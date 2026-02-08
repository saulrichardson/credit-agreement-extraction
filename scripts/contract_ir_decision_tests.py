#!/usr/bin/env python
"""
Exploration harness for two ContractIR design decisions:

1) Status-table key strings:
   - verbatim header keys ("Level V Status") vs normalized keys ("Level V")

2) Semantic disambiguation between "base rate" vs "margin" derived functions:
   - no explicit role
   - role encoded in fn_id naming convention
   - role encoded in a structured field (derived[].semantic_role)

This script runs a small set of experiments against a fixed agreement excerpt pack,
writes all prompts/contexts/LLM outputs to runs/<out_run_id>/, and produces a
summary.json with validation + evaluation results.

Usage:
  python scripts/contract_ir_decision_tests.py --source-run-id dan-v2-20260106
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from pipeline.core.config import REQUIRED_MODEL, REQUIRED_REASONING  # noqa: E402
from pipeline.ir.contract_ir_v0_1 import (  # noqa: E402
    ContractIREvalError,
    ContractIRValidationError,
    evaluate_function,
    validate_contract_ir,
)
from pipeline.evidence.indexing import DEFAULT_GATEWAY_URL, _ensure_gateway_client_sync  # type: ignore  # noqa: E402


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _iter_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def build_excerpt_pack(snippets_path: Path, anchor_ids: List[str]) -> str:
    by_anchor: Dict[str, Dict[str, Any]] = {}
    for rec in _iter_jsonl(snippets_path):
        aid = rec.get("anchor_id")
        if isinstance(aid, str) and aid not in by_anchor:
            by_anchor[aid] = rec

    blocks: List[str] = []
    for aid in anchor_ids:
        rec = by_anchor.get(aid)
        if not rec:
            blocks.append(f"[[{aid}]]\n<MISSING SNIPPET>")
            continue
        snippet = str(rec.get("snippet") or "").rstrip()
        blocks.append(f"[[{aid}]]\n{snippet}")
    return "\n\n".join(blocks).strip() + "\n"


def _render_prompt(template: str, *, task: str, contexts: str) -> str:
    out = template
    out = out.replace("{{TASK}}", task)
    out = out.replace("{{CONTEXTS}}", contexts)
    return out


def _render_repair_prompt(*, raw_json: str, errors: List[ContractIRValidationError]) -> str:
    err_list = [asdict(e) for e in errors]
    return (
        "Your previous response did not validate against the ContractIR v0.1 schema.\n"
        "Return corrected JSON ONLY. Do not include commentary or code fences.\n\n"
        "VALIDATION_ERRORS_JSON:\n"
        f"{json.dumps(err_list, indent=2)}\n\n"
        "PREVIOUS_JSON:\n"
        f"{raw_json}\n"
    )


def _validate_anchor_ids_in_context(doc: Any, allowed_anchor_ids: List[str]) -> List[ContractIRValidationError]:
    allowed = set(allowed_anchor_ids)
    out: List[ContractIRValidationError] = []

    def _walk(obj: Any, path: List[str]) -> None:
        if isinstance(obj, dict):
            for k, v in obj.items():
                if k in ("source_refs", "anchor_ids"):
                    if isinstance(v, list):
                        for idx, aid in enumerate(v):
                            if isinstance(aid, str) and aid in allowed:
                                continue
                            out.append(
                                ContractIRValidationError(
                                    code="anchor_not_in_context",
                                    message=f"Anchor id {aid!r} is not in excerpt anchors {sorted(allowed)}",
                                    json_path="/" + "/".join(path + [k, str(idx)]),
                                )
                            )
                    else:
                        out.append(
                            ContractIRValidationError(
                                code="anchor_not_in_context",
                                message=f"{k} must be an array",
                                json_path="/" + "/".join(path + [k]),
                            )
                        )
                else:
                    _walk(v, path + [k])
        elif isinstance(obj, list):
            for i, v in enumerate(obj):
                _walk(v, path + [str(i)])

    _walk(doc, [])
    return out


@dataclass(frozen=True)
class EvalSpec:
    fn_id: str
    args: Mapping[str, Any]
    indices: Mapping[str, Mapping[str, str]]
    expected_rate: str


@dataclass(frozen=True)
class Experiment:
    exp_id: str
    item_id: str
    anchor_ids: List[str]
    task: str
    evals: List[EvalSpec]
    notes: Optional[str] = None


def _extract_string_cells(contract_ir: Mapping[str, Any], table_id: str, column_name: str) -> List[str]:
    out: List[str] = []
    for t in contract_ir.get("tables", []) or []:
        if not isinstance(t, dict) or t.get("table_id") != table_id:
            continue
        for row in t.get("rows", []) or []:
            if not isinstance(row, dict):
                continue
            cells = row.get("cells") or {}
            if not isinstance(cells, dict):
                continue
            node = cells.get(column_name)
            if not isinstance(node, dict):
                continue
            lit = node.get("lit")
            if isinstance(lit, dict) and lit.get("type") == "string" and isinstance(lit.get("value"), str):
                out.append(lit["value"])
    return out


def _extract_semantic_roles(contract_ir: Mapping[str, Any]) -> Dict[str, Any]:
    roles: Dict[str, Any] = {}
    for fn in contract_ir.get("derived", []) or []:
        if not isinstance(fn, dict):
            continue
        fn_id = fn.get("fn_id")
        if not isinstance(fn_id, str):
            continue
        roles[fn_id] = fn.get("semantic_role")
    return roles


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-run-id", default="dan-v2-20260106")
    parser.add_argument("--out-run-id", default=None)
    parser.add_argument("--gateway-url", default=DEFAULT_GATEWAY_URL)
    parser.add_argument("--prompt", default="prompts/contract_ir_v0_1_compile_v1.txt")
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--timeout-seconds", type=float, default=240.0)
    args = parser.parse_args()

    out_run_id = args.out_run_id or f"contractir-decision-tests-{time.strftime('%Y%m%d-%H%M%S')}"
    out_dir = PROJECT_ROOT / "runs" / out_run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    prompt_path = PROJECT_ROOT / args.prompt
    template = _read_text(prompt_path)

    # Agreements under test.
    modern_item_id = "0001140361-23-000046_9"
    modern_abr_anchor = "A0118"  # ABR definition
    modern_margin_anchor = "A0146"  # Applicable Margin table with Level I..VI Status headers

    legacy_item_id = "0000950134-96-000714_4"
    legacy_abr_anchor = "A0021"
    legacy_margin_anchor = "A0026"

    indices_for_modern_abr = {
        "PrimeRate": {"2023-01-03": "0.0500"},
        "NYFRBRate": {"2023-01-03": "0.0520"},
        "AdjustedTermSOFRRate1M": {"2023-01-03": "0.0490"},
    }

    experiments: List[Experiment] = [
        Experiment(
            exp_id="E1_status_keys_verbatim_header",
            item_id=modern_item_id,
            anchor_ids=[modern_margin_anchor],
            task=(
                f"Build ContractIR for item_id={modern_item_id}.\n"
                "Goal: extract the 'Applicable Margin' table for ABR Loans as a table object, and expose a derived function.\n"
                "Requirements:\n"
                "- Create table_id: ApplicableMarginByStatus\n"
                "- columns MUST include: status (string), abr_margin_bps (bps)\n"
                "- Extract ONLY the row whose label is exactly 'ABR Loans and Canadian Prime Rate Loans'.\n"
                "- derived fn_id MUST be ApplicableMarginABR\n"
                "- semantic_role MUST be margin\n"
                "- IMPORTANT: the status string MUST be EXACTLY the column header text from the table (e.g., 'Level V Status').\n"
                "- args: status (type=string)\n"
                "- expr MUST compute: bps_to_rate(lookup(\"ApplicableMarginByStatus\",\"status\",status,\"abr_margin_bps\"))\n"
            ),
            evals=[
                EvalSpec(
                    fn_id="ApplicableMarginABR",
                    args={"status": "Level V Status"},
                    indices={},
                    expected_rate="0.0005",
                )
            ],
        ),
        Experiment(
            exp_id="E2_status_keys_normalized",
            item_id=modern_item_id,
            anchor_ids=[modern_margin_anchor],
            task=(
                f"Build ContractIR for item_id={modern_item_id}.\n"
                "Goal: extract the 'Applicable Margin' table for ABR Loans as a table object, and expose a derived function.\n"
                "Requirements:\n"
                "- Create table_id: ApplicableMarginByStatus\n"
                "- columns MUST include: status (string), abr_margin_bps (bps)\n"
                "- Extract ONLY the row whose label is exactly 'ABR Loans and Canadian Prime Rate Loans'.\n"
                "- derived fn_id MUST be ApplicableMarginABR\n"
                "- semantic_role MUST be margin\n"
                "- IMPORTANT: normalize status keys by removing the trailing word 'Status' from the header.\n"
                "  Example: 'Level V Status' becomes 'Level V'. Use exactly 'Level I'..'Level VI'.\n"
                "- args: status (type=string)\n"
                "- expr MUST compute: bps_to_rate(lookup(\"ApplicableMarginByStatus\",\"status\",status,\"abr_margin_bps\"))\n"
            ),
            evals=[
                EvalSpec(
                    fn_id="ApplicableMarginABR",
                    args={"status": "Level V"},
                    indices={},
                    expected_rate="0.0005",
                )
            ],
        ),
        Experiment(
            exp_id="E7_status_code_plus_label",
            item_id=modern_item_id,
            anchor_ids=[modern_margin_anchor],
            task=(
                f"Build ContractIR for item_id={modern_item_id}.\n"
                "Goal: extract the 'Applicable Margin' table row for ABR Loans as a table object, and expose a derived function.\n"
                "Requirements:\n"
                "- Create table_id: ApplicableMarginByStatus\n"
                "- columns MUST include: status_code (string), status_label (string), abr_margin_bps (bps)\n"
                "- Extract ONLY the row whose label is exactly 'ABR Loans and Canadian Prime Rate Loans'.\n"
                "- status_label MUST be EXACTLY the column header text (e.g., 'Level V Status').\n"
                "- status_code MUST be the Roman numeral extracted from the header (I, II, III, IV, V, VI).\n"
                "- derived fn_id MUST be ApplicableMarginABR\n"
                "- semantic_role MUST be margin\n"
                "- args: status_code (type=string)\n"
                "- expr MUST compute: bps_to_rate(lookup(\"ApplicableMarginByStatus\",\"status_code\",status_code,\"abr_margin_bps\"))\n"
            ),
            evals=[
                EvalSpec(
                    fn_id="ApplicableMarginABR",
                    args={"status_code": "V"},
                    indices={},
                    expected_rate="0.0005",
                )
            ],
        ),
        Experiment(
            exp_id="E3_combined_semantic_role_normalized_keys",
            item_id=modern_item_id,
            anchor_ids=[modern_abr_anchor, modern_margin_anchor],
            task=(
                f"Build ContractIR for item_id={modern_item_id}.\n"
                "Goal: encode the ABR definition AND the Applicable Margin table (ABR row) in ONE ContractIR JSON.\n"
                "Requirements:\n"
                "- IMPORTANT: set derived.semantic_role for EVERY derived function:\n"
                "  - ABR MUST have semantic_role='base_rate'\n"
                "  - ApplicableMarginABR MUST have semantic_role='margin'\n"
                "\n"
                "A) ABR (base rate definition)\n"
                "- derived fn_id MUST be ABR\n"
                "- args: date (type=date)\n"
                "- indices MUST include PrimeRate (unit=rate), NYFRBRate (unit=rate), AdjustedTermSOFRRate1M (unit=rate)\n"
                "- expr MUST compute: max(PrimeRate, NYFRBRate + 0.5%, AdjustedTermSOFRRate1M + 1.0%)\n"
                "- If the excerpt includes conditional logic excluding clause (c), record that as an open_item.\n"
                "\n"
                "B) Applicable Margin table (ABR row)\n"
                "- Create table_id: ApplicableMarginByStatus\n"
                "- columns MUST include: status (string), abr_margin_bps (bps)\n"
                "- IMPORTANT: normalize status keys by removing the trailing word 'Status' (use 'Level I'..'Level VI').\n"
                "- derived fn_id MUST be ApplicableMarginABR\n"
                "- args: status (type=string)\n"
                "- expr MUST compute: bps_to_rate(lookup(\"ApplicableMarginByStatus\",\"status\",status,\"abr_margin_bps\"))\n"
            ),
            evals=[
                EvalSpec(
                    fn_id="ABR",
                    args={"date": "2023-01-03"},
                    indices=indices_for_modern_abr,
                    expected_rate="0.0590",
                ),
                EvalSpec(
                    fn_id="ApplicableMarginABR",
                    args={"status": "Level V"},
                    indices={},
                    expected_rate="0.0005",
                ),
            ],
        ),
        Experiment(
            exp_id="E9_combined_semantic_role_plus_status_code",
            item_id=modern_item_id,
            anchor_ids=[modern_abr_anchor, modern_margin_anchor],
            task=(
                f"Build ContractIR for item_id={modern_item_id}.\n"
                "Goal: encode the ABR definition AND the Applicable Margin table (ABR row) in ONE ContractIR JSON.\n"
                "Requirements:\n"
                "- IMPORTANT: set derived.semantic_role for EVERY derived function:\n"
                "  - ABR MUST have semantic_role='base_rate'\n"
                "  - ApplicableMarginABR MUST have semantic_role='margin'\n"
                "\n"
                "A) ABR (base rate definition)\n"
                "- derived fn_id MUST be ABR\n"
                "- args: date (type=date)\n"
                "- indices MUST include PrimeRate (unit=rate), NYFRBRate (unit=rate), AdjustedTermSOFRRate1M (unit=rate)\n"
                "- expr MUST compute: max(PrimeRate, NYFRBRate + 0.5%, AdjustedTermSOFRRate1M + 1.0%)\n"
                "- If the excerpt includes conditional logic excluding clause (c), record that as an open_item.\n"
                "\n"
                "B) Applicable Margin table (ABR row)\n"
                "- Create table_id: ApplicableMarginByStatus\n"
                "- columns MUST include: status_code (string), status_label (string), abr_margin_bps (bps)\n"
                "- Extract ONLY the row whose label is exactly 'ABR Loans and Canadian Prime Rate Loans'.\n"
                "- rows MUST include exactly 6 rows, one per Status column (Level I Status through Level VI Status).\n"
                "- status_label MUST be EXACTLY the column header text (e.g., 'Level V Status').\n"
                "- status_code MUST be the Roman numeral extracted from the header (I, II, III, IV, V, VI).\n"
                "- derived fn_id MUST be ApplicableMarginABR\n"
                "- args: status_code (type=string)\n"
                "- expr MUST compute: bps_to_rate(lookup(\"ApplicableMarginByStatus\",\"status_code\",status_code,\"abr_margin_bps\"))\n"
            ),
            evals=[
                EvalSpec(
                    fn_id="ABR",
                    args={"date": "2023-01-03"},
                    indices=indices_for_modern_abr,
                    expected_rate="0.0590",
                ),
                EvalSpec(
                    fn_id="ApplicableMarginABR",
                    args={"status_code": "V"},
                    indices={},
                    expected_rate="0.0005",
                ),
            ],
        ),
        Experiment(
            exp_id="E10_combined_explicit_row_label",
            item_id=modern_item_id,
            anchor_ids=[modern_abr_anchor, modern_margin_anchor],
            task=(
                f"Build ContractIR for item_id={modern_item_id}.\n"
                "Goal: encode the ABR definition AND the Applicable Margin table (ABR row) in ONE ContractIR JSON.\n"
                "Requirements:\n"
                "- IMPORTANT: set derived.semantic_role for EVERY derived function:\n"
                "  - ABR MUST have semantic_role='base_rate'\n"
                "  - ApplicableMarginABR MUST have semantic_role='margin'\n"
                "\n"
                "A) ABR (base rate definition)\n"
                "- derived fn_id MUST be ABR\n"
                "- args: date (type=date)\n"
                "- indices MUST include PrimeRate (unit=rate), NYFRBRate (unit=rate), AdjustedTermSOFRRate1M (unit=rate)\n"
                "- expr MUST compute: max(PrimeRate, NYFRBRate + 0.5%, AdjustedTermSOFRRate1M + 1.0%)\n"
                "- If the excerpt includes conditional logic excluding clause (c), record that as an open_item.\n"
                "\n"
                "B) Applicable Margin table (ABR row ONLY)\n"
                "- Create table_id: ApplicableMarginByStatus\n"
                "- columns MUST include: status (string), abr_margin_bps (bps)\n"
                "- IMPORTANT: the ABR row is the row whose left-most label text is EXACTLY: 'ABR Loans and Canadian Prime Rate Loans'.\n"
                "- Extract ONLY that row into your table (do not use the Term Benchmark Loans row).\n"
                "- derived fn_id MUST be ApplicableMarginABR\n"
                "- semantic_role MUST be margin\n"
                "- args: status (type=string)\n"
                "- IMPORTANT: normalize status keys by removing the trailing word 'Status'. Use exactly: 'Level I'..'Level VI'.\n"
                "- expr MUST compute: bps_to_rate(lookup(\"ApplicableMarginByStatus\",\"status\",status,\"abr_margin_bps\"))\n"
            ),
            evals=[
                EvalSpec(
                    fn_id="ABR",
                    args={"date": "2023-01-03"},
                    indices=indices_for_modern_abr,
                    expected_rate="0.0590",
                ),
                EvalSpec(
                    fn_id="ApplicableMarginABR",
                    args={"status": "Level V"},
                    indices={},
                    expected_rate="0.0005",
                ),
            ],
        ),
        Experiment(
            exp_id="E8_abr_plus_margin_semantic_role_field_leverage_ratio",
            item_id=legacy_item_id,
            anchor_ids=[legacy_abr_anchor, legacy_margin_anchor],
            task=(
                f"Build ContractIR for item_id={legacy_item_id}.\n"
                "Goal: encode the ABR definition AND the Applicable Margin table (by Leverage Ratio) in ONE ContractIR JSON.\n"
                "Requirements:\n"
                "- IMPORTANT: set derived.semantic_role for EVERY derived function:\n"
                "  - ABR MUST have semantic_role='base_rate'\n"
                "  - ApplicableMarginABR MUST have semantic_role='margin'\n"
                "\n"
                "A) ABR (base rate definition)\n"
                "- derived fn_id MUST be ABR\n"
                "- args: date (type=date)\n"
                "- indices MUST include PrimeRate (unit=rate) and FederalFundsEffectiveRate (unit=rate)\n"
                "- expr MUST compute: round up to the next 1/16 of 1% of max( PrimeRate, FederalFundsEffectiveRate + 0.5% ).\n"
                "- Use round_up_to_increment with increment=0.000625 and the addend 0.005.\n"
                "\n"
                "B) Applicable Margin table (by Leverage Ratio)\n"
                "- Create table_id: ApplicableMarginByLeverageRatio\n"
                "- columns MUST include: leverage_ratio_bucket (string), abr_margin_bps (bps), eurodollar_margin_bps (bps)\n"
                "- rows MUST include all table rows; store margins as bps (e.g., 0.125% => 12.5 bps).\n"
                "- derived fn_id MUST be ApplicableMarginABR\n"
                "- semantic_role MUST be margin\n"
                "- args: leverage_ratio_bucket (type=string)\n"
                "- expr MUST compute: bps_to_rate(lookup(\"ApplicableMarginByLeverageRatio\",\"leverage_ratio_bucket\",leverage_ratio_bucket,\"abr_margin_bps\"))\n"
            ),
            evals=[
                EvalSpec(
                    fn_id="ABR",
                    args={"date": "1996-01-05"},
                    indices={
                        "PrimeRate": {"1996-01-05": "0.0551"},
                        "FederalFundsEffectiveRate": {"1996-01-05": "0.0503"},
                    },
                    expected_rate="0.055625",
                ),
                EvalSpec(
                    fn_id="ApplicableMarginABR",
                    args={"leverage_ratio_bucket": "Greater than 5.00 to 1"},
                    indices={},
                    expected_rate="0.00125",
                ),
            ],
        ),
    ]

    complete_response_sync = _ensure_gateway_client_sync()

    run_summary: Dict[str, Any] = {
        "schema_version": "contractir_decision_tests_v0",
        "source_run_id": args.source_run_id,
        "out_run_id": out_run_id,
        "model": REQUIRED_MODEL,
        "reasoning_effort": REQUIRED_REASONING,
        "gateway_url": args.gateway_url,
        "prompt_path": str(prompt_path),
        "prompt_sha": None,
        "experiments": [],
    }

    for exp in experiments:
        exp_dir = out_dir / exp.exp_id
        exp_dir.mkdir(parents=True, exist_ok=True)

        snippets_path = PROJECT_ROOT / "runs" / args.source_run_id / "retrieval_v2" / f"{exp.item_id}_snippets.jsonl"
        if not snippets_path.exists():
            raise FileNotFoundError(f"Missing retrieval snippets: {snippets_path}")

        contexts = build_excerpt_pack(snippets_path, exp.anchor_ids)
        (exp_dir / "contexts.txt").write_text(contexts, encoding="utf-8")

        prompt = _render_prompt(template, task=exp.task, contexts=contexts)
        (exp_dir / "prompt.txt").write_text(prompt, encoding="utf-8")

        attempt = 0
        repair_rounds_used: Optional[int] = None
        raw: Optional[str] = None
        parsed: Any = None
        validation_errors: List[ContractIRValidationError] = []
        anchor_errors: List[ContractIRValidationError] = []

        while True:
            attempt += 1
            raw = complete_response_sync(
                model=REQUIRED_MODEL,
                prompt=prompt,
                base_url=args.gateway_url,
                reasoning={"effort": REQUIRED_REASONING},
                temperature=0,
                timeout=args.timeout_seconds,
            )
            (exp_dir / f"raw_attempt_{attempt}.txt").write_text(raw, encoding="utf-8")

            try:
                parsed = json.loads(raw)
                _write_json(exp_dir / f"parsed_attempt_{attempt}.json", parsed)
            except Exception as e:
                parsed = None
                validation_errors = [ContractIRValidationError(code="json_parse", message=str(e), json_path="/")]

            if parsed is not None:
                validation_errors = validate_contract_ir(parsed)
                if not validation_errors:
                    anchor_errors = _validate_anchor_ids_in_context(parsed, exp.anchor_ids)

            if parsed is not None and not validation_errors and not anchor_errors:
                break

            if attempt >= args.max_attempts:
                break

            if parsed is None:
                continue

            repair_rounds_used = (repair_rounds_used or 0) + 1
            prompt = _render_repair_prompt(raw_json=raw, errors=(validation_errors or anchor_errors))
            (exp_dir / f"repair_prompt_round_{repair_rounds_used}.txt").write_text(prompt, encoding="utf-8")

        exp_result: Dict[str, Any] = {
            "exp_id": exp.exp_id,
            "item_id": exp.item_id,
            "anchor_ids": exp.anchor_ids,
            "attempts_used": attempt,
            "repair_rounds_used": repair_rounds_used,
            "parse_success": parsed is not None,
            "schema_valid": bool(parsed is not None and not validation_errors),
            "anchor_context_valid": bool(parsed is not None and not validation_errors and not anchor_errors),
            "evals": [],
            "status_values": None,
            "semantic_roles": None,
        }

        if validation_errors:
            _write_json(exp_dir / "validation_errors.json", [asdict(e) for e in validation_errors])
        if anchor_errors:
            _write_json(exp_dir / "anchor_context_errors.json", [asdict(e) for e in anchor_errors])

        if parsed is not None and not validation_errors and not anchor_errors:
            _write_json(exp_dir / "contractir_validated.json", parsed)

            # Optional introspection fields used for decision-making.
            status_values: Dict[str, Any] = {}
            status_values["status"] = _extract_string_cells(parsed, table_id="ApplicableMarginByStatus", column_name="status")
            status_values["status_code"] = _extract_string_cells(
                parsed, table_id="ApplicableMarginByStatus", column_name="status_code"
            )
            status_values["status_label"] = _extract_string_cells(
                parsed, table_id="ApplicableMarginByStatus", column_name="status_label"
            )
            status_values["leverage_ratio_bucket"] = _extract_string_cells(
                parsed, table_id="ApplicableMarginByLeverageRatio", column_name="leverage_ratio_bucket"
            )
            exp_result["status_values"] = {k: v for k, v in status_values.items() if v}
            exp_result["semantic_roles"] = _extract_semantic_roles(parsed)

            for spec in exp.evals:
                ev = {"fn_id": spec.fn_id, "expected_rate": spec.expected_rate, "args": dict(spec.args)}
                try:
                    out = evaluate_function(parsed, fn_id=spec.fn_id, args=spec.args, indices=spec.indices)
                    got = str(out.value)
                    ev["got_rate"] = got
                    ev["numeric_correct"] = got == spec.expected_rate
                    ev["eval_success"] = True
                except ContractIREvalError as e:
                    ev["eval_success"] = False
                    ev["numeric_correct"] = False
                    ev["error"] = str(e)
                exp_result["evals"].append(ev)

            _write_json(exp_dir / "result.json", exp_result)
        else:
            _write_json(exp_dir / "result.json", exp_result)

        run_summary["experiments"].append(exp_result)

    _write_json(out_dir / "summary.json", run_summary)
    print(f"Wrote decision test artifacts to {out_dir}")
    print(f"Summary: {out_dir / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
