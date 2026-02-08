#!/usr/bin/env python
"""
Feasibility harness for ContractIR v0.1 (AST JSON, strict validation, deterministic eval).

This is intentionally a small, concrete bed for iterating:
  - Build excerpt packs (anchor-tagged snippets)
  - Prompt the LLM to emit ContractIR JSON
  - Validate via JSON Schema + light anchor-id checks
  - Optional repair rounds (feed validation errors back)
  - Evaluate one derived function with synthetic scenario inputs
  - Compare to ground truth

Usage (requires a running gateway at http://127.0.0.1:8000):
  python scripts/contract_ir_feasibility_harness.py --source-run-id dan-v2-20260106
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from pipeline.core.config import REQUIRED_MODEL, REQUIRED_REASONING
from pipeline.ir.contract_ir_v0_1 import (
    ContractIREvalError,
    ContractIRValidationError,
    TypedValue,
    evaluate_function,
    validate_contract_ir,
)
from pipeline.evidence.indexing import DEFAULT_GATEWAY_URL, _ensure_gateway_client_sync  # type: ignore


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


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
    """Light provenance gate: all referenced anchors must come from the excerpt pack.

    This is intentionally *not* a semantic citation check. It only enforces that:
      - every anchor id in anchor_ids/source_refs is one of the anchors we provided.
    """

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
                                    message=f"Anchor id {aid!r} is not in the provided excerpt anchors {sorted(allowed)}",
                                    json_path="/" + "/".join(path + [k, str(idx)]),
                                )
                            )
                    else:
                        out.append(
                            ContractIRValidationError(
                                code="anchor_not_in_context",
                                message=f"{k} must be an array of anchor ids",
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
class Case:
    case_id: str
    item_id: str
    snippets_path: Path
    anchor_ids: List[str]
    task: str
    fn_id: str
    fn_args: Dict[str, Any]
    indices: Dict[str, Dict[str, str]]
    expected_rate: str  # decimal-string


def _decimal_str(x: Decimal) -> str:
    # Normalize for comparison; keep full precision but strip trailing zeros.
    s = format(x, "f")
    if "." in s:
        s = s.rstrip("0").rstrip(".") or "0"
    return s


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source-run-id", required=True, help="Run ID containing retrieval_v2 snippet packs.")
    ap.add_argument(
        "--out-run-id",
        default=None,
        help="Output run ID for harness artifacts (default: contract-ir-feasibility-<ts>).",
    )
    ap.add_argument(
        "--prompt",
        default="prompts/contract_ir_v0_1_compile_v1.txt",
        help="Prompt template path.",
    )
    ap.add_argument("--gateway-url", default=DEFAULT_GATEWAY_URL)
    ap.add_argument("--max-repair-rounds", type=int, default=2)
    args = ap.parse_args()

    source_run_dir = Path("runs") / args.source_run_id
    if not source_run_dir.exists():
        raise SystemExit(f"Source run not found: {source_run_dir}")

    out_run_id = args.out_run_id or f"contract-ir-feasibility-{int(time.time())}"
    out_dir = Path("runs") / out_run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    prompt_path = Path(args.prompt)
    template = _read_text(prompt_path)

    # --- Select a few concrete agreements/excerpts from the existing run artifacts ----
    retrieval_dir = source_run_dir / "retrieval_v2"
    cases: List[Case] = [
        Case(
            case_id="A1_adjusted_libor",
            item_id="0000950144-96-000010_7",
            snippets_path=retrieval_dir / "0000950144-96-000010_7_snippets.jsonl",
            anchor_ids=["A0401"],
            task=(
                "Build ContractIR for item_id=0000950144-96-000010_7.\n"
                "Goal: encode the definition for Adjusted London Interbank Offered Rate as a derived function.\n"
                "Requirements:\n"
                "- derived fn_id MUST be AdjustedLondonInterbankOfferedRate\n"
                "- semantic_role MUST be base_rate\n"
                "- args: date (type=date), euro_dollar_reserve_pct (type=decimal)\n"
                "- indices MUST include LondonInterbankOfferedRate (unit=rate)\n"
                "- expr MUST compute: ceil( LIBOR / (1.00 - reserve_pct) ) to the next higher 1/100th of 1%.\n"
                "- IMPORTANT: the 1.00 in (1.00 - reserve_pct) MUST be a decimal literal (type=decimal), not a rate.\n"
                "- Use round_up_to_increment with increment=0.0001.\n"
            ),
            fn_id="AdjustedLondonInterbankOfferedRate",
            fn_args={"date": "1995-09-26", "euro_dollar_reserve_pct": "0.10"},
            indices={"LondonInterbankOfferedRate": {"1995-09-26": "0.0525"}},
            expected_rate="0.0584",
        ),
        Case(
            case_id="A2_abr",
            item_id="0000950134-96-000714_4",
            snippets_path=retrieval_dir / "0000950134-96-000714_4_snippets.jsonl",
            anchor_ids=["A0021"],
            task=(
                "Build ContractIR for item_id=0000950134-96-000714_4.\n"
                "Goal: encode the definition for ABR as a derived function.\n"
                "Requirements:\n"
                "- derived fn_id MUST be ABR\n"
                "- semantic_role MUST be base_rate\n"
                "- args: date (type=date)\n"
                "- indices MUST include PrimeRate (unit=rate) and FederalFundsEffectiveRate (unit=rate)\n"
                "- expr MUST compute: round up to the next 1/16 of 1% of max( PrimeRate, FederalFundsEffectiveRate + 0.5% ).\n"
                "- Use round_up_to_increment with increment=0.000625 and the addend 0.005.\n"
            ),
            fn_id="ABR",
            fn_args={"date": "1996-01-05"},
            indices={
                "PrimeRate": {"1996-01-05": "0.0551"},
                "FederalFundsEffectiveRate": {"1996-01-05": "0.0503"},
            },
            expected_rate="0.055625",
        ),
        Case(
            case_id="B1_applicable_margin_table",
            item_id="0000950134-96-000040_2",
            snippets_path=retrieval_dir / "0000950134-96-000040_2_snippets.jsonl",
            anchor_ids=["A0049"],
            task=(
                "Build ContractIR for item_id=0000950134-96-000040_2.\n"
                "Goal: extract the 'Applicable Margin' table as a table object, and expose a derived function.\n"
                "Requirements:\n"
                "- Create table_id: ApplicableMarginByRating\n"
                "- columns MUST include: rating (string), eurodollar_bps (bps), base_rate_bps (bps)\n"
                "- rows MUST include all table rows; store margins as bps (e.g., 0.375% => 37.5 bps).\n"
                "- derived fn_id MUST be ApplicableMarginEurodollar\n"
                "- semantic_role MUST be margin\n"
                "- args: rating (type=string)\n"
                "- expr MUST compute: bps_to_rate(lookup(\"ApplicableMarginByRating\",\"rating\",rating,\"eurodollar_bps\"))\n"
            ),
            fn_id="ApplicableMarginEurodollar",
            fn_args={"rating": "BBB/Baa2"},
            indices={},
            expected_rate="0.0075",
        ),
        Case(
            case_id="B2_applicable_margin_by_status_code",
            item_id="0001140361-23-000046_9",
            snippets_path=retrieval_dir / "0001140361-23-000046_9_snippets.jsonl",
            anchor_ids=["A0146"],
            task=(
                "Build ContractIR for item_id=0001140361-23-000046_9.\n"
                "Goal: extract the 'Applicable Margin' table row for ABR Loans as a table object, and expose a derived function.\n"
                "Requirements:\n"
                "- Create table_id: ApplicableMarginByStatus\n"
                "- columns MUST include: status_code (string), status_label (string), abr_margin_bps (bps)\n"
                "- Extract ONLY the row whose label is exactly 'ABR Loans and Canadian Prime Rate Loans'.\n"
                "- rows MUST include exactly 6 rows, one per Status column (Level I Status through Level VI Status).\n"
                "- status_label MUST be EXACTLY the column header text (e.g., 'Level V Status').\n"
                "- status_code MUST be the Roman numeral extracted from the header (I, II, III, IV, V, VI).\n"
                "- derived fn_id MUST be ApplicableMarginABR\n"
                "- semantic_role MUST be margin\n"
                "- args: status_code (type=string)\n"
                "- expr MUST compute: bps_to_rate(lookup(\"ApplicableMarginByStatus\",\"status_code\",status_code,\"abr_margin_bps\"))\n"
            ),
            fn_id="ApplicableMarginABR",
            fn_args={"status_code": "V"},
            indices={},
            expected_rate="0.0005",
        ),
    ]

    complete_response_sync = _ensure_gateway_client_sync()
    summary: Dict[str, Any] = {"schema_version": "contract_ir_feasibility_v0", "cases": []}

    for case in cases:
        case_dir = out_dir / case.case_id
        case_dir.mkdir(parents=True, exist_ok=True)

        contexts = build_excerpt_pack(case.snippets_path, case.anchor_ids)
        (case_dir / "contexts.txt").write_text(contexts, encoding="utf-8")

        prompt = _render_prompt(template, task=case.task, contexts=contexts)
        (case_dir / "prompt.txt").write_text(prompt, encoding="utf-8")

        attempt = 0
        repair_rounds_used: Optional[int] = None
        raw: str | None = None
        parsed: Any = None
        validation_errors: List[ContractIRValidationError] = []

        while True:
            attempt += 1
            raw = complete_response_sync(
                model=REQUIRED_MODEL,
                prompt=prompt,
                base_url=args.gateway_url,
                reasoning={"effort": REQUIRED_REASONING},
                temperature=0.0,
                max_output_tokens=None,
                timeout=600.0,
            )
            (case_dir / f"llm_output_attempt_{attempt}.txt").write_text(raw, encoding="utf-8")

            try:
                parsed = json.loads(raw)
            except json.JSONDecodeError as exc:
                validation_errors = [
                    ContractIRValidationError(code="json_parse", message=str(exc), json_path="/")
                ]
            else:
                validation_errors = validate_contract_ir(parsed)
                if not validation_errors:
                    validation_errors = _validate_anchor_ids_in_context(parsed, case.anchor_ids)

            if not validation_errors:
                repair_rounds_used = (attempt - 1)
                break

            if attempt > 1 + max(0, args.max_repair_rounds):
                repair_rounds_used = None
                break

            prompt = _render_repair_prompt(raw_json=raw, errors=validation_errors)

        record: Dict[str, Any] = {
            "case_id": case.case_id,
            "item_id": case.item_id,
            "valid_on_first_try": bool(repair_rounds_used == 0),
            "repair_rounds_used": repair_rounds_used,
            "compile_validate_success": bool(repair_rounds_used is not None),
            "eval_success": False,
            "numeric_correct": False,
            "expected_rate": case.expected_rate,
            "observed_rate": None,
            "failure_type": None,
        }

        if repair_rounds_used is None:
            record["failure_type"] = (
                "json_parse"
                if any(e.code == "json_parse" for e in validation_errors)
                else (validation_errors[0].code if validation_errors else "unknown")
            )
            (case_dir / "validation_errors.json").write_text(
                json.dumps([asdict(e) for e in validation_errors], indent=2),
                encoding="utf-8",
            )
            summary["cases"].append(record)
            continue

        (case_dir / "contract_ir.json").write_text(json.dumps(parsed, indent=2), encoding="utf-8")

        try:
            out: TypedValue = evaluate_function(parsed, fn_id=case.fn_id, args=case.fn_args, indices=case.indices)
            if out.kind != "rate":
                raise TypeError(f"Expected rate output, got {out.kind}")
            observed = _decimal_str(out.value)
            record["observed_rate"] = observed
            record["eval_success"] = True
            record["numeric_correct"] = (observed == case.expected_rate)
        except Exception as exc:
            record["failure_type"] = "eval"
            (case_dir / "eval_error.txt").write_text(str(exc), encoding="utf-8")

        summary["cases"].append(record)

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # Print a tiny human-readable summary for quick iteration.
    ok = sum(1 for c in summary["cases"] if c["numeric_correct"])
    total = len(summary["cases"])
    print(f"[ok] wrote {out_dir}/summary.json")
    print(f"[summary] numeric_correct: {ok}/{total}")
    for c in summary["cases"]:
        status = "OK" if c["numeric_correct"] else "FAIL"
        rr = c["repair_rounds_used"]
        print(f"- {status} {c['case_id']} (repair_rounds={rr}) expected={c['expected_rate']} observed={c['observed_rate']}")


if __name__ == "__main__":
    main()
