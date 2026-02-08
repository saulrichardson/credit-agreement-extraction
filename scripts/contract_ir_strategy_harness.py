#!/usr/bin/env python
"""
Strategy harness for ContractIR v0.1 prompts and edge cases.

Goal: Compare prompt/structure strategies for:
  - base rate extraction (including one-off conditions)
  - margin pricing extraction (including 2D grids)

This script is intentionally artifact-first:
  - writes prompt/context/raw LLM outputs + validated JSON into runs/<out_run_id>/
  - validates strict ContractIR schema + anchor-in-context gates
  - runs deterministic evaluation for selected functions with synthetic inputs

Usage:
  python scripts/contract_ir_strategy_harness.py --source-run-id dan-v2-20260106
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

from pipeline.core.config import REQUIRED_MODEL, REQUIRED_REASONING  # noqa: E402
from pipeline.ir.contract_ir_v0_1 import (  # noqa: E402
    ContractIREvalError,
    ContractIRValidationError,
    evaluate_function,
    validate_contract_ir,
)
from pipeline.evidence.indexing import DEFAULT_GATEWAY_URL, _ensure_gateway_client_sync  # type: ignore  # noqa: E402


def _decimal_str(x: Decimal) -> str:
    s = format(x, "f")
    if "." in s:
        s = s.rstrip("0").rstrip(".") or "0"
    return s


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


def _validate_label_cells_in_context(doc: Any, *, contexts: str) -> List[ContractIRValidationError]:
    """Optional grounding gate: if a table includes *_label columns, ensure those exact strings appear in CONTEXT.

    This is intentionally narrow and non-magical:
      - It only applies when the model outputs explicit label cells (column names ending with '_label')
      - It checks exact substring presence in the excerpt pack text
    """

    out: List[ContractIRValidationError] = []
    ctx_text = contexts or ""

    tables = doc.get("tables") if isinstance(doc, dict) else None
    if not isinstance(tables, list):
        return out

    for ti, table in enumerate(tables):
        if not isinstance(table, dict):
            continue
        rows = table.get("rows")
        if not isinstance(rows, list):
            continue
        for ri, row in enumerate(rows):
            if not isinstance(row, dict):
                continue
            cells = row.get("cells")
            if not isinstance(cells, dict):
                continue
            for col_name, cell in cells.items():
                if not isinstance(col_name, str) or not col_name.endswith("_label"):
                    continue
                if not isinstance(cell, dict):
                    continue
                lit = cell.get("lit")
                if not isinstance(lit, dict):
                    continue
                if lit.get("type") != "string":
                    continue
                val = lit.get("value")
                if not isinstance(val, str) or not val:
                    continue
                if val not in ctx_text:
                    out.append(
                        ContractIRValidationError(
                            code="label_not_in_context",
                            message=f"Label {val!r} for column {col_name!r} not found in CONTEXT text",
                            json_path=f"/tables/{ti}/rows/{ri}/cells/{col_name}/lit/value",
                        )
                    )

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
    prompt_path: str
    item_id: str
    anchor_ids: List[str]
    task: str
    evals: List[EvalSpec]
    context_clip_chars: Optional[int] = None
    context_stop_before: Optional[str] = None
    expect_open_items_min: Optional[int] = None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-run-id", default="dan-v2-20260106")
    parser.add_argument("--out-run-id", default=None)
    parser.add_argument("--gateway-url", default=DEFAULT_GATEWAY_URL)
    parser.add_argument("--max-repair-rounds", type=int, default=2)
    parser.add_argument("--timeout-seconds", type=float, default=600.0)
    parser.add_argument(
        "--exp-filter",
        default=None,
        help="If provided, only run experiments whose exp_id contains this substring.",
    )
    args = parser.parse_args()

    out_run_id = args.out_run_id or f"contractir-strategy-tests-{time.strftime('%Y%m%d-%H%M%S')}"
    out_dir = PROJECT_ROOT / "runs" / out_run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Agreements / anchors used as ground for edge cases ----------------------------------------
    # Status grid (2D table: loan_type row x status columns).
    status_grid_item_a = "0001140361-23-000046_9"
    status_grid_anchor_a = "A0146"

    status_grid_item_b = "0001140361-23-000246_8"
    status_grid_anchor_b = "A0146"

    # Leverage ratio table (rows = leverage buckets; columns = ABR vs Eurodollar).
    leverage_item = "0000950134-96-000714_4"
    leverage_anchor = "A0026"

    # Rating table (rows = rating bucket; columns = Eurodollar vs Base Rate).
    rating_item = "0000950134-96-000040_2"
    rating_anchor = "A0049"

    # ABR with a "clause (c) exclusion" style conditional inside the definition.
    abr_conditional_item = "0001140361-23-000046_9"
    abr_conditional_anchor = "A0118"

    # Term SOFR definition with an explicit 2.50% floor.
    term_sofr_item = "0001819974-23-000003_2"
    term_sofr_anchor = "A0446"

    # 1996-style agreement with "Alternate Base Rate" and Pricing Level grids.
    abr_pricing_levels_item = "0000912057-96-000124_5"
    abr_pricing_levels_base_rate_anchor = "A0070"
    abr_pricing_levels_margins_anchor = "A0076"  # ABR margin + commitment fee
    abr_pricing_levels_euro_margin_anchor = "A0080"  # includes Eurodollar margin table

    # Amendment with "Applicable Margin" table by S&P rating across loan types (ABR/LIBOR/CD).
    sp_rating_grid_item = "0000041850-96-000001_5"
    sp_rating_grid_anchor = "A0015"
    sp_rating_grid_with_conditional_anchor = "A0018"  # includes the "+0.15% if Advances > $10,000,000" clause

    # Note with a net-income-driven LIBOR spread schedule (range buckets).
    net_income_grid_item = "0000355948-96-000003_2"
    net_income_grid_anchor = "A0069"

    # Agreement with 2D conditional margin schedule by two ratios (range + conjunctions).
    ratios_margin_item = "0000720032-96-000002_2"
    ratios_margin_anchor = "A0125"
    index_rate_item = "0000720032-96-000002_2"
    index_rate_anchor = "A0949"

    # AT&T Capital agreement: Base Rate definition + 2D status-grid margin table (I-IV) + facility fee schedule.
    att_item = "0000950117-96-000184_5"
    att_base_rate_anchor = "A0046"
    att_status_margin_anchor = "A0295"
    att_facility_fee_anchor = "A0300"

    # Reserve adjustment clause (Adjusted LIBO via Euro Reserve Percentage).
    reserve_adjust_item = "0000950152-96-000050_2"
    reserve_adjust_anchor = "A0139"

    # Leverage-ratio buckets margin schedule (ranges).
    leverage_bucket_item = "0000906318-96-000011_6"
    leverage_bucket_anchor = "A0019"

    # Modern agreement with small leverage threshold tables (>= / <) for applicable margins + commitment fee.
    modern_small_tables_item = "0000950170-23-000122_2"
    modern_applicable_rate_anchor = "A0575"
    modern_commitment_fee_anchor = "A0593"

    indices_for_modern_abr = {
        "PrimeRate": {"2023-01-03": "0.0500"},
        "NYFRBRate": {"2023-01-03": "0.0520"},
        "AdjustedTermSOFRRate1M": {"2023-01-03": "0.0490"},
    }

    indices_for_alt_base_rate_rounding = {
        "PrimeRate": {"1997-03-31": "0.05491"},
        "FederalFundsRate": {"1997-03-31": "0.04982"},
    }

    indices_for_simple_base_rate = {
        "PrimeRate": {"1996-12-31": "0.0525"},
        "FederalFundsRate": {"1996-12-31": "0.0475"},
    }

    indices_for_index_rate = {
        "BankersTrustPrimeRate": {"1996-06-28": "0.0550"},
        "ChemicalBankPrimeRate": {"1996-06-28": "0.0540"},
        "CitibankPrimeRate": {"1996-06-28": "0.0565"},
        "MorganGuarantyPrimeRate": {"1996-06-28": "0.0535"},
        "ChaseManhattanPrimeRate": {"1996-06-28": "0.0555"},
        "CPRate": {"1996-06-28": "0.0520"},
        "FederalFundsRate": {"1996-06-28": "0.0500"},
    }

    experiments: List[Experiment] = [
        # --- Margin: specialized (status grid) -----------------------------------------------------
        Experiment(
            exp_id="M1A_status_grid_specialized_ABR_row_itemA",
            prompt_path="prompts/contract_ir_margin_v1_specialized.txt",
            item_id=status_grid_item_a,
            anchor_ids=[status_grid_anchor_a],
            task=(
                f"Build ContractIR for item_id={status_grid_item_a}.\n"
                "Goal: extract ONLY the 'ABR Loans and Canadian Prime Rate Loans' row from the Applicable Margin status grid.\n"
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
            exp_id="M1B_status_grid_specialized_TermBenchmark_row_itemA",
            prompt_path="prompts/contract_ir_margin_v1_specialized.txt",
            item_id=status_grid_item_a,
            anchor_ids=[status_grid_anchor_a],
            task=(
                f"Build ContractIR for item_id={status_grid_item_a}.\n"
                "Goal: extract ONLY the 'Term Benchmark Loans, RFR Loans, and Money Market Loans' row from the Applicable Margin status grid.\n"
                "Requirements:\n"
                "- Create table_id: ApplicableMarginTermBenchmarkByStatus\n"
                "- columns MUST include: status_code (string), status_label (string), term_benchmark_margin_bps (bps)\n"
                "- Extract ONLY the row whose label is exactly 'Term Benchmark Loans, RFR Loans, and Money Market Loans'.\n"
                "- rows MUST include exactly 6 rows, one per Status column (Level I Status through Level VI Status).\n"
                "- status_label MUST be EXACTLY the column header text (e.g., 'Level V Status').\n"
                "- status_code MUST be the Roman numeral extracted from the header (I, II, III, IV, V, VI).\n"
                "- derived fn_id MUST be ApplicableMarginTermBenchmark\n"
                "- semantic_role MUST be margin\n"
                "- args: status_code (type=string)\n"
                "- expr MUST compute: bps_to_rate(lookup(\"ApplicableMarginTermBenchmarkByStatus\",\"status_code\",status_code,\"term_benchmark_margin_bps\"))\n"
            ),
            evals=[
                EvalSpec(
                    fn_id="ApplicableMarginTermBenchmark",
                    args={"status_code": "V"},
                    indices={},
                    expected_rate="0.0105",
                )
            ],
        ),
        # --- Margin: lookup2 (status grid normalized into long table) ------------------------------
        Experiment(
            exp_id="M2_status_grid_lookup2_longtable_itemA",
            prompt_path="prompts/contract_ir_margin_v1_lookup2.txt",
            item_id=status_grid_item_a,
            anchor_ids=[status_grid_anchor_a],
            task=(
                f"Build ContractIR for item_id={status_grid_item_a}.\n"
                "Goal: normalize the Applicable Margin status grid into a long table and expose ONE generic lookup function.\n"
                "Requirements:\n"
                "- Create table_id: ApplicableMarginGrid\n"
                "- columns MUST include: loan_type_code (string), status_code (string), margin_bps (bps)\n"
                "- loan_type_code MUST be exactly one of: 'ABR', 'TERM_BENCHMARK'\n"
                "- status_code MUST be the Roman numeral extracted from the Status headers (I, II, III, IV, V, VI).\n"
                "- rows MUST include exactly 12 rows:\n"
                "  - 6 for loan_type_code='ABR' using the 'ABR Loans and Canadian Prime Rate Loans' row\n"
                "  - 6 for loan_type_code='TERM_BENCHMARK' using the 'Term Benchmark Loans, RFR Loans, and Money Market Loans' row\n"
                "- derived fn_id MUST be ApplicableMargin\n"
                "- semantic_role MUST be margin\n"
                "- args: loan_type_code (type=string), status_code (type=string)\n"
                "- expr MUST compute: bps_to_rate(lookup2(\"ApplicableMarginGrid\",\"loan_type_code\",loan_type_code,\"status_code\",status_code,\"margin_bps\"))\n"
            ),
            evals=[
                EvalSpec(
                    fn_id="ApplicableMargin",
                    args={"loan_type_code": "ABR", "status_code": "V"},
                    indices={},
                    expected_rate="0.0005",
                ),
                EvalSpec(
                    fn_id="ApplicableMargin",
                    args={"loan_type_code": "TERM_BENCHMARK", "status_code": "V"},
                    indices={},
                    expected_rate="0.0105",
                ),
            ],
        ),
        # Repeat status grid on another agreement to increase sample size.
        Experiment(
            exp_id="M3A_status_grid_specialized_ABR_row_itemB",
            prompt_path="prompts/contract_ir_margin_v1_specialized.txt",
            item_id=status_grid_item_b,
            anchor_ids=[status_grid_anchor_b],
            task=(
                f"Build ContractIR for item_id={status_grid_item_b}.\n"
                "Goal: extract ONLY the 'ABR Loans and Canadian Prime Rate Loans' row from the Applicable Margin status grid.\n"
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
            exp_id="M3B_status_grid_lookup2_longtable_itemB",
            prompt_path="prompts/contract_ir_margin_v1_lookup2.txt",
            item_id=status_grid_item_b,
            anchor_ids=[status_grid_anchor_b],
            task=(
                f"Build ContractIR for item_id={status_grid_item_b}.\n"
                "Goal: normalize the Applicable Margin status grid into a long table and expose ONE generic lookup function.\n"
                "Requirements:\n"
                "- Create table_id: ApplicableMarginGrid\n"
                "- columns MUST include: loan_type_code (string), status_code (string), margin_bps (bps)\n"
                "- loan_type_code MUST be exactly one of: 'ABR', 'TERM_BENCHMARK'\n"
                "- status_code MUST be the Roman numeral extracted from the Status headers (I, II, III, IV, V, VI).\n"
                "- rows MUST include exactly 12 rows (6 per loan_type_code).\n"
                "- derived fn_id MUST be ApplicableMargin\n"
                "- semantic_role MUST be margin\n"
                "- args: loan_type_code (type=string), status_code (type=string)\n"
                "- expr MUST compute: bps_to_rate(lookup2(\"ApplicableMarginGrid\",\"loan_type_code\",loan_type_code,\"status_code\",status_code,\"margin_bps\"))\n"
            ),
            evals=[
                EvalSpec(
                    fn_id="ApplicableMargin",
                    args={"loan_type_code": "ABR", "status_code": "V"},
                    indices={},
                    expected_rate="0.0005",
                ),
                EvalSpec(
                    fn_id="ApplicableMargin",
                    args={"loan_type_code": "TERM_BENCHMARK", "status_code": "V"},
                    indices={},
                    expected_rate="0.0105",
                ),
            ],
        ),
        # --- Margin: leverage ratio table (2 columns) ----------------------------------------------
        Experiment(
            exp_id="M4A_leverage_ratio_specialized_ABR",
            prompt_path="prompts/contract_ir_margin_v1_specialized.txt",
            item_id=leverage_item,
            anchor_ids=[leverage_anchor],
            task=(
                f"Build ContractIR for item_id={leverage_item}.\n"
                "Goal: extract the Applicable Margin table and expose an ABR-only margin lookup.\n"
                "Requirements:\n"
                "- Create table_id: ApplicableMarginByLeverageRatio\n"
                "- columns MUST include: leverage_ratio_bucket (string), abr_margin_bps (bps)\n"
                "- rows MUST include all leverage ratio buckets from the table; store values as bps.\n"
                "- derived fn_id MUST be ApplicableMarginABR\n"
                "- semantic_role MUST be margin\n"
                "- args: leverage_ratio_bucket (type=string)\n"
                "- expr MUST compute: bps_to_rate(lookup(\"ApplicableMarginByLeverageRatio\",\"leverage_ratio_bucket\",leverage_ratio_bucket,\"abr_margin_bps\"))\n"
            ),
            evals=[
                EvalSpec(
                    fn_id="ApplicableMarginABR",
                    args={"leverage_ratio_bucket": "Greater than 5.00 to 1"},
                    indices={},
                    expected_rate="0.00125",
                )
            ],
        ),
        Experiment(
            exp_id="M4B_leverage_ratio_lookup2_longtable",
            prompt_path="prompts/contract_ir_margin_v1_lookup2.txt",
            item_id=leverage_item,
            anchor_ids=[leverage_anchor],
            task=(
                f"Build ContractIR for item_id={leverage_item}.\n"
                "Goal: normalize the Applicable Margin leverage ratio table into a long table and expose ONE generic lookup function.\n"
                "Requirements:\n"
                "- Create table_id: ApplicableMarginGrid\n"
                "- columns MUST include: loan_type_code (string), leverage_ratio_bucket (string), margin_bps (bps)\n"
                "- loan_type_code MUST be exactly one of: 'ABR', 'EURODOLLAR'\n"
                "- rows MUST include all combinations present in the table.\n"
                "- derived fn_id MUST be ApplicableMargin\n"
                "- semantic_role MUST be margin\n"
                "- args: loan_type_code (type=string), leverage_ratio_bucket (type=string)\n"
                "- expr MUST compute: bps_to_rate(lookup2(\"ApplicableMarginGrid\",\"loan_type_code\",loan_type_code,\"leverage_ratio_bucket\",leverage_ratio_bucket,\"margin_bps\"))\n"
            ),
            evals=[
                EvalSpec(
                    fn_id="ApplicableMargin",
                    args={"loan_type_code": "ABR", "leverage_ratio_bucket": "Greater than 5.00 to 1"},
                    indices={},
                    expected_rate="0.00125",
                ),
                EvalSpec(
                    fn_id="ApplicableMargin",
                    args={"loan_type_code": "EURODOLLAR", "leverage_ratio_bucket": "Greater than 5.00 to 1"},
                    indices={},
                    expected_rate="0.01125",
                ),
            ],
        ),
        # --- Margin: rating table (2 columns) -----------------------------------------------------
        Experiment(
            exp_id="M5A_rating_specialized_Eurodollar",
            prompt_path="prompts/contract_ir_margin_v1_specialized.txt",
            item_id=rating_item,
            anchor_ids=[rating_anchor],
            task=(
                f"Build ContractIR for item_id={rating_item}.\n"
                "Goal: extract the Applicable Margin rating table and expose a Eurodollar-only margin lookup.\n"
                "Requirements:\n"
                "- Create table_id: ApplicableMarginByRating\n"
                "- columns MUST include: rating (string), eurodollar_bps (bps), base_rate_bps (bps)\n"
                "- rows MUST include all rating buckets from the table; store values as bps.\n"
                "- derived fn_id MUST be ApplicableMarginEurodollar\n"
                "- semantic_role MUST be margin\n"
                "- args: rating (type=string)\n"
                "- expr MUST compute: bps_to_rate(lookup(\"ApplicableMarginByRating\",\"rating\",rating,\"eurodollar_bps\"))\n"
            ),
            evals=[
                EvalSpec(
                    fn_id="ApplicableMarginEurodollar",
                    args={"rating": "BBB/Baa2"},
                    indices={},
                    expected_rate="0.0075",
                )
            ],
        ),
        Experiment(
            exp_id="M5B_rating_lookup2_longtable",
            prompt_path="prompts/contract_ir_margin_v1_lookup2.txt",
            item_id=rating_item,
            anchor_ids=[rating_anchor],
            task=(
                f"Build ContractIR for item_id={rating_item}.\n"
                "Goal: normalize the Applicable Margin rating table into a long table and expose ONE generic lookup function.\n"
                "Requirements:\n"
                "- Create table_id: ApplicableMarginGrid\n"
                "- columns MUST include: loan_type_code (string), rating (string), margin_bps (bps)\n"
                "- loan_type_code MUST be exactly one of: 'EURODOLLAR', 'BASE_RATE'\n"
                "- rows MUST include all combinations present in the table.\n"
                "- derived fn_id MUST be ApplicableMargin\n"
                "- semantic_role MUST be margin\n"
                "- args: loan_type_code (type=string), rating (type=string)\n"
                "- expr MUST compute: bps_to_rate(lookup2(\"ApplicableMarginGrid\",\"loan_type_code\",loan_type_code,\"rating\",rating,\"margin_bps\"))\n"
            ),
            evals=[
                EvalSpec(
                    fn_id="ApplicableMargin",
                    args={"loan_type_code": "EURODOLLAR", "rating": "BBB/Baa2"},
                    indices={},
                    expected_rate="0.0075",
                ),
                EvalSpec(
                    fn_id="ApplicableMargin",
                    args={"loan_type_code": "BASE_RATE", "rating": "BBB/Baa2"},
                    indices={},
                    expected_rate="0",
                ),
            ],
        ),
        # --- Base rate: ABR conditional (open_items vs boolean arg) --------------------------------
        Experiment(
            exp_id="B1_ABR_conditional_open_item",
            prompt_path="prompts/contract_ir_base_rate_v1_open_items.txt",
            item_id=abr_conditional_item,
            anchor_ids=[abr_conditional_anchor],
            task=(
                f"Build ContractIR for item_id={abr_conditional_item}.\n"
                "Goal: encode the ABR definition as a derived function.\n"
                "Requirements:\n"
                "- derived fn_id MUST be ABR\n"
                "- semantic_role MUST be base_rate\n"
                "- args: date (type=date)\n"
                "- indices MUST include PrimeRate (unit=rate), NYFRBRate (unit=rate), AdjustedTermSOFRRate1M (unit=rate)\n"
                "- expr MUST compute: max(PrimeRate, NYFRBRate + 0.5%, AdjustedTermSOFRRate1M + 1.0%)\n"
                "- If the excerpt contains logic excluding clause (c) in some circumstances, record that as a blocking open_item.\n"
            ),
            evals=[
                EvalSpec(
                    fn_id="ABR",
                    args={"date": "2023-01-03"},
                    indices=indices_for_modern_abr,
                    expected_rate="0.0590",
                )
            ],
        ),
        Experiment(
            exp_id="B2_ABR_conditional_boolean_arg",
            prompt_path="prompts/contract_ir_base_rate_v1_bool_args.txt",
            item_id=abr_conditional_item,
            anchor_ids=[abr_conditional_anchor],
            task=(
                f"Build ContractIR for item_id={abr_conditional_item}.\n"
                "Goal: encode the ABR definition as a derived function with a boolean arg for the clause (c) exclusion.\n"
                "Requirements:\n"
                "- derived fn_id MUST be ABR\n"
                "- semantic_role MUST be base_rate\n"
                "- args: date (type=date), exclude_clause_c (type=bool)\n"
                "- indices MUST include PrimeRate (unit=rate), NYFRBRate (unit=rate), AdjustedTermSOFRRate1M (unit=rate)\n"
                "- expr MUST compute: if(exclude_clause_c, max(PrimeRate, NYFRBRate + 0.5%), max(PrimeRate, NYFRBRate + 0.5%, AdjustedTermSOFRRate1M + 1.0%))\n"
            ),
            evals=[
                EvalSpec(
                    fn_id="ABR",
                    args={"date": "2023-01-03", "exclude_clause_c": False},
                    indices=indices_for_modern_abr,
                    expected_rate="0.0590",
                ),
                EvalSpec(
                    fn_id="ABR",
                    args={"date": "2023-01-03", "exclude_clause_c": True},
                    indices=indices_for_modern_abr,
                    expected_rate="0.0570",
                ),
            ],
        ),
        # --- Base rate: Term SOFR with floor -------------------------------------------------------
        Experiment(
            exp_id="B3_TermSOFR_floor_open_item",
            prompt_path="prompts/contract_ir_base_rate_v1_open_items.txt",
            item_id=term_sofr_item,
            anchor_ids=[term_sofr_anchor],
            task=(
                f"Build ContractIR for item_id={term_sofr_item}.\n"
                "Goal: encode the definition of Term SOFR (1M) including the explicit 2.50% floor.\n"
                "Requirements:\n"
                "- derived fn_id MUST be TermSOFR1M\n"
                "- semantic_role MUST be base_rate\n"
                "- args: date (type=date)\n"
                "- indices MUST include TermSOFRReferenceRate1M (unit=rate)\n"
                "- expr MUST compute: max(index(TermSOFRReferenceRate1M,date), 0.025)\n"
                "- If the excerpt contains publication fallback logic or lookback rules, record those as open_items (do not guess date shifting).\n"
            ),
            evals=[
                EvalSpec(
                    fn_id="TermSOFR1M",
                    args={"date": "2023-01-03"},
                    indices={"TermSOFRReferenceRate1M": {"2023-01-03": "0.0200", "2023-01-04": "0.0300"}},
                    expected_rate="0.025",
                ),
                EvalSpec(
                    fn_id="TermSOFR1M",
                    args={"date": "2023-01-04"},
                    indices={"TermSOFRReferenceRate1M": {"2023-01-03": "0.0200", "2023-01-04": "0.0300"}},
                    expected_rate="0.03",
                ),
            ],
        ),
        # --- Base rate: Alternate Base Rate w/ rounding (1996 agreement) ---------------------------
        Experiment(
            exp_id="B4_AlternateBaseRate_rounding_1996",
            prompt_path="prompts/contract_ir_base_rate_v1_open_items.txt",
            item_id=abr_pricing_levels_item,
            anchor_ids=[abr_pricing_levels_base_rate_anchor],
            task=(
                f"Build ContractIR for item_id={abr_pricing_levels_item}.\n"
                "Goal: encode the definition of ALTERNATE BASE RATE.\n"
                "Requirements:\n"
                "- derived fn_id MUST be AlternateBaseRate\n"
                "- semantic_role MUST be base_rate\n"
                "- args: date (type=date)\n"
                "- indices MUST include PrimeRate (unit=rate), FederalFundsRate (unit=rate)\n"
                "- expr MUST compute: round_up_to_increment(max(PrimeRate, FederalFundsRate + 0.5%), 0.0001)\n"
            ),
            evals=[
                EvalSpec(
                    fn_id="AlternateBaseRate",
                    args={"date": "1997-03-31"},
                    indices=indices_for_alt_base_rate_rounding,
                    expected_rate="0.055",
                )
            ],
        ),
        # --- Margin: Pricing Level tables (ABR margin + commitment fee + Eurodollar margin) -------
        Experiment(
            exp_id="M6_PricingLevel_ABR_margin_table_1996",
            prompt_path="prompts/contract_ir_margin_v1_specialized.txt",
            item_id=abr_pricing_levels_item,
            anchor_ids=[abr_pricing_levels_margins_anchor],
            task=(
                f"Build ContractIR for item_id={abr_pricing_levels_item}.\n"
                "Goal: extract ONLY the 'APPLICABLE ALTERNATE BASE RATE MARGIN' Pricing Level schedule.\n"
                "Requirements:\n"
                "- Create table_id: ApplicableABRMarginByPricingLevel\n"
                "- columns MUST include: pricing_level (string), abr_margin_bps (bps)\n"
                "- rows MUST include Pricing Levels: I, II, III, IV.\n"
                "- derived fn_id MUST be ApplicableABRMargin\n"
                "- semantic_role MUST be margin\n"
                "- args: pricing_level (type=string)\n"
                "- expr MUST compute: bps_to_rate(lookup(\"ApplicableABRMarginByPricingLevel\",\"pricing_level\",pricing_level,\"abr_margin_bps\"))\n"
            ),
            evals=[
                EvalSpec(
                    fn_id="ApplicableABRMargin",
                    args={"pricing_level": "III"},
                    indices={},
                    expected_rate="0.0025",
                )
            ],
        ),
        Experiment(
            exp_id="M7_PricingLevel_commitment_fee_table_1996",
            prompt_path="prompts/contract_ir_margin_v1_specialized.txt",
            item_id=abr_pricing_levels_item,
            anchor_ids=[abr_pricing_levels_margins_anchor],
            task=(
                f"Build ContractIR for item_id={abr_pricing_levels_item}.\n"
                "Goal: extract ONLY the 'APPLICABLE COMMITMENT FEE RATE' Pricing Level schedule.\n"
                "Requirements:\n"
                "- Create table_id: ApplicableCommitmentFeeRateByPricingLevel\n"
                "- columns MUST include: pricing_level (string), commitment_fee_bps (bps)\n"
                "- rows MUST include Pricing Levels: I, II, III, IV.\n"
                "- derived fn_id MUST be ApplicableCommitmentFeeRate\n"
                "- semantic_role MUST be margin\n"
                "- args: pricing_level (type=string)\n"
                "- expr MUST compute: bps_to_rate(lookup(\"ApplicableCommitmentFeeRateByPricingLevel\",\"pricing_level\",pricing_level,\"commitment_fee_bps\"))\n"
            ),
            evals=[
                EvalSpec(
                    fn_id="ApplicableCommitmentFeeRate",
                    args={"pricing_level": "IV"},
                    indices={},
                    expected_rate="0.00125",
                )
            ],
        ),
        Experiment(
            exp_id="M8_PricingLevel_eurodollar_margin_table_1996",
            prompt_path="prompts/contract_ir_margin_v1_specialized.txt",
            item_id=abr_pricing_levels_item,
            anchor_ids=[abr_pricing_levels_euro_margin_anchor],
            task=(
                f"Build ContractIR for item_id={abr_pricing_levels_item}.\n"
                "Goal: extract ONLY the 'APPLICABLE EURODOLLAR RATE MARGIN' Pricing Level schedule.\n"
                "Requirements:\n"
                "- Create table_id: ApplicableEurodollarMarginByPricingLevel\n"
                "- columns MUST include: pricing_level (string), eurodollar_margin_bps (bps)\n"
                "- rows MUST include Pricing Levels: I, II, III, IV.\n"
                "- derived fn_id MUST be ApplicableEurodollarMargin\n"
                "- semantic_role MUST be margin\n"
                "- args: pricing_level (type=string)\n"
                "- expr MUST compute: bps_to_rate(lookup(\"ApplicableEurodollarMarginByPricingLevel\",\"pricing_level\",pricing_level,\"eurodollar_margin_bps\"))\n"
            ),
            evals=[
                EvalSpec(
                    fn_id="ApplicableEurodollarMargin",
                    args={"pricing_level": "IV"},
                    indices={},
                    expected_rate="0.0075",
                )
            ],
        ),
        # --- Margin: S&P rating grid (ABR/LIBOR/CD) with conditional add-on ------------------------
        Experiment(
            exp_id="M9_SP_rating_lookup2_longtable_LIBOR",
            prompt_path="prompts/contract_ir_margin_v1_lookup2.txt",
            item_id=sp_rating_grid_item,
            anchor_ids=[sp_rating_grid_anchor, sp_rating_grid_with_conditional_anchor],
            task=(
                f"Build ContractIR for item_id={sp_rating_grid_item}.\n"
                "Goal: normalize the 'APPLICABLE MARGIN' S&P rating grid into a long table and expose ONE generic lookup function.\n"
                "Requirements:\n"
                "- Create table_id: ApplicableMarginGrid\n"
                "- columns MUST include: loan_type_code (string), sp_rating_bucket (string), margin_bps (bps)\n"
                "- loan_type_code MUST be exactly one of: 'ABR', 'LIBOR', 'CD'\n"
                "- sp_rating_bucket MUST match the rating bucket text from the excerpt (e.g., 'BB+').\n"
                "- Convert percent values to bps (e.g., 1.50% => 150 bps).\n"
                "- derived fn_id MUST be ApplicableMargin\n"
                "- semantic_role MUST be margin\n"
                "- args: loan_type_code (type=string), sp_rating_bucket (type=string)\n"
                "- expr MUST compute: bps_to_rate(lookup2(\"ApplicableMarginGrid\",\"loan_type_code\",loan_type_code,\"sp_rating_bucket\",sp_rating_bucket,\"margin_bps\"))\n"
                "- The excerpt also contains a conditional add-on (+0.15% per annum when Advances > $10,000,000). Do NOT bake that into the table; record it as a blocking open_item.\n"
            ),
            evals=[
                EvalSpec(
                    fn_id="ApplicableMargin",
                    args={"loan_type_code": "LIBOR", "sp_rating_bucket": "BB+"},
                    indices={},
                    expected_rate="0.015",
                )
            ],
            expect_open_items_min=1,
        ),
        # --- Margin: net income tiers (range buckets + special case) -------------------------------
        Experiment(
            exp_id="M10_NetIncome_spread_nested_if",
            prompt_path="prompts/contract_ir_margin_v1_specialized.txt",
            item_id=net_income_grid_item,
            anchor_ids=[net_income_grid_anchor],
            task=(
                f"Build ContractIR for item_id={net_income_grid_item}.\n"
                "Goal: encode the 'Libor Option Rate' spread schedule as a derived function.\n"
                "Requirements:\n"
                "- derived fn_id MUST be LiborOptionSpread\n"
                "- semantic_role MUST be margin\n"
                "- args: net_income (type=money)\n"
                "- returns MUST be rate\n"
                "- Implement the tiering as nested if/lt/lte/gt/gte comparisons on net_income.\n"
                "- Convert bps to rate constants (e.g., 150 bps => 0.015).\n"
                "- The table includes a 'Below 0' row that references Prime Rate (not a LIBOR spread). Do NOT guess that behavior; record it as a blocking open_item.\n"
            ),
            evals=[
                EvalSpec(
                    fn_id="LiborOptionSpread",
                    args={"net_income": "3000000"},
                    indices={},
                    expected_rate="0.015",
                )
            ],
            expect_open_items_min=1,
        ),
        # --- Margin: 2D ratio condition schedule (range + conjunction/or) --------------------------
        Experiment(
            exp_id="M11_RatioConditionMargin_nested_if",
            prompt_path="prompts/contract_ir_margin_v1_specialized.txt",
            item_id=ratios_margin_item,
            anchor_ids=[ratios_margin_anchor],
            task=(
                f"Build ContractIR for item_id={ratios_margin_item}.\n"
                "Goal: encode the Applicable Margin adjustment criteria (Interest Coverage Ratio + Fixed Charge Coverage Ratio) as a derived function.\n"
                "Requirements:\n"
                "- derived fn_id MUST be ApplicableMarginLIBO\n"
                "- semantic_role MUST be margin\n"
                "- args: interest_coverage_ratio (type=decimal), fixed_charge_coverage_ratio (type=decimal)\n"
                "- returns MUST be rate\n"
                "- Implement the 3-row criteria as nested if with and/or and comparisons.\n"
                "- Use rate literals for the margin values (2.50% => 0.025, 2.25% => 0.0225, 2.00% => 0.02).\n"
            ),
            evals=[
                EvalSpec(
                    fn_id="ApplicableMarginLIBO",
                    args={"interest_coverage_ratio": "2.60", "fixed_charge_coverage_ratio": "1.45"},
                    indices={},
                    expected_rate="0.02",
                )
            ],
        ),
        # --- Base rate: Index Rate (max of multiple bank primes, CP, Fed Funds + 0.5) -------------
        Experiment(
            exp_id="B5_IndexRate_max_of_primes_cp_fedfunds",
            prompt_path="prompts/contract_ir_base_rate_v1_open_items.txt",
            item_id=index_rate_item,
            anchor_ids=[index_rate_anchor],
            task=(
                f"Build ContractIR for item_id={index_rate_item}.\n"
                "Goal: encode the Index Rate definition.\n"
                "Requirements:\n"
                "- derived fn_id MUST be IndexRate\n"
                "- semantic_role MUST be base_rate\n"
                "- args: date (type=date)\n"
                "- indices MUST include the following series_id exactly:\n"
                "  - BankersTrustPrimeRate\n"
                "  - ChemicalBankPrimeRate\n"
                "  - CitibankPrimeRate\n"
                "  - MorganGuarantyPrimeRate\n"
                "  - ChaseManhattanPrimeRate\n"
                "  - CPRate\n"
                "  - FederalFundsRate\n"
                "- expr MUST compute: max(max(BankersTrustPrimeRate, ChemicalBankPrimeRate, CitibankPrimeRate, MorganGuarantyPrimeRate, ChaseManhattanPrimeRate), CPRate, FederalFundsRate + 0.5%)\n"
            ),
            evals=[
                EvalSpec(
                    fn_id="IndexRate",
                    args={"date": "1996-06-28"},
                    indices=indices_for_index_rate,
                    expected_rate="0.0565",
                )
            ],
        ),
        # --- Margin: 2D status table (I-IV) normalized via lookup2 --------------------------------
        Experiment(
            exp_id="M12_StatusMarginGrid_lookup2_I_to_IV",
            prompt_path="prompts/contract_ir_margin_v1_lookup2.txt",
            item_id=att_item,
            anchor_ids=[att_status_margin_anchor],
            task=(
                f"Build ContractIR for item_id={att_item}.\n"
                "Goal: normalize the Applicable Margin table (Euro-Dollar Loans vs CD Loans) by Status levels (I-IV).\n"
                "Requirements:\n"
                "- Create table_id: ApplicableMarginGrid\n"
                "- columns MUST include: loan_type_code (string), status_code (string), margin_bps (bps)\n"
                "- loan_type_code MUST be exactly one of: 'EURODOLLAR', 'CD'\n"
                "- status_code MUST be exactly one of: 'I','II','III','IV'\n"
                "- Convert percent values to bps (e.g., 0.2000% => 20 bps).\n"
                "- rows MUST include exactly 8 rows (2 loan types x 4 statuses)\n"
                "- derived fn_id MUST be ApplicableMargin\n"
                "- semantic_role MUST be margin\n"
                "- args: loan_type_code (type=string), status_code (type=string)\n"
                "- expr MUST compute: bps_to_rate(lookup2(\"ApplicableMarginGrid\",\"loan_type_code\",loan_type_code,\"status_code\",status_code,\"margin_bps\"))\n"
            ),
            evals=[
                EvalSpec(
                    fn_id="ApplicableMargin",
                    args={"loan_type_code": "EURODOLLAR", "status_code": "III"},
                    indices={},
                    expected_rate="0.00375",
                )
            ],
        ),
        # --- Margin: facility fee schedule (1D mapping) -------------------------------------------
        Experiment(
            exp_id="M13_FacilityFeeRate_by_status",
            prompt_path="prompts/contract_ir_margin_v1_specialized.txt",
            item_id=att_item,
            anchor_ids=[att_facility_fee_anchor],
            task=(
                f"Build ContractIR for item_id={att_item}.\n"
                "Goal: extract the Facility Fee Rate schedule by Status level.\n"
                "Requirements:\n"
                "- Create table_id: FacilityFeeRateByStatus\n"
                "- columns MUST include: status_code (string), facility_fee_bps (bps)\n"
                "- rows MUST include Status levels: I, II, III, IV.\n"
                "- Convert percent values to bps (e.g., 0.0500% => 5 bps).\n"
                "- derived fn_id MUST be FacilityFeeRate\n"
                "- semantic_role MUST be margin\n"
                "- args: status_code (type=string)\n"
                "- expr MUST compute: bps_to_rate(lookup(\"FacilityFeeRateByStatus\",\"status_code\",status_code,\"facility_fee_bps\"))\n"
            ),
            evals=[
                EvalSpec(
                    fn_id="FacilityFeeRate",
                    args={"status_code": "III"},
                    indices={},
                    expected_rate="0.00125",
                )
            ],
        ),
        # --- Base rate: simple Base Rate definition (Prime vs Fed Funds + 0.5) ---------------------
        Experiment(
            exp_id="B6_BaseRate_simple_prime_vs_fedfunds",
            prompt_path="prompts/contract_ir_base_rate_v1_open_items.txt",
            item_id=att_item,
            anchor_ids=[att_base_rate_anchor],
            task=(
                f"Build ContractIR for item_id={att_item}.\n"
                "Goal: encode the Base Rate definition.\n"
                "Requirements:\n"
                "- derived fn_id MUST be BaseRate\n"
                "- semantic_role MUST be base_rate\n"
                "- args: date (type=date)\n"
                "- indices MUST include PrimeRate (unit=rate), FederalFundsRate (unit=rate)\n"
                "- expr MUST compute: max(index(PrimeRate,date), index(FederalFundsRate,date) + 0.5%)\n"
            ),
            evals=[
                EvalSpec(
                    fn_id="BaseRate",
                    args={"date": "1996-12-31"},
                    indices=indices_for_simple_base_rate,
                    expected_rate="0.0525",
                )
            ],
        ),
        # --- Base rate: reserve adjustment / adjusted LIBO -----------------------------------------
        Experiment(
            exp_id="B7_AdjustedLIBO_div_by_one_minus_reserve",
            prompt_path="prompts/contract_ir_base_rate_v1_open_items.txt",
            item_id=reserve_adjust_item,
            anchor_ids=[reserve_adjust_anchor],
            task=(
                f"Build ContractIR for item_id={reserve_adjust_item}.\n"
                "Goal: encode the Euro Reserve Percentage adjustment formula.\n"
                "Requirements:\n"
                "- derived fn_id MUST be AdjustedLIBO\n"
                "- semantic_role MUST be base_rate\n"
                "- args: libo_rate (type=rate), euro_reserve_pct (type=decimal)\n"
                "- returns MUST be rate\n"
                "- expr MUST compute: div(libo_rate, sub(1.00, euro_reserve_pct))\n"
            ),
            evals=[
                EvalSpec(
                    fn_id="AdjustedLIBO",
                    args={"libo_rate": "0.045", "euro_reserve_pct": "0.10"},
                    indices={},
                    expected_rate="0.05",
                )
            ],
        ),
        # --- Margin: leverage ratio buckets (ranges) ----------------------------------------------
        Experiment(
            exp_id="M14_LeverageBucketMargin_nested_if",
            prompt_path="prompts/contract_ir_margin_v1_specialized.txt",
            item_id=leverage_bucket_item,
            anchor_ids=[leverage_bucket_anchor],
            task=(
                f"Build ContractIR for item_id={leverage_bucket_item}.\n"
                "Goal: encode the Applicable Borrowing Margin schedule for Eurodollar Loans (by leverage ratio buckets).\n"
                "Requirements:\n"
                "- derived fn_id MUST be EurodollarBorrowingMargin\n"
                "- semantic_role MUST be margin\n"
                "- args: leverage_ratio (type=decimal)\n"
                "- returns MUST be rate\n"
                "- Implement the bucket logic as nested if with comparisons.\n"
                "- Use rate literals from the excerpt (e.g., 1.125% => 0.01125, 0.9375% => 0.009375).\n"
            ),
            evals=[
                EvalSpec(
                    fn_id="EurodollarBorrowingMargin",
                    args={"leverage_ratio": "2.50"},
                    indices={},
                    expected_rate="0.009375",
                )
            ],
        ),
        # --- Margin: modern 2x2 tables (>= / < thresholds) ----------------------------------------
        Experiment(
            exp_id="M15_ModernApplicableRate_lookup2",
            prompt_path="prompts/contract_ir_margin_v1_lookup2.txt",
            item_id=modern_small_tables_item,
            anchor_ids=[modern_applicable_rate_anchor],
            task=(
                f"Build ContractIR for item_id={modern_small_tables_item}.\n"
                "Goal: normalize the Applicable Rate table for Revolving Loans (Base Rate Loans vs Eurocurrency Rate Loans) into a long table.\n"
                "Requirements:\n"
                "- Create table_id: ApplicableRateGrid\n"
                "- columns MUST include: loan_type_code (string), leverage_bucket (string), margin_bps (bps)\n"
                "- loan_type_code MUST be exactly one of: 'BASE_RATE', 'EUROCURRENCY'\n"
                "- leverage_bucket MUST be exactly one of: 'GE_3.00', 'LT_3.00'\n"
                "- Convert percent values to bps (e.g., 1.75% => 175 bps).\n"
                "- rows MUST include exactly 4 rows (2 loan types x 2 buckets)\n"
                "- derived fn_id MUST be ApplicableRate\n"
                "- semantic_role MUST be margin\n"
                "- args: loan_type_code (type=string), leverage_bucket (type=string)\n"
                "- expr MUST compute: bps_to_rate(lookup2(\"ApplicableRateGrid\",\"loan_type_code\",loan_type_code,\"leverage_bucket\",leverage_bucket,\"margin_bps\"))\n"
            ),
            evals=[
                EvalSpec(
                    fn_id="ApplicableRate",
                    args={"loan_type_code": "EUROCURRENCY", "leverage_bucket": "GE_3.00"},
                    indices={},
                    expected_rate="0.0275",
                )
            ],
        ),
        Experiment(
            exp_id="M16_ModernCommitmentFee_by_bucket",
            prompt_path="prompts/contract_ir_margin_v1_specialized.txt",
            item_id=modern_small_tables_item,
            anchor_ids=[modern_commitment_fee_anchor],
            task=(
                f"Build ContractIR for item_id={modern_small_tables_item}.\n"
                "Goal: extract the Commitment Fee table by leverage ratio threshold.\n"
                "Requirements:\n"
                "- Create table_id: CommitmentFeeByBucket\n"
                "- columns MUST include: leverage_bucket (string), commitment_fee_bps (bps)\n"
                "- leverage_bucket MUST be exactly one of: 'GE_4.75', 'LT_4.75'\n"
                "- Convert percent values to bps (e.g., 0.50% => 50 bps).\n"
                "- rows MUST include exactly 2 rows.\n"
                "- derived fn_id MUST be CommitmentFeeRate\n"
                "- semantic_role MUST be margin\n"
                "- args: leverage_bucket (type=string)\n"
                "- expr MUST compute: bps_to_rate(lookup(\"CommitmentFeeByBucket\",\"leverage_bucket\",leverage_bucket,\"commitment_fee_bps\"))\n"
            ),
            evals=[
                EvalSpec(
                    fn_id="CommitmentFeeRate",
                    args={"leverage_bucket": "GE_4.75"},
                    indices={},
                    expected_rate="0.005",
                )
            ],
        ),
        # --- Stress tests: try to break conformance / avoid hallucinations -------------------------
        Experiment(
            exp_id="S1_MultiAnchor_noise_margin_plus_fee",
            prompt_path="prompts/contract_ir_margin_v1_lookup2.txt",
            item_id=att_item,
            anchor_ids=[att_status_margin_anchor, att_facility_fee_anchor],
            task=(
                f"Build ContractIR for item_id={att_item}.\n"
                "Goal: normalize ONLY the Applicable Margin table (Euro-Dollar Loans vs CD Loans) by Status levels.\n"
                "Requirements:\n"
                "- Create table_id: ApplicableMarginGrid\n"
                "- columns MUST include: loan_type_code (string), status_code (string), margin_bps (bps)\n"
                "- loan_type_code MUST be exactly one of: 'EURODOLLAR', 'CD'\n"
                "- status_code MUST be exactly one of: 'I','II','III','IV'\n"
                "- Convert percent values to bps (e.g., 0.2000% => 20 bps).\n"
                "- rows MUST include exactly 8 rows (2 loan types x 4 statuses)\n"
                "- derived fn_id MUST be ApplicableMargin\n"
                "- semantic_role MUST be margin\n"
                "- args: loan_type_code (type=string), status_code (type=string)\n"
                "- expr MUST compute: bps_to_rate(lookup2(\"ApplicableMarginGrid\",\"loan_type_code\",loan_type_code,\"status_code\",status_code,\"margin_bps\"))\n"
                "- Ignore any Facility Fee language in the context.\n"
            ),
            evals=[
                EvalSpec(
                    fn_id="ApplicableMargin",
                    args={"loan_type_code": "EURODOLLAR", "status_code": "III"},
                    indices={},
                    expected_rate="0.00375",
                )
            ],
        ),
        Experiment(
            exp_id="S2_Truncated_context_forces_guessing_bad",
            prompt_path="prompts/contract_ir_margin_v1_lookup2.txt",
            item_id=att_item,
            anchor_ids=[att_status_margin_anchor],
            context_stop_before="CD Loans",
            task=(
                f"Build ContractIR for item_id={att_item}.\n"
                "Goal: normalize the Applicable Margin table (Euro-Dollar Loans vs CD Loans) by Status levels.\n"
                "Requirements:\n"
                "- Create table_id: ApplicableMarginGrid\n"
                "- columns MUST include: loan_type_code (string), loan_type_label (string), status_code (string), status_label (string), margin_bps (bps)\n"
                "- loan_type_code MUST be exactly one of: 'EURODOLLAR', 'CD'\n"
                "- loan_type_label MUST be the exact row label text from the excerpt table (e.g., 'Euro-Dollar Loans', 'CD Loans').\n"
                "- status_code MUST be exactly one of: 'I','II','III','IV'\n"
                "- status_label MUST be EXACTLY the header text (e.g., 'Level III Status').\n"
                "- Convert percent values to bps (e.g., 0.2000% => 20 bps).\n"
                "- rows MUST include exactly 8 rows (2 loan types x 4 statuses)\n"
                "- derived fn_id MUST be ApplicableMargin\n"
                "- semantic_role MUST be margin\n"
                "- args: loan_type_code (type=string), status_code (type=string)\n"
                "- expr MUST compute: bps_to_rate(lookup2(\"ApplicableMarginGrid\",\"loan_type_code\",loan_type_code,\"status_code\",status_code,\"margin_bps\"))\n"
            ),
            evals=[
                EvalSpec(
                    fn_id="ApplicableMargin",
                    args={"loan_type_code": "CD", "status_code": "IV"},
                    indices={},
                    expected_rate="0.00675",
                )
            ],
        ),
        Experiment(
            exp_id="S3_Truncated_context_should_open_item_not_guess",
            prompt_path="prompts/contract_ir_margin_v1_lookup2.txt",
            item_id=att_item,
            anchor_ids=[att_status_margin_anchor],
            context_stop_before="CD Loans",
            task=(
                f"Build ContractIR for item_id={att_item}.\n"
                "Goal: normalize the Applicable Margin table (Euro-Dollar Loans vs CD Loans) by Status levels.\n"
                "Requirements:\n"
                "- If any required row/cell is not explicitly present in the provided CONTEXT, DO NOT GUESS values.\n"
                "  Instead: emit a blocking open_item describing the missing cells, and set tables=[] and derived=[].\n"
            ),
            evals=[],
            expect_open_items_min=1,
        ),
        Experiment(
            exp_id="S4_Ambiguous_row_selection_should_open_item",
            prompt_path="prompts/contract_ir_margin_v1_specialized.txt",
            item_id=att_item,
            anchor_ids=[att_status_margin_anchor],
            task=(
                f"Build ContractIR for item_id={att_item}.\n"
                "Goal: extract the Applicable Margin table.\n"
                "Requirements:\n"
                "- The table contains multiple loan-type rows and this TASK does not specify which row to extract.\n"
                "- You MUST NOT guess which row is intended.\n"
                "- Emit a blocking open_item describing the ambiguity, and set tables=[] and derived=[].\n"
            ),
            evals=[],
            expect_open_items_min=1,
        ),
        Experiment(
            exp_id="S5_Truncated_context_default_prompt_policy_open_item",
            prompt_path="prompts/contract_ir_margin_v1_lookup2.txt",
            item_id=att_item,
            anchor_ids=[att_status_margin_anchor],
            context_stop_before="CD Loans",
            task=(
                f"Build ContractIR for item_id={att_item}.\n"
                "Goal: normalize the Applicable Margin table (Euro-Dollar Loans vs CD Loans) by Status levels.\n"
                "Requirements:\n"
                "- Create table_id: ApplicableMarginGrid\n"
                "- columns MUST include: loan_type_code (string), status_code (string), margin_bps (bps)\n"
                "- loan_type_code MUST be exactly one of: 'EURODOLLAR', 'CD'\n"
                "- status_code MUST be exactly one of: 'I','II','III','IV'\n"
                "- Convert percent values to bps (e.g., 0.2000% => 20 bps).\n"
                "- rows MUST include exactly 8 rows (2 loan types x 4 statuses)\n"
                "- derived fn_id MUST be ApplicableMargin\n"
                "- semantic_role MUST be margin\n"
                "- args: loan_type_code (type=string), status_code (type=string)\n"
                "- expr MUST compute: bps_to_rate(lookup2(\"ApplicableMarginGrid\",\"loan_type_code\",loan_type_code,\"status_code\",status_code,\"margin_bps\"))\n"
            ),
            evals=[],
            expect_open_items_min=1,
        ),
        Experiment(
            exp_id="S6_Truncated_context_specialized_row_should_open_item",
            prompt_path="prompts/contract_ir_margin_v1_specialized.txt",
            item_id=att_item,
            anchor_ids=[att_status_margin_anchor],
            context_stop_before="CD Loans",
            task=(
                f"Build ContractIR for item_id={att_item}.\n"
                "Goal: extract ONLY the 'CD Loans' row from the Applicable Margin table.\n"
                "Requirements:\n"
                "- Create table_id: ApplicableMarginCDByStatus\n"
                "- columns MUST include: status_code (string), status_label (string), cd_margin_bps (bps)\n"
                "- rows MUST include exactly 4 rows for Status levels I, II, III, IV.\n"
                "- derived fn_id MUST be ApplicableMarginCD\n"
                "- semantic_role MUST be margin\n"
                "- args: status_code (type=string)\n"
                "- expr MUST compute: bps_to_rate(lookup(\"ApplicableMarginCDByStatus\",\"status_code\",status_code,\"cd_margin_bps\"))\n"
            ),
            evals=[],
            expect_open_items_min=1,
        ),
    ]

    if args.exp_filter:
        experiments = [e for e in experiments if args.exp_filter in e.exp_id]
        if not experiments:
            raise ValueError(f"No experiments matched exp-filter={args.exp_filter!r}")

    complete_response_sync = _ensure_gateway_client_sync()

    run_summary: Dict[str, Any] = {
        "schema_version": "contractir_strategy_tests_v0",
        "source_run_id": args.source_run_id,
        "out_run_id": out_run_id,
        "model": REQUIRED_MODEL,
        "reasoning_effort": REQUIRED_REASONING,
        "gateway_url": args.gateway_url,
        "experiments": [],
    }

    for exp in experiments:
        exp_dir = out_dir / exp.exp_id
        exp_dir.mkdir(parents=True, exist_ok=True)

        snippets_path = PROJECT_ROOT / "runs" / args.source_run_id / "retrieval_v2" / f"{exp.item_id}_snippets.jsonl"
        if not snippets_path.exists():
            raise FileNotFoundError(f"Missing retrieval snippets: {snippets_path}")

        template = _read_text(PROJECT_ROOT / exp.prompt_path)
        contexts_full = build_excerpt_pack(snippets_path, exp.anchor_ids)
        contexts = contexts_full
        if exp.context_stop_before is not None:
            stop_idx = contexts.find(exp.context_stop_before)
            if stop_idx < 0:
                raise ValueError(
                    f"Experiment {exp.exp_id}: context_stop_before={exp.context_stop_before!r} not found in contexts"
                )
            contexts = contexts[:stop_idx].rstrip() + "\n"
        if exp.context_stop_before is not None or exp.context_clip_chars is not None:
            (exp_dir / "contexts_full.txt").write_text(contexts_full, encoding="utf-8")
        if exp.context_clip_chars is not None:
            contexts = contexts[: exp.context_clip_chars].rstrip() + "\n"
        (exp_dir / "contexts.txt").write_text(contexts, encoding="utf-8")

        prompt = _render_prompt(template, task=exp.task, contexts=contexts)
        (exp_dir / "prompt.txt").write_text(prompt, encoding="utf-8")

        attempt = 0
        repair_rounds_used: Optional[int] = None
        raw: Optional[str] = None
        parsed: Any = None
        validation_errors: List[ContractIRValidationError] = []
        anchor_errors: List[ContractIRValidationError] = []
        label_errors: List[ContractIRValidationError] = []

        while True:
            attempt += 1
            raw = complete_response_sync(
                model=REQUIRED_MODEL,
                prompt=prompt,
                base_url=args.gateway_url,
                reasoning={"effort": REQUIRED_REASONING},
                temperature=0.0,
                max_output_tokens=None,
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
                    if not anchor_errors:
                        label_errors = _validate_label_cells_in_context(parsed, contexts=contexts)

            if parsed is not None and not validation_errors and not anchor_errors and not label_errors:
                repair_rounds_used = (attempt - 1)
                break

            if attempt > 1 + max(0, args.max_repair_rounds):
                repair_rounds_used = None
                break

            prompt = _render_repair_prompt(raw_json=raw or "", errors=(validation_errors or anchor_errors or label_errors))
            (exp_dir / f"repair_prompt_round_{attempt-1}.txt").write_text(prompt, encoding="utf-8")

        open_items_count: Optional[int] = None
        if parsed is not None and not validation_errors and not anchor_errors and not label_errors:
            open_items = parsed.get("open_items")
            if isinstance(open_items, list):
                open_items_count = len(open_items)

        exp_result: Dict[str, Any] = {
            "exp_id": exp.exp_id,
            "prompt_path": exp.prompt_path,
            "item_id": exp.item_id,
            "anchor_ids": exp.anchor_ids,
            "attempts_used": attempt,
            "repair_rounds_used": repair_rounds_used,
            "parse_success": parsed is not None,
            "schema_valid": bool(parsed is not None and not validation_errors),
            "anchor_context_valid": bool(parsed is not None and not validation_errors and not anchor_errors),
            "label_context_valid": bool(parsed is not None and not validation_errors and not anchor_errors and not label_errors),
            "open_items_count": open_items_count,
            "evals": [],
        }

        if exp.expect_open_items_min is not None and open_items_count is not None:
            exp_result["expect_open_items_min"] = exp.expect_open_items_min
            exp_result["open_items_ok"] = open_items_count >= exp.expect_open_items_min

        if validation_errors:
            _write_json(exp_dir / "validation_errors.json", [asdict(e) for e in validation_errors])
        if anchor_errors:
            _write_json(exp_dir / "anchor_context_errors.json", [asdict(e) for e in anchor_errors])
        if label_errors:
            _write_json(exp_dir / "label_context_errors.json", [asdict(e) for e in label_errors])

        if parsed is not None and not validation_errors and not anchor_errors and not label_errors:
            _write_json(exp_dir / "contractir_validated.json", parsed)

            for spec in exp.evals:
                ev = {"fn_id": spec.fn_id, "expected_rate": spec.expected_rate, "args": dict(spec.args)}
                try:
                    out = evaluate_function(parsed, fn_id=spec.fn_id, args=spec.args, indices=spec.indices)
                    if out.kind != "rate":
                        raise TypeError(f"Expected rate output, got {out.kind}")
                    got = _decimal_str(out.value)
                    expected_norm = _decimal_str(Decimal(spec.expected_rate))
                    ev["got_rate"] = got
                    ev["numeric_correct"] = (got == expected_norm)
                    ev["eval_success"] = True
                except ContractIREvalError as e:
                    ev["eval_success"] = False
                    ev["numeric_correct"] = False
                    ev["error"] = str(e)
                except Exception as e:
                    ev["eval_success"] = False
                    ev["numeric_correct"] = False
                    ev["error"] = str(e)
                exp_result["evals"].append(ev)

        _write_json(exp_dir / "result.json", exp_result)
        run_summary["experiments"].append(exp_result)

    _write_json(out_dir / "summary.json", run_summary)

    # Human-readable rollup (for quick iteration).
    eval_total = 0
    eval_ok = 0
    for exp in run_summary["experiments"]:
        for ev in exp.get("evals") or []:
            eval_total += 1
            if ev.get("numeric_correct"):
                eval_ok += 1

    print(f"Wrote strategy test artifacts to {out_dir}")
    print(f"Summary: {out_dir / 'summary.json'}")
    print(f"[eval] numeric_correct: {eval_ok}/{eval_total}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
