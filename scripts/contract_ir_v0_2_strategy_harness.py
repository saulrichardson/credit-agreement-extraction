#!/usr/bin/env python
"""
Strategy harness for ContractIR v0.2 prompts and scalability operators.

This script is a forward-looking (breaking-change OK) harness that validates:
  - strict schema validation (ContractIR v0.2)
  - anchor-in-context grounding gates
  - deterministic evaluation for selected functions (including lookup_range)

Usage:
  python scripts/contract_ir_v0_2_strategy_harness.py --source-run-id dan-v2-20260106
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
from pipeline.ir.contract_ir_v0_2 import (  # noqa: E402
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
        "Your previous response did not validate against the ContractIR v0.2 schema.\n"
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


def _validate_contract_and_source_ids(doc: Any, *, item_id: str) -> List[ContractIRValidationError]:
    """Hard gate: require contract_id and sources[*].item_id to match the experiment item_id.

    This makes downstream deterministic merging possible (base_rate/spread/fee passes).
    """

    out: List[ContractIRValidationError] = []
    if not isinstance(doc, dict):
        return out

    cid = doc.get("contract_id")
    if cid != item_id:
        out.append(
            ContractIRValidationError(
                code="contract_id_mismatch",
                message=f"contract_id must equal item_id {item_id!r}; got {cid!r}",
                json_path="/contract_id",
            )
        )

    sources = doc.get("sources")
    if isinstance(sources, list):
        for si, s in enumerate(sources):
            if not isinstance(s, dict):
                continue
            sid = s.get("item_id")
            if sid != item_id:
                out.append(
                    ContractIRValidationError(
                        code="source_item_id_mismatch",
                        message=f"sources[{si}].item_id must equal item_id {item_id!r}; got {sid!r}",
                        json_path=f"/sources/{si}/item_id",
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
    expect_open_items_min: Optional[int] = None
    context_clip_chars: Optional[int] = None
    context_stop_before: Optional[str] = None
    expect_tables_empty: Optional[bool] = None
    expect_derived_empty: Optional[bool] = None
    freeze_tables_on_repair: bool = False


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

    out_run_id = args.out_run_id or f"contractir-v0_2-strategy-tests-{time.strftime('%Y%m%d-%H%M%S')}"
    out_dir = PROJECT_ROOT / "runs" / out_run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Agreements / anchors -------------------------------------------------------------
    # Term SOFR definition with an explicit floor + fallback logic (open_item required).
    term_sofr_item = "0001819974-23-000003_2"
    term_sofr_anchor = "A0446"

    # AT&T Capital agreement: Applicable Margin table by Status + Facility Fee schedule.
    att_item = "0000950117-96-000184_5"
    att_status_margin_anchor = "A0295"
    att_facility_fee_anchor = "A0300"

    # Leverage ratio bucket schedule (range tiers).
    leverage_bucket_item = "0000906318-96-000011_6"
    leverage_bucket_anchor = "A0019"

    # 2D ratio-based schedule (Interest Coverage Ratio + Fixed Charge Coverage Ratio).
    ratios_margin_item = "0000720032-96-000002_2"
    ratios_margin_anchor = "A0125"

    # ABR definition with explicit conditional (clause (c) excluded in some cases).
    abr_conditional_item = "0001140361-23-000046_9"
    abr_conditional_anchor = "A0118"

    # 1996-style Alternate Base Rate with rounding to next 1/100 of 1%.
    alt_base_rate_item = "0000912057-96-000124_5"
    alt_base_rate_anchor = "A0070"

    # 1996-style schedules: ABR margin + commitment fee by pricing level.
    pricing_levels_item = "0000912057-96-000124_5"
    pricing_levels_anchor = "A0076"

    # Applicable Margin table by Senior Long Term Debt Rating.
    rating_grid_item = "0000950134-96-000040_2"
    rating_grid_anchor = "A0049"

    # Reserve adjustment clause (LIBO / (1 - Euro Reserve Percentage)).
    reserve_adjust_item = "0000950152-96-000050_2"
    reserve_adjust_anchor = "A0139"

    # Modern commitment fee schedule by leverage ratio (range thresholds).
    modern_commitment_fee_item = "0000950170-23-000122_2"
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

    indices_for_adjusted_libo = {
        "LIBORRate": {"1996-12-31": "0.0450"},
    }

    experiments: List[Experiment] = [
        Experiment(
            exp_id="B1_TermSOFR_floor_open_item_v2",
            prompt_path="prompts/contract_ir_base_rate_v2.txt",
            item_id=term_sofr_item,
            anchor_ids=[term_sofr_anchor],
            task=(
                f"Build ContractIR v0.2 for item_id={term_sofr_item}.\n"
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
                )
            ],
            expect_open_items_min=1,
        ),
        Experiment(
            exp_id="B2_ABR_conditional_open_item_v2",
            prompt_path="prompts/contract_ir_base_rate_v2.txt",
            item_id=abr_conditional_item,
            anchor_ids=[abr_conditional_anchor],
            task=(
                f"Build ContractIR v0.2 for item_id={abr_conditional_item}.\n"
                "Goal: encode the ABR definition as a derived function.\n"
                "Requirements:\n"
                "- derived fn_id MUST be ABR\n"
                "- semantic_role MUST be base_rate\n"
                "- args: date (type=date)\n"
                "- indices MUST include PrimeRate (unit=rate), NYFRBRate (unit=rate), AdjustedTermSOFRRate1M (unit=rate)\n"
                "- expr MUST compute the PRIMARY definition:\n"
                "    max(\n"
                "      index(PrimeRate,date),\n"
                "      index(NYFRBRate,date) + 0.005,\n"
                "      index(AdjustedTermSOFRRate1M,date) + 0.01\n"
                "    )\n"
                "- The excerpt contains conditional logic excluding clause (c) when ABR is used as an alternate rate under Section 2.8.\n"
                "  Do NOT encode that conditional branch in this pass.\n"
                "  Instead, emit at least one blocking open_item describing the missing branch/date shifting logic.\n"
            ),
            evals=[
                EvalSpec(
                    fn_id="ABR",
                    args={"date": "2023-01-03"},
                    indices=indices_for_modern_abr,
                    expected_rate="0.059",
                )
            ],
            expect_open_items_min=1,
        ),
        Experiment(
            exp_id="B3_AlternateBaseRate_round_up_increment_v2",
            prompt_path="prompts/contract_ir_base_rate_v2.txt",
            item_id=alt_base_rate_item,
            anchor_ids=[alt_base_rate_anchor],
            task=(
                f"Build ContractIR v0.2 for item_id={alt_base_rate_item}.\n"
                "Goal: encode the definition of the Alternate Base Rate, including its rounding rule.\n"
                "Requirements:\n"
                "- derived fn_id MUST be AlternateBaseRate\n"
                "- semantic_role MUST be base_rate\n"
                "- args: date (type=date)\n"
                "- indices MUST include PrimeRate (unit=rate), FederalFundsRate (unit=rate)\n"
                "- expr MUST compute:\n"
                "    round_up_to_increment(\n"
                "      max(index(PrimeRate,date), index(FederalFundsRate,date) + 0.005),\n"
                "      0.0001\n"
                "    )\n"
                "  because 1/100 of 1% = 0.0001.\n"
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
        Experiment(
            exp_id="S1_SpreadGrid_lookup2_status_I_IV_v2",
            prompt_path="prompts/contract_ir_spread_v2_lookup2.txt",
            item_id=att_item,
            anchor_ids=[att_status_margin_anchor],
            task=(
                f"Build ContractIR v0.2 for item_id={att_item}.\n"
                "Goal: normalize the Applicable Margin table (Euro-Dollar Loans vs CD Loans) by Status levels I-IV.\n"
                "Requirements:\n"
                "- Create table_id: ApplicableMarginGrid\n"
                "- columns MUST include: loan_type_code (string), status_code (string), margin_bps (bps)\n"
                "- loan_type_code MUST be exactly one of: 'EURODOLLAR', 'CD'\n"
                "- status_code MUST be exactly one of: 'I','II','III','IV'\n"
                "- Convert percent values to bps (e.g., 0.2000% => 20 bps).\n"
                "- rows MUST include exactly 8 rows (2 loan types x 4 statuses)\n"
                "- derived fn_id MUST be ApplicableMargin\n"
                "- semantic_role MUST be spread\n"
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
            exp_id="S1B_TruncatedStatusGrid_should_open_item_v2",
            prompt_path="prompts/contract_ir_spread_v2_lookup2.txt",
            item_id=att_item,
            anchor_ids=[att_status_margin_anchor],
            task=(
                f"Build ContractIR v0.2 for item_id={att_item}.\n"
                "Goal: normalize the Applicable Margin table (Euro-Dollar Loans vs CD Loans) by Status levels I-IV.\n"
                "This is a stress test: the CONTEXT will be truncated before the 'CD Loans' row.\n"
                "Requirements:\n"
                "- rows MUST include exactly 8 rows (2 loan types x 4 statuses)\n"
                "- If the CD Loans row is missing from CONTEXT, DO NOT GUESS; emit a blocking open_item and return tables=[] and derived=[].\n"
            ),
            evals=[],
            expect_open_items_min=1,
            context_stop_before="CD Loans",
            expect_tables_empty=True,
            expect_derived_empty=True,
        ),
        Experiment(
            exp_id="F1_FacilityFeeRate_by_status_v2",
            prompt_path="prompts/contract_ir_fee_rate_v2_specialized.txt",
            item_id=att_item,
            anchor_ids=[att_facility_fee_anchor],
            task=(
                f"Build ContractIR v0.2 for item_id={att_item}.\n"
                "Goal: extract the Facility Fee Rate schedule by Status level.\n"
                "Requirements:\n"
                "- Create table_id: FacilityFeeRateByStatus\n"
                "- columns MUST include: status_code (string), facility_fee_bps (bps)\n"
                "- rows MUST include Status levels: I, II, III, IV.\n"
                "- Convert percent values to bps (e.g., 0.0500% => 5 bps).\n"
                "- derived fn_id MUST be FacilityFeeRate\n"
                "- semantic_role MUST be fee_rate\n"
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
        Experiment(
            exp_id="F2_CommitmentFeeRange_lookup_range_leverage_ratio_v2",
            prompt_path="prompts/contract_ir_fee_rate_v2_lookup_range.txt",
            item_id=modern_commitment_fee_item,
            anchor_ids=[modern_commitment_fee_anchor],
            task=(
                f"Build ContractIR v0.2 for item_id={modern_commitment_fee_item}.\n"
                "Goal: encode ONLY the FIRST commitment fee schedule table in the excerpt.\n"
                "It is introduced by: '- at any time on or after the Second Amendment Effective Date...'\n"
                "Requirements:\n"
                "- Create table_id: CommitmentFeeByLeverage\n"
                "- columns MUST include: bucket_label (string), lower_bound (decimal), lower_cmp (string), upper_bound (decimal), upper_cmp (string), fee_bps (bps)\n"
                "- Convert percent values to bps (e.g., 0.50% => 50 bps).\n"
                "- derived fn_id MUST be CommitmentFeeRate\n"
                "- semantic_role MUST be fee_rate\n"
                "- args: leverage_ratio (type=decimal)\n"
                "- expr MUST compute: bps_to_rate(lookup_range(\"CommitmentFeeByLeverage\", leverage_ratio, \"lower_bound\", \"lower_cmp\", \"upper_bound\", \"upper_cmp\", \"fee_bps\"))\n"
            ),
            evals=[
                EvalSpec(
                    fn_id="CommitmentFeeRate",
                    args={"leverage_ratio": "5.00"},
                    indices={},
                    expected_rate="0.005",
                ),
                EvalSpec(
                    fn_id="CommitmentFeeRate",
                    args={"leverage_ratio": "2.00"},
                    indices={},
                    expected_rate="0.00375",
                ),
            ],
            context_stop_before="- at any time on or after the Revolving Facility Amendment Date",
        ),
        Experiment(
            exp_id="S2_SpreadRange_lookup_range_leverage_ratio_v2",
            prompt_path="prompts/contract_ir_spread_v2_lookup_range.txt",
            item_id=leverage_bucket_item,
            anchor_ids=[leverage_bucket_anchor],
            task=(
                f"Build ContractIR v0.2 for item_id={leverage_bucket_item}.\n"
                "Goal: encode the Eurodollar borrowing margin schedule as a range-based table.\n"
                "Requirements:\n"
                "- Create table_id: EurodollarBorrowingMarginByLeverage\n"
                "- columns MUST include: bucket_label (string), lower_bound (decimal), lower_cmp (string), upper_bound (decimal), upper_cmp (string), margin_bps (bps)\n"
                "- Use lower_cmp in {'gt','gte'} and upper_cmp in {'lt','lte'}.\n"
                "- Omit lower_bound/lower_cmp for an unbounded lower range; omit upper_bound/upper_cmp for an unbounded upper range.\n"
                "- Convert percent values to bps (e.g., 1.125% => 112.5 bps).\n"
                "- derived fn_id MUST be EurodollarBorrowingMargin\n"
                "- semantic_role MUST be spread\n"
                "- args: leverage_ratio (type=decimal)\n"
                "- expr MUST compute: bps_to_rate(lookup_range(\"EurodollarBorrowingMarginByLeverage\", leverage_ratio, \"lower_bound\", \"lower_cmp\", \"upper_bound\", \"upper_cmp\", \"margin_bps\"))\n"
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
        Experiment(
            exp_id="S5_RatioSchedule_lookup_rule_2D_v2",
            prompt_path="prompts/contract_ir_spread_v2_lookup_rule.txt",
            item_id=ratios_margin_item,
            anchor_ids=[ratios_margin_anchor],
            task=(
                f"Build ContractIR v0.2 for item_id={ratios_margin_item}.\n"
                "Goal: encode the Applicable Margin schedule for LIBO Rate Advances using a RULE TABLE.\n"
                "Requirements:\n"
                "- Create table_id: ApplicableMarginByCoverageRatios\n"
                "- columns MUST include: rule_label (string), predicate (bool), margin_bps (bps)\n"
                "- For each rule in the excerpt, create one row with:\n"
                "  - rule_label: short label from the excerpt condition\n"
                "  - predicate: boolean condition over the function args\n"
                "  - margin_bps: margin percent converted to bps (e.g., 2.50% => 250)\n"
                "- derived fn_id MUST be ApplicableMarginLIBO\n"
                "- semantic_role MUST be spread\n"
                "- args: interest_coverage_ratio (type=decimal), fixed_charge_coverage_ratio (type=decimal)\n"
                "- expr MUST compute: bps_to_rate(lookup_rule(\"ApplicableMarginByCoverageRatios\",\"predicate\",\"margin_bps\"))\n"
            ),
            evals=[
                EvalSpec(
                    fn_id="ApplicableMarginLIBO",
                    args={"interest_coverage_ratio": "1.50", "fixed_charge_coverage_ratio": "2.00"},
                    indices={},
                    expected_rate="0.025",
                ),
                EvalSpec(
                    fn_id="ApplicableMarginLIBO",
                    args={"interest_coverage_ratio": "2.00", "fixed_charge_coverage_ratio": "1.30"},
                    indices={},
                    expected_rate="0.0225",
                ),
                EvalSpec(
                    fn_id="ApplicableMarginLIBO",
                    args={"interest_coverage_ratio": "3.00", "fixed_charge_coverage_ratio": "1.50"},
                    indices={},
                    expected_rate="0.02",
                ),
            ],
            freeze_tables_on_repair=True,
        ),
        Experiment(
            exp_id="S5B_TruncatedRatioSchedule_should_open_item_v2",
            prompt_path="prompts/contract_ir_spread_v2_lookup_rule.txt",
            item_id=ratios_margin_item,
            anchor_ids=[ratios_margin_anchor],
            task=(
                f"Build ContractIR v0.2 for item_id={ratios_margin_item}.\n"
                "Goal: encode the Applicable Margin schedule for LIBO Rate Advances using a RULE TABLE.\n"
                "This is a stress test: the CONTEXT will be truncated before the final margin value.\n"
                "Requirements:\n"
                "- If the excerpt does not contain all margin values for all rules, DO NOT GUESS.\n"
                "  Emit a blocking open_item and return tables=[] and derived=[].\n"
            ),
            evals=[],
            expect_open_items_min=1,
            context_stop_before="2.00%",
            expect_tables_empty=True,
            expect_derived_empty=True,
            freeze_tables_on_repair=True,
        ),
        Experiment(
            exp_id="S3_SpreadGrid_lookup2_rating_bucket_v2",
            prompt_path="prompts/contract_ir_spread_v2_lookup2.txt",
            item_id=rating_grid_item,
            anchor_ids=[rating_grid_anchor],
            task=(
                f"Build ContractIR v0.2 for item_id={rating_grid_item}.\n"
                "Goal: normalize the Applicable Margin table by Senior Long Term Debt Rating into a long table.\n"
                "Requirements:\n"
                "- Create table_id: ApplicableMarginByRating\n"
                "- columns MUST include: loan_type_code (string), rating_bucket (string), margin_bps (bps)\n"
                "- loan_type_code MUST be exactly one of: 'EURODOLLAR', 'BASE_RATE'\n"
                "- rows MUST include exactly 10 rows (2 loan types x 5 rating buckets)\n"
                "- Convert percent values to bps.\n"
                "- derived fn_id MUST be ApplicableMargin\n"
                "- semantic_role MUST be spread\n"
                "- args: loan_type_code (type=string), rating_bucket (type=string)\n"
                "- expr MUST compute: bps_to_rate(lookup2(\"ApplicableMarginByRating\",\"loan_type_code\",loan_type_code,\"rating_bucket\",rating_bucket,\"margin_bps\"))\n"
            ),
            evals=[
                EvalSpec(
                    fn_id="ApplicableMargin",
                    args={"loan_type_code": "EURODOLLAR", "rating_bucket": "BBB/Baa2"},
                    indices={},
                    expected_rate="0.0075",
                )
            ],
        ),
        Experiment(
            exp_id="S3B_TruncatedRatingGrid_should_open_item_v2",
            prompt_path="prompts/contract_ir_spread_v2_lookup2.txt",
            item_id=rating_grid_item,
            anchor_ids=[rating_grid_anchor],
            task=(
                f"Build ContractIR v0.2 for item_id={rating_grid_item}.\n"
                "Goal: normalize the Applicable Margin table by Senior Long Term Debt Rating into a long table.\n"
                "This is a stress test: the CONTEXT will be truncated before the final rating bucket row.\n"
                "Requirements:\n"
                "- rows MUST include exactly 10 rows (2 loan types x 5 rating buckets)\n"
                "- If the final rating bucket row is missing from CONTEXT, DO NOT GUESS; emit a blocking open_item and return tables=[] and derived=[].\n"
            ),
            evals=[],
            expect_open_items_min=1,
            context_stop_before="less than BBB-/Baa3",
            expect_tables_empty=True,
            expect_derived_empty=True,
        ),
        Experiment(
            exp_id="S4_ABR_margin_by_pricing_level_v2",
            prompt_path="prompts/contract_ir_spread_v2_specialized.txt",
            item_id=pricing_levels_item,
            anchor_ids=[pricing_levels_anchor],
            task=(
                f"Build ContractIR v0.2 for item_id={pricing_levels_item}.\n"
                "Goal: extract the Applicable Alternate Base Rate Margin schedule by Applicable Pricing Level.\n"
                "Requirements:\n"
                "- Create table_id: ABRMarginByPricingLevel\n"
                "- columns MUST include: pricing_level_code (string), margin_bps (bps)\n"
                "- pricing_level_code MUST be exactly one of: 'I','II','III','IV'\n"
                "- Convert the bps values (already expressed in basis points) into bps literal values.\n"
                "- derived fn_id MUST be ABRMargin\n"
                "- semantic_role MUST be spread\n"
                "- args: pricing_level_code (type=string)\n"
                "- expr MUST compute: bps_to_rate(lookup(\"ABRMarginByPricingLevel\",\"pricing_level_code\",pricing_level_code,\"margin_bps\"))\n"
            ),
            evals=[
                EvalSpec(
                    fn_id="ABRMargin",
                    args={"pricing_level_code": "II"},
                    indices={},
                    expected_rate="0.005",
                )
            ],
        ),
        Experiment(
            exp_id="F3_CommitmentFee_by_pricing_level_v2",
            prompt_path="prompts/contract_ir_fee_rate_v2_specialized.txt",
            item_id=pricing_levels_item,
            anchor_ids=[pricing_levels_anchor],
            task=(
                f"Build ContractIR v0.2 for item_id={pricing_levels_item}.\n"
                "Goal: extract the Applicable Commitment Fee Rate schedule by Applicable Pricing Level.\n"
                "Requirements:\n"
                "- Create table_id: CommitmentFeeByPricingLevel\n"
                "- columns MUST include: pricing_level_code (string), fee_bps (bps)\n"
                "- pricing_level_code MUST be exactly one of: 'I','II','III','IV'\n"
                "- The excerpt expresses values in basis points; store them as bps literal values.\n"
                "- derived fn_id MUST be CommitmentFeeRate\n"
                "- semantic_role MUST be fee_rate\n"
                "- args: pricing_level_code (type=string)\n"
                "- expr MUST compute: bps_to_rate(lookup(\"CommitmentFeeByPricingLevel\",\"pricing_level_code\",pricing_level_code,\"fee_bps\"))\n"
            ),
            evals=[
                EvalSpec(
                    fn_id="CommitmentFeeRate",
                    args={"pricing_level_code": "IV"},
                    indices={},
                    expected_rate="0.00125",
                )
            ],
        ),
        Experiment(
            exp_id="B4_AdjustedLIBOR_reserve_pct_v2",
            prompt_path="prompts/contract_ir_base_rate_v2.txt",
            item_id=reserve_adjust_item,
            anchor_ids=[reserve_adjust_anchor],
            task=(
                f"Build ContractIR v0.2 for item_id={reserve_adjust_item}.\n"
                "Goal: encode the reserve-adjusted LIBO rate (LIBO / (1 - Euro Reserve Percentage)).\n"
                "Requirements:\n"
                "- derived fn_id MUST be AdjustedLIBOR\n"
                "- semantic_role MUST be base_rate\n"
                "- args: date (type=date), euro_reserve_pct (type=decimal)\n"
                "- indices MUST include LIBORRate (unit=rate)\n"
                "- expr MUST compute: div(index(LIBORRate,date), sub(1.00, euro_reserve_pct))\n"
                "- The excerpt states the reserve adjustment applies only when the Bank maintains reserves / determines costs increased.\n"
                "  Emit at least one blocking open_item describing that conditional applicability (do not guess when it applies).\n"
            ),
            evals=[
                EvalSpec(
                    fn_id="AdjustedLIBOR",
                    args={"date": "1996-12-31", "euro_reserve_pct": "0.10"},
                    indices=indices_for_adjusted_libo,
                    expected_rate="0.05",
                )
            ],
            expect_open_items_min=1,
        ),
    ]

    if args.exp_filter:
        experiments = [e for e in experiments if args.exp_filter in e.exp_id]
        if not experiments:
            raise ValueError(f"No experiments matched exp-filter={args.exp_filter!r}")

    complete_response_sync = _ensure_gateway_client_sync()

    run_summary: Dict[str, Any] = {
        "schema_version": "contractir_v0_2_strategy_tests_v0",
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
        baseline_table_shapes: Optional[Dict[str, int]] = None

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
                if not validation_errors and not anchor_errors:
                    validation_errors = _validate_contract_and_source_ids(parsed, item_id=exp.item_id)

            if parsed is not None and not validation_errors and not anchor_errors and exp.freeze_tables_on_repair:
                shapes: Dict[str, int] = {}
                tables = parsed.get("tables")
                if isinstance(tables, list):
                    for t in tables:
                        if not isinstance(t, dict):
                            continue
                        tid = t.get("table_id")
                        if not isinstance(tid, str) or not tid:
                            continue
                        rows = t.get("rows")
                        shapes[tid] = (len(rows) if isinstance(rows, list) else -1)
                if baseline_table_shapes is None:
                    baseline_table_shapes = shapes
                elif shapes != baseline_table_shapes:
                    validation_errors = [
                        ContractIRValidationError(
                            code="table_shape_changed",
                            message=(
                                "Table shapes changed across repair attempts. "
                                f"baseline={baseline_table_shapes} current={shapes}. "
                                "Do not add/remove rule rows; only correct predicates/values."
                            ),
                            json_path="/tables",
                        )
                    ]

            open_items_count: Optional[int] = None
            if parsed is not None and not validation_errors and not anchor_errors:
                open_items = parsed.get("open_items")
                if isinstance(open_items, list):
                    open_items_count = len(open_items)

                if exp.expect_open_items_min is not None:
                    if open_items_count is None or open_items_count < exp.expect_open_items_min:
                        validation_errors = [
                            ContractIRValidationError(
                                code="open_items_min",
                                message=(
                                    "Expected at least "
                                    f"{exp.expect_open_items_min} open_items for this task, "
                                    f"got {open_items_count}."
                                ),
                                json_path="/open_items",
                            )
                        ]

            if parsed is not None and not validation_errors and not anchor_errors:
                if exp.expect_tables_empty:
                    tables = parsed.get("tables")
                    if not isinstance(tables, list) or tables:
                        validation_errors = [
                            ContractIRValidationError(
                                code="tables_expected_empty",
                                message="Expected tables=[] for this task.",
                                json_path="/tables",
                            )
                        ]
                if exp.expect_derived_empty and not validation_errors:
                    derived = parsed.get("derived")
                    if not isinstance(derived, list) or derived:
                        validation_errors = [
                            ContractIRValidationError(
                                code="derived_expected_empty",
                                message="Expected derived=[] for this task.",
                                json_path="/derived",
                            )
                        ]

            # Semantic gate: if this experiment includes eval specs, require that evaluation succeeds
            # and matches the expected numeric results (deterministic, no model code execution).
            if parsed is not None and not validation_errors and not anchor_errors and exp.evals:
                eval_gate_errors: List[ContractIRValidationError] = []
                for spec in exp.evals:
                    try:
                        out = evaluate_function(parsed, fn_id=spec.fn_id, args=spec.args, indices=spec.indices)
                        if out.kind != "rate":
                            raise TypeError(f"Expected rate output, got {out.kind}")
                        got = _decimal_str(out.value)
                        expected_norm = _decimal_str(Decimal(spec.expected_rate))
                        if got != expected_norm:
                            eval_gate_errors.append(
                                ContractIRValidationError(
                                    code="eval_numeric_mismatch",
                                    message=(
                                        f"Evaluation mismatch for fn_id={spec.fn_id}: got_rate={got} "
                                        f"expected_rate={expected_norm} args={dict(spec.args)}"
                                    ),
                                    json_path="/derived",
                                )
                            )
                    except Exception as e:
                        eval_gate_errors.append(
                            ContractIRValidationError(
                                code="eval_error",
                                message=f"Evaluation error for fn_id={spec.fn_id} args={dict(spec.args)}: {e}",
                                json_path="/derived",
                            )
                        )

                if eval_gate_errors:
                    validation_errors = eval_gate_errors

            if parsed is not None and not validation_errors and not anchor_errors:
                repair_rounds_used = (attempt - 1)
                break

            if attempt > 1 + max(0, args.max_repair_rounds):
                repair_rounds_used = None
                break

            prompt = _render_repair_prompt(raw_json=raw or "", errors=(validation_errors or anchor_errors))
            (exp_dir / f"repair_prompt_round_{attempt-1}.txt").write_text(prompt, encoding="utf-8")

        # open_items_count computed within the attempt loop (after schema + anchor checks).

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

        if parsed is not None and not validation_errors and not anchor_errors:
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

    eval_total = 0
    eval_ok = 0
    for exp in run_summary["experiments"]:
        for ev in exp.get("evals") or []:
            eval_total += 1
            if ev.get("numeric_correct"):
                eval_ok += 1

    print(f"Wrote ContractIR v0.2 strategy test artifacts to {out_dir}")
    print(f"Summary: {out_dir / 'summary.json'}")
    print(f"[eval] numeric_correct: {eval_ok}/{eval_total}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
