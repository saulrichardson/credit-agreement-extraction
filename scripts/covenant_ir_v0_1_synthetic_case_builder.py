#!/usr/bin/env python
"""
Build a synthetic CovenantIR v0.1 "run" under runs/<run_id>/ for stress-testing.

This produces minimal artifacts needed by:
  - scripts/covenant_ir_v0_1_one_pass_harness.py
  - scripts/covenant_ir_v0_1_batch_runner.py

Specifically, for each synthetic item_id we write:
  - runs/<run_id>/normalized/<item_id>/anchors.tsv
  - runs/<run_id>/retrieval_v2/<item_id>_snippets.jsonl

All snippets are categorized as "financial_covenant" so they are included in the excerpt
pack by default. The prompt itself is responsible for excluding non-financial covenants.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence


@dataclass(frozen=True)
class SyntheticItem:
    item_id: str
    blocks: Mapping[str, str]  # anchor_id -> snippet text
    notes: str | None = None


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _anchor_rows(anchor_ids: Sequence[str]) -> str:
    # anchors.tsv is used only for stable ordering + allowed anchor_id set.
    # Start/end offsets are not used by CovenantIR harness, but must be integers.
    lines = ["anchor_id\tanchor_type\tstart\tend\tlabel\n"]
    offset = 0
    for aid in anchor_ids:
        start = offset
        end = offset + 1
        offset += 10
        lines.append(f"{aid}\tsentence\t{start}\t{end}\t{aid}\n")
    return "".join(lines)


def _snippet_records(item: SyntheticItem, *, anchor_order: Sequence[str]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for aid in anchor_order:
        snippet = item.blocks.get(aid, "")
        out.append(
            {
                "item_id": item.item_id,
                "anchor_id": aid,
                "categories": ["financial_covenant"],
                "label": "financial_covenant",
                "type": "sentence",
                "start": 0,
                "end": 0,
                "buckets": ["financial_covenant"],
                "toc_chunk_id": None,
                "toc_title": None,
                "snippet": snippet,
                "snippet_start": 0,
                "snippet_end": max(0, len(snippet)),
            }
        )
    return out


def _define_items() -> List[SyntheticItem]:
    # These are deliberately "agreement-like" snippets that cover:
    #  - max/min ratio tests
    #  - springing applicability
    #  - multiple covenants + out-of-scope negative covenant bait
    #  - date-range schedules (lookup_rule / lookup_range)
    #  - fiscal-year label schedules (string keys)
    #  - "positive amount" semantics
    #  - incurrence tests (applies_when)
    #  - duplicate/truncated excerpts (use most complete)
    #  - true missing schedule (should block)
    #  - ambiguous comparator (should block)

    return [
        SyntheticItem(
            item_id="synth_fin_01_simple_max_leverage",
            blocks={
                "A0001": (
                    "6.14 MAXIMUM TOTAL LEVERAGE RATIO.\n"
                    "The Borrower shall not permit the Consolidated Total Leverage Ratio, as of the last day of any Fiscal Quarter,\n"
                    "to exceed 3.50 to 1.00.\n"
                )
            },
            notes="Single max ratio covenant; no schedule; should encode as ratio <= 3.50.",
        ),
        SyntheticItem(
            item_id="synth_fin_02_springing_fccr",
            blocks={
                "A0001": (
                    "6.15 MINIMUM FIXED CHARGE COVERAGE RATIO.\n"
                    "If at any time Revolving Loans are outstanding, the Borrower shall not permit the Consolidated Fixed Charge Coverage Ratio\n"
                    "to be less than 1.10 to 1.00, measured as of the last day of each Fiscal Quarter.\n"
                )
            },
            notes="Springing covenant; should use applies_when=revolving_loans_outstanding and test=fccr >= 1.10.",
        ),
        SyntheticItem(
            item_id="synth_fin_03_multi_plus_negative_bait",
            blocks={
                "A0001": (
                    "7.1 LIENS.\n"
                    "The Borrower shall not create or permit to exist any Lien on any property, other than Permitted Liens.\n"
                    "For example, purchase money liens securing indebtedness not exceeding $5,000,000 are permitted.\n"
                ),
                "A0002": (
                    "6.16 MINIMUM LIQUIDITY.\n"
                    "The Borrower shall maintain Liquidity of not less than $10,000,000 at all times.\n\n"
                    "6.17 MAXIMUM CAPITAL EXPENDITURES.\n"
                    "The Borrower shall not permit Consolidated Capital Expenditures for any Fiscal Year to exceed $7,000,000;\n"
                    "provided that unused amounts in any Fiscal Year may be carried forward and added to the limitation for the immediately succeeding Fiscal Year.\n\n"
                    "6.18 MINIMUM CONSOLIDATED NET WORTH.\n"
                    "The Borrower shall not permit Consolidated Net Worth at any time to be less than $50,000,000.\n"
                ),
            },
            notes="Multiple covenants; includes an out-of-scope negative covenant with numbers to ensure exclusion.",
        ),
        SyntheticItem(
            item_id="synth_fin_04_date_range_schedule",
            blocks={
                "A0001": (
                    "6.19 MAXIMUM FIRST LIEN LEVERAGE RATIO.\n"
                    "The Borrower shall not permit the Consolidated First Lien Leverage Ratio, as of the last day of each Fiscal Quarter,\n"
                    "to exceed the applicable ratio set forth below:\n\n"
                    "[[TABLE]]\n"
                    "Period                                           Maximum Ratio\n"
                    "----------------------------------------------------------------\n"
                    "January 1, 2026 through and including June 30, 2026      4.50 to 1.00\n"
                    "July 1, 2026 through and including December 31, 2026     4.25 to 1.00\n"
                    "January 1, 2027 and thereafter                            4.00 to 1.00\n"
                    "[[/TABLE]]\n"
                )
            },
            notes="Schedule keyed by date ranges; should encode via lookup_range or lookup_rule without overlaps/gaps.",
        ),
        SyntheticItem(
            item_id="synth_fin_05_fiscal_year_label_schedule",
            blocks={
                "A0001": (
                    "6.20 MINIMUM EBITDA.\n"
                    "The Borrower shall not permit Consolidated EBITDA for any Fiscal Year to be less than the amount set forth below for such Fiscal Year:\n\n"
                    "[[TABLE]]\n"
                    "Fiscal Year     Minimum EBITDA\n"
                    "------------------------------\n"
                    "2026            $50,000,000\n"
                    "2027            $55,000,000\n"
                    "2028 and thereafter  $60,000,000\n"
                    "[[/TABLE]]\n"
                )
            },
            notes="Non-date schedule; should not guess fiscal-year boundaries; encode with string key / lookup_rule.",
        ),
        SyntheticItem(
            item_id="synth_fin_06_positive_net_income",
            blocks={
                "A0001": (
                    "6.21 NET INCOME.\n"
                    "Consolidated Net Income for the Fiscal Year 2026 shall be a positive amount.\n"
                )
            },
            notes="No explicit numeric threshold; prompt instructs 'positive amount' => strictly > 0 (do not block).",
        ),
        SyntheticItem(
            item_id="synth_fin_07_acquisition_stepup",
            blocks={
                "A0001": (
                    "6.22 MAXIMUM LEVERAGE RATIO.\n"
                    "The Borrower shall not permit the Consolidated Total Leverage Ratio, as of the last day of each Fiscal Quarter, to exceed 4.00 to 1.00;\n"
                    "provided that following a Material Acquisition, the Borrower may elect to increase the maximum Consolidated Total Leverage Ratio to 4.50 to 1.00\n"
                    "for the four Fiscal Quarters ending after the closing of such Material Acquisition.\n"
                )
            },
            notes="Piecewise threshold; should use if(has_material_acquisition_recent_4q, 4.50, 4.00) or a single upstream boolean compliance var.",
        ),
        SyntheticItem(
            item_id="synth_fin_08_incurrence_test",
            blocks={
                "A0001": (
                    "6.23 INCURRENCE LEVERAGE TEST.\n"
                    "The Borrower may Incur additional Indebtedness only if, immediately after giving Pro Forma effect thereto,\n"
                    "the Consolidated Total Leverage Ratio does not exceed 3.00 to 1.00.\n"
                )
            },
            notes="Incurrence-style financial test; should model as applies_when (incurring debt) + ratio threshold.",
        ),
        SyntheticItem(
            item_id="synth_fin_09_duplicate_truncated_vs_complete",
            blocks={
                "A0001": (
                    "6.24 MINIMUM INTEREST COVERAGE RATIO.\n"
                    "The Borrower shall not permit the Consolidated Interest Coverage Ratio as of the last day of any Fiscal Quarter to be less than:\n\n"
                    "[[TABLE]]\n"
                    "Period                    Minimum Ratio\n"
                    "---------------------------------------\n"
                    "Through June 30, 2026      2.00 to 1.00\n"
                    "July 1, 2026 and thereafter\n"
                    "[[/TABLE]]\n"
                ),
                "A0002": (
                    "6.24 MINIMUM INTEREST COVERAGE RATIO.\n"
                    "The Borrower shall not permit the Consolidated Interest Coverage Ratio as of the last day of any Fiscal Quarter to be less than:\n\n"
                    "[[TABLE]]\n"
                    "Period                    Minimum Ratio\n"
                    "---------------------------------------\n"
                    "Through June 30, 2026      2.00 to 1.00\n"
                    "July 1, 2026 and thereafter 2.50 to 1.00\n"
                    "[[/TABLE]]\n"
                ),
            },
            notes="Overlapping anchors: one truncated, one complete. Should not block; use complete copy.",
        ),
        SyntheticItem(
            item_id="synth_fin_10_missing_schedule_should_block",
            blocks={
                "A0001": (
                    "6.25 MAXIMUM SENIOR SECURED LEVERAGE RATIO.\n"
                    "The Borrower shall not permit the Consolidated Senior Secured Leverage Ratio to exceed the applicable ratio set forth below:\n\n"
                    "[[TABLE]]\n"
                    "Period                    Maximum Ratio\n"
                    "---------------------------------------\n"
                    "Date hereof until\n"
                    "[[/TABLE]]\n"
                )
            },
            notes="Truly missing threshold values; should emit blocking open_items and return no covenants/tables/derived.",
        ),
        SyntheticItem(
            item_id="synth_fin_11_ambiguous_comparator_should_block",
            blocks={
                "A0001": (
                    "6.26 CURRENT RATIO.\n"
                    "The Borrower shall maintain a Current Ratio of 1.20 to 1.00 as of the last day of each Fiscal Quarter.\n"
                )
            },
            notes="Comparator ambiguous ('maintain a ratio of 1.20'); should block rather than guessing >= or <=.",
        ),
    ]


def build_synthetic_run(*, base_dir: Path, run_id: str, overwrite: bool) -> Path:
    run_dir = base_dir / "runs" / run_id
    retrieval_dir = run_dir / "retrieval_v2"
    normalized_dir = run_dir / "normalized"

    items = _define_items()
    if run_dir.exists() and not overwrite:
        raise SystemExit(f"Run dir already exists: {run_dir} (pass --overwrite to rebuild)")

    retrieval_dir.mkdir(parents=True, exist_ok=True)
    normalized_dir.mkdir(parents=True, exist_ok=True)

    # Top-level manifest-like note for humans.
    _write_json(
        run_dir / "synthetic_cases.json",
        {
            "schema_version": "synthetic_covenant_ir_cases_v1",
            "created_at": int(time.time()),
            "run_id": run_id,
            "items": [asdict(i) for i in items],
        },
    )

    for item in items:
        anchor_ids = sorted(item.blocks.keys())
        anchors_tsv = _anchor_rows(anchor_ids)
        _write_text(normalized_dir / item.item_id / "anchors.tsv", anchors_tsv)

        snippet_path = retrieval_dir / f"{item.item_id}_snippets.jsonl"
        recs = _snippet_records(item, anchor_order=anchor_ids)
        snippet_path.parent.mkdir(parents=True, exist_ok=True)
        snippet_path.write_text(
            "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in recs),
            encoding="utf-8",
        )

    return run_dir


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", default="synth-covenantir-finA1-20260116")
    ap.add_argument("--base-dir", default=".")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    run_dir = build_synthetic_run(base_dir=Path(args.base_dir), run_id=args.run_id, overwrite=bool(args.overwrite))
    print(f"Wrote synthetic run to {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

