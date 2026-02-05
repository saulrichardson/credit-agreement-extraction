#!/usr/bin/env python3
"""
Bundle CovenantIR-vs-CEA artifacts into a review-friendly folder layout.

Goal
----
Make it easy for a new person to open one folder per item and see:
  1) the exact paragraph text we extracted from (source_text.txt)
  2) the CEA baseline row for that paragraph (baseline.json)
  3) our CovenantIR output + status (covenantir_validated.json, covenantir_result.json)
  4) the comparator outcome + best-match covenant summary (comparison.json)

This is intentionally a *copy-based* bundle (no symlinks) so the folder can be shared/moved.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8", errors="replace"))


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _copy_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def _safe_str(x: Any) -> str:
    return "" if x is None else str(x)


@dataclass(frozen=True)
class IndexRow:
    run_id: str
    item_id: str
    disposition: str
    covenantir_status: str
    baseline_name: str
    baseline_limit_raw: str
    baseline_start_date_iso: str
    baseline_end_date_iso: str
    best_match_title: str
    best_match_covenant_id: str
    best_match_limit_match: str
    best_match_threshold_values: str
    item_dir: str


def _flatten_thresholds(values: Any, max_values: int = 12) -> str:
    if not isinstance(values, list):
        return ""
    vals = [str(v) for v in values if v is not None]
    if len(vals) > max_values:
        vals = vals[:max_values] + ["…"]
    return ";".join(vals)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True, help="CEA run_id under runs/<run_id>/ (e.g., cea-1995q3).")
    ap.add_argument("--report", required=True, help="Comparator report JSON (covenantir_v0_1_vs_cea_*.v3.json).")
    ap.add_argument(
        "--covenantir-dir",
        required=True,
        help="Directory containing per-item CovenantIR outputs (subdirs by item_id).",
    )
    ap.add_argument("--base-dir", default=".", help="Repo base dir")
    ap.add_argument("--out-dir", required=True, help="Output bundle directory (will be created; must not already exist).")
    ap.add_argument("--include-prompt-context", action="store_true", help="Copy contexts.txt + prompt.txt when present.")
    ap.add_argument("--max-items", type=int, default=10_000)
    args = ap.parse_args()

    base_dir = Path(args.base_dir)
    run_id = str(args.run_id)
    run_dir = base_dir / "runs" / run_id
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.exists():
        raise SystemExit(f"Missing manifest.json: {manifest_path}")

    covenantir_dir = Path(args.covenantir_dir)
    if not covenantir_dir.exists():
        raise SystemExit(f"Missing covenantir-dir: {covenantir_dir}")

    report_path = Path(args.report)
    if not report_path.exists():
        raise SystemExit(f"Missing report JSON: {report_path}")

    out_dir = Path(args.out_dir)
    if out_dir.exists():
        raise SystemExit(f"out-dir already exists: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=False)

    manifest = _read_json(manifest_path)
    items = manifest.get("items")
    if not isinstance(items, list) or not items:
        raise SystemExit(f"manifest.json has no items: {manifest_path}")

    report = _read_json(report_path)
    report_items = report.get("items")
    if not isinstance(report_items, list):
        raise SystemExit(f"report JSON has no items list: {report_path}")
    report_by_item_id: Dict[str, Dict[str, Any]] = {}
    for it in report_items:
        if not isinstance(it, dict):
            continue
        iid = it.get("item_id")
        if isinstance(iid, str) and iid:
            report_by_item_id[iid] = it

    # Bundle files.
    (out_dir / "items").mkdir(parents=True, exist_ok=True)
    _copy_file(manifest_path, out_dir / "manifest.json")
    _copy_file(report_path, out_dir / "report.json")

    readme = (
        f"CEA CovenantIR review bundle\n"
        f"===========================\n\n"
        f"run_id: {run_id}\n\n"
        f"Folder layout:\n"
        f"- items/<ITEM_ID>/\n"
        f"  - source_text.txt              (canonical paragraph text)\n"
        f"  - baseline.json                (CEA baseline row + provenance)\n"
        f"  - covenantir_result.json       (status/attempt counts)\n"
        f"  - covenantir_validated.json    (CovenantIR v0.1 output when status=ok)\n"
        f"  - comparison.json              (baseline vs best_match + disposition)\n"
    )
    if args.include_prompt_context:
        readme += (
            f"  - contexts.txt                 (excerpt pack fed to the model)\n"
            f"  - prompt.txt                   (rendered prompt fed to the model)\n"
        )
    readme += (
        f"\nIndexes:\n"
        f"- index.csv   (all items)\n"
        f"- issues.csv  (items where disposition != ok_baseline_matched)\n"
    )
    _write_text(out_dir / "README.txt", readme)

    index_rows: List[IndexRow] = []
    for raw in items[: max(0, int(args.max_items))]:
        if not isinstance(raw, dict):
            continue
        item_id = str(raw.get("item_id") or "")
        if not item_id:
            continue

        item_dir = out_dir / "items" / item_id
        item_dir.mkdir(parents=True, exist_ok=True)

        # Source text.
        canonical_path = run_dir / "normalized" / item_id / "canonical.txt"
        if canonical_path.exists():
            _copy_file(canonical_path, item_dir / "source_text.txt")
        else:
            _write_text(item_dir / "source_text.txt", "<missing canonical.txt>\n")

        # Baseline + provenance.
        baseline = raw.get("baseline") if isinstance(raw.get("baseline"), dict) else {}
        source = raw.get("source") if isinstance(raw.get("source"), dict) else {}
        _write_json(item_dir / "baseline.json", {"item_id": item_id, "baseline": baseline, "source": source})

        # CovenantIR outputs.
        cov_item_dir = covenantir_dir / item_id
        cov_result = cov_item_dir / "result.json"
        cov_validated = cov_item_dir / "covenantir_validated.json"
        if cov_result.exists():
            _copy_file(cov_result, item_dir / "covenantir_result.json")
        else:
            _write_text(item_dir / "covenantir_result.json", "{\n  \"status\": \"missing\"\n}\n")
        if cov_validated.exists():
            _copy_file(cov_validated, item_dir / "covenantir_validated.json")

        if args.include_prompt_context:
            for name in ("contexts.txt", "prompt.txt"):
                src = cov_item_dir / name
                if src.exists():
                    _copy_file(src, item_dir / name)

        # Comparator view.
        rep_it = report_by_item_id.get(item_id)
        if rep_it is None:
            _write_json(item_dir / "comparison.json", {"item_id": item_id, "error": "missing_in_report"})
            disposition = "missing_in_report"
            covenantir_status = "unknown"
            best = {}
        else:
            # Keep comparison.json small-ish but complete for review.
            keep = {
                "item_id": item_id,
                "disposition": rep_it.get("disposition"),
                "baseline": rep_it.get("baseline"),
                "covenantir": rep_it.get("covenantir"),
                "match": rep_it.get("match"),
                "best_match": rep_it.get("best_match"),
                "candidates": rep_it.get("candidates"),
            }
            _write_json(item_dir / "comparison.json", keep)
            disposition = _safe_str(rep_it.get("disposition"))
            covenantir_status = _safe_str(((rep_it.get("covenantir") or {}) if isinstance(rep_it.get("covenantir"), dict) else {}).get("status"))
            best = rep_it.get("best_match") if isinstance(rep_it.get("best_match"), dict) else {}

        base_name = _safe_str(baseline.get("name"))
        base_limit = _safe_str(baseline.get("limit"))
        base_start = _safe_str(rep_it.get("baseline", {}).get("start_date_iso") if rep_it else "")
        base_end = _safe_str(rep_it.get("baseline", {}).get("end_date_iso") if rep_it else "")

        index_rows.append(
            IndexRow(
                run_id=run_id,
                item_id=item_id,
                disposition=disposition,
                covenantir_status=covenantir_status,
                baseline_name=base_name,
                baseline_limit_raw=base_limit,
                baseline_start_date_iso=base_start,
                baseline_end_date_iso=base_end,
                best_match_title=_safe_str(best.get("title")),
                best_match_covenant_id=_safe_str(best.get("covenant_id")),
                best_match_limit_match=_safe_str(best.get("limit_match")),
                best_match_threshold_values=_flatten_thresholds(best.get("threshold_values")),
                item_dir=str(item_dir.relative_to(out_dir)),
            )
        )

    # Write index CSVs.
    def _write_index_csv(path: Path, rows: List[IndexRow]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(IndexRow.__annotations__.keys()))
            w.writeheader()
            for r in rows:
                w.writerow(r.__dict__)

    _write_index_csv(out_dir / "index.csv", index_rows)
    issues = [r for r in index_rows if r.disposition != "ok_baseline_matched"]
    _write_index_csv(out_dir / "issues.csv", issues)

    print(f"[done] wrote bundle: {out_dir}")
    print(f"[index] {out_dir / 'index.csv'}")
    print(f"[issues] {out_dir / 'issues.csv'} ({len(issues)}/{len(index_rows)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

