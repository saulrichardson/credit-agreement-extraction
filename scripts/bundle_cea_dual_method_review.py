#!/usr/bin/env python3
"""
Bundle CEA baseline + source text + TWO extraction methods into one review folder.

Use case
--------
You want a single, shareable folder where a reviewer can open items/<ITEM_ID>/ and see:
  - what the paragraph says (source_text.txt)
  - what the CEA baseline claims (baseline.json)
  - what CovenantIR produced (covenantir_validated.json + covenantir_result.json)
  - what prompt_v1_short produced (prompt_v1_short_output.txt)
  - the comparator's per-item judgment for each method

This is copy-based (no symlinks) so the bundle can be moved to Dropbox/shared.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional


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


def _parse_timestamp_like_to_iso(s: str) -> str:
    s = (s or "").strip()
    # "YYYY-MM-DD HH:MM:SS" -> YYYY-MM-DD
    m = re.match(r"^(\d{4}-\d{2}-\d{2})\s+\d{2}:\d{2}:\d{2}$", s)
    if m:
        return m.group(1)
    return ""


def _flatten_thresholds(values: Any, max_values: int = 12) -> str:
    if not isinstance(values, list):
        return ""
    vals = [str(v) for v in values if v is not None]
    if len(vals) > max_values:
        vals = vals[:max_values] + ["…"]
    return ";".join(vals)


@dataclass(frozen=True)
class IndexRow:
    run_id: str
    item_id: str
    baseline_name: str
    baseline_limit_raw: str
    baseline_start_date: str
    baseline_end_date: str
    covenantir_status: str
    covenantir_disposition: str
    covenantir_best_title: str
    covenantir_best_thresholds: str
    prompt_v1_short_status: str
    prompt_v1_short_disposition: str
    prompt_v1_short_best_name: str
    prompt_v1_short_best_limit: str
    triage: str
    item_dir: str


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True, help="CEA run_id under runs/<run_id>/ (e.g., cea-1995q3).")

    ap.add_argument("--covenantir-report", required=True, help="CovenantIR-vs-CEA report JSON (v3 comparator).")
    ap.add_argument("--covenantir-dir", required=True, help="Directory with per-item CovenantIR outputs (subdirs by item_id).")

    ap.add_argument(
        "--prompt-v1-short-report",
        required=True,
        help="prompt_v1_short comparator report JSON (scripts/compare_prompt_v1_short_to_cea_baseline.py output).",
    )
    ap.add_argument(
        "--prompt-v1-short-output-dir",
        default=None,
        help=(
            "Directory containing <ITEM_ID>.txt outputs for prompt_v1_short (defaults to runs/<run_id>/llm_qa/<prompt_subdir>/ "
            "where <prompt_subdir> comes from the report JSON)."
        ),
    )

    ap.add_argument("--base-dir", default=".", help="Repo base dir")
    ap.add_argument("--out-dir", required=True, help="Output bundle directory (created; must not already exist).")
    ap.add_argument("--include-covenantir-prompt-context", action="store_true", help="Copy contexts.txt + prompt.txt when present.")
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

    cov_report_path = Path(args.covenantir_report)
    if not cov_report_path.exists():
        raise SystemExit(f"Missing covenantir-report: {cov_report_path}")

    prompt_report_path = Path(args.prompt_v1_short_report)
    if not prompt_report_path.exists():
        raise SystemExit(f"Missing prompt-v1-short-report: {prompt_report_path}")

    out_dir = Path(args.out_dir)
    if out_dir.exists():
        raise SystemExit(f"out-dir already exists: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=False)

    manifest = _read_json(manifest_path)
    items = manifest.get("items")
    if not isinstance(items, list) or not items:
        raise SystemExit(f"manifest.json has no items: {manifest_path}")

    cov_report = _read_json(cov_report_path)
    cov_report_items = cov_report.get("items")
    if not isinstance(cov_report_items, list):
        raise SystemExit(f"covenantir-report has no items list: {cov_report_path}")
    cov_by_id: Dict[str, Dict[str, Any]] = {}
    for it in cov_report_items:
        if isinstance(it, dict) and isinstance(it.get("item_id"), str):
            cov_by_id[it["item_id"]] = it

    prompt_report = _read_json(prompt_report_path)
    prompt_report_items = prompt_report.get("items")
    if not isinstance(prompt_report_items, list):
        raise SystemExit(f"prompt-v1-short-report has no items list: {prompt_report_path}")
    prompt_by_id: Dict[str, Dict[str, Any]] = {}
    for it in prompt_report_items:
        if isinstance(it, dict) and isinstance(it.get("item_id"), str):
            prompt_by_id[it["item_id"]] = it

    prompt_subdir = str(prompt_report.get("prompt_subdir") or "")
    if not prompt_subdir:
        raise SystemExit(f"prompt-v1-short-report missing prompt_subdir: {prompt_report_path}")

    prompt_out_dir = (
        Path(args.prompt_v1_short_output_dir)
        if args.prompt_v1_short_output_dir
        else (run_dir / "llm_qa" / prompt_subdir)
    )
    if not prompt_out_dir.exists():
        raise SystemExit(f"prompt_v1_short_output_dir not found: {prompt_out_dir}")

    # Bundle roots.
    (out_dir / "items").mkdir(parents=True, exist_ok=True)
    _copy_file(manifest_path, out_dir / "manifest.json")
    _copy_file(cov_report_path, out_dir / "covenantir_report.json")
    _copy_file(prompt_report_path, out_dir / "prompt_v1_short_report.json")

    readme = (
        f"CEA dual-method covenant review bundle\n"
        f"=====================================\n\n"
        f"run_id: {run_id}\n\n"
        f"This bundle collocates:\n"
        f"- baseline (CEA)\n"
        f"- source paragraph text\n"
        f"- CovenantIR output + comparison\n"
        f"- prompt_v1_short output + comparison\n\n"
        f"Folder layout:\n"
        f"- items/<ITEM_ID>/\n"
        f"  - source_text.txt\n"
        f"  - baseline.json\n"
        f"  - covenantir_result.json\n"
        f"  - covenantir_validated.json (if available)\n"
        f"  - covenantir_comparison.json\n"
        f"  - prompt_v1_short_output.txt\n"
        f"  - prompt_v1_short_comparison.json\n"
        f"  - combined_summary.json\n\n"
        f"Indexes:\n"
        f"- index.csv (all items)\n"
        f"- issues.csv (items where triage != both_match)\n"
    )
    if args.include_covenantir_prompt_context:
        readme += (
            f"\nCovenantIR prompt artifacts (when present):\n"
            f"- items/<ITEM_ID>/contexts.txt\n"
            f"- items/<ITEM_ID>/prompt.txt\n"
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

        # Baseline + provenance (from manifest).
        baseline = raw.get("baseline") if isinstance(raw.get("baseline"), dict) else {}
        source = raw.get("source") if isinstance(raw.get("source"), dict) else {}
        _write_json(item_dir / "baseline.json", {"item_id": item_id, "baseline": baseline, "source": source})

        baseline_name = _safe_str(baseline.get("name"))
        baseline_limit_raw = _safe_str(baseline.get("limit"))
        baseline_start = _parse_timestamp_like_to_iso(_safe_str(baseline.get("start_date")))
        baseline_end = _parse_timestamp_like_to_iso(_safe_str(baseline.get("end_date")))

        # CovenantIR outputs + per-item comparison.
        cov_item_dir = covenantir_dir / item_id
        cov_result = cov_item_dir / "result.json"
        cov_validated = cov_item_dir / "covenantir_validated.json"
        if cov_result.exists():
            _copy_file(cov_result, item_dir / "covenantir_result.json")
        else:
            _write_text(item_dir / "covenantir_result.json", "{\n  \"status\": \"missing\"\n}\n")
        if cov_validated.exists():
            _copy_file(cov_validated, item_dir / "covenantir_validated.json")
        if args.include_covenantir_prompt_context:
            for name in ("contexts.txt", "prompt.txt"):
                src = cov_item_dir / name
                if src.exists():
                    _copy_file(src, item_dir / name)

        cov_rep = cov_by_id.get(item_id)
        if cov_rep is None:
            _write_json(item_dir / "covenantir_comparison.json", {"item_id": item_id, "error": "missing_in_covenantir_report"})
            cov_disp = "missing_in_report"
            cov_status = "unknown"
            cov_best = {}
        else:
            keep = {
                "item_id": item_id,
                "disposition": cov_rep.get("disposition"),
                "baseline": cov_rep.get("baseline"),
                "covenantir": cov_rep.get("covenantir"),
                "match": cov_rep.get("match"),
                "best_match": cov_rep.get("best_match"),
                "candidates": cov_rep.get("candidates"),
            }
            _write_json(item_dir / "covenantir_comparison.json", keep)
            cov_disp = _safe_str(cov_rep.get("disposition"))
            cov_status = _safe_str(
                ((cov_rep.get("covenantir") or {}) if isinstance(cov_rep.get("covenantir"), dict) else {}).get("status")
            )
            cov_best = cov_rep.get("best_match") if isinstance(cov_rep.get("best_match"), dict) else {}

            # Prefer the comparator's normalized baseline dates if present.
            cov_base = cov_rep.get("baseline") if isinstance(cov_rep.get("baseline"), dict) else {}
            if cov_base.get("start_date_iso"):
                baseline_start = _safe_str(cov_base.get("start_date_iso"))
            if cov_base.get("end_date_iso"):
                baseline_end = _safe_str(cov_base.get("end_date_iso"))

        # prompt_v1_short output + per-item comparison.
        prompt_output_path = prompt_out_dir / f"{item_id}.txt"
        if prompt_output_path.exists():
            _copy_file(prompt_output_path, item_dir / "prompt_v1_short_output.txt")
        else:
            # Preserve errors/skips if present.
            err_path = prompt_out_dir / f"{item_id}.error.txt"
            if err_path.exists():
                _copy_file(err_path, item_dir / "prompt_v1_short_output.error.txt")
            else:
                _write_text(item_dir / "prompt_v1_short_output.txt", "<missing prompt output>\n")

        prompt_rep = prompt_by_id.get(item_id)
        if prompt_rep is None:
            _write_json(item_dir / "prompt_v1_short_comparison.json", {"item_id": item_id, "error": "missing_in_prompt_report"})
            prompt_disp = "missing_in_report"
            prompt_status = "unknown"
            prompt_best = {}
        else:
            keep = {
                "item_id": item_id,
                "disposition": prompt_rep.get("disposition"),
                "baseline": prompt_rep.get("baseline"),
                "prompt_v1_short": prompt_rep.get("prompt_v1_short"),
                "match": prompt_rep.get("match"),
                "best_match": prompt_rep.get("best_match"),
                "candidates": prompt_rep.get("candidates"),
            }
            _write_json(item_dir / "prompt_v1_short_comparison.json", keep)
            prompt_disp = _safe_str(prompt_rep.get("disposition"))
            prompt_status = _safe_str(
                ((prompt_rep.get("prompt_v1_short") or {}) if isinstance(prompt_rep.get("prompt_v1_short"), dict) else {}).get("status")
            )
            prompt_best = prompt_rep.get("best_match") if isinstance(prompt_rep.get("best_match"), dict) else {}

        cov_ok = cov_disp == "ok_baseline_matched"
        prompt_ok = prompt_disp == "ok_baseline_matched"
        if cov_ok and prompt_ok:
            triage = "both_match"
        elif cov_ok and not prompt_ok:
            triage = "only_covenantir_match"
        elif prompt_ok and not cov_ok:
            triage = "only_prompt_v1_short_match"
        else:
            triage = "neither_match"

        combined = {
            "item_id": item_id,
            "baseline": {
                "name": baseline_name,
                "limit": baseline_limit_raw,
                "start_date": baseline_start or None,
                "end_date": baseline_end or None,
            },
            "triage": triage,
            "covenantir": {
                "status": cov_status,
                "disposition": cov_disp,
                "best_match": cov_best,
            },
            "prompt_v1_short": {
                "status": prompt_status,
                "disposition": prompt_disp,
                "best_match": prompt_best,
            },
        }
        _write_json(item_dir / "combined_summary.json", combined)

        index_rows.append(
            IndexRow(
                run_id=run_id,
                item_id=item_id,
                baseline_name=baseline_name,
                baseline_limit_raw=baseline_limit_raw,
                baseline_start_date=baseline_start,
                baseline_end_date=baseline_end,
                covenantir_status=cov_status,
                covenantir_disposition=cov_disp,
                covenantir_best_title=_safe_str(cov_best.get("title")),
                covenantir_best_thresholds=_flatten_thresholds(cov_best.get("threshold_values")),
                prompt_v1_short_status=prompt_status,
                prompt_v1_short_disposition=prompt_disp,
                prompt_v1_short_best_name=_safe_str(prompt_best.get("name")),
                prompt_v1_short_best_limit=_safe_str(prompt_best.get("limit")),
                triage=triage,
                item_dir=str(item_dir.relative_to(out_dir)),
            )
        )

    def _write_index_csv(path: Path, rows: List[IndexRow]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(IndexRow.__annotations__.keys()))
            w.writeheader()
            for r in rows:
                w.writerow(r.__dict__)

    _write_index_csv(out_dir / "index.csv", index_rows)
    issues = [r for r in index_rows if r.triage != "both_match"]
    _write_index_csv(out_dir / "issues.csv", issues)

    print(f"[done] wrote bundle: {out_dir}")
    print(f"[index] {out_dir / 'index.csv'}")
    print(f"[issues] {out_dir / 'issues.csv'} ({len(issues)}/{len(index_rows)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

