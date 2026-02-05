#!/usr/bin/env python3
"""
Compact a CEA dual-method review bundle by removing per-item "ancillary" files.

Input
-----
A bundle produced by scripts/bundle_cea_dual_method_review.py, which has:
  <IN_DIR>/
    README.txt
    cea-1995q3/
      index.csv
      issues.csv
      items/<ITEM_ID>/
        baseline.json
        source_text.txt
        covenantir_validated.json
        prompt_v1_short_output.txt
        combined_summary.json
        ... extra debug/comparison files ...
    cea-1995q4/
      ...

Output
------
A copy-based compact bundle that preserves item-level reviewability but keeps only:
  - baseline.json
  - source_text.txt
  - covenantir_validated.json
  - prompt_v1_short_output.txt
  - comparison.json (renamed from combined_summary.json)

It also writes a root README summarizing match rates:
  - Each method vs baseline
  - Both methods vs each other (triage counts)

This script does NOT make any new LLM calls; it only repackages artifacts.
"""

from __future__ import annotations

import argparse
import csv
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List


def _copy_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def _read_csv_dicts(path: Path) -> List[Dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        return [dict(row) for row in reader]


def _pct(n: int, d: int) -> str:
    if d <= 0:
        return "NA"
    return f"{(100.0 * n / d):.2f}%"


def _find_run_dirs(bundle_root: Path) -> List[Path]:
    runs: List[Path] = []
    for child in sorted(bundle_root.iterdir()):
        if not child.is_dir():
            continue
        if (child / "index.csv").exists() and (child / "items").is_dir():
            runs.append(child)
    if not runs:
        raise SystemExit(f"No run dirs found under: {bundle_root}")
    return runs


@dataclass(frozen=True)
class RunStats:
    run_id: str
    total_items: int
    covenantir_matched: int
    prompt_matched: int
    triage_counts: Dict[str, int]


def _compute_run_stats(run_dir: Path) -> RunStats:
    index_path = run_dir / "index.csv"
    if not index_path.exists():
        raise SystemExit(f"Missing index.csv: {index_path}")
    rows = _read_csv_dicts(index_path)
    total = len(rows)

    covenantir_matched = sum(1 for r in rows if (r.get("covenantir_disposition") or "") == "ok_baseline_matched")
    prompt_matched = sum(1 for r in rows if (r.get("prompt_v1_short_disposition") or "") == "ok_baseline_matched")

    triage_counts: Dict[str, int] = {}
    for r in rows:
        triage = (r.get("triage") or "").strip() or "missing"
        triage_counts[triage] = triage_counts.get(triage, 0) + 1

    return RunStats(
        run_id=run_dir.name,
        total_items=total,
        covenantir_matched=covenantir_matched,
        prompt_matched=prompt_matched,
        triage_counts=triage_counts,
    )


def _iter_item_dirs(run_dir: Path) -> Iterable[Path]:
    items_dir = run_dir / "items"
    if not items_dir.is_dir():
        raise SystemExit(f"Missing items/ dir: {items_dir}")
    for child in sorted(items_dir.iterdir()):
        if child.is_dir():
            yield child


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _render_root_readme(stats: List[RunStats]) -> str:
    total_items = sum(s.total_items for s in stats)
    total_cov = sum(s.covenantir_matched for s in stats)
    total_prompt = sum(s.prompt_matched for s in stats)
    total_triage: Dict[str, int] = {}
    for s in stats:
        for k, v in s.triage_counts.items():
            total_triage[k] = total_triage.get(k, 0) + v

    lines: List[str] = []
    lines.append("CEA covenant extraction comparison (compact dual-method bundle)")
    lines.append("==============================================================")
    lines.append("")
    lines.append("What this is")
    lines.append("------------")
    lines.append("This folder is a copy-based review bundle for the CEA (handcheck) baseline.")
    lines.append("For each item_id, it collocates:")
    lines.append("  - baseline (CEA handcheck)")
    lines.append("  - source paragraph text")
    lines.append("  - CovenantIR v0.1 output (structured + validated)")
    lines.append("  - prompt_v1_short_openai output (legacy JSON-list)")
    lines.append("  - a per-item comparison summary")
    lines.append("")
    lines.append("Match rates vs baseline (headline)")
    lines.append("-------------------------------")
    lines.append(f"Total items: {total_items}")
    lines.append(
        f"- prompt_v1_short_openai matched baseline: {total_prompt}/{total_items} ({_pct(total_prompt, total_items)})"
    )
    lines.append(f"- CovenantIR v0.1 matched baseline:       {total_cov}/{total_items} ({_pct(total_cov, total_items)})")
    lines.append("")
    lines.append("Match overlap between the two methods (triage)")
    lines.append("---------------------------------------------")
    # These triage labels are defined by baseline matching:
    # - both_match: both methods matched baseline
    # - only_*: only one method matched baseline
    # - neither_match: neither matched baseline
    for key in ("both_match", "only_prompt_v1_short_match", "only_covenantir_match", "neither_match"):
        if key in total_triage:
            lines.append(f"- {key}: {total_triage[key]}/{total_items} ({_pct(total_triage[key], total_items)})")
    # Include any other unexpected labels
    extra = sorted(k for k in total_triage.keys() if k not in {"both_match", "only_prompt_v1_short_match", "only_covenantir_match", "neither_match"})
    for k in extra:
        lines.append(f"- {k}: {total_triage[k]}/{total_items} ({_pct(total_triage[k], total_items)})")
    lines.append("")
    lines.append("Per-run breakdown")
    lines.append("-----------------")
    for s in stats:
        lines.append(f"{s.run_id}:")
        lines.append(f"  items: {s.total_items}")
        lines.append(
            f"  prompt_v1_short_openai matched baseline: {s.prompt_matched}/{s.total_items} ({_pct(s.prompt_matched, s.total_items)})"
        )
        lines.append(
            f"  CovenantIR v0.1 matched baseline:       {s.covenantir_matched}/{s.total_items} ({_pct(s.covenantir_matched, s.total_items)})"
        )
        for key in ("both_match", "only_prompt_v1_short_match", "only_covenantir_match", "neither_match"):
            if key in s.triage_counts:
                lines.append(f"  {key}: {s.triage_counts[key]}/{s.total_items} ({_pct(s.triage_counts[key], s.total_items)})")
        lines.append("")
    lines.append("How to review quickly")
    lines.append("---------------------")
    lines.append("1) Start with <RUN_ID>/issues.csv to focus on disagreements/failures.")
    lines.append("2) Pick an item_id and open the corresponding folder:")
    lines.append("   <RUN_ID>/items/<ITEM_ID>/")
    lines.append("3) Read source_text.txt, then compare:")
    lines.append("   - baseline.json")
    lines.append("   - covenantir_validated.json")
    lines.append("   - prompt_v1_short_output.txt")
    lines.append("   - comparison.json")
    lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", required=True, help="Input dual-method bundle root (review_bundle_v2_dual).")
    ap.add_argument("--out-dir", required=True, help="Output compact bundle root (must not exist).")
    args = ap.parse_args()

    in_dir = Path(args.in_dir)
    if not in_dir.exists():
        raise SystemExit(f"in-dir not found: {in_dir}")

    out_dir = Path(args.out_dir)
    if out_dir.exists():
        raise SystemExit(f"out-dir already exists: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=False)

    run_dirs = _find_run_dirs(in_dir)
    stats = [_compute_run_stats(r) for r in run_dirs]
    _write_text(out_dir / "README.txt", _render_root_readme(stats))

    # Copy per-run artifacts + per-item minimal set.
    for run_in in run_dirs:
        run_out = out_dir / run_in.name
        run_out.mkdir(parents=True, exist_ok=False)

        # Copy indexes (these are the primary navigation surface).
        for name in ("index.csv", "issues.csv"):
            src = run_in / name
            if not src.exists():
                raise SystemExit(f"Missing required run-level file: {src}")
            _copy_file(src, run_out / name)

        # Optional run-level provenance files (few files; useful for deeper debugging).
        for name in ("manifest.json", "covenantir_report.json", "prompt_v1_short_report.json"):
            src = run_in / name
            if src.exists():
                _copy_file(src, run_out / name)

        # Run-level README describing the compact item layout.
        _write_text(
            run_out / "README.txt",
            (
                f"{run_in.name} (compact dual-method bundle)\n"
                f"{'=' * (len(run_in.name) + 29)}\n\n"
                f"Item folders are under items/<ITEM_ID>/ and contain only:\n"
                f"- source_text.txt\n"
                f"- baseline.json\n"
                f"- covenantir_validated.json\n"
                f"- prompt_v1_short_output.txt\n"
                f"- comparison.json\n"
            ),
        )

        (run_out / "items").mkdir(parents=True, exist_ok=False)
        for item_in in _iter_item_dirs(run_in):
            item_id = item_in.name
            item_out = run_out / "items" / item_id
            item_out.mkdir(parents=True, exist_ok=False)

            required = {
                "baseline.json": "baseline.json",
                "source_text.txt": "source_text.txt",
                "covenantir_validated.json": "covenantir_validated.json",
                "prompt_v1_short_output.txt": "prompt_v1_short_output.txt",
                "combined_summary.json": "comparison.json",
            }
            for src_name, dst_name in required.items():
                src = item_in / src_name
                if not src.exists():
                    raise SystemExit(f"Missing required item file: {src}")
                _copy_file(src, item_out / dst_name)

    print(f"[done] wrote compact bundle: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

