#!/usr/bin/env python3
"""
Export CEA run inputs into a single easy-to-consume file for re-running extraction.

This takes an existing CEA run (runs/<run-id>/manifest.json + normalized/<item_id>/canonical.txt)
and produces a JSONL file where each line is one item:
  - item_id
  - paragraph text (canonical)
  - snippet text (retrieval_v2; what was fed to the model)
  - baseline + provenance metadata from manifest.json

Why:
  The run directory structure is great for pipelines but awkward for humans.
  This exporter creates a single "open a file and iterate" artifact.

Usage:
  python scripts/export_cea_inputs_for_rerun.py \\
    --run-ids cea-1995q3 cea-1995q4 \\
    --out-jsonl scratch/cea_1995q3q4_inputs.jsonl \\
    --out-txt scratch/cea_1995q3q4_paragraphs.txt
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def _read_json(path: Path) -> Any:
    return json.loads(_read_text(path))


def _iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    if not path.exists():
        return
    with path.open("r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if isinstance(obj, dict):
                yield obj


def _first_snippet(snippets_path: Path) -> str:
    for rec in _iter_jsonl(snippets_path):
        snip = rec.get("snippet")
        if isinstance(snip, str) and snip.strip():
            return snip
    return ""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-ids", nargs="+", required=True, help="CEA run ids under runs/<run-id>/")
    ap.add_argument("--base-dir", default=".", help="Repo base dir")
    ap.add_argument("--out-jsonl", required=True, help="Write JSONL records here")
    ap.add_argument("--out-txt", default="", help="Optional: write a human-readable text file here")
    args = ap.parse_args()

    base_dir = Path(args.base_dir)

    out_jsonl = Path(args.out_jsonl)
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)

    out_txt: Optional[Path] = Path(args.out_txt) if args.out_txt else None
    if out_txt:
        out_txt.parent.mkdir(parents=True, exist_ok=True)

    n = 0
    with out_jsonl.open("w", encoding="utf-8") as fj, (out_txt.open("w", encoding="utf-8") if out_txt else open("/dev/null", "w")) as ft:
        for run_id in args.run_ids:
            run_dir = base_dir / "runs" / run_id
            manifest_path = run_dir / "manifest.json"
            manifest = _read_json(manifest_path)
            items = manifest.get("items")
            if not isinstance(items, list):
                raise SystemExit(f"manifest.json has no items list: {manifest_path}")

            for it in items:
                if not isinstance(it, dict):
                    continue
                item_id = str(it.get("item_id") or "")
                if not item_id:
                    continue
                canonical_path = run_dir / "normalized" / item_id / "canonical.txt"
                paragraph = _read_text(canonical_path) if canonical_path.exists() else ""

                snippets_path = run_dir / "retrieval_v2" / f"{item_id}_snippets.jsonl"
                snippet = _first_snippet(snippets_path) if snippets_path.exists() else ""

                rec = {
                    "run_id": run_id,
                    "item_id": item_id,
                    "paragraph": paragraph,
                    "snippet": snippet,
                    "baseline": it.get("baseline"),
                    "source": it.get("source"),
                }
                fj.write(json.dumps(rec, ensure_ascii=False) + "\n")

                if out_txt:
                    ft.write("=== ITEM ===\n")
                    ft.write(f"run_id: {run_id}\n")
                    ft.write(f"item_id: {item_id}\n")
                    baseline = it.get("baseline") if isinstance(it.get("baseline"), dict) else {}
                    ft.write(f"baseline_name: {baseline.get('name')}\n")
                    ft.write(f"baseline_value: {baseline.get('limit')}\n")
                    src = it.get("source") if isinstance(it.get("source"), dict) else {}
                    ft.write(f"source_file: {src.get('file')}\n")
                    ft.write(f"segment_no: {src.get('segment_no')}\n")
                    ft.write(f"row_index: {src.get('row_index')}\n")
                    ft.write("--- paragraph ---\n")
                    ft.write(paragraph.strip() + "\n\n")

                n += 1

    print(f"Wrote JSONL: {out_jsonl} ({n} items)")
    if out_txt:
        print(f"Wrote TXT: {out_txt} ({n} items)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

