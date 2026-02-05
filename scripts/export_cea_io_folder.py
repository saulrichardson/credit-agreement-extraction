#!/usr/bin/env python3
"""
Export a CEA run's per-item inputs + outputs into a simple folder structure.

User goal:
  Create two directories:
    - inputs/: one file per item containing the exact input text
    - outputs/: one file per item containing the model output

This is intended for "no tarballs" sharing / re-running extraction in a different workflow.

By default, the input text is the *exact snippet* that was fed to the model:
  runs/<run_id>/retrieval_v2/<item_id>_snippets.jsonl  (field: "snippet")

And the output text is copied from:
  runs/<run_id>/llm_qa/<prompt_subdir>/<item_id>.txt

Example:
  python scripts/export_cea_io_folder.py \\
    --run-id cea-1995q3 \\
    --prompt-subdir prompt_v1_short_openai \\
    --out-dir /path/to/cea_prompt_v1_short_basics \\
    --overwrite
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any, Dict, Iterable, Optional


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def _read_json(path: Path) -> Any:
    return json.loads(_read_text(path))


def _iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
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


def _first_snippet(snippets_jsonl: Path) -> Optional[str]:
    if not snippets_jsonl.exists():
        return None
    for rec in _iter_jsonl(snippets_jsonl):
        snip = rec.get("snippet")
        if isinstance(snip, str) and snip.strip():
            return snip
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True, help="Run id under runs/<run-id>/")
    ap.add_argument("--prompt-subdir", required=True, help="Subdir under runs/<run-id>/llm_qa/")
    ap.add_argument("--base-dir", default=".", help="Repo base dir")
    ap.add_argument("--out-dir", required=True, help="Destination folder root")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing per-item files")
    args = ap.parse_args()

    base_dir = Path(args.base_dir)
    run_id = str(args.run_id)
    run_dir = base_dir / "runs" / run_id
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.exists():
        raise SystemExit(f"Missing manifest.json: {manifest_path}")

    manifest = _read_json(manifest_path)
    items = manifest.get("items")
    if not isinstance(items, list) or not items:
        raise SystemExit(f"manifest.json has no items list: {manifest_path}")

    out_root = Path(args.out_dir)
    inputs_dir = out_root / "inputs" / run_id
    outputs_dir = out_root / "outputs" / run_id
    inputs_dir.mkdir(parents=True, exist_ok=True)
    outputs_dir.mkdir(parents=True, exist_ok=True)

    llm_dir = run_dir / "llm_qa" / str(args.prompt_subdir)
    if not llm_dir.exists():
        raise SystemExit(f"Missing llm output dir: {llm_dir}")

    missing_snippet = 0
    missing_output = 0
    written = 0

    for it in items:
        if not isinstance(it, dict):
            continue
        item_id = str(it.get("item_id") or "")
        if not item_id:
            continue

        snippets_path = run_dir / "retrieval_v2" / f"{item_id}_snippets.jsonl"
        snippet = _first_snippet(snippets_path)
        if snippet is None:
            missing_snippet += 1
            continue

        in_path = inputs_dir / f"{item_id}.txt"
        if in_path.exists() and not args.overwrite:
            raise SystemExit(f"Refusing to overwrite existing file (pass --overwrite): {in_path}")
        in_path.write_text(snippet, encoding="utf-8")

        out_src = llm_dir / f"{item_id}.txt"
        if not out_src.exists():
            missing_output += 1
            continue

        out_dst = outputs_dir / f"{item_id}.txt"
        if out_dst.exists() and not args.overwrite:
            raise SystemExit(f"Refusing to overwrite existing file (pass --overwrite): {out_dst}")
        shutil.copyfile(out_src, out_dst)

        written += 1

    if missing_snippet:
        raise SystemExit(f"Missing snippet text for {missing_snippet} items under {run_id} (retrieval_v2/*.jsonl)")
    if missing_output:
        raise SystemExit(f"Missing model output for {missing_output} items under {run_id} (llm_qa/{args.prompt_subdir}/*.txt)")

    print(f"Wrote {written} inputs to {inputs_dir}")
    print(f"Wrote {written} outputs to {outputs_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

