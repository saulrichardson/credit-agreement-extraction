#!/usr/bin/env python3
"""
Create runs/<run_id>/manifest.json from runs/<run_id>/local_sample_manifest.json.

Why this exists
--------------
Local sample runs built by scripts/build_local_covenant_sample_run.py write:
  - runs/<run_id>/retrieval_v2/<item_id>_snippets.jsonl
  - runs/<run_id>/normalized/<item_id>/...
  - runs/<run_id>/local_sample_manifest.json   (NOT manifest.json)

But the standard pipeline runner `poetry run python -m pipeline.run structured-v2`
expects runs/<run_id>/manifest.json with an "items" list.

This script bridges that gap without changing the local sample manifest format.
It does NOT run any LLM calls.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8", errors="replace"))


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True, help="Existing local sample run under runs/<run-id>/")
    ap.add_argument("--base-dir", default=".", help="Repo base dir containing runs/<run-id>/")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite runs/<run-id>/manifest.json if it exists.")
    args = ap.parse_args()

    run_dir = Path(args.base_dir) / "runs" / str(args.run_id)
    local_manifest_path = run_dir / "local_sample_manifest.json"
    out_manifest_path = run_dir / "manifest.json"

    if not local_manifest_path.exists():
        raise SystemExit(f"Missing local sample manifest: {local_manifest_path}")
    if out_manifest_path.exists() and not args.overwrite:
        raise SystemExit(f"Refusing to overwrite existing manifest.json: {out_manifest_path} (pass --overwrite)")

    local = _read_json(local_manifest_path)
    items = local.get("items")
    if not isinstance(items, list) or not items:
        raise SystemExit(f"local_sample_manifest.json has no items: {local_manifest_path}")

    out_items: List[Dict[str, Any]] = []
    for it in items:
        if not isinstance(it, dict):
            continue
        item_id = it.get("item_id")
        if not isinstance(item_id, str) or not item_id.strip():
            continue
        out_items.append(
            {
                "item_id": item_id.strip(),
                # Preserve provenance/baseline even though structured-v2 doesn't require it.
                "row_index": it.get("row_index"),
                "baseline": it.get("baseline"),
                "source": it.get("source"),
            }
        )

    if not out_items:
        raise SystemExit(f"No valid items found in {local_manifest_path}")

    manifest = {
        "schema_version": "pipeline_manifest_from_local_sample_manifest_v1",
        "created_at": int(time.time()),
        "source_local_sample_manifest": str(local_manifest_path),
        "items": out_items,
    }
    _write_json(out_manifest_path, manifest)
    print(f"Wrote {out_manifest_path} (items={len(out_items)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

