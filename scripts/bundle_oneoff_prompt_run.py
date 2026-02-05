#!/usr/bin/env python3
"""
Create a minimal, shareable folder for a one-off prompt run.

Bundle contents (under runs/<run_id>/deliverables/<bundle_name>/):
  - prompt.txt                       (prompt template used)
  - report.json                      (oneoff_prompt_run.json summary, if present)
  - items/<item_id>/agreement.txt    (normalized canonical_annotated.txt, or canonical.txt fallback)
  - items/<item_id>/llm_output.txt   (raw model output when present)
  - items/<item_id>/status.txt       (present when output is missing, e.g., skipped)

This intentionally omits indexing/retrieval internals; it's meant for quick review/sharing.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def _read_manifest_item_ids(manifest_path: Path) -> list[str]:
    doc = json.loads(manifest_path.read_text(encoding="utf-8"))
    items = doc.get("items") or []
    out: list[str] = []
    for rec in items:
        if isinstance(rec, dict) and isinstance(rec.get("item_id"), str) and rec["item_id"].strip():
            out.append(rec["item_id"].strip())
    if not out:
        raise RuntimeError(f"No item_ids found in manifest: {manifest_path}")
    return out


def _copy_text(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--output-subdir", required=True, help="Subdir under runs/<run_id>/llm_qa/ containing <item_id>.txt outputs.")
    ap.add_argument(
        "--bundle-name",
        default=None,
        help="Folder name under runs/<run_id>/deliverables/ (defaults to output_subdir + '_bundle').",
    )
    ap.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete an existing bundle folder before writing a new one.",
    )
    args = ap.parse_args()

    run_dir = Path("runs") / args.run_id
    if not run_dir.exists():
        raise SystemExit(f"Run dir not found: {run_dir}")

    manifest_path = run_dir / "manifest.json"
    if not manifest_path.exists():
        raise SystemExit(f"Missing manifest: {manifest_path}")

    output_dir = run_dir / "llm_qa" / args.output_subdir
    if not output_dir.exists():
        raise SystemExit(f"Missing output dir: {output_dir}")

    bundle_name = args.bundle_name or f"{args.output_subdir}_bundle"
    bundle_dir = run_dir / "deliverables" / bundle_name
    if bundle_dir.exists():
        if not args.overwrite:
            raise SystemExit(f"Bundle already exists: {bundle_dir} (pass --overwrite to recreate)")
        shutil.rmtree(bundle_dir)
    (bundle_dir / "items").mkdir(parents=True, exist_ok=True)

    # Prompt (preserve original filename when available via report).
    prompt_used = run_dir / "prompt_used.txt"
    if not prompt_used.exists():
        raise SystemExit(f"Missing prompt_used.txt: {prompt_used}")

    # Report (optional)
    report = run_dir / "report" / "oneoff_prompt_run.json"
    prompt_filename = prompt_used.name
    if report.exists():
        _copy_text(report, bundle_dir / "report.json")
        try:
            report_doc = json.loads(report.read_text(encoding="utf-8"))
            original = report_doc.get("prompt_original_path")
            if isinstance(original, str) and original.strip():
                prompt_filename = Path(original.strip()).name or prompt_filename
        except Exception:
            # If report parsing fails, fall back to prompt_used.txt name.
            pass

    _copy_text(prompt_used, bundle_dir / prompt_filename)

    # Items
    item_ids = _read_manifest_item_ids(manifest_path)
    for item_id in item_ids:
        item_dir = bundle_dir / "items" / item_id
        item_dir.mkdir(parents=True, exist_ok=True)

        # Agreement: prefer canonical_annotated with anchors, then canonical.
        ann = run_dir / "normalized" / item_id / "canonical_annotated.txt"
        can = run_dir / "normalized" / item_id / "canonical.txt"
        if ann.exists():
            _copy_text(ann, item_dir / "agreement.txt")
        elif can.exists():
            _copy_text(can, item_dir / "agreement.txt")
        else:
            (item_dir / "status.txt").write_text(
                "missing agreement (expected normalized/<item_id>/canonical_annotated.txt)\n",
                encoding="utf-8",
            )

        out_txt = output_dir / f"{item_id}.txt"
        err_txt = output_dir / f"{item_id}.error.txt"
        skip_txt = output_dir / f"{item_id}.skipped.txt"
        if out_txt.exists():
            _copy_text(out_txt, item_dir / "llm_output.txt")
        elif err_txt.exists():
            _copy_text(err_txt, item_dir / "status.txt")
        elif skip_txt.exists():
            _copy_text(skip_txt, item_dir / "status.txt")
        else:
            # No output artifact at all (unexpected but keep bundle complete).
            (item_dir / "status.txt").write_text("missing llm output\n", encoding="utf-8")

    print(f"[done] bundle: {bundle_dir}")


if __name__ == "__main__":
    main()
