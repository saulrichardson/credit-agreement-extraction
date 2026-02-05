#!/usr/bin/env python
"""
Build a local "run" folder from a covenant sample CSV (handcheck/openllm packs).

Purpose
-------
The CovenantIR harness expects the standard artifacts:
  - runs/<run_id>/normalized/<item_id>/anchors.tsv
  - runs/<run_id>/retrieval_v2/<item_id>_snippets.jsonl

These Dropbox sample packs already provide:
  - per-row excerpt text: chunk_text
  - per-row baseline covenant fields (name/limit/start/end)
  - per-row link to full agreement text: local_agreement_path (optional)

This script adapts each CSV row into a minimal retrieval_v2 snippet pack so we can run
new CovenantIR extraction calls and compare to the baseline.

Design choices (intentionally explicit; no silent heuristics):
  - One CSV row => one synthetic item_id => one excerpt-pack anchor (A0001).
  - Text fed to the model is EXACTLY `chunk_text` by default (not the full agreement file).
  - Full agreement text is recorded for provenance/debug only.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple


def _read_rows(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8", errors="replace", newline="") as fh:
        reader = csv.DictReader(fh)
        return [dict(r) for r in reader]


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _ensure_str(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, str):
        return v
    return str(v)


def _get_col(row: Mapping[str, Any], *names: str) -> Tuple[str, Optional[str]]:
    """Return (value, column_name_used)."""
    for n in names:
        if n in row:
            return (_ensure_str(row.get(n)), n)
    return ("", None)


def _derive_item_id(row: Mapping[str, Any]) -> str:
    # Preferred: segment_id = "<member>::<seg_no>"
    seg, _ = _get_col(row, "segment_id")
    if seg and "::" in seg:
        member_part, seg_no_part = seg.split("::", 1)
        member_part = member_part.strip()
        seg_no_part = seg_no_part.strip()
        if member_part.endswith(".nc"):
            member_part = member_part[: -len(".nc")]
        seg_no_part = seg_no_part or "0"
        return f"{member_part}_{seg_no_part}"

    member, _ = _get_col(row, "member")
    seg_no, _ = _get_col(row, "seg_no")
    if member:
        m = member.strip()
        if m.endswith(".nc"):
            m = m[: -len(".nc")]
        if seg_no.strip():
            return f"{m}_{seg_no.strip()}"
        return m

    custom_id, _ = _get_col(row, "custom_id", "handcheck_row_id")
    if custom_id.strip():
        return f"row_{custom_id.strip()}"

    raise ValueError("Unable to derive item_id (missing segment_id/member/custom_id).")


def _anchors_tsv(anchor_ids: Iterable[str]) -> str:
    lines = ["anchor_id\tanchor_type\tstart\tend\tlabel\n"]
    offset = 0
    for aid in anchor_ids:
        start = offset
        end = offset + 1
        offset += 10
        lines.append(f"{aid}\tsentence\t{start}\t{end}\t{aid}\n")
    return "".join(lines)


def _snippet_record(*, item_id: str, anchor_id: str, snippet: str) -> Dict[str, Any]:
    return {
        "item_id": item_id,
        "anchor_id": anchor_id,
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
        "snippet_end": len(snippet),
    }


@dataclass(frozen=True)
class BaselineCols:
    name: str
    limit: str
    start_date: str
    end_date: str
    name_col: str
    limit_col: str
    start_col: str | None
    end_col: str | None


def _detect_baseline_cols(rows: List[Mapping[str, Any]]) -> BaselineCols:
    if not rows:
        raise ValueError("CSV has no rows")
    cols = set(rows[0].keys())

    def _pick(*candidates: str) -> str:
        for c in candidates:
            if c in cols:
                return c
        raise ValueError(f"Missing required column; tried {candidates} (available={sorted(cols)})")

    name_col = _pick("openllm_name", "name")
    limit_col = _pick("openllm_limit", "limit")

    start_col = None
    end_col = None
    for c in ("openllm_start_date", "start date", "start_date"):
        if c in cols:
            start_col = c
            break
    for c in ("openllm_end_date", "end date", "end_date"):
        if c in cols:
            end_col = c
            break

    # Values are recorded per-row later; here we just return column names.
    return BaselineCols(
        name="",
        limit="",
        start_date="",
        end_date="",
        name_col=name_col,
        limit_col=limit_col,
        start_col=start_col,
        end_col=end_col,
    )


def build_run_from_csv(
    *,
    csv_path: Path,
    out_run_dir: Path,
    dropbox_base: Path,
    text_mode: str,
    overwrite: bool,
) -> Dict[str, Any]:
    if out_run_dir.exists() and not overwrite:
        raise SystemExit(f"Refusing to overwrite existing run dir: {out_run_dir} (pass --overwrite)")
    out_run_dir.mkdir(parents=True, exist_ok=True)

    rows = _read_rows(csv_path)
    cols = _detect_baseline_cols(rows)

    normalized_dir = out_run_dir / "normalized"
    retrieval_dir = out_run_dir / "retrieval_v2"

    items: List[Dict[str, Any]] = []
    seen: set[str] = set()

    for idx, row in enumerate(rows):
        item_id = _derive_item_id(row)
        if item_id in seen:
            raise ValueError(f"Duplicate derived item_id {item_id!r} at row {idx}")
        seen.add(item_id)

        chunk_text, _ = _get_col(row, "chunk_text")
        local_path_rel, _ = _get_col(row, "local_agreement_path")
        local_path_abs = str((dropbox_base / Path(local_path_rel)).resolve()) if local_path_rel else ""

        baseline_name = _ensure_str(row.get(cols.name_col))
        baseline_limit = _ensure_str(row.get(cols.limit_col))
        baseline_start = _ensure_str(row.get(cols.start_col)) if cols.start_col else ""
        baseline_end = _ensure_str(row.get(cols.end_col)) if cols.end_col else ""

        if text_mode == "chunk_text":
            text_for_model = chunk_text
        elif text_mode == "full_document":
            if not local_path_rel:
                raise ValueError(f"Row {idx} missing local_agreement_path; cannot use --text-mode full_document")
            full_path = dropbox_base / Path(local_path_rel)
            if not full_path.exists():
                raise FileNotFoundError(f"Missing full document at {full_path}")
            text_for_model = full_path.read_text(encoding="utf-8", errors="replace")
        else:
            raise ValueError(f"Unknown text_mode: {text_mode}")

        # Minimal anchor catalog: one excerpt pack anchor.
        anchor_ids = ["A0001"]
        _write_text(normalized_dir / item_id / "anchors.tsv", _anchors_tsv(anchor_ids))

        rec = _snippet_record(item_id=item_id, anchor_id="A0001", snippet=text_for_model)
        out_path = retrieval_dir / f"{item_id}_snippets.jsonl"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(rec, ensure_ascii=False) + "\n", encoding="utf-8")

        items.append(
            {
                "row_index": idx,
                "item_id": item_id,
                "baseline": {
                    "name": baseline_name,
                    "limit": baseline_limit,
                    "start_date": baseline_start,
                    "end_date": baseline_end,
                    "name_col": cols.name_col,
                    "limit_col": cols.limit_col,
                    "start_col": cols.start_col,
                    "end_col": cols.end_col,
                },
                "source": {
                    "csv": str(csv_path),
                    "chunk_text_chars": len(chunk_text),
                    "local_agreement_path_rel": local_path_rel,
                    "local_agreement_path_abs": local_path_abs,
                },
            }
        )

    manifest = {
        "schema_version": "local_covenant_sample_run_v1",
        "created_at": int(time.time()),
        "text_mode": text_mode,
        "dropbox_base": str(dropbox_base),
        "csv_path": str(csv_path),
        "items": items,
    }
    _write_json(out_run_dir / "local_sample_manifest.json", manifest)
    return manifest


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", dest="csv_path", required=True)
    ap.add_argument("--run-id", required=True, help="Output under runs/<run-id>/")
    ap.add_argument("--base-dir", default=".", help="Repo base dir (writes under <base-dir>/runs/<run-id>/)")
    ap.add_argument(
        "--dropbox-base",
        default="/Users/saulrichardson/Dropbox/edgar",
        help="Base dir used to resolve local_agreement_path relative paths",
    )
    ap.add_argument("--text-mode", choices=["chunk_text", "full_document"], default="chunk_text")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    base_dir = Path(args.base_dir)
    out_run_dir = base_dir / "runs" / args.run_id
    manifest = build_run_from_csv(
        csv_path=Path(args.csv_path),
        out_run_dir=out_run_dir,
        dropbox_base=Path(args.dropbox_base),
        text_mode=str(args.text_mode),
        overwrite=bool(args.overwrite),
    )
    print(f"Wrote run dir: {out_run_dir}")
    print(f"Items: {len(manifest.get('items') or [])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

