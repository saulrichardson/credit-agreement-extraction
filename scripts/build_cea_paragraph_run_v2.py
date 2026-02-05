#!/usr/bin/env python3
"""
Build a minimal v2 run from the CEA (credit_agreement_analysis_*) covenant spreadsheets.

Goal
----
Convert rows from the CEA "Covenants" sheet into a standard runs/<run_id>/ layout so we can:
  - run prompt-based extraction via `pipeline.run structured-v2`
  - keep audit artifacts (manifest.json, canonical.txt, retrieval_v2 snippets)

This script does NOT run any LLM calls. It only builds run artifacts.

Inputs (CEA formats observed)
----------------------------
- .xlsx with a sheet named "Covenants" containing at least:
  - file (string path, typically ending in *.nc)
  - segment_no (int)
  - para (string snippet/paragraph text)
  - financial_covenant (string, may be null)
  - value (number, may be null)
  - cov_start_date / cov_end_date (optional; may be null)

Output
------
runs/<run_id>/
  manifest.json
  normalized/<item_id>/canonical.txt
  normalized/<item_id>/anchors.tsv
  retrieval_v2/<item_id>_snippets.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import time
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _norm_str(v: Any) -> str:
    # pandas uses NaN/NaT sentinels; treat them as empty.
    try:
        if pd.isna(v):
            return ""
    except Exception:
        pass
    if v is None:
        return ""
    if isinstance(v, str):
        return v
    return str(v)


def _parse_accession_from_file_path(file_path: str) -> str:
    name = Path(file_path).name
    if name.endswith(".nc"):
        name = name[: -len(".nc")]
    # Keep only safe chars.
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("_")
    return name or "unknown"


def _safe_token(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", s).strip("_")


def _anchors_tsv_single(*, anchor_id: str, text_len: int) -> str:
    return "anchor_id\tanchor_type\tstart\tend\tlabel\n" + f"{anchor_id}\tsentence\t0\t{text_len}\t{anchor_id}\n"


def _snippet_record(*, item_id: str, anchor_id: str, snippet: str, category: str) -> Dict[str, Any]:
    return {
        "item_id": item_id,
        "anchor_id": anchor_id,
        "categories": [category],
        "label": category,
        "type": "sentence",
        "start": 0,
        "end": len(snippet),
        "buckets": [category],
        "toc_chunk_id": None,
        "toc_title": None,
        "snippet": snippet,
        "snippet_start": 0,
        "snippet_end": len(snippet),
    }


def _parse_decimal_like(value: str) -> Optional[Decimal]:
    s = _norm_str(value)
    if not s:
        return None
    s = s.replace(",", "")
    s = re.sub(r"^\$", "", s)
    m = re.match(r"^(-?\d+(?:\.\d+)?)", s)
    if not m:
        return None
    try:
        return Decimal(m.group(1))
    except (InvalidOperation, ValueError):
        return None


@dataclass(frozen=True)
class CeaRow:
    row_idx: int
    file_path: str
    segment_no: str
    para: str
    financial_covenant: str
    value: str
    cov_start_date: str
    cov_end_date: str


def _read_cea_rows(
    *,
    xlsx_path: Path,
    sheet: str,
    text_col: str,
    include_missing_baseline: bool,
) -> List[CeaRow]:
    df = pd.read_excel(xlsx_path, sheet_name=sheet)

    required = {"file", "segment_no", text_col}
    missing = sorted(required - set(df.columns))
    if missing:
        raise SystemExit(
            f"Missing required column(s) in {xlsx_path} sheet={sheet}: {', '.join(missing)} "
            f"(available={', '.join(map(str, df.columns))})"
        )

    rows: List[CeaRow] = []
    for idx, r in df.iterrows():
        para_raw = r.get(text_col)
        para = _norm_str(para_raw)
        if not para.strip():
            continue

        baseline_name = _norm_str(r.get("financial_covenant"))
        baseline_value = _norm_str(r.get("value"))

        if not include_missing_baseline:
            if not baseline_name.strip():
                continue
            if not baseline_value.strip():
                continue
            if _parse_decimal_like(baseline_value) is None:
                continue

        rows.append(
            CeaRow(
                row_idx=int(idx),
                file_path=_norm_str(r.get("file")),
                segment_no=_norm_str(r.get("segment_no")),
                para=para,
                financial_covenant=baseline_name,
                value=baseline_value,
                cov_start_date=_norm_str(r.get("cov_start_date")),
                cov_end_date=_norm_str(r.get("cov_end_date")),
            )
        )
    return rows


def build_run_from_cea(
    *,
    xlsx_path: Path,
    sheet: str,
    out_run_dir: Path,
    text_col: str,
    category: str,
    limit: Optional[int],
    overwrite: bool,
    include_missing_baseline: bool,
) -> Dict[str, Any]:
    if out_run_dir.exists():
        if not overwrite:
            raise SystemExit(f"Refusing to overwrite existing run dir: {out_run_dir} (pass --overwrite)")
        shutil.rmtree(out_run_dir)

    out_run_dir.mkdir(parents=True, exist_ok=True)

    rows = _read_cea_rows(
        xlsx_path=xlsx_path,
        sheet=sheet,
        text_col=text_col,
        include_missing_baseline=include_missing_baseline,
    )
    if limit is not None:
        rows = rows[: max(0, int(limit))]

    if not rows:
        raise SystemExit(f"No rows with non-empty {text_col!r} found in {xlsx_path} sheet={sheet}.")

    normalized_dir = out_run_dir / "normalized"
    retrieval_dir = out_run_dir / "retrieval_v2"

    items: List[Dict[str, Any]] = []
    seen: set[str] = set()

    for row in rows:
        accession = _parse_accession_from_file_path(row.file_path) if row.file_path else "unknown"
        seg = _safe_token(row.segment_no) or "seg"
        rid = f"r{row.row_idx:05d}"
        item_id = f"{accession}_{seg}_{rid}"

        if item_id in seen:
            raise SystemExit(f"Duplicate derived item_id {item_id!r}; unexpected collision.")
        seen.add(item_id)

        # Canonical text for audit (single-anchor).
        _write_text(normalized_dir / item_id / "canonical.txt", row.para)
        _write_text(normalized_dir / item_id / "anchors.tsv", _anchors_tsv_single(anchor_id="A0001", text_len=len(row.para)))

        # retrieval_v2 snippet pack: one record that spans the whole paragraph.
        rec = _snippet_record(item_id=item_id, anchor_id="A0001", snippet=row.para, category=category)
        out_path = retrieval_dir / f"{item_id}_snippets.jsonl"
        _write_text(out_path, json.dumps(rec, ensure_ascii=False) + "\n")

        items.append(
            {
                "item_id": item_id,
                "source": {
                    "cea_xlsx": str(xlsx_path),
                    "sheet": sheet,
                    "row_index": row.row_idx,
                    "file": row.file_path,
                    "segment_no": row.segment_no,
                    "text_col": text_col,
                },
                "baseline": {
                    "name": row.financial_covenant or None,
                    "limit": row.value or None,
                    "start_date": row.cov_start_date or None,
                    "end_date": row.cov_end_date or None,
                },
            }
        )

    manifest: Dict[str, Any] = {
        "schema_version": "cea_paragraph_run_v1",
        "created_at": int(time.time()),
        "source": {
            "xlsx_path": str(xlsx_path),
            "sheet": sheet,
            "text_col": text_col,
            "category": category,
        },
        "items": items,
    }
    _write_json(out_run_dir / "manifest.json", manifest)
    return manifest


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--xlsx", required=True, help="Path to credit_agreement_analysis_*.xlsx")
    ap.add_argument("--sheet", default="Covenants")
    ap.add_argument("--run-id", required=True, help="Output under runs/<run-id>/")
    ap.add_argument("--base-dir", default=".", help="Repo base dir (writes under <base-dir>/runs/<run-id>/)")
    ap.add_argument("--text-col", default="para", help="Column containing the paragraph text.")
    ap.add_argument("--category", default="financial_covenant", help="Category tag for retrieval_v2 snippet records.")
    ap.add_argument("--limit", type=int, default=None, help="Optional max number of rows to include (for quick tests).")
    ap.add_argument(
        "--include-missing-baseline",
        action="store_true",
        help="Include rows that are missing CEA baseline fields (financial_covenant/value). Default behavior filters them out.",
    )
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    base_dir = Path(args.base_dir)
    out_run_dir = base_dir / "runs" / str(args.run_id)
    manifest = build_run_from_cea(
        xlsx_path=Path(args.xlsx),
        sheet=str(args.sheet),
        out_run_dir=out_run_dir,
        text_col=str(args.text_col),
        category=str(args.category),
        limit=args.limit,
        overwrite=bool(args.overwrite),
        include_missing_baseline=bool(args.include_missing_baseline),
    )
    print(f"Wrote run dir: {out_run_dir}")
    print(f"Items: {len(manifest.get('items') or [])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
