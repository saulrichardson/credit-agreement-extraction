from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .anchors import load_anchor_catalog
from .config import Paths
from .utils import assert_exists


@dataclass(frozen=True)
class AnchorBlock:
    anchor_id: str
    anchor_type: str
    order: int
    start: int | None
    end: int | None
    text: str


def _parse_canonical_annotated(text: str) -> list[tuple[str, str]]:
    """Parse canonical_annotated.txt into ordered (anchor_id, block_text) pairs."""

    marker_re = re.compile(r"^\[\[(A\d{4,})\]\]\s*$")
    blocks: list[tuple[str, str]] = []

    current_id: str | None = None
    current_lines: list[str] = []

    for line in text.splitlines():
        m = marker_re.match(line)
        if m:
            if current_id is not None:
                blocks.append((current_id, "\n".join(current_lines).strip("\n")))
            current_id = m.group(1)
            current_lines = []
            continue
        current_lines.append(line)

    if current_id is not None:
        blocks.append((current_id, "\n".join(current_lines).strip("\n")))

    if not blocks:
        raise RuntimeError("No anchors found in canonical_annotated.txt (missing [[A####]] markers?)")
    return blocks


def _extract_table_payload(block_text: str) -> str | None:
    m = re.search(r"\[\[TABLE\]\]\s*(.*?)\s*\[\[/TABLE\]\]", block_text, flags=re.DOTALL)
    if not m:
        return None
    return m.group(1).strip()


def _is_md_separator_row(row: list[str]) -> bool:
    if not row:
        return False
    return all(re.fullmatch(r"-{3,}", cell.strip()) for cell in row if cell.strip())


def _parse_markdown_table(table_text: str) -> list[list[str]] | None:
    """Parse a markdown table into a 2D matrix of cell strings.

    Returns None when the payload doesn't look like a markdown table.
    """

    lines = [ln.rstrip("\n") for ln in table_text.splitlines() if ln.strip()]
    md_lines = [ln for ln in lines if "|" in ln]
    if len(md_lines) < 2:
        return None

    matrix: list[list[str]] = []
    for ln in md_lines:
        if not ln.strip().startswith("|"):
            continue
        # Keep empty interior cells; strip outer pipes.
        raw_cells = ln.strip().strip("|").split("|")
        cells = [c.strip() for c in raw_cells]
        if not cells:
            continue
        matrix.append(cells)

    if not matrix:
        return None

    # Drop the markdown separator row if present as the second row.
    if len(matrix) >= 2 and _is_md_separator_row(matrix[1]):
        matrix = [matrix[0]] + matrix[2:]

    return matrix or None


def _drop_empty_columns(matrix: list[list[str]]) -> list[list[str]]:
    if not matrix:
        return matrix
    ncols = max(len(r) for r in matrix)
    padded = [r + [""] * (ncols - len(r)) for r in matrix]
    keep: list[int] = []
    for j in range(ncols):
        col_vals = [padded[i][j].strip() for i in range(len(padded))]
        if any(v for v in col_vals):
            keep.append(j)
    if not keep:
        return matrix
    squeezed: list[list[str]] = []
    for r in padded:
        squeezed.append([r[j] for j in keep])
    return squeezed


def _drop_empty_rows(matrix: list[list[str]]) -> list[list[str]]:
    out: list[list[str]] = []
    for r in matrix:
        if any(c.strip() for c in r):
            out.append(r)
    return out


def parse_table(block_text: str) -> dict[str, Any] | None:
    """Parse the first [[TABLE]] ... [[/TABLE]] block inside an anchor block."""

    payload = _extract_table_payload(block_text)
    if payload is None:
        return None

    matrix = _parse_markdown_table(payload)
    if matrix is None:
        # Preserve raw payload even when we cannot structure it.
        return {
            "raw": payload,
            "columns": [],
            "rows": [],
            "structured": False,
        }

    matrix = _drop_empty_rows(_drop_empty_columns(matrix))
    if not matrix:
        return {
            "raw": payload,
            "columns": [],
            "rows": [],
            "structured": False,
        }

    header = matrix[0]
    columns = [c.strip() for c in header]
    if columns and not columns[0]:
        columns[0] = "ROW_LABEL"

    rows_out: list[dict[str, Any]] = []
    for r in matrix[1:]:
        row = r + [""] * (len(columns) - len(r))
        row_label = row[0].strip()
        cells: dict[str, str] = {}
        for idx in range(1, len(columns)):
            col = columns[idx] or f"COL_{idx}"
            cells[col] = row[idx].strip()
        if not row_label and not any(v for v in cells.values()):
            continue
        rows_out.append({"row_label": row_label, "cells": cells})

    return {
        "raw": payload,
        "columns": columns,
        "rows": rows_out,
        "structured": True,
    }


def build_doc_ir(paths: Paths, item_id: str) -> dict[str, Any]:
    annotated_path = assert_exists(
        paths.normalized_dir / item_id / "canonical_annotated.txt",
        message=f"Missing canonical_annotated.txt for {item_id}: run normalize first.",
    )
    catalog = load_anchor_catalog(paths, item_id)
    pairs = _parse_canonical_annotated(annotated_path.read_text())

    anchors: list[dict[str, Any]] = []
    tables: list[dict[str, Any]] = []

    for order, (anchor_id, block_text) in enumerate(pairs):
        info = catalog.get(anchor_id) or {}
        anchor_type = str(info.get("anchor_type") or "unknown")
        start = info.get("start")
        end = info.get("end")
        anchor_rec = {
            "anchor_id": anchor_id,
            "anchor_type": anchor_type,
            "order": order,
            "start": start,
            "end": end,
            "text": block_text.strip(),
        }
        anchors.append(anchor_rec)

        if anchor_type == "table":
            parsed = parse_table(block_text) or {"raw": "", "columns": [], "rows": [], "structured": False}
            tables.append(
                {
                    "table_anchor_id": anchor_id,
                    "order": order,
                    **parsed,
                }
            )

    return {"item_id": item_id, "anchors": anchors, "tables": tables}


def write_doc_ir(out_path: Path, doc_ir: dict[str, Any]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(doc_ir, indent=2))


def load_doc_ir(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def iter_anchor_text(doc_ir: dict[str, Any]) -> Iterable[tuple[str, str]]:
    for a in doc_ir.get("anchors", []):
        yield str(a.get("anchor_id")), str(a.get("text") or "")

