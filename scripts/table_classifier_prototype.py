#!/usr/bin/env python3
"""Prototype table classifier / header inference for EDGAR segments.

Usage:
    python scripts/table_classifier_prototype.py path/to/file.html [...]

This does NOT modify the main pipeline. It prints diagnostic output so we can
evaluate heuristics before wiring them into CanonicalDocumentBuilder.
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

from bs4 import BeautifulSoup, Tag


PLACEHOLDER_GLYPHS = {"■", "¨", "□", "☐", "&nbsp;", "\xa0"}


@dataclass
class TableSummary:
    index: int
    classification: str
    num_rows: int
    num_cols: int
    placeholder_ratio: float
    avg_cell_len: float
    header_pairs: list[str]


def is_placeholder(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return True
    if stripped in PLACEHOLDER_GLYPHS:
        return True
    if all(ch in {"·", ".", "-"} for ch in stripped):
        return True
    return False


def normalize_text(text: str) -> str:
    return " ".join(text.split())


def iter_rows(table: Tag) -> Iterable[List[str]]:
    for row in table.find_all("tr", recursive=True):
        cells = row.find_all(["td", "th"], recursive=False)
        if not cells:
            cells = row.find_all(["td", "th"])
        values = [normalize_text(cell.get_text(separator=" ", strip=True)) for cell in cells]
        if any(values):
            yield values


def classify_table(rows: List[List[str]]) -> str:
    if not rows:
        return "empty"
    num_cols = max(len(row) for row in rows) or 0
    cell_lengths = [len(cell) for row in rows for cell in row if cell]
    avg_len = sum(cell_lengths) / len(cell_lengths) if cell_lengths else 0.0
    total_cells = sum(len(row) for row in rows)
    placeholder_cells = sum(1 for row in rows for cell in row if is_placeholder(cell))
    placeholder_ratio = placeholder_cells / total_cells if total_cells else 0.0
    contains_numeric = any(re.search(r"\d", cell) for row in rows for cell in row)
    contains_th = any("<th" in str(row).lower() for row in rows)

    if num_cols <= 2 and (avg_len > 60 or placeholder_ratio > 0.3):
        return "layout_prose"
    if num_cols >= 3 and (contains_numeric or contains_th):
        return "data"
    if placeholder_ratio > 0.5:
        return "layout_prose"
    return "unknown"


def classify_rows(rows: List[List[str]]) -> List[str]:
    labels: List[str] = []
    for row in rows:
        tokens = [cell for cell in row if cell]
        if not tokens:
            labels.append("value")
            continue
        ends_with_colon = all(cell.endswith(":") for cell in tokens)
        has_parentheses = any("(" in cell and ")" in cell for cell in tokens)
        avg_len = sum(len(cell) for cell in tokens) / len(tokens)
        mostly_numeric = sum(bool(re.search(r"\d", cell)) for cell in tokens) >= len(tokens) / 2
        if ends_with_colon or has_parentheses:
            labels.append("header")
        elif avg_len > 60 and not mostly_numeric:
            labels.append("header")
        else:
            labels.append("value")
    return labels


def infer_headers(rows: List[List[str]], labels: List[str]) -> list[str]:
    results: list[str] = []
    pending_headers: list[str] = []
    for row, label in zip(rows, labels, strict=False):
        if label == "header":
            header_text = " | ".join(cell for cell in row if cell)
            pending_headers.append(header_text)
            continue
        value_cells = [cell for cell in row if cell and not is_placeholder(cell)]
        header = " | ".join(pending_headers) if pending_headers else ""
        value = " | ".join(value_cells)
        if header or value:
            results.append(f"{header}: {value}".strip(": "))
        pending_headers = []
    return results


def analyze_file(path: Path) -> list[TableSummary]:
    soup = BeautifulSoup(path.read_text(encoding="utf-8"), "lxml")
    summaries: list[TableSummary] = []
    for idx, table in enumerate(soup.find_all("table"), start=1):
        rows = list(iter_rows(table))
        classification = classify_table(rows)
        labels = classify_rows(rows)
        headers = infer_headers(rows, labels) if classification == "data" else []
        num_rows = len(rows)
        num_cols = max((len(row) for row in rows), default=0)
        total_cells = sum(len(row) for row in rows)
        placeholder_cells = sum(1 for row in rows for cell in row if is_placeholder(cell))
        cell_lengths = [len(cell) for row in rows for cell in row if cell]
        avg_len = sum(cell_lengths) / len(cell_lengths) if cell_lengths else 0.0
        placeholder_ratio = placeholder_cells / total_cells if total_cells else 0.0
        summaries.append(
            TableSummary(
                index=idx,
                classification=classification,
                num_rows=num_rows,
                num_cols=num_cols,
                placeholder_ratio=placeholder_ratio,
                avg_cell_len=avg_len,
                header_pairs=headers,
            )
        )
    return summaries


def main() -> None:
    parser = argparse.ArgumentParser(description="Prototype table classifier for EDGAR samples.")
    parser.add_argument("files", nargs="+", type=Path)
    args = parser.parse_args()
    for path in args.files:
        print(f"=== {path} ===")
        summaries = analyze_file(path)
        if not summaries:
            print("  (no tables)")
            continue
        for summary in summaries[:5]:
            print(
                f"  Table {summary.index}: class={summary.classification} "
                f"rows={summary.num_rows} cols={summary.num_cols} "
                f"placeholder={summary.placeholder_ratio:.2f} avg_len={summary.avg_cell_len:.1f}"
            )
            for header in summary.header_pairs[:3]:
                print(f"    header/value: {header}")
        if len(summaries) > 5:
            print(f"  ... {len(summaries) - 5} more tables")


if __name__ == "__main__":
    main()
