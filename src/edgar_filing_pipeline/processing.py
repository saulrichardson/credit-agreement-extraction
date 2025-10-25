from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

from bs4 import BeautifulSoup
from tabulate import tabulate

SGML_WRAPPER_TAGS = {
    "s",
    "c",
    "sec-header",
    "sec-body",
    "page",
    "sec-document",
    "sec-issuer",
}

WHITESPACE_RE = re.compile(r"\s+")
MULTI_NEWLINE_RE = re.compile(r"\n{3,}")


@dataclass
class NormalizedTables:
    markers: List[str]
    markdown: List[str]
    html: List[str]


@dataclass
class NormalizedSegment:
    text: str
    tables: NormalizedTables


def _clean_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = MULTI_NEWLINE_RE.sub("\n\n", text)
    lines = [WHITESPACE_RE.sub(" ", line).strip() for line in text.split("\n")]
    cleaned = "\n".join(line for line in lines if line is not None)
    return cleaned.strip()


def _table_to_markdown(table) -> str:
    rows: List[List[str]] = []
    header: List[str] | None = None

    for tr in table.find_all("tr"):
        cells = tr.find_all(["th", "td"])
        if not cells:
            continue

        cell_text = [
            cell.get_text(separator=" ", strip=True).replace("\xa0", " ")
            for cell in cells
        ]

        if any(cell.name == "th" for cell in cells) and header is None:
            header = cell_text
            continue

        rows.append(cell_text)

    if not rows and not header:
        return ""

    width = max(len(header) if header else 0, *(len(row) for row in rows)) if (rows or header) else 0
    if header and len(header) < width:
        header = header + [""] * (width - len(header))

    normalized_rows = [
        row + [""] * (width - len(row))
        for row in rows
    ]

    if header:
        return tabulate(normalized_rows, headers=header, tablefmt="github")
    return tabulate(normalized_rows, tablefmt="github")


def normalize_html(html: str) -> NormalizedSegment:
    soup = BeautifulSoup(html, "lxml")

    for tag_name in SGML_WRAPPER_TAGS:
        for node in soup.find_all(tag_name):
            node.unwrap()

    table_markers: List[str] = []
    table_markdown: List[str] = []
    table_html: List[str] = []

    tables = soup.find_all("table")
    for idx, table in enumerate(tables):
        marker = f"__TABLE_{idx + 1}__"
        markdown = _table_to_markdown(table)
        table_markers.append(marker)
        table_markdown.append(markdown)
        table_html.append(str(table))
        table.replace_with(soup.new_string(f"\n{marker}\n"))

    text = soup.get_text(separator="\n")
    cleaned_text = _clean_text(text)

    return NormalizedSegment(
        text=cleaned_text,
        tables=NormalizedTables(
            markers=table_markers,
            markdown=table_markdown,
            html=table_html,
        ),
    )


def build_checksum(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def tables_to_json_dict(tables: NormalizedTables) -> Tuple[str, str]:
    markdown_payload = [
        {"marker": marker, "markdown": markdown}
        for marker, markdown in zip(tables.markers, tables.markdown)
    ]
    html_payload = [
        {"marker": marker, "html": html}
        for marker, html in zip(tables.markers, tables.html)
    ]
    return json.dumps(markdown_payload), json.dumps(html_payload)
