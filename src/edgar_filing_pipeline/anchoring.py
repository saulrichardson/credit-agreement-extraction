from __future__ import annotations

import csv
import hashlib
import html
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List, Optional

from bs4 import BeautifulSoup, NavigableString, Tag

BLOCK_LEVEL_TAGS = {
    "p",
    "div",
    "section",
    "article",
    "li",
    "ul",
    "ol",
    "table",
    "tr",
    "td",
    "th",
    "h1",
    "h2",
    "h3",
    "h4",
    "h5",
    "h6",
    "br",
    "hr",
    "pre",
}

ABBREVIATIONS = {
    "mr.",
    "mrs.",
    "ms.",
    "dr.",
    "prof.",
    "sr.",
    "jr.",
    "inc.",
    "ltd.",
    "co.",
    "corp.",
    "u.s.",
    "u.k.",
    "no.",
    "fig.",
    "art.",
    "sec.",
    "ch.",
    "dept.",
    "assn.",
    "bros.",
    "st.",
    "viz.",
    "vs.",
    "etc.",
    "i.e.",
    "e.g.",
}

ANCHOR_KIND_SENTENCE = "s"
ANCHOR_KIND_DEFINITION = "d"
ANCHOR_KIND_TABLE_CELL = "t"
ANCHOR_KIND_PARAGRAPH = "p"
ANCHOR_KIND_HEADING = "h"

PLACEHOLDER_GLYPHS = {"■", "¨", "□", "☐", "\xa0", "&nbsp;", "•"}


class MachineContentError(RuntimeError):
    """Raised when the segment is classified as machine-readable payload (XML/XBRL/etc.)."""


@dataclass
class ParagraphSegment:
    start: int
    end: int
    text: str
    is_heading: bool


@dataclass
class TableCellAnchor:
    anchor_id: str
    start: int
    end: int
    table_index: int
    row_index: int
    col_index: int
    row_header: str | None
    col_header: str | None
    text: str
    parent_heading: str | None


@dataclass
class AnchorRecord:
    anchor_id: str
    kind: str
    start: int
    end: int
    checksum: str
    context_pre: str
    context_post: str
    prev_id: str | None = None
    next_id: str | None = None
    parent_heading: str | None = None
    attributes: dict[str, str] = field(default_factory=dict)

    def to_tsv_row(self) -> dict[str, str]:
        payload = {
            "anchor_id": self.anchor_id,
            "kind": self.kind,
            "start": str(self.start),
            "end": str(self.end),
            "checksum": self.checksum,
            "context_pre": self.context_pre,
            "context_post": self.context_post,
            "prev_id": self.prev_id or "",
            "next_id": self.next_id or "",
            "parent_heading": self.parent_heading or "",
            "attributes_json": json.dumps(self.attributes, separators=(",", ":")),
        }
        return payload


@dataclass
class CanonicalDocument:
    text: str
    anchors: list[AnchorRecord]


class CanonicalDocumentBuilder:
    def __init__(self, *, html_text: str, source_id: str = "doc0") -> None:
        if not html_text:
            raise ValueError("html_text must be non-empty")
        self.html_text = html_text
        self.source_id = source_id
        if self._looks_like_machine_payload(html_text):
            raise MachineContentError("Segment classified as machine-readable payload; skipping.")
        self._soup = BeautifulSoup(html_text, "lxml")
        for node in self._soup(["script", "style"]):
            node.decompose()

    def build(self) -> CanonicalDocument:
        chunks: list[str] = []
        paragraph_segments: list[ParagraphSegment] = []
        table_cell_anchors: list[TableCellAnchor] = []

        offset = 0
        active_heading_id: str | None = None
        table_index = 0

        def append_text(text: str, is_heading: bool = False) -> ParagraphSegment | None:
            nonlocal offset, active_heading_id
            normalized = self._normalize_whitespace(text)
            if not normalized:
                return None
            if chunks and not chunks[-1].endswith("\n"):
                chunks.append("\n")
                offset += 1
            start = offset
            chunks.append(normalized)
            offset += len(normalized)
            chunks.append("\n")
            offset += 1
            segment = ParagraphSegment(
                start=start,
                end=start + len(normalized),
                text=normalized,
                is_heading=is_heading,
            )
            paragraph_segments.append(segment)
            if is_heading:
                active_heading_id = None  # will be assigned once headings processed
            return segment

        for element in self._iter_blocks(self._soup.body or self._soup):
            if isinstance(element, str):
                append_text(element, is_heading=False)
                continue
            if element.name == "table":
                cell_anchors, text_block = self._process_table(
                    table_tag=element,
                    table_index=table_index,
                    current_offset=offset,
                    active_heading_id=active_heading_id,
                )
                if text_block:
                    if chunks and not chunks[-1].endswith("\n"):
                        chunks.append("\n")
                        offset += 1
                    start = offset
                    chunks.append(text_block)
                    offset += len(text_block)
                    chunks.append("\n")
                    offset += 1
                    paragraph_segments.append(
                        ParagraphSegment(
                            start=start,
                            end=start + len(text_block),
                            text=text_block,
                            is_heading=False,
                        )
                    )
                for cell_anchor in cell_anchors:
                    table_cell_anchors.append(cell_anchor)
                table_index += 1
                continue
            append_text(element.get_text(separator=" "))

        canonical_text = "".join(chunks).rstrip("\n")
        paragraph_segments = [seg for seg in paragraph_segments if seg.end > seg.start]
        if not paragraph_segments and not table_cell_anchors:
            raise ValueError("No textual content detected; cannot build anchors.")

        paragraph_records = self._build_paragraph_anchors(
            canonical_text=canonical_text,
            paragraphs=paragraph_segments,
        )
        heading_ids = {
            record.anchor_id: record
            for record in paragraph_records
            if record.kind == ANCHOR_KIND_HEADING
        }
        sentence_records = self._build_sentence_anchors(
            canonical_text=canonical_text,
            paragraphs=paragraph_segments,
            heading_records=heading_ids,
        )
        table_records = self._build_table_records(canonical_text, table_cell_anchors)

        anchors = paragraph_records + sentence_records + table_records
        anchors.sort(key=lambda rec: rec.start)
        self._link_neighbor_ids(anchors)
        self._assign_parent_heading(anchors)

        return CanonicalDocument(text=canonical_text, anchors=anchors)

    def _iter_blocks(self, root: Tag) -> Iterable[Tag | str]:
        for child in root.children:
            if isinstance(child, NavigableString):
                text = self._normalize_whitespace(str(child))
                if text:
                    yield text
                continue
            if not isinstance(child, Tag):
                continue
            if child.name in {"script", "style"}:
                continue
            if child.name == "table":
                yield child
                continue
            if child.get_text(strip=True) == "":
                continue
            if child.name in BLOCK_LEVEL_TAGS:
                yield child
            else:
                text = child.get_text(separator=" ")
                normalized = self._normalize_whitespace(text)
                if normalized:
                    yield normalized

    def _normalize_whitespace(self, text: str) -> str:
        decoded = html.unescape(text)
        collapsed = re.sub(r"[ \t\r\f\v]+", " ", decoded)
        collapsed = re.sub(r"\n{2,}", "\n", collapsed)
        cleaned = collapsed.strip()
        return cleaned

    def _looks_like_machine_payload(self, text: str) -> bool:
        stripped = text.lstrip()
        prefix = stripped[:512].lower()
        if stripped.startswith("<?xml") or stripped.lower().startswith("<xbrl"):
            return True
        if prefix.startswith("<xbrli:") or "<xbrli:" in prefix:
            return True
        if "begin 644" in stripped[:200]:
            return True
        if stripped.startswith("%PDF"):
            return True
        return False

    def _clean_cell_text(self, text: str) -> str:
        if text is None:
            return ""
        cleaned = text.replace("\xa0", " ").strip()
        if not cleaned:
            return ""
        if cleaned in PLACEHOLDER_GLYPHS:
            return ""
        if len(cleaned) <= 3 and all(ch in PLACEHOLDER_GLYPHS for ch in cleaned):
            return ""
        if cleaned in {"—", "-", "–"}:
            return ""
        return cleaned

    def _is_placeholder_cell(self, text: str) -> bool:
        if text is None:
            return True
        stripped = text.replace("\xa0", " ").strip()
        if not stripped:
            return True
        if stripped in PLACEHOLDER_GLYPHS:
            return True
        if len(stripped) <= 3 and all(ch in PLACEHOLDER_GLYPHS for ch in stripped):
            return True
        return False

    def _classify_structured_table(self, grid: List[List[str]]) -> str:
        if not grid:
            return "empty"
        num_cols = max(len(row) for row in grid) if grid else 0
        total_cells = sum(len(row) for row in grid)
        placeholder_cells = sum(1 for row in grid for cell in row if self._is_placeholder_cell(cell))
        placeholder_ratio = placeholder_cells / total_cells if total_cells else 0.0
        cell_lengths = [len(cell) for row in grid for cell in row if cell]
        avg_len = sum(cell_lengths) / len(cell_lengths) if cell_lengths else 0.0
        contains_numeric = any(re.search(r"\d", cell) for row in grid for cell in row if cell)
        if num_cols <= 2 and (avg_len > 60 or placeholder_ratio > 0.3):
            return "layout_prose"
        if num_cols >= 3 and contains_numeric:
            return "data"
        if placeholder_ratio > 0.5:
            return "layout_prose"
        return "unknown"

    def _serialize_layout_table(self, grid: List[List[str]]) -> str:
        lines: list[str] = []
        for row in grid:
            cleaned_cells = [self._clean_cell_text(cell) for cell in row if self._clean_cell_text(cell)]
            if not cleaned_cells:
                continue
            if len(cleaned_cells) >= 2:
                label = cleaned_cells[0]
                value = " ".join(cleaned_cells[1:]).strip()
                if value and len(label) <= 120:
                    lines.append(f"{label}: {value}")
                else:
                    lines.append(" ".join(cleaned_cells))
            else:
                lines.append(cleaned_cells[0])
        return "\n".join(lines)

    def _markdown_escape(self, text: str) -> str:
        return text.replace("|", "\\|").replace("\n", " ")

    def _render_markdown_table(self, grid: List[List[str]]) -> str:
        if not grid:
            return ""
        grid = [row for row in grid if any(cell for cell in row)]
        if not grid:
            return ""
        headers = [self._markdown_escape(cell) if cell else f"Column {idx + 1}"
                   for idx, cell in enumerate(grid[0])]
        data_rows: list[list[str]] = []
        for row in grid[1:]:
            cleaned = [self._markdown_escape(cell) for cell in row]
            if any(cell for cell in cleaned):
                data_rows.append(cleaned)
        if not data_rows:
            return ""
        header_line = "| " + " | ".join(headers) + " |"
        separator_line = "| " + " | ".join("---" for _ in headers) + " |"
        row_lines = ["| " + " | ".join(row) + " |" for row in data_rows]
        return "\n".join([header_line, separator_line, *row_lines])

    def _build_paragraph_anchors(
        self,
        *,
        canonical_text: str,
        paragraphs: List[ParagraphSegment],
    ) -> List[AnchorRecord]:
        anchors: list[AnchorRecord] = []
        counter = 0
        for segment in paragraphs:
            counter += 1
            anchor_id = f"p{counter:06d}"
            kind = ANCHOR_KIND_HEADING if self._is_heading(segment.text) else ANCHOR_KIND_PARAGRAPH
            checksum = self._hash_text(segment.text)
            ctx_pre, ctx_post = self._context_hashes(canonical_text, segment.start, segment.end)
            anchors.append(
                AnchorRecord(
                    anchor_id=anchor_id,
                    kind=kind,
                    start=segment.start,
                    end=segment.end,
                    checksum=checksum,
                    context_pre=ctx_pre,
                    context_post=ctx_post,
                )
            )
        return anchors

    def _build_sentence_anchors(
        self,
        *,
        canonical_text: str,
        paragraphs: List[ParagraphSegment],
        heading_records: dict[str, AnchorRecord],
    ) -> List[AnchorRecord]:
        anchors: list[AnchorRecord] = []
        sentence_counter = 0
        active_heading_id: str | None = None
        heading_iter = sorted(heading_records.values(), key=lambda rec: rec.start)
        heading_idx = 0

        for paragraph in paragraphs:
            while heading_idx < len(heading_iter) and heading_iter[heading_idx].end <= paragraph.start:
                heading_idx += 1
            if heading_idx < len(heading_iter) and heading_iter[heading_idx].start <= paragraph.start:
                active_heading_id = heading_iter[heading_idx].anchor_id

            sentences = self._split_sentences(paragraph.text, base_offset=paragraph.start)
            for start, end in sentences:
                snippet = canonical_text[start:end].strip()
                if not snippet:
                    continue
                if all(ch in {'.', '*', '•', '·', ' ', '\u00a0'} for ch in snippet):
                    continue
                sentence_counter += 1
                anchor_id = f"s{sentence_counter:06d}"
                kind = (
                    ANCHOR_KIND_DEFINITION
                    if self._looks_like_definition(snippet)
                    else ANCHOR_KIND_SENTENCE
                )
                checksum = self._hash_text(snippet)
                ctx_pre, ctx_post = self._context_hashes(canonical_text, start, end)
                anchors.append(
                    AnchorRecord(
                        anchor_id=anchor_id,
                        kind=kind,
                        start=start,
                        end=end,
                        checksum=checksum,
                        context_pre=ctx_pre,
                        context_post=ctx_post,
                        parent_heading=active_heading_id,
                    )
                )
        return anchors

    def _build_table_records(
        self,
        canonical_text: str,
        table_cell_anchors: List[TableCellAnchor],
    ) -> List[AnchorRecord]:
        anchors: list[AnchorRecord] = []
        for cell in table_cell_anchors:
            snippet = canonical_text[cell.start:cell.end]
            checksum = self._hash_text(snippet)
            ctx_pre, ctx_post = self._context_hashes(canonical_text, cell.start, cell.end)
            attributes = {
                "table_index": str(cell.table_index),
                "row_index": str(cell.row_index),
                "col_index": str(cell.col_index),
            }
            if cell.row_header:
                attributes["row_header"] = cell.row_header
            if cell.col_header:
                attributes["col_header"] = cell.col_header
            anchors.append(
                AnchorRecord(
                    anchor_id=cell.anchor_id,
                    kind=ANCHOR_KIND_TABLE_CELL,
                    start=cell.start,
                    end=cell.end,
                    checksum=checksum,
                    context_pre=ctx_pre,
                    context_post=ctx_post,
                    parent_heading=cell.parent_heading,
                    attributes=attributes,
                )
            )
        return anchors

    def _hash_text(self, text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def _context_hashes(self, text: str, start: int, end: int, window: int = 80) -> tuple[str, str]:
        pre_start = max(0, start - window)
        post_end = min(len(text), end + window)
        pre = text[pre_start:start]
        post = text[end:post_end]
        pre_hash = hashlib.sha256(pre.encode("utf-8")).hexdigest()
        post_hash = hashlib.sha256(post.encode("utf-8")).hexdigest()
        return pre_hash, post_hash

    def _is_heading(self, text: str) -> bool:
        stripped = text.strip()
        if not stripped:
            return False
        if len(stripped) <= 140 and stripped.isupper():
            return True
        if stripped.startswith("Section ") or stripped.startswith("ARTICLE "):
            return True
        if stripped.endswith(":") and len(stripped.split()) <= 10:
            return True
        return False

    def _looks_like_definition(self, text: str) -> bool:
        return bool(re.search(r'\"[^"]+\"\s+means\s+', text))

    def _split_sentences(self, text: str, *, base_offset: int) -> List[tuple[int, int]]:
        stripped = text.lstrip()
        if stripped.startswith("|") and "| ---" in stripped:
            start_offset = base_offset + (len(text) - len(stripped))
            return [(start_offset, base_offset + len(text))]

        sentences: list[tuple[int, int]] = []
        start = 0
        i = 0
        length = len(text)

        while i < length:
            char = text[i]
            if char in {".", "?", "!"}:
                next_char = text[i + 1] if i + 1 < length else ""
                next_non_space = self._next_non_space_char(text, i + 1)
                snippet = text[start : i + 1]
                if snippet:
                    token = snippet.split()[-1].lower()
                    if token in ABBREVIATIONS:
                        if next_non_space and (next_non_space.isalnum() or next_non_space == "_"):
                            i += 1
                            continue
                if next_char and next_char.islower():
                    i += 1
                    continue
                end = i + 1
                trimmed = text[start:end].strip()
                if trimmed:
                    sentences.append((base_offset + start + (len(text[start:]) - len(text[start:].lstrip())), base_offset + end))
                start = end
            i += 1

        if start < length:
            remaining = text[start:].strip()
            if remaining:
                local_start = base_offset + start + (len(text[start:]) - len(text[start:].lstrip()))
                sentences.append((local_start, base_offset + len(text)))
        return sentences

    def _next_non_space_char(self, text: str, index: int) -> str:
        length = len(text)
        while index < length:
            ch = text[index]
            if not ch.isspace():
                return ch
            index += 1
        return ""

    def _process_table(
        self,
        *,
        table_tag: Tag,
        table_index: int,
        current_offset: int,
        active_heading_id: str | None,
    ) -> tuple[List[TableCellAnchor], str]:
        if table_tag.find("tr"):
            return self._process_structured_table(
                table_tag=table_tag,
                table_index=table_index,
                current_offset=current_offset,
                active_heading_id=active_heading_id,
            )
        return self._process_fds_table(
            table_tag=table_tag,
            table_index=table_index,
            current_offset=current_offset,
            active_heading_id=active_heading_id,
        )

    def _process_structured_table(
        self,
        *,
        table_tag: Tag,
        table_index: int,
        current_offset: int,
        active_heading_id: str | None,
    ) -> tuple[List[TableCellAnchor], str]:
        rows = table_tag.find_all("tr")
        if not rows:
            raise ValueError("Table reported having <tr> but none were parsed.")

        raw_grid: list[list[str]] = []
        row_headers: list[str | None] = []
        col_headers: list[str] = []
        spans: dict[tuple[int, int], tuple[int, int, str]] = {}

        max_cols = 0
        for row_idx, row in enumerate(rows):
            cells: list[str] = []
            col_idx = 0
            while (row_idx, col_idx) in spans:
                span_rowspan, span_colspan, span_text = spans[(row_idx, col_idx)]
                cells.append(span_text)
                if span_rowspan > 1:
                    spans[(row_idx + 1, col_idx)] = (span_rowspan - 1, span_colspan, span_text)
                del spans[(row_idx, col_idx)]
                col_idx += 1

            for cell in row.find_all(["th", "td"]):
                while (row_idx, col_idx) in spans:
                    span_rowspan, span_colspan, span_text = spans[(row_idx, col_idx)]
                    cells.append(span_text)
                    if span_rowspan > 1:
                        spans[(row_idx + 1, col_idx)] = (span_rowspan - 1, span_colspan, span_text)
                    del spans[(row_idx, col_idx)]
                    col_idx += 1

                cell_text = self._normalize_whitespace(cell.get_text(separator=" "))
                rowspan = int(cell.get("rowspan", "1") or "1")
                colspan = int(cell.get("colspan", "1") or "1")

                cells.append(cell_text)
                if rowspan > 1 or colspan > 1:
                    for r in range(rowspan):
                        for c in range(colspan):
                            if r == 0 and c == 0:
                                continue
                            spans[(row_idx + r, col_idx + c)] = (rowspan - r, colspan - c, cell_text)

                col_idx += 1

            if len(cells) > max_cols:
                max_cols = len(cells)
            raw_grid.append(list(cells))

        if not raw_grid:
            raise ValueError("Structured table grid empty after parsing.")

        clean_grid = [[self._clean_cell_text(cell) for cell in row] for row in raw_grid]
        classification = self._classify_structured_table(raw_grid)
        if classification == "layout_prose":
            serialized = self._serialize_layout_table(raw_grid)
            return [], serialized

        grid = clean_grid

        for row in grid:
            if len(row) < max_cols:
                row.extend([""] * (max_cols - len(row)))

        markdown = self._render_markdown_table(grid)
        return [], markdown

    def _process_fds_table(
        self,
        *,
        table_tag: Tag,
        table_index: int,
        current_offset: int,
        active_heading_id: str | None,
    ) -> tuple[List[TableCellAnchor], str]:
        raw = table_tag.decode()
        pattern = re.compile(r"<([a-z0-9-]+)>(.*)", re.IGNORECASE)
        lines = raw.splitlines()
        table_rows: list[list[str]] = []

        for row_idx, line in enumerate(lines, start=1):
            match = pattern.match(line.strip())
            if not match:
                continue
            key = match.group(1).upper()
            value = self._normalize_whitespace(match.group(2))
            if not value:
                continue
            table_rows.append([key, value])
        if not table_rows:
            return [], ""
        markdown_grid = [["Field", "Value"], *table_rows]
        markdown = self._render_markdown_table(markdown_grid)
        return [], markdown

    def _link_neighbor_ids(self, anchors: List[AnchorRecord]) -> None:
        for idx, anchor in enumerate(anchors):
            prev_anchor = anchors[idx - 1] if idx > 0 else None
            next_anchor = anchors[idx + 1] if idx + 1 < len(anchors) else None
            anchor.prev_id = prev_anchor.anchor_id if prev_anchor else None
            anchor.next_id = next_anchor.anchor_id if next_anchor else None

    def _assign_parent_heading(self, anchors: List[AnchorRecord]) -> None:
        active_heading: str | None = None
        for anchor in anchors:
            if anchor.kind == ANCHOR_KIND_HEADING:
                active_heading = anchor.anchor_id
                continue
            if anchor.parent_heading:
                continue
            anchor.parent_heading = active_heading


def write_canonical_bundle(
    *,
    canonical_document: CanonicalDocument,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=False)
    canonical_path = output_dir / "canonical.txt"
    anchors_path = output_dir / "anchors.tsv"
    with canonical_path.open("w", encoding="utf-8") as handle:
        handle.write(canonical_document.text)

    fieldnames = [
        "anchor_id",
        "kind",
        "start",
        "end",
        "checksum",
        "context_pre",
        "context_post",
        "prev_id",
        "next_id",
        "parent_heading",
        "attributes_json",
    ]
    with anchors_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for record in canonical_document.anchors:
            writer.writerow(record.to_tsv_row())


def build_prompt_view(canonical_document: CanonicalDocument) -> str:
    lines: list[str] = []
    for anchor in canonical_document.anchors:
        if anchor.kind not in {
            ANCHOR_KIND_SENTENCE,
            ANCHOR_KIND_DEFINITION,
            ANCHOR_KIND_TABLE_CELL,
        }:
            continue
        snippet = canonical_document.text[anchor.start:anchor.end].strip()
        if not snippet:
            raise ValueError(f"Anchor {anchor.anchor_id} has empty snippet.")
        if anchor.kind == ANCHOR_KIND_TABLE_CELL:
            normalized_value = re.sub(r"\s+", " ", snippet)
            row_header = anchor.attributes.get("row_header")
            col_header = anchor.attributes.get("col_header")
            header_parts: list[str] = []
            if row_header:
                header_parts.append(row_header)
            if col_header:
                header_parts.append(col_header)
            if header_parts:
                header_text = " | ".join(header_parts)
            else:
                table_idx = anchor.attributes.get("table_index")
                row_idx = anchor.attributes.get("row_index")
                col_idx = anchor.attributes.get("col_index")
                header_text = f"Table {table_idx} r{row_idx} c{col_idx}"
            lines.append(f"⟦{anchor.anchor_id}⟧ {header_text}: {normalized_value}")
        else:
            if snippet.startswith("|") and "| ---" in snippet:
                table_block = snippet.strip()
                lines.append(f"⟦{anchor.anchor_id}⟧ ```markdown\n{table_block}\n```")
            else:
                normalized_sentence = re.sub(r"\s+", " ", snippet)
                lines.append(f"⟦{anchor.anchor_id}⟧ {normalized_sentence}")
    return "\n".join(lines)


def write_prompt_view(*, prompt_view: str, output_dir: Path) -> None:
    prompt_path = output_dir / "prompt_view.txt"
    with prompt_path.open("w", encoding="utf-8") as handle:
        handle.write(prompt_view)
