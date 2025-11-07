from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence

from .canonicalizer import CanonicalCharSource

CONTEXT_WINDOW = 80
DEFAULT_CHUNK_SIZE = 1000


@dataclass(frozen=True)
class SourceSpan:
    source_id: str
    start: int
    end: int


@dataclass(frozen=True)
class AnchorHint:
    heading_like: bool = False
    table_like: bool = False
    xref_like: bool = False


@dataclass(frozen=True)
class Anchor:
    anchor_id: str
    kind: str
    start: int
    end: int
    source_spans: List[SourceSpan]
    context_before_hash: str
    context_after_hash: str
    text_checksum: str
    hint: AnchorHint


@dataclass
class AnchorMap:
    anchors: List[Anchor]
    canonical_length: int

    def __iter__(self) -> Iterable[Anchor]:
        return iter(self.anchors)


def build_anchors(
    *,
    canonical_text: str,
    char_sources: Sequence[Optional[CanonicalCharSource]],
    source_id: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> AnchorMap:
    """Create paragraph and chunk anchors for ``canonical_text``."""
    if len(canonical_text) != len(char_sources):
        raise ValueError("canonical_text and char_sources length mismatch")

    paragraph_spans = list(_iter_paragraph_spans(canonical_text))
    anchors: List[Anchor] = []

    paragraph_counter = 1
    chunk_counter = 1

    for start, end in paragraph_spans:
        if end <= start:
            continue
        anchor_id = f"s{paragraph_counter:06d}"
        anchors.append(
            _make_anchor(
                anchor_id=anchor_id,
                kind="paragraph",
                text=canonical_text,
                start=start,
                end=end,
                char_sources=char_sources,
            )
        )
        paragraph_counter += 1

        remaining = end - start
        if remaining > chunk_size:
            chunk_starts = _chunk_offsets(
                canonical_text,
                start=start,
                end=end,
                chunk_size=chunk_size,
            )
            for chunk_start, chunk_end in chunk_starts:
                if chunk_end - chunk_start <= 0:
                    continue
                chunk_id = f"c{chunk_counter:06d}"
                anchors.append(
                    _make_anchor(
                        anchor_id=chunk_id,
                        kind="chunk",
                        text=canonical_text,
                        start=chunk_start,
                        end=chunk_end,
                        char_sources=char_sources,
                    )
                )
                chunk_counter += 1

    anchors.sort(key=lambda a: a.start)
    return AnchorMap(anchors=anchors, canonical_length=len(canonical_text))


def _iter_paragraph_spans(text: str) -> Iterable[tuple[int, int]]:
    length = len(text)
    cursor = 0

    while cursor < length:
        # Skip leading newlines
        while cursor < length and text[cursor] == "\n":
            cursor += 1
        if cursor >= length:
            break
        start = cursor
        while cursor < length:
            if (
                text[cursor] == "\n"
                and cursor + 1 < length
                and text[cursor + 1] == "\n"
            ):
                break
            cursor += 1
        end = cursor
        yield start, end

        # Advance past blank lines
        while cursor < length and text[cursor] == "\n":
            cursor += 1


def _chunk_offsets(
    text: str,
    *,
    start: int,
    end: int,
    chunk_size: int,
) -> Iterable[tuple[int, int]]:
    cursor = start
    while cursor < end:
        chunk_end = min(end, cursor + chunk_size)
        if chunk_end < end:
            chunk_end = _advance_to_boundary(text, chunk_end, end)
        yield cursor, chunk_end
        if chunk_end == end:
            break
        cursor = chunk_end


def _advance_to_boundary(text: str, candidate: int, upper_bound: int) -> int:
    index = candidate
    while index < upper_bound and not text[index].isspace():
        index += 1
        if index - candidate > 200:
            break
    return index if index > candidate else candidate


def _make_anchor(
    *,
    anchor_id: str,
    kind: str,
    text: str,
    start: int,
    end: int,
    char_sources: Sequence[Optional[CanonicalCharSource]],
) -> Anchor:
    if start < 0 or end > len(text):
        raise ValueError("anchor span out of bounds")

    source_spans = _collect_source_spans(char_sources[start:end])
    context_before_hash = _hash_text(text[max(0, start - CONTEXT_WINDOW) : start])
    context_after_hash = _hash_text(text[end : min(len(text), end + CONTEXT_WINDOW)])
    text_checksum = _hash_text(text[start:end])
    hint = AnchorHint(
        heading_like=_is_heading_like(text[start:end]),
        table_like=False,
        xref_like=_is_xref_like(text[start:end]),
    )
    return Anchor(
        anchor_id=anchor_id,
        kind=kind,
        start=start,
        end=end,
        source_spans=source_spans,
        context_before_hash=context_before_hash,
        context_after_hash=context_after_hash,
        text_checksum=text_checksum,
        hint=hint,
    )


def _collect_source_spans(
    window_sources: Sequence[Optional[CanonicalCharSource]],
) -> List[SourceSpan]:
    spans: List[SourceSpan] = []
    current: Optional[SourceSpan] = None

    for pointer in window_sources:
        if pointer is None:
            current = None
            continue
        if (
            current
            and pointer.source_id == current.source_id
            and pointer.start == current.end
        ):
            current = SourceSpan(pointer.source_id, current.start, pointer.end)
            spans[-1] = current
        else:
            current = SourceSpan(pointer.source_id, pointer.start, pointer.end)
            spans.append(current)
    return spans


def _hash_text(payload: str) -> str:
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _is_heading_like(payload: str) -> bool:
    stripped = payload.strip()
    if not stripped:
        return False
    if len(stripped) > 120:
        return False
    alpha = sum(1 for c in stripped if c.isalpha())
    if alpha == 0:
        return False
    uppercase = sum(1 for c in stripped if c.isalpha() and c == c.upper())
    return uppercase / alpha > 0.85


def _is_xref_like(payload: str) -> bool:
    lowered = payload.lower()
    return any(
        marker in lowered
        for marker in ("section ", "article ", "schedule ", "exhibit ")
    )
