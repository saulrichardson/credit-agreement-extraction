from __future__ import annotations

"""Shared helpers for splitting prose into clause-level spans."""

import re
from typing import List, Tuple

ABBREVIATIONS = {
    "mr",
    "mrs",
    "ms",
    "dr",
    "prof",
    "sr",
    "jr",
    "st",
    "no",
    "sec",
    "art",
    "fig",
    "ex",
    "dept",
    "inc",
    "corp",
    "co",
    "ltd",
    "plc",
    "llc",
    "etc",
    "e.g",
    "i.e",
    "vs",
    "v",
    "u.s",
    "u.k",
    "jan",
    "feb",
    "mar",
    "apr",
    "may",
    "jun",
    "jul",
    "aug",
    "sep",
    "sept",
    "oct",
    "nov",
    "dec",
}

CHUNK_BOUNDARY_CHARS = {",", ";", ":", "(", ")", "[", "]", "{", "}", "\"", "'", "“", "”", "‘", "’"}
ABBREVIATION_SEQUENCE = re.compile(r"(?:[A-Za-z]\.){2,}")


def _skip_forward(text: str, idx: int) -> int:
    length = len(text)
    while idx < length and text[idx].isspace():
        idx += 1
    return idx


def _token_before(text: str, idx: int) -> str:
    j = idx - 1
    while j >= 0 and text[j].isspace():
        j -= 1
    end = j + 1
    while j >= 0 and text[j].isalpha():
        j -= 1
    return text[j + 1 : end]


def _chunk_for_abbreviation(text: str, idx: int) -> str:
    start = idx
    while start > 0 and not text[start - 1].isspace() and text[start - 1] not in CHUNK_BOUNDARY_CHARS:
        start -= 1
    return text[start : idx + 1]


def _is_abbreviation(text: str, idx: int) -> bool:
    chunk = _chunk_for_abbreviation(text, idx)
    if ABBREVIATION_SEQUENCE.fullmatch(chunk):
        return True
    token = _token_before(text, idx).lower().strip(".")
    return bool(token) and token in ABBREVIATIONS


def _is_decimal_or_section(text: str, idx: int) -> bool:
    prev_char = text[idx - 1] if idx > 0 else ""
    next_char = text[idx + 1] if idx + 1 < len(text) else ""
    if prev_char.isdigit() and (next_char.isdigit() or next_char in {"(", ")"}):
        return True
    return False


def _is_ellipsis(text: str, idx: int) -> bool:
    prev_char = text[idx - 1] if idx > 0 else ""
    next_char = text[idx + 1] if idx + 1 < len(text) else ""
    return prev_char == "." or next_char == "."


def _should_split(text: str, idx: int) -> bool:
    mark = text[idx]
    if mark not in ".!?":
        return False
    if mark == ".":
        if (
            idx + 2 < len(text)
            and text[idx + 1].isalpha()
            and text[idx + 2] == "."
        ):
            return False
        if _is_ellipsis(text, idx) or _is_decimal_or_section(text, idx) or _is_abbreviation(text, idx):
            return False
    next_idx = _skip_forward(text, idx + 1)
    if next_idx < len(text) and text[next_idx].islower():
        return False
    return True


def sentence_boundaries(text: str) -> List[Tuple[int, int]]:
    """Return (start, end) spans for sentences within ``text`` (relative to the original string)."""
    spans: List[Tuple[int, int]] = []
    length = len(text)
    start = _skip_forward(text, 0)
    idx = start

    while idx < length:
        char = text[idx]
        if char in ".!?":
            if _should_split(text, idx):
                end = idx + 1
                if start < end:
                    spans.append((start, end))
                idx = _skip_forward(text, end)
                start = idx
                continue
        idx += 1

    if start < length:
        tail = text[start:].strip()
        if tail:
            spans.append((start, length))
    return spans
