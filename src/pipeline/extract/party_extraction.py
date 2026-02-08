from __future__ import annotations

import bisect
import re
from typing import Any


MAX_EXPLICIT_LENDERS = 40

_LENDER_DOUBLE_AS_RE = re.compile(
    r"(?:^|[\n:])\s*(?P<name>[^\n:]+?),\s+as\s+[^\n]{0,140}?\s+and\s+as\s+(?:a\s+)?lender\b",
    flags=re.IGNORECASE,
)
_LENDER_COMMA_SINGLE_RE = re.compile(
    r"(?:^|[\n:])\s*(?P<name>[^\n:]+?),\s+as\s+(?:a\s+)?lender\b",
    flags=re.IGNORECASE,
)
_LENDER_SPACE_SINGLE_RE = re.compile(
    r"(?:^|[\n:])\s*(?P<name>[^\n:]{3,}?)\s+as\s+(?:a\s+)?lender\b",
    flags=re.IGNORECASE,
)


def normalize_hay(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "")).strip().lower()


def _clean_party_name(name: str) -> str:
    cleaned = re.sub(r"\s+", " ", (name or "")).strip()
    cleaned = re.sub(r"^[0-9]{1,3}\s+", "", cleaned)

    if re.search(r"-{5,}", cleaned):
        parts = [p.strip(" \t\r\n,;:-") for p in re.split(r"-{5,}", cleaned) if p.strip()]
        if parts:
            cleaned = parts[-1]

    cleaned = cleaned.strip(" \t\r\n,;:-")
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def extract_lender_names_from_signature_text(text: str) -> list[tuple[int, str]]:
    """Extract lender entity names from signature-style snippet text.

    Returns pairs of (local_offset_in_snippet, lender_name).
    """

    raw = text or ""
    if not raw.strip():
        return []

    hay = raw.lower()
    if not ("as a lender" in hay or "as lender" in hay):
        return []
    if not any(k in hay for k in ("by:", "name:", "title:")):
        return []

    matches: list[tuple[int, int, str]] = []
    used_spans: list[tuple[int, int]] = []

    def _overlaps(span: tuple[int, int]) -> bool:
        a, b = span
        for x, y in used_spans:
            if a < y and x < b:
                return True
        return False

    def _add(pattern: re.Pattern[str]) -> None:
        for m in pattern.finditer(raw):
            span = m.span()
            if _overlaps(span):
                continue
            name = _clean_party_name(m.group("name"))
            if not name:
                continue
            used_spans.append(span)
            matches.append((m.start("name"), span[1], name))

    _add(_LENDER_DOUBLE_AS_RE)
    _add(_LENDER_COMMA_SINGLE_RE)
    _add(_LENDER_SPACE_SINGLE_RE)

    matches_sorted = sorted(matches, key=lambda t: t[0])
    out: list[tuple[int, str]] = []
    seen: set[str] = set()
    for local_offset, _end, name in matches_sorted:
        norm = normalize_hay(name)
        if not norm or norm in seen:
            continue
        seen.add(norm)
        out.append((local_offset, name))
    return out


def extract_lenders_from_snippets(
    *,
    snippets: list[dict[str, Any]],
    catalog: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[str]]:
    """Deterministically extract lenders from snippet evidence.

    Returns:
      - lenders: list of dicts compatible with Party-like objects:
          {"name": str, "source_refs": [anchor_id], "notes": None}
      - lender_source_refs: sorted unique anchor IDs
    """

    lenders: list[dict[str, Any]] = []
    lender_source_refs: set[str] = set()
    seen: set[str] = set()

    ordered = sorted(catalog.values(), key=lambda a: int(a.get("start", 0)))
    starts = [int(a.get("start", 0)) for a in ordered]
    snippet_anchor_ids: set[str] = {
        str(rec.get("anchor_id"))
        for rec in snippets
        if isinstance(rec.get("anchor_id"), str) and isinstance(rec.get("snippet"), str)
    }

    def _anchor_for_pos(pos: int) -> str | None:
        if not ordered:
            return None
        idx = bisect.bisect_right(starts, pos) - 1
        if idx < 0:
            idx = 0
        for j in range(max(0, idx - 3), min(len(ordered), idx + 4)):
            info = ordered[j]
            if int(info.get("start", 0)) <= pos < int(info.get("end", 0)):
                return str(info.get("anchor_id") or "")
        j = max(0, idx)
        while j < len(ordered) and int(ordered[j].get("start", 0)) <= pos:
            info = ordered[j]
            if int(info.get("start", 0)) <= pos < int(info.get("end", 0)):
                return str(info.get("anchor_id") or "")
            j += 1
        return None

    for rec in snippets:
        aid = rec.get("anchor_id")
        snippet = rec.get("snippet")
        if not isinstance(aid, str) or not isinstance(snippet, str):
            continue

        snippet_start = rec.get("snippet_start") if isinstance(rec.get("snippet_start"), int) else None
        extracted = extract_lender_names_from_signature_text(snippet)
        if not extracted:
            continue

        for local_offset, name in extracted:
            norm = normalize_hay(name)
            if not norm or norm in seen:
                continue
            seen.add(norm)

            resolved_anchor = aid
            if snippet_start is not None:
                global_pos = int(snippet_start) + int(local_offset)
                mapped = _anchor_for_pos(global_pos)
                # Keep evidence grounded to anchors that actually exist in the snippet pack.
                if mapped and mapped in snippet_anchor_ids:
                    resolved_anchor = mapped

            lenders.append({"name": name, "source_refs": [resolved_anchor], "notes": None})
            lender_source_refs.add(resolved_anchor)

    return (lenders, sorted(lender_source_refs))


def _looks_like_lender_grid(text: str) -> bool:
    raw = text or ""
    hay = normalize_hay(raw)
    if "commitment" not in hay:
        return False

    has_large_number = bool(re.search(r"\b\d{1,3}(?:,\d{3}){1,}\b", raw)) or bool(re.search(r"\b\d{6,}\b", raw))
    if not has_large_number:
        return False

    has_entity = bool(
        re.search(
            r"\b(bank|n\.a\.|llc|l\.p\.|inc|corp|corporation|company|plc|s\.a\.|ag)\b",
            hay,
        )
    )
    return has_entity


def looks_like_enumerated_lender_list(text: str) -> bool:
    """Check whether snippet text likely contains an enumerated lender list."""

    raw = text or ""
    hay = normalize_hay(raw)

    if _looks_like_lender_grid(raw):
        return True

    lender_mentions = len(re.findall(r"\bas\s+(?:a\s+)?lender\b", hay))
    by_mentions = hay.count("by:") + hay.count("name:") + hay.count("title:")
    if lender_mentions < 2 or by_mentions < 2:
        return False

    has_entity = bool(
        re.search(
            r"\b(bank|n\.a\.|llc|l\.p\.|inc|corp|corporation|company|plc|s\.a\.|ag)\b",
            hay,
        )
    )
    return has_entity
