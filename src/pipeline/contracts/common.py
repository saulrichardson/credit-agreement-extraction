from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any


ANCHOR_ID_RE = re.compile(r"^A\d{4,}$")


def normalize_ws(value: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"expected string, got {type(value).__name__}")
    return value.strip()


def validate_anchor_id(value: str) -> str:
    anchor_id = normalize_ws(value)
    if not ANCHOR_ID_RE.fullmatch(anchor_id):
        raise ValueError(f"invalid anchor id {anchor_id!r} (expected like 'A0001')")
    return anchor_id


def validate_anchor_id_list(value: Any, *, allow_empty: bool, label: str) -> list[str]:
    if not isinstance(value, list):
        raise TypeError(f"{label} must be a JSON array")
    cleaned: list[str] = []
    seen: set[str] = set()
    for idx, raw in enumerate(value):
        try:
            anchor_id = validate_anchor_id(raw)
        except Exception as exc:
            raise ValueError(f"{label}[{idx}] invalid anchor id: {exc}") from exc
        if anchor_id in seen:
            continue
        seen.add(anchor_id)
        cleaned.append(anchor_id)
    if not allow_empty and not cleaned:
        raise ValueError(f"{label} must be a non-empty list of anchor ids")
    return cleaned


def validate_trimmed_string_list(value: Any, *, label: str, max_items: int | None = None) -> list[str]:
    if not isinstance(value, list):
        raise TypeError(f"{label} must be a JSON array")
    cleaned: list[str] = []
    for idx, raw in enumerate(value):
        if not isinstance(raw, str):
            raise TypeError(f"{label}[{idx}] must be a string")
        s = raw.strip()
        if not s:
            continue
        cleaned.append(s)
    if max_items is not None and len(cleaned) > max_items:
        raise ValueError(f"{label} must have <= {max_items} items (got {len(cleaned)})")
    return cleaned


@dataclass(frozen=True)
class StructuredValidationContext:
    item_id: str
    allowed_anchor_ids: set[str]
    # Best-effort: tags derived from snippet metadata (categories/buckets/label parts).
    anchor_tags: dict[str, set[str]]
    # Raw snippet text per anchor, for evidence-based validation heuristics.
    snippet_text_by_anchor: dict[str, str]
