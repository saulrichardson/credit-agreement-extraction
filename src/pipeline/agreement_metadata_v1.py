from __future__ import annotations

import asyncio
import bisect
import json
import re
import time
import traceback
from pathlib import Path
from typing import Any, Iterable

from pydantic import BaseModel, ConfigDict, Field, field_validator

from .anchors import load_anchor_catalog
from .config import Paths, REQUIRED_MODEL, REQUIRED_REASONING, prompt_hash, update_manifest
from .llm.gateway import DEFAULT_GATEWAY_URL, _ensure_gateway_client_async
from .utils import assert_exists

ANCHOR_ID_RE = re.compile(r"^A\d{4,}$")

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


def _validate_anchor_ids(anchor_ids: list[str]) -> list[str]:
    if not isinstance(anchor_ids, list):
        raise TypeError("source_refs must be a JSON array")
    if not anchor_ids:
        raise ValueError("source_refs must be a non-empty array")
    cleaned: list[str] = []
    for idx, raw in enumerate(anchor_ids):
        if not isinstance(raw, str):
            raise TypeError(f"anchor IDs must be strings; got {type(raw).__name__} at index {idx}")
        anchor_id = raw.strip()
        if not ANCHOR_ID_RE.fullmatch(anchor_id):
            raise ValueError(f"invalid anchor_id {anchor_id!r} (expected like 'A0001')")
        cleaned.append(anchor_id)
    return cleaned


class Party(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    source_refs: list[str] = Field(default_factory=list)
    notes: str | None = None

    @field_validator("name")
    @classmethod
    def _name_validate(cls, value: str) -> str:
        if not isinstance(value, str):
            raise TypeError("name must be a string")
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("name must be non-empty")
        return cleaned

    @field_validator("source_refs")
    @classmethod
    def _source_refs_validate(cls, value: list[str]) -> list[str]:
        return _validate_anchor_ids(value)


class PartyRole(BaseModel):
    model_config = ConfigDict(extra="forbid")

    role_label: str
    party_name: str
    source_refs: list[str] = Field(default_factory=list)
    notes: str | None = None

    @field_validator("role_label")
    @classmethod
    def _role_label_validate(cls, value: str) -> str:
        if not isinstance(value, str):
            raise TypeError("role_label must be a string")
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("role_label must be non-empty")
        return cleaned

    @field_validator("party_name")
    @classmethod
    def _party_name_validate(cls, value: str) -> str:
        if not isinstance(value, str):
            raise TypeError("party_name must be a string")
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("party_name must be non-empty")
        return cleaned

    @field_validator("source_refs")
    @classmethod
    def _source_refs_validate(cls, value: list[str]) -> list[str]:
        return _validate_anchor_ids(value)


class Money(BaseModel):
    model_config = ConfigDict(extra="forbid")

    amount: float | None = None
    currency: str | None = Field(default=None, description="3-letter currency code when available")

    @field_validator("amount")
    @classmethod
    def _amount_validate(cls, value: float | None) -> float | None:
        if value is None:
            return None
        if not isinstance(value, (int, float)):
            raise TypeError("amount must be a JSON number")
        if not float(value) == float(value):  # NaN
            raise ValueError("amount must not be NaN")
        if float(value) <= 0:
            raise ValueError("amount must be > 0 when provided")
        return float(value)


class Facility(BaseModel):
    model_config = ConfigDict(extra="forbid")

    facility_name: str | None = None
    facility_types: list[str] = Field(default_factory=list)
    committed_amount: Money | None = None
    currency: str | None = None
    maturity_date: str | None = Field(default=None, description="YYYY-MM-DD when available")
    termination_date: str | None = Field(default=None, description="YYYY-MM-DD when available")
    source_refs: list[str] = Field(default_factory=list)
    notes: str | None = None

    @field_validator("source_refs")
    @classmethod
    def _source_refs_validate(cls, value: list[str]) -> list[str]:
        return _validate_anchor_ids(value)

    @field_validator("facility_types")
    @classmethod
    def _facility_types_validate(cls, value: list[str]) -> list[str]:
        if not isinstance(value, list):
            raise TypeError("facility_types must be a JSON array")
        if not value:
            raise ValueError("facility_types must be non-empty when a facility object is provided")
        allowed = {"revolver", "term_loan", "letters_of_credit", "swingline", "delayed_draw", "other"}
        cleaned: list[str] = []
        for idx, raw in enumerate(value):
            if not isinstance(raw, str):
                raise TypeError(f"facility_types entries must be strings; got {type(raw).__name__} at index {idx}")
            v = raw.strip()
            if v not in allowed:
                raise ValueError(f"invalid facility_type {v!r} (allowed: {', '.join(sorted(allowed))})")
            cleaned.append(v)
        # de-dupe, preserve order
        seen: set[str] = set()
        out: list[str] = []
        for v in cleaned:
            if v in seen:
                continue
            seen.add(v)
            out.append(v)
        return out

    @field_validator("committed_amount")
    @classmethod
    def _committed_amount_validate(cls, value: Money | None) -> Money | None:
        if value is None:
            return None
        if value.amount is None:
            raise ValueError("committed_amount.amount must be provided when committed_amount is not null")
        return value


class AgreementMetadataArtifact(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str

    borrowers: list[Party] = Field(default_factory=list)
    guarantors: list[Party] = Field(default_factory=list)
    agents: list[PartyRole] = Field(default_factory=list)
    arrangers: list[PartyRole] = Field(default_factory=list)
    lenders: list[Party] = Field(default_factory=list)

    lenders_present_but_omitted: bool = False
    lenders_source_refs: list[str] = Field(default_factory=list)

    facilities: list[Facility] = Field(default_factory=list)
    notes: str | None = None

    @field_validator("lenders_source_refs")
    @classmethod
    def _lenders_source_refs_validate(cls, value: list[str]) -> list[str]:
        # This field may be empty only when lenders_present_but_omitted == false.
        if not isinstance(value, list):
            raise TypeError("lenders_source_refs must be a JSON array")
        if not value:
            return []
        return _validate_anchor_ids(value)


def _load_snippets_v2(paths: Paths, item_id: str) -> list[dict]:
    path = assert_exists(
        paths.run_dir / "retrieval_v2" / f"{item_id}_snippets.jsonl",
        message=f"Missing v2 snippets for {item_id}: run retrieve-v2 first.",
    )
    snippets: list[dict] = []
    with path.open() as fh:
        for line in fh:
            if not line.strip():
                continue
            try:
                snippets.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    if not snippets:
        raise RuntimeError(f"No snippets parsed for {item_id} (file: {path})")
    return snippets


def _filter_snippets(snippets: list[dict], categories: Iterable[str]) -> list[dict]:
    wanted = {c.strip().lower() for c in categories if c.strip()}
    if not wanted:
        return snippets
    filtered: list[dict] = []
    for rec in snippets:
        rec_categories = [c.lower() for c in (rec.get("categories") or []) if isinstance(c, str)]
        rec_buckets = [c.lower() for c in (rec.get("buckets") or []) if isinstance(c, str)]
        label = rec.get("label") or ""
        rec_label_parts = [c.strip().lower() for c in str(label).split(",") if c.strip()]
        if any(c in wanted for c in rec_categories + rec_buckets + rec_label_parts):
            filtered.append(rec)
    return filtered


def _render_snippet_block(snippets: list[dict]) -> str:
    blocks: list[str] = []
    for rec in snippets:
        aid = rec.get("anchor_id") or "UNK"
        label = rec.get("label") or rec.get("type")
        toc_title = rec.get("toc_title")
        toc_chunk_id = rec.get("toc_chunk_id")
        snippet_text = (rec.get("snippet") or "").strip()
        header = f"[[{aid}]]"
        lines: list[str] = [header]
        if label:
            lines.append(f"label: {label}")
        if isinstance(toc_title, str) and toc_title.strip():
            if isinstance(toc_chunk_id, int):
                lines.append(f"toc: {toc_title.strip()} (chunk {toc_chunk_id})")
            else:
                lines.append(f"toc: {toc_title.strip()}")
        lines.append(snippet_text)
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


def _normalize_hay(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "")).strip().lower()


def _tokenize(text: str) -> list[str]:
    # Keep alphanumerics; drop punctuation for robust matching.
    return [t for t in re.split(r"[^a-z0-9]+", _normalize_hay(text)) if t]


def _contains_all_tokens(hay: str, needle: str) -> bool:
    hay_tokens = set(_tokenize(hay))
    want = _tokenize(needle)
    if not want:
        return False
    return all(t in hay_tokens for t in want)


def _clean_party_name(name: str) -> str:
    cleaned = re.sub(r"\s+", " ", (name or "")).strip()
    cleaned = re.sub(r"^[0-9]{1,3}\s+", "", cleaned)  # drop signature page numbers like "90 "

    # Signature blocks often include long dashed separators between signatory fields and entity names.
    # If present, keep the last segment (most likely the entity name).
    if re.search(r"-{5,}", cleaned):
        parts = [p.strip(" \t\r\n,;:-") for p in re.split(r"-{5,}", cleaned) if p.strip()]
        if parts:
            cleaned = parts[-1]

    cleaned = cleaned.strip(" \t\r\n,;:-")
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def _extract_lender_names_from_signature_text(text: str) -> list[tuple[int, str]]:
    """Extract lender entity names from signature-page style text.

    We rely on local textual patterns like:
      "FOO BANK, as a Lender"
      "BAR BANK, as Co-Agent and as a Lender"
      "DRESDNER BANK AG, New York ... Branches as a Lender"

    This is deliberately conservative and only runs on snippet blocks that already
    look signature-like (By/Name/Title markers).
    """

    raw = text or ""
    if not raw.strip():
        return []

    # Guardrail: only parse signature-like blocks to avoid capturing operative clauses.
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

    # Order matters: the simpler patterns can mis-capture in the presence of "and as a Lender".
    _add(_LENDER_DOUBLE_AS_RE)
    _add(_LENDER_COMMA_SINGLE_RE)
    _add(_LENDER_SPACE_SINGLE_RE)

    # Sort by occurrence order; de-dupe by normalized name.
    matches_sorted = sorted(matches, key=lambda t: t[0])
    out: list[tuple[int, str]] = []
    seen: set[str] = set()
    for _a, _b, name in matches_sorted:
        norm = _normalize_hay(name)
        if not norm or norm in seen:
            continue
        seen.add(norm)
        out.append((_a, name))
    return out


def _extract_lenders_from_snippets(
    *,
    snippets: list[dict],
    catalog: dict[str, dict[str, Any]],
) -> tuple[list[Party], list[str]]:
    """Deterministically extract lenders from signature-page snippets.

    Returns:
      - lenders: list[Party] (each with a single anchor in source_refs)
      - lender_source_refs: sorted unique anchor IDs supporting the list
    """

    lenders: list[Party] = []
    lender_source_refs: set[str] = set()
    seen: set[str] = set()

    ordered = sorted(catalog.values(), key=lambda a: int(a.get("start", 0)))
    starts = [int(a.get("start", 0)) for a in ordered]

    def _anchor_for_pos(pos: int) -> str | None:
        if not ordered:
            return None
        idx = bisect.bisect_right(starts, pos) - 1
        if idx < 0:
            idx = 0
        # Scan local neighborhood (anchors are in order; most hits are close).
        for j in range(max(0, idx - 3), min(len(ordered), idx + 4)):
            info = ordered[j]
            if int(info.get("start", 0)) <= pos < int(info.get("end", 0)):
                return str(info.get("anchor_id") or "")
        # Fallback: linear forward scan (rare).
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

        snippet_start = rec.get("snippet_start")
        if not isinstance(snippet_start, int):
            # retrieval_v2 should always emit snippet_start; if absent, fall back to the record's anchor_id.
            snippet_start = None

        # Extract names with local offsets, then map to true anchor IDs using snippet_start offsets.
        extracted = _extract_lender_names_from_signature_text(snippet)
        if not extracted:
            continue
        for local_offset, name in extracted:
            norm = _normalize_hay(name)
            if not norm or norm in seen:
                continue
            seen.add(norm)

            resolved_anchor = aid
            if snippet_start is not None:
                global_pos = int(snippet_start) + int(local_offset)
                mapped = _anchor_for_pos(global_pos)
                if mapped:
                    resolved_anchor = mapped

            lenders.append(Party(name=name, source_refs=[resolved_anchor], notes=None))
            lender_source_refs.add(resolved_anchor)

    return (lenders, sorted(lender_source_refs))


def _month_names(month: int) -> list[str]:
    names = [
        "january",
        "february",
        "march",
        "april",
        "may",
        "june",
        "july",
        "august",
        "september",
        "october",
        "november",
        "december",
    ]
    if 1 <= month <= 12:
        name = names[month - 1]
        return [name, name[:3]]
    return []


def _validate_iso_date_against_evidence(date_iso: str, evidence: str) -> bool:
    """Heuristic: allow ISO dates when evidence contains matching MonthName + day + year."""

    m = re.fullmatch(r"(\d{4})-(\d{2})-(\d{2})", date_iso.strip())
    if not m:
        return False
    year = int(m.group(1))
    month = int(m.group(2))
    day = int(m.group(3))
    hay = _normalize_hay(evidence)
    if str(year) not in hay:
        return False
    # Require month name (Oct/October) to reduce false positives from random numbers.
    if not any(name in hay for name in _month_names(month)):
        return False
    # Require a standalone day number.
    if not re.search(rf"\b{day}\b", hay):
        return False
    return True


def _evidence_text_for_refs(snippet_by_anchor: dict[str, str], anchor_ids: list[str]) -> str:
    parts: list[str] = []
    for aid in anchor_ids:
        txt = snippet_by_anchor.get(aid)
        if txt:
            parts.append(txt)
    return "\n\n".join(parts)


def _require_supported_string(
    *,
    field: str,
    value: str,
    source_refs: list[str],
    snippet_by_anchor: dict[str, str],
) -> None:
    evidence = _evidence_text_for_refs(snippet_by_anchor, source_refs)
    if not evidence:
        raise RuntimeError(f"{field}: no evidence text found for cited anchors {source_refs}")
    if _normalize_hay(value) not in _normalize_hay(evidence):
        # Fall back to token containment for capitalization/punctuation changes.
        if not _contains_all_tokens(evidence, value):
            # Provide a concrete hint for retries: where does this value appear (if anywhere)?
            candidates: list[str] = []
            for aid, txt in snippet_by_anchor.items():
                if not txt:
                    continue
                if _normalize_hay(value) in _normalize_hay(txt) or _contains_all_tokens(txt, value):
                    candidates.append(aid)
            candidates = sorted(set(candidates))
            if candidates:
                raise RuntimeError(
                    f"{field}: value {value!r} not supported by cited anchors {source_refs}. "
                    f"Candidate anchors containing this value: {candidates[:12]}"
                )
            raise RuntimeError(f"{field}: value {value!r} not supported by cited anchors {source_refs}")


def _supported_by_refs(*, value: str, source_refs: list[str], snippet_by_anchor: dict[str, str]) -> bool:
    evidence = _evidence_text_for_refs(snippet_by_anchor, source_refs)
    if not evidence:
        return False
    if _normalize_hay(value) in _normalize_hay(evidence):
        return True
    return _contains_all_tokens(evidence, value)


def _candidate_anchors_for_value(
    *,
    value: str,
    snippet_by_anchor: dict[str, str],
    catalog: dict[str, dict[str, Any]],
    max_candidates: int = 8,
) -> list[str]:
    if not value.strip():
        return []
    candidates: list[str] = []
    for aid, txt in snippet_by_anchor.items():
        if not txt:
            continue
        if _normalize_hay(value) in _normalize_hay(txt) or _contains_all_tokens(txt, value):
            candidates.append(aid)

    def _order(aid: str) -> int:
        info = catalog.get(aid) or {}
        try:
            return int(info.get("order", 10**9))
        except Exception:
            return 10**9

    deduped = sorted(set(candidates), key=_order)
    return deduped[: max(0, int(max_candidates))]


def _candidate_anchors_for_facility_type(
    *,
    facility_type: str,
    snippet_by_anchor: dict[str, str],
    catalog: dict[str, dict[str, Any]],
    max_candidates: int = 6,
) -> list[str]:
    ft = facility_type.strip()
    if not ft:
        return []

    needles: list[str] = []
    if ft == "revolver":
        needles = ["revolving", "revolver"]
    elif ft == "term_loan":
        needles = ["term loan"]
    elif ft == "letters_of_credit":
        needles = ["letter of credit", "letters of credit", "l/c"]
    elif ft == "swingline":
        needles = ["swingline"]
    elif ft == "delayed_draw":
        needles = ["delayed draw"]
    elif ft == "other":
        return []
    else:
        return []

    candidates: list[str] = []
    for aid, txt in snippet_by_anchor.items():
        if not txt:
            continue
        hay = _normalize_hay(txt)
        if any(n in hay for n in needles):
            candidates.append(aid)

    def _order(aid: str) -> int:
        info = catalog.get(aid) or {}
        try:
            return int(info.get("order", 10**9))
        except Exception:
            return 10**9

    deduped = sorted(set(candidates), key=_order)
    return deduped[: max(0, int(max_candidates))]


def _dedupe_anchor_ids(anchor_ids: list[str], *, catalog: dict[str, dict[str, Any]], max_len: int = 12) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for aid in anchor_ids:
        if not isinstance(aid, str):
            continue
        a = aid.strip()
        if not a or a in seen:
            continue
        seen.add(a)
        out.append(a)

    def _order(aid: str) -> int:
        info = catalog.get(aid) or {}
        try:
            return int(info.get("order", 10**9))
        except Exception:
            return 10**9

    ordered = sorted(out, key=_order)
    return ordered[: max(1, int(max_len))] if ordered else []


def _auto_fix_party_refs(
    *,
    field: str,
    value: str,
    obj: Any,
    snippet_by_anchor: dict[str, str],
    catalog: dict[str, dict[str, Any]],
    corrections: list[dict],
) -> None:
    """If the model cited the wrong anchor, deterministically swap to a correct one.

    This is not a fallback for missing evidence: we only change refs when the value
    is present in *some* snippet and the current refs do not support it.
    """

    source_refs = list(getattr(obj, "source_refs", []) or [])
    if not source_refs:
        return
    if _supported_by_refs(value=value, source_refs=source_refs, snippet_by_anchor=snippet_by_anchor):
        return

    candidates = _candidate_anchors_for_value(value=value, snippet_by_anchor=snippet_by_anchor, catalog=catalog)
    if not candidates:
        return

    new_refs = [candidates[0]]
    setattr(obj, "source_refs", new_refs)
    corrections.append(
        {
            "field": field,
            "value": value,
            "old_source_refs": source_refs,
            "new_source_refs": new_refs,
            "reason": "auto-fixed source_refs to an anchor whose snippet contains the value",
        }
    )


def _require_supported_currency(
    *,
    currency: str,
    source_refs: list[str],
    snippet_by_anchor: dict[str, str],
) -> None:
    evidence = _evidence_text_for_refs(snippet_by_anchor, source_refs)
    hay = _normalize_hay(evidence)
    cur = currency.strip().upper()
    if not re.fullmatch(r"[A-Z]{3}", cur):
        raise RuntimeError(f"currency: invalid code {currency!r}")
    # Strict: require explicit evidence. We do not treat '$' as USD.
    if cur.lower() in hay:
        return
    if cur == "USD":
        if ("u.s." in hay or "united states" in hay) and "dollar" in hay:
            return
    if cur == "EUR" and "euro" in hay:
        return
    if cur == "GBP" and "pound" in hay:
        return
    raise RuntimeError(f"currency: {cur} not explicitly supported by cited anchors {source_refs}")


def _require_supported_number(
    *,
    field: str,
    value: float,
    source_refs: list[str],
    snippet_by_anchor: dict[str, str],
) -> None:
    evidence = _evidence_text_for_refs(snippet_by_anchor, source_refs)
    if not evidence:
        raise RuntimeError(f"{field}: no evidence text found for cited anchors {source_refs}")

    # Prefer integer matching when possible (committed amounts are typically integers).
    candidates: list[str] = []
    if abs(value - round(value)) < 1e-6:
        iv = int(round(value))
        candidates.extend([str(iv), f"{iv:,}"])
    else:
        rendered = f"{value:.6f}".rstrip("0").rstrip(".")
        if rendered:
            candidates.append(rendered)
            # Add a comma-formatted variant when possible.
            if "." in rendered:
                lhs, rhs = rendered.split(".", 1)
                if lhs.isdigit():
                    candidates.append(f"{int(lhs):,}.{rhs}")
            elif rendered.isdigit():
                candidates.append(f"{int(rendered):,}")

    for cand in candidates:
        if not cand:
            continue
        if re.search(rf"(?<!\\d){re.escape(cand)}(?!\\d)", evidence):
            return

    raise RuntimeError(f"{field}: value {value!r} not supported by cited anchors {source_refs}")


def _require_supported_facility_types(
    *,
    facility_types: list[str],
    source_refs: list[str],
    snippet_by_anchor: dict[str, str],
) -> None:
    evidence = _evidence_text_for_refs(snippet_by_anchor, source_refs)
    hay = _normalize_hay(evidence)
    for ft in facility_types:
        if ft == "revolver":
            if "revolving" in hay or "revolver" in hay:
                continue
            raise RuntimeError("facility_types: revolver not supported by evidence")
        if ft == "term_loan":
            if "term loan" in hay:
                continue
            raise RuntimeError("facility_types: term_loan not supported by evidence")
        if ft == "letters_of_credit":
            if "letter of credit" in hay or "letters of credit" in hay or "l/c" in hay:
                continue
            raise RuntimeError("facility_types: letters_of_credit not supported by evidence")
        if ft == "swingline":
            if "swingline" in hay:
                continue
            raise RuntimeError("facility_types: swingline not supported by evidence")
        if ft == "delayed_draw":
            if "delayed draw" in hay:
                continue
            raise RuntimeError("facility_types: delayed_draw not supported by evidence")
        if ft == "other":
            # Allowed, but caller should explain in notes.
            continue
        raise RuntimeError(f"facility_types: unsupported value {ft!r}")


def _looks_like_lender_grid(text: str) -> bool:
    """Heuristic: lender commitment/addresses grid (table-style evidence)."""

    raw = text or ""
    hay = _normalize_hay(raw)
    if "commitment" not in hay:
        return False

    # Lender commitment schedules typically contain large numeric amounts.
    has_large_number = bool(re.search(r"\b\d{1,3}(?:,\d{3}){1,}\b", raw)) or bool(re.search(r"\b\d{6,}\b", raw))
    if not has_large_number:
        return False

    # And typically contain lender entity names (Bank, LLC, etc.)
    has_entity = bool(
        re.search(
            r"\b(bank|n\.a\.|llc|l\.p\.|inc|corp|corporation|company|plc|s\.a\.|ag)\b",
            hay,
        )
    )
    if not has_entity:
        return False

    return True


def _looks_like_enumerated_lender_list(text: str) -> bool:
    """Heuristic guardrail for lenders_present_but_omitted evidence.

    We allow two main evidence shapes:
    1) A commitment/addresses grid (numbers + "commitment" + entity markers)
    2) Signature-page enumerations where multiple parties sign "as a Lender" with By/Name/Title blocks

    We intentionally reject schedule references that *name* a lender schedule but do not enumerate lenders.
    """

    raw = text or ""
    hay = _normalize_hay(raw)

    if _looks_like_lender_grid(raw):
        return True

    # Signature lists: repeated "... as a Lender" plus signatory lines.
    lender_mentions = len(re.findall(r"\bas\s+(?:a\s+)?lender\b", hay))
    by_mentions = hay.count("by:") + hay.count("name:") + hay.count("title:")

    if lender_mentions < 2 or by_mentions < 2:
        return False

    # Require at least one "entity-ish" marker to reduce false positives.
    has_entity = bool(
        re.search(
            r"\b(bank|n\.a\.|llc|l\.p\.|inc|corp|corporation|company|plc|s\.a\.|ag)\b",
            hay,
        )
    )
    if not has_entity:
        return False

    return True


def _render_prompt(template: str, snippets_block: str) -> str:
    template = template.strip()
    if "{document}" in template:
        return template.replace("{document}", snippets_block)
    if "{snippets}" in template:
        return template.replace("{snippets}", snippets_block)
    if "INPUT" in template and template.rstrip().endswith("follows.)"):
        # Prompt already ends with an input marker; append directly.
        return f"{template}\n\n{snippets_block}"
    return f"{template}\n\n{snippets_block}"


def _retry_prompt(base_prompt: str, *, attempt: int, error: str) -> str:
    if attempt == 2:
        rules = (
            "Your previous response was not valid. Fix it.\n"
            "- Output MUST be a single JSON object (no markdown, no code fences, no prose).\n"
            "- JSON MUST match the exact shape described in the prompt.\n"
            "- Every party/role/facility object MUST include non-empty source_refs.\n"
            "- amount MUST be a JSON number (no quotes, no commas).\n"
        )
    else:
        rules = (
            "STRICT MODE (final attempt):\n"
            "- Output ONLY JSON.\n"
            "- Include ALL required top-level keys exactly.\n"
            "- Do NOT include any extra keys.\n"
            "- If you cannot support a field from the snippets, use null/[] and explain in notes.\n"
        )
    return f"{base_prompt}\n\n=== RETRY REQUIRED ===\nError: {error}\n\n{rules}"


async def _call_gateway(*, client: Any, prompt: str, model: str, temperature: float, reasoning: str | None) -> str:
    result = await client.complete_response(
        model=model,
        input_messages=[{"role": "user", "content": prompt}],
        reasoning={"effort": reasoning} if reasoning else None,
        temperature=temperature,
        max_output_tokens=None,
        metadata=None,
    )
    if isinstance(result, dict):
        return result.get("text") or ""
    return str(result)


def run_agreement_metadata_v1(
    paths: Paths,
    item_ids: Iterable[str],
    prompt_path: Path,
    *,
    categories: Iterable[str] | None = None,
    model: str | None = None,
    gateway_url: str | None = None,
    temperature: float = 0.0,
    reasoning: str | None = None,
    gateway_timeout: float | None = None,
    concurrency: int = 3,
    attempts: int = 3,
    output_subdir: str | None = None,
) -> None:
    """Extract agreement parties/roles + facility headline terms from v2 retrieval snippets.

    This stage is strict and fails loudly:
    - Output must be valid JSON and validate against AgreementMetadataArtifact.
    - Anchor IDs must be real A#### anchors present in anchors.tsv.
    - Empty/garbage outputs are treated as errors (no silent success).
    """

    assert_exists(prompt_path, message=f"Agreement metadata prompt not found: {prompt_path}")

    model = REQUIRED_MODEL
    reasoning = REQUIRED_REASONING
    prompt_template = prompt_path.read_text()
    prompt_digest = prompt_hash(prompt_path)

    out_root = paths.run_dir / "agreement_metadata"
    resolved_output_subdir = output_subdir or prompt_path.stem
    out_dir = out_root / resolved_output_subdir
    out_dir.mkdir(parents=True, exist_ok=True)

    wanted_categories = list(categories) if categories is not None else ["metadata", "fundamental"]

    items = list(item_ids)
    attempts = max(1, int(attempts))

    GatewayAgentClient = _ensure_gateway_client_async()

    async def _run_async() -> None:
        sem = asyncio.Semaphore(max(1, concurrency))
        resolved_gateway_url = gateway_url or DEFAULT_GATEWAY_URL
        failures: list[tuple[str, str]] = []

        async with GatewayAgentClient(base_url=resolved_gateway_url, timeout=gateway_timeout or 600.0) as client:

            async def _process(item_id: str) -> None:
                async with sem:
                    try:
                        snippets = _load_snippets_v2(paths, item_id)
                        if wanted_categories:
                            snippets = _filter_snippets(snippets, wanted_categories)
                        if not snippets:
                            raise RuntimeError(
                                "No snippets after category filter. "
                                f"Requested categories: {', '.join(wanted_categories)}"
                            )

                        snippet_block = _render_snippet_block(snippets)
                        base_prompt = _render_prompt(prompt_template, snippet_block)

                        catalog = load_anchor_catalog(paths, item_id)
                        snippet_by_anchor = {
                            str(rec.get("anchor_id")): str(rec.get("snippet") or "")
                            for rec in snippets
                            if isinstance(rec.get("anchor_id"), str)
                        }
                        rec_by_anchor = {
                            str(rec.get("anchor_id")): rec
                            for rec in snippets
                            if isinstance(rec.get("anchor_id"), str)
                        }

                        def _write_outputs(
                            *,
                            artifact_json: Any,
                            attempts_used: int,
                            auto_corrections: list[dict] | None = None,
                        ) -> None:
                            artifact_path = out_dir / f"{item_id}.json"
                            artifact_path.write_text(json.dumps(artifact_json, indent=2))
                            meta_path = out_dir / f"{item_id}.meta.json"
                            meta_path.write_text(
                                json.dumps(
                                    {
                                        "schema_version": "agreement_metadata_v1_artifact_meta",
                                        "stage": "agreement_metadata_v1",
                                        "run_id": paths.run_id,
                                        "item_id": item_id,
                                        "created_at": int(time.time()),
                                        "gateway_url": resolved_gateway_url,
                                        "model": model,
                                        "reasoning_effort": reasoning,
                                        "temperature": float(temperature),
                                        "prompt": str(prompt_path),
                                        "prompt_sha256": prompt_digest,
                                        "attempts_used": int(attempts_used),
                                        "categories": wanted_categories,
                                        "auto_corrections": auto_corrections or [],
                                    },
                                    indent=2,
                                )
                            )

                        last_error = "unknown"
                        for attempt in range(1, attempts + 1):
                            prompt = base_prompt if attempt == 1 else _retry_prompt(base_prompt, attempt=attempt, error=last_error)
                            raw_text = await _call_gateway(
                                client=client,
                                prompt=prompt,
                                model=model,
                                temperature=temperature,
                                reasoning=reasoning,
                            )
                            (out_dir / f"{item_id}.attempt{attempt}.raw.txt").write_text(raw_text)

                            try:
                                payload = json.loads(raw_text)
                            except json.JSONDecodeError as exc:
                                last_error = f"invalid JSON: {exc}"
                                continue

                            try:
                                artifact = AgreementMetadataArtifact.model_validate(payload)
                            except Exception as exc:
                                last_error = f"schema validation failed: {exc}"
                                continue

                            if artifact.schema_version != "agreement_metadata_v1":
                                last_error = f"schema_version mismatch: expected 'agreement_metadata_v1' got {artifact.schema_version!r}"
                                continue

                            # Deterministic fix-up (non-silent): models sometimes cite a nearby
                            # anchor ID even though the value appears in a different snippet.
                            # When the value exists somewhere in the provided snippets, we can
                            # safely re-point source_refs to a supporting anchor.
                            auto_corrections: list[dict] = []
                            for p in artifact.borrowers:
                                _auto_fix_party_refs(
                                    field="borrowers.name",
                                    value=p.name,
                                    obj=p,
                                    snippet_by_anchor=snippet_by_anchor,
                                    catalog=catalog,
                                    corrections=auto_corrections,
                                )
                            for p in artifact.guarantors:
                                _auto_fix_party_refs(
                                    field="guarantors.name",
                                    value=p.name,
                                    obj=p,
                                    snippet_by_anchor=snippet_by_anchor,
                                    catalog=catalog,
                                    corrections=auto_corrections,
                                )
                            for p in artifact.lenders:
                                _auto_fix_party_refs(
                                    field="lenders.name",
                                    value=p.name,
                                    obj=p,
                                    snippet_by_anchor=snippet_by_anchor,
                                    catalog=catalog,
                                    corrections=auto_corrections,
                                )
                            for r in artifact.agents:
                                _auto_fix_party_refs(
                                    field="agents.party_name",
                                    value=r.party_name,
                                    obj=r,
                                    snippet_by_anchor=snippet_by_anchor,
                                    catalog=catalog,
                                    corrections=auto_corrections,
                                )
                            for r in artifact.arrangers:
                                _auto_fix_party_refs(
                                    field="arrangers.party_name",
                                    value=r.party_name,
                                    obj=r,
                                    snippet_by_anchor=snippet_by_anchor,
                                    catalog=catalog,
                                    corrections=auto_corrections,
                                )

                            # Deterministic lenders extraction (signature-aware): improves reliability when
                            # lender lists appear only in signature pages and the model omits/mis-handles them.
                            det_lenders, det_refs = _extract_lenders_from_snippets(snippets=snippets, catalog=catalog)
                            det_count = len(det_lenders)
                            if det_count:
                                if det_count <= MAX_EXPLICIT_LENDERS:
                                    # Prefer deterministic extraction when it increases completeness.
                                    if det_count > len(artifact.lenders) or artifact.lenders_present_but_omitted:
                                        old_count = len(artifact.lenders)
                                        old_present = bool(artifact.lenders_present_but_omitted)
                                        old_refs = list(artifact.lenders_source_refs or [])
                                        artifact.lenders = det_lenders
                                        artifact.lenders_present_but_omitted = False
                                        artifact.lenders_source_refs = det_refs
                                        auto_corrections.append(
                                            {
                                                "field": "lenders",
                                                "value": f"deterministic lenders extracted (count={det_count})",
                                                "old_source_refs": old_refs,
                                                "new_source_refs": det_refs,
                                                "reason": (
                                                    "replaced model lenders output with deterministic extraction from signature snippets "
                                                    f"(old_count={old_count}, old_present_but_omitted={old_present})"
                                                ),
                                            }
                                        )
                                else:
                                    # Too many lenders to list: enforce omitted=true with strong evidence refs.
                                    if (not artifact.lenders_present_but_omitted) or artifact.lenders:
                                        old_count = len(artifact.lenders)
                                        old_present = bool(artifact.lenders_present_but_omitted)
                                        old_refs = list(artifact.lenders_source_refs or [])
                                        artifact.lenders = []
                                        artifact.lenders_present_but_omitted = True
                                        artifact.lenders_source_refs = det_refs
                                        auto_corrections.append(
                                            {
                                                "field": "lenders_present_but_omitted",
                                                "value": f"deterministic lenders extracted (count={det_count})",
                                                "old_source_refs": old_refs,
                                                "new_source_refs": det_refs,
                                                "reason": (
                                                    "set lenders_present_but_omitted=true due to deterministic extraction exceeding max "
                                                    f"(max={MAX_EXPLICIT_LENDERS}, old_count={old_count}, old_present_but_omitted={old_present})"
                                                ),
                                            }
                                        )
                                    elif artifact.lenders_present_but_omitted and not artifact.lenders_source_refs:
                                        artifact.lenders_source_refs = det_refs
                                        auto_corrections.append(
                                            {
                                                "field": "lenders_source_refs",
                                                "value": f"filled from deterministic lender evidence (count={det_count})",
                                                "old_source_refs": [],
                                                "new_source_refs": det_refs,
                                                "reason": "model omitted lenders_source_refs but signature snippets contained an enumerated lender list",
                                            }
                                        )

                            # Post-correction consistency checks for lenders fields.
                            if artifact.lenders_present_but_omitted and not artifact.lenders_source_refs:
                                last_error = "lenders_present_but_omitted is true but lenders_source_refs is empty"
                                continue
                            if artifact.lenders_present_but_omitted and artifact.lenders:
                                last_error = "lenders_present_but_omitted is true but lenders[] is non-empty"
                                continue
                            if not artifact.lenders_present_but_omitted and not artifact.lenders and artifact.lenders_source_refs:
                                last_error = "lenders_source_refs must be [] when no lenders are extracted and lenders_present_but_omitted is false"
                                continue
                            if artifact.lenders and not artifact.lenders_source_refs:
                                # Fill automatically from lender objects; this is a pure structural consistency fix.
                                artifact.lenders_source_refs = sorted(
                                    {aid for p in artifact.lenders for aid in (p.source_refs or []) if isinstance(aid, str)}
                                )

                            # Facility evidence fix-ups: models sometimes cite only one anchor even though the
                            # evidence for facility types lives elsewhere in the snippet pack. We add (not replace)
                            # anchors that contain the type keywords, then re-dedupe in doc order. If we still
                            # cannot support the type evidence, validation will fail loudly later.
                            for f in artifact.facilities:
                                old_refs = list(f.source_refs or [])
                                extra: list[str] = []
                                for ft in f.facility_types:
                                    extra.extend(
                                        _candidate_anchors_for_facility_type(
                                            facility_type=ft,
                                            snippet_by_anchor=snippet_by_anchor,
                                            catalog=catalog,
                                        )
                                    )
                                merged = _dedupe_anchor_ids(old_refs + extra, catalog=catalog, max_len=12)
                                if merged and merged != old_refs:
                                    f.source_refs = merged
                                    auto_corrections.append(
                                        {
                                            "field": "facilities.source_refs",
                                            "value": f"added anchors to support facility_types (types={f.facility_types})",
                                            "old_source_refs": old_refs,
                                            "new_source_refs": merged,
                                            "reason": "added anchors containing facility type keywords to support evidence validation",
                                        }
                                    )

                            # Ensure at least one meaningful extraction was produced.
                            has_any = bool(
                                artifact.borrowers
                                or artifact.guarantors
                                or artifact.agents
                                or artifact.arrangers
                                or artifact.lenders
                                or artifact.lenders_present_but_omitted
                                or artifact.facilities
                            )
                            if not has_any:
                                last_error = "output contained no extracted data (all arrays empty)"
                                continue

                            # Facilities: if currency is not explicitly supported by the cited anchors,
                            # null it out rather than failing the entire stage (no guessing). Record it.
                            for f in artifact.facilities:
                                if f.committed_amount and f.committed_amount.currency:
                                    try:
                                        _require_supported_currency(
                                            currency=f.committed_amount.currency,
                                            source_refs=f.source_refs,
                                            snippet_by_anchor=snippet_by_anchor,
                                        )
                                    except Exception as exc:
                                        old = f.committed_amount.currency
                                        f.committed_amount.currency = None
                                        auto_corrections.append(
                                            {
                                                "field": "facilities.committed_amount.currency",
                                                "value": old,
                                                "old_source_refs": list(f.source_refs),
                                                "new_source_refs": list(f.source_refs),
                                                "reason": f"nulled unsupported currency (no explicit evidence): {exc}",
                                            }
                                        )
                                if f.currency:
                                    try:
                                        _require_supported_currency(
                                            currency=f.currency,
                                            source_refs=f.source_refs,
                                            snippet_by_anchor=snippet_by_anchor,
                                        )
                                    except Exception as exc:
                                        old = f.currency
                                        f.currency = None
                                        auto_corrections.append(
                                            {
                                                "field": "facilities.currency",
                                                "value": old,
                                                "old_source_refs": list(f.source_refs),
                                                "new_source_refs": list(f.source_refs),
                                                "reason": f"nulled unsupported currency (no explicit evidence): {exc}",
                                            }
                                        )

                            # Validate that every referenced anchor exists in anchors.tsv.
                            def _check_anchors(anchor_ids: list[str], *, ctx: str) -> None:
                                missing = [a for a in anchor_ids if a not in catalog]
                                if missing:
                                    raise RuntimeError(f"{ctx}: unknown anchor_ids (examples): {missing[:10]}")

                            for p in artifact.borrowers:
                                _check_anchors(p.source_refs, ctx="borrowers.source_refs")
                            for p in artifact.guarantors:
                                _check_anchors(p.source_refs, ctx="guarantors.source_refs")
                            for r in artifact.agents:
                                _check_anchors(r.source_refs, ctx="agents.source_refs")
                            for r in artifact.arrangers:
                                _check_anchors(r.source_refs, ctx="arrangers.source_refs")
                            for p in artifact.lenders:
                                _check_anchors(p.source_refs, ctx="lenders.source_refs")
                            if artifact.lenders_source_refs:
                                _check_anchors(artifact.lenders_source_refs, ctx="lenders_source_refs")
                            for f in artifact.facilities:
                                _check_anchors(f.source_refs, ctx="facilities.source_refs")

                            # Validate that claimed values are supported by cited anchors (no guessing).
                            try:
                                for p in artifact.borrowers:
                                    _require_supported_string(
                                        field="borrowers.name",
                                        value=p.name,
                                        source_refs=p.source_refs,
                                        snippet_by_anchor=snippet_by_anchor,
                                    )
                                for p in artifact.guarantors:
                                    _require_supported_string(
                                        field="guarantors.name",
                                        value=p.name,
                                        source_refs=p.source_refs,
                                        snippet_by_anchor=snippet_by_anchor,
                                    )
                                for p in artifact.lenders:
                                    _require_supported_string(
                                        field="lenders.name",
                                        value=p.name,
                                        source_refs=p.source_refs,
                                        snippet_by_anchor=snippet_by_anchor,
                                    )
                                for r in artifact.agents:
                                    _require_supported_string(
                                        field="agents.party_name",
                                        value=r.party_name,
                                        source_refs=r.source_refs,
                                        snippet_by_anchor=snippet_by_anchor,
                                    )
                                    if not _contains_all_tokens(
                                        _evidence_text_for_refs(snippet_by_anchor, r.source_refs), r.role_label
                                    ):
                                        raise RuntimeError(
                                            f"agents.role_label {r.role_label!r} not supported by cited anchors {r.source_refs}"
                                        )
                                for r in artifact.arrangers:
                                    _require_supported_string(
                                        field="arrangers.party_name",
                                        value=r.party_name,
                                        source_refs=r.source_refs,
                                        snippet_by_anchor=snippet_by_anchor,
                                    )
                                    if not _contains_all_tokens(
                                        _evidence_text_for_refs(snippet_by_anchor, r.source_refs), r.role_label
                                    ):
                                        raise RuntimeError(
                                            f"arrangers.role_label {r.role_label!r} not supported by cited anchors {r.source_refs}"
                                        )

                                if artifact.lenders_present_but_omitted:
                                    # Conservative: only allow omission if evidence looks like an enumerated lender
                                    # list (grid/table OR signature-page lender list), not just a schedule reference.
                                    ok = False
                                    for aid in artifact.lenders_source_refs:
                                        rec = rec_by_anchor.get(aid) or {}
                                        snippet = str(rec.get("snippet") or "")
                                        if len(snippet) < 200:
                                            continue
                                        if _looks_like_enumerated_lender_list(snippet):
                                            ok = True
                                            break
                                    if not ok:
                                        raise RuntimeError(
                                            "lenders_present_but_omitted is true but cited anchors do not include an enumerated lender grid/list"
                                        )

                                for f in artifact.facilities:
                                    if f.facility_name:
                                        _require_supported_string(
                                            field="facilities.facility_name",
                                            value=f.facility_name,
                                            source_refs=f.source_refs,
                                            snippet_by_anchor=snippet_by_anchor,
                                        )
                                    _require_supported_facility_types(
                                        facility_types=f.facility_types,
                                        source_refs=f.source_refs,
                                        snippet_by_anchor=snippet_by_anchor,
                                    )
                                    if f.committed_amount and f.committed_amount.amount is not None:
                                        _require_supported_number(
                                            field="facilities.committed_amount.amount",
                                            value=float(f.committed_amount.amount),
                                            source_refs=f.source_refs,
                                            snippet_by_anchor=snippet_by_anchor,
                                        )
                                    if f.committed_amount and f.committed_amount.currency:
                                        _require_supported_currency(
                                            currency=f.committed_amount.currency,
                                            source_refs=f.source_refs,
                                            snippet_by_anchor=snippet_by_anchor,
                                        )
                                    if f.currency:
                                        _require_supported_currency(
                                            currency=f.currency,
                                            source_refs=f.source_refs,
                                            snippet_by_anchor=snippet_by_anchor,
                                        )
                                    if f.committed_amount and f.committed_amount.currency and f.currency:
                                        if f.committed_amount.currency.strip().upper() != f.currency.strip().upper():
                                            raise RuntimeError(
                                                "facilities: committed_amount.currency must match currency when both are provided"
                                            )
                                    if f.maturity_date:
                                        evidence = _evidence_text_for_refs(snippet_by_anchor, f.source_refs)
                                        if not _validate_iso_date_against_evidence(f.maturity_date, evidence):
                                            raise RuntimeError(
                                                f"maturity_date {f.maturity_date!r} not supported by cited anchors {f.source_refs}"
                                            )
                                    if f.termination_date:
                                        evidence = _evidence_text_for_refs(snippet_by_anchor, f.source_refs)
                                        if not _validate_iso_date_against_evidence(f.termination_date, evidence):
                                            raise RuntimeError(
                                                f"termination_date {f.termination_date!r} not supported by cited anchors {f.source_refs}"
                                            )
                            except Exception as exc:
                                last_error = f"evidence validation failed: {exc}"
                                continue

                            # Success.
                            _write_outputs(
                                artifact_json=artifact.model_dump(mode="json"),
                                attempts_used=attempt,
                                auto_corrections=auto_corrections,
                            )
                            return

                        failures.append((item_id, last_error))
                        err_path = out_dir / f"{item_id}.final.error.txt"
                        err_path.write_text(last_error)
                    except Exception as exc:
                        failures.append((item_id, str(exc)))
                        err_path = out_dir / f"{item_id}.error.txt"
                        err_path.write_text(f"{exc}\n\n{traceback.format_exc()}")

            await asyncio.gather(*(_process(i) for i in items))

        if failures:
            err_summary = out_dir / "errors.txt"
            err_summary.write_text("\n".join(f"{item}: {msg}" for item, msg in failures))
            raise RuntimeError(f"Agreement metadata extraction failed for {len(failures)} items; see {err_summary}")

    asyncio.run(_run_async())

    manifest_path = paths.manifest_path
    if manifest_path.exists():
        update_manifest(
            manifest_path,
            agreement_metadata_prompt=str(prompt_path),
            agreement_metadata_prompt_sha256=prompt_digest,
            agreement_metadata_output_subdir=resolved_output_subdir,
            agreement_metadata_categories=wanted_categories,
        )
