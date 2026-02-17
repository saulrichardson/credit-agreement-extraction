from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Dict, Sequence

import yaml

from pipeline.core.config import DocumentSelection


DocFilter = Callable[[Dict[str, Any], Dict[str, Any]], bool]


def _normalize_sequence(seq_raw: str) -> str:
    seq_raw = str(seq_raw or "").strip()
    if not seq_raw:
        return ""
    try:
        # EDGAR <SEQUENCE> is numeric; normalize "02" -> "2".
        return str(int(seq_raw))
    except ValueError:
        return seq_raw.lstrip("0") or seq_raw


def _load_json_or_yaml(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    if path.suffix.lower() in {".yml", ".yaml"}:
        return yaml.safe_load(path.read_text())
    return json.loads(path.read_text())


def _load_item_ids_from_file(path: Path) -> list[str]:
    raw = _load_json_or_yaml(path)
    if isinstance(raw, dict):
        item_ids = raw.get("item_ids")
    else:
        item_ids = raw

    if not isinstance(item_ids, list) or not item_ids:
        raise ValueError(f"{path} must contain a non-empty item_ids list")

    normalized: list[str] = []
    for idx, raw_item in enumerate(item_ids):
        value = str(raw_item or "").strip()
        if not value:
            raise ValueError(f"{path} item_ids[{idx}] is empty")
        normalized.append(value)
    return normalized


def _allowed_accession_sequences_from_item_ids(item_ids: Sequence[str]) -> dict[str, set[str]]:
    allowed: dict[str, set[str]] = {}
    for raw in item_ids:
        item_id = str(raw or "").strip()
        if not item_id:
            continue
        parts = item_id.split("_")
        if len(parts) < 2:
            raise ValueError(f"Invalid item_id (expected '<accession>_<sequence>'): {item_id!r}")
        accession = parts[0].strip()
        seq_raw = "_".join(parts[1:]).strip()
        seq = _normalize_sequence(seq_raw)
        if not accession or not seq:
            raise ValueError(f"Invalid item_id (empty accession/sequence after parsing): {item_id!r}")
        allowed.setdefault(accession, set()).add(seq)
    if not allowed:
        raise ValueError("No valid item_ids provided (empty allowlist).")
    return allowed


def _normalize_doc_type_prefixes(prefixes: Sequence[str]) -> tuple[str, ...]:
    normalized: list[str] = []
    seen: set[str] = set()
    for raw in prefixes:
        value = str(raw or "").strip().upper()
        if not value or value in seen:
            continue
        seen.add(value)
        normalized.append(value)
    return tuple(normalized)


def build_document_selection(
    *,
    item_ids_file: str | None,
    doc_type_prefixes: Sequence[str] | None,
) -> tuple[DocumentSelection, DocFilter]:
    item_ids: tuple[str, ...] = ()
    allowed_sequences: dict[str, set[str]] | None = None
    if item_ids_file:
        source_path = Path(item_ids_file)
        item_ids = tuple(_load_item_ids_from_file(source_path))
        allowed_sequences = _allowed_accession_sequences_from_item_ids(item_ids)
        item_ids_source = str(source_path)
    else:
        item_ids_source = None

    normalized_prefixes = _normalize_doc_type_prefixes(doc_type_prefixes or ())
    if doc_type_prefixes and not normalized_prefixes:
        raise ValueError("doc_type_prefixes must contain at least one non-empty string")

    selection = DocumentSelection(
        item_ids_source=item_ids_source,
        item_ids=item_ids,
        doc_type_prefixes=normalized_prefixes,
    )

    if allowed_sequences is None and not normalized_prefixes:
        return selection, lambda _submission, _document: True

    def _predicate(submission: Dict[str, Any], document: Dict[str, Any]) -> bool:
        if allowed_sequences is not None:
            accession = str(submission.get("accession") or "").strip()
            seq_raw = str(document.get("sequence") or "").strip()
            if not accession or not seq_raw:
                return False
            seq = _normalize_sequence(seq_raw)
            permitted = allowed_sequences.get(accession)
            if not permitted or seq not in permitted:
                return False

        if normalized_prefixes:
            doc_type = str(document.get("type") or "").strip().upper()
            if not doc_type:
                return False
            if not any(doc_type.startswith(prefix) for prefix in normalized_prefixes):
                return False

        return True

    return selection, _predicate


def serialize_document_selection(selection: DocumentSelection) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "mode": "allow_all",
        "item_ids_source": selection.item_ids_source,
        "item_ids": list(selection.item_ids),
        "doc_type_prefixes": list(selection.doc_type_prefixes),
    }
    if selection.item_ids or selection.doc_type_prefixes:
        out["mode"] = "restricted"
    return out
