from __future__ import annotations

import importlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Callable, Any

import yaml

from .config import FilterSpec


def load_filter_spec(path: Path) -> FilterSpec:
    if not path.exists():
        raise FileNotFoundError(f"Filter spec not found: {path}")
    if path.suffix.lower() in {".yml", ".yaml"}:
        data = yaml.safe_load(path.read_text())
    else:
        data = json.loads(path.read_text())
    if not isinstance(data, dict):
        raise ValueError("Filter spec must be a mapping")
    return FilterSpec.from_mapping(data)


def serialize_filter_spec(spec: FilterSpec) -> Dict:
    return {k: v for k, v in asdict(spec).items() if v is not None}


def keep_all(submission: Dict[str, Any], document: Dict[str, Any]) -> bool:
    """Accept every document; used as the default when no filter is provided."""
    return True


def _normalize_sequence(seq_raw: str) -> str:
    seq_raw = str(seq_raw or "").strip()
    if not seq_raw:
        return ""
    try:
        # EDGAR <SEQUENCE> is numeric. Normalize "02" -> "2" to reduce brittle matching.
        return str(int(seq_raw))
    except ValueError:
        return seq_raw.lstrip("0") or seq_raw


def _load_json_or_yaml(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    if path.suffix.lower() in {".yml", ".yaml"}:
        data = yaml.safe_load(path.read_text())
    else:
        data = json.loads(path.read_text())
    return data


def _allowed_accession_sequences_from_item_ids(item_ids: list[str]) -> dict[str, set[str]]:
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


def keep_item_ids(item_ids: list[str]) -> Callable[[Dict[str, Any], Dict[str, Any]], bool]:
    """Factory: keep only a fixed allowlist of item_ids (accession + sequence).

    item_id format is `<accession>_<sequence>` where sequence is normalized as an integer string
    (e.g., "02" becomes "2") to match EDGAR <SEQUENCE> semantics.
    """

    if not isinstance(item_ids, list) or not item_ids:
        raise ValueError("item_ids must be a non-empty list")

    allowed = _allowed_accession_sequences_from_item_ids([str(x) for x in item_ids])

    def _predicate(submission: Dict[str, Any], document: Dict[str, Any]) -> bool:
        accession = str(submission.get("accession") or "").strip()
        seq_raw = str(document.get("sequence") or "").strip()
        if not accession or not seq_raw:
            return False
        seq = _normalize_sequence(seq_raw)
        permitted = allowed.get(accession)
        return bool(permitted and seq in permitted)

    return _predicate


def keep_item_ids_from_file(path: str) -> Callable[[Dict[str, Any], Dict[str, Any]], bool]:
    """Factory: keep only item_ids loaded from a JSON/YAML file.

    Supported formats:
      - mapping with `item_ids: [...]` (recommended; can include metadata fields)
      - bare list: ["<accession>_<sequence>", ...]
    """

    file_path = Path(path)
    data = _load_json_or_yaml(file_path)

    item_ids: Any
    if isinstance(data, dict):
        item_ids = data.get("item_ids")
    else:
        item_ids = data

    if not isinstance(item_ids, list) or not item_ids:
        raise ValueError(f"{file_path} must contain a non-empty item_ids list")

    return keep_item_ids([str(x) for x in item_ids])


def keep_doc_type_prefixes(prefixes: list[str]) -> Callable[[Dict[str, Any], Dict[str, Any]], bool]:
    """Factory: keep only documents whose <TYPE> starts with one of the provided prefixes.

    Example: prefixes=["EX-10"] matches EX-10, EX-10.1, EX-10.01, etc.
    """

    if not isinstance(prefixes, list) or not prefixes:
        raise ValueError("prefixes must be a non-empty list")
    normalized = [str(p).strip().upper() for p in prefixes if str(p).strip()]
    if not normalized:
        raise ValueError("prefixes must contain at least one non-empty string")

    def _predicate(submission: Dict[str, Any], document: Dict[str, Any]) -> bool:
        doc_type = str(document.get("type") or "").strip().upper()
        if not doc_type:
            return False
        return any(doc_type.startswith(prefix) for prefix in normalized)

    return _predicate


def all_of(filters: list[Dict[str, Any]]) -> Callable[[Dict[str, Any], Dict[str, Any]], bool]:
    """Factory: boolean AND over nested filter specs.

    `filters` is a list of filter spec mappings, each like:
      {"doc_filter_path": "module:function", "doc_filter_kwargs": {...}}
    """

    if not isinstance(filters, list) or not filters:
        raise ValueError("filters must be a non-empty list of filter spec mappings")

    predicates: list[Callable[[Dict[str, Any], Dict[str, Any]], bool]] = []
    for idx, mapping in enumerate(filters):
        if not isinstance(mapping, dict):
            raise ValueError(f"filters[{idx}] must be a mapping, got {type(mapping).__name__}")
        spec = FilterSpec.from_mapping(mapping)
        predicates.append(load_doc_filter(spec))

    def _predicate(submission: Dict[str, Any], document: Dict[str, Any]) -> bool:
        return all(pred(submission, document) for pred in predicates)

    return _predicate


def any_of(filters: list[Dict[str, Any]]) -> Callable[[Dict[str, Any], Dict[str, Any]], bool]:
    """Factory: boolean OR over nested filter specs."""

    if not isinstance(filters, list) or not filters:
        raise ValueError("filters must be a non-empty list of filter spec mappings")

    predicates: list[Callable[[Dict[str, Any], Dict[str, Any]], bool]] = []
    for idx, mapping in enumerate(filters):
        if not isinstance(mapping, dict):
            raise ValueError(f"filters[{idx}] must be a mapping, got {type(mapping).__name__}")
        spec = FilterSpec.from_mapping(mapping)
        predicates.append(load_doc_filter(spec))

    def _predicate(submission: Dict[str, Any], document: Dict[str, Any]) -> bool:
        return any(pred(submission, document) for pred in predicates)

    return _predicate


def not_(filter: Dict[str, Any]) -> Callable[[Dict[str, Any], Dict[str, Any]], bool]:
    """Factory: boolean NOT over a nested filter spec.

    `filter` is a filter spec mapping like:
      {"doc_filter_path": "module:function", "doc_filter_kwargs": {...}}
    """

    if not isinstance(filter, dict):
        raise ValueError(f"filter must be a mapping, got {type(filter).__name__}")
    spec = FilterSpec.from_mapping(filter)
    pred = load_doc_filter(spec)

    def _predicate(submission: Dict[str, Any], document: Dict[str, Any]) -> bool:
        return not pred(submission, document)

    return _predicate


def keep_item_ids_from_manifest(manifest_path: str) -> Callable[[Dict[str, Any], Dict[str, Any]], bool]:
    """Factory: keep only (accession, sequence) pairs listed in another run's manifest.

    Intended for "curate a sample run once, then re-run downstream stages deterministically".

    Expects a manifest with an `items` list containing `accession` + `sequence` per item.
    """

    path = Path(manifest_path)
    if not path.exists():
        raise FileNotFoundError(f"Manifest not found: {path}")

    manifest = json.loads(path.read_text())
    items = manifest.get("items")
    if not isinstance(items, list) or not items:
        raise ValueError(f"Manifest contains no items: {path}")

    allowed: dict[str, set[str]] = {}
    for item in items:
        if not isinstance(item, dict):
            raise ValueError(f"Manifest item must be a mapping; got {type(item).__name__}: {item!r}")
        acc = str(item.get("accession") or "").strip()
        seq_raw = str(item.get("sequence") or "").strip()
        if not acc or not seq_raw:
            # Fall back to parsing item_id when accession/sequence aren't present.
            item_id = str(item.get("item_id") or "").strip()
            if "_" not in item_id:
                raise ValueError(f"Manifest item missing accession/sequence and unparseable item_id: {item!r}")
            acc, seq_raw = item_id.split("_", 1)
        seq = _normalize_sequence(seq_raw)
        if not seq:
            raise ValueError(f"Manifest item has empty sequence: {item!r}")
        allowed.setdefault(acc, set()).add(seq)

    if not allowed:
        raise ValueError(f"No allowed (accession, sequence) pairs derived from manifest: {path}")

    def _predicate(submission: Dict[str, Any], document: Dict[str, Any]) -> bool:
        accession = str(submission.get("accession") or "").strip()
        seq_raw = str(document.get("sequence") or "").strip()
        if not accession or not seq_raw:
            return False
        seq = _normalize_sequence(seq_raw)
        permitted = allowed.get(accession)
        return bool(permitted and seq in permitted)

    return _predicate


def load_doc_filter(spec: FilterSpec) -> Callable[[Dict[str, Any], Dict[str, Any]], bool]:
    if ":" not in spec.doc_filter_path:
        raise ValueError("doc_filter_path must be 'module:function'")
    module_name, func_name = spec.doc_filter_path.split(":", 1)
    module = importlib.import_module(module_name)
    func = getattr(module, func_name, None)
    if func is None or not callable(func):
        raise ValueError(f"Callable {spec.doc_filter_path} not found")
    if spec.doc_filter_kwargs is None:
        return func

    try:
        predicate = func(**spec.doc_filter_kwargs)
    except TypeError as exc:
        raise TypeError(
            f"{spec.doc_filter_path} could not be called as a factory with doc_filter_kwargs. "
            f"If your filter is a predicate (submission, document)->bool, omit doc_filter_kwargs. "
            f"kwargs={spec.doc_filter_kwargs!r}"
        ) from exc
    if predicate is None or not callable(predicate):
        raise ValueError(
            f"{spec.doc_filter_path} returned a non-callable predicate: {predicate!r}. "
            "Factories must return (submission, document)->bool."
        )
    return predicate
