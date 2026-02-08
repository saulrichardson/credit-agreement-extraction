from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Set, Tuple

from pipeline.ir.contract_ir_v0_2 import ContractIRValidationError, validate_contract_ir


class ContractIRMergeError(ValueError):
    pass


@dataclass(frozen=True)
class MergeConflict:
    code: str
    message: str
    path: str


def merge_contract_ir_v0_2(docs: Iterable[Mapping[str, Any]]) -> Dict[str, Any]:
    """Deterministically merge multiple ContractIR v0.2 documents.

    This is intended for the "subtask prompts + deterministic merge" strategy:
      - base_rate pass (derived only; tables empty)
      - spread pass (tables + derived)
      - fee_rate pass (tables + derived)

    Hard merge rules (fail fast):
      - Every input must validate as ContractIR v0.2.
      - contract_id must match across docs.
      - table_id and fn_id must be unique across docs (no implicit overwrites).
      - indices must not conflict on series_id (unit must match).

    Sources strategy:
      - Emit a single excerpt_pack source whose anchor_ids is the union of all input sources[*].anchor_ids.
      - This avoids source_id collisions (prompts typically use "S1" everywhere) and preserves provenance
        via derived.source_refs / row.source_refs.
    """

    docs_list = [dict(d) for d in docs]
    if not docs_list:
        raise ContractIRMergeError("No documents provided for merge")

    conflicts: List[MergeConflict] = []
    validated_docs: List[Dict[str, Any]] = []

    for i, doc in enumerate(docs_list):
        errs = validate_contract_ir(doc)
        if errs:
            first = errs[0]
            raise ContractIRMergeError(
                f"Input doc #{i} failed ContractIR v0.2 validation at {first.json_path}: {first.message}"
            )
        validated_docs.append(doc)

    contract_ids = {str(d.get("contract_id") or "") for d in validated_docs}
    contract_ids.discard("")
    if len(contract_ids) != 1:
        raise ContractIRMergeError(f"Expected exactly one contract_id across docs, got {sorted(contract_ids)}")
    contract_id = next(iter(contract_ids))

    # ---- sources: union anchor ids ----------------------------------------------------
    item_ids: Set[str] = set()
    anchor_ids: Set[str] = set()
    for d in validated_docs:
        for s in d.get("sources", []) or []:
            if not isinstance(s, dict):
                continue
            item_id = s.get("item_id")
            if isinstance(item_id, str) and item_id:
                item_ids.add(item_id)
            for aid in s.get("anchor_ids", []) or []:
                if isinstance(aid, str) and aid:
                    anchor_ids.add(aid)
    if len(item_ids) > 1:
        raise ContractIRMergeError(f"Expected a single item_id across sources, got {sorted(item_ids)}")
    item_id = next(iter(item_ids)) if item_ids else contract_id

    merged_sources: List[Dict[str, Any]] = [
        {
            "source_id": "S1",
            "kind": "excerpt_pack",
            "item_id": item_id,
            "anchor_ids": sorted(anchor_ids),
            "notes": None,
        }
    ]

    # ---- indices ----------------------------------------------------
    by_series: Dict[str, Dict[str, Any]] = {}
    for d in validated_docs:
        for idx in d.get("indices", []) or []:
            if not isinstance(idx, dict):
                continue
            sid = idx.get("series_id")
            if not isinstance(sid, str) or not sid:
                continue
            if sid in by_series:
                prev = by_series[sid]
                if prev.get("unit") != idx.get("unit"):
                    conflicts.append(
                        MergeConflict(
                            code="index_unit_conflict",
                            message=f"Series {sid!r} unit mismatch: {prev.get('unit')!r} vs {idx.get('unit')!r}",
                            path=f"/indices/{sid}",
                        )
                    )
                continue
            by_series[sid] = dict(idx)

    # ---- tables -----------------------------------------------------
    by_table_id: Dict[str, Dict[str, Any]] = {}
    for d in validated_docs:
        for t in d.get("tables", []) or []:
            if not isinstance(t, dict):
                continue
            tid = t.get("table_id")
            if not isinstance(tid, str) or not tid:
                continue
            if tid in by_table_id:
                conflicts.append(
                    MergeConflict(
                        code="table_id_conflict",
                        message=f"Duplicate table_id {tid!r} across merged docs",
                        path=f"/tables/{tid}",
                    )
                )
                continue
            by_table_id[tid] = dict(t)

    # ---- derived ----------------------------------------------------
    by_fn_id: Dict[str, Dict[str, Any]] = {}
    for d in validated_docs:
        for fn in d.get("derived", []) or []:
            if not isinstance(fn, dict):
                continue
            fid = fn.get("fn_id")
            if not isinstance(fid, str) or not fid:
                continue
            if fid in by_fn_id:
                conflicts.append(
                    MergeConflict(
                        code="fn_id_conflict",
                        message=f"Duplicate fn_id {fid!r} across merged docs",
                        path=f"/derived/{fid}",
                    )
                )
                continue
            by_fn_id[fid] = dict(fn)

    # ---- open_items -------------------------------------------------
    merged_open_items: List[Dict[str, Any]] = []
    for d in validated_docs:
        for oi in d.get("open_items", []) or []:
            if isinstance(oi, dict):
                merged_open_items.append(dict(oi))

    if conflicts:
        first = conflicts[0]
        raise ContractIRMergeError(f"Merge conflict {first.code} at {first.path}: {first.message}")

    merged_doc: Dict[str, Any] = {
        "schema_version": "contract_ir_v0_2",
        "contract_id": contract_id,
        "sources": merged_sources,
        "indices": [by_series[k] for k in sorted(by_series.keys())],
        "tables": [by_table_id[k] for k in sorted(by_table_id.keys())],
        "derived": [by_fn_id[k] for k in sorted(by_fn_id.keys())],
        "open_items": merged_open_items,
    }

    # Sanity: merged doc must validate.
    merged_errs: List[ContractIRValidationError] = validate_contract_ir(merged_doc)
    if merged_errs:
        first = merged_errs[0]
        raise ContractIRMergeError(
            f"Merged doc failed ContractIR v0.2 validation at {first.json_path}: {first.message}"
        )

    return merged_doc

