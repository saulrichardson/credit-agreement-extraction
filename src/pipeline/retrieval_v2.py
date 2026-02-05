from __future__ import annotations

import json
from typing import Iterable, Dict, Any

from .anchors import load_anchor_catalog
from .config import Paths
from .schemas_v2 import IndexingSelectionV2Artifact
from .utils import assert_exists, load_manifest, prompt_view_path


def _window(text: str, start: int, end: int, bandwidth_chars: int) -> Dict[str, Any]:
    a = max(0, start - bandwidth_chars)
    b = min(len(text), end + bandwidth_chars)
    return {"snippet": text[a:b], "snippet_start": a, "snippet_end": b}


def render_snippets_v2(paths: Paths, item_ids: Iterable[str], bandwidth: int = 400) -> None:
    out_dir = paths.run_dir / "retrieval_v2"
    out_dir.mkdir(parents=True, exist_ok=True)

    toc_subdir: str | None = None
    toc_item_ids: set[str] | None = None
    toc_manifest: dict | None = None
    if paths.manifest_path.exists():
        toc_manifest = load_manifest(paths.manifest_path)
        toc_subdir = toc_manifest.get("toc_v1_output_subdir") if isinstance(toc_manifest, dict) else None
        if isinstance(toc_subdir, str) and not toc_subdir.strip():
            toc_subdir = None
        if toc_subdir and isinstance(toc_manifest, dict):
            raw_ids = toc_manifest.get("toc_v1_item_ids")
            if raw_ids is None:
                toc_item_ids = None
            elif isinstance(raw_ids, list) and all(isinstance(v, str) for v in raw_ids):
                toc_item_ids = {v for v in raw_ids if v.strip()}
            else:
                raise RuntimeError("manifest toc_v1_item_ids must be a list of strings when provided")

    for item_id in item_ids:
        selection_json = assert_exists(
            paths.run_dir / "indexing_v2" / f"{item_id}_anchors.json",
            message=f"Missing anchor JSON for {item_id}: run index-v2 first.",
        )
        pv = prompt_view_path(paths, item_id)

        text = pv.read_text()
        selection_doc = json.loads(selection_json.read_text())
        artifact = IndexingSelectionV2Artifact.model_validate(selection_doc)
        selection = artifact.selection

        catalog = load_anchor_catalog(paths, item_id)

        # Optional/strict: attach TOC chunk title/ids to each snippet for higher-level retrieval context.
        # If the manifest indicates toc-v1 has been run, we treat missing/invalid TOC as an error (no silent failure).
        toc_by_order: list[dict] = []
        if toc_subdir:
            toc_path = paths.run_dir / "toc_v1" / toc_subdir / f"{item_id}.json"
            if not toc_path.exists():
                toc_expected = toc_item_ids is None or item_id in toc_item_ids
                if toc_expected:
                    raise RuntimeError(
                        f"TOC missing for {item_id}: expected {toc_path} because manifest indicates toc-v1 ran for this item "
                        f"(toc_v1_output_subdir={toc_subdir!r})"
                    )
            else:
                try:
                    toc_doc = json.loads(toc_path.read_text())
                except json.JSONDecodeError as exc:
                    raise RuntimeError(f"TOC is not valid JSON for {item_id}: {toc_path}") from exc
                if not isinstance(toc_doc, dict):
                    raise RuntimeError(f"TOC must be a JSON object for {item_id}: {toc_path}")
                toc_chunks = toc_doc.get("chunks")
                if not isinstance(toc_chunks, list) or not toc_chunks:
                    raise RuntimeError(f"TOC chunks missing/empty for {item_id}: {toc_path}")
                for ch in toc_chunks:
                    if not isinstance(ch, dict):
                        raise RuntimeError(f"TOC chunk entry must be an object for {item_id}: {toc_path}")
                    if not isinstance(ch.get("start_order"), int) or not isinstance(ch.get("end_order"), int):
                        raise RuntimeError(f"TOC chunk missing start_order/end_order for {item_id}: {toc_path}")
                    if not isinstance(ch.get("chunk_id"), int) or not isinstance(ch.get("title"), str):
                        raise RuntimeError(f"TOC chunk missing chunk_id/title for {item_id}: {toc_path}")
                    toc_by_order.append(ch)

        def _toc_for_anchor(aid: str) -> dict | None:
            if not toc_by_order:
                return None
            info = catalog.get(aid)
            if not info:
                return None
            order = int(info["order"])
            for ch in toc_by_order:
                if int(ch["start_order"]) <= order <= int(ch["end_order"]):
                    return ch
            return None

        categories_by_anchor: dict[str, list[str]] = {}
        buckets_by_anchor: dict[str, set[str]] = {}

        def _add(anchor_id: str, bucket: str) -> None:
            buckets_by_anchor.setdefault(anchor_id, set()).add(bucket)

        for anchor_id in selection.metadata_anchors:
            _add(anchor_id, "metadata")
        for anchor_id in selection.agreement_date_anchors:
            _add(anchor_id, "agreement_dates")
        for anchor_id in selection.fundamental_anchors:
            _add(anchor_id, "fundamental")
        for anchor_id in selection.key_date_definitions_anchors:
            # Keep these explicitly-tagged anchors available to downstream fundamentals extraction,
            # while also merging them into the core "fundamental" bucket so default filters include them.
            _add(anchor_id, "key_date_definitions")
            _add(anchor_id, "fundamental")
        for anchor_id in selection.pricing_anchors:
            _add(anchor_id, "pricing")
        for anchor_id in selection.base_rate_anchors:
            _add(anchor_id, "base_rate")
        for anchor_id in selection.spread_anchors:
            _add(anchor_id, "spread")
        for anchor_id in selection.fee_anchors:
            _add(anchor_id, "fee")
        for anchor_id in selection.financial_covenant_anchors:
            _add(anchor_id, "financial_covenant")

        # Optional: include the full definitions section span as a separate bucket when provided.
        # This is a pure expansion of the anchor range; downstream consumers can decide whether
        # to use it (do not feed it to an LLM by default).
        if selection.definitions_anchor_range is not None:
            dr = selection.definitions_anchor_range
            if dr.start_anchor not in catalog:
                raise RuntimeError(f"definitions_anchor_range.start_anchor {dr.start_anchor} not found in catalog for {item_id}")
            if dr.end_anchor not in catalog:
                raise RuntimeError(f"definitions_anchor_range.end_anchor {dr.end_anchor} not found in catalog for {item_id}")
            start_order = int(catalog[dr.start_anchor]["order"])
            end_order = int(catalog[dr.end_anchor]["order"])
            if start_order > end_order:
                raise RuntimeError(
                    f"definitions_anchor_range has start after end for {item_id}: "
                    f"{dr.start_anchor} (order={start_order}) > {dr.end_anchor} (order={end_order})"
                )

            ordered = sorted(catalog.values(), key=lambda a: int(a["order"]))
            for info in ordered[start_order : end_order + 1]:
                _add(str(info["anchor_id"]), "definitions")
        for aid, buckets in buckets_by_anchor.items():
            categories_by_anchor[aid] = sorted({b for b in buckets})

        if not categories_by_anchor:
            raise RuntimeError(f"No anchors selected for {item_id} in {selection_json}")

        def _order(aid: str) -> int:
            info = catalog.get(aid)
            if not info:
                raise RuntimeError(f"Indexing v2 selected unknown anchor_id {aid} for {item_id}")
            return int(info["order"])

        ordered_anchor_ids = sorted(categories_by_anchor.keys(), key=_order)

        out_file = out_dir / f"{item_id}_snippets.jsonl"
        with out_file.open("w") as fh:
            for anchor_id in ordered_anchor_ids:
                info = catalog[anchor_id]
                start = int(info["start"])
                end = int(info["end"])
                anchor_type = str(info["anchor_type"])
                categories = categories_by_anchor.get(anchor_id) or []
                label = ",".join(categories) if categories else anchor_type
                window = _window(text, start, end, bandwidth_chars=bandwidth)
                toc = _toc_for_anchor(anchor_id)
                rec = {
                    "item_id": item_id,
                    "anchor_id": anchor_id,
                    "categories": categories,
                    "label": label,
                    "type": anchor_type,
                    "start": start,
                    "end": end,
                    "buckets": sorted(buckets_by_anchor.get(anchor_id, set())),
                    "toc_chunk_id": int(toc["chunk_id"]) if toc else None,
                    "toc_title": str(toc["title"]) if toc else None,
                    **window,
                }
                fh.write(json.dumps(rec) + "\n")
