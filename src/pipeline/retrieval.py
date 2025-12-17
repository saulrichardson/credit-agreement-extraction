from __future__ import annotations

import json
from typing import Iterable, Dict, Any

from .anchors import load_anchor_catalog
from .config import Paths
from .schemas import IndexingSelectionArtifact
from .utils import assert_exists, prompt_view_path


def _window(text: str, start: int, end: int, bandwidth_chars: int) -> Dict[str, Any]:
    a = max(0, start - bandwidth_chars)
    b = min(len(text), end + bandwidth_chars)
    return {"snippet": text[a:b], "snippet_start": a, "snippet_end": b}


def render_snippets(paths: Paths, item_ids: Iterable[str], bandwidth: int = 400) -> None:
    out_dir = paths.retrieval_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    for item_id in item_ids:
        selection_json = assert_exists(
            paths.indexing_dir / f"{item_id}_anchors.json",
            message=f"Missing anchor JSON for {item_id}: run indexing first.",
        )
        pv = prompt_view_path(paths, item_id)

        text = pv.read_text()
        selection_doc = json.loads(selection_json.read_text())
        artifact = IndexingSelectionArtifact.model_validate(selection_doc)
        selection = artifact.selection

        catalog = load_anchor_catalog(paths, item_id)

        categories_by_anchor: dict[str, list[str]] = {}
        for aid in selection.fundamental_anchors:
            categories_by_anchor.setdefault(aid, []).append("fundamental")
        for aid in selection.pricing_anchors:
            categories_by_anchor.setdefault(aid, []).append("pricing")
        for aid in selection.financial_covenant_anchors:
            categories_by_anchor.setdefault(aid, []).append("financial_covenant")

        if not categories_by_anchor:
            raise RuntimeError(f"No anchors selected for {item_id} in {selection_json}")

        def _order(aid: str) -> int:
            info = catalog.get(aid)
            if not info:
                raise RuntimeError(f"Indexing selected unknown anchor_id {aid} for {item_id}")
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
                rec = {
                    "item_id": item_id,
                    "anchor_id": anchor_id,
                    "categories": categories,
                    "label": label,
                    "type": anchor_type,
                    "start": start,
                    "end": end,
                    **window,
                }
                fh.write(json.dumps(rec) + "\n")
