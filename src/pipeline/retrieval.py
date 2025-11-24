from __future__ import annotations

import json
from typing import Iterable, Dict, Any

from .config import Paths
from .utils import assert_exists


def _window(text: str, start: int, end: int, bandwidth_chars: int) -> Dict[str, Any]:
    a = max(0, start - bandwidth_chars)
    b = min(len(text), end + bandwidth_chars)
    return {"snippet": text[a:b], "snippet_start": a, "snippet_end": b}


def render_snippets(paths: Paths, item_ids: Iterable[str], bandwidth: int = 400) -> None:
    out_dir = paths.retrieval_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    for item_id in item_ids:
        anchor_json = assert_exists(
            paths.indexing_dir / f"{item_id}_anchors.json",
            message=f"Missing anchor JSON for {item_id}: run indexing first.",
        )
        pv = assert_exists(paths.prompt_views_dir / item_id / "prompt_view.txt")

        text = pv.read_text()
        anchors_doc = json.loads(anchor_json.read_text())
        anchors = anchors_doc.get("anchors") or []
        if not anchors:
            raise RuntimeError(f"No anchors found in {anchor_json}")

        out_file = out_dir / f"{item_id}_snippets.jsonl"
        with out_file.open("w") as fh:
            for anchor in anchors:
                start = int(anchor.get("start", 0))
                end = int(anchor.get("end", 0))
                window = _window(text, start, end, bandwidth_chars=bandwidth)
                rec = {
                    "item_id": item_id,
                    "anchor_id": anchor.get("anchor_id"),
                    "label": anchor.get("label"),
                    "type": anchor.get("type"),
                    "start": start,
                    "end": end,
                    **window,
                }
                fh.write(json.dumps(rec) + "\n")
