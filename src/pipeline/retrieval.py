from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from .config import Paths
from .indexing import IndexingNotImplemented


def render_snippets(paths: Paths, accessions: Iterable[str], bandwidth: int = 4) -> None:
    out_dir = paths.retrieval_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    for acc in accessions:
        anchor_json = paths.indexing_dir / f"{acc}.json"
        if not anchor_json.exists():
            raise FileNotFoundError(
                f"Missing anchor JSON for {acc}: {anchor_json}. Run indexing first (or implement it)."
            )
        pv = paths.prompt_views_dir / acc / "prompt_view.txt"
        if not pv.exists():
            raise FileNotFoundError(f"Missing prompt_view for {acc}: {pv}")
        # Stub: without anchor JSON format, we cannot render snippets.
        raise IndexingNotImplemented(
            "Snippet rendering requires anchor JSON. Implement retrieval once indexing is available."
        )

