from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from .config import Paths, prompt_hash, record_manifest


class IndexingNotImplemented(RuntimeError):
    pass


def run_indexing(paths: Paths, accessions: Iterable[str], prompt_path: Path) -> None:
    if not prompt_path.exists():
        raise FileNotFoundError(f"Indexing prompt not found: {prompt_path}")
    out_dir = paths.indexing_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    prompt_digest = prompt_hash(prompt_path)

    for acc in accessions:
        pv = paths.prompt_views_dir / acc / "prompt_view.txt"
        if not pv.exists():
            raise FileNotFoundError(f"Missing prompt_view for {acc}: {pv}")
        # Stub: raise to indicate integration point
        raise IndexingNotImplemented(
            "Indexing is not implemented. Plug in your LLM/gateway call here and write anchors JSON."
        )

    # Update manifest with prompt info
    manifest_path = paths.manifest_path
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        manifest["indexing_prompt"] = str(prompt_path)
        manifest["indexing_prompt_sha256"] = prompt_digest
        record_manifest(manifest_path, manifest)

