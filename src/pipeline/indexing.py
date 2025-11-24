from __future__ import annotations

from pathlib import Path
from typing import Iterable

from .config import Paths, prompt_hash, update_manifest
from .utils import assert_exists


class IndexingNotImplemented(RuntimeError):
    pass


def run_indexing(paths: Paths, item_ids: Iterable[str], prompt_path: Path) -> None:
    assert_exists(prompt_path, message=f"Indexing prompt not found: {prompt_path}")
    out_dir = paths.indexing_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    prompt_digest = prompt_hash(prompt_path)

    for item_id in item_ids:
        pv = assert_exists(paths.prompt_views_dir / item_id / "prompt_view.txt")
        raise IndexingNotImplemented("Indexing not implemented. Plug in your LLM/gateway call here.")

    manifest_path = paths.manifest_path
    if manifest_path.exists():
        update_manifest(
            manifest_path,
            indexing_prompt=str(prompt_path),
            indexing_prompt_sha256=prompt_digest,
        )
