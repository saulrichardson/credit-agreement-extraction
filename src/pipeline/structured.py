from __future__ import annotations

from pathlib import Path
from typing import Iterable

from .config import Paths, prompt_hash, update_manifest
from .utils import assert_exists


class StructuredNotImplemented(RuntimeError):
    pass


def run_structured(paths: Paths, item_ids: Iterable[str], prompt_path: Path) -> None:
    assert_exists(prompt_path, message=f"Structured prompt not found: {prompt_path}")
    out_dir = paths.structured_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    prompt_digest = prompt_hash(prompt_path)

    for item_id in item_ids:
        snippets_path = paths.retrieval_dir / f"{item_id}_snippets.jsonl"
        assert_exists(snippets_path, message=f"Missing snippets for {item_id}: run retrieve first.")
        raise StructuredNotImplemented("Structured extraction not implemented. Integrate your LLM/gateway here.")

    manifest_path = paths.manifest_path
    if manifest_path.exists():
        update_manifest(
            manifest_path,
            structured_prompt=str(prompt_path),
            structured_prompt_sha256=prompt_digest,
        )
