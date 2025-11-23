from __future__ import annotations

from pathlib import Path
from typing import Iterable

from .config import Paths, prompt_hash
from .indexing import IndexingNotImplemented


def run_structured(paths: Paths, accessions: Iterable[str], prompt_path: Path) -> None:
    if not prompt_path.exists():
        raise FileNotFoundError(f"Structured prompt not found: {prompt_path}")
    out_dir = paths.structured_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    prompt_digest = prompt_hash(prompt_path)

    for acc in accessions:
        snippets = paths.retrieval_dir / f"{acc}_snippets.txt"
        if not snippets.exists():
            raise FileNotFoundError(f"Missing snippets for {acc}: {snippets}")
        raise IndexingNotImplemented(
            "Structured extraction not implemented. Integrate your LLM/gateway call here."
        )

    # Could record prompt hash in manifest if desired

