from __future__ import annotations

from typing import Iterable

from .config import Paths


def run_validation(paths: Paths, accessions: Iterable[str]) -> None:
    # Placeholder for QA, reverse highlighter, etc.
    val_dir = paths.validation_dir
    val_dir.mkdir(parents=True, exist_ok=True)
    raise RuntimeError("Validation not implemented yet. Add QA checks here.")
