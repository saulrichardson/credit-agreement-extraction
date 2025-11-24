from __future__ import annotations

import json
from pathlib import Path
import re
from typing import List, Dict, Any

from .config import Paths


def load_manifest(path: Path) -> Dict:
    if not path.exists():
        raise FileNotFoundError(f"Manifest not found: {path}")
    return json.loads(path.read_text())


def manifest_accessions(manifest: Dict) -> List[str]:
    accs = [entry["accession"] for entry in manifest.get("accessions", [])]
    if not accs:
        raise RuntimeError("Manifest contains no accessions; run ingest first.")
    return accs


def manifest_items(manifest: Dict) -> List[Dict[str, Any]]:
    items = manifest.get("items")
    if not items:
        raise RuntimeError("Manifest contains no items; run ingest first.")
    return items


def safe_item_id(accession: str, sequence: str, fallback_idx: int | None = None) -> str:
    seq = sequence if sequence else (f"doc{fallback_idx:02d}" if fallback_idx is not None else "doc")
    def _clean(s: str) -> str:
        return re.sub(r"[^A-Za-z0-9._-]+", "_", s)
    return f"{_clean(accession)}_{_clean(seq)}"


def read_accessions_file(path: Path) -> List[str]:
    if not path.exists():
        raise FileNotFoundError(f"accessions-file not found: {path}")
    accs = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    if not accs:
        raise RuntimeError("accessions-file is empty.")
    return accs


def assert_exists(path: Path, message: str | None = None) -> Path:
    if not path.exists():
        raise FileNotFoundError(message or f"Missing required file: {path}")
    return path


def prompt_view_path(paths: Paths, item_id: str) -> Path:
    """Return the prompt_view.txt path, checking new normalized/ then legacy prompt_views/."""
    preferred = paths.normalized_dir / item_id / "prompt_view.txt"
    if preferred.exists():
        return preferred
    legacy = paths.legacy_prompt_views_dir / item_id / "prompt_view.txt"
    if legacy.exists():
        return legacy
    raise FileNotFoundError(
        f"Missing prompt_view for {item_id}: checked {preferred} and {legacy}"
    )
