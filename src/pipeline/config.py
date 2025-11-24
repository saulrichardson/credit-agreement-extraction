from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional


def _hash_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


@dataclass(frozen=True)
class Paths:
    root: Path
    run_id: str

    @property
    def run_dir(self) -> Path:
        return self.root / "runs" / self.run_id

    @property
    def ingest_dir(self) -> Path:
        return self.run_dir / "ingest"

    @property
    def prompt_views_dir(self) -> Path:
        return self.run_dir / "prompt_views"

    @property
    def indexing_dir(self) -> Path:
        return self.run_dir / "indexing"

    @property
    def retrieval_dir(self) -> Path:
        return self.run_dir / "retrieval"

    @property
    def structured_dir(self) -> Path:
        return self.run_dir / "structured"

    @property
    def validation_dir(self) -> Path:
        return self.run_dir / "validation"

    @property
    def deliverables_dir(self) -> Path:
        return self.run_dir / "deliverables"

    @property
    def manifest_path(self) -> Path:
        return self.run_dir / "manifest.json"


@dataclass
class RunConfig:
    run_id: str
    base_dir: Path = Path(".")
    workers: int = 4
    bandwidth: int = 4  # snippet context sentences

    def paths(self) -> Paths:
        return Paths(root=self.base_dir, run_id=self.run_id)


@dataclass
class FilterSpec:
    # Single, user-supplied Python callable to decide if a document should be kept.
    # Path format: "module:function". The callable signature must be (submission_dict, document_dict) -> bool.
    # submission_dict keys: accession, cik, form_type, filing_date, documents[*].
    # document_dict keys: type, filename, sequence, content, (plus any SGML fields).
    doc_filter_path: str

    @classmethod
    def from_mapping(cls, mapping: Dict) -> "FilterSpec":
        path = mapping.get("doc_filter_path")
        if not path:
            raise ValueError("doc_filter_path is required (module:function)")
        return cls(doc_filter_path=path)


def record_manifest(path: Path, payload: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


def update_manifest(path: Path, **fields: Dict) -> Dict:
    """Load, merge, and persist manifest atomically."""
    if not path.exists():
        raise FileNotFoundError(f"Manifest not found: {path}")
    manifest = json.loads(path.read_text())
    manifest.update(fields)
    record_manifest(path, manifest)
    return manifest


def prompt_hash(prompt_path: Path) -> str:
    return _hash_file(prompt_path)
