from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

# Pipeline-wide defaults for gateway calls
REQUIRED_MODEL = "openai:gpt-5-nano"
REQUIRED_REASONING = "medium"


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
    def normalized_dir(self) -> Path:
        return self.run_dir / "normalized"

    @property
    def indexing_dir(self) -> Path:
        return self.run_dir / "indexing"

    @property
    def retrieval_dir(self) -> Path:
        return self.run_dir / "retrieval"

    @property
    def structured_dir(self) -> Path:
        return self.run_dir / "llm_qa"

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
    #
    # doc_filter_path format: "module:function"
    #
    # Two supported callable shapes:
    # 1) Predicate:
    #    (submission_dict, document_dict) -> bool
    # 2) Factory (when doc_filter_kwargs is provided):
    #    (**doc_filter_kwargs) -> predicate(submission_dict, document_dict) -> bool
    #
    # submission_dict keys: accession, cik, form_type, filing_date, documents[*].
    # document_dict keys: type, filename, sequence, content, (plus any SGML fields).
    doc_filter_path: str
    doc_filter_kwargs: Dict[str, Any] | None = None

    @classmethod
    def from_mapping(cls, mapping: Dict) -> "FilterSpec":
        path = mapping.get("doc_filter_path")
        if not path or not isinstance(path, str):
            raise ValueError("doc_filter_path is required (module:function)")
        kwargs = mapping.get("doc_filter_kwargs")
        if kwargs is not None and not isinstance(kwargs, dict):
            raise ValueError("doc_filter_kwargs must be a mapping when provided")
        return cls(doc_filter_path=path, doc_filter_kwargs=kwargs)


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
