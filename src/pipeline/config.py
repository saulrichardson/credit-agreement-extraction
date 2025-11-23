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
    accession_in: Optional[List[str]] = None
    cik_in: Optional[List[str]] = None
    exhibit_glob: Optional[str] = None
    form_in: Optional[List[str]] = None
    date_from: Optional[str] = None
    date_to: Optional[str] = None

    @classmethod
    def from_mapping(cls, mapping: Dict) -> "FilterSpec":
        return cls(
            accession_in=mapping.get("accession_in"),
            cik_in=mapping.get("cik_in"),
            exhibit_glob=mapping.get("exhibit_glob"),
            form_in=mapping.get("form_in"),
            date_from=mapping.get("date_from"),
            date_to=mapping.get("date_to"),
        )


def record_manifest(path: Path, payload: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


def prompt_hash(prompt_path: Path) -> str:
    return _hash_file(prompt_path)

