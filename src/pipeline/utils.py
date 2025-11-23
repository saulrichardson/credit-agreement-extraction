from __future__ import annotations

import json
from pathlib import Path
from typing import List, Dict


def load_manifest(path: Path) -> Dict:
    if not path.exists():
        raise FileNotFoundError(f"Manifest not found: {path}")
    return json.loads(path.read_text())


def manifest_accessions(manifest: Dict) -> List[str]:
    accs = [entry["accession"] for entry in manifest.get("accessions", [])]
    if not accs:
        raise RuntimeError("Manifest contains no accessions; run ingest first.")
    return accs

