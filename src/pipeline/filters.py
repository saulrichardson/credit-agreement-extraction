from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Dict

import yaml

from .config import FilterSpec


def load_filter_spec(path: Path) -> FilterSpec:
    if not path.exists():
        raise FileNotFoundError(f"Filter spec not found: {path}")
    if path.suffix.lower() in {".yml", ".yaml"}:
        data = yaml.safe_load(path.read_text())
    else:
        data = json.loads(path.read_text())
    if not isinstance(data, dict):
        raise ValueError("Filter spec must be a mapping")
    return FilterSpec.from_mapping(data)


def serialize_filter_spec(spec: FilterSpec) -> Dict:
    return {k: v for k, v in asdict(spec).items() if v is not None}

