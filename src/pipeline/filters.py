from __future__ import annotations

import importlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Callable, Any

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


def load_doc_filter(spec: FilterSpec) -> Callable[[Dict[str, Any], Dict[str, Any]], bool]:
    if ":" not in spec.doc_filter_path:
        raise ValueError("doc_filter_path must be 'module:function'")
    module_name, func_name = spec.doc_filter_path.split(":", 1)
    module = importlib.import_module(module_name)
    func = getattr(module, func_name, None)
    if func is None or not callable(func):
        raise ValueError(f"Callable {spec.doc_filter_path} not found")
    return func
