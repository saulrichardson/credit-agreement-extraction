from __future__ import annotations

from pathlib import Path
from typing import Any

from pipeline.core.config import Paths


def load_anchor_catalog(paths: Paths, item_id: str) -> dict[str, dict[str, Any]]:
    """Load anchor spans from normalization output.

    Source of truth: `runs/<run_id>/normalized/<item_id>/anchors.tsv`.

    Returns a dict keyed by anchor_id with:
      - anchor_id (str)
      - anchor_type (str): "sentence" | "bullet" | "table"
      - start (int): character offset into canonical.txt
      - end (int): character offset into canonical.txt
      - order (int): stable document order (0-based)
    """

    tsv_path = paths.normalized_dir / item_id / "anchors.tsv"
    if not tsv_path.exists():
        raise FileNotFoundError(f"anchors.tsv not found for {item_id}: expected {tsv_path}")

    catalog: dict[str, dict[str, Any]] = {}
    with tsv_path.open() as fh:
        header = fh.readline()
        if not header:
            raise RuntimeError(f"anchors.tsv is empty: {tsv_path}")

        order = 0
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 5:
                raise ValueError(f"Invalid anchors.tsv row (expected 5 columns): {line!r}")
            anchor_id, anchor_type, start, end, _label = parts[:5]
            try:
                start_i = int(start)
                end_i = int(end)
            except ValueError as exc:
                raise ValueError(f"Invalid start/end offsets in anchors.tsv row: {line!r}") from exc

            catalog[anchor_id] = {
                "anchor_id": anchor_id,
                "anchor_type": anchor_type,
                "start": start_i,
                "end": end_i,
                "order": order,
            }
            order += 1

    if not catalog:
        raise RuntimeError(f"No anchors parsed from {tsv_path}")

    return catalog

