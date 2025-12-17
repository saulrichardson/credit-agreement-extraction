#!/usr/bin/env python
"""
Generate covenant snippets JSONL for given items by selecting financial_covenant anchors
and nearby covenant tables from anchors.tsv.

Output: runs/<run-id>/retrieval/<item>_snippets_covenant.jsonl

Usage:
  poetry run python scripts/gen_covenant_snippets.py \
    --run-id pi-toc-sample \
    --items 0001731122-23-000003_2
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import List, Set

from pipeline.schemas import IndexingSelectionArtifact


def load_financial_covenant_ids(index_path: Path) -> Set[str]:
    artifact = IndexingSelectionArtifact.model_validate_json(index_path.read_text())
    return set(artifact.selection.financial_covenant_anchors)


def collect_snippets(
    canonical: str,
    anchors_tsv: Path,
    fc_ids: Set[str],
    table_proximity: int = 500,
) -> List[dict]:
    rows: List[dict] = []
    with anchors_tsv.open() as fh:
        reader = list(csv.DictReader(fh, delimiter="\t"))

    # map anchor_id -> row
    row_by_id = {r["anchor_id"]: r for r in reader}
    for aid in fc_ids:
        r = row_by_id.get(aid)
        if not r:
            continue
        start, end = int(r["start"]), int(r["end"])
        snippet = canonical[start:end]
        rows.append(
            {
                "anchor_id": aid,
                "label": "financial_covenant",
                "type": r.get("anchor_type"),
                "start": start,
                "end": end,
                "snippet": snippet,
            }
        )
        # look ahead for nearby table
        # find next row by position
        for nxt in reader:
            ns, ne = int(nxt["start"]), int(nxt["end"])
            if ns >= end and ns - end <= table_proximity and nxt.get("anchor_type") == "table":
                tid = nxt["anchor_id"]
                t_snip = canonical[ns:ne]
                rows.append(
                    {
                        "anchor_id": tid,
                        "label": "financial_covenant",
                        "type": "table",
                        "start": ns,
                        "end": ne,
                        "snippet": t_snip,
                    }
                )
                break
    # dedupe by anchor_id
    dedup = {}
    for r in rows:
        dedup[r["anchor_id"]] = r
    return list(dedup.values())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--items", required=True, help="Comma-separated item_ids")
    args = ap.parse_args()

    base = Path("runs")
    base /= args.run_id

    items = [x.strip() for x in args.items.split(",") if x.strip()]
    for item in items:
        canon_path = base / "normalized" / item / "canonical.txt"
        anchors_tsv = base / "normalized" / item / "anchors.tsv"
        index_path = base / "indexing" / f"{item}_anchors_pricing_highlevel_simple.json"
        if not index_path.exists():
            index_path = base / "indexing" / f"{item}_anchors.json"
        out_path = base / "retrieval" / f"{item}_snippets_covenant.jsonl"
        if not index_path.exists():
            print(f"[skip] missing index file for {item}: {index_path}")
            continue
        canonical = canon_path.read_text()
        fc_ids = load_financial_covenant_ids(index_path)
        rows = collect_snippets(canonical, anchors_tsv, fc_ids)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w") as fh:
            for r in rows:
                fh.write(json.dumps({"item_id": item, **r}) + "\n")
        print(f"[cov snippets] {item}: {len(rows)} snippets -> {out_path}")


if __name__ == "__main__":
    main()
