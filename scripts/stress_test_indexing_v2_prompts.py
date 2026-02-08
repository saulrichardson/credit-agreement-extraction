#!/usr/bin/env python
"""
Stress-test single-pass (Option A) indexing-v2 prompts across a fixed set of agreements.

What this does
--------------
For each prompt provided:
  1) Clone inputs (normalized + manifest) from an existing source run into a new run_id.
  2) Run indexing_v2 once per item (LLM call; single pass).
  3) Compute offline diagnostics from the resulting anchor selections:
       - anchor counts per bucket
       - table anchor counts per bucket (based on anchors.tsv anchor_type)
       - definitions_anchor_range length (when present)
       - excerpt-pack size proxies after deterministic structural expansion

What this intentionally does NOT do
----------------------------------
- No deterministic semantic backfill (no keyword/%/bps scanning to add anchors).
- No second-pass indexing calls.
- No downstream pricing/covenant extraction calls (can be layered on later).

This script is for comparing prompt strategies, not for production extraction.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"

import sys

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from pipeline.core.anchors import load_anchor_catalog  # noqa: E402
from pipeline.core.config import Paths  # noqa: E402
from pipeline.ir.contract_ir_v0_2_flow import (  # noqa: E402
    ContractIRFlowError,
    DEFAULT_ANCHOR_GAP_FILL_UP_TO,
    DEFAULT_ANCHOR_NEIGHBOR_PAD,
    prepare_run_inputs_from_source,
)
from pipeline.evidence.excerpt_packs import expand_anchor_ids  # noqa: E402
from pipeline.evidence.indexing_v2 import run_indexing_v2  # noqa: E402
from pipeline.schemas_v2 import IndexingSelectionV2Artifact  # noqa: E402


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _manifest_item_ids(manifest: Mapping[str, Any]) -> List[str]:
    items = manifest.get("items")
    if not isinstance(items, list):
        raise RuntimeError("manifest.items missing or not a list")
    out: List[str] = []
    for rec in items:
        if not isinstance(rec, dict):
            continue
        item_id = rec.get("item_id")
        if isinstance(item_id, str) and item_id.strip():
            out.append(item_id.strip())
    if not out:
        raise RuntimeError("No item_id values found in manifest.items")
    return out


def _range_len_in_anchors(catalog: Mapping[str, Mapping[str, Any]], start_anchor: str, end_anchor: str) -> Optional[int]:
    a = catalog.get(start_anchor) or {}
    b = catalog.get(end_anchor) or {}
    ao = a.get("order")
    bo = b.get("order")
    if not isinstance(ao, int) or not isinstance(bo, int):
        return None
    lo = min(int(ao), int(bo))
    hi = max(int(ao), int(bo))
    return int(hi - lo + 1)


def _sum_anchor_chars(
    *,
    catalog: Mapping[str, Mapping[str, Any]],
    anchor_ids: Sequence[str],
) -> int:
    total = 0
    for aid in anchor_ids:
        info = catalog.get(aid) or {}
        start = info.get("start")
        end = info.get("end")
        if not isinstance(start, int) or not isinstance(end, int):
            continue
        if end < start:
            continue
        total += int(end - start)
    return int(total)


def _table_count(*, catalog: Mapping[str, Mapping[str, Any]], anchor_ids: Sequence[str]) -> int:
    n = 0
    for aid in anchor_ids:
        info = catalog.get(aid) or {}
        if str(info.get("anchor_type") or "") == "table":
            n += 1
    return int(n)


@dataclass(frozen=True)
class BucketStats:
    seed_count: int
    table_count: int
    expanded_count: int
    expanded_chars: int


def _bucket_stats(
    *,
    catalog: Mapping[str, Mapping[str, Any]],
    seed_anchor_ids: Sequence[str],
) -> BucketStats:
    expanded = expand_anchor_ids(
        catalog=catalog,
        seed_anchor_ids=seed_anchor_ids,
        fill_gaps_up_to=int(DEFAULT_ANCHOR_GAP_FILL_UP_TO),
        neighbor_pad=int(DEFAULT_ANCHOR_NEIGHBOR_PAD),
    )
    return BucketStats(
        seed_count=int(len(seed_anchor_ids)),
        table_count=_table_count(catalog=catalog, anchor_ids=seed_anchor_ids),
        expanded_count=int(len(expanded)),
        expanded_chars=_sum_anchor_chars(catalog=catalog, anchor_ids=expanded),
    )


def _coherence_ratio(subset: Sequence[str], superset: Sequence[str]) -> Optional[float]:
    sub = {a for a in subset if isinstance(a, str) and a.strip()}
    if not sub:
        return None
    sup = {a for a in superset if isinstance(a, str) and a.strip()}
    return float(len(sub & sup) / len(sub))


def _run_one_prompt(
    *,
    source_paths: Paths,
    prompt_path: Path,
    out_run_id: str,
    item_ids: Sequence[str],
    gateway_url: Optional[str],
    temperature: float,
) -> Dict[str, Any]:
    dest_paths = Paths(root=source_paths.root, run_id=out_run_id)
    prepare_run_inputs_from_source(dest_paths=dest_paths, source_paths=source_paths)

    run_indexing_v2(
        dest_paths,
        item_ids,
        prompt_path,
        gateway_url=gateway_url,
        temperature=float(temperature),
        concurrency=1,
        attempts=3,
    )

    per_item: Dict[str, Any] = {}
    for item_id in item_ids:
        catalog = load_anchor_catalog(dest_paths, item_id)
        sel_path = dest_paths.run_dir / "indexing_v2" / f"{item_id}_anchors.json"
        artifact = IndexingSelectionV2Artifact.model_validate(_read_json(sel_path))
        sel = artifact.selection

        defs_len: Optional[int] = None
        if sel.definitions_anchor_range is not None:
            defs_len = _range_len_in_anchors(
                catalog,
                start_anchor=sel.definitions_anchor_range.start_anchor,
                end_anchor=sel.definitions_anchor_range.end_anchor,
            )

        per_item[item_id] = {
            "definitions_anchor_range": (
                {
                    "start_anchor": sel.definitions_anchor_range.start_anchor,
                    "end_anchor": sel.definitions_anchor_range.end_anchor,
                    "range_len_anchors": defs_len,
                }
                if sel.definitions_anchor_range is not None
                else None
            ),
            "bucket_seed_counts": {
                "metadata": len(sel.metadata_anchors),
                "fundamental": len(sel.fundamental_anchors),
                "pricing": len(sel.pricing_anchors),
                "base_rate": len(sel.base_rate_anchors),
                "spread": len(sel.spread_anchors),
                "fee": len(sel.fee_anchors),
                "financial_covenant": len(sel.financial_covenant_anchors),
            },
            "bucket_table_counts": {
                "metadata": _table_count(catalog=catalog, anchor_ids=sel.metadata_anchors),
                "fundamental": _table_count(catalog=catalog, anchor_ids=sel.fundamental_anchors),
                "pricing": _table_count(catalog=catalog, anchor_ids=sel.pricing_anchors),
                "base_rate": _table_count(catalog=catalog, anchor_ids=sel.base_rate_anchors),
                "spread": _table_count(catalog=catalog, anchor_ids=sel.spread_anchors),
                "fee": _table_count(catalog=catalog, anchor_ids=sel.fee_anchors),
                "financial_covenant": _table_count(catalog=catalog, anchor_ids=sel.financial_covenant_anchors),
            },
            "excerpt_pack_proxy": {
                "base_rate": _bucket_stats(catalog=catalog, seed_anchor_ids=sel.base_rate_anchors).__dict__,
                "spread": _bucket_stats(catalog=catalog, seed_anchor_ids=sel.spread_anchors).__dict__,
                "fee": _bucket_stats(catalog=catalog, seed_anchor_ids=sel.fee_anchors).__dict__,
                "financial_covenant": _bucket_stats(catalog=catalog, seed_anchor_ids=sel.financial_covenant_anchors).__dict__,
            },
            "coherence": {
                "base_rate_in_pricing": _coherence_ratio(sel.base_rate_anchors, sel.pricing_anchors),
                "spread_in_pricing": _coherence_ratio(sel.spread_anchors, sel.pricing_anchors),
                "fee_in_pricing": _coherence_ratio(sel.fee_anchors, sel.pricing_anchors),
            },
        }

    out = {
        "run_id": out_run_id,
        "prompt_path": str(prompt_path),
        "prompt_stem": prompt_path.stem,
        "source_run_id": source_paths.run_id,
        "created_at": int(time.time()),
        "items": per_item,
    }
    _write_json(dest_paths.run_dir / "indexing_v2_stress_test_summary.json", out)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source-run-id", required=True)
    ap.add_argument(
        "--prompt",
        action="append",
        dest="prompts",
        required=True,
        help="Indexing-v2 prompt file (repeatable).",
    )
    ap.add_argument("--run-id-prefix", required=True)
    ap.add_argument("--base-dir", default=".")
    ap.add_argument("--gateway-url", default=None)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--item-id", action="append", dest="item_ids", default=None)
    args = ap.parse_args()

    source_paths = Paths(root=Path(args.base_dir), run_id=str(args.source_run_id))
    manifest = _read_json(source_paths.manifest_path)
    if not isinstance(manifest, dict):
        raise RuntimeError(f"manifest must be a JSON object: {source_paths.manifest_path}")

    if args.item_ids:
        item_ids = [s for s in args.item_ids if isinstance(s, str) and s.strip()]
    else:
        item_ids = _manifest_item_ids(manifest)

    prompts = [Path(p) for p in args.prompts]
    for p in prompts:
        if not p.exists():
            raise RuntimeError(f"Prompt not found: {p}")

    combined: Dict[str, Any] = {
        "schema_version": "indexing_v2_prompt_stress_test_v1",
        "created_at": int(time.time()),
        "source_run_id": source_paths.run_id,
        "run_id_prefix": str(args.run_id_prefix),
        "item_count": len(item_ids),
        "prompt_runs": [],
    }

    for prompt_path in prompts:
        out_run_id = f"{args.run_id_prefix}-{prompt_path.stem}"
        try:
            rec = _run_one_prompt(
                source_paths=source_paths,
                prompt_path=prompt_path,
                out_run_id=out_run_id,
                item_ids=item_ids,
                gateway_url=str(args.gateway_url) if args.gateway_url else None,
                temperature=float(args.temperature),
            )
        except ContractIRFlowError as exc:
            raise RuntimeError(f"Failed to clone inputs for prompt {prompt_path}: {exc}") from exc

        combined["prompt_runs"].append(
            {
                "prompt_path": str(prompt_path),
                "out_run_id": out_run_id,
                "summary_path": str(Paths(root=Path(args.base_dir), run_id=out_run_id).run_dir / "indexing_v2_stress_test_summary.json"),
            }
        )

    stamp = time.strftime("%Y%m%d_%H%M%S")
    out_path = PROJECT_ROOT / "scratch" / f"indexing_v2_prompt_stress_test_{stamp}.json"
    _write_json(out_path, combined)
    print(str(out_path))


if __name__ == "__main__":
    main()

