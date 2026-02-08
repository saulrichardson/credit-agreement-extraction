#!/usr/bin/env python
"""
Evaluate deterministic retrieval/indexing backfill strategies against an existing run.

Goal
----
We want to understand whether current pricing excerpt packs are missing key pricing
tables/schedules because indexing selected "mentions" of pricing terms but missed the
actual definition + grid/table anchors.

This script makes NO LLM calls. It reads:
  - runs/<run_id>/indexing_v2/<item_id>_anchors.json (seed anchor IDs per bucket)
  - runs/<run_id>/normalized/<item_id>/canonical.txt and anchors.tsv (anchor spans/types)

It then simulates different deterministic "completion" strategies and reports whether
the resulting anchor sets include table anchors that look like pricing schedules.
"""

from __future__ import annotations

import argparse
import json
import re
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
from pipeline.evidence.excerpt_packs import expand_anchor_ids, order_anchor_ids  # noqa: E402


SPREAD_KEYWORDS = (
    "applicable margin",
    "applicable rate",
    "margin",
    "spread",
    "pricing table",
    "pricing grid",
    "pricing level",
    "performance pricing",
)

FEE_KEYWORDS = (
    "facility fee",
    "facility fee rate",
    "commitment fee",
    "unused commitment",
    "unused line",
    "utilization fee",
    "letter of credit fee",
    "l/c fee",
    "l/c fee rate",
    "fronting fee",
    "issuance fee",
)

RATE_LIKE_RE = re.compile(r"(\d+(?:\.\d+)?\s*%|\bbps\b|\bbasis point)", flags=re.IGNORECASE)


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _anchor_text(canonical_text: str, catalog: Mapping[str, Mapping[str, Any]], anchor_id: str) -> str:
    info = catalog.get(anchor_id) or {}
    start = info.get("start")
    end = info.get("end")
    if not isinstance(start, int) or not isinstance(end, int) or end <= start:
        return ""
    return canonical_text[start:end]


def _has_any_keyword(text: str, keywords: Sequence[str]) -> bool:
    t = (text or "").lower()
    return any(k in t for k in keywords)


def _anchor_order(catalog: Mapping[str, Mapping[str, Any]], anchor_id: str) -> Optional[int]:
    info = catalog.get(anchor_id) or {}
    o = info.get("order")
    return int(o) if isinstance(o, int) else None


def _anchors_by_order(catalog: Mapping[str, Mapping[str, Any]]) -> Dict[int, str]:
    out: Dict[int, str] = {}
    for aid, info in (catalog or {}).items():
        if not isinstance(aid, str) or not isinstance(info, dict):
            continue
        o = info.get("order")
        if isinstance(o, int):
            out[o] = aid
    return out


def _orders_in_range(
    *, catalog: Mapping[str, Mapping[str, Any]], start_anchor: str, end_anchor: str
) -> Optional[Tuple[int, int]]:
    so = _anchor_order(catalog, start_anchor)
    eo = _anchor_order(catalog, end_anchor)
    if so is None or eo is None:
        return None
    return (min(so, eo), max(so, eo))


@dataclass(frozen=True)
class BackfillResult:
    seed_anchor_ids: List[str]
    auto_added: List[Dict[str, str]]


def _defs_table_backfill(
    *,
    canonical_text: str,
    catalog: Mapping[str, Mapping[str, Any]],
    seed_anchor_ids: Sequence[str],
    definitions_anchor_range: Optional[Mapping[str, str]],
    bucket: str,
    lookaround: int = 2,
) -> BackfillResult:
    """Deterministically add pricing table anchors from the definitions range.

    Strategy:
      - If definitions_anchor_range is present, scan table anchors within that range.
      - If a nearby sentence/paragraph anchor mentions a bucket keyword, add both the
        header anchor and the table anchor as additional seeds for that bucket.
    """

    if bucket not in ("spread", "fee"):
        raise ValueError("bucket must be 'spread' or 'fee'")

    keywords = SPREAD_KEYWORDS if bucket == "spread" else FEE_KEYWORDS
    dr = definitions_anchor_range or None
    if not dr:
        return BackfillResult(seed_anchor_ids=list(seed_anchor_ids), auto_added=[])

    start_anchor = dr.get("start_anchor")
    end_anchor = dr.get("end_anchor")
    if not isinstance(start_anchor, str) or not isinstance(end_anchor, str):
        return BackfillResult(seed_anchor_ids=list(seed_anchor_ids), auto_added=[])

    orders = _orders_in_range(catalog=catalog, start_anchor=start_anchor, end_anchor=end_anchor)
    if orders is None:
        return BackfillResult(seed_anchor_ids=list(seed_anchor_ids), auto_added=[])
    start_order, end_order = orders

    by_order = _anchors_by_order(catalog)
    existing = set(a.strip() for a in seed_anchor_ids if isinstance(a, str) and a.strip())
    additions: List[str] = []
    auto: List[Dict[str, str]] = []

    def _maybe_add(aid: str, *, reason: str) -> None:
        if not aid or aid in existing:
            return
        existing.add(aid)
        additions.append(aid)
        auto.append({"anchor_id": aid, "reason": reason})

    # Scan for candidate tables in definitions range.
    for o in range(int(start_order), int(end_order) + 1):
        aid = by_order.get(o)
        if not aid:
            continue
        info = catalog.get(aid) or {}
        if info.get("anchor_type") != "table":
            continue

        table_txt = _anchor_text(canonical_text, catalog, aid)
        # Quick filter: avoid pulling in TOC / irrelevant tables unless something hints pricing.
        # (Still allow if header sentence mentions pricing.)
        table_has_rateish = bool(RATE_LIKE_RE.search(table_txt))

        header_anchor: Optional[str] = None
        header_txt: str = ""
        for d in range(1, int(lookaround) + 1):
            for oo in (o - d, o + d):
                ha = by_order.get(oo)
                if not ha:
                    continue
                hinfo = catalog.get(ha) or {}
                if hinfo.get("anchor_type") not in ("sentence", "paragraph"):
                    continue
                ht = _anchor_text(canonical_text, catalog, ha)
                if _has_any_keyword(ht, keywords):
                    header_anchor = ha
                    header_txt = ht
                    break
            if header_anchor:
                break

        if not header_anchor:
            # Table-only keyword match fallback (less reliable than header match).
            if not _has_any_keyword(table_txt, keywords):
                continue
            header_anchor = aid  # no separate header anchor available

        # If this doesn't look like it has rates at all, skip (keeps context smaller).
        if not table_has_rateish:
            continue

        if header_anchor != aid:
            _maybe_add(
                header_anchor,
                reason=f"defs-table-backfill({bucket}): header near table {aid} contains pricing keyword(s)",
            )
        _maybe_add(
            aid,
            reason=f"defs-table-backfill({bucket}): table appears to contain rate-like values and is near pricing keyword(s)",
        )

    return BackfillResult(seed_anchor_ids=order_anchor_ids(catalog=catalog, anchor_ids=list(seed_anchor_ids) + additions), auto_added=auto)


def _global_table_backfill(
    *,
    canonical_text: str,
    catalog: Mapping[str, Mapping[str, Any]],
    seed_anchor_ids: Sequence[str],
    bucket: str,
    lookaround: int = 2,
) -> BackfillResult:
    """More aggressive version of defs-table-backfill: scans the whole doc."""

    if bucket not in ("spread", "fee"):
        raise ValueError("bucket must be 'spread' or 'fee'")
    keywords = SPREAD_KEYWORDS if bucket == "spread" else FEE_KEYWORDS

    by_order = _anchors_by_order(catalog)
    existing = set(a.strip() for a in seed_anchor_ids if isinstance(a, str) and a.strip())
    additions: List[str] = []
    auto: List[Dict[str, str]] = []

    def _maybe_add(aid: str, *, reason: str) -> None:
        if not aid or aid in existing:
            return
        existing.add(aid)
        additions.append(aid)
        auto.append({"anchor_id": aid, "reason": reason})

    for o, aid in sorted(by_order.items()):
        info = catalog.get(aid) or {}
        if info.get("anchor_type") != "table":
            continue

        table_txt = _anchor_text(canonical_text, catalog, aid)
        table_has_rateish = bool(RATE_LIKE_RE.search(table_txt))
        if not table_has_rateish:
            continue

        # Require some nearby keyword to avoid hoovering TOC / random numeric tables.
        header_anchor: Optional[str] = None
        for d in range(1, int(lookaround) + 1):
            for oo in (o - d, o + d):
                ha = by_order.get(oo)
                if not ha:
                    continue
                hinfo = catalog.get(ha) or {}
                if hinfo.get("anchor_type") not in ("sentence", "paragraph"):
                    continue
                ht = _anchor_text(canonical_text, catalog, ha)
                if _has_any_keyword(ht, keywords):
                    header_anchor = ha
                    break
            if header_anchor:
                break

        if not header_anchor and not _has_any_keyword(table_txt, keywords):
            continue

        if header_anchor and header_anchor != aid:
            _maybe_add(header_anchor, reason=f"global-table-backfill({bucket}): header near table {aid} has keyword(s)")
        _maybe_add(aid, reason=f"global-table-backfill({bucket}): table has rate-like values and pricing keyword signal")

    return BackfillResult(seed_anchor_ids=order_anchor_ids(catalog=catalog, anchor_ids=list(seed_anchor_ids) + additions), auto_added=auto)


def _pricing_window_table_backfill(
    *,
    canonical_text: str,
    catalog: Mapping[str, Mapping[str, Any]],
    seed_anchor_ids: Sequence[str],
    pricing_anchor_ids: Sequence[str],
    bucket: str,
    window_radius: int = 30,
    lookaround: int = 2,
) -> BackfillResult:
    """Backfill tables near the LLM-selected pricing_anchors neighborhood.

    This stays in-lane with the "LLM global reader" strategy:
      - LLM identifies a broad set of pricing anchors (pricing_anchors).
      - We deterministically add nearby table anchors with keyword/rate signals,
        rather than scanning the entire document blindly.
    """

    if bucket not in ("spread", "fee"):
        raise ValueError("bucket must be 'spread' or 'fee'")
    keywords = SPREAD_KEYWORDS if bucket == "spread" else FEE_KEYWORDS

    center_orders: List[int] = []
    for aid in pricing_anchor_ids:
        o = _anchor_order(catalog, aid)
        if o is not None:
            center_orders.append(int(o))
    if not center_orders:
        return BackfillResult(seed_anchor_ids=list(seed_anchor_ids), auto_added=[])

    by_order = _anchors_by_order(catalog)
    existing = set(a.strip() for a in seed_anchor_ids if isinstance(a, str) and a.strip())
    additions: List[str] = []
    auto: List[Dict[str, str]] = []

    def _maybe_add(aid: str, *, reason: str) -> None:
        if not aid or aid in existing:
            return
        existing.add(aid)
        additions.append(aid)
        auto.append({"anchor_id": aid, "reason": reason})

    def _near_pricing(o: int) -> bool:
        return any(abs(int(o) - int(c)) <= int(window_radius) for c in center_orders)

    for o, aid in sorted(by_order.items()):
        if not _near_pricing(int(o)):
            continue
        info = catalog.get(aid) or {}
        if info.get("anchor_type") != "table":
            continue

        table_txt = _anchor_text(canonical_text, catalog, aid)
        if not RATE_LIKE_RE.search(table_txt):
            continue

        header_anchor: Optional[str] = None
        for d in range(1, int(lookaround) + 1):
            for oo in (o - d, o + d):
                ha = by_order.get(oo)
                if not ha:
                    continue
                hinfo = catalog.get(ha) or {}
                if hinfo.get("anchor_type") not in ("sentence", "paragraph"):
                    continue
                ht = _anchor_text(canonical_text, catalog, ha)
                if _has_any_keyword(ht, keywords):
                    header_anchor = ha
                    break
            if header_anchor:
                break

        if not header_anchor and not _has_any_keyword(table_txt, keywords):
            continue

        if header_anchor and header_anchor != aid:
            _maybe_add(
                header_anchor,
                reason=f"pricing-window-backfill({bucket}): header near table {aid} has keyword(s) (within {window_radius} anchors of pricing_anchors)",
            )
        _maybe_add(
            aid,
            reason=f"pricing-window-backfill({bucket}): table has rate-like values and pricing keyword signal (within {window_radius} anchors of pricing_anchors)",
        )

    return BackfillResult(seed_anchor_ids=order_anchor_ids(catalog=catalog, anchor_ids=list(seed_anchor_ids) + additions), auto_added=auto)


def _expanded(*, catalog: Mapping[str, Mapping[str, Any]], seed_anchor_ids: Sequence[str], fill: int, pad: int) -> List[str]:
    return expand_anchor_ids(catalog=catalog, seed_anchor_ids=seed_anchor_ids, fill_gaps_up_to=fill, neighbor_pad=pad)


def _count_rate_tables(*, canonical_text: str, catalog: Mapping[str, Mapping[str, Any]], anchor_ids: Sequence[str]) -> int:
    n = 0
    for aid in anchor_ids:
        info = catalog.get(aid) or {}
        if info.get("anchor_type") != "table":
            continue
        txt = _anchor_text(canonical_text, catalog, aid)
        if RATE_LIKE_RE.search(txt):
            n += 1
    return n


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--base-dir", default=".")
    ap.add_argument("--out", default=None, help="Optional JSON output path")
    ap.add_argument("--fill", type=int, default=12)
    ap.add_argument("--pad", type=int, default=2)
    args = ap.parse_args()

    paths = Paths(root=Path(args.base_dir), run_id=args.run_id)
    idx_dir = paths.run_dir / "indexing_v2"
    if not idx_dir.exists():
        raise SystemExit(f"Missing indexing_v2 dir: {idx_dir}")

    rows: List[Dict[str, Any]] = []
    for fp in sorted(idx_dir.glob("*_anchors.json")):
        item_id = fp.name.replace("_anchors.json", "")
        doc = _read_json(fp)
        sel = doc.get("selection") or {}

        catalog = load_anchor_catalog(paths, item_id)
        canonical_text = (paths.run_dir / "normalized" / item_id / "canonical.txt").read_text(
            encoding="utf-8", errors="replace"
        )

        dr = sel.get("definitions_anchor_range") if isinstance(sel, dict) else None
        pricing_seeds = sel.get("pricing_anchors") or []
        spread_seeds = sel.get("spread_anchors") or []
        fee_seeds = sel.get("fee_anchors") or []

        # Baseline expansion.
        spread_base = _expanded(catalog=catalog, seed_anchor_ids=spread_seeds, fill=int(args.fill), pad=int(args.pad))
        fee_base = _expanded(catalog=catalog, seed_anchor_ids=fee_seeds, fill=int(args.fill), pad=int(args.pad))

        # Option 1: definitions-range table backfill.
        spread_defs = _defs_table_backfill(
            canonical_text=canonical_text,
            catalog=catalog,
            seed_anchor_ids=spread_seeds,
            definitions_anchor_range=dr if isinstance(dr, dict) else None,
            bucket="spread",
        )
        fee_defs = _defs_table_backfill(
            canonical_text=canonical_text,
            catalog=catalog,
            seed_anchor_ids=fee_seeds,
            definitions_anchor_range=dr if isinstance(dr, dict) else None,
            bucket="fee",
        )
        spread_defs_exp = _expanded(catalog=catalog, seed_anchor_ids=spread_defs.seed_anchor_ids, fill=int(args.fill), pad=int(args.pad))
        fee_defs_exp = _expanded(catalog=catalog, seed_anchor_ids=fee_defs.seed_anchor_ids, fill=int(args.fill), pad=int(args.pad))

        # Option 2: global table backfill.
        spread_glob = _global_table_backfill(
            canonical_text=canonical_text,
            catalog=catalog,
            seed_anchor_ids=spread_seeds,
            bucket="spread",
        )
        fee_glob = _global_table_backfill(
            canonical_text=canonical_text,
            catalog=catalog,
            seed_anchor_ids=fee_seeds,
            bucket="fee",
        )
        spread_glob_exp = _expanded(catalog=catalog, seed_anchor_ids=spread_glob.seed_anchor_ids, fill=int(args.fill), pad=int(args.pad))
        fee_glob_exp = _expanded(catalog=catalog, seed_anchor_ids=fee_glob.seed_anchor_ids, fill=int(args.fill), pad=int(args.pad))

        # Option 3: windowed scan around pricing_anchors.
        spread_win = _pricing_window_table_backfill(
            canonical_text=canonical_text,
            catalog=catalog,
            seed_anchor_ids=spread_seeds,
            pricing_anchor_ids=pricing_seeds,
            bucket="spread",
        )
        fee_win = _pricing_window_table_backfill(
            canonical_text=canonical_text,
            catalog=catalog,
            seed_anchor_ids=fee_seeds,
            pricing_anchor_ids=pricing_seeds,
            bucket="fee",
        )
        spread_win_exp = _expanded(catalog=catalog, seed_anchor_ids=spread_win.seed_anchor_ids, fill=int(args.fill), pad=int(args.pad))
        fee_win_exp = _expanded(catalog=catalog, seed_anchor_ids=fee_win.seed_anchor_ids, fill=int(args.fill), pad=int(args.pad))

        row = {
            "item_id": item_id,
            "spread": {
                "seed_count": len(spread_seeds),
                "baseline": {"expanded_count": len(spread_base), "rate_tables": _count_rate_tables(canonical_text=canonical_text, catalog=catalog, anchor_ids=spread_base)},
                "defs_backfill": {
                    "auto_added_count": len(spread_defs.auto_added),
                    "expanded_count": len(spread_defs_exp),
                    "rate_tables": _count_rate_tables(canonical_text=canonical_text, catalog=catalog, anchor_ids=spread_defs_exp),
                },
                "global_backfill": {
                    "auto_added_count": len(spread_glob.auto_added),
                    "expanded_count": len(spread_glob_exp),
                    "rate_tables": _count_rate_tables(canonical_text=canonical_text, catalog=catalog, anchor_ids=spread_glob_exp),
                },
                "pricing_window_backfill": {
                    "auto_added_count": len(spread_win.auto_added),
                    "expanded_count": len(spread_win_exp),
                    "rate_tables": _count_rate_tables(canonical_text=canonical_text, catalog=catalog, anchor_ids=spread_win_exp),
                },
            },
            "fee": {
                "seed_count": len(fee_seeds),
                "baseline": {"expanded_count": len(fee_base), "rate_tables": _count_rate_tables(canonical_text=canonical_text, catalog=catalog, anchor_ids=fee_base)},
                "defs_backfill": {
                    "auto_added_count": len(fee_defs.auto_added),
                    "expanded_count": len(fee_defs_exp),
                    "rate_tables": _count_rate_tables(canonical_text=canonical_text, catalog=catalog, anchor_ids=fee_defs_exp),
                },
                "global_backfill": {
                    "auto_added_count": len(fee_glob.auto_added),
                    "expanded_count": len(fee_glob_exp),
                    "rate_tables": _count_rate_tables(canonical_text=canonical_text, catalog=catalog, anchor_ids=fee_glob_exp),
                },
                "pricing_window_backfill": {
                    "auto_added_count": len(fee_win.auto_added),
                    "expanded_count": len(fee_win_exp),
                    "rate_tables": _count_rate_tables(canonical_text=canonical_text, catalog=catalog, anchor_ids=fee_win_exp),
                },
            },
        }
        rows.append(row)

    # Summaries
    def _sum(bucket: str, variant: str) -> int:
        return sum(1 for r in rows if (r[bucket][variant]["rate_tables"] or 0) > 0)

    summary = {
        "run_id": args.run_id,
        "fill_gaps_up_to": int(args.fill),
        "neighbor_pad": int(args.pad),
        "items": len(rows),
        "spread_has_rate_table": {
            "baseline": _sum("spread", "baseline"),
            "defs_backfill": _sum("spread", "defs_backfill"),
            "global_backfill": _sum("spread", "global_backfill"),
            "pricing_window_backfill": _sum("spread", "pricing_window_backfill"),
        },
        "fee_has_rate_table": {
            "baseline": _sum("fee", "baseline"),
            "defs_backfill": _sum("fee", "defs_backfill"),
            "global_backfill": _sum("fee", "global_backfill"),
            "pricing_window_backfill": _sum("fee", "pricing_window_backfill"),
        },
        "rows": rows,
    }

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(str(out_path))
    else:
        print(json.dumps(summary, indent=2, sort_keys=True))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
