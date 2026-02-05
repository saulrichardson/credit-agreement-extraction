from __future__ import annotations

from typing import Any, Dict, List, Mapping, Sequence


class ExcerptPackError(RuntimeError):
    pass


def order_anchor_ids(*, catalog: Mapping[str, Mapping[str, Any]], anchor_ids: Sequence[str]) -> List[str]:
    """Return unique anchor_ids sorted by document order from anchors.tsv (catalog['order'])."""

    cleaned: List[str] = []
    seen: set[str] = set()
    for aid in anchor_ids:
        if not isinstance(aid, str):
            continue
        a = aid.strip()
        if not a or a in seen:
            continue
        cleaned.append(a)
        seen.add(a)

    def _order(aid: str) -> int:
        info = catalog.get(aid)
        if not info:
            raise ExcerptPackError(f"Unknown anchor_id {aid!r} (not present in anchors.tsv)")
        order = info.get("order")
        if not isinstance(order, int):
            raise ExcerptPackError(f"Invalid anchors.tsv catalog entry for {aid!r}: missing int 'order'")
        return order

    return sorted(cleaned, key=_order)


def expand_anchor_ids(
    *,
    catalog: Mapping[str, Mapping[str, Any]],
    seed_anchor_ids: Sequence[str],
    fill_gaps_up_to: int,
    neighbor_pad: int,
) -> List[str]:
    """Expand a seed anchor set into a more 'complete' excerpt pack.

    This implements the "Seed + deterministic completion" strategy:
      - Start with anchor IDs selected by an LLM that read the full document (seeds).
      - Deterministically add nearby anchors so we don't drop required threshold/table anchors.

    Expansion rules:
      1) Gap fill: If two consecutive seed anchors are within `fill_gaps_up_to` anchors (by order),
         include all anchors between them (inclusive).
      2) Neighbor pad: For each seed anchor, include +/- `neighbor_pad` adjacent anchors.

    Returns a deduped, ordered list of anchor IDs.
    """

    ordered_seeds = order_anchor_ids(catalog=catalog, anchor_ids=seed_anchor_ids)
    if not ordered_seeds:
        return []

    # Build order <-> anchor_id mappings from the catalog.
    anchor_by_order: Dict[int, str] = {}
    order_by_anchor: Dict[str, int] = {}
    for aid, info in (catalog or {}).items():
        if not isinstance(aid, str) or not isinstance(info, dict):
            continue
        order = info.get("order")
        if not isinstance(order, int):
            continue
        anchor_by_order[order] = aid
        order_by_anchor[aid] = order

    seed_orders = [order_by_anchor[a] for a in ordered_seeds]
    expanded_orders: set[int] = set(seed_orders)

    # Fill small gaps between consecutive seeds.
    if fill_gaps_up_to > 0:
        for a, b in zip(seed_orders, seed_orders[1:]):
            gap = int(b) - int(a)
            if gap <= 0:
                continue
            if gap <= int(fill_gaps_up_to):
                for o in range(int(a), int(b) + 1):
                    expanded_orders.add(o)

    # Add neighbor pad around each seed.
    if neighbor_pad > 0:
        for o in seed_orders:
            for d in range(1, int(neighbor_pad) + 1):
                expanded_orders.add(int(o) - d)
                expanded_orders.add(int(o) + d)

    expanded = [anchor_by_order[o] for o in sorted(expanded_orders) if o in anchor_by_order]
    return order_anchor_ids(catalog=catalog, anchor_ids=expanded)


def build_excerpt_pack_from_canonical(
    *,
    canonical_text: str,
    catalog: Mapping[str, Mapping[str, Any]],
    anchor_ids: Sequence[str],
) -> str:
    """Render an excerpt pack as anchored blocks using full anchor text spans.

    Each block has the form:
      [[A0001]]
      <exact anchor text from canonical.txt>

    This is intentionally NOT a "window snippet" strategy; we slice by anchors.tsv spans so each
    selected anchor is structurally complete.
    """

    ordered = order_anchor_ids(catalog=catalog, anchor_ids=anchor_ids)
    blocks: List[str] = []
    for aid in ordered:
        info = catalog.get(aid)
        if not info:
            raise ExcerptPackError(f"Unknown anchor_id {aid!r} (not present in anchors.tsv)")
        start = info.get("start")
        end = info.get("end")
        if not isinstance(start, int) or not isinstance(end, int):
            raise ExcerptPackError(f"Invalid anchors.tsv entry for {aid!r}: start/end must be ints")
        if start < 0 or end < 0 or end < start:
            raise ExcerptPackError(f"Invalid anchors.tsv entry for {aid!r}: start={start} end={end}")
        if start > len(canonical_text) or end > len(canonical_text):
            raise ExcerptPackError(
                f"Anchor span out of bounds for {aid!r}: start={start} end={end} len(canonical)={len(canonical_text)}"
            )
        txt = canonical_text[start:end].rstrip()
        blocks.append(f"[[{aid}]]\n{txt}" if txt else f"[[{aid}]]\n<EMPTY ANCHOR TEXT>")
    return "\n\n".join(blocks).strip() + "\n"

