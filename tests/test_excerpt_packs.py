from __future__ import annotations

import pytest

from pipeline.excerpt_packs import ExcerptPackError, build_excerpt_pack_from_canonical, expand_anchor_ids


def _catalog_for_text(text: str) -> tuple[str, dict[str, dict[str, int]]]:
    # Build a tiny deterministic catalog with fixed-width anchors.
    # Anchor i covers text[i*3:(i+1)*3].
    cat: dict[str, dict[str, int]] = {}
    for i in range(0, len(text) // 3):
        aid = f"A{i+1:04d}"
        cat[aid] = {"order": i, "start": i * 3, "end": (i + 1) * 3}
    return text, cat


def test_expand_anchor_ids_gap_fill_and_neighbor_pad() -> None:
    canonical, catalog = _catalog_for_text("aaabbbcccdddeeefffggghhh")

    # Seed anchors A0002 (order=1) and A0005 (order=4) have a gap of 3, so
    # fill_gaps_up_to=3 should include A0002..A0005, plus neighbor_pad=1 adds A0001 and A0006.
    expanded = expand_anchor_ids(
        catalog=catalog,
        seed_anchor_ids=["A0002", "A0005"],
        fill_gaps_up_to=3,
        neighbor_pad=1,
    )
    assert expanded == ["A0001", "A0002", "A0003", "A0004", "A0005", "A0006"]

    excerpt = build_excerpt_pack_from_canonical(canonical_text=canonical, catalog=catalog, anchor_ids=expanded)
    assert "[[A0001]]" in excerpt
    assert "[[A0006]]" in excerpt
    assert "aaa" in excerpt  # A0001 span
    assert "fff" in excerpt  # A0006 span


def test_expand_anchor_ids_raises_on_unknown_anchor() -> None:
    canonical, catalog = _catalog_for_text("aaabbbccc")
    with pytest.raises(ExcerptPackError):
        expand_anchor_ids(catalog=catalog, seed_anchor_ids=["A9999"], fill_gaps_up_to=3, neighbor_pad=1)


def test_build_excerpt_pack_from_canonical_raises_on_bad_spans() -> None:
    canonical = "abcdef"
    catalog = {"A0001": {"order": 0, "start": 10, "end": 12}}
    with pytest.raises(ExcerptPackError):
        build_excerpt_pack_from_canonical(canonical_text=canonical, catalog=catalog, anchor_ids=["A0001"])

