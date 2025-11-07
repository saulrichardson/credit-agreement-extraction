#!/usr/bin/env python3
from __future__ import annotations

import argparse
import itertools
import textwrap
from pathlib import Path
from typing import Iterable

from edgar_filing_pipeline.document_prep import build_anchors, canonicalize
from edgar_filing_pipeline.segment import SegmentExtractor, TarSegmentReader


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect canonicalization and anchoring for a segment."
    )
    parser.add_argument(
        "--tar",
        type=Path,
        required=True,
        help="Path to the .nc.tar.gz archive containing the segment.",
    )
    parser.add_argument(
        "--member",
        type=str,
        help="Specific member file inside the tar (defaults to first .nc file).",
    )
    parser.add_argument(
        "--segment-index",
        type=int,
        default=1,
        help="1-based segment index within the member file.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=1000,
        help="Chunk anchor size threshold (default: 1000).",
    )
    parser.add_argument(
        "--max-anchors",
        type=int,
        default=8,
        help="Number of anchors to preview (default: 8).",
    )
    parser.add_argument(
        "--show-text",
        action="store_true",
        help="Print the full canonical text to stdout.",
    )
    return parser.parse_args()


def select_member(reader: TarSegmentReader, explicit: str | None) -> str:
    members = sorted(m for m in reader.list_members() if m.endswith(".nc"))
    if not members:
        raise SystemExit("No .nc members found in the tar archive.")
    if explicit:
        if explicit not in members:
            raise SystemExit(f"{explicit} not present. Available: {', '.join(members[:10])}")
        return explicit
    return members[0]


def preview_anchors(anchors, text: str, limit: int) -> Iterable[str]:
    for anchor in itertools.islice(anchors, limit):
        excerpt = text[anchor.start : anchor.end].strip().replace("\n", " ↩ ")
        snippet = textwrap.shorten(excerpt, width=180, placeholder="…")
        metadata = f"{anchor.anchor_id} [{anchor.kind}]"
        hints = []
        if anchor.hint.heading_like:
            hints.append("heading_like")
        if anchor.hint.table_like:
            hints.append("table_like")
        if anchor.hint.xref_like:
            hints.append("xref_like")
        hint_str = f" hints={','.join(hints)}" if hints else ""
        yield f"{metadata} span=({anchor.start},{anchor.end}){hint_str}\n    {snippet}"


def summarize_sources(anchors, text: str) -> Iterable[str]:
    for anchor in anchors:
        if not anchor.source_spans:
            continue
        sample_span = anchor.source_spans[0]
        raw_fragment = text[anchor.start : min(anchor.end, anchor.start + 80)]
        normalized_fragment = raw_fragment.replace("\n", " ")
        snippet = textwrap.shorten(normalized_fragment, 60)
        yield (
            f"{anchor.anchor_id}: sources={len(anchor.source_spans)} "
            f"first_source=({sample_span.start},{sample_span.end}) "
            f"canonical_sample={snippet}"
        )


def main() -> None:
    args = parse_args()

    reader = TarSegmentReader(args.tar)
    try:
        member_name = select_member(reader, args.member)
        raw_text = reader.read_member(member_name)
    finally:
        reader.close()

    extractor = SegmentExtractor(raw_text)
    seg_index = args.segment_index - 1
    if seg_index < 0 or seg_index >= len(extractor):
        raise SystemExit(f"Segment index out of range (1-{len(extractor)}).")
    segment_html = extractor.get_segment_html(seg_index)

    canonical = canonicalize(
        segment_html,
        source_id=f"{args.tar.name}:{member_name}:{args.segment_index}",
        treat_as_html=True,
    )
    anchors = build_anchors(
        canonical_text=canonical.text,
        char_sources=canonical.char_sources,
        source_id=canonical.source_id,
        chunk_size=args.chunk_size,
    )

    print("=== Canonicalization Summary ===")
    print(f"Canonical length: {len(canonical.text)} characters")
    print(f"Anchor count: {len(anchors.anchors)}")

    print("\n=== Anchor Preview ===")
    for line in preview_anchors(anchors.anchors, canonical.text, args.max_anchors):
        print(line)

    print("\n=== Anchor Source Diagnostics ===")
    for line in itertools.islice(summarize_sources(anchors.anchors, canonical.text), args.max_anchors):
        print(line)

    if args.show_text:
        print("\n=== Canonical Text ===")
        print(canonical.text)


if __name__ == "__main__":
    main()
