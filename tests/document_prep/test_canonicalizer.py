from __future__ import annotations

import random
from html import unescape

from edgar_filing_pipeline.document_prep.canonicalizer import (
    CanonicalCharSource,
    canonicalize,
)
from edgar_filing_pipeline.document_prep.anchoring import build_anchors
from edgar_filing_pipeline.segment import SegmentExtractor, TarSegmentReader


def _load_sample_html() -> str:
    sample_tar = "data/daily_filings/1996/QTR1/19960103.nc.tar.gz"
    reader = TarSegmentReader(sample_tar)
    try:
        member = "0000075176-96-000001.nc"
        text = reader.read_member(member)
        extractor = SegmentExtractor(text)
        return extractor.get_segment_html(0)
    finally:
        reader.close()


def test_canonicalize_basic_properties():
    random.seed(42)
    raw_html = _load_sample_html()
    result = canonicalize(raw_html, source_id="sample", treat_as_html=True)

    assert result.text, "canonical text should not be empty"
    assert "<" not in result.text and ">" not in result.text
    assert len(result.text) == len(result.char_sources)

    # Sample a few characters that originate from the source and verify mapping.
    indices = [
        idx
        for idx, pointer in enumerate(result.char_sources)
        if pointer is not None and result.text[idx].strip()
    ]
    sample_indices = random.sample(indices, min(25, len(indices)))
    for idx in sample_indices:
        pointer: CanonicalCharSource = result.char_sources[idx]  # type: ignore[assignment]
        original_slice = raw_html[pointer.start : pointer.end]
        reconstructed = unescape(original_slice)
        # normalise smart punctuation in the expectation check
        assert result.text[idx] in reconstructed or reconstructed.strip() == ""


def test_anchor_builder_covers_document():
    raw_html = _load_sample_html()
    canonical = canonicalize(raw_html, source_id="sample", treat_as_html=True)

    anchors = build_anchors(
        canonical_text=canonical.text,
        char_sources=canonical.char_sources,
        source_id=canonical.source_id,
        chunk_size=500,
    )

    assert anchors.anchors, "anchors should not be empty"
    assert anchors.canonical_length == len(canonical.text)

    # Ensure paragraph anchors span the document without overlaps outside bounds.
    for anchor in anchors.anchors:
        assert 0 <= anchor.start < anchor.end <= len(canonical.text)
        if anchor.source_spans:
            for span in anchor.source_spans:
                assert span.start < span.end

    # Coverage check: every non-whitespace character appears in at least one anchor.
    coverage = [False] * len(canonical.text)
    for anchor in anchors.anchors:
        for i in range(anchor.start, anchor.end):
            coverage[i] = True

    non_whitespace_indices = [
        idx for idx, char in enumerate(canonical.text) if char.strip()
    ]
    uncovered = [idx for idx in non_whitespace_indices if not coverage[idx]]
    assert not uncovered, "all substantive characters should be covered by anchors"
