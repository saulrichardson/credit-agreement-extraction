# Document Preparation & Anchoring – Design Snapshot

This note captures the agreed decisions for implementing Component 1 (document
preparation & anchoring). It locks the external interfaces, storage layout, and
behavioural guarantees before coding begins.

## Artifacts

- **Canonical text**: UTF‑8 file named `{segment_id}.canonical.txt` (gzip
  optional) containing the normalized document text.
- **Anchor map**: Columnar Parquet file `{segment_id}.anchors.parquet` with one
  row per anchor.
- **Index metadata**: JSON file `{segment_id}.index.json` summarising schema
  versions, hashes, counts, and source descriptors.

Outputs are stored under `data/prepared/{segment_id}/` (configurable by the
caller).

For manual inspection, use `python scripts/inspect_document_prep.py` to preview
canonical text snippets and anchor metadata for any segment.

## Canonicalization Rules

1. Strip non-visible content (`<script>`, `<style>`, comments, hidden nodes).
2. Decode HTML entities.
3. Normalize whitespace: collapse consecutive spaces/tabs, harmonise newline
   handling, ensure blank-line separation between block boundaries.
4. Normalize punctuation: convert smart quotes/dashes to ASCII equivalents while
   preserving text semantics.
5. Retain a **per-character position map** from canonical indices to the
   original byte range and source identifier. Synthetic characters (e.g.
   inserted blank lines) use `null` to indicate no direct source span.

The canonicalization algorithm never guesses document structure; it performs
deterministic text cleaning guided only by tag boundaries.

## Anchoring Policy

- **Paragraph anchors**: one anchor per paragraph (text bounded by blank lines).
- **Chunk anchors**: additional anchors inside paragraphs longer than
  `CHUNK_SIZE` (default 1 000 chars), aligned to whitespace boundaries.
- Anchor IDs use zero-padded counters: `s000001` for paragraphs,
  `c000001` for chunks.
- Each anchor stores:
  - canonical span `{from, to}`
  - aggregated original source spans (source id + byte ranges)
  - 80-character context hashes before/after the span (SHA‑256 hex)
  - anchor `kind` (`paragraph` or `chunk`)
  - optional hint flags (`heading_like`, `table_like`, `xref_like`)

Context hashes are deterministic and form the basis of future re-anchoring.

## Interfaces

### Canonicalization

```python
from edgar_filing_pipeline.document_prep import canonicalize

result = canonicalize(
    raw_text,
    source_id="segment_id",
    treat_as_html=True,
)
```

`result` contains:

- `text`: canonical string
- `char_sources`: list of per-character source spans or `None`
- `source_id`: echo of the input identifier

### Anchoring

```python
from edgar_filing_pipeline.document_prep import build_anchors

anchors = build_anchors(
    canonical_text=result.text,
    char_sources=result.char_sources,
    source_id=result.source_id,
    chunk_size=1000,
)
```

Returns a list of `Anchor` dataclasses with the fields outlined above.

## Quality Guarantees

- Length equality: `len(text) == len(char_sources)`.
- Coverage: ≥ 98 % of non-empty paragraphs yield a paragraph anchor.
- Anchor spans fall within the canonical text bounds.
- Context hashes allow deterministic re-identification.
- Canonical text contains no residual markup (`<`, `>`).

These decisions ground the implementation of phases 1–2; later phases build on
these artefacts without revisiting core behaviour.
