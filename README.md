edgar-filing-pipeline
=====================

This package bundles the utilities we need to extract, catalogue, and analyse
segment-level filings from the SEC EDGAR tar archives.  It is intended to live
next to (but independent from) the larger covenant extraction repository so
that we can eventually promote it into its own project or submodule.

Key features
------------

* A light-weight `EdgarFiling` object that wraps an individual segment and
  exposes the HTML, plain-text, and table representations we need for
  downstream regex and LLM workflows.
* Manifest generation that walks the `.tar.gz` archives and records segment
  metadata (file name, segment number, document type, table presence, etc.).
* Extraction helpers that materialise the raw HTML for selected segments while
  keeping metadata in sync.

The API is intentionally simple so we can embed it inside larger pipelines or
call it from notebooks.  See `scripts/build_manifest.py` and
`scripts/extract_segments.py` for end-to-end examples.

Quick start
-----------

```bash
cd edgar_filing_pipeline
python -m pip install -e .

# Build a manifest for every .nc document in a tarball directory
python scripts/build_manifest.py \
    --tar-root /path/to/daily_filings/2003/QTR2 \
    --output manifest_2003_q2.parquet

# Extract the HTML for a filtered subset of segments
python scripts/extract_segments.py \
    --manifest manifest_2003_q2.parquet \
    --tar-root /path/to/daily_filings \
+   --output-dir ./html_segments
```

Once extracted, the resulting files can be wrapped with `EdgarFiling` to access
plain text, Markdown-, or DataFrame-rendered tables suitable for LLM prompts.
