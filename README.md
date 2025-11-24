# EDGAR LLM Pipeline

Run-scoped pipeline for turning SEC EDGAR daily tarballs into LLM-ready chunks and outputs. Filters decide which filings/exhibits to pull; the same flow works for any type.

## What it does
- Ingest filtered filings from .nc tarballs.
- Normalize HTML to text, split into anchors, and produce snippets.
- Hand off to your LLM for indexing and QA/extraction (pluggable hooks).

## Run it
```bash
poetry install

# end-to-end
pipeline all --run-id demo \
  --tarball data/20230103.nc.tar.gz \
  --filters filters.yaml \
  --prompt-index prompts/prompt_all_comprehensive_v2.txt \
  --prompt-structured prompts/dg_v3.txt
```

## Artifacts
```
runs/<run_id>/
  ingest/       # extracted filing HTML
  normalized/   # canonical text, anchors, prompt_view.txt
  indexing/     # anchor JSON (LLM-produced)
  retrieval/    # snippets
  llm_qa/       # LLM QA / extraction outputs
  validation/   # optional QA outputs
  deliverables/ # final rollups
  manifest.json # filters, prompts, accessions, paths
```

## Status
LLM calls are stubbed—wire your client into `pipeline/indexing.py` and `pipeline/structured.py` (llm_qa stage). Errors surface early to avoid silent failures.
