# EDGAR LLM Pipeline (run-scoped)

Run-scoped, deterministic pipeline to pull SEC EDGAR filings from daily tarballs, normalize them, carve out anchors/snippets, and drive LLM-based QA/extraction. Works for any filing/exhibit type the filters select—EX‑10 is just a default example.

## Key ideas
- **Run-scoped artifacts:** `runs/<run_id>/` holds the full chain (ingest → normalized → indexing → retrieval → llm_qa → validation → deliverables). No shared scratch.
- **Deterministic + fail-fast:** Required inputs must exist; empty filters raise early errors.
- **Explicit targeting:** Ingest uses a JSON/YAML filter spec (accession/cik/form/exhibit glob/date) so you only process what you ask for.
- **LLM pluggable points:** Prompts live in `prompts/`; indexing and llm_qa stages are clear integration hooks for your gateway/LLM.

## Quick start
```bash
poetry install

# Ingest a tarball for specific accessions
pipeline ingest --run-id demo --tarball data/20230103.nc.tar.gz --accessions-file acc.txt

# Build prompt views for that run
pipeline normalize --run-id demo

# Index (anchors) with a chosen prompt
pipeline index --run-id demo --prompt prompts/prompt_all_comprehensive_v2.txt

# Render snippets
pipeline retrieve --run-id demo --bandwidth 4

# LLM QA / extraction
pipeline structured --run-id demo --prompt prompts/dg_v3.txt
```

## Layout
```
runs/<run_id>/
  ingest/         # extracted filing HTML
  normalized/     # prompt_view.txt + anchors.tsv + canonical.txt
  indexing/       # anchor JSON (LLM-produced)
  retrieval/      # snippets around anchors
  llm_qa/         # LLM QA / extraction outputs
  validation/     # QA outputs (optional)
  deliverables/   # final rollups
  manifest.json   # filters, prompts, accessions, paths
```

## Status
This is a clean scaffolding. Gateway/LLM calls are stubbed; wire your client into `pipeline/indexing.py` and `pipeline/structured.py` (`llm_qa` stage). The code fails fast when required inputs are missing to avoid silent degradation.
