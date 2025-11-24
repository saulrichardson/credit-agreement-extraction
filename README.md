# Credit Agreement Pipeline (clean run-scoped build)

This repo provides a minimal, deterministic pipeline to extract EX‑10 exhibits from SEC daily tarballs, build prompt views, run anchor indexing, render snippets, and perform structured extraction — all scoped to a single `run_id` folder.

## Key ideas
- **Run-scoped artifacts only:** Everything lives under `runs/<run_id>/` (ingest → normalized → indexing → retrieval → structured → validation → deliverables). No global scratch.
- **Fail fast:** Missing inputs or empty filters raise errors.
- **Explicit filters:** Ingest uses a JSON/YAML filter spec (accession/cik/glob/date) to avoid pulling whole tarballs.
- **No hidden magic:** Prompts are explicit files in `prompts/`; pipelines take paths as args.

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

# Structured extraction
pipeline structured --run-id demo --prompt prompts/dg_v3.txt
```

## Layout
```
runs/<run_id>/
  ingest/         # extracted EX‑10 HTML
  normalized/     # prompt_view.txt + anchors.tsv + canonical.txt
  indexing/       # anchor JSON
  retrieval/      # snippets
  structured/     # structured JSON
  validation/     # QA outputs (optional)
  deliverables/   # final rollups
  manifest.json   # filters, prompts, accessions, paths
```

## Status
This is a clean scaffolding. Gateway/LLM calls are stubbed for now; integration should be wired into `pipeline/indexing.py` and `pipeline/structured.py` where indicated. The code fails if required inputs are missing to avoid silent degradation.
