# Pipeline Map (Locked Recursive Workflow)

This repository now has one supported extraction flow.

## Stage graph

```mermaid
flowchart LR
  A[ingest] --> B[normalize]
  B --> C[index-v2]
  C --> D[retrieve-v2]
  D --> E1[structured-v2 pricing]
  D --> E2[structured-v2 covenant]
  E1 --> F1[definitions-compiler-v1 pricing]
  E2 --> F2[definitions-compiler-v1 covenant]
  F1 --> G1[blocking-terms-compiler-v1 pricing]
  F2 --> G2[blocking-terms-compiler-v1 covenant]
  G1 --> H1[compustat-overlay-v1 pricing]
  G2 --> H2[compustat-overlay-v1 covenant]
  D --> I[agreement-metadata]
  E1 --> J[analysis-export-v2]
  E2 --> J
  F1 --> J
  F2 --> J
  G1 --> J
  G2 --> J
  H1 --> J
  H2 --> J
  I --> J
```

## Canonical orchestrator

- `pipeline all-v2-full`

The full orchestrator is the only supported batch flow. It enforces the strict recursive methodology for pricing and covenant outputs before export.

## Artifact contract

- `runs/<run_id>/indexing_v2/<item_id>_anchors.json`
- `runs/<run_id>/retrieval_v2/<item_id>_snippets.jsonl`
- `runs/<run_id>/llm_qa/<subdir>/<item_id>.json`
- `runs/<run_id>/definitions_compiler_v1/<subdir>/<item_id>__compiled.json`
- `runs/<run_id>/blocking_terms_compiler_v1/<subdir>/<item_id>__compiled.json`
- `runs/<run_id>/compustat_overlay_v1/<subdir>/<item_id>__compustat_overlay.json`
- `runs/<run_id>/agreement_metadata/<subdir>/<item_id>.json`
- `runs/<run_id>/analysis_export/analysis_export_v2/<item_id>.json`
- `runs/<run_id>/analysis_export/analysis_export_v2/records.jsonl`

## Retired workflows

Retired and unsupported in the codebase:

- `toc-v1`
- `definitions-v2`
- `analysis-export` (v1)
- `contract-pricing*`
- `contractir-v0-2*`
