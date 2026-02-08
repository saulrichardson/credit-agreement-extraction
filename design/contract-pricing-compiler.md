# Contract Pricing Compiler (Workflow)

This repo now has a **separate, indexing-free** workflow for extracting pricing from credit agreements.

The goal is to encode:
- **All pricing regimes** (amendments, sustainability pricing, alternate grids, default-rate regimes, etc.)
- Pricing attached to **rate options** (loan types and fee labels)
- With **strict, evidence-cited JSON** suitable for downstream reasoning and programmatic retrieval

This workflow is intentionally:
- **Reasoning-first** (LLM compiles meaning from tables/footnotes)
- **Anti-RAG** (no vector search; no embedding index)
- **Evidence grounded** (every extracted number should carry anchor references)
- **Strict** (JSON-only, Pydantic-validated, retry then fail)

## High-level flow

1) **Ingest**: pick which EDGAR exhibits to include in the run
2) **Normalize**: produce `canonical_annotated.txt` with anchor markers and `[[TABLE]]` blocks
3) **Doc IR** (deterministic): parse anchors + tables into a structured intermediate representation (`doc_ir`)
4) **Contract pricing** (LLM): compile each pricing table into a strict schema, then deterministically assemble regimes

Unlike the legacy v1 pipeline (preserved on branch `legacy/v1-run-scoped`), this workflow skips indexing + snippet retrieval.

## How to run

### End-to-end (one command)

```bash
poetry run pipeline contract-pricing-flow \
  --run-id dan \
  --tarball data/20230103.nc.tar.gz \
  --filters filters/test_sample_2_agreements.json \
  --prompt prompts/contract_pricing_table_v1.txt \
  --gateway-url http://127.0.0.1:8000
```

### Explicit stages

```bash
poetry run pipeline ingest \
  --run-id dan \
  --tarball data/20230103.nc.tar.gz \
  --filters filters/test_sample_2_agreements.json

poetry run pipeline normalize --run-id dan

poetry run pipeline contract-pricing \
  --run-id dan \
  --prompt prompts/contract_pricing_table_v1.txt \
  --gateway-url http://127.0.0.1:8000
```

## Inputs / outputs

### Inputs

- `runs/<run_id>/normalized/<item_id>/canonical_annotated.txt`
  - Anchored blocks: `[[A0001]] ...`
  - Tables encoded as `[[TABLE]] ... [[/TABLE]]` inside anchors with `anchor_type == "table"`

### Deterministic IR

- `runs/<run_id>/doc_ir/<item_id>.json`
  - Produced by `src/pipeline/pricing/doc_ir.py`
  - Contains:
    - `anchors[]`: ordered blocks with anchor_id + text
    - `tables[]`: parsed tables (columns + row_label + cells), plus raw markdown payload

### Contract pricing outputs

- `runs/<run_id>/contract_pricing/<subdir>/<item_id>.json`
  - Produced by `src/pipeline/pricing/contract_pricing.py`
  - Strictly validated by Pydantic models in `src/pipeline/pricing/contract_schemas.py`
  - Model:
    - `pricing_regimes[]`: each regime contains `grids[]`, `adjustments[]`, and `flat_items[]`
    - `grid.cells[]` references `tier_id` + `rate_option_id` so downstream logic can reason over matrices

### Context bundles (debugging / provenance)

- `runs/<run_id>/contract_pricing/_context/<subdir>/<item_id>.json`
- `runs/<run_id>/contract_pricing/_context/<subdir>/<item_id>/<table_anchor_id>.json`

These are the exact inputs used for each per-table LLM call, kept so the pipeline is debuggable.

## Strictness + failure mode

The contract pricing step is configured as:
- **Strict JSON** (no salvage)
- **Pydantic validation** (extra keys forbidden)
- **3 attempts** per item (re-prompted with the prior error)
- Then **fail loudly**

This is intended to make failures obvious and to keep downstream systems simple.

## Notes

- The pricing compiler currently focuses on tables and table-adjacent footnotes/adjustments. If an agreement encodes pricing primarily in prose, the workflow may miss it.
- Table selection is heuristic; tightening this can reduce accidental inclusion of non-pricing tables.
