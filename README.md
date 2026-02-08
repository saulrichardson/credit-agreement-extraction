# EDGAR LLM QA Pipeline

Purpose-built for getting accurate, verifiable answers from LLMs over SEC EDGAR filings. Every run leaves an evidence trail (anchors → snippets → outputs) so answers can be checked.

For a newcomer-oriented walkthrough of the two core extraction pipelines (financial covenants + pricing-kernel), see `PIPELINES_COVENANT_AND_PRICING.md`.

## Design principles
- **Evidence first:** Normalize filings, mark anchors, and keep surrounding snippets so every answer has a provenance trail.
- **Run-scoped artifacts:** All artifacts stay under `runs/<run_id>/`; inputs and prompts are recorded in the manifest. Re-running with the same `run_id` overwrites artifacts, so pick a new `run_id` to keep prior runs.
- **Precise targeting:** Filters (accession/cik/form/exhibit/date) decide what to ingest—no uncontrolled crawling.
- **Pluggable LLM steps:** Indexing and QA/extraction are clear hook points for your LLM or gateway.

## Methodology (one consistent contract)

This repo uses a single methodology: **run-scoped, anchor-grounded evidence** + deterministic evidence packaging + strict, validated JSON artifacts.

The codebase contains multiple pipelines that produce different artifact families (e.g., discovery JSON vs computable kernels vs pricing regime coverage), but they should all follow the same evidence/validation contract.

Start here:
- `METHODOLOGY.md`
- `PIPELINES_COVENANT_AND_PRICING.md`
- `ROADMAP.md`
- `design/PIPELINE_MAP.md` (functional stage map + artifact graph)

Legacy snapshots (kept out of the canonical branch):
- `legacy/v1-run-scoped` — older v1 `ingest → normalize → index → retrieve → structured` CLI flow
- `legacy/pricing-as-code-v1` — priority-ordered “pricing as code” model

## Run it
```bash
poetry install

poetry run pipeline all-v2 \
  --run-id demo \
  --tarball data/20230103.nc.tar.gz \
  --filters filters/test_sample_2_agreements.json \
  --gateway-url http://127.0.0.1:8000

# Filters
- By default, the pipeline accepts every document in the whitelisted filings (`keep_all`).
- Provide `--filters path/to/spec.json` (JSON/YAML with `doc_filter_path: "module:function"` and optional `doc_filter_kwargs`) to restrict exhibits. When `doc_filter_kwargs` is present, `doc_filter_path` is treated as a factory that returns a predicate.
- For curated allowlists, use `pipeline.filters:keep_item_ids_from_file` with `doc_filter_kwargs: {path: "datasets/your_dataset.json"}` (expects `item_ids: [...]`).
- For exhibit-type filtering, use `pipeline.filters:keep_doc_type_prefixes` with `doc_filter_kwargs: {prefixes: ["EX-10"]}`.
- Compose filters with `pipeline.filters:all_of`, `pipeline.filters:any_of`, and `pipeline.filters:not_`.
```

## Artifacts
```
runs/<run_id>/
  ingest/       # extracted filing HTML
  normalized/   # canonical text, anchors, prompt_view.txt
  toc_v1/       # optional chunk TOC (LLM-produced)
  indexing_v2/  # anchor buckets (LLM-produced; strict JSON)
  retrieval_v2/ # evidence snippet packs (deterministic)
  llm_qa/       # extraction outputs (prompt-driven; typically strict JSON)
  definitions_v2/ # definition grounding outputs
  agreement_metadata/ # parties/roles + facility headline terms
  analysis_export/ # analysis-ready record joins
  contract_pricing/ # pricing regime compiler outputs
  contractir_v0_2/ # pricing kernel (ContractIR v0.2) outputs
  manifest.json # filters, prompts, accessions, paths
```

## Source layout
The codebase is organized by pipeline role, not by historical iteration:

```
src/pipeline/
  core/      # run paths, manifest helpers, anchor loading, run_id validation
  llm/       # gateway client + strict JSON retry policy
  evidence/  # ingest, normalize, toc, indexing, retrieval, excerpt packs
  extract/   # structured extraction, definitions grounding, metadata/facility extraction
  pricing/   # pricing compiler/planner/doc_ir/schema modules
  compile/   # definitions/blocking-term/overlay compilers
  ir/        # ContractIR and CovenantIR validators/evaluators/flows
  cli/       # Click command surface
```

This layout is strict-forward: modules consume canonical run artifacts (`runs/<run_id>/...`) and do not include legacy module-path aliases.

## Status
LLM calls run via the local `agent-gateway` submodule. Canonical pipeline stages enforce strict JSON outputs and fail loudly on schema/provenance errors (see `METHODOLOGY.md`).

## Gateway-powered indexing (optional)
- The repo now vendors the lightweight [agent-gateway](agent-gateway) as a git submodule. After cloning run `git submodule update --init --recursive`, then start it with your keys (`cd agent-gateway && make serve`) so the pipeline can call `/v1/responses`.
- Model policy (enforced): all gateway calls use `openai:gpt-5-nano` with `reasoning=medium`. CLI flags/environment won’t override this.
- Run indexing to enable LLM anchor selection: `pipeline index-v2 --run-id demo --prompt prompts/indexing_v2.txt --gateway-url http://127.0.0.1:8000`. The run fails fast if the gateway is unreachable or returns invalid JSON / invalid anchor IDs.
- Outputs land in `runs/<run_id>/indexing_v2/{item_id}_anchors.json` as a **selection-only** artifact (anchor IDs grouped by category). Retrieval joins that selection with `normalized/<item_id>/anchors.tsv` (spans) to render snippet packs.

### Indexing v2

The v2 indexing prompt returns **streamlined anchor buckets** (arrays of anchor IDs) with an optional full definitions section span for downstream definition work.

```bash
pipeline index-v2 \
  --run-id demo \
  --prompt prompts/indexing_v2.txt \
  --gateway-url http://127.0.0.1:8000
```

Artifacts:
- `runs/<run_id>/indexing_v2/{item_id}_anchors.json`

To run retrieval + structured on v2 outputs:

```bash
pipeline retrieve-v2 --run-id demo
pipeline structured-v2 --run-id demo --prompt prompts/prompt_pricing_second_pass_dg_nano_v2_tuned_v2.txt --gateway-url http://127.0.0.1:8000
```

Artifacts:
- `runs/<run_id>/retrieval_v2/{item_id}_snippets.jsonl`
- `runs/<run_id>/llm_qa/<prompt_stem>/{item_id}.json`

Notes:
- `index-v2` may deterministically add signature-page anchors into `metadata_anchors` when it detects an execution block (e.g., “IN WITNESS WHEREOF”). These additions are recorded in `auto_added_anchors` inside `runs/<run_id>/indexing_v2/{item_id}_anchors.json` so the behavior is not silent.
- `index-v2` can optionally emit `key_date_definitions_anchors` (e.g., “Maturity Date means …”) so retrieval can include those targeted definitions without sending the entire `definitions_anchor_range`.

### TOC v1 (Hebbia-style chunk TOC; no embeddings)

This stage builds a **chunk-level table of contents** over the whole agreement using:
- deterministic adaptive chunking over anchors
- reasoning-first LLM summaries (no embeddings)

It is designed so retrieval can attach a stable, high-level “where am I in the document?” label to every snippet.

```bash
pipeline toc-v1 \
  --run-id demo \
  --prompt-chunk prompts/toc_chunk_v1.txt \
  --prompt-high-level prompts/toc_high_level_v1.txt \
  --gateway-url http://127.0.0.1:8000
```

Artifacts:
- `runs/<run_id>/toc_v1/<output_subdir>/{item_id}.json`

When present, `retrieve-v2` tags each snippet record with:
- `toc_chunk_id`
- `toc_title`

### Definitions v2 (pipeline-native, dg-driven term resolution)

Definitions v2 resolves **exact agreement definition language** for the terms used in your structured outputs (metrics + rate terms).
It is designed to avoid a separate per-term “canonicalization” pass by requiring the dg v2 output to include:
- `metrics[*].contract_term` (verbatim substring from the snippets when available)
- `metrics[*].search_tokens` (3–6 verbatim tokens/phrases to locate the definition in full text)

Run it after `structured-v2`:

```bash
pipeline definitions-v2 \
  --run-id demo \
  --qa-subdir <structured_output_subdir> \
  --definitions-prompt prompts/definitions_v2_metrics_rates.txt \
  --gateway-url http://127.0.0.1:8000
```

Artifacts:
- `runs/<run_id>/definitions_v2/<output_subdir>/*__definition.json`
- `runs/<run_id>/definitions_v2/<output_subdir>/*__context.txt`

This stage fails loudly (no silent skipping) when dg outputs are missing required term fields or when definitions cannot be grounded.

### Facility fundamentals (facility separation + key dates)

This stage extracts a dedicated **facility fundamentals** record from the v2 snippet pack:
- Multi-facility separation (one facility object per distinct facility/tranche when terms differ)
- Agreement-level Closing Date / Effective Date when explicitly stated (including relative definitions)
- Facility-level availability + maturity/termination, capturing **relative/conditional** language verbatim (no computed dates)

```bash
pipeline facility-fundamentals \
  --run-id demo \
  --prompt prompts/facility_fundamentals_v1.txt \
  --gateway-url http://127.0.0.1:8000
```

Artifacts:
- `runs/<run_id>/facility_fundamentals/<output_subdir>/{item_id}.json`
- `runs/<run_id>/facility_fundamentals/<output_subdir>/{item_id}.meta.json`

Notes:
- Indexing v2 supports an optional targeted bucket `key_date_definitions_anchors` so definitions like “Maturity Date means …” can be included in retrieval without sending the entire definitions span.

### Agreement metadata (parties/roles + facility headline terms)

This stage extracts agreement metadata from the v2 snippet pack:
- Parties + roles (borrowers / guarantors / agents / arrangers / lenders when explicitly enumerated)
- Facility headline terms (types, commitments, currency, maturity/termination dates) **only when explicitly supported**

```bash
pipeline agreement-metadata \
  --run-id demo \
  --prompt prompts/agreement_metadata_v1.txt \
  --gateway-url http://127.0.0.1:8000
```

Artifacts:
- `runs/<run_id>/agreement_metadata/<output_subdir>/{item_id}.json`
- `runs/<run_id>/agreement_metadata/<output_subdir>/{item_id}.meta.json`

Notes:
- Agreement metadata is evidence-validated. If the model cites the wrong anchor ID for a party name (but the name appears elsewhere in the snippet pack), the pipeline may deterministically re-point `source_refs` to a supporting anchor and record the change in `{item_id}.meta.json` under `auto_corrections`.
- Currency is only kept when explicitly supported by the cited anchors; otherwise it is nulled (and recorded) rather than guessed.
- Lenders are additionally extracted deterministically from signature-page patterns (e.g., “as a Lender”) when present in the snippet pack; this is recorded in `{item_id}.meta.json` under `auto_corrections` (no silent enrichment).

### Analysis export (single record per agreement)

This stage produces an analysis-ready record per agreement (and a JSONL bundle) by joining:
- Agreement metadata outputs
- Definitions-v2 outputs
- Source pointers (canonical text + anchors + snippet packs)

```bash
pipeline analysis-export \
  --run-id demo \
  --agreement-metadata-subdir <agreement_metadata_output_subdir> \
  --definitions-v2-subdir <definitions_v2_output_subdir>
```

Artifacts:
- `runs/<run_id>/analysis_export/<output_subdir>/{item_id}.json`
- `runs/<run_id>/analysis_export/<output_subdir>/records.jsonl`
- `runs/<run_id>/analysis_export/<output_subdir>/export.meta.json`
- `runs/<run_id>/analysis_export/<output_subdir>/tables/*.csv`

### All-v2 flow (one command end-to-end)

Runs ingest → normalize → index-v2 → retrieve-v2 → structured-v2 → definitions-v2 → agreement-metadata → analysis-export:

Note: `filters/test_sample_2_agreements.json` currently selects **18 documents (`item_id`s)** across **17 SEC filings (`accession`s)** (one filing contains two EX-10 exhibits). Most pipeline stages operate on `item_id` as the unit of work.

```bash
pipeline all-v2 \
  --run-id demo \
  --tarball data/20230103.nc.tar.gz \
  --filters filters/test_sample_2_agreements.json \
  --gateway-url http://127.0.0.1:8000
```

## Contract pricing compiler (new workflow; no indexing/retrieval)
This workflow extracts **all pricing regimes** and attaches pricing to **rate options** by:
- Building a deterministic `doc_ir` from `normalized/<item_id>/canonical_annotated.txt` (anchors + parsed tables)
- Running a **per-table** LLM compilation pass into a strict JSON schema (Pydantic-validated; 3 attempts then fail)

End-to-end (single command):
```bash
pipeline contract-pricing-flow \
  --run-id dan \
  --tarball data/20230103.nc.tar.gz \
  --filters filters/test_sample_2_agreements.json \
  --prompt prompts/contract_pricing_table_v1.txt \
  --gateway-url http://127.0.0.1:8000
```

Manual (explicit stages):
```bash
pipeline ingest --run-id dan --tarball data/20230103.nc.tar.gz --filters filters/test_sample_2_agreements.json
pipeline normalize --run-id dan
pipeline contract-pricing --run-id dan --prompt prompts/contract_pricing_table_v1.txt --gateway-url http://127.0.0.1:8000
```

Artifacts:
- `runs/<run_id>/doc_ir/<item_id>.json` (deterministic parsing output; tables + nearby anchor text)
- `runs/<run_id>/contract_pricing/<subdir>/<item_id>.json` (validated pricing IR)
- `runs/<run_id>/contract_pricing/_context/<subdir>/<item_id>.json` (what we sent into the tablepass)

Details: see `docs/contract-pricing-compiler.md`.
