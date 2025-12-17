# EDGAR LLM QA Pipeline

Purpose-built for getting accurate, verifiable answers from LLMs over SEC EDGAR filings. Every run leaves an evidence trail (anchors → snippets → outputs) so answers can be checked.

## Design principles
- **Evidence first:** Normalize filings, mark anchors, and keep surrounding snippets so every answer has a provenance trail.
- **Run-scoped artifacts:** All artifacts stay under `runs/<run_id>/`; inputs and prompts are recorded in the manifest. Re-running with the same `run_id` overwrites artifacts, so pick a new `run_id` to keep prior runs.
- **Precise targeting:** Filters (accession/cik/form/exhibit/date) decide what to ingest—no uncontrolled crawling.
- **Pluggable LLM steps:** Indexing and QA/extraction are clear hook points for your LLM or gateway.

## Run it
```bash
poetry install

pipeline all --run-id demo \
  --tarball data/20230103.nc.tar.gz \
  --prompt-index prompts/comprehensive_indexing.txt \
  --prompt-structured prompts/dg_v3.txt

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
  indexing/     # anchor JSON (LLM-produced)
  retrieval/    # evidence snippets
  llm_qa/       # LLM QA / extraction outputs
  validation/   # optional QA checks
  deliverables/ # final rollups
  manifest.json # filters, prompts, accessions, paths
```

## Status
LLM calls are stubbed; plug your client into `pipeline/indexing.py` and `pipeline/structured.py` (llm_qa stage). Errors surface early to avoid silent failures.

## Gateway-powered indexing (optional)
- The repo now vendors the lightweight [agent-gateway](agent-gateway) as a git submodule. After cloning run `git submodule update --init --recursive`, then start it with your keys (`cd agent-gateway && make serve`) so the pipeline can call `/v1/responses`.
- Model policy (enforced): all gateway calls use `openai:gpt-5-nano` with `reasoning=medium`. CLI flags/environment won’t override this.
- Run indexing to enable LLM anchor selection: `pipeline index --run-id demo --prompt prompts/comprehensive_indexing.txt --gateway-url http://127.0.0.1:8000`. The run fails fast if the gateway is unreachable or returns zero anchors. `--max-anchors` has been removed; all anchors are sent.
- Outputs land in `runs/<run_id>/indexing/{item_id}_anchors.json` as a **selection-only** artifact (anchor IDs grouped by category). Retrieval joins that selection with `normalized/<item_id>/anchors.tsv` (spans) to render snippets.

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
