# EDGAR LLM QA Pipeline

Purpose-built for getting accurate, verifiable answers from LLMs over SEC EDGAR filings. Every run is isolated, reproducible, and leaves an evidence trail (anchors → snippets → outputs) so answers can be checked.

## Design principles
- **Evidence first:** Normalize filings, mark anchors, and keep surrounding snippets so every answer has a provenance trail.
- **Deterministic runs:** All artifacts stay under `runs/<run_id>/`; inputs and prompts are recorded in the manifest.
- **Precise targeting:** Filters (accession/cik/form/exhibit/date) decide what to ingest—no uncontrolled crawling.
- **Pluggable LLM steps:** Indexing and QA/extraction are clear hook points for your LLM or gateway.

## Run it
```bash
poetry install

pipeline all --run-id demo \
  --tarball data/20230103.nc.tar.gz \
  --prompt-index prompts/prompt_all_comprehensive_v2.txt \
  --prompt-structured prompts/dg_v3.txt

# Filters
- By default, the pipeline accepts every document in the whitelisted filings (`keep_all`).
- Provide `--filters path/to/spec.json` (JSON/YAML with `doc_filter_path: "module:function"`) to restrict exhibits, e.g., to EX-10 only.
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
- Run indexing with a model to enable LLM anchor selection: `pipeline index --run-id demo --prompt prompts/prompt_all_comprehensive_v2.txt --model openai:gpt-4o-mini --gateway-url http://127.0.0.1:8000`. If `--model` is omitted, the pipeline keeps all heuristic anchors. When a model is provided, the run now FAILS if the gateway is unreachable or returns zero anchors (no silent fallback). `--max-anchors` has been removed; all anchors are sent.
- Outputs land in `runs/<run_id>/indexing/{item_id}_anchors.json`, which retrieval consumes.
