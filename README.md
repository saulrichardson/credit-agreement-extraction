# Credit Agreement Extraction Pipeline (Locked Recursive Workflow)

This repository now supports one production workflow only:

`ingest -> normalize -> index-v2 -> retrieve-v2 -> structured-v2 (pricing + covenant) -> recursive definitions compiler -> recursive blocking terms compiler -> compustat overlay -> agreement-metadata -> analysis-export-v2`

Legacy workflows have been retired and are intentionally unsupported.

## Install

```bash
poetry install
```

## End-to-end run

```bash
poetry run pipeline all-v2-full \
  --run-id demo-recursive \
  --tarball /path/to/filings.tar.gz \
  --item-ids-file /path/to/item_ids.json \
  --gateway-url http://127.0.0.1:8000
```

`all-v2-full` is the canonical orchestrator. It runs the full recursive methodology and exports analysis records.

## Slurm orchestration

Use `scripts/slurm_schedule_pipeline.py` for shard scheduling on Torch/Slurm.

```bash
python scripts/slurm_schedule_pipeline.py \
  --workflow all-v2-full \
  --run-id-prefix debt-batch-a \
  --item-ids-file /path/to/item_ids.json \
  --tarball /path/to/filings_1.tar.gz \
  --tarball /path/to/filings_2.tar.gz \
  --base-dir /path/to/repo \
  --submit
```

## Run artifacts

```text
runs/<run_id>/
  ingest/
  normalized/
  indexing_v2/
  retrieval_v2/
  llm_qa/
  definitions_compiler_v1/
  blocking_terms_compiler_v1/
  compustat_overlay_v1/
  agreement_metadata/
  analysis_export/analysis_export_v2/
  manifest.json
```

## Core commands

```bash
poetry run pipeline --help
```

Available command surface is intentionally minimal:

- `ingest`
- `normalize`
- `index-v2`
- `retrieve-v2`
- `structured-v2`
- `definitions-compiler-v1`
- `blocking-terms-compiler-v1`
- `compustat-overlay-v1`
- `compustat-overlay-eval-v1`
- `agreement-metadata`
- `analysis-export-v2`
- `all-v2-full`

## Viewer

Render a per-item stage viewer with prompts and anchor-linked agreement text:

```bash
python scripts/render_run_stage_viewer_html.py \
  --base-dir . \
  --run-id <run_id>
```

The viewer is focused on core LLM outputs for the locked workflow:

- indexing
- agreement metadata
- pricing structured + recursive metrics + recursive overlay
- covenant structured + recursive metrics + recursive overlay

## Non-goals

- Backward compatibility with retired workflows.
- Supporting legacy `definitions-v2`, `analysis-export-v1`, `toc-v1`, `contract-pricing`, or ContractIR CLI paths.
