# Lands' End Chunk Run

This folder holds a fresh canonical bundle and chunk plan derived from the source HTML in `docs/artifacts/source_html/lands_end_term_loan_ex4_1.html`. Use these staged artifacts for the chunk-first workflow described in `docs/artifacts/chunking/README.txt`.

## Generated artifacts

| Artifact | Path |
| --- | --- |
| Canonical bundle (text + anchors + prompt view) | `runs/lands_end_chunk_run/canonical/` |
| Chunk plan (size=40, stride=20) | `runs/lands_end_chunk_run/chunks/chunks_40_20.json` |

## Runbook (chunking approach)

1. **(Done)** Build canonical bundle:
   ```bash
   poetry run edgar-pipeline anchor-document \
     --input-html docs/artifacts/source_html/lands_end_term_loan_ex4_1.html \
     --output-dir runs/lands_end_chunk_run/canonical
   ```
2. **(Done)** Generate chunk windows:
   ```bash
   poetry run edgar-pipeline chunk-document \
     --canonical-dir runs/lands_end_chunk_run/canonical \
     --output runs/lands_end_chunk_run/chunks/chunks_40_20.json \
     --chunk-size 40 --stride 20
   ```
3. **Score chunks (needs OPENAI_API_KEY)**
   ```bash
   export OPENAI_API_KEY=sk-...
   poetry run edgar-pipeline score-plan \
     --plan runs/lands_end_chunk_run/chunks/chunks_40_20.json \
     --output runs/lands_end_chunk_run/chunks/chunks_40_20_scores.json \
     --question "Explain the performance pricing / applicable margin structure."
   ```
4. **Feed the scored chunks into downstream prompts** (e.g., pricing extraction):
   ```bash
   poetry run python scripts/run_pricing_extraction.py chunk \
     --prompt docs/artifacts/prompts/extraction.txt \
     --prompt-view runs/lands_end_chunk_run/canonical/prompt_view.txt \
     --chunk-scores runs/lands_end_chunk_run/chunks/chunks_40_20_scores.json \
     --output runs/lands_end_chunk_run/chunks/chunk_pricing.json
   ```

Replace the question/prompt paths as needed. The scoring/extraction steps will fail until an OpenAI key is present in the environment.
