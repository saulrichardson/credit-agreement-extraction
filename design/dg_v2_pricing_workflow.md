# DG v2 pricing workflow (traditional / PI track)

This repo has two pricing-related tracks:

- **DG v2 (traditional)**: normalize → index-v2 → retrieve-v2 → structured-v2 (DG JSON) → definitions-v2
- **ContractIR (imposed structure)**: separate namespace under `src/pipeline/contract_ir_*` (not covered here)

This document describes the **DG v2 (traditional)** flow, where artifacts are written under `runs/<run_id>/`.

## Key artifacts (where things live)

Given `--run-id <run_id>`:

- Normalized text (source-of-truth text for hit/anchor mapping)
  - `runs/<run_id>/normalized/<item_id>/canonical.txt`
- Indexing selections (anchor sets + definitions section range)
  - `runs/<run_id>/indexing_v2/<item_id>_anchors.json`
  - `selection.definitions_anchor_range` contains `{start_anchor,end_anchor}` when a definitions section is detected
- Retrieval snippet packs (used as LLM input for structured-v2)
  - `runs/<run_id>/retrieval_v2/<item_id>_snippets.jsonl`
- DG structured outputs (pricing JSON)
  - `runs/<run_id>/llm_qa/<qa_subdir>/<item_id>.txt`
- Standalone metric definitions compiler (metrics-only, AST-first)
  - `runs/<run_id>/definitions_compiler_v1/<compiler_subdir>/`
  - Per-metric artifacts:
    - `.../<item_id>__<metric_slug>__contexts.txt` (selected anchored evidence blocks)
    - `.../<item_id>__<metric_slug>__compile.raw.txt` (raw LLM response, attempt 1)
    - `.../<item_id>__<metric_slug>__compile.raw.attempt2.txt` (only present if a repair round was needed)
    - `.../<item_id>__<metric_slug>__compiled.json` (validated JSON)
  - Per-item aggregate:
    - `.../<item_id>__compiled.json`
  - Summaries:
    - `issues.txt` (non-fatal issues like “no metrics for this item”)
    - `errors.txt` (hard failures like “could not parse/validate JSON after retries”)
- Blocking terms compiler (recursive dependency term definitions; optional)
  - `runs/<run_id>/blocking_terms_compiler_v1/<terms_subdir>/`
  - Per-term artifacts mirror the metric compiler (`...__contexts.txt`, `...__compile.raw.txt`, `...__compiled.json`)
  - Per-item aggregate:
    - `.../<item_id>__compiled.json`
  - Dependency graph (debug/audit):
    - `.../<item_id>__dependency_graph.json`
- Compustat overlay (post-processing; optional)
  - `runs/<run_id>/compustat_overlay_v1/<overlay_subdir>/`
  - Per-term overlay:
    - `.../<item_id>__<term_slug>__compustat_overlay.json`
  - Per-item aggregate:
    - `.../<item_id>__compustat_overlay.json`
- Definitions outputs (per metric / per base-rate term)
  - `runs/<run_id>/definitions_v2/<definitions_subdir>/`
  - Files are per-term: `...__context.txt`, `...__definition.txt`, `...__definition.json`
  - Summaries:
    - `issues.txt` (non-fatal issues like “no terms for this item” or “term not found”)
    - `errors.txt` (hard failures like invalid JSON shape from the definitions model)
- Audits / reports
  - `runs/<run_id>/report/dg_json_audit_<qa_subdir>.json`

## Prompts (repo-local)

- Indexing v2 (anchor selection + definitions span)
  - `prompts/indexing_v2.txt`
- DG structured prompt variants (pricing JSON)
  - `prompts/prompt_pricing_second_pass_dg_nano_v2_tuned_v2_strict_metric_consistency_v2.txt`
  - `prompts/prompt_pricing_second_pass_dg_nano_v2_tuned_v2_strict_metric_consistency_v4.txt`
- Definitions v2 (per-term definitions JSON)
  - `prompts/definitions_v2_metrics_rates_anchorrefs_v1.txt`
- Standalone definitions compiler (metrics-only, AST-first)
  - `prompts/definitions_compiler_v1_metrics_ast_v2.txt`
- Blocking terms compiler (dependency terms, AST-first)
  - `prompts/blocking_terms_compiler_v1_ast_v2.txt`
- Compustat overlay (post-processing)
  - `prompts/compustat_overlay_v1.txt`
  - `datasets/compustat_allowlist_quarterly_v1.json` (expand this allowlist to widen mapping coverage)
  - `datasets/compustat_allowlist_quarterly_q_only_v1.json` (optional stricter variant: only codes ending in `q`)

## Recommended commands (manual)

Run the full v2 pipeline stages explicitly:

1) **Index-v2**

```bash
poetry run pipeline index-v2 \
  --run-id <run_id> \
  --prompt prompts/indexing_v2.txt
```

2) **Retrieve-v2**

```bash
poetry run pipeline retrieve-v2 --run-id <run_id>
```

3) **Structured-v2**

```bash
poetry run pipeline structured-v2 \
  --run-id <run_id> \
  --prompt prompts/prompt_pricing_second_pass_dg_nano_v2_tuned_v2_strict_metric_consistency_v4.txt \
  --output-subdir dg_strict_metric_consistency_v4_full
```

4) **Audit structured outputs**

```bash
poetry run python scripts/audit_dg_json_failures.py \
  --run-id <run_id> \
  --qa-subdir dg_strict_metric_consistency_v4_full
```

5) **Definitions-v2 (uses definitions anchor range when available)**

```bash
poetry run pipeline definitions-v2 \
  --run-id <run_id> \
  --qa-subdir dg_strict_metric_consistency_v4_full \
  --definitions-prompt prompts/definitions_v2_metrics_rates_anchorrefs_v1.txt \
  --output-subdir defs_anchorrefs_v1_metric_consistency_v4_full
```

Optional (recommended for PI-style metric definitions): **Definitions compiler v1 (metrics-only, AST-first)**

This runs **one LLM call per metric** (from the DG structured output `metrics[]`) using deterministic evidence selection:
- term + search_token hits in `canonical.txt`
- plus (when present) the indexing_v2 `selection.definitions_anchor_range` hint

```bash
poetry run pipeline definitions-compiler-v1 \
  --run-id <run_id> \
  --qa-subdir dg_strict_metric_consistency_v4_full \
  --compiler-prompt prompts/definitions_compiler_v1_metrics_ast_v2.txt \
  --output-subdir compiled_metrics_v2_ast_v1
```

Optional: recursively resolve dependency terms (depth 1 shown; increase if needed):

```bash
poetry run pipeline blocking-terms-compiler-v1 \
  --run-id <run_id> \
  --metrics-output-subdir compiled_metrics_v2_ast_v1 \
  --term-compiler-prompt prompts/blocking_terms_compiler_v1_ast_v2.txt \
  --output-subdir blocking_terms_v2_ast_v1_depth1 \
  --max-depth 1 \
  --max-terms 200
```

Optional: map extracted definitions to a Compustat-style quarterly formula (overlay stage):

```bash
poetry run pipeline compustat-overlay-v1 \
  --run-id <run_id> \
  --overlay-prompt prompts/compustat_overlay_v1.txt \
  --metrics-output-subdir compiled_metrics_v2_ast_v1 \
  --blocking-terms-output-subdir blocking_terms_v2_ast_v1_depth1 \
  --output-subdir compustat_overlay_v1
```

## One-command runner (script)

For repeatable experiments, use:

```bash
poetry run python scripts/run_dg_v2_pricing_flow.py \
  --run-id <run_id> \
  --structured-prompt prompts/prompt_pricing_second_pass_dg_nano_v2_tuned_v2_strict_metric_consistency_v4.txt \
  --qa-subdir dg_strict_metric_consistency_v4_full \
  --definitions-prompt prompts/definitions_v2_metrics_rates_anchorrefs_v1.txt \
  --definitions-subdir defs_anchorrefs_v1_metric_consistency_v4_full
```

Optional: include the standalone metric definitions compiler in the same run:

```bash
poetry run python scripts/run_dg_v2_pricing_flow.py \
  --run-id <run_id> \
  --qa-subdir dg_strict_metric_consistency_v4_full \
  --compiler-prompt prompts/definitions_compiler_v1_metrics_ast_v2.txt \
  --compiler-subdir compiled_metrics_v2_ast_v1
```

Optional: restrict to a subset of items:

```bash
poetry run python scripts/run_dg_v2_pricing_flow.py \
  --run-id <run_id> \
  --items 0000950144-96-000010_7,0000950170-23-000122_2 \
  --qa-subdir experiment_20260112 \
  --structured-prompt prompts/prompt_pricing_second_pass_dg_nano_v2_tuned_v2_strict_metric_consistency_v4.txt
```

## Notes on recent robustness fixes

- `src/pipeline/structured_v2.py` enforces **strict JSON output** (retry-then-fail). On repeated failure it writes a `.raw.txt` sidecar and a per-item `.error.txt`.
- `src/pipeline/structured_v2.py` does **not** drop `definitions` bucket snippets by default. If you want to exclude definitions section text from a structured prompt, pass an explicit category filter when running `structured-v2`.
- `src/pipeline/definitions_v2.py` uses `selection.definitions_anchor_range` (when present) to prefer term hits inside the definitions span.
- `src/pipeline/definitions_v2.py` treats “no terms” and “term not found” as **issues** (recorded in `issues.txt`) rather than hard failures.
