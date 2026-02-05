# Definition Compiler v2 (AST-first) — Schema + Prompts

This repo now supports an **AST-first** definition representation that is **Compustat-free** at extraction time (Compustat can be added later as an overlay stage).

The goal of this redesign is:
- Extract a grounded **verbatim definition** for a target metric/term.
- Represent the **calculation structure** as a best-effort **expression AST**.
- Identify the other defined terms that are **numeric inputs** to compute the target (the recursion driver).
- Capture non-math **clauses** (GAAP/consolidation/timing/exclusions/etc.) strictly as verbatim substrings of the definition.

## Prompts

- Definitions compiler prompt (metrics/terms):
  - `prompts/definitions_compiler_v1_metrics_ast_v2.txt`
- Blocking terms compiler prompt (used to resolve input terms):
  - `prompts/blocking_terms_compiler_v1_ast_v2.txt`

Both prompts output the same schema.

## JSON Schema (single-target)

Top-level:
```json
{
  "schema_version": "definition_compiler_v2_ast_v1",
  "definitions": [ { "name": "...", "...": "..." } ],
  "unresolved_dependencies": [ { "term": "...", "referenced_by": ["..."] } ]
}
```

`definitions[]` (exactly one object):
- `name`: must equal the input target term name
- `contract_term`: the defined term as written in the agreement (may differ from `name`)
- `definition_verbatim`: verbatim definition text (or null)
- `expression_ast`: best-effort AST (or null)
- `input_terms`: other defined terms that are numeric inputs to compute the target (list)
- `clauses`: verbatim, non-math computation conditions (list of `{kind,text}`)
- `source_refs`: anchor IDs covering the definition language
- `needs_more_context`: boolean flag when definition appears incomplete in provided blocks
- `confidence`: high|medium|low
- `notes`: list of short strings

`unresolved_dependencies` is provided by the LLM and validated for internal consistency with `input_terms` (the validator does not repair or synthesize missing dependencies).

## Validator (strict; no repairs)

The v2 validator lives in `src/pipeline/definitions_compiler_v1.py` as:
- `_validate_compiler_output_v2_ast(...)`

Key behaviors:
- Enforces the strict top-level + per-definition JSON shape (no extra keys, no missing keys).
- Verifies `definition_verbatim` grounding against the provided contexts (with conservative normalization for quotes/whitespace/punctuation).
- Verifies `source_refs` coverage of `definition_verbatim` (definition must be a substring of the concatenated cited anchors).
- Enforces substring grounding rules for `input_terms[]`, `clauses[].text`, and all text-bearing `expression_ast` nodes.
- Enforces internal consistency: `expression_ast` term leaves must appear in `input_terms[]`, and `unresolved_dependencies` must match `input_terms[]` (deduped).

## Smoke-tested run (gateway)

Validated on real agreements in:
- `runs/defgraph-iter-20260129/definitions_compiler_v1/compiled_metrics_v2_ast_v1_validated2/`
- `runs/defgraph-iter-20260129/blocking_terms_compiler_v1/blocking_terms_v2_ast_v1_validated2_depth1/`

These were produced by running the CLI through the gateway (no local stubbing).

## How to run (example)

Definitions compiler (per-run, limited item_ids):
```bash
poetry run pipeline definitions-compiler-v1 \
  --run-id defgraph-iter-20260129 \
  --qa-subdir dg_strict_metric_consistency_v4_full \
  --compiler-prompt prompts/definitions_compiler_v1_metrics_ast_v2.txt \
  --output-subdir compiled_metrics_v2_ast_v1_validated2 \
  --item-id 0001193125-23-000306_2 \
  --item-id 0000950170-23-000122_2
```

Resolve dependency terms (depth 1):
```bash
poetry run pipeline blocking-terms-compiler-v1 \
  --run-id defgraph-iter-20260129 \
  --metrics-output-subdir compiled_metrics_v2_ast_v1_validated2 \
  --term-compiler-prompt prompts/blocking_terms_compiler_v1_ast_v2.txt \
  --output-subdir blocking_terms_v2_ast_v1_validated2_depth1 \
  --max-depth 1 \
  --max-terms 50 \
  --item-id 0001193125-23-000306_2 \
  --item-id 0000950170-23-000122_2
```

## Compustat overlay (optional; post-processing)

The **definition compiler v2 AST schema is Compustat-free at extraction time**.

If you want a best-effort mapping to a Compustat-style quarterly formula, run the overlay stage after you have:
- metric definitions (from `definitions-compiler-v1`), and optionally
- recursively resolved dependency-term definitions (from `blocking-terms-compiler-v1`).

The overlay runs one LLM call per *term* and validates that any produced formula uses only an allowlisted set of
quarterly Compustat variables.

Prompt:
- `prompts/compustat_overlay_v1.txt`
Allowlist (edit/expand this to widen the mapping space):
- `datasets/compustat_allowlist_quarterly_v1.json`
- (optional) `datasets/compustat_allowlist_quarterly_q_only_v1.json` (only codes ending in `q`)

Example (overlay both metric definitions and dependency-term definitions):
```bash
poetry run pipeline compustat-overlay-v1 \
  --run-id defgraph-iter-20260129 \
  --overlay-prompt prompts/compustat_overlay_v1.txt \
  --metrics-output-subdir compiled_metrics_v2_ast_v1_validated2 \
  --blocking-terms-output-subdir blocking_terms_v2_ast_v1_validated2_depth1 \
  --output-subdir compustat_overlay_v1_defgraph_iter_20260129
```

Artifacts:
- `runs/<run_id>/compustat_overlay_v1/<output_subdir>/<item_id>__<term_slug>__compustat_overlay.json`
- `runs/<run_id>/compustat_overlay_v1/<output_subdir>/<item_id>__compustat_overlay.json`
