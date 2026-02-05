# Covenant + Pricing Pipelines (newcomer guide)

This repo has two “core extraction” pipelines relevant to credit agreements:

1. **Financial covenants** → CovenantIR v0.1 (computable boolean compliance tests)
2. **Pricing** (benchmark/base rate + spread/margin + fee rates) → ContractIR v0.2 “pricing kernel” (multi-pass, merged)

Note: there is also a separate **contract pricing compiler** workflow (`pipeline contract-pricing-flow` / `pipeline contract-pricing`)
that compiles pricing from normalized tables; this guide is specifically about the CovenantIR + ContractIR v0.2 pipelines.

Both pipelines share the same upstream concept of a **run**:

- Everything is run-scoped under `runs/<run_id>/`
- Each agreement/exhibit has a stable `item_id`
- Documents are normalized into **canonical text** that is broken into **anchors** (`A0001`, `A0002`, …)
  - Anchors are the “provenance unit”: model outputs cite anchor IDs, and we can verify by jumping back to the cited anchor text.

If you only remember one thing:

- **Indexing (LLM)** decides *where* in the document the relevant content is (as anchor IDs).
- **Excerpt packs (deterministic)** assemble the exact anchor text we send to the extractor(s).
- **Extraction (LLM)** outputs a strict JSON IR.
- **Validation (deterministic)** rejects schema/logic errors loudly.

---

## Companion “simple prompt” runs (optional recall)

In addition to the validated IR pipelines (CovenantIR / ContractIR), we also keep a pair of
*simple, non-validated* prompts around for recall/triage and side-by-side review.

These are designed to work with the same **indexing_v2 anchor buckets** and the same
**full-anchor excerpt pack** construction (no window truncation).

Prompts:
- Financial covenants (JSON list): `prompts/prompt_v1_short.txt`
- Performance pricing (JSON object): `prompts/prompt_pricing_second_pass_dg_nano_v4.txt`

Runner:
- `scripts/run_prompt_over_excerpt_packs.py`

Example: run the covenant prompt on the covenant anchor bucket for one agreement:

```bash
python scripts/run_prompt_over_excerpt_packs.py \
  --run-id <run_id> \
  --prompt prompts/prompt_v1_short.txt \
  --bucket financial_covenant \
  --item-id <item_id> \
  --out-dir scratch/simple_prompts/<item_id>/covenants \
  --overwrite
```

Example: run the pricing prompt on the union of pricing/base_rate/spread/fee anchors:

```bash
python scripts/run_prompt_over_excerpt_packs.py \
  --run-id <run_id> \
  --prompt prompts/prompt_pricing_second_pass_dg_nano_v4.txt \
  --bucket pricing_union \
  --item-id <item_id> \
  --out-dir scratch/simple_prompts/<item_id>/pricing \
  --overwrite
```

The output folder is artifact-first (prompt used, excerpt pack, rendered prompt, and raw model output).

---

## Third-pass refiners on top of the “simple prompts”

These are the two PI-driven workflows that sit *on top of* the less-structured prompts:

1) **Pricing metric definitions** (third pass)
2) **Covenant non-numeric limit formulas** (third pass)

They are intentionally narrow and artifact-first:
- They do **not** try to produce full ContractIR/CovenantIR.
- They do produce auditable JSON outputs with strict **anchor provenance** guardrails.

### Pricing definitions (third pass)

Goal:
- Given the metrics found in the performance pricing output, extract formal **definition language** from the agreement.

Inputs:
- A run with `normalized/` + `indexing_v2/` (for canonical text + anchor spans).
- A directory of pricing “second pass” outputs (per item), typically from the simple prompt run.

Runner:
- `scripts/pricing_definitions_third_pass_runner.py`

Recommended mode (Option B: targeted full-document indexing, always-on)

This is a two-step flow:
1) Run a dedicated *definitions-indexing* pass (LLM reads the full document and selects definition anchors **per metric**).
2) Run the definitions extractor over the resulting anchor pack.

Step 1: targeted definitions-indexing (writes to `runs/<run_id>/indexing_pricing_definitions_v1/`)

```bash
python scripts/pricing_definitions_indexing_v1_runner.py \
  --run-id <run_id> \
  --pricing-second-pass-dir <pricing_second_pass_dir> \
  --prompt prompts/indexing_pricing_definitions_v1.txt \
  --all \
  --overwrite
```

Step 2: definitions extraction using the targeted indexing selection

```bash
python scripts/pricing_definitions_third_pass_runner.py \
  --run-id <run_id> \
  --pricing-second-pass-dir <pricing_second_pass_dir> \
  --prompt prompts/prompt_pricing_third_pass_dg_v2.txt \
  --strategy pricing_definitions_indexing_v1 \
  --all \
  --out-dir scratch/pricing_definitions_third_pass/<run_id> \
  --overwrite
```

Notes:
- This avoids heuristic “term hit” retrieval by using an LLM as the full-document reader that selects definition anchors.
- The extractor still enforces anchor provenance: `source_refs` must appear in the provided excerpt pack.

Fallback mode (auto strategy selection; older):

```bash
python scripts/pricing_definitions_third_pass_runner.py \
  --run-id <run_id> \
  --pricing-second-pass-dir <pricing_second_pass_dir> \
  --prompt prompts/prompt_pricing_third_pass_dg_v2.txt \
  --auto \
  --all \
  --out-dir scratch/pricing_definitions_third_pass/<run_id> \
  --overwrite
```

What “auto” does:
- If indexing found a `definitions_anchor_range`, it tries:
  1) filtered definitions span
  2) definitions span + pricing_union anchors
- If there is no definitions span, it tries:
  1) definitions span + pricing_union (will be mostly pricing_union)
  2) term-hits anywhere in the doc + pricing_union

Key outputs:
- `.../best/<item_id>/llm_output.txt` — JSON object with `metrics[*].definition_verbatim` + `source_refs`
- `.../best/<item_id>/definition_chunks.txt` — exact anchor blocks shown to the model
- `.../report.json` — per-item strategy outcomes and validation counts

Validation guardrails:
- `source_refs` must be a list of anchors that appear in `definition_chunks.txt` (no hallucinated anchors).

### Covenant limit formulas (third pass)

Goal:
- Keep covenant extraction simple, but still recover computable thresholds when the covenant limit is **not a single number**
  (e.g., “the sum of $X plus 90% of Net Income…”).

Inputs:
- Covenant “second pass” outputs directory (per item), produced by the simple covenant prompt run.
  - Must include each item’s `llm_output.txt` (JSON list of covenant rows)
  - Must include the exact excerpt pack used (`excerpt_pack.txt` or `contexts.txt`)

Runner:
- `scripts/covenant_limit_formula_third_pass_runner.py`

Run over all items in a covenant second-pass output directory:

```bash
python scripts/covenant_limit_formula_third_pass_runner.py \
  --covenant-second-pass-dir <covenant_second_pass_dir> \
  --prompt prompts/covenant_limit_formula_third_pass_v1.txt \
  --all \
  --out-dir scratch/covenant_limit_formula_third_pass/<run_id> \
  --overwrite
```

Selection rule (deterministic):
- Only rows with `limit == null` and a non-empty `limit_text` are sent to the model.
- Rows that look like **definitions** (e.g., `"TERM" means ...`) are skipped (guardrail against false positives).
- If there are no such rows for an item, the script **skips the LLM call** and writes an empty schema-valid output.

Output schema:
- Produces a lightweight AST (`lit` / `var` / `op`) so the limit becomes computable without requiring full CovenantIR.

Validation guardrails:
- No unknown anchors in `source_refs` (must cite only anchors present in the excerpt pack).
- Every variable used in the AST must appear in `limit_expr.args`.

---

## Setup (once)

```bash
poetry install
```

Start the local gateway (required for any LLM calls):

```bash
cd agent-gateway
make serve
```

By default, the pipeline uses `GATEWAY_URL` or falls back to `http://127.0.0.1:8000` (see `src/pipeline/indexing.py`).

---

## Common upstream stages (shared by both pipelines)

### 1) Ingest (tarball → extracted HTML)

```bash
poetry run pipeline ingest \
  --run-id <run_id> \
  --tarball <path/to/*.nc.tar.gz> \
  --filters <filters/*.json>
```

Writes:
- `runs/<run_id>/ingest/`
- `runs/<run_id>/manifest.json`

### 2) Normalize (HTML → canonical text + anchors)

```bash
poetry run pipeline normalize --run-id <run_id>
```

Key artifacts per agreement:
- `runs/<run_id>/normalized/<item_id>/prompt_view.txt` (canonical text with `[[A####]]` markers)
- `runs/<run_id>/normalized/<item_id>/anchors.tsv` (anchor spans + ordering)

### 3) Indexing v2 (LLM global reader → anchor buckets)

This produces a selection artifact: “these anchors look like pricing”, “these anchors look like financial covenants”, etc.

For a general index with covenant anchors:

```bash
poetry run pipeline index-v2 \
  --run-id <run_id> \
  --prompt prompts/indexing_v2.txt \
  --gateway-url http://127.0.0.1:8000
```

Writes:
- `runs/<run_id>/indexing_v2/<item_id>_anchors.json`

### 4) Retrieval v2 (index selection → snippet packs)

This produces a JSONL “snippet pack” from the indexing selection. Each record includes:
- `anchor_id`
- `categories` (e.g., `financial_covenant`, `base_rate`, `spread`)
- a **window snippet** around the anchor span (useful for many LLM tasks)

```bash
poetry run pipeline retrieve-v2 --run-id <run_id>
```

Writes:
- `runs/<run_id>/retrieval_v2/<item_id>_snippets.jsonl`

Important nuance:
- CovenantIR and ContractIR v0.2 do **not** rely on the window snippets for core extraction.
- They build **excerpt packs** from the canonical anchor spans so the extracted context is structurally complete (no truncated tables).

---

## Financial covenant pipeline (CovenantIR v0.1)

### What it produces

For each agreement, it produces a `covenantir_validated.json` with:
- `covenants[*].test`: a computable boolean expression (AST) for compliance
- optional `covenants[*].applies_when`: a computable boolean “springing” trigger
- optional `tables[]` + `lookup_*` ops for schedules (strict evaluation)
- strict `source_refs` pointing back to anchors

Core logic lives in:
- Prompt: `prompts/covenant_ir_financial_v0_1.txt`
- Schema: `schemas/covenant_ir_v0_1.schema.json`
- Validator + evaluator: `src/pipeline/covenant_ir_v0_1.py`
- One-pass runner (LLM + repair + validation): `scripts/covenant_ir_v0_1_one_pass_harness.py`
- Batch runner (runs many items, writes `summary.json`): `scripts/covenant_ir_v0_1_batch_runner.py`

### How to run (batch over a run)

Prereq: you must already have `runs/<run_id>/retrieval_v2/<item_id>_snippets.jsonl` from `pipeline retrieve-v2`.

```bash
poetry run python scripts/covenant_ir_v0_1_batch_runner.py \
  --run-id <run_id> \
  --out-dir runs/<run_id>/covenantir_v0_1 \
  --prompt prompts/covenant_ir_financial_v0_1.txt \
  --max-items 99999 \
  --attempts 3
```

Outputs:
- `runs/<run_id>/covenantir_v0_1/summary.json` (per-item statuses)
- `runs/<run_id>/covenantir_v0_1/<item_id>/...` (full per-item trace)

### How to run (single item, for debugging)

```bash
poetry run python scripts/covenant_ir_v0_1_one_pass_harness.py \
  --run-id <run_id> \
  --item-id <item_id> \
  --out-dir scratch/covenantir_debug/<item_id> \
  --prompt prompts/covenant_ir_financial_v0_1.txt \
  --attempts 3
```

### How to debug failures

Look inside the per-item output folder. Key files:

- `contexts.txt` — the exact excerpt pack text fed to the model (full anchor text, not windows)
- `prompt.txt` — the full system prompt including TASK + CONTEXT
- `raw_attempt_N.txt` — raw model output (JSON-as-text)
- `validation_errors_attempt_N.json` — why attempt N failed validation
- `repair_appendix_attempt_N.txt` — what we appended to the prompt for the next attempt
- `covenantir_validated.json` — final validated artifact (or a “salvage blocked” artifact if the model never validated)
- `result.json` — one-line status used by batch summaries

---

## Pricing pipeline (ContractIR v0.2 “pricing kernel”)

This is the structured, multi-pass pricing approach (benchmark/base rate → spread/margin → fee rates), with strict validation and a deterministic merge.

Core logic lives in:
- Flow + LLM repair loops + excerpt pack building: `src/pipeline/contract_ir_v0_2_flow.py`
- CLI wrapper: `src/pipeline/contract_ir_v0_2_cli.py` (exposed as `pipeline contractir-v0-2 ...`)
- Validator + evaluator: `src/pipeline/contract_ir_v0_2.py`
- Merge: `src/pipeline/contract_ir_v0_2_merge.py`
- Excerpt-pack utilities (anchor ordering + gap-fill): `src/pipeline/excerpt_packs.py`

Prompts:
- Indexing prompt (pricing-kernel optimized): `prompts/indexing_v2_pricing_kernel_agentic_v2.txt`
- Base-rate extraction: `prompts/contract_ir_base_rate_v2.txt`
- Spread/margin extraction (tries strategies in order; first valid wins):
  - `prompts/contract_ir_spread_v2_lookup2.txt`
  - `prompts/contract_ir_spread_v2_lookup_range.txt`
  - `prompts/contract_ir_spread_v2_lookup_rule.txt`
  - `prompts/contract_ir_spread_v2_specialized.txt`
- Fee extraction (tries strategies in order; first valid wins):
  - `prompts/contract_ir_fee_rate_v2_lookup2.txt`
  - `prompts/contract_ir_fee_rate_v2_lookup_range.txt`
  - `prompts/contract_ir_fee_rate_v2_specialized.txt`

### How to run (end-to-end)

This command runs:
- indexing_v2
- retrieval_v2
- base-rate pass
- spread pass (strategy loop)
- fee pass (strategy loop)
- deterministic merge

If you already have a “source run” with normalized inputs:

```bash
poetry run pipeline contractir-v0-2 flow \
  --run-id <new_run_id> \
  --source-run-id <existing_run_id> \
  --gateway-url http://127.0.0.1:8000
```

Or start from tarballs:

```bash
poetry run pipeline contractir-v0-2 flow \
  --run-id <run_id> \
  --tarball <path/to/*.nc.tar.gz> \
  --filters <filters/*.json> \
  --gateway-url http://127.0.0.1:8000
```

To restrict to a few agreements while iterating:

```bash
poetry run pipeline contractir-v0-2 flow \
  --run-id <run_id> \
  --source-run-id <existing_run_id> \
  --item-id <item_id_1> \
  --item-id <item_id_2> \
  --gateway-url http://127.0.0.1:8000
```

### Output layout

Writes under:
- `runs/<run_id>/contractir_v0_2/summary.json`
- `runs/<run_id>/contractir_v0_2/items/<item_id>/...`

Per item:
- `base_rate/` (single prompt)
  - `anchor_expansion.json` (seed anchors + deterministic expansion parameters)
  - `contexts.txt`, `prompt.txt`, `raw_attempt_N.txt`, `parsed_attempt_N.json`, `validation_errors_attempt_N.json`
  - `contractir_validated.json` (base-rate-only ContractIR doc)
- `spread/` (strategy prompts)
  - `anchor_expansion.json`
  - `prompt_1/`, `prompt_2/`, … (trace per strategy)
  - `contractir_validated.json` (the selected spread result)
- `fee/` (strategy prompts)
  - same structure as `spread/`
- `contractir_merged.json` (deterministic merge of base + spread + fee)

### Offline validate / eval tools (no LLM calls)

Validate:

```bash
poetry run pipeline contractir-v0-2 validate --path runs/<run_id>/contractir_v0_2/items/<item_id>/contractir_merged.json
```

Evaluate a derived function:

```bash
poetry run pipeline contractir-v0-2 eval \
  --path runs/<run_id>/contractir_v0_2/items/<item_id>/contractir_merged.json \
  --fn-id <fn_id> \
  --args-json '{"as_of_date":"2020-06-30", "some_input":"1.23"}'
```
