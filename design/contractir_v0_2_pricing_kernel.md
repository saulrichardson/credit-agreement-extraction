# ContractIR v0.2 — Pricing Kernel (Base / Spread / Fees)

This document describes the **new, imposed-structure pricing-kernel approach** for credit agreements.

Scope (intentionally narrow):
- **Base rate definitions** (ABR / Base Rate / Adjusted LIBOR / Term SOFR, etc.)
- **Pricing spreads / margins** (Applicable Margin tables, grids, tiers)
- **Fee rates** (facility fee, commitment fee schedules)

Non-goals (for now):
- Full cashflow modeling, accrual across periods, business-day adjustments, etc.

The key principle is: **do not ask the LLM for “all-in pricing” in one shot**. We extract components with strict structure and merge deterministically.

---

## What this is

### Imposed structure (ContractIR)
ContractIR is a strict JSON intermediate representation with:
- A small AST grammar (`lit` / `var` / `op`)
- A strict operator allowlist
- Explicit provenance requirements (`source_refs` for every derived fn and every table row)
- Explicit semantic roles so “rate-like” numbers are not conflated:
  - `base_rate` (benchmark/base rate definition)
  - `spread` (margin component only)
  - `fee_rate` (facility/commitment fees)

**Files**
- Schema: `schemas/contract_ir_v0_2.schema.json`
- Validator + deterministic evaluator: `src/pipeline/contract_ir_v0_2.py`

### Prompt strategy (subtasks)
Each pricing component is extracted with a **separate prompt** and a **separate excerpt pack**.

This avoids the major failure mode we saw in full-table / full-pricing prompts:
the model will “complete the grid” or “complete the formula” when context is incomplete.

**Files (prompts)**
- Base rate: `prompts/contract_ir_base_rate_v2.txt`
- Spread (2D grids via `lookup2`): `prompts/contract_ir_spread_v2_lookup2.txt`
- Spread (range tiers via `lookup_range`): `prompts/contract_ir_spread_v2_lookup_range.txt`
- Spread (2D conditional schedules via rule tables + `lookup_rule`): `prompts/contract_ir_spread_v2_lookup_rule.txt`
- Spread (single-row targeted extraction): `prompts/contract_ir_spread_v2_specialized.txt`
- Fee rate (simple schedules): `prompts/contract_ir_fee_rate_v2_specialized.txt`
- Fee rate (range tiers): `prompts/contract_ir_fee_rate_v2_lookup_range.txt`

### Deterministic merge (no model)
The final per-agreement artifact should be produced by merging the component outputs deterministically.

Hard rules:
- Fail loudly on duplicate `table_id` / `fn_id`
- Fail loudly on conflicting index series units
- Require `contract_id == item_id` so outputs are mergeable across passes

**Files**
- Merge utility: `src/pipeline/contract_ir_v0_2_merge.py`

---

## Where artifacts live

This repo is “run-scoped”. Most artifacts live under:
`runs/<run_id>/...`

Relevant upstream artifacts (used to build excerpt packs):
- Indexing v2 selections: `runs/<run_id>/indexing_v2/<item_id>_anchors.json`
- Snippet packs: `runs/<run_id>/retrieval_v2/<item_id>_snippets.jsonl`

ContractIR strategy harness outputs:
- `runs/<out_run_id>/<experiment_id>/`
  - `prompt.txt` (exact prompt sent)
  - `contexts.txt` (exact excerpt pack)
  - `raw_attempt_*.txt`, `parsed_attempt_*.json` (LLM raw + parsed outputs)
  - `validation_errors.json` (if any)
  - `contractir_validated.json` (only when valid)
  - `result.json` (per-experiment metrics)
- `runs/<out_run_id>/summary.json` (aggregate)

---

## Commands (formalized)

These commands are intentionally **separate** from the legacy DG pricing pipeline.
They live under the `pipeline contractir-v0-2 ...` namespace.

### Note on “priority” (legacy vs ContractIR)

This repo previously contained an older, separate **pricing-as-code** workflow with **priority-ordered** rule evaluation.

That methodology is intentionally **not present** on the canonical branch (to avoid two competing compute semantics).
It is preserved as a snapshot on:
- branch `legacy/pricing-as-code-v1` (see `design/legacy/pricing_schema_v1.md`)

That legacy workflow is **not used** by ContractIR v0.2 (or CovenantIR v0.1).
In the ContractIR / CovenantIR direction, rule tables are evaluated via `lookup_rule(...)` with **strict semantics**:
- exactly 1 row must match for a given input point
- 0 matches ⇒ hard error (`NoMatchingRow`)
- >1 matches ⇒ hard error (`MultipleMatchingRows`)

This is intentional: we want ambiguity/overlap in extracted grids to fail loudly rather than be “resolved” by implicit precedence.

All commands assume you are running inside the Poetry environment:

```bash
poetry run pipeline --help
poetry run pipeline contractir-v0-2 --help
```

### Prereqs: produce snippet packs for a run (LLM calls happen later)

The ContractIR prompts operate on **excerpt packs** built from `indexing_v2` + `retrieval_v2`.
If you already have a run like `dan-v2-20260106`, you can skip this.

```bash
# 1) Choose anchors (LLM call) using the v2 indexing schema
poetry run pipeline index-v2 --run-id <run_id> --prompt prompts/indexing_v2.txt

# 2) Materialize the anchor snippets into retrieval_v2/*.jsonl
poetry run pipeline retrieve-v2 --run-id <run_id> --bandwidth 400
```

---

## End-to-end flow (Option C)

There is a single command that runs:
- input preparation (either ingest+normalize, or clone from an existing run for cleanliness)
- `indexing_v2` using a pricing-kernel-optimized prompt
- `retrieval_v2`
- three structured LLM passes (base_rate, spread, fee)
- deterministic merge per agreement

### 1) Clone from an existing run (recommended for iteration)

This copies `runs/<source_run_id>/normalized` and the manifest into a new run, then performs indexing/retrieval and the ContractIR passes.

```bash
poetry run pipeline contractir-v0-2 flow \
  --run-id contractir-v0_2-flow-dev \
  --source-run-id dan-v2-20260106
```

Limit to one agreement while iterating:

```bash
poetry run pipeline contractir-v0-2 flow \
  --run-id contractir-v0_2-flow-dev \
  --source-run-id dan-v2-20260106 \
  --item-id 0000950134-96-000040_2
```

### 2) Start from tarballs (ingest + normalize)

```bash
poetry run pipeline contractir-v0-2 flow \
  --run-id contractir-v0_2-flow-dev \
  --tarball data/daily_filings/2023/QTR1/20230103.nc.tar.gz \
  --filters filters/test_sample_2_agreements.json
```

### Outputs

Artifacts are written under:
- `runs/<run_id>/indexing_v2/` (now includes pricing-kernel buckets)
- `runs/<run_id>/retrieval_v2/`
- `runs/<run_id>/contractir_v0_2/items/<item_id>/`
  - `base_rate/` (prompt/contexts/raw/validated)
  - `spread/` (prompt strategies attempted under `prompt_1/`, `prompt_2/`, …)
  - `fee/` (prompt strategies attempted under `prompt_1/`, `prompt_2/`, …)
  - `contractir_merged.json` (final merged ContractIR for the item)
- `runs/<run_id>/contractir_v0_2/summary.json`

---

## Indexing prompt for the pricing kernel

The end-to-end flow uses a pricing-kernel-optimized indexing prompt by default:
- `prompts/indexing_v2_pricing_kernel.txt`

This prompt extends indexing_v2 with additional evidence buckets that directly drive the three ContractIR passes:
- `base_rate_anchors`
- `spread_anchors`
- `fee_anchors`

`pricing_anchors` remains available as a union bucket for “all borrower economics” when needed.

### 1) Run the v0.2 strategy harness (LLM calls)

This is a repeatable testbed that runs controlled experiments + stress tests.

```bash
poetry run pipeline contractir-v0-2 strategy-harness \
  --source-run-id dan-v2-20260106 \
  --out-run-id contractir-v0_2-strategy-tests-dev
```

Run a single experiment (substring match on exp_id):

```bash
poetry run pipeline contractir-v0-2 strategy-harness \
  --source-run-id dan-v2-20260106 \
  --out-run-id contractir-v0_2-strategy-tests-dev \
  --exp-filter S5_RatioSchedule
```

### 2) Validate a ContractIR v0.2 JSON file (offline)

```bash
poetry run pipeline contractir-v0-2 validate --path runs/.../contractir_validated.json
```

### 3) Evaluate a derived function (offline)

```bash
poetry run pipeline contractir-v0-2 eval \
  --path runs/.../contractir_validated.json \
  --fn-id ApplicableMargin \
  --args-json '{"loan_type_code":"CD","status_code":"IV"}'
```

With indices:

```bash
poetry run pipeline contractir-v0-2 eval \
  --path runs/.../contractir_validated.json \
  --fn-id ABR \
  --args-json '{"date":"2023-01-03"}' \
  --indices-json '{"PrimeRate":{"2023-01-03":"0.0500"},"NYFRBRate":{"2023-01-03":"0.0520"}}'
```

### 4) Merge component artifacts (offline)

```bash
poetry run pipeline contractir-v0-2 merge \
  --input runs/.../base_rate.json \
  --input runs/.../spread.json \
  --input runs/.../fee_rate.json \
  --out merged_contract_ir.json
```

---

## Notes on hard requirements that matter

These are deliberate “fail fast” constraints that improved reliability:
- `contract_id` MUST equal `item_id` (enforced by harness + prompts)
- `source_refs` MUST reference anchors actually present in the excerpt pack
- For rule tables, `predicate` MUST be a bool AST node (not a string)
- When context is incomplete, output MUST use blocking `open_items` rather than invent rows/values
