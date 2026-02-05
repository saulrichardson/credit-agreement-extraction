# Methodology (First-Stab Standardization)

This repo is best understood as a **run-scoped evidence pipeline** with a single consistent methodology:

1) normalize to anchored canonical text  
2) select evidence deterministically and/or via an LLM global reader  
3) package evidence deterministically (so model inputs are auditable)  
4) compile into strict artifacts (JSON-only + validation + fail-loud)

There are multiple *pipelines* in the codebase, but they should not represent multiple methodologies. They should be different **artifact families** produced under the same evidence/validation contract.

If you want a “what runs when / what files exist” view (not conceptual), see:
- `design/PIPELINE_MAP.md`

## Core concepts (shared across everything)

### 1) `run_id` (reproducibility boundary)

Every execution writes artifacts under:

`runs/<run_id>/...`

The run manifest is the single canonical “what happened” record:

`runs/<run_id>/manifest.json` (written by `src/pipeline/ingest.py` and updated by later stages).

### 2) `item_id` (unit of work)

Most stages operate per exhibit / agreement instance (an EDGAR document) using a stable `item_id` (typically `<accession>_<sequence>`).

### 3) `anchor_id` (provenance unit)

Normalized documents are split into anchors (`A0001`, `A0002`, …). An “answer” is only considered grounded if it cites anchor IDs, and those anchor IDs can be resolved back to canonical text.

Normalized artifacts live under:

- `runs/<run_id>/normalized/<item_id>/canonical.txt`
- `runs/<run_id>/normalized/<item_id>/anchors.tsv`
- `runs/<run_id>/normalized/<item_id>/canonical_annotated.txt`

Normalization code: `src/pipeline/normalize.py`.

### 4) Evidence selection vs evidence packaging

The repo intentionally separates:

- **Selection**: decide *which* anchors matter (typically an LLM “global reader”).
- **Packaging**: deterministically assemble the exact text the next step sees (snippets or excerpt packs).

This makes it possible to audit model behavior and to make downstream steps smaller/faster.

### 5) Artifact-first contract (default policy)

When an approach is “production-grade” in this repo, it should follow:

- JSON-only output (no markdown/prose in the primary artifact)
- strict validation (Pydantic and/or JSON Schema)
- retries use explicit validation error feedback
- final failure is loud (no silent partial outputs)
- raw model output is saved as a debug sidecar, not the product surface

## Conventions (so stages compose cleanly)

These are the conventions the canonical v2 stages follow today:

- **Deterministic vs LLM-backed is explicit**
  - Deterministic stages write reproducible artifacts from upstream inputs (no gateway calls).
  - LLM-backed stages call the gateway and produce strict-ish artifacts plus debug sidecars.

- **Prompts are versioned by path + hash**
  - Prompt templates live in `prompts/`.
  - Stages record both the prompt path and its SHA-256 in `runs/<run_id>/manifest.json`.

- **Output subdirectories are part of the experiment boundary**
  - When a stage supports `--output-subdir`, it selects a subfolder under its stage directory.
  - The resolved subdir is recorded in the manifest so downstream joins can reference it (see `analysis-export`).


## Artifact families (pipelines)

These are the major artifact families the repo can produce. The intent is:
- keep the *methodology* consistent across them
- allow multiple outputs when they serve different downstream goals (compute vs coverage vs discovery)

### A) DG v2 pricing JSON (discovery / upstream for definitions)

Goal:
- Produce a strict-ish pricing JSON that is good for **discovery + follow-on definition work**.

Primary code entrypoints:
- CLI: `src/pipeline/run.py` (`structured-v2`, `definitions-v2`, `all-v2`)
- Structured runner: `src/pipeline/structured_v2.py`
- Definitions pass: `src/pipeline/definitions_v2.py`

### B) Definition Graph (AST-first definitions + dependency terms + optional Compustat overlay)

Goal:
- Extract and validate **verbatim definition language** + a best-effort **expression AST** for metrics/terms.
- Resolve dependency terms recursively.
- Optionally map to Compustat variables later as a separate overlay.

Primary code entrypoints:
- Definitions compiler: `src/pipeline/definitions_compiler_v1.py`
- Blocking terms: `src/pipeline/blocking_terms_compiler_v1.py`
- Overlay: `src/pipeline/compustat_overlay_v1.py`

### C) ContractIR v0.2 (imposed-structure pricing kernel)

Goal:
- Encode **computable** base-rate / spread / fee logic as a strict IR with deterministic evaluation semantics.

Key semantic choice:
- Strict schedule semantics (0 matches or >1 matches is a hard error), no implicit precedence/priority.

Primary code entrypoints:
- CLI namespace: `src/pipeline/contract_ir_v0_2_cli.py`
- Validator + evaluator: `src/pipeline/contract_ir_v0_2.py`
- Merge: `src/pipeline/contract_ir_v0_2_merge.py`

### D) Contract pricing compiler (table/regime coverage)

Goal:
- Enumerate pricing tables/regimes/adjustments with strict evidence grounding.
- Optimized for *coverage* of schedules/regimes rather than “compute a single rate”.

Primary code entrypoints:
- Doc IR: `src/pipeline/doc_ir.py`
- Contract pricing: `src/pipeline/contract_pricing.py`
- Schema/models: `src/pipeline/contract_schemas.py`

### E) CovenantIR v0.1 (financial covenants)

Goal:
- Encode financial covenants as **computable compliance tests** (boolean expressions) with strict provenance.

Primary code entrypoint:
- Validator + evaluator: `src/pipeline/covenant_ir_v0_1.py`

### F) Pricing-as-code v1 (legacy)

Goal:
- A computation-oriented pricing model with **priority-ordered rules**.

Status:
- Treat as legacy / research unless explicitly elevated.
- Snapshot preserved on branch: `legacy/pricing-as-code-v1`

Notes:
- This approach is intentionally not present on the canonical branch to avoid two competing compute semantics.

## “Latest” methodology choice (proposed default)

This is the default stance for organization (can be revised):

- Canonical *computable* pricing output: **ContractIR v0.2**
- Canonical pricing *coverage/regimes* output: **Contract pricing compiler**
- Canonical definition semantics: **Definition Graph (AST-first)**
- DG v2 remains useful as a **discovery + triage** tool, and as an upstream for the definition graph, but is not the canonical compute artifact.
