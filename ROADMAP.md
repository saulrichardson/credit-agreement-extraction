# Repo Autopsy + Modernization Roadmap (2026-02-04)

This file records an artifact-first snapshot of the repo state and a concrete plan to reorganize around the **latest, strict, validated** extraction approaches (with performance + design guardrails).

Branch created for this work: `modernize/reorg-2026-02-04`.

## Snapshot (What’s Here)

Top-level:
- `src/pipeline/` — the main Python package (`pipeline` console script).
- `scripts/` — many one-off runners/audits/renderers (currently not part of the installed package).
- `prompts/` — prompt templates (indexing, structured extraction, compilers).
- `schemas/` — JSON Schemas (ContractIR v0.1/v0.2, CovenantIR v0.1).
- `agent-gateway/` — submodule providing a local `/v1/responses` gateway + a small Python client.
- `runs/`, `scratch/`, `logs/` — runtime artifacts (ignored by git).

Important repo fact right now:
- The working tree contains **substantial uncommitted work** (new pipeline modules, scripts, prompts, and tests) that represent the “new direction”, but is not yet captured in git history.

## “Old” vs “New” (Practical Definition)

**Old (baseline / legacy workflow):**
- The v1 pipeline shape: `ingest → normalize → index → retrieve → structured`.
- “Structured” stages persist raw model output (`.txt`) without strict JSON parsing/validation.
- `pipelines/` click entrypoints duplicate the main CLI and import internal helpers.

**New (directionally correct / latest approach):**
- v2 indexing/retrieval (`index-v2`, `retrieve-v2`) that produces richer anchor buckets and optionally tags snippets with TOC metadata.
- Strict IRs + validators:
  - `ContractIR v0.2` (pricing-kernel) with deterministic evaluation + hard validation gates.
  - `CovenantIR v0.1` (financial covenants) reusing the same AST semantics.
- “Compiler-style” stages that enforce **JSON-only + Pydantic validation + retry-then-fail** (e.g., contract pricing compiler).
- A real `pytest` suite that exercises core invariants (schema + provenance + evaluator rules).

## Target Direction (Design Principles)

1. **Strict artifacts as the product surface**
   - Every stage writes a *typed* artifact (`.json` / `.jsonl`) validated by Pydantic and/or JSON Schema.
   - Raw model text remains as a debug sidecar, never as the only output.

2. **Separation of concerns**
   - CLI wiring ≠ pipeline orchestration ≠ domain models/IR ≠ I/O/artifact layout ≠ gateway client.

3. **Fail fast and loud**
   - No silent skipping, no “best effort” fallbacks unless explicitly requested and recorded in output metadata.

4. **Performance as a first-class constraint**
   - Deterministic pre-filtering and table/anchor selection to reduce LLM payload.
   - Skip-existing + overwrite semantics are explicit and consistent across stages.
   - Avoid repeated schema loads / repeated heavy parsing where caching is safe.

## Proposed Re-org (Concrete)

Keep the public package name `pipeline` (avoid breaking everything), but restructure internals:

```
src/pipeline/
  cli/                # click groups + option parsing only
  stages/             # ingest/normalize/index/retrieve/structured/toc/...
  artifacts/          # Pydantic models for artifacts + path helpers
  ir/                 # ContractIR/CovenantIR + AST + evaluators
  llm/                # gateway/http client + retry/backoff + JSON-only helpers
  prompts/            # prompt loading + prompt registry (paths + hashes)
  datasets/           # small, versioned JSON inputs (filters/allowlists)
```

Move or retire:
- `pipelines/` → remove (or mark `legacy/`) once `pipeline` CLI fully covers flows.
- `scripts/` → keep, but label as `tools/` and make sure each script is:
  - explicit about inputs/outputs
  - compatible with `poetry run python ...`
  - not depended on by “product” CLI paths unless promoted into `src/pipeline/`

## Phased Plan (Suggested)

### Phase 0 — Baseline capture (1 short PR)
- Decide what subset of the current uncommitted “new” work is ready to be the baseline.
- Commit it on this branch so the modernization can be incremental and reviewable.

### Phase 1 — Make the “latest path” explicit
- Declare one blessed end-to-end flow (likely `all-v2` + ContractIR/CovenantIR flows).
- Mark the v1 pipeline path as legacy and document its intended fate (remove vs keep).

### Phase 2 — Strict outputs everywhere
- Upgrade any remaining raw-text-only stages to:
  - enforce JSON-only outputs
  - validate (Pydantic / JSON Schema)
  - persist structured artifacts + `.raw.txt` sidecars

### Phase 3 — Refactor structure (no behavioral changes)
- Split `src/pipeline/run.py` into `src/pipeline/cli/*.py`.
- Introduce `stages/` modules and keep them pure/functional where possible.
- Centralize the gateway client + retry policy (remove duplicated `_call_gateway` patterns).

### Phase 4 — Prompt + artifact registry
- Organize `prompts/` by pipeline + version.
- Add a small registry file (YAML/JSON) mapping:
  - CLI command → default prompt(s) → expected schema → output directories

### Phase 5 — Performance + ergonomics
- Add skip-existing/overwrite flags to long-running stages consistently.
- Cache JSON schema loads and compiled validators where safe.
- Add microbenchmarks for normalize + doc_ir + validators (optional).

## Decisions Needed (Pick One Per Item)

1. **Dependency tooling**
   - Keep Poetry (current), or migrate to `uv` for faster installs/locks?
2. **Gateway coupling**
   - Keep importing the submodule client (current), or replace with a small internal HTTP client to remove `sys.path` hacks?
3. **What to version-control**
   - Prompts: track in repo (recommended for reproducibility) vs keep local-only.
   - Docs: track Markdown but ignore PDFs/aux/logs (recommended).
4. **Python version floor**
   - Keep `>=3.10` vs raise to `>=3.11` (aligns with `agent-gateway`, cleaner typing, faster runtime).

