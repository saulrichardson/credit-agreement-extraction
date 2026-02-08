# Repo Autopsy (What Exists Today)

This is a **code-grounded** snapshot of how the repo is organized right now (as of the current `main`/`canonical/one-methodology` tip).

If you want the **single conceptual contract**, read `METHODOLOGY.md`.
If you want the **functional stage graph + artifact IO**, read `design/PIPELINE_MAP.md`.

---

## The organizing unit is the run directory

Almost everything in this repo is built around a single reproducibility boundary:

`runs/<run_id>/...`

The manifest (`runs/<run_id>/manifest.json`) is the run-scoped “what happened” record written/updated by stages.

Code: `src/pipeline/core/config.py` (`Paths`, `update_manifest`) + `src/pipeline/evidence/ingest.py` (manifest creation).

---

## “Same preprocessing, various LLM Q/A parts” — yes, by design

The shared upstream is:

1) **Ingest**: tarballs + filters → `runs/<run_id>/ingest/` + `manifest.json`  
   Code: `src/pipeline/evidence/ingest.py`, CLI: `src/pipeline/cli/ingest.py`

2) **Normalize**: HTML → canonical text + anchors (`A0001…`)  
   Writes `runs/<run_id>/normalized/<item_id>/{canonical.txt,anchors.tsv,canonical_annotated.txt,prompt_view.txt}`  
   Code: `src/pipeline/evidence/normalize.py`, CLI: `src/pipeline/cli/ingest.py`

After that, the repo intentionally supports *multiple* LLM-backed transforms that play **different roles**:

| Role | What it does | Entry points | Artifacts |
|---|---|---|---|
| Evidence selection (global read) | LLM reads the full annotated doc and returns anchor buckets | `pipeline index-v2`, `src/pipeline/evidence/indexing_v2.py` | `runs/<run_id>/indexing_v2/<item_id>_anchors.json` |
| Navigation/context (optional) | LLM builds a TOC-like map used to tag snippets with “where in doc” context | `pipeline toc-v1`, `src/pipeline/evidence/toc_v1.py` | `runs/<run_id>/toc_v1/<subdir>/<item_id>.json` |
| Evidence packaging (deterministic) | Join anchor IDs + anchor spans → render snippet packs | `pipeline retrieve-v2`, `src/pipeline/evidence/retrieval_v2.py` | `runs/<run_id>/retrieval_v2/<item_id>_snippets.jsonl` |
| Extraction (prompt-driven) | LLM compiles snippet packs into “discovery JSON” (DG-ish output) | `pipeline structured-v2`, `src/pipeline/extract/structured_v2.py` | `runs/<run_id>/llm_qa/<subdir>/<item_id>.*` |
| Grounding pass | LLM re-asks over narrow context to pin down definition language | `pipeline definitions-v2`, `src/pipeline/extract/definitions_v2.py` | `runs/<run_id>/definitions_v2/<subdir>/*__definition.json` |
| Agreement-level metadata | LLM extracts parties/roles + headline facility terms | `pipeline agreement-metadata`, `src/pipeline/extract/agreement_metadata_v1.py` | `runs/<run_id>/agreement_metadata/<subdir>/<item_id>.json` |
| Facility fundamentals | LLM extracts facility separation + key dates | `pipeline facility-fundamentals`, `src/pipeline/extract/facility_fundamentals_v1.py` | `runs/<run_id>/facility_fundamentals/<subdir>/<item_id>.json` |

So: the “multiple Q/A parts” are not accidental duplication — they are separate transforms in the overall DAG.

The canonical end-to-end “v2” DAG is implemented by `pipeline all-v2`:
- Code: `src/pipeline/cli/all_v2.py`

---

## The repo has multiple “artifact families” (not multiple methodologies)

These are different outputs built under the same run-scoped / anchor-grounded methodology:

### A) DG-style discovery JSON (“structured-v2”)
- Purpose: discovery/triage and upstream term targeting (not computable semantics).
- Code: `src/pipeline/extract/structured_v2.py`

### B) Definitions grounding (“definitions-v2”)
- Purpose: recover verbatim definition text + provenance for the terms found in the discovery JSON.
- Code: `src/pipeline/extract/definitions_v2.py`

### C) Definition Graph compiler (AST-first)
- Purpose: compile definition language into an AST and resolve dependencies.
- Code: `src/pipeline/compile/definitions_compiler_v1.py`, `src/pipeline/compile/blocking_terms_compiler_v1.py`, `src/pipeline/compile/compustat_overlay_v1.py`

### D) ContractIR v0.2 (pricing kernel; computable semantics)
- Purpose: imposed-structure computable pricing logic (base rate / spread / fees) with strict evaluation semantics.
- CLI: `src/pipeline/ir/contract_ir_v0_2_cli.py`
- Core: `src/pipeline/ir/contract_ir_v0_2.py`

### E) Contract pricing compiler (pricing regimes/tables coverage)
- Purpose: table/regime coverage using `doc_ir` + per-table compilation (indexing-free by design).
- CLI: `src/pipeline/cli/contract_pricing.py`
- Core: `src/pipeline/pricing/contract_pricing.py`, `src/pipeline/pricing/doc_ir.py`

### F) CovenantIR v0.1 (financial covenants; computable semantics)
- Purpose: computable covenant tests with strict provenance + hard validation.
- Core validator/evaluator: `src/pipeline/ir/covenant_ir_v0_1.py`
- Current extraction harnesses live in `scripts/` (e.g., `scripts/covenant_ir_v0_1_one_pass_harness.py`).

---

## Where the “two approaches to the same problem” actually are

The main “dual approach” in the repo is **evidence targeting**:

1) **Index-v2 → retrieve-v2 → snippet-pack extraction**
   - Good for: “global reader picks anchors” + targeted downstream prompts
   - Used by: `structured-v2`, `agreement-metadata`, `facility-fundamentals`, ContractIR flows, CovenantIR harnesses

2) **DocIR/table compiler (indexing-free)**
   - Good for: pricing schedules/regimes that are primarily table-driven; exhaustive table coverage
   - Used by: `contract-pricing`, `contract-pricing-v3`

These can coexist under one methodology because both still produce:
- run-scoped artifacts
- anchor-grounded provenance (anchors still exist; they’re just selected differently)
- deterministic packaging (either retrieval packs or doc_ir table packs)

---

## Practical “old vs new” (what you’ll see on disk)

The “new” (canonical) layout is:
- `indexing_v2/`, `retrieval_v2/`, `llm_qa/`

You may also encounter older runs containing:
- `indexing/`, `retrieval/`, `structured/`

Legacy snapshots are explicitly preserved as branches:
- `legacy/v1-run-scoped`
- `legacy/pricing-as-code-v1`

---

## Current organization pain points (worth fixing)

These are grounded in current code, not preferences:

1) **Some research/extraction flows still live only in `scripts/`**, not as first-class pipeline stages (notably CovenantIR extraction harnesses).  
   The methodology is consistent, but the *organizational surface* is split.

2) **Legacy artifacts exist on disk** (older runs with `structured/<item_id>.txt` or `llm_qa/<item_id>.txt`).  
   The canonical branch now prefers `.json` artifacts, but tooling should remain explicit about legacy compatibility.
