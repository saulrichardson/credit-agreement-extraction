# Target Structure (Canonical “One Methodology”)

This doc proposes a **clean, canonical structure** for the repo that matches the methodology contract in `METHODOLOGY.md`:

- run-scoped artifacts (`runs/<run_id>/...`)
- anchor-grounded provenance (`A0001…`)
- deterministic evidence packaging
- strict artifacts *when you choose to turn validation on*, without entangling prompt iteration with validator iteration

It is written to reduce “dueling approaches” by making the *roles* explicit instead of letting them appear as parallel ad-hoc flows.

---

## What “organized” means here (non-goals included)

Organized does **not** mean “one giant pipeline that does everything”.

Organized means:
- **One shared upstream contract** (ingest/normalize/anchors + run manifest)
- **Clear stage boundaries** with explicit IO under `runs/<run_id>/`
- **Multiple outputs** (artifact families) are fine, but they are produced under the same evidence/packaging contract
- **Prompts can iterate without touching validators**, and validators can harden without rewriting prompts

Non-goals (for now):
- Forcing every stage to be fully validated today (you explicitly want prompt iteration freedom)
- Deleting experimental harnesses (we can re-home them, but not pretend they don’t exist)

---

## Canonical “roles” (how to name/group code)

Instead of grouping files by “approach”, group by **stage role**:

1) `ingest/` — tarball ingestion + manifest
2) `normalize/` — canonical text + anchors
3) `evidence/`
   - selection (LLM global read): `index-v2`
   - packaging (deterministic): `retrieve-v2` / excerpt packs
   - optional navigation tags: `toc-v1`
4) `extract/` — prompt-driven compilation from packaged evidence
   - structured discovery JSON (`structured-v2`)
   - agreement metadata / facility fundamentals
   - definition grounding (`definitions-v2`)
5) `compile/` — downstream compilers that build stricter derived artifacts (ASTs, overlays)
6) `ir/` — imposed-structure IRs + validators/evaluators (ContractIR, CovenantIR)
7) `pricing/` — pricing regime/table compilers (`contract-pricing*`)
8) `llm/` — gateway + strict JSON retry policy (shared infrastructure)

This keeps “multiple pipelines” from looking like “multiple methodologies”.

---

## Concrete repo layout proposal (incremental, not a big-bang)

### Short-term (incremental, low-risk)

- Keep `src/pipeline/*.py` flat for now, but:
  - move **gateway plumbing** out of `src/pipeline/indexing.py` into `src/pipeline/llm/gateway.py`
  - standardize “JSON artifacts use `.json`” (no JSON-in-`.txt`)
  - keep *compatibility re-exports* in old locations so scripts don’t all break at once

### Mid-term (cleaner package boundaries)

Move to subpackages and leave thin shim modules for compatibility:

```
src/pipeline/
  core/                 # Paths, manifest helpers, run_id, filesystem contracts
  llm/                  # gateway client + strict_json + prompt hashing helpers
  evidence/             # index-v2, retrieve-v2, toc-v1, excerpt packs
  extract/              # structured-v2, definitions-v2, metadata, facility
  compile/              # definition graph + blocking terms + overlays
  pricing/              # contract_pricing* + planner + doc_ir helpers
  ir/                   # contract_ir_v0_2*, covenant_ir_v0_1*
  cli/                  # thin click glue only
```

---

## Prompt iteration vs validation iteration (how to keep them decoupled)

You asked for the ability to:
- improve prompts without touching validation yet
- later harden validation without rewriting prompts
- and sometimes do both

The cleanest separation is:

1) Extraction stages always write:
   - a parsed artifact (JSON)
   - raw output sidecar(s)

2) Validation is a separate *deterministic stage* that:
   - reads the parsed artifact(s)
   - emits a validation report (and optionally a “validated” marker artifact)

This avoids per-command feature flags and keeps the conceptual pipeline clean:
- prompts live in `prompts/`
- validators live in `src/pipeline/*validator*` modules
- you run validators when you care, not while you’re iterating prompts

---

## Tradeoffs (explicit)

1) **Big-bang reorg vs incremental**
   - Big-bang: faster “end state”, higher breakage risk.
   - Incremental: keeps the repo runnable at each step; costs more steps.

2) **Single CLI surface vs scripts**
   - Full CLI: discoverable, consistent artifacts, better for “canonical”.
   - Scripts: faster iteration for research; but they fragment organization unless clearly treated as harnesses/tools.

3) **Strict validation everywhere vs “validate later”**
   - Strict everywhere: safer outputs, but blocks prompt iteration early.
   - Validate later: enables prompt iteration, but you must consciously opt into hard gates before claiming “production”.

