# Design Notes (tracked)

This folder contains **tracked, versioned** design docs for the canonical methodology on this branch.

Key repo fact:
- `docs/` is intentionally **git-ignored** (see `.gitignore`) because it often contains local artifacts (PDF/TeX logs, sample outputs, etc.).
- Anything in `design/` should be safe to version-control: **Markdown only**, no generated artifacts.

If you only read one doc first:
- `../METHODOLOGY.md` — the single canonical methodology contract (run-scoped + anchors + deterministic packaging + strict artifacts).

## What lives here

Canonical methodology docs:
- `contractir_v0_2_pricing_kernel.md` — imposed-structure pricing kernel (computable).
- `covenantir_v0_1.md` — financial covenants as computable tests.
- `contract-pricing-compiler.md` — pricing regime compiler (coverage-first, indexing-free).
- `dg_v2_pricing_workflow.md` — DG v2 discovery/triage workflow (upstream for definition work).
- `definition_compiler_v2_ast_v1.md` — AST-first definition compiler (metrics/terms).
- `normalize-tables.md` — normalization invariants around table preservation.

Notes / future directions:
- `notes/` — conceptual sketches that are not necessarily implemented.

Legacy references:
- `legacy/` — docs for methodologies intentionally *not* present on the canonical branch.

## Legacy snapshots

This repo keeps legacy methodologies as branches so the canonical branch stays conceptually clean:
- `legacy/v1-run-scoped`
- `legacy/pricing-as-code-v1`

