# Methodology (Locked Recursive Pipeline)

This repository now follows one strict methodology.

## Objective

Extract pricing and covenant outputs from debt agreements using anchor-grounded evidence, then produce recursive definition and Compustat overlay artifacts for downstream analysis.

## Canonical flow

1. `ingest`
2. `normalize`
3. `index-v2`
4. `retrieve-v2`
5. `structured-v2` (pricing)
6. `structured-v2` (covenant)
7. `definitions-compiler-v1` (pricing + covenant)
8. `blocking-terms-compiler-v1` (pricing + covenant)
9. `compustat-overlay-v1` (pricing + covenant)
10. `agreement-metadata`
11. `analysis-export-v2`

The full orchestrator is:

- `pipeline all-v2-full`

## Grounding contract

- Every extracted claim must be attributable to anchor-linked evidence from `index-v2` / `retrieve-v2`.
- Recursive definition compilation must emit explicit term-level outputs.
- Overlay output is based on the compiled recursive definitions/terms and the Compustat allowlist.
- Export fails loudly when required stage artifacts are missing.

## Non-goals

- Backward compatibility for retired workflows.
- Legacy TOC, definitions-v2, contract-pricing, or ContractIR command paths.
