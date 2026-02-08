# CovenantIR v0.1 (financial covenants)

Goal: encode **financial covenants** from credit agreements as **computable compliance tests** so downstream evaluation is mechanical:

- each covenant has an optional `applies_when` (springing triggers) and a required `test`
- both are strict boolean expressions over typed inputs (dates, ratios, money, etc.)

Primary artifacts:

- Schema: `schemas/covenant_ir_v0_1.schema.json`
- Validator + evaluator: `src/pipeline/ir/covenant_ir_v0_1.py`
- Extraction prompt: `prompts/covenant_ir_financial_v0_1.txt`
- One-pass harness (LLM + repair + validate): `scripts/covenant_ir_v0_1_one_pass_harness.py`
- Batch runner (pressure-testing): `scripts/covenant_ir_v0_1_batch_runner.py`

## Strict lookup semantics (no priority)

Like ContractIR, CovenantIR uses the shared AST + table evaluation semantics from `src/pipeline/ir/contract_ir_v0_2.py`.

In particular, rule/schedule evaluation is strict:

- `lookup_rule(...)`: **exactly 1 row must match** (0 matches or >1 matches is a hard error)
- `lookup_range(...)`: must resolve to **exactly 1 row** for a given key (overlaps/gaps become runtime errors)

There is **no implicit precedence / priority** in `lookup_rule` evaluation.

Note: an older pricing-as-code workflow used **priority-ordered** rule evaluation, but that is separate from the ContractIR/CovenantIR direction and is intentionally not present on the canonical branch.
It is preserved as a snapshot on branch `legacy/pricing-as-code-v1` (see `design/legacy/pricing_schema_v1.md`).

## Precision-first extraction mode (Option 2)

The financial covenant extraction prompt is currently operated in a **precision-first** mode:

- Either encode **all** financial covenants in the excerpt pack as computable tests, or
- If any covenant can’t be encoded from the provided context, **fail the whole item** by emitting blocking `open_items` and returning:
  - `covenants = []`
  - `tables = []`
  - `derived = []`

The one-pass harness enforces this policy via `validate_precision_first_policy(...)` (see `src/pipeline/ir/covenant_ir_v0_1.py`).

## Future direction: Option 3 (table type inference)

If prompt-only reliability is insufficient for schedule/rule table encoding (especially `lookup_rule(...)` predicate typing),
see: `notes/covenantir_v0_1_option3_table_type_inference.md`.
