# CovenantIR v0.1 — Option 3 (future): table type inference / compilation

This document captures a **future direction** we may want to implement if prompt-only reliability (Option 1) is insufficient.

## Motivation (what breaks today)

In practice, models sometimes produce **semantically correct** table content (especially for rule tables via `lookup_rule(...)`) but make a **bookkeeping mistake** in `tables[].columns[]`:

- Predicate **cells** are boolean AST expressions (e.g., `gte(as_of_date, "2008-06-30")`)
- But the declared predicate column type is `"string"` instead of `"bool"`
- The validator correctly rejects this with `lookup_rule_predicate_not_bool`

This is a “type declaration drift” problem: the model got the hard part right (the logic), but failed on a metadata field.

## Option 3 idea

Stop requiring the extractor to be the source of truth for **table column types**.

Instead:

1. Treat extracted tables as *raw* (shape-correct) data structures.
2. Run a deterministic **table compiler** that infers `tables[].columns[].type` from:
   - Cell AST node types (e.g., `lit.type == "date"` → column is `"date"`)
   - AST operator return types (e.g., `gte(...)` returns bool → column is `"bool"`)
   - How the table is referenced (e.g., `lookup_rule(table_id, predicate_col, value_col)` implies `predicate_col` must be `"bool"`)
3. Validate the compiled representation (strictly) before evaluation.

The net effect is that we stop asking the LLM to maintain redundant “schema glue” it frequently gets wrong.

## Implementation sketch

### 1) Two-phase validation

Split validation into:

- **Structural validation** (shape + required keys)
  - Does the JSON match the schema shape?
  - Are AST nodes well-formed?
  - Do rows contain `cells` mappings, etc.

- **Compilation validation**
  - Infer and/or check types across each table column.
  - Reject mixed-type columns (unless explicitly allowed).
  - Enforce operator-specific requirements (e.g., `lookup_rule` predicate col must be bool).

### 2) Where the compiler could live

- `src/pipeline/covenant_ir_v0_1.py` (as a CovenantIR-specific compiler), or
- shared code near the ContractIR AST/table evaluator (`src/pipeline/contract_ir_v0_2.py`) if we want a single table-typing system.

### 3) Schema change options

We’d need to decide whether:

- **A. Keep the existing schema** and allow `type` to be `"unknown"` or optional at extraction time, filled in by the compiler, or
- **B. Introduce a new schema version** (`covenant_ir_v0_2`) where column types are derived artifacts (not required from the model).

Both are viable; (B) is cleaner but breaking.

## Trade-offs

Pros:

- Removes an entire class of “LLM bookkeeping” failures without changing strict semantics.
- Makes the system more robust to minor prompt/model regressions.
- Compiler errors can be made extremely actionable (e.g., “column predicate used in lookup_rule must be bool; inferred bool from cells but declared string”).

Cons:

- Adds implementation complexity (type inference, unification, error reporting).
- Ambiguity must be handled explicitly (e.g., a column with both `lit("decimal", ...)` and `lit("money", ...)`).
- If inference is wrong, it becomes harder to attribute blame to the extractor vs. compiler.

## Current status

Not implemented. We are currently operating under Option 1:

- strict validator stays strict
- we rely on improved prompts + a targeted repair loop to fix `lookup_rule` predicate typing errors

