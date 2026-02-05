# `agreement_pricing_v1` (Computation-Oriented Pricing Schema)

> Legacy snapshot: this “pricing-as-code” methodology is preserved on branch `legacy/pricing-as-code-v1`.
> The canonical branch intentionally does not include the implementation or CLI wiring for this approach.

This document describes the **new target Pydantic schema** in:

- `src/pipeline/pricing_schema_v1.py`

Goal: encode the **complete pricing structure** of a credit agreement (margins + fees + tiering + overrides + adjustments) so downstream pricing calculations are **mechanical** and do not require re-reading the contract.

This is intentionally **not** a “table parser schema”. It’s a **pricing engine schema**: tables are just one possible evidence source.

---

## Status / relationship to ContractIR

This document describes the pricing-as-code pipeline exposed as:
- `pipeline agreement-pricing-v1` (`src/pipeline/agreement_pricing_v1.py`)

It is **separate** from the current ContractIR / CovenantIR direction (pricing-kernel + covenant extraction):
- ContractIR v0.2: `src/pipeline/contract_ir_v0_2.py`
- CovenantIR v0.1: `src/pipeline/covenant_ir_v0_1.py`

In particular, `agreement_pricing_v1` uses **priority-ordered** rule evaluation (see `priority` in `src/pipeline/pricing_schema_v1.py`),
but ContractIR / CovenantIR uses `lookup_rule(...)` with **strict** semantics (0 or >1 matching rows is a hard error, and there is no implicit precedence).

If you are iterating on the newer ContractIR / CovenantIR pipelines, you can ignore this schema. See `../contractir_v0_2_pricing_kernel.md` for the ContractIR pricing-kernel approach.

---

## Why a new schema?

Across the sample agreements in `runs/dan/normalized/*/canonical_annotated.txt`, pricing is encoded in multiple ways:

1) **Tiered grids** in explicit `[[TABLE]]` blocks
   - Example: `runs/dan/normalized/0001193125-23-000306_2/canonical_annotated.txt:856` (“Applicable Margin” table by Fixed Charge Coverage Ratio).

2) **Table-like grids inside prose definitions** (not necessarily tagged as `[[TABLE]]`)
   - Example: `runs/dan/normalized/0000041850-96-000001_5/canonical_annotated.txt:15` (rating → ABR/LIBOR/CD margins).

3) **Overrides and fallback rules** in prose
   - Initial-period overrides (“set to Level III until Dec 31, 2021”) and default overrides (“may be set to Level III”).
   - Missing-certificate fallbacks (“set to highest until delivered”).

4) **Conditional alternative pricing**
   - Example: letter of credit fee lower if cash-collateralized:
     `runs/dan/normalized/0000041850-96-000001_5/canonical_annotated.txt:27`

5) **Multi-agency rating logic** (split ratings, optional Fitch inclusion, timing)
   - Example: `runs/dan/normalized/0001140361-23-000046_9/canonical_annotated.txt:1849` (`Status` + split rating + third agency + leverage override + effective date rules).

The existing `ContractPricingModel` (`src/pipeline/contract_schemas.py`) is a good **table extraction** schema, but it is not sufficient for “pricing calculations are trivial” because:

- tier selection can be non-numeric (ratings) and is often stateful/override-driven,
- many key rules are not in tables,
- “applies_when” is free text rather than a computable condition.

---

## Core concepts in `agreement_pricing_v1`

### 1) Variables (`PricingVariable`)
Every non-trivial pricing rule depends on explicit inputs.

Examples of variables you should expect to see for common agreements:
- `sp_rating` (kind=`rating`)
- `moodys_rating` (kind=`rating`)
- `fixed_charge_coverage_ratio` (kind=`number`, unit=`ratio`)
- `average_revolver_usage_pct` (kind=`number`, unit=`percent`)
- `event_of_default_continuing` (kind=`boolean`)
- `financials_delivered_on_time` (kind=`boolean`)
- `aggregate_advances_usd` (kind=`number`, unit=`usd`)
- `is_cash_collateralized` (kind=`boolean`)

Variables exist to make downstream calculations a pure function of:
`(agreement_pricing_model, inputs) -> outputs`.

### 2) Computable conditions (`Condition`, `BoolExpr`, `NumExpr`)
Every rule has:
- contract-grounded evidence (`EvidenceText`)
- an optional machine-evaluable expression (`expr`)

Long-term direction: treat `expr` as mandatory for production use; in the short-term the pipeline can populate `expr` incrementally while still failing loudly where computability is required.

### 3) Tiering schemes (`TieringScheme`)
Tiering is where contracts vary the most (ratings vs ratios vs usage vs bespoke rules).

A `TieringScheme` contains:
- tier definitions (`Tier`) with labels + evidence
- priority-ordered tier rules (`TierRule`) that assign which tier applies
- a `default_tier_id` for fall-through

Example: Kimco “Status” tiers (Level I–VI) are defined in prose with multi-agency rating thresholds:
`runs/dan/normalized/0001140361-23-000046_9/canonical_annotated.txt:1849`

Important: complicated, stateful, or timing-dependent aspects of tiering can be recorded under `TieringScheme.notes` with evidence anchors (so we never lose fidelity), while the computable core goes into `TierRule.when.expr`.

### 4) Pricing schedules (`FixedSchedule` / `TieredSchedule`)
Schedules are the “numbers” part:

- `FixedSchedule`: constant values (possibly per rate option).
- `TieredSchedule`: a tier × rate_option matrix keyed by a `tiering_scheme_id`.

Both support:
- `source_table_anchor_id` if derived from an actual `[[TABLE]]`
- `source_refs` evidence anchors even when the numbers are in prose

### 5) Pricing parameters (`PricingParameter`)
A `PricingParameter` is a named output you want to compute, like:
- Applicable Margin (margin_bps)
- Facility Fee Rate (fee_rate_bps)
- L/C Fee Rate (fee_rate_bps)
- Unused Line Fee Percentage (fee_rate_bps)

It contains prioritized `PricingParameterRule`s. Each rule specifies:
- `when` (condition)
- `schedule` (fixed or tiered)
- optional `adjustments` and `constraints`

This rule-list structure is how you model:
- initial-period overrides
- default overrides
- “if certificate missing, use highest”
- alternative grids (sustainability pricing, etc.)

---

## Example mapping notes (what “complete pricing structure” means)

### A) Rating tier grid + threshold adjustment (1996 amendment)
From:
- `runs/dan/normalized/0000041850-96-000001_5/canonical_annotated.txt:15` (rating → margin by loan type)
- `runs/dan/normalized/0000041850-96-000001_5/canonical_annotated.txt:18` (+0.15% if Advances > $10,000,000)

Schema approach:
- `PricingVariable`: `sp_rating` (rating), `aggregate_advances_usd` (number/usd)
- `TieringScheme`: rating tiers (B+ or lower, BB-, BB, BB+, BBB- or higher)
- `PricingParameter`: `applicable_margin`
  - schedule: `TieredSchedule` mapping rating tier × {ABR, LIBOR, CD} → margin_bps
  - adjustment: +15 bps when `aggregate_advances_usd > 10_000_000`

### B) Ratio-based tiers + quarterly re-determination + default overrides (2023 credit agreement)
From:
- `runs/dan/normalized/0001193125-23-000306_2/canonical_annotated.txt:856` (Applicable Margin, FCCR tiers)
- `runs/dan/normalized/0001193125-23-000306_2/canonical_annotated.txt:873` (override to Level III until 2021-12-31)
- `runs/dan/normalized/0001193125-23-000306_2/canonical_annotated.txt:882` (fallback to Level III if certificate missing)
- `runs/dan/normalized/0001193125-23-000306_2/canonical_annotated.txt:888` (retroactive correction)

Schema approach:
- `PricingVariable`: `fixed_charge_coverage_ratio`, `certificate_delivered_on_time`, `event_of_default_continuing`, `required_lenders_directed_level_iii`
- `TieringScheme`: tiers based on ratio comparisons
- `PricingParameter`: `applicable_margin`
  - Rule (highest priority): initial-period override to Level III (requires a `date` variable)
  - Rule: if certificate missing -> Level III (or highest tier)
  - Rule: base tiered schedule

Note: retroactive correction is stateful; it can be captured in `notes` + variables (e.g., `certificate_corrected`) if you want the evaluator to model it.

### C) Multi-agency rating “Status” tiers (Kimco example)
From:
- `runs/dan/normalized/0001140361-23-000046_9/canonical_annotated.txt:1849` (Status definition, including split/third agency and leverage override)

Schema approach:
- `PricingVariable`: S&P + Moody’s ratings, plus optional Fitch rating, plus leverage ratio, plus “fitch_included” boolean.
- `TieringScheme`: tiers Level I–VI, with rules using `rating_at_least` expressions.
- Store complicated aggregation/timing rules under `TieringScheme.notes` with citations until the evaluator supports them fully.

---

## What this schema does NOT yet guarantee

This schema is the *target shape*. Building an extractor that populates it fully (especially `Condition.expr`) still requires:
- a robust “conditions → expression” compiler,
- handling stateful/timing rules when needed for downstream calculations,
- model-side validation that each adjustment applies to the correct parameter/rate option.

The schema is designed so those can be added without changing the output shape.
