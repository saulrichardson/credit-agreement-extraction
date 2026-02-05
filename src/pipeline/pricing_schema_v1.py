from __future__ import annotations

import re
from typing import Annotated, Literal, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

# -----------------------------------------------------------------------------
# Purpose
# -----------------------------------------------------------------------------
#
# This schema is intentionally designed for *computation*:
# - Pricing inputs (variables/metrics/ratings/events) are explicit.
# - Tiering is explicit and machine-evaluable (when `expr` is present).
# - Pricing outputs are schedules (fixed or tiered) + adjustments + constraints.
# - Everything is anchored to evidence via `source_refs` (A####).
#
# It is expected that an extraction pipeline may initially populate `Condition.expr`
# sparsely; the long-term goal is to make `expr` mandatory for production use.


ANCHOR_ID_RE = re.compile(r"^A\d{4,}$")
_NO_SPACES_ID_RE = re.compile(r"^[a-z0-9_]{1,80}$", flags=re.IGNORECASE)
_PLACEHOLDER_IDS = {
    "no_spaces_id",
    "id",
    "grid_id",
    "tier_id",
    "rate_option_id",
    "adjustment_id",
    "item_id",
    "facility_id",
    "parameter_id",
    "var_id",
    "scheme_id",
    "rule_id",
}


def _validate_anchor_ids(anchor_ids: list[str]) -> list[str]:
    if not isinstance(anchor_ids, list):
        raise TypeError("source_refs must be a JSON array")

    cleaned: list[str] = []
    for idx, raw in enumerate(anchor_ids):
        if not isinstance(raw, str):
            raise TypeError(f"anchor IDs must be strings; got {type(raw).__name__} at index {idx}")
        anchor_id = raw.strip()
        if not ANCHOR_ID_RE.fullmatch(anchor_id):
            raise ValueError(f"invalid anchor_id {anchor_id!r} (expected like 'A0001')")
        cleaned.append(anchor_id)
    return cleaned


def _validate_anchor_ids_nonempty(anchor_ids: list[str]) -> list[str]:
    cleaned = _validate_anchor_ids(anchor_ids)
    if not cleaned:
        raise ValueError("source_refs must contain at least one anchor_id")
    return cleaned


def _validate_no_spaces_id(value: str) -> str:
    if not isinstance(value, str):
        raise TypeError("id fields must be strings")
    v = value.strip()
    if not v:
        raise ValueError("id fields must be non-empty")
    if any(ch.isspace() for ch in v):
        raise ValueError(f"invalid id {v!r}: spaces are not allowed")
    if v.lower() in _PLACEHOLDER_IDS:
        raise ValueError(f"invalid id {v!r}: placeholder value not allowed")
    if "verbatim" in v.lower():
        raise ValueError(f"invalid id {v!r}: placeholder value not allowed")
    if not _NO_SPACES_ID_RE.fullmatch(v):
        raise ValueError(f"invalid id {v!r}: expected [a-z0-9_], length <= 80")
    return v


class EvidenceText(BaseModel):
    """A contract-grounded statement with explicit anchor references."""

    model_config = ConfigDict(extra="forbid")

    text: str = Field(description="Verbatim or near-verbatim text from the agreement.")
    source_refs: list[str] = Field(default_factory=list, description="Anchor IDs supporting the text.")

    @field_validator("source_refs")
    @classmethod
    def _source_refs_validate(cls, value: list[str]) -> list[str]:
        return _validate_anchor_ids_nonempty(value)


# -----------------------------------------------------------------------------
# Variables / inputs (evaluation-time)
# -----------------------------------------------------------------------------


class PricingVariable(BaseModel):
    """A named input required to evaluate pricing deterministically."""

    model_config = ConfigDict(extra="forbid")

    var_id: str = Field(description="Stable variable ID referenced by expressions (no spaces).")
    label_raw: str = Field(description="Human label (prefer verbatim from agreement when possible).")
    kind: Literal["number", "boolean", "string", "date", "rating"] = Field(
        description="Evaluation-time type."
    )
    unit: str | None = Field(
        default=None,
        description="Unit hint (e.g., 'ratio', 'usd', 'bps', 'percent', 'rating_grade').",
    )
    notes: str | None = None
    source_refs: list[str] = Field(default_factory=list)

    @field_validator("var_id")
    @classmethod
    def _var_id_validate(cls, value: str) -> str:
        return _validate_no_spaces_id(value)

    @field_validator("source_refs")
    @classmethod
    def _source_refs_validate(cls, value: list[str]) -> list[str]:
        return _validate_anchor_ids(value)


# -----------------------------------------------------------------------------
# Expressions (for computable conditions)
# -----------------------------------------------------------------------------


class NumConst(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["const"] = "const"
    value: float


class NumVar(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["var"] = "var"
    var_id: str

    @field_validator("var_id")
    @classmethod
    def _var_id_validate(cls, value: str) -> str:
        return _validate_no_spaces_id(value)


class NumAdd(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["add"] = "add"
    args: list["NumExpr"]


class NumSub(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["sub"] = "sub"
    left: "NumExpr"
    right: "NumExpr"


class NumMax(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["max"] = "max"
    args: list["NumExpr"]


class NumMin(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["min"] = "min"
    args: list["NumExpr"]


NumExpr = Annotated[
    Union[NumConst, NumVar, NumAdd, NumSub, NumMax, NumMin],
    Field(discriminator="type"),
]


class BoolConst(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["const"] = "const"
    value: bool


class BoolVar(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["var"] = "var"
    var_id: str

    @field_validator("var_id")
    @classmethod
    def _var_id_validate(cls, value: str) -> str:
        return _validate_no_spaces_id(value)


class BoolNot(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["not"] = "not"
    arg: "BoolExpr"


class BoolAnd(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["and"] = "and"
    args: list["BoolExpr"]


class BoolOr(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["or"] = "or"
    args: list["BoolExpr"]


class NumCompare(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["num_compare"] = "num_compare"
    op: Literal["<", "<=", ">", ">=", "==", "!="]
    left: NumExpr
    right: NumExpr


class RatingAtLeast(BaseModel):
    """rating(var) is >= grade_raw on the agency's scale (e.g., S&P 'BBB+ or better')."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["rating_at_least"] = "rating_at_least"
    rating_var_id: str
    agency: Literal["sp", "moodys", "fitch", "other"]
    grade_raw: str = Field(description="Grade threshold, verbatim (e.g., 'BBB+', 'A2').")

    @field_validator("rating_var_id")
    @classmethod
    def _rating_var_id_validate(cls, value: str) -> str:
        return _validate_no_spaces_id(value)


BoolExpr = Annotated[
    Union[BoolConst, BoolVar, BoolNot, BoolAnd, BoolOr, NumCompare, RatingAtLeast],
    Field(discriminator="type"),
]


class Condition(BaseModel):
    """A condition expressed as evidence text plus (optionally) a machine-evaluable expression."""

    model_config = ConfigDict(extra="forbid")

    evidence: EvidenceText
    expr: BoolExpr | None = Field(
        default=None,
        description="Optional machine-evaluable condition. Omit if not yet computable.",
    )


# -----------------------------------------------------------------------------
# Facilities / rate options (objects being priced)
# -----------------------------------------------------------------------------


class Facility(BaseModel):
    model_config = ConfigDict(extra="forbid")

    facility_id: str
    label_raw: str
    facility_type: Literal[
        "revolver",
        "term_loan",
        "delayed_draw_term_loan",
        "swingline",
        "letter_of_credit",
        "other",
    ] = "other"
    currency: str | None = None
    source_refs: list[str] = Field(default_factory=list)

    @field_validator("facility_id")
    @classmethod
    def _facility_id_validate(cls, value: str) -> str:
        return _validate_no_spaces_id(value)

    @field_validator("source_refs")
    @classmethod
    def _source_refs_validate(cls, value: list[str]) -> list[str]:
        return _validate_anchor_ids(value)


class RateOption(BaseModel):
    """A borrower choice (loan type) or a fee-rate label."""

    model_config = ConfigDict(extra="forbid")

    rate_option_id: str
    label_raw: str
    kind: Literal["loan", "fee"]
    facility_id: str | None = None
    fee_basis: str | None = Field(
        default=None,
        description='Fee basis like "undrawn", "letters_of_credit", "facility_commitment", etc.',
    )
    source_refs: list[str] = Field(default_factory=list)

    @field_validator("rate_option_id")
    @classmethod
    def _rate_option_id_validate(cls, value: str) -> str:
        return _validate_no_spaces_id(value)

    @field_validator("facility_id")
    @classmethod
    def _facility_id_ref_validate(cls, value: str | None) -> str | None:
        return _validate_no_spaces_id(value) if value is not None else None

    @field_validator("source_refs")
    @classmethod
    def _source_refs_validate(cls, value: list[str]) -> list[str]:
        return _validate_anchor_ids_nonempty(value)


# -----------------------------------------------------------------------------
# Tiering schemes (how tiers are selected)
# -----------------------------------------------------------------------------


class Tier(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tier_id: str
    label_raw: str
    source_refs: list[str] = Field(default_factory=list)

    @field_validator("tier_id")
    @classmethod
    def _tier_id_validate(cls, value: str) -> str:
        return _validate_no_spaces_id(value)

    @field_validator("source_refs")
    @classmethod
    def _source_refs_validate(cls, value: list[str]) -> list[str]:
        return _validate_anchor_ids_nonempty(value)


class TierRule(BaseModel):
    """Priority-ordered rule assigning an active tier."""

    model_config = ConfigDict(extra="forbid")

    rule_id: str
    tier_id: str
    when: Condition
    priority: int = Field(
        default=100,
        description="Lower numbers run first. If multiple match, the smallest priority wins.",
    )
    source_refs: list[str] = Field(default_factory=list)

    @field_validator("rule_id")
    @classmethod
    def _rule_id_validate(cls, value: str) -> str:
        return _validate_no_spaces_id(value)

    @field_validator("tier_id")
    @classmethod
    def _tier_id_ref_validate(cls, value: str) -> str:
        return _validate_no_spaces_id(value)

    @field_validator("source_refs")
    @classmethod
    def _source_refs_validate(cls, value: list[str]) -> list[str]:
        return _validate_anchor_ids(value)


class TieringScheme(BaseModel):
    """Defines tiers and the rules to select which tier applies."""

    model_config = ConfigDict(extra="forbid")

    scheme_id: str
    label_raw: str
    input_var_ids: list[str] = Field(default_factory=list)
    tiers: list[Tier] = Field(default_factory=list)
    rules: list[TierRule] = Field(default_factory=list)
    default_tier_id: str

    notes: list[EvidenceText] = Field(
        default_factory=list,
        description="Non-computable but important tiering rules (e.g., 'split rating' or timing).",
    )
    source_refs: list[str] = Field(default_factory=list)

    @field_validator("scheme_id")
    @classmethod
    def _scheme_id_validate(cls, value: str) -> str:
        return _validate_no_spaces_id(value)

    @field_validator("input_var_ids")
    @classmethod
    def _input_var_ids_validate(cls, value: list[str]) -> list[str]:
        if not isinstance(value, list):
            raise TypeError("input_var_ids must be a JSON array")
        return [_validate_no_spaces_id(v) for v in value]

    @field_validator("default_tier_id")
    @classmethod
    def _default_tier_id_validate(cls, value: str) -> str:
        return _validate_no_spaces_id(value)

    @field_validator("source_refs")
    @classmethod
    def _source_refs_validate(cls, value: list[str]) -> list[str]:
        return _validate_anchor_ids(value)

    @model_validator(mode="after")
    def _cross_validate(self) -> "TieringScheme":
        tier_ids = {t.tier_id for t in self.tiers}
        if self.default_tier_id not in tier_ids:
            raise ValueError(
                f"default_tier_id {self.default_tier_id!r} not found in tiers (have: {sorted(tier_ids)})"
            )
        for r in self.rules:
            if r.tier_id not in tier_ids:
                raise ValueError(f"TierRule {r.rule_id!r} references unknown tier_id {r.tier_id!r}")
        return self


# -----------------------------------------------------------------------------
# Pricing schedules (fixed or tiered) and rules
# -----------------------------------------------------------------------------


class RateValue(BaseModel):
    model_config = ConfigDict(extra="forbid")

    rate_option_id: str
    value_bps: float = Field(description="Basis points (bps). Percent values must be converted (1.75% -> 175).")
    source_refs: list[str] = Field(default_factory=list)

    @field_validator("rate_option_id")
    @classmethod
    def _rate_option_id_ref_validate(cls, value: str) -> str:
        return _validate_no_spaces_id(value)

    @field_validator("source_refs")
    @classmethod
    def _source_refs_validate(cls, value: list[str]) -> list[str]:
        return _validate_anchor_ids_nonempty(value)


class TierRateValue(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tier_id: str
    rate_option_id: str
    value_bps: float = Field(description="Basis points (bps). Percent values must be converted (1.75% -> 175).")
    source_refs: list[str] = Field(default_factory=list)

    @field_validator("tier_id")
    @classmethod
    def _tier_id_ref_validate(cls, value: str) -> str:
        return _validate_no_spaces_id(value)

    @field_validator("rate_option_id")
    @classmethod
    def _rate_option_id_ref_validate(cls, value: str) -> str:
        return _validate_no_spaces_id(value)

    @field_validator("source_refs")
    @classmethod
    def _source_refs_validate(cls, value: list[str]) -> list[str]:
        return _validate_anchor_ids_nonempty(value)


class FixedSchedule(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["fixed"] = "fixed"

    values: list[RateValue] = Field(default_factory=list)
    source_table_anchor_id: str | None = Field(
        default=None,
        description="If derived from a table, the table anchor_id.",
    )
    source_refs: list[str] = Field(default_factory=list)

    @field_validator("source_table_anchor_id")
    @classmethod
    def _table_anchor_validate(cls, value: str | None) -> str | None:
        if value is None:
            return None
        v = value.strip()
        if not ANCHOR_ID_RE.fullmatch(v):
            raise ValueError(f"invalid table_anchor_id {v!r} (expected like 'A0001')")
        return v

    @field_validator("source_refs")
    @classmethod
    def _source_refs_validate(cls, value: list[str]) -> list[str]:
        return _validate_anchor_ids_nonempty(value)


class TieredSchedule(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["tiered"] = "tiered"

    tiering_scheme_id: str
    cells: list[TierRateValue] = Field(default_factory=list)
    source_table_anchor_id: str | None = Field(
        default=None,
        description="If derived from a table, the table anchor_id.",
    )
    source_refs: list[str] = Field(default_factory=list)

    @field_validator("tiering_scheme_id")
    @classmethod
    def _tiering_scheme_id_validate(cls, value: str) -> str:
        return _validate_no_spaces_id(value)

    @field_validator("source_table_anchor_id")
    @classmethod
    def _table_anchor_validate(cls, value: str | None) -> str | None:
        if value is None:
            return None
        v = value.strip()
        if not ANCHOR_ID_RE.fullmatch(v):
            raise ValueError(f"invalid table_anchor_id {v!r} (expected like 'A0001')")
        return v

    @field_validator("source_refs")
    @classmethod
    def _source_refs_validate(cls, value: list[str]) -> list[str]:
        return _validate_anchor_ids_nonempty(value)


PricingSchedule = Annotated[Union[FixedSchedule, TieredSchedule], Field(discriminator="type")]


class PricingConstraint(BaseModel):
    """Constraints applied to the computed value after schedule + adjustments."""

    model_config = ConfigDict(extra="forbid")

    constraint_id: str
    min_bps: float | None = None
    max_bps: float | None = None
    applies_to_rate_option_ids: list[str] = Field(default_factory=list)
    evidence: EvidenceText
    source_refs: list[str] = Field(default_factory=list)

    @field_validator("constraint_id")
    @classmethod
    def _constraint_id_validate(cls, value: str) -> str:
        return _validate_no_spaces_id(value)

    @field_validator("applies_to_rate_option_ids")
    @classmethod
    def _applies_to_rate_option_ids_validate(cls, value: list[str]) -> list[str]:
        if not isinstance(value, list):
            raise TypeError("applies_to_rate_option_ids must be a JSON array")
        return [_validate_no_spaces_id(v) for v in value]

    @field_validator("source_refs")
    @classmethod
    def _source_refs_validate(cls, value: list[str]) -> list[str]:
        return _validate_anchor_ids_nonempty(value)


class PricingAdjustment(BaseModel):
    """An additive delta in bps applied to a schedule value when a condition holds."""

    model_config = ConfigDict(extra="forbid")

    adjustment_id: str
    label: str
    delta_bps: float
    applies_to_rate_option_ids: list[str] = Field(default_factory=list)
    when: Condition
    floor_bps: float | None = None
    cap_bps: float | None = None
    source_refs: list[str] = Field(default_factory=list)

    @field_validator("adjustment_id")
    @classmethod
    def _adjustment_id_validate(cls, value: str) -> str:
        return _validate_no_spaces_id(value)

    @field_validator("applies_to_rate_option_ids")
    @classmethod
    def _applies_to_rate_option_ids_validate(cls, value: list[str]) -> list[str]:
        if not isinstance(value, list):
            raise TypeError("applies_to_rate_option_ids must be a JSON array")
        return [_validate_no_spaces_id(v) for v in value]

    @field_validator("source_refs")
    @classmethod
    def _source_refs_validate(cls, value: list[str]) -> list[str]:
        return _validate_anchor_ids_nonempty(value)


class PricingParameterRule(BaseModel):
    """A prioritized rule that selects a schedule + adjustments for a pricing parameter."""

    model_config = ConfigDict(extra="forbid")

    rule_id: str
    priority: int = Field(default=100)
    when: Condition
    schedule: PricingSchedule
    adjustments: list[PricingAdjustment] = Field(default_factory=list)
    constraints: list[PricingConstraint] = Field(default_factory=list)
    source_refs: list[str] = Field(default_factory=list)

    @field_validator("rule_id")
    @classmethod
    def _rule_id_validate(cls, value: str) -> str:
        return _validate_no_spaces_id(value)

    @field_validator("source_refs")
    @classmethod
    def _source_refs_validate(cls, value: list[str]) -> list[str]:
        return _validate_anchor_ids_nonempty(value)


class PricingParameter(BaseModel):
    """A named pricing output (margin, fee rate, etc.) that can be evaluated."""

    model_config = ConfigDict(extra="forbid")

    parameter_id: str
    label_raw: str
    kind: Literal[
        "margin_bps",
        "fee_rate_bps",
        "all_in_rate_bps",
        "default_interest_spread_bps",
        "other_bps",
    ] = "other_bps"
    rate_option_ids: list[str] = Field(default_factory=list)
    rules: list[PricingParameterRule] = Field(default_factory=list)
    source_refs: list[str] = Field(default_factory=list)

    @field_validator("parameter_id")
    @classmethod
    def _parameter_id_validate(cls, value: str) -> str:
        return _validate_no_spaces_id(value)

    @field_validator("rate_option_ids")
    @classmethod
    def _rate_option_ids_validate(cls, value: list[str]) -> list[str]:
        if not isinstance(value, list):
            raise TypeError("rate_option_ids must be a JSON array")
        return [_validate_no_spaces_id(v) for v in value]

    @field_validator("source_refs")
    @classmethod
    def _source_refs_validate(cls, value: list[str]) -> list[str]:
        return _validate_anchor_ids(value)

    @model_validator(mode="after")
    def _cross_validate(self) -> "PricingParameter":
        if not self.rules:
            raise ValueError("PricingParameter.rules must not be empty")
        if not any(r.priority == min(rr.priority for rr in self.rules) for r in self.rules):
            # Should never happen, but keeps rule ordering intent explicit.
            raise ValueError("PricingParameter.rules priorities are inconsistent")
        return self


# -----------------------------------------------------------------------------
# Agreement-level model
# -----------------------------------------------------------------------------


class AgreementInfo(BaseModel):
    model_config = ConfigDict(extra="forbid")

    title: str | None = None
    as_of_date: str | None = Field(default=None, description="YYYY-MM-DD when available")
    currency: str | None = None
    source_refs: list[str] = Field(default_factory=list)

    @field_validator("source_refs")
    @classmethod
    def _source_refs_validate(cls, value: list[str]) -> list[str]:
        return _validate_anchor_ids(value)


class AgreementPricingModelV1(BaseModel):
    """Computation-oriented pricing schema."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["agreement_pricing_v1"] = "agreement_pricing_v1"

    issuer: str | None = None
    agreement: AgreementInfo = Field(default_factory=AgreementInfo)

    facilities: list[Facility] = Field(default_factory=list)
    rate_options: list[RateOption] = Field(default_factory=list)

    variables: list[PricingVariable] = Field(default_factory=list)
    tiering_schemes: list[TieringScheme] = Field(default_factory=list)
    pricing_parameters: list[PricingParameter] = Field(default_factory=list)

    @model_validator(mode="after")
    def _cross_validate(self) -> "AgreementPricingModelV1":
        facility_ids = {f.facility_id for f in self.facilities}
        rate_option_ids = {r.rate_option_id for r in self.rate_options}
        var_ids = {v.var_id for v in self.variables}
        scheme_ids = {s.scheme_id for s in self.tiering_schemes}

        # facility references
        for ro in self.rate_options:
            if ro.facility_id and ro.facility_id not in facility_ids:
                raise ValueError(
                    f"RateOption {ro.rate_option_id!r} references unknown facility_id {ro.facility_id!r}"
                )

        # scheme -> variable references
        for s in self.tiering_schemes:
            for vid in s.input_var_ids:
                if vid not in var_ids:
                    raise ValueError(f"TieringScheme {s.scheme_id!r} references unknown var_id {vid!r}")

        # schedule references
        for p in self.pricing_parameters:
            for rid in p.rate_option_ids:
                if rid not in rate_option_ids:
                    raise ValueError(f"PricingParameter {p.parameter_id!r} references unknown rate_option_id {rid!r}")
            for rule in p.rules:
                schedule = rule.schedule
                if isinstance(schedule, TieredSchedule):
                    if schedule.tiering_scheme_id not in scheme_ids:
                        raise ValueError(
                            f"PricingParameterRule {rule.rule_id!r} references unknown tiering_scheme_id {schedule.tiering_scheme_id!r}"
                        )
                # validate adjustment/constraint rate option IDs if present
                for adj in rule.adjustments:
                    for rid in adj.applies_to_rate_option_ids:
                        if rid not in rate_option_ids:
                            raise ValueError(
                                f"PricingAdjustment {adj.adjustment_id!r} references unknown rate_option_id {rid!r}"
                            )
                for c in rule.constraints:
                    for rid in c.applies_to_rate_option_ids:
                        if rid not in rate_option_ids:
                            raise ValueError(
                                f"PricingConstraint {c.constraint_id!r} references unknown rate_option_id {rid!r}"
                            )

        return self

