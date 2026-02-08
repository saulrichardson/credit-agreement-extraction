from __future__ import annotations

"""
pricing_schema_v2

Goal: represent corporate credit agreement pricing in a way that is:
- expressive across formats (tables, prose, ASCII tables)
- grounded (every claim cites anchor IDs)
- easy to evaluate downstream (piecewise rules + explicit bases/indices)

This is intentionally *not* tied to any specific input format (HTML/table parsing).
It is a semantic representation that an LLM can populate from anchored text.

Notes
- This schema is a forward-looking successor to pricing_schema_v1 and the current
  contract_pricing_v3 output schema (contract_schemas.py).
- It adds first-class support for:
  - index-based rates (e.g. "150 bps over LIBOR", "Prime Rate")
  - money amounts (e.g. "$75,000 facility fee")
  - explicit charge bases (e.g. unused commitment vs L/C face amount)
"""

import re
from decimal import Decimal
from typing import Annotated, Literal, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


ANCHOR_ID_RE = re.compile(r"^A\d{4,}$")
NO_SPACES_ID_RE = re.compile(r"^[a-z0-9_]{1,80}$")


def _validate_anchor_ids(anchor_ids: list[str]) -> list[str]:
    if not isinstance(anchor_ids, list):
        raise TypeError("source_refs must be a JSON array")
    cleaned: list[str] = []
    for idx, raw in enumerate(anchor_ids):
        if not isinstance(raw, str):
            raise TypeError(f"anchor IDs must be strings; got {type(raw).__name__} at index {idx}")
        aid = raw.strip()
        if not ANCHOR_ID_RE.fullmatch(aid):
            raise ValueError(f"invalid anchor_id {aid!r} (expected like 'A0001')")
        cleaned.append(aid)
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
    if not NO_SPACES_ID_RE.fullmatch(v):
        raise ValueError(f"invalid id {v!r}: expected [a-z0-9_], length <= 80")
    return v


class EvidenceText(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str = Field(description="Verbatim or near-verbatim text grounded in the agreement.")
    source_refs: list[str] = Field(default_factory=list, description="Anchor IDs supporting the text.")

    @field_validator("source_refs")
    @classmethod
    def _source_refs_validate(cls, value: list[str]) -> list[str]:
        return _validate_anchor_ids_nonempty(value)


class Condition(BaseModel):
    """A condition used to gate regimes / pricing branches.

    We keep this evidence-first. A future iteration can add a machine-evaluable expression tree,
    but the key requirement for extraction is the anchored text.
    """

    model_config = ConfigDict(extra="forbid")

    evidence: EvidenceText


class Money(BaseModel):
    model_config = ConfigDict(extra="forbid")

    currency: str = Field(default="USD", description="ISO currency code when known (default USD).")
    amount: Decimal = Field(description="Amount in currency units (e.g., 75000).")
    source_refs: list[str] = Field(default_factory=list)

    @field_validator("source_refs")
    @classmethod
    def _source_refs_validate(cls, value: list[str]) -> list[str]:
        return _validate_anchor_ids_nonempty(value)


class RateIndex(BaseModel):
    """A base reference index for interest rates.

    Examples: Term SOFR, Daily Simple SOFR, LIBOR, Prime Rate, Base Rate, CD Rate, Index Rate.
    """

    model_config = ConfigDict(extra="forbid")

    index_id: str
    label_raw: str
    kind: Literal[
        "sofr",
        "libor",
        "prime",
        "base_rate",
        "cd",
        "index_rate",
        "other",
    ] = "other"
    tenor: str | None = Field(default=None, description="If applicable (e.g., '1m', '3m').")
    source_refs: list[str] = Field(default_factory=list)

    @field_validator("index_id")
    @classmethod
    def _index_id_validate(cls, value: str) -> str:
        return _validate_no_spaces_id(value)

    @field_validator("source_refs")
    @classmethod
    def _source_refs_validate(cls, value: list[str]) -> list[str]:
        return _validate_anchor_ids_nonempty(value)


class RateFormulaIndexPlusSpread(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["index_plus_spread"] = "index_plus_spread"
    index_id: str
    spread_bps: Decimal = Field(description="Spread in basis points (e.g., 150 for 1.50%).")
    source_refs: list[str] = Field(default_factory=list)

    @field_validator("index_id")
    @classmethod
    def _index_id_validate(cls, value: str) -> str:
        return _validate_no_spaces_id(value)

    @field_validator("source_refs")
    @classmethod
    def _source_refs_validate(cls, value: list[str]) -> list[str]:
        return _validate_anchor_ids_nonempty(value)


class RateFormulaFixedPercent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["fixed_percent"] = "fixed_percent"
    rate_percent: Decimal = Field(
        description="Annualized percent rate (e.g., 2.50 for 2.50% per annum)."
    )
    source_refs: list[str] = Field(default_factory=list)

    @field_validator("source_refs")
    @classmethod
    def _source_refs_validate(cls, value: list[str]) -> list[str]:
        return _validate_anchor_ids_nonempty(value)


RateFormula = Annotated[
    Union[RateFormulaIndexPlusSpread, RateFormulaFixedPercent],
    Field(discriminator="type"),
]


class ChargeRate(BaseModel):
    """A value that is a rate (either index-based or fixed)."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["rate"] = "rate"
    formula: RateFormula


class ChargeMoney(BaseModel):
    """A value that is a money amount."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["money"] = "money"
    money: Money


ChargeValue = Annotated[Union[ChargeRate, ChargeMoney], Field(discriminator="type")]


class ChargeBase(BaseModel):
    """What the rate/fee is applied to (for downstream calculation)."""

    model_config = ConfigDict(extra="forbid")

    base_kind: Literal[
        "outstanding_principal",
        "facility_commitment",
        "unused_commitment",
        "lc_face_amount",
        "lc_drawing",
        "other",
        "none",
    ]
    label_raw: str | None = Field(
        default=None,
        description="Optional verbatim-ish label for the base (e.g., 'average daily unused commitment').",
    )
    source_refs: list[str] = Field(default_factory=list)

    @field_validator("source_refs")
    @classmethod
    def _source_refs_validate(cls, value: list[str]) -> list[str]:
        return _validate_anchor_ids_nonempty(value)


class ChargeBranch(BaseModel):
    model_config = ConfigDict(extra="forbid")

    when: Condition
    value: ChargeValue
    source_refs: list[str] = Field(default_factory=list)

    @field_validator("source_refs")
    @classmethod
    def _source_refs_validate(cls, value: list[str]) -> list[str]:
        return _validate_anchor_ids_nonempty(value)


class PiecewiseCharge(BaseModel):
    model_config = ConfigDict(extra="forbid")

    branches: list[ChargeBranch] = Field(default_factory=list)
    default_value: ChargeValue | None = None

    @model_validator(mode="after")
    def _nonempty(self) -> "PiecewiseCharge":
        if not self.branches and self.default_value is None:
            raise ValueError("PiecewiseCharge must have at least one branch or a default_value")
        return self


class Charge(BaseModel):
    """A single pricing output with evaluation semantics."""

    model_config = ConfigDict(extra="forbid")

    charge_id: str
    label_raw: str
    kind: Literal[
        "interest_rate",
        "margin_spread",
        "commitment_fee",
        "lc_fee",
        "facility_fee",
        "upfront_fee",
        "default_rate",
        "other",
    ] = "other"
    applies_to_facility_id: str | None = None
    applies_to_rate_option_label_raw: str | None = Field(
        default=None,
        description="Optional human label of the loan/fee option (e.g., 'ABR', 'LIBO Rate').",
    )
    base: ChargeBase | None = None
    value: PiecewiseCharge
    source_refs: list[str] = Field(default_factory=list)

    @field_validator("charge_id")
    @classmethod
    def _charge_id_validate(cls, value: str) -> str:
        return _validate_no_spaces_id(value)

    @field_validator("applies_to_facility_id")
    @classmethod
    def _facility_id_validate(cls, value: str | None) -> str | None:
        return _validate_no_spaces_id(value) if value is not None else None

    @field_validator("source_refs")
    @classmethod
    def _source_refs_validate(cls, value: list[str]) -> list[str]:
        return _validate_anchor_ids_nonempty(value)


class PricingRegime(BaseModel):
    model_config = ConfigDict(extra="forbid")

    regime_id: str
    label: str
    applies_when: Condition
    charges: list[Charge] = Field(default_factory=list)
    source_refs: list[str] = Field(default_factory=list)

    @field_validator("regime_id")
    @classmethod
    def _regime_id_validate(cls, value: str) -> str:
        return _validate_no_spaces_id(value)

    @field_validator("source_refs")
    @classmethod
    def _source_refs_validate(cls, value: list[str]) -> list[str]:
        return _validate_anchor_ids_nonempty(value)

    @model_validator(mode="after")
    def _nonempty(self) -> "PricingRegime":
        if not self.charges:
            raise ValueError("PricingRegime.charges must not be empty")
        return self


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


class AgreementPricingModelV2(BaseModel):
    """Expressive, evaluation-friendly pricing model."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["agreement_pricing_v2"] = "agreement_pricing_v2"

    issuer: str | None = None
    agreement: AgreementInfo = Field(default_factory=AgreementInfo)

    rate_indices: list[RateIndex] = Field(default_factory=list)
    regimes: list[PricingRegime] = Field(default_factory=list)

    @model_validator(mode="after")
    def _cross_validate(self) -> "AgreementPricingModelV2":
        if not self.regimes:
            raise ValueError("AgreementPricingModelV2.regimes must not be empty")
        index_ids = {r.index_id for r in self.rate_indices}
        for regime in self.regimes:
            for charge in regime.charges:
                # Validate index references inside rate formulas.
                for branch in charge.value.branches:
                    v = branch.value
                    if isinstance(v, ChargeRate):
                        f = v.formula
                        if isinstance(f, RateFormulaIndexPlusSpread) and f.index_id not in index_ids:
                            raise ValueError(
                                f"Charge {charge.charge_id!r} references unknown index_id {f.index_id!r}"
                            )
                if charge.value.default_value is not None and isinstance(charge.value.default_value, ChargeRate):
                    f = charge.value.default_value.formula
                    if isinstance(f, RateFormulaIndexPlusSpread) and f.index_id not in index_ids:
                        raise ValueError(
                            f"Charge {charge.charge_id!r} references unknown index_id {f.index_id!r}"
                        )
        return self

