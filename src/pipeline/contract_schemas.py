from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

ANCHOR_ID_RE = re.compile(r"^A\d{4,}$")


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


class EvidenceText(BaseModel):
    """A text statement with explicit anchor references that support it."""

    model_config = ConfigDict(extra="forbid")

    text: str = Field(description="Verbatim or near-verbatim text describing a condition or rule.")
    source_refs: list[str] = Field(default_factory=list, description="Anchor IDs supporting this text.")

    @field_validator("source_refs")
    @classmethod
    def _source_refs_validate(cls, value: list[str]) -> list[str]:
        return _validate_anchor_ids(value)


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


class RateOption(BaseModel):
    """A borrower choice (loan type) or a fee rate label seen in pricing tables."""

    model_config = ConfigDict(extra="forbid")

    rate_option_id: str = Field(description="Stable ID for referencing this option in grids (no spaces).")
    label_raw: str = Field(description="Exact label from a table header or row label (verbatim).")
    kind: Literal["loan", "fee"]

    # Optional (used when kind == 'fee')
    fee_basis: str | None = Field(
        default=None,
        description='Fee basis like "undrawn", "letters_of_credit", "facility_commitment", or null if unclear.',
    )

    source_refs: list[str] = Field(default_factory=list)

    @field_validator("source_refs")
    @classmethod
    def _source_refs_validate(cls, value: list[str]) -> list[str]:
        return _validate_anchor_ids(value)


class TierTest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    metric_label_raw: str = Field(description="Verbatim metric label (e.g., leverage ratio label).")
    op: Literal["<", "<=", ">", ">=", "==", "!="]
    value: float
    source_refs: list[str] = Field(default_factory=list)

    @field_validator("source_refs")
    @classmethod
    def _source_refs_validate(cls, value: list[str]) -> list[str]:
        return _validate_anchor_ids(value)


class PricingTier(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tier_id: str
    label_raw: str
    tests: list[TierTest] = Field(default_factory=list)
    source_refs: list[str] = Field(default_factory=list)

    @field_validator("source_refs")
    @classmethod
    def _source_refs_validate(cls, value: list[str]) -> list[str]:
        return _validate_anchor_ids(value)


class PricingCell(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tier_id: str
    rate_option_id: str
    value_bps: float = Field(description="Basis points (bps). Percent values must be converted (1.75% -> 175).")
    source_refs: list[str] = Field(default_factory=list)

    @field_validator("source_refs")
    @classmethod
    def _source_refs_validate(cls, value: list[str]) -> list[str]:
        return _validate_anchor_ids(value)


class PricingGrid(BaseModel):
    """A single pricing table or schedule encoded as a tier x rate_option matrix."""

    model_config = ConfigDict(extra="forbid")

    grid_id: str
    table_anchor_id: str = Field(description="Anchor ID whose text contains the [[TABLE]] block.")
    facility_id: str | None = None

    tier_metric_label_raw: str | None = Field(
        default=None,
        description="Verbatim label for the tiering axis metric (if applicable).",
    )
    tiers: list[PricingTier]
    rate_option_ids: list[str] = Field(description="IDs of rate options covered by this grid.")
    cells: list[PricingCell]

    source_refs: list[str] = Field(default_factory=list)

    @field_validator("table_anchor_id")
    @classmethod
    def _table_anchor_validate(cls, value: str) -> str:
        if not isinstance(value, str):
            raise TypeError("table_anchor_id must be a string")
        v = value.strip()
        if not ANCHOR_ID_RE.fullmatch(v):
            raise ValueError(f"invalid table_anchor_id {v!r} (expected like 'A0001')")
        return v

    @field_validator("source_refs")
    @classmethod
    def _source_refs_validate(cls, value: list[str]) -> list[str]:
        return _validate_anchor_ids(value)


class PricingAdjustment(BaseModel):
    """A delta applied on top of an otherwise-applicable grid value."""

    model_config = ConfigDict(extra="forbid")

    adjustment_id: str
    label: str
    delta_bps: float
    applies_to_rate_option_ids: list[str] = Field(default_factory=list)
    applies_when: EvidenceText
    floor_bps: float | None = None
    cap_bps: float | None = None
    source_refs: list[str] = Field(default_factory=list)

    @field_validator("source_refs")
    @classmethod
    def _source_refs_validate(cls, value: list[str]) -> list[str]:
        return _validate_anchor_ids(value)


class FlatPricingItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    item_id: str
    rate_option_id: str
    value_bps: float
    applies_when: EvidenceText | None = None
    source_refs: list[str] = Field(default_factory=list)

    @field_validator("source_refs")
    @classmethod
    def _source_refs_validate(cls, value: list[str]) -> list[str]:
        return _validate_anchor_ids(value)


class PricingRegime(BaseModel):
    model_config = ConfigDict(extra="forbid")

    regime_id: str
    label: str
    applies_when: EvidenceText
    grids: list[PricingGrid] = Field(default_factory=list)
    adjustments: list[PricingAdjustment] = Field(default_factory=list)
    flat_items: list[FlatPricingItem] = Field(default_factory=list)
    source_refs: list[str] = Field(default_factory=list)

    @field_validator("source_refs")
    @classmethod
    def _source_refs_validate(cls, value: list[str]) -> list[str]:
        return _validate_anchor_ids(value)


class ContractPricingModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    issuer: str | None = None
    agreement: AgreementInfo = Field(default_factory=AgreementInfo)
    rate_options: list[RateOption] = Field(default_factory=list)
    pricing_regimes: list[PricingRegime] = Field(default_factory=list)


class ContractPricingArtifact(BaseModel):
    """Artifact persisted under runs/<run_id>/contract_pricing/<item_id>.json."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["contract_pricing_v1"]
    stage: Literal["contract_pricing"]
    run_id: str
    item_id: str
    created_at: int

    gateway_url: str
    model: str
    reasoning_effort: str
    temperature: float

    prompt: str
    prompt_sha256: str

    attempts_used: int
    pricing: ContractPricingModel


class ContractPricingTableExtract(BaseModel):
    """LLM output for a single pricing table + nearby footnotes."""

    model_config = ConfigDict(extra="forbid")

    rate_options: list[RateOption] = Field(default_factory=list)
    grid: PricingGrid
    adjustments: list[PricingAdjustment] = Field(default_factory=list)
    flat_items: list[FlatPricingItem] = Field(default_factory=list)
