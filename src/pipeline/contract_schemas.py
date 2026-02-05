from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

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
    """A text statement with explicit anchor references that support it."""

    model_config = ConfigDict(extra="forbid")

    text: str = Field(description="Verbatim or near-verbatim text describing a condition or rule.")
    source_refs: list[str] = Field(default_factory=list, description="Anchor IDs supporting this text.")

    @field_validator("source_refs")
    @classmethod
    def _source_refs_validate(cls, value: list[str]) -> list[str]:
        return _validate_anchor_ids_nonempty(value)


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

    @field_validator("rate_option_id")
    @classmethod
    def _rate_option_id_validate(cls, value: str) -> str:
        return _validate_no_spaces_id(value)

    @field_validator("source_refs")
    @classmethod
    def _source_refs_validate(cls, value: list[str]) -> list[str]:
        return _validate_anchor_ids_nonempty(value)


class TierTest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    metric_label_raw: str = Field(description="Verbatim metric label (e.g., leverage ratio label).")
    op: Literal["<", "<=", ">", ">=", "==", "!="]
    value: float
    source_refs: list[str] = Field(default_factory=list)

    @field_validator("op", mode="before")
    @classmethod
    def _op_validate(cls, value: str) -> str:
        """Normalize common operator variants emitted by LLMs.

        Keep the canonical set small so downstream logic stays deterministic.
        """

        if not isinstance(value, str):
            return value
        v = value.strip()

        # Some models will emit the literal unicode escape sequence inside JSON (e.g., "\\u2264")
        # instead of the actual character (e.g., "≤"). Decode it if possible.
        if len(v) == 6 and v.lower().startswith("\\u"):
            try:
                v = chr(int(v[2:], 16))
            except Exception:
                pass

        aliases = {
            "=": "==",
            "≤": "<=",
            "≥": ">=",
            "≠": "!=",
            "<>": "!=",
        }
        return aliases.get(v, v)

    @field_validator("source_refs")
    @classmethod
    def _source_refs_validate(cls, value: list[str]) -> list[str]:
        return _validate_anchor_ids_nonempty(value)


class PricingTier(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tier_id: str
    label_raw: str
    tests: list[TierTest] = Field(default_factory=list)
    source_refs: list[str] = Field(default_factory=list)

    @field_validator("tier_id")
    @classmethod
    def _tier_id_validate(cls, value: str) -> str:
        return _validate_no_spaces_id(value)

    @field_validator("source_refs")
    @classmethod
    def _source_refs_validate(cls, value: list[str]) -> list[str]:
        return _validate_anchor_ids_nonempty(value)


class PricingCell(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tier_id: str
    rate_option_id: str
    value_bps: float = Field(description="Basis points (bps). Percent values must be converted (1.75% -> 175).")
    source_refs: list[str] = Field(default_factory=list)

    @field_validator("source_refs")
    @classmethod
    def _source_refs_validate(cls, value: list[str]) -> list[str]:
        return _validate_anchor_ids_nonempty(value)


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

    @field_validator("grid_id")
    @classmethod
    def _grid_id_validate(cls, value: str) -> str:
        return _validate_no_spaces_id(value)

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
        return _validate_anchor_ids_nonempty(value)


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

    @field_validator("adjustment_id")
    @classmethod
    def _adjustment_id_validate(cls, value: str) -> str:
        return _validate_no_spaces_id(value)

    @field_validator("source_refs")
    @classmethod
    def _source_refs_validate(cls, value: list[str]) -> list[str]:
        return _validate_anchor_ids_nonempty(value)


class FlatPricingItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    item_id: str
    rate_option_id: str
    value_bps: float
    applies_when: EvidenceText | None = None
    source_refs: list[str] = Field(default_factory=list)

    @field_validator("item_id")
    @classmethod
    def _item_id_validate(cls, value: str) -> str:
        return _validate_no_spaces_id(value)

    @field_validator("source_refs")
    @classmethod
    def _source_refs_validate(cls, value: list[str]) -> list[str]:
        return _validate_anchor_ids_nonempty(value)


class PricingRegime(BaseModel):
    model_config = ConfigDict(extra="forbid")

    regime_id: str
    label: str
    applies_when: EvidenceText
    grids: list[PricingGrid] = Field(default_factory=list)
    adjustments: list[PricingAdjustment] = Field(default_factory=list)
    flat_items: list[FlatPricingItem] = Field(default_factory=list)
    source_refs: list[str] = Field(default_factory=list)

    @field_validator("regime_id")
    @classmethod
    def _regime_id_validate(cls, value: str) -> str:
        return _validate_no_spaces_id(value)

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

    schema_version: Literal["contract_pricing_v1", "contract_pricing_v2", "contract_pricing_v3"]
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

    # Optional v2 planner metadata (kept optional so v1 artifacts still validate).
    planner_prompt: str | None = None
    planner_prompt_sha256: str | None = None
    planner_model: str | None = None
    planner_reasoning_effort: str | None = None
    planner_temperature: float | None = None
    planner_max_steps: int | None = None


class ContractPricingTableExtract(BaseModel):
    """LLM output for a single pricing table + nearby footnotes."""

    model_config = ConfigDict(extra="forbid")

    rate_options: list[RateOption] = Field(default_factory=list)
    grid: PricingGrid
    adjustments: list[PricingAdjustment] = Field(default_factory=list)
    flat_items: list[FlatPricingItem] = Field(default_factory=list)
