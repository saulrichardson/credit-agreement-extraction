from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


_ANCHOR_ID_RE = r"^A\d{4,}$"


def _validate_anchor_ids(values: list[str]) -> list[str]:
    import re

    if not isinstance(values, list) or not values:
        raise ValueError("source_refs must be a non-empty list")
    for v in values:
        if not isinstance(v, str) or not re.match(_ANCHOR_ID_RE, v):
            raise ValueError(f"Invalid anchor_id in source_refs: {v!r}")
    return values


class SemanticScanChunkRef(BaseModel):
    model_config = ConfigDict(extra="forbid")

    start_order: int
    end_order: int
    anchor_ids: list[str] = Field(min_length=1)

    @field_validator("anchor_ids")
    @classmethod
    def _anchor_ids_validate(cls, value: list[str]) -> list[str]:
        return _validate_anchor_ids(value)


class SemanticPricingDefinition(BaseModel):
    model_config = ConfigDict(extra="forbid")

    defined_term: str
    definition_summary: str
    source_refs: list[str] = Field(min_length=1)

    @field_validator("source_refs")
    @classmethod
    def _source_refs_validate(cls, value: list[str]) -> list[str]:
        return _validate_anchor_ids(value)


class SemanticPricingAdjustmentCandidate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    adjustment_anchor_id: str
    label: str
    affects_kind: Literal[
        "applicable_margin",
        "applicable_rate",
        "facility_fee",
        "commitment_fee",
        "lc_fee",
        "l_c_fee",
        "fronting_fee",
        "default_rate",
        "other_pricing",
        # Common model slips / aliases (normalized in validator)
        "fee",
        "margin",
    ]
    delta_bps: float | None = None
    applies_when_text: str | None = None
    affects: str | None = None
    source_refs: list[str] = Field(min_length=1)

    @field_validator("adjustment_anchor_id")
    @classmethod
    def _adj_anchor_validate(cls, value: str) -> str:
        value = str(value or "").strip()
        if not value:
            raise ValueError("adjustment_anchor_id is required")
        import re

        if not re.match(_ANCHOR_ID_RE, value):
            raise ValueError(f"Invalid adjustment_anchor_id: {value!r}")
        return value

    @field_validator("source_refs")
    @classmethod
    def _source_refs_validate(cls, value: list[str]) -> list[str]:
        return _validate_anchor_ids(value)

    @field_validator("affects_kind")
    @classmethod
    def _affects_kind_normalize(cls, value: str) -> str:
        # Allow common underscored variants from LLM output, but normalize to canonical values.
        if value == "l_c_fee":
            return "lc_fee"
        if value == "fee":
            return "other_pricing"
        if value == "margin":
            return "applicable_margin"
        return value


Axis = Literal["rows", "columns", "unknown"]
RateKind = Literal["loan", "fee", "unknown"]
Confidence = Literal["high", "medium", "low"]


class SemanticPricingTableCandidate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    table_anchor_id: str
    confidence: Confidence
    rationale: str

    tiers_axis: Axis = "unknown"
    rate_options_axis: Axis = "unknown"
    rate_option_kind: RateKind = "unknown"
    fee_basis: str | None = None
    defined_term: str | None = None
    tier_metric_label_raw: str | None = None
    regime_hint_anchor_id: str | None = None

    source_refs: list[str] = Field(min_length=1)

    @field_validator("table_anchor_id")
    @classmethod
    def _table_anchor_validate(cls, value: str) -> str:
        value = str(value or "").strip()
        if not value:
            raise ValueError("table_anchor_id is required")
        import re

        if not re.match(_ANCHOR_ID_RE, value):
            raise ValueError(f"Invalid table_anchor_id: {value!r}")
        return value

    @field_validator("source_refs")
    @classmethod
    def _source_refs_validate(cls, value: list[str]) -> list[str]:
        return _validate_anchor_ids(value)

    @field_validator("rate_option_kind", mode="before")
    @classmethod
    def _rate_option_kind_normalize(cls, value: str) -> str:
        """Normalize common LLM-emitted variants into the small RateKind enum.

        This keeps downstream logic deterministic while allowing the scan model to be expressive.
        """

        if not isinstance(value, str):
            return value
        v = value.strip().lower()

        # Explicit aliases seen in practice.
        if v in {"facility_fee", "commitment_fee", "lc_fee", "l_c_fee", "fronting_fee", "default_rate"}:
            return "fee"
        if v in {"margin", "applicable_margin", "loan_margin"}:
            return "loan"

        # Soft normalization for other fee-ish variants (e.g., "letter_of_credit_fee").
        if "fee" in v:
            return "fee"

        return value

    @field_validator("rate_options_axis")
    @classmethod
    def _axes_validate(cls, value: Axis, info) -> Axis:  # type: ignore[override]
        tiers_axis = info.data.get("tiers_axis")
        if tiers_axis in ("rows", "columns") and value in ("rows", "columns") and tiers_axis == value:
            raise ValueError("tiers_axis and rate_options_axis cannot be the same when both are known")
        return value


class SemanticPricingScanChunkV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["semantic_pricing_scan_chunk_v1"]
    item_id: str
    chunk: SemanticScanChunkRef

    # Table candidates observed in this chunk (only for tables the model is confident are pricing related).
    pricing_table_candidates: list[SemanticPricingTableCandidate] = Field(default_factory=list)

    # Explicit MARGIN/FEE/GRID adjustments in prose/footnotes that should be encoded downstream as PricingAdjustment.
    # This is intentionally narrow: do NOT include benchmark-index definitions (e.g., "Adjusted Term SOFR Rate").
    required_pricing_adjustments: list[SemanticPricingAdjustmentCandidate] = Field(default_factory=list)

    # Key pricing definitions (Applicable Margin, Commitment Fee, etc.) when present.
    definitions: list[SemanticPricingDefinition] = Field(default_factory=list)


class SemanticPricingScanSummaryV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["semantic_pricing_scan_summary_v1"]
    item_id: str

    n_anchors: int
    n_chunks: int
    covered_anchor_ids: list[str] = Field(min_length=1)

    pricing_table_candidates: list[SemanticPricingTableCandidate] = Field(default_factory=list)
    required_pricing_adjustments: list[SemanticPricingAdjustmentCandidate] = Field(default_factory=list)
    definitions: list[SemanticPricingDefinition] = Field(default_factory=list)

    @field_validator("covered_anchor_ids")
    @classmethod
    def _covered_validate(cls, value: list[str]) -> list[str]:
        return _validate_anchor_ids(value)
