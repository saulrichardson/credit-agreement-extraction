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


def _validate_anchor_ids_nonempty(anchor_ids: list[str]) -> list[str]:
    cleaned = _validate_anchor_ids(anchor_ids)
    if not cleaned:
        raise ValueError("source_refs must contain at least one anchor_id")
    return cleaned


class PricingTablePlan(BaseModel):
    """Semantic hints for compiling a single pricing table."""

    model_config = ConfigDict(extra="forbid")

    table_anchor_id: str
    tiers_axis: Literal["rows", "columns"]
    rate_options_axis: Literal["rows", "columns"]

    # Coarse semantic: does the "rate option" represent a loan-type choice or a fee label?
    rate_option_kind: Literal["loan", "fee"]
    fee_basis: str | None = None

    # Optional: when the label isn't fully present in the table itself (common for 1-row status fee tables),
    # we allow pointing at a defined-term label in neighboring anchors.
    defined_term: str | None = None

    # Optional: verbatim metric label for the tier axis (e.g., "Consolidated Total Leverage Ratio").
    tier_metric_label_raw: str | None = None

    # Optional: anchor id that introduces a distinct regime, if the planner decides this table belongs to it.
    regime_hint_anchor_id: str | None = None

    rationale: str = Field(description="Short explanation grounded in cited anchors.")
    source_refs: list[str] = Field(default_factory=list)

    @field_validator("table_anchor_id")
    @classmethod
    def _table_anchor_id_validate(cls, value: str) -> str:
        if not isinstance(value, str):
            raise TypeError("table_anchor_id must be a string")
        v = value.strip()
        if not ANCHOR_ID_RE.fullmatch(v):
            raise ValueError(f"invalid table_anchor_id {v!r} (expected like 'A0001')")
        return v

    @field_validator("regime_hint_anchor_id")
    @classmethod
    def _regime_hint_anchor_id_validate(cls, value: str | None) -> str | None:
        if value is None:
            return None
        if not isinstance(value, str):
            raise TypeError("regime_hint_anchor_id must be a string or null")
        v = value.strip()
        if not ANCHOR_ID_RE.fullmatch(v):
            raise ValueError(f"invalid regime_hint_anchor_id {v!r} (expected like 'A0001')")
        return v

    @field_validator("source_refs")
    @classmethod
    def _source_refs_validate(cls, value: list[str]) -> list[str]:
        return _validate_anchor_ids_nonempty(value)

    @field_validator("rate_options_axis")
    @classmethod
    def _axes_not_equal(cls, value: str, info):  # type: ignore[override]
        # Pydantic doesn't provide sibling values here reliably across versions; enforce in plan validator below.
        return value


class ContractPricingPlanV2(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["contract_pricing_plan_v2"]
    item_id: str
    selected_tables: list[PricingTablePlan] = Field(default_factory=list)

    @field_validator("selected_tables")
    @classmethod
    def _validate_selected_tables(cls, value: list[PricingTablePlan]) -> list[PricingTablePlan]:
        if not value:
            raise ValueError("selected_tables must be non-empty (planner must select at least one pricing table)")
        seen: set[str] = set()
        for t in value:
            if t.table_anchor_id in seen:
                raise ValueError(f"duplicate table_anchor_id in selected_tables: {t.table_anchor_id}")
            seen.add(t.table_anchor_id)
            if t.tiers_axis == t.rate_options_axis:
                raise ValueError(
                    f"invalid axis plan for {t.table_anchor_id}: tiers_axis and rate_options_axis must differ"
                )
        return value
