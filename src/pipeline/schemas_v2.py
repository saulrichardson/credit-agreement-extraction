from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


ANCHOR_ID_RE = re.compile(r"^A\d{4,}$")


def _validate_anchor_id(value: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"anchor ID must be a string; got {type(value).__name__}")
    anchor_id = value.strip()
    if not ANCHOR_ID_RE.fullmatch(anchor_id):
        raise ValueError(f"invalid anchor_id {anchor_id!r} (expected like 'A0001')")
    return anchor_id


def _validate_anchor_ids(anchor_ids: list[str]) -> list[str]:
    if not isinstance(anchor_ids, list):
        raise TypeError("anchor ids must be a JSON array")
    seen: set[str] = set()
    cleaned: list[str] = []
    for idx, raw in enumerate(anchor_ids):
        anchor_id = _validate_anchor_id(raw)
        if anchor_id in seen:
            raise ValueError(f"duplicate anchor_id {anchor_id!r} within the same bucket")
        seen.add(anchor_id)
        cleaned.append(anchor_id)
    return cleaned


class AnchorRangeV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    start_anchor: str
    end_anchor: str

    @field_validator("start_anchor", "end_anchor")
    @classmethod
    def _validate_anchor_id(cls, value: str) -> str:
        return _validate_anchor_id(value)


class IndexingSelectionV2(BaseModel):
    """Strict schema for v2 indexing responses (selection + metadata per anchor)."""

    model_config = ConfigDict(extra="forbid")

    metadata_anchors: list[str] = Field(default_factory=list)
    agreement_date_anchors: list[str] = Field(
        default_factory=list,
        description=(
            "Anchors that explicitly state or define agreement-level key dates such as Closing Date, "
            "Effective Date / Restatement Effective Date / Amendment Effective Date, including preamble "
            '"dated as of" lines and definitions like "Closing Date means ...". '
            "This bucket is intended to be a small, targeted input for facility fundamentals extraction."
        ),
    )
    fundamental_anchors: list[str] = Field(default_factory=list)
    key_date_definitions_anchors: list[str] = Field(
        default_factory=list,
        description=(
            "Optional: anchors that explicitly define key facility date terms (e.g., 'Maturity Date means ...', "
            "'Termination Date means ...', 'Revolving Credit Commitment Termination Date means ...'). "
            "These are often located inside the definitions section; this bucket exists so downstream "
            "fundamentals extraction can include the relevant definitions without ingesting the entire definitions span."
        ),
    )
    pricing_anchors: list[str] = Field(default_factory=list)
    base_rate_anchors: list[str] = Field(
        default_factory=list,
        description=(
            "Pricing-kernel subset: anchors that define benchmark/base-rate definitions and the mechanics needed "
            "to compute them (ABR/Base Rate, Adjusted LIBOR, Term SOFR, rounding, reserve adjustments, etc.)."
        ),
    )
    spread_anchors: list[str] = Field(
        default_factory=list,
        description=(
            "Pricing-kernel subset: anchors that define pricing spreads/margins (Applicable Margin grids/tables, "
            "performance pricing, tiers) and the definitions/mechanics needed to apply them."
        ),
    )
    fee_anchors: list[str] = Field(
        default_factory=list,
        description=(
            "Pricing-kernel subset: anchors that define borrower-paid fee rates/schedules (commitment/utilization/"
            "LC/fronting/facility fees) and the definitions/mechanics needed to apply them."
        ),
    )
    financial_covenant_anchors: list[str] = Field(default_factory=list)
    definitions_anchor_range: AnchorRangeV1 | None = Field(
        default=None,
        description=(
            "If the agreement contains a dedicated definitions/glossary section, optionally provide a contiguous "
            "range of anchors that covers it. Null if not applicable."
        ),
    )

    @field_validator(
        "metadata_anchors",
        "agreement_date_anchors",
        "fundamental_anchors",
        "key_date_definitions_anchors",
        "pricing_anchors",
        "base_rate_anchors",
        "spread_anchors",
        "fee_anchors",
        "financial_covenant_anchors",
    )
    @classmethod
    def _validate_anchor_ids(cls, value: list[str]) -> list[str]:
        return _validate_anchor_ids(value)


class IndexingSelectionV2Artifact(BaseModel):
    """Pipeline artifact persisted under runs/<run_id>/indexing_v2/<item_id>_anchors.json."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["indexing_selection_v2"]
    stage: Literal["indexing_v2"]
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
    selection: IndexingSelectionV2

    auto_added_anchors: list[dict] = Field(
        default_factory=list,
        description=(
            "Deterministic post-processing additions applied after the model selection. "
            "Each entry is an object {anchor_id, bucket, reason}."
        ),
    )

    @field_validator("auto_added_anchors")
    @classmethod
    def _auto_added_validate(cls, value: list[dict]) -> list[dict]:
        if not isinstance(value, list):
            raise TypeError("auto_added_anchors must be a JSON array")
        cleaned: list[dict] = []
        allowed_buckets = {
            "metadata",
            "agreement_dates",
            "fundamental",
            "key_date_definitions",
            "pricing",
            "base_rate",
            "spread",
            "fee",
            "definitions",
            "financial_covenant",
        }
        for idx, raw in enumerate(value):
            if not isinstance(raw, dict):
                raise TypeError(f"auto_added_anchors[{idx}] must be an object")
            anchor_id = raw.get("anchor_id")
            bucket = raw.get("bucket")
            reason = raw.get("reason")
            if not isinstance(anchor_id, str) or not ANCHOR_ID_RE.fullmatch(anchor_id.strip()):
                raise ValueError(f"auto_added_anchors[{idx}].anchor_id must be like 'A0001'")
            if not isinstance(bucket, str) or bucket.strip() not in allowed_buckets:
                raise ValueError(f"auto_added_anchors[{idx}].bucket must be one of {sorted(allowed_buckets)}")
            if not isinstance(reason, str) or not reason.strip():
                raise ValueError(f"auto_added_anchors[{idx}].reason must be a non-empty string")
            cleaned.append({"anchor_id": anchor_id.strip(), "bucket": bucket.strip(), "reason": reason.strip()})
        return cleaned
