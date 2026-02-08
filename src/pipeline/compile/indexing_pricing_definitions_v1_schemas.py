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
            raise ValueError(f"duplicate anchor_id {anchor_id!r} within the same metric")
        seen.add(anchor_id)
        cleaned.append(anchor_id)
    return cleaned


class PricingMetricInputV1(BaseModel):
    """A compact record of the metrics we asked the indexer to locate definitions for."""

    model_config = ConfigDict(extra="forbid")

    name: str
    description: str | None = None


class PricingDefinitionsIndexingMetricV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    contract_term: str
    definition_anchor_ids: list[str] = Field(default_factory=list)
    confidence: Literal["high", "medium", "low"]
    notes: list[str] = Field(default_factory=list)

    @field_validator("definition_anchor_ids")
    @classmethod
    def _validate_anchor_ids(cls, value: list[str]) -> list[str]:
        return _validate_anchor_ids(value)


class PricingDefinitionsIndexingSelectionV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["pricing_definitions_indexing_v1"]
    metrics: list[PricingDefinitionsIndexingMetricV1]


class PricingDefinitionsIndexingArtifactV1(BaseModel):
    """Artifact persisted under runs/<run_id>/indexing_pricing_definitions_v1/."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["pricing_definitions_indexing_v1_artifact"]
    stage: Literal["indexing_pricing_definitions_v1"]
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
    pricing_second_pass_path: str
    metrics_in: list[PricingMetricInputV1]

    selection: PricingDefinitionsIndexingSelectionV1
