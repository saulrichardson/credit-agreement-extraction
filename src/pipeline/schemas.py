from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

ANCHOR_ID_RE = re.compile(r"^A\d{4,}$")


class IndexingSelection(BaseModel):
    """Strict schema for the model's indexing response (selection-only).

    The model must output JSON with exactly these keys and arrays of anchor IDs:
      - fundamental_anchors
      - pricing_anchors
      - financial_covenant_anchors

    Anchor IDs are expected to match the normalization format: "A0001", "A0123", ...
    """

    model_config = ConfigDict(extra="forbid")

    fundamental_anchors: list[str] = Field(description="Anchor IDs expressing core deal structure.")
    pricing_anchors: list[str] = Field(description="Anchor IDs expressing pricing/economics.")
    financial_covenant_anchors: list[str] = Field(description="Anchor IDs expressing financial covenants.")

    @field_validator(
        "fundamental_anchors",
        "pricing_anchors",
        "financial_covenant_anchors",
    )
    @classmethod
    def _validate_anchor_ids(cls, value: list[str]) -> list[str]:
        if not isinstance(value, list):
            raise TypeError("must be a JSON array")

        seen: set[str] = set()
        cleaned: list[str] = []
        for idx, raw in enumerate(value):
            if not isinstance(raw, str):
                raise TypeError(f"anchor IDs must be strings; got {type(raw).__name__} at index {idx}")
            anchor_id = raw.strip()
            if not ANCHOR_ID_RE.fullmatch(anchor_id):
                raise ValueError(f"invalid anchor_id {anchor_id!r} (expected like 'A0001')")
            if anchor_id in seen:
                raise ValueError(f"duplicate anchor_id {anchor_id!r} within the same bucket")
            seen.add(anchor_id)
            cleaned.append(anchor_id)
        return cleaned


class IndexingSelectionArtifact(BaseModel):
    """Pipeline artifact persisted under runs/<run_id>/indexing/<item_id>_anchors.json."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["indexing_selection_v1"]
    stage: Literal["indexing"]
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
    selection: IndexingSelection
