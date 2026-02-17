from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from pipeline.contracts.common import (
    StructuredValidationContext,
    normalize_ws,
    validate_anchor_id_list,
    validate_trimmed_string_list,
)


COMPARATOR_VALUES = ("<=", ">=", "=", "between")


class CovenantSimpleMetricV2(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    metric_id: str
    name: str
    contract_term: str
    search_tokens: list[str]
    source_refs: list[str]
    notes: str = ""

    @field_validator("metric_id", "name", "contract_term", "notes")
    @classmethod
    def _trim_strings(cls, value: str) -> str:
        return normalize_ws(value)

    @field_validator("search_tokens")
    @classmethod
    def _search_tokens(cls, value: Any) -> list[str]:
        # Prompt asks for 3-6 tokens, but explicitly allows [] when evidence is unavailable.
        return validate_trimmed_string_list(value, label="metrics[].search_tokens", max_items=6)

    @field_validator("source_refs")
    @classmethod
    def _source_refs(cls, value: Any) -> list[str]:
        return validate_anchor_id_list(value, allow_empty=False, label="metrics[].source_refs")


class CovenantSimpleRowV2(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    covenant_id: str
    name: str
    metric_refs: list[str]
    comparator: Literal["<=", ">=", "=", "between"] | None = None
    limit: float | None = None
    limit_text: str
    start_date: str | None = None
    end_date: str | None = None
    source_refs: list[str]
    notes: str = ""

    @field_validator("covenant_id", "name", "limit_text", "notes")
    @classmethod
    def _trim_strings(cls, value: str) -> str:
        return normalize_ws(value)

    @field_validator("metric_refs")
    @classmethod
    def _metric_refs(cls, value: Any) -> list[str]:
        if not isinstance(value, list):
            raise TypeError("covenants[].metric_refs must be a JSON array")
        cleaned: list[str] = []
        seen: set[str] = set()
        for idx, raw in enumerate(value):
            if not isinstance(raw, str):
                raise TypeError(f"covenants[].metric_refs[{idx}] must be a string")
            ref = raw.strip()
            if not ref:
                continue
            if ref in seen:
                continue
            seen.add(ref)
            cleaned.append(ref)
        if not cleaned:
            raise ValueError("covenants[].metric_refs must be non-empty")
        return cleaned

    @field_validator("source_refs")
    @classmethod
    def _source_refs(cls, value: Any) -> list[str]:
        return validate_anchor_id_list(value, allow_empty=False, label="covenants[].source_refs")

    @field_validator("start_date", "end_date")
    @classmethod
    def _dates(cls, value: str | None) -> str | None:
        if value is None:
            return None
        s = normalize_ws(value)
        # Prompt contract: "MM/DD/YYYY" format only.
        if len(s) != 10 or s[2] != "/" or s[5] != "/":
            raise ValueError(f"expected MM/DD/YYYY date string or null, got {value!r}")
        return s


class CovenantSimpleV2(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    schema_version: Literal["covenant_simple_v2"]
    covenants: list[CovenantSimpleRowV2] = Field(default_factory=list)
    metrics: list[CovenantSimpleMetricV2] = Field(default_factory=list)
    notes: str = ""

    @field_validator("notes")
    @classmethod
    def _notes(cls, value: str) -> str:
        return normalize_ws(value)


def empty_payload(*, reason: str) -> dict[str, Any]:
    return {
        "schema_version": "covenant_simple_v2",
        "covenants": [],
        "metrics": [],
        "notes": reason.strip(),
    }


def validate(payload: object, *, ctx: StructuredValidationContext) -> dict[str, Any]:
    """Validate covenant_simple_v2 payload (schema + semantic checks)."""

    try:
        doc = CovenantSimpleV2.model_validate(payload)
    except Exception as exc:
        raise ValueError(f"schema validation failed: {exc}") from exc

    # Enforce unique IDs and resolvable references.
    metric_ids = [m.metric_id for m in doc.metrics]
    if len(set(metric_ids)) != len(metric_ids):
        raise ValueError("duplicate metric_id values found in metrics[]")
    covenant_ids = [c.covenant_id for c in doc.covenants]
    if len(set(covenant_ids)) != len(covenant_ids):
        raise ValueError("duplicate covenant_id values found in covenants[]")

    metric_id_set = set(metric_ids)
    missing_metric_refs: set[str] = set()
    for cov in doc.covenants:
        for ref in cov.metric_refs:
            if ref not in metric_id_set:
                missing_metric_refs.add(ref)
    if missing_metric_refs:
        raise ValueError(f"covenants[].metric_refs contains unknown metric_id(s): {sorted(missing_metric_refs)}")

    # Anchor provenance: every source_ref must be present in the snippet pack used for this call.
    bad_source_refs: set[str] = set()
    for cov in doc.covenants:
        for aid in cov.source_refs:
            if aid not in ctx.allowed_anchor_ids:
                bad_source_refs.add(aid)
    for met in doc.metrics:
        for aid in met.source_refs:
            if aid not in ctx.allowed_anchor_ids:
                bad_source_refs.add(aid)
    if bad_source_refs:
        examples = ", ".join(sorted(bad_source_refs)[:10])
        raise ValueError(
            "source_refs contains anchors not present in the provided snippet pack "
            f"(count={len(bad_source_refs)}; examples={examples})"
        )

    # Comparator value is already constrained by Literal when non-null.
    # Limit/limit_text: prompt allows limit=null when qualitative. We accept both.
    return doc.model_dump(mode="json")


def retry_prompt(base_prompt: str, attempt: int, error: str, previous_output: str) -> str:
    _ = previous_output
    if attempt == 2:
        rules = (
            "Fix your previous response. Output MUST be valid JSON ONLY (no prose, no markdown).\n"
            "Return exactly this top-level shape:\n"
            "{\n"
            '  \"schema_version\": \"covenant_simple_v2\",\n'
            '  \"covenants\": [ ... ],\n'
            '  \"metrics\": [ ... ],\n'
            '  \"notes\": \"\"\n'
            "}\n"
            "- Do not add extra top-level keys.\n"
            "- Every covenant/metric MUST have non-empty source_refs with real anchor IDs from the input.\n"
            "- start_date/end_date MUST be null or \"MM/DD/YYYY\".\n"
        )
    else:
        rules = (
            "STRICT MODE (final attempt):\n"
            "- Return JSON ONLY.\n"
            "- Top-level keys MUST be exactly: schema_version, covenants, metrics, notes.\n"
            "- comparator MUST be one of: <=, >=, =, between, or null.\n"
            "- covenant.metric_refs MUST reference metric_id values present in metrics[].\n"
            "- No extra keys anywhere.\n"
        )
    return f"{base_prompt}\n\n=== RETRY REQUIRED ===\nError: {error}\n\n{rules}"

