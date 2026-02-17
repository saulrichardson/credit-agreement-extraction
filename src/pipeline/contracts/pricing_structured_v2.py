from __future__ import annotations

import re
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from pipeline.contracts.common import (
    StructuredValidationContext,
    normalize_ws,
    validate_anchor_id_list,
    validate_trimmed_string_list,
)


ISO_DATE_RE = re.compile(r"^\\d{4}-\\d{2}-\\d{2}$")


def _validate_iso_date_or_empty(value: str) -> str:
    s = normalize_ws(value)
    if not s:
        return ""
    if not ISO_DATE_RE.fullmatch(s):
        raise ValueError(f"expected ISO date YYYY-MM-DD or empty string, got {value!r}")
    return s


class PricingAgreementV2(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    title: str
    section_refs: list[str]
    currency: str
    as_of_date: str

    @field_validator("title", "currency")
    @classmethod
    def _trim(cls, value: str) -> str:
        return normalize_ws(value)

    @field_validator("as_of_date")
    @classmethod
    def _as_of_date(cls, value: str) -> str:
        return _validate_iso_date_or_empty(value)

    @field_validator("section_refs")
    @classmethod
    def _section_refs(cls, value: Any) -> list[str]:
        return validate_trimmed_string_list(value, label="agreement.section_refs", max_items=50)


class PricingMetricV2(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    name: str
    contract_term: str
    search_tokens: list[str]
    description: str
    formula: str
    requires_non_compustat: bool
    source_refs: list[str]

    @field_validator("name", "contract_term", "description", "formula")
    @classmethod
    def _trim(cls, value: str) -> str:
        return normalize_ws(value)

    @field_validator("search_tokens")
    @classmethod
    def _search_tokens(cls, value: Any) -> list[str]:
        # Prompt: 3-6 tokens, but allows [] when not available.
        return validate_trimmed_string_list(value, label="metrics[].search_tokens", max_items=6)

    @field_validator("source_refs")
    @classmethod
    def _source_refs(cls, value: Any) -> list[str]:
        return validate_anchor_id_list(value, allow_empty=False, label="metrics[].source_refs")


class TierConditionAtomV1(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    metric: str
    op: Literal["<=", "<", ">=", ">"]
    value: float

    @field_validator("metric")
    @classmethod
    def _metric(cls, value: str) -> str:
        return normalize_ws(value)


class PricingTierV2(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    tier_id: str
    display: str
    conditions: list[TierConditionAtomV1] | None = None
    source_refs: list[str]

    @field_validator("tier_id", "display")
    @classmethod
    def _trim(cls, value: str) -> str:
        return normalize_ws(value)

    @field_validator("source_refs")
    @classmethod
    def _source_refs(cls, value: Any) -> list[str]:
        return validate_anchor_id_list(value, allow_empty=False, label="tier_schemes[].tiers[].source_refs")


class PricingTierSchemeV2(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    scheme_id: str
    classification: Literal["ratings_based", "financial_metrics", "flat"]
    tiers: list[PricingTierV2]
    source_refs: list[str]

    @field_validator("scheme_id")
    @classmethod
    def _scheme_id(cls, value: str) -> str:
        return normalize_ws(value)

    @field_validator("tiers")
    @classmethod
    def _tiers_non_empty(cls, value: list[PricingTierV2]) -> list[PricingTierV2]:
        if not value:
            raise ValueError("tier_schemes[].tiers must be non-empty")
        return value

    @field_validator("source_refs")
    @classmethod
    def _source_refs(cls, value: Any) -> list[str]:
        return validate_anchor_id_list(value, allow_empty=False, label="tier_schemes[].source_refs")


class RateByLevelV1(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    tier_id: str
    bps: float

    @field_validator("tier_id")
    @classmethod
    def _tier_id(cls, value: str) -> str:
        return normalize_ws(value)


class RateConditionAtomV1(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    metric: str
    op: Literal["<=", "<", ">=", ">"]
    value: float

    @field_validator("metric")
    @classmethod
    def _metric(cls, value: str) -> str:
        return normalize_ws(value)


class PricingRateV2(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    rate_id: str
    type: Literal["margin", "fee"]
    tier_scheme_ref: str
    units: Literal["bps"]
    by_level: list[RateByLevelV1]
    source_refs: list[str]

    # Optional, only when multi-regime pricing applies.
    condition: RateConditionAtomV1 | None = None

    # Required for margin rates.
    base_rate: str | None = None
    base_rate_type: Literal["sofr", "libor", "prime", "other"] | None = None
    base_rate_contract_term: str | None = None
    base_rate_search_tokens: list[str] | None = None

    # Required for fee rates.
    fee_basis: Literal["undrawn", "letters_of_credit", "facility_commitment", "other"] | None = None
    calculation_basis: Literal["per_annum", "flat"] | None = None

    @field_validator("rate_id", "tier_scheme_ref")
    @classmethod
    def _trim(cls, value: str) -> str:
        return normalize_ws(value)

    @field_validator("by_level")
    @classmethod
    def _by_level_non_empty(cls, value: list[RateByLevelV1]) -> list[RateByLevelV1]:
        if not value:
            raise ValueError("rates[].by_level must be non-empty")
        # Ensure stable de-dupe of tier_id.
        seen: set[str] = set()
        unique: list[RateByLevelV1] = []
        for row in value:
            if row.tier_id in seen:
                continue
            seen.add(row.tier_id)
            unique.append(row)
        return unique

    @field_validator("source_refs")
    @classmethod
    def _source_refs(cls, value: Any) -> list[str]:
        return validate_anchor_id_list(value, allow_empty=False, label="facilities[].rates[].source_refs")

    @field_validator("base_rate", "base_rate_contract_term")
    @classmethod
    def _base_rate_strings(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return normalize_ws(value)

    @field_validator("base_rate_search_tokens")
    @classmethod
    def _base_rate_search_tokens(cls, value: Any) -> list[str] | None:
        if value is None:
            return None
        return validate_trimmed_string_list(value, label="rates[].base_rate_search_tokens", max_items=6)


class PricingFacilityV2(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    facility_id: str
    committed_amount: float | None = None
    currency: str
    maturity_date: str | None = None
    rate_type: Literal["fixed", "floating"]
    rates: list[PricingRateV2]
    source_refs: list[str]

    @field_validator("facility_id", "currency")
    @classmethod
    def _trim(cls, value: str) -> str:
        return normalize_ws(value)

    @field_validator("maturity_date")
    @classmethod
    def _maturity_date(cls, value: str | None) -> str | None:
        if value is None:
            return None
        s = normalize_ws(value)
        # We do not enforce a date format here; agreements vary. Keep it strict "string or null".
        return s

    @field_validator("rates")
    @classmethod
    def _rates_non_empty(cls, value: list[PricingRateV2]) -> list[PricingRateV2]:
        # Prompt requires non-empty when pricing is present, but we enforce at the document level
        # (based on evidence heuristics). Facility-level empty rates are not meaningful.
        if not value:
            raise ValueError("facilities[].rates must be non-empty")
        return value

    @field_validator("source_refs")
    @classmethod
    def _source_refs(cls, value: Any) -> list[str]:
        return validate_anchor_id_list(value, allow_empty=False, label="facilities[].source_refs")


class PricingStructuredV2(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    schema_version: Literal["pricing_structured_v2"]

    issuer: str
    agreement: PricingAgreementV2
    metrics: list[PricingMetricV2] = Field(default_factory=list)
    tier_schemes: list[PricingTierSchemeV2] = Field(default_factory=list)
    facilities: list[PricingFacilityV2] = Field(default_factory=list)
    overrides: list[Any] = Field(default_factory=list)

    @field_validator("issuer")
    @classmethod
    def _issuer(cls, value: str) -> str:
        return normalize_ws(value)

    @field_validator("overrides")
    @classmethod
    def _overrides_currently_disabled(cls, value: Any) -> list[Any]:
        if not isinstance(value, list):
            raise TypeError("overrides must be a JSON array")
        if value:
            raise ValueError(
                "overrides[] is not currently supported (prompt contract does not define its shape). "
                "Return overrides=[] for now."
            )
        return []


def empty_payload(*, reason: str) -> dict[str, Any]:
    # Pricing pass should not silently succeed without evidence; callers decide whether this is allowed.
    _ = reason
    return {
        "schema_version": "pricing_structured_v2",
        "issuer": "",
        "agreement": {"title": "", "section_refs": [], "currency": "", "as_of_date": ""},
        "metrics": [],
        "tier_schemes": [],
        "facilities": [],
        "overrides": [],
    }


_BANNED_METRIC_SUBSTRINGS = (
    "pricing level",
    "applicable pricing level",
    "pricing tier",
    "pricing regime",
)


def _is_disallowed_financial_covenant_anchor(ctx: StructuredValidationContext, anchor_id: str) -> bool:
    tags = {t.lower() for t in ctx.anchor_tags.get(anchor_id, set()) if isinstance(t, str)}
    # The prompt rule is "Do NOT use anchors labeled only (financial_covenant)".
    return bool(tags) and tags.issubset({"financial_covenant"})


def validate(payload: object, *, ctx: StructuredValidationContext) -> dict[str, Any]:
    """Validate pricing_structured_v2 payload (schema + semantic checks)."""

    try:
        doc = PricingStructuredV2.model_validate(payload)
    except Exception as exc:
        raise ValueError(f"schema validation failed: {exc}") from exc

    # Top-level: keep these checks explicit so failures are actionable.
    if not isinstance(doc.issuer, str):
        raise ValueError("issuer must be a string")

    # Metrics: enforce no banned tier/outcome labels as metric names.
    bad_metric_names: list[str] = []
    seen_metric_names: set[str] = set()
    for m in doc.metrics:
        lower = m.name.lower()
        if any(sub in lower for sub in _BANNED_METRIC_SUBSTRINGS):
            bad_metric_names.append(m.name)
        if m.name in seen_metric_names:
            raise ValueError(f"duplicate metrics[].name found: {m.name!r}")
        seen_metric_names.add(m.name)
        for aid in m.source_refs:
            if aid not in ctx.allowed_anchor_ids:
                raise ValueError(f"metrics[].source_refs includes anchor not in snippet pack: {aid}")
            if _is_disallowed_financial_covenant_anchor(ctx, aid):
                raise ValueError(
                    f"pricing metrics may not cite anchors labeled only as financial_covenant (anchor={aid})"
                )
    if bad_metric_names:
        raise ValueError(f"metrics[].name includes banned pricing-tier/outcome label(s): {bad_metric_names}")

    # Tier schemes: IDs unique, tiers unique per scheme, and provenance.
    scheme_ids = [s.scheme_id for s in doc.tier_schemes]
    if len(set(scheme_ids)) != len(scheme_ids):
        raise ValueError("duplicate tier_schemes[].scheme_id values found")

    scheme_by_id = {s.scheme_id: s for s in doc.tier_schemes}
    tier_ids_by_scheme: dict[str, set[str]] = {}
    for scheme in doc.tier_schemes:
        for aid in scheme.source_refs:
            if aid not in ctx.allowed_anchor_ids:
                raise ValueError(f"tier_schemes[].source_refs includes anchor not in snippet pack: {aid}")
            if _is_disallowed_financial_covenant_anchor(ctx, aid):
                raise ValueError(
                    f"pricing tier_schemes may not cite anchors labeled only as financial_covenant (anchor={aid})"
                )
        tier_ids: set[str] = set()
        for tier in scheme.tiers:
            if tier.tier_id in tier_ids:
                raise ValueError(f"duplicate tier_id {tier.tier_id!r} within scheme {scheme.scheme_id!r}")
            tier_ids.add(tier.tier_id)
            for aid in tier.source_refs:
                if aid not in ctx.allowed_anchor_ids:
                    raise ValueError(f"tiers[].source_refs includes anchor not in snippet pack: {aid}")
                if _is_disallowed_financial_covenant_anchor(ctx, aid):
                    raise ValueError(
                        f"pricing tiers may not cite anchors labeled only as financial_covenant (anchor={aid})"
                    )
        tier_ids_by_scheme[scheme.scheme_id] = tier_ids

    # Facilities: IDs unique, all refs resolvable, and margin/fee required helper fields.
    facility_ids = [f.facility_id for f in doc.facilities]
    if len(set(facility_ids)) != len(facility_ids):
        raise ValueError("duplicate facilities[].facility_id values found")

    total_rates = 0
    for fac in doc.facilities:
        for aid in fac.source_refs:
            if aid not in ctx.allowed_anchor_ids:
                raise ValueError(f"facilities[].source_refs includes anchor not in snippet pack: {aid}")
            if _is_disallowed_financial_covenant_anchor(ctx, aid):
                raise ValueError(
                    f"pricing facilities may not cite anchors labeled only as financial_covenant (anchor={aid})"
                )

        for rate in fac.rates:
            total_rates += 1
            if rate.tier_scheme_ref not in scheme_by_id:
                raise ValueError(f"rates[].tier_scheme_ref not found in tier_schemes: {rate.tier_scheme_ref!r}")
            allowed_tier_ids = tier_ids_by_scheme.get(rate.tier_scheme_ref, set())
            for row in rate.by_level:
                if row.tier_id not in allowed_tier_ids:
                    raise ValueError(
                        "rates[].by_level contains tier_id not present in referenced tier scheme "
                        f"(tier_id={row.tier_id!r}, scheme={rate.tier_scheme_ref!r})"
                    )

            for aid in rate.source_refs:
                if aid not in ctx.allowed_anchor_ids:
                    raise ValueError(f"rates[].source_refs includes anchor not in snippet pack: {aid}")
                if _is_disallowed_financial_covenant_anchor(ctx, aid):
                    raise ValueError(
                        f"pricing rates may not cite anchors labeled only as financial_covenant (anchor={aid})"
                    )

            if rate.type == "margin":
                missing = [
                    key
                    for key in (
                        "base_rate",
                        "base_rate_type",
                        "base_rate_contract_term",
                        "base_rate_search_tokens",
                    )
                    if getattr(rate, key) is None
                ]
                if missing:
                    raise ValueError(
                        f"margin rate missing required base-rate helper fields: {missing} (rate_id={rate.rate_id!r})"
                    )
            elif rate.type == "fee":
                if rate.fee_basis is None or rate.calculation_basis is None:
                    raise ValueError(
                        f"fee rate missing required fee fields fee_basis/calculation_basis (rate_id={rate.rate_id!r})"
                    )
            else:
                raise ValueError(f"unknown rate.type: {rate.type!r}")

    # Evidence-based "no empty pricing": if the prompt included pricing-tagged snippets with obvious numeric pricing
    # content, then facilities/rates must not be empty.
    pricing_numeric_evidence = False
    for anchor_id in ctx.allowed_anchor_ids:
        tags = {t.lower() for t in ctx.anchor_tags.get(anchor_id, set()) if isinstance(t, str)}
        if "pricing" not in tags:
            continue
        snippet = (ctx.snippet_text_by_anchor.get(anchor_id) or "").lower()
        if not snippet:
            continue
        has_number = bool(re.search(r"\\b\\d+(?:\\.\\d+)?\\b", snippet))
        has_pricing_keyword = any(
            kw in snippet
            for kw in (
                "applicable margin",
                "commitment fee",
                "facility fee",
                "letter of credit",
                "l/c",
                "basis points",
                "bps",
                "margin",
                "spread",
                "ab r",
                "abr",
                "term sofr",
                "eurocurrency",
            )
        )
        if has_number and has_pricing_keyword:
            pricing_numeric_evidence = True
            break

    if pricing_numeric_evidence and (not doc.facilities or total_rates == 0):
        raise ValueError(
            "pricing evidence appears present in the snippet pack (pricing-tagged snippet has numeric pricing keywords), "
            "but output contains no facilities/rates"
        )

    # Anchor provenance: already checked per-object above. Also ensure no unknown anchors sneak in.
    return doc.model_dump(mode="json")


def retry_prompt(base_prompt: str, attempt: int, error: str, previous_output: str) -> str:
    _ = previous_output
    if attempt == 2:
        rules = (
            "Fix your previous response. Output MUST be valid JSON ONLY (no prose, no markdown).\n"
            "Return exactly this top-level shape:\n"
            "{\n"
            '  \"schema_version\": \"pricing_structured_v2\",\n'
            '  \"issuer\": \"\",\n'
            '  \"agreement\": {\"title\":\"\",\"section_refs\":[],\"currency\":\"\",\"as_of_date\":\"\"},\n'
            '  \"metrics\": [],\n'
            '  \"tier_schemes\": [],\n'
            '  \"facilities\": [],\n'
            '  \"overrides\": []\n'
            "}\n"
            "- Do not add extra top-level keys.\n"
            "- overrides MUST be an empty array.\n"
            "- Every metrics/tier_schemes/tiers/facilities/rates object MUST include non-empty source_refs.\n"
            "- Every margin rate MUST include: base_rate, base_rate_type, base_rate_contract_term, base_rate_search_tokens.\n"
            "- Every fee rate MUST include: fee_basis, calculation_basis.\n"
        )
    else:
        rules = (
            "STRICT MODE (final attempt):\n"
            "- Return JSON ONLY.\n"
            "- Top-level keys MUST be exactly: schema_version, issuer, agreement, metrics, tier_schemes, facilities, overrides.\n"
            "- margin rates MUST have base-rate helper fields populated.\n"
            "- fee rates MUST have fee_basis and calculation_basis.\n"
            "- rates[].tier_scheme_ref MUST match a tier_schemes[].scheme_id.\n"
            "- rates[].by_level[].tier_id MUST match the tier IDs in that scheme.\n"
            "- No extra keys anywhere.\n"
        )
    return f"{base_prompt}\n\n=== RETRY REQUIRED ===\nError: {error}\n\n{rules}"
