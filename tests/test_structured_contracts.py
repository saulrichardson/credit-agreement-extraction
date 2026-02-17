from __future__ import annotations

import pytest

from pipeline.contracts.common import StructuredValidationContext
from pipeline.contracts.covenant_simple_v2 import validate as validate_covenant
from pipeline.contracts.pricing_structured_v2 import validate as validate_pricing


def _ctx(*, allowed: set[str], tags: dict[str, set[str]] | None = None, snippets: dict[str, str] | None = None):
    return StructuredValidationContext(
        item_id="TEST_ITEM",
        allowed_anchor_ids=set(allowed),
        anchor_tags=tags or {},
        snippet_text_by_anchor=snippets or {},
    )


def test_covenant_simple_v2_valid_payload_passes():
    payload = {
        "schema_version": "covenant_simple_v2",
        "covenants": [
            {
                "covenant_id": "COV-001",
                "name": "Interest Coverage Ratio",
                "metric_refs": ["MET-001"],
                "comparator": ">=",
                "limit": 3.5,
                "limit_text": "minimum Interest Coverage Ratio of 3.50 to 1.00",
                "start_date": None,
                "end_date": None,
                "source_refs": ["A0001", "A0002"],
                "notes": "",
            }
        ],
        "metrics": [
            {
                "metric_id": "MET-001",
                "name": "Interest Coverage Ratio",
                "contract_term": "Interest Coverage Ratio",
                "search_tokens": ["Interest Coverage Ratio", "coverage ratio", "3.50 to 1.00"],
                "source_refs": ["A0001"],
                "notes": "",
            }
        ],
        "notes": "",
    }
    ctx = _ctx(allowed={"A0001", "A0002"})
    out = validate_covenant(payload, ctx=ctx)
    assert out["schema_version"] == "covenant_simple_v2"
    assert out["covenants"][0]["metric_refs"] == ["MET-001"]


def test_covenant_simple_v2_rejects_unknown_metric_refs():
    payload = {
        "schema_version": "covenant_simple_v2",
        "covenants": [
            {
                "covenant_id": "COV-001",
                "name": "Interest Coverage Ratio",
                "metric_refs": ["MET-999"],
                "comparator": ">=",
                "limit": 3.5,
                "limit_text": "minimum Interest Coverage Ratio of 3.50 to 1.00",
                "start_date": None,
                "end_date": None,
                "source_refs": ["A0001"],
                "notes": "",
            }
        ],
        "metrics": [],
        "notes": "",
    }
    ctx = _ctx(allowed={"A0001"})
    with pytest.raises(ValueError, match="unknown metric_id"):
        validate_covenant(payload, ctx=ctx)


def test_covenant_simple_v2_rejects_source_refs_not_in_pack():
    payload = {
        "schema_version": "covenant_simple_v2",
        "covenants": [],
        "metrics": [
            {
                "metric_id": "MET-001",
                "name": "Interest Coverage Ratio",
                "contract_term": "Interest Coverage Ratio",
                "search_tokens": [],
                "source_refs": ["A9999"],
                "notes": "",
            }
        ],
        "notes": "",
    }
    ctx = _ctx(allowed={"A0001"})
    with pytest.raises(ValueError, match="not present in the provided snippet pack"):
        validate_covenant(payload, ctx=ctx)


def test_pricing_structured_v2_minimal_valid_payload_passes():
    payload = {
        "schema_version": "pricing_structured_v2",
        "issuer": "ACME Corp.",
        "agreement": {"title": "", "section_refs": [], "currency": "USD", "as_of_date": ""},
        "metrics": [
            {
                "name": "Consolidated Leverage Ratio",
                "contract_term": "Consolidated Leverage Ratio",
                "search_tokens": ["Consolidated Leverage Ratio", "Leverage Ratio", "ratio"],
                "description": "",
                "formula": "",
                "requires_non_compustat": True,
                "source_refs": ["A0001"],
            }
        ],
        "tier_schemes": [
            {
                "scheme_id": "SCHEME-1",
                "classification": "financial_metrics",
                "tiers": [
                    {
                        "tier_id": "Tier1",
                        "display": "<= 3.00:1.00",
                        "conditions": [{"metric": "Consolidated Leverage Ratio", "op": "<=", "value": 3.0}],
                        "source_refs": ["A0001"],
                    }
                ],
                "source_refs": ["A0001"],
            }
        ],
        "facilities": [
            {
                "facility_id": "revolver",
                "committed_amount": None,
                "currency": "USD",
                "maturity_date": None,
                "rate_type": "floating",
                "rates": [
                    {
                        "rate_id": "R1",
                        "type": "margin",
                        "tier_scheme_ref": "SCHEME-1",
                        "units": "bps",
                        "by_level": [{"tier_id": "Tier1", "bps": 175.0}],
                        "source_refs": ["A0001"],
                        "base_rate": "ABR",
                        "base_rate_type": "prime",
                        "base_rate_contract_term": "ABR",
                        "base_rate_search_tokens": ["ABR", "Base Rate", "prime rate"],
                    }
                ],
                "source_refs": ["A0001"],
            }
        ],
        "overrides": [],
    }
    ctx = _ctx(
        allowed={"A0001", "A0002"},
        tags={"A0001": {"pricing"}, "A0002": {"financial_covenant"}},
        snippets={"A0001": "Applicable Margin: 1.75% (175 bps) for ABR Loans"},
    )
    out = validate_pricing(payload, ctx=ctx)
    assert out["schema_version"] == "pricing_structured_v2"
    assert out["facilities"][0]["rates"][0]["base_rate_type"] == "prime"


def test_pricing_structured_v2_rejects_margin_missing_base_rate_helpers():
    payload = {
        "schema_version": "pricing_structured_v2",
        "issuer": "",
        "agreement": {"title": "", "section_refs": [], "currency": "", "as_of_date": ""},
        "metrics": [],
        "tier_schemes": [
            {
                "scheme_id": "SCHEME-1",
                "classification": "flat",
                "tiers": [{"tier_id": "Tier1", "display": "flat", "conditions": None, "source_refs": ["A0001"]}],
                "source_refs": ["A0001"],
            }
        ],
        "facilities": [
            {
                "facility_id": "revolver",
                "committed_amount": None,
                "currency": "USD",
                "maturity_date": None,
                "rate_type": "floating",
                "rates": [
                    {
                        "rate_id": "R1",
                        "type": "margin",
                        "tier_scheme_ref": "SCHEME-1",
                        "units": "bps",
                        "by_level": [{"tier_id": "Tier1", "bps": 100.0}],
                        "source_refs": ["A0001"],
                    }
                ],
                "source_refs": ["A0001"],
            }
        ],
        "overrides": [],
    }
    ctx = _ctx(
        allowed={"A0001"},
        tags={"A0001": {"pricing"}},
        snippets={"A0001": "Applicable Margin 1.00%"},
    )
    with pytest.raises(ValueError, match="missing required base-rate helper fields"):
        validate_pricing(payload, ctx=ctx)


def test_pricing_structured_v2_rejects_financial_covenant_only_sources():
    payload = {
        "schema_version": "pricing_structured_v2",
        "issuer": "",
        "agreement": {"title": "", "section_refs": [], "currency": "", "as_of_date": ""},
        "metrics": [
            {
                "name": "Consolidated Leverage Ratio",
                "contract_term": "Consolidated Leverage Ratio",
                "search_tokens": [],
                "description": "",
                "formula": "",
                "requires_non_compustat": True,
                "source_refs": ["A0002"],
            }
        ],
        "tier_schemes": [],
        "facilities": [],
        "overrides": [],
    }
    ctx = _ctx(
        allowed={"A0001", "A0002"},
        tags={"A0001": {"pricing"}, "A0002": {"financial_covenant"}},
        snippets={"A0002": "Financial covenant definition..."},
    )
    with pytest.raises(ValueError, match="may not cite anchors labeled only as financial_covenant"):
        validate_pricing(payload, ctx=ctx)

