from __future__ import annotations

import pytest

from pipeline.compile.definitions_compiler_v1 import _validate_compiler_output_v2_ast


def test_definitions_compiler_rejects_source_refs_not_in_contexts():
    raw_json = {
        "schema_version": "definition_compiler_v2_ast_v1",
        "definitions": [
            {
                "name": "Fixed Charge Coverage Ratio",
                "contract_term": None,
                "definition_verbatim": None,
                "expression_ast": None,
                "input_terms": [],
                "clauses": [],
                "source_refs": ["A0002"],
                "needs_more_context": False,
                "confidence": "low",
                "notes": [],
            }
        ],
        "unresolved_dependencies": [],
    }

    with pytest.raises(RuntimeError, match="not present in the provided CONTEXT BLOCKS"):
        _validate_compiler_output_v2_ast(
            item_id="contract_1",
            metric_name="Fixed Charge Coverage Ratio",
            raw_json=raw_json,
            allowed_anchor_ids={"A0001"},
            contexts_text="[[A0001]]\nFixed Charge Coverage Ratio means ...\n",
        )


def test_definitions_compiler_rejects_expression_ast_terms_not_listed_in_input_terms():
    raw_json = {
        "schema_version": "definition_compiler_v2_ast_v1",
        "definitions": [
            {
                "name": "Debt/Cash Flow Ratio",
                "contract_term": "Debt/Cash Flow Ratio",
                "definition_verbatim": "\"Debt/Cash Flow Ratio\" means the ratio of (i) Consolidated Funded Debt to (ii) Consolidated Cash Flow",
                "expression_ast": {
                    "type": "op",
                    "op": "/",
                    "args": [
                        {"type": "term", "term": "Consolidated Funded Debt"},
                        {"type": "term", "term": "Consolidated Cash Flow"},
                    ],
                },
                "input_terms": [],
                "clauses": [],
                "source_refs": ["A0001"],
                "needs_more_context": False,
                "confidence": "high",
                "notes": [],
            }
        ],
        "unresolved_dependencies": [],
    }

    with pytest.raises(RuntimeError, match=r"must be present in input_terms\[\]"):
        _validate_compiler_output_v2_ast(
            item_id="contract_1",
            metric_name="Debt/Cash Flow Ratio",
            raw_json=raw_json,
            allowed_anchor_ids={"A0001"},
            contexts_text="[[A0001]]\n"
            "\"Debt/Cash Flow Ratio\" means the ratio of (i) Consolidated Funded Debt to (ii) Consolidated Cash Flow\n",
        )


def test_definitions_compiler_does_not_repair_input_terms():
    raw_json = {
        "schema_version": "definition_compiler_v2_ast_v1",
        "definitions": [
            {
                "name": "Fixed Charge Coverage Ratio",
                "contract_term": "Fixed Charge Coverage Ratio",
                "definition_verbatim": "\"Fixed Charge Coverage Ratio\" means the ratio of a) EBITDA to b) Fixed Charges",
                "expression_ast": None,
                "input_terms": ["a) EBITDA", "b) Fixed Charges"],
                "clauses": [],
                "source_refs": ["A0001"],
                "needs_more_context": False,
                "confidence": "high",
                "notes": [],
            }
        ],
        "unresolved_dependencies": [
            {"term": "a) EBITDA", "referenced_by": ["Fixed Charge Coverage Ratio"]},
            {"term": "b) Fixed Charges", "referenced_by": ["Fixed Charge Coverage Ratio"]},
        ],
    }

    cleaned, _issues = _validate_compiler_output_v2_ast(
        item_id="contract_1",
        metric_name="Fixed Charge Coverage Ratio",
        raw_json=raw_json,
        allowed_anchor_ids={"A0001"},
        contexts_text="[[A0001]]\n"
        "\"Fixed Charge Coverage Ratio\" means the ratio of a) EBITDA to b) Fixed Charges\n",
    )

    m = cleaned["definitions"][0]
    assert m["input_terms"] == ["a) EBITDA", "b) Fixed Charges"]
