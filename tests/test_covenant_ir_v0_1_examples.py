from __future__ import annotations

import importlib.util

from pipeline.ir.covenant_ir_v0_1 import evaluate_covenant, validate_covenant_ir, validate_precision_first_policy


def _lit(typ: str, value: object) -> dict:
    return {"lit": {"type": typ, "value": value}}


def _var(name: str) -> dict:
    return {"var": name}


def _op(op: str, *args: object) -> dict:
    return {"op": op, "args": list(args)}


def _build_a0515_range() -> dict:
    table_id = "t_leverage_ratio_thresholds"
    threshold_expr = _op(
        "lookup_range",
        _lit("string", table_id),
        _var("as_of_date"),
        _lit("string", "lower"),
        _lit("string", "lower_cmp"),
        _lit("string", "upper"),
        _lit("string", "upper_cmp"),
        _lit("string", "max_ratio"),
    )
    return {
        "schema_version": "covenant_ir_v0_1",
        "contract_id": "0000950134-96-000714_4",
        "sources": [
            {"source_id": "s1", "kind": "snippets", "item_id": "0000950134-96-000714_4", "anchor_ids": ["A0515"]},
        ],
        "tables": [
            {
                "table_id": table_id,
                "columns": [
                    {"name": "lower", "type": "date"},
                    {"name": "lower_cmp", "type": "string"},
                    {"name": "upper", "type": "date"},
                    {"name": "upper_cmp", "type": "string"},
                    {"name": "max_ratio", "type": "decimal"},
                ],
                "rows": [
                    {
                        "row_id": "through-1999-06-30",
                        "cells": {"upper": _lit("date", "1999-06-30"), "upper_cmp": _lit("string", "lte"), "max_ratio": _lit("decimal", "5.50")},
                        "source_refs": ["A0515"],
                    },
                    {
                        "row_id": "1999-07-01-through-2000-06-30",
                        "cells": {
                            "lower": _lit("date", "1999-07-01"),
                            "lower_cmp": _lit("string", "gte"),
                            "upper": _lit("date", "2000-06-30"),
                            "upper_cmp": _lit("string", "lte"),
                            "max_ratio": _lit("decimal", "5.00"),
                        },
                        "source_refs": ["A0515"],
                    },
                    {
                        "row_id": "2000-07-01+",
                        "cells": {"lower": _lit("date", "2000-07-01"), "lower_cmp": _lit("string", "gte"), "max_ratio": _lit("decimal", "4.50")},
                        "source_refs": ["A0515"],
                    },
                ],
            }
        ],
        "covenants": [
            {
                "covenant_id": "leverage_ratio",
                "title": "Leverage Ratio",
                "test": {
                    "args": [{"name": "as_of_date", "type": "date"}, {"name": "ratio", "type": "decimal"}],
                    "returns": "bool",
                    "expr": _op("lte", _var("ratio"), threshold_expr),
                    "source_refs": ["A0515"],
                },
                "source_refs": ["A0515"],
            }
        ],
    }


def _build_a0515_rule() -> dict:
    table_id = "t_leverage_ratio_rules"
    threshold_expr = _op("lookup_rule", _lit("string", table_id), _lit("string", "predicate"), _lit("string", "max_ratio"))
    return {
        "schema_version": "covenant_ir_v0_1",
        "contract_id": "0000950134-96-000714_4",
        "sources": [
            {"source_id": "s1", "kind": "snippets", "item_id": "0000950134-96-000714_4", "anchor_ids": ["A0515"]},
        ],
        "tables": [
            {
                "table_id": table_id,
                "columns": [{"name": "predicate", "type": "bool"}, {"name": "max_ratio", "type": "decimal"}],
                "rows": [
                    {
                        "row_id": "through-1999-06-30",
                        "cells": {"predicate": _op("lte", _var("as_of_date"), _lit("date", "1999-06-30")), "max_ratio": _lit("decimal", "5.50")},
                        "source_refs": ["A0515"],
                    },
                    {
                        "row_id": "1999-07-01-through-2000-06-30",
                        "cells": {
                            "predicate": _op(
                                "and",
                                _op("gte", _var("as_of_date"), _lit("date", "1999-07-01")),
                                _op("lte", _var("as_of_date"), _lit("date", "2000-06-30")),
                            ),
                            "max_ratio": _lit("decimal", "5.00"),
                        },
                        "source_refs": ["A0515"],
                    },
                    {
                        "row_id": "2000-07-01+",
                        "cells": {"predicate": _op("gte", _var("as_of_date"), _lit("date", "2000-07-01")), "max_ratio": _lit("decimal", "4.50")},
                        "source_refs": ["A0515"],
                    },
                ],
            }
        ],
        "covenants": [
            {
                "covenant_id": "leverage_ratio",
                "title": "Leverage Ratio",
                "test": {
                    "args": [{"name": "as_of_date", "type": "date"}, {"name": "ratio", "type": "decimal"}],
                    "returns": "bool",
                    "expr": _op("lte", _var("ratio"), threshold_expr),
                    "source_refs": ["A0515"],
                },
                "source_refs": ["A0515"],
            }
        ],
    }


def test_precision_first_policy_rejects_partial_outputs_with_blocking_open_items() -> None:
    doc = _build_a0515_range()
    doc["open_items"] = [
        {
            "issue": "Missing schedule row for a covenant threshold",
            "source_refs": ["A0515"],
            "suggested_parameters": ["as_of_date"],
            "blocking": True,
        }
    ]

    # Base doc is schema-valid even with open_items added.
    assert validate_covenant_ir(doc) == []

    policy_errors = validate_precision_first_policy(doc)
    assert any(e.json_path == "/covenants" for e in policy_errors)
    assert any(e.json_path == "/tables" for e in policy_errors)

    # If we "fail the whole item" (no partial outputs), the policy passes.
    blocked = _build_a0515_range()
    blocked["open_items"] = doc["open_items"]
    blocked["covenants"] = []
    blocked["tables"] = []
    blocked["derived"] = []
    assert validate_covenant_ir(blocked) == []
    assert validate_precision_first_policy(blocked) == []


def test_covenant_ir_a0515_rule_and_range_match() -> None:
    doc_range = _build_a0515_range()
    doc_rule = _build_a0515_rule()

    assert validate_covenant_ir(doc_range) == []
    assert validate_covenant_ir(doc_rule) == []

    cases = [
        ({"as_of_date": "1999-05-01", "ratio": "5.60"}, False),
        ({"as_of_date": "1999-05-01", "ratio": "5.50"}, True),
        ({"as_of_date": "1999-07-15", "ratio": "5.10"}, False),
        ({"as_of_date": "1999-07-15", "ratio": "5.00"}, True),
        ({"as_of_date": "2001-01-01", "ratio": "4.60"}, False),
        ({"as_of_date": "2001-01-01", "ratio": "4.50"}, True),
    ]

    for args, expected in cases:
        r1 = evaluate_covenant(doc_range, covenant_id="leverage_ratio", args=args)
        r2 = evaluate_covenant(doc_rule, covenant_id="leverage_ratio", args=args)
        assert r1.applicable is True and r1.passed == expected
        assert r2.applicable is True and r2.passed == expected


def test_covenant_ir_harness_repair_prompt_mentions_lookup_rule_predicate_col_bool() -> None:
    """Regression test for a common LLM failure mode.

    Some models output correct predicate cell expressions, but mistakenly declare the predicate column type as
    \"string\" instead of \"bool\". Under Option 1, we do not deterministically repair this; instead we ensure
    the repair prompt explicitly tells the model how to fix it.
    """

    doc = _build_a0515_rule()
    assert validate_covenant_ir(doc) == []

    # Introduce the observed failure mode: predicate cells are bool-exprs, but the predicate column is typed as string.
    doc["tables"][0]["columns"][0]["type"] = "string"
    errs = validate_covenant_ir(doc)
    assert any(e.code == "lookup_rule_predicate_not_bool" for e in errs)

    spec = importlib.util.spec_from_file_location(
        "covenant_harness",
        "scripts/covenant_ir_v0_1_one_pass_harness.py",
    )
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    repair = mod._render_repair_prompt(raw_json="{}", errors=errs)
    assert "predicate column MUST be declared as type \"bool\"" in repair


def test_covenant_ir_supports_derived_fn_refs_for_bool_exprs() -> None:
    table_id = "t_leverage_ratio_thresholds"
    threshold_expr = _op(
        "lookup_range",
        _lit("string", table_id),
        _var("as_of_date"),
        _lit("string", "lower"),
        _lit("string", "lower_cmp"),
        _lit("string", "upper"),
        _lit("string", "upper_cmp"),
        _lit("string", "max_ratio"),
    )
    test_expr = _op("lte", _var("ratio"), threshold_expr)
    doc = {
        "schema_version": "covenant_ir_v0_1",
        "contract_id": "0000950134-96-000714_4",
        "sources": [
            {"source_id": "s1", "kind": "snippets", "item_id": "0000950134-96-000714_4", "anchor_ids": ["A0515"]},
        ],
        "tables": [
            {
                "table_id": table_id,
                "columns": [
                    {"name": "lower", "type": "date"},
                    {"name": "lower_cmp", "type": "string"},
                    {"name": "upper", "type": "date"},
                    {"name": "upper_cmp", "type": "string"},
                    {"name": "max_ratio", "type": "decimal"},
                ],
                "rows": [
                    {
                        "row_id": "through-1999-06-30",
                        "cells": {"upper": _lit("date", "1999-06-30"), "upper_cmp": _lit("string", "lte"), "max_ratio": _lit("decimal", "5.50")},
                        "source_refs": ["A0515"],
                    }
                ],
            }
        ],
        "derived": [
            {
                "fn_id": "fn_leverage_ratio_test",
                "semantic_role": "covenant_test",
                "args": [{"name": "as_of_date", "type": "date"}, {"name": "ratio", "type": "decimal"}],
                "returns": "bool",
                "expr": test_expr,
                "source_refs": ["A0515"],
            }
        ],
        "covenants": [
            {
                "covenant_id": "leverage_ratio",
                "title": "Leverage Ratio",
                "test": {"fn_id": "fn_leverage_ratio_test"},
                "source_refs": ["A0515"],
            }
        ],
    }

    assert validate_covenant_ir(doc) == []
    r = evaluate_covenant(doc, covenant_id="leverage_ratio", args={"as_of_date": "1999-05-01", "ratio": "5.60"})
    assert r.applicable is True and r.passed is False


def test_covenant_ir_rejects_non_bool_derived_fn_refs_for_bool_exprs() -> None:
    doc = {
        "schema_version": "covenant_ir_v0_1",
        "contract_id": "c1",
        "sources": [{"source_id": "s1", "kind": "snippets", "item_id": "c1", "anchor_ids": ["A0001"]}],
        "tables": [],
        "derived": [
            {
                "fn_id": "fn_not_bool",
                "semantic_role": "other",
                "args": [{"name": "x", "type": "decimal"}],
                "returns": "decimal",
                "expr": _var("x"),
                "source_refs": ["A0001"],
            }
        ],
        "covenants": [
            {"covenant_id": "c", "title": "C", "test": {"fn_id": "fn_not_bool"}, "source_refs": ["A0001"]},
        ],
    }

    errs = validate_covenant_ir(doc)
    assert any(e.code == "derived_fn_wrong_return" for e in errs)


def test_covenant_ir_a0437_tangible_net_worth_min_with_equity_adjustment() -> None:
    table_id = "t_tangible_net_worth_minimums"
    base_min_expr = _op("lookup_rule", _lit("string", table_id), _lit("string", "predicate"), _lit("string", "base_min_tnw"))
    adj_expr = _op("mul", _lit("decimal", "0.90"), _var("equity_issuance_tnw_increase"))
    required_min_expr = _op("add", base_min_expr, adj_expr)
    doc = {
        "schema_version": "covenant_ir_v0_1",
        "contract_id": "0000950152-96-000050_2",
        "sources": [{"source_id": "s1", "kind": "snippets", "item_id": "0000950152-96-000050_2", "anchor_ids": ["A0437"]}],
        "tables": [
            {
                "table_id": table_id,
                "columns": [{"name": "predicate", "type": "bool"}, {"name": "base_min_tnw", "type": "money"}],
                "rows": [
                    {
                        "row_id": "through-1996-03-02",
                        "cells": {"predicate": _op("lte", _var("as_of_date"), _lit("date", "1996-03-02")), "base_min_tnw": _lit("money", "51000000")},
                        "source_refs": ["A0437"],
                    },
                    {
                        "row_id": "1996-03-03-through-1997-03-01",
                        "cells": {
                            "predicate": _op(
                                "and",
                                _op("gte", _var("as_of_date"), _lit("date", "1996-03-03")),
                                _op("lte", _var("as_of_date"), _lit("date", "1997-03-01")),
                            ),
                            "base_min_tnw": _lit("money", "53000000"),
                        },
                        "source_refs": ["A0437"],
                    },
                    {
                        "row_id": "1997-03-02-through-1998-02-28",
                        "cells": {
                            "predicate": _op(
                                "and",
                                _op("gte", _var("as_of_date"), _lit("date", "1997-03-02")),
                                _op("lte", _var("as_of_date"), _lit("date", "1998-02-28")),
                            ),
                            "base_min_tnw": _lit("money", "58000000"),
                        },
                        "source_refs": ["A0437"],
                    },
                    {
                        "row_id": "1998-03-01+",
                        "cells": {"predicate": _op("gte", _var("as_of_date"), _lit("date", "1998-03-01")), "base_min_tnw": _lit("money", "65000000")},
                        "source_refs": ["A0437"],
                    },
                ],
            }
        ],
        "covenants": [
            {
                "covenant_id": "tangible_net_worth",
                "title": "Tangible Net Worth",
                "test": {
                    "args": [
                        {"name": "as_of_date", "type": "date"},
                        {"name": "tnw", "type": "money"},
                        {"name": "equity_issuance_tnw_increase", "type": "money"},
                    ],
                    "returns": "bool",
                    "expr": _op("gte", _var("tnw"), required_min_expr),
                    "source_refs": ["A0437"],
                },
                "source_refs": ["A0437"],
            }
        ],
    }

    assert validate_covenant_ir(doc) == []

    cases = [
        ({"as_of_date": "1996-03-02", "tnw": "50999999", "equity_issuance_tnw_increase": "0"}, False),
        ({"as_of_date": "1996-03-02", "tnw": "51000000", "equity_issuance_tnw_increase": "0"}, True),
        # Base=58,000,000; adjustment=0.90*2,000,000=1,800,000 => min=59,800,000
        ({"as_of_date": "1997-06-01", "tnw": "59700000", "equity_issuance_tnw_increase": "2000000"}, False),
        ({"as_of_date": "1997-06-01", "tnw": "59800000", "equity_issuance_tnw_increase": "2000000"}, True),
        # Base=65,000,000; adjustment=0.90*10,000,000=9,000,000 => min=74,000,000
        ({"as_of_date": "1999-01-01", "tnw": "74000000", "equity_issuance_tnw_increase": "10000000"}, True),
    ]
    for args, expected in cases:
        r = evaluate_covenant(doc, covenant_id="tangible_net_worth", args=args)
        assert r.applicable is True and r.passed == expected


def test_covenant_ir_a0443_minimum_current_ratio() -> None:
    table_id = "t_minimum_current_ratio"
    threshold_expr = _op("lookup_rule", _lit("string", table_id), _lit("string", "predicate"), _lit("string", "min_ratio"))
    ratio_expr = _op("div", _var("current_assets"), _var("current_liabilities"))
    doc = {
        "schema_version": "covenant_ir_v0_1",
        "contract_id": "0000950152-96-000050_2",
        "sources": [{"source_id": "s1", "kind": "snippets", "item_id": "0000950152-96-000050_2", "anchor_ids": ["A0443"]}],
        "tables": [
            {
                "table_id": table_id,
                "columns": [{"name": "predicate", "type": "bool"}, {"name": "min_ratio", "type": "decimal"}],
                "rows": [
                    {
                        "row_id": "through-1996-03-31",
                        "cells": {"predicate": _op("lte", _var("as_of_date"), _lit("date", "1996-03-31")), "min_ratio": _lit("decimal", "1.35")},
                        "source_refs": ["A0443"],
                    },
                    {
                        "row_id": "1996-04-01+",
                        "cells": {"predicate": _op("gte", _var("as_of_date"), _lit("date", "1996-04-01")), "min_ratio": _lit("decimal", "1.50")},
                        "source_refs": ["A0443"],
                    },
                ],
            }
        ],
        "covenants": [
            {
                "covenant_id": "minimum_current_ratio",
                "title": "Minimum Current Ratio",
                "test": {
                    "args": [
                        {"name": "as_of_date", "type": "date"},
                        {"name": "current_assets", "type": "money"},
                        {"name": "current_liabilities", "type": "money"},
                    ],
                    "returns": "bool",
                    "expr": _op("gte", ratio_expr, threshold_expr),
                    "source_refs": ["A0443"],
                },
                "source_refs": ["A0443"],
            }
        ],
    }

    assert validate_covenant_ir(doc) == []

    cases = [
        ({"as_of_date": "1996-03-31", "current_assets": "134", "current_liabilities": "100"}, False),
        ({"as_of_date": "1996-03-31", "current_assets": "135", "current_liabilities": "100"}, True),
        ({"as_of_date": "1996-04-01", "current_assets": "149", "current_liabilities": "100"}, False),
        ({"as_of_date": "1996-04-01", "current_assets": "150", "current_liabilities": "100"}, True),
    ]
    for args, expected in cases:
        r = evaluate_covenant(doc, covenant_id="minimum_current_ratio", args=args)
        assert r.applicable is True and r.passed == expected


def test_covenant_ir_lookup_rule_requires_bool_predicate_column() -> None:
    doc = {
        "schema_version": "covenant_ir_v0_1",
        "contract_id": "c1",
        "sources": [{"source_id": "s1", "kind": "snippets", "item_id": "c1", "anchor_ids": ["A0001"]}],
        "tables": [
            {
                "table_id": "T",
                "columns": [
                    {"name": "predicate", "type": "string"},
                    {"name": "v", "type": "decimal"},
                ],
                "rows": [
                    {
                        "row_id": "r1",
                        "cells": {
                            "predicate": _lit("string", "x > 1"),
                            "v": _lit("decimal", "1.0"),
                        },
                        "source_refs": ["A0001"],
                    }
                ],
            }
        ],
        "covenants": [
            {
                "covenant_id": "c",
                "title": "C",
                "test": {
                    "args": [{"name": "x", "type": "decimal"}],
                    "returns": "bool",
                    "expr": _op(
                        "gt",
                        _op("lookup_rule", _lit("string", "T"), _lit("string", "predicate"), _lit("string", "v")),
                        _lit("decimal", "0.0"),
                    ),
                    "source_refs": ["A0001"],
                },
                "source_refs": ["A0001"],
            }
        ],
    }
    errs = validate_covenant_ir(doc)
    assert any(e.code == "lookup_rule_predicate_not_bool" for e in errs)


def test_covenant_ir_rejects_undefined_var_in_inline_expr() -> None:
    doc = _build_a0515_range()
    # Drop the 'ratio' arg even though the expr references it.
    doc["covenants"][0]["test"]["args"] = [{"name": "as_of_date", "type": "date"}]
    errs = validate_covenant_ir(doc)
    assert any(e.code == "undefined_var" for e in errs)


def test_covenant_ir_rejects_undefined_var_used_in_table_predicate() -> None:
    doc = _build_a0515_rule()
    # The rule table predicate references as_of_date, but lookup_rule(...) itself does not.
    # If args omit as_of_date, evaluation would fail; treat as invalid CovenantIR.
    doc["covenants"][0]["test"]["args"] = [{"name": "ratio", "type": "decimal"}]
    errs = validate_covenant_ir(doc)
    assert any(e.code == "undefined_var" and "uses table" in e.message for e in errs)


def test_covenant_ir_rejects_wrong_operator_arity() -> None:
    doc = _build_a0515_range()
    doc["covenants"][0]["test"]["expr"] = _op("not", _var("ratio"), _var("ratio"))
    errs = validate_covenant_ir(doc)
    assert any(e.code == "op_arity" for e in errs)


def test_covenant_ir_rejects_duplicate_covenant_ids() -> None:
    doc = _build_a0515_range()
    dup = _build_a0515_range()
    dup["covenants"][0]["covenant_id"] = doc["covenants"][0]["covenant_id"]
    doc["covenants"].append(dup["covenants"][0])
    errs = validate_covenant_ir(doc)
    assert any(e.code == "duplicate_covenant_id" for e in errs)


def test_covenant_ir_rejects_placeholder_date_literals() -> None:
    doc = _build_a0515_range()
    doc["tables"][0]["rows"][0]["cells"]["upper"]["lit"]["value"] = "YYYY-01-01"
    errs = validate_covenant_ir(doc)
    assert any(e.code == "date_literal_format" for e in errs)
