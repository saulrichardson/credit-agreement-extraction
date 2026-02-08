#!/usr/bin/env python
"""
Local exploration harness for CovenantIR v0.1 design choices.

It builds two encodings for the same covenant logic:
  - Range schedule: lookup_range(...)
  - Rule schedule: lookup_rule(...) with per-row predicates

Then validates + evaluates both encodings on a small set of test cases,
and prints simple size/complexity metrics to help pick defaults.

Usage:
  python scripts/covenant_ir_design_harness.py
"""

from __future__ import annotations

import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from pipeline.ir.covenant_ir_v0_1 import evaluate_covenant, validate_covenant_ir  # noqa: E402


def _lit(typ: str, value: Any) -> Dict[str, Any]:
    return {"lit": {"type": typ, "value": value}}


def _var(name: str) -> Dict[str, Any]:
    return {"var": name}


def _op(op: str, *args: Any) -> Dict[str, Any]:
    return {"op": op, "args": list(args)}


def _count_ast_nodes(node: Any) -> int:
    if not isinstance(node, dict):
        return 0
    total = 1
    if "args" in node and isinstance(node.get("args"), list):
        for a in node["args"]:
            total += _count_ast_nodes(a)
    if "lit" in node and isinstance(node.get("lit"), dict):
        return total
    if "var" in node:
        return total
    return total


def _doc_metrics(doc: Mapping[str, Any]) -> Dict[str, Any]:
    encoded = json.dumps(doc, sort_keys=True)
    ast_nodes = 0
    table_cells = 0
    for cov in doc.get("covenants", []) or []:
        if not isinstance(cov, dict):
            continue
        for k in ("test", "applies_when"):
            spec = cov.get(k)
            if isinstance(spec, dict) and "expr" in spec:
                ast_nodes += _count_ast_nodes(spec.get("expr"))
    for fn in doc.get("derived", []) or []:
        if isinstance(fn, dict):
            ast_nodes += _count_ast_nodes(fn.get("expr"))
    for table in doc.get("tables", []) or []:
        if not isinstance(table, dict):
            continue
        for row in table.get("rows", []) or []:
            if not isinstance(row, dict):
                continue
            cells = row.get("cells")
            if not isinstance(cells, dict):
                continue
            for node in cells.values():
                if not isinstance(node, dict):
                    continue
                table_cells += 1
                ast_nodes += _count_ast_nodes(node)
    return {
        "json_bytes": len(encoded.encode("utf-8")),
        "tables": len(doc.get("tables", []) or []),
        "table_cells": table_cells,
        "derived": len(doc.get("derived", []) or []),
        "ast_nodes": ast_nodes,
    }


def build_examples() -> List[Dict[str, Any]]:
    examples: List[Dict[str, Any]] = []

    # Example: 0000950170-23-000122_2 / A2675
    # Net Secured Leverage Ratio max threshold schedule by quarter-end date, springing by utilization >= 30%.
    examples.append(
        {
            "example_id": "A2675_net_secured_leverage_ratio_springing",
            "covenant_id": "net_secured_leverage_ratio",
            "supports_range": True,
            "cases": [
                {
                    "args": {"as_of_date": "2019-09-30", "ratio": "7.30", "utilization_pct": "0.35"},
                    "expected": {"applicable": True, "passed": False},
                },
                {
                    "args": {"as_of_date": "2019-09-30", "ratio": "7.25", "utilization_pct": "0.35"},
                    "expected": {"applicable": True, "passed": True},
                },
                {
                    "args": {"as_of_date": "2019-09-30", "ratio": "9.99", "utilization_pct": "0.20"},
                    "expected": {"applicable": False, "passed": None},
                },
                {
                    "args": {"as_of_date": "2021-03-31", "ratio": "6.26", "utilization_pct": "0.35"},
                    "expected": {"applicable": True, "passed": False},
                },
            ],
        }
    )

    # Example: 0000950134-96-000714_4 / A0515
    examples.append(
        {
            "example_id": "A0515_leverage_ratio_date_schedule",
            "covenant_id": "leverage_ratio",
            "supports_range": True,
            "cases": [
                {"args": {"as_of_date": "1999-05-01", "ratio": "5.60"}, "expected": {"applicable": True, "passed": False}},
                {"args": {"as_of_date": "1999-05-01", "ratio": "5.50"}, "expected": {"applicable": True, "passed": True}},
                {"args": {"as_of_date": "1999-07-15", "ratio": "5.10"}, "expected": {"applicable": True, "passed": False}},
                {"args": {"as_of_date": "1999-07-15", "ratio": "5.00"}, "expected": {"applicable": True, "passed": True}},
                {"args": {"as_of_date": "2001-01-01", "ratio": "4.60"}, "expected": {"applicable": True, "passed": False}},
            ],
        }
    )

    # Example: 0000950134-96-000714_4 / A0520
    examples.append(
        {
            "example_id": "A0520_interest_coverage_ratio_min_schedule",
            "covenant_id": "interest_coverage_ratio",
            "supports_range": True,
            "cases": [
                {"args": {"as_of_date": "1999-05-01", "ratio": "1.40"}, "expected": {"applicable": True, "passed": False}},
                {"args": {"as_of_date": "1999-05-01", "ratio": "1.50"}, "expected": {"applicable": True, "passed": True}},
                {"args": {"as_of_date": "1999-07-01", "ratio": "1.90"}, "expected": {"applicable": True, "passed": False}},
                {"args": {"as_of_date": "1999-07-01", "ratio": "2.00"}, "expected": {"applicable": True, "passed": True}},
            ],
        }
    )

    # Example: 0000950152-96-000050_2 / A0437
    # Tangible Net Worth minimum schedule, with automatic increases based on equity issuance.
    examples.append(
        {
            "example_id": "A0437_tangible_net_worth_min_with_equity_adjustment",
            "covenant_id": "tangible_net_worth",
            # We focus on lookup_rule for pressure testing (range encoding omitted for now).
            "supports_range": False,
            "cases": [
                {"args": {"as_of_date": "1996-03-02", "tnw": "50999999", "equity_issuance_tnw_increase": "0"}, "expected": {"applicable": True, "passed": False}},
                {"args": {"as_of_date": "1996-03-02", "tnw": "51000000", "equity_issuance_tnw_increase": "0"}, "expected": {"applicable": True, "passed": True}},
                # Base=58,000,000; adjustment=0.90*2,000,000=1,800,000 => min=59,800,000
                {"args": {"as_of_date": "1997-06-01", "tnw": "59700000", "equity_issuance_tnw_increase": "2000000"}, "expected": {"applicable": True, "passed": False}},
                {"args": {"as_of_date": "1997-06-01", "tnw": "59800000", "equity_issuance_tnw_increase": "2000000"}, "expected": {"applicable": True, "passed": True}},
                # Base=65,000,000; adjustment=0.90*10,000,000=9,000,000 => min=74,000,000
                {"args": {"as_of_date": "1999-01-01", "tnw": "74000000", "equity_issuance_tnw_increase": "10000000"}, "expected": {"applicable": True, "passed": True}},
            ],
        }
    )

    # Example: 0000950152-96-000050_2 / A0443
    # Minimum Current Ratio schedule; ratio = current_assets / current_liabilities.
    examples.append(
        {
            "example_id": "A0443_minimum_current_ratio",
            "covenant_id": "minimum_current_ratio",
            "supports_range": False,
            "cases": [
                {"args": {"as_of_date": "1996-03-31", "current_assets": "134", "current_liabilities": "100"}, "expected": {"applicable": True, "passed": False}},
                {"args": {"as_of_date": "1996-03-31", "current_assets": "135", "current_liabilities": "100"}, "expected": {"applicable": True, "passed": True}},
                {"args": {"as_of_date": "1996-04-01", "current_assets": "149", "current_liabilities": "100"}, "expected": {"applicable": True, "passed": False}},
                {"args": {"as_of_date": "1996-04-01", "current_assets": "150", "current_liabilities": "100"}, "expected": {"applicable": True, "passed": True}},
            ],
        }
    )

    return examples


def build_ir_range(example: Mapping[str, Any]) -> Dict[str, Any]:
    ex_id = example["example_id"]
    covenant_id = example["covenant_id"]

    if ex_id == "A2675_net_secured_leverage_ratio_springing":
        table_id = "t_net_secured_leverage_ratio_thresholds"
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
        applies_expr = _op("gte", _var("utilization_pct"), _lit("decimal", "0.30"))

        doc = {
            "schema_version": "covenant_ir_v0_1",
            "contract_id": "0000950170-23-000122_2",
            "sources": [
                {"source_id": "s1", "kind": "snippets", "item_id": "0000950170-23-000122_2", "anchor_ids": ["A2675"]},
            ],
            "tables": [
                {
                    "table_id": table_id,
                    "description": "Quarter-end thresholds for Consolidated Net Secured Leverage Ratio",
                    "columns": [
                        {"name": "lower", "type": "date"},
                        {"name": "lower_cmp", "type": "string"},
                        {"name": "upper", "type": "date"},
                        {"name": "upper_cmp", "type": "string"},
                        {"name": "max_ratio", "type": "decimal"},
                    ],
                    "rows": [
                        {
                            "row_id": "2018Q3-2019Q3",
                            "cells": {
                                "lower": _lit("date", "2018-09-30"),
                                "lower_cmp": _lit("string", "gte"),
                                "upper": _lit("date", "2019-09-30"),
                                "upper_cmp": _lit("string", "lte"),
                                "max_ratio": _lit("decimal", "7.25"),
                            },
                            "source_refs": ["A2675"],
                        },
                        {
                            "row_id": "2019Q4-2020Q4",
                            "cells": {
                                "lower": _lit("date", "2019-12-31"),
                                "lower_cmp": _lit("string", "gte"),
                                "upper": _lit("date", "2020-12-31"),
                                "upper_cmp": _lit("string", "lte"),
                                "max_ratio": _lit("decimal", "6.75"),
                            },
                            "source_refs": ["A2675"],
                        },
                        {
                            "row_id": "2021Q1+",
                            "cells": {
                                "lower": _lit("date", "2021-03-31"),
                                "lower_cmp": _lit("string", "gte"),
                                "max_ratio": _lit("decimal", "6.25"),
                            },
                            "source_refs": ["A2675"],
                        },
                    ],
                }
            ],
            "covenants": [
                {
                    "covenant_id": covenant_id,
                    "title": "Consolidated Net Secured Leverage Ratio",
                    "category": "financial_covenant",
                    "applies_when": {
                        "args": [
                            {"name": "utilization_pct", "type": "decimal"},
                        ],
                        "returns": "bool",
                        "expr": applies_expr,
                        "source_refs": ["A2675"],
                    },
                    "test": {
                        "args": [
                            {"name": "as_of_date", "type": "date"},
                            {"name": "ratio", "type": "decimal"},
                        ],
                        "returns": "bool",
                        "expr": test_expr,
                        "source_refs": ["A2675"],
                    },
                    "source_refs": ["A2675"],
                    "notes": "Springing covenant: applies only when utilization_pct >= 0.30 (details simplified into utilization_pct input).",
                }
            ],
        }
        return doc

    if ex_id == "A0515_leverage_ratio_date_schedule":
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
                    "description": "Leverage Ratio maximums by date period",
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
                            "cells": {
                                "upper": _lit("date", "1999-06-30"),
                                "upper_cmp": _lit("string", "lte"),
                                "max_ratio": _lit("decimal", "5.50"),
                            },
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
                            "cells": {
                                "lower": _lit("date", "2000-07-01"),
                                "lower_cmp": _lit("string", "gte"),
                                "max_ratio": _lit("decimal", "4.50"),
                            },
                            "source_refs": ["A0515"],
                        },
                    ],
                }
            ],
            "covenants": [
                {
                    "covenant_id": covenant_id,
                    "title": "Leverage Ratio",
                    "category": "financial_covenant",
                    "test": {
                        "args": [{"name": "as_of_date", "type": "date"}, {"name": "ratio", "type": "decimal"}],
                        "returns": "bool",
                        "expr": test_expr,
                        "source_refs": ["A0515"],
                    },
                    "source_refs": ["A0515"],
                }
            ],
        }
        return doc

    if ex_id == "A0520_interest_coverage_ratio_min_schedule":
        table_id = "t_icr_thresholds"
        threshold_expr = _op(
            "lookup_range",
            _lit("string", table_id),
            _var("as_of_date"),
            _lit("string", "lower"),
            _lit("string", "lower_cmp"),
            _lit("string", "upper"),
            _lit("string", "upper_cmp"),
            _lit("string", "min_ratio"),
        )
        test_expr = _op("gte", _var("ratio"), threshold_expr)
        doc = {
            "schema_version": "covenant_ir_v0_1",
            "contract_id": "0000950134-96-000714_4",
            "sources": [
                {"source_id": "s1", "kind": "snippets", "item_id": "0000950134-96-000714_4", "anchor_ids": ["A0520"]},
            ],
            "tables": [
                {
                    "table_id": table_id,
                    "description": "Interest Coverage Ratio minimums by date period",
                    "columns": [
                        {"name": "lower", "type": "date"},
                        {"name": "lower_cmp", "type": "string"},
                        {"name": "upper", "type": "date"},
                        {"name": "upper_cmp", "type": "string"},
                        {"name": "min_ratio", "type": "decimal"},
                    ],
                    "rows": [
                        {
                            "row_id": "through-1999-06-30",
                            "cells": {
                                "upper": _lit("date", "1999-06-30"),
                                "upper_cmp": _lit("string", "lte"),
                                "min_ratio": _lit("decimal", "1.50"),
                            },
                            "source_refs": ["A0520"],
                        },
                        {
                            "row_id": "1999-07-01+",
                            "cells": {
                                "lower": _lit("date", "1999-07-01"),
                                "lower_cmp": _lit("string", "gte"),
                                "min_ratio": _lit("decimal", "2.00"),
                            },
                            "source_refs": ["A0520"],
                        },
                    ],
                }
            ],
            "covenants": [
                {
                    "covenant_id": covenant_id,
                    "title": "Interest Coverage Ratio",
                    "category": "financial_covenant",
                    "test": {
                        "args": [{"name": "as_of_date", "type": "date"}, {"name": "ratio", "type": "decimal"}],
                        "returns": "bool",
                        "expr": test_expr,
                        "source_refs": ["A0520"],
                    },
                    "source_refs": ["A0520"],
                }
            ],
        }
        return doc

    raise ValueError(f"Unknown example_id: {ex_id}")


def build_ir_rule(example: Mapping[str, Any]) -> Dict[str, Any]:
    ex_id = example["example_id"]
    covenant_id = example["covenant_id"]

    def _date(s: str) -> Dict[str, Any]:
        return _lit("date", s)

    if ex_id == "A2675_net_secured_leverage_ratio_springing":
        table_id = "t_net_secured_leverage_ratio_rules"
        threshold_expr = _op("lookup_rule", _lit("string", table_id), _lit("string", "predicate"), _lit("string", "max_ratio"))
        test_expr = _op("lte", _var("ratio"), threshold_expr)
        applies_expr = _op("gte", _var("utilization_pct"), _lit("decimal", "0.30"))

        doc = {
            "schema_version": "covenant_ir_v0_1",
            "contract_id": "0000950170-23-000122_2",
            "sources": [
                {"source_id": "s1", "kind": "snippets", "item_id": "0000950170-23-000122_2", "anchor_ids": ["A2675"]},
            ],
            "tables": [
                {
                    "table_id": table_id,
                    "description": "Quarter-end thresholds as rule predicates",
                    "columns": [
                        {"name": "predicate", "type": "bool"},
                        {"name": "max_ratio", "type": "decimal"},
                    ],
                    "rows": [
                        {
                            "row_id": "2018Q3-2019Q3",
                            "cells": {
                                "predicate": _op(
                                    "and",
                                    _op("gte", _var("as_of_date"), _date("2018-09-30")),
                                    _op("lte", _var("as_of_date"), _date("2019-09-30")),
                                ),
                                "max_ratio": _lit("decimal", "7.25"),
                            },
                            "source_refs": ["A2675"],
                        },
                        {
                            "row_id": "2019Q4-2020Q4",
                            "cells": {
                                "predicate": _op(
                                    "and",
                                    _op("gte", _var("as_of_date"), _date("2019-12-31")),
                                    _op("lte", _var("as_of_date"), _date("2020-12-31")),
                                ),
                                "max_ratio": _lit("decimal", "6.75"),
                            },
                            "source_refs": ["A2675"],
                        },
                        {
                            "row_id": "2021Q1+",
                            "cells": {
                                "predicate": _op("gte", _var("as_of_date"), _date("2021-03-31")),
                                "max_ratio": _lit("decimal", "6.25"),
                            },
                            "source_refs": ["A2675"],
                        },
                    ],
                }
            ],
            "covenants": [
                {
                    "covenant_id": covenant_id,
                    "title": "Consolidated Net Secured Leverage Ratio",
                    "category": "financial_covenant",
                    "applies_when": {
                        "args": [{"name": "utilization_pct", "type": "decimal"}],
                        "returns": "bool",
                        "expr": applies_expr,
                        "source_refs": ["A2675"],
                    },
                    "test": {
                        "args": [{"name": "as_of_date", "type": "date"}, {"name": "ratio", "type": "decimal"}],
                        "returns": "bool",
                        "expr": test_expr,
                        "source_refs": ["A2675"],
                    },
                    "source_refs": ["A2675"],
                }
            ],
        }
        return doc

    if ex_id == "A0515_leverage_ratio_date_schedule":
        table_id = "t_leverage_ratio_rules"
        threshold_expr = _op("lookup_rule", _lit("string", table_id), _lit("string", "predicate"), _lit("string", "max_ratio"))
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
                    "description": "Leverage Ratio maximums as rule predicates",
                    "columns": [
                        {"name": "predicate", "type": "bool"},
                        {"name": "max_ratio", "type": "decimal"},
                    ],
                    "rows": [
                        {
                            "row_id": "through-1999-06-30",
                            "cells": {
                                "predicate": _op("lte", _var("as_of_date"), _date("1999-06-30")),
                                "max_ratio": _lit("decimal", "5.50"),
                            },
                            "source_refs": ["A0515"],
                        },
                        {
                            "row_id": "1999-07-01-through-2000-06-30",
                            "cells": {
                                "predicate": _op(
                                    "and",
                                    _op("gte", _var("as_of_date"), _date("1999-07-01")),
                                    _op("lte", _var("as_of_date"), _date("2000-06-30")),
                                ),
                                "max_ratio": _lit("decimal", "5.00"),
                            },
                            "source_refs": ["A0515"],
                        },
                        {
                            "row_id": "2000-07-01+",
                            "cells": {
                                "predicate": _op("gte", _var("as_of_date"), _date("2000-07-01")),
                                "max_ratio": _lit("decimal", "4.50"),
                            },
                            "source_refs": ["A0515"],
                        },
                    ],
                }
            ],
            "covenants": [
                {
                    "covenant_id": covenant_id,
                    "title": "Leverage Ratio",
                    "category": "financial_covenant",
                    "test": {
                        "args": [{"name": "as_of_date", "type": "date"}, {"name": "ratio", "type": "decimal"}],
                        "returns": "bool",
                        "expr": test_expr,
                        "source_refs": ["A0515"],
                    },
                    "source_refs": ["A0515"],
                }
            ],
        }
        return doc

    if ex_id == "A0520_interest_coverage_ratio_min_schedule":
        table_id = "t_icr_rules"
        threshold_expr = _op("lookup_rule", _lit("string", table_id), _lit("string", "predicate"), _lit("string", "min_ratio"))
        test_expr = _op("gte", _var("ratio"), threshold_expr)
        doc = {
            "schema_version": "covenant_ir_v0_1",
            "contract_id": "0000950134-96-000714_4",
            "sources": [
                {"source_id": "s1", "kind": "snippets", "item_id": "0000950134-96-000714_4", "anchor_ids": ["A0520"]},
            ],
            "tables": [
                {
                    "table_id": table_id,
                    "description": "Interest Coverage Ratio minimums as rule predicates",
                    "columns": [
                        {"name": "predicate", "type": "bool"},
                        {"name": "min_ratio", "type": "decimal"},
                    ],
                    "rows": [
                        {
                            "row_id": "through-1999-06-30",
                            "cells": {
                                "predicate": _op("lte", _var("as_of_date"), _date("1999-06-30")),
                                "min_ratio": _lit("decimal", "1.50"),
                            },
                            "source_refs": ["A0520"],
                        },
                        {
                            "row_id": "1999-07-01+",
                            "cells": {
                                "predicate": _op("gte", _var("as_of_date"), _date("1999-07-01")),
                                "min_ratio": _lit("decimal", "2.00"),
                            },
                            "source_refs": ["A0520"],
                        },
                    ],
                }
            ],
            "covenants": [
                {
                    "covenant_id": covenant_id,
                    "title": "Interest Coverage Ratio",
                    "category": "financial_covenant",
                    "test": {
                        "args": [{"name": "as_of_date", "type": "date"}, {"name": "ratio", "type": "decimal"}],
                        "returns": "bool",
                        "expr": test_expr,
                        "source_refs": ["A0520"],
                    },
                    "source_refs": ["A0520"],
                }
            ],
        }
        return doc

    if ex_id == "A0437_tangible_net_worth_min_with_equity_adjustment":
        table_id = "t_tangible_net_worth_minimums"
        base_min_expr = _op("lookup_rule", _lit("string", table_id), _lit("string", "predicate"), _lit("string", "base_min_tnw"))
        adj_expr = _op("mul", _lit("decimal", "0.90"), _var("equity_issuance_tnw_increase"))
        required_min_expr = _op("add", base_min_expr, adj_expr)
        test_expr = _op("gte", _var("tnw"), required_min_expr)

        return {
            "schema_version": "covenant_ir_v0_1",
            "contract_id": "0000950152-96-000050_2",
            "sources": [
                {"source_id": "s1", "kind": "snippets", "item_id": "0000950152-96-000050_2", "anchor_ids": ["A0437"]},
            ],
            "tables": [
                {
                    "table_id": table_id,
                    "description": "Tangible Net Worth minimum schedule (base values)",
                    "columns": [
                        {"name": "predicate", "type": "bool"},
                        {"name": "base_min_tnw", "type": "money"},
                    ],
                    "rows": [
                        {
                            "row_id": "through-1996-03-02",
                            "cells": {
                                "predicate": _op("lte", _var("as_of_date"), _lit("date", "1996-03-02")),
                                "base_min_tnw": _lit("money", "51000000"),
                            },
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
                            "cells": {
                                "predicate": _op("gte", _var("as_of_date"), _lit("date", "1998-03-01")),
                                "base_min_tnw": _lit("money", "65000000"),
                            },
                            "source_refs": ["A0437"],
                        },
                    ],
                }
            ],
            "covenants": [
                {
                    "covenant_id": covenant_id,
                    "title": "Tangible Net Worth",
                    "category": "financial_covenant",
                    "test": {
                        "args": [
                            {"name": "as_of_date", "type": "date"},
                            {"name": "tnw", "type": "money"},
                            {"name": "equity_issuance_tnw_increase", "type": "money"},
                        ],
                        "returns": "bool",
                        "expr": test_expr,
                        "source_refs": ["A0437"],
                    },
                    "source_refs": ["A0437"],
                    "notes": (
                        "Encodes only the base schedule + a single input 'equity_issuance_tnw_increase' (assumed cumulative as-of-date) "
                        "with a 0.90 multiplier per the covenant text."
                    ),
                }
            ],
        }

    if ex_id == "A0443_minimum_current_ratio":
        table_id = "t_minimum_current_ratio"
        threshold_expr = _op("lookup_rule", _lit("string", table_id), _lit("string", "predicate"), _lit("string", "min_ratio"))
        ratio_expr = _op("div", _var("current_assets"), _var("current_liabilities"))
        test_expr = _op("gte", ratio_expr, threshold_expr)

        return {
            "schema_version": "covenant_ir_v0_1",
            "contract_id": "0000950152-96-000050_2",
            "sources": [
                {"source_id": "s1", "kind": "snippets", "item_id": "0000950152-96-000050_2", "anchor_ids": ["A0443"]},
            ],
            "tables": [
                {
                    "table_id": table_id,
                    "description": "Minimum Consolidated Current Ratio thresholds by date",
                    "columns": [
                        {"name": "predicate", "type": "bool"},
                        {"name": "min_ratio", "type": "decimal"},
                    ],
                    "rows": [
                        {
                            "row_id": "through-1996-03-31",
                            "cells": {
                                "predicate": _op("lte", _var("as_of_date"), _lit("date", "1996-03-31")),
                                "min_ratio": _lit("decimal", "1.35"),
                            },
                            "source_refs": ["A0443"],
                        },
                        {
                            "row_id": "1996-04-01+",
                            "cells": {
                                "predicate": _op("gte", _var("as_of_date"), _lit("date", "1996-04-01")),
                                "min_ratio": _lit("decimal", "1.50"),
                            },
                            "source_refs": ["A0443"],
                        },
                    ],
                }
            ],
            "covenants": [
                {
                    "covenant_id": covenant_id,
                    "title": "Minimum Current Ratio",
                    "category": "financial_covenant",
                    "test": {
                        "args": [
                            {"name": "as_of_date", "type": "date"},
                            {"name": "current_assets", "type": "money"},
                            {"name": "current_liabilities", "type": "money"},
                        ],
                        "returns": "bool",
                        "expr": test_expr,
                        "source_refs": ["A0443"],
                    },
                    "source_refs": ["A0443"],
                }
            ],
        }

    raise ValueError(f"Unknown example_id: {ex_id}")


def main() -> int:
    examples = build_examples()
    for ex in examples:
        ex_id = ex["example_id"]
        print(f"\n=== {ex_id} ===")

        for variant in ("range_inline", "range_derived", "rule_inline", "rule_derived"):
            if variant.startswith("range") and not ex.get("supports_range", True):
                continue
            if variant.startswith("range"):
                doc = build_ir_range(ex)
            else:
                doc = build_ir_rule(ex)

            if variant.endswith("derived"):
                _lift_inline_exprs_to_derived(doc)

            errs = validate_covenant_ir(doc)
            if errs:
                print(f"\n[{variant}] INVALID:")
                for e in errs[:10]:
                    print(" ", asdict(e))
                continue
            m = _doc_metrics(doc)
            print(f"\n[{variant}] metrics={m}")

            for case in ex["cases"]:
                result = evaluate_covenant(doc, covenant_id=ex["covenant_id"], args=case["args"])
                exp = case["expected"]
                ok = result.applicable == exp["applicable"] and result.passed == exp["passed"]
                status = "OK" if ok else "MISMATCH"
                print(f"  - {status} args={case['args']} -> applicable={result.applicable} passed={result.passed}")

    return 0


def _lift_inline_exprs_to_derived(doc: Dict[str, Any]) -> None:
    covs = doc.get("covenants")
    if not isinstance(covs, list) or not covs:
        return
    cov = covs[0]
    if not isinstance(cov, dict):
        return
    doc.setdefault("derived", [])
    derived = doc["derived"]
    if not isinstance(derived, list):
        raise TypeError("doc['derived'] must be a list when present")

    def _lift(field: str, fn_id: str) -> None:
        spec = cov.get(field)
        if not isinstance(spec, dict) or "expr" not in spec:
            return
        derived.append(
            {
                "fn_id": fn_id,
                "semantic_role": "covenant_test",
                "args": spec.get("args", []) or [],
                "returns": spec.get("returns"),
                "expr": spec.get("expr"),
                "source_refs": spec.get("source_refs", []),
            }
        )
        cov[field] = {"fn_id": fn_id}

    _lift("applies_when", "fn_applies_when")
    _lift("test", "fn_test")


if __name__ == "__main__":
    raise SystemExit(main())
