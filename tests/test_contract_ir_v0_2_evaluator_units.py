from __future__ import annotations

from decimal import Decimal

import pytest

from pipeline.contract_ir_v0_2 import MultipleMatchingRows, NoMatchingRow, evaluate_expr


def _var(name: str) -> dict:
    return {"var": name}


def _op(op: str, *args: object) -> dict:
    return {"op": op, "args": list(args)}


def test_money_div_money_returns_decimal_ratio() -> None:
    tv = evaluate_expr(
        expr=_op("div", _var("a"), _var("b")),
        arg_defs=[{"name": "a", "type": "money"}, {"name": "b", "type": "money"}],
        args={"a": "150", "b": "100"},
        indices={},
        tables={},
    )
    assert tv.kind == "decimal"
    assert tv.value == Decimal("1.5")


def test_rate_div_rate_returns_decimal_ratio() -> None:
    tv = evaluate_expr(
        expr=_op("div", _var("a"), _var("b")),
        arg_defs=[{"name": "a", "type": "rate"}, {"name": "b", "type": "rate"}],
        args={"a": "0.05", "b": "0.025"},
        indices={},
        tables={},
    )
    assert tv.kind == "decimal"
    assert tv.value == Decimal("2")


def test_money_div_decimal_preserves_money() -> None:
    tv = evaluate_expr(
        expr=_op("div", _var("a"), _var("b")),
        arg_defs=[{"name": "a", "type": "money"}, {"name": "b", "type": "decimal"}],
        args={"a": "1000", "b": "2"},
        indices={},
        tables={},
    )
    assert tv.kind == "money"
    assert tv.value == Decimal("500")


def test_rate_div_decimal_preserves_rate() -> None:
    tv = evaluate_expr(
        expr=_op("div", _var("a"), _var("b")),
        arg_defs=[{"name": "a", "type": "rate"}, {"name": "b", "type": "decimal"}],
        args={"a": "0.05", "b": "2"},
        indices={},
        tables={},
    )
    assert tv.kind == "rate"
    assert tv.value == Decimal("0.025")


def test_lookup_rule_multiple_matching_rows_raises() -> None:
    table_id = "t"
    tables = {
        table_id: {
            "table_id": table_id,
            "rows": [
                {"row_id": "r1", "cells": {"predicate": {"lit": {"type": "bool", "value": True}}, "v": {"lit": {"type": "decimal", "value": "1"}}}},
                {"row_id": "r2", "cells": {"predicate": {"lit": {"type": "bool", "value": True}}, "v": {"lit": {"type": "decimal", "value": "2"}}}},
            ],
        }
    }
    expr = _op(
        "lookup_rule",
        {"lit": {"type": "string", "value": table_id}},
        {"lit": {"type": "string", "value": "predicate"}},
        {"lit": {"type": "string", "value": "v"}},
    )
    with pytest.raises(MultipleMatchingRows):
        evaluate_expr(expr=expr, arg_defs=[], args={}, indices={}, tables=tables)


def test_lookup_rule_no_matching_rows_raises() -> None:
    table_id = "t"
    tables = {
        table_id: {
            "table_id": table_id,
            "rows": [
                {"row_id": "r1", "cells": {"predicate": {"lit": {"type": "bool", "value": False}}, "v": {"lit": {"type": "decimal", "value": "1"}}}},
            ],
        }
    }
    expr = _op(
        "lookup_rule",
        {"lit": {"type": "string", "value": table_id}},
        {"lit": {"type": "string", "value": "predicate"}},
        {"lit": {"type": "string", "value": "v"}},
    )
    with pytest.raises(NoMatchingRow):
        evaluate_expr(expr=expr, arg_defs=[], args={}, indices={}, tables=tables)
