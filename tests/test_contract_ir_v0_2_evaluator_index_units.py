from __future__ import annotations

import pytest

from pipeline.ir.contract_ir_v0_2 import evaluate_expr


def _lit(typ: str, value: object) -> dict:
    return {"lit": {"type": typ, "value": value}}


def _var(name: str) -> dict:
    return {"var": name}


def _op(op: str, *args: object) -> dict:
    return {"op": op, "args": list(args)}


def test_index_uses_declared_unit_decimal() -> None:
    expr = _op("index", _lit("string", "S"), _var("d"))
    out = evaluate_expr(
        expr=expr,
        arg_defs=[{"name": "d", "type": "date"}],
        args={"d": "2020-01-01"},
        indices={"S": {"2020-01-01": "1.23"}},
        tables={},
        index_units={"S": "decimal"},
    )
    assert out.kind == "decimal"
    assert str(out.value) == "1.23"


def test_index_uses_declared_unit_money() -> None:
    expr = _op("index", _lit("string", "Cash"), _var("d"))
    out = evaluate_expr(
        expr=expr,
        arg_defs=[{"name": "d", "type": "date"}],
        args={"d": "2020-01-01"},
        indices={"Cash": {"2020-01-01": "1000000"}},
        tables={},
        index_units={"Cash": "money"},
    )
    assert out.kind == "money"
    assert str(out.value) == "1000000"


def test_index_missing_unit_is_error() -> None:
    expr = _op("index", _lit("string", "S"), _var("d"))
    with pytest.raises(KeyError):
        evaluate_expr(
            expr=expr,
            arg_defs=[{"name": "d", "type": "date"}],
            args={"d": "2020-01-01"},
            indices={"S": {"2020-01-01": "1.23"}},
            tables={},
            index_units={},
        )


def test_index_bool_parsing() -> None:
    expr = _op("index", _lit("string", "flag"), _var("d"))
    out = evaluate_expr(
        expr=expr,
        arg_defs=[{"name": "d", "type": "date"}],
        args={"d": "2020-01-01"},
        indices={"flag": {"2020-01-01": "true"}},
        tables={},
        index_units={"flag": "bool"},
    )
    assert out.kind == "bool"
    assert out.value is True

