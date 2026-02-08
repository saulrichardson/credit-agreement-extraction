from __future__ import annotations

import pytest

from pipeline.ir.contract_ir_v0_2 import NoMatchingRow, evaluate_expr


def _lit(typ: str, value: object) -> dict:
    return {"lit": {"type": typ, "value": value}}


def _var(name: str) -> dict:
    return {"var": name}


def _op(op: str, *args: object) -> dict:
    return {"op": op, "args": list(args)}


def test_lookup_rule_supports_or_and_2d_grid_like_a0125() -> None:
    table_id = "ApplicableMarginByCoverageRatios"
    tables = {
        table_id: {
            "table_id": table_id,
            "rows": [
                {
                    "row_id": "row1",
                    "cells": {
                        "predicate": _op(
                            "or",
                            _op("lt", _var("interest_coverage_ratio"), _lit("decimal", "1.75")),
                            _op("lt", _var("fixed_charge_coverage_ratio"), _lit("decimal", "1.20")),
                        ),
                        "margin_bps": _lit("bps", "250"),
                    },
                },
                {
                    "row_id": "row2",
                    "cells": {
                        "predicate": _op(
                            "and",
                            _op(
                                "and",
                                _op("gte", _var("interest_coverage_ratio"), _lit("decimal", "1.75")),
                                _op("lt", _var("interest_coverage_ratio"), _lit("decimal", "2.50")),
                            ),
                            _op(
                                "and",
                                _op("gte", _var("fixed_charge_coverage_ratio"), _lit("decimal", "1.20")),
                                _op("lt", _var("fixed_charge_coverage_ratio"), _lit("decimal", "1.40")),
                            ),
                        ),
                        "margin_bps": _lit("bps", "225"),
                    },
                },
                {
                    "row_id": "row3",
                    "cells": {
                        "predicate": _op(
                            "and",
                            _op("gte", _var("interest_coverage_ratio"), _lit("decimal", "2.50")),
                            _op("gte", _var("fixed_charge_coverage_ratio"), _lit("decimal", "1.40")),
                        ),
                        "margin_bps": _lit("bps", "200"),
                    },
                },
            ],
        }
    }

    expr = _op(
        "bps_to_rate",
        _op(
            "lookup_rule",
            _lit("string", table_id),
            _lit("string", "predicate"),
            _lit("string", "margin_bps"),
        ),
    )

    def eval_margin(icr: str, fccr: str) -> str:
        out = evaluate_expr(
            expr=expr,
            arg_defs=[
                {"name": "interest_coverage_ratio", "type": "decimal"},
                {"name": "fixed_charge_coverage_ratio", "type": "decimal"},
            ],
            args={"interest_coverage_ratio": icr, "fixed_charge_coverage_ratio": fccr},
            indices={},
            tables=tables,
        )
        assert out.kind == "rate"
        return str(out.value)

    assert eval_margin("1.74", "1.19") == "0.025"
    # Boundary should land in row2 (row1 is strict lt)
    assert eval_margin("1.75", "1.20") == "0.0225"
    assert eval_margin("2.49", "1.39") == "0.0225"
    assert eval_margin("2.50", "1.40") == "0.02"

    # Gap region (ICR >= 2.50 but FCCR < 1.40) should fail loudly.
    with pytest.raises(NoMatchingRow):
        eval_margin("2.60", "1.30")

