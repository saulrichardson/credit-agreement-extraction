import pytest

from pipeline.compile.compustat_formula import CompustatFormulaError, evaluate_compustat_formula


def test_evaluate_compustat_formula_basic() -> None:
    out = evaluate_compustat_formula(
        "(oibdpq - capxq) / xintq",
        {"oibdpq": 100.0, "capxq": 20.0, "xintq": 10.0},
    )
    assert out == pytest.approx(8.0)


def test_evaluate_compustat_formula_unknown_var_raises() -> None:
    with pytest.raises(KeyError):
        evaluate_compustat_formula("niq + xintq", {"niq": 1.0})


def test_evaluate_compustat_formula_rejects_calls() -> None:
    with pytest.raises(CompustatFormulaError):
        evaluate_compustat_formula("__import__('os').system('echo nope')", {})


def test_evaluate_compustat_formula_rejects_power() -> None:
    with pytest.raises(CompustatFormulaError):
        evaluate_compustat_formula("niq ** 2", {"niq": 2.0})
