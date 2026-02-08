from __future__ import annotations

import ast
import operator
from typing import Mapping


class CompustatFormulaError(RuntimeError):
    pass


_BIN_OPS: dict[type[ast.operator], object] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
}

_UNARY_OPS: dict[type[ast.unaryop], object] = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}


def evaluate_compustat_formula(formula: str, values: Mapping[str, float]) -> float:
    """Evaluate a Compustat-style formula safely against a dict of variable values.

    Supported syntax:
    - Binary ops: + - * /
    - Unary ops: + -
    - Parentheses
    - Variables: names like dlttq, oibdpq, gdwlamq_fn1 (letters/digits/underscores)
    - Numeric literals (int/float)

    Disallowed:
    - Function calls
    - Attribute access
    - Subscripts
    - Any operator beyond + - * /
    """

    if not isinstance(formula, str) or not formula.strip():
        raise CompustatFormulaError("formula must be a non-empty string")

    # Normalize the provided values to be case-insensitive.
    norm_values = {str(k).lower(): float(v) for k, v in values.items()}

    try:
        expr = ast.parse(formula, mode="eval")
    except SyntaxError as exc:
        raise CompustatFormulaError(f"invalid formula syntax: {exc}") from exc

    def _eval(node: ast.AST) -> float:
        if isinstance(node, ast.Expression):
            return _eval(node.body)

        if isinstance(node, ast.Constant):
            if isinstance(node.value, (int, float)):
                return float(node.value)
            raise CompustatFormulaError("only numeric constants are allowed in compustat_formula")

        if isinstance(node, ast.Name):
            key = node.id.lower()
            if key not in norm_values:
                raise KeyError(f"missing variable value for {node.id!r}")
            return float(norm_values[key])

        if isinstance(node, ast.UnaryOp):
            op_fn = _UNARY_OPS.get(type(node.op))
            if op_fn is None:
                raise CompustatFormulaError("unsupported unary operator in compustat_formula")
            return float(op_fn(_eval(node.operand)))  # type: ignore[misc]

        if isinstance(node, ast.BinOp):
            op_fn = _BIN_OPS.get(type(node.op))
            if op_fn is None:
                raise CompustatFormulaError("unsupported binary operator in compustat_formula")
            left = _eval(node.left)
            right = _eval(node.right)
            try:
                return float(op_fn(left, right))  # type: ignore[misc]
            except ZeroDivisionError as exc:
                raise ZeroDivisionError(f"division by zero in compustat_formula: {formula!r}") from exc

        # Explicitly reject anything else (calls, attributes, subscripts, etc.).
        raise CompustatFormulaError(f"disallowed expression in compustat_formula: {type(node).__name__}")

    return _eval(expr)

