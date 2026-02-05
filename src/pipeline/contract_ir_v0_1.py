from __future__ import annotations

import json
import re
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

import jsonschema

ANCHOR_ID_RE = re.compile(r"^A\d{4,}$")


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


SCHEMA_PATH = _project_root() / "schemas" / "contract_ir_v0_1.schema.json"


@dataclass(frozen=True)
class ContractIRValidationError:
    code: str
    message: str
    json_path: str


def load_schema() -> Dict[str, Any]:
    if not SCHEMA_PATH.exists():
        raise FileNotFoundError(f"ContractIR schema missing: {SCHEMA_PATH}")
    return json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))


def validate_contract_ir(doc: Any) -> List[ContractIRValidationError]:
    """Validate ContractIR v0.1 with hard gates.

    Hard gates (v0.1):
      - Reject any JSON that fails schema validation
      - Reject any operator not in allowlist (schema-enforced)
      - Reject untyped literals / non-string numeric literal values (schema-enforced)
      - Require provenance for every derived function and every table row (schema-enforced)
      - Require derived.semantic_role for every derived function (schema-enforced)

    Returns a list of structured errors; empty list means valid.
    """

    schema = load_schema()
    validator = jsonschema.Draft202012Validator(schema)
    errors: List[ContractIRValidationError] = []
    for err in sorted(validator.iter_errors(doc), key=lambda e: list(e.absolute_path)):
        json_path = "/" + "/".join(str(p) for p in err.absolute_path)
        errors.append(
            ContractIRValidationError(
                code="schema",
                message=err.message,
                json_path=json_path,
            )
        )

    # Additional light checks that are difficult to express cleanly in JSON Schema.
    if not errors:
        errors.extend(_validate_anchor_id_fields(doc))

    return errors


def _validate_anchor_id_fields(doc: Any) -> List[ContractIRValidationError]:
    """Ensure anchor refs follow the anchor ID convention everywhere they appear.

    This is intentionally light-weight (no semantic checks, no second pass).
    """

    out: List[ContractIRValidationError] = []

    def _walk(obj: Any, path: List[str]) -> None:
        if isinstance(obj, dict):
            for k, v in obj.items():
                if k in ("source_refs", "anchor_ids"):
                    if isinstance(v, list):
                        for idx, aid in enumerate(v):
                            if not isinstance(aid, str) or not ANCHOR_ID_RE.fullmatch(aid.strip()):
                                out.append(
                                    ContractIRValidationError(
                                        code="anchor_id",
                                        message=f"Invalid anchor id {aid!r} (expected like 'A0001')",
                                        json_path="/" + "/".join(path + [k, str(idx)]),
                                    )
                                )
                    else:
                        out.append(
                            ContractIRValidationError(
                                code="anchor_id",
                                message=f"{k} must be an array of anchor ids",
                                json_path="/" + "/".join(path + [k]),
                            )
                        )
                else:
                    _walk(v, path + [k])
        elif isinstance(obj, list):
            for i, v in enumerate(obj):
                _walk(v, path + [str(i)])

    _walk(doc, [])
    return out


# ------------------------- Evaluator -------------------------------------------------


class ContractIREvalError(Exception):
    pass


class MissingArgument(ContractIREvalError):
    def __init__(self, name: str) -> None:
        super().__init__(f"Missing argument: {name}")
        self.name = name


class MissingSeriesValue(ContractIREvalError):
    def __init__(self, series_id: str, date: str) -> None:
        super().__init__(f"Missing series value: {series_id} @ {date}")
        self.series_id = series_id
        self.date = date


class UnknownOperator(ContractIREvalError):
    def __init__(self, op: str) -> None:
        super().__init__(f"Unknown operator: {op}")
        self.op = op


class NoMatchingRow(ContractIREvalError):
    def __init__(self, table_id: str, key_column: str, key_value: str) -> None:
        super().__init__(f"No matching row in table {table_id!r} for {key_column} == {key_value!r}")
        self.table_id = table_id
        self.key_column = key_column
        self.key_value = key_value


class MultipleMatchingRows(ContractIREvalError):
    def __init__(self, table_id: str, key_column: str, key_value: str, matches: int) -> None:
        super().__init__(f"Multiple matching rows in table {table_id!r} for {key_column} == {key_value!r} ({matches})")
        self.table_id = table_id
        self.key_column = key_column
        self.key_value = key_value
        self.matches = matches


@dataclass(frozen=True)
class TypedValue:
    kind: str  # "decimal" | "rate" | "bps" | "money" | "string" | "bool" | "date"
    value: Any

    def as_decimal(self) -> Decimal:
        if self.kind not in ("decimal", "rate", "bps", "money"):
            raise TypeError(f"Expected numeric kind, got {self.kind}")
        assert isinstance(self.value, Decimal)
        return self.value

    def as_string(self) -> str:
        if self.kind != "string":
            raise TypeError(f"Expected string kind, got {self.kind}")
        assert isinstance(self.value, str)
        return self.value

    def as_bool(self) -> bool:
        if self.kind != "bool":
            raise TypeError(f"Expected bool kind, got {self.kind}")
        assert isinstance(self.value, bool)
        return self.value


def evaluate_function(
    contract_ir: Mapping[str, Any],
    *,
    fn_id: str,
    args: Mapping[str, Any],
    indices: Mapping[str, Mapping[str, str]] | None = None,
) -> TypedValue:
    """Evaluate a derived function by fn_id.

    Args:
      contract_ir: validated ContractIR document
      fn_id: function id in contract_ir["derived"]
      args: mapping from arg name -> python literal (str/bool) or decimal-string for numeric
      indices: mapping series_id -> (date -> decimal-string)
    """

    fn = None
    for rec in contract_ir.get("derived", []) or []:
        if isinstance(rec, dict) and rec.get("fn_id") == fn_id:
            fn = rec
            break
    if fn is None:
        raise KeyError(f"Function not found: {fn_id}")

    env: Dict[str, TypedValue] = {}
    for arg_def in fn.get("args", []) or []:
        if not isinstance(arg_def, dict):
            continue
        name = str(arg_def.get("name") or "")
        kind = str(arg_def.get("type") or "")
        if not name:
            continue
        if name not in args:
            raise MissingArgument(name)
        env[name] = _coerce_arg(kind, args[name])

    tables_by_id: Dict[str, Any] = {}
    for t in contract_ir.get("tables", []) or []:
        if isinstance(t, dict) and isinstance(t.get("table_id"), str):
            tables_by_id[t["table_id"]] = t

    ctx = _EvalContext(
        env=env,
        indices=indices or {},
        tables=tables_by_id,
    )
    return _eval_ast(ctx, fn.get("expr"))


def _coerce_arg(kind: str, value: Any) -> TypedValue:
    if kind in ("decimal", "rate", "bps", "money"):
        if isinstance(value, (int, float)):
            # Fail loudly: numeric args must be supplied as strings to avoid float footguns.
            raise TypeError(f"Numeric argument must be a string, got {type(value).__name__}")
        if not isinstance(value, str):
            raise TypeError(f"Numeric argument must be a string, got {type(value).__name__}")
        return TypedValue(kind=kind, value=_parse_decimal(value))
    if kind == "string":
        if not isinstance(value, str):
            raise TypeError(f"String argument must be a string, got {type(value).__name__}")
        return TypedValue(kind="string", value=value)
    if kind == "bool":
        if not isinstance(value, bool):
            raise TypeError(f"Bool argument must be a boolean, got {type(value).__name__}")
        return TypedValue(kind="bool", value=value)
    if kind == "date":
        if not isinstance(value, str):
            raise TypeError(f"Date argument must be a string, got {type(value).__name__}")
        return TypedValue(kind="date", value=value)
    raise TypeError(f"Unknown arg type: {kind}")


def _parse_decimal(s: str) -> Decimal:
    try:
        return Decimal(s)
    except InvalidOperation as exc:
        raise ValueError(f"Invalid decimal literal: {s!r}") from exc


@dataclass
class _EvalContext:
    env: Dict[str, TypedValue]
    indices: Mapping[str, Mapping[str, str]]
    tables: Mapping[str, Any]


def _eval_ast(ctx: _EvalContext, node: Any) -> TypedValue:
    if not isinstance(node, dict):
        raise TypeError(f"AST node must be an object; got {type(node).__name__}")

    if "lit" in node:
        lit = node.get("lit")
        if not isinstance(lit, dict):
            raise TypeError("lit must be an object")
        lit_type = str(lit.get("type") or "")
        value = lit.get("value")
        if lit_type in ("decimal", "rate", "bps", "money", "integer"):
            if not isinstance(value, str):
                raise TypeError("numeric literal values must be strings")
            return TypedValue(kind=("decimal" if lit_type == "integer" else lit_type), value=_parse_decimal(value))
        if lit_type == "string":
            if not isinstance(value, str):
                raise TypeError("string literal value must be a string")
            return TypedValue(kind="string", value=value)
        if lit_type == "bool":
            if not isinstance(value, bool):
                raise TypeError("bool literal value must be boolean")
            return TypedValue(kind="bool", value=value)
        if lit_type == "date":
            if not isinstance(value, str):
                raise TypeError("date literal value must be a string")
            return TypedValue(kind="date", value=value)
        raise TypeError(f"Unknown literal type: {lit_type}")

    if "var" in node:
        name = node.get("var")
        if not isinstance(name, str) or not name.strip():
            raise TypeError("var must be a non-empty string")
        if name not in ctx.env:
            raise MissingArgument(name)
        return ctx.env[name]

    if "op" in node:
        op = node.get("op")
        args = node.get("args")
        if not isinstance(op, str):
            raise TypeError("op must be a string")
        if not isinstance(args, list):
            raise TypeError("args must be a list")
        return _eval_op(ctx, op, args)

    raise TypeError("AST node must be one of: lit, var, op")


def _eval_op(ctx: _EvalContext, op: str, args: List[Any]) -> TypedValue:
    if op == "index":
        if len(args) != 2:
            raise TypeError("index(series_id, date) expects 2 args")
        series_id = _eval_ast(ctx, args[0]).as_string()
        date_val = _eval_ast(ctx, args[1])
        date = date_val.value if date_val.kind == "date" else str(date_val.value)
        series = ctx.indices.get(series_id)
        if not series or date not in series:
            raise MissingSeriesValue(series_id, date)
        return TypedValue(kind="rate", value=_parse_decimal(series[date]))

    if op in ("add", "sub", "mul", "div", "max", "min"):
        if len(args) < 2:
            raise TypeError(f"{op} expects at least 2 args")
        vals = [_eval_ast(ctx, a) for a in args]
        kinds = [v.kind for v in vals]
        nums = [v.as_decimal() for v in vals]

        def _all_same_kind(expected: str | None = None) -> bool:
            if not kinds:
                return False
            if expected is not None and any(k != expected for k in kinds):
                return False
            return all(k == kinds[0] for k in kinds)

        def _choose_kind_for_mul(a: str, b: str) -> str:
            # Minimal unit rules for early-stage feasibility testing.
            if "money" in (a, b) and "rate" in (a, b):
                return "money"
            if a == "rate" and b == "decimal":
                return "rate"
            if a == "decimal" and b == "rate":
                return "rate"
            if a == "money" and b == "decimal":
                return "money"
            if a == "decimal" and b == "money":
                return "money"
            if a == b:
                return a
            return "decimal"

        def _choose_kind_for_div(a: str, b: str) -> str:
            if a == "rate" and b == "decimal":
                return "rate"
            if a == "money" and b == "decimal":
                return "money"
            if a == b:
                return a
            return "decimal"

        def _choose_kind_for_extrema(extrema_kinds: list[str]) -> str:
            # Allow a narrow, safe set of mixed-kind comparisons for max/min to
            # reduce footguns from LLMs tagging constant floors as "decimal"
            # instead of "rate"/"money"/"bps" when the intent is unambiguous.
            allowed = set(extrema_kinds)
            if allowed <= {"rate", "decimal"}:
                return "rate" if "rate" in allowed else "decimal"
            if allowed <= {"money", "decimal"}:
                return "money" if "money" in allowed else "decimal"
            if allowed <= {"bps", "decimal"}:
                return "bps" if "bps" in allowed else "decimal"
            raise TypeError(f"max/min requires compatible numeric kinds; got {extrema_kinds}")
        if op == "add":
            if not _all_same_kind():
                raise TypeError(f"add requires matching numeric kinds; got {kinds}")
            out = sum(nums[1:], start=nums[0])
            out_kind = kinds[0]
        elif op == "sub":
            if not _all_same_kind():
                raise TypeError(f"sub requires matching numeric kinds; got {kinds}")
            out = nums[0]
            for n in nums[1:]:
                out -= n
            out_kind = kinds[0]
        elif op == "mul":
            out = nums[0]
            out_kind = kinds[0]
            for i, n in enumerate(nums[1:], start=1):
                out *= n
                out_kind = _choose_kind_for_mul(out_kind, kinds[i])
        elif op == "div":
            out = nums[0]
            out_kind = kinds[0]
            for i, n in enumerate(nums[1:], start=1):
                out /= n
                out_kind = _choose_kind_for_div(out_kind, kinds[i])
        elif op == "max":
            out = max(nums)
            out_kind = _choose_kind_for_extrema(kinds)
        else:  # min
            out = min(nums)
            out_kind = _choose_kind_for_extrema(kinds)
        return TypedValue(kind=out_kind, value=out)

    if op == "round_up_to_increment":
        if len(args) != 2:
            raise TypeError("round_up_to_increment(x, increment) expects 2 args")
        x_val = _eval_ast(ctx, args[0])
        x = x_val.as_decimal()
        inc = _eval_ast(ctx, args[1]).as_decimal()
        if inc <= 0:
            raise ValueError("increment must be positive")
        # Round upwards ("next higher") to an increment.
        q = x / inc
        out = (q.to_integral_value(rounding="ROUND_CEILING") * inc)
        return TypedValue(kind=x_val.kind, value=out)

    if op == "bps_to_rate":
        if len(args) != 1:
            raise TypeError("bps_to_rate(bps) expects 1 arg")
        bps = _eval_ast(ctx, args[0]).as_decimal()
        return TypedValue(kind="rate", value=(bps / Decimal("10000")))

    if op == "lookup":
        # lookup(table_id, key_column, key_value, value_column)
        if len(args) != 4:
            raise TypeError("lookup(table_id, key_column, key_value, value_column) expects 4 args")
        table_id = _eval_ast(ctx, args[0]).as_string()
        key_column = _eval_ast(ctx, args[1]).as_string()
        key_value = _eval_ast(ctx, args[2]).as_string()
        value_column = _eval_ast(ctx, args[3]).as_string()
        table = ctx.tables.get(table_id)
        if not isinstance(table, dict):
            raise KeyError(f"Table not found: {table_id}")
        rows = table.get("rows") or []
        matches: List[Dict[str, Any]] = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            cells = row.get("cells") or {}
            if not isinstance(cells, dict):
                continue
            key_node = cells.get(key_column)
            if not isinstance(key_node, dict):
                continue
            try:
                k_val = _eval_ast(ctx, key_node)
            except Exception:
                continue
            if k_val.kind == "string" and k_val.value == key_value:
                matches.append(row)
        if not matches:
            raise NoMatchingRow(table_id, key_column, key_value)
        if len(matches) > 1:
            raise MultipleMatchingRows(table_id, key_column, key_value, len(matches))
        cell = (matches[0].get("cells") or {}).get(value_column)
        if not isinstance(cell, dict):
            raise KeyError(f"Missing value cell {value_column!r} in table {table_id!r}")
        return _eval_ast(ctx, cell)

    if op == "lookup2":
        # lookup2(table_id, key1_column, key1_value, key2_column, key2_value, value_column)
        if len(args) != 6:
            raise TypeError(
                "lookup2(table_id, key1_column, key1_value, key2_column, key2_value, value_column) expects 6 args"
            )
        table_id = _eval_ast(ctx, args[0]).as_string()
        key1_column = _eval_ast(ctx, args[1]).as_string()
        key1_value = _eval_ast(ctx, args[2]).as_string()
        key2_column = _eval_ast(ctx, args[3]).as_string()
        key2_value = _eval_ast(ctx, args[4]).as_string()
        value_column = _eval_ast(ctx, args[5]).as_string()

        table = ctx.tables.get(table_id)
        if not isinstance(table, dict):
            raise KeyError(f"Table not found: {table_id}")

        rows = table.get("rows") or []
        matches: List[Dict[str, Any]] = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            cells = row.get("cells") or {}
            if not isinstance(cells, dict):
                continue

            key1_node = cells.get(key1_column)
            key2_node = cells.get(key2_column)
            if not isinstance(key1_node, dict) or not isinstance(key2_node, dict):
                continue
            try:
                k1_val = _eval_ast(ctx, key1_node)
                k2_val = _eval_ast(ctx, key2_node)
            except Exception:
                continue
            if k1_val.kind == "string" and k2_val.kind == "string" and k1_val.value == key1_value and k2_val.value == key2_value:
                matches.append(row)

        if not matches:
            raise NoMatchingRow(table_id, f"{key1_column}+{key2_column}", f"{key1_value}+{key2_value}")
        if len(matches) > 1:
            raise MultipleMatchingRows(table_id, f"{key1_column}+{key2_column}", f"{key1_value}+{key2_value}", len(matches))

        cell = (matches[0].get("cells") or {}).get(value_column)
        if not isinstance(cell, dict):
            raise KeyError(f"Missing value cell {value_column!r} in table {table_id!r}")
        return _eval_ast(ctx, cell)

    if op == "if":
        if len(args) != 3:
            raise TypeError("if(cond, then, else) expects 3 args")
        cond = _eval_ast(ctx, args[0]).as_bool()
        return _eval_ast(ctx, args[1] if cond else args[2])

    if op in ("eq", "lt", "lte", "gt", "gte"):
        if len(args) != 2:
            raise TypeError(f"{op} expects 2 args")
        a = _eval_ast(ctx, args[0])
        b = _eval_ast(ctx, args[1])
        if a.kind == "string" and b.kind == "string":
            av = a.value
            bv = b.value
        else:
            av = a.as_decimal()
            bv = b.as_decimal()
        if op == "eq":
            return TypedValue(kind="bool", value=(av == bv))
        if op == "lt":
            return TypedValue(kind="bool", value=(av < bv))
        if op == "lte":
            return TypedValue(kind="bool", value=(av <= bv))
        if op == "gt":
            return TypedValue(kind="bool", value=(av > bv))
        return TypedValue(kind="bool", value=(av >= bv))

    if op == "and":
        if len(args) < 2:
            raise TypeError("and expects at least 2 args")
        return TypedValue(kind="bool", value=all(_eval_ast(ctx, a).as_bool() for a in args))

    if op == "or":
        if len(args) < 2:
            raise TypeError("or expects at least 2 args")
        return TypedValue(kind="bool", value=any(_eval_ast(ctx, a).as_bool() for a in args))

    if op == "not":
        if len(args) != 1:
            raise TypeError("not expects 1 arg")
        return TypedValue(kind="bool", value=not _eval_ast(ctx, args[0]).as_bool())

    raise UnknownOperator(op)
