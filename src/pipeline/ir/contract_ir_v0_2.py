from __future__ import annotations

import json
import re
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

import jsonschema

ANCHOR_ID_RE = re.compile(r"^A\d{4,}$")
ISO_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
NUMERIC_LITERAL_RE = re.compile(r"^-?\d+(?:\.\d+)?$")
IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _project_root() -> Path:
    return Path(__file__).resolve().parents[3]


SCHEMA_PATH = _project_root() / "schemas" / "contract_ir_v0_2.schema.json"


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
    """Validate ContractIR v0.2 with hard gates.

    Hard gates (v0.2):
      - Reject any JSON that fails schema validation
      - Reject any operator not in allowlist (schema-enforced)
      - Reject untyped literals / non-string numeric literal values (schema-enforced)
      - Require provenance for every derived function and every table row (schema-enforced)
      - Require derived.semantic_role for every derived function (schema-enforced)
      - Require open_items[*].suggested_parameters (schema-enforced)

    Returns a list of structured errors; empty list means valid.
    """

    schema = load_schema()
    validator = jsonschema.Draft202012Validator(schema)
    errors: List[ContractIRValidationError] = []
    for err in sorted(validator.iter_errors(doc), key=lambda e: list(e.absolute_path)):
        json_path = "/" + "/".join(str(p) for p in err.absolute_path)
        msg = err.message
        # jsonschema often emits unhelpful "not valid under any of the given schemas" for oneOf/anyOf.
        # Include a few nested context messages to make LLM repairs much more likely to succeed.
        if getattr(err, "context", None):
            ctx_msgs: List[str] = []
            for sub in err.context:
                m = getattr(sub, "message", None)
                if isinstance(m, str) and m and m not in ctx_msgs:
                    ctx_msgs.append(m)
                if len(ctx_msgs) >= 3:
                    break
            if ctx_msgs:
                msg = f"{msg} (context: {'; '.join(ctx_msgs)})"
        errors.append(
            ContractIRValidationError(
                code="schema",
                message=msg,
                json_path=json_path,
            )
        )

    # Additional light checks that are difficult to express cleanly in JSON Schema.
    if not errors:
        errors.extend(_validate_anchor_id_fields(doc))
        errors.extend(_validate_source_refs_in_context(doc))
        errors.extend(_validate_bool_column_cells(doc))
        errors.extend(_validate_table_cells_are_ast_nodes(doc))
        errors.extend(_validate_table_cell_literal_value_types(doc))
        errors.extend(_validate_table_cells_match_column_types(doc))
        errors.extend(_validate_derived_arg_names(doc))
        errors.extend(_validate_expr_vars_declared(doc))
        errors.extend(_validate_index_series_declared(doc))
        errors.extend(_validate_index_call_types(doc))
        errors.extend(_validate_numeric_literal_formats(doc))
        errors.extend(_validate_rate_literal_magnitude(doc))
        errors.extend(_validate_operator_arity(doc))
        errors.extend(_validate_numeric_op_operands(doc))
        errors.extend(_validate_add_sub_kind_consistency(doc))
        errors.extend(_validate_lookup_calls(doc))
        errors.extend(_validate_lookup_range_calls(doc))
        errors.extend(_validate_lookup_rule_calls(doc))

    return errors


def _validate_table_cells_are_ast_nodes(doc: Any) -> List[ContractIRValidationError]:
    """Ensure table row cell values are AST nodes (dict) or null.

    JSON Schema currently does not validate the shape of table cell values. Without this gate,
    common model outputs like "upper_cmp": "lt" validate but fail deterministically at eval time.
    """

    if not isinstance(doc, dict):
        return []

    out: List[ContractIRValidationError] = []
    tables = doc.get("tables")
    if not isinstance(tables, list):
        return out

    for ti, table in enumerate(tables):
        if not isinstance(table, dict):
            continue
        rows = table.get("rows")
        if not isinstance(rows, list):
            continue
        for ri, row in enumerate(rows):
            if not isinstance(row, dict):
                continue
            cells = row.get("cells")
            if not isinstance(cells, dict):
                continue
            for col, cell in cells.items():
                if cell is None:
                    continue
                if not isinstance(cell, dict):
                    out.append(
                        ContractIRValidationError(
                            code="table_cell_not_ast",
                            message=f"Table cell {col!r} must be an AST node object or null; got {type(cell).__name__}",
                            json_path=f"/tables/{ti}/rows/{ri}/cells/{col}",
                        )
                    )
                    continue
                if not ("lit" in cell or "var" in cell or "op" in cell):
                    out.append(
                        ContractIRValidationError(
                            code="table_cell_not_ast",
                            message=f"Table cell {col!r} must be an AST node with one of: lit, var, op",
                            json_path=f"/tables/{ti}/rows/{ri}/cells/{col}",
                        )
                    )
    return out


def _validate_table_cell_literal_value_types(doc: Any) -> List[ContractIRValidationError]:
    """Validate literal value types inside table cells (schema does not validate table cell ASTs).

    Examples of failure modes this catches:
      - {"lit":{"type":"bool","value":"true"}}  (value must be boolean)
      - {"lit":{"type":"bool","value":{...}}}   (value must be boolean, not nested AST)
      - {"lit":{"type":"date","value":123}}     (value must be ISO date string)
    """

    if not isinstance(doc, dict):
        return []

    out: List[ContractIRValidationError] = []
    tables = doc.get("tables")
    if not isinstance(tables, list):
        return out

    for ti, table in enumerate(tables):
        if not isinstance(table, dict):
            continue
        rows = table.get("rows")
        if not isinstance(rows, list):
            continue
        for ri, row in enumerate(rows):
            if not isinstance(row, dict):
                continue
            cells = row.get("cells")
            if not isinstance(cells, dict):
                continue
            for col, cell in cells.items():
                if not isinstance(cell, dict):
                    continue
                lit = cell.get("lit")
                if not isinstance(lit, dict):
                    continue
                lit_type = lit.get("type")
                value = lit.get("value")
                if lit_type == "string" and not isinstance(value, str):
                    out.append(
                        ContractIRValidationError(
                            code="literal_value_type",
                            message=f"Invalid string literal value type {type(value).__name__}; expected string",
                            json_path=f"/tables/{ti}/rows/{ri}/cells/{col}/lit/value",
                        )
                    )
                if lit_type == "bool" and not isinstance(value, bool):
                    out.append(
                        ContractIRValidationError(
                            code="literal_value_type",
                            message=f"Invalid bool literal value type {type(value).__name__}; expected boolean",
                            json_path=f"/tables/{ti}/rows/{ri}/cells/{col}/lit/value",
                        )
                    )
                if lit_type == "date":
                    if not isinstance(value, str) or not ISO_DATE_RE.fullmatch(value.strip()):
                        out.append(
                            ContractIRValidationError(
                                code="literal_value_type",
                                message="Invalid date literal; expected ISO YYYY-MM-DD string",
                                json_path=f"/tables/{ti}/rows/{ri}/cells/{col}/lit/value",
                            )
                        )

    return out


def _validate_table_cells_match_column_types(doc: Any) -> List[ContractIRValidationError]:
    """Enforce that table cells respect their declared column types.

    Policy (v0.2):
      - For non-bool columns (string/decimal/rate/bps/money/integer/date), cells must be:
          * null, OR
          * a literal AST node ({"lit": {"type": <col_type>, "value": ...}})
        (decimal columns may also use integer literals.)
      - For bool columns, cells may be any AST node (lit bool or a boolean expression AST),
        to support lookup_rule predicate tables.

    Motivation: models sometimes embed expressions (e.g. lookup_rule or comparisons) inside string key columns,
    which validates structurally but is not a stable/evaluatable representation of a lookup table.
    """

    if not isinstance(doc, dict):
        return []

    out: List[ContractIRValidationError] = []
    tables = doc.get("tables")
    if not isinstance(tables, list):
        return out

    for ti, table in enumerate(tables):
        if not isinstance(table, dict):
            continue
        table_id = table.get("table_id")
        col_types: Dict[str, str] = {}
        for c in table.get("columns", []) or []:
            if not isinstance(c, dict):
                continue
            name = c.get("name")
            typ = c.get("type")
            if isinstance(name, str) and isinstance(typ, str):
                col_types[name] = typ

        rows = table.get("rows")
        if not isinstance(rows, list):
            continue
        for ri, row in enumerate(rows):
            if not isinstance(row, dict):
                continue
            cells = row.get("cells")
            if not isinstance(cells, dict):
                continue
            for col, cell in cells.items():
                expected = col_types.get(col)
                if not isinstance(expected, str) or not expected:
                    continue
                if cell is None:
                    continue
                if not isinstance(cell, dict):
                    # table_cell_not_ast covers the type; skip duplicate messaging.
                    continue
                if expected == "bool":
                    continue
                lit = cell.get("lit")
                if not isinstance(lit, dict):
                    out.append(
                        ContractIRValidationError(
                            code="table_cell_not_literal",
                            message=(
                                f"Table {table_id!r} column {col!r} is type {expected!r} and must be a literal AST node (or null); "
                                "only bool columns may contain expressions (for lookup_rule predicates)"
                            ),
                            json_path=f"/tables/{ti}/rows/{ri}/cells/{col}",
                        )
                    )
                    continue
                lit_type = lit.get("type")
                if expected == "decimal" and lit_type in ("decimal", "integer"):
                    continue
                if lit_type != expected:
                    out.append(
                        ContractIRValidationError(
                            code="table_cell_literal_type_mismatch",
                            message=(
                                f"Table {table_id!r} column {col!r} is type {expected!r} but cell literal has type {lit_type!r}"
                            ),
                            json_path=f"/tables/{ti}/rows/{ri}/cells/{col}/lit/type",
                        )
                    )

    return out


def _validate_derived_arg_names(doc: Any) -> List[ContractIRValidationError]:
    """Ensure derived args use stable identifier-like names.

    Motivation: models sometimes emit arg names with spaces ("Index Rate") which makes downstream
    wiring brittle and is almost always accidental.
    """

    if not isinstance(doc, dict):
        return []

    out: List[ContractIRValidationError] = []
    derived = doc.get("derived")
    if not isinstance(derived, list):
        return out

    for di, fn in enumerate(derived):
        if not isinstance(fn, dict):
            continue
        args = fn.get("args") or []
        if not isinstance(args, list):
            continue
        for ai, a in enumerate(args):
            if not isinstance(a, dict):
                continue
            name = a.get("name")
            if not isinstance(name, str) or not IDENTIFIER_RE.fullmatch(name.strip()):
                out.append(
                    ContractIRValidationError(
                        code="arg_name_invalid",
                        message=(
                            "derived.args[*].name must be an identifier like 'as_of_date' (letters/digits/underscore; no spaces)"
                        ),
                        json_path=f"/derived/{di}/args/{ai}/name",
                    )
                )

    return out


def _validate_numeric_op_operands(doc: Any) -> List[ContractIRValidationError]:
    """Reject obvious non-numeric operands in numeric operators (prevents eval-time TypeError).

    This is intentionally conservative: it only fires when an operand is *definitely* non-numeric
    (e.g., a string literal or a var declared type=string).
    """

    if not isinstance(doc, dict):
        return []

    numeric_ops = {"add", "sub", "mul", "div", "max", "min", "round_up_to_increment", "bps_to_rate"}
    bool_ops = {"eq", "lt", "lte", "gt", "gte", "and", "or", "not"}
    numeric_types = {"rate", "bps", "decimal", "money", "integer"}
    non_numeric_types = {"string", "bool", "date"}

    out: List[ContractIRValidationError] = []

    for di, fn in _iter_derived(doc):
        # Build var -> type map for this derived function.
        arg_types: Dict[str, str] = {}
        for a in fn.get("args", []) or []:
            if isinstance(a, dict) and isinstance(a.get("name"), str) and isinstance(a.get("type"), str):
                arg_types[a["name"]] = a["type"]

        expr = fn.get("expr")
        for path, n in _iter_ast_nodes(expr):
            op = n.get("op")
            if op not in numeric_ops:
                continue
            args = n.get("args")
            if not isinstance(args, list):
                continue

            def _definitely_non_numeric(node: Any) -> Optional[str]:
                if not isinstance(node, dict):
                    return None
                if "lit" in node:
                    lit = node.get("lit")
                    if isinstance(lit, dict) and isinstance(lit.get("type"), str):
                        lt = lit["type"]
                        if lt in non_numeric_types:
                            return lt
                        return None
                if "var" in node and isinstance(node.get("var"), str):
                    vt = arg_types.get(node["var"])
                    if vt in non_numeric_types:
                        return vt
                    return None
                if "op" in node and isinstance(node.get("op"), str):
                    # Boolean-valued ops used in numeric context are always wrong.
                    if node["op"] in bool_ops:
                        return "bool_expr"
                return None

            for ai, a in enumerate(args):
                bad = _definitely_non_numeric(a)
                if bad is None:
                    continue
                out.append(
                    ContractIRValidationError(
                        code="numeric_op_arg_not_numeric",
                        message=f"{op} expects numeric operands; got {bad!r}",
                        json_path="/" + "/".join(["derived", str(di), "expr"] + path + ["args", str(ai)]),
                    )
                )

    return out


def _iter_ast_nodes(node: Any) -> Iterable[Tuple[List[str], Dict[str, Any]]]:
    """Yield (json_path_parts, ast_node) pairs for AST nodes within a subtree."""

    out: List[Tuple[List[str], Dict[str, Any]]] = []

    def _walk(obj: Any, path: List[str]) -> None:
        if isinstance(obj, dict):
            # AST nodes are always dicts with one of these keys.
            if "lit" in obj or "var" in obj or "op" in obj:
                out.append((path, obj))
            for k, v in obj.items():
                if isinstance(v, (dict, list)):
                    _walk(v, path + [k])
        elif isinstance(obj, list):
            for i, v in enumerate(obj):
                if isinstance(v, (dict, list)):
                    _walk(v, path + [str(i)])

    _walk(node, [])
    return out


def _tables_by_id(doc: Mapping[str, Any]) -> Dict[str, Dict[str, Any]]:
    tables: Dict[str, Dict[str, Any]] = {}
    for t in doc.get("tables", []) or []:
        if isinstance(t, dict) and isinstance(t.get("table_id"), str) and t.get("table_id"):
            tables[t["table_id"]] = t
    return tables


def _table_col_types(table: Mapping[str, Any]) -> Dict[str, str]:
    cols: Dict[str, str] = {}
    for c in table.get("columns", []) or []:
        if not isinstance(c, dict):
            continue
        name = c.get("name")
        typ = c.get("type")
        if isinstance(name, str) and isinstance(typ, str):
            cols[name] = typ
    return cols


def _string_lit(node: Any) -> Optional[str]:
    if not isinstance(node, dict):
        return None
    lit = node.get("lit")
    if not isinstance(lit, dict):
        return None
    if lit.get("type") != "string":
        return None
    val = lit.get("value")
    return val if isinstance(val, str) else None


def _iter_derived(doc: Mapping[str, Any]) -> Iterable[Tuple[int, Dict[str, Any]]]:
    derived = doc.get("derived")
    if not isinstance(derived, list):
        return []
    out: List[Tuple[int, Dict[str, Any]]] = []
    for i, fn in enumerate(derived):
        if isinstance(fn, dict):
            out.append((i, fn))
    return out


def _referenced_table_ids(expr: Any) -> List[str]:
    table_ops = {"lookup", "lookup2", "lookup_range", "lookup_rule"}
    ids: List[str] = []
    for _, n in _iter_ast_nodes(expr):
        op = n.get("op")
        if op not in table_ops:
            continue
        args = n.get("args")
        if not isinstance(args, list) or not args:
            continue
        tid = _string_lit(args[0])
        if isinstance(tid, str) and tid and tid not in ids:
            ids.append(tid)
    return ids


def _collect_var_names_in_ast(expr: Any) -> List[str]:
    names: List[str] = []
    for _, n in _iter_ast_nodes(expr):
        if "var" in n:
            v = n.get("var")
            if isinstance(v, str) and v.strip() and v not in names:
                names.append(v)
    return names


def _validate_expr_vars_declared(doc: Any) -> List[ContractIRValidationError]:
    """Ensure that every {"var":"x"} referenced in an expression is declared as an argument.

    Motivation:
      - JSON Schema cannot express "vars in expr must be in args".
      - Without this, artifacts can validate but fail deterministically at eval-time (MissingArgument).
    """

    if not isinstance(doc, dict):
        return []

    tables = _tables_by_id(doc)
    out: List[ContractIRValidationError] = []

    for di, fn in _iter_derived(doc):
        arg_names = set()
        for a in fn.get("args", []) or []:
            if isinstance(a, dict) and isinstance(a.get("name"), str) and a.get("name"):
                arg_names.add(a["name"])

        expr = fn.get("expr")
        for var_name in _collect_var_names_in_ast(expr):
            if var_name not in arg_names:
                out.append(
                    ContractIRValidationError(
                        code="var_not_declared",
                        message=f"Expression references var {var_name!r} not present in derived.args",
                        json_path=f"/derived/{di}/args",
                    )
                )

        # lookup_rule predicates live inside tables; validate vars there too for referenced tables.
        for table_id in _referenced_table_ids(expr):
            table = tables.get(table_id)
            if not table:
                continue
            for ri, row in enumerate(table.get("rows", []) or []):
                if not isinstance(row, dict):
                    continue
                cells = row.get("cells")
                if not isinstance(cells, dict):
                    continue
                for col_name, cell in cells.items():
                    # Only AST dicts can have vars.
                    for var_name in _collect_var_names_in_ast(cell):
                        if var_name not in arg_names:
                            out.append(
                                ContractIRValidationError(
                                    code="var_not_declared",
                                    message=(
                                        f"Table {table_id!r} cell {col_name!r} references var {var_name!r} "
                                        "not present in derived.args"
                                    ),
                                    json_path=f"/tables/{table_id}/rows/{ri}/cells/{col_name}",
                                )
                            )
    return out


def _validate_index_series_declared(doc: Any) -> List[ContractIRValidationError]:
    """Ensure that any index(series_id, ...) call references a series declared in contract_ir['indices']."""

    if not isinstance(doc, dict):
        return []

    declared: set[str] = set()
    for idx in doc.get("indices", []) or []:
        if isinstance(idx, dict) and isinstance(idx.get("series_id"), str) and idx.get("series_id"):
            declared.add(idx["series_id"])

    out: List[ContractIRValidationError] = []

    for di, fn in _iter_derived(doc):
        expr = fn.get("expr")
        for path, n in _iter_ast_nodes(expr):
            if n.get("op") != "index":
                continue
            args = n.get("args")
            if not isinstance(args, list) or len(args) != 2:
                continue
            series_id = _string_lit(args[0])
            if isinstance(series_id, str) and series_id and series_id not in declared:
                out.append(
                    ContractIRValidationError(
                        code="index_series_not_declared",
                        message=f"index() references series_id {series_id!r} not present in contract_ir.indices",
                        json_path="/" + "/".join(["derived", str(di), "expr"] + path + ["args", "0", "lit", "value"]),
                    )
                )
    return out


def _validate_index_call_types(doc: Any) -> List[ContractIRValidationError]:
    """Validate common index() call constraints that schema does not express.

    - index(series_id, date) arity is validated elsewhere.
    - series_id MUST be a string literal (stable wiring).
    - date arg MUST be a variable declared as type=date (no hardcoded date literals).

    Motivation: prevents eval-time errors like MissingSeriesValue due to passing arbitrary strings for date.
    """

    if not isinstance(doc, dict):
        return []

    out: List[ContractIRValidationError] = []

    for di, fn in _iter_derived(doc):
        # Build var -> type map for this derived function.
        arg_types: Dict[str, str] = {}
        for a in fn.get("args", []) or []:
            if isinstance(a, dict) and isinstance(a.get("name"), str) and isinstance(a.get("type"), str):
                arg_types[a["name"]] = a["type"]

        expr = fn.get("expr")
        for path, n in _iter_ast_nodes(expr):
            if n.get("op") != "index":
                continue
            args = n.get("args")
            if not isinstance(args, list) or len(args) != 2:
                continue

            series_id = _string_lit(args[0])
            if series_id is None:
                out.append(
                    ContractIRValidationError(
                        code="index_series_id_not_literal",
                        message="index() series_id must be an AST string literal node",
                        json_path="/" + "/".join(["derived", str(di), "expr"] + path + ["args", "0"]),
                    )
                )

            date_arg = args[1]
            if isinstance(date_arg, dict) and "lit" in date_arg:
                lit = date_arg.get("lit")
                if isinstance(lit, dict) and lit.get("type") == "date":
                    out.append(
                        ContractIRValidationError(
                            code="index_date_arg_not_var",
                            message="index() date arg must be a date var (type=date), not a hardcoded date literal",
                            json_path="/" + "/".join(["derived", str(di), "expr"] + path + ["args", "1", "lit", "value"]),
                        )
                    )
                    continue
                out.append(
                    ContractIRValidationError(
                        code="index_date_arg_not_date",
                        message="index() date arg must be a var declared as type=date",
                        json_path="/" + "/".join(["derived", str(di), "expr"] + path + ["args", "1", "lit", "type"]),
                    )
                )
                continue

            if isinstance(date_arg, dict) and "var" in date_arg and isinstance(date_arg.get("var"), str):
                var_name = date_arg["var"]
                var_type = arg_types.get(var_name)
                if var_type == "date":
                    continue
                # If the var isn't declared at all, var_not_declared will fire; keep this as a type-mismatch only.
                if var_type is not None:
                    out.append(
                        ContractIRValidationError(
                            code="index_date_arg_not_date",
                            message=f"index() date var {var_name!r} must be declared as type=date (got {var_type!r})",
                            json_path="/" + "/".join(["derived", str(di), "expr"] + path + ["args", "1", "var"]),
                        )
                    )
                continue

            # Other expressions for dates are not supported in v0.2 (fail loudly).
            out.append(
                ContractIRValidationError(
                    code="index_date_arg_not_date",
                    message="index() date arg must be a var declared as type=date",
                    json_path="/" + "/".join(["derived", str(di), "expr"] + path + ["args", "1"]),
                )
            )

    return out


def _validate_numeric_literal_formats(doc: Any) -> List[ContractIRValidationError]:
    """Ensure numeric literal string values are parseable by Decimal (no '%', '$', commas, etc.).

    This is intentionally strict so evaluation is deterministic and caller-independent.
    """

    if not isinstance(doc, dict):
        return []

    out: List[ContractIRValidationError] = []

    def _check_lit(*, lit_type: str, value: Any, json_path: str) -> None:
        if lit_type not in ("decimal", "rate", "bps", "money", "integer"):
            return
        if value is None:
            out.append(
                ContractIRValidationError(
                    code="numeric_literal_value_type",
                    message=(
                        f"Invalid {lit_type} literal value None; expected a plain decimal string like '0.005' "
                        "(for missing bounds, set the CELL to null, not lit.value)"
                    ),
                    json_path=json_path,
                )
            )
            return
        if not isinstance(value, str):
            out.append(
                ContractIRValidationError(
                    code="numeric_literal_value_type",
                    message=(
                        f"Invalid {lit_type} literal value type {type(value).__name__}; expected a plain decimal string like '0.005'"
                    ),
                    json_path=json_path,
                )
            )
            return
        if not NUMERIC_LITERAL_RE.fullmatch(value.strip()):
            out.append(
                ContractIRValidationError(
                    code="numeric_literal_format",
                    message=f"Invalid {lit_type} literal {value!r}; expected a plain decimal string like '0.005'",
                    json_path=json_path,
                )
            )

    # Table cells
    for ti, table in enumerate(doc.get("tables", []) or []):
        if not isinstance(table, dict):
            continue
        for ri, row in enumerate(table.get("rows", []) or []):
            if not isinstance(row, dict):
                continue
            cells = row.get("cells")
            if not isinstance(cells, dict):
                continue
            for col, cell in cells.items():
                for path, n in _iter_ast_nodes(cell):
                    lit = n.get("lit")
                    if not isinstance(lit, dict):
                        continue
                    lit_type = str(lit.get("type") or "")
                    _check_lit(
                        lit_type=lit_type,
                        value=lit.get("value"),
                        json_path="/" + "/".join(["tables", str(ti), "rows", str(ri), "cells", str(col)] + path + ["lit", "value"]),
                    )

    # Derived exprs
    for di, fn in _iter_derived(doc):
        expr = fn.get("expr")
        for path, n in _iter_ast_nodes(expr):
            lit = n.get("lit")
            if not isinstance(lit, dict):
                continue
            lit_type = str(lit.get("type") or "")
            _check_lit(
                lit_type=lit_type,
                value=lit.get("value"),
                json_path="/" + "/".join(["derived", str(di), "expr"] + path + ["lit", "value"]),
            )

    return out


def _validate_lookup_calls(doc: Any) -> List[ContractIRValidationError]:
    """Validate lookup/lookup2/lookup_range call sites against table schemas."""

    if not isinstance(doc, dict):
        return []

    tables = _tables_by_id(doc)
    out: List[ContractIRValidationError] = []
    checked_value_cells: set[tuple[str, str]] = set()

    def _looks_like_condition_string(s: str) -> bool:
        t = str(s).lower()
        if not t.strip():
            return False
        # Common patterns we want to reject for string-keyed lookup tables:
        #  - "< 1.75 : 1", ">= 2.50 : 1", "greater than 3.00x"
        #  - conjunction/disjunction of such conditions.
        has_digit = bool(re.search(r"\d", t))
        has_ineq = any(sym in t for sym in ("<", ">", "≤", "≥"))
        has_ratio = bool(re.search(r":\s*1\b", t))
        has_words = ("less than" in t) or ("greater than" in t) or ("at least" in t) or ("not less than" in t)
        has_logic = (" and " in t) or (" or " in t)
        return (has_digit and (has_ineq or has_words)) or (has_ratio and has_digit) or (has_logic and has_ineq and has_digit)

    def _iter_string_cell_values(*, table: Mapping[str, Any], col: str) -> Iterable[Tuple[int, str]]:
        rows = table.get("rows") or []
        if not isinstance(rows, list):
            return
        for ri, row in enumerate(rows):
            if not isinstance(row, dict):
                continue
            cells = row.get("cells")
            if not isinstance(cells, dict):
                continue
            cell = cells.get(col)
            v = _string_lit(cell)
            if isinstance(v, str) and v.strip():
                yield ri, v

    def _walk_expr(expr: Any, *, path_prefix: List[str], arg_types: Mapping[str, str]) -> None:
        for path, n in _iter_ast_nodes(expr):
            op = n.get("op")
            if op not in ("lookup", "lookup2", "lookup_range"):
                continue
            args = n.get("args")
            if not isinstance(args, list):
                continue
            table_id = _string_lit(args[0]) if len(args) >= 1 else None
            if not isinstance(table_id, str) or not table_id:
                continue
            table = tables.get(table_id)
            if table is None:
                out.append(
                    ContractIRValidationError(
                        code="lookup_table_missing",
                        message=f"{op} references missing table_id {table_id!r}",
                        json_path="/" + "/".join(path_prefix + path + ["args", "0", "lit", "value"]),
                    )
                )
                continue
            cols = _table_col_types(table)

            def _require_rows_have_value_cell(*, value_col: Optional[str]) -> None:
                if not isinstance(value_col, str) or not value_col:
                    return
                key = (table_id, value_col)
                if key in checked_value_cells:
                    return
                checked_value_cells.add(key)
                rows = table.get("rows") or []
                if not isinstance(rows, list):
                    return
                for ri, row in enumerate(rows):
                    if not isinstance(row, dict):
                        continue
                    cells = row.get("cells")
                    if not isinstance(cells, dict):
                        continue
                    cell = cells.get(value_col)
                    if not isinstance(cell, dict):
                        out.append(
                            ContractIRValidationError(
                                code="lookup_value_cell_missing",
                                message=f"Table {table_id!r} row {ri} is missing required value cell {value_col!r}",
                                json_path=f"/tables/{table_id}/rows/{ri}/cells/{value_col}",
                            )
                        )

            def _require_string_lit_arg(arg_idx: int, *, label: str) -> Optional[str]:
                if len(args) <= arg_idx:
                    return None
                val = _string_lit(args[arg_idx])
                if not isinstance(val, str) or not val:
                    out.append(
                        ContractIRValidationError(
                            code="lookup_arg_not_string_literal",
                            message=f"{op} {label} must be an AST string literal node",
                            json_path="/" + "/".join(path_prefix + path + ["args", str(arg_idx)]),
                        )
                    )
                    return None
                return val

            def _require_col(col_lit_idx: int, *, err_code: str) -> None:
                if len(args) <= col_lit_idx:
                    return
                col_name = _string_lit(args[col_lit_idx])
                if not isinstance(col_name, str) or not col_name:
                    return
                if col_name not in cols:
                    out.append(
                        ContractIRValidationError(
                            code=err_code,
                            message=f"{op} references column {col_name!r} not declared in table {table_id!r}",
                            json_path="/" + "/".join(path_prefix + path + ["args", str(col_lit_idx), "lit", "value"]),
                        )
                    )

            def _var_name(node: Any) -> Optional[str]:
                if not isinstance(node, dict):
                    return None
                v = node.get("var")
                return v if isinstance(v, str) and v else None

            def _lit_type(node: Any) -> Optional[str]:
                if not isinstance(node, dict):
                    return None
                lit = node.get("lit")
                if not isinstance(lit, dict):
                    return None
                t = lit.get("type")
                return t if isinstance(t, str) else None

            if op == "lookup" and len(args) == 4:
                _require_string_lit_arg(0, label="table_id")
                _require_string_lit_arg(1, label="key_column")
                _require_string_lit_arg(3, label="value_column")
                _require_col(1, err_code="lookup_key_col_missing")
                _require_col(3, err_code="lookup_value_col_missing")
                _require_rows_have_value_cell(value_col=_string_lit(args[3]))

                key_col = _string_lit(args[1])
                if isinstance(key_col, str) and key_col and cols.get(key_col) not in (None, "string"):
                    out.append(
                        ContractIRValidationError(
                            code="lookup_key_col_not_string",
                            message=f"lookup key_column {key_col!r} must be type 'string' in table {table_id!r}",
                            json_path="/" + "/".join(path_prefix + path + ["args", "1", "lit", "value"]),
                        )
                    )

                # Reject common model failure mode: encoding numeric threshold logic into string-keyed lookup tables.
                if isinstance(key_col, str) and key_col:
                    if "condition" in key_col.lower():
                        out.append(
                            ContractIRValidationError(
                                code="lookup_key_is_condition_string",
                                message=(
                                    f"lookup key_column {key_col!r} appears to be a condition label; "
                                    "do not encode numeric thresholds as string keys. Use lookup_range for numeric ranges "
                                    "or lookup_rule for boolean predicates."
                                ),
                                json_path="/" + "/".join(path_prefix + path + ["args", "1", "lit", "value"]),
                            )
                        )
                    for ri, v in _iter_string_cell_values(table=table, col=key_col):
                        if _looks_like_condition_string(v):
                            out.append(
                                ContractIRValidationError(
                                    code="lookup_key_is_condition_string",
                                    message=(
                                        f"lookup table {table_id!r} key cell looks like a numeric condition {v!r}; "
                                        "do not encode inequality/range logic as string keys. Use lookup_range/lookup_rule instead."
                                    ),
                                    json_path=f"/tables/{table_id}/rows/{ri}/cells/{key_col}/lit/value",
                                )
                            )
                            break

                key_val_var = _var_name(args[2])
                if key_val_var is not None:
                    vtype = arg_types.get(key_val_var)
                    if vtype is not None and vtype != "string":
                        out.append(
                            ContractIRValidationError(
                                code="lookup_key_value_not_string",
                                message=(
                                    f"lookup key_value var {key_val_var!r} must be declared as type=string (got {vtype!r}); "
                                    "use lookup_range for numeric schedules"
                                ),
                                json_path="/" + "/".join(path_prefix + path + ["args", "2", "var"]),
                            )
                        )
                else:
                    ltype = _lit_type(args[2])
                    if ltype is not None and ltype != "string":
                        out.append(
                            ContractIRValidationError(
                                code="lookup_key_value_not_string",
                                message=(
                                    f"lookup key_value literal must be type=string (got {ltype!r}); use lookup_range for numeric schedules"
                                ),
                                json_path="/" + "/".join(path_prefix + path + ["args", "2", "lit", "type"]),
                            )
                        )

                key_col = _string_lit(args[1])
                key_val = _string_lit(args[2])
                if isinstance(key_col, str) and isinstance(key_val, str) and key_col and key_val and key_col == key_val:
                    out.append(
                        ContractIRValidationError(
                            code="lookup_key_value_is_column_name",
                            message=(
                                f"lookup key_value is the literal column name {key_val!r}; expected a variable (or a real key string)"
                            ),
                            json_path="/" + "/".join(path_prefix + path + ["args", "2"]),
                        )
                    )
            if op == "lookup2" and len(args) == 6:
                _require_string_lit_arg(0, label="table_id")
                _require_string_lit_arg(1, label="key1_column")
                _require_string_lit_arg(3, label="key2_column")
                _require_string_lit_arg(5, label="value_column")
                _require_col(1, err_code="lookup2_key1_col_missing")
                _require_col(3, err_code="lookup2_key2_col_missing")
                _require_col(5, err_code="lookup2_value_col_missing")
                _require_rows_have_value_cell(value_col=_string_lit(args[5]))

                key1_col = _string_lit(args[1])
                if isinstance(key1_col, str) and key1_col and cols.get(key1_col) not in (None, "string"):
                    out.append(
                        ContractIRValidationError(
                            code="lookup2_key_col_not_string",
                            message=f"lookup2 key1_column {key1_col!r} must be type 'string' in table {table_id!r}",
                            json_path="/" + "/".join(path_prefix + path + ["args", "1", "lit", "value"]),
                        )
                    )
                key2_col = _string_lit(args[3])
                if isinstance(key2_col, str) and key2_col and cols.get(key2_col) not in (None, "string"):
                    out.append(
                        ContractIRValidationError(
                            code="lookup2_key_col_not_string",
                            message=f"lookup2 key2_column {key2_col!r} must be type 'string' in table {table_id!r}",
                            json_path="/" + "/".join(path_prefix + path + ["args", "3", "lit", "value"]),
                        )
                    )

                # Reject common model failure mode: encoding numeric threshold logic into string-keyed lookup2 tables.
                # For pricing schedules keyed by ratios (ICR/FCCR/leverage), the keys must be numeric inputs (lookup_range)
                # or boolean predicates (lookup_rule), not free-form condition strings.
                for arg_idx, col_name in ((1, key1_col), (3, key2_col)):
                    if not isinstance(col_name, str) or not col_name:
                        continue
                    if "condition" in col_name.lower():
                        out.append(
                            ContractIRValidationError(
                                code="lookup_key_is_condition_string",
                                message=(
                                    f"lookup2 key_column {col_name!r} appears to be a condition label; "
                                    "do not encode numeric thresholds as string keys. Use lookup_range for numeric ranges "
                                    "or lookup_rule for boolean predicates."
                                ),
                                json_path="/" + "/".join(path_prefix + path + ["args", str(arg_idx), "lit", "value"]),
                            )
                        )
                    for ri, v in _iter_string_cell_values(table=table, col=col_name):
                        if _looks_like_condition_string(v):
                            out.append(
                                ContractIRValidationError(
                                    code="lookup_key_is_condition_string",
                                    message=(
                                        f"lookup2 table {table_id!r} key cell looks like a numeric condition {v!r}; "
                                        "do not encode inequality/range logic as string keys. Use lookup_range/lookup_rule instead."
                                    ),
                                    json_path=f"/tables/{table_id}/rows/{ri}/cells/{col_name}/lit/value",
                                )
                            )
                            break

                for arg_idx in (2, 4):
                    key_var = _var_name(args[arg_idx])
                    if key_var is not None:
                        vtype = arg_types.get(key_var)
                        if vtype is not None and vtype != "string":
                            out.append(
                                ContractIRValidationError(
                                    code="lookup2_key_value_not_string",
                                    message=(
                                        f"lookup2 key value var {key_var!r} must be declared as type=string (got {vtype!r}); "
                                        "use lookup_range for numeric schedules"
                                    ),
                                    json_path="/" + "/".join(path_prefix + path + ["args", str(arg_idx), "var"]),
                                )
                            )
                    else:
                        ltype = _lit_type(args[arg_idx])
                        if ltype is not None and ltype != "string":
                            out.append(
                                ContractIRValidationError(
                                    code="lookup2_key_value_not_string",
                                    message=(
                                        f"lookup2 key value literal must be type=string (got {ltype!r}); use lookup_range for numeric schedules"
                                    ),
                                    json_path="/" + "/".join(path_prefix + path + ["args", str(arg_idx), "lit", "type"]),
                                )
                            )

                key1_col = _string_lit(args[1])
                key1_val = _string_lit(args[2])
                if (
                    isinstance(key1_col, str)
                    and isinstance(key1_val, str)
                    and key1_col
                    and key1_val
                    and key1_col == key1_val
                ):
                    out.append(
                        ContractIRValidationError(
                            code="lookup_key_value_is_column_name",
                            message=(
                                f"lookup2 key1_value is the literal column name {key1_val!r}; expected a variable (or a real key string)"
                            ),
                            json_path="/" + "/".join(path_prefix + path + ["args", "2"]),
                        )
                    )
                key2_val = _string_lit(args[4])
                if (
                    isinstance(key2_col, str)
                    and isinstance(key2_val, str)
                    and key2_col
                    and key2_val
                    and key2_col == key2_val
                ):
                    out.append(
                        ContractIRValidationError(
                            code="lookup_key_value_is_column_name",
                            message=(
                                f"lookup2 key2_value is the literal column name {key2_val!r}; expected a variable (or a real key string)"
                            ),
                            json_path="/" + "/".join(path_prefix + path + ["args", "4"]),
                        )
                    )
            if op == "lookup_range" and len(args) == 7:
                _require_string_lit_arg(0, label="table_id")
                _require_string_lit_arg(2, label="lower_bound_col")
                _require_string_lit_arg(3, label="lower_cmp_col")
                _require_string_lit_arg(4, label="upper_bound_col")
                _require_string_lit_arg(5, label="upper_cmp_col")
                _require_string_lit_arg(6, label="value_col")
                _require_col(2, err_code="lookup_range_lower_col_missing")
                _require_col(3, err_code="lookup_range_lower_cmp_col_missing")
                _require_col(4, err_code="lookup_range_upper_col_missing")
                _require_col(5, err_code="lookup_range_upper_cmp_col_missing")
                _require_col(6, err_code="lookup_range_value_col_missing")
                _require_rows_have_value_cell(value_col=_string_lit(args[6]))

    for di, fn in _iter_derived(doc):
        arg_types: Dict[str, str] = {}
        for a in fn.get("args", []) or []:
            if isinstance(a, dict) and isinstance(a.get("name"), str) and isinstance(a.get("type"), str):
                arg_types[a["name"]] = a["type"]
        _walk_expr(fn.get("expr"), path_prefix=["derived", str(di), "expr"], arg_types=arg_types)

    return out


def _validate_lookup_range_calls(doc: Any) -> List[ContractIRValidationError]:
    """Validate lookup_range tables beyond schema-level checks.

    - The comparator columns referenced by lookup_range MUST be declared as type=string.
    - When a row provides a bound, the corresponding comparator cell MUST exist and be one of:
        lower: "gt" | "gte"
        upper: "lt" | "lte"

    Motivation: catch common model errors like encoding comparator columns as numeric flags ("gte": 1),
    which validate but are not evaluatable.
    """

    if not isinstance(doc, dict):
        return []

    tables = _tables_by_id(doc)
    out: List[ContractIRValidationError] = []

    for di, fn in _iter_derived(doc):
        arg_types: Dict[str, str] = {}
        for a in fn.get("args", []) or []:
            if isinstance(a, dict) and isinstance(a.get("name"), str) and isinstance(a.get("type"), str):
                arg_types[a["name"]] = a["type"]

        expr = fn.get("expr")
        for path, n in _iter_ast_nodes(expr):
            if n.get("op") != "lookup_range":
                continue
            args = n.get("args")
            if not isinstance(args, list) or len(args) != 7:
                # arity validation handles the rest
                continue

            table_id = _string_lit(args[0])
            key_node = args[1]
            lower_bound_col = _string_lit(args[2])
            lower_cmp_col = _string_lit(args[3])
            upper_bound_col = _string_lit(args[4])
            upper_cmp_col = _string_lit(args[5])

            if not isinstance(table_id, str) or not table_id:
                continue
            table = tables.get(table_id)
            if not isinstance(table, dict):
                continue
            col_types = _table_col_types(table)

            # Key kind must match bound kinds when bounds are typed (prevents string keys vs decimal bounds).
            def _infer_key_kind(node: Any) -> Optional[str]:
                if not isinstance(node, dict):
                    return None
                if "var" in node and isinstance(node.get("var"), str):
                    return arg_types.get(node["var"])
                if "lit" in node:
                    lit = node.get("lit")
                    if isinstance(lit, dict) and isinstance(lit.get("type"), str):
                        return "decimal" if lit["type"] == "integer" else lit["type"]
                return None

            key_kind = _infer_key_kind(key_node)
            lb_kind = col_types.get(lower_bound_col) if isinstance(lower_bound_col, str) else None
            ub_kind = col_types.get(upper_bound_col) if isinstance(upper_bound_col, str) else None
            bound_kind = lb_kind or ub_kind
            if lb_kind and ub_kind and lb_kind != ub_kind:
                out.append(
                    ContractIRValidationError(
                        code="lookup_range_bound_kind_mismatch",
                        message=f"lookup_range lower/upper bound column kinds disagree: {lb_kind!r} vs {ub_kind!r}",
                        json_path="/" + "/".join(["derived", str(di), "expr"] + path),
                    )
                )
            if key_kind and bound_kind and key_kind != bound_kind:
                out.append(
                    ContractIRValidationError(
                        code="lookup_range_key_kind_mismatch",
                        message=(
                            f"lookup_range key kind {key_kind!r} does not match bound kind {bound_kind!r}; "
                            "use lookup() or lookup2() for string-keyed tables"
                        ),
                        json_path="/" + "/".join(["derived", str(di), "expr"] + path + ["args", "1"]),
                    )
                )

            def _col_type(col_name: Optional[str]) -> Optional[str]:
                if not isinstance(col_name, str) or not col_name:
                    return None
                return col_types.get(col_name)

            # Comparator columns must be string-typed.
            if isinstance(lower_cmp_col, str) and lower_cmp_col and _col_type(lower_cmp_col) not in (None, "string"):
                out.append(
                    ContractIRValidationError(
                        code="lookup_range_cmp_col_not_string",
                        message=(
                            f"lookup_range lower_cmp_col {lower_cmp_col!r} must be type 'string' "
                            f"in table {table_id!r} (got {_col_type(lower_cmp_col)!r})"
                        ),
                        json_path="/" + "/".join(["derived", str(di), "expr"] + path + ["args", "3", "lit", "value"]),
                    )
                )
            if isinstance(upper_cmp_col, str) and upper_cmp_col and _col_type(upper_cmp_col) not in (None, "string"):
                out.append(
                    ContractIRValidationError(
                        code="lookup_range_cmp_col_not_string",
                        message=(
                            f"lookup_range upper_cmp_col {upper_cmp_col!r} must be type 'string' "
                            f"in table {table_id!r} (got {_col_type(upper_cmp_col)!r})"
                        ),
                        json_path="/" + "/".join(["derived", str(di), "expr"] + path + ["args", "5", "lit", "value"]),
                    )
                )

            rows = table.get("rows") or []
            if not isinstance(rows, list):
                continue
            for ri, row in enumerate(rows):
                if not isinstance(row, dict):
                    continue
                cells = row.get("cells")
                if not isinstance(cells, dict):
                    continue

                def _present(col_name: Optional[str]) -> bool:
                    if not isinstance(col_name, str) or not col_name:
                        return False
                    return isinstance(cells.get(col_name), dict)

                def _cmp_value(col_name: Optional[str]) -> Optional[str]:
                    if not isinstance(col_name, str) or not col_name:
                        return None
                    node = cells.get(col_name)
                    if node is None:
                        return None
                    if not isinstance(node, dict):
                        # table_cell_not_ast will catch the shape; keep a targeted error too.
                        return None
                    return _string_lit(node)

                # Lower bound => requires lower comparator.
                if _present(lower_bound_col):
                    cv = _cmp_value(lower_cmp_col)
                    if cv is None:
                        out.append(
                            ContractIRValidationError(
                                code="lookup_range_cmp_cell_invalid",
                                message=(
                                    f"Row requires a lower comparator in {lower_cmp_col!r} when lower bound is present; "
                                    "expected a string literal 'gt' or 'gte'"
                                ),
                                json_path=f"/tables/{table_id}/rows/{ri}/cells/{lower_cmp_col}",
                            )
                        )
                    elif cv not in ("gt", "gte"):
                        out.append(
                            ContractIRValidationError(
                                code="lookup_range_cmp_cell_invalid",
                                message=f"Invalid lower comparator {cv!r}; expected 'gt' or 'gte'",
                                json_path=f"/tables/{table_id}/rows/{ri}/cells/{lower_cmp_col}",
                            )
                        )

                # Upper bound => requires upper comparator.
                if _present(upper_bound_col):
                    cv = _cmp_value(upper_cmp_col)
                    if cv is None:
                        out.append(
                            ContractIRValidationError(
                                code="lookup_range_cmp_cell_invalid",
                                message=(
                                    f"Row requires an upper comparator in {upper_cmp_col!r} when upper bound is present; "
                                    "expected a string literal 'lt' or 'lte'"
                                ),
                                json_path=f"/tables/{table_id}/rows/{ri}/cells/{upper_cmp_col}",
                            )
                        )
                    elif cv not in ("lt", "lte"):
                        out.append(
                            ContractIRValidationError(
                                code="lookup_range_cmp_cell_invalid",
                                message=f"Invalid upper comparator {cv!r}; expected 'lt' or 'lte'",
                                json_path=f"/tables/{table_id}/rows/{ri}/cells/{upper_cmp_col}",
                            )
                        )

    return out


def _validate_rate_literal_magnitude(doc: Any) -> List[ContractIRValidationError]:
    """Reject obviously mis-scaled rate literals (e.g., 2.25 meaning 225%).

    ContractIR v0.2 represents rates as decimal fractions:
      - 2.25%  -> "0.0225"
      - 50 bps -> either "0.005" (rate) or "50" (bps)

    We keep this gate intentionally simple: any *literal* of type "rate" must have
    absolute value <= 1.0. This catches common model mistakes where a percent value
    is encoded as a fraction (2.25 instead of 0.0225).
    """

    if not isinstance(doc, dict):
        return []

    out: List[ContractIRValidationError] = []

    def _check_rate(*, value: Any, json_path: str) -> None:
        if not isinstance(value, str) or not NUMERIC_LITERAL_RE.fullmatch(value.strip()):
            return  # other gates handle format/type
        try:
            d = Decimal(value)
        except InvalidOperation:
            return
        if abs(d) > Decimal("1.0"):
            out.append(
                ContractIRValidationError(
                    code="rate_literal_magnitude",
                    message=(
                        f"Rate literal {value!r} is out of bounds; rates must be decimal fractions "
                        "(example: 2.25% -> '0.0225')"
                    ),
                    json_path=json_path,
                )
            )

    # Table cells
    for ti, table in enumerate(doc.get("tables", []) or []):
        if not isinstance(table, dict):
            continue
        for ri, row in enumerate(table.get("rows", []) or []):
            if not isinstance(row, dict):
                continue
            cells = row.get("cells")
            if not isinstance(cells, dict):
                continue
            for col, cell in cells.items():
                for path, n in _iter_ast_nodes(cell):
                    lit = n.get("lit")
                    if not isinstance(lit, dict):
                        continue
                    if str(lit.get("type") or "") != "rate":
                        continue
                    _check_rate(
                        value=lit.get("value"),
                        json_path="/" + "/".join(["tables", str(ti), "rows", str(ri), "cells", str(col)] + path + ["lit", "value"]),
                    )

    # Derived exprs
    for di, fn in _iter_derived(doc):
        expr = fn.get("expr")
        for path, n in _iter_ast_nodes(expr):
            lit = n.get("lit")
            if not isinstance(lit, dict):
                continue
            if str(lit.get("type") or "") != "rate":
                continue
            _check_rate(
                value=lit.get("value"),
                json_path="/" + "/".join(["derived", str(di), "expr"] + path + ["lit", "value"]),
            )

    return out


def _validate_operator_arity(doc: Any) -> List[ContractIRValidationError]:
    """Validate operator arity (argument counts) for evaluatable ASTs.

    JSON Schema validates operator names but not their expected argument counts. This gate
    catches common issues like lookup2 being called with 5 args.
    """

    if not isinstance(doc, dict):
        return []

    exact: Dict[str, int] = {
        "index": 2,
        "round_up_to_increment": 2,
        "bps_to_rate": 1,
        "lookup": 4,
        "lookup2": 6,
        "lookup_range": 7,
        "lookup_rule": 3,
        "if": 3,
        "not": 1,
        "eq": 2,
        "lt": 2,
        "lte": 2,
        "gt": 2,
        "gte": 2,
    }
    min_arity: Dict[str, int] = {
        "add": 2,
        "sub": 2,
        "mul": 2,
        "div": 2,
        "max": 2,
        "min": 2,
        "and": 2,
        "or": 2,
    }

    out: List[ContractIRValidationError] = []

    for di, fn in _iter_derived(doc):
        expr = fn.get("expr")
        for path, n in _iter_ast_nodes(expr):
            op = n.get("op")
            if not isinstance(op, str):
                continue
            args = n.get("args")
            if not isinstance(args, list):
                continue
            argc = len(args)
            if op in exact and argc != exact[op]:
                out.append(
                    ContractIRValidationError(
                        code="op_arity",
                        message=f"{op} expects {exact[op]} args (got {argc})",
                        json_path="/" + "/".join(["derived", str(di), "expr"] + path + ["args"]),
                    )
                )
            if op in min_arity and argc < min_arity[op]:
                out.append(
                    ContractIRValidationError(
                        code="op_arity",
                        message=f"{op} expects at least {min_arity[op]} args (got {argc})",
                        json_path="/" + "/".join(["derived", str(di), "expr"] + path + ["args"]),
                    )
                )

    return out


def _validate_add_sub_kind_consistency(doc: Any) -> List[ContractIRValidationError]:
    """Reject add/sub operations whose operands can be inferred to have incompatible numeric kinds.

    The evaluator requires add/sub operands to have the same numeric kind (rate+rate, money+money, etc.).
    This gate catches high-impact cases where we can infer mismatch statically, e.g.:
      add(index(FederalFundsRate, date), {"lit":{"type":"decimal","value":"0.005"}})
    """

    if not isinstance(doc, dict):
        return []

    # Build index unit map for inference.
    index_units: Dict[str, str] = {}
    for idx in doc.get("indices", []) or []:
        if isinstance(idx, dict) and isinstance(idx.get("series_id"), str) and isinstance(idx.get("unit"), str):
            index_units[idx["series_id"]] = idx["unit"]

    tables = _tables_by_id(doc)

    def _infer_kind(node: Any, *, arg_types: Mapping[str, str]) -> Optional[str]:
        if not isinstance(node, dict):
            return None
        if "lit" in node:
            lit = node.get("lit")
            if not isinstance(lit, dict):
                return None
            t = lit.get("type")
            if t == "integer":
                return "decimal"
            return t if isinstance(t, str) else None
        if "var" in node:
            v = node.get("var")
            return arg_types.get(v) if isinstance(v, str) else None
        if "op" not in node:
            return None
        op = node.get("op")
        args = node.get("args")
        if not isinstance(op, str) or not isinstance(args, list):
            return None

        if op == "index" and len(args) == 2:
            sid = _string_lit(args[0])
            if isinstance(sid, str):
                return index_units.get(sid)
            return None
        if op == "bps_to_rate" and len(args) == 1:
            return "rate"
        if op == "round_up_to_increment" and len(args) == 2:
            return _infer_kind(args[0], arg_types=arg_types)

        if op in ("lookup", "lookup2", "lookup_range") and isinstance(args, list):
            # Return type is the referenced value column's type when fully literal-wired.
            table_id = _string_lit(args[0]) if len(args) >= 1 else None
            value_col = None
            if op == "lookup" and len(args) == 4:
                value_col = _string_lit(args[3])
            if op == "lookup2" and len(args) == 6:
                value_col = _string_lit(args[5])
            if op == "lookup_range" and len(args) == 7:
                value_col = _string_lit(args[6])
            if isinstance(table_id, str) and isinstance(value_col, str):
                table = tables.get(table_id)
                if isinstance(table, dict):
                    return _table_col_types(table).get(value_col)
            return None

        if op == "lookup_rule" and len(args) == 3:
            table_id = _string_lit(args[0])
            value_col = _string_lit(args[2])
            if isinstance(table_id, str) and isinstance(value_col, str):
                table = tables.get(table_id)
                if isinstance(table, dict):
                    return _table_col_types(table).get(value_col)
            return None

        if op == "if" and len(args) == 3:
            k1 = _infer_kind(args[1], arg_types=arg_types)
            k2 = _infer_kind(args[2], arg_types=arg_types)
            if k1 is not None and k1 == k2:
                return k1
            return None

        if op in ("eq", "lt", "lte", "gt", "gte", "and", "or", "not"):
            return "bool"

        if op in ("add", "sub", "mul", "div", "max", "min"):
            # Avoid deep inference here; only infer if all operands infer cleanly to same kind.
            inferred = [_infer_kind(a, arg_types=arg_types) for a in args]
            inferred = [k for k in inferred if k is not None]
            if inferred and all(k == inferred[0] for k in inferred):
                return inferred[0]
            return None

        return None

    out: List[ContractIRValidationError] = []
    for di, fn in _iter_derived(doc):
        arg_types: Dict[str, str] = {}
        for a in fn.get("args", []) or []:
            if isinstance(a, dict) and isinstance(a.get("name"), str) and isinstance(a.get("type"), str):
                arg_types[a["name"]] = a["type"]

        expr = fn.get("expr")
        for path, n in _iter_ast_nodes(expr):
            op = n.get("op")
            if op not in ("add", "sub"):
                continue
            args = n.get("args")
            if not isinstance(args, list) or len(args) < 2:
                continue
            kinds = [_infer_kind(a, arg_types=arg_types) for a in args]
            known = [k for k in kinds if isinstance(k, str)]
            if len(set(known)) >= 2:
                out.append(
                    ContractIRValidationError(
                        code="add_sub_kind_mismatch",
                        message=f"{op} operands must have matching numeric kinds; got {known}",
                        json_path="/" + "/".join(["derived", str(di), "expr"] + path),
                    )
                )

    return out


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


def _validate_source_refs_in_context(doc: Any) -> List[ContractIRValidationError]:
    """Ensure every `source_refs` anchor exists in the declared `sources[*].anchor_ids` list.

    This makes ContractIR artifacts self-contained and retrieval-agnostic: callers do not need
    external context packs to validate grounding; they only need the `sources` block.
    """

    if not isinstance(doc, dict):
        return []

    sources = doc.get("sources")
    if not isinstance(sources, list):
        return []

    allowed: set[str] = set()
    for s in sources:
        if not isinstance(s, dict):
            continue
        anchor_ids = s.get("anchor_ids")
        if not isinstance(anchor_ids, list):
            continue
        for aid in anchor_ids:
            if not isinstance(aid, str):
                continue
            cleaned = aid.strip()
            if cleaned and ANCHOR_ID_RE.fullmatch(cleaned):
                allowed.add(cleaned)

    out: list[ContractIRValidationError] = []

    def _walk(obj: Any, path: list[str]) -> None:
        if isinstance(obj, dict):
            for k, v in obj.items():
                if k == "source_refs":
                    if isinstance(v, list):
                        for idx, aid in enumerate(v):
                            if isinstance(aid, str) and aid.strip() in allowed:
                                continue
                            out.append(
                                ContractIRValidationError(
                                    code="anchor_not_in_context",
                                    message=(
                                        f"Anchor id {aid!r} is not present in sources[*].anchor_ids "
                                        f"(allowed={sorted(allowed)[:50]})"
                                    ),
                                    json_path="/" + "/".join(path + [k, str(idx)]),
                                )
                            )
                    else:
                        # Schema should catch this, but keep a helpful error for direct callers.
                        out.append(
                            ContractIRValidationError(
                                code="anchor_not_in_context",
                                message="source_refs must be an array of anchor ids",
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


def _validate_bool_column_cells(doc: Any) -> List[ContractIRValidationError]:
    """Ensure that table columns declared as bool are not populated with non-bool literals.

    This is a narrow hard gate to avoid an easy, high-impact failure mode when using rule tables:
      - column type is declared "bool" (e.g., predicate)
      - model encodes the predicate as a string literal instead of an AST bool expression
    """

    out: List[ContractIRValidationError] = []
    if not isinstance(doc, dict):
        return out

    tables = doc.get("tables")
    if not isinstance(tables, list):
        return out

    for ti, table in enumerate(tables):
        if not isinstance(table, dict):
            continue
        columns = table.get("columns")
        if not isinstance(columns, list):
            continue
        col_types: Dict[str, str] = {}
        for col in columns:
            if not isinstance(col, dict):
                continue
            name = col.get("name")
            typ = col.get("type")
            if isinstance(name, str) and isinstance(typ, str):
                col_types[name] = typ

        rows = table.get("rows")
        if not isinstance(rows, list):
            continue

        for ri, row in enumerate(rows):
            if not isinstance(row, dict):
                continue
            cells = row.get("cells")
            if not isinstance(cells, dict):
                continue
            for col_name, cell in cells.items():
                if col_types.get(col_name) != "bool":
                    continue
                if not isinstance(cell, dict):
                    continue
                lit = cell.get("lit")
                if not isinstance(lit, dict):
                    continue
                if lit.get("type") != "bool":
                    out.append(
                        ContractIRValidationError(
                            code="bool_cell_type",
                            message=(
                                f"Column {col_name!r} is declared type 'bool' but has literal type {lit.get('type')!r}; "
                                "use a bool literal or an operator/var node that evaluates to bool."
                            ),
                            json_path=f"/tables/{ti}/rows/{ri}/cells/{col_name}/lit/type",
                        )
                    )

    return out


def _validate_lookup_rule_calls(doc: Any) -> List[ContractIRValidationError]:
    """Validate lookup_rule(table_id, predicate_col, value_col) call sites against table schemas.

    Today, JSON Schema cannot easily express cross-references like:
      - "the predicate_col referenced by lookup_rule must be a bool-typed column in that table"

    Without this check, artifacts can pass schema validation but fail deterministically at evaluation-time.
    """

    if not isinstance(doc, dict):
        return []

    # Map table_id -> (col_name -> col_type)
    table_cols: Dict[str, Dict[str, str]] = {}
    table_objs: Dict[str, Dict[str, Any]] = {}
    for t in doc.get("tables", []) or []:
        if not isinstance(t, dict):
            continue
        table_id = t.get("table_id")
        if not isinstance(table_id, str) or not table_id:
            continue
        cols: Dict[str, str] = {}
        for c in t.get("columns", []) or []:
            if not isinstance(c, dict):
                continue
            name = c.get("name")
            typ = c.get("type")
            if isinstance(name, str) and isinstance(typ, str):
                cols[name] = typ
        table_cols[table_id] = cols
        table_objs[table_id] = t

    def _string_lit(node: Any) -> Optional[str]:
        if not isinstance(node, dict):
            return None
        lit = node.get("lit")
        if not isinstance(lit, dict):
            return None
        if lit.get("type") != "string":
            return None
        val = lit.get("value")
        return val if isinstance(val, str) else None

    out: List[ContractIRValidationError] = []

    def _walk(node: Any, path: List[str]) -> None:
        if isinstance(node, dict):
            op = node.get("op")
            if op == "lookup_rule":
                args = node.get("args")
                if isinstance(args, list) and len(args) == 3:
                    table_id = _string_lit(args[0])
                    predicate_col = _string_lit(args[1])
                    value_col = _string_lit(args[2])

                    # Enforce literal wiring for lookup_rule call sites (like lookup/lookup2/lookup_range).
                    if table_id is None:
                        out.append(
                            ContractIRValidationError(
                                code="lookup_rule_arg_not_string_literal",
                                message="lookup_rule table_id must be an AST string literal node",
                                json_path="/" + "/".join(path + ["args", "0"]),
                            )
                        )
                    if predicate_col is None:
                        out.append(
                            ContractIRValidationError(
                                code="lookup_rule_arg_not_string_literal",
                                message="lookup_rule predicate_col must be an AST string literal node",
                                json_path="/" + "/".join(path + ["args", "1"]),
                            )
                        )
                    if value_col is None:
                        out.append(
                            ContractIRValidationError(
                                code="lookup_rule_arg_not_string_literal",
                                message="lookup_rule value_col must be an AST string literal node",
                                json_path="/" + "/".join(path + ["args", "2"]),
                            )
                        )

                    if isinstance(table_id, str) and isinstance(predicate_col, str) and isinstance(value_col, str):
                        cols = table_cols.get(table_id)
                        if cols is None:
                            out.append(
                                ContractIRValidationError(
                                    code="lookup_rule_table_missing",
                                    message=f"lookup_rule references missing table_id {table_id!r}",
                                    json_path="/" + "/".join(path + ["args", "0", "lit", "value"]),
                                )
                            )
                        else:
                            pred_type = cols.get(predicate_col)
                            if pred_type != "bool":
                                out.append(
                                    ContractIRValidationError(
                                        code="lookup_rule_predicate_not_bool",
                                        message=(
                                            f"lookup_rule predicate_col {predicate_col!r} must be a bool column in table {table_id!r}; "
                                            f"got {pred_type!r}"
                                        ),
                                        json_path="/" + "/".join(path + ["args", "1", "lit", "value"]),
                                    )
                                )
                            if value_col not in cols:
                                out.append(
                                    ContractIRValidationError(
                                        code="lookup_rule_value_col_missing",
                                        message=(
                                            f"lookup_rule value_col {value_col!r} is not declared in table {table_id!r} columns"
                                        ),
                                        json_path="/" + "/".join(path + ["args", "2", "lit", "value"]),
                                    )
                                )

                            # Ensure every row provides both predicate and value cells (schema doesn't require this).
                            t = table_objs.get(table_id) or {}
                            rows = t.get("rows") or []
                            if isinstance(rows, list):
                                for ri, row in enumerate(rows):
                                    if not isinstance(row, dict):
                                        continue
                                    cells = row.get("cells")
                                    if not isinstance(cells, dict):
                                        continue
                                    pred_cell = cells.get(predicate_col)
                                    if pred_cell is None or not isinstance(pred_cell, dict):
                                        out.append(
                                            ContractIRValidationError(
                                                code="lookup_rule_predicate_cell_missing",
                                                message=(
                                                    f"lookup_rule requires predicate cell {predicate_col!r} in every row of table {table_id!r}"
                                                ),
                                                json_path=f"/tables/{table_id}/rows/{ri}/cells/{predicate_col}",
                                            )
                                        )
                                    val_cell = cells.get(value_col)
                                    if val_cell is None or not isinstance(val_cell, dict):
                                        out.append(
                                            ContractIRValidationError(
                                                code="lookup_rule_value_cell_missing",
                                                message=(
                                                    f"lookup_rule requires value cell {value_col!r} in every row of table {table_id!r}"
                                                ),
                                                json_path=f"/tables/{table_id}/rows/{ri}/cells/{value_col}",
                                            )
                                        )

            for k, v in node.items():
                if k == "args" and isinstance(v, list):
                    for i, child in enumerate(v):
                        _walk(child, path + [k, str(i)])
                elif isinstance(v, (dict, list)):
                    _walk(v, path + [k])
        elif isinstance(node, list):
            for i, v in enumerate(node):
                _walk(v, path + [str(i)])

    # Only derived functions are evaluated; focus on those.
    derived = doc.get("derived")
    if isinstance(derived, list):
        for di, fn in enumerate(derived):
            if not isinstance(fn, dict):
                continue
            _walk(fn.get("expr"), ["derived", str(di), "expr"])

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
    def __init__(self, table_id: str, key_column: str, key_value: str, count: int) -> None:
        super().__init__(
            f"Multiple matching rows in table {table_id!r} for {key_column} == {key_value!r} (count={count})"
        )
        self.table_id = table_id
        self.key_column = key_column
        self.key_value = key_value
        self.count = count


@dataclass(frozen=True)
class TypedValue:
    kind: str
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

    def as_date(self) -> str:
        if self.kind != "date":
            raise TypeError(f"Expected date kind, got {self.kind}")
        assert isinstance(self.value, str)
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

    tables_by_id: Dict[str, Any] = {}
    for t in contract_ir.get("tables", []) or []:
        if isinstance(t, dict) and isinstance(t.get("table_id"), str):
            tables_by_id[t["table_id"]] = t

    index_units: Dict[str, str] = {}
    for idx in contract_ir.get("indices", []) or []:
        if not isinstance(idx, dict):
            continue
        series_id = idx.get("series_id")
        unit = idx.get("unit")
        if isinstance(series_id, str) and isinstance(unit, str):
            index_units[series_id] = unit
    return evaluate_expr(
        expr=fn.get("expr"),
        arg_defs=fn.get("args", []) or [],
        args=args,
        indices=indices or {},
        tables=tables_by_id,
        index_units=index_units,
    )


def evaluate_expr(
    *,
    expr: Any,
    arg_defs: List[Any],
    args: Mapping[str, Any],
    indices: Mapping[str, Mapping[str, str]],
    tables: Mapping[str, Any],
    index_units: Mapping[str, str] | None = None,
) -> TypedValue:
    """Evaluate an AST node with an explicit argument specification.

    This is the shared evaluation entrypoint for other IRs that reuse the ContractIR AST/table semantics.
    """

    env: Dict[str, TypedValue] = {}
    for arg_def in arg_defs:
        if not isinstance(arg_def, dict):
            continue
        name = str(arg_def.get("name") or "")
        kind = str(arg_def.get("type") or "")
        if not name:
            continue
        if name not in args:
            raise MissingArgument(name)
        env[name] = _coerce_arg(kind, args[name])

    ctx = _EvalContext(
        env=env,
        indices=indices,
        tables=tables,
        index_units=index_units or {},
    )
    return _eval_ast(ctx, expr)


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
        if not ISO_DATE_RE.fullmatch(value.strip()):
            raise ValueError(f"Date argument must be ISO YYYY-MM-DD, got {value!r}")
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
    index_units: Mapping[str, str]


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
        unit = ctx.index_units.get(series_id)
        if unit is None:
            raise KeyError(
                f"Index series {series_id!r} is missing a declared unit; pass index_units or define contract_ir['indices']"
            )
        raw = series[date]
        if unit in ("rate", "bps", "decimal", "money"):
            if isinstance(raw, Decimal):
                value = raw
            elif isinstance(raw, int):
                value = Decimal(str(raw))
            elif isinstance(raw, float):
                raise TypeError(
                    f"Index value for {series_id!r} must not be float; pass a string decimal instead (date={date})"
                )
            elif isinstance(raw, str):
                value = _parse_decimal(raw)
            else:
                raise TypeError(
                    f"Index value for {series_id!r} must be a string decimal (or Decimal/int), got {type(raw).__name__}"
                )
            return TypedValue(kind=unit, value=value)
        if unit == "bool":
            if isinstance(raw, bool):
                return TypedValue(kind="bool", value=raw)
            if isinstance(raw, str):
                lowered = raw.strip().lower()
                if lowered in ("true", "false"):
                    return TypedValue(kind="bool", value=(lowered == "true"))
            raise TypeError(
                f"Index value for {series_id!r} must be boolean or 'true'/'false' string, got {raw!r} (date={date})"
            )
        if unit == "string":
            if not isinstance(raw, str):
                raw = str(raw)
            return TypedValue(kind="string", value=raw)
        raise TypeError(f"Unsupported index unit {unit!r} for series {series_id!r}")

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
            # Unit rules (minimal + fail-loud defaults):
            # - money/money, rate/rate, bps/bps are dimensionless ratios => decimal
            # - money/decimal preserves money
            # - rate|bps divided by decimal preserves rate|bps
            if a == b and a in ("money", "rate", "bps"):
                return "decimal"
            if a in ("rate", "bps") and b == "decimal":
                return a
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
            if (
                k1_val.kind == "string"
                and k2_val.kind == "string"
                and k1_val.value == key1_value
                and k2_val.value == key2_value
            ):
                matches.append(row)

        if not matches:
            raise NoMatchingRow(table_id, f"{key1_column}+{key2_column}", f"{key1_value}+{key2_value}")
        if len(matches) > 1:
            raise MultipleMatchingRows(table_id, f"{key1_column}+{key2_column}", f"{key1_value}+{key2_value}", len(matches))

        cell = (matches[0].get("cells") or {}).get(value_column)
        if not isinstance(cell, dict):
            raise KeyError(f"Missing value cell {value_column!r} in table {table_id!r}")
        return _eval_ast(ctx, cell)

    if op == "lookup_range":
        # lookup_range(table_id, key_value, lower_bound_col, lower_cmp_col, upper_bound_col, upper_cmp_col, value_col)
        if len(args) != 7:
            raise TypeError(
                "lookup_range(table_id, key_value, lower_bound_col, lower_cmp_col, upper_bound_col, upper_cmp_col, value_col) expects 7 args"
            )
        table_id = _eval_ast(ctx, args[0]).as_string()
        def _as_orderable(tv: TypedValue) -> tuple[str, Decimal | str]:
            if tv.kind in ("decimal", "rate", "bps", "money"):
                return (tv.kind, tv.as_decimal())
            if tv.kind == "date":
                return ("date", tv.as_date())
            if tv.kind == "string":
                return ("string", tv.as_string())
            raise TypeError(f"lookup_range key must be numeric/date/string; got {tv.kind}")

        key_val = _eval_ast(ctx, args[1])
        key_kind, key_ord = _as_orderable(key_val)
        lower_bound_col = _eval_ast(ctx, args[2]).as_string()
        lower_cmp_col = _eval_ast(ctx, args[3]).as_string()
        upper_bound_col = _eval_ast(ctx, args[4]).as_string()
        upper_cmp_col = _eval_ast(ctx, args[5]).as_string()
        value_col = _eval_ast(ctx, args[6]).as_string()

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

            def _maybe_get_numeric(col: str) -> Optional[tuple[str, Decimal]]:
                node = cells.get(col)
                if node is None:
                    return None
                if not isinstance(node, dict):
                    raise TypeError(f"Expected AST node for {col!r} in table {table_id!r}")
                tv = _eval_ast(ctx, node)
                k, v = _as_orderable(tv)
                if not isinstance(v, Decimal):
                    raise TypeError(f"Expected numeric bound for {col!r} in table {table_id!r}; got {k}")
                return (k, v)

            def _maybe_get_orderable(col: str) -> Optional[tuple[str, Decimal | str]]:
                node = cells.get(col)
                if node is None:
                    return None
                if not isinstance(node, dict):
                    raise TypeError(f"Expected AST node for {col!r} in table {table_id!r}")
                tv = _eval_ast(ctx, node)
                return _as_orderable(tv)

            def _maybe_get_string(col: str) -> Optional[str]:
                node = cells.get(col)
                if node is None:
                    return None
                if not isinstance(node, dict):
                    raise TypeError(f"Expected AST node for {col!r} in table {table_id!r}")
                return _eval_ast(ctx, node).as_string()

            # NOTE: lookup_range supports numeric and date/string kinds. Date ordering relies on ISO YYYY-MM-DD
            # lexicographic ordering, which matches chronological ordering.
            if key_kind in ("date", "string"):
                lower = _maybe_get_orderable(lower_bound_col)
                upper = _maybe_get_orderable(upper_bound_col)
            else:
                lower = _maybe_get_numeric(lower_bound_col)
                upper = _maybe_get_numeric(upper_bound_col)
            lower_cmp = _maybe_get_string(lower_cmp_col)
            upper_cmp = _maybe_get_string(upper_cmp_col)

            # Reject rows that have neither bound: they are ill-formed for range lookup.
            if lower is None and upper is None:
                continue

            ok = True

            if lower is not None:
                lb_kind, lb = lower
                if lb_kind != key_kind:
                    raise TypeError(
                        f"lookup_range kind mismatch for lower bound: key is {key_kind}, lower is {lb_kind}"
                    )
                if lower_cmp not in ("gt", "gte"):
                    raise TypeError(f"Invalid lower comparator {lower_cmp!r}; expected 'gt' or 'gte'")
                if isinstance(key_ord, Decimal) and isinstance(lb, Decimal):
                    ok = ok and (key_ord > lb if lower_cmp == "gt" else key_ord >= lb)
                else:
                    assert isinstance(key_ord, str) and isinstance(lb, str)
                    ok = ok and (key_ord > lb if lower_cmp == "gt" else key_ord >= lb)
            if upper is not None:
                ub_kind, ub = upper
                if ub_kind != key_kind:
                    raise TypeError(
                        f"lookup_range kind mismatch for upper bound: key is {key_kind}, upper is {ub_kind}"
                    )
                if upper_cmp not in ("lt", "lte"):
                    raise TypeError(f"Invalid upper comparator {upper_cmp!r}; expected 'lt' or 'lte'")
                if isinstance(key_ord, Decimal) and isinstance(ub, Decimal):
                    ok = ok and (key_ord < ub if upper_cmp == "lt" else key_ord <= ub)
                else:
                    assert isinstance(key_ord, str) and isinstance(ub, str)
                    ok = ok and (key_ord < ub if upper_cmp == "lt" else key_ord <= ub)

            if ok:
                matches.append(row)

        if not matches:
            raise NoMatchingRow(table_id, "range", str(key_val.value))
        if len(matches) > 1:
            raise MultipleMatchingRows(table_id, "range", str(key_val.value), len(matches))

        cell = (matches[0].get("cells") or {}).get(value_col)
        if not isinstance(cell, dict):
            raise KeyError(f"Missing value cell {value_col!r} in table {table_id!r}")
        return _eval_ast(ctx, cell)

    if op == "lookup_rule":
        # lookup_rule(table_id, predicate_col, value_col)
        if len(args) != 3:
            raise TypeError("lookup_rule(table_id, predicate_col, value_col) expects 3 args")

        table_id = _eval_ast(ctx, args[0]).as_string()
        predicate_col = _eval_ast(ctx, args[1]).as_string()
        value_col = _eval_ast(ctx, args[2]).as_string()

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
            predicate_node = cells.get(predicate_col)
            if not isinstance(predicate_node, dict):
                raise KeyError(f"Missing predicate cell {predicate_col!r} in table {table_id!r}")
            if _eval_ast(ctx, predicate_node).as_bool():
                matches.append(row)

        if not matches:
            raise NoMatchingRow(table_id, "rule", predicate_col)
        if len(matches) > 1:
            raise MultipleMatchingRows(table_id, "rule", predicate_col, len(matches))

        cell = (matches[0].get("cells") or {}).get(value_col)
        if not isinstance(cell, dict):
            raise KeyError(f"Missing value cell {value_col!r} in table {table_id!r}")
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
        if a.kind == "date" or b.kind == "date":
            if a.kind != "date" or b.kind != "date":
                raise TypeError(f"{op} date comparisons require both operands to be 'date'")
            av = a.as_date()
            bv = b.as_date()
        elif a.kind == "string" and b.kind == "string":
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
