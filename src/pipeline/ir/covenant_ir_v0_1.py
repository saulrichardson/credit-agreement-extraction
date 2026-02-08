from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import jsonschema

from pipeline.ir.contract_ir_v0_2 import ISO_DATE_RE, NUMERIC_LITERAL_RE, TypedValue, evaluate_expr  # Reuse AST + table semantics.

ANCHOR_ID_RE = re.compile(r"^A\d{4,}$")


def _project_root() -> Path:
    return Path(__file__).resolve().parents[3]


SCHEMA_PATH = _project_root() / "schemas" / "covenant_ir_v0_1.schema.json"


@dataclass(frozen=True)
class CovenantIRValidationError:
    code: str
    message: str
    json_path: str


def load_schema() -> Dict[str, Any]:
    if not SCHEMA_PATH.exists():
        raise FileNotFoundError(f"CovenantIR schema missing: {SCHEMA_PATH}")
    return json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))


def validate_covenant_ir(doc: Any) -> List[CovenantIRValidationError]:
    """Validate CovenantIR v0.1 with hard gates.

    Hard gates (v0.1):
      - Reject any JSON that fails schema validation.
      - Ensure anchor ids follow convention everywhere they appear.
      - Ensure every `source_refs` anchor exists in `sources[*].anchor_ids`.
      - Ensure covenant test/applies_when refs point to existing bool-returning derived fns.

    Returns a list of structured errors; empty list means valid.
    """

    schema = load_schema()
    validator = jsonschema.Draft202012Validator(schema)
    errors: List[CovenantIRValidationError] = []
    for err in sorted(validator.iter_errors(doc), key=lambda e: list(e.absolute_path)):
        json_path = "/" + "/".join(str(p) for p in err.absolute_path)
        errors.append(CovenantIRValidationError(code="schema", message=err.message, json_path=json_path))

    # Add explicit shape errors for bool_expr_spec to avoid opaque "oneOf failed" schema messages.
    # This is safe to run even when schema validation fails because it only checks types/keys.
    errors.extend(_validate_bool_expr_spec_shapes(doc))

    if errors:
        return errors

    errors.extend(_validate_anchor_id_fields(doc))
    errors.extend(_validate_source_refs_in_context(doc))
    errors.extend(_validate_id_uniqueness(doc))
    errors.extend(_validate_table_cells_are_ast_nodes(doc))
    errors.extend(_validate_bool_column_cells(doc))
    errors.extend(_validate_literal_value_formats(doc))
    errors.extend(_validate_operator_arities(doc))
    errors.extend(_validate_table_operator_calls(doc))
    errors.extend(_validate_expr_vars_declared(doc))
    errors.extend(_validate_bool_expr_fn_refs(doc))
    return errors


def _is_ast_node(x: Any) -> bool:
    if not isinstance(x, dict):
        return False
    keys = set(x.keys())
    return bool(keys & {"lit", "var", "op"}) and keys.issubset({"lit", "var", "op"})


def _validate_literal_value_formats(doc: Any) -> List[CovenantIRValidationError]:
    """Enforce strict literal value formats for deterministic evaluation.

    Motivation:
      - Prevent hallucinated placeholders like "YYYY-01-01" from silently validating.
      - Keep numeric literals parseable (no '$', '%', commas, etc.) so evaluation is deterministic.

    Policy (v0.1):
      - date literals MUST be ISO YYYY-MM-DD strings.
      - numeric literals (decimal/rate/bps/money/integer/int) MUST be plain decimal strings.
    """

    if not isinstance(doc, dict):
        return []

    out: List[CovenantIRValidationError] = []

    def _walk(x: Any, path: List[str]) -> None:
        if isinstance(x, dict):
            if _is_ast_node(x):
                lit = x.get("lit")
                if isinstance(lit, dict):
                    lit_type = lit.get("type")
                    value = lit.get("value")

                    if lit_type == "date":
                        if not isinstance(value, str) or not ISO_DATE_RE.fullmatch(value.strip()):
                            out.append(
                                CovenantIRValidationError(
                                    code="date_literal_format",
                                    message=f"Invalid date literal {value!r}; expected ISO YYYY-MM-DD string",
                                    json_path="/" + "/".join(path + ["lit", "value"]),
                                )
                            )
                    elif lit_type in ("decimal", "rate", "bps", "money", "integer", "int"):
                        if value is None:
                            out.append(
                                CovenantIRValidationError(
                                    code="numeric_literal_value_type",
                                    message=(
                                        f"Invalid {lit_type} literal value None; expected a plain decimal string like '0.005' "
                                        "(for missing bounds, set the CELL to null, not lit.value)"
                                    ),
                                    json_path="/" + "/".join(path + ["lit", "value"]),
                                )
                            )
                        elif not isinstance(value, str):
                            out.append(
                                CovenantIRValidationError(
                                    code="numeric_literal_value_type",
                                    message=(
                                        f"Invalid {lit_type} literal value type {type(value).__name__}; expected a plain decimal string like '0.005'"
                                    ),
                                    json_path="/" + "/".join(path + ["lit", "value"]),
                                )
                            )
                        elif not NUMERIC_LITERAL_RE.fullmatch(value.strip()):
                            out.append(
                                CovenantIRValidationError(
                                    code="numeric_literal_format",
                                    message=f"Invalid {lit_type} literal {value!r}; expected a plain decimal string like '0.005'",
                                    json_path="/" + "/".join(path + ["lit", "value"]),
                                )
                            )

            for k, v in x.items():
                _walk(v, path + [k])
        elif isinstance(x, list):
            for i, v in enumerate(x):
                _walk(v, path + [str(i)])

    _walk(doc, [])
    return out


def validate_precision_first_policy(doc: Any) -> List[CovenantIRValidationError]:
    """Validate "precision-first" CovenantIR policy.

    Policy (used by the financial covenant one-pass harness):
      - If open_items is non-empty, then the output must NOT contain partial covenants.
      - Concretely: covenants=[], tables=[], and derived=[] (if present).

    Motivation:
      - This prevents downstream code from accidentally treating a partial extraction as complete.
      - It also forces the extractor to be explicit: either fully encodable, or explicitly blocked.
    """

    if not isinstance(doc, dict):
        return []

    open_items = doc.get("open_items") or []
    if not isinstance(open_items, list):
        return []

    # Precision-first interpretation:
    # - Any open_items means the item is not fully encodable from the excerpt pack.
    # - Therefore open_items must be blocking, and the extractor must return no partial outputs.
    if not open_items:
        return []

    out: List[CovenantIRValidationError] = []

    for i, oi in enumerate(open_items):
        if not isinstance(oi, dict):
            continue
        if oi.get("blocking") is not True:
            out.append(
                CovenantIRValidationError(
                    code="precision_first_open_item_not_blocking",
                    message="precision-first policy: any open_item must have blocking=true",
                    json_path=f"/open_items/{i}/blocking",
                )
            )

    covenants = doc.get("covenants")
    if not (isinstance(covenants, list) and len(covenants) == 0):
        out.append(
            CovenantIRValidationError(
                code="precision_first_partial_output",
                message="precision-first policy: blocking open_items require covenants=[] (no partial covenants allowed)",
                json_path="/covenants",
            )
        )

    tables = doc.get("tables")
    if not (isinstance(tables, list) and len(tables) == 0):
        out.append(
            CovenantIRValidationError(
                code="precision_first_partial_output",
                message="precision-first policy: blocking open_items require tables=[] (no partial tables allowed)",
                json_path="/tables",
            )
        )

    if "derived" in doc:
        derived = doc.get("derived")
        if not (isinstance(derived, list) and len(derived) == 0):
            out.append(
                CovenantIRValidationError(
                    code="precision_first_partial_output",
                    message="precision-first policy: blocking open_items require derived=[] (no partial derived fns allowed)",
                    json_path="/derived",
                )
            )

    return out


def _validate_anchor_id_fields(doc: Any) -> List[CovenantIRValidationError]:
    out: List[CovenantIRValidationError] = []

    def _walk(obj: Any, path: List[str]) -> None:
        if isinstance(obj, dict):
            for k, v in obj.items():
                if k in ("source_refs", "anchor_ids"):
                    if isinstance(v, list):
                        for idx, aid in enumerate(v):
                            if not isinstance(aid, str) or not ANCHOR_ID_RE.fullmatch(aid.strip()):
                                out.append(
                                    CovenantIRValidationError(
                                        code="anchor_id",
                                        message=f"Invalid anchor id {aid!r} (expected like 'A0001')",
                                        json_path="/" + "/".join(path + [k, str(idx)]),
                                    )
                                )
                    else:
                        out.append(
                            CovenantIRValidationError(
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


def _validate_source_refs_in_context(doc: Any) -> List[CovenantIRValidationError]:
    """Ensure every `source_refs` anchor exists in the declared `sources[*].anchor_ids` list."""

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

    out: List[CovenantIRValidationError] = []

    def _walk(obj: Any, path: List[str]) -> None:
        if isinstance(obj, dict):
            for k, v in obj.items():
                if k == "source_refs":
                    if isinstance(v, list):
                        for idx, aid in enumerate(v):
                            if isinstance(aid, str) and aid.strip() in allowed:
                                continue
                            out.append(
                                CovenantIRValidationError(
                                    code="anchor_not_in_context",
                                    message=(
                                        f"Anchor id {aid!r} is not present in sources[*].anchor_ids "
                                        f"(allowed={sorted(allowed)[:50]})"
                                    ),
                                    json_path="/" + "/".join(path + [k, str(idx)]),
                                )
                            )
                    else:
                        out.append(
                            CovenantIRValidationError(
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


def _validate_bool_expr_fn_refs(doc: Any) -> List[CovenantIRValidationError]:
    """Validate that any covenant bool expr spec referencing a derived fn points to a bool-returning fn."""

    if not isinstance(doc, dict):
        return []

    derived_by_id: Dict[str, Dict[str, Any]] = {}
    for fn in doc.get("derived", []) or []:
        if not isinstance(fn, dict):
            continue
        fn_id = fn.get("fn_id")
        if isinstance(fn_id, str):
            derived_by_id[fn_id] = fn

    out: List[CovenantIRValidationError] = []
    covenants = doc.get("covenants")
    if not isinstance(covenants, list):
        return out

    def _check_spec(spec: Any, json_path: str) -> None:
        if not isinstance(spec, dict):
            return
        if set(spec.keys()) != {"fn_id"}:
            return
        fn_id = spec.get("fn_id")
        if not isinstance(fn_id, str) or not fn_id:
            return
        fn = derived_by_id.get(fn_id)
        if fn is None:
            out.append(
                CovenantIRValidationError(
                    code="missing_derived_fn",
                    message=f"Referenced derived fn_id {fn_id!r} not found in derived[]",
                    json_path=json_path + "/fn_id",
                )
            )
            return
        if fn.get("returns") != "bool":
            out.append(
                CovenantIRValidationError(
                    code="derived_fn_wrong_return",
                    message=f"Referenced derived fn_id {fn_id!r} must return 'bool', got {fn.get('returns')!r}",
                    json_path=json_path + "/fn_id",
                )
            )

    for ci, cov in enumerate(covenants):
        if not isinstance(cov, dict):
            continue
        _check_spec(cov.get("test"), f"/covenants/{ci}/test")
        if "applies_when" in cov:
            _check_spec(cov.get("applies_when"), f"/covenants/{ci}/applies_when")

    return out


def _validate_bool_expr_spec_shapes(doc: Any) -> List[CovenantIRValidationError]:
    """Produce explicit, actionable validation errors for bool_expr_spec objects.

    Motivation:
      - The JSON Schema uses oneOf for bool_expr_spec, which often yields opaque errors like
        "is not valid under any of the given schemas" when something is missing.
      - These explicit errors make LLM repair loops significantly more reliable.
    """

    if not isinstance(doc, dict):
        return []

    covenants = doc.get("covenants")
    if not isinstance(covenants, list):
        return []

    out: List[CovenantIRValidationError] = []

    def _check_spec(spec: Any, json_path: str) -> None:
        if spec is None:
            out.append(
                CovenantIRValidationError(
                    code="bool_expr_spec_null",
                    message="bool_expr_spec must be omitted or an object (not null)",
                    json_path=json_path,
                )
            )
            return
        if not isinstance(spec, dict):
            out.append(
                CovenantIRValidationError(
                    code="bool_expr_spec_type",
                    message=f"bool_expr_spec must be an object; got {type(spec).__name__}",
                    json_path=json_path,
                )
            )
            return

        # Derived fn reference form
        if set(spec.keys()) == {"fn_id"}:
            fn_id = spec.get("fn_id")
            if not isinstance(fn_id, str) or not fn_id.strip():
                out.append(
                    CovenantIRValidationError(
                        code="bool_expr_spec_fn_id",
                        message="bool_expr_spec fn_id must be a non-empty string",
                        json_path=json_path + "/fn_id",
                    )
                )
            return

        # Inline expr form
        required = ("args", "returns", "expr", "source_refs")
        for k in required:
            if k not in spec:
                out.append(
                    CovenantIRValidationError(
                        code="inline_expr_missing_field",
                        message=f"inline bool expression must include {k!r}",
                        json_path=json_path,
                    )
                )

        returns = spec.get("returns")
        if returns is not None and returns != "bool":
            out.append(
                CovenantIRValidationError(
                    code="inline_expr_returns_not_bool",
                    message=f"inline covenant expression must declare returns='bool'; got {returns!r}",
                    json_path=json_path + "/returns",
                )
            )

        # source_refs should be non-empty list (schema also enforces when schema passes)
        sr = spec.get("source_refs")
        if sr is not None and not (isinstance(sr, list) and sr):
            out.append(
                CovenantIRValidationError(
                    code="inline_expr_source_refs",
                    message="inline covenant expression source_refs must be a non-empty array",
                    json_path=json_path + "/source_refs",
                )
            )

    for ci, cov in enumerate(covenants):
        if not isinstance(cov, dict):
            continue
        if "test" in cov:
            _check_spec(cov.get("test"), f"/covenants/{ci}/test")
        if "applies_when" in cov:
            _check_spec(cov.get("applies_when"), f"/covenants/{ci}/applies_when")

    return out


def _is_ast_node(obj: Any) -> bool:
    if not isinstance(obj, dict):
        return False
    if set(obj.keys()) == {"var"}:
        return isinstance(obj.get("var"), str)
    if set(obj.keys()) == {"lit"} and isinstance(obj.get("lit"), dict):
        lit = obj["lit"]
        return isinstance(lit.get("type"), str) and "value" in lit
    if set(obj.keys()) == {"op", "args"}:
        return isinstance(obj.get("op"), str) and isinstance(obj.get("args"), list)
    return False


def _validate_id_uniqueness(doc: Any) -> List[CovenantIRValidationError]:
    """Hard gate: IDs must be unique within their respective arrays.

    Rationale:
      - Duplicate IDs create ambiguous evaluation / downstream linking (e.g., two covenants with the same covenant_id).
      - JSON Schema can't conveniently enforce uniqueness across arbitrary object arrays.
    """

    if not isinstance(doc, dict):
        return []

    out: List[CovenantIRValidationError] = []

    def _check_unique(*, array_key: str, id_key: str, code: str) -> None:
        arr = doc.get(array_key)
        if not isinstance(arr, list):
            return
        seen: Dict[str, int] = {}
        for i, rec in enumerate(arr):
            if not isinstance(rec, dict):
                continue
            rid = rec.get(id_key)
            if not isinstance(rid, str) or not rid.strip():
                continue
            rid = rid.strip()
            if rid in seen:
                out.append(
                    CovenantIRValidationError(
                        code=code,
                        message=f"Duplicate {id_key} {rid!r} in {array_key} (first at index {seen[rid]})",
                        json_path=f"/{array_key}/{i}/{id_key}",
                    )
                )
            else:
                seen[rid] = i

    _check_unique(array_key="sources", id_key="source_id", code="duplicate_source_id")
    _check_unique(array_key="indices", id_key="series_id", code="duplicate_series_id")
    _check_unique(array_key="tables", id_key="table_id", code="duplicate_table_id")
    _check_unique(array_key="derived", id_key="fn_id", code="duplicate_fn_id")
    _check_unique(array_key="covenants", id_key="covenant_id", code="duplicate_covenant_id")

    return out


def _validate_table_cells_are_ast_nodes(doc: Any) -> List[CovenantIRValidationError]:
    """Hard gate: table cell values must be AST nodes.

    Why:
      - The schema allows `cells` to be an object without constraining the values.
      - Deterministic evaluation (`lookup_range` / `lookup_rule`) requires AST nodes in table cells.
    """

    if not isinstance(doc, dict):
        return []

    tables = doc.get("tables")
    if not isinstance(tables, list):
        return []

    out: List[CovenantIRValidationError] = []

    for ti, t in enumerate(tables):
        if not isinstance(t, dict):
            continue
        cols = t.get("columns") or []
        col_names: set[str] = set()
        if isinstance(cols, list):
            for c in cols:
                if isinstance(c, dict) and isinstance(c.get("name"), str):
                    col_names.add(c["name"])

        rows = t.get("rows") or []
        if not isinstance(rows, list):
            continue
        for ri, row in enumerate(rows):
            if not isinstance(row, dict):
                continue
            cells = row.get("cells")
            if not isinstance(cells, dict):
                continue
            for col_name, cell in cells.items():
                if isinstance(col_name, str) and col_names and col_name not in col_names:
                    out.append(
                        CovenantIRValidationError(
                            code="table_unknown_column",
                            message=f"Table row cell references unknown column {col_name!r} (not declared in table.columns)",
                            json_path=f"/tables/{ti}/rows/{ri}/cells/{col_name}",
                        )
                    )
                if not _is_ast_node(cell):
                    out.append(
                        CovenantIRValidationError(
                            code="table_cell_not_ast",
                            message=(
                                "Table cell must be an AST node object "
                                "(one of {'lit':...} / {'var':...} / {'op':...,'args':[...]})."
                            ),
                            json_path=f"/tables/{ti}/rows/{ri}/cells/{col_name}",
                        )
                    )

    return out


def _walk_ast_nodes(root: Any, *, path: List[str], out: List[Tuple[List[str], Dict[str, Any]]]) -> None:
    """Collect all AST nodes reachable from root, along with their JSON-pointer-ish paths."""

    if isinstance(root, dict):
        if "lit" in root or "var" in root or "op" in root:
            out.append((path, root))
            if root.get("op") and isinstance(root.get("args"), list):
                for i, child in enumerate(root["args"]):
                    _walk_ast_nodes(child, path=path + ["args", str(i)], out=out)
            return

        for k, v in root.items():
            if isinstance(v, (dict, list)):
                _walk_ast_nodes(v, path=path + [k], out=out)
        return

    if isinstance(root, list):
        for i, v in enumerate(root):
            if isinstance(v, (dict, list)):
                _walk_ast_nodes(v, path=path + [str(i)], out=out)


def _validate_operator_arities(doc: Any) -> List[CovenantIRValidationError]:
    """Hard gate: operator nodes must have the correct number of args.

    Motivation:
      - The schema enforces operator *names*, but not arity.
      - If arity is wrong, evaluation fails at runtime; treat as invalid CovenantIR.
    """

    if not isinstance(doc, dict):
        return []

    # Exact arity requirements (op -> n_args)
    exact: Dict[str, int] = {
        "index": 2,
        "round_up_to_increment": 2,
        "bps_to_rate": 1,
        "lookup": 4,
        "lookup2": 6,
        "lookup_range": 7,
        "lookup_rule": 3,
        "if": 3,
        "eq": 2,
        "lt": 2,
        "lte": 2,
        "gt": 2,
        "gte": 2,
        "not": 1,
    }

    # Minimum arity requirements (op -> min_args)
    minimum: Dict[str, int] = {
        "add": 2,
        "sub": 2,
        "mul": 2,
        "div": 2,
        "max": 2,
        "min": 2,
        "and": 2,
        "or": 2,
    }

    nodes: List[Tuple[List[str], Dict[str, Any]]] = []
    _walk_ast_nodes(doc, path=[], out=nodes)

    out: List[CovenantIRValidationError] = []
    for path, node in nodes:
        op = node.get("op")
        if not isinstance(op, str):
            continue
        args = node.get("args")
        if not isinstance(args, list):
            continue

        if op in exact:
            want = exact[op]
            if len(args) != want:
                out.append(
                    CovenantIRValidationError(
                        code="op_arity",
                        message=f"Operator {op!r} expects {want} args, got {len(args)}",
                        json_path="/" + "/".join(path + ["args"]),
                    )
                )
        elif op in minimum:
            want = minimum[op]
            if len(args) < want:
                out.append(
                    CovenantIRValidationError(
                        code="op_arity",
                        message=f"Operator {op!r} expects at least {want} args, got {len(args)}",
                        json_path="/" + "/".join(path + ["args"]),
                    )
                )

    return out


def _validate_table_operator_calls(doc: Any) -> List[CovenantIRValidationError]:
    """Validate lookup*/lookup_range/lookup_rule call sites against declared tables/columns.

    This is a hard gate because any mismatch makes evaluation fail or become ambiguous.
    """

    if not isinstance(doc, dict):
        return []

    # Map table_id -> (col_name -> col_type)
    table_cols: Dict[str, Dict[str, str]] = {}
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

    out: List[CovenantIRValidationError] = []

    nodes: List[Tuple[List[str], Dict[str, Any]]] = []
    _walk_ast_nodes(doc, path=[], out=nodes)

    def _err(code: str, msg: str, path: List[str]) -> None:
        out.append(CovenantIRValidationError(code=code, message=msg, json_path="/" + "/".join(path)))

    for path, node in nodes:
        op = node.get("op")
        if not isinstance(op, str):
            continue
        args = node.get("args")
        if not isinstance(args, list):
            continue

        if op == "lookup_rule":
            if len(args) != 3:
                continue  # arity handled elsewhere
            table_id = _string_lit(args[0])
            if table_id is None:
                _err(
                    "lookup_rule_table_id_not_string_lit",
                    "lookup_rule table_id must be an AST string literal",
                    path + ["args", "0"],
                )
                continue
            predicate_col = _string_lit(args[1])
            if predicate_col is None:
                _err(
                    "lookup_rule_predicate_col_not_string_lit",
                    "lookup_rule predicate_col must be an AST string literal",
                    path + ["args", "1"],
                )
                continue
            value_col = _string_lit(args[2])
            if value_col is None:
                _err(
                    "lookup_rule_value_col_not_string_lit",
                    "lookup_rule value_col must be an AST string literal",
                    path + ["args", "2"],
                )
                continue

            cols = table_cols.get(table_id)
            if cols is None:
                _err(
                    "lookup_rule_table_missing",
                    f"lookup_rule references missing table_id {table_id!r}",
                    path + ["args", "0", "lit", "value"],
                )
                continue

            if predicate_col not in cols:
                _err(
                    "lookup_rule_predicate_col_missing",
                    f"lookup_rule predicate_col {predicate_col!r} is not declared in table {table_id!r} columns",
                    path + ["args", "1", "lit", "value"],
                )
            else:
                pred_type = cols.get(predicate_col)
                if pred_type != "bool":
                    _err(
                        "lookup_rule_predicate_not_bool",
                        (
                            f"lookup_rule predicate_col {predicate_col!r} must be a bool column in table {table_id!r}; "
                            f"got {pred_type!r}"
                        ),
                        path + ["args", "1", "lit", "value"],
                    )

            if value_col not in cols:
                _err(
                    "lookup_rule_value_col_missing",
                    f"lookup_rule value_col {value_col!r} is not declared in table {table_id!r} columns",
                    path + ["args", "2", "lit", "value"],
                )

        if op == "lookup_range":
            if len(args) != 7:
                continue
            table_id = _string_lit(args[0])
            if table_id is None:
                _err(
                    "lookup_range_table_id_not_string_lit",
                    "lookup_range table_id must be an AST string literal",
                    path + ["args", "0"],
                )
                continue
            cols = table_cols.get(table_id)
            if cols is None:
                _err(
                    "lookup_range_table_missing",
                    f"lookup_range references missing table_id {table_id!r}",
                    path + ["args", "0", "lit", "value"],
                )
                continue

            lower_bound_col = _string_lit(args[2])
            lower_cmp_col = _string_lit(args[3])
            upper_bound_col = _string_lit(args[4])
            upper_cmp_col = _string_lit(args[5])
            value_col = _string_lit(args[6])

            for idx, name, code in [
                (2, lower_bound_col, "lookup_range_lower_bound_col_not_string_lit"),
                (3, lower_cmp_col, "lookup_range_lower_cmp_col_not_string_lit"),
                (4, upper_bound_col, "lookup_range_upper_bound_col_not_string_lit"),
                (5, upper_cmp_col, "lookup_range_upper_cmp_col_not_string_lit"),
                (6, value_col, "lookup_range_value_col_not_string_lit"),
            ]:
                if name is None:
                    _err(code, "lookup_range column identifiers must be AST string literals", path + ["args", str(idx)])

            for col_name, kind, col_code in [
                (lower_bound_col, None, "lookup_range_col_missing"),
                (lower_cmp_col, "string", "lookup_range_cmp_col_type"),
                (upper_bound_col, None, "lookup_range_col_missing"),
                (upper_cmp_col, "string", "lookup_range_cmp_col_type"),
                (value_col, None, "lookup_range_col_missing"),
            ]:
                if col_name is None:
                    continue
                if col_name not in cols:
                    _err(
                        col_code,
                        f"lookup_range references missing column {col_name!r} in table {table_id!r}",
                        path + ["args"],
                    )
                    continue
                if kind is not None and cols.get(col_name) != kind:
                    _err(
                        col_code,
                        f"lookup_range comparator column {col_name!r} must have type {kind!r} in table {table_id!r}; got {cols.get(col_name)!r}",
                        path + ["args"],
                    )

        if op == "lookup":
            if len(args) != 4:
                continue
            table_id = _string_lit(args[0])
            if table_id is None:
                _err("lookup_table_id_not_string_lit", "lookup table_id must be an AST string literal", path + ["args", "0"])
                continue
            cols = table_cols.get(table_id)
            if cols is None:
                _err("lookup_table_missing", f"lookup references missing table_id {table_id!r}", path + ["args", "0", "lit", "value"])
                continue
            key_col = _string_lit(args[1])
            if key_col is None:
                _err("lookup_key_col_not_string_lit", "lookup key_column must be an AST string literal", path + ["args", "1"])
            elif key_col not in cols:
                _err("lookup_key_col_missing", f"lookup key_column {key_col!r} is not declared in table {table_id!r} columns", path + ["args"])
            value_col = _string_lit(args[3])
            if value_col is None:
                _err("lookup_value_col_not_string_lit", "lookup value_column must be an AST string literal", path + ["args", "3"])
            elif value_col not in cols:
                _err("lookup_value_col_missing", f"lookup value_column {value_col!r} is not declared in table {table_id!r} columns", path + ["args"])

        if op == "lookup2":
            if len(args) != 6:
                continue
            table_id = _string_lit(args[0])
            if table_id is None:
                _err("lookup2_table_id_not_string_lit", "lookup2 table_id must be an AST string literal", path + ["args", "0"])
                continue
            cols = table_cols.get(table_id)
            if cols is None:
                _err("lookup2_table_missing", f"lookup2 references missing table_id {table_id!r}", path + ["args", "0", "lit", "value"])
                continue
            key1_col = _string_lit(args[1])
            key2_col = _string_lit(args[3])
            value_col = _string_lit(args[5])
            if key1_col is None:
                _err("lookup2_key1_col_not_string_lit", "lookup2 key1_column must be an AST string literal", path + ["args", "1"])
            elif key1_col not in cols:
                _err("lookup2_key1_col_missing", f"lookup2 key1_column {key1_col!r} is not declared in table {table_id!r} columns", path + ["args"])
            if key2_col is None:
                _err("lookup2_key2_col_not_string_lit", "lookup2 key2_column must be an AST string literal", path + ["args", "3"])
            elif key2_col not in cols:
                _err("lookup2_key2_col_missing", f"lookup2 key2_column {key2_col!r} is not declared in table {table_id!r} columns", path + ["args"])
            if value_col is None:
                _err("lookup2_value_col_not_string_lit", "lookup2 value_column must be an AST string literal", path + ["args", "5"])
            elif value_col not in cols:
                _err("lookup2_value_col_missing", f"lookup2 value_column {value_col!r} is not declared in table {table_id!r} columns", path + ["args"])

    return out


def _validate_expr_vars_declared(doc: Any) -> List[CovenantIRValidationError]:
    """Hard gate: every var node used by an expression must be declared in that expression's args.

    This includes vars referenced inside table predicates used by lookup_rule/lookup_range.
    """

    if not isinstance(doc, dict):
        return []

    tables_by_id: Dict[str, Tuple[int, Dict[str, Any]]] = {}
    for ti, t in enumerate(doc.get("tables", []) or []):
        if not isinstance(t, dict):
            continue
        table_id = t.get("table_id")
        if isinstance(table_id, str) and table_id:
            tables_by_id[table_id] = (ti, t)

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

    def _collect_vars_and_tables(node: Any, *, base_path: List[str]) -> Tuple[List[Tuple[str, str]], List[str]]:
        vars_found: List[Tuple[str, str]] = []
        tables_found: List[str] = []

        def _walk(n: Any, p: List[str]) -> None:
            if not isinstance(n, dict):
                return
            if "var" in n:
                v = n.get("var")
                if isinstance(v, str) and v.strip():
                    vars_found.append((v, "/" + "/".join(p + ["var"])))
            op = n.get("op")
            args = n.get("args")
            if isinstance(op, str) and isinstance(args, list) and op in ("lookup", "lookup2", "lookup_range", "lookup_rule"):
                if args:
                    tid = _string_lit(args[0])
                    if isinstance(tid, str) and tid:
                        tables_found.append(tid)
            if isinstance(args, list):
                for i, child in enumerate(args):
                    _walk(child, p + ["args", str(i)])

        _walk(node, base_path)
        return (vars_found, sorted(set(tables_found)))

    # Pre-compute vars used inside each table's cells (including nested expressions).
    table_vars: Dict[str, List[Tuple[str, str]]] = {}
    for table_id, (ti, table) in tables_by_id.items():
        vars_in_table: List[Tuple[str, str]] = []
        for ri, row in enumerate(table.get("rows", []) or []):
            if not isinstance(row, dict):
                continue
            cells = row.get("cells")
            if not isinstance(cells, dict):
                continue
            for col_name, cell in cells.items():
                if not isinstance(col_name, str) or not isinstance(cell, dict):
                    continue
                (vars_found, tables_found) = _collect_vars_and_tables(
                    cell, base_path=["tables", str(ti), "rows", str(ri), "cells", col_name]
                )
                vars_in_table.extend(vars_found)
                # We deliberately ignore tables_found here; nested table references in table cells are allowed,
                # but validating their variable dependencies is non-trivial and rare in current use.
        table_vars[table_id] = vars_in_table

    out: List[CovenantIRValidationError] = []

    def _declared_arg_names(arg_defs: Any) -> set[str]:
        names: set[str] = set()
        if not isinstance(arg_defs, list):
            return names
        for a in arg_defs:
            if not isinstance(a, dict):
                continue
            name = a.get("name")
            if isinstance(name, str) and name.strip():
                names.add(name.strip())
        return names

    def _check_expr(*, expr: Any, arg_defs: Any, expr_path: str) -> None:
        declared = _declared_arg_names(arg_defs)
        if not declared:
            # Schema allows empty args in some contexts, but for CovenantIR evaluation we need vars declared.
            # We don't fail on empty args unless we see var usage.
            declared = set()

        (vars_found, tables_found) = _collect_vars_and_tables(expr, base_path=expr_path.strip("/").split("/") if expr_path else [])
        for var_name, var_path in vars_found:
            if var_name not in declared:
                out.append(
                    CovenantIRValidationError(
                        code="undefined_var",
                        message=f"Expression at {expr_path} references var {var_name!r} but it is not declared in args",
                        json_path=var_path,
                    )
                )

        for table_id in tables_found:
            for var_name, var_path in table_vars.get(table_id, []):
                if var_name not in declared:
                    out.append(
                        CovenantIRValidationError(
                            code="undefined_var",
                            message=(
                                f"Expression at {expr_path} uses table {table_id!r} whose cells reference var {var_name!r}, "
                                "but it is not declared in args"
                            ),
                            json_path=var_path,
                        )
                    )

    # Derived functions
    for di, fn in enumerate(doc.get("derived", []) or []):
        if not isinstance(fn, dict):
            continue
        _check_expr(expr=fn.get("expr"), arg_defs=fn.get("args"), expr_path=f"/derived/{di}/expr")

    # Inline exprs in covenants
    for ci, cov in enumerate(doc.get("covenants", []) or []):
        if not isinstance(cov, dict):
            continue
        for field in ("applies_when", "test"):
            spec = cov.get(field)
            if not isinstance(spec, dict):
                continue
            if set(spec.keys()) == {"fn_id"}:
                continue
            if "expr" in spec:
                _check_expr(expr=spec.get("expr"), arg_defs=spec.get("args"), expr_path=f"/covenants/{ci}/{field}/expr")

    return out


def _validate_bool_column_cells(doc: Any) -> List[CovenantIRValidationError]:
    """Ensure table columns declared as bool are not populated with non-bool literals.

    This is a narrow hard gate to avoid a high-impact failure mode:
      - a rule table declares `predicate` as type bool
      - the model populates predicate cells with a string literal ("x > 1") instead of an AST bool
    """

    out: List[CovenantIRValidationError] = []
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
                        CovenantIRValidationError(
                            code="bool_cell_type",
                            message=(
                                f"Column {col_name!r} is declared type 'bool' but has literal type {lit.get('type')!r}; "
                                "use a bool literal or an operator/var node that evaluates to bool."
                            ),
                            json_path=f"/tables/{ti}/rows/{ri}/cells/{col_name}/lit/type",
                        )
                    )

    return out


@dataclass(frozen=True)
class CovenantEvaluationResult:
    covenant_id: str
    applicable: bool
    passed: Optional[bool]
    details: Dict[str, Any]


class CovenantIREvalError(Exception):
    pass


class UnknownCovenant(CovenantIREvalError):
    def __init__(self, covenant_id: str) -> None:
        super().__init__(f"Unknown covenant: {covenant_id}")
        self.covenant_id = covenant_id


def evaluate_covenant(
    covenant_ir: Mapping[str, Any],
    *,
    covenant_id: str,
    args: Mapping[str, Any],
    indices: Mapping[str, Mapping[str, str]] | None = None,
) -> CovenantEvaluationResult:
    cov = None
    for rec in covenant_ir.get("covenants", []) or []:
        if isinstance(rec, dict) and rec.get("covenant_id") == covenant_id:
            cov = rec
            break
    if cov is None:
        raise UnknownCovenant(covenant_id)

    tables_by_id: Dict[str, Any] = {}
    for t in covenant_ir.get("tables", []) or []:
        if isinstance(t, dict) and isinstance(t.get("table_id"), str):
            tables_by_id[t["table_id"]] = t

    derived_by_id: Dict[str, Any] = {}
    for fn in covenant_ir.get("derived", []) or []:
        if isinstance(fn, dict) and isinstance(fn.get("fn_id"), str):
            derived_by_id[fn["fn_id"]] = fn

    index_units: Dict[str, str] = {}
    for idx in covenant_ir.get("indices", []) or []:
        if not isinstance(idx, dict):
            continue
        series_id = idx.get("series_id")
        unit = idx.get("unit")
        if isinstance(series_id, str) and isinstance(unit, str):
            index_units[series_id] = unit

    applies_when = cov.get("applies_when")
    applicable = True
    if applies_when is not None:
        app_val = _eval_bool_expr_spec(
            spec=applies_when,
            args=args,
            indices=indices or {},
            tables=tables_by_id,
            derived_by_id=derived_by_id,
            index_units=index_units,
        )
        applicable = app_val

    if not applicable:
        return CovenantEvaluationResult(
            covenant_id=covenant_id,
            applicable=False,
            passed=None,
            details={"status": "not_applicable"},
        )

    passed = _eval_bool_expr_spec(
        spec=cov.get("test"),
        args=args,
        indices=indices or {},
        tables=tables_by_id,
        derived_by_id=derived_by_id,
        index_units=index_units,
    )
    return CovenantEvaluationResult(
        covenant_id=covenant_id,
        applicable=True,
        passed=passed,
        details={"status": "pass" if passed else "fail"},
    )


def _eval_bool_expr_spec(
    *,
    spec: Any,
    args: Mapping[str, Any],
    indices: Mapping[str, Mapping[str, str]],
    tables: Mapping[str, Any],
    derived_by_id: Mapping[str, Any],
    index_units: Mapping[str, str],
) -> bool:
    tv = _eval_expr_spec(
        spec=spec,
        args=args,
        indices=indices,
        tables=tables,
        derived_by_id=derived_by_id,
        index_units=index_units,
    )
    return tv.as_bool()


def _eval_expr_spec(
    *,
    spec: Any,
    args: Mapping[str, Any],
    indices: Mapping[str, Mapping[str, str]],
    tables: Mapping[str, Any],
    derived_by_id: Mapping[str, Any],
    index_units: Mapping[str, str],
) -> TypedValue:
    if not isinstance(spec, dict):
        raise TypeError(f"expr spec must be an object; got {type(spec).__name__}")

    # Derived function reference form: {"fn_id": "..."}
    if set(spec.keys()) == {"fn_id"}:
        fn_id = spec.get("fn_id")
        if not isinstance(fn_id, str) or not fn_id:
            raise TypeError("expr spec fn_id must be a non-empty string")
        fn = derived_by_id.get(fn_id)
        if not isinstance(fn, dict):
            raise KeyError(f"Derived fn not found: {fn_id}")
        return evaluate_expr(
            expr=fn.get("expr"),
            arg_defs=fn.get("args", []) or [],
            args=args,
            indices=indices,
            tables=tables,
            index_units=index_units,
        )

    # Inline form: {"args": [...], "returns": "...", "expr": {...}, "source_refs": [...]}
    arg_defs = spec.get("args", []) or []
    tv = evaluate_expr(
        expr=spec.get("expr"),
        arg_defs=arg_defs,
        args=args,
        indices=indices,
        tables=tables,
        index_units=index_units,
    )
    declared = spec.get("returns")
    if isinstance(declared, str):
        expected = "decimal" if declared == "integer" else declared
        if tv.kind != expected:
            raise TypeError(f"Inline expr returns {tv.kind}, but schema declares returns={declared!r}")
    return tv
