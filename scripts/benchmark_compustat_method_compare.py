#!/usr/bin/env python3
from __future__ import annotations

import ast
import argparse
import csv
import json
import math
import random
import re
import sys
import time
from collections import Counter
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Iterable


class BenchmarkError(RuntimeError):
    pass


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


WEIGHTS = {
    "grounded_definition_rate": 30.0,
    "formula_validity_rate": 25.0,
    "metric_coverage_rate": 20.0,
    "non_compustat_explanation_rate": 15.0,
    "pairwise_judge_rate": 10.0,
}

FORMULA_ALLOWED_RE = re.compile(r"^[A-Za-z0-9_+\-*/().,\s]+$")
FORMULA_IDENT_RE = re.compile(r"\b([A-Za-z_][A-Za-z0-9_]*)\b")
FORMULA_NUMBER_RE = re.compile(r"\b\d+(?:\.\d+)?\b")
FORMULA_OPERATOR_RE = re.compile(r"[+\-*/]")
FORMULA_CASE_KEYWORD_RE = re.compile(r"\b(case|when|then|end)\b", flags=re.IGNORECASE)
FORMULA_BOOLEAN_KEYWORD_RE = re.compile(r"\b(and|or|not)\b", flags=re.IGNORECASE)

RESIDUAL_TOL_ABS = 1e-9
RESIDUAL_TOL_REL = 1e-7
RESIDUAL_TARGET_VALID_SAMPLES = 24
RESIDUAL_MAX_ATTEMPTS = 120
OUTCOME_METRIC_CANONICAL_TOKENS = (
    "pricingregime",
    "pricinglevel",
    "pricingtier",
    "applicablemargin",
    "applicablefee",
    "applicablespread",
)


@dataclass
class NormalizedRecord:
    item_id: str
    metric_name: str
    method: str
    expected: bool
    returned: bool
    definition_present: bool
    source_refs_valid: bool
    definition_grounded: bool
    formula_present: bool
    formula_valid: bool
    formula: str | None
    formula_variables: list[str]
    requires_non_compustat: bool
    explanation_required: bool
    explanation_quality_ok: bool
    source_refs: list[str]
    notes_count: int
    limitations_count: int
    non_compustat_reason: str | None


@dataclass
class FormulaDiff:
    a_formula: str | None
    b_formula: str | None
    a_normalized: str | None
    b_normalized: str | None
    formulas_identical_normalized: bool
    shared_variables: dict[str, int]
    a_only_variables: dict[str, int]
    b_only_variables: dict[str, int]
    shared_operators: dict[str, int]
    a_only_operators: dict[str, int]
    b_only_operators: dict[str, int]
    shared_numbers: dict[str, int]
    a_only_numbers: dict[str, int]
    b_only_numbers: dict[str, int]
    variable_jaccard: float | None
    operator_jaccard: float | None
    residual_expression: str | None
    residual_status: str
    residual_equivalent: bool | None
    residual_error: str | None
    residual_edge_flags: list[str]
    residual_valid_samples: int
    residual_skipped_samples: int
    residual_max_abs_error: float | None
    residual_mean_abs_error: float | None


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def _read_json(path: Path) -> Any:
    return json.loads(_read_text(path))


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, sort_keys=True) + "\n")


def _stable_metric_name(name: Any) -> str | None:
    if not isinstance(name, str):
        return None
    cleaned = name.strip()
    return cleaned or None


def _load_item_ids(dataset_path: Path) -> list[str]:
    doc = _read_json(dataset_path)
    if not isinstance(doc, dict):
        raise BenchmarkError(f"Dataset file must be object: {dataset_path}")
    item_ids = doc.get("item_ids")
    if not isinstance(item_ids, list) or not all(isinstance(x, str) and x.strip() for x in item_ids):
        raise BenchmarkError(f"Dataset item_ids must be list[str]: {dataset_path}")
    return [x.strip() for x in item_ids]


def _load_allowlist_codes(path: Path) -> set[str]:
    doc = _read_json(path)
    if not isinstance(doc, dict):
        raise BenchmarkError(f"Allowlist file must be object: {path}")
    variables = doc.get("variables")
    if not isinstance(variables, list):
        raise BenchmarkError(f"Allowlist variables must be list: {path}")
    out: set[str] = set()
    for row in variables:
        if not isinstance(row, dict):
            continue
        code = row.get("code")
        if isinstance(code, str) and code.strip():
            out.add(code.strip().lower())
    if not out:
        raise BenchmarkError(f"Allowlist produced zero variable codes: {path}")
    return out


def _load_anchor_ids(run_dir: Path, item_id: str) -> set[str]:
    anchors_tsv = run_dir / "normalized" / item_id / "anchors.tsv"
    if not anchors_tsv.exists():
        return set()
    rows = _read_text(anchors_tsv).splitlines()
    if not rows:
        return set()
    header = rows[0].split("\t")
    try:
        idx = header.index("anchor_id")
    except ValueError:
        return set()
    out: set[str] = set()
    for line in rows[1:]:
        parts = line.split("\t")
        if idx >= len(parts):
            continue
        aid = parts[idx].strip()
        if aid:
            out.add(aid)
    return out


def _iter_metric_names_from_conditions(conditions: Any) -> set[str]:
    out: set[str] = set()
    if isinstance(conditions, dict):
        for key in ("all", "any"):
            rows = conditions.get(key)
            if not isinstance(rows, list):
                continue
            for row in rows:
                if not isinstance(row, dict):
                    continue
                metric = _stable_metric_name(row.get("metric"))
                if metric:
                    out.add(metric)
    elif isinstance(conditions, list):
        for row in conditions:
            if not isinstance(row, dict):
                continue
            metric = _stable_metric_name(row.get("metric"))
            if metric:
                out.add(metric)
    return out


def _extract_expected_metrics(pricing_doc: dict[str, Any]) -> list[str]:
    ordered: list[str] = []
    seen: set[str] = set()

    def add(name: Any) -> None:
        metric = _stable_metric_name(name)
        if not metric or metric in seen:
            return
        seen.add(metric)
        ordered.append(metric)

    metrics = pricing_doc.get("metrics")
    if isinstance(metrics, list):
        for row in metrics:
            if isinstance(row, dict):
                add(row.get("name"))

    tier_schemes = pricing_doc.get("tier_schemes")
    if isinstance(tier_schemes, list):
        for scheme in tier_schemes:
            if not isinstance(scheme, dict):
                continue
            add(scheme.get("metric"))
            tiers = scheme.get("tiers")
            if not isinstance(tiers, list):
                continue
            for tier in tiers:
                if isinstance(tier, dict):
                    for metric in _iter_metric_names_from_conditions(tier.get("conditions")):
                        add(metric)

    facilities = pricing_doc.get("facilities")
    if isinstance(facilities, list):
        for facility in facilities:
            if not isinstance(facility, dict):
                continue
            rates = facility.get("rates")
            if not isinstance(rates, list):
                continue
            for rate in rates:
                if not isinstance(rate, dict):
                    continue
                condition = rate.get("condition")
                if isinstance(condition, dict):
                    add(condition.get("metric"))
    return ordered


def _is_outcome_metric_name(metric_name: str) -> bool:
    canonical = re.sub(r"[^a-z0-9]+", "", str(metric_name or "").lower())
    if not canonical:
        return False
    return any(token in canonical for token in OUTCOME_METRIC_CANONICAL_TOKENS)


def _load_pricing_doc(pricing_dir: Path, item_id: str) -> dict[str, Any]:
    candidates = [
        pricing_dir / item_id / "llm_output.txt",
        pricing_dir / f"{item_id}.json",
        pricing_dir / f"{item_id}.txt",
        pricing_dir / item_id / f"{item_id}.txt",
    ]
    for path in candidates:
        if not path.exists():
            continue
        doc = _read_json(path)
        if isinstance(doc, dict):
            return doc
    raise BenchmarkError(f"Missing pricing second-pass JSON for {item_id} under {pricing_dir}")


def _load_method_a_item(
    *,
    run_dir: Path,
    item_id: str,
    definitions_subdir: str,
    overlay_subdir: str,
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    defs_path = run_dir / "definitions_compiler_v1" / definitions_subdir / f"{item_id}__compiled.json"
    ovl_path = run_dir / "compustat_overlay_v1" / overlay_subdir / f"{item_id}__compustat_overlay.json"

    definitions_by_metric: dict[str, dict[str, Any]] = {}
    overlays_by_metric: dict[str, dict[str, Any]] = {}

    if defs_path.exists():
        defs_doc = _read_json(defs_path)
        defs = defs_doc.get("definitions") if isinstance(defs_doc, dict) else None
        if isinstance(defs, list):
            for row in defs:
                if not isinstance(row, dict):
                    continue
                metric = _stable_metric_name(row.get("metric_name")) or _stable_metric_name(row.get("name"))
                if not metric:
                    continue
                definitions_by_metric[metric] = row

    if ovl_path.exists():
        ovl_doc = _read_json(ovl_path)
        overlays = ovl_doc.get("overlays") if isinstance(ovl_doc, dict) else None
        if isinstance(overlays, list):
            for row in overlays:
                if not isinstance(row, dict):
                    continue
                metric = _stable_metric_name(row.get("term"))
                if not metric:
                    continue
                overlays_by_metric[metric] = row

    return definitions_by_metric, overlays_by_metric


def _discover_method_b_item_dir(method_b_dir: Path, item_id: str, report_doc: dict[str, Any] | None) -> Path | None:
    if isinstance(report_doc, dict):
        for row in report_doc.get("items") or []:
            if not isinstance(row, dict):
                continue
            if str(row.get("item_id") or "") != item_id:
                continue
            best = row.get("best")
            if isinstance(best, dict):
                item_out = best.get("item_out")
                if isinstance(item_out, str) and item_out.strip():
                    p = Path(item_out.strip())
                    if not p.is_absolute():
                        p = (method_b_dir / p).resolve()
                    if p.exists():
                        return p

    direct_best = method_b_dir / "best" / item_id
    if (direct_best / "llm_output.txt").exists():
        return direct_best

    # Non-auto mode layouts: <strategy>/<item_id>/llm_output.txt
    for strategy_dir in sorted([p for p in method_b_dir.iterdir() if p.is_dir()]):
        candidate = strategy_dir / item_id
        if (candidate / "llm_output.txt").exists():
            return candidate

    attempts_dir = method_b_dir / "attempts"
    if attempts_dir.exists():
        for strategy_dir in sorted([p for p in attempts_dir.iterdir() if p.is_dir()]):
            candidate = strategy_dir / item_id
            if (candidate / "llm_output.txt").exists():
                return candidate
    return None


def _load_method_b_item(
    *,
    method_b_dir: Path,
    item_id: str,
    report_doc: dict[str, Any] | None,
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]], Path | None]:
    item_dir = _discover_method_b_item_dir(method_b_dir, item_id, report_doc)
    if item_dir is None:
        return {}, [], None
    llm_output_path = item_dir / "llm_output.txt"
    if not llm_output_path.exists():
        return {}, [], item_dir
    doc = _read_json(llm_output_path)
    if not isinstance(doc, dict):
        return {}, [], item_dir
    metrics = doc.get("metrics")
    unresolved = doc.get("unresolved_dependencies")
    by_metric: dict[str, dict[str, Any]] = {}
    if isinstance(metrics, list):
        for row in metrics:
            if not isinstance(row, dict):
                continue
            metric = _stable_metric_name(row.get("metric_name")) or _stable_metric_name(row.get("name"))
            if not metric:
                continue
            by_metric[metric] = row
    unresolved_rows = unresolved if isinstance(unresolved, list) else []
    return by_metric, [r for r in unresolved_rows if isinstance(r, dict)], item_dir


def _formula_identifiers(formula: str) -> list[str]:
    return [m.group(1).lower() for m in FORMULA_IDENT_RE.finditer(formula or "")]


def _normalize_formula_text(formula: str | None) -> str | None:
    if not isinstance(formula, str):
        return None
    cleaned = formula.strip()
    if not cleaned:
        return None
    return re.sub(r"\s+", "", cleaned).lower()


def _counter_to_sorted_dict(counter: Counter[str]) -> dict[str, int]:
    return {k: int(counter[k]) for k in sorted(counter.keys())}


def _jaccard_from_keys(a: Counter[str], b: Counter[str]) -> float | None:
    keys_a = set(a.keys())
    keys_b = set(b.keys())
    union = keys_a | keys_b
    if not union:
        return None
    inter = keys_a & keys_b
    return len(inter) / len(union)


def _detect_non_arithmetic_flags(formula: str | None, *, side: str) -> list[str]:
    if not isinstance(formula, str) or not formula.strip():
        return []
    txt = formula.strip()
    flags: list[str] = []
    if FORMULA_CASE_KEYWORD_RE.search(txt):
        flags.append(f"{side}_case_construct")
    if FORMULA_BOOLEAN_KEYWORD_RE.search(txt):
        flags.append(f"{side}_boolean_keyword")
    if any(ch in txt for ch in "<>="):
        flags.append(f"{side}_comparison_operator")
    if "," in txt:
        flags.append(f"{side}_comma_token")
    if "**" in txt:
        flags.append(f"{side}_power_operator")
    if "//" in txt:
        flags.append(f"{side}_floor_division_operator")
    return sorted(set(flags))


def _validate_arithmetic_ast(node: ast.AST, names_out: set[str]) -> None:
    if isinstance(node, ast.Expression):
        _validate_arithmetic_ast(node.body, names_out)
        return
    if isinstance(node, ast.BinOp):
        if not isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div)):
            raise ValueError(f"unsupported_operator:{type(node.op).__name__}")
        _validate_arithmetic_ast(node.left, names_out)
        _validate_arithmetic_ast(node.right, names_out)
        return
    if isinstance(node, ast.UnaryOp):
        if not isinstance(node.op, (ast.UAdd, ast.USub)):
            raise ValueError(f"unsupported_unary_operator:{type(node.op).__name__}")
        _validate_arithmetic_ast(node.operand, names_out)
        return
    if isinstance(node, ast.Name):
        ident = str(node.id or "").strip().lower()
        if not ident:
            raise ValueError("empty_identifier")
        names_out.add(ident)
        return
    if isinstance(node, ast.Constant):
        value = node.value
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("non_numeric_constant")
        if not math.isfinite(float(value)):
            raise ValueError("non_finite_constant")
        return
    if isinstance(node, ast.Num):  # pragma: no cover - py<3.8 compatibility
        value = node.n
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("non_numeric_constant")
        if not math.isfinite(float(value)):
            raise ValueError("non_finite_constant")
        return
    raise ValueError(f"unsupported_syntax:{type(node).__name__}")


def _parse_arithmetic_formula(expr: str) -> tuple[ast.Expression | None, set[str], str | None]:
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError as exc:
        return None, set(), f"syntax_error:{exc.msg}"
    names: set[str] = set()
    try:
        _validate_arithmetic_ast(tree, names)
    except ValueError as exc:
        return None, names, str(exc)
    return tree, names, None


def _eval_arithmetic_ast(node: ast.AST, env: dict[str, float]) -> float:
    if isinstance(node, ast.Expression):
        return _eval_arithmetic_ast(node.body, env)
    if isinstance(node, ast.BinOp):
        left = _eval_arithmetic_ast(node.left, env)
        right = _eval_arithmetic_ast(node.right, env)
        if isinstance(node.op, ast.Add):
            out = left + right
        elif isinstance(node.op, ast.Sub):
            out = left - right
        elif isinstance(node.op, ast.Mult):
            out = left * right
        elif isinstance(node.op, ast.Div):
            if abs(right) < 1e-12:
                raise ZeroDivisionError("near-zero denominator")
            out = left / right
        else:  # pragma: no cover - guarded by validator
            raise ValueError(f"unsupported_operator:{type(node.op).__name__}")
        if not math.isfinite(out):
            raise OverflowError("non-finite evaluation")
        return float(out)
    if isinstance(node, ast.UnaryOp):
        value = _eval_arithmetic_ast(node.operand, env)
        if isinstance(node.op, ast.UAdd):
            return +value
        if isinstance(node.op, ast.USub):
            return -value
        raise ValueError(f"unsupported_unary_operator:{type(node.op).__name__}")  # pragma: no cover
    if isinstance(node, ast.Name):
        key = str(node.id or "").strip().lower()
        if key not in env:
            raise KeyError(f"missing_variable:{key}")
        return float(env[key])
    if isinstance(node, ast.Constant):
        value = float(node.value)
        if not math.isfinite(value):
            raise OverflowError("non-finite_constant")
        return value
    if isinstance(node, ast.Num):  # pragma: no cover - py<3.8 compatibility
        value = float(node.n)
        if not math.isfinite(value):
            raise OverflowError("non-finite_constant")
        return value
    raise ValueError(f"unsupported_syntax:{type(node).__name__}")  # pragma: no cover


def _residual_numeric_analysis(
    *,
    expr_a: ast.Expression,
    expr_b: ast.Expression,
    variables: set[str],
) -> tuple[str, bool | None, int, int, float | None, float | None]:
    rng = random.Random(20260206)
    valid = 0
    skipped = 0
    errors: list[float] = []
    max_attempts = max(RESIDUAL_MAX_ATTEMPTS, RESIDUAL_TARGET_VALID_SAMPLES * 3)
    vars_sorted = sorted(v for v in variables if isinstance(v, str) and v)

    for _attempt in range(max_attempts):
        if valid >= RESIDUAL_TARGET_VALID_SAMPLES:
            break
        env: dict[str, float] = {}
        for v in vars_sorted:
            mag = 0.25 + (19.75 * rng.random())
            sign = -1.0 if rng.random() < 0.5 else 1.0
            env[v] = sign * mag
        try:
            aval = _eval_arithmetic_ast(expr_a, env)
            bval = _eval_arithmetic_ast(expr_b, env)
        except (ZeroDivisionError, OverflowError, ValueError, KeyError):
            skipped += 1
            continue
        diff = aval - bval
        if not math.isfinite(diff):
            skipped += 1
            continue
        errors.append(abs(diff))
        valid += 1

    if valid == 0:
        return "insufficient_valid_samples", None, valid, skipped, None, None
    max_abs = max(errors)
    mean_abs = sum(errors) / len(errors)
    equivalent = max_abs <= RESIDUAL_TOL_ABS
    if equivalent:
        return "numerically_equivalent", True, valid, skipped, max_abs, mean_abs
    return "numerically_different", False, valid, skipped, max_abs, mean_abs


def _formula_residual_diff(
    *,
    formula_a: str | None,
    formula_b: str | None,
    normalized_a: str | None,
    normalized_b: str | None,
    normalized_equal: bool,
) -> tuple[str | None, str, bool | None, str | None, list[str], int, int, float | None, float | None]:
    if normalized_a is None or normalized_b is None:
        return None, "missing_formula", None, None, [], 0, 0, None, None

    residual_expression = f"({normalized_a})-({normalized_b})"
    if normalized_equal:
        return residual_expression, "identical_normalized", True, None, [], 0, 0, 0.0, 0.0

    edge_flags = []
    edge_flags.extend(_detect_non_arithmetic_flags(formula_a, side="a"))
    edge_flags.extend(_detect_non_arithmetic_flags(formula_b, side="b"))
    edge_flags = sorted(set(edge_flags))
    if edge_flags:
        return (
            residual_expression,
            "unsupported_non_arithmetic",
            None,
            "non_arithmetic_construct_detected",
            edge_flags,
            0,
            0,
            None,
            None,
        )

    tree_a, names_a, err_a = _parse_arithmetic_formula(normalized_a)
    if err_a or tree_a is None:
        return residual_expression, "parse_error", None, f"a:{err_a or 'parse_failed'}", edge_flags, 0, 0, None, None
    tree_b, names_b, err_b = _parse_arithmetic_formula(normalized_b)
    if err_b or tree_b is None:
        return residual_expression, "parse_error", None, f"b:{err_b or 'parse_failed'}", edge_flags, 0, 0, None, None

    status, equivalent, valid, skipped, max_abs, mean_abs = _residual_numeric_analysis(
        expr_a=tree_a,
        expr_b=tree_b,
        variables=names_a | names_b,
    )
    return residual_expression, status, equivalent, None, edge_flags, valid, skipped, max_abs, mean_abs


def _formula_structural_diff(formula_a: str | None, formula_b: str | None) -> FormulaDiff:
    fa = _normalize_formula_text(formula_a)
    fb = _normalize_formula_text(formula_b)

    ids_a = Counter(_formula_identifiers(fa or ""))
    ids_b = Counter(_formula_identifiers(fb or ""))
    ops_a = Counter(FORMULA_OPERATOR_RE.findall(fa or ""))
    ops_b = Counter(FORMULA_OPERATOR_RE.findall(fb or ""))
    nums_a = Counter(FORMULA_NUMBER_RE.findall(fa or ""))
    nums_b = Counter(FORMULA_NUMBER_RE.findall(fb or ""))

    shared_ids = ids_a & ids_b
    a_only_ids = ids_a - ids_b
    b_only_ids = ids_b - ids_a

    shared_ops = ops_a & ops_b
    a_only_ops = ops_a - ops_b
    b_only_ops = ops_b - ops_a

    shared_nums = nums_a & nums_b
    a_only_nums = nums_a - nums_b
    b_only_nums = nums_b - nums_a

    normalized_equal = fa is not None and fa == fb
    (
        residual_expression,
        residual_status,
        residual_equivalent,
        residual_error,
        residual_edge_flags,
        residual_valid_samples,
        residual_skipped_samples,
        residual_max_abs_error,
        residual_mean_abs_error,
    ) = _formula_residual_diff(
        formula_a=formula_a,
        formula_b=formula_b,
        normalized_a=fa,
        normalized_b=fb,
        normalized_equal=normalized_equal,
    )

    return FormulaDiff(
        a_formula=formula_a,
        b_formula=formula_b,
        a_normalized=fa,
        b_normalized=fb,
        formulas_identical_normalized=normalized_equal,
        shared_variables=_counter_to_sorted_dict(shared_ids),
        a_only_variables=_counter_to_sorted_dict(a_only_ids),
        b_only_variables=_counter_to_sorted_dict(b_only_ids),
        shared_operators=_counter_to_sorted_dict(shared_ops),
        a_only_operators=_counter_to_sorted_dict(a_only_ops),
        b_only_operators=_counter_to_sorted_dict(b_only_ops),
        shared_numbers=_counter_to_sorted_dict(shared_nums),
        a_only_numbers=_counter_to_sorted_dict(a_only_nums),
        b_only_numbers=_counter_to_sorted_dict(b_only_nums),
        variable_jaccard=_jaccard_from_keys(ids_a, ids_b),
        operator_jaccard=_jaccard_from_keys(ops_a, ops_b),
        residual_expression=residual_expression,
        residual_status=residual_status,
        residual_equivalent=residual_equivalent,
        residual_error=residual_error,
        residual_edge_flags=residual_edge_flags,
        residual_valid_samples=residual_valid_samples,
        residual_skipped_samples=residual_skipped_samples,
        residual_max_abs_error=residual_max_abs_error,
        residual_mean_abs_error=residual_mean_abs_error,
    )


def _formula_valid(formula: Any, variables: Any, allowlist_codes: set[str]) -> tuple[bool, list[str], str | None]:
    if not isinstance(formula, str) or not formula.strip():
        return False, [], "formula_missing"
    formula_txt = formula.strip()
    if not FORMULA_ALLOWED_RE.fullmatch(formula_txt):
        return False, [], "formula_has_invalid_chars"
    ids = _formula_identifiers(formula_txt)
    if not ids:
        return False, [], "formula_has_no_identifiers"
    unknown = sorted(set(v for v in ids if v not in allowlist_codes))
    if unknown:
        return False, sorted(set(ids)), f"formula_uses_non_allowlist_codes:{','.join(unknown[:10])}"

    vars_norm = sorted(
        set(
            str(v).strip().lower()
            for v in (variables or [])
            if isinstance(v, str) and str(v).strip()
        )
    )
    id_set = sorted(set(ids))
    if vars_norm and vars_norm != id_set:
        return False, id_set, "declared_variables_do_not_match_formula_identifiers"
    return True, id_set, None


def _source_refs_valid(refs: Any, anchor_ids: set[str]) -> tuple[list[str], bool]:
    if not isinstance(refs, list):
        return [], False
    cleaned = [str(x).strip() for x in refs if isinstance(x, str) and str(x).strip()]
    if not cleaned:
        return [], False
    if not anchor_ids:
        return cleaned, False
    ok = all(r in anchor_ids for r in cleaned)
    return cleaned, ok


def _score_method(records: list[NormalizedRecord]) -> dict[str, Any]:
    if not records:
        return {
            "grounded_definition_rate": 0.0,
            "formula_validity_rate": 0.0,
            "metric_coverage_rate": 0.0,
            "non_compustat_explanation_rate": 0.0,
            "deterministic_points": 0.0,
            "counts": {},
        }

    total = len(records)
    grounded = sum(1 for r in records if r.definition_grounded)
    formula_valid = sum(1 for r in records if r.formula_valid)
    covered = sum(1 for r in records if r.returned)
    explanation_required = [r for r in records if r.explanation_required]
    if explanation_required:
        explanation_ok = sum(1 for r in explanation_required if r.explanation_quality_ok)
        explanation_rate = explanation_ok / len(explanation_required)
    else:
        explanation_ok = 0
        explanation_rate = 1.0

    grounded_rate = grounded / total
    formula_rate = formula_valid / total
    coverage_rate = covered / total

    deterministic_points = (
        WEIGHTS["grounded_definition_rate"] * grounded_rate
        + WEIGHTS["formula_validity_rate"] * formula_rate
        + WEIGHTS["metric_coverage_rate"] * coverage_rate
        + WEIGHTS["non_compustat_explanation_rate"] * explanation_rate
    )
    return {
        "grounded_definition_rate": grounded_rate,
        "formula_validity_rate": formula_rate,
        "metric_coverage_rate": coverage_rate,
        "non_compustat_explanation_rate": explanation_rate,
        "deterministic_points": deterministic_points,
        "counts": {
            "total_expected_metrics": total,
            "grounded_definitions": grounded,
            "formula_valid": formula_valid,
            "returned_metrics": covered,
            "explanations_required": len(explanation_required),
            "explanations_ok": explanation_ok,
        },
    }


def _format_counts_for_csv(counts: dict[str, int]) -> str:
    if not counts:
        return ""
    return ";".join(f"{k}:{v}" for k, v in sorted(counts.items()))


def _summarize_formula_diffs(rows: list[FormulaDiff]) -> dict[str, Any]:
    if not rows:
        return {
            "total_pairs": 0,
            "identical_normalized_count": 0,
            "any_variable_delta_count": 0,
            "any_operator_delta_count": 0,
            "any_number_delta_count": 0,
            "missing_formula_in_any_count": 0,
            "mean_variable_jaccard": None,
            "mean_operator_jaccard": None,
            "residual_status_counts": {},
            "residual_equivalent_count": 0,
            "residual_non_equivalent_count": 0,
            "residual_unresolved_count": 0,
        }
    total = len(rows)
    identical = sum(1 for r in rows if r.formulas_identical_normalized)
    var_delta = sum(1 for r in rows if r.a_only_variables or r.b_only_variables)
    op_delta = sum(1 for r in rows if r.a_only_operators or r.b_only_operators)
    num_delta = sum(1 for r in rows if r.a_only_numbers or r.b_only_numbers)
    missing_any = sum(1 for r in rows if (r.a_normalized is None or r.b_normalized is None))
    var_j = [r.variable_jaccard for r in rows if r.variable_jaccard is not None]
    op_j = [r.operator_jaccard for r in rows if r.operator_jaccard is not None]
    residual_status_counts = dict(Counter(r.residual_status for r in rows))
    residual_equivalent_count = sum(1 for r in rows if r.residual_equivalent is True)
    residual_non_equivalent_count = sum(1 for r in rows if r.residual_equivalent is False)
    residual_unresolved_count = sum(1 for r in rows if r.residual_equivalent is None)
    return {
        "total_pairs": total,
        "identical_normalized_count": identical,
        "any_variable_delta_count": var_delta,
        "any_operator_delta_count": op_delta,
        "any_number_delta_count": num_delta,
        "missing_formula_in_any_count": missing_any,
        "mean_variable_jaccard": (sum(var_j) / len(var_j)) if var_j else None,
        "mean_operator_jaccard": (sum(op_j) / len(op_j)) if op_j else None,
        "residual_status_counts": residual_status_counts,
        "residual_equivalent_count": residual_equivalent_count,
        "residual_non_equivalent_count": residual_non_equivalent_count,
        "residual_unresolved_count": residual_unresolved_count,
    }


def _build_judge_prompt(metric_name: str, method_a: dict[str, Any], method_b: dict[str, Any]) -> str:
    return (
        "You are comparing two extraction methods for one covenant/pricing metric.\n"
        "Pick the better output quality based on:\n"
        "1) Definition grounding and evidence refs\n"
        "2) Formula plausibility and allowlist discipline\n"
        "3) Clarity of non-Compustat explanation when needed\n\n"
        "Return JSON only with EXACT keys winner, reason.\n"
        "winner must be one of: A, B, tie.\n"
        "reason must be <= 40 words.\n\n"
        f"METRIC: {metric_name}\n\n"
        "METHOD_A:\n"
        f"{json.dumps(method_a, indent=2, sort_keys=True)}\n\n"
        "METHOD_B:\n"
        f"{json.dumps(method_b, indent=2, sort_keys=True)}\n"
    )


def _run_pairwise_judge(
    *,
    samples: list[dict[str, Any]],
    model: str,
    reasoning: str,
    temperature: float,
    timeout_seconds: float,
    gateway_url: str | None,
) -> tuple[list[dict[str, Any]], float | None, float | None]:
    if not samples:
        return [], None, None

    from src.pipeline.evidence.indexing import DEFAULT_GATEWAY_URL, _ensure_gateway_client_sync

    complete = _ensure_gateway_client_sync()
    base_url = gateway_url or DEFAULT_GATEWAY_URL

    judged: list[dict[str, Any]] = []
    a_points = 0.0
    b_points = 0.0
    decided = 0

    for sample in samples:
        metric_name = str(sample.get("metric_name") or "")
        prompt = _build_judge_prompt(metric_name, sample["method_a"], sample["method_b"])
        winner = None
        reason = None
        raw_text = None
        error = None
        try:
            raw_text = complete(
                model=model,
                prompt=prompt,
                base_url=base_url,
                reasoning={"effort": reasoning} if reasoning else None,
                temperature=temperature,
                max_output_tokens=None,
                timeout=timeout_seconds,
            )
            parsed = json.loads(str(raw_text or "").strip())
            winner_val = str(parsed.get("winner") or "").strip().lower()
            if winner_val in {"a", "b", "tie"}:
                winner = winner_val
            reason_val = parsed.get("reason")
            if isinstance(reason_val, str):
                reason = reason_val.strip()
            if winner is None:
                error = f"invalid_winner:{winner_val!r}"
        except Exception as exc:  # pragma: no cover - network path
            error = f"{type(exc).__name__}: {exc}"

        if winner in {"a", "b", "tie"}:
            decided += 1
            if winner == "a":
                a_points += 1.0
            elif winner == "b":
                b_points += 1.0
            else:
                a_points += 0.5
                b_points += 0.5

        judged.append(
            {
                "item_id": sample["item_id"],
                "metric_name": metric_name,
                "winner": winner,
                "reason": reason,
                "error": error,
                "raw_response": raw_text,
            }
        )

    if decided == 0:
        return judged, None, None
    return judged, a_points / decided, b_points / decided


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark Compustat mapping quality between method A (definitions+overlay) "
            "and method B (one-step definitions+mapping prompt)."
        )
    )
    parser.add_argument("--run-id", required=True, help="Run ID containing normalized/, definitions_compiler_v1/, compustat_overlay_v1/.")
    parser.add_argument("--method-a-definitions-subdir", required=True, help="Subdir under runs/<run_id>/definitions_compiler_v1/.")
    parser.add_argument("--method-a-overlay-subdir", required=True, help="Subdir under runs/<run_id>/compustat_overlay_v1/.")
    parser.add_argument("--method-b-dir", required=True, help="Path to pricing_definitions_third_pass_runner outputs.")
    parser.add_argument(
        "--pricing-second-pass-dir",
        required=True,
        help="Directory containing pricing second-pass JSON outputs used to derive expected metric names.",
    )
    parser.add_argument("--dataset", default="datasets/test_sample_2_agreements.json")
    parser.add_argument("--allowlist", default="datasets/compustat_allowlist_quarterly_v1.json")
    parser.add_argument("--out-dir", default=None, help="Output directory (default: scratch/benchmarks/compustat_method_compare_<timestamp>/)")
    parser.add_argument("--judge-model", default=None, help="Optional high-quality model name for pairwise A/B judge.")
    parser.add_argument("--judge-reasoning", default="medium")
    parser.add_argument("--judge-temperature", type=float, default=0.0)
    parser.add_argument("--judge-timeout-seconds", type=float, default=180.0)
    parser.add_argument("--judge-max-samples", type=int, default=120)
    parser.add_argument("--judge-gateway-url", default=None)
    parser.add_argument(
        "--include-outcome-metrics",
        action="store_true",
        help=(
            "Include outcome-label metrics (e.g., PricingRegime/PricingLevel/ApplicableMargin). "
            "By default these are excluded from benchmark rows."
        ),
    )
    parser.add_argument("--base-dir", default=".")
    args = parser.parse_args()

    root = Path(args.base_dir).resolve()
    run_dir = root / "runs" / str(args.run_id)
    if not run_dir.exists():
        raise SystemExit(f"run-id not found: {run_dir}")

    method_b_dir = Path(args.method_b_dir).resolve()
    if not method_b_dir.exists():
        raise SystemExit(f"method-b-dir not found: {method_b_dir}")

    pricing_second_pass_dir = Path(args.pricing_second_pass_dir).resolve()
    if not pricing_second_pass_dir.exists():
        raise SystemExit(f"pricing-second-pass-dir not found: {pricing_second_pass_dir}")

    dataset_path = Path(args.dataset)
    if not dataset_path.is_absolute():
        dataset_path = (root / dataset_path).resolve()
    allowlist_path = Path(args.allowlist)
    if not allowlist_path.is_absolute():
        allowlist_path = (root / allowlist_path).resolve()

    item_ids = _load_item_ids(dataset_path)
    allowlist_codes = _load_allowlist_codes(allowlist_path)

    out_dir = Path(args.out_dir).resolve() if args.out_dir else (
        root / "scratch" / "benchmarks" / f"compustat_method_compare_{time.strftime('%Y%m%d_%H%M%S')}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    report_doc_for_method_b = None
    report_path_for_method_b = method_b_dir / "report.json"
    if report_path_for_method_b.exists():
        try:
            loaded_report = _read_json(report_path_for_method_b)
            if isinstance(loaded_report, dict):
                report_doc_for_method_b = loaded_report
        except Exception:
            report_doc_for_method_b = None

    records_a: list[NormalizedRecord] = []
    records_b: list[NormalizedRecord] = []
    paired_rows: list[dict[str, Any]] = []
    formula_diffs: list[FormulaDiff] = []
    formula_diff_rows: list[dict[str, Any]] = []
    zero_metric_details: list[dict[str, Any]] = []
    filtered_outcome_metrics_details: list[dict[str, Any]] = []

    for item_id in item_ids:
        anchor_ids = _load_anchor_ids(run_dir, item_id)
        pricing_doc = _load_pricing_doc(pricing_second_pass_dir, item_id)
        expected_metrics_raw = _extract_expected_metrics(pricing_doc)
        excluded_metrics = [m for m in expected_metrics_raw if _is_outcome_metric_name(m)]
        if args.include_outcome_metrics:
            expected_metrics = expected_metrics_raw
            excluded_metrics = []
        else:
            expected_metrics = [m for m in expected_metrics_raw if not _is_outcome_metric_name(m)]
        if excluded_metrics:
            filtered_outcome_metrics_details.append(
                {
                    "item_id": item_id,
                    "excluded_metrics": excluded_metrics,
                }
            )
        expected_set = set(expected_metrics)

        defs_a, ovl_a = _load_method_a_item(
            run_dir=run_dir,
            item_id=item_id,
            definitions_subdir=str(args.method_a_definitions_subdir),
            overlay_subdir=str(args.method_a_overlay_subdir),
        )
        b_map, b_unresolved, b_item_dir = _load_method_b_item(
            method_b_dir=method_b_dir,
            item_id=item_id,
            report_doc=report_doc_for_method_b,
        )

        if not expected_metrics:
            zero_metric_details.append(
                {
                    "item_id": item_id,
                    "method_a_returned_metric_count": len(set(defs_a.keys()) | set(ovl_a.keys())),
                    "method_b_returned_metric_count": len(b_map),
                    "method_b_item_dir": str(b_item_dir) if b_item_dir else None,
                }
            )
            continue

        unresolved_by_metric_b: dict[str, list[dict[str, Any]]] = {m: [] for m in expected_set}
        for row in b_unresolved:
            refs = row.get("referenced_by")
            if not isinstance(refs, list):
                continue
            for metric in refs:
                metric_name = _stable_metric_name(metric)
                if metric_name and metric_name in unresolved_by_metric_b:
                    unresolved_by_metric_b[metric_name].append(row)

        for metric_name in expected_metrics:
            d_a = defs_a.get(metric_name, {})
            o_a = ovl_a.get(metric_name, {})

            definition_verbatim_a = d_a.get("definition_verbatim")
            source_refs_a, refs_valid_a = _source_refs_valid(d_a.get("source_refs"), anchor_ids)
            definition_present_a = isinstance(definition_verbatim_a, str) and bool(definition_verbatim_a.strip())
            grounded_a = definition_present_a and refs_valid_a

            formula_a = o_a.get("compustat_formula")
            vars_a = o_a.get("compustat_variables")
            formula_valid_a, vars_norm_a, _formula_err_a = _formula_valid(formula_a, vars_a, allowlist_codes)
            formula_present_a = isinstance(formula_a, str) and bool(formula_a.strip())

            match_type_a = str(o_a.get("match_type") or "").strip().lower()
            limitations_a = o_a.get("limitations") if isinstance(o_a.get("limitations"), list) else []
            notes_a = o_a.get("notes") if isinstance(o_a.get("notes"), list) else []
            explanation_required_a = (not formula_valid_a) or match_type_a == "none"
            explanation_quality_a = (len(limitations_a) > 0 or len(notes_a) > 0) if explanation_required_a else True

            rec_a = NormalizedRecord(
                item_id=item_id,
                metric_name=metric_name,
                method="A",
                expected=True,
                returned=bool(metric_name in defs_a or metric_name in ovl_a),
                definition_present=definition_present_a,
                source_refs_valid=refs_valid_a,
                definition_grounded=grounded_a,
                formula_present=formula_present_a,
                formula_valid=formula_valid_a,
                formula=str(formula_a).strip() if isinstance(formula_a, str) and formula_a.strip() else None,
                formula_variables=vars_norm_a,
                requires_non_compustat=match_type_a == "none",
                explanation_required=explanation_required_a,
                explanation_quality_ok=explanation_quality_a,
                source_refs=source_refs_a,
                notes_count=len([n for n in notes_a if isinstance(n, str)]),
                limitations_count=len([n for n in limitations_a if isinstance(n, str)]),
                non_compustat_reason=None,
            )
            records_a.append(rec_a)

            d_b = b_map.get(metric_name, {})
            definition_verbatim_b = d_b.get("definition_verbatim")
            source_refs_b, refs_valid_b = _source_refs_valid(d_b.get("source_refs"), anchor_ids)
            definition_present_b = isinstance(definition_verbatim_b, str) and bool(definition_verbatim_b.strip())
            grounded_b = definition_present_b and refs_valid_b

            formula_b = d_b.get("formula_compustat")
            vars_b = d_b.get("compustat_variables")
            formula_valid_b, vars_norm_b, _formula_err_b = _formula_valid(formula_b, vars_b, allowlist_codes)
            formula_present_b = isinstance(formula_b, str) and bool(formula_b.strip())
            requires_non_compustat_b = bool(d_b.get("requires_non_compustat"))
            non_compustat_reason_b = d_b.get("non_compustat_reason")
            notes_b = d_b.get("notes") if isinstance(d_b.get("notes"), list) else []
            refinements_b = d_b.get("refinements") if isinstance(d_b.get("refinements"), list) else []

            explanation_required_b = requires_non_compustat_b or (not formula_valid_b)
            explanation_quality_b = True
            if explanation_required_b:
                has_reason = isinstance(non_compustat_reason_b, str) and bool(non_compustat_reason_b.strip())
                has_supporting_notes = bool(notes_b or refinements_b or unresolved_by_metric_b.get(metric_name))
                explanation_quality_b = has_reason or has_supporting_notes

            rec_b = NormalizedRecord(
                item_id=item_id,
                metric_name=metric_name,
                method="B",
                expected=True,
                returned=metric_name in b_map,
                definition_present=definition_present_b,
                source_refs_valid=refs_valid_b,
                definition_grounded=grounded_b,
                formula_present=formula_present_b,
                formula_valid=formula_valid_b,
                formula=str(formula_b).strip() if isinstance(formula_b, str) and formula_b.strip() else None,
                formula_variables=vars_norm_b,
                requires_non_compustat=requires_non_compustat_b,
                explanation_required=explanation_required_b,
                explanation_quality_ok=explanation_quality_b,
                source_refs=source_refs_b,
                notes_count=len([n for n in notes_b if isinstance(n, str)]),
                limitations_count=len([n for n in refinements_b if isinstance(n, str)]),
                non_compustat_reason=str(non_compustat_reason_b).strip()
                if isinstance(non_compustat_reason_b, str) and non_compustat_reason_b.strip()
                else None,
            )
            records_b.append(rec_b)
            formula_diff = _formula_structural_diff(rec_a.formula, rec_b.formula)

            paired_rows.append(
                {
                    "item_id": item_id,
                    "metric_name": metric_name,
                    "expected": True,
                    "method_a": asdict(rec_a),
                    "method_b": asdict(rec_b),
                    "formula_diff": asdict(formula_diff),
                    "method_b_item_dir": str(b_item_dir) if b_item_dir else None,
                }
            )
            formula_diff_rows.append(
                {
                    "item_id": item_id,
                    "metric_name": metric_name,
                    **asdict(formula_diff),
                }
            )
            formula_diffs.append(formula_diff)

    method_a_scores = _score_method(records_a)
    method_b_scores = _score_method(records_b)
    zero_metric_item_ids = [row["item_id"] for row in zero_metric_details]
    zero_metric_summary = {
        "count": len(zero_metric_details),
        "item_ids": zero_metric_item_ids,
        "method_a_returned_metrics_total": int(sum(int(row["method_a_returned_metric_count"]) for row in zero_metric_details)),
        "method_b_returned_metrics_total": int(sum(int(row["method_b_returned_metric_count"]) for row in zero_metric_details)),
        "details": zero_metric_details,
    }

    judge_samples_in: list[dict[str, Any]] = []
    for row in paired_rows:
        judge_samples_in.append(
            {
                "item_id": row["item_id"],
                "metric_name": row["metric_name"],
                "method_a": {
                    "returned": row["method_a"]["returned"],
                    "definition_grounded": row["method_a"]["definition_grounded"],
                    "formula": row["method_a"]["formula"],
                    "formula_valid": row["method_a"]["formula_valid"],
                    "requires_non_compustat": row["method_a"]["requires_non_compustat"],
                    "explanation_quality_ok": row["method_a"]["explanation_quality_ok"],
                },
                "method_b": {
                    "returned": row["method_b"]["returned"],
                    "definition_grounded": row["method_b"]["definition_grounded"],
                    "formula": row["method_b"]["formula"],
                    "formula_valid": row["method_b"]["formula_valid"],
                    "requires_non_compustat": row["method_b"]["requires_non_compustat"],
                    "explanation_quality_ok": row["method_b"]["explanation_quality_ok"],
                    "non_compustat_reason": row["method_b"]["non_compustat_reason"],
                },
            }
        )

    judge_samples = judge_samples_in[: max(0, int(args.judge_max_samples))]
    judge_rows: list[dict[str, Any]] = []
    judge_rate_a: float | None = None
    judge_rate_b: float | None = None
    if args.judge_model:
        judge_rows, judge_rate_a, judge_rate_b = _run_pairwise_judge(
            samples=judge_samples,
            model=str(args.judge_model),
            reasoning=str(args.judge_reasoning),
            temperature=float(args.judge_temperature),
            timeout_seconds=float(args.judge_timeout_seconds),
            gateway_url=args.judge_gateway_url,
        )
    else:
        judge_rows = [
            {
                "item_id": row["item_id"],
                "metric_name": row["metric_name"],
                "winner": None,
                "reason": None,
                "error": "judge_disabled",
                "raw_response": None,
            }
            for row in judge_samples
        ]

    deterministic_weight_total = (
        WEIGHTS["grounded_definition_rate"]
        + WEIGHTS["formula_validity_rate"]
        + WEIGHTS["metric_coverage_rate"]
        + WEIGHTS["non_compustat_explanation_rate"]
    )
    if judge_rate_a is None or judge_rate_b is None:
        composite_a = (float(method_a_scores["deterministic_points"]) / deterministic_weight_total) * 100.0
        composite_b = (float(method_b_scores["deterministic_points"]) / deterministic_weight_total) * 100.0
        judge_component_a = None
        judge_component_b = None
        effective_weight_total = deterministic_weight_total
    else:
        judge_component_a = WEIGHTS["pairwise_judge_rate"] * judge_rate_a
        judge_component_b = WEIGHTS["pairwise_judge_rate"] * judge_rate_b
        composite_a = float(method_a_scores["deterministic_points"]) + judge_component_a
        composite_b = float(method_b_scores["deterministic_points"]) + judge_component_b
        effective_weight_total = deterministic_weight_total + WEIGHTS["pairwise_judge_rate"]

    normalized_records_path = out_dir / "normalized_records.jsonl"
    _write_jsonl(
        normalized_records_path,
        [asdict(r) for r in records_a] + [asdict(r) for r in records_b],
    )
    formula_diff_path = out_dir / "formula_diff.jsonl"
    _write_jsonl(formula_diff_path, formula_diff_rows)
    formula_diff_summary = _summarize_formula_diffs(formula_diffs)

    per_metric_csv_path = out_dir / "per_metric.csv"
    with per_metric_csv_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            [
                "item_id",
                "metric_name",
                "a_returned",
                "a_definition_grounded",
                "a_formula_valid",
                "a_explanation_quality_ok",
                "a_formula",
                "b_returned",
                "b_definition_grounded",
                "b_formula_valid",
                "b_explanation_quality_ok",
                "b_formula",
                "shared_variables",
                "a_only_variables",
                "b_only_variables",
                "shared_operators",
                "a_only_operators",
                "b_only_operators",
                "shared_numbers",
                "a_only_numbers",
                "b_only_numbers",
                "formulas_identical_normalized",
                "variable_jaccard",
                "operator_jaccard",
                "residual_status",
                "residual_equivalent",
                "residual_max_abs_error",
                "residual_mean_abs_error",
                "residual_valid_samples",
                "residual_skipped_samples",
                "residual_edge_flags",
                "residual_expression",
                "residual_error",
                "judge_winner",
            ]
        )

        judge_by_key = {(r["item_id"], r["metric_name"]): (r.get("winner") or "") for r in judge_rows}
        formula_diff_by_key = {(r["item_id"], r["metric_name"]): r for r in formula_diff_rows}
        for row in paired_rows:
            a = row["method_a"]
            b = row["method_b"]
            winner = judge_by_key.get((row["item_id"], row["metric_name"]), "")
            fd = formula_diff_by_key.get((row["item_id"], row["metric_name"]), {})
            writer.writerow(
                [
                    row["item_id"],
                    row["metric_name"],
                    "true" if a["returned"] else "false",
                    "true" if a["definition_grounded"] else "false",
                    "true" if a["formula_valid"] else "false",
                    "true" if a["explanation_quality_ok"] else "false",
                    a["formula"] or "",
                    "true" if b["returned"] else "false",
                    "true" if b["definition_grounded"] else "false",
                    "true" if b["formula_valid"] else "false",
                    "true" if b["explanation_quality_ok"] else "false",
                    b["formula"] or "",
                    _format_counts_for_csv(fd.get("shared_variables") or {}),
                    _format_counts_for_csv(fd.get("a_only_variables") or {}),
                    _format_counts_for_csv(fd.get("b_only_variables") or {}),
                    _format_counts_for_csv(fd.get("shared_operators") or {}),
                    _format_counts_for_csv(fd.get("a_only_operators") or {}),
                    _format_counts_for_csv(fd.get("b_only_operators") or {}),
                    _format_counts_for_csv(fd.get("shared_numbers") or {}),
                    _format_counts_for_csv(fd.get("a_only_numbers") or {}),
                    _format_counts_for_csv(fd.get("b_only_numbers") or {}),
                    "true" if fd.get("formulas_identical_normalized") else "false",
                    fd.get("variable_jaccard"),
                    fd.get("operator_jaccard"),
                    fd.get("residual_status"),
                    "" if fd.get("residual_equivalent") is None else ("true" if fd.get("residual_equivalent") else "false"),
                    fd.get("residual_max_abs_error"),
                    fd.get("residual_mean_abs_error"),
                    fd.get("residual_valid_samples"),
                    fd.get("residual_skipped_samples"),
                    ";".join(fd.get("residual_edge_flags") or []),
                    fd.get("residual_expression") or "",
                    fd.get("residual_error") or "",
                    winner,
                ]
            )

    judge_samples_path = out_dir / "judge_samples.jsonl"
    _write_jsonl(judge_samples_path, judge_rows)

    report = {
        "schema_version": "compustat_method_compare_v1",
        "created_at": int(time.time()),
        "inputs": {
            "run_id": args.run_id,
            "dataset": str(dataset_path),
            "allowlist": str(allowlist_path),
            "pricing_second_pass_dir": str(pricing_second_pass_dir),
            "method_a_definitions_subdir": args.method_a_definitions_subdir,
            "method_a_overlay_subdir": args.method_a_overlay_subdir,
            "method_b_dir": str(method_b_dir),
        },
        "weights": WEIGHTS,
        "effective_weight_total": effective_weight_total,
        "methods": {
            "A_two_step": method_a_scores,
            "B_one_step": method_b_scores,
        },
        "zero_metric_items": zero_metric_summary,
        "formula_structure_summary": formula_diff_summary,
        "metric_filtering": {
            "include_outcome_metrics": bool(args.include_outcome_metrics),
            "excluded_metric_count": int(
                sum(len(row.get("excluded_metrics") or []) for row in filtered_outcome_metrics_details)
            ),
            "excluded_by_item": filtered_outcome_metrics_details,
        },
        "judge": {
            "enabled": bool(args.judge_model),
            "model": args.judge_model,
            "samples_requested": int(args.judge_max_samples),
            "samples_evaluated": len(judge_rows),
            "pairwise_judge_rate_a": judge_rate_a,
            "pairwise_judge_rate_b": judge_rate_b,
        },
        "composite": {
            "A_two_step": {
                "score": composite_a,
                "judge_component": judge_component_a,
            },
            "B_one_step": {
                "score": composite_b,
                "judge_component": judge_component_b,
            },
            "winner": "A_two_step" if composite_a > composite_b else ("B_one_step" if composite_b > composite_a else "tie"),
        },
        "artifacts": {
            "report_json": str(out_dir / "report.json"),
            "per_metric_csv": str(per_metric_csv_path),
            "judge_samples_jsonl": str(judge_samples_path),
            "normalized_records_jsonl": str(normalized_records_path),
            "formula_diff_jsonl": str(formula_diff_path),
        },
    }
    _write_json(out_dir / "report.json", report)

    print("[done] compustat method comparison benchmark complete")
    print(str(out_dir / "report.json"))
    print(str(per_metric_csv_path))
    print(str(judge_samples_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
