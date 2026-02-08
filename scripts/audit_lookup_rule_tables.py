#!/usr/bin/env python
"""
Audit ContractIR v0.2 artifacts that use lookup_rule(...) tables.

Goal (artifact-first):
  - Estimate how often rule tables are well-formed (exactly one match),
    have gaps (no matching row), or overlap (multiple matching rows)
    for representative boundary test points derived from the predicates.

This is NOT a theorem prover; it uses a deterministic sampling strategy:
  - Extract per-variable threshold constants from predicate expressions.
  - Generate boundary + midpoint sample values per variable.
  - Evaluate each row predicate on the Cartesian product (capped).

Outputs:
  - Per-table summary (coverage / overlaps / gaps)
  - Examples of points that trigger gaps or overlaps
  - If overlaps occur, whether overlapping rows produce conflicting outputs.

Example:
  python scripts/audit_lookup_rule_tables.py --runs-root runs --max-files 200
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass, asdict
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"

import sys

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from pipeline.ir.contract_ir_v0_2 import evaluate_expr, validate_contract_ir  # noqa: E402


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8", errors="replace"))


def _iter_contractir_files(runs_root: Path) -> Iterable[Path]:
    for p in runs_root.rglob("contractir_validated.json"):
        yield p


def _walk_ast(node: Any) -> Iterable[dict]:
    if not isinstance(node, dict):
        return
    yield node
    if "args" in node and isinstance(node.get("args"), list):
        for a in node["args"]:
            yield from _walk_ast(a)
    if "lit" in node and isinstance(node.get("lit"), dict):
        return


def _is_string_lit(node: Any) -> Optional[str]:
    if not isinstance(node, dict):
        return None
    lit = node.get("lit")
    if not isinstance(lit, dict):
        return None
    if lit.get("type") != "string":
        return None
    v = lit.get("value")
    return v if isinstance(v, str) else None


def _is_decimal_lit(node: Any) -> Optional[Decimal]:
    if not isinstance(node, dict):
        return None
    lit = node.get("lit")
    if not isinstance(lit, dict):
        return None
    if lit.get("type") not in ("decimal", "rate", "bps", "money", "integer"):
        return None
    v = lit.get("value")
    if not isinstance(v, str):
        return None
    try:
        return Decimal(v)
    except Exception:
        return None


def _is_date_lit(node: Any) -> Optional[str]:
    if not isinstance(node, dict):
        return None
    lit = node.get("lit")
    if not isinstance(lit, dict):
        return None
    if lit.get("type") != "date":
        return None
    v = lit.get("value")
    return v if isinstance(v, str) else None


def _is_var(node: Any) -> Optional[str]:
    if not isinstance(node, dict):
        return None
    v = node.get("var")
    return v if isinstance(v, str) else None


def _parse_iso_date(s: str) -> Tuple[int, int, int]:
    # Minimal parser to avoid importing datetime for speed.
    y, m, d = s.split("-")
    return (int(y), int(m), int(d))


def _iso_date_add_days(s: str, delta_days: int) -> str:
    # Use datetime only here; clearer and less bug-prone than manual month arithmetic.
    import datetime as _dt

    y, m, d = _parse_iso_date(s)
    dt = _dt.date(y, m, d) + _dt.timedelta(days=delta_days)
    return dt.isoformat()


@dataclass(frozen=True)
class LookupRuleCall:
    fn_id: str
    table_id: str
    predicate_col: str
    value_col: str


@dataclass(frozen=True)
class SamplePoint:
    args: Dict[str, Any]
    match_count: int
    matched_row_ids: List[str]
    values: List[str]


@dataclass(frozen=True)
class TableAudit:
    contract_file: str
    contract_id: str
    fn_id: str
    table_id: str
    predicate_col: str
    value_col: str
    variables: List[str]
    sample_points: int
    ok_points: int
    gap_points: int
    overlap_points: int
    overlap_conflicting_points: int
    example_gaps: List[SamplePoint]
    example_overlaps: List[SamplePoint]


def _extract_lookup_rule_calls(contract_ir: Mapping[str, Any]) -> List[LookupRuleCall]:
    out: List[LookupRuleCall] = []
    for fn in contract_ir.get("derived", []) or []:
        if not isinstance(fn, dict):
            continue
        fn_id = fn.get("fn_id")
        if not isinstance(fn_id, str):
            continue
        expr = fn.get("expr")
        for node in _walk_ast(expr):
            if node.get("op") != "lookup_rule":
                continue
            args = node.get("args")
            if not isinstance(args, list) or len(args) != 3:
                continue
            table_id = _is_string_lit(args[0])
            predicate_col = _is_string_lit(args[1])
            value_col = _is_string_lit(args[2])
            if not (isinstance(table_id, str) and isinstance(predicate_col, str) and isinstance(value_col, str)):
                # Dynamic lookup_rule is possible in schema but we skip for this audit.
                continue
            out.append(
                LookupRuleCall(
                    fn_id=fn_id,
                    table_id=table_id,
                    predicate_col=predicate_col,
                    value_col=value_col,
                )
            )
    return out


def _tables_by_id(contract_ir: Mapping[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for t in contract_ir.get("tables", []) or []:
        if isinstance(t, dict) and isinstance(t.get("table_id"), str):
            out[t["table_id"]] = t
    return out


def _arg_defs_by_fn(contract_ir: Mapping[str, Any]) -> Dict[str, List[dict]]:
    out: Dict[str, List[dict]] = {}
    for fn in contract_ir.get("derived", []) or []:
        if not isinstance(fn, dict):
            continue
        fn_id = fn.get("fn_id")
        if not isinstance(fn_id, str):
            continue
        args = fn.get("args") or []
        if isinstance(args, list):
            out[fn_id] = [a for a in args if isinstance(a, dict)]
    return out


def _extract_thresholds_from_predicate(node: Any) -> Tuple[Dict[str, List[Decimal]], Dict[str, List[str]]]:
    """Return (numeric_thresholds, date_thresholds) by var name."""

    num: Dict[str, List[Decimal]] = {}
    dates: Dict[str, List[str]] = {}

    for n in _walk_ast(node):
        op = n.get("op")
        if op not in ("lt", "lte", "gt", "gte", "eq"):
            continue
        args = n.get("args")
        if not isinstance(args, list) or len(args) != 2:
            continue
        a, b = args
        var = _is_var(a)
        if var is None:
            var = _is_var(b)
            if var is None:
                continue
            other = a
        else:
            other = b

        dv = _is_decimal_lit(other)
        if dv is not None:
            num.setdefault(var, []).append(dv)
            continue
        dt = _is_date_lit(other)
        if dt is not None:
            dates.setdefault(var, []).append(dt)
            continue

    # de-dupe + stable ordering
    num = {k: sorted(set(v)) for k, v in num.items()}
    dates = {k: sorted(set(v)) for k, v in dates.items()}
    return num, dates


def _extract_var_names(node: Any) -> List[str]:
    vars: set[str] = set()
    for n in _walk_ast(node):
        v = _is_var(n)
        if v:
            vars.add(v)
    return sorted(vars)


def _decimal_samples(thresholds: Sequence[Decimal]) -> List[Decimal]:
    if not thresholds:
        return [Decimal("0"), Decimal("1")]
    uniq = sorted(set(thresholds))

    # Pick a small epsilon relative to scale.
    # Use 1e-4 by default, but avoid epsilon larger than smallest gap/10.
    eps = Decimal("0.0001")
    if len(uniq) >= 2:
        gaps = [uniq[i + 1] - uniq[i] for i in range(len(uniq) - 1)]
        min_gap = min(gaps)
        if min_gap > 0:
            eps = min(eps, min_gap / Decimal("10"))
    if eps <= 0:
        eps = Decimal("0.0001")

    samples: set[Decimal] = set()
    for t in uniq:
        samples.add(t)
        samples.add(t - eps)
        samples.add(t + eps)

    # Midpoints between thresholds.
    for i in range(len(uniq) - 1):
        a = uniq[i]
        b = uniq[i + 1]
        mid = (a + b) / Decimal("2")
        samples.add(mid)

    # Extremes: slightly below min and above max.
    samples.add(uniq[0] - Decimal("1"))
    samples.add(uniq[-1] + Decimal("1"))
    return sorted(samples)


def _date_samples(thresholds: Sequence[str]) -> List[str]:
    if not thresholds:
        return ["2000-01-01"]
    uniq = sorted(set(thresholds))
    samples: set[str] = set()
    for d in uniq:
        samples.add(d)
        # day before / after to exercise lt/lte boundaries
        samples.add(_iso_date_add_days(d, -1))
        samples.add(_iso_date_add_days(d, 1))
    return sorted(samples)


def _cartesian_product_limited(options: Mapping[str, Sequence[Any]], max_points: int) -> List[Dict[str, Any]]:
    keys = list(options.keys())
    if not keys:
        return [{}]
    lists = [list(options[k]) for k in keys]
    total = 1
    for xs in lists:
        total *= max(1, len(xs))
    # If tiny, do full product.
    if total <= max_points:
        out: List[Dict[str, Any]] = []

        def _recurse(i: int, acc: Dict[str, Any]) -> None:
            if i == len(keys):
                out.append(dict(acc))
                return
            k = keys[i]
            for v in lists[i]:
                acc[k] = v
                _recurse(i + 1, acc)

        _recurse(0, {})
        return out

    # Otherwise: deterministic sampling by stepping through indices.
    # We stride through each dimension with a prime-ish step.
    out = []
    strides = [1 for _ in lists]
    for i in range(len(strides) - 2, -1, -1):
        strides[i] = strides[i + 1] * len(lists[i + 1])

    def _decode(idx: int) -> Dict[str, Any]:
        row: Dict[str, Any] = {}
        for k, xs, stride in zip(keys, lists, strides):
            pos = (idx // stride) % len(xs)
            row[k] = xs[pos]
        return row

    step = 7919  # prime
    idx = 0
    seen = set()
    while len(out) < max_points:
        row = _decode(idx)
        key = tuple((k, row[k]) for k in keys)
        if key not in seen:
            seen.add(key)
            out.append(row)
        idx = (idx + step) % total
        if len(seen) >= total:
            break
    return out


def audit_one_table(
    *,
    contract_file: Path,
    contract_ir: Mapping[str, Any],
    call: LookupRuleCall,
    max_points: int,
    keep_examples: int,
) -> Optional[TableAudit]:
    contract_id = str(contract_ir.get("contract_id") or "")
    tables = _tables_by_id(contract_ir)
    table = tables.get(call.table_id)
    if not isinstance(table, dict):
        return None

    # Locate predicate nodes and extract variable thresholds.
    rows = table.get("rows") or []
    if not isinstance(rows, list) or not rows:
        return None

    num_thresholds: Dict[str, List[Decimal]] = {}
    date_thresholds: Dict[str, List[str]] = {}
    variables: set[str] = set()
    for row in rows:
        if not isinstance(row, dict):
            continue
        cells = row.get("cells")
        if not isinstance(cells, dict):
            continue
        pred = cells.get(call.predicate_col)
        if not isinstance(pred, dict):
            continue
        variables.update(_extract_var_names(pred))
        n, d = _extract_thresholds_from_predicate(pred)
        for k, vs in n.items():
            num_thresholds.setdefault(k, []).extend(vs)
        for k, vs in d.items():
            date_thresholds.setdefault(k, []).extend(vs)

    # De-dupe
    num_thresholds = {k: sorted(set(vs)) for k, vs in num_thresholds.items()}
    date_thresholds = {k: sorted(set(vs)) for k, vs in date_thresholds.items()}

    # Use derived fn args to determine types; fallback to decimal for unknown.
    fn_args = _arg_defs_by_fn(contract_ir).get(call.fn_id, [])
    arg_types: Dict[str, str] = {}
    for a in fn_args:
        name = a.get("name")
        typ = a.get("type")
        if isinstance(name, str) and isinstance(typ, str):
            arg_types[name] = typ

    options: Dict[str, List[Any]] = {}
    for var in sorted(variables):
        typ = arg_types.get(var, "decimal")
        if typ == "date":
            options[var] = _date_samples(date_thresholds.get(var, []))
        elif typ in ("decimal", "rate", "bps", "money", "integer"):
            options[var] = [str(x) for x in _decimal_samples(num_thresholds.get(var, []))]
        else:
            # For non-numeric, we can't infer good values; use a single placeholder.
            options[var] = ["<UNSPECIFIED>"]

    sample_args = _cartesian_product_limited(options, max_points=max_points)

    ok = 0
    gaps = 0
    overlaps = 0
    overlap_conflicting = 0
    example_gaps: List[SamplePoint] = []
    example_overlaps: List[SamplePoint] = []

    for args in sample_args:
        matched: List[Tuple[str, str]] = []  # (row_id, value_str)
        for row in rows:
            if not isinstance(row, dict):
                continue
            cells = row.get("cells")
            if not isinstance(cells, dict):
                continue
            pred = cells.get(call.predicate_col)
            if not isinstance(pred, dict):
                continue
            # Evaluate predicate in the environment of the derived function args.
            try:
                tv = evaluate_expr(expr=pred, arg_defs=fn_args, args=args, indices={}, tables=tables)
            except Exception:
                # If we can't evaluate the predicate (missing vars, etc.), skip this row for this point.
                continue
            if tv.kind != "bool":
                continue
            if not tv.value:
                continue
            # Evaluate output cell value.
            val_node = cells.get(call.value_col)
            if not isinstance(val_node, dict):
                continue
            try:
                vtv = evaluate_expr(expr=val_node, arg_defs=fn_args, args=args, indices={}, tables=tables)
                value_str = f"{vtv.kind}:{vtv.value}"
            except Exception as exc:
                value_str = f"ERROR:{type(exc).__name__}"
            row_id = row.get("row_id")
            matched.append((str(row_id) if row_id is not None else "<null>", value_str))

        if len(matched) == 1:
            ok += 1
            continue
        if len(matched) == 0:
            gaps += 1
            if len(example_gaps) < keep_examples:
                example_gaps.append(
                    SamplePoint(
                        args=dict(args),
                        match_count=0,
                        matched_row_ids=[],
                        values=[],
                    )
                )
            continue

        overlaps += 1
        values = [v for _, v in matched]
        if len(set(values)) > 1:
            overlap_conflicting += 1
        if len(example_overlaps) < keep_examples:
            example_overlaps.append(
                SamplePoint(
                    args=dict(args),
                    match_count=len(matched),
                    matched_row_ids=[rid for rid, _ in matched],
                    values=values,
                )
            )

    return TableAudit(
        contract_file=str(contract_file),
        contract_id=contract_id,
        fn_id=call.fn_id,
        table_id=call.table_id,
        predicate_col=call.predicate_col,
        value_col=call.value_col,
        variables=sorted(variables),
        sample_points=len(sample_args),
        ok_points=ok,
        gap_points=gaps,
        overlap_points=overlaps,
        overlap_conflicting_points=overlap_conflicting,
        example_gaps=example_gaps,
        example_overlaps=example_overlaps,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-root", default="runs")
    parser.add_argument("--max-files", type=int, default=200)
    parser.add_argument("--max-points", type=int, default=4000)
    parser.add_argument("--keep-examples", type=int, default=3)
    parser.add_argument("--require-valid", action="store_true", help="skip artifacts that fail current validate_contract_ir()")
    parser.add_argument("--json-out", default=None)
    args = parser.parse_args()

    runs_root = Path(args.runs_root)
    if not runs_root.exists():
        raise SystemExit(f"runs root not found: {runs_root}")

    audits: List[TableAudit] = []
    scanned = 0
    for p in _iter_contractir_files(runs_root):
        if scanned >= args.max_files:
            break
        scanned += 1
        try:
            doc = _read_json(p)
        except Exception:
            continue
        if not isinstance(doc, dict):
            continue
        if args.require_valid:
            if validate_contract_ir(doc):
                continue
        raw = json.dumps(doc)
        if "\"op\": \"lookup_rule\"" not in raw:
            continue

        calls = _extract_lookup_rule_calls(doc)
        for call in calls:
            audit = audit_one_table(
                contract_file=p,
                contract_ir=doc,
                call=call,
                max_points=args.max_points,
                keep_examples=args.keep_examples,
            )
            if audit is not None:
                audits.append(audit)

    # Summary
    print(f"Scanned files: {scanned}")
    print(f"lookup_rule tables audited: {len(audits)}")

    total_points = sum(a.sample_points for a in audits)
    total_ok = sum(a.ok_points for a in audits)
    total_gaps = sum(a.gap_points for a in audits)
    total_overlaps = sum(a.overlap_points for a in audits)
    total_overlap_conflicts = sum(a.overlap_conflicting_points for a in audits)
    if total_points:
        print(
            "Aggregate sample results: "
            f"ok={total_ok}/{total_points} "
            f"gaps={total_gaps}/{total_points} "
            f"overlaps={total_overlaps}/{total_points} "
            f"overlap_conflicts={total_overlap_conflicts}/{total_points}"
        )

    # Print per-table highlights (gaps/overlaps only).
    for a in audits:
        if a.gap_points == 0 and a.overlap_points == 0:
            continue
        print("\n---")
        print(
            f"{a.contract_id} :: {a.fn_id} :: {a.table_id} "
            f"(vars={a.variables}, samples={a.sample_points})"
        )
        if a.gap_points:
            print(f"  gaps={a.gap_points}  example={a.example_gaps[0].args if a.example_gaps else None}")
        if a.overlap_points:
            print(
                f"  overlaps={a.overlap_points} conflicting={a.overlap_conflicting_points} "
                f"example={a.example_overlaps[0].args if a.example_overlaps else None}"
            )

    if args.json_out:
        out_path = Path(args.json_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps([asdict(a) for a in audits], indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(f"\nWrote JSON: {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
