#!/usr/bin/env python3
"""
Compare CovenantIR v0.1 outputs against the CEA baseline for paragraph runs.

Goal
----
Given:
  - a CEA paragraph run under runs/<run_id>/ (e.g., cea-1995q3, cea-1995q4)
  - a CovenantIR batch output directory (produced by scripts/covenant_ir_v0_1_batch_runner.py)

Produce:
  - a JSON report with per-item match diagnostics + aggregate counts.

This is "Option B" for the CEA rerun workflow:
  - We do NOT export CovenantIR into the legacy prompt_v1_short JSON-list format.
  - We compare baseline rows directly against CovenantIR's typed IR by extracting numeric thresholds from:
      - covenant test expressions (including derived fns)
      - referenced schedule tables (lookup_range / lookup_rule / lookup / lookup2)

Why this works for CEA:
  - The CEA baseline is a (name, limit[, start/end]) row per paragraph.
  - For A1 CovenantIR, covenant-defined metrics are external inputs; thresholds are literal numbers/schedule cells.
  - Matching is therefore "identity-ish tokens" + "does the baseline numeric limit appear as a threshold".
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from datetime import date
from decimal import Decimal, InvalidOperation
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8", errors="replace"))


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _norm_ws(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip())


_TOKEN_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)


def _tokenize(s: str) -> List[str]:
    return [m.group(0).lower() for m in _TOKEN_RE.finditer(s or "")]


_MISSING_STRINGS = {"", "<na>", "na", "nan", "none", "null", "n/a"}


def _is_missing_str(s: str) -> bool:
    return _norm_ws(s).lower() in _MISSING_STRINGS


def _parse_decimal_like(value: Any) -> Optional[Decimal]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        try:
            return Decimal(str(value)).quantize(Decimal("0.0001"))
        except Exception:
            return None
    if not isinstance(value, str):
        value = str(value)
    s = _norm_ws(value)
    if _is_missing_str(s):
        return None
    s = s.replace(",", "")
    s = re.sub(r"^\s*\$\s*", "", s).strip()
    # Handle negatives expressed as parentheses: "(200000)" -> -200000
    if s.startswith("(") and s.endswith(")"):
        inner = s[1:-1].strip()
        inner = re.sub(r"^\s*\$\s*", "", inner).strip()
        s = f"-{inner}"
    # Take leading number (handles "2.25 to 1.00" etc.)
    m = re.match(r"^(-?(?:\d+(?:\.\d+)?|\.\d+))", s)
    if not m:
        return None
    try:
        return Decimal(m.group(1)).quantize(Decimal("0.0001"))
    except (InvalidOperation, ValueError):
        return None


def _parse_timestamp_like_to_iso(s: str) -> Optional[str]:
    """Parse the CEA baseline date strings into ISO YYYY-MM-DD when possible.

    Example baseline format observed in runs/cea-1995q3/manifest.json:
      "1995-12-31 00:00:00"
    """

    s = _norm_ws(s)
    if _is_missing_str(s):
        return None
    # "YYYY-MM-DD HH:MM:SS" -> YYYY-MM-DD
    m = re.match(r"^(\d{4}-\d{2}-\d{2})\s+\d{2}:\d{2}:\d{2}$", s)
    if m:
        return m.group(1)
    # "M/D/YYYY" convenience fallback
    m2 = re.match(r"^(\d{1,2})/(\d{1,2})/(\d{2,4})$", s)
    if not m2:
        return None
    mm = int(m2.group(1))
    dd = int(m2.group(2))
    yy = int(m2.group(3))
    if yy < 100:
        yy = 2000 + yy
    try:
        return date(yy, mm, dd).isoformat()
    except ValueError:
        return None


def _limit_matches(*, baseline_limit: Decimal, extracted_limit: Decimal) -> bool:
    if extracted_limit == baseline_limit:
        return True
    # percent<->decimal normalization
    if baseline_limit > 1 and baseline_limit <= 100:
        as_decimal = (baseline_limit / Decimal("100")).quantize(Decimal("0.0001"))
        if extracted_limit == as_decimal:
            return True
    if baseline_limit >= 0 and baseline_limit < 1:
        as_percent = (baseline_limit * Decimal("100")).quantize(Decimal("0.0001"))
        if extracted_limit == as_percent:
            return True
    return False


def _preview(text: str, limit: int = 260) -> str:
    t = _norm_ws(text)
    if len(t) <= limit:
        return t
    return t[: limit - 3] + "..."


# For evaluation, we need to distinguish "same covenant type" vs "same numeric limit".
# Baseline names are short; remove generic tokens so overlap has signal.
_GENERIC_IDENTITY_TOKENS = {
    "ratio",
    "minimum",
    "maximum",
    "min",
    "max",
    "consolidated",
    "total",
    "the",
    "a",
    "an",
    "and",
    "or",
}


def _identity_tokens_from_baseline(baseline_name: str) -> List[str]:
    toks = [t for t in _tokenize(baseline_name) if len(t) >= 4]
    if not toks:
        toks = [t for t in _tokenize(baseline_name) if t]
    identity = [t for t in toks if t not in _GENERIC_IDENTITY_TOKENS]
    return identity or toks


def _identity_overlap_tokens(*, baseline_tokens: Sequence[str], candidate_tokens: Sequence[str]) -> List[str]:
    return sorted(set(baseline_tokens) & set(candidate_tokens))


def _jaccard(a: Sequence[str], b: Sequence[str]) -> float:
    sa = set(a)
    sb = set(b)
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def _seq_ratio(a: str, b: str) -> float:
    a = _norm_ws(a).lower()
    b = _norm_ws(b).lower()
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b).ratio()


def _is_ast_node(x: Any) -> bool:
    if not isinstance(x, dict):
        return False
    # CovenantIR AST nodes use a minimal tagged-union shape:
    #   - literal: {"lit": {"type": "...", "value": ...}}
    #   - variable: {"var": "name"}
    #   - operator: {"op": "lte", "args": [<AST>, ...]}
    #
    # This mirrors the ContractIR/CovenantIR evaluator shape used across the repo.
    if "lit" in x and set(x.keys()) == {"lit"}:
        return isinstance(x.get("lit"), dict)
    if "var" in x and set(x.keys()) == {"var"}:
        return isinstance(x.get("var"), str)
    if "op" in x and "args" in x and set(x.keys()) == {"op", "args"}:
        return isinstance(x.get("op"), str) and isinstance(x.get("args"), list)
    return False


def _iter_ast_nodes(x: Any) -> Iterable[Dict[str, Any]]:
    if isinstance(x, dict):
        # Yield AST nodes only (not every dict), but recurse through dict values
        # so we can find nested AST expressions under "expr", "cells", etc.
        if _is_ast_node(x):
            yield x
            if "args" in x and isinstance(x.get("args"), list):
                for a in x["args"]:
                    yield from _iter_ast_nodes(a)
            return
        for v in x.values():
            yield from _iter_ast_nodes(v)
    elif isinstance(x, list):
        for v in x:
            yield from _iter_ast_nodes(v)


def _lit_value(n: Any) -> Tuple[Optional[str], Any]:
    if not isinstance(n, dict) or "lit" not in n:
        return (None, None)
    lit = n.get("lit")
    if not isinstance(lit, dict):
        return (None, None)
    return (lit.get("type"), lit.get("value"))


def _var_name(n: Any) -> Optional[str]:
    if isinstance(n, dict) and "var" in n:
        v = n.get("var")
        if isinstance(v, str) and v.strip():
            return v.strip()
    return None


@dataclass(frozen=True)
class TableRef:
    table_id: str
    value_col: str


def _extract_table_refs(expr: Any) -> List[TableRef]:
    """Extract (table_id, value_col) from lookup ops in an AST expression."""

    refs: List[TableRef] = []
    for n in _iter_ast_nodes(expr):
        op = n.get("op")
        args = n.get("args")
        if not isinstance(op, str) or not isinstance(args, list):
            continue
        if op == "lookup_range" and len(args) == 7:
            t0, table_id = _lit_value(args[0])
            t6, value_col = _lit_value(args[6])
            if t0 == "string" and isinstance(table_id, str) and t6 == "string" and isinstance(value_col, str):
                refs.append(TableRef(table_id=table_id, value_col=value_col))
        if op == "lookup_rule" and len(args) == 3:
            t0, table_id = _lit_value(args[0])
            t2, value_col = _lit_value(args[2])
            if t0 == "string" and isinstance(table_id, str) and t2 == "string" and isinstance(value_col, str):
                refs.append(TableRef(table_id=table_id, value_col=value_col))
        if op == "lookup" and len(args) == 4:
            t0, table_id = _lit_value(args[0])
            t3, value_col = _lit_value(args[3])
            if t0 == "string" and isinstance(table_id, str) and t3 == "string" and isinstance(value_col, str):
                refs.append(TableRef(table_id=table_id, value_col=value_col))
        if op == "lookup2" and len(args) == 6:
            t0, table_id = _lit_value(args[0])
            t5, value_col = _lit_value(args[5])
            if t0 == "string" and isinstance(table_id, str) and t5 == "string" and isinstance(value_col, str):
                refs.append(TableRef(table_id=table_id, value_col=value_col))

    seen: set[tuple[str, str]] = set()
    out: List[TableRef] = []
    for r in refs:
        key = (r.table_id, r.value_col)
        if key in seen:
            continue
        seen.add(key)
        out.append(r)
    return out


def _resolve_expr_from_bool_expr_spec(spec: Any, *, derived_map: Mapping[str, Any]) -> Optional[Any]:
    """Return the underlying AST expr for a bool_expr_spec (inline or derived fn ref)."""

    if not isinstance(spec, dict):
        return None
    if "fn_id" in spec:
        fn_id = spec.get("fn_id")
        if not isinstance(fn_id, str) or not fn_id:
            return None
        df = derived_map.get(fn_id)
        if isinstance(df, dict):
            return df.get("expr")
        return None
    # Inline spec shape: {"args": [...], "returns":"bool", "expr":<AST>, "source_refs":[...]}
    return spec.get("expr")


def _collect_thresholds_from_expr(expr: Any, *, table_map: Mapping[str, Any]) -> Tuple[set[Decimal], List[Dict[str, Any]], set[str]]:
    thresholds: set[Decimal] = set()
    sources: List[Dict[str, Any]] = []
    date_literals: set[str] = set()

    for n in _iter_ast_nodes(expr):
        ltyp, lval = _lit_value(n)
        if ltyp in ("decimal", "rate", "bps", "money", "integer", "int") and isinstance(lval, str):
            d = _parse_decimal_like(lval)
            if d is not None:
                thresholds.add(d)
                sources.append({"kind": "literal_anywhere", "type": ltyp, "value": str(d)})
        if ltyp == "date" and isinstance(lval, str):
            date_literals.add(lval)

    # Comparator thresholds: catch common "metric_var <= 3.50" patterns.
    for n in _iter_ast_nodes(expr):
        op = n.get("op")
        args = n.get("args")
        if not isinstance(op, str) or not isinstance(args, list) or len(args) != 2:
            continue
        if op not in ("lt", "lte", "gt", "gte", "eq"):
            continue
        left_var = _var_name(args[0])
        right_var = _var_name(args[1])
        ltyp, lval = _lit_value(args[0])
        rtyp, rval = _lit_value(args[1])

        if ltyp == "date" and isinstance(lval, str):
            date_literals.add(lval)
        if rtyp == "date" and isinstance(rval, str):
            date_literals.add(rval)

        if left_var and rtyp in ("decimal", "rate", "bps", "money", "integer", "int") and isinstance(rval, str):
            d = _parse_decimal_like(rval)
            if d is not None:
                thresholds.add(d)
                sources.append({"kind": "direct_cmp", "op": op, "var": left_var, "value": str(d)})
        if right_var and ltyp in ("decimal", "rate", "bps", "money", "integer", "int") and isinstance(lval, str):
            d = _parse_decimal_like(lval)
            if d is not None:
                thresholds.add(d)
                sources.append({"kind": "direct_cmp", "op": op, "var": right_var, "value": str(d)})

    # Schedule/table thresholds: follow lookup refs.
    for ref in _extract_table_refs(expr):
        table = table_map.get(ref.table_id)
        if not isinstance(table, dict):
            continue
        rows = table.get("rows")
        if not isinstance(rows, list):
            continue
        for row in rows:
            if not isinstance(row, dict):
                continue
            cells = row.get("cells")
            if not isinstance(cells, dict):
                continue
            cell = cells.get(ref.value_col)
            # Value cells are often plain literals, but for 2D tables the model may encode them
            # as an expression (e.g., if(is_adjusted_oak, <adj>, <cons>)). For CEA matching we
            # treat ANY numeric literal reachable in the value-cell AST as a candidate threshold.
            for n in _iter_ast_nodes(cell):
                ntyp, nval = _lit_value(n)
                if ntyp in ("decimal", "rate", "bps", "money", "integer", "int") and isinstance(nval, str):
                    d = _parse_decimal_like(nval)
                    if d is not None:
                        thresholds.add(d)
                        sources.append(
                            {
                                "kind": "table_value_ast",
                                "table_id": ref.table_id,
                                "value_col": ref.value_col,
                                "value": str(d),
                            }
                        )
                if ntyp == "date" and isinstance(nval, str):
                    date_literals.add(nval)

            # Also collect date literals from other schedule cells (bounds, keys, etc.).
            for v in cells.values():
                for n in _iter_ast_nodes(v):
                    vtyp, vval = _lit_value(n)
                    if vtyp == "date" and isinstance(vval, str):
                        date_literals.add(vval)

    return (thresholds, sources, date_literals)


def _limit_in_text(*, limit: Decimal, text: str, required_keywords: Sequence[str]) -> Tuple[bool, Optional[str]]:
    """Heuristic evidence check: is the baseline limit actually mentioned in the excerpt text?"""

    if not text:
        return (False, None)
    tlow = _norm_ws(text).lower()
    if required_keywords and not any(k.lower() in tlow for k in required_keywords[:3]):
        return (False, None)

    expected: List[Decimal] = [limit]
    if limit > 1 and limit <= 100:
        expected.append((limit / Decimal("100")).quantize(Decimal("0.0001")))
    if limit >= 0 and limit < 1:
        expected.append((limit * Decimal("100")).quantize(Decimal("0.0001")))
    expected_set = set(expected)

    num_re = re.compile(
        r"""
        (?P<tok>
            \(?\s*
            \$?\s*
            -?\s*
            (?:
                \d{1,3}(?:,\d{3})+(?:\.\d+)?  # 1,234.56
                |
                \d+(?:\.\d+)?                 # 1234.56
                |
                \.\d+                         # .65
            )
            \s*\)?
            %?
        )
        """,
        re.VERBOSE,
    )
    for m in num_re.finditer(text):
        d = _parse_decimal_like(m.group("tok"))
        if d is None:
            continue
        if d in expected_set:
            return (True, "parsed")
        if abs(d) in expected_set:
            return (True, "parsed_abs")
    return (False, None)


@dataclass(frozen=True)
class CovenantSummary:
    covenant_id: str
    title: str
    overlap_tokens: List[str]
    name_jaccard: float
    seq_ratio: float
    combined_score: float
    limit_match: bool
    threshold_values: List[str]
    threshold_sources: List[Dict[str, Any]]
    schedule_dates: List[str]


def _summarize_item(
    *,
    item: Mapping[str, Any],
    run_dir: Path,
    covenantir_dir: Path,
) -> Dict[str, Any]:
    item_id = str(item.get("item_id") or "")
    baseline = item.get("baseline") if isinstance(item.get("baseline"), dict) else {}

    baseline_name = _norm_ws(str(baseline.get("name") or ""))
    baseline_limit_raw = _norm_ws(str(baseline.get("limit") or ""))
    baseline_limit = _parse_decimal_like(baseline_limit_raw)
    baseline_start_iso = _parse_timestamp_like_to_iso(str(baseline.get("start_date") or ""))
    baseline_end_iso = _parse_timestamp_like_to_iso(str(baseline.get("end_date") or ""))

    baseline_tokens = _identity_tokens_from_baseline(baseline_name) if baseline_name else []

    canonical_path = run_dir / "normalized" / item_id / "canonical.txt"
    excerpt_text = canonical_path.read_text(encoding="utf-8", errors="replace") if canonical_path.exists() else ""

    baseline_name_in_excerpt = False
    if baseline_tokens and excerpt_text:
        elow = excerpt_text.lower()
        baseline_name_in_excerpt = any(t.lower() in elow for t in baseline_tokens[:3])

    baseline_limit_in_excerpt = False
    baseline_limit_in_excerpt_kind: Optional[str] = None
    if baseline_limit is not None and excerpt_text:
        baseline_limit_in_excerpt, baseline_limit_in_excerpt_kind = _limit_in_text(
            limit=baseline_limit, text=excerpt_text, required_keywords=baseline_tokens
        )

    item_out_dir = covenantir_dir / item_id
    result_path = item_out_dir / "result.json"
    validated_path = item_out_dir / "covenantir_validated.json"

    out: Dict[str, Any] = {
        "item_id": item_id,
        "baseline": {
            "name": baseline_name or None,
            "limit_raw": baseline_limit_raw or None,
            "limit_decimal": str(baseline_limit) if baseline_limit is not None else None,
            "start_date_iso": baseline_start_iso,
            "end_date_iso": baseline_end_iso,
        },
        "excerpt": {
            "canonical_path": str(canonical_path),
            "excerpt_len": len(excerpt_text),
            "excerpt_preview": _preview(excerpt_text, limit=260) if excerpt_text else "",
            "baseline_name_in_excerpt": baseline_name_in_excerpt,
            "baseline_limit_in_excerpt": baseline_limit_in_excerpt,
            "baseline_limit_in_excerpt_kind": baseline_limit_in_excerpt_kind,
        },
        "covenantir": {
            "status": "missing_output",
            "result_path": str(result_path),
            "validated_path": str(validated_path),
            "covenants_count": None,
            "open_items_total": None,
            "open_items_blocking": None,
        },
        "match": {
            "baseline_identity_tokens": baseline_tokens,
            "baseline_identity_ok_any": False,
            "baseline_identity_overlap_best": [],
            "baseline_limit_found_any": False,
            "baseline_limit_found_best": False,
        },
        "candidates": [],
        "best_match": None,
        "disposition": None,
    }

    if not result_path.exists():
        out["disposition"] = "missing_output"
        return out

    try:
        result_obj = _read_json(result_path)
    except Exception as exc:
        out["covenantir"]["status"] = "invalid_result_json"
        out["disposition"] = f"error_result_json:{type(exc).__name__}"
        return out

    status = str(result_obj.get("status") or "unknown")
    out["covenantir"]["status"] = status
    out["covenantir"]["covenants_count"] = result_obj.get("covenants_count")
    out["covenantir"]["open_items_total"] = result_obj.get("open_items_total")
    out["covenantir"]["open_items_blocking"] = result_obj.get("open_items_blocking")

    # If the CovenantIR run did not produce a validated IR, we can't do threshold matching.
    if status != "ok" or not validated_path.exists():
        out["disposition"] = f"covenantir_{status}"
        return out

    try:
        doc = _read_json(validated_path)
    except Exception as exc:
        out["covenantir"]["status"] = "invalid_validated_json"
        out["disposition"] = f"error_validated_json:{type(exc).__name__}"
        return out

    covenants = doc.get("covenants")
    tables = doc.get("tables")
    derived = doc.get("derived")
    if not isinstance(covenants, list):
        covenants = []
    if not isinstance(tables, list):
        tables = []
    if not isinstance(derived, list):
        derived = []

    table_map: Dict[str, Any] = {}
    for t in tables:
        if not isinstance(t, dict):
            continue
        tid = t.get("table_id")
        if isinstance(tid, str) and tid:
            table_map[tid] = t

    derived_map: Dict[str, Any] = {}
    for d in derived:
        if not isinstance(d, dict):
            continue
        fn_id = d.get("fn_id")
        if isinstance(fn_id, str) and fn_id:
            derived_map[fn_id] = d

    summaries: List[CovenantSummary] = []
    all_thresholds: set[Decimal] = set()

    for cov in covenants:
        if not isinstance(cov, dict):
            continue
        cid = _norm_ws(str(cov.get("covenant_id") or ""))
        title = _norm_ws(str(cov.get("title") or ""))
        test_spec = cov.get("test")
        expr = _resolve_expr_from_bool_expr_spec(test_spec, derived_map=derived_map)
        if expr is None:
            continue

        thresholds, threshold_sources, dates = _collect_thresholds_from_expr(expr, table_map=table_map)
        all_thresholds |= thresholds

        candidate_tokens = _tokenize(" ".join([cid, title]))
        overlap = _identity_overlap_tokens(baseline_tokens=baseline_tokens, candidate_tokens=candidate_tokens)
        name_j = _jaccard(baseline_tokens, candidate_tokens)
        seq = _seq_ratio(baseline_name, title)
        combined = 0.7 * name_j + 0.3 * seq

        limit_match = False
        if baseline_limit is not None:
            limit_match = any(_limit_matches(baseline_limit=baseline_limit, extracted_limit=t) for t in thresholds)

        summaries.append(
            CovenantSummary(
                covenant_id=cid,
                title=title,
                overlap_tokens=overlap,
                name_jaccard=round(name_j, 6),
                seq_ratio=round(seq, 6),
                combined_score=round(combined, 6),
                limit_match=bool(limit_match),
                threshold_values=[str(d) for d in sorted(thresholds)],
                threshold_sources=threshold_sources,
                schedule_dates=sorted(dates),
            )
        )

    # Baseline value found anywhere in the output.
    if baseline_limit is not None:
        out["match"]["baseline_limit_found_any"] = any(
            _limit_matches(baseline_limit=baseline_limit, extracted_limit=t) for t in all_thresholds
        )

    # Sort candidates: prioritize limit match + identity overlap + combined score.
    summaries_sorted = sorted(
        summaries,
        key=lambda s: (
            int(s.limit_match),
            len(s.overlap_tokens),
            s.combined_score,
        ),
        reverse=True,
    )

    out["candidates"] = [
        {
            "covenant_id": s.covenant_id,
            "title": s.title,
            "identity_overlap_tokens": s.overlap_tokens,
            "name_jaccard": float(f"{s.name_jaccard:.6f}"),
            "seq_ratio": float(f"{s.seq_ratio:.6f}"),
            "combined_score": float(f"{s.combined_score:.6f}"),
            "limit_match": bool(s.limit_match),
            "threshold_values": s.threshold_values,
            "schedule_dates": s.schedule_dates,
        }
        for s in summaries_sorted[:10]
    ]

    if summaries_sorted:
        best = summaries_sorted[0]
        out["best_match"] = {
            "covenant_id": best.covenant_id,
            "title": best.title,
            "combined_score": float(f"{best.combined_score:.6f}"),
            "identity_overlap_tokens": best.overlap_tokens,
            "limit_match": bool(best.limit_match),
            "threshold_values": best.threshold_values,
            "schedule_dates": best.schedule_dates,
            "threshold_sources": best.threshold_sources[:20],
        }
        out["match"]["baseline_identity_overlap_best"] = best.overlap_tokens
        out["match"]["baseline_identity_ok_any"] = any(bool(s.overlap_tokens) for s in summaries)
        out["match"]["baseline_limit_found_best"] = bool(best.limit_match)

    # Disposition buckets:
    if baseline_limit is None or _is_missing_str(baseline_name):
        out["disposition"] = "no_baseline_label_or_value"
        return out

    ok_matched = any(s.limit_match and bool(s.overlap_tokens) for s in summaries)
    if ok_matched:
        out["disposition"] = "ok_baseline_matched"
    elif out["match"]["baseline_limit_found_any"] is True:
        out["disposition"] = "ok_baseline_limit_matched_wrong_covenant"
    else:
        if out.get("excerpt", {}).get("baseline_limit_in_excerpt") is True:
            out["disposition"] = "ok_baseline_limit_missing_despite_in_excerpt"
        else:
            out["disposition"] = "ok_baseline_limit_not_in_excerpt"

    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True, help="CEA run_id under runs/<run_id>/")
    ap.add_argument("--covenantir-dir", required=True, help="Directory containing per-item CovenantIR outputs (subdirs by item_id).")
    ap.add_argument("--base-dir", default=".", help="Repo base dir")
    ap.add_argument("--out", required=True, help="Path to write JSON report")
    ap.add_argument("--max-items", type=int, default=10_000)
    ap.add_argument(
        "--item-id",
        dest="item_ids",
        action="append",
        default=[],
        help="Optional explicit item_id to score (repeatable). If omitted, scores all items in manifest order.",
    )
    args = ap.parse_args()

    base_dir = Path(args.base_dir)
    run_dir = base_dir / "runs" / args.run_id
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.exists():
        raise SystemExit(f"Missing manifest.json: {manifest_path}")

    manifest = _read_json(manifest_path)
    items = manifest.get("items")
    if not isinstance(items, list) or not items:
        raise SystemExit(f"manifest.json has no items: {manifest_path}")

    covenantir_dir = Path(args.covenantir_dir)
    if not covenantir_dir.exists():
        raise SystemExit(f"covenantir-dir not found: {covenantir_dir}")

    if args.item_ids:
        wanted = {str(i).strip() for i in args.item_ids if str(i).strip()}
        items = [it for it in items if isinstance(it, dict) and str(it.get("item_id") or "") in wanted]

    per_item: List[Dict[str, Any]] = []
    for it in items[: max(0, int(args.max_items))]:
        if not isinstance(it, dict):
            continue
        per_item.append(_summarize_item(item=it, run_dir=run_dir, covenantir_dir=covenantir_dir))

    # Aggregate counts.
    by_disposition: Dict[str, int] = {}
    by_covenantir_status: Dict[str, int] = {}
    for r in per_item:
        disp = str(r.get("disposition") or "unknown")
        by_disposition[disp] = by_disposition.get(disp, 0) + 1
        cov_status = str(((r.get("covenantir") or {}) if isinstance(r.get("covenantir"), dict) else {}).get("status") or "unknown")
        by_covenantir_status[cov_status] = by_covenantir_status.get(cov_status, 0) + 1

    report = {
        "schema_version": "covenantir_v0_1_vs_cea_baseline_report_v1",
        "run_id": args.run_id,
        "covenantir_dir": str(covenantir_dir),
        "counts": {
            "items_total": len(per_item),
            "ok_baseline_matched": by_disposition.get("ok_baseline_matched", 0),
            "by_disposition": dict(sorted(by_disposition.items(), key=lambda kv: (-kv[1], kv[0]))),
            "by_covenantir_status": dict(sorted(by_covenantir_status.items(), key=lambda kv: (-kv[1], kv[0]))),
        },
        "items": per_item,
    }
    _write_json(Path(args.out), report)

    print(f"Wrote {args.out}")
    print(f"Counts: {report['counts']['by_disposition']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
