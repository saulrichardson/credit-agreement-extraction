#!/usr/bin/env python
"""
Compare CovenantIR outputs (our current extractor) against a local covenant-sample baseline.

Baseline format
---------------
This script expects the run to have been built by:
  scripts/build_local_covenant_sample_run.py

That builder writes:
  runs/<run_id>/local_sample_manifest.json

which contains, per item_id:
  - baseline: name/limit/start/end
  - provenance pointers (chunk_text length, local_agreement_path)

Inputs
------
- CovenantIR outputs directory from the batch runner, with per-item subfolders that contain:
    covenantir_validated.json
    result.json

Output
------
- A JSON report with per-item match diagnostics + aggregate counts.
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


def _iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if isinstance(obj, dict):
                yield obj


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _norm_ws(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip())


_MISSING_STRINGS = {"", "<na>", "na", "nan", "none", "null"}


def _is_missing_str(s: str) -> bool:
    return _norm_ws(s).lower() in _MISSING_STRINGS


def _parse_us_date_to_iso(s: str) -> Optional[str]:
    s = _norm_ws(s)
    if _is_missing_str(s):
        return None
    m = re.match(r"^(\d{1,2})/(\d{1,2})/(\d{2,4})$", s)
    if not m:
        return None
    mm = int(m.group(1))
    dd = int(m.group(2))
    yy = int(m.group(3))
    if yy < 100:
        # Conservative pivot: sample data is 2000s/2010s.
        yy = 2000 + yy
    try:
        d = date(yy, mm, dd)
    except ValueError:
        return None
    return d.isoformat()


def _parse_decimal_like(s: str) -> Optional[Decimal]:
    s = _norm_ws(s)
    if _is_missing_str(s):
        return None

    # Remove common formatting.
    cleaned = s.replace(",", "").replace("$", "").replace("%", "")

    # If the value is ratio-like "1.50:1.00", take the first component.
    if ":" in cleaned:
        cleaned = cleaned.split(":", 1)[0]

    # If the string contains extra tokens, take the first numeric-looking token.
    m = re.match(r"^-?\d+(?:\.\d+)?", cleaned)
    if m:
        cleaned = m.group(0)

    if not cleaned:
        return None
    try:
        return Decimal(cleaned)
    except InvalidOperation:
        return None


def _limit_matches_threshold_values(*, baseline_limit: Decimal, threshold_values: Sequence[str]) -> bool:
    """Return True if baseline_limit matches any threshold value (with simple percent/decimal normalization)."""

    thresholds = [_parse_decimal_like(v) for v in threshold_values]
    thresholds = [d for d in thresholds if d is not None]
    if any(d == baseline_limit for d in thresholds):
        return True

    # baseline=75 matches 0.75 in output; baseline=0.15 matches "15%"-style output.
    if baseline_limit > 1 and baseline_limit <= 100:
        as_decimal = (baseline_limit / Decimal("100")).quantize(Decimal("0.0001"))
        if any(d == as_decimal for d in thresholds):
            return True
    if baseline_limit >= 0 and baseline_limit < 1:
        as_percent = (baseline_limit * Decimal("100")).quantize(Decimal("0.0001"))
        if any(d == as_percent for d in thresholds):
            return True
    return False


def _tokenize(s: str) -> List[str]:
    s = (s or "").lower()
    s = re.sub(r"[^a-z0-9]+", " ", s)
    toks = [t for t in s.split() if t]
    return toks


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


# For evaluation, we need to distinguish "same covenant type" vs "same numeric limit".
# The baseline names are short (e.g., "interest coverage ratio"), and many covenants share
# generic tokens like "ratio"/"maximum"/"minimum". We treat those as non-identifying.
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


def _identity_overlap_tokens(*, baseline_tokens: Sequence[str], candidate_tokens: Sequence[str]) -> Tuple[List[str], List[str]]:
    """Return (identity_tokens, overlap_tokens) for baseline-vs-candidate covenant identity.

    - identity_tokens: baseline tokens with generic words removed
    - overlap_tokens: intersection(identity_tokens, candidate_tokens)

    If the baseline has no identity tokens after filtering (rare), we fall back to using the
    full baseline tokens so we never treat "empty identity" as a mismatch.
    """

    identity_tokens = [t for t in baseline_tokens if t and t not in _GENERIC_IDENTITY_TOKENS]
    if not identity_tokens:
        identity_tokens = [t for t in baseline_tokens if t]
    overlap = sorted(set(identity_tokens) & set(candidate_tokens))
    return (sorted(set(identity_tokens)), overlap)


_NUM_TOKEN_RE = re.compile(r"(?<![A-Za-z0-9])(\d{1,3}(?:,\d{3})*(?:\.\d+)?|\d+(?:\.\d+)?|\.\d+)(%?)(?![A-Za-z0-9])")


def _extract_numeric_mentions(text: str) -> List[Dict[str, Any]]:
    """Extract numeric tokens from raw excerpt text for baseline-evidence checks.

    This is intentionally heuristic and used ONLY for evaluation diagnostics
    (e.g., "baseline limit is not in the excerpt we fed to the model").
    """

    out: List[Dict[str, Any]] = []
    for m in _NUM_TOKEN_RE.finditer(text or ""):
        raw = m.group(1)
        pct = m.group(2) == "%"
        d = _parse_decimal_like(raw)
        if d is None:
            continue
        out.append({"raw": raw, "value": str(d), "is_percent": pct, "start": m.start(), "end": m.end()})
    return out


def _limit_in_text(
    *,
    limit: Decimal,
    text: str,
    required_keywords: Sequence[str] | None = None,
    window_chars: int = 120,
) -> Tuple[bool, Optional[str]]:
    """Return (present, kind) where kind matches the report match-kind labels.

    If required_keywords is provided, require that at least one of those keywords occurs
    within a small window around the numeric mention. This reduces false positives like
    matching a covenant ratio limit against an unrelated fee percentage in the excerpt.
    """

    mentions = _extract_numeric_mentions(text)
    vals: List[Tuple[Decimal, bool]] = []
    for m in mentions:
        try:
            v = Decimal(str(m.get("value") or ""))
        except Exception:
            continue
        vals.append((v, bool(m.get("is_percent"))))

    kw_norm: List[str] = []
    if required_keywords:
        for k in required_keywords:
            kk = _norm_ws(str(k)).lower()
            if kk:
                kw_norm.append(kk)

    def _kw_ok(start: int, end: int) -> bool:
        if not kw_norm:
            return True
        lo = max(0, start - int(window_chars))
        hi = min(len(text), end + int(window_chars))
        win = (text[lo:hi] or "").lower()
        return any(k in win for k in kw_norm)

    # Exact match.
    for m in mentions:
        try:
            v = Decimal(str(m.get("value") or ""))
        except Exception:
            continue
        if v == limit and _kw_ok(int(m.get("start") or 0), int(m.get("end") or 0)):
            return (True, "exact")

    # baseline=0.15 matches "15%" in the excerpt.
    if limit >= 0 and limit < 1:
        want_pct = (limit * Decimal("100")).quantize(Decimal("0.0001"))
        for m in mentions:
            try:
                v = Decimal(str(m.get("value") or ""))
            except Exception:
                continue
            if bool(m.get("is_percent")) and v == want_pct and _kw_ok(int(m.get("start") or 0), int(m.get("end") or 0)):
                return (True, "decimal_to_percent")

    # baseline=75 matches 0.75 in the excerpt.
    if limit > 1 and limit <= 100:
        want_decimal = (limit / Decimal("100")).quantize(Decimal("0.0001"))
        for m in mentions:
            try:
                v = Decimal(str(m.get("value") or ""))
            except Exception:
                continue
            if v == want_decimal and _kw_ok(int(m.get("start") or 0), int(m.get("end") or 0)):
                return (True, "percent_to_decimal")

    return (False, None)


def _preview(text: str, *, limit: int = 260) -> str:
    t = _norm_ws(text)
    if len(t) <= limit:
        return t
    return t[: max(0, limit - 1)] + "…"


_OUT_OF_SCOPE_ISSUE_KEYWORDS = (
    "negative covenant",
    "negative covenants",
    "indebtedness",
    "liens",
    "restricted payments",
    "restricted payment",
    "investments",
    "investment",
    "asset sales",
    "asset sale",
    "fundamental changes",
    "fundamental change",
)


_IN_SCOPE_OVERRIDE_KEYWORDS = (
    "ratio",
    "coverage",
    "net worth",
    "tangible",
    "ebitda",
    "capital expenditures",
    "capex",
    "leverage",
    "liquidity",
    "current ratio",
    "fixed charge",
    "interest coverage",
)


def _looks_out_of_scope_issue(issue: str) -> bool:
    t = _norm_ws(issue).lower()
    if not t:
        return False
    if any(k in t for k in _OUT_OF_SCOPE_ISSUE_KEYWORDS) and not any(k in t for k in _IN_SCOPE_OVERRIDE_KEYWORDS):
        return True
    return False


def _lit_value(node: Any) -> Tuple[Optional[str], Optional[str]]:
    """Return (lit_type, lit_value_as_str) for an AST literal node."""
    if not isinstance(node, dict):
        return (None, None)
    lit = node.get("lit")
    if not isinstance(lit, dict):
        return (None, None)
    typ = lit.get("type")
    val = lit.get("value")
    if not isinstance(typ, str):
        return (None, None)
    if typ == "date" and isinstance(val, str):
        return (typ, val)
    if typ in ("decimal", "rate", "bps", "money", "integer") and isinstance(val, str):
        return (typ, val)
    if typ == "string" and isinstance(val, str):
        return (typ, val)
    if typ == "bool" and isinstance(val, bool):
        return (typ, "true" if val else "false")
    return (None, None)


def _var_name(node: Any) -> Optional[str]:
    if not isinstance(node, dict):
        return None
    v = node.get("var")
    return v if isinstance(v, str) and v.strip() else None


def _iter_ast_nodes(node: Any) -> Iterable[Dict[str, Any]]:
    if isinstance(node, dict):
        yield node
        args = node.get("args")
        if isinstance(args, list):
            for a in args:
                yield from _iter_ast_nodes(a)
    elif isinstance(node, list):
        for a in node:
            yield from _iter_ast_nodes(a)


@dataclass(frozen=True)
class TableRef:
    table_id: str
    value_col: str


_METRIC_VAR_KEYWORDS = (
    "ratio",
    "coverage",
    "leverage",
    "debt",
    "net_worth",
    "networth",
    "worth",
    "income",
    "ebitda",
    "capex",
    "capital_expenditures",
    "liquidity",
    "fixed_charge",
    "interest_coverage",
    "tangible",
)


def _looks_like_metric_var(name: str) -> bool:
    n = name.lower()
    return any(k in n for k in _METRIC_VAR_KEYWORDS)


def _extract_table_refs(expr: Any) -> List[TableRef]:
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
    # Deduplicate preserving order
    seen: set[tuple[str, str]] = set()
    out: List[TableRef] = []
    for r in refs:
        key = (r.table_id, r.value_col)
        if key in seen:
            continue
        seen.add(key)
        out.append(r)
    return out


@dataclass
class CovenantSummary:
    covenant_id: str
    title: str
    name_score: float
    seq_score: float
    var_score: float
    combined_score: float
    threshold_values: List[str]  # Decimal strings
    threshold_sources: List[Dict[str, Any]]
    schedule_dates: List[str]  # ISO dates (strings)


def _collect_thresholds_from_expr(expr: Any, *, table_map: Mapping[str, Any]) -> Tuple[set[Decimal], List[Dict[str, Any]], set[str]]:
    thresholds: set[Decimal] = set()
    sources: List[Dict[str, Any]] = []
    date_literals: set[str] = set()

    # Any numeric literal values that appear in the expression tree (including arithmetic factors like 0.75).
    # This is intentionally broad for auditability; downstream matching can apply heuristics (e.g., percent scaling).
    for n in _iter_ast_nodes(expr):
        ltyp, lval = _lit_value(n)
        if ltyp in ("decimal", "rate", "bps", "money", "integer") and isinstance(lval, str):
            d = _parse_decimal_like(lval)
            if d is not None:
                thresholds.add(d)
                sources.append({"kind": "literal_anywhere", "type": ltyp, "value": str(d)})
        if ltyp == "date" and isinstance(lval, str):
            date_literals.add(lval)

    # Direct comparator thresholds.
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

        # Track date literals for diagnostics (e.g., as_of_date == 2009-12-31).
        if ltyp == "date" and isinstance(lval, str):
            date_literals.add(lval)
        if rtyp == "date" and isinstance(rval, str):
            date_literals.add(rval)

        if left_var and rtyp in ("decimal", "rate", "bps", "money", "integer") and isinstance(rval, str):
            if _looks_like_metric_var(left_var):
                d = _parse_decimal_like(rval)
                if d is not None:
                    thresholds.add(d)
                    sources.append({"kind": "direct_cmp", "op": op, "var": left_var, "value": str(d)})
        if right_var and ltyp in ("decimal", "rate", "bps", "money", "integer") and isinstance(lval, str):
            if _looks_like_metric_var(right_var):
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
            ctyp, cval = _lit_value(cell)
            if ctyp in ("decimal", "rate", "bps", "money", "integer") and isinstance(cval, str):
                d = _parse_decimal_like(cval)
                if d is not None:
                    thresholds.add(d)
                    sources.append({"kind": "table_value", "table_id": ref.table_id, "value_col": ref.value_col, "value": str(d)})
            # Collect date literals from any date-type cells (common for lookup_range bounds).
            for v in cells.values():
                vtyp, vval = _lit_value(v)
                if vtyp == "date" and isinstance(vval, str):
                    date_literals.add(vval)

    return (thresholds, sources, date_literals)


def _resolve_expr_from_bool_expr_spec(spec: Any, *, derived_map: Mapping[str, Any]) -> Optional[Any]:
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
    return spec.get("expr")


def _collect_var_tokens(expr: Any) -> List[str]:
    toks: List[str] = []
    for n in _iter_ast_nodes(expr):
        v = _var_name(n)
        if not v:
            continue
        toks.extend([t for t in re.split(r"[^a-z0-9]+", v.lower().replace("_", " ")) if t])
    return toks


def _summarize_item(
    *,
    item: Mapping[str, Any],
    covenantir_dir: Path,
    retrieval_dir: Path,
) -> Dict[str, Any]:
    item_id = str(item.get("item_id") or "")
    baseline = item.get("baseline") if isinstance(item.get("baseline"), dict) else {}

    baseline_name = _norm_ws(str(baseline.get("name") or ""))
    baseline_limit_raw = _norm_ws(str(baseline.get("limit") or ""))
    baseline_start_raw = _norm_ws(str(baseline.get("start_date") or ""))
    baseline_end_raw = _norm_ws(str(baseline.get("end_date") or ""))

    baseline_limit = _parse_decimal_like(baseline_limit_raw)
    baseline_start_iso = _parse_us_date_to_iso(baseline_start_raw)
    baseline_end_iso = _parse_us_date_to_iso(baseline_end_raw)

    snippet_text = ""
    snippet_path = retrieval_dir / f"{item_id}_snippets.jsonl"
    if snippet_path.exists():
        for rec in _iter_jsonl(snippet_path):
            snip = rec.get("snippet")
            if isinstance(snip, str) and snip.strip():
                snippet_text = snip
                break

    baseline_name_in_excerpt = False
    if baseline_name and snippet_text:
        baseline_name_in_excerpt = _norm_ws(baseline_name).lower() in _norm_ws(snippet_text).lower()

    baseline_limit_in_excerpt = False
    baseline_limit_in_excerpt_kind: Optional[str] = None
    if baseline_limit is not None and snippet_text:
        req_keywords = [t for t in _tokenize(baseline_name) if len(t) >= 4]
        baseline_limit_in_excerpt, baseline_limit_in_excerpt_kind = _limit_in_text(
            limit=baseline_limit,
            text=snippet_text,
            required_keywords=req_keywords,
        )

    out: Dict[str, Any] = {
        "item_id": item_id,
        "baseline": {
            "name": baseline_name,
            "limit_raw": baseline_limit_raw,
            "limit_decimal": str(baseline_limit) if baseline_limit is not None else None,
            "start_date_raw": baseline_start_raw,
            "start_date_iso": baseline_start_iso,
            "end_date_raw": baseline_end_raw,
            "end_date_iso": baseline_end_iso,
        },
        "excerpt": {
            "snippet_found": bool(snippet_text),
            "snippet_len": len(snippet_text) if snippet_text else 0,
            "snippet_preview": _preview(snippet_text, limit=260) if snippet_text else "",
            "baseline_name_in_excerpt": baseline_name_in_excerpt,
            "baseline_limit_in_excerpt": baseline_limit_in_excerpt,
            "baseline_limit_in_excerpt_kind": baseline_limit_in_excerpt_kind,
            "baseline_start_date_in_excerpt": bool(baseline_start_raw and baseline_start_raw in snippet_text),
            "baseline_end_date_in_excerpt": bool(baseline_end_raw and baseline_end_raw in snippet_text),
        },
        "covenantir": {
            "status": "missing",
            "covenants_count": None,
            "open_items_blocking": None,
            "open_item_issue": None,
        },
        "match": {
            "baseline_limit_found_any": False,
            "baseline_limit_found_best": False,
            "baseline_limit_match_kind_any": None,
            "baseline_limit_match_kind_best": None,
            "baseline_start_date_found_any": False,
            "baseline_end_date_found_any": False,
            "baseline_identity_tokens": [],
            "baseline_identity_overlap_best": [],
            "baseline_identity_ok_best": False,
        },
        "disposition": None,
        "candidates": [],
        "best_match": None,
    }

    item_dir = covenantir_dir / item_id
    validated_path = item_dir / "covenantir_validated.json"
    result_path = item_dir / "result.json"
    if not validated_path.exists():
        return out

    doc = _read_json(validated_path)
    result = _read_json(result_path) if result_path.exists() else {}

    open_items = doc.get("open_items") if isinstance(doc, dict) else None
    covenants = doc.get("covenants") if isinstance(doc, dict) else None
    tables = doc.get("tables") if isinstance(doc, dict) else None
    derived = doc.get("derived") if isinstance(doc, dict) else None

    out["covenantir"]["status"] = str(result.get("status") or ("blocked" if open_items else "ok"))
    out["covenantir"]["covenants_count"] = len(covenants) if isinstance(covenants, list) else None
    out["covenantir"]["open_items_blocking"] = int(result.get("open_items_blocking") or 0) if isinstance(result, dict) else None

    if isinstance(open_items, list) and open_items:
        # Precision-first means no covenants/tables are present; issue is the main info.
        issue = open_items[0].get("issue") if isinstance(open_items[0], dict) else None
        out["covenantir"]["open_item_issue"] = str(issue or "")

        issue_text = out["covenantir"]["open_item_issue"] or ""
        if _looks_out_of_scope_issue(issue_text):
            out["disposition"] = "blocked_out_of_scope_issue"
        else:
            btoks = set(_tokenize(baseline_name))
            itoks = set(_tokenize(issue_text))
            if btoks and (btoks & itoks):
                out["disposition"] = "blocked_baseline_missing_terms"
            else:
                out["disposition"] = "blocked_other_missing_terms"
        return out

    table_map: Dict[str, Any] = {}
    if isinstance(tables, list):
        for t in tables:
            if isinstance(t, dict) and isinstance(t.get("table_id"), str):
                table_map[t["table_id"]] = t

    derived_map: Dict[str, Any] = {}
    if isinstance(derived, list):
        for df in derived:
            if isinstance(df, dict) and isinstance(df.get("fn_id"), str):
                derived_map[df["fn_id"]] = df

    baseline_tokens = _tokenize(baseline_name)
    identity_tokens, _ = _identity_overlap_tokens(baseline_tokens=baseline_tokens, candidate_tokens=[])
    out["match"]["baseline_identity_tokens"] = identity_tokens

    summaries: List[CovenantSummary] = []
    all_thresholds: set[Decimal] = set()
    all_dates: set[str] = set()

    if isinstance(covenants, list):
        for cov in covenants:
            if not isinstance(cov, dict):
                continue
            cid = str(cov.get("covenant_id") or "")
            title = str(cov.get("title") or "")

            # Extract thresholds/dates from test expr (and referenced tables).
            test_spec = cov.get("test")
            expr = _resolve_expr_from_bool_expr_spec(test_spec, derived_map=derived_map)
            if expr is None:
                thresholds: set[Decimal] = set()
                sources: List[Dict[str, Any]] = []
                dates: set[str] = set()
            else:
                thresholds, sources, dates = _collect_thresholds_from_expr(expr, table_map=table_map)

            all_thresholds |= thresholds
            all_dates |= dates

            cov_tokens = _tokenize(" ".join([cid, title]))
            jac = _jaccard(baseline_tokens, cov_tokens)
            seq = _seq_ratio(baseline_name, title or cid)
            var_toks = _collect_var_tokens(expr) if expr is not None else []
            var_score = _jaccard(baseline_tokens, var_toks)
            combined = 0.5 * seq + 0.3 * jac + 0.2 * var_score

            summaries.append(
                CovenantSummary(
                    covenant_id=cid,
                    title=title,
                    name_score=jac,
                    seq_score=seq,
                    var_score=var_score,
                    combined_score=combined,
                    threshold_values=[str(d) for d in sorted(thresholds)],
                    threshold_sources=sources,
                    schedule_dates=sorted(dates),
                )
            )

    # Global presence checks (baseline value/date appears anywhere in the output).
    if baseline_limit is not None:
        if baseline_limit in all_thresholds:
            out["match"]["baseline_limit_found_any"] = True
            out["match"]["baseline_limit_match_kind_any"] = "exact"
        else:
            # Heuristic: baseline limit sometimes appears in percent units (75) while IR encodes decimal factors (0.75).
            # Only apply for plausible percentages.
            if baseline_limit > 1 and baseline_limit <= 100:
                as_decimal = (baseline_limit / Decimal("100")).quantize(Decimal("0.0001"))
                if as_decimal in all_thresholds:
                    out["match"]["baseline_limit_found_any"] = True
                    out["match"]["baseline_limit_match_kind_any"] = "percent_to_decimal"
            elif baseline_limit >= 0 and baseline_limit < 1:
                as_percent = (baseline_limit * Decimal("100")).quantize(Decimal("0.0001"))
                if as_percent in all_thresholds:
                    out["match"]["baseline_limit_found_any"] = True
                    out["match"]["baseline_limit_match_kind_any"] = "decimal_to_percent"
    if baseline_start_iso and baseline_start_iso in all_dates:
        out["match"]["baseline_start_date_found_any"] = True
    if baseline_end_iso and baseline_end_iso in all_dates:
        out["match"]["baseline_end_date_found_any"] = True

    # Pick best match covenant.
    summaries_sorted = sorted(
        summaries,
        key=lambda s: (
            -s.combined_score,
            -(baseline_limit is not None and _limit_matches_threshold_values(baseline_limit=baseline_limit, threshold_values=s.threshold_values)),
            -len(_tokenize(s.title)),
        ),
    )
    top = summaries_sorted[:5]
    out["candidates"] = [
        {
            "covenant_id": c.covenant_id,
            "title": c.title,
            "name_score": float(f"{c.name_score:.3f}"),
            "seq_score": float(f"{c.seq_score:.3f}"),
            "var_score": float(f"{c.var_score:.3f}"),
            "combined_score": float(f"{c.combined_score:.3f}"),
            "identity_overlap_tokens": _identity_overlap_tokens(
                baseline_tokens=baseline_tokens,
                candidate_tokens=_tokenize(" ".join([c.covenant_id, c.title])),
            )[1],
            "threshold_values": c.threshold_values,
            "schedule_dates": c.schedule_dates,
        }
        for c in top
    ]

    if top:
        best = top[0]
        out["best_match"] = {
            "covenant_id": best.covenant_id,
            "title": best.title,
            "combined_score": float(f"{best.combined_score:.3f}"),
            "threshold_values": best.threshold_values,
            "schedule_dates": best.schedule_dates,
            "threshold_sources": best.threshold_sources[:20],
        }

        # Baseline covenant identity (name/type) match: require overlap on non-generic baseline tokens.
        _, overlap = _identity_overlap_tokens(
            baseline_tokens=baseline_tokens,
            candidate_tokens=_tokenize(" ".join([best.covenant_id, best.title])),
        )
        out["match"]["baseline_identity_overlap_best"] = overlap
        out["match"]["baseline_identity_ok_best"] = bool(overlap)

        if baseline_limit is not None:
            best_thresholds = [_parse_decimal_like(v) for v in best.threshold_values]
            best_thresholds = [d for d in best_thresholds if d is not None]
            if any(d == baseline_limit for d in best_thresholds):
                out["match"]["baseline_limit_found_best"] = True
                out["match"]["baseline_limit_match_kind_best"] = "exact"
            elif baseline_limit > 1 and baseline_limit <= 100:
                as_decimal = (baseline_limit / Decimal("100")).quantize(Decimal("0.0001"))
                if any(d == as_decimal for d in best_thresholds):
                    out["match"]["baseline_limit_found_best"] = True
                    out["match"]["baseline_limit_match_kind_best"] = "percent_to_decimal"
            elif baseline_limit >= 0 and baseline_limit < 1:
                as_percent = (baseline_limit * Decimal("100")).quantize(Decimal("0.0001"))
                if any(d == as_percent for d in best_thresholds):
                    out["match"]["baseline_limit_found_best"] = True
                    out["match"]["baseline_limit_match_kind_best"] = "decimal_to_percent"

    # Final disposition classification (evaluation-only).
    status = str(out["covenantir"].get("status") or "")
    if status == "ok":
        cov_count = out["covenantir"].get("covenants_count")
        if cov_count == 0:
            # Often indicates the excerpt isn't actually a financial-covenant clause under our scope.
            excerpt_l = _norm_ws(snippet_text).lower()
            if "negative covenants" in excerpt_l or "article vii" in excerpt_l:
                out["disposition"] = "ok_zero_covenants_likely_negative_covenant_excerpt"
            else:
                out["disposition"] = "ok_zero_covenants"
        elif out["match"]["baseline_limit_found_best"] is True and out["match"].get("baseline_identity_ok_best") is True:
            out["disposition"] = "ok_baseline_matched"
        elif out["match"]["baseline_limit_found_best"] is True:
            out["disposition"] = "ok_baseline_limit_matched_wrong_covenant"
        else:
            if baseline_limit is None:
                out["disposition"] = "ok_no_baseline_limit"
            elif out.get("excerpt", {}).get("baseline_limit_in_excerpt") is True:
                out["disposition"] = "ok_baseline_limit_missing_despite_in_excerpt"
            else:
                out["disposition"] = "ok_baseline_limit_not_in_excerpt"

    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True, help="Local sample run_id under runs/<run-id>/")
    ap.add_argument("--covenantir-out-dir", required=True, help="Scratch output dir from CovenantIR batch runner")
    ap.add_argument("--base-dir", default=".", help="Repo base dir")
    ap.add_argument("--out", required=True, help="Path to write JSON report")
    args = ap.parse_args()

    base_dir = Path(args.base_dir)
    run_dir = base_dir / "runs" / args.run_id
    manifest_path = run_dir / "local_sample_manifest.json"
    if not manifest_path.exists():
        raise SystemExit(f"Missing local_sample_manifest.json: {manifest_path} (run build_local_covenant_sample_run.py first)")

    manifest = _read_json(manifest_path)
    items = manifest.get("items")
    if not isinstance(items, list) or not items:
        raise SystemExit(f"Manifest items missing/empty: {manifest_path}")

    retrieval_dir = run_dir / "retrieval_v2"
    if not retrieval_dir.exists():
        raise SystemExit(f"Missing retrieval_v2 dir: {retrieval_dir}")

    covenantir_dir = Path(args.covenantir_out_dir)
    per_item: List[Dict[str, Any]] = []
    for it in items:
        if not isinstance(it, dict):
            continue
        per_item.append(_summarize_item(item=it, covenantir_dir=covenantir_dir, retrieval_dir=retrieval_dir))

    # Aggregate counts
    counts: Dict[str, int] = {}
    match_counts: Dict[str, int] = {
        "baseline_limit_found_any": 0,
        "baseline_limit_found_best": 0,
        "baseline_start_date_found_any": 0,
        "baseline_end_date_found_any": 0,
        "baseline_identity_ok_best": 0,
    }
    disposition_counts: Dict[str, int] = {}
    for r in per_item:
        status = str(((r.get("covenantir") or {}) if isinstance(r.get("covenantir"), dict) else {}).get("status") or "unknown")
        counts[status] = counts.get(status, 0) + 1
        disp = str(r.get("disposition") or "unknown")
        disposition_counts[disp] = disposition_counts.get(disp, 0) + 1
        m = r.get("match")
        if isinstance(m, dict):
            for k in match_counts:
                if m.get(k) is True:
                    match_counts[k] += 1

    report = {
        "schema_version": "covenantir_vs_local_baseline_report_v2",
        "run_id": args.run_id,
        "covenantir_out_dir": str(covenantir_dir),
        "counts_by_status": counts,
        "counts_by_disposition": disposition_counts,
        "match_counts": match_counts,
        "items": per_item,
    }
    _write_json(Path(args.out), report)
    print(f"Wrote report: {args.out}")
    print(f"Counts: {counts}")
    print(f"Disposition counts: {disposition_counts}")
    print(f"Match counts: {match_counts}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
