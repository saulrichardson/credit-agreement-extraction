#!/usr/bin/env python3
"""
Compare prompt_v1_short extractions vs CEA (credit_agreement_analysis_*) baseline rows.

This is for runs created by scripts/build_cea_paragraph_run_v2.py.

Inputs
------
- runs/<run_id>/manifest.json
    items[*].baseline.name   (from CEA: financial_covenant)
    items[*].baseline.limit  (from CEA: value)
    items[*].baseline.start_date / end_date (optional; currently not scored)
- runs/<run_id>/retrieval_v2/<item_id>_snippets.jsonl
    snippet text that was fed to the model (one record per item)
- runs/<run_id>/llm_qa/<prompt_subdir>/<item_id>.txt
    raw model output (expected: JSON list with objects having keys:
      name, limit, start date, end date)

Output
------
- JSON report with per-item disposition + aggregate counts, written to --out.

Matching definition (evaluation-only)
-------------------------------------
Row-level success requires BOTH:
1) identity match: overlap on baseline covenant name tokens (>=4 chars, with light generic filtering)
2) limit match: extracted limit equals baseline value (with percent<->decimal normalization allowed)
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8", errors="replace"))


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")


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


def _norm_ws(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


_TOKEN_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)


def _tokenize(text: str) -> List[str]:
    return [m.group(0).lower() for m in _TOKEN_RE.finditer(text or "")]


# Avoid over-weighting generic name parts.
_GENERIC_TOKENS = {
    "ratio",
    "minimum",
    "maximum",
    "consolidated",
}


def _identity_tokens_from_baseline(baseline_name: str) -> List[str]:
    toks = [t for t in _tokenize(baseline_name) if len(t) >= 4]
    out: List[str] = []
    for t in toks:
        if t in _GENERIC_TOKENS:
            continue
        if t not in out:
            out.append(t)
    return out or toks


def _identity_overlap_tokens(*, baseline_tokens: Sequence[str], candidate_tokens: Sequence[str]) -> List[str]:
    return sorted(set(baseline_tokens) & set(candidate_tokens))


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
    if not s or s.lower() in {"na", "n/a", "none", "<na>", "null", "nan"}:
        return None
    s = s.replace(",", "")
    s = re.sub(r"^\s*\$\s*", "", s).strip()
    # Handle negatives expressed as parentheses: "(200000)" -> -200000
    if s.startswith("(") and s.endswith(")"):
        inner = s[1:-1].strip()
        inner = re.sub(r"^\s*\$\s*", "", inner).strip()
        s = f"-{inner}"
    # Take the leading number (handles "0.65:1" or "2.25 to 1.00").
    m = re.match(r"^(-?(?:\d+(?:\.\d+)?|\.\d+))", s)
    if not m:
        return None
    try:
        return Decimal(m.group(1)).quantize(Decimal("0.0001"))
    except (InvalidOperation, ValueError):
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


def _limit_in_text(*, limit: Decimal, text: str, required_keywords: Sequence[str]) -> Tuple[bool, Optional[str]]:
    if not text:
        return (False, None)
    tlow = _norm_ws(text).lower()
    # Require at least one baseline keyword to appear to reduce false positives.
    if required_keywords and not any(k.lower() in tlow for k in required_keywords[:3]):
        return (False, None)

    expected: List[Decimal] = [limit]
    if limit > 1 and limit <= 100:
        expected.append((limit / Decimal("100")).quantize(Decimal("0.0001")))
    if limit >= 0 and limit < 1:
        expected.append((limit * Decimal("100")).quantize(Decimal("0.0001")))
    expected_set = set(expected)

    # Robust path: parse numeric tokens from text and compare as Decimals.
    #
    # We intentionally handle:
    # - commas: "200,000"
    # - currency: "$200,000"
    # - parens-negatives: "(200,000)"
    # - leading-dot decimals: ".65"
    # - ratios: "2.00 to 1.0" (we parse "2.00")
    num_re = re.compile(
        r"""
        (?P<tok>
            \(?\s*            # optional paren open
            \$?\s*            # optional currency
            -?\s*             # optional minus
            (?:               # number body
                \d{1,3}(?:,\d{3})+(?:\.\d+)?  # 1,234.56
                |
                \d+(?:\.\d+)?                 # 1234.56
                |
                \.\d+                         # .65
            )
            \s*\)?            # optional paren close
            %?                # optional percent sign
        )
        """,
        re.VERBOSE,
    )
    for m in num_re.finditer(text):
        tok = m.group("tok")
        d = _parse_decimal_like(tok)
        if d is None:
            continue
        if d in expected_set:
            return (True, "parsed")
        if abs(d) in expected_set:
            return (True, "parsed_abs")

    return (False, None)


@dataclass(frozen=True)
class ExtractedRow:
    name: str
    limit: Optional[Decimal]
    start_date: str
    end_date: str


def _parse_extracted_list(raw_text: str) -> Tuple[Optional[List[ExtractedRow]], Optional[str]]:
    try:
        obj = json.loads(raw_text)
    except Exception:
        return (None, "invalid_json")
    if not isinstance(obj, list):
        return (None, "not_a_list")
    out: List[ExtractedRow] = []
    for el in obj:
        if not isinstance(el, dict):
            continue
        name = _norm_ws(el.get("name") or "")
        limit = _parse_decimal_like(el.get("limit"))
        start_date = _norm_ws(el.get("start date") or "")
        end_date = _norm_ws(el.get("end date") or "")
        out.append(ExtractedRow(name=name, limit=limit, start_date=start_date, end_date=end_date))
    return (out, None)


def _summarize_item(
    *,
    item: Mapping[str, Any],
    run_dir: Path,
    prompt_subdir: str,
) -> Dict[str, Any]:
    item_id = str(item.get("item_id") or "")
    baseline = item.get("baseline") if isinstance(item.get("baseline"), dict) else {}

    baseline_name = _norm_ws(str(baseline.get("name") or ""))
    if baseline_name.lower() in {"na", "n/a", "none", "<na>", "null", "nan"}:
        baseline_name = ""
    baseline_limit_raw = _norm_ws(str(baseline.get("limit") or ""))
    if baseline_limit_raw.lower() in {"na", "n/a", "none", "<na>", "null", "nan"}:
        baseline_limit_raw = ""
    baseline_limit = _parse_decimal_like(baseline_limit_raw)

    snippet_text = ""
    snippet_path = run_dir / "retrieval_v2" / f"{item_id}_snippets.jsonl"
    if snippet_path.exists():
        for rec in _iter_jsonl(snippet_path):
            snip = rec.get("snippet")
            if isinstance(snip, str) and snip.strip():
                snippet_text = snip
                break

    baseline_tokens = _identity_tokens_from_baseline(baseline_name) if baseline_name else []
    baseline_name_in_excerpt = bool(baseline_name and snippet_text and baseline_name.lower() in snippet_text.lower())

    baseline_limit_in_excerpt = False
    baseline_limit_in_excerpt_kind: Optional[str] = None
    if baseline_limit is not None and snippet_text:
        baseline_limit_in_excerpt, baseline_limit_in_excerpt_kind = _limit_in_text(
            limit=baseline_limit,
            text=snippet_text,
            required_keywords=baseline_tokens,
        )

    llm_dir = run_dir / "llm_qa" / prompt_subdir
    raw_path = llm_dir / f"{item_id}.txt"
    err_path = llm_dir / f"{item_id}.error.txt"

    out: Dict[str, Any] = {
        "item_id": item_id,
        "baseline": {
            "name": baseline_name or None,
            "limit_raw": baseline_limit_raw or None,
            "limit_decimal": str(baseline_limit) if baseline_limit is not None else None,
        },
        "excerpt": {
            "snippet_found": bool(snippet_text),
            "snippet_len": len(snippet_text) if snippet_text else 0,
            "snippet_preview": _preview(snippet_text, limit=260) if snippet_text else "",
            "baseline_name_in_excerpt": baseline_name_in_excerpt,
            "baseline_limit_in_excerpt": baseline_limit_in_excerpt,
            "baseline_limit_in_excerpt_kind": baseline_limit_in_excerpt_kind,
        },
        "prompt_v1_short": {
            "status": "missing_output",
            "raw_path": str(raw_path),
            "error_path": str(err_path),
            "parsed_count": None,
            "parse_error_kind": None,
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

    # Baseline presence gates (CEA can include unlabeled / missing-value rows).
    if not baseline_name and baseline_limit is None:
        out["disposition"] = "no_baseline_label_or_value"
        return out
    if not baseline_name:
        out["disposition"] = "no_baseline_label"
        return out
    if baseline_limit is None:
        out["disposition"] = "no_baseline_value"
        return out

    if err_path.exists():
        out["prompt_v1_short"]["status"] = "error"
        out["disposition"] = "error_gateway"
        return out
    if not raw_path.exists():
        out["disposition"] = "missing_output"
        return out

    raw_text = raw_path.read_text(encoding="utf-8", errors="replace")
    parsed, parse_err = _parse_extracted_list(raw_text)
    if parsed is None:
        out["prompt_v1_short"]["status"] = "invalid_output"
        out["prompt_v1_short"]["parse_error_kind"] = parse_err
        out["disposition"] = "error_invalid_json"
        return out

    out["prompt_v1_short"]["status"] = "ok"
    out["prompt_v1_short"]["parsed_count"] = len(parsed)

    candidates: List[Dict[str, Any]] = []
    for r in parsed:
        overlap = _identity_overlap_tokens(baseline_tokens=baseline_tokens, candidate_tokens=_tokenize(r.name))
        name_score = len(overlap)
        limit_match = bool(r.limit is not None and _limit_matches(baseline_limit=baseline_limit, extracted_limit=r.limit))

        if overlap:
            out["match"]["baseline_identity_ok_any"] = True
        if limit_match:
            out["match"]["baseline_limit_found_any"] = True

        candidates.append(
            {
                "name": r.name,
                "limit": str(r.limit) if r.limit is not None else None,
                "start_date": r.start_date,
                "end_date": r.end_date,
                "identity_overlap_tokens": overlap,
                "name_score": name_score,
                "limit_match": limit_match,
            }
        )

    candidates_sorted = sorted(
        candidates,
        key=lambda c: (
            int(c.get("name_score") or 0),
            int(bool(c.get("limit_match"))),
            len(_tokenize(str(c.get("name") or ""))),
        ),
        reverse=True,
    )
    out["candidates"] = candidates_sorted[:10]

    if candidates_sorted:
        best = candidates_sorted[0]
        out["best_match"] = {
            "name": best.get("name"),
            "limit": best.get("limit"),
            "start_date": best.get("start_date"),
            "end_date": best.get("end_date"),
        }
        out["match"]["baseline_identity_overlap_best"] = list(best.get("identity_overlap_tokens") or [])
        out["match"]["baseline_limit_found_best"] = bool(best.get("limit_match"))

    # Disposition buckets.
    baseline_matched_any = any(bool(c.get("limit_match")) and bool(c.get("identity_overlap_tokens")) for c in candidates_sorted)
    if baseline_matched_any:
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
    ap.add_argument("--run-id", required=True, help="CEA run_id under runs/<run-id>/")
    ap.add_argument("--prompt-subdir", default="prompt_v1_short", help="Subdir under runs/<run-id>/llm_qa/")
    ap.add_argument("--base-dir", default=".", help="Repo base dir")
    ap.add_argument("--out", required=True, help="Path to write JSON report")
    args = ap.parse_args()

    run_dir = Path(args.base_dir) / "runs" / args.run_id
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.exists():
        raise SystemExit(f"Missing manifest.json: {manifest_path}")
    if not (run_dir / "retrieval_v2").exists():
        raise SystemExit(f"Missing retrieval_v2 dir: {run_dir / 'retrieval_v2'}")

    manifest = _read_json(manifest_path)
    items = manifest.get("items")
    if not isinstance(items, list) or not items:
        raise SystemExit(f"manifest.json has no items: {manifest_path}")

    per_item: List[Dict[str, Any]] = []
    for it in items:
        if not isinstance(it, dict):
            continue
        per_item.append(_summarize_item(item=it, run_dir=run_dir, prompt_subdir=str(args.prompt_subdir)))

    counts_by_status: Dict[str, int] = {}
    counts_by_disp: Dict[str, int] = {}
    match_counts: Dict[str, int] = {
        "baseline_identity_ok_any": 0,
        "baseline_limit_found_any": 0,
        "baseline_limit_found_best": 0,
    }
    for r in per_item:
        status = str(((r.get("prompt_v1_short") or {}) if isinstance(r.get("prompt_v1_short"), dict) else {}).get("status") or "unknown")
        counts_by_status[status] = counts_by_status.get(status, 0) + 1
        disp = str(r.get("disposition") or "unknown")
        counts_by_disp[disp] = counts_by_disp.get(disp, 0) + 1
        m = r.get("match")
        if isinstance(m, dict):
            for k in match_counts:
                if m.get(k) is True:
                    match_counts[k] += 1

    report = {
        "schema_version": "prompt_v1_short_vs_cea_baseline_report_v3",
        "run_id": args.run_id,
        "prompt_subdir": str(args.prompt_subdir),
        "counts_by_status": counts_by_status,
        "counts_by_disposition": counts_by_disp,
        "match_counts": match_counts,
        "items": per_item,
    }
    _write_json(Path(args.out), report)
    print(f"Wrote report: {args.out}")
    print(f"Counts by status: {counts_by_status}")
    print(f"Counts by disposition: {counts_by_disp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
