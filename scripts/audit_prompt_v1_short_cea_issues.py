#!/usr/bin/env python3
"""
Audit non-success cases from prompt_v1_short vs CEA baseline reports.

Goal:
  For every non-success item, decide whether the baseline numeric value is actually
  present in the paragraph (what the model saw). This lets us separate:
    - "model missed something that was written in the paragraph"
    - "baseline value is not written in the paragraph (row/paragraph mismatch or incomplete paragraph)"

Inputs:
  - One or more report JSONs produced by scripts/compare_prompt_v1_short_to_cea_baseline.py
    (e.g., scratch/prompt_v1_short_vs_cea_1995q4_v3.json)
  - On-disk paragraph text under runs/<run_id>/normalized/<item_id>/canonical.txt
  - Model output under runs/<run_id>/llm_qa/<prompt_subdir>/<item_id>.txt

Outputs:
  - A CSV with per-item audit flags
  - A JSON summary with aggregate counts
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def _read_json(path: Path) -> Any:
    return json.loads(_read_text(path))


def _norm_ws(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


_TOKEN_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)


def _tokenize(text: str) -> List[str]:
    return [m.group(0).lower() for m in _TOKEN_RE.finditer(text or "")]


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
    if s.startswith("(") and s.endswith(")"):
        inner = s[1:-1].strip()
        inner = re.sub(r"^\s*\$\s*", "", inner).strip()
        s = f"-{inner}"
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
    if baseline_limit > 1 and baseline_limit <= 100:
        as_decimal = (baseline_limit / Decimal("100")).quantize(Decimal("0.0001"))
        if extracted_limit == as_decimal:
            return True
    if baseline_limit >= 0 and baseline_limit < 1:
        as_percent = (baseline_limit * Decimal("100")).quantize(Decimal("0.0001"))
        if extracted_limit == as_percent:
            return True
    return False


def _limit_in_text(*, limit: Decimal, text: str, required_keywords: Sequence[str]) -> Tuple[bool, Optional[str]]:
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
class ExtractedRow:
    name: str
    limit: Optional[Decimal]


def _parse_model_output(raw: str) -> List[ExtractedRow]:
    try:
        obj = json.loads(raw)
    except Exception:
        return []
    if not isinstance(obj, list):
        return []
    out: List[ExtractedRow] = []
    for el in obj:
        if not isinstance(el, dict):
            continue
        name = _norm_ws(el.get("name") or "")
        limit = _parse_decimal_like(el.get("limit"))
        out.append(ExtractedRow(name=name, limit=limit))
    return out


def _best_match_by_name_overlap(
    *,
    baseline_tokens: Sequence[str],
    rows: Sequence[ExtractedRow],
) -> Optional[ExtractedRow]:
    if not rows:
        return None
    best: Optional[ExtractedRow] = None
    best_score = -1
    for r in rows:
        score = len(set(baseline_tokens) & set(_tokenize(r.name)))
        if score > best_score:
            best = r
            best_score = score
    return best


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reports", nargs="+", required=True, help="One or more report JSON paths")
    ap.add_argument("--base-dir", default=".", help="Repo base dir")
    ap.add_argument("--out-csv", required=True, help="CSV output path")
    ap.add_argument("--out-json", required=True, help="JSON summary output path")
    args = ap.parse_args()

    base_dir = Path(args.base_dir)

    rows_out: List[Dict[str, Any]] = []
    summary = {
        "total_issues": 0,
        "by_disposition": Counter(),
        "baseline_value_in_paragraph": Counter(),
        "model_extracted_baseline_value": Counter(),
        "decision": Counter(),
        "decision_by_disposition": defaultdict(Counter),
    }

    for report_path_str in args.reports:
        report_path = Path(report_path_str)
        report = _read_json(report_path)
        run_id = str(report.get("run_id") or "")
        prompt_subdir = str(report.get("prompt_subdir") or "")
        if not run_id:
            raise SystemExit(f"Missing run_id in report: {report_path}")
        run_dir = base_dir / "runs" / run_id
        items = report.get("items")
        if not isinstance(items, list):
            raise SystemExit(f"Missing items list in report: {report_path}")

        for it in items:
            if not isinstance(it, dict):
                continue
            disp = str(it.get("disposition") or "")
            if disp == "ok_baseline_matched":
                continue
            item_id = str(it.get("item_id") or "")
            if not item_id:
                continue

            baseline = it.get("baseline") if isinstance(it.get("baseline"), dict) else {}
            baseline_name = str(baseline.get("name") or "")
            baseline_limit_raw = str(baseline.get("limit_raw") or baseline.get("limit_decimal") or "")
            baseline_limit = _parse_decimal_like(baseline_limit_raw)
            baseline_tokens = _identity_tokens_from_baseline(baseline_name) if baseline_name else []

            canonical_path = run_dir / "normalized" / item_id / "canonical.txt"
            paragraph = _read_text(canonical_path) if canonical_path.exists() else ""

            baseline_name_mentioned = False
            if baseline_tokens and paragraph:
                plow = paragraph.lower()
                baseline_name_mentioned = any(t.lower() in plow for t in baseline_tokens[:3])

            baseline_value_in_paragraph = False
            baseline_value_in_paragraph_kind: Optional[str] = None
            if baseline_limit is not None and paragraph:
                baseline_value_in_paragraph, baseline_value_in_paragraph_kind = _limit_in_text(
                    limit=baseline_limit, text=paragraph, required_keywords=baseline_tokens
                )

            llm_path = run_dir / "llm_qa" / prompt_subdir / f"{item_id}.txt"
            llm_raw = _read_text(llm_path) if llm_path.exists() else ""
            extracted = _parse_model_output(llm_raw)

            model_extracted_baseline_value = False
            if baseline_limit is not None:
                for r in extracted:
                    if r.limit is None:
                        continue
                    if _limit_matches(baseline_limit=baseline_limit, extracted_limit=r.limit):
                        model_extracted_baseline_value = True
                        break

            best_by_name = _best_match_by_name_overlap(baseline_tokens=baseline_tokens, rows=extracted)
            best_name = best_by_name.name if best_by_name else ""
            best_limit = str(best_by_name.limit) if best_by_name and best_by_name.limit is not None else ""

            # Primary decision axis the user asked for.
            if baseline_value_in_paragraph and not model_extracted_baseline_value:
                decision = "model_missed_value_that_is_written"
            elif not baseline_value_in_paragraph:
                if baseline_name_mentioned:
                    decision = "baseline_value_not_written_but_covenant_named"
                else:
                    decision = "baseline_value_not_written_and_covenant_not_named"
            else:
                # Baseline value is in paragraph AND model extracted it somewhere.
                # (If this were also matched on name, it would have been ok_baseline_matched.)
                decision = "model_found_value_but_not_scored_as_match"

            row = {
                "run_id": run_id,
                "prompt_subdir": prompt_subdir,
                "item_id": item_id,
                "disposition": disp,
                "decision": decision,
                "baseline_name": baseline_name,
                "baseline_value": baseline_limit_raw,
                "baseline_value_in_paragraph": baseline_value_in_paragraph,
                "baseline_value_in_paragraph_kind": baseline_value_in_paragraph_kind or "",
                "baseline_name_mentioned": baseline_name_mentioned,
                "model_extracted_baseline_value": model_extracted_baseline_value,
                "model_best_name_by_baseline_tokens": best_name,
                "model_best_limit_by_baseline_tokens": best_limit,
                "canonical_path": str(canonical_path) if canonical_path.exists() else "",
                "llm_output_path": str(llm_path) if llm_path.exists() else "",
            }
            rows_out.append(row)

            summary["total_issues"] += 1
            summary["by_disposition"][disp] += 1
            summary["baseline_value_in_paragraph"][str(bool(baseline_value_in_paragraph)).lower()] += 1
            summary["model_extracted_baseline_value"][str(bool(model_extracted_baseline_value)).lower()] += 1
            summary["decision"][decision] += 1
            summary["decision_by_disposition"][disp][decision] += 1

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "run_id",
        "prompt_subdir",
        "item_id",
        "disposition",
        "decision",
        "baseline_name",
        "baseline_value",
        "baseline_value_in_paragraph",
        "baseline_value_in_paragraph_kind",
        "baseline_name_mentioned",
        "model_extracted_baseline_value",
        "model_best_name_by_baseline_tokens",
        "model_best_limit_by_baseline_tokens",
        "canonical_path",
        "llm_output_path",
    ]
    with out_csv.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        for r in rows_out:
            w.writerow({k: r.get(k, "") for k in fieldnames})

    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    # Convert Counters to plain dicts for JSON.
    summary_json = dict(summary)
    summary_json["by_disposition"] = dict(summary["by_disposition"])
    summary_json["baseline_value_in_paragraph"] = dict(summary["baseline_value_in_paragraph"])
    summary_json["model_extracted_baseline_value"] = dict(summary["model_extracted_baseline_value"])
    summary_json["decision"] = dict(summary["decision"])
    summary_json["decision_by_disposition"] = {k: dict(v) for k, v in summary["decision_by_disposition"].items()}

    out_json.write_text(json.dumps(summary_json, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(f"Wrote CSV: {out_csv}")
    print(f"Wrote JSON: {out_json}")
    print(json.dumps(summary_json, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
