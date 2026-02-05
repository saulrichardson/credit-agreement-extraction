#!/usr/bin/env python3
"""
Generate an issue-focused LaTeX writeup for prompt_v1_short extractions vs the CEA baseline.

This is reviewer-oriented and artifact-first:
  - For each non-success example, embed:
      (1) model extraction output (verbatim JSON)
      (2) baseline reference fields (from manifest.json)
      (3) an excerpt of the agreement paragraph (from normalized/<item_id>/canonical.txt)
      (4) a short explanation of what happened (simple language)

The examples are grouped into three buckets:
  - Extraction gap: the paragraph contains the baseline value but the model output does not match it
  - Evidence mismatch: the paragraph appears to be about something else than the baseline row
  - Blocked / incomplete paragraph: the paragraph names the covenant but does not include the specific numeric threshold

Usage:
  python scripts/generate_prompt_v1_short_cea_issues_tex.py \\
    --reports scratch/prompt_v1_short_vs_cea_1995q4_v2.json scratch/prompt_v1_short_vs_cea_1995q3_v2.json \\
    --out-tex scratch/cea_prompt_v1_short_issues.tex \\
    --compile
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


MAX_BLOCK_CHARS = 40000
EXCERPT_WINDOW_CHARS = 900


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def _read_json(path: Path) -> Any:
    return json.loads(_read_text(path))


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _truncate(text: str, limit: int = MAX_BLOCK_CHARS) -> str:
    if len(text) <= limit:
        return text
    head = text[:limit]
    return f"{head}\n\n[...truncated: {len(text)} chars total, showing first {limit}...]\n"


def _tex_escape_inline(text: str) -> str:
    repl = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(repl.get(ch, ch) for ch in (text or ""))


def _verbatim_block(text: str) -> str:
    return (
        "\\begin{Verbatim}[breaklines=true,breakanywhere=true,fontsize=\\small]\n"
        + (text or "")
        + "\n\\end{Verbatim}\n"
    )


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


def _candidate_number_strings(limit: Decimal) -> List[str]:
    # Used to find a good excerpt window around the numeric value.
    out: List[str] = []
    d = limit
    out.extend(
        [
            str(d),
            f"{d:.0f}",
            f"{d:.2f}",
            f"{d:.3f}",
            f"{d:.4f}",
        ]
    )
    # Comma-separated integer formatting for large numbers.
    try:
        if d == d.to_integral_value():
            out.append(f"{int(abs(d)):,}")
            out.append(f"({int(abs(d)):,})")
            out.append(f"$({int(abs(d)):,})")
            out.append(f"${int(abs(d)):,}")
    except Exception:
        pass
    # Common alternate representations.
    if d >= 0 and d < 1:
        out.append(str(d).lstrip("0"))  # ".55"
        as_percent = (d * Decimal("100")).quantize(Decimal("0.0001"))
        out.extend([f"{as_percent:.0f}%", f"{as_percent:.2f}%"])
    if d > 1 and d <= 100:
        as_decimal = (d / Decimal("100")).quantize(Decimal("0.0001"))
        out.extend([str(as_decimal), f"{as_decimal:.2f}", f"{as_decimal:.4f}"])
    # Dedupe while preserving order.
    seen: set[str] = set()
    uniq: list[str] = []
    for s in out:
        s = str(s or "")
        if not s:
            continue
        if s in seen:
            continue
        seen.add(s)
        uniq.append(s)
    return uniq


def _excerpt_window(*, text: str, anchor: Optional[str], window: int = EXCERPT_WINDOW_CHARS) -> str:
    if not text:
        return ""
    if not anchor:
        return text[:window]
    hay = text
    idx = hay.lower().find(anchor.lower())
    if idx < 0:
        return text[:window]
    start = max(0, idx - window // 3)
    end = min(len(text), start + window)
    return text[start:end]


def _pretty_json(obj: Any) -> str:
    return json.dumps(obj, indent=2, ensure_ascii=False, sort_keys=True)


def _explain(
    *,
    disposition: str,
    baseline_name: str,
    baseline_limit_raw: str,
    baseline_limit_in_excerpt: bool,
    baseline_name_in_excerpt: bool,
    best_match_name: Optional[str],
    best_match_limit: Optional[str],
) -> str:
    # Keep this short and non-technical (no "token overlap", "heuristics", etc).
    bname = baseline_name.strip() or "the baseline covenant"
    blim = baseline_limit_raw.strip() or "the baseline value"
    mname = (best_match_name or "").strip()
    mlim = (best_match_limit or "").strip()

    if disposition == "ok_baseline_limit_missing_despite_in_excerpt":
        if mlim:
            return (
                f'The paragraph includes the value {blim} for "{bname}", but the model output used {mlim}'
                + (f' for "{mname}".' if mname else ".")
            )
        return f'The paragraph includes the value {blim} for "{bname}", but the model output did not include that value.'

    if disposition == "ok_baseline_limit_matched_wrong_covenant":
        if mname:
            return (
                f'The model output includes the number {blim}, but it is attached to "{mname}" rather than "{bname}".'
            )
        return f'The model output includes the number {blim}, but it is not attached to "{bname}".'

    if disposition == "ok_baseline_limit_not_in_excerpt":
        if baseline_name_in_excerpt and not baseline_limit_in_excerpt:
            return f'The paragraph names "{bname}", but the specific numeric threshold {blim} is not written in this paragraph.'
        if not baseline_name_in_excerpt and mname:
            return f'This paragraph appears to be about "{mname}", not "{bname}". The baseline value {blim} is not shown here.'
        return f'The baseline value {blim} is not shown in this paragraph for "{bname}".'

    return f'The baseline row "{bname}" with value {blim} did not match the model output for this paragraph.'


@dataclass(frozen=True)
class Issue:
    run_id: str
    prompt_subdir: str
    item_id: str
    disposition: str
    baseline_name: str
    baseline_limit_raw: str
    baseline_start_date: str
    baseline_end_date: str
    baseline_name_in_excerpt: bool
    baseline_limit_in_excerpt: bool
    baseline_limit_in_excerpt_kind: Optional[str]
    best_match_name: Optional[str]
    best_match_limit: Optional[str]
    source_file: str
    source_row_index: Optional[int]
    source_segment_no: str


def _load_manifest_index(run_dir: Path) -> Dict[str, Dict[str, Any]]:
    manifest_path = run_dir / "manifest.json"
    manifest = _read_json(manifest_path)
    items = manifest.get("items")
    if not isinstance(items, list):
        raise ValueError(f"manifest.json has no items list: {manifest_path}")
    out: Dict[str, Dict[str, Any]] = {}
    for it in items:
        if not isinstance(it, dict):
            continue
        item_id = str(it.get("item_id") or "")
        if item_id:
            out[item_id] = it
    return out


def _issue_bucket(issue: Issue) -> str:
    if issue.disposition == "ok_baseline_limit_missing_despite_in_excerpt":
        return "Extraction gap"
    if issue.disposition == "ok_baseline_limit_matched_wrong_covenant":
        return "Evidence mismatch"
    if issue.disposition == "ok_baseline_limit_not_in_excerpt":
        if issue.baseline_name_in_excerpt and not issue.baseline_limit_in_excerpt:
            return "Blocked / incomplete paragraph"
        return "Evidence mismatch"
    return "Other"


def _compile_latex(tex_path: Path) -> Path:
    out_dir = tex_path.parent
    cmd = [
        "pdflatex",
        "-interaction=nonstopmode",
        "-halt-on-error",
        f"-output-directory={out_dir}",
        str(tex_path),
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    pdf_path = tex_path.with_suffix(".pdf")
    if not pdf_path.exists():
        raise FileNotFoundError(f"Expected PDF not found: {pdf_path}")
    return pdf_path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reports", nargs="+", required=True, help="One or more comparison report JSON paths")
    ap.add_argument("--base-dir", default=".", help="Repo base dir")
    ap.add_argument("--out-tex", required=True, help="Where to write the LaTeX file")
    ap.add_argument("--compile", action="store_true", help="Compile PDF via pdflatex")
    args = ap.parse_args()

    base_dir = Path(args.base_dir)
    now = datetime.now(timezone.utc).astimezone()

    issues: List[Issue] = []
    for report_path_str in args.reports:
        report_path = Path(report_path_str)
        report = _read_json(report_path)
        run_id = str(report.get("run_id") or "")
        prompt_subdir = str(report.get("prompt_subdir") or "")
        if not run_id:
            raise SystemExit(f"Missing run_id in report: {report_path}")
        run_dir = base_dir / "runs" / run_id
        manifest_index = _load_manifest_index(run_dir)

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

            excerpt = it.get("excerpt") if isinstance(it.get("excerpt"), dict) else {}
            baseline = it.get("baseline") if isinstance(it.get("baseline"), dict) else {}
            best = it.get("best_match") if isinstance(it.get("best_match"), dict) else {}

            m_it = manifest_index.get(item_id, {})
            m_baseline = m_it.get("baseline") if isinstance(m_it.get("baseline"), dict) else {}
            m_source = m_it.get("source") if isinstance(m_it.get("source"), dict) else {}

            issues.append(
                Issue(
                    run_id=run_id,
                    prompt_subdir=prompt_subdir,
                    item_id=item_id,
                    disposition=disp,
                    baseline_name=str(baseline.get("name") or m_baseline.get("name") or ""),
                    baseline_limit_raw=str(baseline.get("limit_raw") or baseline.get("limit_decimal") or m_baseline.get("limit") or ""),
                    baseline_start_date=str(m_baseline.get("start_date") or ""),
                    baseline_end_date=str(m_baseline.get("end_date") or ""),
                    baseline_name_in_excerpt=bool(excerpt.get("baseline_name_in_excerpt") is True),
                    baseline_limit_in_excerpt=bool(excerpt.get("baseline_limit_in_excerpt") is True),
                    baseline_limit_in_excerpt_kind=excerpt.get("baseline_limit_in_excerpt_kind") if isinstance(excerpt.get("baseline_limit_in_excerpt_kind"), str) else None,
                    best_match_name=str(best.get("name") or "") if best else None,
                    best_match_limit=str(best.get("limit") or "") if best else None,
                    source_file=str(m_source.get("file") or ""),
                    source_row_index=int(m_source.get("row_index")) if isinstance(m_source.get("row_index"), int) else None,
                    source_segment_no=str(m_source.get("segment_no") or ""),
                )
            )

    issues_sorted = sorted(issues, key=lambda i: (i.run_id, _issue_bucket(i), i.item_id))

    buckets: Dict[str, List[Issue]] = {}
    for iss in issues_sorted:
        buckets.setdefault(_issue_bucket(iss), []).append(iss)

    total = len(issues_sorted)
    header = rf"""
\documentclass[11pt]{{article}}
\usepackage[margin=1in]{{geometry}}
\usepackage[T1]{{fontenc}}
\usepackage[utf8]{{inputenc}}
\usepackage{{hyperref}}
\usepackage{{fvextra}}
\usepackage{{xcolor}}

\title{{CEA (1995 Q3/Q4): prompt\_v1\_short issue review}}
\author{{Generated {now.strftime("%Y-%m-%d %H:%M:%S %Z")}}}
\date{{}}

\begin{{document}}
\maketitle

\section*{{Summary}}
\begin{{itemize}}
  \item Total non-success cases: {total}
"""
    for bucket_name in ["Extraction gap", "Evidence mismatch", "Blocked / incomplete paragraph", "Other"]:
        n = len(buckets.get(bucket_name, []))
        header += f"  \\item { _tex_escape_inline(bucket_name) }: {n}\n"
    header += "\\end{itemize}\n\n"

    body_parts: List[str] = [header]

    for bucket_name in ["Extraction gap", "Evidence mismatch", "Blocked / incomplete paragraph", "Other"]:
        bucket_items = buckets.get(bucket_name, [])
        if not bucket_items:
            continue
        body_parts.append(f"\\section*{{{_tex_escape_inline(bucket_name)}}}\n")
        for iss in bucket_items:
            run_dir = base_dir / "runs" / iss.run_id
            canonical_path = run_dir / "normalized" / iss.item_id / "canonical.txt"
            paragraph = _read_text(canonical_path) if canonical_path.exists() else ""

            llm_out_path = run_dir / "llm_qa" / iss.prompt_subdir / f"{iss.item_id}.txt"
            llm_raw = _read_text(llm_out_path) if llm_out_path.exists() else ""

            baseline_limit = _parse_decimal_like(iss.baseline_limit_raw)
            anchor: Optional[str] = None
            if baseline_limit is not None:
                # Prefer anchoring on the numeric value (or common renderings of it).
                for cand in _candidate_number_strings(baseline_limit):
                    if cand and cand.lower() in paragraph.lower():
                        anchor = cand
                        break
            if not anchor and iss.baseline_name:
                anchor = iss.baseline_name
            excerpt = _excerpt_window(text=paragraph, anchor=anchor, window=EXCERPT_WINDOW_CHARS)

            baseline_obj = {
                "run_id": iss.run_id,
                "item_id": iss.item_id,
                "source_file": iss.source_file or None,
                "segment_no": iss.source_segment_no or None,
                "row_index": iss.source_row_index,
                "baseline": {
                    "financial_covenant": iss.baseline_name or None,
                    "value": iss.baseline_limit_raw or None,
                    "start_date": iss.baseline_start_date or None,
                    "end_date": iss.baseline_end_date or None,
                },
            }

            explanation = _explain(
                disposition=iss.disposition,
                baseline_name=iss.baseline_name,
                baseline_limit_raw=iss.baseline_limit_raw,
                baseline_limit_in_excerpt=iss.baseline_limit_in_excerpt,
                baseline_name_in_excerpt=iss.baseline_name_in_excerpt,
                best_match_name=iss.best_match_name,
                best_match_limit=iss.best_match_limit,
            )

            body_parts.append(
                "\\subsection*{"
                + _tex_escape_inline(f"{iss.run_id} — {iss.item_id}")
                + "}\n"
                + "\\textbf{Disposition}: "
                + _tex_escape_inline(iss.disposition)
                + "\\\\\n"
                + "\\textbf{Baseline reference}\\\\\n"
                + _verbatim_block(_truncate(_pretty_json(baseline_obj)))
                + "\\textbf{Model extraction (verbatim)}\\\\\n"
                + _verbatim_block(_truncate(llm_raw.strip()))
                + "\\textbf{Agreement excerpt (paragraph window)}\\\\\n"
                + _verbatim_block(_truncate(excerpt.strip()))
                + "\\textbf{What happened}\\\\\n"
                + _tex_escape_inline(explanation)
                + "\n\n"
            )

    body_parts.append("\\end{document}\n")

    out_tex = Path(args.out_tex)
    _write_text(out_tex, "".join(body_parts))
    print(f"Wrote TeX: {out_tex}")

    if args.compile:
        pdf_path = _compile_latex(out_tex)
        print(f"Wrote PDF: {pdf_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
