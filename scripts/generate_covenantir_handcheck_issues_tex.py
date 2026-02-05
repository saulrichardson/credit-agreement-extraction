#!/usr/bin/env python3
"""
Generate an issue-focused LaTeX writeup for the CovenantIR handcheck evaluation.

This is deliberately artifact-first and reviewer-friendly:
  - For each non-success example, embed:
      (1) our extraction (validated JSON; verbatim)
      (2) the baseline reference fields (handcheck row; verbatim)
      (3) a short explanation of what went wrong

The examples are grouped into the same three issue buckets used in the PI summary:
  - Extraction gap (baseline present in excerpt but missed)
  - Evidence mismatch / out-of-scope excerpt
  - Blocked due to incomplete excerpt (missing schedule/terms/truncation)

Usage (Handcheck 30 example):
  python scripts/generate_covenantir_handcheck_issues_tex.py \
    --run-id local-handcheck-covenants-30-20260116 \
    --covenantir-out-dir scratch/covenantir_v0_1_handcheck_30_eval_20260116 \
    --comparison-report scratch/covenantir_v0_1_handcheck_30_eval_20260116/comparison_report_v5.json \
    --out-tex runs/local-handcheck-covenants-30-20260116/deliverables/covenantir_eval_bundle_20260117/PI_issues_writeup.tex \
    --compile
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence


MAX_BLOCK_CHARS = 40000
EXCERPT_WINDOW_CHARS = 900

DISP_EXTRACTION_GAP = {"ok_baseline_limit_missing_despite_in_excerpt"}
DISP_EVIDENCE_MISMATCH = {
    "ok_baseline_limit_matched_wrong_covenant",
    "ok_baseline_limit_not_in_excerpt",
    "ok_zero_covenants_likely_negative_covenant_excerpt",
}
DISP_BLOCKED_INCOMPLETE = {
    "blocked_baseline_missing_terms",
    "blocked_other_missing_terms",
    "blocked_out_of_scope_issue",
}


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def _read_json(path: Path) -> Any:
    return json.loads(_read_text(path))


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _to_pretty_json(obj: Any) -> str:
    return json.dumps(obj, indent=2, ensure_ascii=False, sort_keys=True)


def _truncate(text: str, limit: int = MAX_BLOCK_CHARS) -> str:
    if len(text) <= limit:
        return text
    head = text[:limit]
    return f"{head}\n\n[...truncated: {len(text)} chars total, showing first {limit}...]\n"


def _tex_escape_inline(text: str) -> str:
    # For inline LaTeX text only (not code blocks).
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


def _iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    if not path.exists():
        return
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


def _first_snippet_text(snippet_jsonl: Path) -> str:
    for rec in _iter_jsonl(snippet_jsonl):
        snippet = rec.get("snippet")
        if isinstance(snippet, str) and snippet.strip():
            return snippet
    return ""


def _compact_validated_output(validated: Mapping[str, Any], *, covenant_id: Optional[str] = None) -> Dict[str, Any]:
    """Produce a compact, review-friendly subset of covenantir_validated.json for embedding."""
    out: Dict[str, Any] = {
        "schema_version": validated.get("schema_version"),
        "contract_id": validated.get("contract_id"),
        "open_items": validated.get("open_items"),
    }

    covenants = validated.get("covenants")
    if isinstance(covenants, list) and covenant_id:
        filtered = [c for c in covenants if isinstance(c, dict) and str(c.get("covenant_id") or "") == covenant_id]
        out["covenants"] = filtered if filtered else covenants
    else:
        out["covenants"] = covenants

    tables = validated.get("tables")
    if isinstance(tables, list):
        compact_tables: list[dict[str, Any]] = []
        for t in tables[:3]:
            if not isinstance(t, dict):
                continue
            table_id = t.get("table_id")
            rows = t.get("rows")
            compact_rows: list[Any] = []
            if isinstance(rows, list):
                compact_rows = rows[:3]
            compact_tables.append({"table_id": table_id, "rows_preview": compact_rows})
        if compact_tables:
            out["tables_preview"] = compact_tables
    return out


def _baseline_obj(rep: Mapping[str, Any]) -> Dict[str, Any]:
    baseline = rep.get("baseline") if isinstance(rep.get("baseline"), dict) else {}
    return {
        "name": baseline.get("name"),
        "limit": baseline.get("limit_raw") or baseline.get("limit_decimal"),
        "start_date": baseline.get("start_date_raw") or baseline.get("start_date_iso"),
        "end_date": baseline.get("end_date_raw") or baseline.get("end_date_iso"),
    }


def _snippet_preview(rep: Mapping[str, Any], *, max_chars: int = 260) -> str:
    excerpt = rep.get("excerpt") if isinstance(rep.get("excerpt"), dict) else {}
    prev = excerpt.get("snippet_preview")
    if not isinstance(prev, str):
        return ""
    prev = prev.strip()
    if not prev:
        return ""
    if len(prev) <= max_chars:
        return prev
    return prev[:max_chars].rstrip() + "…"


_GENERIC_TOKENS = {
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
    "of",
    "to",
}


def _tokenize(text: str) -> List[str]:
    toks = re.findall(r"[a-z0-9]+", (text or "").lower())
    return [t for t in toks if t]


def _try_parse_decimal(s: str) -> Optional[Decimal]:
    s = (s or "").strip()
    if not s or s.lower() in {"<na>", "na", "nan", "none", "null"}:
        return None
    cleaned = s.replace(",", "").replace("$", "").replace("%", "")
    # ratio-like "3.50:1.00" => first component
    if ":" in cleaned:
        cleaned = cleaned.split(":", 1)[0]
    m = re.match(r"^-?\d+(?:\.\d+)?", cleaned)
    if m:
        cleaned = m.group(0)
    if not cleaned:
        return None
    try:
        return Decimal(cleaned)
    except InvalidOperation:
        return None


def _limit_variants(limit_raw: str) -> List[str]:
    d = _try_parse_decimal(limit_raw)
    if d is None:
        return [limit_raw.strip()] if limit_raw else []

    # Avoid over-broad matches for tiny integers like "1" or "2".
    if d == d.to_integral_value():
        try:
            as_int = int(d)
        except Exception:
            as_int = None
        # For small integers, searching for the "limit" number is usually too noisy.
        if as_int is None or abs(as_int) < 10:
            return []

    variants: List[str] = []
    if limit_raw:
        variants.append(limit_raw.strip())

    # Non-integer numeric formatting variants: 3.50 -> 3.5, etc.
    base = format(d, "f")
    base = base.rstrip("0").rstrip(".") if "." in base else base
    if base:
        variants.append(base)

    # Two-decimal form (common in agreements): 3.5 -> 3.50
    try:
        variants.append(str(d.quantize(Decimal("0.01"))))
    except Exception:
        pass

    # Percent form (helpful for 0.15 vs 15%). Do NOT include bare "15" without "%".
    try:
        if d >= 0 and d < 1:
            pct = d * Decimal("100")
            pct_s = format(pct, "f")
            pct_s = pct_s.rstrip("0").rstrip(".") if "." in pct_s else pct_s
            if pct_s:
                variants.append(pct_s + "%")
    except Exception:
        pass

    out: List[str] = []
    seen: set[str] = set()
    for v in variants:
        v = (v or "").strip()
        if not v:
            continue
        k = v.lower()
        if k in seen:
            continue
        seen.add(k)
        out.append(v)
    return out


def _find_first_index(haystack: str, needles: Sequence[str]) -> Optional[int]:
    if not haystack:
        return None
    h = haystack.lower()
    best: Optional[int] = None
    for n in needles:
        n = (n or "").strip()
        if not n:
            continue
        idx = h.find(n.lower())
        if idx < 0:
            continue
        if best is None or idx < best:
            best = idx
    return best


def _clip_window(text: str, center: int, window: int = EXCERPT_WINDOW_CHARS) -> str:
    if not text:
        return ""
    center = max(0, min(len(text), center))
    start = max(0, center - window)
    end = min(len(text), center + window)
    prefix = "[…]\n" if start > 0 else ""
    suffix = "\n[…]" if end < len(text) else ""
    return prefix + text[start:end].strip() + suffix


def _relevant_excerpt_blocks(
    *, snippet_text: str, baseline_name: str, baseline_limit_raw: str, best_title: str
) -> List[Dict[str, str]]:
    """Return a small number of excerpt blocks to show what text was available for this example."""
    if not snippet_text.strip():
        return [{"label": "Relevant text", "text": "[missing agreement excerpt text]"}]

    name_tokens = [t for t in _tokenize(baseline_name) if t not in _GENERIC_TOKENS and len(t) >= 4]
    best_tokens = [t for t in _tokenize(best_title) if t not in _GENERIC_TOKENS and len(t) >= 4]
    limit_terms = _limit_variants(baseline_limit_raw)

    idx_name = _find_first_index(snippet_text, [baseline_name] + name_tokens)
    idx_best = _find_first_index(snippet_text, best_tokens)
    idx_limit = _find_first_index(snippet_text, limit_terms)

    blocks: List[Dict[str, str]] = []
    idx_cov = idx_name if idx_name is not None else idx_best
    if idx_cov is not None and idx_limit is not None and abs(idx_cov - idx_limit) <= EXCERPT_WINDOW_CHARS:
        center = int((idx_cov + idx_limit) / 2)
        blocks.append(
            {
                "label": "Text around the covenant mention and the reference number",
                "text": _clip_window(snippet_text, center),
            }
        )
    else:
        if idx_cov is not None:
            blocks.append({"label": "Text around the covenant mention", "text": _clip_window(snippet_text, idx_cov)})
        if idx_limit is not None and limit_terms:
            blocks.append(
                {"label": "Text around the reference number / schedule", "text": _clip_window(snippet_text, idx_limit)}
            )

    if not blocks:
        blocks.append(
            {
                "label": "Text (start of excerpt)",
                "text": snippet_text[: (2 * EXCERPT_WINDOW_CHARS)].strip()
                + ("\n[…]" if len(snippet_text) > 2 * EXCERPT_WINDOW_CHARS else ""),
            }
        )

    # Deduplicate (common when name+number occur close together).
    deduped: List[Dict[str, str]] = []
    seen: set[str] = set()
    for b in blocks:
        key = re.sub(r"\s+", " ", b["text"]).strip()[:500]
        if key in seen:
            continue
        seen.add(key)
        deduped.append(b)
    return deduped[:2]


def _pick_missing_detail(open_issue: str) -> str:
    t = (open_issue or "").lower()
    if not t:
        return "key details needed to write a precise rule"
    # Use token matching to avoid false positives like "compuTABLE".
    toks = set(re.findall(r"[a-z]+", t))
    if "schedule" in toks or "table" in toks or "threshold" in toks:
        return "the schedule/table of limits needed to compute the covenant"
    if "truncated" in toks or "cut" in toks and "off" in toks:
        return "the rest of the clause (the text cuts off before the details)"
    if "capex" in toks or ("capital" in toks and "expenditures" in toks):
        return "the year-by-year capital expenditure limits (and any carry-forward rules)"
    if "definition" in toks or ("defined" in toks and "term" in toks):
        return "the missing definition(s) for one or more defined terms"
    return "some required definitions or numeric values"


def _explain_plain(rep: Mapping[str, Any]) -> str:
    disp = str(rep.get("disposition") or "")
    excerpt = rep.get("excerpt") if isinstance(rep.get("excerpt"), dict) else {}
    name_in = excerpt.get("baseline_name_in_excerpt")
    limit_in = excerpt.get("baseline_limit_in_excerpt")

    cov = rep.get("covenantir") if isinstance(rep.get("covenantir"), dict) else {}
    status = str(cov.get("status") or "")
    open_issue = str(cov.get("open_item_issue") or "").strip()

    best = rep.get("best_match") if isinstance(rep.get("best_match"), dict) else {}
    best_title = str(best.get("title") or "").strip()
    best_id = str(best.get("covenant_id") or "").strip()
    best_thresholds = best.get("threshold_values")
    if not isinstance(best_thresholds, list):
        best_thresholds = []
    best_thresholds = [str(x) for x in best_thresholds if x is not None][:12]

    baseline = rep.get("baseline") if isinstance(rep.get("baseline"), dict) else {}
    baseline_name = str(baseline.get("name") or "").strip()
    baseline_limit = str(baseline.get("limit_raw") or baseline.get("limit_decimal") or "").strip()

    expected_bits: List[str] = []
    if baseline_name:
        expected_bits.append(baseline_name)
    if baseline_limit:
        expected_bits.append(f"limit {baseline_limit}")
    expected = "; ".join(expected_bits) if expected_bits else "the reference covenant"

    observed = best_title or best_id or "no financial covenant test"

    # Plain-English, no evaluator field names / jargon.
    if disp in DISP_EXTRACTION_GAP:
        return (
            f"Based on the handchecked reference, we expected {expected}. The text we provided contains that covenant "
            "and the number, but it does not appear in the system’s result. "
            f"Instead, the system focused on “{observed}”. "
            "In other words: the necessary information was in the text, but it wasn’t captured."
        )

    if disp in DISP_EVIDENCE_MISMATCH:
        if disp == "ok_baseline_limit_not_in_excerpt":
            return (
                f"Based on the handchecked reference, we expected {expected}. The reference number does not appear in "
                "the text we provided, so the system cannot recover it from this text."
            )
        if disp == "ok_zero_covenants_likely_negative_covenant_excerpt":
            return (
                f"Based on the handchecked reference, we expected {expected}. The text we provided appears to be a "
                "general restrictions section, not a financial covenant section, so there was no financial covenant "
                "rule to extract."
            )
        if disp == "ok_baseline_limit_matched_wrong_covenant":
            threshold_note = ""
            if baseline_limit and best_thresholds:
                threshold_note = f" It includes a schedule of numbers that happens to include {baseline_limit}."
            return (
                f"Based on the handchecked reference, we expected {expected}. The system did find a covenant, but it "
                f"returned “{observed}”, which is a different covenant than the reference. This usually happens when "
                "the text lists a schedule of limits without clearly stating which covenant the schedule belongs to."
                + threshold_note
            )

    if disp in DISP_BLOCKED_INCOMPLETE or status == "blocked":
        missing_detail = _pick_missing_detail(open_issue)
        return (
            f"Based on the handchecked reference, we expected {expected}. The system did not guess. It stopped because "
            f"the text we provided does not include {missing_detail}. Without those details, it cannot write a precise, "
            "checkable rule."
        )

    # Fallback (should not happen for issue-only selection).
    return (
        f"We expected {expected}. The outcome does not fall into a known issue category for this writeup. "
        "Please refer to the embedded reference fields and extraction output."
    )


def _run_latexmk(tex_path: Path) -> None:
    workdir = tex_path.parent
    cmd = [
        "latexmk",
        "-xelatex",
        "-interaction=nonstopmode",
        "-halt-on-error",
        tex_path.name,
    ]
    subprocess.run(cmd, cwd=str(workdir), check=True)
    subprocess.run(["latexmk", "-c", tex_path.name], cwd=str(workdir), check=True)


def build_tex(*, run_id: str, covenantir_out_dir: Path, comparison_report: Mapping[str, Any], repo_root: Path) -> str:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    run_dir = repo_root / "runs" / run_id
    retrieval_dir = run_dir / "retrieval_v2"
    if not retrieval_dir.exists():
        raise SystemExit(f"Missing retrieval_v2 dir: {retrieval_dir}")

    report_items = comparison_report.get("items")
    if not isinstance(report_items, list):
        raise SystemExit("comparison_report missing items[]")
    report_items = [it for it in report_items if isinstance(it, dict)]

    # Select the non-success items only (everything except strict success).
    issue_items: list[dict[str, Any]] = [it for it in report_items if str(it.get("disposition") or "") != "ok_baseline_matched"]

    # Group by the three PI buckets.
    extraction_gap = [it for it in issue_items if str(it.get("disposition") or "") in DISP_EXTRACTION_GAP]
    evidence_mismatch = [it for it in issue_items if str(it.get("disposition") or "") in DISP_EVIDENCE_MISMATCH]
    blocked_incomplete = [it for it in issue_items if str(it.get("disposition") or "") in DISP_BLOCKED_INCOMPLETE]

    # Fail loudly if the buckets don't partition the issue set (keeps the report honest).
    covered = {str(it.get("item_id") or "") for it in (extraction_gap + evidence_mismatch + blocked_incomplete)}
    all_issue_ids = {str(it.get("item_id") or "") for it in issue_items}
    missing = sorted(i for i in all_issue_ids - covered if i)
    if missing:
        raise SystemExit(f"Unclassified issue items (update bucketing rules): {missing}")

    # Stable ordering within buckets.
    extraction_gap = sorted(extraction_gap, key=lambda it: str(it.get("item_id") or ""))
    evidence_mismatch = sorted(evidence_mismatch, key=lambda it: str(it.get("item_id") or ""))
    blocked_incomplete = sorted(blocked_incomplete, key=lambda it: str(it.get("item_id") or ""))

    tex: List[str] = []
    tex.append(r"\documentclass[11pt]{article}")
    tex.append(r"\usepackage[margin=1in]{geometry}")
    tex.append(r"\usepackage{iftex}")
    tex.append(r"\ifPDFTeX")
    tex.append(r"\usepackage[T1]{fontenc}")
    tex.append(r"\usepackage[utf8]{inputenc}")
    tex.append(r"\else")
    tex.append(r"\usepackage{fontspec}")
    tex.append(r"\fi")
    tex.append(r"\usepackage{hyperref}")
    tex.append(r"\usepackage{parskip}")
    tex.append(r"\usepackage{xcolor}")
    tex.append(r"\usepackage{fvextra}")
    tex.append(r"\DefineVerbatimEnvironment{Verbatim}{Verbatim}{breaklines=true,breakanywhere=true,fontsize=\small}")
    tex.append(r"\sloppy")
    tex.append(r"\setlength{\emergencystretch}{3em}")
    tex.append(r"\title{" + _tex_escape_inline("Financial Covenant Extraction Eval — Issues (Handcheck 30)") + r"}")
    tex.append(r"\author{}")
    tex.append(r"\date{" + _tex_escape_inline(now) + r"}")
    tex.append(r"\begin{document}")
    tex.append(r"\maketitle")

    tex.append(r"\section*{What this document contains}")
    tex.append(
        _tex_escape_inline(
            "This writeup focuses only on the non-success rows from the Handcheck-30 evaluation. "
            "For each issue row, we embed (1) our extraction JSON, (2) the baseline reference fields, "
            "and (3) a short explanation of what went wrong."
        )
    )

    tex.append(r"\section*{Issue counts (row-level)}")
    tex.append(r"\begin{itemize}")
    tex.append(r"\item " + _tex_escape_inline(f"Extraction gap (baseline present in excerpt but missed): {len(extraction_gap)}"))
    tex.append(r"\item " + _tex_escape_inline(f"Evidence mismatch / out-of-scope excerpt: {len(evidence_mismatch)}"))
    tex.append(
        r"\item "
        + _tex_escape_inline(
            f"Blocked due to incomplete excerpt (missing schedule/terms/truncation): {len(blocked_incomplete)}"
        )
    )
    tex.append(r"\end{itemize}")

    def _add_item(rep: Mapping[str, Any], *, disp: str) -> None:
        item_id = str(rep.get("item_id") or "").strip()
        tex.append(r"\subsection*{" + _tex_escape_inline(f"Item: {item_id}") + r"}")

        # 1) Extraction JSON (first, per request)
        validated_path = covenantir_out_dir / item_id / "covenantir_validated.json"
        tex.append(r"\paragraph{Our extraction (validated JSON; verbatim / compact view)}")
        if validated_path.exists():
            try:
                validated = _read_json(validated_path)
                # For readability: show only the best-match covenant except for extraction-gap rows (show all).
                best = rep.get("best_match") if isinstance(rep.get("best_match"), dict) else {}
                best_cid = str(best.get("covenant_id") or "").strip()
                cid_filter: Optional[str] = None
                if disp != "ok_baseline_limit_missing_despite_in_excerpt" and best_cid:
                    cid_filter = best_cid
                compact = _compact_validated_output(validated if isinstance(validated, dict) else {}, covenant_id=cid_filter)
                tex.append(_verbatim_block(_truncate(_to_pretty_json(compact))))
            except Exception as e:
                tex.append(_verbatim_block(f"[error reading validated json: {e}]"))
        else:
            tex.append(_verbatim_block("[missing covenantir_validated.json]"))

        # 2) Baseline reference fields (second, per request)
        tex.append(r"\paragraph{Baseline reference (handcheck row fields; verbatim)}")
        baseline_obj = _baseline_obj(rep)
        tex.append(_verbatim_block(_truncate(_to_pretty_json(baseline_obj))))

        # 2.5) Relevant agreement excerpt(s)
        tex.append(r"\paragraph{Relevant text from the agreement (verbatim)}")
        snippet_text = _first_snippet_text(retrieval_dir / f"{item_id}_snippets.jsonl")
        best = rep.get("best_match") if isinstance(rep.get("best_match"), dict) else {}
        best_title = str(best.get("title") or "").strip()
        blocks = _relevant_excerpt_blocks(
            snippet_text=snippet_text,
            baseline_name=str(baseline_obj.get("name") or ""),
            baseline_limit_raw=str(baseline_obj.get("limit") or ""),
            best_title=best_title,
        )
        for b in blocks:
            if len(blocks) > 1:
                tex.append(_tex_escape_inline(f"{b['label']}:"))
            tex.append(_verbatim_block(_truncate(b["text"], limit=2 * EXCERPT_WINDOW_CHARS)))

        # 3) Explanation (simple, last)
        tex.append(r"\paragraph{What went wrong}")
        tex.append(_tex_escape_inline(_explain_plain(rep)))

    tex.append(r"\section*{Extraction gap (baseline present in excerpt but missed)}")
    for rep in extraction_gap:
        _add_item(rep, disp=str(rep.get("disposition") or ""))

    tex.append(r"\section*{Evidence mismatch / out-of-scope excerpt}")
    for rep in evidence_mismatch:
        _add_item(rep, disp=str(rep.get("disposition") or ""))

    tex.append(r"\section*{Blocked due to incomplete excerpt}")
    for rep in blocked_incomplete:
        _add_item(rep, disp=str(rep.get("disposition") or ""))

    tex.append(r"\section*{Reproducibility}")
    tex.append(_tex_escape_inline(f"Run id: {run_id}"))
    tex.append(r"\\")
    tex.append(_tex_escape_inline(f"CovenantIR output dir: {str(covenantir_out_dir)}"))
    tex.append(r"\\")
    tex.append(_tex_escape_inline("Comparison report: ") + _tex_escape_inline(str(comparison_report.get("schema_version") or "")))

    tex.append(r"\end{document}")
    return "\n".join(tex).rstrip() + "\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--covenantir-out-dir", required=True)
    ap.add_argument("--comparison-report", required=True)
    ap.add_argument("--out-tex", required=True)
    ap.add_argument("--compile", action="store_true", help="Compile PDF with latexmk after writing .tex")
    args = ap.parse_args()

    repo_root = Path(".").resolve()
    covenantir_out_dir = Path(args.covenantir_out_dir)
    if not covenantir_out_dir.exists():
        raise SystemExit(f"Missing CovenantIR output dir: {covenantir_out_dir}")

    report_path = Path(args.comparison_report)
    if not report_path.exists():
        raise SystemExit(f"Missing comparison report: {report_path}")
    report = _read_json(report_path)

    tex = build_tex(run_id=args.run_id, covenantir_out_dir=covenantir_out_dir, comparison_report=report, repo_root=repo_root)
    out_tex = Path(args.out_tex)
    _write_text(out_tex, tex)
    print(f"[done] wrote tex: {out_tex}")

    if args.compile:
        _run_latexmk(out_tex)
        print(f"[done] compiled pdf: {out_tex.with_suffix('.pdf')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
