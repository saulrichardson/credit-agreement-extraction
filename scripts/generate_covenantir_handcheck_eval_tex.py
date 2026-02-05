#!/usr/bin/env python3
"""
Generate a PI-facing LaTeX report (and optional PDF) for the CovenantIR handcheck evaluation.

Design goals:
  - Artifact-first: embed the actual snippet inputs and extraction JSON in verbatim blocks.
  - Keep it readable: explain what "match" means at a high level, summarize match-up results, then show a few examples.
  - Examples include only: reference data, source text, and our extraction.

Usage:
  python scripts/generate_covenantir_handcheck_eval_tex.py \
    --run-id local-handcheck-covenants-30-20260116 \
    --covenantir-out-dir scratch/covenantir_v0_1_handcheck_30_eval_20260116 \
    --comparison-report scratch/covenantir_v0_1_handcheck_30_eval_20260116/comparison_report_v5.json \
    --out-tex runs/local-handcheck-covenants-30-20260116/deliverables/covenantir_eval_bundle_20260117/PI_writeup.tex
"""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple


MAX_BLOCK_CHARS = 25000


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def _read_json(path: Path) -> Any:
    return json.loads(_read_text(path))


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


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


def _tex_tt(text: str) -> str:
    return r"\texttt{" + _tex_escape_inline(text) + r"}"


def _verbatim_block(text: str) -> str:
    # fvextra's Verbatim supports wrapping.
    return (
        "\\begin{Verbatim}[breaklines=true,breakanywhere=true,fontsize=\\small]\n"
        + (text or "")
        + "\n\\end{Verbatim}\n"
    )


def _first_snippet_text(snippet_jsonl: Path) -> Tuple[str, Optional[str]]:
    """Return (snippet_text, anchor_id)."""
    if not snippet_jsonl.exists():
        return ("", None)
    for rec in _iter_jsonl(snippet_jsonl):
        snippet = rec.get("snippet")
        if isinstance(snippet, str) and snippet.strip():
            anchor_id = rec.get("anchor_id")
            return (snippet, anchor_id if isinstance(anchor_id, str) else None)
    return ("", None)


def _to_pretty_json(obj: Any) -> str:
    return json.dumps(obj, indent=2, ensure_ascii=False, sort_keys=True)

def _compact_validated_output(validated: Mapping[str, Any], *, covenant_id: Optional[str] = None) -> Dict[str, Any]:
    """Produce a compact, review-friendly subset of covenantir_validated.json for embedding.

    If covenant_id is provided and the output contains covenants, include only that covenant
    when possible (keeps the PDF short while still grounded in the actual artifact).
    """
    out: Dict[str, Any] = {
        "schema_version": validated.get("schema_version"),
        "contract_id": validated.get("contract_id"),
        "open_items": validated.get("open_items"),
    }
    covenants = validated.get("covenants")
    if isinstance(covenants, list) and covenant_id:
        filtered = [c for c in covenants if isinstance(c, dict) and str(c.get("covenant_id") or "") == covenant_id]
        out["covenants"] = filtered if filtered else covenants[:1]
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
                compact_rows = rows[:3]  # show only first few rows
            compact_tables.append({"table_id": table_id, "rows_preview": compact_rows})
        if compact_tables:
            out["tables_preview"] = compact_tables
    return out


def _iter_nodes(node: Any) -> Iterable[Dict[str, Any]]:
    if isinstance(node, dict):
        yield node
        for v in node.values():
            yield from _iter_nodes(v)
    elif isinstance(node, list):
        for v in node:
            yield from _iter_nodes(v)


_NUM_LIT_TYPES = {"decimal", "rate", "bps", "money", "integer"}


def _expr_has_numeric_literal(expr: Any) -> bool:
    for n in _iter_nodes(expr):
        lit = n.get("lit") if isinstance(n, dict) else None
        if isinstance(lit, dict) and lit.get("type") in _NUM_LIT_TYPES:
            return True
    return False


def _read_manifest_items(run_dir: Path) -> list[dict[str, Any]]:
    manifest_path = run_dir / "local_sample_manifest.json"
    if not manifest_path.exists():
        raise SystemExit(f"Missing local_sample_manifest.json: {manifest_path}")
    doc = _read_json(manifest_path)
    items = doc.get("items")
    if not isinstance(items, list) or not items:
        raise SystemExit(f"Manifest items missing/empty: {manifest_path}")
    out: list[dict[str, Any]] = []
    for it in items:
        if not isinstance(it, dict):
            continue
        item_id = it.get("item_id")
        baseline = it.get("baseline")
        if isinstance(item_id, str) and item_id.strip() and isinstance(baseline, dict):
            out.append({"item_id": item_id.strip(), "baseline": baseline})
    if not out:
        raise SystemExit(f"No usable item records in manifest: {manifest_path}")
    return out


def _first_item_id_with_disposition(report_items: list[dict[str, Any]], disposition: str) -> Optional[str]:
    for it in report_items:
        if str(it.get("disposition") or "") != disposition:
            continue
        item_id = it.get("item_id")
        if isinstance(item_id, str) and item_id.strip():
            return item_id.strip()
    return None


def build_tex(*, run_id: str, covenantir_out_dir: Path, comparison_report: Mapping[str, Any], repo_root: Path) -> str:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    run_dir = repo_root / "runs" / run_id
    retrieval_dir = run_dir / "retrieval_v2"
    manifest_items = _read_manifest_items(run_dir)

    # Summarize match-up results (baseline vs extraction) without embedding evaluator artifacts per example.
    total = len(manifest_items)
    report_items = comparison_report.get("items")
    if not isinstance(report_items, list):
        raise SystemExit("comparison_report missing items[]")
    report_items = [it for it in report_items if isinstance(it, dict)]

    counts_by_status = comparison_report.get("counts_by_status") if isinstance(comparison_report.get("counts_by_status"), dict) else {}
    counts_by_disp = comparison_report.get("counts_by_disposition") if isinstance(comparison_report.get("counts_by_disposition"), dict) else {}

    matched = int(counts_by_disp.get("ok_baseline_matched") or 0)
    blocked = int(counts_by_status.get("blocked") or 0)

    # Aggregate the "mismatch" bucket into a couple of core categories that matter for match-up.
    extraction_gap = int(counts_by_disp.get("ok_baseline_limit_missing_despite_in_excerpt") or 0)
    evidence_mismatch = (
        int(counts_by_disp.get("ok_baseline_limit_matched_wrong_covenant") or 0)
        + int(counts_by_disp.get("ok_baseline_limit_not_in_excerpt") or 0)
        + int(counts_by_disp.get("ok_zero_covenants_likely_negative_covenant_excerpt") or 0)
    )
    blocked_incomplete = (
        int(counts_by_disp.get("blocked_baseline_missing_terms") or 0)
        + int(counts_by_disp.get("blocked_other_missing_terms") or 0)
        + int(counts_by_disp.get("blocked_out_of_scope_issue") or 0)
    )

    # Representative examples keyed off the match-up outcomes (few).
    example_ids: list[tuple[str, str, str]] = []
    ex_ok = _first_item_id_with_disposition(report_items, "ok_baseline_matched")
    ex_wrong = _first_item_id_with_disposition(report_items, "ok_baseline_limit_matched_wrong_covenant")
    ex_miss = _first_item_id_with_disposition(report_items, "ok_baseline_limit_missing_despite_in_excerpt")
    ex_block = _first_item_id_with_disposition(report_items, "blocked_baseline_missing_terms") or _first_item_id_with_disposition(report_items, "blocked_other_missing_terms") or _first_item_id_with_disposition(report_items, "blocked_out_of_scope_issue")
    if ex_ok:
        example_ids.append(("Matched baseline (reference vs extraction)", ex_ok, "ok_baseline_matched"))
    if ex_wrong:
        example_ids.append(("Mismatch driven by evidence mismatch (baseline not actually in excerpt)", ex_wrong, "ok_baseline_limit_matched_wrong_covenant"))
    if ex_miss:
        example_ids.append(("Extraction gap (baseline appears in excerpt but covenant is missed)", ex_miss, "ok_baseline_limit_missing_despite_in_excerpt"))
    if ex_block:
        example_ids.append(("Blocked (excerpt missing required terms/schedule)", ex_block, "blocked"))

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
    tex.append(r"\usepackage{longtable}")
    tex.append(r"\DefineVerbatimEnvironment{Verbatim}{Verbatim}{breaklines=true,breakanywhere=true,fontsize=\small}")
    tex.append(r"\sloppy")
    tex.append(r"\setlength{\emergencystretch}{3em}")
    tex.append(r"\title{" + _tex_escape_inline("Financial Covenant Extraction Evaluation (Handcheck 30)") + r"}")
    # Keep the report PI-facing: no author attribution line.
    tex.append(r"\author{}")
    tex.append(r"\date{" + _tex_escape_inline(now) + r"}")
    tex.append(r"\begin{document}")
    tex.append(r"\maketitle")

    tex.append(r"\section*{Objective}")
    tex.append(
        _tex_escape_inline(
            "The goal is to converge on robust, general, and evaluatable financial covenant extraction logic for credit agreements. "
            "In particular, we want covenant representations that can be validated and tested as computable boolean conditions (e.g., "
            "interest coverage ratio >= X, leverage ratio <= Y, or schedule-based thresholds)."
        )
    )
    tex.append("")
    tex.append(
        _tex_escape_inline(
            "This report focuses on empirical behavior over real examples: where extraction already works, where it fails, "
            "and which failure modes are most important to close in order to reach production-grade robustness."
        )
    )

    tex.append(r"\section*{Dataset (handcheck reference + source text)}")
    tex.append(
        _tex_escape_inline(
            "We use a 30-example handcheck pack. For each example, the covenant name/limit/dates shown as reference data are taken from "
            "the handcheck dataset (i.e., the handcheck row is the reference label for that example). The source excerpt text is the "
            "evidence intended to contain that covenant."
        )
    )

    tex.append(r"\section*{Method}")
    tex.append(
        _tex_escape_inline(
            "We ran the current financial covenant extraction pipeline on the 30 examples and summarized the resulting outputs "
            "as artifacts (reference row, source excerpt, and extracted structured output)."
        )
    )

    tex.append(r"\section*{Artifacts (what each file represents)}")
    tex.append(
        _tex_escape_inline(
            "Each covenant example is stored as a small set of artifacts so a reviewer can verify match-up against the original text. "
            "The core reviewer-facing artifacts are:"
        )
    )
    tex.append(r"\begin{itemize}")
    tex.append(
        r"\item "
        + _tex_tt("baseline.json")
        + " "
        + _tex_escape_inline(
            "— the handcheck reference label for that example (name / limit / start date / end date)."
        )
    )
    tex.append(
        r"\item "
        + _tex_tt("input_snippet.txt")
        + " "
        + _tex_escape_inline(
            "— the source excerpt text used as evidence (what the extractor was given for that example)."
        )
    )
    tex.append(
        r"\item "
        + _tex_tt("covenantir_validated.json")
        + " "
        + _tex_escape_inline(
            "— our extracted structured covenant output after validation/normalization (the object we compare to the reference row)."
        )
    )
    tex.append(r"\end{itemize}")

    tex.append(
        _tex_escape_inline(
            "Some bundles also include optional debugging/reproducibility artifacts (not required for the main review):"
        )
    )
    tex.append(r"\begin{itemize}")
    tex.append(
        r"\item "
        + _tex_tt("input_snippet.json")
        + " "
        + _tex_escape_inline(
            "— the same excerpt plus metadata (anchor id, offsets, categorization), to make runs fully reproducible."
        )
    )
    tex.append(
        r"\item "
        + _tex_tt("comparison.json")
        + " "
        + _tex_escape_inline(
            "— detailed match-up diagnostics (candidate matches + scoring + final disposition) used to build the summary counts."
        )
    )
    tex.append(
        r"\item "
        + _tex_tt("result.json")
        + " "
        + _tex_escape_inline(
            "— pipeline status bookkeeping (attempts, counts, output directory pointers)."
        )
    )
    tex.append(r"\end{itemize}")

    tex.append(r"\section*{Match-up results (reference vs extraction)}")
    tex.append(
        _tex_escape_inline(
            "We focus on the core question: does the extracted output contain the same covenant (by name/type) and limit as the reference row?"
        )
    )
    tex.append("")
    tex.append(r"\begin{itemize}")
    tex.append(r"\item " + _tex_escape_inline(f"Total examples: {total}"))
    tex.append(r"\item " + _tex_escape_inline(f"Matched baseline: {matched}"))
    tex.append(r"\item " + _tex_escape_inline(f"Extraction gap (baseline present in excerpt but missed): {extraction_gap}"))
    tex.append(r"\item " + _tex_escape_inline(f"Evidence mismatch / out-of-scope excerpt: {evidence_mismatch}"))
    tex.append(r"\item " + _tex_escape_inline(f"Blocked due to incomplete excerpt (missing schedule/terms/truncation): {blocked_incomplete}"))
    tex.append(r"\end{itemize}")

    tex.append(r"\section*{Representative examples (verbatim)}")
    tex.append(
        _tex_escape_inline(
            "Below are a small number of representative examples. For each, we embed only: "
            "(1) reference data, (2) the source excerpt text, and (3) our extracted structured output."
        )
    )

    report_map: dict[str, dict[str, Any]] = {}
    for it in report_items:
        item_id = it.get("item_id")
        if isinstance(item_id, str) and item_id.strip():
            report_map[item_id.strip()] = it

    def _add_example(item_id: str, *, disp: str) -> None:
        rep = report_map.get(item_id, {})
        baseline = rep.get("baseline") if isinstance(rep.get("baseline"), dict) else {}

        # Prefer the raw baseline fields (name/limit/start/end).
        baseline_obj = {
            "name": baseline.get("name"),
            "limit": baseline.get("limit_raw") or baseline.get("limit_decimal"),
            "start_date": baseline.get("start_date_raw") or baseline.get("start_date_iso"),
            "end_date": baseline.get("end_date_raw") or baseline.get("end_date_iso"),
        }

        tex.append(r"\subsection*{" + _tex_escape_inline(f"Example: {item_id}") + r"}")

        tex.append(r"\paragraph{Reference data (handcheck row; verbatim)}")
        tex.append(_verbatim_block(_truncate(_to_pretty_json(baseline_obj))))

        # Input snippet
        snippet_path = retrieval_dir / f"{item_id}_snippets.jsonl"
        snippet_text, anchor_id = _first_snippet_text(snippet_path)
        tex.append(r"\paragraph{Source excerpt (verbatim)}")
        if anchor_id:
            tex.append(_tex_escape_inline(f"Anchor id: {anchor_id}"))
        tex.append(_verbatim_block(_truncate(snippet_text)))

        # Extraction JSON
        validated_path = covenantir_out_dir / item_id / "covenantir_validated.json"
        tex.append(r"\paragraph{Our extraction (validated JSON; compact view)}")
        if validated_path.exists():
            try:
                validated = _read_json(validated_path)
                # In most cases, filter to the best_match covenant for readability.
                # For "extraction gap" examples, show all covenants (so the missing one is visible).
                best = rep.get("best_match") if isinstance(rep.get("best_match"), dict) else {}
                best_cid = str(best.get("covenant_id") or "").strip()
                cid_filter: Optional[str] = None
                if disp != "ok_baseline_limit_missing_despite_in_excerpt" and best_cid:
                    cid_filter = best_cid
                compact = _compact_validated_output(
                    validated if isinstance(validated, dict) else {},
                    covenant_id=cid_filter,
                )
                tex.append(_verbatim_block(_truncate(_to_pretty_json(compact))))
            except Exception as e:
                tex.append(_verbatim_block(f"[error reading validated json: {e}]"))
        else:
            tex.append(_verbatim_block("[missing covenantir_validated.json]"))

    for title, item_id, disp in example_ids:
        tex.append(r"\subsection{" + _tex_escape_inline(title) + r"}")
        _add_example(item_id, disp=disp)

    tex.append(r"\section*{Reproducibility}")
    tex.append(_tex_escape_inline(f"Run id: {run_id}"))
    tex.append(r"\\")
    tex.append(_tex_escape_inline(f"CovenantIR output dir: {str(covenantir_out_dir)}"))

    tex.append(r"\end{document}")
    return "\n".join(tex).rstrip() + "\n"


def _run_latexmk(tex_path: Path) -> None:
    # Compile in the tex file directory so aux/pdf land alongside the tex.
    workdir = tex_path.parent
    cmd = [
        "latexmk",
        "-xelatex",
        "-interaction=nonstopmode",
        "-halt-on-error",
        tex_path.name,
    ]
    subprocess.run(cmd, cwd=str(workdir), check=True)

    # Keep the deliverable folder tidy: remove aux/log/etc but keep the PDF.
    subprocess.run(["latexmk", "-c", tex_path.name], cwd=str(workdir), check=True)


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
    report = _read_json(Path(args.comparison_report))
    tex = build_tex(run_id=args.run_id, covenantir_out_dir=covenantir_out_dir, comparison_report=report, repo_root=repo_root)

    out_tex = Path(args.out_tex)
    _write_text(out_tex, tex)
    print(f"[done] wrote tex: {out_tex}")

    if args.compile:
        _run_latexmk(out_tex)
        pdf_path = out_tex.with_suffix(".pdf")
        print(f"[done] compiled pdf: {pdf_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
