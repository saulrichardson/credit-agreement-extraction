#!/usr/bin/env python
"""
Generate a SHORT LaTeX PDF report focused on core failure modes, with wrapped code blocks.

Intent:
  - Embed relevant LLM outputs directly in the PDF (structured dg output + definitions outputs).
  - Avoid LaTeX overhang by using fvextra Verbatim with breaklines.
  - Keep the report short and pointed (a few concrete examples).

Usage:
  python scripts/generate_llm_audit_core_issues_tex.py --run-id dan-v2-20260106
  python scripts/generate_llm_audit_core_issues_tex.py --run-id dan-v2-20260106 --out runs/dan-v2-20260106/report/llm_audit_core_issues.tex
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


MAX_BLOCK_CHARS = 25000


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def _read_json(path: Path) -> Any:
    return json.loads(_read_text(path))


def _iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _truncate(text: str, limit: int = MAX_BLOCK_CHARS) -> str:
    if len(text) <= limit:
        return text
    head = text[:limit]
    return f"{head}\n\n[...truncated: {len(text)} chars total, showing first {limit}...]\n"


def _json_excerpt(obj: Any) -> str:
    return json.dumps(obj, indent=2, ensure_ascii=False)


def _count_occurrences(haystack: str, needle: str) -> int:
    if not needle:
        return 0
    return len(re.findall(re.escape(needle), haystack, flags=re.IGNORECASE))


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
    return "".join(repl.get(ch, ch) for ch in text)


def _verbatim_block(text: str) -> str:
    # fvextra's Verbatim supports wrapping.
    return (
        "\\begin{Verbatim}[breaklines=true,breakanywhere=true,fontsize=\\small]\n"
        + text
        + "\n\\end{Verbatim}\n"
    )


def _load_snippet_pack(snippets_path: Path) -> Tuple[Dict[str, Dict[str, Any]], str]:
    by_anchor: dict[str, dict[str, Any]] = {}
    all_text: list[str] = []
    for rec in _iter_jsonl(snippets_path):
        anchor_id = rec.get("anchor_id")
        if isinstance(anchor_id, str) and anchor_id and anchor_id not in by_anchor:
            by_anchor[anchor_id] = rec
        snippet = rec.get("snippet")
        if isinstance(snippet, str):
            all_text.append(snippet)
    return by_anchor, "\n".join(all_text)


def _get_rate_with_token(doc: Dict[str, Any], token: str) -> Dict[str, Any] | None:
    for facility in doc.get("facilities", []) or []:
        if not isinstance(facility, dict):
            continue
        for rate in facility.get("rates", []) or []:
            if not isinstance(rate, dict):
                continue
            toks = rate.get("base_rate_search_tokens") or []
            if isinstance(toks, list) and token in toks:
                return rate
    return None


def _get_tier_condition_atoms(doc: Dict[str, Any], metric_name: str, max_examples: int = 4) -> List[Dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for scheme in doc.get("tier_schemes", []) or []:
        if not isinstance(scheme, dict):
            continue
        scheme_id = scheme.get("scheme_id")
        for tier in scheme.get("tiers", []) or []:
            if not isinstance(tier, dict):
                continue
            tier_id = tier.get("tier_id")
            conds = tier.get("conditions")
            if not isinstance(conds, list):
                continue
            for cond in conds:
                if not isinstance(cond, dict):
                    continue
                if cond.get("metric") == metric_name:
                    out.append(
                        {
                            "scheme_id": scheme_id,
                            "tier_id": tier_id,
                            "condition": cond,
                            "source_refs": tier.get("source_refs"),
                        }
                    )
                    if len(out) >= max_examples:
                        return out
    return out


def _metric_names(doc: Dict[str, Any]) -> List[str]:
    names: list[str] = []
    for m in doc.get("metrics", []) or []:
        if isinstance(m, dict) and isinstance(m.get("name"), str):
            names.append(m["name"])
    return names


def _extract_context_block(context_text: str, anchor_id: str) -> str | None:
    # Context blocks are formatted as:
    #   [[A0123]]
    #   <text...>
    #
    # Be careful to match literal square brackets, not backslash-escaped ones.
    pattern = re.compile(rf"^\[\[{re.escape(anchor_id)}\]\]\s*$", flags=re.MULTILINE)
    m = pattern.search(context_text)
    if not m:
        return None
    start = m.start()
    next_m = re.compile(r"^\[\[A\d{4}\]\]\s*$", flags=re.MULTILINE).search(context_text, pos=m.end())
    end = next_m.start() if next_m else len(context_text)
    return context_text[start:end].strip()


def build_core_report(run_dir: Path) -> str:
    audit = _read_json(run_dir / "audit_report.json") if (run_dir / "audit_report.json").exists() else {}
    manifest = _read_json(run_dir / "manifest.json") if (run_dir / "manifest.json").exists() else {}

    qa_subdir = audit.get("qa_subdir") or manifest.get("definitions_v2_qa_subdir") or ""
    defs_subdir = audit.get("definitions_subdir") or manifest.get("definitions_v2_output_subdir") or ""

    structured_dir = run_dir / "llm_qa" / qa_subdir
    retrieval_dir = run_dir / "retrieval_v2"
    definitions_dir = run_dir / "definitions_v2" / defs_subdir

    # Pick a few concrete, high-signal examples.
    ex1_item = "0000950144-96-000010_7"  # missing metric in metrics[] + LIBOR token injection
    ex2_item = "0001193125-23-000306_2"  # definitions anchor mis-citation
    ex3_item = "0000950144-96-000006_2"  # runaway indexing + empty base_rate_search_tokens

    tex: list[str] = []
    tex.append(r"\documentclass[11pt]{article}")
    tex.append(r"\usepackage[margin=1in]{geometry}")
    tex.append(r"\usepackage[T1]{fontenc}")
    tex.append(r"\usepackage[utf8]{inputenc}")
    tex.append(r"\usepackage{hyperref}")
    tex.append(r"\usepackage{parskip}")
    tex.append(r"\usepackage{xcolor}")
    tex.append(r"\usepackage{fvextra}")
    tex.append(r"\DefineVerbatimEnvironment{Verbatim}{Verbatim}{breaklines=true,breakanywhere=true,fontsize=\small}")
    tex.append(r"\sloppy")
    tex.append(r"\setlength{\emergencystretch}{3em}")
    tex.append(r"\title{" + _tex_escape_inline(f"Core Issues Audit: {run_dir.name}") + r"}")
    tex.append(r"\author{" + _tex_escape_inline("Generated by scripts/generate_llm_audit_core_issues_tex.py") + r"}")
    tex.append(r"\date{}")
    tex.append(r"\begin{document}")
    tex.append(r"\maketitle")

    # Summary
    structured_issues = audit.get("structured_issues", []) or []
    definitions_issues = audit.get("definitions_issues", []) or []
    tex.append(r"\section*{Summary}")
    tex.append(
        _tex_escape_inline(
            "This report focuses on a few core failure modes, with verbatim excerpts from the run artifacts."
        )
    )
    tex.append("")
    tex.append(r"\begin{itemize}")
    tex.append(r"\item " + _tex_escape_inline("Structured output can be internally inconsistent (tier conditions reference a metric that is not declared in metrics[])."))
    tex.append(r"\item " + _tex_escape_inline("Verbatim-required helper fields can include ungrounded tokens (e.g., 'LIBOR' not present in evidence pack)."))
    tex.append(r"\item " + _tex_escape_inline("Definitions stage can cite the wrong anchor_refs even when definition text is present."))
    tex.append(r"\item " + _tex_escape_inline("Indexing/retrieval can balloon context (hundreds of anchors/snippets), and structured output can still omit required base-rate helpers."))
    tex.append(r"\end{itemize}")

    # Example 1
    tex.append(r"\section{Example 1: Structured output inconsistency + ungrounded token}")
    tex.append(r"\textbf{Item:} " + _tex_escape_inline(ex1_item))

    dg_path = structured_dir / f"{ex1_item}.txt"
    snips_path = retrieval_dir / f"{ex1_item}_snippets.jsonl"
    dg_raw = _read_text(dg_path).strip()
    dg = json.loads(dg_raw)
    by_anchor, all_snip_text = _load_snippet_pack(snips_path)

    # (A) Missing metric declaration for tier conditions
    tex.append(r"\subsection*{A) Tier conditions reference an undeclared metric}")
    tex.append(_tex_escape_inline("DG output (verbatim):"))
    tex.append(_verbatim_block(_truncate(dg_raw)))
    tex.append(_tex_escape_inline("Evidence snippet (A0387 pricing table):"))
    tex.append(_verbatim_block(_truncate(str(by_anchor.get("A0387", {}).get("snippet", "MISSING A0387")))))

    # (B) Ungrounded LIBOR token
    tex.append(r"\subsection*{B) base\_rate\_search\_tokens include 'LIBOR' even though evidence pack has 0 matches}")
    libor_count = _count_occurrences(all_snip_text, "LIBOR")
    tex.append(_tex_escape_inline(f"Token count in snippet pack (case-insensitive): LIBOR = {libor_count}"))
    rate_with_libor = _get_rate_with_token(dg, "LIBOR")
    tex.append(_tex_escape_inline("DG output excerpt (a rate object containing LIBOR in base_rate_search_tokens):"))
    tex.append(_verbatim_block(_truncate(_json_excerpt(rate_with_libor))))
    tex.append(_tex_escape_inline("Evidence snippet (A0401 contains 'London Interbank Offered Rate' text):"))
    tex.append(_verbatim_block(_truncate(str(by_anchor.get("A0401", {}).get("snippet", "MISSING A0401")))))

    # Example 2
    tex.append(r"\section{Example 2: Definitions cite wrong anchor\_refs}")
    tex.append(r"\textbf{Item:} " + _tex_escape_inline(ex2_item))
    def_txt_path = definitions_dir / f"{ex2_item}__US_Base_Rate__definition.txt"
    ctx_path = definitions_dir / f"{ex2_item}__US_Base_Rate__context.txt"
    def_raw = _read_text(def_txt_path).strip()
    ctx = _read_text(ctx_path)

    tex.append(_tex_escape_inline("Definitions output (US Base Rate):"))
    tex.append(_verbatim_block(_truncate(def_raw)))
    tex.append(_tex_escape_inline("Context block that contains the actual definition line (A0827):"))
    tex.append(_verbatim_block(_truncate(_extract_context_block(ctx, "A0827") or "MISSING [[A0827]]")))
    tex.append(_tex_escape_inline("Context block that the model cited instead (A1143):"))
    tex.append(_verbatim_block(_truncate(_extract_context_block(ctx, "A1143") or "MISSING [[A1143]]")))

    # Example 3
    tex.append(r"\section{Example 3: Indexing/retrieval ballooning + base\_rate helpers still missing}")
    tex.append(r"\textbf{Item:} " + _tex_escape_inline(ex3_item))
    snips3_path = retrieval_dir / f"{ex3_item}_snippets.jsonl"
    snip_count = sum(1 for _ in _iter_jsonl(snips3_path))
    dg3_path = structured_dir / f"{ex3_item}.txt"
    dg3_raw = _read_text(dg3_path).strip()
    tex.append(_tex_escape_inline(f"Retrieval snippets: {snip_count} (v2)."))
    tex.append(_tex_escape_inline("DG output (verbatim):"))
    tex.append(_verbatim_block(_truncate(dg3_raw)))

    tex.append(r"\end{document}")
    return "\n".join(tex) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True)
    ap.add_argument(
        "--out",
        help="Output .tex path (default: runs/<run-id>/report/llm_audit_core_issues.tex)",
    )
    args = ap.parse_args()

    run_dir = Path("runs") / args.run_id
    if not run_dir.exists():
        raise SystemExit(f"Run directory not found: {run_dir}")

    out_path = Path(args.out) if args.out else (run_dir / "report" / "llm_audit_core_issues.tex")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tex = build_core_report(run_dir)
    out_path.write_text(tex, encoding="utf-8")
    print(f"[ok] wrote {out_path}")


if __name__ == "__main__":
    main()
