#!/usr/bin/env python
"""
Generate a condensed LaTeX audit report for a run, grounded in on-disk artifacts.

This is intentionally artifact-first:
  - Evidence is pulled from retrieval snippet packs and definitions context files.
  - Structured outputs are shown as JSON excerpts from llm_qa outputs.
  - Findings are tied to the run's audit_report.json (if present).

Usage:
  python scripts/generate_llm_audit_tex.py --run-id dan-v2-20260106
  python scripts/generate_llm_audit_tex.py --run-id dan-v2-20260106 --out runs/dan-v2-20260106/report/llm_audit.tex
  python scripts/generate_llm_audit_tex.py --run-id dan-v2-20260106 --items 0000720032-96-000002_2,0000950144-96-000010_7
"""

from __future__ import annotations

import argparse
import json
import re
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


MAX_BLOCK_CHARS = 2200
MAX_ISSUE_EXAMPLES_PER_KIND = 3


def _tex_escape(text: str) -> str:
    # For inline LaTeX text (NOT verbatim blocks).
    replacements = {
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
    return "".join(replacements.get(ch, ch) for ch in text)


def _truncate_block(text: str, limit: int = MAX_BLOCK_CHARS) -> str:
    if len(text) <= limit:
        return text
    head = text[:limit]
    return f"{head}\n\n[...truncated: {len(text)} chars total, showing first {limit}...]\n"


def _json_dumps(obj: Any) -> str:
    return json.dumps(obj, indent=2, ensure_ascii=False, sort_keys=False)


def _wrap_paragraphs(text: str, width: int = 100) -> str:
    # Used only for human-readable notes; do NOT apply to evidence blocks like tables.
    out_lines: list[str] = []
    for line in text.splitlines():
        if not line.strip():
            out_lines.append("")
            continue
        out_lines.extend(textwrap.fill(line, width=width).splitlines())
    return "\n".join(out_lines)


@dataclass(frozen=True)
class StructuredLoadResult:
    data: Dict[str, Any]
    duplicate_keys: List[str]


def _loads_json_detect_duplicates(text: str) -> StructuredLoadResult:
    duplicates: list[str] = []

    def hook(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        obj: dict[str, Any] = {}
        for key, value in pairs:
            if key in obj:
                duplicates.append(key)
            obj[key] = value
        return obj

    data = json.loads(text, object_pairs_hook=hook)
    if not isinstance(data, dict):
        raise ValueError("Expected top-level JSON object in structured output")
    return StructuredLoadResult(data=data, duplicate_keys=duplicates)


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def _read_json(path: Path) -> Any:
    return json.loads(_read_text(path))


def _iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _load_snippets(snippets_path: Path) -> Tuple[List[Dict[str, Any]], Dict[str, Dict[str, Any]], str]:
    records: list[dict[str, Any]] = []
    by_anchor: dict[str, dict[str, Any]] = {}
    concatenated: list[str] = []
    for rec in _iter_jsonl(snippets_path):
        records.append(rec)
        anchor_id = rec.get("anchor_id")
        if isinstance(anchor_id, str) and anchor_id and anchor_id not in by_anchor:
            by_anchor[anchor_id] = rec
        snippet = rec.get("snippet")
        if isinstance(snippet, str):
            concatenated.append(snippet)
    return records, by_anchor, "\n".join(concatenated)


def _count_token_occurrences(haystack: str, token: str) -> int:
    if not token:
        return 0
    # Case-insensitive literal search (not regex).
    return len(re.findall(re.escape(token), haystack, flags=re.IGNORECASE))


def _find_facility(structured: Dict[str, Any], facility_id: str) -> Optional[Dict[str, Any]]:
    for facility in structured.get("facilities", []) or []:
        if isinstance(facility, dict) and facility.get("facility_id") == facility_id:
            return facility
    return None


def _find_rate(structured: Dict[str, Any], rate_id: str) -> Optional[Dict[str, Any]]:
    for facility in structured.get("facilities", []) or []:
        if not isinstance(facility, dict):
            continue
        for rate in facility.get("rates", []) or []:
            if isinstance(rate, dict) and rate.get("rate_id") == rate_id:
                return rate
    return None


def _find_tier_scheme(structured: Dict[str, Any], scheme_id: str) -> Optional[Dict[str, Any]]:
    for scheme in structured.get("tier_schemes", []) or []:
        if isinstance(scheme, dict) and scheme.get("scheme_id") == scheme_id:
            return scheme
    return None


def _find_metric(structured: Dict[str, Any], name: str) -> Optional[Dict[str, Any]]:
    for metric in structured.get("metrics", []) or []:
        if isinstance(metric, dict) and metric.get("name") == name:
            return metric
    return None


def _iter_conditions(structured: Dict[str, Any]) -> Iterable[Tuple[str, str, Dict[str, Any], List[str]]]:
    """Yield (scheme_id, tier_id, condition_atom, source_refs)."""
    for scheme in structured.get("tier_schemes", []) or []:
        if not isinstance(scheme, dict):
            continue
        scheme_id = str(scheme.get("scheme_id") or "")
        for tier in scheme.get("tiers", []) or []:
            if not isinstance(tier, dict):
                continue
            tier_id = str(tier.get("tier_id") or "")
            source_refs = tier.get("source_refs") or []
            if not isinstance(source_refs, list) or not all(isinstance(a, str) for a in source_refs):
                source_refs = []
            conds = tier.get("conditions")
            if not conds:
                continue
            if not isinstance(conds, list):
                continue
            for cond in conds:
                if isinstance(cond, dict):
                    yield scheme_id, tier_id, cond, source_refs


def _missing_condition_metrics(structured: Dict[str, Any]) -> List[str]:
    defined = {
        m.get("name")
        for m in structured.get("metrics", []) or []
        if isinstance(m, dict) and isinstance(m.get("name"), str) and m.get("name").strip()
    }
    used = set()
    for _scheme_id, _tier_id, cond, _refs in _iter_conditions(structured):
        metric = cond.get("metric")
        if isinstance(metric, str) and metric.strip():
            used.add(metric.strip())
    missing = sorted(used - defined)
    return missing


def _rates_with_search_token(structured: Dict[str, Any], token: str) -> List[Dict[str, Any]]:
    hits: list[dict[str, Any]] = []
    for facility in structured.get("facilities", []) or []:
        if not isinstance(facility, dict):
            continue
        for rate in facility.get("rates", []) or []:
            if not isinstance(rate, dict):
                continue
            toks = rate.get("base_rate_search_tokens") or []
            if isinstance(toks, list) and token in toks:
                hits.append(rate)
    return hits


def _metrics_with_search_token(structured: Dict[str, Any], token: str) -> List[Dict[str, Any]]:
    hits: list[dict[str, Any]] = []
    for metric in structured.get("metrics", []) or []:
        if not isinstance(metric, dict):
            continue
        toks = metric.get("search_tokens") or []
        if isinstance(toks, list) and token in toks:
            hits.append(metric)
    return hits


def _normalize_ws(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _split_context_blocks(context_text: str) -> List[Tuple[str, str]]:
    """Split a definitions context.txt into ordered (anchor_id, block_text) pairs."""
    # Blocks look like:
    # [[A0379]]
    # ...
    blocks: list[tuple[str, str]] = []
    pattern = re.compile(r"^\[\[(A\d{4})\]\]\s*$", flags=re.MULTILINE)
    matches = list(pattern.finditer(context_text))
    for idx, match in enumerate(matches):
        anchor_id = match.group(1)
        start = match.start()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(context_text)
        blocks.append((anchor_id, context_text[start:end].strip()))
    return blocks


def _best_context_blocks_for_term(context_text: str, term: str, *, max_blocks: int = 2) -> List[Tuple[str, str]]:
    """Heuristically pick blocks that likely contain the actual definition line for term."""
    t = term.strip()
    if not t:
        return []
    t_norm = t.lower()
    scored: list[tuple[int, str, str]] = []
    for anchor_id, block in _split_context_blocks(context_text):
        b_norm = block.lower()
        if t_norm not in b_norm:
            continue
        score = 0
        score += 5  # contains term
        if " shall mean" in b_norm or " means" in b_norm:
            score += 3
        if ":" in block:
            score += 1
        # Prefer blocks where the term appears near the top (term line)
        first_pos = b_norm.find(t_norm)
        if 0 <= first_pos <= 80:
            score += 2
        scored.append((score, anchor_id, block))
    scored.sort(key=lambda x: (-x[0], x[1]))
    return [(a, b) for _s, a, b in scored[:max_blocks]]


def _extract_context_anchor_block(context_text: str, anchor_id: str) -> Optional[str]:
    # Context files use blocks like:
    # [[A0379]]
    # ...
    pattern = re.compile(rf"^\[\[{re.escape(anchor_id)}\]\]\s*$", flags=re.MULTILINE)
    match = pattern.search(context_text)
    if not match:
        return None
    start = match.end()
    next_match = re.compile(r"^\[\[A\d{4}\]\]\s*$", flags=re.MULTILINE).search(
        context_text, pos=start
    )
    end = next_match.start() if next_match else len(context_text)
    return context_text[match.start() : end].strip()


def _default_case_studies() -> List[str]:
    # Curated selection intended to cover: good, structured failures, indexing blow-up, definitions anchor issues.
    return [
        "0000720032-96-000002_2",  # good-ish end-to-end (structured + definitions)
        "0000950144-96-000010_7",  # missing metric definition + ungrounded token
        "0000950144-96-000006_2",  # runaway indexing/retrieval + empty base_rate_search_tokens
        "0001140361-23-000046_9",  # definition anchor split (Money Market Loans)
        "0001193125-23-000306_2",  # compound: duplicate metric name + missing defs + noisy definitions anchors
    ]


def _render_verbatim_block(text: str) -> str:
    return "\\begin{verbatim}\n" + text + "\n\\end{verbatim}\n"


def _render_kv_table(rows: Sequence[Tuple[str, str]]) -> str:
    out: list[str] = []
    out.append("\\begin{tabular}{p{0.44\\linewidth} p{0.52\\linewidth}}")
    out.append("\\hline")
    for k, v in rows:
        out.append(f"{_tex_escape(k)} & {_tex_escape(v)} \\\\")
    out.append("\\hline")
    out.append("\\end{tabular}")
    out.append("")
    return "\n".join(out)


def _load_item_issues(audit: Dict[str, Any], item_id: str) -> List[Dict[str, Any]]:
    return [iss for iss in audit.get("structured_issues", []) if iss.get("item_id") == item_id]


def _load_definition_issues_for_item(audit: Dict[str, Any], item_id: str) -> List[Dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for iss in audit.get("definitions_issues", []):
        detail = iss.get("detail")
        if isinstance(detail, str) and detail.startswith(item_id + "__"):
            out.append(iss)
    return out


def _find_definitions_for_item(definitions_dir: Path, item_id: str) -> List[Path]:
    return sorted(definitions_dir.glob(f"{item_id}__*__definition.json"))


def _read_definition(definitions_dir: Path, item_id: str, term: str) -> Dict[str, Any]:
    definition_path = definitions_dir / f"{item_id}__{term}__definition.json"
    return _read_json(definition_path)


def _read_definition_context(definitions_dir: Path, item_id: str, term: str) -> str:
    return _read_text(definitions_dir / f"{item_id}__{term}__context.txt")


def _write_tex(out_path: Path, tex: str) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(tex, encoding="utf-8")


def build_report(run_dir: Path, items: Sequence[str], out_path: Path) -> str:
    audit_path = run_dir / "audit_report.json"
    manifest_path = run_dir / "manifest.json"

    audit = _read_json(audit_path) if audit_path.exists() else {}
    manifest = _read_json(manifest_path) if manifest_path.exists() else {}

    qa_subdir = audit.get("qa_subdir") or manifest.get("definitions_v2_qa_subdir") or ""
    definitions_subdir = audit.get("definitions_subdir") or manifest.get("definitions_v2_output_subdir") or ""

    indexing_dir = run_dir / "indexing_v2"
    retrieval_dir = run_dir / "retrieval_v2"
    structured_dir = run_dir / "llm_qa" / qa_subdir if qa_subdir else (run_dir / "llm_qa")
    definitions_dir = run_dir / "definitions_v2" / definitions_subdir if definitions_subdir else (run_dir / "definitions_v2")

    now = manifest.get("created_at") or ""

    prompt_index = manifest.get("indexing_v2_prompt") or ""
    prompt_structured = manifest.get("structured_v2_prompt") or ""
    prompt_definitions = manifest.get("definitions_v2_prompt") or ""

    structured_issues = audit.get("structured_issues", []) or []
    definitions_issues = audit.get("definitions_issues", []) or []

    tex: list[str] = []
    tex.append(r"\documentclass[11pt]{article}")
    tex.append(r"\usepackage[margin=1in]{geometry}")
    tex.append(r"\usepackage[T1]{fontenc}")
    tex.append(r"\usepackage[utf8]{inputenc}")
    tex.append(r"\usepackage{hyperref}")
    tex.append(r"\usepackage{parskip}")
    tex.append(r"\usepackage{longtable}")
    tex.append(r"\usepackage{array}")
    tex.append(r"\title{" + _tex_escape(f"LLM Run Audit: {run_dir.name}") + r"}")
    tex.append(
        r"\author{"
        + _tex_escape("Generated by scripts/generate_llm_audit_tex.py")
        + r"}"
    )
    tex.append(r"\date{" + _tex_escape(str(now)) + r"}")
    tex.append(r"\begin{document}")
    tex.append(r"\maketitle")

    tex.append(r"\section{Scope}")
    scope_rows = [
        ("Run directory", str(run_dir)),
        ("QA subdir", qa_subdir or "(missing)"),
        ("Definitions subdir", definitions_subdir or "(missing)"),
        ("Indexing prompt", prompt_index or "(missing)"),
        ("Structured prompt", prompt_structured or "(missing)"),
        ("Definitions prompt", prompt_definitions or "(missing)"),
    ]
    tex.append(_render_kv_table(scope_rows))

    tex.append(r"\section{High-level findings (from audit\_report.json)}")
    tex.append(r"\begin{itemize}")
    tex.append(
        r"\item "
        + _tex_escape(
            f"Structured issues: {len(structured_issues)} across {len({i.get('item_id') for i in structured_issues if i.get('item_id')})} items."
        )
    )
    tex.append(
        r"\item "
        + _tex_escape(
            f"Definitions issues: {len(definitions_issues)} (heuristic) across {len({(i.get('detail') or '').split('__')[0] for i in definitions_issues})} items."
        )
    )
    idx_sum = audit.get("indexing_summary", {})
    ret_sum = audit.get("retrieval_summary", {})
    if idx_sum:
        tex.append(
            r"\item "
            + _tex_escape(
                f"Indexing totals (all items): metadata={idx_sum.get('totals', {}).get('metadata_anchors')}, "
                f"fundamental={idx_sum.get('totals', {}).get('fundamental_anchors')}, "
                f"pricing={idx_sum.get('totals', {}).get('pricing_anchors')}, "
                f"covenant={idx_sum.get('totals', {}).get('financial_covenant_anchors')}, "
                f"auto_added={idx_sum.get('totals', {}).get('auto_added_anchors')}."
            )
        )
        tex.append(
            r"\item "
            + _tex_escape(
                f"Indexing max-by-bucket: {idx_sum.get('max_by_bucket', {})}."
            )
        )
    if ret_sum:
        tex.append(
            r"\item "
            + _tex_escape(
                f"Retrieval snippet counts: min={ret_sum.get('snippet_count_min')}, "
                f"median={ret_sum.get('snippet_count_median')}, "
                f"max={ret_sum.get('snippet_count_max')} (max item={ret_sum.get('max_item')})."
            )
        )
    tex.append(r"\end{itemize}")

    tex.append(r"\section{Case studies}")
    tex.append(
        _tex_escape(
            "These were selected to cover: a good end-to-end case, structured-stage failures, an indexing/retrieval blow-up, and a definitions anchor-ref failure mode."
        )
    )
    tex.append("")
    tex.append(r"\begin{itemize}")
    for item in items:
        tex.append(r"\item " + _tex_escape(item))
    tex.append(r"\end{itemize}")

    for item_id in items:
        tex.append(r"\clearpage")
        tex.append(r"\section{" + _tex_escape(item_id) + r"}")

        indexing_path = indexing_dir / f"{item_id}_anchors.json"
        snippets_path = retrieval_dir / f"{item_id}_snippets.jsonl"
        structured_path = structured_dir / f"{item_id}.txt"

        indexing = _read_json(indexing_path) if indexing_path.exists() else {}
        selection = indexing.get("selection", {}) if isinstance(indexing, dict) else {}

        snippet_records: list[dict[str, Any]] = []
        snippets_by_anchor: dict[str, dict[str, Any]] = {}
        snippet_text_all = ""
        if snippets_path.exists():
            snippet_records, snippets_by_anchor, snippet_text_all = _load_snippets(snippets_path)

        structured = StructuredLoadResult(data={}, duplicate_keys=[])
        if structured_path.exists():
            structured_text = _read_text(structured_path)
            try:
                structured = _loads_json_detect_duplicates(structured_text)
            except Exception as exc:  # noqa: BLE001
                tex.append(_tex_escape(f"ERROR: failed to parse structured JSON: {exc}"))

        item_structured_stats = (audit.get("structured_stats") or {}).get(item_id, {})
        item_def_stats = (audit.get("definitions_stats") or {}).get(item_id, {})

        stage_rows = [
            ("Indexing: metadata anchors", str(len(selection.get("metadata_anchors", []) or []))),
            ("Indexing: fundamental anchors", str(len(selection.get("fundamental_anchors", []) or []))),
            ("Indexing: pricing anchors", str(len(selection.get("pricing_anchors", []) or []))),
            ("Indexing: covenant anchors", str(len(selection.get("financial_covenant_anchors", []) or []))),
            ("Indexing: auto-added anchors", str(len(indexing.get("auto_added_anchors", []) or []))),
            ("Retrieval: snippet count", str(len(snippet_records))),
            ("Structured: facilities", str(item_structured_stats.get("facilities", "?"))),
            ("Structured: metrics_defined", str(item_structured_stats.get("metrics_defined", "?"))),
            ("Structured: tier_condition_metrics", str(item_structured_stats.get("tier_condition_metrics", "?"))),
            ("Definitions: definitions", str(item_def_stats.get("definitions", "?"))),
            ("Definitions: null_definition_text", str(item_def_stats.get("null_definition_text", "?"))),
            ("Definitions: suspicious_anchor_refs", str(item_def_stats.get("suspicious_anchor_refs", "?"))),
        ]
        tex.append(_render_kv_table(stage_rows))

        item_structured_issues = _load_item_issues(audit, item_id)
        item_definition_issues = _load_definition_issues_for_item(audit, item_id)

        if structured.duplicate_keys:
            tex.append(
                _tex_escape(
                    f"NOTE: structured JSON had duplicate object keys: {sorted(set(structured.duplicate_keys))}."
                )
            )
            tex.append("")

        if item_structured_issues or item_definition_issues:
            tex.append(r"\subsection{Issues flagged by audit}")
            tex.append(r"\begin{itemize}")
            for iss in item_structured_issues:
                tex.append(
                    r"\item "
                    + _tex_escape(f"structured: {iss.get('kind')} :: {iss.get('detail')}")
                )
            for iss in item_definition_issues:
                tex.append(
                    r"\item "
                    + _tex_escape(f"definitions: {iss.get('kind')} :: {iss.get('detail')}")
                )
            tex.append(r"\end{itemize}")

        # Case-study specific evidence blocks (curated; keeps report compact and relevant).
        tex.append(r"\subsection{Evidence}")
        tex.append(
            _tex_escape(
                "All evidence blocks below are pulled verbatim from this run's artifacts (retrieval snippets / structured JSON / definitions context)."
            )
        )
        tex.append("")
        # Generic, issue-driven evidence blocks (scales to many items).
        if not item_structured_issues and not item_definition_issues:
            tex.append(
                _tex_escape(
                    "No issues were flagged for this item by audit_report.json; this section is intentionally minimal."
                )
            )
            tex.append("")
        else:
            # Structured issues
            for iss in item_structured_issues:
                kind = iss.get("kind")
                detail = iss.get("detail")
                if kind == "duplicate_metric_name" and isinstance(detail, str):
                    tex.append(r"\subsubsection{" + _tex_escape(f"Structured: duplicate_metric_name ({detail})") + r"}")
                    dup = detail
                    dupe_metrics = [
                        m for m in (structured.data.get("metrics", []) or [])
                        if isinstance(m, dict) and m.get("name") == dup
                    ]
                    tex.append(_tex_escape("Structured excerpt (duplicated metrics):"))
                    tex.append(_render_verbatim_block(_truncate_block(_json_dumps(dupe_metrics))))

                elif kind == "missing_metric_definitions" and isinstance(detail, list):
                    missing_metrics = [m for m in detail if isinstance(m, str)]
                    # Cross-check: compute missing from structure too, just for sanity in the report.
                    computed_missing = _missing_condition_metrics(structured.data)
                    tex.append(r"\subsubsection{" + _tex_escape("Structured: tier condition metrics missing from metrics[]") + r"}")
                    tex.append(
                        _tex_escape(
                            f"Audit-reported missing metrics: {missing_metrics}. Computed missing metrics: {computed_missing}."
                        )
                    )
                    tex.append("")
                    for metric_name in missing_metrics[:MAX_ISSUE_EXAMPLES_PER_KIND]:
                        tex.append(r"\paragraph{" + _tex_escape(metric_name) + r"}")
                        # Show any tier conditions referencing it.
                        refs: list[str] = []
                        examples: list[dict[str, Any]] = []
                        for scheme_id, tier_id, cond, source_refs in _iter_conditions(structured.data):
                            if cond.get("metric") == metric_name:
                                examples.append(
                                    {
                                        "scheme_id": scheme_id,
                                        "tier_id": tier_id,
                                        "condition": cond,
                                        "source_refs": source_refs,
                                    }
                                )
                                refs.extend(source_refs)
                                if len(examples) >= 3:
                                    break
                        tex.append(_tex_escape("Structured excerpt (tier condition atoms):"))
                        tex.append(_render_verbatim_block(_truncate_block(_json_dumps(examples))))
                        tex.append(_tex_escape("Structured excerpt (metrics list):"))
                        tex.append(
                            _render_verbatim_block(
                                _truncate_block(_json_dumps({"metrics": structured.data.get("metrics", []) or []}))
                            )
                        )
                        # Evidence snippet(s)
                        unique_refs = [a for a in dict.fromkeys(refs) if isinstance(a, str)]
                        if unique_refs:
                            tex.append(_tex_escape(f"Retrieval evidence (first cited anchors for {metric_name}):"))
                            for anchor_id in unique_refs[:2]:
                                tex.append(
                                    _render_verbatim_block(
                                        _truncate_block(snippets_by_anchor.get(anchor_id, {}).get("snippet", f"MISSING {anchor_id}"))
                                    )
                                )

                elif kind == "base_rate_search_tokens_empty" and isinstance(detail, list):
                    tex.append(r"\subsubsection{" + _tex_escape("Structured: base_rate_search_tokens_empty") + r"}")
                    for rate_id in [r for r in detail if isinstance(r, str)][:MAX_ISSUE_EXAMPLES_PER_KIND]:
                        tex.append(r"\paragraph{" + _tex_escape(rate_id) + r"}")
                        rate = _find_rate(structured.data, rate_id)
                        tex.append(_tex_escape("Structured excerpt (rate object):"))
                        tex.append(_render_verbatim_block(_truncate_block(_json_dumps(rate))))
                        if isinstance(rate, dict):
                            refs = rate.get("source_refs") or []
                            if isinstance(refs, list) and refs:
                                tex.append(_tex_escape("Retrieval evidence (first rate source_ref):"))
                                a0 = refs[0]
                                if isinstance(a0, str):
                                    tex.append(
                                        _render_verbatim_block(
                                            _truncate_block(snippets_by_anchor.get(a0, {}).get("snippet", f"MISSING {a0}"))
                                        )
                                    )

                elif kind == "base_rate_search_token_not_in_snippet_pack" and isinstance(detail, list):
                    tex.append(r"\subsubsection{" + _tex_escape("Structured: base_rate_search_token_not_in_snippet_pack") + r"}")
                    for token in [t for t in detail if isinstance(t, str)][:MAX_ISSUE_EXAMPLES_PER_KIND]:
                        count = _count_token_occurrences(snippet_text_all, token)
                        tex.append(
                            _tex_escape(
                                f"Token presence check (case-insensitive): token='{token}' occurs {count} times in the retrieval snippet pack."
                            )
                        )
                        rates = _rates_with_search_token(structured.data, token)
                        tex.append(_tex_escape("Structured excerpt (rates containing this token):"))
                        tex.append(_render_verbatim_block(_truncate_block(_json_dumps(rates[:2]))))

                elif kind == "metric_search_token_not_in_snippet_pack" and isinstance(detail, list):
                    tex.append(r"\subsubsection{" + _tex_escape("Structured: metric_search_token_not_in_snippet_pack") + r"}")
                    for token in [t for t in detail if isinstance(t, str)][:MAX_ISSUE_EXAMPLES_PER_KIND]:
                        count = _count_token_occurrences(snippet_text_all, token)
                        tex.append(
                            _tex_escape(
                                f"Token presence check (case-insensitive): token='{token}' occurs {count} times in the retrieval snippet pack."
                            )
                        )
                        metrics = _metrics_with_search_token(structured.data, token)
                        tex.append(_tex_escape("Structured excerpt (metrics containing this token):"))
                        tex.append(_render_verbatim_block(_truncate_block(_json_dumps(metrics[:2]))))

                elif kind == "metric_contract_term_not_in_snippet_pack" and isinstance(detail, list):
                    tex.append(r"\subsubsection{" + _tex_escape("Structured: metric_contract_term_not_in_snippet_pack") + r"}")
                    for term in [t for t in detail if isinstance(t, str)][:MAX_ISSUE_EXAMPLES_PER_KIND]:
                        count = _count_token_occurrences(snippet_text_all, term)
                        tex.append(
                            _tex_escape(
                                f"Contract-term presence check (case-insensitive): phrase={term!r} occurs {count} times in the retrieval snippet pack."
                            )
                        )
                        metrics = [
                            m for m in (structured.data.get("metrics", []) or [])
                            if isinstance(m, dict) and _normalize_ws(str(m.get("contract_term") or "")) == _normalize_ws(term)
                        ]
                        tex.append(_tex_escape("Structured excerpt (matching metric objects):"))
                        tex.append(_render_verbatim_block(_truncate_block(_json_dumps(metrics[:2]))))

                else:
                    # Keep unknown kinds visible (fail loud).
                    tex.append(
                        _tex_escape(
                            f"NOTE: no renderer implemented for structured issue kind={kind!r}; raw issue detail={detail!r}."
                        )
                    )
                    tex.append("")

            # Definitions issues
            for iss in item_definition_issues:
                kind = iss.get("kind")
                detail = iss.get("detail")
                if kind == "definition_anchor_refs_suspicious" and isinstance(detail, str):
                    tex.append(r"\subsubsection{" + _tex_escape("Definitions: definition_anchor_refs_suspicious") + r"}")
                    tex.append(_tex_escape(f"Flagged file: {detail}"))
                    def_path = definitions_dir / detail
                    if not def_path.exists():
                        tex.append(_tex_escape("MISSING: definitions output file not found."))
                        tex.append("")
                        continue
                    parsed = _read_json(def_path)
                    term = parsed.get("term") if isinstance(parsed, dict) else None
                    anchor_refs = parsed.get("anchor_refs") if isinstance(parsed, dict) else None
                    tex.append(_tex_escape("Definition JSON excerpt:"))
                    tex.append(_render_verbatim_block(_truncate_block(_json_dumps(parsed))))
                    # context file name is same stem with __context.txt
                    ctx_path = def_path.with_name(def_path.name.replace("__definition.json", "__context.txt"))
                    if ctx_path.exists() and isinstance(term, str) and term.strip():
                        ctx = _read_text(ctx_path)
                        # Show cited blocks
                        if isinstance(anchor_refs, list) and all(isinstance(a, str) for a in anchor_refs):
                            tex.append(_tex_escape("Context evidence (cited anchor_refs):"))
                            for a in anchor_refs[:2]:
                                block = _extract_context_anchor_block(ctx, a) or f"MISSING [[{a}]]"
                                tex.append(_render_verbatim_block(_truncate_block(block)))
                        # Show best blocks that contain the term line
                        best = _best_context_blocks_for_term(ctx, term, max_blocks=2)
                        if best:
                            tex.append(_tex_escape("Context evidence (best-matching blocks containing the term):"))
                            for a, block in best:
                                tex.append(_render_verbatim_block(_truncate_block(block)))
                    else:
                        tex.append(_tex_escape("MISSING: context file not found or term missing; cannot show anchor alignment evidence."))
                        tex.append("")
                else:
                    tex.append(
                        _tex_escape(
                            f"NOTE: no renderer implemented for definitions issue kind={kind!r}; raw issue detail={detail!r}."
                        )
                    )
                    tex.append("")

        # Definitions coverage summary for the item (kept short).
        def_files = _find_definitions_for_item(definitions_dir, item_id) if definitions_dir.exists() else []
        if def_files:
            tex.append(r"\subsection{Definitions outputs (inventory)}")
            tex.append(
                _tex_escape(
                    f"Definitions JSON files for this item: {len(def_files)}. (Showing filenames only.)"
                )
            )
            tex.append(_render_verbatim_block("\n".join(p.name for p in def_files[:30]) + ("\n..." if len(def_files) > 30 else "")))

    tex.append(r"\clearpage")
    tex.append(r"\section{Notes / next steps}")
    tex.append(r"\begin{itemize}")
    tex.append(
        r"\item "
        + _tex_escape(
            "If search_tokens are meant to be used for later retrieval, requiring them to be verbatim (or at least present in the canonical text) avoids introducing ungrounded synonyms like 'LIBOR' when the agreement uses 'London Interbank Offered Rate'."
        )
    )
    tex.append(
        r"\item "
        + _tex_escape(
            "For definitions, anchor_refs should cite an anchor that contains the term line (e.g., '\"US Base Rate\" means ...'). If anchor segmentation splits the term line from the body, consider allowing multiple anchor_refs for a single definition candidate."
        )
    )
    tex.append(
        r"\item "
        + _tex_escape(
            "Indexing: adding hard caps or explicit selection budgets per bucket would prevent runaway packs (e.g., 671 'fundamental' anchors)."
        )
    )
    tex.append(r"\end{itemize}")

    tex.append(r"\end{document}")
    return "\n".join(tex) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True)
    ap.add_argument(
        "--items",
        help="Comma-separated item_ids; default: curated set covering good+bad cases",
    )
    ap.add_argument(
        "--issues-only",
        action="store_true",
        help="If set, ignore --items and render the union of items mentioned in audit_report.json structured_issues/definitions_issues.",
    )
    ap.add_argument(
        "--out",
        help="Output .tex path (default: runs/<run-id>/report/llm_audit.tex)",
    )
    args = ap.parse_args()

    run_dir = Path("runs") / args.run_id
    if not run_dir.exists():
        raise SystemExit(f"Run directory not found: {run_dir}")

    if args.issues_only:
        audit_path = run_dir / "audit_report.json"
        if not audit_path.exists():
            raise SystemExit(f"--issues-only requires audit_report.json at {audit_path}")
        audit = json.loads(audit_path.read_text())
        structured_items = {i.get("item_id") for i in audit.get("structured_issues", []) if isinstance(i, dict)}
        def_items: set[str] = set()
        for iss in audit.get("definitions_issues", []) or []:
            if not isinstance(iss, dict):
                continue
            detail = iss.get("detail")
            if isinstance(detail, str) and "__" in detail:
                def_items.add(detail.split("__")[0])
        items = sorted({i for i in structured_items if isinstance(i, str)} | def_items)
    else:
        items = (
            [x.strip() for x in args.items.split(",") if x.strip()]
            if args.items
            else _default_case_studies()
        )

    out_path = Path(args.out) if args.out else (run_dir / "report" / "llm_audit.tex")
    tex = build_report(run_dir=run_dir, items=items, out_path=out_path)
    _write_tex(out_path, tex)
    print(f"[ok] wrote {out_path}")


if __name__ == "__main__":
    main()
