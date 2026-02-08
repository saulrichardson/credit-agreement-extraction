#!/usr/bin/env python
"""
Render a self-contained HTML viewer for the pricing "definitions → Compustat" workflow,
plus optional third-pass prompt experiments.

Goal: open a single HTML file in a browser and:
  - navigate by agreement (item_id)
  - switch between multiple stage outputs (DG structured output, definitions compiler, blocking terms, third-pass outputs, etc.)
  - click any anchor id (A####) inside an output to jump/highlight the cited support in the FULL agreement text

This script makes NO LLM calls. It only reads run artifacts:
  - runs/<run_id>/normalized/<item_id>/canonical_annotated.txt (preferred) or canonical.txt + anchors.tsv (fallback)
  - runs/<run_id>/indexing_v2/<item_id>_anchors.json
  - runs/<run_id>/llm_qa/<qa_subdir>/<item_id>.txt
  - runs/<run_id>/definitions_compiler_v1/<subdir>/...
  - runs/<run_id>/blocking_terms_compiler_v1/<subdir>/...
  - (optional) a third-pass runner output dir produced by scripts/pricing_definitions_third_pass_runner.py

Default output:
  - If --third-pass-out-dir is provided: <third-pass-out-dir>/workflow_viewer.html
  - Else: runs/<run_id>/pricing_workflow_viewer.html
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from pipeline.core.anchors import load_anchor_catalog  # noqa: E402
from pipeline.core.config import Paths  # noqa: E402


ANCHOR_ID_RE = re.compile(r"^A\d{4,}$")
ANNOTATED_ANCHOR_LINE_RE = re.compile(r"^\[\[(A\d{4,})\]\]\s*$")
ANCHOR_FINDER_RE = re.compile(r"(A\d{4,})")


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def _read_text_or_none(path: Path) -> Optional[str]:
    try:
        return _read_text(path)
    except FileNotFoundError:
        return None


def _read_json(path: Path) -> Any:
    return json.loads(_read_text(path))


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _safe_slug(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "_", (s or "").strip()).strip("_") or "term"


def _parse_canonical_annotated(text: str) -> List[Dict[str, Any]]:
    """Parse canonical_annotated.txt into ordered anchor blocks."""

    blocks: List[Dict[str, Any]] = []
    current_id: Optional[str] = None
    current_lines: List[str] = []

    for raw_line in text.splitlines():
        m = ANNOTATED_ANCHOR_LINE_RE.match(raw_line.strip())
        if m:
            if current_id is not None:
                blocks.append({"anchor_id": current_id, "text": "\n".join(current_lines).rstrip("\n")})
            current_id = m.group(1)
            current_lines = []
            continue
        # Preserve original line breaks.
        if current_id is None:
            current_id = "__PREFACE__"
            current_lines = []
        current_lines.append(raw_line)

    if current_id is not None:
        blocks.append({"anchor_id": current_id, "text": "\n".join(current_lines).rstrip("\n")})

    if blocks and blocks[0].get("anchor_id") == "__PREFACE__" and not str(blocks[0].get("text") or "").strip():
        blocks = blocks[1:]

    cleaned: List[Dict[str, Any]] = []
    for b in blocks:
        aid = str(b.get("anchor_id") or "").strip()
        txt = str(b.get("text") or "")
        if aid != "__PREFACE__" and not ANCHOR_ID_RE.fullmatch(aid):
            raise ValueError(f"Invalid anchor marker in canonical_annotated parse: {aid!r}")
        cleaned.append({"anchor_id": aid, "text": txt})
    return cleaned


def _anchor_text_from_catalog(
    canonical_text: str, catalog: Mapping[str, Mapping[str, Any]], anchor_id: str
) -> Optional[str]:
    info = catalog.get(anchor_id)
    if not info:
        return None
    start = info.get("start")
    end = info.get("end")
    if not isinstance(start, int) or not isinstance(end, int):
        return None
    a = max(0, start)
    b = min(len(canonical_text), end)
    if b <= a:
        return None
    txt = canonical_text[a:b].rstrip()
    return txt if txt.strip() else None


def _fallback_blocks_from_canonical(paths: Paths, item_id: str) -> List[Dict[str, Any]]:
    canonical_path = paths.run_dir / "normalized" / item_id / "canonical.txt"
    if not canonical_path.exists():
        raise FileNotFoundError(f"Missing canonical.txt for {item_id}: expected {canonical_path}")
    canonical_text = _read_text(canonical_path)

    catalog = load_anchor_catalog(paths, item_id)
    ordered = sorted(catalog.values(), key=lambda a: int(a["order"]))
    blocks: List[Dict[str, Any]] = []
    for info in ordered:
        aid = str(info.get("anchor_id") or "")
        if not ANCHOR_ID_RE.fullmatch(aid):
            continue
        txt = _anchor_text_from_catalog(canonical_text, catalog, aid) or ""
        blocks.append({"anchor_id": aid, "text": txt})
    return blocks


def _load_document_blocks(paths: Paths, item_id: str) -> Tuple[str, List[Dict[str, Any]]]:
    annotated_path = paths.run_dir / "normalized" / item_id / "canonical_annotated.txt"
    if annotated_path.exists():
        return (str(annotated_path), _parse_canonical_annotated(_read_text(annotated_path)))
    return (str(paths.run_dir / "normalized" / item_id / "canonical.txt"), _fallback_blocks_from_canonical(paths, item_id))


def _extract_anchor_ids_from_json(doc: Any) -> Set[str]:
    """Find anchor IDs anywhere in a JSON-like object.

    This is intentionally permissive: any string leaf matching A#### is treated as an anchor id.
    """

    anchors: Set[str] = set()

    def _walk(x: Any) -> None:
        if isinstance(x, dict):
            for v in x.values():
                _walk(v)
            return
        if isinstance(x, list):
            for v in x:
                _walk(v)
            return
        if isinstance(x, str):
            s = x.strip()
            if ANCHOR_ID_RE.fullmatch(s):
                anchors.add(s)
                return
            for m in ANCHOR_FINDER_RE.finditer(s):
                aid = m.group(1)
                if ANCHOR_ID_RE.fullmatch(aid):
                    anchors.add(aid)

    _walk(doc)
    return anchors


def _extract_anchor_ids_from_text(text: str) -> Set[str]:
    return {m.group(1) for m in ANCHOR_FINDER_RE.finditer(text or "") if ANCHOR_ID_RE.fullmatch(m.group(1))}


def _load_manifest(paths: Paths) -> Dict[str, Any]:
    mp = paths.manifest_path
    if not mp.exists():
        return {}
    doc = _read_json(mp)
    if isinstance(doc, dict):
        return doc
    return {}


def _extract_metric_names_from_dg(doc: Any) -> List[str]:
    if not isinstance(doc, dict):
        return []
    metrics = doc.get("metrics")
    if not isinstance(metrics, list):
        return []
    out: List[str] = []
    seen: Set[str] = set()
    for m in metrics:
        if not isinstance(m, dict):
            continue
        name = m.get("name")
        if not isinstance(name, str) or not name.strip():
            continue
        n = name.strip()
        if n in seen:
            continue
        seen.add(n)
        out.append(n)
    return out


def _extract_blocking_terms_from_aggregate(doc: Any) -> List[str]:
    if not isinstance(doc, dict):
        return []
    ud = doc.get("unresolved_dependencies")
    if not isinstance(ud, list):
        return []
    out: List[str] = []
    seen: Set[str] = set()
    for row in ud:
        if not isinstance(row, dict):
            continue
        term = row.get("term")
        if not isinstance(term, str) or not term.strip():
            continue
        t = term.strip()
        if t in seen:
            continue
        seen.add(t)
        out.append(t)
    return out


def _load_outputs_for_item(
    *,
    paths: Paths,
    item_id: str,
    doc_anchor_ids: Set[str],
    qa_subdir: str | None,
    definitions_subdir: str | None,
    blocking_terms_subdir: str | None,
    third_pass_out_dir: Path | None,
    third_pass_strategies: Sequence[str],
) -> List[Dict[str, Any]]:
    outputs: List[Dict[str, Any]] = []

    def _add_json(*, stage: str, label: str, path: Path, output_id: str) -> None:
        if not path.exists():
            return
        try:
            doc = _read_json(path)
        except Exception as exc:
            outputs.append(
                {
                    "output_id": output_id,
                    "stage": stage,
                    "label": label,
                    "kind": "text",
                    "path": str(path),
                    "text": f"(error reading JSON) {type(exc).__name__}: {exc}\n\nRAW:\n{_read_text(path)}",
                    "cited_anchor_ids": [],
                    "unknown_anchor_ids": [],
                }
            )
            return
        cited = sorted(_extract_anchor_ids_from_json(doc), key=lambda a: int(a[1:]))
        unknown = sorted([a for a in cited if a not in doc_anchor_ids], key=lambda a: int(a[1:]))
        outputs.append(
            {
                "output_id": output_id,
                "stage": stage,
                "label": label,
                "kind": "json",
                "path": str(path),
                "json": doc,
                "cited_anchor_ids": cited,
                "unknown_anchor_ids": unknown,
            }
        )

    def _add_text(*, stage: str, label: str, path: Path, output_id: str) -> None:
        if not path.exists():
            return
        text = _read_text(path)
        cited = sorted(_extract_anchor_ids_from_text(text), key=lambda a: int(a[1:]))
        unknown = sorted([a for a in cited if a not in doc_anchor_ids], key=lambda a: int(a[1:]))
        outputs.append(
            {
                "output_id": output_id,
                "stage": stage,
                "label": label,
                "kind": "text",
                "path": str(path),
                "text": text,
                "cited_anchor_ids": cited,
                "unknown_anchor_ids": unknown,
            }
        )

    # Stage: indexing_v2 selection
    _add_json(
        stage="indexing_v2",
        label="indexing_v2 selection",
        path=paths.run_dir / "indexing_v2" / f"{item_id}_anchors.json",
        output_id="indexing_v2_selection",
    )

    # Stage: DG structured output
    dg_doc: Any = None
    if qa_subdir:
        dg_path = paths.run_dir / "llm_qa" / qa_subdir / f"{item_id}.txt"
        if dg_path.exists():
            try:
                dg_doc = _read_json(dg_path)
            except Exception:
                dg_doc = None
            _add_json(stage="dg_structured_v2", label=f"DG output ({qa_subdir})", path=dg_path, output_id="dg_output")

    metric_names = _extract_metric_names_from_dg(dg_doc)

    # Stage: definitions compiler v1
    defs_agg: Any = None
    if definitions_subdir:
        defs_dir = paths.run_dir / "definitions_compiler_v1" / definitions_subdir
        _add_json(
            stage="definitions_compiler_v1",
            label=f"definitions_compiler_v1 aggregate ({definitions_subdir})",
            path=defs_dir / f"{item_id}__compiled.json",
            output_id="definitions_compiler_v1_aggregate",
        )
        agg_path = defs_dir / f"{item_id}__compiled.json"
        if agg_path.exists():
            try:
                defs_agg = _read_json(agg_path)
            except Exception:
                defs_agg = None

        for metric in metric_names:
            slug = _safe_slug(metric)
            prefix = f"{item_id}__{slug}__"
            # Curate a stable, useful set; include pass1/pass2 variants when present.
            for name, kind in (
                ("contexts.pass1.txt", "text"),
                ("contexts.pass2.txt", "text"),
                ("contexts.txt", "text"),
                ("definition_finder.json", "json"),
                ("definition_finder.raw.txt", "text"),
                ("compile.pass1.raw.txt", "text"),
                ("compile.pass2.raw.txt", "text"),
                ("compile.raw.txt", "text"),
                ("compiled.pass1.json", "json"),
                ("compiled.pass2.json", "json"),
                ("compiled.json", "json"),
            ):
                p = defs_dir / f"{prefix}{name}"
                oid = f"defs::{slug}::{name}"
                lbl = f"defs[{metric}]::{name}"
                if kind == "json":
                    _add_json(stage="definitions_compiler_v1", label=lbl, path=p, output_id=oid)
                else:
                    _add_text(stage="definitions_compiler_v1", label=lbl, path=p, output_id=oid)

            # Also include any repair attempts that exist (attempt2/attempt3 etc.).
            for raw_path in sorted(defs_dir.glob(f"{prefix}compile.raw.attempt*.txt")):
                _add_text(
                    stage="definitions_compiler_v1",
                    label=f"defs[{metric}]::{raw_path.name.split(prefix, 1)[-1]}",
                    path=raw_path,
                    output_id=f"defs::{slug}::{raw_path.name.split(prefix, 1)[-1]}",
                )

    # Stage: blocking terms compiler v1
    if blocking_terms_subdir:
        bdir = paths.run_dir / "blocking_terms_compiler_v1" / blocking_terms_subdir
        _add_json(
            stage="blocking_terms_compiler_v1",
            label=f"blocking_terms_compiler_v1 aggregate ({blocking_terms_subdir})",
            path=bdir / f"{item_id}__compiled.json",
            output_id="blocking_terms_compiler_v1_aggregate",
        )

        terms = _extract_blocking_terms_from_aggregate(defs_agg)
        for term in terms:
            slug = _safe_slug(term)
            prefix = f"{item_id}__{slug}__"
            for name, kind in (
                ("contexts.pass1.txt", "text"),
                ("contexts.pass2.txt", "text"),
                ("contexts.txt", "text"),
                ("compile.pass1.raw.txt", "text"),
                ("compile.pass2.raw.txt", "text"),
                ("compile.raw.txt", "text"),
                ("compiled.pass1.json", "json"),
                ("compiled.pass2.json", "json"),
                ("compiled.json", "json"),
            ):
                p = bdir / f"{prefix}{name}"
                oid = f"bterms::{slug}::{name}"
                lbl = f"bterms[{term}]::{name}"
                if kind == "json":
                    _add_json(stage="blocking_terms_compiler_v1", label=lbl, path=p, output_id=oid)
                else:
                    _add_text(stage="blocking_terms_compiler_v1", label=lbl, path=p, output_id=oid)

            for raw_path in sorted(bdir.glob(f"{prefix}compile.raw.attempt*.txt")):
                _add_text(
                    stage="blocking_terms_compiler_v1",
                    label=f"bterms[{term}]::{raw_path.name.split(prefix, 1)[-1]}",
                    path=raw_path,
                    output_id=f"bterms::{slug}::{raw_path.name.split(prefix, 1)[-1]}",
                )

    # Stage: third-pass prompt experiment outputs (from pricing_definitions_third_pass_runner)
    if third_pass_out_dir and third_pass_out_dir.exists():
        # Include the prompt templates copied to the output dir (for reproducibility).
        for p in sorted(third_pass_out_dir.glob("prompt*.txt")):
            _add_text(stage="third_pass", label=f"third_pass::{p.name}", path=p, output_id=f"third_pass_prompt::{p.name}")

        for strategy in third_pass_strategies:
            item_out = third_pass_out_dir / strategy / item_id
            if not item_out.exists():
                continue
            # Curated stable artifacts.
            for fname, kind in (
                ("metrics_list.json", "json"),
                ("definition_chunks.txt", "text"),
                ("prompt_rendered.txt", "text"),
                ("llm_output.txt", "json"),
                ("validation.json", "json"),
                ("meta.json", "json"),
                ("error.txt", "text"),
            ):
                p = item_out / fname
                oid = f"third_pass::{strategy}::{fname}"
                lbl = f"third_pass[{strategy}]::{fname}"
                if kind == "json":
                    _add_json(stage="third_pass", label=lbl, path=p, output_id=oid)
                else:
                    _add_text(stage="third_pass", label=lbl, path=p, output_id=oid)

    # Stable sort: stage then label.
    stage_rank = {
        "third_pass": 10,
        "blocking_terms_compiler_v1": 20,
        "definitions_compiler_v1": 30,
        "dg_structured_v2": 40,
        "indexing_v2": 50,
    }
    outputs_sorted = sorted(
        outputs,
        key=lambda o: (
            stage_rank.get(str(o.get("stage")), 999),
            str(o.get("stage")),
            str(o.get("label")),
        ),
    )
    return outputs_sorted


def _default_output_id(outputs: Sequence[Dict[str, Any]]) -> Optional[str]:
    """Pick a sensible default output to show first."""

    # Prefer third-pass LLM output; else definitions aggregate; else DG output; else indexing.
    for pred in (
        lambda o: str(o.get("output_id", "")).startswith("third_pass::") and str(o.get("label", "")).endswith("llm_output.txt"),
        lambda o: o.get("output_id") == "definitions_compiler_v1_aggregate",
        lambda o: o.get("output_id") == "dg_output",
        lambda o: o.get("output_id") == "indexing_v2_selection",
    ):
        for o in outputs:
            if pred(o):
                return str(o.get("output_id"))
    return str(outputs[0].get("output_id")) if outputs else None


def _html_template(*, title: str, embedded_json: str) -> str:
    return f"""<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>{title}</title>
    <style>
      :root {{
        --bg: #0b1020;
        --panel: #101735;
        --panel2: #0f1630;
        --border: rgba(255,255,255,0.10);
        --text: rgba(255,255,255,0.92);
        --muted: rgba(255,255,255,0.70);
        --accent: #6ee7ff;
        --accent2: #a78bfa;
        --warn: #fbbf24;
        --bad: #fb7185;
        --good: #34d399;
        --mono: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace;
        --sans: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial;
      }}

      html, body {{
        margin: 0;
        padding: 0;
        height: 100%;
        background: var(--bg);
        color: var(--text);
        font-family: var(--sans);
      }}

      .app {{
        display: grid;
        grid-template-columns: 300px 1fr 1fr;
        grid-template-rows: auto 1fr;
        height: 100vh;
      }}

      header {{
        grid-column: 1 / -1;
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 16px;
        padding: 12px 16px;
        border-bottom: 1px solid var(--border);
        background: linear-gradient(180deg, rgba(255,255,255,0.04), rgba(255,255,255,0.0));
      }}
      .title {{
        display: flex;
        flex-direction: column;
        gap: 2px;
      }}
      .h1 {{
        font-weight: 700;
        font-size: 14px;
        letter-spacing: 0.2px;
      }}
      .sub {{
        font-family: var(--mono);
        font-size: 11px;
        color: var(--muted);
      }}

      .controls {{
        display: flex;
        align-items: center;
        gap: 10px;
      }}

      .chip {{
        display: inline-flex;
        align-items: center;
        gap: 8px;
        padding: 6px 10px;
        border-radius: 999px;
        background: rgba(255,255,255,0.05);
        border: 1px solid var(--border);
        color: var(--text);
        font-size: 12px;
        cursor: pointer;
        user-select: none;
      }}
      .chip.active {{
        border-color: rgba(110,231,255,0.35);
        box-shadow: 0 0 0 1px rgba(110,231,255,0.15) inset;
        color: var(--accent);
      }}

      aside {{
        border-right: 1px solid var(--border);
        background: rgba(255,255,255,0.02);
        overflow: hidden;
      }}
      .aside-inner {{
        display: flex;
        flex-direction: column;
        height: 100%;
      }}
      .search {{
        padding: 10px 10px 6px 10px;
      }}
      input {{
        width: 100%;
        padding: 8px 10px;
        border-radius: 10px;
        border: 1px solid var(--border);
        background: rgba(255,255,255,0.03);
        color: var(--text);
        font-family: var(--mono);
        font-size: 12px;
        outline: none;
      }}
      input:focus {{
        border-color: rgba(110,231,255,0.35);
        box-shadow: 0 0 0 2px rgba(110,231,255,0.12);
      }}

      .item-list {{
        padding: 6px 8px 10px 8px;
        overflow: auto;
        flex: 1;
      }}
      .item {{
        padding: 10px 10px;
        border-radius: 12px;
        border: 1px solid transparent;
        cursor: pointer;
        margin-bottom: 8px;
        background: rgba(255,255,255,0.02);
      }}
      .item:hover {{
        border-color: rgba(255,255,255,0.10);
      }}
      .item.active {{
        border-color: rgba(167,139,250,0.35);
        box-shadow: 0 0 0 1px rgba(167,139,250,0.15) inset;
      }}
      .id {{
        font-family: var(--mono);
        font-size: 12px;
        color: var(--text);
      }}
      .meta {{
        margin-top: 6px;
        display: flex;
        flex-wrap: wrap;
        gap: 6px;
        font-size: 11px;
        color: var(--muted);
        font-family: var(--mono);
      }}
      .badge {{
        padding: 2px 8px;
        border-radius: 999px;
        border: 1px solid var(--border);
        background: rgba(255,255,255,0.03);
      }}
      .badge.good {{
        border-color: rgba(52, 211, 153, 0.35);
        color: var(--good);
      }}
      .badge.warn {{
        border-color: rgba(251, 191, 36, 0.35);
        color: var(--warn);
      }}
      .badge.bad {{
        border-color: rgba(251, 113, 133, 0.35);
        color: var(--bad);
      }}

      .panel {{
        overflow: hidden;
        background: rgba(255,255,255,0.015);
      }}
      .panel-inner {{
        height: 100%;
        display: flex;
        flex-direction: column;
        padding: 12px 12px 12px 12px;
      }}

      .section-title {{
        font-size: 12px;
        color: var(--muted);
        font-family: var(--mono);
        margin-bottom: 8px;
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 10px;
      }}

      .output-controls {{
        display: flex;
        align-items: center;
        gap: 10px;
        margin-bottom: 8px;
      }}
      .output-controls select {{
        flex: 1;
        min-width: 220px;
        padding: 8px 10px;
        border-radius: 10px;
        border: 1px solid var(--border);
        background: rgba(255,255,255,0.03);
        color: var(--text);
        font-family: var(--mono);
        font-size: 12px;
        outline: none;
      }}
      .pathline {{
        font-family: var(--mono);
        font-size: 11px;
        color: var(--muted);
        margin: 4px 0 8px 0;
        overflow: hidden;
        text-overflow: ellipsis;
        white-space: nowrap;
      }}

      pre {{
        margin: 0;
        padding: 12px 12px;
        border: 1px solid var(--border);
        border-radius: 12px;
        background: rgba(0,0,0,0.22);
        overflow: auto;
        font-family: var(--mono);
        font-size: 12px;
        line-height: 1.35;
        white-space: pre-wrap;
      }}

      .anchor-ref {{
        color: var(--accent);
        cursor: pointer;
        text-decoration: underline;
        text-decoration-color: rgba(110,231,255,0.35);
      }}

      .doc-controls {{
        display: flex;
        align-items: center;
        gap: 10px;
        margin-bottom: 8px;
      }}
      .doc-count {{
        font-family: var(--mono);
        font-size: 11px;
        color: var(--muted);
      }}
      #doc-view {{
        overflow: auto;
        padding-right: 2px;
      }}
      .anchor-block {{
        border: 1px solid var(--border);
        border-radius: 12px;
        overflow: hidden;
        background: rgba(0,0,0,0.16);
        margin-bottom: 10px;
      }}
      .anchor-header {{
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 10px;
        padding: 8px 10px;
        background: rgba(255,255,255,0.03);
        border-bottom: 1px solid var(--border);
        font-family: var(--mono);
        font-size: 12px;
        color: var(--muted);
      }}
      .aid {{
        color: var(--text);
        cursor: pointer;
      }}
      .tags {{
        font-size: 11px;
        color: var(--muted);
      }}

      .highlight {{
        outline: 2px solid rgba(110,231,255,0.35);
        box-shadow: 0 0 0 2px rgba(110,231,255,0.12);
      }}
    </style>
  </head>
  <body>
    <div class="app" id="app">
      <header>
        <div class="title">
          <div class="h1" id="viewer-title">Pricing Workflow Viewer</div>
          <div class="sub" id="run-id"></div>
        </div>
        <div class="controls">
          <div class="chip" id="toggle-only-cited">Only cited anchors</div>
        </div>
      </header>

      <aside>
        <div class="aside-inner">
          <div class="search">
            <input id="item-filter" placeholder="Filter item_id…" />
          </div>
          <div class="item-list" id="item-list"></div>
        </div>
      </aside>

      <main class="panel" id="output-panel">
        <div class="panel-inner">
          <div class="section-title" id="output-title">Output</div>
          <div class="output-controls">
            <select id="output-select"></select>
          </div>
          <div class="pathline" id="output-path"></div>
          <pre id="output-view"></pre>
        </div>
      </main>

      <main class="panel" id="doc-panel">
        <div class="panel-inner">
          <div class="section-title" id="doc-title">Agreement text</div>
          <div class="doc-controls">
            <div class="search" style="flex: 1; min-width: 240px;">
              <input id="doc-filter" placeholder="Filter (anchor id like A0123 or text substring)…" />
            </div>
            <div class="doc-count" id="doc-count"></div>
          </div>
          <div id="doc-view"></div>
        </div>
      </main>
    </div>

    <script type="application/json" id="run-data">
{embedded_json}
    </script>
    <script>
      function escapeHtml(s) {{
        return (s || '')
          .replaceAll('&', '&amp;')
          .replaceAll('<', '&lt;')
          .replaceAll('>', '&gt;')
          .replaceAll('\"', '&quot;')
          .replaceAll(\"'\", '&#39;');
      }}

      function selectDocAnchor(anchorId) {{
        const el = document.getElementById('doc-' + anchorId);
        if (!el) return;
        el.scrollIntoView({{ behavior: 'smooth', block: 'start' }});
        el.classList.add('highlight');
        window.setTimeout(() => el.classList.remove('highlight'), 1200);
      }}

      function renderTextWithAnchorLinks(text) {{
        const safe = escapeHtml(text || '');
        return safe.replace(/A\\d{{4,}}/g, (m) => `<span class=\"anchor-ref\" data-anchor=\"${{m}}\">${{m}}</span>`);
      }}

      const raw = document.getElementById('run-data').textContent;
      const DATA = JSON.parse(raw);
      document.getElementById('run-id').textContent = `run_id=${{DATA.run_id}}` + (DATA.viewer_label ? `  ·  ${{DATA.viewer_label}}` : '');

      let state = {{
        selectedItemId: null,
        selectedOutputId: null,
        itemFilter: '',
        docFilter: '',
        onlyCited: false,
      }};

      function getSelectedItem() {{
        if (!state.selectedItemId) {{
          const ids = Object.keys(DATA.items || {{}}).sort();
          state.selectedItemId = ids.length ? ids[0] : null;
        }}
        return state.selectedItemId ? DATA.items[state.selectedItemId] : null;
      }}

      function getSelectedOutput(item) {{
        if (!item) return null;
        const outs = item.outputs || [];
        if (!outs.length) return null;
        if (!state.selectedOutputId || !outs.some((o) => o.output_id === state.selectedOutputId)) {{
          state.selectedOutputId = item.default_output_id || outs[0].output_id;
        }}
        return outs.find((o) => o.output_id === state.selectedOutputId) || outs[0];
      }}

      function stageBadge(stage) {{
        if (stage === 'third_pass') return 'third_pass';
        if (stage === 'definitions_compiler_v1') return 'defs';
        if (stage === 'blocking_terms_compiler_v1') return 'bterms';
        if (stage === 'dg_structured_v2') return 'dg';
        if (stage === 'indexing_v2') return 'index';
        return stage;
      }}

      function renderItemList() {{
        const box = document.getElementById('item-list');
        box.innerHTML = '';

        const filter = (state.itemFilter || '').trim();
        const ids = Object.keys(DATA.items || {{}}).sort();
        for (const itemId of ids) {{
          if (filter && !itemId.includes(filter)) continue;
          const it = DATA.items[itemId];
          const outs = it.outputs || [];
          const stages = new Set(outs.map((o) => o.stage));
          const unknown = outs.reduce((acc, o) => acc + ((o.unknown_anchor_ids || []).length ? 1 : 0), 0);

          const div = document.createElement('div');
          div.className = 'item' + (itemId === state.selectedItemId ? ' active' : '');
          div.addEventListener('click', () => {{
            state.selectedItemId = itemId;
            state.selectedOutputId = null; // reset output per item
            render();
          }});

          const idDiv = document.createElement('div');
          idDiv.className = 'id';
          idDiv.textContent = itemId;

          const meta = document.createElement('div');
          meta.className = 'meta';

          for (const s of Array.from(stages).sort()) {{
            const b = document.createElement('div');
            b.className = 'badge good';
            b.textContent = stageBadge(s);
            meta.appendChild(b);
          }}

          const b2 = document.createElement('div');
          b2.className = 'badge ' + (outs.length ? 'good' : 'warn');
          b2.textContent = `outputs=${{outs.length}}`;
          meta.appendChild(b2);

          if (unknown) {{
            const b3 = document.createElement('div');
            b3.className = 'badge bad';
            b3.textContent = `unknown_refs=${{unknown}}`;
            meta.appendChild(b3);
          }}

          const docCount = (it && it.doc_blocks) ? it.doc_blocks.length : 0;
          const b4 = document.createElement('div');
          b4.className = 'badge';
          b4.textContent = `anchors=${{docCount}}`;
          meta.appendChild(b4);

          div.appendChild(idDiv);
          div.appendChild(meta);
          box.appendChild(div);
        }}
      }}

      function renderOutputPanel() {{
        const item = getSelectedItem();
        const title = document.getElementById('output-title');
        const pathLine = document.getElementById('output-path');
        const pre = document.getElementById('output-view');
        const sel = document.getElementById('output-select');

        sel.innerHTML = '';
        if (!item) {{
          title.textContent = 'Output';
          pathLine.textContent = '';
          pre.textContent = '(no items)';
          return;
        }}

        const outs = item.outputs || [];
        if (!outs.length) {{
          title.textContent = `Output :: ${{item.item_id}}`;
          pathLine.textContent = '';
          pre.textContent = '(no outputs found)';
          return;
        }}

        // Build select options grouped by stage.
        const byStage = new Map();
        for (const o of outs) {{
          if (!byStage.has(o.stage)) byStage.set(o.stage, []);
          byStage.get(o.stage).push(o);
        }}

        const stageOrder = ['third_pass', 'blocking_terms_compiler_v1', 'definitions_compiler_v1', 'dg_structured_v2', 'indexing_v2'];
        const stages = Array.from(byStage.keys()).sort((a, b) => stageOrder.indexOf(a) - stageOrder.indexOf(b));
        for (const st of stages) {{
          const grp = document.createElement('optgroup');
          grp.label = st;
          const arr = (byStage.get(st) || []).slice().sort((a,b) => (a.label || '').localeCompare(b.label || ''));
          for (const o of arr) {{
            const opt = document.createElement('option');
            opt.value = o.output_id;
            opt.textContent = o.label;
            grp.appendChild(opt);
          }}
          sel.appendChild(grp);
        }}

        const out = getSelectedOutput(item);
        sel.value = out ? out.output_id : (outs[0].output_id || '');
        sel.addEventListener('change', () => {{
          state.selectedOutputId = sel.value;
          renderOutputPanel();
          renderDocPanel();
        }}, {{ once: true }});

        title.textContent = `Output :: ${{item.item_id}}`;
        pathLine.textContent = out ? (out.path || '') : '';

        let rawText = '';
        if (!out) {{
          rawText = '(no output selected)';
        }} else if (out.kind === 'json') {{
          rawText = JSON.stringify(out.json, null, 2);
        }} else {{
          rawText = out.text || '';
        }}

        if (out && out.unknown_anchor_ids && out.unknown_anchor_ids.length) {{
          rawText = `=== WARNING: unknown anchor ids (not found in this agreement) ===\\n` + out.unknown_anchor_ids.join(', ') + `\\n\\n` + rawText;
        }}

        pre.innerHTML = renderTextWithAnchorLinks(rawText);
        pre.querySelectorAll('.anchor-ref').forEach((el) => {{
          el.addEventListener('click', () => selectDocAnchor(el.dataset.anchor));
        }});
      }}

      function renderDocPanel() {{
        const item = getSelectedItem();
        const out = getSelectedOutput(item);
        const docBox = document.getElementById('doc-view');
        const docTitle = document.getElementById('doc-title');
        const docCount = document.getElementById('doc-count');
        docBox.innerHTML = '';

        if (!item) {{
          docTitle.textContent = 'Agreement text';
          docCount.textContent = '';
          return;
        }}

        const blocks = item.doc_blocks || [];
        const filter = (state.docFilter || '').trim().toLowerCase();
        const cited = new Set((out && out.cited_anchor_ids) ? out.cited_anchor_ids : []);

        docTitle.textContent = `Agreement text :: ${{item.item_id}}`;

        const frag = document.createDocumentFragment();
        let shown = 0;
        for (const b of blocks) {{
          const aid = b.anchor_id;
          const text = b.text || '';
          const isRealAnchor = /^A\\d{{4,}}$/.test(aid);
          if (state.onlyCited && isRealAnchor && !cited.has(aid)) continue;
          if (filter) {{
            const hayA = (aid || '').toLowerCase();
            const hayT = text.toLowerCase();
            if (!(hayA.includes(filter) || hayT.includes(filter))) continue;
          }}

          const wrapper = document.createElement('div');
          wrapper.className = 'anchor-block';
          if (isRealAnchor) wrapper.id = 'doc-' + aid;

          const header = document.createElement('div');
          header.className = 'anchor-header';

          const left = document.createElement('div');
          left.className = 'aid';
          left.textContent = aid;
          if (isRealAnchor) left.addEventListener('click', () => selectDocAnchor(aid));

          const right = document.createElement('div');
          right.className = 'tags';
          if (isRealAnchor) {{
            right.textContent = state.onlyCited ? 'cited' : (cited.has(aid) ? 'cited' : '');
          }} else {{
            right.textContent = 'preface';
          }}

          header.appendChild(left);
          header.appendChild(right);

          const pre = document.createElement('pre');
          pre.innerHTML = escapeHtml(text);

          wrapper.appendChild(header);
          wrapper.appendChild(pre);
          frag.appendChild(wrapper);
          shown += 1;
        }}
        docBox.appendChild(frag);
        const citedLabel = state.onlyCited ? ' (cited only)' : '';
        const citedCount = cited.size ? ` · cited=${{cited.size}}` : '';
        docCount.textContent = `showing ${{shown}} / ${{blocks.length}} blocks${{citedLabel}}${{citedCount}}`;
      }}

      const toggleOnlyCited = document.getElementById('toggle-only-cited');
      toggleOnlyCited.addEventListener('click', () => {{
        state.onlyCited = !state.onlyCited;
        toggleOnlyCited.classList.toggle('active', state.onlyCited);
        renderDocPanel();
      }});

      document.getElementById('item-filter').addEventListener('input', (e) => {{
        state.itemFilter = e.target.value || '';
        renderItemList();
      }});

      document.getElementById('doc-filter').addEventListener('input', (e) => {{
        state.docFilter = e.target.value || '';
        renderDocPanel();
      }});

      function render() {{
        renderItemList();
        renderOutputPanel();
        renderDocPanel();
      }}

      render();
    </script>
  </body>
</html>
"""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--base-dir", default=".")
    ap.add_argument("--qa-subdir", default=None, help="llm_qa subdir for DG structured outputs (defaults from manifest when available).")
    ap.add_argument("--definitions-subdir", default=None, help="definitions_compiler_v1 output subdir (defaults from manifest).")
    ap.add_argument("--blocking-terms-subdir", default=None, help="blocking_terms_compiler_v1 output subdir (defaults from manifest).")
    ap.add_argument(
        "--third-pass-out-dir",
        default=None,
        help="Optional third-pass runner output dir (from scripts/pricing_definitions_third_pass_runner.py).",
    )
    ap.add_argument("--only-items", default=None, help="Comma-separated item_id list to include.")
    ap.add_argument("--max-items", type=int, default=0, help="If >0, only include first N items (sorted).")
    ap.add_argument("--out", default=None, help="Output HTML path.")
    args = ap.parse_args()

    paths = Paths(root=Path(args.base_dir), run_id=args.run_id)
    if not paths.run_dir.exists():
        raise SystemExit(f"Run dir not found: {paths.run_dir}")

    manifest = _load_manifest(paths)

    qa_subdir = args.qa_subdir
    if not qa_subdir:
        v = manifest.get("definitions_compiler_v1_qa_subdir")
        if isinstance(v, str) and v.strip():
            qa_subdir = v.strip()
        else:
            # Best-effort fallback: prefer a common subdir if present.
            candidate = paths.run_dir / "llm_qa" / "dg_strict_metric_consistency_v4_full"
            qa_subdir = "dg_strict_metric_consistency_v4_full" if candidate.exists() else None

    definitions_subdir = args.definitions_subdir
    if not definitions_subdir:
        v = manifest.get("definitions_compiler_v1_output_subdir")
        if isinstance(v, str) and v.strip():
            definitions_subdir = v.strip()

    blocking_terms_subdir = args.blocking_terms_subdir
    if not blocking_terms_subdir:
        v = manifest.get("blocking_terms_compiler_v1_output_subdir")
        if isinstance(v, str) and v.strip():
            blocking_terms_subdir = v.strip()

    third_pass_out_dir: Path | None = Path(args.third_pass_out_dir) if args.third_pass_out_dir else None
    third_pass_report: dict[str, Any] | None = None
    third_pass_strategies: list[str] = []
    viewer_label = ""
    if third_pass_out_dir and third_pass_out_dir.exists():
        rp = third_pass_out_dir / "report.json"
        if rp.exists():
            try:
                third_pass_report = _read_json(rp)
            except Exception:
                third_pass_report = None
        if isinstance(third_pass_report, dict):
            viewer_label = f"third_pass_out_dir={third_pass_out_dir}"
            third_pass_strategies = list(dict.fromkeys(third_pass_report.get("strategies") or []))  # type: ignore[arg-type]
        if not third_pass_strategies:
            # Fall back to directory inspection: strategy dirs are immediate children containing item_id folders.
            candidates = [p.name for p in third_pass_out_dir.iterdir() if p.is_dir() and p.name not in {"attempts", "best"}]
            third_pass_strategies = sorted([c for c in candidates if (third_pass_out_dir / c).is_dir()])

    # Item selection
    item_ids: List[str] = []
    if args.only_items:
        item_ids = [s.strip() for s in str(args.only_items).split(",") if s.strip()]
    elif isinstance(third_pass_report, dict) and isinstance(third_pass_report.get("items"), list):
        for row in third_pass_report.get("items") or []:
            if isinstance(row, dict) and isinstance(row.get("item_id"), str):
                item_ids.append(row["item_id"].strip())
    else:
        # Prefer manifest order if present; else use normalized.
        items = manifest.get("items")
        if isinstance(items, list) and items:
            for it in items:
                if isinstance(it, dict) and isinstance(it.get("item_id"), str):
                    item_ids.append(it["item_id"].strip())
        else:
            norm = paths.run_dir / "normalized"
            if norm.exists():
                item_ids = [p.name for p in norm.iterdir() if p.is_dir()]

    item_ids = sorted(set([x for x in item_ids if x]))
    if args.max_items and int(args.max_items) > 0:
        item_ids = item_ids[: int(args.max_items)]
    if not item_ids:
        raise SystemExit("No item_ids selected (use --only-items or provide a third-pass report with items).")

    out_path: Path
    if args.out:
        out_path = Path(args.out)
    elif third_pass_out_dir:
        out_path = third_pass_out_dir / "workflow_viewer.html"
    else:
        out_path = paths.run_dir / "pricing_workflow_viewer.html"

    items_payload: Dict[str, Any] = {}
    for item_id in item_ids:
        try:
            doc_source, blocks = _load_document_blocks(paths, item_id)
        except Exception as exc:
            # If we can't load the doc, fail loudly: without the doc we can't do provenance.
            raise RuntimeError(f"Failed to load agreement text for item_id={item_id}: {exc}") from exc

        doc_anchor_ids = {b["anchor_id"] for b in blocks if isinstance(b.get("anchor_id"), str) and ANCHOR_ID_RE.fullmatch(b["anchor_id"])}
        outputs = _load_outputs_for_item(
            paths=paths,
            item_id=item_id,
            doc_anchor_ids=doc_anchor_ids,
            qa_subdir=qa_subdir,
            definitions_subdir=definitions_subdir,
            blocking_terms_subdir=blocking_terms_subdir,
            third_pass_out_dir=third_pass_out_dir,
            third_pass_strategies=third_pass_strategies,
        )

        items_payload[item_id] = {
            "item_id": item_id,
            "doc_source_path": doc_source,
            "doc_blocks": blocks,
            "outputs": outputs,
            "default_output_id": _default_output_id(outputs),
        }

    data = {
        "run_id": args.run_id,
        "viewer_label": viewer_label,
        "qa_subdir": qa_subdir,
        "definitions_subdir": definitions_subdir,
        "blocking_terms_subdir": blocking_terms_subdir,
        "third_pass_out_dir": str(third_pass_out_dir) if third_pass_out_dir else None,
        "items": items_payload,
    }
    embedded = json.dumps(data, indent=2, sort_keys=True)
    embedded = embedded.replace("</script>", "<\\/script>")

    html = _html_template(title=f"Pricing Workflow Viewer :: {args.run_id}", embedded_json=embedded)
    _write_text(out_path, html)
    print(str(out_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

