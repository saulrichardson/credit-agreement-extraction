#!/usr/bin/env python
"""
Render a self-contained HTML viewer for:
  - definitions_compiler_v1 outputs (v2 AST)
  - blocking_terms_compiler_v1 outputs (recursive defined terms)
  - compustat_overlay_v1 outputs (candidate Compustat mappings)

Goal: open a single HTML file in a browser and:
  - pick an agreement (item_id)
  - switch between pipeline stages
  - inspect each term output (compiled JSON + raw LLM output + contexts, when available)
  - click any cited anchor id (A####) to jump/highlight its support in the full agreement text

This script makes NO LLM calls. It only reads run artifacts:
  - runs/<run_id>/normalized/<item_id>/canonical_annotated.txt (preferred) or canonical.txt + anchors.tsv (fallback)
  - runs/<run_id>/definitions_compiler_v1/<subdir>/<item_id>__compiled.json
  - runs/<run_id>/blocking_terms_compiler_v1/<subdir>/<item_id>__compiled.json
  - runs/<run_id>/compustat_overlay_v1/<subdir>/<item_id>__compustat_overlay.json

Default output:
  runs/<run_id>/definitions_compustat_viewer.html
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from pipeline.anchors import load_anchor_catalog  # noqa: E402
from pipeline.config import Paths  # noqa: E402


ANCHOR_ID_RE = re.compile(r"^A\d{4,}$")
ANNOTATED_ANCHOR_LINE_RE = re.compile(r"^\[\[(A\d{4,})\]\]\s*$")


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def _read_text_or_none(path: Path) -> Optional[str]:
    try:
        return _read_text(path)
    except FileNotFoundError:
        return None


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _parse_canonical_annotated(text: str) -> List[Dict[str, Any]]:
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
        annotated_text = _read_text(annotated_path)
        return (str(annotated_path), _parse_canonical_annotated(annotated_text))

    return (
        str(paths.run_dir / "normalized" / item_id / "canonical.txt"),
        _fallback_blocks_from_canonical(paths, item_id),
    )


def _safe_slug(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "_", s).strip("_") or "term"


def _find_best_raw_text(path_dir: Path, stem_prefix: str, *, suffixes: Sequence[str]) -> Optional[Dict[str, Any]]:
    candidates: List[Path] = []
    seen: set[Path] = set()

    def add_if_exists(p: Path) -> None:
        if not p.exists():
            return
        rp = p.resolve()
        if rp in seen:
            return
        seen.add(rp)
        candidates.append(p)

    for suf in suffixes:
        add_if_exists(path_dir / f"{stem_prefix}{suf}")

    # Always include wildcard matches too (even if explicit suffixes exist),
    # since some stages may emit "repair" variants alongside a default ".raw.txt".
    for p in sorted(path_dir.glob(f"{stem_prefix}*.txt")):
        add_if_exists(p)

    if not candidates:
        return None

    def key(p: Path) -> Tuple[int, int]:
        name = p.name
        if "repair" in name:
            base = 50
        elif name.endswith(".raw.txt"):
            base = 40
        elif ".raw.attempt" in name:
            base = 30
        elif ".pass2." in name:
            base = 20
        elif ".pass1." in name:
            base = 10
        else:
            base = 0

        m = re.search(r"attempt(\d+)", name)
        attempt = int(m.group(1)) if m else 0
        return (base, attempt)

    best = sorted(candidates, key=key, reverse=True)[0]
    return {"path": str(best), "text": _read_text(best)}


def _iter_anchor_ids_from_source_refs(obj: Any) -> Iterable[str]:
    def _walk(x: Any) -> Iterable[str]:
        if isinstance(x, dict):
            for k, v in x.items():
                if k == "source_refs" and isinstance(v, list):
                    for aid in v:
                        if isinstance(aid, str):
                            cleaned = aid.strip()
                            if ANCHOR_ID_RE.fullmatch(cleaned):
                                yield cleaned
                else:
                    yield from _walk(v)
        elif isinstance(x, list):
            for v in x:
                yield from _walk(v)

    yield from _walk(obj)


def _load_manifest(paths: Paths) -> Dict[str, Any]:
    manifest_path = paths.run_dir / "manifest.json"
    if not manifest_path.exists():
        return {}
    try:
        doc = _read_json(manifest_path)
    except Exception:
        return {}
    return doc if isinstance(doc, dict) else {}


def _load_prompt_bundle(paths: Paths, manifest: Dict[str, Any]) -> List[Dict[str, Any]]:
    prompt_keys = [
        ("definitions_compiler_v1_prompt", "definitions_compiler_v1_prompt"),
        ("blocking_terms_compiler_v1_prompt", "blocking_terms_compiler_v1_prompt"),
        ("compustat_overlay_v1_prompt", "compustat_overlay_v1_prompt"),
    ]

    out: List[Dict[str, Any]] = []
    for key, label in prompt_keys:
        ref = manifest.get(key)
        if not isinstance(ref, str) or not ref.strip():
            continue
        p = Path(ref.strip())
        if not p.is_absolute():
            p = (paths.root / p).resolve()
        txt = _read_text_or_none(p)
        out.append(
            {
                "label": label,
                "path": ref.strip(),
                "text": txt,
                "error": None if txt is not None else f"Prompt file not found at: {p}",
            }
        )
    return out


def _load_stage_aggregate(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    doc = _read_json(path)
    if not isinstance(doc, dict):
        return None
    return {"path": str(path), "json": doc}


def _terms_from_definitions_aggregate(stage_dir: Path, item_id: str, agg_json: Dict[str, Any]) -> List[Dict[str, Any]]:
    defs = agg_json.get("definitions") or []
    out: List[Dict[str, Any]] = []
    if not isinstance(defs, list):
        return out

    for d in defs:
        if not isinstance(d, dict):
            continue
        term = d.get("name")
        if not isinstance(term, str) or not term.strip():
            continue
        slug = _safe_slug(term)

        raw = _find_best_raw_text(
            stage_dir,
            f"{item_id}__{slug}__compile",
            suffixes=(
                ".raw.txt",
                ".pass2.raw.txt",
                ".pass1.raw.txt",
            ),
        )
        contexts = _find_best_raw_text(
            stage_dir,
            f"{item_id}__{slug}__contexts",
            suffixes=(
                ".txt",
                ".pass2.txt",
                ".pass1.txt",
            ),
        )

        source_refs = [x for x in (d.get("source_refs") or []) if isinstance(x, str)]
        cited = sorted(set(a.strip() for a in source_refs if ANCHOR_ID_RE.fullmatch(a.strip())), key=lambda s: int(s[1:]))

        out.append(
            {
                "term": term,
                "slug": slug,
                "compiled": d,
                "raw": raw,
                "contexts": contexts,
                "cited_anchor_ids": cited,
            }
        )

    return sorted(out, key=lambda x: str(x.get("term") or "").lower())


def _terms_from_compustat_overlay_aggregate(
    stage_dir: Path,
    item_id: str,
    agg_json: Dict[str, Any],
    *,
    definitions_terms: Mapping[str, Dict[str, Any]],
    blocking_terms_terms: Mapping[str, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    overlays = agg_json.get("overlays") or []
    out: List[Dict[str, Any]] = []
    if not isinstance(overlays, list):
        return out

    for o in overlays:
        if not isinstance(o, dict):
            continue
        term = o.get("term")
        if not isinstance(term, str) or not term.strip():
            continue
        slug = _safe_slug(term)

        raw = _find_best_raw_text(
            stage_dir,
            f"{item_id}__{slug}__compustat_overlay",
            suffixes=(
                ".raw.txt",
                ".raw.attempt4.txt",
                ".raw.attempt3.txt",
                ".raw.attempt2.txt",
            ),
        )

        support_stage = None
        support_def = None
        if term in blocking_terms_terms:
            support_stage = "blocking_terms"
            support_def = blocking_terms_terms[term]
        elif term in definitions_terms:
            support_stage = "definitions"
            support_def = definitions_terms[term]

        cited: List[str] = []
        if support_def is not None:
            cited = sorted(set(_iter_anchor_ids_from_source_refs(support_def)), key=lambda s: int(s[1:]))

        out.append(
            {
                "term": term,
                "slug": slug,
                "compiled": o,
                "raw": raw,
                "support": {
                    "stage": support_stage,
                    "compiled": support_def,
                },
                "cited_anchor_ids": cited,
            }
        )

    return sorted(out, key=lambda x: str(x.get("term") or "").lower())


def _read_errors_or_none(path: Path) -> Optional[str]:
    txt = _read_text_or_none(path)
    if txt is None:
        return None
    return txt.strip() or None


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

      header {{
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 14px 16px;
        border-bottom: 1px solid var(--border);
        background: rgba(16, 23, 53, 0.75);
        backdrop-filter: blur(8px);
      }}

      .app {{
        display: grid;
        grid-template-columns: 320px 1fr 1fr;
        grid-template-rows: auto 1fr;
        height: 100vh;
      }}

      aside {{
        border-right: 1px solid var(--border);
        background: rgba(15, 22, 48, 0.7);
      }}

      .aside-inner {{
        height: calc(100vh - 60px);
        display: flex;
        flex-direction: column;
      }}

      .search {{
        padding: 12px;
        border-bottom: 1px solid var(--border);
      }}

      input, select {{
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

      .item-list {{
        padding: 10px;
        overflow: auto;
        display: flex;
        flex-direction: column;
        gap: 8px;
      }}

      .item {{
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 10px 10px;
        cursor: pointer;
        background: rgba(255,255,255,0.02);
      }}
      .item:hover {{
        border-color: rgba(110,231,255,0.25);
      }}
      .item.active {{
        border-color: rgba(110,231,255,0.55);
        box-shadow: 0 0 0 2px rgba(110,231,255,0.12);
      }}
      .item .id {{
        font-family: var(--mono);
        font-size: 12px;
      }}
      .item .meta {{
        margin-top: 4px;
        font-size: 11px;
        color: var(--muted);
      }}

      .panel {{
        overflow: hidden;
        display: flex;
        flex-direction: column;
      }}

      .panel-inner {{
        height: calc(100vh - 60px);
        overflow: auto;
        padding: 12px 12px 16px 12px;
      }}

      .title {{
        display: flex;
        flex-direction: column;
        gap: 4px;
      }}
      .h1 {{
        font-size: 16px;
        font-weight: 700;
      }}
      .sub {{
        font-size: 12px;
        color: var(--muted);
        font-family: var(--mono);
      }}

      .controls {{
        display: flex;
        align-items: center;
        gap: 10px;
      }}
      .chip {{
        font-family: var(--mono);
        font-size: 11px;
        padding: 6px 10px;
        border-radius: 999px;
        border: 1px solid var(--border);
        background: rgba(255,255,255,0.03);
        color: var(--text);
        cursor: pointer;
        user-select: none;
      }}
      .chip.on {{
        border-color: rgba(110,231,255,0.35);
        box-shadow: 0 0 0 2px rgba(110,231,255,0.12);
      }}

      .section-title {{
        font-weight: 650;
        margin: 2px 0 10px 0;
      }}

      .toolbar {{
        display: flex;
        gap: 10px;
        flex-wrap: wrap;
        margin-bottom: 10px;
      }}
      .toolbar .group {{
        display: flex;
        gap: 8px;
        align-items: center;
        flex-wrap: wrap;
      }}
      .toolbar label {{
        font-family: var(--mono);
        font-size: 11px;
        color: var(--muted);
      }}

      .kv {{
        font-family: var(--mono);
        font-size: 11px;
        color: var(--muted);
        margin-bottom: 8px;
      }}

      pre {{
        margin: 0;
        white-space: pre-wrap;
        word-break: break-word;
        font-family: var(--mono);
        font-size: 12px;
        line-height: 1.45;
        padding: 12px;
        border-radius: 12px;
        border: 1px solid var(--border);
        background: rgba(0,0,0,0.12);
      }}

      a.anchor {{
        color: var(--accent);
        text-decoration: none;
        border-bottom: 1px dashed rgba(110,231,255,0.4);
        cursor: pointer;
      }}
      a.anchor:hover {{
        color: white;
      }}

      .doc-controls {{
        display: flex;
        gap: 10px;
        align-items: center;
        margin-bottom: 10px;
      }}
      .doc-count {{
        font-family: var(--mono);
        font-size: 11px;
        color: var(--muted);
      }}

      .doc-block {{
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 10px 12px;
        margin-bottom: 10px;
        background: rgba(255,255,255,0.02);
      }}
      .doc-anchor {{
        font-family: var(--mono);
        font-size: 12px;
        color: var(--accent2);
        margin-bottom: 8px;
        display: inline-block;
      }}
      .doc-text {{
        white-space: pre-wrap;
        font-family: var(--mono);
        font-size: 12px;
        line-height: 1.45;
        color: rgba(255,255,255,0.88);
      }}
      .highlight {{
        box-shadow: 0 0 0 2px rgba(110,231,255,0.25);
        border-color: rgba(110,231,255,0.55);
      }}

      details {{
        margin-top: 10px;
        border: 1px solid var(--border);
        border-radius: 12px;
        background: rgba(255,255,255,0.02);
        padding: 10px 12px;
      }}
      details summary {{
        cursor: pointer;
        font-family: var(--mono);
        font-size: 12px;
        color: var(--muted);
      }}

      .anchor-list {{
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
        margin: 8px 0 10px 0;
      }}
      .anchor-pill {{
        font-family: var(--mono);
        font-size: 11px;
        padding: 4px 8px;
        border-radius: 999px;
        border: 1px solid var(--border);
        background: rgba(255,255,255,0.03);
      }}

      .modal {{
        display: none;
        position: fixed;
        left: 0;
        top: 0;
        right: 0;
        bottom: 0;
        background: rgba(0,0,0,0.55);
        align-items: center;
        justify-content: center;
        padding: 18px;
        z-index: 99;
      }}
      .modal.show {{
        display: flex;
      }}
      .modal-card {{
        width: min(1100px, 98vw);
        max-height: 92vh;
        overflow: hidden;
        border: 1px solid var(--border);
        border-radius: 14px;
        background-color: var(--panel);
        box-shadow: 0 30px 80px rgba(0,0,0,0.55);
        display: flex;
        flex-direction: column;
      }}
      .modal-head {{
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 10px;
        padding: 12px 14px;
        border-bottom: 1px solid var(--border);
      }}
      .modal-body {{
        padding: 0 14px 14px 14px;
        overflow: auto;
      }}
      .modal-body select {{
        width: 100%;
        margin: 8px 0 12px 0;
      }}
    </style>
  </head>
  <body>
    <div class="app" id="app">
      <header style="grid-column: 1 / -1;">
        <div class="title">
          <div class="h1">Definitions → Blocking Terms → Compustat Overlay Viewer</div>
          <div class="sub" id="run-id"></div>
        </div>
        <div class="controls">
          <div class="chip" id="toggle-only-cited">Only cited anchors</div>
          <div class="chip" id="toggle-prompts">Prompts</div>
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

      <main class="panel">
        <div class="panel-inner">
          <div class="section-title" id="out-title">Output</div>
          <div class="toolbar">
            <div class="group" style="min-width: 220px; flex: 1;">
              <label for="stage-select">Stage</label>
              <select id="stage-select"></select>
            </div>
            <div class="group" style="min-width: 260px; flex: 2;">
              <label for="term-select">Term</label>
              <select id="term-select"></select>
            </div>
            <div class="group">
              <div class="chip on" id="toggle-view-compiled">Compiled</div>
              <div class="chip" id="toggle-view-raw">Raw</div>
              <div class="chip" id="toggle-view-contexts">Contexts</div>
              <div class="chip" id="toggle-view-aggregate">Aggregate</div>
            </div>
          </div>

          <div class="kv" id="out-meta"></div>
          <div class="kv" id="out-cited-meta"></div>
          <div class="anchor-list" id="out-cited-list"></div>

          <pre id="out-view"></pre>

          <details id="support-details" style="display:none;">
            <summary>Support definition (for overlay terms)</summary>
            <pre id="support-view"></pre>
          </details>

          <details id="errors-details" style="display:none;">
            <summary>Stage errors (if any)</summary>
            <pre id="errors-view"></pre>
          </details>
        </div>
      </main>

      <main class="panel">
        <div class="panel-inner">
          <div class="section-title" id="doc-title">Agreement text</div>
          <div class="doc-controls">
            <div class="search" style="flex: 1; min-width: 240px; padding: 0; border: 0;">
              <input id="doc-filter" placeholder="Filter (anchor id like A0123 or text substring)…" />
            </div>
            <div class="doc-count" id="doc-count"></div>
          </div>
          <div id="doc-view"></div>
        </div>
      </main>
    </div>

    <div class="modal" id="prompts-modal" aria-hidden="true">
      <div class="modal-card">
        <div class="modal-head">
          <div style="font-weight:650;">Prompts</div>
          <div class="chip" id="prompts-close">Close</div>
        </div>
        <div class="modal-body">
          <select id="prompts-select"></select>
          <div class="kv" id="prompts-path"></div>
          <pre id="prompts-text"></pre>
        </div>
      </div>
    </div>

    <script type="application/json" id="run-data">
{embedded_json}
    </script>
    <script>
      const data = JSON.parse(document.getElementById('run-data').textContent);

      function escapeHtml(s) {{
        return (s || '')
          .replaceAll('&', '&amp;')
          .replaceAll('<', '&lt;')
          .replaceAll('>', '&gt;')
          .replaceAll('\\"', '&quot;')
          .replaceAll(\"'\", '&#39;');
      }}

      function anchorSortKey(aid) {{
        const m = /^A(\\d+)$/.exec(aid);
        return m ? parseInt(m[1], 10) : 1e12;
      }}

      function selectDocAnchor(anchorId) {{
        const el = document.getElementById('doc-' + anchorId);
        if (!el) return;
        el.scrollIntoView({{ behavior: 'smooth', block: 'start' }});
        el.classList.add('highlight');
        window.setTimeout(() => el.classList.remove('highlight'), 1200);
      }}

      function linkifyAnchors(text) {{
        const re = /\\bA\\d{{4,}}\\b/g;
        let out = '';
        let last = 0;
        while (true) {{
          const m = re.exec(text);
          if (!m) break;
          out += escapeHtml(text.slice(last, m.index));
          const aid = m[0];
          out += `<a class=\"anchor\" data-anchor=\"${{aid}}\">${{aid}}</a>`;
          last = m.index + aid.length;
        }}
        out += escapeHtml(text.slice(last));
        return out;
      }}

      function prettyJson(obj) {{
        try {{
          return JSON.stringify(obj, null, 2);
        }} catch (e) {{
          return String(obj);
        }}
      }}

      function renderJsonWithAnchorLinks(obj) {{
        return linkifyAnchors(prettyJson(obj));
      }}

      const stageLabels = {{
        definitions: 'Definitions',
        blocking_terms: 'Blocking terms',
        compustat_overlay: 'Compustat overlay',
      }};

      let state = {{
        itemFilter: '',
        docFilter: '',
        onlyCited: false,
        selectedItemId: null,
        selectedStage: 'definitions',
        selectedTerm: '__aggregate__',
        viewMode: 'compiled',
      }};

      function stageData(item) {{
        if (!item) return null;
        return (item.stages || {{}})[state.selectedStage] || null;
      }}

      function currentTermEntry(item) {{
        const sd = stageData(item);
        if (!sd) return null;
        const terms = sd.terms || [];
        for (const t of terms) {{
          if (t && t.term === state.selectedTerm) return t;
        }}
        return null;
      }}

      function currentCitedAnchors(item) {{
        const entry = currentTermEntry(item);
        if (entry && Array.isArray(entry.cited_anchor_ids)) {{
          return entry.cited_anchor_ids.slice().sort((a,b) => anchorSortKey(a)-anchorSortKey(b));
        }}
        return [];
      }}

      function renderItemList() {{
        const list = document.getElementById('item-list');
        const q = (state.itemFilter || '').toLowerCase();
        const ids = Object.keys(data.items || {{}}).sort();
        list.innerHTML = '';

        for (const id of ids) {{
          if (q && !id.toLowerCase().includes(q)) continue;
          const item = data.items[id];
          const defs = (item.stages && item.stages.definitions && item.stages.definitions.terms) ? item.stages.definitions.terms.length : 0;
          const blk = (item.stages && item.stages.blocking_terms && item.stages.blocking_terms.terms) ? item.stages.blocking_terms.terms.length : 0;
          const ovl = (item.stages && item.stages.compustat_overlay && item.stages.compustat_overlay.terms) ? item.stages.compustat_overlay.terms.length : 0;

          const el = document.createElement('div');
          el.className = 'item' + (state.selectedItemId === id ? ' active' : '');
          el.innerHTML = `<div class=\"id\">${{escapeHtml(id)}}</div><div class=\"meta\">defs=${{defs}} • blocking=${{blk}} • overlay=${{ovl}}</div>`;
          el.addEventListener('click', () => {{
            state.selectedItemId = id;
            state.selectedTerm = '__aggregate__';
            syncTermSelect();
            render();
          }});
          list.appendChild(el);
        }}
      }}

      function syncStageSelect() {{
        const sel = document.getElementById('stage-select');
        sel.innerHTML = '';
        for (const key of Object.keys(stageLabels)) {{
          const opt = document.createElement('option');
          opt.value = key;
          opt.textContent = stageLabels[key];
          if (key === state.selectedStage) opt.selected = true;
          sel.appendChild(opt);
        }}
      }}

      function syncTermSelect() {{
        const sel = document.getElementById('term-select');
        const item = data.items[state.selectedItemId] || null;
        const sd = stageData(item);
        const terms = (sd && sd.terms) ? sd.terms : [];

        sel.innerHTML = '';
        const aggOpt = document.createElement('option');
        aggOpt.value = '__aggregate__';
        aggOpt.textContent = '(aggregate)';
        sel.appendChild(aggOpt);

        for (const t of terms) {{
          const opt = document.createElement('option');
          opt.value = t.term;
          opt.textContent = t.term;
          sel.appendChild(opt);
        }}

        const has = Array.from(sel.options).some(o => o.value === state.selectedTerm);
        if (!has) state.selectedTerm = '__aggregate__';
        sel.value = state.selectedTerm;
      }}

      function renderOutputPanel() {{
        const title = document.getElementById('out-title');
        const view = document.getElementById('out-view');
        const meta = document.getElementById('out-meta');
        const citedMeta = document.getElementById('out-cited-meta');
        const citedList = document.getElementById('out-cited-list');
        const supportDetails = document.getElementById('support-details');
        const supportView = document.getElementById('support-view');
        const errorsDetails = document.getElementById('errors-details');
        const errorsView = document.getElementById('errors-view');

        const item = data.items[state.selectedItemId] || null;
        if (!item) {{
          title.textContent = 'Output';
          view.textContent = '(select an agreement)';
          meta.textContent = '';
          citedMeta.textContent = '';
          citedList.innerHTML = '';
          supportDetails.style.display = 'none';
          errorsDetails.style.display = 'none';
          return;
        }}

        const sd = stageData(item);
        title.textContent = `${{stageLabels[state.selectedStage]}} output`;
        const stagePath = (sd && sd.aggregate && sd.aggregate.path) ? sd.aggregate.path : '(missing)';
        meta.textContent = `item_id=${{state.selectedItemId}} • stage=${{state.selectedStage}} • aggregate=${{stagePath}}`;

        const cited = currentCitedAnchors(item);
        citedMeta.textContent = cited.length ? `cited anchors: ${{cited.length}}` : 'cited anchors: (none)';
        citedList.innerHTML = '';
        for (const aid of cited) {{
          const pill = document.createElement('div');
          pill.className = 'anchor-pill';
          pill.innerHTML = `<a class=\"anchor\" data-anchor=\"${{aid}}\">${{aid}}</a>`;
          citedList.appendChild(pill);
        }}

        const entry = currentTermEntry(item);

        let payload = null;
        if (state.selectedTerm === '__aggregate__') {{
          payload = (sd && sd.aggregate && sd.aggregate.json) ? sd.aggregate.json : null;
        }} else {{
          if (state.viewMode === 'compiled') {{
            payload = entry ? entry.compiled : null;
          }} else if (state.viewMode === 'raw') {{
            payload = (entry && entry.raw) ? entry.raw.text : null;
          }} else if (state.viewMode === 'contexts') {{
            payload = (entry && entry.contexts) ? entry.contexts.text : null;
          }} else if (state.viewMode === 'aggregate') {{
            payload = (sd && sd.aggregate && sd.aggregate.json) ? sd.aggregate.json : null;
          }}
        }}

        if (payload === null || payload === undefined) {{
          view.textContent = '(no output)';
        }} else if (typeof payload === 'string') {{
          view.innerHTML = linkifyAnchors(payload);
        }} else {{
          view.innerHTML = renderJsonWithAnchorLinks(payload);
        }}

        if (state.selectedStage === 'compustat_overlay' && entry && entry.support && entry.support.compiled) {{
          supportDetails.style.display = 'block';
          supportView.innerHTML = renderJsonWithAnchorLinks(entry.support.compiled);
        }} else {{
          supportDetails.style.display = 'none';
          supportView.textContent = '';
        }}

        if (sd && sd.errors_text) {{
          errorsDetails.style.display = 'block';
          errorsView.textContent = sd.errors_text;
        }} else {{
          errorsDetails.style.display = 'none';
          errorsView.textContent = '';
        }}
      }}

      function renderDocPanel() {{
        const title = document.getElementById('doc-title');
        const view = document.getElementById('doc-view');
        const count = document.getElementById('doc-count');
        const item = data.items[state.selectedItemId] || null;
        if (!item) {{
          title.textContent = 'Agreement text';
          view.innerHTML = '';
          count.textContent = '';
          return;
        }}

        title.textContent = `Agreement text • ${{state.selectedItemId}}`;
        const q = (state.docFilter || '').trim().toLowerCase();
        const cited = new Set(state.onlyCited ? currentCitedAnchors(item) : []);

        let shown = 0;
        view.innerHTML = '';

        for (const b of (item.doc_blocks || [])) {{
          const aid = b.anchor_id;
          const txt = b.text || '';

          if (aid !== '__PREFACE__' && cited.size && !cited.has(aid)) continue;

          if (q) {{
            if (aid !== '__PREFACE__' && q.startsWith('a') && /^a\\d+$/.test(q)) {{
              if ((aid || '').toLowerCase() !== q) continue;
            }} else {{
              if (!(aid || '').toLowerCase().includes(q) && !txt.toLowerCase().includes(q)) continue;
            }}
          }}

          const block = document.createElement('div');
          block.className = 'doc-block';
          if (aid && aid !== '__PREFACE__') block.id = 'doc-' + aid;

          const head = document.createElement('div');
          head.className = 'doc-anchor';
          if (aid === '__PREFACE__') {{
            head.textContent = '(preface)';
          }} else {{
            head.innerHTML = `<a class=\"anchor\" data-anchor=\"${{aid}}\">${{aid}}</a>`;
          }}

          const body = document.createElement('div');
          body.className = 'doc-text';
          body.textContent = txt;

          block.appendChild(head);
          block.appendChild(body);
          view.appendChild(block);
          shown += 1;
        }}

        count.textContent = `blocks shown: ${{shown}} / ${{(item.doc_blocks || []).length}}`;
      }}

      function wireAnchorClicks() {{
        document.body.addEventListener('click', (e) => {{
          const a = e.target && e.target.closest ? e.target.closest('a.anchor') : null;
          if (!a) return;
          const aid = a.getAttribute('data-anchor');
          if (!aid) return;
          e.preventDefault();
          selectDocAnchor(aid);
        }});
      }}

      function wireControls() {{
        document.getElementById('item-filter').addEventListener('input', (e) => {{
          state.itemFilter = e.target.value || '';
          renderItemList();
        }});

        document.getElementById('doc-filter').addEventListener('input', (e) => {{
          state.docFilter = e.target.value || '';
          renderDocPanel();
        }});

        document.getElementById('stage-select').addEventListener('change', (e) => {{
          state.selectedStage = e.target.value;
          state.selectedTerm = '__aggregate__';
          syncTermSelect();
          render();
        }});

        document.getElementById('term-select').addEventListener('change', (e) => {{
          state.selectedTerm = e.target.value;
          render();
        }});

        function setViewMode(mode) {{
          state.viewMode = mode;
          const modes = ['compiled', 'raw', 'contexts', 'aggregate'];
          for (const m of modes) {{
            const el = document.getElementById('toggle-view-' + m);
            if (!el) continue;
            if (m === mode) el.classList.add('on');
            else el.classList.remove('on');
          }}
          renderOutputPanel();
        }}

        document.getElementById('toggle-view-compiled').addEventListener('click', () => setViewMode('compiled'));
        document.getElementById('toggle-view-raw').addEventListener('click', () => setViewMode('raw'));
        document.getElementById('toggle-view-contexts').addEventListener('click', () => setViewMode('contexts'));
        document.getElementById('toggle-view-aggregate').addEventListener('click', () => setViewMode('aggregate'));

        document.getElementById('toggle-only-cited').addEventListener('click', () => {{
          state.onlyCited = !state.onlyCited;
          const el = document.getElementById('toggle-only-cited');
          if (state.onlyCited) el.classList.add('on');
          else el.classList.remove('on');
          renderDocPanel();
        }});

        // Prompts modal
        const modal = document.getElementById('prompts-modal');
        const closeBtn = document.getElementById('prompts-close');
        const toggleBtn = document.getElementById('toggle-prompts');
        const select = document.getElementById('prompts-select');
        const pathEl = document.getElementById('prompts-path');
        const textEl = document.getElementById('prompts-text');

        function renderPrompt() {{
          const idx = parseInt(select.value, 10);
          const p = (data.prompts || [])[idx];
          if (!p) return;
          pathEl.textContent = p.path || '';
          textEl.textContent = p.text || p.error || '(missing)';
        }}

        function openModal() {{
          modal.classList.add('show');
          select.innerHTML = '';
          const prompts = data.prompts || [];
          if (!prompts.length) {{
            const opt = document.createElement('option');
            opt.value = '0';
            opt.textContent = '(no prompts found in manifest)';
            select.appendChild(opt);
            pathEl.textContent = '';
            textEl.textContent = '';
            return;
          }}
          for (let i = 0; i < prompts.length; i++) {{
            const p = prompts[i];
            const opt = document.createElement('option');
            opt.value = String(i);
            opt.textContent = p.label;
            select.appendChild(opt);
          }}
          select.value = '0';
          renderPrompt();
        }}

        function closeModal() {{
          modal.classList.remove('show');
        }}

        toggleBtn.addEventListener('click', openModal);
        closeBtn.addEventListener('click', closeModal);
        select.addEventListener('change', renderPrompt);
        modal.addEventListener('click', (e) => {{
          if (e.target === modal) closeModal();
        }});
        document.addEventListener('keydown', (e) => {{
          if (e.key === 'Escape') closeModal();
        }});
      }}

      function render() {{
        renderItemList();
        syncStageSelect();
        syncTermSelect();
        renderOutputPanel();
        renderDocPanel();
      }}

      function init() {{
        document.getElementById('run-id').textContent = 'run_id=' + (data.run_id || '');
        const ids = Object.keys(data.items || {{}}).sort();
        state.selectedItemId = ids.length ? ids[0] : null;
        syncStageSelect();
        syncTermSelect();
        wireAnchorClicks();
        wireControls();
        render();
      }}

      init();
    </script>
  </body>
</html>
"""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--base-dir", default=".")
    ap.add_argument("--out", default=None, help="Output HTML path (default: runs/<run_id>/definitions_compustat_viewer.html)")
    ap.add_argument("--max-items", type=int, default=0, help="If >0, only include first N items (sorted)")
    ap.add_argument("--only-items", default=None, help="Comma-separated item_id list to include")
    ap.add_argument("--definitions-subdir", default=None, help="Override definitions_compiler_v1 output_subdir")
    ap.add_argument("--blocking-terms-subdir", default=None, help="Override blocking_terms_compiler_v1 output_subdir")
    ap.add_argument("--compustat-overlay-subdir", default=None, help="Override compustat_overlay_v1 output_subdir")
    args = ap.parse_args()

    paths = Paths(root=Path(args.base_dir), run_id=args.run_id)
    run_dir = paths.run_dir

    manifest = _load_manifest(paths)
    definitions_subdir = (args.definitions_subdir or manifest.get("definitions_compiler_v1_output_subdir") or "").strip()
    blocking_subdir = (args.blocking_terms_subdir or manifest.get("blocking_terms_compiler_v1_output_subdir") or "").strip()
    overlay_subdir = (args.compustat_overlay_subdir or manifest.get("compustat_overlay_v1_output_subdir") or "").strip()

    out_path = Path(args.out) if args.out else (run_dir / "definitions_compustat_viewer.html")

    if not definitions_subdir:
        raise SystemExit(
            "definitions subdir not found; pass --definitions-subdir or ensure manifest has definitions_compiler_v1_output_subdir"
        )
    if not blocking_subdir:
        raise SystemExit(
            "blocking terms subdir not found; pass --blocking-terms-subdir or ensure manifest has blocking_terms_compiler_v1_output_subdir"
        )
    if not overlay_subdir:
        raise SystemExit(
            "overlay subdir not found; pass --compustat-overlay-subdir or ensure manifest has compustat_overlay_v1_output_subdir"
        )

    item_ids: List[str] = []
    if args.only_items:
        item_ids = [s.strip() for s in str(args.only_items).split(",") if s.strip()]
    else:
        for it in (manifest.get("items") or []):
            if isinstance(it, dict) and isinstance(it.get("item_id"), str):
                item_ids.append(it["item_id"])

    item_ids = sorted(set(item_ids))
    if args.max_items and int(args.max_items) > 0:
        item_ids = item_ids[: int(args.max_items)]

    defs_dir = run_dir / "definitions_compiler_v1" / definitions_subdir
    blk_dir = run_dir / "blocking_terms_compiler_v1" / blocking_subdir
    ovl_dir = run_dir / "compustat_overlay_v1" / overlay_subdir

    items: Dict[str, Any] = {}
    for item_id in item_ids:
        doc_source, blocks = _load_document_blocks(paths, item_id)

        defs_agg = _load_stage_aggregate(defs_dir / f"{item_id}__compiled.json")
        blk_agg = _load_stage_aggregate(blk_dir / f"{item_id}__compiled.json")
        ovl_agg = _load_stage_aggregate(ovl_dir / f"{item_id}__compustat_overlay.json")

        defs_terms: List[Dict[str, Any]] = []
        defs_by_term: Dict[str, Dict[str, Any]] = {}
        if defs_agg and isinstance(defs_agg.get("json"), dict):
            defs_terms = _terms_from_definitions_aggregate(defs_dir, item_id, defs_agg["json"])
            for t in defs_terms:
                if isinstance(t.get("term"), str) and isinstance(t.get("compiled"), dict):
                    defs_by_term[t["term"]] = t["compiled"]

        blk_terms: List[Dict[str, Any]] = []
        blk_by_term: Dict[str, Dict[str, Any]] = {}
        if blk_agg and isinstance(blk_agg.get("json"), dict):
            blk_terms = _terms_from_definitions_aggregate(blk_dir, item_id, blk_agg["json"])
            for t in blk_terms:
                if isinstance(t.get("term"), str) and isinstance(t.get("compiled"), dict):
                    blk_by_term[t["term"]] = t["compiled"]

        ovl_terms: List[Dict[str, Any]] = []
        if ovl_agg and isinstance(ovl_agg.get("json"), dict):
            ovl_terms = _terms_from_compustat_overlay_aggregate(
                ovl_dir,
                item_id,
                ovl_agg["json"],
                definitions_terms=defs_by_term,
                blocking_terms_terms=blk_by_term,
            )

        items[item_id] = {
            "item_id": item_id,
            "doc_source_path": doc_source,
            "doc_blocks": blocks,
            "stages": {
                "definitions": {
                    "subdir": definitions_subdir,
                    "aggregate": defs_agg,
                    "terms": defs_terms,
                    "errors_text": _read_errors_or_none(defs_dir / "errors.txt"),
                },
                "blocking_terms": {
                    "subdir": blocking_subdir,
                    "aggregate": blk_agg,
                    "terms": blk_terms,
                    "errors_text": _read_errors_or_none(blk_dir / "errors.txt"),
                },
                "compustat_overlay": {
                    "subdir": overlay_subdir,
                    "aggregate": ovl_agg,
                    "terms": ovl_terms,
                    "errors_text": _read_errors_or_none(ovl_dir / "errors.txt"),
                },
            },
        }

    prompts = _load_prompt_bundle(paths, manifest)
    data = {
        "run_id": args.run_id,
        "definitions_subdir": definitions_subdir,
        "blocking_terms_subdir": blocking_subdir,
        "compustat_overlay_subdir": overlay_subdir,
        "items": items,
        "prompts": prompts,
    }

    embedded = json.dumps(data, indent=2, sort_keys=True)
    embedded = embedded.replace("</script>", "<\\/script>")

    html = _html_template(title=f"Definitions/Compustat Viewer :: {args.run_id}", embedded_json=embedded)
    _write_text(out_path, html)
    print(str(out_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
