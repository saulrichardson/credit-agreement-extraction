#!/usr/bin/env python
"""
Render a self-contained HTML viewer for facility fundamentals with clickable provenance.

Goal: open a single HTML file in a browser and:
  - view Facility Fundamentals JSON (facility_fundamentals_v1)
  - view the full agreement text (canonical_annotated when available)
  - click any cited anchor id (A####) in the JSON to jump/highlight the relevant agreement block

This script makes NO LLM calls. It only reads run artifacts:
  - runs/<run_id>/normalized/<item_id>/canonical_annotated.txt (preferred) or canonical.txt + anchors.tsv (fallback)
  - runs/<run_id>/facility_fundamentals/<subdir>/<item_id>.json

Default output:
  runs/<run_id>/facility_fundamentals_viewer_<subdir>.html
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


def _iter_anchor_ids_from_source_refs(obj: Any) -> Iterable[str]:
    """Find anchor IDs anywhere under `source_refs` lists."""

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
            # Some files might have a header before the first anchor marker; we keep it attached
            # to a synthetic preface block so it's still visible.
            current_id = "__PREFACE__"
            current_lines = []
        current_lines.append(raw_line)

    if current_id is not None:
        blocks.append({"anchor_id": current_id, "text": "\n".join(current_lines).rstrip("\n")})

    # Drop empty preface if present.
    if blocks and blocks[0].get("anchor_id") == "__PREFACE__" and not str(blocks[0].get("text") or "").strip():
        blocks = blocks[1:]

    # Sanity: ensure anchor_id shape for real anchors.
    cleaned: List[Dict[str, Any]] = []
    for b in blocks:
        aid = str(b.get("anchor_id") or "").strip()
        txt = str(b.get("text") or "")
        if aid != "__PREFACE__" and not ANCHOR_ID_RE.fullmatch(aid):
            # If parsing went wrong, fail loudly; we don't want a viewer that silently mis-links.
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
    """Fallback when canonical_annotated.txt is missing: use canonical.txt + anchors.tsv spans."""

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
    """Return (source_path_string, blocks)."""

    annotated_path = paths.run_dir / "normalized" / item_id / "canonical_annotated.txt"
    if annotated_path.exists():
        annotated_text = _read_text(annotated_path)
        return (str(annotated_path), _parse_canonical_annotated(annotated_text))
    # Fallback: canonical spans
    return (str(paths.run_dir / "normalized" / item_id / "canonical.txt"), _fallback_blocks_from_canonical(paths, item_id))


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
        grid-template-columns: 280px 1fr 1fr;
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

      header .title {{
        display: flex;
        flex-direction: column;
        gap: 2px;
      }}
      header .title .h1 {{
        font-weight: 650;
        letter-spacing: 0.2px;
      }}
      header .title .sub {{
        font-size: 12px;
        color: var(--muted);
        font-family: var(--mono);
      }}

      header .controls {{
        display: flex;
        align-items: center;
        gap: 10px;
        flex-wrap: wrap;
      }}

      .chip {{
        font-family: var(--mono);
        font-size: 12px;
        padding: 6px 10px;
        border-radius: 999px;
        border: 1px solid var(--border);
        background: rgba(255,255,255,0.04);
        color: var(--text);
        cursor: pointer;
        user-select: none;
      }}
      .chip.active {{
        border-color: rgba(110,231,255,0.6);
        background: rgba(110,231,255,0.08);
      }}

      aside {{
        border-right: 1px solid var(--border);
        background: rgba(255,255,255,0.02);
        overflow: auto;
      }}

      .aside-inner {{
        padding: 12px;
        display: flex;
        flex-direction: column;
        gap: 10px;
      }}

      .search {{
        display: flex;
        gap: 8px;
        align-items: center;
      }}
      .search input {{
        width: 100%;
        padding: 8px 10px;
        border-radius: 10px;
        border: 1px solid var(--border);
        background: rgba(255,255,255,0.03);
        color: var(--text);
        font-family: var(--mono);
        font-size: 12px;
      }}

      .item-list {{
        display: flex;
        flex-direction: column;
        gap: 6px;
      }}
      .item {{
        padding: 10px;
        border: 1px solid var(--border);
        border-radius: 12px;
        background: rgba(255,255,255,0.02);
        cursor: pointer;
      }}
      .item.active {{
        border-color: rgba(167,139,250,0.6);
        background: rgba(167,139,250,0.08);
      }}
      .item .id {{
        font-family: var(--mono);
        font-size: 12px;
        color: var(--text);
        word-break: break-all;
      }}
      .item .meta {{
        margin-top: 4px;
        font-size: 11px;
        color: var(--muted);
        display: flex;
        gap: 8px;
      }}
      .badge {{
        font-family: var(--mono);
        font-size: 10px;
        padding: 2px 6px;
        border-radius: 999px;
        border: 1px solid var(--border);
        background: rgba(255,255,255,0.03);
      }}
      .badge.good {{ border-color: rgba(52,211,153,0.5); color: rgba(52,211,153,0.95); }}
      .badge.warn {{ border-color: rgba(251,191,36,0.6); color: rgba(251,191,36,0.95); }}

      main {{
        overflow: hidden;
        display: flex;
        flex-direction: column;
      }}

      .panel {{
        height: 100%;
        overflow: auto;
        border-right: 1px solid var(--border);
      }}
      .panel:last-child {{
        border-right: none;
      }}

      .panel-inner {{
        padding: 12px;
      }}

      .section-title {{
        font-family: var(--mono);
        font-size: 12px;
        color: var(--muted);
        margin-bottom: 8px;
      }}

      pre {{
        margin: 0;
        padding: 12px;
        border-radius: 12px;
        border: 1px solid var(--border);
        background: rgba(255,255,255,0.02);
        font-family: var(--mono);
        font-size: 12px;
        line-height: 1.35;
        white-space: pre-wrap;
        word-break: break-word;
      }}

      .anchor-ref {{
        color: var(--accent);
        text-decoration: underline;
        cursor: pointer;
      }}

      .doc-controls {{
        display: flex;
        gap: 8px;
        align-items: center;
        flex-wrap: wrap;
        margin-bottom: 10px;
      }}

      .doc-count {{
        font-family: var(--mono);
        font-size: 11px;
        color: var(--muted);
      }}

      .anchor-block {{
        margin-bottom: 10px;
      }}
      .anchor-header {{
        display: flex;
        align-items: baseline;
        justify-content: space-between;
        gap: 10px;
        margin-bottom: 6px;
      }}
      .anchor-header .aid {{
        font-family: var(--mono);
        color: var(--accent);
        cursor: pointer;
      }}
      .anchor-header .tags {{
        font-family: var(--mono);
        font-size: 11px;
        color: var(--muted);
      }}

      .highlight {{
        outline: 2px solid rgba(110,231,255,0.45);
        box-shadow: 0 0 0 4px rgba(110,231,255,0.10);
      }}

      .modal {{
        position: fixed;
        inset: 0;
        background: rgba(0,0,0,0.55);
        display: none;
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
        background: linear-gradient(180deg, rgba(255,255,255,0.05), rgba(255,255,255,0.0));
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
      .modal-title {{
        font-weight: 650;
      }}
      .modal-sub {{
        padding: 0 14px 10px 14px;
        color: var(--muted);
        font-family: var(--mono);
        font-size: 11px;
      }}
      .modal-body {{
        padding: 0 14px 14px 14px;
        overflow: auto;
      }}
      .modal-body select {{
        width: 100%;
        margin: 8px 0 12px 0;
        padding: 8px 10px;
        border-radius: 10px;
        border: 1px solid var(--border);
        background: rgba(255,255,255,0.03);
        color: var(--text);
        font-family: var(--mono);
        font-size: 12px;
      }}
    </style>
  </head>
  <body>
    <div class="app" id="app">
      <header>
        <div class="title">
          <div class="h1" id="viewer-title">Facility Fundamentals Viewer</div>
          <div class="sub" id="run-id"></div>
        </div>
        <div class="controls">
          <div class="chip" id="toggle-only-cited">Only cited anchors</div>
          <div class="chip" id="toggle-prompt">Prompt</div>
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

      <main class="panel" id="json-panel">
        <div class="panel-inner">
          <div class="section-title" id="json-title">Facility fundamentals JSON</div>
          <pre id="json-view"></pre>
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

    <div class="modal" id="prompt-modal" aria-hidden="true">
      <div class="modal-card">
        <div class="modal-head">
          <div class="modal-title">Prompt</div>
          <div class="controls" style="gap: 8px;">
            <div class="chip" id="prompt-close">Close</div>
          </div>
        </div>
        <div class="modal-body">
          <select id="prompt-select"></select>
          <div class="modal-sub" id="prompt-path"></div>
          <pre id="prompt-text"></pre>
        </div>
      </div>
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

      function renderJsonWithAnchorLinks(obj) {{
        if (!obj) return '(no output)';
        const text = JSON.stringify(obj, null, 2);
        const safe = escapeHtml(text);
        return safe.replace(/A\\d{{4,}}/g, (m) => `<span class=\"anchor-ref\" data-anchor=\"${{m}}\">${{m}}</span>`);
      }}

      const raw = document.getElementById('run-data').textContent;
      const DATA = JSON.parse(raw);
      document.getElementById('run-id').textContent = `run_id=${{DATA.run_id}}  ·  facility_subdir=${{DATA.facility_subdir}}`;

      let state = {{
        selectedItemId: null,
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

      function renderItemList() {{
        const box = document.getElementById('item-list');
        box.innerHTML = '';

        const filter = (state.itemFilter || '').trim();
        const ids = Object.keys(DATA.items || {{}}).sort();
        for (const itemId of ids) {{
          if (filter && !itemId.includes(filter)) continue;
          const it = DATA.items[itemId];
          const hasFund = !!(it && it.fundamentals && it.fundamentals.json);
          const fundJson = hasFund ? it.fundamentals.json : null;
          const citedCount = (it && it.cited_anchor_ids) ? it.cited_anchor_ids.length : 0;
          const docCount = (it && it.doc_blocks) ? it.doc_blocks.length : 0;
          const facilityCount = Array.isArray(fundJson && fundJson.facilities) ? fundJson.facilities.length : 0;
          const lenderCount = Array.isArray(fundJson && fundJson.lenders) ? fundJson.lenders.length : 0;
          const lendersOmitted = !!(fundJson && fundJson.lenders_present_but_omitted);
          const facilityEndCount = Array.isArray(fundJson && fundJson.facilities)
            ? fundJson.facilities.filter((f) => f && f.facility_end).length
            : 0;

          const div = document.createElement('div');
          div.className = 'item' + (itemId === state.selectedItemId ? ' active' : '');
          div.addEventListener('click', () => {{
            state.selectedItemId = itemId;
            render();
          }});

          const idDiv = document.createElement('div');
          idDiv.className = 'id';
          idDiv.textContent = itemId;

          const meta = document.createElement('div');
          meta.className = 'meta';
          const b1 = document.createElement('div');
          b1.className = 'badge ' + (hasFund ? 'good' : 'warn');
          b1.textContent = hasFund ? 'fundamentals' : 'missing';
          const b2 = document.createElement('div');
          b2.className = 'badge';
          b2.textContent = `cited=${{citedCount}}`;
          const b3 = document.createElement('div');
          b3.className = 'badge';
          b3.textContent = `anchors=${{docCount}}`;
          const b4 = document.createElement('div');
          b4.className = 'badge';
          b4.textContent = `facilities=${{facilityCount}}`;
          const b5 = document.createElement('div');
          b5.className = 'badge';
          b5.textContent = lendersOmitted ? 'lenders=omitted' : `lenders=${{lenderCount}}`;
          const b6 = document.createElement('div');
          b6.className = 'badge';
          b6.textContent = `facility_end=${{facilityEndCount}}`;
          meta.appendChild(b1);
          meta.appendChild(b2);
          meta.appendChild(b3);
          meta.appendChild(b4);
          meta.appendChild(b5);
          meta.appendChild(b6);

          div.appendChild(idDiv);
          div.appendChild(meta);
          box.appendChild(div);
        }}
      }}

      function renderJsonPanel() {{
        const item = getSelectedItem();
        const pre = document.getElementById('json-view');
        const title = document.getElementById('json-title');
        if (!item) {{
          title.textContent = 'Facility fundamentals JSON';
          pre.textContent = '(no items)';
          return;
        }}
        title.textContent = `Facility fundamentals JSON :: ${{item.item_id}}`;
        pre.innerHTML = renderJsonWithAnchorLinks(item.fundamentals ? item.fundamentals.json : null);
        pre.querySelectorAll('.anchor-ref').forEach((el) => {{
          el.addEventListener('click', () => selectDocAnchor(el.dataset.anchor));
        }});
      }}

      function renderDocPanel() {{
        const item = getSelectedItem();
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
        const cited = new Set(item.cited_anchor_ids || []);
        const filter = (state.docFilter || '').trim().toLowerCase();

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
          if (isRealAnchor) {{
            left.addEventListener('click', () => selectDocAnchor(aid));
          }}

          const right = document.createElement('div');
          right.className = 'tags';
          if (isRealAnchor) {{
            right.textContent = cited.has(aid) ? 'cited' : '';
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
        docCount.textContent = `showing ${{shown}} / ${{blocks.length}} blocks` + (state.onlyCited ? ' (cited only)' : '');
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

      // Prompt modal
      const togglePrompt = document.getElementById('toggle-prompt');
      const promptModal = document.getElementById('prompt-modal');
      const promptSelect = document.getElementById('prompt-select');
      const promptPath = document.getElementById('prompt-path');
      const promptText = document.getElementById('prompt-text');
      const promptClose = document.getElementById('prompt-close');

      function closePrompt() {{
        promptModal.classList.remove('show');
        promptModal.setAttribute('aria-hidden', 'true');
      }}

      function renderPrompt(idx) {{
        const prompts = DATA.prompts || [];
        const p = prompts[idx] || null;
        if (!p) {{
          promptPath.textContent = '(no prompt)';
          promptText.textContent = '';
          return;
        }}
        const ref = p.path ? ('Prompt: ' + p.path) : 'Prompt';
        promptPath.textContent = ref + (p.error ? ('  |  ' + p.error) : '');
        promptText.textContent = p.text || '(prompt text not available)';
      }}

      if (!DATA.prompts || !DATA.prompts.length) {{
        togglePrompt.style.display = 'none';
      }} else {{
        promptSelect.innerHTML = '';
        DATA.prompts.forEach((p, idx) => {{
          const opt = document.createElement('option');
          opt.value = String(idx);
          opt.textContent = p.label || p.path || ('prompt ' + idx);
          promptSelect.appendChild(opt);
        }});
        promptSelect.addEventListener('change', () => {{
          renderPrompt(parseInt(promptSelect.value, 10) || 0);
        }});

        togglePrompt.addEventListener('click', () => {{
          promptModal.classList.add('show');
          promptModal.setAttribute('aria-hidden', 'false');
          promptSelect.value = '0';
          renderPrompt(0);
        }});

        promptClose.addEventListener('click', closePrompt);
        promptModal.addEventListener('click', (e) => {{
          if (e.target === promptModal) closePrompt();
        }});
        document.addEventListener('keydown', (e) => {{
          if (e.key === 'Escape') closePrompt();
        }});
      }}

      function render() {{
        renderItemList();
        renderJsonPanel();
        renderDocPanel();
      }}

      render();
    </script>
  </body>
</html>
"""


def _load_prompt_bundle(paths: Paths) -> List[Dict[str, Any]]:
    """Best-effort: include facility fundamentals prompt + indexing v2 prompt from manifest."""

    manifest_path = paths.run_dir / "manifest.json"
    if not manifest_path.exists():
        return []
    manifest = _read_json(manifest_path)

    prompt_entries: List[Tuple[str, str]] = []
    for key, label in (
        ("facility_fundamentals_prompt", "facility_fundamentals_prompt"),
        ("indexing_v2_prompt", "indexing_v2_prompt"),
    ):
        ref = manifest.get(key)
        if isinstance(ref, str) and ref.strip():
            prompt_entries.append((label, ref.strip()))

    out: List[Dict[str, Any]] = []
    for label, ref in prompt_entries:
        p = Path(ref)
        if not p.is_absolute():
            p = (paths.root / p).resolve()
        txt = _read_text_or_none(p)
        out.append(
            {
                "label": label,
                "path": ref,
                "text": txt,
                "error": None if txt is not None else f"Prompt file not found at: {p}",
            }
        )
    return out


def _load_facility_fundamentals(paths: Paths, *, subdir: str, item_id: str) -> Optional[Dict[str, Any]]:
    path = paths.run_dir / "facility_fundamentals" / subdir / f"{item_id}.json"
    if not path.exists():
        return None
    return {"path": str(path), "json": _read_json(path)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--base-dir", default=".")
    ap.add_argument(
        "--facility-subdir",
        required=True,
        help="Subfolder under runs/<run_id>/facility_fundamentals/ containing {item_id}.json outputs.",
    )
    ap.add_argument(
        "--out",
        default=None,
        help="Output HTML path (default: runs/<run_id>/facility_fundamentals_viewer_<facility_subdir>.html)",
    )
    ap.add_argument("--max-items", type=int, default=0, help="If >0, only include first N items (sorted)")
    ap.add_argument("--only-items", default=None, help="Comma-separated item_id list to include")
    args = ap.parse_args()

    paths = Paths(root=Path(args.base_dir), run_id=args.run_id)
    run_dir = paths.run_dir
    facility_subdir = str(args.facility_subdir).strip()
    if not facility_subdir:
        raise SystemExit("--facility-subdir must be non-empty")

    out_path = (
        Path(args.out)
        if args.out
        else (run_dir / f"facility_fundamentals_viewer_{facility_subdir}.html")
    )

    # Determine items to include.
    item_ids: List[str] = []
    if args.only_items:
        item_ids = [s.strip() for s in args.only_items.split(",") if s.strip()]
    else:
        manifest_path = run_dir / "manifest.json"
        if manifest_path.exists():
            manifest = _read_json(manifest_path)
            for it in (manifest.get("items") or []):
                if isinstance(it, dict) and isinstance(it.get("item_id"), str):
                    item_ids.append(it["item_id"])
        else:
            norm = run_dir / "normalized"
            if norm.exists():
                item_ids = [p.name for p in norm.iterdir() if p.is_dir()]

    item_ids = sorted(set(item_ids))
    if args.max_items and int(args.max_items) > 0:
        item_ids = item_ids[: int(args.max_items)]

    # Load payloads.
    items: Dict[str, Any] = {}
    for item_id in item_ids:
        fundamentals = _load_facility_fundamentals(paths, subdir=facility_subdir, item_id=item_id)
        if fundamentals is None:
            # Still include the document so you can see what's missing.
            fundamentals_json = None
            cited: List[str] = []
        else:
            fundamentals_json = fundamentals.get("json")
            cited = sorted(set(_iter_anchor_ids_from_source_refs(fundamentals_json)), key=lambda s: int(s[1:]))

        doc_source, blocks = _load_document_blocks(paths, item_id)
        items[item_id] = {
            "item_id": item_id,
            "fundamentals": fundamentals,
            "cited_anchor_ids": cited,
            "doc_source_path": doc_source,
            "doc_blocks": blocks,
        }

    prompts = _load_prompt_bundle(paths)
    data = {"run_id": args.run_id, "facility_subdir": facility_subdir, "items": items, "prompts": prompts}
    embedded = json.dumps(data, indent=2, sort_keys=True)
    embedded = embedded.replace("</script>", "<\\/script>")

    html = _html_template(title=f"Facility Fundamentals Viewer :: {args.run_id}", embedded_json=embedded)
    _write_text(out_path, html)
    print(str(out_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
