#!/usr/bin/env python
"""
Render a self-contained HTML provenance viewer for a run.

Goal: open a single HTML file in a browser and:
  - view Pricing JSON (ContractIR v0.2 merged) and/or Covenant JSON (CovenantIR v0.1 validated)
  - click any anchor id (A####) inside the JSON to jump/highlight the cited source text

This script makes NO LLM calls. It only reads run artifacts:
  - runs/<run_id>/normalized/<item_id>/canonical.txt + anchors.tsv
  - runs/<run_id>/contractir_v0_2/items/<item_id>/contractir_merged.json
  - runs/<run_id>/covenantir_v0_1/<item_id>/covenantir_validated.json

Default output:
  runs/<run_id>/provenance_viewer.html
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"

import sys

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from pipeline.anchors import load_anchor_catalog  # noqa: E402
from pipeline.config import Paths  # noqa: E402


ANCHOR_ID_RE = re.compile(r"^A\d{4,}$")


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _read_text_or_none(path: Path) -> Optional[str]:
    try:
        return path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return None


def _anchor_text(canonical_text: str, catalog: Mapping[str, Mapping[str, Any]], anchor_id: str) -> Optional[str]:
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


def _iter_anchor_ids(obj: Any) -> Iterable[str]:
    """Find anchor ids anywhere in source_refs / anchor_ids lists."""

    def _walk(x: Any) -> Iterable[str]:
        if isinstance(x, dict):
            for k, v in x.items():
                if k in ("source_refs", "anchor_ids") and isinstance(v, list):
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


def _load_item_payload(
    *,
    paths: Paths,
    item_id: str,
    include_pricing: bool,
    include_covenants: bool,
) -> Optional[Dict[str, Any]]:
    run_dir = paths.run_dir

    canonical_path = run_dir / "normalized" / item_id / "canonical.txt"
    if not canonical_path.exists():
        return None
    canonical_text = canonical_path.read_text(encoding="utf-8", errors="replace")

    catalog = load_anchor_catalog(paths, item_id)

    pricing_doc = None
    pricing_path = run_dir / "contractir_v0_2" / "items" / item_id / "contractir_merged.json"
    if include_pricing and pricing_path.exists():
        pricing_doc = _read_json(pricing_path)

    covenant_doc = None
    covenant_path = run_dir / "covenantir_v0_1" / item_id / "covenantir_validated.json"
    if include_covenants and covenant_path.exists():
        covenant_doc = _read_json(covenant_path)

    if pricing_doc is None and covenant_doc is None:
        return None

    # Only include anchor texts that are actually cited by the extracted JSON.
    cited: Set[str] = set()
    if pricing_doc is not None:
        cited.update(_iter_anchor_ids(pricing_doc))
    if covenant_doc is not None:
        cited.update(_iter_anchor_ids(covenant_doc))

    anchors: Dict[str, Dict[str, Any]] = {}
    for aid in sorted(cited, key=lambda s: int(s[1:]) if ANCHOR_ID_RE.fullmatch(s) else 10**9):
        txt = _anchor_text(canonical_text, catalog, aid)
        if txt is None:
            continue
        anchors[aid] = {
            "anchor_type": catalog.get(aid, {}).get("anchor_type"),
            "order": catalog.get(aid, {}).get("order"),
            "text": txt,
        }

    return {
        "item_id": item_id,
        "pricing": {"path": str(pricing_path), "json": pricing_doc} if pricing_doc is not None else None,
        "covenants": {"path": str(covenant_path), "json": covenant_doc} if covenant_doc is not None else None,
        "anchors": anchors,
    }


def _load_prompt_bundle(*, paths: Paths, include_pricing: bool, include_covenants: bool) -> Dict[str, Any]:
    run_dir = paths.run_dir
    prompts: Dict[str, Any] = {}

    if include_covenants:
        summary_path = run_dir / "covenantir_v0_1" / "summary.json"
        if summary_path.exists():
            summary = _read_json(summary_path)
            prompt_ref = summary.get("prompt")
            if isinstance(prompt_ref, str) and prompt_ref.strip():
                ref = prompt_ref.strip()
                prompt_path = Path(ref)
                if not prompt_path.is_absolute():
                    prompt_path = (paths.root / prompt_path).resolve()
                prompt_text = _read_text_or_none(prompt_path)
                prompts["covenants_template"] = {
                    "kind": "template",
                    "path": ref,
                    "text": prompt_text,
                    "error": None if prompt_text is not None else f"Prompt file not found at: {prompt_path}",
                }

    # Pricing prompts are multi-part (indexing + base_rate/spread/fee). Only include
    # them in the bundle when pricing outputs are present.
    if include_pricing:
        summary_path = run_dir / "contractir_v0_2" / "summary.json"
        if summary_path.exists():
            summary = _read_json(summary_path)
            refs: List[str] = []
            for k in ("indexing_prompt", "base_rate_prompt"):
                v = summary.get(k)
                if isinstance(v, str) and v.strip():
                    refs.append(v.strip())
            for k in ("fee_prompts", "spread_prompts"):
                v = summary.get(k)
                if isinstance(v, list):
                    for p in v:
                        if isinstance(p, str) and p.strip():
                            refs.append(p.strip())
            # De-dup in stable order.
            seen: Set[str] = set()
            unique_refs = [r for r in refs if not (r in seen or seen.add(r))]
            entries: List[Dict[str, Any]] = []
            for ref in unique_refs:
                prompt_path = Path(ref)
                if not prompt_path.is_absolute():
                    prompt_path = (paths.root / prompt_path).resolve()
                prompt_text = _read_text_or_none(prompt_path)
                entries.append(
                    {
                        "kind": "template",
                        "path": ref,
                        "text": prompt_text,
                        "error": None if prompt_text is not None else f"Prompt file not found at: {prompt_path}",
                    }
                )
            if entries:
                prompts["pricing_templates"] = entries

    return prompts


def _html_template(*, title: str, embedded_json: str) -> str:
    # NOTE: embedded_json is inserted verbatim into a <script type="application/json"> tag.
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
        --sans: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, "Apple Color Emoji", "Segoe UI Emoji";
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
      .anchor-header .atype {{
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
    </style>
  </head>
  <body>
    <div class="app" id="app">
      <header>
        <div class="title">
          <div class="h1" id="viewer-title">Provenance Viewer</div>
          <div class="sub" id="run-id"></div>
        </div>
        <div class="controls">
          <div class="chip active" id="toggle-pricing">Pricing</div>
          <div class="chip active" id="toggle-covenants">Covenants</div>
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
          <div class="section-title" id="json-title">LLM JSON</div>
          <pre id="json-view"></pre>
        </div>
      </main>

      <main class="panel" id="anchors-panel">
        <div class="panel-inner">
          <div class="section-title">Cited source anchors</div>
          <div class="search" style="margin-bottom: 10px;">
            <input id="anchor-filter" placeholder="Filter anchors (e.g., A0414)…" />
          </div>
          <div id="anchors-view"></div>
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
        <div class="modal-sub" id="prompt-path"></div>
        <div class="modal-body">
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

      function renderJsonWithAnchorLinks(obj) {{
        if (!obj) return '(no output)';
        const text = JSON.stringify(obj, null, 2);
        const safe = escapeHtml(text);
        return safe.replace(/A\\d{{4,}}/g, (m) => `<span class=\"anchor-ref\" data-anchor=\"${{m}}\">${{m}}</span>`);
      }}

      function anchorSortKey(aid) {{
        const m = /^A(\\d+)$/.exec(aid);
        return m ? parseInt(m[1], 10) : 1e12;
      }}

      function selectAnchor(anchorId) {{
        const el = document.getElementById('anchor-' + anchorId);
        if (!el) return;
        el.scrollIntoView({{ behavior: 'smooth', block: 'start' }});
        el.classList.add('highlight');
        window.setTimeout(() => el.classList.remove('highlight'), 1200);
      }}

      const raw = document.getElementById('run-data').textContent;
      const DATA = JSON.parse(raw);
      document.getElementById('run-id').textContent = DATA.run_id;

      const ANY_PRICING = Object.values(DATA.items || {{}}).some((it) => !!it.pricing);
      const ANY_COVENANTS = Object.values(DATA.items || {{}}).some((it) => !!it.covenants);

      let state = {{
        showPricing: ANY_PRICING,
        showCovenants: ANY_COVENANTS,
        selectedItemId: null,
        itemFilter: '',
        anchorFilter: ''
      }};

      const togglePricing = document.getElementById('toggle-pricing');
      const toggleCovenants = document.getElementById('toggle-covenants');

      // Title + controls adapt automatically when only one artifact type exists.
      const viewerTitle = document.getElementById('viewer-title');
      if (ANY_PRICING && ANY_COVENANTS) {{
        viewerTitle.textContent = 'Provenance Viewer';
      }} else if (ANY_COVENANTS) {{
        viewerTitle.textContent = 'Covenant Provenance Viewer';
      }} else if (ANY_PRICING) {{
        viewerTitle.textContent = 'Pricing Provenance Viewer';
      }} else {{
        viewerTitle.textContent = 'Provenance Viewer';
      }}

      if (!ANY_PRICING) {{
        state.showPricing = false;
        togglePricing.classList.remove('active');
        togglePricing.style.display = 'none';
      }}
      if (!ANY_COVENANTS) {{
        state.showCovenants = false;
        toggleCovenants.classList.remove('active');
        toggleCovenants.style.display = 'none';
      }}

      const togglePrompt = document.getElementById('toggle-prompt');
      const promptModal = document.getElementById('prompt-modal');
      const promptPath = document.getElementById('prompt-path');
      const promptText = document.getElementById('prompt-text');
      const promptClose = document.getElementById('prompt-close');

      function pickPrompt() {{
        // Prefer covenant prompt in covenant-only runs; otherwise fall back to any pricing prompt.
        const prompts = DATA.prompts || {{}};
        if (prompts.covenants_template && prompts.covenants_template.text) return prompts.covenants_template;
        if (prompts.covenants_template) return prompts.covenants_template;
        const arr = prompts.pricing_templates || [];
        return arr.length ? arr[0] : null;
      }}

      const PROMPT = pickPrompt();
      if (!PROMPT) {{
        togglePrompt.style.display = 'none';
      }} else {{
        togglePrompt.addEventListener('click', () => {{
          promptModal.classList.add('show');
          promptModal.setAttribute('aria-hidden', 'false');
          const ref = PROMPT.path ? ('Prompt: ' + PROMPT.path) : 'Prompt';
          promptPath.textContent = ref + (PROMPT.error ? ('  |  ' + PROMPT.error) : '');
          promptText.textContent = PROMPT.text || '(prompt text not available)';
        }});
      }}

      function closePrompt() {{
        promptModal.classList.remove('show');
        promptModal.setAttribute('aria-hidden', 'true');
      }}

      promptClose.addEventListener('click', closePrompt);
      promptModal.addEventListener('click', (e) => {{
        if (e.target === promptModal) closePrompt();
      }});
      document.addEventListener('keydown', (e) => {{
        if (e.key === 'Escape') closePrompt();
      }});

      togglePricing.addEventListener('click', () => {{
        state.showPricing = !state.showPricing;
        togglePricing.classList.toggle('active', state.showPricing);
        render();
      }});
      toggleCovenants.addEventListener('click', () => {{
        state.showCovenants = !state.showCovenants;
        toggleCovenants.classList.toggle('active', state.showCovenants);
        render();
      }});
      document.getElementById('item-filter').addEventListener('input', (e) => {{
        state.itemFilter = e.target.value || '';
        renderItemList();
      }});
      document.getElementById('anchor-filter').addEventListener('input', (e) => {{
        state.anchorFilter = (e.target.value || '').trim().toUpperCase();
        renderAnchors();
      }});

      function getSelectedItem() {{
        if (!state.selectedItemId) {{
          const keys = Object.keys(DATA.items);
          state.selectedItemId = keys.length ? keys[0] : null;
        }}
        return state.selectedItemId ? DATA.items[state.selectedItemId] : null;
      }}

      function renderItemList() {{
        const list = document.getElementById('item-list');
        list.innerHTML = '';
        const filter = (state.itemFilter || '').toLowerCase().trim();

        const itemIds = Object.keys(DATA.items).sort();
        for (const itemId of itemIds) {{
          if (filter && !itemId.toLowerCase().includes(filter)) continue;
          const item = DATA.items[itemId];
          const hasPricing = !!item.pricing;
          const hasCov = !!item.covenants;

          const div = document.createElement('div');
          div.className = 'item' + (state.selectedItemId === itemId ? ' active' : '');
          div.addEventListener('click', () => {{
            state.selectedItemId = itemId;
            render();
          }});

          const id = document.createElement('div');
          id.className = 'id';
          id.textContent = itemId;

          const meta = document.createElement('div');
          meta.className = 'meta';
          if (ANY_PRICING) {{
            const b1 = document.createElement('span');
            b1.className = 'badge ' + (hasPricing ? 'good' : 'warn');
            b1.textContent = hasPricing ? 'pricing' : 'no pricing';
            meta.appendChild(b1);
          }}
          if (ANY_COVENANTS) {{
            const b2 = document.createElement('span');
            b2.className = 'badge ' + (hasCov ? 'good' : 'warn');
            b2.textContent = hasCov ? 'covenants' : 'no covenants';
            meta.appendChild(b2);
          }}

          div.appendChild(id);
          div.appendChild(meta);
          list.appendChild(div);
        }}
      }}

      function renderJson() {{
        const item = getSelectedItem();
        const pre = document.getElementById('json-view');
        const title = document.getElementById('json-title');

        if (!item) {{
          title.textContent = 'LLM JSON';
          pre.textContent = '(no items)';
          return;
        }}

        const blocks = [];
        const wantPricing = state.showPricing && item.pricing;
        const wantCov = state.showCovenants && item.covenants;
        const modeCount = (wantPricing ? 1 : 0) + (wantCov ? 1 : 0);

        if (wantPricing) {{
          const body = JSON.stringify(item.pricing.json, null, 2);
          blocks.push(modeCount > 1 ? ('=== PRICING (ContractIR v0.2 merged) ===\\n' + body) : body);
        }}
        if (wantCov) {{
          const body = JSON.stringify(item.covenants.json, null, 2);
          blocks.push(modeCount > 1 ? ('=== COVENANTS (CovenantIR v0.1 validated) ===\\n' + body) : body);
        }}
        title.textContent = `LLM JSON :: ${{item.item_id}}`;
        const raw = blocks.length ? blocks.join('\\n\\n') : '(no selected outputs)';
        const safe = escapeHtml(raw).replace(/A\\d{{4,}}/g, (m) => `<span class=\"anchor-ref\" data-anchor=\"${{m}}\">${{m}}</span>`);
        pre.innerHTML = safe;

        // Delegate clicks on anchor refs in JSON.
        pre.querySelectorAll('.anchor-ref').forEach((el) => {{
          el.addEventListener('click', () => selectAnchor(el.dataset.anchor));
        }});
      }}

      function renderAnchors() {{
        const item = getSelectedItem();
        const box = document.getElementById('anchors-view');
        box.innerHTML = '';
        if (!item) return;

        const filter = state.anchorFilter;
        const anchorIds = Object.keys(item.anchors || {{}}).sort((a,b) => anchorSortKey(a) - anchorSortKey(b));
        for (const aid of anchorIds) {{
          if (filter && !aid.includes(filter)) continue;
          const info = item.anchors[aid];

          const wrapper = document.createElement('div');
          wrapper.className = 'anchor-block';
          wrapper.id = 'anchor-' + aid;

          const header = document.createElement('div');
          header.className = 'anchor-header';

          const left = document.createElement('div');
          left.className = 'aid';
          left.textContent = aid;
          left.addEventListener('click', () => selectAnchor(aid));

          const right = document.createElement('div');
          right.className = 'atype';
          right.textContent = (info.anchor_type || 'anchor') + (Number.isInteger(info.order) ? ` · order=${{info.order}}` : '');

          header.appendChild(left);
          header.appendChild(right);

          const pre = document.createElement('pre');
          pre.innerHTML = escapeHtml(info.text || '');

          wrapper.appendChild(header);
          wrapper.appendChild(pre);
          box.appendChild(wrapper);
        }}
      }}

      function render() {{
        renderItemList();
        renderJson();
        renderAnchors();
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
    ap.add_argument("--out", default=None, help="Output HTML path (default: runs/<run_id>/provenance_viewer.html)")
    ap.add_argument("--pricing", action="store_true", help="Include pricing outputs (default: on)")
    ap.add_argument("--covenants", action="store_true", help="Include covenant outputs (default: on)")
    ap.add_argument("--max-items", type=int, default=0, help="If >0, only include first N items (sorted)")
    ap.add_argument("--only-items", default=None, help="Comma-separated item_id list to include")
    args = ap.parse_args()

    include_pricing = True if (not args.pricing and not args.covenants) else bool(args.pricing)
    include_covenants = True if (not args.pricing and not args.covenants) else bool(args.covenants)

    paths = Paths(root=Path(args.base_dir), run_id=args.run_id)
    run_dir = paths.run_dir
    out_path = Path(args.out) if args.out else (run_dir / "provenance_viewer.html")

    item_ids: List[str] = []
    if args.only_items:
        item_ids = [s.strip() for s in args.only_items.split(",") if s.strip()]
    else:
        # Prefer manifest ordering if present; otherwise use what exists in normalized.
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

    items: Dict[str, Any] = {}
    for item_id in item_ids:
        payload = _load_item_payload(
            paths=paths,
            item_id=item_id,
            include_pricing=include_pricing,
            include_covenants=include_covenants,
        )
        if payload is None:
            continue
        items[item_id] = payload

    prompts = _load_prompt_bundle(paths=paths, include_pricing=include_pricing, include_covenants=include_covenants)
    data = {"run_id": args.run_id, "items": items, "prompts": prompts}
    embedded = json.dumps(data, indent=2, sort_keys=True)
    # Avoid breaking out of the script tag in pathological cases.
    embedded = embedded.replace("</script>", "<\\/script>")

    html = _html_template(title=f"Provenance Viewer :: {args.run_id}", embedded_json=embedded)
    _write_text(out_path, html)
    print(str(out_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
