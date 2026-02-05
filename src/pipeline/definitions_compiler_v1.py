from __future__ import annotations

import asyncio
import json
import re
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from .anchors import load_anchor_catalog
from .config import Paths, REQUIRED_MODEL, REQUIRED_REASONING, prompt_hash, update_manifest
from .excerpt_packs import build_excerpt_pack_from_canonical, expand_anchor_ids
from .llm.gateway import DEFAULT_GATEWAY_URL, _ensure_gateway_client_async
from .indexing_pricing_definitions_v1_schemas import PricingDefinitionsIndexingSelectionV1
from .schemas_v2 import IndexingSelectionV2Artifact
from .utils import assert_exists, prompt_view_path


@dataclass(frozen=True)
class MetricTarget:
    name: str
    contract_term_hint: str
    search_tokens: list[str]


def _load_dg_output(paths: Paths, item_id: str, qa_subdir: str) -> Dict[str, Any]:
    path_json = paths.structured_dir / qa_subdir / f"{item_id}.json"
    path_txt = paths.structured_dir / qa_subdir / f"{item_id}.txt"
    path = path_json if path_json.exists() else path_txt
    if not path.exists():
        raise FileNotFoundError(
            "Missing structured output for definition compilation. "
            f"Expected {path_json} (preferred) or legacy {path_txt}."
        )
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"dg output is not valid JSON for {item_id}: {path}") from exc
    if not isinstance(doc, dict):
        raise RuntimeError(f"dg output must be a JSON object for {item_id}: {path}")
    return doc


def _extract_metric_targets(doc: Dict[str, Any]) -> List[MetricTarget]:
    metrics: list[MetricTarget] = []
    raw_metrics = doc.get("metrics", []) or []
    if not isinstance(raw_metrics, list):
        raise RuntimeError("dg output metrics must be a list")

    for m in raw_metrics:
        if not isinstance(m, dict):
            continue
        name = m.get("name")
        if not isinstance(name, str) or not name.strip():
            continue
        contract_term = m.get("contract_term")
        contract_term_hint = contract_term.strip() if isinstance(contract_term, str) else ""
        search_tokens = m.get("search_tokens")
        toks: list[str] = []
        if isinstance(search_tokens, list) and all(isinstance(t, str) for t in search_tokens):
            toks = [t.strip() for t in search_tokens if t.strip()]
        metrics.append(
            MetricTarget(
                name=name.strip(),
                contract_term_hint=contract_term_hint,
                search_tokens=toks,
            )
        )

    # De-dupe by name (stable order).
    seen: set[str] = set()
    unique: list[MetricTarget] = []
    for mt in metrics:
        if mt.name in seen:
            continue
        seen.add(mt.name)
        unique.append(mt)
    return unique


def _find_term_hits(text: str, term: str) -> List[Tuple[int, int]]:
    """Return case-insensitive matches for a term, tolerant to whitespace differences."""

    parts = [p for p in re.split(r"\s+", term.strip()) if p]
    if not parts:
        return []
    pattern = r"\s+".join(re.escape(p) for p in parts)
    return [(m.start(), m.end()) for m in re.finditer(pattern, text, flags=re.IGNORECASE)]


def _match_hits_to_anchors(
    anchors: Dict[str, Dict[str, Any]], hits: List[Tuple[int, int]]
) -> List[str]:
    ordered = sorted(anchors.values(), key=lambda a: a["start"])
    anchor_ids: List[str] = []
    for start, _end in hits:
        for info in ordered:
            if info["start"] <= start < info["end"]:
                anchor_ids.append(info["anchor_id"])
                break
    seen = set()
    unique: list[str] = []
    for aid in anchor_ids:
        if aid in seen:
            continue
        seen.add(aid)
        unique.append(aid)
    return unique


def _definitions_anchor_ids_from_indexing_v2(
    paths: Paths, item_id: str, anchors: Dict[str, Dict[str, Any]]
) -> Tuple[List[str], Optional[Tuple[int, int]]]:
    path = paths.run_dir / "indexing_v2" / f"{item_id}_anchors.json"
    if not path.exists():
        return ([], None)

    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Indexing v2 artifact is not valid JSON for {item_id}: {path}") from exc

    artifact = IndexingSelectionV2Artifact.model_validate(doc)
    dr = artifact.selection.definitions_anchor_range
    if dr is None:
        return ([], None)

    if dr.start_anchor not in anchors:
        raise RuntimeError(
            f"Indexing v2 definitions_anchor_range.start_anchor {dr.start_anchor!r} not found in anchors.tsv for {item_id}"
        )
    if dr.end_anchor not in anchors:
        raise RuntimeError(
            f"Indexing v2 definitions_anchor_range.end_anchor {dr.end_anchor!r} not found in anchors.tsv for {item_id}"
        )

    start_order = int(anchors[dr.start_anchor]["order"])
    end_order = int(anchors[dr.end_anchor]["order"])
    if start_order > end_order:
        raise RuntimeError(
            f"Indexing v2 definitions_anchor_range has start after end for {item_id}: "
            f"{dr.start_anchor} (order={start_order}) > {dr.end_anchor} (order={end_order})"
        )

    ordered = sorted(anchors.values(), key=lambda a: int(a["order"]))
    anchor_ids = [str(info["anchor_id"]) for info in ordered[start_order : end_order + 1]]

    start_pos = int(anchors[dr.start_anchor]["start"])
    end_pos = int(anchors[dr.end_anchor]["end"])
    if start_pos > end_pos:
        raise RuntimeError(
            f"Indexing v2 definitions_anchor_range has start_pos after end_pos for {item_id}: {start_pos} > {end_pos}"
        )

    return (anchor_ids, (start_pos, end_pos))


def _pricing_union_anchor_ids_from_indexing_v2(paths: Paths, item_id: str, anchors: Dict[str, Dict[str, Any]]) -> List[str]:
    """Return union(pricing,base_rate,spread,fee) anchors from indexing_v2 in doc order."""

    path = paths.run_dir / "indexing_v2" / f"{item_id}_anchors.json"
    if not path.exists():
        return []

    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Indexing v2 artifact is not valid JSON for {item_id}: {path}") from exc

    artifact = IndexingSelectionV2Artifact.model_validate(doc)
    sel = artifact.selection

    seeds: list[str] = (
        list(sel.pricing_anchors or [])
        + list(sel.base_rate_anchors or [])
        + list(sel.spread_anchors or [])
        + list(sel.fee_anchors or [])
    )

    # Stable de-dupe; keep only anchors that exist in this document.
    seen: set[str] = set()
    merged: list[str] = []
    for aid in seeds:
        if not isinstance(aid, str):
            continue
        a = aid.strip()
        if not a or a in seen:
            continue
        if a not in anchors:
            raise RuntimeError(
                f"Indexing v2 selected unknown pricing_union anchor_id {a!r} for {item_id} (not present in anchors.tsv)"
            )
        seen.add(a)
        merged.append(a)

    return sorted(merged, key=lambda a: int(anchors[a]["order"]))


def _financial_covenant_anchor_ids_from_indexing_v2(
    paths: Paths, item_id: str, anchors: Dict[str, Dict[str, Any]]
) -> List[str]:
    """Return financial_covenant_anchors from indexing_v2 in doc order."""

    path = paths.run_dir / "indexing_v2" / f"{item_id}_anchors.json"
    if not path.exists():
        return []

    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Indexing v2 artifact is not valid JSON for {item_id}: {path}") from exc

    artifact = IndexingSelectionV2Artifact.model_validate(doc)
    seeds = list(artifact.selection.financial_covenant_anchors or [])

    seen: set[str] = set()
    merged: list[str] = []
    for aid in seeds:
        if not isinstance(aid, str):
            continue
        a = aid.strip()
        if not a or a in seen:
            continue
        if a not in anchors:
            raise RuntimeError(
                f"Indexing v2 selected unknown financial_covenant anchor_id {a!r} for {item_id} (not present in anchors.tsv)"
            )
        seen.add(a)
        merged.append(a)

    return sorted(merged, key=lambda a: int(anchors[a]["order"]))


def _select_context_anchor_ids(
    *,
    canonical_text: str,
    anchors: Dict[str, Dict[str, Any]],
    candidate_anchor_ids: List[str],
    contract_term: str,
    search_tokens: List[str],
    max_anchors: int = 14,
    include_neighbors_before: int = 1,
    include_neighbors_after: int = 2,
    allowed_anchor_ids: Optional[set[str]] = None,
) -> List[str]:
    """Down-select candidate anchors to a bounded context set, preferring definition-like text."""

    def _anchor_text(aid: str) -> str:
        info = anchors.get(aid)
        if not info:
            return ""
        start = int(info["start"])
        end = int(info["end"])
        return canonical_text[start:end]

    def _score(text: str) -> int:
        hay = text.lower()
        score = 0
        # Strong match: the contract term itself is being defined in this block.
        if contract_term:
            term_parts = [p for p in re.split(r"\s+", contract_term.lower().strip()) if p]
            term_pat = r"\s+".join(re.escape(p) for p in term_parts)
            if term_pat:
                if re.search(rf"[\"“]?\s*{term_pat}\s*[\"”]?\s+(?:shall\s+)?mean(?:s)?\b", hay):
                    score += 30
                if re.search(rf"[\"“]?\s*{term_pat}\s*[\"”]?\s+is\s+defined\s+as\b", hay):
                    score += 28
        # Definition trigger phrases (tolerate punctuation/newlines).
        if re.search(r"\bshall\s+mean\b", hay):
            score += 12
        if re.search(r"\bmeans\b", hay):
            score += 6
        if re.search(r"\bis\s+defined\s+as\b", hay):
            score += 10
        if re.search(r"\bshall\s+have\s+the\s+meaning\b", hay) or re.search(r"\bhas\s+the\s+meaning\s+assigned\b", hay):
            score += 10
        # Common definition headings.
        if hay.strip().startswith(("definitions", "defined terms", "section 1")):
            score += 2
        if contract_term and contract_term.lower() in hay:
            score += 5
        for tok in search_tokens:
            if tok.lower() in hay:
                score += 1
        return score

    candidate_ids = [aid for aid in candidate_anchor_ids if not allowed_anchor_ids or aid in allowed_anchor_ids]
    scored: list[tuple[int, int, str]] = []
    for aid in candidate_ids:
        info = anchors.get(aid)
        if not info:
            continue
        start = int(info["start"])
        scored.append((_score(_anchor_text(aid)), start, aid))

    if not scored:
        return []

    # Prefer higher score; tie-breaker by earlier document position.
    scored_sorted = sorted(scored, key=lambda t: (-t[0], t[1]))

    ordered = sorted(anchors.values(), key=lambda a: int(a["order"]))
    id_by_order = [str(info["anchor_id"]) for info in ordered]
    order_by_id = {str(info["anchor_id"]): int(info["order"]) for info in ordered}

    selected: list[str] = []
    for score_val, _pos, aid in scored_sorted:
        if aid in selected:
            continue

        # Add the anchor itself.
        selected.append(aid)

        # Expand neighbors for multi-anchor definitions.
        o = order_by_id.get(aid)
        if o is None:
            continue

        # Expand before.
        for j in range(1, include_neighbors_before + 1):
            if len(selected) >= max_anchors:
                break
            idx = o - j
            if idx < 0:
                break
            nid = id_by_order[idx]
            if allowed_anchor_ids and nid not in allowed_anchor_ids:
                continue
            if nid not in selected:
                selected.append(nid)

        # Expand after.
        for j in range(1, include_neighbors_after + 1):
            if len(selected) >= max_anchors:
                break
            idx = o + j
            if idx >= len(id_by_order):
                break
            nid = id_by_order[idx]
            if allowed_anchor_ids and nid not in allowed_anchor_ids:
                continue
            if nid not in selected:
                selected.append(nid)

        if len(selected) >= max_anchors:
            break

        # If we hit a strong definition indicator ("shall mean"/"is defined as"/etc.),
        # prefer staying focused on that definition rather than mixing in usage mentions.
        if score_val >= 10:
            break

    # Preserve document order.
    selected_sorted = sorted(selected, key=lambda a: order_by_id.get(a, 10**9))
    return selected_sorted[:max_anchors]


def _build_context_blocks(
    canonical_text: str,
    anchors: Dict[str, Dict[str, Any]],
    anchor_ids: List[str],
) -> str:
    blocks: list[str] = []
    for aid in anchor_ids:
        info = anchors.get(aid)
        if not info:
            continue
        start = int(info["start"])
        end = int(info["end"])
        snippet = canonical_text[start:end].strip()
        if _is_noise_anchor_text(snippet):
            continue
        if not snippet:
            continue
        blocks.append(f"[[{aid}]]\n{snippet}")
    return "\n\n".join(blocks)


def _render_prompt(
    template: str,
    *,
    metric_name: str,
    contract_term_hint: str,
    search_tokens: List[str],
    contexts_block: str,
) -> str:
    out = template
    out = out.replace("{{METRIC_NAME}}", metric_name)
    out = out.replace("{{CONTRACT_TERM_HINT}}", contract_term_hint)
    out = out.replace("{{SEARCH_TOKENS}}", json.dumps(search_tokens))
    out = out.replace("{{CONTEXTS}}", contexts_block)
    return out


def _render_definition_finder_prompt(template: str, *, metrics_json: str, document: str) -> str:
    template = template.strip()
    missing: list[str] = []
    for key in ("{metrics_json}", "{document}"):
        if key not in template:
            missing.append(key)
    if missing:
        raise RuntimeError(f"Definition-finder prompt template missing required placeholders: {', '.join(missing)}")
    return template.replace("{metrics_json}", metrics_json).replace("{document}", document)


def _retry_prompt(
    base_prompt: str,
    *,
    attempt: int,
    error: str,
    previous_output: str,
) -> str:
    """Ask the model to repair invalid JSON / schema drift without re-doing the task."""

    trimmed_prev = previous_output.strip()
    if len(trimmed_prev) > 12_000:
        # Keep the most recent tail; errors are usually near the start but the model can still
        # re-emit the full object if needed. We avoid unbounded prompt growth on repeated failures.
        trimmed_prev = trimmed_prev[-12_000:]

    return (
        f"{base_prompt.strip()}\n\n"
        f"=== REPAIR MODE (attempt {attempt}) ===\n"
        "Your previous output was invalid JSON or failed schema validation.\n"
        f"ERROR: {error}\n\n"
        "Return corrected JSON only.\n"
        "- Do NOT include code fences.\n"
        "- Keep the same content unless required to satisfy the schema.\n"
        "- Ensure the top-level object has exactly: schema_version, definitions, unresolved_dependencies.\n"
        "- Ensure definitions[0] has exactly these keys:\n"
        "  name, contract_term, definition_verbatim, expression_ast, input_terms, clauses,\n"
        "  source_refs, needs_more_context, confidence, notes\n"
        "- Do NOT include unresolved_dependencies inside definitions[0] (top-level only).\n"
        "- Do NOT include literal newlines inside JSON strings.\n"
        "- Replace any line breaks in copied text with a single space (do not emit backslash-n sequences).\n\n"
        "- Do NOT “clean up” the source text. If the context contains duplicated words (e.g., \"Agent Agents\"), keep them.\n\n"
        "- If grounding fails, you likely dropped/changed a small word (e.g., \"any\"). Re-copy the definition_verbatim EXACTLY.\n\n"
        "Previous output:\n"
        f"{trimmed_prev}\n"
    )


async def _call_gateway(
    *,
    client: Any,
    prompt: str,
    model: str,
    temperature: float,
    reasoning: str | None,
) -> str:
    reasoning_payload = {"effort": reasoning} if reasoning else None
    result = await client.complete_response(
        model=model,
        input_messages=[{"role": "user", "content": prompt}],
        reasoning=reasoning_payload,
        temperature=temperature,
        max_output_tokens=None,
        metadata=None,
    )
    if isinstance(result, dict):
        return result.get("text") or ""
    return str(result)


async def _call_gateway_with_retry(
    *,
    client: Any,
    prompt: str,
    model: str,
    temperature: float,
    reasoning: str | None,
    attempts: int = 3,
) -> str:
    last_exc: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            text = await _call_gateway(
                client=client,
                prompt=prompt,
                model=model,
                temperature=temperature,
                reasoning=reasoning,
            )
            if text and text.strip():
                return text
            raise RuntimeError("Gateway returned empty response text.")
        except Exception as exc:  # pragma: no cover - network/errors
            last_exc = exc
            await asyncio.sleep(1.5 * attempt)
    raise last_exc if last_exc else RuntimeError("Gateway call failed with unknown error.")


def _safe_slug(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "_", s).strip("_") or "metric"


def _is_noise_anchor_text(snippet: str) -> bool:
    """Return True for anchors that are clearly non-content artifacts.

    Some canonical.txt files include standalone anchors for page numbers and footer/version
    strings. Those can break substring grounding if the model correctly omits them when
    copying a definition. We keep this conservative and pattern-based.
    """

    s = snippet.strip()
    if not s:
        return True
    # Standalone page numbers like "20"
    if re.fullmatch(r"\d{1,4}", s):
        return True
    # Footer/version markers like "#96465179v1" or "#95537764v15 AMERICAS/2023466857.21"
    if re.fullmatch(r"#\d+v\d+(?:\s+AMERICAS/\d+(?:\.\d+)?)?", s):
        return True
    return False


def _filter_noise_anchor_ids(
    *,
    canonical_text: str,
    anchors: Dict[str, Dict[str, Any]],
    anchor_ids: List[str],
) -> List[str]:
    kept: list[str] = []
    for aid in anchor_ids:
        info = anchors.get(aid)
        if not info:
            continue
        start = int(info["start"])
        end = int(info["end"])
        snippet = canonical_text[start:end]
        if _is_noise_anchor_text(snippet):
            continue
        kept.append(aid)
    return kept


_QUOTE_TRANSLATION = str.maketrans(
    {
        "“": '"',
        "”": '"',
        "„": '"',
        "‟": '"',
        "‘": "'",
        "’": "'",
        "‚": "'",
        "‛": "'",
    }
)


def _norm_for_substring(s: str) -> str:
    """Normalize text for robust substring grounding checks.

    We keep this intentionally conservative: it should reduce false positives caused by
    typographic quotes and hard line breaks, without masking true hallucinations.
    """

    s2 = s.replace("\u00a0", " ").translate(_QUOTE_TRANSLATION)
    # Some model outputs include literal backslash escapes (e.g., "\\n") inside strings.
    # Treat those as whitespace for grounding checks.
    s2 = s2.replace("\\n", " ").replace("\\r", " ").replace("\\t", " ")
    s2 = re.sub(r"\s+", " ", s2).strip()

    # Treat quoting as non-semantic for substring grounding checks.
    #
    # Important: documents sometimes contain *unbalanced* quotes (e.g., `Interest Expense" shall mean ...`),
    # which can cause naive quote-handling to join words across the quote boundary and trigger false
    # substring mismatches (e.g., `Expense" shall` -> `Expenseshall`).
    #
    # Use spaces for double-quotes to preserve token boundaries; strip apostrophes to keep possessives
    # from breaking matches (e.g., `Borrower's` vs `Borrowers`).
    s2 = s2.replace('"', " ").replace("'", "")
    s2 = re.sub(r"\s+", " ", s2).strip()

    # Treat common punctuation as non-semantic for substring grounding checks.
    # This helps avoid brittle failures like `7.01(b)).` vs `7.01(b) ).` when text is split across anchors.
    for ch in "()[]{}":
        s2 = s2.replace(ch, " ")
    for ch in ",;:.":
        s2 = s2.replace(ch, " ")
    s2 = re.sub(r"\s+", " ", s2).strip()
    return s2


def _validate_compiler_output(
    *,
    item_id: str,
    metric_name: str,
    raw_json: Dict[str, Any],
    allowed_anchor_ids: set[str],
    contexts_text: str,
) -> Tuple[Dict[str, Any], List[str]]:
    """Light validation: raise on unusable structure; return (cleaned_doc, issues)."""

    issues: list[str] = []
    if not isinstance(raw_json, dict):
        raise RuntimeError("compiler output must be a JSON object")

    def _contexts_plain_text(block: str) -> str:
        """Strip marker-only lines so multi-block definitions can be checked as contiguous text."""

        lines: list[str] = []
        for line in block.splitlines():
            s = line.strip()
            if not s:
                lines.append("")
                continue
            # Drop our injected context markers.
            if re.fullmatch(r"\[\[[^\]]+\]\]", s):
                continue
            # Drop table sentinels to avoid blocking substring checks across table blocks.
            if s in {"[[TABLE]]", "[[/TABLE]]"}:
                continue
            # Drop common page-number artifacts.
            if re.fullmatch(r"\d{1,4}", s):
                continue
            if re.fullmatch(r"-\d+-", s):
                continue
            if re.fullmatch(r"[A-Z]-\d+", s):
                continue
            # Drop common hash/version artifacts that are not contract text.
            if s.startswith("#") and len(s) < 96:
                continue
            lines.append(line)
        return "\n".join(lines)

    # Tolerate missing unresolved_dependencies (we normalize it deterministically later),
    # but fail on extra keys to keep the artifact shape stable.
    allowed_top_keys = {"metrics", "unresolved_dependencies"}
    extra = sorted(set(raw_json.keys()) - allowed_top_keys)
    if extra:
        raise RuntimeError(f"compiler output keys must only include metrics, unresolved_dependencies (extra={extra})")
    if "metrics" not in raw_json:
        raise RuntimeError("compiler output missing required key: metrics")
    if "unresolved_dependencies" not in raw_json:
        raw_json["unresolved_dependencies"] = []
        issues.append("missing_unresolved_dependencies_filled")

    metrics = raw_json.get("metrics")
    if not isinstance(metrics, list) or len(metrics) != 1 or not isinstance(metrics[0], dict):
        raise RuntimeError("compiler output metrics must be a list with exactly one metric object")

    m = metrics[0]
    if m.get("name") != metric_name:
        raise RuntimeError(f"compiler metric.name must equal input METRIC_NAME {metric_name!r}; got {m.get('name')!r}")

    contract_term = m.get("contract_term")
    if contract_term is not None and not isinstance(contract_term, str):
        raise RuntimeError("compiler metric.contract_term must be string or null")

    # Anchor refs validation (soft: unknown anchors => issue; but shape must be list[str]).
    src_refs = m.get("source_refs")
    if not isinstance(src_refs, list) or not all(isinstance(a, str) for a in src_refs):
        raise RuntimeError("compiler metric.source_refs must be a list of strings")
    unknown = [a for a in src_refs if a not in allowed_anchor_ids]
    if unknown:
        raise RuntimeError(
            "compiler metric.source_refs contains anchors not present in the provided CONTEXT BLOCKS "
            f"(examples={unknown[:10]})"
        )

    # Definition substring check (soft).
    dv = m.get("definition_verbatim")
    if dv is not None:
        if not isinstance(dv, str):
            raise RuntimeError("compiler metric.definition_verbatim must be string or null")
        if dv.strip():
            if not src_refs:
                raise RuntimeError("compiler metric.source_refs must be non-empty when definition_verbatim is provided")

            def _rebuild_from_source_refs() -> str | None:
                # Parse contexts_text into (anchor_id -> (position, snippet_text)).
                pos_by_id: dict[str, int] = {}
                txt_by_id: dict[str, str] = {}
                cur: str | None = None
                buf: list[str] = []
                pos = -1
                for line in contexts_text.splitlines():
                    s = line.strip()
                    m_header = re.fullmatch(r"\[\[([^\]]+)\]\]", s)
                    if m_header:
                        if cur is not None:
                            txt_by_id[cur] = "\n".join(buf).strip()
                        pos += 1
                        cur = m_header.group(1)
                        pos_by_id[cur] = pos
                        buf = []
                        continue
                    if cur is not None:
                        buf.append(line)
                if cur is not None:
                    txt_by_id[cur] = "\n".join(buf).strip()

                def _is_noise(snippet: str) -> bool:
                    s2 = snippet.strip()
                    return bool(
                        not s2
                        or re.fullmatch(r"\d{1,4}", s2)
                        or re.fullmatch(r"-\d+-", s2)
                        or re.fullmatch(r"[A-Z]-\d+", s2)
                    )

                parts: list[tuple[int, str]] = []
                for aid in src_refs:
                    if aid not in txt_by_id:
                        continue
                    snippet = txt_by_id[aid]
                    if _is_noise(snippet):
                        continue
                    parts.append((pos_by_id.get(aid, 10**9), snippet))

                if not parts:
                    return None

                joined = " ".join(
                    re.sub(r"\s+", " ", t).strip() for _p, t in sorted(parts, key=lambda x: x[0]) if t.strip()
                )
                joined = re.sub(r"\s+", " ", joined).strip()
                return joined if joined else None

            dv_norm = _norm_for_substring(dv)
            ctx_norm = _norm_for_substring(_contexts_plain_text(contexts_text))
            if dv_norm and dv_norm not in ctx_norm:
                # If the model referenced anchors, we can reconstruct an exact verbatim definition from the
                # selected anchor spans. This avoids false "not substring" issues due to minor copying drift
                # and makes the output more auditable/deterministic.
                rebuilt = _rebuild_from_source_refs()
                if rebuilt:
                    rebuilt_norm = _norm_for_substring(rebuilt)
                    if rebuilt_norm and rebuilt_norm in ctx_norm:
                        m["definition_verbatim"] = rebuilt
                        dv = rebuilt
                    else:
                        issues.append("definition_verbatim_not_substring_of_contexts")
                else:
                    issues.append("definition_verbatim_not_substring_of_contexts")

            # Ensure the definition text is actually attributable to the cited source_refs anchors.
            # (We allow multi-anchor definitions but require that source_refs cover the emitted verbatim.)
            dv_norm2 = _norm_for_substring(str(m.get("definition_verbatim") or ""))
            if dv_norm2:
                # Reuse the same reconstruction logic (but without requiring a mismatch first).
                def _src_text() -> str:
                    txt_by_id: dict[str, str] = {}
                    cur: str | None = None
                    buf: list[str] = []
                    for line in contexts_text.splitlines():
                        s = line.strip()
                        m_header = re.fullmatch(r"\[\[([^\]]+)\]\]", s)
                        if m_header:
                            if cur is not None:
                                txt_by_id[cur] = "\n".join(buf).strip()
                            cur = m_header.group(1)
                            buf = []
                            continue
                        if cur is not None:
                            buf.append(line)
                    if cur is not None:
                        txt_by_id[cur] = "\n".join(buf).strip()

                    parts: list[str] = []
                    for aid in src_refs:
                        snippet = txt_by_id.get(aid, "").strip()
                        if not snippet:
                            continue
                        # Drop obvious page-number artifacts inside the cited span.
                        if re.fullmatch(r"\d{1,4}", snippet):
                            continue
                        if re.fullmatch(r"-\d+-", snippet):
                            continue
                        if re.fullmatch(r"[A-Z]-\d+", snippet):
                            continue
                        parts.append(snippet)
                    return re.sub(r"\s+", " ", " ".join(parts)).strip()

                src_norm = _norm_for_substring(_src_text())
                if dv_norm2 and dv_norm2 not in src_norm:
                    # If the model paraphrased/reformatted the definition_verbatim but provided the right
                    # source_refs, prefer rewriting the verbatim definition deterministically from source_refs.
                    # This keeps outputs auditable and avoids brittle failures on formatting differences.
                    rebuilt = _rebuild_from_source_refs()
                    if rebuilt:
                        m["definition_verbatim"] = rebuilt
                        dv = rebuilt
                        dv_norm2 = _norm_for_substring(rebuilt)
                        src_norm = _norm_for_substring(_src_text())

                if dv_norm2 and dv_norm2 not in src_norm:
                    raise RuntimeError(
                        "compiler metric.definition_verbatim must be a substring of the concatenated source_refs text "
                        "(source_refs appears incomplete or mismatched)"
                    )

                # Deterministic continuation completion: if the definition ends with punctuation that strongly
                # suggests truncation (e.g., trailing comma) and the next anchors in the contexts look like
                # continuation (not a new definition), extend definition_verbatim and source_refs forward.
                # This is especially common when page-number artifacts split a definition across anchors.
                dv_now = str(m.get("definition_verbatim") or "").strip()
                if dv_now and dv_now.endswith((",", ";")):
                    # Build ordered (anchor_id, snippet) list from contexts_text.
                    ordered: list[tuple[str, str]] = []
                    cur = None
                    buf = []
                    for line in contexts_text.splitlines():
                        s = line.strip()
                        m_header = re.fullmatch(r"\[\[([^\]]+)\]\]", s)
                        if m_header:
                            if cur is not None:
                                ordered.append((cur, "\n".join(buf).strip()))
                            cur = m_header.group(1)
                            buf = []
                            continue
                        if cur is not None:
                            buf.append(line)
                    if cur is not None:
                        ordered.append((cur, "\n".join(buf).strip()))

                    pos = {aid: idx for idx, (aid, _txt) in enumerate(ordered)}

                    def _is_noise(snippet: str) -> bool:
                        s2 = snippet.strip()
                        return bool(
                            not s2
                            or re.fullmatch(r"\d{1,4}", s2)
                            or re.fullmatch(r"-\d+-", s2)
                            or re.fullmatch(r"[A-Z]-\d+", s2)
                        )

                    def _starts_new_definition(snippet: str) -> bool:
                        s2 = snippet.strip().lower()
                        if not s2:
                            return False
                        # Common definitional openers: quoted term + "means/shall mean/is defined as/has the meaning".
                        if s2.startswith(('"', "“")):
                            return bool(
                                re.search(r"\bshall\s+mean\b", s2)
                                or re.search(r"\bmeans\b", s2)
                                or re.search(r"\bis\s+defined\s+as\b", s2)
                                or re.search(r"\bhas\s+the\s+meaning\b", s2)
                            )
                        return False

                    # Expand forward from the last cited anchor (by doc order) to capture continuation.
                    cited = [a for a in src_refs if a in pos]
                    if cited:
                        last = max(cited, key=lambda a: pos[a])
                        start_idx = pos[last] + 1
                        added: list[str] = []
                        for _idx in range(start_idx, min(len(ordered), start_idx + 8)):
                            aid, snippet = ordered[_idx]
                            if _is_noise(snippet):
                                added.append(aid)
                                continue
                            if _starts_new_definition(snippet):
                                break
                            added.append(aid)
                            # Stop after a small continuation window (in anchors), to avoid drifting into unrelated text.
                            if len([a for a in added if not _is_noise(dict(ordered).get(a, ""))]) >= 3:
                                break

                        # If we added anything, extend source_refs + rebuild definition_verbatim deterministically.
                        if added:
                            # Preserve existing refs, then append new ones in doc order.
                            new_refs = list(src_refs)
                            for aid in added:
                                if aid not in new_refs:
                                    new_refs.append(aid)
                            # Rebuild definition text from the expanded refs (dropping noise anchors).
                            parts: list[str] = []
                            for aid in new_refs:
                                snip = dict(ordered).get(aid, "").strip()
                                if _is_noise(snip):
                                    continue
                                parts.append(snip)
                            rebuilt = re.sub(r"\s+", " ", " ".join(parts)).strip()
                            if rebuilt:
                                m["source_refs"] = new_refs
                                m["definition_verbatim"] = rebuilt
                                dv = rebuilt

    # definition_parsed structural validation (hard; downstream code expects stable keys).
    dp = m.get("definition_parsed")
    if dp is None:
        # Be tolerant to minor schema omissions; downstream expects the object, so we fill.
        dp = {}
        m["definition_parsed"] = dp
        issues.append("missing_definition_parsed_filled")
    if not isinstance(dp, dict):
        raise RuntimeError("compiler metric.definition_parsed must be an object")
    expected_dp_keys = {"numerator", "denominator", "components", "adjustments", "measurement_period"}
    extra_dp = sorted(set(dp.keys()) - expected_dp_keys)
    if extra_dp:
        raise RuntimeError(
            "compiler metric.definition_parsed contains unexpected keys "
            f"{extra_dp}"
        )
    missing_dp = sorted(expected_dp_keys - set(dp.keys()))
    if missing_dp:
        # Fill missing keys with safe defaults; this is deterministic and reduces brittleness
        # without changing the semantics of extracted text.
        for k in missing_dp:
            if k in {"components", "adjustments"}:
                dp[k] = []
            else:
                dp[k] = None
        issues.append(f"definition_parsed_missing_keys_filled: {missing_dp}")
    for k in ("numerator", "denominator", "measurement_period"):
        if dp.get(k) is not None and not isinstance(dp.get(k), str):
            raise RuntimeError(f"compiler metric.definition_parsed.{k} must be string or null")
    for k in ("components", "adjustments"):
        v = dp.get(k)
        if not isinstance(v, list) or not all(isinstance(x, str) for x in v):
            raise RuntimeError(f"compiler metric.definition_parsed.{k} must be list[str]")

    requires_non_compustat = m.get("requires_non_compustat")
    if not isinstance(requires_non_compustat, bool):
        raise RuntimeError("compiler metric.requires_non_compustat must be boolean")

    # Common model typo: non_compustust_reason -> non_compustat_reason.
    if m.get("non_compustat_reason") is None and isinstance(m.get("non_compustust_reason"), str):
        m["non_compustat_reason"] = m.get("non_compustust_reason")
        try:
            del m["non_compustust_reason"]
        except Exception:
            pass
        issues.append("non_compustat_reason_key_typo_corrected")

    non_compustat_reason = m.get("non_compustat_reason")
    if non_compustat_reason is not None and not isinstance(non_compustat_reason, str):
        raise RuntimeError("compiler metric.non_compustat_reason must be string or null")

    blocking_terms = m.get("blocking_terms")
    if not isinstance(blocking_terms, list) or not all(isinstance(t, str) for t in blocking_terms):
        raise RuntimeError("compiler metric.blocking_terms must be list[str]")

    # New: typed dependencies (calculation vs context), formula AST, and non-math rules.
    # These fields make the definition representation more robust than numerator/denominator alone.
    dependencies = m.get("dependencies")
    if dependencies is None:
        dependencies = []
        m["dependencies"] = dependencies
        issues.append("missing_dependencies_filled")
    if not isinstance(dependencies, list) or not all(isinstance(d, dict) for d in dependencies):
        raise RuntimeError("compiler metric.dependencies must be list[object]")
    for idx, d in enumerate(dependencies):
        term = d.get("term")
        kind = d.get("kind")
        if term is not None and not isinstance(term, str):
            raise RuntimeError(f"compiler metric.dependencies[{idx}].term must be string")
        if kind is not None and not isinstance(kind, str):
            raise RuntimeError(f"compiler metric.dependencies[{idx}].kind must be string")
        if isinstance(kind, str) and kind not in {"calculation", "context"}:
            issues.append(f"dependency_kind_invalid[{idx}]")

    rules = m.get("rules")
    if rules is None:
        rules = []
        m["rules"] = rules
        issues.append("missing_rules_filled")
    if not isinstance(rules, list) or not all(isinstance(r, dict) for r in rules):
        raise RuntimeError("compiler metric.rules must be list[object]")
    for idx, r in enumerate(rules):
        kind = r.get("kind")
        text = r.get("text")
        if kind is not None and not isinstance(kind, str):
            raise RuntimeError(f"compiler metric.rules[{idx}].kind must be string")
        if text is not None and not isinstance(text, str):
            raise RuntimeError(f"compiler metric.rules[{idx}].text must be string")

    formula_ast = m.get("formula_ast")
    if formula_ast is not None and not isinstance(formula_ast, dict):
        raise RuntimeError("compiler metric.formula_ast must be object or null")

    confidence = m.get("confidence")
    if confidence not in {"high", "medium", "low"}:
        raise RuntimeError("compiler metric.confidence must be 'high'|'medium'|'low'")

    notes = m.get("notes")
    if not isinstance(notes, list) or not all(isinstance(n, str) for n in notes):
        raise RuntimeError("compiler metric.notes must be list[str]")

    # Substring grounding checks for parsed fields (soft). Use whitespace-normalized comparisons to tolerate line breaks.
    if isinstance(dv, str) and dv.strip():
        dv_hay = _norm_for_substring(dv).lower()

        for k in ("numerator", "denominator", "measurement_period"):
            v = dp.get(k)
            if isinstance(v, str) and v.strip():
                if _norm_for_substring(v).lower() not in dv_hay:
                    issues.append(f"definition_parsed_{k}_not_substring")

        for k in ("components", "adjustments"):
            for idx, v in enumerate(dp.get(k) or []):
                if isinstance(v, str) and v.strip():
                    if _norm_for_substring(v).lower() not in dv_hay:
                        issues.append(f"definition_parsed_{k}_not_substring[{idx}]")

        for idx, t in enumerate(blocking_terms):
            if t and t.strip():
                if _norm_for_substring(t).lower() not in dv_hay:
                    issues.append(f"blocking_term_not_substring[{idx}]")

        for idx, d in enumerate(dependencies):
            term = d.get("term") if isinstance(d, dict) else None
            if isinstance(term, str) and term.strip():
                if _norm_for_substring(term).lower() not in dv_hay:
                    issues.append(f"dependency_term_not_substring[{idx}]")

        for idx, r in enumerate(rules):
            text = r.get("text") if isinstance(r, dict) else None
            if isinstance(text, str) and text.strip():
                if _norm_for_substring(text).lower() not in dv_hay:
                    issues.append(f"rule_text_not_substring[{idx}]")

    # Compustat allowlist check (soft).
    allow = {
        "dlttq",
        "dlcq",
        "dd1q",
        "npq",
        "ltq",
        "xintq",
        "cheq",
        "actq",
        "lctq",
        "revtq",
        "oibdpq",
        "oiadpq",
        "dpq",
        "niq",
        "saleq",
        "atq",
        "seqq",
        "capxq",
        "txtq",
    }
    vars_ = m.get("compustat_variables")
    if not isinstance(vars_, list) or not all(isinstance(v, str) for v in vars_):
        raise RuntimeError("compiler metric.compustat_variables must be a list of strings")
    bad_vars = [v for v in vars_ if v not in allow]
    if bad_vars:
        issues.append(f"unknown_compustat_variables: {bad_vars[:10]}")

    formula = m.get("formula_compustat")
    if formula is not None and not isinstance(formula, str):
        raise RuntimeError("compiler metric.formula_compustat must be string or null")

    if formula is None and requires_non_compustat is False:
        raise RuntimeError("compiler metric.requires_non_compustat must be true when formula_compustat is null")

    # Guardrail: if the model provided a Compustat formula but the definition clearly includes bespoke
    # adjustments or other defined terms, keep the formula (it may still be a useful approximation) but
    # force requires_non_compustat=true so downstream consumers don't treat it as exact.
    calc_dep_terms = [
        d.get("term")
        for d in dependencies
        if isinstance(d, dict) and d.get("kind") == "calculation" and isinstance(d.get("term"), str)
    ]
    has_bespoke_deps = bool(calc_dep_terms) if dependencies else bool(blocking_terms)
    if (
        isinstance(formula, str)
        and formula.strip()
        and requires_non_compustat is False
        and (
            has_bespoke_deps
            or (isinstance(dp, dict) and bool((dp.get("adjustments") or [])))
        )
    ):
        m["requires_non_compustat"] = True
        requires_non_compustat = True
        if not non_compustat_reason:
            m["non_compustat_reason"] = (
                "Compustat formula provided as an approximation; definition includes bespoke adjustments and/or other defined terms."
            )
        issues.append("compustat_formula_flagged_as_non_compustat")

    # Deterministic fallback: if the model does not populate blocking_terms, but it also
    # cannot provide a Compustat formula (and we *do* have a grounded definition), derive
    # blocking terms from grounded parsed components. This makes the output usable as a
    # driver for a second-pass "define the blockers" workflow (PI feedback).
    if (
        isinstance(dv, str)
        and dv.strip()
        and formula is None
        and requires_non_compustat is True
        and (not blocking_terms)
        and isinstance(dp, dict)
    ):
        dv_hay = _norm_for_substring(dv).lower()
        candidates: list[str] = []

        def _maybe_add(val: str) -> None:
            v = val.strip()
            if not v:
                return
            # Strip common enumeration prefixes like "(i)" / "(ii)" / "(a)" / "a)" that appear in ratio parts.
            v = re.sub(r"^\(\s*(?:[ivx]+|[a-z]|\d+)\s*\)\s*", "", v, flags=re.IGNORECASE).strip()
            v = re.sub(r"^(?:[a-z]|\d+)\)\s+", "", v, flags=re.IGNORECASE).strip()
            v = re.sub(r"^(?:[a-z]|\d+)\.\s+", "", v, flags=re.IGNORECASE).strip()
            if not v:
                return
            if _norm_for_substring(v).lower() not in dv_hay:
                return
            candidates.append(v)

        # Prefer ratio parts; then include other key components/adjustments.
        for v in (dp.get("numerator"), dp.get("denominator")):
            if isinstance(v, str):
                _maybe_add(v)
        for v in dp.get("components") or []:
            if isinstance(v, str):
                _maybe_add(v)
        for v in dp.get("adjustments") or []:
            if isinstance(v, str) and any(ch.isupper() for ch in v):
                _maybe_add(v)

        # Only set if we found anything grounded.
        if candidates:
            blocking_terms = candidates
            m["blocking_terms"] = candidates

    # Normalize unresolved deps deterministically from typed dependencies (preferred) or blocking_terms
    # (fallback for older prompts).
    def _norm(s: str) -> str:
        s2 = s.strip()
        # Strip common quote characters (ASCII + curly quotes).
        s2 = s2.strip("\"'“”‘’")
        # Strip trailing punctuation that often appears in copied term tokens.
        s2 = s2.strip(" ,.;:()[]{}")
        return re.sub(r"\s+", " ", s2).strip().lower()

    def _strip_leading_articles(s: str) -> str:
        return re.sub(r"^\s*(?:the|a|an)\s+", "", s.strip(), flags=re.IGNORECASE).strip()

    self_terms = {_norm(metric_name), _norm(_strip_leading_articles(metric_name))}
    if isinstance(contract_term, str) and contract_term.strip():
        self_terms.add(_norm(contract_term))
        self_terms.add(_norm(_strip_leading_articles(contract_term)))

    def _collect_formula_ast_terms(node: Any) -> list[str]:
        out: list[str] = []
        stack: list[Any] = [node]
        while stack:
            cur = stack.pop()
            if not isinstance(cur, dict):
                continue
            t = cur.get("type")
            if t == "term":
                term = cur.get("term")
                if isinstance(term, str) and term.strip():
                    out.append(term.strip())
                continue
            if t == "binary":
                stack.append(cur.get("lhs"))
                stack.append(cur.get("rhs"))
                continue
            if t == "unary":
                stack.append(cur.get("arg"))
                continue
            if t == "function":
                args = cur.get("args")
                if isinstance(args, list):
                    stack.extend(args)
                continue
        # Stable de-dupe, preserve order.
        seen: set[str] = set()
        return [t for t in out if not (t in seen or seen.add(t))]

    def _normalize_blocking_term_display(term: str) -> str | None:
        """Heuristic normalization for blocking terms.

        Goal: keep likely defined/bespoke terms (PI feedback) and drop obvious sentence fragments / table labels.
        This intentionally errs on the side of being conservative (drop only clearly non-term patterns).
        """

        s = term.strip()
        if not s:
            return None

        # Drop bullet-style adjustment fragments (not defined terms).
        if re.match(r"^[-–—•]\s*", s):
            return None

        # Strip common quote characters (ASCII + curly quotes).
        s = s.strip("\"'“”‘’").strip()

        # Strip enumeration prefixes like "(i)" / "(ii)".
        s = re.sub(r"^\(\s*(?:[ivx]+|[a-z]|\d+)\s*\)\s*", "", s, flags=re.IGNORECASE).strip()
        s = re.sub(r"^(?:[a-z]|\d+)\)\s+", "", s, flags=re.IGNORECASE).strip()
        s = re.sub(r"^(?:[a-z]|\d+)\.\s+", "", s, flags=re.IGNORECASE).strip()

        # Strip leading articles.
        s = _strip_leading_articles(s)

        # Strip common leading modifiers that are not part of the defined term.
        s = re.sub(r"^(?:initial|subsequent|each|any|such)\s+", "", s, flags=re.IGNORECASE).strip()

        # Drop obvious comparator/row-label phrases (often tier labels).
        if re.search(r"\b(equal|greater|less)\b", s, flags=re.IGNORECASE) and re.search(r"\d", s):
            return None

        # Drop ratio-phrases; we want the underlying defined terms (EBITDA, Total Debt, etc.).
        if s.lower().startswith("ratio of "):
            return None

        # Strip trailing qualifiers that frequently appear in ratio parts/components.
        trailer_patterns = [
            r"\s+as\s+of\b.*$",
            r"\s+based\s+on\b.*$",
            r"\s+to\s+the\s+extent\b.*$",
            r"\s+set\s+forth\b.*$",
            r"\s+achieved\b.*$",
            r"\s+less\b.*$",
            r"\s+for\s+such\s+period\b.*$",
            r"\s+for\s+the\s+fiscal\s+period\b.*$",
            r"\s+for\s+the\s+period\b.*$",
        ]
        for pat in trailer_patterns:
            s2 = re.sub(pat, "", s, flags=re.IGNORECASE).strip()
            if s2 and s2 != s:
                s = s2

        s = re.sub(r"\s+", " ", s).strip()
        s = s.strip(" ,.;:()[]{}").strip()
        if not s:
            return None
        if not any(ch.isalpha() for ch in s):
            return None
        if not any(ch.isupper() for ch in s):
            return None

        words = s.split()
        # Very long phrases are usually not defined terms; keep the pipeline focused.
        if len(words) > 12:
            return None
        # Sentence fragments tend to be lower-case openers with many words.
        if s[:1].islower() and len(words) > 4:
            return None

        return s

    def _normalize_formula_ast(node: Any) -> Any:
        """Normalize formula_ast so that 'term' nodes are reserved for likely defined terms.

        If a model emits long phrases / lower-case fragments as {"type":"term", ...}, downgrade those leaves
        to {"type":"opaque_text", ...}. This keeps formula_ast useful without turning generic phrases into
        dependency graph nodes.
        """

        if node is None:
            return None
        if not isinstance(node, dict):
            return node
        t = node.get("type")
        if t == "term":
            raw = node.get("term")
            if not isinstance(raw, str) or not raw.strip():
                return {"type": "opaque_text", "text": ""}
            normed = _normalize_blocking_term_display(raw)
            if not normed:
                issues.append("formula_ast_term_downgraded_to_opaque_text")
                return {"type": "opaque_text", "text": raw.strip()}
            if normed != raw.strip():
                issues.append("formula_ast_term_normalized")
            return {"type": "term", "term": normed}
        if t == "binary":
            out = dict(node)
            out["lhs"] = _normalize_formula_ast(node.get("lhs"))
            out["rhs"] = _normalize_formula_ast(node.get("rhs"))
            return out
        if t == "unary":
            out = dict(node)
            out["arg"] = _normalize_formula_ast(node.get("arg"))
            return out
        if t == "function":
            out = dict(node)
            args = node.get("args")
            if isinstance(args, list):
                out["args"] = [_normalize_formula_ast(a) for a in args]
            return out
        return node

    # Prefer typed dependencies. If missing/empty (older prompts), infer as "calculation" from blocking_terms.
    dep_by_key: dict[str, dict[str, str]] = {}
    dep_order: list[str] = []

    def _upsert_dep(term: str, kind: str, *, issue_tag: str | None = None) -> None:
        normed = _normalize_blocking_term_display(term)
        if not normed:
            return
        k = _norm(normed)
        if not k or k in self_terms:
            return
        if k not in dep_by_key:
            dep_by_key[k] = {"term": normed, "kind": kind}
            dep_order.append(k)
            if issue_tag:
                issues.append(issue_tag)
        else:
            # Upgrade context -> calculation if any evidence suggests it is a numeric input.
            if dep_by_key[k]["kind"] != "calculation" and kind == "calculation":
                dep_by_key[k]["kind"] = "calculation"
                if issue_tag:
                    issues.append(issue_tag)

    had_typed = False
    for idx, d in enumerate(dependencies):
        if not isinstance(d, dict):
            continue
        term = d.get("term")
        kind = d.get("kind")
        if not isinstance(term, str) or not term.strip():
            continue
        if kind not in {"calculation", "context"}:
            continue
        had_typed = True
        _upsert_dep(term, kind)

    if not had_typed:
        for t in blocking_terms:
            if not isinstance(t, str) or not t.strip():
                continue
            _upsert_dep(t, "calculation", issue_tag="dependencies_inferred_from_blocking_terms")

    # Ensure any term referenced in formula_ast is treated as a calculation dependency.
    formula_ast = _normalize_formula_ast(formula_ast)
    m["formula_ast"] = formula_ast
    for t in _collect_formula_ast_terms(formula_ast):
        _upsert_dep(t, "calculation", issue_tag="dependency_inferred_from_formula_ast")

    # Deterministic upgrade: if a dependency term appears inside the ratio parts, treat it as a numeric input.
    # This avoids misclassifications like marking "Unrestricted Cash" as context when it is explicitly
    # subtracted in the numerator.
    if isinstance(dp, dict):
        ratio_bits: list[str] = []
        for k2 in ("numerator", "denominator"):
            v2 = dp.get(k2)
            if isinstance(v2, str) and v2.strip():
                ratio_bits.append(v2)
        if ratio_bits:
            ratio_hay = _norm_for_substring(" ".join(ratio_bits)).lower()
            for k in dep_order:
                if dep_by_key.get(k, {}).get("kind") == "calculation":
                    continue
                term = dep_by_key.get(k, {}).get("term")
                if isinstance(term, str) and term.strip():
                    if _norm_for_substring(term).lower() in ratio_hay:
                        dep_by_key[k]["kind"] = "calculation"
                        issues.append("dependency_upgraded_to_calculation_from_ratio_part")

    cleaned_dependencies = [{"term": dep_by_key[k]["term"], "kind": dep_by_key[k]["kind"]} for k in dep_order]
    m["dependencies"] = cleaned_dependencies

    # Keep blocking_terms as a compatibility alias for "all dependency terms" (untyped).
    m["blocking_terms"] = [d["term"] for d in cleaned_dependencies]

    # unresolved_dependencies is the recursion driver; default to calculation terms only.
    unresolved_terms = [dep_by_key[k]["term"] for k in dep_order if dep_by_key[k]["kind"] == "calculation"]
    raw_json["unresolved_dependencies"] = [{"term": t, "referenced_by": [metric_name]} for t in unresolved_terms]

    return raw_json, issues


def _validate_compiler_output_v2_ast(
    *,
    item_id: str,
    metric_name: str,
    raw_json: Dict[str, Any],
    allowed_anchor_ids: set[str],
    contexts_text: str,
) -> Tuple[Dict[str, Any], List[str]]:
    """Strict validation for the definition compiler v2 AST schema (no repairs).

    Policy: rely only on what the LLM returns. This function MUST NOT "fix up" outputs:
    - No field promotion/moves
    - No dropping extra keys
    - No rebuilding text from anchors
    - No inferring missing dependencies

    It only validates structure + grounding. If validation fails, the caller should re-prompt/retry.
    """

    issues: list[str] = []

    if not isinstance(raw_json, dict):
        raise RuntimeError("compiler output must be a JSON object")

    def _contexts_plain_text(block: str) -> str:
        """Remove injected marker-only lines; keep everything else as-is."""

        out: list[str] = []
        for line in block.splitlines():
            s = line.strip()
            if re.fullmatch(r"\[\[[^\]]+\]\]", s):
                continue
            if s in {"[[TABLE]]", "[[/TABLE]]"}:
                continue
            out.append(line)
        return "\n".join(out)

    def _anchor_text_map(block: str) -> dict[str, str]:
        txt_by_id: dict[str, str] = {}
        cur: str | None = None
        buf: list[str] = []
        for line in block.splitlines():
            s = line.strip()
            m_header = re.fullmatch(r"\[\[([^\]]+)\]\]", s)
            if m_header:
                if cur is not None:
                    txt_by_id[cur] = "\n".join(buf).strip()
                cur = m_header.group(1)
                buf = []
                continue
            if cur is not None:
                buf.append(line)
        if cur is not None:
            txt_by_id[cur] = "\n".join(buf).strip()
        return txt_by_id

    allowed_top_keys = {"schema_version", "definitions", "unresolved_dependencies"}
    extra_top = sorted(set(raw_json.keys()) - allowed_top_keys)
    missing_top = sorted(allowed_top_keys - set(raw_json.keys()))
    if missing_top or extra_top:
        raise RuntimeError(
            "compiler output must have exactly keys {schema_version, definitions, unresolved_dependencies} "
            f"(missing={missing_top}, extra={extra_top})"
        )

    schema_version = raw_json.get("schema_version")
    if schema_version != "definition_compiler_v2_ast_v1":
        raise RuntimeError("compiler output schema_version must equal 'definition_compiler_v2_ast_v1'")

    defs = raw_json.get("definitions")
    if not isinstance(defs, list) or len(defs) != 1 or not isinstance(defs[0], dict):
        raise RuntimeError("compiler output definitions must be a list with exactly one definition object")
    d0 = defs[0]

    allowed_def_keys = {
        "name",
        "contract_term",
        "definition_verbatim",
        "expression_ast",
        "input_terms",
        "clauses",
        "source_refs",
        "needs_more_context",
        "confidence",
        "notes",
    }
    extra_def = sorted(set(d0.keys()) - allowed_def_keys)
    missing_def = sorted(allowed_def_keys - set(d0.keys()))
    if missing_def or extra_def:
        raise RuntimeError(
            "compiler definition object must have exactly the required keys "
            f"(missing={missing_def}, extra={extra_def})"
        )

    if d0.get("name") != metric_name:
        raise RuntimeError(
            f"compiler definition.name must equal input TARGET_TERM_NAME {metric_name!r}; got {d0.get('name')!r}"
        )

    contract_term = d0.get("contract_term")
    if contract_term is not None and not isinstance(contract_term, str):
        raise RuntimeError("compiler definition.contract_term must be string or null")
    if isinstance(contract_term, str) and contract_term.strip() and contract_term.strip()[0] in "\"'“”‘’":
        raise RuntimeError("compiler definition.contract_term must not include surrounding quote characters")

    dv = d0.get("definition_verbatim")
    if dv is not None and not isinstance(dv, str):
        raise RuntimeError("compiler definition.definition_verbatim must be string or null")

    expression_ast = d0.get("expression_ast")
    if expression_ast is not None and not isinstance(expression_ast, dict):
        raise RuntimeError("compiler definition.expression_ast must be object or null")

    input_terms = d0.get("input_terms")
    if not isinstance(input_terms, list) or not all(isinstance(t, str) for t in input_terms):
        raise RuntimeError("compiler definition.input_terms must be list[str]")

    clauses = d0.get("clauses")
    if not isinstance(clauses, list) or not all(isinstance(r, dict) for r in clauses):
        raise RuntimeError("compiler definition.clauses must be list[object]")

    src_refs = d0.get("source_refs")
    if not isinstance(src_refs, list) or not all(isinstance(a, str) for a in src_refs):
        raise RuntimeError("compiler definition.source_refs must be a list of strings")
    unknown = [a for a in src_refs if a not in allowed_anchor_ids]
    if unknown:
        raise RuntimeError(
            "compiler definition.source_refs contains anchors not present in the provided CONTEXT BLOCKS "
            f"(examples={unknown[:10]})"
        )

    needs_more_context = d0.get("needs_more_context")
    if not isinstance(needs_more_context, bool):
        raise RuntimeError("compiler definition.needs_more_context must be boolean")

    confidence = d0.get("confidence")
    if confidence not in {"high", "medium", "low"}:
        raise RuntimeError("compiler definition.confidence must be 'high'|'medium'|'low'")

    notes = d0.get("notes")
    if not isinstance(notes, list) or not all(isinstance(n, str) for n in notes):
        raise RuntimeError("compiler definition.notes must be list[str]")

    unresolved = raw_json.get("unresolved_dependencies")
    if not isinstance(unresolved, list) or not all(isinstance(x, dict) for x in unresolved):
        raise RuntimeError("compiler output unresolved_dependencies must be list[object]")
    for idx, dep in enumerate(unresolved):
        extra_dep = sorted(set(dep.keys()) - {"term", "referenced_by"})
        missing_dep = sorted({"term", "referenced_by"} - set(dep.keys()))
        if missing_dep or extra_dep:
            raise RuntimeError(
                f"compiler unresolved_dependencies[{idx}] must have exactly keys {{term, referenced_by}} "
                f"(missing={missing_dep}, extra={extra_dep})"
            )
        term = dep.get("term")
        referenced_by = dep.get("referenced_by")
        if not isinstance(term, str) or not term.strip():
            raise RuntimeError(f"compiler unresolved_dependencies[{idx}].term must be non-empty string")
        if not isinstance(referenced_by, list) or not all(isinstance(r, str) for r in referenced_by):
            raise RuntimeError(f"compiler unresolved_dependencies[{idx}].referenced_by must be list[str]")
        if referenced_by != [metric_name]:
            raise RuntimeError(
                f"compiler unresolved_dependencies[{idx}].referenced_by must equal [TARGET_TERM_NAME]={metric_name!r}"
            )

    # "No definition found" contract must be consistent (prompt rules).
    has_definition = isinstance(dv, str) and dv.strip()
    if not has_definition:
        if contract_term is not None:
            raise RuntimeError("when definition_verbatim is null, contract_term must be null")
        if src_refs:
            raise RuntimeError("when definition_verbatim is null, source_refs must be []")
        if expression_ast is not None:
            raise RuntimeError("when definition_verbatim is null, expression_ast must be null")
        if input_terms:
            raise RuntimeError("when definition_verbatim is null, input_terms must be []")
        if clauses:
            raise RuntimeError("when definition_verbatim is null, clauses must be []")
        if needs_more_context is not False:
            raise RuntimeError("when definition_verbatim is null, needs_more_context must be false")
        if confidence != "low":
            raise RuntimeError("when definition_verbatim is null, confidence must be 'low'")
        if unresolved:
            raise RuntimeError("when definition_verbatim is null, unresolved_dependencies must be []")
        return raw_json, issues

    # With a definition, contract_term must be present and source_refs must be non-empty.
    if not isinstance(contract_term, str) or not contract_term.strip():
        raise RuntimeError("when definition_verbatim is provided, contract_term must be a non-empty string")
    if not src_refs:
        raise RuntimeError("when definition_verbatim is provided, source_refs must be non-empty")

    dv_norm = _norm_for_substring(dv).lower()
    ctx_norm = _norm_for_substring(_contexts_plain_text(contexts_text)).lower()
    if dv_norm not in ctx_norm:
        diag = ""
        try:
            # Best-effort: align on the definitional start to show the first mismatch.
            start_phrase = _norm_for_substring(f"{contract_term} means").lower()
            idx = ctx_norm.find(start_phrase) if start_phrase else -1
            if idx != -1:
                ctx_tail = ctx_norm[idx:]
                max_len = min(len(dv_norm), len(ctx_tail))
                mismatch = None
                for i in range(max_len):
                    if dv_norm[i] != ctx_tail[i]:
                        mismatch = i
                        break
                if mismatch is not None:
                    a0 = max(0, mismatch - 60)
                    a1 = min(len(dv_norm), mismatch + 60)
                    b0 = max(0, mismatch - 60)
                    b1 = min(len(ctx_tail), mismatch + 60)
                    diag = f" (mismatch_near_dv={dv_norm[a0:a1]!r}, mismatch_near_ctx={ctx_tail[b0:b1]!r})"
        except Exception:
            diag = ""
        raise RuntimeError(
            "compiler definition.definition_verbatim must be a substring of provided contexts (after normalization)"
            + diag
        )

    txt_by_id = _anchor_text_map(contexts_text)
    src_text = " ".join(txt_by_id.get(a, "") for a in src_refs if a in txt_by_id).strip()
    if not src_text:
        raise RuntimeError("compiler definition.source_refs must reference anchors present in CONTEXT BLOCKS")
    if dv_norm not in _norm_for_substring(src_text).lower():
        raise RuntimeError(
            "compiler definition.definition_verbatim must be a substring of the concatenated source_refs text "
            "(source_refs appears incomplete or mismatched)"
        )

    # Substring grounding rules (prompt says MUST).
    for idx, t in enumerate(input_terms):
        if not isinstance(t, str) or not t.strip():
            raise RuntimeError(f"compiler definition.input_terms[{idx}] must be non-empty string")
        if not any(ch.isupper() for ch in t if ch.isalpha()):
            raise RuntimeError(
                f"compiler definition.input_terms[{idx}] must look like a defined term (must contain an uppercase letter)"
            )
        if _norm_for_substring(t).lower() not in dv_norm:
            raise RuntimeError(
                f"compiler definition.input_terms[{idx}] must be a verbatim substring of definition_verbatim"
            )

    for idx, r in enumerate(clauses):
        extra_clause = sorted(set(r.keys()) - {"kind", "text"})
        missing_clause = sorted({"kind", "text"} - set(r.keys()))
        if missing_clause or extra_clause:
            raise RuntimeError(
                f"compiler definition.clauses[{idx}] must have exactly keys {{kind, text}} "
                f"(missing={missing_clause}, extra={extra_clause})"
            )
        kind = r.get("kind")
        text = r.get("text")
        if not isinstance(kind, str) or not kind.strip():
            raise RuntimeError(f"compiler definition.clauses[{idx}].kind must be non-empty string")
        if not isinstance(text, str) or not text.strip():
            raise RuntimeError(f"compiler definition.clauses[{idx}].text must be non-empty string")
        if _norm_for_substring(text).lower() not in dv_norm:
            raise RuntimeError(
                f"compiler definition.clauses[{idx}].text must be a verbatim substring of definition_verbatim"
            )

    # expression_ast grounding + shape checks (best-effort but must be grounded when present).
    allowed_ops = {"+", "-", "*", "/"}
    allowed_fns = {"min", "max", "greater_of", "lesser_of"}
    input_term_set = {t for t in input_terms if isinstance(t, str)}

    def _validate_ast(node: Any, *, path: str) -> None:
        if node is None:
            return
        if not isinstance(node, dict):
            raise RuntimeError(f"compiler definition.expression_ast{path} must be object")
        node_type = node.get("type")
        if node_type not in {"term", "number", "op", "function", "opaque_text"}:
            raise RuntimeError(f"compiler definition.expression_ast{path}.type is invalid")
        if node_type == "term":
            term = node.get("term")
            if not isinstance(term, str) or not term.strip():
                raise RuntimeError(f"compiler definition.expression_ast{path}.term must be non-empty string")
            if term not in input_term_set:
                raise RuntimeError(
                    f"compiler definition.expression_ast{path}.term must be present in input_terms[] (exact match)"
                )
            if _norm_for_substring(term).lower() not in dv_norm:
                raise RuntimeError(
                    f"compiler definition.expression_ast{path}.term must be a verbatim substring of definition_verbatim"
                )
            return
        if node_type == "number":
            value = node.get("value")
            if not isinstance(value, (int, float)):
                raise RuntimeError(f"compiler definition.expression_ast{path}.value must be number")
            raw = node.get("raw")
            if raw is not None:
                if not isinstance(raw, str) or not raw.strip():
                    raise RuntimeError(f"compiler definition.expression_ast{path}.raw must be string or null")
                if _norm_for_substring(raw).lower() not in dv_norm:
                    raise RuntimeError(
                        f"compiler definition.expression_ast{path}.raw must be a verbatim substring of definition_verbatim"
                    )
            return
        if node_type == "op":
            op = node.get("op")
            args = node.get("args")
            if op not in allowed_ops:
                raise RuntimeError(f"compiler definition.expression_ast{path}.op is invalid")
            if not isinstance(args, list) or len(args) < 2:
                raise RuntimeError(f"compiler definition.expression_ast{path}.args must be list with >=2 elements")
            for i, a in enumerate(args):
                _validate_ast(a, path=f"{path}.args[{i}]")
            return
        if node_type == "function":
            name = node.get("name")
            args = node.get("args")
            if name not in allowed_fns:
                raise RuntimeError(f"compiler definition.expression_ast{path}.name is invalid")
            if not isinstance(args, list) or len(args) < 1:
                raise RuntimeError(f"compiler definition.expression_ast{path}.args must be list with >=1 element")
            for i, a in enumerate(args):
                _validate_ast(a, path=f"{path}.args[{i}]")
            return
        if node_type == "opaque_text":
            text = node.get("text")
            if not isinstance(text, str):
                raise RuntimeError(f"compiler definition.expression_ast{path}.text must be string")
            if text.strip() and _norm_for_substring(text).lower() not in dv_norm:
                raise RuntimeError(
                    f"compiler definition.expression_ast{path}.text must be a verbatim substring of definition_verbatim"
                )
            return

    _validate_ast(expression_ast, path="")

    # unresolved_dependencies must match input_terms (deduped) per prompt contract.
    dep_terms = [d.get("term") for d in unresolved]
    if not all(isinstance(t, str) for t in dep_terms):
        raise RuntimeError("compiler output unresolved_dependencies[].term must be string")
    if len(dep_terms) != len(set(dep_terms)):
        raise RuntimeError("compiler output unresolved_dependencies must be deduped (no duplicate term entries)")
    if set(dep_terms) != set(input_terms):
        raise RuntimeError("compiler output unresolved_dependencies must list the same set of terms as input_terms[]")

    return raw_json, issues


def run_definitions_compiler_v1(
    paths: Paths,
    item_ids: Iterable[str],
    *,
    qa_subdir: str,
    compiler_prompt_path: Path,
    definition_finder_prompt_path: Path | None = None,
    model: str | None = None,
    gateway_url: str | None = None,
    temperature: float = 0.0,
    reasoning: str | None = None,
    gateway_timeout: float | None = None,
    concurrency: int = 2,
    output_subdir: str = "compiled_v1",
    attempts: int = 3,
) -> None:
    """Compile structured metric definitions in one LLM call per metric.

    This is a standalone stage: it performs deterministic evidence selection (canonical text search + optional
    definitions section span) and then calls the LLM once per metric to produce a richer schema (verbatim
    definition + best-effort calculation AST + dependency terms).

    Note: Compustat mapping is intentionally a separate post-processing stage (see compustat_overlay_v1).
    """

    assert_exists(compiler_prompt_path, message=f"Definitions compiler prompt not found: {compiler_prompt_path}")

    # Enforce model + reasoning defaults for repeatability.
    model = REQUIRED_MODEL
    reasoning = REQUIRED_REASONING

    prompt_template = compiler_prompt_path.read_text(encoding="utf-8")
    prompt_digest = prompt_hash(compiler_prompt_path)

    # PI feedback #1 (Option B fallback): when the restricted-context attempt fails to find a definition,
    # fall back to a full-document "definition finder" (anchor selector) and re-run the same compiler prompt.
    definition_finder_prompt_path = definition_finder_prompt_path or (paths.root / "prompts" / "indexing_pricing_definitions_v1.txt")
    assert_exists(
        definition_finder_prompt_path,
        message=f"Definition-finder prompt not found: {definition_finder_prompt_path}",
    )
    definition_finder_template = definition_finder_prompt_path.read_text(encoding="utf-8")
    definition_finder_prompt_digest = prompt_hash(definition_finder_prompt_path)

    out_dir = paths.run_dir / "definitions_compiler_v1" / output_subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    for stale_name in ("errors.txt", "issues.txt"):
        stale = out_dir / stale_name
        if stale.exists():
            stale.unlink()

    items = list(item_ids)
    GatewayAgentClient = _ensure_gateway_client_async()
    sem = asyncio.Semaphore(max(1, concurrency))

    hard_errors: list[tuple[str, str]] = []
    soft_issues: list[tuple[str, str]] = []

    async def _process_metric(
        *,
        item_id: str,
        metric: MetricTarget,
        canonical_text: str,
        anchors: Dict[str, Dict[str, Any]],
        definitions_anchor_ids: List[str],
        definitions_span: Optional[Tuple[int, int]],
        pricing_union_anchor_ids: List[str],
        financial_covenant_anchor_ids: List[str],
        client: Any,
    ) -> None:
        async with sem:
            safe_metric = _safe_slug(metric.name)
            context_path = out_dir / f"{item_id}__{safe_metric}__contexts.txt"
            raw_path = out_dir / f"{item_id}__{safe_metric}__compile.raw.txt"
            parsed_path = out_dir / f"{item_id}__{safe_metric}__compiled.json"

            # Additional artifacts for the two-pass flow.
            pass1_context_path = out_dir / f"{item_id}__{safe_metric}__contexts.pass1.txt"
            pass1_raw_base = out_dir / f"{item_id}__{safe_metric}__compile.pass1.raw.txt"
            pass1_compiled_path = out_dir / f"{item_id}__{safe_metric}__compiled.pass1.json"

            pass2_context_path = out_dir / f"{item_id}__{safe_metric}__contexts.pass2.txt"
            pass2_raw_base = out_dir / f"{item_id}__{safe_metric}__compile.pass2.raw.txt"
            pass2_compiled_path = out_dir / f"{item_id}__{safe_metric}__compiled.pass2.json"

            finder_raw_path = out_dir / f"{item_id}__{safe_metric}__definition_finder.raw.txt"
            finder_json_path = out_dir / f"{item_id}__{safe_metric}__definition_finder.json"

            def _has_grounded_definition(doc: dict[str, Any]) -> bool:
                defs = doc.get("definitions")
                if not isinstance(defs, list) or not defs or not isinstance(defs[0], dict):
                    return False
                d0 = defs[0]
                dv = d0.get("definition_verbatim")
                srefs = d0.get("source_refs")
                return bool(isinstance(dv, str) and dv.strip() and isinstance(srefs, list) and any(isinstance(x, str) and x.strip() for x in srefs))

            async def _compile_with_contexts(
                *,
                pass_tag: str,
                selected_anchor_ids: list[str],
                contexts_block: str,
                context_out_path: Path,
                raw_base_path: Path,
                compiled_out_path: Path,
            ) -> dict[str, Any]:
                context_out_path.write_text(contexts_block + "\n", encoding="utf-8")

                rendered = _render_prompt(
                    prompt_template,
                    metric_name=metric.name,
                    contract_term_hint=metric.contract_term_hint,
                    search_tokens=metric.search_tokens,
                    contexts_block=contexts_block,
                )

                last_error = "unknown"
                last_raw = ""
                for attempt in range(1, max(1, attempts) + 1):
                    prompt = rendered
                    if attempt > 1:
                        prompt = _retry_prompt(
                            rendered,
                            attempt=attempt,
                            error=last_error,
                            previous_output=last_raw,
                        )

                    raw_attempt = raw_base_path if attempt == 1 else raw_base_path.with_name(
                        raw_base_path.name.replace(".raw.", f".raw.attempt{attempt}.")
                    )

                    try:
                        raw = await _call_gateway(
                            client=client,
                            prompt=prompt,
                            model=model,
                            temperature=temperature,
                            reasoning=reasoning,
                        )
                    except Exception as exc:  # pragma: no cover - network/errors
                        last_error = f"gateway error: {exc}"
                        await asyncio.sleep(1.5 * attempt)
                        continue

                    if not raw or not raw.strip():
                        last_error = "gateway returned empty response text"
                        await asyncio.sleep(1.5 * attempt)
                        continue

                    raw_attempt.write_text(raw, encoding="utf-8")
                    last_raw = raw

                    try:
                        parsed = json.loads(raw)
                    except json.JSONDecodeError as exc:
                        last_error = f"invalid JSON: {exc}"
                        await asyncio.sleep(1.5 * attempt)
                        continue

                    try:
                        cleaned, issues = _validate_compiler_output_v2_ast(
                            item_id=item_id,
                            metric_name=metric.name,
                            raw_json=parsed,
                            allowed_anchor_ids=set(selected_anchor_ids),
                            contexts_text=contexts_block,
                        )
                    except Exception as exc:
                        last_error = f"schema validation failed: {exc}"
                        await asyncio.sleep(1.5 * attempt)
                        continue

                    for iss in issues:
                        soft_issues.append((f"{item_id}::{metric.name}::{pass_tag}", iss))

                    # Always write the successful raw output to the base path for easier browsing.
                    raw_base_path.write_text(raw, encoding="utf-8")
                    compiled_out_path.write_text(json.dumps(cleaned, indent=2) + "\n", encoding="utf-8")
                    return cleaned

                raise RuntimeError(
                    f"compiler output could not be parsed/validated for metric={metric.name!r} after {attempts} attempt(s): {last_error}"
                )

            # Pass 1: use a *portion* of the document:
            # - identified pricing anchors (pricing/base_rate/spread/fee union)
            # - plus the full definitions section span (if detected)
            restricted_anchor_ids = list(
                dict.fromkeys(
                    (pricing_union_anchor_ids or []) + (financial_covenant_anchor_ids or []) + (definitions_anchor_ids or [])
                )
            )
            restricted_set: set[str] | None = set(restricted_anchor_ids) if restricted_anchor_ids else None

            # Evidence selection: term hits inside the restricted slice, preferring hits inside definitions span.
            def_start: int | None = None
            def_end: int | None = None
            if definitions_span is not None:
                def_start, def_end = int(definitions_span[0]), int(definitions_span[1])

            primary = metric.contract_term_hint.strip() if metric.contract_term_hint.strip() else metric.name
            hits = _find_term_hits(canonical_text, primary) if primary else []

            def _anchors_from_hits(hs: list[tuple[int, int]], *, allowed: set[str] | None) -> list[str]:
                ids = _match_hits_to_anchors(anchors, hs)
                if allowed is None:
                    return ids
                return [aid for aid in ids if aid in allowed]

            allowed_for_pass1: set[str] | None = restricted_set
            if hits and def_start is not None and def_end is not None and definitions_anchor_ids:
                hits_in_defs = [(s, e) for (s, e) in hits if def_start <= s < def_end]
                if hits_in_defs:
                    allowed_for_pass1 = set(definitions_anchor_ids)
                    hits = hits_in_defs

            if not hits:
                token_hits: list[tuple[int, int]] = []
                for tok in metric.search_tokens:
                    token_hits.extend(_find_term_hits(canonical_text, tok))
                if token_hits and def_start is not None and def_end is not None and definitions_anchor_ids:
                    token_hits_in_defs = [(s, e) for (s, e) in token_hits if def_start <= s < def_end]
                    if token_hits_in_defs:
                        allowed_for_pass1 = set(definitions_anchor_ids)
                        hits = token_hits_in_defs
                    else:
                        hits = token_hits
                else:
                    hits = token_hits

            pass1_doc: dict[str, Any] | None = None
            if hits:
                candidate_anchor_ids = _anchors_from_hits(hits, allowed=allowed_for_pass1)
                if candidate_anchor_ids:
                    selected_anchor_ids = _select_context_anchor_ids(
                        canonical_text=canonical_text,
                        anchors=anchors,
                        candidate_anchor_ids=candidate_anchor_ids,
                        contract_term=metric.contract_term_hint or metric.name,
                        search_tokens=metric.search_tokens,
                        max_anchors=64,
                        include_neighbors_before=2,
                        include_neighbors_after=4,
                        allowed_anchor_ids=allowed_for_pass1,
                    )
                    if selected_anchor_ids:
                        contexts_block = _build_context_blocks(canonical_text, anchors, selected_anchor_ids)
                        pass1_doc = await _compile_with_contexts(
                            pass_tag="pass1_restricted",
                            selected_anchor_ids=selected_anchor_ids,
                            contexts_block=contexts_block,
                            context_out_path=pass1_context_path,
                            raw_base_path=pass1_raw_base,
                            compiled_out_path=pass1_compiled_path,
                        )

            if pass1_doc and _has_grounded_definition(pass1_doc):
                context_path.write_text(pass1_context_path.read_text(encoding="utf-8", errors="replace"), encoding="utf-8")
                raw_path.write_text(pass1_raw_base.read_text(encoding="utf-8", errors="replace"), encoding="utf-8")
                parsed_path.write_text(json.dumps(pass1_doc, indent=2) + "\n", encoding="utf-8")
                return

            # Pass 2 (Option B): if pass 1 fails, fall back to the full document definition-finder.
            finder_anchor_ids: list[str] = []
            try:
                all_anchor_ids = [str(info["anchor_id"]) for info in sorted(anchors.values(), key=lambda a: int(a["order"]))]
                full_doc_block = build_excerpt_pack_from_canonical(canonical_text=canonical_text, catalog=anchors, anchor_ids=all_anchor_ids)
                metrics_json = json.dumps(
                    [{"name": metric.name, "description": metric.contract_term_hint or None}],
                    indent=2,
                    sort_keys=True,
                )
                finder_prompt = _render_definition_finder_prompt(
                    definition_finder_template,
                    metrics_json=metrics_json,
                    document=full_doc_block,
                )
                finder_raw = await _call_gateway_with_retry(
                    client=client,
                    prompt=finder_prompt,
                    model=model,
                    temperature=temperature,
                    reasoning=reasoning,
                    attempts=3,
                )
                finder_raw_path.write_text(finder_raw, encoding="utf-8")

                finder_payload = json.loads(finder_raw)
                selection = PricingDefinitionsIndexingSelectionV1.model_validate(finder_payload)
                finder_json_path.write_text(selection.model_dump_json(indent=2) + "\n", encoding="utf-8")

                if len(selection.metrics) != 1 or selection.metrics[0].name != metric.name:
                    raise RuntimeError(
                        "definition finder returned unexpected metric coverage "
                        f"(got={[m.name for m in selection.metrics]}; expected={[metric.name]})"
                    )
                finder_anchor_ids = list(selection.metrics[0].definition_anchor_ids or [])
                invalid = [a for a in finder_anchor_ids if a not in anchors]
                if invalid:
                    raise RuntimeError(f"definition finder returned unknown anchors for {item_id}: {invalid[:10]}")
            except Exception as exc:
                soft_issues.append((f"{item_id}::{metric.name}::__definition_finder__", f"{type(exc).__name__}: {exc}"))
                finder_anchor_ids = []

            if finder_anchor_ids:
                expanded = expand_anchor_ids(
                    catalog=anchors,
                    seed_anchor_ids=finder_anchor_ids,
                    fill_gaps_up_to=24,
                    neighbor_pad=3,
                )
                contexts_block = _build_context_blocks(canonical_text, anchors, expanded)
                pass2_doc = await _compile_with_contexts(
                    pass_tag="pass2_definition_finder",
                    selected_anchor_ids=expanded,
                    contexts_block=contexts_block,
                    context_out_path=pass2_context_path,
                    raw_base_path=pass2_raw_base,
                    compiled_out_path=pass2_compiled_path,
                )

                context_path.write_text(pass2_context_path.read_text(encoding="utf-8", errors="replace"), encoding="utf-8")
                raw_path.write_text(pass2_raw_base.read_text(encoding="utf-8", errors="replace"), encoding="utf-8")
                parsed_path.write_text(json.dumps(pass2_doc, indent=2) + "\n", encoding="utf-8")
                return

            # If we got a schema-valid pass 1 doc, keep it even if it didn't locate a definition.
            if pass1_doc:
                soft_issues.append(
                    (
                        f"{item_id}::{metric.name}",
                        "Definition not located in restricted contexts; definition-finder returned no anchors.",
                    )
                )
                parsed_path.write_text(json.dumps(pass1_doc, indent=2) + "\n", encoding="utf-8")
                return

            # Deterministic non-LLM output: no evidence found to provide contexts.
            msg = (
                f"Definition not located for metric={metric.name!r} "
                f"(contract_term_hint={metric.contract_term_hint!r}; search_tokens={metric.search_tokens!r})."
            )
            soft_issues.append((f"{item_id}::{metric.name}", msg))
            parsed_path.write_text(
                json.dumps(
                    {
                        "schema_version": "definition_compiler_v2_ast_v1",
                        "definitions": [
                            {
                                "name": metric.name,
                                "contract_term": None,
                                "definition_verbatim": None,
                                "expression_ast": None,
                                "input_terms": [],
                                "clauses": [],
                                "source_refs": [],
                                "needs_more_context": False,
                                "confidence": "low",
                                "notes": [msg],
                            }
                        ],
                        "unresolved_dependencies": [],
                    },
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
            return

    async def _process_item(item_id: str, client: Any) -> None:
        try:
            dg = _load_dg_output(paths, item_id, qa_subdir)
        except Exception as exc:
            err_path = out_dir / f"{item_id}__dg_output.error.txt"
            err_path.write_text(f"{exc}\n\n{traceback.format_exc()}", encoding="utf-8")
            hard_errors.append((f"{item_id}::__dg_output__", str(exc)))
            return

        try:
            metric_targets = _extract_metric_targets(dg)
        except Exception as exc:
            err_path = out_dir / f"{item_id}__dg_output.schema.error.txt"
            err_path.write_text(f"{exc}\n\n{traceback.format_exc()}", encoding="utf-8")
            hard_errors.append((f"{item_id}::__dg_schema__", str(exc)))
            return

        if not metric_targets:
            msg = f"No metrics found in dg output for {item_id} (qa_subdir={qa_subdir})."
            soft_issues.append((f"{item_id}::__no_metrics__", msg))
            return

        canonical_text = prompt_view_path(paths, item_id).read_text(encoding="utf-8", errors="replace")
        anchors = load_anchor_catalog(paths, item_id)
        definitions_anchor_ids, definitions_span = _definitions_anchor_ids_from_indexing_v2(paths, item_id, anchors)
        pricing_union_anchor_ids = _pricing_union_anchor_ids_from_indexing_v2(paths, item_id, anchors)
        financial_covenant_anchor_ids = _financial_covenant_anchor_ids_from_indexing_v2(paths, item_id, anchors)

        # Compile each metric (concurrent under semaphore).
        results = await asyncio.gather(
            *(
                _process_metric(
                    item_id=item_id,
                    metric=mt,
                    canonical_text=canonical_text,
                    anchors=anchors,
                    definitions_anchor_ids=definitions_anchor_ids,
                    definitions_span=definitions_span,
                    pricing_union_anchor_ids=pricing_union_anchor_ids,
                    financial_covenant_anchor_ids=financial_covenant_anchor_ids,
                    client=client,
                )
                for mt in metric_targets
            ),
            return_exceptions=True,
        )
        for res in results:
            if isinstance(res, Exception):
                hard_errors.append((f"{item_id}::__metric_task__", str(res)))

        # Build an item-level aggregate for convenience.
        compiled: list[dict] = []
        deps: dict[str, set[str]] = {}
        for mt in metric_targets:
            safe_metric = _safe_slug(mt.name)
            parsed_path = out_dir / f"{item_id}__{safe_metric}__compiled.json"
            if not parsed_path.exists():
                continue
            try:
                doc = json.loads(parsed_path.read_text(encoding="utf-8"))
            except Exception:
                continue
            if not isinstance(doc, dict):
                continue
            defs = doc.get("definitions")
            if isinstance(defs, list) and defs and isinstance(defs[0], dict):
                compiled.append(defs[0])
            ud = doc.get("unresolved_dependencies")
            if isinstance(ud, list):
                for row in ud:
                    if not isinstance(row, dict):
                        continue
                    term = row.get("term")
                    ref_by = row.get("referenced_by")
                    if not isinstance(term, str) or not term.strip():
                        continue
                    if not isinstance(ref_by, list) or not all(isinstance(v, str) for v in ref_by):
                        continue
                    deps.setdefault(term.strip(), set()).update(v.strip() for v in ref_by if v.strip())

        item_out = out_dir / f"{item_id}__compiled.json"
        unresolved = [{"term": t, "referenced_by": sorted(list(rb))} for t, rb in sorted(deps.items())]
        item_out.write_text(
            json.dumps(
                {
                    "schema_version": "definition_compiler_v2_ast_v1",
                    "definitions": compiled,
                    "unresolved_dependencies": unresolved,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

    async def _runner() -> None:
        async with GatewayAgentClient(
            base_url=gateway_url or DEFAULT_GATEWAY_URL,
            timeout=gateway_timeout or 600.0,
        ) as client:
            results = await asyncio.gather(*(_process_item(item_id, client) for item_id in items), return_exceptions=True)
            for res in results:
                if isinstance(res, Exception):
                    hard_errors.append(("__runner__", str(res)))

    asyncio.run(_runner())

    if paths.manifest_path.exists():
        update_manifest(
            paths.manifest_path,
            definitions_compiler_v1_prompt=str(compiler_prompt_path),
            definitions_compiler_v1_prompt_sha256=prompt_digest,
            definitions_compiler_v1_definition_finder_prompt=str(definition_finder_prompt_path),
            definitions_compiler_v1_definition_finder_prompt_sha256=definition_finder_prompt_digest,
            definitions_compiler_v1_qa_subdir=qa_subdir,
            definitions_compiler_v1_output_subdir=output_subdir,
            definitions_compiler_v1_created_at=int(time.time()),
        )

    if soft_issues:
        (out_dir / "issues.txt").write_text("\n".join(f"{k}: {v}" for k, v in soft_issues) + "\n", encoding="utf-8")

    if hard_errors:
        (out_dir / "errors.txt").write_text("\n".join(f"{k}: {v}" for k, v in hard_errors) + "\n", encoding="utf-8")
        raise RuntimeError(
            f"Definitions compiler v1 completed with hard errors (count={len(hard_errors)}); see {out_dir / 'errors.txt'}"
        )
