from __future__ import annotations

import asyncio
import json
import re
import traceback
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from pipeline.core.anchors import load_anchor_catalog
from pipeline.core.config import Paths, REQUIRED_MODEL, REQUIRED_REASONING, prompt_hash, update_manifest
from pipeline.llm.gateway import DEFAULT_GATEWAY_URL, _ensure_gateway_client_async
from pipeline.schemas_v2 import IndexingSelectionV2Artifact
from pipeline.utils import assert_exists, prompt_view_path


def _load_dg_output(paths: Paths, item_id: str, qa_subdir: str) -> Dict[str, Any]:
    """Load the dg-v2 structured output for an item.

    This must be valid JSON. We fail loudly here because downstream definition extraction
    depends on contract_term/search_tokens being present in the dg output.
    """

    path = paths.structured_dir / qa_subdir / f"{item_id}.json"
    if not path.exists():
        raise FileNotFoundError(
            "Missing structured output for definition targeting. "
            f"Expected {path}."
        )
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"dg output is not valid JSON for {item_id}: {path}") from exc
    if not isinstance(doc, dict):
        raise RuntimeError(f"dg output must be a JSON object for {item_id}: {path}")
    return doc


def _extract_terms(doc: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Extract definition targets from the dg-v2 output.

    Expected (prompt-enforced) fields:
    - metrics[*].contract_term (str, may be empty when evidence is missing)
    - metrics[*].search_tokens (list[str], may be empty when evidence is missing)
    - facilities[*].rates[*].base_rate_contract_term (str, may be empty)
    - facilities[*].rates[*].base_rate_search_tokens (list[str], may be empty)

    We return a normalized list of term objects:
      {"term_id": "...", "term_type": "metric|rate_term", "contract_term": "...", "search_tokens": [...]}
    """

    terms: List[Dict[str, Any]] = []

    for metric in doc.get("metrics", []) or []:
        if not isinstance(metric, dict):
            continue
        metric_name = metric.get("name")
        if not isinstance(metric_name, str) or not metric_name.strip():
            continue

        contract_term = metric.get("contract_term")
        if contract_term is None:
            raise RuntimeError(
                "dg output is missing metrics[*].contract_term; rerun structured-v2 with a prompt that outputs it."
            )
        if not isinstance(contract_term, str):
            raise RuntimeError("dg output metrics[*].contract_term must be a string")

        search_tokens = metric.get("search_tokens")
        if search_tokens is None:
            raise RuntimeError(
                "dg output is missing metrics[*].search_tokens; rerun structured-v2 with a prompt that outputs it."
            )
        if not isinstance(search_tokens, list) or not all(isinstance(t, str) for t in search_tokens):
            raise RuntimeError("dg output metrics[*].search_tokens must be a list of strings")

        terms.append(
            {
                "term_id": metric_name.strip(),
                "term_type": "metric",
                "contract_term": contract_term.strip(),
                "search_tokens": [t.strip() for t in search_tokens if t.strip()],
            }
        )

    for fac in doc.get("facilities", []) or []:
        if not isinstance(fac, dict):
            continue
        for rate in fac.get("rates", []) or []:
            if not isinstance(rate, dict):
                continue
            base_rate = rate.get("base_rate")
            if not isinstance(base_rate, str) or not base_rate.strip():
                continue

            base_rate_contract_term = rate.get("base_rate_contract_term")
            if base_rate_contract_term is None:
                raise RuntimeError(
                    "dg output is missing facilities[*].rates[*].base_rate_contract_term; "
                    "rerun structured-v2 with a prompt that outputs it."
                )
            if not isinstance(base_rate_contract_term, str):
                raise RuntimeError("dg output facilities[*].rates[*].base_rate_contract_term must be a string")

            base_rate_search_tokens = rate.get("base_rate_search_tokens")
            if base_rate_search_tokens is None:
                raise RuntimeError(
                    "dg output is missing facilities[*].rates[*].base_rate_search_tokens; "
                    "rerun structured-v2 with a prompt that outputs it."
                )
            if not isinstance(base_rate_search_tokens, list) or not all(
                isinstance(t, str) for t in base_rate_search_tokens
            ):
                raise RuntimeError("dg output facilities[*].rates[*].base_rate_search_tokens must be a list of strings")

            terms.append(
                {
                    "term_id": base_rate.strip(),
                    "term_type": "rate_term",
                    "contract_term": base_rate_contract_term.strip(),
                    "search_tokens": [t.strip() for t in base_rate_search_tokens if t.strip()],
                }
            )

    # De-duplicate by (term_type, normalized contract_term when present, else term_id).
    seen: set[Tuple[str, str]] = set()
    unique: List[Dict[str, Any]] = []
    for rec in terms:
        term_type = str(rec.get("term_type") or "")
        contract_term = str(rec.get("contract_term") or "")
        term_id = str(rec.get("term_id") or "")
        basis = contract_term if contract_term.strip() else term_id
        norm = re.sub(r"\s+", " ", basis).strip().lower()
        key = (term_type, norm)
        if key in seen:
            continue
        seen.add(key)
        unique.append(rec)
    return unique


def _load_snippets_v2(paths: Paths, item_id: str) -> List[Dict[str, Any]]:
    path = assert_exists(
        paths.run_dir / "retrieval_v2" / f"{item_id}_snippets.jsonl",
        message=f"Missing v2 snippets for {item_id}: run retrieve-v2 first.",
    )
    snippets: List[Dict[str, Any]] = []
    with path.open() as fh:
        for line in fh:
            if not line.strip():
                continue
            try:
                snippets.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    if not snippets:
        raise RuntimeError(f"No snippets parsed for {item_id} (file: {path})")
    return snippets


def _render_snippet_block(snippets: List[Dict[str, Any]]) -> str:
    blocks: List[str] = []
    for rec in snippets:
        aid = rec.get("anchor_id") or "UNK"
        label = rec.get("label") or rec.get("type")
        toc_title = rec.get("toc_title")
        toc_chunk_id = rec.get("toc_chunk_id")
        snippet_text = (rec.get("snippet") or "").strip()
        header = f"[[{aid}]]"
        lines: List[str] = [header]
        if label:
            lines.append(f"label: {label}")
        if isinstance(toc_title, str) and toc_title.strip():
            if isinstance(toc_chunk_id, int):
                lines.append(f"toc: {toc_title.strip()} (chunk {toc_chunk_id})")
            else:
                lines.append(f"toc: {toc_title.strip()}")
        lines.append(snippet_text)
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


def _render_canonical_prompt(term: str, term_type: str, snippets_block: str) -> str:
    raise RuntimeError(
        "Canonical term resolution via LLM has been removed for definitions-v2. "
        "Provide contract_term/search_tokens in the dg-v2 output instead."
    )


def _render_definition_prompt(template: str, term: str, term_type: str, contexts_block: str) -> str:
    out = template
    out = out.replace("{{TERM}}", term)
    out = out.replace("{{TERM_TYPE}}", term_type)
    out = out.replace("{{CONTEXTS}}", contexts_block)
    return out


def _find_term_hits(text: str, term: str) -> List[Tuple[int, int]]:
    """Return case-insensitive matches for a term, tolerant to whitespace differences.

    Canonical text often contains hard line breaks or multiple spaces inside terms
    (e.g., table cells or wrapped defined terms). We treat any run of whitespace
    in the search term as flexible `\\s+` so tokens like "RFR Loans" still match
    "RFR\\nLoans" in the canonical text.
    """

    parts = [p for p in re.split(r"\s+", term.strip()) if p]
    if not parts:
        return []
    pattern = r"\s+".join(re.escape(p) for p in parts)
    return [(m.start(), m.end()) for m in re.finditer(pattern, text, flags=re.IGNORECASE)]


def _match_hits_to_anchors(
    anchors: Dict[str, Dict[str, Any]], hits: List[Tuple[int, int]]
) -> List[str]:
    # Build ordered list for deterministic scan.
    ordered = sorted(anchors.values(), key=lambda a: a["start"])
    anchor_ids: List[str] = []
    for start, _end in hits:
        for info in ordered:
            if info["start"] <= start < info["end"]:
                anchor_ids.append(info["anchor_id"])
                break
    # preserve order, de-dupe
    seen = set()
    unique = []
    for aid in anchor_ids:
        if aid in seen:
            continue
        seen.add(aid)
        unique.append(aid)
    return unique


def _definitions_anchor_ids_from_indexing_v2(
    paths: Paths, item_id: str, anchors: Dict[str, Dict[str, Any]]
) -> Tuple[List[str], Optional[Tuple[int, int]]]:
    """Return (definitions_anchor_ids, (start_pos,end_pos)) from indexing_v2 if available.

    Contract:
    - If indexing_v2/<item_id>_anchors.json is missing or does not include definitions_anchor_range, return ([], None).
    - If definitions_anchor_range is present, expand it to the full anchor_id list (inclusive) and also return the
      corresponding character span into canonical.txt to support fast filtering of term hits.
    """

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


def _build_contexts(
    text: str,
    anchors: Dict[str, Dict[str, Any]],
    anchor_ids: List[str],
    bandwidth_chars: int = 400,
) -> str:
    blocks: List[str] = []
    for aid in anchor_ids:
        info = anchors.get(aid)
        if not info:
            continue
        start = int(info["start"])
        end = int(info["end"])
        a = max(0, start - bandwidth_chars)
        b = min(len(text), end + bandwidth_chars)
        snippet = text[a:b]
        blocks.append(f"[[{aid}]]\n{snippet}")
    return "\n\n".join(blocks)

def _select_context_anchor_ids(
    *,
    canonical_text: str,
    anchors: Dict[str, Dict[str, Any]],
    candidate_anchor_ids: List[str],
    contract_term: str,
    search_tokens: List[str],
    max_anchors: int = 12,
    include_neighbors_before: int = 0,
    include_neighbors_after: int = 0,
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
        if " means " in hay or " shall mean " in hay or " is defined as " in hay:
            score += 10
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
        score = _score(_anchor_text(aid))
        scored.append((score, start, aid))

    # Prefer higher score; tie-breaker by document position.
    scored_sorted = sorted(scored, key=lambda t: (-t[0], t[1]))

    # Deterministic neighbor expansion (useful for multi-anchor definitions). We expand around the best anchors
    # until we hit max_anchors.
    ordered = sorted(anchors.values(), key=lambda a: int(a["order"]))
    id_by_order = [str(info["anchor_id"]) for info in ordered]
    order_by_id = {str(info["anchor_id"]): int(info["order"]) for info in ordered}

    selected: list[str] = []
    seen: set[str] = set()

    def _maybe_add(aid: str) -> None:
        if not aid:
            return
        if allowed_anchor_ids is not None and aid not in allowed_anchor_ids:
            return
        if aid in seen:
            return
        seen.add(aid)
        selected.append(aid)

    for _score, _start, aid in scored_sorted:
        _maybe_add(aid)
        if include_neighbors_before or include_neighbors_after:
            base_order = order_by_id.get(aid)
            if base_order is not None:
                for delta in range(-int(include_neighbors_before), int(include_neighbors_after) + 1):
                    if delta == 0:
                        continue
                    j = base_order + delta
                    if 0 <= j < len(id_by_order):
                        _maybe_add(id_by_order[j])
        if len(selected) >= max(1, max_anchors):
            break

    # Re-sort in stable doc order.
    return sorted(selected, key=lambda x: int(anchors.get(x, {}).get("start", 0)))


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
    return result.get("text") if isinstance(result, dict) else str(result)


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
            return await _call_gateway(
                client=client,
                prompt=prompt,
                model=model,
                temperature=temperature,
                reasoning=reasoning,
            )
        except Exception as exc:  # pragma: no cover - network/errors
            last_exc = exc
            await asyncio.sleep(1.5 * attempt)
    raise last_exc if last_exc else RuntimeError("Gateway call failed with unknown error.")


def run_definitions_v2(
    paths: Paths,
    item_ids: Iterable[str],
    *,
    qa_subdir: str,
    definitions_prompt_path: Path,
    model: str | None = None,
    gateway_url: str | None = None,
    temperature: float = 0.0,
    reasoning: str | None = None,
    gateway_timeout: float | None = None,
    concurrency: int = 3,
    output_subdir: str = "definitions_v2",
) -> None:
    """Extract metric/rate term definitions using dg-provided contract_term/search_tokens.

    This stage is intentionally strict:
    - dg output must be valid JSON and include contract_term/search_tokens for every term
    - missing/invalid inputs or malformed definition JSON are treated as errors
    """

    model = REQUIRED_MODEL
    reasoning = REQUIRED_REASONING

    assert_exists(definitions_prompt_path, message=f"Definitions prompt not found: {definitions_prompt_path}")
    definitions_template = definitions_prompt_path.read_text()
    prompt_digest = prompt_hash(definitions_prompt_path)

    out_dir = paths.run_dir / "definitions_v2" / output_subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    # Avoid stale summaries from previous runs.
    for stale_name in ("errors.txt", "issues.txt"):
        stale_path = out_dir / stale_name
        if stale_path.exists():
            stale_path.unlink()

    items = list(item_ids)
    GatewayAgentClient = _ensure_gateway_client_async()
    term_sem = asyncio.Semaphore(max(1, concurrency))
    # Hard errors abort the stage (invalid inputs or invalid definition JSON outputs).
    # Soft issues are recorded but do not fail the stage (e.g., no terms for an item,
    # or a term can't be found in the canonical text given the dg-provided tokens).
    hard_errors: List[Tuple[str, str]] = []
    soft_issues: List[Tuple[str, str]] = []

    async def _process_term(
        item_id: str,
        term_id: str,
        term_type: str,
        contract_term: str,
        search_tokens: List[str],
        snippets_block: str,
        canonical_text: str,
        anchors: Dict[str, Dict[str, Any]],
        definitions_anchor_ids: List[str],
        definitions_span: Optional[Tuple[int, int]],
        client: Any,
    ) -> None:
        async with term_sem:
            safe_term = re.sub(r"[^A-Za-z0-9_-]+", "_", term_id).strip("_") or "term"
            last_raw: str | None = None
            try:
                # Prefer contract_term hits; if absent, fall back to term_id; then fall back to search_tokens.
                hits: List[Tuple[int, int]] = []
                using_definitions_span = False
                allowed_anchor_ids: Optional[set[str]] = None

                def_start: int | None = None
                def_end: int | None = None
                if definitions_span is not None:
                    def_start, def_end = int(definitions_span[0]), int(definitions_span[1])
                    if def_start < 0 or def_end < 0 or def_start >= def_end:
                        raise RuntimeError(
                            f"Invalid definitions_span for {item_id}: start={def_start} end={def_end}"
                        )

                primary = contract_term.strip() if contract_term.strip() else term_id
                if primary:
                    hits = _find_term_hits(canonical_text, primary)

                # Prefer hits that occur inside the dedicated definitions section when available.
                if hits and def_start is not None and def_end is not None:
                    hits_in_defs = [(s, e) for (s, e) in hits if def_start <= s < def_end]
                    if hits_in_defs:
                        hits = hits_in_defs
                        using_definitions_span = True
                        allowed_anchor_ids = set(definitions_anchor_ids)

                if not hits:
                    if not search_tokens:
                        raise RuntimeError(
                            f"No term hits for {term_id} and no search_tokens provided (contract_term={contract_term!r})."
                        )

                    # Try search tokens within definitions section first (if we have a span), then fall back
                    # to whole-document token hits.
                    token_hits: List[Tuple[int, int]] = []
                    for tok in search_tokens:
                        token_hits.extend(_find_term_hits(canonical_text, tok))

                    if token_hits and def_start is not None and def_end is not None:
                        token_hits_in_defs = [(s, e) for (s, e) in token_hits if def_start <= s < def_end]
                        if token_hits_in_defs:
                            hits = token_hits_in_defs
                            using_definitions_span = True
                            allowed_anchor_ids = set(definitions_anchor_ids)
                        else:
                            hits = token_hits
                    else:
                        hits = token_hits

                if not hits:
                    msg = (
                        f"No matches found in canonical text for term_id={term_id!r} "
                        f"(contract_term={contract_term!r}; search_tokens={search_tokens!r})."
                    )
                    not_found_path = out_dir / f"{item_id}__{safe_term}__definition.not_found.json"
                    not_found_path.write_text(json.dumps({"term_id": term_id, "term_type": term_type, "error": msg}, indent=2))
                    soft_issues.append((f"{item_id}::{term_id}", msg))
                    return

                candidate_anchor_ids = _match_hits_to_anchors(anchors, hits)
                if not candidate_anchor_ids:
                    msg = (
                        f"Found text matches but could not map to anchors for term_id={term_id!r} "
                        f"(contract_term={contract_term!r}; search_tokens={search_tokens!r})."
                    )
                    not_found_path = out_dir / f"{item_id}__{safe_term}__definition.not_found.json"
                    not_found_path.write_text(json.dumps({"term_id": term_id, "term_type": term_type, "error": msg}, indent=2))
                    soft_issues.append((f"{item_id}::{term_id}", msg))
                    return

                anchor_ids = _select_context_anchor_ids(
                    canonical_text=canonical_text,
                    anchors=anchors,
                    candidate_anchor_ids=candidate_anchor_ids,
                    contract_term=contract_term,
                    search_tokens=search_tokens,
                    max_anchors=12,
                    include_neighbors_before=1 if using_definitions_span else 0,
                    include_neighbors_after=2 if using_definitions_span else 0,
                    allowed_anchor_ids=allowed_anchor_ids,
                )
                contexts_block = _build_contexts(canonical_text, anchors, anchor_ids)
                ctx_path = out_dir / f"{item_id}__{safe_term}__context.txt"
                ctx_path.write_text(contexts_block)
                if not contexts_block.strip():
                    raise RuntimeError(f"Empty contexts block for {term_id} (anchors selected: {anchor_ids})")

                term_for_prompt = contract_term.strip() if contract_term.strip() else term_id
                def_prompt = _render_definition_prompt(definitions_template, term_for_prompt, term_type, contexts_block)
                max_output_attempts = 3

                for out_attempt in range(1, max_output_attempts + 1):
                    def_raw = await _call_gateway_with_retry(
                        client=client,
                        prompt=def_prompt,
                        model=model,
                        temperature=temperature,
                        reasoning=reasoning,
                    )
                    last_raw = def_raw
                    def_path = out_dir / f"{item_id}__{safe_term}__definition.txt"
                    def_path.write_text(def_raw)

                    try:
                        parsed = json.loads(def_raw)
                        parsed_path = out_dir / f"{item_id}__{safe_term}__definition.json"
                        if not isinstance(parsed, dict):
                            raise RuntimeError("definition output must be a JSON object")
                        # Basic shape validation (fail loudly).
                        required_keys = {"term", "term_type", "definition_text", "anchor_refs", "candidates", "notes"}
                        missing_keys = sorted(required_keys - set(parsed.keys()))
                        if missing_keys:
                            raise RuntimeError(f"definition output missing required keys: {', '.join(missing_keys)}")

                        def _norm(s: str) -> str:
                            return re.sub(r"\s+", " ", s).strip().lower()

                        parsed_term = parsed.get("term")
                        if not isinstance(parsed_term, str) or not parsed_term.strip():
                            raise RuntimeError("definition output term must be a non-empty string")
                        if _norm(parsed_term) != _norm(term_for_prompt):
                            raise RuntimeError(
                                f"definition output term mismatch: expected {term_for_prompt!r} got {parsed_term!r}"
                            )
                        if parsed.get("term_type") != term_type:
                            raise RuntimeError(
                                f"definition output term_type mismatch: expected {term_type!r} got {parsed.get('term_type')!r}"
                            )
                        definition_text = parsed.get("definition_text")
                        if definition_text is not None and not isinstance(definition_text, str):
                            raise RuntimeError("definition output definition_text must be a string or null")
                        anchor_refs = parsed.get("anchor_refs")
                        if not isinstance(anchor_refs, list) or not all(isinstance(a, str) for a in anchor_refs):
                            raise RuntimeError("definition output anchor_refs must be a list of strings")
                        allowed_ctx_anchors = set(anchor_ids)
                        unknown_anchors = [a for a in anchor_refs if a not in allowed_ctx_anchors]
                        if unknown_anchors:
                            raise RuntimeError(
                                "definition output contains anchor_refs not present in the provided CONTEXTS "
                                f"(examples={unknown_anchors[:10]})"
                            )

                        candidates = parsed.get("candidates")
                        if not isinstance(candidates, list):
                            raise RuntimeError("definition output candidates must be a list")
                        for idx, cand in enumerate(candidates):
                            if not isinstance(cand, dict):
                                raise RuntimeError(f"definition output candidates[{idx}] must be an object")
                            if "definition_text" not in cand or "anchor_refs" not in cand:
                                raise RuntimeError(f"definition output candidates[{idx}] missing keys")
                            cand_anchors = cand.get("anchor_refs")
                            if not isinstance(cand_anchors, list) or not all(isinstance(a, str) for a in cand_anchors):
                                raise RuntimeError(
                                    f"definition output candidates[{idx}].anchor_refs must be a list of strings"
                                )
                            cand_unknown = [a for a in cand_anchors if a not in allowed_ctx_anchors]
                            if cand_unknown:
                                raise RuntimeError(
                                    f"definition output candidates[{idx}] contains anchor_refs not present in the provided CONTEXTS "
                                    f"(examples={cand_unknown[:10]})"
                                )

                        parsed_path.write_text(json.dumps(parsed, indent=2))
                        break
                    except Exception as exc:
                        if out_attempt < max_output_attempts:
                            await asyncio.sleep(0.75 * out_attempt)
                            continue
                        raise
            except Exception as exc:  # pragma: no cover - network/errors
                err_path = out_dir / f"{item_id}__{safe_term}__definition.error.txt"
                payload = f"{exc}\n\n{traceback.format_exc()}"
                if last_raw:
                    payload = f"{payload}\n\nLAST_OUTPUT:\n{last_raw}"
                err_path.write_text(payload)
                hard_errors.append((f"{item_id}::{term_id}", str(exc)))

    async def _process_item(item_id: str, client: Any) -> None:
        try:
            doc = _load_dg_output(paths, item_id, qa_subdir)
        except Exception as exc:
            msg = str(exc)
            err_path = out_dir / f"{item_id}__dg_output.error.txt"
            err_path.write_text(f"{exc}\n\n{traceback.format_exc()}")
            hard_errors.append((f"{item_id}::__dg_output__", msg))
            return

        try:
            terms = _extract_terms(doc)
        except Exception as exc:
            msg = str(exc)
            err_path = out_dir / f"{item_id}__dg_output.schema.error.txt"
            err_path.write_text(f"{exc}\n\n{traceback.format_exc()}")
            hard_errors.append((f"{item_id}::__dg_schema__", msg))
            return

        if not terms:
            msg = f"No definition terms extracted from dg output for {item_id} (qa_subdir={qa_subdir})."
            note_path = out_dir / f"{item_id}__no_terms.issue.txt"
            note_path.write_text(msg)
            soft_issues.append((f"{item_id}::__no_terms__", msg))
            return

        snippets = _load_snippets_v2(paths, item_id)
        snippets_block = _render_snippet_block(snippets)
        canonical_text = prompt_view_path(paths, item_id).read_text()
        anchors = load_anchor_catalog(paths, item_id)
        definitions_anchor_ids, definitions_span = _definitions_anchor_ids_from_indexing_v2(paths, item_id, anchors)

        results = await asyncio.gather(
            *(
                _process_term(
                    item_id,
                    str(t.get("term_id") or ""),
                    str(t.get("term_type") or ""),
                    str(t.get("contract_term") or ""),
                    list(t.get("search_tokens") or []),
                    snippets_block,
                    canonical_text,
                    anchors,
                    definitions_anchor_ids,
                    definitions_span,
                    client,
                )
                for t in terms
            ),
            return_exceptions=True,
        )
        for res in results:
            if isinstance(res, Exception):
                hard_errors.append((f"{item_id}::__task__", str(res)))

    async def _runner() -> None:
        async with GatewayAgentClient(
            base_url=gateway_url or DEFAULT_GATEWAY_URL,
            timeout=gateway_timeout or 600.0,
        ) as client:
            results = await asyncio.gather(
                *(_process_item(item_id, client) for item_id in items),
                return_exceptions=True,
            )
            for res in results:
                if isinstance(res, Exception):
                    hard_errors.append(("__runner__", str(res)))

    asyncio.run(_runner())

    manifest_path = paths.manifest_path
    if manifest_path.exists():
        update_manifest(
            manifest_path,
            definitions_v2_prompt=str(definitions_prompt_path),
            definitions_v2_prompt_sha256=prompt_digest,
            definitions_v2_qa_subdir=qa_subdir,
            definitions_v2_output_subdir=output_subdir,
        )

    if soft_issues:
        issues_path = out_dir / "issues.txt"
        issues_path.write_text("\n".join(f"{item}: {msg}" for item, msg in soft_issues))

    if hard_errors:
        err_path = out_dir / "errors.txt"
        err_path.write_text("\n".join(f"{item}: {msg}" for item, msg in hard_errors))
        raise RuntimeError(
            f"Definitions v2 completed with hard errors (count={len(hard_errors)}); see {err_path}"
        )
