from __future__ import annotations

import asyncio
import json
import re
import time
import traceback
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from pipeline.core.anchors import load_anchor_catalog
from pipeline.core.config import Paths, REQUIRED_MODEL, REQUIRED_REASONING, prompt_hash, update_manifest
from pipeline.compile.definitions_compiler_v1 import (
    _build_context_blocks,
    _filter_noise_anchor_ids,
    _definitions_anchor_ids_from_indexing_v2,
    _find_term_hits,
    _match_hits_to_anchors,
    _pricing_union_anchor_ids_from_indexing_v2,
    _retry_prompt,
    _select_context_anchor_ids,
    _validate_compiler_output_v2_ast,
)  # reuse stable schema validation + context selection
from pipeline.evidence.excerpt_packs import build_excerpt_pack_from_canonical
from pipeline.llm.gateway import DEFAULT_GATEWAY_URL, _ensure_gateway_client_async
from pipeline.schemas_v2 import IndexingSelectionV2Artifact
from pipeline.utils import assert_exists, prompt_view_path


@dataclass(frozen=True)
class BlockingTermTarget:
    term: str
    referenced_by: list[str]


def _safe_slug(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "_", s).strip("_") or "term"


def _render_prompt(
    template: str,
    *,
    term: str,
    contract_term_hint: str,
    search_tokens: list[str],
    contexts_block: str,
) -> str:
    out = template
    out = out.replace("{{METRIC_NAME}}", term)
    out = out.replace("{{CONTRACT_TERM_HINT}}", contract_term_hint)
    out = out.replace("{{SEARCH_TOKENS}}", json.dumps(search_tokens))
    out = out.replace("{{CONTEXTS}}", contexts_block)
    return out


def _load_metric_compiler_aggregate(
    *,
    paths: Paths,
    item_id: str,
    metrics_output_subdir: str,
) -> Dict[str, Any]:
    """Load item-level aggregate output from definitions_compiler_v1.

    Expected path:
      runs/<run_id>/definitions_compiler_v1/<metrics_output_subdir>/<item_id>__compiled.json
    """

    compiled_path = paths.run_dir / "definitions_compiler_v1" / metrics_output_subdir / f"{item_id}__compiled.json"
    if not compiled_path.exists():
        raise FileNotFoundError(
            f"Missing definitions_compiler_v1 aggregate for {item_id}: expected {compiled_path}"
        )
    try:
        doc = json.loads(compiled_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"definitions_compiler_v1 aggregate is not valid JSON for {item_id}: {compiled_path}") from exc
    if not isinstance(doc, dict):
        raise RuntimeError(f"definitions_compiler_v1 aggregate must be a JSON object for {item_id}: {compiled_path}")
    return doc


def _extract_blocking_term_targets(
    doc: Dict[str, Any],
) -> List[BlockingTermTarget]:
    """Extract dependency targets from the metric compiler aggregate.

    Default behavior: only uses doc.unresolved_dependencies (which the definitions compiler now
    normalizes to calculation dependencies).
    """

    ud = doc.get("unresolved_dependencies")
    if ud is None:
        raise RuntimeError("definitions_compiler_v1 aggregate missing unresolved_dependencies")
    if not isinstance(ud, list):
        raise RuntimeError("definitions_compiler_v1 aggregate unresolved_dependencies must be a list")

    by_term: dict[str, set[str]] = {}
    for row in ud:
        if not isinstance(row, dict):
            continue
        term = row.get("term")
        referenced_by = row.get("referenced_by")
        if not isinstance(term, str) or not term.strip():
            continue
        if not isinstance(referenced_by, list) or not all(isinstance(x, str) for x in referenced_by):
            continue
        by_term.setdefault(term.strip(), set()).update(x.strip() for x in referenced_by if x.strip())

    # Stable order (alphabetical by term) so the output is reproducible.
    return [BlockingTermTarget(term=t, referenced_by=sorted(list(rb))) for t, rb in sorted(by_term.items())]


def _canonical_term_key(term: str) -> str:
    s = term.strip()
    if not s:
        return ""
    s = s.strip("\"'“”‘’").strip()
    s = re.sub(r"^\(\s*(?:[ivx]+|[a-z]|\d+)\s*\)\s*", "", s, flags=re.IGNORECASE).strip()
    s = re.sub(r"^(?:[a-z]|\d+)\)\s+", "", s, flags=re.IGNORECASE).strip()
    s = re.sub(r"^(?:[a-z]|\d+)\.\s+", "", s, flags=re.IGNORECASE).strip()
    s = s.strip(" ,.;:()[]{}").strip()
    s = re.sub(r"\s+", " ", s).strip()
    return s.lower()


def _normalize_term_display(term: str) -> str:
    s = term.strip()
    s = s.strip("\"'“”‘’").strip()
    s = re.sub(r"^\(\s*(?:[ivx]+|[a-z]|\d+)\s*\)\s*", "", s, flags=re.IGNORECASE).strip()
    s = re.sub(r"^(?:[a-z]|\d+)\)\s+", "", s, flags=re.IGNORECASE).strip()
    s = re.sub(r"^(?:[a-z]|\d+)\.\s+", "", s, flags=re.IGNORECASE).strip()
    s = s.strip(" ,.;:()[]{}").strip()
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _financial_covenant_anchor_ids_from_indexing_v2(paths: Paths, item_id: str, anchors: Dict[str, Dict[str, Any]]) -> list[str]:
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


def run_blocking_terms_compiler_v1(
    paths: Paths,
    item_ids: Iterable[str],
    *,
    metrics_output_subdir: str,
    term_compiler_prompt_path: Path,
    model: str | None = None,
    gateway_url: str | None = None,
    temperature: float = 0.0,
    reasoning: str | None = None,
    gateway_timeout: float | None = None,
    concurrency: int = 2,
    attempts: int = 3,
    output_subdir: str = "blocking_terms_v1",
    max_depth: int = 1,
    max_terms: int = 200,
) -> None:
    """Resolve blocking terms that prevent Compustat evaluation.

    Inputs:
      - Item-level aggregate JSON from definitions_compiler_v1 (per item_id), which includes unresolved_dependencies.
    Behavior:
      - For each blocking term, first try a targeted context slice:
          - definitions section range (if detected by indexing_v2), plus
          - pricing/covenant anchors (if present in indexing_v2),
        and fall back to passing the *entire document* as anchored context blocks if needed.
      - Optionally recurse over newly discovered unresolved_dependencies up to max_depth.
    Outputs:
      - One per-term JSON (schema identical to definitions_compiler_v1 metric outputs).
      - Item-level aggregate file collecting term outputs.
    """

    assert_exists(term_compiler_prompt_path, message=f"Blocking-terms compiler prompt not found: {term_compiler_prompt_path}")

    # Enforce model + reasoning defaults for repeatability.
    model = REQUIRED_MODEL
    reasoning = REQUIRED_REASONING

    prompt_template = term_compiler_prompt_path.read_text(encoding="utf-8")
    prompt_digest = prompt_hash(term_compiler_prompt_path)

    out_dir = paths.run_dir / "blocking_terms_compiler_v1" / output_subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    for stale_name in ("errors.txt", "issues.txt"):
        stale_path = out_dir / stale_name
        if stale_path.exists():
            stale_path.unlink()

    items = list(item_ids)
    GatewayAgentClient = _ensure_gateway_client_async()
    # IMPORTANT: create loop-bound asyncio primitives inside the running loop.
    # Creating Semaphore before asyncio.run(...) can bind it to a different loop
    # on Python 3.9 compute nodes.
    term_sem: asyncio.Semaphore | None = None

    hard_errors: list[tuple[str, str]] = []
    soft_issues: list[tuple[str, str]] = []

    if max_depth < 1:
        raise ValueError("--max-depth must be >= 1")
    if max_terms < 1:
        raise ValueError("--max-terms must be >= 1")

    async def _process_term(
        *,
        item_id: str,
        target: BlockingTermTarget,
        canonical_text: str,
        anchors: Dict[str, Dict[str, Any]],
        definitions_anchor_ids: list[str],
        definitions_span: tuple[int, int] | None,
        pricing_union_anchor_ids: list[str],
        financial_covenant_anchor_ids: list[str],
        full_contexts_block: str,
        full_allowed_anchor_ids: set[str],
        client: Any,
    ) -> tuple[bool, list[str]]:
        if term_sem is None:
            raise RuntimeError("internal error: blocking-terms semaphore was not initialized")
        async with term_sem:
            term_display = _normalize_term_display(target.term)
            term_key = _canonical_term_key(term_display)
            if not term_display or not term_key:
                soft_issues.append((f"{item_id}::{target.term}", "Skipping empty/invalid blocking term string."))
                return (False, [])

            safe_term = _safe_slug(term_display)
            context_path = out_dir / f"{item_id}__{safe_term}__contexts.txt"
            raw_base_path = out_dir / f"{item_id}__{safe_term}__compile.raw.txt"
            compiled_path = out_dir / f"{item_id}__{safe_term}__compiled.json"

            pass1_context_path = out_dir / f"{item_id}__{safe_term}__contexts.pass1.txt"
            pass1_raw_base = out_dir / f"{item_id}__{safe_term}__compile.pass1.raw.txt"
            pass1_compiled_path = out_dir / f"{item_id}__{safe_term}__compiled.pass1.json"

            pass2_raw_base = out_dir / f"{item_id}__{safe_term}__compile.pass2.raw.txt"

            def _write_fallback_compiled(message: str) -> None:
                compiled_path.write_text(
                    json.dumps(
                        {
                            "schema_version": "definition_compiler_v2_ast_v1",
                            "metric_name": term_display,
                            "definitions": [
                                {
                                    "name": term_display,
                                    "contract_term": None,
                                    "definition_verbatim": None,
                                    "expression_ast": None,
                                    "input_terms": [],
                                    "clauses": [],
                                    "source_refs": [],
                                    "needs_more_context": False,
                                    "confidence": "low",
                                    "notes": [message],
                                }
                            ],
                            "unresolved_dependencies": [],
                        },
                        indent=2,
                    )
                    + "\n",
                    encoding="utf-8",
                )

            if compiled_path.exists():
                try:
                    existing = json.loads(compiled_path.read_text(encoding="utf-8"))
                    deps = [
                        row.get("term")
                        for row in (existing.get("unresolved_dependencies") or [])
                        if isinstance(row, dict)
                    ]
                    return (True, [d for d in deps if isinstance(d, str) and d.strip()])
                except Exception:
                    soft_issues.append(
                        (f"{item_id}::{term_display}", f"Existing compiled term JSON unreadable; recomputing: {compiled_path}")
                    )

            def _has_grounded_definition(doc: dict[str, Any]) -> bool:
                defs = doc.get("definitions")
                if not isinstance(defs, list) or not defs or not isinstance(defs[0], dict):
                    return False
                d0 = defs[0]
                dv = d0.get("definition_verbatim")
                srefs = d0.get("source_refs")
                return bool(isinstance(dv, str) and dv.strip() and isinstance(srefs, list) and len(srefs) > 0)

            async def _compile_with_contexts(
                *,
                pass_tag: str,
                contexts_block: str,
                allowed_anchor_ids: set[str],
                raw_base_path: Path,
                compiled_out_path: Path,
            ) -> dict[str, Any]:
                rendered = _render_prompt(
                    prompt_template,
                    term=term_display,
                    contract_term_hint=term_display,
                    search_tokens=[],
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
                            metric_name=term_display,
                            raw_json=parsed,
                            allowed_anchor_ids=allowed_anchor_ids,
                            contexts_text=contexts_block,
                        )
                    except Exception as exc:
                        last_error = f"schema validation failed: {exc}"
                        await asyncio.sleep(1.5 * attempt)
                        continue

                    for iss in issues:
                        soft_issues.append((f"{item_id}::{term_display}::{pass_tag}", iss))

                    raw_base_path.write_text(raw, encoding="utf-8")
                    compiled_out_path.write_text(json.dumps(cleaned, indent=2) + "\n", encoding="utf-8")
                    return cleaned

                raise RuntimeError(
                    f"blocking term output could not be parsed/validated for term={term_display!r} "
                    f"after {attempts} attempt(s): {last_error}"
                )

            # Pass 1: targeted context pack from:
            #   - indexing-identified anchors (pricing/covenant) and
            #   - the full definitions section anchors (if available),
            # plus a small local slice around term hits (anywhere in the document).
            #
            # Rationale: long definitions often span many anchors, and term hits can miss
            # bridging anchors. Including the full definitions section reduces "truncated
            # context" failures, while the local slice helps when a term is defined outside
            # the definitions section.
            primary = term_display
            hits = _find_term_hits(canonical_text, primary) if primary else []

            local_slice_anchor_ids: list[str] = []
            if hits:
                candidate_anchor_ids = _match_hits_to_anchors(anchors, hits)
                if candidate_anchor_ids:
                    local_slice_anchor_ids = _select_context_anchor_ids(
                        canonical_text=canonical_text,
                        anchors=anchors,
                        candidate_anchor_ids=candidate_anchor_ids,
                        contract_term=term_display,
                        search_tokens=[],
                        max_anchors=96,
                        include_neighbors_before=2,
                        include_neighbors_after=6,
                        allowed_anchor_ids=None,
                    )

            # Prefer a small, term-local slice when we have hits. Fall back to the full definitions
            # section (if identified), otherwise to other indexing anchors.
            #
            # Rationale: passing the entire definitions section can be extremely large and can make
            # verbatim copying harder; a focused slice tends to improve grounding reliability.
            if local_slice_anchor_ids:
                base_pass1_anchor_ids = local_slice_anchor_ids
            elif definitions_anchor_ids:
                base_pass1_anchor_ids = list(dict.fromkeys(definitions_anchor_ids))
            else:
                base_pass1_anchor_ids = list(
                    dict.fromkeys((pricing_union_anchor_ids or []) + (financial_covenant_anchor_ids or []))
                )

            # Stable union of anchor IDs (preserve order; de-dupe).
            selected_anchor_ids: list[str] = []
            seen: set[str] = set()
            for aid in base_pass1_anchor_ids:
                if aid and aid not in seen:
                    selected_anchor_ids.append(aid)
                    seen.add(aid)

            pass1_doc: dict[str, Any] | None = None
            if selected_anchor_ids:
                selected_anchor_ids = _filter_noise_anchor_ids(
                    canonical_text=canonical_text,
                    anchors=anchors,
                    anchor_ids=selected_anchor_ids,
                )
                selected_anchor_ids = sorted(
                    selected_anchor_ids,
                    key=lambda aid: int(anchors.get(aid, {}).get("order", 10**9)),
                )
                contexts_block = _build_context_blocks(canonical_text, anchors, selected_anchor_ids)
                pass1_context_path.write_text(contexts_block, encoding="utf-8")
                try:
                    pass1_doc = await _compile_with_contexts(
                        pass_tag="pass1_restricted",
                        contexts_block=contexts_block,
                        allowed_anchor_ids=set(selected_anchor_ids),
                        raw_base_path=pass1_raw_base,
                        compiled_out_path=pass1_compiled_path,
                    )
                except Exception as exc:
                    # Pass 1 failures should not prevent the full-document fallback pass.
                    soft_issues.append(
                        (
                            f"{item_id}::{term_display}::pass1_restricted",
                            f"Pass 1 failed; falling back to full document contexts: {exc}",
                        )
                    )
                    pass1_doc = None

            if pass1_doc and _has_grounded_definition(pass1_doc):
                context_path.write_text(pass1_context_path.read_text(encoding="utf-8", errors="replace"), encoding="utf-8")
                raw_base_path.write_text(pass1_raw_base.read_text(encoding="utf-8", errors="replace"), encoding="utf-8")
                compiled_path.write_text(json.dumps(pass1_doc, indent=2) + "\n", encoding="utf-8")
                deps = [row.get("term") for row in (pass1_doc.get("unresolved_dependencies") or []) if isinstance(row, dict)]
                return (True, [d for d in deps if isinstance(d, str) and d.strip()])

            # Pass 2: full document fallback.
            try:
                pass2_doc = await _compile_with_contexts(
                    pass_tag="pass2_full_document",
                    contexts_block=full_contexts_block,
                    allowed_anchor_ids=full_allowed_anchor_ids,
                    raw_base_path=pass2_raw_base,
                    compiled_out_path=compiled_path,
                )
            except Exception as exc:
                msg = f"Pass 2 full-document compilation failed for blocking term={term_display!r}: {exc}"
                soft_issues.append((f"{item_id}::{term_display}::pass2_full_document", msg))
                _write_fallback_compiled(msg)
                return (False, [])
            raw_base_path.write_text(pass2_raw_base.read_text(encoding="utf-8", errors="replace"), encoding="utf-8")
            context_path.write_text(f"(used full document contexts: {item_id}__contexts.full.txt)\n", encoding="utf-8")

            deps = [row.get("term") for row in (pass2_doc.get("unresolved_dependencies") or []) if isinstance(row, dict)]
            return (True, [d for d in deps if isinstance(d, str) and d.strip()])

    async def _process_item(item_id: str, client: Any) -> None:
        try:
            metric_doc = _load_metric_compiler_aggregate(paths=paths, item_id=item_id, metrics_output_subdir=metrics_output_subdir)
        except FileNotFoundError as exc:
            soft_issues.append((f"{item_id}::__missing_metrics__", str(exc)))
            return
        except Exception as exc:
            err_path = out_dir / f"{item_id}__metrics_compiler.error.txt"
            err_path.write_text(f"{exc}\n\n{traceback.format_exc()}", encoding="utf-8")
            hard_errors.append((f"{item_id}::__metrics_compiler__", str(exc)))
            return

        try:
            targets = _extract_blocking_term_targets(metric_doc)
        except Exception as exc:
            err_path = out_dir / f"{item_id}__metrics_compiler.schema.error.txt"
            err_path.write_text(f"{exc}\n\n{traceback.format_exc()}", encoding="utf-8")
            hard_errors.append((f"{item_id}::__metrics_compiler_schema__", str(exc)))
            return

        if not targets:
            soft_issues.append((f"{item_id}::__no_blocking_terms__", "No unresolved_dependencies / blocking terms found."))
            return

        canonical_text = prompt_view_path(paths, item_id).read_text(encoding="utf-8", errors="replace")
        anchors = load_anchor_catalog(paths, item_id)
        ordered_infos = sorted(anchors.values(), key=lambda a: int(a["order"]))
        all_anchor_ids = [str(info["anchor_id"]) for info in ordered_infos]
        all_anchor_ids = _filter_noise_anchor_ids(
            canonical_text=canonical_text,
            anchors=anchors,
            anchor_ids=all_anchor_ids,
        )
        full_allowed_anchor_ids = set(all_anchor_ids)

        full_contexts_block = build_excerpt_pack_from_canonical(
            canonical_text=canonical_text,
            catalog=anchors,
            anchor_ids=all_anchor_ids,
        )
        (out_dir / f"{item_id}__contexts.full.txt").write_text(full_contexts_block, encoding="utf-8")

        # Indexing hints for targeted pass 1.
        definitions_anchor_ids, definitions_span = _definitions_anchor_ids_from_indexing_v2(paths, item_id, anchors)
        pricing_union_anchor_ids = _pricing_union_anchor_ids_from_indexing_v2(paths, item_id, anchors)
        financial_covenant_anchor_ids = _financial_covenant_anchor_ids_from_indexing_v2(paths, item_id, anchors)

        # Persist the extracted initial targets as an artifact to make the stage deterministic/auditable.
        (out_dir / f"{item_id}__blocking_term_targets.json").write_text(
            json.dumps([t.__dict__ for t in targets], indent=2) + "\n",
            encoding="utf-8",
        )

        # Recursive resolution: expand over newly discovered unresolved_dependencies up to max_depth.
        term_display_by_key: dict[str, str] = {}
        depth_by_key: dict[str, int] = {}
        root_refs_by_key: dict[str, set[str]] = defaultdict(set)
        edges_by_key: dict[str, set[str]] = defaultdict(set)

        frontier: list[str] = []
        for t in targets:
            disp = _normalize_term_display(t.term)
            key = _canonical_term_key(disp)
            if not disp or not key:
                continue
            term_display_by_key.setdefault(key, disp)
            depth_by_key.setdefault(key, 1)
            root_refs_by_key[key].update([x for x in (t.referenced_by or []) if isinstance(x, str) and x.strip()])
            frontier.append(key)

        # Stable de-dupe; preserve order.
        seen_frontier: set[str] = set()
        frontier = [k for k in frontier if not (k in seen_frontier or seen_frontier.add(k))]

        compiled_keys: set[str] = set()
        term_cap_hit = False
        for depth in range(1, max_depth + 1):
            current = [k for k in frontier if depth_by_key.get(k) == depth and k not in compiled_keys]
            if not current:
                break

            current_targets = [
                BlockingTermTarget(term=term_display_by_key[k], referenced_by=sorted(list(root_refs_by_key.get(k) or set())))
                for k in current
            ]

            results = await asyncio.gather(
                *(
                    _process_term(
                        item_id=item_id,
                        target=t,
                        canonical_text=canonical_text,
                        anchors=anchors,
                        definitions_anchor_ids=definitions_anchor_ids,
                        definitions_span=definitions_span,
                        pricing_union_anchor_ids=pricing_union_anchor_ids,
                        financial_covenant_anchor_ids=financial_covenant_anchor_ids,
                        full_contexts_block=full_contexts_block,
                        full_allowed_anchor_ids=full_allowed_anchor_ids,
                        client=client,
                    )
                    for t in current_targets
                ),
                return_exceptions=True,
            )

            next_frontier: list[str] = []
            for k, res in zip(current, results):
                if isinstance(res, Exception):
                    hard_errors.append((f"{item_id}::__term_task__::{term_display_by_key.get(k, k)}", str(res)))
                    compiled_keys.add(k)
                    continue

                ok, deps = res
                compiled_keys.add(k)
                if not ok:
                    continue

                for dep in deps:
                    dep_disp = _normalize_term_display(dep)
                    dep_key = _canonical_term_key(dep_disp)
                    if not dep_disp or not dep_key:
                        continue

                    if dep_key not in term_display_by_key and len(term_display_by_key) >= max_terms:
                        if not term_cap_hit:
                            soft_issues.append(
                                (
                                    f"{item_id}::__max_terms__",
                                    f"Reached max_terms={max_terms}; skipping discovery of additional dependency terms.",
                                )
                            )
                            term_cap_hit = True
                        continue

                    edges_by_key[k].add(dep_key)
                    term_display_by_key.setdefault(dep_key, dep_disp)
                    root_refs_by_key[dep_key].update(root_refs_by_key.get(k, set()))

                    if dep_key not in depth_by_key:
                        depth_by_key[dep_key] = depth + 1
                    if depth_by_key.get(dep_key) == depth + 1 and depth + 1 <= max_depth:
                        next_frontier.append(dep_key)

            # Stable de-dupe for next layer.
            seen_next: set[str] = set()
            frontier = [k for k in next_frontier if not (k in seen_next or seen_next.add(k))]

        # Build an item-level aggregate for convenience.
        compiled_terms: list[dict[str, Any]] = []
        deps: dict[str, set[str]] = {}
        successful_keys: set[str] = set()
        for k in sorted(compiled_keys, key=lambda kk: term_display_by_key.get(kk, kk)):
            safe_term = _safe_slug(term_display_by_key.get(k, k))
            compiled_path = out_dir / f"{item_id}__{safe_term}__compiled.json"
            if not compiled_path.exists():
                continue
            try:
                doc = json.loads(compiled_path.read_text(encoding="utf-8"))
            except Exception:
                continue
            if not isinstance(doc, dict):
                continue
            defs = doc.get("definitions")
            if isinstance(defs, list) and defs and isinstance(defs[0], dict):
                compiled_terms.append(defs[0])
                successful_keys.add(k)
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

        unresolved: list[dict[str, Any]] = []
        for t, rb in sorted(deps.items()):
            dep_key = _canonical_term_key(_normalize_term_display(t))
            # If we successfully compiled this dependency term during this run, it is no longer unresolved
            # at the item level (it is a resolved node in the graph, though it remains an edge).
            if dep_key and dep_key in successful_keys:
                continue
            unresolved.append({"term": t, "referenced_by": sorted(list(rb))})
        (out_dir / f"{item_id}__compiled.json").write_text(
            json.dumps(
                {
                    "schema_version": "definition_compiler_v2_ast_v1",
                    "definitions": compiled_terms,
                    "unresolved_dependencies": unresolved,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

        # Persist a lightweight dependency graph for auditing/debugging (term keys + depths + root refs).
        graph_terms: list[dict[str, Any]] = []
        for k in sorted(term_display_by_key.keys(), key=lambda kk: (depth_by_key.get(kk, 10**9), term_display_by_key.get(kk, kk))):
            graph_terms.append(
                {
                    "term": term_display_by_key.get(k, k),
                    "term_key": k,
                    "depth": int(depth_by_key.get(k, 0) or 0),
                    "root_referenced_by": sorted(list(root_refs_by_key.get(k, set()))),
                    "compiled": k in compiled_keys,
                }
            )
        graph_edges: list[dict[str, Any]] = []
        for src_key, dst_keys in sorted(edges_by_key.items(), key=lambda kv: term_display_by_key.get(kv[0], kv[0])):
            for dst_key in sorted(dst_keys, key=lambda kk: term_display_by_key.get(kk, kk)):
                graph_edges.append(
                    {
                        "from": term_display_by_key.get(src_key, src_key),
                        "to": term_display_by_key.get(dst_key, dst_key),
                    }
                )

        (out_dir / f"{item_id}__dependency_graph.json").write_text(
            json.dumps(
                {
                    "schema_version": "blocking_terms_dependency_graph_v1",
                    "item_id": item_id,
                    "created_at": int(time.time()),
                    "max_depth": int(max_depth),
                    "terms": graph_terms,
                    "edges": graph_edges,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

    async def _runner() -> None:
        nonlocal term_sem
        term_sem = asyncio.Semaphore(max(1, concurrency))
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
            blocking_terms_compiler_v1_prompt=str(term_compiler_prompt_path),
            blocking_terms_compiler_v1_prompt_sha256=prompt_digest,
            blocking_terms_compiler_v1_input_metrics_subdir=metrics_output_subdir,
            blocking_terms_compiler_v1_output_subdir=output_subdir,
            blocking_terms_compiler_v1_max_depth=int(max_depth),
            blocking_terms_compiler_v1_max_terms=int(max_terms),
            blocking_terms_compiler_v1_created_at=int(time.time()),
        )

    if soft_issues:
        (out_dir / "issues.txt").write_text("\n".join(f"{k}: {v}" for k, v in soft_issues) + "\n", encoding="utf-8")

    if hard_errors:
        (out_dir / "errors.txt").write_text("\n".join(f"{k}: {v}" for k, v in hard_errors) + "\n", encoding="utf-8")
        raise RuntimeError(
            f"Blocking terms compiler v1 completed with hard errors (count={len(hard_errors)}); see {out_dir / 'errors.txt'}"
        )
