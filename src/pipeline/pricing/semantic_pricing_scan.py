from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pipeline.pricing.contract_pricing import _call_gateway
from pipeline.pricing.contract_pricing_plan_schemas import ContractPricingPlanV2
from pipeline.pricing.semantic_pricing_scan_schemas import (
    SemanticPricingScanChunkV1,
    SemanticPricingScanSummaryV1,
    SemanticPricingTableCandidate,
)
from pipeline.utils import assert_exists


def _estimate_tokens(text: str) -> int:
    # Rough heuristic: 1 token ≈ 4 characters.
    return max(0, len(text) // 4)


@dataclass(frozen=True)
class AnchorRec:
    anchor_id: str
    anchor_type: str
    order: int
    text: str


def _coerce_anchors(doc_ir: dict[str, Any]) -> list[AnchorRec]:
    anchors: list[AnchorRec] = []
    for a in doc_ir.get("anchors") or []:
        if not isinstance(a, dict):
            continue
        anchor_id = str(a.get("anchor_id") or "").strip()
        if not anchor_id:
            continue
        anchors.append(
            AnchorRec(
                anchor_id=anchor_id,
                anchor_type=str(a.get("anchor_type") or "").strip() or "unknown",
                order=int(a.get("order") or 0),
                text=str(a.get("text") or ""),
            )
        )
    anchors_sorted = sorted(anchors, key=lambda x: x.order)
    return anchors_sorted


def build_anchor_chunks(
    *,
    doc_ir: dict[str, Any],
    max_chunk_chars: int = 20_000,
    overlap_anchors: int = 5,
    max_anchors_per_chunk: int = 120,
) -> list[list[AnchorRec]]:
    """Chunk the full anchor stream with structural boundaries only (no heuristics).

    Guarantees:
    - Every anchor appears in at least one chunk.
    - Chunks never split anchors.
    """

    anchors = _coerce_anchors(doc_ir)
    if not anchors:
        raise ValueError("doc_ir has no anchors to scan")

    max_chunk_chars = max(1_000, int(max_chunk_chars))
    overlap_anchors = max(0, int(overlap_anchors))
    max_anchors_per_chunk = max(1, int(max_anchors_per_chunk))

    chunks: list[list[AnchorRec]] = []
    i = 0
    while i < len(anchors):
        current: list[AnchorRec] = []
        chars = 0
        while i < len(anchors) and len(current) < max_anchors_per_chunk:
            a = anchors[i]
            # Include a small fixed overhead for JSON keys + separators.
            add = len(a.text) + len(a.anchor_id) + 32
            if current and (chars + add) > max_chunk_chars:
                break
            current.append(a)
            chars += add
            i += 1

        if not current:
            # Single anchor exceeds max; still emit it to guarantee coverage.
            current = [anchors[i]]
            i += 1

        chunks.append(current)

        if overlap_anchors and i < len(anchors):
            i = max(0, i - overlap_anchors)

    # Sanity: ensure coverage
    covered = {a.anchor_id for chunk in chunks for a in chunk}
    missing = [a.anchor_id for a in anchors if a.anchor_id not in covered]
    if missing:
        raise RuntimeError(f"Chunking bug: {len(missing)} anchors were not covered (e.g., {missing[:3]})")

    return chunks


def _render_scan_prompt(template: str, *, chunk_json: str) -> str:
    template = template.strip()
    if "{chunk_json}" not in template:
        raise ValueError("Scan prompt template must include a `{chunk_json}` placeholder.")
    return template.replace("{chunk_json}", chunk_json)


def _render_plan_prompt(template: str, *, scan_summary_json: str, tables_pack_json: str) -> str:
    template = template.strip()
    if "{scan_summary_json}" not in template or "{tables_pack_json}" not in template:
        raise ValueError(
            "Semantic-scan planner prompt must include `{scan_summary_json}` and `{tables_pack_json}` placeholders."
        )
    return template.replace("{scan_summary_json}", scan_summary_json).replace("{tables_pack_json}", tables_pack_json)


def _tables_pack(doc_ir: dict[str, Any]) -> dict[str, Any]:
    tables = []
    for t in doc_ir.get("tables") or []:
        if not isinstance(t, dict):
            continue
        table_anchor_id = str(t.get("table_anchor_id") or "").strip()
        if not table_anchor_id:
            continue
        tables.append(
            {
                "table_anchor_id": table_anchor_id,
                "order": int(t.get("order") or 0),
                "structured": bool(t.get("structured")),
                "columns": t.get("columns"),
                "rows": t.get("rows"),
                "raw": t.get("raw"),
            }
        )
    return {
        "item_id": str(doc_ir.get("item_id") or ""),
        "n_tables": len(tables),
        "tables": sorted(tables, key=lambda x: int(x.get("order") or 0)),
    }


def _dedupe_table_candidates(candidates: list[SemanticPricingTableCandidate]) -> list[SemanticPricingTableCandidate]:
    order = {"high": 3, "medium": 2, "low": 1}
    best: dict[str, SemanticPricingTableCandidate] = {}
    for c in candidates:
        prev = best.get(c.table_anchor_id)
        if prev is None:
            best[c.table_anchor_id] = c
            continue
        if order.get(c.confidence, 0) > order.get(prev.confidence, 0):
            best[c.table_anchor_id] = c
            continue
        if order.get(c.confidence, 0) == order.get(prev.confidence, 0):
            # Merge evidence; keep existing rationale (first seen) to stay stable.
            merged_refs = sorted(set(prev.source_refs + c.source_refs))
            prev.source_refs = merged_refs  # type: ignore[misc]
    # Keep document order stable by sorting by table order if present in refs isn't accessible; fallback by id.
    return list(best.values())


async def semantic_pricing_scan(
    *,
    doc_ir: dict[str, Any],
    scan_prompt_path: Path,
    planner_prompt_path: Path,
    client: Any,
    scan_model: str,
    scan_reasoning: str | None,
    scan_temperature: float,
    planner_model: str,
    planner_reasoning: str | None,
    planner_temperature: float,
    max_chunk_chars: int = 20_000,
    overlap_anchors: int = 5,
    max_anchors_per_chunk: int = 120,
    attempts_per_chunk: int = 3,
    chunk_concurrency: int = 4,
    planner_attempts: int = 3,
    require_tables: bool = True,
    restrict_table_candidates_to_table_anchors: bool = True,
    restrict_plan_to_table_ids: bool = True,
    output_dir: Path | None = None,
) -> tuple[SemanticPricingScanSummaryV1, ContractPricingPlanV2]:
    """Scan the entire anchor stream (coverage-first) and then produce a pricing plan.

    This is the "Hebbia-like" guarantee: the model is shown every anchor via sequential chunks,
    then asked to synthesize a pricing-table plan grounded in those citations.
    """

    assert_exists(scan_prompt_path, message=f"Scan prompt not found: {scan_prompt_path}")
    assert_exists(planner_prompt_path, message=f"Planner prompt not found: {planner_prompt_path}")

    item_id = str(doc_ir.get("item_id") or "")
    anchors = _coerce_anchors(doc_ir)
    anchors_by_id = {a.anchor_id: a for a in anchors}
    table_ids = {str(t.get("table_anchor_id")) for t in (doc_ir.get("tables") or []) if isinstance(t, dict) and t.get("table_anchor_id")}
    if require_tables and not table_ids:
        raise ValueError(
            f"semantic_pricing_scan requires doc_ir.tables to be non-empty (item_id={item_id!r}). "
            "This document has no parsed tables, so table-based semantic scan planning cannot produce a ContractPricingPlanV2."
        )

    scan_template = scan_prompt_path.read_text()
    planner_template = planner_prompt_path.read_text()

    chunks = build_anchor_chunks(
        doc_ir=doc_ir,
        max_chunk_chars=max_chunk_chars,
        overlap_anchors=overlap_anchors,
        max_anchors_per_chunk=max_anchors_per_chunk,
    )

    seen_anchor_ids = {a.anchor_id for chunk in chunks for a in chunk}
    chunk_concurrency = max(1, int(chunk_concurrency))

    sem = asyncio.Semaphore(chunk_concurrency)

    async def _scan_chunk(idx: int, chunk: list[AnchorRec]) -> SemanticPricingScanChunkV1:
        async with sem:
            chunk_anchor_ids = [a.anchor_id for a in chunk]
            chunk_table_anchor_ids = [
                a.anchor_id
                for a in chunk
                if a.anchor_type == "table" or "[[TABLE]]" in (a.text or "")
            ]

            chunk_payload = {
                "item_id": item_id,
                "start_order": chunk[0].order,
                "end_order": chunk[-1].order,
                "table_anchor_ids_in_chunk": chunk_table_anchor_ids,
                "anchors": [
                    {"anchor_id": a.anchor_id, "anchor_type": a.anchor_type, "order": a.order, "text": a.text}
                    for a in chunk
                ],
            }
            chunk_json = json.dumps(chunk_payload, indent=2)
            prompt = _render_scan_prompt(scan_template, chunk_json=chunk_json)

            last_err: str | None = None
            parsed: SemanticPricingScanChunkV1 | None = None
            for attempt in range(1, max(1, int(attempts_per_chunk)) + 1):
                attempt_prompt = prompt
                if last_err:
                    attempt_prompt = (
                        f"{prompt}\n\n=== VALIDATION_ERRORS ===\n{last_err}\n\n"
                        "Regenerate the JSON to satisfy the schema and the errors above. Output JSON only."
                    )
                try:
                    raw = await _call_gateway(
                        client=client,
                        prompt=attempt_prompt,
                        model=scan_model,
                        temperature=scan_temperature,
                        reasoning=scan_reasoning,
                    )
                except Exception as exc:
                    msg = str(exc).strip()
                    if not msg:
                        msg = f"{type(exc).__name__}"
                    else:
                        msg = f"{type(exc).__name__}: {msg}"
                    last_err = f"Gateway call failed: {msg}"
                    continue
                try:
                    obj = SemanticPricingScanChunkV1.model_validate_json(raw)
                    if obj.item_id != item_id:
                        raise ValueError(f"item_id mismatch: got {obj.item_id!r}, expected {item_id!r}")

                    allowed = set(chunk_anchor_ids)
                    allowed_table_ids = set(chunk_table_anchor_ids)

                    def _check_refs(label: str, refs: list[str]) -> None:
                        bad = sorted(set(refs) - allowed)
                        if bad:
                            raise ValueError(f"{label} has source_refs outside this chunk: {', '.join(bad)}")

                    _check_refs("chunk.anchor_ids", obj.chunk.anchor_ids)
                    for c in obj.pricing_table_candidates:
                        if c.table_anchor_id not in allowed:
                            raise ValueError(
                                f"pricing_table_candidates includes table_anchor_id not in chunk: {c.table_anchor_id}"
                            )
                        if restrict_table_candidates_to_table_anchors and c.table_anchor_id not in allowed_table_ids:
                            # Fail loudly with a concrete allowlist to prevent the model from "upcasting" prose anchors
                            # into table candidates when the chunk contains no actual [[TABLE]] anchors.
                            hint = ", ".join(sorted(allowed_table_ids)) if allowed_table_ids else "(none in this chunk)"
                            raise ValueError(
                                "pricing_table_candidates.table_anchor_id must be one of CHUNK_JSON.table_anchor_ids_in_chunk; "
                                f"got {c.table_anchor_id}. Allowed table_anchor_ids_in_chunk: {hint}"
                            )
                        _check_refs(f"table_candidate[{c.table_anchor_id}].source_refs", c.source_refs)
                        if c.regime_hint_anchor_id:
                            if c.regime_hint_anchor_id not in allowed:
                                raise ValueError(
                                    f"table_candidate[{c.table_anchor_id}].regime_hint_anchor_id "
                                    f"is outside chunk: {c.regime_hint_anchor_id}"
                                )

                    for adj in obj.required_pricing_adjustments:
                        _check_refs(
                            f"required_pricing_adjustments[{adj.adjustment_anchor_id}].source_refs",
                            adj.source_refs,
                        )
                        if adj.adjustment_anchor_id not in allowed:
                            raise ValueError(
                                "required_pricing_adjustments includes adjustment_anchor_id not in chunk: "
                                f"{adj.adjustment_anchor_id}"
                            )
                        if adj.adjustment_anchor_id not in adj.source_refs:
                            raise ValueError(
                                f"required_pricing_adjustments[{adj.adjustment_anchor_id}] must include its own anchor_id in source_refs"
                            )

                    for d in obj.definitions:
                        _check_refs(f"definitions[{d.defined_term}].source_refs", d.source_refs)

                    parsed = obj
                    break
                except Exception as exc:
                    last_err = str(exc)
                    continue

            if parsed is None:
                raise RuntimeError(
                    f"Semantic scan failed for chunk {idx+1}/{len(chunks)} "
                    f"(orders {chunk[0].order}-{chunk[-1].order}). Last error: {last_err}"
                )

            if output_dir is not None:
                output_dir.mkdir(parents=True, exist_ok=True)
                (output_dir / f"chunk_{idx:04d}.json").write_text(json.dumps(parsed.model_dump(), indent=2))

            return parsed

    # Run scan across all chunks with bounded concurrency.
    results = await asyncio.gather(*(_scan_chunk(i, c) for i, c in enumerate(chunks)))

    all_candidates: list[SemanticPricingTableCandidate] = []
    all_pricing_adjustments = []
    all_definitions = []

    for parsed in results:
        all_candidates.extend(parsed.pricing_table_candidates)
        all_pricing_adjustments.extend(parsed.required_pricing_adjustments)
        all_definitions.extend(parsed.definitions)

    covered_anchor_ids = sorted(seen_anchor_ids, key=lambda aid: anchors_by_id.get(aid).order if aid in anchors_by_id else 10**9)
    summary = SemanticPricingScanSummaryV1(
        schema_version="semantic_pricing_scan_summary_v1",
        item_id=item_id,
        n_anchors=len(anchors),
        n_chunks=len(chunks),
        covered_anchor_ids=covered_anchor_ids,
        pricing_table_candidates=_dedupe_table_candidates(all_candidates),
        required_pricing_adjustments=all_pricing_adjustments,
        definitions=all_definitions,
    )

    if output_dir is not None:
        (output_dir / "summary.json").write_text(json.dumps(summary.model_dump(), indent=2))

    # --- Planner: synthesize a ContractPricingPlanV2 from scan summary + full table pack ------------

    tables_pack = _tables_pack(doc_ir)
    plan_prompt = _render_plan_prompt(
        planner_template,
        scan_summary_json=json.dumps(summary.model_dump(), indent=2),
        tables_pack_json=json.dumps(tables_pack, indent=2),
    )

    plan_last_err: str | None = None
    plan_model: ContractPricingPlanV2 | None = None
    for attempt in range(1, max(1, int(planner_attempts)) + 1):
        attempt_prompt = plan_prompt
        if plan_last_err:
            attempt_prompt = (
                f"{plan_prompt}\n\n=== VALIDATION_ERRORS ===\n{plan_last_err}\n\n"
                "Regenerate the JSON to satisfy the schema and the errors above. Output JSON only."
            )
        try:
            raw = await _call_gateway(
                client=client,
                prompt=attempt_prompt,
                model=planner_model,
                temperature=planner_temperature,
                reasoning=planner_reasoning,
            )
        except Exception as exc:
            msg = str(exc).strip()
            if not msg:
                msg = f"{type(exc).__name__}"
            else:
                msg = f"{type(exc).__name__}: {msg}"
            plan_last_err = f"Gateway call failed: {msg}"
            continue
        try:
            plan = ContractPricingPlanV2.model_validate_json(raw)
            if plan.item_id != item_id:
                raise ValueError(f"item_id mismatch: plan has {plan.item_id!r}, expected {item_id!r}")
            if restrict_plan_to_table_ids:
                # Table IDs must exist.
                missing = sorted({t.table_anchor_id for t in plan.selected_tables} - table_ids)
                if missing:
                    raise ValueError("Plan selected unknown table_anchor_id(s): " + ", ".join(missing))
            else:
                # Anchor IDs must exist (plan may select non-table anchors as extraction units).
                plan_ids = {t.table_anchor_id for t in plan.selected_tables}
                missing = sorted(plan_ids - set(anchors_by_id.keys()))
                if missing:
                    raise ValueError("Plan selected unknown anchor_id(s): " + ", ".join(missing))
            plan_model = plan
            break
        except Exception as exc:
            plan_last_err = str(exc)
            continue

    if plan_model is None:
        raise RuntimeError(f"Semantic-scan planner failed after {planner_attempts} attempts. Last error: {plan_last_err}")

    if output_dir is not None:
        (output_dir / "plan.json").write_text(json.dumps(plan_model.model_dump(), indent=2))

    return summary, plan_model


async def semantic_pricing_scan_only(
    *,
    doc_ir: dict[str, Any],
    scan_prompt_path: Path,
    client: Any,
    scan_model: str,
    scan_reasoning: str | None,
    scan_temperature: float,
    max_chunk_chars: int = 20_000,
    overlap_anchors: int = 5,
    max_anchors_per_chunk: int = 120,
    attempts_per_chunk: int = 3,
    chunk_concurrency: int = 4,
    restrict_table_candidates_to_table_anchors: bool = True,
    output_dir: Path | None = None,
) -> SemanticPricingScanSummaryV1:
    """Scan the entire anchor stream (coverage-first) without producing a downstream plan.

    This is useful when you want to build your own compiler/IR synthesis step that is *not*
    expressed as a ContractPricingPlanV2 of tables.
    """

    assert_exists(scan_prompt_path, message=f"Scan prompt not found: {scan_prompt_path}")

    item_id = str(doc_ir.get("item_id") or "")
    anchors = _coerce_anchors(doc_ir)
    anchors_by_id = {a.anchor_id: a for a in anchors}

    scan_template = scan_prompt_path.read_text()

    chunks = build_anchor_chunks(
        doc_ir=doc_ir,
        max_chunk_chars=max_chunk_chars,
        overlap_anchors=overlap_anchors,
        max_anchors_per_chunk=max_anchors_per_chunk,
    )

    seen_anchor_ids = {a.anchor_id for chunk in chunks for a in chunk}
    chunk_concurrency = max(1, int(chunk_concurrency))
    sem = asyncio.Semaphore(chunk_concurrency)

    async def _scan_chunk(idx: int, chunk: list[AnchorRec]) -> SemanticPricingScanChunkV1:
        async with sem:
            chunk_anchor_ids = [a.anchor_id for a in chunk]
            chunk_table_anchor_ids = [
                a.anchor_id
                for a in chunk
                if a.anchor_type == "table" or "[[TABLE]]" in (a.text or "")
            ]

            chunk_payload = {
                "item_id": item_id,
                "start_order": chunk[0].order,
                "end_order": chunk[-1].order,
                "table_anchor_ids_in_chunk": chunk_table_anchor_ids,
                "anchors": [
                    {"anchor_id": a.anchor_id, "anchor_type": a.anchor_type, "order": a.order, "text": a.text}
                    for a in chunk
                ],
            }
            chunk_json = json.dumps(chunk_payload, indent=2)
            prompt = _render_scan_prompt(scan_template, chunk_json=chunk_json)

            last_err: str | None = None
            parsed: SemanticPricingScanChunkV1 | None = None
            for attempt in range(1, max(1, int(attempts_per_chunk)) + 1):
                attempt_prompt = prompt
                if last_err:
                    attempt_prompt = (
                        f"{prompt}\n\n=== VALIDATION_ERRORS ===\n{last_err}\n\n"
                        "Regenerate the JSON to satisfy the schema and the errors above. Output JSON only."
                    )
                try:
                    raw = await _call_gateway(
                        client=client,
                        prompt=attempt_prompt,
                        model=scan_model,
                        temperature=scan_temperature,
                        reasoning=scan_reasoning,
                    )
                except Exception as exc:
                    msg = str(exc).strip()
                    if not msg:
                        msg = f"{type(exc).__name__}"
                    else:
                        msg = f"{type(exc).__name__}: {msg}"
                    last_err = f"Gateway call failed: {msg}"
                    continue
                try:
                    obj = SemanticPricingScanChunkV1.model_validate_json(raw)
                    if obj.item_id != item_id:
                        raise ValueError(f"item_id mismatch: got {obj.item_id!r}, expected {item_id!r}")

                    allowed = set(chunk_anchor_ids)
                    allowed_table_ids = set(chunk_table_anchor_ids)

                    def _check_refs(label: str, refs: list[str]) -> None:
                        bad = sorted(set(refs) - allowed)
                        if bad:
                            raise ValueError(f"{label} has source_refs outside this chunk: {', '.join(bad)}")

                    _check_refs("chunk.anchor_ids", obj.chunk.anchor_ids)
                    for c in obj.pricing_table_candidates:
                        if c.table_anchor_id not in allowed:
                            raise ValueError(
                                f"pricing_table_candidates includes table_anchor_id not in chunk: {c.table_anchor_id}"
                            )
                        if restrict_table_candidates_to_table_anchors and c.table_anchor_id not in allowed_table_ids:
                            hint = ", ".join(sorted(allowed_table_ids)) if allowed_table_ids else "(none in this chunk)"
                            raise ValueError(
                                "pricing_table_candidates.table_anchor_id must be one of CHUNK_JSON.table_anchor_ids_in_chunk; "
                                f"got {c.table_anchor_id}. Allowed table_anchor_ids_in_chunk: {hint}"
                            )
                        _check_refs(f"table_candidate[{c.table_anchor_id}].source_refs", c.source_refs)
                        if c.regime_hint_anchor_id:
                            if c.regime_hint_anchor_id not in allowed:
                                raise ValueError(
                                    f"table_candidate[{c.table_anchor_id}].regime_hint_anchor_id "
                                    f"is outside chunk: {c.regime_hint_anchor_id}"
                                )

                    for adj in obj.required_pricing_adjustments:
                        _check_refs(
                            f"required_pricing_adjustments[{adj.adjustment_anchor_id}].source_refs",
                            adj.source_refs,
                        )
                        if adj.adjustment_anchor_id not in allowed:
                            raise ValueError(
                                "required_pricing_adjustments includes adjustment_anchor_id not in chunk: "
                                f"{adj.adjustment_anchor_id}"
                            )
                        if adj.adjustment_anchor_id not in adj.source_refs:
                            raise ValueError(
                                f"required_pricing_adjustments[{adj.adjustment_anchor_id}] must include its own anchor_id in source_refs"
                            )

                    for d in obj.definitions:
                        _check_refs(f"definitions[{d.defined_term}].source_refs", d.source_refs)

                    parsed = obj
                    break
                except Exception as exc:
                    last_err = str(exc)
                    continue

            if parsed is None:
                raise RuntimeError(
                    f"Semantic scan failed for chunk {idx+1}/{len(chunks)} "
                    f"(orders {chunk[0].order}-{chunk[-1].order}). Last error: {last_err}"
                )

            if output_dir is not None:
                output_dir.mkdir(parents=True, exist_ok=True)
                (output_dir / f"chunk_{idx:04d}.json").write_text(json.dumps(parsed.model_dump(), indent=2))

            return parsed

    results = await asyncio.gather(*(_scan_chunk(i, c) for i, c in enumerate(chunks)))

    all_candidates: list[SemanticPricingTableCandidate] = []
    all_pricing_adjustments = []
    all_definitions = []

    for parsed in results:
        all_candidates.extend(parsed.pricing_table_candidates)
        all_pricing_adjustments.extend(parsed.required_pricing_adjustments)
        all_definitions.extend(parsed.definitions)

    covered_anchor_ids = sorted(
        seen_anchor_ids,
        key=lambda aid: anchors_by_id.get(aid).order if aid in anchors_by_id else 10**9,
    )
    summary = SemanticPricingScanSummaryV1(
        schema_version="semantic_pricing_scan_summary_v1",
        item_id=item_id,
        n_anchors=len(anchors),
        n_chunks=len(chunks),
        covered_anchor_ids=covered_anchor_ids,
        pricing_table_candidates=_dedupe_table_candidates(all_candidates),
        required_pricing_adjustments=all_pricing_adjustments,
        definitions=all_definitions,
    )

    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "summary.json").write_text(json.dumps(summary.model_dump(), indent=2))

    return summary
