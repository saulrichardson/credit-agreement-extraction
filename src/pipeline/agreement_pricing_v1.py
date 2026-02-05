from __future__ import annotations

import asyncio
import json
import time
import traceback
from pathlib import Path
from typing import Any, Iterable

from .config import Paths, REQUIRED_MODEL, REQUIRED_REASONING, prompt_hash, update_manifest
from .contract_pricing import _call_gateway
from .doc_ir import build_doc_ir, write_doc_ir
from .pricing_schema_v1 import AgreementPricingModelV1
from .semantic_pricing_scan import semantic_pricing_scan_only
from .semantic_pricing_scan_schemas import SemanticPricingScanSummaryV1
from .utils import assert_exists

# Reuse gateway helper/constants from indexing to avoid duplicating config.
from .indexing import (  # type: ignore
    _ensure_gateway_client_async,
    DEFAULT_GATEWAY_URL,
)


_PLACEHOLDER_TEXT_MARKERS = (
    "verbatim",
    "no_spaces_id",
    "output schema",
    "context_json",
)


def _has_placeholder_text(value: str) -> bool:
    hay = (value or "").strip().lower()
    return any(marker in hay for marker in _PLACEHOLDER_TEXT_MARKERS)


def _render_prompt(template: str, *, context_json: str) -> str:
    template = template.strip()
    if "{context_json}" not in template:
        raise ValueError("Compiler prompt template must include a `{context_json}` placeholder.")
    return template.replace("{context_json}", context_json)


def _truncate_for_prompt(text: str, *, limit: int = 8_000) -> str:
    """Keep retry prompts small enough to stay effective.

    Pydantic can emit very long error strings; feeding them back verbatim often causes the model to fail
    to respond with valid JSON. We keep the most actionable prefix and point to the saved error file.
    """

    s = str(text or "")
    if len(s) <= limit:
        return s
    return s[:limit].rstrip() + f"\n... (truncated; {len(s)} chars total)\n"


def _collect_source_refs(obj: Any) -> set[str]:
    """Recursively collect any source_refs lists from a (nested) JSON-like structure."""

    out: set[str] = set()
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k == "source_refs" and isinstance(v, list):
                for aid in v:
                    if isinstance(aid, str) and aid.strip():
                        out.add(aid.strip())
                continue
            out.update(_collect_source_refs(v))
        return out
    if isinstance(obj, list):
        for v in obj:
            out.update(_collect_source_refs(v))
        return out
    return out


def _validate_schedule_completeness(model: AgreementPricingModelV1) -> None:
    """Hard structural checks to make the extracted pricing machine-evaluable.

    These checks are intentionally format-agnostic; they validate *internal consistency*:
    - Fixed schedules cover exactly the rate options the parameter claims to apply to.
    - Tiered schedules cover a full tier x rate_option matrix for the scheme/parameter.
    """

    schemes_by_id = {s.scheme_id: s for s in model.tiering_schemes}

    for p in model.pricing_parameters:
        expected_rate_option_ids = list(p.rate_option_ids or [])
        expected_rate_set = set(expected_rate_option_ids)
        if not expected_rate_set:
            raise ValueError(f"PricingParameter {p.parameter_id!r} has empty rate_option_ids")

        for rule in p.rules:
            sched = rule.schedule
            if sched.type == "fixed":
                actual = [v.rate_option_id for v in sched.values]
                actual_set = set(actual)
                if actual_set != expected_rate_set:
                    missing = sorted(expected_rate_set - actual_set)
                    extra = sorted(actual_set - expected_rate_set)
                    raise ValueError(
                        "FixedSchedule rate_option coverage mismatch for "
                        f"parameter={p.parameter_id!r} rule={rule.rule_id!r}: "
                        + (
                            (f"missing={missing}; " if missing else "")
                            + (f"extra={extra}; " if extra else "")
                        ).strip()
                    )
                if len(actual) != len(actual_set):
                    raise ValueError(
                        f"FixedSchedule has duplicate rate_option_id values for parameter={p.parameter_id!r} rule={rule.rule_id!r}"
                    )
                continue

            if sched.type == "tiered":
                scheme = schemes_by_id.get(sched.tiering_scheme_id)
                if scheme is None:
                    raise ValueError(
                        f"TieredSchedule references unknown tiering_scheme_id {sched.tiering_scheme_id!r}"
                    )
                tier_ids = [t.tier_id for t in scheme.tiers]
                tier_set = set(tier_ids)
                if not tier_set:
                    raise ValueError(f"TieringScheme {scheme.scheme_id!r} has no tiers")

                expected_pairs = {(tid, rid) for tid in tier_set for rid in expected_rate_set}
                actual_pairs = {(c.tier_id, c.rate_option_id) for c in sched.cells}
                missing_pairs = sorted(expected_pairs - actual_pairs)
                extra_pairs = sorted(actual_pairs - expected_pairs)
                if missing_pairs:
                    raise ValueError(
                        "TieredSchedule missing tier/rate_option cells for "
                        f"parameter={p.parameter_id!r} rule={rule.rule_id!r}: "
                        f"missing_pairs_count={len(missing_pairs)} (e.g., {missing_pairs[:5]})"
                    )
                if extra_pairs:
                    raise ValueError(
                        "TieredSchedule has unexpected tier/rate_option cells for "
                        f"parameter={p.parameter_id!r} rule={rule.rule_id!r}: "
                        f"extra_pairs_count={len(extra_pairs)} (e.g., {extra_pairs[:5]})"
                    )

                # Dedupe protection.
                if len(actual_pairs) != len(sched.cells):
                    raise ValueError(
                        f"TieredSchedule has duplicate tier/rate_option cells for parameter={p.parameter_id!r} rule={rule.rule_id!r}"
                    )
                continue

            raise ValueError(f"Unknown schedule type: {getattr(sched, 'type', None)!r}")


def _validate_agreement_pricing_output(
    *,
    raw_text: str,
    context: dict[str, Any],
    valid_anchor_ids: set[str],
    required_anchor_ids: list[str],
) -> AgreementPricingModelV1:
    try:
        parsed = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"LLM output is not valid JSON: {exc}") from exc

    model = AgreementPricingModelV1.model_validate(parsed)

    dumped = model.model_dump()
    used_anchor_ids = _collect_source_refs(dumped)

    unknown = sorted(set(used_anchor_ids) - valid_anchor_ids)
    if unknown:
        raise ValueError(f"Output references unknown anchor_id(s): {', '.join(unknown[:12])}")

    required_missing = sorted(set(required_anchor_ids) - set(used_anchor_ids))
    if required_missing:
        raise ValueError(
            "Missing required anchor references (coverage failure): " + ", ".join(required_missing[:20])
        )

    # Stronger enforcement: any scan-identified "pricing table candidate" anchors must be represented
    # as schedules (not just cited somewhere in passing).
    #
    # We treat source_table_anchor_id as the canonical link from a PricingParameterRule.schedule back
    # to the anchor that contains the schedule.
    pricing_table_anchor_ids = [
        str(a).strip()
        for a in (context.get("pricing_table_anchor_ids") or [])
        if isinstance(a, str) and str(a).strip()
    ]
    if pricing_table_anchor_ids:
        schedule_sources: set[str] = set()
        for p in model.pricing_parameters:
            for rule in p.rules:
                sched = rule.schedule
                if getattr(sched, "source_table_anchor_id", None):
                    schedule_sources.add(str(sched.source_table_anchor_id))
                # Fall back to schedule.source_refs (some schedules may be prose-derived without a clear
                # single "table anchor"; still must cite the anchor explicitly).
                schedule_sources.update(getattr(sched, "source_refs", []) or [])
                if getattr(sched, "type", None) == "tiered":
                    for cell in getattr(sched, "cells", []) or []:
                        schedule_sources.update(getattr(cell, "source_refs", []) or [])
                if getattr(sched, "type", None) == "fixed":
                    for val in getattr(sched, "values", []) or []:
                        schedule_sources.update(getattr(val, "source_refs", []) or [])
        missing_pricing_tables = sorted(set(pricing_table_anchor_ids) - schedule_sources)
        if missing_pricing_tables:
            raise ValueError(
                "Missing pricing table candidates in schedules (must be represented via schedule.source_table_anchor_id "
                "or schedule/source cell source_refs): "
                + ", ".join(missing_pricing_tables[:20])
            )

    # Stronger enforcement: scan-identified pricing adjustment anchors must be represented somewhere
    # in a PricingAdjustment or PricingConstraint (not just cited in an unrelated field).
    adjustment_anchor_ids = [
        str(a).strip()
        for a in (context.get("adjustment_anchor_ids") or [])
        if isinstance(a, str) and str(a).strip()
    ]
    if adjustment_anchor_ids:
        adjustment_sources: set[str] = set()
        for p in model.pricing_parameters:
            for rule in p.rules:
                for adj in rule.adjustments:
                    adjustment_sources.update(adj.source_refs or [])
                    adjustment_sources.update(adj.when.evidence.source_refs or [])
                    if adj.when.expr is not None:
                        # expr has no source_refs, but keep structural hook in case we extend.
                        pass
                for c in rule.constraints:
                    adjustment_sources.update(c.source_refs or [])
                    adjustment_sources.update(c.evidence.source_refs or [])
        missing_adjs = sorted(set(adjustment_anchor_ids) - adjustment_sources)
        if missing_adjs:
            raise ValueError(
                "Missing required adjustment anchors in adjustments/constraints: " + ", ".join(missing_adjs[:20])
            )

    # Placeholder text checks: these are the most common "template copy" failure mode.
    failures: list[str] = []
    for ro in model.rate_options:
        if _has_placeholder_text(ro.label_raw):
            failures.append(f"RateOption.label_raw appears to be placeholder text: {ro.label_raw!r}")
    for f in model.facilities:
        if _has_placeholder_text(f.label_raw):
            failures.append(f"Facility.label_raw appears to be placeholder text: {f.label_raw!r}")
    for v in model.variables:
        if _has_placeholder_text(v.label_raw):
            failures.append(f"PricingVariable.label_raw appears to be placeholder text: {v.label_raw!r}")
    for s in model.tiering_schemes:
        if _has_placeholder_text(s.label_raw):
            failures.append(f"TieringScheme.label_raw appears to be placeholder text: {s.label_raw!r}")
        for t in s.tiers:
            if _has_placeholder_text(t.label_raw):
                failures.append(f"Tier.label_raw appears to be placeholder text: {t.label_raw!r}")
    for p in model.pricing_parameters:
        if _has_placeholder_text(p.label_raw):
            failures.append(f"PricingParameter.label_raw appears to be placeholder text: {p.label_raw!r}")

    if failures:
        raise ValueError("Placeholder text detected: " + "; ".join(failures[:8]))

    _validate_schedule_completeness(model)

    # Keep context unused for now (reserved for future enforceable invariants).
    _ = context
    return model


def _orders_by_anchor_id(doc_ir: dict[str, Any]) -> dict[str, int]:
    out: dict[str, int] = {}
    for a in doc_ir.get("anchors") or []:
        if not isinstance(a, dict):
            continue
        aid = str(a.get("anchor_id") or "").strip()
        if not aid:
            continue
        out[aid] = int(a.get("order") or 0)
    return out


def _anchor_payload_by_id(doc_ir: dict[str, Any]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for a in doc_ir.get("anchors") or []:
        if isinstance(a, dict) and a.get("anchor_id"):
            out[str(a["anchor_id"])] = a
    return out


def _select_pricing_anchor_ids(
    *,
    summary: SemanticPricingScanSummaryV1,
    orders: dict[str, int],
    neighbor_window: int = 2,
) -> tuple[list[str], list[str], list[str], list[str]]:
    """Return (required_anchor_ids, context_anchor_ids).

    required_anchor_ids: minimal set that must be represented in output (coverage constraint)
    context_anchor_ids: expanded set provided to the compiler (includes neighbors and supporting refs)
    """

    required: set[str] = set()
    context: set[str] = set()
    pricing_table_anchor_ids: set[str] = set()
    adjustment_anchor_ids: set[str] = set()

    for c in summary.pricing_table_candidates:
        pricing_table_anchor_ids.add(c.table_anchor_id)
        required.add(c.table_anchor_id)
        context.add(c.table_anchor_id)
        context.update(c.source_refs or [])
        if c.regime_hint_anchor_id:
            required.add(c.regime_hint_anchor_id)
            context.add(c.regime_hint_anchor_id)

    for adj in summary.required_pricing_adjustments:
        adjustment_anchor_ids.add(adj.adjustment_anchor_id)
        required.add(adj.adjustment_anchor_id)
        context.add(adj.adjustment_anchor_id)
        context.update(adj.source_refs or [])

    for d in summary.definitions:
        context.update(d.source_refs or [])

    # Expand the compiler context with deterministic neighbors (not pricing heuristics; just locality).
    neighbor_window = max(0, int(neighbor_window))
    if neighbor_window:
        by_order = {v: k for k, v in orders.items()}
        seed_ids = set(context)
        for aid in list(seed_ids):
            o = orders.get(aid)
            if o is None:
                continue
            for oo in range(o - neighbor_window, o + neighbor_window + 1):
                n_id = by_order.get(oo)
                if n_id:
                    context.add(n_id)

    required_ids = sorted(required, key=lambda aid: orders.get(aid, 10**9))
    context_ids = sorted(context, key=lambda aid: orders.get(aid, 10**9))
    pricing_table_ids = sorted(pricing_table_anchor_ids, key=lambda aid: orders.get(aid, 10**9))
    adjustment_ids = sorted(adjustment_anchor_ids, key=lambda aid: orders.get(aid, 10**9))
    return required_ids, context_ids, pricing_table_ids, adjustment_ids


def run_agreement_pricing_v1(
    paths: Paths,
    item_ids: Iterable[str],
    *,
    compiler_prompt_path: Path,
    scan_prompt_path: Path,
    gateway_url: str | None = None,
    compiler_model: str | None = None,
    compiler_reasoning: str | None = None,
    compiler_temperature: float = 0.0,
    scan_model: str = "openai:gpt-5-mini",
    scan_reasoning: str | None = "high",
    scan_temperature: float = 0.0,
    scan_max_chunk_chars: int = 20_000,
    scan_overlap_anchors: int = 5,
    scan_max_anchors_per_chunk: int = 120,
    scan_attempts_per_chunk: int = 3,
    neighbor_window: int = 2,
    concurrency: int = 1,
    attempts: int = 3,
    output_subdir: str = "agreement_pricing_v1",
) -> None:
    """Extract pricing as a computation-oriented program (AgreementPricingModelV1).

    This is explicitly a move away from "pricing grids" as the primary representation.
    Tables, bullets, and prose are treated as just anchored text; the output is a rule/program model.
    """

    assert_exists(compiler_prompt_path, message=f"Compiler prompt not found: {compiler_prompt_path}")
    assert_exists(scan_prompt_path, message=f"Scan prompt not found: {scan_prompt_path}")

    compiler_model = compiler_model or REQUIRED_MODEL
    compiler_reasoning = compiler_reasoning or REQUIRED_REASONING

    compiler_prompt_sha = prompt_hash(compiler_prompt_path)
    compiler_template = compiler_prompt_path.read_text()

    out_root = paths.run_dir / "agreement_pricing" / output_subdir
    out_root.mkdir(parents=True, exist_ok=True)

    doc_ir_dir = paths.run_dir / "doc_ir"
    doc_ir_dir.mkdir(parents=True, exist_ok=True)

    scan_root = paths.run_dir / "semantic_scan" / output_subdir
    scan_root.mkdir(parents=True, exist_ok=True)

    items = list(item_ids)
    GatewayAgentClient = _ensure_gateway_client_async()
    sem = asyncio.Semaphore(max(1, concurrency))
    errors: list[tuple[str, str]] = []

    async def _process(item_id: str, client: Any) -> None:
        async with sem:
            try:
                doc_ir = build_doc_ir(paths, item_id)
                write_doc_ir(doc_ir_dir / f"{item_id}.json", doc_ir)

                orders = _orders_by_anchor_id(doc_ir)
                anchors_by_id = _anchor_payload_by_id(doc_ir)
                valid_anchor_ids = set(anchors_by_id.keys())

                # 1) Coverage-first semantic scan over all anchors (Hebbia-like).
                scan_item_dir = scan_root / item_id
                scan_item_dir.mkdir(parents=True, exist_ok=True)
                effective_scan_model = str(scan_model or "").strip()
                if not effective_scan_model:
                    raise ValueError("scan_model must be non-empty")
                effective_scan_reasoning = scan_reasoning

                summary = await semantic_pricing_scan_only(
                    doc_ir=doc_ir,
                    scan_prompt_path=scan_prompt_path,
                    client=client,
                    scan_model=effective_scan_model,
                    scan_reasoning=effective_scan_reasoning,
                    scan_temperature=scan_temperature,
                    max_chunk_chars=scan_max_chunk_chars,
                    overlap_anchors=scan_overlap_anchors,
                    max_anchors_per_chunk=scan_max_anchors_per_chunk,
                    attempts_per_chunk=scan_attempts_per_chunk,
                    restrict_table_candidates_to_table_anchors=False,
                    output_dir=scan_item_dir,
                )

                required_anchor_ids, context_anchor_ids, pricing_table_anchor_ids, adjustment_anchor_ids = (
                    _select_pricing_anchor_ids(
                    summary=summary,
                    orders=orders,
                    neighbor_window=neighbor_window,
                )
                )

                if not required_anchor_ids:
                    raise RuntimeError(
                        "Semantic scan did not identify any pricing anchors to compile "
                        f"(item_id={item_id!r})."
                    )

                anchors_payload: list[dict[str, Any]] = []
                for aid in context_anchor_ids:
                    a = anchors_by_id.get(aid)
                    if not a:
                        continue
                    anchors_payload.append(
                        {
                            "anchor_id": str(a.get("anchor_id") or ""),
                            "anchor_type": str(a.get("anchor_type") or ""),
                            "order": int(a.get("order") or 0),
                            "text": str(a.get("text") or ""),
                        }
                    )

                context = {
                    "item_id": item_id,
                    "required_anchor_ids": required_anchor_ids,
                    "pricing_table_anchor_ids": pricing_table_anchor_ids,
                    "adjustment_anchor_ids": adjustment_anchor_ids,
                    "anchors": anchors_payload,
                }

                # Persist the exact compilation context for audit/debug (this is the "artifact of truth").
                (out_root / f"{item_id}.context.json").write_text(json.dumps(context, indent=2))

                prompt = _render_prompt(compiler_template, context_json=json.dumps(context, indent=2))

                last_err: str | None = None
                raw: str | None = None
                for attempt in range(1, max(1, int(attempts)) + 1):
                    attempt_prompt = prompt
                    if last_err:
                        err_for_prompt = _truncate_for_prompt(last_err)
                        attempt_prompt = (
                            f"{prompt}\n\n=== VALIDATION_ERRORS ===\n{err_for_prompt}\n\n"
                            "Regenerate the JSON to satisfy the schema and the errors above. Output JSON only."
                        )
                    raw = await _call_gateway(
                        client=client,
                        prompt=attempt_prompt,
                        model=compiler_model,
                        temperature=compiler_temperature,
                        reasoning=compiler_reasoning,
                    )
                    (out_root / f"{item_id}.raw_attempt{attempt}.txt").write_text(raw)
                    try:
                        model = _validate_agreement_pricing_output(
                            raw_text=raw,
                            context=context,
                            valid_anchor_ids=valid_anchor_ids,
                            required_anchor_ids=required_anchor_ids,
                        )
                        out_path = out_root / f"{item_id}.json"
                        out_path.write_text(json.dumps(model.model_dump(), indent=2))

                        meta = {
                            "schema_version": "agreement_pricing_v1_artifact",
                            "stage": "agreement_pricing",
                            "run_id": paths.run_id,
                            "item_id": item_id,
                            "created_at": int(time.time()),
                            "gateway_url": gateway_url or DEFAULT_GATEWAY_URL,
                            "compiler_model": compiler_model,
                            "compiler_reasoning_effort": compiler_reasoning,
                            "compiler_temperature": compiler_temperature,
                        "compiler_prompt": str(compiler_prompt_path),
                        "compiler_prompt_sha256": compiler_prompt_sha,
                        "scan_prompt": str(scan_prompt_path),
                            "required_anchor_ids": required_anchor_ids,
                            "context_anchor_ids": context_anchor_ids,
                        }
                        (out_root / f"{item_id}.meta.json").write_text(json.dumps(meta, indent=2))
                        return
                    except Exception as exc:
                        last_err = str(exc)
                        (out_root / f"{item_id}.validation_attempt{attempt}.error.txt").write_text(last_err)
                        continue

                raise RuntimeError(f"Compiler failed after {attempts} attempts. Last error: {last_err}.")
            except Exception as exc:
                msg = str(exc).strip()
                if not msg:
                    msg = f"{type(exc).__name__}"
                else:
                    msg = f"{type(exc).__name__}: {msg}"
                errors.append((item_id, msg))
                (out_root / f"{item_id}.error.txt").write_text(f"{exc}\n\n{traceback.format_exc()}")

    async def _runner() -> None:
        async with GatewayAgentClient(
            base_url=gateway_url or DEFAULT_GATEWAY_URL,
            timeout=1200.0,
        ) as client:
            await asyncio.gather(*(_process(item_id, client) for item_id in items))

    asyncio.run(_runner())

    manifest_path = paths.manifest_path
    if manifest_path.exists():
        update_manifest(
            manifest_path,
            agreement_pricing_v1_prompt=str(compiler_prompt_path),
            agreement_pricing_v1_prompt_sha256=compiler_prompt_sha,
        )

    if errors:
        joined = "; ".join(f"{item}: {msg}" for item, msg in errors)
        raise RuntimeError(f"Agreement pricing v1 completed with errors: {joined}")
