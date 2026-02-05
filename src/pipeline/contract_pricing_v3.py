from __future__ import annotations

import asyncio
import json
import time
import traceback
from pathlib import Path
from typing import Any, Iterable, Literal

from .config import Paths, REQUIRED_MODEL, REQUIRED_REASONING, prompt_hash, update_manifest
from .contract_pricing import (
    _apply_rate_option_remap,
    _build_regimes_from_tables,
    _call_gateway,
    _merge_rate_options,
    _render_table_prompt,
    _validate_pricing_output,
    _validate_table_extract,
)
from .contract_pricing_plan_schemas import ContractPricingPlanV2
from .contract_pricing_planner import plan_contract_pricing_v2
from .contract_pricing_planner_table_pack import plan_contract_pricing_table_pack
from .contract_schemas import (
    ContractPricingArtifact,
    ContractPricingModel,
    ContractPricingTableExtract,
    EvidenceText,
    PricingAdjustment,
    PricingRegime,
    RateOption,
)
from .doc_ir import build_doc_ir, write_doc_ir
from .doc_tools_full import DocBrowserFull
from .table_critic_schemas import TableCritique
from .semantic_pricing_scan import semantic_pricing_scan
from .utils import assert_exists

from .llm.gateway import DEFAULT_GATEWAY_URL, _ensure_gateway_client_async


def _render_global_adjustments_prompt(
    template: str,
    *,
    adjustment_anchors_json: str,
    rate_options_json: str,
) -> str:
    template = template.strip()
    if "{adjustment_anchors_json}" not in template or "{rate_options_json}" not in template:
        raise ValueError(
            "Global-adjustments prompt must include `{adjustment_anchors_json}` and `{rate_options_json}` placeholders."
        )
    return template.replace("{adjustment_anchors_json}", adjustment_anchors_json).replace(
        "{rate_options_json}", rate_options_json
    )


async def _compile_global_pricing_adjustments_from_scan(
    *,
    client: Any,
    prompt_path: Path,
    model: str,
    reasoning: str,
    temperature: float,
    rate_options: list[RateOption],
    adjustment_anchors: list[dict[str, str]],
    valid_anchor_ids: set[str],
    required_anchor_ids: list[str],
    attempts: int = 3,
) -> list[PricingAdjustment]:
    """Compile scan-identified numeric adjustments that are not table-local.

    This is intentionally semantic (LLM-compiled) rather than heuristic parsing.
    """

    assert_exists(prompt_path, message=f"Global-adjustments prompt not found: {prompt_path}")
    template = prompt_path.read_text()

    prompt = _render_global_adjustments_prompt(
        template,
        adjustment_anchors_json=json.dumps(adjustment_anchors, indent=2),
        rate_options_json=json.dumps([ro.model_dump() for ro in rate_options], indent=2),
    )

    last_err: str | None = None
    for attempt in range(1, max(1, int(attempts)) + 1):
        attempt_prompt = prompt
        if last_err:
            attempt_prompt = (
                f"{prompt}\n\n=== VALIDATION_ERRORS ===\n{last_err}\n\n"
                "Regenerate the JSON to satisfy the schema and the errors above. Output JSON only."
            )
        raw = await _call_gateway(
            client=client,
            prompt=attempt_prompt,
            model=model,
            temperature=temperature,
            reasoning=reasoning,
        )
        try:
            parsed = json.loads(raw)
            if not isinstance(parsed, list):
                raise ValueError("Expected a JSON array of PricingAdjustment objects.")

            adjustments: list[PricingAdjustment] = []
            for idx, obj in enumerate(parsed):
                if not isinstance(obj, dict):
                    raise ValueError(f"Adjustment at index {idx} must be an object.")
                adj = PricingAdjustment.model_validate(obj)

                # Anchor integrity: forbid hallucinated anchor IDs.
                used = set(adj.source_refs or []) | set(adj.applies_when.source_refs or [])
                unknown = sorted(used - valid_anchor_ids)
                if unknown:
                    raise ValueError(
                        f"Adjustment {adj.adjustment_id} references unknown anchor_id(s): {', '.join(unknown)}"
                    )
                adjustments.append(adj)

            # Coverage: every required anchor must be referenced somewhere.
            referenced: set[str] = set()
            for adj in adjustments:
                referenced.update(adj.source_refs or [])
                referenced.update(adj.applies_when.source_refs or [])
            missing = sorted(set(required_anchor_ids) - referenced)
            if missing:
                raise ValueError(
                    "Missing required adjustment anchor references in compiled adjustments: " + ", ".join(missing)
                )

            return adjustments
        except Exception as exc:
            last_err = str(exc)
            continue

    raise RuntimeError(f"Global adjustment compilation failed after {attempts} attempts. Last error: {last_err}")


def run_contract_pricing_v3(
    paths: Paths,
    item_ids: Iterable[str],
    *,
    planner_prompt_path: Path,
    table_prompt_path: Path,
    gateway_url: str | None = None,
    planner_model: str = "openai:gpt-5-mini",
    planner_reasoning: str = "high",
    planner_temperature: float = 0.0,
    planner_mode: Literal["tool", "table_pack", "semantic_scan", "semantic_scan_any_anchor"] = "tool",
    scan_prompt_path: Path | None = None,
    scan_model: str | None = None,
    scan_reasoning: str | None = None,
    scan_temperature: float = 0.0,
    scan_max_chunk_chars: int = 20_000,
    scan_overlap_anchors: int = 5,
    scan_max_anchors_per_chunk: int = 120,
    scan_attempts_per_chunk: int = 3,
    planner_max_steps: int = 40,
    planner_attempts_per_step: int = 3,
    critic_prompt_path: Path | None = None,
    critic_model: str | None = None,
    critic_reasoning: str | None = None,
    critic_temperature: float = 0.0,
    compiler_temperature: float = 0.0,
    gateway_timeout: float | None = None,
    concurrency: int = 1,
    attempts: int = 3,
    output_subdir: str = "contract_pricing_v3",
) -> None:
    """End-to-end (v3): doc IR -> agentic planning (no heuristics, no embeddings) -> per-table compile -> validation.

    v3 goal: maximize fidelity/robustness by removing hand-coded pricing heuristics.

    Planner modes:
    - tool: agentic tool loop over full doc_ir (anchors + tables) via deterministic tools.
    - table_pack: one-shot planning from all parsed tables provided up-front (no tools).
    - semantic_scan: host-driven coverage scan over all anchors (chunked) -> plan from scan summary + full tables.
    - semantic_scan_any_anchor: like semantic_scan, but allows selecting non-table anchors as extraction units
      (supports agreements where pricing is encoded in prose/"ASCII tables" and doc_ir.tables may be empty).
    """

    assert_exists(planner_prompt_path, message=f"Planner prompt not found: {planner_prompt_path}")
    assert_exists(table_prompt_path, message=f"Table prompt not found: {table_prompt_path}")
    if planner_mode in ("semantic_scan", "semantic_scan_any_anchor"):
        if scan_prompt_path is None:
            raise ValueError("scan_prompt_path is required when planner_mode='semantic_scan'")
        assert_exists(scan_prompt_path, message=f"Scan prompt not found: {scan_prompt_path}")
    if critic_model:
        if critic_prompt_path is None:
            raise ValueError("critic_prompt_path is required when critic_model is set")
        assert_exists(critic_prompt_path, message=f"Critic prompt not found: {critic_prompt_path}")

    planner_prompt_sha = prompt_hash(planner_prompt_path)
    _ = planner_prompt_path.read_text()

    table_prompt_sha = prompt_hash(table_prompt_path)
    table_prompt_template = table_prompt_path.read_text()

    critic_prompt_template: str | None = None
    critic_prompt_sha: str | None = None
    if critic_model and critic_prompt_path is not None:
        critic_prompt_sha = prompt_hash(critic_prompt_path)
        critic_prompt_template = critic_prompt_path.read_text()

    compiler_model = REQUIRED_MODEL
    compiler_reasoning = REQUIRED_REASONING

    out_root = paths.run_dir / "contract_pricing" / output_subdir
    out_root.mkdir(parents=True, exist_ok=True)

    doc_ir_dir = paths.run_dir / "doc_ir"
    doc_ir_dir.mkdir(parents=True, exist_ok=True)

    plan_dir = paths.run_dir / "contract_pricing_plan" / output_subdir
    plan_dir.mkdir(parents=True, exist_ok=True)

    context_dir = paths.run_dir / "contract_pricing" / "_context" / output_subdir
    context_dir.mkdir(parents=True, exist_ok=True)

    scan_root = paths.run_dir / "semantic_scan" / output_subdir
    scan_root.mkdir(parents=True, exist_ok=True)

    items = list(item_ids)
    GatewayAgentClient = _ensure_gateway_client_async()
    sem = asyncio.Semaphore(max(1, concurrency))
    errors: list[tuple[str, str]] = []

    async def _process(item_id: str, client: Any) -> None:
        async with sem:
            try:
                # 1) Build + persist doc IR
                doc_ir = build_doc_ir(paths, item_id)
                write_doc_ir(doc_ir_dir / f"{item_id}.json", doc_ir)

                browser = DocBrowserFull(doc_ir)

                # 2) Planning (no heuristics)
                scan_summary = None
                if planner_mode in ("semantic_scan", "semantic_scan_any_anchor"):
                    scan_item_dir = scan_root / item_id
                    scan_item_dir.mkdir(parents=True, exist_ok=True)
                    effective_scan_model = scan_model or planner_model
                    effective_scan_reasoning = scan_reasoning or planner_reasoning
                    summary, plan_model = await semantic_pricing_scan(
                        doc_ir=doc_ir,
                        scan_prompt_path=scan_prompt_path,
                        planner_prompt_path=planner_prompt_path,
                        client=client,
                        scan_model=effective_scan_model,
                        scan_reasoning=effective_scan_reasoning,
                        scan_temperature=scan_temperature,
                        planner_model=planner_model,
                        planner_reasoning=planner_reasoning,
                        planner_temperature=planner_temperature,
                        max_chunk_chars=scan_max_chunk_chars,
                        overlap_anchors=scan_overlap_anchors,
                        max_anchors_per_chunk=scan_max_anchors_per_chunk,
                        attempts_per_chunk=scan_attempts_per_chunk,
                        planner_attempts=max(1, planner_attempts_per_step),
                        require_tables=(planner_mode == "semantic_scan"),
                        restrict_table_candidates_to_table_anchors=(planner_mode == "semantic_scan"),
                        restrict_plan_to_table_ids=(planner_mode == "semantic_scan"),
                        output_dir=scan_item_dir,
                    )
                    scan_summary = summary
                elif planner_mode == "table_pack":
                    plan_model = await plan_contract_pricing_table_pack(
                        doc_ir=doc_ir,
                        planner_prompt_path=planner_prompt_path,
                        gateway_url=gateway_url,
                        planner_model=planner_model,
                        planner_reasoning=planner_reasoning,
                        planner_temperature=planner_temperature,
                        gateway_timeout=gateway_timeout or 600.0,
                        attempts=max(1, planner_attempts_per_step),
                        client=client,
                    )
                elif planner_mode == "tool":
                    plan_model = await plan_contract_pricing_v2(
                        browser=browser,  # type: ignore[arg-type]
                        planner_prompt_path=planner_prompt_path,
                        gateway_url=gateway_url,
                        planner_model=planner_model,
                        planner_reasoning=planner_reasoning,
                        planner_temperature=planner_temperature,
                        gateway_timeout=gateway_timeout or 600.0,
                        catalog_max_tables=None,
                        max_steps=planner_max_steps,
                        attempts_per_step=planner_attempts_per_step,
                        client=client,
                    )
                else:
                    raise ValueError(f"Unknown planner_mode: {planner_mode!r}")
                (plan_dir / f"{item_id}.json").write_text(json.dumps(plan_model.model_dump(), indent=2))

                # 3) Per-table extraction driven by plan
                per_table_dir = context_dir / item_id
                per_table_dir.mkdir(parents=True, exist_ok=True)

                table_contexts: list[dict[str, Any]] = []
                extracts_by_table: dict[str, ContractPricingTableExtract] = {}

                anchors_by_id = {
                    str(a.get("anchor_id")): a
                    for a in (doc_ir.get("anchors") or [])
                    if isinstance(a, dict)
                }
                valid_anchor_ids = set(anchors_by_id.keys())

                # "Global" adjustment anchors discovered by the full-document scan.
                #
                # Important: ContractPricingModel.PricingAdjustment requires a numeric delta_bps, so we only
                # enforce/propagate scan adjustments that include an explicit numeric delta.
                numeric_adjustment_anchor_ids: list[str] = []
                global_adjustment_anchors = None
                if scan_summary is not None and scan_summary.required_pricing_adjustments:
                    numeric = [a for a in scan_summary.required_pricing_adjustments if a.delta_bps is not None]
                    adj_ids = sorted({a.adjustment_anchor_id for a in numeric})
                    numeric_adjustment_anchor_ids = adj_ids
                    global_adjustment_anchors = [
                        {"anchor_id": aid, "text": str((anchors_by_id.get(aid) or {}).get("text") or "").strip()}
                        for aid in adj_ids
                        if str((anchors_by_id.get(aid) or {}).get("text") or "").strip()
                    ]

                for table_plan in plan_model.selected_tables:
                    table_anchor_id = table_plan.table_anchor_id
                    try:
                        table_context = browser.open_table(table_anchor_id, neighbor_window=10)
                    except KeyError:
                        # Allow "semantic_scan_any_anchor" and tool planners to select non-table anchors as
                        # extraction units (e.g., pricing grids encoded as fixed-width text).
                        anchor_ctx = browser.open_anchor(table_anchor_id, neighbor_window=10)
                        anchor = anchor_ctx.get("anchor") or {}
                        table_context = {
                            "item_id": browser.item_id,
                            "table": {
                                "table_anchor_id": table_anchor_id,
                                "order": anchor.get("order") or 0,
                                "structured": False,
                                "columns": [],
                                "rows": [],
                                "raw": str(anchor.get("text") or ""),
                                "table_score": None,
                                "table_score_signals": [],
                                "regime_hint": None,
                            },
                            "neighbor_anchors": anchor_ctx.get("neighbors") or [],
                            "tail_anchors": [],
                            "adjustment_hints": [],
                        }
                    table_context["table_plan"] = table_plan.model_dump()
                    if global_adjustment_anchors:
                        table_context["global_adjustment_anchors"] = global_adjustment_anchors

                    # Optional regime override from plan.
                    if table_plan.regime_hint_anchor_id:
                        hint_anchor = anchors_by_id.get(table_plan.regime_hint_anchor_id)
                        hint_text = str((hint_anchor or {}).get("text") or "").strip()
                        if not hint_text:
                            raise RuntimeError(
                                f"Plan referenced regime_hint_anchor_id {table_plan.regime_hint_anchor_id} "
                                f"but that anchor text is missing/empty."
                            )
                        table_context["table"]["regime_hint"] = {
                            "anchor_id": table_plan.regime_hint_anchor_id,
                            "text": hint_text,
                        }

                    table_contexts.append(table_context)
                    (per_table_dir / f"{table_anchor_id}.json").write_text(
                        json.dumps(table_context, indent=2)
                    )

                    rendered = _render_table_prompt(
                        table_prompt_template, json.dumps(table_context, indent=2)
                    )
                    last_err: str | None = None
                    used = 0
                    for attempt in range(1, max(1, attempts) + 1):
                        used = attempt
                        prompt = rendered
                        if last_err:
                            prompt = (
                                f"{rendered}\n\n=== VALIDATION_ERRORS ===\n{last_err}\n\n"
                                "Regenerate the JSON to satisfy the schema and the errors above. Output JSON only."
                            )
                        try:
                            raw = await _call_gateway(
                                client=client,
                                prompt=prompt,
                                model=compiler_model,
                                temperature=compiler_temperature,
                                reasoning=compiler_reasoning,
                            )
                        except Exception as exc:
                            last_err = f"Gateway call failed: {type(exc).__name__}: {exc}"
                            continue
                        try:
                            try:
                                ex = ContractPricingTableExtract.model_validate_json(raw)
                            except Exception:
                                # Robustness: some models will echo extra top-level keys from TABLE_CONTEXT_JSON
                                # (e.g., "table_plan", "adjustment_hints"). We keep the schema strict but drop
                                # unknown top-level keys and retry parsing once.
                                parsed = json.loads(raw)
                                if isinstance(parsed, dict):
                                    allowed = set(ContractPricingTableExtract.model_fields.keys())
                                    cleaned = {k: v for k, v in parsed.items() if k in allowed}
                                    ex = ContractPricingTableExtract.model_validate(cleaned)
                                else:
                                    raise
                            if ex.grid.table_anchor_id != table_anchor_id:
                                raise ValueError(
                                    f"grid.table_anchor_id {ex.grid.table_anchor_id!r} "
                                    f"does not match expected {table_anchor_id!r}"
                                )
                            _validate_table_extract(ex, table_context=table_context)
                            if critic_model and critic_prompt_template:
                                critic_prompt = critic_prompt_template
                                if "{table_context_json}" not in critic_prompt or "{table_extract_json}" not in critic_prompt:
                                    raise ValueError(
                                        "Critic prompt template must include {table_context_json} and {table_extract_json} placeholders"
                                    )
                                critic_prompt = critic_prompt.replace(
                                    "{table_context_json}", json.dumps(table_context, indent=2)
                                ).replace("{table_extract_json}", json.dumps(ex.model_dump(), indent=2))
                                critic_raw = await _call_gateway(
                                    client=client,
                                    prompt=critic_prompt,
                                    model=critic_model,
                                    temperature=critic_temperature,
                                    reasoning=critic_reasoning,
                                )
                                critique = TableCritique.model_validate_json(critic_raw)
                                (per_table_dir / f"{table_anchor_id}.critic.json").write_text(
                                    json.dumps(critique.model_dump(), indent=2)
                                )
                                if critique.has_blockers:
                                    issues = [
                                        f"{i.code}: {i.message} (refs={','.join(i.source_refs or [])})"
                                        for i in critique.issues
                                    ]
                                    last_err = (
                                        "CRITIC flagged issues: "
                                        + "; ".join(issues[:6])
                                        + (
                                            f"; fix_instructions: {critique.fix_instructions}"
                                            if critique.fix_instructions
                                            else ""
                                        )
                                    )
                                    continue
                            extracts_by_table[table_anchor_id] = ex
                            break
                        except Exception as exc:
                            last_err = str(exc)

                    if table_anchor_id not in extracts_by_table:
                        raise RuntimeError(
                            f"Table {table_anchor_id} failed after {used} attempts. Last error: {last_err}"
                        )

                # 4) Canonicalize rate option IDs across tables and apply remap.
                global_rate_options, remap = _merge_rate_options(list(extracts_by_table.values()))
                for tid, ex in list(extracts_by_table.items()):
                    extracts_by_table[tid] = _apply_rate_option_remap(ex, remap)

                # 5) Assemble regimes using regime hints (if present).
                regimes = _build_regimes_from_tables(
                    table_contexts=table_contexts, extracts_by_table=extracts_by_table
                )

                pricing_model = ContractPricingModel(
                    issuer=None,
                    agreement={},
                    rate_options=global_rate_options,
                    pricing_regimes=regimes,
                )

                # 5.5) If the scan found explicit numeric adjustments that are NOT captured in any table-local
                # extraction, compile them directly and attach as a global regime.
                if scan_summary is not None and numeric_adjustment_anchor_ids:
                    referenced: set[str] = set()
                    for regime in pricing_model.pricing_regimes:
                        for adj in regime.adjustments:
                            referenced.update(adj.source_refs or [])
                            referenced.update(adj.applies_when.source_refs or [])
                    missing = sorted(set(numeric_adjustment_anchor_ids) - referenced)
                    if missing:
                        prompt_path = paths.root / "prompts" / "contract_pricing_adjustments_from_scan_v1.txt"
                        adjustment_anchors = [
                            {
                                "anchor_id": aid,
                                "text": str((anchors_by_id.get(aid) or {}).get("text") or "").strip(),
                            }
                            for aid in missing
                            if str((anchors_by_id.get(aid) or {}).get("text") or "").strip()
                        ]
                        compiled = await _compile_global_pricing_adjustments_from_scan(
                            client=client,
                            prompt_path=prompt_path,
                            model=compiler_model,
                            reasoning=compiler_reasoning,
                            temperature=compiler_temperature,
                            rate_options=pricing_model.rate_options,
                            adjustment_anchors=adjustment_anchors,
                            valid_anchor_ids=valid_anchor_ids,
                            required_anchor_ids=missing,
                            attempts=max(1, attempts),
                        )
                        pricing_model.pricing_regimes.append(
                            PricingRegime(
                                regime_id="global_adjustments",
                                label="Global pricing adjustments",
                                applies_when=EvidenceText(
                                    text="Applies when the condition in each adjustment is met.",
                                    source_refs=missing,
                                ),
                                grids=[],
                                adjustments=compiled,
                                flat_items=[],
                                source_refs=missing,
                            )
                        )

                # 5.75) Repair: some models omit TierTest.source_refs even when a test is present.
                #
                # This is a schema-required field, and failing here blocks the entire item even when the
                # test is clearly grounded by the tier/grid/table. For robustness, inherit evidence from
                # the parent tier (or the grid/table anchor) when missing.
                for regime in pricing_model.pricing_regimes:
                    for grid in regime.grids:
                        grid_fallback = [grid.table_anchor_id] if grid.table_anchor_id else []
                        for tier in grid.tiers:
                            tier_fallback = tier.source_refs or grid_fallback
                            for test in tier.tests:
                                # Pydantic models are mutable; validate_assignment is not enabled, so we can repair in-place.
                                if getattr(test, "source_refs", None) == []:
                                    test.source_refs = list(tier_fallback)  # type: ignore[assignment]

                context = {
                    "item_id": item_id,
                    "tables": [tc["table"] for tc in table_contexts],
                    # Adjustment enforcement:
                    # - In semantic_scan mode, require that all scan-identified required adjustments are represented.
                    # - In other modes, keep empty (no heuristic enforcement).
                    "adjustment_hints": (
                        [
                            {"anchor_id": aid}
                            for aid in numeric_adjustment_anchor_ids
                        ]
                        if scan_summary is not None
                        else []
                    ),
                }
                pricing_model = _validate_pricing_output(
                    raw_text=json.dumps(pricing_model.model_dump()),
                    context=context,
                )

                artifact = ContractPricingArtifact(
                    schema_version="contract_pricing_v3",
                    stage="contract_pricing",
                    run_id=paths.run_id,
                    item_id=item_id,
                    created_at=int(time.time()),
                    gateway_url=gateway_url or DEFAULT_GATEWAY_URL,
                    model=compiler_model,
                    reasoning_effort=compiler_reasoning,
                    temperature=compiler_temperature,
                    prompt=str(table_prompt_path),
                    prompt_sha256=table_prompt_sha,
                    attempts_used=attempts,
                    pricing=pricing_model,
                    planner_prompt=str(planner_prompt_path),
                    planner_prompt_sha256=planner_prompt_sha,
                    planner_model=planner_model,
                    planner_reasoning_effort=planner_reasoning,
                    planner_temperature=planner_temperature,
                    planner_max_steps=planner_max_steps,
                )
                (out_root / f"{item_id}.json").write_text(json.dumps(artifact.model_dump(), indent=2))

                (context_dir / f"{item_id}.json").write_text(
                    json.dumps(
                        {
                            "item_id": item_id,
                            "planner": {
                                "planner_prompt": str(planner_prompt_path),
                                "planner_prompt_sha256": planner_prompt_sha,
                                "planner_model": planner_model,
                                "planner_reasoning_effort": planner_reasoning,
                                "planner_temperature": planner_temperature,
                                "planner_max_steps": planner_max_steps,
                            },
                            "plan": plan_model.model_dump(),
                            "tables_planned": [t.table_anchor_id for t in plan_model.selected_tables],
                        },
                        indent=2,
                    )
                )
                return
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
            timeout=gateway_timeout or 600.0,
        ) as client:
            await asyncio.gather(*(_process(item_id, client) for item_id in items))

    asyncio.run(_runner())

    manifest_path = paths.manifest_path
    if manifest_path.exists():
        update_manifest(
            manifest_path,
            contract_pricing_v3_table_prompt=str(table_prompt_path),
            contract_pricing_v3_table_prompt_sha256=table_prompt_sha,
            contract_pricing_v3_planner_prompt=str(planner_prompt_path),
            contract_pricing_v3_planner_prompt_sha256=planner_prompt_sha,
        )

    if errors:
        joined = "; ".join(f"{item}: {msg}" for item, msg in errors)
        raise RuntimeError(f"Contract pricing v3 completed with errors: {joined}")
