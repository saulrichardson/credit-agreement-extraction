from __future__ import annotations

import asyncio
import json
import time
import traceback
from pathlib import Path
from typing import Any, Iterable

from .config import Paths, REQUIRED_MODEL, REQUIRED_REASONING, prompt_hash, update_manifest
from .contract_pricing import (
    _apply_rate_option_remap,
    _build_regimes_from_tables,
    _call_gateway,
    _extract_simple_fronting_fees,
    _merge_rate_options,
    _render_table_prompt,
    _try_extract_single_row_status_fee_table,
    _validate_pricing_output,
    _validate_table_extract,
)
from .contract_pricing_plan_schemas import ContractPricingPlanV2
from .contract_pricing_planner import plan_contract_pricing_v2
from .contract_schemas import (
    ContractPricingArtifact,
    ContractPricingModel,
    ContractPricingTableExtract,
    RateOption,
)
from .doc_ir import build_doc_ir, write_doc_ir
from .doc_tools import DocBrowser
from .pricing_index import build_pricing_index
from .utils import assert_exists

from .llm.gateway import DEFAULT_GATEWAY_URL, _ensure_gateway_client_async


def run_contract_pricing_v2(
    paths: Paths,
    item_ids: Iterable[str],
    *,
    planner_prompt_path: Path,
    table_prompt_path: Path,
    gateway_url: str | None = None,
    planner_model: str = "openai:gpt-5-mini",
    planner_reasoning: str = "high",
    planner_temperature: float = 0.0,
    planner_max_steps: int = 12,
    planner_attempts_per_step: int = 3,
    compiler_temperature: float = 0.0,
    gateway_timeout: float | None = None,
    concurrency: int = 2,
    attempts: int = 3,
    output_subdir: str = "contract_pricing_v2",
) -> None:
    """End-to-end (v2): doc IR -> agentic planning (no embeddings) -> per-table compile -> strict validation.

    v2 differs from v1 by inserting a planner that can navigate doc_ir via deterministic tools:
    - The planner selects tables and axis semantics (rows vs columns).
    - The table compiler receives table_plan hints but is still schema-validated and grounded.
    """

    assert_exists(planner_prompt_path, message=f"Planner prompt not found: {planner_prompt_path}")
    assert_exists(table_prompt_path, message=f"Table prompt not found: {table_prompt_path}")

    planner_prompt_sha = prompt_hash(planner_prompt_path)
    planner_prompt_template = planner_prompt_path.read_text()

    table_prompt_sha = prompt_hash(table_prompt_path)
    table_prompt_template = table_prompt_path.read_text()

    # Compiler model is enforced for consistency/cost. Planner is configurable.
    compiler_model = REQUIRED_MODEL
    compiler_reasoning = REQUIRED_REASONING

    out_root = paths.run_dir / "contract_pricing" / output_subdir
    out_root.mkdir(parents=True, exist_ok=True)

    doc_ir_dir = paths.run_dir / "doc_ir"
    doc_ir_dir.mkdir(parents=True, exist_ok=True)

    pricing_index_dir = paths.run_dir / "pricing_index"
    pricing_index_dir.mkdir(parents=True, exist_ok=True)

    plan_dir = paths.run_dir / "contract_pricing_plan" / output_subdir
    plan_dir.mkdir(parents=True, exist_ok=True)

    context_dir = paths.run_dir / "contract_pricing" / "_context" / output_subdir
    context_dir.mkdir(parents=True, exist_ok=True)

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

                # 2) Deterministic pricing index (ranked table candidates; audit only)
                pricing_index = build_pricing_index(doc_ir)
                (pricing_index_dir / f"{item_id}.json").write_text(json.dumps(pricing_index, indent=2))

                browser = DocBrowser(doc_ir)

                # 3) Agentic planning (Hebbia-like navigation, no embeddings)
                # Reuse the same gateway client so we don't spin up separate sessions per item.
                plan_model: ContractPricingPlanV2 = await plan_contract_pricing_v2(
                    browser=browser,
                    planner_prompt_path=planner_prompt_path,
                    gateway_url=gateway_url,
                    planner_model=planner_model,
                    planner_reasoning=planner_reasoning,
                    planner_temperature=planner_temperature,
                    gateway_timeout=gateway_timeout or 600.0,
                    max_steps=planner_max_steps,
                    attempts_per_step=planner_attempts_per_step,
                    client=client,
                )
                (plan_dir / f"{item_id}.json").write_text(json.dumps(plan_model.model_dump(), indent=2))

                # 4) Per-table extraction driven by plan
                per_table_dir = context_dir / item_id
                per_table_dir.mkdir(parents=True, exist_ok=True)

                table_contexts: list[dict[str, Any]] = []
                extracts_by_table: dict[str, ContractPricingTableExtract] = {}

                # Convenience for resolving regime hint text when planner provides only an anchor id.
                anchors_by_id = {str(a.get("anchor_id")): a for a in (doc_ir.get("anchors") or []) if isinstance(a, dict)}

                for table_plan in plan_model.selected_tables:
                    table_anchor_id = table_plan.table_anchor_id

                    table_context = browser.open_table(table_anchor_id, neighbor_window=6)
                    table_context["table_plan"] = table_plan.model_dump()

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
                    (per_table_dir / f"{table_anchor_id}.json").write_text(json.dumps(table_context, indent=2))

                    deterministic = _try_extract_single_row_status_fee_table(table_context=table_context)
                    if deterministic is not None:
                        _validate_table_extract(deterministic, table_context=table_context)
                        hinted = {
                            h["anchor_id"]
                            for h in table_context.get("adjustment_hints", [])
                            if isinstance(h, dict) and h.get("anchor_id")
                        }
                        if hinted:
                            referenced: set[str] = set()
                            for adj in deterministic.adjustments:
                                referenced.update(adj.source_refs or [])
                                referenced.update(adj.applies_when.source_refs or [])
                            missing = sorted(hinted - referenced)
                            if missing:
                                # Deterministic path couldn't encode required adjustment anchors; fall back to LLM.
                                deterministic = None
                        if deterministic is not None:
                            extracts_by_table[table_anchor_id] = deterministic
                            continue

                    rendered = _render_table_prompt(table_prompt_template, json.dumps(table_context, indent=2))
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
                            ex = ContractPricingTableExtract.model_validate_json(raw)
                            if ex.grid.table_anchor_id != table_anchor_id:
                                raise ValueError(
                                    f"grid.table_anchor_id {ex.grid.table_anchor_id!r} does not match expected {table_anchor_id!r}"
                                )
                            _validate_table_extract(ex, table_context=table_context)
                            hinted = {
                                h["anchor_id"]
                                for h in table_context.get("adjustment_hints", [])
                                if isinstance(h, dict) and h.get("anchor_id")
                            }
                            if hinted:
                                referenced: set[str] = set()
                                for adj in ex.adjustments:
                                    referenced.update(adj.source_refs or [])
                                    referenced.update(adj.applies_when.source_refs or [])
                                missing = sorted(hinted - referenced)
                                if missing:
                                    raise ValueError(
                                        f"Missing adjustments referencing anchors: {', '.join(missing)}"
                                    )
                            extracts_by_table[table_anchor_id] = ex
                            break
                        except Exception as exc:
                            last_err = str(exc)

                    if table_anchor_id not in extracts_by_table:
                        raise RuntimeError(
                            f"Table {table_anchor_id} failed after {used} attempts. Last error: {last_err}"
                        )

                # 5) Canonicalize rate option IDs across tables and apply remap.
                global_rate_options, remap = _merge_rate_options(list(extracts_by_table.values()))
                for tid, ex in list(extracts_by_table.items()):
                    extracts_by_table[tid] = _apply_rate_option_remap(ex, remap)

                # Attach deterministic flat items (e.g., fronting fee clauses) when obvious.
                flat_fronting = _extract_simple_fronting_fees({"anchors": list(doc_ir.get("anchors") or [])})
                if flat_fronting:
                    if not any(ro.rate_option_id == "fronting_fee" for ro in global_rate_options):
                        global_rate_options.append(
                            RateOption(
                                rate_option_id="fronting_fee",
                                label_raw="Fronting Fee",
                                kind="fee",
                                fee_basis="letters_of_credit",
                                source_refs=[fi.source_refs[0] for fi in flat_fronting if fi.source_refs],
                            )
                        )

                # 6) Assemble regimes using regime hints (planner override supported via table.regime_hint).
                regimes = _build_regimes_from_tables(table_contexts=table_contexts, extracts_by_table=extracts_by_table)
                if flat_fronting:
                    for reg in regimes:
                        if reg.regime_id == "default":
                            reg.flat_items.extend(flat_fronting)

                pricing_model = ContractPricingModel(
                    issuer=None,
                    agreement={},
                    rate_options=global_rate_options,
                    pricing_regimes=regimes,
                )

                # Final validation: ensure every planned table is represented and hint constraints are respected.
                context = {
                    "item_id": item_id,
                    "tables": [tc["table"] for tc in table_contexts],
                    "adjustment_hints": list(
                        {
                            str(h.get("anchor_id")): h
                            for tc in table_contexts
                            for h in (tc.get("adjustment_hints") or [])
                            if isinstance(h, dict) and h.get("anchor_id")
                        }.values()
                    ),
                }
                pricing_model = _validate_pricing_output(
                    raw_text=json.dumps(pricing_model.model_dump()),
                    context=context,
                )

                artifact = ContractPricingArtifact(
                    schema_version="contract_pricing_v2",
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
                    # v2 extras (optional in schema)
                    planner_prompt=str(planner_prompt_path),
                    planner_prompt_sha256=planner_prompt_sha,
                    planner_model=planner_model,
                    planner_reasoning_effort=planner_reasoning,
                    planner_temperature=planner_temperature,
                    planner_max_steps=planner_max_steps,
                )
                (out_root / f"{item_id}.json").write_text(json.dumps(artifact.model_dump(), indent=2))

                # Persist a lightweight per-item context summary for audit/debug.
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
            contract_pricing_v2_table_prompt=str(table_prompt_path),
            contract_pricing_v2_table_prompt_sha256=table_prompt_sha,
            contract_pricing_v2_planner_prompt=str(planner_prompt_path),
            contract_pricing_v2_planner_prompt_sha256=planner_prompt_sha,
        )

    if errors:
        joined = "; ".join(f"{item}: {msg}" for item, msg in errors)
        raise RuntimeError(f"Contract pricing v2 completed with errors: {joined}")
