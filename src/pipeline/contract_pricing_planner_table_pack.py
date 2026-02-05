from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

from .config import REQUIRED_MODEL, REQUIRED_REASONING
from .contract_pricing_plan_schemas import ContractPricingPlanV2
from .utils import assert_exists

from .llm.gateway import DEFAULT_GATEWAY_URL, _ensure_gateway_client_async


def _render_table_pack_prompt(template: str, tables_pack_json: str) -> str:
    template = template.strip()
    if "{tables_pack_json}" not in template:
        raise ValueError(
            "Planner prompt template must include a `{tables_pack_json}` placeholder so the table pack can be injected."
        )
    return template.replace("{tables_pack_json}", tables_pack_json)


async def _call_gateway_messages(
    *,
    client: Any,
    messages: list[dict[str, Any]],
    model: str,
    temperature: float,
    reasoning: str | None,
) -> str:
    reasoning_payload = {"effort": reasoning} if reasoning else None
    result = await client.complete_response(
        model=model,
        input_messages=messages,
        reasoning=reasoning_payload,
        temperature=temperature,
        max_output_tokens=None,
        metadata=None,
    )
    if isinstance(result, dict):
        text = str(result.get("text") or "")
        meta = result.get("meta") or {}
        response = meta.get("response") if isinstance(meta, dict) else None
        if isinstance(response, dict) and response.get("error"):
            raise RuntimeError(f"Gateway response.error: {response.get('error')}")
        if not text.strip() and isinstance(response, dict):
            out_text = response.get("output_text")
            if isinstance(out_text, list):
                text = "".join(str(x) for x in out_text)
        if not text.strip():
            if isinstance(response, dict):
                raise RuntimeError(
                    "Gateway returned empty text. "
                    f"response.status={response.get('status')!r} model={response.get('model')!r}"
                )
            raise RuntimeError("Gateway returned empty text (no response.completed payload captured).")
        return text
    return str(result)


def _retry_message(*, attempt: int, error: str) -> str:
    if attempt >= 2:
        return (
            "STRICT MODE: your previous response was not valid JSON or failed schema validation.\n"
            "- Output MUST be a single JSON object.\n"
            "- Do NOT include markdown or prose.\n"
            f"- Error: {error}\n"
        )
    return (
        "Your previous response was not valid JSON or failed schema validation.\n"
        "- Output MUST be a single JSON object only.\n"
        f"- Error: {error}\n"
    )


async def plan_contract_pricing_table_pack(
    *,
    doc_ir: dict[str, Any],
    planner_prompt_path: Path,
    gateway_url: str | None,
    planner_model: str,
    planner_reasoning: str | None,
    planner_temperature: float,
    gateway_timeout: float,
    attempts: int = 3,
    client: Any | None = None,
) -> ContractPricingPlanV2:
    """Plan pricing tables from a full "table pack" (no tool loop, no heuristics).

    This approach gives the model ALL parsed tables up-front, which tends to improve recall on
    pricing tables compared to tool-driven sampling, at the cost of a larger prompt.
    """

    assert_exists(planner_prompt_path, message=f"Planner prompt not found: {planner_prompt_path}")

    if not planner_model:
        planner_model = REQUIRED_MODEL
    planner_reasoning = planner_reasoning or REQUIRED_REASONING

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

    tables_pack = {
        "item_id": str(doc_ir.get("item_id") or ""),
        "n_tables": len(tables),
        "tables": sorted(tables, key=lambda x: int(x.get("order") or 0)),
    }

    prompt_template = planner_prompt_path.read_text()
    rendered = _render_table_pack_prompt(prompt_template, json.dumps(tables_pack, indent=2))

    messages: list[dict[str, Any]] = [
        {
            "role": "system",
            "content": (
                "You are a planning agent. Output JSON only. "
                "Return a ContractPricingPlanV2 JSON object."
            ),
        },
        {"role": "user", "content": rendered},
    ]

    async def _run_with_client(active_client: Any) -> ContractPricingPlanV2:
        last_err: str | None = None
        for attempt in range(1, max(1, int(attempts)) + 1):
            prompt_messages = list(messages)
            if last_err:
                prompt_messages.append({"role": "user", "content": _retry_message(attempt=attempt, error=last_err)})
            raw = await _call_gateway_messages(
                client=active_client,
                messages=prompt_messages,
                model=planner_model,
                temperature=planner_temperature,
                reasoning=planner_reasoning,
            )
            try:
                parsed = json.loads(raw)
            except json.JSONDecodeError as exc:
                last_err = str(exc)
                await asyncio.sleep(min(0.5, 0.1 * attempt))
                continue

            if not isinstance(parsed, dict):
                last_err = f"Expected a JSON object, got {type(parsed).__name__}"
                await asyncio.sleep(min(0.5, 0.1 * attempt))
                continue

            # Accept either the plan directly or wrapped in {"action":"final","plan":...}.
            if str(parsed.get("schema_version") or "").strip() == "contract_pricing_plan_v2":
                plan = parsed
            else:
                action = str(parsed.get("action") or "").strip().lower()
                if action == "final" and isinstance(parsed.get("plan"), dict):
                    plan = parsed["plan"]
                else:
                    last_err = "Response did not match expected plan schema or final wrapper"
                    await asyncio.sleep(min(0.5, 0.1 * attempt))
                    continue

            try:
                plan_model = ContractPricingPlanV2.model_validate(plan)
            except Exception as exc:
                last_err = str(exc)
                await asyncio.sleep(min(0.5, 0.1 * attempt))
                continue

            expected_item_id = str(doc_ir.get("item_id") or "")
            if plan_model.item_id != expected_item_id:
                last_err = f"item_id mismatch: plan has {plan_model.item_id!r}, expected {expected_item_id!r}"
                await asyncio.sleep(min(0.5, 0.1 * attempt))
                continue

            return plan_model

        raise RuntimeError(f"Table-pack planner failed after {attempts} attempts. Last error: {last_err}")

    if client is not None:
        return await _run_with_client(client)

    GatewayAgentClient = _ensure_gateway_client_async()
    async with GatewayAgentClient(
        base_url=gateway_url or DEFAULT_GATEWAY_URL,
        timeout=gateway_timeout,
    ) as active_client:
        return await _run_with_client(active_client)
