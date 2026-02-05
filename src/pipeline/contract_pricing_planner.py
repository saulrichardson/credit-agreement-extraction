from __future__ import annotations

import asyncio
import json
import traceback
from pathlib import Path
from typing import Any

from .config import REQUIRED_MODEL, REQUIRED_REASONING
from .contract_pricing_plan_schemas import ContractPricingPlanV2
from .doc_tools import DocBrowser
from .utils import assert_exists

# Reuse gateway helper/constants from indexing to avoid duplicating config.
from .indexing import (  # type: ignore
    _ensure_gateway_client_async,
    DEFAULT_GATEWAY_URL,
)


def _render_planner_prompt(template: str, doc_catalog_json: str) -> str:
    template = template.strip()
    if "{doc_catalog_json}" not in template:
        raise ValueError(
            "Planner prompt template must include a `{doc_catalog_json}` placeholder so the document catalog can be injected."
        )
    return template.replace("{doc_catalog_json}", doc_catalog_json)


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
            # Fallback: in rare cases we fail to collect deltas; use completed payload.
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
            "STRICT MODE: your previous response was not valid JSON.\n"
            "- Output MUST be a single JSON object.\n"
            "- Do NOT include markdown or prose.\n"
            f"- Error: {error}\n"
        )
    return (
        "Your previous response was not valid JSON.\n"
        "- Output MUST be a single JSON object only.\n"
        f"- Error: {error}\n"
    )


async def plan_contract_pricing_v2(
    *,
    browser: DocBrowser,
    planner_prompt_path: Path,
    gateway_url: str | None,
    planner_model: str,
    planner_reasoning: str | None,
    planner_temperature: float,
    gateway_timeout: float,
    catalog_max_tables: int | None = 60,
    max_steps: int,
    attempts_per_step: int,
    client: Any | None = None,
) -> ContractPricingPlanV2:
    """Agentic planner loop (no embeddings).

    The LLM "drives" via explicit JSON tool calls. The host executes deterministic tools over doc_ir
    and feeds results back. Final output is a validated ContractPricingPlanV2.
    """

    assert_exists(planner_prompt_path, message=f"Planner prompt not found: {planner_prompt_path}")

    # Planner is intentionally configurable (user wants a bigger model), but keep defaults sane.
    if not planner_model:
        planner_model = REQUIRED_MODEL
    planner_reasoning = planner_reasoning or REQUIRED_REASONING

    prompt_template = planner_prompt_path.read_text()
    catalog = browser.catalog(max_tables=catalog_max_tables)
    rendered = _render_planner_prompt(prompt_template, json.dumps(catalog, indent=2))

    messages: list[dict[str, Any]] = [
        {
            "role": "system",
            "content": (
                "You are a tool-using planning agent. "
                "You must follow the Tool protocol and output JSON only."
            ),
        },
        {"role": "user", "content": rendered},
    ]

    async def _run_with_client(active_client: Any) -> ContractPricingPlanV2:
        for step in range(max(1, int(max_steps))):
            raw: str | None = None
            parsed: dict[str, Any] | None = None
            last_err: str | None = None

            for attempt in range(1, max(1, int(attempts_per_step)) + 1):
                try:
                    raw = await _call_gateway_messages(
                        client=active_client,
                        messages=messages,
                        model=planner_model,
                        temperature=planner_temperature,
                        reasoning=planner_reasoning,
                    )
                except Exception as exc:
                    last_err = f"Gateway call failed: {type(exc).__name__}: {exc}"
                    # Retry without mutating the conversation; no model output was produced.
                    await asyncio.sleep(min(1.0, 0.1 * attempt))
                    continue
                try:
                    parsed_obj = json.loads(raw)
                except json.JSONDecodeError as exc:
                    last_err = str(exc)
                    messages.append({"role": "user", "content": _retry_message(attempt=attempt, error=last_err)})
                    continue

                if not isinstance(parsed_obj, dict):
                    last_err = f"Expected a JSON object, got {type(parsed_obj).__name__}"
                    messages.append({"role": "user", "content": _retry_message(attempt=attempt, error=last_err)})
                    continue

                parsed = parsed_obj
                break

            if parsed is None:
                raise RuntimeError(f"Planner step {step+1} failed: {last_err or 'unknown error'}")

            action = str(parsed.get("action") or "").strip().lower()
            # Back-compat / robustness: some models may skip the outer wrapper and output the plan directly.
            if not action and str(parsed.get("schema_version") or "").strip() == "contract_pricing_plan_v2":
                action = "final"
                parsed = {"action": "final", "plan": parsed}
            if not action and parsed.get("tool") and parsed.get("args"):
                action = "tool"

            if action == "tool":
                tool = str(parsed.get("tool") or "").strip()
                args = parsed.get("args") or {}
                if not isinstance(args, dict):
                    raise ValueError("Planner tool call args must be a JSON object")

                if tool == "catalog":
                    max_tables = args.get("max_tables")
                    result = browser.catalog(max_tables=max_tables if max_tables is None else int(max_tables))
                elif tool == "search":
                    result = browser.search(
                        query=str(args.get("query") or ""),
                        scope=str(args.get("scope") or "all").lower(),  # type: ignore[arg-type]
                        limit=int(args.get("limit") or 20),
                        case_sensitive=bool(args.get("case_sensitive") or False),
                    )
                elif tool == "open_table":
                    result = browser.open_table(
                        str(args.get("table_anchor_id") or ""),
                        neighbor_window=int(args.get("neighbor_window") or 6),
                    )
                elif tool == "open_anchor":
                    result = browser.open_anchor(
                        str(args.get("anchor_id") or ""),
                        neighbor_window=int(args.get("neighbor_window") or 0),
                    )
                elif tool == "open_anchor_range":
                    open_anchor_range = getattr(browser, "open_anchor_range", None)
                    if not callable(open_anchor_range):
                        raise ValueError("Planner tool open_anchor_range is not supported by this browser")
                    result = open_anchor_range(
                        start_order=int(args.get("start_order") or 0),
                        limit=int(args.get("limit") or 50),
                    )
                else:
                    raise ValueError(f"Unknown planner tool: {tool!r}")

                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "TOOL_RESULT_JSON:\n"
                            + json.dumps({"tool": tool, "result": result}, indent=2)
                            + "\n\nContinue by outputting the next tool call JSON, or the final plan JSON."
                        ),
                    }
                )
                continue

            if action == "final":
                plan = parsed.get("plan")
                if not isinstance(plan, dict):
                    raise ValueError("Planner final output must include plan as a JSON object")
                try:
                    plan_model = ContractPricingPlanV2.model_validate(plan)
                except Exception as exc:
                    # Keep the agentic loop alive: ask the planner to fix schema issues.
                    messages.append(
                        {
                            "role": "user",
                            "content": (
                                "FINAL_PLAN_VALIDATION_ERRORS:\n"
                                f"{exc}\n\n"
                                "Return a corrected {\"action\":\"final\",\"plan\":...} JSON object only."
                            ),
                        }
                    )
                    continue

                if plan_model.item_id != browser.item_id:
                    messages.append(
                        {
                            "role": "user",
                            "content": (
                                "FINAL_PLAN_VALIDATION_ERRORS:\n"
                                f"item_id mismatch: plan has {plan_model.item_id!r}, expected {browser.item_id!r}\n\n"
                                "Return a corrected {\"action\":\"final\",\"plan\":...} JSON object only."
                            ),
                        }
                    )
                    continue

                return plan_model

            raise ValueError(f"Planner returned unknown action {action!r}; expected 'tool' or 'final'")

        raise RuntimeError(f"Planner exceeded max_steps={max_steps}")

    if client is not None:
        return await _run_with_client(client)

    GatewayAgentClient = _ensure_gateway_client_async()
    async with GatewayAgentClient(
        base_url=gateway_url or DEFAULT_GATEWAY_URL,
        timeout=gateway_timeout,
    ) as active_client:
        return await _run_with_client(active_client)


def run_contract_pricing_planner_v2(
    paths,
    item_ids,
    *,
    doc_ir_by_item: dict[str, dict[str, Any]],
    planner_prompt_path: Path,
    gateway_url: str | None = None,
    planner_model: str | None = None,
    planner_reasoning: str | None = None,
    planner_temperature: float = 0.0,
    gateway_timeout: float = 600.0,
    max_steps: int = 12,
    attempts_per_step: int = 3,
) -> dict[str, ContractPricingPlanV2]:
    """Sync wrapper for planning across multiple items (runs per-item loops sequentially)."""

    results: dict[str, ContractPricingPlanV2] = {}
    for item_id in item_ids:
        doc_ir = doc_ir_by_item[item_id]
        browser = DocBrowser(doc_ir)
        plan = asyncio.run(
            plan_contract_pricing_v2(
                browser=browser,
                planner_prompt_path=planner_prompt_path,
                gateway_url=gateway_url,
                planner_model=planner_model or REQUIRED_MODEL,
                planner_reasoning=planner_reasoning or REQUIRED_REASONING,
                planner_temperature=planner_temperature,
                gateway_timeout=gateway_timeout,
                max_steps=max_steps,
                attempts_per_step=attempts_per_step,
            )
        )
        results[item_id] = plan
    return results


def format_planner_error(exc: Exception) -> str:
    return f"{exc}\n\n{traceback.format_exc()}"
