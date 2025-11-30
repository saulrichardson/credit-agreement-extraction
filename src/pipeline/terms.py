from __future__ import annotations

import asyncio
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List

from .config import Paths
from .indexing import _ensure_gateway_client_async, DEFAULT_GATEWAY_URL, DEFAULT_MODEL
from .utils import assert_exists


def _load_metrics(item_id: str, qa_dir: Path) -> List[Dict[str, Any]]:
    """Load metrics array from a Q/A JSON file if present."""

    path = qa_dir / f"{item_id}.json"
    if not path.exists():
        return []
    doc = json.loads(path.read_text())
    metrics = doc.get("metrics") or []
    return metrics if isinstance(metrics, list) else []


def _load_snippets(paths: Paths, item_id: str) -> List[Dict[str, Any]]:
    path = assert_exists(
        paths.retrieval_dir / f"{item_id}_snippets.jsonl",
        message=f"Missing snippets for {item_id}: run retrieve first.",
    )
    out: List[Dict[str, Any]] = []
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def _select_snippets(snippets: List[Dict[str, Any]], limit: int | None = None) -> List[Dict[str, Any]]:
    """Return all snippets (or the first N when limit is set).

    We intentionally do *not* filter by metric name to mirror the Q/A input: the
    model sees the same snippet set the downstream Q/A saw.
    """

    if not snippets:
        return []
    if limit is None or limit <= 0:
        return snippets
    return snippets[:limit]


def _render_prompt(metric_name: str, snippet_recs: List[Dict[str, Any]]) -> str:
    formatted = []
    for rec in snippet_recs:
        aid = rec.get("anchor_id") or "UNK"
        txt = (rec.get("snippet") or "").strip()
        if len(txt) > 500:
            txt = txt[:500] + "..."
        formatted.append(f"[[{aid}]] {txt}")

    snippets_block = "\n\n".join(formatted)
    prompt = (
        "You are extracting the exact legal term wording from the credit agreement.\n"
        f"Metric (from prior Q/A): {metric_name}\n"
        "Below are all snippets supplied to the prior Q/A. Return JSON exactly:\n"
        '{"term": "<exact term as written>", "anchor_refs": ["<anchor_id>", ...]}\n'
        "Rules: copy the term text from the snippets; do not invent or rename; if uncertain, repeat the closest term seen; use anchor_refs from the provided snippets.\n"
        "Snippets:\n\n"
        f"{snippets_block}"
    )
    return prompt


async def _call_gateway_raw(client: Any, prompt: str, model: str, temperature: float, reasoning: str | None) -> str:
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


def run_terms_lookup(
    paths: Paths,
    item_ids: Iterable[str],
    *,
    qa_dir: Path,
    model: str | None = None,
    gateway_url: str | None = None,
    temperature: float = 0.0,
    reasoning: str | None = None,
    gateway_timeout: float | None = None,
    concurrency: int = 3,
    output_subdir: str = "terms_from_qa",
) -> None:
    """Resolve exact agreement term names for metrics by asking the gateway over their snippets."""

    out_dir = paths.structured_dir / output_subdir
    out_dir.mkdir(parents=True, exist_ok=True)

    items = list(item_ids)
    GatewayAgentClient = _ensure_gateway_client_async()
    sem = asyncio.Semaphore(max(1, concurrency))

    async def _process(item_id: str, client: Any) -> None:
        async with sem:
            metrics = _load_metrics(item_id, qa_dir)
            if not metrics:
                return
            snippets = _load_snippets(paths, item_id)
            for metric in metrics:
                name = metric.get("name") or metric.get("metric_id") or metric.get("description") or ""
                if not name:
                    continue
                relevant = _select_snippets(snippets, limit=None)
                prompt = _render_prompt(name, relevant)
                raw_text = await _call_gateway_raw(
                    client=client,
                    prompt=prompt,
                    model=model or DEFAULT_MODEL,
                    temperature=temperature,
                    reasoning=reasoning,
                )
                safe_name = re.sub(r"[^A-Za-z0-9_-]+", "_", name).strip("_") or "metric"
                out_path = out_dir / f"{item_id}__{safe_name}.txt"
                out_path.write_text(raw_text)

    async def _runner() -> None:
        async with GatewayAgentClient(
            base_url=gateway_url or DEFAULT_GATEWAY_URL,
            timeout=gateway_timeout or 30.0,
        ) as client:
            await asyncio.gather(*(_process(item_id, client) for item_id in items))

    asyncio.run(_runner())
