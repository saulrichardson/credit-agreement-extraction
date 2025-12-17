from __future__ import annotations

import asyncio
import json
import traceback
from pathlib import Path
from typing import Any, Iterable, List, Literal, Tuple

from .config import Paths, prompt_hash, update_manifest, REQUIRED_MODEL, REQUIRED_REASONING
from .utils import assert_exists

# Reuse the gateway helpers/constants from indexing to avoid duplicating config.
from .indexing import (  # type: ignore
    _ensure_gateway_client_async,
    DEFAULT_GATEWAY_URL,
)

StructuredInputMode = Literal["snippets", "full_document"]


def _load_snippets(paths: Paths, item_id: str) -> List[dict]:
    path = assert_exists(
        paths.retrieval_dir / f"{item_id}_snippets.jsonl",
        message=f"Missing snippets for {item_id}: run retrieve first.",
    )
    snippets: List[dict] = []
    with path.open() as fh:
        for line in fh:
            if not line.strip():
                continue
            try:
                snippets.append(json.loads(line))
            except json.JSONDecodeError:
                # Skip malformed lines but keep going.
                continue
    if not snippets:
        raise RuntimeError(f"No snippets parsed for {item_id} (file: {path})")
    return snippets


def _load_full_document(paths: Paths, item_id: str) -> str:
    path = assert_exists(
        paths.normalized_dir / item_id / "canonical_annotated.txt",
        message=f"Missing canonical_annotated.txt for {item_id}: run normalize first.",
    )
    text = path.read_text()
    if not text.strip():
        raise RuntimeError(f"Empty canonical_annotated.txt for {item_id} (file: {path})")
    return text


def _render_snippet_block(snippets: List[dict]) -> str:
    blocks: List[str] = []
    for rec in snippets:
        aid = rec.get("anchor_id") or "UNK"
        label = rec.get("label") or rec.get("type")
        header = f"[[{aid}]]"
        if label:
            header = f"{header} ({label})"
        snippet_text = (rec.get("snippet") or "").strip()
        blocks.append(f"{header}\n{snippet_text}")
    return "\n\n".join(blocks)


def _render_prompt(template: str, snippets_block: str) -> str:
    template = template.strip()
    if "{document}" in template:
        return template.replace("{document}", snippets_block)
    if "{snippets}" in template:
        return template.replace("{snippets}", snippets_block)
    if "=== INPUT" in template:
        return f"{template}\n{snippets_block}"
    return f"{template}\n\n=== INPUT ===\n{snippets_block}"


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


def run_structured(
    paths: Paths,
    item_ids: Iterable[str],
    prompt_path: Path,
    *,
    input_mode: StructuredInputMode = "snippets",
    model: str | None = None,
    gateway_url: str | None = None,
    temperature: float = 0.0,
    reasoning: str | None = None,
    gateway_timeout: float | None = None,
    concurrency: int = 3,
    output_subdir: str | None = None,
) -> None:
    """Run structured LLM extraction via agent-gateway.

    Input modes:
    - snippets: reads retrieval output (`*_snippets.jsonl`) and renders an anchor snippet block.
    - full_document: reads normalized full document (`normalized/<item_id>/canonical_annotated.txt`).

    - Renders the user prompt with the input appended or substituted.
    - Calls the gateway with bounded async concurrency and persists both raw and parsed outputs.
    """

    assert_exists(prompt_path, message=f"Structured prompt not found: {prompt_path}")

    # Hard enforcement of model + reasoning defaults.
    model = REQUIRED_MODEL
    reasoning = REQUIRED_REASONING

    prompt_digest = prompt_hash(prompt_path)
    prompt_template = prompt_path.read_text()

    # Prepare output directories
    root_out = paths.structured_dir
    out_dir = root_out / (output_subdir or prompt_path.stem)
    out_dir.mkdir(parents=True, exist_ok=True)

    items = list(item_ids)

    GatewayAgentClient = _ensure_gateway_client_async()
    sem = asyncio.Semaphore(max(1, concurrency))
    errors: List[Tuple[str, str]] = []

    async def _process(item_id: str, client: Any) -> None:
        async with sem:
            try:
                if input_mode == "snippets":
                    snippets = _load_snippets(paths, item_id)
                    input_block = _render_snippet_block(snippets)
                elif input_mode == "full_document":
                    input_block = _load_full_document(paths, item_id)
                else:
                    raise ValueError(f"Unknown structured input_mode: {input_mode!r}")

                rendered_prompt = _render_prompt(prompt_template, input_block)
                raw_text = await _call_gateway(
                    client=client,
                    prompt=rendered_prompt,
                    model=model,
                    temperature=temperature,
                    reasoning=reasoning,
                )

                # Persist raw output only.
                raw_path = out_dir / f"{item_id}.txt"
                raw_path.write_text(raw_text)
            except Exception as exc:  # pragma: no cover - defensive
                errors.append((item_id, str(exc)))
                err_path = out_dir / f"{item_id}.error.txt"
                err_path.write_text(f"{exc}\n\n{traceback.format_exc()}")

    async def _runner() -> None:
        async with GatewayAgentClient(
            base_url=gateway_url or DEFAULT_GATEWAY_URL,
            timeout=gateway_timeout or 30.0,
        ) as client:
            await asyncio.gather(*(_process(item_id, client) for item_id in items))

    asyncio.run(_runner())

    manifest_path = paths.manifest_path
    if manifest_path.exists():
        update_manifest(
            manifest_path,
            structured_prompt=str(prompt_path),
            structured_prompt_sha256=prompt_digest,
        )

    if errors:
        joined = "; ".join(f"{item}: {msg}" for item, msg in errors)
        raise RuntimeError(f"Structured extraction completed with errors: {joined}")
