from __future__ import annotations

import asyncio
import json
import traceback
from pathlib import Path
from typing import Any, Iterable, List, Tuple

from pipeline.core.config import Paths, prompt_hash, update_manifest, REQUIRED_MODEL, REQUIRED_REASONING
from pipeline.llm.strict_json import StrictJsonFailure, call_strict_json
from pipeline.llm.gateway import DEFAULT_GATEWAY_URL, _ensure_gateway_client_async
from pipeline.utils import assert_exists


def _load_snippets_v2(paths: Paths, item_id: str) -> List[dict]:
    path = assert_exists(
        paths.run_dir / "retrieval_v2" / f"{item_id}_snippets.jsonl",
        message=f"Missing v2 snippets for {item_id}: run retrieve-v2 first.",
    )
    snippets: List[dict] = []
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


def _filter_snippets(snippets: List[dict], categories: Iterable[str]) -> List[dict]:
    wanted = {c.strip().lower() for c in categories if c.strip()}
    if not wanted:
        return snippets
    filtered: List[dict] = []
    for rec in snippets:
        rec_categories = [c.lower() for c in (rec.get("categories") or []) if isinstance(c, str)]
        rec_buckets = [c.lower() for c in (rec.get("buckets") or []) if isinstance(c, str)]
        label = rec.get("label") or ""
        rec_label_parts = [c.strip().lower() for c in str(label).split(",") if c.strip()]
        if any(c in wanted for c in rec_categories + rec_buckets + rec_label_parts):
            filtered.append(rec)
    return filtered


def _drop_definitions_only(snippets: List[dict]) -> List[dict]:
    """Remove snippets that are *only* in the 'definitions' bucket.

    Indexing v2 can optionally expand a large definitions section anchor range. Those anchors are
    useful for the definitions pass, but they should not be fed wholesale into structured pricing
    extraction prompts (too large / not targeted).

    We keep any snippet that is tagged with other buckets (e.g., pricing/base_rate) even if it
    also lies inside the definitions range.
    """

    kept: List[dict] = []
    for rec in snippets:
        cats = [c.lower() for c in (rec.get("categories") or []) if isinstance(c, str) and c.strip()]
        buckets = [c.lower() for c in (rec.get("buckets") or []) if isinstance(c, str) and c.strip()]
        tags = cats or buckets
        if tags and all(t == "definitions" for t in tags):
            continue
        kept.append(rec)
    return kept


def _render_snippet_block(snippets: List[dict]) -> str:
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


def _render_prompt(template: str, snippets_block: str) -> str:
    template = template.strip()
    if "{document}" in template:
        return template.replace("{document}", snippets_block)
    if "{snippets}" in template:
        return template.replace("{snippets}", snippets_block)
    if "=== INPUT" in template:
        return f"{template}\n{snippets_block}"
    return f"{template}\n\n=== INPUT ===\n{snippets_block}"


def _retry_prompt(base_prompt: str, attempt: int, error: str, previous_output: str) -> str:
    _ = attempt
    _ = previous_output
    rules = (
        "Your previous response was invalid.\n"
        "- Output MUST be valid JSON.\n"
        "- Output MUST contain JSON ONLY (no markdown, no code fences, no prose).\n"
        "- If you are unsure, return the best-effort JSON in the expected shape rather than commentary.\n"
    )
    return f"{base_prompt}\n\n=== RETRY REQUIRED ===\nError: {error}\n\n{rules}"


def validate_structured_v2_payload(payload: object) -> dict | list:
    """Validate structured-v2 payload shape.

    Domain-level semantics are enforced by downstream stages; this validator isolates
    shape checks from prompt execution flow.
    """

    if not isinstance(payload, (dict, list)):
        raise ValueError(f"structured-v2 payload must be object or array; got {type(payload).__name__}")
    return payload


def run_structured_v2(
    paths: Paths,
    item_ids: Iterable[str],
    prompt_path: Path,
    *,
    model: str | None = None,
    gateway_url: str | None = None,
    temperature: float = 0.0,
    reasoning: str | None = None,
    gateway_timeout: float | None = None,
    concurrency: int = 3,
    attempts: int = 3,
    output_subdir: str | None = None,
    categories: Iterable[str] | None = None,
) -> None:
    """Run structured LLM extraction over v2 retrieval snippets (strict JSON output)."""

    assert_exists(prompt_path, message=f"Structured prompt not found: {prompt_path}")

    # Hard enforcement of model + reasoning defaults.
    model = REQUIRED_MODEL
    reasoning = REQUIRED_REASONING

    prompt_digest = prompt_hash(prompt_path)
    prompt_template = prompt_path.read_text()

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
                snippets = _load_snippets_v2(paths, item_id)
                if categories:
                    snippets = _filter_snippets(snippets, categories)
                if not snippets:
                    if categories:
                        raise RuntimeError(
                            "No snippets after category filter. "
                            f"Requested categories: {', '.join(categories or [])}"
                        )
                    raise RuntimeError(f"No snippets parsed for {item_id} after loading.")
                input_block = _render_snippet_block(snippets)

                rendered_prompt = _render_prompt(prompt_template, input_block)
                try:
                    parsed, _raw, _attempts_used = await call_strict_json(
                        client=client,
                        prompt=rendered_prompt,
                        model=model,
                        temperature=temperature,
                        reasoning=reasoning,
                        attempts=attempts,
                        retry_prompt=_retry_prompt,
                        allowed_root_types=(dict, list),
                        validate=validate_structured_v2_payload,
                    )
                except StrictJsonFailure as exc:
                    raw_sidecar = out_dir / f"{item_id}.raw.txt"
                    raw_sidecar.write_text(exc.last_raw_text or "", encoding="utf-8")
                    raise RuntimeError(
                        f"structured-v2 failed strict JSON after {attempts} attempt(s). Last error: {exc.last_error}"
                    ) from exc

                artifact_path = out_dir / f"{item_id}.json"
                artifact_path.write_text(json.dumps(parsed, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            except Exception as exc:  # pragma: no cover - defensive
                errors.append((item_id, str(exc)))
                err_path = out_dir / f"{item_id}.error.txt"
                err_path.write_text(f"{exc}\n\n{traceback.format_exc()}")

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
            structured_v2_prompt=str(prompt_path),
            structured_v2_prompt_sha256=prompt_digest,
        )

    if errors:
        joined = "; ".join(f"{item}: {msg}" for item, msg in errors)
        raise RuntimeError(f"Structured extraction (v2) completed with errors: {joined}")
