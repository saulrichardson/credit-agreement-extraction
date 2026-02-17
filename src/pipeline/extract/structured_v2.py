from __future__ import annotations

import asyncio
import json
import traceback
from pathlib import Path
from typing import Any, Iterable, List, Tuple

from pipeline.contracts.common import StructuredValidationContext
from pipeline.contracts.registry import StructuredContractName, get_structured_contract
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


def run_structured_v2(
    paths: Paths,
    item_ids: Iterable[str],
    prompt_path: Path,
    *,
    contract: StructuredContractName,
    model: str | None = None,
    gateway_url: str | None = None,
    temperature: float = 0.0,
    reasoning: str | None = None,
    gateway_timeout: float | None = None,
    concurrency: int = 3,
    attempts: int = 3,
    output_subdir: str | None = None,
    categories: Iterable[str] | None = None,
    skip_existing: bool = False,
    allow_empty_after_filter: bool = False,
) -> None:
    """Run structured LLM extraction over v2 retrieval snippets (strict JSON output)."""

    assert_exists(prompt_path, message=f"Structured prompt not found: {prompt_path}")
    contract_spec = get_structured_contract(contract)

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
    errors: List[Tuple[str, str]] = []

    async def _process(item_id: str, client: Any, sem: asyncio.Semaphore) -> None:
        async with sem:
            try:
                artifact_path = out_dir / f"{item_id}.json"
                if skip_existing and artifact_path.exists():
                    return

                snippets = _load_snippets_v2(paths, item_id)
                if categories:
                    snippets = _filter_snippets(snippets, categories)
                if not snippets:
                    if allow_empty_after_filter:
                        empty = contract_spec.empty_payload(
                            "no snippets after category filter"
                            if categories
                            else "no snippets after loading"
                        )
                        artifact_path.write_text(json.dumps(empty, indent=2, sort_keys=True) + "\n", encoding="utf-8")
                        return
                    if categories:
                        raise RuntimeError(
                            "No snippets after category filter. "
                            f"Requested categories: {', '.join(categories or [])}"
                        )
                    raise RuntimeError(f"No snippets parsed for {item_id} after loading.")

                allowed_anchor_ids: set[str] = set()
                anchor_tags: dict[str, set[str]] = {}
                snippet_text_by_anchor: dict[str, str] = {}
                for rec in snippets:
                    aid = rec.get("anchor_id")
                    if not isinstance(aid, str) or not aid.strip():
                        continue
                    anchor_id = aid.strip()
                    allowed_anchor_ids.add(anchor_id)
                    snippet_text_by_anchor[anchor_id] = str(rec.get("snippet") or "")
                    tags: set[str] = set()
                    for raw in (rec.get("categories") or []):
                        if isinstance(raw, str) and raw.strip():
                            tags.add(raw.strip().lower())
                    for raw in (rec.get("buckets") or []):
                        if isinstance(raw, str) and raw.strip():
                            tags.add(raw.strip().lower())
                    label = rec.get("label") or ""
                    for part in str(label).split(","):
                        if part.strip():
                            tags.add(part.strip().lower())
                    anchor_tags[anchor_id] = tags

                ctx = StructuredValidationContext(
                    item_id=item_id,
                    allowed_anchor_ids=allowed_anchor_ids,
                    anchor_tags=anchor_tags,
                    snippet_text_by_anchor=snippet_text_by_anchor,
                )
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
                        retry_prompt=contract_spec.retry_prompt,
                        allowed_root_types=contract_spec.allowed_root_types,
                        validate=lambda payload: contract_spec.validate(payload, ctx),
                    )
                except StrictJsonFailure as exc:
                    raw_sidecar = out_dir / f"{item_id}.raw.txt"
                    raw_sidecar.write_text(exc.last_raw_text or "", encoding="utf-8")
                    raise RuntimeError(
                        f"structured-v2 failed strict JSON after {attempts} attempt(s). Last error: {exc.last_error}"
                    ) from exc

                artifact_path.write_text(json.dumps(parsed, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            except Exception as exc:  # pragma: no cover - defensive
                errors.append((item_id, str(exc)))
                err_path = out_dir / f"{item_id}.error.txt"
                err_path.write_text(f"{exc}\n\n{traceback.format_exc()}")

    async def _runner() -> None:
        sem = asyncio.Semaphore(max(1, concurrency))
        async with GatewayAgentClient(
            base_url=gateway_url or DEFAULT_GATEWAY_URL,
            timeout=gateway_timeout or 600.0,
        ) as client:
            await asyncio.gather(*(_process(item_id, client, sem) for item_id in items))

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
