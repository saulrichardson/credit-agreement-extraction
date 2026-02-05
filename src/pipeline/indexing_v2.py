from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from typing import Iterable

from .anchors import load_anchor_catalog
from .config import Paths, REQUIRED_MODEL, REQUIRED_REASONING, prompt_hash, update_manifest
from .indexing import DEFAULT_GATEWAY_URL, _ensure_gateway_client_async
from .llm.strict_json import StrictJsonFailure, call_strict_json
from .schemas_v2 import IndexingSelectionV2, IndexingSelectionV2Artifact
from .utils import assert_exists


def _canonical_annotated_text(paths: Paths, item_id: str) -> str:
    annotated = paths.normalized_dir / item_id / "canonical_annotated.txt"
    if not annotated.exists():
        raise FileNotFoundError(f"Annotated text missing for {item_id}: expected {annotated}")
    text = annotated.read_text().strip()
    if not text:
        raise RuntimeError(f"Empty canonical_annotated.txt for {item_id}")
    return text


def _render_prompt(template: str, doc_block: str) -> str:
    template = template.strip()
    if "{document}" not in template:
        raise ValueError(
            "Indexing v2 prompt template must include a `{document}` placeholder so the full "
            "annotated document can be injected."
        )
    return template.replace("{document}", doc_block)


def _retry_prompt(base_prompt: str, attempt: int, error: str, previous_output: str) -> str:
    _ = previous_output
    if attempt == 2:
        rules = (
            "Your previous response was not valid. Fix it.\n"
            "- Output MUST be a single JSON object (no markdown, no code fences, no prose).\n"
            "- JSON MUST match this exact schema:\n"
            "  {\"metadata_anchors\": [...], \"agreement_date_anchors\": [...], \"fundamental_anchors\": [...], "
            "\"key_date_definitions_anchors\": [...], "
            "\"pricing_anchors\": [...], "
            "\"base_rate_anchors\": [...], \"spread_anchors\": [...], \"fee_anchors\": [...], "
            "\"financial_covenant_anchors\": [...], "
            "\"definitions_anchor_range\": {\"start_anchor\":\"A0001\",\"end_anchor\":\"A0002\"} | null}\n"
            "- Each entry MUST be a string anchor ID like \"A0001\".\n"
        )
    else:
        rules = (
            "STRICT MODE (final attempt):\n"
            "- Return ONLY JSON with exactly these keys:\n"
            "  metadata_anchors, agreement_date_anchors, fundamental_anchors, key_date_definitions_anchors, pricing_anchors, "
            "base_rate_anchors, spread_anchors, fee_anchors, "
            "financial_covenant_anchors, definitions_anchor_range\n"
            "- Do not include any other keys.\n"
            "- Do not include any text before/after the JSON.\n"
            "- If a bucket has no anchors, return an empty array for that bucket (but keep the key).\n"
        )
    return f"{base_prompt}\n\n=== RETRY REQUIRED ===\nError: {error}\n\n{rules}"


def run_indexing_v2(
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
    skip_existing: bool = False,
) -> None:
    """Index anchors via the agent-gateway using v2 schema (selection + metadata)."""

    assert_exists(prompt_path, message=f"Indexing v2 prompt not found: {prompt_path}")

    model = REQUIRED_MODEL
    reasoning = REQUIRED_REASONING
    out_dir = paths.run_dir / "indexing_v2"
    out_dir.mkdir(parents=True, exist_ok=True)

    prompt_digest = prompt_hash(prompt_path)
    prompt_template = prompt_path.read_text()

    item_list = list(item_ids)
    attempts = max(1, attempts)

    async def _run_async(items: list[str]) -> None:
        GatewayAgentClient = _ensure_gateway_client_async()
        sem = asyncio.Semaphore(max(1, concurrency))
        resolved_gateway_url = gateway_url or DEFAULT_GATEWAY_URL

        failures: list[tuple[str, str]] = []

        async with GatewayAgentClient(base_url=resolved_gateway_url, timeout=gateway_timeout or 600.0) as client:

            async def _process(item_id: str) -> None:
                async with sem:
                    json_path = out_dir / f"{item_id}_anchors.json"
                    if skip_existing and json_path.exists():
                        return

                    catalog = load_anchor_catalog(paths, item_id)
                    doc_block = _canonical_annotated_text(paths, item_id)
                    base_prompt = _render_prompt(prompt_template, doc_block)

                    def _validate(payload: object) -> IndexingSelectionV2:
                        try:
                            selection = IndexingSelectionV2.model_validate(payload)
                        except Exception as exc:
                            raise ValueError(f"schema validation failed: {exc}") from exc

                        invalid: list[str] = []
                        for bucket in (
                            selection.metadata_anchors,
                            selection.agreement_date_anchors,
                            selection.fundamental_anchors,
                            selection.key_date_definitions_anchors,
                            selection.pricing_anchors,
                            selection.base_rate_anchors,
                            selection.spread_anchors,
                            selection.fee_anchors,
                            selection.financial_covenant_anchors,
                        ):
                            for anchor_id in bucket:
                                if anchor_id not in catalog:
                                    invalid.append(anchor_id)
                        if selection.definitions_anchor_range is not None:
                            dr = selection.definitions_anchor_range
                            if dr.start_anchor not in catalog:
                                invalid.append(dr.start_anchor)
                            if dr.end_anchor not in catalog:
                                invalid.append(dr.end_anchor)
                        if invalid:
                            examples = ", ".join(sorted(set(invalid))[:10])
                            raise ValueError(
                                "returned anchor_id values not present in anchors.tsv "
                                f"(count={len(invalid)}; examples={examples})"
                            )

                        if selection.definitions_anchor_range is not None:
                            dr = selection.definitions_anchor_range
                            start_order = int(catalog[dr.start_anchor]["order"])
                            end_order = int(catalog[dr.end_anchor]["order"])
                            if start_order > end_order:
                                raise ValueError(
                                    "definitions_anchor_range has start_anchor after end_anchor "
                                    f"(start={dr.start_anchor} order={start_order}; end={dr.end_anchor} order={end_order})"
                                )

                        return selection

                    try:
                        selection, _raw_text, attempts_used = await call_strict_json(
                            client=client,
                            prompt=base_prompt,
                            model=model or REQUIRED_MODEL,
                            temperature=temperature,
                            reasoning=reasoning,
                            attempts=attempts,
                            retry_prompt=_retry_prompt,
                            allowed_root_types=(dict,),
                            validate=_validate,
                        )
                    except StrictJsonFailure as exc:
                        failures.append((item_id, exc.last_error))
                        return

                    auto_added_anchors: list[dict] = []

                    artifact = IndexingSelectionV2Artifact(
                        schema_version="indexing_selection_v2",
                        stage="indexing_v2",
                        run_id=paths.run_id,
                        item_id=item_id,
                        created_at=int(time.time()),
                        gateway_url=resolved_gateway_url,
                        model=model,
                        reasoning_effort=reasoning,
                        temperature=float(temperature),
                        prompt=str(prompt_path),
                        prompt_sha256=prompt_digest,
                        attempts_used=attempts_used,
                        selection=selection,
                        auto_added_anchors=auto_added_anchors,
                    )
                    json_path.write_text(artifact.model_dump_json(indent=2))
                    return

            await asyncio.gather(*(_process(i) for i in items))

        if failures:
            joined = "; ".join(f"{item_id}: {err}" for item_id, err in failures)
            raise RuntimeError(f"Indexing v2 failed for {len(failures)} items: {joined}")

    asyncio.run(_run_async(item_list))

    manifest_path = paths.manifest_path
    if manifest_path.exists():
        update_manifest(
            manifest_path,
            indexing_v2_prompt=str(prompt_path),
            indexing_v2_prompt_sha256=prompt_digest,
        )
