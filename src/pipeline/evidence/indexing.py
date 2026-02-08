from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from typing import Iterable

from pipeline.core.config import Paths, prompt_hash, update_manifest, REQUIRED_MODEL, REQUIRED_REASONING
from pipeline.core.anchors import load_anchor_catalog
from pipeline.llm.gateway import (
    DEFAULT_GATEWAY_URL,
    GatewayUnavailable as IndexingGatewayUnavailable,
    _ensure_gateway_client_async,
    _ensure_gateway_client_sync,
)
from pipeline.schemas import IndexingSelection, IndexingSelectionArtifact
from pipeline.utils import assert_exists


DEFAULT_MODEL = REQUIRED_MODEL


def _canonical_annotated_text(paths: Paths, item_id: str) -> str:
    """Load the full annotated canonical text (with [[A####]] markers) for the item.

    This is produced by the normalization stage at:
      runs/<run_id>/normalized/<item_id>/canonical_annotated.txt
    """

    annotated = paths.normalized_dir / item_id / "canonical_annotated.txt"
    if not annotated.exists():
        raise FileNotFoundError(f"Annotated text missing for {item_id}: expected {annotated}")
    return annotated.read_text().strip()


def _render_prompt(template: str, doc_block: str) -> str:
    """Blend the user template with the full annotated document block.

    We require the template to contain `{document}` to avoid "magical" fallback behavior and to
    keep the model contract explicit.
    """

    template = template.strip()
    if "{document}" not in template:
        raise ValueError(
            "Indexing prompt template must include a `{document}` placeholder so the full "
            "annotated document can be injected."
        )
    return template.replace("{document}", doc_block)

def _retry_prompt(base_prompt: str, *, attempt: int, error: str, max_anchor_id: str) -> str:
    """Generate an explicit retry prompt for strict JSON output."""

    if attempt == 2:
        rules = (
            "Your previous response was not valid. Fix it.\n"
            "- Output MUST be a single JSON object (no markdown, no code fences, no prose).\n"
            "- JSON MUST match this exact schema:\n"
            "  {\"fundamental_anchors\": [...], \"pricing_anchors\": [...], \"financial_covenant_anchors\": [...]}\n"
            "- Each value MUST be a JSON array of UNIQUE anchor IDs like \"A0001\".\n"
            f"- Valid anchor IDs for this document range from A0001 to {max_anchor_id}.\n"
        )
    else:
        rules = (
            "STRICT MODE (final attempt):\n"
            "- Return ONLY JSON with exactly these keys:\n"
            "  fundamental_anchors, pricing_anchors, financial_covenant_anchors\n"
            "- Do not include any other keys.\n"
            "- Do not include any text before/after the JSON.\n"
            f"- Valid anchor IDs range from A0001 to {max_anchor_id}.\n"
            "- If a bucket has no anchors, return an empty array for that bucket (but keep the key).\n"
        )

    return f"{base_prompt}\n\n=== RETRY REQUIRED ===\nError: {error}\n\n{rules}"


def run_indexing(
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
) -> None:
    """Index anchors via the agent-gateway (selection-only, strict JSON).

    Contract:
    - The model must return a single JSON object matching IndexingSelection exactly.
    - We validate with Pydantic and reject any response that is not valid JSON or does not
      match the schema.
    - We also validate that every anchor_id returned exists in anchors.tsv for the item.
    - We retry up to 3 attempts per item; after that the run fails loudly.

    Output:
      runs/<run_id>/indexing/{item_id}_anchors.json
        - Contains the model selection only (plus minimal metadata).
        - Does NOT embed start/end/type spans (those live in anchors.tsv).
    """

    assert_exists(prompt_path, message=f"Indexing prompt not found: {prompt_path}")

    # Hard enforcement of model + reasoning defaults.
    model = REQUIRED_MODEL
    reasoning = REQUIRED_REASONING
    out_dir = paths.indexing_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    prompt_digest = prompt_hash(prompt_path)
    prompt_template = prompt_path.read_text()

    item_list = list(item_ids)

    async def _run_async(items: list[str]) -> None:
        GatewayAgentClient = _ensure_gateway_client_async()
        sem = asyncio.Semaphore(max(1, concurrency))
        resolved_gateway_url = gateway_url or DEFAULT_GATEWAY_URL

        failures: list[tuple[str, str]] = []

        async with GatewayAgentClient(base_url=resolved_gateway_url, timeout=gateway_timeout or 30.0) as client:

            async def _process(item_id: str) -> None:
                async with sem:
                    catalog = load_anchor_catalog(paths, item_id)
                    max_anchor_id = max(catalog.keys(), key=lambda aid: int(aid[1:]))

                    doc_block = _canonical_annotated_text(paths, item_id)
                    base_prompt = _render_prompt(prompt_template, doc_block)

                    # Remove stale sidecars from older runs; we only write JSON in this mode.
                    for legacy in (
                        out_dir / f"{item_id}_anchors.txt",
                        out_dir / f"{item_id}_anchors.retry.txt",
                        out_dir / f"{item_id}_anchors.error.txt",
                        out_dir / f"{item_id}_anchors.retry.error.txt",
                        out_dir / f"{item_id}_anchors.final.error.txt",
                    ):
                        if legacy.exists():
                            legacy.unlink()

                    last_error = "unknown"
                    for attempt in (1, 2, 3):
                        prompt = base_prompt
                        if attempt > 1:
                            prompt = _retry_prompt(
                                base_prompt,
                                attempt=attempt,
                                error=last_error,
                                max_anchor_id=max_anchor_id,
                            )

                        result = await client.complete_response(
                            model=model or DEFAULT_MODEL,
                            input_messages=[{"role": "user", "content": prompt}],
                            reasoning={"effort": reasoning} if reasoning else None,
                            temperature=temperature,
                            max_output_tokens=None,
                            metadata=None,
                        )

                        raw_text = result.get("text") if isinstance(result, dict) else str(result)
                        try:
                            payload = json.loads(raw_text)
                        except json.JSONDecodeError as exc:
                            last_error = f"invalid JSON: {exc}"
                            continue

                        try:
                            selection = IndexingSelection.model_validate(payload)
                        except Exception as exc:
                            last_error = f"schema validation failed: {exc}"
                            continue

                        invalid: list[str] = []
                        for aid in (
                            selection.fundamental_anchors
                            + selection.pricing_anchors
                            + selection.financial_covenant_anchors
                        ):
                            if aid not in catalog:
                                invalid.append(aid)
                        if invalid:
                            examples = ", ".join(invalid[:10])
                            last_error = (
                                f"returned anchor_id values not present in anchors.tsv "
                                f"(count={len(invalid)}; examples={examples})"
                            )
                            continue

                        artifact = IndexingSelectionArtifact(
                            schema_version="indexing_selection_v1",
                            stage="indexing",
                            run_id=paths.run_id,
                            item_id=item_id,
                            created_at=int(time.time()),
                            gateway_url=resolved_gateway_url,
                            model=model,
                            reasoning_effort=reasoning,
                            temperature=float(temperature),
                            prompt=str(prompt_path),
                            prompt_sha256=prompt_digest,
                            attempts_used=attempt,
                            selection=selection,
                        )
                        json_path = out_dir / f"{item_id}_anchors.json"
                        json_path.write_text(artifact.model_dump_json(indent=2))
                        return

                    failures.append((item_id, last_error))

            await asyncio.gather(*(_process(i) for i in items))

        if failures:
            joined = "; ".join(f"{item_id}: {err}" for item_id, err in failures)
            raise RuntimeError(f"Indexing failed for {len(failures)} items: {joined}")

    asyncio.run(_run_async(item_list))

    manifest_path = paths.manifest_path
    if manifest_path.exists():
        update_manifest(
            manifest_path,
            indexing_prompt=str(prompt_path),
            indexing_prompt_sha256=prompt_digest,
        )
