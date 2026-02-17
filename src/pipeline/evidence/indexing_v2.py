from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import Iterable

from pipeline.core.anchors import load_anchor_catalog
from pipeline.core.config import Paths, REQUIRED_MODEL, REQUIRED_REASONING, prompt_hash, update_manifest
from pipeline.llm.gateway import DEFAULT_GATEWAY_URL, _ensure_gateway_client_async
from pipeline.llm.strict_json import StrictJsonFailure, call_strict_json
from pipeline.schemas_v2 import IndexingSelectionV2, IndexingSelectionV2Artifact
from pipeline.utils import assert_exists, prompt_view_path

# Indexing v2 runs with enforced reasoning=medium. On long agreements, low output budgets can be
# fully consumed by reasoning tokens, yielding empty assistant text. Keep a high enough cap so the
# model can emit the final JSON payload after reasoning.
INDEXING_V2_MAX_OUTPUT_TOKENS = 24000
_INDEXING_V2_ANCHOR_BUCKET_KEYS = (
    "metadata_anchors",
    "agreement_date_anchors",
    "fundamental_anchors",
    "key_date_definitions_anchors",
    "pricing_anchors",
    "base_rate_anchors",
    "spread_anchors",
    "fee_anchors",
    "financial_covenant_anchors",
)


def _selection_has_any_anchor(selection: IndexingSelectionV2) -> bool:
    return any(
        (
            selection.metadata_anchors,
            selection.agreement_date_anchors,
            selection.fundamental_anchors,
            selection.key_date_definitions_anchors,
            selection.pricing_anchors,
            selection.base_rate_anchors,
            selection.spread_anchors,
            selection.fee_anchors,
            selection.financial_covenant_anchors,
        )
    ) or selection.definitions_anchor_range is not None


def _dedupe_preserve(values: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out


def _normalize_indexing_payload(payload: object) -> object:
    """Coerce minor list-shape issues before strict schema validation.

    The model occasionally repeats an anchor ID inside the same bucket. Dedupe the
    repeated IDs while preserving order so downstream selection remains stable.
    """

    if not isinstance(payload, dict):
        return payload

    normalized = dict(payload)
    for key in _INDEXING_V2_ANCHOR_BUCKET_KEYS:
        raw = normalized.get(key)
        if not isinstance(raw, list):
            continue
        deduped: list[object] = []
        seen_ids: set[str] = set()
        for value in raw:
            if isinstance(value, str):
                anchor_id = value.strip()
                if not anchor_id or anchor_id in seen_ids:
                    continue
                seen_ids.add(anchor_id)
                deduped.append(anchor_id)
            else:
                # Preserve non-string values so schema validation still surfaces invalid types.
                deduped.append(value)
        normalized[key] = deduped
    return normalized


def _pick_anchor_ids_by_terms(
    *,
    ordered_anchor_ids: list[str],
    anchor_text: dict[str, str],
    terms: list[str],
    limit: int,
) -> list[str]:
    selected: list[str] = []
    lowered_terms = [t.lower() for t in terms if t.strip()]
    if not lowered_terms:
        return selected
    for aid in ordered_anchor_ids:
        txt = (anchor_text.get(aid) or "").lower()
        if not txt:
            continue
        if any(term in txt for term in lowered_terms):
            selected.append(aid)
            if len(selected) >= limit:
                break
    return selected


def _apply_empty_selection_fallback(
    *,
    paths: Paths,
    item_id: str,
    selection: IndexingSelectionV2,
    catalog: dict[str, dict[str, int | str]],
) -> tuple[IndexingSelectionV2, list[dict]]:
    """Deterministically seed anchors when the model returns an empty selection.

    This preserves indexing as the relevance gate while preventing hard pipeline stops on
    rare all-empty model outputs.
    """

    if _selection_has_any_anchor(selection):
        return selection, []

    canonical = prompt_view_path(paths, item_id).read_text(encoding="utf-8")
    ordered_anchor_ids = [
        aid
        for aid, _info in sorted(
            catalog.items(),
            key=lambda kv: int(kv[1]["order"]),
        )
    ]
    anchor_text: dict[str, str] = {}
    for aid in ordered_anchor_ids:
        info = catalog[aid]
        start = int(info["start"])
        end = int(info["end"])
        if start < 0 or end < 0 or end < start:
            continue
        anchor_text[aid] = canonical[start:end]

    metadata = _pick_anchor_ids_by_terms(
        ordered_anchor_ids=ordered_anchor_ids,
        anchor_text=anchor_text,
        terms=["credit agreement", "dated as of", "among", "borrower", "lender"],
        limit=5,
    )
    agreement_dates = _pick_anchor_ids_by_terms(
        ordered_anchor_ids=ordered_anchor_ids,
        anchor_text=anchor_text,
        terms=[
            "closing date",
            "effective date",
            "maturity date",
            "termination date",
            "restatement effective date",
            "amendment effective date",
        ],
        limit=8,
    )
    key_date_definitions = _pick_anchor_ids_by_terms(
        ordered_anchor_ids=ordered_anchor_ids,
        anchor_text=anchor_text,
        terms=[
            "closing date means",
            "effective date means",
            "maturity date means",
            "termination date means",
        ],
        limit=8,
    )
    base_rate = _pick_anchor_ids_by_terms(
        ordered_anchor_ids=ordered_anchor_ids,
        anchor_text=anchor_text,
        terms=[
            "base rate",
            "abr",
            "alternate base rate",
            "term sofr",
            "adjusted libor",
            "benchmark",
            "federal funds rate",
        ],
        limit=12,
    )
    spread = _pick_anchor_ids_by_terms(
        ordered_anchor_ids=ordered_anchor_ids,
        anchor_text=anchor_text,
        terms=[
            "applicable margin",
            "margin",
            "spread",
            "pricing level",
            "pricing grid",
            "leverage ratio",
        ],
        limit=12,
    )
    fee = _pick_anchor_ids_by_terms(
        ordered_anchor_ids=ordered_anchor_ids,
        anchor_text=anchor_text,
        terms=[
            "commitment fee",
            "facility fee",
            "utilization fee",
            "letter of credit fee",
            "fronting fee",
            "unused fee",
        ],
        limit=12,
    )
    pricing = _dedupe_preserve(
        base_rate
        + spread
        + fee
        + _pick_anchor_ids_by_terms(
            ordered_anchor_ids=ordered_anchor_ids,
            anchor_text=anchor_text,
            terms=["interest rate", "rate per annum", "eurodollar rate", "sofr"],
            limit=10,
        )
    )[:18]
    financial_covenant = _pick_anchor_ids_by_terms(
        ordered_anchor_ids=ordered_anchor_ids,
        anchor_text=anchor_text,
        terms=[
            "financial covenant",
            "maximum leverage ratio",
            "minimum interest coverage",
            "fixed charge coverage ratio",
            "consolidated leverage ratio",
        ],
        limit=12,
    )
    fundamental = _dedupe_preserve(
        agreement_dates
        + key_date_definitions
        + _pick_anchor_ids_by_terms(
            ordered_anchor_ids=ordered_anchor_ids,
            anchor_text=anchor_text,
            terms=[
                "revolving credit",
                "term loan",
                "commitment",
                "facility",
                "maturity",
                "availability period",
            ],
            limit=12,
        )
    )[:18]

    # Final guardrail: if all keyword buckets missed, seed with earliest anchors.
    if not any((metadata, agreement_dates, fundamental, pricing, base_rate, spread, fee, financial_covenant)):
        seed = ordered_anchor_ids[:8]
        metadata = seed[:3]
        fundamental = seed[:8]
        pricing = seed[:8]

    fallback_by_bucket: dict[str, list[str]] = {
        "metadata": metadata,
        "agreement_dates": agreement_dates,
        "fundamental": fundamental,
        "key_date_definitions": key_date_definitions,
        "pricing": pricing,
        "base_rate": base_rate,
        "spread": spread,
        "fee": fee,
        "financial_covenant": financial_covenant,
    }

    auto_added_anchors: list[dict] = []
    for bucket, aids in fallback_by_bucket.items():
        for aid in _dedupe_preserve(aids):
            auto_added_anchors.append(
                {
                    "anchor_id": aid,
                    "bucket": bucket,
                    "reason": "empty_selection_fallback_keyword_match",
                }
            )

    seeded = selection.model_copy(
        update={
            "metadata_anchors": _dedupe_preserve(metadata),
            "agreement_date_anchors": _dedupe_preserve(agreement_dates),
            "fundamental_anchors": _dedupe_preserve(fundamental),
            "key_date_definitions_anchors": _dedupe_preserve(key_date_definitions),
            "pricing_anchors": _dedupe_preserve(pricing),
            "base_rate_anchors": _dedupe_preserve(base_rate),
            "spread_anchors": _dedupe_preserve(spread),
            "fee_anchors": _dedupe_preserve(fee),
            "financial_covenant_anchors": _dedupe_preserve(financial_covenant),
        }
    )
    return seeded, auto_added_anchors


def validate_indexing_selection_v2_payload(
    payload: object,
    *,
    catalog: dict[str, dict[str, int | str]],
) -> IndexingSelectionV2:
    """Validate an indexing-v2 payload against schema and run-local anchors."""

    payload = _normalize_indexing_payload(payload)

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

                    try:
                        selection, _raw_text, attempts_used = await call_strict_json(
                            client=client,
                            prompt=base_prompt,
                            model=model or REQUIRED_MODEL,
                            temperature=temperature,
                            reasoning=reasoning,
                            max_output_tokens=INDEXING_V2_MAX_OUTPUT_TOKENS,
                            attempts=attempts,
                            retry_prompt=_retry_prompt,
                            allowed_root_types=(dict,),
                            validate=lambda payload: validate_indexing_selection_v2_payload(payload, catalog=catalog),
                        )
                    except StrictJsonFailure as exc:
                        (out_dir / f"{item_id}_anchors.error.txt").write_text(
                            f"{exc.last_error}\n",
                            encoding="utf-8",
                        )
                        (out_dir / f"{item_id}_anchors.raw.txt").write_text(
                            exc.last_raw_text or "",
                            encoding="utf-8",
                        )
                        failures.append((item_id, exc.last_error))
                        return

                    selection, auto_added_anchors = _apply_empty_selection_fallback(
                        paths=paths,
                        item_id=item_id,
                        selection=selection,
                        catalog=catalog,
                    )

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
