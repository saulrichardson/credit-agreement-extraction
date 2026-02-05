from __future__ import annotations

import asyncio
import json
import re
import time
from pathlib import Path
from typing import Any, Iterable, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .anchors import load_anchor_catalog
from .config import Paths, REQUIRED_MODEL, REQUIRED_REASONING, prompt_hash, update_manifest
from .indexing import DEFAULT_GATEWAY_URL, _ensure_gateway_client_async  # type: ignore
from .llm.strict_json import StrictJsonFailure, call_strict_json
from .utils import assert_exists


ANCHOR_ID_RE = re.compile(r"^A\d{4,}$")
ISO_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
CURRENCY_RE = re.compile(r"^[A-Z]{3}$")


def _validate_anchor_ids(anchor_ids: list[str], *, allow_empty: bool = False) -> list[str]:
    if not isinstance(anchor_ids, list):
        raise TypeError("source_refs must be a JSON array")
    if not anchor_ids and not allow_empty:
        raise ValueError("source_refs must be a non-empty array")
    cleaned: list[str] = []
    for idx, raw in enumerate(anchor_ids):
        if not isinstance(raw, str):
            raise TypeError(f"anchor IDs must be strings; got {type(raw).__name__} at index {idx}")
        anchor_id = raw.strip()
        if not ANCHOR_ID_RE.fullmatch(anchor_id):
            raise ValueError(f"invalid anchor_id {anchor_id!r} (expected like 'A0001')")
        cleaned.append(anchor_id)
    return cleaned


class Money(BaseModel):
    model_config = ConfigDict(extra="forbid")

    amount: float | None = None
    currency: str | None = Field(default=None, description="3-letter ISO currency code when explicitly stated")

    @field_validator("amount")
    @classmethod
    def _amount_validate(cls, value: float | None) -> float | None:
        if value is None:
            return None
        if not isinstance(value, (int, float)):
            raise TypeError("amount must be a JSON number")
        if not float(value) == float(value):  # NaN
            raise ValueError("amount must not be NaN")
        if float(value) <= 0:
            raise ValueError("amount must be > 0 when provided")
        return float(value)

    @field_validator("currency")
    @classmethod
    def _currency_validate(cls, value: str | None) -> str | None:
        if value is None:
            return None
        if not isinstance(value, str):
            raise TypeError("currency must be a string")
        cur = value.strip().upper()
        if not cur:
            return None
        if not CURRENCY_RE.fullmatch(cur):
            raise ValueError("currency must be a 3-letter ISO code like 'USD' when provided")
        return cur


class DateTerm(BaseModel):
    model_config = ConfigDict(extra="forbid")

    date: str | None = Field(default=None, description="YYYY-MM-DD when explicitly stated; otherwise null")
    text: str | None = Field(
        default=None,
        description=(
            "Verbatim date expression when the agreement uses relative/conditional language "
            '(e.g., "the date that is the fifth anniversary of the Closing Date", "the earlier of ...").'
        ),
    )
    notes: str | None = None

    @field_validator("date")
    @classmethod
    def _date_validate(cls, value: str | None) -> str | None:
        if value is None:
            return None
        if not isinstance(value, str):
            raise TypeError("date must be a string")
        cleaned = value.strip()
        if not cleaned:
            return None
        if not ISO_DATE_RE.fullmatch(cleaned):
            raise ValueError("date must be in YYYY-MM-DD format when provided")
        return cleaned

    @field_validator("text")
    @classmethod
    def _text_validate(cls, value: str | None) -> str | None:
        if value is None:
            return None
        if not isinstance(value, str):
            raise TypeError("text must be a string")
        cleaned = value.strip()
        if not cleaned:
            return None
        return cleaned

    @model_validator(mode="after")
    def _at_least_one(self) -> "DateTerm":
        # Allow fully-null objects in case the model outputs placeholders; downstream users can ignore them.
        # The prompt should push the model to use null instead of placeholder objects.
        return self


class AgreementDateTerm(DateTerm):
    model_config = ConfigDict(extra="forbid")

    source_refs: list[str] = Field(default_factory=list)

    @field_validator("source_refs")
    @classmethod
    def _source_refs_validate(cls, value: list[str]) -> list[str]:
        return _validate_anchor_ids(value)


class FacilityFundamentals(BaseModel):
    model_config = ConfigDict(extra="forbid")

    facility_name: str | None = None
    facility_types: list[str] = Field(default_factory=list)
    committed_amount: Money | None = None
    currency: str | None = Field(default=None, description="3-letter ISO currency code when explicitly stated")

    availability_start: DateTerm | None = None
    availability_end: DateTerm | None = None
    maturity: DateTerm | None = None
    termination: DateTerm | None = None

    source_refs: list[str] = Field(default_factory=list)
    notes: str | None = None

    @field_validator("facility_name")
    @classmethod
    def _facility_name_validate(cls, value: str | None) -> str | None:
        if value is None:
            return None
        if not isinstance(value, str):
            raise TypeError("facility_name must be a string")
        cleaned = value.strip()
        if not cleaned:
            return None
        return cleaned

    @field_validator("facility_types")
    @classmethod
    def _facility_types_validate(cls, value: list[str]) -> list[str]:
        if not isinstance(value, list):
            raise TypeError("facility_types must be a JSON array")
        if not value:
            raise ValueError("facility_types must be non-empty when a facility object is provided")
        allowed = {"revolver", "term_loan", "letters_of_credit", "swingline", "delayed_draw", "other"}
        cleaned: list[str] = []
        for idx, raw in enumerate(value):
            if not isinstance(raw, str):
                raise TypeError(f"facility_types entries must be strings; got {type(raw).__name__} at index {idx}")
            v = raw.strip()
            if v not in allowed:
                raise ValueError(f"invalid facility_type {v!r} (allowed: {', '.join(sorted(allowed))})")
            cleaned.append(v)
        # de-dupe, preserve order
        seen: set[str] = set()
        out: list[str] = []
        for v in cleaned:
            if v in seen:
                continue
            seen.add(v)
            out.append(v)
        return out

    @field_validator("currency")
    @classmethod
    def _currency_validate(cls, value: str | None) -> str | None:
        if value is None:
            return None
        if not isinstance(value, str):
            raise TypeError("currency must be a string")
        cur = value.strip().upper()
        if not cur:
            return None
        if not CURRENCY_RE.fullmatch(cur):
            raise ValueError("currency must be a 3-letter ISO code like 'USD' when provided")
        return cur

    @field_validator("source_refs")
    @classmethod
    def _source_refs_validate(cls, value: list[str]) -> list[str]:
        return _validate_anchor_ids(value)

    @field_validator("committed_amount")
    @classmethod
    def _committed_amount_validate(cls, value: Money | None) -> Money | None:
        if value is None:
            return None
        if value.amount is None:
            raise ValueError("committed_amount.amount must be provided when committed_amount is not null")
        return value


class FacilityFundamentalsArtifact(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["facility_fundamentals_v1"]

    agreement_closing_date: AgreementDateTerm | None = None
    agreement_effective_date: AgreementDateTerm | None = None

    facilities: list[FacilityFundamentals] = Field(default_factory=list)
    notes: str | None = None


def _load_snippets_v2(paths: Paths, item_id: str) -> list[dict]:
    path = assert_exists(
        paths.run_dir / "retrieval_v2" / f"{item_id}_snippets.jsonl",
        message=f"Missing v2 snippets for {item_id}: run retrieve-v2 first.",
    )
    snippets: list[dict] = []
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


def _filter_snippets(snippets: list[dict], categories: Iterable[str]) -> list[dict]:
    wanted = {c.strip().lower() for c in categories if isinstance(c, str) and c.strip()}
    if not wanted:
        return snippets
    filtered: list[dict] = []
    for rec in snippets:
        rec_categories = [c.lower() for c in (rec.get("categories") or []) if isinstance(c, str)]
        rec_buckets = [c.lower() for c in (rec.get("buckets") or []) if isinstance(c, str)]
        label = rec.get("label") or ""
        rec_label_parts = [c.strip().lower() for c in str(label).split(",") if c.strip()]
        if any(c in wanted for c in rec_categories + rec_buckets + rec_label_parts):
            filtered.append(rec)
    return filtered


def _render_snippet_block(snippets: list[dict]) -> str:
    blocks: list[str] = []
    for rec in snippets:
        aid = rec.get("anchor_id") or "UNK"
        label = rec.get("label") or rec.get("type")
        toc_title = rec.get("toc_title")
        toc_chunk_id = rec.get("toc_chunk_id")
        snippet_text = (rec.get("snippet") or "").strip()
        header = f"[[{aid}]]"
        lines: list[str] = [header]
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
    _ = previous_output
    if attempt == 2:
        rules = (
            "Your previous response was not valid. Fix it.\n"
            "- Output MUST be a single JSON object (no markdown, no code fences, no prose).\n"
            "- JSON MUST match the exact shape described in the prompt.\n"
            "- Every facility object MUST include non-empty source_refs.\n"
            "- agreement_closing_date / agreement_effective_date objects MUST include non-empty source_refs when provided.\n"
            "- Dates MUST be YYYY-MM-DD when you provide a `date` field.\n"
        )
    else:
        rules = (
            "STRICT MODE (final attempt):\n"
            "- Output ONLY JSON.\n"
            "- Include ALL required top-level keys exactly.\n"
            "- Do NOT include any extra keys.\n"
            "- If you cannot support a field from the snippets, use null/[] and explain briefly in notes.\n"
        )
    return f"{base_prompt}\n\n=== RETRY REQUIRED ===\nError: {error}\n\n{rules}"


def run_facility_fundamentals_v1(
    paths: Paths,
    item_ids: Iterable[str],
    prompt_path: Path,
    *,
    categories: Iterable[str] | None = None,
    model: str | None = None,
    gateway_url: str | None = None,
    temperature: float = 0.0,
    reasoning: str | None = None,
    gateway_timeout: float | None = None,
    concurrency: int = 3,
    attempts: int = 3,
    output_subdir: str | None = None,
) -> None:
    """Extract facility fundamentals (dates + facility separation) from v2 retrieval snippets.

    This stage is intentionally LLM-driven (no regex-based extraction).
    It is strict about JSON shape + anchor refs, but does not attempt to deterministically
    infer or normalize relative date expressions.
    """

    assert_exists(prompt_path, message=f"Facility fundamentals prompt not found: {prompt_path}")

    model = REQUIRED_MODEL
    reasoning = REQUIRED_REASONING
    prompt_template = prompt_path.read_text()
    prompt_digest = prompt_hash(prompt_path)

    out_root = paths.run_dir / "facility_fundamentals"
    resolved_output_subdir = output_subdir or prompt_path.stem
    out_dir = out_root / resolved_output_subdir
    out_dir.mkdir(parents=True, exist_ok=True)

    wanted_categories = (
        list(categories)
        if categories is not None
        else ["agreement_dates", "fundamental", "key_date_definitions"]
    )

    items = list(item_ids)
    attempts = max(1, int(attempts))

    GatewayAgentClient = _ensure_gateway_client_async()

    async def _run_async() -> None:
        sem = asyncio.Semaphore(max(1, concurrency))
        resolved_gateway_url = gateway_url or DEFAULT_GATEWAY_URL
        failures: list[tuple[str, str]] = []

        async with GatewayAgentClient(base_url=resolved_gateway_url, timeout=gateway_timeout or 600.0) as client:

            async def _process(item_id: str) -> None:
                async with sem:
                    try:
                        snippets = _load_snippets_v2(paths, item_id)
                        if wanted_categories:
                            snippets = _filter_snippets(snippets, wanted_categories)
                        if not snippets:
                            raise RuntimeError(
                                "No snippets after category filter. "
                                f"Requested categories: {', '.join(wanted_categories)}"
                            )

                        snippet_block = _render_snippet_block(snippets)
                        base_prompt = _render_prompt(prompt_template, snippet_block)

                        catalog = load_anchor_catalog(paths, item_id)

                        def _validate(payload: object) -> FacilityFundamentalsArtifact:
                            try:
                                artifact = FacilityFundamentalsArtifact.model_validate(payload)
                            except Exception as exc:
                                raise ValueError(f"schema validation failed: {exc}") from exc

                            if artifact.schema_version != "facility_fundamentals_v1":
                                raise ValueError(
                                    "schema_version mismatch: expected 'facility_fundamentals_v1' "
                                    f"got {artifact.schema_version!r}"
                                )

                            # Validate that every referenced anchor exists in anchors.tsv.
                            def _check_anchors(anchor_ids: list[str], *, ctx: str) -> None:
                                missing = [a for a in anchor_ids if a not in catalog]
                                if missing:
                                    raise ValueError(f"{ctx}: unknown anchor_ids (examples): {missing[:10]}")

                            if artifact.agreement_closing_date is not None:
                                _check_anchors(
                                    artifact.agreement_closing_date.source_refs,
                                    ctx="agreement_closing_date.source_refs",
                                )
                            if artifact.agreement_effective_date is not None:
                                _check_anchors(
                                    artifact.agreement_effective_date.source_refs,
                                    ctx="agreement_effective_date.source_refs",
                                )
                            for i, facility in enumerate(artifact.facilities):
                                _check_anchors(facility.source_refs, ctx=f"facilities[{i}].source_refs")

                            return artifact

                        try:
                            artifact, _raw_text, attempts_used = await call_strict_json(
                                client=client,
                                prompt=base_prompt,
                                model=model,
                                temperature=temperature,
                                reasoning=reasoning,
                                attempts=attempts,
                                retry_prompt=_retry_prompt,
                                allowed_root_types=(dict,),
                                validate=_validate,
                                on_attempt=lambda attempt, raw_text: (out_dir / f"{item_id}.attempt{attempt}.raw.txt").write_text(
                                    raw_text
                                ),
                            )
                        except StrictJsonFailure as exc:
                            failures.append((item_id, exc.last_error))
                            return

                        # Success.
                        artifact_path = out_dir / f"{item_id}.json"
                        artifact_path.write_text(json.dumps(artifact.model_dump(mode="json"), indent=2))
                        meta_path = out_dir / f"{item_id}.meta.json"
                        meta_path.write_text(
                            json.dumps(
                                {
                                    "schema_version": "facility_fundamentals_v1_artifact_meta",
                                    "stage": "facility_fundamentals_v1",
                                    "run_id": paths.run_id,
                                    "item_id": item_id,
                                    "created_at": int(time.time()),
                                    "gateway_url": resolved_gateway_url,
                                    "model": model,
                                    "reasoning_effort": reasoning,
                                    "temperature": float(temperature),
                                    "prompt": str(prompt_path),
                                    "prompt_sha256": prompt_digest,
                                    "attempts_used": attempts_used,
                                },
                                indent=2,
                            )
                        )
                        return
                    except Exception as exc:
                        failures.append((item_id, str(exc)))

            await asyncio.gather(*(_process(i) for i in items))

        if failures:
            joined = "; ".join(f"{item_id}: {err}" for item_id, err in failures)
            raise RuntimeError(f"Facility fundamentals v1 failed for {len(failures)} items: {joined}")

    asyncio.run(_run_async())

    # Best-effort manifest update (only when the run manifest exists).
    manifest_path = paths.manifest_path
    if manifest_path.exists():
        update_manifest(
            manifest_path,
            facility_fundamentals_prompt=str(prompt_path),
            facility_fundamentals_prompt_sha256=prompt_digest,
        )
