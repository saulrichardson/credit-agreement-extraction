from __future__ import annotations

import asyncio
import json
import re
import time
from pathlib import Path
from typing import Any, Iterable, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from pipeline.core.anchors import load_anchor_catalog
from pipeline.core.config import Paths, REQUIRED_MODEL, REQUIRED_REASONING, prompt_hash, update_manifest
from pipeline.llm.gateway import DEFAULT_GATEWAY_URL, _ensure_gateway_client_async
from pipeline.llm.strict_json import StrictJsonFailure, call_strict_json
from pipeline.extract.party_extraction import (
    MAX_EXPLICIT_LENDERS,
    extract_lenders_from_snippets as extract_lenders_from_snippets_det,
    looks_like_enumerated_lender_list,
)
from pipeline.utils import assert_exists


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


class Party(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    source_refs: list[str] = Field(default_factory=list)
    notes: str | None = None

    @field_validator("name")
    @classmethod
    def _name_validate(cls, value: str) -> str:
        if not isinstance(value, str):
            raise TypeError("name must be a string")
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("name must be non-empty")
        return cleaned

    @field_validator("source_refs")
    @classmethod
    def _source_refs_validate(cls, value: list[str]) -> list[str]:
        return _validate_anchor_ids(value)


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
            '(e.g., "the date that is the fifth anniversary of the Maturity Date", "the earlier of ...").'
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
    facility_end: DateTerm | None = None
    facility_end_basis: Literal["maturity", "termination", "availability_end", "mixed", "unknown"] | None = None

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

    @model_validator(mode="after")
    def _facility_end_requires_supporting_refs(self) -> "FacilityFundamentals":
        if self.facility_end is not None and not self.source_refs:
            raise ValueError("facility_end requires non-empty source_refs")
        if self.facility_end_basis is not None and self.facility_end is None:
            raise ValueError("facility_end_basis must be null when facility_end is null")
        return self


class FacilityFundamentalsArtifact(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["facility_fundamentals_v1"]

    agreement_maturity_date: AgreementDateTerm | None = None
    agreement_effective_date: AgreementDateTerm | None = None

    lenders: list[Party] = Field(default_factory=list)
    lenders_present_but_omitted: bool = False
    lenders_source_refs: list[str] = Field(default_factory=list)

    facilities: list[FacilityFundamentals] = Field(default_factory=list)
    notes: str | None = None

    @field_validator("lenders_source_refs")
    @classmethod
    def _lenders_source_refs_validate(cls, value: list[str]) -> list[str]:
        if not isinstance(value, list):
            raise TypeError("lenders_source_refs must be a JSON array")
        if not value:
            return []
        return _validate_anchor_ids(value)


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
            "- agreement_maturity_date / agreement_effective_date objects MUST include non-empty source_refs when provided.\n"
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
        else ["agreement_dates", "fundamental", "key_date_definitions", "metadata"]
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
                        rec_by_anchor = {
                            str(rec.get("anchor_id")): rec
                            for rec in snippets
                            if isinstance(rec.get("anchor_id"), str)
                        }

                        def _check_anchors(anchor_ids: list[str], *, ctx: str) -> None:
                            missing = [a for a in anchor_ids if a not in catalog]
                            if missing:
                                raise ValueError(f"{ctx}: unknown anchor_ids (examples): {missing[:10]}")

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

                            if artifact.agreement_maturity_date is not None:
                                _check_anchors(
                                    artifact.agreement_maturity_date.source_refs,
                                    ctx="agreement_maturity_date.source_refs",
                                )
                            if artifact.agreement_effective_date is not None:
                                _check_anchors(
                                    artifact.agreement_effective_date.source_refs,
                                    ctx="agreement_effective_date.source_refs",
                                )
                            if artifact.lenders_source_refs:
                                _check_anchors(artifact.lenders_source_refs, ctx="lenders_source_refs")
                            for i, lender in enumerate(artifact.lenders):
                                _check_anchors(lender.source_refs, ctx=f"lenders[{i}].source_refs")
                            for i, facility in enumerate(artifact.facilities):
                                _check_anchors(facility.source_refs, ctx=f"facilities[{i}].source_refs")
                                if facility.facility_end is not None and not facility.source_refs:
                                    raise ValueError(f"facilities[{i}].facility_end requires non-empty source_refs")

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

                        auto_corrections: list[dict[str, Any]] = []

                        det_lender_dicts, det_refs = extract_lenders_from_snippets_det(snippets=snippets, catalog=catalog)
                        det_lenders = [Party.model_validate(row) for row in det_lender_dicts]
                        det_count = len(det_lenders)
                        if det_count:
                            if det_count <= MAX_EXPLICIT_LENDERS:
                                if det_count > len(artifact.lenders) or artifact.lenders_present_but_omitted:
                                    old_count = len(artifact.lenders)
                                    old_present = bool(artifact.lenders_present_but_omitted)
                                    old_refs = list(artifact.lenders_source_refs or [])
                                    artifact.lenders = det_lenders
                                    artifact.lenders_present_but_omitted = False
                                    artifact.lenders_source_refs = det_refs
                                    auto_corrections.append(
                                        {
                                            "field": "lenders",
                                            "value": f"deterministic lenders extracted (count={det_count})",
                                            "old_source_refs": old_refs,
                                            "new_source_refs": det_refs,
                                            "reason": (
                                                "replaced model lenders output with deterministic extraction from signature snippets "
                                                f"(old_count={old_count}, old_present_but_omitted={old_present})"
                                            ),
                                        }
                                    )
                            else:
                                if (not artifact.lenders_present_but_omitted) or artifact.lenders:
                                    old_count = len(artifact.lenders)
                                    old_present = bool(artifact.lenders_present_but_omitted)
                                    old_refs = list(artifact.lenders_source_refs or [])
                                    artifact.lenders = []
                                    artifact.lenders_present_but_omitted = True
                                    artifact.lenders_source_refs = det_refs
                                    auto_corrections.append(
                                        {
                                            "field": "lenders_present_but_omitted",
                                            "value": f"deterministic lenders extracted (count={det_count})",
                                            "old_source_refs": old_refs,
                                            "new_source_refs": det_refs,
                                            "reason": (
                                                "set lenders_present_but_omitted=true due to deterministic extraction exceeding max "
                                                f"(max={MAX_EXPLICIT_LENDERS}, old_count={old_count}, old_present_but_omitted={old_present})"
                                            ),
                                        }
                                    )
                                elif artifact.lenders_present_but_omitted and not artifact.lenders_source_refs:
                                    artifact.lenders_source_refs = det_refs
                                    auto_corrections.append(
                                        {
                                            "field": "lenders_source_refs",
                                            "value": f"filled from deterministic lender evidence (count={det_count})",
                                            "old_source_refs": [],
                                            "new_source_refs": det_refs,
                                            "reason": "model omitted lenders_source_refs but snippets contained an enumerated lender list",
                                        }
                                    )

                        if artifact.lenders_present_but_omitted and not artifact.lenders_source_refs:
                            failures.append((item_id, "lenders_present_but_omitted is true but lenders_source_refs is empty"))
                            return
                        if artifact.lenders_present_but_omitted and artifact.lenders:
                            failures.append((item_id, "lenders_present_but_omitted is true but lenders[] is non-empty"))
                            return
                        if not artifact.lenders_present_but_omitted and not artifact.lenders and artifact.lenders_source_refs:
                            failures.append(
                                (
                                    item_id,
                                    "lenders_source_refs must be [] when no lenders are extracted and lenders_present_but_omitted is false",
                                )
                            )
                            return
                        if artifact.lenders and not artifact.lenders_source_refs:
                            old_refs = list(artifact.lenders_source_refs or [])
                            artifact.lenders_source_refs = sorted(
                                {aid for p in artifact.lenders for aid in (p.source_refs or []) if isinstance(aid, str)}
                            )
                            auto_corrections.append(
                                {
                                    "field": "lenders_source_refs",
                                    "value": "filled from lenders[].source_refs",
                                    "old_source_refs": old_refs,
                                    "new_source_refs": artifact.lenders_source_refs,
                                    "reason": "filled missing lenders_source_refs from explicit lenders list",
                                }
                            )

                        if artifact.lenders_present_but_omitted:
                            ok = False
                            for aid in artifact.lenders_source_refs:
                                rec = rec_by_anchor.get(aid) or {}
                                snippet = str(rec.get("snippet") or "")
                                if len(snippet) < 200:
                                    continue
                                if looks_like_enumerated_lender_list(snippet):
                                    ok = True
                                    break
                            if not ok:
                                old_refs = list(artifact.lenders_source_refs or [])
                                artifact.lenders_present_but_omitted = False
                                artifact.lenders_source_refs = []
                                auto_corrections.append(
                                    {
                                        "field": "lenders_present_but_omitted",
                                        "value": False,
                                        "old_source_refs": old_refs,
                                        "new_source_refs": [],
                                        "reason": (
                                            "cleared unsupported lenders_present_but_omitted claim: cited anchors did not "
                                            "contain an enumerated lender grid/list"
                                        ),
                                    }
                                )

                        if artifact.agreement_maturity_date is not None:
                            _check_anchors(artifact.agreement_maturity_date.source_refs, ctx="agreement_maturity_date.source_refs")
                        if artifact.agreement_effective_date is not None:
                            _check_anchors(artifact.agreement_effective_date.source_refs, ctx="agreement_effective_date.source_refs")
                        if artifact.lenders_source_refs:
                            _check_anchors(artifact.lenders_source_refs, ctx="lenders_source_refs")
                        for i, lender in enumerate(artifact.lenders):
                            _check_anchors(lender.source_refs, ctx=f"lenders[{i}].source_refs")
                        for i, facility in enumerate(artifact.facilities):
                            _check_anchors(facility.source_refs, ctx=f"facilities[{i}].source_refs")
                            if facility.facility_end is not None and not facility.source_refs:
                                failures.append((item_id, f"facilities[{i}].facility_end requires non-empty source_refs"))
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
                                    "categories": wanted_categories,
                                    "auto_corrections": auto_corrections,
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
