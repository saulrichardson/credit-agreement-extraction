from __future__ import annotations

import asyncio
import json
import re
import time
import traceback
from pathlib import Path
from typing import Any, Iterable, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from pipeline.core.anchors import load_anchor_catalog
from pipeline.core.config import Paths, REQUIRED_MODEL, REQUIRED_REASONING, prompt_hash, update_manifest
from pipeline.llm.gateway import DEFAULT_GATEWAY_URL, _ensure_gateway_client_async
from pipeline.llm.strict_json import StrictJsonFailure, call_strict_json
from pipeline.utils import assert_exists, prompt_view_path

ANCHOR_ID_RE = re.compile(r"^A\d{4,}$")


class TocChunkV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["toc_chunk_v1"]
    chunk_id: int
    title: str
    summary: str
    topics: list[str] = Field(default_factory=list)
    key_terms: list[str] = Field(default_factory=list)
    notes: str | None = None

    @field_validator("chunk_id")
    @classmethod
    def _chunk_id_validate(cls, value: int) -> int:
        if not isinstance(value, int):
            raise TypeError("chunk_id must be an integer")
        if value <= 0:
            raise ValueError("chunk_id must be >= 1")
        return value

    @field_validator("title", "summary")
    @classmethod
    def _nonempty_str(cls, value: str) -> str:
        if not isinstance(value, str):
            raise TypeError("field must be a string")
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("field must be non-empty")
        return cleaned

    @field_validator("topics", "key_terms")
    @classmethod
    def _string_list(cls, value: list[str]) -> list[str]:
        if not isinstance(value, list):
            raise TypeError("must be a JSON array")
        cleaned: list[str] = []
        for idx, raw in enumerate(value):
            if not isinstance(raw, str):
                raise TypeError(f"entries must be strings; got {type(raw).__name__} at index {idx}")
            v = raw.strip()
            if not v:
                continue
            cleaned.append(v)
        # De-dupe, preserve order.
        seen: set[str] = set()
        out: list[str] = []
        for v in cleaned:
            key = v.lower()
            if key in seen:
                continue
            seen.add(key)
            out.append(v)
        return out


class DocTocChunk(BaseModel):
    model_config = ConfigDict(extra="forbid")

    chunk_id: int
    start_anchor: str
    end_anchor: str
    start_order: int
    end_order: int
    title: str
    summary: str
    topics: list[str] = Field(default_factory=list)
    key_terms: list[str] = Field(default_factory=list)
    notes: str | None = None

    @field_validator("start_anchor", "end_anchor")
    @classmethod
    def _anchor_id_validate(cls, value: str) -> str:
        if not isinstance(value, str):
            raise TypeError("anchor id must be a string")
        v = value.strip()
        if not ANCHOR_ID_RE.fullmatch(v):
            raise ValueError(f"invalid anchor id {value!r}")
        return v


class TocHighLevelV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["toc_high_level_v1"]
    overview: str
    major_parts: list[dict] = Field(default_factory=list)
    suggested_retrieval_queries: list[str] = Field(default_factory=list)
    notes: str | None = None


class DocumentTocV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["document_toc_v1"]
    stage: Literal["toc_v1"]
    run_id: str
    item_id: str
    created_at: int

    gateway_url: str
    model: str
    reasoning_effort: str
    temperature: float

    prompt_chunk: str
    prompt_chunk_sha256: str
    prompt_high_level: str | None = None
    prompt_high_level_sha256: str | None = None

    chunking: dict[str, Any]

    chunks: list[DocTocChunk] = Field(default_factory=list)
    high_level: TocHighLevelV1 | None = None


def _heading_score(text: str) -> int:
    t = re.sub(r"\s+", " ", (text or "")).strip()
    if not t:
        return 0
    u = t.upper()
    score = 0
    if re.match(r"^(SECTION|ARTICLE)\s+\d", u):
        score += 10
    if re.match(r"^(SCHEDULE|EXHIBIT)\b", u):
        score += 7
    if re.match(r"^\d+(?:\.\d+)+\b", t):
        score += 5
    if len(t) <= 120:
        # Uppercase-heavy short lines often correspond to captions/headings.
        letters = [c for c in t if c.isalpha()]
        if letters:
            upper_ratio = sum(1 for c in letters if c.isupper()) / len(letters)
            if upper_ratio >= 0.85:
                score += 3
    return score


def _chunk_anchor_ids(
    *,
    ordered_anchor_ids: list[str],
    anchor_text_by_id: dict[str, str],
    target_chars: int = 14_000,
    min_chars: int = 8_000,
    max_chars: int = 22_000,
    lookback_anchors: int = 40,
) -> list[list[str]]:
    if target_chars <= 0 or min_chars <= 0 or max_chars <= 0:
        raise ValueError("chunk char limits must be > 0")
    if not ordered_anchor_ids:
        return []

    chunks: list[list[str]] = []
    n = len(ordered_anchor_ids)
    i = 0
    while i < n:
        start_i = i
        size = 0
        while i < n and size < target_chars:
            aid = ordered_anchor_ids[i]
            size += len(anchor_text_by_id.get(aid, "")) + 16
            i += 1

        if i >= n:
            chunks.append(ordered_anchor_ids[start_i:n])
            break

        # Try to split at a heading-like anchor near the end of this chunk.
        # We want the NEXT chunk to start at the heading anchor.
        best_split = None
        best_score = 0
        search_start = max(start_i + 1, i - lookback_anchors)
        for split_idx in range(search_start, i):
            aid = ordered_anchor_ids[split_idx]
            sc = _heading_score(anchor_text_by_id.get(aid, ""))
            if sc <= 0:
                continue
            # Ensure current chunk isn't too small if we split here.
            cur_ids = ordered_anchor_ids[start_i:split_idx]
            cur_size = sum(len(anchor_text_by_id.get(x, "")) + 16 for x in cur_ids)
            if cur_size < min_chars:
                continue
            if sc >= best_score:
                best_score = sc
                best_split = split_idx

        if best_split is not None:
            chunks.append(ordered_anchor_ids[start_i:best_split])
            i = best_split
            continue

        # Fallback: hard split at i (bounded by max_chars).
        cur_ids = ordered_anchor_ids[start_i:i]
        cur_size = sum(len(anchor_text_by_id.get(x, "")) + 16 for x in cur_ids)
        if cur_size > max_chars:
            # Back off until within max_chars.
            j = i
            while j > start_i + 1:
                j -= 1
                candidate = ordered_anchor_ids[start_i:j]
                cand_size = sum(len(anchor_text_by_id.get(x, "")) + 16 for x in candidate)
                if cand_size <= max_chars:
                    chunks.append(candidate)
                    i = j
                    break
            else:
                chunks.append(ordered_anchor_ids[start_i:i])
        else:
            chunks.append(cur_ids)

    # Final sanity: no empty chunks.
    out = [c for c in chunks if c]
    if not out:
        raise RuntimeError("chunker produced no chunks")
    return out


def _render_chunk_text(anchor_ids: list[str], anchor_text_by_id: dict[str, str]) -> str:
    blocks: list[str] = []
    for aid in anchor_ids:
        txt = (anchor_text_by_id.get(aid) or "").strip()
        blocks.append(f"[[{aid}]]\n{txt}")
    return "\n\n".join(blocks).strip()


def _render_prompt(template: str, *, chunk_id: int, chunk_text: str) -> str:
    out = template
    out = out.replace("{{CHUNK_ID}}", str(chunk_id))
    out = out.replace("{{CHUNK_TEXT}}", chunk_text)
    return out


def _retry_prompt(base_prompt: str, attempt: int, error: str, previous_output: str) -> str:
    _ = attempt
    _ = previous_output
    strict = (
        "=== RETRY REQUIRED ===\n"
        f"Error: {error}\n\n"
        "Return JSON ONLY matching the exact schema. No extra keys.\n"
    )
    return f"{base_prompt}\n\n{strict}"


def run_toc_v1(
    paths: Paths,
    item_ids: Iterable[str],
    *,
    prompt_chunk_path: Path,
    prompt_high_level_path: Path | None = None,
    output_subdir: str | None = None,
    target_chars: int = 14_000,
    min_chars: int = 8_000,
    max_chars: int = 22_000,
    lookback_anchors: int = 40,
    high_level_chunks: int = 5,
    model: str | None = None,
    gateway_url: str | None = None,
    temperature: float = 0.0,
    reasoning: str | None = None,
    gateway_timeout: float | None = None,
    concurrency: int = 3,
    attempts: int = 3,
) -> None:
    """Build a Hebbia-style TOC via chunk summaries (reasoning-first; no embeddings).

    Output artifact: runs/<run_id>/toc_v1/<output_subdir>/<item_id>.json
    Retrieval v2 can use this artifact to attach chunk titles to snippets.
    """

    assert_exists(prompt_chunk_path, message=f"TOC chunk prompt not found: {prompt_chunk_path}")
    if prompt_high_level_path is not None:
        assert_exists(prompt_high_level_path, message=f"TOC high-level prompt not found: {prompt_high_level_path}")

    model = REQUIRED_MODEL
    reasoning = REQUIRED_REASONING

    chunk_template = prompt_chunk_path.read_text()
    chunk_prompt_digest = prompt_hash(prompt_chunk_path)

    high_template: str | None = None
    high_digest: str | None = None
    if prompt_high_level_path is not None:
        high_template = prompt_high_level_path.read_text()
        high_digest = prompt_hash(prompt_high_level_path)

    out_root = paths.run_dir / "toc_v1"
    resolved_out_subdir = output_subdir or prompt_chunk_path.stem
    out_dir = out_root / resolved_out_subdir
    out_dir.mkdir(parents=True, exist_ok=True)

    items = list(item_ids)
    attempts = max(1, int(attempts))

    GatewayAgentClient = _ensure_gateway_client_async()
    sem = asyncio.Semaphore(max(1, concurrency))
    failures: list[tuple[str, str]] = []

    async def _process_item(item_id: str, client: Any) -> None:
        async with sem:
            try:
                # Canonical text + anchors are deterministic.
                canonical_text = prompt_view_path(paths, item_id).read_text()
                catalog = load_anchor_catalog(paths, item_id)
                ordered = sorted(catalog.values(), key=lambda a: int(a["order"]))
                ordered_ids = [str(a["anchor_id"]) for a in ordered]

                anchor_text_by_id: dict[str, str] = {}
                for info in ordered:
                    aid = str(info["anchor_id"])
                    start = int(info["start"])
                    end = int(info["end"])
                    anchor_text_by_id[aid] = canonical_text[start:end]

                chunks_ids = _chunk_anchor_ids(
                    ordered_anchor_ids=ordered_ids,
                    anchor_text_by_id=anchor_text_by_id,
                    target_chars=target_chars,
                    min_chars=min_chars,
                    max_chars=max_chars,
                    lookback_anchors=lookback_anchors,
                )

                # Per-chunk LLM summaries.
                chunk_nodes: list[DocTocChunk] = []
                for idx, anchor_ids in enumerate(chunks_ids, start=1):
                    chunk_text = _render_chunk_text(anchor_ids, anchor_text_by_id)
                    base_prompt = _render_prompt(chunk_template, chunk_id=idx, chunk_text=chunk_text)

                    def _validate(payload: object) -> TocChunkV1:
                        try:
                            parsed = TocChunkV1.model_validate(payload)
                        except Exception as exc:
                            raise ValueError(f"schema validation failed: {exc}") from exc
                        if parsed.chunk_id != idx:
                            raise ValueError(f"chunk_id mismatch: expected {idx}, got {parsed.chunk_id}")
                        return parsed

                    try:
                        parsed, _raw_text, _attempts_used = await call_strict_json(
                            client=client,
                            prompt=base_prompt,
                            model=model,
                            temperature=temperature,
                            reasoning=reasoning,
                            attempts=attempts,
                            retry_prompt=_retry_prompt,
                            allowed_root_types=(dict,),
                            validate=_validate,
                            on_attempt=lambda attempt, raw_text: (out_dir / f"{item_id}.chunk{idx}.attempt{attempt}.raw.txt").write_text(
                                raw_text
                            ),
                        )
                    except StrictJsonFailure as exc:
                        raise RuntimeError(
                            f"Failed to produce valid toc_chunk_v1 for chunk {idx}: {exc.last_error}"
                        ) from exc

                    start_anchor = anchor_ids[0]
                    end_anchor = anchor_ids[-1]
                    start_order = int(catalog[start_anchor]["order"])
                    end_order = int(catalog[end_anchor]["order"])

                    chunk_nodes.append(
                        DocTocChunk(
                            chunk_id=idx,
                            start_anchor=start_anchor,
                            end_anchor=end_anchor,
                            start_order=start_order,
                            end_order=end_order,
                            title=parsed.title,
                            summary=parsed.summary,
                            topics=parsed.topics,
                            key_terms=parsed.key_terms,
                            notes=parsed.notes,
                        )
                    )

                # High-level organization over the first N chunks (experimental).
                high_level: TocHighLevelV1 | None = None
                if high_template is not None:
                    subset = chunk_nodes[: max(1, int(high_level_chunks))]
                    entries = [
                        {
                            "chunk_id": c.chunk_id,
                            "title": c.title,
                            "summary": c.summary,
                            "topics": c.topics,
                            "key_terms": c.key_terms,
                        }
                        for c in subset
                    ]
                    input_block = json.dumps(entries, indent=2)
                    prompt = high_template.replace("{{CHUNK_ENTRIES}}", input_block)
                    try:
                        high_level, _raw_text, _attempts_used = await call_strict_json(
                            client=client,
                            prompt=prompt,
                            model=model,
                            temperature=temperature,
                            reasoning=reasoning,
                            attempts=1,
                            allowed_root_types=(dict,),
                            validate=lambda payload: TocHighLevelV1.model_validate(payload),
                            on_attempt=lambda attempt, raw_text: (out_dir / f"{item_id}.high_level.raw.txt").write_text(
                                raw_text
                            ),
                        )
                    except StrictJsonFailure as exc:
                        raise RuntimeError(f"High-level TOC output invalid for {item_id}: {exc.last_error}") from exc

                artifact = DocumentTocV1(
                    schema_version="document_toc_v1",
                    stage="toc_v1",
                    run_id=paths.run_id,
                    item_id=item_id,
                    created_at=int(time.time()),
                    gateway_url=gateway_url or DEFAULT_GATEWAY_URL,
                    model=model,
                    reasoning_effort=reasoning,
                    temperature=float(temperature),
                    prompt_chunk=str(prompt_chunk_path),
                    prompt_chunk_sha256=chunk_prompt_digest,
                    prompt_high_level=str(prompt_high_level_path) if prompt_high_level_path else None,
                    prompt_high_level_sha256=high_digest if high_digest else None,
                    chunking={
                        "target_chars": int(target_chars),
                        "min_chars": int(min_chars),
                        "max_chars": int(max_chars),
                        "lookback_anchors": int(lookback_anchors),
                        "high_level_chunks": int(high_level_chunks),
                    },
                    chunks=chunk_nodes,
                    high_level=high_level,
                )
                out_path = out_dir / f"{item_id}.json"
                out_path.write_text(artifact.model_dump_json(indent=2))
            except Exception as exc:
                failures.append((item_id, str(exc)))
                (out_dir / f"{item_id}.error.txt").write_text(f"{exc}\n\n{traceback.format_exc()}")

    async def _runner() -> None:
        async with GatewayAgentClient(
            base_url=gateway_url or DEFAULT_GATEWAY_URL,
            timeout=gateway_timeout or 600.0,
        ) as client:
            await asyncio.gather(*(_process_item(item_id, client) for item_id in items))

    asyncio.run(_runner())

    if failures:
        err_path = out_dir / "errors.txt"
        err_path.write_text("\n".join(f"{item}: {msg}" for item, msg in failures))
        raise RuntimeError(f"toc-v1 failed for {len(failures)} items; see {err_path}")

    manifest_path = paths.manifest_path
    if manifest_path.exists():
        update_manifest(
            manifest_path,
            toc_v1_prompt_chunk=str(prompt_chunk_path),
            toc_v1_prompt_chunk_sha256=chunk_prompt_digest,
            toc_v1_prompt_high_level=str(prompt_high_level_path) if prompt_high_level_path else None,
            toc_v1_prompt_high_level_sha256=high_digest if high_digest else None,
            toc_v1_output_subdir=resolved_out_subdir,
            toc_v1_item_ids=items,
        )
