from __future__ import annotations

import asyncio
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

from .config import Paths, prompt_hash, update_manifest
from .utils import assert_exists, prompt_view_path


DEFAULT_MODEL = os.getenv("INDEXING_MODEL", "openai:gpt-4o-mini")
DEFAULT_GATEWAY_URL = os.getenv("GATEWAY_URL", "http://127.0.0.1:8000")


class IndexingGatewayUnavailable(RuntimeError):
    """Raised when the gateway client cannot be imported or called."""


def _ensure_gateway_client_sync() -> Any:
    """Return the sync helper complete_response_sync from agent-gateway."""

    try:
        from gateway.client import complete_response_sync  # type: ignore

        return complete_response_sync
    except ModuleNotFoundError:
        root = Path(__file__).resolve().parents[2] / "agent-gateway" / "src"
        if root.exists() and str(root) not in sys.path:
            sys.path.append(str(root))
            try:
                from gateway.client import complete_response_sync  # type: ignore

                return complete_response_sync
            except ModuleNotFoundError as exc:  # pragma: no cover
                raise IndexingGatewayUnavailable(
                    "agent-gateway present but import failed; ensure dependencies are installed"
                ) from exc
        raise IndexingGatewayUnavailable(
            "agent-gateway submodule not available; run `git submodule update --init --recursive`"
        )


def _ensure_gateway_client_async() -> Any:
    """Return the async GatewayAgentClient from agent-gateway."""

    try:
        from gateway.client import GatewayAgentClient  # type: ignore

        return GatewayAgentClient
    except ModuleNotFoundError:
        root = Path(__file__).resolve().parents[2] / "agent-gateway" / "src"
        if root.exists() and str(root) not in sys.path:
            sys.path.append(str(root))
            try:
                from gateway.client import GatewayAgentClient  # type: ignore

                return GatewayAgentClient
            except ModuleNotFoundError as exc:  # pragma: no cover
                raise IndexingGatewayUnavailable(
                    "agent-gateway present but import failed; ensure dependencies are installed"
                ) from exc
        raise IndexingGatewayUnavailable(
            "agent-gateway submodule not available; run `git submodule update --init --recursive`"
        )


def _anchor_catalog(paths: Paths, item_id: str) -> Dict[str, Dict[str, Any]]:
    """Load anchor spans from anchors.tsv for an item.

    Returns a dict keyed by anchor_id with start/end/type/label and sort order.
    """

    tsv_candidates = [
        paths.normalized_dir / item_id / "anchors.tsv",
        paths.legacy_prompt_views_dir / item_id / "anchors.tsv",
    ]
    tsv_path = next((p for p in tsv_candidates if p.exists()), None)
    if not tsv_path:
        raise FileNotFoundError(f"anchors.tsv not found for {item_id}")

    catalog: Dict[str, Dict[str, Any]] = {}
    with tsv_path.open() as fh:
        header_skipped = False
        order = 0
        for line in fh:
            if not header_skipped:
                header_skipped = True
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 5:
                continue
            anchor_id, anchor_type, start, end, label = parts[:5]
            try:
                start_i, end_i = int(start), int(end)
            except ValueError:
                continue
            catalog[anchor_id] = {
                "anchor_id": anchor_id,
                "type": anchor_type,
                "label": label,
                "start": start_i,
                "end": end_i,
                "order": order,
            }
            order += 1
    if not catalog:
        raise RuntimeError(f"No anchors parsed for {item_id}")
    return catalog


def _anchor_texts(paths: Paths, item_id: str) -> Dict[str, str]:
    """Map anchor_id -> text block from prompt_view_annotated.txt."""

    candidates = [
        paths.normalized_dir / item_id / "prompt_view_annotated.txt",
        paths.legacy_prompt_views_dir / item_id / "prompt_view_annotated.txt",
    ]
    annotated = next((p for p in candidates if p.exists()), None)
    if not annotated:
        raise FileNotFoundError(f"prompt_view_annotated.txt missing for {item_id}")

    text = annotated.read_text()
    out: Dict[str, List[str]] = {}
    current_id: str | None = None
    current_lines: List[str] = []
    anchor_re = re.compile(r"^\[\[(A\d{4,})\]\]")

    for line in text.splitlines():
        match = anchor_re.match(line.strip())
        if match:
            if current_id:
                out[current_id] = current_lines[:]
            current_id = match.group(1)
            current_lines = []
            continue
        if current_id:
            current_lines.append(line)
    if current_id:
        out[current_id] = current_lines[:]

    return {aid: "\n".join(lines).strip() for aid, lines in out.items()}


def _render_prompt(template: str, anchor_texts: Dict[str, str]) -> str:
    """Blend the user template with a list of anchor snippets."""

    pairs = list(anchor_texts.items())
    formatted_snippets = []
    for aid, snippet in pairs:
        short = snippet.strip()
        if len(short) > 400:
            short = short[:400] + "..."
        formatted_snippets.append(f"{aid}: {short}")

    doc_block = "\n\n".join(formatted_snippets)
    template = template.strip()
    if "{document}" in template:
        return template.replace("{document}", doc_block)
    if "{anchors}" in template:
        return template.replace("{anchors}", doc_block)
    return (
        f"{template}\n\nYou will be given anchors identified by ID and text."
        " Select only those that are relevant to credit agreement economics (rates, fees,"
        " grids, maturity, principal amounts, collateral, covenants, definitions, prepayment,"
        " margin adjustments). Respond strictly with JSON: {\"anchors\": [{\"anchor_id\":"
        " \"A0001\", \"label\": \"short label\"}, ...]}\n\nAnchors:\n\n"
        f"{doc_block}"
    )


def _parse_anchor_selection(raw_text: str) -> List[Tuple[str, str | None]]:
    """Parse the model response into a list of (anchor_id, label)."""

    def _load_json(text: str) -> Any:
        try:
            return json.loads(text)
        except Exception:
            match = re.search(r"\{.*\}", text, flags=re.DOTALL)
            if match:
                return json.loads(match.group(0))
            raise

    payload = _load_json(raw_text)
    # Case 1: expected {"anchors": [...]}
    if isinstance(payload, dict) and "anchors" in payload:
        anchors_raw = payload.get("anchors")
    else:
        anchors_raw = payload

    # Case 2: bucketed keys (fundamental/pricing/financial_covenant)
    bucket_keys = (
        "fundamental_anchors",
        "pricing_anchors",
        "financial_covenant_anchors",
    )
    if isinstance(payload, dict) and any(k in payload for k in bucket_keys):
        results: List[Tuple[str, str | None]] = []
        for key in bucket_keys:
            anchors_list = payload.get(key)
            if not isinstance(anchors_list, list):
                continue
            for aid in anchors_list:
                if isinstance(aid, str):
                    results.append((aid, key.replace("_anchors", "")))
        if results:
            return results

    if not isinstance(anchors_raw, list):
        raise ValueError("Model response missing 'anchors' array")

    results: List[Tuple[str, str | None]] = []
    for entry in anchors_raw:
        if isinstance(entry, str):
            results.append((entry, None))
            continue
        if not isinstance(entry, dict):
            continue
        aid = entry.get("anchor_id") or entry.get("id")
        label = entry.get("label") or entry.get("reason") or entry.get("name")
        if aid:
            results.append((aid, label))
    return results


def _select_anchors_via_gateway(
    *,
    prompt: str,
    model: str,
    gateway_url: str,
    temperature: float,
    reasoning: str | None,
    timeout: float | None = None,
) -> List[Tuple[str, str | None]]:
    complete_response_sync = _ensure_gateway_client_sync()
    reasoning_payload = None
    if reasoning:
        reasoning_payload = {"effort": reasoning}
    raw = complete_response_sync(
        model=model,
        prompt=prompt,
        base_url=gateway_url,
        temperature=temperature,
        reasoning=reasoning_payload,
        timeout=timeout,
    )
    return _parse_anchor_selection(raw)


async def _select_anchors_via_gateway_async(
    *,
    prompt: str,
    model: str,
    gateway_url: str,
    temperature: float,
    reasoning: str | None,
    timeout: float | None = None,
    client: Any,
) -> List[Tuple[str, str | None]]:
    reasoning_payload = None
    if reasoning:
        reasoning_payload = {"effort": reasoning}

    result = await client.complete_response(
        model=model,
        input_messages=[{"role": "user", "content": prompt}],
        reasoning=reasoning_payload,
        temperature=temperature,
        max_output_tokens=None,
        metadata=None,
    )
    raw = result.get("text") if isinstance(result, dict) else result
    return _parse_anchor_selection(raw or "")


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
    """Index anchors via the agent-gateway (if available) or fall back to all anchors.

    - Reads candidate anchors from normalization outputs (anchors.tsv + annotated view).
    - Calls the gateway with a user prompt to pick/label anchors.
    - Writes `{item_id}_anchors.json` expected by retrieval.
    """

    assert_exists(prompt_path, message=f"Indexing prompt not found: {prompt_path}")
    out_dir = paths.indexing_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    prompt_digest = prompt_hash(prompt_path)
    prompt_template = prompt_path.read_text()

    item_list = list(item_ids)

    async def _run_async(items: List[str]) -> None:
        GatewayAgentClient = _ensure_gateway_client_async()
        sem = asyncio.Semaphore(max(1, concurrency))
        async with GatewayAgentClient(
            base_url=gateway_url or DEFAULT_GATEWAY_URL, timeout=gateway_timeout or 30.0
        ) as client:

            async def _process(item_id: str) -> None:
                async with sem:
                    anchor_texts = _anchor_texts(paths, item_id)
                    rendered_prompt = _render_prompt(prompt_template, anchor_texts)

                    result = await client.complete_response(
                        model=model or DEFAULT_MODEL,
                        input_messages=[{"role": "user", "content": rendered_prompt}],
                        reasoning={"effort": reasoning} if reasoning else None,
                        temperature=temperature,
                        max_output_tokens=None,
                        metadata=None,
                    )
                    raw_text = result.get("text") if isinstance(result, dict) else str(result)
                    # Persist raw LLM output verbatim; no post-processing
                    raw_path = out_dir / f"{item_id}_anchors.txt"
                    raw_path.write_text(raw_text)

            await asyncio.gather(*(_process(i) for i in items))

    def _run_sync(items: List[str]) -> None:
        for item_id in items:
            anchor_texts = _anchor_texts(paths, item_id)
            rendered_prompt = _render_prompt(prompt_template, anchor_texts)
            if model:
                complete_response_sync = _ensure_gateway_client_sync()
                raw_text = complete_response_sync(
                    model=model or DEFAULT_MODEL,
                    prompt=rendered_prompt,
                    base_url=gateway_url or DEFAULT_GATEWAY_URL,
                    temperature=temperature,
                    reasoning={"effort": reasoning} if reasoning else None,
                    timeout=gateway_timeout,
                )
            else:
                raw_text = ""

            out_path = out_dir / f"{item_id}_anchors.txt"
            out_path.write_text(str(raw_text))

    if model:
        asyncio.run(_run_async(item_list))
    else:
        _run_sync(item_list)

    manifest_path = paths.manifest_path
    if manifest_path.exists():
        update_manifest(
            manifest_path,
            indexing_prompt=str(prompt_path),
            indexing_prompt_sha256=prompt_digest,
        )
