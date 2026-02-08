from __future__ import annotations

from pathlib import Path
from typing import Tuple

import click

from pipeline.core.config import REQUIRED_REASONING
from pipeline.core.run_id import validate_run_id
from .common import resolve_paths, resolve_selected_item_ids


@click.command(name="toc-v1")
@click.option("--run-id", required=True, callback=validate_run_id)
@click.option(
    "--prompt-chunk",
    "prompt_chunk_path",
    type=click.Path(exists=True, dir_okay=False),
    required=True,
    help="Chunk-level TOC prompt (generates a title/summary/topics per chunk).",
)
@click.option(
    "--prompt-high-level",
    "prompt_high_level_path",
    type=click.Path(exists=True, dir_okay=False),
    required=False,
    help="Optional prompt to build a high-level doc map from the first few chunk TOC entries.",
)
@click.option("--base-dir", default=".", show_default=True)
@click.option("--gateway-url", default=None, help="Gateway base URL; defaults to $GATEWAY_URL.")
@click.option("--temperature", default=0.0, show_default=True, type=float)
@click.option(
    "--reasoning",
    default=REQUIRED_REASONING,
    type=click.Choice(["light", "medium", "heavy"], case_sensitive=False),
    show_default=True,
    help="Reasoning effort (enforced).",
)
@click.option("--gateway-timeout", default=600.0, show_default=True, type=float)
@click.option("--concurrency", default=3, show_default=True, type=int)
@click.option("--attempts", default=3, show_default=True, type=int)
@click.option("--output-subdir", default=None, help="Subfolder under runs/<run_id>/toc_v1/ for outputs.")
@click.option("--target-chars", default=14000, show_default=True, type=int)
@click.option("--min-chars", default=8000, show_default=True, type=int)
@click.option("--max-chars", default=22000, show_default=True, type=int)
@click.option("--lookback-anchors", default=40, show_default=True, type=int)
@click.option("--high-level-chunks", default=5, show_default=True, type=int)
@click.option(
    "--item-id",
    "item_ids",
    multiple=True,
    help="Optional item_id(s) to run toc-v1 for (defaults to all items in the run manifest).",
)
def toc_v1(
    run_id: str,
    prompt_chunk_path: str,
    prompt_high_level_path: str | None,
    base_dir: str,
    gateway_url: str | None,
    temperature: float,
    reasoning: str | None,
    gateway_timeout: float,
    concurrency: int,
    attempts: int,
    output_subdir: str | None,
    target_chars: int,
    min_chars: int,
    max_chars: int,
    lookback_anchors: int,
    high_level_chunks: int,
    item_ids: Tuple[str, ...],
) -> None:
    """Build a TOC via adaptive chunking + LLM summaries (no embeddings)."""

    from pipeline.evidence.toc_v1 import run_toc_v1

    paths = resolve_paths(run_id, base_dir, bandwidth=4)
    _manifest, _items, selected_item_ids = resolve_selected_item_ids(paths, item_ids)

    run_toc_v1(
        paths,
        selected_item_ids,
        prompt_chunk_path=Path(prompt_chunk_path),
        prompt_high_level_path=Path(prompt_high_level_path) if prompt_high_level_path else None,
        output_subdir=output_subdir,
        target_chars=target_chars,
        min_chars=min_chars,
        max_chars=max_chars,
        lookback_anchors=lookback_anchors,
        high_level_chunks=high_level_chunks,
        gateway_url=gateway_url,
        temperature=temperature,
        reasoning=reasoning,
        gateway_timeout=gateway_timeout,
        concurrency=concurrency,
        attempts=attempts,
    )

