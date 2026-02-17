from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import click

from pipeline.core.config import REQUIRED_MODEL, REQUIRED_REASONING
from pipeline.evidence.indexing_v2 import run_indexing_v2
from pipeline.evidence.retrieval_v2 import render_snippets_v2
from pipeline.core.run_id import validate_run_id
from pipeline.extract.structured_v2 import run_structured_v2
from .common import resolve_paths, resolve_selected_item_ids


@click.command(name="index-v2")
@click.option("--run-id", required=True, callback=validate_run_id)
@click.option("--prompt", "prompt_path", type=click.Path(exists=True, dir_okay=False), required=True)
@click.option("--base-dir", default=".", show_default=True)
@click.option(
    "--model",
    default=REQUIRED_MODEL,
    show_default=True,
    help="Gateway model (enforced).",
)
@click.option("--gateway-url", default=None, help="Gateway base URL; defaults to $GATEWAY_URL.")
@click.option("--temperature", default=0.0, show_default=True, type=float)
@click.option(
    "--reasoning",
    default=REQUIRED_REASONING,
    type=click.Choice(["light", "medium", "heavy"], case_sensitive=False),
    show_default=True,
    help="Reasoning effort (enforced).",
)
@click.option(
    "--gateway-timeout",
    default=600.0,
    show_default=True,
    type=float,
    help="Gateway client timeout (seconds) for indexing v2 calls.",
)
@click.option(
    "--concurrency",
    default=3,
    show_default=True,
    type=int,
    help="Number of parallel gateway calls during indexing v2 (only when --model is set).",
)
@click.option(
    "--attempts",
    default=3,
    show_default=True,
    type=int,
    help="Strict JSON/Pydantic attempts per item before failing.",
)
@click.option(
    "--skip-existing",
    is_flag=True,
    help="Skip items with existing runs/<run_id>/indexing_v2/<item_id>_anchors.json outputs.",
)
@click.option(
    "--item-id",
    "item_ids",
    multiple=True,
    help="Optional item_id(s) to index (defaults to all items in the run manifest).",
)
def index_v2(
    run_id: str,
    prompt_path: str,
    base_dir: str,
    model: str,
    gateway_url: str,
    temperature: float,
    reasoning: str | None,
    gateway_timeout: float,
    concurrency: int,
    attempts: int,
    skip_existing: bool,
    item_ids: Tuple[str, ...],
) -> None:
    """Run anchor indexing via agent-gateway (v2 schema)."""

    paths = resolve_paths(run_id, base_dir, bandwidth=4)
    _manifest, _items, selected_item_ids = resolve_selected_item_ids(paths, item_ids)

    run_indexing_v2(
        paths,
        selected_item_ids,
        Path(prompt_path),
        model=model,
        gateway_url=gateway_url,
        temperature=temperature,
        reasoning=reasoning,
        gateway_timeout=gateway_timeout,
        concurrency=concurrency,
        attempts=attempts,
        skip_existing=bool(skip_existing),
    )


@click.command(name="retrieve-v2")
@click.option("--run-id", required=True, callback=validate_run_id)
@click.option("--bandwidth", default=400, show_default=True, type=int)
@click.option("--base-dir", default=".", show_default=True)
@click.option(
    "--item-id",
    "item_ids",
    multiple=True,
    help="Optional item_id(s) to retrieve snippets for (defaults to all items in the run manifest).",
)
def retrieve_v2(run_id: str, bandwidth: int, base_dir: str, item_ids: Tuple[str, ...]) -> None:
    """Render snippets around anchors using indexing v2 outputs."""

    paths = resolve_paths(run_id, base_dir, bandwidth=bandwidth)
    _manifest, _items, selected_item_ids = resolve_selected_item_ids(paths, item_ids)
    render_snippets_v2(paths, selected_item_ids, bandwidth=bandwidth)


@click.command(name="structured-v2")
@click.option("--run-id", required=True, callback=validate_run_id)
@click.option("--prompt", "prompt_path", type=click.Path(exists=True, dir_okay=False), required=True)
@click.option("--base-dir", default=".", show_default=True)
@click.option(
    "--model",
    default=REQUIRED_MODEL,
    show_default=True,
    help="Gateway model for structured extraction (enforced).",
)
@click.option("--gateway-url", default=None, help="Gateway base URL; defaults to $GATEWAY_URL.")
@click.option("--temperature", default=0.0, show_default=True, type=float)
@click.option(
    "--reasoning",
    default=REQUIRED_REASONING,
    type=click.Choice(["light", "medium", "heavy"], case_sensitive=False),
    show_default=True,
    help="Reasoning effort (enforced).",
)
@click.option(
    "--gateway-timeout",
    default=600.0,
    show_default=True,
    type=float,
    help="Gateway client timeout (seconds) for structured calls.",
)
@click.option(
    "--concurrency",
    default=3,
    show_default=True,
    type=int,
    help="Number of parallel gateway calls during structured extraction.",
)
@click.option(
    "--attempts",
    default=3,
    show_default=True,
    type=int,
    help="Strict JSON attempts per item before failing.",
)
@click.option(
    "--output-subdir",
    default=None,
    help="Optional subfolder under runs/<run_id>/llm_qa/ for outputs (defaults to prompt filename stem).",
)
@click.option(
    "--contract",
    required=True,
    type=click.Choice(["pricing_structured_v2", "covenant_simple_v2"], case_sensitive=False),
    help="LLM output contract to enforce (no backwards compatibility).",
)
@click.option(
    "--category",
    "categories",
    multiple=True,
    help="Optional category filter (repeatable). Example: --category financial_covenant",
)
@click.option(
    "--item-id",
    "item_ids",
    multiple=True,
    help="Optional item_id(s) to run structured extraction for (defaults to all items in the run manifest).",
)
@click.option(
    "--skip-existing",
    is_flag=True,
    help="Skip items with existing runs/<run_id>/llm_qa/<output_subdir>/<item_id>.json outputs.",
)
@click.option(
    "--allow-empty-after-filter",
    is_flag=True,
    help=(
        "When category filtering yields no snippets for an item, write an empty structured artifact "
        "instead of failing the run."
    ),
)
def structured_v2(
    run_id: str,
    prompt_path: str,
    base_dir: str,
    model: str | None,
    gateway_url: str | None,
    temperature: float,
    reasoning: str | None,
    gateway_timeout: float,
    concurrency: int,
    attempts: int,
    output_subdir: str | None,
    contract: str,
    categories: Tuple[str, ...],
    item_ids: Tuple[str, ...],
    skip_existing: bool,
    allow_empty_after_filter: bool,
) -> None:
    """Structured extraction over v2 retrieval snippets (strict JSON)."""

    paths = resolve_paths(run_id, base_dir, bandwidth=4)
    _manifest, _items, selected_item_ids = resolve_selected_item_ids(paths, item_ids)
    run_structured_v2(
        paths,
        selected_item_ids,
        Path(prompt_path),
        contract=contract.lower(),  # click Choice returns canonical casing but keep it explicit
        model=model,
        gateway_url=gateway_url,
        temperature=temperature,
        reasoning=reasoning,
        gateway_timeout=gateway_timeout,
        concurrency=concurrency,
        attempts=attempts,
        output_subdir=output_subdir,
        categories=categories,
        skip_existing=bool(skip_existing),
        allow_empty_after_filter=bool(allow_empty_after_filter),
    )
