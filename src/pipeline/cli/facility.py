from __future__ import annotations

from pathlib import Path
from typing import Tuple

import click

from ..config import REQUIRED_MODEL, REQUIRED_REASONING
from ..run_id import validate_run_id
from .common import resolve_paths, resolve_selected_item_ids


@click.command(name="facility-fundamentals")
@click.option("--run-id", required=True, callback=validate_run_id)
@click.option(
    "--prompt",
    "prompt_path",
    type=click.Path(exists=True, dir_okay=False),
    default="prompts/facility_fundamentals_v1.txt",
    show_default=True,
    help="Prompt for facility fundamentals extraction (LLM-driven; strict JSON).",
)
@click.option("--base-dir", default=".", show_default=True)
@click.option(
    "--model",
    default=REQUIRED_MODEL,
    show_default=True,
    help="Gateway model for facility fundamentals extraction (enforced).",
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
    help="Gateway client timeout (seconds) for facility fundamentals calls.",
)
@click.option(
    "--concurrency",
    default=3,
    show_default=True,
    type=int,
    help="Number of parallel gateway calls during facility fundamentals extraction.",
)
@click.option(
    "--attempts",
    default=3,
    show_default=True,
    type=int,
    help="Strict JSON/Pydantic attempts per item before failing.",
)
@click.option(
    "--output-subdir",
    default=None,
    help="Optional subfolder under runs/<run_id>/facility_fundamentals/ for outputs (defaults to prompt filename stem).",
)
@click.option(
    "--category",
    "categories",
    multiple=True,
    help="Optional snippet category filter (repeatable). Default: agreement_dates + fundamental + key_date_definitions.",
)
@click.option(
    "--item-id",
    "item_ids",
    multiple=True,
    help="Optional item_id(s) to run facility fundamentals for (defaults to all items in the run manifest).",
)
def facility_fundamentals(
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
    categories: Tuple[str, ...],
    item_ids: Tuple[str, ...],
) -> None:
    """Extract facility fundamentals (dates + facility separation) from v2 retrieval snippets."""

    from ..facility_fundamentals_v1 import run_facility_fundamentals_v1

    paths = resolve_paths(run_id, base_dir, bandwidth=4)
    _manifest, _items, selected_item_ids = resolve_selected_item_ids(paths, item_ids)

    selected_categories = categories if categories else ("agreement_dates", "fundamental", "key_date_definitions")

    run_facility_fundamentals_v1(
        paths,
        selected_item_ids,
        Path(prompt_path),
        categories=selected_categories,
        model=model,
        gateway_url=gateway_url,
        temperature=temperature,
        reasoning=reasoning,
        gateway_timeout=gateway_timeout,
        concurrency=concurrency,
        attempts=attempts,
        output_subdir=output_subdir,
    )


@click.command(name="agreement-metadata")
@click.option("--run-id", required=True, callback=validate_run_id)
@click.option("--prompt", "prompt_path", type=click.Path(exists=True, dir_okay=False), required=True)
@click.option("--base-dir", default=".", show_default=True)
@click.option(
    "--model",
    default=REQUIRED_MODEL,
    show_default=True,
    help="Gateway model for agreement metadata extraction (enforced).",
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
    help="Gateway client timeout (seconds) for agreement metadata calls.",
)
@click.option(
    "--concurrency",
    default=3,
    show_default=True,
    type=int,
    help="Number of parallel gateway calls during agreement metadata extraction.",
)
@click.option(
    "--attempts",
    default=3,
    show_default=True,
    type=int,
    help="Strict JSON/Pydantic attempts per item before failing.",
)
@click.option(
    "--output-subdir",
    default=None,
    help="Optional subfolder under runs/<run_id>/agreement_metadata/ for outputs (defaults to prompt filename stem).",
)
@click.option(
    "--category",
    "categories",
    multiple=True,
    help="Optional snippet category filter (repeatable). Default: metadata + fundamental.",
)
@click.option(
    "--item-id",
    "item_ids",
    multiple=True,
    help="Optional item_id(s) to run agreement metadata for (defaults to all items in the run manifest).",
)
def agreement_metadata(
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
    categories: Tuple[str, ...],
    item_ids: Tuple[str, ...],
) -> None:
    """Extract parties/roles + facility headline terms from v2 retrieval snippets."""

    from ..agreement_metadata_v1 import run_agreement_metadata_v1

    paths = resolve_paths(run_id, base_dir, bandwidth=4)
    _manifest, _items, selected_item_ids = resolve_selected_item_ids(paths, item_ids)

    selected_categories = categories if categories else ("metadata", "fundamental")

    run_agreement_metadata_v1(
        paths,
        selected_item_ids,
        Path(prompt_path),
        categories=selected_categories,
        model=model,
        gateway_url=gateway_url,
        temperature=temperature,
        reasoning=reasoning,
        gateway_timeout=gateway_timeout,
        concurrency=concurrency,
        attempts=attempts,
        output_subdir=output_subdir,
    )


@click.command(name="analysis-export")
@click.option("--run-id", required=True, callback=validate_run_id)
@click.option("--base-dir", default=".", show_default=True)
@click.option(
    "--agreement-metadata-subdir",
    required=True,
    help="Subfolder under runs/<run_id>/agreement_metadata/ containing <item_id>.json outputs.",
)
@click.option(
    "--definitions-v2-subdir",
    required=True,
    help="Subfolder under runs/<run_id>/definitions_v2/ containing per-term __definition.json outputs.",
)
@click.option(
    "--output-subdir",
    default="analysis_export_v1",
    show_default=True,
    help="Output subfolder under runs/<run_id>/analysis_export/.",
)
@click.option(
    "--item-id",
    "item_ids",
    multiple=True,
    help="Optional item_id(s) to export (defaults to all items in the run manifest).",
)
def analysis_export(
    run_id: str,
    base_dir: str,
    agreement_metadata_subdir: str,
    definitions_v2_subdir: str,
    output_subdir: str,
    item_ids: Tuple[str, ...],
) -> None:
    """Export a single analysis-ready JSON record per agreement."""

    from ..analysis_export_v1 import run_analysis_export_v1

    paths = resolve_paths(run_id, base_dir, bandwidth=4)
    _manifest, _items, selected_item_ids = resolve_selected_item_ids(paths, item_ids)

    run_analysis_export_v1(
        paths,
        selected_item_ids,
        agreement_metadata_subdir=agreement_metadata_subdir,
        definitions_v2_subdir=definitions_v2_subdir,
        output_subdir=output_subdir,
    )

