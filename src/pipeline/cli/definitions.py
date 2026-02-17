from __future__ import annotations

from pathlib import Path
from typing import Tuple

import click

from pipeline.core.config import REQUIRED_MODEL, REQUIRED_REASONING
from pipeline.core.run_id import validate_run_id
from .common import resolve_paths, resolve_selected_item_ids


@click.command(name="definitions-compiler-v1")
@click.option("--run-id", required=True, callback=validate_run_id)
@click.option("--qa-subdir", required=True, help="llm_qa subdir with dg outputs.")
@click.option(
    "--compiler-prompt",
    "compiler_prompt_path",
    type=click.Path(exists=True, dir_okay=False),
    required=True,
)
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
    help="Gateway client timeout (seconds) for compiler calls.",
)
@click.option(
    "--concurrency",
    default=2,
    show_default=True,
    type=int,
    help="Number of parallel gateway calls during compilation (metric-level).",
)
@click.option(
    "--attempts",
    default=3,
    show_default=True,
    type=int,
    help="Retry attempts per metric when the gateway returns empty/throws.",
)
@click.option(
    "--output-subdir",
    default="compiled_v1",
    show_default=True,
    help="Subfolder under runs/<run_id>/definitions_compiler_v1/ for outputs.",
)
@click.option(
    "--item-id",
    "item_ids",
    multiple=True,
    help="Optional item_id(s) to run compiler for (defaults to all items in the run manifest).",
)
def definitions_compiler_v1(
    run_id: str,
    qa_subdir: str,
    compiler_prompt_path: str,
    base_dir: str,
    model: str | None,
    gateway_url: str | None,
    temperature: float,
    reasoning: str | None,
    gateway_timeout: float,
    concurrency: int,
    attempts: int,
    output_subdir: str,
    item_ids: Tuple[str, ...],
) -> None:
    """Compile metric definitions (verbatim + AST + dependency terms) from canonical text and indexing hints."""

    from pipeline.compile.definitions_compiler_v1 import run_definitions_compiler_v1

    paths = resolve_paths(run_id, base_dir, bandwidth=4)
    _manifest, _items, selected_item_ids = resolve_selected_item_ids(paths, item_ids)

    run_definitions_compiler_v1(
        paths,
        selected_item_ids,
        qa_subdir=qa_subdir,
        compiler_prompt_path=Path(compiler_prompt_path),
        model=model,
        gateway_url=gateway_url,
        temperature=temperature,
        reasoning=reasoning,
        gateway_timeout=gateway_timeout,
        concurrency=concurrency,
        output_subdir=output_subdir,
        attempts=attempts,
    )


@click.command(name="blocking-terms-compiler-v1")
@click.option("--run-id", required=True, callback=validate_run_id)
@click.option(
    "--metrics-output-subdir",
    required=True,
    help="definitions_compiler_v1 output subdir to read (e.g., compiled_v1 or compiled_metrics_v1_compustat_v3_full).",
)
@click.option(
    "--term-compiler-prompt",
    "term_compiler_prompt_path",
    type=click.Path(exists=True, dir_okay=False),
    required=True,
)
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
    help="Gateway client timeout (seconds) for term compiler calls.",
)
@click.option(
    "--concurrency",
    default=2,
    show_default=True,
    type=int,
    help="Number of parallel gateway calls during term compilation.",
)
@click.option(
    "--attempts",
    default=3,
    show_default=True,
    type=int,
    help="Retry attempts per blocking term when the gateway returns empty/throws.",
)
@click.option(
    "--max-depth",
    default=1,
    show_default=True,
    type=int,
    help="Max recursion depth over unresolved_dependencies (1 = only initial blocking terms).",
)
@click.option(
    "--max-terms",
    default=200,
    show_default=True,
    type=int,
    help="Max unique terms to compile per item (prevents runaway recursion).",
)
@click.option(
    "--output-subdir",
    default="blocking_terms_v1",
    show_default=True,
    help="Subfolder under runs/<run_id>/blocking_terms_compiler_v1/ for outputs.",
)
@click.option(
    "--item-id",
    "item_ids",
    multiple=True,
    help="Optional item_id(s) to compile blocking terms for (defaults to all items in the run manifest).",
)
def blocking_terms_compiler_v1(
    run_id: str,
    metrics_output_subdir: str,
    term_compiler_prompt_path: str,
    base_dir: str,
    model: str | None,
    gateway_url: str | None,
    temperature: float,
    reasoning: str | None,
    gateway_timeout: float,
    concurrency: int,
    attempts: int,
    max_depth: int,
    max_terms: int,
    output_subdir: str,
    item_ids: Tuple[str, ...],
) -> None:
    """Resolve dependency terms recursively from metric definitions using whole-document contexts."""

    from pipeline.compile.blocking_terms_compiler_v1 import run_blocking_terms_compiler_v1

    paths = resolve_paths(run_id, base_dir, bandwidth=4)
    _manifest, _items, selected_item_ids = resolve_selected_item_ids(paths, item_ids)

    run_blocking_terms_compiler_v1(
        paths,
        selected_item_ids,
        metrics_output_subdir=metrics_output_subdir,
        term_compiler_prompt_path=Path(term_compiler_prompt_path),
        model=model,
        gateway_url=gateway_url,
        temperature=temperature,
        reasoning=reasoning,
        gateway_timeout=gateway_timeout,
        concurrency=concurrency,
        attempts=attempts,
        output_subdir=output_subdir,
        max_depth=max_depth,
        max_terms=max_terms,
    )


@click.command(name="compustat-overlay-v1")
@click.option("--run-id", required=True, callback=validate_run_id)
@click.option(
    "--overlay-prompt",
    "overlay_prompt_path",
    type=click.Path(exists=True, dir_okay=False),
    required=True,
)
@click.option(
    "--compustat-allowlist",
    "compustat_allowlist_path",
    type=click.Path(exists=True, dir_okay=False),
    default=None,
    help=(
        "Path to Compustat variable allowlist JSON (defaults to datasets/compustat_allowlist_quarterly_v1.json). "
        "See schema in that file."
    ),
)
@click.option(
    "--metrics-output-subdir",
    default=None,
    help="definitions_compiler_v1 output subdir to read (item-level __compiled.json).",
)
@click.option(
    "--blocking-terms-output-subdir",
    default=None,
    help="blocking_terms_compiler_v1 output subdir to read (item-level __compiled.json).",
)
@click.option("--base-dir", default=".", show_default=True)
@click.option("--gateway-url", default=None, help="Gateway base URL; defaults to $GATEWAY_URL.")
@click.option("--temperature", default=0.0, show_default=True, type=float)
@click.option(
    "--gateway-timeout",
    default=600.0,
    show_default=True,
    type=float,
    help="Gateway client timeout (seconds) for overlay calls.",
)
@click.option(
    "--concurrency",
    default=4,
    show_default=True,
    type=int,
    help="Number of parallel gateway calls during overlay.",
)
@click.option(
    "--attempts",
    default=3,
    show_default=True,
    type=int,
    help="Retry attempts per term overlay when the gateway returns empty/throws/invalid JSON.",
)
@click.option(
    "--output-subdir",
    default="compustat_overlay_v1",
    show_default=True,
    help="Subfolder under runs/<run_id>/compustat_overlay_v1/ for outputs.",
)
@click.option(
    "--item-id",
    "item_ids",
    multiple=True,
    help="Optional item_id(s) to run overlay for (defaults to all items in the run manifest).",
)
def compustat_overlay_v1(
    run_id: str,
    overlay_prompt_path: str,
    compustat_allowlist_path: str | None,
    metrics_output_subdir: str | None,
    blocking_terms_output_subdir: str | None,
    base_dir: str,
    gateway_url: str | None,
    temperature: float,
    gateway_timeout: float,
    concurrency: int,
    attempts: int,
    output_subdir: str,
    item_ids: Tuple[str, ...],
) -> None:
    """Map extracted v2 AST definitions to Compustat formulas (overlay stage)."""

    from pipeline.compile.compustat_overlay_v1 import run_compustat_overlay_v1

    if not metrics_output_subdir and not blocking_terms_output_subdir:
        raise click.UsageError("Must provide at least one of --metrics-output-subdir or --blocking-terms-output-subdir.")

    paths = resolve_paths(run_id, base_dir, bandwidth=4)
    _manifest, _items, selected_item_ids = resolve_selected_item_ids(paths, item_ids)

    run_compustat_overlay_v1(
        paths,
        selected_item_ids,
        overlay_prompt_path=Path(overlay_prompt_path),
        allowlist_path=Path(compustat_allowlist_path) if compustat_allowlist_path else None,
        output_subdir=output_subdir,
        metrics_output_subdir=metrics_output_subdir,
        blocking_terms_output_subdir=blocking_terms_output_subdir,
        gateway_url=gateway_url,
        temperature=temperature,
        gateway_timeout=gateway_timeout,
        concurrency=concurrency,
        attempts=attempts,
    )


@click.command(name="compustat-overlay-eval-v1")
@click.option("--run-id", required=True, callback=validate_run_id)
@click.option(
    "--overlay-output-subdir",
    default=None,
    help=(
        "compustat_overlay_v1 subdir to read. "
        "Defaults to manifest.compustat_overlay_v1_output_subdir when omitted."
    ),
)
@click.option(
    "--values-path",
    type=click.Path(exists=True, dir_okay=False),
    default=None,
    help=(
        "Path to company-level Compustat values JSON "
        "(schema_version=compustat_company_values_v1)."
    ),
)
@click.option(
    "--synthetic-values/--no-synthetic-values",
    default=False,
    show_default=True,
    help=(
        "Use deterministic synthetic values per item/variable instead of an input values file. "
        "Intended for prototype testing only."
    ),
)
@click.option("--base-dir", default=".", show_default=True)
@click.option(
    "--output-subdir",
    default="compustat_overlay_eval_v1",
    show_default=True,
    help="Subfolder under runs/<run_id>/compustat_overlay_eval_v1/ for outputs.",
)
@click.option(
    "--item-id",
    "item_ids",
    multiple=True,
    help="Optional item_id(s) to evaluate (defaults to all items in the run manifest).",
)
def compustat_overlay_eval_v1(
    run_id: str,
    overlay_output_subdir: str | None,
    values_path: str | None,
    synthetic_values: bool,
    base_dir: str,
    output_subdir: str,
    item_ids: Tuple[str, ...],
) -> None:
    """Evaluate Compustat overlay formulas by substituting company values."""

    from pipeline.compile.compustat_overlay_eval_v1 import run_compustat_overlay_eval_v1

    if bool(values_path) == bool(synthetic_values):
        raise click.UsageError(
            "Provide exactly one value source: --values-path OR --synthetic-values."
        )

    paths = resolve_paths(run_id, base_dir, bandwidth=4)
    manifest, _items, selected_item_ids = resolve_selected_item_ids(paths, item_ids)

    selected_overlay_subdir = (
        overlay_output_subdir
        or (manifest.get("compustat_overlay_v1_output_subdir") if isinstance(manifest, dict) else None)
    )
    if not isinstance(selected_overlay_subdir, str) or not selected_overlay_subdir.strip():
        raise click.UsageError(
            "Could not resolve overlay subdir. Pass --overlay-output-subdir or ensure "
            "manifest has compustat_overlay_v1_output_subdir."
        )

    run_compustat_overlay_eval_v1(
        paths,
        selected_item_ids,
        overlay_output_subdir=selected_overlay_subdir.strip(),
        output_subdir=output_subdir,
        values_path=Path(values_path) if values_path else None,
        synthetic_values=synthetic_values,
    )
