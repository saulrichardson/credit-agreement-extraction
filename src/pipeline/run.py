from __future__ import annotations

from pathlib import Path
from typing import Optional

import click

from .config import FilterSpec, RunConfig, REQUIRED_MODEL, REQUIRED_REASONING
from .filters import load_filter_spec, load_doc_filter
from .ingest import ingest_tarballs
from .normalize import build_prompt_views
from .indexing import run_indexing
from .retrieval import render_snippets
from .structured import run_structured
from .contract_pricing import run_contract_pricing
from .validation import run_validation
from .utils import load_manifest, manifest_items, read_accessions_file
from .run_id import validate_run_id


def resolve_run_config(run_id: str, base_dir: str, workers: int, bandwidth: int) -> RunConfig:
    # Validate even when called programmatically (outside Click).
    run_id_validated = validate_run_id(None, None, run_id)
    return RunConfig(
        run_id=run_id_validated,
        base_dir=Path(base_dir),
        workers=workers,
        bandwidth=bandwidth,
    )


def _resolve_paths(run_id: str, base_dir: str, bandwidth: int) -> RunConfig:
    """Convenience helper to construct RunConfig and paths in one place."""
    return resolve_run_config(run_id, base_dir, workers=4, bandwidth=bandwidth)


def _load_accessions_and_filters(filters_path: Optional[str], accessions_file: Optional[str]):
    """Shared validation for ingest/all commands."""
    accessions = read_accessions_file(Path(accessions_file)) if accessions_file else None

    if not accessions_file and not filters_path:
        raise click.UsageError("Provide either accessions-file or filters to avoid scanning everything.")

    if filters_path:
        spec = load_filter_spec(Path(filters_path))
    else:
        # Default to accepting all documents when no filter is supplied.
        spec = FilterSpec(doc_filter_path="pipeline.filters:keep_all")

    doc_filter = load_doc_filter(spec)
    return accessions, spec, doc_filter


@click.group()
def cli():
    """Run-scoped EX-10 processing pipeline."""


@cli.command()
@click.option(
    "--run-id",
    required=True,
    callback=validate_run_id,
    help="Run identifier (writes under runs/<run_id>/; reruns overwrite artifacts).",
)
@click.option("--tarball", multiple=True, type=click.Path(exists=True, dir_okay=False), required=True)
@click.option(
    "--filters",
    "filters_path",
    type=click.Path(exists=True, dir_okay=False),
    required=False,
    help="Filter spec (JSON/YAML with doc_filter_path and optional doc_filter_kwargs). Defaults to keep_all when omitted.",
)
@click.option("--accessions-file", type=click.Path(exists=True, dir_okay=False), required=False)
@click.option("--base-dir", default=".", show_default=True)
def ingest(run_id: str, tarball, filters_path: Optional[str], accessions_file: Optional[str], base_dir: str):
    """Extract EX-10 HTMLs for selected accessions from tarballs."""
    rc = _resolve_paths(run_id, base_dir, bandwidth=4)
    paths = rc.paths()

    accessions, spec, doc_filter = _load_accessions_and_filters(filters_path, accessions_file)

    ingest_tarballs(paths, [Path(t) for t in tarball], spec, accessions, doc_filter=doc_filter)
    click.echo(f"[ingest] Done. Manifest at {paths.manifest_path}")


@cli.command()
@click.option("--run-id", required=True, callback=validate_run_id)
@click.option("--base-dir", default=".", show_default=True)
def normalize(run_id: str, base_dir: str):
    """Build prompt views from ingested HTML."""
    rc = _resolve_paths(run_id, base_dir, bandwidth=4)
    paths = rc.paths()
    manifest = load_manifest(paths.manifest_path)
    items = manifest_items(manifest)
    build_prompt_views(paths, manifest)
    click.echo(f"[normalize] Built prompt views for {len(items)} items (exhibits).")


@cli.command()
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
    help="Gateway client timeout (seconds) for indexing calls.",
)
@click.option(
    "--concurrency",
    default=3,
    show_default=True,
    type=int,
    help="Number of parallel gateway calls during indexing (only when --model is set).",
)
@click.option(
    "--item-id",
    "item_ids",
    multiple=True,
    help="Optional item_id(s) to index (defaults to all items in the run manifest).",
)
def index(
    run_id: str,
    prompt_path: str,
    base_dir: str,
    model: str,
    gateway_url: str,
    temperature: float,
    reasoning: str | None,
    gateway_timeout: float,
    concurrency: int,
    item_ids: tuple[str, ...],
):
    """Run anchor indexing via agent-gateway (or pass-through when model is omitted)."""
    rc = _resolve_paths(run_id, base_dir, bandwidth=4)
    paths = rc.paths()
    manifest = load_manifest(paths.manifest_path)
    items = manifest_items(manifest)
    manifest_item_ids = {item["item_id"] for item in items}
    if item_ids:
        unknown = sorted(set(item_ids) - manifest_item_ids)
        if unknown:
            raise click.UsageError(f"Unknown --item-id values not present in manifest: {', '.join(unknown)}")
        selected_item_ids = list(item_ids)
    else:
        selected_item_ids = [item["item_id"] for item in items]
    run_indexing(
        paths,
        selected_item_ids,
        Path(prompt_path),
        model=model,
        gateway_url=gateway_url,
        temperature=temperature,
        reasoning=reasoning,
        gateway_timeout=gateway_timeout,
        concurrency=concurrency,
    )

@cli.command()
@click.option("--run-id", required=True, callback=validate_run_id)
@click.option("--bandwidth", default=400, show_default=True, type=int)
@click.option("--base-dir", default=".", show_default=True)
@click.option(
    "--item-id",
    "item_ids",
    multiple=True,
    help="Optional item_id(s) to retrieve snippets for (defaults to all items in the run manifest).",
)
def retrieve(run_id: str, bandwidth: int, base_dir: str, item_ids: tuple[str, ...]):
    """Render snippets around anchors."""
    rc = _resolve_paths(run_id, base_dir, bandwidth=bandwidth)
    paths = rc.paths()
    manifest = load_manifest(paths.manifest_path)
    items = manifest_items(manifest)
    manifest_item_ids = {item["item_id"] for item in items}
    if item_ids:
        unknown = sorted(set(item_ids) - manifest_item_ids)
        if unknown:
            raise click.UsageError(f"Unknown --item-id values not present in manifest: {', '.join(unknown)}")
        selected_item_ids = list(item_ids)
    else:
        selected_item_ids = [item["item_id"] for item in items]
    render_snippets(paths, selected_item_ids, bandwidth=bandwidth)


@cli.command()
@click.option("--run-id", required=True, callback=validate_run_id)
@click.option("--prompt", "prompt_path", type=click.Path(exists=True, dir_okay=False), required=True)
@click.option("--base-dir", default=".", show_default=True)
@click.option(
    "--input-mode",
    default="snippets",
    type=click.Choice(["snippets", "full_document"], case_sensitive=False),
    show_default=True,
    help="Structured input source: retrieval snippets or normalized full document (canonical_annotated.txt).",
)
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
    "--output-subdir",
    default=None,
    help="Optional subfolder under runs/<run_id>/llm_qa/ for outputs (defaults to prompt filename stem).",
)
@click.option(
    "--item-id",
    "item_ids",
    multiple=True,
    help="Optional item_id(s) to run structured extraction for (defaults to all items in the run manifest).",
)
def structured(
    run_id: str,
    prompt_path: str,
    base_dir: str,
    input_mode: str,
    model: str | None,
    gateway_url: str | None,
    temperature: float,
    reasoning: str | None,
    gateway_timeout: float,
    concurrency: int,
    output_subdir: str | None,
    item_ids: tuple[str, ...],
):
    """Structured extraction over snippets."""
    rc = _resolve_paths(run_id, base_dir, bandwidth=4)
    paths = rc.paths()
    input_mode = input_mode.lower()
    manifest = load_manifest(paths.manifest_path)
    items = manifest_items(manifest)
    manifest_item_ids = {item["item_id"] for item in items}
    if item_ids:
        unknown = sorted(set(item_ids) - manifest_item_ids)
        if unknown:
            raise click.UsageError(f"Unknown --item-id values not present in manifest: {', '.join(unknown)}")
        selected_item_ids = list(item_ids)
    else:
        selected_item_ids = [item["item_id"] for item in items]
    run_structured(
        paths,
        selected_item_ids,
        Path(prompt_path),
        input_mode=input_mode,  # type: ignore[arg-type]
        model=model,
        gateway_url=gateway_url,
        temperature=temperature,
        reasoning=reasoning,
        gateway_timeout=gateway_timeout,
        concurrency=concurrency,
        output_subdir=output_subdir,
    )


@cli.command(name="contract-pricing")
@click.option("--run-id", required=True, callback=validate_run_id)
@click.option("--prompt", "prompt_path", type=click.Path(exists=True, dir_okay=False), required=True)
@click.option("--base-dir", default=".", show_default=True)
@click.option("--gateway-url", default=None, help="Gateway base URL; defaults to $GATEWAY_URL.")
@click.option("--temperature", default=0.0, show_default=True, type=float)
@click.option(
    "--gateway-timeout",
    default=600.0,
    show_default=True,
    type=float,
    help="Gateway client timeout (seconds) for contract pricing calls.",
)
@click.option(
    "--concurrency",
    default=2,
    show_default=True,
    type=int,
    help="Number of parallel gateway calls during contract pricing extraction.",
)
@click.option(
    "--attempts",
    default=3,
    show_default=True,
    type=int,
    help="Number of strict JSON/Pydantic attempts per item before failing.",
)
@click.option(
    "--output-subdir",
    default="contract_pricing_v1",
    show_default=True,
    help="Subfolder under runs/<run_id>/contract_pricing/ for outputs.",
)
@click.option(
    "--item-id",
    "item_ids",
    multiple=True,
    help="Optional item_id(s) to extract pricing for (defaults to all items in the run manifest).",
)
def contract_pricing(
    run_id: str,
    prompt_path: str,
    base_dir: str,
    gateway_url: str | None,
    temperature: float,
    gateway_timeout: float,
    concurrency: int,
    attempts: int,
    output_subdir: str,
    item_ids: tuple[str, ...],
):
    """Extract pricing regimes into a contract-like IR (doc IR -> LLM compile)."""
    rc = _resolve_paths(run_id, base_dir, bandwidth=4)
    paths = rc.paths()
    manifest = load_manifest(paths.manifest_path)
    items = manifest_items(manifest)
    manifest_item_ids = {item["item_id"] for item in items}
    if item_ids:
        unknown = sorted(set(item_ids) - manifest_item_ids)
        if unknown:
            raise click.UsageError(f"Unknown --item-id values not present in manifest: {', '.join(unknown)}")
        selected_item_ids = list(item_ids)
    else:
        selected_item_ids = [item["item_id"] for item in items]

    run_contract_pricing(
        paths,
        selected_item_ids,
        Path(prompt_path),
        gateway_url=gateway_url,
        temperature=temperature,
        gateway_timeout=gateway_timeout,
        concurrency=concurrency,
        output_subdir=output_subdir,
        attempts=attempts,
    )


@cli.command(name="contract-pricing-flow")
@click.option(
    "--run-id",
    required=True,
    callback=validate_run_id,
    help="Run identifier (writes under runs/<run_id>/; reruns overwrite artifacts).",
)
@click.option("--tarball", multiple=True, type=click.Path(exists=True, dir_okay=False), required=True)
@click.option("--accessions-file", type=click.Path(exists=True, dir_okay=False), required=False)
@click.option("--prompt", "prompt_path", type=click.Path(exists=True, dir_okay=False), required=True)
@click.option(
    "--filters",
    "filters_path",
    type=click.Path(exists=True, dir_okay=False),
    required=False,
    help="Filter spec (JSON/YAML with doc_filter_path and optional doc_filter_kwargs). Defaults to keep_all when omitted.",
)
@click.option("--base-dir", default=".", show_default=True)
@click.option("--gateway-url", default=None, help="Gateway base URL; defaults to $GATEWAY_URL.")
@click.option("--temperature", default=0.0, show_default=True, type=float)
@click.option(
    "--gateway-timeout",
    default=600.0,
    show_default=True,
    type=float,
    help="Gateway client timeout (seconds) for contract pricing calls.",
)
@click.option(
    "--concurrency",
    default=2,
    show_default=True,
    type=int,
    help="Number of parallel gateway calls during contract pricing extraction.",
)
@click.option(
    "--attempts",
    default=3,
    show_default=True,
    type=int,
    help="Number of strict JSON/Pydantic attempts per item before failing.",
)
@click.option(
    "--output-subdir",
    default="contract_pricing_v1",
    show_default=True,
    help="Subfolder under runs/<run_id>/contract_pricing/ for outputs.",
)
@click.option(
    "--item-id",
    "item_ids",
    multiple=True,
    help="Optional item_id(s) to extract pricing for (defaults to all items in the run manifest).",
)
def contract_pricing_flow(
    run_id: str,
    tarball,
    accessions_file: Optional[str],
    prompt_path: str,
    filters_path: Optional[str],
    base_dir: str,
    gateway_url: str | None,
    temperature: float,
    gateway_timeout: float,
    concurrency: int,
    attempts: int,
    output_subdir: str,
    item_ids: tuple[str, ...],
):
    """Run ingest -> normalize -> contract-pricing (no indexing/retrieval)."""

    rc = _resolve_paths(run_id, base_dir, bandwidth=4)
    paths = rc.paths()

    accessions, spec, doc_filter = _load_accessions_and_filters(filters_path, accessions_file)
    ingest_tarballs(paths, [Path(t) for t in tarball], spec, accessions, doc_filter=doc_filter)

    manifest = load_manifest(paths.manifest_path)
    items = manifest_items(manifest)
    manifest_item_ids = {item["item_id"] for item in items}
    build_prompt_views(paths, manifest)

    if item_ids:
        unknown = sorted(set(item_ids) - manifest_item_ids)
        if unknown:
            raise click.UsageError(f"Unknown --item-id values not present in manifest: {', '.join(unknown)}")
        selected_item_ids = list(item_ids)
    else:
        selected_item_ids = [item["item_id"] for item in items]

    run_contract_pricing(
        paths,
        selected_item_ids,
        Path(prompt_path),
        gateway_url=gateway_url,
        temperature=temperature,
        gateway_timeout=gateway_timeout,
        concurrency=concurrency,
        output_subdir=output_subdir,
        attempts=attempts,
    )
    click.echo(f"[contract-pricing-flow] Completed contract pricing extraction for {len(selected_item_ids)} exhibits.")


@cli.command()
@click.option("--run-id", required=True, callback=validate_run_id)
@click.option(
    "--qa-dir",
    type=click.Path(exists=True, file_okay=False, dir_okay=True),
    required=True,
    help="Directory containing Q/A JSON outputs to read metrics from (e.g., llm_qa/instructions-1)",
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
    help="Gateway client timeout (seconds) for term resolution calls.",
)
@click.option(
    "--concurrency",
    default=3,
    show_default=True,
    type=int,
    help="Number of parallel gateway calls during term resolution.",
)
@click.option(
    "--output-subdir",
    default="terms_from_qa",
    show_default=True,
    help="Subfolder under runs/<run_id>/llm_qa/ for outputs.",
)
def terms(
    run_id: str,
    qa_dir: str,
    base_dir: str,
    model: str | None,
    gateway_url: str | None,
    temperature: float,
    reasoning: str | None,
    gateway_timeout: float,
    concurrency: int,
    output_subdir: str,
):
    """Resolve exact agreement term wording for metrics used in Q/A outputs."""

    from .terms import run_terms_lookup

    rc = _resolve_paths(run_id, base_dir, bandwidth=4)
    paths = rc.paths()
    manifest = load_manifest(paths.manifest_path)
    items = manifest_items(manifest)
    item_ids = [item["item_id"] for item in items]
    run_terms_lookup(
        paths,
        item_ids,
        qa_dir=Path(qa_dir),
        model=model,
        gateway_url=gateway_url,
        temperature=temperature,
        reasoning=reasoning,
        gateway_timeout=gateway_timeout,
        concurrency=concurrency,
        output_subdir=output_subdir,
    )


@cli.command()
@click.option("--run-id", required=True, callback=validate_run_id)
@click.option("--base-dir", default=".", show_default=True)
def validate(run_id: str, base_dir: str):
    """Run QA/validation (stub)."""
    rc = _resolve_paths(run_id, base_dir, bandwidth=4)
    paths = rc.paths()
    manifest = load_manifest(paths.manifest_path)
    items = manifest_items(manifest)
    run_validation(paths, items)


@cli.command()
@click.option("--run-id", required=True, callback=validate_run_id)
@click.option("--tarball", multiple=True, type=click.Path(exists=True, dir_okay=False), required=True)
@click.option("--accessions-file", type=click.Path(exists=True, dir_okay=False), required=False)
@click.option("--prompt-index", type=click.Path(exists=True, dir_okay=False), required=True)
@click.option("--prompt-structured", type=click.Path(exists=True, dir_okay=False), required=True)
@click.option("--bandwidth", default=400, show_default=True, type=int)
@click.option("--base-dir", default=".", show_default=True)
@click.option(
    "--filters",
    "filters_path",
    type=click.Path(exists=True, dir_okay=False),
    required=False,
    help="Filter spec (JSON/YAML with doc_filter_path and optional doc_filter_kwargs). Defaults to keep_all when omitted.",
)
@click.option(
    "--model",
    default=REQUIRED_MODEL,
    show_default=True,
    help="Gateway model for indexing (enforced).",
)
@click.option("--gateway-url", default=None, help="Gateway base URL for indexing; defaults to $GATEWAY_URL.")
@click.option("--temperature", default=0.0, show_default=True, type=float, help="Indexing LLM temperature.")
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
    help="Gateway client timeout (seconds) for indexing calls.",
)
@click.option(
    "--concurrency",
    default=3,
    show_default=True,
    type=int,
    help="Number of parallel gateway calls during indexing (only when --model is set).",
)
def all(
    run_id: str,
    tarball,
    accessions_file: Optional[str],
    prompt_index: str,
    prompt_structured: str,
    bandwidth: int,
    base_dir: str,
    filters_path: Optional[str],
    model: str,
    gateway_url: str,
    temperature: float,
    reasoning: str | None,
    gateway_timeout: float,
    concurrency: int,
):
    """Run ingest -> normalize -> index -> retrieve -> structured."""
    rc = _resolve_paths(run_id, base_dir, bandwidth=bandwidth)
    paths = rc.paths()

    accessions, spec, doc_filter = _load_accessions_and_filters(filters_path, accessions_file)

    ingest_tarballs(paths, [Path(t) for t in tarball], spec, accessions, doc_filter=doc_filter)
    manifest = load_manifest(paths.manifest_path)
    items = manifest_items(manifest)
    item_ids = [item["item_id"] for item in items]
    build_prompt_views(paths, manifest)
    run_indexing(
        paths,
        item_ids,
        Path(prompt_index),
        model=model,
        gateway_url=gateway_url,
        temperature=temperature,
        reasoning=reasoning,
        gateway_timeout=gateway_timeout,
        concurrency=concurrency,
    )
    render_snippets(paths, item_ids, bandwidth=bandwidth)
    run_structured(paths, item_ids, Path(prompt_structured))
    click.echo(f"[all] Completed through structured stage for {len(item_ids)} exhibits.")


def main():
    cli()


if __name__ == "__main__":
    main()
