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
from .validation import run_validation
from .utils import load_manifest, manifest_items, read_accessions_file
from .run_id import validate_run_id


def resolve_run_config(run_id: str, base_dir: str, workers: int, bandwidth: int, namespace: str | None) -> RunConfig:
    # Validate even when called programmatically (outside Click).
    run_id_validated = validate_run_id(None, None, run_id)
    return RunConfig(
        run_id=run_id_validated,
        base_dir=Path(base_dir),
        workers=workers,
        bandwidth=bandwidth,
        namespace=namespace,
    )


def _resolve_paths(run_id: str, base_dir: str, bandwidth: int, namespace: str | None) -> RunConfig:
    """Convenience helper to construct RunConfig and paths in one place."""
    return resolve_run_config(run_id, base_dir, workers=4, bandwidth=bandwidth, namespace=namespace)


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
    help="Run identifier (creates runs/<run_id>/)",
)
@click.option("--tarball", multiple=True, type=click.Path(exists=True, dir_okay=False), required=True)
@click.option(
    "--filters",
    "filters_path",
    type=click.Path(exists=True, dir_okay=False),
    required=False,
    help="Filter spec (JSON/YAML with doc_filter_path). Defaults to keep_all when omitted.",
)
@click.option("--accessions-file", type=click.Path(exists=True, dir_okay=False), required=False)
@click.option("--base-dir", default=".", show_default=True)
@click.option(
    "--resume",
    is_flag=True,
    help="Allow reusing an existing run directory/manifest instead of failing fast.",
)
@click.option(
    "--namespace",
    "--ns",
    "namespace",
    default=None,
    help="Optional namespace under runs/ for grouping run directories (e.g., 'pi').",
)
def ingest(run_id: str, tarball, filters_path: Optional[str], accessions_file: Optional[str], base_dir: str, resume: bool, namespace: str | None):
    """Extract EX-10 HTMLs for selected accessions from tarballs."""
    rc = _resolve_paths(run_id, base_dir, bandwidth=4, namespace=namespace)
    paths = rc.paths()

    if paths.manifest_path.exists() and not resume:
        raise click.UsageError(
            f"Manifest already exists for run_id '{run_id}' at {paths.manifest_path}. "
            "Use --resume to reuse it or pick a new --run-id."
        )

    accessions, spec, doc_filter = _load_accessions_and_filters(filters_path, accessions_file)

    ingest_tarballs(paths, [Path(t) for t in tarball], spec, accessions, doc_filter=doc_filter)
    click.echo(f"[ingest] Done. Manifest at {paths.manifest_path}")


@cli.command()
@click.option("--run-id", required=True, callback=validate_run_id)
@click.option("--base-dir", default=".", show_default=True)
@click.option(
    "--namespace",
    "--ns",
    "namespace",
    default=None,
    help="Optional namespace under runs/ for grouping run directories (e.g., 'pi').",
)
def normalize(run_id: str, base_dir: str, namespace: str | None):
    """Build prompt views from ingested HTML."""
    rc = _resolve_paths(run_id, base_dir, bandwidth=4, namespace=namespace)
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
    "--namespace",
    "--ns",
    "namespace",
    default=None,
    help="Optional namespace under runs/ for grouping run directories (e.g., 'pi').",
)
def index(run_id: str, prompt_path: str, base_dir: str, model: str, gateway_url: str, temperature: float, reasoning: str | None, gateway_timeout: float, concurrency: int, namespace: str | None):
    """Run anchor indexing via agent-gateway (or pass-through when model is omitted)."""
    rc = _resolve_paths(run_id, base_dir, bandwidth=4, namespace=namespace)
    paths = rc.paths()
    manifest = load_manifest(paths.manifest_path)
    items = manifest_items(manifest)
    item_ids = [item["item_id"] for item in items]
    run_indexing(
        paths,
        item_ids,
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
    "--namespace",
    "--ns",
    "namespace",
    default=None,
    help="Optional namespace under runs/ for grouping run directories (e.g., 'pi').",
)
def retrieve(run_id: str, bandwidth: int, base_dir: str, namespace: str | None):
    """Render snippets around anchors."""
    rc = _resolve_paths(run_id, base_dir, bandwidth=bandwidth, namespace=namespace)
    paths = rc.paths()
    manifest = load_manifest(paths.manifest_path)
    items = manifest_items(manifest)
    item_ids = [item["item_id"] for item in items]
    render_snippets(paths, item_ids, bandwidth=bandwidth)


@cli.command()
@click.option("--run-id", required=True, callback=validate_run_id)
@click.option("--prompt", "prompt_path", type=click.Path(exists=True, dir_okay=False), required=True)
@click.option("--base-dir", default=".", show_default=True)
@click.option(
    "--namespace",
    "--ns",
    "namespace",
    default=None,
    help="Optional namespace under runs/ for grouping run directories (e.g., 'pi').",
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
def structured(
    run_id: str,
    prompt_path: str,
    base_dir: str,
    namespace: str | None,
    model: str | None,
    gateway_url: str | None,
    temperature: float,
    reasoning: str | None,
    gateway_timeout: float,
    concurrency: int,
    output_subdir: str | None,
):
    """Structured extraction over snippets."""
    rc = _resolve_paths(run_id, base_dir, bandwidth=4, namespace=namespace)
    paths = rc.paths()
    manifest = load_manifest(paths.manifest_path)
    items = manifest_items(manifest)
    item_ids = [item["item_id"] for item in items]
    run_structured(
        paths,
        item_ids,
        Path(prompt_path),
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
@click.option(
    "--qa-dir",
    type=click.Path(exists=True, file_okay=False, dir_okay=True),
    required=True,
    help="Directory containing Q/A JSON outputs to read metrics from (e.g., llm_qa/instructions-1)",
)
@click.option("--base-dir", default=".", show_default=True)
@click.option(
    "--namespace",
    "--ns",
    "namespace",
    default=None,
    help="Optional namespace under runs/ for grouping run directories (e.g., 'pi').",
)
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
    namespace: str | None,
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

    rc = _resolve_paths(run_id, base_dir, bandwidth=4, namespace=namespace)
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
@click.option(
    "--namespace",
    "--ns",
    "namespace",
    default=None,
    help="Optional namespace under runs/ for grouping run directories (e.g., 'pi').",
)
def validate(run_id: str, base_dir: str, namespace: str | None):
    """Run QA/validation (stub)."""
    rc = _resolve_paths(run_id, base_dir, bandwidth=4, namespace=namespace)
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
    help="Filter spec (JSON/YAML with doc_filter_path). Defaults to keep_all when omitted.",
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
@click.option(
    "--namespace",
    "--ns",
    "namespace",
    default=None,
    help="Optional namespace under runs/ for grouping run directories (e.g., 'pi').",
)
def all(
    run_id: str,
    tarball,
    accessions_file: Optional[str],
    prompt_index: str,
    prompt_structured: str,
    bandwidth: int,
    base_dir: str,
    namespace: str | None,
    filters_path: Optional[str],
    model: str,
    gateway_url: str,
    temperature: float,
    reasoning: str | None,
    gateway_timeout: float,
    concurrency: int,
):
    """Run ingest -> normalize -> index -> retrieve -> structured."""
    rc = _resolve_paths(run_id, base_dir, bandwidth=bandwidth, namespace=namespace)
    paths = rc.paths()

    if paths.manifest_path.exists():
        raise click.UsageError(
            f"Manifest already exists for run_id '{run_id}' at {paths.manifest_path}. "
            "Pick a new --run-id or remove the existing run directory."
        )

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
