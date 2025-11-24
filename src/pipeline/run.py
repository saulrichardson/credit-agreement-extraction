from __future__ import annotations

from pathlib import Path
from typing import Optional

import click

from .config import FilterSpec, RunConfig
from .filters import load_filter_spec, load_doc_filter
from .ingest import ingest_tarballs
from .normalize import build_prompt_views
from .indexing import run_indexing
from .retrieval import render_snippets
from .structured import run_structured
from .validation import run_validation
from .utils import load_manifest, manifest_items, read_accessions_file


def resolve_run_config(run_id: str, base_dir: str, workers: int, bandwidth: int) -> RunConfig:
    return RunConfig(run_id=run_id, base_dir=Path(base_dir), workers=workers, bandwidth=bandwidth)


def _resolve_paths(run_id: str, base_dir: str, bandwidth: int) -> RunConfig:
    """Convenience helper to construct RunConfig and paths in one place."""
    return resolve_run_config(run_id, base_dir, workers=4, bandwidth=bandwidth)


def _load_accessions_and_filters(filters_path: Optional[str], accessions_file: Optional[str]):
    """Shared validation for ingest/all commands."""
    accessions = read_accessions_file(Path(accessions_file)) if accessions_file else None

    if not accessions_file and not filters_path:
        raise click.UsageError("Provide either accessions-file or filters to avoid scanning everything.")
    if not filters_path:
        raise click.UsageError("filters file is required; provide doc_filter_path in it (module:function)")

    spec = load_filter_spec(Path(filters_path))
    doc_filter = load_doc_filter(spec)
    return accessions, spec, doc_filter


@click.group()
def cli():
    """Run-scoped EX-10 processing pipeline."""


@cli.command()
@click.option("--run-id", required=True, help="Run identifier (creates runs/<run_id>/)")
@click.option("--tarball", multiple=True, type=click.Path(exists=True, dir_okay=False), required=True)
@click.option("--filters", "filters_path", type=click.Path(exists=True, dir_okay=False), required=False)
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
@click.option("--run-id", required=True)
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
@click.option("--run-id", required=True)
@click.option("--prompt", "prompt_path", type=click.Path(exists=True, dir_okay=False), required=True)
@click.option("--base-dir", default=".", show_default=True)
def index(run_id: str, prompt_path: str, base_dir: str):
    """Run anchor indexing (all-in-one prompt)."""
    rc = _resolve_paths(run_id, base_dir, bandwidth=4)
    paths = rc.paths()
    manifest = load_manifest(paths.manifest_path)
    items = manifest_items(manifest)
    item_ids = [item["item_id"] for item in items]
    run_indexing(paths, item_ids, Path(prompt_path))


@cli.command()
@click.option("--run-id", required=True)
@click.option("--bandwidth", default=400, show_default=True, type=int)
@click.option("--base-dir", default=".", show_default=True)
def retrieve(run_id: str, bandwidth: int, base_dir: str):
    """Render snippets around anchors."""
    rc = _resolve_paths(run_id, base_dir, bandwidth=bandwidth)
    paths = rc.paths()
    manifest = load_manifest(paths.manifest_path)
    items = manifest_items(manifest)
    item_ids = [item["item_id"] for item in items]
    render_snippets(paths, item_ids, bandwidth=bandwidth)


@cli.command()
@click.option("--run-id", required=True)
@click.option("--prompt", "prompt_path", type=click.Path(exists=True, dir_okay=False), required=True)
@click.option("--base-dir", default=".", show_default=True)
def structured(run_id: str, prompt_path: str, base_dir: str):
    """Structured extraction over snippets."""
    rc = _resolve_paths(run_id, base_dir, bandwidth=4)
    paths = rc.paths()
    manifest = load_manifest(paths.manifest_path)
    items = manifest_items(manifest)
    item_ids = [item["item_id"] for item in items]
    run_structured(paths, item_ids, Path(prompt_path))


@cli.command()
@click.option("--run-id", required=True)
@click.option("--base-dir", default=".", show_default=True)
def validate(run_id: str, base_dir: str):
    """Run QA/validation (stub)."""
    rc = _resolve_paths(run_id, base_dir, bandwidth=4)
    paths = rc.paths()
    manifest = load_manifest(paths.manifest_path)
    items = manifest_items(manifest)
    run_validation(paths, items)


@cli.command()
@click.option("--run-id", required=True)
@click.option("--tarball", multiple=True, type=click.Path(exists=True, dir_okay=False), required=True)
@click.option("--accessions-file", type=click.Path(exists=True, dir_okay=False), required=False)
@click.option("--prompt-index", type=click.Path(exists=True, dir_okay=False), required=True)
@click.option("--prompt-structured", type=click.Path(exists=True, dir_okay=False), required=True)
@click.option("--bandwidth", default=400, show_default=True, type=int)
@click.option("--base-dir", default=".", show_default=True)
@click.option("--filters", "filters_path", type=click.Path(exists=True, dir_okay=False), required=False)
def all(
    run_id: str,
    tarball,
    accessions_file: Optional[str],
    prompt_index: str,
    prompt_structured: str,
    bandwidth: int,
    base_dir: str,
    filters_path: Optional[str],
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
    run_indexing(paths, item_ids, Path(prompt_index))
    render_snippets(paths, item_ids, bandwidth=bandwidth)
    run_structured(paths, item_ids, Path(prompt_structured))
    click.echo(f"[all] Completed through structured stage for {len(item_ids)} exhibits.")


def main():
    cli()


if __name__ == "__main__":
    main()
