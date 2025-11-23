from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import click

from .config import FilterSpec, RunConfig
from .filters import load_filter_spec
from .ingest import ingest_tarballs
from .normalize import build_prompt_views
from .indexing import run_indexing
from .retrieval import render_snippets
from .structured import run_structured
from .validation import run_validation
from .utils import load_manifest, manifest_accessions


def resolve_run_config(run_id: str, base_dir: str, workers: int, bandwidth: int) -> RunConfig:
    return RunConfig(run_id=run_id, base_dir=Path(base_dir), workers=workers, bandwidth=bandwidth)


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
    rc = resolve_run_config(run_id, base_dir, workers=4, bandwidth=4)
    paths = rc.paths()

    if accessions_file:
        accessions = [line.strip() for line in Path(accessions_file).read_text().splitlines() if line.strip()]
    else:
        raise click.UsageError("accessions-file is required to avoid processing whole tarballs.")
    if not accessions:
        raise click.UsageError("accessions-file is empty.")

    spec = FilterSpec()
    if filters_path:
        spec = load_filter_spec(Path(filters_path))

    ingest_tarballs(paths, [Path(t) for t in tarball], spec, accessions)
    click.echo(f"[ingest] Done. Manifest at {paths.manifest_path}")


@cli.command()
@click.option("--run-id", required=True)
@click.option("--base-dir", default=".", show_default=True)
def normalize(run_id: str, base_dir: str):
    """Build prompt views from ingested HTML."""
    rc = resolve_run_config(run_id, base_dir, workers=4, bandwidth=4)
    paths = rc.paths()
    manifest = load_manifest(paths.manifest_path)
    accessions = manifest_accessions(manifest)
    build_prompt_views(paths, accessions)
    click.echo(f"[normalize] Built prompt views for {len(accessions)} accessions.")


@cli.command()
@click.option("--run-id", required=True)
@click.option("--prompt", "prompt_path", type=click.Path(exists=True, dir_okay=False), required=True)
@click.option("--base-dir", default=".", show_default=True)
def index(run_id: str, prompt_path: str, base_dir: str):
    """Run anchor indexing (all-in-one prompt)."""
    rc = resolve_run_config(run_id, base_dir, workers=4, bandwidth=4)
    paths = rc.paths()
    manifest = load_manifest(paths.manifest_path)
    accessions = manifest_accessions(manifest)
    run_indexing(paths, accessions, Path(prompt_path))


@cli.command()
@click.option("--run-id", required=True)
@click.option("--bandwidth", default=4, show_default=True, type=int)
@click.option("--base-dir", default=".", show_default=True)
def retrieve(run_id: str, bandwidth: int, base_dir: str):
    """Render snippets around anchors."""
    rc = resolve_run_config(run_id, base_dir, workers=4, bandwidth=bandwidth)
    paths = rc.paths()
    manifest = load_manifest(paths.manifest_path)
    accessions = manifest_accessions(manifest)
    render_snippets(paths, accessions, bandwidth=bandwidth)


@cli.command()
@click.option("--run-id", required=True)
@click.option("--prompt", "prompt_path", type=click.Path(exists=True, dir_okay=False), required=True)
@click.option("--base-dir", default=".", show_default=True)
def structured(run_id: str, prompt_path: str, base_dir: str):
    """Structured extraction over snippets."""
    rc = resolve_run_config(run_id, base_dir, workers=4, bandwidth=4)
    paths = rc.paths()
    manifest = load_manifest(paths.manifest_path)
    accessions = manifest_accessions(manifest)
    run_structured(paths, accessions, Path(prompt_path))


@cli.command()
@click.option("--run-id", required=True)
@click.option("--base-dir", default=".", show_default=True)
def validate(run_id: str, base_dir: str):
    """Run QA/validation (stub)."""
    rc = resolve_run_config(run_id, base_dir, workers=4, bandwidth=4)
    paths = rc.paths()
    manifest = load_manifest(paths.manifest_path)
    accessions = manifest_accessions(manifest)
    run_validation(paths, accessions)


@cli.command()
@click.option("--run-id", required=True)
@click.option("--tarball", multiple=True, type=click.Path(exists=True, dir_okay=False), required=True)
@click.option("--accessions-file", type=click.Path(exists=True, dir_okay=False), required=True)
@click.option("--prompt-index", type=click.Path(exists=True, dir_okay=False), required=True)
@click.option("--prompt-structured", type=click.Path(exists=True, dir_okay=False), required=True)
@click.option("--bandwidth", default=4, show_default=True, type=int)
@click.option("--base-dir", default=".", show_default=True)
@click.option("--filters", "filters_path", type=click.Path(exists=True, dir_okay=False), required=False)
def all(
    run_id: str,
    tarball,
    accessions_file: str,
    prompt_index: str,
    prompt_structured: str,
    bandwidth: int,
    base_dir: str,
    filters_path: Optional[str],
):
    """Run ingest -> normalize -> index -> retrieve -> structured."""
    rc = resolve_run_config(run_id, base_dir, workers=4, bandwidth=bandwidth)
    paths = rc.paths()

    accessions = [line.strip() for line in Path(accessions_file).read_text().splitlines() if line.strip()]
    if not accessions:
        raise click.UsageError("accessions-file is empty.")
    spec = FilterSpec()
    if filters_path:
        spec = load_filter_spec(Path(filters_path))

    ingest_tarballs(paths, [Path(t) for t in tarball], spec, accessions)
    build_prompt_views(paths, accessions)
    run_indexing(paths, accessions, Path(prompt_index))
    render_snippets(paths, accessions, bandwidth=bandwidth)
    run_structured(paths, accessions, Path(prompt_structured))
    click.echo(f"[all] Completed through structured stage for {len(accessions)} accessions.")


def main():
    cli()


if __name__ == "__main__":
    main()
