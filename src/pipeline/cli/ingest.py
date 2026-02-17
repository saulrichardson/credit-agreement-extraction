from __future__ import annotations

from pathlib import Path
from typing import Optional

import click

from pipeline.evidence.ingest import ingest_tarballs
from pipeline.evidence.normalize import build_prompt_views
from pipeline.core.run_id import validate_run_id
from pipeline.utils import load_manifest, manifest_items
from .common import load_accessions_and_selection, resolve_paths


@click.command()
@click.option(
    "--run-id",
    required=True,
    callback=validate_run_id,
    help="Run identifier (writes under runs/<run_id>/; reruns overwrite artifacts).",
)
@click.option("--tarball", multiple=True, type=click.Path(exists=True, dir_okay=False), required=True)
@click.option(
    "--item-ids-file",
    type=click.Path(exists=True, dir_okay=False),
    required=False,
    help="JSON/YAML dataset with item_ids allowlist (mapping with item_ids or a bare list).",
)
@click.option(
    "--doc-type-prefix",
    "doc_type_prefixes",
    multiple=True,
    required=False,
    help="Repeatable SGML <TYPE> prefix allowlist (e.g., EX-10).",
)
@click.option("--accessions-file", type=click.Path(exists=True, dir_okay=False), required=False)
@click.option("--base-dir", default=".", show_default=True)
def ingest(
    run_id: str,
    tarball,
    item_ids_file: Optional[str],
    doc_type_prefixes: tuple[str, ...],
    accessions_file: Optional[str],
    base_dir: str,
) -> None:
    """Extract selected exhibit HTMLs for chosen filings/documents from tarballs."""

    paths = resolve_paths(run_id, base_dir, bandwidth=4)

    accessions, selection, doc_filter = load_accessions_and_selection(
        accessions_file=accessions_file,
        item_ids_file=item_ids_file,
        doc_type_prefixes=doc_type_prefixes,
    )

    ingest_tarballs(paths, [Path(t) for t in tarball], selection, accessions, doc_filter=doc_filter)
    click.echo(f"[ingest] Done. Manifest at {paths.manifest_path}")


@click.command()
@click.option("--run-id", required=True, callback=validate_run_id)
@click.option("--base-dir", default=".", show_default=True)
def normalize(run_id: str, base_dir: str) -> None:
    """Build prompt views from ingested HTML."""

    paths = resolve_paths(run_id, base_dir, bandwidth=4)
    manifest = load_manifest(paths.manifest_path)
    items = manifest_items(manifest)
    build_prompt_views(paths, manifest)
    click.echo(f"[normalize] Built prompt views for {len(items)} items (exhibits).")
