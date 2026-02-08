from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import click

from pipeline.core.config import FilterSpec, Paths, RunConfig
from pipeline.filters import load_doc_filter, load_filter_spec
from pipeline.core.run_id import validate_run_id
from pipeline.utils import load_manifest, manifest_items, read_accessions_file


def resolve_run_config(run_id: str, base_dir: str, *, workers: int, bandwidth: int) -> RunConfig:
    """Construct a validated RunConfig.

    We validate run_id even when called programmatically (outside Click) so run directory layout
    is always safe/consistent.
    """

    run_id_validated = validate_run_id(None, None, run_id)
    return RunConfig(
        run_id=run_id_validated,
        base_dir=Path(base_dir),
        workers=workers,
        bandwidth=bandwidth,
    )


def resolve_paths(run_id: str, base_dir: str, *, bandwidth: int, workers: int = 4) -> Paths:
    """Convenience helper to construct Paths (RunConfig + .paths()) in one place."""

    rc = resolve_run_config(run_id, base_dir, workers=workers, bandwidth=bandwidth)
    return rc.paths()


def load_accessions_and_filters(filters_path: Optional[str], accessions_file: Optional[str]):
    """Shared validation for ingest / all-v2 commands."""

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


def resolve_selected_item_ids(paths: Paths, item_ids: Tuple[str, ...]) -> Tuple[dict, list[dict], list[str]]:
    """Load the run manifest and resolve which item_ids to operate on.

    Policy:
    - If the user provides --item-id, fail loudly on unknown values.
    - Otherwise default to all items in the manifest.
    """

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

    return manifest, items, selected_item_ids

