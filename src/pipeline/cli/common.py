from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import click

from pipeline.core.config import Paths, RunConfig
from pipeline.evidence.selection import build_document_selection
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


def load_accessions_and_selection(
    *,
    accessions_file: Optional[str],
    item_ids_file: Optional[str],
    doc_type_prefixes: Tuple[str, ...],
):
    """Shared ingest selector validation for CLI entrypoints.

    We require at least one explicit selector so ingest does not scan all exhibits by accident.
    """

    accessions = read_accessions_file(Path(accessions_file)) if accessions_file else None

    if not accessions_file and not item_ids_file and not doc_type_prefixes:
        raise click.UsageError(
            "Provide at least one selector: --accessions-file, --item-ids-file, or --doc-type-prefix."
        )

    selection, doc_filter = build_document_selection(
        item_ids_file=item_ids_file,
        doc_type_prefixes=doc_type_prefixes,
    )
    return accessions, selection, doc_filter


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
