from __future__ import annotations

from pathlib import Path
from typing import Optional

import click

from pipeline.run import _load_accessions_and_filters, resolve_run_config
from pipeline.ingest import ingest_tarballs
from pipeline.normalize import build_prompt_views
from pipeline.contract_pricing import run_contract_pricing
from pipeline.utils import load_manifest, manifest_items


@click.command()
@click.option("--run-id", required=True, help="Run identifier.")
@click.option("--base-dir", default=".", show_default=True)
@click.option("--tarball", multiple=True, type=click.Path(exists=True, dir_okay=False), required=True)
@click.option(
    "--filters",
    "filters_path",
    type=click.Path(exists=True, dir_okay=False),
    required=True,
    help="Filter spec JSON/YAML (doc_filter_path + optional doc_filter_kwargs).",
)
@click.option("--accessions-file", type=click.Path(exists=True, dir_okay=False), required=False)
@click.option("--pricing-prompt", type=click.Path(exists=True, dir_okay=False), required=True)
@click.option("--gateway-url", default=None, help="Gateway base URL; defaults to $GATEWAY_URL or 127.0.0.1:8000.")
@click.option("--temperature", default=0.0, show_default=True, type=float)
@click.option("--gateway-timeout", default=600.0, show_default=True, type=float)
@click.option("--concurrency", default=2, show_default=True, type=int, help="Parallel gateway calls during pricing extraction.")
@click.option("--attempts", default=3, show_default=True, type=int, help="Strict JSON/Pydantic attempts per item before failing.")
@click.option("--output-subdir", default="contract_pricing_v1", show_default=True)
def main(
    run_id: str,
    base_dir: str,
    tarball: tuple,
    filters_path: str,
    accessions_file: Optional[str],
    pricing_prompt: str,
    gateway_url: Optional[str],
    temperature: float,
    gateway_timeout: float,
    concurrency: int,
    attempts: int,
    output_subdir: str,
):
    """End-to-end pricing compiler flow: ingest -> normalize -> contract pricing.

    Notes:
    - This flow does NOT run indexing/retrieval; it extracts pricing regimes directly from normalized tables.
    - The underlying contract pricing step enforces model + reasoning (see pipeline.config).
    """

    rc = resolve_run_config(run_id, base_dir, workers=4, bandwidth=4)
    paths = rc.paths()

    accessions, spec, doc_filter = _load_accessions_and_filters(filters_path, accessions_file)
    ingest_tarballs(paths, [Path(t) for t in tarball], spec, accessions, doc_filter=doc_filter)

    manifest = load_manifest(paths.manifest_path)
    build_prompt_views(paths, manifest)

    items = manifest_items(manifest)
    item_ids = [item["item_id"] for item in items]
    run_contract_pricing(
        paths,
        item_ids,
        Path(pricing_prompt),
        gateway_url=gateway_url,
        temperature=temperature,
        gateway_timeout=gateway_timeout,
        concurrency=concurrency,
        output_subdir=output_subdir,
        attempts=attempts,
    )


if __name__ == "__main__":
    main()

