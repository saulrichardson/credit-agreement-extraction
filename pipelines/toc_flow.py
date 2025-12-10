from __future__ import annotations

from pathlib import Path
from typing import Optional

import click

from pipeline.run import _load_accessions_and_filters, resolve_run_config
from pipeline.config import REQUIRED_MODEL, REQUIRED_REASONING
from pipeline.ingest import ingest_tarballs
from pipeline.normalize import build_prompt_views
from pipeline.indexing import run_indexing
from pipeline.retrieval import render_snippets
from pipeline.structured import run_structured
from pipeline.utils import load_manifest, manifest_items


@click.command()
@click.option("--run-id", required=True, help="Run identifier.")
@click.option("--namespace", "--ns", default=None, help="Optional namespace under runs/ (e.g., 'pi').")
@click.option("--base-dir", default=".", show_default=True)
@click.option("--tarball", multiple=True, type=click.Path(exists=True, dir_okay=False), required=True)
@click.option("--filters", "filters_path", type=click.Path(exists=True, dir_okay=False), required=True, help="Filter spec JSON/YAML (doc_filter_path).")
@click.option("--accessions-file", type=click.Path(exists=True, dir_okay=False), required=False)
@click.option("--index-prompt", type=click.Path(exists=True, dir_okay=False), required=True, help="Prompt for TOC/anchor selection.")
@click.option("--definitions-prompt", type=click.Path(exists=True, dir_okay=False), required=True, help="Prompt for definitions extraction.")
@click.option("--verification-prompt", type=click.Path(exists=True, dir_okay=False), required=False, help="Optional prompt for verification/consistency checks.")
@click.option("--model", default=REQUIRED_MODEL, show_default=True, help="Gateway model (enforced).")
@click.option("--gateway-url", default=None, help="Gateway base URL; defaults to $GATEWAY_URL or 127.0.0.1:8000.")
@click.option("--temperature", default=0.0, show_default=True, type=float)
@click.option("--reasoning", default=REQUIRED_REASONING, show_default=True, type=click.Choice(["light", "medium", "heavy"], case_sensitive=False))
@click.option("--gateway-timeout", default=600.0, show_default=True, type=float)
@click.option("--concurrency", default=3, show_default=True, type=int, help="Parallel gateway calls for indexing/structured.")
@click.option("--bandwidth", default=400, show_default=True, type=int, help="Snippet context characters for retrieval.")
@click.option("--definitions-subdir", default="definitions_v1", show_default=True, help="llm_qa subdir for definitions outputs.")
@click.option("--verification-subdir", default="verification_v1", show_default=True, help="llm_qa subdir for verification outputs.")
def main(
    run_id: str,
    namespace: Optional[str],
    base_dir: str,
    tarball: tuple,
    filters_path: str,
    accessions_file: Optional[str],
    index_prompt: str,
    definitions_prompt: str,
    verification_prompt: Optional[str],
    model: Optional[str],
    gateway_url: Optional[str],
    temperature: float,
    reasoning: Optional[str],
    gateway_timeout: float,
    concurrency: int,
    bandwidth: int,
    definitions_subdir: str,
    verification_subdir: str,
):
    """End-to-end TOC-based flow: ingest -> normalize -> index -> retrieve -> definitions -> (optional) verification."""

    rc = resolve_run_config(run_id, base_dir, workers=4, bandwidth=bandwidth, namespace=namespace)
    paths = rc.paths()

    # Enforce required model/reasoning regardless of CLI input.
    model = REQUIRED_MODEL
    reasoning = REQUIRED_REASONING

    # Ingest
    accessions, spec, doc_filter = _load_accessions_and_filters(filters_path, accessions_file)
    ingest_tarballs(paths, [Path(t) for t in tarball], spec, accessions, doc_filter=doc_filter)

    # Normalize
    manifest = load_manifest(paths.manifest_path)
    build_prompt_views(paths, manifest)

    # Index (TOC/anchors)
    items = manifest_items(manifest)
    item_ids = [item["item_id"] for item in items]
    run_indexing(
        paths,
        item_ids,
        Path(index_prompt),
        model=REQUIRED_MODEL,
        gateway_url=gateway_url,
        temperature=temperature,
        reasoning=REQUIRED_REASONING,
        gateway_timeout=gateway_timeout,
        concurrency=concurrency,
    )

    # Retrieval
    render_snippets(paths, item_ids, bandwidth=bandwidth)

    # Definitions extraction
    run_structured(
        paths,
        item_ids,
        Path(definitions_prompt),
        model=REQUIRED_MODEL,
        gateway_url=gateway_url,
        temperature=temperature,
        reasoning=REQUIRED_REASONING,
        gateway_timeout=gateway_timeout,
        concurrency=concurrency,
        output_subdir=definitions_subdir,
    )

    # Optional verification
    if verification_prompt:
        run_structured(
            paths,
            item_ids,
            Path(verification_prompt),
            model=REQUIRED_MODEL,
            gateway_url=gateway_url,
            temperature=temperature,
            reasoning=REQUIRED_REASONING,
            gateway_timeout=gateway_timeout,
            concurrency=concurrency,
            output_subdir=verification_subdir,
        )


if __name__ == "__main__":
    main()
