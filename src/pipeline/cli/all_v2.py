from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import click

from pipeline.core.config import REQUIRED_REASONING
from pipeline.evidence.ingest import ingest_tarballs
from pipeline.evidence.normalize import build_prompt_views
from pipeline.core.run_id import validate_run_id
from pipeline.evidence.toc_v1 import run_toc_v1
from .common import load_accessions_and_filters, resolve_paths, resolve_selected_item_ids


@click.command(name="all-v2")
@click.option("--run-id", required=True, callback=validate_run_id)
@click.option("--tarball", multiple=True, type=click.Path(exists=True, dir_okay=False), required=True)
@click.option("--accessions-file", type=click.Path(exists=True, dir_okay=False), required=False)
@click.option(
    "--prompt-index-v2",
    type=click.Path(exists=True, dir_okay=False),
    default="prompts/indexing_v2.txt",
    show_default=True,
    help="Prompt for index-v2 anchor selection.",
)
@click.option(
    "--prompt-structured-v2",
    type=click.Path(exists=True, dir_okay=False),
    default="prompts/prompt_pricing_second_pass_dg_nano_v2_tuned_v2.txt",
    show_default=True,
    help="Prompt for structured-v2 (dg) extraction.",
)
@click.option(
    "--prompt-definitions-v2",
    type=click.Path(exists=True, dir_okay=False),
    default="prompts/definitions_v2_metrics_rates.txt",
    show_default=True,
    help="Prompt for definitions-v2 extraction.",
)
@click.option(
    "--prompt-agreement-metadata",
    type=click.Path(exists=True, dir_okay=False),
    default="prompts/agreement_metadata_v1.txt",
    show_default=True,
    help="Prompt for agreement-metadata extraction.",
)
@click.option(
    "--prompt-toc-chunk",
    type=click.Path(exists=True, dir_okay=False),
    default="prompts/toc_chunk_v1.txt",
    show_default=True,
    help="Prompt for toc-v1 chunk summaries.",
)
@click.option(
    "--prompt-toc-high-level",
    type=click.Path(exists=True, dir_okay=False),
    default="prompts/toc_high_level_v1.txt",
    show_default=True,
    help="Prompt for toc-v1 high-level index (built from the first few chunk summaries).",
)
@click.option("--bandwidth", default=400, show_default=True, type=int, help="Snippet window size for retrieve-v2.")
@click.option("--base-dir", default=".", show_default=True)
@click.option(
    "--filters",
    "filters_path",
    type=click.Path(exists=True, dir_okay=False),
    required=False,
    help="Filter spec (JSON/YAML with doc_filter_path and optional doc_filter_kwargs). Defaults to keep_all when omitted.",
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
    help="Gateway client timeout (seconds) for all gateway calls in the flow.",
)
@click.option(
    "--concurrency",
    default=3,
    show_default=True,
    type=int,
    help="Number of parallel gateway calls during LLM stages.",
)
@click.option(
    "--structured-output-subdir",
    default=None,
    help="Optional llm_qa subdir for structured-v2 outputs (defaults to structured prompt stem).",
)
@click.option(
    "--definitions-output-subdir",
    default=None,
    help="Optional definitions_v2 subdir (defaults to structured output subdir).",
)
@click.option(
    "--agreement-metadata-output-subdir",
    default=None,
    help="Optional agreement_metadata output subdir (defaults to agreement-metadata prompt stem).",
)
@click.option(
    "--analysis-output-subdir",
    default="analysis_export_v1",
    show_default=True,
    help="Output subfolder under runs/<run_id>/analysis_export/.",
)
@click.option(
    "--item-id",
    "item_ids",
    multiple=True,
    help="Optional item_id(s) to run the flow for (defaults to all items in the run manifest).",
)
def all_v2(
    run_id: str,
    tarball,
    accessions_file: Optional[str],
    prompt_index_v2: str,
    prompt_structured_v2: str,
    prompt_definitions_v2: str,
    prompt_agreement_metadata: str,
    prompt_toc_chunk: str,
    prompt_toc_high_level: str,
    bandwidth: int,
    base_dir: str,
    filters_path: Optional[str],
    gateway_url: str | None,
    temperature: float,
    reasoning: str | None,
    gateway_timeout: float,
    concurrency: int,
    structured_output_subdir: str | None,
    definitions_output_subdir: str | None,
    agreement_metadata_output_subdir: str | None,
    analysis_output_subdir: str,
    item_ids: Tuple[str, ...],
) -> None:
    """Run ingest -> normalize -> toc-v1 -> index-v2 -> retrieve-v2 -> structured-v2 -> definitions-v2 -> agreement-metadata -> analysis-export."""

    from pipeline.extract.agreement_metadata_v1 import run_agreement_metadata_v1
    from pipeline.extract.analysis_export_v1 import run_analysis_export_v1
    from pipeline.extract.definitions_v2 import run_definitions_v2
    from pipeline.evidence.indexing_v2 import run_indexing_v2
    from pipeline.evidence.retrieval_v2 import render_snippets_v2
    from pipeline.extract.structured_v2 import run_structured_v2

    paths = resolve_paths(run_id, base_dir, bandwidth=4)

    accessions, spec, doc_filter = load_accessions_and_filters(filters_path, accessions_file)

    ingest_tarballs(paths, [Path(t) for t in tarball], spec, accessions, doc_filter=doc_filter)
    manifest, _items, selected_item_ids = resolve_selected_item_ids(paths, item_ids)

    # Normalize (canonical text + anchors + annotated view).
    build_prompt_views(paths, manifest)

    # TOC v1 (adaptive chunking + reasoning-first summaries). This runs before retrieval so
    # snippet packs can be tagged with toc_title/toc_chunk_id for higher-level organization.
    run_toc_v1(
        paths,
        selected_item_ids,
        prompt_chunk_path=Path(prompt_toc_chunk),
        prompt_high_level_path=Path(prompt_toc_high_level) if prompt_toc_high_level else None,
        gateway_url=gateway_url,
        temperature=temperature,
        reasoning=reasoning,
        gateway_timeout=gateway_timeout,
        concurrency=concurrency,
    )

    # Index + retrieve v2.
    run_indexing_v2(
        paths,
        selected_item_ids,
        Path(prompt_index_v2),
        gateway_url=gateway_url,
        temperature=temperature,
        reasoning=reasoning,
        gateway_timeout=gateway_timeout,
        concurrency=concurrency,
    )
    render_snippets_v2(paths, selected_item_ids, bandwidth=bandwidth)

    # Structured v2 (dg pass).
    qa_subdir = structured_output_subdir or Path(prompt_structured_v2).stem
    run_structured_v2(
        paths,
        selected_item_ids,
        Path(prompt_structured_v2),
        gateway_url=gateway_url,
        temperature=temperature,
        reasoning=reasoning,
        gateway_timeout=gateway_timeout,
        concurrency=concurrency,
        output_subdir=qa_subdir,
        categories=("metadata", "fundamental", "pricing", "financial_covenant"),
    )

    # Definitions v2 (uses dg outputs for contract_term/search_tokens).
    defs_subdir = definitions_output_subdir or qa_subdir
    run_definitions_v2(
        paths,
        selected_item_ids,
        qa_subdir=qa_subdir,
        definitions_prompt_path=Path(prompt_definitions_v2),
        gateway_url=gateway_url,
        temperature=temperature,
        reasoning=reasoning,
        gateway_timeout=gateway_timeout,
        concurrency=concurrency,
        output_subdir=defs_subdir,
    )

    # Agreement metadata (parties + facility headline terms).
    meta_subdir = agreement_metadata_output_subdir or Path(prompt_agreement_metadata).stem
    run_agreement_metadata_v1(
        paths,
        selected_item_ids,
        Path(prompt_agreement_metadata),
        gateway_url=gateway_url,
        temperature=temperature,
        reasoning=reasoning,
        gateway_timeout=gateway_timeout,
        concurrency=concurrency,
        output_subdir=meta_subdir,
    )

    # Final export for analysis (single record per agreement).
    run_analysis_export_v1(
        paths,
        selected_item_ids,
        agreement_metadata_subdir=meta_subdir,
        definitions_v2_subdir=defs_subdir,
        output_subdir=analysis_output_subdir,
    )

    click.echo(f"[all-v2] Completed v2 flow through analysis-export for {len(selected_item_ids)} exhibits.")

