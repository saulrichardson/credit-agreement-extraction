from __future__ import annotations

from pathlib import Path
from typing import Optional

import click

from .config import FilterSpec, RunConfig, REQUIRED_MODEL, REQUIRED_REASONING
from .filters import load_filter_spec, load_doc_filter
from .ingest import ingest_tarballs
from .normalize import build_prompt_views
from .structured_v2 import run_structured_v2
from .contract_pricing import run_contract_pricing
from .retrieval_v2 import render_snippets_v2
from .utils import load_manifest, manifest_items, read_accessions_file
from .run_id import validate_run_id
from .indexing_v2 import run_indexing_v2
from .contract_ir_v0_2_cli import contractir_v0_2_cli


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


# Separate namespace for the imposed-structure pricing-kernel approach (ContractIR v0.2).
cli.add_command(contractir_v0_2_cli)


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


@cli.command(name="index-v2")
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
    help="Gateway client timeout (seconds) for indexing v2 calls.",
)
@click.option(
    "--concurrency",
    default=3,
    show_default=True,
    type=int,
    help="Number of parallel gateway calls during indexing v2 (only when --model is set).",
)
@click.option(
    "--attempts",
    default=3,
    show_default=True,
    type=int,
    help="Strict JSON/Pydantic attempts per item before failing.",
)
@click.option(
    "--skip-existing",
    is_flag=True,
    help="Skip items with existing runs/<run_id>/indexing_v2/<item_id>_anchors.json outputs.",
)
@click.option(
    "--item-id",
    "item_ids",
    multiple=True,
    help="Optional item_id(s) to index (defaults to all items in the run manifest).",
)
def index_v2(
    run_id: str,
    prompt_path: str,
    base_dir: str,
    model: str,
    gateway_url: str,
    temperature: float,
    reasoning: str | None,
    gateway_timeout: float,
    concurrency: int,
    attempts: int,
    skip_existing: bool,
    item_ids: tuple[str, ...],
):
    """Run anchor indexing via agent-gateway (v2 schema)."""
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
    run_indexing_v2(
        paths,
        selected_item_ids,
        Path(prompt_path),
        model=model,
        gateway_url=gateway_url,
        temperature=temperature,
        reasoning=reasoning,
        gateway_timeout=gateway_timeout,
        concurrency=concurrency,
        attempts=attempts,
        skip_existing=bool(skip_existing),
    )

@cli.command(name="retrieve-v2")
@click.option("--run-id", required=True, callback=validate_run_id)
@click.option("--bandwidth", default=400, show_default=True, type=int)
@click.option("--base-dir", default=".", show_default=True)
@click.option(
    "--item-id",
    "item_ids",
    multiple=True,
    help="Optional item_id(s) to retrieve snippets for (defaults to all items in the run manifest).",
)
def retrieve_v2(run_id: str, bandwidth: int, base_dir: str, item_ids: tuple[str, ...]):
    """Render snippets around anchors using indexing v2 outputs."""
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
    render_snippets_v2(paths, selected_item_ids, bandwidth=bandwidth)


@cli.command(name="structured-v2")
@click.option("--run-id", required=True, callback=validate_run_id)
@click.option("--prompt", "prompt_path", type=click.Path(exists=True, dir_okay=False), required=True)
@click.option("--base-dir", default=".", show_default=True)
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
    "--attempts",
    default=3,
    show_default=True,
    type=int,
    help="Strict JSON attempts per item before failing.",
)
@click.option(
    "--output-subdir",
    default=None,
    help="Optional subfolder under runs/<run_id>/llm_qa/ for outputs (defaults to prompt filename stem).",
)
@click.option(
    "--category",
    "categories",
    multiple=True,
    help="Optional category filter (repeatable). Example: --category financial_covenant",
)
@click.option(
    "--item-id",
    "item_ids",
    multiple=True,
    help="Optional item_id(s) to run structured extraction for (defaults to all items in the run manifest).",
)
def structured_v2(
    run_id: str,
    prompt_path: str,
    base_dir: str,
    model: str | None,
    gateway_url: str | None,
    temperature: float,
    reasoning: str | None,
    gateway_timeout: float,
    concurrency: int,
    attempts: int,
    output_subdir: str | None,
    categories: tuple[str, ...],
    item_ids: tuple[str, ...],
):
    """Structured extraction over v2 retrieval snippets (strict JSON)."""
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
    run_structured_v2(
        paths,
        selected_item_ids,
        Path(prompt_path),
        model=model,
        gateway_url=gateway_url,
        temperature=temperature,
        reasoning=reasoning,
        gateway_timeout=gateway_timeout,
        concurrency=concurrency,
        attempts=attempts,
        output_subdir=output_subdir,
        categories=categories,
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


@cli.command(name="contract-pricing-v3")
@click.option("--run-id", required=True, callback=validate_run_id)
@click.option(
    "--planner-prompt",
    "planner_prompt_path",
    type=click.Path(exists=True, dir_okay=False),
    default="prompts/contract_pricing_planner_v3_table_census.txt",
    show_default=True,
    help="Prompt used by the agentic planner to select pricing tables + axis semantics (no heuristics).",
)
@click.option(
    "--table-prompt",
    "table_prompt_path",
    type=click.Path(exists=True, dir_okay=False),
    default="prompts/contract_pricing_table_v2_noheuristics.txt",
    show_default=True,
    help="Prompt used to compile a single pricing table into the strict schema (no heuristic hints).",
)
@click.option("--base-dir", default=".", show_default=True)
@click.option("--gateway-url", default=None, help="Gateway base URL; defaults to $GATEWAY_URL.")
@click.option("--planner-model", default="openai:gpt-5-mini", show_default=True)
@click.option(
    "--planner-mode",
    default="tool",
    show_default=True,
    type=click.Choice(["tool", "table_pack", "semantic_scan"], case_sensitive=False),
    help="Planning mode: tool loop, one-shot table pack, or coverage-first semantic scan.",
)
@click.option(
    "--planner-reasoning",
    default="high",
    show_default=True,
    type=click.Choice(["none", "minimal", "low", "medium", "high", "xhigh"], case_sensitive=False),
)
@click.option("--planner-temperature", default=0.0, show_default=True, type=float)
@click.option("--planner-max-steps", default=40, show_default=True, type=int)
@click.option("--planner-attempts-per-step", default=3, show_default=True, type=int)
@click.option(
    "--scan-prompt",
    "scan_prompt_path",
    type=click.Path(exists=True, dir_okay=False),
    default="prompts/semantic_pricing_scan_chunk_v1.txt",
    show_default=True,
    help="Prompt used for coverage-first semantic scan chunks (only used in planner-mode=semantic_scan).",
)
@click.option(
    "--scan-model",
    default=None,
    help="Optional model for the semantic scan reader. Defaults to --planner-model when omitted.",
)
@click.option(
    "--scan-reasoning",
    default="high",
    show_default=True,
    type=click.Choice(["none", "minimal", "low", "medium", "high", "xhigh"], case_sensitive=False),
)
@click.option("--scan-temperature", default=0.0, show_default=True, type=float)
@click.option(
    "--scan-max-chunk-chars",
    default=20000,
    show_default=True,
    type=int,
    help="Max characters per scan chunk (structural chunking, no heuristics).",
)
@click.option("--scan-overlap-anchors", default=5, show_default=True, type=int)
@click.option("--scan-max-anchors-per-chunk", default=120, show_default=True, type=int)
@click.option("--scan-attempts-per-chunk", default=3, show_default=True, type=int)
@click.option(
    "--critic-prompt",
    "critic_prompt_path",
    type=click.Path(exists=True, dir_okay=False),
    default="prompts/contract_pricing_table_critic_v1.txt",
    show_default=True,
    help="Prompt used by the optional critic to audit a table extract against context.",
)
@click.option(
    "--critic-model",
    default=None,
    help="Optional model to run a critique loop for each table extract (improves fidelity; costs more).",
)
@click.option(
    "--critic-reasoning",
    default="high",
    show_default=True,
    type=click.Choice(["none", "minimal", "low", "medium", "high", "xhigh"], case_sensitive=False),
)
@click.option("--critic-temperature", default=0.0, show_default=True, type=float)
@click.option("--compiler-temperature", default=0.0, show_default=True, type=float)
@click.option(
    "--gateway-timeout",
    default=900.0,
    show_default=True,
    type=float,
    help="Gateway client timeout (seconds) for planner + table compiler calls.",
)
@click.option(
    "--concurrency",
    default=1,
    show_default=True,
    type=int,
    help="Number of parallel gateway calls (per-item) during contract pricing v3 extraction.",
)
@click.option(
    "--attempts",
    default=3,
    show_default=True,
    type=int,
    help="Number of strict JSON/Pydantic attempts per table before failing.",
)
@click.option(
    "--output-subdir",
    default="contract_pricing_v3",
    show_default=True,
    help="Subfolder under runs/<run_id>/contract_pricing/ for outputs.",
)
@click.option(
    "--item-id",
    "item_ids",
    multiple=True,
    help="Optional item_id(s) to extract pricing for (defaults to all items in the run manifest).",
)
def contract_pricing_v3(
    run_id: str,
    planner_prompt_path: str,
    table_prompt_path: str,
    base_dir: str,
    gateway_url: str | None,
    planner_model: str,
    planner_mode: str,
    planner_reasoning: str,
    planner_temperature: float,
    planner_max_steps: int,
    planner_attempts_per_step: int,
    scan_prompt_path: str,
    scan_model: str | None,
    scan_reasoning: str,
    scan_temperature: float,
    scan_max_chunk_chars: int,
    scan_overlap_anchors: int,
    scan_max_anchors_per_chunk: int,
    scan_attempts_per_chunk: int,
    critic_prompt_path: str,
    critic_model: str | None,
    critic_reasoning: str,
    critic_temperature: float,
    compiler_temperature: float,
    gateway_timeout: float,
    concurrency: int,
    attempts: int,
    output_subdir: str,
    item_ids: tuple[str, ...],
):
    """Extract pricing regimes with a no-heuristics agentic planner + strict per-table compilation."""

    from .contract_pricing_v3 import run_contract_pricing_v3

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

    run_contract_pricing_v3(
        paths,
        selected_item_ids,
        planner_prompt_path=Path(planner_prompt_path),
        table_prompt_path=Path(table_prompt_path),
        gateway_url=gateway_url,
        planner_model=planner_model,
        planner_mode=str(planner_mode or "tool").strip().lower(),  # type: ignore[arg-type]
        planner_reasoning=planner_reasoning,
        planner_temperature=planner_temperature,
        planner_max_steps=planner_max_steps,
        planner_attempts_per_step=planner_attempts_per_step,
        scan_prompt_path=Path(scan_prompt_path),
        scan_model=scan_model,
        scan_reasoning=scan_reasoning,
        scan_temperature=scan_temperature,
        scan_max_chunk_chars=scan_max_chunk_chars,
        scan_overlap_anchors=scan_overlap_anchors,
        scan_max_anchors_per_chunk=scan_max_anchors_per_chunk,
        scan_attempts_per_chunk=scan_attempts_per_chunk,
        critic_prompt_path=Path(critic_prompt_path) if critic_model else None,
        critic_model=critic_model,
        critic_reasoning=critic_reasoning,
        critic_temperature=critic_temperature,
        compiler_temperature=compiler_temperature,
        gateway_timeout=gateway_timeout,
        concurrency=concurrency,
        attempts=attempts,
        output_subdir=output_subdir,
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


@cli.command(name="definitions-v2")
@click.option("--run-id", required=True, callback=validate_run_id)
@click.option("--qa-subdir", default="dg-v2-copy", show_default=True, help="llm_qa subdir with dg outputs.")
@click.option(
    "--definitions-prompt",
    "definitions_prompt_path",
    type=click.Path(exists=True, dir_okay=False),
    required=True,
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
    help="Gateway client timeout (seconds) for definitions calls.",
)
@click.option(
    "--concurrency",
    default=3,
    show_default=True,
    type=int,
    help="Number of parallel gateway calls during definitions.",
)
@click.option(
    "--output-subdir",
    default="definitions_v2",
    show_default=True,
    help="Subfolder under runs/<run_id>/definitions_v2/ for outputs.",
)
@click.option(
    "--item-id",
    "item_ids",
    multiple=True,
    help="Optional item_id(s) to run definitions for (defaults to all items in the run manifest).",
)
def definitions_v2(
    run_id: str,
    qa_subdir: str,
    definitions_prompt_path: str,
    base_dir: str,
    model: str | None,
    gateway_url: str | None,
    temperature: float,
    reasoning: str | None,
    gateway_timeout: float,
    concurrency: int,
    output_subdir: str,
    item_ids: tuple[str, ...],
):
    """Resolve metric and rate term definitions using dg outputs (v2)."""

    from .definitions_v2 import run_definitions_v2

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

    run_definitions_v2(
        paths,
        selected_item_ids,
        qa_subdir=qa_subdir,
        definitions_prompt_path=Path(definitions_prompt_path),
        model=model,
        gateway_url=gateway_url,
        temperature=temperature,
        reasoning=reasoning,
        gateway_timeout=gateway_timeout,
        concurrency=concurrency,
        output_subdir=output_subdir,
    )


@cli.command(name="definitions-compiler-v1")
@click.option("--run-id", required=True, callback=validate_run_id)
@click.option("--qa-subdir", required=True, help="llm_qa subdir with dg outputs.")
@click.option(
    "--compiler-prompt",
    "compiler_prompt_path",
    type=click.Path(exists=True, dir_okay=False),
    required=True,
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
    help="Gateway client timeout (seconds) for compiler calls.",
)
@click.option(
    "--concurrency",
    default=2,
    show_default=True,
    type=int,
    help="Number of parallel gateway calls during compilation (metric-level).",
)
@click.option(
    "--attempts",
    default=3,
    show_default=True,
    type=int,
    help="Retry attempts per metric when the gateway returns empty/throws.",
)
@click.option(
    "--output-subdir",
    default="compiled_v1",
    show_default=True,
    help="Subfolder under runs/<run_id>/definitions_compiler_v1/ for outputs.",
)
@click.option(
    "--item-id",
    "item_ids",
    multiple=True,
    help="Optional item_id(s) to run compiler for (defaults to all items in the run manifest).",
)
def definitions_compiler_v1(
    run_id: str,
    qa_subdir: str,
    compiler_prompt_path: str,
    base_dir: str,
    model: str | None,
    gateway_url: str | None,
    temperature: float,
    reasoning: str | None,
    gateway_timeout: float,
    concurrency: int,
    attempts: int,
    output_subdir: str,
    item_ids: tuple[str, ...],
):
    """Compile metric definitions (verbatim + AST + dependency terms) from canonical text and indexing hints."""

    from .definitions_compiler_v1 import run_definitions_compiler_v1

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

    run_definitions_compiler_v1(
        paths,
        selected_item_ids,
        qa_subdir=qa_subdir,
        compiler_prompt_path=Path(compiler_prompt_path),
        model=model,
        gateway_url=gateway_url,
        temperature=temperature,
        reasoning=reasoning,
        gateway_timeout=gateway_timeout,
        concurrency=concurrency,
        output_subdir=output_subdir,
        attempts=attempts,
    )


@cli.command(name="blocking-terms-compiler-v1")
@click.option("--run-id", required=True, callback=validate_run_id)
@click.option(
    "--metrics-output-subdir",
    required=True,
    help="definitions_compiler_v1 output subdir to read (e.g., compiled_v1 or compiled_metrics_v1_compustat_v3_full).",
)
@click.option(
    "--term-compiler-prompt",
    "term_compiler_prompt_path",
    type=click.Path(exists=True, dir_okay=False),
    required=True,
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
    help="Gateway client timeout (seconds) for term compiler calls.",
)
@click.option(
    "--concurrency",
    default=2,
    show_default=True,
    type=int,
    help="Number of parallel gateway calls during term compilation.",
)
@click.option(
    "--attempts",
    default=3,
    show_default=True,
    type=int,
    help="Retry attempts per blocking term when the gateway returns empty/throws.",
)
@click.option(
    "--max-depth",
    default=1,
    show_default=True,
    type=int,
    help="Max recursion depth over unresolved_dependencies (1 = only initial blocking terms).",
)
@click.option(
    "--max-terms",
    default=200,
    show_default=True,
    type=int,
    help="Max unique terms to compile per item (prevents runaway recursion).",
)
@click.option(
    "--output-subdir",
    default="blocking_terms_v1",
    show_default=True,
    help="Subfolder under runs/<run_id>/blocking_terms_compiler_v1/ for outputs.",
)
@click.option(
    "--item-id",
    "item_ids",
    multiple=True,
    help="Optional item_id(s) to compile blocking terms for (defaults to all items in the run manifest).",
)
def blocking_terms_compiler_v1(
    run_id: str,
    metrics_output_subdir: str,
    term_compiler_prompt_path: str,
    base_dir: str,
    model: str | None,
    gateway_url: str | None,
    temperature: float,
    reasoning: str | None,
    gateway_timeout: float,
    concurrency: int,
    attempts: int,
    max_depth: int,
    max_terms: int,
    output_subdir: str,
    item_ids: tuple[str, ...],
):
    """Resolve dependency terms recursively from metric definitions using whole-document contexts."""

    from .blocking_terms_compiler_v1 import run_blocking_terms_compiler_v1

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

    run_blocking_terms_compiler_v1(
        paths,
        selected_item_ids,
        metrics_output_subdir=metrics_output_subdir,
        term_compiler_prompt_path=Path(term_compiler_prompt_path),
        model=model,
        gateway_url=gateway_url,
        temperature=temperature,
        reasoning=reasoning,
        gateway_timeout=gateway_timeout,
        concurrency=concurrency,
        attempts=attempts,
        output_subdir=output_subdir,
        max_depth=max_depth,
        max_terms=max_terms,
    )


@cli.command(name="compustat-overlay-v1")
@click.option("--run-id", required=True, callback=validate_run_id)
@click.option(
    "--overlay-prompt",
    "overlay_prompt_path",
    type=click.Path(exists=True, dir_okay=False),
    required=True,
)
@click.option(
    "--compustat-allowlist",
    "compustat_allowlist_path",
    type=click.Path(exists=True, dir_okay=False),
    default=None,
    help=(
        "Path to Compustat variable allowlist JSON (defaults to datasets/compustat_allowlist_quarterly_v1.json). "
        "See schema in that file."
    ),
)
@click.option(
    "--metrics-output-subdir",
    default=None,
    help="definitions_compiler_v1 output subdir to read (item-level __compiled.json).",
)
@click.option(
    "--blocking-terms-output-subdir",
    default=None,
    help="blocking_terms_compiler_v1 output subdir to read (item-level __compiled.json).",
)
@click.option("--base-dir", default=".", show_default=True)
@click.option("--gateway-url", default=None, help="Gateway base URL; defaults to $GATEWAY_URL.")
@click.option("--temperature", default=0.0, show_default=True, type=float)
@click.option(
    "--gateway-timeout",
    default=600.0,
    show_default=True,
    type=float,
    help="Gateway client timeout (seconds) for overlay calls.",
)
@click.option(
    "--concurrency",
    default=4,
    show_default=True,
    type=int,
    help="Number of parallel gateway calls during overlay.",
)
@click.option(
    "--attempts",
    default=3,
    show_default=True,
    type=int,
    help="Retry attempts per term overlay when the gateway returns empty/throws/invalid JSON.",
)
@click.option(
    "--output-subdir",
    default="compustat_overlay_v1",
    show_default=True,
    help="Subfolder under runs/<run_id>/compustat_overlay_v1/ for outputs.",
)
@click.option(
    "--item-id",
    "item_ids",
    multiple=True,
    help="Optional item_id(s) to run overlay for (defaults to all items in the run manifest).",
)
def compustat_overlay_v1(
    run_id: str,
    overlay_prompt_path: str,
    compustat_allowlist_path: str | None,
    metrics_output_subdir: str | None,
    blocking_terms_output_subdir: str | None,
    base_dir: str,
    gateway_url: str | None,
    temperature: float,
    gateway_timeout: float,
    concurrency: int,
    attempts: int,
    output_subdir: str,
    item_ids: tuple[str, ...],
):
    """Map extracted v2 AST definitions to Compustat formulas (overlay stage)."""

    from .compustat_overlay_v1 import run_compustat_overlay_v1

    if not metrics_output_subdir and not blocking_terms_output_subdir:
        raise click.UsageError("Must provide at least one of --metrics-output-subdir or --blocking-terms-output-subdir.")

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

    run_compustat_overlay_v1(
        paths,
        selected_item_ids,
        overlay_prompt_path=Path(overlay_prompt_path),
        allowlist_path=Path(compustat_allowlist_path) if compustat_allowlist_path else None,
        output_subdir=output_subdir,
        metrics_output_subdir=metrics_output_subdir,
        blocking_terms_output_subdir=blocking_terms_output_subdir,
        gateway_url=gateway_url,
        temperature=temperature,
        gateway_timeout=gateway_timeout,
        concurrency=concurrency,
        attempts=attempts,
    )


@cli.command(name="facility-fundamentals")
@click.option("--run-id", required=True, callback=validate_run_id)
@click.option(
    "--prompt",
    "prompt_path",
    type=click.Path(exists=True, dir_okay=False),
    default="prompts/facility_fundamentals_v1.txt",
    show_default=True,
    help="Prompt for facility fundamentals extraction (LLM-driven; strict JSON).",
)
@click.option("--base-dir", default=".", show_default=True)
@click.option(
    "--model",
    default=REQUIRED_MODEL,
    show_default=True,
    help="Gateway model for facility fundamentals extraction (enforced).",
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
    help="Gateway client timeout (seconds) for facility fundamentals calls.",
)
@click.option(
    "--concurrency",
    default=3,
    show_default=True,
    type=int,
    help="Number of parallel gateway calls during facility fundamentals extraction.",
)
@click.option(
    "--attempts",
    default=3,
    show_default=True,
    type=int,
    help="Strict JSON/Pydantic attempts per item before failing.",
)
@click.option(
    "--output-subdir",
    default=None,
    help="Optional subfolder under runs/<run_id>/facility_fundamentals/ for outputs (defaults to prompt filename stem).",
)
@click.option(
    "--category",
    "categories",
    multiple=True,
    help="Optional snippet category filter (repeatable). Default: agreement_dates + fundamental + key_date_definitions.",
)
@click.option(
    "--item-id",
    "item_ids",
    multiple=True,
    help="Optional item_id(s) to run facility fundamentals for (defaults to all items in the run manifest).",
)
def facility_fundamentals(
    run_id: str,
    prompt_path: str,
    base_dir: str,
    model: str | None,
    gateway_url: str | None,
    temperature: float,
    reasoning: str | None,
    gateway_timeout: float,
    concurrency: int,
    attempts: int,
    output_subdir: str | None,
    categories: tuple[str, ...],
    item_ids: tuple[str, ...],
):
    """Extract facility fundamentals (dates + facility separation) from v2 retrieval snippets."""

    from .facility_fundamentals_v1 import run_facility_fundamentals_v1

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

    selected_categories = categories if categories else ("agreement_dates", "fundamental", "key_date_definitions")

    run_facility_fundamentals_v1(
        paths,
        selected_item_ids,
        Path(prompt_path),
        categories=selected_categories,
        model=model,
        gateway_url=gateway_url,
        temperature=temperature,
        reasoning=reasoning,
        gateway_timeout=gateway_timeout,
        concurrency=concurrency,
        attempts=attempts,
        output_subdir=output_subdir,
    )


@cli.command(name="agreement-metadata")
@click.option("--run-id", required=True, callback=validate_run_id)
@click.option("--prompt", "prompt_path", type=click.Path(exists=True, dir_okay=False), required=True)
@click.option("--base-dir", default=".", show_default=True)
@click.option(
    "--model",
    default=REQUIRED_MODEL,
    show_default=True,
    help="Gateway model for agreement metadata extraction (enforced).",
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
    help="Gateway client timeout (seconds) for agreement metadata calls.",
)
@click.option(
    "--concurrency",
    default=3,
    show_default=True,
    type=int,
    help="Number of parallel gateway calls during agreement metadata extraction.",
)
@click.option(
    "--attempts",
    default=3,
    show_default=True,
    type=int,
    help="Strict JSON/Pydantic attempts per item before failing.",
)
@click.option(
    "--output-subdir",
    default=None,
    help="Optional subfolder under runs/<run_id>/agreement_metadata/ for outputs (defaults to prompt filename stem).",
)
@click.option(
    "--category",
    "categories",
    multiple=True,
    help="Optional snippet category filter (repeatable). Default: metadata + fundamental.",
)
@click.option(
    "--item-id",
    "item_ids",
    multiple=True,
    help="Optional item_id(s) to run agreement metadata for (defaults to all items in the run manifest).",
)
def agreement_metadata(
    run_id: str,
    prompt_path: str,
    base_dir: str,
    model: str | None,
    gateway_url: str | None,
    temperature: float,
    reasoning: str | None,
    gateway_timeout: float,
    concurrency: int,
    attempts: int,
    output_subdir: str | None,
    categories: tuple[str, ...],
    item_ids: tuple[str, ...],
):
    """Extract parties/roles + facility headline terms from v2 retrieval snippets."""

    from .agreement_metadata_v1 import run_agreement_metadata_v1

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

    selected_categories = categories if categories else ("metadata", "fundamental")

    run_agreement_metadata_v1(
        paths,
        selected_item_ids,
        Path(prompt_path),
        categories=selected_categories,
        model=model,
        gateway_url=gateway_url,
        temperature=temperature,
        reasoning=reasoning,
        gateway_timeout=gateway_timeout,
        concurrency=concurrency,
        attempts=attempts,
        output_subdir=output_subdir,
    )


@cli.command(name="analysis-export")
@click.option("--run-id", required=True, callback=validate_run_id)
@click.option("--base-dir", default=".", show_default=True)
@click.option(
    "--agreement-metadata-subdir",
    required=True,
    help="Subfolder under runs/<run_id>/agreement_metadata/ containing <item_id>.json outputs.",
)
@click.option(
    "--definitions-v2-subdir",
    required=True,
    help="Subfolder under runs/<run_id>/definitions_v2/ containing per-term __definition.json outputs.",
)
@click.option(
    "--output-subdir",
    default="analysis_export_v1",
    show_default=True,
    help="Output subfolder under runs/<run_id>/analysis_export/.",
)
@click.option(
    "--item-id",
    "item_ids",
    multiple=True,
    help="Optional item_id(s) to export (defaults to all items in the run manifest).",
)
def analysis_export(
    run_id: str,
    base_dir: str,
    agreement_metadata_subdir: str,
    definitions_v2_subdir: str,
    output_subdir: str,
    item_ids: tuple[str, ...],
):
    """Export a single analysis-ready JSON record per agreement."""

    from .analysis_export_v1 import run_analysis_export_v1

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

    run_analysis_export_v1(
        paths,
        selected_item_ids,
        agreement_metadata_subdir=agreement_metadata_subdir,
        definitions_v2_subdir=definitions_v2_subdir,
        output_subdir=output_subdir,
    )


@cli.command(name="toc-v1")
@click.option("--run-id", required=True, callback=validate_run_id)
@click.option(
    "--prompt-chunk",
    "prompt_chunk_path",
    type=click.Path(exists=True, dir_okay=False),
    required=True,
    help="Chunk-level TOC prompt (generates a title/summary/topics per chunk).",
)
@click.option(
    "--prompt-high-level",
    "prompt_high_level_path",
    type=click.Path(exists=True, dir_okay=False),
    required=False,
    help="Optional prompt to build a high-level doc map from the first few chunk TOC entries.",
)
@click.option("--base-dir", default=".", show_default=True)
@click.option("--gateway-url", default=None, help="Gateway base URL; defaults to $GATEWAY_URL.")
@click.option("--temperature", default=0.0, show_default=True, type=float)
@click.option(
    "--reasoning",
    default=REQUIRED_REASONING,
    type=click.Choice(["light", "medium", "heavy"], case_sensitive=False),
    show_default=True,
    help="Reasoning effort (enforced).",
)
@click.option("--gateway-timeout", default=600.0, show_default=True, type=float)
@click.option("--concurrency", default=3, show_default=True, type=int)
@click.option("--attempts", default=3, show_default=True, type=int)
@click.option("--output-subdir", default=None, help="Subfolder under runs/<run_id>/toc_v1/ for outputs.")
@click.option("--target-chars", default=14000, show_default=True, type=int)
@click.option("--min-chars", default=8000, show_default=True, type=int)
@click.option("--max-chars", default=22000, show_default=True, type=int)
@click.option("--lookback-anchors", default=40, show_default=True, type=int)
@click.option("--high-level-chunks", default=5, show_default=True, type=int)
@click.option(
    "--item-id",
    "item_ids",
    multiple=True,
    help="Optional item_id(s) to run toc-v1 for (defaults to all items in the run manifest).",
)
def toc_v1(
    run_id: str,
    prompt_chunk_path: str,
    prompt_high_level_path: str | None,
    base_dir: str,
    gateway_url: str | None,
    temperature: float,
    reasoning: str | None,
    gateway_timeout: float,
    concurrency: int,
    attempts: int,
    output_subdir: str | None,
    target_chars: int,
    min_chars: int,
    max_chars: int,
    lookback_anchors: int,
    high_level_chunks: int,
    item_ids: tuple[str, ...],
):
    """Build a TOC via adaptive chunking + LLM summaries (no embeddings)."""

    from .toc_v1 import run_toc_v1

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

    run_toc_v1(
        paths,
        selected_item_ids,
        prompt_chunk_path=Path(prompt_chunk_path),
        prompt_high_level_path=Path(prompt_high_level_path) if prompt_high_level_path else None,
        output_subdir=output_subdir,
        target_chars=target_chars,
        min_chars=min_chars,
        max_chars=max_chars,
        lookback_anchors=lookback_anchors,
        high_level_chunks=high_level_chunks,
        gateway_url=gateway_url,
        temperature=temperature,
        reasoning=reasoning,
        gateway_timeout=gateway_timeout,
        concurrency=concurrency,
        attempts=attempts,
    )


@cli.command(name="all-v2")
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
    item_ids: tuple[str, ...],
):
    """Run ingest -> normalize -> toc-v1 -> index-v2 -> retrieve-v2 -> structured-v2 -> definitions-v2 -> agreement-metadata -> analysis-export."""

    from .analysis_export_v1 import run_analysis_export_v1
    from .agreement_metadata_v1 import run_agreement_metadata_v1
    from .definitions_v2 import run_definitions_v2
    from .indexing_v2 import run_indexing_v2
    from .retrieval_v2 import render_snippets_v2
    from .structured_v2 import run_structured_v2
    from .toc_v1 import run_toc_v1

    rc = _resolve_paths(run_id, base_dir, bandwidth=4)
    paths = rc.paths()

    accessions, spec, doc_filter = _load_accessions_and_filters(filters_path, accessions_file)

    ingest_tarballs(paths, [Path(t) for t in tarball], spec, accessions, doc_filter=doc_filter)
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


def main():
    cli()


if __name__ == "__main__":
    main()
