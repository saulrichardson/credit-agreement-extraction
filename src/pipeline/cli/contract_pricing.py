from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import click

from ..contract_pricing import run_contract_pricing
from ..ingest import ingest_tarballs
from ..normalize import build_prompt_views
from ..run_id import validate_run_id
from ..utils import load_manifest, manifest_items
from .common import load_accessions_and_filters, resolve_paths, resolve_selected_item_ids


@click.command(name="contract-pricing")
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
    item_ids: Tuple[str, ...],
) -> None:
    """Extract pricing regimes into a contract-like IR (doc IR -> LLM compile)."""

    paths = resolve_paths(run_id, base_dir, bandwidth=4)
    _manifest, _items, selected_item_ids = resolve_selected_item_ids(paths, item_ids)

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


@click.command(name="contract-pricing-v3")
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
    item_ids: Tuple[str, ...],
) -> None:
    """Extract pricing regimes with a no-heuristics agentic planner + strict per-table compilation."""

    from ..contract_pricing_v3 import run_contract_pricing_v3

    paths = resolve_paths(run_id, base_dir, bandwidth=4)
    _manifest, _items, selected_item_ids = resolve_selected_item_ids(paths, item_ids)

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


@click.command(name="contract-pricing-flow")
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
    item_ids: Tuple[str, ...],
) -> None:
    """Run ingest -> normalize -> contract-pricing (no indexing/retrieval)."""

    paths = resolve_paths(run_id, base_dir, bandwidth=4)

    accessions, spec, doc_filter = load_accessions_and_filters(filters_path, accessions_file)
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

