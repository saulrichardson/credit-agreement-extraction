from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Optional, Tuple

import click

from pipeline.core.config import REQUIRED_REASONING
from pipeline.core.run_id import validate_run_id
from pipeline.evidence.ingest import ingest_tarballs
from pipeline.evidence.normalize import build_prompt_views
from .common import load_accessions_and_selection, resolve_paths, resolve_selected_item_ids


@click.command(name="all-v2-full")
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
    "--prompt-pricing-structured-v2",
    type=click.Path(exists=True, dir_okay=False),
    default="prompts/prompt_pricing_second_pass_dg_nano_v2_tuned_v2.txt",
    show_default=True,
    help="Prompt for pricing structured-v2 extraction.",
)
@click.option(
    "--prompt-covenant-structured-v2",
    type=click.Path(exists=True, dir_okay=False),
    default="prompts/prompt_v1_short.txt",
    show_default=True,
    help="Prompt for covenant structured-v2 extraction.",
)
@click.option(
    "--prompt-agreement-metadata",
    type=click.Path(exists=True, dir_okay=False),
    default="prompts/agreement_metadata_v1.txt",
    show_default=True,
    help="Prompt for agreement-metadata extraction.",
)
@click.option(
    "--prompt-metrics-compiler",
    type=click.Path(exists=True, dir_okay=False),
    default="prompts/definitions_compiler_v1_metrics_ast_v2.txt",
    show_default=True,
    help="Prompt for recursive metric definitions compiler (pricing + covenant).",
)
@click.option(
    "--prompt-blocking-terms-compiler",
    type=click.Path(exists=True, dir_okay=False),
    default="prompts/blocking_terms_compiler_v1_ast_v2.txt",
    show_default=True,
    help="Prompt for recursive dependency-term compiler (pricing + covenant).",
)
@click.option(
    "--prompt-compustat-overlay",
    type=click.Path(exists=True, dir_okay=False),
    default="prompts/compustat_overlay_v1.txt",
    show_default=True,
    help="Prompt for Compustat overlay compilation (pricing + covenant).",
)
@click.option("--bandwidth", default=400, show_default=True, type=int, help="Snippet window size for retrieve-v2.")
@click.option("--base-dir", default=".", show_default=True)
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
    "--attempts",
    default=3,
    show_default=True,
    type=int,
    help="Retry attempts per unit for strict JSON stages.",
)
@click.option(
    "--structured-output-subdir",
    default=None,
    help="Optional llm_qa subdir for pricing structured-v2 outputs (defaults to pricing prompt stem).",
)
@click.option(
    "--covenant-structured-output-subdir",
    default="covenant_simple_v1",
    show_default=True,
    help="Subfolder under runs/<run_id>/llm_qa/ for covenant outputs.",
)
@click.option(
    "--covenant-category",
    "covenant_categories",
    multiple=True,
    help="Optional snippet category filter for covenants (repeatable). Default: financial_covenant.",
)
@click.option(
    "--agreement-metadata-output-subdir",
    default=None,
    help="Optional agreement_metadata output subdir (defaults to agreement-metadata prompt stem).",
)
@click.option(
    "--pricing-metrics-output-subdir",
    default="compiled_pricing_metrics_recursive_ast_v2",
    show_default=True,
    help="Subfolder under runs/<run_id>/definitions_compiler_v1/ for pricing metric aggregates.",
)
@click.option(
    "--pricing-blocking-output-subdir",
    default="blocking_pricing_terms_recursive_ast_v2_depth1",
    show_default=True,
    help="Subfolder under runs/<run_id>/blocking_terms_compiler_v1/ for pricing dependency-term aggregates.",
)
@click.option(
    "--pricing-overlay-output-subdir",
    default="compustat_overlay_pricing_recursive_ast_v2",
    show_default=True,
    help="Subfolder under runs/<run_id>/compustat_overlay_v1/ for pricing overlay aggregates.",
)
@click.option(
    "--covenant-metrics-output-subdir",
    default="compiled_covenant_metrics_recursive_ast_v2",
    show_default=True,
    help="Subfolder under runs/<run_id>/definitions_compiler_v1/ for covenant metric aggregates.",
)
@click.option(
    "--covenant-blocking-output-subdir",
    default="blocking_covenant_terms_recursive_ast_v2_depth1",
    show_default=True,
    help="Subfolder under runs/<run_id>/blocking_terms_compiler_v1/ for covenant dependency-term aggregates.",
)
@click.option(
    "--covenant-overlay-output-subdir",
    default="compustat_overlay_covenant_recursive_ast_v2",
    show_default=True,
    help="Subfolder under runs/<run_id>/compustat_overlay_v1/ for covenant overlay aggregates.",
)
@click.option(
    "--recursive-max-depth",
    default=1,
    show_default=True,
    type=int,
    help="Max recursion depth for blocking terms (1 = initial blocking terms only).",
)
@click.option(
    "--recursive-max-terms",
    default=200,
    show_default=True,
    type=int,
    help="Max unique dependency terms per item during recursive blocking-term compilation.",
)
@click.option(
    "--analysis-output-subdir",
    default="analysis_export_v2",
    show_default=True,
    help="Output subfolder under runs/<run_id>/analysis_export/.",
)
@click.option(
    "--item-id",
    "item_ids",
    multiple=True,
    help="Optional item_id(s) to run the flow for (defaults to all items in the run manifest).",
)
@click.option(
    "--skip-existing",
    is_flag=True,
    help="Skip items with existing outputs where supported (indexing, structured, metadata).",
)
def all_v2_full(
    run_id: str,
    tarball,
    accessions_file: Optional[str],
    prompt_index_v2: str,
    prompt_pricing_structured_v2: str,
    prompt_covenant_structured_v2: str,
    prompt_agreement_metadata: str,
    prompt_metrics_compiler: str,
    prompt_blocking_terms_compiler: str,
    prompt_compustat_overlay: str,
    bandwidth: int,
    base_dir: str,
    item_ids_file: Optional[str],
    doc_type_prefixes: Tuple[str, ...],
    gateway_url: str | None,
    temperature: float,
    reasoning: str | None,
    gateway_timeout: float,
    concurrency: int,
    attempts: int,
    structured_output_subdir: str | None,
    covenant_structured_output_subdir: str,
    covenant_categories: Tuple[str, ...],
    agreement_metadata_output_subdir: str | None,
    pricing_metrics_output_subdir: str,
    pricing_blocking_output_subdir: str,
    pricing_overlay_output_subdir: str,
    covenant_metrics_output_subdir: str,
    covenant_blocking_output_subdir: str,
    covenant_overlay_output_subdir: str,
    recursive_max_depth: int,
    recursive_max_terms: int,
    analysis_output_subdir: str,
    item_ids: Tuple[str, ...],
    skip_existing: bool,
) -> None:
    """Run the locked pipeline: ingest -> normalize -> index/retrieve -> pricing+covenant structured -> recursive definitions/terms/overlay -> metadata -> analysis-export-v2."""

    from pipeline.compile.blocking_terms_compiler_v1 import run_blocking_terms_compiler_v1
    from pipeline.compile.compustat_overlay_v1 import run_compustat_overlay_v1
    from pipeline.compile.definitions_compiler_v1 import run_definitions_compiler_v1
    from pipeline.extract.agreement_metadata_v1 import run_agreement_metadata_v1
    from pipeline.extract.analysis_export_v2 import run_analysis_export_v2
    from pipeline.extract.structured_v2 import run_structured_v2
    from pipeline.evidence.indexing_v2 import run_indexing_v2
    from pipeline.evidence.retrieval_v2 import render_snippets_v2

    if bandwidth <= 0:
        raise click.UsageError("--bandwidth must be > 0.")
    if concurrency <= 0:
        raise click.UsageError("--concurrency must be > 0.")
    if gateway_timeout <= 0:
        raise click.UsageError("--gateway-timeout must be > 0.")
    if attempts <= 0:
        raise click.UsageError("--attempts must be > 0.")
    if recursive_max_depth <= 0:
        raise click.UsageError("--recursive-max-depth must be > 0.")
    if recursive_max_terms <= 0:
        raise click.UsageError("--recursive-max-terms must be > 0.")

    paths = resolve_paths(run_id, base_dir, bandwidth=bandwidth)
    paths.run_dir.mkdir(parents=True, exist_ok=True)
    status_path = paths.run_dir / "all_v2_status.json"
    status_doc: dict[str, object] = {
        "schema_version": "all_v2_status_v2_recursive",
        "run_id": run_id,
        "started_at": int(time.time()),
        "status": "running",
        "stages": [],
    }

    def _write_status() -> None:
        status_path.write_text(json.dumps(status_doc, indent=2), encoding="utf-8")

    def _run_stage(name: str, fn) -> None:
        started = time.time()
        rec: dict[str, object] = {"stage": name, "status": "running", "started_at": int(started)}
        stages = status_doc.get("stages")
        if isinstance(stages, list):
            stages.append(rec)
        _write_status()
        try:
            fn()
            rec["status"] = "ok"
            rec["finished_at"] = int(time.time())
            rec["duration_seconds"] = round(time.time() - started, 3)
            _write_status()
        except Exception as exc:
            rec["status"] = "error"
            rec["error"] = str(exc)
            rec["finished_at"] = int(time.time())
            rec["duration_seconds"] = round(time.time() - started, 3)
            status_doc["status"] = "error"
            status_doc["error_stage"] = name
            status_doc["error"] = str(exc)
            status_doc["finished_at"] = int(time.time())
            _write_status()
            raise RuntimeError(f"[all-v2-full] stage failed: {name}: {exc}") from exc

    _write_status()

    accessions, selection, doc_filter = load_accessions_and_selection(
        accessions_file=accessions_file,
        item_ids_file=item_ids_file,
        doc_type_prefixes=doc_type_prefixes,
    )

    _run_stage(
        "ingest",
        lambda: ingest_tarballs(paths, [Path(t) for t in tarball], selection, accessions, doc_filter=doc_filter),
    )
    manifest, _items, selected_item_ids = resolve_selected_item_ids(paths, item_ids)
    status_doc["item_count"] = len(selected_item_ids)
    _write_status()

    _run_stage("normalize", lambda: build_prompt_views(paths, manifest))

    _run_stage(
        "index-v2",
        lambda: run_indexing_v2(
            paths,
            selected_item_ids,
            Path(prompt_index_v2),
            gateway_url=gateway_url,
            temperature=temperature,
            reasoning=reasoning,
            gateway_timeout=gateway_timeout,
            concurrency=concurrency,
            attempts=attempts,
            skip_existing=bool(skip_existing),
        ),
    )
    _run_stage("retrieve-v2", lambda: render_snippets_v2(paths, selected_item_ids, bandwidth=bandwidth))

    pricing_qa_subdir = structured_output_subdir or Path(prompt_pricing_structured_v2).stem
    _run_stage(
        "pricing-structured-v2",
        lambda: run_structured_v2(
            paths,
            selected_item_ids,
            Path(prompt_pricing_structured_v2),
            contract="pricing_structured_v2",
            gateway_url=gateway_url,
            temperature=temperature,
            reasoning=reasoning,
            gateway_timeout=gateway_timeout,
            concurrency=concurrency,
            attempts=attempts,
            output_subdir=pricing_qa_subdir,
            categories=("metadata", "fundamental", "pricing", "financial_covenant"),
            skip_existing=bool(skip_existing),
        ),
    )

    cov_categories = covenant_categories if covenant_categories else ("financial_covenant",)
    covenant_qa_subdir = covenant_structured_output_subdir or Path(prompt_covenant_structured_v2).stem
    _run_stage(
        "covenant-structured-v2",
        lambda: run_structured_v2(
            paths,
            selected_item_ids,
            Path(prompt_covenant_structured_v2),
            contract="covenant_simple_v2",
            gateway_url=gateway_url,
            temperature=temperature,
            reasoning=reasoning,
            gateway_timeout=gateway_timeout,
            concurrency=concurrency,
            attempts=attempts,
            output_subdir=covenant_qa_subdir,
            categories=cov_categories,
            skip_existing=bool(skip_existing),
            allow_empty_after_filter=True,
        ),
    )

    _run_stage(
        "pricing-metrics-recursive",
        lambda: run_definitions_compiler_v1(
            paths,
            selected_item_ids,
            qa_subdir=pricing_qa_subdir,
            compiler_prompt_path=Path(prompt_metrics_compiler),
            gateway_url=gateway_url,
            temperature=temperature,
            reasoning=reasoning,
            gateway_timeout=gateway_timeout,
            concurrency=concurrency,
            output_subdir=pricing_metrics_output_subdir,
            attempts=attempts,
        ),
    )
    _run_stage(
        "pricing-blocking-recursive",
        lambda: run_blocking_terms_compiler_v1(
            paths,
            selected_item_ids,
            metrics_output_subdir=pricing_metrics_output_subdir,
            term_compiler_prompt_path=Path(prompt_blocking_terms_compiler),
            gateway_url=gateway_url,
            temperature=temperature,
            reasoning=reasoning,
            gateway_timeout=gateway_timeout,
            concurrency=concurrency,
            attempts=attempts,
            output_subdir=pricing_blocking_output_subdir,
            max_depth=recursive_max_depth,
            max_terms=recursive_max_terms,
        ),
    )
    _run_stage(
        "pricing-overlay-recursive",
        lambda: run_compustat_overlay_v1(
            paths,
            selected_item_ids,
            overlay_prompt_path=Path(prompt_compustat_overlay),
            output_subdir=pricing_overlay_output_subdir,
            metrics_output_subdir=pricing_metrics_output_subdir,
            blocking_terms_output_subdir=pricing_blocking_output_subdir,
            gateway_url=gateway_url,
            temperature=temperature,
            gateway_timeout=gateway_timeout,
            concurrency=concurrency,
            attempts=attempts,
        ),
    )

    _run_stage(
        "covenant-metrics-recursive",
        lambda: run_definitions_compiler_v1(
            paths,
            selected_item_ids,
            qa_subdir=covenant_qa_subdir,
            compiler_prompt_path=Path(prompt_metrics_compiler),
            gateway_url=gateway_url,
            temperature=temperature,
            reasoning=reasoning,
            gateway_timeout=gateway_timeout,
            concurrency=concurrency,
            output_subdir=covenant_metrics_output_subdir,
            attempts=attempts,
        ),
    )
    _run_stage(
        "covenant-blocking-recursive",
        lambda: run_blocking_terms_compiler_v1(
            paths,
            selected_item_ids,
            metrics_output_subdir=covenant_metrics_output_subdir,
            term_compiler_prompt_path=Path(prompt_blocking_terms_compiler),
            gateway_url=gateway_url,
            temperature=temperature,
            reasoning=reasoning,
            gateway_timeout=gateway_timeout,
            concurrency=concurrency,
            attempts=attempts,
            output_subdir=covenant_blocking_output_subdir,
            max_depth=recursive_max_depth,
            max_terms=recursive_max_terms,
        ),
    )
    _run_stage(
        "covenant-overlay-recursive",
        lambda: run_compustat_overlay_v1(
            paths,
            selected_item_ids,
            overlay_prompt_path=Path(prompt_compustat_overlay),
            output_subdir=covenant_overlay_output_subdir,
            metrics_output_subdir=covenant_metrics_output_subdir,
            blocking_terms_output_subdir=covenant_blocking_output_subdir,
            gateway_url=gateway_url,
            temperature=temperature,
            gateway_timeout=gateway_timeout,
            concurrency=concurrency,
            attempts=attempts,
        ),
    )

    meta_subdir = agreement_metadata_output_subdir or Path(prompt_agreement_metadata).stem
    _run_stage(
        "agreement-metadata",
        lambda: run_agreement_metadata_v1(
            paths,
            selected_item_ids,
            Path(prompt_agreement_metadata),
            gateway_url=gateway_url,
            temperature=temperature,
            reasoning=reasoning,
            gateway_timeout=gateway_timeout,
            concurrency=concurrency,
            attempts=attempts,
            output_subdir=meta_subdir,
            skip_existing=bool(skip_existing),
        ),
    )

    _run_stage(
        "analysis-export-v2",
        lambda: run_analysis_export_v2(
            paths,
            selected_item_ids,
            agreement_metadata_subdir=meta_subdir,
            pricing_qa_subdir=pricing_qa_subdir,
            covenant_qa_subdir=covenant_qa_subdir,
            pricing_metrics_output_subdir=pricing_metrics_output_subdir,
            pricing_blocking_output_subdir=pricing_blocking_output_subdir,
            pricing_overlay_output_subdir=pricing_overlay_output_subdir,
            covenant_metrics_output_subdir=covenant_metrics_output_subdir,
            covenant_blocking_output_subdir=covenant_blocking_output_subdir,
            covenant_overlay_output_subdir=covenant_overlay_output_subdir,
            output_subdir=analysis_output_subdir,
        ),
    )

    status_doc["status"] = "ok"
    status_doc["finished_at"] = int(time.time())
    _write_status()

    click.echo(
        "[all-v2-full] Completed locked recursive pipeline "
        f"(pricing + covenant + recursive definitions/overlay + analysis-export-v2) for {len(selected_item_ids)} exhibits."
    )
