from __future__ import annotations

import json
import subprocess
import sys
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import click

from pipeline.ir.contract_ir_v0_2 import evaluate_function, validate_contract_ir
from pipeline.ir.contract_ir_v0_2_flow import (
    ContractIRFlowError,
    prepare_run_inputs_from_source,
    run_contractir_v0_2_flow,
)
from pipeline.ir.contract_ir_v0_2_merge import merge_contract_ir_v0_2
from pipeline.core.config import FilterSpec, Paths
from pipeline.filters import load_filter_spec, load_doc_filter
from pipeline.evidence.ingest import ingest_tarballs
from pipeline.evidence.normalize import build_prompt_views
from pipeline.utils import load_manifest, manifest_items, read_accessions_file


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise click.ClickException(f"Invalid JSON: {path}") from exc


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _decimal_str(x: Decimal) -> str:
    s = format(x, "f")
    if "." in s:
        s = s.rstrip("0").rstrip(".") or "0"
    return s


@click.group(name="contractir-v0-2")
def contractir_v0_2_cli() -> None:
    """ContractIR v0.2 pricing-kernel tools (base rate / spread / fees)."""


@contractir_v0_2_cli.command("validate")
@click.option("--path", "path_str", type=click.Path(exists=True, dir_okay=False), required=True)
def validate_cmd(path_str: str) -> None:
    """Validate a ContractIR v0.2 JSON artifact (offline)."""

    path = Path(path_str)
    doc = _read_json(path)
    errors = validate_contract_ir(doc)
    if errors:
        for e in errors:
            click.echo(f"[{e.code}] {e.json_path}: {e.message}")
        raise click.ClickException(f"Validation failed with {len(errors)} error(s).")
    click.echo("OK")


@contractir_v0_2_cli.command("eval")
@click.option("--path", "path_str", type=click.Path(exists=True, dir_okay=False), required=True)
@click.option("--fn-id", required=True)
@click.option("--args-json", required=True, help="JSON object mapping arg name -> value.")
@click.option("--indices-json", required=False, default=None, help="Optional JSON object of series_id -> {date -> value}.")
def eval_cmd(path_str: str, fn_id: str, args_json: str, indices_json: Optional[str]) -> None:
    """Evaluate a derived function with deterministic evaluation (offline)."""

    path = Path(path_str)
    contract_ir = _read_json(path)
    try:
        args = json.loads(args_json)
    except json.JSONDecodeError as exc:
        raise click.ClickException("--args-json must be valid JSON") from exc
    if not isinstance(args, dict):
        raise click.ClickException("--args-json must be a JSON object")

    indices: Mapping[str, Mapping[str, str]] | None = None
    if indices_json is not None:
        try:
            idx = json.loads(indices_json)
        except json.JSONDecodeError as exc:
            raise click.ClickException("--indices-json must be valid JSON") from exc
        if not isinstance(idx, dict):
            raise click.ClickException("--indices-json must be a JSON object")
        indices = idx  # type: ignore[assignment]

    out = evaluate_function(contract_ir, fn_id=fn_id, args=args, indices=indices)
    val = out.value
    if isinstance(val, Decimal):
        rendered: Any = _decimal_str(val)
    else:
        rendered = val
    click.echo(json.dumps({"kind": out.kind, "value": rendered}))


@contractir_v0_2_cli.command("merge")
@click.option(
    "--input",
    "inputs",
    type=click.Path(exists=True, dir_okay=False),
    multiple=True,
    required=True,
    help="Input ContractIR v0.2 JSON files to merge.",
)
@click.option("--out", "out_path_str", type=click.Path(dir_okay=False), required=True)
def merge_cmd(inputs: tuple[str, ...], out_path_str: str) -> None:
    """Merge multiple ContractIR v0.2 artifacts deterministically (offline)."""

    docs = []
    for p in inputs:
        docs.append(_read_json(Path(p)))
    merged = merge_contract_ir_v0_2(docs)
    out_path = Path(out_path_str)
    _write_json(out_path, merged)
    click.echo(str(out_path))


@contractir_v0_2_cli.command("strategy-harness")
@click.option("--source-run-id", default="dan-v2-20260106", show_default=True)
@click.option("--out-run-id", default=None)
@click.option("--exp-filter", default=None)
def strategy_harness_cmd(source_run_id: str, out_run_id: Optional[str], exp_filter: Optional[str]) -> None:
    """Run the ContractIR v0.2 strategy harness (LLM calls, produces run artifacts)."""

    repo_root = Path(__file__).resolve().parents[2]
    script = repo_root / "scripts" / "contract_ir_v0_2_strategy_harness.py"
    if not script.exists():
        raise click.ClickException(f"Missing harness script: {script}")

    cmd = [sys.executable, str(script), "--source-run-id", source_run_id]
    if out_run_id:
        cmd.extend(["--out-run-id", out_run_id])
    if exp_filter:
        cmd.extend(["--exp-filter", exp_filter])

    # We intentionally stream output to the user (long-running LLM calls).
    rc = subprocess.call(cmd, cwd=str(repo_root))
    if rc != 0:
        raise click.ClickException(f"Strategy harness failed with exit code {rc}")


@contractir_v0_2_cli.command("flow")
@click.option("--run-id", required=True, help="Output run identifier (writes under runs/<run_id>/).")
@click.option(
    "--source-run-id",
    default=None,
    help="Optional existing run-id to clone inputs from (copies normalized + manifest for cleanliness).",
)
@click.option(
    "--tarball",
    multiple=True,
    type=click.Path(exists=True, dir_okay=False),
    required=False,
    help="Tarball(s) to ingest when not using --source-run-id.",
)
@click.option(
    "--filters",
    "filters_path",
    type=click.Path(exists=True, dir_okay=False),
    required=False,
    help="Filter spec JSON/YAML (doc_filter_path + optional doc_filter_kwargs) when ingesting.",
)
@click.option("--accessions-file", type=click.Path(exists=True, dir_okay=False), required=False)
@click.option("--base-dir", default=".", show_default=True)
@click.option(
    "--indexing-prompt",
    "indexing_prompt_path",
    default="prompts/indexing_v2_pricing_kernel_agentic_v2.txt",
    show_default=True,
    type=click.Path(exists=True, dir_okay=False),
)
@click.option("--retrieval-bandwidth", default=400, show_default=True, type=int)
@click.option("--gateway-url", default=None, help="Gateway base URL; defaults to $GATEWAY_URL.")
@click.option("--timeout-seconds", default=600.0, show_default=True, type=float)
@click.option("--temperature", default=0.0, show_default=True, type=float)
@click.option("--attempts", default=3, show_default=True, type=int, help="Max validation/repair attempts per pass.")
@click.option("--item-id", "item_ids", multiple=True, help="Optional item_id(s) to run (defaults to all items).")
def flow_cmd(
    run_id: str,
    source_run_id: Optional[str],
    tarball: tuple[str, ...],
    filters_path: Optional[str],
    accessions_file: Optional[str],
    base_dir: str,
    indexing_prompt_path: str,
    retrieval_bandwidth: int,
    gateway_url: Optional[str],
    timeout_seconds: float,
    temperature: float,
    attempts: int,
    item_ids: tuple[str, ...],
) -> None:
    """End-to-end pricing-kernel run: inputs -> indexing_v2 -> retrieval_v2 -> ContractIR v0.2 passes -> merge."""

    paths = Paths(root=Path(base_dir), run_id=run_id)

    if source_run_id:
        source_paths = Paths(root=Path(base_dir), run_id=source_run_id)
        try:
            prepare_run_inputs_from_source(dest_paths=paths, source_paths=source_paths)
        except ContractIRFlowError as exc:
            raise click.ClickException(str(exc)) from exc
    else:
        if not tarball or not filters_path:
            raise click.UsageError("Provide either --source-run-id OR (--tarball ... AND --filters ...).")

        accessions = read_accessions_file(Path(accessions_file)) if accessions_file else None
        spec = load_filter_spec(Path(filters_path)) if filters_path else FilterSpec(doc_filter_path="pipeline.filters:keep_all")
        doc_filter = load_doc_filter(spec)
        ingest_tarballs(paths, [Path(t) for t in tarball], spec, accessions, doc_filter=doc_filter)
        manifest = load_manifest(paths.manifest_path)
        build_prompt_views(paths, manifest)

    base_rate_prompt = Path("prompts/contract_ir_base_rate_v2.txt")
    spread_prompts = [
        Path("prompts/contract_ir_spread_v2_lookup2.txt"),
        Path("prompts/contract_ir_spread_v2_lookup_range.txt"),
        Path("prompts/contract_ir_spread_v2_lookup_rule.txt"),
        Path("prompts/contract_ir_spread_v2_specialized.txt"),
    ]
    fee_prompts = [
        Path("prompts/contract_ir_fee_rate_v2_lookup2.txt"),
        Path("prompts/contract_ir_fee_rate_v2_lookup_range.txt"),
        Path("prompts/contract_ir_fee_rate_v2_specialized.txt"),
    ]

    # Validate prompt files exist early (fail loud).
    if not base_rate_prompt.exists():
        raise click.ClickException(f"Missing base rate prompt: {base_rate_prompt}")
    for p in spread_prompts:
        if not p.exists():
            raise click.ClickException(f"Missing spread prompt: {p}")
    for p in fee_prompts:
        if not p.exists():
            raise click.ClickException(f"Missing fee prompt: {p}")

    # Ensure item_ids are valid if provided (fail loud before any LLM calls).
    if item_ids:
        manifest = load_manifest(paths.manifest_path)
        manifest_ids = {item["item_id"] for item in manifest_items(manifest)}
        unknown = sorted(set(item_ids) - manifest_ids)
        if unknown:
            raise click.ClickException(f"Unknown --item-id values not present in manifest: {', '.join(unknown)}")

    try:
        summary_path = run_contractir_v0_2_flow(
            paths=paths,
            item_ids=list(item_ids) if item_ids else None,
            indexing_prompt_path=Path(indexing_prompt_path),
            retrieval_bandwidth=int(retrieval_bandwidth),
            base_rate_prompt_path=base_rate_prompt,
            spread_prompt_paths=spread_prompts,
            fee_prompt_paths=fee_prompts,
            gateway_url=gateway_url,
            timeout_seconds=float(timeout_seconds),
            temperature=float(temperature),
            attempts=int(attempts),
            concurrency=1,
        )
    except ContractIRFlowError as exc:
        raise click.ClickException(str(exc)) from exc

    click.echo(str(summary_path))
