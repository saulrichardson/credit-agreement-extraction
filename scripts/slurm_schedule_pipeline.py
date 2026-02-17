#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import shlex
import subprocess
import time
from pathlib import Path
from typing import Any, Iterable, Sequence

import yaml

RUN_ID_RE = re.compile(r"^[A-Za-z0-9._-]+$")
RUN_ID_MAX_LEN = 80
GATEWAY_URL_TOKEN = "__GATEWAY_URL_TOKEN__"


def _load_json_or_yaml(path: Path) -> Any:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() in {".yml", ".yaml"}:
        return yaml.safe_load(text)
    return json.loads(text)


def _load_item_ids(path: Path) -> list[str]:
    raw = _load_json_or_yaml(path)
    if isinstance(raw, dict):
        item_ids = raw.get("item_ids")
    else:
        item_ids = raw

    if not isinstance(item_ids, list) or not item_ids:
        raise ValueError(f"{path} must contain a non-empty item_ids list")

    out: list[str] = []
    seen: set[str] = set()
    for idx, raw_item in enumerate(item_ids):
        value = str(raw_item or "").strip()
        if not value:
            raise ValueError(f"{path} item_ids[{idx}] is empty")
        if value in seen:
            continue
        seen.add(value)
        out.append(value)
    if not out:
        raise ValueError(f"No valid item_ids after normalization: {path}")
    return out


def _load_tarballs(args_tarballs: Sequence[str], tarballs_file: str | None) -> list[str]:
    values: list[str] = [str(v).strip() for v in args_tarballs if str(v).strip()]

    if tarballs_file:
        p = Path(tarballs_file)
        if not p.exists():
            raise FileNotFoundError(f"tarballs-file not found: {p}")
        for line in p.read_text(encoding="utf-8").splitlines():
            t = line.strip()
            if not t or t.startswith("#"):
                continue
            values.append(t)

    if not values:
        raise ValueError("Provide at least one --tarball or --tarballs-file")

    dedup: list[str] = []
    seen: set[str] = set()
    for t in values:
        if t in seen:
            continue
        seen.add(t)
        dedup.append(t)

    for t in dedup:
        if not Path(t).exists():
            raise FileNotFoundError(f"Tarball not found: {t}")

    return dedup


def _load_item_tarball_map(
    *,
    csv_path: str | None,
    item_id_key: str,
    tarball_key: str,
) -> dict[str, list[str]]:
    if not csv_path:
        return {}

    p = Path(csv_path)
    if not p.exists():
        raise FileNotFoundError(f"item-id-tarball-map-csv not found: {p}")

    out: dict[str, list[str]] = {}
    seen_pairs: set[tuple[str, str]] = set()
    with p.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        if not reader.fieldnames:
            raise ValueError(f"CSV has no header: {p}")
        if item_id_key not in reader.fieldnames:
            raise ValueError(f"CSV missing item id column {item_id_key!r}: {p}")
        if tarball_key not in reader.fieldnames:
            raise ValueError(f"CSV missing tarball column {tarball_key!r}: {p}")

        for row_idx, row in enumerate(reader, start=2):
            item_id = str(row.get(item_id_key) or "").strip()
            tarball = str(row.get(tarball_key) or "").strip()
            if not item_id or not tarball:
                continue
            pair = (item_id, tarball)
            if pair in seen_pairs:
                continue
            seen_pairs.add(pair)
            out.setdefault(item_id, []).append(tarball)

    return out


def _chunk(values: Sequence[str], size: int) -> list[list[str]]:
    if size <= 0:
        raise ValueError("shard-size must be > 0")
    return [list(values[i : i + size]) for i in range(0, len(values), size)]


def _make_run_id(prefix: str, shard_idx: int) -> str:
    raw = f"{prefix}-s{shard_idx:04d}"
    if len(raw) <= RUN_ID_MAX_LEN and RUN_ID_RE.fullmatch(raw):
        return raw

    digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:8]
    trimmed = re.sub(r"[^A-Za-z0-9._-]", "-", prefix).strip("-._")
    keep = max(8, RUN_ID_MAX_LEN - len("-s0000-") - len(digest))
    trimmed = trimmed[:keep]
    candidate = f"{trimmed}-s{shard_idx:04d}-{digest}"
    if len(candidate) > RUN_ID_MAX_LEN:
        candidate = candidate[:RUN_ID_MAX_LEN]
    if not RUN_ID_RE.fullmatch(candidate):
        raise ValueError(f"Failed to build valid run_id from prefix={prefix!r}, shard={shard_idx}")
    return candidate


def _q(parts: Iterable[object]) -> str:
    return " ".join(shlex.quote(str(p)) for p in parts)


def _render_command(parts: Iterable[object], *, gateway_url_shell_var: bool = False) -> str:
    rendered = _q(parts)
    if gateway_url_shell_var:
        rendered = rendered.replace(shlex.quote(GATEWAY_URL_TOKEN), '"${GATEWAY_URL}"')
    return rendered


def _build_indexing_commands(
    *,
    pipeline_cmd: Sequence[str],
    run_id: str,
    shard_item_ids_file: Path,
    tarballs: Sequence[str],
    accessions_file: str | None,
    doc_type_prefixes: Sequence[str],
    prompt_index_v2: str,
    gateway_url: str,
    temperature: float,
    reasoning: str,
    gateway_timeout: float,
    concurrency: int,
    attempts: int,
    bandwidth: int,
    resume: bool,
    gateway_url_shell_var: bool = False,
) -> list[str]:
    ingest_cmd: list[object] = [*pipeline_cmd, "ingest", "--run-id", run_id]
    for t in tarballs:
        ingest_cmd.extend(["--tarball", t])
    ingest_cmd.extend(["--item-ids-file", str(shard_item_ids_file)])
    if accessions_file:
        ingest_cmd.extend(["--accessions-file", accessions_file])
    for dt in doc_type_prefixes:
        ingest_cmd.extend(["--doc-type-prefix", dt])

    normalize_cmd = [*pipeline_cmd, "normalize", "--run-id", run_id]

    index_cmd: list[object] = [
        *pipeline_cmd,
        "index-v2",
        "--run-id",
        run_id,
        "--prompt",
        prompt_index_v2,
        "--gateway-url",
        GATEWAY_URL_TOKEN if gateway_url_shell_var else gateway_url,
        "--temperature",
        str(float(temperature)),
        "--reasoning",
        reasoning,
        "--gateway-timeout",
        str(float(gateway_timeout)),
        "--concurrency",
        str(int(concurrency)),
        "--attempts",
        str(int(attempts)),
    ]
    if resume:
        index_cmd.append("--skip-existing")

    retrieve_cmd = [
        *pipeline_cmd,
        "retrieve-v2",
        "--run-id",
        run_id,
        "--bandwidth",
        str(int(bandwidth)),
    ]

    return [
        _q(ingest_cmd),
        _q(normalize_cmd),
        _render_command(index_cmd, gateway_url_shell_var=gateway_url_shell_var),
        _q(retrieve_cmd),
    ]


def _build_all_v2_full_command(
    *,
    pipeline_cmd: Sequence[str],
    run_id: str,
    shard_item_ids_file: Path,
    tarballs: Sequence[str],
    accessions_file: str | None,
    doc_type_prefixes: Sequence[str],
    prompt_index_v2: str,
    prompt_pricing_structured_v2: str,
    prompt_covenant_structured_v2: str,
    prompt_agreement_metadata: str,
    prompt_metrics_compiler: str,
    prompt_blocking_terms_compiler: str,
    prompt_compustat_overlay: str,
    structured_output_subdir: str | None,
    covenant_structured_output_subdir: str,
    pricing_metrics_output_subdir: str,
    pricing_blocking_output_subdir: str,
    pricing_overlay_output_subdir: str,
    covenant_metrics_output_subdir: str,
    covenant_blocking_output_subdir: str,
    covenant_overlay_output_subdir: str,
    covenant_categories: Sequence[str],
    recursive_max_depth: int,
    recursive_max_terms: int,
    attempts: int,
    analysis_output_subdir: str,
    gateway_url: str,
    temperature: float,
    reasoning: str,
    gateway_timeout: float,
    concurrency: int,
    bandwidth: int,
    resume: bool,
    gateway_url_shell_var: bool = False,
) -> str:
    cmd: list[object] = [
        *pipeline_cmd,
        "all-v2-full",
        "--run-id",
        run_id,
        "--item-ids-file",
        str(shard_item_ids_file),
        "--prompt-index-v2",
        prompt_index_v2,
        "--prompt-pricing-structured-v2",
        prompt_pricing_structured_v2,
        "--prompt-covenant-structured-v2",
        prompt_covenant_structured_v2,
        "--prompt-agreement-metadata",
        prompt_agreement_metadata,
        "--prompt-metrics-compiler",
        prompt_metrics_compiler,
        "--prompt-blocking-terms-compiler",
        prompt_blocking_terms_compiler,
        "--prompt-compustat-overlay",
        prompt_compustat_overlay,
        "--covenant-structured-output-subdir",
        covenant_structured_output_subdir,
        "--pricing-metrics-output-subdir",
        pricing_metrics_output_subdir,
        "--pricing-blocking-output-subdir",
        pricing_blocking_output_subdir,
        "--pricing-overlay-output-subdir",
        pricing_overlay_output_subdir,
        "--covenant-metrics-output-subdir",
        covenant_metrics_output_subdir,
        "--covenant-blocking-output-subdir",
        covenant_blocking_output_subdir,
        "--covenant-overlay-output-subdir",
        covenant_overlay_output_subdir,
        "--recursive-max-depth",
        str(int(recursive_max_depth)),
        "--recursive-max-terms",
        str(int(recursive_max_terms)),
        "--attempts",
        str(int(attempts)),
        "--analysis-output-subdir",
        analysis_output_subdir,
        "--gateway-url",
        GATEWAY_URL_TOKEN if gateway_url_shell_var else gateway_url,
        "--temperature",
        str(float(temperature)),
        "--reasoning",
        reasoning,
        "--gateway-timeout",
        str(float(gateway_timeout)),
        "--concurrency",
        str(int(concurrency)),
        "--bandwidth",
        str(int(bandwidth)),
    ]
    if structured_output_subdir:
        cmd.extend(["--structured-output-subdir", structured_output_subdir])
    for cat in covenant_categories:
        if str(cat).strip():
            cmd.extend(["--covenant-category", str(cat).strip()])
    for t in tarballs:
        cmd.extend(["--tarball", t])
    if accessions_file:
        cmd.extend(["--accessions-file", accessions_file])
    for dt in doc_type_prefixes:
        cmd.extend(["--doc-type-prefix", dt])
    if resume:
        cmd.append("--skip-existing")
    return _render_command(cmd, gateway_url_shell_var=gateway_url_shell_var)


def _write_sbatch_script(
    *,
    path: Path,
    job_name: str,
    log_dir: Path,
    project_root: Path,
    account: str | None,
    partition: str | None,
    qos: str | None,
    time_limit: str,
    cpus_per_task: int,
    mem_gb: int,
    pre_commands: Sequence[str],
    commands: Sequence[str],
    post_commands: Sequence[str],
) -> None:
    if not commands:
        raise ValueError("commands must be non-empty")

    lines: list[str] = [
        "#!/usr/bin/env bash",
        f"#SBATCH --job-name={job_name}",
        f"#SBATCH --output={log_dir}/{job_name}.%j.out",
        f"#SBATCH --error={log_dir}/{job_name}.%j.err",
        f"#SBATCH --time={time_limit}",
        f"#SBATCH --cpus-per-task={int(cpus_per_task)}",
        f"#SBATCH --mem={int(mem_gb)}G",
    ]
    if account:
        lines.append(f"#SBATCH --account={account}")
    if partition:
        lines.append(f"#SBATCH --partition={partition}")
    if qos:
        lines.append(f"#SBATCH --qos={qos}")

    lines.extend(
        [
            "",
            "set -euo pipefail",
            f"cd {shlex.quote(str(project_root))}",
            "export PYTHONUNBUFFERED=1",
            "",
        ]
    )
    for cmd in pre_commands:
        lines.append(cmd)
    for cmd in commands:
        lines.append(cmd)
    for cmd in post_commands:
        lines.append(cmd)

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    path.chmod(0o755)


def _submit_sbatch(script_path: Path) -> str:
    proc = subprocess.run(
        ["sbatch", str(script_path)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"sbatch failed for {script_path}: {proc.stderr.strip() or proc.stdout.strip()}")

    out = (proc.stdout or "").strip()
    m = re.search(r"Submitted batch job\s+(\d+)", out)
    if not m:
        raise RuntimeError(f"Could not parse sbatch output for {script_path}: {out}")
    return m.group(1)


def main() -> int:
    parser = argparse.ArgumentParser(description="Shard and schedule pipeline jobs on Slurm (Torch-friendly).")
    parser.add_argument("--workflow", choices=["indexing-retrieval", "all-v2-full"], default="all-v2-full")
    parser.add_argument("--run-id-prefix", required=True)
    parser.add_argument("--item-ids-file", required=True, help="JSON/YAML with item_ids list.")
    parser.add_argument("--shard-size", type=int, default=250)
    parser.add_argument("--max-shards", type=int, default=None)

    parser.add_argument("--tarball", action="append", default=[], help="Repeatable tarball path.")
    parser.add_argument("--tarballs-file", default=None, help="Optional text file listing tarballs (one per line).")
    parser.add_argument(
        "--item-id-tarball-map-csv",
        default=None,
        help="Optional CSV mapping item IDs to tarball paths to shrink each shard's tarball scan set.",
    )
    parser.add_argument(
        "--item-id-map-key",
        default="item_id",
        help="Column name in --item-id-tarball-map-csv that contains item IDs.",
    )
    parser.add_argument(
        "--item-id-map-tarball-key",
        default="tarball_path",
        help="Column name in --item-id-tarball-map-csv that contains tarball paths.",
    )
    parser.add_argument("--accessions-file", default=None)
    parser.add_argument("--doc-type-prefix", action="append", default=[])

    parser.add_argument("--base-dir", default=".", help="Project directory where `poetry run pipeline ...` is executed.")
    parser.add_argument(
        "--pipeline-cmd",
        default="poetry run pipeline",
        help="Command prefix used to invoke the pipeline CLI (e.g. 'poetry run pipeline' or '.venv/bin/pipeline').",
    )
    parser.add_argument("--scratch-dir", default=None, help="Output directory for generated shard files/sbatch scripts.")
    parser.add_argument(
        "--pythonpath",
        action="append",
        default=[],
        help="Optional path(s) to prepend to PYTHONPATH for each job. Repeatable.",
    )

    parser.add_argument("--account", default=None)
    parser.add_argument("--partition", default=None)
    parser.add_argument("--qos", default=None)
    parser.add_argument("--time", default="08:00:00")
    parser.add_argument("--cpus-per-task", type=int, default=4)
    parser.add_argument("--mem-gb", type=int, default=24)

    parser.add_argument("--gateway-url", default="http://127.0.0.1:8000")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--reasoning", choices=["light", "medium", "heavy"], default="medium")
    parser.add_argument("--gateway-timeout", type=float, default=600.0)
    parser.add_argument("--concurrency", type=int, default=3)
    parser.add_argument("--attempts", type=int, default=3)
    parser.add_argument("--bandwidth", type=int, default=400)

    parser.add_argument("--prompt-index-v2", default="prompts/indexing_v2.txt")
    parser.add_argument(
        "--prompt-pricing-structured-v2",
        default="prompts/prompt_pricing_second_pass_dg_nano_v2_tuned_v2.txt",
    )
    parser.add_argument("--prompt-covenant-structured-v2", default="prompts/prompt_v1_short.txt")
    parser.add_argument("--prompt-agreement-metadata", default="prompts/agreement_metadata_v1.txt")
    parser.add_argument("--prompt-metrics-compiler", default="prompts/definitions_compiler_v1_metrics_ast_v2.txt")
    parser.add_argument("--prompt-blocking-terms-compiler", default="prompts/blocking_terms_compiler_v1_ast_v2.txt")
    parser.add_argument("--prompt-compustat-overlay", default="prompts/compustat_overlay_v1.txt")
    parser.add_argument("--structured-output-subdir", default=None)
    parser.add_argument("--covenant-structured-output-subdir", default="covenant_simple_v1")
    parser.add_argument("--pricing-metrics-output-subdir", default="compiled_pricing_metrics_recursive_ast_v2")
    parser.add_argument("--pricing-blocking-output-subdir", default="blocking_pricing_terms_recursive_ast_v2_depth1")
    parser.add_argument("--pricing-overlay-output-subdir", default="compustat_overlay_pricing_recursive_ast_v2")
    parser.add_argument("--covenant-metrics-output-subdir", default="compiled_covenant_metrics_recursive_ast_v2")
    parser.add_argument("--covenant-blocking-output-subdir", default="blocking_covenant_terms_recursive_ast_v2_depth1")
    parser.add_argument("--covenant-overlay-output-subdir", default="compustat_overlay_covenant_recursive_ast_v2")
    parser.add_argument("--covenant-category", action="append", default=[])
    parser.add_argument("--recursive-max-depth", type=int, default=1)
    parser.add_argument("--recursive-max-terms", type=int, default=200)
    parser.add_argument("--analysis-output-subdir", default="analysis_export_v2")

    parser.add_argument("--resume", action="store_true", help="Enable stage-level --skip-existing flags where available.")
    parser.add_argument(
        "--start-local-gateway",
        action="store_true",
        help="Start a local gateway process inside each Slurm job and point pipeline --gateway-url to it.",
    )
    parser.add_argument(
        "--gateway-cmd",
        default=".venv/bin/gateway",
        help="Gateway command used when --start-local-gateway is set (e.g. '.venv/bin/gateway' or '.venv/bin/python -m gateway').",
    )
    parser.add_argument(
        "--gateway-port",
        type=int,
        default=8000,
        help="Gateway port used when --start-local-gateway is enabled.",
    )
    parser.add_argument(
        "--gateway-port-base",
        type=int,
        default=None,
        help="Optional base port for per-shard local gateways (actual port = base + shard_index - 1).",
    )
    parser.add_argument(
        "--gateway-wait-seconds",
        type=int,
        default=45,
        help="Max seconds to wait for local gateway health when --start-local-gateway is enabled.",
    )
    parser.add_argument(
        "--gateway-port-retries",
        type=int,
        default=25,
        help="Additional sequential ports to try if local gateway port is busy (0 = only configured port).",
    )
    parser.add_argument("--submit", action="store_true", help="Submit generated sbatch scripts immediately.")

    args = parser.parse_args()

    if args.shard_size <= 0:
        raise SystemExit("--shard-size must be > 0")
    if args.max_shards is not None and args.max_shards <= 0:
        raise SystemExit("--max-shards must be > 0 when provided")
    if args.cpus_per_task <= 0:
        raise SystemExit("--cpus-per-task must be > 0")
    if args.mem_gb <= 0:
        raise SystemExit("--mem-gb must be > 0")
    if args.concurrency <= 0:
        raise SystemExit("--concurrency must be > 0")
    if args.attempts <= 0:
        raise SystemExit("--attempts must be > 0")
    if args.bandwidth <= 0:
        raise SystemExit("--bandwidth must be > 0")
    if args.recursive_max_depth <= 0:
        raise SystemExit("--recursive-max-depth must be > 0")
    if args.recursive_max_terms <= 0:
        raise SystemExit("--recursive-max-terms must be > 0")
    if args.gateway_timeout <= 0:
        raise SystemExit("--gateway-timeout must be > 0")
    if args.gateway_port <= 0:
        raise SystemExit("--gateway-port must be > 0")
    if args.gateway_port_base is not None and args.gateway_port_base <= 0:
        raise SystemExit("--gateway-port-base must be > 0 when provided")
    if args.gateway_wait_seconds <= 0:
        raise SystemExit("--gateway-wait-seconds must be > 0")
    if args.gateway_port_retries < 0:
        raise SystemExit("--gateway-port-retries must be >= 0")

    project_root = Path(args.base_dir).resolve()
    if not project_root.exists():
        raise SystemExit(f"--base-dir does not exist: {project_root}")
    try:
        pipeline_cmd = shlex.split(str(args.pipeline_cmd))
    except ValueError as exc:
        raise SystemExit(f"Invalid --pipeline-cmd: {exc}") from exc
    if not pipeline_cmd:
        raise SystemExit("--pipeline-cmd must not be empty")

    resolved_gateway_url = str(args.gateway_url)
    gateway_cmd: list[str] | None = None
    if args.start_local_gateway:
        try:
            gateway_cmd = shlex.split(str(args.gateway_cmd))
        except ValueError as exc:
            raise SystemExit(f"Invalid --gateway-cmd: {exc}") from exc
        if not gateway_cmd:
            raise SystemExit("--gateway-cmd must not be empty when --start-local-gateway is set")
        first = gateway_cmd[0]
        if "/" in first:
            raw_gateway_bin = Path(first)
            gateway_bin_path = raw_gateway_bin if raw_gateway_bin.is_absolute() else (project_root / raw_gateway_bin)
            if not gateway_bin_path.exists():
                raise SystemExit(f"--gateway-cmd executable not found: {gateway_bin_path}")
            gateway_cmd[0] = str(gateway_bin_path)
        resolved_gateway_url = f"http://127.0.0.1:{int(args.gateway_port)}"

    item_ids_path = Path(args.item_ids_file)
    if not item_ids_path.exists():
        raise SystemExit(f"--item-ids-file not found: {item_ids_path}")
    item_ids = _load_item_ids(item_ids_path)

    tarballs = _load_tarballs(args.tarball, args.tarballs_file)
    item_tarball_map = _load_item_tarball_map(
        csv_path=args.item_id_tarball_map_csv,
        item_id_key=str(args.item_id_map_key),
        tarball_key=str(args.item_id_map_tarball_key),
    )
    tarball_order = {t: i for i, t in enumerate(tarballs)}

    for p in [
        args.prompt_index_v2,
        args.prompt_pricing_structured_v2,
        args.prompt_covenant_structured_v2,
        args.prompt_agreement_metadata,
        args.prompt_metrics_compiler,
        args.prompt_blocking_terms_compiler,
        args.prompt_compustat_overlay,
    ]:
        if not (project_root / p).exists() and not Path(p).exists():
            raise SystemExit(f"Prompt not found: {p}")

    ts = time.strftime("%Y%m%d_%H%M%S")
    scratch_root = Path(args.scratch_dir).resolve() if args.scratch_dir else project_root / "scratch" / "slurm" / f"{args.run_id_prefix}-{ts}"
    shard_dir = scratch_root / "shards"
    sbatch_dir = scratch_root / "sbatch"
    log_dir = scratch_root / "logs"
    scratch_root.mkdir(parents=True, exist_ok=True)
    shard_dir.mkdir(parents=True, exist_ok=True)
    sbatch_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    shard_lists = _chunk(item_ids, int(args.shard_size))
    if args.max_shards is not None:
        shard_lists = shard_lists[: int(args.max_shards)]

    manifest: dict[str, Any] = {
        "schema_version": "slurm_pipeline_schedule_v1",
        "created_at": int(time.time()),
        "workflow": args.workflow,
        "run_id_prefix": args.run_id_prefix,
        "project_root": str(project_root),
        "item_ids_source": str(item_ids_path.resolve()),
        "item_ids_total": len(item_ids),
        "shard_size": int(args.shard_size),
        "shard_count": len(shard_lists),
        "tarballs": tarballs,
        "item_id_tarball_map_csv": str(Path(args.item_id_tarball_map_csv).resolve())
        if args.item_id_tarball_map_csv
        else None,
        "item_id_map_key": str(args.item_id_map_key),
        "item_id_map_tarball_key": str(args.item_id_map_tarball_key),
        "gateway_url": resolved_gateway_url,
        "reasoning": args.reasoning,
        "concurrency": int(args.concurrency),
        "attempts": int(args.attempts),
        "resume": bool(args.resume),
        "pipeline_cmd": pipeline_cmd,
        "pythonpath": list(args.pythonpath),
        "start_local_gateway": bool(args.start_local_gateway),
        "gateway_cmd": gateway_cmd,
        "gateway_port": int(args.gateway_port),
        "gateway_port_base": int(args.gateway_port_base) if args.gateway_port_base is not None else None,
        "gateway_wait_seconds": int(args.gateway_wait_seconds),
        "gateway_port_retries": int(args.gateway_port_retries),
        "submitted": bool(args.submit),
        "shards": [],
    }

    for idx, shard_item_ids in enumerate(shard_lists, start=1):
        run_id = _make_run_id(args.run_id_prefix, idx)
        shard_gateway_port = (
            int(args.gateway_port_base) + idx - 1 if args.gateway_port_base is not None else int(args.gateway_port)
        )
        shard_tarballs: list[str] = tarballs
        if item_tarball_map:
            selected: list[str] = []
            seen_selected: set[str] = set()
            for item_id in shard_item_ids:
                for raw_tarball in item_tarball_map.get(item_id, []):
                    tarball = str(raw_tarball).strip()
                    if not tarball or tarball in seen_selected:
                        continue
                    seen_selected.add(tarball)
                    selected.append(tarball)
            if selected:
                missing = [t for t in selected if not Path(t).exists()]
                if missing:
                    preview = ", ".join(missing[:3])
                    suffix = " ..." if len(missing) > 3 else ""
                    raise FileNotFoundError(f"Mapped tarball(s) not found for run_id={run_id}: {preview}{suffix}")
                shard_tarballs = sorted(selected, key=lambda t: (tarball_order.get(t, 10**9), t))

        shard_file = shard_dir / f"{run_id}.item_ids.json"
        shard_payload = {
            "name": f"{args.run_id_prefix}-shard-{idx:04d}",
            "item_ids": shard_item_ids,
        }
        shard_file.write_text(json.dumps(shard_payload, indent=2) + "\n", encoding="utf-8")

        if args.workflow == "indexing-retrieval":
            commands = _build_indexing_commands(
                pipeline_cmd=pipeline_cmd,
                run_id=run_id,
                shard_item_ids_file=shard_file,
                tarballs=shard_tarballs,
                accessions_file=args.accessions_file,
                doc_type_prefixes=args.doc_type_prefix,
                prompt_index_v2=args.prompt_index_v2,
                gateway_url=(
                    f"http://127.0.0.1:{shard_gateway_port}" if args.start_local_gateway else resolved_gateway_url
                ),
                temperature=float(args.temperature),
                reasoning=args.reasoning,
                gateway_timeout=float(args.gateway_timeout),
                concurrency=int(args.concurrency),
                attempts=int(args.attempts),
                bandwidth=int(args.bandwidth),
                resume=bool(args.resume),
                gateway_url_shell_var=bool(args.start_local_gateway),
            )
        else:
            commands = [
                _build_all_v2_full_command(
                    pipeline_cmd=pipeline_cmd,
                    run_id=run_id,
                    shard_item_ids_file=shard_file,
                    tarballs=shard_tarballs,
                    accessions_file=args.accessions_file,
                    doc_type_prefixes=args.doc_type_prefix,
                    prompt_index_v2=args.prompt_index_v2,
                    prompt_pricing_structured_v2=args.prompt_pricing_structured_v2,
                    prompt_covenant_structured_v2=args.prompt_covenant_structured_v2,
                    prompt_agreement_metadata=args.prompt_agreement_metadata,
                    prompt_metrics_compiler=args.prompt_metrics_compiler,
                    prompt_blocking_terms_compiler=args.prompt_blocking_terms_compiler,
                    prompt_compustat_overlay=args.prompt_compustat_overlay,
                    structured_output_subdir=args.structured_output_subdir,
                    covenant_structured_output_subdir=args.covenant_structured_output_subdir,
                    pricing_metrics_output_subdir=args.pricing_metrics_output_subdir,
                    pricing_blocking_output_subdir=args.pricing_blocking_output_subdir,
                    pricing_overlay_output_subdir=args.pricing_overlay_output_subdir,
                    covenant_metrics_output_subdir=args.covenant_metrics_output_subdir,
                    covenant_blocking_output_subdir=args.covenant_blocking_output_subdir,
                    covenant_overlay_output_subdir=args.covenant_overlay_output_subdir,
                    covenant_categories=args.covenant_category,
                    recursive_max_depth=int(args.recursive_max_depth),
                    recursive_max_terms=int(args.recursive_max_terms),
                    attempts=int(args.attempts),
                    analysis_output_subdir=args.analysis_output_subdir,
                    gateway_url=(
                        f"http://127.0.0.1:{shard_gateway_port}" if args.start_local_gateway else resolved_gateway_url
                    ),
                    temperature=float(args.temperature),
                    reasoning=args.reasoning,
                    gateway_timeout=float(args.gateway_timeout),
                    concurrency=int(args.concurrency),
                    bandwidth=int(args.bandwidth),
                    resume=bool(args.resume),
                    gateway_url_shell_var=bool(args.start_local_gateway),
                )
            ]

        job_name = run_id[:64]
        sbatch_path = sbatch_dir / f"{run_id}.sbatch"
        pre_commands: list[str] = []
        post_commands: list[str] = []
        if args.pythonpath:
            resolved_pythonpaths: list[str] = []
            for raw in args.pythonpath:
                p = Path(raw)
                resolved = p if p.is_absolute() else (project_root / p)
                resolved_pythonpaths.append(str(resolved))
            joined = ":".join(resolved_pythonpaths) + ':${PYTHONPATH:-}'
            pre_commands.append(f"export PYTHONPATH={shlex.quote(joined)}")
        if args.start_local_gateway:
            if gateway_cmd is None:
                raise SystemExit("internal error: gateway_cmd was not resolved")
            gateway_log_path = log_dir / f"{job_name}.gateway.log"
            gateway_start_cmd = _q([*gateway_cmd, "--host", "127.0.0.1", "--port"])
            pre_commands.extend(
                [
                    f"GATEWAY_LOG={shlex.quote(str(gateway_log_path))}",
                    'GATEWAY_PID=""',
                    "GATEWAY_READY=0",
                    f"GATEWAY_PORT_BASE={int(shard_gateway_port)}",
                    f"GATEWAY_PORT_RETRIES={int(args.gateway_port_retries)}",
                    'GATEWAY_PORT=""',
                    'GATEWAY_URL=""',
                    "cleanup_gateway() {",
                    '  if [ -n "${GATEWAY_PID:-}" ] && kill -0 "$GATEWAY_PID" 2>/dev/null; then',
                    '    kill "$GATEWAY_PID" || true',
                    '    wait "$GATEWAY_PID" || true',
                    "  fi",
                    "}",
                    "trap cleanup_gateway EXIT",
                    'for PORT_OFFSET in $(seq 0 "$GATEWAY_PORT_RETRIES"); do',
                    '  GATEWAY_PORT=$((GATEWAY_PORT_BASE + PORT_OFFSET))',
                    '  GATEWAY_URL="http://127.0.0.1:${GATEWAY_PORT}"',
                    f"  {_q(['cd', project_root / 'agent-gateway'])}",
                    "  export ENVIRONMENT=development",
                    f'  {gateway_start_cmd} "$GATEWAY_PORT" > "$GATEWAY_LOG" 2>&1 &',
                    "  GATEWAY_PID=$!",
                    f"  {_q(['cd', project_root])}",
                    f"  for i in $(seq 1 {int(args.gateway_wait_seconds)}); do",
                    '    if ! kill -0 "$GATEWAY_PID" 2>/dev/null; then',
                    "      break",
                    "    fi",
                    '    if curl -fsS -m 2 "${GATEWAY_URL}/healthz" >/dev/null; then',
                    "      GATEWAY_READY=1",
                    "      break",
                    "    fi",
                    "    sleep 1",
                    "  done",
                    '  if [ "$GATEWAY_READY" -eq 1 ]; then',
                    "    break",
                    "  fi",
                    '  if [ -n "${GATEWAY_PID:-}" ] && kill -0 "$GATEWAY_PID" 2>/dev/null; then',
                    '    kill "$GATEWAY_PID" || true',
                    '    wait "$GATEWAY_PID" || true',
                    "  fi",
                    "done",
                    'if [ "$GATEWAY_READY" -ne 1 ]; then',
                    '  echo "Failed to start local gateway after trying $((GATEWAY_PORT_RETRIES + 1)) port(s) from ${GATEWAY_PORT_BASE}" >&2',
                    '  tail -n 80 "$GATEWAY_LOG" >&2 || true',
                    "  exit 1",
                    "fi",
                    'echo "[gateway] ${GATEWAY_URL}"',
                ]
            )
        _write_sbatch_script(
            path=sbatch_path,
            job_name=job_name,
            log_dir=log_dir,
            project_root=project_root,
            account=args.account,
            partition=args.partition,
            qos=args.qos,
            time_limit=args.time,
            cpus_per_task=int(args.cpus_per_task),
            mem_gb=int(args.mem_gb),
            pre_commands=pre_commands,
            commands=commands,
            post_commands=post_commands,
        )

        rec: dict[str, Any] = {
            "shard_index": idx,
            "run_id": run_id,
            "item_count": len(shard_item_ids),
            "item_ids_file": str(shard_file),
            "tarball_count": len(shard_tarballs),
            "tarballs": shard_tarballs,
            "sbatch_script": str(sbatch_path),
            "commands": commands,
            "job_id": None,
        }

        if args.submit:
            job_id = _submit_sbatch(sbatch_path)
            rec["job_id"] = job_id

        manifest["shards"].append(rec)

    manifest_path = scratch_root / "schedule_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    if args.submit:
        print(f"[submitted] {len(manifest['shards'])} jobs")
    else:
        print(f"[generated] {len(manifest['shards'])} sbatch scripts (use --submit to submit)")
    print(str(manifest_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
