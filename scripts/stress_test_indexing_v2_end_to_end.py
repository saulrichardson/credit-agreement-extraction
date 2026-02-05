#!/usr/bin/env python
"""
End-to-end stress test for SINGLE-PASS (Option A) indexing-v2 prompt strategies.

This script is intentionally artifact-first and run-scoped:
  - Clone normalized inputs from a source run into new run(s)
  - Run indexing_v2 once per item (LLM call; single pass)
  - Materialize retrieval_v2 snippets (for provenance + downstream harnesses)
  - Run pricing extraction (ContractIR v0.2 flow: base_rate / spread / fee)
  - Run financial covenant extraction (CovenantIR v0.1 one-pass harness) on the same run

Important constraints (per user direction):
  - No deterministic/heuristic semantic backfill of missing tables/anchors.
  - Indexing quality is driven by the indexing prompt + single LLM call behavior.

Usage example:
  .venv/bin/python scripts/stress_test_indexing_v2_end_to_end.py \\
    --source-run-id full17-pricing-covenants-20260121-002143 \\  # 18 item_ids across 17 accessions (test_sample_2_agreements)
    --run-id-prefix idxA-e2e-20260122-hard4 \\
    --prompt prompts/indexing_v2_pricing_kernel.txt \\
    --prompt prompts/indexing_v2_pricing_kernel_agentic_v2.txt \\
    --item-id 0000950170-23-000122_2 \\
    --item-id 0001193125-23-000306_2 \\
    --item-id 0001140361-23-000246_8 \\
    --item-id 0000720032-96-000002_2
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from pipeline.config import Paths, REQUIRED_MODEL, REQUIRED_REASONING  # noqa: E402
from pipeline.contract_ir_v0_2_flow import (  # noqa: E402
    ContractIRFlowError,
    prepare_run_inputs_from_source,
    run_contractir_v0_2_flow,
)
from pipeline.indexing_v2 import run_indexing_v2  # noqa: E402
from pipeline.retrieval_v2 import render_snippets_v2  # noqa: E402
from pipeline.utils import load_manifest, manifest_items  # noqa: E402


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _prompt_short_name(path: Path) -> str:
    stem = path.stem
    # Keep run_ids under 80 chars; prompt stems can get long.
    return stem[:40]


def _validate_item_ids(source_paths: Paths, item_ids: Sequence[str]) -> List[str]:
    manifest = load_manifest(source_paths.manifest_path)
    items = manifest_items(manifest)
    known = {it.get("item_id") for it in items if isinstance(it, dict)}
    unknown = sorted({i for i in item_ids if i not in known})
    if unknown:
        raise SystemExit(f"Unknown item_id(s) not present in source manifest: {', '.join(unknown)}")
    return list(item_ids)


def _run_covenantir_batch(
    *,
    run_id: str,
    base_dir: Path,
    out_dir: Path,
    item_ids: Sequence[str],
    attempts: int,
    temperature: float,
    timeout_seconds: float,
) -> Tuple[int, Path]:
    cmd = [
        sys.executable,
        str(base_dir / "scripts" / "covenant_ir_v0_1_batch_runner.py"),
        "--run-id",
        run_id,
        "--base-dir",
        str(base_dir),
        "--out-dir",
        str(out_dir),
        "--prompt",
        "prompts/covenant_ir_financial_v0_1.txt",
        "--attempts",
        str(int(attempts)),
        "--temperature",
        str(float(temperature)),
        "--timeout-seconds",
        str(float(timeout_seconds)),
        "--max-items",
        str(int(len(item_ids))),
        "--model",
        REQUIRED_MODEL,
        "--reasoning",
        REQUIRED_REASONING,
    ]
    for item_id in item_ids:
        cmd.extend(["--item-id", item_id])

    proc = subprocess.run(cmd, cwd=str(base_dir), check=False)
    summary_path = out_dir / "summary.json"
    return (int(proc.returncode), summary_path)


def _run_one_prompt(
    *,
    source_paths: Paths,
    prompt_path: Path,
    out_run_id: str,
    item_ids: Sequence[str],
    retrieval_bandwidth: int,
    attempts: int,
    gateway_url: Optional[str],
    temperature: float,
    timeout_seconds: float,
) -> Dict[str, Any]:
    dest_paths = Paths(root=source_paths.root, run_id=out_run_id)
    prepare_run_inputs_from_source(dest_paths=dest_paths, source_paths=source_paths)

    # 1) Indexing (single-pass, prompt-driven).
    run_indexing_v2(
        dest_paths,
        item_ids,
        prompt_path,
        gateway_url=gateway_url,
        temperature=float(temperature),
        reasoning=REQUIRED_REASONING,
        gateway_timeout=float(timeout_seconds),
        concurrency=1,
        attempts=3,
        skip_existing=False,
    )

    # 2) Retrieval (provenance + snippet packs consumed by CovenantIR harness).
    render_snippets_v2(dest_paths, item_ids, bandwidth=int(retrieval_bandwidth))

    # 3) Pricing extractor (ContractIR v0.2 multi-pass kernel).
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

    pricing_summary_path = run_contractir_v0_2_flow(
        paths=dest_paths,
        item_ids=list(item_ids),
        indexing_prompt_path=prompt_path,  # recorded in summary.json even when skip_indexing=True
        retrieval_bandwidth=int(retrieval_bandwidth),
        base_rate_prompt_path=base_rate_prompt,
        spread_prompt_paths=spread_prompts,
        fee_prompt_paths=fee_prompts,
        gateway_url=gateway_url,
        timeout_seconds=float(timeout_seconds),
        temperature=float(temperature),
        attempts=int(attempts),
        concurrency=1,
        skip_indexing=True,
        skip_retrieval=True,
        fail_on_item_errors=False,
    )

    # 4) Covenant extractor (one-pass CovenantIR v0.1).
    cov_out_dir = dest_paths.run_dir / "covenantir_v0_1"
    cov_rc, cov_summary_path = _run_covenantir_batch(
        run_id=out_run_id,
        base_dir=dest_paths.root,
        out_dir=cov_out_dir,
        item_ids=item_ids,
        attempts=int(attempts),
        temperature=float(temperature),
        timeout_seconds=float(timeout_seconds),
    )

    out = {
        "schema_version": "indexing_v2_end_to_end_stress_test_v1",
        "created_at": int(time.time()),
        "source_run_id": source_paths.run_id,
        "run_id": out_run_id,
        "prompt_path": str(prompt_path),
        "item_ids": list(item_ids),
        "retrieval_bandwidth": int(retrieval_bandwidth),
        "llm": {
            "model": REQUIRED_MODEL,
            "reasoning_effort": REQUIRED_REASONING,
            "temperature": float(temperature),
            "timeout_seconds": float(timeout_seconds),
        },
        "pricing": {"summary_path": str(pricing_summary_path)},
        "covenants": {"exit_code": int(cov_rc), "summary_path": str(cov_summary_path)},
    }
    _write_json(dest_paths.run_dir / "end_to_end_stress_test_summary.json", out)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source-run-id", required=True)
    ap.add_argument("--run-id-prefix", required=True)
    ap.add_argument("--base-dir", default=".")
    ap.add_argument("--gateway-url", default=None)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--timeout-seconds", type=float, default=600.0)
    ap.add_argument("--attempts", type=int, default=3)
    ap.add_argument("--retrieval-bandwidth", type=int, default=400)
    ap.add_argument(
        "--prompt",
        action="append",
        dest="prompts",
        required=True,
        help="Indexing-v2 prompt file (repeatable). Each prompt creates its own run_id.",
    )
    ap.add_argument("--item-id", action="append", dest="item_ids", default=[])
    args = ap.parse_args()

    base_dir = Path(args.base_dir)
    source_paths = Paths(root=base_dir, run_id=str(args.source_run_id))
    if not source_paths.run_dir.exists():
        raise SystemExit(f"Source run does not exist: {source_paths.run_dir}")

    prompt_paths = [Path(p) for p in args.prompts]
    for p in prompt_paths:
        if not p.exists():
            raise SystemExit(f"Prompt not found: {p}")

    if not args.item_ids:
        raise SystemExit("Provide --item-id (repeatable) for the hard subset you want to stress test.")
    item_ids = _validate_item_ids(source_paths, [i for i in args.item_ids if isinstance(i, str) and i.strip()])

    combined: Dict[str, Any] = {
        "schema_version": "indexing_v2_end_to_end_stress_test_bundle_v1",
        "created_at": int(time.time()),
        "source_run_id": source_paths.run_id,
        "run_id_prefix": str(args.run_id_prefix),
        "item_ids": list(item_ids),
        "prompt_runs": [],
    }

    for prompt_path in prompt_paths:
        out_run_id = f"{args.run_id_prefix}-{_prompt_short_name(prompt_path)}"
        rec = _run_one_prompt(
            source_paths=source_paths,
            prompt_path=prompt_path,
            out_run_id=out_run_id,
            item_ids=item_ids,
            retrieval_bandwidth=int(args.retrieval_bandwidth),
            attempts=int(args.attempts),
            gateway_url=str(args.gateway_url) if args.gateway_url else None,
            temperature=float(args.temperature),
            timeout_seconds=float(args.timeout_seconds),
        )
        combined["prompt_runs"].append(
            {
                "prompt_path": str(prompt_path),
                "run_id": out_run_id,
                "summary_path": str(Paths(root=base_dir, run_id=out_run_id).run_dir / "end_to_end_stress_test_summary.json"),
            }
        )

    stamp = time.strftime("%Y%m%d_%H%M%S")
    out_path = PROJECT_ROOT / "scratch" / f"indexing_v2_end_to_end_stress_test_{stamp}.json"
    _write_json(out_path, combined)
    print(str(out_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
