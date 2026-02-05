#!/usr/bin/env python
"""
Run the "traditional DG" v2 pricing pipeline end-to-end on an existing run:
  normalize (already done) -> index-v2 -> retrieve-v2 -> structured-v2 -> audit -> definitions-v2

This is intended for iterative prompt experiments where you want deterministic artifacts on disk
and a repeatable command sequence.

Example:
  poetry run python scripts/run_dg_v2_pricing_flow.py \
    --run-id pi-dg-iter-20260112 \
    --index-prompt prompts/indexing_v2.txt \
    --structured-prompt prompts/prompt_pricing_second_pass_dg_nano_v2_tuned_v2_strict_metric_consistency_v4.txt \
    --qa-subdir dg_strict_metric_consistency_v4_full \
    --definitions-prompt prompts/definitions_v2_metrics_rates_anchorrefs_v1.txt \
    --definitions-subdir defs_anchorrefs_v1_metric_consistency_v4_full
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import List


def _run(cmd: List[str]) -> None:
    subprocess.run(cmd, check=True)


def _split_items(items: str | None) -> list[str]:
    if not items:
        return []
    return [x.strip() for x in items.split(",") if x.strip()]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--items", help="Comma-separated item_ids (defaults to all from manifest when omitted)")

    ap.add_argument("--index-prompt", default="prompts/indexing_v2.txt")

    ap.add_argument(
        "--structured-prompt",
        default="prompts/prompt_pricing_second_pass_dg_nano_v2_tuned_v2_strict_metric_consistency_v4.txt",
    )
    ap.add_argument(
        "--qa-subdir",
        default=None,
        help="Output subdir under runs/<run_id>/llm_qa/ (defaults to structured prompt stem when omitted)",
    )

    ap.add_argument("--definitions-prompt", default="prompts/definitions_v2_metrics_rates_anchorrefs_v1.txt")
    ap.add_argument(
        "--definitions-subdir",
        default=None,
        help="Output subdir under runs/<run_id>/definitions_v2/ (defaults to --qa-subdir when omitted)",
    )

    ap.add_argument(
        "--compiler-prompt",
        default=None,
        help=(
            "Optional: run the standalone metric definitions compiler stage after structured-v2. "
            "Path to prompt file (e.g., prompts/definitions_compiler_v1_metrics_ast_v2.txt)."
        ),
    )
    ap.add_argument(
        "--compiler-subdir",
        default=None,
        help="Optional: output subdir under runs/<run_id>/definitions_compiler_v1/ (defaults to 'compiled_v1').",
    )

    ap.add_argument("--concurrency", type=int, default=3)
    args = ap.parse_args()

    items = _split_items(args.items)

    qa_subdir = args.qa_subdir
    if qa_subdir is None:
        qa_subdir = Path(args.structured_prompt).stem

    defs_subdir = args.definitions_subdir or qa_subdir

    base_cmd = ["poetry", "run", "python", "-m", "src.pipeline.run"]

    # 1) Indexing (v2)
    cmd = base_cmd + [
        "index-v2",
        "--run-id",
        args.run_id,
        "--prompt",
        args.index_prompt,
        "--concurrency",
        str(args.concurrency),
    ]
    for item in items:
        cmd += ["--item-id", item]
    _run(cmd)

    # 2) Retrieval (v2)
    cmd = base_cmd + ["retrieve-v2", "--run-id", args.run_id]
    for item in items:
        cmd += ["--item-id", item]
    _run(cmd)

    # 3) Structured extraction (v2)
    cmd = base_cmd + [
        "structured-v2",
        "--run-id",
        args.run_id,
        "--prompt",
        args.structured_prompt,
        "--output-subdir",
        qa_subdir,
        "--concurrency",
        str(args.concurrency),
    ]
    for item in items:
        cmd += ["--item-id", item]
    _run(cmd)

    # 4) Audit structured outputs (JSON validity + lightweight grounding checks)
    _run(
        [
            "poetry",
            "run",
            "python",
            "scripts/audit_dg_json_failures.py",
            "--run-id",
            args.run_id,
            "--qa-subdir",
            qa_subdir,
        ]
    )

    # 4b) Optional: compile rich metric definitions (single call per metric).
    if args.compiler_prompt:
        cmd = base_cmd + [
            "definitions-compiler-v1",
            "--run-id",
            args.run_id,
            "--qa-subdir",
            qa_subdir,
            "--compiler-prompt",
            args.compiler_prompt,
            "--output-subdir",
            args.compiler_subdir or "compiled_v1",
            "--concurrency",
            str(args.concurrency),
        ]
        for item in items:
            cmd += ["--item-id", item]
        _run(cmd)

    # 5) Definitions v2 (uses indexing_v2 definitions_anchor_range if present)
    cmd = base_cmd + [
        "definitions-v2",
        "--run-id",
        args.run_id,
        "--qa-subdir",
        qa_subdir,
        "--definitions-prompt",
        args.definitions_prompt,
        "--output-subdir",
        defs_subdir,
        "--concurrency",
        str(args.concurrency),
    ]
    for item in items:
        cmd += ["--item-id", item]
    _run(cmd)

    print(f"[done] run_id={args.run_id} qa_subdir={qa_subdir} definitions_subdir={defs_subdir}")
    print(f"[audit] runs/{args.run_id}/report/dg_json_audit_{qa_subdir}.json")
    print(f"[defs]  runs/{args.run_id}/definitions_v2/{defs_subdir}/")


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as exc:
        print(f"[error] command failed with exit code {exc.returncode}", file=sys.stderr)
        raise
