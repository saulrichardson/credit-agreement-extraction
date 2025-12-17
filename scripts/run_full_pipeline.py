#!/usr/bin/env python
"""
End-to-end runner: indexing -> retrieval -> pricing structured -> metric definitions -> covenant snippets -> covenants combined.

Usage:
  poetry run python scripts/run_full_pipeline.py \
    --run-id pi-toc-sample \
    --items 0001731122-23-000003_2 \
    --gateway-url http://127.0.0.1:8000 \
    --model openai:gpt-5-nano
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import List


def run_cmd(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True)


def load_items(run_dir: Path) -> List[str]:
    manifest = json.loads((run_dir / "manifest.json").read_text())
    return [rec["item_id"] for rec in manifest.get("items", [])]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--items", help="Comma-separated item_ids; default: all from manifest")
    ap.add_argument("--gateway-url", default="http://127.0.0.1:8000")
    ap.add_argument("--model", default="openai:gpt-5-nano")
    ap.add_argument("--index-prompt", default="prompts/prompt_pricing_highlevel_simple_v1.txt")
    ap.add_argument("--pricing-prompt", default="vendor/edgar-covenants/prompts/prompt_pricing_second_pass_dg_nano_v2.txt")
    ap.add_argument("--temp", type=float, default=0.0)
    args = ap.parse_args()

    base = Path("runs")
    base /= args.run_id

    items = (
        [x.strip() for x in args.items.split(",") if x.strip()]
        if args.items
        else load_items(base)
    )

    # 1) Indexing
    run_cmd(
        [
            "poetry",
            "run",
            "python",
            "-m",
            "pipeline.run",
            "index",
            "--run-id",
            args.run_id,
            "--prompt",
            args.index_prompt,
            "--model",
            args.model,
            "--gateway-url",
            args.gateway_url,
            "--reasoning",
            "medium",
            "--temperature",
            str(args.temp),
        ]
    )

    # 2) Retrieval
    run_cmd(
        [
            "poetry",
            "run",
            "python",
            "-m",
            "pipeline.run",
            "retrieve",
            "--run-id",
            args.run_id,
        ]
    )

    # 3) Pricing structured extraction
    out_subdir = "pricing_second_pass_dg_nano_v2"
    run_cmd(
        [
            "poetry",
            "run",
            "python",
            "-m",
            "pipeline.run",
            "structured",
            "--run-id",
            args.run_id,
            "--prompt",
            args.pricing_prompt,
            "--model",
            args.model,
            "--gateway-url",
            args.gateway_url,
            "--reasoning",
            "medium",
            "--temperature",
            str(args.temp),
            "--output-subdir",
            out_subdir,
        ]
    )

    # 4) Covenant snippets (includes tables near covenants)
    run_cmd(
        [
            "poetry",
            "run",
            "python",
            "scripts/gen_covenant_snippets.py",
            "--run-id",
            args.run_id,
            "--items",
            ",".join(items),
        ]
    )

    # 5) Per-item: metric definitions + covenants combined
    for item in items:
        snippets_path = base / "retrieval" / f"{item}_snippets_highsimple.jsonl"
        if not snippets_path.exists():
            snippets_path = base / "retrieval" / f"{item}_snippets.jsonl"
        pricing_dir = base / "llm_qa" / out_subdir
        canonical = base / "normalized" / item / "canonical.txt"
        # metric definitions
        run_cmd(
            [
                "poetry",
                "run",
                "python",
                "scripts/run_metric_definitions.py",
                "--run-id",
                args.run_id,
                "--item",
                item,
                "--pricing-json",
                str(pricing_dir),
                "--snippets",
                str(snippets_path),
                "--canonical",
                str(canonical),
                "--gateway-url",
                args.gateway_url,
                "--model",
                args.model,
                "--temperature",
                str(args.temp),
            ]
        )
        # covenants combined
        cov_snips = base / "retrieval" / f"{item}_snippets_covenant.jsonl"
        cov_out = base / "llm_qa" / "covenants" / f"{item}_covenants_combined.json"
        run_cmd(
            [
                "poetry",
                "run",
                "python",
                "scripts/run_covenants_combined.py",
                "--run-id",
                args.run_id,
                "--item",
                item,
                "--snippets",
                str(cov_snips),
                "--out",
                str(cov_out),
                "--prompt",
                "prompts/covenants-v1.txt",
                "--gateway-url",
                args.gateway_url,
                "--model",
                args.model,
                "--temperature",
                str(args.temp),
            ]
        )
        print(f"[done] {item}")


if __name__ == "__main__":
    main()
