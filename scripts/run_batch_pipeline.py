#!/usr/bin/env python
"""
Batch runner for metric definitions + combined covenants across multiple items.

It wraps:
  - scripts/run_metric_definitions.py
  - scripts/run_covenants_combined.py

Assumes file layout:
  runs/<ns>/<run>/llm_qa/pricing_second_pass_dg_nano_v2_<item>.txt  (or shared v2 file)
  runs/<ns>/<run>/retrieval/<item>_snippets_highsimple.jsonl
  runs/<ns>/<run>/normalized/<item>/canonical.txt
  runs/<ns>/<run>/retrieval/<item>_snippets_covenant.jsonl

Usage:
  poetry run python scripts/run_batch_pipeline.py \
    --run-id pi-toc-sample --namespace pi \
    --items 0001731122-23-000003_2,ITEM2 \
    --gateway-url http://127.0.0.1:8000 \
    --model openai:gpt-5-nano
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


def resolve_pricing_json(ns: str | None, run_id: str, item: str) -> Path:
    base = Path("runs")
    if ns:
        base /= ns
    base /= run_id / "llm_qa"
    candidate_item = base / f"pricing_second_pass_dg_nano_v2_{item}.txt"
    candidate_shared = base / "pricing_second_pass_dg_nano_v2.txt"
    if candidate_item.exists():
        return candidate_item
    if candidate_shared.exists():
        return candidate_shared
    raise FileNotFoundError(f"Pricing JSON not found for {item}: {candidate_item} or {candidate_shared}")


def resolve_path(base: Path) -> Path:
    if not base.exists():
        raise FileNotFoundError(str(base))
    return base


def run_cmd(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--namespace", default=None)
    ap.add_argument("--items", required=True, help="Comma-separated list of item ids")
    ap.add_argument("--gateway-url", default="http://127.0.0.1:8000")
    ap.add_argument("--model", default="openai:gpt-5-nano")
    ap.add_argument("--temp", type=float, default=0.0)
    args = ap.parse_args()

    ns = args.namespace
    run_id = args.run_id
    items = [x.strip() for x in args.items.split(",") if x.strip()]

    for item in items:
        print(f"==> Processing {item}")
        pricing_json = resolve_pricing_json(ns, run_id, item)
        base = Path("runs")
        if ns:
            base /= ns
        base /= run_id
        snippets = resolve_path(base / "retrieval" / f"{item}_snippets_highsimple.jsonl")
        canonical = resolve_path(base / "normalized" / item / "canonical.txt")
        covenant_snips = resolve_path(base / "retrieval" / f"{item}_snippets_covenant.jsonl")

        # Metric definitions
        run_cmd(
            [
                "poetry",
                "run",
                "python",
                "scripts/run_metric_definitions.py",
                "--run-id",
                run_id,
                *(["--namespace", ns] if ns else []),
                "--item",
                item,
                "--pricing-json",
                str(pricing_json),
                "--snippets",
                str(snippets),
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

        # Covenants combined
        cov_out = base / "llm_qa" / "covenants" / f"{item}_covenants_combined.json"
        run_cmd(
            [
                "poetry",
                "run",
                "python",
                "scripts/run_covenants_combined.py",
                "--run-id",
                run_id,
                *(["--namespace", ns] if ns else []),
                "--item",
                item,
                "--snippets",
                str(covenant_snips),
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

        print(f"   Metric defs -> runs/{ns+'/'+run_id if ns else run_id}/llm_qa/metric_definitions/{item}_metric_definitions_clean.json")
        print(f"   Covenants   -> {cov_out}")


if __name__ == "__main__":
    main()
