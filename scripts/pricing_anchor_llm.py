#!/usr/bin/env python3
"""Send prompt_view chunks to an LLM and aggregate anchors for whole table."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from openai import OpenAI

SYSTEM_PROMPT = "You are reading a partial segment of a loan agreement."

USER_PROMPT_TEMPLATE = """
Each line begins with an anchor [[sXXXXX]].

1. Identify anchors: List every anchor whose text includes a numeric interest rate, margin, or percentage from a performance pricing grid.
  - Include all anchors that are part of such a table if any line in it contains a number.
  - Ignore definitions of terms or formulas that define variables.

2. Detect performance pricing presence: Indicate whether the document contains any reference to performance-based or leverage-based pricing tiers (for example, a “pricing level,” “grid,” or “Applicable Margin” table that changes with financial ratios).

Return your answer as JSON in the following format:
{
  "anchors": ["s000126", "s000127", "..."],
  "has_performance_pricing": true
}

Output only valid JSON. Document chunk below:
---
{chunk}
---
"""

DEFAULT_OUTPUT_DIR = Path("docs/artifacts/anchor_screen")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Chunked LLM anchor screener")
    parser.add_argument("prompt_view", type=Path)
    parser.add_argument("--chunk-size", type=int, default=200, help="Number of lines per chunk")
    parser.add_argument("--model", default="gpt-5-nano")
    parser.add_argument("--reasoning", default="medium")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    lines = args.prompt_view.read_text(encoding="utf-8").splitlines()
    client = OpenAI()
    anchors: list[str] = []
    has_pricing = False
    for start in range(0, len(lines), args.chunk_size):
        chunk = "\n".join(lines[start : start + args.chunk_size])
        response = client.responses.create(
            model=args.model,
            reasoning={"effort": args.reasoning},
            input=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": USER_PROMPT_TEMPLATE.format(chunk=chunk)},
            ],
        )
        payload = json.loads(response.output_text.strip())
        anchors.extend(payload.get("anchors", []))
        has_pricing = has_pricing or bool(payload.get("has_performance_pricing"))
    anchors = sorted({aid.strip(): None for aid in anchors}.keys())
    output_path = args.output or DEFAULT_OUTPUT_DIR / f"{args.prompt_view.stem}_pricing_anchors.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps({"anchors": anchors, "has_performance_pricing": has_pricing}, indent=2),
        encoding="utf-8",
    )
    print(f"Anchor screen -> {output_path}")


if __name__ == "__main__":
    main()
