#!/usr/bin/env python3
"""Send prompt_view.txt to an LLM to locate numeric financial covenant anchors."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from openai import OpenAI

SYSTEM_PROMPT = (
    "You are an expert credit analyst focused on covenant compliance. Given an anchorized "
    "prompt view, find sentences or table rows that state numeric financial covenant tests "
    "(leverage ratios, interest coverage, debt service ratios, fixed charge coverage, minimum liquidity, "
    "net worth tests, step-down schedules, etc.). Reply with valid JSON only."
)

USER_PROMPT_TEMPLATE = """
You will be given a full prompt view where each line begins with an anchor marker like ⟦s000123⟧.
Return JSON exactly as:
{{
  "document_has_financial_covenants": <true|false>,
  "anchors": [
    {{
      "anchor_id": "...",
      "excerpt": "...",
      "metric": "...",
      "reason": "..."
    }}
  ]
}}

Rules:
1. Flag an anchor only if it contains a numeric financial covenant requirement or ratio threshold.
   Examples: maximum Total Net Leverage Ratio of 4.00 to 1.00, minimum Interest Coverage Ratio of 2.25x,
   minimum liquidity of $50,000,000, step-down schedule for leverage, or similar objective tests that
   govern borrower performance. Ignore qualitative covenants without numbers.
2. `document_has_financial_covenants` must be true exactly when at least one anchor qualifies.
3. `metric` should be a concise label such as "Total Net Leverage Ratio" or "Minimum Liquidity".
4. Excerpts must be ≤200 characters copied verbatim from the prompt view.
5. If nothing qualifies, return an empty anchors list and set the boolean to false.
6. Never invent anchor IDs.

PROMPT VIEW START
---
{prompt_view}
---
PROMPT VIEW END
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Detect financial covenant anchors by forwarding prompt_view.txt to an LLM."
    )
    parser.add_argument("prompt_view", type=Path, help="Path to prompt_view.txt")
    parser.add_argument(
        "--model",
        default="gpt-5-mini",
        help="OpenAI model name (default: gpt-5-mini)",
    )
    parser.add_argument(
        "--reasoning",
        default="medium",
        help="Reasoning effort hint for the Responses API (default: medium)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Destination JSON file (defaults to <prompt_view>_covenant_anchors.json)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prompt_text = args.prompt_view.read_text(encoding="utf-8")
    user_prompt = USER_PROMPT_TEMPLATE.format(prompt_view=prompt_text)

    client = OpenAI()
    response = client.responses.create(
        model=args.model,
        reasoning={"effort": args.reasoning},
        input=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
    )
    payload = response.output_text.strip()
    data = json.loads(payload)

    output_path = (
        args.output
        if args.output is not None
        else args.prompt_view.with_name(args.prompt_view.stem + "_covenant_anchors.json")
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    print(f"Covenant analysis written to {output_path}")


if __name__ == "__main__":
    main()
