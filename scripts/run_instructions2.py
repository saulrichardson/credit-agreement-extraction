#!/usr/bin/env python
"""
Run the instructions-2 prompt against a snippets JSONL for a single item.

Usage:
  poetry run python scripts/run_instructions2.py \
    --run-id pi-toc-sample \
    --namespace pi \
    --item 0001731122-23-000003_2 \
    --snippets runs/pi/pi-toc-sample/retrieval/0001731122-23-000003_2_snippets_highsimple.jsonl \
    --out runs/pi/pi-toc-sample/llm_qa/instructions2_hot/0001731122-23-000003_2_instructions2.txt \
    --model openai:gpt-5-nano \
    --gateway-url http://127.0.0.1:8000 \
    --temperature 0
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import httpx


def load_snippets(snippets_path: Path, labels: set[str]) -> list[str]:
    snippets: list[str] = []
    with snippets_path.open() as fh:
        for line in fh:
            obj = json.loads(line)
            if obj.get("label") not in labels:
                continue
            snippets.append(f"[{obj['anchor_id']}] {obj['snippet']}")
    return snippets


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--namespace", default=None)
    ap.add_argument("--item", required=True)
    ap.add_argument(
        "--snippets",
        help="Path to snippets JSONL (default: runs/<ns>/<run>/retrieval/<item>_snippets.jsonl)",
    )
    ap.add_argument(
        "--prompt",
        default="prompts/instructions-2.txt",
        help="System prompt file for instructions-2",
    )
    ap.add_argument(
        "--out",
        required=True,
        help="Output path for raw model text",
    )
    ap.add_argument("--model", default="openai:gpt-5-nano")
    ap.add_argument("--gateway-url", default="http://127.0.0.1:8000")
    ap.add_argument("--temperature", type=float, default=0.0)
    args = ap.parse_args()

    base_dir = Path(".")
    run_dir = base_dir / "runs"
    if args.namespace:
        run_dir = run_dir / args.namespace
    run_dir = run_dir / args.run_id

    snippets_path = (
        Path(args.snippets)
        if args.snippets
        else run_dir / "retrieval" / f"{args.item}_snippets.jsonl"
    )
    prompt_path = Path(args.prompt)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    labels = {"fundamental", "pricing"}
    snippet_lines = load_snippets(snippets_path, labels)

    system_prompt = prompt_path.read_text().strip()
    user_content = system_prompt + "\n\nSNIPPETS:\n" + "\n\n".join(snippet_lines)

    payload = {
        "model": args.model,
        "input": [
            {"role": "user", "content": user_content},
        ],
        "stream": True,
        "temperature": args.temperature,
    }

    parts: list[str] = []
    with httpx.stream(
        "POST",
        f"{args.gateway_url}/v1/responses",
        json=payload,
        timeout=None,
    ) as resp:
        resp.raise_for_status()
        for raw in resp.iter_lines():
            if raw is None:
                continue
            line = raw.decode() if isinstance(raw, (bytes, bytearray)) else raw
            if not line.startswith("data:"):
                continue
            data = json.loads(line[5:])
            t = data.get("type")
            if t == "response.output_text.delta":
                delta = data.get("delta", "")
                if isinstance(delta, str):
                    parts.append(delta)
            elif t == "response.content_part.delta":
                delta = data.get("delta", "")
                if isinstance(delta, str):
                    parts.append(delta)
            elif t == "response.completed":
                break

    text = "".join(parts)
    out_path.write_text(text)
    print(f"wrote {out_path} ({len(text)} chars)")


if __name__ == "__main__":
    main()
