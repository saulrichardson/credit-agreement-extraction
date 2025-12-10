#!/usr/bin/env python
"""
Run the covenant extraction prompt once over all covenant snippets for an item.

Usage:
  poetry run python scripts/run_covenants_combined.py \
    --run-id pi-toc-sample \
    --namespace pi \
    --item 0001731122-23-000003_2 \
    --snippets runs/pi/pi-toc-sample/retrieval/0001731122-23-000003_2_snippets_covenant.jsonl \
    --out runs/pi/pi-toc-sample/llm_qa/covenants/0001731122-23-000003_2_covenants_combined.json \
    --prompt prompts/covenants-v1.txt \
    --gateway-url http://127.0.0.1:8000 \
    --model openai:gpt-5-nano
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import httpx


def stream_completion(url: str, payload: dict) -> str:
    parts: list[str] = []
    with httpx.stream("POST", url, json=payload, timeout=None) as resp:
        if resp.status_code >= 400:
            try:
                detail = resp.json()
            except Exception:
                detail = resp.text
            raise RuntimeError(f"Gateway error {resp.status_code}: {detail}")
        for raw in resp.iter_lines():
            if not raw:
                continue
            line = raw if isinstance(raw, str) else raw.decode()
            if not line.startswith("data:"):
                continue
            evt = json.loads(line[5:])
            evt_type = evt.get("type")
            if evt_type in {"response.output_text.delta", "response.content_part.delta"}:
                delta = evt.get("delta", "")
                if isinstance(delta, str):
                    parts.append(delta)
            elif evt_type == "response.completed":
                break
    return "".join(parts)


def load_snippets(snippets_path: Path) -> list[str]:
    snips: list[str] = []
    with snippets_path.open() as fh:
        for line in fh:
            obj = json.loads(line)
            snips.append(f"[{obj.get('anchor_id')}] {obj.get('snippet')}")
    return snips


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--namespace", default=None)
    ap.add_argument("--item", required=True)
    ap.add_argument("--snippets", required=True, help="Path to covenant snippets JSONL")
    ap.add_argument("--prompt", default="prompts/covenants-v1.txt")
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="openai:gpt-5-nano")
    ap.add_argument("--gateway-url", default="http://127.0.0.1:8000")
    ap.add_argument("--temperature", type=float, default=0.0)
    args = ap.parse_args()

    prompt = Path(args.prompt).read_text().strip()
    snippets = load_snippets(Path(args.snippets))
    user_content = "\n\n".join(snippets)

    payload = {
        "model": args.model,
        "input": [
            {"role": "system", "content": prompt},
            {"role": "user", "content": user_content},
        ],
        "reasoning": {"effort": "medium"},
        "temperature": args.temperature,
        "stream": True,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    text = stream_completion(f"{args.gateway_url}/v1/responses", payload)
    out_path.write_text(text)
    print(f"wrote {out_path} ({len(text)} chars)")


if __name__ == "__main__":
    main()
