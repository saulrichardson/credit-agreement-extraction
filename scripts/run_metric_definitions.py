#!/usr/bin/env python
"""
Derive pricing metric definitions from snippets and canonical text.

Steps:
1) Read pricing JSON to get metric_ids.
2) Ask LLM (prompt_metric_core_terms_v1.txt) to return regex_phrase/search_tokens per metric using provided snippets.
3) Sweep canonical text with search_tokens to collect mention snippets.
4) Ask LLM (prompt_metric_definition_v1.txt) to return the shortest verbatim definition per metric from the mentions.

Usage (example):
  poetry run python scripts/run_metric_definitions.py \
    --run-id pi-toc-sample \
    --namespace pi \
    --item 0001731122-23-000003_2 \
    --pricing-json runs/pi/pi-toc-sample/llm_qa/pricing_second_pass_dg_nano_v2.txt \
    --snippets runs/pi/pi-toc-sample/retrieval/0001731122-23-000003_2_snippets_highsimple.jsonl \
    --canonical runs/pi/pi-toc-sample/normalized/0001731122-23-000003_2/canonical.txt \
    --gateway-url http://127.0.0.1:8000 \
    --model openai:gpt-5-nano
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List

import httpx


def load_snippets(snippets_path: Path) -> list[str]:
    snippets: list[str] = []
    with snippets_path.open() as fh:
        for line in fh:
            obj = json.loads(line)
            snippets.append(f"[{obj.get('anchor_id')}] {obj.get('snippet')}")
    return snippets


def stream_completion(url: str, payload: dict) -> str:
    parts: list[str] = []
    with httpx.stream("POST", url, json=payload, timeout=None) as resp:
        resp.raise_for_status()
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


def derive_metrics(pricing: dict) -> list[str]:
    metrics: set[str] = set()
    for m in pricing.get("metrics", []):
        name = m.get("name")
        if name:
            metrics.add(name)
    for scheme in pricing.get("tier_schemes", []):
        for tier in scheme.get("tiers", []):
            for cond in tier.get("conditions", []):
                if cond.get("metric"):
                    metrics.add(cond["metric"])
    return sorted(metrics)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--namespace", default=None)
    ap.add_argument("--item", required=True)
    ap.add_argument("--pricing-json", help="Path to pricing JSON", required=True)
    ap.add_argument("--snippets", required=True, help="Path to snippets JSONL")
    ap.add_argument("--canonical", required=True, help="Path to canonical.txt")
    ap.add_argument(
        "--gateway-url", default="http://127.0.0.1:8000", help="LLM gateway base URL"
    )
    ap.add_argument("--model", default="openai:gpt-5-nano")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--window", type=int, default=200, help="Context window for mentions")
    ap.add_argument(
        "--out-dir",
        help="Base output dir",
        default=None,
    )
    args = ap.parse_args()

    run_dir = Path("runs")
    if args.namespace:
        run_dir /= args.namespace
    run_dir /= args.run_id

    out_dir = Path(args.out_dir) if args.out_dir else run_dir / "llm_qa" / "metric_definitions"
    out_dir.mkdir(parents=True, exist_ok=True)

    pricing_path = Path(args.pricing_json)
    if pricing_path.is_dir():
        pricing_path = pricing_path / f"{args.item}.txt"
    snippets_path = Path(args.snippets)
    canonical_path = Path(args.canonical)

    pricing = json.loads(pricing_path.read_text())
    metric_ids = derive_metrics(pricing)
    snippets_text = load_snippets(snippets_path)

    # Step 1: core terms + tokens
    core_prompt = Path("prompts/prompt_metric_core_terms_v1.txt").read_text().strip()
    core_payload = {
        "model": args.model,
        "input": [
            {"role": "system", "content": core_prompt},
            {
                "role": "user",
                "content": json.dumps(
                    {"metric_ids": metric_ids, "pricing_json": pricing, "snippets": snippets_text}
                ),
            },
        ],
        "reasoning": {"effort": "medium"},
        "temperature": args.temperature,
        "stream": True,
    }
    core_resp = stream_completion(f"{args.gateway_url}/v1/responses", core_payload)
    core_path = out_dir / f"{args.item}_core_terms.json"
    core_path.write_text(core_resp)

    try:
        core_data = json.loads(core_resp)
    except json.JSONDecodeError:
        raise SystemExit(f"LLM core-terms response is not valid JSON: {core_path}")

    tokens_map: Dict[str, List[str]] = {}
    for entry in core_data.get("metrics", []):
        mid = entry.get("metric_id")
        if not mid:
            continue
        toks = entry.get("search_tokens") or []
        regex_phrase = entry.get("regex_phrase")
        if regex_phrase:
            toks.append(regex_phrase)
        if not toks:
            # fallback: split metric id
            toks = re.findall(r"[A-Za-z]+", mid)
        tokens_map[mid] = toks

    # Step 2: mentions collection
    canon = canonical_path.read_text()
    mentions_path = out_dir / f"{args.item}_metric_mentions.jsonl"
    with mentions_path.open("w") as fh:
        for mid, toks in tokens_map.items():
            seen_snips: set[str] = set()
            for tok in toks:
                for match in re.finditer(re.escape(tok), canon, flags=re.IGNORECASE):
                    s = max(0, match.start() - args.window)
                    e = min(len(canon), match.end() + args.window)
                    snippet = canon[s:e]
                    if snippet in seen_snips:
                        continue
                    seen_snips.add(snippet)
                    fh.write(
                        json.dumps(
                            {
                                "metric": mid,
                                "anchor": None,
                                "start": s,
                                "end": e,
                                "snippet": snippet,
                            }
                        )
                        + "\n"
                    )

    # Step 3: definition extraction
    def_prompt = Path("prompts/prompt_metric_definition_v1.txt").read_text().strip()
    definitions: Dict[str, dict] = {}
    for mid in tokens_map:
        snips = []
        with mentions_path.open() as fh:
            for line in fh:
                obj = json.loads(line)
                if obj.get("metric") == mid:
                    snips.append(obj)
        user_content = "\n\n".join(
            f"[{s['anchor'] or 'N/A'}] {s['snippet']}" for s in snips
        )
        payload = {
            "model": args.model,
            "input": [
                {"role": "system", "content": def_prompt},
                {"role": "user", "content": user_content},
                {"role": "user", "content": json.dumps({"metric_id": mid})},
            ],
            "reasoning": {"effort": "medium"},
            "temperature": args.temperature,
            "stream": True,
        }
        defs_resp = stream_completion(f"{args.gateway_url}/v1/responses", payload)
        try:
            definitions[mid] = json.loads(defs_resp)
        except json.JSONDecodeError:
            definitions[mid] = {"metric": mid, "anchors": [], "verbatim": None, "error": "invalid JSON"}

    defs_path = out_dir / f"{args.item}_metric_definitions_clean.json"
    defs_path.write_text(json.dumps(definitions, indent=2))

    print(f"core terms -> {core_path}")
    print(f"mentions    -> {mentions_path}")
    print(f"definitions -> {defs_path}")


if __name__ == "__main__":
    main()
