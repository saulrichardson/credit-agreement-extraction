#!/usr/bin/env python3
"""Assemble pricing excerpts per mode and call the pricing extraction LLM."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable

from openai import OpenAI
from edgar_filing_pipeline.prompt_view import (
    load_prompt_view,
    render_anchor_snippets,
    slice_by_anchor_range,
)

DEFAULT_PROMPT_PATH = Path("prompts/extraction.txt")
DEFAULT_PROMPT_VIEW = Path("docs/artifacts/semantic/prompt_view.txt")
DEFAULT_SEMANTIC_SCORES = Path("docs/artifacts/semantic/semantic_scores.txt")
DEFAULT_CHUNK_SCORES = Path("docs/artifacts/chunking/chunk_scores.txt")
DEFAULT_ANCHOR_JSON = Path("docs/artifacts/anchor_screen/prompt_view_pricing_anchors.json")
DEFAULT_OUTPUT_DIR = Path("docs/artifacts/pricing_extraction")
DEFAULT_PAYLOAD_DIR = Path("docs/artifacts/pricing_payloads")

def gather_ranges_from_scores(scores_path: Path) -> list[tuple[str, str]]:
    data = json.loads(scores_path.read_text())
    ranges: list[tuple[str, str]] = []
    for item in data.get("scores", []):
        if float(item.get("score", 0)) == 1.0:
            rng = item.get("range") or []
            if len(rng) == 2:
                ranges.append((rng[0], rng[1]))
    return ranges


def gather_anchor_ids_from_json(anchor_json: Path) -> list[str]:
    data = json.loads(anchor_json.read_text())
    anchors = data.get("anchors", [])
    return [anchor for anchor in anchors if isinstance(anchor, str)]


def gather_anchor_ids_from_first_pass(result_path: Path, groups: Iterable[str]) -> list[str]:
    data = json.loads(result_path.read_text())
    seen: set[str] = set()
    ordered: list[str] = []
    for group in groups:
        values = data.get(group)
        if not isinstance(values, list):
            continue
        for anchor in values:
            if isinstance(anchor, str) and anchor not in seen:
                seen.add(anchor)
                ordered.append(anchor)
    return ordered


def build_document(mode: str, args: argparse.Namespace) -> str:
    prompt_view = load_prompt_view(args.prompt_view)
    snippets: list[str] = []
    if mode == "semantic":
        for start_id, end_id in gather_ranges_from_scores(args.semantic_scores):
            try:
                snippets.append(slice_by_anchor_range(prompt_view, start_id, end_id))
            except KeyError as exc:  # noqa: PERF203
                print(f"[semantic] Skipping range {start_id}-{end_id}: {exc}", file=sys.stderr)
    elif mode == "chunk":
        for start_id, end_id in gather_ranges_from_scores(args.chunk_scores):
            try:
                snippets.append(slice_by_anchor_range(prompt_view, start_id, end_id))
            except KeyError as exc:  # noqa: PERF203
                print(f"[chunk] Skipping range {start_id}-{end_id}: {exc}", file=sys.stderr)
    elif mode == "anchors":
        if args.first_pass_result:
            ordered_ids = gather_anchor_ids_from_first_pass(
                args.first_pass_result,
                args.pricing_anchor_groups,
            )
        else:
            ordered_ids = gather_anchor_ids_from_json(args.anchor_json)
        anchor_snippets, missing = render_anchor_snippets(
            prompt_view,
            ordered_ids,
            bandwidth=max(0, args.anchor_bandwidth),
        )
        for anchor_id in missing:
            print(f"[anchors] Anchor {anchor_id} missing from prompt view", file=sys.stderr)
        snippets.extend(anchor_snippets)
    else:  # pragma: no cover - defensive
        raise ValueError(f"Unknown mode {mode}")

    if not snippets:
        raise SystemExit(f"No snippets gathered for mode '{mode}'. Did you run the upstream workflow?")
    return "\n\n".join(snippets)


def save_payload(mode: str, payload_text: str, prompt_path: Path) -> None:
    DEFAULT_PAYLOAD_DIR.mkdir(parents=True, exist_ok=True)
    payload_path = DEFAULT_PAYLOAD_DIR / f"{mode}_payload.txt"
    payload_path.write_text(
        f"PROMPT_FILE: {prompt_path}\n\n---DOCUMENT---\n{payload_text}\n",
        encoding="utf-8",
    )


def call_llm(prompt_text: str, document_text: str) -> str:
    client = OpenAI()
    response = client.responses.create(
        model="gpt-5-nano",
        reasoning={"effort": "medium"},
        input=[
            {
                "role": "user",
                "content": f"{prompt_text}\n\nDocument excerpt:\n{document_text}",
            }
        ],
    )
    return response.output_text.strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run pricing extraction on selected snippets.")
    parser.add_argument("mode", choices=["semantic", "chunk", "anchors"], help="Snippet selection mode")
    parser.add_argument("--prompt", type=Path, default=DEFAULT_PROMPT_PATH)
    parser.add_argument("--prompt-view", dest="prompt_view", type=Path, default=DEFAULT_PROMPT_VIEW)
    parser.add_argument("--semantic-scores", dest="semantic_scores", type=Path, default=DEFAULT_SEMANTIC_SCORES)
    parser.add_argument("--chunk-scores", dest="chunk_scores", type=Path, default=DEFAULT_CHUNK_SCORES)
    parser.add_argument("--anchor-json", dest="anchor_json", type=Path, default=DEFAULT_ANCHOR_JSON)
    parser.add_argument(
        "--first-pass-result",
        dest="first_pass_result",
        type=Path,
        help="Optional JSON classification output; when present, pricing anchors are derived from the listed groups.",
    )
    parser.add_argument(
        "--pricing-anchor-groups",
        dest="pricing_anchor_groups",
        default="fundamental_anchors,pricing_anchors",
        help="Comma-separated JSON keys to pull from --first-pass-result for pricing context.",
    )
    parser.add_argument(
        "--anchor-bandwidth",
        dest="anchor_bandwidth",
        type=int,
        default=1,
        help="Number of neighbouring anchors to include on each side when materialising anchor snippets.",
    )
    parser.add_argument("--output", type=Path, default=None, help="Optional output JSON path")
    args = parser.parse_args()
    args.pricing_anchor_groups = [
        group.strip()
        for group in str(args.pricing_anchor_groups).split(",")
        if group.strip()
    ]
    if args.first_pass_result and not args.pricing_anchor_groups:
        raise SystemExit("At least one pricing anchor group must be specified when using --first-pass-result.")
    return args


def main() -> None:
    args = parse_args()
    prompt_text = args.prompt.read_text(encoding="utf-8").strip()
    document_text = build_document(args.mode, args)
    save_payload(args.mode, document_text, args.prompt)
    result_text = call_llm(prompt_text, document_text)

    output_path = (
        args.output
        if args.output is not None
        else DEFAULT_OUTPUT_DIR / f"{args.mode}_pricing.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(result_text + "\n", encoding="utf-8")
    print(f"Saved pricing extraction output -> {output_path}")


if __name__ == "__main__":
    main()
