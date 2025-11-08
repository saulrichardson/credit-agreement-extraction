#!/usr/bin/env python3
"""Assemble pricing excerpts per mode and call the pricing extraction LLM."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Iterable, Sequence

from openai import OpenAI

DEFAULT_PROMPT_PATH = Path("prompts/extraction.txt")
DEFAULT_PROMPT_VIEW = Path("docs/artifacts/semantic/prompt_view.txt")
DEFAULT_SEMANTIC_SCORES = Path("docs/artifacts/semantic/semantic_scores.txt")
DEFAULT_CHUNK_SCORES = Path("docs/artifacts/chunking/chunk_scores.txt")
DEFAULT_ANCHOR_JSON = Path("docs/artifacts/anchor_screen/prompt_view_pricing_anchors.json")
DEFAULT_OUTPUT_DIR = Path("docs/artifacts/pricing_extraction")
DEFAULT_PAYLOAD_DIR = Path("docs/artifacts/pricing_payloads")

ANCHOR_RE = re.compile(r"\s*⟦([^⟧]+)⟧")


def load_prompt_lines(prompt_view: Path) -> tuple[list[str], dict[str, int], dict[int, int]]:
    lines = prompt_view.read_text(encoding="utf-8").splitlines()
    anchor_to_idx: dict[str, int] = {}
    sorted_positions: list[int] = []
    for idx, line in enumerate(lines):
        match = ANCHOR_RE.match(line)
        if match:
            anchor_id = match.group(1)
            anchor_to_idx[anchor_id] = idx
            sorted_positions.append(idx)
    next_index: dict[int, int] = {}
    sorted_positions.sort()
    for i, current_idx in enumerate(sorted_positions):
        if i + 1 < len(sorted_positions):
            next_index[current_idx] = sorted_positions[i + 1]
    if sorted_positions:
        next_index.setdefault(sorted_positions[-1], len(lines))
    return lines, anchor_to_idx, next_index


def slice_by_range(
    lines: Sequence[str],
    anchor_map: dict[str, int],
    start_id: str,
    end_id: str,
) -> str:
    if start_id not in anchor_map or end_id not in anchor_map:
        missing = start_id if start_id not in anchor_map else end_id
        raise KeyError(f"Anchor {missing} not found in prompt view")
    start_idx = anchor_map[start_id]
    end_idx = anchor_map[end_id]
    if end_idx < start_idx:
        start_idx, end_idx = end_idx, start_idx
    snippet = lines[start_idx : end_idx + 1]
    return "\n".join(snippet).strip()


def gather_ranges_from_scores(scores_path: Path) -> list[tuple[str, str]]:
    data = json.loads(scores_path.read_text())
    ranges: list[tuple[str, str]] = []
    for item in data.get("scores", []):
        if float(item.get("score", 0)) == 1.0:
            rng = item.get("range") or []
            if len(rng) == 2:
                ranges.append((rng[0], rng[1]))
    return ranges


def gather_anchor_ids(anchor_json: Path) -> list[str]:
    data = json.loads(anchor_json.read_text())
    anchors = data.get("anchors", [])
    return [anchor for anchor in anchors if isinstance(anchor, str)]


def build_document(mode: str, args: argparse.Namespace) -> str:
    lines, anchor_map, next_anchor_idx = load_prompt_lines(args.prompt_view)
    snippets: list[str] = []
    if mode == "semantic":
        for start_id, end_id in gather_ranges_from_scores(args.semantic_scores):
            try:
                snippets.append(slice_by_range(lines, anchor_map, start_id, end_id))
            except KeyError as exc:  # noqa: PERF203
                print(f"[semantic] Skipping range {start_id}-{end_id}: {exc}", file=sys.stderr)
    elif mode == "chunk":
        for start_id, end_id in gather_ranges_from_scores(args.chunk_scores):
            try:
                snippets.append(slice_by_range(lines, anchor_map, start_id, end_id))
            except KeyError as exc:  # noqa: PERF203
                print(f"[chunk] Skipping range {start_id}-{end_id}: {exc}", file=sys.stderr)
    elif mode == "anchors":
        ordered_ids = gather_anchor_ids(args.anchor_json)
        for anchor_id in ordered_ids:
            if anchor_id not in anchor_map:
                print(f"[anchors] Anchor {anchor_id} missing from prompt view", file=sys.stderr)
                continue
            idx = anchor_map[anchor_id]
            end_idx = next_anchor_idx.get(idx, len(lines))
            snippets.append("\n".join(lines[idx:end_idx]))
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
    parser.add_argument("--output", type=Path, default=None, help="Optional output JSON path")
    return parser.parse_args()


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
