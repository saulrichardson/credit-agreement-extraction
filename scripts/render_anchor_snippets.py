#!/usr/bin/env python3
"""Materialise prompt_view excerpts around classified anchors."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

from edgar_filing_pipeline.prompt_view import (
    PromptView,
    load_prompt_view,
    render_anchor_snippets,
)

DEFAULT_GROUPS = ("fundamental_anchors", "pricing_anchors", "financial_covenant_anchors")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Gather prompt_view snippets around classified anchors with a configurable bandwidth."
    )
    parser.add_argument("--prompt-view", required=True, type=Path, help="Path to prompt_view.txt")
    parser.add_argument(
        "--classification",
        required=True,
        type=Path,
        help="JSON file containing anchor arrays (e.g., output of prompt_all_comprehensive_v2).",
    )
    parser.add_argument(
        "--groups",
        default=",".join(DEFAULT_GROUPS),
        help="Comma-separated JSON keys to extract from --classification (default: fundamental_anchors,pricing_anchors,financial_covenant_anchors)",
    )
    parser.add_argument(
        "--bandwidth",
        type=int,
        default=2,
        help="Number of neighbouring anchors to include on each side of every selected anchor (default: 2).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional output path. Defaults to <classification_stem>_snippets.txt next to the classification file.",
    )
    parser.add_argument(
        "--include-headers",
        action="store_true",
        help="Prepend section headers (the group name) before each snippet block.",
    )
    return parser.parse_args()


def gather_snippets(
    view: PromptView, group_name: str, anchors: Sequence[str], bandwidth: int
) -> tuple[list[str], list[str]]:
    snippets, missing = render_anchor_snippets(view, anchors, bandwidth=bandwidth)
    if not snippets:
        return [], missing
    return snippets, missing


def main() -> None:
    args = parse_args()
    prompt_view = load_prompt_view(args.prompt_view)
    data = json.loads(args.classification.read_text(encoding="utf-8"))

    group_names = [group.strip() for group in args.groups.split(",") if group.strip()]
    if not group_names:
        raise SystemExit("At least one group must be provided via --groups.")

    all_snippets: list[str] = []
    has_content = False
    for group in group_names:
        anchors = data.get(group) or []
        if not isinstance(anchors, list) or not anchors:
            continue
        snippets, missing = gather_snippets(prompt_view, group, anchors, args.bandwidth)
        for anchor_id in missing:
            print(f"[warn] Anchor {anchor_id} from group '{group}' not found in prompt view.", file=sys.stderr)
        if not snippets:
            continue
        has_content = True
        if args.include_headers:
            header = group.replace("_", " ").title()
            all_snippets.append(f"### {header}")
        all_snippets.extend(snippets)

    if not has_content:
        raise SystemExit("No snippets could be gathered for the requested groups.")

    output_path = (
        args.output
        if args.output is not None
        else args.classification.with_name(f"{args.classification.stem}_snippets.txt")
    )
    output_path.write_text("\n\n".join(all_snippets).strip() + "\n", encoding="utf-8")
    print(f"Wrote snippets -> {output_path}")


if __name__ == "__main__":
    main()
