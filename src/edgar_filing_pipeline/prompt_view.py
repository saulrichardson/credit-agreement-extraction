from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

ANCHOR_PATTERN = re.compile(r"\s*⟦([^⟧]+)⟧")


@dataclass
class PromptView:
    lines: list[str]
    anchor_to_line: dict[str, int]
    anchor_order: list[str]
    anchor_positions: dict[str, int]
    next_anchor_line: dict[int, int]


def load_prompt_view(path: Path) -> PromptView:
    lines = path.read_text(encoding="utf-8").splitlines()
    anchor_to_line: dict[str, int] = {}
    anchor_order: list[str] = []

    for idx, line in enumerate(lines):
        match = ANCHOR_PATTERN.match(line)
        if match:
            anchor_id = match.group(1)
            anchor_to_line[anchor_id] = idx
            anchor_order.append(anchor_id)

    anchor_positions = {anchor_id: pos for pos, anchor_id in enumerate(anchor_order)}
    next_anchor_line: dict[int, int] = {}
    for pos, anchor_id in enumerate(anchor_order[:-1]):
        current_idx = anchor_to_line[anchor_id]
        next_id = anchor_order[pos + 1]
        next_anchor_line[current_idx] = anchor_to_line[next_id]
    if anchor_order:
        last_idx = anchor_to_line[anchor_order[-1]]
        next_anchor_line.setdefault(last_idx, len(lines))

    return PromptView(
        lines=lines,
        anchor_to_line=anchor_to_line,
        anchor_order=anchor_order,
        anchor_positions=anchor_positions,
        next_anchor_line=next_anchor_line,
    )


def slice_by_anchor_range(view: PromptView, start_anchor: str, end_anchor: str) -> str:
    if start_anchor not in view.anchor_to_line or end_anchor not in view.anchor_to_line:
        missing = start_anchor if start_anchor not in view.anchor_to_line else end_anchor
        raise KeyError(f"Anchor {missing} not found in prompt view")
    start_idx = view.anchor_to_line[start_anchor]
    end_idx = view.anchor_to_line[end_anchor]
    if end_idx < start_idx:
        start_idx, end_idx = end_idx, start_idx
    snippet = view.lines[start_idx : end_idx + 1]
    return "\n".join(snippet).strip()


def render_anchor_snippets(
    view: PromptView,
    anchor_ids: Sequence[str],
    *,
    bandwidth: int = 0,
) -> tuple[list[str], list[str]]:
    if not anchor_ids or not view.anchor_order:
        return [], list(anchor_ids)

    width = max(0, bandwidth)
    missing: list[str] = []
    selected_positions: set[int] = set()
    total = len(view.anchor_order)

    for anchor_id in anchor_ids:
        pos = view.anchor_positions.get(anchor_id)
        if pos is None:
            missing.append(anchor_id)
            continue
        start = max(0, pos - width)
        end = min(total - 1, pos + width)
        for idx in range(start, end + 1):
            selected_positions.add(idx)

    if not selected_positions:
        return [], missing

    sorted_positions = sorted(selected_positions)
    merged: list[tuple[int, int]] = []
    block_start = sorted_positions[0]
    block_end = block_start
    for pos in sorted_positions[1:]:
        if pos == block_end + 1:
            block_end = pos
        else:
            merged.append((block_start, block_end))
            block_start = block_end = pos
    merged.append((block_start, block_end))

    snippets: list[str] = []
    for start_pos, end_pos in merged:
        start_anchor = view.anchor_order[start_pos]
        end_anchor = view.anchor_order[end_pos]
        start_line = view.anchor_to_line[start_anchor]
        end_line_idx = view.anchor_to_line[end_anchor]
        end_line = view.next_anchor_line.get(end_line_idx, len(view.lines))
        excerpt = "\n".join(view.lines[start_line:end_line]).strip()
        if excerpt:
            snippets.append(excerpt)

    return snippets, missing
