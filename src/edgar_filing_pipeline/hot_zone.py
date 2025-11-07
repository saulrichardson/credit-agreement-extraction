from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence


@dataclass(frozen=True)
class PromptEntry:
    anchor_id: str
    text: str


@dataclass(frozen=True)
class SelectedSegment:
    seg_id: str
    name: str
    range: Sequence[str]
    score: float
    verdict: str | None


def _parse_prompt_view(path: Path) -> List[PromptEntry]:
    entries: List[PromptEntry] = []
    current_id: str | None = None
    buffer: List[str] = []

    def flush() -> None:
        nonlocal current_id, buffer
        if current_id is None:
            buffer = []
            return
        entries.append(
            PromptEntry(anchor_id=current_id, text="".join(buffer).rstrip("\n") + "\n")
        )
        current_id = None
        buffer = []

    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.lstrip()
            if stripped.startswith("⟦") and "⟧" in stripped.split(" ", 1)[0]:
                flush()
                raw_id = stripped.split("⟧", 1)[0]
                anchor_id = raw_id.strip()[1:]
                buffer.append(line)
                current_id = anchor_id
            else:
                buffer.append(line)
        flush()

    if not entries:
        raise ValueError(f"No anchors parsed from {path}")
    return entries


def _load_selected_segments(
    scores_path: Path,
    threshold: float,
    verdict: str | None,
) -> tuple[str, str, List[SelectedSegment]]:
    data = json.loads(scores_path.read_text())
    question = data.get("question", "")
    plan_path = data.get("plan_path", "")
    segments: List[SelectedSegment] = []
    for item in data.get("scores", []):
        score = float(item.get("score", 0))
        if score < threshold:
            continue
        item_verdict = item.get("verdict")
        if verdict and item_verdict != verdict:
            continue
        seg_range = item.get("range")
        if not seg_range or len(seg_range) != 2:
            raise ValueError(f"Segment {item.get('seg_id')} missing valid range")
        segments.append(
            SelectedSegment(
                seg_id=item.get("seg_id", ""),
                name=item.get("name", ""),
                range=tuple(seg_range),
                score=score,
                verdict=item_verdict,
            )
        )
    if not segments:
        raise ValueError(
            f"No segments in {scores_path} met threshold {threshold}"
            + (f" with verdict={verdict}" if verdict else "")
        )
    return question, plan_path, segments


def _build_anchor_index(entries: Sequence[PromptEntry]) -> dict[str, int]:
    index: dict[str, int] = {}
    for idx, entry in enumerate(entries):
        if entry.anchor_id in index:
            raise ValueError(f"Duplicate anchor id {entry.anchor_id} in prompt view")
        index[entry.anchor_id] = idx
    return index


def build_hot_zone(
    scores_path: Path,
    prompt_view_path: Path,
    output_path: Path,
    threshold: float = 1.0,
    verdict: str | None = None,
) -> None:
    entries = _parse_prompt_view(prompt_view_path)
    anchor_index = _build_anchor_index(entries)
    question, plan_path, segments = _load_selected_segments(scores_path, threshold, verdict)
    segments.sort(key=lambda seg: anchor_index.get(seg.range[0], float("inf")))

    included = [False] * len(entries)
    for seg in segments:
        start_id, end_id = seg.range
        if start_id not in anchor_index or end_id not in anchor_index:
            raise ValueError(f"Segment {seg.seg_id} references unknown anchors {seg.range}")
        start_idx = anchor_index[start_id]
        end_idx = anchor_index[end_id]
        if start_idx > end_idx:
            raise ValueError(f"Segment {seg.seg_id} has inverted range {seg.range}")
        for idx in range(start_idx, end_idx + 1):
            included[idx] = True

    lines: List[str] = []
    lines.append(f"# Pricing Hot Zone (score ≥ {threshold})\n")
    lines.append(f"Source plan: {plan_path}")
    lines.append(f"Scores: {scores_path}")
    if question:
        lines.append(f"Question: {question}")
    lines.append(f"Prompt view: {prompt_view_path}")
    lines.append("")
    lines.append("## Included segments")
    for seg in segments:
        lines.append(
            f"- {seg.seg_id}: {seg.name or 'unnamed'} (range {seg.range[0]}–{seg.range[1]}, score={seg.score})"
        )
    total_included = sum(1 for flag in included if flag)
    lines.append("")
    lines.append(f"Total anchors included: {total_included}\n")
    lines.append("## Text\n")
    for idx, entry in enumerate(entries):
        if included[idx]:
            lines.append(entry.text.rstrip("\n"))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines).rstrip("\n") + "\n", encoding="utf-8")
