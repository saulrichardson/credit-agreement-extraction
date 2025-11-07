"""Validation helpers for LLM-produced semantic plans."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Iterable, Mapping


def load_sentence_anchor_ids(anchors_path: Path) -> list[str]:
    return [record["anchor_id"] for record in load_sentence_anchor_records(anchors_path)]


def load_sentence_anchor_records(anchors_path: Path) -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    with anchors_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            anchor_id = row["anchor_id"]
            if anchor_id.startswith("s"):
                records.append(row)
    if not records:
        raise ValueError(f"No sentence anchors found in {anchors_path}")
    records.sort(key=lambda row: int(row["start"]))
    return records


def validate_plan(plan: Mapping[str, Any], sentence_ids: Iterable[str]) -> list[str]:
    ordered = list(sentence_ids)
    ordinal = {aid: idx for idx, aid in enumerate(ordered)}
    errors: list[str] = []

    segments = plan.get("segments")
    if not isinstance(segments, list) or not segments:
        return ["segments array missing or empty"]

    consumed: list[str] = []
    last_idx = -1
    for segment in segments:
        rng = segment.get("range")
        seg_id = segment.get("seg_id", "<unknown>")
        if not isinstance(rng, list) or len(rng) != 2:
            errors.append(f"segment {seg_id} range must be [start,end]")
            continue
        start, end = rng
        if start not in ordinal or end not in ordinal:
            errors.append(
                f"segment {seg_id} references unknown anchors {start}-{end}"
            )
            continue
        start_idx, end_idx = ordinal[start], ordinal[end]
        if start_idx > end_idx:
            errors.append(f"segment {seg_id} start {start} > end {end}")
            continue
        if start_idx != last_idx + 1:
            errors.append(
                f"segment {seg_id} begins at {start} but expected contiguous anchor"
            )
        consumed.extend(ordered[start_idx : end_idx + 1])
        last_idx = end_idx

    if last_idx != len(ordered) - 1:
        errors.append(
            f"segments end at {ordered[last_idx] if last_idx >= 0 else 'N/A'} but expected {ordered[-1]}"
        )

    valid_ids = set(ordered)

    def _check_spans(entries: list[Mapping[str, Any]], label: str) -> None:
        for entry in entries:
            spans = entry.get("spans", [])
            name = entry.get("name", "<unknown>")
            for span in spans:
                if not isinstance(span, list) or len(span) != 2:
                    errors.append(f"{label} {name} span must be [start,end]")
                    continue
                start, end = span
                if start not in valid_ids or end not in valid_ids:
                    errors.append(
                        f"{label} {name} references unknown anchors {start}-{end}"
                    )

    overlays = plan.get("overlays", [])
    if isinstance(overlays, list):
        _check_spans(overlays, "overlay")

    frames = plan.get("frames", [])
    if isinstance(frames, list):
        _check_spans(frames, "frame")
        for frame in frames:
            deps = frame.get("dependencies", [])
            name = frame.get("name", "<unknown>")
            for dep in deps:
                if dep not in valid_ids:
                    errors.append(
                        f"frame {name} dependency {dep} is not a known sentence anchor"
                    )
    return errors
