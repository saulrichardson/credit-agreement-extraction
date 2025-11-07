"""Chunk canonical documents into fixed-size anchor windows."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

from .plan_validation import load_sentence_anchor_records


@dataclass
class Chunk:
    chunk_id: str
    start_id: str
    end_id: str
    summary: str
    text: str


def build_chunks(
    canonical_dir: Path,
    *,
    chunk_size: int = 40,
    stride: int = 20,
    max_snippet_chars: int = 1200,
) -> dict:
    anchors_path = canonical_dir / "anchors.tsv"
    canonical_path = canonical_dir / "canonical.txt"
    prompt_path = canonical_dir / "prompt_view.txt"

    if not anchors_path.exists() or not canonical_path.exists() or not prompt_path.exists():
        raise FileNotFoundError(
            f"Expected canonical bundle (anchors.tsv, canonical.txt, prompt_view.txt) under {canonical_dir}"
        )

    anchor_records = load_sentence_anchor_records(anchors_path)
    canonical_text = canonical_path.read_text(encoding="utf-8")
    prompt_lines = prompt_path.read_text(encoding="utf-8").splitlines()

    chunks: List[dict] = []
    total = len(anchor_records)
    if chunk_size <= 0 or stride <= 0:
        raise ValueError("chunk_size and stride must be positive integers")

    def slice_prompt(start_id: str, end_id: str) -> str:
        capture: List[str] = []
        grabbing = False
        for line in prompt_lines:
            if line.startswith(f"⟦{start_id}⟧"):
                grabbing = True
            if grabbing:
                capture.append(line)
            if line.startswith(f"⟦{end_id}⟧"):
                break
        return "\n".join(capture)

    for idx in range(0, total, stride):
        window = anchor_records[idx : idx + chunk_size]
        if not window:
            continue
        start_id = window[0]["anchor_id"]
        end_id = window[-1]["anchor_id"]
        start_offset = int(window[0]["start"])
        end_offset = int(window[-1]["end"])
        snippet = canonical_text[start_offset:end_offset]
        summary = snippet.strip()[:max_snippet_chars]
        prompt_snippet = slice_prompt(start_id, end_id)
        chunks.append(
            {
                "seg_id": f"chunk{len(chunks)+1:04d}",
                "name": f"Chunk {start_id}-{end_id}",
                "range": [start_id, end_id],
                "summary": summary,
                "tags": [],
                "text": prompt_snippet,
            }
        )

    return {
        "source": str(canonical_dir),
        "chunk_size": chunk_size,
        "stride": stride,
        "segments": chunks,
        "keywords": [],
    }
