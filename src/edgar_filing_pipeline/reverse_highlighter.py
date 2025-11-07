"""Reverse highlighter utilities: split answers into claims and map them to anchors."""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

try:  # pragma: no cover
    from openai import OpenAI
except ImportError as exc:  # pragma: no cover
    raise RuntimeError(
        "openai package is required for reverse highlighting. Install with `pip install openai`."
    ) from exc


ANCHOR_PROMPT = """You map claims to anchor IDs from the supplied source text. Anchors appear as ⟦s000127⟧ or ⟦t10r02c03⟧ before their text.
For the claim below, return JSON:
{
  "status": "supported" | "unsupported",
  "anchors": ["s000127", ...],
  "rationale": "short explanation"
}
Rules:
- Only cite anchors that contain the exact supporting language.
- Use "unsupported" with ["N/A"] if no anchor supports the claim.
"""


@dataclass
class Claim:
    text: str
    index: int


@dataclass
class ClaimSupport:
    claim: Claim
    status: str
    anchors: List[str]
    rationale: str


class SupportMapper:
    def __init__(self, model: str = "gpt-5-nano", reasoning: str = "medium", max_attempts: int = 3) -> None:
        if not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY must be set for reverse highlighting.")
        self._client = OpenAI()
        self._model = model
        self._reasoning = reasoning
        self._max_attempts = max_attempts

    def map_claim(self, claim: Claim, source_text: str) -> ClaimSupport:
        user_prompt = f"Source text:\n<<<\n{source_text}\n>>>\n\nClaim #{claim.index}: {claim.text.strip()}"
        last_error: Exception | None = None
        for _ in range(self._max_attempts):
            try:
                response = self._client.responses.create(
                    model=self._model,
                    reasoning={"effort": self._reasoning},
                    input=[
                        {"role": "system", "content": ANCHOR_PROMPT},
                        {"role": "user", "content": user_prompt},
                    ],
                )
                payload = response.output_text.strip()
                data = json.loads(payload)
                status = data.get("status", "unsupported")
                anchors = data.get("anchors", [])
                rationale = data.get("rationale", "")
                if not isinstance(anchors, list):
                    anchors = []
                anchors = [str(a).strip() for a in anchors if str(a).strip()]
                return ClaimSupport(claim=claim, status=status, anchors=anchors, rationale=rationale)
            except Exception as exc:  # pragma: no cover
                last_error = exc
                time.sleep(1)
        raise RuntimeError(f"Support mapping failed for claim {claim.index}: {last_error}")


CLAUSE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")


def split_prose(text: str) -> List[Claim]:
    sentences = []
    for idx, chunk in enumerate(CLAUSE_SPLIT_RE.split(text.strip()), start=1):
        stripped = chunk.strip()
        if stripped:
            sentences.append(Claim(text=stripped, index=idx))
    return sentences


def split_code(text: str) -> List[Claim]:
    claims: List[Claim] = []
    current: List[str] = []
    idx = 1
    for line in text.splitlines():
        if not line.strip():
            if current:
                claims.append(Claim(text="\n".join(current).strip(), index=idx))
                idx += 1
                current = []
            continue
        current.append(line.rstrip())
    if current:
        claims.append(Claim(text="\n".join(current).strip(), index=idx))
    return claims


def parse_prompt_view(path: Path) -> dict[str, str]:
    anchors: dict[str, str] = {}
    current_id: str | None = None
    buffer: List[str] = []

    def flush() -> None:
        nonlocal current_id, buffer
        if current_id is None:
            buffer = []
            return
        anchors[current_id] = "".join(buffer).strip()
        current_id = None
        buffer = []

    with path.open() as handle:
        for line in handle:
            stripped = line.strip()
            if stripped.startswith("⟦") and "⟧" in stripped.split(" ", 1)[0]:
                flush()
                anchor_id = stripped.split("⟧", 1)[0][1:]
                current_id = anchor_id
                buffer.append(line)
            else:
                buffer.append(line)
        flush()
    return anchors


def build_html(claims: List[ClaimSupport], anchor_map: dict[str, str]) -> str:
    parts = ["<html><body>", "<h1>Reverse Highlight Results</h1>"]
    for result in claims:
        parts.append(f"<section><h2>Claim {result.claim.index}</h2>")
        parts.append(f"<p>{result.claim.text}</p>")
        parts.append(f"<p>Status: <strong>{result.status}</strong></p>")
        parts.append(f"<p>{result.rationale}</p>")
        if result.status != "unsupported":
            for anchor in result.anchors:
                snippet = anchor_map.get(anchor, "(anchor not found)")
                parts.append(f"<h4>{anchor}</h4><pre>{snippet}</pre>")
        parts.append("</section>")
    parts.append("</body></html>")
    return "\n".join(parts)
