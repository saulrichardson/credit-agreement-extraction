"""LLM-backed relevance scoring for planner segments."""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

try:  # pragma: no cover - optional dependency validated at runtime
    from openai import OpenAI
except ImportError as exc:  # pragma: no cover
    raise RuntimeError(
        "openai package is required for scoring. Install with `pip install openai`."
    ) from exc


SCORER_SYSTEM_PROMPT = """You evaluate segments from a credit agreement to determine how directly they help us understand the borrower’s pricing structure. Score 1 when the segment contains the core pricing mechanics (actual rates/margins, leverage tiers, benchmark fallback or adjustment logic, fee formulas, etc.). Score 0.5 when it provides required supporting context (definitions or covenants that directly feed the pricing math). Score 0 when it has no bearing on pricing. Respond in JSON only."""

SCORER_USER_TEMPLATE = """Question: {question}

Segment metadata:
- id: {seg_id}
- name: {name}
- summary: {summary}
- tags: {tags}
- global_keywords: {keywords}

Return JSON exactly:
{{
  "score": <0|0.5|1>,
  "verdict": "<include|skip>",
  "rationale": "<concise reason tied to pricing>"
}}

Rules:
- Score 1 (pricing engine) only when the segment exposes concrete pricing mechanics (rate/margin tables, leverage tiers, benchmark fallback/adjustment logic, fee formulas, liquidity triggers that directly change rates).
- Score 0.5 (pricing support) when it provides definitions, covenants, or measurement/reporting conditions that you must know to interpret/apply the pricing engine but do not restate the mechanics.
- Score 0 when it is unrelated to pricing."""


@dataclass
class ScoringConfig:
    model: str = "gpt-5-nano"
    reasoning_effort: str = "medium"
    max_attempts: int = 3


@dataclass
class SegmentScore:
    seg_id: str
    name: str
    score: float
    verdict: str
    rationale: str
    range: list[str]
    tags: list[str]


class SegmentScorer:
    def __init__(self, config: Optional[ScoringConfig] = None) -> None:
        if not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY must be set to run the scoring module.")
        self._client = OpenAI()
        self._config = config or ScoringConfig()

    def score_segments(
        self,
        plan: Dict[str, Any],
        question: str,
    ) -> List[SegmentScore]:
        keywords = ", ".join(plan.get("keywords", [])) or "N/A"
        segments = plan.get("segments", [])
        results: List[SegmentScore] = []
        for segment in segments:
            prompt = SCORER_USER_TEMPLATE.format(
                question=question.strip(),
                seg_id=segment.get("seg_id", "<unknown>"),
                name=segment.get("name", ""),
                summary=segment.get("summary", ""),
                tags=", ".join(segment.get("tags", [])),
                keywords=keywords,
            )
            payload = self._call_model(prompt)
            try:
                parsed = json.loads(payload)
            except json.JSONDecodeError as exc:
                raise RuntimeError(
                    f"Failed to parse scorer output for {segment.get('seg_id')}: {payload}"
                ) from exc
            score = float(parsed.get("score", 0))
            verdict = parsed.get("verdict", "skip")
            rationale = parsed.get("rationale", "")
            results.append(
                SegmentScore(
                    seg_id=segment.get("seg_id", ""),
                    name=segment.get("name", ""),
                    score=score,
                    verdict=verdict,
                    rationale=rationale,
                    range=segment.get("range", []),
                    tags=segment.get("tags", []),
                )
            )
        results.sort(key=lambda item: item.score, reverse=True)
        return results

    def _call_model(self, user_prompt: str) -> str:
        last_error: Exception | None = None
        for _ in range(self._config.max_attempts):
            try:
                response = self._client.responses.create(
                    model=self._config.model,
                    reasoning={"effort": self._config.reasoning_effort},
                    input=[
                        {"role": "system", "content": SCORER_SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt},
                    ],
                )
                return response.output_text.strip()
            except Exception as exc:  # pragma: no cover - network/LLM failure
                last_error = exc
                time.sleep(1)
        raise RuntimeError(f"Scoring failed after {self._config.max_attempts} attempts: {last_error}")
