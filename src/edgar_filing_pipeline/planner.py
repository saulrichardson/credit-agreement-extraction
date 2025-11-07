"""LLM-backed semantic planner for anchored documents."""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

try:  # pragma: no cover - optional dependency validated at runtime
    from openai import OpenAI
except ImportError as exc:  # pragma: no cover - defensive
    raise RuntimeError(
        "openai package is required for planning. Install with `pip install openai`."
    ) from exc


_SYSTEM_PROMPT = """You are building a semantic index for a credit agreement. Partition the document into contiguous segments that maximize within-segment cohesion and minimize cross-segment leakage. Think information-theoretically: each segment should cover a coherent block (definitions, pricing mechanics, covenants, defaults, etc.) while staying small enough for downstream retrieval. Use only anchor IDs provided in the prompt view. Output JSON only."""

_CONTRACT_PROMPT = """JSON schema:\n{\n  \"segments\": [{\"seg_id\":\"segNN\",\"name\":\"...\",\"range\":[\"s#####\",\"s#####\"],\"summary\":\"...\",\"tags\":[...] }],\n  \"overlays\": [{\"name\":\"...\",\"spans\":[[\"s#####\",\"s#####\"],...],\"purpose\":\"...\"}],\n  \"frames\": [{\"name\":\"...\",\"goal\":\"...\",\"candidate_ranges\":[[\"s#####\",\"s#####\"]],\"dependencies\":[\"s#####\",...],\"signals\":[...] }],\n  \"gaps\": [\"...\"],\n  \"keywords\": [\"...\"],\n  \"cross_refs\": [\"...\"]\n}\nRules:\n1. Segments must cover s000001 through the final anchor exactly once, in order. Avoid trivial spans (<20 anchors) unless necessary and avoid mega-spans (>400 anchors) unless unavoidable.\n2. Overlays/frames may overlap but must cite valid anchor IDs. Omit \"cells\" unless real table anchors exist.\n3. Prioritize overlays/frames that surface pricing mechanics (Applicable Margin, Benchmark Replacement), leverage/liquidity covenants, and any benchmark fallback or exception dependencies.\n4. Keywords should be phrases retrieval can search to re-surface these sections.\n5. If you cannot comply, return {\"error\":\"reason\"}. JSON only."""


@dataclass
class PlannerConfig:
    model: str = "gpt-5-nano"
    reasoning_effort: str = "medium"
    max_attempts: int = 3
    focus_hint: str = "pricing mechanics and covenants"


class SemanticPlanner:
    def __init__(self, config: Optional[PlannerConfig] = None) -> None:
        if not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY must be set to use the semantic planner.")
        self._client = OpenAI()
        self._config = config or PlannerConfig()

    def build_plan(self, prompt_view: str) -> dict[str, Any]:
        user_prompt = (
            f"FOCUS_HINT:\n{self._config.focus_hint}\n\nCONTRACT:\n{_CONTRACT_PROMPT}\n\n"
            f"PROMPT VIEW:\n---\n{prompt_view}\n---"
        )

        last_error: Exception | None = None
        for attempt in range(1, self._config.max_attempts + 1):
            try:
                response = self._client.responses.create(
                    model=self._config.model,
                    reasoning={"effort": self._config.reasoning_effort},
                    input=[
                        {"role": "system", "content": _SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt},
                    ],
                )
                payload = response.output_text.strip()
                return json.loads(payload)
            except Exception as exc:  # pragma: no cover - network/LLM failure
                last_error = exc
                time.sleep(1)
        raise RuntimeError(f"Planner failed after {self._config.max_attempts} attempts: {last_error}")


def load_prompt_view(bundle_dir: Path) -> str:
    prompt_path = bundle_dir / "prompt_view.txt"
    if not prompt_path.exists():
        raise FileNotFoundError(f"Prompt view not found at {prompt_path}")
    return prompt_path.read_text(encoding="utf-8")


def load_anchors_path(bundle_dir: Path) -> Path:
    anchors_path = bundle_dir / "anchors.tsv"
    if not anchors_path.exists():
        raise FileNotFoundError(f"anchors.tsv not found in {bundle_dir}")
    return anchors_path

