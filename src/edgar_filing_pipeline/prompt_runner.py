"""Utility to run arbitrary prompts against OpenAI models."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path

try:  # pragma: no cover - optional dependency validated at runtime
    from openai import OpenAI
except ImportError as exc:  # pragma: no cover
    raise RuntimeError(
        "openai package is required for prompt execution. Install with `pip install openai`."
    ) from exc


PLACEHOLDER = "{{SOURCE_TEXT}}"


@dataclass
class PromptRunnerConfig:
    model: str = "gpt-5-nano"
    reasoning_effort: str = "medium"
    max_attempts: int = 3
    system_prompt: str = "You are a careful assistant. Follow the user instructions exactly."  # noqa: E501


class PromptRunner:
    def __init__(self, config: PromptRunnerConfig | None = None) -> None:
        if not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY must be set to run prompts.")
        self._client = OpenAI()
        self._config = config or PromptRunnerConfig()

    @staticmethod
    def render_prompt(template_path: Path, source_path: Path) -> str:
        template = template_path.read_text()
        source = source_path.read_text()
        if PLACEHOLDER not in template:
            raise ValueError(
                f"Prompt template {template_path} missing placeholder {PLACEHOLDER}."
            )
        return template.replace(PLACEHOLDER, source.strip())

    def run_prompt(self, prompt_text: str) -> str:
        last_error: Exception | None = None
        for _ in range(self._config.max_attempts):
            try:
                response = self._client.responses.create(
                    model=self._config.model,
                    reasoning={"effort": self._config.reasoning_effort},
                    input=[
                        {
                            "role": "system",
                            "content": self._config.system_prompt,
                        },
                        {"role": "user", "content": prompt_text},
                    ],
                )
                return response.output_text.strip()
            except Exception as exc:  # pragma: no cover - network/LLM failure
                last_error = exc
                time.sleep(1)
        raise RuntimeError(
            f"Prompt execution failed after {self._config.max_attempts} attempts: {last_error}"
        )
