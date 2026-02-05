from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Callable, TypeVar


T = TypeVar("T")


RetryPromptFn = Callable[[str, int, str, str], str]
OnAttemptFn = Callable[[int, str], None]
ValidateFn = Callable[[Any], T]


@dataclass(frozen=True)
class StrictJsonFailure(RuntimeError):
    """Raised when a strict-JSON gateway call fails after all attempts."""

    attempts: int
    attempts_used: int
    last_error: str
    last_raw_text: str

    def __str__(self) -> str:
        return (
            "Strict JSON call failed "
            f"(attempts={self.attempts}, attempts_used={self.attempts_used}): {self.last_error}"
        )


def default_retry_prompt(base_prompt: str, attempt: int, error: str, previous_output: str) -> str:
    """Default repair prompt enforcing JSON-only output.

    Stages can supply a stronger retry prompt describing exact keys/schema; this is a generic fallback.
    """

    _ = previous_output  # stages may include this; default keeps prompts short.
    rules = (
        "Your previous response was invalid.\n"
        "- Output MUST be valid JSON.\n"
        "- Output MUST contain JSON ONLY (no markdown, no code fences, no prose).\n"
        "- Do not include any text before or after the JSON.\n"
    )
    return f"{base_prompt}\n\n=== RETRY REQUIRED (attempt {attempt}) ===\nError: {error}\n\n{rules}"


async def call_strict_json(
    *,
    client: Any,
    prompt: str,
    model: str,
    temperature: float,
    reasoning: str | None,
    attempts: int = 3,
    retry_prompt: RetryPromptFn | None = None,
    allowed_root_types: tuple[type, ...] = (dict,),
    validate: ValidateFn[T] | None = None,
    on_attempt: OnAttemptFn | None = None,
) -> tuple[T, str, int]:
    """Call the gateway and require JSON-only output, retrying on parse/validation errors.

    This helper standardizes the *policy* (retry loop, JSON-only requirement, root-type check).
    Domain validation stays in stage modules via the `validate` callback.

    Returns:
      (validated_value, raw_text, attempts_used)
    """

    attempts = max(1, int(attempts))
    retry_prompt = retry_prompt or default_retry_prompt
    validate = validate or (lambda payload: payload)  # type: ignore[assignment]

    last_error = "unknown"
    last_raw_text = ""
    previous_output = ""
    attempts_used = 0

    for attempt in range(1, attempts + 1):
        attempts_used = attempt
        attempt_prompt = prompt if attempt == 1 else retry_prompt(prompt, attempt, last_error, previous_output)

        try:
            result = await client.complete_response(
                model=model,
                input_messages=[{"role": "user", "content": attempt_prompt}],
                reasoning={"effort": reasoning} if reasoning else None,
                temperature=temperature,
                max_output_tokens=None,
                metadata=None,
            )
        except Exception as exc:  # pragma: no cover - network/errors
            last_error = f"gateway call failed: {type(exc).__name__}: {exc}"
            continue

        raw_text = result.get("text") if isinstance(result, dict) else str(result)
        raw_text = raw_text if isinstance(raw_text, str) else str(raw_text)
        last_raw_text = raw_text
        previous_output = raw_text

        if on_attempt is not None:
            on_attempt(attempt, raw_text)

        if not raw_text or not raw_text.strip():
            last_error = "empty response text"
            continue

        try:
            payload = json.loads(raw_text)
        except json.JSONDecodeError as exc:
            last_error = f"invalid JSON: {exc}"
            continue

        if allowed_root_types and not isinstance(payload, allowed_root_types):
            allowed = ", ".join(t.__name__ for t in allowed_root_types)
            last_error = f"expected JSON root type {allowed}; got {type(payload).__name__}"
            continue

        try:
            validated = validate(payload)
        except Exception as exc:
            last_error = str(exc) or f"{type(exc).__name__}: {exc}"
            continue

        return validated, raw_text, attempt

    raise StrictJsonFailure(
        attempts=attempts,
        attempts_used=attempts_used,
        last_error=last_error,
        last_raw_text=last_raw_text,
    )

