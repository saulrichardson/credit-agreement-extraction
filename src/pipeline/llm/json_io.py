from __future__ import annotations

import json
from typing import Any


def parse_json_response(raw_text: str, *, allowed_root_types: tuple[type, ...]) -> Any:
    """Parse JSON text and enforce an allowed root type contract.

    This function is intentionally pure and side-effect free so callers can reuse it
    in both retry and non-retry flows.
    """

    if not isinstance(raw_text, str) or not raw_text.strip():
        raise ValueError("empty response text")

    try:
        payload = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON: {exc}") from exc

    if allowed_root_types and not isinstance(payload, allowed_root_types):
        allowed = ", ".join(t.__name__ for t in allowed_root_types)
        raise ValueError(f"expected JSON root type {allowed}; got {type(payload).__name__}")

    return payload


async def ask_json_response(
    *,
    client: Any,
    prompt: str,
    model: str,
    temperature: float,
    reasoning: str | None,
    max_output_tokens: int | None,
) -> str:
    """Send one model request and return raw text.

    This function is request-only; it does not parse or validate returned text.
    """

    result = await client.complete_response(
        model=model,
        input_messages=[{"role": "user", "content": prompt}],
        reasoning={"effort": reasoning} if reasoning else None,
        temperature=temperature,
        max_output_tokens=max_output_tokens,
        metadata=None,
    )
    raw_text = result.get("text") if isinstance(result, dict) else str(result)
    return raw_text if isinstance(raw_text, str) else str(raw_text)
