#!/usr/bin/env python3
"""Batch runner that submits agreement text to an OpenAI model with retries.

The script reads a system prompt (optional) plus a user prompt template,
injects the full agreement text, and sends the combined messages to gpt-5-mini
(or another model if specified). Responses are written as per-agreement ``.txt``
files. Requests execute in parallel with
retry/backoff to tolerate transient API failures.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable, Optional

import requests

API_URL = "https://api.openai.com/v1/responses"
DEFAULT_MODEL = "gpt-5-mini"
DEFAULT_USER_TEMPLATE = "<<BEGIN DOC>>\n{{document}}\n<<END DOC>>"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Submit agreement text to an OpenAI reasoning model."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Directory containing agreement .txt files to process.",
    )
    parser.add_argument(
        "--system-prompt-file",
        type=Path,
        help="Optional system prompt file whose content is sent as a separate "
        "system message ahead of each agreement.",
    )
    parser.add_argument(
        "--prompt-file",
        type=Path,
        help="User prompt template. Use '{{document}}' as a placeholder for the "
        "agreement body; if omitted, the document text is appended after the prompt. "
        "If not provided, a minimal default template wrapping the document between "
        "'<<BEGIN DOC>>' and '<<END DOC>>' is used.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory where response .txt files are saved.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Model name to invoke (default: {DEFAULT_MODEL}).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Maximum concurrent requests (default: 4).",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=5,
        help="Maximum attempts per file on failure (default: 5).",
    )
    parser.add_argument(
        "--initial-backoff",
        type=float,
        default=2.0,
        help="Initial backoff in seconds for retry (default: 2.0).",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="Explicit API key. Falls back to OPENAI_API_KEY env var.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=120.0,
        help="HTTP request timeout in seconds (default: 120).",
    )
    parser.add_argument(
        "--pattern",
        default="*.txt",
        help="Glob pattern for selecting agreements (default: *.txt).",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip files that already have an output file present.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List the files that would be processed without calling the API.",
    )
    return parser.parse_args()


def build_payload(
    *,
    system_prompt: Optional[str],
    user_template: str,
    document: str,
    model: str,
) -> dict:
    if "{{document}}" in user_template:
        user_message = user_template.replace("{{document}}", document)
    else:
        user_message = f"{user_template.strip()}\n\n{document}"

    messages = []
    if system_prompt:
        messages.append(
            {
                "role": "system",
                "content": [{"type": "input_text", "text": system_prompt}],
            }
        )
    messages.append(
        {
            "role": "user",
            "content": [{"type": "input_text", "text": user_message}],
        }
    )

    return {
        "model": model,
        "input": messages,
        "reasoning": {"effort": "medium"},
    }


def extract_text_response(payload: dict) -> Optional[str]:
    # Responses API delivers assistant text in different shapes depending
    # on the SDK; cover the current formats defensively.
    if "output_text" in payload and payload["output_text"]:
        return payload["output_text"]

    output = payload.get("output") or payload.get("data")
    if not output:
        return None

    for message in output:
        # Some SDKs wrap response items under "content" -> [{type, text}]
        for part in message.get("content", []):
            if part.get("type") in {"output_text", "text"} and part.get("text"):
                return part["text"]

        # Others expose a direct "text" field.
        text = message.get("text")
        if text:
            return text

    return None


def call_model(
    headers: dict,
    payload: dict,
    timeout: float,
    max_retries: int,
    initial_backoff: float,
) -> str:
    for attempt in range(1, max_retries + 1):
        try:
            response = requests.post(
                API_URL,
                headers=headers,
                data=json.dumps(payload),
                timeout=timeout,
            )
        except requests.RequestException as exc:
            error = f"request error: {exc}"
        else:
            if response.status_code == 200:
                data = response.json()
                text = extract_text_response(data)
                if text is None:
                    raise RuntimeError("Response missing assistant text content.")
                return text

            error = f"HTTP {response.status_code}: {response.text}"

        if attempt == max_retries:
            raise RuntimeError(f"Exceeded retries ({max_retries}): {error}")

        delay = initial_backoff * (2 ** (attempt - 1))
        delay += random.random()  # Add jitter to desynchronise retries.
        time.sleep(delay)

    raise RuntimeError("Unreachable retry loop exit.")


def submit_file(
    headers: dict,
    system_prompt: Optional[str],
    user_template: str,
    file_path: Path,
    output_path: Path,
    model: str,
    timeout: float,
    max_retries: int,
    initial_backoff: float,
) -> None:
    document_text = file_path.read_text(encoding="utf-8")
    payload = build_payload(
        system_prompt=system_prompt,
        user_template=user_template,
        document=document_text,
        model=model,
    )
    response_text = call_model(
        headers=headers,
        payload=payload,
        timeout=timeout,
        max_retries=max_retries,
        initial_backoff=initial_backoff,
    )
    output_path.write_text(response_text, encoding="utf-8")


def iter_files(directory: Path, pattern: str) -> Iterable[Path]:
    yield from sorted(directory.glob(pattern))


def main() -> None:
    args = parse_args()
    api_key = args.api_key or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("Error: API key not provided. Use --api-key or set OPENAI_API_KEY.")

    input_dir = args.input_dir
    if not input_dir.is_dir():
        raise SystemExit(f"Error: input directory not found: {input_dir}")

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.prompt_file:
        prompt_template = args.prompt_file.read_text(encoding="utf-8")
    else:
        prompt_template = DEFAULT_USER_TEMPLATE
    system_prompt = (
        args.system_prompt_file.read_text(encoding="utf-8")
        if args.system_prompt_file
        else None
    )

    files = list(iter_files(input_dir, args.pattern))
    if not files:
        print("No input files matched the provided pattern.", file=sys.stderr)
        return

    if args.skip_existing:
        filtered = []
        for file_path in files:
            out_path = output_dir / file_path.name
            if out_path.exists():
                continue
            filtered.append(file_path)
        files = filtered

    if not files:
        print("No files left to process after filtering.", file=sys.stderr)
        return

    if args.dry_run:
        for file_path in files:
            print(file_path)
        return

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    lock = threading.Lock()
    success_count = 0
    failure_count = 0

    def worker(file_path: Path) -> tuple[Path, Optional[str]]:
        output_path = output_dir / file_path.name
        try:
            submit_file(
                headers=headers,
                system_prompt=system_prompt,
                user_template=prompt_template,
                file_path=file_path,
                output_path=output_path,
                model=args.model,
                timeout=args.timeout,
                max_retries=args.max_retries,
                initial_backoff=args.initial_backoff,
            )
            return (file_path, None)
        except Exception as exc:  # noqa: BLE001
            return (file_path, str(exc))

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        future_map: dict[Future[tuple[Path, Optional[str]]], Path] = {}
        for path in files:
            future_map[executor.submit(worker, path)] = path

        for future in as_completed(future_map):
            file_path = future_map[future]
            file_path_str = str(file_path)
            try:
                _, error = future.result()
            except Exception as exc:  # noqa: BLE001
                error = str(exc)

            with lock:
                if error:
                    failure_count += 1
                    print(f"[FAIL] {file_path_str}: {error}", file=sys.stderr)
                else:
                    success_count += 1
                    print(f"[OK] {file_path_str}")

    print(f"Completed: {success_count} succeeded, {failure_count} failed.")
    if failure_count:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
