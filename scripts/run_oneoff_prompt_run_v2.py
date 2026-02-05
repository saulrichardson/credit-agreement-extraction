#!/usr/bin/env python3
"""
One-off runner: execute a prompt over a v2 snippet pack, producing a fully reproducible run folder.

Goal
  - Create runs/<dest_run_id>/ as a snapshot of:
      - manifest.json
      - normalized/ (canonical text + anchors)
      - indexing_v2/ (anchor selection buckets)
      - retrieval_v2/ (snippet windows around selected anchors)
  - Run a prompt over *scoped* subsets of retrieval_v2 snippets (by category), and store:
      - llm_inputs/<output_subdir>/<item_id>.snippets.txt   (exact snippet block fed to the prompt)
      - llm_inputs/<output_subdir>/<item_id>.prompt.txt     (exact rendered prompt sent to the model)
      - llm_qa/<output_subdir>/<item_id>.txt                (raw model output)
      - llm_qa/<output_subdir>/<item_id>.error.txt          (if the call failed)
      - report/oneoff_prompt_run.json                       (summary + provenance)

This is intentionally "artifact-first": you should be able to audit *exactly* what text was
fed to the model and what it returned, per item, without relying on external files.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import shutil
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Optional

# Ensure the repo root is on sys.path so imports like `src.pipeline.*` work even when this
# script is invoked as `python scripts/...`.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.pipeline.config import REQUIRED_MODEL, REQUIRED_REASONING, Paths  # noqa: E402
from src.pipeline.indexing import DEFAULT_GATEWAY_URL, _ensure_gateway_client_async  # noqa: E402
from src.pipeline.utils import load_manifest, manifest_items  # noqa: E402

# Reuse the snippet rendering + filtering behavior from structured_v2 so "one-off"
# runs match the existing structured-v2 execution semantics.
from src.pipeline import structured_v2 as sv2  # noqa: E402


@dataclass(frozen=True)
class ItemResult:
    item_id: str
    status: str  # ok | skipped_no_snippets | error
    n_snippets_total: int
    n_snippets_used: int
    error: Optional[str] = None


def _copy_tree(src: Path, dst: Path) -> None:
    if not src.exists():
        raise FileNotFoundError(f"Missing required source directory: {src}")
    if dst.exists():
        raise FileExistsError(f"Destination already exists: {dst}")
    shutil.copytree(src, dst)


def _copy_file(src: Path, dst: Path) -> None:
    if not src.exists():
        raise FileNotFoundError(f"Missing required source file: {src}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def _split_categories(values: Iterable[str]) -> list[str]:
    out: list[str] = []
    for v in values:
        if not v:
            continue
        # Allow both repeated --category and comma-separated convenience.
        parts = [p.strip() for p in str(v).split(",")]
        out.extend([p for p in parts if p])
    # Preserve order, but de-dupe.
    seen: set[str] = set()
    deduped: list[str] = []
    for c in out:
        key = c.lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(c)
    return deduped


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


async def _call_gateway_with_retry(
    *,
    client: Any,
    prompt: str,
    model: str,
    temperature: float,
    reasoning: str | None,
    attempts: int,
) -> str:
    # Mirror structured_v2 retry semantics (sleep backoff, fail on empty output).
    last_exc: Exception | None = None
    for attempt in range(1, max(1, attempts) + 1):
        try:
            reasoning_payload = {"effort": reasoning} if reasoning else None
            result = await client.complete_response(
                model=model,
                input_messages=[{"role": "user", "content": prompt}],
                reasoning=reasoning_payload,
                temperature=temperature,
                max_output_tokens=None,
                metadata=None,
            )
            text = result.get("text") if isinstance(result, dict) else str(result)
            if isinstance(text, str) and text.strip():
                return text
            raise RuntimeError("Gateway returned empty response text.")
        except Exception as exc:  # pragma: no cover - network/errors
            last_exc = exc
            await asyncio.sleep(1.5 * attempt)
    raise last_exc if last_exc else RuntimeError("Gateway call failed with unknown error.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source-run-id", required=True, help="Existing run_id containing normalized/indexing_v2/retrieval_v2.")
    ap.add_argument("--dest-run-id", required=True, help="New run_id to create under runs/<dest-run-id>/.")
    ap.add_argument("--prompt", required=True, help="Path to a prompt template (text file).")
    ap.add_argument(
        "--category",
        action="append",
        default=[],
        help="Category filter for snippets (repeatable; also accepts comma-separated lists).",
    )
    ap.add_argument(
        "--output-subdir",
        default=None,
        help="Output folder under runs/<dest>/llm_qa/ (defaults to prompt stem).",
    )
    ap.add_argument("--gateway-url", default=None, help="Gateway base URL; defaults to $GATEWAY_URL or 127.0.0.1:8000.")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--concurrency", type=int, default=3)
    ap.add_argument("--attempts", type=int, default=3)
    ap.add_argument(
        "--overwrite",
        action="store_true",
        help="Dangerous: delete an existing dest run folder before writing a new snapshot.",
    )
    args = ap.parse_args()

    source_run_id = args.source_run_id
    dest_run_id = args.dest_run_id
    prompt_path = Path(args.prompt)
    categories = _split_categories(args.category)
    if not categories:
        raise SystemExit("Provide at least one --category to make scoping explicit (pricing vs covenant, etc.).")

    source_run_dir = Path("runs") / source_run_id
    dest_run_dir = Path("runs") / dest_run_id

    if not source_run_dir.exists():
        raise SystemExit(f"Source run dir not found: {source_run_dir}")

    if dest_run_dir.exists():
        if not args.overwrite:
            raise SystemExit(
                f"Destination run dir already exists: {dest_run_dir}\n"
                "Pass --overwrite to delete and recreate it."
            )
        shutil.rmtree(dest_run_dir)

    # Snapshot core artifacts from source -> dest
    dest_run_dir.mkdir(parents=True, exist_ok=True)

    _copy_file(source_run_dir / "manifest.json", dest_run_dir / "manifest.json")
    _copy_tree(source_run_dir / "normalized", dest_run_dir / "normalized")
    _copy_tree(source_run_dir / "indexing_v2", dest_run_dir / "indexing_v2")
    _copy_tree(source_run_dir / "retrieval_v2", dest_run_dir / "retrieval_v2")

    prompt_used_path = dest_run_dir / "prompt_used.txt"
    _copy_file(prompt_path, prompt_used_path)
    prompt_template = prompt_used_path.read_text(encoding="utf-8")

    manifest = load_manifest(dest_run_dir / "manifest.json")
    items = manifest_items(manifest)
    item_ids = [item["item_id"] for item in items if isinstance(item.get("item_id"), str)]
    if not item_ids:
        raise SystemExit("Manifest has no items; cannot run.")

    # Output dirs
    out_subdir = args.output_subdir or prompt_path.stem
    llm_out_dir = dest_run_dir / "llm_qa" / out_subdir
    llm_in_dir = dest_run_dir / "llm_inputs" / out_subdir
    report_dir = dest_run_dir / "report"
    llm_out_dir.mkdir(parents=True, exist_ok=True)
    llm_in_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    dest_paths = Paths(root=Path("."), run_id=dest_run_id)

    gateway_url = args.gateway_url or os.getenv("GATEWAY_URL") or DEFAULT_GATEWAY_URL
    model = REQUIRED_MODEL
    reasoning = REQUIRED_REASONING

    sem = asyncio.Semaphore(max(1, int(args.concurrency)))
    results: list[ItemResult] = []

    GatewayAgentClient = _ensure_gateway_client_async()

    async def _process(item_id: str, client: Any) -> None:
        async with sem:
            try:
                snippets = sv2._load_snippets_v2(dest_paths, item_id)  # noqa: SLF001
                scoped = sv2._filter_snippets(snippets, categories)  # noqa: SLF001

                if not scoped:
                    # Keep an explicit artifact instead of silently skipping.
                    (llm_out_dir / f"{item_id}.skipped.txt").write_text(
                        f"skipped: no snippets matched categories={categories}\n",
                        encoding="utf-8",
                    )
                    results.append(
                        ItemResult(
                            item_id=item_id,
                            status="skipped_no_snippets",
                            n_snippets_total=len(snippets),
                            n_snippets_used=0,
                        )
                    )
                    return

                snippet_block = sv2._render_snippet_block(scoped)  # noqa: SLF001
                rendered_prompt = sv2._render_prompt(prompt_template, snippet_block)  # noqa: SLF001

                (llm_in_dir / f"{item_id}.snippets.txt").write_text(snippet_block, encoding="utf-8")
                (llm_in_dir / f"{item_id}.prompt.txt").write_text(rendered_prompt, encoding="utf-8")

                text = await _call_gateway_with_retry(
                    client=client,
                    prompt=rendered_prompt,
                    model=model,
                    temperature=float(args.temperature),
                    reasoning=reasoning,
                    attempts=int(args.attempts),
                )
                (llm_out_dir / f"{item_id}.txt").write_text(text, encoding="utf-8")
                results.append(
                    ItemResult(
                        item_id=item_id,
                        status="ok",
                        n_snippets_total=len(snippets),
                        n_snippets_used=len(scoped),
                    )
                )
            except Exception as exc:
                err_text = f"{type(exc).__name__}: {exc}"
                (llm_out_dir / f"{item_id}.error.txt").write_text(err_text + "\n", encoding="utf-8")
                results.append(
                    ItemResult(
                        item_id=item_id,
                        status="error",
                        n_snippets_total=0,
                        n_snippets_used=0,
                        error=err_text,
                    )
                )

    async def _runner() -> None:
        async with GatewayAgentClient(base_url=gateway_url, timeout=600.0) as client:
            await asyncio.gather(*(_process(item_id, client) for item_id in item_ids))

    asyncio.run(_runner())

    report = {
        "created_at_utc": _utc_now_iso(),
        "source_run_id": source_run_id,
        "dest_run_id": dest_run_id,
        "prompt_original_path": str(prompt_path),
        "prompt_used_path": str(prompt_used_path),
        "categories": categories,
        "gateway_url": gateway_url,
        "model": model,
        "reasoning": reasoning,
        "temperature": float(args.temperature),
        "attempts": int(args.attempts),
        "concurrency": int(args.concurrency),
        "items": [asdict(r) for r in results],
        "counts": {
            "ok": sum(1 for r in results if r.status == "ok"),
            "skipped_no_snippets": sum(1 for r in results if r.status == "skipped_no_snippets"),
            "error": sum(1 for r in results if r.status == "error"),
        },
    }
    (report_dir / "oneoff_prompt_run.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    # Make it obvious in CLI output where to look.
    print(f"[done] wrote runs/{dest_run_id}/ (prompt snapshot + llm_inputs + llm_qa)")
    print(f"[report] runs/{dest_run_id}/report/oneoff_prompt_run.json")
    print(f"[outputs] runs/{dest_run_id}/llm_qa/{out_subdir}/")

    # Exit non-zero if any item failed, but keep artifacts on disk.
    if report["counts"]["error"]:
        raise SystemExit(2)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        raise SystemExit(130)
