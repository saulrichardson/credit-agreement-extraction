#!/usr/bin/env python
"""
Batch runner for the CovenantIR v0.1 one-pass harness.

Goal:
  - Pressure-test extraction behavior across real retrieval_v2 snippet packs
  - Produce a summary.json with per-item status ("ok" / "blocked" / "invalid")

This intentionally shells out to:
  scripts/covenant_ir_v0_1_one_pass_harness.py

…so we keep a single source of truth for prompt formatting + validation gates.
"""

from __future__ import annotations

import asyncio
import argparse
import json
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


def _iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if isinstance(obj, dict):
                yield obj


def _has_financial_covenant(snippets_path: Path) -> bool:
    for rec in _iter_jsonl(snippets_path):
        cats = rec.get("categories") or []
        if isinstance(cats, list) and "financial_covenant" in cats:
            return True
    return False


def _item_id_from_snippets_filename(path: Path) -> Optional[str]:
    name = path.name
    if not name.endswith("_snippets.jsonl"):
        return None
    return name[: -len("_snippets.jsonl")]


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")


@dataclass(frozen=True)
class ItemResult:
    item_id: str
    status: str
    ok: bool
    attempts_used: Optional[int]
    covenants_count: Optional[int]
    open_items_total: Optional[int]
    open_items_blocking: Optional[int]
    exit_code: int
    elapsed_s: float


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--base-dir", default=".")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--prompt", default="prompts/covenant_ir_financial_v0_1.txt")
    ap.add_argument("--attempts", type=int, default=3)
    ap.add_argument("--model", default=None)
    ap.add_argument("--reasoning", default=None)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--timeout-seconds", type=float, default=600.0)
    ap.add_argument("--max-items", type=int, default=10)
    ap.add_argument(
        "--concurrency",
        type=int,
        default=1,
        help="Number of parallel subprocesses to run (1 = sequential; higher is faster but increases gateway load).",
    )
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="List candidate items and exit without running LLM calls.")
    ap.add_argument(
        "--item-id",
        dest="item_ids",
        action="append",
        default=[],
        help="Optional explicit item_id to run (repeatable). If omitted, scans retrieval_v2/*.jsonl.",
    )
    args = ap.parse_args()

    base_dir = Path(args.base_dir)
    run_dir = base_dir / "runs" / args.run_id
    retrieval_dir = run_dir / "retrieval_v2"
    if not retrieval_dir.exists():
        raise SystemExit(f"Missing retrieval_v2 dir: {retrieval_dir}")

    candidates: List[str] = []
    if args.item_ids:
        candidates = [i for i in args.item_ids if isinstance(i, str) and i.strip()]
    else:
        for path in sorted(retrieval_dir.glob("*_snippets.jsonl")):
            item_id = _item_id_from_snippets_filename(path)
            if not item_id:
                continue
            if _has_financial_covenant(path):
                candidates.append(item_id)

    candidates = candidates[: max(0, int(args.max_items))]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.dry_run:
        _write_json(out_dir / "candidates.json", {"run_id": args.run_id, "candidates": candidates})
        print(f"[dry-run] candidates={len(candidates)} wrote={out_dir / 'candidates.json'}")
        return 0

    def _run_one_sync(*, idx: int, item_id: str) -> ItemResult:
        item_out = out_dir / item_id
        item_out.mkdir(parents=True, exist_ok=True)

        result_path = item_out / "result.json"
        if args.skip_existing and result_path.exists():
            try:
                existing = json.loads(result_path.read_text(encoding="utf-8"))
            except Exception:
                existing = {}
            status = str(existing.get("status") or "unknown")
            ok = bool(existing.get("ok"))
            print(f"[{idx}/{len(candidates)}] SKIP item={item_id} out={item_out}")
            return ItemResult(
                item_id=item_id,
                status=status,
                ok=ok,
                attempts_used=existing.get("attempts_used"),
                covenants_count=existing.get("covenants_count"),
                open_items_total=existing.get("open_items_total"),
                open_items_blocking=existing.get("open_items_blocking"),
                exit_code=-1,
                elapsed_s=0.0,
            )

        cmd = [
            sys.executable,
            str(base_dir / "scripts" / "covenant_ir_v0_1_one_pass_harness.py"),
            "--run-id",
            args.run_id,
            "--item-id",
            item_id,
            "--base-dir",
            str(base_dir),
            "--out-dir",
            str(item_out),
            "--prompt",
            str(base_dir / args.prompt),
            "--attempts",
            str(args.attempts),
            "--temperature",
            str(args.temperature),
            "--timeout-seconds",
            str(args.timeout_seconds),
        ]
        if args.model:
            cmd.extend(["--model", args.model])
        if args.reasoning:
            cmd.extend(["--reasoning", args.reasoning])

        t0 = time.time()
        proc = subprocess.run(cmd, cwd=str(base_dir), check=False)
        rc = int(proc.returncode)
        elapsed_s = time.time() - t0

        result_obj: Dict[str, Any] = {}
        if result_path.exists():
            try:
                result_obj = json.loads(result_path.read_text(encoding="utf-8"))
            except Exception:
                result_obj = {}

        status = str(result_obj.get("status") or ("invalid" if rc else "ok"))
        ok = bool(result_obj.get("ok") if "ok" in result_obj else (rc == 0))
        print(f"[{idx}/{len(candidates)}] item={item_id} status={status} rc={rc} elapsed_s={elapsed_s:.1f}")
        return ItemResult(
            item_id=item_id,
            status=status,
            ok=ok,
            attempts_used=result_obj.get("attempts_used"),
            covenants_count=result_obj.get("covenants_count"),
            open_items_total=result_obj.get("open_items_total"),
            open_items_blocking=result_obj.get("open_items_blocking"),
            exit_code=rc,
            elapsed_s=float(f"{elapsed_s:.3f}"),
        )

    async def _run_one_async(*, idx: int, item_id: str, sem: asyncio.Semaphore) -> ItemResult:
        async with sem:
            item_out = out_dir / item_id
            item_out.mkdir(parents=True, exist_ok=True)

            result_path = item_out / "result.json"
            if args.skip_existing and result_path.exists():
                try:
                    existing = json.loads(result_path.read_text(encoding="utf-8"))
                except Exception:
                    existing = {}
                status = str(existing.get("status") or "unknown")
                ok = bool(existing.get("ok"))
                print(f"[{idx}/{len(candidates)}] SKIP item={item_id} out={item_out}")
                return ItemResult(
                    item_id=item_id,
                    status=status,
                    ok=ok,
                    attempts_used=existing.get("attempts_used"),
                    covenants_count=existing.get("covenants_count"),
                    open_items_total=existing.get("open_items_total"),
                    open_items_blocking=existing.get("open_items_blocking"),
                    exit_code=-1,
                    elapsed_s=0.0,
                )

            cmd = [
                sys.executable,
                str(base_dir / "scripts" / "covenant_ir_v0_1_one_pass_harness.py"),
                "--run-id",
                args.run_id,
                "--item-id",
                item_id,
                "--base-dir",
                str(base_dir),
                "--out-dir",
                str(item_out),
                "--prompt",
                str(base_dir / args.prompt),
                "--attempts",
                str(args.attempts),
                "--temperature",
                str(args.temperature),
                "--timeout-seconds",
                str(args.timeout_seconds),
            ]
            if args.model:
                cmd.extend(["--model", args.model])
            if args.reasoning:
                cmd.extend(["--reasoning", args.reasoning])

            t0 = time.time()
            proc = await asyncio.create_subprocess_exec(*cmd, cwd=str(base_dir))
            rc = int(await proc.wait())
            elapsed_s = time.time() - t0

            result_obj: Dict[str, Any] = {}
            if result_path.exists():
                try:
                    result_obj = json.loads(result_path.read_text(encoding="utf-8"))
                except Exception:
                    result_obj = {}

            status = str(result_obj.get("status") or ("invalid" if rc else "ok"))
            ok = bool(result_obj.get("ok") if "ok" in result_obj else (rc == 0))
            print(f"[{idx}/{len(candidates)}] item={item_id} status={status} rc={rc} elapsed_s={elapsed_s:.1f}")
            return ItemResult(
                item_id=item_id,
                status=status,
                ok=ok,
                attempts_used=result_obj.get("attempts_used"),
                covenants_count=result_obj.get("covenants_count"),
                open_items_total=result_obj.get("open_items_total"),
                open_items_blocking=result_obj.get("open_items_blocking"),
                exit_code=rc,
                elapsed_s=float(f"{elapsed_s:.3f}"),
            )

    results: List[ItemResult] = []
    if int(args.concurrency) <= 1:
        for idx, item_id in enumerate(candidates, start=1):
            results.append(_run_one_sync(idx=idx, item_id=item_id))
    else:
        sem = asyncio.Semaphore(max(1, int(args.concurrency)))

        async def _runner() -> List[ItemResult]:
            tasks = [
                asyncio.create_task(_run_one_async(idx=idx, item_id=item_id, sem=sem))
                for idx, item_id in enumerate(candidates, start=1)
            ]
            return list(await asyncio.gather(*tasks))

        results = asyncio.run(_runner())

    summary = {
        "run_id": args.run_id,
        "prompt": args.prompt,
        "attempts": int(args.attempts),
        "model": args.model,
        "reasoning": args.reasoning,
        "temperature": float(args.temperature),
        "timeout_seconds": float(args.timeout_seconds),
        "max_items": int(args.max_items),
        "concurrency": int(args.concurrency),
        "results": [r.__dict__ for r in results],
    }
    _write_json(out_dir / "summary.json", summary)

    by_status: Dict[str, int] = {}
    for r in results:
        by_status[r.status] = by_status.get(r.status, 0) + 1
    print(f"Wrote {out_dir / 'summary.json'}")
    print(f"Counts: {by_status}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
