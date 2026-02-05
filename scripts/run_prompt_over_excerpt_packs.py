#!/usr/bin/env python3
"""
Run a (non-IR) prompt over an excerpt pack built from anchor IDs.

This is intended for "simple prompt" companions that we run alongside the
validated pipelines (ContractIR, CovenantIR), using the SAME indexing_v2 anchor
seeds and the SAME excerpt-pack construction (full anchor spans + gap-fill/pad).

Artifacts (out-dir/<item_id>/):
  - excerpt_pack.txt          (exact [[A####]] blocks fed to the prompt)
  - prompt_rendered.txt       (exact prompt text sent to the gateway)
  - llm_output.txt            (raw model output)
  - error.txt                 (if gateway call failed)
  - meta.json                 (seed anchors + expansion settings)
  - prompt_used.txt           (copy of the prompt template)

This script does NOT attempt to validate or parse outputs; it is an artifact-
first runner for side-by-side review and prompt iteration.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

# Ensure repo root import works when invoked as `python scripts/...`.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.pipeline.anchors import load_anchor_catalog  # noqa: E402
from src.pipeline.config import REQUIRED_MODEL, REQUIRED_REASONING, Paths  # noqa: E402
from src.pipeline.excerpt_packs import build_excerpt_pack_from_canonical, expand_anchor_ids  # noqa: E402
from src.pipeline.indexing import DEFAULT_GATEWAY_URL, _ensure_gateway_client_sync  # noqa: E402
from src.pipeline.schemas_v2 import IndexingSelectionV2Artifact  # noqa: E402
from src.pipeline.utils import assert_exists, prompt_view_path  # noqa: E402


class PromptRunError(RuntimeError):
    pass


@dataclass(frozen=True)
class BucketSpec:
    name: str
    description: str


BUCKETS: dict[str, BucketSpec] = {
    "financial_covenant": BucketSpec(
        name="financial_covenant",
        description="Use indexing_v2.selection.financial_covenant_anchors",
    ),
    "pricing_only": BucketSpec(
        name="pricing_only",
        description="Use indexing_v2.selection.pricing_anchors",
    ),
    "pricing_union": BucketSpec(
        name="pricing_union",
        description="Union of pricing/base_rate/spread/fee anchor buckets",
    ),
    "base_rate": BucketSpec(
        name="base_rate",
        description="Use indexing_v2.selection.base_rate_anchors",
    ),
    "spread": BucketSpec(
        name="spread",
        description="Use indexing_v2.selection.spread_anchors",
    ),
    "fee": BucketSpec(
        name="fee",
        description="Use indexing_v2.selection.fee_anchors",
    ),
}


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8", errors="replace"))


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _split_csv(values: Iterable[str]) -> list[str]:
    out: list[str] = []
    for v in values:
        if not v:
            continue
        parts = [p.strip() for p in str(v).split(",")]
        out.extend([p for p in parts if p])
    return out


def _render_prompt(template: str, contexts: str) -> str:
    t = template.strip()
    if "{document}" in t:
        return t.replace("{document}", contexts)
    if "{snippets}" in t:
        return t.replace("{snippets}", contexts)
    if "=== INPUT ===" in t:
        return f"{t}\n{contexts}"
    return f"{t}\n\n=== INPUT ===\n{contexts}"


def _seed_anchors(selection: Any, bucket: str) -> list[str]:
    # selection is IndexingSelectionV2 (pydantic model) but type hints are loose here.
    if bucket == "financial_covenant":
        return list(selection.financial_covenant_anchors or [])
    if bucket == "pricing_only":
        return list(selection.pricing_anchors or [])
    if bucket == "pricing_union":
        merged: list[str] = []
        for arr in (
            selection.pricing_anchors or [],
            selection.base_rate_anchors or [],
            selection.spread_anchors or [],
            selection.fee_anchors or [],
        ):
            merged.extend(list(arr))
        # Preserve order but de-dupe.
        seen: set[str] = set()
        deduped: list[str] = []
        for a in merged:
            if a in seen:
                continue
            seen.add(a)
            deduped.append(a)
        return deduped
    if bucket == "base_rate":
        return list(selection.base_rate_anchors or [])
    if bucket == "spread":
        return list(selection.spread_anchors or [])
    if bucket == "fee":
        return list(selection.fee_anchors or [])
    raise PromptRunError(f"Unknown bucket: {bucket!r}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True, help="Existing run folder under runs/<run-id>/ with normalized/ + indexing_v2/.")
    ap.add_argument("--prompt", required=True, help="Prompt template path (text file).")
    ap.add_argument(
        "--bucket",
        required=True,
        choices=sorted(BUCKETS.keys()),
        help="Which indexing_v2 anchor bucket(s) to build the excerpt pack from.",
    )
    ap.add_argument("--item-id", action="append", default=[], help="Item ID(s) to run (repeatable; also supports comma-separated).")
    ap.add_argument("--out-dir", required=True, help="Output directory (will be created).")
    ap.add_argument("--fill-gaps-up-to", type=int, default=12)
    ap.add_argument("--neighbor-pad", type=int, default=2)
    ap.add_argument("--gateway-url", default=None)
    ap.add_argument("--model", default=REQUIRED_MODEL)
    ap.add_argument("--reasoning", default=REQUIRED_REASONING)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--attempts", type=int, default=3)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    run_id = str(args.run_id)
    run_dir = Path("runs") / run_id
    if not run_dir.exists():
        raise SystemExit(f"Run dir not found: {run_dir}")

    prompt_path = Path(args.prompt)
    assert_exists(prompt_path, message=f"Prompt not found: {prompt_path}")
    prompt_template = prompt_path.read_text(encoding="utf-8", errors="replace")

    item_ids = _split_csv(args.item_id)
    if not item_ids:
        raise SystemExit("Provide at least one --item-id (repeatable; comma-separated supported).")

    out_dir = Path(args.out_dir)
    if out_dir.exists():
        if not args.overwrite:
            raise SystemExit(f"out-dir already exists: {out_dir} (pass --overwrite to delete)")
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Copy prompt into out dir for reproducibility.
    _write_text(out_dir / prompt_path.name, prompt_template)

    complete = _ensure_gateway_client_sync()
    gateway_url = args.gateway_url or DEFAULT_GATEWAY_URL

    report_items: list[dict[str, Any]] = []

    paths = Paths(root=Path("."), run_id=run_id)

    for item_id in item_ids:
        item_out = out_dir / item_id
        item_out.mkdir(parents=True, exist_ok=True)

        started = time.time()
        meta: dict[str, Any] = {
            "run_id": run_id,
            "item_id": item_id,
            "bucket": args.bucket,
            "bucket_description": BUCKETS[args.bucket].description,
            "fill_gaps_up_to": int(args.fill_gaps_up_to),
            "neighbor_pad": int(args.neighbor_pad),
            "model": args.model,
            "reasoning": args.reasoning,
            "temperature": float(args.temperature),
            "gateway_url": gateway_url,
            "prompt_original_path": str(prompt_path),
        }

        try:
            catalog = load_anchor_catalog(paths, item_id)
            canonical_text = prompt_view_path(paths, item_id).read_text(encoding="utf-8", errors="replace")

            sel_path = assert_exists(
                run_dir / "indexing_v2" / f"{item_id}_anchors.json",
                message=f"Missing indexing_v2 anchors JSON for {item_id}: {run_dir}/indexing_v2/{item_id}_anchors.json",
            )
            sel_doc = _read_json(sel_path)
            sel_art = IndexingSelectionV2Artifact.model_validate(sel_doc)
            selection = sel_art.selection

            seeds = _seed_anchors(selection, args.bucket)
            if not seeds:
                raise PromptRunError(f"indexing_v2 returned empty seeds for bucket={args.bucket!r}")

            expanded = expand_anchor_ids(
                catalog=catalog,
                seed_anchor_ids=seeds,
                fill_gaps_up_to=int(args.fill_gaps_up_to),
                neighbor_pad=int(args.neighbor_pad),
            )
            if not expanded:
                raise PromptRunError("anchor expansion produced empty anchor list")

            contexts = build_excerpt_pack_from_canonical(
                canonical_text=canonical_text,
                catalog=catalog,
                anchor_ids=expanded,
            )
            rendered = _render_prompt(prompt_template, contexts)

            _write_text(item_out / "excerpt_pack.txt", contexts)
            _write_text(item_out / "prompt_rendered.txt", rendered)
            _write_json(
                item_out / "meta.json",
                {
                    **meta,
                    "seed_anchor_ids": seeds,
                    "expanded_anchor_ids": expanded,
                    "indexing_v2_path": str(sel_path),
                },
            )

            last_err: str | None = None
            output_text: str | None = None
            for attempt in range(1, max(1, int(args.attempts)) + 1):
                try:
                    output_text = complete(
                        model=args.model,
                        prompt=rendered,
                        base_url=gateway_url,
                        reasoning={"effort": args.reasoning} if args.reasoning else None,
                        temperature=float(args.temperature),
                        max_output_tokens=None,
                        timeout=600.0,
                    )
                    if isinstance(output_text, str) and output_text.strip():
                        break
                    last_err = "gateway returned empty text"
                except Exception as exc:  # pragma: no cover - network
                    last_err = f"{type(exc).__name__}: {exc}"
                    output_text = None
            if output_text is None or not output_text.strip():
                raise PromptRunError(last_err or "gateway failed")

            _write_text(item_out / "llm_output.txt", output_text)
            report_items.append(
                {
                    "item_id": item_id,
                    "status": "ok",
                    "secs": round(time.time() - started, 3),
                }
            )

        except Exception as exc:
            _write_text(item_out / "error.txt", f"{type(exc).__name__}: {exc}\n")
            report_items.append(
                {
                    "item_id": item_id,
                    "status": "error",
                    "error": f"{type(exc).__name__}: {exc}",
                    "secs": round(time.time() - started, 3),
                }
            )

    report = {
        "schema_version": "prompt_over_excerpt_pack_v1",
        "created_at": int(time.time()),
        "run_id": run_id,
        "prompt": str(prompt_path),
        "bucket": args.bucket,
        "bucket_description": BUCKETS[args.bucket].description,
        "fill_gaps_up_to": int(args.fill_gaps_up_to),
        "neighbor_pad": int(args.neighbor_pad),
        "model": args.model,
        "reasoning": args.reasoning,
        "temperature": float(args.temperature),
        "gateway_url": gateway_url,
        "items": report_items,
        "counts": {
            "ok": sum(1 for r in report_items if r.get("status") == "ok"),
            "error": sum(1 for r in report_items if r.get("status") == "error"),
        },
    }
    _write_json(out_dir / "report.json", report)

    ok = report["counts"]["ok"]
    err = report["counts"]["error"]
    print(f"[done] wrote {out_dir} (ok={ok} error={err})")
    if err:
        raise SystemExit(2)


if __name__ == "__main__":
    main()

