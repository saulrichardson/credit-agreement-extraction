#!/usr/bin/env python3
"""
Targeted definitions-indexing pass (Option B):

Given:
  - a run with normalized/<item_id>/canonical_annotated.txt (full document with [[A####]] blocks)
  - pricing second-pass JSON outputs per item (less-structured prompt)

This pass asks an LLM (with full-document context) to locate the *definition anchors* for
each pricing metric found in the second pass.

Output (per item):
  runs/<run_id>/indexing_pricing_definitions_v1/<item_id>_pricing_definitions.json

This is intentionally separate from the definitions extraction pass: this stage selects anchors only.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Iterable


class PricingDefinitionsIndexingError(RuntimeError):
    pass


# Ensure repo root import works when invoked as `python scripts/...`.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.pipeline.core.anchors import load_anchor_catalog  # noqa: E402
from src.pipeline.core.config import REQUIRED_MODEL, REQUIRED_REASONING, Paths, prompt_hash, update_manifest  # noqa: E402
from src.pipeline.evidence.indexing import DEFAULT_GATEWAY_URL, _ensure_gateway_client_async  # noqa: E402
from src.pipeline.compile.indexing_pricing_definitions_v1_schemas import (  # noqa: E402
    PricingDefinitionsIndexingArtifactV1,
    PricingDefinitionsIndexingSelectionV1,
    PricingMetricInputV1,
)


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


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


def _canonical_annotated_text(paths: Paths, item_id: str) -> str:
    annotated = paths.normalized_dir / item_id / "canonical_annotated.txt"
    if not annotated.exists():
        raise FileNotFoundError(f"Annotated text missing for {item_id}: expected {annotated}")
    text = _read_text(annotated).strip()
    if not text:
        raise RuntimeError(f"Empty canonical_annotated.txt for {item_id}")
    return text


def _render_prompt(template: str, *, metrics_json: str, document: str) -> str:
    template = template.strip()
    missing: list[str] = []
    for key in ("{metrics_json}", "{document}"):
        if key not in template:
            missing.append(key)
    if missing:
        raise PricingDefinitionsIndexingError(
            f"Prompt template missing required placeholders: {', '.join(missing)}"
        )
    return template.replace("{metrics_json}", metrics_json).replace("{document}", document)


def _retry_prompt(base_prompt: str, *, attempt: int, error: str, metric_names: list[str], max_anchor_id: str) -> str:
    names = json.dumps(metric_names, indent=2)
    if attempt == 2:
        rules = (
            "Your previous response was not valid. Fix it.\n"
            "- Output MUST be a single JSON object (no markdown, no code fences, no prose).\n"
            "- JSON MUST match this exact schema:\n"
            "  {\"schema_version\":\"pricing_definitions_indexing_v1\",\"metrics\":[{\"name\":\"...\",\"contract_term\":\"...\",\"definition_anchor_ids\":[\"A0001\"],\"confidence\":\"high|medium|low\",\"notes\":[\"...\"]}]}\n"
            "- The `metrics` list MUST contain exactly one entry for EACH input metric name.\n"
            "- Each `definition_anchor_ids` MUST be a JSON array (use [] if not found; never null).\n"
            f"- Valid anchor IDs for this document range from A0001 to {max_anchor_id}.\n"
            f"- Metric names you MUST cover exactly: {names}\n"
        )
    else:
        rules = (
            "STRICT MODE (final attempt):\n"
            "- Return ONLY JSON with exactly these keys: schema_version, metrics\n"
            "- Do not include any other keys.\n"
            "- Do not include any text before/after the JSON.\n"
            "- `metrics` must contain exactly one entry per input metric name.\n"
            "- If you cannot find a definition, use definition_anchor_ids: [] and contract_term: \"\".\n"
            f"- Valid anchor IDs range from A0001 to {max_anchor_id}.\n"
        )
    return f"{base_prompt}\n\n=== RETRY REQUIRED ===\nError: {error}\n\n{rules}"


async def _call_gateway(*, client, prompt: str, model: str, temperature: float, reasoning: str | None) -> str:
    result = await client.complete_response(
        model=model,
        input_messages=[{"role": "user", "content": prompt}],
        reasoning={"effort": reasoning} if reasoning else None,
        temperature=temperature,
        max_output_tokens=None,
        metadata=None,
    )
    if isinstance(result, dict):
        return result.get("text") or ""
    return str(result)


def _load_pricing_second_pass_doc(pricing_dir: Path, item_id: str) -> tuple[Path, dict[str, Any]]:
    candidates = [
        pricing_dir / item_id / "llm_output.txt",
        pricing_dir / f"{item_id}.txt",
        pricing_dir / item_id / f"{item_id}.txt",
    ]
    for p in candidates:
        if p.exists():
            doc = json.loads(_read_text(p))
            if not isinstance(doc, dict):
                raise PricingDefinitionsIndexingError(f"pricing second-pass output must be a JSON object: {p}")
            return (p, doc)
    raise PricingDefinitionsIndexingError(
        f"Missing pricing second-pass output for {item_id} under {pricing_dir} (expected llm_output.txt or <item_id>.txt)"
    )


def _extract_metric_names(pricing_doc: dict[str, Any]) -> list[str]:
    """Extract metric names robustly from pricing output (covers tier conditions, not only metrics[])."""

    ordered: list[str] = []
    seen: set[str] = set()

    def _add(name: Any) -> None:
        if not isinstance(name, str):
            return
        n = name.strip()
        if not n or n in seen:
            return
        seen.add(n)
        ordered.append(n)

    metrics = pricing_doc.get("metrics") or []
    if isinstance(metrics, list):
        for m in metrics:
            if isinstance(m, dict):
                _add(m.get("name"))

    tier_schemes = pricing_doc.get("tier_schemes") or []
    if isinstance(tier_schemes, list):
        for scheme in tier_schemes:
            if not isinstance(scheme, dict):
                continue
            _add(scheme.get("metric"))
            tiers = scheme.get("tiers") or []
            if not isinstance(tiers, list):
                continue
            for tier in tiers:
                if not isinstance(tier, dict):
                    continue
                cond = tier.get("conditions")
                if isinstance(cond, dict):
                    for key in ("all", "any"):
                        arr = cond.get(key)
                        if not isinstance(arr, list):
                            continue
                        for clause in arr:
                            if isinstance(clause, dict):
                                _add(clause.get("metric"))

    facilities = pricing_doc.get("facilities") or []
    if isinstance(facilities, list):
        for fac in facilities:
            if not isinstance(fac, dict):
                continue
            rates = fac.get("rates") or []
            if not isinstance(rates, list):
                continue
            for rate in rates:
                if not isinstance(rate, dict):
                    continue
                cond = rate.get("condition")
                if isinstance(cond, dict):
                    _add(cond.get("metric"))

    # Drop obvious non-metrics / placeholders.
    return [m for m in ordered if m and m not in {"NA", "RatingScoreMetric", "RatingCategoryMetric"}]


def _extract_metric_inputs(pricing_doc: dict[str, Any]) -> list[PricingMetricInputV1]:
    metric_names = _extract_metric_names(pricing_doc)

    desc_by_name: dict[str, str] = {}
    seen: set[str] = set()
    for m in pricing_doc.get("metrics") or []:
        if not isinstance(m, dict):
            continue
        name = m.get("name")
        if not isinstance(name, str) or not name.strip():
            continue
        n = name.strip()
        if n in seen:
            continue
        seen.add(n)
        desc = m.get("description")
        if isinstance(desc, str) and desc.strip():
            desc_by_name[n] = desc.strip()

    metrics_in: list[PricingMetricInputV1] = []
    for name in metric_names:
        metrics_in.append(PricingMetricInputV1(name=name, description=desc_by_name.get(name)))
    return metrics_in


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True, help="Existing run folder under runs/<run-id>/ with normalized/ + indexing_v2/.")
    ap.add_argument("--pricing-second-pass-dir", required=True, help="Directory containing pricing second-pass outputs per item.")
    ap.add_argument("--prompt", default="prompts/indexing_pricing_definitions_v1.txt")
    ap.add_argument("--out-dir", default=None, help="Output directory. Default: runs/<run_id>/indexing_pricing_definitions_v1")
    ap.add_argument("--all", action="store_true", help="Run for all item_ids present under runs/<run-id>/normalized/.")
    ap.add_argument("--item-id", action="append", default=[], help="Item ID(s) to run (repeatable; supports comma-separated).")
    ap.add_argument("--skip-existing", action="store_true", help="Skip items with existing output JSON.")
    ap.add_argument("--gateway-url", default=None)
    ap.add_argument("--model", default=REQUIRED_MODEL)
    ap.add_argument("--reasoning", default=REQUIRED_REASONING)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--attempts", type=int, default=3)
    ap.add_argument("--timeout-seconds", type=float, default=600.0)
    ap.add_argument("--concurrency", type=int, default=3)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    run_id = str(args.run_id)
    paths = Paths(root=Path("."), run_id=run_id)
    if not paths.run_dir.exists():
        raise SystemExit(f"Run dir not found: {paths.run_dir}")

    if args.all and args.item_id:
        raise SystemExit("Use either --all or --item-id, not both.")

    if args.all:
        normalized_dir = paths.run_dir / "normalized"
        if not normalized_dir.exists():
            raise SystemExit(f"normalized dir not found: {normalized_dir}")
        item_ids = sorted([p.name for p in normalized_dir.iterdir() if p.is_dir()])
        if not item_ids:
            raise SystemExit(f"No items found under: {normalized_dir}")
    else:
        item_ids = _split_csv(args.item_id)
        if not item_ids:
            raise SystemExit("Provide at least one --item-id (repeatable; comma-separated supported) or pass --all.")

    pricing_second_pass_dir = Path(args.pricing_second_pass_dir)
    if not pricing_second_pass_dir.exists():
        raise SystemExit(f"pricing-second-pass-dir not found: {pricing_second_pass_dir}")

    prompt_path = Path(args.prompt)
    if not prompt_path.exists():
        raise SystemExit(f"Prompt not found: {prompt_path}")
    prompt_template = _read_text(prompt_path)
    prompt_digest = prompt_hash(prompt_path)

    out_dir = Path(args.out_dir) if args.out_dir else (paths.run_dir / "indexing_pricing_definitions_v1")
    if out_dir.exists():
        if not args.overwrite:
            raise SystemExit(f"out-dir already exists: {out_dir} (pass --overwrite to delete)")
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_text(out_dir / prompt_path.name, prompt_template)

    GatewayAgentClient = _ensure_gateway_client_async()
    resolved_gateway_url = args.gateway_url or DEFAULT_GATEWAY_URL
    attempts = max(1, int(args.attempts))
    concurrency = max(1, int(args.concurrency))
    temperature = float(args.temperature)
    timeout = float(args.timeout_seconds)

    report: dict[str, Any] = {
        "schema_version": "pricing_definitions_indexing_v1_runner_v1",
        "created_at": int(time.time()),
        "run_id": run_id,
        "prompt": str(prompt_path),
        "prompt_sha256": prompt_digest,
        "pricing_second_pass_dir": str(pricing_second_pass_dir),
        "out_dir": str(out_dir),
        "llm": {
            "gateway_url": resolved_gateway_url,
            "model": args.model,
            "reasoning_effort": args.reasoning,
            "temperature": temperature,
            "timeout_seconds": timeout,
            "attempts": attempts,
            "concurrency": concurrency,
        },
        "counts": {"ok": 0, "error": 0, "skipped_no_metrics": 0, "skipped_existing": 0},
        "items": [],
    }

    async def _run_async() -> None:
        sem = asyncio.Semaphore(concurrency)
        failures: list[tuple[str, str]] = []

        async with GatewayAgentClient(base_url=resolved_gateway_url, timeout=timeout) as client:

            async def _process(item_id: str) -> None:
                started = time.time()
                async with sem:
                    out_path = out_dir / f"{item_id}_pricing_definitions.json"
                    if args.skip_existing and out_path.exists():
                        report["counts"]["skipped_existing"] += 1
                        report["items"].append({"item_id": item_id, "status": "skipped_existing", "secs": 0.0})
                        return

                    pricing_path, pricing_doc = _load_pricing_second_pass_doc(pricing_second_pass_dir, item_id)
                    metrics_in = _extract_metric_inputs(pricing_doc)
                    metric_names = [m.name for m in metrics_in]

                    if not metric_names:
                        # No metrics discovered => nothing to index. Write an empty, schema-valid selection.
                        selection = PricingDefinitionsIndexingSelectionV1(
                            schema_version="pricing_definitions_indexing_v1",
                            metrics=[],
                        )
                        artifact = PricingDefinitionsIndexingArtifactV1(
                            schema_version="pricing_definitions_indexing_v1_artifact",
                            stage="indexing_pricing_definitions_v1",
                            run_id=run_id,
                            item_id=item_id,
                            created_at=int(time.time()),
                            gateway_url=resolved_gateway_url,
                            model=args.model,
                            reasoning_effort=args.reasoning,
                            temperature=temperature,
                            prompt=str(prompt_path),
                            prompt_sha256=prompt_digest,
                            attempts_used=0,
                            pricing_second_pass_path=str(pricing_path),
                            metrics_in=metrics_in,
                            selection=selection,
                        )
                        _write_text(out_path, artifact.model_dump_json(indent=2) + "\n")
                        report["counts"]["skipped_no_metrics"] += 1
                        report["items"].append(
                            {
                                "item_id": item_id,
                                "status": "skipped_no_metrics",
                                "metrics_count": 0,
                                "secs": round(time.time() - started, 3),
                            }
                        )
                        return

                    catalog = load_anchor_catalog(paths, item_id)
                    max_anchor_id = max(catalog.keys(), key=lambda aid: int(aid[1:]))

                    doc_block = _canonical_annotated_text(paths, item_id)
                    metrics_json = json.dumps([m.model_dump() for m in metrics_in], indent=2, sort_keys=True)
                    base_prompt = _render_prompt(prompt_template, metrics_json=metrics_json, document=doc_block)

                    last_error = "unknown"
                    for attempt in range(1, attempts + 1):
                        prompt = base_prompt
                        if attempt > 1:
                            prompt = _retry_prompt(
                                base_prompt,
                                attempt=attempt,
                                error=last_error,
                                metric_names=metric_names,
                                max_anchor_id=max_anchor_id,
                            )
                        try:
                            raw_text = await _call_gateway(
                                client=client,
                                prompt=prompt,
                                model=args.model,
                                temperature=temperature,
                                reasoning=args.reasoning,
                            )
                        except Exception as exc:
                            last_error = f"gateway call failed: {exc}"
                            continue

                        try:
                            payload = json.loads(raw_text)
                        except json.JSONDecodeError as exc:
                            last_error = f"invalid JSON: {exc}"
                            continue

                        try:
                            selection = PricingDefinitionsIndexingSelectionV1.model_validate(payload)
                        except Exception as exc:
                            last_error = f"schema validation failed: {exc}"
                            continue

                        # Enforce: exactly one output per input metric name.
                        out_names = [m.name for m in selection.metrics]
                        if len(out_names) != len(set(out_names)):
                            last_error = "duplicate metric names in output"
                            continue

                        missing = [n for n in metric_names if n not in out_names]
                        extra = [n for n in out_names if n not in metric_names]
                        if missing or extra:
                            last_error = f"metric coverage mismatch (missing={missing[:6]} extra={extra[:6]})"
                            continue

                        invalid: list[str] = []
                        for rec in selection.metrics:
                            for aid in rec.definition_anchor_ids:
                                if aid not in catalog:
                                    invalid.append(aid)
                        if invalid:
                            examples = ", ".join(sorted(set(invalid))[:10])
                            last_error = (
                                "returned anchor_id values not present in anchors.tsv "
                                f"(count={len(invalid)}; examples={examples})"
                            )
                            continue

                        artifact = PricingDefinitionsIndexingArtifactV1(
                            schema_version="pricing_definitions_indexing_v1_artifact",
                            stage="indexing_pricing_definitions_v1",
                            run_id=run_id,
                            item_id=item_id,
                            created_at=int(time.time()),
                            gateway_url=resolved_gateway_url,
                            model=args.model,
                            reasoning_effort=args.reasoning,
                            temperature=temperature,
                            prompt=str(prompt_path),
                            prompt_sha256=prompt_digest,
                            attempts_used=attempt,
                            pricing_second_pass_path=str(pricing_path),
                            metrics_in=metrics_in,
                            selection=selection,
                        )
                        _write_text(out_path, artifact.model_dump_json(indent=2) + "\n")
                        report["counts"]["ok"] += 1
                        report["items"].append(
                            {
                                "item_id": item_id,
                                "status": "ok",
                                "metrics_count": len(metric_names),
                                "anchors_total": sum(len(m.definition_anchor_ids) for m in selection.metrics),
                                "secs": round(time.time() - started, 3),
                            }
                        )
                        return

                    failures.append((item_id, last_error))
                    report["counts"]["error"] += 1
                    report["items"].append(
                        {
                            "item_id": item_id,
                            "status": "error",
                            "error": last_error,
                            "secs": round(time.time() - started, 3),
                        }
                    )

            await asyncio.gather(*(_process(i) for i in item_ids))

        if failures:
            joined = "; ".join(f"{item_id}: {err}" for item_id, err in failures[:5])
            raise RuntimeError(f"pricing definitions indexing failed for {len(failures)} items: {joined}")

    asyncio.run(_run_async())

    _write_json(out_dir / "report.json", report)

    # Record prompt provenance in the manifest if present (best-effort; do not create manifests here).
    if paths.manifest_path.exists():
        update_manifest(
            paths.manifest_path,
            indexing_pricing_definitions_v1_prompt=str(prompt_path),
            indexing_pricing_definitions_v1_prompt_sha256=prompt_digest,
        )

    print(f"[done] wrote {out_dir} (items={len(item_ids)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
