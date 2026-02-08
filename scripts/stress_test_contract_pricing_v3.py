from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

# Allow running as a script without installing the package.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.pipeline.run import resolve_run_config  # noqa: E402
from src.pipeline.utils import load_manifest, manifest_items  # noqa: E402


Mode = Literal["tool", "table_pack", "semantic_scan", "semantic_scan_any_anchor"]


@dataclass(frozen=True)
class ModeConfig:
    mode: Mode
    planner_prompt: Path
    scan_prompt: Path | None


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text())


def _safe_read_text(path: Path, limit: int = 10_000) -> str:
    try:
        return path.read_text()[:limit]
    except Exception:
        return ""


def _summarize_output(
    *,
    out_path: Path,
) -> dict[str, Any]:
    d = _load_json(out_path)
    pricing = d.get("pricing") or {}
    regimes = pricing.get("pricing_regimes") or []
    rate_options = pricing.get("rate_options") or []

    table_ids: list[str] = []
    n_grids = 0
    n_adjustments = 0
    source_refs: set[str] = set()

    for r in regimes:
        aw = (r.get("applies_when") or {}) if isinstance(r, dict) else {}
        source_refs.update(aw.get("source_refs") or [])

        grids = r.get("grids") or []
        n_grids += len(grids)
        for g in grids:
            if isinstance(g, dict) and g.get("table_anchor_id"):
                table_ids.append(str(g.get("table_anchor_id")))
            source_refs.update((g.get("source_refs") or []) if isinstance(g, dict) else [])

        for a in (r.get("adjustments") or []) if isinstance(r, dict) else []:
            n_adjustments += 1
            source_refs.update((a.get("source_refs") or []) if isinstance(a, dict) else [])
            a_aw = (a.get("applies_when") or {}) if isinstance(a, dict) else {}
            source_refs.update(a_aw.get("source_refs") or [])

    return {
        "n_rate_options": len(rate_options),
        "n_regimes": len(regimes),
        "n_grids": n_grids,
        "table_anchor_ids": sorted({t for t in table_ids if t}),
        "n_adjustments": n_adjustments,
        "n_unique_source_refs": len({r for r in source_refs if isinstance(r, str) and r}),
    }


def _summarize_plan(plan_path: Path) -> dict[str, Any]:
    d = _load_json(plan_path)
    selected = d.get("selected_tables") or []
    table_ids = [t.get("table_anchor_id") for t in selected if isinstance(t, dict) and t.get("table_anchor_id")]
    return {
        "n_selected_tables": len(selected),
        "selected_table_anchor_ids": table_ids,
    }


def _extract_adjustment_warnings(
    *,
    output_path: Path,
    doc_ir_path: Path,
) -> list[dict[str, Any]]:
    """Heuristic sanity checks to flag likely misapplied adjustments.

    This is for stress-test reporting only (NOT a production rule engine).
    """

    try:
        out = _load_json(output_path)
        doc_ir = _load_json(doc_ir_path)
    except Exception:
        return []

    anchors_by_id: dict[str, str] = {
        str(a.get("anchor_id")): str(a.get("text") or "")
        for a in (doc_ir.get("anchors") or [])
        if isinstance(a, dict) and a.get("anchor_id")
    }

    rate_option_label: dict[str, str] = {
        str(ro.get("rate_option_id")): str(ro.get("label_raw") or "")
        for ro in ((out.get("pricing") or {}).get("rate_options") or [])
        if isinstance(ro, dict) and ro.get("rate_option_id")
    }

    warnings: list[dict[str, Any]] = []
    regimes = ((out.get("pricing") or {}).get("pricing_regimes") or [])
    for r in regimes:
        if not isinstance(r, dict):
            continue
        for adj in (r.get("adjustments") or []):
            if not isinstance(adj, dict):
                continue
            srcs = [s for s in (adj.get("source_refs") or []) if isinstance(s, str)]
            applies_to = [rid for rid in (adj.get("applies_to_rate_option_ids") or []) if isinstance(rid, str)]
            labels = [rate_option_label.get(rid, "") for rid in applies_to]

            evidence_text = "\n".join(anchors_by_id.get(s, "") for s in srcs).lower()
            labels_text = " ".join(labels).lower()

            # Very lightweight mismatch flags (intentionally conservative).
            if ("l/c" in evidence_text or "letter of credit" in evidence_text) and (
                labels_text and ("l/c" not in labels_text and "letter of credit" not in labels_text)
            ):
                warnings.append(
                    {
                        "kind": "lc_mismatch",
                        "adjustment_id": adj.get("adjustment_id"),
                        "source_refs": srcs,
                        "applies_to_rate_option_ids": applies_to,
                        "applies_to_labels": labels,
                        "evidence_preview": evidence_text[:220],
                    }
                )
            if ("facility fee" in evidence_text) and (labels_text and "facility fee" not in labels_text):
                warnings.append(
                    {
                        "kind": "facility_fee_mismatch",
                        "adjustment_id": adj.get("adjustment_id"),
                        "source_refs": srcs,
                        "applies_to_rate_option_ids": applies_to,
                        "applies_to_labels": labels,
                        "evidence_preview": evidence_text[:220],
                    }
                )
            if ("applicable margin" in evidence_text) and (labels_text and "margin" not in labels_text):
                # Not perfect: loan labels won't contain "margin", but flags cases where evidence is explicitly
                # margin-only while applied to non-loan fee options.
                pass

    return warnings


def _run_mode(
    *,
    paths,
    item_ids: list[str],
    mode_cfg: ModeConfig,
    output_subdir: str,
    planner_model: str,
    planner_reasoning: str,
    planner_max_steps: int,
    planner_attempts_per_step: int,
    scan_model: str | None,
    scan_reasoning: str,
    scan_max_chunk_chars: int,
    scan_overlap_anchors: int,
    scan_max_anchors_per_chunk: int,
    scan_attempts_per_chunk: int,
    gateway_timeout: float,
    concurrency: int,
    attempts: int,
) -> tuple[float, str | None]:
    from src.pipeline.pricing.contract_pricing_v3 import run_contract_pricing_v3

    start = time.time()
    err: str | None = None
    try:
        run_contract_pricing_v3(
            paths,
            item_ids,
            planner_prompt_path=mode_cfg.planner_prompt,
            table_prompt_path=Path("prompts/contract_pricing_table_v2_noheuristics.txt"),
            planner_mode=mode_cfg.mode,
            planner_model=planner_model,
            planner_reasoning=planner_reasoning,
            planner_temperature=0.0,
            planner_max_steps=planner_max_steps,
            planner_attempts_per_step=planner_attempts_per_step,
            scan_prompt_path=mode_cfg.scan_prompt,
            scan_model=scan_model,
            scan_reasoning=scan_reasoning,
            scan_temperature=0.0,
            scan_max_chunk_chars=scan_max_chunk_chars,
            scan_overlap_anchors=scan_overlap_anchors,
            scan_max_anchors_per_chunk=scan_max_anchors_per_chunk,
            scan_attempts_per_chunk=scan_attempts_per_chunk,
            critic_prompt_path=None,
            critic_model=None,
            critic_reasoning="high",
            critic_temperature=0.0,
            compiler_temperature=0.0,
            gateway_timeout=gateway_timeout,
            concurrency=concurrency,
            attempts=attempts,
            output_subdir=output_subdir,
        )
    except KeyboardInterrupt:
        # Allow partial artifacts to be inspected; caller can decide whether to stop.
        err = "KeyboardInterrupt"
    except Exception as exc:
        # run_contract_pricing_v3 raises when any item failed; treat as a mode-level error but still evaluate.
        err = f"{type(exc).__name__}: {exc}".strip()
    elapsed = time.time() - start
    return elapsed, err


def main() -> None:
    parser = argparse.ArgumentParser(description="Stress-test contract-pricing-v3 planner modes.")
    parser.add_argument("--run-id", default="dan")
    parser.add_argument("--base-dir", default=".")
    parser.add_argument("--item-id", action="append", default=[], help="Repeatable; defaults to all manifest items.")
    parser.add_argument(
        "--mode",
        action="append",
        default=[],
        choices=["tool", "table_pack", "semantic_scan", "semantic_scan_any_anchor"],
        help="Repeatable; defaults to all modes.",
    )
    parser.add_argument("--planner-model", default="openai:gpt-5-mini")
    parser.add_argument("--planner-reasoning", default="high")
    parser.add_argument("--planner-max-steps", type=int, default=40)
    parser.add_argument("--planner-attempts-per-step", type=int, default=3)
    parser.add_argument("--scan-model", default=None)
    parser.add_argument("--scan-reasoning", default="high")
    parser.add_argument("--scan-max-chunk-chars", type=int, default=20_000)
    parser.add_argument("--scan-overlap-anchors", type=int, default=5)
    parser.add_argument("--scan-max-anchors-per-chunk", type=int, default=120)
    parser.add_argument("--scan-attempts-per-chunk", type=int, default=3)
    parser.add_argument("--gateway-timeout", type=float, default=1200.0)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--attempts", type=int, default=3)
    parser.add_argument("--report-name", default=None, help="Optional stable name; defaults to timestamp.")
    args = parser.parse_args()

    rc = resolve_run_config(args.run_id, args.base_dir, workers=4, bandwidth=4)
    paths = rc.paths()

    manifest = load_manifest(paths.manifest_path)
    items = manifest_items(manifest)
    all_item_ids = [it["item_id"] for it in items]
    if args.item_id:
        unknown = sorted(set(args.item_id) - set(all_item_ids))
        if unknown:
            raise SystemExit(f"Unknown --item-id values not present in manifest: {', '.join(unknown)}")
        item_ids = list(args.item_id)
    else:
        item_ids = all_item_ids

    modes: list[Mode] = [
        m for m in (args.mode or ["tool", "table_pack", "semantic_scan", "semantic_scan_any_anchor"]) if m
    ]  # type: ignore[list-item]

    now = int(time.time())
    report_name = args.report_name or f"{now}"

    deliver_dir = paths.deliverables_dir / "pricing_stress" / report_name
    deliver_dir.mkdir(parents=True, exist_ok=True)

    mode_cfgs: dict[Mode, ModeConfig] = {
        "tool": ModeConfig(
            mode="tool",
            planner_prompt=Path("prompts/contract_pricing_planner_v3_table_census.txt"),
            scan_prompt=None,
        ),
        "table_pack": ModeConfig(
            mode="table_pack",
            planner_prompt=Path("prompts/contract_pricing_planner_v3_table_pack.txt"),
            scan_prompt=None,
        ),
        "semantic_scan": ModeConfig(
            mode="semantic_scan",
            planner_prompt=Path("prompts/contract_pricing_planner_v4_semantic_scan.txt"),
            scan_prompt=Path("prompts/semantic_pricing_scan_chunk_v1.txt"),
        ),
        "semantic_scan_any_anchor": ModeConfig(
            mode="semantic_scan_any_anchor",
            planner_prompt=Path("prompts/contract_pricing_planner_v5_semantic_scan_any_anchor.txt"),
            scan_prompt=Path("prompts/semantic_pricing_scan_chunk_v2_any_anchor.txt"),
        ),
    }

    # Run each mode and then evaluate outputs.
    mode_results: dict[str, Any] = {
        "run_id": args.run_id,
        "item_ids": item_ids,
        "modes": modes,
        "created_at": now,
        "aborted": False,
        "modes_results": {},
    }

    for mode in modes:
        cfg = mode_cfgs[mode]
        output_subdir = f"stress_{report_name}_{mode}"

        elapsed, mode_err = _run_mode(
            paths=paths,
            item_ids=item_ids,
            mode_cfg=cfg,
            output_subdir=output_subdir,
            planner_model=args.planner_model,
            planner_reasoning=args.planner_reasoning,
            planner_max_steps=args.planner_max_steps,
            planner_attempts_per_step=args.planner_attempts_per_step,
            scan_model=args.scan_model,
            scan_reasoning=args.scan_reasoning,
            scan_max_chunk_chars=args.scan_max_chunk_chars,
            scan_overlap_anchors=args.scan_overlap_anchors,
            scan_max_anchors_per_chunk=args.scan_max_anchors_per_chunk,
            scan_attempts_per_chunk=args.scan_attempts_per_chunk,
            gateway_timeout=args.gateway_timeout,
            concurrency=args.concurrency,
            attempts=args.attempts,
        )

        out_root = paths.run_dir / "contract_pricing" / output_subdir
        plan_root = paths.run_dir / "contract_pricing_plan" / output_subdir
        doc_ir_root = paths.run_dir / "doc_ir"

        per_item: list[dict[str, Any]] = []
        for item_id in item_ids:
            out_path = out_root / f"{item_id}.json"
            err_path = out_root / f"{item_id}.error.txt"
            plan_path = plan_root / f"{item_id}.json"
            doc_ir_path = doc_ir_root / f"{item_id}.json"

            row: dict[str, Any] = {"item_id": item_id, "success": out_path.exists()}
            if out_path.exists():
                row.update(_summarize_output(out_path=out_path))
            else:
                row["error"] = (_safe_read_text(err_path, limit=600) or "").strip().splitlines()[:6]

            if plan_path.exists():
                row.update(_summarize_plan(plan_path))
            else:
                row["n_selected_tables"] = None
                row["selected_table_anchor_ids"] = None

            if out_path.exists() and doc_ir_path.exists():
                row["adjustment_warnings"] = _extract_adjustment_warnings(
                    output_path=out_path,
                    doc_ir_path=doc_ir_path,
                )
            else:
                row["adjustment_warnings"] = []

            per_item.append(row)

        mode_results["modes_results"][mode] = {
            "output_subdir": output_subdir,
            "elapsed_s": elapsed,
            "mode_level_error": mode_err,
            "per_item": per_item,
        }

        if mode_err == "KeyboardInterrupt":
            mode_results["aborted"] = True
            break

    # Write JSON + a compact markdown report for quick review.
    (deliver_dir / "results.json").write_text(json.dumps(mode_results, indent=2))

    completed_modes: list[Mode] = [m for m in modes if m in mode_results["modes_results"]]

    lines: list[str] = []
    lines.append(f"# contract-pricing-v3 stress test ({report_name})")
    lines.append("")
    lines.append(f"- run_id: `{args.run_id}`")
    lines.append(f"- items: `{len(item_ids)}`")
    lines.append(f"- modes_requested: `{', '.join(modes)}`")
    lines.append(f"- modes_completed: `{', '.join(completed_modes)}`")
    if mode_results.get("aborted"):
        lines.append("- aborted: `true`")
    lines.append("")

    for mode in completed_modes:
        mr = mode_results["modes_results"][mode]
        per_item = mr["per_item"]
        ok = sum(1 for r in per_item if r.get("success"))
        lines.append(f"## Mode: `{mode}`")
        lines.append(f"- output_subdir: `runs/{args.run_id}/contract_pricing/{mr['output_subdir']}`")
        lines.append(f"- elapsed_s: `{mr['elapsed_s']:.1f}`")
        if mr.get("mode_level_error"):
            lines.append(f"- mode_level_error: `{mr['mode_level_error']}`")
        lines.append(f"- success: `{ok}/{len(per_item)}`")

        # Aggregate rough metrics for successful items.
        succ = [r for r in per_item if r.get("success")]
        if succ:
            avg = lambda k: sum(float(r.get(k) or 0) for r in succ) / max(1, len(succ))
            lines.append(
                f"- avg_selected_tables: `{avg('n_selected_tables'):.2f}`; "
                f"avg_grids: `{avg('n_grids'):.2f}`; "
                f"avg_regimes: `{avg('n_regimes'):.2f}`; "
                f"avg_adjustments: `{avg('n_adjustments'):.2f}`"
            )

        # List failures (if any)
        failures = [r for r in per_item if not r.get("success")]
        if failures:
            lines.append("")
            lines.append("### Failures")
            for r in failures:
                lines.append(f"- `{r['item_id']}`: {json.dumps(r.get('error') or [])}")

        # Flag adjustment warnings
        warn_items = [
            (r["item_id"], r.get("adjustment_warnings") or [])
            for r in per_item
            if (r.get("adjustment_warnings") or [])
        ]
        if warn_items:
            lines.append("")
            lines.append("### Adjustment warnings (heuristic)")
            for item_id, warns in warn_items:
                lines.append(f"- `{item_id}`: `{len(warns)}` warnings")

        lines.append("")

    (deliver_dir / "report.md").write_text("\n".join(lines) + "\n")
    print(str(deliver_dir / "report.md"))


if __name__ == "__main__":
    main()
