#!/usr/bin/env python3
"""
Third-pass helper: extract computable formulas for covenant LIMITS (thresholds).

Inputs:
  - Covenant second-pass outputs from the less-structured prompt (prompt_v1_short.txt),
    typically produced by scripts/run_prompt_over_excerpt_packs.py
  - The corresponding excerpt_pack.txt (anchor blocks) used for the covenant extraction

Outputs (artifact-first, per item_id):
  - excerpt_pack.txt          (copied from the second-pass dir for provenance)
  - rows_json.txt             (the prior extraction rows we fed into this pass)
  - prompt_rendered.txt
  - llm_output.txt
  - meta.json
  - validation.json

This pass is intentionally narrow: it does NOT attempt to produce full CovenantIR.
It only produces threshold formulas for rows where limit is null + limit_text exists.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Iterable


class CovenantLimitFormulaThirdPassError(RuntimeError):
    pass


# Ensure repo root import works when invoked as `python scripts/...`.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.pipeline.core.config import REQUIRED_MODEL, REQUIRED_REASONING  # noqa: E402
from src.pipeline.evidence.indexing import DEFAULT_GATEWAY_URL, _ensure_gateway_client_sync  # noqa: E402


ANCHOR_RE = re.compile(r"\[\[(A\d{4})\]\]")


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


def _extract_anchor_ids(excerpt: str) -> set[str]:
    return set(ANCHOR_RE.findall(excerpt or ""))


def _load_second_pass_rows(cov_dir: Path, item_id: str) -> list[dict[str, Any]]:
    candidates = [
        cov_dir / item_id / "llm_output.txt",
        cov_dir / f"{item_id}.txt",
        cov_dir / item_id / f"{item_id}.txt",
    ]
    for p in candidates:
        if p.exists():
            doc = json.loads(_read_text(p))
            if not isinstance(doc, list):
                raise CovenantLimitFormulaThirdPassError(f"second-pass covenant output must be a JSON list: {p}")
            rows: list[dict[str, Any]] = []
            for el in doc:
                if isinstance(el, dict):
                    rows.append(el)
            return rows
    raise CovenantLimitFormulaThirdPassError(
        f"Missing covenant second-pass output for {item_id} under {cov_dir} (expected llm_output.txt or <item_id>.txt)"
    )


def _load_excerpt_pack(cov_dir: Path, item_id: str) -> str:
    candidates = [
        cov_dir / item_id / "excerpt_pack.txt",
        cov_dir / item_id / "contexts.txt",
    ]
    for p in candidates:
        if p.exists():
            return _read_text(p)
    raise CovenantLimitFormulaThirdPassError(f"Missing excerpt_pack.txt for {item_id} under {cov_dir}/{item_id}/")


def _looks_like_definition(text: str) -> bool:
    hay = (text or "").lower()
    if not hay.strip():
        return False
    # Definitions: '"TERM" means ...' or 'shall mean', 'is defined as'.
    return (" means" in hay and '"' in hay[:40]) or ("shall mean" in hay) or ("is defined as" in hay)


def _select_formula_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for r in rows:
        if not isinstance(r, dict):
            continue
        if r.get("limit") is not None:
            continue
        limit_text = r.get("limit_text")
        if not isinstance(limit_text, str) or not limit_text.strip():
            continue
        # Guardrail: if the prior pass accidentally extracted a *definition* as a covenant row, skip it here.
        if _looks_like_definition(limit_text):
            continue
        selected.append(r)
    return selected


def _render_prompt(template: str, *, item_id: str, excerpt: str, rows: list[dict[str, Any]]) -> str:
    if "{item_id}" not in template or "{excerpt}" not in template or "{rows_json}" not in template:
        raise CovenantLimitFormulaThirdPassError("Prompt template must contain {item_id}, {excerpt}, and {rows_json} placeholders")
    return (
        template.replace("{item_id}", item_id)
        .replace("{excerpt}", excerpt)
        .replace("{rows_json}", json.dumps(rows, indent=2, sort_keys=True))
        .strip()
    )


def _collect_expr_vars(node: Any) -> set[str]:
    out: set[str] = set()
    if isinstance(node, dict):
        if "var" in node and isinstance(node.get("var"), str):
            out.add(node["var"])
        if "args" in node and isinstance(node.get("args"), list):
            for a in node["args"]:
                out |= _collect_expr_vars(a)
    elif isinstance(node, list):
        for el in node:
            out |= _collect_expr_vars(el)
    return out


def _validate_limit_formula_output(*, doc: Any, anchors_present: set[str]) -> dict[str, Any]:
    out: dict[str, Any] = {"json_ok": False}
    if not isinstance(doc, dict):
        out["error"] = f"expected top-level object, got {type(doc).__name__}"
        return out
    if doc.get("schema_version") != "covenant_limit_formula_v1":
        out["error"] = f"unexpected schema_version: {doc.get('schema_version')!r}"
        return out
    limits = doc.get("limits")
    if not isinstance(limits, list):
        out["error"] = "missing/invalid limits list"
        return out
    out["json_ok"] = True
    out["limits_count"] = len(limits)

    unknown_anchors: set[str] = set()
    missing_expr: list[int] = []
    vars_not_declared: list[dict[str, Any]] = []

    for i, rec in enumerate(limits):
        if not isinstance(rec, dict):
            continue
        # Anchors
        for field in ("source_refs",):
            srefs = rec.get(field)
            if isinstance(srefs, list):
                for s in srefs:
                    if isinstance(s, str) and s.strip() and s.strip() not in anchors_present:
                        unknown_anchors.add(s.strip())

        le = rec.get("limit_expr")
        if not isinstance(le, dict):
            missing_expr.append(i)
            continue
        srefs = le.get("source_refs")
        if isinstance(srefs, list):
            for s in srefs:
                if isinstance(s, str) and s.strip() and s.strip() not in anchors_present:
                    unknown_anchors.add(s.strip())

        args = le.get("args")
        declared: set[str] = set()
        if isinstance(args, list):
            for a in args:
                if isinstance(a, dict) and isinstance(a.get("name"), str):
                    declared.add(a["name"])

        expr = le.get("expr")
        if expr is not None:
            used = _collect_expr_vars(expr)
            undeclared = sorted(used - declared)
            if undeclared:
                vars_not_declared.append({"index": i, "undeclared_vars": undeclared})

    out["unknown_source_refs"] = sorted(unknown_anchors)
    out["missing_limit_expr_idx"] = missing_expr
    out["vars_not_declared"] = vars_not_declared
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--covenant-second-pass-dir", required=True, help="Directory containing covenant second-pass outputs per item.")
    ap.add_argument("--prompt", default="prompts/covenant_limit_formula_third_pass_v1.txt")
    ap.add_argument("--all", action="store_true", help="Run for all item directories present under covenant-second-pass-dir.")
    ap.add_argument("--item-id", action="append", default=[], help="Item ID(s) to run (repeatable; supports comma-separated).")
    ap.add_argument("--out-dir", required=True, help="Output directory.")
    ap.add_argument("--model", default=REQUIRED_MODEL)
    ap.add_argument("--reasoning", default=REQUIRED_REASONING)
    ap.add_argument("--gateway-url", default=None)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--attempts", type=int, default=3)
    ap.add_argument("--timeout-seconds", type=float, default=600.0)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    cov_dir = Path(args.covenant_second_pass_dir)
    if not cov_dir.exists():
        raise SystemExit(f"covenant-second-pass-dir not found: {cov_dir}")

    if args.all and args.item_id:
        raise SystemExit("Use either --all or --item-id, not both.")

    if args.all:
        item_ids = sorted([p.name for p in cov_dir.iterdir() if p.is_dir()])
        if not item_ids:
            raise SystemExit(f"No items found under: {cov_dir}")
    else:
        item_ids = _split_csv(args.item_id)
        if not item_ids:
            raise SystemExit("Provide at least one --item-id (repeatable; comma-separated supported) or pass --all.")

    prompt_path = Path(args.prompt)
    if not prompt_path.exists():
        raise SystemExit(f"Prompt not found: {prompt_path}")
    prompt_template = _read_text(prompt_path)

    out_dir = Path(args.out_dir)
    if out_dir.exists():
        if not args.overwrite:
            raise SystemExit(f"out-dir already exists: {out_dir} (pass --overwrite to delete)")
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Copy prompt for reproducibility.
    _write_text(out_dir / prompt_path.name, prompt_template)

    complete = _ensure_gateway_client_sync()
    gateway_url = args.gateway_url or DEFAULT_GATEWAY_URL

    report: dict[str, Any] = {
        "schema_version": "covenant_limit_formula_third_pass_runner_v1",
        "created_at": int(time.time()),
        "covenant_second_pass_dir": str(cov_dir),
        "prompt": str(prompt_path),
        "llm": {
            "model": args.model,
            "reasoning_effort": args.reasoning,
            "temperature": float(args.temperature),
            "gateway_url": gateway_url,
            "timeout_seconds": float(args.timeout_seconds),
            "attempts": int(args.attempts),
        },
        "items": [],
    }

    for item_id in item_ids:
        started = time.time()
        item_out = out_dir / item_id
        item_out.mkdir(parents=True, exist_ok=True)

        all_rows = _load_second_pass_rows(cov_dir, item_id)
        excerpt = _load_excerpt_pack(cov_dir, item_id)
        anchors_present = _extract_anchor_ids(excerpt)

        rows = _select_formula_rows(all_rows)

        # For auditability, write the exact inputs we fed this pass.
        _write_text(item_out / "excerpt_pack.txt", excerpt)
        _write_text(item_out / "rows_json.txt", json.dumps(rows, indent=2, sort_keys=True) + "\n")

        _write_json(
            item_out / "selection_meta.json",
            {
                "rows_total": len(all_rows),
                "rows_selected_for_formula": len(rows),
                "skipped_definition_like_rows": sum(
                    1
                    for r in all_rows
                    if isinstance(r, dict)
                    and r.get("limit") is None
                    and isinstance(r.get("limit_text"), str)
                    and _looks_like_definition(r["limit_text"])
                ),
            },
        )

        # Skip the LLM call entirely when there's nothing to do.
        if not rows:
            empty = {
                "schema_version": "covenant_limit_formula_v1",
                "item_id": item_id,
                "limits": [],
                "unresolved_inputs": [],
            }
            _write_text(item_out / "llm_output.txt", json.dumps(empty, indent=2, sort_keys=True) + "\n")
            _write_json(item_out / "validation.json", {"json_ok": True, "limits_count": 0, "unknown_source_refs": []})
            report["items"].append(
                {
                    "item_id": item_id,
                    "status": "skipped_no_formula_rows",
                    "secs": round(time.time() - started, 3),
                }
            )
            continue

        rendered = _render_prompt(prompt_template, item_id=item_id, excerpt=excerpt, rows=rows)
        _write_text(item_out / "prompt_rendered.txt", rendered)
        _write_json(
            item_out / "meta.json",
            {
                "item_id": item_id,
                "anchors_present_count": len(anchors_present),
                "rows_count": len(rows),
                "model": args.model,
                "reasoning": args.reasoning,
                "temperature": float(args.temperature),
                "gateway_url": gateway_url,
            },
        )

        last_err: str | None = None
        output_text: str | None = None
        for _attempt in range(1, max(1, int(args.attempts)) + 1):
            try:
                output_text = complete(
                    model=args.model,
                    prompt=rendered,
                    base_url=gateway_url,
                    reasoning={"effort": args.reasoning} if args.reasoning else None,
                    temperature=float(args.temperature),
                    max_output_tokens=None,
                    timeout=float(args.timeout_seconds),
                )
                if isinstance(output_text, str) and output_text.strip():
                    break
                last_err = "gateway returned empty text"
                output_text = None
            except Exception as exc:  # pragma: no cover - network
                last_err = f"{type(exc).__name__}: {exc}"
                output_text = None

        if output_text is None or not output_text.strip():
            _write_text(item_out / "error.txt", f"{last_err or 'gateway_failed'}\n")
            report["items"].append(
                {
                    "item_id": item_id,
                    "status": "error",
                    "error": last_err or "gateway_failed",
                    "secs": round(time.time() - started, 3),
                }
            )
            continue

        _write_text(item_out / "llm_output.txt", output_text)

        try:
            parsed = json.loads(output_text)
        except Exception as exc:
            validation = {"json_ok": False, "json_error": f"{type(exc).__name__}: {exc}"}
            _write_json(item_out / "validation.json", validation)
            report["items"].append(
                {
                    "item_id": item_id,
                    "status": "invalid_json",
                    "secs": round(time.time() - started, 3),
                }
            )
            continue

        validation = _validate_limit_formula_output(doc=parsed, anchors_present=anchors_present)
        _write_json(item_out / "validation.json", validation)
        report["items"].append(
            {
                "item_id": item_id,
                "status": "ok" if validation.get("json_ok") else "invalid_shape",
                "validation": validation,
                "secs": round(time.time() - started, 3),
            }
        )

    _write_json(out_dir / "report.json", report)
    print(f"[done] wrote {out_dir} (items={len(item_ids)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
