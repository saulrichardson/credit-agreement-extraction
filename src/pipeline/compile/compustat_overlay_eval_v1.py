from __future__ import annotations

import hashlib
import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from pipeline.compile.compustat_formula import evaluate_compustat_formula
from pipeline.core.config import Paths, update_manifest
from pipeline.utils import assert_exists


@dataclass(frozen=True)
class _OverlayFormula:
    term: str
    formula: str | None
    variables: list[str]
    match_type: str | None
    confidence: str | None
    source_schema_version: str
    selected_candidate_idx: int | None


def _normalize_vars(values: dict[str, float]) -> dict[str, float]:
    return {str(k).strip().lower(): float(v) for k, v in values.items()}


def _load_company_values(path: Path) -> tuple[dict[str, str], dict[str, dict[str, float]]]:
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Compustat company values file is not valid JSON: {path}") from exc

    if not isinstance(doc, dict):
        raise RuntimeError(f"Compustat company values must be a JSON object: {path}")
    if doc.get("schema_version") != "compustat_company_values_v1":
        raise RuntimeError(
            "Compustat company values schema_version must be "
            f"'compustat_company_values_v1' (got {doc.get('schema_version')!r}): {path}"
        )

    rows = doc.get("companies")
    if not isinstance(rows, list) or not rows:
        raise RuntimeError(f"Compustat company values companies[] must be a non-empty list: {path}")

    item_to_company: dict[str, str] = {}
    values_by_company: dict[str, dict[str, float]] = {}

    for idx, row in enumerate(rows):
        if not isinstance(row, dict):
            raise RuntimeError(f"Compustat company values companies[{idx}] must be object: {path}")

        company_id = row.get("company_id")
        if not isinstance(company_id, str) or not company_id.strip():
            raise RuntimeError(f"Compustat company values companies[{idx}].company_id must be non-empty string: {path}")
        company_id = company_id.strip()
        if company_id in values_by_company:
            raise RuntimeError(f"Duplicate company_id in values file: {company_id!r}")

        item_ids = row.get("item_ids")
        if not isinstance(item_ids, list) or not item_ids or not all(isinstance(v, str) and v.strip() for v in item_ids):
            raise RuntimeError(
                f"Compustat company values companies[{idx}].item_ids must be non-empty list[str]: {path}"
            )

        values = row.get("values")
        if not isinstance(values, dict):
            raise RuntimeError(f"Compustat company values companies[{idx}].values must be object: {path}")
        if not all(isinstance(k, str) and k.strip() for k in values.keys()):
            raise RuntimeError(f"Compustat company values companies[{idx}].values keys must be non-empty strings: {path}")
        if not all(isinstance(v, (int, float)) for v in values.values()):
            raise RuntimeError(f"Compustat company values companies[{idx}].values must be numeric: {path}")

        normalized_values = _normalize_vars(values)
        values_by_company[company_id] = normalized_values

        for item_id in item_ids:
            key = item_id.strip()
            prior = item_to_company.get(key)
            if prior is not None:
                raise RuntimeError(
                    f"Item id {key!r} appears under multiple companies ({prior!r}, {company_id!r}) in {path}"
                )
            item_to_company[key] = company_id

    return item_to_company, values_by_company


def _extract_formula_payload(overlay: dict[str, Any]) -> _OverlayFormula:
    schema_version = overlay.get("schema_version")
    if schema_version == "compustat_overlay_candidates_v1":
        selected_idx = overlay.get("selected_candidate_idx")
        if not isinstance(selected_idx, int):
            raise RuntimeError("overlay.selected_candidate_idx must be int for compustat_overlay_candidates_v1")
        candidates = overlay.get("candidates")
        if not isinstance(candidates, list):
            raise RuntimeError("overlay.candidates must be list for compustat_overlay_candidates_v1")
        if selected_idx < 0 or selected_idx >= len(candidates):
            raise RuntimeError(
                "overlay.selected_candidate_idx out of bounds for compustat_overlay_candidates_v1"
            )
        selected = candidates[selected_idx]
        if not isinstance(selected, dict):
            raise RuntimeError("overlay.candidates[selected_candidate_idx] must be object")
        vars_ = selected.get("compustat_variables")
        if not isinstance(vars_, list) or not all(isinstance(v, str) for v in vars_):
            raise RuntimeError("overlay selected candidate compustat_variables must be list[str]")
        return _OverlayFormula(
            term=str(overlay.get("term") or ""),
            formula=selected.get("compustat_formula") if isinstance(selected.get("compustat_formula"), str) else None,
            variables=[v.lower() for v in vars_],
            match_type=selected.get("match_type") if isinstance(selected.get("match_type"), str) else None,
            confidence=selected.get("confidence") if isinstance(selected.get("confidence"), str) else None,
            source_schema_version=schema_version,
            selected_candidate_idx=selected_idx,
        )

    if schema_version != "compustat_overlay_v1":
        raise RuntimeError(f"Unsupported overlay schema_version: {schema_version!r}")

    vars_ = overlay.get("compustat_variables")
    if not isinstance(vars_, list) or not all(isinstance(v, str) for v in vars_):
        raise RuntimeError("overlay compustat_variables must be list[str]")

    return _OverlayFormula(
        term=str(overlay.get("term") or ""),
        formula=overlay.get("compustat_formula") if isinstance(overlay.get("compustat_formula"), str) else None,
        variables=[v.lower() for v in vars_],
        match_type=overlay.get("match_type") if isinstance(overlay.get("match_type"), str) else None,
        confidence=overlay.get("confidence") if isinstance(overlay.get("confidence"), str) else None,
        source_schema_version=schema_version,
        selected_candidate_idx=None,
    )


def _variables_from_formula(formula: str) -> list[str]:
    found = [m.group(0).lower() for m in re.finditer(r"\b[a-z][a-z0-9_]*\b", formula)]
    out: list[str] = []
    seen: set[str] = set()
    for var in found:
        if var not in seen:
            seen.add(var)
            out.append(var)
    return out


def _synthetic_value(*, item_id: str, variable: str) -> float:
    h = hashlib.sha256(f"{item_id}::{variable}".encode("utf-8")).hexdigest()
    # Stable, non-zero positive range: 10.0 to 910.0
    base = int(h[:8], 16) % 9000
    return float(base) / 10.0 + 10.0


def run_compustat_overlay_eval_v1(
    paths: Paths,
    item_ids: Iterable[str],
    *,
    overlay_output_subdir: str,
    output_subdir: str,
    values_path: Path | None = None,
    synthetic_values: bool = False,
) -> None:
    """Evaluate Compustat overlay formulas using per-company variable values."""

    if synthetic_values and values_path is not None:
        raise RuntimeError("Provide either synthetic_values=true or values_path, not both.")
    if not synthetic_values and values_path is None:
        raise RuntimeError("Provide one input source for values: synthetic_values=true or values_path.")
    if values_path is not None:
        assert_exists(values_path, message=f"Compustat company values file not found: {values_path}")

    out_dir = paths.run_dir / "compustat_overlay_eval_v1" / output_subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    for stale_name in ("errors.txt",):
        stale_path = out_dir / stale_name
        if stale_path.exists():
            stale_path.unlink()

    item_to_company: dict[str, str] = {}
    values_by_company: dict[str, dict[str, float]] = {}
    if values_path is not None:
        item_to_company, values_by_company = _load_company_values(values_path)

    item_errors: list[tuple[str, str]] = []
    summary_items: list[dict[str, Any]] = []
    totals: dict[str, int] = {
        "items_processed": 0,
        "terms_total": 0,
        "terms_evaluated": 0,
        "terms_no_formula": 0,
        "terms_missing_values": 0,
        "terms_eval_error": 0,
        "items_missing_overlay": 0,
        "items_parse_error": 0,
        "items_missing_company_values": 0,
    }

    for item_id in item_ids:
        overlay_path = (
            paths.run_dir
            / "compustat_overlay_v1"
            / overlay_output_subdir
            / f"{item_id}__compustat_overlay.json"
        )
        if not overlay_path.exists():
            item_errors.append((item_id, f"Overlay aggregate not found: {overlay_path}"))
            totals["items_missing_overlay"] += 1
            summary_items.append(
                {
                    "item_id": item_id,
                    "company_id": None,
                    "status": "missing_overlay",
                    "terms_total": 0,
                    "terms_evaluated": 0,
                    "terms_no_formula": 0,
                    "terms_missing_values": 0,
                    "terms_eval_error": 0,
                }
            )
            continue

        try:
            aggregate = json.loads(overlay_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            item_errors.append((item_id, f"Overlay aggregate is invalid JSON: {overlay_path}: {exc}"))
            totals["items_parse_error"] += 1
            summary_items.append(
                {
                    "item_id": item_id,
                    "company_id": None,
                    "status": "overlay_json_error",
                    "terms_total": 0,
                    "terms_evaluated": 0,
                    "terms_no_formula": 0,
                    "terms_missing_values": 0,
                    "terms_eval_error": 0,
                }
            )
            continue

        overlays = aggregate.get("overlays")
        if not isinstance(overlays, list):
            item_errors.append((item_id, f"Overlay aggregate overlays[] missing or invalid: {overlay_path}"))
            totals["items_parse_error"] += 1
            summary_items.append(
                {
                    "item_id": item_id,
                    "company_id": None,
                    "status": "overlay_schema_error",
                    "terms_total": 0,
                    "terms_evaluated": 0,
                    "terms_no_formula": 0,
                    "terms_missing_values": 0,
                    "terms_eval_error": 0,
                }
            )
            continue

        try:
            parsed = [
                _extract_formula_payload(o)
                for o in overlays
                if isinstance(o, dict)
            ]
        except Exception as exc:
            item_errors.append((item_id, f"Failed to parse overlay rows: {exc}"))
            totals["items_parse_error"] += 1
            summary_items.append(
                {
                    "item_id": item_id,
                    "company_id": None,
                    "status": "overlay_row_error",
                    "terms_total": 0,
                    "terms_evaluated": 0,
                    "terms_no_formula": 0,
                    "terms_missing_values": 0,
                    "terms_eval_error": 0,
                }
            )
            continue

        if synthetic_values:
            needed_vars: set[str] = set()
            for p in parsed:
                if isinstance(p.formula, str) and p.formula.strip():
                    if p.variables:
                        needed_vars.update(v.lower() for v in p.variables)
                    else:
                        needed_vars.update(_variables_from_formula(p.formula))
            company_id = f"synthetic::{item_id}"
            values = {v: _synthetic_value(item_id=item_id, variable=v) for v in sorted(needed_vars)}
        else:
            company_id = item_to_company.get(item_id)
            if company_id is None:
                item_errors.append((item_id, "No company mapping for item_id in values file"))
                totals["items_missing_company_values"] += 1
                summary_items.append(
                    {
                        "item_id": item_id,
                        "company_id": None,
                        "status": "missing_company_values",
                        "terms_total": 0,
                        "terms_evaluated": 0,
                        "terms_no_formula": 0,
                        "terms_missing_values": 0,
                        "terms_eval_error": 0,
                    }
                )
                continue
            values = values_by_company[company_id]

        term_results: list[dict[str, Any]] = []
        item_counts: dict[str, int] = {
            "terms_total": 0,
            "terms_evaluated": 0,
            "terms_no_formula": 0,
            "terms_missing_values": 0,
            "terms_eval_error": 0,
        }

        for p in parsed:
            formula = p.formula.strip() if isinstance(p.formula, str) else None
            required_vars = [v.lower() for v in p.variables if isinstance(v, str) and v.strip()]
            if formula and not required_vars:
                required_vars = _variables_from_formula(formula)

            status: str
            value: float | None = None
            error: str | None = None
            missing: list[str] = []

            if not formula:
                status = "no_formula"
            else:
                missing = [v for v in required_vars if v.lower() not in values]
                if missing:
                    status = "missing_values"
                else:
                    try:
                        value = float(evaluate_compustat_formula(formula, values))
                        status = "evaluated"
                    except Exception as exc:  # pragma: no cover
                        status = "eval_error"
                        error = str(exc)

            item_counts["terms_total"] += 1
            if status == "evaluated":
                item_counts["terms_evaluated"] += 1
            elif status == "no_formula":
                item_counts["terms_no_formula"] += 1
            elif status == "missing_values":
                item_counts["terms_missing_values"] += 1
            elif status == "eval_error":
                item_counts["terms_eval_error"] += 1

            term_results.append(
                {
                    "term": p.term,
                    "status": status,
                    "value": value,
                    "error": error,
                    "missing_variables": missing,
                    "compustat_formula": formula,
                    "compustat_variables": required_vars,
                    "match_type": p.match_type,
                    "confidence": p.confidence,
                    "overlay_schema_version": p.source_schema_version,
                    "selected_candidate_idx": p.selected_candidate_idx,
                }
            )

        item_doc = {
            "schema_version": "compustat_overlay_eval_v1",
            "item_id": item_id,
            "company_id": company_id,
            "synthetic_values": synthetic_values,
            "overlay_input_subdir": overlay_output_subdir,
            "values_source": str(values_path) if values_path is not None else "synthetic",
            "created_at": int(time.time()),
            "values": dict(sorted(values.items())),
            "results": term_results,
            "summary": item_counts,
        }
        (out_dir / f"{item_id}__compustat_overlay_eval.json").write_text(
            json.dumps(item_doc, indent=2) + "\n",
            encoding="utf-8",
        )

        totals["items_processed"] += 1
        for k in item_counts:
            totals[k] += item_counts[k]

        summary_items.append(
            {
                "item_id": item_id,
                "company_id": company_id,
                "status": "ok",
                **item_counts,
            }
        )

    summary_doc = {
        "schema_version": "compustat_overlay_eval_summary_v1",
        "created_at": int(time.time()),
        "run_id": paths.run_id,
        "overlay_input_subdir": overlay_output_subdir,
        "output_subdir": output_subdir,
        "synthetic_values": synthetic_values,
        "values_source": str(values_path) if values_path is not None else "synthetic",
        "totals": totals,
        "items": summary_items,
        "item_error_count": len(item_errors),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary_doc, indent=2) + "\n", encoding="utf-8")

    if item_errors:
        (out_dir / "errors.txt").write_text(
            "\n".join(f"{item}: {msg}" for item, msg in item_errors) + "\n",
            encoding="utf-8",
        )

    if paths.manifest_path.exists():
        update_manifest(
            paths.manifest_path,
            compustat_overlay_eval_v1_overlay_input_subdir=overlay_output_subdir,
            compustat_overlay_eval_v1_output_subdir=output_subdir,
            compustat_overlay_eval_v1_values_source=str(values_path) if values_path else "synthetic",
            compustat_overlay_eval_v1_created_at=int(time.time()),
        )
