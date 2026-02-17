from __future__ import annotations

import json
from pathlib import Path

from pipeline.compile.compustat_overlay_eval_v1 import run_compustat_overlay_eval_v1
from pipeline.core.config import Paths


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _make_paths(tmp_path: Path, *, run_id: str = "run-test") -> Paths:
    run_dir = tmp_path / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    _write_json(
        run_dir / "manifest.json",
        {
            "run_id": run_id,
            "items": [{"item_id": "item-a"}],
        },
    )
    return Paths(root=tmp_path, run_id=run_id)


def _base_overlay_row(*, term: str, formula: str | None, variables: list[str]) -> dict:
    return {
        "schema_version": "compustat_overlay_v1",
        "term": term,
        "compustat_formula": formula,
        "compustat_variables": variables,
        "match_type": "approximate" if formula else "none",
        "confidence": "medium" if formula else "low",
        "limitations": [] if formula else ["No direct mapping"],
        "notes": ["assumption: test mapping"] if formula else ["next_step: manual mapping"],
    }


def test_compustat_overlay_eval_v1_synthetic_values(tmp_path: Path) -> None:
    paths = _make_paths(tmp_path)
    item_id = "item-a"
    overlay_subdir = "overlay-subdir"

    _write_json(
        paths.run_dir / "compustat_overlay_v1" / overlay_subdir / f"{item_id}__compustat_overlay.json",
        {
            "schema_version": "compustat_overlay_v1",
            "item_id": item_id,
            "overlays": [
                _base_overlay_row(
                    term="Debt/Cash Flow Ratio",
                    formula="dlttq / oancfy",
                    variables=["dlttq", "oancfy"],
                ),
                _base_overlay_row(term="S&P Rating", formula=None, variables=[]),
            ],
        },
    )

    run_compustat_overlay_eval_v1(
        paths,
        [item_id],
        overlay_output_subdir=overlay_subdir,
        output_subdir="eval-out",
        synthetic_values=True,
    )

    out_path = paths.run_dir / "compustat_overlay_eval_v1" / "eval-out" / f"{item_id}__compustat_overlay_eval.json"
    out = json.loads(out_path.read_text(encoding="utf-8"))
    assert out["synthetic_values"] is True
    statuses = [row["status"] for row in out["results"]]
    assert statuses == ["evaluated", "no_formula"]
    assert out["summary"]["terms_evaluated"] == 1
    assert out["summary"]["terms_no_formula"] == 1


def test_compustat_overlay_eval_v1_missing_values(tmp_path: Path) -> None:
    paths = _make_paths(tmp_path)
    item_id = "item-a"
    overlay_subdir = "overlay-subdir"

    _write_json(
        paths.run_dir / "compustat_overlay_v1" / overlay_subdir / f"{item_id}__compustat_overlay.json",
        {
            "schema_version": "compustat_overlay_v1",
            "item_id": item_id,
            "overlays": [
                _base_overlay_row(
                    term="Debt/Cash Flow Ratio",
                    formula="dlttq / oancfy",
                    variables=["dlttq", "oancfy"],
                ),
            ],
        },
    )

    values_path = paths.root / "values.json"
    _write_json(
        values_path,
        {
            "schema_version": "compustat_company_values_v1",
            "companies": [
                {
                    "company_id": "company-a",
                    "item_ids": [item_id],
                    "values": {"dlttq": 120.0},
                }
            ],
        },
    )

    run_compustat_overlay_eval_v1(
        paths,
        [item_id],
        overlay_output_subdir=overlay_subdir,
        output_subdir="eval-out",
        values_path=values_path,
    )

    out_path = paths.run_dir / "compustat_overlay_eval_v1" / "eval-out" / f"{item_id}__compustat_overlay_eval.json"
    out = json.loads(out_path.read_text(encoding="utf-8"))
    result = out["results"][0]
    assert result["status"] == "missing_values"
    assert result["missing_variables"] == ["oancfy"]


def test_compustat_overlay_eval_v1_candidates_schema_selected_formula(tmp_path: Path) -> None:
    paths = _make_paths(tmp_path)
    item_id = "item-a"
    overlay_subdir = "overlay-subdir"

    _write_json(
        paths.run_dir / "compustat_overlay_v1" / overlay_subdir / f"{item_id}__compustat_overlay.json",
        {
            "schema_version": "compustat_overlay_v1",
            "item_id": item_id,
            "overlays": [
                {
                    "schema_version": "compustat_overlay_candidates_v1",
                    "term": "EBITDA",
                    "selected_candidate_idx": 1,
                    "candidates": [
                        {
                            "why": "No direct mapping",
                            "compustat_formula": None,
                            "compustat_variables": [],
                            "match_type": "none",
                            "confidence": "low",
                            "limitations": ["No direct formula"],
                            "notes": ["next_step: manual work"],
                        },
                        {
                            "why": "Approximate mapping",
                            "compustat_formula": "niq + xintq",
                            "compustat_variables": ["niq", "xintq"],
                            "match_type": "approximate",
                            "confidence": "medium",
                            "limitations": ["Approximation"],
                            "notes": ["assumption: proxy", "ranking: best candidate"],
                        },
                    ],
                }
            ],
        },
    )

    values_path = paths.root / "values.json"
    _write_json(
        values_path,
        {
            "schema_version": "compustat_company_values_v1",
            "companies": [
                {
                    "company_id": "company-a",
                    "item_ids": [item_id],
                    "values": {"niq": 10.0, "xintq": 2.0},
                }
            ],
        },
    )

    run_compustat_overlay_eval_v1(
        paths,
        [item_id],
        overlay_output_subdir=overlay_subdir,
        output_subdir="eval-out",
        values_path=values_path,
    )

    out_path = paths.run_dir / "compustat_overlay_eval_v1" / "eval-out" / f"{item_id}__compustat_overlay_eval.json"
    out = json.loads(out_path.read_text(encoding="utf-8"))
    result = out["results"][0]
    assert result["status"] == "evaluated"
    assert result["selected_candidate_idx"] == 1
    assert result["value"] == 12.0


def test_compustat_overlay_eval_v1_allows_empty_company_values_for_no_formula(tmp_path: Path) -> None:
    paths = _make_paths(tmp_path)
    item_id = "item-a"
    overlay_subdir = "overlay-subdir"

    _write_json(
        paths.run_dir / "compustat_overlay_v1" / overlay_subdir / f"{item_id}__compustat_overlay.json",
        {
            "schema_version": "compustat_overlay_v1",
            "item_id": item_id,
            "overlays": [
                _base_overlay_row(term="S&P Rating", formula=None, variables=[]),
            ],
        },
    )

    values_path = paths.root / "values.json"
    _write_json(
        values_path,
        {
            "schema_version": "compustat_company_values_v1",
            "companies": [
                {
                    "company_id": "company-a",
                    "item_ids": [item_id],
                    "values": {},
                }
            ],
        },
    )

    run_compustat_overlay_eval_v1(
        paths,
        [item_id],
        overlay_output_subdir=overlay_subdir,
        output_subdir="eval-out",
        values_path=values_path,
    )

    out_path = paths.run_dir / "compustat_overlay_eval_v1" / "eval-out" / f"{item_id}__compustat_overlay_eval.json"
    out = json.loads(out_path.read_text(encoding="utf-8"))
    assert out["results"][0]["status"] == "no_formula"
