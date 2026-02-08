#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


PROHIBITED_NAME_RULES: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\bpricing\s*(level|regime|tier)s?\b", flags=re.IGNORECASE), "synthetic pricing outcome label"),
    (re.compile(r"\bapplicable\s*(margin|fee|spread)\s*(level|regime|tier)?\b", flags=re.IGNORECASE), "outcome label, not driver"),
]


@dataclass
class Violation:
    item_id: str
    metric_name: str
    violation_type: str
    reason: str
    source_refs: list[str]
    ratings_context: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            "item_id": self.item_id,
            "metric_name": self.metric_name,
            "violation_type": self.violation_type,
            "reason": self.reason,
            "source_refs": self.source_refs,
            "ratings_context": self.ratings_context,
        }


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _iter_metric_names_from_conditions(conditions: Any) -> set[str]:
    out: set[str] = set()
    if isinstance(conditions, list):
        for row in conditions:
            if isinstance(row, dict):
                metric = row.get("metric")
                if isinstance(metric, str) and metric.strip():
                    out.add(metric.strip())
        return out
    if isinstance(conditions, dict):
        for key in ("all", "any"):
            rows = conditions.get(key)
            if not isinstance(rows, list):
                continue
            for row in rows:
                if isinstance(row, dict):
                    metric = row.get("metric")
                    if isinstance(metric, str) and metric.strip():
                        out.add(metric.strip())
        return out
    return out


def _ratings_context_metrics(doc: dict[str, Any]) -> set[str]:
    ratings_metrics: set[str] = set()
    schemes_by_id: dict[str, dict[str, Any]] = {}
    for scheme in doc.get("tier_schemes") or []:
        if not isinstance(scheme, dict):
            continue
        scheme_id = scheme.get("scheme_id")
        if isinstance(scheme_id, str) and scheme_id.strip():
            schemes_by_id[scheme_id.strip()] = scheme

    for scheme in schemes_by_id.values():
        if scheme.get("classification") != "ratings_based":
            continue
        metric = scheme.get("metric")
        if isinstance(metric, str) and metric.strip():
            ratings_metrics.add(metric.strip())
        for tier in scheme.get("tiers") or []:
            if not isinstance(tier, dict):
                continue
            ratings_metrics.update(_iter_metric_names_from_conditions(tier.get("conditions")))

    for facility in doc.get("facilities") or []:
        if not isinstance(facility, dict):
            continue
        for rate in facility.get("rates") or []:
            if not isinstance(rate, dict):
                continue
            tier_scheme_ref = rate.get("tier_scheme_ref")
            if not isinstance(tier_scheme_ref, str) or not tier_scheme_ref.strip():
                continue
            scheme = schemes_by_id.get(tier_scheme_ref.strip())
            if not isinstance(scheme, dict) or scheme.get("classification") != "ratings_based":
                continue
            cond = rate.get("condition")
            if isinstance(cond, dict):
                metric = cond.get("metric")
                if isinstance(metric, str) and metric.strip():
                    ratings_metrics.add(metric.strip())

    return ratings_metrics


def _is_allowed_ratings_metric(name: str) -> bool:
    return bool(re.search(r"\b(rating|status|notch|grade)\b", name, flags=re.IGNORECASE))


def _audit_item(item_id: str, doc: dict[str, Any]) -> list[Violation]:
    out: list[Violation] = []
    ratings_context = _ratings_context_metrics(doc)
    metrics = doc.get("metrics")
    if not isinstance(metrics, list):
        return out

    for row in metrics:
        if not isinstance(row, dict):
            continue
        name = row.get("name")
        if not isinstance(name, str) or not name.strip():
            continue
        metric_name = name.strip()
        refs = [r.strip() for r in (row.get("source_refs") or []) if isinstance(r, str) and r.strip()]
        in_ratings_context = metric_name in ratings_context
        is_ratings_name = _is_allowed_ratings_metric(metric_name)

        for pattern, reason in PROHIBITED_NAME_RULES:
            if not pattern.search(metric_name):
                continue
            if in_ratings_context and is_ratings_name:
                continue
            out.append(
                Violation(
                    item_id=item_id,
                    metric_name=metric_name,
                    violation_type="non_driver_metric_name",
                    reason=reason,
                    source_refs=refs,
                    ratings_context=in_ratings_context,
                )
            )
            break

    return out


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Audit pricing extraction outputs for metric names that encode pricing outcomes "
            "instead of concrete tier drivers."
        )
    )
    parser.add_argument("--run-id", required=True, help="Run ID under runs/<run-id>/")
    parser.add_argument("--qa-subdir", required=True, help="Subdir under runs/<run-id>/llm_qa/")
    parser.add_argument("--base-dir", default=".", help="Repository root (default: current directory)")
    parser.add_argument(
        "--out-json",
        default=None,
        help="Output JSON path (default: runs/<run-id>/report/pricing_metric_driver_audit_<qa_subdir>.json)",
    )
    parser.add_argument(
        "--out-csv",
        default=None,
        help="Output CSV path (default: runs/<run-id>/report/pricing_metric_driver_audit_<qa_subdir>.csv)",
    )
    parser.add_argument("--fail-on-violations", action="store_true", help="Exit non-zero if violations are found")
    args = parser.parse_args()

    root = Path(args.base_dir).resolve()
    run_dir = root / "runs" / str(args.run_id)
    qa_dir = run_dir / "llm_qa" / str(args.qa_subdir)
    if not qa_dir.exists():
        raise SystemExit(f"qa-subdir not found: {qa_dir}")

    stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(args.qa_subdir)).strip("_") or "qa"
    report_dir = run_dir / "report"
    report_dir.mkdir(parents=True, exist_ok=True)
    out_json = Path(args.out_json) if args.out_json else (report_dir / f"pricing_metric_driver_audit_{stem}.json")
    out_csv = Path(args.out_csv) if args.out_csv else (report_dir / f"pricing_metric_driver_audit_{stem}.csv")

    violations: list[Violation] = []
    scanned_items = 0
    for path in sorted(qa_dir.glob("*.json")):
        scanned_items += 1
        try:
            doc = _read_json(path)
        except Exception as exc:
            violations.append(
                Violation(
                    item_id=path.stem,
                    metric_name="",
                    violation_type="invalid_json",
                    reason=f"{type(exc).__name__}: {exc}",
                    source_refs=[],
                    ratings_context=False,
                )
            )
            continue
        if not isinstance(doc, dict):
            violations.append(
                Violation(
                    item_id=path.stem,
                    metric_name="",
                    violation_type="invalid_shape",
                    reason="top-level output is not a JSON object",
                    source_refs=[],
                    ratings_context=False,
                )
            )
            continue
        violations.extend(_audit_item(path.stem, doc))

    payload = {
        "schema_version": "pricing_metric_driver_audit_v1",
        "run_id": str(args.run_id),
        "qa_subdir": str(args.qa_subdir),
        "scanned_items": scanned_items,
        "violation_count": len(violations),
        "violations": [v.as_dict() for v in violations],
    }
    out_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    with out_csv.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["item_id", "metric_name", "violation_type", "reason", "source_refs", "ratings_context"])
        for v in violations:
            writer.writerow(
                [
                    v.item_id,
                    v.metric_name,
                    v.violation_type,
                    v.reason,
                    ";".join(v.source_refs),
                    "true" if v.ratings_context else "false",
                ]
            )

    print(f"[done] scanned_items={scanned_items} violations={len(violations)}")
    print(str(out_json))
    print(str(out_csv))

    if args.fail_on_violations and violations:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
