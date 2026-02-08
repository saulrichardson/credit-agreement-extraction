#!/usr/bin/env python3
"""
Build a run-level, item-by-item scorecard across the full extraction pipeline.

This script is artifact-first: it reads existing run outputs and emits:
  - JSON scorecard
  - CSV scorecard
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _iter_item_ids_from_manifest(run_dir: Path) -> List[str]:
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.exists():
        return []
    try:
        manifest = _read_json(manifest_path)
    except Exception:
        return []
    rows = manifest.get("items")
    if not isinstance(rows, list):
        return []
    item_ids: List[str] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        item_id = row.get("item_id")
        if isinstance(item_id, str) and item_id:
            item_ids.append(item_id)
    return sorted(set(item_ids))


def _iter_item_ids_from_ingest(run_dir: Path) -> List[str]:
    ingest = run_dir / "ingest"
    if not ingest.exists():
        return []
    out: List[str] = []
    for path in ingest.glob("*.html"):
        out.append(path.stem)
    return sorted(set(out))


def _item_ids(run_dir: Path) -> List[str]:
    ids = _iter_item_ids_from_manifest(run_dir)
    if ids:
        return ids
    return _iter_item_ids_from_ingest(run_dir)


def _latest_file(paths: Sequence[Path]) -> Optional[Path]:
    files = [p for p in paths if p.exists()]
    if not files:
        return None
    return sorted(files, key=lambda p: p.stat().st_mtime, reverse=True)[0]


def _glob_latest(root: Path, pattern: str) -> Optional[Path]:
    return _latest_file(list(root.glob(pattern)))


def _stage_status_from_files(*, exists: bool, warnings: int = 0, errors: int = 0) -> str:
    if errors > 0:
        return "error"
    if not exists:
        return "missing"
    if warnings > 0:
        return "warning"
    return "ok"


def _load_contractir_summary(run_dir: Path) -> Dict[str, Dict[str, Any]]:
    summary_path = run_dir / "contractir_v0_2" / "summary.json"
    if not summary_path.exists():
        return {}
    try:
        summary = _read_json(summary_path)
    except Exception:
        return {}
    out: Dict[str, Dict[str, Any]] = {}
    rows = summary.get("items")
    if not isinstance(rows, list):
        return out
    for row in rows:
        if not isinstance(row, dict):
            continue
        item_id = row.get("item_id")
        if isinstance(item_id, str) and item_id:
            out[item_id] = row
    return out


def _load_covenant_result_map(run_dir: Path) -> Dict[str, Dict[str, Any]]:
    # Preferred source: per-item result.json staging dir.
    staged = run_dir / "covenantir_v0_1"
    out: Dict[str, Dict[str, Any]] = {}
    if staged.exists():
        for item_dir in sorted(staged.iterdir()):
            if not item_dir.is_dir():
                continue
            result_path = item_dir / "result.json"
            if not result_path.exists():
                continue
            try:
                row = _read_json(result_path)
            except Exception:
                continue
            if isinstance(row, dict):
                out[item_dir.name] = row
        if out:
            return out

    # Fallback: newest batch summary.
    batch_root = run_dir / "covenant_ir_v0_1"
    if not batch_root.exists():
        return {}
    summary_path = _glob_latest(batch_root, "*/summary.json")
    if summary_path is None:
        return {}
    try:
        summary = _read_json(summary_path)
    except Exception:
        return {}
    rows = summary.get("results")
    if not isinstance(rows, list):
        return {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        item_id = row.get("item_id")
        if isinstance(item_id, str) and item_id:
            out[item_id] = row
    return out


def _load_provenance_map(run_dir: Path, explicit_path: Optional[Path]) -> Tuple[Optional[Path], Dict[str, Dict[str, Any]]]:
    chosen = explicit_path
    if chosen is None:
        report_dir = run_dir / "report"
        chosen = _glob_latest(report_dir, "provenance_audit*.json")
        if chosen is None:
            fallback = run_dir / "provenance_audit.json"
            chosen = fallback if fallback.exists() else None
    if chosen is None or not chosen.exists():
        return (None, {})
    try:
        report = _read_json(chosen)
    except Exception:
        return (chosen, {})
    rows = report.get("items")
    if not isinstance(rows, dict):
        return (chosen, {})
    out: Dict[str, Dict[str, Any]] = {}
    for item_id, row in rows.items():
        if isinstance(item_id, str) and isinstance(row, dict):
            out[item_id] = row
    return (chosen, out)


def _pricing_dir_for_version(run_dir: Path, version: str) -> Optional[Path]:
    root = run_dir / "contract_pricing"
    if not root.exists():
        return None
    candidates: List[Path] = []
    for child in root.iterdir():
        if not child.is_dir():
            continue
        name = child.name
        if name == "_context":
            continue
        if version == "v3":
            if "contract_pricing_v3" in name:
                candidates.append(child)
        else:
            if "contract_pricing_v3" not in name and name.startswith("contract_pricing_"):
                candidates.append(child)
    return _latest_file(candidates)


def _count_statuses(values: Iterable[str]) -> Dict[str, int]:
    c = Counter(values)
    return dict(sorted(c.items(), key=lambda kv: (-kv[1], kv[0])))


def _has_financial_covenant_snippet(run_dir: Path, item_id: str) -> Optional[bool]:
    snippets_path = run_dir / "retrieval_v2" / f"{item_id}_snippets.jsonl"
    if not snippets_path.exists():
        return None
    try:
        with snippets_path.open("r", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                if not isinstance(obj, dict):
                    continue
                cats = obj.get("categories")
                if isinstance(cats, list) and "financial_covenant" in cats:
                    return True
    except Exception:
        return None
    return False


@dataclass(frozen=True)
class StageRecord:
    status: str
    details: Dict[str, Any]


def _score_status(status: str) -> int:
    if status in {"ok", "skipped_no_financial_covenants"}:
        return 0
    if status in {"error", "missing"}:
        return 5
    if status in {"blocked_artifact", "blocked_legacy"}:
        return 4
    if status in {"warning", "unknown"}:
        return 2
    return 0


def _stage_ingest(run_dir: Path, item_id: str) -> StageRecord:
    p = run_dir / "ingest" / f"{item_id}.html"
    return StageRecord(status="ok" if p.exists() else "missing", details={"path": str(p), "exists": p.exists()})


def _stage_normalize(run_dir: Path, item_id: str) -> StageRecord:
    base = run_dir / "normalized" / item_id
    canonical = base / "canonical.txt"
    anchors = base / "anchors.tsv"
    exists = canonical.exists() and anchors.exists()
    return StageRecord(
        status="ok" if exists else "missing",
        details={"canonical_path": str(canonical), "anchors_path": str(anchors), "exists": exists},
    )


def _stage_toc(run_dir: Path, item_id: str) -> StageRecord:
    matches = sorted((run_dir / "toc_v1").glob(f"*/{item_id}.json"))
    p = _latest_file(matches)
    return StageRecord(
        status="ok" if p is not None else "missing",
        details={"path": str(p) if p else None, "matches": len(matches)},
    )


def _stage_indexing(run_dir: Path, item_id: str) -> StageRecord:
    p = run_dir / "indexing_v2" / f"{item_id}_anchors.json"
    return StageRecord(status="ok" if p.exists() else "missing", details={"path": str(p), "exists": p.exists()})


def _stage_retrieval(run_dir: Path, item_id: str) -> StageRecord:
    p = run_dir / "retrieval_v2" / f"{item_id}_snippets.jsonl"
    return StageRecord(status="ok" if p.exists() else "missing", details={"path": str(p), "exists": p.exists()})


def _stage_structured(run_dir: Path, item_id: str) -> StageRecord:
    qa_root = run_dir / "llm_qa"
    json_paths = sorted(qa_root.glob(f"*/{item_id}.json"))
    err_paths = sorted(qa_root.glob(f"*/{item_id}.error.txt"))
    raw_paths = sorted(qa_root.glob(f"*/{item_id}.raw.txt"))
    exists = bool(json_paths)
    status = _stage_status_from_files(exists=exists, warnings=1 if (exists and err_paths) else 0, errors=0 if exists else len(err_paths))
    return StageRecord(
        status=status,
        details={
            "json_path": str(_latest_file(json_paths)) if json_paths else None,
            "error_count": len(err_paths),
            "raw_count": len(raw_paths),
        },
    )


def _stage_doc_ir(run_dir: Path, item_id: str) -> StageRecord:
    p = run_dir / "doc_ir" / f"{item_id}.json"
    return StageRecord(status="ok" if p.exists() else "missing", details={"path": str(p), "exists": p.exists()})


def _stage_pricing_index(run_dir: Path, item_id: str) -> StageRecord:
    p = run_dir / "pricing_index" / f"{item_id}.json"
    return StageRecord(status="ok" if p.exists() else "missing", details={"path": str(p), "exists": p.exists()})


def _stage_definitions_v2(run_dir: Path, item_id: str) -> StageRecord:
    defs_root = run_dir / "definitions_v2"
    files = sorted(defs_root.glob(f"*/*{item_id}__*__definition.json"))
    return StageRecord(
        status="ok" if files else "missing",
        details={"definition_json_count": len(files), "sample_path": str(files[0]) if files else None},
    )


def _stage_facility(run_dir: Path, item_id: str) -> StageRecord:
    files = sorted((run_dir / "facility_fundamentals").glob(f"*/{item_id}.json"))
    p = _latest_file(files)
    return StageRecord(
        status="ok" if p is not None else "missing",
        details={"path": str(p) if p else None, "matches": len(files)},
    )


def _stage_agreement_metadata(run_dir: Path, item_id: str) -> StageRecord:
    root = run_dir / "agreement_metadata"
    json_files = sorted(root.glob(f"*/{item_id}.json"))
    err_files = sorted(root.glob(f"*/{item_id}*.error.txt"))
    final_err_files = sorted(root.glob(f"*/{item_id}*.final.error.txt"))
    exists = bool(json_files)
    warnings = len(err_files) + len(final_err_files)
    status = _stage_status_from_files(exists=exists, warnings=warnings, errors=0 if exists else warnings)
    return StageRecord(
        status=status,
        details={
            "json_path": str(_latest_file(json_files)) if json_files else None,
            "error_count": len(err_files),
            "final_error_count": len(final_err_files),
        },
    )


def _stage_analysis_export(run_dir: Path, item_id: str) -> StageRecord:
    files = sorted((run_dir / "analysis_export").glob(f"*/{item_id}.json"))
    p = _latest_file(files)
    return StageRecord(
        status="ok" if p is not None else "missing",
        details={"path": str(p) if p else None, "matches": len(files)},
    )


def _stage_definitions_compiler(run_dir: Path, item_id: str) -> StageRecord:
    files = sorted((run_dir / "definitions_compiler_v1").glob(f"*/{item_id}__compiled.json"))
    p = _latest_file(files)
    return StageRecord(
        status="ok" if p is not None else "missing",
        details={"path": str(p) if p else None, "matches": len(files)},
    )


def _stage_blocking_terms(run_dir: Path, item_id: str) -> StageRecord:
    files = sorted((run_dir / "blocking_terms_compiler_v1").glob(f"*/{item_id}__compiled.json"))
    p = _latest_file(files)
    return StageRecord(
        status="ok" if p is not None else "missing",
        details={"path": str(p) if p else None, "matches": len(files)},
    )


def _stage_compustat_overlay(run_dir: Path, item_id: str) -> StageRecord:
    files = sorted((run_dir / "compustat_overlay_v1").glob(f"*/{item_id}__compustat_overlay.json"))
    p = _latest_file(files)
    return StageRecord(
        status="ok" if p is not None else "missing",
        details={"path": str(p) if p else None, "matches": len(files)},
    )


def _stage_contract_pricing(run_dir: Path, item_id: str, *, version: str) -> StageRecord:
    pricing_dir = _pricing_dir_for_version(run_dir, version=version)
    if pricing_dir is None:
        return StageRecord(status="missing", details={"path": None, "warning": False, "error": False})
    json_path = pricing_dir / f"{item_id}.json"
    warning_path = pricing_dir / f"{item_id}.warning.txt"
    error_path = pricing_dir / f"{item_id}.error.txt"
    exists = json_path.exists()
    status = _stage_status_from_files(
        exists=exists,
        warnings=1 if warning_path.exists() else 0,
        errors=1 if error_path.exists() else 0,
    )
    return StageRecord(
        status=status,
        details={
            "path": str(json_path),
            "exists": exists,
            "warning": warning_path.exists(),
            "error": error_path.exists(),
        },
    )


def _stage_contractir(run_dir: Path, item_id: str, summary_map: Mapping[str, Mapping[str, Any]]) -> StageRecord:
    merged = run_dir / "contractir_v0_2" / "items" / item_id / "contractir_merged.json"
    row = summary_map.get(item_id)
    exists = merged.exists()
    if exists and isinstance(row, dict):
        ok = bool(row.get("ok"))
        status = "ok" if ok else "error"
    elif exists:
        status = "ok"
    elif isinstance(row, dict):
        status = "error"
    else:
        status = "missing"
    return StageRecord(
        status=status,
        details={
            "merged_path": str(merged),
            "exists": exists,
            "summary_ok": bool(row.get("ok")) if isinstance(row, dict) else None,
            "base_rate_attempts": (row.get("base_rate") or {}).get("attempts_used") if isinstance(row, dict) else None,
            "spread_attempts": (row.get("spread") or {}).get("attempts_used") if isinstance(row, dict) else None,
            "fee_attempts": (row.get("fee") or {}).get("attempts_used") if isinstance(row, dict) else None,
        },
    )


def _stage_covenantir(run_dir: Path, item_id: str, result_map: Mapping[str, Mapping[str, Any]]) -> StageRecord:
    staged_dir = run_dir / "covenantir_v0_1" / item_id
    validated_path = staged_dir / "covenantir_validated.json"
    blocked_artifact_path = staged_dir / "blocked_artifact.json"
    row = result_map.get(item_id)
    has_financial_cov = _has_financial_covenant_snippet(run_dir, item_id)
    if isinstance(row, dict):
        raw_status = str(row.get("status") or "unknown")
        if raw_status == "ok":
            status = "ok"
        elif raw_status == "blocked_artifact":
            status = "blocked_artifact"
        elif raw_status in {"blocked", "blocked_invalid_output"}:
            status = "blocked_legacy"
        else:
            status = "error"
    elif validated_path.exists():
        status = "unknown"
    elif has_financial_cov is False:
        status = "skipped_no_financial_covenants"
    else:
        status = "missing"
    return StageRecord(
        status=status,
        details={
            "result_status": str(row.get("status")) if isinstance(row, dict) and row.get("status") is not None else None,
            "blocked_reason": row.get("blocked_reason") if isinstance(row, dict) else None,
            "ok": bool(row.get("ok")) if isinstance(row, dict) and "ok" in row else None,
            "open_items_total": row.get("open_items_total") if isinstance(row, dict) else None,
            "open_items_blocking": row.get("open_items_blocking") if isinstance(row, dict) else None,
            "attempts_used": row.get("attempts_used") if isinstance(row, dict) else None,
            "has_financial_covenants_in_retrieval": has_financial_cov,
            "validated_exists": validated_path.exists(),
            "blocked_artifact_exists": blocked_artifact_path.exists(),
            "item_dir": str(staged_dir),
        },
    )


def _stage_provenance(item_id: str, provenance_map: Mapping[str, Mapping[str, Any]]) -> StageRecord:
    row = provenance_map.get(item_id)
    if not isinstance(row, dict):
        return StageRecord(
            status="missing",
            details={
                "pricing_anchor_missing_count": None,
                "pricing_anchor_empty_count": None,
                "pricing_unsupported_literals_count": None,
                "covenants_anchor_missing_count": None,
                "covenants_anchor_empty_count": None,
                "covenants_unsupported_literals_count": None,
            },
        )
    pricing = row.get("pricing") if isinstance(row.get("pricing"), dict) else {}
    cov = row.get("covenants") if isinstance(row.get("covenants"), dict) else {}
    pa_miss = int(pricing.get("anchor_missing_count") or 0)
    pa_empty = int(pricing.get("anchor_empty_count") or 0)
    pa_unsup = int(pricing.get("unsupported_literals_count") or 0)
    ca_miss = int(cov.get("anchor_missing_count") or 0)
    ca_empty = int(cov.get("anchor_empty_count") or 0)
    ca_unsup = int(cov.get("unsupported_literals_count") or 0)
    if (pa_miss + pa_empty + ca_miss + ca_empty) > 0:
        status = "error"
    elif (pa_unsup + ca_unsup) > 0:
        status = "warning"
    else:
        status = "ok"
    return StageRecord(
        status=status,
        details={
            "pricing_anchor_missing_count": pa_miss,
            "pricing_anchor_empty_count": pa_empty,
            "pricing_unsupported_literals_count": pa_unsup,
            "covenants_anchor_missing_count": ca_miss,
            "covenants_anchor_empty_count": ca_empty,
            "covenants_unsupported_literals_count": ca_unsup,
        },
    )


def _critical_path_status(statuses: Sequence[str]) -> str:
    if any(s in {"error", "missing"} for s in statuses):
        return "error"
    if any(s in {"warning", "blocked_artifact", "blocked_legacy", "unknown"} for s in statuses):
        return "warning"
    return "ok"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--base-dir", default=".")
    ap.add_argument("--out-json", default=None)
    ap.add_argument("--out-csv", default=None)
    ap.add_argument("--provenance-json", default=None, help="Optional explicit provenance audit JSON path.")
    args = ap.parse_args()

    base_dir = Path(args.base_dir)
    run_dir = base_dir / "runs" / args.run_id
    if not run_dir.exists():
        raise SystemExit(f"Run directory not found: {run_dir}")

    item_ids = _item_ids(run_dir)
    if not item_ids:
        raise SystemExit(f"No item ids found for run: {args.run_id}")

    provenance_path, provenance_map = _load_provenance_map(
        run_dir=run_dir,
        explicit_path=Path(args.provenance_json) if args.provenance_json else None,
    )
    contractir_summary = _load_contractir_summary(run_dir)
    covenant_result_map = _load_covenant_result_map(run_dir)

    stage_names = [
        "ingest",
        "normalize",
        "toc_v1",
        "indexing_v2",
        "retrieval_v2",
        "structured_v2",
        "doc_ir",
        "pricing_index",
        "definitions_v2",
        "facility_fundamentals",
        "agreement_metadata",
        "analysis_export",
        "definitions_compiler_v1",
        "blocking_terms_compiler_v1",
        "compustat_overlay_v1",
        "contract_pricing_v1",
        "contract_pricing_v3",
        "contractir_v0_2",
        "covenantir_v0_1",
        "provenance_audit",
    ]
    core_stage_names = [
        "ingest",
        "normalize",
        "toc_v1",
        "indexing_v2",
        "retrieval_v2",
        "structured_v2",
        "doc_ir",
        "pricing_index",
        "contract_pricing_v3",
        "contractir_v0_2",
        "covenantir_v0_1",
        "provenance_audit",
    ]

    stage_status_counter: Dict[str, Counter[str]] = {name: Counter() for name in stage_names}
    score_rows: List[Dict[str, Any]] = []

    for item_id in item_ids:
        stages: Dict[str, StageRecord] = {
            "ingest": _stage_ingest(run_dir, item_id),
            "normalize": _stage_normalize(run_dir, item_id),
            "toc_v1": _stage_toc(run_dir, item_id),
            "indexing_v2": _stage_indexing(run_dir, item_id),
            "retrieval_v2": _stage_retrieval(run_dir, item_id),
            "structured_v2": _stage_structured(run_dir, item_id),
            "doc_ir": _stage_doc_ir(run_dir, item_id),
            "pricing_index": _stage_pricing_index(run_dir, item_id),
            "definitions_v2": _stage_definitions_v2(run_dir, item_id),
            "facility_fundamentals": _stage_facility(run_dir, item_id),
            "agreement_metadata": _stage_agreement_metadata(run_dir, item_id),
            "analysis_export": _stage_analysis_export(run_dir, item_id),
            "definitions_compiler_v1": _stage_definitions_compiler(run_dir, item_id),
            "blocking_terms_compiler_v1": _stage_blocking_terms(run_dir, item_id),
            "compustat_overlay_v1": _stage_compustat_overlay(run_dir, item_id),
            "contract_pricing_v1": _stage_contract_pricing(run_dir, item_id, version="v1"),
            "contract_pricing_v3": _stage_contract_pricing(run_dir, item_id, version="v3"),
            "contractir_v0_2": _stage_contractir(run_dir, item_id, summary_map=contractir_summary),
            "covenantir_v0_1": _stage_covenantir(run_dir, item_id, result_map=covenant_result_map),
            "provenance_audit": _stage_provenance(item_id, provenance_map),
        }

        for name, rec in stages.items():
            stage_status_counter[name][rec.status] += 1

        stage_statuses = [stages[name].status for name in stage_names]
        critical_stage_statuses = [stages[name].status for name in core_stage_names]
        failing_stages = [
            name
            for name in stage_names
            if stages[name].status not in {"ok", "skipped_no_financial_covenants"}
        ]
        score = sum(_score_status(s) for s in stage_statuses)
        prov = stages["provenance_audit"].details
        score += min(
            10,
            int(prov.get("pricing_unsupported_literals_count") or 0)
            + int(prov.get("covenants_unsupported_literals_count") or 0),
        )
        critical_status = _critical_path_status(critical_stage_statuses)

        row: Dict[str, Any] = {
            "item_id": item_id,
            "critical_status": critical_status,
            "improvement_score": score,
            "failing_stages": failing_stages,
            "needs_attention": bool(failing_stages),
        }
        for name in stage_names:
            row[f"{name}_status"] = stages[name].status
            row[f"{name}_details"] = stages[name].details

        row["covenant_open_items_blocking"] = stages["covenantir_v0_1"].details.get("open_items_blocking")
        row["covenant_blocked_reason"] = stages["covenantir_v0_1"].details.get("blocked_reason")
        row["provenance_pricing_unsupported_literals_count"] = prov.get("pricing_unsupported_literals_count")
        row["provenance_covenants_unsupported_literals_count"] = prov.get("covenants_unsupported_literals_count")
        row["provenance_pricing_anchor_missing_count"] = prov.get("pricing_anchor_missing_count")
        row["provenance_covenants_anchor_missing_count"] = prov.get("covenants_anchor_missing_count")
        score_rows.append(row)

    score_rows.sort(key=lambda r: (-int(r.get("improvement_score") or 0), r["item_id"]))

    by_critical = _count_statuses([str(r.get("critical_status") or "") for r in score_rows])
    by_improvement_band = defaultdict(int)
    for row in score_rows:
        sc = int(row.get("improvement_score") or 0)
        if sc >= 25:
            by_improvement_band["high"] += 1
        elif sc >= 10:
            by_improvement_band["medium"] += 1
        else:
            by_improvement_band["low"] += 1

    stage_status_counts = {
        stage: dict(sorted(counter.items(), key=lambda kv: (-kv[1], kv[0])))
        for stage, counter in stage_status_counter.items()
    }

    out_json = Path(args.out_json) if args.out_json else (run_dir / "report" / f"run_scorecard_{args.run_id}.json")
    out_csv = Path(args.out_csv) if args.out_csv else (run_dir / "report" / f"run_scorecard_{args.run_id}.csv")

    report = {
        "run_id": args.run_id,
        "generated_at_utc": _now_iso(),
        "run_dir": str(run_dir),
        "item_count": len(item_ids),
        "provenance_report_path": str(provenance_path) if provenance_path else None,
        "stage_names": stage_names,
        "summary": {
            "by_critical_status": by_critical,
            "by_improvement_band": dict(sorted(by_improvement_band.items())),
            "stage_status_counts": stage_status_counts,
        },
        "items": score_rows,
    }
    _write_json(out_json, report)

    csv_columns = [
        "item_id",
        "critical_status",
        "improvement_score",
        "needs_attention",
    ]
    csv_columns.extend([f"{name}_status" for name in stage_names])
    csv_columns.extend(
        [
            "covenant_blocked_reason",
            "covenant_open_items_blocking",
            "provenance_pricing_unsupported_literals_count",
            "provenance_covenants_unsupported_literals_count",
            "provenance_pricing_anchor_missing_count",
            "provenance_covenants_anchor_missing_count",
            "failing_stages",
        ]
    )

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=csv_columns)
        writer.writeheader()
        for row in score_rows:
            flat = {k: row.get(k) for k in csv_columns}
            flat["failing_stages"] = ";".join(row.get("failing_stages") or [])
            writer.writerow(flat)

    print(str(out_json))
    print(str(out_csv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
