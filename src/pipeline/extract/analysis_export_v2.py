from __future__ import annotations

import csv
import json
import time
from contextlib import ExitStack
from pathlib import Path
from typing import Any, Iterable, Literal

from pydantic import BaseModel, ConfigDict, Field

from pipeline.core.config import Paths, record_manifest, update_manifest
from pipeline.utils import load_manifest, manifest_items


class AnalysisRecordV2(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["analysis_record_v2"]
    run_id: str
    item_id: str
    created_at: int

    accession: str | None = None
    sequence: str | None = None
    filename: str | None = None
    source_path: str | None = None

    canonical_text_path: str | None = None
    anchors_tsv_path: str | None = None
    indexing_v2_path: str | None = None
    retrieval_v2_path: str | None = None

    agreement_metadata_path: str | None = None
    agreement_metadata: dict[str, Any] | None = None
    agreement_metadata_meta_path: str | None = None
    agreement_metadata_meta: dict[str, Any] | None = None

    pricing_structured_path: str | None = None
    pricing_structured: dict[str, Any] | list[Any] | None = None

    covenant_structured_path: str | None = None
    covenant_structured: dict[str, Any] | list[Any] | None = None

    pricing_metrics_recursive_path: str | None = None
    pricing_metrics_recursive: dict[str, Any] | None = None
    pricing_blocking_recursive_path: str | None = None
    pricing_blocking_recursive: dict[str, Any] | None = None
    pricing_overlay_recursive_path: str | None = None
    pricing_overlay_recursive: dict[str, Any] | None = None

    covenant_metrics_recursive_path: str | None = None
    covenant_metrics_recursive: dict[str, Any] | None = None
    covenant_blocking_recursive_path: str | None = None
    covenant_blocking_recursive: dict[str, Any] | None = None
    covenant_overlay_recursive_path: str | None = None
    covenant_overlay_recursive: dict[str, Any] | None = None

    errors: list[str] = Field(default_factory=list)


def _rel(path: Path) -> str:
    try:
        return str(path.relative_to(Path(".")))
    except Exception:
        return str(path)


def _load_json_dict(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Invalid JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError(f"Expected JSON object in {path}, got {type(payload).__name__}")
    return payload


def _load_json_any(path: Path) -> dict[str, Any] | list[Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Invalid JSON: {path}") from exc
    if not isinstance(payload, (dict, list)):
        raise RuntimeError(f"Expected JSON object/array in {path}, got {type(payload).__name__}")
    return payload


def _coerce_null_str(value: Any) -> Any:
    if isinstance(value, str) and value.strip().lower() == "null":
        return None
    return value


def _normalize_covenant_payload(payload: dict[str, Any] | list[Any]) -> dict[str, Any] | list[Any]:
    """Normalize common prompt_v1_short quirks without mutating run artifacts."""

    if not isinstance(payload, dict):
        return payload

    covs = payload.get("covenants")
    if isinstance(covs, list):
        for cov in covs:
            if not isinstance(cov, dict):
                continue
            for key in ("start_date", "end_date", "comparator"):
                if key in cov:
                    cov[key] = _coerce_null_str(cov.get(key))
    return payload


def _csv_join(values: Any) -> str:
    if values is None:
        return ""
    if isinstance(values, (list, tuple)):
        return ",".join(str(v) for v in values if v is not None)
    return str(values)


def _compiled_definitions_rows(doc: Any) -> list[dict[str, Any]]:
    if not isinstance(doc, dict):
        return []
    defs = doc.get("definitions")
    if not isinstance(defs, list):
        return []
    return [row for row in defs if isinstance(row, dict)]


def _recursive_paths_for_item(
    *,
    paths: Paths,
    item_id: str,
    pricing_metrics_output_subdir: str,
    pricing_blocking_output_subdir: str,
    pricing_overlay_output_subdir: str,
    covenant_metrics_output_subdir: str,
    covenant_blocking_output_subdir: str,
    covenant_overlay_output_subdir: str,
) -> dict[str, Path]:
    return {
        "pricing_metrics": paths.run_dir / "definitions_compiler_v1" / pricing_metrics_output_subdir / f"{item_id}__compiled.json",
        "pricing_blocking": paths.run_dir / "blocking_terms_compiler_v1" / pricing_blocking_output_subdir / f"{item_id}__compiled.json",
        "pricing_overlay": paths.run_dir / "compustat_overlay_v1" / pricing_overlay_output_subdir / f"{item_id}__compustat_overlay.json",
        "covenant_metrics": paths.run_dir / "definitions_compiler_v1" / covenant_metrics_output_subdir / f"{item_id}__compiled.json",
        "covenant_blocking": paths.run_dir / "blocking_terms_compiler_v1" / covenant_blocking_output_subdir / f"{item_id}__compiled.json",
        "covenant_overlay": paths.run_dir / "compustat_overlay_v1" / covenant_overlay_output_subdir / f"{item_id}__compustat_overlay.json",
    }


def run_analysis_export_v2(
    paths: Paths,
    item_ids: Iterable[str],
    *,
    agreement_metadata_subdir: str,
    pricing_qa_subdir: str,
    covenant_qa_subdir: str,
    pricing_metrics_output_subdir: str,
    pricing_blocking_output_subdir: str,
    pricing_overlay_output_subdir: str,
    covenant_metrics_output_subdir: str,
    covenant_blocking_output_subdir: str,
    covenant_overlay_output_subdir: str,
    output_subdir: str = "analysis_export_v2",
) -> None:
    """Export analysis-ready JSON artifacts per item (single record per agreement).

    This locked export bundles:
    - pricing/covenant structured-v2 outputs
    - recursive metric definitions + recursive blocking terms + recursive overlay outputs
    - agreement metadata + source pointers
    """

    manifest = load_manifest(paths.manifest_path)
    item_rows = {row["item_id"]: row for row in manifest_items(manifest)}

    out_root = paths.run_dir / "analysis_export"
    out_dir = out_root / output_subdir
    out_dir.mkdir(parents=True, exist_ok=True)

    bundle_path = out_dir / "records.jsonl"
    errors: list[tuple[str, str]] = []

    metadata_dir = paths.run_dir / "agreement_metadata" / agreement_metadata_subdir
    if not metadata_dir.exists():
        raise FileNotFoundError(
            f"Agreement metadata directory not found: expected {metadata_dir} (set --agreement-metadata-subdir)"
        )

    pricing_structured_dir = paths.structured_dir / pricing_qa_subdir
    if not pricing_structured_dir.exists():
        raise FileNotFoundError(
            f"Pricing structured output directory not found: expected {pricing_structured_dir} (set --pricing-qa-subdir)"
        )

    covenant_structured_dir = paths.structured_dir / covenant_qa_subdir
    if not covenant_structured_dir.exists():
        raise FileNotFoundError(
            f"Covenant structured output directory not found: expected {covenant_structured_dir} (set --covenant-qa-subdir)"
        )

    expected_recursive_dirs = {
        "pricing_metrics": paths.run_dir / "definitions_compiler_v1" / pricing_metrics_output_subdir,
        "pricing_blocking": paths.run_dir / "blocking_terms_compiler_v1" / pricing_blocking_output_subdir,
        "pricing_overlay": paths.run_dir / "compustat_overlay_v1" / pricing_overlay_output_subdir,
        "covenant_metrics": paths.run_dir / "definitions_compiler_v1" / covenant_metrics_output_subdir,
        "covenant_blocking": paths.run_dir / "blocking_terms_compiler_v1" / covenant_blocking_output_subdir,
        "covenant_overlay": paths.run_dir / "compustat_overlay_v1" / covenant_overlay_output_subdir,
    }
    for label, dir_path in expected_recursive_dirs.items():
        if not dir_path.exists():
            raise FileNotFoundError(f"Recursive output directory not found for {label}: expected {dir_path}")

    record_manifest(
        out_dir / "export.meta.json",
        {
            "schema_version": "analysis_export_v2_meta",
            "run_id": paths.run_id,
            "created_at": int(time.time()),
            "agreement_metadata_subdir": agreement_metadata_subdir,
            "pricing_qa_subdir": pricing_qa_subdir,
            "covenant_qa_subdir": covenant_qa_subdir,
            "pricing_metrics_output_subdir": pricing_metrics_output_subdir,
            "pricing_blocking_output_subdir": pricing_blocking_output_subdir,
            "pricing_overlay_output_subdir": pricing_overlay_output_subdir,
            "covenant_metrics_output_subdir": covenant_metrics_output_subdir,
            "covenant_blocking_output_subdir": covenant_blocking_output_subdir,
            "covenant_overlay_output_subdir": covenant_overlay_output_subdir,
            "output_subdir": output_subdir,
        },
    )

    tables_dir = out_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    with ExitStack() as stack:
        bundle = stack.enter_context(bundle_path.open("w"))

        borrowers_f = stack.enter_context((tables_dir / "borrowers.csv").open("w", newline=""))
        guarantors_f = stack.enter_context((tables_dir / "guarantors.csv").open("w", newline=""))
        agents_f = stack.enter_context((tables_dir / "agents.csv").open("w", newline=""))
        arrangers_f = stack.enter_context((tables_dir / "arrangers.csv").open("w", newline=""))
        lenders_f = stack.enter_context((tables_dir / "lenders.csv").open("w", newline=""))
        lenders_meta_f = stack.enter_context((tables_dir / "lenders_meta.csv").open("w", newline=""))
        facilities_f = stack.enter_context((tables_dir / "facilities.csv").open("w", newline=""))
        definitions_f = stack.enter_context((tables_dir / "definitions.csv").open("w", newline=""))
        covenants_f = stack.enter_context((tables_dir / "covenants.csv").open("w", newline=""))
        covenant_metrics_f = stack.enter_context((tables_dir / "covenant_metrics.csv").open("w", newline=""))

        borrowers_w = csv.DictWriter(
            borrowers_f,
            fieldnames=["run_id", "item_id", "name", "source_refs", "notes"],
        )
        guarantors_w = csv.DictWriter(
            guarantors_f,
            fieldnames=["run_id", "item_id", "name", "source_refs", "notes"],
        )
        agents_w = csv.DictWriter(
            agents_f,
            fieldnames=["run_id", "item_id", "role_label", "party_name", "source_refs", "notes"],
        )
        arrangers_w = csv.DictWriter(
            arrangers_f,
            fieldnames=["run_id", "item_id", "role_label", "party_name", "source_refs", "notes"],
        )
        lenders_w = csv.DictWriter(
            lenders_f,
            fieldnames=["run_id", "item_id", "name", "source_refs", "notes"],
        )
        lenders_meta_w = csv.DictWriter(
            lenders_meta_f,
            fieldnames=[
                "run_id",
                "item_id",
                "lenders_present_but_omitted",
                "lenders_source_refs",
                "notes",
                "auto_corrections_count",
            ],
        )
        facilities_w = csv.DictWriter(
            facilities_f,
            fieldnames=[
                "run_id",
                "item_id",
                "facility_name",
                "facility_types",
                "committed_amount_amount",
                "committed_amount_currency",
                "currency",
                "maturity_date",
                "termination_date",
                "source_refs",
                "notes",
            ],
        )
        definitions_w = csv.DictWriter(
            definitions_f,
            fieldnames=[
                "run_id",
                "item_id",
                "source",
                "term",
                "term_type",
                "definition_text",
                "anchor_refs",
                "notes",
            ],
        )
        covenants_w = csv.DictWriter(
            covenants_f,
            fieldnames=[
                "run_id",
                "item_id",
                "covenant_id",
                "name",
                "metric_refs",
                "comparator",
                "limit",
                "limit_text",
                "start_date",
                "end_date",
                "source_refs",
                "notes",
            ],
        )
        covenant_metrics_w = csv.DictWriter(
            covenant_metrics_f,
            fieldnames=[
                "run_id",
                "item_id",
                "metric_id",
                "name",
                "contract_term",
                "search_tokens",
                "source_refs",
                "notes",
            ],
        )

        for w in (
            borrowers_w,
            guarantors_w,
            agents_w,
            arrangers_w,
            lenders_w,
            lenders_meta_w,
            facilities_w,
            definitions_w,
            covenants_w,
            covenant_metrics_w,
        ):
            w.writeheader()

        for item_id in item_ids:
            created_at = int(time.time())
            row = item_rows.get(item_id) or {}
            record = AnalysisRecordV2(
                schema_version="analysis_record_v2",
                run_id=paths.run_id,
                item_id=item_id,
                created_at=created_at,
                accession=row.get("accession"),
                sequence=row.get("sequence"),
                filename=row.get("filename"),
                source_path=row.get("path"),
            )

            canonical = paths.normalized_dir / item_id / "canonical.txt"
            anchors_tsv = paths.normalized_dir / item_id / "anchors.tsv"
            indexing_v2 = paths.run_dir / "indexing_v2" / f"{item_id}_anchors.json"
            retrieval_v2 = paths.run_dir / "retrieval_v2" / f"{item_id}_snippets.jsonl"
            for label, path in (
                ("canonical.txt", canonical),
                ("anchors.tsv", anchors_tsv),
                ("indexing_v2", indexing_v2),
                ("retrieval_v2", retrieval_v2),
            ):
                if not path.exists():
                    record.errors.append(f"missing {label}: {path}")

            record.canonical_text_path = _rel(canonical) if canonical.exists() else None
            record.anchors_tsv_path = _rel(anchors_tsv) if anchors_tsv.exists() else None
            record.indexing_v2_path = _rel(indexing_v2) if indexing_v2.exists() else None
            record.retrieval_v2_path = _rel(retrieval_v2) if retrieval_v2.exists() else None

            metadata_path = metadata_dir / f"{item_id}.json"
            record.agreement_metadata_path = _rel(metadata_path) if metadata_path.exists() else None
            if metadata_path.exists():
                try:
                    record.agreement_metadata = _load_json_dict(metadata_path)
                except Exception as exc:
                    record.errors.append(f"agreement_metadata invalid: {exc}")
            else:
                record.errors.append(f"missing agreement_metadata: {metadata_path}")

            metadata_meta_path = metadata_dir / f"{item_id}.meta.json"
            record.agreement_metadata_meta_path = _rel(metadata_meta_path) if metadata_meta_path.exists() else None
            if metadata_meta_path.exists():
                try:
                    record.agreement_metadata_meta = _load_json_dict(metadata_meta_path)
                except Exception as exc:
                    record.errors.append(f"agreement_metadata meta invalid: {exc}")

            pricing_structured_path = pricing_structured_dir / f"{item_id}.json"
            record.pricing_structured_path = _rel(pricing_structured_path) if pricing_structured_path.exists() else None
            if pricing_structured_path.exists():
                try:
                    record.pricing_structured = _load_json_any(pricing_structured_path)
                except Exception as exc:
                    record.errors.append(f"pricing structured invalid: {exc}")
            else:
                record.errors.append(f"missing pricing structured: {pricing_structured_path}")

            covenant_structured_path = covenant_structured_dir / f"{item_id}.json"
            record.covenant_structured_path = (
                _rel(covenant_structured_path) if covenant_structured_path.exists() else None
            )
            if covenant_structured_path.exists():
                try:
                    record.covenant_structured = _normalize_covenant_payload(_load_json_any(covenant_structured_path))
                except Exception as exc:
                    record.errors.append(f"covenant structured invalid: {exc}")
            else:
                record.errors.append(f"missing covenant structured: {covenant_structured_path}")

            rec_paths = _recursive_paths_for_item(
                paths=paths,
                item_id=item_id,
                pricing_metrics_output_subdir=pricing_metrics_output_subdir,
                pricing_blocking_output_subdir=pricing_blocking_output_subdir,
                pricing_overlay_output_subdir=pricing_overlay_output_subdir,
                covenant_metrics_output_subdir=covenant_metrics_output_subdir,
                covenant_blocking_output_subdir=covenant_blocking_output_subdir,
                covenant_overlay_output_subdir=covenant_overlay_output_subdir,
            )
            for label, path in rec_paths.items():
                if not path.exists():
                    record.errors.append(f"missing {label}: {path}")

            try:
                if rec_paths["pricing_metrics"].exists():
                    record.pricing_metrics_recursive_path = _rel(rec_paths["pricing_metrics"])
                    record.pricing_metrics_recursive = _load_json_dict(rec_paths["pricing_metrics"])
            except Exception as exc:
                record.errors.append(f"pricing metrics recursive invalid: {exc}")

            try:
                if rec_paths["pricing_blocking"].exists():
                    record.pricing_blocking_recursive_path = _rel(rec_paths["pricing_blocking"])
                    record.pricing_blocking_recursive = _load_json_dict(rec_paths["pricing_blocking"])
            except Exception as exc:
                record.errors.append(f"pricing blocking recursive invalid: {exc}")

            try:
                if rec_paths["pricing_overlay"].exists():
                    record.pricing_overlay_recursive_path = _rel(rec_paths["pricing_overlay"])
                    record.pricing_overlay_recursive = _load_json_dict(rec_paths["pricing_overlay"])
            except Exception as exc:
                record.errors.append(f"pricing overlay recursive invalid: {exc}")

            try:
                if rec_paths["covenant_metrics"].exists():
                    record.covenant_metrics_recursive_path = _rel(rec_paths["covenant_metrics"])
                    record.covenant_metrics_recursive = _load_json_dict(rec_paths["covenant_metrics"])
            except Exception as exc:
                record.errors.append(f"covenant metrics recursive invalid: {exc}")

            try:
                if rec_paths["covenant_blocking"].exists():
                    record.covenant_blocking_recursive_path = _rel(rec_paths["covenant_blocking"])
                    record.covenant_blocking_recursive = _load_json_dict(rec_paths["covenant_blocking"])
            except Exception as exc:
                record.errors.append(f"covenant blocking recursive invalid: {exc}")

            try:
                if rec_paths["covenant_overlay"].exists():
                    record.covenant_overlay_recursive_path = _rel(rec_paths["covenant_overlay"])
                    record.covenant_overlay_recursive = _load_json_dict(rec_paths["covenant_overlay"])
            except Exception as exc:
                record.errors.append(f"covenant overlay recursive invalid: {exc}")

            item_out = out_dir / f"{item_id}.json"
            item_out.write_text(record.model_dump_json(indent=2), encoding="utf-8")
            bundle.write(record.model_dump_json() + "\n")

            if record.errors:
                errors.append((item_id, "; ".join(record.errors)))
                continue

            meta = record.agreement_metadata or {}
            for p in meta.get("borrowers") or []:
                borrowers_w.writerow(
                    {
                        "run_id": record.run_id,
                        "item_id": record.item_id,
                        "name": p.get("name"),
                        "source_refs": _csv_join(p.get("source_refs")),
                        "notes": p.get("notes"),
                    }
                )
            for p in meta.get("guarantors") or []:
                guarantors_w.writerow(
                    {
                        "run_id": record.run_id,
                        "item_id": record.item_id,
                        "name": p.get("name"),
                        "source_refs": _csv_join(p.get("source_refs")),
                        "notes": p.get("notes"),
                    }
                )
            for r in meta.get("agents") or []:
                agents_w.writerow(
                    {
                        "run_id": record.run_id,
                        "item_id": record.item_id,
                        "role_label": r.get("role_label"),
                        "party_name": r.get("party_name"),
                        "source_refs": _csv_join(r.get("source_refs")),
                        "notes": r.get("notes"),
                    }
                )
            for r in meta.get("arrangers") or []:
                arrangers_w.writerow(
                    {
                        "run_id": record.run_id,
                        "item_id": record.item_id,
                        "role_label": r.get("role_label"),
                        "party_name": r.get("party_name"),
                        "source_refs": _csv_join(r.get("source_refs")),
                        "notes": r.get("notes"),
                    }
                )
            for p in meta.get("lenders") or []:
                lenders_w.writerow(
                    {
                        "run_id": record.run_id,
                        "item_id": record.item_id,
                        "name": p.get("name"),
                        "source_refs": _csv_join(p.get("source_refs")),
                        "notes": p.get("notes"),
                    }
                )
            lenders_meta_w.writerow(
                {
                    "run_id": record.run_id,
                    "item_id": record.item_id,
                    "lenders_present_but_omitted": meta.get("lenders_present_but_omitted"),
                    "lenders_source_refs": _csv_join(meta.get("lenders_source_refs")),
                    "notes": meta.get("notes"),
                    "auto_corrections_count": (
                        len((record.agreement_metadata_meta or {}).get("auto_corrections") or [])
                        if isinstance(record.agreement_metadata_meta, dict)
                        else ""
                    ),
                }
            )
            for f in meta.get("facilities") or []:
                committed = f.get("committed_amount") or {}
                facilities_w.writerow(
                    {
                        "run_id": record.run_id,
                        "item_id": record.item_id,
                        "facility_name": f.get("facility_name"),
                        "facility_types": _csv_join(f.get("facility_types")),
                        "committed_amount_amount": committed.get("amount"),
                        "committed_amount_currency": committed.get("currency"),
                        "currency": f.get("currency"),
                        "maturity_date": f.get("maturity_date"),
                        "termination_date": f.get("termination_date"),
                        "source_refs": _csv_join(f.get("source_refs")),
                        "notes": f.get("notes"),
                    }
                )

            definition_sources = [
                ("pricing_metrics_recursive", _compiled_definitions_rows(record.pricing_metrics_recursive)),
                ("pricing_terms_recursive", _compiled_definitions_rows(record.pricing_blocking_recursive)),
                ("covenant_metrics_recursive", _compiled_definitions_rows(record.covenant_metrics_recursive)),
                ("covenant_terms_recursive", _compiled_definitions_rows(record.covenant_blocking_recursive)),
            ]
            for source, rows in definition_sources:
                for d in rows:
                    definitions_w.writerow(
                        {
                            "run_id": record.run_id,
                            "item_id": record.item_id,
                            "source": source,
                            "term": d.get("name") or d.get("metric_name"),
                            "term_type": d.get("term_type"),
                            "definition_text": d.get("definition_verbatim"),
                            "anchor_refs": _csv_join(d.get("source_refs")),
                            "notes": _csv_join(d.get("notes")),
                        }
                    )

            cov_doc = record.covenant_structured
            if isinstance(cov_doc, dict):
                for cov in cov_doc.get("covenants") or []:
                    if not isinstance(cov, dict):
                        continue
                    covenants_w.writerow(
                        {
                            "run_id": record.run_id,
                            "item_id": record.item_id,
                            "covenant_id": cov.get("covenant_id"),
                            "name": cov.get("name"),
                            "metric_refs": _csv_join(cov.get("metric_refs")),
                            "comparator": cov.get("comparator"),
                            "limit": cov.get("limit"),
                            "limit_text": cov.get("limit_text"),
                            "start_date": cov.get("start_date"),
                            "end_date": cov.get("end_date"),
                            "source_refs": _csv_join(cov.get("source_refs")),
                            "notes": cov.get("notes"),
                        }
                    )
                for m in cov_doc.get("metrics") or []:
                    if not isinstance(m, dict):
                        continue
                    covenant_metrics_w.writerow(
                        {
                            "run_id": record.run_id,
                            "item_id": record.item_id,
                            "metric_id": m.get("metric_id"),
                            "name": m.get("name"),
                            "contract_term": m.get("contract_term"),
                            "search_tokens": _csv_join(m.get("search_tokens")),
                            "source_refs": _csv_join(m.get("source_refs")),
                            "notes": m.get("notes"),
                        }
                    )

    if errors:
        err_path = out_dir / "errors.txt"
        err_path.write_text("\n".join(f"{item}: {msg}" for item, msg in errors), encoding="utf-8")
        raise RuntimeError(f"analysis-export-v2 completed with errors (count={len(errors)}); see {err_path}")

    manifest_path = paths.manifest_path
    if manifest_path.exists():
        update_manifest(
            manifest_path,
            analysis_export_v2_output_subdir=output_subdir,
            analysis_export_v2_agreement_metadata_subdir=agreement_metadata_subdir,
            analysis_export_v2_pricing_qa_subdir=pricing_qa_subdir,
            analysis_export_v2_covenant_qa_subdir=covenant_qa_subdir,
            analysis_export_v2_pricing_metrics_output_subdir=pricing_metrics_output_subdir,
            analysis_export_v2_pricing_blocking_output_subdir=pricing_blocking_output_subdir,
            analysis_export_v2_pricing_overlay_output_subdir=pricing_overlay_output_subdir,
            analysis_export_v2_covenant_metrics_output_subdir=covenant_metrics_output_subdir,
            analysis_export_v2_covenant_blocking_output_subdir=covenant_blocking_output_subdir,
            analysis_export_v2_covenant_overlay_output_subdir=covenant_overlay_output_subdir,
        )
