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


class AnalysisRecordV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["analysis_record_v1"]
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

    toc_v1_path: str | None = None
    toc_v1: dict[str, Any] | None = None

    agreement_metadata_path: str | None = None
    agreement_metadata: dict[str, Any] | None = None
    agreement_metadata_meta_path: str | None = None
    agreement_metadata_meta: dict[str, Any] | None = None

    definitions_v2_dir: str | None = None
    definitions_v2: list[dict[str, Any]] = Field(default_factory=list)

    errors: list[str] = Field(default_factory=list)


def _rel(path: Path) -> str:
    try:
        return str(path.relative_to(Path(".")))
    except Exception:
        return str(path)


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Invalid JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError(f"Expected JSON object in {path}, got {type(payload).__name__}")
    return payload


def _definition_files(definitions_dir: Path, item_id: str) -> list[Path]:
    return sorted(definitions_dir.glob(f"{item_id}__*__definition.json"))


def _csv_join(values: Any) -> str:
    if values is None:
        return ""
    if isinstance(values, (list, tuple)):
        return ",".join(str(v) for v in values if v is not None)
    return str(values)


def run_analysis_export_v1(
    paths: Paths,
    item_ids: Iterable[str],
    *,
    agreement_metadata_subdir: str,
    definitions_v2_subdir: str,
    output_subdir: str = "analysis_export_v1",
) -> None:
    """Export analysis-ready JSON artifacts per item (single record per agreement).

    This stage is strict and does not silently skip missing artifacts:
    - Writes per-item JSON and a JSONL bundle
    - Writes errors.txt and raises if any item is missing required inputs
    """

    manifest = load_manifest(paths.manifest_path)
    item_rows = {row["item_id"]: row for row in manifest_items(manifest)}
    toc_subdir = manifest.get("toc_v1_output_subdir") if isinstance(manifest, dict) else None
    if isinstance(toc_subdir, str) and not toc_subdir.strip():
        toc_subdir = None

    toc_item_ids: set[str] | None = None
    if toc_subdir and isinstance(manifest, dict):
        raw_ids = manifest.get("toc_v1_item_ids")
        if raw_ids is None:
            toc_item_ids = None
        elif isinstance(raw_ids, list) and all(isinstance(v, str) for v in raw_ids):
            toc_item_ids = {v for v in raw_ids if v.strip()}
        else:
            raise RuntimeError("manifest toc_v1_item_ids must be a list of strings when provided")

    out_root = paths.run_dir / "analysis_export"
    out_dir = out_root / output_subdir
    out_dir.mkdir(parents=True, exist_ok=True)

    bundle_path = out_dir / "records.jsonl"
    errors: list[tuple[str, str]] = []

    definitions_dir = paths.run_dir / "definitions_v2" / definitions_v2_subdir
    if not definitions_dir.exists():
        raise FileNotFoundError(
            f"Definitions v2 directory not found: expected {definitions_dir} (set --definitions-v2-subdir)"
        )

    metadata_dir = paths.run_dir / "agreement_metadata" / agreement_metadata_subdir
    if not metadata_dir.exists():
        raise FileNotFoundError(
            f"Agreement metadata directory not found: expected {metadata_dir} (set --agreement-metadata-subdir)"
        )

    # Write a small meta manifest for reproducibility.
    record_manifest(
        out_dir / "export.meta.json",
        {
            "schema_version": "analysis_export_v1_meta",
            "run_id": paths.run_id,
            "created_at": int(time.time()),
            "agreement_metadata_subdir": agreement_metadata_subdir,
            "definitions_v2_subdir": definitions_v2_subdir,
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
        toc_chunks_f = stack.enter_context((tables_dir / "toc_chunks.csv").open("w", newline=""))

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
            fieldnames=["run_id", "item_id", "term", "term_type", "definition_text", "anchor_refs", "notes"],
        )
        toc_chunks_w = csv.DictWriter(
            toc_chunks_f,
            fieldnames=["run_id", "item_id", "chunk_id", "start_anchor", "end_anchor", "title", "topics"],
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
            toc_chunks_w,
        ):
            w.writeheader()

        for item_id in item_ids:
            created_at = int(time.time())
            row = item_rows.get(item_id) or {}
            record = AnalysisRecordV1(
                schema_version="analysis_record_v1",
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

            if toc_subdir:
                toc_path = paths.run_dir / "toc_v1" / toc_subdir / f"{item_id}.json"
                record.toc_v1_path = _rel(toc_path) if toc_path.exists() else None
                if toc_path.exists():
                    try:
                        record.toc_v1 = _load_json(toc_path)
                    except Exception as exc:
                        record.errors.append(f"toc_v1 invalid: {exc}")
                else:
                    toc_expected = toc_item_ids is None or item_id in toc_item_ids
                    if toc_expected:
                        record.errors.append(f"missing toc_v1: {toc_path}")

            metadata_path = metadata_dir / f"{item_id}.json"
            record.agreement_metadata_path = _rel(metadata_path) if metadata_path.exists() else None
            if metadata_path.exists():
                try:
                    record.agreement_metadata = _load_json(metadata_path)
                except Exception as exc:
                    record.errors.append(f"agreement_metadata invalid: {exc}")
            else:
                record.errors.append(f"missing agreement_metadata: {metadata_path}")

            metadata_meta_path = metadata_dir / f"{item_id}.meta.json"
            record.agreement_metadata_meta_path = _rel(metadata_meta_path) if metadata_meta_path.exists() else None
            if metadata_meta_path.exists():
                try:
                    record.agreement_metadata_meta = _load_json(metadata_meta_path)
                except Exception as exc:
                    record.errors.append(f"agreement_metadata meta invalid: {exc}")

            record.definitions_v2_dir = _rel(definitions_dir)
            defs: list[dict[str, Any]] = []
            for def_path in _definition_files(definitions_dir, item_id):
                try:
                    defs.append(_load_json(def_path))
                except Exception as exc:
                    record.errors.append(f"definitions_v2 invalid: {def_path}: {exc}")
            record.definitions_v2 = defs

            # Persist per-item record and append to JSONL bundle.
            item_out = out_dir / f"{item_id}.json"
            item_out.write_text(record.model_dump_json(indent=2))
            bundle.write(record.model_dump_json() + "\n")

            if record.errors:
                errors.append((item_id, "; ".join(record.errors)))
            else:
                # Flatten selected outputs into analysis-friendly CSV tables.
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
                for d in record.definitions_v2:
                    definitions_w.writerow(
                        {
                            "run_id": record.run_id,
                            "item_id": record.item_id,
                            "term": d.get("term"),
                            "term_type": d.get("term_type"),
                            "definition_text": d.get("definition_text"),
                            "anchor_refs": _csv_join(d.get("anchor_refs")),
                            "notes": d.get("notes"),
                        }
                    )

                if isinstance(record.toc_v1, dict):
                    for ch in record.toc_v1.get("chunks") or []:
                        if not isinstance(ch, dict):
                            continue
                        toc_chunks_w.writerow(
                            {
                                "run_id": record.run_id,
                                "item_id": record.item_id,
                                "chunk_id": ch.get("chunk_id"),
                                "start_anchor": ch.get("start_anchor"),
                                "end_anchor": ch.get("end_anchor"),
                                "title": ch.get("title"),
                                "topics": _csv_join(ch.get("topics")),
                            }
                        )

    if errors:
        err_path = out_dir / "errors.txt"
        err_path.write_text("\n".join(f"{item}: {msg}" for item, msg in errors))
        raise RuntimeError(f"analysis-export-v1 completed with errors (count={len(errors)}); see {err_path}")

    manifest_path = paths.manifest_path
    if manifest_path.exists():
        update_manifest(
            manifest_path,
            analysis_export_output_subdir=output_subdir,
            analysis_export_agreement_metadata_subdir=agreement_metadata_subdir,
            analysis_export_definitions_v2_subdir=definitions_v2_subdir,
        )
