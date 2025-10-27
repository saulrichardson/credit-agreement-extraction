from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from .extractor import extract_from_manifest
from .filters import FilterContextBuilder, FilterFn
from .manifest import build_manifest
from .metadata import MetadataStore
from .processing import (
    NormalizedSegment,
    build_checksum,
    normalize_html,
    tables_to_json_dict,
)
from .segment import SegmentExtractor, TarSegmentReader
from .identifiers import SegmentKey


def read_manifest(path: Path) -> pd.DataFrame:
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    if path.suffix == ".csv":
        return pd.read_csv(path)
    raise ValueError("Manifest must be parquet or csv.")


def normalize_from_manifest(
    manifest: pd.DataFrame,
    *,
    tar_root: Path,
    limit: int | None = None,
) -> pd.DataFrame:
    tar_root = Path(tar_root)

    readers: Dict[Path, TarSegmentReader] = {}
    extractors: Dict[tuple[Path, str], SegmentExtractor] = {}
    records: List[dict] = []

    processed = 0

    for row in manifest.itertuples(index=False):
        tarfile_name = getattr(row, "tarfile")
        tar_path = tar_root / tarfile_name

        reader = readers.get(tar_path)
        if reader is None:
            reader = TarSegmentReader(tar_path)
            readers[tar_path] = reader

        file_name = getattr(row, "file")
        segment_no = int(getattr(row, "segment_no"))
        segment_index = segment_no - 1
        segment_key = SegmentKey(tarfile_name, file_name, segment_no)

        extractor_key = (tar_path, file_name)
        extractor = extractors.get(extractor_key)
        if extractor is None:
            extractor = SegmentExtractor(reader.read_member(file_name))
            extractors[extractor_key] = extractor

        html = extractor.get_segment_html(segment_index)
        normalized: NormalizedSegment = normalize_html(html)

        markdown_json, html_json = tables_to_json_dict(normalized.tables)
        checksum = build_checksum(normalized.text)

        records.append(
            {
                "segment_id": segment_key.id,
                "segment_digest": segment_key.digest(),
                "tarfile": tarfile_name,
                "file": file_name,
                "segment_no": segment_no,
                "doc_type": getattr(row, "doc_type", None),
                "header_json": getattr(row, "header_json", None),
                "text": normalized.text,
                "tables_markdown": markdown_json,
                "tables_html": html_json,
                "num_tables": len(normalized.tables.markers),
                "encoding": reader.last_encoding,
                "checksum_text": checksum,
                "num_chars": len(normalized.text),
            }
        )

        processed += 1
        if limit and processed >= limit:
            break

    for reader in readers.values():
        reader.close()

    return pd.DataFrame.from_records(records)


@dataclass
class PipelineResult:
    manifest_path: Path
    extraction_records: List[dict]
    normalized_path: Optional[Path]


def build_manifest_to_path(
    *,
    tar_root: Path,
    output_path: Path,
    pattern: str | None = None,
    limit: int | None = None,
) -> pd.DataFrame:
    df = build_manifest(tar_root, pattern=pattern, limit=limit)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.suffix == ".parquet":
        df.to_parquet(output_path, index=False)
    else:
        df.to_csv(output_path, index=False)
    return df


def extract_segments_to_dir(
    *,
    manifest: pd.DataFrame,
    tar_root: Path,
    output_dir: Path,
    metadata_out: Path | None = None,
    convert_html: bool = False,
    table_markers: bool = False,
    filter_fn: FilterFn | None = None,
    filter_context_builder: FilterContextBuilder | None = None,
) -> List[dict]:
    records = extract_from_manifest(
        manifest,
        tar_root=tar_root,
        output_dir=output_dir,
        convert_html=convert_html,
        replace_tables_with_markers=table_markers,
        filter_fn=filter_fn,
        filter_context_builder=filter_context_builder,
    )
    if metadata_out:
        store = MetadataStore(metadata_out)
        store.add_records(records)
        store.save()
    return records


def run_pipeline(
    *,
    tar_root: Path,
    manifest_path: Path,
    extract_dir: Path,
    normalized_output: Optional[Path] = None,
    pattern: str | None = None,
    limit: int | None = None,
    metadata_out: Path | None = None,
    convert_html: bool = False,
    table_markers: bool = False,
    filter_fn: FilterFn | None = None,
    filter_context_builder: FilterContextBuilder | None = None,
) -> PipelineResult:
    manifest = build_manifest_to_path(
        tar_root=tar_root,
        output_path=manifest_path,
        pattern=pattern,
        limit=limit,
    )
    extraction_records = extract_segments_to_dir(
        manifest=manifest,
        tar_root=tar_root,
        output_dir=extract_dir,
        metadata_out=metadata_out,
        convert_html=convert_html,
        table_markers=table_markers,
        filter_fn=filter_fn,
        filter_context_builder=filter_context_builder,
    )
    normalized_path = None
    if normalized_output:
        normalized_df = normalize_from_manifest(
            manifest,
            tar_root=tar_root,
        )
        normalized_output.parent.mkdir(parents=True, exist_ok=True)
        normalized_df.to_parquet(normalized_output, index=False)
        normalized_path = normalized_output

    return PipelineResult(
        manifest_path=manifest_path,
        extraction_records=extraction_records,
        normalized_path=normalized_path,
    )
