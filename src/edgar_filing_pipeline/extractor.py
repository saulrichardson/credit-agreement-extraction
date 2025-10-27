from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import pandas as pd

from .filters import FilterContext, FilterContextBuilder, FilterFn, default_filter
from .identifiers import SegmentKey
from .segment import SegmentExtractor, TarSegmentReader


def _build_output_name(
    file_name: str,
    segment_no: int,
    internal_id: Optional[int] = None,
    suffix: str = ".html",
) -> str:
    stem = file_name.replace(".nc", "")
    parts = [stem, f"seg{segment_no}"]
    if internal_id is not None:
        parts.append(f"id{internal_id}")
    return "_".join(parts) + suffix


def extract_from_manifest(
    manifest: pd.DataFrame,
    *,
    tar_root: Path | str,
    output_dir: Path | str,
    convert_html: bool = False,
    replace_tables_with_markers: bool = False,
    filter_fn: FilterFn | None = None,
    filter_context_builder: FilterContextBuilder | None = None,
) -> List[dict]:
    """Materialise the segments described in ``manifest`` to ``output_dir``.

    The manifest must contain at least ``tarfile``, ``file`` (or ``file_x``),
    and ``segment_no`` columns.  If ``internal_id`` is provided it will be
    encoded into the output filename and echoed in the returned metadata.

    A custom ``filter_fn`` may be supplied to decide whether individual
    segments should be skipped. The callable receives the SGML header,
    raw HTML, and a context dictionary (which always contains the
    manifest row under ``"manifest_row"``) and should return either
    ``None`` to keep the segment or a string reason explaining why it
    should be skipped. Use ``filter_context_builder`` to enrich the
    context with derived data (e.g. lookups from a metadata store).
    """

    tar_root = Path(tar_root)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    required_cols = {"tarfile"}
    file_col = None
    for candidate in ["file", "file_x"]:
        if candidate in manifest.columns:
            file_col = candidate
            break
    if file_col is None:
        raise KeyError("manifest must contain 'file' or 'file_x' column")
    required_cols.add(file_col)
    if "segment_no" not in manifest.columns and "segment_no_x" not in manifest.columns:
        raise KeyError("manifest must contain 'segment_no' (or segment_no_x)")

    seg_col = "segment_no" if "segment_no" in manifest.columns else "segment_no_x"

    readers: Dict[Path, TarSegmentReader] = {}
    extractors: Dict[tuple[Path, str], SegmentExtractor] = {}
    records: List[dict] = []

    active_filter = filter_fn or default_filter

    for row in manifest.itertuples(index=False):
        tar_path = tar_root / getattr(row, "tarfile")
        reader = readers.get(tar_path)
        if reader is None:
            reader = TarSegmentReader(tar_path)
            readers[tar_path] = reader

        file_name = getattr(row, file_col)
        segment_no = int(getattr(row, seg_col))
        internal_id = getattr(row, "internal_id", None)
        segment_index = segment_no - 1  # manifest is 1-based

        extractor_key = (tar_path, file_name)
        extractor = extractors.get(extractor_key)
        if extractor is None:
            extractor = SegmentExtractor(reader.read_member(file_name))
            extractors[extractor_key] = extractor

        header = extractor.get_segment_header(segment_index)
        raw_html = extractor.get_segment_html(segment_index)
        segment_key = SegmentKey(getattr(row, "tarfile"), file_name, segment_no)
        manifest_row = row._asdict() if hasattr(row, "_asdict") else row.__dict__
        manifest_row = dict(manifest_row)
        manifest_row.setdefault("segment_id", segment_key.id)
        manifest_row.setdefault("segment_digest", segment_key.digest())

        context: FilterContext = {
            "manifest_row": manifest_row,
            "segment_id": manifest_row["segment_id"],
        }
        if filter_context_builder:
            extra_context = filter_context_builder(manifest_row)
            if extra_context:
                context.update(extra_context)

        skip_reason = active_filter(header, raw_html, context)
        if skip_reason:
            records.append(
                {
                    "segment_id": segment_key.id,
                    "segment_digest": segment_key.digest(),
                    "tarfile": getattr(row, "tarfile"),
                    "file": file_name,
                    "segment_no": segment_no,
                    "internal_id": internal_id,
                    "output_path": None,
                    "has_table": extractor.has_tables(segment_index),
                    "skipped": True,
                    "skip_reason": skip_reason,
                }
            )
            continue

        if convert_html:
            html, table_dict = extractor.get_segment_text(
                segment_index,
                convert_html=True,
                replace_tables_with_markers=replace_tables_with_markers,
            )
            has_table = bool(table_dict)
        else:
            html = raw_html
            has_table = extractor.has_tables(segment_index)

        out_name = _build_output_name(file_name, segment_no, internal_id)
        out_path = output_dir / out_name
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(html, encoding="utf-8")

        records.append(
            {
                    "segment_id": segment_key.id,
                    "segment_digest": segment_key.digest(),
                    "tarfile": getattr(row, "tarfile"),
                "file": file_name,
                "segment_no": segment_no,
                "internal_id": internal_id,
                "output_path": str(out_path),
                "has_table": has_table,
                "skipped": False,
                "skip_reason": None,
            }
        )

    for reader in readers.values():
        reader.close()

    return records
