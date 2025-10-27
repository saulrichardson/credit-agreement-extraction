from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Iterable

import pandas as pd

from .filters import (
    detect_binary_by_doc_type,
    detect_binary_by_header_filename,
    extract_doc_type,
)
from .identifiers import SegmentKey
from .segment import SegmentExtractor, TarSegmentReader


def iter_tarfiles(tar_root: Path, pattern: str | None) -> Iterable[Path]:
    yield from sorted(
        p for p in tar_root.glob("**/*.tar.gz") if pattern is None or pattern in p.name
    )


def build_manifest(
    tar_root: Path,
    *,
    pattern: str | None = None,
    limit: int | None = None,
) -> pd.DataFrame:
    records: list[dict] = []
    count = 0
    skipped_counts: Counter[str] = Counter()

    for tar_path in iter_tarfiles(tar_root, pattern):
        reader = TarSegmentReader(tar_path)
        for member_name in reader.list_members():
            if not member_name.endswith(".nc"):
                continue
            extractor = SegmentExtractor(reader.read_member(member_name))
            for idx in range(len(extractor)):
                header = extractor.get_segment_header(idx)
                html = extractor.get_segment_html(idx)
                doc_type = extract_doc_type(header)
                skip_reason = detect_binary_by_doc_type(doc_type)
                if not skip_reason:
                    skip_reason = detect_binary_by_header_filename(header)
                if skip_reason:
                    skipped_counts[skip_reason] += 1
                    continue
                segment_no = idx + 1
                segment_key = SegmentKey(tar_path.name, member_name, segment_no)
                records.append(
                    {
                        "segment_id": segment_key.id,
                        "segment_digest": segment_key.digest(),
                        "tarfile": tar_path.name,
                        "file": member_name,
                        "segment_no": segment_no,
                        "doc_type": doc_type,
                        "has_table": extractor.has_tables(idx),
                        "num_chars": len(html),
                        "header_json": json.dumps(header),
                    }
                )
                count += 1
                if limit and count >= limit:
                    reader.close()
                    df = pd.DataFrame(records)
                    if skipped_counts:
                        df.attrs["binary_segments_skipped"] = dict(skipped_counts)
                    return df
        reader.close()
    df = pd.DataFrame(records)
    if skipped_counts:
        df.attrs["binary_segments_skipped"] = dict(skipped_counts)
    return df
