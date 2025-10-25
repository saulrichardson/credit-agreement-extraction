#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Iterable, List

import pandas as pd

from edgar_filing_pipeline.filters import (
    detect_binary_by_doc_type,
    detect_binary_by_header_filename,
    extract_doc_type,
)
from edgar_filing_pipeline.segment import SegmentExtractor, TarSegmentReader


def iter_tarfiles(tar_root: Path, pattern: str | None) -> Iterable[Path]:
    yield from sorted(
        p
        for p in tar_root.glob("**/*.tar.gz")
        if pattern is None or pattern in p.name
    )


def build_manifest(
    tar_root: Path,
    *,
    pattern: str | None = None,
    limit: int | None = None,
) -> pd.DataFrame:
    records: List[dict] = []
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
                records.append(
                    {
                        "tarfile": tar_path.name,
                        "file": member_name,
                        "segment_no": idx + 1,
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a segment-level manifest for EDGAR tar files.")
    parser.add_argument("--tar-root", type=Path, required=True, help="Directory containing .nc.tar.gz archives")
    parser.add_argument("--output", type=Path, required=True, help="Destination .parquet or .csv file")
    parser.add_argument("--pattern", type=str, help="Only process tarfiles whose name contains this substring")
    parser.add_argument("--limit", type=int, help="Maximum number of segments to record (for testing)")
    args = parser.parse_args()

    manifest = build_manifest(args.tar_root, pattern=args.pattern, limit=args.limit)
    skipped_summary = manifest.attrs.get("binary_segments_skipped")
    if skipped_summary:
        total = sum(skipped_summary.values())
        reason_fragments = ", ".join(f"{reason}={count}" for reason, count in sorted(skipped_summary.items()))
        print(f"Skipped {total} binary segments ({reason_fragments})")
    if args.output.suffix == ".parquet":
        manifest.to_parquet(args.output, index=False)
    else:
        manifest.to_csv(args.output, index=False)


if __name__ == "__main__":
    main()
