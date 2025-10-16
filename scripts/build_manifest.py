#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, List

import pandas as pd

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

    for tar_path in iter_tarfiles(tar_root, pattern):
        reader = TarSegmentReader(tar_path)
        for member_name in reader.list_members():
            if not member_name.endswith(".nc"):
                continue
            extractor = SegmentExtractor(reader.read_member(member_name))
            for idx in range(len(extractor)):
                header = extractor.get_segment_header(idx)
                html = extractor.get_segment_html(idx)
                records.append(
                    {
                        "tarfile": tar_path.name,
                        "file": member_name,
                        "segment_no": idx + 1,
                        "doc_type": header.get("doc_type"),
                        "has_table": extractor.has_tables(idx),
                        "num_chars": len(html),
                        "header_json": json.dumps(header),
                    }
                )
                count += 1
                if limit and count >= limit:
                    reader.close()
                    return pd.DataFrame(records)
        reader.close()
    return pd.DataFrame(records)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a segment-level manifest for EDGAR tar files.")
    parser.add_argument("--tar-root", type=Path, required=True, help="Directory containing .nc.tar.gz archives")
    parser.add_argument("--output", type=Path, required=True, help="Destination .parquet or .csv file")
    parser.add_argument("--pattern", type=str, help="Only process tarfiles whose name contains this substring")
    parser.add_argument("--limit", type=int, help="Maximum number of segments to record (for testing)")
    args = parser.parse_args()

    manifest = build_manifest(args.tar_root, pattern=args.pattern, limit=args.limit)
    if args.output.suffix == ".parquet":
        manifest.to_parquet(args.output, index=False)
    else:
        manifest.to_csv(args.output, index=False)


if __name__ == "__main__":
    main()
