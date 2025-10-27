#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from edgar_filing_pipeline.manifest import build_manifest


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
