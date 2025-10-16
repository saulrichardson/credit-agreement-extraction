#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from edgar_filing_pipeline.extractor import extract_from_manifest
from edgar_filing_pipeline.metadata import MetadataStore


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract EDGAR segments described in a manifest.")
    parser.add_argument("--manifest", type=Path, required=True, help="Parquet/CSV manifest file")
    parser.add_argument("--tar-root", type=Path, required=True, help="Root directory containing tarballs (grouped by quarter/year)")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory to write segment HTML")
    parser.add_argument("--metadata-out", type=Path, help="Optional path to store extraction metadata (.parquet or .csv)")
    parser.add_argument("--convert-html", action="store_true", help="Convert segments to plain text instead of preserving HTML")
    parser.add_argument("--table-markers", action="store_true", help="When converting HTML, replace tables with markers")
    args = parser.parse_args()

    if args.manifest.suffix == ".parquet":
        manifest = pd.read_parquet(args.manifest)
    else:
        manifest = pd.read_csv(args.manifest)

    records = extract_from_manifest(
        manifest,
        tar_root=args.tar_root,
        output_dir=args.output_dir,
        convert_html=args.convert_html,
        replace_tables_with_markers=args.table_markers,
    )

    if args.metadata_out:
        store = MetadataStore(args.metadata_out)
        store.add_records(records)
        store.save()


if __name__ == "__main__":
    main()
