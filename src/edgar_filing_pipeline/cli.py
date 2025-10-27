from __future__ import annotations

import argparse
from importlib import import_module
from pathlib import Path
from typing import Any, Callable, List, Optional

from .filters import FilterContextBuilder, FilterFn
from .workflow import (
    build_manifest_to_path,
    extract_segments_to_dir,
    normalize_from_manifest,
    read_manifest,
    run_pipeline,
)


def _load_callable(spec: str) -> Callable[..., Any]:
    module_path, sep, attr = spec.rpartition(":")
    if not sep:
        module_path, sep, attr = spec.rpartition(".")
    if not sep:
        raise ValueError(f"Invalid callable reference '{spec}'. Use module:attribute.")
    module = import_module(module_path)
    try:
        func = getattr(module, attr)
    except AttributeError as exc:  # pragma: no cover - defensive branch
        raise ValueError(f"Callable '{attr}' not found in module '{module_path}'.") from exc
    if not callable(func):
        raise ValueError(f"Object '{spec}' is not callable.")
    return func


def _resolve_filter_functions(
    filter_path: Optional[str],
    context_path: Optional[str],
) -> tuple[Optional[FilterFn], Optional[FilterContextBuilder]]:
    filter_fn: Optional[FilterFn] = None
    context_builder: Optional[FilterContextBuilder] = None
    if filter_path:
        filter_fn = _load_callable(filter_path)
    if context_path:
        context_builder = _load_callable(context_path)
    return filter_fn, context_builder


def _cmd_build_manifest(args: argparse.Namespace) -> None:
    manifest = build_manifest_to_path(
        tar_root=Path(args.tar_root),
        output_path=Path(args.output),
        pattern=args.pattern,
        limit=args.limit,
    )
    skipped_summary = manifest.attrs.get("binary_segments_skipped")
    if skipped_summary:
        total = sum(skipped_summary.values())
        fragments = ", ".join(
            f"{reason}={count}" for reason, count in sorted(skipped_summary.items())
        )
        print(f"Skipped {total} binary segments ({fragments})")
    print(f"Wrote manifest with {len(manifest)} rows to {args.output}")


def _cmd_extract(args: argparse.Namespace) -> None:
    manifest = read_manifest(Path(args.manifest))
    filter_fn, context_builder = _resolve_filter_functions(
        args.filter, args.filter_context
    )
    records = extract_segments_to_dir(
        manifest=manifest,
        tar_root=Path(args.tar_root),
        output_dir=Path(args.output_dir),
        metadata_out=Path(args.metadata_out) if args.metadata_out else None,
        convert_html=args.convert_html,
        table_markers=args.table_markers,
        filter_fn=filter_fn,
        filter_context_builder=context_builder,
    )
    skipped = sum(1 for record in records if record["skipped"])
    extracted = len(records) - skipped
    print(
        f"Extraction complete: {extracted} written, {skipped} skipped "
        f"(see metadata for details)."
    )


def _cmd_normalize(args: argparse.Namespace) -> None:
    manifest = read_manifest(Path(args.manifest))
    normalized_df = normalize_from_manifest(
        manifest,
        tar_root=Path(args.tar_root),
        limit=args.limit,
    )
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    normalized_df.to_parquet(output_path, index=False)
    print(
        f"Normalized {len(normalized_df)} segments -> {output_path} "
        f"(tables={int(normalized_df['num_tables'].sum())})."
    )


def _cmd_run(args: argparse.Namespace) -> None:
    filter_fn, context_builder = _resolve_filter_functions(
        args.filter, args.filter_context
    )
    result = run_pipeline(
        tar_root=Path(args.tar_root),
        manifest_path=Path(args.manifest_out),
        extract_dir=Path(args.extract_dir),
        normalized_output=Path(args.normalized_out) if args.normalized_out else None,
        pattern=args.pattern,
        limit=args.limit,
        metadata_out=Path(args.metadata_out) if args.metadata_out else None,
        convert_html=args.convert_html,
        table_markers=args.table_markers,
        filter_fn=filter_fn,
        filter_context_builder=context_builder,
    )
    print(f"Manifest written to {result.manifest_path}")
    print(f"Extracted {len(result.extraction_records)} segments into {args.extract_dir}")
    if result.normalized_path:
        print(f"Normalized output: {result.normalized_path}")


def _add_filter_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--filter",
        help="Optional callable (module:func) that decides whether to skip segments.",
    )
    parser.add_argument(
        "--filter-context",
        help="Optional callable (module:func) that receives the manifest row and returns "
        "extra context for the filter.",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="EDGAR filing pipeline CLI (manifest, extract, normalize)."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    build = subparsers.add_parser("build-manifest", help="Scan tarballs and write a manifest.")
    build.add_argument("--tar-root", required=True, type=Path)
    build.add_argument("--output", required=True, type=Path)
    build.add_argument("--pattern", type=str)
    build.add_argument("--limit", type=int)
    build.set_defaults(func=_cmd_build_manifest)

    extract = subparsers.add_parser("extract", help="Materialize manifest segments to disk.")
    extract.add_argument("--manifest", required=True, type=Path)
    extract.add_argument("--tar-root", required=True, type=Path)
    extract.add_argument("--output-dir", required=True, type=Path)
    extract.add_argument("--metadata-out", type=Path)
    extract.add_argument("--convert-html", action="store_true")
    extract.add_argument("--table-markers", action="store_true")
    _add_filter_args(extract)
    extract.set_defaults(func=_cmd_extract)

    normalize = subparsers.add_parser(
        "normalize", help="Normalize manifest segments into text + markdown tables."
    )
    normalize.add_argument("--manifest", required=True, type=Path)
    normalize.add_argument("--tar-root", required=True, type=Path)
    normalize.add_argument("--output", required=True, type=Path)
    normalize.add_argument("--limit", type=int)
    normalize.set_defaults(func=_cmd_normalize)

    run = subparsers.add_parser("run", help="Run manifest, extraction, and optional normalization.")
    run.add_argument("--tar-root", required=True, type=Path)
    run.add_argument("--manifest-out", required=True, type=Path)
    run.add_argument("--extract-dir", required=True, type=Path)
    run.add_argument("--normalized-out", type=Path)
    run.add_argument("--pattern", type=str)
    run.add_argument("--limit", type=int)
    run.add_argument("--metadata-out", type=Path)
    run.add_argument("--convert-html", action="store_true")
    run.add_argument("--table-markers", action="store_true")
    _add_filter_args(run)
    run.set_defaults(func=_cmd_run)

    return parser


def main(argv: Optional[List[str]] = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        args.func(args)
    except Exception as exc:  # noqa: BLE001
        parser.exit(1, f"Error: {exc}\n")
