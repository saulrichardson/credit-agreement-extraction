from __future__ import annotations

import argparse
import json
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
from .planner import (
    PlannerConfig,
    SemanticPlanner,
    load_anchors_path as _planner_load_anchors_path,
    load_prompt_view,
)
from .plan_validation import load_sentence_anchor_ids, validate_plan
from .segment_scorer import ScoringConfig, SegmentScorer
from .chunking import build_chunks
from .anchoring import (
    CanonicalDocumentBuilder,
    build_prompt_view,
    write_canonical_bundle,
    write_prompt_view,
    MachineContentError,
)
from .prompt_runner import PromptRunner, PromptRunnerConfig
from .reverse_highlighter import (
    SupportMapper,
    split_prose,
    split_code,
    parse_prompt_view,
    build_html,
)
from .hot_zone import build_hot_zone


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


def _cmd_anchor_document(args: argparse.Namespace) -> None:
    input_path = Path(args.input_html)
    if not input_path.exists():
        raise FileNotFoundError(f"{input_path} does not exist")
    html_text = input_path.read_text(encoding="utf-8")
    try:
        builder = CanonicalDocumentBuilder(html_text=html_text)
    except MachineContentError as exc:
        print(f"Skipping {input_path}: {exc}")
        return
    canonical_document = builder.build()
    output_dir = Path(args.output_dir)
    if output_dir.exists():
        raise FileExistsError(f"{output_dir} already exists; refusing to overwrite.")
    write_canonical_bundle(canonical_document=canonical_document, output_dir=output_dir)
    prompt_view = build_prompt_view(canonical_document)
    write_prompt_view(prompt_view=prompt_view, output_dir=output_dir)
    print(f"Canonical document written to {output_dir}")


def _cmd_plan_document(args: argparse.Namespace) -> None:
    bundle_dir = Path(args.canonical_dir)
    prompt_view = load_prompt_view(bundle_dir)
    anchors_path = (
        Path(args.anchors)
        if args.anchors
        else _planner_load_anchors_path(bundle_dir)
    )

    config = PlannerConfig(
        model=args.model,
        reasoning_effort=args.reasoning,
        max_attempts=args.max_attempts,
        focus_hint=args.focus_hint,
    )
    planner = SemanticPlanner(config=config)
    plan = planner.build_plan(prompt_view)

    output_path = Path(args.output) if args.output else bundle_dir / "planner_semantic_index.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(plan, indent=2))
    print(f"Planner output written to {output_path}")

    if args.skip_validation:
        return

    sentence_ids = load_sentence_anchor_ids(anchors_path)
    errors = validate_plan(plan, sentence_ids)
    if errors:
        print("Validation failed:")
        for err in errors:
            print(f" - {err}")
        raise SystemExit(1)
    print(f"Plan validated against {anchors_path}")


def _cmd_score_plan(args: argparse.Namespace) -> None:
    plan_path = Path(args.plan)
    plan = json.loads(plan_path.read_text())
    config = ScoringConfig(
        model=args.model,
        reasoning_effort=args.reasoning,
        max_attempts=args.max_attempts,
    )
    scorer = SegmentScorer(config=config)
    results = scorer.score_segments(plan, args.question)
    output = {
        "question": args.question,
        "plan_path": str(plan_path),
        "model": args.model,
        "scores": [
            {
                "seg_id": item.seg_id,
                "name": item.name,
                "range": item.range,
                "tags": item.tags,
                "score": item.score,
                "verdict": item.verdict,
                "rationale": item.rationale,
            }
            for item in results
        ],
    }
    output_path = Path(args.output) if args.output else plan_path.with_name(plan_path.stem + "_scores.json")
    output_path.write_text(json.dumps(output, indent=2))
    print(f"Wrote scores for {len(results)} segments -> {output_path}")
    if results:
        print("Top segments:")
        for item in results[: min(3, len(results))]:
            print(
                f" - {item.seg_id} ({item.name}): score={item.score} "
                f"verdict={item.verdict}"
            )


def _cmd_chunk_document(args: argparse.Namespace) -> None:
    canonical_dir = Path(args.canonical_dir)
    output_path = Path(args.output)
    chunk_data = build_chunks(
        canonical_dir,
        chunk_size=args.chunk_size,
        stride=args.stride,
        max_snippet_chars=args.max_snippet_chars,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(chunk_data, indent=2))
    print(
        f"Wrote {len(chunk_data['segments'])} chunks "
        f"(size={args.chunk_size}, stride={args.stride}) -> {output_path}"
    )


def _cmd_build_hot_zone(args: argparse.Namespace) -> None:
    build_hot_zone(
        scores_path=Path(args.scores),
        prompt_view_path=Path(args.prompt_view),
        output_path=Path(args.output),
        threshold=args.threshold,
        verdict=args.verdict,
    )
    print(
        f"Hot zone written to {args.output} "
        f"(threshold={args.threshold}, verdict={args.verdict or 'any'})"
    )


def _cmd_run_prompt(args: argparse.Namespace) -> None:
    prompt_path = Path(args.prompt)
    source_path = Path(args.source)
    rendered_prompt = PromptRunner.render_prompt(prompt_path, source_path)
    config = PromptRunnerConfig(
        model=args.model,
        reasoning_effort=args.reasoning,
        max_attempts=args.max_attempts,
        system_prompt=args.system_prompt,
    )
    runner = PromptRunner(config=config)
    response = runner.run_prompt(rendered_prompt)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(response + "\n", encoding="utf-8")
    print(f"Prompt response written to {output_path}")


def _cmd_reverse_highlight(args: argparse.Namespace) -> None:
    answer_text = Path(args.answer).read_text()
    source_text = Path(args.source).read_text()
    claims = split_prose(answer_text) if args.mode == "prose" else split_code(answer_text)
    mapper = SupportMapper(
        model=args.model,
        reasoning=args.reasoning,
        max_attempts=args.max_attempts,
    )
    results = [mapper.map_claim(claim, source_text) for claim in claims]

    payload = {
        "answer_path": str(args.answer),
        "source_path": str(args.source),
        "mode": args.mode,
        "claims": [
            {
                "index": item.claim.index,
                "text": item.claim.text,
                "status": item.status,
                "anchors": item.anchors,
                "rationale": item.rationale,
            }
            for item in results
        ],
    }
    json_path = Path(args.output_json)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(payload, indent=2))

    anchor_map = parse_prompt_view(Path(args.prompt_view))
    html = build_html(results, anchor_map)
    html_path = Path(args.output_html)
    html_path.parent.mkdir(parents=True, exist_ok=True)
    html_path.write_text(html)
    print(
        f"Reverse highlight outputs written to {json_path} "
        f"and {html_path}"
    )


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

    anchor_doc = subparsers.add_parser(
        "anchor-document",
        help="Generate canonical text and anchors for a single HTML file.",
    )
    anchor_doc.add_argument("--input-html", required=True, type=Path)
    anchor_doc.add_argument("--output-dir", required=True, type=Path)
    anchor_doc.set_defaults(func=_cmd_anchor_document)

    plan = subparsers.add_parser(
        "plan-document",
        help="Run the semantic planner against a canonical bundle and validate the JSON output.",
    )
    plan.add_argument(
        "--canonical-dir",
        required=True,
        type=Path,
        help="Directory containing canonical.txt, anchors.tsv, prompt_view.txt",
    )
    plan.add_argument(
        "--output",
        type=Path,
        help="Path to write the planner JSON (default: canonical-dir/plan.json)",
    )
    plan.add_argument(
        "--focus-hint",
        default="pricing mechanics and covenants",
        help="Optional focus hint passed to the LLM planner.",
    )
    plan.add_argument("--model", default="gpt-5-nano")
    plan.add_argument("--reasoning", default="medium")
    plan.add_argument("--max-attempts", type=int, default=3)
    plan.add_argument(
        "--anchors",
        type=Path,
        help="Optional anchors.tsv path (defaults to canonical-dir/anchors.tsv)",
    )
    plan.add_argument(
        "--skip-validation",
        action="store_true",
        help="Skip validation (not recommended)",
    )
    plan.set_defaults(func=_cmd_plan_document)

    score = subparsers.add_parser(
        "score-plan",
        help="Score planner segments for relevance to a specific question (LLM-backed).",
    )
    score.add_argument("--plan", required=True, type=Path, help="Planner JSON file.")
    score.add_argument(
        "--question",
        required=False,
        default="Does this segment help us fully understand the pricing the company will face?",
        help="Focus question (defaults to the pricing-structure prompt).",
    )
    score.add_argument("--output", type=Path, help="Where to write the score JSON (defaults to plan_scores.json).")
    score.add_argument("--model", default="gpt-5-nano")
    score.add_argument("--reasoning", default="medium")
    score.add_argument("--max-attempts", type=int, default=3)
    score.set_defaults(func=_cmd_score_plan)

    chunk = subparsers.add_parser(
        "chunk-document",
        help="Create fixed-size sentence-anchor chunks from a canonical bundle.",
    )
    chunk.add_argument("--canonical-dir", required=True, type=Path)
    chunk.add_argument("--output", required=True, type=Path)
    chunk.add_argument("--chunk-size", type=int, default=40)
    chunk.add_argument("--stride", type=int, default=20)
    chunk.add_argument("--max-snippet-chars", type=int, default=1200)
    chunk.set_defaults(func=_cmd_chunk_document)

    hot_zone = subparsers.add_parser(
        "build-hot-zone",
        help="Aggregate high-scoring segments into a single prompt-ready artifact.",
    )
    hot_zone.add_argument("--scores", required=True, type=Path, help="score-plan JSON file")
    hot_zone.add_argument(
        "--prompt-view",
        required=True,
        type=Path,
        help="prompt_view.txt containing inline anchors",
    )
    hot_zone.add_argument("--output", required=True, type=Path)
    hot_zone.add_argument(
        "--threshold",
        type=float,
        default=1.0,
        help="Minimum score required to include a segment (default: 1.0)",
    )
    hot_zone.add_argument(
        "--verdict",
        help="Optional verdict filter (e.g., include). Omit to allow any verdict.",
    )
    hot_zone.set_defaults(func=_cmd_build_hot_zone)

    run_prompt = subparsers.add_parser(
        "run-prompt",
        help="Render a prompt template with source text and call the OpenAI Responses API.",
    )
    run_prompt.add_argument("--prompt", required=True, type=Path, help="Prompt template file with {{SOURCE_TEXT}} placeholder")
    run_prompt.add_argument("--source", required=True, type=Path, help="Source text to inject into the template")
    run_prompt.add_argument("--output", required=True, type=Path, help="Where to write the model response")
    run_prompt.add_argument("--system-prompt", default="You are a careful assistant. Follow the user instructions exactly.")
    run_prompt.add_argument("--model", default="gpt-5-nano")
    run_prompt.add_argument("--reasoning", default="medium")
    run_prompt.add_argument("--max-attempts", type=int, default=3)
    run_prompt.set_defaults(func=_cmd_run_prompt)

    reverse = subparsers.add_parser(
        "reverse-highlight",
        help="Split an answer into claims, map them to anchors, and emit highlight artifacts.",
    )
    reverse.add_argument("--answer", required=True, type=Path, help="Answer text (prose or code)")
    reverse.add_argument(
        "--source",
        required=True,
        type=Path,
        help="Source text with inline anchors (e.g., chunks_score_1.txt)",
    )
    reverse.add_argument(
        "--prompt-view",
        required=True,
        type=Path,
        help="prompt_view.txt for retrieving anchor snippets",
    )
    reverse.add_argument("--mode", choices=["prose", "code"], default="prose")
    reverse.add_argument("--output-json", required=True, type=Path)
    reverse.add_argument("--output-html", required=True, type=Path)
    reverse.add_argument("--model", default="gpt-5-nano")
    reverse.add_argument("--reasoning", default="medium")
    reverse.add_argument("--max-attempts", type=int, default=3)
    reverse.set_defaults(func=_cmd_reverse_highlight)

    return parser


def main(argv: Optional[List[str]] = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        args.func(args)
    except Exception as exc:  # noqa: BLE001
        parser.exit(1, f"Error: {exc}\n")


if __name__ == "__main__":  # pragma: no cover
    main()
