#!/usr/bin/env python3
"""
Bundle a CovenantIR-vs-baseline evaluation into a PI-shareable folder organized by agreement.

This is intentionally "artifact-first": it copies the exact source text, baseline reference
fields, our validated CovenantIR extraction, and the per-item evaluation record.

It does NOT make any LLM calls.

Inputs
------
- runs/<run_id>/local_sample_manifest.json
    - per item_id: baseline name/limit/start/end + a pointer to the original agreement text
- runs/<run_id>/retrieval_v2/<item_id>_snippets.jsonl
    - the snippet text we fed to the extractor for this item
- scratch/<covenantir_out_dir>/<item_id>/covenantir_validated.json (+ result.json)
    - our extraction output
- comparison_report_v5.json (or later)
    - per-item evaluation record (baseline vs best-match covenant)

Output
------
out_dir/
  PI_writeup.md
  report.json                       (the comparison report copied verbatim)
  index.json                        (bundle index; agreements -> items)
  reference/
    baseline.csv                    (copy of the handcheck CSV used to build the run)
  agreements/<agreement_name>/
    source/<original_agreement_filename>
    items/<item_id>/
      input_snippet.txt
      input_snippet.json
      baseline.json
      covenantir_validated.json
      result.json
      comparison.json
"""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8", errors="replace"))


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _copy_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def _iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if isinstance(obj, dict):
                yield obj


def _first_snippet_record(snippet_jsonl: Path) -> Optional[Dict[str, Any]]:
    if not snippet_jsonl.exists():
        return None
    for rec in _iter_jsonl(snippet_jsonl):
        return rec
    return None


def _extract_covenant_titles(validated: Mapping[str, Any]) -> List[str]:
    covenants = validated.get("covenants")
    if not isinstance(covenants, list):
        return []
    out: List[str] = []
    for c in covenants:
        if not isinstance(c, dict):
            continue
        cid = str(c.get("covenant_id") or "").strip()
        title = str(c.get("title") or "").strip()
        if cid and title:
            out.append(f"{cid}: {title}")
        elif title:
            out.append(title)
        elif cid:
            out.append(cid)
    return out


def _render_writeup_md(
    *,
    run_id: str,
    covenantir_out_dir: str,
    report: Mapping[str, Any],
    per_item_inputs: Mapping[str, Mapping[str, Any]],
    per_item_extractions: Mapping[str, Mapping[str, Any]],
) -> str:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    counts_by_status = report.get("counts_by_status") if isinstance(report.get("counts_by_status"), dict) else {}
    counts_by_disp = report.get("counts_by_disposition") if isinstance(report.get("counts_by_disposition"), dict) else {}
    match_counts = report.get("match_counts") if isinstance(report.get("match_counts"), dict) else {}
    items = report.get("items") if isinstance(report.get("items"), list) else []

    total = len(items)
    success = int(counts_by_disp.get("ok_baseline_matched") or 0)
    limit_only = int(match_counts.get("baseline_limit_found_best") or 0)
    ok = int(counts_by_status.get("ok") or 0)
    blocked = int(counts_by_status.get("blocked") or 0)

    def _example_item_ids(disp: str, k: int = 2) -> List[str]:
        out: List[str] = []
        for it in items:
            if not isinstance(it, dict):
                continue
            if str(it.get("disposition") or "") != disp:
                continue
            item_id = str(it.get("item_id") or "")
            if item_id:
                out.append(item_id)
            if len(out) >= k:
                break
        return out

    # Representative examples for each non-success bucket (PI-friendly).
    example_buckets = [
        ("ok_baseline_limit_matched_wrong_covenant", "Right number, wrong covenant type"),
        ("ok_baseline_limit_missing_despite_in_excerpt", "True miss: baseline is present in snippet but not extracted"),
        ("ok_baseline_limit_not_in_excerpt", "Insufficient evidence: baseline limit not present in snippet"),
        ("ok_zero_covenants_likely_negative_covenant_excerpt", "Out-of-scope excerpt (negative covenants, not a financial test)"),
        ("blocked_baseline_missing_terms", "Blocked: missing schedule/terms needed for baseline covenant"),
        ("blocked_other_missing_terms", "Blocked: missing terms for a different covenant in the excerpt"),
        ("blocked_out_of_scope_issue", "Blocked: excerpt truncated / out-of-scope open item"),
    ]

    md: List[str] = []
    md.append("# Financial Covenant Extraction Eval (Handcheck 30)")
    md.append("")
    md.append(f"Date: {now} (UTC)")
    md.append("")
    md.append("## What we tested")
    md.append("")
    md.append(
        "We evaluated our current one-pass financial covenant extractor (CovenantIR v0.1) against a 30-example "
        "handcheck pack (each example = one agreement + one baseline covenant row)."
    )
    md.append("")
    md.append(
        "Goal context: this evaluation is part of converging on robust, general, and evaluatable financial covenant "
        "extraction logic (computable boolean tests) across credit agreements."
    )
    md.append("")
    md.append("## Success definition (row-level)")
    md.append("")
    md.append(
        "For each example row, we treat **success** as: the extractor returns a covenant that matches the baseline "
        "covenant type/name (identity match) **and** recovers the baseline numeric limit."
    )
    md.append("")
    md.append("Operationally:")
    md.append("- Identity match: overlap on non-generic tokens from the baseline covenant name (e.g., `interest`, `coverage`, `leverage`, `fixed`, `charge`).")
    md.append("- Limit match: baseline limit equals a threshold value found in the extracted covenant test expression and/or referenced schedule tables (with percent↔decimal normalization).")
    md.append("")
    md.append("## High-level results")
    md.append("")
    md.append(f"- Total examples: **{total}**")
    md.append(f"- Status: **{ok} ok**, **{blocked} blocked**")
    md.append(f"- Baseline-matched (strict success): **{success}/{total}**")
    md.append(f"- Limit-only matched (ignoring covenant identity): **{limit_only}/{total}** (overcounts when a different covenant happens to share the same number)")
    md.append("")
    md.append("Disposition counts:")
    md.append("")
    # Stable ordering: success first, then others by count desc.
    md.append("| Disposition | Count |")
    md.append("|---|---:|")
    md.append(f"| ok_baseline_matched | {int(counts_by_disp.get('ok_baseline_matched') or 0)} |")
    for disp, cnt in sorted(counts_by_disp.items(), key=lambda kv: (-int(kv[1] or 0), str(kv[0]))):
        if disp == "ok_baseline_matched":
            continue
        md.append(f"| {disp} | {int(cnt or 0)} |")
    md.append("")
    md.append("## What we did (pipeline, artifact-first)")
    md.append("")
    md.append("1. Built a local run manifest from the handcheck CSV, preserving the baseline fields per row.")
    md.append("2. Stored the exact covenant snippet text used as input (the text we fed to the model).")
    md.append("3. Ran CovenantIR extraction (already completed earlier; no new calls in this bundling pass).")
    md.append("4. Compared baseline→best-match covenant using identity+limit matching, and recorded a per-row disposition.")
    md.append("")
    md.append("Run identifiers:")
    md.append(f"- Snippet pack run_id: `{run_id}`")
    md.append(f"- CovenantIR output dir: `{covenantir_out_dir}`")
    md.append("")
    md.append("## Where it worked vs didn’t (what to pay attention to)")
    md.append("")
    md.append("### Worked")
    md.append("")
    md.append(
        "Most rows succeeded because the snippet contained an explicit covenant test (often a single threshold or a schedule), "
        "and the extractor produced a computable boolean test with thresholds captured either as AST literals or table lookups."
    )
    md.append("")
    md.append("### Didn’t work (failure modes)")
    md.append("")
    md.append(
        "The remaining failures fall into a small number of buckets. The most important distinction is whether the **evidence "
        "needed for the baseline covenant is actually present in the snippet**."
    )
    md.append("")
    for disp, label in example_buckets:
        cnt = int(counts_by_disp.get(disp) or 0)
        if cnt <= 0:
            continue
        md.append(f"#### {label} (`{disp}`) — {cnt} row(s)")
        ex_ids = _example_item_ids(disp, k=2)
        for item_id in ex_ids:
            it = next((x for x in items if isinstance(x, dict) and x.get("item_id") == item_id), None)
            if not isinstance(it, dict):
                continue
            baseline = it.get("baseline") if isinstance(it.get("baseline"), dict) else {}
            excerpt = it.get("excerpt") if isinstance(it.get("excerpt"), dict) else {}
            cov = it.get("covenantir") if isinstance(it.get("covenantir"), dict) else {}
            best = it.get("best_match") if isinstance(it.get("best_match"), dict) else {}
            snippet_preview = str(excerpt.get("snippet_preview") or "").strip()
            open_issue = str(cov.get("open_item_issue") or "").strip()

            md.append(f"- Example `{item_id}`")
            md.append(f"  - Baseline: **{baseline.get('name')}** limit **{baseline.get('limit_raw')}**")
            if snippet_preview:
                md.append(f"  - Snippet preview: “{snippet_preview}”")
            if best:
                md.append(f"  - Best-match extracted covenant: **{best.get('title')}** (score {best.get('combined_score')})")
            titles = _extract_covenant_titles(per_item_extractions.get(item_id, {}))
            if titles:
                md.append(f"  - Extracted covenants: {', '.join(titles[:4])}{' …' if len(titles) > 4 else ''}")
            if open_issue:
                md.append(f"  - Blocking issue: {open_issue}")
            # Evidence flags (helpful for PI: whether the snippet plausibly contains the baseline).
            md.append(
                "  - Evidence checks: "
                f"name_in_snippet={bool(excerpt.get('baseline_name_in_excerpt'))}, "
                f"limit_in_snippet={bool(excerpt.get('baseline_limit_in_excerpt'))}"
            )
        md.append("")

    md.append("## Bundle contents (how to review)")
    md.append("")
    md.append("This folder is organized by agreement. For each agreement you’ll find:")
    md.append("- The original full text file used for provenance.")
    md.append("- The exact snippet text used as extractor input.")
    md.append("- The baseline reference fields (name/limit/start/end).")
    md.append("- Our validated CovenantIR extraction JSON.")
    md.append("- The evaluation record for that row (baseline vs best-match covenant).")
    md.append("")
    md.append("## Recommended next steps")
    md.append("")
    md.append("1. Fix the one true extraction miss where the baseline covenant is present in the snippet but not extracted.")
    md.append("2. Decide how we want to treat rows where the baseline covenant is not present in the snippet evidence (data/retrieval mismatch).")
    md.append("3. For blocked cases, decide whether to (a) require retrieval to include missing schedules, or (b) allow partial extraction when the baseline covenant is extractable.")
    md.append("")

    return "\n".join(md).rstrip() + "\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True, help="Run id under runs/<run-id>/ (must contain local_sample_manifest.json + retrieval_v2/)")
    ap.add_argument("--covenantir-out-dir", required=True, help="CovenantIR batch output dir (contains <item_id>/covenantir_validated.json)")
    ap.add_argument("--comparison-report", required=True, help="Comparison report JSON (e.g., comparison_report_v5.json)")
    ap.add_argument("--out-dir", required=True, help="Destination folder to write the bundle into")
    ap.add_argument("--overwrite", action="store_true", help="Delete out-dir before writing if it exists")
    args = ap.parse_args()

    run_dir = Path("runs") / args.run_id
    manifest_path = run_dir / "local_sample_manifest.json"
    retrieval_dir = run_dir / "retrieval_v2"
    if not manifest_path.exists():
        raise SystemExit(f"Missing manifest: {manifest_path}")
    if not retrieval_dir.exists():
        raise SystemExit(f"Missing retrieval dir: {retrieval_dir}")

    covenantir_dir = Path(args.covenantir_out_dir)
    if not covenantir_dir.exists():
        raise SystemExit(f"Missing CovenantIR out dir: {covenantir_dir}")

    report_path = Path(args.comparison_report)
    if not report_path.exists():
        raise SystemExit(f"Missing comparison report: {report_path}")

    out_dir = Path(args.out_dir)
    if out_dir.exists():
        if not args.overwrite:
            raise SystemExit(f"Out dir already exists: {out_dir} (pass --overwrite to recreate)")
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = _read_json(manifest_path)
    items = manifest.get("items")
    if not isinstance(items, list) or not items:
        raise SystemExit(f"Manifest items missing/empty: {manifest_path}")

    report = _read_json(report_path)
    report_items = report.get("items")
    if not isinstance(report_items, list):
        raise SystemExit(f"Comparison report missing items[]: {report_path}")
    report_map: Dict[str, Dict[str, Any]] = {}
    for rec in report_items:
        if isinstance(rec, dict) and isinstance(rec.get("item_id"), str) and rec["item_id"].strip():
            report_map[rec["item_id"].strip()] = rec

    # Copy reference CSV (source-of-truth for baseline data).
    csv_path = manifest.get("csv_path")
    if isinstance(csv_path, str) and csv_path.strip():
        src_csv = Path(csv_path.strip())
        if src_csv.exists():
            _copy_file(src_csv, out_dir / "reference" / "baseline.csv")

    # Copy report verbatim for auditability.
    _copy_file(report_path, out_dir / "report.json")

    # Build agreement index and copy per-item artifacts.
    agreements_index: Dict[str, Any] = {
        "schema_version": "covenantir_handcheck_bundle_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_id": args.run_id,
        "covenantir_out_dir": str(covenantir_dir),
        "comparison_report": str(report_path),
        "agreements": {},
    }

    per_item_inputs: Dict[str, Dict[str, Any]] = {}
    per_item_extractions: Dict[str, Dict[str, Any]] = {}

    copied_agreements: set[str] = set()
    for it in items:
        if not isinstance(it, dict):
            continue
        item_id = str(it.get("item_id") or "").strip()
        if not item_id:
            continue

        source = it.get("source") if isinstance(it.get("source"), dict) else {}
        agreement_path_abs = str(source.get("local_agreement_path_abs") or "").strip()
        if not agreement_path_abs:
            raise SystemExit(f"Item {item_id} missing local_agreement_path_abs in manifest: {manifest_path}")

        agreement_path = Path(agreement_path_abs)
        if not agreement_path.exists():
            raise SystemExit(f"Missing agreement source text for {item_id}: {agreement_path}")

        agreement_name = agreement_path.name
        agreement_dir = out_dir / "agreements" / agreement_name.replace(".txt", "")
        (agreement_dir / "items").mkdir(parents=True, exist_ok=True)

        # Copy full agreement text once.
        if agreement_path_abs not in copied_agreements:
            _copy_file(agreement_path, agreement_dir / "source" / agreement_path.name)
            copied_agreements.add(agreement_path_abs)

        # Item folder
        item_dir = agreement_dir / "items" / item_id
        item_dir.mkdir(parents=True, exist_ok=True)

        # Input snippet used for extraction
        snippet_rec = _first_snippet_record(retrieval_dir / f"{item_id}_snippets.jsonl")
        if snippet_rec is not None:
            per_item_inputs[item_id] = snippet_rec
            _write_json(item_dir / "input_snippet.json", snippet_rec)
            snip = snippet_rec.get("snippet")
            if isinstance(snip, str):
                (item_dir / "input_snippet.txt").write_text(snip, encoding="utf-8")

        # Baseline reference fields (from manifest)
        baseline = it.get("baseline") if isinstance(it.get("baseline"), dict) else {}
        baseline_out = {
            "item_id": item_id,
            "row_index": it.get("row_index"),
            "baseline": baseline,
            "source": {
                "csv": source.get("csv"),
                "local_agreement_path_abs": agreement_path_abs,
                "chunk_text_chars": source.get("chunk_text_chars"),
            },
        }
        _write_json(item_dir / "baseline.json", baseline_out)

        # CovenantIR extraction artifacts
        cov_item_dir = covenantir_dir / item_id
        validated_path = cov_item_dir / "covenantir_validated.json"
        result_path = cov_item_dir / "result.json"
        if validated_path.exists():
            _copy_file(validated_path, item_dir / "covenantir_validated.json")
            try:
                per_item_extractions[item_id] = _read_json(validated_path)
            except Exception:
                per_item_extractions[item_id] = {}
        if result_path.exists():
            _copy_file(result_path, item_dir / "result.json")

        # Per-item evaluation record (comparison report)
        comp = report_map.get(item_id)
        if comp is not None:
            _write_json(item_dir / "comparison.json", comp)

        # Index entry
        agreements_index["agreements"].setdefault(agreement_name.replace(".txt", ""), {"source_filename": agreement_path.name, "items": []})
        agreements_index["agreements"][agreement_name.replace(".txt", "")]["items"].append(item_id)

    _write_json(out_dir / "index.json", agreements_index)

    # Write a PI-facing markdown writeup into the bundle, grounded in the report + copied artifacts.
    writeup = _render_writeup_md(
        run_id=args.run_id,
        covenantir_out_dir=str(covenantir_dir),
        report=report,
        per_item_inputs=per_item_inputs,
        per_item_extractions=per_item_extractions,
    )
    (out_dir / "PI_writeup.md").write_text(writeup, encoding="utf-8")

    print(f"[done] bundle written: {out_dir}")
    print(f"  agreements: {len(agreements_index['agreements'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
