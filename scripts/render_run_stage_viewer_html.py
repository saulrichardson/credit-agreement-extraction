#!/usr/bin/env python
"""
Render a self-contained HTML viewer for full run-stage artifacts.

Goal:
- Browse agreements (item_id) for one run or a group of runs.
- Inspect stage outputs (indexing, retrieval, structured outputs, definitions, metadata).
- See full agreement text with anchor blocks.
- Click any cited anchor (A####) in an output to jump to the corresponding location
  in the agreement text.

This script makes no LLM calls. It only reads run artifacts.

Useful defaults:
- Includes only agreements with at least one useful downstream output by default.
  "Useful downstream" is true when any of:
    * pricing_structured has non-empty facilities/metrics/rates/overrides
    * recursive pricing definitions compiler aggregate has at least one usable definition
    * covenant_structured has non-empty covenants/metrics
    * recursive covenant definitions compiler aggregate has at least one usable definition

Examples:
  python scripts/render_run_stage_viewer_html.py \
    --run-id cea100-credit-cg-20260211c-s0001 \
    --base-dir /scratch/sxr203/projects/credit-agreement-extraction

  python scripts/render_run_stage_viewer_html.py \
    --run-glob 'cea100-credit-cg-20260211c-s00*' \
    --base-dir /scratch/sxr203/projects/credit-agreement-extraction \
    --out /scratch/sxr203/projects/credit-agreement-extraction/runs/cea100-c-useful-viewer.html
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from pipeline.core.anchors import load_anchor_catalog  # noqa: E402
from pipeline.core.config import Paths  # noqa: E402


ANCHOR_ID_RE = re.compile(r"^A\d{4,}$")
ANCHOR_FINDER_RE = re.compile(r"(A\d{4,})")
ANNOTATED_ANCHOR_LINE_RE = re.compile(r"^\[\[(A\d{4,})\]\]\s*$")

# Recursive workflow subdir candidates (new outputs first, older names as fallback).
PRICING_METRICS_SUBDIRS = [
    "compiled_pricing_metrics_recursive_ast_v2",
    "compiled_pricing_metrics_recursive_ast_v1",
]
PRICING_BLOCKING_SUBDIRS = [
    "blocking_pricing_terms_recursive_ast_v2_depth1",
    "blocking_pricing_terms_recursive_ast_v1_depth1",
]
PRICING_OVERLAY_SUBDIRS = [
    "compustat_overlay_pricing_recursive_ast_v2",
    "compustat_overlay_pricing_recursive_ast_v1",
]

COVENANT_METRICS_SUBDIRS = [
    "compiled_covenant_metrics_recursive_ast_v2",
    "compiled_covenant_metrics_recursive_ast_v1",
]
COVENANT_BLOCKING_SUBDIRS = [
    "blocking_covenant_terms_recursive_ast_v2_depth1",
    "blocking_covenant_terms_recursive_ast_v1_depth1",
]
COVENANT_OVERLAY_SUBDIRS = [
    "compustat_overlay_covenant_recursive_ast_v2",
    "compustat_overlay_covenant_recursive_ast_v1",
]

# Stage-level prompt candidates (most likely/current first; falls back to older/default variants).
STAGE_PROMPT_CANDIDATES: dict[str, list[str]] = {
    "indexing_v2": [
        "prompts/indexing_v2.txt",
        "prompts/comprehensive_indexing.txt",
    ],
    "agreement_metadata": [
        "prompts/agreement_metadata_v1.txt",
    ],
    "pricing_structured": [
        "prompts/prompt_pricing_second_pass_dg_nano_v2_tuned_v2.txt",
    ],
    "pricing_metrics_recursive": [
        "prompts/definitions_compiler_v1_metrics_ast_v2.txt",
    ],
    "pricing_blocking_recursive": [
        "prompts/blocking_terms_compiler_v1_ast_v2.txt",
    ],
    "pricing_overlay_recursive": [
        "prompts/compustat_overlay_v1.txt",
    ],
    "covenant_structured": [
        "prompts/prompt_v1_short.txt",
    ],
    "covenant_metrics_recursive": [
        "prompts/definitions_compiler_v1_metrics_ast_v2.txt",
    ],
    "covenant_blocking_recursive": [
        "prompts/blocking_terms_compiler_v1_ast_v2.txt",
    ],
    "covenant_overlay_recursive": [
        "prompts/compustat_overlay_v1.txt",
    ],
}

# Companion artifacts that are logically part of the system prompt contract for a stage.
# Example: overlay prompts depend on the Compustat allowlist.
STAGE_PROMPT_COMPANION_CANDIDATES: dict[str, list[str]] = {
    "pricing_overlay_recursive": [
        "datasets/compustat_allowlist_quarterly_v1.json",
        "datasets/compustat_allowlist_quarterly_q_only_v1.json",
    ],
    "covenant_overlay_recursive": [
        "datasets/compustat_allowlist_quarterly_v1.json",
        "datasets/compustat_allowlist_quarterly_q_only_v1.json",
    ],
}


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def _read_json(path: Path) -> Any:
    return json.loads(_read_text(path))


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _safe_slug(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "_", (s or "").strip()).strip("_") or "viewer"


def _as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _anchor_sort_key(aid: str) -> tuple[int, str]:
    if isinstance(aid, str) and ANCHOR_ID_RE.fullmatch(aid):
        return (0, f"{int(aid[1:]):09d}")
    return (1, str(aid))


def _extract_anchor_ids_from_json(doc: Any) -> Set[str]:
    anchors: Set[str] = set()

    def _walk(x: Any) -> None:
        if isinstance(x, dict):
            for v in x.values():
                _walk(v)
            return
        if isinstance(x, list):
            for v in x:
                _walk(v)
            return
        if isinstance(x, str):
            s = x.strip()
            if ANCHOR_ID_RE.fullmatch(s):
                anchors.add(s)
                return
            for m in ANCHOR_FINDER_RE.finditer(s):
                aid = m.group(1)
                if ANCHOR_ID_RE.fullmatch(aid):
                    anchors.add(aid)

    _walk(doc)
    return anchors


def _extract_anchor_ids_from_text(text: str) -> Set[str]:
    return {m.group(1) for m in ANCHOR_FINDER_RE.finditer(text or "") if ANCHOR_ID_RE.fullmatch(m.group(1))}


def _parse_jsonl(path: Path) -> list[Any]:
    rows: list[Any] = []
    for i, raw in enumerate(_read_text(path).splitlines(), start=1):
        s = raw.strip()
        if not s:
            continue
        try:
            rows.append(json.loads(s))
        except Exception as exc:
            rows.append({"_line": i, "_parse_error": f"{type(exc).__name__}: {exc}", "_raw": raw})
    return rows


def _resolve_stage_prompts(root: Path) -> dict[str, dict[str, str]]:
    out: dict[str, dict[str, str]] = {}
    for stage, candidates in STAGE_PROMPT_CANDIDATES.items():
        for rel in candidates:
            p = root / rel
            if not p.exists():
                continue
            prompt_text = _read_text(p)

            companion_paths: list[str] = []
            companion_text_blocks: list[str] = []
            for companion_rel in STAGE_PROMPT_COMPANION_CANDIDATES.get(stage, []):
                cp = root / companion_rel
                if not cp.exists():
                    continue
                companion_paths.append(str(cp))
                companion_text_blocks.append(
                    "\n\n"
                    f"===== BEGIN COMPANION ARTIFACT: {cp} =====\n"
                    f"{_read_text(cp)}\n"
                    f"===== END COMPANION ARTIFACT: {cp} =====\n"
                )
                # Use one canonical companion artifact to keep the viewer concise.
                break

            merged_text = prompt_text + "".join(companion_text_blocks)
            out[stage] = {
                "path": str(p),
                "text": merged_text,
                "companion_paths": companion_paths,
            }
            break
    return out


def _parse_canonical_annotated(text: str) -> List[Dict[str, Any]]:
    blocks: List[Dict[str, Any]] = []
    current_id: Optional[str] = None
    current_lines: List[str] = []

    for raw_line in text.splitlines():
        m = ANNOTATED_ANCHOR_LINE_RE.match(raw_line.strip())
        if m:
            if current_id is not None:
                blocks.append({"anchor_id": current_id, "text": "\n".join(current_lines).rstrip("\n")})
            current_id = m.group(1)
            current_lines = []
            continue

        if current_id is None:
            current_id = "__PREFACE__"
            current_lines = []
        current_lines.append(raw_line)

    if current_id is not None:
        blocks.append({"anchor_id": current_id, "text": "\n".join(current_lines).rstrip("\n")})

    if blocks and blocks[0].get("anchor_id") == "__PREFACE__" and not str(blocks[0].get("text") or "").strip():
        blocks = blocks[1:]

    cleaned: List[Dict[str, Any]] = []
    for b in blocks:
        aid = str(b.get("anchor_id") or "").strip()
        txt = str(b.get("text") or "")
        if aid != "__PREFACE__" and not ANCHOR_ID_RE.fullmatch(aid):
            raise ValueError(f"Invalid anchor marker in canonical_annotated parse: {aid!r}")
        cleaned.append({"anchor_id": aid, "text": txt})
    return cleaned


def _anchor_text_from_catalog(
    canonical_text: str,
    catalog: Mapping[str, Mapping[str, Any]],
    anchor_id: str,
) -> Optional[str]:
    info = catalog.get(anchor_id)
    if not info:
        return None
    start = info.get("start")
    end = info.get("end")
    if not isinstance(start, int) or not isinstance(end, int):
        return None
    a = max(0, start)
    b = min(len(canonical_text), end)
    if b <= a:
        return None
    txt = canonical_text[a:b].rstrip()
    return txt if txt.strip() else None


def _fallback_blocks_from_canonical(paths: Paths, item_id: str) -> List[Dict[str, Any]]:
    canonical_path = paths.run_dir / "normalized" / item_id / "canonical.txt"
    if not canonical_path.exists():
        raise FileNotFoundError(f"Missing canonical.txt for {item_id}: expected {canonical_path}")
    canonical_text = _read_text(canonical_path)

    catalog = load_anchor_catalog(paths, item_id)
    ordered = sorted(catalog.values(), key=lambda a: int(a["order"]))
    blocks: List[Dict[str, Any]] = []
    for info in ordered:
        aid = str(info.get("anchor_id") or "")
        if not ANCHOR_ID_RE.fullmatch(aid):
            continue
        txt = _anchor_text_from_catalog(canonical_text, catalog, aid) or ""
        blocks.append({"anchor_id": aid, "text": txt})
    return blocks


def _load_document_blocks(paths: Paths, item_id: str) -> Tuple[str, List[Dict[str, Any]]]:
    annotated_path = paths.run_dir / "normalized" / item_id / "canonical_annotated.txt"
    if annotated_path.exists():
        return (str(annotated_path), _parse_canonical_annotated(_read_text(annotated_path)))
    return (
        str(paths.run_dir / "normalized" / item_id / "canonical.txt"),
        _fallback_blocks_from_canonical(paths, item_id),
    )


def _load_manifest(run_dir: Path) -> Dict[str, Any]:
    mp = run_dir / "manifest.json"
    if not mp.exists():
        return {}
    doc = _read_json(mp)
    return doc if isinstance(doc, dict) else {}


def _resolve_artifact_path(*, root: Path, run_dir: Path, rel_or_abs: Any) -> Optional[Path]:
    if not isinstance(rel_or_abs, str) or not rel_or_abs.strip():
        return None
    p = Path(rel_or_abs.strip())
    if p.is_absolute():
        return p

    candidates: list[Path] = [root / p]
    if not (len(p.parts) >= 1 and p.parts[0] == "runs"):
        candidates.append(run_dir / p)

    for c in candidates:
        if c.exists():
            return c
    return candidates[0]


def _records_candidates(*, run_dir: Path, manifest: Mapping[str, Any], analysis_subdir: Optional[str]) -> list[Path]:
    cands: list[Path] = []

    if isinstance(analysis_subdir, str) and analysis_subdir.strip():
        sub = analysis_subdir.strip().strip("/")
        cands.extend(
            [
                run_dir / sub / "records.jsonl",
                run_dir / "analysis_export" / sub / "records.jsonl",
            ]
        )

    for key in ("analysis_export_v2_output_subdir", "analysis_export_output_subdir"):
        v = manifest.get(key)
        if isinstance(v, str) and v.strip():
            cands.append(run_dir / "analysis_export" / v.strip() / "records.jsonl")

    cands.extend(
        [
            run_dir / "analysis_export" / "analysis_export_v2" / "records.jsonl",
            run_dir / "analysis_export_v2" / "records.jsonl",
            run_dir / "analysis_export" / "records.jsonl",
        ]
    )

    uniq: list[Path] = []
    seen: Set[str] = set()
    for p in cands:
        k = str(p)
        if k in seen:
            continue
        seen.add(k)
        uniq.append(p)
    return uniq


def _choose_records_path(*, run_dir: Path, manifest: Mapping[str, Any], analysis_subdir: Optional[str]) -> Path:
    candidates = _records_candidates(run_dir=run_dir, manifest=manifest, analysis_subdir=analysis_subdir)
    for p in candidates:
        if p.exists():
            return p
    msg = "\n".join(f"  - {c}" for c in candidates)
    raise FileNotFoundError(f"No analysis records.jsonl found for run_dir={run_dir}. Checked:\n{msg}")


def _load_records(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for i, raw in enumerate(_read_text(path).splitlines(), start=1):
        s = raw.strip()
        if not s:
            continue
        try:
            doc = json.loads(s)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"Invalid JSON in records file {path} line={i}: {exc}") from exc
        if not isinstance(doc, dict):
            raise RuntimeError(f"Expected JSON object in records file {path} line={i}, got {type(doc).__name__}")
        rows.append(doc)
    return rows


def _first_existing(paths: Sequence[Path]) -> Optional[Path]:
    for p in paths:
        if p.exists():
            return p
    return None


def _recursive_compiled_path_candidates(
    *,
    run_dir: Path,
    stage_dir: str,
    subdirs: Sequence[str],
    item_id: str,
    suffix: str,
) -> list[Path]:
    return [run_dir / stage_dir / sub / f"{item_id}{suffix}" for sub in subdirs]


def _has_usable_recursive_compiled(doc: Any) -> bool:
    if not isinstance(doc, dict):
        return False
    defs = doc.get("definitions")
    if not isinstance(defs, list):
        return False
    for row in defs:
        if not isinstance(row, dict):
            continue
        dv = row.get("definition_verbatim")
        src = row.get("source_refs")
        if isinstance(dv, str) and dv.strip():
            return True
        if isinstance(src, list) and any(isinstance(x, str) and x.strip() for x in src):
            return True
    return False


def _load_recursive_useful_flags(*, run_dir: Path, item_id: str) -> dict[str, bool]:
    pricing_metrics_path = _first_existing(
        _recursive_compiled_path_candidates(
            run_dir=run_dir,
            stage_dir="definitions_compiler_v1",
            subdirs=PRICING_METRICS_SUBDIRS,
            item_id=item_id,
            suffix="__compiled.json",
        )
    )
    covenant_metrics_path = _first_existing(
        _recursive_compiled_path_candidates(
            run_dir=run_dir,
            stage_dir="definitions_compiler_v1",
            subdirs=COVENANT_METRICS_SUBDIRS,
            item_id=item_id,
            suffix="__compiled.json",
        )
    )

    pricing_recursive = False
    if pricing_metrics_path is not None:
        try:
            pricing_recursive = _has_usable_recursive_compiled(_read_json(pricing_metrics_path))
        except Exception:
            pricing_recursive = False

    covenant_recursive = False
    if covenant_metrics_path is not None:
        try:
            covenant_recursive = _has_usable_recursive_compiled(_read_json(covenant_metrics_path))
        except Exception:
            covenant_recursive = False

    return {
        "pricing_recursive_definitions": pricing_recursive,
        "covenant_recursive_definitions": covenant_recursive,
    }


def _pricing_structured_useful(doc: Any) -> bool:
    if not isinstance(doc, dict):
        return bool(doc)
    facilities = _as_list(doc.get("facilities"))
    metrics = _as_list(doc.get("metrics"))
    overrides = _as_list(doc.get("overrides"))
    if metrics or overrides:
        return True
    for f in facilities:
        if not isinstance(f, dict):
            continue
        if _as_list(f.get("rates")):
            return True
        if f.get("committed_amount") not in (None, "", 0):
            return True
    return len(facilities) > 0


def _covenant_structured_useful(doc: Any) -> bool:
    if not isinstance(doc, dict):
        return bool(doc)
    return bool(_as_list(doc.get("covenants")) or _as_list(doc.get("metrics")))


def _useful_flags(
    record: Mapping[str, Any],
    *,
    useful_mode: str,
    run_dir: Path,
    item_id: str,
) -> dict[str, bool]:
    pricing_structured = record.get("pricing_structured")
    covenant_structured = record.get("covenant_structured")
    recursive = _load_recursive_useful_flags(run_dir=run_dir, item_id=item_id)

    ps = _pricing_structured_useful(pricing_structured)
    cs = _covenant_structured_useful(covenant_structured)
    # Strict mode: only recursive definition aggregates count as definition usefulness.
    pd = bool(recursive.get("pricing_recursive_definitions"))
    cd = bool(recursive.get("covenant_recursive_definitions"))
    if useful_mode == "definitions":
        any_downstream = bool(pd or cd)
    elif useful_mode == "structured_or_definitions":
        any_downstream = bool(ps or pd or cs or cd)
    else:
        raise ValueError(f"Unsupported useful_mode={useful_mode!r}")

    return {
        "pricing_structured": ps,
        "pricing_definitions": pd,
        "covenant_structured": cs,
        "covenant_definitions": cd,
        "pricing_definitions_recursive": bool(recursive.get("pricing_recursive_definitions")),
        "covenant_definitions_recursive": bool(recursive.get("covenant_recursive_definitions")),
        "any_downstream": any_downstream,
    }


def _analysis_record_summary(record: Mapping[str, Any], useful: Mapping[str, bool]) -> dict[str, Any]:
    pricing_structured = record.get("pricing_structured")
    covenant_structured = record.get("covenant_structured")
    if not isinstance(pricing_structured, dict):
        pricing_structured = {}
    if not isinstance(covenant_structured, dict):
        covenant_structured = {}

    def _compiled_defs_count(doc: Any) -> int:
        if not isinstance(doc, dict):
            return 0
        rows = doc.get("definitions")
        if not isinstance(rows, list):
            return 0
        return sum(1 for row in rows if isinstance(row, dict))

    def _overlay_count(doc: Any) -> int:
        if not isinstance(doc, dict):
            return 0
        rows = doc.get("overlays")
        if not isinstance(rows, list):
            return 0
        return sum(1 for row in rows if isinstance(row, dict))

    pricing_metrics_recursive = record.get("pricing_metrics_recursive")
    pricing_blocking_recursive = record.get("pricing_blocking_recursive")
    pricing_overlay_recursive = record.get("pricing_overlay_recursive")
    covenant_metrics_recursive = record.get("covenant_metrics_recursive")
    covenant_blocking_recursive = record.get("covenant_blocking_recursive")
    covenant_overlay_recursive = record.get("covenant_overlay_recursive")

    return {
        "schema_version": record.get("schema_version"),
        "run_id": record.get("run_id"),
        "item_id": record.get("item_id"),
        "source_path": record.get("source_path"),
        "canonical_text_path": record.get("canonical_text_path"),
        "anchors_tsv_path": record.get("anchors_tsv_path"),
        "indexing_v2_path": record.get("indexing_v2_path"),
        "retrieval_v2_path": record.get("retrieval_v2_path"),
        "agreement_metadata_path": record.get("agreement_metadata_path"),
        "pricing_structured_path": record.get("pricing_structured_path"),
        "covenant_structured_path": record.get("covenant_structured_path"),
        "pricing_metrics_recursive_path": record.get("pricing_metrics_recursive_path"),
        "pricing_blocking_recursive_path": record.get("pricing_blocking_recursive_path"),
        "pricing_overlay_recursive_path": record.get("pricing_overlay_recursive_path"),
        "covenant_metrics_recursive_path": record.get("covenant_metrics_recursive_path"),
        "covenant_blocking_recursive_path": record.get("covenant_blocking_recursive_path"),
        "covenant_overlay_recursive_path": record.get("covenant_overlay_recursive_path"),
        "errors": record.get("errors"),
        "useful_flags": dict(useful),
        "counts": {
            "pricing_facilities": len(_as_list(pricing_structured.get("facilities"))),
            "pricing_metrics": len(_as_list(pricing_structured.get("metrics"))),
            "pricing_overrides": len(_as_list(pricing_structured.get("overrides"))),
            "covenants": len(_as_list(covenant_structured.get("covenants"))),
            "covenant_metrics": len(_as_list(covenant_structured.get("metrics"))),
            "pricing_metrics_recursive_rows": _compiled_defs_count(pricing_metrics_recursive),
            "pricing_blocking_recursive_rows": _compiled_defs_count(pricing_blocking_recursive),
            "pricing_overlay_recursive_rows": _overlay_count(pricing_overlay_recursive),
            "covenant_metrics_recursive_rows": _compiled_defs_count(covenant_metrics_recursive),
            "covenant_blocking_recursive_rows": _compiled_defs_count(covenant_blocking_recursive),
            "covenant_overlay_recursive_rows": _overlay_count(covenant_overlay_recursive),
        },
    }


def _default_output_id(outputs: Sequence[Mapping[str, Any]]) -> Optional[str]:
    for pred in (
        lambda o: o.get("output_id") == "pricing_structured",
        lambda o: o.get("output_id") == "covenant_structured",
        lambda o: o.get("output_id") == "pricing_metrics_recursive",
        lambda o: o.get("output_id") == "covenant_metrics_recursive",
        lambda o: o.get("output_id") == "pricing_overlay_recursive",
        lambda o: o.get("output_id") == "covenant_overlay_recursive",
        lambda o: o.get("output_id") == "indexing_v2",
        lambda o: o.get("output_id") == "retrieval_v2",
    ):
        for o in outputs:
            if pred(o):
                return str(o.get("output_id"))
    return str(outputs[0].get("output_id")) if outputs else None


def _load_outputs_for_record(
    *,
    root: Path,
    run_dir: Path,
    record: Mapping[str, Any],
    useful: Mapping[str, bool],
    doc_anchor_ids: Set[str],
) -> list[dict[str, Any]]:
    outputs: list[dict[str, Any]] = []
    item_id = str(record.get("item_id") or "").strip()
    if not item_id:
        return outputs

    def _push_output(
        *,
        stage: str,
        label: str,
        output_id: str,
        kind: str,
        path: Optional[Path],
        doc: Any = None,
        text: Optional[str] = None,
    ) -> None:
        if kind not in {"json", "text"}:
            raise ValueError(f"Invalid output kind={kind!r} for stage={stage} output_id={output_id}")

        if kind == "json":
            cited = sorted(_extract_anchor_ids_from_json(doc), key=_anchor_sort_key)
            payload: dict[str, Any] = {
                "output_id": output_id,
                "stage": stage,
                "label": label,
                "kind": "json",
                "path": str(path) if path else None,
                "json": doc,
                "cited_anchor_ids": cited,
            }
        else:
            txt = text or ""
            cited = sorted(_extract_anchor_ids_from_text(txt), key=_anchor_sort_key)
            payload = {
                "output_id": output_id,
                "stage": stage,
                "label": label,
                "kind": "text",
                "path": str(path) if path else None,
                "text": txt,
                "cited_anchor_ids": cited,
            }

        payload["unknown_anchor_ids"] = sorted([a for a in payload["cited_anchor_ids"] if a not in doc_anchor_ids], key=_anchor_sort_key)
        outputs.append(payload)

    def _add_missing(stage: str, label: str, output_id: str, expected: Any) -> None:
        _push_output(
            stage=stage,
            label=label,
            output_id=output_id,
            kind="text",
            path=None,
            text=f"(missing artifact)\nexpected={expected}",
        )

    def _add_json_file(stage: str, label: str, output_id: str, path_hint: Any, *, required: bool = False) -> None:
        p = _resolve_artifact_path(root=root, run_dir=run_dir, rel_or_abs=path_hint)
        if p is None:
            if required:
                _add_missing(stage, label, output_id, path_hint)
            return
        if not p.exists():
            if required:
                _add_missing(stage, label, output_id, str(p))
            return
        try:
            doc = _read_json(p)
        except Exception as exc:
            _push_output(
                stage=stage,
                label=label,
                output_id=output_id,
                kind="text",
                path=p,
                text=f"(error reading JSON) {type(exc).__name__}: {exc}\n\nRAW:\n{_read_text(p)}",
            )
            return
        _push_output(stage=stage, label=label, output_id=output_id, kind="json", path=p, doc=doc)

    def _add_jsonl_file(stage: str, label: str, output_id: str, path_hint: Any, *, required: bool = False) -> None:
        p = _resolve_artifact_path(root=root, run_dir=run_dir, rel_or_abs=path_hint)
        if p is None:
            if required:
                _add_missing(stage, label, output_id, path_hint)
            return
        if not p.exists():
            if required:
                _add_missing(stage, label, output_id, str(p))
            return
        rows = _parse_jsonl(p)
        _push_output(stage=stage, label=label, output_id=output_id, kind="json", path=p, doc=rows)

    def _add_json_obj(stage: str, label: str, output_id: str, obj: Any, *, path_hint: Any = None) -> None:
        p = _resolve_artifact_path(root=root, run_dir=run_dir, rel_or_abs=path_hint) if path_hint else None
        _push_output(stage=stage, label=label, output_id=output_id, kind="json", path=p, doc=obj)

    def _add_first_existing_json(stage: str, label: str, output_id: str, candidates: Sequence[Path]) -> None:
        p = _first_existing(candidates)
        if p is None:
            return
        try:
            doc = _read_json(p)
        except Exception as exc:
            _push_output(
                stage=stage,
                label=label,
                output_id=output_id,
                kind="text",
                path=p,
                text=f"(error reading JSON) {type(exc).__name__}: {exc}\n\nRAW:\n{_read_text(p)}",
            )
            return
        _push_output(stage=stage, label=label, output_id=output_id, kind="json", path=p, doc=doc)

    _add_json_file(
        "indexing_v2",
        "indexing_v2 selection",
        "indexing_v2",
        record.get("indexing_v2_path"),
        required=True,
    )
    _add_jsonl_file(
        "retrieval_v2",
        "retrieval_v2 snippets",
        "retrieval_v2",
        record.get("retrieval_v2_path"),
        required=True,
    )
    _add_json_file(
        "agreement_metadata",
        "agreement_metadata_v1",
        "agreement_metadata",
        record.get("agreement_metadata_path"),
        required=True,
    )
    _add_json_file(
        "pricing_structured",
        "pricing_structured",
        "pricing_structured",
        record.get("pricing_structured_path"),
        required=True,
    )
    _add_json_file(
        "covenant_structured",
        "covenant_structured",
        "covenant_structured",
        record.get("covenant_structured_path"),
        required=True,
    )
    # Recursive definitions + overlay workflow.
    _add_first_existing_json(
        "pricing_metrics_recursive",
        "definitions_compiler_v1 (pricing aggregate, recursive)",
        "pricing_metrics_recursive",
        _recursive_compiled_path_candidates(
            run_dir=run_dir,
            stage_dir="definitions_compiler_v1",
            subdirs=PRICING_METRICS_SUBDIRS,
            item_id=item_id,
            suffix="__compiled.json",
        ),
    )
    _add_first_existing_json(
        "pricing_blocking_recursive",
        "blocking_terms_compiler_v1 (pricing aggregate, recursive)",
        "pricing_blocking_recursive",
        _recursive_compiled_path_candidates(
            run_dir=run_dir,
            stage_dir="blocking_terms_compiler_v1",
            subdirs=PRICING_BLOCKING_SUBDIRS,
            item_id=item_id,
            suffix="__compiled.json",
        ),
    )
    _add_first_existing_json(
        "pricing_overlay_recursive",
        "compustat_overlay_v1 (pricing aggregate, recursive)",
        "pricing_overlay_recursive",
        _recursive_compiled_path_candidates(
            run_dir=run_dir,
            stage_dir="compustat_overlay_v1",
            subdirs=PRICING_OVERLAY_SUBDIRS,
            item_id=item_id,
            suffix="__compustat_overlay.json",
        ),
    )
    _add_first_existing_json(
        "covenant_metrics_recursive",
        "definitions_compiler_v1 (covenant aggregate, recursive)",
        "covenant_metrics_recursive",
        _recursive_compiled_path_candidates(
            run_dir=run_dir,
            stage_dir="definitions_compiler_v1",
            subdirs=COVENANT_METRICS_SUBDIRS,
            item_id=item_id,
            suffix="__compiled.json",
        ),
    )
    _add_first_existing_json(
        "covenant_blocking_recursive",
        "blocking_terms_compiler_v1 (covenant aggregate, recursive)",
        "covenant_blocking_recursive",
        _recursive_compiled_path_candidates(
            run_dir=run_dir,
            stage_dir="blocking_terms_compiler_v1",
            subdirs=COVENANT_BLOCKING_SUBDIRS,
            item_id=item_id,
            suffix="__compiled.json",
        ),
    )
    _add_first_existing_json(
        "covenant_overlay_recursive",
        "compustat_overlay_v1 (covenant aggregate, recursive)",
        "covenant_overlay_recursive",
        _recursive_compiled_path_candidates(
            run_dir=run_dir,
            stage_dir="compustat_overlay_v1",
            subdirs=COVENANT_OVERLAY_SUBDIRS,
            item_id=item_id,
            suffix="__compustat_overlay.json",
        ),
    )

    _add_json_obj(
        "analysis_export_v2",
        "analysis record summary",
        "analysis_record_summary",
        _analysis_record_summary(record, useful),
        path_hint=None,
    )

    stage_rank = {
        "indexing_v2": 10,
        "retrieval_v2": 20,
        "agreement_metadata": 30,
        "pricing_structured": 40,
        "pricing_metrics_recursive": 50,
        "pricing_blocking_recursive": 55,
        "pricing_overlay_recursive": 58,
        "covenant_structured": 70,
        "covenant_metrics_recursive": 75,
        "covenant_blocking_recursive": 78,
        "covenant_overlay_recursive": 80,
        "analysis_export_v2": 90,
    }
    outputs.sort(key=lambda o: (stage_rank.get(str(o.get("stage")), 999), str(o.get("label", ""))))
    return outputs


def _html_template(*, title: str, embedded_json: str) -> str:
    return f"""<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>{title}</title>
    <style>
      :root {{
        --bg: #0b1020;
        --panel: #101735;
        --border: rgba(255,255,255,0.10);
        --text: rgba(255,255,255,0.92);
        --muted: rgba(255,255,255,0.70);
        --accent: #6ee7ff;
        --accent2: #a78bfa;
        --warn: #fbbf24;
        --bad: #fb7185;
        --good: #34d399;
        --mono: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace;
        --sans: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial;
      }}

      html, body {{
        margin: 0;
        padding: 0;
        height: 100%;
        background: var(--bg);
        color: var(--text);
        font-family: var(--sans);
      }}

      .app {{
        display: grid;
        grid-template-columns: 340px 1fr 1fr;
        grid-template-rows: auto 1fr;
        height: 100vh;
      }}

      header {{
        grid-column: 1 / -1;
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 16px;
        padding: 12px 16px;
        border-bottom: 1px solid var(--border);
        background: linear-gradient(180deg, rgba(255,255,255,0.04), rgba(255,255,255,0.0));
      }}
      .title {{
        display: flex;
        flex-direction: column;
        gap: 2px;
      }}
      .h1 {{
        font-weight: 700;
        font-size: 14px;
        letter-spacing: 0.2px;
      }}
      .sub {{
        font-family: var(--mono);
        font-size: 11px;
        color: var(--muted);
      }}

      .controls {{
        display: flex;
        align-items: center;
        gap: 10px;
      }}

      .chip {{
        display: inline-flex;
        align-items: center;
        gap: 8px;
        padding: 6px 10px;
        border-radius: 999px;
        background: rgba(255,255,255,0.05);
        border: 1px solid var(--border);
        color: var(--text);
        font-size: 12px;
        cursor: pointer;
        user-select: none;
      }}
      .chip.active {{
        border-color: rgba(110,231,255,0.35);
        box-shadow: 0 0 0 1px rgba(110,231,255,0.15) inset;
        color: var(--accent);
      }}

      aside {{
        border-right: 1px solid var(--border);
        background: rgba(255,255,255,0.02);
        overflow: hidden;
      }}
      .aside-inner {{
        display: flex;
        flex-direction: column;
        height: 100%;
      }}
      .search {{
        padding: 10px 10px 6px 10px;
      }}
      input {{
        width: 100%;
        padding: 8px 10px;
        border-radius: 10px;
        border: 1px solid var(--border);
        background: rgba(255,255,255,0.03);
        color: var(--text);
        font-family: var(--mono);
        font-size: 12px;
        outline: none;
      }}
      input:focus {{
        border-color: rgba(110,231,255,0.35);
        box-shadow: 0 0 0 2px rgba(110,231,255,0.12);
      }}

      .item-list {{
        padding: 6px 8px 10px 8px;
        overflow: auto;
        flex: 1;
      }}
      .item {{
        padding: 10px 10px;
        border-radius: 12px;
        border: 1px solid transparent;
        cursor: pointer;
        margin-bottom: 8px;
        background: rgba(255,255,255,0.02);
      }}
      .item:hover {{
        border-color: rgba(255,255,255,0.10);
      }}
      .item.active {{
        border-color: rgba(167,139,250,0.35);
        box-shadow: 0 0 0 1px rgba(167,139,250,0.15) inset;
      }}
      .id {{
        font-family: var(--mono);
        font-size: 12px;
        color: var(--text);
      }}
      .meta {{
        margin-top: 6px;
        display: flex;
        flex-wrap: wrap;
        gap: 6px;
        font-size: 11px;
        color: var(--muted);
        font-family: var(--mono);
      }}
      .badge {{
        padding: 2px 8px;
        border-radius: 999px;
        border: 1px solid var(--border);
        background: rgba(255,255,255,0.03);
      }}
      .badge.good {{
        border-color: rgba(52, 211, 153, 0.35);
        color: var(--good);
      }}
      .badge.warn {{
        border-color: rgba(251, 191, 36, 0.35);
        color: var(--warn);
      }}
      .badge.bad {{
        border-color: rgba(251, 113, 133, 0.35);
        color: var(--bad);
      }}

      .panel {{
        overflow: hidden;
        background: rgba(255,255,255,0.015);
      }}
      .panel-inner {{
        height: 100%;
        display: flex;
        flex-direction: column;
        padding: 12px 12px 12px 12px;
      }}

      .section-title {{
        font-size: 12px;
        color: var(--muted);
        font-family: var(--mono);
        margin-bottom: 8px;
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 10px;
      }}

      .output-controls {{
        display: flex;
        align-items: center;
        gap: 10px;
        margin-bottom: 8px;
      }}
      .output-controls select {{
        flex: 1;
        min-width: 220px;
        padding: 8px 10px;
        border-radius: 10px;
        border: 1px solid var(--border);
        background: rgba(255,255,255,0.03);
        color: var(--text);
        font-family: var(--mono);
        font-size: 12px;
        outline: none;
      }}
      .pathline {{
        font-family: var(--mono);
        font-size: 11px;
        color: var(--muted);
        margin: 4px 0 8px 0;
        overflow: hidden;
        text-overflow: ellipsis;
        white-space: nowrap;
      }}

      pre {{
        margin: 0;
        padding: 12px 12px;
        border: 1px solid var(--border);
        border-radius: 12px;
        background: rgba(0,0,0,0.22);
        overflow: auto;
        font-family: var(--mono);
        font-size: 12px;
        line-height: 1.35;
        white-space: pre-wrap;
      }}

      .anchor-ref {{
        color: var(--accent);
        cursor: pointer;
        text-decoration: underline;
        text-decoration-color: rgba(110,231,255,0.35);
      }}

      .doc-controls {{
        display: flex;
        align-items: center;
        gap: 10px;
        margin-bottom: 8px;
      }}
      .doc-count {{
        font-family: var(--mono);
        font-size: 11px;
        color: var(--muted);
      }}
      #doc-view {{
        overflow: auto;
        padding-right: 2px;
      }}
      .anchor-block {{
        border: 1px solid var(--border);
        border-radius: 12px;
        overflow: hidden;
        background: rgba(0,0,0,0.16);
        margin-bottom: 10px;
      }}
      .anchor-header {{
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 10px;
        padding: 8px 10px;
        background: rgba(255,255,255,0.03);
        border-bottom: 1px solid var(--border);
        font-family: var(--mono);
        font-size: 12px;
        color: var(--muted);
      }}
      .aid {{
        color: var(--text);
        cursor: pointer;
      }}
      .tags {{
        font-size: 11px;
        color: var(--muted);
      }}

      .highlight {{
        outline: 2px solid rgba(110,231,255,0.35);
        box-shadow: 0 0 0 2px rgba(110,231,255,0.12);
      }}

      .modal {{
        position: fixed;
        inset: 0;
        display: none;
        align-items: center;
        justify-content: center;
        background: rgba(3, 8, 20, 0.72);
        z-index: 1000;
        padding: 20px;
      }}
      .modal.open {{
        display: flex;
      }}
      .modal-card {{
        width: min(1200px, calc(100vw - 40px));
        max-height: calc(100vh - 40px);
        border: 1px solid var(--border);
        border-radius: 14px;
        background: var(--panel);
        box-shadow: 0 20px 60px rgba(0,0,0,0.45);
        display: flex;
        flex-direction: column;
        overflow: hidden;
      }}
      .modal-head {{
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 12px;
        padding: 10px 12px;
        border-bottom: 1px solid var(--border);
        background: rgba(255,255,255,0.03);
      }}
      .modal-body {{
        padding: 12px;
        overflow: auto;
      }}
      .modal-body pre {{
        max-height: calc(100vh - 200px);
      }}
    </style>
  </head>
  <body>
    <div class="app" id="app">
      <header>
        <div class="title">
          <div class="h1" id="viewer-title">Run Stage Viewer</div>
          <div class="sub" id="run-id"></div>
        </div>
        <div class="controls">
          <div class="chip" id="toggle-only-cited">Only cited anchors</div>
          <div class="chip" id="toggle-prompts">Show prompts</div>
        </div>
      </header>

      <aside>
        <div class="aside-inner">
          <div class="search">
            <input id="item-filter" placeholder="Filter run_id or item_id..." />
          </div>
          <div class="item-list" id="item-list"></div>
        </div>
      </aside>

      <main class="panel" id="output-panel">
        <div class="panel-inner">
          <div class="section-title" id="output-title">Core LLM output</div>
          <div class="output-controls">
            <select id="output-select"></select>
          </div>
          <div class="pathline" id="output-path"></div>
          <pre id="output-view"></pre>
        </div>
      </main>

      <main class="panel" id="doc-panel">
        <div class="panel-inner">
          <div class="section-title" id="doc-title">Agreement text</div>
          <div class="doc-controls">
            <div class="search" style="flex: 1; min-width: 240px;">
              <input id="doc-filter" placeholder="Filter (anchor id like A0123 or text substring)..." />
            </div>
            <div class="doc-count" id="doc-count"></div>
          </div>
          <div id="doc-view"></div>
        </div>
      </main>
    </div>

    <div id="prompt-modal" class="modal" aria-hidden="true">
      <div class="modal-card">
        <div class="modal-head">
          <div class="section-title" id="prompt-title" style="margin-bottom: 0;">System prompt</div>
          <div class="chip" id="prompt-close">Close</div>
        </div>
        <div class="modal-body">
          <div class="output-controls">
            <select id="prompt-select"></select>
          </div>
          <div class="pathline" id="prompt-path"></div>
          <pre id="prompt-view"></pre>
        </div>
      </div>
    </div>

    <script type="application/json" id="run-data">
{embedded_json}
    </script>
    <script>
      function escapeHtml(s) {{
        return (s || '')
          .replaceAll('&', '&amp;')
          .replaceAll('<', '&lt;')
          .replaceAll('>', '&gt;')
          .replaceAll('\\"', '&quot;')
          .replaceAll("'", '&#39;');
      }}

      function anchorSortKey(aid) {{
        if (/^A\\d{{4,}}$/.test(aid || '')) {{
          return Number.parseInt(aid.slice(1), 10);
        }}
        return Number.MAX_SAFE_INTEGER;
      }}

      function selectDocAnchor(anchorId) {{
        const el = document.getElementById('doc-' + anchorId);
        if (!el) return;
        el.scrollIntoView({{ behavior: 'smooth', block: 'start' }});
        el.classList.add('highlight');
        window.setTimeout(() => el.classList.remove('highlight'), 1200);
      }}

      function renderTextWithAnchorLinks(text) {{
        const safe = escapeHtml(text || '');
        return safe.replace(/A\\d{{4,}}/g, (m) => `<span class="anchor-ref" data-anchor="${{m}}">${{m}}</span>`);
      }}

      const raw = document.getElementById('run-data').textContent;
      const DATA = JSON.parse(raw);
      document.getElementById('run-id').textContent =
        `runs=${{(DATA.run_ids || []).length}}  items=${{Object.keys(DATA.items || {{}}).length}}  useful_only=${{DATA.include_only_useful ? 'true' : 'false'}}  useful_mode=${{DATA.useful_mode || 'definitions'}}`;

      let state = {{
        selectedKey: null,
        selectedOutputId: null,
        selectedPromptStage: null,
        itemFilter: '',
        docFilter: '',
        onlyCited: false,
        showPrompts: false,
      }};

      const CORE_STAGES = new Set([
        'indexing_v2',
        'agreement_metadata',
        'pricing_structured',
        'pricing_metrics_recursive',
        'pricing_overlay_recursive',
        'covenant_structured',
        'covenant_metrics_recursive',
        'covenant_overlay_recursive',
      ]);

      function isCoreStage(stage) {{
        return CORE_STAGES.has(stage || '');
      }}

      function getSortedKeys() {{
        return Object.keys(DATA.items || {{}}).sort((a, b) => {{
          const ia = DATA.items[a] || {{}};
          const ib = DATA.items[b] || {{}};
          const ra = ia.run_id || '';
          const rb = ib.run_id || '';
          if (ra !== rb) return ra.localeCompare(rb);
          return (ia.item_id || '').localeCompare(ib.item_id || '');
        }});
      }}

      function getSelectedItem() {{
        if (!state.selectedKey) {{
          const keys = getSortedKeys();
          state.selectedKey = keys.length ? keys[0] : null;
        }}
        return state.selectedKey ? DATA.items[state.selectedKey] : null;
      }}

      function getSelectedOutput(item, outputList = null) {{
        if (!item) return null;
        const outs = Array.isArray(outputList) ? outputList : (item.outputs || []);
        if (!outs.length) return null;
        if (!state.selectedOutputId || !outs.some((o) => o.output_id === state.selectedOutputId)) {{
          const preferred = item.default_output_id;
          if (preferred && outs.some((o) => o.output_id === preferred)) {{
            state.selectedOutputId = preferred;
          }} else {{
            state.selectedOutputId = outs[0].output_id;
          }}
        }}
        return outs.find((o) => o.output_id === state.selectedOutputId) || outs[0];
      }}

      function stageBadge(stage) {{
        const map = {{
          indexing_v2: 'index',
          retrieval_v2: 'retrieval',
          agreement_metadata: 'metadata',
          pricing_structured: 'pricing',
          pricing_metrics_recursive: 'pricing_metrics_rec',
          pricing_blocking_recursive: 'pricing_terms_rec',
          pricing_overlay_recursive: 'pricing_overlay_rec',
          covenant_structured: 'covenant',
          covenant_metrics_recursive: 'covenant_metrics_rec',
          covenant_blocking_recursive: 'covenant_terms_rec',
          covenant_overlay_recursive: 'covenant_overlay_rec',
          analysis_export_v2: 'analysis'
        }};
        return map[stage] || stage;
      }}

      function promptStageOrder() {{
        return [
          'indexing_v2',
          'agreement_metadata',
          'pricing_structured',
          'pricing_metrics_recursive',
          'pricing_blocking_recursive',
          'pricing_overlay_recursive',
          'covenant_structured',
          'covenant_metrics_recursive',
          'covenant_blocking_recursive',
          'covenant_overlay_recursive',
        ];
      }}

      function setPromptModalOpen(open) {{
        state.showPrompts = !!open;
        const modal = document.getElementById('prompt-modal');
        const btn = document.getElementById('toggle-prompts');
        modal.classList.toggle('open', state.showPrompts);
        modal.setAttribute('aria-hidden', state.showPrompts ? 'false' : 'true');
        btn.classList.toggle('active', state.showPrompts);
      }}

      function renderItemList() {{
        const box = document.getElementById('item-list');
        box.innerHTML = '';

        const filter = (state.itemFilter || '').trim().toLowerCase();
        const keys = getSortedKeys();
        for (const k of keys) {{
          const it = DATA.items[k];
          const runId = it.run_id || '';
          const itemId = it.item_id || '';
          const keyText = `${{runId}}::${{itemId}}`.toLowerCase();
          if (filter && !keyText.includes(filter)) continue;

          const outs = it.outputs || [];
          const stages = new Set(outs.map((o) => o.stage));
          const unknown = outs.reduce((acc, o) => acc + ((o.unknown_anchor_ids || []).length ? 1 : 0), 0);
          const useful = it.useful_flags || {{}};

          const div = document.createElement('div');
          div.className = 'item' + (k === state.selectedKey ? ' active' : '');
          div.addEventListener('click', () => {{
            state.selectedKey = k;
            state.selectedOutputId = null;
            render();
          }});

          const idDiv = document.createElement('div');
          idDiv.className = 'id';
          idDiv.textContent = `${{runId}} :: ${{itemId}}`;

          const meta = document.createElement('div');
          meta.className = 'meta';

          const sortedStages = Array.from(stages).sort();
          const displayStages = sortedStages.filter((s) => isCoreStage(s));
          const stageChips = displayStages.length ? displayStages : sortedStages;
          for (const s of stageChips) {{
            const b = document.createElement('div');
            b.className = 'badge';
            b.textContent = stageBadge(s);
            meta.appendChild(b);
          }}

          const badgeMap = [
            ['PS', useful.pricing_structured],
            ['PD', useful.pricing_definitions],
            ['CS', useful.covenant_structured],
            ['CD', useful.covenant_definitions],
          ];
          for (const [label, ok] of badgeMap) {{
            const b = document.createElement('div');
            b.className = 'badge ' + (ok ? 'good' : 'warn');
            b.textContent = label;
            meta.appendChild(b);
          }}

          const b2 = document.createElement('div');
          b2.className = 'badge ' + ((it.useful_flags && it.useful_flags.any_downstream) ? 'good' : 'warn');
          b2.textContent = 'useful=' + ((it.useful_flags && it.useful_flags.any_downstream) ? '1' : '0');
          meta.appendChild(b2);
          if (unknown) {{
            const b3 = document.createElement('div');
            b3.className = 'badge bad';
            b3.textContent = `unknown_refs=${{unknown}}`;
            meta.appendChild(b3);
          }}

          div.appendChild(idDiv);
          div.appendChild(meta);
          box.appendChild(div);
        }}
      }}

      function renderOutputPanel() {{
        const item = getSelectedItem();
        const title = document.getElementById('output-title');
        const pathLine = document.getElementById('output-path');
        const pre = document.getElementById('output-view');
        const sel = document.getElementById('output-select');

        sel.innerHTML = '';
        if (!item) {{
          title.textContent = 'Core LLM output';
          pathLine.textContent = '';
          pre.textContent = '(no items)';
          renderPromptPanel(null);
          return;
        }}

        const allOuts = item.outputs || [];
        const coreOuts = allOuts.filter((o) => isCoreStage(o.stage));
        const outs = coreOuts.length ? coreOuts : allOuts;

        if (!outs.length) {{
          title.textContent = `Core LLM output :: ${{item.run_id}} :: ${{item.item_id}}`;
          pathLine.textContent = '';
          pre.textContent = '(no outputs found)';
          renderPromptPanel(null);
          return;
        }}

        const byStage = new Map();
        for (const o of outs) {{
          if (!byStage.has(o.stage)) byStage.set(o.stage, []);
          byStage.get(o.stage).push(o);
        }}

        const stageOrder = [
          'indexing_v2',
          'agreement_metadata',
          'pricing_structured',
          'pricing_metrics_recursive',
          'pricing_blocking_recursive',
          'pricing_overlay_recursive',
          'covenant_structured',
          'covenant_metrics_recursive',
          'covenant_blocking_recursive',
          'covenant_overlay_recursive',
        ];
        const stages = Array.from(byStage.keys()).sort((a, b) => {{
          const ai = stageOrder.indexOf(a);
          const bi = stageOrder.indexOf(b);
          const av = ai === -1 ? 999 : ai;
          const bv = bi === -1 ? 999 : bi;
          if (av !== bv) return av - bv;
          return String(a || '').localeCompare(String(b || ''));
        }});
        for (const st of stages) {{
          const grp = document.createElement('optgroup');
          grp.label = st;
          const arr = (byStage.get(st) || []).slice().sort((a,b) => (a.label || '').localeCompare(b.label || ''));
          for (const o of arr) {{
            const opt = document.createElement('option');
            opt.value = o.output_id;
            opt.textContent = o.label;
            grp.appendChild(opt);
          }}
          sel.appendChild(grp);
        }}

        const out = getSelectedOutput(item, outs);
        sel.value = out ? out.output_id : (outs[0].output_id || '');
        sel.onchange = () => {{
          state.selectedOutputId = sel.value;
          renderOutputPanel();
          renderDocPanel();
        }};

        title.textContent = `Core LLM output :: ${{item.run_id}} :: ${{item.item_id}}`;
        pathLine.textContent = out ? (out.path || '') : '';

        let rawText = '';
        if (!out) {{
          rawText = '(no output selected)';
        }} else if (out.kind === 'json') {{
          rawText = JSON.stringify(out.json, null, 2);
        }} else {{
          rawText = out.text || '';
        }}

        if (out && out.unknown_anchor_ids && out.unknown_anchor_ids.length) {{
          rawText =
            `=== WARNING: unknown anchor ids (not found in this agreement) ===\\n` +
            out.unknown_anchor_ids.sort((a,b) => anchorSortKey(a) - anchorSortKey(b)).join(', ') +
            `\\n\\n` + rawText;
        }}

        pre.innerHTML = renderTextWithAnchorLinks(rawText);
        pre.querySelectorAll('.anchor-ref').forEach((el) => {{
          el.addEventListener('click', () => selectDocAnchor(el.dataset.anchor));
        }});
        renderPromptPanel(out);
      }}

      function renderPromptPanel(selectedOutput) {{
        const prompts = DATA.stage_prompts || {{}};
        const title = document.getElementById('prompt-title');
        const sel = document.getElementById('prompt-select');
        const pathLine = document.getElementById('prompt-path');
        const pre = document.getElementById('prompt-view');

        sel.innerHTML = '';
        const stages = Object.keys(prompts);
        if (!stages.length) {{
          title.textContent = 'System prompt';
          pathLine.textContent = '';
          pre.textContent = '(no stage prompt files found)';
          return;
        }}

        const order = promptStageOrder();
        const sortedStages = stages.slice().sort((a, b) => {{
          const ai = order.indexOf(a);
          const bi = order.indexOf(b);
          const av = ai === -1 ? 999 : ai;
          const bv = bi === -1 ? 999 : bi;
          if (av !== bv) return av - bv;
          return String(a || '').localeCompare(String(b || ''));
        }});

        if (selectedOutput && selectedOutput.stage && prompts[selectedOutput.stage]) {{
          state.selectedPromptStage = selectedOutput.stage;
        }}
        if (!state.selectedPromptStage || !prompts[state.selectedPromptStage]) {{
          state.selectedPromptStage = sortedStages[0];
        }}

        for (const st of sortedStages) {{
          const opt = document.createElement('option');
          opt.value = st;
          opt.textContent = `${{stageBadge(st)}} :: ${{st}}`;
          sel.appendChild(opt);
        }}
        sel.value = state.selectedPromptStage;
        sel.onchange = () => {{
          state.selectedPromptStage = sel.value;
          renderPromptPanel(null);
        }};

        const chosen = prompts[state.selectedPromptStage] || {{}};
        title.textContent = `System prompt :: ${{stageBadge(state.selectedPromptStage)}}`;
        const pathParts = [];
        if (chosen.path) pathParts.push(chosen.path);
        const companions = Array.isArray(chosen.companion_paths) ? chosen.companion_paths : [];
        for (const cp of companions) pathParts.push(`companion: ${{cp}}`);
        pathLine.textContent = pathParts.join('  |  ');
        pre.textContent = chosen.text || '(prompt file is empty)';
      }}

      function renderDocPanel() {{
        const item = getSelectedItem();
        const out = getSelectedOutput(item);
        const docBox = document.getElementById('doc-view');
        const docTitle = document.getElementById('doc-title');
        const docCount = document.getElementById('doc-count');
        docBox.innerHTML = '';

        if (!item) {{
          docTitle.textContent = 'Agreement text';
          docCount.textContent = '';
          return;
        }}

        const blocks = item.doc_blocks || [];
        const filter = (state.docFilter || '').trim().toLowerCase();
        const cited = new Set((out && out.cited_anchor_ids) ? out.cited_anchor_ids : []);

        docTitle.textContent = `Agreement text :: ${{item.run_id}} :: ${{item.item_id}}`;

        const frag = document.createDocumentFragment();
        let shown = 0;
        for (const b of blocks) {{
          const aid = b.anchor_id;
          const text = b.text || '';
          const isRealAnchor = /^A\\d{{4,}}$/.test(aid);
          if (state.onlyCited && isRealAnchor && !cited.has(aid)) continue;
          if (filter) {{
            const hayA = (aid || '').toLowerCase();
            const hayT = text.toLowerCase();
            if (!(hayA.includes(filter) || hayT.includes(filter))) continue;
          }}

          const wrapper = document.createElement('div');
          wrapper.className = 'anchor-block';
          if (isRealAnchor) wrapper.id = 'doc-' + aid;

          const header = document.createElement('div');
          header.className = 'anchor-header';

          const left = document.createElement('div');
          left.className = 'aid';
          left.textContent = aid;
          if (isRealAnchor) left.addEventListener('click', () => selectDocAnchor(aid));

          const right = document.createElement('div');
          right.className = 'tags';
          if (isRealAnchor) {{
            right.textContent = state.onlyCited ? 'cited' : (cited.has(aid) ? 'cited' : '');
          }} else {{
            right.textContent = 'preface';
          }}

          header.appendChild(left);
          header.appendChild(right);

          const pre = document.createElement('pre');
          pre.innerHTML = escapeHtml(text);

          wrapper.appendChild(header);
          wrapper.appendChild(pre);
          frag.appendChild(wrapper);
          shown += 1;
        }}
        docBox.appendChild(frag);
        const citedLabel = state.onlyCited ? ' (cited only)' : '';
        const citedCount = cited.size ? ` · cited=${{cited.size}}` : '';
        docCount.textContent = `showing ${{shown}} / ${{blocks.length}} blocks${{citedLabel}}${{citedCount}}`;
      }}

      document.getElementById('toggle-only-cited').addEventListener('click', () => {{
        state.onlyCited = !state.onlyCited;
        document.getElementById('toggle-only-cited').classList.toggle('active', state.onlyCited);
        renderDocPanel();
      }});

      document.getElementById('toggle-prompts').addEventListener('click', () => {{
        setPromptModalOpen(!state.showPrompts);
        renderPromptPanel(getSelectedOutput(getSelectedItem()));
      }});

      document.getElementById('prompt-close').addEventListener('click', () => {{
        setPromptModalOpen(false);
      }});

      document.getElementById('prompt-modal').addEventListener('click', (e) => {{
        if (e.target && e.target.id === 'prompt-modal') {{
          setPromptModalOpen(false);
        }}
      }});

      document.addEventListener('keydown', (e) => {{
        if (e.key === 'Escape' && state.showPrompts) {{
          setPromptModalOpen(false);
        }}
      }});

      document.getElementById('item-filter').addEventListener('input', (e) => {{
        state.itemFilter = e.target.value || '';
        renderItemList();
      }});

      document.getElementById('doc-filter').addEventListener('input', (e) => {{
        state.docFilter = e.target.value || '';
        renderDocPanel();
      }});

      function render() {{
        renderItemList();
        renderOutputPanel();
        renderDocPanel();
      }}

      render();
    </script>
  </body>
</html>
"""


def _expand_run_ids(raw: Iterable[str]) -> list[str]:
    out: list[str] = []
    for s in raw:
        if not isinstance(s, str):
            continue
        for part in s.split(","):
            v = part.strip()
            if v:
                out.append(v)
    return out


def _select_run_dirs(*, root: Path, run_ids: Sequence[str], run_glob: Optional[str]) -> list[Path]:
    dirs: list[Path] = []
    if run_ids:
        for rid in run_ids:
            p = root / "runs" / rid
            if not p.exists():
                raise FileNotFoundError(f"Run dir not found for run_id={rid}: {p}")
            dirs.append(p)
    if run_glob:
        for p in sorted((root / "runs").glob(run_glob)):
            if p.is_dir():
                dirs.append(p)

    uniq: list[Path] = []
    seen: set[str] = set()
    for p in dirs:
        k = str(p.resolve())
        if k in seen:
            continue
        seen.add(k)
        uniq.append(p)
    return uniq


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-dir", default=".", help="Project root containing runs/")
    ap.add_argument("--run-id", action="append", default=[], help="Run id to include; can be repeated or comma-separated.")
    ap.add_argument("--run-glob", default=None, help="Glob under runs/ (example: 'cea100-credit-cg-20260211c-s00*').")
    ap.add_argument("--analysis-subdir", default=None, help="Analysis subdir name (for records.jsonl detection).")
    ap.add_argument(
        "--useful-mode",
        default="definitions",
        choices=["definitions", "structured_or_definitions"],
        help=(
            "How to decide whether an agreement is useful downstream. "
            "'definitions' (default) keeps agreements with usable pricing/covenant definitions rows. "
            "'structured_or_definitions' also counts non-empty structured pricing/covenant outputs."
        ),
    )
    ap.add_argument("--include-all", action="store_true", help="Include non-useful agreements too.")
    ap.add_argument("--max-items", type=int, default=0, help="If >0, cap included agreements after sorting.")
    ap.add_argument("--out", default=None, help="Output HTML path.")
    args = ap.parse_args()

    root = Path(args.base_dir)
    if not root.exists():
        raise SystemExit(f"base-dir not found: {root}")

    run_ids = _expand_run_ids(args.run_id)
    run_dirs = _select_run_dirs(root=root, run_ids=run_ids, run_glob=args.run_glob)
    if not run_dirs:
        raise SystemExit("No run dirs selected. Provide --run-id and/or --run-glob.")

    include_only_useful = not bool(args.include_all)
    items_payload: dict[str, Any] = {}
    total_records = 0
    useful_records = 0

    for run_dir in run_dirs:
        run_id = run_dir.name
        paths = Paths(root=root, run_id=run_id)
        manifest = _load_manifest(run_dir)
        records_path = _choose_records_path(run_dir=run_dir, manifest=manifest, analysis_subdir=args.analysis_subdir)
        records = _load_records(records_path)

        for record in records:
            total_records += 1
            item_id = str(record.get("item_id") or "").strip()
            if not item_id:
                raise RuntimeError(f"Invalid record without item_id in {records_path}")

            useful = _useful_flags(
                record,
                useful_mode=str(args.useful_mode),
                run_dir=run_dir,
                item_id=item_id,
            )
            if useful.get("any_downstream"):
                useful_records += 1
            if include_only_useful and not useful.get("any_downstream", False):
                continue

            try:
                doc_source, blocks = _load_document_blocks(paths, item_id)
            except Exception as exc:
                raise RuntimeError(f"Failed to load agreement text for run_id={run_id} item_id={item_id}: {exc}") from exc

            doc_anchor_ids = {
                b["anchor_id"]
                for b in blocks
                if isinstance(b.get("anchor_id"), str) and ANCHOR_ID_RE.fullmatch(str(b["anchor_id"]))
            }
            outputs = _load_outputs_for_record(
                root=root,
                run_dir=run_dir,
                record=record,
                useful=useful,
                doc_anchor_ids=doc_anchor_ids,
            )

            key = f"{run_id}::{item_id}"
            items_payload[key] = {
                "key": key,
                "run_id": run_id,
                "item_id": item_id,
                "doc_source_path": doc_source,
                "doc_blocks": blocks,
                "outputs": outputs,
                "default_output_id": _default_output_id(outputs),
                "useful_flags": useful,
            }

    keys_sorted = sorted(
        items_payload.keys(),
        key=lambda k: (str(items_payload[k].get("run_id", "")), str(items_payload[k].get("item_id", ""))),
    )
    if args.max_items and int(args.max_items) > 0:
        keys_sorted = keys_sorted[: int(args.max_items)]
        items_payload = {k: items_payload[k] for k in keys_sorted}

    if not items_payload:
        raise SystemExit(
            "No agreements selected after filtering. "
            "Try --include-all or verify analysis records contain useful downstream outputs."
        )

    if args.out:
        out_path = Path(args.out)
    elif len(run_dirs) == 1:
        out_path = run_dirs[0] / "run_stage_viewer.html"
    else:
        slug = _safe_slug(args.run_glob or "multi_run")
        out_path = root / "runs" / f"run_stage_viewer_{slug}.html"

    payload = {
        "viewer": "run_stage_viewer_html",
        "generated_at": int(time.time()),
        "run_ids": [p.name for p in run_dirs],
        "include_only_useful": include_only_useful,
        "useful_mode": str(args.useful_mode),
        "total_records_seen": total_records,
        "useful_records_seen": useful_records,
        "stage_prompts": _resolve_stage_prompts(root),
        "items": items_payload,
    }
    embedded = json.dumps(payload, indent=2, sort_keys=True)
    embedded = embedded.replace("</script>", "<\\/script>")

    html = _html_template(title="Run Stage Viewer", embedded_json=embedded)
    _write_text(out_path, html)
    print(str(out_path))
    print(
        f"selected_items={len(items_payload)} total_records_seen={total_records} "
        f"useful_records_seen={useful_records} useful_only={include_only_useful} useful_mode={args.useful_mode}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
