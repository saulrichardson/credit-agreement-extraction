#!/usr/bin/env python3
"""
Compare prompt_v1_short extractions vs a local sample baseline (handcheck/openllm packs).

This is the analogue of scripts/compare_covenantir_to_local_sample_baseline.py but for
the simpler prompt_v1_short JSON LIST output.

Inputs
------
- runs/<run_id>/local_sample_manifest.json
    baseline fields per item: name/limit/start/end
- runs/<run_id>/retrieval_v2/<item_id>_snippets.jsonl
    snippet text used as the model input
- runs/<run_id>/llm_qa/<prompt_subdir>/<item_id>.txt
    raw model output (should be a JSON list)

Output
------
- JSON report with per-item disposition + aggregate counts, written to --out.

Notes
-----
This script is intentionally strict:
- The model output must parse as a JSON LIST. Otherwise we mark invalid_json and move on.
- Matching is evaluation-only: name identity overlap + baseline numeric limit match.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8", errors="replace"))


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")


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


def _norm_ws(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


_TOKEN_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)


def _tokenize(text: str) -> List[str]:
    return [m.group(0).lower() for m in _TOKEN_RE.finditer(text or "")]


_GENERIC_TOKENS = {
    "ratio",
    "minimum",
    "maximum",
    "consolidated",
    "leverage",
    "coverage",
    "net",
    "worth",
    "debt",
    "ebitda",
}


def _identity_tokens_from_baseline(baseline_name: str) -> List[str]:
    toks = [t for t in _tokenize(baseline_name) if len(t) >= 4]
    out: List[str] = []
    for t in toks:
        if t in _GENERIC_TOKENS:
            continue
        if t not in out:
            out.append(t)
    # If everything was filtered away, fall back to original >=4-char tokens.
    return out or toks


def _identity_overlap_tokens(*, baseline_tokens: Sequence[str], candidate_tokens: Sequence[str]) -> List[str]:
    bt = set(baseline_tokens)
    ct = set(candidate_tokens)
    overlap = sorted(bt & ct)
    return overlap


def _parse_decimal_like(value: Any) -> Optional[Decimal]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        try:
            return Decimal(str(value)).quantize(Decimal("0.0001"))
        except Exception:
            return None
    if not isinstance(value, str):
        value = str(value)
    s = _norm_ws(value)
    if not s or s.lower() in {"na", "n/a", "none", "<na>", "null"}:
        return None
    # Remove currency symbols and commas.
    s = s.replace(",", "")
    s = re.sub(r"^\$", "", s)
    # Some sources contain ratio suffix like ":1" or "to 1.00"—take the leading number.
    m = re.match(r"^(-?\d+(?:\.\d+)?)", s)
    if not m:
        return None
    try:
        return Decimal(m.group(1)).quantize(Decimal("0.0001"))
    except (InvalidOperation, ValueError):
        return None


def _limit_matches(*, baseline_limit: Decimal, extracted_limit: Decimal) -> bool:
    if extracted_limit == baseline_limit:
        return True
    # Allow percent<->decimal normalization (evaluation tolerance).
    if baseline_limit > 1 and baseline_limit <= 100:
        as_decimal = (baseline_limit / Decimal("100")).quantize(Decimal("0.0001"))
        if extracted_limit == as_decimal:
            return True
    if baseline_limit >= 0 and baseline_limit < 1:
        as_percent = (baseline_limit * Decimal("100")).quantize(Decimal("0.0001"))
        if extracted_limit == as_percent:
            return True
    return False


def _preview(text: str, limit: int = 260) -> str:
    t = _norm_ws(text)
    if len(t) <= limit:
        return t
    return t[: limit - 3] + "..."


def _limit_in_text(*, limit: Decimal, text: str, required_keywords: Sequence[str]) -> Tuple[bool, Optional[str]]:
    """Check if a baseline numeric limit appears in the snippet, with required keywords gating.

    This is used only for evaluation bucketizing (present-in-snippet vs not).
    """

    if not text or not required_keywords:
        return (False, None)

    tlow = _norm_ws(text).lower()
    if not all(k.lower() in tlow for k in required_keywords[:3]):
        # Require at least a few baseline name keywords to reduce false positives.
        return (False, None)

    # Direct numeric forms.
    candidates = {str(limit), f"{limit:.2f}", f"{limit:.3f}", f"{limit:.0f}"}

    # Percent/decimal alternates.
    if limit > 1 and limit <= 100:
        as_decimal = (limit / Decimal("100")).quantize(Decimal("0.0001"))
        candidates.update({str(as_decimal), f"{as_decimal:.4f}", f"{as_decimal:.2f}"})
    if limit >= 0 and limit < 1:
        as_percent = (limit * Decimal("100")).quantize(Decimal("0.0001"))
        candidates.update({str(as_percent), f"{as_percent:.2f}", f"{as_percent:.4f}"})

    for c in sorted({c for c in candidates if c}):
        if c in text:
            return (True, "literal")
    return (False, None)


@dataclass(frozen=True)
class ExtractedRow:
    name: str
    limit: Optional[Decimal]
    start_date: str
    end_date: str


def _parse_extracted_list(raw_text: str) -> Tuple[Optional[List[ExtractedRow]], Optional[str]]:
    """Return (parsed, error_kind)."""
    try:
        obj = json.loads(raw_text)
    except Exception:
        return (None, "invalid_json")
    if not isinstance(obj, list):
        return (None, "not_a_list")
    out: List[ExtractedRow] = []
    for el in obj:
        if not isinstance(el, dict):
            continue
        name = _norm_ws(el.get("name") or "")
        limit = _parse_decimal_like(el.get("limit"))
        start_date = _norm_ws(el.get("start date") or "")
        end_date = _norm_ws(el.get("end date") or "")
        out.append(ExtractedRow(name=name, limit=limit, start_date=start_date, end_date=end_date))
    return (out, None)


def _summarize_item(
    *,
    item: Mapping[str, Any],
    run_dir: Path,
    prompt_subdir: str,
) -> Dict[str, Any]:
    item_id = str(item.get("item_id") or "")
    baseline = item.get("baseline") if isinstance(item.get("baseline"), dict) else {}

    baseline_name = _norm_ws(str(baseline.get("name") or ""))
    baseline_limit_raw = _norm_ws(str(baseline.get("limit") or ""))
    baseline_limit = _parse_decimal_like(baseline_limit_raw)

    retrieval_dir = run_dir / "retrieval_v2"
    snippet_text = ""
    snippet_path = retrieval_dir / f"{item_id}_snippets.jsonl"
    if snippet_path.exists():
        for rec in _iter_jsonl(snippet_path):
            snip = rec.get("snippet")
            if isinstance(snip, str) and snip.strip():
                snippet_text = snip
                break

    baseline_tokens = _identity_tokens_from_baseline(baseline_name)
    baseline_name_in_excerpt = bool(baseline_name and snippet_text and baseline_name.lower() in snippet_text.lower())

    baseline_limit_in_excerpt = False
    baseline_limit_in_excerpt_kind: Optional[str] = None
    if baseline_limit is not None and snippet_text:
        baseline_limit_in_excerpt, baseline_limit_in_excerpt_kind = _limit_in_text(
            limit=baseline_limit,
            text=snippet_text,
            required_keywords=baseline_tokens,
        )

    llm_dir = run_dir / "llm_qa" / prompt_subdir
    raw_path = llm_dir / f"{item_id}.txt"
    err_path = llm_dir / f"{item_id}.error.txt"

    out: Dict[str, Any] = {
        "item_id": item_id,
        "baseline": {
            "name": baseline_name,
            "limit_raw": baseline_limit_raw,
            "limit_decimal": str(baseline_limit) if baseline_limit is not None else None,
        },
        "excerpt": {
            "snippet_found": bool(snippet_text),
            "snippet_len": len(snippet_text) if snippet_text else 0,
            "snippet_preview": _preview(snippet_text, limit=260) if snippet_text else "",
            "baseline_name_in_excerpt": baseline_name_in_excerpt,
            "baseline_limit_in_excerpt": baseline_limit_in_excerpt,
            "baseline_limit_in_excerpt_kind": baseline_limit_in_excerpt_kind,
        },
        "prompt_v1_short": {
            "status": "missing_output",
            "raw_path": str(raw_path),
            "error_path": str(err_path),
            "parsed_count": None,
            "parse_error_kind": None,
        },
        "match": {
            "baseline_identity_tokens": baseline_tokens,
            "baseline_identity_ok_any": False,
            "baseline_identity_overlap_best": [],
            "baseline_limit_found_any": False,
            "baseline_limit_found_best": False,
            "baseline_limit_match_kind_best": None,
        },
        "candidates": [],
        "best_match": None,
        "disposition": None,
    }

    if err_path.exists():
        out["prompt_v1_short"]["status"] = "error"
        out["disposition"] = "error_gateway"
        return out
    if not raw_path.exists():
        return out

    raw_text = raw_path.read_text(encoding="utf-8", errors="replace")
    parsed, parse_err = _parse_extracted_list(raw_text)
    if parsed is None:
        out["prompt_v1_short"]["status"] = "invalid_output"
        out["prompt_v1_short"]["parse_error_kind"] = parse_err
        out["disposition"] = "error_invalid_json"
        return out

    out["prompt_v1_short"]["status"] = "ok"
    out["prompt_v1_short"]["parsed_count"] = len(parsed)

    # Build candidate summaries and compute matches.
    best: Optional[ExtractedRow] = None
    best_score: Tuple[int, int, int] | None = None
    candidates: List[Dict[str, Any]] = []
    for r in parsed:
        overlap = _identity_overlap_tokens(baseline_tokens=baseline_tokens, candidate_tokens=_tokenize(r.name))
        name_score = len(overlap)
        limit_match = False
        if baseline_limit is not None and r.limit is not None:
            limit_match = _limit_matches(baseline_limit=baseline_limit, extracted_limit=r.limit)
        score = (name_score, int(limit_match), len(_tokenize(r.name)))
        if best_score is None or score > best_score:
            best_score = score
            best = r
        candidates.append(
            {
                "name": r.name,
                "limit": str(r.limit) if r.limit is not None else None,
                "start_date": r.start_date,
                "end_date": r.end_date,
                "identity_overlap_tokens": overlap,
                "name_score": name_score,
                "limit_match": bool(limit_match),
            }
        )

        if overlap:
            out["match"]["baseline_identity_ok_any"] = True
        if baseline_limit is not None and r.limit is not None and _limit_matches(baseline_limit=baseline_limit, extracted_limit=r.limit):
            out["match"]["baseline_limit_found_any"] = True

    # Keep top candidates for inspection.
    candidates_sorted = sorted(candidates, key=lambda c: (int(c["name_score"]), int(c["limit_match"]), len(_tokenize(c["name"]))), reverse=True)
    out["candidates"] = candidates_sorted[:10]

    if best is not None:
        overlap_best = _identity_overlap_tokens(baseline_tokens=baseline_tokens, candidate_tokens=_tokenize(best.name))
        out["match"]["baseline_identity_overlap_best"] = overlap_best
        out["best_match"] = {
            "name": best.name,
            "limit": str(best.limit) if best.limit is not None else None,
            "start_date": best.start_date,
            "end_date": best.end_date,
        }
        if baseline_limit is not None and best.limit is not None and _limit_matches(baseline_limit=baseline_limit, extracted_limit=best.limit):
            out["match"]["baseline_limit_found_best"] = True
            out["match"]["baseline_limit_match_kind_best"] = "exact_or_percent_decimal"

    # Final disposition.
    if len(parsed) == 0:
        out["disposition"] = "ok_zero_covenants"
        return out

    if baseline_limit is None:
        out["disposition"] = "ok_no_baseline_limit"
        return out

    # Strict success: any row where BOTH identity tokens overlap and limit matches.
    baseline_matched_any = False
    for c in candidates:
        if not c.get("identity_overlap_tokens"):
            continue
        if c.get("limit_match") is True:
            baseline_matched_any = True
            break

    if baseline_matched_any:
        out["disposition"] = "ok_baseline_matched"
    elif out["match"]["baseline_limit_found_any"] is True:
        out["disposition"] = "ok_baseline_limit_matched_wrong_covenant"
    else:
        if out.get("excerpt", {}).get("baseline_limit_in_excerpt") is True:
            out["disposition"] = "ok_baseline_limit_missing_despite_in_excerpt"
        else:
            out["disposition"] = "ok_baseline_limit_not_in_excerpt"

    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True, help="Local sample run_id under runs/<run-id>/")
    ap.add_argument("--prompt-subdir", default="prompt_v1_short", help="Subdir under runs/<run-id>/llm_qa/")
    ap.add_argument("--base-dir", default=".", help="Repo base dir")
    ap.add_argument("--out", required=True, help="Path to write JSON report")
    args = ap.parse_args()

    run_dir = Path(args.base_dir) / "runs" / args.run_id
    local_manifest_path = run_dir / "local_sample_manifest.json"
    if not local_manifest_path.exists():
        raise SystemExit(f"Missing local_sample_manifest.json: {local_manifest_path}")
    if not (run_dir / "retrieval_v2").exists():
        raise SystemExit(f"Missing retrieval_v2 dir: {run_dir / 'retrieval_v2'}")

    local = _read_json(local_manifest_path)
    items = local.get("items")
    if not isinstance(items, list) or not items:
        raise SystemExit(f"Manifest items missing/empty: {local_manifest_path}")

    per_item: List[Dict[str, Any]] = []
    for it in items:
        if not isinstance(it, dict):
            continue
        per_item.append(_summarize_item(item=it, run_dir=run_dir, prompt_subdir=str(args.prompt_subdir)))

    counts_by_status: Dict[str, int] = {}
    counts_by_disp: Dict[str, int] = {}
    match_counts: Dict[str, int] = {
        "baseline_identity_ok_any": 0,
        "baseline_limit_found_any": 0,
        "baseline_limit_found_best": 0,
    }
    for r in per_item:
        status = str(((r.get("prompt_v1_short") or {}) if isinstance(r.get("prompt_v1_short"), dict) else {}).get("status") or "unknown")
        counts_by_status[status] = counts_by_status.get(status, 0) + 1
        disp = str(r.get("disposition") or "unknown")
        counts_by_disp[disp] = counts_by_disp.get(disp, 0) + 1
        m = r.get("match")
        if isinstance(m, dict):
            for k in match_counts:
                if m.get(k) is True:
                    match_counts[k] += 1

    report = {
        "schema_version": "prompt_v1_short_vs_local_baseline_report_v1",
        "run_id": args.run_id,
        "prompt_subdir": str(args.prompt_subdir),
        "counts_by_status": counts_by_status,
        "counts_by_disposition": counts_by_disp,
        "match_counts": match_counts,
        "items": per_item,
    }
    _write_json(Path(args.out), report)
    print(f"Wrote report: {args.out}")
    print(f"Counts by status: {counts_by_status}")
    print(f"Counts by disposition: {counts_by_disp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

