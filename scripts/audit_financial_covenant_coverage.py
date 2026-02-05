#!/usr/bin/env python3
"""
Audit what *financial covenant* patterns actually appear in a real run's retrieval_v2 snippet packs.

Goal
----
Make the "Option A (financial covenants only)" scope concrete and evidence-based:
  - What kinds of covenant language show up?
  - How often do we see schedules, springing triggers, "right to cure", etc.?
  - Which anchor_ids are representative examples for each pattern bucket?

This script is intentionally heuristic. It does NOT attempt to fully parse covenants; it just
provides an artifact-first coverage report to guide:
  - prompt improvements
  - schema/operator extensions (if needed)
  - validation priorities

Usage
-----
  poetry run python scripts/audit_financial_covenant_coverage.py \
    --run-id dan-v2-20260106 \
    --out scratch/financial_covenant_coverage__dan-v2-20260106.json
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


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


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _shorten(text: str, *, limit: int) -> str:
    t = " ".join(text.strip().split())
    if len(t) <= limit:
        return t
    return t[: max(0, limit - 1)] + "…"


_RE_TABLE = re.compile(r"\[\[TABLE\]\]", re.IGNORECASE)
_RE_SPRINGING = re.compile(r"\bIf at any time\b|\bso long as\b.*\bshall\b", re.IGNORECASE | re.DOTALL)
_RE_CURE = re.compile(r"\bRight to Cure\b|\bCure\b.*\bcompliance\b", re.IGNORECASE)
_RE_SCHEDULE_DATE_PHRASES = re.compile(r"\bthrough\b|\band thereafter\b|\bcommencing\b", re.IGNORECASE)
_RE_PROHIBITION = re.compile(r"\bshall not permit\b|\bPermit\b.*\bto be\b", re.IGNORECASE | re.DOTALL)
_RE_COMPARATOR_PHRASES = re.compile(
    r"\b(not less than|at least|no less than|not more than|no more than|not exceed|shall not exceed|less than|greater than)\b",
    re.IGNORECASE,
)
_RE_DEFINITIONAL_ADJUSTMENT = re.compile(r"\bprovided\b|\bexcluded\b|\bfor the purpose of\b", re.IGNORECASE)

_METRIC_PATTERNS: List[Tuple[str, re.Pattern[str]]] = [
    ("leverage_ratio", re.compile(r"\bLeverage Ratio\b", re.IGNORECASE)),
    ("interest_coverage_ratio", re.compile(r"\bInterest Coverage Ratio\b", re.IGNORECASE)),
    ("fixed_charge_coverage_ratio", re.compile(r"\bFixed Charge Coverage Ratio\b", re.IGNORECASE)),
    ("debt_service_ratio", re.compile(r"\bDebt Service\b", re.IGNORECASE)),
    ("ebitda", re.compile(r"\bEBITDA\b", re.IGNORECASE)),
    ("net_worth", re.compile(r"\b(Net Worth|Tangible Net Worth)\b", re.IGNORECASE)),
    ("liquidity", re.compile(r"\bLiquidity\b", re.IGNORECASE)),
    ("current_ratio", re.compile(r"\bCurrent Ratio\b", re.IGNORECASE)),
]


@dataclass(frozen=True)
class SnippetHit:
    item_id: str
    anchor_id: str
    buckets: List[str]
    metrics: List[str]
    snippet_preview: str


def _classify(snippet: str) -> Tuple[List[str], List[str]]:
    buckets: List[str] = []
    metrics: List[str] = []

    if _RE_TABLE.search(snippet):
        buckets.append("has_table")
    if _RE_SCHEDULE_DATE_PHRASES.search(snippet):
        buckets.append("schedule_language")
    if _RE_SPRINGING.search(snippet):
        buckets.append("springing_language")
    if _RE_CURE.search(snippet):
        buckets.append("cure_language")
    if _RE_PROHIBITION.search(snippet):
        buckets.append("prohibition_style")
    if _RE_COMPARATOR_PHRASES.search(snippet):
        buckets.append("comparator_phrases")
    if _RE_DEFINITIONAL_ADJUSTMENT.search(snippet):
        buckets.append("definitional_adjustments")

    for name, pat in _METRIC_PATTERNS:
        if pat.search(snippet):
            metrics.append(name)

    return (sorted(set(buckets)), sorted(set(metrics)))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--base-dir", default=".")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-examples-per-bucket", type=int, default=8)
    ap.add_argument("--preview-chars", type=int, default=220)
    args = ap.parse_args()

    base_dir = Path(args.base_dir)
    run_dir = base_dir / "runs" / args.run_id
    retrieval_dir = run_dir / "retrieval_v2"
    if not retrieval_dir.exists():
        raise SystemExit(f"Missing retrieval_v2 dir: {retrieval_dir}")

    hits: List[SnippetHit] = []
    bucket_examples: Dict[str, List[Dict[str, Any]]] = {}
    bucket_counts: Dict[str, int] = {}
    metric_counts: Dict[str, int] = {}

    total_snippets = 0
    total_financial_snippets = 0

    for path in sorted(retrieval_dir.glob("*_snippets.jsonl")):
        item_id = path.name[: -len("_snippets.jsonl")]
        for rec in _iter_jsonl(path):
            total_snippets += 1
            cats = rec.get("categories") or []
            if not (isinstance(cats, list) and "financial_covenant" in cats):
                continue
            total_financial_snippets += 1

            anchor_id = rec.get("anchor_id")
            snippet = rec.get("snippet")
            if not isinstance(anchor_id, str) or not isinstance(snippet, str):
                continue

            buckets, metrics = _classify(snippet)

            for b in buckets:
                bucket_counts[b] = bucket_counts.get(b, 0) + 1
                if b not in bucket_examples:
                    bucket_examples[b] = []
                if len(bucket_examples[b]) < max(0, int(args.max_examples_per_bucket)):
                    bucket_examples[b].append(
                        {
                            "item_id": item_id,
                            "anchor_id": anchor_id,
                            "preview": _shorten(snippet, limit=int(args.preview_chars)),
                        }
                    )

            for m in metrics:
                metric_counts[m] = metric_counts.get(m, 0) + 1

            hits.append(
                SnippetHit(
                    item_id=item_id,
                    anchor_id=anchor_id,
                    buckets=buckets,
                    metrics=metrics,
                    snippet_preview=_shorten(snippet, limit=int(args.preview_chars)),
                )
            )

    report = {
        "run_id": args.run_id,
        "retrieval_dir": str(retrieval_dir),
        "stats": {
            "total_snippets": total_snippets,
            "financial_covenant_snippets": total_financial_snippets,
            "unique_items": len({h.item_id for h in hits}),
        },
        "bucket_counts": dict(sorted(bucket_counts.items(), key=lambda kv: (-kv[1], kv[0]))),
        "metric_counts": dict(sorted(metric_counts.items(), key=lambda kv: (-kv[1], kv[0]))),
        "bucket_examples": bucket_examples,
        # Full per-snippet rows can be large; include but keep previews short.
        "hits": [
            {
                "item_id": h.item_id,
                "anchor_id": h.anchor_id,
                "buckets": h.buckets,
                "metrics": h.metrics,
                "snippet_preview": h.snippet_preview,
            }
            for h in hits
        ],
    }

    _write_json(Path(args.out), report)
    print(f"Wrote {args.out} (financial snippets={total_financial_snippets}, items={report['stats']['unique_items']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
