#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence


def _boolish(value: object) -> bool:
    raw = str(value or "").strip().lower()
    return raw in {"1", "true", "t", "yes", "y"}


def _floatish(value: object) -> float | None:
    if value is None:
        return None
    raw = str(value).strip()
    if not raw:
        return None
    try:
        return float(raw)
    except Exception:
        return None


def _intish_str(value: object) -> str | None:
    raw = str(value or "").strip()
    if not raw:
        return None
    try:
        if "." in raw:
            return str(int(float(raw)))
        return str(int(raw))
    except Exception:
        # Fall back to raw, but keep it only if it looks reasonable.
        return raw if raw.isdigit() else None


def _guess_quarter(month: int) -> str:
    if month <= 0 or month > 12:
        raise ValueError(f"Invalid month={month} for quarter inference")
    if month <= 3:
        return "QTR1"
    if month <= 6:
        return "QTR2"
    if month <= 9:
        return "QTR3"
    return "QTR4"


def _tarball_path(daily_filings_root: Path, tarfile: str) -> Path:
    # Expected tarfile shape: YYYYMMDD.nc.tar.gz
    if len(tarfile) < 8:
        raise ValueError(f"tarfile does not look like YYYYMMDD...: {tarfile!r}")
    year = int(tarfile[0:4])
    month = int(tarfile[4:6])
    qtr = _guess_quarter(month)
    return daily_filings_root / str(year) / qtr / tarfile


def _default_batch_merged_path(*, year: int, qtr: int) -> Path:
    # Mirror the convention in /projects/.../processed_output
    return Path(
        f"/projects/rps/dlg340/greenwaldlab/edgar/output/legacy/processed_output/"
        f"batch_merged_{year}_q{qtr}_v3_debt_gpt-4.1-nano.csv"
    )


@dataclass(frozen=True)
class Candidate:
    item_id: str
    accession: str
    sequence: str
    tarfile: str
    tarball_path: Path
    doc_type: str
    conformed_name: str
    confidence: float
    reason: str
    needs_more_text: bool
    kw_credit: bool
    kw_coverage: bool
    kw_covenant: bool


def _iter_candidates(
    csv_path: Path,
    *,
    daily_filings_root: Path,
    doc_type_prefixes: Sequence[str],
    min_confidence: float,
    require_is_ex10: bool,
    require_kw_credit: bool,
    strict_tarball_exists: bool,
) -> Iterable[Candidate]:
    wanted_prefixes = tuple(p.strip().upper() for p in doc_type_prefixes if p.strip())
    if not wanted_prefixes:
        raise ValueError("doc_type_prefixes must not be empty")
    if min_confidence < 0 or min_confidence > 1:
        raise ValueError("--min-confidence must be in [0,1]")

    with csv_path.open(newline="") as f:
        r = csv.DictReader(f)
        if not r.fieldnames:
            raise ValueError(f"CSV has no header: {csv_path}")
        required = {
            "accession-number",
            "segment_no_x",
            "tarfile",
            "doc_type",
            "conformed-name",
            "isDebtContract",
            "confidence",
            "debtClassificationReason",
        }
        missing = sorted(required - set(r.fieldnames))
        if missing:
            raise ValueError(f"CSV missing required columns: {missing} (file={csv_path})")

        for row in r:
            if not _boolish(row.get("isDebtContract")):
                continue

            conf = _floatish(row.get("confidence"))
            if conf is None or conf < float(min_confidence):
                continue

            doc_type = str(row.get("doc_type") or "").strip().upper()
            if not any(doc_type.startswith(pfx) for pfx in wanted_prefixes):
                continue

            if require_is_ex10 and not _boolish(row.get("is_ex10")):
                continue

            kw_credit = _boolish(row.get("kw_credit"))
            if require_kw_credit and not kw_credit:
                continue

            accession = str(row.get("accession-number") or "").strip()
            sequence = _intish_str(row.get("segment_no_x"))
            if not accession or not sequence:
                continue

            tarfile = str(row.get("tarfile") or "").strip()
            if not tarfile:
                continue

            tarball_path = _tarball_path(daily_filings_root, tarfile)
            if strict_tarball_exists and not tarball_path.exists():
                raise FileNotFoundError(f"Missing tarball: {tarball_path} (tarfile={tarfile!r})")

            yield Candidate(
                item_id=f"{accession}_{sequence}",
                accession=accession,
                sequence=sequence,
                tarfile=tarfile,
                tarball_path=tarball_path,
                doc_type=doc_type,
                conformed_name=str(row.get("conformed-name") or "").strip(),
                confidence=float(conf),
                reason=str(row.get("debtClassificationReason") or "").strip(),
                needs_more_text=_boolish(row.get("needsMoreText")),
                kw_credit=kw_credit,
                kw_coverage=_boolish(row.get("kw_coverage")),
                kw_covenant=_boolish(row.get("kw_covenant")),
            )


def _stable_unique(values: Iterable[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for v in values:
        if v in seen:
            continue
        seen.add(v)
        out.append(v)
    return out


def _apply_limits(
    candidates: list[Candidate],
    *,
    limit: int | None,
    max_per_tarfile: int | None,
) -> list[Candidate]:
    if limit is not None and limit <= 0:
        raise ValueError("--limit must be > 0 when provided")
    if max_per_tarfile is not None and max_per_tarfile <= 0:
        raise ValueError("--max-per-tarfile must be > 0 when provided")

    if limit is None and max_per_tarfile is None:
        return candidates

    out: list[Candidate] = []
    per_tf: dict[str, int] = defaultdict(int)
    for c in candidates:
        if max_per_tarfile is not None and per_tf[c.tarfile] >= max_per_tarfile:
            continue
        out.append(c)
        per_tf[c.tarfile] += 1
        if limit is not None and len(out) >= limit:
            break
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Select candidate debt/credit agreements from precomputed isdebt batch_merged CSV outputs "
            "and emit a pipeline-compatible item_ids JSON + tarballs list."
        )
    )
    ap.add_argument("--input-csv", type=str, default=None, help="Path to batch_merged_*_debt_*.csv")
    ap.add_argument("--year", type=int, default=None, help="Convenience: year for default input CSV path")
    ap.add_argument("--qtr", type=int, default=None, help="Convenience: quarter (1-4) for default input CSV path")

    ap.add_argument(
        "--daily-filings-root",
        type=str,
        default="/projects/rps/dlg340/greenwaldlab/edgar/daily_filings",
        help="Root directory containing YYYY/QTR*/YYYYMMDD.nc.tar.gz tarballs.",
    )
    ap.add_argument(
        "--doc-type-prefix",
        action="append",
        default=["EX-10"],
        help="Repeatable doc_type prefix allowlist (default: EX-10).",
    )
    ap.add_argument("--min-confidence", type=float, default=0.8, help="Minimum isdebt confidence threshold.")
    ap.add_argument(
        "--require-is-ex10",
        action="store_true",
        help="Require is_ex10=true in the merged CSV (in addition to doc_type prefix filtering).",
    )
    ap.add_argument(
        "--require-kw-credit",
        action="store_true",
        help="Require kw_credit=true (regex hint) to reduce false positives.",
    )
    ap.add_argument(
        "--strict-tarball-exists",
        dest="strict_tarball_exists",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Fail if any selected row references a tarball path that does not exist.",
    )

    ap.add_argument("--limit", type=int, default=None, help="Optional cap on number of selected item_ids.")
    ap.add_argument(
        "--max-per-tarfile",
        type=int,
        default=None,
        help="Optional per-tarfile cap to diversify selections across days.",
    )

    ap.add_argument("--out-dir", type=str, required=True, help="Output directory to write dataset + tarballs + report.")
    ap.add_argument("--dataset-name", type=str, default=None, help="Dataset name (default derived from input).")
    ap.add_argument("--notes", type=str, default="", help="Optional notes string embedded in the dataset JSON.")

    args = ap.parse_args(argv)

    if args.input_csv:
        csv_path = Path(args.input_csv)
    else:
        if args.year is None or args.qtr is None:
            raise SystemExit("Provide --input-csv OR (--year and --qtr).")
        csv_path = _default_batch_merged_path(year=int(args.year), qtr=int(args.qtr))

    if not csv_path.exists():
        raise SystemExit(f"Input CSV not found: {csv_path}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    daily_root = Path(args.daily_filings_root)
    if not daily_root.exists():
        raise SystemExit(f"daily_filings_root not found: {daily_root}")

    raw_candidates = list(
        _iter_candidates(
            csv_path,
            daily_filings_root=daily_root,
            doc_type_prefixes=args.doc_type_prefix,
            min_confidence=float(args.min_confidence),
            require_is_ex10=bool(args.require_is_ex10),
            require_kw_credit=bool(args.require_kw_credit),
            strict_tarball_exists=bool(args.strict_tarball_exists),
        )
    )
    # Sort by descending confidence, then stable by item_id.
    raw_candidates.sort(key=lambda c: (-c.confidence, c.item_id))

    selected = _apply_limits(
        raw_candidates,
        limit=args.limit,
        max_per_tarfile=args.max_per_tarfile,
    )

    item_ids = _stable_unique(c.item_id for c in selected)
    tarballs = _stable_unique(str(c.tarball_path) for c in selected)

    if not item_ids:
        raise SystemExit("No candidates matched the filters. Loosen filters or check input CSV.")

    dataset_name = args.dataset_name
    if not dataset_name:
        dataset_name = f"isdebt-{csv_path.stem}"

    dataset = {
        "name": dataset_name,
        "description": (
            "Auto-selected candidate debt/credit agreements from isdebt LLM batch outputs "
            f"(source={str(csv_path)})."
        ),
        "created_at": int(time.time()),
        "source_csv": str(csv_path),
        "filters": {
            "doc_type_prefixes": [p for p in args.doc_type_prefix if str(p).strip()],
            "min_confidence": float(args.min_confidence),
            "require_is_ex10": bool(args.require_is_ex10),
            "require_kw_credit": bool(args.require_kw_credit),
        },
        "notes": str(args.notes or ""),
        "item_ids": item_ids,
    }

    (out_dir / "item_ids.json").write_text(json.dumps(dataset, indent=2) + "\n", encoding="utf-8")
    (out_dir / "tarballs_used.txt").write_text("\n".join(tarballs) + "\n", encoding="utf-8")

    # Write a human review report.
    report_path = out_dir / "candidates.csv"
    with report_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "item_id",
                "accession",
                "sequence",
                "tarfile",
                "tarball_path",
                "doc_type",
                "conformed_name",
                "confidence",
                "needs_more_text",
                "kw_credit",
                "kw_coverage",
                "kw_covenant",
                "reason",
            ],
        )
        w.writeheader()
        for c in selected:
            w.writerow(
                {
                    "item_id": c.item_id,
                    "accession": c.accession,
                    "sequence": c.sequence,
                    "tarfile": c.tarfile,
                    "tarball_path": str(c.tarball_path),
                    "doc_type": c.doc_type,
                    "conformed_name": c.conformed_name,
                    "confidence": f"{c.confidence:.4f}",
                    "needs_more_text": str(bool(c.needs_more_text)),
                    "kw_credit": str(bool(c.kw_credit)),
                    "kw_coverage": str(bool(c.kw_coverage)),
                    "kw_covenant": str(bool(c.kw_covenant)),
                    "reason": c.reason,
                }
            )

    counts = Counter()
    for c in selected:
        counts["selected"] += 1
        counts[f"doc_type:{c.doc_type}"] += 1
        counts[f"tarfile:{c.tarfile}"] += 1
        if c.kw_covenant:
            counts["kw_covenant_true"] += 1
        if c.kw_credit:
            counts["kw_credit_true"] += 1
        if c.kw_coverage:
            counts["kw_coverage_true"] += 1

    summary = {
        "schema_version": "isdebt_selection_summary_v1",
        "created_at": int(time.time()),
        "input_csv": str(csv_path),
        "out_dir": str(out_dir),
        "total_candidates_matched": len(raw_candidates),
        "selected_count": len(selected),
        "unique_item_ids": len(item_ids),
        "unique_tarballs": len(tarballs),
        "counts": dict(counts),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
