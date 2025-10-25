#!/usr/bin/env python3
"""Aggregate normalized pricing JSON into wide/long CSV representations.

The script scans an input directory for JSON files (or JSON-flavoured text files),
validates their contents, and emits:

* vectors_wide.csv
* vectors_long.csv
* subjects_catalog.csv
* issues.log

See the in-repo README or CLI ``--help`` for usage details.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict, OrderedDict
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Dict, Iterable, List, Optional

SUBJECT_PATTERN = "^[a-z0-9]+(?:_[a-z0-9]+)*_bps$"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Flatten normalized loan pricing JSON into analytic CSV outputs."
    )
    parser.add_argument(
        "--in-dir",
        required=True,
        type=Path,
        help="Directory containing JSON response files.",
    )
    parser.add_argument(
        "--out-dir",
        required=True,
        type=Path,
        help="Directory where CSV outputs and issues.log will be written.",
    )
    parser.add_argument(
        "--pattern",
        default="*.json",
        help="Glob pattern for selecting input files (default: *.json). "
        "Use, for example, '*.txt' if the JSON is stored with .txt suffix.",
    )
    return parser.parse_args()


def is_valid_subject(subject: str) -> bool:
    import re

    return bool(re.match(SUBJECT_PATTERN, subject))


def decimal_from_value(value) -> Optional[Decimal]:
    """Convert supported types to Decimal while preserving precision."""
    if isinstance(value, (int, float)):
        value = str(value)
    if isinstance(value, str):
        candidate = value.strip()
        if not candidate:
            return None
        try:
            return Decimal(candidate)
        except InvalidOperation:
            return None
    return None


def decimal_to_string(value: Decimal) -> str:
    text = format(value, "f")
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return text


def iter_files(directory: Path, pattern: str) -> Iterable[Path]:
    return sorted(directory.glob(pattern))


def load_json(path: Path, issues: List[str]) -> Optional[dict]:
    try:
        raw = path.read_text(encoding="utf-8")
    except Exception as exc:  # noqa: BLE001
        issues.append(f"{path.name}: unable to read file ({exc}).")
        return None
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        issues.append(f"{path.name}: invalid JSON ({exc}).")
        return None
    if not isinstance(data, dict):
        issues.append(f"{path.name}: top-level JSON is not an object; skipping file.")
        return None
    return data


def main() -> None:
    args = parse_args()

    in_dir: Path = args.in_dir
    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    files = list(iter_files(in_dir, args.pattern))
    issues: List[str] = []

    deals: Dict[str, List[OrderedDict[str, object]]] = {}
    subject_max_tiers: Dict[str, int] = defaultdict(int)
    subject_deal_presence: Dict[str, set[str]] = defaultdict(set)
    all_subjects: set[str] = set()

    if not files:
        issues.append("No files matched the provided pattern.")

    for file_path in files:
        data = load_json(file_path, issues)
        if data is None:
            continue

        deal_id = file_path.stem
        subjects_raw = data.get("subjects")
        rows_raw = data.get("rows")

        if not isinstance(subjects_raw, list):
            issues.append(f"{file_path.name}: 'subjects' is missing or not a list; skipping file.")
            continue
        if not isinstance(rows_raw, list):
            issues.append(f"{file_path.name}: 'rows' is missing or not a list; skipping file.")
            continue

        valid_subjects: List[str] = []
        for subject in subjects_raw:
            if not isinstance(subject, str):
                issues.append(f"{file_path.name}: subject {subject!r} is not a string; skipping subject.")
                continue
            if not subject.endswith("_bps") or not is_valid_subject(subject):
                issues.append(f"{file_path.name}: subject '{subject}' is malformed; skipping subject.")
                continue
            valid_subjects.append(subject)

        if not valid_subjects:
            issues.append(f"{file_path.name}: no valid subjects after validation; skipping file.")
            continue

        seen_row_ids: set[str] = set()
        deal_rows: List[OrderedDict[str, object]] = []

        for row in rows_raw:
            if not isinstance(row, dict):
                issues.append(f"{file_path.name}: encountered non-object row; skipping row.")
                continue

            row_id = row.get("row_id")
            if not isinstance(row_id, str):
                issues.append(f"{file_path.name}: row missing string 'row_id'; skipping row.")
                continue
            if row_id in seen_row_ids:
                issues.append(f"{file_path.name}: duplicate row_id '{row_id}' encountered.")
            else:
                seen_row_ids.add(row_id)

            row_subjects: Dict[str, List[Decimal]] = {}
            for subject in valid_subjects:
                raw_raw = row.get(subject, None)
                if raw_raw is None:
                    raw_values: List = []
                elif isinstance(raw_raw, list):
                    raw_values = raw_raw
                else:
                    raw_values = [raw_raw]

                converted: List[Decimal] = []
                for idx, raw_value in enumerate(raw_values, start=1):
                    converted_value = decimal_from_value(raw_value)
                    if converted_value is None:
                        issues.append(
                            f"{file_path.name}: row '{row_id}' subject '{subject}' "
                            f"value at position {idx} is non-numeric; dropping entry."
                        )
                        continue
                    converted.append(converted_value)

                if subject not in row:
                    issues.append(
                        f"{file_path.name}: row '{row_id}' missing subject '{subject}'; treated as empty."
                    )

                if len(converted) > 1:
                    issues.append(
                        f"{file_path.name}: row '{row_id}' subject '{subject}' has multiple values; "
                        "keeping the first and discarding the rest."
                    )
                    converted = converted[:1]

                row_subjects[subject] = converted

                subject_max_tiers[subject] = max(subject_max_tiers[subject], len(converted))
                if converted:
                    subject_deal_presence[subject].add(deal_id)

            ordered_row = OrderedDict()
            ordered_row["row_id"] = row_id
            ordered_row["subjects"] = row_subjects
            if any(row_subjects[subject] for subject in row_subjects):
                deal_rows.append(ordered_row)
            else:
                issues.append(
                    f"{file_path.name}: row '{row_id}' has no numeric subject values; dropping row."
                )

        if not deal_rows:
            issues.append(f"{file_path.name}: no valid rows parsed; skipping file.")
            continue

        deals[deal_id] = deal_rows
        all_subjects.update(valid_subjects)

    all_subjects_sorted = sorted(all_subjects)

    wide_path = out_dir / "vectors_wide.csv"
    long_path = out_dir / "vectors_long.csv"
    catalog_path = out_dir / "subjects_catalog.csv"
    issues_path = out_dir / "issues.log"

    # Write long form CSV
    with long_path.open("w", newline="", encoding="utf-8") as long_file:
        writer = csv.writer(long_file)
        writer.writerow(["deal_id", "row_id", "subject_id", "bps_value"])

        for deal_id in sorted(deals):
            for row in deals[deal_id]:
                row_id = row["row_id"]
                subjects_map: Dict[str, List[Decimal]] = row["subjects"]  # type: ignore[assignment]
                for subject in all_subjects_sorted:
                    values = subjects_map.get(subject, [])
                    if values:
                        writer.writerow([deal_id, row_id, subject, decimal_to_string(values[0])])

    # Prepare header for wide CSV
    header = ["deal_id", "row_id"]
    for subject in all_subjects_sorted:
        header.append(subject)

    with wide_path.open("w", newline="", encoding="utf-8") as wide_file:
        writer = csv.writer(wide_file)
        writer.writerow(header)

        for deal_id in sorted(deals):
            for row in deals[deal_id]:
                row_id = row["row_id"]
                subjects_map: Dict[str, List[Decimal]] = row["subjects"]  # type: ignore[assignment]
                row_out: List[str] = [deal_id, row_id]

                for subject in all_subjects_sorted:
                    values = subjects_map.get(subject, [])
                    row_out.append(decimal_to_string(values[0]) if values else "")

                writer.writerow(row_out)

    with catalog_path.open("w", newline="", encoding="utf-8") as catalog_file:
        writer = csv.writer(catalog_file)
        writer.writerow(["subject_id", "max_tiers", "appears_in_n_deals"])
        for subject in all_subjects_sorted:
            max_tiers = subject_max_tiers.get(subject, 0)
            deals_count = len(subject_deal_presence.get(subject, set()))
            writer.writerow([subject, max_tiers, deals_count])

    if issues:
        issues_content = "\n".join(issues)
    else:
        issues_content = "No issues encountered."

    issues_path.write_text(issues_content, encoding="utf-8")


if __name__ == "__main__":
    main()
