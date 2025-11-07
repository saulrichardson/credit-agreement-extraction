#!/usr/bin/env python3
"""Validate LLM-produced segmentation artifacts against anchors.tsv.

Usage:
    python scripts/validate_plan.py --plan planner_semantic_index_entropy.json \
        --anchors /tmp/pricing_grid_sample/anchors.tsv

The validator focuses on the contract we established:
    * segments cover every sentence anchor exactly once, in order
    * overlays/frames reference valid anchor ids
The script exits non-zero if any check fails and prints concise diagnostics.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from edgar_filing_pipeline.plan_validation import (
    load_sentence_anchor_ids,
    validate_plan as validate_plan_dict,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate planner output against anchors")
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--anchors", required=True, type=Path)
    args = parser.parse_args()

    sentence_ids = load_sentence_anchor_ids(args.anchors)
    plan = json.loads(args.plan.read_text())
    errors = validate_plan_dict(plan, sentence_ids)

    if errors:
        print("Validation failed:")
        for line in errors:
            print(f" - {line}")
        return 1

    print("Plan validated successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
