#!/usr/bin/env python
"""
Artifact-first provenance + hallucination audit for a run folder.

This script does NOT make any LLM calls. It inspects:
  - Pricing outputs: runs/<run_id>/contractir_v0_2/items/<item_id>/contractir_merged.json
  - Covenant outputs: runs/<run_id>/covenantir_v0_1/<item_id>/covenantir_validated.json
  - Source text: runs/<run_id>/normalized/<item_id>/canonical.txt + anchors.tsv spans

Checks performed (heuristic, but useful):
  1) Provenance integrity:
     - every referenced anchor id exists in anchors.tsv
     - every referenced anchor span extracts non-empty text from canonical.txt
  2) Literal support (anti-hallucination heuristic):
     - numeric/date literals in the JSON should appear (in some textual form) in the
       canonical text of the referenced anchors.

Outputs:
  - JSON report summarizing issues per item.
  - Optional per-item "review packet" text files for flagged items, embedding:
      * the extracted JSON (pretty-printed)
      * the exact anchor texts referenced by the output (from canonical spans)

Note: literal-support checks can produce false positives because documents may render
rates as fractions, basis points, or differently-rounded values.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"

import sys

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from pipeline.core.anchors import load_anchor_catalog  # noqa: E402
from pipeline.core.config import Paths  # noqa: E402


ANCHOR_ID_RE = re.compile(r"^A\\d{4,}$")


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _iter_anchor_refs(obj: Any) -> Iterable[Tuple[str, str]]:
    """Yield (json_path, anchor_id) for every anchor referenced under source_refs/anchor_ids."""

    def _walk(x: Any, path: List[str]) -> Iterable[Tuple[str, str]]:
        if isinstance(x, dict):
            for k, v in x.items():
                if k in ("source_refs", "anchor_ids") and isinstance(v, list):
                    for i, aid in enumerate(v):
                        if isinstance(aid, str):
                            yield ("/" + "/".join(path + [k, str(i)]), aid.strip())
                else:
                    yield from _walk(v, path + [k])
        elif isinstance(x, list):
            for i, v in enumerate(x):
                yield from _walk(v, path + [str(i)])

    yield from _walk(obj, [])


def _is_ast_node(x: Any) -> bool:
    if not isinstance(x, dict):
        return False
    # ContractIR/CovenantIR AST nodes are exactly one of these keys.
    keys = set(x.keys())
    return bool(keys & {"lit", "var", "op"}) and keys.issubset({"lit", "var", "op"})


@dataclass(frozen=True)
class LiteralRef:
    json_path: str
    lit_type: str
    value: Any
    source_refs: Tuple[str, ...]


def _iter_literals_with_provenance(obj: Any) -> Iterable[LiteralRef]:
    """Yield LiteralRef for AST literal nodes, carrying nearest enclosing source_refs."""

    def _walk(x: Any, path: List[str], current_source_refs: Sequence[str]) -> Iterable[LiteralRef]:
        if isinstance(x, dict):
            next_source_refs = current_source_refs
            if "source_refs" in x and isinstance(x.get("source_refs"), list):
                next_source_refs = [a.strip() for a in x["source_refs"] if isinstance(a, str)]

            if _is_ast_node(x) and "lit" in x and isinstance(x.get("lit"), dict):
                lit = x["lit"]
                lit_type = lit.get("type")
                value = lit.get("value")
                if isinstance(lit_type, str):
                    yield LiteralRef(
                        json_path="/" + "/".join(path),
                        lit_type=lit_type,
                        value=value,
                        source_refs=tuple(next_source_refs),
                    )

            for k, v in x.items():
                yield from _walk(v, path + [k], next_source_refs)
        elif isinstance(x, list):
            for i, v in enumerate(x):
                yield from _walk(v, path + [str(i)], current_source_refs)

    yield from _walk(obj, [], [])


def _anchor_text(
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
    txt = canonical_text[a:b]
    if not isinstance(txt, str):
        return None
    txt = txt.strip()
    return txt or None


def _normalize_for_search(text: str) -> str:
    # Keep digits and punctuation; just normalize whitespace and case.
    return " ".join((text or "").lower().split())


def _decimal_from_str(x: Any) -> Optional[Decimal]:
    if not isinstance(x, str):
        return None
    s = x.strip()
    if not s:
        return None
    try:
        return Decimal(s)
    except InvalidOperation:
        return None


def _rate_support_candidates(rate_str: str) -> List[str]:
    """Generate a small set of plausible textual renderings for a rate literal."""

    d = _decimal_from_str(rate_str)
    if d is None:
        return []

    # rate literals are fractions, e.g. 0.025 for 2.5%
    pct = d * Decimal("100")
    bps = d * Decimal("10000")

    candidates: List[str] = []

    def _fmt(dec: Decimal, places: int) -> str:
        q = Decimal(10) ** -places
        s = str(dec.quantize(q))
        # Trim trailing zeros but keep at least 1 decimal when it was fractional.
        if "." in s:
            s = s.rstrip("0").rstrip(".") or "0"
        return s

    # Percent formats: "2.5%" / "2.50%" / "2.500%"
    candidates.append(_fmt(pct, 1) + "%")
    candidates.append(_fmt(pct, 2) + "%")
    candidates.append(_fmt(pct, 3) + "%")
    candidates.append(_fmt(pct, 4) + "%")
    # Sometimes there's a space before the percent sign.
    candidates.append(_fmt(pct, 2) + " %")

    # Bps formats if integer.
    if bps == bps.to_integral_value():
        bps_i = str(int(bps))
        candidates.extend(
            [
                f"{bps_i} bps",
                f"{bps_i}bp",
                f"{bps_i} basis points",
                f"{bps_i} basis point",
            ]
        )

    # De-dup while preserving order.
    out: List[str] = []
    seen: set[str] = set()
    for c in candidates:
        c = c.strip()
        if not c or c in seen:
            continue
        out.append(c)
        seen.add(c)
    return out


def _decimal_support_candidates(dec_str: str) -> List[str]:
    d = _decimal_from_str(dec_str)
    if d is None:
        return []
    candidates: List[str] = []
    candidates.append(str(d))  # as-is

    # Common "show 2 decimals" rendering for ratios.
    try:
        candidates.append(str(d.quantize(Decimal("0.01"))))
    except Exception:
        pass

    # If it's an integer, also try common agreement renderings with commas/$.
    if d == d.to_integral_value():
        sign = "-" if d < 0 else ""
        i = int(abs(d))
        with_commas = f"{i:,}"
        candidates.extend(
            [
                f"{sign}{i}",
                f"{sign}{with_commas}",
                f"{sign}${with_commas}",
                f"{sign}${with_commas}.00",
            ]
        )

    out: List[str] = []
    seen: set[str] = set()
    for c in candidates:
        c = c.strip()
        if not c or c in seen:
            continue
        out.append(c)
        seen.add(c)
    return out


def _literal_candidates(lit: LiteralRef) -> List[str]:
    if lit.lit_type == "rate" and isinstance(lit.value, str):
        return _rate_support_candidates(lit.value)
    if lit.lit_type in ("decimal", "int") and isinstance(lit.value, str):
        return _decimal_support_candidates(lit.value)
    if lit.lit_type == "date" and isinstance(lit.value, str):
        v = lit.value.strip()
        if not v:
            return []
        # Prefer ISO date, but agreements often use "December 31, 1996" or "12/31/1996".
        out = [v]
        m = re.match(r"^(\d{4})-(\d{2})-(\d{2})$", v)
        if m:
            yyyy, mm, dd = m.group(1), m.group(2), m.group(3)
            out.append(f"{mm}/{dd}/{yyyy}")
            out.append(f"{mm}-{dd}-{yyyy}")
            month_names = [
                "January",
                "February",
                "March",
                "April",
                "May",
                "June",
                "July",
                "August",
                "September",
                "October",
                "November",
                "December",
            ]
            try:
                mi = int(mm)
                di = int(dd)
            except Exception:
                mi = 0
                di = 0
            if 1 <= mi <= 12 and 1 <= di <= 31:
                month = month_names[mi - 1]
                abbr = month[:3]
                out.append(f"{month} {di}, {yyyy}")
                out.append(f"{abbr} {di}, {yyyy}")
                out.append(f"{abbr}. {di}, {yyyy}")
                out.append(f"{di} {month} {yyyy}")
        # De-dup
        dedup: List[str] = []
        seen: set[str] = set()
        for c in out:
            c = c.strip()
            if not c or c in seen:
                continue
            dedup.append(c)
            seen.add(c)
        return dedup
    return []


def _search_candidates(text: str, candidates: Sequence[str]) -> bool:
    hay = _normalize_for_search(text)
    for c in candidates:
        if not c:
            continue
        if _normalize_for_search(c) in hay:
            return True
    return False


def _build_review_packet(
    *,
    out_path: Path,
    title: str,
    extracted_json: Any,
    anchor_texts: Mapping[str, str],
) -> None:
    parts: List[str] = []
    parts.append(title)
    parts.append("=" * len(title))
    parts.append("")
    parts.append("EXTRACTED_JSON")
    parts.append("------------")
    parts.append(json.dumps(extracted_json, indent=2, sort_keys=True))
    parts.append("")
    parts.append("SOURCE_ANCHORS (canonical.txt spans)")
    parts.append("-------------------------------")
    for aid in sorted(
        anchor_texts.keys(),
        key=lambda s: int(re.sub(r"^A", "", s)) if ANCHOR_ID_RE.fullmatch(s) else 10**9,
    ):
        parts.append(f"[[{aid}]]")
        parts.append(anchor_texts[aid].rstrip())
        parts.append("")
    _write_text(out_path, "\n".join(parts).rstrip() + "\n")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--base-dir", default=".")
    ap.add_argument("--out", default=None, help="Path to write JSON report (defaults to runs/<run_id>/provenance_audit.json)")
    ap.add_argument("--write-review-packets", action="store_true")
    ap.add_argument(
        "--review-packets-dir",
        default=None,
        help="Directory for review packets (defaults to runs/<run_id>/provenance_review/)",
    )
    ap.add_argument("--max-review-packets", type=int, default=5, help="Max number of review packets to write (most-flagged first).")
    args = ap.parse_args()

    paths = Paths(root=Path(args.base_dir), run_id=args.run_id)
    run_dir = paths.run_dir

    pricing_dir = run_dir / "contractir_v0_2" / "items"
    cov_dir = run_dir / "covenantir_v0_1"

    out_path = Path(args.out) if args.out else (run_dir / "provenance_audit.json")
    review_dir = Path(args.review_packets_dir) if args.review_packets_dir else (run_dir / "provenance_review")

    # Collect candidate item ids from whichever outputs exist.
    item_ids: set[str] = set()
    if pricing_dir.exists():
        for p in pricing_dir.iterdir():
            if p.is_dir():
                item_ids.add(p.name)
    if cov_dir.exists():
        for p in cov_dir.iterdir():
            if p.is_dir():
                item_ids.add(p.name)

    per_item: Dict[str, Any] = {}
    for item_id in sorted(item_ids):
        catalog = load_anchor_catalog(paths, item_id)
        canonical_path = run_dir / "normalized" / item_id / "canonical.txt"
        canonical_text = canonical_path.read_text(encoding="utf-8", errors="replace") if canonical_path.exists() else ""

        item_rec: Dict[str, Any] = {"item_id": item_id}

        # --- Pricing audit ---
        merged_path = pricing_dir / item_id / "contractir_merged.json"
        if merged_path.exists():
            doc = _read_json(merged_path)

            anchor_missing: List[Dict[str, str]] = []
            anchor_empty: List[Dict[str, str]] = []
            for jpath, aid in _iter_anchor_refs(doc):
                if not aid or not ANCHOR_ID_RE.fullmatch(aid):
                    continue
                if aid not in catalog:
                    anchor_missing.append({"json_path": jpath, "anchor_id": aid})
                    continue
                txt = _anchor_text(canonical_text, catalog, aid)
                if not txt:
                    anchor_empty.append({"json_path": jpath, "anchor_id": aid})

            # Literal support checks (only when we have some provenance).
            unsupported_literals: List[Dict[str, Any]] = []
            checked = 0
            for lit in _iter_literals_with_provenance(doc):
                candidates = _literal_candidates(lit)
                if not candidates:
                    continue
                checked += 1
                # If the literal has no provenance, flag it as unsupported immediately.
                if not lit.source_refs:
                    unsupported_literals.append(
                        {
                            "json_path": lit.json_path,
                            "lit_type": lit.lit_type,
                            "value": lit.value,
                            "reason": "no_source_refs",
                        }
                    )
                    continue
                # Search only in the cited anchor texts.
                evidence_text = "\n".join(
                    [t for a in lit.source_refs if (t := _anchor_text(canonical_text, catalog, a)) is not None]
                )
                if not evidence_text.strip():
                    unsupported_literals.append(
                        {
                            "json_path": lit.json_path,
                            "lit_type": lit.lit_type,
                            "value": lit.value,
                            "source_refs": list(lit.source_refs),
                            "reason": "empty_anchor_text",
                        }
                    )
                    continue
                if not _search_candidates(evidence_text, candidates):
                    unsupported_literals.append(
                        {
                            "json_path": lit.json_path,
                            "lit_type": lit.lit_type,
                            "value": lit.value,
                            "source_refs": list(lit.source_refs),
                            "candidates": candidates[:8],
                            "reason": "not_found_in_source_refs_text",
                        }
                    )

            item_rec["pricing"] = {
                "contractir_merged_path": str(merged_path),
                "anchor_missing_count": len(anchor_missing),
                "anchor_empty_count": len(anchor_empty),
                "unsupported_literals_count": len(unsupported_literals),
                "literals_checked_count": checked,
                "anchor_missing_examples": anchor_missing[:5],
                "anchor_empty_examples": anchor_empty[:5],
                "unsupported_literal_examples": unsupported_literals[:10],
            }

        # --- Covenant audit ---
        cov_path = cov_dir / item_id / "covenantir_validated.json"
        if cov_path.exists():
            doc = _read_json(cov_path)
            anchor_missing: List[Dict[str, str]] = []
            anchor_empty: List[Dict[str, str]] = []
            for jpath, aid in _iter_anchor_refs(doc):
                if not aid or not ANCHOR_ID_RE.fullmatch(aid):
                    continue
                if aid not in catalog:
                    anchor_missing.append({"json_path": jpath, "anchor_id": aid})
                    continue
                txt = _anchor_text(canonical_text, catalog, aid)
                if not txt:
                    anchor_empty.append({"json_path": jpath, "anchor_id": aid})

            unsupported_literals: List[Dict[str, Any]] = []
            checked = 0
            for lit in _iter_literals_with_provenance(doc):
                candidates = _literal_candidates(lit)
                if not candidates:
                    continue
                checked += 1
                if not lit.source_refs:
                    unsupported_literals.append(
                        {
                            "json_path": lit.json_path,
                            "lit_type": lit.lit_type,
                            "value": lit.value,
                            "reason": "no_source_refs",
                        }
                    )
                    continue
                evidence_text = "\n".join(
                    [t for a in lit.source_refs if (t := _anchor_text(canonical_text, catalog, a)) is not None]
                )
                if not evidence_text.strip():
                    unsupported_literals.append(
                        {
                            "json_path": lit.json_path,
                            "lit_type": lit.lit_type,
                            "value": lit.value,
                            "source_refs": list(lit.source_refs),
                            "reason": "empty_anchor_text",
                        }
                    )
                    continue
                if not _search_candidates(evidence_text, candidates):
                    unsupported_literals.append(
                        {
                            "json_path": lit.json_path,
                            "lit_type": lit.lit_type,
                            "value": lit.value,
                            "source_refs": list(lit.source_refs),
                            "candidates": candidates[:8],
                            "reason": "not_found_in_source_refs_text",
                        }
                    )

            item_rec["covenants"] = {
                "covenantir_validated_path": str(cov_path),
                "anchor_missing_count": len(anchor_missing),
                "anchor_empty_count": len(anchor_empty),
                "unsupported_literals_count": len(unsupported_literals),
                "literals_checked_count": checked,
                "anchor_missing_examples": anchor_missing[:5],
                "anchor_empty_examples": anchor_empty[:5],
                "unsupported_literal_examples": unsupported_literals[:10],
            }

        per_item[item_id] = item_rec

    # Rank for review packets.
    ranking: List[Tuple[int, str]] = []
    for item_id, rec in per_item.items():
        score = 0
        for k in ("pricing", "covenants"):
            if k not in rec:
                continue
            score += int(rec[k].get("anchor_missing_count") or 0) * 100
            score += int(rec[k].get("anchor_empty_count") or 0) * 50
            score += int(rec[k].get("unsupported_literals_count") or 0)
        ranking.append((score, item_id))
    ranking.sort(key=lambda t: (-t[0], t[1]))

    report = {
        "run_id": args.run_id,
        "items_count": len(per_item),
        "items": per_item,
        "review_ranking": [{"item_id": item_id, "score": score} for score, item_id in ranking],
    }
    _write_json(out_path, report)

    if args.write_review_packets:
        written = 0
        for score, item_id in ranking:
            if written >= max(0, int(args.max_review_packets)):
                break
            if score <= 0:
                break

            rec = per_item[item_id]
            catalog = load_anchor_catalog(paths, item_id)
            canonical_path = run_dir / "normalized" / item_id / "canonical.txt"
            canonical_text = canonical_path.read_text(encoding="utf-8", errors="replace") if canonical_path.exists() else ""

            # Prefer pricing merged for review; otherwise covenants.
            extracted_json: Any = None
            title = f"{args.run_id} :: {item_id}"

            anchor_ids: set[str] = set()
            pricing_path = pricing_dir / item_id / "contractir_merged.json"
            cov_path = cov_dir / item_id / "covenantir_validated.json"
            if pricing_path.exists():
                extracted_json = _read_json(pricing_path)
                title += " (pricing)"
                for _, aid in _iter_anchor_refs(extracted_json):
                    if aid and ANCHOR_ID_RE.fullmatch(aid):
                        anchor_ids.add(aid)
            elif cov_path.exists():
                extracted_json = _read_json(cov_path)
                title += " (covenants)"
                for _, aid in _iter_anchor_refs(extracted_json):
                    if aid and ANCHOR_ID_RE.fullmatch(aid):
                        anchor_ids.add(aid)
            else:
                continue

            anchor_texts: Dict[str, str] = {}
            for aid in sorted(anchor_ids, key=lambda s: int(s[1:]) if ANCHOR_ID_RE.fullmatch(s) else 10**9):
                txt = _anchor_text(canonical_text, catalog, aid)
                if txt:
                    anchor_texts[aid] = txt
            out_file = review_dir / f"{item_id}.review.txt"
            _build_review_packet(out_path=out_file, title=title, extracted_json=extracted_json, anchor_texts=anchor_texts)
            written += 1

    print(str(out_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
