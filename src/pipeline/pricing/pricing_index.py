from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

_PERCENT_RE = re.compile(r"\b\d+(?:\.\d+)?\s*%")
_BPS_RE = re.compile(r"\b\d+(?:\.\d+)?\s*(?:bps|bp|basis points?)\b", flags=re.IGNORECASE)


@dataclass(frozen=True)
class PricingTableScore:
    table_anchor_id: str
    order: int
    score: int
    signals: list[str]

    percent_count: int
    bps_count: int

    is_toc_like: bool
    is_signature_like: bool
    is_lender_commitment_like: bool


def _table_text(table: dict[str, Any]) -> str:
    parts: list[str] = []
    raw = table.get("raw")
    if isinstance(raw, str) and raw.strip():
        parts.append(raw)

    cols = table.get("columns") or []
    if isinstance(cols, list):
        parts.extend(str(c) for c in cols if isinstance(c, str) and c.strip())

    rows = table.get("rows") or []
    if isinstance(rows, list):
        for row in rows:
            if not isinstance(row, dict):
                continue
            row_label = row.get("row_label")
            if isinstance(row_label, str) and row_label.strip():
                parts.append(row_label)
            cells = row.get("cells") or {}
            if isinstance(cells, dict):
                for val in cells.values():
                    if isinstance(val, str) and val.strip():
                        parts.append(val)

    return "\n".join(parts)


def _looks_like_table_of_contents(table: dict[str, Any], text: str) -> bool:
    hay = (text or "").lower()
    if "table of contents" in hay:
        return True

    columns = [str(c or "").strip().lower() for c in (table.get("columns") or [])]
    has_page_col = any(c == "page" for c in columns)

    rows = table.get("rows") or []
    row_labels: list[str] = []
    if isinstance(rows, list):
        for row in rows:
            if isinstance(row, dict):
                row_labels.append(str(row.get("row_label") or "").strip())

    section_rows = sum(1 for rl in row_labels if rl.lower().startswith("section "))
    article_rows = sum(1 for rl in row_labels if rl.lower().startswith("article "))

    # Very common ToC pattern: Page column + many SECTION rows.
    if has_page_col and section_rows >= 3:
        return True
    if has_page_col and (section_rows + article_rows) >= 4:
        return True

    # Fallback: repeated SECTION/ARTICLE tokens plus page numbers.
    if has_page_col and (hay.count("section ") + hay.count("article ")) >= 6:
        return True

    return False


def _looks_like_signature_table(text: str) -> bool:
    hay = (text or "").lower()
    # Signature blocks often contain "By:", "Name:", "Title:" as table rows.
    return ("by:" in hay and "name:" in hay) or ("by:" in hay and "title:" in hay)


def _looks_like_lender_commitment_table(text: str) -> bool:
    hay = (text or "").lower()
    # Common non-pricing schedule: lender commitments / applicable percentages.
    if "applicable percentage" in hay and "lender" in hay and "commitment" in hay:
        return True
    # Another common header combination.
    if "revolving lender" in hay and "commitment" in hay and "%" in hay:
        return True
    if "term lender" in hay and "commitment" in hay and "%" in hay:
        return True
    return False


def score_pricing_table(table: dict[str, Any]) -> PricingTableScore:
    table_anchor_id = str(table.get("table_anchor_id") or "")
    order = int(table.get("order") or 0)

    text = _table_text(table)
    hay = text.lower()

    percent_count = len(_PERCENT_RE.findall(text))
    bps_count = len(_BPS_RE.findall(text))

    signals: list[str] = []
    score = 0

    # Positive signals: explicit pricing numerics.
    if percent_count:
        signals.append(f"has_percent:{percent_count}")
        score += 2
        if percent_count >= 3:
            score += 2

    if bps_count:
        signals.append(f"has_bps:{bps_count}")
        score += 3

    # Strong pricing keywords (only meaningful if there are numeric signals too).
    strong_keywords = [
        "applicable margin",
        "applicable rate",
        "pricing level",
        "pricing grid",
        "commitment fee",
        "facility fee",
        "unused fee",
        "fronting fee",
        "letter of credit fee",
        "l/c fee",
        "margin",
        "spread",
    ]
    strong_hits = [k for k in strong_keywords if k in hay]
    if strong_hits:
        signals.append("kw_strong:" + ",".join(strong_hits[:5]))
        score += 3

    # Benchmark terms often appear in true pricing tables.
    benchmark_keywords = [
        "sofr",
        "term sofr",
        "libor",
        "eurocurrency",
        "base rate",
        "abr",
        "prime",
    ]
    bench_hits = [k for k in benchmark_keywords if k in hay]
    if bench_hits:
        signals.append("kw_benchmark:" + ",".join(bench_hits[:5]))
        score += 1

    # Status-based tables often omit explicit labels like "Applicable Margin" but still contain many % entries.
    if "level i" in hay and "status" in hay and percent_count >= 2:
        signals.append("looks_like_status_grid")
        score += 2

    is_toc_like = _looks_like_table_of_contents(table, text)
    if is_toc_like:
        signals.append("penalty:toc_like")
        score -= 10

    is_signature_like = _looks_like_signature_table(text)
    if is_signature_like:
        signals.append("penalty:signature_like")
        score -= 8

    is_lender_commitment_like = _looks_like_lender_commitment_table(text)
    if is_lender_commitment_like:
        signals.append("penalty:lender_commitment_like")
        score -= 12

    # Guardrail: if we saw zero numeric pricing markers, don't let generic keywords trick us into scoring high.
    if percent_count == 0 and bps_count == 0:
        score = min(score, 1)
        if score:
            signals.append("cap:no_numeric_markers")

    return PricingTableScore(
        table_anchor_id=table_anchor_id,
        order=order,
        score=score,
        signals=signals,
        percent_count=percent_count,
        bps_count=bps_count,
        is_toc_like=is_toc_like,
        is_signature_like=is_signature_like,
        is_lender_commitment_like=is_lender_commitment_like,
    )


def build_pricing_index(doc_ir: dict[str, Any]) -> dict[str, Any]:
    """Deterministic pricing index over a document IR (no embeddings, no chunk RAG).

    This is intentionally simple and auditable:
    - Scores every parsed table using lexical + numeric signals.
    - Emits a ranked list that downstream compiler stages can use.
    """

    tables = list(doc_ir.get("tables") or [])
    scored: list[PricingTableScore] = []
    for t in tables:
        if not isinstance(t, dict):
            continue
        scored.append(score_pricing_table(t))

    scored_sorted = sorted(scored, key=lambda s: (s.score, -s.order), reverse=True)

    return {
        "item_id": doc_ir.get("item_id"),
        "tables_scored": [
            {
                "table_anchor_id": s.table_anchor_id,
                "order": s.order,
                "score": s.score,
                "signals": s.signals,
                "percent_count": s.percent_count,
                "bps_count": s.bps_count,
                "is_toc_like": s.is_toc_like,
                "is_signature_like": s.is_signature_like,
                "is_lender_commitment_like": s.is_lender_commitment_like,
            }
            for s in scored_sorted
        ],
    }
