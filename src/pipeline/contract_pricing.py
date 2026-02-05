from __future__ import annotations

import asyncio
import json
import re
import time
import traceback
from pathlib import Path
from typing import Any, Iterable

from .config import Paths, REQUIRED_MODEL, REQUIRED_REASONING, prompt_hash, update_manifest
from .doc_ir import build_doc_ir, write_doc_ir
from .contract_schemas import (
    ContractPricingArtifact,
    ContractPricingModel,
    ContractPricingTableExtract,
    EvidenceText,
    FlatPricingItem,
    PricingAdjustment,
    PricingCell,
    PricingGrid,
    PricingRegime,
    PricingTier,
    RateOption,
)
from .pricing_index import build_pricing_index, score_pricing_table
from .pricing_heuristics import (
    looks_like_adjustment_anchor,
    looks_like_definition_anchor,
    looks_like_pricing_anchor,
    looks_like_regime_intro,
)
from .utils import assert_exists

# Reuse gateway helper/constants from indexing to avoid duplicating config.
from .indexing import (  # type: ignore
    _ensure_gateway_client_async,
    DEFAULT_GATEWAY_URL,
)

_PLACEHOLDER_TEXT_MARKERS = (
    "verbatim label",
    "verbatim",
    "no_spaces_id",
    "table_context_json",
    "output schema",
)

_DEFINED_TERM_RE = re.compile(r'[“"]\s*([^”"]{2,120}?)\s*[”"]\s*:')


def _extract_defined_term_before_table(anchor_text: str) -> str | None:
    """Best-effort: find the last defined term label preceding a [[TABLE]] marker."""

    if not isinstance(anchor_text, str) or not anchor_text.strip():
        return None

    idx = anchor_text.find("[[TABLE]]")
    prefix = anchor_text[:idx] if idx != -1 else anchor_text
    matches = list(_DEFINED_TERM_RE.finditer(prefix))
    if not matches:
        return None
    return matches[-1].group(1).strip() or None


def _extract_defined_term_for_table(
    *, neighbor_anchors: list[dict[str, Any]], table_anchor_id: str
) -> str | None:
    """Find the most likely defined-term label for a table using nearby anchor text.

    In many normalized filings, the table anchor itself contains only the [[TABLE]] payload, while
    the immediately preceding sentence anchor contains the definition label (e.g., “Facility Fee Rate”: ...).
    """

    if not table_anchor_id:
        return None
    if not isinstance(neighbor_anchors, list) or not neighbor_anchors:
        return None

    table_idx: int | None = None
    for i, a in enumerate(neighbor_anchors):
        if isinstance(a, dict) and str(a.get("anchor_id") or "") == table_anchor_id:
            table_idx = i
            break

    # Prefer scanning backwards from the table anchor.
    if table_idx is not None:
        for j in range(table_idx - 1, -1, -1):
            a = neighbor_anchors[j]
            if not isinstance(a, dict):
                continue
            txt = a.get("text")
            if not isinstance(txt, str) or not txt.strip():
                continue
            matches = list(_DEFINED_TERM_RE.finditer(txt))
            if matches:
                return matches[-1].group(1).strip() or None

    # Fallback: full concatenation, but use the last match overall.
    joined = "\n\n".join(
        str(a.get("text") or "") for a in neighbor_anchors if isinstance(a, dict) and a.get("text")
    )
    matches = list(_DEFINED_TERM_RE.finditer(joined))
    if matches:
        return matches[-1].group(1).strip() or None

    return None


def _parse_percent_to_bps(raw: str) -> float:
    import re

    txt = (raw or "").strip()
    m = re.fullmatch(r"(\d+(?:\.\d+)?)\s*%", txt)
    if not m:
        raise ValueError(f"Not a percent value: {raw!r}")
    return float(m.group(1)) * 100.0


def _extract_adjustments_from_hints(
    *,
    adjustment_hints: list[dict[str, Any]],
    applies_to_rate_option_ids: list[str],
) -> list[PricingAdjustment]:
    import re

    out: list[PricingAdjustment] = []
    for hint in adjustment_hints or []:
        if not isinstance(hint, dict):
            continue
        aid = str(hint.get("anchor_id") or "").strip()
        txt = str(hint.get("text") or "").strip()
        if not aid or not txt:
            continue

        hay = txt.lower()
        # Determine sign (explicitly prefer reduced/decreased when present).
        sign = 1.0
        if any(k in hay for k in ["reduc", "decreas", "lower"]):
            sign = -1.0
        elif any(k in hay for k in ["increas", "raise", "higher", "addit"]):
            sign = 1.0

        # Extract magnitude.
        # Common patterns:
        # - "reduced by 1 basis point"
        # - "reduced by one (1) basis point"
        # - "increased by (2) bps"
        mag: float | None = None
        m = re.search(
            r"\(\s*(\d+(?:\.\d+)?)\s*\)\s*(?:basis point|basis points|\bbp\b|\bbps\b)",
            txt,
            flags=re.I,
        )
        if m:
            mag = float(m.group(1))
        else:
            m = re.search(r"(\d+(?:\.\d+)?)\s*(?:basis point|basis points|\bbp\b|\bbps\b)", txt, flags=re.I)
            if m:
                mag = float(m.group(1))

        if mag is None:
            word_map = {
                "one": 1.0,
                "two": 2.0,
                "three": 3.0,
                "four": 4.0,
                "five": 5.0,
                "six": 6.0,
                "seven": 7.0,
                "eight": 8.0,
                "nine": 9.0,
                "ten": 10.0,
            }
            for word, val in word_map.items():
                if re.search(rf"\b{word}\b\s*(?:basis point|basis points|\bbp\b|\bbps\b)", txt, flags=re.I):
                    mag = val
                    break

        if mag is None:
            continue

        delta_bps = sign * mag

        out.append(
            PricingAdjustment(
                adjustment_id=_slugify_id(f"adj_{aid.lower()}"),
                label="bps adjustment",
                delta_bps=delta_bps,
                applies_to_rate_option_ids=list(applies_to_rate_option_ids or []),
                applies_when=EvidenceText(text=txt, source_refs=[aid]),
                floor_bps=None,
                cap_bps=None,
                source_refs=[aid],
            )
        )
    return out


def _try_extract_single_row_status_fee_table(
    *, table_context: dict[str, Any]
) -> ContractPricingTableExtract | None:
    """Deterministic extraction for 1-row 'Level I Status ...' fee rate tables.

    Common pattern in credit agreements:
      “Facility Fee Rate”: the applicable percentage per annum set forth below based upon the Status...
      [[TABLE]]  Level I Status | ... | Level VI Status
               0.100% | ... | 0.300%
    """

    table = table_context.get("table") or {}
    if not isinstance(table, dict):
        return None

    if not table.get("structured"):
        return None

    cols = table.get("columns") or []
    rows = table.get("rows") or []
    if not isinstance(cols, list) or not isinstance(rows, list) or len(rows) != 1:
        return None

    def _is_status_col(c: str) -> bool:
        hay = (c or "").strip().lower()
        return "status" in hay and hay.startswith("level ")

    status_cols = [str(c) for c in cols if isinstance(c, str) and c.strip()]
    if len(status_cols) < 4:
        return None
    if not all(_is_status_col(c) for c in status_cols):
        return None

    row0 = rows[0] if isinstance(rows[0], dict) else None
    if not row0:
        return None

    row_label = str(row0.get("row_label") or "").strip()
    cells = row0.get("cells") or {}
    if not row_label or not isinstance(cells, dict):
        return None

    # Interpret: row_label is the value for the first status column; remaining values are in cells keyed by column.
    values_by_status: dict[str, str] = {status_cols[0]: row_label}
    for col in status_cols[1:]:
        v = str(cells.get(col) or "").strip()
        if not v:
            return None
        values_by_status[col] = v

    # Ensure all values are percentages.
    try:
        bps_by_status = {k: _parse_percent_to_bps(v) for k, v in values_by_status.items()}
    except Exception:
        return None

    table_anchor_id = str(table.get("table_anchor_id") or "").strip()
    if not table_anchor_id:
        return None

    neighbor_anchors = list(table_context.get("neighbor_anchors") or [])
    fee_label = _extract_defined_term_for_table(
        neighbor_anchors=neighbor_anchors,
        table_anchor_id=table_anchor_id,
    )
    if not fee_label:
        return None

    rate_option_id = _slugify_id(fee_label)
    rate_option = RateOption(
        rate_option_id=rate_option_id,
        label_raw=fee_label,
        kind="fee",
        fee_basis=None,
        source_refs=[table_anchor_id],
    )

    tiers: list[PricingTier] = []
    cells_out: list[PricingCell] = []
    for status_label in status_cols:
        tier_id = _slugify_id(status_label)
        tiers.append(PricingTier(tier_id=tier_id, label_raw=status_label, tests=[], source_refs=[table_anchor_id]))
        cells_out.append(
            PricingCell(
                tier_id=tier_id,
                rate_option_id=rate_option_id,
                value_bps=bps_by_status[status_label],
                source_refs=[table_anchor_id],
            )
        )

    adjustments = _extract_adjustments_from_hints(
        adjustment_hints=list(table_context.get("adjustment_hints") or []),
        applies_to_rate_option_ids=[rate_option_id],
    )

    grid = PricingGrid(
        grid_id=_slugify_id(f"grid_{table_anchor_id.lower()}"),
        table_anchor_id=table_anchor_id,
        facility_id=None,
        tier_metric_label_raw="Status",
        tiers=tiers,
        rate_option_ids=[rate_option_id],
        cells=cells_out,
        source_refs=[table_anchor_id],
    )

    extract = ContractPricingTableExtract(
        rate_options=[rate_option],
        grid=grid,
        adjustments=adjustments,
        flat_items=[],
    )
    return extract


def _looks_like_pricing_table(table: dict[str, Any]) -> bool:
    # Deprecated in favor of deterministic scoring; kept as a narrow wrapper for callers.
    score = score_pricing_table(table)
    return score.score >= 3


def _looks_like_pricing_anchor(text: str) -> bool:
    return looks_like_pricing_anchor(text)


def _looks_like_definition_anchor(text: str) -> bool:
    return looks_like_definition_anchor(text)


def _looks_like_regime_intro(text: str) -> bool:
    return looks_like_regime_intro(text)


def _looks_like_adjustment_anchor(text: str) -> bool:
    return looks_like_adjustment_anchor(text)


def build_pricing_context(doc_ir: dict[str, Any], *, neighbor_window: int = 6, header_anchors: int = 40) -> dict[str, Any]:
    anchors: list[dict[str, Any]] = list(doc_ir.get("anchors") or [])
    tables: list[dict[str, Any]] = list(doc_ir.get("tables") or [])

    anchors_by_id = {a["anchor_id"]: a for a in anchors}
    anchors_by_order = {int(a["order"]): a for a in anchors if isinstance(a.get("order"), int)}
    max_order = max(anchors_by_order.keys()) if anchors_by_order else -1

    def _neighbors(order: int) -> list[int]:
        return [o for o in range(max(0, order - neighbor_window), min(max_order, order + neighbor_window) + 1)]

    included: dict[str, set[str]] = {}

    def _include(anchor_id: str, reason: str) -> None:
        if anchor_id not in anchors_by_id:
            return
        included.setdefault(anchor_id, set()).add(reason)

    # 0) Always include a header slice (title/date/parties often live here).
    for a in anchors[: max(0, header_anchors)]:
        _include(a["anchor_id"], "header_slice")

    # 1) Pricing tables (+ neighbors)
    selected_tables: list[dict[str, Any]] = []
    table_scores: dict[str, Any] = {}
    for t in tables:
        score = score_pricing_table(t)
        table_scores[str(score.table_anchor_id)] = {
            "score": score.score,
            "signals": score.signals,
        }
        if score.score < 3:
            continue
        selected_tables.append(t)
        table_anchor_id = str(t.get("table_anchor_id"))
        _include(table_anchor_id, f"pricing_table:{table_anchor_id}")
        order = int(t.get("order") or 0)
        for o in _neighbors(order):
            _include(anchors_by_order[o]["anchor_id"], f"pricing_table_neighbor:{table_anchor_id}")

    # 2) Pricing anchors (non-table) (+ neighbors)
    for a in anchors:
        txt = str(a.get("text") or "")
        if not _looks_like_pricing_anchor(txt):
            continue
        aid = str(a.get("anchor_id"))
        _include(aid, "pricing_keyword_match")
        order = int(a.get("order") or 0)
        for o in _neighbors(order):
            _include(anchors_by_order[o]["anchor_id"], f"pricing_keyword_neighbor:{aid}")

    # 3) Definitions that help disambiguate benchmark/bases (+ neighbors)
    for a in anchors:
        txt = str(a.get("text") or "")
        if not _looks_like_definition_anchor(txt):
            continue
        aid = str(a.get("anchor_id"))
        _include(aid, "definition_anchor_match")
        order = int(a.get("order") or 0)
        for o in _neighbors(order):
            _include(anchors_by_order[o]["anchor_id"], f"definition_neighbor:{aid}")

    # Finalize anchors list in stable order.
    selected_anchor_ids = {aid for aid in included.keys()}
    selected_anchors: list[dict[str, Any]] = []
    for a in anchors:
        aid = a["anchor_id"]
        if aid not in selected_anchor_ids:
            continue
        txt = str(a.get("text") or "")
        selected_anchors.append(
            {
                "anchor_id": aid,
                "anchor_type": a.get("anchor_type"),
                "order": a.get("order"),
                "text": txt,
                "reasons": sorted(included.get(aid) or []),
            }
        )

    adjustment_hints: list[dict[str, Any]] = []
    for a in selected_anchors:
        txt = str(a.get("text") or "")
        if _looks_like_adjustment_anchor(txt):
            adjustment_hints.append({"anchor_id": a["anchor_id"], "text": txt.strip()})

    # Tables in stable order.
    selected_tables_sorted = sorted(selected_tables, key=lambda x: int(x.get("order") or 0))
    selected_tables_out: list[dict[str, Any]] = []
    for t in selected_tables_sorted:
        order = int(t.get("order") or 0)
        # Determine a regime hint by scanning backwards for a strong "intro" line.
        regime_hint: dict[str, Any] | None = None
        for back in range(order - 1, max(-1, order - neighbor_window - 1), -1):
            a = anchors_by_order.get(back)
            if not a:
                continue
            txt = str(a.get("text") or "")
            if _looks_like_regime_intro(txt):
                regime_hint = {"anchor_id": a.get("anchor_id"), "text": txt.strip()}
                break

        selected_tables_out.append(
            {
                "table_anchor_id": t.get("table_anchor_id"),
                "order": order,
                "regime_hint": regime_hint,
                "structured": t.get("structured"),
                "columns": t.get("columns"),
                "rows": t.get("rows"),
                "raw": t.get("raw"),
                "table_score": (table_scores.get(str(t.get("table_anchor_id"))) or {}).get("score"),
                "table_score_signals": (table_scores.get(str(t.get("table_anchor_id"))) or {}).get("signals"),
                "reasons": ["pricing_table_score>=3"],
            }
        )

    return {
        "item_id": doc_ir.get("item_id"),
        "selection": {
            "neighbor_window": neighbor_window,
            "header_anchors": header_anchors,
        },
        "tables": selected_tables_out,
        "anchors": selected_anchors,
        "adjustment_hints": adjustment_hints,
    }


def _render_table_prompt(template: str, table_context_json: str) -> str:
    template = template.strip()
    if "{table_context_json}" in template:
        return template.replace("{table_context_json}", table_context_json)
    return f"{template}\n\n=== TABLE_CONTEXT_JSON ===\n{table_context_json}"


async def _call_gateway(
    *,
    client: Any,
    prompt: str,
    model: str,
    temperature: float,
    reasoning: str | None,
) -> str:
    reasoning_payload = {"effort": reasoning} if reasoning else None
    result = await client.complete_response(
        model=model,
        input_messages=[{"role": "user", "content": prompt}],
        reasoning=reasoning_payload,
        temperature=temperature,
        max_output_tokens=None,
        metadata=None,
    )
    if isinstance(result, dict):
        return result.get("text") or ""
    return str(result)


def _validate_pricing_output(
    *,
    raw_text: str,
    context: dict[str, Any],
) -> ContractPricingModel:
    try:
        parsed = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"LLM output is not valid JSON: {exc}") from exc

    model = ContractPricingModel.model_validate(parsed)

    # Structural cross-check: every table_anchor_id in context must appear in some grid.
    input_table_ids = {str(t.get("table_anchor_id")) for t in context.get("tables", []) if t.get("table_anchor_id")}
    output_table_ids: set[str] = set()
    for regime in model.pricing_regimes:
        for grid in regime.grids:
            output_table_ids.add(grid.table_anchor_id)
    missing = sorted(input_table_ids - output_table_ids)
    if missing:
        raise ValueError(f"Missing pricing grids for table anchors: {', '.join(missing)}")

    # Regime correctness: when we detected a regime_hint for a table, require that the table lives
    # inside a regime whose applies_when cites that hint anchor.
    hint_by_table: dict[str, str] = {}
    for t in context.get("tables", []):
        tid = str(t.get("table_anchor_id") or "")
        hint = t.get("regime_hint") or None
        if not tid or not hint:
            continue
        hid = str(hint.get("anchor_id") or "")
        if hid:
            hint_by_table[tid] = hid

    if hint_by_table:
        # Build mapping: table_anchor_id -> regimes that include it.
        table_to_regimes: dict[str, list[Any]] = {}
        for regime in model.pricing_regimes:
            regime_tables = {g.table_anchor_id for g in regime.grids}
            for tid in regime_tables:
                table_to_regimes.setdefault(tid, []).append(regime)

        failures: list[str] = []
        for table_id, hint_anchor_id in sorted(hint_by_table.items()):
            regimes = table_to_regimes.get(table_id) or []
            if not regimes:
                continue
            ok = any(hint_anchor_id in (reg.applies_when.source_refs or []) for reg in regimes)
            if not ok:
                failures.append(f"{table_id} missing applies_when.source_refs including hint {hint_anchor_id}")
        if failures:
            raise ValueError("Regime hint mismatch: " + "; ".join(failures))

    # Adjustment hints: if the context provided explicit adjustment anchors, they must be represented.
    hinted_adjustments = {str(h.get("anchor_id")) for h in context.get("adjustment_hints", []) if h.get("anchor_id")}
    if hinted_adjustments:
        referenced: set[str] = set()
        for regime in model.pricing_regimes:
            for adj in regime.adjustments:
                referenced.update(adj.source_refs or [])
                referenced.update(adj.applies_when.source_refs or [])
        missing_adj = sorted(hinted_adjustments - referenced)
        if missing_adj:
            raise ValueError(f"Missing PricingAdjustment(s) referencing anchors: {', '.join(missing_adj)}")

    # Referential integrity: rate_option_ids in grids must exist.
    global_rate_option_ids = {ro.rate_option_id for ro in model.rate_options}
    for regime in model.pricing_regimes:
        for grid in regime.grids:
            unknown = sorted(set(grid.rate_option_ids) - global_rate_option_ids)
            if unknown:
                raise ValueError(
                    f"Grid {grid.grid_id} references unknown rate_option_ids: {', '.join(unknown)}"
                )
            tier_ids = {t.tier_id for t in grid.tiers}
            for cell in grid.cells:
                if cell.tier_id not in tier_ids:
                    raise ValueError(
                        f"Grid {grid.grid_id} has cell with unknown tier_id {cell.tier_id!r}"
                    )
                if cell.rate_option_id not in global_rate_option_ids:
                    raise ValueError(
                        f"Grid {grid.grid_id} has cell with unknown rate_option_id {cell.rate_option_id!r}"
                    )

    return model


def _table_surface_text(table: dict[str, Any]) -> str:
    # Similar to pricing_index._table_text, but kept local to avoid importing internals.
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
                parts.extend(str(v) for v in cells.values() if isinstance(v, str) and v.strip())
    return "\n".join(parts)


def _has_placeholder_text(value: str) -> bool:
    hay = (value or "").strip().lower()
    return any(marker in hay for marker in _PLACEHOLDER_TEXT_MARKERS)


def _grounded_in_table(label_raw: str, surface_text: str) -> bool:
    label = (label_raw or "").strip()
    if not label:
        return False
    # Grounding check for labels against the table surface text.
    #
    # We want to catch template-copy hallucinations, but we also need to tolerate:
    # - line wraps inside a single label (e.g., "no\n rating") and
    # - fixed-width tables where numeric columns can appear between words in the raw text.
    import re

    def _norm(s: str) -> str:
        return re.sub(r"\s+", " ", (s or "").lower()).strip()

    label_norm = _norm(label)
    surface_norm = _norm(surface_text or "")

    # Fast path: normalized substring match.
    if label_norm and label_norm in surface_norm:
        return True

    # Fallback: token-in-order match within a bounded span.
    #
    # Example that motivated this:
    #   label:   "BBB-/BBB or below (including no rating)"
    #   surface: "BBB-/BBB or below (including no 1.50% rating)"
    tokens = re.findall(r"[a-z0-9][a-z0-9%./+-]*", label_norm)
    if not tokens:
        return False

    idx = 0
    first: int | None = None
    for tok in tokens:
        pos = surface_norm.find(tok, idx)
        if pos == -1:
            return False
        if first is None:
            first = pos
        idx = pos + len(tok)

    # Avoid matching tokens that are scattered arbitrarily across the whole table/neighbor context.
    if first is not None and (idx - first) > 500:
        return False

    return True


_LABEL_STOPWORDS = {
    "a",
    "an",
    "and",
    "any",
    "as",
    "at",
    "by",
    "for",
    "from",
    "in",
    "into",
    "is",
    "it",
    "its",
    "of",
    "on",
    "or",
    "that",
    "the",
    "to",
    "under",
    "with",
    "without",
}


def _grounded_in_table_relaxed_label(label_raw: str, surface_text: str) -> bool:
    """Relaxed grounding used ONLY for unstructured/pseudo-table label_raw fields.

    For prose-encoded "tables", the extracted label is often still grounded but not in
    the same word order as the source (e.g., "interest on drawings..." vs "drawings ...
    interest"). We still want to reject hallucinations, but we don't want label word
    order to be a hard blocker for unstructured contexts.

    Strategy:
    - Require all "meaningful" tokens (non-stopwords, len>=4) to appear somewhere.
    - Require those token occurrences to fall within a bounded character span so the
      tokens aren't scattered arbitrarily across the whole neighbor window.
    """

    label = (label_raw or "").strip()
    surface = (surface_text or "").strip()
    if not label or not surface:
        return False

    label_norm = re.sub(r"\s+", " ", label.lower()).strip()
    surface_norm = re.sub(r"\s+", " ", surface.lower()).strip()

    tokens = [
        tok
        for tok in re.findall(r"[a-z0-9]+", label_norm)
        if len(tok) >= 4 and tok not in _LABEL_STOPWORDS
    ]
    if not tokens:
        return False

    positions: list[int] = []
    for tok in tokens:
        pos = surface_norm.find(tok)
        if pos != -1:
            positions.append(pos)

    # Require substantial token overlap, but don't require exact word-for-word match.
    # This prevents long, mostly-correct labels from being rejected due to a couple missing tokens.
    overlap = len(positions) / max(1, len(tokens))
    if overlap < 0.7:
        return False
    if len(positions) < 3:
        return False

    return (max(positions) - min(positions)) <= 500


def _validate_table_extract(extract: ContractPricingTableExtract, *, table_context: dict[str, Any]) -> None:
    """Fail loudly when the table extract is obviously ungrounded or internally inconsistent."""

    table = table_context.get("table") or {}
    table_anchor_id = str(table.get("table_anchor_id") or "")
    if not table_anchor_id:
        raise ValueError("TABLE_CONTEXT_JSON.table.table_anchor_id is missing")

    surface_parts: list[str] = [_table_surface_text(table)]
    for a in table_context.get("neighbor_anchors", []) or []:
        if isinstance(a, dict):
            txt = a.get("text")
            if isinstance(txt, str) and txt.strip():
                surface_parts.append(txt.strip())
    surface = "\n\n".join(p for p in surface_parts if p)

    errors: list[str] = []

    table_signals = table.get("table_score_signals") or []
    if not isinstance(table_signals, list):
        table_signals = []

    def _is_status_label(text: str) -> bool:
        hay = (text or "").lower()
        return "status" in hay or hay.startswith("level ")

    is_unstructured = not bool(table.get("structured", True))

    # Rate options must be grounded in the table surface.
    for ro in extract.rate_options:
        if _has_placeholder_text(ro.label_raw):
            errors.append(f"rate_option.label_raw appears to be a placeholder: {ro.label_raw!r}")
        if not is_unstructured and not _grounded_in_table(ro.label_raw, surface):
            errors.append(f"rate_option.label_raw not found in table text: {ro.label_raw!r}")

    # Tiers must be grounded in the table surface.
    for tier in extract.grid.tiers:
        if _has_placeholder_text(tier.label_raw):
            errors.append(f"tier.label_raw appears to be a placeholder: {tier.label_raw!r}")
        if not _grounded_in_table(tier.label_raw, surface):
            errors.append(f"tier.label_raw not found in table text: {tier.label_raw!r}")

    # Axis sanity for "Status"-style pricing tables:
    # - Status levels ("Level I Status", etc.) should be tiers, not rate options.
    if "looks_like_status_grid" in table_signals:
        status_tiers = [t for t in extract.grid.tiers if _is_status_label(t.label_raw)]
        status_rate_options = [ro for ro in extract.rate_options if _is_status_label(ro.label_raw)]
        if not status_tiers:
            errors.append("table looks like a status grid but no tier labels look like statuses")
        if status_rate_options and len(status_rate_options) >= max(1, len(status_tiers)):
            errors.append("table looks like a status grid but status labels were encoded as rate options (axis swap)")

    # Every cell must cite the table anchor (minimum evidence).
    for cell in extract.grid.cells:
        if table_anchor_id not in (cell.source_refs or []):
            errors.append(
                f"cell missing table_anchor_id in source_refs: tier_id={cell.tier_id!r} "
                f"rate_option_id={cell.rate_option_id!r}"
            )

    # Referential integrity within the extract.
    rate_option_ids = [ro.rate_option_id for ro in extract.rate_options]
    if len(set(rate_option_ids)) != len(rate_option_ids):
        errors.append("duplicate rate_option_id values within rate_options")

    grid_rate_option_ids = list(extract.grid.rate_option_ids or [])
    if len(set(grid_rate_option_ids)) != len(grid_rate_option_ids):
        errors.append("duplicate ids in grid.rate_option_ids")

    unknown_grid_rate_options = sorted(set(grid_rate_option_ids) - set(rate_option_ids))
    if unknown_grid_rate_options:
        errors.append(
            "grid.rate_option_ids contains ids missing from rate_options: "
            + ", ".join(unknown_grid_rate_options)
        )

    tier_ids = [t.tier_id for t in extract.grid.tiers]
    if len(set(tier_ids)) != len(tier_ids):
        errors.append("duplicate tier_id values within grid.tiers")

    # Require a full tier x rate_option matrix (pricing tables are typically complete grids).
    expected_pairs = {(tid, roid) for tid in tier_ids for roid in grid_rate_option_ids}
    found_pairs = {(c.tier_id, c.rate_option_id) for c in extract.grid.cells}
    missing_pairs = sorted(expected_pairs - found_pairs)
    extra_pairs = sorted(found_pairs - expected_pairs)
    if missing_pairs:
        errors.append(f"grid.cells missing {len(missing_pairs)} tier/rate_option combinations")
    if extra_pairs:
        errors.append(f"grid.cells has {len(extra_pairs)} unexpected tier/rate_option combinations")

    # Placeholder guardrails on IDs (schema enforces format, but not semantic grounding).
    for ro in extract.rate_options:
        if _has_placeholder_text(ro.rate_option_id):
            errors.append(f"rate_option_id appears to be a placeholder: {ro.rate_option_id!r}")
    if _has_placeholder_text(extract.grid.grid_id):
        errors.append(f"grid_id appears to be a placeholder: {extract.grid.grid_id!r}")

    if errors:
        raise ValueError("Table extract failed grounding checks: " + "; ".join(errors))


def _slugify_id(raw: str, *, max_len: int = 64) -> str:
    import re

    s = (raw or "").strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    if not s:
        s = "id"
    if len(s) > max_len:
        s = s[:max_len].rstrip("_")
    return s


def _merge_rate_options(extracts: list[ContractPricingTableExtract]) -> tuple[list[RateOption], dict[str, str]]:
    """Return (global rate_options, remap old_id -> new_id)."""

    # Canonicalize by the semantic identity of the option, not the model-generated ID.
    canonical: dict[tuple[str, str, str | None], RateOption] = {}
    remap: dict[str, str] = {}

    for ex in extracts:
        for ro in ex.rate_options:
            key = (ro.label_raw, ro.kind, ro.fee_basis)
            if key not in canonical:
                base = _slugify_id(ro.label_raw)
                # Ensure uniqueness across collisions.
                candidate = base
                i = 2
                used_ids = {v.rate_option_id for v in canonical.values()}
                while candidate in used_ids:
                    candidate = f"{base}_{i}"
                    i += 1
                canonical[key] = RateOption(
                    rate_option_id=candidate,
                    label_raw=ro.label_raw,
                    kind=ro.kind,
                    fee_basis=ro.fee_basis,
                    source_refs=ro.source_refs,
                )
            remap[ro.rate_option_id] = canonical[key].rate_option_id

    return list(canonical.values()), remap


def _apply_rate_option_remap(extract: ContractPricingTableExtract, remap: dict[str, str]) -> ContractPricingTableExtract:
    # Remap rate options inside the extract's grid + adjustments + flat items.
    grid = extract.grid
    grid.rate_option_ids = [remap.get(x, x) for x in grid.rate_option_ids]
    for cell in grid.cells:
        cell.rate_option_id = remap.get(cell.rate_option_id, cell.rate_option_id)
    for adj in extract.adjustments:
        adj.applies_to_rate_option_ids = [remap.get(x, x) for x in adj.applies_to_rate_option_ids]
    for fi in extract.flat_items:
        fi.rate_option_id = remap.get(fi.rate_option_id, fi.rate_option_id)
    return extract


def _build_regimes_from_tables(
    *,
    table_contexts: list[dict[str, Any]],
    extracts_by_table: dict[str, ContractPricingTableExtract],
) -> list[PricingRegime]:
    # Group by regime_hint.anchor_id when present, else "default".
    grouped: dict[str, list[str]] = {}
    meta: dict[str, dict[str, Any]] = {}
    for tc in table_contexts:
        t = tc["table"]
        table_anchor_id = str(t.get("table_anchor_id"))
        hint = t.get("regime_hint") or None
        if hint and hint.get("anchor_id"):
            key = str(hint["anchor_id"])
            meta.setdefault(key, {"label": str(hint.get("text") or "").strip(), "source_refs": [key]})
        else:
            key = "default"
            meta.setdefault(
                key,
                {
                    "label": "Default pricing regime",
                    "source_refs": [],
                },
            )
        grouped.setdefault(key, []).append(table_anchor_id)

    regimes: list[PricingRegime] = []
    for key, table_ids in grouped.items():
        label = meta[key]["label"] if key in meta else "Pricing regime"
        applies_text = label if key != "default" else "Default pricing (no explicit regime hint for these tables)."
        source_refs = meta[key].get("source_refs") or []
        if key == "default":
            # Cite the first table as evidence of existence.
            source_refs = [table_ids[0]] if table_ids else []
        grids = []
        adjustments = []
        flat_items = []
        for tid in table_ids:
            ex = extracts_by_table[tid]
            grids.append(ex.grid)
            adjustments.extend(ex.adjustments)
            flat_items.extend(ex.flat_items)
        regimes.append(
            PricingRegime(
                regime_id=_slugify_id(key),
                label=label,
                applies_when={"text": applies_text, "source_refs": source_refs},
                grids=grids,
                adjustments=adjustments,
                flat_items=flat_items,
                source_refs=source_refs,
            )
        )
    return regimes


def _extract_simple_fronting_fees(context: dict[str, Any]) -> list[FlatPricingItem]:
    """Deterministic extraction for obvious 'fronting fee' clauses (percent -> bps)."""

    import re

    out: list[FlatPricingItem] = []
    for a in context.get("anchors", []):
        txt = str(a.get("text") or "")
        hay = txt.lower()
        if "fronting fee" not in hay:
            continue
        # Capture the first explicit percentage (e.g., 0.125%).
        m = re.search(r"(\d+(?:\.\d+)?)\s*%", txt)
        if not m:
            continue
        pct = float(m.group(1))
        bps = pct * 100.0
        aid = str(a.get("anchor_id"))
        out.append(
            FlatPricingItem(
                item_id=f"fronting_fee_{aid.lower()}",
                rate_option_id="fronting_fee",  # will be remapped if we include a RateOption later
                value_bps=bps,
                applies_when=None,
                source_refs=[aid],
            )
        )
    return out


def run_contract_pricing(
    paths: Paths,
    item_ids: Iterable[str],
    prompt_path: Path,
    *,
    gateway_url: str | None = None,
    temperature: float = 0.0,
    reasoning: str | None = None,
    gateway_timeout: float | None = None,
    concurrency: int = 2,
    output_subdir: str = "contract_pricing_v1",
    attempts: int = 3,
) -> None:
    """End-to-end (v1): build doc IR -> build pricing context -> per-table extraction -> deterministic regime assembly.

    This is a deliberate rewrite away from a single monolithic prompt:
    - We extract each pricing table independently (smaller context, fewer omissions).
    - We assemble pricing regimes deterministically using regime_hint anchors.
    """

    assert_exists(prompt_path, message=f"Contract pricing prompt not found: {prompt_path}")

    # Enforce model and reasoning for now (consistent pipeline behavior).
    model = REQUIRED_MODEL
    reasoning = REQUIRED_REASONING

    prompt_digest = prompt_hash(prompt_path)
    table_prompt_template = prompt_path.read_text()

    out_root = paths.run_dir / "contract_pricing" / output_subdir
    out_root.mkdir(parents=True, exist_ok=True)

    doc_ir_dir = paths.run_dir / "doc_ir"
    doc_ir_dir.mkdir(parents=True, exist_ok=True)

    context_dir = paths.run_dir / "contract_pricing" / "_context" / output_subdir
    context_dir.mkdir(parents=True, exist_ok=True)

    items = list(item_ids)
    GatewayAgentClient = _ensure_gateway_client_async()
    sem = asyncio.Semaphore(max(1, concurrency))
    errors: list[tuple[str, str]] = []

    async def _process(item_id: str, client: Any) -> None:
        async with sem:
            try:
                # 1) Build + persist doc IR (anchors + parsed tables)
                doc_ir = build_doc_ir(paths, item_id)
                doc_ir_path = doc_ir_dir / f"{item_id}.json"
                write_doc_ir(doc_ir_path, doc_ir)

                # 1b) Deterministic pricing index (ranked table candidates)
                pricing_index_dir = paths.run_dir / "pricing_index"
                pricing_index_dir.mkdir(parents=True, exist_ok=True)
                pricing_index = build_pricing_index(doc_ir)
                (pricing_index_dir / f"{item_id}.json").write_text(json.dumps(pricing_index, indent=2))

                # 2) Build + persist context pack
                context = build_pricing_context(doc_ir)
                context_path = context_dir / f"{item_id}.json"
                context_path.write_text(json.dumps(context, indent=2))
                if not (context.get("tables") or []):
                    raise RuntimeError(
                        "No pricing tables detected (score>=3). "
                        "Inspect runs/<run_id>/pricing_index/<item_id>.json for scored candidates."
                    )

                # 3) Per-table extraction (strict JSON + Pydantic, retries)
                # Persist per-table contexts for audit/debug.
                per_table_dir = context_dir / item_id
                per_table_dir.mkdir(parents=True, exist_ok=True)

                anchors_by_order = {int(a["order"]): a for a in doc_ir.get("anchors", []) if isinstance(a.get("order"), int)}
                max_order = max(anchors_by_order.keys()) if anchors_by_order else -1

                def _neighbor_slice(order: int, k: int) -> list[dict[str, Any]]:
                    out: list[dict[str, Any]] = []
                    for o in range(max(0, order - k), min(max_order, order + k) + 1):
                        a = anchors_by_order.get(o)
                        if not a:
                            continue
                        out.append({"anchor_id": a.get("anchor_id"), "anchor_type": a.get("anchor_type"), "order": o, "text": a.get("text")})
                    return out

                def _range_slice(start_order: int, end_order: int) -> list[dict[str, Any]]:
                    out: list[dict[str, Any]] = []
                    for o in range(max(0, start_order), min(max_order, end_order) + 1):
                        a = anchors_by_order.get(o)
                        if not a:
                            continue
                        out.append({"anchor_id": a.get("anchor_id"), "anchor_type": a.get("anchor_type"), "order": o, "text": a.get("text")})
                    return out

                table_orders = sorted(
                    [int(t.get("order") or 0) for t in context.get("tables", []) if t.get("order") is not None]
                )

                table_contexts: list[dict[str, Any]] = []
                extracts_by_table: dict[str, ContractPricingTableExtract] = {}

                for t in context.get("tables", []):
                    table_anchor_id = str(t.get("table_anchor_id"))
                    order = int(t.get("order") or 0)
                    neighbor_anchors = _neighbor_slice(order, context["selection"]["neighbor_window"])

                    # Adjustment hints are usually in the immediate "tail" after a table, before the next table.
                    # Using the full neighbor window creates false positives (a later table's footnote leaking
                    # into an earlier table's context).
                    next_table_order = None
                    for o in table_orders:
                        if o > order:
                            next_table_order = o
                            break
                    tail_end = (next_table_order - 1) if next_table_order is not None else min(
                        max_order, order + context["selection"]["neighbor_window"]
                    )
                    tail_anchors = _range_slice(order + 1, tail_end)
                    adjustment_hints = [
                        {"anchor_id": a["anchor_id"], "text": str(a.get("text") or "").strip()}
                        for a in tail_anchors
                        if _looks_like_adjustment_anchor(str(a.get("text") or ""))
                    ]
                    table_context = {
                        "item_id": item_id,
                        "table": t,
                        "neighbor_anchors": neighbor_anchors,
                        "adjustment_hints": adjustment_hints,
                    }
                    table_contexts.append(table_context)
                    (per_table_dir / f"{table_anchor_id}.json").write_text(json.dumps(table_context, indent=2))

                    # Deterministic fast path for common 1-row status fee tables ("Facility Fee Rate", "L/C Fee Rate", etc.).
                    deterministic = _try_extract_single_row_status_fee_table(table_context=table_context)
                    if deterministic is not None:
                        # Ensure deterministic output still respects our hint contract.
                        hinted = {h["anchor_id"] for h in table_context.get("adjustment_hints", []) if h.get("anchor_id")}
                        if hinted:
                            referenced: set[str] = set()
                            for adj in deterministic.adjustments:
                                referenced.update(adj.source_refs or [])
                                referenced.update(adj.applies_when.source_refs or [])
                            missing = sorted(hinted - referenced)
                            if missing:
                                deterministic = None
                        if deterministic is not None:
                            _validate_table_extract(deterministic, table_context=table_context)
                            extracts_by_table[table_anchor_id] = deterministic
                            continue

                    rendered = _render_table_prompt(table_prompt_template, json.dumps(table_context, indent=2))
                    last_err: str | None = None
                    used = 0
                    for attempt in range(1, max(1, attempts) + 1):
                        used = attempt
                        prompt = rendered
                        if last_err:
                            prompt = (
                                f"{rendered}\n\n=== VALIDATION_ERRORS ===\n{last_err}\n\n"
                                "Regenerate the JSON to satisfy the schema and the errors above. Output JSON only."
                            )
                        try:
                            raw = await _call_gateway(
                                client=client,
                                prompt=prompt,
                                model=model,
                                temperature=temperature,
                                reasoning=reasoning,
                            )
                        except Exception as exc:
                            last_err = f"Gateway call failed: {type(exc).__name__}: {exc}"
                            continue
                        try:
                            ex = ContractPricingTableExtract.model_validate_json(raw)
                            if ex.grid.table_anchor_id != table_anchor_id:
                                raise ValueError(
                                    f"grid.table_anchor_id {ex.grid.table_anchor_id!r} does not match expected {table_anchor_id!r}"
                                )
                            _validate_table_extract(ex, table_context=table_context)
                            # If we provided explicit adjustment hints, require the output to encode them.
                            hinted = {h["anchor_id"] for h in table_context.get("adjustment_hints", []) if h.get("anchor_id")}
                            if hinted:
                                referenced: set[str] = set()
                                for adj in ex.adjustments:
                                    referenced.update(adj.source_refs or [])
                                    referenced.update(adj.applies_when.source_refs or [])
                                missing = sorted(hinted - referenced)
                                if missing:
                                    raise ValueError(
                                        f"Missing adjustments referencing anchors: {', '.join(missing)}"
                                    )
                            extracts_by_table[table_anchor_id] = ex
                            break
                        except Exception as exc:
                            last_err = str(exc)
                    if table_anchor_id not in extracts_by_table:
                        raise RuntimeError(f"Table {table_anchor_id} failed after {used} attempts. Last error: {last_err}")

                # 4) Canonicalize rate option IDs across tables and apply remap.
                global_rate_options, remap = _merge_rate_options(list(extracts_by_table.values()))
                for tid, ex in list(extracts_by_table.items()):
                    extracts_by_table[tid] = _apply_rate_option_remap(ex, remap)

                # Ensure deterministic fee extraction has a canonical option if present.
                flat_fronting = _extract_simple_fronting_fees(context)
                if flat_fronting:
                    # Add canonical fronting fee option if not already present.
                    if not any(ro.rate_option_id == "fronting_fee" for ro in global_rate_options):
                        global_rate_options.append(
                            RateOption(
                                rate_option_id="fronting_fee",
                                label_raw="Fronting Fee",
                                kind="fee",
                                fee_basis="letters_of_credit",
                                source_refs=[fi.source_refs[0] for fi in flat_fronting if fi.source_refs],
                            )
                        )
                    # Attach to default regime after regimes are built.

                # 5) Assemble regimes deterministically using regime_hint.
                regimes = _build_regimes_from_tables(table_contexts=table_contexts, extracts_by_table=extracts_by_table)

                # Attach deterministic flat items to default regime (if any).
                if flat_fronting:
                    for reg in regimes:
                        if reg.regime_id == "default":
                            reg.flat_items.extend(flat_fronting)

                pricing_model = ContractPricingModel(
                    issuer=None,
                    agreement={},
                    rate_options=global_rate_options,
                    pricing_regimes=regimes,
                )

                # Final validation (ensures all tables accounted for + hints satisfied + adjustments present when hinted).
                pricing_model = _validate_pricing_output(raw_text=json.dumps(pricing_model.model_dump()), context=context)

                artifact = ContractPricingArtifact(
                    schema_version="contract_pricing_v1",
                    stage="contract_pricing",
                    run_id=paths.run_id,
                    item_id=item_id,
                    created_at=int(time.time()),
                    gateway_url=gateway_url or DEFAULT_GATEWAY_URL,
                    model=model,
                    reasoning_effort=reasoning,
                    temperature=temperature,
                    prompt=str(prompt_path),
                    prompt_sha256=prompt_digest,
                    attempts_used=attempts,
                    pricing=pricing_model,
                )
                out_path = out_root / f"{item_id}.json"
                out_path.write_text(json.dumps(artifact.model_dump(), indent=2))
                return
            except Exception as exc:
                errors.append((item_id, str(exc)))
                err_path = out_root / f"{item_id}.error.txt"
                err_path.write_text(f"{exc}\n\n{traceback.format_exc()}")

    async def _runner() -> None:
        async with GatewayAgentClient(
            base_url=gateway_url or DEFAULT_GATEWAY_URL,
            timeout=gateway_timeout or 600.0,
        ) as client:
            await asyncio.gather(*(_process(item_id, client) for item_id in items))

    asyncio.run(_runner())

    manifest_path = paths.manifest_path
    if manifest_path.exists():
        update_manifest(
            manifest_path,
            contract_pricing_prompt=str(prompt_path),
            contract_pricing_prompt_sha256=prompt_digest,
        )

    if errors:
        joined = "; ".join(f"{item}: {msg}" for item, msg in errors)
        raise RuntimeError(f"Contract pricing completed with errors: {joined}")
