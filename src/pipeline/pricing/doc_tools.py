from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Literal

from pipeline.pricing.pricing_index import score_pricing_table
from pipeline.pricing.pricing_heuristics import looks_like_adjustment_anchor, looks_like_regime_intro


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


def _snippet(text: str, *, idx: int, query_len: int, window: int = 90) -> str:
    if not text:
        return ""
    a = max(0, idx - window)
    b = min(len(text), idx + query_len + window)
    return text[a:b].replace("\n", " ").strip()


@dataclass(frozen=True)
class DocBrowser:
    """Deterministic, no-embeddings document browser over doc_ir.

    Intended use: expose "full context" as explicit, auditable tools that an LLM planner can call.
    """

    doc_ir: dict[str, Any]

    def __post_init__(self) -> None:
        if not isinstance(self.doc_ir, dict):
            raise TypeError("doc_ir must be a dict")
        if not isinstance(self.doc_ir.get("anchors"), list):
            raise ValueError("doc_ir.anchors missing or invalid")
        if not isinstance(self.doc_ir.get("tables"), list):
            raise ValueError("doc_ir.tables missing or invalid")

    @property
    def item_id(self) -> str:
        return str(self.doc_ir.get("item_id") or "")

    def _anchors(self) -> list[dict[str, Any]]:
        return list(self.doc_ir.get("anchors") or [])

    def _tables(self) -> list[dict[str, Any]]:
        return list(self.doc_ir.get("tables") or [])

    def _anchors_by_id(self) -> dict[str, dict[str, Any]]:
        out: dict[str, dict[str, Any]] = {}
        for a in self._anchors():
            if isinstance(a, dict) and a.get("anchor_id"):
                out[str(a["anchor_id"])] = a
        return out

    def _anchors_by_order(self) -> dict[int, dict[str, Any]]:
        out: dict[int, dict[str, Any]] = {}
        for a in self._anchors():
            if not isinstance(a, dict):
                continue
            order = a.get("order")
            if isinstance(order, int):
                out[order] = a
        return out

    def _tables_by_id(self) -> dict[str, dict[str, Any]]:
        out: dict[str, dict[str, Any]] = {}
        for t in self._tables():
            if isinstance(t, dict) and t.get("table_anchor_id"):
                out[str(t["table_anchor_id"])] = t
        return out

    def catalog(self, *, max_tables: int | None = None) -> dict[str, Any]:
        """Compact, planner-friendly catalog of the whole document."""

        anchors = self._anchors()
        tables = self._tables()
        summaries: list[dict[str, Any]] = []
        for t in tables:
            if not isinstance(t, dict):
                continue
            table_anchor_id = str(t.get("table_anchor_id") or "")
            if not table_anchor_id:
                continue
            score = score_pricing_table(t)
            cols = [str(c) for c in (t.get("columns") or []) if isinstance(c, str) and c.strip()]
            row_labels: list[str] = []
            for row in (t.get("rows") or []) if isinstance(t.get("rows"), list) else []:
                if isinstance(row, dict):
                    rl = str(row.get("row_label") or "").strip()
                    if rl:
                        row_labels.append(rl)
            raw = t.get("raw")
            raw_preview = ""
            if isinstance(raw, str) and raw.strip():
                raw_preview = raw.strip().replace("\n", " ")[:240]

            summaries.append(
                {
                    "table_anchor_id": table_anchor_id,
                    "order": int(t.get("order") or 0),
                    "structured": bool(t.get("structured")),
                    "n_columns": len(cols),
                    "n_rows": len(t.get("rows") or []) if isinstance(t.get("rows"), list) else 0,
                    "columns_preview": cols[:10],
                    "row_labels_preview": row_labels[:8],
                    "raw_preview": raw_preview,
                    "pricing_score": score.score,
                    "pricing_signals": score.signals,
                }
            )

        summaries_sorted = sorted(summaries, key=lambda x: (int(x.get("pricing_score") or 0), -int(x.get("order") or 0)), reverse=True)
        if max_tables is not None:
            summaries_sorted = summaries_sorted[: max(0, int(max_tables))]

        return {
            "item_id": self.item_id,
            "n_anchors": len(anchors),
            "n_tables": len(tables),
            "tables": summaries_sorted,
        }

    def open_anchor(self, anchor_id: str, *, neighbor_window: int = 0) -> dict[str, Any]:
        anchors_by_id = self._anchors_by_id()
        anchors_by_order = self._anchors_by_order()
        anchor_id = str(anchor_id or "").strip()
        if not anchor_id:
            raise ValueError("anchor_id is required")
        a = anchors_by_id.get(anchor_id)
        if not a:
            raise KeyError(f"Unknown anchor_id: {anchor_id}")

        order = int(a.get("order") or 0)
        max_order = max(anchors_by_order.keys()) if anchors_by_order else order

        neighbors: list[dict[str, Any]] = []
        k = max(0, int(neighbor_window))
        for o in range(max(0, order - k), min(max_order, order + k) + 1):
            rec = anchors_by_order.get(o)
            if not rec:
                continue
            neighbors.append(
                {
                    "anchor_id": rec.get("anchor_id"),
                    "anchor_type": rec.get("anchor_type"),
                    "order": rec.get("order"),
                    "text": rec.get("text"),
                }
            )

        return {
            "item_id": self.item_id,
            "anchor": {
                "anchor_id": a.get("anchor_id"),
                "anchor_type": a.get("anchor_type"),
                "order": a.get("order"),
                "text": a.get("text"),
            },
            "neighbor_window": k,
            "neighbors": neighbors,
        }

    def open_table(
        self,
        table_anchor_id: str,
        *,
        neighbor_window: int = 6,
    ) -> dict[str, Any]:
        tables_by_id = self._tables_by_id()
        anchors_by_order = self._anchors_by_order()

        table_anchor_id = str(table_anchor_id or "").strip()
        if not table_anchor_id:
            raise ValueError("table_anchor_id is required")
        t = tables_by_id.get(table_anchor_id)
        if not t:
            raise KeyError(f"Unknown table_anchor_id: {table_anchor_id}")

        order = int(t.get("order") or 0)
        max_order = max(anchors_by_order.keys()) if anchors_by_order else order
        k = max(0, int(neighbor_window))

        def _slice(start: int, end: int) -> list[dict[str, Any]]:
            out: list[dict[str, Any]] = []
            for o in range(max(0, start), min(max_order, end) + 1):
                a = anchors_by_order.get(o)
                if not a:
                    continue
                out.append(
                    {
                        "anchor_id": a.get("anchor_id"),
                        "anchor_type": a.get("anchor_type"),
                        "order": a.get("order"),
                        "text": a.get("text"),
                    }
                )
            return out

        neighbor_anchors = _slice(order - k, order + k)

        # Tail slice: from just after this table until just before the next table (or end of window).
        table_orders = sorted(int(x.get("order") or 0) for x in self._tables() if isinstance(x, dict) and x.get("order") is not None)
        next_table_order = None
        for o in table_orders:
            if o > order:
                next_table_order = o
                break
        tail_end = (next_table_order - 1) if next_table_order is not None else min(max_order, order + k)
        tail_anchors = _slice(order + 1, tail_end)

        adjustment_candidates = [
            {"anchor_id": a["anchor_id"], "text": str(a.get("text") or "").strip()}
            for a in tail_anchors
            if looks_like_adjustment_anchor(str(a.get("text") or ""))
        ]

        # Regime intro candidate: scan backwards within the neighbor window for a strong intro line.
        regime_hint = None
        for back in range(order - 1, max(-1, order - k - 1), -1):
            a = anchors_by_order.get(back)
            if not a:
                continue
            txt = str(a.get("text") or "").strip()
            if looks_like_regime_intro(txt):
                regime_hint = {"anchor_id": a.get("anchor_id"), "text": txt}
                break

        score = score_pricing_table(t)

        return {
            "item_id": self.item_id,
            "table": {
                "table_anchor_id": t.get("table_anchor_id"),
                "order": t.get("order"),
                "structured": t.get("structured"),
                "columns": t.get("columns"),
                "rows": t.get("rows"),
                "raw": t.get("raw"),
                # Keep naming aligned with contract_pricing.build_pricing_context outputs so the same
                # validators/compilers can be reused.
                "table_score": score.score,
                "table_score_signals": score.signals,
                "regime_hint": regime_hint,
            },
            "neighbor_anchors": neighbor_anchors,
            "tail_anchors": tail_anchors,
            "adjustment_hints": adjustment_candidates,
        }

    def search(
        self,
        query: str,
        *,
        scope: Literal["anchors", "tables", "all"] = "all",
        limit: int = 20,
        case_sensitive: bool = False,
    ) -> dict[str, Any]:
        query = str(query or "").strip()
        if not query:
            raise ValueError("query is required")
        limit = max(1, int(limit))

        flags = 0 if case_sensitive else re.IGNORECASE
        pattern = re.compile(re.escape(query), flags=flags)

        results: list[dict[str, Any]] = []

        if scope in ("anchors", "all"):
            for a in self._anchors():
                if not isinstance(a, dict):
                    continue
                text = str(a.get("text") or "")
                m = pattern.search(text)
                if not m:
                    continue
                results.append(
                    {
                        "kind": "anchor",
                        "anchor_id": a.get("anchor_id"),
                        "order": a.get("order"),
                        "anchor_type": a.get("anchor_type"),
                        "snippet": _snippet(text, idx=m.start(), query_len=len(query)),
                    }
                )
                if len(results) >= limit:
                    break

        if scope in ("tables", "all") and len(results) < limit:
            for t in self._tables():
                if not isinstance(t, dict):
                    continue
                text = _table_text(t)
                m = pattern.search(text)
                if not m:
                    continue
                results.append(
                    {
                        "kind": "table",
                        "table_anchor_id": t.get("table_anchor_id"),
                        "order": t.get("order"),
                        "snippet": _snippet(text, idx=m.start(), query_len=len(query)),
                    }
                )
                if len(results) >= limit:
                    break

        return {
            "item_id": self.item_id,
            "query": query,
            "scope": scope,
            "limit": limit,
            "results": results,
        }
