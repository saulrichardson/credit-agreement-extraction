from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Literal


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
class DocBrowserFull:
    """Deterministic document browser over doc_ir with NO hand-coded pricing heuristics.

    This is intentionally "dumb but complete":
    - It exposes the entire parsed document (anchors + tables) via auditable tools.
    - It does not score or classify tables.
    - It does not attempt to detect pricing adjustments or regimes.
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
        """Planner-friendly catalog of the whole document (NO scoring)."""

        anchors = self._anchors()
        tables = self._tables()
        summaries: list[dict[str, Any]] = []
        for t in tables:
            if not isinstance(t, dict):
                continue
            table_anchor_id = str(t.get("table_anchor_id") or "")
            if not table_anchor_id:
                continue
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
                }
            )

        summaries_sorted = sorted(summaries, key=lambda x: int(x.get("order") or 0))
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

    def open_table(self, table_anchor_id: str, *, neighbor_window: int = 6) -> dict[str, Any]:
        """Open a table with deterministic surrounding anchor slices.

        - neighbor_anchors: +/- neighbor_window anchors around the table anchor order.
        - tail_anchors: anchors after the table until the next table (exclusive).

        This function does NOT try to infer which anchors are pricing adjustments/regime intros.
        """

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

        table_orders = sorted(
            int(x.get("order") or 0)
            for x in self._tables()
            if isinstance(x, dict) and x.get("order") is not None
        )
        next_table_order = None
        for o in table_orders:
            if o > order:
                next_table_order = o
                break
        tail_end = (next_table_order - 1) if next_table_order is not None else min(max_order, order + k)
        tail_anchors = _slice(order + 1, tail_end)

        return {
            "item_id": self.item_id,
            "table": {
                "table_anchor_id": t.get("table_anchor_id"),
                "order": t.get("order"),
                "structured": t.get("structured"),
                "columns": t.get("columns"),
                "rows": t.get("rows"),
                "raw": t.get("raw"),
                # Compatibility keys used by validators; intentionally empty.
                "table_score": None,
                "table_score_signals": [],
                "regime_hint": None,
            },
            "neighbor_anchors": neighbor_anchors,
            "tail_anchors": tail_anchors,
            # Intentionally empty: no heuristics.
            "adjustment_hints": [],
        }

    def open_anchor_range(self, *, start_order: int, limit: int = 50) -> dict[str, Any]:
        """Return a deterministic "page" of anchors starting at start_order.

        This supports sequential, human-like reading without any retrieval heuristics.
        """

        anchors_by_order = self._anchors_by_order()
        if not anchors_by_order:
            return {"item_id": self.item_id, "start_order": int(start_order), "limit": int(limit), "anchors": []}

        start = max(0, int(start_order))
        limit = max(1, int(limit))
        max_order = max(anchors_by_order.keys())

        end = min(max_order, start + limit - 1)
        anchors: list[dict[str, Any]] = []
        for o in range(start, end + 1):
            a = anchors_by_order.get(o)
            if not a:
                continue
            anchors.append(
                {
                    "anchor_id": a.get("anchor_id"),
                    "anchor_type": a.get("anchor_type"),
                    "order": a.get("order"),
                    "text": a.get("text"),
                }
            )

        next_start = (end + 1) if end < max_order else None
        return {
            "item_id": self.item_id,
            "start_order": start,
            "limit": limit,
            "end_order": end,
            "next_start_order": next_start,
            "anchors": anchors,
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
