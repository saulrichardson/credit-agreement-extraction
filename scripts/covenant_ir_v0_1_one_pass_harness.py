#!/usr/bin/env python
"""
One-pass CovenantIR v0.1 extraction harness (financial covenants only).

This is intentionally a small, artifact-first testbed:
  - Build an excerpt pack from retrieval_v2 snippets filtered to financial_covenant
  - Run ONE LLM call (with optional repair attempts) using a CovenantIR prompt
  - Validate with validate_covenant_ir() + additional hard gates
  - Write raw/parsed/validated artifacts to an output directory

Usage:
  poetry run python scripts/covenant_ir_v0_1_one_pass_harness.py \
    --run-id dan-v2-20260106 \
    --item-id 0000950134-96-000714_4 \
    --out-dir scratch/covenantir_v0_1_one_pass/0000950134-96-000714_4 \
    --prompt prompts/covenant_ir_financial_v0_1.txt \
    --attempts 3 \
    --model openai:gpt-5-nano
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from pipeline.core.anchors import load_anchor_catalog
from pipeline.core.config import Paths, REQUIRED_MODEL, REQUIRED_REASONING
from pipeline.ir.covenant_ir_v0_1 import CovenantIRValidationError, validate_covenant_ir, validate_precision_first_policy
from pipeline.evidence.indexing import DEFAULT_GATEWAY_URL, _ensure_gateway_client_sync
from pipeline.utils import prompt_view_path


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


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


def _order_anchor_ids(*, catalog: Mapping[str, Mapping[str, Any]], anchor_ids: Sequence[str]) -> List[str]:
    def _order(aid: str) -> int:
        info = catalog.get(aid)
        if not info:
            # Keep unknown anchors at the end; the run is expected to be internally consistent.
            return 1_000_000_000
        return int(info["order"])

    return sorted({a.strip() for a in anchor_ids if isinstance(a, str) and a.strip()}, key=_order)


def _expand_anchor_ids(
    *,
    catalog: Mapping[str, Mapping[str, Any]],
    seed_anchor_ids: Sequence[str],
    fill_gaps_up_to: int = 8,
    neighbor_pad: int = 0,
) -> List[str]:
    """Expand a seed anchor set into a more 'complete' excerpt pack.

    Motivation (retrieval robustness):
      - indexing can select headings but miss the immediately-following anchors that contain the numeric thresholds
        (e.g., a heading anchor for a covenant followed by an anchor containing the actual covenant requirement).
      - retrieval_v2 windows can also make it hard to reason about 'missing middle' anchors.

    Strategy:
      - sort seed anchors by anchor order (from anchors.tsv)
      - fill gaps between consecutive seed anchors when the gap is small (<= fill_gaps_up_to)
      - optionally pad each seed anchor by +/- neighbor_pad anchors

    This stays in the 'full-context indexing' lane: the LLM still picks the seeds from the full document;
    we just deterministically ensure local completeness.
    """

    # Build order <-> anchor_id mappings from the catalog.
    order_by_anchor: Dict[str, int] = {}
    anchor_by_order: Dict[int, str] = {}
    for aid, info in (catalog or {}).items():
        if not isinstance(aid, str) or not isinstance(info, dict):
            continue
        order = info.get("order")
        if not isinstance(order, int):
            continue
        order_by_anchor[aid] = order
        anchor_by_order[order] = aid

    seed_orders = sorted({order_by_anchor.get(a) for a in seed_anchor_ids if isinstance(order_by_anchor.get(a), int)})
    if not seed_orders:
        return _order_anchor_ids(catalog=catalog, anchor_ids=seed_anchor_ids)

    expanded_orders: set[int] = set(seed_orders)

    # Fill small gaps between seeds (captures "missing threshold anchors" patterns).
    if fill_gaps_up_to > 0:
        for a, b in zip(seed_orders, seed_orders[1:]):
            gap = int(b) - int(a)
            if gap <= 0:
                continue
            if gap <= int(fill_gaps_up_to):
                for o in range(int(a), int(b) + 1):
                    expanded_orders.add(o)

    # Optional neighbor padding.
    if neighbor_pad > 0:
        for o in seed_orders:
            for d in range(1, int(neighbor_pad) + 1):
                expanded_orders.add(int(o) - d)
                expanded_orders.add(int(o) + d)

    expanded = [anchor_by_order[o] for o in sorted(expanded_orders) if o in anchor_by_order]
    return _order_anchor_ids(catalog=catalog, anchor_ids=expanded)


def _build_excerpt_pack(
    snippets_path: Path, *, anchor_ids: Sequence[str], catalog: Mapping[str, Mapping[str, Any]], canonical_text: str
) -> str:
    by_anchor: Dict[str, Dict[str, Any]] = {}
    for rec in _iter_jsonl(snippets_path):
        aid = rec.get("anchor_id")
        if isinstance(aid, str) and aid not in by_anchor:
            by_anchor[aid] = rec

    blocks: List[str] = []
    for aid in _order_anchor_ids(catalog=catalog, anchor_ids=anchor_ids):
        info = catalog.get(aid)
        # Prefer full-anchor text from canonical.txt (anchors.tsv spans).
        # retrieval_v2 snippets are windows and can truncate/overlap; for extraction we want
        # the exact anchor text so the excerpt pack is "full and complete" per-anchor.
        if info and isinstance(info.get("start"), int) and isinstance(info.get("end"), int):
            a = max(0, int(info["start"]))
            b = min(len(canonical_text), int(info["end"]))
            anchor_text = canonical_text[a:b].rstrip()
            if anchor_text:
                blocks.append(f"[[{aid}]]\n{anchor_text}")
                continue

        # Fallback to retrieval_v2 window snippet if canonical text isn't available.
        rec = by_anchor.get(aid)
        if rec:
            snippet = str(rec.get("snippet") or "").rstrip()
            blocks.append(f"[[{aid}]]\n{snippet}")
        else:
            blocks.append(f"[[{aid}]]\n<MISSING SNIPPET>")
    return "\n\n".join(blocks).strip() + "\n"


def _render_prompt(template: str, *, task: str, contexts: str) -> str:
    out = template
    out = out.replace("{{TASK}}", task)
    out = out.replace("{{CONTEXTS}}", contexts)
    return out


def _render_repair_prompt(*, raw_json: str, errors: List[CovenantIRValidationError]) -> str:
    err_list = [asdict(e) for e in errors]
    codes = sorted({e.code for e in errors if isinstance(e.code, str)})
    extra: List[str] = []
    if "lookup_rule_predicate_not_bool" in codes:
        extra.append(
            "- If you use lookup_rule(table_id, predicate_col, value_col): the predicate column MUST be declared as type \"bool\" in tables[].columns.\n"
            "  - Example: tables[].columns includes {\"name\":\"predicate\",\"type\":\"bool\"}.\n"
            "  - Do NOT declare predicate columns as \"string\" when predicate cells are boolean expressions."
        )
    extra_block = ""
    if extra:
        extra_block = "Additional repair hints for your specific errors:\n" + "\n".join(extra) + "\n\n"
    parts = [
        "=====================\n",
        "REPAIR INSTRUCTIONS\n",
        "=====================\n",
        "Your previous response did not validate against the CovenantIR v0.1 schema and/or hard gates.\n",
        "Return corrected JSON ONLY. Do not include commentary or code fences.\n",
        "You MAY regenerate the entire JSON from scratch using the TASK + CONTEXT above.\n\n",
        "Common structural pitfalls to fix:\n",
        "- bool_expr_spec must be either:\n",
        "  - {\"fn_id\": \"...\"} OR\n",
        "  - {\"args\": [...], \"returns\": \"bool\", \"expr\": <AST>, \"source_refs\": [\"A0001\", ...]}\n",
        "- tables[*].table_id must be a plain string (e.g., \"schedule_1\"), NOT an AST node.\n",
        "- tables[*].rows[*] must have shape:\n",
        "  {\"row_id\": \"...\"|null, \"cells\": {\"col\": <AST>, ...}, \"source_refs\": [\"A0001\", ...]}\n",
        "  Do NOT put column keys directly on the row object; they must be inside \"cells\".\n",
        "- Do NOT put \"source_refs\" inside AST nodes. AST nodes must be ONLY one of:\n",
        "  - {\"lit\": {...}}  OR  {\"var\": \"...\"}  OR  {\"op\": \"...\", \"args\": [...]}\n",
        "- open_items[*].suggested_parameters must be an array of strings (NOT an object).\n",
        "- if(...) MUST have exactly 3 args. For boolean gating (\"if X then requirement Y\"), prefer: or(not(X), Y).\n",
        "- Do NOT compare against placeholder threshold vars like required_* / minimum_* / maximum_* / limit_*.\n",
        "  - If the excerpt defines a required/minimum term, encode that definition as an expression or a schedule lookup.\n",
        "- Do NOT omit an in-scope financial covenant heading found in CONTEXT. Either encode it as a covenant, or emit blocking open_items and return no covenants/tables/derived.\n",
        "- Precision-first mode: if open_items is non-empty, then covenants=[], tables=[], derived=[].\n\n",
    ]
    if extra_block:
        parts.append(extra_block)
    parts.extend(
        [
            "VALIDATION_ERRORS_JSON:\n",
            f"{json.dumps(err_list, indent=2)}\n\n",
            "PREVIOUS_JSON:\n",
            f"{raw_json}\n",
        ]
    )
    return "".join(parts)


ANCHOR_ID_RE = re.compile(r"^A\d{4,}$")

# Heuristic coverage gates (harness-only):
# The excerpt pack can contain multiple financial covenants; models sometimes omit one without blocking.
# We apply a lightweight heading-coverage check to force completeness in precision-first mode.
# Heading detectors for covenant sections.
#
# Important: we anchor these at the start of a line (MULTILINE) to avoid false positives
# from mid-sentence parenthetical list items like "... of (a) Tangible Net Worth, plus (b) Consolidated Debt."
_RE_HEADING_NUM = re.compile(r"^\s*(\d+(?:\.\d+)+)\s+([A-Z][A-Z0-9 /()&-]{2,})\.", re.MULTILINE)
_RE_HEADING_ALPHA = re.compile(r"^\s*([a-z])\.\s+([A-Z][A-Za-z0-9 /()&-]{2,})\.", re.MULTILINE)
_RE_HEADING_NUM_ANYCASE = re.compile(
    r"^\s*(\d+(?:\.\d+)+(?:\([a-z]\))?)\.?\s+([A-Za-z][A-Za-z0-9 /()&'’.,-]{2,}?)\.",
    re.MULTILINE,
)
_RE_HEADING_PAREN_ALPHA = re.compile(
    r"^\s*\(\s*([a-z])\s*\)\s+([A-Za-z][A-Za-z0-9 /()&'’.,-]{2,}?)\.",
    re.MULTILINE | re.IGNORECASE,
)
_RE_HEADING_PAREN_ALPHA_CLOSE = re.compile(
    r"^\s*([a-z])\)\s+([A-Za-z][A-Za-z0-9 /()&'’.,-]{2,}?)\.",
    re.MULTILINE | re.IGNORECASE,
)

# Headings that are *usually* out of scope (negative/operational covenants).
_HEADING_EXCLUDE_KEYWORDS = (
    "dividend",
    "dividends",
    "distribution",
    "distributions",
    "restricted payment",
    "restricted payments",
    "investment",
    "investments",
    "lien",
    "liens",
    "asset sale",
    "asset sales",
    "fundamental change",
    "fundamental changes",
    "affiliate",
    "affiliates",
    "guarant",
    "erisa",
    "report",
    "reports",
    "information",
    "insurance",
    "tax",
    "taxes",
    "loan",
    "loans",
    "contingent obligation",
    "contingent obligations",
    "merger",
    "mergers",
    "consolidation",
    "sale and leaseback",
    "accounts receivable",
    "acquisition",
    "acquisitions",
    "indebtedness",
    "subsidiary indebtedness",
    "subsidiary",
    "chief executive",
    "change in location",
    "location of chief executive",
    "executive offices",
    "offices and assets",
)

# If a heading includes one of these, treat it as in-scope even if it also includes excluded words
# (e.g., "Indebtedness to Tangible Net Worth" contains "indebtedness" but is a financial ratio test).
_HEADING_INCLUDE_OVERRIDE_KEYWORDS = (
    "ratio",
    "coverage",
    "net income",
    "net worth",
    "tangible",
    "ebitda",
    "capital expenditures",
    "capex",
    "debt service",
    "leverage",
    "liquidity",
    "current ratio",
    "turnover",
    "fixed charge",
)

_OPEN_ITEM_EXCLUDE_KEYWORDS = (
    "negative covenant",
    "negative covenants",
    "indebtedness",
    "liens",
    "restricted payments",
    "restricted payment",
    "investments",
    "investment",
    "asset sales",
    "asset sale",
    "fundamental changes",
    "fundamental change",
    "affiliate transactions",
    "affiliate transaction",
    "reporting",
    "information delivery",
    "baskets",
    "exceptions",
)


def _has_include_override_keyword(text: str) -> bool:
    """Match include-override keywords on word boundaries (avoid substring false positives).

    Example false positive we explicitly avoid:
      "acce**ratio**n"  -> should NOT count as containing the keyword "ratio".
    """

    # Normalize aggressively so keyword checks are robust to common EDGAR artifacts:
    # - non-breaking spaces between words (e.g., "net worth")
    # - underscores in ids (e.g., "net_worth")
    # - multiple whitespace/newlines
    raw = (text or "").replace("\u00a0", " ").replace("_", " ")
    t = " ".join(raw.strip().split()).lower()
    if not t:
        return False
    for kw in _HEADING_INCLUDE_OVERRIDE_KEYWORDS:
        kw_norm = " ".join((kw or "").replace("\u00a0", " ").strip().split()).lower()
        if not kw_norm:
            continue
        # Allow flexible whitespace between keyword tokens (e.g., "net   worth").
        parts = [p for p in kw_norm.split(" ") if p]
        if not parts:
            continue
        pat = r"\b" + r"\s+".join(re.escape(p) for p in parts) + r"\b"
        if re.search(pat, t):
            return True
    return False


def _canonicalize_structural(
    doc: Any, *, item_id: str, allowed_anchor_ids: Sequence[str]
) -> Tuple[Any, List[Dict[str, Any]]]:
    """Apply narrow, deterministic structural fixes before validation.

    This is intentionally conservative: it fixes mechanical schema violations that are
    unambiguous (e.g., missing required keys like table rows' row_id=null), without
    inventing covenant logic or rewriting AST semantics.
    """

    if not isinstance(doc, dict):
        return (doc, [])

    allowed = [a.strip() for a in allowed_anchor_ids if isinstance(a, str) and ANCHOR_ID_RE.fullmatch(a.strip())]
    allowed_set = set(allowed)

    out = doc
    changes: List[Dict[str, Any]] = []

    def _set_default(key: str, value: Any) -> None:
        if key not in out:
            out[key] = value
            changes.append({"op": "set_default", "path": f"/{key}"})

    # Ensure top-level required keys exist.
    _set_default("schema_version", "covenant_ir_v0_1")
    _set_default("contract_id", item_id)
    _set_default("sources", [])
    _set_default("indices", [])
    _set_default("tables", [])
    _set_default("derived", [])
    _set_default("open_items", [])
    _set_default("covenants", [])

    # Normalize contract_id and sources[*].item_id to the known item_id for this run.
    if out.get("contract_id") != item_id:
        changes.append({"op": "set", "path": "/contract_id", "to": item_id})
        out["contract_id"] = item_id

    sources = out.get("sources")
    if not isinstance(sources, list) or not sources:
        out["sources"] = [
            {
                "source_id": "S1",
                "kind": "excerpt_pack",
                "item_id": item_id,
                "anchor_ids": allowed,
                "notes": None,
            }
        ]
        changes.append({"op": "set", "path": "/sources", "reason": "missing_or_empty"})
    else:
        for si, s in enumerate(sources):
            if not isinstance(s, dict):
                continue
            if s.get("item_id") != item_id:
                s["item_id"] = item_id
                changes.append({"op": "set", "path": f"/sources/{si}/item_id", "to": item_id})
            if not isinstance(s.get("anchor_ids"), list) or not s.get("anchor_ids"):
                s["anchor_ids"] = allowed
                changes.append({"op": "set", "path": f"/sources/{si}/anchor_ids", "reason": "missing_or_empty"})

    def _filter_anchor_list(v: Any) -> Any:
        if not isinstance(v, list):
            return v
        before = [a for a in v if isinstance(a, str)]
        return [a for a in before if a in allowed_set]

    def _walk(obj: Any, path: List[str]) -> None:
        if isinstance(obj, dict):
            for k, v in list(obj.items()):
                if k in ("source_refs", "anchor_ids"):
                    new_v = _filter_anchor_list(v)
                    if isinstance(v, list) and new_v != v:
                        removed = [a for a in v if isinstance(a, str) and a not in allowed_set]
                        if removed:
                            changes.append({"op": "filter_anchors", "path": "/" + "/".join(path + [k]), "removed": removed})
                    obj[k] = new_v
                elif isinstance(v, (dict, list)):
                    _walk(v, path + [k])
        elif isinstance(obj, list):
            for i, v in enumerate(obj):
                if isinstance(v, (dict, list)):
                    _walk(v, path + [str(i)])

    _walk(out, [])

    # Arg specs do not allow "integer" types (schema + evaluator only accept decimal/rate/bps/money for numeric args).
    def _coerce_integer_arg_types(obj: Any, path: List[str]) -> None:
        if isinstance(obj, dict):
            args = obj.get("args")
            if isinstance(args, list):
                for ai, arg in enumerate(args):
                    if not isinstance(arg, dict):
                        continue
                    if arg.get("type") == "integer":
                        arg["type"] = "decimal"
                        changes.append(
                            {
                                "op": "coerce",
                                "path": "/" + "/".join(path + ["args", str(ai), "type"]),
                                "from": "integer",
                                "to": "decimal",
                            }
                        )
            for k, v in obj.items():
                if isinstance(v, (dict, list)):
                    _coerce_integer_arg_types(v, path + [k])
        elif isinstance(obj, list):
            for i, v in enumerate(obj):
                if isinstance(v, (dict, list)):
                    _coerce_integer_arg_types(v, path + [str(i)])

    _coerce_integer_arg_types(out, [])

    # Some models confuse lookup()/lookup2() arg ordering (swapping key_column with key_value).
    # This is a deterministic, schema-motivated fix:
    #   - key*_column args must be AST string literals
    #   - key*_value args are usually vars/lits (not column identifiers)
    def _is_lit_string_node(n: Any) -> bool:
        return (
            isinstance(n, dict)
            and isinstance(n.get("lit"), dict)
            and n["lit"].get("type") == "string"
            and isinstance(n["lit"].get("value"), str)
            and bool(n["lit"].get("value").strip())
        )

    def _fix_lookup_arg_order(obj: Any, path: List[str]) -> None:
        if isinstance(obj, dict):
            op = obj.get("op")
            args = obj.get("args")
            if isinstance(op, str) and isinstance(args, list):
                if op == "lookup" and len(args) == 4:
                    # Expected: [table_id(str), key_column(str), key_value, value_column(str)]
                    if not _is_lit_string_node(args[1]) and _is_lit_string_node(args[2]):
                        args[1], args[2] = args[2], args[1]
                        changes.append({"op": "swap", "path": "/" + "/".join(path + ["args", "1..2"]), "reason": "lookup_key_column_order"})
                if op == "lookup2" and len(args) == 6:
                    # Expected: [table_id(str), key1_col(str), key1_val, key2_col(str), key2_val, value_col(str)]
                    if not _is_lit_string_node(args[1]) and _is_lit_string_node(args[2]):
                        args[1], args[2] = args[2], args[1]
                        changes.append(
                            {"op": "swap", "path": "/" + "/".join(path + ["args", "1..2"]), "reason": "lookup2_key1_col_order"}
                        )
                    if not _is_lit_string_node(args[3]) and _is_lit_string_node(args[4]):
                        args[3], args[4] = args[4], args[3]
                        changes.append(
                            {"op": "swap", "path": "/" + "/".join(path + ["args", "3..4"]), "reason": "lookup2_key2_col_order"}
                        )

            for k, v in obj.items():
                if isinstance(v, (dict, list)):
                    _fix_lookup_arg_order(v, path + [k])
        elif isinstance(obj, list):
            for i, v in enumerate(obj):
                if isinstance(v, (dict, list)):
                    _fix_lookup_arg_order(v, path + [str(i)])

    _fix_lookup_arg_order(out, [])

    # open_items[*].suggested_parameters must be an array of strings.
    open_items = out.get("open_items")
    if isinstance(open_items, list):
        for oi_i, oi in enumerate(open_items):
            if not isinstance(oi, dict):
                continue
            sp = oi.get("suggested_parameters")
            if isinstance(sp, dict):
                oi["suggested_parameters"] = [f"{k}: {v}" for k, v in sp.items()]
                changes.append({"op": "coerce", "path": f"/open_items/{oi_i}/suggested_parameters", "from": "object"})
            elif isinstance(sp, str):
                oi["suggested_parameters"] = [sp]
                changes.append({"op": "coerce", "path": f"/open_items/{oi_i}/suggested_parameters", "from": "string"})

    # Table rows must include row_id (schema allows null but requires presence).
    tables = out.get("tables")
    if isinstance(tables, list):
        # Collect table ids referenced by AST lookup_* operators so we can repair missing tables[*].table_id.
        table_ops = {"lookup", "lookup2", "lookup_range", "lookup_rule"}
        referenced_table_ids: List[str] = []

        def _walk_ast(obj: Any) -> None:
            if isinstance(obj, dict):
                op = obj.get("op")
                args = obj.get("args")
                if isinstance(op, str) and op in table_ops and isinstance(args, list) and args:
                    first = args[0]
                    if (
                        isinstance(first, dict)
                        and isinstance(first.get("lit"), dict)
                        and first["lit"].get("type") == "string"
                        and isinstance(first["lit"].get("value"), str)
                    ):
                        tid = first["lit"]["value"]
                        if tid and tid not in referenced_table_ids:
                            referenced_table_ids.append(tid)
                for v in obj.values():
                    _walk_ast(v)
            elif isinstance(obj, list):
                for v in obj:
                    _walk_ast(v)

        _walk_ast(out)

        used_table_ids = {
            t.get("table_id")
            for t in tables
            if isinstance(t, dict) and isinstance(t.get("table_id"), str) and t.get("table_id")
        }
        available_table_ids = [tid for tid in referenced_table_ids if tid not in used_table_ids]

        missing_table_id_slots = [ti for ti, t in enumerate(tables) if isinstance(t, dict) and not t.get("table_id")]

        inferred_ids_by_table_index: Dict[int, str] = {}
        if missing_table_id_slots and len(missing_table_id_slots) == len(available_table_ids):
            for ti, inferred in zip(missing_table_id_slots, available_table_ids, strict=False):
                if isinstance(inferred, str) and inferred:
                    inferred_ids_by_table_index[ti] = inferred

        for ti, t in enumerate(tables):
            if not isinstance(t, dict):
                continue
            table_level_source_refs: List[str] | None = None
            if isinstance(t.get("source_refs"), list):
                table_level_source_refs = [a for a in t["source_refs"] if isinstance(a, str)]

            # tables[*].table_id must be a plain string. Models sometimes emit an AST string literal.
            tid = t.get("table_id")
            if isinstance(tid, dict):
                lit = tid.get("lit")
                if (
                    isinstance(lit, dict)
                    and lit.get("type") == "string"
                    and isinstance(lit.get("value"), str)
                    and lit.get("value").strip()
                ):
                    t["table_id"] = lit["value"]
                    changes.append({"op": "coerce", "path": f"/tables/{ti}/table_id", "from": "lit_string"})
            if not isinstance(t.get("table_id"), str) or not str(t.get("table_id") or "").strip():
                inferred = inferred_ids_by_table_index.get(ti)
                if isinstance(inferred, str) and inferred:
                    t["table_id"] = inferred
                    changes.append({"op": "set", "path": f"/tables/{ti}/table_id", "reason": "inferred_from_lookup"})
                else:
                    t["table_id"] = f"table_{ti + 1}"
                    changes.append({"op": "set", "path": f"/tables/{ti}/table_id", "reason": "generated"})

            col_names: List[str] = []
            cols = t.get("columns")
            if isinstance(cols, list):
                for c in cols:
                    name = c.get("name") if isinstance(c, dict) else None
                    if isinstance(name, str) and name:
                        col_names.append(name)
            col_set = set(col_names)

            rows = t.get("rows")
            if not isinstance(rows, list):
                continue
            for ri, row in enumerate(rows):
                if isinstance(row, dict) and "row_id" not in row:
                    row["row_id"] = None
                    changes.append({"op": "set_default", "path": f"/tables/{ti}/rows/{ri}/row_id"})

                # table_row objects must wrap column values in "cells": {col_name -> AST node}.
                # Some model outputs place the column keys directly on the row object.
                if isinstance(row, dict) and "cells" not in row and col_set:
                    extracted: Dict[str, Any] = {}
                    for k in list(row.keys()):
                        if k in col_set:
                            v = row.pop(k)
                            # Allow open-ended schedule bounds by omitting missing cells entirely.
                            if v is None:
                                changes.append(
                                    {"op": "drop_null_cell", "path": f"/tables/{ti}/rows/{ri}/{k}", "reason": "non_ast"}
                                )
                                continue
                            extracted[k] = v
                    if extracted:
                        row["cells"] = extracted
                        changes.append(
                            {
                                "op": "wrap_cells",
                                "path": f"/tables/{ti}/rows/{ri}/cells",
                                "columns": sorted(extracted.keys()),
                            }
                        )

                # table_row disallows additional properties; drop anything besides row_id/cells/source_refs.
                if isinstance(row, dict):
                    sr = row.get("source_refs")
                    if (
                        (not isinstance(sr, list) or not sr)
                        and isinstance(table_level_source_refs, list)
                        and table_level_source_refs
                    ):
                        row["source_refs"] = [a for a in table_level_source_refs if a in allowed_set]
                        changes.append(
                            {"op": "set", "path": f"/tables/{ti}/rows/{ri}/source_refs", "reason": "copied_from_table"}
                        )
                    sr = row.get("source_refs")
                    if (not isinstance(sr, list) or not sr) and allowed:
                        # Last-resort provenance: point at the whole excerpt pack rather than failing schema.
                        row["source_refs"] = allowed
                        changes.append(
                            {
                                "op": "set",
                                "path": f"/tables/{ti}/rows/{ri}/source_refs",
                                "reason": "default_excerpt_anchors",
                            }
                        )

                    cells = row.get("cells")
                    if isinstance(cells, dict):
                        null_keys = [k for k, v in cells.items() if v is None]
                        if null_keys:
                            for k in null_keys:
                                cells.pop(k, None)
                            changes.append(
                                {"op": "drop_null_cells", "path": f"/tables/{ti}/rows/{ri}/cells", "keys": sorted(null_keys)}
                            )

                    allowed_row_keys = {"row_id", "cells", "source_refs"}
                    extra_keys = [k for k in row.keys() if k not in allowed_row_keys]
                    if extra_keys:
                        for k in extra_keys:
                            row.pop(k, None)
                        changes.append(
                            {"op": "drop_keys", "path": f"/tables/{ti}/rows/{ri}", "keys": sorted(extra_keys)}
                        )

            # table objects disallow additional properties; drop unknown keys (e.g., caption/source_refs).
            allowed_table_keys = {"table_id", "description", "columns", "rows"}
            extra_keys = [k for k in t.keys() if k not in allowed_table_keys]
            if extra_keys:
                for k in extra_keys:
                    t.pop(k, None)
                changes.append({"op": "drop_keys", "path": f"/tables/{ti}", "keys": sorted(extra_keys)})

    # covenants[*].source_refs is required; if missing, copy from test/applies_when source_refs when present.
    covenants = out.get("covenants")
    if isinstance(covenants, list):
        for ci, cov in enumerate(covenants):
            if not isinstance(cov, dict):
                continue
            sr = cov.get("source_refs")
            if isinstance(sr, list) and sr:
                continue
            copied: List[str] | None = None
            test = cov.get("test")
            if isinstance(test, dict) and isinstance(test.get("source_refs"), list) and test.get("source_refs"):
                copied = [a for a in test["source_refs"] if isinstance(a, str)]
            applies = cov.get("applies_when")
            if copied is None and isinstance(applies, dict) and isinstance(applies.get("source_refs"), list) and applies.get("source_refs"):
                copied = [a for a in applies["source_refs"] if isinstance(a, str)]
            if copied:
                cov["source_refs"] = [a for a in copied if a in allowed_set]
                changes.append({"op": "set", "path": f"/covenants/{ci}/source_refs", "reason": "copied_from_expr"})

    # LLMs often omit inline expr source_refs or mis-nest applies_when under test; fix these mechanically.
    covenants = out.get("covenants")
    if isinstance(covenants, list):
        for ci, cov in enumerate(covenants):
            if not isinstance(cov, dict):
                continue

            cov_sr = cov.get("source_refs")
            cov_sr_list = [a for a in cov_sr if isinstance(a, str) and a in allowed_set] if isinstance(cov_sr, list) else []

            test = cov.get("test")
            if isinstance(test, dict):
                # applies_when belongs on the covenant, not inside test.
                if "applies_when" in test:
                    nested = test.pop("applies_when")
                    if "applies_when" not in cov and isinstance(nested, dict):
                        cov["applies_when"] = nested
                        changes.append({"op": "move", "path": f"/covenants/{ci}/applies_when", "from": "test.applies_when"})
                    else:
                        changes.append({"op": "drop_keys", "path": f"/covenants/{ci}/test", "keys": ["applies_when"]})

                # Inline test expressions must carry non-empty source_refs; if omitted, reuse covenant source_refs.
                if "fn_id" not in test:
                    test_sr = test.get("source_refs")
                    if (not isinstance(test_sr, list) or not test_sr) and cov_sr_list:
                        test["source_refs"] = cov_sr_list
                        changes.append({"op": "set", "path": f"/covenants/{ci}/test/source_refs", "reason": "copied_from_covenant"})

            applies = cov.get("applies_when")
            if isinstance(applies, dict) and "fn_id" not in applies:
                applies_sr = applies.get("source_refs")
                if (not isinstance(applies_sr, list) or not applies_sr) and cov_sr_list:
                    applies["source_refs"] = cov_sr_list
                    changes.append(
                        {"op": "set", "path": f"/covenants/{ci}/applies_when/source_refs", "reason": "copied_from_covenant"}
                    )

    # Force sources[*].anchor_ids to reflect the excerpt pack for self-contained provenance.
    sources = out.get("sources")
    if isinstance(sources, list):
        for si, s in enumerate(sources):
            if isinstance(s, dict):
                s["anchor_ids"] = allowed
                changes.append({"op": "set", "path": f"/sources/{si}/anchor_ids", "reason": "force_excerpt_anchors"})

    return (out, changes)


def _iter_ast_nodes(node: Any) -> Iterable[Dict[str, Any]]:
    if isinstance(node, dict):
        yield node
        args = node.get("args")
        if isinstance(args, list):
            for a in args:
                yield from _iter_ast_nodes(a)
    elif isinstance(node, list):
        for a in node:
            yield from _iter_ast_nodes(a)


def _var_name(node: Any) -> str | None:
    if not isinstance(node, dict):
        return None
    v = node.get("var")
    if isinstance(v, str) and v.strip():
        return v.strip()
    return None


_THRESHOLD_VAR_RE = re.compile(r"^(min|max|minimum|maximum|required|threshold|limit)(?:_|$)", re.IGNORECASE)


def _validate_open_items_in_scope(doc: Any) -> List[CovenantIRValidationError]:
    """Hard-ish gate: open_items must not be about out-of-scope non-financial covenants."""

    if not isinstance(doc, dict):
        return []
    open_items = doc.get("open_items") or []
    if not isinstance(open_items, list) or not open_items:
        return []

    out: List[CovenantIRValidationError] = []
    for i, oi in enumerate(open_items):
        if not isinstance(oi, dict):
            continue
        issue = oi.get("issue")
        if not isinstance(issue, str) or not issue.strip():
            continue
        text = " ".join(issue.strip().split()).lower()
        if any(k in text for k in _OPEN_ITEM_EXCLUDE_KEYWORDS) and not _has_include_override_keyword(text):
            out.append(
                CovenantIRValidationError(
                    code="open_item_out_of_scope",
                    message=(
                        "open_items must not block for out-of-scope negative/operational covenants. "
                        "Per prompt scope, ignore negative-covenant baskets/restrictions even if truncated. "
                        "Only block for missing/ambiguous data required to encode an in-scope financial covenant test. "
                        f"(out_of_scope_issue={issue.strip()!r})"
                    ),
                    json_path=f"/open_items/{i}/issue",
                )
            )
    return out


def _resolve_bool_expr_spec_expr(spec: Any, *, derived_map: Mapping[str, Any]) -> Any | None:
    if not isinstance(spec, dict):
        return None
    if "fn_id" in spec:
        fn_id = spec.get("fn_id")
        if not isinstance(fn_id, str) or not fn_id.strip():
            return None
        df = derived_map.get(fn_id.strip())
        if isinstance(df, dict):
            return df.get("expr")
        return None
    return spec.get("expr")


def _validate_no_threshold_placeholders(doc: Any) -> List[CovenantIRValidationError]:
    """Hard-ish gate: avoid placeholder threshold vars like min_* without grounding.

    In A1 financial covenant extraction, covenant thresholds must be:
      - explicit numeric literals, OR
      - retrieved from an explicit schedule table via lookup_* inside the expression.

    A direct comparison like `fixed_charge_coverage_ratio >= min_fixed_charge_coverage_ratio`
    is not evaluatable unless the required threshold is present in the CovenantIR itself.
    """

    if not isinstance(doc, dict):
        return []
    open_items = doc.get("open_items") or []
    if isinstance(open_items, list) and open_items:
        return []  # precision-first blocked state; no covenants should be present anyway

    covenants = doc.get("covenants") or []
    if not isinstance(covenants, list) or not covenants:
        return []

    derived_map: Dict[str, Any] = {}
    derived = doc.get("derived") or []
    if isinstance(derived, list):
        for df in derived:
            if isinstance(df, dict) and isinstance(df.get("fn_id"), str):
                derived_map[df["fn_id"]] = df

    out: List[CovenantIRValidationError] = []

    for ci, cov in enumerate(covenants):
        if not isinstance(cov, dict):
            continue

        for field in ("test", "applies_when"):
            spec = cov.get(field)
            if spec is None:
                continue
            expr = _resolve_bool_expr_spec_expr(spec, derived_map=derived_map)
            if expr is None:
                continue
            for n in _iter_ast_nodes(expr):
                op = n.get("op")
                args = n.get("args")
                if not isinstance(op, str) or not isinstance(args, list) or len(args) != 2:
                    continue
                if op not in ("lt", "lte", "gt", "gte", "eq"):
                    continue
                left = _var_name(args[0])
                right = _var_name(args[1])
                if not left or not right:
                    continue
                if _THRESHOLD_VAR_RE.search(left) or _THRESHOLD_VAR_RE.search(right):
                    out.append(
                        CovenantIRValidationError(
                            code="threshold_placeholder_var",
                            message=(
                                f"Found threshold-like placeholder var comparison {left!r} {op} {right!r}. "
                                "In A1 mode, required thresholds must be explicit numeric literals or schedule lookups; "
                                "do not leave thresholds as free input variables."
                            ),
                            json_path=f"/covenants/{ci}/{field}",
                        )
                    )
    return out


def _validate_no_out_of_scope_covenants(doc: Any) -> List[CovenantIRValidationError]:
    """Hard-ish gate: extracted covenants must respect financial-covenant scope."""

    if not isinstance(doc, dict):
        return []
    open_items = doc.get("open_items") or []
    if isinstance(open_items, list) and open_items:
        return []  # precision-first blocked state

    covenants = doc.get("covenants") or []
    if not isinstance(covenants, list) or not covenants:
        return []

    out: List[CovenantIRValidationError] = []
    for i, cov in enumerate(covenants):
        if not isinstance(cov, dict):
            continue
        cid = str(cov.get("covenant_id") or "")
        title = str(cov.get("title") or "")
        hay = " ".join([cid, title]).strip().lower()
        if not hay:
            continue

        # Out-of-scope keywords are allowed only when they are part of a clearly financial test
        # (e.g., "indebtedness to net worth ratio" includes "ratio").
        looks_out_of_scope = any(k in hay for k in _OPEN_ITEM_EXCLUDE_KEYWORDS) or any(k in hay for k in _HEADING_EXCLUDE_KEYWORDS)
        if looks_out_of_scope and not _has_include_override_keyword(hay):
            out.append(
                CovenantIRValidationError(
                    code="out_of_scope_covenant",
                    message=(
                        "Extracted a covenant that appears out of scope (negative/operational covenant) for this run. "
                        "Per prompt scope, exclude negative covenant restrictions/baskets; only encode financial covenants. "
                        f"(covenant_id={cid!r}, title={title!r})"
                    ),
                    json_path=f"/covenants/{i}",
                )
            )

    return out


def _validate_contract_and_source_ids(doc: Any, *, item_id: str) -> List[CovenantIRValidationError]:
    out: List[CovenantIRValidationError] = []
    if not isinstance(doc, dict):
        return out

    cid = doc.get("contract_id")
    if cid != item_id:
        out.append(
            CovenantIRValidationError(
                code="contract_id_mismatch",
                message=f"contract_id must equal item_id {item_id!r}; got {cid!r}",
                json_path="/contract_id",
            )
        )

    sources = doc.get("sources")
    if isinstance(sources, list):
        for si, s in enumerate(sources):
            if not isinstance(s, dict):
                continue
            sid = s.get("item_id")
            if sid != item_id:
                out.append(
                    CovenantIRValidationError(
                        code="source_item_id_mismatch",
                        message=f"sources[{si}].item_id must equal item_id {item_id!r}; got {sid!r}",
                        json_path=f"/sources/{si}/item_id",
                    )
                )
    return out


def _validate_anchor_ids_in_context(doc: Any, *, allowed_anchor_ids: Sequence[str]) -> List[CovenantIRValidationError]:
    allowed = set(allowed_anchor_ids)
    out: List[CovenantIRValidationError] = []

    def _walk(obj: Any, path: List[str]) -> None:
        if isinstance(obj, dict):
            for k, v in obj.items():
                if k in ("source_refs", "anchor_ids"):
                    if isinstance(v, list):
                        for idx, aid in enumerate(v):
                            if isinstance(aid, str) and aid in allowed:
                                continue
                            out.append(
                                CovenantIRValidationError(
                                    code="anchor_not_in_context",
                                    message=f"Anchor id {aid!r} is not in excerpt anchors {sorted(allowed)}",
                                    json_path="/" + "/".join(path + [k, str(idx)]),
                                )
                            )
                    else:
                        out.append(
                            CovenantIRValidationError(
                                code="anchor_not_in_context",
                                message=f"{k} must be an array",
                                json_path="/" + "/".join(path + [k]),
                            )
                        )
                else:
                    _walk(v, path + [k])
        elif isinstance(obj, list):
            for i, v in enumerate(obj):
                _walk(v, path + [str(i)])

    _walk(doc, [])
    return out


def _count_open_items(doc: Any) -> tuple[int, int]:
    """Return (open_items_total, open_items_blocking)."""

    if not isinstance(doc, dict):
        return (0, 0)

    open_items = doc.get("open_items") or []
    if not isinstance(open_items, list):
        return (0, 0)

    total = 0
    blocking = 0
    for oi in open_items:
        if not isinstance(oi, dict):
            continue
        total += 1
        if oi.get("blocking") is True:
            blocking += 1
    return (total, blocking)


def _to_blocked_artifact(
    *,
    item_id: str,
    run_id: str,
    attempts_used: int,
    blocked_reason: str,
    cov_anchors: Sequence[str],
    covenantir_validated: Mapping[str, Any],
    out_dir: Path,
    last_errors: Sequence[CovenantIRValidationError],
) -> Dict[str, Any]:
    """Build a deterministic sidecar record for blocked CovenantIR items."""

    open_items_total, open_items_blocking = _count_open_items(covenantir_validated)
    covs = covenantir_validated.get("covenants")
    tables = covenantir_validated.get("tables")
    derived = covenantir_validated.get("derived")
    open_items = covenantir_validated.get("open_items")

    error_rows = [asdict(e) for e in (last_errors or [])]
    error_codes: List[str] = sorted({str(e.get("code") or "") for e in error_rows if isinstance(e, dict)})

    return {
        "schema_version": "covenant_ir_v0_1_blocked_artifact_v1",
        "artifact_type": "covenantir_blocked_artifact",
        "item_id": item_id,
        "run_id": run_id,
        "attempts_used": int(attempts_used),
        "blocked_reason": blocked_reason,
        "open_items_total": int(open_items_total),
        "open_items_blocking": int(open_items_blocking),
        "source_anchor_ids": [a for a in cov_anchors if isinstance(a, str)],
        "covenantir_validated_path": str(out_dir / "covenantir_validated.json"),
        "covenantir_summary": {
            "covenants_count": len(covs) if isinstance(covs, list) else 0,
            "tables_count": len(tables) if isinstance(tables, list) else 0,
            "derived_count": len(derived) if isinstance(derived, list) else 0,
            "open_items_count": len(open_items) if isinstance(open_items, list) else 0,
        },
        "error_codes": [c for c in error_codes if c],
        "error_count": len(error_rows),
        "error_examples": error_rows[:10],
        "open_item_examples": open_items[:5] if isinstance(open_items, list) else [],
    }


def _parse_excerpt_pack_blocks(contexts: str) -> Dict[str, str]:
    """Parse the harness-built excerpt pack into {anchor_id: text} blocks."""

    blocks: Dict[str, List[str]] = {}
    current: str | None = None
    for raw_line in (contexts or "").splitlines():
        line = raw_line.rstrip("\n")
        m = re.match(r"^\[\[(A\d{4,})\]\]$", line.strip())
        if m:
            current = m.group(1)
            blocks.setdefault(current, [])
            continue
        if current is not None:
            blocks[current].append(line)
    return {aid: "\n".join(lines).strip() for aid, lines in blocks.items()}


def _heading_in_scope(heading_text: str) -> bool:
    h = " ".join((heading_text or "").strip().split()).lower()
    if not h:
        return False
    include_override = _has_include_override_keyword(h)
    exclude = any(k in h for k in _HEADING_EXCLUDE_KEYWORDS)
    if exclude and not include_override:
        return False
    # Heuristic: only treat it as a financial-covenant heading if it looks like a specific covenant,
    # not a generic section label (e.g., "COVENANTS", "FINANCIAL COVENANTS").
    if h in {"covenants", "financial covenants", "financial covenant"}:
        return False
    return True


_HEADING_EXCLUDE_VERBS = (
    " shall ",
    " may ",
    " will ",
    " must ",
    " permit ",
    " permits ",
    " provide ",
    " provides ",
    " provided ",
    " including ",
    " in computing ",
    " for the purpose ",
)


def _looks_like_covenant_heading_label(name: str) -> bool:
    """Heuristic: distinguish short heading labels from long definitional list items.

    We want to catch things like:
      "(a) Consolidated Leverage Ratio."
    but avoid misclassifying definition enumerations like:
      "(v) all amounts deducted in computing net income ... ."
    """

    n = " ".join((name or "").strip().split())
    if not n:
        return False
    if len(n) > 80:
        return False
    hay = f" {n.lower()} "
    if any(v in hay for v in _HEADING_EXCLUDE_VERBS):
        return False
    return True


def _validate_heading_coverage(doc: Any, *, contexts: str) -> List[CovenantIRValidationError]:
    """Hard-ish gate (harness-only): don't allow silent omission of covenant headings."""

    if not isinstance(doc, dict):
        return []
    open_items = doc.get("open_items") or []
    if isinstance(open_items, list) and open_items:
        return []  # already blocked by policy

    covenants = doc.get("covenants") or []
    if not isinstance(covenants, list):
        return []

    blocks = _parse_excerpt_pack_blocks(contexts)
    heading_to_anchors: Dict[str, set[str]] = {}
    for aid, text in blocks.items():
        # Numeric headings like "6.14 NET INCOME."
        for m in _RE_HEADING_NUM.finditer(text):
            sec = m.group(1).strip()
            name = m.group(2).strip()
            heading = f"{sec} {name}"
            if _heading_in_scope(heading):
                heading_to_anchors.setdefault(heading, set()).add(aid)
        # More permissive numeric headings like "7.1. financial covenants." / "6.11(b) Total Leverage Ratio."
        for m in _RE_HEADING_NUM_ANYCASE.finditer(text):
            sec = m.group(1).strip()
            name = m.group(2).strip()
            # Avoid false positives on numeric thresholds (e.g., "1.00 or ...") by requiring the heading name
            # to begin with an uppercase letter, which is typical for section headings.
            if not name or not name[:1].isupper():
                continue
            heading = f"{sec} {name}"
            if (
                _heading_in_scope(heading)
                and _looks_like_covenant_heading_label(name)
                and _has_include_override_keyword(heading)
            ):
                heading_to_anchors.setdefault(heading, set()).add(aid)
        # Letter headings like "f. Tangible Net Worth."
        for m in _RE_HEADING_ALPHA.finditer(text):
            sec = m.group(1).strip()
            name = m.group(2).strip()
            heading = f"{sec}. {name}"
            # Lettered headings show up in many non-covenant lists (definitions, notice requirements, etc.).
            # To avoid false positives, require a stronger "looks financial" signal for alpha headings.
            if (
                _heading_in_scope(heading)
                and _looks_like_covenant_heading_label(name)
                and _has_include_override_keyword(heading)
            ):
                heading_to_anchors.setdefault(heading, set()).add(aid)
        # Parenthetical letter headings like "(b) Maximum Debt to Tangible Net Worth Ratio."
        for m in _RE_HEADING_PAREN_ALPHA.finditer(text):
            sec = m.group(1).strip()
            name = m.group(2).strip()
            heading = f"{sec}) {name}"
            if (
                _heading_in_scope(heading)
                and _looks_like_covenant_heading_label(name)
                and _has_include_override_keyword(heading)
            ):
                heading_to_anchors.setdefault(heading, set()).add(aid)
        # Letter headings like "b) Maximum Leverage Ratio."
        for m in _RE_HEADING_PAREN_ALPHA_CLOSE.finditer(text):
            sec = m.group(1).strip()
            name = m.group(2).strip()
            heading = f"{sec}) {name}"
            if (
                _heading_in_scope(heading)
                and _looks_like_covenant_heading_label(name)
                and _has_include_override_keyword(heading)
            ):
                heading_to_anchors.setdefault(heading, set()).add(aid)

    if not heading_to_anchors:
        return []

    # If the model returned no covenants but also no open_items, that's always suspicious in this harness.
    out: List[CovenantIRValidationError] = []
    if not covenants:
        out.append(
            CovenantIRValidationError(
                code="missing_all_covenants",
                message="No covenants were returned, but the excerpt pack contained financial-covenant anchors.",
                json_path="/covenants",
            )
        )
        return out

    def _covenant_mentions_heading(cov: Dict[str, Any], *, sec: str, name: str) -> bool:
        cid = cov.get("covenant_id")
        title = cov.get("title")
        hay = " ".join([str(cid or ""), str(title or "")]).lower()
        name_norm = " ".join((name or "").strip().split()).lower()
        if name_norm and name_norm in hay:
            return True
        # Many agreements label headings with "Maximum/Minimum ..." but extracted titles often omit that adjective
        # while still clearly encoding the same covenant (direction is captured in the comparator itself).
        name_norm_wo_adj = re.sub(r"^(maximum|minim(?:um)?|max|min)\\s+", "", name_norm).strip()
        if name_norm_wo_adj and name_norm_wo_adj in hay:
            return True
        sec_norm = (sec or "").strip()
        if sec_norm and sec_norm[0].isdigit():
            if sec_norm.lower() in hay:
                return True
            if sec_norm.replace(".", "_").lower() in hay:
                return True
            if sec_norm.replace(".", "-").lower() in hay:
                return True
        return False

    missing: List[tuple[str, List[str]]] = []
    for heading, anchors in sorted(heading_to_anchors.items()):
        # heading is either "<sec> <NAME>" (numeric) or "<letter>. <Name>" (alpha)
        parts = heading.split(" ", 1)
        sec = parts[0].rstrip(".")
        name = parts[1] if len(parts) > 1 else ""

        matched = False
        for cov in covenants:
            if not isinstance(cov, dict):
                continue
            if _covenant_mentions_heading(cov, sec=sec, name=name):
                matched = True
                break

        if matched:
            continue

        missing.append((heading, sorted(list(anchors))[:8]))

    if not missing:
        return []

    # Emit one error per missing heading (capped) to keep repair prompts reasonable.
    for heading, anchors in missing[:10]:
        out.append(
            CovenantIRValidationError(
                code="missing_heading_coverage",
                message=(
                    f"Detected covenant heading {heading!r} in CONTEXT, but no covenant_id/title appears to encode it "
                    f"(anchors_with_heading={anchors}). "
                    "Do not omit covenants in precision-first mode: either encode it, or emit blocking open_items and return no covenants."
                ),
                json_path="/covenants",
            )
        )

    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--item-id", required=True)
    ap.add_argument("--base-dir", default=".")
    ap.add_argument("--snippets-path", default=None, help="Override retrieval_v2 snippets path; defaults to runs/<run_id>/retrieval_v2/<item_id>_snippets.jsonl")
    ap.add_argument(
        "--anchor-id",
        dest="anchor_ids",
        action="append",
        default=[],
        help="Optional explicit anchor_id to include in excerpt pack (repeatable). If omitted, uses all anchors categorized as financial_covenant.",
    )
    ap.add_argument("--prompt", default="prompts/covenant_ir_financial_v0_1.txt")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--attempts", type=int, default=3)
    ap.add_argument("--model", default=REQUIRED_MODEL)
    ap.add_argument("--reasoning", default=REQUIRED_REASONING)
    ap.add_argument("--gateway-url", default=None)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--timeout-seconds", type=float, default=600.0)
    args = ap.parse_args()

    item_id: str = args.item_id
    paths = Paths(root=Path(args.base_dir), run_id=args.run_id)
    catalog = load_anchor_catalog(paths, item_id)

    snippets_path = (
        Path(args.snippets_path)
        if args.snippets_path
        else (paths.run_dir / "retrieval_v2" / f"{item_id}_snippets.jsonl")
    )
    if not snippets_path.exists():
        raise SystemExit(f"snippets not found: {snippets_path}")

    if args.anchor_ids:
        cov_anchors = [a for a in args.anchor_ids if isinstance(a, str)]
    else:
        cov_anchors = []
        for rec in _iter_jsonl(snippets_path):
            cats = rec.get("categories") or []
            if not isinstance(cats, list):
                continue
            if "financial_covenant" not in cats:
                continue
            aid = rec.get("anchor_id")
            if isinstance(aid, str):
                cov_anchors.append(aid)

    cov_anchors = _order_anchor_ids(catalog=catalog, anchor_ids=cov_anchors)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if not cov_anchors:
        _write_json(
            out_dir / "result.json",
            {
                "attempts_used": 0,
                "covenants_count": 0,
                "item_id": item_id,
                "ok": True,
                "open_items_blocking": 0,
                "open_items_total": 0,
                "out_dir": str(out_dir),
                "run_id": args.run_id,
                "status": "skipped_no_financial_covenants",
            },
        )
        _write_json(
            out_dir / "anchor_expansion.json",
            {
                "item_id": item_id,
                "seed_anchor_ids": [],
                "expanded_anchor_ids": [],
                "added_anchor_ids": [],
                "fill_gaps_up_to": 12,
                "neighbor_pad": 2,
            },
        )
        print(f"No financial_covenant anchors found in snippets: {snippets_path}")
        return 0

    # Expand the seed anchor set to make excerpt packs more complete within local covenant sections.
    cov_anchors_seed = list(cov_anchors)
    cov_anchors = _expand_anchor_ids(catalog=catalog, seed_anchor_ids=cov_anchors_seed, fill_gaps_up_to=12, neighbor_pad=2)

    canonical_text = prompt_view_path(paths, item_id).read_text(encoding="utf-8", errors="replace")
    contexts = _build_excerpt_pack(snippets_path, anchor_ids=cov_anchors, catalog=catalog, canonical_text=canonical_text)
    template = _read_text(Path(args.prompt))
    task = (
        f"Item: {item_id}\n"
        "Extract all financial covenants in the CONTEXT excerpt pack and return CovenantIR v0.1 JSON.\n"
        "Use one covenant per distinct financial test. Use open_items for missing/ambiguous terms.\n"
        "IMPORTANT: Ignore non-financial/negative covenants (e.g., Indebtedness/Liens/Restricted Payments) even if truncated.\n"
        "Do NOT emit open_items about out-of-scope negative covenants; only block for missing data needed to encode an in-scope financial covenant.\n"
    )
    base_prompt = _render_prompt(template, task=task, contexts=contexts)
    prompt = base_prompt

    _write_json(
        out_dir / "anchor_expansion.json",
        {
            "item_id": item_id,
            "seed_anchor_ids": cov_anchors_seed,
            "expanded_anchor_ids": cov_anchors,
            "added_anchor_ids": [a for a in cov_anchors if a not in set(cov_anchors_seed)],
            "fill_gaps_up_to": 12,
            "neighbor_pad": 2,
        },
    )
    _write_text(out_dir / "contexts.txt", contexts)
    _write_text(out_dir / "prompt.txt", base_prompt)

    complete_response_sync = _ensure_gateway_client_sync()
    gateway_url = args.gateway_url or DEFAULT_GATEWAY_URL

    raw: str | None = None
    parsed: Any = None
    last_errors: List[CovenantIRValidationError] = []

    attempt = 0
    while attempt < max(1, int(args.attempts)):
        attempt += 1
        _write_text(out_dir / f"prompt_attempt_{attempt}.txt", prompt)
        raw = complete_response_sync(
            model=args.model,
            prompt=prompt,
            base_url=gateway_url,
            reasoning={"effort": args.reasoning} if args.reasoning else None,
            temperature=float(args.temperature),
            max_output_tokens=None,
            timeout=float(args.timeout_seconds),
        )
        _write_text(out_dir / f"raw_attempt_{attempt}.txt", raw)

        try:
            parsed = json.loads(raw)
            _write_json(out_dir / f"parsed_attempt_{attempt}.json", parsed)
        except Exception as exc:
            parsed = None
            last_errors = [CovenantIRValidationError(code="json_parse", message=str(exc), json_path="/")]

        if parsed is not None:
            (canonicalized, canon_changes) = _canonicalize_structural(
                parsed, item_id=item_id, allowed_anchor_ids=cov_anchors
            )
            if canon_changes:
                _write_json(out_dir / f"canonicalized_attempt_{attempt}.json", canonicalized)
                _write_json(out_dir / f"canonicalization_changes_attempt_{attempt}.json", canon_changes)
            parsed = canonicalized

            last_errors = validate_covenant_ir(parsed)
            if not last_errors:
                last_errors.extend(validate_precision_first_policy(parsed))
            if not last_errors:
                last_errors.extend(_validate_open_items_in_scope(parsed))
            if not last_errors:
                last_errors.extend(_validate_anchor_ids_in_context(parsed, allowed_anchor_ids=cov_anchors))
            if not last_errors:
                last_errors.extend(_validate_contract_and_source_ids(parsed, item_id=item_id))
            if not last_errors:
                last_errors.extend(_validate_heading_coverage(parsed, contexts=contexts))
            if not last_errors:
                last_errors.extend(_validate_no_threshold_placeholders(parsed))
            if not last_errors:
                last_errors.extend(_validate_no_out_of_scope_covenants(parsed))

        if parsed is not None and not last_errors:
            (open_items_total, open_items_blocking) = _count_open_items(parsed)
            covs = parsed.get("covenants") if isinstance(parsed, dict) else None
            n_covs = len(covs) if isinstance(covs, list) else 0

            # In precision-first mode, blocking open_items is a valid output state — but during harness
            # pressure-tests we want to avoid "over-blocking" when the covenant is encodable (A1).
            # Therefore: if the model blocks and we still have attempts left, ask it to try again.
            if open_items_blocking and attempt < max(1, int(args.attempts)):
                last_errors = [
                    CovenantIRValidationError(
                        code="blocked_but_retrying",
                        message=(
                            "You returned blocking open_items. Retry and attempt to fully encode ALL in-scope financial covenants "
                            "as computable tests. Only emit blocking open_items if the CONTEXT is truly missing threshold values / "
                            "schedule rows / required terms. Do NOT block for A1 definitional complexity (treat adjusted metrics as inputs)."
                        ),
                        json_path="/open_items",
                    )
                ]
                _write_json(out_dir / f"validation_errors_attempt_{attempt}.json", [asdict(e) for e in last_errors])
                repair_appendix = _render_repair_prompt(raw_json=raw or "", errors=last_errors)
                _write_text(out_dir / f"repair_appendix_attempt_{attempt}.txt", repair_appendix)
                prompt = base_prompt + "\n\n" + repair_appendix
                continue

            _write_json(out_dir / "covenantir_validated.json", parsed)
            if open_items_blocking > 0:
                blocked_artifact = _to_blocked_artifact(
                    item_id=item_id,
                    run_id=args.run_id,
                    attempts_used=attempt,
                    blocked_reason="model_reported_missing_context",
                    cov_anchors=cov_anchors,
                    covenantir_validated=parsed if isinstance(parsed, dict) else {},
                    out_dir=out_dir,
                    last_errors=[],
                )
                _write_json(out_dir / "blocked_artifact.json", blocked_artifact)
            result = {
                "item_id": item_id,
                "run_id": args.run_id,
                "attempts_used": attempt,
                "ok": (open_items_blocking == 0),
                "status": "ok" if open_items_blocking == 0 else "blocked_artifact",
                "blocked_reason": "model_reported_missing_context" if open_items_blocking > 0 else None,
                "covenants_count": n_covs,
                "open_items_total": open_items_total,
                "open_items_blocking": open_items_blocking,
                "out_dir": str(out_dir),
            }
            _write_json(out_dir / "result.json", result)

            if open_items_blocking:
                print(
                    f"BLOCKED_ARTIFACT attempts_used={attempt} blocking_open_items={open_items_blocking} covenants={n_covs} out={out_dir}"
                )
                return 0

            print(f"OK attempts_used={attempt} covenants={n_covs} out={out_dir}")
            return 0

        _write_json(out_dir / f"validation_errors_attempt_{attempt}.json", [asdict(e) for e in last_errors])
        repair_appendix = _render_repair_prompt(raw_json=raw or "", errors=last_errors)
        _write_text(out_dir / f"repair_appendix_attempt_{attempt}.txt", repair_appendix)
        prompt = base_prompt + "\n\n" + repair_appendix

    # Produce a schema-valid, precision-first blocked CovenantIR document so downstream tooling can proceed
    # deterministically even when the model fails to emit valid CovenantIR after repair attempts.
    err_list = [asdict(e) for e in (last_errors or [])]
    issue = (
        f"Model output invalid after {attempt} attempt(s); see raw_attempt_*.txt and validation_errors_attempt_*.json. "
        f"Last_errors={json.dumps(err_list[:10])}"
    )
    salvage = {
        "schema_version": "covenant_ir_v0_1",
        "contract_id": item_id,
        "sources": [
            {
                "source_id": "S1",
                "kind": "excerpt_pack",
                "item_id": item_id,
                "anchor_ids": cov_anchors,
                "notes": None,
            }
        ],
        "indices": [],
        "tables": [],
        "derived": [],
        "open_items": [
            {
                "issue": issue,
                "source_refs": cov_anchors[: min(20, len(cov_anchors))],
                "suggested_parameters": [
                    "Inspect raw_attempt_*.txt and validation_errors_attempt_*.json to see the exact failure mode.",
                    "If this is a systematic failure, try rerunning with a higher-capability model.",
                ],
                "blocking": True,
            }
        ],
        "covenants": [],
    }
    _write_json(out_dir / "covenantir_validated.json", salvage)
    blocked_artifact = _to_blocked_artifact(
        item_id=item_id,
        run_id=args.run_id,
        attempts_used=attempt,
        blocked_reason="invalid_output_after_repair",
        cov_anchors=cov_anchors,
        covenantir_validated=salvage,
        out_dir=out_dir,
        last_errors=last_errors,
    )
    _write_json(out_dir / "blocked_artifact.json", blocked_artifact)
    result = {
        "item_id": item_id,
        "run_id": args.run_id,
        "attempts_used": attempt,
        "ok": False,
        "status": "blocked_artifact",
        "blocked_reason": "invalid_output_after_repair",
        "legacy_status": "blocked_invalid_output",
        "covenants_count": 0,
        "open_items_total": 1,
        "open_items_blocking": 1,
        "out_dir": str(out_dir),
    }
    _write_json(out_dir / "result.json", result)

    print(f"BLOCKED_ARTIFACT_INVALID_OUTPUT attempts_used={attempt} out={out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
