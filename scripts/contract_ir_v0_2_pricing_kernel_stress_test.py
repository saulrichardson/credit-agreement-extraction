#!/usr/bin/env python
"""
Stress test harness for the ContractIR v0.2 pricing-kernel approach.

Scope:
  - Two-pass pricing kernel: base-rate pass + spread/margin pass
  - Retrieval is abstracted away: we feed excerpt packs (contexts) directly.
  - Uses strict schema validation + additional hard gates:
      * anchor ids used MUST be in the provided context anchors
      * contract_id and sources[*].item_id MUST match the item_id for the case
  - Runs deterministic evaluation as an internal consistency check:
      * base-rate derived functions evaluate under synthetic index inputs
      * spread derived functions evaluate by picking keys/values from extracted tables when possible

Inputs:
  - Real examples: pull pricing-labeled snippets from an existing run's retrieval_v2/*.jsonl
  - Synthetic examples: embed several adversarial pricing snippets directly in this script

Outputs:
  - Writes a run folder under runs/<out_run_id>/ with per-case artifacts:
      contexts.txt, prompt.txt, raw attempts, validation errors, validated json, eval report, etc.
  - Writes summary.json with counts + per-case outcomes.

Usage:
  python scripts/contract_ir_v0_2_pricing_kernel_stress_test.py \
    --source-run-id dan-v2-20260106 \
    --max-items 18 \
    --model openai:gpt-5-nano
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from dataclasses import asdict, dataclass
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from pipeline.config import REQUIRED_MODEL, REQUIRED_REASONING  # noqa: E402
from pipeline.contract_ir_v0_2 import (  # noqa: E402
    ContractIREvalError,
    ContractIRValidationError,
    MultipleMatchingRows,
    NoMatchingRow,
    evaluate_function,
    validate_contract_ir,
)
from pipeline.anchors import load_anchor_catalog  # noqa: E402
from pipeline.indexing import DEFAULT_GATEWAY_URL, _ensure_gateway_client_sync  # type: ignore  # noqa: E402
from pipeline.config import Paths  # noqa: E402
from pipeline.excerpt_packs import build_excerpt_pack_from_canonical, expand_anchor_ids  # noqa: E402
from pipeline.utils import prompt_view_path  # noqa: E402


ANCHOR_ID_RE = re.compile(r"^A\d{4,}$")

# --- Keyword heuristics (real-example anchor selection) -----------------------------------

_BASE_RATE_KEYWORDS = (
    "prime rate",
    "federal funds",
    "nyfrb",
    "sofr",
    "term sofr",
    "libor",
    "eurodollar base rate",
    "alternate base rate",
    "adjusted libo",
    "adjusted libor",
    "reference rate",
    "reserve percentage",
    "rounded up",
    "round up",
    "floor",
)

_SPREAD_KEYWORDS = (
    "applicable margin",
    "applicable rate",
    "margin",
    "spread",
    "basis points",
    "bps",
    "pricing level",
    "eurodollar loans",
    "base rate loans",
)


def _norm(s: str) -> str:
    return " ".join((s or "").lower().split())


def _keyword_score(text: str, keywords: Sequence[str]) -> int:
    t = _norm(text)
    score = 0
    for kw in keywords:
        if kw in t:
            score += 1
    return score


def _parse_snippets(snippets_path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with snippets_path.open("r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if isinstance(obj, dict):
                rows.append(obj)
    return rows


def _anchor_order(aid: str) -> int:
    m = re.match(r"^A(\d+)$", aid.strip())
    if not m:
        return 1_000_000_000
    return int(m.group(1))


def _build_context_from_records(records: Sequence[Dict[str, Any]], *, anchor_ids: Sequence[str]) -> str:
    by_anchor: Dict[str, Dict[str, Any]] = {}
    for rec in records:
        aid = rec.get("anchor_id")
        if isinstance(aid, str) and aid not in by_anchor:
            by_anchor[aid] = rec

    blocks: List[str] = []
    for aid in sorted({a for a in anchor_ids if isinstance(a, str) and a.strip()}, key=_anchor_order):
        rec = by_anchor.get(aid)
        if not rec:
            blocks.append(f"[[{aid}]]\n<MISSING SNIPPET>")
            continue
        snippet = str(rec.get("snippet") or "").rstrip()
        blocks.append(f"[[{aid}]]\n{snippet}")
    return "\n\n".join(blocks).strip() + "\n"


def _build_context_from_canonical(
    *,
    canonical_text: str,
    catalog: Mapping[str, Mapping[str, Any]],
    seed_anchor_ids: Sequence[str],
    fill_gaps_up_to: int,
    neighbor_pad: int,
) -> Tuple[str, List[str]]:
    expanded = expand_anchor_ids(
        catalog=catalog,
        seed_anchor_ids=seed_anchor_ids,
        fill_gaps_up_to=int(fill_gaps_up_to),
        neighbor_pad=int(neighbor_pad),
    )
    ctx = build_excerpt_pack_from_canonical(canonical_text=canonical_text, catalog=catalog, anchor_ids=expanded)
    return ctx, expanded


def _select_pricing_anchors(
    records: Sequence[Dict[str, Any]],
    *,
    max_base_rate: int,
    max_spread: int,
) -> Tuple[List[str], List[str]]:
    base: List[Tuple[int, str]] = []
    spread: List[Tuple[int, str]] = []
    for rec in records:
        aid = rec.get("anchor_id")
        if not isinstance(aid, str) or not ANCHOR_ID_RE.fullmatch(aid.strip()):
            continue
        cats = rec.get("categories") or []
        if not (isinstance(cats, list) and "pricing" in cats):
            continue
        snippet = str(rec.get("snippet") or "")
        b = _keyword_score(snippet, _BASE_RATE_KEYWORDS)
        s = _keyword_score(snippet, _SPREAD_KEYWORDS)

        # Heuristic: "Base Rate Loans" rows are usually margin tables; require more signal for base-rate formulas.
        if b >= 2 and b >= s:
            base.append((b, aid))
        if s >= 1 and s > b:
            spread.append((s, aid))

    # Sort by score desc, then anchor order asc (stable).
    base_ids = [aid for _, aid in sorted(base, key=lambda t: (-t[0], _anchor_order(t[1])))][: max_base_rate]
    spread_ids = [aid for _, aid in sorted(spread, key=lambda t: (-t[0], _anchor_order(t[1])))][: max_spread]

    # De-dup across lists (prefer base_rate classification when ambiguous).
    spread_ids = [aid for aid in spread_ids if aid not in set(base_ids)]
    return base_ids, spread_ids


# --- Prompt runner (LLM + repair) --------------------------------------------------------


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _render_prompt(template: str, *, task: str, contexts: str) -> str:
    return template.replace("{{TASK}}", task).replace("{{CONTEXTS}}", contexts)


def _render_repair_prompt(*, raw_json: str, errors: List[ContractIRValidationError]) -> str:
    err_list = [asdict(e) for e in errors]
    return (
        "Your previous response did not validate against the ContractIR v0.2 schema and/or hard gates.\n"
        "Return corrected JSON ONLY. Do not include commentary or code fences.\n\n"
        "REPAIR REMINDERS (common failure modes):\n"
        "- AST nodes must be ONLY one of:\n"
        "  1) {\"lit\": {\"type\": \"...\", \"value\": ...}}\n"
        "  2) {\"var\": \"name\"}\n"
        "  3) {\"op\": \"op_name\", \"args\": [<node>, ...]}\n"
        "  DO NOT use {\"add\": [...]} or {\"index\": [...]} — use {\"op\":\"add\",\"args\":[...]} etc.\n"
        "- NEVER use bare strings/numbers where an AST node is required. Wrap strings as {\"lit\":{\"type\":\"string\",\"value\":\"...\"}}.\n"
        "- derived[*].args must be an array of {\"name\": ..., \"type\": ...} objects (NOT AST nodes).\n"
        "- Do NOT use placeholder arg names like \"...\". Every arg name must be an identifier (e.g., date, leverage_ratio).\n"
        "- source_refs must be an array of anchor-id strings like \"A0001\" (NOT objects).\n"
        "- index(series_id, date): MUST be {\"op\":\"index\",\"args\":[{\"lit\":{\"type\":\"string\",\"value\":\"SeriesId\"}}, {\"var\":\"date\"}]}\n"
        "  - series_id MUST be an AST string literal node (not a bare string)\n"
        "  - NOTE: indices[].series_id (the declaration list) is a PLAIN STRING field, not an AST node\n"
        "  - the 2nd arg MUST be a var declared type=date; do NOT hardcode date literals\n"
        "- For add/sub, operands must have matching numeric kinds. Example: FedFundsRate + 0.50% uses a RATE literal {\"lit\":{\"type\":\"rate\",\"value\":\"0.005\"}} (not decimal).\n"
        "- Numeric literal values MUST be strings (e.g. \"0.005\"), never numbers.\n"
        "- If a range bound is missing/unbounded, set the CELL to null (e.g. \"lower_bound\": null), not lit.value=null.\n"
        "- Every indices[] entry MUST include both series_id AND unit.\n\n"
        "VALIDATION_ERRORS_JSON:\n"
        f"{json.dumps(err_list, indent=2)}\n\n"
        "PREVIOUS_JSON:\n"
        f"{raw_json}\n"
    )


def _validate_anchor_ids_in_context(doc: Any, allowed_anchor_ids: Sequence[str]) -> List[ContractIRValidationError]:
    allowed = set([a.strip() for a in allowed_anchor_ids if isinstance(a, str) and ANCHOR_ID_RE.fullmatch(a.strip())])
    out: List[ContractIRValidationError] = []

    def _walk(obj: Any, path: List[str]) -> None:
        if isinstance(obj, dict):
            for k, v in obj.items():
                if k in ("source_refs", "anchor_ids"):
                    if isinstance(v, list):
                        for idx, aid in enumerate(v):
                            if isinstance(aid, str) and aid.strip() in allowed:
                                continue
                            out.append(
                                ContractIRValidationError(
                                    code="anchor_not_in_context",
                                    message=f"Anchor id {aid!r} is not in provided context anchors {sorted(allowed)[:50]}",
                                    json_path="/" + "/".join(path + [k, str(idx)]),
                                )
                            )
                    else:
                        out.append(
                            ContractIRValidationError(
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


def _validate_contract_and_source_ids(doc: Any, *, item_id: str) -> List[ContractIRValidationError]:
    out: List[ContractIRValidationError] = []
    if not isinstance(doc, dict):
        return out

    cid = doc.get("contract_id")
    if cid != item_id:
        out.append(
            ContractIRValidationError(
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
                    ContractIRValidationError(
                        code="source_item_id_mismatch",
                        message=f"sources[{si}].item_id must equal item_id {item_id!r}; got {sid!r}",
                        json_path=f"/sources/{si}/item_id",
                    )
                )
    return out


def _validate_pass_policy(
    doc: Any,
    *,
    pass_kind: str,
) -> List[ContractIRValidationError]:
    """Enforce pass-level policies that are not part of the JSON Schema.

    pass_kind:
      - "base_rate": tables MUST be []; all derived semantic_role MUST be base_rate
      - "spread": all derived semantic_role MUST be spread; if open_items non-empty then derived=[] and tables=[]
    """

    out: List[ContractIRValidationError] = []
    if not isinstance(doc, dict):
        return out

    derived = doc.get("derived") or []
    tables = doc.get("tables") or []
    open_items = doc.get("open_items") or []

    if pass_kind == "base_rate":
        if not (isinstance(tables, list) and len(tables) == 0):
            out.append(
                ContractIRValidationError(
                    code="pass_policy",
                    message="base-rate pass policy: tables must be []",
                    json_path="/tables",
                )
            )
        if isinstance(derived, list):
            for di, d in enumerate(derived):
                if not isinstance(d, dict):
                    continue
                if d.get("semantic_role") != "base_rate":
                    out.append(
                        ContractIRValidationError(
                            code="pass_policy",
                            message="base-rate pass policy: every derived.semantic_role must be 'base_rate'",
                            json_path=f"/derived/{di}/semantic_role",
                        )
                    )

    if pass_kind == "spread":
        if isinstance(derived, list):
            for di, d in enumerate(derived):
                if not isinstance(d, dict):
                    continue
                if d.get("semantic_role") != "spread":
                    out.append(
                        ContractIRValidationError(
                            code="pass_policy",
                            message="spread pass policy: every derived.semantic_role must be 'spread'",
                            json_path=f"/derived/{di}/semantic_role",
                        )
                    )
        if isinstance(open_items, list) and len(open_items) > 0:
            if not (isinstance(derived, list) and len(derived) == 0):
                out.append(
                    ContractIRValidationError(
                        code="pass_policy",
                        message="spread pass policy: open_items requires derived=[] (no partial outputs)",
                        json_path="/derived",
                    )
                )
            if not (isinstance(tables, list) and len(tables) == 0):
                out.append(
                    ContractIRValidationError(
                        code="pass_policy",
                        message="spread pass policy: open_items requires tables=[] (no partial outputs)",
                        json_path="/tables",
                    )
                )

    return out


@dataclass(frozen=True)
class PromptAttemptResult:
    ok: bool
    attempts_used: int
    schema_valid: bool
    anchor_context_valid: bool
    pass_policy_valid: bool
    open_items_count: int
    derived_count: int
    tables_count: int
    validation_errors: List[Dict[str, Any]]


def _run_prompt_with_repairs(
    *,
    client,
    out_dir: Path,
    item_id: str,
    prompt_path: Path,
    task: str,
    contexts: str,
    allowed_anchor_ids: Sequence[str],
    pass_kind: str,
    attempts: int,
    gateway_url: str,
    timeout_seconds: float,
    temperature: float,
    model: str,
    reasoning: str,
    max_output_tokens: Optional[int],
) -> Tuple[Optional[dict], PromptAttemptResult]:
    template = prompt_path.read_text(encoding="utf-8", errors="replace")
    prompt = _render_prompt(template, task=task, contexts=contexts)

    _write_text(out_dir / "contexts.txt", contexts)
    _write_text(out_dir / "prompt.txt", prompt)

    raw: Optional[str] = None
    parsed: Any = None
    last_errors: List[ContractIRValidationError] = []

    for attempt in range(1, max(1, int(attempts)) + 1):
        raw = client(
            model=model,
            prompt=prompt,
            base_url=gateway_url,
            reasoning={"effort": reasoning} if reasoning else None,
            temperature=float(temperature),
            max_output_tokens=max_output_tokens,
            timeout=float(timeout_seconds),
        )
        _write_text(out_dir / f"raw_attempt_{attempt}.txt", raw)

        try:
            parsed = json.loads(raw)
            _write_json(out_dir / f"parsed_attempt_{attempt}.json", parsed)
        except Exception as exc:
            parsed = None
            last_errors = [ContractIRValidationError(code="json_parse", message=str(exc), json_path="/")]

        if parsed is not None:
            last_errors = validate_contract_ir(parsed)
            if not last_errors:
                last_errors.extend(_validate_anchor_ids_in_context(parsed, allowed_anchor_ids))
            if not last_errors:
                last_errors.extend(_validate_contract_and_source_ids(parsed, item_id=item_id))
            if not last_errors:
                last_errors.extend(_validate_pass_policy(parsed, pass_kind=pass_kind))

        if parsed is not None and not last_errors:
            _write_json(out_dir / "contractir_validated.json", parsed)
            open_items = parsed.get("open_items") if isinstance(parsed, dict) else []
            derived = parsed.get("derived") if isinstance(parsed, dict) else []
            tables = parsed.get("tables") if isinstance(parsed, dict) else []
            return parsed, PromptAttemptResult(
                ok=True,
                attempts_used=attempt,
                schema_valid=True,
                anchor_context_valid=True,
                pass_policy_valid=True,
                open_items_count=len(open_items) if isinstance(open_items, list) else 0,
                derived_count=len(derived) if isinstance(derived, list) else 0,
                tables_count=len(tables) if isinstance(tables, list) else 0,
                validation_errors=[],
            )

        _write_json(out_dir / f"validation_errors_attempt_{attempt}.json", [asdict(e) for e in last_errors])
        prompt = _render_repair_prompt(raw_json=raw or "", errors=last_errors)

    # Failed attempts.
    schema_valid = False
    anchor_context_valid = False
    pass_policy_valid = False
    if parsed is not None:
        schema_valid = len([e for e in last_errors if e.code == "schema"]) == 0
        anchor_context_valid = len([e for e in last_errors if e.code == "anchor_not_in_context"]) == 0
        pass_policy_valid = len([e for e in last_errors if e.code == "pass_policy"]) == 0

    open_items = parsed.get("open_items") if isinstance(parsed, dict) else []
    derived = parsed.get("derived") if isinstance(parsed, dict) else []
    tables = parsed.get("tables") if isinstance(parsed, dict) else []
    return None, PromptAttemptResult(
        ok=False,
        attempts_used=max(1, int(attempts)),
        schema_valid=schema_valid,
        anchor_context_valid=anchor_context_valid,
        pass_policy_valid=pass_policy_valid,
        open_items_count=len(open_items) if isinstance(open_items, list) else 0,
        derived_count=len(derived) if isinstance(derived, list) else 0,
        tables_count=len(tables) if isinstance(tables, list) else 0,
        validation_errors=[asdict(e) for e in last_errors],
    )


# --- Internal evaluation (consistency checks) ---------------------------------------------


def _decimal_str(x: Decimal) -> str:
    s = format(x, "f")
    if "." in s:
        s = s.rstrip("0").rstrip(".") or "0"
    return s


def _default_value_for_type(typ: str) -> Any:
    if typ == "date":
        return "2000-01-01"
    if typ == "decimal":
        # Avoid common divide-by-(1 - x) pitfalls when x=1.0.
        return "0.10"
    if typ == "rate":
        return "0.05"
    if typ == "bps":
        return "50"
    if typ == "money":
        return "100"
    if typ == "integer":
        return "1"
    if typ == "bool":
        return True
    if typ == "string":
        return "X"
    return "X"


def _indices_fixture(contract_ir: Mapping[str, Any], *, date: str) -> Dict[str, Dict[str, str]]:
    out: Dict[str, Dict[str, str]] = {}
    for idx in contract_ir.get("indices") or []:
        if not isinstance(idx, dict):
            continue
        series_id = idx.get("series_id")
        unit = idx.get("unit")
        if not isinstance(series_id, str) or not series_id.strip():
            continue
        val = "0.05"
        if unit == "rate":
            val = "0.05"
        elif unit == "bps":
            val = "50"
        elif unit == "decimal":
            val = "1.0"
        elif unit == "money":
            val = "100"
        out[series_id] = {date: val}
    return out


def _iter_nodes(node: Any) -> Iterable[Dict[str, Any]]:
    if isinstance(node, dict):
        yield node
        for v in node.values():
            yield from _iter_nodes(v)
    elif isinstance(node, list):
        for v in node:
            yield from _iter_nodes(v)


def _first_op(node: Any, op_name: str) -> Optional[Dict[str, Any]]:
    for n in _iter_nodes(node):
        if isinstance(n, dict) and n.get("op") == op_name and isinstance(n.get("args"), list):
            return n
    return None


def _lit_value(node: Any) -> Optional[Any]:
    if not isinstance(node, dict):
        return None
    lit = node.get("lit")
    if not isinstance(lit, dict):
        return None
    return lit.get("value")


def _get_table(contract_ir: Mapping[str, Any], table_id: str) -> Optional[Dict[str, Any]]:
    for t in contract_ir.get("tables") or []:
        if not isinstance(t, dict):
            continue
        if t.get("table_id") == table_id:
            return t
    return None


def _first_row_cell_literal(table: Mapping[str, Any], col: str) -> Optional[Any]:
    rows = table.get("rows") or []
    if not isinstance(rows, list):
        return None
    for r in rows:
        if not isinstance(r, dict):
            continue
        cells = r.get("cells")
        if not isinstance(cells, dict):
            continue
        v = cells.get(col)
        val = _lit_value(v)
        if val is not None:
            return val
    return None


def _pick_lookup2_args(
    *,
    contract_ir: Mapping[str, Any],
    expr: Mapping[str, Any],
) -> Dict[str, Any]:
    args = expr.get("args") or []
    if not isinstance(args, list) or len(args) != 6:
        return {}
    table_id = _lit_value(args[0])
    key1_col = _lit_value(args[1])
    key1_val_node = args[2]
    key2_col = _lit_value(args[3])
    key2_val_node = args[4]
    if not all(isinstance(x, str) for x in (table_id, key1_col, key2_col)):
        return {}
    table = _get_table(contract_ir, str(table_id))
    if not table:
        return {}
    # pick from first row
    v1 = _first_row_cell_literal(table, str(key1_col))
    v2 = _first_row_cell_literal(table, str(key2_col))
    out: Dict[str, Any] = {}
    if isinstance(key1_val_node, dict) and "var" in key1_val_node and isinstance(key1_val_node["var"], str) and v1 is not None:
        out[key1_val_node["var"]] = v1
    if isinstance(key2_val_node, dict) and "var" in key2_val_node and isinstance(key2_val_node["var"], str) and v2 is not None:
        out[key2_val_node["var"]] = v2
    return out


def _pick_lookup_args(
    *,
    contract_ir: Mapping[str, Any],
    expr: Mapping[str, Any],
) -> Dict[str, Any]:
    args = expr.get("args") or []
    if not isinstance(args, list) or len(args) != 4:
        return {}
    table_id = _lit_value(args[0])
    key_col = _lit_value(args[1])
    key_val_node = args[2]
    if not all(isinstance(x, str) for x in (table_id, key_col)):
        return {}
    table = _get_table(contract_ir, str(table_id))
    if not table:
        return {}
    v = _first_row_cell_literal(table, str(key_col))
    out: Dict[str, Any] = {}
    if isinstance(key_val_node, dict) and "var" in key_val_node and isinstance(key_val_node["var"], str) and v is not None:
        out[key_val_node["var"]] = v
    return out


def _pick_lookup_range_args(
    *,
    contract_ir: Mapping[str, Any],
    expr: Mapping[str, Any],
) -> Dict[str, Any]:
    args = expr.get("args") or []
    if not isinstance(args, list) or len(args) != 7:
        return {}
    table_id = _lit_value(args[0])
    key_val_node = args[1]
    lower_col = _lit_value(args[2])
    lower_cmp_col = _lit_value(args[3])
    upper_col = _lit_value(args[4])
    upper_cmp_col = _lit_value(args[5])
    if not all(isinstance(x, str) for x in (table_id, lower_col, lower_cmp_col, upper_col, upper_cmp_col)):
        return {}
    table = _get_table(contract_ir, str(table_id))
    if not table:
        return {}

    # Pick a value satisfying the first row with any bounds.
    rows = table.get("rows") or []
    if not isinstance(rows, list) or not rows:
        return {}
    row = rows[0]
    if not isinstance(row, dict):
        return {}
    cells = row.get("cells")
    if not isinstance(cells, dict):
        return {}

    def _dec(x: Any) -> Optional[Decimal]:
        if x is None:
            return None
        try:
            return Decimal(str(x))
        except Exception:
            return None

    lower = _dec(_lit_value(cells.get(str(lower_col))))
    upper = _dec(_lit_value(cells.get(str(upper_col))))
    lower_cmp = _lit_value(cells.get(str(lower_cmp_col)))
    upper_cmp = _lit_value(cells.get(str(upper_cmp_col)))

    # Conservative picks:
    if lower is not None and upper is not None:
        pick = (lower + upper) / Decimal("2")
    elif lower is not None and upper is None:
        pick = lower + Decimal("0.01")
    elif lower is None and upper is not None:
        pick = upper - Decimal("0.01")
    else:
        pick = Decimal("0")

    # If strict bounds, move slightly inside.
    if lower is not None and str(lower_cmp).lower() == "gt":
        pick = max(pick, lower + Decimal("0.01"))
    if upper is not None and str(upper_cmp).lower() == "lt":
        pick = min(pick, upper - Decimal("0.01"))

    out: Dict[str, Any] = {}
    if isinstance(key_val_node, dict) and "var" in key_val_node and isinstance(key_val_node["var"], str):
        out[key_val_node["var"]] = _decimal_str(pick)
    return out


def _suggest_args_from_expr(contract_ir: Mapping[str, Any], expr: Any) -> Dict[str, Any]:
    # Prefer structured hints from table lookups.
    op = _first_op(expr, "lookup2")
    if op:
        return _pick_lookup2_args(contract_ir=contract_ir, expr=op)
    op = _first_op(expr, "lookup")
    if op:
        return _pick_lookup_args(contract_ir=contract_ir, expr=op)
    op = _first_op(expr, "lookup_range")
    if op:
        return _pick_lookup_range_args(contract_ir=contract_ir, expr=op)
    # lookup_rule is intentionally not auto-solved in this harness (too many shapes); rely on targeted evals instead.
    return {}


def _eval_contract_ir_consistency(doc: Mapping[str, Any]) -> Dict[str, Any]:
    """Evaluate every derived function with best-effort argument synthesis.

    This is NOT a correctness oracle vs the agreement; it is a structural integrity check:
      - expr is evaluatable under some plausible inputs
      - table wiring (key columns, literal types) is coherent
    """

    date = "2000-01-01"
    indices = _indices_fixture(doc, date=date)
    derived = doc.get("derived") or []
    out: Dict[str, Any] = {"date": date, "derived": [], "ok": True}

    if not isinstance(derived, list):
        return out

    for d in derived:
        if not isinstance(d, dict):
            continue
        fn_id = str(d.get("fn_id") or "")
        args_defs = d.get("args") or []
        expr = d.get("expr")
        returns = d.get("returns")

        args: Dict[str, Any] = {}
        if isinstance(args_defs, list):
            for a in args_defs:
                if not isinstance(a, dict):
                    continue
                name = a.get("name")
                typ = a.get("type")
                if isinstance(name, str) and isinstance(typ, str):
                    args[name] = _default_value_for_type(typ)

        if isinstance(expr, dict):
            args.update(_suggest_args_from_expr(doc, expr))

        entry: Dict[str, Any] = {"fn_id": fn_id, "returns": returns, "args": args, "ok": False}
        try:
            tv = evaluate_function(doc, fn_id=fn_id, args=args, indices=indices)
            val = tv.value
            if isinstance(val, Decimal):
                rendered: Any = _decimal_str(val)
            else:
                rendered = val
            entry.update({"ok": True, "kind": tv.kind, "value": rendered})
        except (NoMatchingRow, MultipleMatchingRows) as exc:
            # This is a common outcome for range/rule tables if our guessed args land in a gap.
            entry.update({"ok": False, "error_kind": type(exc).__name__, "error": str(exc)})
            out["ok"] = False
        except ContractIREvalError as exc:
            entry.update({"ok": False, "error_kind": "ContractIREvalError", "error": str(exc)})
            out["ok"] = False
        except Exception as exc:  # defensive
            entry.update({"ok": False, "error_kind": type(exc).__name__, "error": str(exc)})
            out["ok"] = False
        out["derived"].append(entry)

    return out


# --- Cases ------------------------------------------------------------------------------


@dataclass(frozen=True)
class CaseSpec:
    case_id: str
    item_id: str
    pass_kind: str  # "base_rate" or "spread"
    contexts: str
    allowed_anchor_ids: List[str]
    task: str
    prompt_paths: List[str]  # attempted in order (first success wins)


def _synthetic_cases() -> List[CaseSpec]:
    """Synthetic, retrieval-agnostic cases with known structure."""

    # Note: we keep item_id synthetic; contract_id must match.
    cases: List[CaseSpec] = []

    # Base rate: Alternate Base Rate = max(Prime, FedFunds + 0.50%) rounded up to nearest 1/100 of 1%.
    cases.append(
        CaseSpec(
            case_id="syn_base_rate_alt_base_rate_rounding",
            item_id="SYNTH_BASE_1",
            pass_kind="base_rate",
            allowed_anchor_ids=["A0001"],
            task=(
                "Build ContractIR v0.2 for item_id=SYNTH_BASE_1.\n"
                "Goal: encode the Alternate Base Rate definition and rounding.\n"
                "Requirements:\n"
                "- derived fn_id MUST be AlternateBaseRate\n"
                "- semantic_role MUST be base_rate\n"
                "- args: date (type=date)\n"
                "- indices MUST include PrimeRate (unit=rate), FederalFundsRate (unit=rate)\n"
                "- expr MUST compute: round_up_to_increment(max(index(PrimeRate,date), add(index(FederalFundsRate,date), 0.005)), 0.0001)\n"
            ),
            contexts=(
                "[[A0001]]\n"
                "\"Alternate Base Rate\" means, for any day, an interest rate per annum equal to the greater of (a) the Prime Rate and "
                "(b) the Federal Funds Effective Rate plus one-half of one percent (0.50%). The Alternate Base Rate shall be rounded up "
                "to the nearest 1/100 of 1%.\n"
            ),
            prompt_paths=["prompts/contract_ir_base_rate_v2.txt"],
        )
    )

    # Base rate: reserve-adjusted LIBOR.
    cases.append(
        CaseSpec(
            case_id="syn_base_rate_adjusted_libo_reserve_pct",
            item_id="SYNTH_BASE_2",
            pass_kind="base_rate",
            allowed_anchor_ids=["A0001"],
            task=(
                "Build ContractIR v0.2 for item_id=SYNTH_BASE_2.\n"
                "Goal: encode the reserve-adjusted LIBO rate.\n"
                "Requirements:\n"
                "- derived fn_id MUST be AdjustedLIBOR\n"
                "- semantic_role MUST be base_rate\n"
                "- args: date (type=date), euro_reserve_pct (type=decimal)\n"
                "- indices MUST include LIBORRate (unit=rate)\n"
                "- expr MUST compute: div(index(LIBORRate,date), sub(1.00, euro_reserve_pct))\n"
            ),
            contexts=(
                "[[A0001]]\n"
                "\"Adjusted LIBOR\" means, for any day, a rate per annum equal to LIBOR divided by (1.00 minus the Euro Reserve Percentage).\n"
            ),
            prompt_paths=["prompts/contract_ir_base_rate_v2.txt"],
        )
    )

    # Spread: 2D rule-table over ICR/FCCR (the canonical "2D grid" case).
    cases.append(
        CaseSpec(
            case_id="syn_spread_lookup_rule_2d_icr_fccr",
            item_id="SYNTH_SPR_1",
            pass_kind="spread",
            allowed_anchor_ids=["A0001"],
            task=(
                "Build ContractIR v0.2 for item_id=SYNTH_SPR_1.\n"
                "Goal: encode the Applicable Margin schedule as a RULE TABLE.\n"
                "Requirements:\n"
                "- Create table_id: ApplicableMarginByCoverageRatios\n"
                "- columns MUST include: rule_label (string), predicate (bool), margin_bps (bps)\n"
                "- derived fn_id MUST be ApplicableMarginLIBO\n"
                "- semantic_role MUST be spread\n"
                "- args: interest_coverage_ratio (type=decimal), fixed_charge_coverage_ratio (type=decimal)\n"
                "- expr MUST compute: bps_to_rate(lookup_rule(\"ApplicableMarginByCoverageRatios\",\"predicate\",\"margin_bps\"))\n"
                "- Preserve OR/AND exactly as written.\n"
            ),
            contexts=(
                "[[A0001]]\n"
                "Applicable Margin (LIBO Rate Advances):\n"
                "  - Row 1: ICR < 1.75 OR FCCR < 1.20  -> 2.50%\n"
                "  - Row 2: ICR >= 1.75 AND FCCR >= 1.20 AND ICR < 2.50 AND FCCR < 1.40 -> 2.25%\n"
                "  - Row 3: ICR >= 2.50 AND FCCR >= 1.40 -> 2.00%\n"
            ),
            prompt_paths=["prompts/contract_ir_spread_v2_lookup_rule.txt"],
        )
    )

    # Spread: leverage-based range schedule.
    cases.append(
        CaseSpec(
            case_id="syn_spread_lookup_range_leverage",
            item_id="SYNTH_SPR_2",
            pass_kind="spread",
            allowed_anchor_ids=["A0001"],
            task=(
                "Build ContractIR v0.2 for item_id=SYNTH_SPR_2.\n"
                "Goal: encode the Eurodollar margin schedule by leverage ratio.\n"
                "Requirements:\n"
                "- Create table_id: EurodollarMarginByLeverage\n"
                "- columns MUST include: bucket_label (string), lower_bound (decimal), lower_cmp (string), upper_bound (decimal), upper_cmp (string), margin_bps (bps)\n"
                "- derived fn_id MUST be EurodollarBorrowingMargin\n"
                "- semantic_role MUST be spread\n"
                "- args: leverage_ratio (type=decimal)\n"
                "- expr MUST compute: bps_to_rate(lookup_range(\"EurodollarMarginByLeverage\", leverage_ratio, \"lower_bound\",\"lower_cmp\",\"upper_bound\",\"upper_cmp\",\"margin_bps\"))\n"
            ),
            contexts=(
                "[[A0001]]\n"
                "Applicable Margin for Eurodollar Loans based on Consolidated Leverage Ratio:\n"
                "  - < 2.00 to 1.00: 150 bps\n"
                "  - >= 2.00 to 1.00 and < 3.00 to 1.00: 175 bps\n"
                "  - >= 3.00 to 1.00: 200 bps\n"
            ),
            prompt_paths=["prompts/contract_ir_spread_v2_lookup_range.txt"],
        )
    )

    # Spread: rating-bucket x loan-type grid (lookup2).
    cases.append(
        CaseSpec(
            case_id="syn_spread_lookup2_rating_x_loan_type",
            item_id="SYNTH_SPR_3",
            pass_kind="spread",
            allowed_anchor_ids=["A0001"],
            task=(
                "Build ContractIR v0.2 for item_id=SYNTH_SPR_3.\n"
                "Goal: encode the Applicable Margin grid as a 2D lookup2 table.\n"
                "Requirements:\n"
                "- Create table_id: ApplicableMarginGrid\n"
                "- columns MUST include: rating_bucket (string), loan_type (string), margin_bps (bps)\n"
                "- derived fn_id MUST be ApplicableMargin\n"
                "- semantic_role MUST be spread\n"
                "- args: rating_bucket (type=string), loan_type (type=string)\n"
                "- expr MUST compute: bps_to_rate(lookup2(\"ApplicableMarginGrid\",\"rating_bucket\",rating_bucket,\"loan_type\",loan_type,\"margin_bps\"))\n"
            ),
            contexts=(
                "[[A0001]]\n"
                "Applicable Margin:\n"
                "Rating    Term SOFR Loans   Base Rate Loans\n"
                "BBB/Baa2  175 bps          75 bps\n"
                "BB/Ba2    250 bps          150 bps\n"
            ),
            prompt_paths=["prompts/contract_ir_spread_v2_lookup2.txt"],
        )
    )

    return cases


def _load_manifest_item_ids(run_dir: Path) -> List[str]:
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.exists():
        raise SystemExit(f"Missing manifest.json: {manifest_path}")
    doc = json.loads(manifest_path.read_text(encoding="utf-8"))
    items = doc.get("items")
    if not isinstance(items, list):
        raise SystemExit(f"manifest.json missing items[]: {manifest_path}")
    out: List[str] = []
    for it in items:
        if isinstance(it, dict) and isinstance(it.get("item_id"), str):
            out.append(it["item_id"])
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source-run-id", default="dan-v2-20260106", help="Run id that contains retrieval_v2 snippets (real examples).")
    ap.add_argument("--out-run-id", default=None, help="Output run id (writes under runs/<out-run-id>/).")
    ap.add_argument("--gateway-url", default=DEFAULT_GATEWAY_URL)
    ap.add_argument("--model", default=REQUIRED_MODEL)
    ap.add_argument("--reasoning", default=REQUIRED_REASONING)
    ap.add_argument("--timeout-seconds", type=float, default=600.0)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--attempts", type=int, default=3, help="Max validation/repair attempts per pass.")
    ap.add_argument("--max-output-tokens", type=int, default=4000)
    ap.add_argument("--max-items", type=int, default=18)
    ap.add_argument("--max-base-rate-anchors", type=int, default=8)
    ap.add_argument("--max-spread-anchors", type=int, default=10)
    ap.add_argument("--include-synthetic", action="store_true", help="Include synthetic cases (recommended).")
    ap.add_argument("--skip-real", action="store_true", help="Skip real examples and run only synthetic.")
    ap.add_argument("--item-id", action="append", default=[], help="Optional item_id(s) to run (repeatable).")
    ap.add_argument("--case-filter", default=None, help="If provided, only run cases whose id contains this substring.")
    args = ap.parse_args()

    out_run_id = args.out_run_id or f"pricing-kernel-stress-{time.strftime('%Y%m%d-%H%M%S')}"
    out_dir = PROJECT_ROOT / "runs" / out_run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    client = _ensure_gateway_client_sync()
    gateway_url = str(args.gateway_url)

    base_rate_prompt = Path("prompts/contract_ir_base_rate_v2.txt")
    spread_prompts = [
        Path("prompts/contract_ir_spread_v2_lookup2.txt"),
        Path("prompts/contract_ir_spread_v2_lookup_range.txt"),
        Path("prompts/contract_ir_spread_v2_lookup_rule.txt"),
        Path("prompts/contract_ir_spread_v2_specialized.txt"),
    ]

    # --- Build cases -------------------------------------------------------------------
    cases: List[CaseSpec] = []

    if args.include_synthetic:
        cases.extend(_synthetic_cases())

    if not args.skip_real:
        source_run_dir = PROJECT_ROOT / "runs" / str(args.source_run_id)
        retrieval_dir = source_run_dir / "retrieval_v2"
        if not retrieval_dir.exists():
            raise SystemExit(f"Missing retrieval_v2 dir: {retrieval_dir}")

        source_paths = Paths(root=PROJECT_ROOT, run_id=str(args.source_run_id))

        if args.item_id:
            item_ids = [i for i in args.item_id if isinstance(i, str) and i.strip()]
        else:
            item_ids = _load_manifest_item_ids(source_run_dir)
        item_ids = item_ids[: max(0, int(args.max_items))]

        for item_id in item_ids:
            snippets_path = retrieval_dir / f"{item_id}_snippets.jsonl"
            if not snippets_path.exists():
                continue
            records = _parse_snippets(snippets_path)
            base_anchors, spread_anchors = _select_pricing_anchors(
                records,
                max_base_rate=int(args.max_base_rate_anchors),
                max_spread=int(args.max_spread_anchors),
            )

            if base_anchors:
                catalog = load_anchor_catalog(source_paths, item_id)
                canonical_text = prompt_view_path(source_paths, item_id).read_text(encoding="utf-8", errors="replace")
                contexts, allowed = _build_context_from_canonical(
                    canonical_text=canonical_text,
                    catalog=catalog,
                    seed_anchor_ids=base_anchors,
                    fill_gaps_up_to=12,
                    neighbor_pad=2,
                )
                cases.append(
                    CaseSpec(
                        case_id=f"real_{item_id}__base_rate",
                        item_id=item_id,
                        pass_kind="base_rate",
                        contexts=contexts,
                        allowed_anchor_ids=allowed,
                        task=(
                            f"Build ContractIR v0.2 for item_id={item_id}.\n"
                            "Extract ALL benchmark/base-rate definitions present in CONTEXT.\n"
                            "Do not encode spreads/margins/fees in this pass.\n"
                        ),
                        prompt_paths=[str(base_rate_prompt)],
                    )
                )

            if spread_anchors:
                catalog = load_anchor_catalog(source_paths, item_id)
                canonical_text = prompt_view_path(source_paths, item_id).read_text(encoding="utf-8", errors="replace")
                contexts, allowed = _build_context_from_canonical(
                    canonical_text=canonical_text,
                    catalog=catalog,
                    seed_anchor_ids=spread_anchors,
                    fill_gaps_up_to=12,
                    neighbor_pad=2,
                )
                cases.append(
                    CaseSpec(
                        case_id=f"real_{item_id}__spread",
                        item_id=item_id,
                        pass_kind="spread",
                        contexts=contexts,
                        allowed_anchor_ids=allowed,
                        task=(
                            f"Build ContractIR v0.2 for item_id={item_id}.\n"
                            "Extract ALL spread/margin schedules present in CONTEXT.\n"
                            "Do not encode benchmark/base-rate definitions or fees in this pass.\n"
                        ),
                        prompt_paths=[str(p) for p in spread_prompts],
                    )
                )

    if args.case_filter:
        cases = [c for c in cases if args.case_filter in c.case_id]
        if not cases:
            raise SystemExit(f"No cases matched --case-filter={args.case_filter!r}")

    # --- Execute cases -----------------------------------------------------------------
    run_summary: Dict[str, Any] = {
        "schema_version": "pricing_kernel_stress_v0",
        "run_id": out_run_id,
        "created_at": int(time.time()),
        "source_run_id": None if args.skip_real else str(args.source_run_id),
        "gateway_url": gateway_url,
        "model": str(args.model),
        "reasoning_effort": str(args.reasoning),
        "temperature": float(args.temperature),
        "attempts": int(args.attempts),
        "cases_total": len(cases),
        "cases": [],
        "counts_by_status": {},
    }

    counts: Dict[str, int] = {}
    # Persist progress as we go so an interrupted run still leaves a useful summary.json.
    _write_json(out_dir / "summary.json", run_summary)

    for case in cases:
        print(f"[case] {case.case_id} ({case.pass_kind})")
        case_dir = out_dir / case.case_id
        case_dir.mkdir(parents=True, exist_ok=True)

        case_report: Dict[str, Any] = {
            "case_id": case.case_id,
            "item_id": case.item_id,
            "pass_kind": case.pass_kind,
            "allowed_anchor_ids": case.allowed_anchor_ids,
            "prompt_attempts": [],
            "selected_prompt": None,
            "status": "invalid",
            "eval": None,
        }

        validated: Optional[dict] = None
        selected_prompt: Optional[str] = None
        selected_attempt: Optional[PromptAttemptResult] = None
        blocked_fallback: Optional[dict] = None
        blocked_fallback_prompt: Optional[str] = None
        blocked_fallback_attempt: Optional[PromptAttemptResult] = None

        for pi, prompt_path_str in enumerate(case.prompt_paths, start=1):
            print(f"  [prompt {pi}/{len(case.prompt_paths)}] {prompt_path_str}")
            prompt_path = PROJECT_ROOT / prompt_path_str
            attempt_dir = case_dir / f"prompt_{pi}"
            attempt_dir.mkdir(parents=True, exist_ok=True)

            doc, res = _run_prompt_with_repairs(
                client=client,
                out_dir=attempt_dir,
                item_id=case.item_id,
                prompt_path=prompt_path,
                task=case.task,
                contexts=case.contexts,
                allowed_anchor_ids=case.allowed_anchor_ids,
                pass_kind=case.pass_kind,
                attempts=int(args.attempts),
                gateway_url=gateway_url,
                timeout_seconds=float(args.timeout_seconds),
                temperature=float(args.temperature),
                model=str(args.model),
                reasoning=str(args.reasoning),
                max_output_tokens=int(args.max_output_tokens) if args.max_output_tokens else None,
            )

            case_report["prompt_attempts"].append({"prompt": prompt_path_str, **asdict(res)})

            if doc is None:
                continue

            # Spread pass: a validated output that only contains open_items is considered "blocked".
            # Keep it as a fallback, but continue trying other prompt strategies since another prompt
            # may be able to encode the schedule (e.g., a 2D grid requiring lookup_rule).
            if case.pass_kind == "spread":
                open_items = doc.get("open_items") if isinstance(doc, dict) else []
                open_items_count = len(open_items) if isinstance(open_items, list) else 0
                if open_items_count > 0:
                    if blocked_fallback is None:
                        blocked_fallback = doc
                        blocked_fallback_prompt = prompt_path_str
                        blocked_fallback_attempt = res
                    continue

            # Successful validation for this prompt.
            validated = doc
            selected_prompt = prompt_path_str
            selected_attempt = res
            break

        # If no prompt produced an unblocked extraction, but at least one produced a validated "blocked" doc,
        # keep the earliest such doc so the run folder has a canonical blocked output.
        if validated is None and blocked_fallback is not None:
            validated = blocked_fallback
            selected_prompt = blocked_fallback_prompt
            selected_attempt = blocked_fallback_attempt

        case_report["selected_prompt"] = selected_prompt

        if validated is None:
            case_report["status"] = "invalid"
            counts["invalid"] = counts.get("invalid", 0) + 1
            run_summary["cases"].append(case_report)
            run_summary["counts_by_status"] = counts
            _write_json(out_dir / "summary.json", run_summary)
            continue

        # Spread pass can be "blocked" per prompt policy (open_items => no derived/tables).
        open_items = validated.get("open_items") if isinstance(validated, dict) else []
        derived = validated.get("derived") if isinstance(validated, dict) else []
        tables = validated.get("tables") if isinstance(validated, dict) else []
        open_items_count = len(open_items) if isinstance(open_items, list) else 0
        derived_count = len(derived) if isinstance(derived, list) else 0
        tables_count = len(tables) if isinstance(tables, list) else 0

        if case.pass_kind == "spread" and open_items_count > 0:
            case_report["status"] = "blocked"
            counts["blocked"] = counts.get("blocked", 0) + 1
            run_summary["cases"].append(case_report)
            run_summary["counts_by_status"] = counts
            _write_json(out_dir / "summary.json", run_summary)
            continue

        # Consistency eval
        eval_report = _eval_contract_ir_consistency(validated)
        _write_json(case_dir / "eval_report.json", eval_report)
        case_report["eval"] = eval_report

        if not eval_report.get("ok", False):
            case_report["status"] = "eval_error"
            counts["eval_error"] = counts.get("eval_error", 0) + 1
        else:
            case_report["status"] = "ok"
            counts["ok"] = counts.get("ok", 0) + 1

        run_summary["cases"].append(case_report)
        run_summary["counts_by_status"] = counts
        _write_json(out_dir / "summary.json", run_summary)

    run_summary["counts_by_status"] = counts
    _write_json(out_dir / "summary.json", run_summary)
    print(f"[done] wrote {out_dir / 'summary.json'}")
    print(json.dumps({"run_id": out_run_id, "counts_by_status": counts, "cases_total": len(cases)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
