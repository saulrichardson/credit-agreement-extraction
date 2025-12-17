from __future__ import annotations

import asyncio
import json
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
    FlatPricingItem,
    PricingRegime,
    RateOption,
)
from .utils import assert_exists

# Reuse gateway helper/constants from indexing to avoid duplicating config.
from .indexing import (  # type: ignore
    _ensure_gateway_client_async,
    DEFAULT_GATEWAY_URL,
)


def _looks_like_pricing_table(table: dict[str, Any]) -> bool:
    raw = str(table.get("raw") or "")
    hay = raw.lower()
    if "%" in raw or "bps" in hay or "basis point" in hay:
        return True

    keywords = [
        "applicable margin",
        "applicable rate",
        "margin",
        "spread",
        "fee",
        "commitment fee",
        "facility fee",
        "l/c fee",
        "letter of credit",
        "fronting",
        "sofr",
        "libor",
        "eurocurrency",
        "base rate",
        "abr",
        "prime",
        "rate",
    ]
    return any(k in hay for k in keywords)


def _looks_like_pricing_anchor(text: str) -> bool:
    hay = text.lower()
    keywords = [
        "applicable margin",
        "applicable rate",
        "default rate",
        "fronting fee",
        "letter of credit fee",
        "letter of credit fees",
        "l/c fee rate",
        "commitment fee",
        "facility fee rate",
        "unused fee",
        "utilization",
        "basis point",
        "bps",
        "pricing grid",
        "sustainability grid",
        "sustainability metric grid",
    ]
    # Do NOT treat '%' alone as a pricing signal; that explodes context size on long exhibits.
    # We only include anchors when explicit pricing keywords are present.
    return any(k in hay for k in keywords)


def _looks_like_definition_anchor(text: str) -> bool:
    # Conservative: include only classic definitions we want nearby for pricing.
    hay = text.strip()
    if not hay:
        return False
    if " means " not in hay and " shall mean " not in hay:
        return False
    targets = [
        '"Base Rate"',
        '"ABR"',
        '"Eurocurrency Rate"',
        '"SOFR"',
        '"Term SOFR"',
        '"Adjusted Term SOFR Rate"',
        '"Adjusted CD Rate"',
        '"CD Base Rate"',
        '"London Interbank Offered Rate"',
        '"Applicable Margin"',
        '"Applicable Rate"',
        '"Facility Fee"',
        '"Commitment Fee"',
        '"L/C Fee',
    ]
    return any(t in hay for t in targets)


def _looks_like_regime_intro(text: str) -> bool:
    """Heuristic: lines that introduce a distinct pricing applicability regime."""

    hay = (text or "").strip().lower()
    if not hay:
        return False

    # Very common in amendments/agreements: bullet-like regime intros.
    if hay.startswith("-"):
        if any(k in hay for k in ["prior to", "on or after", "from and after", "until", "while", "during"]):
            return True
        if hay.startswith("- if "):
            return True

    # Common "alternate grid" intros.
    if "notwithstanding" in hay and any(k in hay for k in ["if ", "while ", "from and after", "on and after"]):
        return True

    return False


def _looks_like_adjustment_anchor(text: str) -> bool:
    """Heuristic: footnotes/clauses that adjust a grid by +/- X bps."""

    hay = (text or "").lower()
    if not hay:
        return False
    has_bps = any(k in hay for k in ["basis point", "basis points", " bps", " bp"])
    if not has_bps:
        return False
    return any(k in hay for k in ["reduced", "increased", "additional", "decreased", "increase", "reduce"])


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
    for t in tables:
        if not _looks_like_pricing_table(t):
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
                "reasons": ["pricing_table_match"],
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

                # 2) Build + persist context pack
                context = build_pricing_context(doc_ir)
                context_path = context_dir / f"{item_id}.json"
                context_path.write_text(json.dumps(context, indent=2))

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
