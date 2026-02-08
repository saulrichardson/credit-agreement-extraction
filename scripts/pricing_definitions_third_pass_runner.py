#!/usr/bin/env python3
"""
Run a pricing-metric definitions "third pass" over an agreement, using:
  - Metrics list extracted from a prior pricing (second-pass) JSON output
  - Definition text assembled from anchor spans (prefer indexing_v2.definitions_anchor_range)

This is intentionally artifact-first and run-scoped. It writes a folder per item_id:
  - definition_chunks.txt
  - metrics_list.json
  - prompt_rendered.txt
  - llm_output.txt
  - meta.json
  - validation.json
  - compiled_metrics/*.compiled.json
  - compiled_metrics_bundle.json

The goal is to make "pricing definitions" iterative + auditable without forcing the full
ContractIR/CovenantIR strict schemas.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence


class PricingDefinitionsThirdPassError(RuntimeError):
    pass


# Ensure repo root import works when invoked as `python scripts/...`.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.pipeline.core.anchors import load_anchor_catalog  # noqa: E402
from src.pipeline.core.config import REQUIRED_MODEL, REQUIRED_REASONING, Paths, prompt_hash  # noqa: E402
from src.pipeline.evidence.excerpt_packs import build_excerpt_pack_from_canonical, expand_anchor_ids  # noqa: E402
from src.pipeline.evidence.indexing import DEFAULT_GATEWAY_URL, _ensure_gateway_client_sync  # noqa: E402
from src.pipeline.compile.indexing_pricing_definitions_v1_schemas import (  # noqa: E402
    PricingDefinitionsIndexingArtifactV1,
    PricingDefinitionsIndexingSelectionV1,
)
from src.pipeline.schemas_v2 import IndexingSelectionV2Artifact  # noqa: E402
from src.pipeline.utils import assert_exists, prompt_view_path  # noqa: E402


@dataclass(frozen=True)
class StrategySpec:
    name: str
    description: str


@dataclass(frozen=True)
class CompustatAllowlistEntry:
    code: str
    description: str


STRATEGIES: dict[str, StrategySpec] = {
    "pricing_definitions_indexing_v1": StrategySpec(
        name="pricing_definitions_indexing_v1",
        description=(
            "Use runs/<run_id>/indexing_pricing_definitions_v1/<item_id>_pricing_definitions.json "
            "(LLM full-doc targeted indexing) as the source of definition anchors."
        ),
    ),
    "definition_finder_full_doc": StrategySpec(
        name="definition_finder_full_doc",
        description=(
            "Run a full-document definition finder inline (LLM anchor selector using prompts/indexing_pricing_definitions_v1.txt), "
            "then extract definitions using only the returned definition anchors (+deterministic completion)."
        ),
    ),
    "definitions_span": StrategySpec(
        name="definitions_span",
        description="Use indexing_v2.selection.definitions_anchor_range span (all anchors in span).",
    ),
    "definitions_span_filtered": StrategySpec(
        name="definitions_span_filtered",
        description="Use definitions span, but keep only anchors likely relevant to requested metrics (+neighbors).",
    ),
    "definitions_plus_pricing_union": StrategySpec(
        name="definitions_plus_pricing_union",
        description="Definitions span (filtered when possible) plus pricing_union anchors from indexing_v2 (gap-fill + pad).",
    ),
    "term_hits_plus_pricing_union": StrategySpec(
        name="term_hits_plus_pricing_union",
        description=(
            "Find likely definition anchors by searching the full document for metric terms (text hits), "
            "then merge with pricing_union anchors (gap-fill + pad). Works even when definitions_anchor_range is missing."
        ),
    ),
}


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _read_json(path: Path) -> Any:
    return json.loads(_read_text(path))


def _split_csv(values: Iterable[str]) -> list[str]:
    out: list[str] = []
    for v in values:
        if not v:
            continue
        parts = [p.strip() for p in str(v).split(",")]
        out.extend([p for p in parts if p])
    return out


def _load_compustat_allowlist(path: Path) -> list[CompustatAllowlistEntry]:
    try:
        doc = _read_json(path)
    except Exception as exc:
        raise PricingDefinitionsThirdPassError(f"Compustat allowlist is not valid JSON: {path} ({type(exc).__name__}: {exc})") from exc

    if not isinstance(doc, dict):
        raise PricingDefinitionsThirdPassError(f"Compustat allowlist must be a JSON object: {path}")
    if doc.get("schema_version") != "compustat_allowlist_v1":
        raise PricingDefinitionsThirdPassError(
            f"Compustat allowlist schema_version must be 'compustat_allowlist_v1' (got {doc.get('schema_version')!r}): {path}"
        )
    if doc.get("frequency") != "quarterly":
        raise PricingDefinitionsThirdPassError(
            f"Compustat allowlist frequency must be 'quarterly' (got {doc.get('frequency')!r}): {path}"
        )

    rows = doc.get("variables")
    if not isinstance(rows, list) or not rows:
        raise PricingDefinitionsThirdPassError(f"Compustat allowlist variables must be a non-empty list: {path}")

    entries: list[CompustatAllowlistEntry] = []
    for idx, row in enumerate(rows):
        if not isinstance(row, dict):
            raise PricingDefinitionsThirdPassError(f"Compustat allowlist variables[{idx}] must be object: {path}")
        code = row.get("code")
        description = row.get("description")
        if not isinstance(code, str) or not code.strip():
            raise PricingDefinitionsThirdPassError(f"Compustat allowlist variables[{idx}].code must be non-empty string: {path}")
        if not isinstance(description, str) or not description.strip():
            raise PricingDefinitionsThirdPassError(
                f"Compustat allowlist variables[{idx}].description must be non-empty string: {path}"
            )
        entries.append(CompustatAllowlistEntry(code=code.strip().lower(), description=description.strip()))

    deduped: dict[str, CompustatAllowlistEntry] = {}
    for e in entries:
        if e.code not in deduped:
            deduped[e.code] = e
    return [deduped[k] for k in sorted(deduped.keys())]


def _render_allowlist_table(allowlist: list[CompustatAllowlistEntry]) -> str:
    lines = [
        "| Code | Description |",
        "|------|-------------|",
    ]
    for e in allowlist:
        lines.append(f"| {e.code} | {e.description} |")
    return "\n".join(lines)


def _safe_slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "_", text or "").strip("_") or "metric"


def _load_pricing_second_pass_doc(pricing_dir: Path, item_id: str) -> dict[str, Any]:
    candidates = [
        pricing_dir / item_id / "llm_output.txt",
        pricing_dir / f"{item_id}.json",
        pricing_dir / f"{item_id}.txt",
        pricing_dir / item_id / f"{item_id}.json",
        pricing_dir / item_id / f"{item_id}.txt",
    ]
    for p in candidates:
        if p.exists():
            doc = json.loads(_read_text(p))
            if not isinstance(doc, dict):
                raise PricingDefinitionsThirdPassError(f"pricing second-pass output must be a JSON object: {p}")
            return doc
    raise PricingDefinitionsThirdPassError(
        f"Missing pricing second-pass output for {item_id} under {pricing_dir} "
        "(expected llm_output.txt or <item_id>.json/.txt)"
    )


def _camel_to_spaces(s: str) -> str:
    # Conservative camel splitter; keeps acronyms reasonably intact.
    s2 = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", s)
    s2 = s2.replace("_", " ")
    s2 = re.sub(r"\s+", " ", s2).strip()
    return s2


_TOKEN_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)


def _tokens(text: str) -> list[str]:
    return [m.group(0).lower() for m in _TOKEN_RE.finditer(text or "")]


def _metric_search_terms(metric_name: str) -> list[str]:
    # Build a small set of robust search terms that are likely to appear in contract definitions.
    spaced = _camel_to_spaces(metric_name)
    toks = [t for t in _tokens(spaced) if len(t) >= 4]
    # De-emphasize generic words but keep them if that's all we have.
    generic = {"ratio", "total", "consolidated", "minimum", "maximum"}
    strong = [t for t in toks if t not in generic]
    if strong:
        return strong[:6]
    return toks[:6]


def _extract_metric_names(pricing_doc: dict[str, Any]) -> list[str]:
    ordered: list[str] = []
    seen: set[str] = set()

    def _add(name: Any) -> None:
        if not isinstance(name, str):
            return
        n = name.strip()
        if not n:
            return
        if n in seen:
            return
        seen.add(n)
        ordered.append(n)

    # 1) Explicit metrics list.
    metrics = pricing_doc.get("metrics") or []
    if isinstance(metrics, list):
        for m in metrics:
            if isinstance(m, dict):
                _add(m.get("name"))

    # 2) Tier conditions (multi-metric tiers).
    tier_schemes = pricing_doc.get("tier_schemes") or []
    if isinstance(tier_schemes, list):
        for scheme in tier_schemes:
            if not isinstance(scheme, dict):
                continue
            _add(scheme.get("metric"))
            tiers = scheme.get("tiers") or []
            if not isinstance(tiers, list):
                continue
            for tier in tiers:
                if not isinstance(tier, dict):
                    continue
                cond = tier.get("conditions")
                if isinstance(cond, dict):
                    for key in ("all", "any"):
                        arr = cond.get(key)
                        if not isinstance(arr, list):
                            continue
                        for clause in arr:
                            if isinstance(clause, dict):
                                _add(clause.get("metric"))

    # 3) Rate-level conditions (rare but supported by prompt).
    facilities = pricing_doc.get("facilities") or []
    if isinstance(facilities, list):
        for fac in facilities:
            if not isinstance(fac, dict):
                continue
            rates = fac.get("rates") or []
            if not isinstance(rates, list):
                continue
            for rate in rates:
                if not isinstance(rate, dict):
                    continue
                cond = rate.get("condition")
                if isinstance(cond, dict):
                    _add(cond.get("metric"))

    # Drop obvious non-metrics.
    ordered = [m for m in ordered if m and m not in {"NA", "RatingScore", "RatingCategory", "RatingScoreMetric"}]
    return ordered


def _anchor_ids_between(*, catalog: dict[str, dict[str, Any]], start_anchor: str, end_anchor: str) -> list[str]:
    if start_anchor not in catalog:
        raise PricingDefinitionsThirdPassError(f"definitions span start_anchor not found in anchors.tsv: {start_anchor}")
    if end_anchor not in catalog:
        raise PricingDefinitionsThirdPassError(f"definitions span end_anchor not found in anchors.tsv: {end_anchor}")

    start_order = int(catalog[start_anchor]["order"])
    end_order = int(catalog[end_anchor]["order"])
    if start_order > end_order:
        raise PricingDefinitionsThirdPassError(
            f"definitions span start_anchor after end_anchor: {start_anchor} (order={start_order}) > {end_anchor} (order={end_order})"
        )

    ordered = sorted(catalog.values(), key=lambda a: int(a["order"]))
    return [str(info["anchor_id"]) for info in ordered[start_order : end_order + 1]]


def _definitions_anchor_ids_from_indexing(paths: Paths, item_id: str, catalog: dict[str, dict[str, Any]]) -> list[str]:
    sel_path = paths.run_dir / "indexing_v2" / f"{item_id}_anchors.json"
    if not sel_path.exists():
        return []
    sel_doc = _read_json(sel_path)
    sel_art = IndexingSelectionV2Artifact.model_validate(sel_doc)
    dr = sel_art.selection.definitions_anchor_range
    if dr is None:
        return []
    return _anchor_ids_between(catalog=catalog, start_anchor=dr.start_anchor, end_anchor=dr.end_anchor)


def _definitions_anchor_ids_from_pricing_definitions_indexing_v1(
    paths: Paths,
    item_id: str,
    *,
    metric_names: Sequence[str],
    catalog: dict[str, dict[str, Any]],
) -> list[str]:
    """Load targeted definitions indexing output and return the union of selected anchors in doc order."""

    artifact_path = paths.run_dir / "indexing_pricing_definitions_v1" / f"{item_id}_pricing_definitions.json"
    if not artifact_path.exists():
        raise PricingDefinitionsThirdPassError(
            "Missing pricing definitions indexing artifact. Run:\n"
            f"  python scripts/pricing_definitions_indexing_v1_runner.py --run-id {paths.run_id} "
            f"--pricing-second-pass-dir <dir> --all\n"
            f"Expected: {artifact_path}"
        )

    try:
        artifact_doc = _read_json(artifact_path)
    except Exception as exc:
        raise PricingDefinitionsThirdPassError(f"Failed to read JSON: {artifact_path} ({type(exc).__name__}: {exc})") from exc

    artifact = PricingDefinitionsIndexingArtifactV1.model_validate(artifact_doc)
    selection = artifact.selection

    by_name = {m.name: m for m in selection.metrics}
    missing = [n for n in metric_names if n not in by_name]
    extra = [n for n in by_name.keys() if n not in metric_names]
    if missing or extra:
        raise PricingDefinitionsThirdPassError(
            f"pricing definitions indexing metric mismatch for {item_id}: "
            f"missing={missing[:6]} extra={extra[:6]} (path={artifact_path})"
        )

    merged: list[str] = []
    for n in metric_names:
        merged.extend(list(by_name[n].definition_anchor_ids or []))

    # Dedup while preserving doc order.
    order = {aid: int(catalog[aid]["order"]) for aid in catalog}
    merged = list(dict.fromkeys(merged))
    return sorted(merged, key=lambda a: order.get(a, 10**9))


def _run_definition_finder_full_doc(
    *,
    complete: Any,
    prompt_template: str,
    metric_names: Sequence[str],
    canonical_text: str,
    catalog: dict[str, dict[str, Any]],
    model: str,
    reasoning: str,
    temperature: float,
    gateway_url: str,
    timeout_seconds: float,
    attempts: int,
) -> tuple[list[str], PricingDefinitionsIndexingSelectionV1, str, str]:
    """Run the full-document definition finder and return (seed_anchor_ids, selection, rendered_prompt, raw_text)."""

    template = prompt_template.strip()
    missing: list[str] = []
    for key in ("{metrics_json}", "{document}"):
        if key not in template:
            missing.append(key)
    if missing:
        raise PricingDefinitionsThirdPassError(
            f"definition finder prompt missing required placeholders: {', '.join(missing)}"
        )

    ordered = sorted(catalog.values(), key=lambda a: int(a["order"]))
    all_anchor_ids = [str(info["anchor_id"]) for info in ordered]
    document = build_excerpt_pack_from_canonical(
        canonical_text=canonical_text,
        catalog=catalog,
        anchor_ids=all_anchor_ids,
    )

    metrics_json = json.dumps(
        [{"name": n, "description": None} for n in metric_names],
        indent=2,
        sort_keys=True,
    )
    rendered = template.replace("{metrics_json}", metrics_json).replace("{document}", document)

    last_error: str | None = None
    for attempt in range(1, max(1, int(attempts)) + 1):
        try:
            raw_text = complete(
                model=model,
                prompt=rendered,
                base_url=gateway_url,
                reasoning={"effort": reasoning} if reasoning else None,
                temperature=float(temperature),
                max_output_tokens=None,
                timeout=float(timeout_seconds),
            )
        except Exception as exc:  # pragma: no cover - network
            last_error = f"{type(exc).__name__}: {exc}"
            continue

        if not isinstance(raw_text, str) or not raw_text.strip():
            last_error = "gateway returned empty text"
            continue

        try:
            payload = json.loads(raw_text)
        except Exception as exc:
            last_error = f"invalid JSON: {type(exc).__name__}: {exc}"
            continue

        try:
            selection = PricingDefinitionsIndexingSelectionV1.model_validate(payload)
        except Exception as exc:
            last_error = f"schema validation failed: {type(exc).__name__}: {exc}"
            continue

        out_names = [m.name for m in selection.metrics]
        missing_names = [n for n in metric_names if n not in out_names]
        extra_names = [n for n in out_names if n not in metric_names]
        if missing_names or extra_names:
            last_error = f"metric coverage mismatch (missing={missing_names[:6]} extra={extra_names[:6]})"
            continue

        invalid: list[str] = []
        for rec in selection.metrics:
            for aid in rec.definition_anchor_ids:
                if aid not in catalog:
                    invalid.append(aid)
        if invalid:
            last_error = f"returned anchor_id values not present in anchors.tsv (examples={sorted(set(invalid))[:10]})"
            continue

        merged: list[str] = []
        by_name = {m.name: m for m in selection.metrics}
        for n in metric_names:
            merged.extend(list(by_name[n].definition_anchor_ids or []))
        # Stable de-dupe in doc order.
        merged = list(dict.fromkeys(merged))
        order_map = {aid: int(catalog[aid]["order"]) for aid in catalog}
        seeds = sorted(merged, key=lambda a: order_map.get(a, 10**9))
        return (seeds, selection, rendered, raw_text)

    raise PricingDefinitionsThirdPassError(
        f"definition finder failed after {attempts} attempt(s): {last_error or 'unknown'}"
    )


def _find_term_hits(text: str, term: str) -> list[tuple[int, int]]:
    """Return case-insensitive matches for a term, tolerant to whitespace differences."""
    parts = [p for p in re.split(r"\s+", (term or "").strip()) if p]
    if not parts:
        return []
    pattern = r"\s+".join(re.escape(p) for p in parts)
    return [(m.start(), m.end()) for m in re.finditer(pattern, text, flags=re.IGNORECASE)]


def _match_hits_to_anchors(catalog: dict[str, dict[str, Any]], hits: list[tuple[int, int]]) -> list[str]:
    ordered = sorted(catalog.values(), key=lambda a: int(a["start"]))
    anchor_ids: list[str] = []
    for start, _end in hits:
        for info in ordered:
            if int(info["start"]) <= start < int(info["end"]):
                anchor_ids.append(str(info["anchor_id"]))
                break
    # stable de-dupe
    seen: set[str] = set()
    unique: list[str] = []
    for aid in anchor_ids:
        if aid in seen:
            continue
        seen.add(aid)
        unique.append(aid)
    return unique


def _anchor_neighbor_pad(catalog: dict[str, dict[str, Any]], anchor_ids: list[str], *, neighbor_pad: int) -> list[str]:
    if not anchor_ids:
        return []
    order = {aid: int(catalog[aid]["order"]) for aid in catalog}
    ordered_all = sorted(order.items(), key=lambda kv: kv[1])
    by_order = [aid for aid, _ in ordered_all]
    idx = {aid: i for i, aid in enumerate(by_order)}
    out_set: set[str] = set()
    for aid in anchor_ids:
        i = idx.get(aid)
        if i is None:
            continue
        for j in range(max(0, i - neighbor_pad), min(len(by_order), i + neighbor_pad + 1)):
            out_set.add(by_order[j])
    return [aid for aid in by_order if aid in out_set]


def _term_hit_anchor_ids_anywhere(
    *,
    canonical_text: str,
    catalog: dict[str, dict[str, Any]],
    metric_names: Sequence[str],
    neighbor_pad: int = 2,
    max_anchors: int = 120,
) -> list[str]:
    hit_anchor_ids: list[str] = []
    for name in metric_names:
        # Prefer searching the "spaced" form, which resembles contract terms.
        term = _camel_to_spaces(name)
        hits = _find_term_hits(canonical_text, term)
        hit_anchor_ids.extend(_match_hits_to_anchors(catalog, hits))

    # Stable de-dupe while preserving order of appearance in doc.
    seen: set[str] = set()
    deduped: list[str] = []
    for aid in hit_anchor_ids:
        if aid in seen:
            continue
        seen.add(aid)
        deduped.append(aid)

    padded = _anchor_neighbor_pad(catalog, deduped, neighbor_pad=neighbor_pad)
    return padded


def _definition_like_score(text: str) -> int:
    hay = (text or "").lower()
    score = 0
    if "means" in hay:
        score += 2
    if "shall mean" in hay:
        score += 6
    if "is defined as" in hay:
        score += 6
    if hay.strip().startswith(("definitions", "defined terms", "section 1")):
        score += 1
    return score


def _filter_definitions_span(
    *,
    canonical_text: str,
    catalog: dict[str, dict[str, Any]],
    span_anchor_ids: list[str],
    metric_names: Sequence[str],
    neighbor_pad: int = 1,
    max_anchors: int = 120,
) -> list[str]:
    if not span_anchor_ids:
        return []

    # Build a compact term set from metric names.
    term_tokens: set[str] = set()
    for name in metric_names:
        for t in _metric_search_terms(name):
            term_tokens.add(t)

    def _anchor_text(aid: str) -> str:
        info = catalog.get(aid)
        if not info:
            return ""
        start = int(info["start"])
        end = int(info["end"])
        return canonical_text[start:end]

    # Select anchors that look definition-like AND/or mention at least one metric token.
    picked: list[str] = []
    for aid in span_anchor_ids:
        text = _anchor_text(aid)
        hay_tokens = set(_tokens(text))
        tok_hit = bool(term_tokens and (term_tokens & hay_tokens))
        def_hit = _definition_like_score(text) >= 2
        if tok_hit or def_hit:
            picked.append(aid)

    # If we got nothing, fall back to the full span (we don't want to silently drop context).
    if not picked:
        return span_anchor_ids

    # Expand by neighbors within the span.
    order_map = {aid: int(catalog[aid]["order"]) for aid in span_anchor_ids if aid in catalog}
    ordered_span = sorted(span_anchor_ids, key=lambda a: order_map.get(a, 10**9))
    idx_map = {aid: i for i, aid in enumerate(ordered_span)}
    out_set: set[str] = set()
    for aid in picked:
        i = idx_map.get(aid)
        if i is None:
            continue
        for j in range(max(0, i - neighbor_pad), min(len(ordered_span), i + neighbor_pad + 1)):
            out_set.add(ordered_span[j])

    out = [aid for aid in ordered_span if aid in out_set]
    return out


def _pricing_union_seeds(selection: Any) -> list[str]:
    merged: list[str] = []
    for arr in (
        selection.pricing_anchors or [],
        selection.base_rate_anchors or [],
        selection.spread_anchors or [],
        selection.fee_anchors or [],
    ):
        merged.extend(list(arr))
    # Preserve order but de-dupe.
    seen: set[str] = set()
    deduped: list[str] = []
    for a in merged:
        if a in seen:
            continue
        seen.add(a)
        deduped.append(a)
    return deduped


def _render_prompt(
    template: str,
    metrics_list: list[str],
    definition_chunks: str,
    *,
    compustat_allowlist_table: str,
) -> str:
    required = ("{metrics_list}", "{definition_chunks}", "{compustat_allowlist_table}")
    if any(k not in template for k in required):
        raise PricingDefinitionsThirdPassError(
            "Prompt template must contain placeholders: "
            "{metrics_list}, {definition_chunks}, {compustat_allowlist_table}"
        )
    return (
        template.replace("{metrics_list}", json.dumps(metrics_list, indent=2))
        .replace("{definition_chunks}", definition_chunks)
        .replace("{compustat_allowlist_table}", compustat_allowlist_table)
        .strip()
    )


ANCHOR_RE = re.compile(r"\[\[(A\d{4})\]\]")


def _extract_anchor_ids(excerpt: str) -> set[str]:
    return set(ANCHOR_RE.findall(excerpt or ""))


def _validate_output(doc: Any, metrics_expected: Sequence[str], *, anchors_present: set[str]) -> dict[str, Any]:
    out: dict[str, Any] = {"json_ok": False}
    if not isinstance(doc, dict):
        out["error"] = f"expected top-level object, got {type(doc).__name__}"
        return out
    metrics = doc.get("metrics")
    if not isinstance(metrics, list):
        out["error"] = "missing/invalid metrics list"
        return out
    out["json_ok"] = True
    out["metrics_expected"] = list(metrics_expected)
    out["metrics_returned_count"] = len(metrics)

    by_name: dict[str, dict[str, Any]] = {}
    for m in metrics:
        if isinstance(m, dict) and isinstance(m.get("name"), str):
            by_name[m["name"]] = m

    missing = [n for n in metrics_expected if n not in by_name]
    out["missing_metrics"] = missing

    defined = 0
    defined_with_refs = 0
    unknown_refs: set[str] = set()
    for n in metrics_expected:
        rec = by_name.get(n)
        if not rec:
            continue
        if rec.get("definition_verbatim"):
            defined += 1
            srefs = rec.get("source_refs")
            if isinstance(srefs, list):
                cleaned = [x.strip() for x in srefs if isinstance(x, str) and x.strip()]
                for x in cleaned:
                    if x not in anchors_present:
                        unknown_refs.add(x)
                if cleaned and not any(x not in anchors_present for x in cleaned):
                    defined_with_refs += 1
    out["metrics_with_definition_verbatim"] = defined
    out["metrics_with_definition_and_refs"] = defined_with_refs
    out["unknown_source_refs"] = sorted(unknown_refs)
    return out


def _unresolved_for_metric(unresolved: Any, metric_name: str) -> list[dict[str, Any]]:
    if not isinstance(unresolved, list):
        return []
    out: list[dict[str, Any]] = []
    seen: set[tuple[str, tuple[str, ...]]] = set()
    for row in unresolved:
        if not isinstance(row, dict):
            continue
        term = row.get("term")
        refs = row.get("referenced_by")
        if not isinstance(term, str) or not term.strip():
            continue
        if not isinstance(refs, list) or not all(isinstance(x, str) for x in refs):
            continue
        refs_clean = [x.strip() for x in refs if x.strip()]
        if metric_name not in refs_clean:
            continue
        key = (term.strip(), tuple(sorted(set(refs_clean))))
        if key in seen:
            continue
        seen.add(key)
        out.append({"term": term.strip(), "referenced_by": sorted(set(refs_clean))})
    return out


def _write_per_metric_compiled_jsons(item_out: Path, parsed_doc: dict[str, Any]) -> dict[str, Any]:
    metrics = parsed_doc.get("metrics")
    if not isinstance(metrics, list):
        return {"written_count": 0, "paths": []}

    unresolved = parsed_doc.get("unresolved_dependencies")
    compiled_dir = item_out / "compiled_metrics"
    if compiled_dir.exists():
        shutil.rmtree(compiled_dir)
    compiled_dir.mkdir(parents=True, exist_ok=True)

    written_paths: list[str] = []
    bundle_metrics: list[dict[str, Any]] = []
    seen_names: set[str] = set()

    for row in metrics:
        if not isinstance(row, dict):
            continue
        name = row.get("name")
        if not isinstance(name, str) or not name.strip():
            continue
        metric_name = name.strip()
        if metric_name in seen_names:
            continue
        seen_names.add(metric_name)

        metric_payload = dict(row)
        metric_payload["name"] = metric_name
        unresolved_metric = _unresolved_for_metric(unresolved, metric_name)
        compiled_doc = {
            "schema_version": "pricing_metric_definition_compiled_v1",
            "metric_name": metric_name,
            "metric": metric_payload,
            "unresolved_dependencies": unresolved_metric,
        }

        out_path = compiled_dir / f"{_safe_slug(metric_name)}.compiled.json"
        _write_json(out_path, compiled_doc)
        written_paths.append(str(out_path))
        bundle_metrics.append(compiled_doc)

    _write_json(
        item_out / "compiled_metrics_bundle.json",
        {
            "schema_version": "pricing_metric_definition_compiled_bundle_v1",
            "metrics": bundle_metrics,
        },
    )
    return {"written_count": len(bundle_metrics), "paths": written_paths}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True, help="Existing run folder under runs/<run-id>/ with normalized/ + indexing_v2/.")
    ap.add_argument("--pricing-second-pass-dir", required=True, help="Directory containing pricing second-pass outputs per item.")
    ap.add_argument("--prompt", default="prompts/prompt_pricing_third_pass_dg_v2.txt")
    ap.add_argument(
        "--compustat-allowlist",
        default="datasets/compustat_allowlist_quarterly_v1.json",
        help="Compustat allowlist JSON injected into the one-step prompt as a table.",
    )
    ap.add_argument(
        "--definition-finder-prompt",
        default="prompts/indexing_pricing_definitions_v1.txt",
        help="Prompt used by the full-document definition finder (strategy=definition_finder_full_doc).",
    )
    ap.add_argument(
        "--strategy",
        action="append",
        default=[],
        choices=sorted(STRATEGIES.keys()),
        help="Context strategy to run (repeatable). Defaults to definitions_span.",
    )
    ap.add_argument(
        "--auto",
        action="store_true",
        help=(
            "Auto-select strategies per item (minimize calls): "
            "if definitions span exists: try definitions_span then fallback to definitions_plus_pricing_union then definition_finder_full_doc; "
            "else: try definitions_plus_pricing_union then fallback to term_hits_plus_pricing_union then definition_finder_full_doc."
        ),
    )
    ap.add_argument("--all", action="store_true", help="Run for all item_ids present under runs/<run-id>/normalized/.")
    ap.add_argument("--item-id", action="append", default=[], help="Item ID(s) to run (repeatable; supports comma-separated).")
    ap.add_argument("--out-dir", required=True, help="Output directory.")
    ap.add_argument("--model", default=REQUIRED_MODEL)
    ap.add_argument("--reasoning", default=REQUIRED_REASONING)
    ap.add_argument("--gateway-url", default=None)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--attempts", type=int, default=3)
    ap.add_argument("--timeout-seconds", type=float, default=600.0)
    ap.add_argument("--fill-gaps-up-to", type=int, default=12)
    ap.add_argument("--neighbor-pad", type=int, default=2)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    strategies = list(dict.fromkeys(args.strategy or ["definitions_span"]))
    if args.auto:
        strategies = []

    run_id = str(args.run_id)
    paths = Paths(root=Path("."), run_id=run_id)
    if not paths.run_dir.exists():
        raise SystemExit(f"Run dir not found: {paths.run_dir}")

    if args.all and args.item_id:
        raise SystemExit("Use either --all or --item-id, not both.")

    if args.all:
        normalized_dir = paths.run_dir / "normalized"
        if not normalized_dir.exists():
            raise SystemExit(f"normalized dir not found: {normalized_dir}")
        item_ids = sorted([p.name for p in normalized_dir.iterdir() if p.is_dir()])
        if not item_ids:
            raise SystemExit(f"No items found under: {normalized_dir}")
    else:
        item_ids = _split_csv(args.item_id)
        if not item_ids:
            raise SystemExit("Provide at least one --item-id (repeatable; comma-separated supported) or pass --all.")

    pricing_second_pass_dir = Path(args.pricing_second_pass_dir)
    if not pricing_second_pass_dir.exists():
        raise SystemExit(f"pricing-second-pass-dir not found: {pricing_second_pass_dir}")

    prompt_path = Path(args.prompt)
    assert_exists(prompt_path, message=f"Prompt not found: {prompt_path}")
    prompt_template = _read_text(prompt_path)
    prompt_digest = prompt_hash(prompt_path)

    compustat_allowlist_path = Path(args.compustat_allowlist)
    assert_exists(compustat_allowlist_path, message=f"Compustat allowlist not found: {compustat_allowlist_path}")
    allowlist = _load_compustat_allowlist(compustat_allowlist_path)
    compustat_allowlist_table = _render_allowlist_table(allowlist)
    compustat_allowlist_digest = prompt_hash(compustat_allowlist_path)

    definition_finder_prompt_path = Path(args.definition_finder_prompt)
    assert_exists(definition_finder_prompt_path, message=f"Definition-finder prompt not found: {definition_finder_prompt_path}")
    definition_finder_template = _read_text(definition_finder_prompt_path)
    definition_finder_prompt_digest = prompt_hash(definition_finder_prompt_path)

    out_dir = Path(args.out_dir)
    if out_dir.exists():
        if not args.overwrite:
            raise SystemExit(f"out-dir already exists: {out_dir} (pass --overwrite to delete)")
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Copy prompt for reproducibility.
    _write_text(out_dir / prompt_path.name, prompt_template)
    _write_text(out_dir / definition_finder_prompt_path.name, definition_finder_template)

    complete = _ensure_gateway_client_sync()
    gateway_url = args.gateway_url or DEFAULT_GATEWAY_URL

    report: dict[str, Any] = {
        "schema_version": "pricing_definitions_third_pass_runner_v1",
        "created_at": int(time.time()),
        "run_id": run_id,
        "pricing_second_pass_dir": str(pricing_second_pass_dir),
        "prompt": str(prompt_path),
        "prompt_sha256": prompt_digest,
        "definition_finder_prompt": str(definition_finder_prompt_path),
        "definition_finder_prompt_sha256": definition_finder_prompt_digest,
        "compustat_allowlist": str(compustat_allowlist_path),
        "compustat_allowlist_sha256": compustat_allowlist_digest,
        "strategies": strategies,
        "llm": {
            "model": args.model,
            "reasoning_effort": args.reasoning,
            "temperature": float(args.temperature),
            "gateway_url": gateway_url,
            "timeout_seconds": float(args.timeout_seconds),
            "attempts": int(args.attempts),
        },
        "items": [],
    }

    if args.auto:
        (out_dir / "attempts").mkdir(parents=True, exist_ok=True)
        (out_dir / "best").mkdir(parents=True, exist_ok=True)
    else:
        for strategy in strategies:
            (out_dir / strategy).mkdir(parents=True, exist_ok=True)

    for item_id in item_ids:
        started = time.time()
        pricing_doc = _load_pricing_second_pass_doc(pricing_second_pass_dir, item_id)
        metric_names = _extract_metric_names(pricing_doc)

        canonical_text = _read_text(prompt_view_path(paths, item_id))
        catalog = load_anchor_catalog(paths, item_id)

        sel_path = assert_exists(
            paths.run_dir / "indexing_v2" / f"{item_id}_anchors.json",
            message=f"Missing indexing_v2 anchors JSON for {item_id}: {paths.run_dir}/indexing_v2/{item_id}_anchors.json",
        )
        sel_art = IndexingSelectionV2Artifact.model_validate(_read_json(sel_path))
        selection = sel_art.selection

        span_anchor_ids = _definitions_anchor_ids_from_indexing(paths, item_id, catalog)

        per_strategy: dict[str, Any] = {}
        best: dict[str, Any] | None = None
        best_strategy: str | None = None

        if args.auto:
            if span_anchor_ids:
                run_strategies = ["definitions_span", "definitions_plus_pricing_union", "definition_finder_full_doc"]
            else:
                run_strategies = ["definitions_plus_pricing_union", "term_hits_plus_pricing_union", "definition_finder_full_doc"]
        else:
            run_strategies = strategies

        for strategy in run_strategies:
            item_out = (out_dir / "attempts" / strategy / item_id) if args.auto else (out_dir / strategy / item_id)
            item_out.mkdir(parents=True, exist_ok=True)

            # Build definition chunks according to strategy.
            if strategy == "pricing_definitions_indexing_v1":
                seeds = _definitions_anchor_ids_from_pricing_definitions_indexing_v1(
                    paths,
                    item_id,
                    metric_names=metric_names,
                    catalog=catalog,
                )
                defs_anchor_ids = (
                    expand_anchor_ids(
                        catalog=catalog,
                        seed_anchor_ids=seeds,
                        fill_gaps_up_to=int(args.fill_gaps_up_to),
                        neighbor_pad=int(args.neighbor_pad),
                    )
                    if seeds
                    else []
                )
            elif strategy == "definition_finder_full_doc":
                if not metric_names:
                    defs_anchor_ids = []
                else:
                    seeds, selection_obj, _finder_rendered, finder_raw = _run_definition_finder_full_doc(
                        complete=complete,
                        prompt_template=definition_finder_template,
                        metric_names=metric_names,
                        canonical_text=canonical_text,
                        catalog=catalog,
                        model=args.model,
                        reasoning=args.reasoning,
                        temperature=float(args.temperature),
                        gateway_url=gateway_url,
                        timeout_seconds=float(args.timeout_seconds),
                        attempts=int(args.attempts),
                    )
                    _write_text(item_out / "definition_finder_llm_output.txt", finder_raw)
                    _write_json(item_out / "definition_finder_selection.json", selection_obj.model_dump())
                    defs_anchor_ids = (
                        expand_anchor_ids(
                            catalog=catalog,
                            seed_anchor_ids=seeds,
                            fill_gaps_up_to=int(args.fill_gaps_up_to),
                            neighbor_pad=int(args.neighbor_pad),
                        )
                        if seeds
                        else []
                    )
            elif strategy == "definitions_span":
                defs_anchor_ids = span_anchor_ids
            elif strategy == "definitions_span_filtered":
                defs_anchor_ids = _filter_definitions_span(
                    canonical_text=canonical_text,
                    catalog=catalog,
                    span_anchor_ids=span_anchor_ids,
                    metric_names=metric_names,
                    neighbor_pad=1,
                    max_anchors=140,
                )
            elif strategy == "definitions_plus_pricing_union":
                defs_base = _filter_definitions_span(
                    canonical_text=canonical_text,
                    catalog=catalog,
                    span_anchor_ids=span_anchor_ids,
                    metric_names=metric_names,
                    neighbor_pad=1,
                    max_anchors=120,
                )
                seeds = _pricing_union_seeds(selection)
                if seeds:
                    expanded = expand_anchor_ids(
                        catalog=catalog,
                        seed_anchor_ids=seeds,
                        fill_gaps_up_to=int(args.fill_gaps_up_to),
                        neighbor_pad=int(args.neighbor_pad),
                    )
                else:
                    expanded = []
                # Merge in doc order.
                order = {aid: int(catalog[aid]["order"]) for aid in catalog}
                merged = list(dict.fromkeys(defs_base + expanded))
                defs_anchor_ids = sorted(merged, key=lambda a: order.get(a, 10**9))
            elif strategy == "term_hits_plus_pricing_union":
                hit_ids = _term_hit_anchor_ids_anywhere(
                    canonical_text=canonical_text,
                    catalog=catalog,
                    metric_names=metric_names,
                    neighbor_pad=2,
                    max_anchors=140,
                )
                seeds = _pricing_union_seeds(selection)
                expanded = (
                    expand_anchor_ids(
                        catalog=catalog,
                        seed_anchor_ids=seeds,
                        fill_gaps_up_to=int(args.fill_gaps_up_to),
                        neighbor_pad=int(args.neighbor_pad),
                    )
                    if seeds
                    else []
                )
                order = {aid: int(catalog[aid]["order"]) for aid in catalog}
                merged = list(dict.fromkeys(hit_ids + expanded))
                defs_anchor_ids = sorted(merged, key=lambda a: order.get(a, 10**9))
            else:
                raise PricingDefinitionsThirdPassError(f"Unknown strategy: {strategy}")

            if not defs_anchor_ids:
                _write_text(item_out / "error.txt", f"No definition anchors available for strategy={strategy}\n")
                per_strategy[strategy] = {"status": "error", "error": "no_definition_anchors"}
                continue

            definition_chunks = build_excerpt_pack_from_canonical(
                canonical_text=canonical_text,
                catalog=catalog,
                anchor_ids=defs_anchor_ids,
            )
            anchors_present = _extract_anchor_ids(definition_chunks)

            rendered = _render_prompt(
                prompt_template,
                metric_names,
                definition_chunks,
                compustat_allowlist_table=compustat_allowlist_table,
            )

            _write_text(item_out / "definition_chunks.txt", definition_chunks)
            _write_json(item_out / "metrics_list.json", metric_names)
            _write_text(item_out / "prompt_rendered.txt", rendered)
            _write_json(
                item_out / "meta.json",
                {
                    "item_id": item_id,
                    "strategy": strategy,
                    "strategy_description": STRATEGIES[strategy].description,
                    "metrics_count": len(metric_names),
                    "definition_anchor_ids_count": len(defs_anchor_ids),
                    "definition_anchor_ids_preview": defs_anchor_ids[:25],
                    "anchors_present_count": len(anchors_present),
                    "definitions_span_present": bool(span_anchor_ids),
                    "indexing_v2_path": str(sel_path),
                    "model": args.model,
                    "reasoning": args.reasoning,
                    "temperature": float(args.temperature),
                    "gateway_url": gateway_url,
                    "prompt": str(prompt_path),
                    "prompt_sha256": prompt_digest,
                    "definition_finder_prompt": str(definition_finder_prompt_path),
                    "definition_finder_prompt_sha256": definition_finder_prompt_digest,
                    "compustat_allowlist": str(compustat_allowlist_path),
                    "compustat_allowlist_sha256": compustat_allowlist_digest,
                },
            )

            last_err: str | None = None
            output_text: str | None = None
            for _attempt in range(1, max(1, int(args.attempts)) + 1):
                try:
                    output_text = complete(
                        model=args.model,
                        prompt=rendered,
                        base_url=gateway_url,
                        reasoning={"effort": args.reasoning} if args.reasoning else None,
                        temperature=float(args.temperature),
                        max_output_tokens=None,
                        timeout=float(args.timeout_seconds),
                    )
                    if isinstance(output_text, str) and output_text.strip():
                        break
                    last_err = "gateway returned empty text"
                    output_text = None
                except Exception as exc:  # pragma: no cover - network
                    last_err = f"{type(exc).__name__}: {exc}"
                    output_text = None

            if output_text is None or not output_text.strip():
                _write_text(item_out / "error.txt", f"{last_err or 'gateway_failed'}\n")
                per_strategy[strategy] = {"status": "error", "error": last_err or "gateway_failed"}
                continue

            _write_text(item_out / "llm_output.txt", output_text)

            try:
                parsed = json.loads(output_text)
            except Exception as exc:
                _write_json(
                    item_out / "validation.json",
                    {"json_ok": False, "json_error": f"{type(exc).__name__}: {exc}"},
                )
                per_strategy[strategy] = {"status": "invalid_json"}
                continue

            validation = _validate_output(parsed, metric_names, anchors_present=anchors_present)
            _write_json(item_out / "validation.json", validation)
            per_metric = _write_per_metric_compiled_jsons(item_out, parsed)
            _write_json(item_out / "compiled_metrics_index.json", per_metric)
            per_strategy[strategy] = {"status": "ok", **validation}

            score = int(validation.get("metrics_with_definition_and_refs") or 0)
            if best is None or score > int(best.get("metrics_with_definition_and_refs") or 0):
                best = {"strategy": strategy, "score": score, "validation": validation, "item_out": str(item_out)}
                best_strategy = strategy

            # Early exit: perfect coverage.
            if args.auto and metric_names and score >= len(metric_names):
                break

        report["items"].append(
            {
                "item_id": item_id,
                "secs": round(time.time() - started, 3),
                "metrics_count": len(metric_names),
                "per_strategy": per_strategy,
                "best": best,
            }
        )

        if args.auto and best_strategy:
            # Copy best artifacts into out_dir/best/<item_id>/ for easy browsing.
            src_dir = Path(str(best.get("item_out"))) if isinstance(best, dict) and best.get("item_out") else None
            if src_dir and src_dir.exists():
                best_dir = out_dir / "best" / item_id
                best_dir.mkdir(parents=True, exist_ok=True)
                # Copy a small, stable set of artifacts.
                for fname in (
                    "definition_chunks.txt",
                    "metrics_list.json",
                    "prompt_rendered.txt",
                    "llm_output.txt",
                    "meta.json",
                    "validation.json",
                ):
                    src_path = src_dir / fname
                    if src_path.exists():
                        shutil.copy2(src_path, best_dir / fname)
                _write_text(best_dir / "best_strategy.txt", f"{best_strategy}\n")

    _write_json(out_dir / "report.json", report)
    print(f"[done] wrote {out_dir} (items={len(item_ids)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
