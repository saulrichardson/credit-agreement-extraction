from __future__ import annotations

import asyncio
import json
import re
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from .config import Paths, REQUIRED_MODEL, REQUIRED_REASONING, prompt_hash, update_manifest
from .llm.gateway import DEFAULT_GATEWAY_URL, _ensure_gateway_client_async
from .utils import assert_exists


@dataclass(frozen=True)
class OverlayTarget:
    item_id: str
    term: str
    definition_verbatim: str | None
    expression_ast: dict[str, Any] | None
    input_terms: list[str]
    clauses: list[dict[str, Any]]


def _safe_slug(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "_", s).strip("_") or "term"


@dataclass(frozen=True)
class CompustatAllowlistEntry:
    code: str
    description: str


def _load_compustat_allowlist(path: Path) -> list[CompustatAllowlistEntry]:
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Compustat allowlist is not valid JSON: {path}") from exc

    if not isinstance(doc, dict):
        raise RuntimeError(f"Compustat allowlist must be a JSON object: {path}")
    if doc.get("schema_version") != "compustat_allowlist_v1":
        raise RuntimeError(
            f"Compustat allowlist schema_version must be 'compustat_allowlist_v1' (got {doc.get('schema_version')!r}): {path}"
        )
    if doc.get("frequency") != "quarterly":
        raise RuntimeError(
            f"Compustat allowlist frequency must be 'quarterly' (got {doc.get('frequency')!r}): {path}"
        )

    rows = doc.get("variables")
    if not isinstance(rows, list) or not rows:
        raise RuntimeError(f"Compustat allowlist variables must be a non-empty list: {path}")

    entries: list[CompustatAllowlistEntry] = []
    for idx, row in enumerate(rows):
        if not isinstance(row, dict):
            raise RuntimeError(f"Compustat allowlist variables[{idx}] must be an object: {path}")
        code = row.get("code")
        desc = row.get("description")
        if not isinstance(code, str) or not code.strip():
            raise RuntimeError(f"Compustat allowlist variables[{idx}].code must be non-empty string: {path}")
        if not isinstance(desc, str) or not desc.strip():
            raise RuntimeError(f"Compustat allowlist variables[{idx}].description must be non-empty string: {path}")
        entries.append(CompustatAllowlistEntry(code=code.strip(), description=desc.strip()))

    # Stable de-dupe / validation.
    seen: set[str] = set()
    deduped: list[CompustatAllowlistEntry] = []
    for e in entries:
        k = e.code.lower()
        if k in seen:
            continue
        seen.add(k)
        deduped.append(CompustatAllowlistEntry(code=k, description=e.description))

    return sorted(deduped, key=lambda e: e.code)


def _render_allowlist_table(allowlist: list[CompustatAllowlistEntry]) -> str:
    lines = [
        "| Code | Description |",
        "|------|-------------|",
    ]
    for e in allowlist:
        lines.append(f"| {e.code} | {e.description} |")
    return "\n".join(lines)


def _render_prompt(
    template: str,
    *,
    term: str,
    definition_verbatim: str | None,
    expression_ast: dict[str, Any] | None,
    input_terms: list[str],
    clauses: list[dict[str, Any]],
    compustat_allowlist_table: str,
) -> str:
    out = template
    out = out.replace("{{TERM}}", term)
    out = out.replace("{{DEFINITION_VERBATIM}}", definition_verbatim or "null")
    out = out.replace("{{EXPRESSION_AST_JSON}}", json.dumps(expression_ast, ensure_ascii=False) if expression_ast is not None else "null")
    out = out.replace("{{INPUT_TERMS_JSON}}", json.dumps(input_terms, ensure_ascii=False))
    out = out.replace("{{CLAUSES_JSON}}", json.dumps(clauses, ensure_ascii=False))
    out = out.replace("{{COMPUSTAT_ALLOWLIST_TABLE}}", compustat_allowlist_table)
    return out


def _validate_overlay_output(
    *,
    term: str,
    raw_json: Dict[str, Any],
    compustat_allowlist: set[str],
    financial_services_codes: set[str],
) -> Dict[str, Any]:
    if not isinstance(raw_json, dict):
        raise RuntimeError("overlay output must be a JSON object")

    schema_version = raw_json.get("schema_version")
    if schema_version == "compustat_overlay_candidates_v1":
        allowed_top_keys = {"schema_version", "term", "selected_candidate_idx", "candidates"}
        extra = sorted(set(raw_json.keys()) - allowed_top_keys)
        missing = sorted(allowed_top_keys - set(raw_json.keys()))
        if missing or extra:
            raise RuntimeError(
                "overlay output must have exactly the required keys "
                f"(missing={missing}, extra={extra})"
            )

        if raw_json.get("term") != term:
            raise RuntimeError(f"overlay output term must equal input term {term!r}; got {raw_json.get('term')!r}")

        selected_idx = raw_json.get("selected_candidate_idx")
        if not isinstance(selected_idx, int):
            raise RuntimeError("overlay output selected_candidate_idx must be an integer")

        candidates = raw_json.get("candidates")
        if not isinstance(candidates, list) or not all(isinstance(c, dict) for c in candidates):
            raise RuntimeError("overlay output candidates must be list[object]")
        if not (2 <= len(candidates) <= 4):
            raise RuntimeError("overlay output candidates must have between 2 and 4 entries")
        if selected_idx < 0 or selected_idx >= len(candidates):
            raise RuntimeError("overlay output selected_candidate_idx out of range for candidates[]")

        def _validate_formula_and_vars(*, formula: str, vars_: list[str]) -> None:
            # Basic formula hygiene: allow only operators + - * / ( ) . whitespace and allowlist codes.
            allowed_chars = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_()+-*/. \\t\\n\\r")
            if any(ch not in allowed_chars for ch in formula):
                raise RuntimeError("overlay output compustat_formula contains disallowed characters")

            identifiers = re.findall(r"\b[a-z][a-z0-9_]*\b", formula.lower())
            bad = [c for c in identifiers if c not in compustat_allowlist]
            if bad:
                raise RuntimeError(
                    f"overlay output compustat_formula uses identifiers outside allowlist (examples={bad[:10]})"
                )

            if any(v not in compustat_allowlist for v in vars_):
                raise RuntimeError("overlay output compustat_variables must only contain allowlisted codes")

            if set(vars_) != set(identifiers):
                raise RuntimeError(
                    "overlay output compustat_variables must list exactly the codes used in compustat_formula"
                )

        def _validate_candidate(*, idx: int, cand: dict[str, Any]) -> dict[str, Any]:
            allowed_keys = {
                "why",
                "compustat_formula",
                "compustat_variables",
                "match_type",
                "confidence",
                "limitations",
                "notes",
            }
            extra = sorted(set(cand.keys()) - allowed_keys)
            missing = sorted(allowed_keys - set(cand.keys()))
            if missing or extra:
                raise RuntimeError(
                    f"overlay output candidates[{idx}] must have exactly the required keys "
                    f"(missing={missing}, extra={extra})"
                )

            why = cand.get("why")
            if not isinstance(why, str) or not why.strip():
                raise RuntimeError(f"overlay output candidates[{idx}].why must be a non-empty string")

            match_type = cand.get("match_type")
            if match_type not in {"exact", "approximate", "none"}:
                raise RuntimeError(f"overlay output candidates[{idx}].match_type is invalid")

            confidence = cand.get("confidence")
            if confidence not in {"high", "medium", "low"}:
                raise RuntimeError(f"overlay output candidates[{idx}].confidence is invalid")
            if match_type == "approximate" and confidence == "high":
                raise RuntimeError(
                    f"overlay output candidates[{idx}].confidence must not be 'high' when match_type is 'approximate'"
                )

            limitations = cand.get("limitations")
            if not isinstance(limitations, list) or not all(isinstance(x, str) for x in limitations):
                raise RuntimeError(f"overlay output candidates[{idx}].limitations must be list[str]")
            bad_lim = [
                x for x in limitations if x.strip().lower().startswith(("assumption:", "alternative:", "next_step:"))
            ]
            if bad_lim:
                raise RuntimeError(
                    f"overlay output candidates[{idx}].limitations entries must be plain statements "
                    "and must not start with 'assumption:'/'alternative:'/'next_step:'"
                )

            notes = cand.get("notes")
            if not isinstance(notes, list) or not all(isinstance(x, str) for x in notes):
                raise RuntimeError(f"overlay output candidates[{idx}].notes must be list[str]")

            formula = cand.get("compustat_formula")
            if formula is not None and not isinstance(formula, str):
                raise RuntimeError(f"overlay output candidates[{idx}].compustat_formula must be string or null")

            vars_ = cand.get("compustat_variables")
            if not isinstance(vars_, list) or not all(isinstance(v, str) for v in vars_):
                raise RuntimeError(f"overlay output candidates[{idx}].compustat_variables must be list[str]")
            if len(vars_) != len(set(vars_)):
                raise RuntimeError(
                    f"overlay output candidates[{idx}].compustat_variables must be deduped (no duplicates)"
                )

            if formula is None:
                if match_type != "none":
                    raise RuntimeError(
                        f"overlay output candidates[{idx}].match_type must be 'none' when compustat_formula is null"
                    )
                if vars_:
                    raise RuntimeError(
                        f"overlay output candidates[{idx}].compustat_variables must be [] when compustat_formula is null"
                    )
                if not limitations:
                    raise RuntimeError(
                        f"overlay output candidates[{idx}].limitations must be non-empty when match_type is 'none'"
                    )
                if not any(n.strip().lower().startswith("next_step:") for n in notes):
                    raise RuntimeError(
                        f"overlay output candidates[{idx}].notes must include at least one 'next_step:' entry when match_type is 'none'"
                    )
                return cand

            if match_type == "none":
                raise RuntimeError(
                    f"overlay output candidates[{idx}].match_type must not be 'none' when compustat_formula is provided"
                )
            if match_type == "exact" and limitations:
                raise RuntimeError(
                    f"overlay output candidates[{idx}].limitations must be [] when match_type is 'exact'"
                )
            if match_type == "approximate" and not limitations:
                raise RuntimeError(
                    f"overlay output candidates[{idx}].limitations must be non-empty when match_type is 'approximate'"
                )
            if match_type == "approximate" and not any(n.strip().lower().startswith("assumption:") for n in notes):
                raise RuntimeError(
                    f"overlay output candidates[{idx}].notes must include at least one 'assumption:' entry when match_type is 'approximate'"
                )

            fs_used = sorted({v for v in vars_ if v in financial_services_codes})
            if fs_used:
                if confidence != "low":
                    raise RuntimeError(
                        f"overlay output candidates[{idx}].confidence must be 'low' when using Financial Services variables "
                        f"(used={fs_used})"
                    )
                has_fs_assumption = any(
                    n.strip().lower().startswith("assumption:")
                    and (
                        "financial services" in n.lower()
                        or any(code in n.lower() for code in fs_used)
                    )
                    for n in notes
                )
                if not has_fs_assumption:
                    raise RuntimeError(
                        f"overlay output candidates[{idx}].notes must include an 'assumption:' explicitly justifying use of "
                        f"Financial Services variables (used={fs_used})"
                    )

            _validate_formula_and_vars(formula=formula, vars_=vars_)
            return cand

        cleaned_candidates: list[dict[str, Any]] = []
        any_formula = False
        for idx, cand in enumerate(candidates):
            cleaned = _validate_candidate(idx=idx, cand=cand)
            cleaned_candidates.append(cleaned)
            if cleaned.get("compustat_formula") is not None:
                any_formula = True

        # Selected candidate must include an explicit ranking explanation (auditability).
        selected_notes = cleaned_candidates[selected_idx].get("notes") or []
        if not any(n.strip().lower().startswith("ranking:") for n in selected_notes):
            raise RuntimeError(
                "overlay output selected candidate notes must include at least one 'ranking:' entry "
                "explaining why it was selected over other candidates"
            )

        if any_formula and cleaned_candidates[selected_idx].get("compustat_formula") is None:
            raise RuntimeError(
                "overlay output selected_candidate_idx must point to a candidate with a non-null compustat_formula "
                "when any candidate provides a formula"
            )

        # Enforce selection consistency: prefer exact when available; otherwise never select lower confidence
        # when a higher-confidence candidate exists.
        if any(c.get("match_type") == "exact" for c in cleaned_candidates):
            if cleaned_candidates[selected_idx].get("match_type") != "exact":
                raise RuntimeError(
                    "overlay output selected_candidate_idx must point to an 'exact' candidate when any candidate is 'exact'"
                )

        confidence_rank = {"high": 3, "medium": 2, "low": 1}
        selectable = [
            c
            for c in cleaned_candidates
            if c.get("compustat_formula") is not None and c.get("match_type") in {"exact", "approximate"}
        ]
        if selectable:
            max_rank = max(confidence_rank.get(c.get("confidence"), 0) for c in selectable)
            sel_rank = confidence_rank.get(cleaned_candidates[selected_idx].get("confidence"), 0)
            if sel_rank < max_rank:
                raise RuntimeError(
                    "overlay output selected_candidate_idx must point to a candidate with the maximum confidence "
                    "among candidates providing a formula"
                )

        # Term-specific enforcement where we know an "exact" Compustat mapping is not realistic.
        if term.strip() == "EBITDA":
            for idx, cand in enumerate(cleaned_candidates):
                if cand.get("compustat_formula") is not None and cand.get("match_type") == "exact":
                    raise RuntimeError(
                        f"overlay output candidates[{idx}].match_type must not be 'exact' for TERM='EBITDA' "
                        "(contract EBITDA definitions are bespoke; require 'approximate')"
                    )

            needed = {"niq", "xintq", "txtq", "dpq"}
            has_standard = any(set(c.get("compustat_variables") or []) == needed for c in cleaned_candidates)
            if not has_standard:
                raise RuntimeError(
                    "overlay output must include an EBITDA candidate using variables {niq,xintq,txtq,dpq} "
                    "when those codes are allowlisted"
                )
            if "oibdpq" in compustat_allowlist:
                has_oibdpq = any(set(c.get("compustat_variables") or []) == {"oibdpq"} for c in cleaned_candidates)
                if not has_oibdpq:
                    raise RuntimeError(
                        "overlay output must include an EBITDA single-variable candidate using {oibdpq} "
                        "when oibdpq is allowlisted"
                    )

        # Return a cleaned object (structure unchanged; candidate dicts already validated).
        return {
            "schema_version": "compustat_overlay_candidates_v1",
            "term": term,
            "selected_candidate_idx": selected_idx,
            "candidates": cleaned_candidates,
        }

    if schema_version != "compustat_overlay_v1":
        raise RuntimeError("overlay output schema_version must equal 'compustat_overlay_v1' or 'compustat_overlay_candidates_v1'")

    # Legacy (single-output) schema.
    allowed_keys = {
        "schema_version",
        "term",
        "compustat_formula",
        "compustat_variables",
        "match_type",
        "confidence",
        "limitations",
        "notes",
    }
    extra = sorted(set(raw_json.keys()) - allowed_keys)
    missing = sorted(allowed_keys - set(raw_json.keys()))
    if missing or extra:
        raise RuntimeError(f"overlay output must have exactly the required keys (missing={missing}, extra={extra})")

    if raw_json.get("term") != term:
        raise RuntimeError(f"overlay output term must equal input term {term!r}; got {raw_json.get('term')!r}")

    match_type = raw_json.get("match_type")
    if match_type not in {"exact", "approximate", "none"}:
        raise RuntimeError("overlay output match_type must be 'exact'|'approximate'|'none'")

    confidence = raw_json.get("confidence")
    if confidence not in {"high", "medium", "low"}:
        raise RuntimeError("overlay output confidence must be 'high'|'medium'|'low'")

    limitations = raw_json.get("limitations")
    if not isinstance(limitations, list) or not all(isinstance(x, str) for x in limitations):
        raise RuntimeError("overlay output limitations must be list[str]")

    notes = raw_json.get("notes")
    if not isinstance(notes, list) or not all(isinstance(x, str) for x in notes):
        raise RuntimeError("overlay output notes must be list[str]")

    bad_lim = [x for x in limitations if x.strip().lower().startswith(("assumption:", "alternative:", "next_step:"))]
    if bad_lim:
        raise RuntimeError(
            "overlay output limitations entries must be plain statements and must not start with "
            "'assumption:'/'alternative:'/'next_step:' (move those to notes[])"
        )

    formula = raw_json.get("compustat_formula")
    if formula is not None and not isinstance(formula, str):
        raise RuntimeError("overlay output compustat_formula must be string or null")

    vars_ = raw_json.get("compustat_variables")
    if not isinstance(vars_, list) or not all(isinstance(v, str) for v in vars_):
        raise RuntimeError("overlay output compustat_variables must be list[str]")
    if len(vars_) != len(set(vars_)):
        raise RuntimeError("overlay output compustat_variables must be deduped (no duplicates)")

    if formula is None:
        if match_type != "none":
            raise RuntimeError("overlay output match_type must be 'none' when compustat_formula is null")
        if vars_:
            raise RuntimeError("overlay output compustat_variables must be [] when compustat_formula is null")
        if not limitations:
            raise RuntimeError("overlay output limitations must be non-empty when match_type is 'none'")
        if not any(n.strip().lower().startswith("next_step:") for n in notes):
            raise RuntimeError(
                "overlay output notes must include at least one 'next_step:' entry when match_type is 'none'"
            )
        return raw_json

    if match_type == "none":
        raise RuntimeError("overlay output match_type must not be 'none' when compustat_formula is provided")
    if match_type == "approximate" and confidence == "high":
        raise RuntimeError("overlay output confidence must not be 'high' when match_type is 'approximate'")
    if match_type != "exact" and not limitations:
        raise RuntimeError("overlay output limitations must be non-empty when match_type is 'approximate'")
    if match_type == "exact" and limitations:
        raise RuntimeError("overlay output limitations must be [] when match_type is 'exact'")
    if match_type == "approximate" and not any(n.strip().lower().startswith("assumption:") for n in notes):
        raise RuntimeError("overlay output notes must include at least one 'assumption:' entry when match_type is 'approximate'")
    if not any(n.strip().lower().startswith("alternative:") for n in notes):
        raise RuntimeError("overlay output notes must include at least one 'alternative:' entry when compustat_formula is provided")

    allowed_chars = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_()+-*/. \\t\\n\\r")
    if any(ch not in allowed_chars for ch in formula):
        raise RuntimeError("overlay output compustat_formula contains disallowed characters")

    identifiers = re.findall(r"\b[a-z][a-z0-9_]*\b", formula.lower())
    bad = [c for c in identifiers if c not in compustat_allowlist]
    if bad:
        raise RuntimeError(f"overlay output compustat_formula uses identifiers outside allowlist (examples={bad[:10]})")

    fs_used = sorted({c for c in identifiers if c in financial_services_codes})
    if fs_used:
        if confidence != "low":
            raise RuntimeError(
                "overlay output confidence must be 'low' when compustat_formula uses Financial Services variables "
                f"(used={fs_used})"
            )
        has_fs_assumption = any(
            n.strip().lower().startswith("assumption:")
            and (
                "financial services" in n.lower()
                or any(code in n.lower() for code in fs_used)
            )
            for n in notes
        )
        if not has_fs_assumption:
            raise RuntimeError(
                "overlay output notes must include an 'assumption:' explicitly justifying use of Financial Services variables "
                f"(used={fs_used})"
            )

    if any(v not in compustat_allowlist for v in vars_):
        raise RuntimeError("overlay output compustat_variables must only contain allowlisted codes")

    if set(vars_) != set(identifiers):
        raise RuntimeError("overlay output compustat_variables must list exactly the codes used in compustat_formula")

    return raw_json


def _load_definitions_from_aggregate(path: Path) -> list[dict[str, Any]]:
    doc = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(doc, dict):
        raise RuntimeError(f"Aggregate file must be a JSON object: {path}")
    if doc.get("schema_version") != "definition_compiler_v2_ast_v1":
        raise RuntimeError(f"Unexpected schema_version in aggregate: {path}")
    defs = doc.get("definitions")
    if not isinstance(defs, list) or not all(isinstance(d, dict) for d in defs):
        raise RuntimeError(f"Aggregate definitions must be list[object]: {path}")
    return defs


def run_compustat_overlay_v1(
    paths: Paths,
    item_ids: Iterable[str],
    *,
    overlay_prompt_path: Path,
    allowlist_path: Path | None = None,
    output_subdir: str,
    metrics_output_subdir: str | None,
    blocking_terms_output_subdir: str | None,
    gateway_url: str | None = None,
    temperature: float = 0.0,
    gateway_timeout: float | None = None,
    concurrency: int = 4,
    attempts: int = 3,
) -> None:
    """Run a Compustat mapping overlay on previously-extracted v2 AST definitions."""

    assert_exists(overlay_prompt_path, message=f"Compustat overlay prompt not found: {overlay_prompt_path}")

    # Enforce model + reasoning defaults for repeatability.
    model = REQUIRED_MODEL
    reasoning = REQUIRED_REASONING

    allowlist_path = allowlist_path or (paths.root / "datasets" / "compustat_allowlist_quarterly_v1.json")
    assert_exists(allowlist_path, message=f"Compustat allowlist not found: {allowlist_path}")
    allowlist = _load_compustat_allowlist(allowlist_path)
    compustat_allowlist = {e.code for e in allowlist}
    financial_services_codes = {e.code for e in allowlist if "financial services" in e.description.lower()}
    compustat_allowlist_table = _render_allowlist_table(allowlist)

    prompt_template = overlay_prompt_path.read_text(encoding="utf-8")
    prompt_digest = prompt_hash(overlay_prompt_path)
    allowlist_digest = prompt_hash(allowlist_path)

    out_dir = paths.run_dir / "compustat_overlay_v1" / output_subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    for stale_name in ("errors.txt", "issues.txt"):
        stale_path = out_dir / stale_name
        if stale_path.exists():
            stale_path.unlink()

    GatewayAgentClient = _ensure_gateway_client_async()
    sem = asyncio.Semaphore(max(1, concurrency))

    hard_errors: list[tuple[str, str]] = []
    soft_issues: list[tuple[str, str]] = []

    def _targets_for_item(item_id: str) -> list[OverlayTarget]:
        defs: list[dict[str, Any]] = []

        if metrics_output_subdir:
            metrics_agg = paths.run_dir / "definitions_compiler_v1" / metrics_output_subdir / f"{item_id}__compiled.json"
            if metrics_agg.exists():
                defs.extend(_load_definitions_from_aggregate(metrics_agg))

        if blocking_terms_output_subdir:
            terms_agg = (
                paths.run_dir / "blocking_terms_compiler_v1" / blocking_terms_output_subdir / f"{item_id}__compiled.json"
            )
            if terms_agg.exists():
                defs.extend(_load_definitions_from_aggregate(terms_agg))

        by_name: dict[str, dict[str, Any]] = {}
        for d in defs:
            name = d.get("name")
            if not isinstance(name, str) or not name.strip():
                continue
            # Prefer later entries (blocking terms can supersede metrics if names collide).
            by_name[name.strip()] = d

        targets: list[OverlayTarget] = []
        for name, d in sorted(by_name.items(), key=lambda kv: kv[0]):
            dv = d.get("definition_verbatim")
            if dv is not None and not isinstance(dv, str):
                dv = None
            expr = d.get("expression_ast")
            if expr is not None and not isinstance(expr, dict):
                expr = None
            input_terms = d.get("input_terms") if isinstance(d.get("input_terms"), list) else []
            input_terms = [t for t in input_terms if isinstance(t, str)]
            clauses = d.get("clauses") if isinstance(d.get("clauses"), list) else []
            clauses = [c for c in clauses if isinstance(c, dict)]
            targets.append(
                OverlayTarget(
                    item_id=item_id,
                    term=name,
                    definition_verbatim=dv,
                    expression_ast=expr,
                    input_terms=input_terms,
                    clauses=clauses,
                )
            )
        return targets

    async def _call_gateway(*, client: Any, prompt: str) -> str:
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

    async def _process_target(client: Any, target: OverlayTarget) -> dict[str, Any] | None:
        async with sem:
            safe_term = _safe_slug(target.term)
            raw_base = out_dir / f"{target.item_id}__{safe_term}__compustat_overlay.raw.txt"
            out_path = out_dir / f"{target.item_id}__{safe_term}__compustat_overlay.json"

            if out_path.exists():
                try:
                    return json.loads(out_path.read_text(encoding="utf-8"))
                except Exception:
                    soft_issues.append((f"{target.item_id}::{target.term}", f"Existing overlay JSON unreadable; recomputing: {out_path}"))

            rendered = _render_prompt(
                prompt_template,
                term=target.term,
                definition_verbatim=target.definition_verbatim,
                expression_ast=target.expression_ast,
                input_terms=target.input_terms,
                clauses=target.clauses,
                compustat_allowlist_table=compustat_allowlist_table,
            )

            last_error = "unknown"
            last_raw = ""
            for attempt in range(1, max(1, attempts) + 1):
                prompt = rendered
                if attempt > 1:
                    prompt = (
                        f"{rendered.strip()}\n\n"
                        f"=== REPAIR MODE (attempt {attempt}) ===\n"
                        "Your previous output was invalid JSON or failed schema validation.\n"
                        f"ERROR: {last_error}\n\n"
                        "Return corrected JSON only (no code fences). Keep the same intended mapping.\n\n"
                        "Previous output:\n"
                        f"{last_raw.strip()}\n"
                    )

                try:
                    raw = await _call_gateway(client=client, prompt=prompt)
                except Exception as exc:  # pragma: no cover
                    last_error = f"gateway error: {exc}"
                    await asyncio.sleep(1.5 * attempt)
                    continue

                if not raw or not raw.strip():
                    last_error = "gateway returned empty response text"
                    await asyncio.sleep(1.5 * attempt)
                    continue

                raw_attempt = raw_base if attempt == 1 else raw_base.with_name(
                    raw_base.name.replace(".raw.", f".raw.attempt{attempt}.")
                )
                raw_attempt.write_text(raw, encoding="utf-8")
                last_raw = raw

                try:
                    parsed = json.loads(raw)
                except json.JSONDecodeError as exc:
                    last_error = f"invalid JSON: {exc}"
                    await asyncio.sleep(1.5 * attempt)
                    continue

                try:
                    cleaned = _validate_overlay_output(
                        term=target.term,
                        raw_json=parsed,
                        compustat_allowlist=compustat_allowlist,
                        financial_services_codes=financial_services_codes,
                    )
                except Exception as exc:
                    last_error = f"schema validation failed: {exc}"
                    await asyncio.sleep(1.5 * attempt)
                    continue

                raw_base.write_text(raw, encoding="utf-8")
                out_path.write_text(json.dumps(cleaned, indent=2) + "\n", encoding="utf-8")
                return cleaned

            hard_errors.append((f"{target.item_id}::{target.term}", f"Failed after {attempts} attempt(s): {last_error}"))
            return None

    async def _process_item(client: Any, item_id: str) -> None:
        try:
            targets = _targets_for_item(item_id)
        except Exception as exc:
            err_path = out_dir / f"{item_id}__inputs.error.txt"
            err_path.write_text(f"{exc}\n\n{traceback.format_exc()}", encoding="utf-8")
            hard_errors.append((f"{item_id}::__inputs__", str(exc)))
            return

        if not targets:
            soft_issues.append((f"{item_id}::__no_targets__", "No definitions found in provided aggregates; skipping."))
            return

        results = await asyncio.gather(*(_process_target(client, t) for t in targets), return_exceptions=True)
        overlays: list[dict[str, Any]] = []
        for t, res in zip(targets, results):
            if isinstance(res, Exception):
                hard_errors.append((f"{item_id}::{t.term}", str(res)))
                continue
            if isinstance(res, dict):
                overlays.append(res)

        # Item-level aggregate for convenience.
        (out_dir / f"{item_id}__compustat_overlay.json").write_text(
            json.dumps(
                {
                    "schema_version": "compustat_overlay_v1",
                    "item_id": item_id,
                    "created_at": int(time.time()),
                    "overlays": overlays,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

    async def _runner() -> None:
        async with GatewayAgentClient(
            base_url=gateway_url or DEFAULT_GATEWAY_URL,
            timeout=gateway_timeout or 600.0,
        ) as client:
            await asyncio.gather(*(_process_item(client, item_id) for item_id in item_ids), return_exceptions=False)

    asyncio.run(_runner())

    if soft_issues:
        (out_dir / "issues.txt").write_text("\n".join(f"{k}: {v}" for k, v in soft_issues) + "\n", encoding="utf-8")

    if hard_errors:
        (out_dir / "errors.txt").write_text("\n".join(f"{k}: {v}" for k, v in hard_errors) + "\n", encoding="utf-8")

    if paths.manifest_path.exists():
        update_manifest(
            paths.manifest_path,
            compustat_overlay_v1_prompt=str(overlay_prompt_path),
            compustat_overlay_v1_prompt_sha256=prompt_digest,
            compustat_overlay_v1_allowlist=str(allowlist_path),
            compustat_overlay_v1_allowlist_sha256=allowlist_digest,
            compustat_overlay_v1_output_subdir=output_subdir,
            compustat_overlay_v1_metrics_input_subdir=metrics_output_subdir,
            compustat_overlay_v1_terms_input_subdir=blocking_terms_output_subdir,
            compustat_overlay_v1_created_at=int(time.time()),
        )

    if hard_errors:
        raise RuntimeError(f"Compustat overlay v1 completed with errors (count={len(hard_errors)}); see {out_dir/'errors.txt'}")
