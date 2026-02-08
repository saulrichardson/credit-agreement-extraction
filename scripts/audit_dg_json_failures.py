#!/usr/bin/env python
"""
Audit DG structured-v2 outputs for:
  - JSON parse failures
  - missing/typed fields in the expected schema
  - ungrounded helper fields (contract_term/search_tokens not present in snippet pack)

This is meant to be a lightweight, artifact-first diagnostic tool to guide prompt iteration.

Example:
  poetry run python scripts/audit_dg_json_failures.py \
    --run-id dan-v2-20260106 \
    --qa-subdir prompt_pricing_second_pass_dg_nano_v2_tuned_v2_strict
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


@dataclass(frozen=True)
class Issue:
    item_id: str
    kind: str
    json_path: str
    detail: str


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def _read_json(path: Path) -> Any:
    return json.loads(_read_text(path))


def _iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _snippet_pack_text(snippets_path: Path) -> str:
    parts: list[str] = []
    for rec in _iter_jsonl(snippets_path):
        snippet = rec.get("snippet")
        if isinstance(snippet, str) and snippet.strip():
            parts.append(snippet)
    return "\n".join(parts)


def _contains_case_insensitive(haystack: str, needle: str) -> bool:
    if not needle:
        return False
    return needle.lower() in haystack.lower()


def _has_alpha(s: str) -> bool:
    return any(ch.isalpha() for ch in s)


def _looks_like_numeric_rate(s: str) -> bool:
    """Heuristic: detect strings like '0%', '0.24%', '175 bps', etc."""
    s_norm = s.strip().lower()
    if not s_norm:
        return False
    # Common percent forms.
    if s_norm.endswith("%"):
        core = s_norm[:-1].strip()
        try:
            float(core)
            return True
        except ValueError:
            return False
    # Common bps forms.
    for suffix in ("bps", "bp"):
        if s_norm.endswith(suffix):
            core = s_norm[: -len(suffix)].strip()
            try:
                float(core)
                return True
            except ValueError:
                return False
    # Pure number.
    try:
        float(s_norm)
        return True
    except ValueError:
        return False


def _require_key(obj: dict, key: str, item_id: str, path: str, issues: list[Issue]) -> Any:
    if key not in obj:
        issues.append(Issue(item_id=item_id, kind="missing_key", json_path=f"{path}/{key}", detail="missing"))
        return None
    return obj.get(key)


def _expect_type(val: Any, typ: type, item_id: str, path: str, issues: list[Issue]) -> bool:
    if not isinstance(val, typ):
        issues.append(Issue(item_id=item_id, kind="type_error", json_path=path, detail=f"expected {typ.__name__}"))
        return False
    return True


def audit_one(
    *,
    item_id: str,
    raw_text: str,
    snippet_text: Optional[str],
) -> Tuple[Optional[dict], list[Issue]]:
    issues: list[Issue] = []
    try:
        doc = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        issues.append(Issue(item_id=item_id, kind="json_parse_error", json_path="/", detail=str(exc)))
        return None, issues

    if not isinstance(doc, dict):
        issues.append(Issue(item_id=item_id, kind="type_error", json_path="/", detail="top-level must be object"))
        return None, issues

    # Top-level required keys for dg-v2 prompts.
    for k in ("issuer", "agreement", "metrics", "tier_schemes", "facilities", "overrides"):
        _require_key(doc, k, item_id, "/", issues)

    # Agreement object (light checks; allow empty strings).
    ag = doc.get("agreement")
    if ag is not None:
        if not isinstance(ag, dict):
            issues.append(Issue(item_id=item_id, kind="type_error", json_path="/agreement", detail="must be object"))
        else:
            for k in ("title", "section_refs", "currency", "as_of_date"):
                _require_key(ag, k, item_id, "/agreement", issues)

    # Metrics: ensure helper fields exist and are typed; optionally check grounding against snippet pack.
    metric_names: set[str] = set()
    metrics = doc.get("metrics")
    if isinstance(metrics, list):
        for mi, m in enumerate(metrics):
            p = f"/metrics/{mi}"
            if not isinstance(m, dict):
                issues.append(Issue(item_id=item_id, kind="type_error", json_path=p, detail="must be object"))
                continue
            name = _require_key(m, "name", item_id, p, issues)
            if isinstance(name, str) and name.strip():
                metric_names.add(name.strip())
            contract_term = _require_key(m, "contract_term", item_id, p, issues)
            search_tokens = _require_key(m, "search_tokens", item_id, p, issues)
            _require_key(m, "source_refs", item_id, p, issues)

            if contract_term is not None and not isinstance(contract_term, str):
                issues.append(Issue(item_id=item_id, kind="type_error", json_path=f"{p}/contract_term", detail="must be string"))
            if search_tokens is not None and (
                not isinstance(search_tokens, list) or not all(isinstance(t, str) for t in search_tokens)
            ):
                issues.append(
                    Issue(item_id=item_id, kind="type_error", json_path=f"{p}/search_tokens", detail="must be list[str]")
                )

            if snippet_text is not None:
                if isinstance(contract_term, str) and contract_term.strip():
                    if not _contains_case_insensitive(snippet_text, contract_term.strip()):
                        issues.append(
                            Issue(
                                item_id=item_id,
                                kind="metric_contract_term_not_in_snippet_pack",
                                json_path=f"{p}/contract_term",
                                detail=contract_term.strip(),
                            )
                        )
                if isinstance(search_tokens, list):
                    for ti, tok in enumerate(search_tokens):
                        tok_s = tok.strip()
                        if not tok_s:
                            continue
                        if not _contains_case_insensitive(snippet_text, tok_s):
                            issues.append(
                                Issue(
                                    item_id=item_id,
                                    kind="metric_search_token_not_in_snippet_pack",
                                    json_path=f"{p}/search_tokens/{ti}",
                                    detail=tok_s,
                                )
                            )

    # Tier schemes: metric consistency check (metric referenced in tier conditions must be in metrics[]).
    tier_schemes = doc.get("tier_schemes")
    if isinstance(tier_schemes, list):
        for si, s in enumerate(tier_schemes):
            sp = f"/tier_schemes/{si}"
            if not isinstance(s, dict):
                issues.append(Issue(item_id=item_id, kind="type_error", json_path=sp, detail="must be object"))
                continue
            _require_key(s, "scheme_id", item_id, sp, issues)
            classification = _require_key(s, "classification", item_id, sp, issues)
            tiers = _require_key(s, "tiers", item_id, sp, issues)
            if tiers is not None and not isinstance(tiers, list):
                issues.append(Issue(item_id=item_id, kind="type_error", json_path=f"{sp}/tiers", detail="must be list"))
                continue
            if isinstance(tiers, list):
                for ti, t in enumerate(tiers):
                    tp = f"{sp}/tiers/{ti}"
                    if not isinstance(t, dict):
                        issues.append(Issue(item_id=item_id, kind="type_error", json_path=tp, detail="must be object"))
                        continue
                    _require_key(t, "tier_id", item_id, tp, issues)
                    _require_key(t, "display", item_id, tp, issues)
                    conds = _require_key(t, "conditions", item_id, tp, issues)
                    if classification == "financial_metrics" and conds is None:
                        issues.append(
                            Issue(
                                item_id=item_id,
                                kind="tier_conditions_null_for_financial_metrics",
                                json_path=f"{tp}/conditions",
                                detail="",
                            )
                        )
                    if conds is None:
                        continue
                    if conds is not None and conds is not None and not (conds is None or isinstance(conds, list)):
                        issues.append(Issue(item_id=item_id, kind="type_error", json_path=f"{tp}/conditions", detail="must be list or null"))
                        continue
                    if isinstance(conds, list):
                        for ci, c in enumerate(conds):
                            cp = f"{tp}/conditions/{ci}"
                            if not isinstance(c, dict):
                                issues.append(Issue(item_id=item_id, kind="type_error", json_path=cp, detail="must be object"))
                                continue
                            metric = c.get("metric")
                            if isinstance(metric, str) and metric.strip():
                                if metric.strip() not in metric_names:
                                    issues.append(
                                        Issue(
                                            item_id=item_id,
                                            kind="tier_condition_metric_not_in_metrics_list",
                                            json_path=f"{cp}/metric",
                                            detail=metric.strip(),
                                        )
                                    )

    # Facilities/rates: base-rate helpers grounding.
    facilities = doc.get("facilities")
    if isinstance(facilities, list):
        for fi, f in enumerate(facilities):
            fp = f"/facilities/{fi}"
            if not isinstance(f, dict):
                issues.append(Issue(item_id=item_id, kind="type_error", json_path=fp, detail="must be object"))
                continue
            rates = f.get("rates")
            if not isinstance(rates, list):
                continue
            for ri, r in enumerate(rates):
                rp = f"{fp}/rates/{ri}"
                if not isinstance(r, dict):
                    continue
                cond = r.get("condition")
                if isinstance(cond, dict):
                    cm = cond.get("metric")
                    if isinstance(cm, str) and cm.strip():
                        if cm.strip() not in metric_names:
                            issues.append(
                                Issue(
                                    item_id=item_id,
                                    kind="rate_condition_metric_not_in_metrics_list",
                                    json_path=f"{rp}/condition/metric",
                                    detail=cm.strip(),
                                )
                            )
                if r.get("type") != "margin":
                    continue
                base_rate = r.get("base_rate")
                toks = r.get("base_rate_search_tokens")
                by_level = r.get("by_level")
                if not isinstance(base_rate, str) or not base_rate.strip():
                    issues.append(Issue(item_id=item_id, kind="base_rate_missing_or_empty", json_path=f"{rp}/base_rate", detail=""))
                elif not _has_alpha(base_rate) and _looks_like_numeric_rate(base_rate):
                    issues.append(
                        Issue(
                            item_id=item_id,
                            kind="base_rate_looks_numeric",
                            json_path=f"{rp}/base_rate",
                            detail=base_rate.strip(),
                        )
                    )
                if not isinstance(toks, list) or not all(isinstance(t, str) for t in toks):
                    issues.append(Issue(item_id=item_id, kind="type_error", json_path=f"{rp}/base_rate_search_tokens", detail="must be list[str]"))
                elif base_rate and isinstance(base_rate, str) and base_rate.strip():
                    if not [t for t in toks if t.strip()]:
                        issues.append(
                            Issue(item_id=item_id, kind="base_rate_search_tokens_empty", json_path=f"{rp}/base_rate_search_tokens", detail=base_rate.strip())
                        )
                if by_level is not None and isinstance(by_level, list) and not by_level:
                    # If we emitted a margin rate at all, we expect at least one by_level bps value. An empty
                    # by_level usually indicates we failed to extract the numeric margin values (or encoded them elsewhere).
                    issues.append(Issue(item_id=item_id, kind="margin_by_level_empty", json_path=f"{rp}/by_level", detail=""))
                if snippet_text is not None and isinstance(toks, list):
                    for ti, tok in enumerate(toks):
                        tok_s = tok.strip()
                        if not tok_s:
                            continue
                        if not _contains_case_insensitive(snippet_text, tok_s):
                            issues.append(
                                Issue(
                                    item_id=item_id,
                                    kind="base_rate_search_token_not_in_snippet_pack",
                                    json_path=f"{rp}/base_rate_search_tokens/{ti}",
                                    detail=tok_s,
                                )
                            )

    return doc, issues


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--qa-subdir", required=True, help="Subdir under runs/<run_id>/llm_qa/ containing DG outputs.")
    ap.add_argument(
        "--item-id",
        dest="item_ids",
        action="append",
        default=[],
        help="Optional item_id(s) to audit (repeatable). If omitted, audits all manifest items (unless --only-present).",
    )
    ap.add_argument(
        "--only-present",
        action="store_true",
        help="Audit only the item_ids that have a *.json (or legacy *.txt) output present in the qa-subdir.",
    )
    ap.add_argument(
        "--snippets-dir",
        default="retrieval_v2",
        help="Directory under runs/<run_id>/ containing <item_id>_snippets.jsonl (default: retrieval_v2).",
    )
    ap.add_argument(
        "--out",
        default=None,
        help="Optional output JSON path (default: runs/<run_id>/report/dg_json_audit_<qa_subdir>.json).",
    )
    args = ap.parse_args()

    run_dir = Path("runs") / args.run_id
    structured_dir = run_dir / "llm_qa" / args.qa_subdir
    if not structured_dir.exists():
        raise SystemExit(f"Missing structured output dir: {structured_dir}")

    # Resolve item_ids to audit.
    if args.only_present:
        present: list[str] = []
        for p in structured_dir.glob("*.json"):
            present.append(p.stem)
        for p in structured_dir.glob("*.txt"):
            # Skip error sidecars from legacy runs.
            if p.name.endswith(".error.txt"):
                continue
            present.append(p.stem)
        item_ids = sorted(set(present))
    elif args.item_ids:
        item_ids = [v for v in args.item_ids if isinstance(v, str) and v.strip()]
    else:
        manifest_path = run_dir / "manifest.json"
        manifest = _read_json(manifest_path) if manifest_path.exists() else {}
        items = manifest.get("items") if isinstance(manifest, dict) else None
        if not isinstance(items, list) or not items:
            raise SystemExit(f"Manifest missing/empty items list: {manifest_path}")
        item_ids = [
            row.get("item_id")
            for row in items
            if isinstance(row, dict) and isinstance(row.get("item_id"), str)
        ]

    snippets_dir = run_dir / args.snippets_dir
    out_path = Path(args.out) if args.out else (run_dir / "report" / f"dg_json_audit_{args.qa_subdir}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    total = 0
    parse_ok = 0
    missing_files: list[str] = []
    all_issues: list[Issue] = []

    for item_id in item_ids:
        if not item_id:
            continue
        total += 1
        json_path = structured_dir / f"{item_id}.json"
        txt_path = structured_dir / f"{item_id}.txt"
        input_path: Path | None = json_path if json_path.exists() else (txt_path if txt_path.exists() else None)
        if input_path is None:
            missing_files.append(item_id)
            all_issues.append(
                Issue(
                    item_id=item_id,
                    kind="missing_output_file",
                    json_path=f"{json_path} | {txt_path}",
                    detail="",
                )
            )
            continue

        raw = _read_text(input_path)
        snippet_text: Optional[str] = None
        snips_path = snippets_dir / f"{item_id}_snippets.jsonl"
        if snips_path.exists():
            try:
                snippet_text = _snippet_pack_text(snips_path)
            except Exception:
                snippet_text = None

        doc, issues = audit_one(item_id=item_id, raw_text=raw, snippet_text=snippet_text)
        if doc is not None:
            parse_ok += 1
        all_issues.extend(issues)

    # Aggregate counts by kind/path.
    kind_counts: dict[str, int] = {}
    path_counts: dict[str, int] = {}
    for iss in all_issues:
        kind_counts[iss.kind] = kind_counts.get(iss.kind, 0) + 1
        path_counts[iss.json_path] = path_counts.get(iss.json_path, 0) + 1

    payload = {
        "schema_version": "dg_json_audit_v0",
        "run_id": args.run_id,
        "qa_subdir": args.qa_subdir,
        "total_items": total,
        "json_parse_ok": parse_ok,
        "json_parse_failed": total - parse_ok,
        "missing_output_files": missing_files,
        "issue_counts_by_kind": dict(sorted(kind_counts.items(), key=lambda kv: (-kv[1], kv[0]))),
        "issue_counts_by_json_path_top": sorted(
            [{"json_path": p, "count": c} for p, c in path_counts.items()],
            key=lambda r: (-int(r["count"]), str(r["json_path"])),
        )[:40],
        "issues": [
            {"item_id": i.item_id, "kind": i.kind, "json_path": i.json_path, "detail": i.detail} for i in all_issues
        ],
    }
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    # Human-readable summary
    print(f"run_id={args.run_id} qa_subdir={args.qa_subdir}")
    print(f"items_total={total} json_parse_ok={parse_ok} json_parse_failed={total - parse_ok}")
    if missing_files:
        print(f"missing_output_files={len(missing_files)}")
    top_kinds = list(payload["issue_counts_by_kind"].items())[:10]
    if top_kinds:
        print("top_issue_kinds:")
        for k, c in top_kinds:
            print(f"  - {k}: {c}")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
