from __future__ import annotations

import json
import shutil
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from .anchors import load_anchor_catalog
from .config import REQUIRED_MODEL, REQUIRED_REASONING, Paths
from .contract_ir_v0_2 import ContractIRValidationError, validate_contract_ir
from .contract_ir_v0_2_merge import merge_contract_ir_v0_2
from .excerpt_packs import ExcerptPackError, build_excerpt_pack_from_canonical, expand_anchor_ids, order_anchor_ids
from .indexing import DEFAULT_GATEWAY_URL, _ensure_gateway_client_sync
from .indexing_v2 import run_indexing_v2
from .retrieval_v2 import render_snippets_v2
from .schemas_v2 import IndexingSelectionV2Artifact
from .utils import assert_exists, load_manifest, manifest_items, prompt_view_path


class ContractIRFlowError(RuntimeError):
    pass


DEFAULT_ANCHOR_GAP_FILL_UP_TO = 12
DEFAULT_ANCHOR_NEIGHBOR_PAD = 2


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _expand_for_excerpt_pack(
    *,
    catalog: Mapping[str, Mapping[str, Any]],
    seed_anchor_ids: Sequence[str],
    fill_gaps_up_to: int,
    neighbor_pad: int,
) -> List[str]:
    """Expand seed anchors deterministically so excerpt packs are locally complete."""

    try:
        return expand_anchor_ids(
            catalog=catalog,
            seed_anchor_ids=seed_anchor_ids,
            fill_gaps_up_to=int(fill_gaps_up_to),
            neighbor_pad=int(neighbor_pad),
        )
    except ExcerptPackError as exc:
        raise ContractIRFlowError(str(exc)) from exc


def _render_prompt(template: str, *, task: str, contexts: str) -> str:
    out = template
    out = out.replace("{{TASK}}", task)
    out = out.replace("{{CONTEXTS}}", contexts)
    return out


def _render_repair_prompt(*, raw_json: str, errors: List[ContractIRValidationError]) -> str:
    err_list = [asdict(e) for e in errors]
    return (
        "Your previous response did not validate against the ContractIR v0.2 schema and/or hard gates.\n"
        "Return corrected JSON ONLY. Do not include commentary or code fences.\n\n"
        "Common fixes:\n"
        "- indices[*] entries MUST include both series_id and unit.\n"
        "  - unit MUST be one of: rate, bps, decimal, money, bool, string (never 'percent').\n"
        "- Numeric literal values must be plain decimal strings (no '%', '$', or commas).\n"
        "- AST nodes must be ONLY one of: {\"lit\":...}, {\"var\":...}, or {\"op\":\"...\",\"args\":[...]}.\n"
        "- Important: derived[*].args is a list of argument objects {\"name\":\"...\",\"type\":\"...\"} (NOT AST nodes).\n"
        "  Only AST operator nodes use \"args\": [<AST node>, ...].\n"
        "- lookup/lookup2/lookup_range/lookup_rule require table_id and column-name args to be AST string literals.\n\n"
        "VALIDATION_ERRORS_JSON:\n"
        f"{json.dumps(err_list, indent=2)}\n\n"
        "PREVIOUS_JSON:\n"
        f"{raw_json}\n"
    )


def _validate_anchor_ids_in_context(doc: Any, allowed_anchor_ids: Sequence[str]) -> List[ContractIRValidationError]:
    allowed = set(allowed_anchor_ids)
    out: List[ContractIRValidationError] = []

    def _walk(obj: Any, path: List[str]) -> None:
        if isinstance(obj, dict):
            for k, v in obj.items():
                if k in ("source_refs", "anchor_ids"):
                    if isinstance(v, list):
                        for idx, aid in enumerate(v):
                            if isinstance(aid, str) and aid in allowed:
                                continue
                            out.append(
                                ContractIRValidationError(
                                    code="anchor_not_in_context",
                                    message=f"Anchor id {aid!r} is not in excerpt anchors {sorted(allowed)}",
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
    """Hard gate: require contract_id and sources[*].item_id to match the item_id."""

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


def _run_contractir_pass(
    *,
    complete_response_sync,
    pass_dir: Path,
    item_id: str,
    prompt_path: Path,
    task: str,
    contexts: str,
    allowed_anchor_ids: Sequence[str],
    attempts: int,
    gateway_url: str,
    timeout_seconds: float,
    temperature: float,
    model: str,
    reasoning: str,
) -> Tuple[Optional[dict], Dict[str, Any]]:
    """Run one prompt with repair loops; returns (validated_doc|None, metrics)."""

    template = _read_text(prompt_path)
    prompt = _render_prompt(template, task=task, contexts=contexts)

    _write_text(pass_dir / "contexts.txt", contexts)
    _write_text(pass_dir / "prompt.txt", prompt)

    attempt = 0
    raw: Optional[str] = None
    parsed: Any = None
    last_errors: List[ContractIRValidationError] = []

    while attempt < max(1, attempts):
        attempt += 1
        try:
            raw = complete_response_sync(
                model=model,
                prompt=prompt,
                base_url=gateway_url,
                reasoning={"effort": reasoning} if reasoning else None,
                temperature=float(temperature),
                max_output_tokens=None,
                timeout=timeout_seconds,
            )
            _write_text(pass_dir / f"raw_attempt_{attempt}.txt", raw)
        except Exception as exc:
            raw = None
            parsed = None
            last_errors = [ContractIRValidationError(code="gateway_error", message=str(exc), json_path="/")]
            _write_json(pass_dir / f"validation_errors_attempt_{attempt}.json", [asdict(e) for e in last_errors])
            prompt = _render_repair_prompt(raw_json="", errors=last_errors)
            continue

        try:
            parsed = json.loads(raw)
            _write_json(pass_dir / f"parsed_attempt_{attempt}.json", parsed)
        except Exception as exc:
            parsed = None
            last_errors = [ContractIRValidationError(code="json_parse", message=str(exc), json_path="/")]

        if parsed is not None:
            last_errors = validate_contract_ir(parsed)
            if not last_errors:
                last_errors.extend(_validate_anchor_ids_in_context(parsed, allowed_anchor_ids))
            if not last_errors:
                last_errors.extend(_validate_contract_and_source_ids(parsed, item_id=item_id))

        if parsed is not None and not last_errors:
            _write_json(pass_dir / "contractir_validated.json", parsed)
            return parsed, {"ok": True, "attempts_used": attempt, "prompt": str(prompt_path)}

        _write_json(pass_dir / f"validation_errors_attempt_{attempt}.json", [asdict(e) for e in last_errors])
        prompt = _render_repair_prompt(raw_json=raw or "", errors=last_errors)

    return None, {"ok": False, "attempts_used": attempt, "prompt": str(prompt_path), "errors": [asdict(e) for e in last_errors]}


def _copytree_clean(src: Path, dst: Path) -> None:
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)


def prepare_run_inputs_from_source(*, dest_paths: Paths, source_paths: Paths) -> None:
    """Clone run inputs (normalized + manifest) into a new run for cleanliness."""

    if dest_paths.run_dir.exists():
        raise ContractIRFlowError(f"Destination run already exists: {dest_paths.run_dir}")

    source_manifest = assert_exists(source_paths.manifest_path, message=f"Missing source manifest: {source_paths.manifest_path}")
    source_normalized = assert_exists(
        source_paths.normalized_dir, message=f"Missing source normalized dir: {source_paths.normalized_dir}"
    )

    dest_paths.run_dir.mkdir(parents=True, exist_ok=True)
    _copytree_clean(source_normalized, dest_paths.normalized_dir)

    manifest = json.loads(source_manifest.read_text(encoding="utf-8"))
    if isinstance(manifest, dict):
        manifest["run_id"] = dest_paths.run_id
        manifest["cloned_from_run_id"] = source_paths.run_id
        manifest["cloned_at"] = int(time.time())
    _write_json(dest_paths.manifest_path, manifest)


def run_contractir_v0_2_flow(
    *,
    paths: Paths,
    item_ids: Optional[Sequence[str]] = None,
    indexing_prompt_path: Path,
    retrieval_bandwidth: int,
    base_rate_prompt_path: Path,
    spread_prompt_paths: Sequence[Path],
    fee_prompt_paths: Sequence[Path],
    gateway_url: Optional[str] = None,
    timeout_seconds: float = 600.0,
    temperature: float = 0.0,
    attempts: int = 3,
    concurrency: int = 1,
    skip_indexing: bool = False,
    skip_retrieval: bool = False,
    fail_on_item_errors: bool = True,
) -> Path:
    """End-to-end pricing-kernel extraction flow (indexing_v2 -> retrieval_v2 -> ContractIR passes -> merge).

    Notes:
    - This is intentionally strict and run-scoped: failures are surfaced as errors (fail fast & loudly).
    - LLM calls are structured into separate passes (base_rate, spread, fee) and merged deterministically.
    """

    if concurrency != 1:
        raise ContractIRFlowError("concurrency != 1 not implemented yet (keep LLM calls sequential for now)")

    manifest = load_manifest(paths.manifest_path)
    items = manifest_items(manifest)
    manifest_item_ids = [item["item_id"] for item in items]

    selected_item_ids: List[str]
    if item_ids:
        unknown = sorted(set(item_ids) - set(manifest_item_ids))
        if unknown:
            raise ContractIRFlowError(f"Unknown item_id(s) not present in manifest: {', '.join(unknown)}")
        selected_item_ids = list(item_ids)
    else:
        selected_item_ids = list(manifest_item_ids)

    # 1) Indexing v2: select anchors with pricing-kernel evidence buckets.
    if not skip_indexing:
        run_indexing_v2(
            paths,
            selected_item_ids,
            indexing_prompt_path,
            model=REQUIRED_MODEL,
            gateway_url=gateway_url,
            temperature=float(temperature),
            reasoning=REQUIRED_REASONING,
            gateway_timeout=float(timeout_seconds),
            concurrency=1,
            attempts=3,
        )

    # 2) Retrieval v2: materialize snippet windows for selected anchors.
    if not skip_retrieval:
        render_snippets_v2(paths, selected_item_ids, bandwidth=int(retrieval_bandwidth))

    # 3) Structured passes (LLM) and deterministic merge.
    out_dir = paths.run_dir / "contractir_v0_2"
    out_dir.mkdir(parents=True, exist_ok=True)

    complete_response_sync = _ensure_gateway_client_sync()
    resolved_gateway_url = gateway_url or DEFAULT_GATEWAY_URL

    run_summary: Dict[str, Any] = {
        "schema_version": "contractir_v0_2_flow_v0",
        "run_id": paths.run_id,
        "created_at": int(time.time()),
        "model": REQUIRED_MODEL,
        "reasoning_effort": REQUIRED_REASONING,
        "gateway_url": resolved_gateway_url,
        "indexing_prompt": str(indexing_prompt_path),
        "retrieval_bandwidth": int(retrieval_bandwidth),
        "excerpt_pack": {
            "mode": "full_anchor_text_v1",
            "anchor_expansion": {
                "fill_gaps_up_to": int(DEFAULT_ANCHOR_GAP_FILL_UP_TO),
                "neighbor_pad": int(DEFAULT_ANCHOR_NEIGHBOR_PAD),
            },
        },
        "base_rate_prompt": str(base_rate_prompt_path),
        "spread_prompts": [str(p) for p in spread_prompt_paths],
        "fee_prompts": [str(p) for p in fee_prompt_paths],
        "items": [],
    }

    item_errors: List[Dict[str, Any]] = []

    for item_id in selected_item_ids:
        item_dir = out_dir / "items" / item_id
        item_dir.mkdir(parents=True, exist_ok=True)

        item_record: Dict[str, Any] = {"item_id": item_id, "ok": False}
        try:
            catalog = load_anchor_catalog(paths, item_id)
            canonical_text = prompt_view_path(paths, item_id).read_text(encoding="utf-8", errors="replace")
            selection_path = assert_exists(paths.run_dir / "indexing_v2" / f"{item_id}_anchors.json")
            selection_doc = json.loads(selection_path.read_text(encoding="utf-8"))
            selection_artifact = IndexingSelectionV2Artifact.model_validate(selection_doc)
            selection = selection_artifact.selection

            assert_exists(paths.run_dir / "retrieval_v2" / f"{item_id}_snippets.jsonl")

            # Base-rate pass
            base_anchor_ids = selection.base_rate_anchors
            if not base_anchor_ids:
                raise ContractIRFlowError(
                    f"{item_id}: indexing_v2 produced empty base_rate_anchors (cannot run base-rate pass)"
                )
            base_allowed = _expand_for_excerpt_pack(
                catalog=catalog,
                seed_anchor_ids=base_anchor_ids,
                fill_gaps_up_to=DEFAULT_ANCHOR_GAP_FILL_UP_TO,
                neighbor_pad=DEFAULT_ANCHOR_NEIGHBOR_PAD,
            )
            _write_json(
                item_dir / "base_rate" / "anchor_expansion.json",
                {
                    "seed_anchor_ids": base_anchor_ids,
                    "auto_added": [],
                    "expanded_anchor_ids": base_allowed,
                    "fill_gaps_up_to": int(DEFAULT_ANCHOR_GAP_FILL_UP_TO),
                    "neighbor_pad": int(DEFAULT_ANCHOR_NEIGHBOR_PAD),
                },
            )
            base_contexts = build_excerpt_pack_from_canonical(
                canonical_text=canonical_text, catalog=catalog, anchor_ids=base_allowed
            )
            base_task = (
                f"item_id: {item_id}\n"
                "Extract ALL benchmark/base-rate definitions present in CONTEXT.\n"
                "Do not encode spreads/margins/fees in this pass.\n"
            )
            base_dir = item_dir / "base_rate"
            base_dir.mkdir(parents=True, exist_ok=True)
            base_doc, base_metrics = _run_contractir_pass(
                complete_response_sync=complete_response_sync,
                pass_dir=base_dir,
                item_id=item_id,
                prompt_path=base_rate_prompt_path,
                task=base_task,
                contexts=base_contexts,
                allowed_anchor_ids=base_allowed,
                attempts=int(attempts),
                gateway_url=resolved_gateway_url,
                timeout_seconds=float(timeout_seconds),
                temperature=float(temperature),
                model=REQUIRED_MODEL,
                reasoning=REQUIRED_REASONING,
            )
            if base_doc is None:
                raise ContractIRFlowError(
                    f"{item_id}: base-rate pass failed after {base_metrics.get('attempts_used')} attempt(s)"
                )

            # Spread pass (try prompt strategies)
            spread_anchor_ids = selection.spread_anchors
            spread_doc: Optional[dict] = None
            spread_metrics: Dict[str, Any] = {
                "ok": True,
                "skipped": True,
                "reason": "indexing_v2 produced empty spread_anchors",
            }
            if spread_anchor_ids:
                spread_metrics = {"ok": False}
                spread_allowed = _expand_for_excerpt_pack(
                    catalog=catalog,
                    seed_anchor_ids=spread_anchor_ids,
                    fill_gaps_up_to=DEFAULT_ANCHOR_GAP_FILL_UP_TO,
                    neighbor_pad=DEFAULT_ANCHOR_NEIGHBOR_PAD,
                )
                _write_json(
                    item_dir / "spread" / "anchor_expansion.json",
                    {
                        "seed_anchor_ids": spread_anchor_ids,
                        "auto_added": [],
                        "expanded_anchor_ids": spread_allowed,
                        "fill_gaps_up_to": int(DEFAULT_ANCHOR_GAP_FILL_UP_TO),
                        "neighbor_pad": int(DEFAULT_ANCHOR_NEIGHBOR_PAD),
                    },
                )
                spread_contexts = build_excerpt_pack_from_canonical(
                    canonical_text=canonical_text, catalog=catalog, anchor_ids=spread_allowed
                )
                spread_task = (
                    f"item_id: {item_id}\n"
                    "Extract ALL spread/margin schedules present in CONTEXT.\n"
                    "Do not encode benchmark/base-rate definitions or fees in this pass.\n"
                )
                for pi, p in enumerate(spread_prompt_paths, start=1):
                    attempt_dir = item_dir / "spread" / f"prompt_{pi}"
                    attempt_dir.mkdir(parents=True, exist_ok=True)
                    spread_doc, spread_metrics = _run_contractir_pass(
                        complete_response_sync=complete_response_sync,
                        pass_dir=attempt_dir,
                        item_id=item_id,
                        prompt_path=p,
                        task=spread_task,
                        contexts=spread_contexts,
                        allowed_anchor_ids=spread_allowed,
                        attempts=int(attempts),
                        gateway_url=resolved_gateway_url,
                        timeout_seconds=float(timeout_seconds),
                        temperature=float(temperature),
                        model=REQUIRED_MODEL,
                        reasoning=REQUIRED_REASONING,
                    )
                    if spread_doc is not None:
                        _write_json(item_dir / "spread" / "contractir_validated.json", spread_doc)
                        break
                if spread_doc is None:
                    raise ContractIRFlowError(f"{item_id}: spread pass failed for all prompt strategies")

            # Fee pass (try prompt strategies)
            fee_anchor_ids = selection.fee_anchors
            fee_doc: Optional[dict] = None
            fee_metrics: Dict[str, Any] = {"ok": True, "skipped": True, "reason": "indexing_v2 produced empty fee_anchors"}
            if fee_anchor_ids:
                fee_metrics = {"ok": False}
                fee_allowed = _expand_for_excerpt_pack(
                    catalog=catalog,
                    seed_anchor_ids=fee_anchor_ids,
                    fill_gaps_up_to=DEFAULT_ANCHOR_GAP_FILL_UP_TO,
                    neighbor_pad=DEFAULT_ANCHOR_NEIGHBOR_PAD,
                )
                _write_json(
                    item_dir / "fee" / "anchor_expansion.json",
                    {
                        "seed_anchor_ids": fee_anchor_ids,
                        "auto_added": [],
                        "expanded_anchor_ids": fee_allowed,
                        "fill_gaps_up_to": int(DEFAULT_ANCHOR_GAP_FILL_UP_TO),
                        "neighbor_pad": int(DEFAULT_ANCHOR_NEIGHBOR_PAD),
                    },
                )
                fee_contexts = build_excerpt_pack_from_canonical(
                    canonical_text=canonical_text, catalog=catalog, anchor_ids=fee_allowed
                )
                fee_task = (
                    f"item_id: {item_id}\n"
                    "Extract ALL fee rate schedules present in CONTEXT.\n"
                    "Do not encode benchmark/base-rate definitions or spreads/margins in this pass.\n"
                )
                for pi, p in enumerate(fee_prompt_paths, start=1):
                    attempt_dir = item_dir / "fee" / f"prompt_{pi}"
                    attempt_dir.mkdir(parents=True, exist_ok=True)
                    fee_doc, fee_metrics = _run_contractir_pass(
                        complete_response_sync=complete_response_sync,
                        pass_dir=attempt_dir,
                        item_id=item_id,
                        prompt_path=p,
                        task=fee_task,
                        contexts=fee_contexts,
                        allowed_anchor_ids=fee_allowed,
                        attempts=int(attempts),
                        gateway_url=resolved_gateway_url,
                        timeout_seconds=float(timeout_seconds),
                        temperature=float(temperature),
                        model=REQUIRED_MODEL,
                        reasoning=REQUIRED_REASONING,
                    )
                    if fee_doc is not None:
                        _write_json(item_dir / "fee" / "contractir_validated.json", fee_doc)
                        break
                if fee_doc is None:
                    raise ContractIRFlowError(f"{item_id}: fee pass failed for all prompt strategies")

            # Deterministic merge (offline).
            docs_to_merge = [base_doc]
            if spread_doc is not None:
                docs_to_merge.append(spread_doc)
            if fee_doc is not None:
                docs_to_merge.append(fee_doc)

            merged = merge_contract_ir_v0_2(docs_to_merge)
            _write_json(item_dir / "contractir_merged.json", merged)

            item_record.update(
                {
                    "ok": True,
                    "base_rate": base_metrics,
                    "spread": spread_metrics,
                    "fee": fee_metrics,
                }
            )

        except Exception as exc:
            item_record["ok"] = False
            item_record["error"] = str(exc)
            item_errors.append({"item_id": item_id, "error": str(exc)})

        run_summary["items"].append(item_record)

    summary_path = out_dir / "summary.json"
    _write_json(summary_path, run_summary)

    if fail_on_item_errors and item_errors:
        raise ContractIRFlowError(f"{len(item_errors)} item(s) failed; see {summary_path}")

    return summary_path
