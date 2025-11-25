from __future__ import annotations

import tarfile
from pathlib import Path
from typing import Iterable, List, Tuple, Dict, Any, Callable, Optional

from .config import FilterSpec, Paths, record_manifest
from .filters import serialize_filter_spec
from .utils import safe_item_id


def _member_matches_accessions(member_name: str, accessions: List[str]) -> Tuple[bool, str | None]:
    def _norm(s: str) -> str:
        return s.replace("-", "").replace("_", "")

    for acc in accessions:
        if _norm(acc) in _norm(member_name):
            return True, acc
    return False, None


def _iter_nc_members(tf: tarfile.TarFile, accessions: List[str]) -> Iterable[Tuple[tarfile.TarInfo, str | None]]:
    """Yield .nc members that optionally match provided accessions."""
    for member in tf.getmembers():
        if not (member.isfile() and member.name.lower().endswith(".nc")):
            continue
        if accessions:
            matched, acc = _member_matches_accessions(member.name, accessions)
            if not matched:
                continue
        else:
            acc = None
        yield member, acc


def _parse_submission(text: str) -> Dict[str, Any]:
    """Minimal SGML parser for .nc submissions."""

    def _find_first(tag: str) -> str | None:
        marker = f"<{tag}>"
        idx = text.find(marker)
        if idx == -1:
            return None
        start = idx + len(marker)
        end = text.find("\n", start)
        if end == -1:
            end = len(text)
        return text[start:end].strip()

    accession = _find_first("ACCESSION-NUMBER")
    cik = _find_first("CIK")
    form_type = _find_first("TYPE")
    filing_date = _find_first("FILING-DATE")

    documents: List[Dict[str, Any]] = []
    parts = text.split("<DOCUMENT>")
    for part in parts[1:]:
        doc_section, *_ = part.split("</DOCUMENT>", 1)
        doc_type = None
        filename = None
        seq = None
        for line in doc_section.splitlines():
            if line.startswith("<TYPE>") and doc_type is None:
                doc_type = line.replace("<TYPE>", "", 1).strip()
            elif line.startswith("<SEQUENCE>") and seq is None:
                seq = line.replace("<SEQUENCE>", "", 1).strip()
            elif line.startswith("<FILENAME>") and filename is None:
                filename = line.replace("<FILENAME>", "", 1).strip()
            if doc_type and filename and seq:
                break
        content = None
        if "<TEXT>" in doc_section:
            after = doc_section.split("<TEXT>", 1)[1]
            content = after.split("</TEXT>", 1)[0]
        documents.append({
            "type": doc_type,
            "filename": filename,
            "sequence": seq,
            "content": content,
        })

    return {
        "accession": accession,
        "cik": cik,
        "form_type": form_type,
        "filing_date": filing_date,
        "documents": documents,
    }


def _doc_type_matches(doc_type: str | None, exhibit_types: List[str] | None, exhibit_glob: str | None) -> bool:
    # If no filters provided, accept all document types.
    if not exhibit_types and not exhibit_glob:
        return True
    if doc_type is None:
        return False
    upper = doc_type.upper()
    if exhibit_types:
        for pat in exhibit_types:
            if upper.startswith(pat.upper()):
                return True
    if exhibit_glob:
        return exhibit_glob.lower() in upper.lower()
    return False


def ingest_tarballs(
    paths: Paths,
    tarballs: Iterable[Path],
    filters: FilterSpec,
    accessions: List[str] | None,
    doc_filter: Callable[[Dict[str, Any], Dict[str, Any]], bool],
) -> List[str]:
    out_dir = paths.ingest_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    collected: List[Dict[str, Any]] = []  # per accession
    items: List[Dict[str, Any]] = []      # flat per-document
    accessions = accessions or []
    for tarball in tarballs:
        if not tarball.exists():
            raise FileNotFoundError(f"Tarball not found: {tarball}")
        with tarfile.open(tarball, "r:*") as tf:
            for m, acc_from_list in _iter_nc_members(tf, accessions):
                content = tf.extractfile(m)
                if content is None:
                    continue
                text = content.read()
                try:
                    decoded = text.decode("utf-8", errors="ignore")
                except Exception as exc:  # pragma: no cover
                    raise RuntimeError(f"Failed to decode {m.name}: {exc}") from exc

                submission = _parse_submission(decoded)
                acc = submission.get("accession") or acc_from_list
                if not acc:
                    continue

                cik = submission.get("cik")
                form_type = submission.get("form_type")
                filing_date = submission.get("filing_date")

                docs: List[Dict[str, Any]] = []
                matched_idx = 0
                for doc in submission.get("documents", []):
                    if not doc_filter(submission, doc):
                        continue
                    matched_idx += 1
                    content = doc.get("content") or ""
                    html = content
                    if "<html" not in html.lower():
                        html = f"<html><body><pre>{html}</pre></body></html>"
                    seq = doc.get("sequence") or f"{matched_idx:02d}"
                    fname = doc.get("filename") or f"EX-10-{seq}.html"
                    item_id = safe_item_id(acc, seq, fallback_idx=matched_idx)
                    safe_name = f"{item_id}.html"
                    out_path = out_dir / safe_name
                    out_path.write_text(html)
                    docs.append(
                        {
                            "type": doc.get("type"),
                            "filename": fname,
                            "sequence": seq,
                            "path": str(out_path),
                            "primary": False,  # may be unused downstream
                            "item_id": item_id,
                        }
                    )
                if docs:
                    # choose primary deterministically: lowest sequence number
                    try:
                        primary_doc = min(docs, key=lambda d: int(d.get("sequence") or 1_000_000))
                    except ValueError:
                        primary_doc = docs[0]
                    for d in docs:
                        d["primary"] = d is primary_doc
                    collected.append(
                        {
                            "accession": acc,
                            "cik": cik,
                            "form_type": form_type,
                            "filing_date": filing_date,
                            "documents": docs,
                        }
                    )
                    for d in docs:
                        items.append(
                            {
                                "item_id": d["item_id"],
                                "accession": acc,
                                "sequence": d.get("sequence"),
                                "filename": d.get("filename"),
                                "path": d.get("path"),
                                "primary": d.get("primary"),
                            }
                        )

    if not collected:
        raise RuntimeError("No matching EX-10 exhibits were extracted with the provided filters/accessions.")
    dedup: Dict[str, Dict[str, Any]] = {}
    for entry in collected:
        acc = entry["accession"]
        if acc not in dedup:
            dedup[acc] = entry
    manifest = {
        "run_id": paths.run_id,
        "tarballs": [str(p) for p in tarballs],
        "filters": serialize_filter_spec(filters),
        "accessions": list(dedup.values()),
        "items": items,
    }
    record_manifest(paths.manifest_path, manifest)
    return sorted(dedup.keys())
