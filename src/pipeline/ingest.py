from __future__ import annotations

import tarfile
from pathlib import Path
from typing import Iterable, List, Tuple

from bs4 import BeautifulSoup

from .config import FilterSpec, Paths, record_manifest
from .filters import serialize_filter_spec


def _member_matches_accessions(member_name: str, accessions: List[str]) -> Tuple[bool, str | None]:
    for acc in accessions:
        if acc.replace("-", "") in member_name:
            return True, acc
    return False, None


def _is_ex10_like(name: str) -> bool:
    lowered = name.lower()
    return "ex-10" in lowered or "ex10" in lowered or "exhibit10" in lowered


def ingest_tarballs(
    paths: Paths,
    tarballs: Iterable[Path],
    filters: FilterSpec,
    accessions: List[str],
) -> List[str]:
    out_dir = paths.ingest_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    collected: List[str] = []
    for tarball in tarballs:
        if not tarball.exists():
            raise FileNotFoundError(f"Tarball not found: {tarball}")
        with tarfile.open(tarball, "r:*") as tf:
            members = tf.getmembers()
            for m in members:
                if not m.isfile():
                    continue
                matched, acc = _member_matches_accessions(m.name, accessions)
                if not matched:
                    continue
                if filters.exhibit_glob and not _is_ex10_like(m.name):
                    continue
                # extract file content
                content = tf.extractfile(m)
                if content is None:
                    continue
                text = content.read()
                # basic sanity: ensure it looks like HTML/text
                try:
                    decoded = text.decode("utf-8", errors="ignore")
                except Exception as exc:  # pragma: no cover
                    raise RuntimeError(f"Failed to decode {m.name}: {exc}") from exc
                # quick check for exhibit text
                if "<html" not in decoded.lower():
                    # attempt to wrap in html for downstream tools
                    decoded = f"<html><body><pre>{decoded}</pre></body></html>"
                out_path = out_dir / f"{acc}_EX-10.html"
                out_path.write_text(decoded)
                collected.append(acc)
    if not collected:
        raise RuntimeError("No matching EX-10 exhibits were extracted with the provided filters/accessions.")
    # Deduplicate accessions in manifest
    collected = sorted(set(collected))
    manifest = {
        "run_id": paths.run_id,
        "tarballs": [str(p) for p in tarballs],
        "filters": serialize_filter_spec(filters),
        "accessions": [
            {"accession": acc, "html": str(paths.ingest_dir / f"{acc}_EX-10.html")}
            for acc in collected
        ],
    }
    record_manifest(paths.manifest_path, manifest)
    return collected

