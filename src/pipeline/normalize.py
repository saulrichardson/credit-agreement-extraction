from __future__ import annotations

from pathlib import Path
from typing import Iterable, List

from bs4 import BeautifulSoup

from .config import Paths, record_manifest


def _canonicalize_html(html_path: Path) -> str:
    raw = html_path.read_text(errors="ignore")
    soup = BeautifulSoup(raw, "lxml")
    # Keep text-ish representation
    return soup.get_text("\n")


def build_prompt_views(paths: Paths, accessions: Iterable[str]) -> None:
    pv_root = paths.prompt_views_dir
    pv_root.mkdir(parents=True, exist_ok=True)
    accessions = list(accessions)
    if not accessions:
        raise RuntimeError("No accessions provided to normalize.")

    for acc in accessions:
        html_file = paths.ingest_dir / f"{acc}_EX-10.html"
        if not html_file.exists():
            raise FileNotFoundError(f"Expected HTML missing: {html_file}")
        out_dir = pv_root / acc
        out_dir.mkdir(parents=True, exist_ok=True)

        canonical_text = _canonicalize_html(html_file)
        (out_dir / "canonical.txt").write_text(canonical_text)
        (out_dir / "prompt_view.txt").write_text(canonical_text)
        # Minimal anchors TSV header; real anchors should be produced by downstream indexing
        (out_dir / "anchors.tsv").write_text("anchor_id\tanchor_type\tstart\tend\tlabel\n")

    # augment manifest with prompt_view paths
    manifest_path = paths.manifest_path
    if manifest_path.exists():
        import json

        manifest = json.loads(manifest_path.read_text())
        manifest["prompt_views"] = {
            acc: str(paths.prompt_views_dir / acc / "prompt_view.txt") for acc in accessions
        }
        record_manifest(manifest_path, manifest)

