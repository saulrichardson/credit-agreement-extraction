from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Callable, Dict, Optional

BINARY_DOC_TYPE_KEYWORDS = (
    "GRAPHIC",
    "IMAGE",
    "PDF",
    "PNG",
    "JPG",
    "JPEG",
    "GIF",
    "TIFF",
    "TIF",
)

BINARY_DOC_TYPE_EXACT = {
    "EX-99.G",
    "EX-99.H",
    "EX-99.J",
    "EX-99.K",
    "EX-99.P",
    "EX-99.Q",
    "EX-99.G1",
    "EX-99.G2",
    "EX-99.H1",
}

BINARY_FILENAME_SUFFIXES = {
    ".PDF",
    ".PNG",
    ".JPG",
    ".JPEG",
    ".GIF",
    ".TIFF",
    ".TIF",
    ".BMP",
    ".SVG",
}

_TAG_RE = re.compile(r"<[^>]+>")
_ALNUM_RE = re.compile(r"[A-Za-z0-9]")

DOC_TYPE_KEYS = (
    "doc_type",
    "type",
    "document_type",
    "document-type",
)

FilterContext = Dict[str, Any]
FilterFn = Callable[[Dict[str, str], str, Optional[FilterContext]], Optional[str]]
FilterContextBuilder = Callable[[Dict[str, Any]], Optional[FilterContext]]


def default_filter(header: Dict[str, str], html: str, context: Optional[FilterContext] = None) -> Optional[str]:
    """Default filter that mirrors the historical binary detection behaviour."""
    return detect_binary_segment(header, html)


def extract_doc_type(header: Dict[str, str]) -> Optional[str]:
    for key in DOC_TYPE_KEYS:
        value = header.get(key)
        if value:
            return value
    return None


def detect_binary_by_doc_type(doc_type: Optional[str]) -> Optional[str]:
    if not doc_type:
        return None
    value = doc_type.strip().upper()
    if value in BINARY_DOC_TYPE_EXACT:
        return f"doc_type:{value}"
    if any(keyword in value for keyword in BINARY_DOC_TYPE_KEYWORDS):
        return f"doc_type:{value}"
    return None


def detect_binary_by_header_filename(header: Dict[str, str]) -> Optional[str]:
    for key in ("filename", "file", "name"):
        candidate = header.get(key)
        if not candidate:
            continue
        suffix = Path(candidate).suffix.upper()
        if suffix in BINARY_FILENAME_SUFFIXES:
            return f"filename:{candidate}"
    return None


def _counts_as_image_only(html: str) -> bool:
    without_tags = _TAG_RE.sub(" ", html)
    alnum_chars = _ALNUM_RE.findall(without_tags)
    return len(alnum_chars) < 32


def detect_binary_content(html: str) -> Optional[str]:
    stripped = html.lstrip()
    lowered = stripped.lower()
    if stripped.startswith("%pdf"):
        return "content:pdf_header"
    if "application/pdf" in lowered:
        return "content:pdf_reference"
    if "<object" in lowered or "<embed" in lowered or "<iframe" in lowered:
        return "content:embedded_object"
    if "data:application/pdf" in lowered or "data:image" in lowered:
        return "content:data_uri"
    if "content-transfer-encoding: base64" in lowered or "begin base64" in lowered:
        return "content:base64_attachment"
    if "<img" in lowered and _counts_as_image_only(stripped):
        return "content:image_only"
    return None


def detect_binary_segment(
    header: Dict[str, str],
    html: str,
) -> Optional[str]:
    doc_type = extract_doc_type(header)
    doc_type_reason = detect_binary_by_doc_type(doc_type)
    if doc_type_reason:
        return doc_type_reason

    filename_reason = detect_binary_by_header_filename(header)
    if filename_reason:
        return filename_reason

    content_reason = detect_binary_content(html)
    if content_reason:
        return content_reason

    return None
