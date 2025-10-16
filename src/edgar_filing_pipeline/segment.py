from __future__ import annotations

import os
import re
import tarfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

from bs4 import BeautifulSoup


def create_segments(text: str) -> List[Tuple[int, int, int, int]]:
    """Locate the ``<DOCUMENT>``/``<TEXT>`` boundaries inside an .nc file."""
    doc_matches = list(re.finditer(r"<DOCUMENT>", text))
    start_matches = list(re.finditer(r"<TEXT>", text))
    end_matches = list(re.finditer(r"</TEXT>", text))

    if not doc_matches or not start_matches or not end_matches:
        return []

    start_doc_ub = [m.end() for m in doc_matches]
    start_text_lb = [m.start() for m in start_matches]
    start_text_ub = [m.end() for m in start_matches]
    end_text_lb = [m.start() for m in end_matches]

    n_doc = len(doc_matches)
    n_start = len(start_matches)
    n_end = len(end_matches)

    if n_doc < n_end:
        start_doc_ub = end_text_lb.copy()
    if n_start < n_doc:
        start_text_lb = [0] + start_doc_ub[:-1]
        start_text_ub = start_doc_ub.copy()
    if n_doc < n_start:
        start_doc_ub = start_text_lb.copy()
    if n_end < n_doc:
        end_text_lb = start_doc_ub.copy()

    return [tup for tup in zip(start_doc_ub, start_text_lb, start_text_ub, end_text_lb)]


def get_header_dict(header_text: str) -> Dict[str, str]:
    matches = re.findall(r"<([-a-zA-Z0-9]+)>(.*)\n", header_text)
    return {key.lower(): val for key, val in matches if val}


class SegmentExtractor:
    """Minimal segment parser for EDGAR ``.nc`` documents."""

    def __init__(self, text: str):
        self.text = text
        self.segment_positions = create_segments(text)

    def __len__(self) -> int:
        return len(self.segment_positions)

    def get_segment_header(self, index: int) -> Dict[str, str]:
        head_start, head_end, _, _ = self.segment_positions[index]
        return get_header_dict(self.text[head_start:head_end])

    def get_segment_html(self, index: int) -> str:
        _, _, _, body_end = self.segment_positions[index]
        _, _, body_start, _ = self.segment_positions[index]
        return self.text[body_start:body_end]

    def get_segment_text(
        self,
        index: int,
        *,
        convert_html: bool = False,
        replace_tables_with_markers: bool = False,
    ) -> Tuple[str, Dict[str, str]]:
        """Return the segment text and an optional table dictionary."""
        html = self.get_segment_html(index)
        if not convert_html:
            return html, {}

        soup = BeautifulSoup(html, "lxml")
        table_dict: Dict[str, str] = {}
        if replace_tables_with_markers:
            tables = soup.find_all("table")
            for idx, table in enumerate(tables):
                key = f"__TABLE_{idx}__"
                table_dict[key] = table.prettify()
                table.replace_with(f"\n{key}\n")
        text = soup.get_text(separator="\n").strip()
        return text, table_dict

    def has_tables(self, index: int) -> bool:
        html = self.get_segment_html(index).lower()
        return "<table" in html and "</table" in html


@dataclass
class TarSegment:
    tarfile: str
    member_name: str
    segment_index: int
    html: str
    header: Dict[str, str]
    has_table: bool


class TarSegmentReader:
    """Iterate segments from an EDGAR ``.nc.tar.gz`` archive."""

    def __init__(self, tar_path: os.PathLike[str] | str, *, encoding: str = "utf-8"):
        self.tar_path = Path(tar_path)
        self.encoding = encoding
        self._tar = tarfile.open(self.tar_path, "r:gz")
        self._members = {
            os.path.basename(member.name): member
            for member in self._tar.getmembers()
            if member.isfile()
        }

    def list_members(self) -> Iterable[str]:
        return self._members.keys()

    def read_member(self, member_name: str) -> str:
        member = self._members.get(member_name)
        if member is None:
            raise FileNotFoundError(
                f"{member_name} not found in tar archive {self.tar_path}"
            )
        data = self._tar.extractfile(member).read()
        return data.decode(self.encoding, errors="ignore")

    def iter_segments(self, member_name: str) -> Iterable[TarSegment]:
        raw_text = self.read_member(member_name)
        extractor = SegmentExtractor(raw_text)

        for idx in range(len(extractor)):
            header = extractor.get_segment_header(idx)
            html = extractor.get_segment_html(idx)
            has_table = extractor.has_tables(idx)
            yield TarSegment(
                tarfile=self.tar_path.name,
                member_name=member_name,
                segment_index=idx,
                html=html,
                header=header,
                has_table=has_table,
            )

    def close(self) -> None:
        self._tar.close()
