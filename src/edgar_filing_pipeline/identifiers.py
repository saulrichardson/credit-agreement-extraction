from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Dict

SEGMENT_ID_DELIMITER = "::"


def _normalize_member_name(member_name: str) -> str:
    return member_name.strip()


@dataclass(frozen=True)
class SegmentKey:
    """Uniform identifier for a specific EDGAR segment."""

    tarfile: str
    member_name: str
    segment_no: int

    def as_string(self) -> str:
        return SEGMENT_ID_DELIMITER.join(
            [self.tarfile, _normalize_member_name(self.member_name), str(self.segment_no)]
        )

    def digest(self) -> str:
        payload = self.as_string().encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    @property
    def id(self) -> str:  # pragma: no cover - simple alias
        return self.as_string()

    @classmethod
    def from_manifest_row(cls, row: Dict[str, Any]) -> "SegmentKey":
        tarfile = row.get("tarfile")
        member_name = row.get("file") or row.get("file_x")
        segment_no = row.get("segment_no") or row.get("segment_no_x")
        if tarfile is None or member_name is None or segment_no is None:
            raise ValueError("manifest row missing tarfile/file/segment_no fields")
        return cls(str(tarfile), str(member_name), int(segment_no))


def build_segment_id(tarfile: str, member_name: str, segment_no: int) -> str:
    return SegmentKey(tarfile, member_name, segment_no).id


def build_segment_digest(tarfile: str, member_name: str, segment_no: int) -> str:
    return SegmentKey(tarfile, member_name, segment_no).digest()
