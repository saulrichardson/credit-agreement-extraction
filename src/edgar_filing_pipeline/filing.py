from __future__ import annotations

import json
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import pandas as pd
from bs4 import BeautifulSoup


@dataclass
class EdgarFiling:
    """Representation of a single EDGAR segment.

    Parameters
    ----------
    html_path:
        Path pointing to the saved segment HTML.
    metadata:
        Arbitrary metadata collected during extraction (e.g. tarfile,
        segment number, document type).  Stored verbatim and echoed in
        :meth:`to_dict`.
    """

    html_path: Path
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.html_path = Path(self.html_path)
        if "file_path" not in self.metadata:
            self.metadata["file_path"] = str(self.html_path)

    @cached_property
    def html(self) -> str:
        return self.html_path.read_text(encoding="utf-8", errors="ignore")

    @cached_property
    def soup(self) -> BeautifulSoup:
        return BeautifulSoup(self.html, "lxml")

    @cached_property
    def text(self) -> str:
        return self.soup.get_text(separator="\n").strip()

    @cached_property
    def tables(self) -> list[pd.DataFrame]:
        try:
            return pd.read_html(self.html)
        except ValueError:
            return []

    def tables_as_markdown(self) -> Iterable[str]:
        for table in self.tables:
            yield table.to_markdown(index=False)

    def to_dict(self) -> Dict[str, Any]:
        payload = dict(self.metadata)
        payload.update(
            {
                "html_path": str(self.html_path),
                "num_chars": len(self.html),
                "num_tables": len(self.tables),
            }
        )
        return payload

    def to_json(self) -> str:
        return json.dumps(self.to_dict())

    def regex_search(self, pattern: str, flags: int = 0):
        import re

        return list(re.finditer(pattern, self.text, flags))
