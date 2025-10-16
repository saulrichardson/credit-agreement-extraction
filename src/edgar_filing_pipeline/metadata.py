from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, List, Optional

import pandas as pd


class MetadataStore:
    """Simple wrapper around a Parquet/CSV metadata table."""

    def __init__(self, store_path: Path | str):
        self.store_path = Path(store_path)
        if self.store_path.suffix not in {".parquet", ".csv"}:
            raise ValueError("metadata file must be .parquet or .csv")
        self._df = pd.DataFrame()
        if self.store_path.exists():
            self._load()

    @property
    def df(self) -> pd.DataFrame:
        return self._df.copy()

    def _load(self) -> None:
        if self.store_path.suffix == ".parquet":
            self._df = pd.read_parquet(self.store_path)
        else:
            self._df = pd.read_csv(self.store_path)

    def save(self) -> None:
        self.store_path.parent.mkdir(parents=True, exist_ok=True)
        if self.store_path.suffix == ".parquet":
            self._df.to_parquet(self.store_path, index=False)
        else:
            self._df.to_csv(self.store_path, index=False)

    def add_records(self, records: Iterable[dict]) -> None:
        df = pd.DataFrame(list(records))
        if df.empty:
            return
        if self._df.empty:
            self._df = df
        else:
            self._df = pd.concat([self._df, df], ignore_index=True)

    def query(self, **filters) -> pd.DataFrame:
        df = self._df
        for key, value in filters.items():
            if key not in df.columns:
                raise KeyError(f"{key} not present in metadata columns")
            if isinstance(value, (list, tuple, set)):
                df = df[df[key].isin(value)]
            else:
                df = df[df[key] == value]
        return df.copy()

    def to_json(self) -> str:
        return json.dumps(self._df.to_dict(orient="records"))
