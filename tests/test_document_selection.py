from __future__ import annotations

import json
from pathlib import Path

import click
import pytest

from pipeline.cli.common import load_accessions_and_selection
from pipeline.evidence.selection import build_document_selection, serialize_document_selection


def _write_json(path: Path, payload: object) -> Path:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def test_build_document_selection_from_item_ids_file_normalizes_sequence(tmp_path: Path) -> None:
    dataset_path = _write_json(
        tmp_path / "dataset.json",
        {"item_ids": ["0001111111-23-000001_02"]},
    )

    selection, predicate = build_document_selection(
        item_ids_file=str(dataset_path),
        doc_type_prefixes=(),
    )

    assert selection.item_ids_source == str(dataset_path)
    assert selection.item_ids == ("0001111111-23-000001_02",)
    assert predicate(
        {"accession": "0001111111-23-000001"},
        {"sequence": "2", "type": "EX-10.1"},
    )
    assert not predicate(
        {"accession": "0001111111-23-000001"},
        {"sequence": "3", "type": "EX-10.1"},
    )


def test_build_document_selection_doc_type_prefixes_only() -> None:
    selection, predicate = build_document_selection(
        item_ids_file=None,
        doc_type_prefixes=("ex-10", "EX-10"),
    )

    assert selection.item_ids == ()
    assert selection.doc_type_prefixes == ("EX-10",)
    assert predicate({}, {"type": "EX-10.2"})
    assert not predicate({}, {"type": "EX-99"})


def test_build_document_selection_applies_and_semantics(tmp_path: Path) -> None:
    dataset_path = _write_json(
        tmp_path / "dataset.json",
        {"item_ids": ["0001111111-23-000001_2"]},
    )
    _selection, predicate = build_document_selection(
        item_ids_file=str(dataset_path),
        doc_type_prefixes=("EX-10",),
    )

    assert predicate(
        {"accession": "0001111111-23-000001"},
        {"sequence": "2", "type": "EX-10.3"},
    )
    assert not predicate(
        {"accession": "0001111111-23-000001"},
        {"sequence": "2", "type": "EX-99"},
    )
    assert not predicate(
        {"accession": "0001111111-23-000002"},
        {"sequence": "2", "type": "EX-10.3"},
    )


def test_serialize_document_selection_allow_all() -> None:
    selection, _predicate = build_document_selection(item_ids_file=None, doc_type_prefixes=())
    serialized = serialize_document_selection(selection)
    assert serialized["mode"] == "allow_all"
    assert serialized["item_ids"] == []
    assert serialized["doc_type_prefixes"] == []


def test_load_accessions_and_selection_requires_explicit_selector() -> None:
    with pytest.raises(click.UsageError):
        load_accessions_and_selection(
            accessions_file=None,
            item_ids_file=None,
            doc_type_prefixes=(),
        )

