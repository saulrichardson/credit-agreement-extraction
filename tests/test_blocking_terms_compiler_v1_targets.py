from __future__ import annotations

import pytest

from pipeline.blocking_terms_compiler_v1 import _extract_blocking_term_targets


def test_extract_blocking_term_targets_dedupes_and_sorts():
    doc = {
        "unresolved_dependencies": [
            {"term": "EBITDA", "referenced_by": ["Leverage Ratio"]},
            {"term": "EBITDA", "referenced_by": ["Interest Coverage", "Leverage Ratio"]},
            {"term": " Total Debt ", "referenced_by": ["Leverage Ratio", ""]},
            {"term": "", "referenced_by": ["X"]},
            {"term": None, "referenced_by": ["Y"]},
            {"term": "Bad Row", "referenced_by": "not-a-list"},
        ]
    }

    targets = _extract_blocking_term_targets(doc)
    assert [t.term for t in targets] == ["EBITDA", "Total Debt"]
    assert targets[0].referenced_by == ["Interest Coverage", "Leverage Ratio"]
    assert targets[1].referenced_by == ["Leverage Ratio"]


def test_extract_blocking_term_targets_requires_unresolved_dependencies():
    with pytest.raises(RuntimeError, match="missing unresolved_dependencies"):
        _extract_blocking_term_targets({})

