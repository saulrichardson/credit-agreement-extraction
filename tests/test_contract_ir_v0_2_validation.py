from __future__ import annotations

from pipeline.contract_ir_v0_2 import validate_contract_ir


def _minimal_contract_ir(*, source_anchor_ids: list[str], derived_source_refs: list[str]):
    return {
        "schema_version": "contract_ir_v0_2",
        "contract_id": "contract_1",
        "sources": [
            {
                "source_id": "S1",
                "kind": "excerpt_pack",
                "item_id": "contract_1",
                "anchor_ids": source_anchor_ids,
                "notes": None,
            }
        ],
        "indices": [],
        "tables": [],
        "derived": [
            {
                "fn_id": "ExampleRate",
                "semantic_role": "spread",
                "description": None,
                "args": [],
                "returns": "rate",
                "expr": {"lit": {"type": "rate", "value": "0.01"}},
                "source_refs": derived_source_refs,
            }
        ],
        "open_items": [],
    }


def test_contract_ir_source_refs_must_be_in_sources_anchor_ids():
    doc = _minimal_contract_ir(source_anchor_ids=["A0001"], derived_source_refs=["A0001"])
    errors = validate_contract_ir(doc)
    assert errors == []


def test_contract_ir_rejects_source_refs_not_in_sources():
    doc = _minimal_contract_ir(source_anchor_ids=["A0001"], derived_source_refs=["A9999"])
    errors = validate_contract_ir(doc)
    codes = {e.code for e in errors}
    assert "anchor_not_in_context" in codes


def test_contract_ir_rejects_mis_scaled_rate_literals() -> None:
    doc = _minimal_contract_ir(source_anchor_ids=["A0001"], derived_source_refs=["A0001"])
    # Wrong: 2.25% must be encoded as 0.0225, not 2.25.
    doc["derived"][0]["expr"]["lit"]["value"] = "2.25"
    errors = validate_contract_ir(doc)
    codes = {e.code for e in errors}
    assert "rate_literal_magnitude" in codes


def test_contract_ir_lookup_rule_requires_bool_predicate_column():
    doc = {
        "schema_version": "contract_ir_v0_2",
        "contract_id": "contract_1",
        "sources": [
            {
                "source_id": "S1",
                "kind": "excerpt_pack",
                "item_id": "contract_1",
                "anchor_ids": ["A0001"],
                "notes": None,
            }
        ],
        "indices": [],
        "tables": [
            {
                "table_id": "T",
                "description": None,
                "columns": [
                    {"name": "predicate", "type": "string"},
                    {"name": "v", "type": "bps"},
                ],
                "rows": [
                    {
                        "row_id": None,
                        "cells": {
                            "predicate": {"lit": {"type": "string", "value": "x > 1.0"}},
                            "v": {"lit": {"type": "bps", "value": "25"}},
                        },
                        "source_refs": ["A0001"],
                    }
                ],
            }
        ],
        "derived": [
            {
                "fn_id": "F",
                "semantic_role": "spread",
                "description": None,
                "args": [{"name": "x", "type": "decimal"}],
                "returns": "rate",
                "expr": {
                    "op": "bps_to_rate",
                    "args": [
                        {
                            "op": "lookup_rule",
                            "args": [
                                {"lit": {"type": "string", "value": "T"}},
                                {"lit": {"type": "string", "value": "predicate"}},
                                {"lit": {"type": "string", "value": "v"}},
                            ],
                        }
                    ],
                },
                "source_refs": ["A0001"],
            }
        ],
        "open_items": [],
    }
    errors = validate_contract_ir(doc)
    codes = {e.code for e in errors}
    assert "lookup_rule_predicate_not_bool" in codes


def test_contract_ir_lookup_rule_requires_predicate_and_value_cells_in_each_row():
    doc = {
        "schema_version": "contract_ir_v0_2",
        "contract_id": "contract_1",
        "sources": [
            {
                "source_id": "S1",
                "kind": "excerpt_pack",
                "item_id": "contract_1",
                "anchor_ids": ["A0001"],
                "notes": None,
            }
        ],
        "indices": [],
        "tables": [
            {
                "table_id": "T",
                "description": None,
                "columns": [
                    {"name": "predicate", "type": "bool"},
                    {"name": "v", "type": "bps"},
                ],
                "rows": [
                    {
                        "row_id": None,
                        "cells": {
                            # Missing predicate cell.
                            "v": {"lit": {"type": "bps", "value": "25"}},
                        },
                        "source_refs": ["A0001"],
                    }
                ],
            }
        ],
        "derived": [
            {
                "fn_id": "F",
                "semantic_role": "spread",
                "description": None,
                "args": [{"name": "x", "type": "decimal"}],
                "returns": "bps",
                "expr": {
                    "op": "lookup_rule",
                    "args": [
                        {"lit": {"type": "string", "value": "T"}},
                        {"lit": {"type": "string", "value": "predicate"}},
                        {"lit": {"type": "string", "value": "v"}},
                    ],
                },
                "source_refs": ["A0001"],
            }
        ],
        "open_items": [],
    }

    errors = validate_contract_ir(doc)
    codes = {e.code for e in errors}
    assert "lookup_rule_predicate_cell_missing" in codes


def test_contract_ir_lookup_rule_requires_string_literal_args():
    doc = {
        "schema_version": "contract_ir_v0_2",
        "contract_id": "contract_1",
        "sources": [
            {
                "source_id": "S1",
                "kind": "excerpt_pack",
                "item_id": "contract_1",
                "anchor_ids": ["A0001"],
                "notes": None,
            }
        ],
        "indices": [],
        "tables": [
            {
                "table_id": "T",
                "description": None,
                "columns": [
                    {"name": "predicate", "type": "bool"},
                    {"name": "v", "type": "bps"},
                ],
                "rows": [
                    {
                        "row_id": None,
                        "cells": {
                            "predicate": {"lit": {"type": "bool", "value": True}},
                            "v": {"lit": {"type": "bps", "value": "25"}},
                        },
                        "source_refs": ["A0001"],
                    }
                ],
            }
        ],
        "derived": [
            {
                "fn_id": "F",
                "semantic_role": "spread",
                "description": None,
                "args": [{"name": "col", "type": "string"}],
                "returns": "bps",
                # Wrong: value_col must be a string literal node, not a var.
                "expr": {
                    "op": "lookup_rule",
                    "args": [
                        {"lit": {"type": "string", "value": "T"}},
                        {"lit": {"type": "string", "value": "predicate"}},
                        {"var": "col"},
                    ],
                },
                "source_refs": ["A0001"],
            }
        ],
        "open_items": [],
    }

    errors = validate_contract_ir(doc)
    codes = {e.code for e in errors}
    assert "lookup_rule_arg_not_string_literal" in codes


def test_contract_ir_var_not_declared_is_rejected():
    doc = {
        "schema_version": "contract_ir_v0_2",
        "contract_id": "contract_1",
        "sources": [
            {
                "source_id": "S1",
                "kind": "excerpt_pack",
                "item_id": "contract_1",
                "anchor_ids": ["A0001"],
                "notes": None,
            }
        ],
        "indices": [],
        "tables": [],
        "derived": [
            {
                "fn_id": "F",
                "semantic_role": "spread",
                "description": None,
                "args": [],
                "returns": "rate",
                "expr": {
                    "op": "add",
                    "args": [
                        {"var": "x"},
                        {"lit": {"type": "rate", "value": "0.01"}},
                    ],
                },
                "source_refs": ["A0001"],
            }
        ],
        "open_items": [],
    }

    errors = validate_contract_ir(doc)
    codes = {e.code for e in errors}
    assert "var_not_declared" in codes


def test_contract_ir_index_series_must_be_declared_in_indices():
    doc = {
        "schema_version": "contract_ir_v0_2",
        "contract_id": "contract_1",
        "sources": [
            {
                "source_id": "S1",
                "kind": "excerpt_pack",
                "item_id": "contract_1",
                "anchor_ids": ["A0001"],
                "notes": None,
            }
        ],
        "indices": [],
        "tables": [],
        "derived": [
            {
                "fn_id": "Prime",
                "semantic_role": "base_rate",
                "description": None,
                "args": [{"name": "date", "type": "date"}],
                "returns": "rate",
                "expr": {
                    "op": "index",
                    "args": [
                        {"lit": {"type": "string", "value": "PrimeRate"}},
                        {"var": "date"},
                    ],
                },
                "source_refs": ["A0001"],
            }
        ],
        "open_items": [],
    }

    errors = validate_contract_ir(doc)
    codes = {e.code for e in errors}
    assert "index_series_not_declared" in codes


def test_contract_ir_numeric_literals_must_be_plain_decimal_strings():
    doc = {
        "schema_version": "contract_ir_v0_2",
        "contract_id": "contract_1",
        "sources": [
            {
                "source_id": "S1",
                "kind": "excerpt_pack",
                "item_id": "contract_1",
                "anchor_ids": ["A0001"],
                "notes": None,
            }
        ],
        "indices": [],
        "tables": [
            {
                "table_id": "T",
                "description": None,
                "columns": [
                    {"name": "k", "type": "string"},
                    {"name": "v", "type": "rate"},
                ],
                "rows": [
                    {
                        "row_id": None,
                        "cells": {
                            "k": {"lit": {"type": "string", "value": "X"}},
                            # NOTE: The schema does not validate table cell literals; the hard gate does.
                            "v": {"lit": {"type": "rate", "value": "0.2000%"}},
                        },
                        "source_refs": ["A0001"],
                    }
                ],
            }
        ],
        "derived": [
            {
                "fn_id": "BadRate",
                "semantic_role": "spread",
                "description": None,
                "args": [],
                "returns": "rate",
                "expr": {
                    "op": "lookup",
                    "args": [
                        {"lit": {"type": "string", "value": "T"}},
                        {"lit": {"type": "string", "value": "k"}},
                        {"lit": {"type": "string", "value": "X"}},
                        {"lit": {"type": "string", "value": "v"}},
                    ],
                },
                "source_refs": ["A0001"],
            }
        ],
        "open_items": [],
    }

    errors = validate_contract_ir(doc)
    codes = {e.code for e in errors}
    assert "numeric_literal_format" in codes


def test_contract_ir_lookup_requires_referenced_columns_exist():
    doc = {
        "schema_version": "contract_ir_v0_2",
        "contract_id": "contract_1",
        "sources": [
            {
                "source_id": "S1",
                "kind": "excerpt_pack",
                "item_id": "contract_1",
                "anchor_ids": ["A0001"],
                "notes": None,
            }
        ],
        "indices": [],
        "tables": [
            {
                "table_id": "T",
                "description": None,
                "columns": [
                    {"name": "key", "type": "string"},
                    {"name": "v", "type": "bps"},
                ],
                "rows": [
                    {
                        "row_id": None,
                        "cells": {
                            "key": {"lit": {"type": "string", "value": "X"}},
                            "v": {"lit": {"type": "bps", "value": "25"}},
                        },
                        "source_refs": ["A0001"],
                    }
                ],
            }
        ],
        "derived": [
            {
                "fn_id": "F",
                "semantic_role": "spread",
                "description": None,
                "args": [],
                "returns": "bps",
                "expr": {
                    "op": "lookup",
                    "args": [
                        {"lit": {"type": "string", "value": "T"}},
                        {"lit": {"type": "string", "value": "missing_key"}},
                        {"lit": {"type": "string", "value": "X"}},
                        {"lit": {"type": "string", "value": "v"}},
                    ],
                },
                "source_refs": ["A0001"],
            }
        ],
        "open_items": [],
    }

    errors = validate_contract_ir(doc)
    codes = {e.code for e in errors}
    assert "lookup_key_col_missing" in codes


def test_contract_ir_lookup2_rejects_condition_string_keys():
    doc = {
        "schema_version": "contract_ir_v0_2",
        "contract_id": "contract_1",
        "sources": [
            {
                "source_id": "S1",
                "kind": "excerpt_pack",
                "item_id": "contract_1",
                "anchor_ids": ["A0001"],
                "notes": None,
            }
        ],
        "indices": [],
        "tables": [
            {
                "table_id": "T",
                "description": None,
                "columns": [
                    {"name": "ICR_condition", "type": "string"},
                    {"name": "FCCR_condition", "type": "string"},
                    {"name": "margin", "type": "rate"},
                ],
                "rows": [
                    {
                        "row_id": None,
                        "cells": {
                            "ICR_condition": {"lit": {"type": "string", "value": "< 1.75 : 1"}},
                            "FCCR_condition": {"lit": {"type": "string", "value": "< 1.20 : 1"}},
                            "margin": {"lit": {"type": "rate", "value": "0.025"}},
                        },
                        "source_refs": ["A0001"],
                    }
                ],
            }
        ],
        "derived": [
            {
                "fn_id": "M",
                "semantic_role": "spread",
                "description": None,
                "args": [{"name": "icr", "type": "string"}, {"name": "fccr", "type": "string"}],
                "returns": "rate",
                "expr": {
                    "op": "lookup2",
                    "args": [
                        {"lit": {"type": "string", "value": "T"}},
                        {"lit": {"type": "string", "value": "ICR_condition"}},
                        {"var": "icr"},
                        {"lit": {"type": "string", "value": "FCCR_condition"}},
                        {"var": "fccr"},
                        {"lit": {"type": "string", "value": "margin"}},
                    ],
                },
                "source_refs": ["A0001"],
            }
        ],
        "open_items": [],
    }

    errors = validate_contract_ir(doc)
    codes = {e.code for e in errors}
    assert "lookup_key_is_condition_string" in codes


def test_contract_ir_table_cells_must_be_ast_nodes_or_null():
    doc = {
        "schema_version": "contract_ir_v0_2",
        "contract_id": "contract_1",
        "sources": [
            {
                "source_id": "S1",
                "kind": "excerpt_pack",
                "item_id": "contract_1",
                "anchor_ids": ["A0001"],
                "notes": None,
            }
        ],
        "indices": [],
        "tables": [
            {
                "table_id": "T",
                "description": None,
                "columns": [
                    {"name": "bucket_label", "type": "string"},
                    {"name": "upper_cmp", "type": "string"},
                ],
                "rows": [
                    {
                        "row_id": None,
                        "cells": {
                            "bucket_label": {"lit": {"type": "string", "value": "X"}},
                            # Invalid: cell value is not an AST node.
                            "upper_cmp": "lt",
                        },
                        "source_refs": ["A0001"],
                    }
                ],
            }
        ],
        "derived": [
            {
                "fn_id": "F",
                "semantic_role": "spread",
                "description": None,
                "args": [],
                "returns": "rate",
                "expr": {"lit": {"type": "rate", "value": "0.01"}},
                "source_refs": ["A0001"],
            }
        ],
        "open_items": [],
    }

    errors = validate_contract_ir(doc)
    codes = {e.code for e in errors}
    assert "table_cell_not_ast" in codes


def test_contract_ir_table_non_bool_cells_must_be_literal_nodes():
    doc = {
        "schema_version": "contract_ir_v0_2",
        "contract_id": "contract_1",
        "sources": [
            {
                "source_id": "S1",
                "kind": "excerpt_pack",
                "item_id": "contract_1",
                "anchor_ids": ["A0001"],
                "notes": None,
            }
        ],
        "indices": [],
        "tables": [
            {
                "table_id": "T",
                "description": None,
                "columns": [
                    {"name": "k", "type": "string"},
                    {"name": "v", "type": "rate"},
                ],
                "rows": [
                    {
                        "row_id": None,
                        "cells": {
                            # Invalid: 'k' is string-typed column but cell is an expression AST node.
                            "k": {"op": "eq", "args": [{"var": "x"}, {"lit": {"type": "string", "value": "X"}}]},
                            "v": {"lit": {"type": "rate", "value": "0.01"}},
                        },
                        "source_refs": ["A0001"],
                    }
                ],
            }
        ],
        "derived": [
            {
                "fn_id": "F",
                "semantic_role": "spread",
                "description": None,
                "args": [{"name": "x", "type": "string"}],
                "returns": "rate",
                "expr": {"lit": {"type": "rate", "value": "0.01"}},
                "source_refs": ["A0001"],
            }
        ],
        "open_items": [],
    }

    errors = validate_contract_ir(doc)
    codes = {e.code for e in errors}
    assert "table_cell_not_literal" in codes


def test_contract_ir_operator_arity_is_validated():
    doc = {
        "schema_version": "contract_ir_v0_2",
        "contract_id": "contract_1",
        "sources": [
            {
                "source_id": "S1",
                "kind": "excerpt_pack",
                "item_id": "contract_1",
                "anchor_ids": ["A0001"],
                "notes": None,
            }
        ],
        "indices": [],
        "tables": [],
        "derived": [
            {
                "fn_id": "F",
                "semantic_role": "spread",
                "description": None,
                "args": [{"name": "a", "type": "string"}, {"name": "b", "type": "string"}],
                "returns": "rate",
                # Invalid: lookup2 expects 6 args, but only 5 provided.
                "expr": {
                    "op": "lookup2",
                    "args": [
                        {"lit": {"type": "string", "value": "T"}},
                        {"lit": {"type": "string", "value": "k1"}},
                        {"var": "a"},
                        {"lit": {"type": "string", "value": "k2"}},
                        {"var": "b"},
                    ],
                },
                "source_refs": ["A0001"],
            }
        ],
        "open_items": [],
    }

    errors = validate_contract_ir(doc)
    codes = {e.code for e in errors}
    assert "op_arity" in codes


def test_contract_ir_rejects_non_identifier_arg_names():
    doc = {
        "schema_version": "contract_ir_v0_2",
        "contract_id": "contract_1",
        "sources": [
            {
                "source_id": "S1",
                "kind": "excerpt_pack",
                "item_id": "contract_1",
                "anchor_ids": ["A0001"],
                "notes": None,
            }
        ],
        "indices": [],
        "tables": [],
        "derived": [
            {
                "fn_id": "F",
                "semantic_role": "spread",
                "description": None,
                "args": [{"name": "Index Rate", "type": "rate"}],
                "returns": "rate",
                "expr": {"var": "Index Rate"},
                "source_refs": ["A0001"],
            }
        ],
        "open_items": [],
    }

    errors = validate_contract_ir(doc)
    codes = {e.code for e in errors}
    assert "arg_name_invalid" in codes


def test_contract_ir_rejects_string_operands_in_numeric_ops():
    doc = {
        "schema_version": "contract_ir_v0_2",
        "contract_id": "contract_1",
        "sources": [
            {
                "source_id": "S1",
                "kind": "excerpt_pack",
                "item_id": "contract_1",
                "anchor_ids": ["A0001"],
                "notes": None,
            }
        ],
        "indices": [],
        "tables": [],
        "derived": [
            {
                "fn_id": "F",
                "semantic_role": "spread",
                "description": None,
                "args": [{"name": "x", "type": "string"}],
                "returns": "rate",
                "expr": {
                    "op": "add",
                    "args": [
                        {"var": "x"},
                        {"lit": {"type": "rate", "value": "0.01"}},
                    ],
                },
                "source_refs": ["A0001"],
            }
        ],
        "open_items": [],
    }

    errors = validate_contract_ir(doc)
    codes = {e.code for e in errors}
    assert "numeric_op_arg_not_numeric" in codes


def test_contract_ir_index_requires_date_typed_arg():
    doc = {
        "schema_version": "contract_ir_v0_2",
        "contract_id": "contract_1",
        "sources": [
            {
                "source_id": "S1",
                "kind": "excerpt_pack",
                "item_id": "contract_1",
                "anchor_ids": ["A0001"],
                "notes": None,
            }
        ],
        "indices": [{"series_id": "PrimeRate", "unit": "rate", "description": None}],
        "tables": [],
        "derived": [
            {
                "fn_id": "Prime",
                "semantic_role": "base_rate",
                "description": None,
                # Wrong: as_of_date is declared as string, but index() requires date.
                "args": [{"name": "as_of_date", "type": "string"}],
                "returns": "rate",
                "expr": {
                    "op": "index",
                    "args": [
                        {"lit": {"type": "string", "value": "PrimeRate"}},
                        {"var": "as_of_date"},
                    ],
                },
                "source_refs": ["A0001"],
            }
        ],
        "open_items": [],
    }

    errors = validate_contract_ir(doc)
    codes = {e.code for e in errors}
    assert "index_date_arg_not_date" in codes


def test_contract_ir_add_sub_requires_matching_kinds_when_inferable():
    doc = {
        "schema_version": "contract_ir_v0_2",
        "contract_id": "contract_1",
        "sources": [
            {
                "source_id": "S1",
                "kind": "excerpt_pack",
                "item_id": "contract_1",
                "anchor_ids": ["A0001"],
                "notes": None,
            }
        ],
        "indices": [{"series_id": "FederalFundsRate", "unit": "rate", "description": None}],
        "tables": [],
        "derived": [
            {
                "fn_id": "ABR",
                "semantic_role": "base_rate",
                "description": None,
                "args": [{"name": "date", "type": "date"}],
                "returns": "rate",
                "expr": {
                    "op": "add",
                    "args": [
                        {
                            "op": "index",
                            "args": [
                                {"lit": {"type": "string", "value": "FederalFundsRate"}},
                                {"var": "date"},
                            ],
                        },
                        # Wrong: decimal literal mixed with rate-producing index().
                        {"lit": {"type": "decimal", "value": "0.005"}},
                    ],
                },
                "source_refs": ["A0001"],
            }
        ],
        "open_items": [],
    }

    errors = validate_contract_ir(doc)
    codes = {e.code for e in errors}
    assert "add_sub_kind_mismatch" in codes


def test_contract_ir_lookup_range_requires_string_comparator_columns():
    doc = {
        "schema_version": "contract_ir_v0_2",
        "contract_id": "contract_1",
        "sources": [
            {
                "source_id": "S1",
                "kind": "excerpt_pack",
                "item_id": "contract_1",
                "anchor_ids": ["A0001"],
                "notes": None,
            }
        ],
        "indices": [],
        "tables": [
            {
                "table_id": "T",
                "description": None,
                "columns": [
                    {"name": "bucket_label", "type": "string"},
                    {"name": "lower_bound", "type": "decimal"},
                    # Wrong: comparator columns must be string, not decimal.
                    {"name": "lower_cmp", "type": "decimal"},
                    {"name": "upper_bound", "type": "decimal"},
                    {"name": "upper_cmp", "type": "decimal"},
                    {"name": "spread_bps", "type": "bps"},
                ],
                "rows": [
                    {
                        "row_id": None,
                        "cells": {
                            "bucket_label": {"lit": {"type": "string", "value": "X"}},
                            "lower_bound": None,
                            "lower_cmp": None,
                            "upper_bound": None,
                            "upper_cmp": None,
                            "spread_bps": {"lit": {"type": "bps", "value": "50"}},
                        },
                        "source_refs": ["A0001"],
                    }
                ],
            }
        ],
        "derived": [
            {
                "fn_id": "Spread",
                "semantic_role": "spread",
                "description": None,
                "args": [{"name": "x", "type": "decimal"}],
                "returns": "rate",
                "expr": {
                    "op": "bps_to_rate",
                    "args": [
                        {
                            "op": "lookup_range",
                            "args": [
                                {"lit": {"type": "string", "value": "T"}},
                                {"var": "x"},
                                {"lit": {"type": "string", "value": "lower_bound"}},
                                {"lit": {"type": "string", "value": "lower_cmp"}},
                                {"lit": {"type": "string", "value": "upper_bound"}},
                                {"lit": {"type": "string", "value": "upper_cmp"}},
                                {"lit": {"type": "string", "value": "spread_bps"}},
                            ],
                        }
                    ],
                },
                "source_refs": ["A0001"],
            }
        ],
        "open_items": [],
    }

    errors = validate_contract_ir(doc)
    codes = {e.code for e in errors}
    assert "lookup_range_cmp_col_not_string" in codes


def test_contract_ir_numeric_literal_values_must_not_be_null_in_table_cells():
    doc = {
        "schema_version": "contract_ir_v0_2",
        "contract_id": "contract_1",
        "sources": [
            {
                "source_id": "S1",
                "kind": "excerpt_pack",
                "item_id": "contract_1",
                "anchor_ids": ["A0001"],
                "notes": None,
            }
        ],
        "indices": [],
        "tables": [
            {
                "table_id": "T",
                "description": None,
                "columns": [{"name": "lower_bound", "type": "decimal"}],
                "rows": [
                    {
                        "row_id": None,
                        "cells": {
                            # Invalid: missing bound should be encoded as a null CELL, not lit.value=null.
                            "lower_bound": {"lit": {"type": "decimal", "value": None}},
                        },
                        "source_refs": ["A0001"],
                    }
                ],
            }
        ],
        "derived": [
            {
                "fn_id": "F",
                "semantic_role": "spread",
                "description": None,
                "args": [],
                "returns": "rate",
                "expr": {"lit": {"type": "rate", "value": "0.01"}},
                "source_refs": ["A0001"],
            }
        ],
        "open_items": [],
    }

    errors = validate_contract_ir(doc)
    codes = {e.code for e in errors}
    assert "numeric_literal_value_type" in codes


def test_contract_ir_lookup_key_value_cannot_be_literal_column_name():
    doc = {
        "schema_version": "contract_ir_v0_2",
        "contract_id": "contract_1",
        "sources": [
            {
                "source_id": "S1",
                "kind": "excerpt_pack",
                "item_id": "contract_1",
                "anchor_ids": ["A0001"],
                "notes": None,
            }
        ],
        "indices": [],
        "tables": [
            {
                "table_id": "T",
                "description": None,
                "columns": [
                    {"name": "k1", "type": "string"},
                    {"name": "k2", "type": "string"},
                    {"name": "v", "type": "bps"},
                ],
                "rows": [
                    {
                        "row_id": None,
                        "cells": {
                            "k1": {"lit": {"type": "string", "value": "A"}},
                            "k2": {"lit": {"type": "string", "value": "B"}},
                            "v": {"lit": {"type": "bps", "value": "25"}},
                        },
                        "source_refs": ["A0001"],
                    }
                ],
            }
        ],
        "derived": [
            {
                "fn_id": "F",
                "semantic_role": "spread",
                "description": None,
                "args": [{"name": "k1", "type": "string"}],
                "returns": "bps",
                "expr": {
                    "op": "lookup2",
                    "args": [
                        {"lit": {"type": "string", "value": "T"}},
                        {"lit": {"type": "string", "value": "k1"}},
                        {"var": "k1"},
                        {"lit": {"type": "string", "value": "k2"}},
                        # Invalid: key2_value is just the column name literal, not a real key.
                        {"lit": {"type": "string", "value": "k2"}},
                        {"lit": {"type": "string", "value": "v"}},
                    ],
                },
                "source_refs": ["A0001"],
            }
        ],
        "open_items": [],
    }

    errors = validate_contract_ir(doc)
    codes = {e.code for e in errors}
    assert "lookup_key_value_is_column_name" in codes


def test_contract_ir_index_disallows_hardcoded_date_literals():
    doc = {
        "schema_version": "contract_ir_v0_2",
        "contract_id": "contract_1",
        "sources": [
            {
                "source_id": "S1",
                "kind": "excerpt_pack",
                "item_id": "contract_1",
                "anchor_ids": ["A0001"],
                "notes": None,
            }
        ],
        "indices": [{"series_id": "PrimeRate", "unit": "rate", "description": None}],
        "tables": [],
        "derived": [
            {
                "fn_id": "Prime",
                "semantic_role": "base_rate",
                "description": None,
                "args": [{"name": "date", "type": "date"}],
                "returns": "rate",
                "expr": {
                    "op": "index",
                    "args": [
                        {"lit": {"type": "string", "value": "PrimeRate"}},
                        {"lit": {"type": "date", "value": "2020-01-01"}},
                    ],
                },
                "source_refs": ["A0001"],
            }
        ],
        "open_items": [],
    }

    errors = validate_contract_ir(doc)
    codes = {e.code for e in errors}
    assert "index_date_arg_not_var" in codes


def test_contract_ir_lookup_range_key_kind_must_match_bounds():
    doc = {
        "schema_version": "contract_ir_v0_2",
        "contract_id": "contract_1",
        "sources": [
            {
                "source_id": "S1",
                "kind": "excerpt_pack",
                "item_id": "contract_1",
                "anchor_ids": ["A0001"],
                "notes": None,
            }
        ],
        "indices": [],
        "tables": [
            {
                "table_id": "T",
                "description": None,
                "columns": [
                    {"name": "bucket_label", "type": "string"},
                    {"name": "lower_bound", "type": "decimal"},
                    {"name": "lower_cmp", "type": "string"},
                    {"name": "upper_bound", "type": "decimal"},
                    {"name": "upper_cmp", "type": "string"},
                    {"name": "spread_bps", "type": "bps"},
                ],
                "rows": [
                    {
                        "row_id": None,
                        "cells": {
                            "bucket_label": {"lit": {"type": "string", "value": "X"}},
                            "lower_bound": None,
                            "lower_cmp": None,
                            "upper_bound": {"lit": {"type": "decimal", "value": "2.0"}},
                            "upper_cmp": {"lit": {"type": "string", "value": "lt"}},
                            "spread_bps": {"lit": {"type": "bps", "value": "50"}},
                        },
                        "source_refs": ["A0001"],
                    }
                ],
            }
        ],
        "derived": [
            {
                "fn_id": "Spread",
                "semantic_role": "spread",
                "description": None,
                # Wrong: key is a string, but bounds are decimal.
                "args": [{"name": "x", "type": "string"}],
                "returns": "rate",
                "expr": {
                    "op": "bps_to_rate",
                    "args": [
                        {
                            "op": "lookup_range",
                            "args": [
                                {"lit": {"type": "string", "value": "T"}},
                                {"var": "x"},
                                {"lit": {"type": "string", "value": "lower_bound"}},
                                {"lit": {"type": "string", "value": "lower_cmp"}},
                                {"lit": {"type": "string", "value": "upper_bound"}},
                                {"lit": {"type": "string", "value": "upper_cmp"}},
                                {"lit": {"type": "string", "value": "spread_bps"}},
                            ],
                        }
                    ],
                },
                "source_refs": ["A0001"],
            }
        ],
        "open_items": [],
    }

    errors = validate_contract_ir(doc)
    codes = {e.code for e in errors}
    assert "lookup_range_key_kind_mismatch" in codes


def test_contract_ir_table_bool_literal_values_must_be_boolean():
    doc = {
        "schema_version": "contract_ir_v0_2",
        "contract_id": "contract_1",
        "sources": [
            {
                "source_id": "S1",
                "kind": "excerpt_pack",
                "item_id": "contract_1",
                "anchor_ids": ["A0001"],
                "notes": None,
            }
        ],
        "indices": [],
        "tables": [
            {
                "table_id": "T",
                "description": None,
                "columns": [{"name": "flag", "type": "bool"}],
                "rows": [
                    {
                        "row_id": None,
                        "cells": {
                            # Invalid: bool literal values must be boolean, not strings.
                            "flag": {"lit": {"type": "bool", "value": "true"}},
                        },
                        "source_refs": ["A0001"],
                    }
                ],
            }
        ],
        "derived": [
            {
                "fn_id": "F",
                "semantic_role": "spread",
                "description": None,
                "args": [],
                "returns": "rate",
                "expr": {"lit": {"type": "rate", "value": "0.01"}},
                "source_refs": ["A0001"],
            }
        ],
        "open_items": [],
    }

    errors = validate_contract_ir(doc)
    codes = {e.code for e in errors}
    assert "literal_value_type" in codes


def test_contract_ir_lookup_range_requires_value_cells_present_in_rows():
    doc = {
        "schema_version": "contract_ir_v0_2",
        "contract_id": "contract_1",
        "sources": [
            {
                "source_id": "S1",
                "kind": "excerpt_pack",
                "item_id": "contract_1",
                "anchor_ids": ["A0001"],
                "notes": None,
            }
        ],
        "indices": [],
        "tables": [
            {
                "table_id": "T",
                "description": None,
                "columns": [
                    {"name": "bucket_label", "type": "string"},
                    {"name": "lower_bound", "type": "decimal"},
                    {"name": "lower_cmp", "type": "string"},
                    {"name": "upper_bound", "type": "decimal"},
                    {"name": "upper_cmp", "type": "string"},
                    {"name": "spread_bps", "type": "bps"},
                ],
                "rows": [
                    {
                        "row_id": None,
                        "cells": {
                            "bucket_label": {"lit": {"type": "string", "value": "X"}},
                            "upper_bound": {"lit": {"type": "decimal", "value": "2.0"}},
                            "upper_cmp": {"lit": {"type": "string", "value": "lt"}},
                            # Invalid: spread_bps cell is missing entirely.
                        },
                        "source_refs": ["A0001"],
                    }
                ],
            }
        ],
        "derived": [
            {
                "fn_id": "Spread",
                "semantic_role": "spread",
                "description": None,
                "args": [{"name": "x", "type": "decimal"}],
                "returns": "rate",
                "expr": {
                    "op": "bps_to_rate",
                    "args": [
                        {
                            "op": "lookup_range",
                            "args": [
                                {"lit": {"type": "string", "value": "T"}},
                                {"var": "x"},
                                {"lit": {"type": "string", "value": "lower_bound"}},
                                {"lit": {"type": "string", "value": "lower_cmp"}},
                                {"lit": {"type": "string", "value": "upper_bound"}},
                                {"lit": {"type": "string", "value": "upper_cmp"}},
                                {"lit": {"type": "string", "value": "spread_bps"}},
                            ],
                        }
                    ],
                },
                "source_refs": ["A0001"],
            }
        ],
        "open_items": [],
    }

    errors = validate_contract_ir(doc)
    codes = {e.code for e in errors}
    assert "lookup_value_cell_missing" in codes
