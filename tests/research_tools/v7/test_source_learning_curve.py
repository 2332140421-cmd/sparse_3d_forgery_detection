"""Research-critical contracts for the source learning-curve plan."""

from __future__ import annotations

from research_tools.v7.source_learning_curve.runner import (
    _balanced_selection,
    _label,
    _source_order,
)


def test_balanced_selection_is_unique_and_cell_deterministic() -> None:
    rows = [
        {"charades_source_id": "a", "available_cells": [["fcvg", "delete"]], "fake_variants": [{"generator": "fcvg", "manipulation_operation": "delete", "activityforensics_file": "a"}]},
        {"charades_source_id": "b", "available_cells": [["wan", "add"]], "fake_variants": [{"generator": "wan", "manipulation_operation": "add", "activityforensics_file": "b"}]},
        {"charades_source_id": "c", "available_cells": [["fcvg", "delete"], ["wan", "add"]], "fake_variants": [{"generator": "fcvg", "manipulation_operation": "delete", "activityforensics_file": "c-f"}, {"generator": "wan", "manipulation_operation": "add", "activityforensics_file": "c-w"}]},
    ]
    selected = _balanced_selection(rows, 3)
    assert [row["charades_source_id"] for row in selected] == ["a", "b", "c"]
    assert len({row["charades_source_id"] for row in selected}) == 3


def test_source_order_is_reproducible_and_nested_prefixes_are_valid() -> None:
    source_ids = ["s0", "s1", "s2", "s3", "s4", "s5", "s6", "s7"]
    first = _source_order(source_ids, 20260909)
    assert first == _source_order(source_ids, 20260909)
    assert set(first[:3]).issubset(first[:5])
    assert first != _source_order(source_ids, 20260910)


def test_labels_do_not_depend_on_observation_support() -> None:
    segments = [{"start_s": 10.0, "end_s": 14.0}]
    assert _label("real", 100.0, 101.0, segments) == (0, "REAL_NEGATIVE")
    assert _label("fake", 10.5, 11.5, segments) == (1, "FAKE_MANIPULATION")
    assert _label("fake", 9.5, 10.5, segments) == (None, "BOUNDARY_MIXED")
    assert _label("fake", 20.0, 21.0, segments) == (None, "OUTSIDE_ANNOTATED_MANIPULATION")
