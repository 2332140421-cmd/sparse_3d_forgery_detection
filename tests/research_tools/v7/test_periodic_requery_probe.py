from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from research_tools.v7.periodic_requery_probe.runner import (
    OFFSETS_S,
    _window_label,
    _choose_parents,
)


def test_label_mapping_is_applied_after_fixed_sampling() -> None:
    segments = [{"start_s": 10.0, "end_s": 20.0}]
    assert _window_label("real", 12.0, 13.0, segments) == (0, "REAL_NEGATIVE")
    assert _window_label("fake", 12.0, 13.0, segments) == (1, "FAKE_MANIPULATION")
    assert _window_label("fake", 9.5, 10.5, segments) == (None, "BOUNDARY_MIXED")
    assert _window_label("fake", 21.0, 22.0, segments) == (None, "OUTSIDE_ANNOTATED_MANIPULATION")


def test_frozen_parent_selection_has_three_offsets_without_score_fields() -> None:
    parents, subwindows = _choose_parents()
    assert len(parents) == 32
    assert len(subwindows) == 96
    assert sorted({float(row["offset_s"]) for row in subwindows}) == list(OFFSETS_S)
    assert all(row["selection_rule"] for row in parents if row.get("status") == "PLANNED")
    assert not any("score" in key.lower() for row in subwindows for key in row)


def test_window_ids_keep_query_cohorts_separate() -> None:
    parents, subwindows = _choose_parents()
    assert len({row["window_id"] for row in subwindows}) == len(subwindows)
    for row in subwindows:
        assert row["parent_id"] in {parent["parent_id"] for parent in parents}
        assert row["frame_indices"] == sorted(row["frame_indices"])
        assert len(row["frame_indices"]) == len(row["timestamps_s"])
        if row["timestamps_s"]:
            assert np.all(np.diff(np.asarray(row["timestamps_s"], dtype=float)) > 0)
