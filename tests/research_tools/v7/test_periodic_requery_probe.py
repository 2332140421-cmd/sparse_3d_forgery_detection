from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from sparse3d_forgery.video_input import DecodedFrame, DecodedVideoSample

from research_tools.v7.periodic_requery_probe.runner import (
    OFFSETS_S,
    _build_sequence,
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


def test_build_sequence_converts_geometry_dtype_without_repairing_missing() -> None:
    frames = tuple(
        DecodedFrame(
            source_frame_index=index,
            timestamp_s=float(index) * 0.1,
            rgb=np.zeros((4, 5, 3), dtype=np.uint8),
        )
        for index in range(3)
    )
    decoded = DecodedVideoSample("sample", "video", frames)
    visibility = np.ones((3, 289), dtype=np.bool_)
    geometry = np.ones((3, 289), dtype=np.bool_)
    visibility[1, 7] = False
    geometry[1, 7] = False
    xyz = np.zeros((3, 289, 3), dtype=np.float64)
    xyz[1, 7] = np.nan
    uv = np.zeros((3, 289, 2), dtype=np.float32)
    uv[1, 7] = np.nan
    sequence = _build_sequence(
        decoded,
        uv=uv,
        visibility=visibility,
        xyz=xyz,
        geometry_valid=geometry,
        row={"source_id": "S", "pair_id": "P", "role": "real", "window_id": "W"},
        mode="O",
        parent_id="PARENT",
        cohort_start_s=0.0,
    )
    assert sequence.xyz.dtype == np.float32
    assert np.isnan(sequence.xyz[1, 7]).all()
    assert not sequence.geometry_validity[1, 7]
