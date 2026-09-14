"""Pure-contract tests for the bounded local observation recovery probe."""

import json

import numpy as np

from research_tools.v7.local_observation_recovery_probe.probe import (
    condition_frame_status,
    fixed_members,
    history_evaluation_status,
    layer_counts,
    nearest_frame,
    recent_trace_indices,
    _trajectory_entry,
    validate_roi_rect,
)


def test_nearest_frame_uses_pts_and_identifies_adjacent_case():
    frames = [487, 488]
    pts = [16.249582916249583, 16.28294961628295]
    first = nearest_frame(frames, pts, 16.249583)
    second = nearest_frame(frames, pts, 16.282950)
    assert first["source_frame_index"] == 487
    assert second["source_frame_index"] == 488
    assert second["timestamp_s"] > first["timestamp_s"]
    assert np.isclose(second["timestamp_s"] - first["timestamp_s"], 0.0333667, atol=1e-9)


def test_layer_counts_never_counts_invalid_xyz_as_geometry_or_display():
    uv = np.asarray([[1.0, 2.0], [np.nan, np.nan], [4.0, 5.0]])
    visibility = np.asarray([True, True, False])
    geo = np.asarray([True, True, True])
    xyz = np.asarray([[0.0, 0.0, 1.0], [np.nan, np.nan, np.nan], [0.0, 0.0, 1.0]])
    row = layer_counts(uv, visibility, geo, xyz, group_slots=[0, 1, 2], pair_slots=[(0, 1), (1, 2)])
    assert row["query_count"] == 3
    assert row["uv_available_count"] == 2
    assert row["visibility_count"] == 1
    assert row["geometry_valid_count"] == 1
    assert row["common_relation_support"] == 0
    assert row["final_display_count"] == 1


def test_fixed_members_are_the_original_condition_only():
    visibility = np.asarray([[True, False, True], [False, True, True]])
    np.testing.assert_array_equal(fixed_members(visibility, 0), [0, 2])
    np.testing.assert_array_equal(fixed_members(visibility, 1), [1, 2])


def test_missing_or_prequery_state_is_not_silently_filled():
    row = layer_counts(np.full((2, 2), np.nan), np.zeros(2, dtype=bool), np.zeros(2, dtype=bool), np.full((2, 3), np.nan))
    assert row["visibility_count"] == 0
    assert row["geometry_valid_count"] == 0
    assert row["display_reason"] == "NO_VISIBLE_FINITE_UV"


def test_requery_has_no_prequery_display_state():
    assert condition_frame_status("R", 487, query_start_frame=488) == "NOT_QUERIED"
    assert condition_frame_status("R", 488, query_start_frame=488) == "AVAILABLE"
    assert condition_frame_status("O", 487, query_start_frame=488) == "AVAILABLE"


def test_history_frame_is_not_model_evaluation_frame():
    status = history_evaluation_status(
        initialization_frame=476,
        initialization_pts_s=15.88254921588255,
        frame_index=488,
        frame_pts_s=16.28294961628295,
        model_frame_indices=[491, 493, 496, 499, 502],
    )
    assert status["within_first_history_window"] is True
    assert status["model_evaluation_frame"] is False
    assert status["phase"] == "HISTORY"
    assert np.isclose(status["relative_to_initialization_s"], 0.4004004004004)


def test_roi_validation_is_source_pixel_bookkeeping_only():
    assert validate_roi_rect([110.2, -3, 370.4, 359.9], 480, 360) == (110, 0, 370, 360)


def test_recent_trace_keeps_saved_order_and_limits_to_five_frames():
    assert recent_trace_indices([487, 488, 489, 490, 491, 492], 5) == [1, 2, 3, 4, 5]
    assert recent_trace_indices([487, 488], 1, limit=5) == [0, 1]


def test_trajectory_payload_preserves_frame_order_and_missing_uv():
    payload = _trajectory_entry(
        np.asarray([487, 488], dtype=np.int64),
        np.asarray([16.2495, 16.2829], dtype=np.float64),
        np.asarray([[[1.0, 2.0]], [[np.nan, np.nan]]], dtype=np.float32),
        np.asarray([[True], [False]], dtype=bool),
        np.asarray([[True], [False]], dtype=bool),
    )
    assert payload["frame_indices"] == [487, 488]
    assert payload["uv"][0][0] == [1.0, 2.0]
    assert payload["uv"][1][0] is None
    assert payload["uv_finite"] == [[True], [False]]
