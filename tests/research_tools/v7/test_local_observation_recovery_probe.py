"""Pure-contract tests for the bounded local observation recovery probe."""

import json

import numpy as np

from research_tools.v7.local_observation_recovery_probe.probe import (
    condition_frame_status,
    fixed_members,
    layer_counts,
    nearest_frame,
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
