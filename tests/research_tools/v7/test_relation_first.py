"""Contract tests for the relation-first representation pilot."""

from __future__ import annotations

import numpy as np
import pytest

from research_tools.v7.relation_first.representation import (
    aggregate_component,
    aggregate_window,
    descriptor,
    relation_first_window,
)
from research_tools.v7.relation_first.temporal_derivatives import timestamp_derivatives
from research_tools.v7.paired_signal.protocol import assert_original_video_path


def test_timestamp_aware_first_derivative_uses_pts():
    first, first_valid, _, _ = timestamp_derivatives(
        np.asarray([0.0, 2.0, 5.0]),
        np.asarray([True, True, True]),
        np.asarray([0.0, 0.5, 1.5]),
        np.asarray([0, 1, 2]),
    )
    assert first_valid.tolist() == [False, True, True]
    assert np.isnan(first[0])
    assert first[1:] == pytest.approx([4.0, 3.0])


def test_timestamp_aware_second_derivative_and_uniform_equivalence():
    values = np.asarray([0.0, 1.0, 4.0, 9.0])
    first, first_valid, second, second_valid = timestamp_derivatives(
        values, np.ones(4, dtype=bool), np.arange(4, dtype=np.float64), np.arange(4)
    )
    assert first_valid.tolist() == [False, True, True, True]
    assert second_valid.tolist() == [False, False, True, True]
    assert second[2] == pytest.approx(2.0)
    assert second[3] == pytest.approx(2.0)


def test_no_derivative_crosses_missing_gap():
    _, first_valid, _, second_valid = timestamp_derivatives(
        np.asarray([0.0, 1.0, 2.0, 3.0]),
        np.asarray([True, False, True, True]),
        np.arange(4, dtype=np.float64),
        np.arange(4),
    )
    assert first_valid.tolist() == [False, False, False, True]
    assert second_valid.tolist() == [False, False, False, False]


def test_no_derivative_across_original_frame_gap():
    _, first_valid, _, _ = timestamp_derivatives(
        np.asarray([0.0, 1.0, 2.0]),
        np.ones(3, dtype=bool),
        np.asarray([0.0, 1.0, 2.0]),
        np.asarray([0, 2, 3]),
    )
    assert first_valid.tolist() == [False, False, True]


def test_pair_descriptor_keeps_signed_and_magnitude_statistics():
    value = descriptor(np.asarray([-2.0, -1.0, 1.0, 2.0]), np.ones(4, dtype=bool))
    assert value is not None
    assert value.tolist() == pytest.approx([0.0, 1.5, 1.5, 2.0])


def test_pair_and_component_aggregation_are_equal_weighted():
    component_a = aggregate_component([[0.0] * 4, [2.0] * 4])
    component_b = aggregate_component([[10.0] * 4])
    assert component_a is not None and component_b is not None
    assert component_a["vector"][:4] == pytest.approx([1.0] * 4)
    assert aggregate_window([component_a["vector"], component_b["vector"]])[:4] == pytest.approx([5.5] * 4)


def test_derivative_before_aggregation_preserves_opposite_pair_directions():
    plus = descriptor(np.asarray([0.5, 1.0, 1.5]), np.ones(3, dtype=bool))
    minus = descriptor(np.asarray([-0.5, -1.0, -1.5]), np.ones(3, dtype=bool))
    assert plus is not None and minus is not None
    assert plus[0] == pytest.approx(-minus[0])
    aggregated = aggregate_component([plus, minus])
    assert aggregated is not None
    assert aggregated["descriptor_median"][0] == pytest.approx(0.0)


def test_preview_path_is_rejected_before_media_access(tmp_path):
    with pytest.raises(ValueError, match="preview"):
        assert_original_video_path(tmp_path / "preview_fake.mp4")


def test_frozen_population_and_window_manifests_are_reused():
    root = __import__("pathlib").Path("/root/autodl-tmp/data/sparse_3d_forgery_detection/derived/v7_activityforensics_paired_second_order_pilot_v1")
    if not root.exists():
        pytest.skip("frozen pilot artifacts are not mounted")
    import json

    selected = json.loads((root / "manifests/selected_pairs.json").read_text())
    windows = json.loads((root / "manifests/window_manifest.json").read_text())
    assert len(selected) == 16
    assert len(windows) == 192
    assert {item["pair_id"] for item in windows} == {item["pair_id"] for item in selected}


def test_relation_first_uses_persistent_track_identity(tmp_path):
    xyz = np.asarray(
        [
            [[0, 0, 1], [1, 0, 1], [0, 1, 1]],
            [[0, 0, 1], [1, 0, 1], [0, 1, 1]],
            [[0, 0, 1], [1, 0, 1], [0, 1, 1]],
            [[0, 0, 1], [1, 0, 1], [0, 1, 1]],
            [[0, 0, 1], [1, 0, 1], [0, 1, 1]],
            [[0, 0, 1], [1, 0, 1], [0, 1, 1]],
            [[0, 0, 1], [1, 0, 1], [0, 1, 1]],
            [[0, 0, 1], [1, 0, 1], [0, 1, 1]],
            [[0, 0, 1], [1, 0, 1], [0, 1, 1]],
            [[0, 0, 1], [1, 0, 1], [0, 1, 1]],
        ],
        dtype=np.float32,
    )
    prefix = tmp_path / "particle"
    np.savez(prefix.with_suffix(".npz"), xyz=xyz, geometry_validity=np.ones((10, 3), dtype=bool))
    result = relation_first_window(prefix.with_suffix(".npz"), [{"component_index": 0, "members": [0, 1, 2]}], np.arange(10, dtype=float), np.arange(10))
    assert result["persistent_pair_count"] == 3
    assert result["r1_eligible_pair_count"] == 3
    assert result["r2_eligible_pair_count"] == 3


def test_relation_first_does_not_use_nonpersistent_pairs(tmp_path):
    xyz = np.zeros((10, 3, 3), dtype=np.float32)
    xyz[:, 1, 0] = 1.0
    xyz[:, 2, 1] = 1.0
    valid = np.ones((10, 3), dtype=bool)
    xyz[:3, 2, 0] = np.nan
    prefix = tmp_path / "particle"
    np.savez(prefix.with_suffix(".npz"), xyz=xyz, geometry_validity=valid)
    result = relation_first_window(prefix.with_suffix(".npz"), [{"component_index": 0, "members": [0, 1, 2]}], np.arange(10, dtype=float), np.arange(10))
    assert result["persistent_pair_count"] == 1
    assert result["r2_eligible_pair_count"] == 1
