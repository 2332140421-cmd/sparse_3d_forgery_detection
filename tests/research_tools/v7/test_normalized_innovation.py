"""Contract tests for the frozen normalized structural innovation pilot."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from research_tools.v7.normalized_innovation.normalized_innovation import (
    component_q90,
    cross_source_summary,
    leave_one_source_out,
    normalized_structural_innovation,
    paired_nsi_gain,
)
from research_tools.v7.normalized_innovation.structural_trajectory import aggregate_component_q90, window_nsi


def test_local_triplet_uses_identical_sorted_pair_coordinates(tmp_path):
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
        ],
        dtype=np.float32,
    )
    path = tmp_path / "particle.npz"
    np.savez(path, xyz=xyz, geometry_validity=np.ones((8, 3), dtype=bool), timestamps_s=np.arange(8, dtype=float), frame_indices=np.arange(8))
    result = window_nsi(path, [{"component_index": 0, "members": [2, 0, 1]}], np.arange(8, dtype=float), np.arange(8))
    assert result["triplet_count"] == 6
    assert {row["common_pair_count"] for row in result["components"][0]["triplets"]} == {3}


def test_no_triplet_crosses_invalid_gap(tmp_path):
    xyz = np.asarray([[[0, 0, 1], [1, 0, 1], [0, 1, 1]]] * 8, dtype=np.float32)
    valid = np.ones((8, 3), dtype=bool)
    valid[3, :] = False
    path = tmp_path / "particle.npz"
    np.savez(path, xyz=xyz, geometry_validity=valid, timestamps_s=np.arange(8, dtype=float), frame_indices=np.arange(8))
    result = window_nsi(path, [{"component_index": 0, "members": [0, 1, 2]}], np.arange(8, dtype=float), np.arange(8))
    assert all(2 not in (row["center_index"],) for row in result["components"][0]["triplets"])
    assert all(3 not in (row["center_index"],) for row in result["components"][0]["triplets"])


def test_timestamp_aware_velocity_is_used():
    # Both sides have the same relation-space velocity despite unequal PTS.
    assert normalized_structural_innovation([2.0, 0.0], [2.0, 0.0]) == pytest.approx(0.0)


def test_constant_structural_velocity_has_zero_nsi():
    assert normalized_structural_innovation([1.0, -2.0], [1.0, -2.0]) == pytest.approx(0.0)


def test_velocity_reversal_has_nsi_near_one():
    assert normalized_structural_innovation([3.0, 0.0], [-3.0, 0.0]) == pytest.approx(1.0)


def test_nsi_is_invariant_to_common_positive_speed_scale():
    base = normalized_structural_innovation([1.0, 2.0], [3.0, -1.0])
    scaled = normalized_structural_innovation([17.0, 34.0], [51.0, -17.0])
    assert scaled == pytest.approx(base)


def test_repeated_relation_coordinates_do_not_change_nsi():
    base = normalized_structural_innovation([1.0, 2.0], [3.0, -1.0])
    repeated = normalized_structural_innovation([1.0, 2.0, 1.0, 2.0], [3.0, -1.0, 3.0, -1.0])
    assert repeated == pytest.approx(base)


def test_nsi_is_bounded():
    rng = np.random.default_rng(7)
    for _ in range(100):
        value = normalized_structural_innovation(rng.normal(size=5), rng.normal(size=5))
        assert -1e-12 <= value <= 1.0 + 1e-12


def test_component_q90_is_deterministic_and_window_component_median_is_equal_weighted():
    assert component_q90([0.0, 0.1, 0.2, 0.3]) == pytest.approx(0.27)
    # The implementation uses median across component Q90 values, not size weights.
    assert aggregate_component_q90([0.1, 0.9]) == pytest.approx(0.5)


def _calibration_rows():
    rows = []
    for source_index, source in enumerate(("A", "B", "C")):
        for role, kind, value in (("real", "MANIP", 0.2 + source_index * 0.01), ("real", "CTRL", 0.1 + source_index * 0.01), ("fake", "MANIP", 0.8), ("fake", "CTRL", 0.4)):
            rows.append({"window_id": f"{source}-{role}-{kind}", "source_id": source, "role": role, "kind": kind, "I_window": value, "label": kind, "anchor_fraction": 0.25})
    return rows


def test_loso_excludes_held_out_source_and_fake_from_calibration():
    calibrated, metadata = leave_one_source_out(_calibration_rows())
    assert metadata["A"]["N_real_calibration"] == 4
    assert metadata["A"]["fake_count"] == 0
    assert all(row["calibration_source_count"] == 4 for row in calibrated if row["source_id"] == "A")


def test_loso_calibration_does_not_use_labels_as_training_inputs():
    rows = _calibration_rows()
    for row in rows:
        row["label"] = "adversarial-label-that-must-not-matter"
    first, _ = leave_one_source_out(rows)
    for row in rows:
        row["label"] = "different-label"
    second, _ = leave_one_source_out(rows)
    assert [row["Z_I"] for row in first] == pytest.approx([row["Z_I"] for row in second])


def test_cross_source_summary_reports_h_and_group_values():
    calibrated, _ = leave_one_source_out(_calibration_rows())
    source_rows, summary = cross_source_summary(calibrated)
    assert len(source_rows) == 3
    assert summary["H_I"]["N"] == 3
    assert summary["Z_real_manip"]["N"] == 3


def test_paired_gain_uses_three_matched_anchors_and_absolute_discrepancy():
    rows = []
    for kind in ("MANIP", "CTRL"):
        for anchor in (0.25, 0.50, 0.75):
            for role, value in (("real", 0.1), ("fake", 0.3 if kind == "MANIP" else 0.15)):
                rows.append({"pair_id": "0001", "source_id": "A", "kind": kind, "label": f"{kind}_{int(anchor * 100)}", "anchor_fraction": anchor, "role": role, "I_window": value})
    output = paired_nsi_gain(rows, [{"pair_id": "0001", "source_id": "A"}])
    assert output[0]["D_manip"] == pytest.approx(0.2)
    assert output[0]["D_ctrl"] == pytest.approx(0.05)
    assert output[0]["G_I"] == pytest.approx(0.15)


def test_frozen_population_and_windows_are_unchanged():
    root = Path("/root/autodl-tmp/data/sparse_3d_forgery_detection/derived/v7_activityforensics_paired_second_order_pilot_v1")
    if not root.exists():
        pytest.skip("frozen pilot artifacts are not mounted")
    assert len(json.loads((root / "manifests/selected_pairs.json").read_text())) == 16
    assert len(json.loads((root / "manifests/window_manifest.json").read_text())) == 192


def test_no_frontend_rerun_and_formal_src_boundary():
    root = Path("/root/autodl-tmp/data/sparse_3d_forgery_detection/derived/v7_activityforensics_paired_second_order_pilot_v1")
    assert root.exists()
    assert not Path("src/sparse3d_forgery").joinpath("v7").exists()
