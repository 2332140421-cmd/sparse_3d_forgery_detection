import numpy as np
import pytest

from sparse3d_forgery.experiments.prefix_target_audit import (
    PREFIX_INFERENCE_HORIZONS,
    classify_real,
    common_target_measurements,
    diagnostic_ratio,
    fixed_history_anchor,
    prefix_frame_indices,
    summarize_ratios,
)
from test_temporal_probe import _sequence


@pytest.mark.parametrize("horizon,length", [(1, 9), (2, 10), (4, 12), (8, 16)])
def test_prefix_selection_stops_exactly_at_target(horizon, length):
    indices = np.arange(16, dtype=np.int64) + 20
    selected = prefix_frame_indices(indices, 8, horizon)
    assert selected == tuple(range(20, 20 + length))
    assert not set(indices[length:]) & set(selected)


def test_fixed_anchor_reuses_exact_existing_history_arrays():
    sequence = _sequence()
    sequence = __import__("dataclasses").replace(
        sequence,
        lineage={
            "construction": "history-anchored causal VGGT window",
            "history_count": 8,
            "history_frame_indices": list(range(8)),
            "pass_a_history_only": {"provider": "test"},
        },
    )
    anchor = fixed_history_anchor(sequence)
    np.testing.assert_array_equal(anchor.frame_indices, sequence.frame_indices[:8])
    np.testing.assert_array_equal(anchor.xyz, sequence.xyz[:8])
    assert not np.shares_memory(anchor.xyz, sequence.xyz)


def test_target_comparison_uses_same_track_joint_validity_and_preserves_invalid():
    prefix = np.array([[1, 0, 0], [9, 9, 9]], dtype=np.float32)
    full = np.array([[0, 0, 0], [8, 8, 8]], dtype=np.float32)
    cutoff = np.zeros((2, 3), dtype=np.float32)
    result = common_target_measurements(
        prefix,
        np.array([True, False]),
        full,
        np.array([True, True]),
        cutoff,
        np.array([True, True]),
    )
    assert result["validity"].tolist() == [True, False]
    assert result["motion_validity"].tolist() == [True, False]
    assert result["disagreement"].tolist() == [1.0]
    assert result["motion"].tolist() == [1.0]
    assert result["q_target"] == 1.0


def test_q_target_uses_one_common_set_and_zero_motion_is_undefined():
    xyz = np.zeros((2, 3), dtype=np.float32)
    result = common_target_measurements(
        xyz, np.ones(2, bool), xyz + 1, np.ones(2, bool), xyz, np.ones(2, bool)
    )
    assert result["q_target"] is None
    assert diagnostic_ratio(1.0, 0.0) is None


def test_target_disagreement_does_not_require_cutoff_but_ratio_does():
    prefix = np.array([[1, 0, 0], [2, 0, 0]], dtype=np.float32)
    full = np.zeros((2, 3), dtype=np.float32)
    cutoff = np.zeros((2, 3), dtype=np.float32)
    result = common_target_measurements(
        prefix,
        np.ones(2, bool),
        full,
        np.ones(2, bool),
        cutoff,
        np.array([True, False]),
    )
    assert result["validity"].tolist() == [True, True]
    assert result["motion_validity"].tolist() == [True, False]
    assert result["disagreement"].tolist() == [1.0, 2.0]
    assert result["q_target"] == 1.0


def test_h8_is_not_a_new_inference_horizon():
    assert PREFIX_INFERENCE_HORIZONS == (1, 2, 4)
    assert 8 not in PREFIX_INFERENCE_HORIZONS


def test_prefix_selection_rejects_short_or_unknown_horizon_without_mutation():
    indices = np.arange(10, dtype=np.int64)
    before = indices.copy()
    with pytest.raises(ValueError):
        prefix_frame_indices(indices, 8, 4)
    with pytest.raises(ValueError):
        prefix_frame_indices(indices, 8, 3)
    np.testing.assert_array_equal(indices, before)


def test_invalid_target_nan_is_excluded_without_repair():
    prefix = np.array([[1, 0, 0], [np.nan, np.nan, np.nan]], dtype=np.float32)
    full = np.zeros((2, 3), dtype=np.float32)
    result = common_target_measurements(
        prefix,
        np.array([True, False]),
        full,
        np.ones(2, bool),
        full,
        np.ones(2, bool),
    )
    assert result["disagreement"].tolist() == [1.0]
    assert np.isnan(prefix[1]).all()


def test_primary_classification_has_no_label_or_role_input():
    metrics = {
        "alignment": {str(h): {"median": 1.0} for h in (1, 2, 4, 8)},
        "q_target": {str(h): {"median": 0.75} for h in (1, 2, 4)},
        "correlation": {
            str(h): {"full": 0.5, "prefix": 0.5} for h in (1, 2, 4)
        },
    }
    assert classify_real(metrics, False) == "FRONTEND_CONTEXT_INSTABILITY_PERSISTS"


def test_supported_classification_uses_all_three_preregistered_conditions():
    metrics = {
        "alignment": {
            "1": {"median": 0.5},
            "2": {"median": 0.7},
            "4": {"median": 0.9},
            "8": {"median": 1.0},
        },
        "q_target": {
            "1": {"median": 0.2},
            "2": {"median": 0.4},
            "4": {"median": 0.8},
        },
        "correlation": {
            "1": {"full": 0.8, "prefix": 0.6},
            "2": {"full": 0.7, "prefix": 0.5},
            "4": {"full": 0.6, "prefix": 0.55},
        },
    }
    assert classify_real(metrics, False) == "PREFIX_TARGET_STABILITY_SUPPORTED"


def test_ratio_summary_uses_preregistered_thresholds():
    result = summarize_ratios([0.1, 0.4, 0.8, 1.2])
    assert result["fraction_lt_0_25"] == 0.25
    assert result["fraction_lt_0_5"] == 0.5
    assert result["fraction_lt_1"] == 0.75
    assert result["fraction_ge_1"] == 0.25
