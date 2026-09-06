import numpy as np
import pytest

from sparse3d_forgery.experiments.multiwindow_probe import (
    actual_time_spans,
    aggregate_video_scores,
    arithmetic_window_score,
    bootstrap_auroc_difference,
    classify_probe,
    eligible_window_records,
    exact_artifact_match,
    fixed_window_starts,
)


def test_fixed_starts_and_short_video_deduplication_are_label_blind():
    assert fixed_window_starts(153) == (0, 34, 68, 102, 137)
    assert fixed_window_starts(18) == (0, 1, 2)
    assert fixed_window_starts(15) == ()
    # The sole input is frame count: real/fake and annotations cannot affect selection.
    assert fixed_window_starts(153) == fixed_window_starts(153)


def test_exact_center_reuse_requires_identical_source_indices():
    old = np.arange(68, 84, dtype=np.int64)
    assert exact_artifact_match(old, tuple(range(68, 84)))
    assert not exact_artifact_match(old, tuple(range(69, 85)))


def test_window_score_is_the_same_arithmetic_valid_error_mean():
    errors = np.array([[1.0, 99.0], [3.0, 5.0]])
    validity = np.array([[True, False], [True, True]])
    assert arithmetic_window_score(errors, validity) == 3.0


def test_only_causal_eligibility_filters_windows_not_alignment_residual():
    records = [
        {"status": "eligible_generated", "normalized_residual": 999.0},
        {"status": "causal_ineligible", "normalized_residual": 0.0},
    ]
    assert eligible_window_records(records) == [records[0]]


def test_primary_video_score_is_max_and_secondary_mean_cannot_replace_it():
    result = aggregate_video_scores([1.0, 4.0, 2.0])
    assert result == {"score_max": 4.0, "score_mean_secondary": 7 / 3, "max_window_offset": 1}


def test_actual_spans_use_irregular_timestamps_not_fps():
    timestamps = np.cumsum(np.linspace(0.02, 0.08, 16))
    result = actual_time_spans(timestamps)
    assert result["horizon_delta_s"]["8"] == pytest.approx(timestamps[15] - timestamps[7])
    assert result["window_duration_s"] == pytest.approx(timestamps[-1] - timestamps[0])


def test_bootstrap_is_reproducible():
    real_a = np.array([0.1, 0.2, 0.3])
    fake_a = np.array([0.2, 0.4, 0.5])
    real_b = np.array([0.2, 0.3, 0.4])
    fake_b = np.array([0.3, 0.4, 0.5])
    first = bootstrap_auroc_difference(real_a, fake_a, real_b, fake_b, replicates=100)
    second = bootstrap_auroc_difference(real_a, fake_a, real_b, fake_b, replicates=100)
    assert first == second


def test_decision_uses_preregistered_max_metrics_only():
    aurocs = {
        "zero": {"center": 0.4, "max": 0.5},
        "t0": {"center": 0.4, "max": 0.65},
        "spatial": {"center": 0.4, "max": 0.6},
    }
    assert classify_probe(aurocs, True) == "CENTER_WINDOW_COVERAGE_SUPPORTED"
    aurocs["t0"] = {"center": 0.49, "max": 0.52}
    aurocs["spatial"]["max"] = 0.53
    assert classify_probe(aurocs, False) == "CENTER_WINDOW_NOT_MAIN_CAUSE"
