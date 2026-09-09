"""Research-contract tests for the bounded B0 no-reference diagnostic."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest

from research_tools.v7.b0_no_reference.analyze_b0_no_reference import (
    BOOTSTRAP_REPLICATES,
    BOOTSTRAP_SEED,
    choose_status,
    evaluation,
    bootstrap_median,
    source_score_rows,
    window_score_rows,
)
from research_tools.v7.b0_no_reference.b0_reconstruction import (
    B0_DIMENSIONS,
    b0_observations,
    b0_window_median,
    load_and_reproduce,
)
from research_tools.v7.b0_no_reference.robust_source_balanced_scaler import (
    IQR_DIVISOR,
    MAD_MULTIPLIER,
    SCALE_FLOOR,
    fit_source_balanced_scaler,
    source_balanced_quantile,
)


def _values(*rows: float) -> list[list[float]]:
    return [[value, value + 1.0, value + 2.0, value + 3.0] for value in rows]


def _observation(window_id: str, source: str, role: str, kind: str, value: float, ordinal: int = 0) -> dict:
    return {
        "window_id": window_id,
        "pair_id": window_id.split("_", 1)[0],
        "source_id": source,
        "role": role,
        "kind": kind,
        "anchor_fraction": 0.25,
        "component_index": 0,
        "time_index": ordinal,
        "timestamp_s": float(ordinal),
        "observation_ordinal": ordinal,
        "values": [value, value, value, value],
    }


def test_b0_dimension_and_definition_are_frozen():
    assert B0_DIMENSIONS == ("mean", "std", "p25", "p75")


def test_b0_window_aggregation_is_component_time_median():
    result = {
        "window": {"source_id": "s", "pair_id": "p", "window_id": "w", "role": "real", "kind": "MANIP", "anchor_fraction": 0.25},
        "features": {
            "observations": {
                "K0_S": [
                    {"component_index": 0, "time_index": 0, "timestamp_s": 0.0, "values": [1, 2, 3, 4]},
                    {"component_index": 1, "time_index": 1, "timestamp_s": 1.0, "values": [3, 4, 5, 6]},
                ]
            }
        },
    }
    assert np.allclose(b0_window_median(result), [2, 3, 4, 5])


def test_reconstruction_keeps_k0_only_and_ignores_dynamic_features():
    result = {
        "window": {"source_id": "s", "pair_id": "p", "window_id": "w", "role": "real", "kind": "CTRL", "anchor_fraction": 0.25},
        "features": {
            "observations": {
                "K0_S": [{"component_index": 0, "time_index": 0, "timestamp_s": 0.0, "values": [1, 2, 3, 4]}],
                "K1_delta_s": [{"values": [99, 99, 99, 99]}],
            }
        },
    }
    assert len(b0_observations(result)) == 1
    assert b0_observations(result)[0]["values"] == [1.0, 2.0, 3.0, 4.0]


def test_source_balanced_quantile_gives_each_source_equal_total_weight():
    # A has many low observations; B has one high observation.  The weighted
    # median is the first B value only after A's half mass is exhausted.
    values = {"A": [[0, 0, 0, 0]] * 100, "B": [[100, 100, 100, 100]]}
    assert source_balanced_quantile(values, 0.5).tolist() == [0.0] * 4


def test_source_balanced_quantile_is_deterministic_and_left_continuous():
    values = {"B": _values(2, 4), "A": _values(0, 6)}
    first = source_balanced_quantile(values, 0.5)
    second = source_balanced_quantile(values, 0.5)
    assert np.array_equal(first, second)
    assert first[0] == 2.0


def test_scaler_uses_real_source_values_supplied_by_caller_only():
    scaler = fit_source_balanced_scaler({"real-a": _values(0, 1), "real-b": _values(10, 11)})
    assert scaler.training_sources == ("real-a", "real-b")
    assert np.all(np.isfinite(scaler.center))


def test_scaler_records_fixed_mad_rule():
    scaler = fit_source_balanced_scaler({"a": _values(0, 1, 2), "b": _values(10, 11, 12)})
    details = scaler.as_dict()
    assert details["mad_multiplier"] == MAD_MULTIPLIER
    assert details["iqr_divisor"] == IQR_DIVISOR
    assert details["score_epsilon"] == 1e-8


def test_zero_mad_uses_iqr_fallback_before_floor():
    scaler = fit_source_balanced_scaler({"a": _values(0, 0, 10), "b": _values(0, 0, 10)})
    assert scaler.iqr_fallback_dimensions == (0, 1, 2, 3)
    assert scaler.floor_fallback_dimensions == ()


def test_zero_mad_and_zero_iqr_use_fixed_scale_floor():
    scaler = fit_source_balanced_scaler({"a": _values(2, 2), "b": _values(2, 2)})
    assert scaler.floor_fallback_dimensions == (0, 1, 2, 3)
    assert np.allclose(scaler.scale, SCALE_FLOOR)


def test_score_is_robust_standardized_rms_not_sum_or_covariance():
    scaler = fit_source_balanced_scaler({"a": _values(0, 1, 2), "b": _values(0, 1, 2)})
    value = np.asarray([1, 1, 1, 1], dtype=float)
    expected = float(np.sqrt(np.mean(((value - scaler.center) / (scaler.scale + 1e-8)) ** 2)))
    assert scaler.score(value) == pytest.approx(expected)


def test_score_rejects_nonfinite_or_wrong_shape_without_casting():
    scaler = fit_source_balanced_scaler({"a": _values(0, 1), "b": _values(0, 1)})
    with pytest.raises(ValueError):
        scaler.score([1, 2, 3])
    with pytest.raises(ValueError):
        scaler.score([1, 2, 3, float("nan")])


def test_loso_training_source_excludes_held_out_source():
    held_out = "held"
    scaler = fit_source_balanced_scaler({"a": _values(0, 1), "b": _values(2, 3)})
    assert held_out not in scaler.training_sources
    assert "a" in scaler.training_sources and "b" in scaler.training_sources


def test_fake_and_held_out_real_are_not_in_fitting_contract():
    scaler = fit_source_balanced_scaler({"real-a": _values(0, 1)})
    assert scaler.training_sources == ("real-a",)
    assert scaler.as_dict()["training_sources"] == ["real-a"]


def test_inference_score_does_not_use_paired_real_or_group_labels():
    scaler = fit_source_balanced_scaler({"real-a": _values(0, 1), "real-b": _values(0, 1)})
    value = [20, 20, 20, 20]
    assert scaler.score(value) == scaler.score(value)


def test_window_scores_use_median_and_keep_valid_count():
    rows = [_observation("p_MANIP_25::real", "s", "real", "MANIP", 1, 0), _observation("p_MANIP_25::real", "s", "real", "MANIP", 3, 1)]
    for row in rows:
        row["score"] = row["values"][0]
    output = window_score_rows(rows)
    assert output[0]["score"] == 2.0
    assert output[0]["valid_observation_count"] == 2


def test_source_rows_retain_invalid_source_accounting():
    rows = [{**_observation("p_MANIP_25::real", "valid", "real", "MANIP", 1), "score": 1.0}]
    windows = window_score_rows(rows)
    source_rows = source_score_rows(windows, ["valid", "invalid"])
    invalid = [row for row in source_rows if row["source_id"] == "invalid"]
    assert invalid and all(row["score_median"] is None for row in invalid)


def test_evaluation_uses_source_level_four_group_medians():
    source_rows = []
    values = {("real", "MANIP"): 1, ("fake", "MANIP"): 3, ("real", "CTRL"): 1, ("fake", "CTRL"): 2}
    for (role, kind), value in values.items():
        source_rows.append({"source_id": "s", "role": role, "kind": kind, "score_median": value, "window_count": 1})
    result = evaluation(source_rows, [], [])
    assert result["per_source_deltas"][0]["Delta_manip"] == 2
    assert result["per_source_deltas"][0]["Delta_control"] == 1
    assert result["per_source_deltas"][0]["J_B0"] == 1


def test_evaluation_never_fits_fake_or_heldout_real():
    result = evaluation([], [], [])
    assert result["per_source_deltas"] == []
    assert result["observation_score_count"] == 0


def test_bootstrap_seed_and_replicate_count_are_frozen():
    first = bootstrap_median([1.0, 2.0, 3.0])
    second = bootstrap_median([1.0, 2.0, 3.0])
    assert first == second
    assert first["seed"] == BOOTSTRAP_SEED == 20260909
    assert first["replicates"] == BOOTSTRAP_REPLICATES == 10_000


def test_status_requires_both_delta_and_j_gates():
    positive = {"median": 1.0, "positive_fraction": 1.0, "bootstrap": {"ci95": [0.1, 2.0]}}
    assert choose_status(positive, positive, 10, True, 0.5) == "B0_NO_REFERENCE_SIGNAL_PRESENT"
    unstable = {"median": 1.0, "positive_fraction": 0.5, "bootstrap": {"ci95": [-1.0, 2.0]}}
    assert choose_status(unstable, unstable, 10, True, 0.5) == "B0_PAIRED_ONLY_NOT_NO_REFERENCE"


def test_status_reports_support_before_signal():
    positive = {"median": 1.0, "positive_fraction": 1.0, "bootstrap": {"ci95": [0.1, 2.0]}}
    assert choose_status(positive, positive, 7, True, 0.9) == "B0_SUPPORT_INSUFFICIENT"


def test_reconstruction_exactly_matches_historical_csv(tmp_path: Path):
    (tmp_path / "frontend").mkdir()
    (tmp_path / "metrics").mkdir()
    result = {"status": "COMPLETE", "window": {"source_id": "s", "pair_id": "p", "window_id": "p_MANIP_25::real", "role": "real", "kind": "MANIP", "anchor_fraction": 0.25, "frame_indices": [0], "timestamps_s": [0.0]}, "features": {"observations": {"K0_S": [{"component_index": 0, "time_index": 0, "timestamp_s": 0.0, "values": [1, 2, 3, 4]}]}}}
    (tmp_path / "frontend/window_results.json").write_text(json.dumps([result]))
    with (tmp_path / "metrics/per_window_signal.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["window_id", "K0_S_median"])
        writer.writeheader()
        writer.writerow({"window_id": "p_MANIP_25::real", "K0_S_median": json.dumps([1, 2, 3, 4])})
    observations, windows, reproduction = load_and_reproduce(tmp_path)
    assert len(observations) == len(windows) == 1
    assert reproduction["max_abs_error"] == 0.0


def test_no_frontend_or_formal_src_is_needed_for_reconstruction():
    assert "frontend" not in b0_observations.__module__
    assert "sparse3d_forgery.b0" not in b0_observations.__module__
