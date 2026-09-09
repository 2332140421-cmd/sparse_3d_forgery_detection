"""Tests for the one fixed B0 supervised LOSO probe."""

from __future__ import annotations

import numpy as np
import pytest

from research_tools.v7.b0_supervised_linear_probe.probe import (
    MODEL_CONFIG,
    fit_fold,
    fit_weighted_standardizer,
    main_training_label,
    paired_source_bootstrap,
    primary_status,
    source_auc,
    source_class_weights,
    source_group_medians,
    validate_main_rows,
)


def row(source: str, role: str, kind: str = "MANIP", value: float = 0.0) -> dict:
    return {"source_id": source, "role": role, "kind": kind, "b0": [value, value, value, value]}


def training_rows() -> list[dict]:
    return [row(source, role, value=(1.0 if role == "fake" else 0.0) + index * 0.1) for source in ("a", "b") for index, role in enumerate(("real", "fake"))]


def test_main_label_protocol_is_only_real_fake_manip():
    assert main_training_label(row("a", "real")) == 0
    assert main_training_label(row("a", "fake")) == 1
    assert main_training_label(row("a", "real", "CTRL")) is None


def test_control_windows_are_not_main_training_rows():
    with pytest.raises(ValueError):
        validate_main_rows([row("a", "real", "CTRL"), row("a", "fake", "CTRL")])


def test_each_training_source_requires_both_classes():
    with pytest.raises(ValueError):
        validate_main_rows([row("a", "real"), row("b", "real"), row("b", "fake")])


def test_source_class_weights_equalize_source_and_class_mass_and_mean_one():
    rows = training_rows() + [row("a", "real", value=0.2), row("a", "fake", value=1.2)]
    weights = source_class_weights(rows)
    assert np.mean(weights) == pytest.approx(1.0)
    for source in ("a", "b"):
        for label in (0, 1):
            mass = sum(weight for item, weight in zip(rows, weights) if item["source_id"] == source and main_training_label(item) == label)
            assert mass == pytest.approx(0.5 * len(rows) / len({"a", "b"}))


def test_weighted_standardizer_uses_training_values_and_fixed_zero_scale():
    standardizer = fit_weighted_standardizer([[0, 0, 1, 2], [2, 0, 3, 4]], [1, 1])
    assert standardizer.zero_variance_dimensions == (1,)
    assert standardizer.scale[1] == 1.0
    assert np.allclose(standardizer.transform([[1, 0, 2, 3]]), [[0, 0, 0, 0]])


def test_weighted_standardizer_rejects_bad_inputs():
    with pytest.raises(ValueError):
        fit_weighted_standardizer([[0, 1, 2]], [1])
    with pytest.raises(ValueError):
        fit_weighted_standardizer([[0, 1, 2, float("nan")]], [1])


def test_fold_excludes_held_out_source_from_training():
    rows = training_rows() + [row("held", "real", value=5), row("held", "fake", value=6)]
    fitted = fit_fold(rows, "held")
    assert fitted.held_out_source == "held"
    assert fitted.training_source_count == 2
    assert fitted.training_real_count == 2
    assert fitted.training_fake_count == 2


def test_fold_produces_four_coefficients_and_decision_scores():
    fitted = fit_fold(training_rows() + [row("held", "real", value=5), row("held", "fake", value=6)], "held")
    scores = fitted.score([[5, 5, 5, 5], [6, 6, 6, 6]])
    assert fitted.coefficient.shape == (4,)
    assert scores.shape == (2,)


def test_standardization_and_fit_ignore_held_out_values():
    base = training_rows() + [row("held", "real", value=5), row("held", "fake", value=6)]
    changed = training_rows() + [row("held", "real", value=500), row("held", "fake", value=600)]
    first = fit_fold(base, "held")
    second = fit_fold(changed, "held")
    assert np.allclose(first.standardizer.mean, second.standardizer.mean)
    assert np.allclose(first.standardizer.scale, second.standardizer.scale)
    assert np.allclose(first.coefficient, second.coefficient)
    assert first.intercept == pytest.approx(second.intercept)


def test_fixed_logistic_configuration_is_not_searchable():
    assert MODEL_CONFIG == {
        "model": "logistic_regression",
        "penalty": "l2",
        "C": 1.0,
        "solver": "lbfgs",
        "fit_intercept": True,
        "max_iter": 1000,
        "tol": 1e-6,
        "class_weight": None,
        "random_state": None,
    }


def test_source_auc_is_computed_per_source():
    rows = [
        {"source_id": "a", "role": "real", "kind": "MANIP", "score": 0.1},
        {"source_id": "a", "role": "fake", "kind": "MANIP", "score": 0.9},
        {"source_id": "b", "role": "real", "kind": "MANIP", "score": 0.8},
        {"source_id": "b", "role": "fake", "kind": "MANIP", "score": 0.2},
    ]
    result = source_auc(rows)
    assert [item["auroc"] for item in result] == [1.0, 0.0]


def test_source_auc_ignores_control_for_primary_metric():
    rows = [
        {"source_id": "a", "role": "real", "kind": "MANIP", "score": 0.1},
        {"source_id": "a", "role": "fake", "kind": "MANIP", "score": 0.9},
        {"source_id": "a", "role": "real", "kind": "CTRL", "score": 100.0},
        {"source_id": "a", "role": "fake", "kind": "CTRL", "score": -100.0},
    ]
    assert source_auc(rows)[0]["auroc"] == 1.0


def test_source_group_medians_preserve_delta_and_j_definitions():
    rows = []
    for role, kind, value in (("real", "MANIP", 1), ("fake", "MANIP", 3), ("real", "CTRL", 2), ("fake", "CTRL", 2.5)):
        rows.append({"source_id": "a", "role": role, "kind": kind, "score": value})
    result = source_group_medians(rows)[0]
    assert result["Delta_manip"] == 2
    assert result["Delta_control"] == pytest.approx(0.5)
    assert result["J_B0"] == pytest.approx(1.5)


def test_paired_bootstrap_uses_same_source_indices_for_model_and_baseline():
    rows = [{"model_auroc": 0.8, "baseline_auroc": 0.6}, {"model_auroc": 0.7, "baseline_auroc": 0.7}, {"model_auroc": 0.9, "baseline_auroc": 0.5}]
    first = paired_source_bootstrap(rows)
    second = paired_source_bootstrap(rows)
    assert first == second
    assert first["delta_mean"] == pytest.approx((0.2 + 0 + 0.4) / 3)
    assert first["seed"] == 20260909 and first["replicates"] == 10000


def test_primary_status_requires_model_and_gain_lower_bounds():
    supported = {"model_ci95": [0.6, 0.9], "delta_ci95": [0.1, 0.3]}
    no_gain = {"model_ci95": [0.6, 0.9], "delta_ci95": [-0.1, 0.3]}
    failed = {"model_ci95": [0.4, 0.9], "delta_ci95": [0.1, 0.3]}
    assert primary_status(supported) == "B0_LINEAR_DISCRIMINATION_SUPPORTED_IN_PILOT"
    assert primary_status(no_gain) == "B0_LINEAR_SIGNAL_WITHOUT_CLEAR_BASELINE_GAIN"
    assert primary_status(failed) == "B0_LINEAR_DISCRIMINATION_NOT_ESTABLISHED"


def test_blocked_status_is_reserved_for_input_or_isolation_failure():
    assert primary_status({}, blocked=True) == "B0_LINEAR_PROBE_BLOCKED"


def test_scores_use_only_b0_numeric_vector():
    rows = training_rows() + [row("held", "real", value=5), row("held", "fake", value=6)]
    fitted = fit_fold(rows, "held")
    first = fitted.score([[5, 5, 5, 5]])
    rows[0]["source_id"] = "changed-metadata"
    rows[0]["role"] = "fake"
    assert fitted.score([[5, 5, 5, 5]]) == pytest.approx(first)


def test_control_and_invalid_source_can_be_reported_without_fitting():
    rows = training_rows() + [row("held", "real", value=5), row("held", "fake", value=6), row("held", "real", "CTRL", value=7)]
    fitted = fit_fold(rows[:4] + rows[4:6], "held")
    assert fitted.training_real_count == fitted.training_fake_count == 2
