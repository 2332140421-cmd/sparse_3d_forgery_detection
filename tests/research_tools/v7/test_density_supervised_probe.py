from __future__ import annotations

import numpy as np
import pytest

from research_tools.v7.density_supervised_probe.pilot import (
    _paired_bootstrap,
    _predeclared_status,
    _source_metric_rows,
)


def _source_row(source: str, c289: float, a289: float, d289: float, c64: float) -> dict:
    return {
        "source_id": source,
        "density289_ORDERED_SECOND_auroc": c289,
        "density289_UNORDERED_STATE_auroc": a289,
        "density289_PERMUTED_SECOND_auroc": d289,
        "density64_ORDERED_SECOND_auroc": c64,
    }


def test_paired_bootstrap_uses_original_source_means_for_point_estimates():
    rows = [_source_row(f"S{i}", 0.8, 0.5, 0.4, 0.3) for i in range(12)]
    result = _paired_bootstrap(rows)
    assert result["N"] == 12
    assert result["arms"]["density289_ORDERED_SECOND_auroc"]["mean"] == pytest.approx(0.8)
    assert result["gains"]["C289-A289"]["mean"] == 0.3
    assert result["gains"]["C289-D289"]["mean"] == pytest.approx(0.4)
    assert result["gains"]["C289-C64"]["mean"] == 0.5


def test_predeclared_status_does_not_call_point_estimate_a_supported_ci():
    bootstrap = {
        "arms": {"density289_ORDERED_SECOND_auroc": {"ci95": [0.51, 0.9]}},
        "gains": {
            "C289-A289": {"ci95": [0.01, 0.2]},
            "C289-D289": {"ci95": [-0.1, 0.2]},
            "C289-C64": {"ci95": [0.01, 0.2]},
        },
    }
    status = _predeclared_status(bootstrap, 12)
    assert status["classification"] == "C289_DISCRIMINATION_SUPPORTED_IN_PILOT"
    assert status["temporal_organization"] == "C289_TEMPORAL_ORGANIZATION_NOT_ESTABLISHED"
    assert status["density_gain"] == "DENSITY_289_GAIN_SUPPORTED_IN_PILOT"


def test_source_metrics_keep_density_and_arm_pairs_separate():
    rows = [
        {"source_id": "S1", "kind": "MANIP", "role": "real", "density289_ORDERED_SECOND_score": 0.0, "density289_UNORDERED_STATE_score": 0.0, "density289_PERMUTED_SECOND_score": 0.0, "density64_ORDERED_SECOND_score": 0.0},
        {"source_id": "S1", "kind": "MANIP", "role": "fake", "density289_ORDERED_SECOND_score": 1.0, "density289_UNORDERED_STATE_score": 0.5, "density289_PERMUTED_SECOND_score": 0.2, "density64_ORDERED_SECOND_score": 0.1},
    ]
    result = _source_metric_rows(rows)
    assert len(result) == 1
    assert result[0]["density289_ORDERED_SECOND_auroc"] == 1.0
    assert result[0]["C289-A289"] == 0.0
    assert result[0]["C289-C64"] == 0.0


def test_empty_bootstrap_is_explicitly_unscored():
    result = _paired_bootstrap([])
    assert result["N"] == 0
    assert result["gains"] == {}
    assert np.isfinite(result["seed"])
