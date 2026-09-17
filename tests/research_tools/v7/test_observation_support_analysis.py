"""Small deterministic checks for the read-only observation-support analysis."""

import numpy as np

from research_tools.v7.observation_support_pilot.analyze import (
    _bootstrap,
    _direction_class,
    _integer_counts,
    _transition,
)


def test_direction_classes_and_transitions() -> None:
    assert _direction_class([1.0, 2.0, 0.1]) == "ALL_POSITIVE"
    assert _direction_class([1.0, -1.0, 0.0]) == "CONTAINS_ZERO"
    assert _direction_class([1.0, -1.0, 0.5]) == "MIXED"
    assert _transition(1, -0.2, 0.3) == "FN_TO_TP"
    assert _transition(0, 0.2, -0.1) == "FP_TO_TN"


def test_q_fraction_rounding_preserves_integer_mask_counts() -> None:
    raw = np.asarray([3, 5], dtype=np.int64)
    q = np.zeros((2, 5, 4), dtype=np.float64)
    q[:, :, 0] = np.asarray([[1.0] * 5, [0.6] * 5])
    q[:, :, 1] = np.asarray([[1.0] * 5, [0.8] * 5])
    visibility, geometry = _integer_counts(q, raw)
    assert visibility.tolist() == [[3] * 5, [3] * 5]
    assert geometry.tolist() == [[3] * 5, [4] * 5]


def test_bootstrap_is_paired_and_reproducible() -> None:
    left = {"a": 0.5, "b": 0.75, "c": 0.0}
    right = {"a": 0.25, "b": 0.50, "c": 0.50}
    first = _bootstrap(left, right)
    second = _bootstrap(left, right)
    assert first == second
    assert first["source_count"] == 3
    assert first["mean"] == 0.0
