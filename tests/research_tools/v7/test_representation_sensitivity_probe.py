"""Focused contracts for the bounded representation-sensitivity probe."""

import numpy as np

from research_tools.v7.representation_sensitivity_probe.probe import (
    _apply_operation,
    _pairs_for_members,
    _selected_moved_members,
    _synthetic,
)


def test_pair_identity_and_moved_member_selection_are_deterministic() -> None:
    assert np.array_equal(_pairs_for_members([3, 1, 2]), np.asarray([[1, 2], [1, 3], [2, 3]]))
    first = _selected_moved_members([0, 1, 2, 3, 4, 5, 6, 7])
    second = _selected_moved_members([7, 6, 5, 4, 3, 2, 1, 0])
    assert np.array_equal(first, second)
    assert 1 <= first.size <= 7


def test_rigid_and_global_scale_interventions_preserve_s() -> None:
    sample = _synthetic()
    baseline = sample["base_s"]["s"]
    for operation in ("G1", "G2", "G3"):
        result = _apply_operation(sample, operation, True)
        assert result["status"] == "VALID"
        assert np.allclose(result["s"]["s"], baseline, atol=1e-5)
        assert np.allclose(result["q"]["q"], sample["base_q"]["q"], atol=1e-12)
    target_scale = _apply_operation(sample, "G4", True)
    assert target_scale["status"] == "VALID"
    assert not np.allclose(target_scale["s"]["s"], baseline, atol=1e-5)
    assert np.allclose(target_scale["q"]["q"], sample["base_q"]["q"], atol=1e-12)


def test_mask_m3_records_drop_and_recovery_without_readding_member() -> None:
    sample = _synthetic()
    result = _apply_operation(sample, "M3", True)
    assert result["status"] == "VALID"
    assert result["members"].size == sample["members"].size - 1
    visibility = result["q"]["q"][:, 0]
    assert np.array_equal(visibility, np.asarray([1.0, 1.0, 0.875, 1.0, 1.0]))
