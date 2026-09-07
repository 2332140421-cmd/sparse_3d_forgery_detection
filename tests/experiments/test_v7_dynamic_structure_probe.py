import numpy as np
import pytest

from sparse3d_forgery.experiments.v7_dynamic_structure_probe import (
    ComponentConfig,
    motion_coherent_components,
    structural_differences,
    structure_state,
)


def sample_tracks():
    xyz = np.zeros((5, 4, 3), dtype=np.float32)
    xyz[:, :, 0] = np.array([0.0, 0.1, 0.2, 4.0])[None]
    xyz[:, :3, 1] = np.arange(5)[:, None] * 0.02
    return xyz, np.ones((5, 4), dtype=bool)


def test_grouping_is_deterministic_identity_only_and_motion_coherent():
    xyz, valid = sample_tracks()
    config = ComponentConfig(0.3, 0.01, minimum_size=3, minimum_overlap=4)
    assert motion_coherent_components(xyz, valid, config) == ((0, 1, 2),)
    assert motion_coherent_components(xyz.copy(), valid.copy(), config) == ((0, 1, 2),)


def test_structure_state_is_translation_invariant():
    xyz, valid = sample_tracks()
    first, first_valid = structure_state(xyz, valid, (0, 1, 2))
    shifted, shifted_valid = structure_state(xyz + np.array([9, -3, 2]), valid, (0, 1, 2))
    assert np.array_equal(first_valid, shifted_valid)
    assert np.allclose(first, shifted)


def test_structural_differences_use_real_timestamp_spacing():
    state = np.array([[0.0], [2.0], [8.0]])
    first, first_valid, second, second_valid = structural_differences(
        state, np.ones(3, bool), np.array([0.0, 2.0, 4.0])
    )
    assert first_valid.tolist() == [False, True, True]
    assert np.allclose(first[1:], [[1.0], [3.0]])
    assert second_valid.tolist() == [False, False, True]
    assert second[2, 0] == pytest.approx(1.0)


def test_structural_differences_reject_noncausal_timestamps():
    with pytest.raises(ValueError, match="strictly increasing"):
        structural_differences(np.ones((3, 1)), np.ones(3, bool), np.array([0.0, 1.0, 1.0]))
