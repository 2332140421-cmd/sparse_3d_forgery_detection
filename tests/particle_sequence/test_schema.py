from dataclasses import FrozenInstanceError, fields

import numpy as np
import pytest

from sparse3d_forgery.particle_sequence import ParticleSequence


def test_fields_and_derived_properties(valid_sequence):
    assert valid_sequence.num_frames == 3
    assert valid_sequence.num_tracks == 2
    np.testing.assert_array_equal(
        valid_sequence.observation_validity,
        valid_sequence.visibility & valid_sequence.geometry_validity,
    )
    assert valid_sequence.track_ids.tolist() == [7, 42]


def test_mid_sequence_birth_is_representable(valid_sequence):
    assert not valid_sequence.visibility[0, 1]
    assert valid_sequence.visibility[1, 1]


def test_disappearance_and_reappearance_keep_slot(valid_sequence):
    visibility = valid_sequence.visibility.copy()
    visibility[:, 0] = [True, False, True]
    assert visibility[:, 0].tolist() == [True, False, True]
    assert valid_sequence.track_ids[0] == 7


def test_frozen_dataclass_prevents_field_rebinding(valid_sequence):
    with pytest.raises(FrozenInstanceError):
        valid_sequence.sample_id = "other"


def test_forbidden_fields_are_absent():
    names = {field.name for field in fields(ParticleSequence)}
    forbidden = {
        "label", "split", "generator", "velocity", "acceleration",
        "jerk", "residual", "padding",
    }
    assert names.isdisjoint(forbidden)
