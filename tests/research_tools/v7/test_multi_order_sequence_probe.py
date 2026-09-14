"""Targeted tests for the five-time research-only pilot."""

from types import SimpleNamespace

import numpy as np
import pytest

from research_tools.v7.multi_order_sequence_probe.model import (
    SequenceMLP,
    SetAModel,
    fit_standardizer,
    parameter_count,
    source_class_weights,
)
from research_tools.v7.multi_order_sequence_probe.representation import (
    CONDITIONS,
    build_five_time_unit,
    compute_derivatives,
    condition_inputs,
    deterministic_permutation,
)


def _sequence(times=None):
    times = np.asarray(times if times is not None else [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], dtype=np.float64)
    xyz = np.zeros((len(times), 4, 3), dtype=np.float32)
    xyz[:, 1, 0] = 1.0; xyz[:, 2, 1] = 1.0; xyz[:, 3, 2] = 1.0
    valid = np.ones((len(times), 4), dtype=bool)
    return SimpleNamespace(xyz=xyz, geometry_validity=valid, timestamps_s=times, frame_indices=np.arange(len(times)), track_ids=np.arange(4))


def test_five_target_frames_are_strict_and_share_members_pairs_and_scale():
    row = build_five_time_unit(_sequence(), window_id="w", window_start_s=0.0, member_slots=[0, 1, 2, 3], local_group_id=2)
    assert row["status"] == "VALID"
    assert len(row["frame_indices"]) == 5
    assert np.all(np.diff(row["timestamps_s"]) > 0)
    assert len(row["member_slots"]) == len(row["track_ids"])
    assert row["history_scale"] > 0
    assert row["states"].shape == (5, 4)


def test_repeated_or_nonmonotonic_timestamps_are_rejected():
    row = build_five_time_unit(_sequence([0, .1, .2, .3, .4, .5, .6, .6, .8, .9]), window_id="w", window_start_s=0.0, member_slots=[0, 1, 2], local_group_id=0)
    assert row["status"] == "INVALID"
    assert row["reason"] == "SEQUENCE_TIMESTAMPS_NOT_STRICT"


def test_missing_geometry_does_not_get_filled():
    sequence = _sequence(); sequence.geometry_validity[6, 2] = False
    row = build_five_time_unit(sequence, window_id="w", window_start_s=0.0, member_slots=[0, 1, 2], local_group_id=0)
    assert row["status"] == "INVALID"
    assert row["reason"] == "COMMON_VALID_MEMBERS_LT3"


def test_timestamp_aware_derivatives_handle_unequal_intervals():
    times = np.asarray([0., .1, .3, .6, 1.0])
    states = np.stack([times ** 2] * 4, axis=1)
    velocity, acceleration = compute_derivatives(states, times)
    np.testing.assert_allclose(velocity[:, 0], [.1, .4, .9, 1.6])
    np.testing.assert_allclose(acceleration[:, 0], [2., 2., 2.])


def test_condition_shapes_and_shuffle_recomputes_derivatives():
    states = np.arange(20, dtype=np.float64).reshape(5, 4)
    result = condition_inputs(states, [0., .1, .25, .4, .6], "window")
    assert result["RAW_SEQ"].shape == (40,)
    assert result["MULTI_ORDER_SEQ"].shape == (40,)
    assert result["SHUFFLED_MULTI_ORDER"].shape == (40,)
    assert tuple(result["permutation"]) != tuple(range(5))
    assert not np.array_equal(result["MULTI_ORDER_SEQ"], result["SHUFFLED_MULTI_ORDER"])


def test_set_a_is_permutation_invariant_after_shared_encoding_input():
    states = np.random.default_rng(2).normal(size=(5, 4))
    values = condition_inputs(states, [0., .1, .2, .3, .4], "x")["SET_A_states"]
    shuffled = values[[2, 0, 4, 1, 3]]
    model = SetAModel()
    intervals = np.ones((1, 4), dtype=np.float32) * .1
    import torch
    with torch.no_grad():
        first = model(torch.as_tensor(values[None], dtype=torch.float32), torch.as_tensor(intervals))
        second = model(torch.as_tensor(shuffled[None], dtype=torch.float32), torch.as_tensor(intervals))
    np.testing.assert_allclose(first.numpy(), second.numpy(), atol=1e-6)


def test_sequence_conditions_have_equal_parameters_and_fixed_permutation():
    assert parameter_count("RAW_SEQ") == parameter_count("MULTI_ORDER_SEQ") == parameter_count("SHUFFLED_MULTI_ORDER")
    assert deterministic_permutation("a") == deterministic_permutation("a")
    assert deterministic_permutation("a") != tuple(range(5))


def test_standardizers_share_channels_and_exclude_invalid_values():
    matrices = [np.ones((2, 40)), np.full((1, 40), 3.0)]
    standardizer = fit_standardizer("MULTI_ORDER_SEQ", matrices, np.asarray([1.0, 1.0]))
    assert standardizer.mean.shape == (16,)
    assert np.all(np.isfinite(standardizer.transform(matrices[0])))


def test_source_class_weights_balance_source_and_class():
    rows = [
        {"source_id": "a", "label": 0}, {"source_id": "a", "label": 0}, {"source_id": "a", "label": 1},
        {"source_id": "b", "label": 0}, {"source_id": "b", "label": 1}, {"source_id": "b", "label": 1},
    ]
    weights = source_class_weights(rows)
    assert np.isclose(weights.mean(), 1.0)
    for source in ("a", "b"):
        for label in (0, 1):
            assert np.isclose(weights[[r["source_id"] == source and r["label"] == label for r in rows]].sum(), weights[[r["source_id"] == "a" and r["label"] == 0 for r in rows]].sum())


def test_conditions_are_frozen_names():
    assert CONDITIONS == ("SET_A", "RAW_SEQ", "MULTI_ORDER_SEQ", "SHUFFLED_MULTI_ORDER")
