"""Contracts for the case-specific frozen five-time readout."""

from types import SimpleNamespace

import numpy as np

from research_tools.v7.multi_order_sequence_probe.representation import build_five_time_unit
from research_tools.v7.observation_support_pilot.runner import _q_for_unit as observation_q_for_unit


def _sequence() -> SimpleNamespace:
    timestamps = np.arange(10, dtype=np.float64) / 10.0
    base = np.asarray(
        [[0.0, 0.0, 1.0], [0.1, 0.0, 1.0], [0.0, 0.1, 1.0]],
        dtype=np.float32,
    )
    xyz = np.stack([base + np.asarray([0.001 * index, 0.0, 0.0], dtype=np.float32) for index in range(10)])
    return SimpleNamespace(
        frame_indices=np.arange(10, dtype=np.int64) + 100,
        timestamps_s=timestamps,
        frame_sizes_hw=np.tile(np.asarray([[360, 480]], dtype=np.int64), (10, 1)),
        track_ids=np.arange(3, dtype=np.int64),
        uv=np.ones((10, 3, 2), dtype=np.float32),
        visibility=np.ones((10, 3), dtype=bool),
        geometry_validity=np.ones((10, 3), dtype=bool),
        xyz=xyz,
        provenance={"cohort_start_s": 0.0, "query_cohort": "R::test"},
    )


def test_current_builder_produces_five_time_s_without_fill():
    sequence = _sequence()
    unit = build_five_time_unit(
        sequence,
        window_id="test::fake",
        window_start_s=0.0,
        member_slots=[0, 1, 2],
        local_group_id=7,
    )
    assert unit["status"] == "VALID"
    assert unit["frame_indices"] == [105, 106, 107, 108, 109]
    assert np.asarray(unit["states"]).shape == (5, 4)
    assert np.all(np.isfinite(unit["states"]))


def test_q_reuses_raw_group_denominator_and_explicit_history_reference():
    sequence = _sequence()
    unit = build_five_time_unit(
        sequence,
        window_id="test::fake",
        window_start_s=0.0,
        member_slots=[0, 1, 2],
        local_group_id=7,
    )
    row = {"window_id": "test::fake", "grouping": {"history_array_indices": [0, 1, 2, 3, 4]}}
    identity = {"local_group_id": 7, "array_indices": unit["array_indices"], "track_ids": unit["track_ids"]}
    q, metadata, changed = observation_q_for_unit(row, identity, unit, sequence, {"member_slots": [0, 1, 2]})
    assert q.shape == (5, 4)
    assert np.allclose(q[:, :2], 1.0)
    assert np.allclose(q[:, 2:], 0.0)
    assert metadata["raw_member_count"] == 3
    assert metadata["history_reference_frame_index"] == 104
    assert changed == 0


def test_common_member_failure_is_explicit():
    sequence = _sequence()
    sequence.geometry_validity[8, 2] = False
    sequence.geometry_validity[9, 2] = False
    sequence.xyz[8:, 2] = np.nan
    unit = build_five_time_unit(
        sequence,
        window_id="test::fake",
        window_start_s=0.0,
        member_slots=[0, 1, 2],
        local_group_id=7,
    )
    assert unit["status"] == "INVALID"
    assert unit["reason"] == "COMMON_VALID_MEMBERS_LT3"
