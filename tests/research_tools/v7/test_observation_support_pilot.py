"""Research-contract checks for the observation-support pilot."""

from types import SimpleNamespace

import numpy as np
import torch

from research_tools.v7.observation_support_pilot.runner import (
    _initial_model,
    _q_for_unit,
    _record_matches,
)
from research_tools.v7.geometry_information_pilot.model import parameter_count


def _sequence(visibility: np.ndarray, geometry: np.ndarray) -> SimpleNamespace:
    return SimpleNamespace(
        timestamps_s=np.arange(6, dtype=np.float64),
        frame_indices=np.arange(6, dtype=np.int64) + 100,
        visibility=visibility,
        geometry_validity=geometry,
        uv=np.zeros((6, 4, 2), dtype=np.float32),
        xyz=np.zeros((6, 4, 3), dtype=np.float32),
        track_ids=np.arange(4, dtype=np.int64) + 10,
        provenance={"cohort_start_s": 0.0, "query_cohort": "R"},
    )


def _inputs() -> tuple[dict, dict, dict, SimpleNamespace]:
    row = {"window_id": "S::real::p::b05", "offset_s": 0.5, "grouping": {"history_array_indices": [0]}}
    identity = {"local_group_id": 4, "array_indices": [1, 2, 3, 4, 5], "track_ids": [10, 11, 12]}
    support_unit = {"member_slots": [0, 1, 2], "status": "VALID"}
    group = {"member_slots": [0, 1, 2, 3], "retained": True}
    return row, identity, support_unit, group


def test_q_uses_raw_history_members_not_common_member_count() -> None:
    visibility = np.ones((6, 4), dtype=bool)
    visibility[3:, 3] = False
    geometry = np.ones((6, 4), dtype=bool)
    row, identity, unit, group = _inputs()
    q, metadata, changed = _q_for_unit(row, identity, unit, _sequence(visibility, geometry), group)
    assert metadata["raw_member_count"] == 4
    assert metadata["common_member_count"] == 3
    assert np.allclose(q[:, 0], [1.0, 1.0, 0.75, 0.75, 0.75])
    assert np.allclose(q[:, 2], [0.0, 0.0, -0.25, -0.25, -0.25])
    assert changed == 1


def test_q_history_difference_is_zero_when_support_is_unchanged() -> None:
    visibility = np.ones((6, 4), dtype=bool)
    geometry = np.ones((6, 4), dtype=bool)
    row, identity, unit, group = _inputs()
    q, _, changed = _q_for_unit(row, identity, unit, _sequence(visibility, geometry), group)
    assert np.allclose(q[:, 2:], 0.0)
    assert changed == 0


def test_fused_and_duplicate_models_have_equal_width_and_extra_gradient() -> None:
    fused = _initial_model("STRUCTURE_SUPPORT", 20260909)
    control = _initial_model("STRUCTURE_DUP_CONTROL", 20260909)
    assert parameter_count(fused) == parameter_count(control) == 633
    states = torch.randn(4, 5, 8)
    intervals = torch.full((4, 4), 0.1)
    loss = fused(states, intervals).sum()
    loss.backward()
    assert torch.isfinite(fused.encoder[0].weight.grad[:, 4:]).all()
    assert torch.any(torch.abs(fused.encoder[0].weight.grad[:, 4:]) > 0)


def test_model_identity_rejects_changed_condition_or_input() -> None:
    expected = {"condition": "STRUCTURE_ONLY", "seed": 20260909, "feature_hash": "abc"}
    assert _record_matches({"input_identity": expected}, expected)
    assert not _record_matches({"input_identity": {**expected, "feature_hash": "other"}}, expected)
    assert not _record_matches({"input_identity": {**expected, "condition": "SUPPORT_ONLY"}}, expected)
