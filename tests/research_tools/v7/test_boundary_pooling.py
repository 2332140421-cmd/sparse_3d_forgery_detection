from __future__ import annotations

import numpy as np
import torch

from research_tools.v7.boundary_pooling_probe.grouping import (
    assign_uv_to_masks,
    split_local_groups_by_assignment,
    validate_boundary_partition,
)
from research_tools.v7.boundary_pooling_probe.model import PoolingWindowMLP
from research_tools.v7.local_structural_temporal_probe.model import Batch, WindowMLP, prepare_batch_tensors


def _groups() -> dict[str, object]:
    return {
        "local_diameter_m": 0.30,
        "component_config": {"minimum_size": 3},
        "groups": [{"local_group_id": 0, "parent_component_id": 0, "member_slots": [0, 1, 2, 3], "track_ids": [10, 11, 12, 13], "retained": True}],
    }


def test_no_mask_keeps_valid_points_as_one_background_child() -> None:
    assignment = assign_uv_to_masks(np.asarray([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [1.0, 3.0]]), np.asarray([True, True, True, True]), [], [])
    assert assignment.tolist() == [0, 0, 0, 0]
    split = split_local_groups_by_assignment(_groups(), assignment, minimum_size=3)
    assert split["groups"][0]["member_slots"] == [0, 1, 2, 3]
    assert split["groups"][0]["retained"] is True


def test_boundary_children_are_parent_subsets_and_disjoint() -> None:
    split = split_local_groups_by_assignment(_groups(), np.asarray([1, 1, 2, 0]), minimum_size=1)
    assert {tuple(x["member_slots"]) for x in split["groups"]} == {(0, 1), (2,), (3,)}
    assert validate_boundary_partition(_groups(), split)["all_pass"]


def test_overlapping_masks_choose_smaller_area_then_index() -> None:
    masks = [np.ones((4, 4), dtype=bool), np.ones((4, 4), dtype=bool)]
    result = assign_uv_to_masks(np.asarray([[1.0, 1.0]]), np.asarray([True]), masks, [10.0, 3.0])
    assert result.tolist() == [2]
    result = assign_uv_to_masks(np.asarray([[1.0, 1.0]]), np.asarray([True]), masks, [3.0, 3.0])
    assert result.tolist() == [1]


def test_invalid_uv_is_explicitly_unassigned() -> None:
    result = assign_uv_to_masks(np.asarray([[np.nan, 1.0], [99.0, 2.0]]), np.asarray([True, False]), [], [])
    assert result.tolist() == [-1, -1]


def _triplet(component: int, offset: float) -> dict[str, object]:
    return {"component_index": component, "states": (np.asarray([[1, 2, 3, 4], [1 + offset, 2, 3, 4], [1 + 2 * offset, 2, 3, 4]], dtype=float)).tolist(), "timestamps_s": [0.0, 0.1, 0.2], "pair_ids": [[1, 2]], "common_track_ids": [1, 2]}


def _batch() -> Batch:
    from research_tools.v7.local_structural_temporal_probe.model import build_batch

    examples = [
        {"window_id": "w0", "source_id": "s0", "kind": "MANIP", "role": "real", "triplets": [_triplet(0, 0.1), _triplet(1, 0.2)]},
        {"window_id": "w1", "source_id": "s0", "kind": "MANIP", "role": "fake", "triplets": [_triplet(0, 0.3)]},
        {"window_id": "w2", "source_id": "s1", "kind": "MANIP", "role": "real", "triplets": [_triplet(0, 0.4)]},
        {"window_id": "w3", "source_id": "s1", "kind": "MANIP", "role": "fake", "triplets": [_triplet(0, 0.5)]},
    ]
    batch, _ = build_batch(examples, "ORDERED_SECOND", observation_weights=True)
    return batch


def test_mean_wrapper_matches_existing_window_mlp() -> None:
    batch = _batch()
    old = WindowMLP()
    new = PoolingWindowMLP("mean")
    new.load_state_dict(old.state_dict())
    prepared = prepare_batch_tensors(batch, "cpu")
    old_value = old._forward_prepared(batch, prepared)
    new_value = new._forward_prepared(batch, prepared)
    assert torch.allclose(old_value, new_value, atol=1e-7, rtol=0.0)


def test_max_is_group_level_and_backpropagates() -> None:
    batch = _batch()
    model = PoolingWindowMLP("max")
    prepared = prepare_batch_tensors(batch, "cpu")
    output = model._forward_prepared(batch, prepared)
    assert output.shape == (4,)
    assert sum(parameter.numel() for parameter in model.parameters()) == sum(parameter.numel() for parameter in PoolingWindowMLP("mean").parameters())
    loss = output.sum(); loss.backward()
    assert all(parameter.grad is not None for parameter in model.parameters())
