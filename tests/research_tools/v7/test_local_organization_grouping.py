from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from research_tools.v7.local_organization_probe.grouping import (
    LOCAL_DIAMETER_M,
    build_local_groups,
    build_local_support,
    rebuild_components_fast,
    validate_grouping,
)
from research_tools.v7.local_organization_probe.run_pilot import _visual_member_slots
from sparse3d_forgery.experiments.v7_dynamic_structure_probe import motion_coherent_components
from research_tools.v7.local_structural_temporal_probe.representation import COMPONENT_CONFIG


def _sequence(*, positions: list[float] | None = None, reorder: np.ndarray | None = None) -> SimpleNamespace:
    times = np.arange(20, dtype=np.float64) * 0.05
    base = np.asarray(positions or [0.0, 0.1, 0.2, 0.5, 0.6, 0.7], dtype=np.float64)
    xyz = np.stack([np.stack([base, np.zeros_like(base), np.zeros_like(base)], axis=1) for _ in times])
    valid = np.ones(xyz.shape[:2], dtype=bool)
    track_ids = np.arange(base.size, dtype=np.int64) + 100
    if reorder is not None:
        xyz = xyz[:, reorder]
        valid = valid[:, reorder]
        track_ids = track_ids[reorder]
    return SimpleNamespace(
        xyz=xyz,
        geometry_validity=valid,
        track_ids=track_ids,
        timestamps_s=times,
        frame_indices=np.arange(times.size, dtype=np.int64),
    )


def test_chain_component_is_split_into_local_diameter_groups():
    sequence = _sequence()
    grouping = build_local_groups(sequence, np.arange(10))
    retained = [tuple(row["track_ids"]) for row in grouping["groups"] if row["retained"]]
    assert retained == [(100, 101, 102), (103, 104, 105)]
    assert validate_grouping(grouping, sequence.track_ids)["all_pass"]
    assert all(
        max(sequence.xyz[0, [sequence.track_ids.tolist().index(track) for track in group], 0])
        - min(sequence.xyz[0, [sequence.track_ids.tolist().index(track) for track in group], 0])
        <= LOCAL_DIAMETER_M
        for group in retained
    )


def test_vectorized_old_component_rebuild_matches_existing_rule():
    sequence = _sequence()
    expected = motion_coherent_components(sequence.xyz[:10], sequence.geometry_validity[:10], COMPONENT_CONFIG)
    actual = rebuild_components_fast(sequence.xyz, sequence.geometry_validity, np.arange(10))
    assert sorted(tuple(sorted(item)) for item in actual) == sorted(tuple(sorted(item)) for item in expected)


def test_indirect_old_connection_cannot_cross_local_group_diameter():
    sequence = _sequence(positions=[0.0, 0.1, 0.2, 1.0, 1.1, 1.2])
    grouping = build_local_groups(sequence, np.arange(10))
    retained = [set(row["track_ids"]) for row in grouping["groups"] if row["retained"]]
    assert retained == [{100, 101, 102}, {103, 104, 105}]
    parent = grouping["parent_components"][0]
    pair = next(item for item in parent["pair_evidence"] if item["left_slot"] == 0 and item["right_slot"] == 3)
    assert pair["direct_edge"] is True
    assert pair["merge_allowed"] is False
    assert pair["rejection_reason"] == "LOCAL_DIAMETER_EXCEEDED"


def test_insufficient_history_never_becomes_pair_support():
    sequence = _sequence()
    sequence.geometry_validity[8:10, 0] = False
    sequence.geometry_validity[:2, 2] = False
    grouping = build_local_groups(sequence, np.arange(10))
    assert all(102 not in row["track_ids"] for row in grouping["groups"] if row["retained"])
    all_pairs = [pair for parent in grouping["parent_components"] for pair in parent["pair_evidence"]]
    pair = next(item for item in all_pairs if item["left_slot"] == 0 and item["right_slot"] == 2)
    assert pair["history_overlap_count"] == 6
    assert pair["direct_edge"] is False
    assert pair["merge_allowed"] is False


def test_evaluation_changes_do_not_change_history_groups_or_scales():
    sequence = _sequence()
    grouping = build_local_groups(sequence, np.arange(10))
    support = build_local_support(sequence, window_start_s=0.0, grouping=grouping)
    changed = _sequence()
    changed.xyz[10:, 0, 0] += 100.0
    changed_grouping = build_local_groups(changed, np.arange(10))
    changed_support = build_local_support(changed, window_start_s=0.0, grouping=changed_grouping)
    assert grouping["groups"] == changed_grouping["groups"]
    scales = [row["history_scale"] for row in support["components"]]
    changed_scales = [row["history_scale"] for row in changed_support["components"]]
    assert scales == changed_scales


def test_track_order_changes_preserve_id_canonical_groups():
    sequence = _sequence()
    reorder = np.asarray([3, 0, 5, 2, 1, 4], dtype=np.int64)
    reordered = _sequence(reorder=reorder)
    first = build_local_groups(sequence, np.arange(10))
    second = build_local_groups(reordered, np.arange(10))
    first_ids = [tuple(row["track_ids"]) for row in first["groups"] if row["retained"]]
    second_ids = [tuple(row["track_ids"]) for row in second["groups"] if row["retained"]]
    assert first_ids == second_ids


def test_local_support_keeps_common_pair_identity_and_missing_slots_explicit():
    sequence = _sequence()
    grouping = build_local_groups(sequence, np.arange(10))
    sequence.geometry_validity[14, 0] = False
    support = build_local_support(sequence, window_start_s=0.0, grouping=grouping)
    assert support["support_status"] == "VALID"
    assert support["triplets"]
    for triplet in support["triplets"]:
        expected = {
            tuple(sorted((int(left), int(right))))
            for left in triplet["common_member_indices"]
            for right in triplet["common_member_indices"]
            if left < right
        }
        assert {tuple(pair) for pair in triplet["pair_indices"]} == expected
        assert len(triplet["timestamps_s"]) == 3


def test_small_groups_are_recorded_but_not_retained():
    sequence = _sequence(positions=[0.0, 0.1, 0.5, 0.6, 1.0, 1.1])
    grouping = build_local_groups(sequence, np.arange(10))
    assert grouping["support_insufficient_group_count"] > 0
    assert any(row["status"] == "SUPPORT_INSUFFICIENT_GROUP_SIZE" for row in grouping["groups"])


def test_visualization_reads_old_and_local_member_fields_separately():
    assert _visual_member_slots({"member_indices": [1, 2]}, local=False) == (1, 2)
    assert _visual_member_slots({"member_slots": [3, 4]}, local=True) == (3, 4)
