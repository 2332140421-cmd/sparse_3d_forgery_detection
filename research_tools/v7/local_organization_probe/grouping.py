"""History-only local grouping and support construction for one V7 pilot.

This module deliberately keeps the existing component and state definitions.
It only restricts the relation set inside each existing component with a
predeclared complete-linkage diameter bound.  Evaluation frames never enter
group formation or history-scale calculation.
"""

from __future__ import annotations

from itertools import combinations
import heapq
from typing import Any, Mapping, Sequence
import warnings

import numpy as np

from research_tools.v7.local_structural_temporal_probe.representation import (
    COMPONENT_CONFIG,
    MAX_TARGET_ERROR_S,
    TARGET_OFFSETS_S,
    _history_scale,
    _state_from_common_members,
    _target_frame_matches,
)
from sparse3d_forgery.experiments.v7_dynamic_structure_probe import (
    ComponentConfig,
    motion_coherent_components,
)


LOCAL_DIAMETER_M = 0.30


def rebuild_components_fast(
    xyz: np.ndarray,
    valid: np.ndarray,
    history_indices: Sequence[int],
    config: ComponentConfig = COMPONENT_CONFIG,
) -> tuple[tuple[int, ...], ...]:
    """Vectorized reproduction of the existing direct-edge component rule.

    The numerical predicates are the same as
    ``motion_coherent_components``; only the pairwise implementation is
    vectorized so that the 289-point pilot remains a bounded CPU experiment.
    """

    points = np.asarray(xyz, dtype=np.float64)[np.asarray(history_indices, dtype=np.int64)]
    observed = np.asarray(valid, dtype=bool)[np.asarray(history_indices, dtype=np.int64)]
    if points.ndim != 3 or points.shape[-1] != 3 or observed.shape != points.shape[:2]:
        raise ValueError("expected history xyz [H,N,3] and valid [H,N]")
    count_n = points.shape[1]
    finite_observed = observed & np.all(np.isfinite(points), axis=-1)
    pair_observed = finite_observed[:, :, None] & finite_observed[:, None, :]
    overlap = np.sum(pair_observed, axis=0)
    relative = points[:, None, :, :] - points[:, :, None, :]
    distances = np.linalg.norm(relative, axis=-1)
    first_index = np.argmax(pair_observed, axis=0)
    rows = np.arange(count_n)[:, None]
    cols = np.arange(count_n)[None, :]
    initial = distances[first_index, rows, cols]
    masked_relative = np.where(pair_observed[..., None], relative, np.nan)
    with warnings.catch_warnings(), np.errstate(all="ignore"):
        warnings.simplefilter("ignore", RuntimeWarning)
        median_relative = np.nanmedian(masked_relative, axis=0)
        change = np.linalg.norm(relative - median_relative[None, ...], axis=-1)
        median_change = np.nanmedian(np.where(pair_observed, change, np.nan), axis=0)
    edge_matrix = (
        (overlap >= int(config.minimum_overlap))
        & (initial <= float(config.max_initial_distance))
        & (median_change <= float(config.max_relative_change))
    )
    adjacency = [set() for _ in range(count_n)]
    for left in range(count_n):
        for right in range(left + 1, count_n):
            if bool(edge_matrix[left, right]):
                adjacency[left].add(right)
                adjacency[right].add(left)
    seen: set[int] = set()
    components: list[tuple[int, ...]] = []
    for seed in range(count_n):
        if seed in seen:
            continue
        stack = [seed]
        group: list[int] = []
        seen.add(seed)
        while stack:
            node = stack.pop()
            group.append(node)
            for neighbor in sorted(adjacency[node], reverse=True):
                if neighbor not in seen:
                    seen.add(neighbor)
                    stack.append(neighbor)
        if len(group) >= int(config.minimum_size):
            components.append(tuple(sorted(group)))
    return tuple(components)


def _track_key(track_ids: np.ndarray, members: Sequence[int]) -> tuple[int, ...]:
    return tuple(sorted(int(track_ids[index]) for index in members))


def _pair_key(left: int, right: int) -> tuple[int, int]:
    return (left, right) if left < right else (right, left)


def _pair_evidence(
    xyz: np.ndarray,
    valid: np.ndarray,
    history_indices: np.ndarray,
    left: int,
    right: int,
    config: ComponentConfig,
    local_diameter_m: float,
) -> dict[str, Any]:
    """Reproduce the existing direct-edge test and add the history diameter."""

    pair_valid = (
        valid[history_indices, left]
        & valid[history_indices, right]
        & np.all(np.isfinite(xyz[history_indices, left]), axis=1)
        & np.all(np.isfinite(xyz[history_indices, right]), axis=1)
    )
    overlap_indices = history_indices[pair_valid]
    evidence: dict[str, Any] = {
        "left_slot": int(left),
        "right_slot": int(right),
        "history_overlap_count": int(overlap_indices.size),
        "direct_edge": False,
        "d_history_m": None,
        "initial_distance_m": None,
        "relative_change_median_m": None,
        "merge_allowed": False,
        "rejection_reason": "INSUFFICIENT_HISTORY_OVERLAP",
    }
    if overlap_indices.size == 0:
        return evidence
    relative = xyz[overlap_indices, right] - xyz[overlap_indices, left]
    distances = np.linalg.norm(relative, axis=1)
    evidence["d_history_m"] = float(np.max(distances))
    evidence["initial_distance_m"] = float(np.linalg.norm(relative[0]))
    change = np.linalg.norm(relative - np.median(relative, axis=0), axis=1)
    evidence["relative_change_median_m"] = float(np.median(change))
    direct = (
        int(overlap_indices.size) >= int(config.minimum_overlap)
        and evidence["initial_distance_m"] <= float(config.max_initial_distance)
        and evidence["relative_change_median_m"] <= float(config.max_relative_change)
    )
    evidence["direct_edge"] = bool(direct)
    if not direct:
        evidence["rejection_reason"] = "OLD_DIRECT_EDGE_REJECTED"
        return evidence
    if not np.isfinite(evidence["d_history_m"]):
        evidence["rejection_reason"] = "NONFINITE_HISTORY_DIAMETER"
        return evidence
    if evidence["d_history_m"] > float(local_diameter_m):
        evidence["rejection_reason"] = "LOCAL_DIAMETER_EXCEEDED"
        return evidence
    evidence["merge_allowed"] = True
    evidence["rejection_reason"] = None
    return evidence


def _component_pair_evidence(
    xyz: np.ndarray,
    valid: np.ndarray,
    history_indices: np.ndarray,
    members: Sequence[int],
    config: ComponentConfig,
    local_diameter_m: float,
) -> dict[tuple[int, int], dict[str, Any]]:
    """Compute all pair predicates of one component with vectorized NumPy."""

    slots = np.asarray(tuple(int(item) for item in members), dtype=np.int64)
    points = np.asarray(xyz, dtype=np.float64)[history_indices][:, slots]
    observed = np.asarray(valid, dtype=bool)[history_indices][:, slots]
    observed &= np.all(np.isfinite(points), axis=-1)
    pair_observed = observed[:, :, None] & observed[:, None, :]
    relative = points[:, None, :, :] - points[:, :, None, :]
    distances = np.linalg.norm(relative, axis=-1)
    overlap = np.sum(pair_observed, axis=0)
    first_index = np.argmax(pair_observed, axis=0)
    local_rows = np.arange(slots.size)[:, None]
    local_cols = np.arange(slots.size)[None, :]
    initial = distances[first_index, local_rows, local_cols]
    masked_relative = np.where(pair_observed[..., None], relative, np.nan)
    with warnings.catch_warnings(), np.errstate(all="ignore"):
        warnings.simplefilter("ignore", RuntimeWarning)
        median_relative = np.nanmedian(masked_relative, axis=0)
        change = np.linalg.norm(relative - median_relative[None, ...], axis=-1)
        median_change = np.nanmedian(np.where(pair_observed, change, np.nan), axis=0)
    result: dict[tuple[int, int], dict[str, Any]] = {}
    for left_index in range(slots.size):
        for right_index in range(left_index + 1, slots.size):
            left = int(slots[left_index])
            right = int(slots[right_index])
            count = int(overlap[left_index, right_index])
            d_value = distances[:, left_index, right_index]
            d_history = float(np.nanmax(np.where(pair_observed[:, left_index, right_index], d_value, np.nan))) if count else None
            initial_value = float(initial[left_index, right_index]) if count else None
            change_value = float(median_change[left_index, right_index]) if count else None
            direct = bool(
                count >= int(config.minimum_overlap)
                and initial_value is not None
                and change_value is not None
                and initial_value <= float(config.max_initial_distance)
                and change_value <= float(config.max_relative_change)
            )
            evidence: dict[str, Any] = {
                "left_slot": left,
                "right_slot": right,
                "history_overlap_count": count,
                "direct_edge": direct,
                "d_history_m": d_history,
                "initial_distance_m": initial_value,
                "relative_change_median_m": change_value,
                "merge_allowed": False,
                "rejection_reason": "INSUFFICIENT_HISTORY_OVERLAP" if count < int(config.minimum_overlap) else "OLD_DIRECT_EDGE_REJECTED",
            }
            if direct and d_history is not None and np.isfinite(d_history):
                if d_history <= float(local_diameter_m):
                    evidence["merge_allowed"] = True
                    evidence["rejection_reason"] = None
                else:
                    evidence["rejection_reason"] = "LOCAL_DIAMETER_EXCEEDED"
            elif direct and (d_history is None or not np.isfinite(d_history)):
                evidence["rejection_reason"] = "NONFINITE_HISTORY_DIAMETER"
            result[_pair_key(left, right)] = evidence
    return result


def _canonical_components(
    components: Sequence[Sequence[int]], track_ids: np.ndarray
) -> list[tuple[int, ...]]:
    values = [tuple(sorted((int(item) for item in component), key=lambda item: (int(track_ids[item]), item))) for component in components]
    return sorted(values, key=lambda component: _track_key(track_ids, component))


def _merge_distance(
    left: Sequence[int],
    right: Sequence[int],
    pair_map: Mapping[tuple[int, int], Mapping[str, Any]],
) -> float | None:
    distances: list[float] = []
    for left_slot in left:
        for right_slot in right:
            evidence = pair_map[_pair_key(int(left_slot), int(right_slot))]
            if not bool(evidence["merge_allowed"]) or evidence["d_history_m"] is None:
                return None
            distances.append(float(evidence["d_history_m"]))
    return max(distances) if distances else None


def build_local_groups(
    sequence: Any,
    history_indices: Sequence[int],
    *,
    component_config: ComponentConfig = COMPONENT_CONFIG,
    local_diameter_m: float = LOCAL_DIAMETER_M,
    old_components: Sequence[Sequence[int]] | None = None,
) -> dict[str, Any]:
    """Partition each existing history component by constrained complete linkage.

    The returned group identities are canonicalized by track ID.  Only groups
    meeting the existing minimum component size are marked retained; discarded
    small groups remain explicit in the result.
    """

    if not np.isfinite(local_diameter_m) or local_diameter_m <= 0:
        raise ValueError("local_diameter_m must be finite and positive")
    xyz = np.asarray(sequence.xyz, dtype=np.float64)
    valid = np.asarray(sequence.geometry_validity, dtype=bool)
    track_ids = np.asarray(sequence.track_ids, dtype=np.int64)
    history = np.asarray(history_indices, dtype=np.int64)
    if xyz.ndim != 3 or xyz.shape[-1] != 3 or valid.shape != xyz.shape[:2]:
        raise ValueError("sequence xyz and geometry_validity have incompatible shapes")
    if track_ids.shape != (xyz.shape[1],) or len(np.unique(track_ids)) != track_ids.size:
        raise ValueError("track_ids must be unique and match sequence slots")
    if history.ndim != 1 or np.any(history < 0) or np.any(history >= xyz.shape[0]):
        raise ValueError("history_indices are out of range")

    history_xyz = xyz[history]
    history_valid = valid[history]
    components = old_components if old_components is not None else motion_coherent_components(history_xyz, history_valid, component_config)
    canonical = _canonical_components(components, track_ids)
    parent_rows: list[dict[str, Any]] = []
    all_group_rows: list[dict[str, Any]] = []
    for parent_id, component in enumerate(canonical):
        members = tuple(component)
        pair_map = _component_pair_evidence(xyz, valid, history, members, component_config, float(local_diameter_m))

        active: dict[int, tuple[int, ...]] = {index: (int(member),) for index, member in enumerate(members)}
        heap: list[tuple[float, tuple[int, ...], tuple[int, ...], tuple[int, ...], int, int]] = []

        def push_candidate(left_id: int, right_id: int) -> None:
            left_group = active[left_id]
            right_group = active[right_id]
            left_key = _track_key(track_ids, left_group)
            right_key = _track_key(track_ids, right_group)
            if right_key < left_key:
                left_id, right_id = right_id, left_id
                left_group, right_group = right_group, left_group
                left_key, right_key = right_key, left_key
            distance = _merge_distance(left_group, right_group, pair_map)
            if distance is not None and distance <= float(local_diameter_m):
                heapq.heappush(heap, (float(distance), tuple(sorted(left_key + right_key)), left_key, right_key, left_id, right_id))

        active_ids = sorted(active)
        for left_index, left_id in enumerate(active_ids):
            for right_id in active_ids[left_index + 1 :]:
                push_candidate(left_id, right_id)
        next_group_id = len(active)
        while heap:
            _distance, _combined_key, left_key, right_key, left_id, right_id = heapq.heappop(heap)
            if left_id not in active or right_id not in active:
                continue
            if _track_key(track_ids, active[left_id]) != left_key or _track_key(track_ids, active[right_id]) != right_key:
                continue
            merged = tuple(sorted(active[left_id] + active[right_id], key=lambda item: (int(track_ids[item]), item)))
            del active[left_id]
            del active[right_id]
            merged_id = next_group_id
            next_group_id += 1
            active[merged_id] = merged
            for other_id in sorted(active):
                if other_id != merged_id:
                    push_candidate(merged_id, other_id)
        groups = sorted(active.values(), key=lambda group: _track_key(track_ids, group))

        parent_rows.append(
            {
                "parent_component_id": int(parent_id),
                "track_ids": [int(track_ids[item]) for item in members],
                "member_slots": [int(item) for item in members],
                "pair_evidence": [pair_map[key] for key in sorted(pair_map)],
                "history_frame_count": int(history.size),
            }
        )
        for group in sorted(groups, key=lambda value: _track_key(track_ids, value)):
            all_group_rows.append(
                {
                    "parent_component_id": int(parent_id),
                    "member_slots": [int(item) for item in group],
                    "track_ids": [int(track_ids[item]) for item in group],
                    "track_key": list(_track_key(track_ids, group)),
                    "retained": len(group) >= int(component_config.minimum_size),
                    "status": "RETAINED" if len(group) >= int(component_config.minimum_size) else "SUPPORT_INSUFFICIENT_GROUP_SIZE",
                }
            )

    all_group_rows.sort(key=lambda row: (int(row["parent_component_id"]), tuple(row["track_key"])))
    for local_id, row in enumerate(all_group_rows):
        row["local_group_id"] = int(local_id)
    retained = [row for row in all_group_rows if bool(row["retained"])]
    seen: set[int] = set()
    for row in retained:
        members = tuple(int(item) for item in row["member_slots"])
        if seen.intersection(members):
            raise AssertionError("retained local groups overlap")
        seen.update(members)

    return {
        "local_diameter_m": float(local_diameter_m),
        "component_config": {
            "max_initial_distance": float(component_config.max_initial_distance),
            "max_relative_change": float(component_config.max_relative_change),
            "minimum_size": int(component_config.minimum_size),
            "minimum_overlap": int(component_config.minimum_overlap),
        },
        "history_array_indices": history.astype(int).tolist(),
        "parent_components": parent_rows,
        "groups": all_group_rows,
        "retained_group_count": len(retained),
        "support_insufficient_group_count": len(all_group_rows) - len(retained),
        "retained_track_ids": sorted(int(item) for row in retained for item in row["track_ids"]),
    }


def _local_support_pair(
    xyz: np.ndarray,
    valid: np.ndarray,
    frame_tuple: tuple[int, int, int],
    members: Sequence[int],
    track_ids: np.ndarray,
    scale: float,
    *,
    parent_component_id: int,
    local_group_id: int,
    target_rows: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any] | None, str | None]:
    member_array = np.asarray(members, dtype=np.int64)
    geometry = valid[np.asarray(frame_tuple), :][:, member_array]
    finite_xyz = np.all(np.isfinite(xyz[np.asarray(frame_tuple)][:, member_array, :]), axis=(0, 2))
    common = [member for position, member in enumerate(members) if bool(np.all(geometry[:, position])) and bool(finite_xyz[position])]
    if len(common) < 3:
        return None, "COMMON_VALID_MEMBERS_LT3"
    states, pair_indices, reason = _state_from_common_members(xyz, frame_tuple, common, float(scale))
    if states is None or pair_indices is None or reason is not None:
        return None, reason or "STATE_INVALID"
    slots = tuple(int(item["slot"]) for item in target_rows)
    triplet = {
        "parent_component_id": int(parent_component_id),
        "component_index": int(local_group_id),
        "local_group_id": int(local_group_id),
        "target_slots": list(slots),
        "target_times_s": [float(item["target_time_s"]) for item in target_rows],
        "frame_indices": [int(item["frame_index"]) for item in target_rows],
        "timestamps_s": [float(item["timestamp_s"]) for item in target_rows],
        "match_errors_s": [float(item["match_error_s"]) for item in target_rows],
        "history_scale": float(scale),
        "members_considered": [int(item) for item in members],
        "track_ids_considered": [int(track_ids[item]) for item in members],
        "common_member_indices": [int(item) for item in common],
        "common_track_ids": [int(track_ids[item]) for item in common],
        "pair_indices": [[int(left), int(right)] for left, right in pair_indices],
        "pair_ids": [[int(track_ids[left]), int(track_ids[right])] for left, right in pair_indices],
        "states": states,
        "status": "VALID",
    }
    return triplet, None


def build_local_support(
    sequence: Any,
    *,
    window_start_s: float,
    grouping: Mapping[str, Any],
    max_target_error_s: float = MAX_TARGET_ERROR_S,
) -> dict[str, Any]:
    """Build frozen-identity local triplets with the existing state formulas."""

    xyz = np.asarray(sequence.xyz, dtype=np.float64)
    valid = np.asarray(sequence.geometry_validity, dtype=bool)
    timestamps = np.asarray(sequence.timestamps_s, dtype=np.float64)
    frame_indices = np.asarray(sequence.frame_indices, dtype=np.int64)
    track_ids = np.asarray(sequence.track_ids, dtype=np.int64)
    if xyz.ndim != 3 or xyz.shape[-1] != 3 or valid.shape != xyz.shape[:2] or timestamps.shape != (xyz.shape[0],):
        raise ValueError("sequence arrays have incompatible shapes")
    history_indices = np.flatnonzero(timestamps < float(window_start_s) + 0.5).astype(np.int64)
    evaluation_indices = np.flatnonzero(timestamps >= float(window_start_s) + 0.5).astype(np.int64)
    matches = _target_frame_matches(timestamps, frame_indices, float(window_start_s), max_error_s=max_target_error_s)
    triplets: list[dict[str, Any]] = []
    group_rows: list[dict[str, Any]] = []
    invalid_reasons: list[dict[str, Any]] = []
    retained_groups = [row for row in grouping.get("groups", []) if bool(row.get("retained"))]
    for group in retained_groups:
        members = tuple(int(item) for item in group["member_slots"])
        scale, count, _ = _history_scale(xyz, valid, history_indices, members)
        row = {
            "local_group_id": int(group["local_group_id"]),
            "parent_component_id": int(group["parent_component_id"]),
            "member_slots": [int(item) for item in members],
            "track_ids": [int(track_ids[item]) for item in members],
            "history_scale": scale,
            "history_scale_pair_time_count": int(count),
            "triplet_ids": [],
            "invalid_triplets": [],
        }
        if scale is None:
            row["invalid_triplets"] = [{"target_slots": [0, 1, 2], "reason": "HISTORY_SCALE_INVALID"}]
            invalid_reasons.append({"local_group_id": row["local_group_id"], "reason": "HISTORY_SCALE_INVALID"})
            group_rows.append(row)
            continue
        for start in range(3):
            slots = (start, start + 1, start + 2)
            target_rows = [matches[item] for item in slots]
            if any(item["frame_index"] is None for item in target_rows):
                reason = "TARGET_FRAME_MISSING"
                row["invalid_triplets"].append({"target_slots": list(slots), "reason": reason})
                invalid_reasons.append({"local_group_id": row["local_group_id"], "target_slots": list(slots), "reason": reason})
                continue
            frame_tuple = tuple(int(item["array_index"]) for item in target_rows)
            triplet, reason = _local_support_pair(
                xyz,
                valid,
                frame_tuple,
                members,
                track_ids,
                float(scale),
                parent_component_id=int(group["parent_component_id"]),
                local_group_id=int(group["local_group_id"]),
                target_rows=target_rows,
            )
            if triplet is None:
                row["invalid_triplets"].append({"target_slots": list(slots), "reason": reason})
                invalid_reasons.append({"local_group_id": row["local_group_id"], "target_slots": list(slots), "reason": reason})
                continue
            triplet["triplet_id"] = int(len(triplets))
            triplets.append(triplet)
            row["triplet_ids"].append(int(triplet["triplet_id"]))
        group_rows.append(row)
    return {
        "history_boundary_s": float(window_start_s) + 0.5,
        "history_frame_indices": frame_indices[history_indices].astype(int).tolist(),
        "history_array_indices": history_indices.astype(int).tolist(),
        "evaluation_frame_indices": frame_indices[evaluation_indices].astype(int).tolist(),
        "evaluation_array_indices": evaluation_indices.astype(int).tolist(),
        "target_matches": matches,
        "components": group_rows,
        "triplets": triplets,
        "valid_triplet_count": len(triplets),
        "invalid_reasons": invalid_reasons,
        "support_status": "VALID" if triplets else "NO_VALID_TRIPLET",
        "grouping_local_diameter_m": float(grouping["local_diameter_m"]),
    }


def build_support_from_components(
    sequence: Any,
    *,
    window_start_s: float,
    components: Sequence[Sequence[int]],
    max_target_error_s: float = MAX_TARGET_ERROR_S,
) -> dict[str, Any]:
    """Build the existing support contract from already verified components."""

    xyz = np.asarray(sequence.xyz, dtype=np.float64)
    valid = np.asarray(sequence.geometry_validity, dtype=bool)
    timestamps = np.asarray(sequence.timestamps_s, dtype=np.float64)
    frame_indices = np.asarray(sequence.frame_indices, dtype=np.int64)
    track_ids = np.asarray(sequence.track_ids, dtype=np.int64)
    history_indices = np.flatnonzero(timestamps < float(window_start_s) + 0.5).astype(np.int64)
    evaluation_indices = np.flatnonzero(timestamps >= float(window_start_s) + 0.5).astype(np.int64)
    matches = _target_frame_matches(timestamps, frame_indices, float(window_start_s), max_error_s=max_target_error_s)
    triplets: list[dict[str, Any]] = []
    component_rows: list[dict[str, Any]] = []
    invalid_reasons: list[dict[str, Any]] = []
    for component_index, component in enumerate(components):
        members = tuple(int(item) for item in component)
        scale, scale_count, _ = _history_scale(xyz, valid, history_indices, members)
        row: dict[str, Any] = {
            "component_index": int(component_index),
            "member_indices": [int(item) for item in members],
            "track_ids": [int(track_ids[item]) for item in members],
            "pair_ids": [[int(track_ids[left]), int(track_ids[right])] for left, right in combinations(members, 2)],
            "history_frame_count": int(history_indices.size),
            "history_scale": scale,
            "history_scale_pair_time_count": int(scale_count),
            "validity_source": "ParticleSequence.geometry_validity and finite XYZ",
            "triplet_ids": [],
            "invalid_triplets": [],
        }
        if scale is None:
            reason = "HISTORY_SCALE_INVALID"
            row["invalid_triplets"] = [{"target_slots": [int(start), int(start + 1), int(start + 2)], "reason": reason} for start in range(3)]
            invalid_reasons.append({"component_index": int(component_index), "reason": reason})
            component_rows.append(row)
            continue
        for start in range(3):
            slots = (start, start + 1, start + 2)
            target_rows = [matches[item] for item in slots]
            base = {
                "component_index": int(component_index),
                "target_slots": list(slots),
                "target_times_s": [float(item["target_time_s"]) for item in target_rows],
                "frame_indices": [item["frame_index"] for item in target_rows],
                "timestamps_s": [item["timestamp_s"] for item in target_rows],
                "match_errors_s": [item["match_error_s"] for item in target_rows],
                "history_scale": float(scale),
                "history_scale_pair_time_count": int(scale_count),
                "members_considered": [int(item) for item in members],
                "track_ids_considered": [int(track_ids[item]) for item in members],
                "validity_source": "ParticleSequence.geometry_validity and finite XYZ at all three matched frames",
            }
            if any(item["frame_index"] is None for item in target_rows):
                reason = "TARGET_FRAME_MISSING"
                row["invalid_triplets"].append({"target_slots": list(slots), "reason": reason})
                invalid_reasons.append({"component_index": int(component_index), "target_slots": list(slots), "reason": reason})
                continue
            frame_tuple = tuple(int(item["array_index"]) for item in target_rows)
            member_array = np.asarray(members, dtype=np.int64)
            geometry = valid[np.asarray(frame_tuple), :][:, member_array]
            finite_xyz = np.all(np.isfinite(xyz[np.asarray(frame_tuple)][:, member_array, :]), axis=(0, 2))
            common = [member for position, member in enumerate(members) if bool(np.all(geometry[:, position])) and bool(finite_xyz[position])]
            if len(common) < 3:
                reason = "COMMON_VALID_MEMBERS_LT3"
                row["invalid_triplets"].append({"target_slots": list(slots), "reason": reason, "common_track_ids": [int(track_ids[item]) for item in common]})
                invalid_reasons.append({"component_index": int(component_index), "target_slots": list(slots), "reason": reason})
                continue
            states, pair_indices, reason = _state_from_common_members(xyz, frame_tuple, common, float(scale))
            if states is None or pair_indices is None or reason is not None:
                error_reason = reason or "STATE_INVALID"
                row["invalid_triplets"].append({"target_slots": list(slots), "reason": error_reason})
                invalid_reasons.append({"component_index": int(component_index), "target_slots": list(slots), "reason": error_reason})
                continue
            triplet = {
                **base,
                "triplet_id": int(len(triplets)),
                "common_member_indices": [int(item) for item in common],
                "common_track_ids": [int(track_ids[item]) for item in common],
                "pair_indices": [[int(left), int(right)] for left, right in pair_indices],
                "pair_ids": [[int(track_ids[left]), int(track_ids[right])] for left, right in pair_indices],
                "states": states,
                "status": "VALID",
            }
            triplets.append(triplet)
            row["triplet_ids"].append(int(triplet["triplet_id"]))
        component_rows.append(row)
    return {
        "history_boundary_s": float(window_start_s) + 0.5,
        "history_frame_indices": frame_indices[history_indices].astype(int).tolist(),
        "history_array_indices": history_indices.astype(int).tolist(),
        "evaluation_frame_indices": frame_indices[evaluation_indices].astype(int).tolist(),
        "evaluation_array_indices": evaluation_indices.astype(int).tolist(),
        "target_matches": matches,
        "component_count": len(component_rows),
        "components": component_rows,
        "triplets": triplets,
        "valid_triplet_count": len(triplets),
        "invalid_reasons": invalid_reasons,
        "support_status": "VALID" if triplets else "NO_VALID_TRIPLET",
    }


def json_ready_local_support(support: Mapping[str, Any]) -> dict[str, Any]:
    """Convert local support arrays to JSON without changing values."""

    output = dict(support)
    output["triplets"] = [
        {
            **{key: value for key, value in triplet.items() if key != "states"},
            "states": np.asarray(triplet["states"], dtype=np.float64).tolist(),
        }
        for triplet in support.get("triplets", [])
    ]
    return output


def validate_grouping(grouping: Mapping[str, Any], track_ids: Sequence[int], minimum_size: int = 3) -> dict[str, Any]:
    """Validate the disjointness and complete-linkage constraints of a result."""

    ids = np.asarray(track_ids, dtype=np.int64)
    retained = [row for row in grouping.get("groups", []) if bool(row.get("retained"))]
    seen: set[int] = set()
    checks = {"disjoint": True, "minimum_size": True, "pair_constraints": True, "history_support": True}
    for row in retained:
        members = [int(item) for item in row["member_slots"]]
        if len(members) < minimum_size or seen.intersection(members):
            checks["disjoint"] = False
            checks["minimum_size"] = False
        seen.update(members)
        for item in grouping.get("parent_components", []):
            if int(item["parent_component_id"]) != int(row["parent_component_id"]):
                continue
            evidence = {(int(x["left_slot"]), int(x["right_slot"])): x for x in item.get("pair_evidence", [])}
            for left, right in combinations(members, 2):
                pair = evidence.get(_pair_key(left, right))
                if pair is None or not pair.get("direct_edge") or not pair.get("merge_allowed") or pair.get("d_history_m") is None or float(pair["d_history_m"]) > float(grouping["local_diameter_m"]):
                    checks["pair_constraints"] = False
                if pair is None or int(pair.get("history_overlap_count", 0)) < int(grouping["component_config"]["minimum_overlap"]):
                    checks["history_support"] = False
    checks["all_pass"] = bool(all(checks.values()))
    return checks
