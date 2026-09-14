"""Frozen five-time support and input construction for the V7 pilot.

The only observations consumed here are the existing 289-query particle
arrays and the already frozen history-local H groups.  A valid unit keeps one
set of members, pairs and one history-derived scale for all five target
times.  Invalid support is returned as a reason rather than being padded with
zeros.
"""

from __future__ import annotations

import hashlib
import itertools
from typing import Any, Mapping, Sequence

import numpy as np

from research_tools.v7.local_structural_temporal_probe.representation import (
    MAX_TARGET_ERROR_S,
    TARGET_OFFSETS_S,
    _history_scale,
    _target_frame_matches,
)


CONDITIONS = ("SET_A", "RAW_SEQ", "MULTI_ORDER_SEQ", "SHUFFLED_MULTI_ORDER")
TARGET_OFFSETS_S = tuple(float(x) for x in TARGET_OFFSETS_S)
MIN_COMMON_MEMBERS = 3


def _finite_xyz(xyz: np.ndarray, valid: np.ndarray, frames: Sequence[int], members: Sequence[int]) -> np.ndarray:
    frame_idx = np.asarray(frames, dtype=np.int64)
    member_idx = np.asarray(members, dtype=np.int64)
    return valid[frame_idx][:, member_idx] & np.all(np.isfinite(xyz[frame_idx][:, member_idx]), axis=-1)


def deterministic_permutation(window_id: str) -> tuple[int, ...]:
    """Return a stable, label-blind non-identity permutation for one window."""

    digest = hashlib.sha256(f"v7-multi-order-sequence:{window_id}".encode("utf-8")).digest()
    seed = int.from_bytes(digest[:8], "little", signed=False)
    permutation = np.random.default_rng(seed).permutation(5).astype(np.int64)
    if np.array_equal(permutation, np.arange(5, dtype=np.int64)):
        permutation[:2] = permutation[1], permutation[0]
    return tuple(int(x) for x in permutation)


def compute_derivatives(states: Sequence[Sequence[float]], timestamps_s: Sequence[float]) -> tuple[np.ndarray, np.ndarray]:
    """Compute timestamp-aware v[4,4] and a[3,4] from five states."""

    values = np.asarray(states, dtype=np.float64)
    times = np.asarray(timestamps_s, dtype=np.float64)
    if values.shape != (5, 4) or times.shape != (5,):
        raise ValueError("five states and timestamps are required")
    if not np.all(np.isfinite(values)) or not np.all(np.isfinite(times)) or not np.all(np.diff(times) > 0):
        raise ValueError("states must be finite and timestamps strictly increasing")
    intervals = np.diff(times)
    velocity = np.diff(values, axis=0) / intervals[:, None]
    acceleration = 2.0 * np.diff(velocity, axis=0) / (intervals[1:, None] + intervals[:-1, None])
    return velocity, acceleration


def _state_from_pairs(
    xyz: np.ndarray,
    frame_indices: Sequence[int],
    members: Sequence[int],
    scale: float,
) -> tuple[np.ndarray | None, list[list[int]] | None, str | None]:
    pairs = list(itertools.combinations(tuple(int(x) for x in members), 2))
    if not pairs:
        return None, None, "NO_PAIR_SUPPORT"
    distances = []
    frame_idx = np.asarray(frame_indices, dtype=np.int64)
    for left, right in pairs:
        value = np.linalg.norm(xyz[frame_idx, right] - xyz[frame_idx, left], axis=1)
        if not np.all(np.isfinite(value)):
            return None, None, "NONFINITE_PAIR_DISTANCE"
        distances.append(value / float(scale))
    matrix = np.asarray(distances, dtype=np.float64).T
    if matrix.shape != (5, len(pairs)) or not np.all(np.isfinite(matrix)):
        return None, None, "NONFINITE_NORMALIZED_DISTANCE"
    states = np.stack(
        [
            np.mean(matrix, axis=1),
            np.std(matrix, axis=1),
            np.percentile(matrix, 25, axis=1),
            np.percentile(matrix, 75, axis=1),
        ],
        axis=1,
    )
    return states, [[int(a), int(b)] for a, b in pairs], None


def build_five_time_unit(
    sequence: Any,
    *,
    window_id: str,
    window_start_s: float,
    member_slots: Sequence[int],
    local_group_id: int,
    max_target_error_s: float = MAX_TARGET_ERROR_S,
) -> dict[str, Any]:
    """Build one H-group unit, preserving explicit missing reasons."""

    xyz = np.asarray(sequence.xyz, dtype=np.float64)
    valid = np.asarray(sequence.geometry_validity, dtype=bool)
    timestamps = np.asarray(sequence.timestamps_s, dtype=np.float64)
    source_frames = np.asarray(sequence.frame_indices, dtype=np.int64)
    track_ids = np.asarray(sequence.track_ids, dtype=np.int64)
    members = tuple(sorted(set(int(x) for x in member_slots)))
    if xyz.ndim != 3 or xyz.shape[-1] != 3 or valid.shape != xyz.shape[:2]:
        raise ValueError("sequence xyz/geometry_validity shapes are incompatible")
    if len(members) < MIN_COMMON_MEMBERS:
        return {"status": "INVALID", "reason": "H_GROUP_LT3", "local_group_id": int(local_group_id)}
    if not np.all(np.isfinite(timestamps)) or (timestamps.size > 1 and not np.all(np.diff(timestamps) > 0)):
        return {"status": "INVALID", "reason": "SEQUENCE_TIMESTAMPS_NOT_STRICT", "local_group_id": int(local_group_id)}

    boundary = float(window_start_s) + 0.5
    history_indices = np.flatnonzero(timestamps < boundary).astype(np.int64)
    matches = _target_frame_matches(timestamps, source_frames, float(window_start_s), max_error_s=max_target_error_s)
    if len(matches) != 5 or any(item.get("frame_index") is None for item in matches):
        return {
            "status": "INVALID",
            "reason": "TARGET_FRAME_MISSING",
            "local_group_id": int(local_group_id),
            "target_matches": matches,
        }
    array_indices = tuple(int(item["array_index"]) for item in matches)
    actual_timestamps = np.asarray([float(item["timestamp_s"]) for item in matches], dtype=np.float64)
    if len(set(array_indices)) != 5 or not np.all(np.diff(actual_timestamps) > 0):
        return {"status": "INVALID", "reason": "TARGET_FRAMES_NOT_STRICT", "local_group_id": int(local_group_id), "target_matches": matches}

    all_valid = _finite_xyz(xyz, valid, array_indices, members)
    common = [member for column, member in enumerate(members) if bool(np.all(all_valid[:, column]))]
    if len(common) < MIN_COMMON_MEMBERS:
        return {
            "status": "INVALID",
            "reason": "COMMON_VALID_MEMBERS_LT3",
            "local_group_id": int(local_group_id),
            "target_matches": matches,
            "common_track_ids": [int(track_ids[item]) for item in common],
        }
    scale, scale_count, _ = _history_scale(xyz, valid, history_indices, common)
    if scale is None:
        return {"status": "INVALID", "reason": "HISTORY_SCALE_INVALID", "local_group_id": int(local_group_id), "target_matches": matches}
    states, pair_indices, reason = _state_from_pairs(xyz, array_indices, common, float(scale))
    if states is None or pair_indices is None:
        return {"status": "INVALID", "reason": reason or "STATE_INVALID", "local_group_id": int(local_group_id), "target_matches": matches}
    return {
        "status": "VALID",
        "window_id": str(window_id),
        "local_group_id": int(local_group_id),
        "member_slots": [int(x) for x in common],
        "track_ids": [int(track_ids[x]) for x in common],
        "pair_indices": pair_indices,
        "pair_ids": [[int(track_ids[a]), int(track_ids[b])] for a, b in pair_indices],
        "history_scale": float(scale),
        "history_scale_pair_time_count": int(scale_count),
        "frame_indices": [int(item["frame_index"]) for item in matches],
        "array_indices": [int(x) for x in array_indices],
        "timestamps_s": [float(x) for x in actual_timestamps],
        "target_times_s": [float(item["target_time_s"]) for item in matches],
        "match_errors_s": [float(item["match_error_s"]) for item in matches],
        "states": states,
        "support_semantics": "same five-time common members, pair identities and history scale",
    }


def condition_inputs(states: np.ndarray, timestamps_s: Sequence[float], window_id: str) -> dict[str, np.ndarray | tuple[int, ...]]:
    """Construct all four fixed condition inputs from one five-state unit."""

    values = np.asarray(states, dtype=np.float64)
    times = np.asarray(timestamps_s, dtype=np.float64)
    if values.shape != (5, 4):
        raise ValueError("states must have shape [5,4]")
    velocity, acceleration = compute_derivatives(values, times)
    intervals = np.diff(times).astype(np.float64)
    raw = np.concatenate((values[:3], values[1:4], values[2:5]), axis=0).reshape(-1)
    multi = np.concatenate([np.concatenate((values[i], velocity[i - 1], acceleration[i - 2])) for i in (2, 3, 4)])
    permutation = deterministic_permutation(window_id)
    shuffled_values = values[np.asarray(permutation, dtype=np.int64)]
    shuffled_v, shuffled_a = compute_derivatives(shuffled_values, times)
    shuffled = np.concatenate([np.concatenate((shuffled_values[i], shuffled_v[i - 1], shuffled_a[i - 2])) for i in (2, 3, 4)])
    return {
        "SET_A_states": values.copy(),
        "RAW_SEQ": np.concatenate((raw, intervals)),
        "MULTI_ORDER_SEQ": np.concatenate((multi, intervals)),
        "SHUFFLED_MULTI_ORDER": np.concatenate((shuffled, intervals)),
        "permutation": permutation,
        "intervals_s": intervals,
    }


def condition_feature_matrix(support_row: Mapping[str, Any], window_id: str) -> dict[str, Any]:
    """Build feature matrices for every valid H unit in one window."""

    units = [item for item in support_row.get("units", []) if item.get("status") == "VALID"]
    outputs: dict[str, list[np.ndarray]] = {condition: [] for condition in CONDITIONS}
    permutations: list[tuple[int, ...]] = []
    for unit in units:
        values = np.asarray(unit["states"], dtype=np.float64)
        inputs = condition_inputs(values, unit["timestamps_s"], window_id)
        outputs["SET_A"].append(np.asarray(inputs["SET_A_states"], dtype=np.float64))
        for condition in ("RAW_SEQ", "MULTI_ORDER_SEQ", "SHUFFLED_MULTI_ORDER"):
            outputs[condition].append(np.asarray(inputs[condition], dtype=np.float64))
        permutations.append(tuple(int(x) for x in inputs["permutation"]))
    if not units:
        return {"unit_count": 0, "permutations": [], "features": {condition: None for condition in CONDITIONS}}
    return {
        "unit_count": len(units),
        "permutations": permutations,
        "features": {
            "SET_A": np.stack(outputs["SET_A"], axis=0),
            "RAW_SEQ": np.stack(outputs["RAW_SEQ"], axis=0),
            "MULTI_ORDER_SEQ": np.stack(outputs["MULTI_ORDER_SEQ"], axis=0),
            "SHUFFLED_MULTI_ORDER": np.stack(outputs["SHUFFLED_MULTI_ORDER"], axis=0),
        },
    }


def json_ready(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, tuple):
        return [json_ready(item) for item in value]
    if isinstance(value, list):
        return [json_ready(item) for item in value]
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    return value
