"""Representation and support construction for the bounded V7 pilot.

The adapter consumes already materialized ParticleSequence artifacts.  It
does not read video or rerun any frontend.  Components are built from the
history half of each frozen one-second window, while each local triplet keeps
one common set of track identities and one history-derived scale.
"""

from __future__ import annotations

from itertools import combinations, permutations
from typing import Any, Mapping, Sequence

import numpy as np

from sparse3d_forgery.experiments.v7_dynamic_structure_probe import (
    ComponentConfig,
    motion_coherent_components,
)


COMPONENT_CONFIG = ComponentConfig(
    max_initial_distance=1.0,
    max_relative_change=0.05,
    minimum_size=3,
    minimum_overlap=8,
)
TARGET_OFFSETS_S = (0.5, 0.6, 0.7, 0.8, 0.9)
MAX_TARGET_ERROR_S = 0.05
ARM_NAMES = ("UNORDERED_STATE", "ORDERED_FIRST", "ORDERED_SECOND", "PERMUTED_SECOND")


def _as_float_array(values: Any, *, ndim: int | None = None) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if ndim is not None and array.ndim != ndim:
        raise ValueError(f"expected {ndim}-D numeric array")
    return array


def _pair_ids(track_ids: np.ndarray, members: Sequence[int]) -> list[list[int]]:
    return [
        [int(track_ids[left]), int(track_ids[right])]
        for left, right in combinations(tuple(int(item) for item in members), 2)
    ]


def _history_scale(
    xyz: np.ndarray,
    valid: np.ndarray,
    history_indices: np.ndarray,
    members: Sequence[int],
) -> tuple[float | None, int, list[float]]:
    distances: list[float] = []
    for left, right in combinations(tuple(int(item) for item in members), 2):
        pair_valid = valid[history_indices, left] & valid[history_indices, right]
        if not np.any(pair_valid):
            continue
        values = np.linalg.norm(
            xyz[history_indices[pair_valid], right] - xyz[history_indices[pair_valid], left],
            axis=1,
        )
        distances.extend(float(value) for value in values if np.isfinite(value))
    if not distances:
        return None, 0, []
    scale = float(np.median(np.asarray(distances, dtype=np.float64)))
    if not np.isfinite(scale) or scale <= 0:
        return None, len(distances), distances
    return scale, len(distances), distances


def _state_from_common_members(
    xyz: np.ndarray,
    frame_indices: Sequence[int],
    members: Sequence[int],
    scale: float,
) -> tuple[np.ndarray | None, list[list[int]] | None, str | None]:
    pair_members = list(combinations(tuple(int(item) for item in members), 2))
    if len(pair_members) == 0:
        return None, None, "NO_PAIR_SUPPORT"
    distances: list[np.ndarray] = []
    for left, right in pair_members:
        values = np.linalg.norm(xyz[np.asarray(frame_indices), right] - xyz[np.asarray(frame_indices), left], axis=1)
        if not np.all(np.isfinite(values)):
            return None, None, "NONFINITE_PAIR_DISTANCE"
        distances.append(values / scale)
    normalized = np.asarray(distances, dtype=np.float64).T
    if normalized.shape != (len(frame_indices), len(pair_members)) or not np.all(np.isfinite(normalized)):
        return None, None, "NONFINITE_NORMALIZED_DISTANCE"
    state = np.asarray(
        [
            [
                np.mean(row),
                np.std(row),
                np.percentile(row, 25),
                np.percentile(row, 75),
            ]
            for row in normalized
        ],
        dtype=np.float64,
    )
    return state, [[int(left), int(right)] for left, right in pair_members], None


def _target_frame_matches(
    timestamps_s: np.ndarray,
    source_frame_indices: np.ndarray,
    window_start_s: float,
    *,
    max_error_s: float = MAX_TARGET_ERROR_S,
) -> list[dict[str, Any]]:
    boundary = float(window_start_s) + 0.5
    local_indices = np.arange(timestamps_s.size, dtype=np.int64)
    evaluation = [int(index) for index in local_indices if timestamps_s[index] >= boundary]
    used: set[int] = set()
    matches: list[dict[str, Any]] = []
    for slot, offset in enumerate(TARGET_OFFSETS_S):
        target = float(window_start_s) + float(offset)
        candidates = [
            (abs(float(timestamps_s[index]) - target), index)
            for index in evaluation
            if index not in used and abs(float(timestamps_s[index]) - target) <= max_error_s
        ]
        if not candidates:
            matches.append(
                {
                    "slot": slot,
                    "target_time_s": target,
                    "frame_index": None,
                    "array_index": None,
                    "timestamp_s": None,
                    "match_error_s": None,
                    "status": "MISSING_TARGET_FRAME",
                }
            )
            continue
        error, index = min(candidates, key=lambda item: (item[0], item[1]))
        used.add(index)
        matches.append(
                {
                    "slot": slot,
                    "target_time_s": target,
                    "frame_index": int(source_frame_indices[index]),
                    "array_index": int(index),
                "timestamp_s": float(timestamps_s[index]),
                "match_error_s": float(error),
                "status": "MATCHED",
            }
        )
    return matches


def build_window_support(
    sequence: Any,
    *,
    window_start_s: float,
    component_config: ComponentConfig = COMPONENT_CONFIG,
    max_target_error_s: float = MAX_TARGET_ERROR_S,
) -> dict[str, Any]:
    """Build history-defined components and fixed-support local triplets.

    The returned dictionary is JSON-compatible apart from the NumPy arrays in
    ``triplets[*]["states"]``.  Call :func:`json_ready_support` before writing
    a manifest.  The arrays are intentionally retained for model preparation
    so no second numerical reconstruction is needed.
    """

    xyz = _as_float_array(sequence.xyz, ndim=3)
    valid = np.asarray(sequence.geometry_validity, dtype=bool)
    timestamps = _as_float_array(sequence.timestamps_s, ndim=1)
    frame_indices = np.asarray(sequence.frame_indices, dtype=np.int64)
    track_ids = np.asarray(sequence.track_ids, dtype=np.int64)
    if xyz.shape[-1] != 3 or valid.shape != xyz.shape[:2] or timestamps.shape != (xyz.shape[0],):
        raise ValueError("sequence arrays have incompatible shapes")
    if frame_indices.shape != (xyz.shape[0],) or track_ids.shape != (xyz.shape[1],):
        raise ValueError("sequence index arrays have incompatible shapes")
    if not np.all(np.isfinite(timestamps)) or (timestamps.size > 1 and not np.all(np.diff(timestamps) > 0)):
        raise ValueError("sequence timestamps must be finite and strictly increasing")

    boundary = float(window_start_s) + 0.5
    history_indices = np.flatnonzero(timestamps < boundary).astype(np.int64)
    evaluation_indices = np.flatnonzero(timestamps >= boundary).astype(np.int64)
    if history_indices.size == 0:
        raise ValueError("HISTORY_EMPTY")
    components = motion_coherent_components(xyz[history_indices], valid[history_indices], component_config)
    matches = _target_frame_matches(
        timestamps,
        frame_indices,
        float(window_start_s),
        max_error_s=max_target_error_s,
    )

    component_rows: list[dict[str, Any]] = []
    triplets: list[dict[str, Any]] = []
    invalid_reasons: list[dict[str, Any]] = []
    if not components:
        invalid_reasons.append({"reason": "NO_HISTORY_COMPONENT"})
    for component_index, component in enumerate(components):
        members = tuple(int(item) for item in component)
        scale, scale_count, _ = _history_scale(xyz, valid, history_indices, members)
        component_row: dict[str, Any] = {
            "component_index": int(component_index),
            "member_indices": list(members),
            "track_ids": [int(track_ids[item]) for item in members],
            "pair_ids": _pair_ids(track_ids, members),
            "history_frame_count": int(history_indices.size),
            "history_scale": scale,
            "history_scale_pair_time_count": int(scale_count),
            "validity_source": "ParticleSequence.geometry_validity and finite XYZ",
            "triplet_ids": [],
            "invalid_triplets": [],
        }
        if scale is None:
            reason = "HISTORY_SCALE_INVALID"
            component_row["invalid_triplets"] = [
                {"target_slots": [int(start), int(start + 1), int(start + 2)], "reason": reason}
                for start in range(3)
            ]
            invalid_reasons.append({"component_index": component_index, "reason": reason})
            component_rows.append(component_row)
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
                "members_considered": list(members),
                "track_ids_considered": [int(track_ids[item]) for item in members],
                "validity_source": "ParticleSequence.geometry_validity and finite XYZ at all three matched frames",
            }
            if any(item["frame_index"] is None for item in target_rows):
                reason = "TARGET_FRAME_MISSING"
                component_row["invalid_triplets"].append({"target_slots": list(slots), "reason": reason})
                invalid_reasons.append({"component_index": component_index, "target_slots": list(slots), "reason": reason})
                continue
            frame_tuple = tuple(int(item["array_index"]) for item in target_rows)
            geometry = valid[np.asarray(frame_tuple), :][:, np.asarray(members, dtype=np.int64)]
            finite_xyz = np.all(np.isfinite(xyz[np.asarray(frame_tuple)][:, np.asarray(members, dtype=np.int64), :]), axis=(0, 2))
            common = [member for position, member in enumerate(members) if bool(np.all(geometry[:, position])) and bool(finite_xyz[position])]
            if len(common) < 3:
                reason = "COMMON_VALID_MEMBERS_LT3"
                component_row["invalid_triplets"].append({"target_slots": list(slots), "reason": reason, "common_track_ids": [int(track_ids[item]) for item in common]})
                invalid_reasons.append({"component_index": component_index, "target_slots": list(slots), "reason": reason})
                continue
            states, pair_indices, reason = _state_from_common_members(xyz, frame_tuple, common, float(scale))
            if states is None or pair_indices is None or reason is not None:
                error_reason = reason or "STATE_INVALID"
                component_row["invalid_triplets"].append({"target_slots": list(slots), "reason": error_reason})
                invalid_reasons.append({"component_index": component_index, "target_slots": list(slots), "reason": error_reason})
                continue
            triplet_id = len(triplets)
            triplet = {
                **base,
                "triplet_id": int(triplet_id),
                "common_member_indices": list(common),
                "common_track_ids": [int(track_ids[item]) for item in common],
                "pair_indices": pair_indices,
                "pair_ids": [[int(track_ids[left]), int(track_ids[right])] for left, right in pair_indices],
                "states": states,
                "status": "VALID",
            }
            triplets.append(triplet)
            component_row["triplet_ids"].append(int(triplet_id))
        component_rows.append(component_row)

    return {
        "history_boundary_s": boundary,
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


def json_ready_support(support: Mapping[str, Any]) -> dict[str, Any]:
    """Convert a support record to JSON without changing numerical values."""

    output: dict[str, Any] = {}
    for key, value in support.items():
        if key == "triplets":
            output[key] = [
                {**{name: item_value for name, item_value in item.items() if name != "states"}, "states": np.asarray(item["states"], dtype=np.float64).tolist()}
                for item in value
            ]
        else:
            output[key] = value
    return output


def compute_local_derivatives(states: Sequence[Sequence[float]], timestamps_s: Sequence[float]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return center state, signed first derivative, and signed second derivative."""

    values = _as_float_array(states, ndim=2)
    times = _as_float_array(timestamps_s, ndim=1)
    if values.shape != (3, 4) or times.shape != (3,) or not np.all(np.isfinite(values)):
        raise ValueError("local triplet must contain finite states with shape [3,4]")
    h_minus = float(times[1] - times[0])
    h_plus = float(times[2] - times[1])
    if h_minus <= 0 or h_plus <= 0:
        raise ValueError("local triplet timestamps must be strictly increasing")
    v_minus = (values[1] - values[0]) / h_minus
    v_plus = (values[2] - values[1]) / h_plus
    acceleration = 2.0 * (v_plus - v_minus) / (h_minus + h_plus)
    return values[1].copy(), v_minus, acceleration


def arm_inputs_for_triplet(triplet: Mapping[str, Any]) -> dict[str, np.ndarray]:
    """Build the four fixed 12-D arm inputs for one valid triplet."""

    states = _as_float_array(triplet["states"], ndim=2)
    times = _as_float_array(triplet["timestamps_s"], ndim=1)
    if states.shape != (3, 4) or times.shape != (3,):
        raise ValueError("triplet states/timestamps have incompatible shape")
    center, first, second = compute_local_derivatives(states, times)
    ordered_first = np.concatenate((center, first, np.zeros(4, dtype=np.float64)))[None, :]
    ordered_second = np.concatenate((center, first, second))[None, :]
    unordered = np.stack([np.concatenate((state, state, state)) for state in states], axis=0)

    permuted_rows: list[np.ndarray] = []
    for permutation in permutations(range(3)):
        permuted = states[np.asarray(permutation, dtype=np.int64)]
        perm_center, perm_first, perm_second = compute_local_derivatives(permuted, times)
        permuted_rows.append(np.concatenate((perm_center, perm_first, perm_second)))
    return {
        "UNORDERED_STATE": unordered,
        "ORDERED_FIRST": ordered_first,
        "ORDERED_SECOND": ordered_second,
        "PERMUTED_SECOND": np.stack(permuted_rows, axis=0),
    }


def support_arrays(support: Mapping[str, Any]) -> dict[str, Any]:
    """Return only valid triplets in a compact model-preparation structure."""

    triplets = []
    for triplet in support.get("triplets", []):
        states = np.asarray(triplet["states"], dtype=np.float64)
        if states.shape != (3, 4) or not np.all(np.isfinite(states)):
            continue
        triplets.append(
            {
                "triplet_id": int(triplet["triplet_id"]),
                "component_index": int(triplet["component_index"]),
                "states": states,
                "timestamps_s": np.asarray(triplet["timestamps_s"], dtype=np.float64),
                "target_slots": tuple(int(item) for item in triplet["target_slots"]),
                "pair_ids": tuple(tuple(int(x) for x in pair) for pair in triplet["pair_ids"]),
                "common_track_ids": tuple(int(item) for item in triplet["common_track_ids"]),
            }
        )
    return {"triplets": triplets, "component_indices": tuple(sorted({item["component_index"] for item in triplets}))}
