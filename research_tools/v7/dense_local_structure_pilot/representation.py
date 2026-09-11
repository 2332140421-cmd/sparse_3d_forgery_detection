"""Fixed-edge state and three-arm representation helpers for the pilot."""

from __future__ import annotations

from itertools import permutations
from typing import Any, Mapping, Sequence

import numpy as np

from research_tools.v7.local_structural_temporal_probe.representation import (
    TARGET_OFFSETS_S,
    arm_inputs_for_triplet,
    compute_local_derivatives,
)


def edge_distances(xyz: np.ndarray, edges: Sequence[Mapping[str, Any]], frame: int) -> np.ndarray:
    values = []
    for edge in edges:
        left, right = int(edge["left_query_id"]), int(edge["right_query_id"])
        if np.all(np.isfinite(xyz[frame, left])) and np.all(np.isfinite(xyz[frame, right])):
            values.append(float(np.linalg.norm(xyz[frame, right] - xyz[frame, left])))
    return np.asarray(values, dtype=np.float64)


def state_from_distances(values: np.ndarray, scale: float) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or values.size == 0 or not np.isfinite(scale) or scale <= 0:
        raise ValueError("finite non-empty distances and positive scale are required")
    normalized = values / scale
    if not np.all(np.isfinite(normalized)):
        raise ValueError("normalized distances must be finite")
    return np.asarray(
        [np.mean(normalized), np.std(normalized), np.percentile(normalized, 25), np.percentile(normalized, 75)],
        dtype=np.float64,
    )


def fixed_edge_triplet(
    xyz: np.ndarray,
    timestamps_s: np.ndarray,
    edges: Sequence[Mapping[str, Any]],
    history_indices: Sequence[int],
    target_indices: Sequence[int],
) -> dict[str, Any] | None:
    """Construct one triplet using exactly the same valid edge identities."""

    xyz = np.asarray(xyz, dtype=np.float64)
    timestamps = np.asarray(timestamps_s, dtype=np.float64)
    history = np.asarray(history_indices, dtype=np.int64)
    target = np.asarray(target_indices, dtype=np.int64)
    if xyz.ndim != 3 or xyz.shape[-1] != 3 or timestamps.shape != (xyz.shape[0],):
        raise ValueError("incompatible xyz/timestamp shapes")
    if target.shape != (3,) or history.ndim != 1:
        raise ValueError("history and target indices are invalid")
    valid_edges = []
    for edge in edges:
        left, right = int(edge["left_query_id"]), int(edge["right_query_id"])
        values = []
        for frame in history:
            if np.all(np.isfinite(xyz[frame, left])) and np.all(np.isfinite(xyz[frame, right])):
                values.append(float(np.linalg.norm(xyz[frame, right] - xyz[frame, left])))
        if values:
            valid_edges.append((edge, values))
    all_history = np.asarray([value for _edge, values in valid_edges for value in values], dtype=np.float64)
    if all_history.size == 0:
        return None
    scale = float(np.median(all_history))
    if not np.isfinite(scale) or scale <= 0:
        return None
    common = []
    states = []
    for frame in target:
        values = []
        for edge, _history_values in valid_edges:
            left, right = int(edge["left_query_id"]), int(edge["right_query_id"])
            if np.all(np.isfinite(xyz[frame, left])) and np.all(np.isfinite(xyz[frame, right])):
                values.append(float(np.linalg.norm(xyz[frame, right] - xyz[frame, left])))
        if len(values) < 3:
            return None
        states.append(state_from_distances(np.asarray(values), scale))
        common.append(values)
    if len(valid_edges) < 3:
        return None
    query_ids = sorted({int(edge["left_query_id"]) for edge, _ in valid_edges} | {int(edge["right_query_id"]) for edge, _ in valid_edges})
    if len(query_ids) < 3:
        return None
    return {
        "states": np.asarray(states, dtype=np.float64),
        "timestamps_s": timestamps[target].astype(np.float64),
        "history_scale": scale,
        "edge_ids": [int(edge["edge_id"]) for edge, _ in valid_edges],
        "query_ids": query_ids,
    }


def build_triplet_arms(triplet: Mapping[str, Any]) -> dict[str, np.ndarray]:
    """Use the already frozen 12-D arm definitions without adding features."""

    return arm_inputs_for_triplet(triplet)
