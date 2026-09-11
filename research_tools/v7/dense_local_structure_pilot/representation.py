"""Fixed-edge state and three-arm representation helpers for the pilot."""

from __future__ import annotations

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
    if history.size == 0 or len(edges) == 0:
        return None
    left_ids = np.asarray([int(edge["left_query_id"]) for edge in edges], dtype=np.int64)
    right_ids = np.asarray([int(edge["right_query_id"]) for edge in edges], dtype=np.int64)
    history_left = xyz[history[:, None], left_ids[None, :], :]
    history_right = xyz[history[:, None], right_ids[None, :], :]
    history_valid = np.all(np.isfinite(history_left), axis=2) & np.all(np.isfinite(history_right), axis=2)
    history_distances = np.linalg.norm(history_right - history_left, axis=2)
    valid_edge_mask = np.any(history_valid, axis=0)
    all_history = history_distances[history_valid]
    if all_history.size == 0:
        return None
    scale = float(np.median(all_history))
    if not np.isfinite(scale) or scale <= 0:
        return None
    target_left = xyz[target[:, None], left_ids[None, :], :]
    target_right = xyz[target[:, None], right_ids[None, :], :]
    target_valid = np.all(np.isfinite(target_left), axis=2) & np.all(np.isfinite(target_right), axis=2)
    common_mask = valid_edge_mask & np.all(target_valid, axis=0)
    common_indices = np.flatnonzero(common_mask)
    common_edges = [edges[int(index)] for index in common_indices]
    if len(common_edges) < 3:
        return None
    query_ids = sorted({int(edge["left_query_id"]) for edge in common_edges} | {int(edge["right_query_id"]) for edge in common_edges})
    if len(query_ids) < 3:
        return None
    common_distances = np.linalg.norm(target_right[:, common_indices, :] - target_left[:, common_indices, :], axis=2) / scale
    states = np.asarray(
        [[np.mean(row), np.std(row), np.percentile(row, 25), np.percentile(row, 75)] for row in common_distances],
        dtype=np.float64,
    )
    return {
        "states": np.asarray(states, dtype=np.float64),
        "timestamps_s": timestamps[target].astype(np.float64),
        "history_scale": scale,
        "edge_ids": [int(edge["edge_id"]) for edge in common_edges],
        "query_ids": query_ids,
    }


def build_triplet_arms(triplet: Mapping[str, Any]) -> dict[str, np.ndarray]:
    """Use the already frozen 12-D arm definitions without adding features."""

    return arm_inputs_for_triplet(triplet)
