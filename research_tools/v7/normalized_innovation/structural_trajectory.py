"""Fixed-pair relation trajectories and local NSI triplets.

This module only reorganizes the already materialized V7 particle artifacts.
It deliberately does not run a frontend, infer components, or repair missing
observations.
"""

from __future__ import annotations

from itertools import combinations
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .normalized_innovation import normalized_structural_innovation


EPSILON = 1e-8
MINIMUM_OVERLAP = 8


def aggregate_component_q90(values: Sequence[float]) -> float | None:
    """Aggregate component Q90 scores with equal component weight."""

    finite = np.asarray(list(values), dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    return float(np.median(finite)) if finite.size else None


def _component_scales(xyz: np.ndarray, validity: np.ndarray, members: Sequence[int]) -> np.ndarray:
    """Return the frozen median valid pair-distance scale at each frame."""

    scales = np.full(xyz.shape[0], np.nan, dtype=np.float64)
    member_ids = tuple(sorted(int(item) for item in members))
    for time_index in range(xyz.shape[0]):
        valid_ids = [item for item in member_ids if validity[time_index, item]]
        if len(valid_ids) < 2:
            continue
        points = np.asarray(xyz[time_index, valid_ids], dtype=np.float64)
        distances = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=-1)
        distances = distances[np.triu_indices(len(valid_ids), k=1)]
        distances = distances[np.isfinite(distances)]
        if distances.size:
            scale = float(np.median(distances))
            if np.isfinite(scale) and scale > 0:
                scales[time_index] = scale
    return scales


def _relation_trajectories(
    xyz: np.ndarray,
    validity: np.ndarray,
    timestamps_s: np.ndarray,
    members: Sequence[int],
) -> tuple[dict[tuple[int, int], np.ndarray], dict[tuple[int, int], np.ndarray], np.ndarray]:
    """Build fixed-identity normalized relation trajectories."""

    scales = _component_scales(xyz, validity, members)
    trajectories: dict[tuple[int, int], np.ndarray] = {}
    masks: dict[tuple[int, int], np.ndarray] = {}
    for left, right in combinations(sorted(int(item) for item in members), 2):
        values = np.full(xyz.shape[0], np.nan, dtype=np.float64)
        valid = np.zeros(xyz.shape[0], dtype=bool)
        for time_index in range(xyz.shape[0]):
            if not (validity[time_index, left] and validity[time_index, right]):
                continue
            scale = scales[time_index]
            if not np.isfinite(scale):
                continue
            distance = float(np.linalg.norm(np.asarray(xyz[time_index, right], dtype=np.float64) - np.asarray(xyz[time_index, left], dtype=np.float64)))
            relation = distance / (scale + EPSILON)
            if np.isfinite(relation):
                values[time_index] = relation
                valid[time_index] = True
        trajectories[(left, right)] = values
        masks[(left, right)] = valid
    return trajectories, masks, scales


def _component_triplets(
    xyz: np.ndarray,
    validity: np.ndarray,
    timestamps_s: np.ndarray,
    frame_indices: np.ndarray,
    members: Sequence[int],
    component_index: int,
) -> dict[str, Any]:
    """Compute all valid local triplets for one frozen component."""

    trajectories, masks, scales = _relation_trajectories(xyz, validity, timestamps_s, members)
    persistent = {
        pair: int(np.sum(mask)) >= MINIMUM_OVERLAP
        for pair, mask in masks.items()
    }
    rows: list[dict[str, Any]] = []
    for center in range(1, xyz.shape[0] - 1):
        if frame_indices[center] - frame_indices[center - 1] != 1 or frame_indices[center + 1] - frame_indices[center] != 1:
            continue
        h0 = float(timestamps_s[center] - timestamps_s[center - 1])
        h1 = float(timestamps_s[center + 1] - timestamps_s[center])
        if not (np.isfinite(h0) and np.isfinite(h1) and h0 > 0 and h1 > 0):
            continue
        common_pairs = [
            pair
            for pair in sorted(trajectories)
            if persistent[pair]
            and masks[pair][center - 1]
            and masks[pair][center]
            and masks[pair][center + 1]
        ]
        if not common_pairs:
            continue
        r_minus = np.asarray([trajectories[pair][center - 1] for pair in common_pairs], dtype=np.float64)
        r_zero = np.asarray([trajectories[pair][center] for pair in common_pairs], dtype=np.float64)
        r_plus = np.asarray([trajectories[pair][center + 1] for pair in common_pairs], dtype=np.float64)
        v_minus = (r_zero - r_minus) / h0
        v_plus = (r_plus - r_zero) / h1
        if not (np.all(np.isfinite(v_minus)) and np.all(np.isfinite(v_plus))):
            continue
        step_minus = float(np.linalg.norm(r_zero - r_minus))
        step_plus = float(np.linalg.norm(r_plus - r_zero))
        speed_sum = float(np.linalg.norm(v_minus) + np.linalg.norm(v_plus))
        raw_second = float(np.linalg.norm(v_plus - v_minus))
        innovation = normalized_structural_innovation(v_minus, v_plus, epsilon=EPSILON)
        rows.append(
            {
                "component_index": int(component_index),
                "center_index": int(center),
                "frame_index_prev": int(frame_indices[center - 1]),
                "frame_index": int(frame_indices[center]),
                "frame_index_next": int(frame_indices[center + 1]),
                "timestamp_prev": float(timestamps_s[center - 1]),
                "timestamp": float(timestamps_s[center]),
                "timestamp_next": float(timestamps_s[center + 1]),
                "h0_s": h0,
                "h1_s": h1,
                "common_pair_count": len(common_pairs),
                "structural_innovation": innovation,
                "speed_sum": speed_sum,
                "raw_delta_r_norm": float((step_minus + step_plus) / 2.0),
                "raw_second_difference_norm": raw_second,
            }
        )
    component_rows = [row for row in rows if row["component_index"] == component_index]
    innovations = np.asarray([row["structural_innovation"] for row in component_rows], dtype=np.float64)
    return {
        "component_index": int(component_index),
        "member_count": len(tuple(members)),
        "persistent_pair_count": int(sum(persistent.values())),
        "triplet_count": int(len(component_rows)),
        "common_pair_dimension_median": float(np.median([row["common_pair_count"] for row in component_rows])) if component_rows else None,
        "q90": float(np.percentile(innovations, 90)) if innovations.size else None,
        "median": float(np.median(innovations)) if innovations.size else None,
        "triplets": component_rows,
        "scale_valid_count": int(np.sum(np.isfinite(scales))),
    }


def window_nsi(
    npz_path: str | Path,
    component_rows: Sequence[dict[str, Any]],
    timestamps_s: Sequence[float],
    frame_indices: Sequence[int],
) -> dict[str, Any]:
    """Extract NSI triplets and fixed component/window aggregation."""

    with np.load(npz_path, allow_pickle=False) as arrays:
        xyz = np.asarray(arrays["xyz"], dtype=np.float64)
        validity = np.asarray(arrays["geometry_validity"], dtype=bool)
        stored_timestamps = np.asarray(arrays["timestamps_s"], dtype=np.float64) if "timestamps_s" in arrays else None
        stored_indices = np.asarray(arrays["frame_indices"], dtype=np.int64) if "frame_indices" in arrays else None
    timestamps = np.asarray(timestamps_s, dtype=np.float64)
    indices = np.asarray(frame_indices, dtype=np.int64)
    if xyz.ndim != 3 or xyz.shape[2] != 3 or validity.shape != xyz.shape[:2]:
        raise ValueError("particle arrays have incompatible xyz/validity shapes")
    if timestamps.shape != (xyz.shape[0],) or indices.shape != (xyz.shape[0],):
        raise ValueError("window timestamps/frame indices do not match particle artifact")
    if stored_timestamps is not None and not np.array_equal(stored_timestamps, timestamps):
        raise ValueError("caller timestamps differ from particle artifact")
    if stored_indices is not None and not np.array_equal(stored_indices, indices):
        raise ValueError("caller frame indices differ from particle artifact")
    components = [
        _component_triplets(xyz, validity, timestamps, indices, row["members"], int(row["component_index"]))
        for row in sorted(component_rows, key=lambda item: int(item["component_index"]))
    ]
    valid_components = [item for item in components if item["q90"] is not None]
    q90_values = [item["q90"] for item in valid_components]
    median_values = [item["median"] for item in valid_components]
    all_triplets = [triplet for component in components for triplet in component["triplets"]]
    common_dims = [row["common_pair_count"] for row in all_triplets]
    speed_values = [row["speed_sum"] for row in all_triplets]
    delta_values = [row["raw_delta_r_norm"] for row in all_triplets]
    second_values = [row["raw_second_difference_norm"] for row in all_triplets]
    return {
        "component_count": len(components),
        "component_with_valid_nsi": len(valid_components),
        "triplet_count": len(all_triplets),
        "persistent_pair_count": int(sum(item["persistent_pair_count"] for item in components)),
        "common_pair_dimension_median": float(np.median(common_dims)) if common_dims else None,
        "I_window": aggregate_component_q90(q90_values),
        "component_q90": q90_values,
        "component_median": median_values,
        "speed_sum_median": float(np.median(speed_values)) if speed_values else None,
        "raw_delta_r_median": float(np.median(delta_values)) if delta_values else None,
        "raw_second_difference_median": float(np.median(second_values)) if second_values else None,
        "range_violations": int(sum(not (-1e-10 <= row["structural_innovation"] <= 1.0 + 1e-10) for row in all_triplets)),
        "components": components,
    }
