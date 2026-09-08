"""Derivative-before-aggregation relation representation for the V7 pilot."""

from __future__ import annotations

from itertools import combinations
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .temporal_derivatives import timestamp_derivatives


EPSILON = 1e-12
MINIMUM_OVERLAP = 8
MINIMUM_DESCRIPTOR_SAMPLES = 3


def descriptor(values: np.ndarray, valid: np.ndarray, minimum_samples: int = MINIMUM_DESCRIPTOR_SAMPLES) -> np.ndarray | None:
    """Return the fixed four-number signed/magnitude descriptor for one pair."""

    finite = np.asarray(values, dtype=np.float64)[np.asarray(valid, dtype=bool)]
    finite = finite[np.isfinite(finite)]
    if finite.size < minimum_samples:
        return None
    median = float(np.median(finite))
    return np.asarray(
        [median, np.median(np.abs(finite - median)), np.median(np.abs(finite)), np.percentile(np.abs(finite), 90)],
        dtype=np.float64,
    )


def robust_scale(values: Sequence[Sequence[float]]) -> tuple[np.ndarray, dict[str, Any]]:
    """Use the frozen real-control MAD/IQR/floor rule for arbitrary dimensions."""

    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2 or array.shape[0] == 0:
        raise ValueError("scale values must be a non-empty two-dimensional array")
    center = np.median(array, axis=0)
    mad = np.median(np.abs(array - center), axis=0)
    scale = 1.4826 * mad
    iqr_dimensions: list[int] = []
    floor_dimensions: list[int] = []
    for index in range(array.shape[1]):
        if not np.isfinite(scale[index]) or scale[index] == 0:
            fallback = (np.percentile(array[:, index], 75) - np.percentile(array[:, index], 25)) / 1.349
            if np.isfinite(fallback) and fallback > 0:
                scale[index] = fallback
                iqr_dimensions.append(index)
            else:
                scale[index] = max(1e-6, 1e-6 * float(np.median(np.abs(array[:, index]))))
                floor_dimensions.append(index)
    return scale, {
        "center_median": center.tolist(),
        "scale": scale.tolist(),
        "mad": mad.tolist(),
        "iqr_fallback_dimensions": iqr_dimensions,
        "floor_fallback_dimensions": floor_dimensions,
        "fallback_count": len(iqr_dimensions) + len(floor_dimensions),
        "source": "real_control_windows_only",
    }


def _component_relation_trajectories(
    xyz: np.ndarray,
    geometry_validity: np.ndarray,
    timestamps_s: np.ndarray,
    frame_indices: np.ndarray,
    members: Sequence[int],
) -> dict[str, Any]:
    member_ids = tuple(int(item) for item in members)
    count_t = xyz.shape[0]
    scales = np.full(count_t, np.nan, dtype=np.float64)
    relation_values: dict[tuple[int, int], np.ndarray] = {}
    relation_valid: dict[tuple[int, int], np.ndarray] = {}
    persistent_pairs = 0
    r1_descriptors: list[np.ndarray] = []
    r2_descriptors: list[np.ndarray] = []
    pair_rows: list[dict[str, Any]] = []
    for time_index in range(count_t):
        ids = np.asarray([item for item in member_ids if geometry_validity[time_index, item]], dtype=np.int64)
        if ids.size < 3:
            continue
        points = xyz[time_index, ids].astype(np.float64)
        distances = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=-1)
        distances = distances[np.triu_indices(ids.size, k=1)]
        finite_distances = distances[np.isfinite(distances)]
        if finite_distances.size:
            scale = float(np.median(finite_distances))
            if np.isfinite(scale) and scale > 0:
                scales[time_index] = scale
    for left, right in combinations(member_ids, 2):
        key = (left, right)
        values = np.full(count_t, np.nan, dtype=np.float64)
        valid = np.zeros(count_t, dtype=bool)
        for time_index in range(count_t):
            if not geometry_validity[time_index, left] or not geometry_validity[time_index, right] or not np.isfinite(scales[time_index]):
                continue
            distance = float(np.linalg.norm(xyz[time_index, right].astype(np.float64) - xyz[time_index, left].astype(np.float64)))
            relation = distance / (scales[time_index] + EPSILON)
            if np.isfinite(relation):
                values[time_index] = relation
                valid[time_index] = True
        overlap = int(np.sum(valid))
        persistent = overlap >= MINIMUM_OVERLAP
        if persistent:
            persistent_pairs += 1
        if persistent:
            first, first_valid, second, second_valid = timestamp_derivatives(values, valid, timestamps_s, frame_indices)
            first_descriptor = descriptor(first, first_valid)
            second_descriptor = descriptor(second, second_valid)
        else:
            first_valid = np.zeros(count_t, dtype=bool)
            second_valid = np.zeros(count_t, dtype=bool)
            first_descriptor = None
            second_descriptor = None
        if first_descriptor is not None:
            r1_descriptors.append(first_descriptor)
        if second_descriptor is not None:
            r2_descriptors.append(second_descriptor)
        pair_rows.append(
            {
                "left_track": left,
                "right_track": right,
                "valid_relation_samples": overlap,
                "persistent": persistent,
                "r1_valid_samples": int(np.sum(first_valid)),
                "r2_valid_samples": int(np.sum(second_valid)),
                "r1_eligible": first_descriptor is not None,
                "r2_eligible": second_descriptor is not None,
            }
        )
    return {
        "members": list(member_ids),
        "pair_candidate_count": len(pair_rows),
        "persistent_pair_count": persistent_pairs,
        "r1_eligible_pair_count": len(r1_descriptors),
        "r2_eligible_pair_count": len(r2_descriptors),
        "r1_pair_descriptors": [item.tolist() for item in r1_descriptors],
        "r2_pair_descriptors": [item.tolist() for item in r2_descriptors],
        "pair_rows": pair_rows,
        "scale_valid_count": int(np.sum(np.isfinite(scales))),
    }


def aggregate_component(pair_descriptors: Sequence[Sequence[float]]) -> dict[str, Any] | None:
    """Aggregate eligible pair descriptors with equal pair weight."""

    if not pair_descriptors:
        return None
    values = np.asarray(pair_descriptors, dtype=np.float64)
    median = np.median(values, axis=0)
    iqr = np.percentile(values, 75, axis=0) - np.percentile(values, 25, axis=0)
    return {"pair_count": int(values.shape[0]), "descriptor_median": median.tolist(), "descriptor_iqr": iqr.tolist(), "vector": np.concatenate([median, iqr]).tolist()}


def aggregate_window(component_vectors: Sequence[Sequence[float]]) -> list[float] | None:
    """Aggregate components equally; each component contributes one 8-D vector."""

    if not component_vectors:
        return None
    return np.median(np.asarray(component_vectors, dtype=np.float64), axis=0).tolist()


def relation_first_window(
    npz_path: str | Path,
    component_rows: Sequence[dict[str, Any]],
    timestamps_s: Sequence[float],
    frame_indices: Sequence[int],
) -> dict[str, Any]:
    """Extract R1/R2 descriptors from one existing particle artifact."""

    with np.load(npz_path, allow_pickle=False) as arrays:
        xyz = np.asarray(arrays["xyz"], dtype=np.float64)
        validity = np.asarray(arrays["geometry_validity"], dtype=bool)
    timestamps = np.asarray(timestamps_s, dtype=np.float64)
    indices = np.asarray(frame_indices, dtype=np.int64)
    if xyz.ndim != 3 or validity.shape != xyz.shape[:2] or timestamps.shape != (xyz.shape[0],) or indices.shape != (xyz.shape[0],):
        raise ValueError("particle artifact arrays are incompatible")
    components: list[dict[str, Any]] = []
    r1_vectors: list[list[float]] = []
    r2_vectors: list[list[float]] = []
    persistent_total = r1_total = r2_total = candidate_total = 0
    for row in component_rows:
        trajectories = _component_relation_trajectories(xyz, validity, timestamps, indices, row["members"])
        r1_component = aggregate_component(trajectories["r1_pair_descriptors"])
        r2_component = aggregate_component(trajectories["r2_pair_descriptors"])
        trajectories["component_index"] = int(row["component_index"])
        trajectories["r1_component"] = r1_component
        trajectories["r2_component"] = r2_component
        components.append(trajectories)
        candidate_total += trajectories["pair_candidate_count"]
        persistent_total += trajectories["persistent_pair_count"]
        r1_total += trajectories["r1_eligible_pair_count"]
        r2_total += trajectories["r2_eligible_pair_count"]
        if r1_component is not None:
            r1_vectors.append(r1_component["vector"])
        if r2_component is not None:
            r2_vectors.append(r2_component["vector"])
    return {
        "component_count": len(component_rows),
        "component_with_valid_r1": len(r1_vectors),
        "component_with_valid_r2": len(r2_vectors),
        "pair_candidate_count": candidate_total,
        "persistent_pair_count": persistent_total,
        "r1_eligible_pair_count": r1_total,
        "r2_eligible_pair_count": r2_total,
        "r1_eligible_fraction": float(r1_total / persistent_total) if persistent_total else 0.0,
        "r2_eligible_fraction": float(r2_total / persistent_total) if persistent_total else 0.0,
        "r1_window": aggregate_window(r1_vectors),
        "r2_window": aggregate_window(r2_vectors),
        "components": components,
    }
