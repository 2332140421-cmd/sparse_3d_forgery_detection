"""Small, deterministic helpers for the V7 boundary-partition pilot.

The only new operation here is a first-frame partition of an already frozen
historical local group.  It never creates an edge, merges two historical
groups, or looks at a target frame.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Mapping, Sequence

import numpy as np


def assign_uv_to_masks(
    uv: np.ndarray,
    visibility: np.ndarray,
    masks: Sequence[np.ndarray],
    areas: Sequence[float] | None = None,
) -> np.ndarray:
    """Assign first-frame UV points to masks.

    Returns ``-1`` for an unusable point, ``0`` for a valid background point,
    and ``1..K`` for mask indices.  Overlaps use the smaller original mask
    area, then the stable mask index.  ``masks`` are boolean arrays in the
    original RGB raster coordinate system; no category or confidence is used.
    """

    points = np.asarray(uv, dtype=np.float64)
    valid = np.asarray(visibility, dtype=bool)
    if points.ndim != 2 or points.shape[1] != 2 or valid.shape != (points.shape[0],):
        raise ValueError("uv must have shape [N,2] and visibility shape [N]")
    normalized: list[np.ndarray] = []
    shape: tuple[int, int] | None = None
    for mask in masks:
        value = np.asarray(mask, dtype=bool)
        if value.ndim != 2:
            raise ValueError("instance masks must be [H,W] boolean arrays")
        if shape is None:
            shape = (int(value.shape[0]), int(value.shape[1]))
        if value.shape != shape:
            raise ValueError("all instance masks must share the source raster shape")
        normalized.append(value)
    if areas is None:
        areas_array = np.asarray([np.count_nonzero(mask) for mask in normalized], dtype=np.float64)
    else:
        areas_array = np.asarray(areas, dtype=np.float64)
        if areas_array.shape != (len(normalized),) or np.any(~np.isfinite(areas_array)) or np.any(areas_array < 0):
            raise ValueError("mask areas must be finite and match mask count")
    result = np.full(points.shape[0], -1, dtype=np.int64)
    if shape is None:
        result[valid & np.all(np.isfinite(points), axis=1)] = 0
        return result
    height, width = shape
    for index, point in enumerate(points):
        if not bool(valid[index]) or not np.all(np.isfinite(point)):
            continue
        u, v = float(point[0]), float(point[1])
        if not (0.0 <= u < width and 0.0 <= v < height):
            continue
        x = min(width - 1, max(0, int(np.floor(u))))
        y = min(height - 1, max(0, int(np.floor(v))))
        candidates = [mask_index for mask_index, mask in enumerate(normalized) if bool(mask[y, x])]
        if candidates:
            winner = min(candidates, key=lambda mask_index: (float(areas_array[mask_index]), int(mask_index)))
            result[index] = int(winner) + 1
        else:
            result[index] = 0
    return result


def split_local_groups_by_assignment(
    historical_groups: Mapping[str, Any],
    assignments: Sequence[int],
    *,
    minimum_size: int = 3,
) -> dict[str, Any]:
    """Split each H group by frozen first-frame assignment.

    The returned child groups are disjoint subsets of their H parent.  The
    assignment is never recomputed for later frames.  Key ``0`` is background;
    key ``-1`` is unassigned/invalid and remains explicit.
    """

    values = np.asarray(assignments, dtype=np.int64)
    children: list[dict[str, Any]] = []
    parent_rows = list(historical_groups.get("groups", []))
    for parent in parent_rows:
        slots = tuple(int(item) for item in parent.get("member_slots", []))
        buckets: dict[int, list[int]] = defaultdict(list)
        for slot in slots:
            if slot < 0 or slot >= values.size:
                raise ValueError("assignment slot is outside particle sequence")
            buckets[int(values[slot])].append(slot)
        for key in sorted(buckets):
            members = tuple(sorted(buckets[key]))
            children.append(
                {
                    "parent_component_id": int(parent["parent_component_id"]),
                    "parent_local_group_id": int(parent.get("local_group_id", -1)),
                    "assignment_key": int(key),
                    "member_slots": list(members),
                    "track_ids": [int(item) for item in members],
                    "retained": len(members) >= int(minimum_size),
                    "status": "RETAINED" if len(members) >= int(minimum_size) else "SUPPORT_INSUFFICIENT_GROUP_SIZE",
                }
            )
    children.sort(key=lambda row: (int(row["parent_component_id"]), int(row["parent_local_group_id"]), int(row["assignment_key"]), tuple(row["member_slots"])))
    for local_id, row in enumerate(children):
        row["local_group_id"] = int(local_id)
    result = {
        "local_diameter_m": float(historical_groups.get("local_diameter_m", 0.30)),
        "component_config": dict(historical_groups.get("component_config", {})),
        "history_array_indices": [int(item) for item in historical_groups.get("history_array_indices", [])],
        "parent_components": list(historical_groups.get("parent_components", [])),
        "groups": children,
        "retained_group_count": int(sum(bool(row["retained"]) for row in children)),
        "support_insufficient_group_count": int(sum(not bool(row["retained"]) for row in children)),
        "retained_track_ids": sorted(int(slot) for row in children if row["retained"] for slot in row["member_slots"]),
    }
    return result


def validate_boundary_partition(
    historical_groups: Mapping[str, Any],
    boundary_groups: Mapping[str, Any],
) -> dict[str, bool]:
    """Check the non-negotiable H→B partition invariants."""

    h_by_id = {int(row.get("local_group_id", -1)): set(int(x) for x in row.get("member_slots", [])) for row in historical_groups.get("groups", [])}
    children_by_parent: dict[int, list[set[int]]] = defaultdict(list)
    subset = True
    for row in boundary_groups.get("groups", []):
        parent = int(row.get("parent_local_group_id", -1))
        members = set(int(x) for x in row.get("member_slots", []))
        if parent not in h_by_id or not members.issubset(h_by_id[parent]):
            subset = False
        children_by_parent[parent].append(members)
    disjoint = all(not (left & right) for groups in children_by_parent.values() for i, left in enumerate(groups) for right in groups[i + 1 :])
    return {"child_subset_of_h": subset, "children_disjoint_within_parent": disjoint, "all_pass": bool(subset and disjoint)}
