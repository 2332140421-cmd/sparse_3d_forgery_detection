"""Small, label-blind diagnostics for the frozen V7 component contract.

This module answers a measurement question, not a detector question: does a
denser nested query grid change observation support and local grouping when
the depth, pose, tracking model, time window, and component rule are held
fixed?  The helpers intentionally operate on canonical ``ParticleSequence``
objects and do not fit a model or alter the formal detection chain.
"""

from __future__ import annotations

from collections import deque
from itertools import combinations
from typing import Any, Mapping, Sequence

import numpy as np

from sparse3d_forgery.experiments.v7_dynamic_structure_probe import ComponentConfig
from research_tools.v7.local_structural_temporal_probe.representation import build_window_support


COMPONENT_CONFIG = ComponentConfig(
    max_initial_distance=1.0,
    max_relative_change=0.05,
    minimum_size=3,
    minimum_overlap=8,
)


def query_grid_coordinates(process_size: int, grid_size: int) -> np.ndarray:
    """Return the existing row-major ``(u, v)`` query coordinates."""

    if process_size <= 0 or grid_size <= 0:
        raise ValueError("process_size and grid_size must be positive")
    grid = np.linspace(0, process_size - 1, grid_size + 2, dtype=np.float32)[1:-1]
    uu, vv = np.meshgrid(grid, grid)
    return np.stack((uu.ravel(), vv.ravel()), axis=1)


def nested_query_mapping(
    process_size: int = 256, base_grid_size: int = 8, dense_grid_size: int = 17
) -> list[dict[str, Any]]:
    """Map every 8x8 coordinate to the identical location in a 17x17 grid."""

    base = query_grid_coordinates(process_size, base_grid_size)
    dense = query_grid_coordinates(process_size, dense_grid_size)
    mapping: list[dict[str, Any]] = []
    for base_index in range(base.shape[0]):
        row, col = divmod(base_index, base_grid_size)
        # Both grids use an interior linspace.  The 8-grid locations therefore
        # land at odd (not even) coordinates of the 17-grid.
        dense_index = (2 * row + 1) * dense_grid_size + (2 * col + 1)
        if not np.allclose(base[base_index], dense[dense_index], rtol=0.0, atol=1e-6):
            raise AssertionError("nested grid coordinate mismatch")
        mapping.append(
            {
                "base_index": int(base_index),
                "dense_index": int(dense_index),
                "base_u": float(base[base_index, 0]),
                "base_v": float(base[base_index, 1]),
                "dense_u": float(dense[dense_index, 0]),
                "dense_v": float(dense[dense_index, 1]),
                "track_id_base": int(base_index),
                "track_id_dense": int(dense_index),
            }
        )
    return mapping


def qualified_graph_edges(
    xyz: np.ndarray, valid: np.ndarray, config: ComponentConfig = COMPONENT_CONFIG
) -> set[tuple[int, int]]:
    """Reproduce the frozen direct-edge rule without changing it."""

    xyz = np.asarray(xyz, dtype=np.float64)
    valid = np.asarray(valid, dtype=bool)
    if xyz.ndim != 3 or xyz.shape[-1] != 3 or valid.shape != xyz.shape[:2]:
        raise ValueError("expected xyz [T,N,3] and valid [T,N]")
    edges: set[tuple[int, int]] = set()
    for left in range(xyz.shape[1]):
        for right in range(left + 1, xyz.shape[1]):
            overlap = valid[:, left] & valid[:, right]
            if int(np.sum(overlap)) < config.minimum_overlap:
                continue
            relative = xyz[overlap, right] - xyz[overlap, left]
            if np.linalg.norm(relative[0]) > config.max_initial_distance:
                continue
            change = np.linalg.norm(relative - np.median(relative, axis=0), axis=1)
            if float(np.median(change)) <= config.max_relative_change:
                edges.add((left, right))
    return edges


def _path(edges: set[tuple[int, int]], start: int, goal: int) -> list[int] | None:
    adjacency: dict[int, list[int]] = {}
    for left, right in edges:
        adjacency.setdefault(left, []).append(right)
        adjacency.setdefault(right, []).append(left)
    queue: deque[int] = deque([start])
    previous = {start: None}
    while queue:
        node = queue.popleft()
        if node == goal:
            result: list[int] = []
            while node is not None:
                result.append(node)
                node = previous[node]
            return list(reversed(result))
        for neighbor in sorted(adjacency.get(node, [])):
            if neighbor not in previous:
                previous[neighbor] = node
                queue.append(neighbor)
    return None


def _indirect_examples(
    xyz: np.ndarray,
    valid: np.ndarray,
    members: Sequence[int],
    edges: set[tuple[int, int]],
    limit: int = 3,
) -> list[dict[str, Any]]:
    examples: list[dict[str, Any]] = []
    for left, right in combinations(sorted(int(x) for x in members), 2):
        edge = (left, right) if left < right else (right, left)
        if edge in edges:
            continue
        path = _path(edges, left, right)
        if path is None or len(path) < 3:
            continue
        overlap = valid[:, left] & valid[:, right]
        if not np.any(overlap):
            continue
        relative = xyz[overlap, right] - xyz[overlap, left]
        initial = float(np.linalg.norm(relative[0]))
        change = np.linalg.norm(relative - np.median(relative, axis=0), axis=1)
        examples.append(
            {
                "left_slot": left,
                "right_slot": right,
                "path": path,
                "initial_distance": initial,
                "history_overlap": int(np.sum(overlap)),
                "relative_change_median": float(np.median(change)),
            }
        )
        if len(examples) >= limit:
            break
    return examples


def _triplet_ranges(sequence: Any, support: Mapping[str, Any]) -> dict[str, Any]:
    xyz = np.asarray(sequence.xyz, dtype=np.float64)
    uv = np.asarray(sequence.uv, dtype=np.float64)
    valid = np.asarray(sequence.geometry_validity, dtype=bool)
    for triplet in support.get("triplets", []):
        members = np.asarray(triplet["common_member_indices"], dtype=np.int64)
        if "array_indices" in triplet:
            frames = np.asarray(triplet["array_indices"], dtype=np.int64)
        else:
            source_frames = np.asarray(triplet.get("frame_indices", []), dtype=np.int64)
            sequence_frames = np.asarray(sequence.frame_indices, dtype=np.int64)
            positions = np.searchsorted(sequence_frames, source_frames)
            if source_frames.size != 3 or np.any(positions >= sequence_frames.size) or not np.array_equal(sequence_frames[positions], source_frames):
                continue
            frames = positions.astype(np.int64)
        if members.size < 3 or frames.size != 3:
            continue
        ranges_2d: list[list[float]] = []
        ranges_3d: list[list[float]] = []
        for frame in frames:
            okay = valid[frame, members] & np.all(np.isfinite(uv[frame, members]), axis=1)
            if int(np.sum(okay)) < 3:
                continue
            xy = uv[frame, members[okay]]
            ranges_2d.append((np.max(xy, axis=0) - np.min(xy, axis=0)).astype(float).tolist())
            points = xyz[frame, members[okay]]
            if np.all(np.isfinite(points)):
                ranges_3d.append((np.max(points, axis=0) - np.min(points, axis=0)).astype(float).tolist())
        return {
            "triplet_id": int(triplet["triplet_id"]),
            "common_member_count": int(members.size),
                "source_frame_indices": [int(x) for x in triplet.get("source_frame_indices", triplet.get("frame_indices", []))],
            "timestamps_s": [float(x) for x in triplet["timestamps_s"]],
            "range_2d_per_frame": ranges_2d,
            "range_3d_per_frame": ranges_3d,
        }
    return {
        "triplet_id": None,
        "common_member_count": 0,
        "source_frame_indices": [],
        "timestamps_s": [],
        "range_2d_per_frame": [],
        "range_3d_per_frame": [],
    }


def component_diagnostic(sequence: Any, *, window_start_s: float, label: str = "history64") -> dict[str, Any]:
    """Rebuild one component graph and return support/coverage diagnostics."""

    xyz = np.asarray(sequence.xyz, dtype=np.float64)
    valid = np.asarray(sequence.geometry_validity, dtype=bool)
    timestamps = np.asarray(sequence.timestamps_s, dtype=np.float64)
    history_indices = np.flatnonzero(timestamps < float(window_start_s) + 0.5)
    support = build_window_support(sequence, window_start_s=window_start_s, component_config=COMPONENT_CONFIG)
    edges = qualified_graph_edges(xyz[history_indices], valid[history_indices], COMPONENT_CONFIG)
    components = [tuple(int(x) for x in row["member_indices"]) for row in support["components"]]
    grouped = {member for component in components for member in component}
    component_ids = {member: index for index, component in enumerate(components) for member in component}
    component_edges = {
        edge for edge in edges
        if edge[0] in component_ids and edge[1] in component_ids and component_ids[edge[0]] == component_ids[edge[1]]
    }
    internal_pairs = sum(len(component) * (len(component) - 1) // 2 for component in components)
    valid_triplets = support.get("triplets", [])
    triplet_pair_count = sum(len(triplet.get("pair_indices", [])) for triplet in valid_triplets)
    indirect: list[dict[str, Any]] = []
    for component in components:
        indirect.extend(_indirect_examples(xyz[history_indices], valid[history_indices], component, edges))
    indirect = indirect[:3]
    triplet_range = _triplet_ranges(sequence, support)
    max_component = max((len(component) for component in components), default=0)
    return {
        "density_label": label,
        "history_frame_count": int(history_indices.size),
        "history_valid_point_observations": int(np.sum(valid[history_indices])),
        "history_candidate_points": int(np.sum(np.any(valid[history_indices], axis=0))),
        "qualified_graph_edge_count": int(len(edges)),
        "qualified_graph_edge_count_within_components": int(len(component_edges)),
        "component_count": int(len(components)),
        "component_member_count_total": int(len(grouped)),
        "max_component_members": int(max_component),
        "max_component_share_of_grouped": float(max_component / len(grouped)) if grouped else 0.0,
        "component_internal_pair_count": int(internal_pairs),
        "triplet_measured_pair_count": int(triplet_pair_count),
        "qualified_edge_fraction_of_internal_pairs": float(len(component_edges) / internal_pairs) if internal_pairs else None,
        "valid_triplet_count": int(len(valid_triplets)),
        "selected_triplet_common_members": int(triplet_range["common_member_count"]),
        "triplet_range_2d_per_frame": triplet_range["range_2d_per_frame"],
        "triplet_range_3d_per_frame": triplet_range["range_3d_per_frame"],
        "support_status": str(support["support_status"]),
        "indirect_connectivity_examples": indirect,
        "component_members": [list(component) for component in components],
        "graph_edges": [list(edge) for edge in sorted(edges)],
        "support": support,
    }


def tracking_validity_rows(sequence: Any, support: Mapping[str, Any], *, density_label: str) -> list[dict[str, Any]]:
    """Emit explicit, non-mutually-exclusive observation state fields."""

    uv = np.asarray(sequence.uv, dtype=np.float64)
    visible = np.asarray(sequence.visibility, dtype=bool)
    geometry = np.asarray(sequence.geometry_validity, dtype=bool)
    frame_sizes = np.asarray(sequence.frame_sizes_hw, dtype=np.int64)
    frame_indices = np.asarray(sequence.frame_indices, dtype=np.int64)
    timestamps = np.asarray(sequence.timestamps_s, dtype=np.float64)
    track_ids = np.asarray(sequence.track_ids, dtype=np.int64)
    component_members = {int(x) for row in support.get("components", []) for x in row["member_indices"]}
    selected_members = {
        int(x) for triplet in support.get("triplets", []) for x in triplet.get("common_member_indices", [])
    }
    rows: list[dict[str, Any]] = []
    for frame in range(uv.shape[0]):
        height, width = (int(frame_sizes[frame, 0]), int(frame_sizes[frame, 1]))
        finite = np.all(np.isfinite(uv[frame]), axis=1)
        inside = finite & (uv[frame, :, 0] >= 0) & (uv[frame, :, 0] < width) & (uv[frame, :, 1] >= 0) & (uv[frame, :, 1] < height)
        for slot in range(uv.shape[1]):
            not_visible = not bool(visible[frame, slot])
            uv_bad = not bool(inside[slot])
            geometry_bad = not bool(geometry[frame, slot])
            in_component = slot in component_members
            used = slot in selected_members
            unknown_reasons: list[str] = []
            if not_visible and uv_bad:
                unknown_reasons.append("visibility-masked UV cannot separate occlusion from tracker failure")
            if geometry_bad:
                unknown_reasons.append("geometry invalid source does not expose depth-vs-pose cause")
            rows.append(
                {
                    "density": density_label,
                    "array_index": frame,
                    "source_frame_index": int(frame_indices[frame]),
                    "timestamp_s": float(timestamps[frame]),
                    "track_slot": slot,
                    "track_id": int(track_ids[slot]),
                    "track_not_visible": not_visible,
                    "uv_nonfinite_or_out_of_bounds": uv_bad,
                    "depth_invalid": "UNKNOWN" if geometry_bad else False,
                    "other_geometry_invalid": "UNKNOWN" if geometry_bad else False,
                    "visible_valid_not_in_history_component": bool(visible[frame, slot] and geometry[frame, slot] and not in_component),
                    "in_component_not_in_selected_triplet_common_support": bool(in_component and not used),
                    "used_in_triplet": used,
                    "unknown_reason": "; ".join(unknown_reasons),
                }
            )
    return rows


def compare_sequences(old: Any, new: Any) -> dict[str, Any]:
    """Compare the historical 64-point artifact with the matched rerun."""

    old_uv = np.asarray(old.uv, dtype=np.float64)
    new_uv = np.asarray(new.uv, dtype=np.float64)
    old_vis = np.asarray(old.visibility, dtype=bool)
    new_vis = np.asarray(new.visibility, dtype=bool)
    old_xyz = np.asarray(old.xyz, dtype=np.float64)
    new_xyz = np.asarray(new.xyz, dtype=np.float64)
    same_frames = np.array_equal(old.frame_indices, new.frame_indices)
    timestamp_error = float(np.max(np.abs(old.timestamps_s - new.timestamps_s))) if old.timestamps_s.size else 0.0
    same_sizes = np.array_equal(old.frame_sizes_hw, new.frame_sizes_hw)
    vis_equal = old_vis == new_vis
    both_uv = np.all(np.isfinite(old_uv), axis=-1) & np.all(np.isfinite(new_uv), axis=-1)
    uv_delta = np.linalg.norm(old_uv - new_uv, axis=-1)[both_uv]
    both_xyz = np.all(np.isfinite(old_xyz), axis=-1) & np.all(np.isfinite(new_xyz), axis=-1)
    xyz_delta = np.linalg.norm(old_xyz - new_xyz, axis=-1)[both_xyz]
    disagreement_frames = int(np.sum(np.any(~vis_equal, axis=1)))
    summary = {
        "source_frame_indices_equal": same_frames,
        "max_timestamp_abs_error_s": timestamp_error,
        "frame_sizes_equal": same_sizes,
        "visibility_equal_fraction": float(np.mean(vis_equal)),
        "visibility_disagreement_count": int(np.sum(~vis_equal)),
        "visibility_disagreement_frame_count": disagreement_frames,
        "uv_common_finite_count": int(uv_delta.size),
        "uv_abs_error_median_px": float(np.median(uv_delta)) if uv_delta.size else None,
        "uv_abs_error_p95_px": float(np.percentile(uv_delta, 95)) if uv_delta.size else None,
        "xyz_common_finite_count": int(xyz_delta.size),
        "xyz_abs_error_median_m": float(np.median(xyz_delta)) if xyz_delta.size else None,
        "xyz_abs_error_p95_m": float(np.percentile(xyz_delta, 95)) if xyz_delta.size else None,
    }
    return summary


def density_summary(sequence: Any, support: Mapping[str, Any], *, density_label: str) -> dict[str, Any]:
    """Return compact observation counts for an already-built support object.

    Component reconstruction is deliberately kept in :func:`component_diagnostic`;
    this helper must not invent a window start or silently rebuild a graph with a
    different history boundary.
    """
    geometry = np.asarray(sequence.geometry_validity, dtype=bool)
    visibility = np.asarray(sequence.visibility, dtype=bool)
    return {
        "density": density_label,
        "requested_tracks": int(geometry.shape[1]),
        "visible_points_mean": float(np.mean(np.sum(visibility, axis=1))),
        "geometry_valid_points_mean": float(np.mean(np.sum(geometry, axis=1))),
        "visible_points_min": int(np.min(np.sum(visibility, axis=1))),
        "geometry_valid_points_min": int(np.min(np.sum(geometry, axis=1))),
        "valid_triplet_count": int(len(support.get("triplets", []))),
        "component_count": int(len(support.get("components", []))),
        "support_status": str(support.get("support_status")),
    }


def review_frame_data(
    sequence: Any,
    support: Mapping[str, Any],
    diagnostic: Mapping[str, Any],
    *,
    nested_dense: bool = False,
    base_grid_size: int = 8,
    dense_grid_size: int = 17,
) -> dict[str, Any]:
    """Create small JSON data for the density review page."""

    component_by_slot: dict[int, int] = {}
    for component_index, component in enumerate(diagnostic.get("component_members", [])):
        for slot in component:
            component_by_slot[int(slot)] = component_index
    common = {int(x) for triplet in support.get("triplets", []) for x in triplet.get("common_member_indices", [])}
    edges = diagnostic.get("graph_edges", [])
    frame_sizes = np.asarray(sequence.frame_sizes_hw, dtype=np.int64)
    uv = np.asarray(sequence.uv, dtype=np.float64)
    visible = np.asarray(sequence.visibility, dtype=bool)
    geometry = np.asarray(sequence.geometry_validity, dtype=bool)
    frames: list[dict[str, Any]] = []
    for index in range(uv.shape[0]):
        points = []
        for slot in range(uv.shape[1]):
            if not np.all(np.isfinite(uv[index, slot])):
                continue
            row, col = divmod(slot, dense_grid_size if nested_dense else base_grid_size)
            is_nested_base = (not nested_dense) or (row % 2 == 1 and col % 2 == 1)
            points.append(
                [
                    int(slot),
                    round(float(uv[index, slot, 0]), 3),
                    round(float(uv[index, slot, 1]), 3),
                    bool(visible[index, slot]),
                    bool(geometry[index, slot]),
                    int(component_by_slot.get(slot, -1)),
                    bool(slot in common),
                    bool(is_nested_base),
                ]
            )
        frames.append(
            {
                "array_index": index,
                "source_frame_index": int(sequence.frame_indices[index]),
                "timestamp_s": float(sequence.timestamps_s[index]),
                "height": int(frame_sizes[index, 0]),
                "width": int(frame_sizes[index, 1]),
                "points": points,
            }
        )
    return {
        "track_count": int(uv.shape[1]),
        "frames": frames,
        "common_member_slots": sorted(common),
        "graph_edges_display": edges[:200],
        "graph_edges_total": len(edges),
        "component_internal_pair_count": diagnostic.get("component_internal_pair_count", 0),
        "valid_triplet_count": diagnostic.get("valid_triplet_count", 0),
        # Keep the actual saved support identity beside the display points.
        # The ROI diagnostic uses these records to count unique unordered
        # pairs; it must not infer relationships from the rendered lines.
        "triplets": [
            {
                "triplet_id": int(triplet.get("triplet_id", index)),
                "component_index": int(triplet.get("component_index", -1)),
                "source_frame_indices": [int(value) for value in triplet.get("frame_indices", [])],
                "timestamps_s": [float(value) for value in triplet.get("timestamps_s", [])],
                "common_member_indices": [int(value) for value in triplet.get("common_member_indices", [])],
                "pair_indices": [[int(pair[0]), int(pair[1])] for pair in triplet.get("pair_indices", [])],
            }
            for index, triplet in enumerate(support.get("triplets", []))
        ],
        "components": [
            {
                "component_index": int(index),
                "member_indices": [int(value) for value in members],
            }
            for index, members in enumerate(diagnostic.get("component_members", []))
        ],
    }
