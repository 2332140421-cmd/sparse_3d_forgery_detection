"""Small, source-frame ROI bookkeeping for the observation-density diagnostic.

This module is deliberately an offline reader of the saved 64/289 review
records.  Its immediate caller is ``run_diagnostic.build_review`` and its
output is a development measurement, not a spatial ground-truth or detector
metric.  Keeping the rectangle and pair counting here avoids putting a
browser-only interpretation into the frozen component code.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any


ROI_SCHEMA_VERSION = "v1"
VALID = "VALID"
UNRESOLVED = "UNRESOLVED"
OUTSIDE_DENSITY_POPULATION = "OUTSIDE_DENSITY_POPULATION"
TEMPORAL_ROI_PENDING = "TEMPORAL_ROI_PENDING"
NOT_MODEL_SAMPLE = "NOT_MODEL_SAMPLE"


def display_to_source(
    x: float,
    y: float,
    display_width: float,
    display_height: float,
    source_width: int,
    source_height: int,
) -> tuple[float, float]:
    """Map a displayed canvas coordinate back to original image pixels."""

    if display_width <= 0 or display_height <= 0:
        raise ValueError("display dimensions must be positive")
    if source_width <= 0 or source_height <= 0:
        raise ValueError("source dimensions must be positive")
    return float(x) * source_width / float(display_width), float(y) * source_height / float(display_height)


def source_to_display(
    x: float,
    y: float,
    source_width: int,
    source_height: int,
    display_width: float,
    display_height: float,
) -> tuple[float, float]:
    """Map original image pixels to a displayed canvas coordinate."""

    if source_width <= 0 or source_height <= 0:
        raise ValueError("source dimensions must be positive")
    if display_width <= 0 or display_height <= 0:
        raise ValueError("display dimensions must be positive")
    return float(x) * display_width / float(source_width), float(y) * display_height / float(source_height)


def rect_is_valid(rect: Mapping[str, Any], width: int, height: int) -> bool:
    """Return whether an inclusive-boundary rectangle is non-empty and in-frame."""

    try:
        x, y, w, h = (float(rect[key]) for key in ("x", "y", "w", "h"))
    except (KeyError, TypeError, ValueError):
        return False
    return (
        width > 0
        and height > 0
        and w > 0
        and h > 0
        and x >= 0
        and y >= 0
        and x + w <= width
        and y + h <= height
    )


def point_in_rect(u: float, v: float, rect: Mapping[str, Any]) -> bool:
    """Use inclusive rectangle boundaries so edge points are not discarded."""

    return (
        float(u) >= float(rect["x"])
        and float(u) <= float(rect["x"]) + float(rect["w"])
        and float(v) >= float(rect["y"])
        and float(v) <= float(rect["y"]) + float(rect["h"])
    )


def _pair_key(pair: Sequence[Any]) -> tuple[int, int]:
    left, right = sorted((int(pair[0]), int(pair[1])))
    if left == right:
        raise ValueError("a pair cannot contain the same track twice")
    return left, right


def _frame_for_source_index(detail: Mapping[str, Any], source_frame_index: int) -> Mapping[str, Any] | None:
    for frame in detail.get("frames", []):
        if int(frame.get("source_frame_index", -1)) == int(source_frame_index):
            return frame
    return None


def _point_map(frame: Mapping[str, Any]) -> dict[int, Sequence[Any]]:
    return {int(point[0]): point for point in frame.get("points", [])}


def _inside(point: Sequence[Any] | None, rect: Mapping[str, Any]) -> bool:
    return bool(point is not None and len(point) >= 4 and point_in_rect(float(point[1]), float(point[2]), rect))


def validate_annotation(
    annotation: Mapping[str, Any],
    details_by_window: Mapping[str, Mapping[str, Any]],
    manifest_by_window: Mapping[str, Mapping[str, Any]],
    *,
    timestamp_tolerance_s: float = 1e-4,
) -> dict[str, Any]:
    """Validate one human ROI against the saved source-frame contract."""

    row = dict(annotation)
    window_id = str(row.get("window_id", ""))
    detail = details_by_window.get(window_id)
    manifest = manifest_by_window.get(window_id)
    if detail is None or manifest is None:
        return {"status": OUTSIDE_DENSITY_POPULATION, "window_id": window_id, "reason": "window_id not in frozen density manifest"}
    expected_source = str(manifest.get("source_id", ""))
    if str(row.get("source_id", "")) != expected_source:
        return {"status": UNRESOLVED, "window_id": window_id, "reason": "source_id does not match manifest"}
    role = str(row.get("role", ""))
    if role != str(manifest.get("role", "")):
        return {"status": UNRESOLVED, "window_id": window_id, "reason": "role does not match manifest"}
    try:
        source_frame_index = int(row["source_frame_index"])
        timestamp_s = float(row["timestamp_s"])
    except (KeyError, TypeError, ValueError):
        return {"status": UNRESOLVED, "window_id": window_id, "reason": "source frame and timestamp are required"}
    frame = _frame_for_source_index(detail, source_frame_index)
    if frame is None:
        return {"status": UNRESOLVED, "window_id": window_id, "reason": "source frame is not in saved sequence"}
    frame_timestamp = float(frame["timestamp_s"])
    if abs(frame_timestamp - timestamp_s) > timestamp_tolerance_s:
        return {"status": UNRESOLVED, "window_id": window_id, "reason": "timestamp does not match saved PTS"}
    frame_size = row.get("frame_size_hw")
    expected_size = [int(frame["height"]), int(frame["width"])]
    if frame_size is not None and [int(value) for value in frame_size] != expected_size:
        return {"status": UNRESOLVED, "window_id": window_id, "reason": "frame_size_hw does not match source frame"}
    rect = row.get("rect", row.get("region"))
    if not isinstance(rect, Mapping) or not rect_is_valid(rect, expected_size[1], expected_size[0]):
        return {"status": UNRESOLVED, "window_id": window_id, "reason": "ROI rectangle is empty or outside the source image"}
    return {
        "status": VALID,
        "window_id": window_id,
        "source_id": expected_source,
        "role": role,
        "source_frame_index": source_frame_index,
        "timestamp_s": timestamp_s,
        "frame_size_hw": expected_size,
        "rect": {key: float(rect[key]) for key in ("x", "y", "w", "h")},
        "kind": str(manifest.get("kind", "")),
        "observation": str(row.get("observation", row.get("visible_distortion", ""))),
        "tracking_observation": str(row.get("tracking_observation", "")),
        "notes": str(row.get("notes", "")),
    }


def _triplets_for_frame(detail: Mapping[str, Any], source_frame_index: int) -> list[Mapping[str, Any]]:
    return [
        triplet
        for triplet in detail.get("triplets", [])
        if int(source_frame_index) in {int(value) for value in triplet.get("source_frame_indices", [])}
    ]


def coverage_for_roi(detail: Mapping[str, Any], annotation: Mapping[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Count actual points and unordered pairs for one exact source frame."""

    source_frame_index = int(annotation["source_frame_index"])
    frame = _frame_for_source_index(detail, source_frame_index)
    if frame is None:
        raise ValueError("annotation frame is absent from detail")
    rect = annotation["rect"]
    points = _point_map(frame)
    all_inside = {slot for slot, point in points.items() if _inside(point, rect)}
    visible_slots = {slot for slot in all_inside if bool(points[slot][3])}
    geometry_slots = {slot for slot in all_inside if bool(points[slot][4])}
    component_slots = {slot for slot in geometry_slots if int(points[slot][5]) >= 0}
    triplets = _triplets_for_frame(detail, source_frame_index)
    common_slots: set[int] = set()
    internal_pairs: set[tuple[int, int]] = set()
    boundary_pairs: set[tuple[int, int]] = set()
    outside_pairs: set[tuple[int, int]] = set()
    component_ids: set[int] = set()
    triplet_ids: set[int] = set()
    component_rows: list[dict[str, Any]] = []
    for triplet in triplets:
        triplet_id = int(triplet["triplet_id"])
        component_id = int(triplet["component_index"])
        triplet_ids.add(triplet_id)
        component_ids.add(component_id)
        members = {int(value) for value in triplet.get("common_member_indices", [])}
        in_roi_members = members.intersection(geometry_slots).intersection(all_inside)
        common_slots.update(in_roi_members)
        triplet_internal: set[tuple[int, int]] = set()
        triplet_boundary: set[tuple[int, int]] = set()
        triplet_outside: set[tuple[int, int]] = set()
        for pair in triplet.get("pair_indices", []):
            key = _pair_key(pair)
            left, right = key
            left_inside = left in all_inside and left in members
            right_inside = right in all_inside and right in members
            if left_inside and right_inside:
                triplet_internal.add(key)
                internal_pairs.add(key)
            elif left_inside or right_inside:
                triplet_boundary.add(key)
                boundary_pairs.add(key)
            else:
                triplet_outside.add(key)
                outside_pairs.add(key)
        total_pairs = len({_pair_key(pair) for pair in triplet.get("pair_indices", [])})
        component_rows.append(
            {
                "component_index": component_id,
                "triplet_id": triplet_id,
                "source_frame_index": source_frame_index,
                "triplet_common_members_total": len(members),
                "roi_common_members": len(in_roi_members),
                "actual_internal_pairs_total": total_pairs,
                "roi_internal_pairs": len(triplet_internal),
                "cross_roi_pairs": len(triplet_boundary),
                "outside_pairs": len(triplet_outside),
                "roi_internal_pair_fraction": (len(triplet_internal) / total_pairs) if total_pairs else None,
            }
        )
    model_sample = bool(triplets)
    frame_row = {
        "window_id": str(detail.get("window_id", annotation.get("window_id", ""))),
        "source_id": str(annotation.get("source_id", "")),
        "role": str(annotation.get("role", "")),
        "source_frame_index": source_frame_index,
        "timestamp_s": float(frame["timestamp_s"]),
        "density": str(detail.get("density", "")),
        "roi_x": float(rect["x"]),
        "roi_y": float(rect["y"]),
        "roi_w": float(rect["w"]),
        "roi_h": float(rect["h"]),
        "visible_query_points": len(visible_slots),
        "geometry_valid_points": len(geometry_slots),
        "valid_history_component_points": len(component_slots),
        "valid_triplet_common_points": len(common_slots),
        "internal_pairs_both_endpoints": len(internal_pairs),
        "cross_roi_pairs_one_endpoint": len(boundary_pairs),
        "outside_pairs_both_endpoints": len(outside_pairs),
        "component_ids": sorted(component_ids),
        "triplet_ids": sorted(triplet_ids),
        "model_sample_status": "MODEL_SAMPLE" if model_sample else NOT_MODEL_SAMPLE,
    }
    return frame_row, component_rows


def compare_density_details(
    detail64: Mapping[str, Any],
    detail289: Mapping[str, Any],
    annotation: Mapping[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    """Compare two densities using one exact source frame and one ROI."""

    if detail64.get("window_id") != detail289.get("window_id"):
        raise ValueError("density details must refer to the same window")
    row64, components64 = coverage_for_roi(detail64, annotation)
    row289, components289 = coverage_for_roi(detail289, annotation)
    timestamp_difference = abs(float(row64["timestamp_s"]) - float(row289["timestamp_s"]))
    if timestamp_difference > 1e-6:
        raise ValueError("64/289 source-frame PTS values do not match")
    frame64 = _frame_for_source_index(detail64, int(annotation["source_frame_index"]))
    frame289 = _frame_for_source_index(detail289, int(annotation["source_frame_index"]))
    if [int(frame64["height"]), int(frame64["width"])] != [int(frame289["height"]), int(frame289["width"])]:
        raise ValueError("64/289 source-frame dimensions do not match")
    comparison = {
        "window_id": str(detail64["window_id"]),
        "source_id": str(annotation.get("source_id", "")),
        "role": str(annotation.get("role", "")),
        "source_frame_index": int(annotation["source_frame_index"]),
        "timestamp_s": float(annotation["timestamp_s"]),
        "same_source_frame": True,
        "timestamp_abs_difference_s": timestamp_difference,
        "density64_visible_query_points": row64["visible_query_points"],
        "density289_visible_query_points": row289["visible_query_points"],
        "density64_geometry_valid_points": row64["geometry_valid_points"],
        "density289_geometry_valid_points": row289["geometry_valid_points"],
        "density64_valid_triplet_common_points": row64["valid_triplet_common_points"],
        "density289_valid_triplet_common_points": row289["valid_triplet_common_points"],
        "density64_internal_pairs_both_endpoints": row64["internal_pairs_both_endpoints"],
        "density289_internal_pairs_both_endpoints": row289["internal_pairs_both_endpoints"],
        "density64_cross_roi_pairs_one_endpoint": row64["cross_roi_pairs_one_endpoint"],
        "density289_cross_roi_pairs_one_endpoint": row289["cross_roi_pairs_one_endpoint"],
        "support_status": (
            "LOCAL_SUPPORT_INCREASED"
            if row289["valid_triplet_common_points"] > row64["valid_triplet_common_points"]
            or row289["internal_pairs_both_endpoints"] > row64["internal_pairs_both_endpoints"]
            else "LOCAL_SUPPORT_STILL_INSUFFICIENT"
        ),
        "interpretation": "support counts only; not accuracy, anomaly score, or spatial ground truth",
    }
    return comparison, [row64, row289], components64 + components289


def validate_annotations(
    annotations: Iterable[Mapping[str, Any]],
    details_by_window: Mapping[str, Mapping[str, Any]],
    manifest_by_window: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    return [validate_annotation(item, details_by_window, manifest_by_window) for item in annotations]
