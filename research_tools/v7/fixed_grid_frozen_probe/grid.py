"""Pure fixed-grid and annotation helpers.

The functions in this module deliberately do not inspect labels when creating
the grid.  Labels are applied only after the complete grid has been frozen.
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping, Sequence


def _union(intervals: Iterable[tuple[float, float]]) -> list[tuple[float, float]]:
    values = sorted((float(a), float(b)) for a, b in intervals if b > a)
    merged: list[tuple[float, float]] = []
    for start, end in values:
        if not merged or start > merged[-1][1]:
            merged.append((start, end))
        else:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
    return merged


def interval_union_overlap(start: float, end: float, intervals: Sequence[Mapping[str, Any]]) -> float:
    """Return overlap seconds with the union of half-open annotation intervals."""

    if end <= start:
        return 0.0
    total = 0.0
    for left, right in _union((float(item["start_s"]), float(item["end_s"])) for item in intervals):
        total += max(0.0, min(end, right) - max(start, left))
    return float(total)


def generate_grid_windows(
    timestamps_s: Sequence[float],
    *,
    window_length_s: float = 1.0,
    stride_s: float = 0.5,
) -> list[dict[str, Any]]:
    """Generate label-blind windows from an actual presentation-time axis.

    ``timestamps_s`` are source PTS values.  Returned starts/ends are relative
    to the first PTS, while frame indices and PTS values remain source values.
    A final tail window is added only when it is not already represented.
    """

    pts = [float(value) for value in timestamps_s]
    if not pts or any(not (value == value) for value in pts):
        return []
    if any(right <= left for left, right in zip(pts, pts[1:])):
        raise ValueError("timestamps must be strictly increasing")
    length = float(window_length_s)
    stride = float(stride_s)
    if length <= 0 or stride <= 0:
        raise ValueError("window length and stride must be positive")
    origin = pts[0]
    duration = max(0.0, pts[-1] - origin)
    if duration < length:
        return [{"status": "SHORT_VIDEO", "relative_duration_s": duration, "timestamp_origin_s": origin}]
    starts: list[float] = []
    value = 0.0
    epsilon = 1e-9
    while value + length <= duration + epsilon:
        starts.append(round(value, 9))
        value += stride
    tail = max(0.0, duration - length)
    if not starts or abs(starts[-1] - tail) > epsilon:
        starts.append(round(tail, 9))
    rows: list[dict[str, Any]] = []
    for number, start in enumerate(starts):
        end = min(duration, start + length)
        absolute_start = origin + start
        absolute_end = origin + end
        indices = [index for index, stamp in enumerate(pts) if absolute_start <= stamp < absolute_end or (number == len(starts) - 1 and stamp <= absolute_end)]
        rows.append({
            "grid_index": number,
            "nominal_start_s": float(start),
            "nominal_end_s": float(end),
            "actual_start_pts_s": float(pts[indices[0]]) if indices else None,
            "actual_end_pts_s": float(pts[indices[-1]]) if indices else None,
            "frame_indices": indices,
            "frame_pts_s": [pts[index] for index in indices],
            "tail_window": bool(abs(start - tail) <= epsilon and not abs(start - round(start / stride) * stride) <= epsilon),
            "status": "PLANNED" if indices else "NO_FRAMES",
            "timestamp_origin_s": origin,
            "relative_duration_s": duration,
        })
    return rows


def classify_fake_window(
    start_pts_s: float,
    end_pts_s: float,
    intervals: Sequence[Mapping[str, Any]],
    *,
    origin_pts_s: float = 0.0,
    epsilon_s: float = 1e-6,
) -> dict[str, Any]:
    """Apply annotations after grid creation and report overlap facts."""

    relative_start = float(start_pts_s) - float(origin_pts_s)
    relative_end = float(end_pts_s) - float(origin_pts_s)
    overlap = interval_union_overlap(relative_start, relative_end, intervals)
    length = max(0.0, relative_end - relative_start)
    fraction = overlap / length if length else 0.0
    if fraction >= 1.0 - epsilon_s:
        category = "FAKE_MANIPULATION"
    elif overlap > epsilon_s:
        category = "BOUNDARY_MIXED"
    else:
        category = "OUTSIDE_ANNOTATED_MANIPULATION"
    return {
        "annotation_category": category,
        "annotation_overlap_s": overlap,
        "annotation_overlap_fraction": fraction,
        "relative_start_s": relative_start,
        "relative_end_s": relative_end,
    }


def map_target_frames_to_intervals(
    frame_pts_s: Sequence[float],
    intervals: Sequence[Mapping[str, Any]],
    *,
    origin_pts_s: float = 0.0,
) -> dict[str, Any]:
    """Count model-use timestamps lying in annotated intervals."""

    union = _union((float(item["start_s"]), float(item["end_s"])) for item in intervals)
    hits = [float(stamp) for stamp in frame_pts_s if any(left <= float(stamp) - origin_pts_s < right for left, right in union)]
    return {"target_frame_count": len(frame_pts_s), "target_frames_in_annotation_count": len(hits), "target_pts_in_annotation_s": hits}
