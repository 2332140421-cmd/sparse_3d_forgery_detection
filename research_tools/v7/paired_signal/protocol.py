"""Frozen paired-window protocol and baseline structural measurements.

This module deliberately contains no detector or fitted normality model.  It
only turns the reviewed, exact real/fake population into timestamp-matched
windows and extracts the existing V7 S/Delta-S/Delta2-S representations.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import av
import numpy as np

from sparse3d_forgery.experiments.v7_dynamic_structure_probe import (
    ComponentConfig,
    motion_coherent_components,
    structural_differences,
    structure_state,
)


TIMESCALE_S = 1.0
ANCHOR_FRACTIONS = (0.25, 0.50, 0.75)
COMPONENT_CONFIG = ComponentConfig(
    max_initial_distance=1.0,
    max_relative_change=0.05,
    minimum_overlap=8,
    minimum_size=3,
)
ORDER_NAMES = ("K0_S", "K1_delta_s", "K2_delta2_s")


def assert_original_video_path(path: str | Path) -> Path:
    """Reject preview-derived paths before any formal experiment reads media."""

    value = Path(path)
    lowered = str(value).lower()
    if "preview" in lowered or "preview_fake_h264" in lowered:
        raise ValueError(f"review-only preview is forbidden in formal input: {value}")
    if not value.is_file() or value.stat().st_size <= 0:
        raise FileNotFoundError(value)
    return value


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _finite_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def read_timeline(path: str | Path) -> dict[str, Any]:
    """Read frame PTS in source order without materializing RGB frames."""

    source = assert_original_video_path(path)
    timestamps: list[float] = []
    width = height = None
    codec_name = None
    with av.open(str(source)) as container:
        stream = next((item for item in container.streams if item.type == "video"), None)
        if stream is None:
            raise ValueError(f"no video stream: {source}")
        width, height = int(stream.width), int(stream.height)
        codec_name = str(stream.codec_context.name)
        for frame in container.decode(stream):
            if frame.pts is None or frame.time_base is None:
                raise ValueError(f"missing frame PTS: {source}")
            timestamp = float(frame.pts * frame.time_base)
            if not math.isfinite(timestamp):
                raise ValueError(f"non-finite frame PTS: {source}")
            if timestamps and timestamp <= timestamps[-1]:
                raise ValueError(f"non-increasing frame PTS: {source}")
            timestamps.append(timestamp)
    values = np.asarray(timestamps, dtype=np.float64)
    if values.size < 2:
        raise ValueError(f"not enough frames: {source}")
    periods = np.diff(values)
    return {
        "path": str(source),
        "sha256": file_sha256(source),
        "frame_indices": list(range(int(values.size))),
        "timestamps_s": values.tolist(),
        "frame_count": int(values.size),
        "start_s": float(values[0]),
        "end_s": float(values[-1]),
        "duration_s": float(values[-1] - values[0]),
        "median_frame_period_s": float(np.median(periods)),
        "width": width,
        "height": height,
        "codec_name": codec_name,
    }


def primary_segment(segments: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Choose the longest official segment, then the earliest on ties."""

    normalized = [
        {**dict(segment), "start_s": float(segment["start_s"]), "end_s": float(segment["end_s"])}
        for segment in segments
    ]
    if not normalized:
        raise ValueError("at least one official manipulation segment is required")
    return dict(sorted(normalized, key=lambda item: (-(item["end_s"] - item["start_s"]), item["start_s"]))[0])


def unmodified_blocks(start_s: float, end_s: float, segments: Sequence[Mapping[str, Any]]) -> list[tuple[float, float]]:
    """Return contiguous timeline gaps outside every official segment."""

    intervals = sorted((max(start_s, float(x["start_s"])), min(end_s, float(x["end_s"]))) for x in segments)
    intervals = [(a, b) for a, b in intervals if b > start_s and a < end_s and b > a]
    blocks: list[tuple[float, float]] = []
    cursor = start_s
    for left, right in intervals:
        if left > cursor:
            blocks.append((cursor, left))
        cursor = max(cursor, right)
    if cursor < end_s:
        blocks.append((cursor, end_s))
    return [block for block in blocks if block[1] - block[0] >= TIMESCALE_S]


def choose_control_block(
    common_start_s: float,
    common_end_s: float,
    primary: Mapping[str, Any],
    segments: Sequence[Mapping[str, Any]],
) -> tuple[float, float] | None:
    candidates = unmodified_blocks(common_start_s, common_end_s, segments)
    if not candidates:
        return None
    p_start, p_end = float(primary["start_s"]), float(primary["end_s"])
    return min(
        candidates,
        key=lambda block: (
            max(p_start - block[1], block[0] - p_end, 0.0),
            block[0] > p_start,
            block[0],
        ),
    )


def anchor_interval(block: tuple[float, float], fraction: float) -> tuple[float, float, float]:
    left, right = block
    if right - left < TIMESCALE_S:
        raise ValueError("block cannot contain a one-second window")
    center = left + TIMESCALE_S / 2.0 + fraction * (right - left - TIMESCALE_S)
    return center - TIMESCALE_S / 2.0, center + TIMESCALE_S / 2.0, center


def indices_for_interval(timeline: Mapping[str, Any], interval: tuple[float, float]) -> tuple[list[int], list[float]]:
    timestamps = np.asarray(timeline["timestamps_s"], dtype=np.float64)
    left, right = interval
    indices = np.flatnonzero((timestamps >= left) & (timestamps <= right)).astype(int).tolist()
    if len(indices) < 2:
        raise ValueError(f"one-second interval has fewer than two source frames: {interval}")
    return indices, timestamps[indices].tolist()


def _mapping_for_row(row: Mapping[str, Any], mapping_by_path: Mapping[str, Mapping[str, Any]]) -> Mapping[str, Any]:
    fake = str(row["fake_video_path"])
    marker = "/source/activityforensics/raw/"
    if marker not in fake:
        raise ValueError(f"unexpected ActivityForensics path: {fake}")
    relative = fake.split(marker, 1)[1]
    mapping = mapping_by_path.get(relative)
    if mapping is None or mapping.get("lineage_status") != "EXACT":
        raise ValueError(f"EXACT mapping missing for {fake}")
    return mapping


def build_population(review_rows: Sequence[Mapping[str, Any]], mapping_rows: Sequence[Mapping[str, Any]], limit: int = 16) -> dict[str, Any]:
    """Select the first technically eligible frozen pairs in manifest order."""

    mapping_by_path = {str(item["activityforensics_file"]): item for item in mapping_rows}
    selected: list[dict[str, Any]] = []
    exclusions: list[dict[str, Any]] = []
    for row in review_rows:
        if len(selected) >= limit:
            break
        pair_id = str(row["review_id"])
        reason: str | None = None
        try:
            real_path = assert_original_video_path(row["real_video_path"])
            fake_path = assert_original_video_path(row["fake_video_path"])
            mapping = _mapping_for_row(row, mapping_by_path)
            if row.get("lineage_status") != "EXACT":
                reason = "LINEAGE_NOT_EXACT"
            elif row.get("media_status_real") != "MEDIA_VALID" or row.get("media_status_fake") != "MEDIA_VALID":
                reason = "MEDIA_NOT_VALID"
            else:
                real_timeline = read_timeline(real_path)
                fake_timeline = read_timeline(fake_path)
                tolerance = max(
                    0.10,
                    2.0 * max(real_timeline["median_frame_period_s"], fake_timeline["median_frame_period_s"]),
                )
                if abs(real_timeline["duration_s"] - fake_timeline["duration_s"]) > tolerance:
                    reason = "TIMELINE_ALIGNMENT_UNSAFE"
                segments = list(mapping.get("manipulation_segments", []))
                primary = primary_segment(segments) if segments else None
                if reason is None and (primary is None or primary["end_s"] - primary["start_s"] < TIMESCALE_S):
                    reason = "MANIPULATION_SEGMENT_TOO_SHORT"
                common_start = max(real_timeline["start_s"], fake_timeline["start_s"])
                common_end = min(real_timeline["end_s"], fake_timeline["end_s"])
                control = choose_control_block(common_start, common_end, primary, segments) if reason is None else None
                if reason is None and control is None:
                    reason = "NO_UNMODIFIED_CONTROL_BLOCK"
                windows: list[dict[str, Any]] = []
                if reason is None:
                    manip_block = (primary["start_s"], primary["end_s"])
                    for kind, block in (("MANIP", manip_block), ("CTRL", control)):
                        for fraction in ANCHOR_FRACTIONS:
                            left, right, center = anchor_interval(block, fraction)
                            real_indices, real_times = indices_for_interval(real_timeline, (left, right))
                            fake_indices, fake_times = indices_for_interval(fake_timeline, (left, right))
                            label = f"{kind}_{int(fraction * 100):02d}"
                            windows.append(
                                {
                                    "window_id": f"{pair_id}_{label}",
                                    "pair_id": pair_id,
                                    "source_id": str(row["charades_source_id"]),
                                    "label": label,
                                    "kind": kind,
                                    "anchor_fraction": fraction,
                                    "interval_start_s": left,
                                    "interval_end_s": right,
                                    "center_s": center,
                                    "real": {"video_path": str(real_path), "frame_indices": real_indices, "timestamps_s": real_times},
                                    "fake": {"video_path": str(fake_path), "frame_indices": fake_indices, "timestamps_s": fake_times},
                                }
                            )
        except (OSError, ValueError, RuntimeError) as exc:
            reason = reason or f"TECHNICAL_TIMELINE_FAILURE:{type(exc).__name__}"
        if reason is not None:
            exclusions.append({"pair_id": pair_id, "source_id": row.get("charades_source_id"), "reason": reason})
            continue
        selected.append(
            {
                "pair_id": pair_id,
                "source_id": str(row["charades_source_id"]),
                "generator": mapping["generator"],
                "manipulation_operation": mapping["manipulation_operation"],
                "official_split": mapping["official_split"],
                "lineage_status": mapping["lineage_status"],
                "real_video_path": str(real_path),
                "fake_video_path": str(fake_path),
                "real_sha256": real_timeline["sha256"],
                "fake_sha256": fake_timeline["sha256"],
                "all_manipulation_segments": segments,
                "primary_manipulation_segment": primary,
                "control_block": {"start_s": control[0], "end_s": control[1]},
                "real_timeline": real_timeline,
                "fake_timeline": fake_timeline,
                "windows": windows,
            }
        )
    return {"selected_pairs": selected, "exclusions": exclusions, "requested_limit": limit}


def finite_summary(values: Iterable[float]) -> dict[str, float | int | None]:
    array = np.asarray(list(values), dtype=np.float64)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return {"N": 0, "median": None, "iqr": None, "p10": None, "p90": None}
    return {
        "N": int(array.size),
        "median": float(np.median(array)),
        "iqr": float(np.percentile(array, 75) - np.percentile(array, 25)),
        "p10": float(np.percentile(array, 10)),
        "p90": float(np.percentile(array, 90)),
    }


def robust_scale(real_control_medians: Sequence[Sequence[float]]) -> tuple[np.ndarray, dict[str, Any]]:
    values = np.asarray(real_control_medians, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 4:
        raise ValueError("real-control medians must be [N,4]")
    med = np.median(values, axis=0)
    mad = np.median(np.abs(values - med), axis=0)
    scale = 1.4826 * mad
    fallback_iqr = np.zeros(4, dtype=bool)
    fallback_floor = np.zeros(4, dtype=bool)
    for index in range(4):
        if not np.isfinite(scale[index]) or scale[index] == 0:
            iqr_scale = (np.percentile(values[:, index], 75) - np.percentile(values[:, index], 25)) / 1.349
            if np.isfinite(iqr_scale) and iqr_scale > 0:
                scale[index] = iqr_scale
                fallback_iqr[index] = True
            else:
                scale[index] = max(1e-6, 1e-6 * float(np.median(np.abs(values[:, index]))))
                fallback_floor[index] = True
    return scale, {
        "center_median": med.tolist(),
        "scale": scale.tolist(),
        "mad": mad.tolist(),
        "iqr_fallback_dimensions": np.flatnonzero(fallback_iqr).astype(int).tolist(),
        "floor_fallback_dimensions": np.flatnonzero(fallback_floor).astype(int).tolist(),
        "fallback_count": int(np.sum(fallback_iqr) + np.sum(fallback_floor)),
        "source": "real_control_windows_only",
    }


def window_observation_median(result: Mapping[str, Any], key: str) -> tuple[np.ndarray | None, dict[str, Any]]:
    observations = result.get("observations", {}).get(key, [])
    values = np.asarray([item["values"] for item in observations], dtype=np.float64)
    if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] != 4 or not np.all(np.isfinite(values)):
        return None, {"N": 0, "median": None, "iqr": None}
    return np.median(values, axis=0), {**finite_summary(values.reshape(-1)), "observation_count": int(values.shape[0]), "iqr_vector": (np.percentile(values, 75, axis=0) - np.percentile(values, 25, axis=0)).tolist()}


def pairwise_summary(xyz: np.ndarray, valid: np.ndarray, components: Sequence[Sequence[int]], timestamps_s: Sequence[float]) -> dict[str, Any]:
    """Secondary diagnostic from raw normalized pair distances, not detector input."""

    times = np.asarray(timestamps_s, dtype=np.float64)
    all_distances: list[float] = []
    first: list[float] = []
    second: list[float] = []
    for component in components:
        ids = np.asarray(component, dtype=np.int64)
        for offset, left in enumerate(ids[:-1]):
            for right in ids[offset + 1 :]:
                mask = valid[:, left] & valid[:, right]
                if int(np.sum(mask)) < 3:
                    continue
                distances = np.linalg.norm(xyz[:, right] - xyz[:, left], axis=1).astype(np.float64)
                distances[~mask] = np.nan
                finite = distances[np.isfinite(distances)]
                if finite.size < 3:
                    continue
                scale = float(np.median(finite))
                if not np.isfinite(scale) or scale <= 0:
                    continue
                normalized = finite / scale
                all_distances.extend(normalized.tolist())
                valid_times = times[mask]
                delta = np.diff(normalized) / np.diff(valid_times)
                if delta.size:
                    first.extend(np.abs(delta).tolist())
                if delta.size >= 2:
                    second_delta = np.diff(delta) / np.diff(valid_times[1:])
                    second.extend(np.abs(second_delta).tolist())
    return {
        "pair_count": len(all_distances),
        "distance": finite_summary(all_distances),
        "temporal_mad": float(np.median(np.abs(np.asarray(first) - np.median(first)))) if first else None,
        "temporal_iqr": float(np.percentile(first, 75) - np.percentile(first, 25)) if first else None,
        "first_difference_magnitude": finite_summary(first),
        "second_difference_magnitude": finite_summary(second),
    }


def extract_features(sequence: Any, *, source_id: str, pair_id: str, window_id: str, role: str, kind: str) -> dict[str, Any]:
    """Run the frozen component/state code and retain K0/K1/K2 observations."""

    components = motion_coherent_components(sequence.xyz, sequence.geometry_validity, COMPONENT_CONFIG)
    observations: dict[str, list[dict[str, Any]]] = {key: [] for key in ORDER_NAMES}
    component_rows: list[dict[str, Any]] = []
    valid_s = valid_delta = valid_delta2 = 0
    for component_index, component in enumerate(components):
        state, state_valid = structure_state(sequence.xyz, sequence.geometry_validity, component)
        first, first_valid, second, second_valid = structural_differences(state, state_valid, sequence.timestamps_s)
        valid_s += int(np.sum(state_valid))
        valid_delta += int(np.sum(first_valid))
        valid_delta2 += int(np.sum(second_valid))
        component_rows.append({"component_index": component_index, "members": list(component), "size": len(component), "valid_s_states": int(np.sum(state_valid))})
        for index in np.flatnonzero(state_valid):
            observations["K0_S"].append({"component_index": component_index, "time_index": int(index), "timestamp_s": float(sequence.timestamps_s[index]), "values": state[index].astype(float).tolist()})
        for index in np.flatnonzero(first_valid):
            observations["K1_delta_s"].append({"component_index": component_index, "time_index": int(index), "timestamp_s": float(sequence.timestamps_s[index]), "values": first[index].astype(float).tolist()})
        for index in np.flatnonzero(second_valid):
            observations["K2_delta2_s"].append({"component_index": component_index, "time_index": int(index), "timestamp_s": float(sequence.timestamps_s[index]), "values": second[index].astype(float).tolist()})
    return {
        "source_id": source_id,
        "pair_id": pair_id,
        "window_id": window_id,
        "role": role,
        "kind": kind,
        "component_count": len(components),
        "component_success": bool(components),
        "component_rows": component_rows,
        "valid_s_states": valid_s,
        "valid_delta_s": valid_delta,
        "valid_delta2_s": valid_delta2,
        "s_valid_fraction": float(valid_s / (len(components) * sequence.xyz.shape[0])) if components else 0.0,
        "observations": observations,
        "pairwise": pairwise_summary(sequence.xyz, sequence.geometry_validity, components, sequence.timestamps_s),
    }
