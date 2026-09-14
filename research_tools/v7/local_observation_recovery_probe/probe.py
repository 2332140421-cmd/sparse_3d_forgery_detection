"""Bounded O/R/T diagnostic for the 04LAX fake observation case.

The module intentionally consumes an existing 289-point ParticleSequence for
the O condition.  R is an independent query with the already used
BootsTAPIR implementation, while T is a best-effort official CoTracker3
online check.  No detector score is computed here.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import sys
import time
from types import SimpleNamespace
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from research_tools.v7.local_organization_probe.grouping import (
    build_local_groups,
    build_local_support,
    json_ready_local_support,
    rebuild_components_fast,
    validate_grouping,
)
from research_tools.v7.local_structural_temporal_probe.representation import compute_local_derivatives


CASE_ID = "04LAX_fake_0002_MANIP_25"
SOURCE_ID = "04LAX"
WINDOW_ID = "0002_MANIP_25::fake"
TARGET_TIMES = (16.249583, 16.282950)
TARGET_FRAMES = (487, 488)
DATA_ROOT = Path("/root/autodl-tmp/data/sparse_3d_forgery_detection")
DEFAULT_OUTPUT = DATA_ROOT / "derived/v7_activityforensics_local_observation_recovery_probe_v1"
FAKE_VIDEO = DATA_ROOT / "datasets/v7_core_candidates/activityforensics_charades_v1/source/activityforensics/raw/video/02_wan/04LAX+13.90=22.70=charades@train_delete@04LAX@365@wan.mp4"
REAL_VIDEO = DATA_ROOT / "datasets/v7_core_candidates/activityforensics_charades_v1/source/charades/videos/04LAX.mp4"
PARTICLE = DATA_ROOT / "derived/v7_activityforensics_observation_density_diagnostic_v1/particles/0002_MANIP_25__fake__density289.npz"
PARTICLE_META = PARTICLE.with_suffix(".json")
DETAIL = DATA_ROOT / "derived/v7_activityforensics_boundary_pooling_pilot_v1/review/details/0002_MANIP_25__fake.json"
TAPNET_SOURCE = DATA_ROOT / "external/v7_explicit_geometry/tapnet-c2cbab81cc06092b5f05bfe2da7bfec54e2079c9"
TAPNET_CHECKPOINT = DATA_ROOT / "external/v7_explicit_geometry/checkpoints/causal_bootstapir_checkpoint.pt"
DEPTH_SOURCE = DATA_ROOT / "external/v7_explicit_geometry/ml-depth-pro-9efe5c1def37a26c5367a71df664b18e1306c708"
DEPTH_CHECKPOINT = DATA_ROOT / "external/v7_explicit_geometry/checkpoints/depth_pro.pt"
TRACKER_SHA = "c2cbab81cc06092b5f05bfe2da7bfec54e2079c9"
DEPTH_SHA = "9efe5c1def37a26c5367a71df664b18e1306c708"
COTRACKER_SHA = "82e02e8029753ad4ef13cf06be7f4fc5facdda4d"
COTRACKER_SOURCE = Path("/root/.cache/torch/hub/facebookresearch_co-tracker_main")
COTRACKER_CHECKPOINT = Path("/root/.cache/torch/hub/checkpoints/scaled_online.pth")
COTRACKER_CHECKPOINT_SHA256 = "205d34789f19699d64b22cf93f9b697f15f28d4025240e31532e504109837218"
SEGMENTATION_CACHE_DIR = DATA_ROOT / "derived/v7_activityforensics_boundary_pooling_pilot_v1/masks"


def nearest_frame(frame_indices: Sequence[int], timestamps_s: Sequence[float], target_s: float) -> dict[str, Any]:
    """Map a user time to the nearest saved source frame without rounding FPS."""
    if len(frame_indices) != len(timestamps_s) or len(frame_indices) == 0:
        raise ValueError("frame_indices and timestamps_s must be non-empty and aligned")
    i = min(range(len(frame_indices)), key=lambda j: abs(float(timestamps_s[j]) - float(target_s)))
    return {
        "array_index": int(i),
        "source_frame_index": int(frame_indices[i]),
        "timestamp_s": float(timestamps_s[i]),
        "delta_s": float(timestamps_s[i] - target_s),
    }


def _finite_uv(uv: np.ndarray) -> np.ndarray:
    return np.asarray(np.isfinite(uv).all(axis=-1), dtype=bool)


def layer_counts(
    uv: np.ndarray,
    visibility: np.ndarray,
    geometry_validity: np.ndarray | None = None,
    xyz: np.ndarray | None = None,
    *,
    group_slots: Iterable[int] = (),
    pair_slots: Iterable[Sequence[int]] = (),
) -> dict[str, Any]:
    """Count the actual observation layers for one frame.

    Missing UV/XYZ is never converted to a coordinate.  ``relation_support``
    counts only supplied pairs whose two endpoints are geometry-valid.
    """
    uv = np.asarray(uv)
    visibility = np.asarray(visibility, dtype=bool)
    finite = _finite_uv(uv)
    visible = visibility & finite
    if geometry_validity is None:
        geometry = None
    else:
        geometry = np.asarray(geometry_validity, dtype=bool) & visible
        if xyz is not None:
            geometry &= np.isfinite(np.asarray(xyz)).all(axis=-1)
    group = np.zeros(len(visible), dtype=bool)
    slots = np.asarray(list(group_slots), dtype=int)
    if slots.size:
        group[slots[(slots >= 0) & (slots < len(group))]] = True
    group_valid = int(np.count_nonzero(visible & group))
    relation_support = None
    if geometry is not None:
        relation_support = sum(bool(geometry[a] and geometry[b]) for a, b in pair_slots)
    return {
        "query_count": int(len(visible)),
        "uv_available_count": int(np.count_nonzero(finite)),
        "visibility_count": int(np.count_nonzero(visible)),
        "geometry_valid_count": None if geometry is None else int(np.count_nonzero(geometry)),
        "local_group_count": group_valid,
        "local_group_member_count": group_valid,
        "common_relation_support": relation_support,
        "final_display_count": int(np.count_nonzero(visible)),
        "display_reason": "VISIBLE_FINITE_UV" if np.any(visible) else "NO_VISIBLE_FINITE_UV",
    }


def fixed_members(visibility: np.ndarray, array_index: int) -> np.ndarray:
    """Return IDs visible at the fixed O reference time."""
    v = np.asarray(visibility, dtype=bool)
    if v.ndim != 2 or not 0 <= array_index < v.shape[0]:
        raise ValueError("visibility must be [time, query] and array_index must be valid")
    return np.flatnonzero(v[array_index]).astype(int)


def condition_frame_status(condition: str, source_frame_index: int, *, query_start_frame: int | None = None) -> str:
    """Return an explicit pre-query state instead of freezing or inventing points."""
    if condition == "R" and query_start_frame is not None and int(source_frame_index) < int(query_start_frame):
        return "NOT_QUERIED"
    return "AVAILABLE"


def history_evaluation_status(
    *,
    initialization_frame: int,
    initialization_pts_s: float,
    frame_index: int,
    frame_pts_s: float,
    model_frame_indices: Sequence[int],
    history_window_s: float = 0.5,
) -> dict[str, Any]:
    """Classify a saved frame as query history or model-evaluation input.

    This is a bookkeeping distinction only.  It does not infer that a point
    belongs to a manipulated region, and it never turns a missing observation
    into a valid measurement.
    """
    in_history = float(frame_pts_s) <= float(initialization_pts_s) + float(history_window_s) + 1e-12
    in_model = int(frame_index) in {int(x) for x in model_frame_indices}
    return {
        "initialization_frame": int(initialization_frame),
        "initialization_pts_s": float(initialization_pts_s),
        "frame_index": int(frame_index),
        "frame_pts_s": float(frame_pts_s),
        "relative_to_initialization_s": float(frame_pts_s) - float(initialization_pts_s),
        "within_first_history_window": bool(in_history),
        "model_evaluation_frame": bool(in_model),
        "phase": "MODEL_EVALUATION" if in_model else ("HISTORY" if in_history else "POST_HISTORY"),
    }


def validate_roi_rect(rectangle_xyxy: Sequence[float], width: int, height: int) -> tuple[int, int, int, int]:
    """Validate and clamp a source-pixel ROI; never treat it as a mask/GT."""
    if len(rectangle_xyxy) != 4 or int(width) <= 0 or int(height) <= 0:
        raise ValueError("ROI must contain four coordinates and positive image dimensions")
    x0, y0, x1, y1 = (int(round(float(x))) for x in rectangle_xyxy)
    x0, x1 = max(0, min(x0, int(width) - 1)), max(1, min(x1, int(width)))
    y0, y1 = max(0, min(y0, int(height) - 1)), max(1, min(y1, int(height)))
    if not (x0 < x1 and y0 < y1):
        raise ValueError("ROI must have positive area after clamping")
    return x0, y0, x1, y1


def recent_trace_indices(frame_indices: Sequence[int], current_index: int, limit: int = 5) -> list[int]:
    """Return at most ``limit`` saved frames ending at one array index."""
    if int(limit) <= 0 or not 0 <= int(current_index) < len(frame_indices):
        raise ValueError("current_index must be valid and limit must be positive")
    start = max(0, int(current_index) - int(limit) + 1)
    return list(range(start, int(current_index) + 1))


def _trajectory_entry(frame_indices: np.ndarray, timestamps_s: np.ndarray, uv: np.ndarray, visibility: np.ndarray, geometry_validity: np.ndarray | None) -> dict[str, Any]:
    """Create a small browser-review payload while preserving missing UV explicitly."""
    uv = np.asarray(uv)
    finite = np.isfinite(uv).all(axis=-1)
    safe_uv = [
        [[float(x), float(y)] if finite[t, n] else None for n, (x, y) in enumerate(uv[t])]
        for t in range(uv.shape[0])
    ]
    payload: dict[str, Any] = {
        "frame_indices": np.asarray(frame_indices, dtype=np.int64).astype(int).tolist(),
        "timestamps_s": np.asarray(timestamps_s, dtype=np.float64).astype(float).tolist(),
        "track_ids": list(range(int(uv.shape[1]))),
        "uv": safe_uv,
        "uv_finite": finite.astype(bool).tolist(),
        "visibility": np.asarray(visibility, dtype=bool).tolist(),
    }
    if geometry_validity is not None:
        payload["geometry_validity"] = np.asarray(geometry_validity, dtype=bool).tolist()
    return payload


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, sort_keys=True, allow_nan=False, default=_jsonable)
        handle.write("\n")
    tmp.replace(path)


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _support(detail: Mapping[str, Any]) -> tuple[set[int], list[tuple[int, int]], dict[str, Any]]:
    support = detail.get("support", {}).get("H", {})
    groups = [g for g in support.get("groups", []) if g.get("retained", True)]
    slots = {int(s) for g in groups for s in g.get("member_slots", [])}
    pairs: set[tuple[int, int]] = set()
    for triplet in support.get("triplets", []):
        for pair in triplet.get("pair_member_slots", triplet.get("pair_ids", [])):
            if len(pair) == 2:
                pairs.add(tuple(sorted((int(pair[0]), int(pair[1])))))
    return slots, sorted(pairs), support


def _load_o(output: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    arrays = np.load(PARTICLE, allow_pickle=False)
    detail = _load_json(DETAIL)
    uv = arrays["uv"]
    vis = arrays["visibility"]
    geo = arrays["geometry_validity"]
    xyz = arrays["xyz"]
    group_slots, pairs, support = _support(detail)
    model_frame_indices = [int(x) for x in detail.get("model_frame_indices", [])]
    h_group_count = sum(1 for group in support.get("groups", []) if group.get("retained", True))
    b_support = detail.get("support", {}).get("B", {})
    b_group_count = sum(1 for group in b_support.get("groups", []) if group.get("retained", True))
    ref_index = int(np.argmin(np.abs(arrays["timestamps_s"] - TARGET_TIMES[0])))
    fixed = fixed_members(vis, ref_index)
    rows: list[dict[str, Any]] = []
    for i in range(len(arrays["frame_indices"])):
        c = layer_counts(uv[i], vis[i], geo[i], xyz[i], group_slots=group_slots, pair_slots=pairs)
        fc = layer_counts(uv[i, fixed], vis[i, fixed], geo[i, fixed], xyz[i, fixed], group_slots=range(len(fixed)), pair_slots=[])
        phase = history_evaluation_status(
            initialization_frame=int(arrays["frame_indices"][0]),
            initialization_pts_s=float(arrays["timestamps_s"][0]),
            frame_index=int(arrays["frame_indices"][i]),
            frame_pts_s=float(arrays["timestamps_s"][i]),
            model_frame_indices=model_frame_indices,
        )
        rows.append({
            "condition": "O",
            "array_index": i,
            "source_frame_index": int(arrays["frame_indices"][i]),
            "timestamp_s": float(arrays["timestamps_s"][i]),
            "fixed_query_count": int(len(fixed)),
            "fixed_uv_available_count": fc["uv_available_count"],
            "fixed_visibility_count": fc["visibility_count"],
            "fixed_geometry_valid_count": fc["geometry_valid_count"],
            "fixed_membership": "initial_visibility_at_16.249583s",
            "phase": phase["phase"],
            "model_evaluation_frame": phase["model_evaluation_frame"],
            **c,
            "local_group_definition": "retained H members from saved support",
            "relation_definition": "union of saved H triplet pair members",
        })
    manifest = {
        "case_id": CASE_ID,
        "condition": "O",
        "source_id": SOURCE_ID,
        "role": "fake",
        "window_id": WINDOW_ID,
        "video": {"path": str(FAKE_VIDEO), "status": "AVAILABLE" if FAKE_VIDEO.exists() else "SOURCE_MISSING", "sha256": _sha256(FAKE_VIDEO) if FAKE_VIDEO.exists() else None, "bytes": FAKE_VIDEO.stat().st_size if FAKE_VIDEO.exists() else None},
        "source_frames": arrays["frame_indices"].tolist(),
        "timestamps_s": arrays["timestamps_s"].tolist(),
        "image_size_hw": arrays["frame_sizes_hw"][0].tolist(),
        "query_count": int(len(arrays["track_ids"])),
        "query_initialization": {"source_frame_index": int(arrays["frame_indices"][0]), "timestamp_s": float(arrays["timestamps_s"][0]), "rule": "window first frame"},
        "verification_targets": [nearest_frame(arrays["frame_indices"], arrays["timestamps_s"], t) for t in TARGET_TIMES],
        "history_evaluation": {
            "history_window_s": 0.5,
            "model_frame_indices": model_frame_indices,
            "model_timestamps_s": [float(x) for x in detail.get("model_target_timestamps_s", [])],
            "target_frame_status": [
                history_evaluation_status(
                    initialization_frame=int(arrays["frame_indices"][0]),
                    initialization_pts_s=float(arrays["timestamps_s"][0]),
                    frame_index=int(arrays["frame_indices"][j]),
                    frame_pts_s=float(arrays["timestamps_s"][j]),
                    model_frame_indices=model_frame_indices,
                )
                for j in [ref_index, int(np.argmin(np.abs(arrays["frame_indices"] - TARGET_FRAMES[1])))]
            ],
            "interpretation": "frame 488 is history bookkeeping; it is not a model-evaluation target frame",
        },
        "fixed_reference_members": fixed.tolist(),
        "raw_uv_status": "UNAVAILABLE_CANONICAL_ARTIFACT_MASKS_INVISIBLE_UV_TO_NAN",
        "support": {
            "h_group_count": h_group_count,
            "h_group_slots": sorted(group_slots),
            "h_member_slot_count": len(group_slots),
            "h_pair_count": len(pairs),
            "h_triplet_count": len(support.get("triplets", [])),
            "b_group_count": b_group_count,
            "b_triplet_count": len(b_support.get("triplets", [])),
            "selected_triplet": ({
                "triplet_id": int(support["triplets"][0].get("triplet_id", 0)),
                "common_member_slots": [int(x) for x in support["triplets"][0].get("common_member_slots", [])],
                "pair_member_slots": [[int(x) for x in pair] for pair in support["triplets"][0].get("pair_member_slots", [])],
            } if support.get("triplets") else None),
        },
        "roi": {
            "status": "PENDING_USER_CONFIRMATION",
            "rectangle_xyxy": None,
            "suggested_rectangle_xyxy": [110, 0, 370, 359],
            "suggestion_basis": "粗略覆盖源帧488中人物区域；仅供用户确认/修改，不是空间真值或分割结果",
            "reason": "conversation screenshots are browser captures without a source-pixel calibration file",
        },
        "provenance": _load_json(PARTICLE_META).get("provenance", {}) if PARTICLE_META.exists() else {},
        "artifact": str(PARTICLE),
    }
    return manifest, rows


def _decode_exact(video: Path, frame_indices: Sequence[int]):
    from sparse3d_forgery.video_input import VideoSource, decode_video
    return decode_video(VideoSource(sample_id=CASE_ID, source_video_id=SOURCE_ID, source_locator=video), frame_indices)


def _save_pngs(
    output: Path,
    decoded: Any,
    rows: Sequence[Mapping[str, Any]],
    uv: np.ndarray,
    vis: np.ndarray,
    prefix: str,
    *,
    geometry: np.ndarray | None = None,
    pair_slots: Sequence[Sequence[int]] = (),
) -> list[str]:
    try:
        import cv2
    except ImportError:
        return []
    shot_dir = output / "review" / "screenshots"
    shot_dir.mkdir(parents=True, exist_ok=True)
    names = []
    for i, frame in enumerate(decoded.frames):
        image = np.asarray(frame.rgb)[:, :, ::-1].copy()
        finite = np.isfinite(uv[i]).all(axis=-1) & vis[i]
        if geometry is not None and pair_slots:
            valid_geo = np.asarray(geometry[i], dtype=bool) & finite
            for a, b in pair_slots:
                if not (0 <= int(a) < len(valid_geo) and 0 <= int(b) < len(valid_geo)):
                    continue
                if not (valid_geo[int(a)] and valid_geo[int(b)]):
                    continue
                pa, pb = uv[i, int(a)], uv[i, int(b)]
                cv2.line(image, (int(round(float(pa[0]))), int(round(float(pa[1])))), (int(round(float(pb[0]))), int(round(float(pb[1])))), (255, 180, 40), 1, lineType=cv2.LINE_AA)
        for x, y in uv[i][finite]:
            cv2.circle(image, (int(round(float(x))), int(round(float(y)))), 2, (40, 220, 40), -1, lineType=cv2.LINE_AA)
        name = f"{prefix}_frame_{int(frame.source_frame_index)}.png"
        cv2.imwrite(str(shot_dir / name), image)
        names.append(str(Path("review/screenshots") / name))
    return names


def _save_source_frame(output: Path, decoded: Any, source_frame_index: int) -> str | None:
    """Save one unannotated source frame for pixel-space ROI confirmation."""
    try:
        import cv2
        frame = next(frame for frame in decoded.frames if int(frame.source_frame_index) == int(source_frame_index))
        target = output / "review" / "source_frames" / f"source_frame_{int(source_frame_index)}.png"
        target.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(target), np.asarray(frame.rgb)[:, :, ::-1])
        return str(Path("review/source_frames") / target.name)
    except (ImportError, StopIteration):
        return None


def _reuse_verified_r_geometry(output: Path) -> dict[str, Any] | None:
    """Reuse an already validated R 3-D artifact without rerunning Depth Pro."""
    r_path = output / "conditions" / "R_requery.npz"
    geometry_path = output / "conditions" / "R_geometry.npz"
    geometry_meta = output / "conditions" / "R_geometry.json"
    counts_path = output / "frame_layer_counts_R.csv"
    if not (r_path.is_file() and geometry_path.is_file() and geometry_meta.is_file() and counts_path.is_file()):
        return None
    try:
        with np.load(r_path, allow_pickle=False) as r, np.load(geometry_path, allow_pickle=False) as g:
            required_2d = {"frame_indices", "timestamps_s", "frame_sizes_hw", "track_ids", "uv", "visibility"}
            required_3d = required_2d | {"geometry_validity", "xyz"}
            if not required_2d.issubset(r.files) or not required_3d.issubset(g.files):
                return None
            frames = np.asarray(r["frame_indices"])
            times = np.asarray(r["timestamps_s"])
            expected_frames = np.arange(488, 519, dtype=np.int64)
            if frames.shape != (31,) or not np.array_equal(frames, expected_frames):
                return None
            if not np.array_equal(g["frame_indices"], frames) or not np.allclose(g["timestamps_s"], times, rtol=0, atol=1e-9):
                return None
            if r["frame_sizes_hw"].shape != (31, 2) or g["frame_sizes_hw"].shape != (31, 2) or not np.array_equal(r["frame_sizes_hw"], g["frame_sizes_hw"]):
                return None
            if not np.all(r["frame_sizes_hw"] == np.asarray([360, 480], dtype=np.int64)):
                return None
            if r["uv"].shape != (31, 289, 2) or r["visibility"].shape != (31, 289):
                return None
            if g["uv"].shape != r["uv"].shape or g["visibility"].shape != r["visibility"].shape:
                return None
            if g["geometry_validity"].shape != (31, 289) or g["xyz"].shape != (31, 289, 3):
                return None
            geometry_validity = np.asarray(g["geometry_validity"], dtype=bool)
            xyz_finite = np.isfinite(np.asarray(g["xyz"])).all(axis=-1)
            if np.any(geometry_validity & ~xyz_finite) or np.any(~geometry_validity & xyz_finite):
                return None
            if not np.array_equal(r["track_ids"], np.arange(289, dtype=np.int64)) or not np.array_equal(g["track_ids"], r["track_ids"]):
                return None
        metadata = _load_json(geometry_meta)
        if (
            metadata.get("status") != "COMPLETE_3D_STATE"
            or metadata.get("query_start_frame") != 488
            or metadata.get("tracker_source_sha") != TRACKER_SHA
            or metadata.get("depth_source_sha") != DEPTH_SHA
            or metadata.get("frame_indices") != expected_frames.astype(int).tolist()
        ):
            return None
        screenshots = [str(Path("review/screenshots") / p.name) for p in sorted((output / "review" / "screenshots").glob("R_frame_*.png"))]
        return {
            "condition": "R",
            "status": "COMPLETE_3D_STATE_NO_HB_SUPPORT",
            "query_start_frame": 488,
            "query_start_pts_s": float(times[0]),
            "frame_indices": frames.astype(int).tolist(),
            "timestamps_s": times.astype(float).tolist(),
            "query_count": 289,
            "raw_uv_status": "UNAVAILABLE_TRACKER_WRAPPER_MASKS_INVISIBLE_UV_TO_NAN",
            "artifact": str(r_path),
            "reused_2d": "REUSED_IDENTITY_VERIFIED_R_2D",
            "geometry_status": "REUSED_IDENTITY_VERIFIED_R_3D",
            "geometry_artifact": str(geometry_path),
            "geometry_elapsed_s": 0.0,
            "screenshots": screenshots,
            "elapsed_s": 0.0,
        }
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
        return None


def _load_r_geometry_sequence(geometry_path: Path) -> SimpleNamespace:
    """Load the saved R geometry as an immutable-in-practice review input."""
    required = {
        "frame_indices",
        "timestamps_s",
        "frame_sizes_hw",
        "track_ids",
        "uv",
        "visibility",
        "geometry_validity",
        "xyz",
    }
    with np.load(geometry_path, allow_pickle=False) as arrays:
        if not required.issubset(arrays.files):
            missing = sorted(required.difference(arrays.files))
            raise ValueError(f"R geometry is missing fields: {missing}")
        values = {name: np.array(arrays[name], copy=True) for name in required}
    frame_indices = np.asarray(values["frame_indices"], dtype=np.int64)
    timestamps_s = np.asarray(values["timestamps_s"], dtype=np.float64)
    frame_sizes_hw = np.asarray(values["frame_sizes_hw"], dtype=np.int64)
    track_ids = np.asarray(values["track_ids"], dtype=np.int64)
    uv = np.asarray(values["uv"], dtype=np.float32)
    visibility = np.asarray(values["visibility"], dtype=bool)
    geometry_validity = np.asarray(values["geometry_validity"], dtype=bool)
    xyz = np.asarray(values["xyz"], dtype=np.float32)
    if (
        frame_indices.ndim != 1
        or timestamps_s.shape != frame_indices.shape
        or frame_sizes_hw.shape != (frame_indices.size, 2)
        or uv.shape != (frame_indices.size, track_ids.size, 2)
        or visibility.shape != uv.shape[:2]
        or geometry_validity.shape != uv.shape[:2]
        or xyz.shape != (frame_indices.size, track_ids.size, 3)
    ):
        raise ValueError("R geometry arrays have incompatible shapes")
    if frame_indices.size == 0 or not np.all(np.isfinite(timestamps_s)) or np.any(np.diff(timestamps_s) <= 0):
        raise ValueError("R geometry timestamps must be finite and strictly increasing")
    if len(np.unique(track_ids)) != track_ids.size:
        raise ValueError("R track IDs are not unique")
    finite_xyz = np.isfinite(xyz).all(axis=-1)
    finite_uv = np.isfinite(uv).all(axis=-1)
    if np.any(geometry_validity & ~visibility) or np.any(geometry_validity & ~finite_xyz):
        raise ValueError("R geometry_validity does not satisfy the saved validity contract")
    if np.any(visibility & ~finite_uv):
        raise ValueError("R visibility contains non-finite UV")
    return SimpleNamespace(
        frame_indices=frame_indices,
        timestamps_s=timestamps_s,
        frame_sizes_hw=frame_sizes_hw,
        track_ids=track_ids,
        uv=uv,
        visibility=visibility,
        geometry_validity=geometry_validity,
        xyz=xyz,
    )


def _matching_r_segmentation_cache(query_frame: int) -> dict[str, Any] | None:
    """Find a pre-existing mask at R's query frame without running segmentation."""
    for path in sorted(SEGMENTATION_CACHE_DIR.glob("*.json")):
        try:
            metadata = _load_json(path)
            frame_index = int(metadata.get("frame_index", -1))
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            continue
        if (
            metadata.get("source_video") == str(FAKE_VIDEO)
            and frame_index == int(query_frame)
            and metadata.get("status") == "COMPLETE"
        ):
            return {"path": str(path), "frame_index": int(query_frame), "status": metadata.get("status")}
    return None


def build_r_local_structure(
    sequence: Any,
    *,
    query_start_pts_s: float | None = None,
    query_start_frame: int | None = None,
    source_id: str = SOURCE_ID,
    window_id: str = WINDOW_ID,
    geometry_meta: Mapping[str, Any] | None = None,
    source_artifact: str | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Build R-only H support from the saved new-query geometry.

    The half-second history starts at R's own query PTS.  This is deliberately
    not the O window boundary: R has no pre-query frames and its local IDs,
    history scale, components and triplets must remain independent.
    """
    frame_indices = np.asarray(sequence.frame_indices, dtype=np.int64)
    timestamps_s = np.asarray(sequence.timestamps_s, dtype=np.float64)
    track_ids = np.asarray(sequence.track_ids, dtype=np.int64)
    if query_start_pts_s is None:
        query_start_pts_s = float(timestamps_s[0])
    if query_start_frame is None:
        query_start_frame = int(frame_indices[0])
    query_start_pts_s = float(query_start_pts_s)
    history_indices = np.flatnonzero(timestamps_s < query_start_pts_s + 0.5).astype(np.int64)
    if history_indices.size == 0:
        raise ValueError("R local history is empty")
    components = rebuild_components_fast(sequence.xyz, sequence.geometry_validity, history_indices)
    grouping = build_local_groups(sequence, history_indices, old_components=components)
    grouping_validation = validate_grouping(grouping, track_ids)
    if not grouping_validation["all_pass"]:
        raise ValueError(f"R grouping validation failed: {grouping_validation}")
    support = json_ready_local_support(
        build_local_support(sequence, window_start_s=query_start_pts_s, grouping=grouping)
    )
    for triplet in support["triplets"]:
        center, first, second = compute_local_derivatives(triplet["states"], triplet["timestamps_s"])
        triplet["center_state"] = center.tolist()
        triplet["first_derivative"] = first.tolist()
        triplet["second_derivative"] = second.tolist()

    retained_groups = [row for row in grouping["groups"] if row.get("retained")]
    retained_slots = sorted({int(slot) for row in retained_groups for slot in row["member_slots"]})
    frame_to_triplets: dict[int, list[dict[str, Any]]] = {}
    for triplet in support["triplets"]:
        for frame in triplet["frame_indices"]:
            frame_to_triplets.setdefault(int(frame), []).append(triplet)
    frame_rows: list[dict[str, Any]] = []
    visible = np.asarray(sequence.visibility, dtype=bool)
    geometry = np.asarray(sequence.geometry_validity, dtype=bool)
    finite_xyz = np.isfinite(np.asarray(sequence.xyz)).all(axis=-1)
    for array_index, frame in enumerate(frame_indices.tolist()):
        base = layer_counts(
            sequence.uv[array_index],
            visible[array_index],
            geometry[array_index],
            sequence.xyz[array_index],
            group_slots=retained_slots,
            pair_slots=(),
        )
        group_mask = np.zeros(track_ids.size, dtype=bool)
        group_mask[retained_slots] = True
        group_geometry_count = int(np.count_nonzero(group_mask & geometry[array_index] & finite_xyz[array_index]))
        relation_count = 0
        triplet_count = 0
        for triplet in frame_to_triplets.get(int(frame), []):
            position = triplet["frame_indices"].index(int(frame))
            if position < 0:
                continue
            triplet_count += 1
            for left, right in triplet["pair_indices"]:
                relation_count += int(bool(geometry[array_index, int(left)] and geometry[array_index, int(right)]))
        frame_rows.append(
            {
                "condition": "R",
                "array_index": int(array_index),
                "source_frame_index": int(frame),
                "timestamp_s": float(timestamps_s[array_index]),
                **base,
                "local_group_geometry_valid_count": group_geometry_count,
                "structure_triplet_count": triplet_count,
                "common_relation_support": relation_count if triplet_count else None,
                "geometry_status": "REUSED_IDENTITY_VERIFIED_R_3D",
                "support_status": "VALID_R_LOCAL_H" if triplet_count else "NO_R_LOCAL_H_TRIPLET_AT_FRAME",
                "query_start_frame": int(query_start_frame),
            }
        )
    b_cache = _matching_r_segmentation_cache(int(query_start_frame))
    if b_cache is None:
        b_status = "NOT_COMPUTED_NO_MATCHING_R_HISTORY_START_SEGMENTATION_CACHE"
        b_reason = "Existing masks contain source frame 476, not R query frame 488; O segmentation is not reused."
    else:
        b_status = "MATCHING_R_HISTORY_START_SEGMENTATION_CACHE_AVAILABLE_NOT_USED"
        b_reason = "A matching cache exists, but B was not recomputed in this bounded H-only completion."
    compact_grouping = {
        key: value
        for key, value in grouping.items()
        if key not in {"parent_components"}
    }
    compact_grouping["parent_components"] = [
        {
            key: row[key]
            for key in ("parent_component_id", "track_ids", "member_slots", "history_frame_count")
            if key in row
        }
        for row in grouping["parent_components"]
    ]
    structure = {
        "schema_version": 1,
        "condition": "R",
        "source_id": source_id,
        "window_id": window_id,
        "id_namespace": "R::independent_query_frame_488",
        "old_condition_reused": False,
        "source_artifact": source_artifact,
        "geometry_metadata": dict(geometry_meta or {}),
        "query_start": {"source_frame_index": int(query_start_frame), "timestamp_s": query_start_pts_s},
        "history_window_s": 0.5,
        "history_boundary_s": query_start_pts_s + 0.5,
        "history_frame_indices": frame_indices[history_indices].astype(int).tolist(),
        "history_timestamps_s": timestamps_s[history_indices].astype(float).tolist(),
        "evaluation_frame_indices": frame_indices[timestamps_s >= query_start_pts_s + 0.5].astype(int).tolist(),
        "evaluation_timestamps_s": timestamps_s[timestamps_s >= query_start_pts_s + 0.5].astype(float).tolist(),
        "grouping": compact_grouping,
        "grouping_validation": grouping_validation,
        "support": support,
        "b_status": b_status,
        "b_reason": b_reason,
        "b_cache": b_cache,
        "summary": {
            "parent_component_count": len(grouping["parent_components"]),
            "group_count_total": len(grouping["groups"]),
            "h_retained_group_count": int(grouping["retained_group_count"]),
            "h_retained_member_slot_count": len(retained_slots),
            "h_retained_pair_count": int(sum(len(row["member_slots"]) * (len(row["member_slots"]) - 1) // 2 for row in retained_groups)),
            "h_valid_triplet_count": int(support["valid_triplet_count"]),
            "h_invalid_reason_counts": {},
            "frames_with_triplet_support": sorted(int(frame) for frame, rows in frame_to_triplets.items() if rows),
        },
    }
    from collections import Counter

    structure["summary"]["h_invalid_reason_counts"] = dict(Counter(item["reason"] for item in support["invalid_reasons"]))
    return structure, frame_rows


def _build_r_structure(output: Path) -> dict[str, Any]:
    """Create the H-only R structure artifact from saved arrays, never a frontend run."""
    geometry_path = output / "conditions" / "R_geometry.npz"
    r_path = output / "conditions" / "R_requery.npz"
    metadata_path = output / "conditions" / "R_geometry.json"
    if not geometry_path.is_file() or not r_path.is_file() or not metadata_path.is_file():
        raise FileNotFoundError("verified R_requery, R_geometry and R_geometry metadata are required")
    sequence = _load_r_geometry_sequence(geometry_path)
    with np.load(r_path, allow_pickle=False) as r_arrays:
        for key in ("frame_indices", "timestamps_s", "uv", "visibility"):
            if not np.array_equal(r_arrays[key], getattr(sequence, key), equal_nan=True):
                raise ValueError(f"R geometry and R trajectory disagree in {key}")
    metadata = _load_json(metadata_path)
    structure, rows = build_r_local_structure(
        sequence,
        query_start_pts_s=float(sequence.timestamps_s[0]),
        query_start_frame=int(sequence.frame_indices[0]),
        geometry_meta=metadata,
        source_artifact=str(r_path),
    )
    structure_path = output / "conditions" / "R_structure.json"
    _write_json(structure_path, structure)
    _write_csv(output / "frame_layer_counts_R.csv", rows)
    summary = structure["summary"]
    status = {
        "structure_status": "COMPLETE_R_LOCAL_H",
        "structure_artifact": str(structure_path),
        "id_namespace": structure["id_namespace"],
        "old_condition_reused": False,
        "h_group_count": summary["h_retained_group_count"],
        "h_group_count_total": summary["group_count_total"],
        "h_member_slot_count": summary["h_retained_member_slot_count"],
        "h_retained_pair_count": summary["h_retained_pair_count"],
        "h_valid_triplet_count": summary["h_valid_triplet_count"],
        "h_support_status": structure["support"]["support_status"],
        "b_status": structure["b_status"],
        "b_reason": structure["b_reason"],
        "structure_frame_rows": len(rows),
    }
    return {"structure": structure, "rows": rows, "status": status}


def _run_requery(output: Path, *, max_extra_s: float = 1.0) -> dict[str, Any]:
    started = time.perf_counter()
    result: dict[str, Any] = {"condition": "R", "status": "NOT_RUN"}
    reused_geometry = _reuse_verified_r_geometry(output)
    if reused_geometry is not None:
        reused_geometry["elapsed_s"] = time.perf_counter() - started
        return reused_geometry
    try:
        with np.load(PARTICLE, allow_pickle=False) as z:
            source_start = int(z["frame_indices"][0])
            target = int(z["frame_indices"][int(np.argmin(np.abs(z["timestamps_s"] - TARGET_TIMES[1])))])
            fps_delta = float(np.median(np.diff(z["timestamps_s"])))
            end = target + int(math.ceil(max_extra_s / fps_delta))
        frames = list(range(target, end + 1))
        decoded = _decode_exact(FAKE_VIDEO, frames)
        r_path = output / "conditions" / "R_requery.npz"
        uv = visible = None
        reuse_reason = "NO_EXISTING_R_ARTIFACT"
        if r_path.is_file():
            try:
                with np.load(r_path, allow_pickle=False) as cached:
                    required = {"frame_indices", "timestamps_s", "frame_sizes_hw", "track_ids", "uv", "visibility"}
                    if required.issubset(cached.files):
                        cached_frames = cached["frame_indices"]
                        cached_times = cached["timestamps_s"]
                        cached_uv = cached["uv"]
                        cached_visible = cached["visibility"]
                        identity_ok = (
                            cached_frames.shape == (len(frames),)
                            and np.array_equal(cached_frames, np.asarray(frames, dtype=np.int64))
                            and cached_times.shape == (len(frames),)
                            and np.allclose(cached_times, [f.timestamp_s for f in decoded.frames], rtol=0, atol=1e-9)
                            and cached_uv.shape == (len(frames), 289, 2)
                            and cached_visible.shape == (len(frames), 289)
                            and np.array_equal(cached["track_ids"], np.arange(289, dtype=np.int64))
                        )
                        if identity_ok:
                            uv = np.array(cached_uv, copy=True)
                            visible = np.array(cached_visible, copy=True)
                            reuse_reason = "REUSED_IDENTITY_VERIFIED_R_2D"
                        else:
                            reuse_reason = "EXISTING_R_ARTIFACT_IDENTITY_MISMATCH"
            except Exception as exc:
                reuse_reason = f"EXISTING_R_ARTIFACT_UNREADABLE:{type(exc).__name__}:{exc}"
        tracker = None
        if uv is None or visible is None:
            sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts"))
            from run_v7_explicit_geometry_frontend import OnlineBootsTapir
            tracker = OnlineBootsTapir(TAPNET_SOURCE, TAPNET_CHECKPOINT, process_size=256, grid_size=17)
            uv, visible = tracker.track(decoded)
        uv = np.asarray(uv, dtype=np.float32)
        visible = np.asarray(visible, dtype=bool)
        geo = np.zeros_like(visible, dtype=bool)
        xyz = np.full((*uv.shape[:2], 3), np.nan, dtype=np.float32)
        arrays_path = output / "conditions" / "R_requery.npz"
        arrays_path.parent.mkdir(parents=True, exist_ok=True)
        if reuse_reason != "REUSED_IDENTITY_VERIFIED_R_2D":
            np.savez_compressed(arrays_path, frame_indices=np.asarray(frames, np.int64), timestamps_s=np.asarray([f.timestamp_s for f in decoded.frames], np.float64), frame_sizes_hw=np.asarray([[f.rgb.shape[0], f.rgb.shape[1]] for f in decoded.frames], np.int64), track_ids=np.arange(uv.shape[1], dtype=np.int64), uv=uv.astype(np.float32), visibility=visible, geometry_validity=geo, xyz=xyz)
        rows = []
        for i, frame in enumerate(decoded.frames):
            c = layer_counts(uv[i], visible[i], geo[i], xyz[i])
            rows.append({"condition": "R", "array_index": i, "source_frame_index": int(frame.source_frame_index), "timestamp_s": float(frame.timestamp_s), **c, "geometry_status": "NOT_COMPUTED_2D_REQUERY_ONLY", "support_status": "NOT_COMPUTED_NEW_ID_NO_REUSED_HB", "query_start_frame": target})
        geometry_status = "NOT_ATTEMPTED"
        geometry_error = None
        geometry_artifact = None
        geometry_elapsed = None
        geometry_started = time.perf_counter()
        try:
            sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts"))
            from run_v7_explicit_geometry_frontend import DepthProRunner, rgbd_odometry
            from sparse3d_forgery.experiments.v7_explicit_geometry_frontend import (
                accumulate_world_from_camera,
                causal_first_frame_intrinsics,
                sample_depth_at_uv,
                world_xyz,
            )
            depth_runner = DepthProRunner(DEPTH_SOURCE, DEPTH_CHECKPOINT)
            depths, focals, frame_depth_valid = depth_runner.infer(decoded)
            intrinsics, fixed_focal_px = causal_first_frame_intrinsics(depths, focals)
            adjacent, pair_valid, information = rgbd_odometry(decoded, depths, intrinsics)
            world_from_camera, pose_valid = accumulate_world_from_camera(adjacent, pair_valid)
            sampled_depth, sampled_valid = sample_depth_at_uv(depths, uv)
            observation_valid = visible & sampled_valid & frame_depth_valid[:, None]
            xyz, geo = world_xyz(uv, sampled_depth, intrinsics, world_from_camera, observation_valid, pose_valid)
            geometry_artifact = output / "conditions" / "R_geometry.npz"
            np.savez_compressed(geometry_artifact, frame_indices=np.asarray(frames, np.int64), timestamps_s=np.asarray([f.timestamp_s for f in decoded.frames], np.float64), frame_sizes_hw=np.asarray([[f.rgb.shape[0], f.rgb.shape[1]] for f in decoded.frames], np.int64), track_ids=np.arange(uv.shape[1], dtype=np.int64), uv=uv.astype(np.float32), visibility=visible, geometry_validity=geo, xyz=xyz)
            _write_json(output / "conditions" / "R_geometry.json", {
                "status": "COMPLETE_3D_STATE",
                "artifact": str(geometry_artifact),
                "tracker": "online_bootstapir",
                "tracker_source_sha": TRACKER_SHA,
                "depth": "apple_depth_pro",
                "depth_source_sha": DEPTH_SHA,
                "depth_semantics": "optical_axis_z_depth_interpretation_of_official_depth_output",
                "intrinsics_policy": "CAUSAL_FIRST_FRAME_DEPTH_PRO_FOCAL_ASSUMPTION",
                "fixed_focal_px": float(fixed_focal_px),
                "pose": "open3d_0.19_adjacent_rgbd_odometry",
                "pose_convention": "target_camera_from_source_camera; inverted during world accumulation",
                "query_start_frame": target,
                "frame_indices": frames,
                "geometry_valid_per_frame": np.sum(geo, axis=1).astype(int).tolist(),
                "pose_valid_per_frame": pose_valid.astype(bool).tolist(),
                "observation_valid_per_frame": np.sum(observation_valid, axis=1).astype(int).tolist(),
                "support_status": "NOT_COMPUTED_NEW_ID_NO_REUSED_HB",
            })
            geometry_status = "COMPLETE_3D_STATE_NO_HB_SUPPORT"
            for i, frame in enumerate(decoded.frames):
                c = layer_counts(uv[i], visible[i], geo[i], xyz[i])
                rows[i].update(c, geometry_status=geometry_status, support_status="NOT_COMPUTED_NEW_ID_NO_REUSED_HB")
            geometry_elapsed = time.perf_counter() - geometry_started
        except Exception as exc:  # preserve a precise geometry block while retaining R 2D
            geometry_status = "GEOMETRY_FAILED"
            geometry_error = f"{type(exc).__name__}: {exc}"
            geometry_elapsed = time.perf_counter() - geometry_started
        _write_csv(output / "frame_layer_counts_R.csv", rows)
        result.update({"status": geometry_status if geometry_status != "GEOMETRY_FAILED" else "COMPLETE_2D_ONLY_GEOMETRY_FAILED", "query_start_frame": target, "query_start_pts_s": float(decoded.frames[0].timestamp_s), "frame_indices": frames, "timestamps_s": [float(f.timestamp_s) for f in decoded.frames], "query_count": int(uv.shape[1]), "raw_uv_status": "UNAVAILABLE_TRACKER_WRAPPER_MASKS_INVISIBLE_UV_TO_NAN", "artifact": str(arrays_path), "reused_2d": reuse_reason, "geometry_status": geometry_status, "geometry_artifact": str(geometry_artifact) if geometry_artifact else None, "geometry_elapsed_s": geometry_elapsed, "elapsed_s": time.perf_counter() - started})
        if geometry_error:
            result["geometry_error"] = geometry_error
        result["screenshots"] = _save_pngs(output, decoded, rows, uv, visible, "R", geometry=geo)
        del tracker
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
    except Exception as exc:  # diagnostic boundary: preserve exact failure
        result.update({"status": "FAILED", "error": f"{type(exc).__name__}: {exc}", "elapsed_s": time.perf_counter() - started})
    return result


def _run_cotracker(output: Path, decoded: Any) -> dict[str, Any]:
    started = time.perf_counter()
    result: dict[str, Any] = {"condition": "T", "status": "NOT_RUN", "official_repo": "https://github.com/facebookresearch/co-tracker", "source_commit": COTRACKER_SHA}
    try:
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable for the official online candidate")
        if not COTRACKER_SOURCE.is_dir() or not COTRACKER_CHECKPOINT.is_file():
            raise FileNotFoundError("official CoTracker source or scaled_online.pth is not available in the local cache")
        sys.path.insert(0, str(COTRACKER_SOURCE))
        from cotracker.predictor import CoTrackerOnlinePredictor

        with np.load(PARTICLE, allow_pickle=False) as z:
            original_uv = np.asarray(z["uv"], dtype=np.float32)
            query_uv = np.asarray(original_uv[0], dtype=np.float32)
            frame_indices = np.asarray(z["frame_indices"], dtype=np.int64)
            timestamps_s = np.asarray(z["timestamps_s"], dtype=np.float64)
        if query_uv.shape != (289, 2) or not np.isfinite(query_uv).all():
            raise ValueError("O query frame does not contain 289 finite source-pixel query coordinates")
        rgb = np.stack([np.asarray(frame.rgb, dtype=np.uint8) for frame in decoded.frames], axis=0)
        device = torch.device("cuda")
        video = torch.from_numpy(rgb).permute(0, 3, 1, 2).unsqueeze(0).float().to(device)
        queries = torch.zeros((1, 289, 3), dtype=torch.float32, device=device)
        queries[:, :, 1:] = torch.from_numpy(query_uv).to(device)
        model = CoTrackerOnlinePredictor(checkpoint=str(COTRACKER_CHECKPOINT), window_len=16).to(device).eval()
        step = int(model.step)
        with torch.no_grad():
            # The official online protocol initializes on the first half-window,
            # then returns a full window whose latter half is newly evaluated.
            model(video[:, :step], is_first_step=True, queries=queries, grid_size=0, add_support_grid=False)
            tracks_full = np.full((len(frame_indices), 289, 2), np.nan, dtype=np.float32)
            visible_full = np.zeros((len(frame_indices), 289), dtype=bool)
            calls = []
            for start in range(0, len(frame_indices) - step, step):
                chunk = video[:, start : min(start + 2 * step, len(frame_indices))]
                tracks, visible = model(video_chunk=chunk, grid_size=0, add_support_grid=False)
                if tracks is None or visible is None:
                    continue
                t = tracks[0].detach().float().cpu().numpy()
                v = visible[0].detach().cpu().numpy()
                if v.ndim == 3 and v.shape[-1] == 1:
                    v = v[..., 0]
                if t.ndim != 3 or v.ndim != 2 or t.shape[1:] != (289, 2) or v.shape[1:] != (289,):
                    raise ValueError(f"unexpected CoTracker online output shapes tracks={t.shape}, visibility={v.shape}")
                if not calls:
                    take = min(len(frame_indices), t.shape[0])
                    tracks_full[:take] = t[:take]
                    visible_full[:take] = v[:take]
                    calls.append({"chunk_start": int(start), "output_frames": int(t.shape[0]), "written_range": [0, int(take)]})
                else:
                    dest_start = int(start + step)
                    src_start = int(step)
                    take = min(len(frame_indices) - dest_start, t.shape[0] - src_start)
                    if take > 0:
                        tracks_full[dest_start : dest_start + take] = t[src_start : src_start + take]
                        visible_full[dest_start : dest_start + take] = v[src_start : src_start + take]
                    calls.append({"chunk_start": int(start), "output_frames": int(t.shape[0]), "written_range": [dest_start, int(dest_start + max(take, 0))]})
        geo = np.zeros_like(visible_full, dtype=bool)
        xyz = np.full((len(frame_indices), 289, 3), np.nan, dtype=np.float32)
        artifact = output / "conditions" / "T_cotracker3_online.npz"
        artifact.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            artifact,
            frame_indices=frame_indices,
            timestamps_s=timestamps_s,
            frame_sizes_hw=np.asarray([[frame.rgb.shape[0], frame.rgb.shape[1]] for frame in decoded.frames], dtype=np.int64),
            track_ids=np.arange(289, dtype=np.int64),
            uv=tracks_full,
            visibility=visible_full,
            geometry_validity=geo,
            xyz=xyz,
        )
        t_rows = []
        for i, frame in enumerate(decoded.frames):
            t_rows.append({
                "condition": "T",
                "array_index": int(i),
                "source_frame_index": int(frame.source_frame_index),
                "timestamp_s": float(frame.timestamp_s),
                **layer_counts(tracks_full[i], visible_full[i], geo[i], xyz[i]),
                "geometry_status": "NOT_ATTEMPTED_T_2D_DIAGNOSTIC",
                "support_status": "NOT_COMPUTED_T_NEW_TRACK_IDS",
                "query_start_frame": int(frame_indices[0]),
            })
        _write_csv(output / "frame_layer_counts_T.csv", t_rows)
        screenshots = _save_pngs(output, decoded, t_rows, tracks_full, visible_full, "T", geometry=geo)
        result.update({
            "status": "COMPLETE_2D_TRACKS_NO_3D",
            "artifact": str(artifact),
            "screenshots": screenshots,
            "frame_indices": frame_indices.tolist(),
            "timestamps_s": timestamps_s.tolist(),
            "query_count": 289,
            "query_start_frame": int(frame_indices[0]),
            "query_start_pts_s": float(timestamps_s[0]),
            "query_identity": "O source-frame-0 UV copied only as T query coordinates; T track IDs remain independent",
            "raw_uv_status": "AVAILABLE_OFFICIAL_COTRACKER_PREDICTIONS_WITH_SEPARATE_VISIBILITY",
            "geometry_status": "NOT_ATTEMPTED_T_2D_DIAGNOSTIC",
            "support_status": "NOT_COMPUTED_T_NEW_TRACK_IDS",
            "official_source_path": str(COTRACKER_SOURCE),
            "checkpoint": str(COTRACKER_CHECKPOINT),
            "checkpoint_url": "https://huggingface.co/facebook/cotracker3/resolve/main/scaled_online.pth",
            "checkpoint_sha256": COTRACKER_CHECKPOINT_SHA256,
            "model_step": step,
            "interp_shape": [int(x) for x in model.interp_shape],
            "chunk_calls": calls,
            "note": "official CoTracker3 online tracking completed; this bounded case stores 2D tracks only, so no T 3D/H/B result is claimed",
        })
        del model
    except Exception as exc:
        result.update({"status": "BLOCKED_OFFICIAL_RUNTIME_UNAVAILABLE", "error": f"{type(exc).__name__}: {exc}"})
    result["elapsed_s"] = time.perf_counter() - started
    _write_json(output / "conditions" / "T_status.json", result)
    return result


def _html_page(output: Path, manifest: Mapping[str, Any], statuses: Mapping[str, Any]) -> None:
    # The page is deliberately dependency-free and reads the small case and trajectory JSON payloads.
    page = """<!doctype html><meta charset='utf-8'><title>V7 local observation recovery</title>
<style>body{font:15px sans-serif;background:#111827;color:#e5e7eb;margin:20px}h1{font-size:24px}.grid{display:grid;grid-template-columns:repeat(3,1fr);gap:12px}.panel,.roi{background:#1f2937;padding:12px;border-radius:8px;margin:10px 0}.view,.roi-view{position:relative;background:#000}.view img,.roi-view img{width:100%;display:block}.view canvas,.roi-view canvas{position:absolute;inset:0;width:100%;height:100%}.view canvas{pointer-events:auto;cursor:crosshair}.roi-view canvas{pointer-events:auto}.muted{color:#9ca3af}.ok{color:#86efac}.warn{color:#fbbf24}button{margin:3px;padding:5px}input{width:70px;margin:2px}code{color:#bfdbfe}</style>
<h1>V7 04LAX fake 局部观测恢复</h1>
<p class='muted'>这是单案例观测诊断，不是检测器评价。O/R/T 的 query ID 不跨条件连接；不可见点不生成有效 XYZ。</p>
<div id='facts'></div>
<div><button id='prev'>上一帧</button><button id='next'>下一帧</button><button id='jump'>跳到源帧488</button><span id='frame'></span></div>
<div class='panel'>选点 ID（条件内）：<input id='pointId' type='number' min='0' max='288' value='0'> <label><input id='showIds' type='checkbox' checked>显示 ID</label> <label><input id='showTrail' type='checkbox' checked>显示最近5帧尾线</label>；点击 O/R/T 任一点可选择该条件自己的 ID。跨条件同号不表示同一物理点。</div>
<div class='panel'><b>R 独立 H 结构核对：</b>组 <select id='rGroup'></select> triplet <select id='rTriplet'></select> <label><input id='rGroupLayer' type='checkbox' checked>R组成员</label> <label><input id='rCommonLayer' type='checkbox' checked>共同成员</label> <label><input id='rPairLayer' type='checkbox' checked>有效 pair</label> <span id='rStructureStatus' class='muted'>结构载荷读取中…</span><br><span class='muted'>橙色=保留组成员，青色=当前 triplet 三帧共同成员，紫色连线=实际有效 pair；未高亮的可见点未参与所选结构。R 组/ID 仅在 R 命名空间内解释。</span></div>
<div class='grid'><section class='panel'><h2>O 原有结果</h2><div id='oStatus'></div><div class='view'><img id='oImg'><canvas id='oCanvas'></canvas></div></section>
<section class='panel'><h2>R 重新查询</h2><div id='rStatus'></div><div class='view'><img id='rImg'><canvas id='rCanvas'></canvas></div></section>
<section class='panel'><h2>T CoTracker3 online</h2><div id='tStatus'></div><div class='view'><img id='tImg'><canvas id='tCanvas'></canvas></div></section></div>
<section class='roi'><h2>源帧488粗 ROI（一次确认/修改）</h2>
<p class='warn'>黄色框只是根据用户截图给出的待确认建议，不是空间真值、人体分割或伪造 mask。请在原图像素坐标中修改；确认后可下载 JSON。</p>
<div class='roi-view'><img id='roiImg' src='../review/source_frames/source_frame_488.png'><canvas id='roiCanvas'></canvas></div>
<div>x0 <input id='x0' type='number'> y0 <input id='y0' type='number'> x1 <input id='x1' type='number'> y1 <input id='y1' type='number'>
<label><input id='roiConfirm' type='checkbox'> 我确认这是粗略人工 ROI（仍不是真值）</label><button id='roiSave'>保存/下载 ROI JSON</button></div>
<div id='roiCoord' class='muted'>把鼠标移到原图上查看源像素坐标。</div><div id='roiState' class='warn'>PENDING_USER_CONFIRMATION</div></section>
<p>图例：<span class='ok'>绿色=当前可见且有 UV</span>；状态文字区分未查询、二维轨迹和三维几何。源帧 487/488 的 PTS 是 16.249583/16.282950 s。R 查询启动前显示 NOT QUERIED；T 的不可见预测 UV 与有效观测分开保存。</p>
<script>
let d=null,trace=null,rStructure=null,i=0,selectedPanel='O',selectedId=0,selectedRGroup=0,selectedRTriplet=0;const $=x=>document.getElementById(x);const roiKey='v7-04LAX-source-frame-488-roi';
function esc(x){return String(x).replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]))}
function statusHtml(panel,row,p){const twoD=panel==='O'?'REUSED_CANONICAL_2D':(panel==='R'?'REQUERY_2D_TRACKS':(panel==='T'?'OFFICIAL_COTRACKER3_ONLINE_2D':'UNKNOWN'));const xyz=panel==='O'?'REUSED_CANONICAL_XYZ':(row.geometry_status||'NOT_COMPUTED');const hb=panel==='O'?'SAVED_H_B':(row.support_status||'NOT_COMPUTED');const trip=panel==='O'?'SAVED_H_B_TRIPLET':(panel==='R'?(Number(row.structure_triplet_count||0)>0?`R_LOCAL_H_TRIPLETS=${row.structure_triplet_count}`:'NO_R_LOCAL_H_TRIPLET_AT_FRAME'):'NOT_COMPUTED_T');return `condition: ${p.status||'UNKNOWN'}<br>2D tracking: ${twoD}; frame=${row.source_frame_index}, PTS=${Number(row.timestamp_s).toFixed(6)} s, UV=${row.uv_available_count}, visible=${row.visibility_count}<br>XYZ: ${xyz}; H/B grouping: ${hb}; triplet support: ${trip}`}
function overlay(panel,j){const rec=trace&&trace.conditions[panel],can=$(panel.toLowerCase()+'Canvas');if(!rec||j<0||!can.width)return;const g=can.getContext('2d'),id=panel===selectedPanel?Math.max(0,Math.min(Number(selectedId)||0,rec.track_ids.length-1)):-1;g.clearRect(0,0,can.width,can.height);if(id<0)return;const uv=rec.uv[j]?.[id];const finite=uv&&rec.uv_finite[j]?.[id];if($('showTrail').checked){const start=Math.max(0,j-4);g.strokeStyle='#facc15';g.lineWidth=3;g.beginPath();let active=false;for(let k=start;k<=j;k++){const q=rec.uv[k]?.[id];if(!q||!rec.uv_finite[k]?.[id]||!rec.visibility[k]?.[id]){active=false;continue}if(active)g.lineTo(q[0],q[1]);else g.moveTo(q[0],q[1]);active=true}g.stroke()}if(finite){g.fillStyle='#ffffff';g.strokeStyle='#ef4444';g.lineWidth=2;g.beginPath();g.arc(uv[0],uv[1],5,0,Math.PI*2);g.fill();g.stroke();if($('showIds').checked){g.fillStyle='#ffffff';g.font='bold 14px sans-serif';g.fillText(`id=${id}`,uv[0]+7,uv[1]-7)}}}
function draw(panel){const p=d.conditions[panel],img=$(panel.toLowerCase()+'Img'),can=$(panel.toLowerCase()+'Canvas');if(!p||!p.rows){img.removeAttribute('src');can.getContext('2d').clearRect(0,0,can.width,can.height);$(panel.toLowerCase()+'Status').textContent='NOT_QUERIED';return}const sourceFrame=d.conditions.O.rows[i].source_frame_index;const j=p.rows.findIndex(r=>Number(r.source_frame_index)===Number(sourceFrame));const traceIndex=trace?.conditions?.[panel]?.frame_indices?.indexOf(Number(sourceFrame))??-1;if(j<0||traceIndex<0||!p.screenshots||!p.screenshots[j]){img.removeAttribute('src');can.getContext('2d').clearRect(0,0,can.width,can.height);$(panel.toLowerCase()+'Status').textContent=panel==='R'?'NOT_QUERIED_BEFORE_R_START / OUTSIDE_R_RANGE':(p.status||'NO_RESULT');return}const row=p.rows[j];$(panel.toLowerCase()+'Status').innerHTML=statusHtml(panel,row,p);img.onload=()=>{can.width=img.naturalWidth;can.height=img.naturalHeight;overlay(panel,traceIndex)};img.src='../'+p.screenshots[j];if(img.complete){can.width=img.naturalWidth;can.height=img.naturalHeight;overlay(panel,traceIndex)}}
function populateRSelectors(){if(!rStructure)return;const groups=(rStructure.grouping?.groups||[]).filter(x=>x.retained);const trips=rStructure.support?.triplets||[];$('rGroup').innerHTML=groups.map((x,n)=>`<option value='${n}'>G${x.local_group_id} (${x.track_ids.length} members)</option>`).join('');$('rTriplet').innerHTML=trips.map((x,n)=>`<option value='${n}'>T${x.triplet_id} / G${x.local_group_id} (${x.common_track_ids.length} common)</option>`).join('');selectedRGroup=Math.min(selectedRGroup,Math.max(0,groups.length-1));selectedRTriplet=Math.min(selectedRTriplet,Math.max(0,trips.length-1));$('rGroup').value=String(selectedRGroup);$('rTriplet').value=String(selectedRTriplet)}
function overlayRStructure(j){const s=rStructure,rec=trace?.conditions?.R,can=$('rCanvas');if(!s||!rec||j<0||!can.width)return;const groups=(s.grouping?.groups||[]).filter(x=>x.retained),trips=s.support?.triplets||[],group=groups[selectedRGroup],trip=trips[selectedRTriplet],uv=rec.uv[j]||[],vis=rec.visibility[j]||[],geo=rec.geometry_validity?.[j]||[],g=can.getContext('2d');const point=(slot,colour,radius)=>{const q=uv[slot];if(!q||!rec.uv_finite[j]?.[slot]||!vis[slot])return false;g.beginPath();g.arc(q[0],q[1],radius,0,Math.PI*2);g.fillStyle=colour;g.fill();return true};let groupCount=0,commonCount=0,pairCount=0;if($('rGroupLayer').checked&&group){for(const slot of group.member_slots)if(point(Number(slot),'#f59e0b',5))groupCount++}if($('rCommonLayer').checked&&trip){for(const slot of trip.common_member_indices)if(point(Number(slot),'#22d3ee',7))commonCount++}if($('rPairLayer').checked&&trip){g.strokeStyle='#e879f9';g.lineWidth=2;for(const pair of trip.pair_indices){const a=Number(pair[0]),b=Number(pair[1]),qa=uv[a],qb=uv[b];if(qa&&qb&&rec.uv_finite[j]?.[a]&&rec.uv_finite[j]?.[b]&&vis[a]&&vis[b]&&geo[a]&&geo[b]){g.beginPath();g.moveTo(qa[0],qa[1]);g.lineTo(qb[0],qb[1]);g.stroke();pairCount++}}}const frame=Number(rec.frame_indices[j]);$('rStructureStatus').textContent=`H=${s.summary?.h_retained_group_count??'NA'}组/${s.summary?.h_valid_triplet_count??'NA'} triplet；当前 frame ${frame}：组可见 ${groupCount}，共同成员 ${commonCount}，有效 pair ${pairCount}；B=${s.b_status||'UNKNOWN'}`}
function render(){const o=d.conditions.O;if(!o)return;i=Math.max(0,Math.min(i,o.rows.length-1));$('frame').textContent=` O[${i}/${o.rows.length-1}] source_frame=${o.rows[i].source_frame_index} PTS=${Number(o.rows[i].timestamp_s).toFixed(6)} s phase=${o.rows[i].phase||''}`;draw('O');draw('R');draw('T');if(rStructure){const r=trace?.conditions?.R,j=r?Number(r.frame_indices.indexOf(Number(o.rows[i].source_frame_index))):-1;if(j>=0)requestAnimationFrame(()=>overlayRStructure(j))}}
function rect(){return ['x0','y0','x1','y1'].map(k=>Number($(k).value))}
function drawRoi(){const img=$('roiImg'),c=$('roiCanvas');if(!img.naturalWidth)return;c.width=img.naturalWidth;c.height=img.naturalHeight;const g=c.getContext('2d');g.clearRect(0,0,c.width,c.height);const r=rect();if(r.every(Number.isFinite)){g.strokeStyle='#facc15';g.lineWidth=3;g.strokeRect(r[0],r[1],r[2]-r[0],r[3]-r[1])}}
function updateRoi(){['x0','y0','x1','y1'].forEach(k=>$(k).addEventListener('input',drawRoi));drawRoi();}
Promise.all([fetch('../case_data.json').then(x=>x.json()),fetch('trajectory_data.json').then(x=>x.json())]).then(([x,t])=>{d=x;trace=t;const tr=x.manifest.support.selected_triplet,h=x.manifest.history_evaluation;$('facts').innerHTML=`<p>source=${esc(x.manifest.source_id)} role=${esc(x.manifest.role)} window=${esc(x.manifest.window_id)}; init frame=${x.manifest.query_initialization.source_frame_index} PTS=${Number(x.manifest.query_initialization.timestamp_s).toFixed(6)} s; frame488=${h.target_frame_status[1].phase} (${Number(h.target_frame_status[1].relative_to_initialization_s).toFixed(6)} s after init, model target=${h.target_frame_status[1].model_evaluation_frame}); H groups=${x.manifest.support.h_group_count} (member slots=${x.manifest.support.h_member_slot_count}), B groups=${x.manifest.support.b_group_count}, H pairs=${x.manifest.support.h_pair_count}, H triplets=${x.manifest.support.h_triplet_count}; selected triplet=${tr?tr.triplet_id:'none'}.</p>`;const saved=localStorage.getItem(roiKey);const r=saved?JSON.parse(saved):{rectangle_xyxy:x.manifest.roi.suggested_rectangle_xyxy||[110,0,370,359],confirmed:false};['x0','y0','x1','y1'].forEach((k,n)=>$(k).value=r.rectangle_xyxy[n]);$('roiConfirm').checked=!!r.confirmed;$('roiState').textContent=r.confirmed?'CONFIRMED_USER_LOCAL':'PENDING_USER_CONFIRMATION';const oi=x.conditions.O.rows.findIndex(r=>Number(r.source_frame_index)===488);i=oi>=0?oi:0;render();updateRoi()});
['prev','next'].forEach(k=>$(k).onclick=()=>{i+=k==='next'?1:-1;render()});$('jump').onclick=()=>{if(!d)return;const j=d.conditions.O.rows.findIndex(r=>Number(r.source_frame_index)===488);if(j>=0){i=j;render()}};$('pointId').oninput=()=>{selectedId=Math.max(0,Math.min(288,Number($('pointId').value)||0));render()};['showIds','showTrail'].forEach(k=>$(k).onchange=()=>render());['O','R','T'].forEach(panel=>$(panel.toLowerCase()+'Canvas').onclick=e=>{if(!d||!trace)return;const rec=trace.conditions[panel],sourceFrame=d.conditions.O.rows[i].source_frame_index,j=rec.frame_indices.indexOf(Number(sourceFrame));if(j<0)return;const box=$(panel.toLowerCase()+'Canvas').getBoundingClientRect(),x=(e.clientX-box.left)*$(panel.toLowerCase()+'Canvas').width/box.width,y=(e.clientY-box.top)*$(panel.toLowerCase()+'Canvas').height/box.height;let best=-1,dist=Infinity;rec.uv[j].forEach((q,n)=>{if(!q||!rec.uv_finite[j][n]||!rec.visibility[j][n])return;const dd=(q[0]-x)**2+(q[1]-y)**2;if(dd<dist){dist=dd;best=n}});if(best>=0&&dist<400){selectedId=best;$('pointId').value=best;render()}});$('roiImg').onload=drawRoi;['x0','y0','x1','y1'].forEach(k=>$(k).oninput=drawRoi);$('roiCanvas').onmousemove=e=>{const r=$('roiCanvas').getBoundingClientRect();$('roiCoord').textContent=`源像素 x=${((e.clientX-r.left)*$('roiCanvas').width/r.width).toFixed(1)}, y=${((e.clientY-r.top)*$('roiCanvas').height/r.height).toFixed(1)}（宽480×高360）`};$('roiConfirm').onchange=()=>{$('roiState').textContent=$('roiConfirm').checked?'READY_TO_SAVE_USER_CONFIRMATION':'PENDING_USER_CONFIRMATION'};$('roiSave').onclick=()=>{const r=rect();if(!$('roiConfirm').checked||!(r[0]<r[2]&&r[1]<r[3])){alert('请先确认复选框，并保证 ROI 面积为正');return}const payload={status:'USER_CONFIRMED_SOURCE_PIXEL_ROI',source:'04LAX',role:'fake',window_id:'0002_MANIP_25::fake',source_frame_index:488,pts_s:16.28294961628295,image_size_hw:[360,480],rectangle_xyxy:r,not_ground_truth:true,confirmed_at:new Date().toISOString()};localStorage.setItem(roiKey,JSON.stringify({rectangle_xyxy:r,confirmed:true}));$('roiState').textContent='CONFIRMED_USER_LOCAL; JSON downloaded';const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([JSON.stringify(payload,null,2)],{type:'application/json'}));a.download='roi_mapping_04LAX_frame488.json';a.click()};
['O','R','T'].forEach(panel=>$(panel.toLowerCase()+'Canvas').addEventListener('click',()=>{selectedPanel=panel;render()}));
fetch('../conditions/R_structure.json').then(x=>x.ok?x.json():null).then(s=>{rStructure=s;if(!s){$('rStructureStatus').textContent='R_STRUCTURE_NOT_AVAILABLE';return}populateRSelectors();$('rGroup').onchange=()=>{selectedRGroup=Number($('rGroup').value)||0;render()};$('rTriplet').onchange=()=>{selectedRTriplet=Number($('rTriplet').value)||0;render()};['rGroupLayer','rCommonLayer','rPairLayer'].forEach(k=>$(k).onchange=()=>render());$('rImg').addEventListener('load',()=>{const o=d?.conditions?.O,r=trace?.conditions?.R;if(!o||!r)return;const j=r.frame_indices.indexOf(Number(o.rows[i].source_frame_index));if(j>=0)setTimeout(()=>overlayRStructure(j),0)});render()}).catch(e=>{$('rStructureStatus').textContent='R_STRUCTURE_LOAD_FAILED: '+e});
</script>"""
    review = output / "review"
    review.mkdir(parents=True, exist_ok=True)
    (review / "index.html").write_text(page, encoding="utf-8")


def _motion_stats(delta: np.ndarray, mask: np.ndarray) -> dict[str, Any]:
    values = np.asarray(delta, dtype=np.float64)[np.asarray(mask, dtype=bool)]
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {"n": 0, "nonzero": 0, "mean": None, "median": None, "p95": None, "max": None}
    return {
        "n": int(values.size),
        "nonzero": int(np.count_nonzero(values > 0.0)),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "p95": float(np.percentile(values, 95)),
        "max": float(np.max(values)),
    }


def _t_motion_diagnostics(output: Path, statuses: Mapping[str, Any], manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Check saved T UV at the requested source frames without rerunning T."""
    status = statuses.get("T", {})
    artifact = status.get("artifact")
    if not artifact or not Path(str(artifact)).is_file():
        return {"status": "T_ARTIFACT_UNAVAILABLE"}
    try:
        with np.load(Path(str(artifact)), allow_pickle=False) as z:
            frame_indices = np.asarray(z["frame_indices"], dtype=np.int64)
            uv = np.asarray(z["uv"], dtype=np.float32)
            visibility = np.asarray(z["visibility"], dtype=bool)
        index = {int(frame): int(i) for i, frame in enumerate(frame_indices.tolist())}
        rectangle = manifest.get("roi", {}).get("suggested_rectangle_xyxy") or [110, 0, 370, 359]
        x0, y0, x1, y1 = (float(x) for x in rectangle)
        pairs = ((487, 488), (488, 498), (487, 498))
        result: dict[str, Any] = {
            "status": "T_ARRAYS_VERIFIED",
            "artifact": str(artifact),
            "frame_to_array_index": {str(frame): index.get(frame) for frame in (487, 488, 498)},
            "roi_rectangle_xyxy": [x0, y0, x1, y1],
            "pairs": [],
        }
        for first, second in pairs:
            if first not in index or second not in index:
                result["pairs"].append({"first_frame": first, "second_frame": second, "status": "FRAME_MISSING"})
                continue
            ia, ib = index[first], index[second]
            delta = np.linalg.norm(uv[ib] - uv[ia], axis=-1)
            finite = np.isfinite(uv[ia]).all(axis=-1) & np.isfinite(uv[ib]).all(axis=-1)
            roi = (
                finite
                & (uv[ia, :, 0] >= x0)
                & (uv[ia, :, 0] < x1)
                & (uv[ia, :, 1] >= y0)
                & (uv[ia, :, 1] < y1)
            )
            result["pairs"].append({
                "first_frame": first,
                "second_frame": second,
                "first_array_index": ia,
                "second_array_index": ib,
                "arrays_exact_equal": bool(np.array_equal(uv[ia], uv[ib], equal_nan=True)),
                "first_visibility_count": int(np.count_nonzero(visibility[ia])),
                "second_visibility_count": int(np.count_nonzero(visibility[ib])),
                "all_points": _motion_stats(delta, finite),
                "rough_roi": _motion_stats(delta, roi),
            })
        return result
    except (OSError, KeyError, ValueError) as exc:
        return {"status": "T_ARRAY_READ_FAILED", "error": f"{type(exc).__name__}: {exc}"}


def _write_report(output: Path, manifest: Mapping[str, Any], statuses: Mapping[str, Any]) -> None:
    rows = list(csv.DictReader((output / "frame_layer_counts_O.csv").open(encoding="utf-8")))
    target_rows = [row for row in rows if int(row["source_frame_index"]) in TARGET_FRAMES]
    history = manifest.get("history_evaluation", {})
    r_status = statuses.get("R", {})
    t_status = statuses.get("T", {})
    geometry_meta = {}
    geometry_meta_path = output / "conditions" / "R_geometry.json"
    if geometry_meta_path.is_file():
        try:
            geometry_meta = _load_json(geometry_meta_path)
        except (OSError, json.JSONDecodeError):
            geometry_meta = {}
    r_count_rows = []
    r_count_path = output / "frame_layer_counts_R.csv"
    if r_count_path.is_file():
        r_count_rows = list(csv.DictReader(r_count_path.open(encoding="utf-8")))
    r_key_counts = next((row for row in r_count_rows if int(row["source_frame_index"]) == 488), None)
    r_last_counts = r_count_rows[-1] if r_count_rows else None
    r_structure = {}
    structure_path = output / "conditions" / "R_structure.json"
    if structure_path.is_file():
        try:
            r_structure = _load_json(structure_path)
        except (OSError, json.JSONDecodeError):
            r_structure = {}
    r_summary = r_structure.get("summary", {})
    lines = [
        "# V7 04LAX fake 局部观测恢复诊断结果",
        "",
        "这是一个单案例观测诊断，不是随机评价、检测器训练或 AUROC 实验。",
        "",
        "## 输入定位",
        "",
        f"- source/role/window：`{manifest['source_id']}` / `{manifest['role']}` / `{manifest['window_id']}`。",
        f"- 原视频：`{manifest['video']['path']}`；身份状态：`{manifest['video']['status']}`。",
        f"- 查询初始化：源帧 `{manifest['query_initialization']['source_frame_index']}`，PTS `{manifest['query_initialization']['timestamp_s']:.15f}` s。",
        "- 用户时刻按保存的真实 PTS 映射，而不是按 FPS 推算：",
        f"  - `{TARGET_TIMES[0]:.6f}` s → 源帧 `{TARGET_FRAMES[0]}`，PTS `{target_rows[0]['timestamp_s']}` s；",
        f"  - `{TARGET_TIMES[1]:.6f}` s → 源帧 `{TARGET_FRAMES[1]}`，PTS `{target_rows[1]['timestamp_s']}` s。",
        f"- 两帧相邻，PTS 间隔约 `{float(target_rows[1]['timestamp_s']) - float(target_rows[0]['timestamp_s']):.9f}` s；第一张不是查询初始化帧。",
        f"- frame 488 相对初始化 `{history.get('target_frame_status', [{}, {}])[1].get('relative_to_initialization_s', 'UNKNOWN')}` s；按 0.5 s 历史窗口属于 `{history.get('target_frame_status', [{}, {}])[1].get('phase', 'UNKNOWN')}`，不是模型评分目标。模型实际目标帧来自既有 detail：`{history.get('model_frame_indices', [])}`。",
        f"- O 保存的 H/B 支撑为 H 组 `{manifest.get('support', {}).get('h_group_count')}`（成员 slot `{manifest.get('support', {}).get('h_member_slot_count')}`）、H pair `{manifest.get('support', {}).get('h_pair_count')}`、H triplet `{manifest.get('support', {}).get('h_triplet_count')}`；B 组 `{manifest.get('support', {}).get('b_group_count')}`、B triplet `{manifest.get('support', {}).get('b_triplet_count')}`。首个选定 H triplet 的共同成员仅表示保存的结构支撑，不证明这些成员位于伪造区域。",
        "",
        "## O 原有结果的逐层计数",
        "",
        "`uv_available` 是有限 UV 数，`visibility` 是 visibility 且 UV 有限，`geometry` 还要求 geometry_validity 与有限 XYZ。`local_group_member_count`（CSV 中兼容列名为 local_group_count）是落在已保存 H 保留组成员中的有效点数；`common_relation_support` 是已保存 H triplet pair 两端同时几何有效的关系数。",
        "",
        "| 源帧 | PTS(s) | query | UV | visibility | geometry | H成员有效点 | 关系支撑 | 绘图点 |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in target_rows:
        lines.append("| {source_frame_index} | {timestamp_s} | {query_count} | {uv_available_count} | {visibility_count} | {geometry_valid_count} | {local_group_count} | {common_relation_support} | {final_display_count} |".format(**row))
    lines += [
        "",
        "在源帧 487，289 个 query 全部可见、几何有效并显示；在源帧 488，三者均为 86。这个下降已经存在于 canonical NPZ 的 visibility/UV/geometry 层，不能归因于新页面的显示过滤。canonical NPZ 对不可见点将 UV 写为 NaN，所以未掩码 tracker 预测 UV 不可恢复；这不等于证明 tracker 没有预测输出。",
        "",
        "固定集合（源帧 487 首次核验时刻可见的 289 个 O query ID）在源帧 488 中仍有 86 个同时通过 visibility/geometry；这只是保存 ID 的状态统计，不证明物理对应必然正确。",
        "",
        "## R/T",
        "",
        f"- R：`{r_status.get('status')}`。从源帧 `{r_status.get('query_start_frame', 'UNKNOWN')}`（PTS `{r_status.get('query_start_pts_s', 'UNKNOWN')}`）重新以 17×17/289 点查询；R 的 `R_geometry.npz` 已按帧、PTS、query 数、UV/visibility/XYZ shape 复核并复用。Depth Pro / Open3D RGB-D odometry 形成了独立 R 三维状态；frame488 几何有效 `{r_key_counts.get('geometry_valid_count') if r_key_counts else 'UNKNOWN'}`，末帧 518 几何有效 `{r_last_counts.get('geometry_valid_count') if r_last_counts else 'UNKNOWN'}`，逐帧见 `frame_layer_counts_R.csv`。R 结构状态为 `{r_status.get('structure_status', 'NOT_COMPUTED')}`，只使用 R 自己的 ID、历史尺度、H 分组与 triplet，不连接 O 旧 ID。",
        f"- R 几何元数据：depth `{geometry_meta.get('depth', 'UNKNOWN')}`, pose `{geometry_meta.get('pose', 'UNKNOWN')}`, pose convention `{geometry_meta.get('pose_convention', 'UNKNOWN')}`；复用耗时本轮为 `{r_status.get('geometry_elapsed_s', 'UNKNOWN')}` s（0 表示身份核验后复用，首次计算耗时保留在历史 JSON/日志）。",
        f"- T：`{t_status.get('status')}`。唯一候选为官方 CoTracker3 online（commit `{COTRACKER_SHA}`），本地官方源码和 `scaled_online.pth` 已存在时按官方 chunk API 完成二维轨迹；原始预测 UV 与 visibility 分开保存，未将不可见预测位置当 XYZ。T 当前 `geometry_status={t_status.get('geometry_status', 'UNKNOWN')}`，没有 T 三维/H/B 结果，也不与 O/R ID 连接。阻塞错误：`{t_status.get('error', '无')}`。",
        "",
        "### T 数组和绘图索引核验",
    ]
    t_motion = _t_motion_diagnostics(output, statuses, manifest)
    if t_motion.get("status") == "T_ARRAYS_VERIFIED":
        lines.extend([
            f"- T 不是面板占位：保存的官方预测数组为 `{t_motion['artifact']}`；源帧到数组索引为 `487→{t_motion['frame_to_array_index'].get('487')}`、`488→{t_motion['frame_to_array_index'].get('488')}`、`498→{t_motion['frame_to_array_index'].get('498')}`。页面 `trajectory_data.json` 保留该 NPZ 的每帧顺序，按源帧查索引后绘制，未广播初始 query。",
            "- `arrays_exact_equal` 若为 false 只证明保存的 UV 数组发生变化，不证明跟踪正确；T 无 XYZ/H/B，不能把这些二维变化解释为三维结构或伪造证据。",
            "",
            "| 源帧对 | 数组完全相同 | 全体点位移(px): n / 非零 / median / p95 / max | 粗 ROI 位移(px): n / 非零 / median / p95 / max |",
            "|---|---|---|---|",
        ])
        for pair in t_motion["pairs"]:
            if pair.get("status") == "FRAME_MISSING":
                lines.append(f"| {pair['first_frame']}→{pair['second_frame']} | FRAME_MISSING | NA | NA |")
                continue
            a, r = pair["all_points"], pair["rough_roi"]
            def stat_text(s: Mapping[str, Any]) -> str:
                return f"{s['n']} / {s['nonzero']} / {s['median']:.6f} / {s['p95']:.6f} / {s['max']:.6f}"
            lines.append(f"| {pair['first_frame']}→{pair['second_frame']} | {pair['arrays_exact_equal']} | {stat_text(a)} | {stat_text(r)} |")
        lines.append(f"- 粗 ROI 采用待用户确认的源像素框 `{t_motion['roi_rectangle_xyxy']}`；它用于观察位移，不是空间真值。T 可见数为 frame487=`{t_motion['pairs'][0].get('first_visibility_count', 'UNKNOWN')}`、frame488=`{t_motion['pairs'][0].get('second_visibility_count', 'UNKNOWN')}`、frame498=`{t_motion['pairs'][1].get('second_visibility_count', 'UNKNOWN')}`。")
    else:
        lines.append(f"- T 数组核验状态：`{t_motion.get('status')}`；未用静态网格冒充跟踪结果。错误：`{t_motion.get('error', '无')}`。")
    r_lines = [
        "",
        "## R 独立结构支撑（H）",
        "",
        f"- R 的局部历史从自己的查询起点 frame `{r_structure.get('query_start', {}).get('source_frame_index', r_status.get('query_start_frame', 'UNKNOWN'))}` / PTS `{r_structure.get('query_start', {}).get('timestamp_s', r_status.get('query_start_pts_s', 'UNKNOWN'))}` 开始，边界为 `{r_structure.get('history_boundary_s', 'UNKNOWN')}` s；历史帧与评估帧分别见 `conditions/R_structure.json`。这与 O 的 frame476 起始历史不是同一时间范围，不能当作匹配检测对照。",
        f"- R 结构身份域：`{r_structure.get('id_namespace', r_status.get('id_namespace', 'UNKNOWN'))}`；`old_condition_reused={r_structure.get('old_condition_reused', 'UNKNOWN')}`。几何来源为 `{r_structure.get('source_artifact', 'UNKNOWN')}`，结构计算只读取已保存 `R_geometry.npz`，没有再次运行 tracking/depth/pose。",
        f"- H 父 component `{r_summary.get('parent_component_count', 'UNKNOWN')}`；R 局部组总数 `{r_summary.get('group_count_total', 'UNKNOWN')}`，保留组 `{r_summary.get('h_retained_group_count', 'UNKNOWN')}`，保留成员 slot `{r_summary.get('h_retained_member_slot_count', 'UNKNOWN')}`，历史有效 pair `{r_summary.get('h_retained_pair_count', 'UNKNOWN')}`；有效 H triplet `{r_summary.get('h_valid_triplet_count', 'UNKNOWN')}`。",
        f"- H grouping validation：`{r_structure.get('grouping_validation', {}).get('all_pass', 'UNKNOWN')}`；固定历史尺度为每个 R 局部组自己的历史有效 pair 距离中位数，未复用 O 尺度。",
        f"- B 状态：`{r_structure.get('b_status', r_status.get('b_status', 'UNKNOWN'))}`。{r_structure.get('b_reason', r_status.get('b_reason', ''))} 不使用 O frame476 mask 冒充 R frame488 的分割缓存。",
        "- 用户提供的框目前仍是待确认建议 ROI；因此本轮不能声称某个 R group/pair 已落在用户所指区域。页面可在源帧 488 原像素上确认一次粗框，再人工查看 R 的橙/青/紫图层；这不是空间真值。",
        "",
        "### R 有效 triplet 与多阶表示",
        "",
        "下表列出 R 每个有效 triplet 的共同成员、关系数、三帧真实源索引/PTS、四维 S(t)、一阶和二阶量；完整数组保存在 `conditions/R_structure.json`。",
        "",
        "| triplet | local group | common track IDs | pair 数 | 源帧/PTS | history scale (m) | S(t0); S(t1); S(t2) | 一阶 | 二阶 |",
        "|---:|---:|---|---:|---|---:|---|---|---|",
    ]
    for triplet in r_structure.get("support", {}).get("triplets", []):
        states = triplet.get("states", [])
        state_text = "; ".join("[" + ", ".join(f"{float(value):.5f}" for value in state) + "]" for state in states)
        first_text = "[" + ", ".join(f"{float(value):.5f}" for value in triplet.get("first_derivative", [])) + "]"
        second_text = "[" + ", ".join(f"{float(value):.5f}" for value in triplet.get("second_derivative", [])) + "]"
        frame_text = ", ".join(f"{int(frame)} @ {float(pts):.6f}" for frame, pts in zip(triplet.get("frame_indices", []), triplet.get("timestamps_s", [])))
        r_lines.append(
            f"| {triplet.get('triplet_id')} | {triplet.get('local_group_id')} | `{triplet.get('common_track_ids')}` | {len(triplet.get('pair_ids', []))} | {frame_text} | {float(triplet.get('history_scale')):.6f} | {state_text} | {first_text} | {second_text} |"
        )
    if not r_structure.get("support", {}).get("triplets"):
        r_lines.append("| — | — | — | 0 | — | — | 无有效 triplet | — | — |")
    r_lines += [
        "",
        "R 的结构支撑只能证明新查询轨迹在这些历史/目标帧上形成了可计算的局部状态与多阶量；它不证明旧 O 轨迹跨 frame487→488 的物理对应被恢复。未参与所选组/triplet 的可见点仍保留在页面上，并与实际参与点用不同图层区分。",
    ]
    lines += r_lines + [
        "",
        "## 历史/评估阶段和筛选链",
        "",
        "`query -> 有限UV -> visibility且UV有限 -> geometry_validity且XYZ有限 -> 保存的H/B成员与pair/triplet支撑 -> 页面显示`。frame 487/488 的下降是 canonical O 数组中 visibility/UV/geometry 共同反映的保存状态；这三个布尔层不是三个独立失败证据。frame 488 在首个 0.5 s 历史范围内，而既有模型评分目标从更晚的 frame 491 等开始。",
        "",
        "## ROI 与显示边界",
        "",
        f"用户截图只用于给出粗略建议框，当前 ROI 状态为 `{manifest.get('roi', {}).get('status')}`，建议源像素框 `{manifest.get('roi', {}).get('suggested_rectangle_xyxy')}`；它不是空间真值。页面提供源帧 488 原图、原像素坐标读数、框修改和一次性下载确认 JSON。用户确认前没有 ROI 内外统计；固定 query ID 后续状态与当前落框数量也不会混为同一指标。",
        "",
        "页面：`review/index.html`；访问：",
        "```bash",
        f"python3 -m http.server 8765 --bind 127.0.0.1 --directory {output}",
        "# 浏览器打开 http://127.0.0.1:8765/review/",
        "```",
        "",
        "本结果不运行冻结检测器、不训练、不计算 AUROC，也不把单案例的点消失解释为伪造导致的跟踪失败。R 的前三维状态形成不等于恢复了 O 旧轨迹跨失踪事件的物理对应；T 仅为官方 online 二维替代跟踪诊断。",
    ]
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_trajectory_data(output: Path, paths: Mapping[str, Path]) -> str:
    """Export only review-sized UV/visibility arrays for browser inspection."""
    conditions: dict[str, Any] = {}
    for condition, path in paths.items():
        with np.load(path, allow_pickle=False) as z:
            conditions[condition] = _trajectory_entry(
                z["frame_indices"],
                z["timestamps_s"],
                z["uv"],
                z["visibility"],
                z["geometry_validity"] if "geometry_validity" in z.files else None,
            )
    target = output / "review" / "trajectory_data.json"
    _write_json(target, {"conditions": conditions, "note": "review payload only; IDs are condition-local and UV null means no finite prediction was saved"})
    return str(Path("review/trajectory_data.json"))


def render_existing(output: Path) -> dict[str, Any]:
    """Rebuild review JSON/HTML/report and R H support without running O/R/T."""
    case_path = output / "case_data.json"
    manifest_path = output / "case_manifest.json"
    if not case_path.is_file() or not manifest_path.is_file():
        raise FileNotFoundError("existing case_data.json and case_manifest.json are required")
    case = _load_json(case_path)
    manifest = dict(_load_json(manifest_path))
    statuses = case.get("conditions", {})
    if statuses.get("R", {}).get("geometry_artifact"):
        r_structure = _build_r_structure(output)
        statuses["R"] = {
            **statuses.get("R", {}),
            **r_structure["status"],
            "rows": r_structure["rows"],
        }
    paths: dict[str, Path] = {}
    o_artifact = manifest.get("artifact")
    if o_artifact and Path(str(o_artifact)).is_file():
        paths["O"] = Path(str(o_artifact))
    for condition in ("R", "T"):
        status = statuses.get(condition, {})
        artifact = status.get("geometry_artifact") or status.get("artifact")
        if artifact and Path(str(artifact)).is_file():
            paths[condition] = Path(str(artifact))
    if "O" not in paths:
        raise FileNotFoundError("canonical O artifact is unavailable")
    manifest["trajectory_data"] = _write_trajectory_data(output, paths)
    _write_json(manifest_path, manifest)
    case["manifest"] = manifest
    _write_json(case_path, case)
    _html_page(output, manifest, statuses)
    _write_report(output, manifest, statuses)
    return {"output": str(output), "conditions": statuses, "trajectory_data": manifest["trajectory_data"]}


def run(output: Path, *, run_requery: bool = True, run_cotracker: bool = False) -> dict[str, Any]:
    started = time.perf_counter()
    output.mkdir(parents=True, exist_ok=True)
    manifest, o_rows = _load_o(output)
    (output / "review" / "screenshots").mkdir(parents=True, exist_ok=True)
    with np.load(PARTICLE, allow_pickle=False) as z:
        decoded = _decode_exact(FAKE_VIDEO, [int(x) for x in z["frame_indices"]])
        o_uv, o_vis, o_geo, o_xyz = z["uv"], z["visibility"], z["geometry_validity"], z["xyz"]
    manifest["source_frame_488_image"] = _save_source_frame(output, decoded, 488)
    selected_triplet = manifest.get("support", {}).get("selected_triplet") or {}
    selected_pairs = selected_triplet.get("pair_member_slots", [])
    manifest["screenshots"] = _save_pngs(output, decoded, o_rows, o_uv, o_vis, "O", geometry=o_geo, pair_slots=selected_pairs)
    media = output / "review" / "media"
    media.mkdir(parents=True, exist_ok=True)
    for name, source in (("04LAX_fake.mp4", FAKE_VIDEO), ("04LAX_real_auxiliary.mp4", REAL_VIDEO)):
        target = media / name
        if source.exists() and not target.exists():
            target.symlink_to(source)
    manifest["media"] = {"fake": "review/media/04LAX_fake.mp4", "real_auxiliary": "review/media/04LAX_real_auxiliary.mp4" if REAL_VIDEO.exists() else None}
    _write_json(output / "roi_mapping.json", {
        "status": "PENDING_USER_CONFIRMATION",
        "source": "source frame 488 pixel-space suggestion; conversation screenshots only guided the rough extent",
        "source_image_size_hw": manifest["image_size_hw"],
        "rectangle_xyxy": None,
        "suggested_rectangle_xyxy": manifest["roi"].get("suggested_rectangle_xyxy"),
        "mapping_note": "Suggested rough source-pixel box is not a mask or spatial ground truth; user must confirm or edit it once in review/index.html.",
        "no_roi_counts_emitted": True,
    })
    _write_csv(output / "frame_layer_counts_O.csv", o_rows)
    statuses: dict[str, Any] = {"O": {"status": "REUSED_CANONICAL_289", "rows": o_rows, "screenshots": manifest["screenshots"]}}
    if run_requery:
        statuses["R"] = _run_requery(output)
        if statuses["R"].get("artifact"):
            with np.load(statuses["R"]["artifact"], allow_pickle=False) as z:
                statuses["R"]["rows"] = list(csv.DictReader((output / "frame_layer_counts_R.csv").open(encoding="utf-8")))
    else:
        statuses["R"] = {"status": "NOT_RUN"}
    statuses["T"] = _run_cotracker(output, decoded) if run_cotracker else {"status": "NOT_RUN", "official_repo": "https://github.com/facebookresearch/co-tracker", "source_commit": COTRACKER_SHA}
    if statuses["T"].get("artifact") and (output / "frame_layer_counts_T.csv").is_file():
        statuses["T"]["rows"] = list(csv.DictReader((output / "frame_layer_counts_T.csv").open(encoding="utf-8")))
    if statuses["R"].get("geometry_artifact"):
        r_structure = _build_r_structure(output)
        statuses["R"].update(r_structure["status"])
        statuses["R"]["rows"] = r_structure["rows"]
    trajectory_paths: dict[str, Path] = {"O": PARTICLE}
    if statuses["R"].get("artifact"):
        trajectory_paths["R"] = Path(str(statuses["R"].get("geometry_artifact") or statuses["R"]["artifact"]))
    if statuses["T"].get("artifact"):
        trajectory_paths["T"] = Path(str(statuses["T"]["artifact"]))
    manifest["trajectory_data"] = _write_trajectory_data(output, trajectory_paths)
    _write_json(output / "case_manifest.json", manifest)
    _write_json(output / "case_data.json", {"manifest": manifest, "conditions": statuses, "created_unix": time.time(), "elapsed_s": time.perf_counter() - started})
    _html_page(output, manifest, statuses)
    _write_report(output, manifest, statuses)
    rows = []
    for condition, status in statuses.items():
        if condition == "O":
            continue
        rows.append({"condition": condition, "status": status.get("status"), "elapsed_s": status.get("elapsed_s"), "error": status.get("error", "")})
    _write_csv(output / "condition_status.csv", rows)
    return {"manifest": manifest, "conditions": statuses, "elapsed_s": time.perf_counter() - started}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--no-requery", action="store_true")
    parser.add_argument("--run-cotracker", action="store_true")
    parser.add_argument("--render-existing", action="store_true", help="rebuild review files from existing O/R/T artifacts only")
    args = parser.parse_args(argv)
    if args.render_existing:
        result = render_existing(args.output)
        print(json.dumps({"output": str(args.output), "rendered_existing": True, "trajectory_data": result["trajectory_data"]}, indent=2), flush=True)
        return 0
    result = run(args.output, run_requery=not args.no_requery, run_cotracker=args.run_cotracker)
    print(json.dumps({"output": str(args.output), "elapsed_s": result["elapsed_s"], "R": result["conditions"]["R"].get("status"), "T": result["conditions"]["T"].get("status")}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
