"""Run the V7 fixed periodic re-query observation pilot.

The pilot is deliberately independent from the formal ``src`` detection
chain.  It freezes parent clips and subwindows before labels are applied,
reuses the audited BootsTAPIR/Depth Pro/Open3D implementation, and writes
only small manifests, summaries, and model records to the data disk.  O and
R never share track identities across query cohorts.
"""

from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import json
import os
import signal
import subprocess
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from sparse3d_forgery.particle_sequence import (
    CoordinateSystem,
    Handedness,
    LengthUnit,
    build_particle_sequence,
    load_particle_sequence,
    save_particle_sequence,
)
from sparse3d_forgery.video_input import DecodedVideoSample, VideoSource, decode_video

from research_tools.v7.local_organization_probe.grouping import (
    build_local_groups,
    build_local_support,
    rebuild_components_fast,
)
from research_tools.v7.multi_order_sequence_probe.model import (
    EPOCHS,
    LEARNING_RATE,
    SEEDS,
    WEIGHT_DECAY,
    FeatureStandardizer,
    fit_standardizer,
    make_batch,
    model_state,
    parameter_count,
    score_batch,
    source_class_weights,
    train_one,
)
from research_tools.v7.multi_order_sequence_probe.representation import (
    condition_feature_matrix,
)
from research_tools.v7.observation_density_diagnostic import run_diagnostic as diagnostic


REPO_ROOT = Path(__file__).resolve().parents[3]
DATA_ROOT = Path("/root/autodl-tmp/data/sparse_3d_forgery_detection")
BASE_ROOT = DATA_ROOT / "derived/v7_activityforensics_paired_second_order_pilot_v1"
OUTPUT_ROOT = DATA_ROOT / "derived/v7_activityforensics_periodic_requery_pilot_v1"
FRONTEND_BUDGET_S = 3600.0
TRAIN_BUDGET_S = 900.0
BOOTSTRAP_SEED = 20260909
BOOTSTRAP_REPLICATES = 10_000
OFFSETS_S = (0.0, 0.5, 1.0)
TARGET_OFFSETS_S = (0.5, 0.6, 0.7, 0.8, 0.9)
MODES = ("O", "R")
BASE_CONDITIONS = ("SET_A", "RAW_SEQ", "MULTI_ORDER_SEQ")
CONDITIONS = tuple(f"{mode}_{name.replace('MULTI_ORDER_SEQ', 'MULTI').replace('SET_A', 'SET').replace('RAW_SEQ', 'RAW')}" for mode in MODES for name in BASE_CONDITIONS)
MAX_PARENT_ATTEMPTS = 2
STOP_REQUESTED = False


def _safe(value: str) -> str:
    return "".join(c if c.isalnum() or c in "._-" else "_" for c in value)


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    return value


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(_jsonable(value), indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_head() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()


class Budget:
    """Persist one additive clock per stage without double-counting resume."""

    def __init__(self, root: Path, name: str, budget_s: float) -> None:
        self.path = root / "state" / f"{name}_budget.json"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        old = json.loads(self.path.read_text(encoding="utf-8")) if self.path.is_file() else {}
        if old and abs(float(old.get("budget_s", budget_s)) - float(budget_s)) > 1e-9:
            raise ValueError(f"{name.upper()}_BUDGET_ARGUMENT_MISMATCH")
        self.before = float(old.get("cumulative_s", 0.0))
        self.budget_s = float(budget_s)
        self.started_monotonic = time.monotonic()
        self.started_unix = time.time()

    def elapsed(self) -> float:
        return self.before + max(0.0, time.monotonic() - self.started_monotonic)

    def remaining(self) -> float:
        return max(0.0, self.budget_s - self.elapsed())

    def save(self, stop_reason: str | None = None, **extra: Any) -> None:
        _atomic_json(self.path, {
            "budget_s": self.budget_s,
            "elapsed_before_this_process_s": self.before,
            "process_start_unix": self.started_unix,
            "process_elapsed_s": max(0.0, time.monotonic() - self.started_monotonic),
            "cumulative_s": self.elapsed(),
            "stop_reason": stop_reason,
            **extra,
        })


def _progress(root: Path, stage: str, status: str, completed: int, total: int, budget: Budget | None = None, **extra: Any) -> None:
    _atomic_json(root / "progress.json", {
        "stage": stage,
        "status": status,
        "completed": int(completed),
        "total": int(total),
        "updated_unix": time.time(),
        "cumulative_elapsed_s": budget.elapsed() if budget else None,
        **extra,
    })


def _completed_window_count(results: Mapping[str, Mapping[str, Any]]) -> int:
    return sum(item.get("status") == "FRONTEND_COMPLETE" for item in results.values())


def _persist_parent_completion(
    root: Path,
    result_path: Path,
    results: dict[str, dict[str, Any]],
    generated: Sequence[Mapping[str, Any]],
    *,
    parent_id: str,
    parent_elapsed_s: float,
    budget: Budget,
    parent_completed: int,
    total_windows: int,
) -> float:
    """Atomically persist a completed parent and its matching progress state."""

    for item in generated:
        row = dict(item)
        row["parent_elapsed_s"] = float(parent_elapsed_s)
        results[str(row["window_id"])] = row
    _atomic_json(result_path, list(sorted(results.values(), key=lambda item: str(item["window_id"]))))
    budget.save(None, last_parent=str(parent_id), last_parent_elapsed_s=float(parent_elapsed_s))
    _progress(
        root,
        "frontend",
        "RUNNING",
        _completed_window_count(results),
        total_windows,
        budget=budget,
        completed_parent_count=int(parent_completed),
        current_parent=str(parent_id),
    )
    return float(parent_elapsed_s)


def _window_label(role: str, start_s: float, end_s: float, segments: Sequence[Mapping[str, Any]]) -> tuple[int | None, str]:
    if role == "real":
        return 0, "REAL_NEGATIVE"
    overlaps = [(max(start_s, float(item["start_s"])), min(end_s, float(item["end_s"]))) for item in segments]
    overlaps = [(left, right) for left, right in overlaps if right > left]
    if not overlaps:
        return None, "OUTSIDE_ANNOTATED_MANIPULATION"
    union = sorted(overlaps)
    cursor = union[0][0]
    covered = union[0][1] - union[0][0]
    for left, right in union[1:]:
        if left > cursor:
            cursor = left
            covered += right - left
        else:
            covered += max(0.0, right - cursor)
            cursor = max(cursor, right)
    fully_inside = any(start_s >= float(item["start_s"]) - 1e-9 and end_s <= float(item["end_s"]) + 1e-9 for item in segments)
    return (1, "FAKE_MANIPULATION") if fully_inside else (None, "BOUNDARY_MIXED")


def _timeline_for(pair: Mapping[str, Any], role: str) -> Mapping[str, Any]:
    return pair["real_timeline" if role == "real" else "fake_timeline"]


def _choose_parents() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    pairs = json.loads((BASE_ROOT / "manifests/selected_pairs.json").read_text(encoding="utf-8"))
    windows = json.loads((BASE_ROOT / "manifests/window_manifest.json").read_text(encoding="utf-8"))
    by_key = {(str(row["source_id"]), str(row["role"])): row for row in windows if str(row["kind"]) == "MANIP" and abs(float(row["anchor_fraction"]) - 0.5) < 1e-9}
    parents: list[dict[str, Any]] = []
    subwindows: list[dict[str, Any]] = []
    for pair in pairs:
        source = str(pair["source_id"])
        for role in ("real", "fake"):
            source_row = by_key.get((source, role))
            if source_row is None:
                parents.append({"parent_id": f"{source}::{role}::MANIP50", "source_id": source, "role": role, "status": "PARENT_MISSING"})
                continue
            timeline = _timeline_for(pair, role)
            timestamps = np.asarray(timeline["timestamps_s"], dtype=np.float64)
            frame_indices = np.asarray(timeline["frame_indices"], dtype=np.int64)
            t0 = float(source_row["timestamps_s"][0])
            parent_mask = (timestamps >= t0 - 1e-9) & (timestamps <= t0 + 2.0 + 1e-9)
            parent_frames = frame_indices[parent_mask]
            parent_times = timestamps[parent_mask]
            parent_id = f"{source}::{role}::MANIP50"
            parent = {
                "parent_id": parent_id,
                "source_id": source,
                "pair_id": str(pair["pair_id"]),
                "role": role,
                "kind": "MANIP",
                "selection_rule": "middle MANIP anchor_fraction=0.50 in frozen window manifest",
                "selection_source_window_id": str(source_row["window_id"]),
                "video_path": str(pair["real_video_path"] if role == "real" else pair["fake_video_path"]),
                "video_sha256": str(timeline["sha256"]),
                "cohort_start_s": t0,
                "cohort_start_frame_index": int(parent_frames[0]) if parent_frames.size else None,
                "parent_frame_indices": parent_frames.tolist(),
                "parent_timestamps_s": parent_times.tolist(),
                "parent_frame_count": int(parent_frames.size),
                "timeline_frame_count": int(timestamps.size),
                "timeline_sha256": str(timeline["sha256"]),
                "all_manipulation_segments": list(pair.get("all_manipulation_segments", [])),
                "status": "PLANNED" if parent_frames.size >= 2 else "PARENT_TOO_SHORT",
            }
            parents.append(parent)
            for offset in OFFSETS_S:
                desired_start = t0 + float(offset)
                start_pos = int(np.searchsorted(timestamps, desired_start - 1e-9, side="left"))
                end_pos = int(np.searchsorted(timestamps, desired_start + 1.0 - 1e-9, side="left"))
                indices = frame_indices[start_pos:end_pos]
                stamps = timestamps[start_pos:end_pos]
                start_actual = float(stamps[0]) if stamps.size else desired_start
                end_actual = float(stamps[-1]) if stamps.size else desired_start
                label, category = _window_label(role, start_actual, start_actual + 1.0, parent["all_manipulation_segments"])
                window_id = f"{parent_id}::b{int(round(offset * 10)):02d}"
                subwindows.append({
                    "window_id": window_id,
                    "parent_id": parent_id,
                    "source_id": source,
                    "pair_id": str(pair["pair_id"]),
                    "role": role,
                    "kind": "MANIP",
                    "offset_s": float(offset),
                    "window_length_s": 1.0,
                    "cohort_start_s": t0,
                    "interval_start_s": start_actual,
                    "interval_end_s": end_actual,
                    "frame_indices": indices.tolist(),
                    "timestamps_s": stamps.tolist(),
                    "annotation_category": category,
                    "label": label,
                    "label_applied_after_sampling": True,
                    "status": "PLANNED" if indices.size >= 5 else "SUBWINDOW_TOO_SHORT",
                })
    return parents, subwindows


def prepare(root: Path) -> dict[str, Any]:
    parents, subwindows = _choose_parents()
    root.joinpath("manifests").mkdir(parents=True, exist_ok=True)
    _atomic_json(root / "manifests/parents.json", parents)
    _atomic_json(root / "manifests/subwindows.json", subwindows)
    _atomic_json(root / "manifests/input_manifest.json", {
        "source_manifest": str(BASE_ROOT / "manifests/window_manifest.json"),
        "source_manifest_sha256": _sha256(BASE_ROOT / "manifests/window_manifest.json"),
        "rows": subwindows,
        "population": {"frozen_sources": len({str(x["source_id"]) for x in parents}), "parent_count": len(parents), "subwindow_count": len(subwindows)},
    })
    source_order = [str(pair["source_id"]) for pair in json.loads((BASE_ROOT / "manifests/selected_pairs.json").read_text(encoding="utf-8"))]
    _atomic_json(root / "manifests/execution_plan.json", {
        "source_order": source_order,
        "parent_selection": "middle MANIP slot, anchor_fraction=0.50, before labels and results",
        "offsets_s": list(OFFSETS_S),
        "window_length_s": 1.0,
        "parent_length_s": 2.0,
        "parent_count": len(parents),
        "subwindow_count": len(subwindows),
        "no_score_or_support_selection": True,
    })
    protocol(root)
    return {"parents": len(parents), "subwindows": len(subwindows), "planned_sources": len(set(source_order))}


def protocol(root: Path) -> None:
    _atomic_json(root / "protocol.json", {
        "protocol_id": "v7-periodic-requery-pilot-v1",
        "git_head": _git_head(),
        "question": "fixed periodic re-query versus continuous old-query observation for five-time structural support",
        "population": "16 frozen source pairs; middle MANIP parent slot; two-second parent and offsets 0/0.5/1.0",
        "parent_selection": "deterministic middle MANIP slot from the existing 25/50/75 windows; no score, visibility, or manual selection",
        "O": "one 17x17=289 grid initialized at parent t0 and tracked through the parent; each subwindow slices this same cohort",
        "R": "independent 17x17=289 grid initialized at each nonzero subwindow start; b=0 uses the same initialization as O when identity is exact; no cross-cohort identity or derivative",
        "geometry": {"tracker": "online_bootstapir", "tracker_source_sha": diagnostic.TRACKER_SHA, "tracker_checkpoint": str(diagnostic.TAPNET_CHECKPOINT), "depth": "apple_depth_pro", "depth_source_sha": diagnostic.DEPTH_SHA, "depth_checkpoint": str(diagnostic.DEPTH_CHECKPOINT), "pose": "Open3D 0.19 adjacent RGB-D odometry", "shared_within_parent": True, "coordinates": "first_camera_world, right-handed right/down/forward, meters", "depth_semantics": "optical-axis z-depth", "query_count": 289, "process_size": 256},
        "support": {"history_s": 0.5, "target_offsets_s": list(TARGET_OFFSETS_S), "same_h_grouping_and_scale_per_mode": True, "minimum_common_members": 3, "missing": "preserved; no interpolation or zero fill"},
        "labels": {"real": 0, "fake_fully_inside_union": 1, "boundary_mixed": "descriptive_only", "outside": "descriptive_only", "applied_after_sampling": True},
        "models": {"conditions": list(CONDITIONS), "base_conditions": list(BASE_CONDITIONS), "seeds": list(SEEDS), "epochs": EPOCHS, "optimizer": "Adam", "learning_rate": LEARNING_RATE, "weight_decay": WEIGHT_DECAY, "source_disjoint": True, "standardization": "fold training rows only", "weighted_bce": "equal source and class total weight"},
        "evaluation": {"primary": "R_SET-O_SET source-equal AUROC difference on paired common support", "secondary": ["R_RAW-O_RAW", "R_MULTI-O_MULTI", "R_MULTI-R_RAW"], "bootstrap": {"unit": "source", "seed": BOOTSTRAP_SEED, "replicates": BOOTSTRAP_REPLICATES}, "threshold": "logit >= 0", "pooled": "auxiliary"},
        "budgets": {"frontend_s": FRONTEND_BUDGET_S, "training_s": TRAIN_BUDGET_S, "postprocess_excluded": True},
        "boundaries": ["research development pilot", "no formal src modification", "no new tracker/depth/segmentation", "no ID matching across cohorts", "no spatial ground truth claim", "not a sealed test"],
    })


def _load_parents(root: Path) -> list[dict[str, Any]]:
    return json.loads((root / "manifests/parents.json").read_text(encoding="utf-8"))


def _load_subwindows(root: Path) -> list[dict[str, Any]]:
    return json.loads((root / "manifests/subwindows.json").read_text(encoding="utf-8"))


def _sequence_identity_ok(prefix: Path, row: Mapping[str, Any], mode: str) -> bool:
    try:
        sequence = load_particle_sequence(prefix)
        if not np.array_equal(sequence.frame_indices, np.asarray(row["frame_indices"], dtype=np.int64)):
            return False
        if not np.allclose(sequence.timestamps_s, np.asarray(row["timestamps_s"], dtype=np.float64), atol=1e-7, rtol=0):
            return False
        provenance = sequence.provenance
        return str(provenance.get("query_cohort", "")) == mode and int(sequence.num_tracks) == 289
    except (OSError, ValueError, KeyError, TypeError):
        return False


def _sequence_manifest_identity_ok(prefix: Path, frame_indices: Sequence[int], timestamps_s: Sequence[float], *, source_video_id: str, mode: str, expected_lineage: Mapping[str, Any] | None = None) -> bool:
    try:
        sequence = load_particle_sequence(prefix)
        if not np.array_equal(sequence.frame_indices, np.asarray(frame_indices, dtype=np.int64)):
            return False
        if not np.allclose(sequence.timestamps_s, np.asarray(timestamps_s, dtype=np.float64), atol=1e-7, rtol=0):
            return False
        if sequence.source_video_id != source_video_id or int(sequence.num_tracks) != 289:
            return False
        lineage = sequence.lineage
        provenance = sequence.provenance
        if str(lineage.get("source_id", "")) != source_video_id.split("::", 1)[0] or str(provenance.get("query_cohort", "")) != mode:
            return False
        if expected_lineage and any(str(lineage.get(key, "")) != str(value) for key, value in expected_lineage.items()):
            return False
        return True
    except (OSError, ValueError, KeyError, TypeError):
        return False


def _cached_parent_outputs(parent: Mapping[str, Any], subwindows: Sequence[Mapping[str, Any]], root: Path) -> list[dict[str, Any]] | None:
    """Return reconstructed results only when every required cache is valid."""

    parent_id = str(parent["parent_id"])
    parent_prefix = root / "particles" / _safe(parent_id) / "O_parent"
    expected_video_id = f"{parent['source_id']}::{parent['role']}"
    expected_lineage = {"source_id": parent["source_id"], "pair_id": parent["pair_id"], "role": parent["role"], "parent_id": parent_id}
    if not _sequence_manifest_identity_ok(parent_prefix, parent["parent_frame_indices"], parent["parent_timestamps_s"], source_video_id=expected_video_id, mode="O", expected_lineage=expected_lineage):
        return None
    parent_sequence = load_particle_sequence(parent_prefix)
    outputs: list[dict[str, Any]] = []
    for row in subwindows:
        offset = float(row["offset_s"])
        reuse_b0 = abs(offset) < 1e-9
        prefix_r = parent_prefix if reuse_b0 else root / "particles" / _safe(parent_id) / f"R_{_safe(str(row['window_id']))}"
        if not reuse_b0 and not _sequence_identity_ok(prefix_r, row, "R"):
            return None
        loaded_r = parent_sequence if reuse_b0 else load_particle_sequence(prefix_r)
        outputs.append({
            **dict(row),
            "status": "FRONTEND_COMPLETE",
            "o_sequence_prefix": str(parent_prefix),
            "r_sequence_prefix": str(prefix_r),
            "r_status": "REUSED_O_B0" if reuse_b0 else "FRONTEND_COMPLETE",
            "o_frame_count": int(parent_sequence.num_frames),
            "r_frame_count": int(loaded_r.num_frames),
            "o_query_count": int(parent_sequence.num_tracks),
            "r_query_count": int(loaded_r.num_tracks),
            "o_geometry_valid_fraction": float(np.mean(parent_sequence.geometry_validity)),
            "r_geometry_valid_fraction": float(np.mean(loaded_r.geometry_validity)),
            "shared_geometry": True,
            "parent_frame_count": int(parent_sequence.num_frames),
            "parent_pts_start_s": float(parent_sequence.timestamps_s[0]),
            "parent_pts_end_s": float(parent_sequence.timestamps_s[-1]),
            "fixed_focal_px": None,
            "pose_pair_valid_fraction": None,
            "pose_valid_fraction": None,
            "elapsed_s": 0.0,
            "cache_reused": True,
        })
    return outputs


def _slice_decoded(decoded: DecodedVideoSample, positions: Sequence[int], sample_id: str) -> DecodedVideoSample:
    return DecodedVideoSample(sample_id=sample_id, source_video_id=decoded.source_video_id, frames=tuple(decoded.frames[int(pos)] for pos in positions))


def _positions_for_frames(parent_indices: np.ndarray, frame_indices: Sequence[int]) -> list[int]:
    lookup = {int(value): index for index, value in enumerate(parent_indices.tolist())}
    return [lookup[int(value)] for value in frame_indices]


def _build_sequence(decoded: DecodedVideoSample, *, uv: np.ndarray, visibility: np.ndarray, xyz: np.ndarray, geometry_valid: np.ndarray, row: Mapping[str, Any], mode: str, parent_id: str, cohort_start_s: float) -> Any:
    # Geometry helpers may calculate in float64 (notably during inverse
    # intrinsics and pose multiplication), while the logical ParticleSequence
    # contract requires float32 XYZ.  Convert only at this construction
    # boundary; preserve NaN missing values and never repair invalid geometry.
    xyz_array = np.asarray(xyz)
    geometry_array = np.asarray(geometry_valid, dtype=np.bool_)
    if xyz_array.shape[:2] != geometry_array.shape or xyz_array.shape[-1:] != (3,):
        raise ValueError("xyz and geometry_validity shapes are incompatible before ParticleSequence construction")
    xyz_array = xyz_array.astype(np.float32, copy=False)
    if np.any(geometry_array & ~np.all(np.isfinite(xyz_array), axis=-1)):
        raise ValueError("geometry-valid XYZ must remain finite before ParticleSequence construction")
    if np.any(~geometry_array & ~np.all(np.isnan(xyz_array), axis=-1)):
        raise ValueError("geometry-invalid XYZ must remain NaN before ParticleSequence construction")
    return build_particle_sequence(
        decoded,
        track_ids=np.arange(289, dtype=np.int64),
        xyz=xyz_array,
        uv=np.asarray(uv, dtype=np.float32),
        visibility=np.asarray(visibility, dtype=bool),
        geometry_validity=geometry_array,
        coordinate_system=CoordinateSystem(frame_name="first_camera_world", handedness=Handedness.RIGHT, axis_directions=("right", "down", "forward"), length_unit=LengthUnit.METER, camera_motion_compensated=True, normalization={}),
        lineage={"dataset": "ActivityForensics+Charades", "official_split": "train", "source_id": str(row["source_id"]), "pair_id": str(row["pair_id"]), "role": str(row["role"]), "parent_id": parent_id, "window_id": str(row["window_id"]), "query_cohort": mode},
        provenance={"tracker": "online_bootstapir", "tracker_source_sha": diagnostic.TRACKER_SHA, "depth": "apple_depth_pro", "depth_source_sha": diagnostic.DEPTH_SHA, "depth_semantics": "optical_axis_z_depth", "pose": "open3d_rgbd_odometry", "pose_convention": "target_camera_from_source_camera_inverted", "process_size": 256, "query_grid_size": 17, "query_count": 289, "query_cohort": mode, "cohort_start_s": float(cohort_start_s), "parent_id": parent_id, "cross_cohort_identity": False, "causal_execution": False, "causal_training_reason": "offline parent-window geometry and fixed time support; no target construction"},
    )


def _frontend_parent(parent: Mapping[str, Any], subwindows: Sequence[Mapping[str, Any]], tracker: Any, depth_runner: Any, root: Path) -> list[dict[str, Any]]:
    parent_id = str(parent["parent_id"])
    source_path = Path(str(parent["video_path"]))
    if str(parent.get("status")) != "PLANNED" or not source_path.is_file():
        return [{"window_id": str(row["window_id"]), "status": "SOURCE_MISSING" if not source_path.is_file() else "PARENT_UNAVAILABLE", "error": "parent is not runnable"} for row in subwindows]
    cached = _cached_parent_outputs(parent, subwindows, root)
    if cached is not None:
        return cached
    import torch
    frame_indices = np.asarray(parent["parent_frame_indices"], dtype=np.int64)
    decoded = decode_video(VideoSource(sample_id=f"v7-periodic-{parent_id}", source_video_id=f"{parent['source_id']}::{parent['role']}", source_locator=source_path), frame_indices.tolist())
    depths, focals, frame_depth_valid = depth_runner.infer(decoded)
    intrinsics, fixed_focal_px = __import__("scripts.run_v7_explicit_geometry_frontend", fromlist=["causal_first_frame_intrinsics"]).causal_first_frame_intrinsics(depths, focals)
    frontend = __import__("scripts.run_v7_explicit_geometry_frontend", fromlist=["rgbd_odometry", "accumulate_world_from_camera", "sample_depth_at_uv", "world_xyz"])
    adjacent, pair_valid, information = frontend.rgbd_odometry(decoded, depths, intrinsics)
    world_from_camera, pose_valid = frontend.accumulate_world_from_camera(adjacent, pair_valid)
    outputs: list[dict[str, Any]] = []
    parent_dir = root / "particles" / _safe(parent_id)
    parent_prefix = parent_dir / "O_parent"
    parent_cache_valid = _sequence_manifest_identity_ok(parent_prefix, parent["parent_frame_indices"], parent["parent_timestamps_s"], source_video_id=f"{parent['source_id']}::{parent['role']}", mode="O", expected_lineage={"source_id": parent["source_id"], "pair_id": parent["pair_id"], "role": parent["role"], "parent_id": parent_id})
    if not parent_cache_valid:
        uv_o, visibility_o = tracker.track(decoded)
        observation = visibility_o & frame_depth_valid[:, None]
        sampled, sampled_valid = frontend.sample_depth_at_uv(depths, uv_o)
        observation &= sampled_valid
        xyz_o, geo_o = frontend.world_xyz(uv_o, sampled, intrinsics, world_from_camera, observation, pose_valid)
        parent_row = {"window_id": parent_id, "source_id": parent["source_id"], "pair_id": parent["pair_id"], "role": parent["role"], "frame_indices": frame_indices.tolist(), "timestamps_s": decoded.timestamps_s.tolist()}
        parent_sequence = _build_sequence(decoded, uv=uv_o, visibility=visibility_o, xyz=xyz_o, geometry_valid=geo_o, row=parent_row, mode="O", parent_id=parent_id, cohort_start_s=float(parent["cohort_start_s"]))
        save_particle_sequence(parent_sequence, parent_prefix)
    else:
        parent_sequence = load_particle_sequence(parent_prefix)
        uv_o, visibility_o = np.asarray(parent_sequence.uv), np.asarray(parent_sequence.visibility)
        xyz_o, geo_o = np.asarray(parent_sequence.xyz), np.asarray(parent_sequence.geometry_validity)
    for row in subwindows:
        window_id = str(row["window_id"])
        positions = _positions_for_frames(frame_indices, row["frame_indices"])
        sub_decoded = _slice_decoded(decoded, positions, f"v7-periodic-{window_id}")
        # O is the one continuous parent cohort.  Keep its full parent
        # sequence so every subwindow has the preceding 0.5 s history used
        # by the frozen scale/group/support contract.
        prefix_o = parent_prefix
        prefix_r = root / "particles" / _safe(parent_id) / f"R_{_safe(window_id)}"
        reuse_b0 = abs(float(row["offset_s"])) < 1e-9
        if reuse_b0:
            prefix_r = prefix_o
        elif not parent_cache_valid or not _sequence_identity_ok(prefix_r, row, "R"):
            uv_r, visibility_r = tracker.track(sub_decoded)
            depths_sub = depths[positions]
            intrinsics_sub = intrinsics[positions]
            poses_sub = world_from_camera[positions]
            pose_valid_sub = pose_valid[positions]
            valid_depth_sub = frame_depth_valid[positions]
            sampled_r, sampled_valid_r = frontend.sample_depth_at_uv(depths_sub, uv_r)
            observation_r = visibility_r & sampled_valid_r & valid_depth_sub[:, None]
            xyz_r, geo_r = frontend.world_xyz(uv_r, sampled_r, intrinsics_sub, poses_sub, observation_r, pose_valid_sub)
            seq_r = _build_sequence(sub_decoded, uv=uv_r, visibility=visibility_r, xyz=xyz_r, geometry_valid=geo_r, row=row, mode="R", parent_id=parent_id, cohort_start_s=float(row["interval_start_s"]))
            save_particle_sequence(seq_r, prefix_r)
        loaded_o = parent_sequence
        loaded_r = load_particle_sequence(prefix_r)
        outputs.append({
            **dict(row),
            "status": "FRONTEND_COMPLETE",
            "o_sequence_prefix": str(prefix_o),
            "r_sequence_prefix": str(prefix_r),
            "r_status": "REUSED_O_B0" if reuse_b0 else "FRONTEND_COMPLETE",
            "o_frame_count": int(loaded_o.num_frames),
            "r_frame_count": int(loaded_r.num_frames),
            "o_query_count": int(loaded_o.num_tracks),
            "r_query_count": int(loaded_r.num_tracks),
            "o_geometry_valid_fraction": float(np.mean(loaded_o.geometry_validity)),
            "r_geometry_valid_fraction": float(np.mean(loaded_r.geometry_validity)),
            "shared_geometry": True,
            "parent_frame_count": int(decoded.frame_indices.size),
            "parent_pts_start_s": float(decoded.timestamps_s[0]),
            "parent_pts_end_s": float(decoded.timestamps_s[-1]),
            "fixed_focal_px": float(fixed_focal_px),
            "pose_pair_valid_fraction": float(np.mean(pair_valid)),
            "pose_valid_fraction": float(np.mean(pose_valid)),
            "elapsed_s": None,
        })
    del decoded, depths, uv_o, visibility_o, xyz_o, geo_o, parent_sequence
    gc = getattr(torch, "cuda", None)
    if gc is not None and gc.is_available():
        gc.empty_cache()
    return outputs


def run_frontend(root: Path, budget_s: float, resume: bool) -> dict[str, Any]:
    global STOP_REQUESTED
    budget = Budget(root, "frontend", budget_s)
    parents = _load_parents(root)
    subwindows = _load_subwindows(root)
    result_path = root / "frontend/results.json"
    results = {str(item["window_id"]): item for item in (json.loads(result_path.read_text(encoding="utf-8")) if result_path.is_file() else [])} if resume else {}
    _progress(root, "frontend", "RUNNING", sum(item.get("status") == "FRONTEND_COMPLETE" for item in results.values()), len(subwindows), budget=budget, completed_parent_count=0)
    if budget.remaining() <= 0:
        budget.save("BUDGET_EXHAUSTED")
        completed_now = _completed_window_count(results)
        _progress(root, "frontend", "BUDGET_EXHAUSTED", completed_now, len(subwindows), budget=budget, stop_reason="BUDGET_EXHAUSTED")
        return {"status": "BUDGET_EXHAUSTED", "completed": completed_now, "total": len(subwindows)}
    try:
        import torch
        torch.set_num_threads(4)
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA_UNAVAILABLE_FRONTEND_REQUIRES_GPU")
        from scripts.run_v7_explicit_geometry_frontend import DepthProRunner, OnlineBootsTapir
        print(f"periodic frontend device={torch.cuda.get_device_name(0)}", flush=True)
        tracker = OnlineBootsTapir(diagnostic.TAPNET_SOURCE, diagnostic.TAPNET_CHECKPOINT, process_size=256, grid_size=17)
        depth_runner = DepthProRunner(diagnostic.DEPTH_SOURCE, diagnostic.DEPTH_CHECKPOINT)
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        _atomic_json(root / "state/frontend_error.json", {"status": "FRONTEND_BLOCKED", "error": error})
        budget.save("FRONTEND_BLOCKED", error=error)
        completed_now = _completed_window_count(results)
        _progress(root, "frontend", "FRONTEND_BLOCKED", completed_now, len(subwindows), budget=budget, stop_reason="FRONTEND_BLOCKED", error=error)
        return {"status": "FRONTEND_BLOCKED", "error": error, "completed": completed_now, "total": len(subwindows)}
    old_handler = signal.getsignal(signal.SIGTERM)
    old_int = signal.getsignal(signal.SIGINT)
    def stop(_signal: int, _frame: Any) -> None:
        global STOP_REQUESTED
        STOP_REQUESTED = True
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    parent_completed = 0
    try:
        for parent in parents:
            parent_rows = [row for row in subwindows if str(row["parent_id"]) == str(parent["parent_id"])]
            if all(str(row["window_id"]) in results and str(results[str(row["window_id"])].get("status")) == "FRONTEND_COMPLETE" for row in parent_rows):
                parent_completed += 1
                continue
            if STOP_REQUESTED or budget.remaining() <= 0:
                reason = "STOPPED_SAFE" if STOP_REQUESTED else "BUDGET_EXHAUSTED"
                budget.save(reason)
                completed_now = sum(item.get("status") == "FRONTEND_COMPLETE" for item in results.values())
                _progress(root, "frontend", reason, completed_now, len(subwindows), budget=budget, stop_reason=reason, completed_parent_count=parent_completed)
                break
            started = time.perf_counter()
            try:
                generated = _frontend_parent(parent, parent_rows, tracker, depth_runner, root)
                parent_completed += 1
                elapsed = time.perf_counter() - started
                _persist_parent_completion(root, result_path, results, generated, parent_id=str(parent["parent_id"]), parent_elapsed_s=elapsed, budget=budget, parent_completed=parent_completed, total_windows=len(subwindows))
                print(f"periodic parent {parent_completed}/{len(parents)} {parent['parent_id']} subwindows={len(generated)} elapsed={elapsed:.1f}s remaining={budget.remaining():.1f}s", flush=True)
            except Exception as exc:
                error = {"parent_id": str(parent["parent_id"]), "status": "FRONTEND_FAILED", "error": f"{type(exc).__name__}: {exc}", "traceback": __import__("traceback").format_exc()}
                for row in parent_rows:
                    results[str(row["window_id"])] = {**dict(row), **error, "window_id": str(row["window_id"])}
                _atomic_json(result_path, list(sorted(results.values(), key=lambda item: str(item["window_id"]))))
                budget.save(None, last_parent=str(parent["parent_id"]), last_error=error["error"])
                _progress(root, "frontend", "RUNNING", _completed_window_count(results), len(subwindows), budget=budget, completed_parent_count=parent_completed, current_parent=str(parent["parent_id"]), last_error=error["error"])
                print(f"periodic parent FAILED {parent['parent_id']}: {error['error']}", flush=True)
    finally:
        signal.signal(signal.SIGTERM, old_handler)
        signal.signal(signal.SIGINT, old_int)
        try:
            del tracker, depth_runner
        except UnboundLocalError:
            pass
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
    complete = sum(item.get("status") == "FRONTEND_COMPLETE" for item in results.values())
    status = "COMPLETE" if complete == len(subwindows) else ("STOPPED_SAFE" if STOP_REQUESTED else ("BUDGET_EXHAUSTED" if budget.remaining() <= 0 else "FRONTEND_INCOMPLETE"))
    budget.save(None if status == "COMPLETE" else status)
    _progress(root, "frontend", status, complete, len(subwindows), budget=budget, completed_parent_count=parent_completed, stop_reason=None if status == "COMPLETE" else status)
    return {"status": status, "completed": complete, "total": len(subwindows), "parent_completed": parent_completed, "cumulative_s": budget.elapsed()}


def _support_for_sequence(prefix: Path, row: Mapping[str, Any], mode: str) -> dict[str, Any]:
    sequence = load_particle_sequence(prefix)
    history = np.flatnonzero(sequence.timestamps_s < float(row["interval_start_s"]) + 0.5).astype(np.int64)
    components = rebuild_components_fast(sequence.xyz, sequence.geometry_validity, history)
    grouping = build_local_groups(sequence, history, old_components=components)
    support = build_local_support(sequence, window_start_s=float(row["interval_start_s"]), grouping=grouping)
    built = condition_feature_matrix({"units": support.get("triplets", [])}, str(row["window_id"]))
    valid_units = [item for item in support.get("triplets", []) if item.get("status") == "VALID"]
    # condition_feature_matrix consumes the same support unit schema as the
    # existing multi-order runner; no B segmentation or new grouping is used.
    return {
        "mode": mode,
        "window_id": str(row["window_id"]),
        "source_id": str(row["source_id"]),
        "pair_id": str(row["pair_id"]),
        "role": str(row["role"]),
        "kind": str(row["kind"]),
        "label": row.get("label"),
        "annotation_category": str(row.get("annotation_category")),
        "offset_s": float(row["offset_s"]),
        "interval_start_s": float(row["interval_start_s"]),
        "interval_end_s": float(row["interval_end_s"]),
        "particle_prefix": str(prefix),
        "frame_indices": [int(x) for x in sequence.frame_indices],
        "timestamps_s": [float(x) for x in sequence.timestamps_s],
        "intervals_s": (np.diff(np.asarray(valid_units[0]["timestamps_s"], dtype=np.float64)) if valid_units else np.empty(4, dtype=np.float64)),
        "group_count": int(len(grouping.get("groups", []))),
        "retained_group_count": int(grouping.get("retained_group_count", 0)),
        "support_status": str(support.get("support_status")),
        "valid_unit_count": int(len(valid_units)),
        "support_reasons": sorted({str(item.get("reason")) for item in support.get("invalid_reasons", [])}),
        "grouping": grouping,
        "support": support,
        "features": built["features"],
        "permutations": built["permutations"],
    }


def features(root: Path) -> dict[str, Any]:
    subwindows = _load_subwindows(root)
    result_path = root / "frontend/results.json"
    results = {str(item["window_id"]): item for item in json.loads(result_path.read_text(encoding="utf-8"))} if result_path.is_file() else {}
    support_rows: list[dict[str, Any]] = []
    for index, row in enumerate(subwindows, 1):
        result = results.get(str(row["window_id"]), {})
        for mode, prefix_key in (("O", "o_sequence_prefix"), ("R", "r_sequence_prefix")):
            if str(result.get("status")) != "FRONTEND_COMPLETE" or not result.get(prefix_key):
                support_rows.append({**dict(row), "mode": mode, "support_status": "MISSING_FRONTEND", "valid_unit_count": 0, "features": {base: None for base in BASE_CONDITIONS}, "group_count": 0, "retained_group_count": 0, "particle_prefix": None})
                continue
            try:
                support_rows.append(_support_for_sequence(Path(str(result[prefix_key])), row, mode))
            except Exception as exc:
                support_rows.append({**dict(row), "mode": mode, "support_status": "FEATURE_FAILED", "support_error": f"{type(exc).__name__}: {exc}", "valid_unit_count": 0, "features": {base: None for base in BASE_CONDITIONS}, "group_count": 0, "retained_group_count": 0, "particle_prefix": result.get(prefix_key)})
        if index % 8 == 0:
            _progress(root, "features", "RUNNING", index, len(subwindows), completed_modes=2 * index)
    by_window = defaultdict(dict)
    for row in support_rows:
        by_window[str(row["window_id"])][str(row["mode"])] = row
    for row in support_rows:
        other = by_window[str(row["window_id"])].get("R" if row["mode"] == "O" else "O")
        row["paired_eligible"] = bool(row.get("valid_unit_count", 0) > 0 and other and other.get("valid_unit_count", 0) > 0 and row.get("label") in (0, 1) and other.get("label") in (0, 1))
    root.joinpath("support").mkdir(parents=True, exist_ok=True)
    _atomic_json(root / "support/window_support.json", support_rows)
    coverage_rows = []
    for row in support_rows:
        coverage_rows.append({key: row.get(key) for key in ("window_id", "source_id", "role", "offset_s", "mode", "label", "annotation_category", "support_status", "valid_unit_count", "group_count", "retained_group_count", "paired_eligible")})
    _write_csv(root / "support/coverage.csv", coverage_rows)
    summary = {
        "mode_rows": len(support_rows),
        "subwindow_count": len(subwindows),
        "paired_windows": sum(bool(row.get("paired_eligible")) for row in support_rows if row["mode"] == "O"),
        "o_valid_windows": sum(row["mode"] == "O" and row.get("valid_unit_count", 0) > 0 for row in support_rows),
        "r_valid_windows": sum(row["mode"] == "R" and row.get("valid_unit_count", 0) > 0 for row in support_rows),
        "o_units": sum(int(row.get("valid_unit_count", 0)) for row in support_rows if row["mode"] == "O"),
        "r_units": sum(int(row.get("valid_unit_count", 0)) for row in support_rows if row["mode"] == "R"),
        "offsets": list(OFFSETS_S),
    }
    _atomic_json(root / "support/support_summary.json", summary)
    _progress(root, "features", "COMPLETE", len(subwindows), len(subwindows), **summary)
    return summary


def _feature_rows(root: Path, mode: str, base_condition: str) -> list[dict[str, Any]]:
    rows = json.loads((root / "support/window_support.json").read_text(encoding="utf-8"))
    return [row for row in rows if str(row["mode"]) == mode and int(row.get("valid_unit_count", 0)) > 0 and row.get("features", {}).get(base_condition) is not None]


def _model_from_record(base_condition: str, record: Mapping[str, Any], device: str) -> Any:
    import torch
    from research_tools.v7.multi_order_sequence_probe.model import SequenceMLP, SetAModel
    model = SetAModel() if base_condition == "SET_A" else SequenceMLP()
    state = {name: torch.as_tensor(value, dtype=model.state_dict()[name].dtype) for name, value in record["state_dict"].items()}
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def _standardizer(record: Mapping[str, Any]) -> FeatureStandardizer:
    item = record["standardization"]
    return FeatureStandardizer(str(item["condition"]), np.asarray(item["mean"], dtype=np.float64), np.asarray(item["scale"], dtype=np.float64), tuple(int(x) for x in item.get("zero_variance_dimensions", [])))


def _fit_rows(rows: Sequence[Mapping[str, Any]], base: str) -> tuple[list[dict[str, Any]], Any, dict[str, Any]]:
    values = [{**dict(row), "features": {base: np.asarray(row["features"][base], dtype=np.float64)}} for row in rows]
    weights = source_class_weights(values)
    standardizer = fit_standardizer(base, [row["features"][base] for row in values], weights)
    batch = make_batch(base, values, standardizer)
    batch["window_weights"] = weights
    return values, batch, standardizer


def train(root: Path, budget_s: float, device: str, resume: bool) -> dict[str, Any]:
    global STOP_REQUESTED
    budget = Budget(root, "training", budget_s)
    records_path = root / "models/fold_models.json"
    existing = json.loads(records_path.read_text(encoding="utf-8")).get("records", []) if resume and records_path.is_file() else []
    records = list(existing)
    completed = {(str(item["condition"]), str(item["held_out_source"]), int(item["seed"])) for item in records}
    rows = json.loads((root / "support/window_support.json").read_text(encoding="utf-8"))
    eligible = [row for row in rows if bool(row.get("paired_eligible")) and row.get("label") in (0, 1)]
    sources = sorted({str(row["source_id"]) for row in eligible})
    total = len(sources) * len(CONDITIONS) * len(SEEDS)
    root.joinpath("models").mkdir(parents=True, exist_ok=True)
    _progress(root, "train", "RUNNING", len(completed), total, budget=budget, source_count=len(sources), eligibility="paired_common_support_only")
    with (root / "models/loss_history.jsonl").open("a", encoding="utf-8") as loss_handle:
        for held_out in sources:
            for condition in CONDITIONS:
                mode = condition[0]
                base = {"SET": "SET_A", "RAW": "RAW_SEQ", "MULTI": "MULTI_ORDER_SEQ"}[condition.split("_", 1)[1]]
                train_rows = [row for row in eligible if str(row["source_id"]) != held_out and str(row["mode"]) == mode]
                if not train_rows or {int(row["label"]) for row in train_rows} != {0, 1}:
                    continue
                values, batch, standardizer = _fit_rows(train_rows, base)
                for seed in SEEDS:
                    key = (condition, held_out, int(seed))
                    if key in completed:
                        continue
                    if STOP_REQUESTED or budget.remaining() <= 0:
                        reason = "STOPPED_SAFE" if STOP_REQUESTED else "TRAIN_BUDGET_EXHAUSTED"
                        budget.save(reason)
                        _progress(root, "train", "STOPPED", len(completed), total, budget=budget, stop_reason=reason)
                        _atomic_json(records_path, {"conditions": list(CONDITIONS), "seeds": list(SEEDS), "epochs": EPOCHS, "records": records})
                        return {"status": reason, "completed_models": len(completed), "total_models": total}
                    started = time.perf_counter()
                    def epoch_callback(epoch: int, metrics: Mapping[str, Any]) -> None:
                        print(f"periodic condition={condition} held_out={held_out} seed={seed} epoch={epoch}/{EPOCHS} weighted_bce={float(metrics['loss']):.6f} elapsed={budget.elapsed():.1f}s remaining={budget.remaining():.1f}s", flush=True)
                    model, fit = train_one(base, batch, seed=int(seed), device=device, epochs=EPOCHS, epoch_callback=epoch_callback)
                    record = {"condition": condition, "base_condition": base, "mode": mode, "held_out_source": held_out, "seed": int(seed), "parameter_count": parameter_count(base), "training_source_count": len({str(row["source_id"]) for row in train_rows}), "training_real_count": sum(int(row["label"]) == 0 for row in train_rows), "training_fake_count": sum(int(row["label"]) == 1 for row in train_rows), "training_window_count": len(train_rows), "fit": fit, "standardization": standardizer.as_dict(), "elapsed_s": time.perf_counter() - started, "state_dict": model_state(model)}
                    records.append(record)
                    completed.add(key)
                    _atomic_json(records_path, {"conditions": list(CONDITIONS), "seeds": list(SEEDS), "epochs": EPOCHS, "learning_rate": LEARNING_RATE, "weight_decay": WEIGHT_DECAY, "records": records})
                    for epoch_row in fit["loss_history"]:
                        loss_handle.write(json.dumps({"condition": condition, "held_out_source": held_out, "seed": int(seed), **epoch_row}, allow_nan=False) + "\n")
                    loss_handle.flush()
                    del model
                    _progress(root, "train", "RUNNING", len(completed), total, budget=budget, current_condition=condition, held_out_source=held_out, seed=int(seed))
    status = "COMPLETE" if len(completed) == total else "PARTIAL"
    budget.save(None if status == "COMPLETE" else status)
    _progress(root, "train", status, len(completed), total, budget=budget)
    return {"status": status, "completed_models": len(completed), "total_models": total, "cumulative_s": budget.elapsed()}


def _auroc(labels: Sequence[int], scores: Sequence[float]) -> float | None:
    y = np.asarray(labels, dtype=np.int64)
    s = np.asarray(scores, dtype=np.float64)
    if np.sum(y == 0) == 0 or np.sum(y == 1) == 0:
        return None
    pos, neg = s[y == 1], s[y == 0]
    return float(np.mean((pos[:, None] > neg[None, :]) + 0.5 * (pos[:, None] == neg[None, :])))


def _ap(labels: Sequence[int], scores: Sequence[float]) -> float | None:
    y = np.asarray(labels, dtype=np.int64); s = np.asarray(scores, dtype=np.float64)
    if not np.any(y == 1) or not np.any(y == 0):
        return None
    order = np.argsort(-s, kind="mergesort"); ordered = y[order]; tp = np.cumsum(ordered == 1); precision = tp / np.arange(1, len(ordered) + 1)
    return float(np.sum(precision[ordered == 1]) / np.sum(ordered == 1))


def _classification(labels: Sequence[int], scores: Sequence[float]) -> dict[str, Any]:
    y = np.asarray(labels, dtype=np.int64); pred = np.asarray(scores, dtype=np.float64) >= 0.0
    tn = int(np.sum((y == 0) & ~pred)); fp = int(np.sum((y == 0) & pred)); fn = int(np.sum((y == 1) & ~pred)); tp = int(np.sum((y == 1) & pred))
    precision = tp / (tp + fp) if tp + fp else None; recall = tp / (tp + fn) if tp + fn else None
    f1 = 2 * precision * recall / (precision + recall) if precision is not None and recall is not None and precision + recall else None
    return {"precision": precision, "recall": recall, "f1": f1, "accuracy": (tn + tp) / len(y) if len(y) else None, "tn": tn, "fp": fp, "fn": fn, "tp": tp}


def _bootstrap(values: Mapping[str, float], other: Mapping[str, float] | None = None) -> dict[str, Any]:
    keys = sorted(set(values) & (set(other) if other is not None else set(values)))
    if not keys:
        return {"source_count": 0, "mean": None, "ci95": None, "seed": BOOTSTRAP_SEED, "replicates": BOOTSTRAP_REPLICATES}
    raw = np.asarray([values[key] - other[key] if other is not None else values[key] for key in keys], dtype=np.float64)
    draws = raw[np.random.default_rng(BOOTSTRAP_SEED).integers(0, len(raw), size=(BOOTSTRAP_REPLICATES, len(raw)))].mean(axis=1)
    return {"source_count": len(keys), "sources": keys, "mean": float(np.mean(raw)), "ci95": [float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))], "seed": BOOTSTRAP_SEED, "replicates": BOOTSTRAP_REPLICATES}


def evaluate(root: Path, device: str) -> dict[str, Any]:
    rows = json.loads((root / "support/window_support.json").read_text(encoding="utf-8"))
    records_data = json.loads((root / "models/fold_models.json").read_text(encoding="utf-8")) if (root / "models/fold_models.json").is_file() else {"records": []}
    records = {(str(item["condition"]), str(item["held_out_source"]), int(item["seed"])): item for item in records_data.get("records", [])}
    by_window: dict[str, dict[str, Any]] = {}
    for row in rows:
        window_id = str(row["window_id"])
        base = by_window.setdefault(window_id, {key: row.get(key) for key in ("window_id", "source_id", "pair_id", "role", "offset_s", "label", "annotation_category", "paired_eligible")})
        mode = str(row["mode"])
        base[f"{mode}_support_status"] = row.get("support_status")
        base[f"{mode}_valid_unit_count"] = int(row.get("valid_unit_count", 0))
        base[f"{mode}_group_count"] = int(row.get("retained_group_count", 0))
    for condition in CONDITIONS:
        mode = condition[0]
        base = {"SET": "SET_A", "RAW": "RAW_SEQ", "MULTI": "MULTI_ORDER_SEQ"}[condition.split("_", 1)[1]]
        mode_rows = [row for row in rows if str(row["mode"]) == mode and int(row.get("valid_unit_count", 0)) > 0 and row.get("features", {}).get(base) is not None]
        for held_out in sorted({str(row["source_id"]) for row in mode_rows}):
            held_rows = [row for row in mode_rows if str(row["source_id"]) == held_out]
            for seed in SEEDS:
                record = records.get((condition, held_out, int(seed)))
                if record is None:
                    continue
                model = _model_from_record(base, record, device)
                standardizer = _standardizer(record)
                batch_rows = [{**dict(row), "features": {base: np.asarray(row["features"][base], dtype=np.float64)}} for row in held_rows]
                batch = make_batch(base, batch_rows, standardizer, require_labels=False)
                scores = score_batch(base, model, batch, device)
                for window_id, score in zip(batch["window_ids"], scores):
                    by_window[str(window_id)][f"{condition}_seed_{seed}"] = float(score)
                del model
    final_rows = []
    for row in by_window.values():
        for condition in CONDITIONS:
            values = [row.get(f"{condition}_seed_{seed}") for seed in SEEDS]
            if all(value is not None and np.isfinite(float(value)) for value in values):
                row[condition] = float(np.mean(np.asarray(values, dtype=np.float64)))
                row[f"{condition}_seed_std"] = float(np.std(np.asarray(values, dtype=np.float64)))
                row[f"{condition}_status"] = "SCORED"
            else:
                row[condition] = ""
                row[f"{condition}_status"] = "MISSING_SEED_MODEL"
        final_rows.append(row)
    final_rows.sort(key=lambda row: str(row["window_id"]))
    _write_csv(root / "scores/oof_window_scores.csv", final_rows)
    source_values: dict[str, dict[str, float]] = {condition: {} for condition in CONDITIONS}
    per_source: list[dict[str, Any]] = []
    for source in sorted({str(row["source_id"]) for row in final_rows}):
        eligible = [row for row in final_rows if str(row["source_id"]) == source and row.get("paired_eligible") and row.get("label") in (0, 1)]
        output = {"source_id": source, "window_count": len(eligible), "real_count": sum(row.get("label") == 0 for row in eligible), "fake_count": sum(row.get("label") == 1 for row in eligible)}
        for condition in CONDITIONS:
            usable = [row for row in eligible if row.get(f"{condition}_status") == "SCORED"]
            labels = [int(row["label"]) for row in usable]; scores = [float(row[condition]) for row in usable]
            value = _auroc(labels, scores)
            output[f"{condition}_auroc"] = value
            output[f"{condition}_real_count"] = sum(label == 0 for label in labels)
            output[f"{condition}_fake_count"] = sum(label == 1 for label in labels)
            if value is not None:
                source_values[condition][source] = value
        for left, right, name in (("R_SET", "O_SET", "R_SET_MINUS_O_SET"), ("R_RAW", "O_RAW", "R_RAW_MINUS_O_RAW"), ("R_MULTI", "O_MULTI", "R_MULTI_MINUS_O_MULTI"), ("R_MULTI", "R_RAW", "R_MULTI_MINUS_R_RAW")):
            output[name] = source_values[left].get(source, np.nan) - source_values[right].get(source, np.nan) if source in source_values[left] and source in source_values[right] else ""
        per_source.append(output)
    _write_csv(root / "evaluation/per_source_metrics.csv", per_source)
    summary: dict[str, Any] = {"experiment": "V7 fixed periodic re-query versus continuous old-query matched pilot", "population": {"source_count": len({str(row['source_id']) for row in final_rows}), "subwindow_count": len(final_rows), "paired_eligible_count": sum(bool(row.get('paired_eligible')) for row in final_rows)}, "conditions": {}, "paired_comparisons": {}, "coverage": {}, "limitations": ["development pilot on frozen MANIP parent slots", "overlapping subwindows are correlated", "R-only support is not mixed into paired main gains", "no cross-cohort physical correspondence or spatial ground truth claim"]}
    for condition in CONDITIONS:
        eligible = [row for row in final_rows if row.get("paired_eligible") and row.get("label") in (0, 1) and row.get(f"{condition}_status") == "SCORED"]
        labels = [int(row["label"]) for row in eligible]; scores = [float(row[condition]) for row in eligible]
        summary["conditions"][condition] = {"source_auroc": _bootstrap(source_values[condition]), "pooled_auroc": _auroc(labels, scores), "pooled_ap": _ap(labels, scores), "classification_logit_ge_0": _classification(labels, scores), "window_count": len(eligible), "source_count": len(source_values[condition]), "parameter_count": parameter_count({"SET": "SET_A", "RAW": "RAW_SEQ", "MULTI": "MULTI_ORDER_SEQ"}[condition.split("_", 1)[1]])}
    for left, right, name in (("R_SET", "O_SET", "R_SET_MINUS_O_SET"), ("R_RAW", "O_RAW", "R_RAW_MINUS_O_RAW"), ("R_MULTI", "O_MULTI", "R_MULTI_MINUS_O_MULTI"), ("R_MULTI", "R_RAW", "R_MULTI_MINUS_R_RAW")):
        summary["paired_comparisons"][name] = _bootstrap(source_values[left], source_values[right])
    for mode in MODES:
        coverage_by_offset: dict[str, dict[str, int]] = {}
        for offset in OFFSETS_S:
            offset_rows = [row for row in final_rows if abs(float(row.get("offset_s", -999.0)) - float(offset)) < 1e-9]
            coverage_by_offset[str(offset)] = {
                "valid_windows": sum(
                    any(row.get(f"{mode}_{condition}_status") == "SCORED" for condition in ("SET", "RAW", "MULTI"))
                    for row in offset_rows
                ),
                "unit_count": sum(int(row.get(f"{mode}_valid_unit_count", 0)) for row in offset_rows),
            }
        summary["coverage"][mode] = coverage_by_offset
    _atomic_json(root / "evaluation/summary.json", summary)
    _progress(root, "evaluate", "COMPLETE", len(final_rows), len(final_rows), scored_windows=len(final_rows))
    return summary


def report(root: Path) -> dict[str, Any]:
    summary = json.loads((root / "evaluation/summary.json").read_text(encoding="utf-8")) if (root / "evaluation/summary.json").is_file() else {}
    parents = _load_parents(root) if (root / "manifests/parents.json").is_file() else []
    subwindows = _load_subwindows(root) if (root / "manifests/subwindows.json").is_file() else []
    results = json.loads((root / "frontend/results.json").read_text(encoding="utf-8")) if (root / "frontend/results.json").is_file() else []
    models = json.loads((root / "models/fold_models.json").read_text(encoding="utf-8")) if (root / "models/fold_models.json").is_file() else {"records": []}
    support = json.loads((root / "support/support_summary.json").read_text(encoding="utf-8")) if (root / "support/support_summary.json").is_file() else {}
    frontend_complete = sum(item.get("status") == "FRONTEND_COMPLETE" for item in results)
    complete_parents = 0
    for parent in parents:
        parent_rows = [item for item in results if item.get("parent_id") == parent.get("parent_id")]
        if parent_rows and all(item.get("status") == "FRONTEND_COMPLETE" for item in parent_rows):
            complete_parents += 1
    lines = [
        "# V7 固定重新查询观测与持续旧查询匹配监督对照 pilot",
        "",
        "本报告从当前独立产物生成；它不是正式检测链、sealed-test 或空间定位真值。",
        "",
        "## 直接回答",
        "",
        f"1. 实际完成 source/父片段/分析窗口/模型：{len({str(item.get('source_id')) for item in parents if item.get('status') == 'PLANNED'})}/{complete_parents}/{frontend_complete}/{len(models.get('records', []))}（计划 {len(parents)}/{len(subwindows)}/{len(CONDITIONS)*len({str(item.get('source_id')) for item in parents})*len(SEEDS)}）。",
        f"2. R 五时刻结构支撑：O 有效窗口 {support.get('o_valid_windows', 'NA')}，R 有效窗口 {support.get('r_valid_windows', 'NA')}；配对窗口 {support.get('paired_windows', 'NA')}。有效支撑不是物理对应证明。",
        "3. 增量按 real/fake 和 offset 的完整覆盖见 `support/coverage.csv`；不按分数排除窗口。",
        f"4. R_SET−O_SET：{summary.get('paired_comparisons', {}).get('R_SET_MINUS_O_SET', {})}。",
        f"5. R_RAW−O_RAW：{summary.get('paired_comparisons', {}).get('R_RAW_MINUS_O_RAW', {})}；R_MULTI−O_MULTI：{summary.get('paired_comparisons', {}).get('R_MULTI_MINUS_O_MULTI', {})}。",
        "6. R-only 新增覆盖与检测改善不是同一件事；主差值只使用 O/R 共同支持窗口。",
        "7. 计算成本记录在 `state/frontend_budget.json`、`state/training_budget.json` 及 frontend/results.json。",
        "8. 没有空间真值；不能据此宣称失真部位进入模型，也不能宣称跨 cohort 恢复物理点对应。",
        "",
        "## 条件结果（source 等权 AUROC；括号内为 source bootstrap 95% CI）",
        "",
        "| condition | source AUROC | pooled AUROC | pooled AP | windows | sources |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for condition in CONDITIONS:
        item = summary.get("conditions", {}).get(condition, {}); macro = item.get("source_auroc", {})
        lines.append(f"| {condition} | {macro.get('mean', 'NA')} [{macro.get('ci95', 'NA')}] | {item.get('pooled_auroc', 'NA')} | {item.get('pooled_ap', 'NA')} | {item.get('window_count', 'NA')} | {item.get('source_count', 'NA')} |")
    lines += [
        "",
        "## 冻结与隔离",
        "",
        "- 父片段只来自 frozen manifest 的中间 MANIP 槽位；子窗口和标签映射在任何结果前固定。",
        "- O 在父片段 t0 初始化 289 点，R 在 b=0/0.5/1.0 秒独立初始化；b=0 可复用 O，但批次 ID 不构成物理对应。",
        "- O/R 每个子窗口独立重建 H 组、历史尺度和五时刻共同成员；不跨 cohort 拼接导数。",
        "- 训练使用 paired common support、source-disjoint LOSO、fold 内标准化和 source/class weighted BCE；CTRL 未用于本 pilot 主训练。",
        "- 前端 geometry 在一个父片段内只计算一次并共享；前端输出的全部 PTS 与模型实际五时刻分别保留。",
        "",
        "## 产物",
        "",
        "`protocol.json`、`resolved_config.json`、`manifests/parents.json`、`manifests/subwindows.json`、`frontend/results.json`、`support/window_support.json`、`models/fold_models.json`、`scores/oof_window_scores.csv`、`evaluation/summary.json` 和 `evaluation/per_source_metrics.csv`。",
        "",
    ]
    (root / "report.md").write_text("\n".join(lines), encoding="utf-8")
    _progress(root, "report", "COMPLETE", 1, 1)
    return {"report": str(root / "report.md"), "models": len(models.get("records", [])), "frontend_windows": frontend_complete}


def run_all(root: Path, *, device: str, resume: bool, frontend_budget_s: float, train_budget_s: float) -> dict[str, Any]:
    prepare(root)
    frontend = run_frontend(root, frontend_budget_s, resume=resume)
    features_summary = features(root)
    training = train(root, train_budget_s, device=device, resume=resume)
    evaluation = evaluate(root, device=device) if (root / "models/fold_models.json").is_file() else {}
    report_summary = report(root)
    if frontend.get("status") == "COMPLETE" and training.get("status") == "COMPLETE" and evaluation:
        status = "COMPLETE"
    elif frontend.get("status") == "FRONTEND_BLOCKED":
        status = "FRONTEND_BLOCKED"
    elif frontend.get("status") in {"BUDGET_EXHAUSTED", "STOPPED_SAFE"}:
        status = frontend["status"]
    elif training.get("status") in {"TRAIN_BUDGET_EXHAUSTED", "STOPPED_SAFE"}:
        status = training["status"]
    else:
        status = "PARTIAL"
    final = {"status": status, "updated_unix": time.time(), "frontend": frontend, "features": features_summary, "training": training, "evaluation_present": bool(evaluation), "report": report_summary, "stop_reason": None if status == "COMPLETE" else status}
    _atomic_json(root / "final_status.json", final)
    return final


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("prepare", "frontend", "features", "train", "evaluate", "report", "all"))
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--frontend-budget-s", type=float, default=FRONTEND_BUDGET_S)
    parser.add_argument("--train-budget-s", type=float, default=TRAIN_BUDGET_S)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    root = args.output_root
    root.mkdir(parents=True, exist_ok=True)
    lock_path = root / ".run.lock"
    with lock_path.open("w") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit("RUN_LOCK_HELD")
        _atomic_json(root / "resolved_config.json", {"git_head": _git_head(), "device": args.device, "frontend_budget_s": args.frontend_budget_s, "train_budget_s": args.train_budget_s, "resume": bool(args.resume), "output_root": str(root)})
        global STOP_REQUESTED
        def stop(_signal: int, _frame: Any) -> None:
            STOP_REQUESTED = True
        signal.signal(signal.SIGTERM, stop)
        signal.signal(signal.SIGINT, stop)
        if args.stage == "prepare":
            prepare(root)
        elif args.stage == "frontend":
            if not (root / "manifests/subwindows.json").is_file():
                prepare(root)
            run_frontend(root, args.frontend_budget_s, resume=args.resume)
        elif args.stage == "features":
            features(root)
        elif args.stage == "train":
            train(root, args.train_budget_s, args.device, args.resume)
        elif args.stage == "evaluate":
            evaluate(root, args.device)
        elif args.stage == "report":
            report(root)
        else:
            run_all(root, device=args.device, resume=args.resume, frontend_budget_s=args.frontend_budget_s, train_budget_s=args.train_budget_s)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
