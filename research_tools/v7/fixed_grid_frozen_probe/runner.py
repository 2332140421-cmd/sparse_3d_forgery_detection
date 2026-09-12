"""Independent execution for the V7 fixed-grid frozen-model pilot.

This runner intentionally reuses the already audited 289-point frontend and
local-organization implementation.  It writes only to the data disk and can
be resumed after a process stop; it never changes the formal ``src`` chain or
the historical pilot outputs.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import os
import random
import signal
import subprocess
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from sparse3d_forgery.particle_sequence import (
    CoordinateSystem,
    Handedness,
    LengthUnit,
    build_particle_sequence,
    load_particle_sequence,
    save_particle_sequence,
)
from sparse3d_forgery.video_input import VideoSource, decode_video
from research_tools.v7.boundary_pooling_probe.grouping import (
    assign_uv_to_masks,
    split_local_groups_by_assignment,
    validate_boundary_partition,
)
from research_tools.v7.boundary_pooling_probe.model import PoolingWindowMLP, score_pooling_model
from research_tools.v7.boundary_pooling_probe.pipeline import (
    SEG_WEIGHT,
    _compact_grouping,
    _compact_support,
    _mask_cache_key,
    _polygons_to_masks,
    _predict_masks,
    _sha256 as file_sha256,
)
from research_tools.v7.local_organization_probe.grouping import (
    build_local_groups,
    build_local_support,
    rebuild_components_fast,
)
from research_tools.v7.local_structural_temporal_probe.model import (
    ARM_NAMES,
    WeightedStandardizer,
    build_batch,
)
from research_tools.v7.local_structural_temporal_probe.representation import COMPONENT_CONFIG
from research_tools.v7.observation_density_diagnostic import run_diagnostic as diagnostic
from .grid import classify_fake_window, generate_grid_windows, map_target_frames_to_intervals


REPO_ROOT = Path(__file__).resolve().parents[3]
DATA_ROOT = Path("/root/autodl-tmp/data/sparse_3d_forgery_detection")
SOURCE_ROOT = DATA_ROOT / "derived/v7_activityforensics_boundary_pooling_pilot_v1"
PAIRED_ROOT = DATA_ROOT / "derived/v7_activityforensics_paired_second_order_pilot_v1"
MODEL_ROOT = SOURCE_ROOT / "models/fold_models.json"
OUTPUT_ROOT = DATA_ROOT / "derived/v7_activityforensics_fixed_grid_frozen_probe_v1"
WINDOW_LENGTH_S = 1.0
STRIDE_S = 0.5
SEED = 20260909
CONDITIONS = ("H_MEAN_A", "B_MEAN_A", "B_MEAN_C")
ARM_BY_CONDITION = {"H_MEAN_A": ("H", "UNORDERED_STATE"), "B_MEAN_A": ("B", "UNORDERED_STATE"), "B_MEAN_C": ("B", "ORDERED_SECOND")}
TAPNET_SOURCE = diagnostic.TAPNET_SOURCE
TAPNET_CHECKPOINT = diagnostic.TAPNET_CHECKPOINT
DEPTH_SOURCE = diagnostic.DEPTH_SOURCE
DEPTH_CHECKPOINT = diagnostic.DEPTH_CHECKPOINT
TRACKER_SHA = diagnostic.TRACKER_SHA
DEPTH_SHA = diagnostic.DEPTH_SHA


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    raise TypeError(type(value).__name__)


def _safe(value: str) -> str:
    return "".join(char if char.isalnum() or char in "._-" else "_" for char in value)


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False, default=_json_default) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    names: list[str] = []
    for row in rows:
        for name in row:
            if name not in names:
                names.append(name)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=names, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def _progress(root: Path, phase: str, completed: int, total: int, status: str = "RUNNING", **extra: Any) -> None:
    _atomic_json(root / "progress.json", {"phase": phase, "completed": int(completed), "total": int(total), "status": status, "updated_unix": time.time(), **extra})


def _git_head() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()


def _probe_video(path: Path) -> dict[str, Any]:
    import av

    pts: list[float] = []
    width = height = None
    codec = None
    with av.open(str(path)) as container:
        stream = next(stream for stream in container.streams if stream.type == "video")
        codec = str(stream.codec_context.name)
        for frame in container.decode(stream):
            if frame.pts is None or frame.time_base is None:
                raise RuntimeError(f"PTS_MISSING:{path}")
            stamp = float(frame.pts * frame.time_base)
            if not np.isfinite(stamp) or (pts and stamp <= pts[-1]):
                raise RuntimeError(f"PTS_NOT_STRICT:{path}")
            pts.append(stamp)
            height, width = int(frame.height), int(frame.width)
    if not pts:
        raise RuntimeError(f"NO_VIDEO_FRAMES:{path}")
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": file_sha256(path), "frame_count": len(pts), "frame_indices": list(range(len(pts))), "timestamps_s": pts, "width": width, "height": height, "codec": codec, "origin_pts_s": pts[0], "duration_relative_s": pts[-1] - pts[0]}


def _selected_pairs() -> dict[str, dict[str, Any]]:
    path = PAIRED_ROOT / "manifests/selected_pairs.json"
    values = json.loads(path.read_text(encoding="utf-8"))
    return {str(row["source_id"]): row for row in values}


def prepare_plan(root: Path) -> dict[str, Any]:
    input_rows = json.loads((SOURCE_ROOT / "manifests/input_manifest.json").read_text(encoding="utf-8"))["rows"]
    by_video: dict[tuple[str, str], dict[str, Any]] = {}
    for row in input_rows:
        key = (str(row["source_id"]), str(row["role"]))
        by_video.setdefault(key, {"source_id": key[0], "role": key[1], "pair_id": str(row["pair_id"]), "video_path": str(row["video_path"])})
        if by_video[key]["video_path"] != str(row["video_path"]):
            raise RuntimeError(f"multiple videos for frozen source/role: {key}")
    pairs = _selected_pairs()
    videos: list[dict[str, Any]] = []
    grids: list[dict[str, Any]] = []
    for (source_id, role), item in sorted(by_video.items()):
        path = Path(item["video_path"])
        record: dict[str, Any] = {**item, "video_id": f"{source_id}::{role}", "status": "MISSING" if not path.is_file() else "PENDING"}
        if path.is_file():
            try:
                record.update(_probe_video(path))
                record["status"] = "METADATA_COMPLETE"
            except Exception as exc:
                record.update({"status": "PTS_OR_DECODE_FAILED", "error": f"{type(exc).__name__}: {exc}"})
        if record["status"] == "PENDING":
            rows = generate_grid_windows(record["timestamps_s"], window_length_s=WINDOW_LENGTH_S, stride_s=STRIDE_S)
            if rows and rows[0].get("status") == "SHORT_VIDEO":
                record["status"] = "SHORT_VIDEO"
            annotations = [] if role == "real" else list(pairs.get(source_id, {}).get("all_manipulation_segments", []))
            record["annotation_intervals_relative_s"] = annotations
            for planned in rows:
                if planned.get("status") != "PLANNED":
                    continue
                grid_id = f"{source_id}::{role}::grid{int(planned['grid_index']):04d}"
                start_pts = float(record["origin_pts_s"] + planned["nominal_start_s"])
                end_pts = float(record["origin_pts_s"] + planned["nominal_end_s"])
                annotation = {"annotation_category": "REAL_NEGATIVE"} if role == "real" else classify_fake_window(start_pts, end_pts, annotations, origin_pts_s=float(record["origin_pts_s"]))
                target = {"target_frame_count": 0, "target_frames_in_annotation_count": 0, "target_pts_in_annotation_s": []} if role == "real" else map_target_frames_to_intervals(planned["frame_pts_s"], annotations, origin_pts_s=float(record["origin_pts_s"]))
                grids.append({
                    "grid_window_id": grid_id, "window_id": grid_id, "source_video_id": record["video_id"], "source_id": source_id, "pair_id": record["pair_id"], "role": role, "kind": "GRID",
                    "video_path": str(path), "grid_index": int(planned["grid_index"]), "nominal_start_s": float(planned["nominal_start_s"]), "nominal_end_s": float(planned["nominal_end_s"]),
                    "interval_start_s": start_pts, "interval_end_s": end_pts, "actual_start_pts_s": planned["actual_start_pts_s"], "actual_end_pts_s": planned["actual_end_pts_s"],
                    "frame_indices": list(planned["frame_indices"]), "timestamps_s": list(planned["frame_pts_s"]), "tail_window": bool(planned["tail_window"]), "timestamp_origin_s": float(record["origin_pts_s"]),
                    **annotation, **target, "grid_frozen_before_labels": True,
                })
        videos.append(record)
    source_ids = sorted({str(row["source_id"]) for row in videos})
    random.Random(SEED).shuffle(source_ids)
    plan = {"seed": SEED, "source_order": source_ids, "window_length_s": WINDOW_LENGTH_S, "stride_s": STRIDE_S, "video_count": len(videos), "grid_window_count": len(grids), "videos": videos}
    _atomic_json(root / "manifests/videos.json", videos)
    _atomic_json(root / "manifests/grid_windows.json", grids)
    _atomic_json(root / "manifests/execution_plan.json", {"source_order": source_ids, "seed": SEED, "window_count": len(grids), "rule": "complete source prefix in fixed randomized order; no label or score selection"})
    return plan


def write_protocol(root: Path) -> None:
    records = json.loads(MODEL_ROOT.read_text(encoding="utf-8"))
    model_sha = file_sha256(MODEL_ROOT)
    _atomic_json(root / "protocol.json", {
        "protocol_id": "v7-fixed-grid-frozen-probe-v1", "git_head": _git_head(), "question": "fixed label-blind time coverage versus observation support and frozen response",
        "window_grid": {"length_s": WINDOW_LENGTH_S, "stride_s": STRIDE_S, "time_origin": "first actual decoded PTS per video", "tail_policy": "append one ending at last PTS when needed", "short_video": "record SHORT_VIDEO; no padding"},
        "frontend": {"provider_tracker": "online_bootstapir", "tracker_source_sha": TRACKER_SHA, "tracker_checkpoint": str(TAPNET_CHECKPOINT), "depth": "apple_depth_pro", "depth_source_sha": DEPTH_SHA, "depth_checkpoint": str(DEPTH_CHECKPOINT), "process_size": 256, "query_grid": "17x17 interior linspace on process raster = 289", "visibility": "tracker visibility plus finite depth/pose geometry masks", "depth_semantics": "optical-axis z-depth", "pose": "Open3D RGB-D odometry, target camera-from-source inverted during world accumulation", "coordinates": "first_camera_world, right-handed right/down/forward, meters"},
        "window_initialization": "each grid window decodes its own frames and initializes 289 queries and causal state; no cross-window track identity",
        "representation": {"history_s": 0.5, "target_offsets_s": [0.5, 0.6, 0.7, 0.8, 0.9], "target_tolerance_s": 0.05, "component_config": {"max_initial_distance_m": float(COMPONENT_CONFIG.max_initial_distance), "max_relative_change_m": float(COMPONENT_CONFIG.max_relative_change), "minimum_size": int(COMPONENT_CONFIG.minimum_size), "minimum_overlap": int(COMPONENT_CONFIG.minimum_overlap)}, "H": "existing history local groups", "B": "first-frame segmentation partition of H; no later reassignment"},
        "frozen_scoring": {"conditions": list(CONDITIONS), "models": str(MODEL_ROOT), "models_sha256": model_sha, "available_conditions": records.get("conditions"), "seeds": records.get("seeds"), "held_out_rule": "only held_out_source=s records", "standardization": "saved fold standardizer; no refit", "score": "arithmetic mean of three seed logits; missing support is missing, never zero"},
        "labels": "applied only after grid freeze; real windows negative, fake windows fully within annotation union positive; boundary/outside descriptive only",
        "bootstrap": {"unit": "source", "seed": SEED, "replicates": 10000},
        "budget_s": 7200,
        "boundaries": ["development pilot, not sealed test", "no dense branch", "no training or calibration", "no spatial ground truth claim"],
    })


def _load_grid(root: Path) -> list[dict[str, Any]]:
    return json.loads((root / "manifests/grid_windows.json").read_text(encoding="utf-8"))


def _load_results(root: Path) -> dict[str, dict[str, Any]]:
    path = root / "frontend_results.json"
    if not path.is_file():
        return {}
    return {str(row["window_id"]): row for row in json.loads(path.read_text(encoding="utf-8")).get("results", [])}


def _save_results(root: Path, results: Mapping[str, Mapping[str, Any]]) -> None:
    _atomic_json(root / "frontend_results.json", {"results": list(sorted(results.values(), key=lambda row: str(row["window_id"])))})


def run_frontend(root: Path, budget_s: float, resume: bool = True, max_windows: int | None = None) -> dict[str, Any]:
    """Run the reused 289-point frontend in fixed source/window order."""

    import torch
    from scripts.run_v7_explicit_geometry_frontend import (
        DepthProRunner,
        OnlineBootsTapir,
        accumulate_world_from_camera,
        causal_first_frame_intrinsics,
        persistent_track_ids,
        rgbd_odometry,
        sample_depth_at_uv,
        world_xyz,
    )

    rows = _load_grid(root)
    execution = json.loads((root / "manifests/execution_plan.json").read_text(encoding="utf-8"))
    order = {source: index for index, source in enumerate(execution["source_order"])}
    rows.sort(key=lambda row: (order.get(str(row["source_id"]), 10_000), str(row["role"]), int(row["grid_index"])))
    benchmark_sources = sorted({str(row["source_id"]) for row in rows})
    benchmark_source = benchmark_sources[0] if benchmark_sources else None
    benchmark_ids = [str(row["window_id"]) for row in rows if str(row["source_id"]) == benchmark_source and int(row["grid_index"]) == 0]
    rows = [row for row in rows if str(row["window_id"]) in benchmark_ids] + [row for row in rows if str(row["window_id"]) not in benchmark_ids]
    results = _load_results(root) if resume else {}
    pending = [row for row in rows if str(row["window_id"]) not in results]
    if max_windows is not None:
        pending = pending[: int(max_windows)]
    if not pending:
        _progress(root, "frontend", len(results), len(rows), "REUSED")
        return {"status": "REUSED", "completed": len(results), "total": len(rows)}
    tracker = OnlineBootsTapir(TAPNET_SOURCE, TAPNET_CHECKPOINT, process_size=256, grid_size=17)
    depth_runner = DepthProRunner(DEPTH_SOURCE, DEPTH_CHECKPOINT)
    started_all = time.monotonic()
    stop_requested = False

    def stop(_signum: int, _frame: Any) -> None:
        nonlocal stop_requested
        stop_requested = True

    old_handler = signal.signal(signal.SIGTERM, stop)
    old_int = signal.signal(signal.SIGINT, stop)
    try:
        for number, row in enumerate(pending, 1):
            elapsed = time.monotonic() - started_all
            if stop_requested or elapsed >= float(budget_s):
                _progress(root, "frontend", len(results), len(rows), "STOPPED_SAFE" if stop_requested else "BUDGET_EXHAUSTED", elapsed_s=elapsed)
                break
            window_started = time.perf_counter()
            try:
                decoded = decode_video(VideoSource(sample_id=f"v7-fixed-grid-{row['window_id']}", source_video_id=str(row["source_video_id"]), source_locator=row["video_path"]), row["frame_indices"])
                depths, focals, frame_depth_valid = depth_runner.infer(decoded)
                intrinsics, fixed_focal_px = causal_first_frame_intrinsics(depths, focals)
                adjacent, pair_valid, information = rgbd_odometry(decoded, depths, intrinsics)
                world_from_camera, pose_valid = accumulate_world_from_camera(adjacent, pair_valid)
                tracker.grid_size = 17
                uv, visibility = tracker.track(decoded)
                sampled_depth, sampled_valid = sample_depth_at_uv(depths, uv)
                observation_valid = visibility & sampled_valid & frame_depth_valid[:, None]
                xyz, geometry_valid = world_xyz(uv, sampled_depth, intrinsics, world_from_camera, observation_valid, pose_valid)
                sequence = build_particle_sequence(
                    decoded, track_ids=persistent_track_ids(uv.shape[1]), xyz=xyz, uv=uv, visibility=visibility, geometry_validity=geometry_valid,
                    coordinate_system=CoordinateSystem(frame_name="first_camera_world", handedness=Handedness.RIGHT, axis_directions=("right", "down", "forward"), length_unit=LengthUnit.METER, camera_motion_compensated=True, normalization={}),
                    lineage={"dataset": "ActivityForensics+Charades", "official_split": "train", "source_id": row["source_id"], "pair_id": row["pair_id"], "role": row["role"], "grid_window_id": row["window_id"]},
                    provenance={"tracker": "online_bootstapir", "tracker_source_sha": TRACKER_SHA, "depth": "apple_depth_pro", "depth_source_sha": DEPTH_SHA, "depth_semantics": "optical_axis_z_depth", "pose": "open3d_rgbd_odometry", "pose_convention": "target_camera_from_source_camera_inverted", "process_size": 256, "query_grid_size": 17, "query_count": 289, "window_initialization": "independent", "causal_execution": True, "fixed_focal_px": fixed_focal_px},
                )
                prefix = root / "particles" / f"{_safe(str(row['window_id']))}__density289"
                if not prefix.with_suffix(".npz").is_file():
                    save_particle_sequence(sequence, prefix)
                diag = diagnostic.component_diagnostic(sequence, window_start_s=float(row["interval_start_s"]), label="density289")
                result = {**row, "sequence_prefix": str(prefix), "result_key": f"{row['window_id']}::density289", "frame_indices": [int(x) for x in sequence.frame_indices], "timestamps_s": [float(x) for x in sequence.timestamps_s], "frame_sizes_hw": np.asarray(sequence.frame_sizes_hw).tolist(), "query_count": int(sequence.num_tracks), "tracking_visible_fraction": float(np.mean(visibility)), "geometry_valid_fraction": float(np.mean(geometry_valid)), "valid_points_per_frame": np.sum(geometry_valid, axis=1).astype(int).tolist(), "pose_pair_valid": np.asarray(pair_valid).tolist(), "pose_valid": np.asarray(pose_valid).tolist(), "support_status": str(diag["support_status"]), "valid_triplet_count": int(diag["valid_triplet_count"]), "component_count": int(diag["component_count"]), "max_component_members": int(diag["max_component_members"]), "selected_triplet_common_members": int(diag["selected_triplet_common_members"]), "causal_training_eligible": False, "causal_training_reason": "frozen frontend window observation; no target construction", "elapsed_s": time.perf_counter() - window_started}
                results[str(row["window_id"])] = result
                _save_results(root, results)
                _progress(root, "frontend", len(results), len(rows), "RUNNING", last_window=row["window_id"], elapsed_s=time.monotonic() - started_all)
                print(f"frontend {len(results)}/{len(rows)} source={row['source_id']} role={row['role']} grid={row['grid_index']} new {result['elapsed_s']:.1f}s", flush=True)
                del decoded, depths, uv, visibility, xyz, geometry_valid, sequence
                gc.collect()
                torch.cuda.empty_cache()
            except Exception as exc:
                error = {**row, "result_key": f"{row['window_id']}::density289", "status": "FRONTEND_FAILED", "error": f"{type(exc).__name__}: {exc}"}
                results[str(row["window_id"])] = error
                _save_results(root, results)
                _progress(root, "frontend", len(results), len(rows), "RUNNING", last_error=error["error"])
                print(f"frontend FAILED {row['window_id']}: {error['error']}", flush=True)
    finally:
        signal.signal(signal.SIGTERM, old_handler)
        signal.signal(signal.SIGINT, old_int)
        del tracker, depth_runner
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    complete = sum(bool(item.get("sequence_prefix")) for item in results.values())
    status = "COMPLETE" if complete == len(rows) else ("BUDGET_EXHAUSTED" if not stop_requested else "STOPPED_SAFE")
    if max_windows is not None and len(results) < len(rows) and not stop_requested:
        status = "BENCHMARK_PARTIAL"
    if benchmark_ids:
        _atomic_json(root / "evaluation/benchmark.json", {"source_id": benchmark_source, "window_ids": benchmark_ids, "completed": [item for item in benchmark_ids if item in results], "note": "fixed sorted-source real/fake first complete grid windows; no label/score selection"})
    _progress(root, "frontend", complete, len(rows), status, elapsed_s=time.monotonic() - started_all)
    return {"status": status, "completed": complete, "total": len(rows)}


def _feature_path(root: Path, window_id: str) -> Path:
    return root / "features" / f"{_safe(window_id)}.json"


def build_features(root: Path, resume: bool = True) -> dict[str, Any]:
    """Build the existing H and first-frame segmentation B supports."""

    rows = _load_grid(root)
    results = _load_results(root)
    usable = [row for row in rows if str(row["window_id"]) in results and results[str(row["window_id"])].get("sequence_prefix")]
    mask_dir = root / "masks"
    mask_dir.mkdir(parents=True, exist_ok=True)
    model = None
    mask_cache: dict[str, dict[str, Any]] = {}
    coverage: list[dict[str, Any]] = []
    for number, row in enumerate(usable, 1):
        path = _feature_path(root, str(row["window_id"]))
        if resume and path.is_file():
            feature = json.loads(path.read_text(encoding="utf-8"))
        else:
            result = results[str(row["window_id"])]
            sequence = load_particle_sequence(Path(str(result["sequence_prefix"])))
            history = np.flatnonzero(np.asarray(sequence.timestamps_s) < float(row["interval_start_s"]) + 0.5).astype(np.int64)
            components = rebuild_components_fast(sequence.xyz, sequence.geometry_validity, history, COMPONENT_CONFIG)
            h_grouping = build_local_groups(sequence, history, old_components=components)
            h_support = build_local_support(sequence, window_start_s=float(row["interval_start_s"]), grouping=h_grouping)
            feature = {"identity": {"window_id": str(row["window_id"]), "source_id": str(row["source_id"]), "role": str(row["role"]), "grid_index": int(row["grid_index"]), "particle_prefix": str(result["sequence_prefix"]), "particle_count": int(sequence.num_tracks), "frame_indices": [int(x) for x in sequence.frame_indices], "timestamps_s": [float(x) for x in sequence.timestamps_s]}, "h_grouping": _compact_grouping(h_grouping), "h_support": _compact_support(h_support), "segmentation": {"status": "PENDING"}}
            video = Path(str(row["video_path"]))
            key = _mask_cache_key(str(video), int(sequence.frame_indices[0]))
            mask_path = mask_dir / f"{key}.json"
            if key not in mask_cache:
                if resume and mask_path.is_file():
                    mask_cache[key] = json.loads(mask_path.read_text(encoding="utf-8"))
                elif not video.is_file():
                    mask_cache[key] = {"status": "SOURCE_MISSING", "source_video": str(video), "frame_index": int(sequence.frame_indices[0]), "polygons": [], "areas": []}
                else:
                    if model is None:
                        from ultralytics import YOLO
                        model = YOLO(str(SEG_WEIGHT))
                    try:
                        mask_cache[key] = {"status": "COMPLETE", **_predict_masks(video, int(sequence.frame_indices[0]), model)}
                    except Exception as exc:
                        mask_cache[key] = {"status": "DECODE_FAILED", "source_video": str(video), "frame_index": int(sequence.frame_indices[0]), "polygons": [], "areas": [], "error": f"{type(exc).__name__}: {exc}"}
                _atomic_json(mask_path, mask_cache[key])
            record = mask_cache[key]
            masks = _polygons_to_masks(record) if record.get("status") == "COMPLETE" else []
            assignments = assign_uv_to_masks(np.asarray(sequence.uv)[0], np.asarray(sequence.visibility)[0], masks, record.get("areas", []))
            b_grouping = split_local_groups_by_assignment(h_grouping, assignments, minimum_size=int(COMPONENT_CONFIG.minimum_size))
            for child in b_grouping["groups"]:
                child["track_ids"] = [int(sequence.track_ids[int(slot)]) for slot in child["member_slots"]]
            checks = validate_boundary_partition(h_grouping, b_grouping)
            b_support = build_local_support(sequence, window_start_s=float(row["interval_start_s"]), grouping=b_grouping)
            feature["segmentation"] = {"status": str(record.get("status")), "cache_key": key, "cache_path": str(mask_path), "assignment_counts": {str(int(k)): int(v) for k, v in zip(*np.unique(assignments, return_counts=True))}}
            feature["b_grouping"] = _compact_grouping(b_grouping)
            feature["b_support"] = _compact_support(b_support)
            feature["boundary_checks"] = checks
            _atomic_json(path, feature)
        coverage.append({"window_id": row["window_id"], "source_id": row["source_id"], "role": row["role"], "grid_index": row["grid_index"], "h_support": feature.get("h_support", {}).get("support_status"), "b_support": feature.get("b_support", {}).get("support_status"), "h_triplets": feature.get("h_support", {}).get("valid_triplet_count", 0), "b_triplets": feature.get("b_support", {}).get("valid_triplet_count", 0), "segmentation_status": feature.get("segmentation", {}).get("status"), "frontend_status": results[str(row["window_id"])].get("status", "COMPLETE")})
        if number % 10 == 0:
            _progress(root, "features", number, len(usable), "RUNNING")
    _write_csv(root / "coverage/window_support.csv", coverage)
    source_rows: list[dict[str, Any]] = []
    for source in sorted({str(row["source_id"]) for row in rows}):
        source_all = [row for row in rows if str(row["source_id"]) == source]
        source_done = [row for row in usable if str(row["source_id"]) == source and str(row["window_id"]) in {str(item["window_id"]) for item in usable}]
        source_rows.append({"source_id": source, "planned_videos": len({str(row["role"]) for row in source_all}), "planned_windows": len(source_all), "frontend_windows": len(source_done), "complete_source": len(source_done) == len(source_all), "partial_reason": "" if len(source_done) == len(source_all) else "budget_or_frontend_failure"})
    _write_csv(root / "coverage/source_coverage.csv", source_rows)
    _atomic_json(root / "manifests/feature_summary.json", {"windows": len(coverage), "h_valid": sum(row["h_support"] == "VALID" for row in coverage), "b_valid": sum(row["b_support"] == "VALID" for row in coverage), "segmentation_status": {status: sum(row["segmentation_status"] == status for row in coverage) for status in sorted({str(row["segmentation_status"]) for row in coverage})}})
    _progress(root, "features", len(coverage), len(usable), "COMPLETE")
    return {"windows": len(coverage), "usable": len(usable)}


def _load_models() -> tuple[dict[tuple[str, str, int], tuple[Any, Any]], dict[str, Any]]:
    import torch

    data = json.loads(MODEL_ROOT.read_text(encoding="utf-8"))
    models: dict[tuple[str, str, int], tuple[Any, Any]] = {}
    for record in data.get("records", []):
        condition = str(record["condition"])
        if condition not in CONDITIONS:
            continue
        model_record = record["model"]
        model = PoolingWindowMLP(str(model_record["pooling"]))
        state = {name: torch.as_tensor(value, dtype=model.state_dict()[name].dtype) for name, value in model_record["state_dict"].items()}
        model.load_state_dict(state)
        model.eval()
        std = model_record["standardization"]
        standardizer = WeightedStandardizer(np.asarray(std["mean"], dtype=np.float64), np.asarray(std["scale"], dtype=np.float64), tuple(int(x) for x in std.get("zero_variance_dimensions", [])))
        models[(condition, str(record["held_out_source"]), int(record["seed"]))] = (model, standardizer)
    return models, data


def score_windows(root: Path, resume: bool = True) -> dict[str, Any]:
    rows = _load_grid(root)
    results = _load_results(root)
    models, model_data = _load_models()
    score_rows: list[dict[str, Any]] = []
    for row in rows:
        window_id = str(row["window_id"])
        result = results.get(window_id, {})
        feature_path = _feature_path(root, window_id)
        base: dict[str, Any] = {"window_id": window_id, "source_id": str(row["source_id"]), "role": str(row["role"]), "grid_index": int(row["grid_index"]), "annotation_category": row.get("annotation_category"), "annotation_overlap_s": row.get("annotation_overlap_s"), "annotation_overlap_fraction": row.get("annotation_overlap_fraction"), "frontend_status": result.get("status", "COMPLETE" if result.get("sequence_prefix") else "MISSING"), "model_used_pts_count": len(row.get("timestamps_s", []))}
        if not feature_path.is_file():
            base["score_status"] = "MISSING_FEATURE"
            score_rows.append(base)
            continue
        feature = json.loads(feature_path.read_text(encoding="utf-8"))
        for condition, (organization, arm) in ARM_BY_CONDITION.items():
            support = feature.get("h_support" if organization == "H" else "b_support", {})
            triplets = support.get("triplets", [])
            if not triplets:
                base[f"{condition}_status"] = "NO_VALID_TRIPLET"
                continue
            example = {"window_id": window_id, "source_id": str(row["source_id"]), "role": str(row["role"]), "kind": "MANIP", "triplets": triplets}
            seed_values: list[float] = []
            for seed in model_data.get("seeds", []):
                item = models.get((condition, str(row["source_id"]), int(seed)))
                if item is None:
                    continue
                model, standardizer = item
                batch, _ = build_batch([example], arm, standardizer=standardizer)
                seed_values.append(float(score_pooling_model(model, batch)[0]))
                base[f"{condition}_seed_{seed}"] = seed_values[-1]
            if seed_values:
                base[condition] = float(np.mean(seed_values))
                base[f"{condition}_status"] = "SCORED"
            else:
                base[f"{condition}_status"] = "MISSING_HELDOUT_MODEL"
        base["score_status"] = "SCORED" if any(base.get(f"{condition}_status") == "SCORED" for condition in CONDITIONS) else "NO_MODEL_SCORE"
        score_rows.append(base)
    _write_csv(root / "scores/window_scores.csv", score_rows)
    _progress(root, "score", len(score_rows), len(rows), "COMPLETE")
    return {"windows": len(score_rows), "models": len(models)}


def _auroc(labels: Sequence[int], scores: Sequence[float]) -> float | None:
    labels_array = np.asarray(labels, dtype=np.int64); values = np.asarray(scores, dtype=np.float64)
    positives = values[labels_array == 1]; negatives = values[labels_array == 0]
    if not positives.size or not negatives.size: return None
    return float(np.mean((positives[:, None] > negatives[None, :]).astype(float) + 0.5 * (positives[:, None] == negatives[None, :])))


def _average_precision(labels: Sequence[int], scores: Sequence[float]) -> float | None:
    y = np.asarray(labels, dtype=np.int64); s = np.asarray(scores, dtype=np.float64)
    if not np.any(y == 1) or not np.any(y == 0): return None
    order = np.argsort(-s, kind="mergesort"); yy = y[order]; tp = np.cumsum(yy == 1); precision = tp / np.arange(1, len(yy) + 1); return float(np.sum(precision[yy == 1]) / np.sum(yy == 1))


def _classification(labels: Sequence[int], scores: Sequence[float]) -> dict[str, Any]:
    y = np.asarray(labels, dtype=np.int64); pred = np.asarray(scores) >= 0.0
    tn = int(np.sum((y == 0) & ~pred)); fp = int(np.sum((y == 0) & pred)); fn = int(np.sum((y == 1) & ~pred)); tp = int(np.sum((y == 1) & pred)); precision = tp / (tp + fp) if tp + fp else None; recall = tp / (tp + fn) if tp + fn else None; f1 = 2 * precision * recall / (precision + recall) if precision is not None and recall is not None and precision + recall else None
    return {"precision": precision, "recall": recall, "f1": f1, "accuracy": (tn + tp) / len(y) if len(y) else None, "tn": tn, "fp": fp, "fn": fn, "tp": tp}


def evaluate(root: Path) -> dict[str, Any]:
    with (root / "scores/window_scores.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    per_source: list[dict[str, Any]] = []; summary: dict[str, Any] = {"conditions": {}, "protocol": {"positive": "fake windows fully inside annotation union", "negative": "all real windows", "boundary_and_outside": "descriptive only"}}
    source_ids = sorted({str(row["source_id"]) for row in rows})
    for condition in CONDITIONS:
        source_values: dict[str, float] = {}; pooled_labels: list[int] = []; pooled_scores: list[float] = []
        for source in source_ids:
            selected = []
            for row in rows:
                if str(row["source_id"]) != source or row.get(condition, "") == "" or row.get(f"{condition}_status") != "SCORED": continue
                if row["role"] == "real": label = 0
                elif row.get("annotation_category") == "FAKE_MANIPULATION": label = 1
                else: continue
                selected.append((label, float(row[condition])))
            if selected and {label for label, _ in selected} == {0, 1}:
                value = _auroc([x[0] for x in selected], [x[1] for x in selected]); source_values[source] = value
                pooled_labels.extend(x[0] for x in selected); pooled_scores.extend(x[1] for x in selected)
                per_source.append({"condition": condition, "source_id": source, "n_windows": len(selected), "real_count": sum(x[0] == 0 for x in selected), "fake_count": sum(x[0] == 1 for x in selected), "auroc": value})
        values = list(source_values.values()); mean = float(np.mean(values)) if values else None
        rng = np.random.default_rng(SEED); bootstrap = []
        if values:
            array = np.asarray(values, dtype=float)
            bootstrap = [float(np.mean(array[rng.integers(0, len(array), len(array))])) for _ in range(10000)]
        summary["conditions"][condition] = {"source_count": len(values), "source_mean_auroc": mean, "bootstrap_ci95": [float(np.percentile(bootstrap, 2.5)), float(np.percentile(bootstrap, 97.5))] if bootstrap else None, "pooled_auroc": _auroc(pooled_labels, pooled_scores), "pooled_ap": _average_precision(pooled_labels, pooled_scores), **(_classification(pooled_labels, pooled_scores) if pooled_labels else {})}
    _write_csv(root / "evaluation/per_source_metrics.csv", per_source)
    _atomic_json(root / "evaluation/summary.json", summary)
    _progress(root, "evaluate", len(per_source), len(CONDITIONS) * len(source_ids), "COMPLETE")
    return summary


def build_review(root: Path) -> None:
    rows = list(csv.DictReader((root / "scores/window_scores.csv").open(newline="", encoding="utf-8")))
    payload = json.dumps(rows, ensure_ascii=False)
    html = """<!doctype html><meta charset='utf-8'><title>V7 fixed-grid frozen probe</title><style>body{font:14px sans-serif;background:#101826;color:#eee;margin:2em}table{border-collapse:collapse}td,th{border:1px solid #475569;padding:4px}tr:hover{background:#26354d}.missing{color:#fbbf24}</style><h1>V7 固定时间网格观测与冻结模型评分</h1><p>窗口标签是数据集时间标注；分数是冻结模型 logit，不是概率或空间真值。窗口之间不共享 track identity。</p><input id='q' placeholder='source/window filter'><table id='t'><thead><tr><th>window</th><th>role</th><th>annotation</th><th>H_MEAN_A</th><th>B_MEAN_A</th><th>B_MEAN_C</th><th>score status</th></tr></thead><tbody></tbody></table><script>const rows=__ROWS__;const body=document.querySelector('tbody');function draw(){const q=document.querySelector('#q').value.toLowerCase();body.innerHTML=rows.filter(r=>(r.window_id+' '+r.source_id).toLowerCase().includes(q)).map(r=>'<tr><td>'+r.window_id+'</td><td>'+r.role+'</td><td>'+r.annotation_category+'</td><td>'+fmt(r.H_MEAN_A)+'</td><td>'+fmt(r.B_MEAN_A)+'</td><td>'+fmt(r.B_MEAN_C)+'</td><td>'+r.score_status+'</td></tr>').join('')}function fmt(x){return x==null||x===''?'<span class="missing">—</span>':Number(x).toFixed(5)}document.querySelector('#q').oninput=draw;draw()</script>""".replace("__ROWS__", payload)
    (root / "review/index.html").parent.mkdir(parents=True, exist_ok=True); (root / "review/index.html").write_text(html, encoding="utf-8")


def write_report(root: Path, frontend: Mapping[str, Any], feature: Mapping[str, Any], summary: Mapping[str, Any]) -> None:
    lines = ["# V7 固定时间网格观测与冻结模型评分 pilot", "", f"- Git HEAD: `{_git_head()}`", f"- 前端窗口结果：{frontend.get('completed', 0)}/{frontend.get('total', 0)}；状态：`{frontend.get('status')}`", f"- H/B 特征窗口：{feature.get('windows', 0)}", "- 时间网格先于标签冻结；每个窗口独立初始化 289 查询点。", "- 真实视频窗口只作负类；fake 仅在一秒窗口完整包含于标注区间并集时作主正类；BOUNDARY_MIXED/OUTSIDE 仅描述。", "- 冻结条件：H_MEAN_A、B_MEAN_A、B_MEAN_C；仅使用 held-out source 的旧 fold 模型与原标准化。", "", "## 条件摘要", ""]
    for condition, value in summary.get("conditions", {}).items():
        lines.append(f"- `{condition}`：source mean AUROC={value.get('source_mean_auroc')}, CI={value.get('bootstrap_ci95')}, pooled AUROC={value.get('pooled_auroc')}, AP={value.get('pooled_ap')}，source 数={value.get('source_count')}。")
    lines += ["", "## 限制", "", "本 pilot 不是 sealed-test、不是新模型训练，也没有空间真值。无分数窗口区分为无支撑/缺模型/前端失败，不填零。可评分时间点是离散 model-used PTS，不等同连续像素覆盖。"]
    (root / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def reproduce_old_score(root: Path) -> dict[str, Any]:
    """Check the deserializer/scorer against one previously saved OOF row."""

    old_feature = SOURCE_ROOT / "features/0001_MANIP_25__fake.json"
    old_oof = SOURCE_ROOT / "scores/oof_window_scores.csv"
    if not old_feature.is_file() or not old_oof.is_file():
        return {"status": "UNAVAILABLE"}
    feature = json.loads(old_feature.read_text(encoding="utf-8"))
    with old_oof.open(newline="", encoding="utf-8") as handle:
        expected_rows = {row["window_id"]: row for row in csv.DictReader(handle)}
    expected = expected_rows.get("0001_MANIP_25::fake", {}).get("H_MEAN_A_seed_20260909")
    models, _ = _load_models()
    item = models.get(("H_MEAN_A", "01KML", 20260909))
    if expected is None or item is None:
        return {"status": "UNAVAILABLE"}
    model, standardizer = item
    example = {"window_id": "0001_MANIP_25::fake", "source_id": "01KML", "role": "fake", "kind": "MANIP", "triplets": feature["h_support"]["triplets"]}
    batch, _ = build_batch([example], "UNORDERED_STATE", standardizer=standardizer)
    actual = float(score_pooling_model(model, batch)[0])
    result = {"status": "CHECKED", "window_id": example["window_id"], "condition": "H_MEAN_A", "seed": 20260909, "expected_oof_logit": float(expected), "recomputed_logit": actual, "absolute_error": abs(actual - float(expected)), "tolerance": 1e-6, "pass": abs(actual - float(expected)) <= 1e-6}
    _atomic_json(root / "evaluation/old_score_reproduction.json", result)
    return result


def run(root: Path, *, resume: bool, budget_s: float, max_frontend_windows: int | None = None) -> dict[str, Any]:
    root.mkdir(parents=True, exist_ok=True)
    if not (root / "manifests/grid_windows.json").is_file() or not resume:
        prepare_plan(root)
        write_protocol(root)
    frontend = run_frontend(root, budget_s=budget_s, resume=resume, max_windows=max_frontend_windows)
    if max_frontend_windows is not None:
        return {"frontend": frontend, "status": frontend.get("status")}
    feature = build_features(root, resume=resume)
    scores = score_windows(root, resume=resume)
    reproduction = reproduce_old_score(root)
    summary = evaluate(root)
    build_review(root)
    write_report(root, frontend, feature, summary)
    _atomic_json(root / "final_status.json", {"status": "COMPLETE" if frontend.get("status") == "COMPLETE" else frontend.get("status"), "frontend": frontend, "feature": feature, "scores": scores, "updated_unix": time.time()})
    return {"frontend": frontend, "feature": feature, "scores": scores, "old_score_reproduction": reproduction, "status": frontend.get("status")}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--budget-s", type=float, default=7200.0)
    parser.add_argument("--max-frontend-windows", type=int, default=None, help="bounded smoke run; do not use for the full pilot")
    parser.add_argument("--phase", choices=("plan", "run"), default="run")
    args = parser.parse_args()
    if args.phase == "plan":
        args.output_root.mkdir(parents=True, exist_ok=True); prepare_plan(args.output_root); write_protocol(args.output_root); print(json.dumps({"status": "PLANNED", "output": str(args.output_root)}, indent=2)); return
    print(json.dumps(run(args.output_root, resume=args.resume, budget_s=args.budget_s, max_frontend_windows=args.max_frontend_windows), indent=2, default=_json_default), flush=True)


if __name__ == "__main__":
    main()
