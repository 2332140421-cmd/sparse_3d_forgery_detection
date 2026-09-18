"""V7 matched 17x17 versus nested 33x33 density training pilot.

This is a deliberately thin experiment entry point.  The expensive input is
made from the already frozen source128 parent clips: the old 289-point R UV
and visibility are reused, while a nested 1089-point query is run with the
same tracker.  One depth/pose pass per parent is shared by both densities.
All downstream grouping, five-time support, S/Q construction and the
STRUCTURE_SUPPORT model are the existing V7 implementations.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import signal
import subprocess
import time
import traceback
from collections import Counter, defaultdict
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

from research_tools.v7.observation_support_pilot import runner as observation
from research_tools.v7.observation_density_diagnostic import run_diagnostic as diagnostic
from research_tools.v7.periodic_requery_probe import runner as periodic
from research_tools.v7.query_density_probe.pilot import nested_axes


REPO_ROOT = Path(__file__).resolve().parents[3]
DATA_ROOT = Path("/root/autodl-tmp/data/sparse_3d_forgery_detection")
SOURCE_ROOT = DATA_ROOT / "derived/v7_activityforensics_source128_extension_v1"
OUTPUT_ROOT = DATA_ROOT / "derived/v7_activityforensics_density_training_pilot_v1"
SELECTION_SEED = 20260909
MODEL_SEEDS = (20260909, 20260910, 20260911)
EPOCHS = 200
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
BOOTSTRAP_SEED = 20260909
BOOTSTRAP_REPLICATES = 10_000
PROCESS_SIZE = 256
R17_COUNT = 289
R33_COUNT = 1089
DENSITIES = ("R17", "R33")
CONDITIONS = ("D17_TRAIN", "D33_TRAIN")
VALIDATION_IDS_EXPECTED = 83
MAX_TARGET_ERROR_S = 0.05
STOP_REQUESTED = False


def _jsonable(value: Any, path: str = "$") -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return [_jsonable(item, f"{path}[{i}]") for i, item in enumerate(value)]
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return _jsonable(value.item(), path)
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"non-finite JSON value at {path}: {value!r}")
        return value
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item, f"{path}.{key}") for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item, f"{path}[{i}]") for i, item in enumerate(value)]
    return value


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(_jsonable(value), indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
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


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_head() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()


def safe(value: str) -> str:
    return "".join(c if c.isalnum() or c in "._-" else "_" for c in value)


def progress(root: Path, stage: str, status: str, completed: int, total: int, **extra: Any) -> None:
    atomic_json(root / "progress.json", {"stage": stage, "status": status, "completed": int(completed), "total": int(total), "updated_unix": time.time(), **extra})


def _load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _load_plan_inputs() -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    train_sources = [dict(row) for row in csv.DictReader((SOURCE_ROOT / "manifests/training_source_manifest.csv").open(newline="", encoding="utf-8"))]
    validation_rows = [dict(row) for row in csv.DictReader((SOURCE_ROOT / "manifests/validation_window_manifest.csv").open(newline="", encoding="utf-8"))]
    subwindows = [dict(row) for row in _load_json(SOURCE_ROOT / "manifests/subwindows.json")]
    parents = {str(row["parent_id"]): dict(row) for row in _load_json(SOURCE_ROOT / "manifests/parents.json")}
    support_rows = [dict(row) for row in _load_json(SOURCE_ROOT / "support/window_support.json") if str(row.get("mode")) == "R"]
    support = {str(row["window_id"]): row for row in support_rows}
    if len(train_sources) != 128:
        raise ValueError(f"SOURCE128_TRAINING_SOURCE_COUNT:{len(train_sources)}")
    if len(validation_rows) != VALIDATION_IDS_EXPECTED:
        raise ValueError(f"VALIDATION_MANIFEST_COUNT:{len(validation_rows)}")
    return train_sources, validation_rows, subwindows, parents, support


def choose_sources(train_sources: Sequence[Mapping[str, Any]], support: Mapping[str, Mapping[str, Any]]) -> tuple[list[str], list[str], dict[str, Any]]:
    training_ids = {str(row["source_id"]) for row in train_sources}
    effective = sorted({str(row["source_id"]) for row in support.values() if str(row["source_id"]) in training_ids and int(row.get("valid_unit_count", 0)) > 0 and row.get("label") in (0, 1) and row.get("features", {}).get("SET_A") is not None})
    effective = [sid for sid in effective if sid != "04LAX"]
    ordered = sorted(effective, key=lambda sid: hashlib.sha256(f"v7-density-train-{SELECTION_SEED}:{sid}".encode("utf-8")).hexdigest())
    if len(ordered) < 32:
        raise ValueError(f"EFFECTIVE_TRAIN_SOURCES_INSUFFICIENT:{len(ordered)}")
    selected = ordered[:32]
    # ``training_subsets.json`` stores validation source IDs directly (not
    # row objects).  Keep the frozen 16-source validation plan even though
    # the legacy 83-window evaluation manifest has no rows for two of them.
    validation_sources = sorted({str(source_id) for source_id in _load_json(SOURCE_ROOT / "manifests/training_subsets.json")["validation_sources"]})
    if set(selected) & set(validation_sources):
        raise ValueError("TRAIN_VALIDATION_SOURCE_OVERLAP")
    return selected, validation_sources, {"effective_train_sources": effective, "selection_key": "sha256('v7-density-train-20260909:'+source_id)", "selection_seed": SELECTION_SEED}


def load_or_freeze_plan(root: Path, *, overwrite: bool = False) -> dict[str, Any]:
    protocol_path = root / "protocol.json"
    if protocol_path.is_file() and not overwrite:
        return _load_json(protocol_path)
    train_sources, validation_rows, subwindows, parents, support = _load_plan_inputs()
    selected, validation_sources, selection_info = choose_sources(train_sources, support)
    selected_all = set(selected) | set(validation_sources)
    planned = [row for row in subwindows if str(row.get("source_id")) in selected_all and str(row.get("kind")) == "MANIP"]
    planned.sort(key=lambda row: str(row["window_id"]))
    if len(planned) != 288:
        raise ValueError(f"PLANNED_WINDOW_COUNT:{len(planned)}")
    validation_ids = [str(row["window_id"]) for row in validation_rows]
    missing_val = sorted(set(validation_ids) - {str(row["window_id"]) for row in planned})
    if missing_val:
        raise ValueError(f"VALIDATION_WINDOWS_NOT_IN_PLAN:{missing_val[:3]}")
    root.joinpath("manifests").mkdir(parents=True, exist_ok=True)
    sample = {
        "train_sources": selected,
        "validation_sources": validation_sources,
        "planned_windows": planned,
        "validation_window_ids": validation_ids,
        "nominal": {"train_source_count": 32, "validation_source_count": len(validation_sources), "source_role_offset_window_count": len(planned), "density_count": 2},
        "selection": selection_info,
        "source128_protocol_sha256": sha256(SOURCE_ROOT / "protocol.json"),
    }
    atomic_json(root / "manifests/sample_manifest.json", sample)
    atomic_json(root / "manifests/common_validation_windows.json", validation_ids)
    atomic_json(root / "protocol.json", {
        "protocol_id": "v7-density-training-pilot-v1",
        "git_head": git_head(),
        "source128_root": str(SOURCE_ROOT),
        "sample_manifest_sha256": sha256(root / "manifests/sample_manifest.json"),
        "selection": {"train_source_count": 32, "train_sources": selected, "validation_sources": validation_sources, **selection_info, "04LAX_excluded": True},
        "windows": {"rule": "existing source128 MANIP50", "offsets_s": [0.0, 0.5, 1.0], "window_count": len(planned), "label_rule": "real=0; fake=1 only for existing label-qualified windows; boundary/outside remain descriptive", "validation_window_ids": validation_ids},
        "densities": {"R17": {"grid_size": 17, "query_count": 289, "source": "source128 legal R caches, geometry rebuilt with shared parent depth/pose"}, "R33": {"grid_size": 33, "query_count": 1089, "original_subset_count": 289, "added_count": 800, "axis": "R17 process axis with adjacent midpoints", "source": "new tracker query"}},
        "frontend": {"tracker": "online_bootstapir", "tracker_source_sha": diagnostic.TRACKER_SHA, "checkpoint": str(diagnostic.TAPNET_CHECKPOINT), "depth": "apple_depth_pro", "depth_source_sha": diagnostic.DEPTH_SHA, "process_size": PROCESS_SIZE, "shared_depth_pose_per_parent": True, "query_chunk_size": 64, "no_interpolation_or_zero_fill": True},
        "support": {"history_and_five_time_rule": "existing build_local_groups/build_five_time_unit", "q": "raw historical group member denominator; target visibility/geometry fractions minus last history reference", "minimum_common_members": 3, "max_target_error_s": MAX_TARGET_ERROR_S},
        "training": {"conditions": list(CONDITIONS), "model": "ModalSetModel(8), existing STRUCTURE_SUPPORT", "epochs": EPOCHS, "optimizer": "Adam", "learning_rate": LEARNING_RATE, "weight_decay": WEIGHT_DECAY, "seeds": list(MODEL_SEEDS), "weighted_bce": "source/class balanced window loss, mean normalized", "standardization": "each density fitted on matched training windows only", "mean_local_to_window": True},
        "evaluation": {"primary": "D33/D33 - D17/D17 source-macro AUROC on common dual-density validation windows", "cross_readout": "2x2 train density x validation density with own standardizer", "bootstrap": {"unit": "source", "replicates": BOOTSTRAP_REPLICATES, "seed": BOOTSTRAP_SEED}, "threshold": "logit >= 0", "validation_is_development": True},
        "boundaries": ["no third density", "no new architecture", "no ROI", "no old R7/V5", "no formal src changes", "not sealed test", "no spatial ground truth"],
    })
    write_csv(root / "manifests/common_train_windows.csv", [])
    write_csv(root / "manifests/validation_window_manifest.csv", validation_rows)
    write_csv(root / "manifests/planned_windows.csv", planned)
    return _load_json(protocol_path)


def _load_sample(root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, dict[str, Any]], dict[str, dict[str, Any]], list[dict[str, Any]]]:
    sample = _load_json(root / "manifests/sample_manifest.json")
    train_rows, validation_rows, subwindows, parents, support = _load_plan_inputs()
    train_sources = set(str(x) for x in sample["train_sources"])
    val_ids = set(str(x) for x in sample["validation_window_ids"])
    planned_by_id = {str(row["window_id"]): row for row in subwindows if str(row.get("window_id")) in {str(x["window_id"]) for x in sample["planned_windows"]}}
    if len(planned_by_id) != int(sample.get("nominal", {}).get("source_role_offset_window_count", 288)):
        raise ValueError(f"FROZEN_PLAN_MISMATCH:{len(planned_by_id)}")
    planned = [dict(planned_by_id[key]) for key in sorted(planned_by_id)]
    train = [dict(row) for row in planned if str(row["source_id"]) in train_sources and row.get("label") in (0, 1)]
    validation = [dict(row) for row in planned if str(row["window_id"]) in val_ids]
    train.sort(key=lambda row: str(row["window_id"]))
    validation.sort(key=lambda row: str(row["window_id"]))
    return train, validation, parents, support, planned


def _positions(parent_frames: Sequence[int], frames: Sequence[int]) -> list[int]:
    lookup = {int(frame): index for index, frame in enumerate(parent_frames)}
    try:
        return [lookup[int(frame)] for frame in frames]
    except KeyError as exc:
        raise ValueError(f"WINDOW_FRAME_NOT_IN_PARENT:{exc}") from exc


def _slice(decoded: DecodedVideoSample, positions: Sequence[int], sample_id: str) -> DecodedVideoSample:
    return DecodedVideoSample(sample_id=sample_id, source_video_id=decoded.source_video_id, frames=tuple(decoded.frames[int(pos)] for pos in positions))


def _dense_queries() -> np.ndarray:
    _, dense, _ = nested_axes(PROCESS_SIZE)
    uu, vv = np.meshgrid(dense, dense)
    return np.stack((uu.ravel(), vv.ravel()), axis=1).astype(np.float32)


def _track_with_queries(tracker: Any, decoded: DecodedVideoSample, process_uv: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    torch = tracker.torch
    source_h, source_w = decoded.frames[0].rgb.shape[:2]
    if any(frame.rgb.shape[:2] != (source_h, source_w) for frame in decoded.frames):
        raise ValueError("VARIABLE_RASTER_SIZE")
    process_uv = np.asarray(process_uv, dtype=np.float32)
    if process_uv.ndim != 2 or process_uv.shape[1:] != (2,) or not np.all(np.isfinite(process_uv)):
        raise ValueError("QUERY_ARRAY_INVALID")
    query = np.stack((np.zeros(process_uv.shape[0], np.float32), process_uv[:, 1], process_uv[:, 0]), axis=1)
    frames = np.stack([frame.rgb for frame in decoded.frames])
    tensor = torch.from_numpy(frames).to(tracker.device).permute(0, 3, 1, 2).float()
    tensor = tracker.functional.interpolate(tensor, size=(tracker.process_size, tracker.process_size), mode="bilinear", align_corners=False).permute(0, 2, 3, 1)
    tensor = tensor / 255 * 2 - 1
    query_tensor = torch.from_numpy(query).to(tracker.device)
    with torch.inference_mode():
        first = tensor[None, 0:1]
        feature_grids = tracker.model.get_feature_grids(first, is_training=False)
        query_features = tracker.model.get_query_features(first, is_training=False, query_points=query_tensor[None], feature_grids=feature_grids)
        causal_state = tracker.model.construct_initial_causal_state(query.shape[0], len(query_features.resolutions) - 1)
        causal_state = tracker.tree.map_structure(lambda value: value.to(tracker.device), causal_state)
        tracks: list[np.ndarray] = []
        visibility: list[np.ndarray] = []
        for frame in tensor:
            single = frame[None, None]
            grids = tracker.model.get_feature_grids(single, is_training=False)
            result = tracker.model.estimate_trajectories(single.shape[-3:-1], is_training=False, feature_grids=grids, query_features=query_features, query_points_in_video=None, query_chunk_size=64, causal_context=causal_state, get_causal_context=True)
            causal_state = result["causal_context"]
            tracks.append(result["tracks"][-1][0, :, 0].detach().cpu().numpy())
            occlusion = result["occlusion"][-1][0, :, 0]
            expected = result["expected_dist"][-1][0, :, 0]
            visible = ((1 - torch.sigmoid(occlusion)) * (1 - torch.sigmoid(expected)) > 0.5)
            visibility.append(visible.detach().cpu().numpy())
    from sparse3d_forgery.experiments.v7_explicit_geometry_frontend import RasterScale
    mapping = RasterScale((source_h, source_w), (tracker.process_size, tracker.process_size))
    uv = mapping.process_to_source(np.asarray(tracks, dtype=np.float32))
    vis = np.asarray(visibility, dtype=bool)
    inside = np.all(np.isfinite(uv), axis=-1) & (uv[..., 0] >= 0) & (uv[..., 0] < source_w) & (uv[..., 1] >= 0) & (uv[..., 1] < source_h)
    vis &= inside
    uv[~vis] = np.nan
    return uv.astype(np.float32), vis


def _build_seq(decoded: DecodedVideoSample, uv: np.ndarray, visibility: np.ndarray, xyz: np.ndarray, geometry_valid: np.ndarray, row: Mapping[str, Any], density: str, cohort: str, query_start_s: float) -> Any:
    xyz_array = np.asarray(xyz)
    geo = np.asarray(geometry_valid, dtype=bool)
    if xyz_array.shape[:2] != geo.shape or xyz_array.shape[-1] != 3:
        raise ValueError("XYZ_GEOMETRY_SHAPE_MISMATCH")
    xyz_array = xyz_array.astype(np.float32, copy=False)
    finite_xyz = np.all(np.isfinite(xyz_array), axis=-1)
    if np.any(geo & ~finite_xyz) or np.any(~geo & ~np.all(np.isnan(xyz_array), axis=-1)):
        raise ValueError("XYZ_MASK_FINITE_CONTRACT")
    count = int(uv.shape[1])
    return build_particle_sequence(
        decoded,
        track_ids=np.arange(count, dtype=np.int64),
        xyz=xyz_array,
        uv=np.asarray(uv, dtype=np.float32),
        visibility=np.asarray(visibility, dtype=bool),
        geometry_validity=geo,
        coordinate_system=CoordinateSystem(frame_name="first_camera_world", handedness=Handedness.RIGHT, axis_directions=("right", "down", "forward"), length_unit=LengthUnit.METER, camera_motion_compensated=True, normalization={}),
        lineage={"dataset": "ActivityForensics+Charades", "official_split": "train", "source_id": str(row["source_id"]), "pair_id": str(row["pair_id"]), "role": str(row["role"]), "parent_id": str(row["parent_id"]), "window_id": str(row["window_id"]), "query_cohort": cohort},
        provenance={"tracker": "online_bootstapir", "tracker_source_sha": diagnostic.TRACKER_SHA, "depth": "apple_depth_pro", "depth_source_sha": diagnostic.DEPTH_SHA, "pose": "open3d_rgbd_odometry", "process_size": PROCESS_SIZE, "query_grid_size": 17 if density == "R17" else 33, "query_count": count, "query_cohort": cohort, "cohort_start_s": float(query_start_s), "cross_cohort_identity": False, "density_training_pilot": "v1"},
    )


def _sequence_valid(prefix: Path, row: Mapping[str, Any], density: str, expected_count: int, expected_frame_indices: Sequence[int] | None = None, expected_timestamps_s: Sequence[float] | None = None) -> bool:
    try:
        seq = load_particle_sequence(prefix)
        expected_frames = np.asarray(expected_frame_indices if expected_frame_indices is not None else row["frame_indices"], dtype=np.int64)
        if int(seq.num_tracks) != expected_count or not np.array_equal(seq.frame_indices, expected_frames):
            return False
        expected_timestamps = np.asarray(expected_timestamps_s if expected_timestamps_s is not None else row["timestamps_s"], dtype=np.float64)
        if seq.timestamps_s.shape != expected_timestamps.shape or not np.allclose(seq.timestamps_s, expected_timestamps, atol=1e-7, rtol=0):
            return False
        expected_video = f"{row['source_id']}::{row['role']}"
        if str(seq.source_video_id) != expected_video or str(seq.lineage.get("window_id")) != str(row["window_id"]):
            return False
        if int(seq.provenance.get("query_count", -1)) != expected_count:
            return False
        if str(seq.provenance.get("query_cohort")) not in {"O", "R"}:
            return False
        return True
    except (OSError, ValueError, KeyError, TypeError):
        return False


def _geometry_for_parent(decoded: DecodedVideoSample, depth_runner: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    from scripts.run_v7_explicit_geometry_frontend import accumulate_world_from_camera, causal_first_frame_intrinsics, rgbd_odometry
    depths, focals, depth_valid = depth_runner.infer(decoded)
    intrinsics, _ = causal_first_frame_intrinsics(depths, focals)
    adjacent, pair_valid, _ = rgbd_odometry(decoded, depths, intrinsics)
    world, pose_valid = accumulate_world_from_camera(adjacent, pair_valid)
    return depths, depth_valid, intrinsics, world, pose_valid, pair_valid


def _make_xyz(uv: np.ndarray, vis: np.ndarray, positions: Sequence[int], geometry: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    from scripts.run_v7_explicit_geometry_frontend import sample_depth_at_uv, world_xyz
    depths, depth_valid, intrinsics, world, pose_valid, _ = geometry
    pos = np.asarray(positions, dtype=np.int64)
    depths_s = depths[pos]; intrinsics_s = intrinsics[pos]; world_s = world[pos]; pose_s = pose_valid[pos]; depth_valid_s = depth_valid[pos]
    sampled, sampled_valid = sample_depth_at_uv(depths_s, uv)
    observation_valid = np.asarray(vis, dtype=bool) & sampled_valid & depth_valid_s[:, None]
    xyz, valid = world_xyz(uv, sampled, intrinsics_s, world_s, observation_valid, pose_s)
    return np.asarray(xyz, dtype=np.float32), np.asarray(valid, dtype=bool)


def frontend_smoke(root: Path, train_rows: Sequence[Mapping[str, Any]], parents: Mapping[str, Mapping[str, Any]], support: Mapping[str, Mapping[str, Any]], *, device: str = "cuda") -> dict[str, Any]:
    """Run one real/fake pair, preserving output separately from formal data."""
    import torch
    smoke_root = root / "smoke"
    existing = smoke_root / "summary.json"
    if existing.is_file():
        cached = _load_json(existing)
        if cached.get("status") == "PASS" and all(
            Path(str(item.get("r17_sequence_prefix", ""))).with_suffix(".npz").is_file()
            and Path(str(item.get("r17_sequence_prefix", ""))).with_suffix(".json").is_file()
            and Path(str(item.get("r33_sequence_prefix", ""))).with_suffix(".npz").is_file()
            and Path(str(item.get("r33_sequence_prefix", ""))).with_suffix(".json").is_file()
            for item in cached.get("rows", [])
        ):
            return cached
    chosen: list[Mapping[str, Any]] = []
    for source in sorted({str(row["source_id"]) for row in train_rows}):
        pair = [row for row in train_rows if str(row["source_id"]) == source]
        offsets = sorted({float(row["offset_s"]) for row in pair})
        for offset in offsets:
            same_offset = [row for row in pair if float(row["offset_s"]) == offset]
            by_role = {str(row["role"]): row for row in same_offset}
            if set(by_role) == {"real", "fake"}:
                chosen = [by_role["real"], by_role["fake"]]
                break
        if chosen:
            break
    if len(chosen) != 2:
        raise ValueError("SMOKE_REAL_FAKE_PAIR_MISSING")
    from scripts.run_v7_explicit_geometry_frontend import DepthProRunner, OnlineBootsTapir
    tracker = OnlineBootsTapir(diagnostic.TAPNET_SOURCE, diagnostic.TAPNET_CHECKPOINT, process_size=PROCESS_SIZE, grid_size=17)
    depth_runner = DepthProRunner(diagnostic.DEPTH_SOURCE, diagnostic.DEPTH_CHECKPOINT)
    rows_out = []
    for row in chosen:
        parent = parents[str(row["parent_id"])]
        decoded = decode_video(VideoSource(sample_id=f"density-smoke-{parent['parent_id']}", source_video_id=f"{row['source_id']}::{row['role']}", source_locator=Path(str(parent["video_path"]))), [int(x) for x in parent["parent_frame_indices"]])
        geometry = _geometry_for_parent(decoded, depth_runner)
        old_seq = load_particle_sequence(Path(str(support[str(row["window_id"])] ["particle_prefix"])))
        old_frames = np.asarray(old_seq.frame_indices, dtype=np.int64)
        positions = _positions(parent["parent_frame_indices"], old_frames)
        sub = _slice(decoded, positions, f"density-smoke-{row['window_id']}")
        uv17 = np.asarray(old_seq.uv)
        vis17 = np.asarray(old_seq.visibility)
        geom_sub = tuple(
            np.asarray(value)[positions]
            if np.asarray(value).ndim > 0 and np.asarray(value).shape[0] == len(decoded.frames)
            else value
            for value in geometry
        )
        xyz17, geo17 = _make_xyz(uv17, vis17, list(range(len(old_frames))), geom_sub)
        q33 = _dense_queries()
        uv33, vis33 = _track_with_queries(tracker, sub, q33)
        geom_sub = tuple(
            np.asarray(value)[positions]
            if np.asarray(value).ndim > 0 and np.asarray(value).shape[0] == len(decoded.frames)
            else value
            for value in geometry
        )
        xyz33, geo33 = _make_xyz(uv33, vis33, list(range(len(positions))), geom_sub)
        # The helper above is intentionally exercised through ParticleSequence;
        # no smoke features are used as formal data.
        s17 = _build_seq(sub, uv17, vis17, xyz17, geo17, row, "R17", "O" if float(row["offset_s"]) == 0 else "R", float(row["interval_start_s"]))
        s33 = _build_seq(sub, uv33, vis33, xyz33, geo33, row, "R33", "O" if float(row["offset_s"]) == 0 else "R", float(row["interval_start_s"]))
        p17 = smoke_root / "particles" / f"{safe(str(row['window_id']))}_R17"
        p33 = smoke_root / "particles" / f"{safe(str(row['window_id']))}_R33"
        _save_sequence(p17, s17)
        _save_sequence(p33, s33)
        rows_out.append({"window_id": row["window_id"], "role": row["role"], "r17_frames": int(s17.num_frames), "r33_frames": int(s33.num_frames), "r17_query_count": int(s17.num_tracks), "r33_query_count": int(s33.num_tracks), "r17_geometry_valid": int(np.sum(s17.geometry_validity)), "r33_geometry_valid": int(np.sum(s33.geometry_validity)), "r17_sequence_prefix": str(p17), "r33_sequence_prefix": str(p33)})
        del decoded, sub, s17, s33
    result = {"status": "PASS", "rows": rows_out, "device": str(torch.device(device)), "formal_records_untouched": True, "nested_query_count": R33_COUNT, "original_query_count": R17_COUNT}
    atomic_json(existing, result)
    return result


def _save_sequence(prefix: Path, seq: Any) -> None:
    prefix.parent.mkdir(parents=True, exist_ok=True)
    save_particle_sequence(seq, prefix)


def _recover_frontend_row(root: Path, row: Mapping[str, Any], support: Mapping[str, Mapping[str, Any]]) -> dict[str, Any] | None:
    """Recover a successfully written pair after an interrupted metadata save."""
    old = support.get(str(row["window_id"]))
    if old is None or not old.get("particle_prefix"):
        return None
    try:
        old_seq = load_particle_sequence(Path(str(old["particle_prefix"])))
        expected_frames = np.asarray(old_seq.frame_indices, dtype=np.int64)
        parent_id = str(row["parent_id"])
        p17 = root / "particles" / safe(parent_id) / f"R17_{safe(str(row['window_id']))}"
        p33 = root / "particles" / safe(parent_id) / f"R33_{safe(str(row['window_id']))}"
        if not _sequence_valid(p17, row, "R17", R17_COUNT, expected_frames, old_seq.timestamps_s) or not _sequence_valid(p33, row, "R33", R33_COUNT, expected_frames, old_seq.timestamps_s):
            return None
        seq17 = load_particle_sequence(p17); seq33 = load_particle_sequence(p33)
        return {
            **dict(row),
            "status": "FRONTEND_COMPLETE",
            "r17_sequence_prefix": str(p17),
            "r33_sequence_prefix": str(p33),
            "r17_sequence_frame_indices": expected_frames.tolist(),
            "r33_sequence_frame_indices": expected_frames.tolist(),
            "r17_sequence_timestamps_s": np.asarray(seq17.timestamps_s, dtype=np.float64).tolist(),
            "r33_sequence_timestamps_s": np.asarray(seq33.timestamps_s, dtype=np.float64).tolist(),
            "r17_query_count": R17_COUNT,
            "r33_query_count": R33_COUNT,
            "r17_geometry_valid_fraction": float(np.mean(seq17.geometry_validity)),
            "r33_geometry_valid_fraction": float(np.mean(seq33.geometry_validity)),
            "shared_geometry": True,
            "parent_frame_count": int(expected_frames.size),
            "source128_r17_particle_prefix": str(old["particle_prefix"]),
            "r17_cache_reused_uv_visibility": True,
            "r33_nested_query_mapping": "17-axis midpoint insertion",
            "r17_original_support_set_a_available": bool(old.get("features", {}).get("SET_A") is not None),
            "recovered_from_existing_sequences": True,
        }
    except (OSError, ValueError, KeyError, TypeError):
        return None


def run_frontend(root: Path, train_rows: Sequence[Mapping[str, Any]], validation_rows: Sequence[Mapping[str, Any]], parents: Mapping[str, Mapping[str, Any]], support: Mapping[str, Mapping[str, Any]], *, planned_rows: Sequence[Mapping[str, Any]] | None = None, device: str = "cuda", resume: bool = True, limit_parents: int | None = None) -> dict[str, Any]:
    """Build both density sequences, one parent at a time, with atomic rows."""
    import torch
    global STOP_REQUESTED
    from scripts.run_v7_explicit_geometry_frontend import DepthProRunner, OnlineBootsTapir
    # Frontend processing follows the frozen nominal plan (32 train sources
    # plus all 16 validation sources), while the main evaluation later uses
    # only the 83-row validation manifest.  Keeping these populations
    # separate avoids silently dropping the two validation sources that have
    # no row in that legacy manifest.
    all_rows = list(planned_rows) if planned_rows is not None else list(train_rows) + list(validation_rows)
    selected_parent_ids = sorted({str(row["parent_id"]) for row in all_rows})
    if limit_parents is not None:
        selected_parent_ids = selected_parent_ids[:int(limit_parents)]
    total = len(selected_parent_ids)
    results_path = root / "frontend/results.json"
    existing_rows = (_load_json(results_path) if results_path.is_file() and resume else [])
    results = {str(row.get("window_id")): row for row in existing_rows if row.get("window_id") is not None}
    completed = 0
    progress(root, "frontend", "RUNNING", completed, total)
    if not torch.cuda.is_available() and str(device).startswith("cuda"):
        raise RuntimeError("CUDA_UNAVAILABLE_FRONTEND")
    tracker = OnlineBootsTapir(diagnostic.TAPNET_SOURCE, diagnostic.TAPNET_CHECKPOINT, process_size=PROCESS_SIZE, grid_size=17)
    depth_runner = DepthProRunner(diagnostic.DEPTH_SOURCE, diagnostic.DEPTH_CHECKPOINT)
    for pindex, parent_id in enumerate(selected_parent_ids, 1):
        if STOP_REQUESTED:
            break
        parent = parents[parent_id]
        rows = [row for row in all_rows if str(row["parent_id"]) == parent_id]
        existing_for_parent = {str(row["window_id"]): results.get(str(row["window_id"])) for row in rows}
        # A previous process may have written both sequence artifacts and then
        # failed before its window metadata was saved (or marked the row
        # FAILED).  Validate those artifacts by their actual sequence identity
        # and restore the metadata without rerunning the expensive frontend.
        recovered_any = False
        for row in rows:
            item = existing_for_parent[str(row["window_id"])]
            if item is None or item.get("status") != "FRONTEND_COMPLETE":
                recovered = _recover_frontend_row(root, row, support)
                if recovered is not None:
                    results[str(row["window_id"])] = recovered
                    existing_for_parent[str(row["window_id"])] = recovered
                    recovered_any = True
        if recovered_any:
            atomic_json(results_path, list(sorted(results.values(), key=lambda x: str(x.get("window_id", "")))))
        cache_valid = True
        for row in rows:
            item = existing_for_parent[str(row["window_id"])]
            if item is None or item.get("status") != "FRONTEND_COMPLETE":
                cache_valid = False
                break
            old = support.get(str(row["window_id"]))
            try:
                old_seq = load_particle_sequence(Path(str(old["particle_prefix"]))) if old and old.get("particle_prefix") else None
            except (OSError, ValueError, KeyError, TypeError):
                old_seq = None
            if old_seq is None:
                cache_valid = False
                break
            expected_frames = item.get("r17_sequence_frame_indices") or old_seq.frame_indices.tolist()
            expected_times = item.get("r17_sequence_timestamps_s") or np.asarray(old_seq.timestamps_s, dtype=np.float64).tolist()
            for density in DENSITIES:
                if not _sequence_valid(
                    Path(str(item[f"{density.lower()}_sequence_prefix"])),
                    row,
                    density,
                    R17_COUNT if density == "R17" else R33_COUNT,
                    expected_frames,
                    expected_times,
                ):
                    cache_valid = False
                    break
            if not cache_valid:
                break
        if rows and cache_valid:
            completed += 1
            continue
        started = time.perf_counter()
        atomic_json(root / "state/current_parent.json", {"status": "RUNNING", "parent_id": parent_id, "source_id": parent["source_id"], "role": parent["role"], "index": pindex, "total": total, "started_unix": time.time()})
        try:
            source_path = Path(str(parent["video_path"]))
            if not source_path.is_file():
                raise FileNotFoundError(f"SOURCE_MISSING:{source_path}")
            frame_indices = [int(x) for x in parent["parent_frame_indices"]]
            decoded = decode_video(VideoSource(sample_id=f"density-{parent_id}", source_video_id=f"{parent['source_id']}::{parent['role']}", source_locator=source_path), frame_indices)
            geometry = _geometry_for_parent(decoded, depth_runner)
            for row in rows:
                old = support.get(str(row["window_id"]))
                if old is None or not old.get("particle_prefix"):
                    raise ValueError(f"R17_SOURCE128_SUPPORT_MISSING:{row['window_id']}")
                old_seq = load_particle_sequence(Path(str(old["particle_prefix"])))
                old_frames = np.asarray(old_seq.frame_indices, dtype=np.int64)
                row_frames = np.asarray(row["frame_indices"], dtype=np.int64)
                if not np.all(np.isin(row_frames, old_frames)):
                    raise ValueError(f"R17_FRAME_IDENTITY_MISMATCH:{row['window_id']}:window_not_in_cached_sequence")
                old_positions = _positions(frame_indices, old_frames)
                sub = _slice(decoded, old_positions, f"density-{row['window_id']}")
                uv17 = np.asarray(old_seq.uv, dtype=np.float32); vis17 = np.asarray(old_seq.visibility, dtype=bool)
                geom_sub = tuple(
                    np.asarray(value)[old_positions]
                    if np.asarray(value).ndim > 0 and np.asarray(value).shape[0] == len(frame_indices)
                    else value
                    for value in geometry
                )
                xyz17, geo17 = _make_xyz(uv17, vis17, list(range(len(old_frames))), geom_sub)
                cohort = str(old_seq.provenance.get("query_cohort", ""))
                if cohort not in {"O", "R"}:
                    raise ValueError(f"R17_QUERY_COHORT_MISSING:{row['window_id']}")
                seq17 = _build_seq(sub, uv17, vis17, xyz17, geo17, row, "R17", cohort, float(row["interval_start_s"]))
                q33 = _dense_queries()
                uv33, vis33 = _track_with_queries(tracker, sub, q33)
                xyz33, geo33 = _make_xyz(uv33, vis33, list(range(len(old_frames))), geom_sub)
                seq33 = _build_seq(sub, uv33, vis33, xyz33, geo33, row, "R33", cohort, float(row["interval_start_s"]))
                p17 = root / "particles" / safe(parent_id) / f"R17_{safe(str(row['window_id']))}"
                p33 = root / "particles" / safe(parent_id) / f"R33_{safe(str(row['window_id']))}"
                _save_sequence(p17, seq17); _save_sequence(p33, seq33)
                old_s = np.asarray(old.get("features", {}).get("SET_A"), dtype=np.float64) if old.get("features", {}).get("SET_A") is not None else np.empty((0, 5, 4))
                result_row = {**dict(row), "status": "FRONTEND_COMPLETE", "r17_sequence_prefix": str(p17), "r33_sequence_prefix": str(p33), "r17_sequence_frame_indices": old_frames.tolist(), "r33_sequence_frame_indices": np.asarray(seq33.frame_indices, dtype=np.int64).tolist(), "r17_sequence_timestamps_s": np.asarray(seq17.timestamps_s, dtype=np.float64).tolist(), "r33_sequence_timestamps_s": np.asarray(seq33.timestamps_s, dtype=np.float64).tolist(), "r17_query_count": R17_COUNT, "r33_query_count": R33_COUNT, "r17_geometry_valid_fraction": float(np.mean(seq17.geometry_validity)), "r33_geometry_valid_fraction": float(np.mean(seq33.geometry_validity)), "shared_geometry": True, "parent_frame_count": len(frame_indices), "elapsed_s": time.perf_counter() - started, "source128_r17_particle_prefix": str(old["particle_prefix"]), "r17_cache_reused_uv_visibility": True, "r33_nested_query_mapping": "17-axis midpoint insertion", "r17_original_support_set_a_available": bool(old_s.size)}
                results[str(row["window_id"])] = result_row
                del sub, seq17, seq33
            # Save by window, not only at parent end, so a signal cannot leave
            # a false successful parent record.
            atomic_json(results_path, list(sorted(results.values(), key=lambda x: str(x.get("window_id", x.get("parent_id", ""))))))
            completed += 1
            atomic_json(root / "state/current_parent.json", {"status": "COMPLETE", "parent_id": parent_id, "source_id": parent["source_id"], "role": parent["role"], "elapsed_s": time.perf_counter() - started, "completed_unix": time.time()})
            progress(root, "frontend", "RUNNING", completed, total, current_parent=parent_id, elapsed_s=time.perf_counter() - started)
            print(f"density frontend {completed}/{total} {parent_id} elapsed={time.perf_counter()-started:.1f}s", flush=True)
            del decoded
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except BaseException as exc:
            error = {"status": "FRONTEND_FAILED", "parent_id": parent_id, "source_id": parent["source_id"], "role": parent["role"], "error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc(), "elapsed_s": time.perf_counter() - started}
            atomic_json(root / "state/current_parent.json", error)
            for row in rows:
                results[str(row["window_id"])] = {**dict(row), **error}
            atomic_json(results_path, list(sorted(results.values(), key=lambda x: str(x.get("window_id", "")))))
            progress(root, "frontend", "RUNNING", completed, total, current_parent=parent_id, last_error=error["error"])
            print(f"density frontend FAILED {parent_id}: {error['error']}", flush=True)
            # A failed parent remains retryable on resume; continue to the
            # next frozen parent rather than calling a partial result valid.
        if STOP_REQUESTED:
            break
    status = "COMPLETE" if completed == total else ("STOPPED_SAFE" if STOP_REQUESTED else "PARTIAL")
    progress(root, "frontend", status, completed, total)
    atomic_json(root / "state/frontend_summary.json", {"status": status, "completed_parents": completed, "total_parents": total, "completed_windows": sum(r.get("status") == "FRONTEND_COMPLETE" for r in results.values()), "total_windows": len(all_rows)})
    return {"status": status, "completed_parents": completed, "total_parents": total, "results": results}


def _q_for_sequence(row: Mapping[str, Any], seq: Any, support_item: Mapping[str, Any]) -> tuple[list[dict[str, Any]], np.ndarray]:
    # Build support using the current grouping/target functions, then derive Q
    # at the same output boundary as observation_support_pilot.
    history = np.flatnonzero(np.asarray(seq.timestamps_s) < float(row["interval_start_s"]) + 0.5).astype(np.int64)
    from research_tools.v7.local_organization_probe.grouping import build_local_groups, rebuild_components_fast
    from research_tools.v7.multi_order_sequence_probe.representation import condition_feature_matrix, build_five_time_unit
    components = rebuild_components_fast(seq.xyz, seq.geometry_validity, history)
    grouping = build_local_groups(seq, history, old_components=components)
    groups = {int(group["local_group_id"]): group for group in grouping.get("groups", [])}
    valid_units: list[dict[str, Any]] = []
    for group in grouping.get("groups", []):
        if not bool(group.get("retained")):
            continue
        unit = build_five_time_unit(seq, window_id=str(row["window_id"]), window_start_s=float(row["interval_start_s"]), member_slots=group["member_slots"], local_group_id=int(group["local_group_id"]), max_target_error_s=MAX_TARGET_ERROR_S)
        if unit.get("status") == "VALID":
            valid_units.append(unit)
    if not valid_units:
        return [], np.empty((0, 5, 4), dtype=np.float64)
    q_values: list[np.ndarray] = []
    for unit in valid_units:
        group = groups[int(unit["local_group_id"])]
        q, _, _ = observation._q_for_unit({"window_id": row["window_id"], "grouping": {"history_array_indices": history.tolist()}}, {"local_group_id": unit["local_group_id"], "array_indices": unit["array_indices"], "track_ids": unit["track_ids"]}, unit, seq, group)
        unit["q"] = q
        q_values.append(q)
    return valid_units, np.stack(q_values, axis=0).astype(np.float64)


def _feature_row(root: Path, density: str, row: Mapping[str, Any], seq_prefix: Path) -> dict[str, Any]:
    seq = load_particle_sequence(seq_prefix)
    units, q = _q_for_sequence(row, seq, {})
    if not units:
        return {**dict(row), "density": density, "status": "NO_VALID_FIVE_TIME_UNIT", "valid_unit_count": 0, "input_path": None, "s_shape": [0, 5, 4], "q_shape": [0, 5, 4], "unit_identities": []}
    # Existing condition_feature_matrix is the sole S implementation.
    from research_tools.v7.multi_order_sequence_probe.representation import condition_feature_matrix
    built = condition_feature_matrix({"units": units}, str(row["window_id"]))
    s = np.asarray(built["features"]["SET_A"], dtype=np.float32)
    if s.shape != q.shape or not np.all(np.isfinite(s)) or not np.all(np.isfinite(q)):
        raise ValueError(f"FEATURE_NONFINITE_OR_SHAPE:{density}:{row['window_id']}")
    rel = Path("features") / density / f"{safe(str(row['window_id']))}.npz"
    path = root / rel; path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp.npz")
    np.savez_compressed(temporary, s=s, q=q.astype(np.float32), intervals=np.diff(np.asarray(units[0]["timestamps_s"], dtype=np.float64)))
    os.replace(temporary, path)
    identities = []
    for index, unit in enumerate(units):
        identities.append({"unit_index": index, "local_group_id": int(unit["local_group_id"]), "member_slots": [int(x) for x in unit["member_slots"]], "track_ids": [int(x) for x in unit["track_ids"]], "pair_ids": unit["pair_ids"], "history_scale": float(unit["history_scale"]), "history_scale_pair_time_count": int(unit["history_scale_pair_time_count"]), "array_indices": unit["array_indices"], "frame_indices": unit["frame_indices"], "timestamps_s": unit["timestamps_s"], "target_times_s": unit["target_times_s"]})
    return {**dict(row), "density": density, "status": "VALID", "valid_unit_count": int(s.shape[0]), "input_path": str(rel), "s_shape": list(s.shape), "q_shape": list(q.shape), "intervals_s": np.diff(np.asarray(units[0]["timestamps_s"], dtype=np.float64)).tolist(), "unit_identities": identities, "particle_sequence_prefix": str(seq_prefix)}


def build_features(root: Path, train_rows: Sequence[Mapping[str, Any]], validation_rows: Sequence[Mapping[str, Any]], *, planned_rows: Sequence[Mapping[str, Any]] | None = None, resume: bool = True) -> dict[str, Any]:
    frontend = _load_json(root / "frontend/results.json")
    by_window = {str(item["window_id"]): item for item in frontend if str(item.get("status")) == "FRONTEND_COMPLETE"}
    all_rows = list(planned_rows) if planned_rows is not None else list(train_rows) + list(validation_rows)
    train_ids = {str(row["window_id"]) for row in train_rows}
    manifests: dict[str, list[dict[str, Any]]] = {density: [] for density in DENSITIES}
    total = len(all_rows) * 2
    done = 0
    for row in sorted(all_rows, key=lambda x: str(x["window_id"])):
        front = by_window.get(str(row["window_id"]))
        for density in DENSITIES:
            if front is None:
                feature = {**dict(row), "density": density, "status": "MISSING_FRONTEND", "valid_unit_count": 0, "input_path": None, "s_shape": [0, 5, 4], "q_shape": [0, 5, 4], "unit_identities": []}
            else:
                prefix_key = f"{density.lower()}_sequence_prefix"
                prefix = Path(str(front[prefix_key]))
                rel_manifest = root / "manifests" / f"feature_manifest_{density}.json"
                existing_rows = {str(item["window_id"]): item for item in (_load_json(rel_manifest) if resume and rel_manifest.is_file() else [])}
                if str(row["window_id"]) in existing_rows and existing_rows[str(row["window_id"])].get("status") == "VALID" and existing_rows[str(row["window_id"])].get("input_path") and (root / str(existing_rows[str(row["window_id"])] ["input_path"])).is_file():
                    feature = existing_rows[str(row["window_id"])]
                else:
                    feature = _feature_row(root, density, row, prefix)
            manifests[density].append(feature)
            done += 1
            progress(root, "features", "RUNNING", done, total, current_window=row["window_id"], density=density)
    for density in DENSITIES:
        atomic_json(root / "manifests" / f"feature_manifest_{density}.json", manifests[density])
    common_train = []
    common_val = []
    val_ids = {str(row["window_id"]) for row in validation_rows}
    for row in all_rows:
        by_density = {density: next(item for item in manifests[density] if str(item["window_id"]) == str(row["window_id"])) for density in DENSITIES}
        if all(item.get("status") == "VALID" and int(item.get("valid_unit_count", 0)) > 0 for item in by_density.values()) and row.get("label") in (0, 1):
            if str(row["window_id"]) in val_ids:
                common_val.append(dict(row))
            elif str(row["window_id"]) in train_ids:
                common_train.append(dict(row))
    common_train.sort(key=lambda x: str(x["window_id"])); common_val.sort(key=lambda x: str(x["window_id"]))
    write_csv(root / "manifests/common_train_windows.csv", common_train)
    write_csv(root / "manifests/common_validation_windows.csv", common_val)
    _write_coverage_summary(root, manifests, {str(row["window_id"]) for row in common_train}, {str(row["window_id"]) for row in common_val})
    summary = {"planned_windows": len(all_rows), "frontend_complete_windows": len(by_window), "density_rows": {d: len(manifests[d]) for d in DENSITIES}, "valid_windows": {d: sum(item.get("status") == "VALID" for item in manifests[d]) for d in DENSITIES}, "valid_units": {d: sum(int(item.get("valid_unit_count", 0)) for item in manifests[d]) for d in DENSITIES}, "common_train_windows": len(common_train), "common_validation_windows": len(common_val), "common_train_real": sum(int(item["label"]) == 0 for item in common_train), "common_train_fake": sum(int(item["label"]) == 1 for item in common_train), "common_validation_real": sum(int(item["label"]) == 0 for item in common_val), "common_validation_fake": sum(int(item["label"]) == 1 for item in common_val)}
    atomic_json(root / "support/summary.json", summary)
    progress(root, "features", "COMPLETE", done, total, **summary)
    return summary


def _write_coverage_summary(root: Path, manifests: Mapping[str, Sequence[Mapping[str, Any]]], train_ids: set[str], validation_ids: set[str]) -> None:
    """Persist small, explicit density coverage counts without changing inputs."""
    rows: list[dict[str, Any]] = []
    for density, items in manifests.items():
        for split, wanted in (("train_common", train_ids), ("validation_common", validation_ids), ("planned", None)):
            selected = [dict(item) for item in items if wanted is None or str(item.get("window_id")) in wanted]
            valid = [item for item in selected if item.get("status") == "VALID" and int(item.get("valid_unit_count", 0)) > 0]
            status_counts = Counter(str(item.get("status")) for item in selected if item.get("status") != "VALID")
            rows.append({
                "density": density,
                "split": split,
                "planned_windows": len(selected),
                "valid_windows": len(valid),
                "invalid_windows": len(selected) - len(valid),
                "valid_units": sum(int(item.get("valid_unit_count", 0)) for item in valid),
                "real_valid_windows": sum(item.get("label") == 0 for item in valid),
                "fake_valid_windows": sum(item.get("label") == 1 for item in valid),
                "invalid_statuses": ";".join(f"{key}:{value}" for key, value in sorted(status_counts.items())),
            })
    write_csv(root / "evaluation/coverage_summary.csv", rows)


def _load_feature_rows(root: Path, density: str, split: str) -> list[dict[str, Any]]:
    rows = [dict(row) for row in _load_json(root / "manifests" / f"feature_manifest_{density}.json") if str(row.get("status")) == "VALID" and int(row.get("valid_unit_count", 0)) > 0 and row.get("label") in (0, 1)]
    train_ids = {str(row["window_id"]) for row in csv.DictReader((root / "manifests/common_train_windows.csv").open(newline="", encoding="utf-8"))}
    val_ids = {str(row["window_id"]) for row in csv.DictReader((root / "manifests/common_validation_windows.csv").open(newline="", encoding="utf-8"))}
    wanted = train_ids if split == "train" else val_ids
    return [row for row in rows if str(row["window_id"]) in wanted]


def _feature_hash(root: Path, rows: Sequence[Mapping[str, Any]]) -> str:
    digest = hashlib.sha256()
    for row in sorted(rows, key=lambda x: str(x["window_id"])):
        path = root / str(row["input_path"]); digest.update(str(row["window_id"]).encode()); digest.update(path.read_bytes())
    return digest.hexdigest()


def _make_observation_rows(root: Path, density: str, rows: Sequence[Mapping[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    values = [dict(row) for row in rows]
    for row in values:
        arrays = np.load(root / str(row["input_path"]), allow_pickle=False)
        row["features"] = {"SET_A": np.asarray(arrays["s"], dtype=np.float64), "q": np.asarray(arrays["q"], dtype=np.float64)}
        row["intervals_s"] = np.asarray(arrays["intervals"], dtype=np.float64).tolist()
        arrays.close()
    weights = periodic.source_class_weights(values)
    standardizer_s = periodic.fit_standardizer("SET_A", [row["features"]["SET_A"] for row in values], weights)
    standardizer_q = periodic.fit_standardizer("SET_A", [row["features"]["q"] for row in values], weights)
    # observation_support._batch looks for q via its own standardizer and a
    # fixed condition.  We keep the schema explicit and build the fused state.
    manifest = [{"window_id": row["window_id"], "input_path": row["input_path"]} for row in values]
    return values, standardizer_s, standardizer_q


def _batch(root: Path, rows: Sequence[Mapping[str, Any]], standardizer_s: Any, standardizer_q: Any) -> dict[str, Any]:
    s_parts: list[np.ndarray] = []; q_parts: list[np.ndarray] = []; intervals: list[np.ndarray] = []; indices: list[int] = []; labels: list[int] = []; ids: list[str] = []
    for wi, row in enumerate(rows):
        with np.load(root / str(row["input_path"]), allow_pickle=False) as arrays:
            s = standardizer_s.transform(np.asarray(arrays["s"], dtype=np.float64)); q = standardizer_q.transform(np.asarray(arrays["q"], dtype=np.float64)); inter = np.asarray(arrays["intervals"], dtype=np.float64)
        if s.shape != q.shape or s.ndim != 3 or s.shape[1:] != (5, 4):
            raise ValueError(f"BATCH_SHAPE:{row['window_id']}")
        s_parts.append(s); q_parts.append(q); intervals.extend([inter] * s.shape[0]); indices.extend([wi] * s.shape[0]); labels.append(int(row["label"])); ids.append(str(row["window_id"]))
    if not ids:
        raise ValueError("EMPTY_BATCH")
    return {"states": np.concatenate([np.concatenate(s_parts, axis=0), np.concatenate(q_parts, axis=0)], axis=-1), "intervals": np.asarray(intervals, dtype=np.float64), "window_index": np.asarray(indices, dtype=np.int64), "labels": np.asarray(labels, dtype=np.int64), "window_ids": ids, "window_rows": list(rows), "window_count": len(ids), "window_weights": periodic.source_class_weights(rows)}


def smoke_model_pipeline(root: Path, smoke: Mapping[str, Any], train_rows: Sequence[Mapping[str, Any]], *, device: str = "cuda") -> dict[str, Any]:
    """Exercise feature construction, one training step, save/reload and readout.

    The smoke artifacts are deliberately isolated below ``smoke/`` and are
    never considered formal model records.  This catches an input/model
    wiring error before the expensive 6-model run starts.
    """
    summary_path = root / "smoke/model_summary.json"
    if summary_path.is_file():
        return _load_json(summary_path)
    by_id = {str(row["window_id"]): row for row in train_rows}
    chosen_ids = [str(item["window_id"]) for item in smoke.get("rows", [])]
    chosen = [by_id[item] for item in chosen_ids if item in by_id]
    if len(chosen) != 2 or {str(row["role"]) for row in chosen} != {"real", "fake"}:
        raise ValueError("SMOKE_MODEL_REAL_FAKE_ROWS_MISSING")
    summaries: dict[str, Any] = {}
    for density in DENSITIES:
        feature_rows: list[dict[str, Any]] = []
        for row in chosen:
            front = next(item for item in smoke["rows"] if str(item["window_id"]) == str(row["window_id"]))
            prefix = Path(str(front[f"{density.lower()}_sequence_prefix"]))
            feature_rows.append(_feature_row(root / "smoke", density, row, prefix))
        atomic_json(root / "smoke" / f"feature_manifest_{density}.json", feature_rows)
        if any(item.get("status") != "VALID" or int(item.get("valid_unit_count", 0)) <= 0 for item in feature_rows):
            raise ValueError(f"SMOKE_FEATURE_NO_SUPPORT:{density}")
        values, standardizer_s, standardizer_q = _make_observation_rows(root / "smoke", density, feature_rows)
        batch = _batch(root / "smoke", values, standardizer_s, standardizer_q)
        model, fit = observation._train_one("STRUCTURE_SUPPORT", batch, SELECTION_SEED, device, epochs=1)
        record = {
            "condition": f"SMOKE_{density}",
            "density": density,
            "seed": SELECTION_SEED,
            "standardization": {"s": standardizer_s.as_dict(), "q": standardizer_q.as_dict()},
            "state_dict": {name: value.detach().cpu().numpy().tolist() for name, value in model.state_dict().items()},
            "fit": fit,
        }
        atomic_json(root / "smoke" / f"model_{density}.json", record)
        reloaded = _load_model(record, device)
        scores = _score(reloaded, batch, device)
        if scores.shape != (2,) or not np.all(np.isfinite(scores)):
            raise ValueError(f"SMOKE_MODEL_SCORE_INVALID:{density}")
        summaries[density] = {"window_ids": [str(row["window_id"]) for row in values], "valid_unit_counts": [int(row["valid_unit_count"]) for row in values], "scores": scores.tolist(), "reloaded": True, "fit_epochs": fit.get("epochs"), "parameter_count": int(sum(parameter.numel() for parameter in model.parameters()))}
        del model, reloaded
    result = {"status": "PASS", "conditions": summaries, "formal_records_untouched": True, "epochs": 1}
    atomic_json(summary_path, result)
    return result


def _train_models(root: Path, *, device: str = "cuda", resume: bool = True) -> dict[str, Any]:
    import torch
    train_rows = {density: _load_feature_rows(root, density, "train") for density in DENSITIES}
    if train_rows["R17"] and {int(x["label"]) for x in train_rows["R17"]} != {0, 1}:
        raise ValueError("TRAINING_CLASSES_INCOMPLETE")
    model_path = root / "models/fold_models.json"
    existing = _load_json(model_path).get("records", []) if resume and model_path.is_file() else []
    records: list[dict[str, Any]] = []
    done: set[tuple[str, int]] = set()
    for item in existing:
        if item.get("status") == "TRAIN_COMPLETE" and str(item.get("condition")) in CONDITIONS and int(item.get("seed", -1)) in MODEL_SEEDS:
            records.append(item); done.add((str(item["condition"]), int(item["seed"])))
    total = len(CONDITIONS) * len(MODEL_SEEDS)
    progress(root, "train", "RUNNING", len(done), total)
    for density, condition in zip(DENSITIES, CONDITIONS):
        rows, ss, sq = _make_observation_rows(root, density, train_rows[density])
        batch = _batch(root, rows, ss, sq)
        for seed in MODEL_SEEDS:
            if (condition, int(seed)) in done:
                continue
            started = time.perf_counter()
            model, fit = observation._train_one("STRUCTURE_SUPPORT", batch, int(seed), device, epochs=EPOCHS)
            record = {"condition": condition, "density": density, "base_condition": "STRUCTURE_SUPPORT", "seed": int(seed), "status": "TRAIN_COMPLETE", "training_window_count": len(rows), "training_real_count": sum(int(x["label"]) == 0 for x in rows), "training_fake_count": sum(int(x["label"]) == 1 for x in rows), "training_sources": sorted({str(x["source_id"]) for x in rows}), "parameter_count": int(sum(p.numel() for p in model.parameters())), "standardization": {"s": ss.as_dict(), "q": sq.as_dict()}, "feature_hash": _feature_hash(root, train_rows[density]), "fit": fit, "elapsed_s": time.perf_counter() - started, "device": str(torch.device(device)), "state_dict": {name: value.detach().cpu().numpy().tolist() for name, value in model.state_dict().items()}}
            records.append(record); done.add((condition, int(seed)))
            atomic_json(model_path, {"conditions": list(CONDITIONS), "seeds": list(MODEL_SEEDS), "epochs": EPOCHS, "records": records})
            progress(root, "train", "RUNNING", len(done), total, condition=condition, seed=seed)
            del model
    status = "COMPLETE" if len(done) == total else "PARTIAL"
    progress(root, "train", status, len(done), total)
    return {"status": status, "completed_models": len(done), "total_models": total, "records": records}


def _load_model(record: Mapping[str, Any], device: str) -> Any:
    import torch
    from research_tools.v7.geometry_information_pilot.model import ModalSetModel
    model = ModalSetModel(8)
    model.load_state_dict({name: torch.as_tensor(value, dtype=model.state_dict()[name].dtype) for name, value in record["state_dict"].items()})
    model.to(device).eval()
    return model


def _score(model: Any, batch: Mapping[str, Any], device: str) -> np.ndarray:
    import torch
    states = torch.as_tensor(batch["states"], dtype=torch.float32, device=device); intervals = torch.as_tensor(batch["intervals"], dtype=torch.float32, device=device); indices = torch.as_tensor(batch["window_index"], dtype=torch.long, device=device)
    with torch.no_grad():
        local = model(states, intervals); sums = torch.zeros(int(batch["window_count"]), dtype=local.dtype, device=device); counts = torch.zeros_like(sums); sums.index_add_(0, indices, local); counts.index_add_(0, indices, torch.ones_like(local)); return (sums / counts.clamp_min(1)).detach().cpu().numpy().astype(np.float64)


def _classification(labels: Sequence[int], scores: Sequence[float]) -> dict[str, Any]:
    y = np.asarray(labels, dtype=np.int64); z = np.asarray(scores, dtype=np.float64); pred = z >= 0
    tn = int(np.sum((y == 0) & ~pred)); fp = int(np.sum((y == 0) & pred)); fn = int(np.sum((y == 1) & ~pred)); tp = int(np.sum((y == 1) & pred)); precision = tp / (tp + fp) if tp + fp else None; recall = tp / (tp + fn) if tp + fn else None; f1 = 2 * precision * recall / (precision + recall) if precision is not None and recall is not None and precision + recall else None
    return {"precision": precision, "recall": recall, "f1": f1, "accuracy": float((tn + tp) / len(y)) if len(y) else None, "tn": tn, "fp": fp, "fn": fn, "tp": tp}


def _auroc(labels: Sequence[int], scores: Sequence[float]) -> float | None:
    y = np.asarray(labels, dtype=np.int64); z = np.asarray(scores, dtype=np.float64); p = z[y == 1]; n = z[y == 0]
    if not p.size or not n.size: return None
    return float(np.mean((p[:, None] > n[None, :]) + 0.5 * (p[:, None] == n[None, :])))


def _ap(labels: Sequence[int], scores: Sequence[float]) -> float | None:
    y = np.asarray(labels, dtype=np.int64); z = np.asarray(scores, dtype=np.float64); positives = int(np.sum(y == 1))
    if positives == 0: return None
    order = np.argsort(-z, kind="mergesort"); ordered = y[order]; tp = np.cumsum(ordered == 1); precision = tp / np.arange(1, len(y) + 1); return float(np.sum(precision[ordered == 1]) / positives)


def _source_metrics(rows: Sequence[Mapping[str, Any]], score_map: Mapping[str, float]) -> tuple[dict[str, Any], dict[str, float]]:
    labels = [int(row["label"]) for row in rows]; scores = [float(score_map[str(row["window_id"])]) for row in rows]
    by_source: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows: by_source[str(row["source_id"])].append(row)
    per = {src: _auroc([int(x["label"]) for x in vals], [float(score_map[str(x["window_id"])]) for x in vals]) for src, vals in by_source.items()}
    per = {src: float(value) for src, value in per.items() if value is not None}
    raw = np.asarray(list(per.values()), dtype=np.float64)
    ci = [None, None]
    if raw.size:
        draws = raw[np.random.default_rng(BOOTSTRAP_SEED).integers(0, raw.size, size=(BOOTSTRAP_REPLICATES, raw.size))].mean(axis=1); ci = [float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))]
    return ({"window_count": len(rows), "real_count": int(sum(x == 0 for x in labels)), "fake_count": int(sum(x == 1 for x in labels)), "source_count": len(by_source), "dual_source_count": len(per), "source_macro_auroc": float(raw.mean()) if raw.size else None, "source_macro_ci_low": ci[0], "source_macro_ci_high": ci[1], "pooled_auroc": _auroc(labels, scores), "pooled_ap": _ap(labels, scores), **_classification(labels, scores)}, per)


def _cross_density_metrics(cross_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Summarize the precomputed 2x2 readout without another model forward."""
    normalized = []
    for row in cross_rows:
        item = dict(row)
        item["seed"] = int(item["seed"])
        item["score"] = float(item["score"])
        item["label"] = 0 if "::real::" in str(item["window_id"]) else 1
        normalized.append(item)
    output: list[dict[str, Any]] = []
    pairs = sorted({(str(row["train_density"]), str(row["inference_density"])) for row in normalized})
    for train_density, inference_density in pairs:
        subset = [row for row in normalized if (str(row["train_density"]), str(row["inference_density"])) == (train_density, inference_density)]
        for seed in sorted({int(row["seed"]) for row in subset}):
            by_id = {str(row["window_id"]): row for row in subset if int(row["seed"]) == seed}
            ids = sorted(by_id)
            labels = [int(by_id[item]["label"]) for item in ids]
            scores = [float(by_id[item]["score"]) for item in ids]
            by_source: dict[str, list[str]] = defaultdict(list)
            for item in ids:
                by_source[item.split("::", 1)[0]].append(item)
            source_auroc = [_auroc([int(by_id[item]["label"]) for item in source_ids], [float(by_id[item]["score"]) for item in source_ids]) for source_ids in by_source.values()]
            source_auroc = [float(value) for value in source_auroc if value is not None]
            output.append({"aggregate": "SEED", "train_density": train_density, "inference_density": inference_density, "seed": seed, "window_count": len(ids), "real_count": labels.count(0), "fake_count": labels.count(1), "source_count": len(source_auroc), "source_macro_auroc": float(np.mean(source_auroc)) if source_auroc else None, "pooled_auroc": _auroc(labels, scores), "pooled_ap": _ap(labels, scores), **_classification(labels, scores)})
        by_id: dict[str, list[float]] = defaultdict(list)
        for row in subset:
            by_id[str(row["window_id"])].append(float(row["score"]))
        ids = sorted(by_id)
        labels = [0 if "::real::" in item else 1 for item in ids]
        scores = [float(np.mean(by_id[item])) for item in ids]
        by_source: dict[str, list[int]] = defaultdict(list)
        for index, item in enumerate(ids):
            by_source[item.split("::", 1)[0]].append(index)
        source_auroc = [_auroc([labels[index] for index in indices], [scores[index] for index in indices]) for indices in by_source.values()]
        source_auroc = [float(value) for value in source_auroc if value is not None]
        output.append({"aggregate": "MEAN_LOGIT", "train_density": train_density, "inference_density": inference_density, "seed": "MEAN_LOGIT", "window_count": len(ids), "real_count": labels.count(0), "fake_count": labels.count(1), "source_count": len(source_auroc), "source_macro_auroc": float(np.mean(source_auroc)) if source_auroc else None, "pooled_auroc": _auroc(labels, scores), "pooled_ap": _ap(labels, scores), **_classification(labels, scores)})
    return output


def evaluate(root: Path, *, device: str = "cuda") -> dict[str, Any]:
    import torch
    val_rows = {density: _load_feature_rows(root, density, "validation") for density in DENSITIES}
    common_ids = set(str(row["window_id"]) for row in val_rows["R17"]) & set(str(row["window_id"]) for row in val_rows["R33"])
    common = [row for row in val_rows["R17"] if str(row["window_id"]) in common_ids]
    common.sort(key=lambda x: str(x["window_id"]))
    records = [row for row in _load_json(root / "models/fold_models.json")["records"] if row.get("status") == "TRAIN_COMPLETE"]
    score_rows: dict[str, dict[str, Any]] = {str(row["window_id"]): {"window_id": row["window_id"], "source_id": row["source_id"], "role": row["role"], "label": int(row["label"]), "offset_s": row.get("offset_s"), "annotation_category": row.get("annotation_category"), "r17_valid_unit_count": next(int(x["valid_unit_count"]) for x in val_rows["R17"] if x["window_id"] == row["window_id"]), "r33_valid_unit_count": next(int(x["valid_unit_count"]) for x in val_rows["R33"] if x["window_id"] == row["window_id"]) } for row in common}
    metric_rows: list[dict[str, Any]] = []; per_source_rows: list[dict[str, Any]] = []; per_seed_rows: list[dict[str, Any]] = []; cross_rows: list[dict[str, Any]] = []
    means: dict[tuple[str, str], dict[str, float]] = {}
    for rec in records:
        density = str(rec["density"]); condition = str(rec["condition"]); ss = periodic.FeatureStandardizer(**{**{k: rec["standardization"]["s"][k] for k in ("condition",)}, "mean": np.asarray(rec["standardization"]["s"]["mean"], dtype=np.float64), "scale": np.asarray(rec["standardization"]["s"]["scale"], dtype=np.float64), "zero_variance_dimensions": tuple(int(x) for x in rec["standardization"]["s"].get("zero_variance_dimensions", []))}); sq = periodic.FeatureStandardizer(str(rec["standardization"]["q"]["condition"]), np.asarray(rec["standardization"]["q"]["mean"], dtype=np.float64), np.asarray(rec["standardization"]["q"]["scale"], dtype=np.float64), tuple(int(x) for x in rec["standardization"]["q"].get("zero_variance_dimensions", [])))
        # Own training metrics and validation metrics use the model's own density.
        train_rows = _load_feature_rows(root, density, "train"); train_values, _, _ = _make_observation_rows(root, density, train_rows); train_batch = _batch(root, train_values, ss, sq)
        val_density_rows = _load_feature_rows(root, density, "validation"); val_values, _, _ = _make_observation_rows(root, density, [row for row in val_density_rows if str(row["window_id"]) in common_ids]); val_batch = _batch(root, val_values, ss, sq)
        model = _load_model(rec, device); train_scores = _score(model, train_batch, device); val_scores = _score(model, val_batch, device)
        for split, rows, scores in (("train", train_values, train_scores), ("validation", val_values, val_scores)):
            mapping = {str(row["window_id"]): float(score) for row, score in zip(rows, scores)}
            metrics, per = _source_metrics(rows, mapping); metric_rows.append({"split": split, "condition": condition, "density": density, "seed": int(rec["seed"]), **metrics});
            for src, auc in per.items(): per_source_rows.append({"split": split, "condition": condition, "density": density, "seed": int(rec["seed"]), "source_id": src, "auroc": auc, "window_count": sum(str(x["source_id"]) == src for x in rows), "real_count": sum(str(x["source_id"]) == src and int(x["label"]) == 0 for x in rows), "fake_count": sum(str(x["source_id"]) == src and int(x["label"]) == 1 for x in rows)})
            if split == "validation":
                for row, score in zip(rows, scores): score_rows[str(row["window_id"])][f"{condition}_seed_{rec['seed']}"] = float(score)
        del model
        if torch.cuda.is_available(): torch.cuda.empty_cache()
    # All cross-density reads use each record's own standardizer. Recompute
    # only validation forward, never fit anything on validation.
    for rec in records:
        density = str(rec["density"]); condition = str(rec["condition"]); ss = periodic.FeatureStandardizer(str(rec["standardization"]["s"]["condition"]), np.asarray(rec["standardization"]["s"]["mean"], dtype=np.float64), np.asarray(rec["standardization"]["s"]["scale"], dtype=np.float64), tuple(int(x) for x in rec["standardization"]["s"].get("zero_variance_dimensions", []))); sq = periodic.FeatureStandardizer(str(rec["standardization"]["q"]["condition"]), np.asarray(rec["standardization"]["q"]["mean"], dtype=np.float64), np.asarray(rec["standardization"]["q"]["scale"], dtype=np.float64), tuple(int(x) for x in rec["standardization"]["q"].get("zero_variance_dimensions", [])))
        model = _load_model(rec, device)
        for target in DENSITIES:
            target_rows = [row for row in _load_feature_rows(root, target, "validation") if str(row["window_id"]) in common_ids]; values, _, _ = _make_observation_rows(root, target, target_rows); scores = _score(model, _batch(root, values, ss, sq), device); cross_rows.extend({"train_density": density, "inference_density": target, "condition": condition, "seed": int(rec["seed"]), "window_id": row["window_id"], "score": float(score)} for row, score in zip(values, scores))
        del model
    # seed-mean scores and metrics
    for condition, density in zip(CONDITIONS, DENSITIES):
        mapping = {}
        for row in common:
            vals = [score_rows[str(row["window_id"])].get(f"{condition}_seed_{seed}") for seed in MODEL_SEEDS]
            if not all(value is not None and math.isfinite(float(value)) for value in vals):
                continue
            mapping[str(row["window_id"])] = float(np.mean(np.asarray(vals, dtype=np.float64)))
            score_rows[str(row["window_id"])][f"{condition}_MEAN_LOGIT"] = mapping[str(row["window_id"])]
        rows = [row for row in common if str(row["window_id"]) in mapping]
        metrics, per = _source_metrics(rows, mapping); metric_rows.append({"split": "validation", "condition": condition, "density": density, "seed": "MEAN_LOGIT", **metrics}); means[(condition, "validation")] = mapping
    train_metric_mean = []
    for condition, density in zip(CONDITIONS, DENSITIES):
        train_rows = _load_feature_rows(root, density, "train"); # mean train scores are not needed for the primary paired comparison
        seed_maps = []
        for seed in MODEL_SEEDS:
            candidates = [r for r in metric_rows if r["split"] == "train" and r["condition"] == condition and int(r["seed"]) == seed]
            if candidates: train_metric_mean.extend(candidates)
    write_csv(root / "scores/validation_window_scores.csv", list(score_rows.values()))
    write_csv(root / "evaluation/metrics.csv", metric_rows)
    write_csv(root / "evaluation/per_source_metrics.csv", per_source_rows)
    write_csv(root / "evaluation/cross_density_readout.csv", cross_rows)
    cross_metrics = _cross_density_metrics(cross_rows)
    write_csv(root / "evaluation/cross_density_metrics.csv", cross_metrics)
    d17 = means.get(("D17_TRAIN", "validation"), {}); d33 = means.get(("D33_TRAIN", "validation"), {})
    sources = sorted({str(row["source_id"]) for row in common}); by_source_diff = {}
    for src in sources:
        a = [row for row in common if str(row["source_id"]) == src]; x = _auroc([int(row["label"]) for row in a], [d17[str(row["window_id"])] for row in a]); y = _auroc([int(row["label"]) for row in a], [d33[str(row["window_id"])] for row in a]);
        if x is not None and y is not None: by_source_diff[src] = float(y - x)
    raw = np.asarray(list(by_source_diff.values()), dtype=np.float64); ci = [None, None]
    if raw.size:
        rng = np.random.default_rng(BOOTSTRAP_SEED); draws = raw[rng.integers(0, raw.size, size=(BOOTSTRAP_REPLICATES, raw.size))].mean(axis=1); ci = [float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))]
    paired = [{"comparison": "D33_DTRAIN_D33_VAL-D17_DTRAIN_D17_VAL", "source_count": int(raw.size), "mean_source_macro_difference": float(raw.mean()) if raw.size else None, "ci_low": ci[0], "ci_high": ci[1], "positive_source_count": int(np.sum(raw > 0)), "negative_source_count": int(np.sum(raw < 0)), "tie_source_count": int(np.sum(raw == 0))}]
    paired += [{"comparison": "source", "source_id": src, "difference": value} for src, value in sorted(by_source_diff.items())]
    write_csv(root / "evaluation/paired_comparisons.csv", paired)
    summary = {"validation_common_window_count": len(common), "validation_common_source_count": len({str(row["source_id"]) for row in common}), "validation_real_count": sum(int(row["label"]) == 0 for row in common), "validation_fake_count": sum(int(row["label"]) == 1 for row in common), "metrics": metric_rows, "paired": paired, "model_count": len(records), "cross_density_rows": len(cross_rows), "cross_metrics": cross_metrics, "device": str(torch.device(device))}
    atomic_json(root / "evaluation/summary.json", summary)
    progress(root, "evaluate", "COMPLETE", len(metric_rows), len(metric_rows))
    return summary


def write_report(root: Path, protocol: Mapping[str, Any], frontend_summary: Mapping[str, Any], feature_summary: Mapping[str, Any], train_summary: Mapping[str, Any], evaluation_summary: Mapping[str, Any], smoke: Mapping[str, Any]) -> Path:
    metrics = evaluation_summary.get("metrics", [])
    lines = ["# V7 小规模 real/fake 对称的两密度匹配训练 pilot", "", "本实验固定当前 R 表示、STRUCTURE_SUPPORT 模型和匹配训练/验证窗口，只比较 17×17（289）与嵌套 33×33（1089）查询。验证集是反复使用的开发集合，不是 sealed-test。", "", "## 结果先行"]
    lines += [f"- 前端父片段：{frontend_summary.get('completed_parents')}/{frontend_summary.get('total_parents')}；特征共同训练窗口：{feature_summary.get('common_train_windows')}；共同验证窗口：{feature_summary.get('common_validation_windows')}。", f"- 正式模型：{train_summary.get('completed_models')}/{train_summary.get('total_models')}；smoke：{smoke.get('status')}。", f"- 训练样本为两密度合法 S/Q 和主标签的交集；R33-only 或支撑失败不填零、不进入匹配训练。", "", "| split | condition | seed | windows | real | fake | source-macro AUROC | 95% CI | pooled AUROC | AP | P | R | F1 | ACC |", "|---|---|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|"]
    for row in metrics:
        lines.append(f"| {row.get('split')} | {row.get('condition')} | {row.get('seed')} | {row.get('window_count')} | {row.get('real_count')} | {row.get('fake_count')} | {row.get('source_macro_auroc')} | [{row.get('source_macro_ci_low')}, {row.get('source_macro_ci_high')}] | {row.get('pooled_auroc')} | {row.get('pooled_ap')} | {row.get('precision')} | {row.get('recall')} | {row.get('f1')} | {row.get('accuracy')} |")
    lines += ["", "## 主配对", "", "| comparison | source count | mean difference | 95% CI | positive/tie/negative |", "|---|---:|---:|---|---|"]
    for row in evaluation_summary.get("paired", [])[:1]:
        lines.append(f"| {row.get('comparison')} | {row.get('source_count')} | {row.get('mean_source_macro_difference')} | [{row.get('ci_low')}, {row.get('ci_high')}] | {row.get('positive_source_count')}/{row.get('tie_source_count')}/{row.get('negative_source_count')} |")
    cross_metrics = [row for row in evaluation_summary.get("cross_metrics", []) if row.get("aggregate") == "MEAN_LOGIT"]
    if cross_metrics:
        lines += ["", "## 2×2 密度交叉读出（辅助）", "", "四格使用同一主验证窗口和各训练密度自己的标准化；这是分布适应诊断，不是新增主要假设。", "", "| 训练密度 | 推理密度 | source-macro AUROC | pooled AUROC | AP | F1 | ACC |", "|---|---|---:|---:|---:|---:|---:|"]
        for row in cross_metrics:
            lines.append(f"| {row.get('train_density')} | {row.get('inference_density')} | {row.get('source_macro_auroc')} | {row.get('pooled_auroc')} | {row.get('pooled_ap')} | {row.get('f1')} | {row.get('accuracy')} |")
    lines += ["", "## 覆盖与解释", f"- 计划：288 个窗口（32 个训练 source + {len(protocol['selection']['validation_sources'])} 个既有 validation source，real/fake 各自三个 offset）；两密度各有独立前端记录。", f"- R17 合法窗口：{feature_summary.get('valid_windows', {}).get('R17')}；R33 合法窗口：{feature_summary.get('valid_windows', {}).get('R33')}；共同训练/验证集合只由技术支撑和预先冻结标签决定。", "- 每个父片段仅做一次深度/位姿，R17 使用合法旧 UV/visibility，R33 使用嵌套新查询；两密度使用相同父片段几何结果。R33 联合跟踪可能改变原 289 点 UV，不能把结果解释为仅增加 800 个独立轨迹。", "- 选择 32 个训练 source 依赖旧 R 窗口可用性，再按稳定哈希排序；这是开发性、带可用性筛选的 source 集合。", "- pooled 指标是辅助；窗口重叠且 source 数少，bootstrap 以 source 为单位。无空间真值，不报告定位能力；没有把 component/pair 数或缺失率作为模型输入。", "", "## 产物", "- `protocol.json`、`manifests/sample_manifest.json`、`manifests/common_train_windows.csv`、`manifests/common_validation_windows.csv`。", "- `scores/validation_window_scores.csv`、`evaluation/metrics.csv`、`evaluation/per_source_metrics.csv`、`evaluation/paired_comparisons.csv`、`evaluation/cross_density_readout.csv`、`evaluation/cross_density_metrics.csv`、`evaluation/coverage_summary.csv`。", "- 大型 ParticleSequence、特征、模型和视频保留在数据目录，不提交 Git。", "", "## 结论边界", "若 D33/D33 的 source-macro 点估计上升但区间跨 0，只能说本开发 pilot 有改善倾向；若覆盖增加而检测未一致改善，应暂停密度路线，不自动测试更高密度。结果不支持稳定跨生成器泛化、空间定位或三维物理规律。"]
    path = root / "report.md"; path.write_text("\n".join(lines) + "\n", encoding="utf-8"); return path


def run(root: Path = OUTPUT_ROOT, *, stage: str = "all", device: str = "cuda", resume: bool = True, smoke_only: bool = False, limit_parents: int | None = None) -> dict[str, Any]:
    global STOP_REQUESTED
    root.mkdir(parents=True, exist_ok=True); root.joinpath("state").mkdir(parents=True, exist_ok=True)
    atomic_json(root / "state/launch.json", {"pid": os.getpid(), "git_head": git_head(), "device": device, "started_unix": time.time(), "stage": stage})
    def stop(_signal: int, _frame: Any) -> None:
        global STOP_REQUESTED; STOP_REQUESTED = True
    old_term = signal.getsignal(signal.SIGTERM); old_int = signal.getsignal(signal.SIGINT); signal.signal(signal.SIGTERM, stop); signal.signal(signal.SIGINT, stop)
    started = time.perf_counter()
    try:
        protocol = load_or_freeze_plan(root)
        train_rows, val_rows, parents, support, planned_rows = _load_sample(root)
        smoke = frontend_smoke(root, train_rows, parents, support, device=device)
        if smoke.get("status") != "PASS": raise RuntimeError("SMOKE_FAILED")
        smoke = {**smoke, "model": smoke_model_pipeline(root, smoke, train_rows, device=device)}
        if smoke_only:
            return {"status": "SMOKE_COMPLETE", "smoke": smoke}
        if stage in {"all", "frontend"}:
            frontend_summary = run_frontend(root, train_rows, val_rows, parents, support, planned_rows=planned_rows, device=device, resume=resume, limit_parents=limit_parents)
        else:
            frontend_summary = _load_json(root / "state/frontend_summary.json")
        if stage == "frontend":
            return frontend_summary
        feature_summary = build_features(root, train_rows, val_rows, planned_rows=planned_rows, resume=resume) if stage in {"all", "features"} else _load_json(root / "support/summary.json")
        if stage == "features":
            return feature_summary
        train_summary = _train_models(root, device=device, resume=resume) if stage in {"all", "train"} else {"completed_models": len(_load_json(root / "models/fold_models.json").get("records", [])), "total_models": 6}
        if train_summary.get("completed_models") != 6:
            atomic_json(root / "final_status.json", {"status": "PARTIAL", "stage": "train", "model_count": train_summary.get("completed_models"), "git_head": git_head()}); return train_summary
        evaluation_summary = evaluate(root, device=device)
        report = write_report(root, protocol, frontend_summary, feature_summary, train_summary, evaluation_summary, smoke)
        final = {"status": "COMPLETE", "git_head": git_head(), "report": str(report), "frontend": frontend_summary, "features": feature_summary, "models": train_summary, "evaluation": {"validation_common_window_count": evaluation_summary.get("validation_common_window_count"), "paired": evaluation_summary.get("paired")}, "smoke": smoke, "elapsed_s": time.perf_counter() - started}
        atomic_json(root / "final_status.json", final); progress(root, "report", "COMPLETE", 1, 1, elapsed_s=final["elapsed_s"]); return final
    except BaseException as exc:
        failure = {"status": "FAILED", "error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc(), "git_head": git_head(), "elapsed_s": time.perf_counter() - started}
        atomic_json(root / "state/failure.json", failure); atomic_json(root / "final_status.json", failure); progress(root, "failed", "FAILED", 0, 1, error=failure["error"]); raise
    finally:
        signal.signal(signal.SIGTERM, old_term); signal.signal(signal.SIGINT, old_int)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("all", "frontend", "features", "train"), default="all")
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke-only", action="store_true")
    parser.add_argument("--limit-parents", type=int, default=None)
    args = parser.parse_args(argv)
    result = run(args.output_root, stage=args.stage, device=args.device, resume=args.resume, smoke_only=args.smoke_only, limit_parents=args.limit_parents)
    print(json.dumps(_jsonable({"status": result.get("status"), "report": result.get("report"), "elapsed_s": result.get("elapsed_s")}), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
