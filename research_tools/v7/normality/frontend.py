"""Adapter that runs the already frozen V7 frontend on pilot windows."""

from __future__ import annotations

import gc
import time
from pathlib import Path

import numpy as np

from research_tools.v7.feasibility.run_vript_frontend_probe import _decode
from scripts.run_v7_explicit_geometry_frontend import (
    DepthProRunner,
    OnlineBootsTapir,
    accumulate_world_from_camera,
    causal_first_frame_intrinsics,
    persistent_track_ids,
    rgbd_odometry,
    sample_depth_at_uv,
    tracking_metrics,
    world_xyz,
)
from sparse3d_forgery.particle_sequence import (
    CoordinateSystem,
    Handedness,
    LengthUnit,
    build_particle_sequence,
    save_particle_sequence,
)
from .protocol import COMPONENT_CONFIG, extract_window_features


DATA_REVISION = "acc278efb0ee249d646eef8a6b023595ca7efb93"
TAPNET_SHA = "c2cbab81cc06092b5f05bfe2da7bfec54e2079c9"
DEPTH_PRO_SHA = "9efe5c1def37a26c5367a71df664b18e1306c708"


def run_frontend_window(item: dict[str, object], tracker, depth_runner, output: Path) -> dict[str, object]:
    started = time.perf_counter()
    decoded = _decode(Path(str(item["video_path"])), str(item["source_id"]), list(item["frame_indices"]))
    uv, visibility = tracker.track(decoded)
    depths, focals, frame_depth_valid = depth_runner.infer(decoded)
    intrinsics, fixed_focal_px = causal_first_frame_intrinsics(depths, focals)
    adjacent, pair_valid, information = rgbd_odometry(decoded, depths, intrinsics)
    world_from_camera, pose_valid = accumulate_world_from_camera(adjacent, pair_valid)
    sampled_depth, sampled_valid = sample_depth_at_uv(depths, uv)
    observation_valid = visibility & sampled_valid & frame_depth_valid[:, None]
    xyz, geometry_valid = world_xyz(uv, sampled_depth, intrinsics, world_from_camera, observation_valid, pose_valid)
    sequence = build_particle_sequence(
        decoded,
        track_ids=persistent_track_ids(uv.shape[1]),
        xyz=xyz,
        uv=uv,
        visibility=visibility,
        geometry_validity=geometry_valid,
        coordinate_system=CoordinateSystem(
            frame_name="first_camera_world",
            handedness=Handedness.RIGHT,
            axis_directions=("right", "down", "forward"),
            length_unit=LengthUnit.METER,
            camera_motion_compensated=True,
            normalization={},
        ),
        lineage={
            "dataset": "Vript",
            "revision": DATA_REVISION,
            "official_split": str(item["role"]),
            "population": "V7_REAL_ONLY_NORMALITY_PILOT",
        },
        provenance={
            "tracker": "online_bootstapir",
            "tracker_source_sha": TAPNET_SHA,
            "depth": "apple_depth_pro",
            "depth_source_sha": DEPTH_PRO_SHA,
            "depth_semantics": "optical_axis_z_depth_interpretation_of_official_depth_output",
            "principal_point": "EXPERIMENTAL_CENTER_PRINCIPAL_POINT_ASSUMPTION",
            "distortion": "EXPERIMENTAL_ZERO_DISTORTION_ASSUMPTION",
            "intrinsics_policy": "CAUSAL_FIRST_FRAME_DEPTH_PRO_FOCAL_ASSUMPTION",
            "fixed_focal_px": fixed_focal_px,
            "pose": "open3d_0.19_adjacent_rgbd_odometry",
            "pose_convention": "target_camera_from_source_camera; inverted during world accumulation",
            "causal_execution": True,
            "frontend_baseline_unchanged": True,
        },
    )
    window_id = str(item["window_id"])
    safe_id = window_id.replace("/", "_").replace(":", "_")
    prefix = output / "particles" / safe_id
    save_particle_sequence(sequence, prefix)
    feature_rows = extract_window_features(sequence, source_id=str(item["source_id"]), window_id=window_id)
    peak_memory = None
    if getattr(tracker, "torch", None) is not None and tracker.torch.cuda.is_available():
        peak_memory = int(tracker.torch.cuda.max_memory_allocated())
    result = {
        "window": item,
        "frame_indices": decoded.frame_indices.tolist(),
        "timestamps_s": decoded.timestamps_s.tolist(),
        "focal_px": focals.tolist(),
        "fixed_focal_px": fixed_focal_px,
        "frame_depth_valid": frame_depth_valid.tolist(),
        "pose_pair_valid": pair_valid.tolist(),
        "pose_valid": pose_valid.tolist(),
        "pose_information_trace": [float(np.trace(x)) if np.all(np.isfinite(x)) else None for x in information],
        "tracking": tracking_metrics(visibility),
        "coverage": {
            "requested": int(geometry_valid.size),
            "tracked_fraction": float(np.mean(visibility)),
            "depth_valid_fraction": float(np.mean(observation_valid)),
            "pose_valid_fraction": float(np.mean(pose_valid)),
            "final_geometry_valid_fraction": float(np.mean(geometry_valid)),
        },
        "component_count": feature_rows["component_count"],
        "valid_s_states": feature_rows["valid_s_states"],
        "valid_delta_s": feature_rows["valid_delta_s"],
        "valid_delta2_s": feature_rows["valid_delta2_s"],
        "component_rows": feature_rows["component_rows"],
        "observations": feature_rows["observations"],
        "particle_prefix": str(prefix),
        "peak_gpu_memory_bytes": peak_memory,
        "elapsed_s": time.perf_counter() - started,
        "status": "COMPLETE",
        "causal_training_eligible": True,
    }
    del decoded, sequence, depths, uv, xyz, geometry_valid
    gc.collect()
    if getattr(tracker, "torch", None) is not None and tracker.torch.cuda.is_available():
        tracker.torch.cuda.empty_cache()
    return result
