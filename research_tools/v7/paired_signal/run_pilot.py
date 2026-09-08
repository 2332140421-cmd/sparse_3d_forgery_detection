"""Run the frozen explicit frontend on the paired ActivityForensics pilot.

The script performs the first four-pair feasibility gate before continuing to
the remaining selected pairs.  It never fits a model and it rejects preview
paths before decoding.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from research_tools.v7.feasibility.run_vript_frontend_probe import _decode
from research_tools.v7.paired_signal.protocol import (
    build_population,
    extract_features,
    file_sha256,
)
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


DATA_ROOT = Path("/root/autodl-tmp/data/sparse_3d_forgery_detection")
DATASET_ROOT = DATA_ROOT / "datasets/v7_core_candidates/activityforensics_charades_v1"
OUTPUT_ROOT = DATA_ROOT / "derived/v7_activityforensics_paired_second_order_pilot_v1"
TAPNET_SHA = "c2cbab81cc06092b5f05bfe2da7bfec54e2079c9"
DEPTH_PRO_SHA = "9efe5c1def37a26c5367a71df664b18e1306c708"


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False, ensure_ascii=False) + "\n", encoding="utf-8")


def file_identity(path: Path) -> dict[str, Any]:
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": file_sha256(path)}


def _window_item(window: dict[str, Any], role: str) -> dict[str, Any]:
    side = window[role]
    path = str(side["video_path"])
    if "preview" in path.lower():
        raise ValueError(f"preview path rejected before decode: {path}")
    return {
        "window_id": f"{window['window_id']}::{role}",
        "pair_id": window["pair_id"],
        "source_id": window["source_id"],
        "video_path": path,
        "frame_indices": side["frame_indices"],
        "timestamps_s": side["timestamps_s"],
        "role": role,
        "kind": window["kind"],
        "label": window["label"],
        "interval_start_s": window["interval_start_s"],
        "interval_end_s": window["interval_end_s"],
        "center_s": window["center_s"],
        "anchor_fraction": window["anchor_fraction"],
    }


def run_frontend_window(item: dict[str, Any], tracker: Any, depth_runner: Any, output: Path) -> dict[str, Any]:
    started = time.perf_counter()
    source_path = Path(item["video_path"])
    if "preview" in str(source_path).lower():
        raise ValueError("preview media is forbidden")
    decoded = _decode(source_path, f"{item['pair_id']}-{item['role']}", list(item["frame_indices"]))
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
            "dataset": "ActivityForensics+Charades",
            "source_id": item["source_id"],
            "pair_id": item["pair_id"],
            "official_split": item.get("official_split", "train"),
            "lineage_status": "EXACT",
            "role": item["role"],
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
            "original_video_sha256": file_sha256(source_path),
        },
    )
    safe_id = item["window_id"].replace("/", "_").replace(":", "_")
    prefix = output / "particles" / safe_id
    save_particle_sequence(sequence, prefix)
    features = extract_features(
        sequence,
        source_id=str(item["source_id"]),
        pair_id=str(item["pair_id"]),
        window_id=str(item["window_id"]),
        role=str(item["role"]),
        kind=str(item["kind"]),
    )
    torch = getattr(tracker, "torch", None)
    peak_memory = None
    if torch is not None and torch.cuda.is_available():
        peak_memory = int(torch.cuda.max_memory_allocated())
    quality = {
        "geometry_coverage": float(np.mean(geometry_valid)),
        "tracking_persistence": float(np.mean(np.all(visibility, axis=0))),
        "pose_success": float(np.mean(pair_valid)) if pair_valid.size else 1.0,
        "pose_valid_fraction": float(np.mean(pose_valid)) if pose_valid.size else 1.0,
        "component_count": features["component_count"],
        "component_success": features["component_success"],
        "s_valid_fraction": features["s_valid_fraction"],
        "valid_delta_s": features["valid_delta_s"],
        "valid_delta2_s": features["valid_delta2_s"],
    }
    result = {
        "window": item,
        "source_sha256": file_sha256(source_path),
        "frame_indices": decoded.frame_indices.tolist(),
        "timestamps_s": decoded.timestamps_s.tolist(),
        "focal_px": focals.tolist(),
        "fixed_focal_px": fixed_focal_px,
        "frame_depth_valid": frame_depth_valid.tolist(),
        "pose_pair_valid": pair_valid.tolist(),
        "pose_valid": pose_valid.tolist(),
        "pose_information_trace": [float(np.trace(x)) if np.all(np.isfinite(x)) else None for x in information],
        "tracking": tracking_metrics(visibility),
        "quality": quality,
        "features": features,
        "particle_prefix": str(prefix),
        "peak_gpu_memory_bytes": peak_memory,
        "elapsed_s": time.perf_counter() - started,
        "status": "COMPLETE",
        "causal_training_eligible": True,
        "preview_used": False,
    }
    del decoded, sequence, depths, uv, xyz, geometry_valid
    gc.collect()
    if torch is not None and torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


def gate_passed(results: list[dict[str, Any]], first_pairs: list[dict[str, Any]]) -> bool:
    valid_pairs = 0
    for pair in first_pairs:
        rows = [row for row in results if row.get("window", {}).get("pair_id") == pair["pair_id"] and row.get("window", {}).get("kind") == "MANIP"]
        by_role = {role: [row for row in rows if row.get("window", {}).get("role") == role] for role in ("real", "fake")}
        if all(any(row.get("quality", {}).get("component_success") and row.get("quality", {}).get("valid_delta2_s", 0) > 0 for row in by_role[role]) for role in ("real", "fake")):
            valid_pairs += 1
    return valid_pairs >= 3


def run_pilot(output: Path, *, tapnet_source: Path, tapnet_checkpoint: Path, depth_source: Path, depth_checkpoint: Path, limit: int = 16) -> dict[str, Any]:
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"pilot output already exists: {output}")
    output.mkdir(parents=True, exist_ok=True)
    review_rows = json.loads((DATASET_ROOT / "review/review_manifest.json").read_text(encoding="utf-8"))
    mapping_rows = json.loads((DATASET_ROOT / "paired/manifests/activityforensics_source_mapping.json").read_text(encoding="utf-8"))
    population = build_population(review_rows, mapping_rows, limit=limit)
    write_json(output / "manifests/population.json", population)
    write_json(output / "manifests/selected_pairs.json", population["selected_pairs"])
    windows = [_window_item(window, role) for pair in population["selected_pairs"] for window in pair["windows"] for role in ("real", "fake")]
    write_json(output / "manifests/window_manifest.json", windows)
    meta = {
        "population": "ActivityForensics-Charades EXACT paired pilot",
        "limit": limit,
        "selected_pairs": len(population["selected_pairs"]),
        "fake_count_in_fitting": 0,
        "normality_model_fitting": "NONE",
        "frontend_baseline_unchanged": True,
        "tracker_source_commit": TAPNET_SHA,
        "depth_source_commit": DEPTH_PRO_SHA,
        "tracker_checkpoint": file_identity(tapnet_checkpoint),
        "depth_checkpoint": file_identity(depth_checkpoint),
        "frontend_environment": "/root/autodl-tmp/envs/v7-explicit-geometry",
        "gpu": None,
    }
    try:
        import torch

        meta["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except Exception:
        pass
    write_json(output / "frontend/run_meta.json", meta)
    if len(population["selected_pairs"]) < 12:
        result = {"status": "PAIRED_PILOT_POPULATION_INSUFFICIENT", "selected_pairs": len(population["selected_pairs"]), "exclusions": population["exclusions"], "normality_training": False}
        write_json(output / "run_summary.json", result)
        return result

    tracker = OnlineBootsTapir(tapnet_source, tapnet_checkpoint)
    depth_runner = DepthProRunner(depth_source, depth_checkpoint)
    results: list[dict[str, Any]] = []
    selected = population["selected_pairs"]
    for pair_index, pair in enumerate(selected):
        for window in pair["windows"]:
            for role in ("real", "fake"):
                item = _window_item(window, role)
                print(f"frontend {pair_index + 1}/{len(selected)} {item['window_id']}", flush=True)
                try:
                    result = run_frontend_window(item, tracker, depth_runner, output)
                except Exception as exc:
                    result = {"window": item, "status": "FRONTEND_FAILURE", "error": f"{type(exc).__name__}: {exc}", "preview_used": False}
                results.append(result)
                write_json(output / "frontend/window_results_progress.json", results)
        if pair_index == 3 and not gate_passed(results, selected[:4]):
            status = "FRONTEND_FEASIBILITY_BLOCKED_ON_NEW_CORE_DATA"
            write_json(output / "frontend/window_results.json", results)
            write_json(output / "run_summary.json", {"status": status, "selected_pairs": len(selected), "processed_pairs": pair_index + 1, "normality_training": False, "frontend_gate_passed": False})
            del tracker, depth_runner
            gc.collect()
            return {"status": status, "selected_pairs": len(selected), "processed_pairs": pair_index + 1, "frontend_gate_passed": False}
    del tracker, depth_runner
    gc.collect()
    write_json(output / "frontend/window_results.json", results)
    summary = {"status": "FRONTEND_COMPLETE_FOR_ANALYSIS", "selected_pairs": len(selected), "window_results": len(results), "frontend_gate_passed": True, "normality_training": False, "fake_count_in_fitting": 0}
    write_json(output / "run_summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--tapnet-source", type=Path, required=True)
    parser.add_argument("--tapnet-checkpoint", type=Path, required=True)
    parser.add_argument("--depth-source", type=Path, required=True)
    parser.add_argument("--depth-checkpoint", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=16)
    args = parser.parse_args()
    print(json.dumps(run_pilot(args.output, tapnet_source=args.tapnet_source, tapnet_checkpoint=args.tapnet_checkpoint, depth_source=args.depth_source, depth_checkpoint=args.depth_checkpoint, limit=args.limit), indent=2, sort_keys=True, ensure_ascii=False))


if __name__ == "__main__":
    main()
