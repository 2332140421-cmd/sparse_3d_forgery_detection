#!/usr/bin/env python3
"""Run the fixed real-only V7 explicit-geometry falsification probe."""

import argparse
from dataclasses import replace
import gc
import hashlib
import json
from pathlib import Path
import platform
import sys
import time

import numpy as np

from sparse3d_forgery.experiments.v7_explicit_geometry_frontend import (
    RasterScale,
    accumulate_world_from_camera,
    centered_intrinsics,
    causal_first_frame_intrinsics,
    persistent_track_ids,
    sample_depth_at_uv,
    summarize_distribution,
    world_xyz,
)
from sparse3d_forgery.particle_sequence import (
    CoordinateSystem,
    Handedness,
    LengthUnit,
    build_particle_sequence,
    save_particle_sequence,
)
from sparse3d_forgery.video_input import VideoSource, decode_video


DATA_REVISION = "92e76e78e8c90a1ff7ec9354bee44eb024265e79"
DATA_ROOT = Path("/root/autodl-tmp/data/sparse_3d_forgery_detection")
EXTRACTED_ROOT = DATA_ROOT / "extracted/deeptrace_reward" / DATA_REVISION
PILOT_MANIFEST = DATA_ROOT / "derived/temporal_learnability_probe_v1/pilot_manifest.json"
DEFAULT_OUTPUT = DATA_ROOT / "derived/v7_explicit_geometry_probe_v1"
TAPNET_SHA = "c2cbab81cc06092b5f05bfe2da7bfec54e2079c9"
DEPTH_PRO_SHA = "9efe5c1def37a26c5367a71df664b18e1306c708"


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def file_identity(path: Path) -> dict[str, object]:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def selected_real_windows(limit: int, frames: int) -> list[dict]:
    with PILOT_MANIFEST.open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    items = [item for item in manifest["items"] if item["role"] == "real_val"]
    items.sort(key=lambda item: item["source_video_id"])
    if len(items) < limit:
        raise RuntimeError(f"fixed real-validation population has only {len(items)} items")
    selected = []
    for source in items[:limit]:
        item = dict(source)
        item["frame_indices"] = source["frame_indices"][:frames]
        selected.append(item)
    return selected


def decode(item: dict):
    return decode_video(
        VideoSource(
            sample_id=f"v7-explicit-{item['source_video_id']}",
            source_video_id=item["source_video_id"],
            source_locator=EXTRACTED_ROOT / item["video_path"],
        ),
        item["frame_indices"],
    )


class OnlineBootsTapir:
    def __init__(self, source: Path, checkpoint: Path, *, process_size: int = 256, grid_size: int = 8):
        sys.path.insert(0, str(source))
        import torch
        import torch.nn.functional as functional
        import tree
        from tapnet.torch import tapir_model

        self.torch = torch
        self.functional = functional
        self.tree = tree
        self.process_size = process_size
        self.grid_size = grid_size
        self.device = torch.device("cuda")
        self.model = tapir_model.TAPIR(pyramid_level=1, use_casual_conv=True)
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        self.model.load_state_dict(state)
        self.model.to(self.device).eval()

    def track(self, decoded) -> tuple[np.ndarray, np.ndarray]:
        torch = self.torch
        source_h, source_w = decoded.frames[0].rgb.shape[:2]
        if any(frame.rgb.shape[:2] != (source_h, source_w) for frame in decoded.frames):
            raise RuntimeError("variable raster size inside a window is unsupported")
        mapping = RasterScale((source_h, source_w), (self.process_size, self.process_size))
        grid = np.linspace(0, self.process_size - 1, self.grid_size + 2, dtype=np.float32)[1:-1]
        uu, vv = np.meshgrid(grid, grid)
        query = np.stack((np.zeros(uu.size, np.float32), vv.ravel(), uu.ravel()), axis=1)
        frames = np.stack([frame.rgb for frame in decoded.frames])
        tensor = torch.from_numpy(frames).to(self.device)
        tensor = tensor.permute(0, 3, 1, 2).float()
        tensor = self.functional.interpolate(
            tensor, size=(self.process_size, self.process_size), mode="bilinear", align_corners=False
        ).permute(0, 2, 3, 1)
        tensor = tensor / 255 * 2 - 1
        query_tensor = torch.from_numpy(query).to(self.device)
        with torch.inference_mode():
            first = tensor[None, 0:1]
            feature_grids = self.model.get_feature_grids(first, is_training=False)
            query_features = self.model.get_query_features(
                first,
                is_training=False,
                query_points=query_tensor[None],
                feature_grids=feature_grids,
            )
            causal_state = self.model.construct_initial_causal_state(
                query.shape[0], len(query_features.resolutions) - 1
            )
            causal_state = self.tree.map_structure(lambda value: value.to(self.device), causal_state)
            tracks, visibility = [], []
            for frame in tensor:
                single = frame[None, None]
                grids = self.model.get_feature_grids(single, is_training=False)
                result = self.model.estimate_trajectories(
                    single.shape[-3:-1],
                    is_training=False,
                    feature_grids=grids,
                    query_features=query_features,
                    query_points_in_video=None,
                    query_chunk_size=64,
                    causal_context=causal_state,
                    get_causal_context=True,
                )
                causal_state = result["causal_context"]
                point = result["tracks"][-1][0, :, 0].detach().cpu().numpy()
                occlusion = result["occlusion"][-1][0, :, 0]
                expected = result["expected_dist"][-1][0, :, 0]
                visible = ((1 - torch.sigmoid(occlusion)) * (1 - torch.sigmoid(expected)) > 0.5)
                tracks.append(point)
                visibility.append(visible.detach().cpu().numpy())
        uv_process = np.asarray(tracks, dtype=np.float32)
        uv_source = mapping.process_to_source(uv_process)
        visibility = np.asarray(visibility, dtype=np.bool_)
        inside = (
            np.all(np.isfinite(uv_source), axis=-1)
            & (uv_source[..., 0] >= 0)
            & (uv_source[..., 0] < source_w)
            & (uv_source[..., 1] >= 0)
            & (uv_source[..., 1] < source_h)
        )
        visibility &= inside
        uv_source[~visibility] = np.nan
        return uv_source, visibility


class DepthProRunner:
    def __init__(self, source: Path, checkpoint: Path):
        sys.path.insert(0, str(source / "src"))
        import torch
        import depth_pro
        from depth_pro.depth_pro import DEFAULT_MONODEPTH_CONFIG_DICT

        config = replace(DEFAULT_MONODEPTH_CONFIG_DICT, checkpoint_uri=str(checkpoint))
        self.torch = torch
        self.model, self.transform = depth_pro.create_model_and_transforms(
            config=config, device=torch.device("cuda"), precision=torch.float16
        )
        self.model.eval()

    def infer(self, decoded) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        depths, focals, valid = [], [], []
        with self.torch.inference_mode():
            for frame in decoded.frames:
                transformed = self.transform(frame.rgb)
                prediction = self.model.infer(transformed)
                depth = prediction["depth"].float().cpu().numpy().astype(np.float32)
                focal = float(prediction["focallength_px"].float().cpu())
                depths.append(depth)
                focals.append(focal)
                valid.append(bool(np.all(np.isfinite(depth)) and np.isfinite(focal) and focal > 0))
        return np.stack(depths), np.asarray(focals, np.float64), np.asarray(valid, np.bool_)


def resize_for_pose(rgb: np.ndarray, depth: np.ndarray, k: np.ndarray, max_side: int = 320):
    import cv2

    height, width = depth.shape
    scale = min(1.0, max_side / max(height, width))
    new_w, new_h = max(1, round(width * scale)), max(1, round(height * scale))
    if (new_h, new_w) == (height, width):
        return rgb, depth, k
    color = cv2.resize(rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)
    scaled_depth = cv2.resize(depth, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    scaled_k = k.copy()
    scaled_k[0, 0] *= new_w / width
    scaled_k[1, 1] *= new_h / height
    scaled_k[0, 2] = (k[0, 2] + 0.5) * new_w / width - 0.5
    scaled_k[1, 2] = (k[1, 2] + 0.5) * new_h / height - 0.5
    return color, scaled_depth, scaled_k


def rgbd_odometry(decoded, depths: np.ndarray, intrinsics: np.ndarray):
    import open3d as o3d

    transformations, successes, information = [], [], []
    for t in range(len(decoded.frames) - 1):
        source_rgb, source_depth, source_k = resize_for_pose(
            decoded.frames[t].rgb, depths[t], intrinsics[t]
        )
        target_rgb, target_depth, target_k = resize_for_pose(
            decoded.frames[t + 1].rgb, depths[t + 1], intrinsics[t + 1]
        )
        if source_rgb.shape != target_rgb.shape or not np.allclose(source_k, target_k, rtol=0.03):
            transformations.append(np.full((4, 4), np.nan))
            successes.append(False)
            information.append(np.full((6, 6), np.nan))
            continue
        height, width = source_depth.shape
        intrinsic = o3d.camera.PinholeCameraIntrinsic(
            width, height, source_k[0, 0], source_k[1, 1], source_k[0, 2], source_k[1, 2]
        )
        def make_rgbd(color, depth):
            return o3d.geometry.RGBDImage.create_from_color_and_depth(
                o3d.geometry.Image(np.ascontiguousarray(color)),
                o3d.geometry.Image(np.ascontiguousarray(depth.astype(np.float32))),
                depth_scale=1.0,
                depth_trunc=100.0,
                convert_rgb_to_intensity=True,
            )
        success, transform, info = o3d.pipelines.odometry.compute_rgbd_odometry(
            make_rgbd(source_rgb, source_depth),
            make_rgbd(target_rgb, target_depth),
            intrinsic,
            np.eye(4),
            o3d.pipelines.odometry.RGBDOdometryJacobianFromHybridTerm(),
            o3d.pipelines.odometry.OdometryOption(),
        )
        transformations.append(transform if success else np.full((4, 4), np.nan))
        successes.append(bool(success))
        information.append(info if success else np.full((6, 6), np.nan))
    return np.asarray(transformations), np.asarray(successes, np.bool_), np.asarray(information)


def tracking_metrics(visibility: np.ndarray) -> dict:
    lifetimes = np.sum(visibility, axis=0)
    return {
        "requested_tracks": int(visibility.shape[1]),
        "initialized_tracks": int(visibility.shape[1]),
        "visible_tracks_per_frame": np.sum(visibility, axis=1).astype(int).tolist(),
        "persistent_8_fraction": float(np.mean(np.sum(visibility[:8], axis=0) == min(8, visibility.shape[0]))),
        "persistent_16_fraction": float(np.mean(np.all(visibility, axis=0))) if visibility.shape[0] == 16 else None,
        "lifetime": summarize_distribution(lifetimes),
    }


def process_window(item: dict, track_path: Path, depth_runner: DepthProRunner, output: Path, run_name: str):
    started = time.perf_counter()
    decoded = decode(item)
    with np.load(track_path, allow_pickle=False) as arrays:
        uv = arrays["uv"]
        visibility = arrays["visibility"]
    depths, focals, frame_depth_valid = depth_runner.infer(decoded)
    intrinsics, fixed_focal_px = causal_first_frame_intrinsics(depths, focals)
    adjacent, pair_valid, information = rgbd_odometry(decoded, depths, intrinsics)
    world_from_camera, pose_valid = accumulate_world_from_camera(adjacent, pair_valid)
    sampled_depth, sampled_valid = sample_depth_at_uv(depths, uv)
    observation_valid = visibility & sampled_valid & frame_depth_valid[:, None]
    xyz, geometry_valid = world_xyz(
        uv, sampled_depth, intrinsics, world_from_camera, observation_valid, pose_valid
    )
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
        lineage={"dataset_revision": DATA_REVISION, "official_split": "val"},
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
        },
    )
    prefix = output / "particle_sequences" / run_name / item["source_video_id"]
    save_particle_sequence(sequence, prefix)
    valid_steps = geometry_valid[1:] & geometry_valid[:-1]
    displacement = np.linalg.norm(xyz[1:] - xyz[:-1], axis=-1)
    displacement[~valid_steps] = np.nan
    return {
        "source_video_id": item["source_video_id"],
        "frame_indices": item["frame_indices"],
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
        "same_track_step_distance_m": summarize_distribution(displacement),
        "elapsed_s": time.perf_counter() - started,
        "particle_prefix": str(prefix),
        "world_from_camera": [
            [[float(value) if np.isfinite(value) else None for value in row] for row in matrix]
            for matrix in world_from_camera
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, choices=(2, 8, 32), required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--frames", type=int, choices=(8, 16), default=16)
    parser.add_argument("--phase", choices=("all", "tracking", "geometry"), default="all")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--tapnet-source", type=Path, required=True)
    parser.add_argument("--tapnet-checkpoint", type=Path, required=True)
    parser.add_argument("--depth-source", type=Path, required=True)
    parser.add_argument("--depth-checkpoint", type=Path, required=True)
    args = parser.parse_args()
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    items = selected_real_windows(args.limit, args.frames)
    import open3d
    import torch
    write_json(output / "config.json", {
        "population": "REAL ONLY",
        "frames_per_window": args.frames,
        "tracks": 64,
        "tracker_process_hw": [256, 256],
        "pose_max_side": 320,
        "principal_point": "EXPERIMENTAL_CENTER_PRINCIPAL_POINT_ASSUMPTION",
        "intrinsics_policy": "CAUSAL_FIRST_FRAME_DEPTH_PRO_FOCAL_ASSUMPTION",
        "distortion": "EXPERIMENTAL_ZERO_DISTORTION_ASSUMPTION",
        "execution_phase": args.phase,
        "fake_used": False,
    })
    write_json(output / "environment.json", {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "numpy": np.__version__,
        "open3d": open3d.__version__,
        "environment": "/root/autodl-tmp/envs/v7-explicit-geometry",
        "inherits_system_site_packages": True,
    })
    write_json(output / f"sample_manifest_{args.run_name}.json", {"population": "REAL ONLY", "items": items})
    track_root = output / "tracking" / args.run_name
    track_root.mkdir(parents=True, exist_ok=True)
    if args.phase != "geometry":
        tracker = OnlineBootsTapir(args.tapnet_source, args.tapnet_checkpoint)
        for item in items:
            decoded = decode(item)
            uv, visibility = tracker.track(decoded)
            np.savez_compressed(track_root / f"{item['source_video_id']}.npz", uv=uv, visibility=visibility)
        del tracker
        gc.collect()
    if args.phase == "tracking":
        return
    if any(not (track_root / f"{item['source_video_id']}.npz").is_file() for item in items):
        raise RuntimeError("geometry phase requires an existing track artifact for every fixed window")
    torch.cuda.empty_cache()
    depth_runner = DepthProRunner(args.depth_source, args.depth_checkpoint)
    torch.cuda.reset_peak_memory_stats()
    results = []
    started = time.perf_counter()
    for number, item in enumerate(items, 1):
        result = process_window(
            item, track_root / f"{item['source_video_id']}.npz", depth_runner, output, args.run_name
        )
        results.append(result)
        if not args.quiet:
            print(number, len(items), item["source_video_id"], result["coverage"]["final_geometry_valid_fraction"], flush=True)
    elapsed = time.perf_counter() - started
    write_json(output / f"run_{args.run_name}.json", {"windows": results})
    resource = {
        "run_name": args.run_name,
        "windows": len(items),
        "elapsed_s": elapsed,
        "peak_gpu_bytes": int(torch.cuda.max_memory_allocated()),
    }
    write_json(output / "resource_metrics.json", resource)
    write_json(output / "tracking_metrics.json", {"windows": [x["tracking"] for x in results]})
    write_json(output / "depth_metrics.json", {"windows": [{"id": x["source_video_id"], "focal_px": x["focal_px"], "valid": x["frame_depth_valid"]} for x in results]})
    write_json(output / "pose_metrics.json", {"windows": [{"id": x["source_video_id"], "pair_valid": x["pose_pair_valid"], "information_trace": x["pose_information_trace"]} for x in results]})
    write_json(output / "coverage.json", {"windows": [{"id": x["source_video_id"], **x["coverage"]} for x in results]})
    write_json(output / "xyz_stability.json", {"windows": [{"id": x["source_video_id"], **x["same_track_step_distance_m"]} for x in results]})
    write_json(output / "third_party_assets.json", {
        "tapnet_source_sha": TAPNET_SHA,
        "depth_pro_source_sha": DEPTH_PRO_SHA,
        "boots_tapir_checkpoint": file_identity(args.tapnet_checkpoint),
        "depth_pro_checkpoint": file_identity(args.depth_checkpoint),
        "open3d": "0.19.0",
    })


if __name__ == "__main__":
    main()
