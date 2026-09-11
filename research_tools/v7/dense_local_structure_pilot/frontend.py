"""Chunked 384-grid BootsTAPIR frontend and official YOLO measurements.

The query batch is an execution detail only: global query identities, the
analysis grid, and the returned arrays always retain all 16,384 candidates.
Each batch owns its query features and causal state for the complete window.
"""

from __future__ import annotations

import gc
import hashlib
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from sparse3d_forgery.experiments.v7_explicit_geometry_frontend import (
    RasterScale,
    centered_intrinsics,
    sample_depth_at_uv,
    world_xyz,
)
from sparse3d_forgery.video_input import VideoSource, decode_video

from .grouping import analysis_grid


PROCESS_SIZE = 384
GRID_SPACING = 3
DEFAULT_BATCH_SIZE = 128


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _prepare_frame(rgb: np.ndarray, device: Any, functional: Any) -> Any:
    """Resize one uint8 RGB frame without staging the complete clip on GPU."""

    import torch

    tensor = torch.from_numpy(np.ascontiguousarray(rgb)).to(device)
    tensor = tensor.permute(2, 0, 1).unsqueeze(0).float()
    tensor = functional.interpolate(tensor, size=(PROCESS_SIZE, PROCESS_SIZE), mode="bilinear", align_corners=False)
    return (tensor.permute(0, 2, 3, 1).unsqueeze(1) / 255.0 * 2.0 - 1.0).contiguous()


def _track_batch(
    model: Any,
    feature_grids: Sequence[Any],
    first_frame: Any,
    query_points: np.ndarray,
    *,
    device: Any,
) -> tuple[np.ndarray, np.ndarray]:
    """Track one query batch over every cached frame feature."""

    import torch

    query_tensor = torch.from_numpy(query_points).to(device)
    with torch.inference_mode():
        query_features = model.get_query_features(
            first_frame,
            is_training=False,
            query_points=query_tensor[None],
            feature_grids=feature_grids[0],
        )
        causal_state = model.construct_initial_causal_state(query_points.shape[0], len(query_features.resolutions) - 1)
        causal_state = [{key: value.to(device) for key, value in state.items()} for state in causal_state]
        tracks: list[np.ndarray] = []
        visible: list[np.ndarray] = []
        for feature_grid in feature_grids:
            result = model.estimate_trajectories(
                (PROCESS_SIZE, PROCESS_SIZE),
                is_training=False,
                feature_grids=feature_grid,
                query_features=query_features,
                query_points_in_video=None,
                query_chunk_size=256,
                causal_context=causal_state,
                get_causal_context=True,
            )
            causal_state = result["causal_context"]
            tracks.append(result["tracks"][-1][0, :, 0].detach().cpu().numpy())
            occlusion = result["occlusion"][-1][0, :, 0]
            expected = result["expected_dist"][-1][0, :, 0]
            visible.append((((1 - torch.sigmoid(occlusion)) * (1 - torch.sigmoid(expected))) > 0.5).detach().cpu().numpy())
    return np.asarray(tracks, dtype=np.float32), np.asarray(visible, dtype=bool)


def track_dense_window(
    row: Mapping[str, Any],
    *,
    tapnet_source: Path,
    checkpoint: Path,
    batch_size: int = DEFAULT_BATCH_SIZE,
    query_indices: Sequence[int] | None = None,
    progress_prefix: str = "frontend",
) -> dict[str, Any]:
    """Track all declared queries with externally batched causal state."""

    if batch_size not in {64, 128, 256}:
        raise ValueError("batch_size must be one of 64, 128, 256")
    import torch
    import torch.nn.functional as functional

    started = time.perf_counter()
    decoded = decode_video(
        VideoSource(
            sample_id=f"v7-dense-{row['window_id']}",
            source_video_id=f"{row['source_id']}-{row['role']}",
            source_locator=Path(str(row["video_path"])),
        ),
        [int(index) for index in row["frame_indices"]],
    )
    frames_rgb = np.stack([frame.rgb for frame in decoded.frames]).astype(np.uint8, copy=False)
    source_h, source_w = frames_rgb.shape[1:3]
    grid = analysis_grid(PROCESS_SIZE, GRID_SPACING)
    selected = np.arange(grid.shape[0], dtype=np.int64) if query_indices is None else np.asarray(query_indices, dtype=np.int64)
    if selected.ndim != 1 or np.any(selected < 0) or np.any(selected >= grid.shape[0]) or len(np.unique(selected)) != selected.size:
        raise ValueError("query_indices must be unique global query ids")
    query_points = np.stack((np.zeros(grid.shape[0], np.float32), grid[:, 1], grid[:, 0]), axis=1)

    sys.path.insert(0, str(tapnet_source))
    from tapnet.torch import tapir_model

    model = tapir_model.TAPIR(pyramid_level=1, use_casual_conv=True)
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    device = torch.device("cuda")
    model.to(device).eval()
    torch.cuda.reset_peak_memory_stats(device)

    feature_grids: list[Any] = []
    first_frame = None
    with torch.inference_mode():
        for frame_number, rgb in enumerate(frames_rgb):
            prepared = _prepare_frame(rgb, device, functional)
            if first_frame is None:
                first_frame = prepared
            feature_grids.append(model.get_feature_grids(prepared, is_training=False))
            del prepared
            print(f"{progress_prefix} window={row['window_id']} feature_frame={frame_number + 1}/{len(frames_rgb)}", flush=True)
    assert first_frame is not None

    raw_process = np.full((len(frames_rgb), selected.size, 2), np.nan, dtype=np.float32)
    visibility = np.zeros((len(frames_rgb), selected.size), dtype=bool)
    total_batches = int(np.ceil(selected.size / batch_size))
    for batch_number, start in enumerate(range(0, selected.size, batch_size), 1):
        stop = min(start + batch_size, selected.size)
        batch_raw, batch_visible = _track_batch(
            model,
            feature_grids,
            first_frame,
            query_points[selected[start:stop]],
            device=device,
        )
        raw_process[:, start:stop] = batch_raw
        visibility[:, start:stop] = batch_visible
        allocated_gb = float(torch.cuda.memory_allocated(device) / 1024**3)
        print(
            f"{progress_prefix} window={row['window_id']} query_batch={batch_number}/{total_batches} "
            f"queries={start}:{stop} elapsed={time.perf_counter() - started:.1f}s gpu={allocated_gb:.2f}GiB",
            flush=True,
        )

    mapping = RasterScale((source_h, source_w), (PROCESS_SIZE, PROCESS_SIZE))
    raw_uv = mapping.process_to_source(raw_process)
    inside = np.all(np.isfinite(raw_uv), axis=-1) & (raw_uv[..., 0] >= 0) & (raw_uv[..., 0] < source_w) & (raw_uv[..., 1] >= 0) & (raw_uv[..., 1] < source_h)
    visibility &= inside
    peak_memory = int(torch.cuda.max_memory_allocated(device))
    del model, feature_grids, first_frame
    gc.collect()
    torch.cuda.empty_cache()
    return {
        "row": dict(row),
        "frame_indices": decoded.frame_indices.tolist(),
        "timestamps_s": decoded.timestamps_s.tolist(),
        "frames_rgb": frames_rgb,
        "raw_uv": raw_uv,
        "visibility": visibility,
        "query_ids": selected,
        "query_start_uv_analysis": grid[selected],
        "source_hw": [int(source_h), int(source_w)],
        "analysis_hw": [PROCESS_SIZE, PROCESS_SIZE],
        "query_count": int(selected.size),
        "batch_size": int(batch_size),
        "peak_gpu_bytes": peak_memory,
        "elapsed_s": time.perf_counter() - started,
    }


def normalize_yolo_depth(result: Any, expected_hw: tuple[int, int]) -> np.ndarray:
    depth_object = getattr(result, "depth", None)
    if depth_object is None or not hasattr(depth_object, "data"):
        raise ValueError("YOLO result has no depth.data output")
    data = depth_object.data
    if hasattr(data, "detach"):
        data = data.detach().float().cpu().numpy()
    depth = np.asarray(data)
    while depth.ndim > 2 and depth.shape[0] == 1:
        depth = depth[0]
    if depth.shape != expected_hw:
        raise ValueError(f"depth shape {depth.shape} != source raster {expected_hw}")
    depth = depth.astype(np.float32, copy=False)
    valid = np.isfinite(depth) & (depth > 0)
    if not np.any(valid):
        raise ValueError("YOLO depth has no finite positive pixels")
    depth[~valid] = np.nan
    return depth


def infer_depth_and_masks(
    frames_rgb: np.ndarray,
    *,
    depth_weight: Path,
    seg_weight: Path,
    device: int = 0,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Run depth on every frame and segmentation on the start frame only."""

    import cv2
    import torch
    from ultralytics import YOLO

    depth_started = time.perf_counter()
    depth_model = YOLO(str(depth_weight))
    depths = []
    for rgb in frames_rgb:
        result = depth_model.predict(source=rgb, imgsz=768, device=device, verbose=False)[0]
        depths.append(normalize_yolo_depth(result, tuple(rgb.shape[:2])))
    depth_elapsed = time.perf_counter() - depth_started
    del depth_model
    gc.collect()
    torch.cuda.empty_cache()

    seg_started = time.perf_counter()
    seg_model = YOLO(str(seg_weight))
    segmentation = seg_model.predict(source=frames_rgb[0], device=device, verbose=False)[0]
    masks_original: list[np.ndarray] = []
    if getattr(segmentation, "masks", None) is not None:
        for polygon in segmentation.masks.xy:
            canvas = np.zeros(frames_rgb[0].shape[:2], dtype=np.uint8)
            points = np.rint(np.asarray(polygon, dtype=np.float32)).astype(np.int32)
            if points.shape[0] >= 3:
                cv2.fillPoly(canvas, [points], 1)
            masks_original.append(canvas.astype(bool))
    masks_analysis = np.asarray(
        [cv2.resize(mask.astype(np.uint8), (PROCESS_SIZE, PROCESS_SIZE), interpolation=cv2.INTER_NEAREST).astype(bool) for mask in masks_original],
        dtype=bool,
    ) if masks_original else np.zeros((0, PROCESS_SIZE, PROCESS_SIZE), dtype=bool)
    seg_elapsed = time.perf_counter() - seg_started
    del seg_model, segmentation
    gc.collect()
    torch.cuda.empty_cache()
    return np.stack(depths).astype(np.float32), masks_analysis, {
        "depth_elapsed_s": depth_elapsed,
        "seg_elapsed_s": seg_elapsed,
        "depth_shape": list(depths[0].shape),
        "mask_count": int(masks_analysis.shape[0]),
        "mask_pixel_shape": [PROCESS_SIZE, PROCESS_SIZE],
    }


def build_geometry(
    raw_uv: np.ndarray,
    depths: np.ndarray,
    *,
    focal_px: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Back-project in per-frame camera coordinates without visibility gating."""

    height, width = depths.shape[1:]
    k = centered_intrinsics(height, width, focal_px)
    sampled, valid = sample_depth_at_uv(depths, raw_uv)
    intrinsics = np.repeat(k[None], depths.shape[0], axis=0)
    identity = np.repeat(np.eye(4, dtype=np.float64)[None], depths.shape[0], axis=0)
    pose_valid = np.ones(depths.shape[0], dtype=bool)
    xyz, geometry_validity = world_xyz(raw_uv, sampled, intrinsics, identity, valid, pose_valid)
    return xyz, geometry_validity, {
        "depth_semantics": "official YOLO result.depth.data interpreted as positive z-depth",
        "intrinsics": k.tolist(),
        "intrinsics_policy": "existing fixed focal convention; no unit-intrinsics fallback",
        "coordinate_frame": "camera_coordinates_each_frame",
        "camera_motion_compensation": False,
        "valid_depth_fraction": float(np.mean(valid)),
    }
