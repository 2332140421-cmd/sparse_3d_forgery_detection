"""Fixed 384-grid frontend attempt for the dense local-structure pilot.

The implementation intentionally fails loudly when the declared 384 grid does
not fit the available tracker causal state.  It never falls back to a smaller
grid or silently reuses the old 256 analysis.
"""

from __future__ import annotations

import gc
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from sparse3d_forgery.video_input import VideoSource, decode_video
from sparse3d_forgery.experiments.v7_explicit_geometry_frontend import RasterScale

from .grouping import analysis_grid, assign_groups, fixed_local_edges


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def track_dense_window(row: Mapping[str, Any], *, tapnet_source: Path, checkpoint: Path) -> dict[str, Any]:
    """Run the declared 384/3-pixel BootsTAPIR attempt for one window."""

    import torch
    import torch.nn.functional as functional
    import tree

    started = time.perf_counter()
    decoded = decode_video(
        VideoSource(
            sample_id=f"v7-dense-{row['window_id']}",
            source_video_id=f"{row['source_id']}-{row['role']}",
            source_locator=Path(str(row["video_path"])),
        ),
        [int(index) for index in row["frame_indices"]],
    )
    source_h, source_w = decoded.frames[0].rgb.shape[:2]
    process_size, spacing = 384, 3
    grid = analysis_grid(process_size, spacing)
    sys.path.insert(0, str(tapnet_source))
    from tapnet.torch import tapir_model
    model = tapir_model.TAPIR(pyramid_level=1, use_casual_conv=True)
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    device = torch.device("cuda")
    model.to(device).eval()
    frames = np.stack([frame.rgb for frame in decoded.frames])
    tensor = torch.from_numpy(frames).to(device).permute(0, 3, 1, 2).float()
    tensor = functional.interpolate(tensor, size=(process_size, process_size), mode="bilinear", align_corners=False).permute(0, 2, 3, 1)
    tensor = tensor / 255.0 * 2.0 - 1.0
    query = np.stack((np.zeros(grid.shape[0], np.float32), grid[:, 1], grid[:, 0]), axis=1)
    query_tensor = torch.from_numpy(query).to(device)
    with torch.inference_mode():
        first = tensor[None, 0:1]
        feature_grids = model.get_feature_grids(first, is_training=False)
        query_features = model.get_query_features(first, is_training=False, query_points=query_tensor[None], feature_grids=feature_grids)
        causal_state = model.construct_initial_causal_state(grid.shape[0], len(query_features.resolutions) - 1)
        causal_state = tree.map_structure(lambda value: value.to(device), causal_state)
        tracks, visibility = [], []
        for frame in tensor:
            single = frame[None, None]
            grids = model.get_feature_grids(single, is_training=False)
            result = model.estimate_trajectories(
                single.shape[-3:-1], is_training=False, feature_grids=grids,
                query_features=query_features, query_points_in_video=None,
                query_chunk_size=256, causal_context=causal_state, get_causal_context=True,
            )
            causal_state = result["causal_context"]
            point = result["tracks"][-1][0, :, 0].detach().cpu().numpy()
            occlusion = result["occlusion"][-1][0, :, 0]
            expected = result["expected_dist"][-1][0, :, 0]
            visible = ((1 - torch.sigmoid(occlusion)) * (1 - torch.sigmoid(expected)) > 0.5)
            tracks.append(point)
            visibility.append(visible.detach().cpu().numpy())
    mapping = RasterScale((source_h, source_w), (process_size, process_size))
    raw_uv = mapping.process_to_source(np.asarray(tracks, dtype=np.float32))
    visibility_array = np.asarray(visibility, dtype=bool)
    finite_inside = np.all(np.isfinite(raw_uv), axis=-1) & (raw_uv[..., 0] >= 0) & (raw_uv[..., 0] < source_w) & (raw_uv[..., 1] >= 0) & (raw_uv[..., 1] < source_h)
    visibility_array &= finite_inside
    groups, group_summary = assign_groups(grid, None, image_size=process_size)
    edges = fixed_local_edges(grid, groups, max_neighbors=8)
    del model, tensor, query_features, causal_state
    gc.collect()
    torch.cuda.empty_cache()
    return {
        "row": dict(row),
        "frame_indices": decoded.frame_indices.tolist(),
        "timestamps_s": decoded.timestamps_s.tolist(),
        "raw_uv": raw_uv,
        "visibility": visibility_array,
        "query_start_uv_analysis": grid,
        "group_ids": groups,
        "group_summary": group_summary,
        "edges": edges,
        "elapsed_s": time.perf_counter() - started,
        "source_hw": [int(source_h), int(source_w)],
        "analysis_hw": [process_size, process_size],
        "query_count": int(grid.shape[0]),
    }
