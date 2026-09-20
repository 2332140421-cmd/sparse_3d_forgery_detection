"""V8 raw-video frontend.

This module deliberately has no import from research_tools.v7.  It stores a
frame-level MoGe point map/depth/mask and a window-level short-prefix tracking
summary.  Grouping and learned features are a later stage so that the raw
observation and the model input remain separately auditable.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

from .common import (
    CANVAS_HW,
    CELL_COUNT,
    CELL_SIZE,
    GRID_H,
    GRID_W,
    FRAME_COUNT,
    finite_or_zero,
    read_json,
    sha256_file,
    write_json_atomic,
    write_npz_atomic,
)


def letterbox(frame_bgr: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    h, w = frame_bgr.shape[:2]
    ch, cw = CANVAS_HW
    scale = min(cw / w, ch / h)
    nw, nh = max(1, round(w * scale)), max(1, round(h * scale))
    resized = cv2.resize(frame_bgr, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((ch, cw, 3), dtype=np.uint8)
    left, top = (cw - nw) // 2, (ch - nh) // 2
    canvas[top : top + nh, left : left + nw] = resized
    return canvas, {
        "source_hw": [int(h), int(w)],
        "canvas_hw": list(CANVAS_HW),
        "scale": float(scale),
        "pad_xy": [int(left), int(top)],
        "content_hw": [int(nh), int(nw)],
    }


def _cell_reduce(arr: np.ndarray, valid: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Mean dense map into 16x16 cells without silently deleting empty cells."""
    h, w = arr.shape[:2]
    out = np.zeros((CELL_COUNT, arr.shape[-1]), dtype=np.float32)
    count = np.zeros(CELL_COUNT, dtype=np.float32)
    for y in range(GRID_H):
        for x in range(GRID_W):
            idx = y * GRID_W + x
            sl = arr[y * CELL_SIZE : (y + 1) * CELL_SIZE, x * CELL_SIZE : (x + 1) * CELL_SIZE]
            m = np.ones(sl.shape[:2], dtype=bool) if valid is None else valid[
                y * CELL_SIZE : (y + 1) * CELL_SIZE, x * CELL_SIZE : (x + 1) * CELL_SIZE
            ]
            if m.any():
                out[idx] = sl[m].mean(axis=0)
                count[idx] = float(m.sum())
    return out, count


class MogeV2:
    def __init__(self, checkpoint: str, device: str = "cuda"):
        self.device = torch.device(device)
        os.environ.setdefault("XFORMERS_DISABLED", "1")
        from moge.model.v2 import MoGeModel

        self.model = MoGeModel.from_pretrained(checkpoint).to(self.device).eval()
        self.checkpoint = checkpoint

    @torch.inference_mode()
    def infer(self, rgb_u8: np.ndarray) -> dict[str, np.ndarray]:
        rgb = torch.from_numpy(rgb_u8[:, :, ::-1].copy()).permute(2, 0, 1).float().div(255).to(self.device)
        out = self.model.infer(rgb, num_tokens=1200, resolution_level=0, apply_mask=False, use_fp16=True)
        result: dict[str, np.ndarray] = {}
        for k, v in out.items():
            if isinstance(v, torch.Tensor):
                result[k] = v.detach().float().cpu().numpy()
        points = result.get("points")
        depth = result.get("depth")
        mask = result.get("mask")
        if points is None or depth is None:
            raise RuntimeError("MoGe-2 did not return point/depth maps")
        finite = np.isfinite(points).all(axis=-1) & np.isfinite(depth)
        if mask is not None:
            finite &= mask.astype(bool)
        result["points_raw"] = points.astype(np.float32)
        result["depth_raw"] = depth.astype(np.float32)
        result["validity"] = finite.astype(bool)
        return result


class CoTrackerShortPrefix:
    """Official CoTracker3 short-prefix calls with endpoint padding recorded."""

    def __init__(self, checkpoint: str, device: str = "cuda", grid_size: int = 32):
        from cotracker.predictor import CoTrackerOnlinePredictor

        self.device = torch.device(device)
        self.grid_size = int(grid_size)
        self.model = CoTrackerOnlinePredictor(checkpoint=checkpoint, window_len=16).to(self.device).eval()
        self.checkpoint = checkpoint

    @torch.inference_mode()
    def query(self, frames_rgb: list[np.ndarray], current_index: int) -> dict[str, Any]:
        if current_index < 0 or current_index >= len(frames_rgb):
            raise IndexError(current_index)
        start = max(0, current_index - 2)
        prefix = frames_rgb[start : current_index + 1]
        pad_count = 16 - len(prefix)
        if pad_count > 0:
            prefix = [prefix[0]] * pad_count + prefix
        video = np.stack(prefix, axis=0)
        # CoTracker consumes RGB tensors in [B,T,C,H,W]. Query t=15 is the
        # last real frame; the repeated prefix is never counted as a real PTS.
        x = torch.from_numpy(video).permute(0, 3, 1, 2).unsqueeze(0).float().to(self.device)
        h, w = CANVAS_HW
        yy, xx = np.mgrid[0:self.grid_size, 0:self.grid_size]
        q = np.stack([np.full(xx.size, 15), xx.reshape(-1) * (w - 1) / (self.grid_size - 1), yy.reshape(-1) * (h - 1) / (self.grid_size - 1)], axis=-1)
        queries = torch.from_numpy(q.astype(np.float32))[None].to(self.device)
        self.model(video_chunk=x, is_first_step=True, queries=queries)
        tracks, visibility = self.model(video_chunk=x)
        tracks = tracks[0].detach().float().cpu().numpy()
        visibility = visibility[0].detach().cpu().numpy().astype(bool)
        return {
            "tracks": tracks,
            "visibility": visibility,
            "grid_size": self.grid_size,
            "prefix_start_index": int(start),
            "prefix_length_real": int(current_index - start + 1),
            "endpoint_padding_count": int(pad_count),
            "query_frame_in_padded_prefix": 15,
        }


def decode_frame(cap: cv2.VideoCapture, frame_index: int, expected_pts: float | None = None) -> tuple[np.ndarray, float]:
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
    ok, frame = cap.read()
    if not ok or frame is None:
        raise RuntimeError(f"decode failed at frame {frame_index}")
    pts = float(cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0)
    if not np.isfinite(pts) or pts <= 0:
        pts = float(expected_pts) if expected_pts is not None else float("nan")
    return frame, pts


def _frame_cache_key(video_path: str, frame_index: int) -> str:
    return hashlib.sha256(f"{sha256_file(video_path)}:{frame_index}:v8-canvas256".encode()).hexdigest()[:24]


def extract_window(
    row: dict[str, Any],
    output: str | Path,
    moge: MogeV2,
    tracker: CoTrackerShortPrefix,
    overwrite: bool = False,
) -> dict[str, Any]:
    output = Path(output)
    wid = str(row["window_id"])
    safe = hashlib.sha256(wid.encode()).hexdigest()[:24]
    out_npz = output / "frontend" / f"{safe}.npz"
    out_json = output / "frontend" / f"{safe}.json"
    if out_npz.exists() and out_json.exists() and not overwrite:
        return read_json(out_json)

    video_path = str(row["video_path"])
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video: {video_path}")
    frame_indices = [int(x) for x in row.get("v8_frame_indices", row["frame_indices"][:FRAME_COUNT])]
    expected_pts = [float(x) if x is not None else None for x in row.get("v8_timestamps_s", row.get("timestamps_s", [])[:FRAME_COUNT])]
    rgbs: list[np.ndarray] = []
    pts: list[float] = []
    frame_meta: list[dict[str, Any]] = []
    cell_xyz, cell_rgb, cell_geom, cell_area = [], [], [], []
    frame_cache = output / "raw_frontend" / "frames"
    frame_cache.mkdir(parents=True, exist_ok=True)
    video_sha = sha256_file(video_path)
    for k, fi in enumerate(frame_indices):
        frame, actual_pts = decode_frame(cap, fi, expected_pts[k] if k < len(expected_pts) else None)
        rgb_canvas, mapping = letterbox(frame)
        key = _frame_cache_key(video_path, fi)
        cache_path = frame_cache / f"{key}.npz"
        cache_meta = frame_cache / f"{key}.json"
        if cache_path.exists() and cache_meta.exists():
            cmeta = read_json(cache_meta)
            if cmeta.get("video_sha256") != video_sha or int(cmeta.get("frame_index", -1)) != fi:
                raise RuntimeError(f"frame cache identity mismatch for {wid} frame {fi}")
            with np.load(cache_path, allow_pickle=False) as z:
                pts_map, depth_map, valid_map = z["points_raw"], z["depth_raw"], z["validity"]
                cached_rgb = z["rgb"]
            if cached_rgb.shape != rgb_canvas.shape:
                raise RuntimeError(f"frame cache shape mismatch for {wid} frame {fi}")
            rgb_canvas = cached_rgb
        else:
            pred = moge.infer(rgb_canvas)
            pts_map, depth_map, valid_map = pred["points_raw"], pred["depth_raw"], pred["validity"]
            write_npz_atomic(cache_path, rgb=rgb_canvas, points_raw=pts_map, depth_raw=depth_map, validity=valid_map)
            write_json_atomic(cache_meta, {
                "video_sha256": video_sha, "frame_index": fi, "pts_s": actual_pts,
                "mapping": mapping, "shape": list(rgb_canvas.shape), "provider": "MoGe-2-v2",
            })
        rgb_canvas_rgb = rgb_canvas[:, :, ::-1].copy()
        rgb_cells, _ = _cell_reduce(rgb_canvas_rgb.astype(np.float32) / 255.0)
        xyz_cells, counts = _cell_reduce(pts_map.astype(np.float32), valid_map)
        areas = np.full(CELL_COUNT, CELL_SIZE * CELL_SIZE, dtype=np.float32)
        # Padding cells have zero image area; partial border content is explicit.
        top, left = mapping["pad_xy"][1], mapping["pad_xy"][0]
        nh, nw = mapping["content_hw"]
        for cy in range(GRID_H):
            for cx in range(GRID_W):
                y0, y1 = cy * CELL_SIZE, (cy + 1) * CELL_SIZE
                x0, x1 = cx * CELL_SIZE, (cx + 1) * CELL_SIZE
                iy0, iy1 = max(y0, top), min(y1, top + nh)
                ix0, ix1 = max(x0, left), min(x1, left + nw)
                areas[cy * GRID_W + cx] = max(0, iy1 - iy0) * max(0, ix1 - ix0)
        cell_xyz.append(xyz_cells)
        cell_rgb.append(rgb_cells)
        cell_geom.append(counts > 0)
        cell_area.append(areas)
        rgbs.append(rgb_canvas_rgb)
        pts.append(actual_pts)
        frame_meta.append({"frame_index": fi, "pts_s": actual_pts, "mapping": mapping, "cache_key": key})
    cap.release()

    track_cov, motion_xy, motion_valid, history_avail, tracker_meta = [], [], [], [], []
    all_tracks, all_visibility = [], []
    for ti in range(len(rgbs)):
        tr = tracker.query(rgbs, ti)
        tracks, vis = tr["tracks"], tr["visibility"]
        all_tracks.append(tracks)
        all_visibility.append(vis)
        now_xy = tracks[-1]
        now_vis = vis[-1]
        cov = np.zeros(CELL_COUNT, np.float32)
        mxy = np.zeros((CELL_COUNT, 2), np.float32)
        mv = np.zeros(CELL_COUNT, bool)
        for qy in range(tracker.grid_size):
            for qx in range(tracker.grid_size):
                qi = qy * tracker.grid_size + qx
                cy = min(GRID_H - 1, qy // 2)
                cx = min(GRID_W - 1, qx // 2)
                ci = cy * GRID_W + cx
                cov[ci] += float(now_vis[qi]) / 4.0
                if ti >= 1 and vis[-2, qi] and now_vis[qi]:
                    mxy[ci] += (now_xy[qi] - tracks[-2, qi]) / 4.0
                    mv[ci] = True
        track_cov.append(cov)
        motion_xy.append(mxy)
        motion_valid.append(mv)
        history_avail.append(np.full(CELL_COUNT, bool(ti > 0), dtype=bool))
        tracker_meta.append({k: v for k, v in tr.items() if k not in {"tracks", "visibility"}})
    arr = {
        "cell_xyz": np.asarray(cell_xyz, dtype=np.float32),
        "cell_rgb": np.asarray(cell_rgb, dtype=np.float32),
        "cell_geometry_valid": np.asarray(cell_geom, dtype=bool),
        "cell_area": np.asarray(cell_area, dtype=np.float32),
        "track_coverage": np.asarray(track_cov, dtype=np.float32),
        "motion_xy": np.asarray(motion_xy, dtype=np.float32),
        "motion_valid": np.asarray(motion_valid, dtype=bool),
        "history_available": np.asarray(history_avail, dtype=bool),
        "tracker_tracks": np.asarray(all_tracks, dtype=np.float32),
        "tracker_visibility": np.asarray(all_visibility, dtype=bool),
    }
    write_npz_atomic(out_npz, **arr)
    meta = {
        "status": "FRONTEND_COMPLETE",
        "window_id": wid,
        "source_id": row["source_id"],
        "role": row["role"],
        "label": int(row["label"]),
        "video_path": video_path,
        "video_sha256": video_sha,
        "frame_indices": frame_indices,
        "pts_s": pts,
        "expected_pts_s": expected_pts,
        "frame_meta": frame_meta,
        "canvas_hw": list(CANVAS_HW),
        "cell_size": CELL_SIZE,
        "provider": {
            "geometry": "MoGe-2-v2",
            "geometry_checkpoint": moge.checkpoint,
            "tracker": "CoTracker3-online-short-prefix",
            "tracker_checkpoint": tracker.checkpoint,
            "tracker_grid_size": tracker.grid_size,
            "max_prefix_frames": 3,
        },
        "tracker_meta": tracker_meta,
        "created_unix": time.time(),
        "npz": str(out_npz),
    }
    write_json_atomic(out_json, meta)
    return meta
