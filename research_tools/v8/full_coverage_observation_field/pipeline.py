"""Stageable V8 runner: plan -> frontend -> features -> train -> evaluate -> report."""

from __future__ import annotations

import csv
import json
import math
import os
import shutil
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from PIL import Image

from .common import (
    CELL_COUNT,
    FRAME_COUNT,
    GRID_H,
    GRID_W,
    PROTOCOL_VERSION,
    SEEDS,
    cell_neighbors,
    manifest_rows,
    read_json,
    sha256_file,
    stable_hash,
    write_json_atomic,
    write_npz_atomic,
)
from .frontend import CoTrackerShortPrefix, MogeV2, extract_window
from .model import FullCoverageModel, parameter_count


class FrozenResNet18Spatial:
    """Frozen ImageNet ResNet18 layer-1 feature map projected to base cells.

    The input is the letterboxed RGB frame saved by the new V8 frontend, not a
    V7 feature or trajectory cache.  A layer-1 spatial map is interpolated to
    the 16x16 base-cell lattice; no trainable parameter is introduced here.
    """

    def __init__(self, device: str = "cuda"):
        from torchvision.models import ResNet18_Weights, resnet18

        self.device = torch.device(device)
        weights = ResNet18_Weights.DEFAULT
        net = resnet18(weights=weights).to(self.device).eval()
        for p in net.parameters():
            p.requires_grad_(False)
        self.backbone = torch.nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool, net.layer1).eval()
        self.mean = torch.tensor(weights.transforms().mean, device=self.device, dtype=torch.float32).view(1, 3, 1, 1)
        self.std = torch.tensor(weights.transforms().std, device=self.device, dtype=torch.float32).view(1, 3, 1, 1)
        self.checkpoint = str(torch.hub.get_dir() + "/checkpoints/resnet18-f37072fd.pth")
        self.feature_dim = 64

    @torch.inference_mode()
    def encode(self, frames_rgb: np.ndarray) -> np.ndarray:
        if frames_rgb.ndim != 4 or frames_rgb.shape[-1] != 3:
            raise ValueError(f"frames_rgb must be [T,H,W,3], got {frames_rgb.shape}")
        x = torch.from_numpy(frames_rgb).to(self.device, dtype=torch.float32).permute(0, 3, 1, 2).div_(255.0)
        x = torch.nn.functional.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
        y = self.backbone((x - self.mean) / self.std)
        y = torch.nn.functional.interpolate(y, size=(GRID_H, GRID_W), mode="bilinear", align_corners=False)
        return y.permute(0, 2, 3, 1).reshape(frames_rgb.shape[0], CELL_COUNT, self.feature_dim).float().cpu().numpy()


def _now() -> float:
    return time.time()


def _status_path(out: Path) -> Path:
    return out / "stage_status.json"


def _load_status(out: Path) -> dict[str, Any]:
    p = _status_path(out)
    return read_json(p) if p.exists() else {"status": "NEW", "stages": {}, "updated_unix": _now()}


def _stage(out: Path, name: str, status: str, **fields: Any) -> None:
    s = _load_status(out)
    s.setdefault("stages", {})[name] = {"status": status, "updated_unix": _now(), **fields}
    s["updated_unix"] = _now()
    s["status"] = "RUNNING" if status == "RUNNING" else s.get("status", "RUNNING")
    write_json_atomic(_status_path(out), s)


def _safe_name(s: str) -> str:
    import hashlib

    return hashlib.sha256(s.encode()).hexdigest()[:24]


def build_manifest(out: Path, geometry_root: Path, source_root: Path) -> list[dict[str, Any]]:
    feature_rows = manifest_rows(geometry_root / "inputs" / "feature_manifest.json")
    old_rows = {r["window_id"]: r for r in manifest_rows(source_root / "manifests" / "input_manifest.json")}
    pairs = {r["pair_id"]: r for r in read_json(source_root / "manifests" / "input_pairs.json")}
    split = read_json(geometry_root / "protocol.json")["input"]
    val_sources = set(split["validation_sources"])
    rows: list[dict[str, Any]] = []
    missing = []
    for fr in feature_rows:
        wid = fr["window_id"]
        base = old_rows.get(wid)
        if base is None:
            missing.append(wid)
            continue
        pair = pairs.get(base["pair_id"])
        if pair is None:
            missing.append(wid)
            continue
        video_path = pair["real_video_path"] if base["role"] == "real" else pair["fake_video_path"]
        row = dict(base)
        row.update({
            "split": "validation" if base["source_id"] in val_sources else "train",
            "video_path": video_path,
            "video_sha256": sha256_file(video_path) if Path(video_path).exists() else None,
            "v7_feature_row_present": True,
        })
        source_frames = [int(x) for x in base.get("frame_indices", [])]
        source_pts = [float(x) for x in base.get("timestamps_s", [])]
        if not source_frames:
            raise RuntimeError(f"window {wid} has no frame identity")
        positions = np.rint(np.linspace(0, len(source_frames) - 1, FRAME_COUNT)).astype(int)
        row["v8_frame_indices"] = [source_frames[int(i)] for i in positions]
        row["v8_timestamps_s"] = [source_pts[int(i)] if int(i) < len(source_pts) else None for i in positions]
        row["v8_sampling_positions"] = positions.tolist()
        rows.append(row)
    if missing:
        raise RuntimeError(f"window identity mapping missing {len(missing)} rows: {missing[:3]}")
    if len(rows) != 801:
        raise RuntimeError(f"expected 801 window identities from the frozen queue, got {len(rows)}")
    write_json_atomic(out / "data_manifest.json", {"protocol_version": PROTOCOL_VERSION, "rows": rows, "count": len(rows), "source": "window identity only; no V7 feature reuse"})
    return rows


def write_protocol(out: Path, rows: list[dict[str, Any]], geometry_root: Path, moge_checkpoint: str, tracker_checkpoint: str) -> None:
    train = [r for r in rows if r["split"] == "train"]
    val = [r for r in rows if r["split"] == "validation"]
    protocol = {
        "protocol_id": PROTOCOL_VERSION,
        "git_head": os.popen("git rev-parse HEAD").read().strip(),
        "input": {
            "window_count": len(rows), "train_windows": len(train), "validation_windows": len(val),
            "train_sources": sorted({r["source_id"] for r in train}),
            "validation_sources": sorted({r["source_id"] for r in val}),
            "identity_source": str(geometry_root / "inputs" / "feature_manifest.json"),
        },
        "frontend": {
            "canvas_hw": [256, 256], "cell_size": 16, "frames_per_window": FRAME_COUNT,
            "sampling": "16 PTS-sorted target times from frozen window identity; actual decoded PTS saved",
            "geometry": {"provider": "MoGe-2-v2", "checkpoint": moge_checkpoint, "num_tokens": 1200, "apply_mask": False},
            "tracking": {"provider": "CoTracker3", "checkpoint": tracker_checkpoint, "grid_size": 32, "prefix_frames": 3, "endpoint_padding_allowed": True},
            "old_v7_assets_forbidden": True,
        },
        "rgb_feature": {
            "provider": "torchvision ResNet18 ImageNet",
            "weights_url": "https://download.pytorch.org/models/resnet18-f37072fd.pth",
            "weights_path": "/root/.cache/torch/hub/checkpoints/resnet18-f37072fd.pth",
            "feature_stage": "layer1 spatial feature map",
            "feature_dim_per_cell": 64,
            "frozen": True,
            "computed_from": "V8 raw letterboxed RGB frame cache; no V7 feature reuse",
        },
        "grouping": {"base_cells": "16x16 at 256x256", "adjacency": "4-neighbor", "max_cells_per_component": 4, "cost": "equal available spatial/motion terms; training-only robust scales and 25th percentile threshold"},
        "conditions": {
            "FULL": "RGB cell feature + dense XYZ + observation state + dynamic components and causal associations",
            "RGB_2D": "RGB cell feature + 2D short-prefix state; no XYZ/depth/geometry mask/component input",
        },
        "training": {"conditions": ["FULL", "RGB_2D"], "seeds": list(SEEDS), "epochs": 100, "optimizer": "AdamW", "lr": 0.001, "weight_decay": 0.0001, "effective_batch_windows": 8, "clip_norm": 1.0, "threshold": "logit >= 0", "source_isolation": True},
        "evaluation": {"source_macro": "equal source AUROC", "pooled": True, "bootstrap": {"replicates": 10000, "seed": 20260909, "unit": "source"}, "sealed_test": False},
    }
    write_json_atomic(out / "protocol.json", protocol)


def run_plan(out: Path, geometry_root: Path, source_root: Path, moge_checkpoint: str, tracker_checkpoint: str) -> list[dict[str, Any]]:
    out.mkdir(parents=True, exist_ok=True)
    _stage(out, "plan", "RUNNING")
    rows = build_manifest(out, geometry_root, source_root)
    write_protocol(out, rows, geometry_root, moge_checkpoint, tracker_checkpoint)
    _stage(out, "plan", "COMPLETE", total=len(rows), train=sum(r["split"] == "train" for r in rows), validation=sum(r["split"] == "validation" for r in rows))
    return rows


def _failure(out: Path, stage: str, exc: BaseException) -> None:
    _stage(out, stage, "FAILED", error=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc())
    write_json_atomic(out / "final_status.json", {"status": "FAILED", "failed_stage": stage, "error": f"{type(exc).__name__}: {exc}", "updated_unix": _now()})


def run_frontend(out: Path, rows: list[dict[str, Any]], moge_checkpoint: str, tracker_checkpoint: str, device: str = "cuda", limit: int | None = None) -> None:
    _stage(out, "frontend", "RUNNING", planned=len(rows), completed=0, failed=0)
    moge = MogeV2(moge_checkpoint, device=device)
    tracker = CoTrackerShortPrefix(tracker_checkpoint, device=device, grid_size=32)
    done = fail = 0
    selected = rows if limit is None else rows[:limit]
    results = []
    for row in selected:
        try:
            meta = extract_window(row, out, moge, tracker)
            done += 1
            results.append({"window_id": row["window_id"], "status": meta["status"], "npz": meta["npz"]})
        except Exception as exc:
            fail += 1
            results.append({"window_id": row["window_id"], "status": "FAILED", "error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()})
        write_json_atomic(out / "frontend_manifest.json", {"planned": len(selected), "completed": done, "failed": fail, "results": results})
        _stage(out, "frontend", "RUNNING", planned=len(selected), completed=done, failed=fail)
    status = "COMPLETE" if limit is None and done == len(rows) and fail == 0 else ("PARTIAL" if done else "FAILED")
    _stage(out, "frontend", status, planned=len(selected), completed=done, failed=fail, limit=limit)
    if status == "COMPLETE":
        write_json_atomic(out / "frontend_manifest.json", {"planned": len(selected), "completed": done, "failed": fail, "results": results})


def _frontend_npz(out: Path, row: dict[str, Any]) -> Path:
    return out / "frontend" / f"{_safe_name(row['window_id'])}.npz"


def _compute_scales(out: Path, rows: list[dict[str, Any]]) -> dict[str, float]:
    spatial, motion, costs = [], [], []
    edges = cell_neighbors()
    train = [r for r in rows if r["split"] == "train"]
    for row in train:
        p = _frontend_npz(out, row)
        if not p.exists():
            continue
        with np.load(p, allow_pickle=False) as z:
            xyz, gv, mv, mxy = z["cell_xyz"], z["cell_geometry_valid"], z["motion_valid"], z["motion_xy"]
        for t in range(FRAME_COUNT):
            for i, j in edges:
                if gv[t, i] and gv[t, j]:
                    spatial.append(float(np.linalg.norm(xyz[t, i] - xyz[t, j])))
                if mv[t, i] and mv[t, j]:
                    motion.append(float(np.linalg.norm(mxy[t, i] - mxy[t, j])))
    ss = float(np.median(np.asarray([x for x in spatial if x > 1e-8], dtype=np.float64))) if spatial else 1.0
    ms = float(np.median(np.asarray([x for x in motion if x > 1e-8], dtype=np.float64))) if motion else 1.0
    for row in train:
        p = _frontend_npz(out, row)
        if not p.exists():
            continue
        with np.load(p, allow_pickle=False) as z:
            xyz, gv, mv, mxy = z["cell_xyz"], z["cell_geometry_valid"], z["motion_valid"], z["motion_xy"]
        for t in range(FRAME_COUNT):
            for i, j in edges:
                terms = []
                if gv[t, i] and gv[t, j]: terms.append(float(np.linalg.norm(xyz[t, i] - xyz[t, j])) / ss)
                if mv[t, i] and mv[t, j]: terms.append(float(np.linalg.norm(mxy[t, i] - mxy[t, j])) / ms)
                if terms: costs.append(float(np.mean(terms)))
    threshold = float(np.quantile(np.asarray(costs), 0.25)) if costs else 0.0
    return {"spatial_scale": ss, "motion_scale": ms, "merge_threshold": threshold, "spatial_rows": len(spatial), "motion_rows": len(motion), "cost_rows": len(costs)}


class _UnionFind:
    def __init__(self, n: int): self.p = list(range(n)); self.size = [1] * n
    def find(self, x: int) -> int:
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]; x = self.p[x]
        return x
    def union(self, a: int, b: int, max_size: int) -> bool:
        a, b = self.find(a), self.find(b)
        if a == b or self.size[a] + self.size[b] > max_size: return False
        if a > b: a, b = b, a
        self.p[b] = a; self.size[a] += self.size[b]; return True


def _components(xyz: np.ndarray, gv: np.ndarray, mxy: np.ndarray, mv: np.ndarray, scales: dict[str, float]) -> np.ndarray:
    uf = _UnionFind(CELL_COUNT)
    edges = cell_neighbors(); proposals = []
    for i, j in edges:
        terms = []
        if gv[i] and gv[j]: terms.append(float(np.linalg.norm(xyz[i] - xyz[j])) / max(scales["spatial_scale"], 1e-8))
        if mv[i] and mv[j]: terms.append(float(np.linalg.norm(mxy[i] - mxy[j])) / max(scales["motion_scale"], 1e-8))
        if terms: proposals.append((float(np.mean(terms)), int(i), int(j)))
    for cost, i, j in sorted(proposals, key=lambda z: (z[0], z[1], z[2])):
        if cost <= scales["merge_threshold"]: uf.union(i, j, 4)
    roots = [uf.find(i) for i in range(CELL_COUNT)]
    remap = {r: k for k, r in enumerate(sorted(set(roots)))}
    return np.asarray([remap[r] for r in roots], dtype=np.int64)


def _load_frontend_rgb(out: Path, row: dict[str, Any]) -> np.ndarray:
    """Load the exact V8 letterboxed RGB frames saved by the raw frontend."""
    safe = _safe_name(row["window_id"])
    meta_path = out / "frontend" / f"{safe}.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"FRONTEND_METADATA_MISSING:{row['window_id']}")
    meta = read_json(meta_path)
    frames = []
    for item in meta.get("frame_meta", []):
        key = item.get("cache_key")
        if not key:
            raise RuntimeError(f"FRAME_CACHE_KEY_MISSING:{row['window_id']}")
        frame_path = out / "raw_frontend" / "frames" / f"{key}.npz"
        if not frame_path.exists():
            raise FileNotFoundError(f"FRAME_CACHE_MISSING:{row['window_id']}:{key}")
        with np.load(frame_path, allow_pickle=False) as z:
            bgr = z["rgb"]
        if bgr.shape != (256, 256, 3):
            raise ValueError(f"FRAME_CACHE_SHAPE:{row['window_id']}:{bgr.shape}")
        frames.append(bgr[:, :, ::-1].copy())
    if len(frames) != FRAME_COUNT:
        raise ValueError(f"FRAME_COUNT_MISMATCH:{row['window_id']}:{len(frames)}")
    return np.asarray(frames, dtype=np.uint8)


def _build_feature_arrays(
    out: Path,
    row: dict[str, Any],
    scales: dict[str, float],
    rgb_backbone: FrozenResNet18Spatial,
) -> dict[str, np.ndarray]:
    with np.load(_frontend_npz(out, row), allow_pickle=False) as z:
        rgb = z["cell_rgb"].astype(np.float32)
        xyz = z["cell_xyz"].astype(np.float32)
        gv = z["cell_geometry_valid"].astype(bool)
        area = z["cell_area"].astype(np.float32)
        cov = z["track_coverage"].astype(np.float32)
        motion = z["motion_xy"].astype(np.float32)
        mv = z["motion_valid"].astype(bool)
        hist = z["history_available"].astype(bool)
    comps, prev, assoc = [], np.zeros((FRAME_COUNT, CELL_COUNT), np.int64), np.zeros((FRAME_COUNT, CELL_COUNT), np.float32)
    deltas = np.zeros(FRAME_COUNT, np.float32)
    for t in range(FRAME_COUNT):
        comps.append(_components(xyz[t], gv[t], motion[t], mv[t], scales))
        if t > 0:
            d = float(row["timestamps_s"][t] - row["timestamps_s"][t - 1])
            deltas[t] = d if np.isfinite(d) and d >= 0 else 0.0
            for ci in range(CELL_COUNT):
                if hist[t, ci]:
                    cy, cx = divmod(ci, GRID_W)
                    dx, dy = motion[t, ci] / 16.0
                    px, py = int(round(cx - dx)), int(round(cy - dy))
                    px, py = max(0, min(GRID_W - 1, px)), max(0, min(GRID_H - 1, py))
                    prev[t, ci] = py * GRID_W + px
                    assoc[t, ci] = cov[t, ci]
    comps = np.asarray(comps, dtype=np.int64)
    # Frozen ImageNet spatial features are generated from the new frontend's
    # letterboxed RGB frames.  They are shared by FULL and RGB_2D; only the
    # latter's non-RGB channels remain strictly 2-D tracking state.
    rgb_map = rgb_backbone.encode(_load_frontend_rgb(out, row))
    x_full = np.concatenate([rgb_map, rgb, xyz, gv[..., None].astype(np.float32), cov[..., None], motion, mv[..., None].astype(np.float32), hist[..., None].astype(np.float32), (1.0 - gv[..., None].astype(np.float32))], axis=-1)
    x_rgb = np.concatenate([rgb_map, rgb, cov[..., None], motion, mv[..., None].astype(np.float32)], axis=-1)
    return {"x_full": x_full.astype(np.float32), "x_rgb2d": x_rgb.astype(np.float32), "component_ids": comps, "predecessor": prev, "association_weight": assoc, "history_available": hist, "area": area, "delta_t": deltas}


def run_features(out: Path, rows: list[dict[str, Any]], device: str = "cuda") -> None:
    _stage(out, "features", "RUNNING", planned=len(rows), completed=0, failed=0)
    scales = _compute_scales(out, rows)
    write_json_atomic(out / "grouping_config.json", {"protocol_version": PROTOCOL_VERSION, **scales, "max_cells_per_component": 4, "adjacency": "4-neighbor"})
    feature_dir = out / "features"; feature_dir.mkdir(exist_ok=True)
    rgb_backbone = FrozenResNet18Spatial(device=device if device != "cuda" or torch.cuda.is_available() else "cpu")
    done = fail = 0; reasons = []
    for row in rows:
        try:
            if not _frontend_npz(out, row).exists(): raise FileNotFoundError("FRONTEND_MISSING")
            arr = _build_feature_arrays(out, row, scales, rgb_backbone)
            path = feature_dir / f"{_safe_name(row['window_id'])}.npz"
            write_npz_atomic(path, **arr)
            done += 1
        except Exception as exc:
            fail += 1; reasons.append({"window_id": row["window_id"], "error": f"{type(exc).__name__}: {exc}"})
        _stage(out, "features", "RUNNING", planned=len(rows), completed=done, failed=fail)
    write_json_atomic(out / "feature_schema.json", {"version": "v8-feature-v2", "rgb_backbone": "torchvision ResNet18 ImageNet layer1 (frozen)", "rgb_backbone_checkpoint": "/root/.cache/torch/hub/checkpoints/resnet18-f37072fd.pth", "rgb_backbone_feature_dim": 64, "full_dim": 77, "rgb_2d_dim": 71, "shapes": {"x": [FRAME_COUNT, CELL_COUNT, "D"], "component_ids": [FRAME_COUNT, CELL_COUNT], "predecessor": [FRAME_COUNT, CELL_COUNT]}, "mask_semantics": "all non-padding cells retained; geometry validity is state, not row filtering"})
    write_json_atomic(out / "feature_manifest.json", {"planned": len(rows), "completed": done, "failed": fail, "failures": reasons})
    _stage(out, "features", "COMPLETE" if done == len(rows) and fail == 0 else ("PARTIAL" if done else "FAILED"), planned=len(rows), completed=done, failed=fail)


class WindowDataset(torch.utils.data.Dataset):
    def __init__(self, out: Path, rows: list[dict[str, Any]], condition: str, mean: np.ndarray | None = None, scale: np.ndarray | None = None):
        self.out, self.rows, self.condition = out, rows, condition
        self.mean, self.scale = mean, scale
    def __len__(self): return len(self.rows)
    def __getitem__(self, i: int):
        r = self.rows[i]
        with np.load(self.out / "features" / f"{_safe_name(r['window_id'])}.npz", allow_pickle=False) as z:
            x = z["x_full" if self.condition == "FULL" else "x_rgb2d"].astype(np.float32)
            arr = {k: z[k] for k in ["component_ids", "predecessor", "association_weight", "history_available", "area", "delta_t"]}
        if self.mean is not None: x = (x - self.mean) / self.scale
        return {"x": torch.from_numpy(x), "label": torch.tensor(float(r["label"])), "source": r["source_id"], "window_id": r["window_id"], **{k: torch.from_numpy(v) for k, v in arr.items()}}


def _collate(items: list[dict[str, Any]]) -> dict[str, Any]:
    out = {}
    for k in items[0]:
        if k in {"source", "window_id"}: out[k] = [x[k] for x in items]
        elif k == "label": out[k] = torch.stack([x[k] for x in items])
        else: out[k] = torch.stack([x[k] for x in items])
    return out


def _standardizer(out: Path, rows: list[dict[str, Any]], condition: str) -> tuple[np.ndarray, np.ndarray]:
    vals = []
    key = "x_full" if condition == "FULL" else "x_rgb2d"
    for r in rows:
        with np.load(out / "features" / f"{_safe_name(r['window_id'])}.npz", allow_pickle=False) as z:
            vals.append(z[key].reshape(-1, z[key].shape[-1]))
    a = np.concatenate(vals, axis=0).astype(np.float64)
    mean, scale = a.mean(axis=0).astype(np.float32), a.std(axis=0).astype(np.float32)
    scale[scale < 1e-6] = 1.0
    return mean, scale


def _source_class_weights(rows: list[dict[str, Any]]) -> dict[str, float]:
    counts: dict[tuple[str, int], int] = {}
    for r in rows: counts[(r["source_id"], int(r["label"]))] = counts.get((r["source_id"], int(r["label"])), 0) + 1
    raw = {r["window_id"]: 1.0 / counts[(r["source_id"], int(r["label"]))] for r in rows}
    scale = len(raw) / sum(raw.values())
    return {k: v * scale for k, v in raw.items()}


def _predict(model: FullCoverageModel, loader: torch.utils.data.DataLoader, device: str) -> tuple[np.ndarray, list[str], list[str], np.ndarray]:
    model.eval(); scores=[]; labels=[]; ids=[]; sources=[]
    with torch.no_grad():
        for b in loader:
            x=b["x"].to(device); out=model(x,b["component_ids"].to(device),b["predecessor"].to(device),b["association_weight"].to(device),b["history_available"].to(device),b["delta_t"].to(device),b["area"].to(device))
            scores.extend(out.detach().cpu().numpy().tolist()); labels.extend(b["label"].numpy().tolist()); ids.extend(b["window_id"]); sources.extend(b["source"])
    return np.asarray(scores,float), ids, sources, np.asarray(labels,int)


def run_train(out: Path, rows: list[dict[str, Any]], device: str = "cuda") -> None:
    _stage(out, "train", "RUNNING", planned=6, completed=0, failed=0)
    model_dir = out / "models"; model_dir.mkdir(exist_ok=True)
    train_rows = [r for r in rows if r["split"] == "train"]
    val_rows = [r for r in rows if r["split"] == "validation"]
    completed = failed = 0; records=[]
    for condition in ["FULL", "RGB_2D"]:
        mean, scale = _standardizer(out, train_rows, condition)
        write_json_atomic(model_dir / f"standardizer_{condition}.json", {"condition": condition, "mean": mean.tolist(), "scale": scale.tolist(), "fit_windows": len(train_rows)})
        weights = _source_class_weights(train_rows)
        for seed in SEEDS:
            torch.manual_seed(seed); np.random.seed(seed)
            key = "x_full" if condition == "FULL" else "x_rgb2d"
            with np.load(out / "features" / f"{_safe_name(train_rows[0]['window_id'])}.npz", allow_pickle=False) as z:
                input_dim = int(z[key].shape[-1])
            model = FullCoverageModel(condition, input_dim).to(device)
            opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
            ds = WindowDataset(out, train_rows, condition, mean, scale)
            val_ds = WindowDataset(out, val_rows, condition, mean, scale)
            gen = torch.Generator().manual_seed(seed)
            loader = torch.utils.data.DataLoader(ds, batch_size=8, shuffle=True, generator=gen, collate_fn=_collate, num_workers=0)
            vloader = torch.utils.data.DataLoader(val_ds, batch_size=8, shuffle=False, collate_fn=_collate, num_workers=0)
            history=[]; started=_now()
            for epoch in range(100):
                model.train(); losses=[]
                for b in loader:
                    opt.zero_grad(set_to_none=True)
                    z=model(b["x"].to(device),b["component_ids"].to(device),b["predecessor"].to(device),b["association_weight"].to(device),b["history_available"].to(device),b["delta_t"].to(device),b["area"].to(device))
                    w=torch.as_tensor([weights[i] for i in b["window_id"]],device=device,dtype=z.dtype)
                    loss=torch.nn.functional.binary_cross_entropy_with_logits(z,b["label"].to(device),weight=w)
                    if not torch.isfinite(loss): raise FloatingPointError(f"nonfinite loss {condition} seed {seed} epoch {epoch}")
                    loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step(); losses.append(float(loss.detach().cpu()))
                history.append(float(np.mean(losses)))
            ckpt=model_dir / f"{condition}__seed{seed}.pt"
            torch.save({"condition":condition,"seed":seed,"input_dim":input_dim,"model":model.state_dict(),"optimizer":opt.state_dict(),"epoch":100,"mean":mean,"scale":scale,"parameter_count":parameter_count(model),"loss_history":history},ckpt)
            train_scores, train_ids, train_sources, train_labels = _predict(model, torch.utils.data.DataLoader(ds,batch_size=8,shuffle=False,collate_fn=_collate), device)
            val_scores, val_ids, val_sources, val_labels = _predict(model, vloader, device)
            np.savez_compressed(model_dir / f"scores__{condition}__seed{seed}.npz", train_scores=train_scores, train_labels=train_labels, val_scores=val_scores, val_labels=val_labels)
            records.append({"condition":condition,"seed":seed,"status":"COMPLETE","epoch":100,"parameter_count":parameter_count(model),"trainable_parameter_count":sum(p.numel() for p in model.parameters() if p.requires_grad),"rgb_backbone":"precomputed frozen ResNet18 layer1","input_dim":input_dim,"train_windows":len(train_rows),"validation_windows":len(val_rows),"elapsed_s":_now()-started,"checkpoint":str(ckpt),"loss_first":history[0],"loss_last":history[-1]})
            completed += 1
            write_json_atomic(model_dir / "fold_models.json", {"planned":6,"completed":completed,"failed":failed,"records":records})
            _stage(out,"train","RUNNING",planned=6,completed=completed,failed=failed)
    _stage(out,"train","COMPLETE" if completed == 6 else "PARTIAL",planned=6,completed=completed,failed=failed)


def _metric(y: np.ndarray, z: np.ndarray) -> dict[str, Any]:
    from sklearn.metrics import average_precision_score, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
    pred = z >= 0
    out = {"n": int(len(y)), "real": int((y == 0).sum()), "fake": int((y == 1).sum())}
    out["auroc"] = float(roc_auc_score(y,z)) if len(np.unique(y))==2 else None
    out["ap"] = float(average_precision_score(y,z)) if len(np.unique(y))==2 else None
    out["precision"] = float(precision_score(y,pred,zero_division=0)) if pred.any() else None
    out["recall"] = float(recall_score(y,pred,zero_division=0))
    out["f1"] = float(f1_score(y,pred,zero_division=0)) if pred.any() else None
    out["accuracy"] = float(accuracy_score(y,pred))
    tn,fp,fn,tp=confusion_matrix(y,pred,labels=[0,1]).ravel().tolist(); out.update(tn=tn,fp=fp,fn=fn,tp=tp)
    return out


def _source_macro(rows: list[dict[str, Any]], scores: np.ndarray, labels: np.ndarray, ids: list[str]) -> dict[str, Any]:
    from sklearn.metrics import roc_auc_score
    by: dict[str, list[int]] = {}
    for i,r in enumerate(rows): by.setdefault(r["source_id"],[]).append(i)
    vals={s:float(roc_auc_score(labels[ix],scores[ix])) for s,ix in by.items() if len(set(labels[ix]))==2}
    return {"source_count":len(vals),"mean":float(np.mean(list(vals.values()))) if vals else None,"per_source":vals}


def _bootstrap_diff(a: dict[str,float], b: dict[str,float], reps: int=10000, seed: int=20260909) -> dict[str, Any]:
    src=sorted(set(a)&set(b)); rng=np.random.default_rng(seed); dif=np.asarray([a[s]-b[s] for s in src],float)
    if not len(dif): return {"mean":None,"ci95":[None,None],"source_count":0}
    samples=rng.integers(0,len(dif),size=(reps,len(dif))); boots=dif[samples].mean(axis=1)
    return {"mean":float(dif.mean()),"ci95":[float(np.quantile(boots,.025)),float(np.quantile(boots,.975))],"source_count":len(src),"per_source":{s:float(a[s]-b[s]) for s in src}}


def run_evaluate(out: Path, rows: list[dict[str, Any]], device: str = "cuda") -> dict[str, Any]:
    _stage(out,"evaluate","RUNNING")
    val_rows=[r for r in rows if r["split"]=="validation"]
    all_scores=[]; metrics=[]; per_source=[]
    for condition in ["FULL","RGB_2D"]:
        m=read_json(out / "models" / f"standardizer_{condition}.json"); mean=np.asarray(m["mean"],np.float32); scale=np.asarray(m["scale"],np.float32)
        ds=WindowDataset(out,val_rows,condition,mean,scale); loader=torch.utils.data.DataLoader(ds,batch_size=8,shuffle=False,collate_fn=_collate)
        seed_scores=[]
        for seed in SEEDS:
            rec=torch.load(out / "models" / f"{condition}__seed{seed}.pt",map_location=device,weights_only=False)
            model=FullCoverageModel(condition,int(rec["input_dim"])).to(device); model.load_state_dict(rec["model"])
            z,ids,sources,y=_predict(model,loader,device); seed_scores.append(z)
            mm=_metric(y,z); sm=_source_macro(val_rows,z,y,ids); metrics.append({"condition":condition,"seed":seed,"split":"validation",**mm,"source_macro_auroc":sm["mean"],"source_count":sm["source_count"]})
        avg=np.mean(np.stack(seed_scores),axis=0); mm=_metric(y,avg); sm=_source_macro(val_rows,avg,y,ids); metrics.append({"condition":condition,"seed":"MEAN_LOGIT","split":"validation",**mm,"source_macro_auroc":sm["mean"],"source_count":sm["source_count"]})
        all_scores.append((condition,avg,y,ids,sm["per_source"]))
    base=all_scores[0][4]; ctrl=all_scores[1][4]
    comparison=_bootstrap_diff(base,ctrl)
    summary={"conditions":metrics,"comparison_FULL_minus_RGB_2D":comparison,"validation_windows":len(val_rows),"validation_sources":sorted({r["source_id"] for r in val_rows})}
    write_json_atomic(out / "evaluation" / "summary.json",summary)
    (out/"evaluation").mkdir(exist_ok=True)
    with open(out/"evaluation"/"metrics.csv","w",newline="",encoding="utf-8") as f:
        fields=sorted({k for r in metrics for k in r}); w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(metrics)
    with open(out/"evaluation"/"per_source_differences.csv","w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=["source_id","full_auroc","rgb2d_auroc","difference"]); w.writeheader()
        for s in sorted(set(base)&set(ctrl)): w.writerow({"source_id":s,"full_auroc":base[s],"rgb2d_auroc":ctrl[s],"difference":base[s]-ctrl[s]})
    with open(out/"scores/validation_window_scores.csv","w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=["window_id","source_id","role","label","full_logit","rgb_2d_logit"]); w.writeheader()
        for i,r in enumerate(val_rows): w.writerow({"window_id":r["window_id"],"source_id":r["source_id"],"role":r["role"],"label":r["label"],"full_logit":all_scores[0][1][i],"rgb_2d_logit":all_scores[1][1][i]})
    _stage(out,"evaluate","COMPLETE",validation_windows=len(val_rows),source_count=summary["conditions"][-1]["source_count"])
    return summary


def run_visualizations(out: Path, rows: list[dict[str, Any]], device: str = "cuda") -> None:
    val=[r for r in rows if r["split"]=="validation"]
    vdir=out/"visualizations"; vdir.mkdir(exist_ok=True)
    for r in val:
        if r["role"] != "fake": continue
        if len(list(vdir.glob("*.png"))) >= 8: break
        with np.load(out/"features"/f"{_safe_name(r['window_id'])}.npz",allow_pickle=False) as z: comp=z["component_ids"][-1].reshape(GRID_H,GRID_W)
        with np.load(_frontend_npz(out, r), allow_pickle=False) as z: geo=z["cell_geometry_valid"][-1].reshape(GRID_H,GRID_W)
        rgb=np.zeros((GRID_H,GRID_W,3),dtype=np.uint8); rgb[...,0]=np.where(geo>0,80,200); rgb[...,1]=np.where(geo>0,190,100); rgb[...,2]=60
        Image.fromarray(rgb).resize((256,256),Image.Resampling.NEAREST).save(vdir/f"{_safe_name(r['window_id'])}.png")
    write_json_atomic(vdir/"manifest.json", {"status":"COMPLETE","note":"Grid-level diagnostic maps; color is state/component aid, not a pixel GT."})


def run_report(out: Path, rows: list[dict[str, Any]]) -> None:
    status=_load_status(out); eval_path=out/"evaluation"/"summary.json"; summary=read_json(eval_path) if eval_path.exists() else {}
    lines=["# V8 Full-Coverage 3D Observation Field v1", "", f"- protocol: `{PROTOCOL_VERSION}`", f"- branch: `v8-full-coverage-observation-field`", f"- windows: {len(rows)} (train={sum(r['split']=='train' for r in rows)}, validation={sum(r['split']=='validation' for r in rows)})", "- queue: reused only window identity and raw-video paths; V7 frontend/features/models were not read.", "", "## Status", "", "```json", json.dumps(status,ensure_ascii=False,indent=2), "```", "", "## Evaluation", "", "```json", json.dumps(summary,ensure_ascii=False,indent=2), "```", "", "## Interpretation boundary", "", "V8 的全覆盖指 256×256 处理画面中的非 padding 基础格始终有输出；它不是原始像素级真值。MoGe/CoTracker 的几何和对应质量未在本实验中证明，缺失状态不是伪造标签。FULL−RGB_2D 同时改变了几何观测、动态分组与观测状态，不能单独归因于 XYZ。", ""]
    (out/"report.md").write_text("\n".join(lines),encoding="utf-8")
    stages = status.get("stages", {})
    required = ["plan", "frontend", "features", "train", "evaluate", "report"]
    complete = all(stages.get(k, {}).get("status") == "COMPLETE" for k in required[:-1])
    final = "COMPLETE" if complete else "PARTIAL"
    _stage(out, "report", "COMPLETE", stages_complete=complete)
    write_json_atomic(out/"final_status.json", {"status":final,"protocol_version":PROTOCOL_VERSION,"updated_unix":_now(),"note":"COMPLETE means pipeline stages completed; it does not establish the scientific hypothesis."})


def run_all(args: Any) -> None:
    out=Path(args.output); geometry_root=Path(args.identity_root); source_root=Path(args.source_root)
    out.mkdir(parents=True,exist_ok=True)
    rows = manifest_rows(out/"data_manifest.json") if (out/"data_manifest.json").exists() else run_plan(out,geometry_root,source_root,args.moge_checkpoint,args.tracker_checkpoint)
    try:
        # Refresh the protocol metadata after code changes while preserving the
        # frozen identity rows.  This does not regenerate or alter the plan.
        write_protocol(out, rows, geometry_root, args.moge_checkpoint, args.tracker_checkpoint)
        if args.stage in {"all","frontend"}:
            frontend_state = _load_status(out).get("stages", {}).get("frontend", {})
            # A two-window smoke is intentionally not a formal completion mark.
            if not (frontend_state.get("status") == "COMPLETE" and int(frontend_state.get("planned", -1)) == len(rows)):
                run_frontend(out,rows,args.moge_checkpoint,args.tracker_checkpoint,args.device,args.limit_windows)
        if args.stage in {"all","features"}:
            if not (_load_status(out).get("stages",{}).get("features",{}).get("status")=="COMPLETE"):
                run_features(out,rows,args.device)
        if args.stage in {"all","train"}:
            if not (_load_status(out).get("stages",{}).get("train",{}).get("status")=="COMPLETE"):
                run_train(out,rows,args.device)
        if args.stage in {"all","evaluate"}:
            if not (_load_status(out).get("stages",{}).get("evaluate",{}).get("status")=="COMPLETE"):
                run_evaluate(out,rows,args.device)
        if args.stage in {"all","report"}:
            run_visualizations(out,rows,args.device); run_report(out,rows)
    except Exception as exc:
        _failure(out,args.stage,exc); raise
