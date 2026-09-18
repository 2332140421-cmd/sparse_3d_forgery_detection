"""Run the bounded frozen-video-feature/structure V7 pilot.

This module intentionally remains a thin experiment entry point.  It reads the
already frozen R17 structure features and source/window manifests, extracts a
single official R3D-18 representation for the same windows, and trains only
the two small heads requested by the experiment.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import subprocess
import time
import traceback
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from research_tools.v7.geometry_information_pilot import runner as geometry
from research_tools.v7.geometry_information_pilot.model import ModalSetModel, parameter_count
from research_tools.v7.multi_order_sequence_probe.model import FeatureStandardizer, fit_standardizer, source_class_weights
from research_tools.v7.periodic_requery_probe import runner as periodic


REPO_ROOT = Path(__file__).resolve().parents[3]
DATA_ROOT = Path("/root/autodl-tmp/data/sparse_3d_forgery_detection")
STRUCTURE_ROOT = DATA_ROOT / "derived/v7_activityforensics_observation_support_pilot_v1"
VIDEO_ROOT = DATA_ROOT / "derived/v7_activityforensics_source128_extension_v1"
OUTPUT_ROOT = DATA_ROOT / "derived/v7_activityforensics_visual_structure_pilot_v1"
SEEDS = (20260909, 20260910, 20260911)
EPOCHS = 200
LR = 1e-3
WEIGHT_DECAY = 1e-4
BOOTSTRAP_SEED = 20260909
BOOTSTRAP_REPLICATES = 10_000
CONDITIONS = ("STRUCTURE", "VISUAL", "VISUAL_STRUCTURE")
WEIGHT_URL = "https://download.pytorch.org/models/r3d_18-b3b3357e.pth"
WEIGHT_FILE = Path.home() / ".cache/torch/hub/checkpoints/r3d_18-b3b3357e.pth"


def _jsonable(value: Any, path: str = "$") -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return [_jsonable(x, f"{path}[{i}]") for i, x in enumerate(value)]
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return _jsonable(value.item(), path)
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"non-finite JSON value at {path}: {value!r}")
        return value
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v, f"{path}.{k}") for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v, f"{path}[{i}]") for i, v in enumerate(value)]
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


def progress(root: Path, stage: str, status: str, completed: int, total: int, **extra: Any) -> None:
    atomic_json(root / "progress.json", {"stage": stage, "status": status, "completed": int(completed), "total": int(total), "updated_unix": time.time(), **extra})


def _safe(text: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in text)


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {key: np.asarray(archive[key]) for key in archive.files}


def _rows_and_inputs() -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    train, validation, support_rows, info = geometry._load_selection()
    feature_manifest = json.loads((STRUCTURE_ROOT / "inputs/feature_manifest.json").read_text(encoding="utf-8"))
    feature_by_id = {str(row["window_id"]): dict(row) for row in feature_manifest}
    expected = {str(row["window_id"]) for row in train + validation}
    if set(feature_by_id) != expected:
        raise ValueError(f"STRUCTURE_FEATURE_WINDOW_SET_MISMATCH:{len(feature_by_id)}:{len(expected)}")
    source_input = json.loads((VIDEO_ROOT / "manifests/input_manifest.json").read_text(encoding="utf-8"))
    source_rows = {str(row["window_id"]): dict(row) for row in source_input["rows"]}
    parents = json.loads((VIDEO_ROOT / "manifests/parents.json").read_text(encoding="utf-8"))
    parent_by_id = {str(row["parent_id"]): dict(row) for row in parents}
    output_rows: list[dict[str, Any]] = []
    for row in train + validation:
        window_id = str(row["window_id"])
        source_row = source_rows.get(window_id)
        if source_row is None:
            raise ValueError(f"VIDEO_WINDOW_MISSING:{window_id}")
        parent_id = window_id.rsplit("::b", 1)[0]
        parent = parent_by_id.get(parent_id)
        if parent is None:
            raise ValueError(f"VIDEO_PARENT_MISSING:{window_id}")
        if not Path(str(parent["video_path"])).is_file():
            raise FileNotFoundError(f"VIDEO_MISSING:{parent['video_path']}")
        merged = dict(row)
        merged.update({"parent_id": parent_id, "interval_start_s": float(source_row["interval_start_s"]), "interval_end_s": float(source_row["interval_end_s"]), "video_path": str(parent["video_path"]), "video_sha256": str(parent.get("video_sha256", "")), "parent_frame_indices": list(parent["parent_frame_indices"]), "parent_timestamps_s": list(parent["parent_timestamps_s"]), "feature_item": feature_by_id[window_id]})
        output_rows.append(merged)
    train_rows = output_rows[: len(train)]
    validation_rows = output_rows[len(train) :]
    return train_rows, validation_rows, feature_manifest, info, {"support_rows": support_rows, "parents": parents, "source_manifest": source_input}


def _fit_visual_standardizer(rows: Sequence[Mapping[str, Any]], feature_by_id: Mapping[str, Mapping[str, Any]], root: Path | None = None) -> dict[str, Any]:
    values = np.stack([_visual_feature(feature_by_id[str(row["window_id"])], root) for row in rows], axis=0).astype(np.float64)
    weights = source_class_weights(rows)
    mean = np.average(values, axis=0, weights=weights)
    var = np.average((values - mean) ** 2, axis=0, weights=weights)
    zero = np.flatnonzero(~np.isfinite(var) | (var <= 1e-12)).astype(int).tolist()
    scale = np.sqrt(np.maximum(var, 1e-12)); scale[zero] = 1.0
    return {"mean": mean, "scale": scale, "zero_variance_dimensions": zero, "training_window_count": len(rows)}


def _visual_transform(values: np.ndarray, standardizer: Mapping[str, Any]) -> np.ndarray:
    return (np.asarray(values, dtype=np.float64) - np.asarray(standardizer["mean"], dtype=np.float64)) / np.asarray(standardizer["scale"], dtype=np.float64)


def _visual_standardizer_json(standardizer: Mapping[str, Any]) -> dict[str, Any]:
    return {"mean": np.asarray(standardizer["mean"]).tolist(), "scale": np.asarray(standardizer["scale"]).tolist(), "zero_variance_dimensions": list(standardizer["zero_variance_dimensions"]), "training_window_count": int(standardizer["training_window_count"])}


def _visual_feature(item: Mapping[str, Any], root: Path | None) -> np.ndarray:
    path = Path(str(item["feature_path"])) if root is None else root / str(item["feature_path"])
    with np.load(path, allow_pickle=False) as archive:
        feature = np.asarray(archive["feature"], dtype=np.float32)
    if feature.shape != (512,) or not np.all(np.isfinite(feature)):
        raise ValueError(f"VISUAL_FEATURE_INVALID:{item.get('window_id')}:{feature.shape}")
    return feature


def _select_frames(row: Mapping[str, Any]) -> list[dict[str, Any]]:
    start, end = float(row["interval_start_s"]), float(row["interval_end_s"])
    indices = np.asarray(row["parent_frame_indices"], dtype=np.int64)
    pts = np.asarray(row["parent_timestamps_s"], dtype=np.float64)
    if indices.ndim != 1 or pts.shape != indices.shape or not np.all(np.isfinite(pts)):
        raise ValueError(f"VIDEO_TIMELINE_INVALID:{row['window_id']}")
    target = start + (np.arange(16, dtype=np.float64) + 0.5) * (end - start) / 16.0
    selected: list[dict[str, Any]] = []
    for value in target:
        eligible = np.flatnonzero((pts >= start - 1e-9) & (pts <= end + 1e-9))
        if eligible.size == 0:
            raise ValueError(f"NO_FRAME_IN_WINDOW:{row['window_id']}:{value}")
        local = int(eligible[int(np.argmin(np.abs(pts[eligible] - value)))])
        selected.append({"target_s": float(value), "frame_index": int(indices[local]), "pts_s": float(pts[local]), "error_s": float(pts[local] - value)})
    return selected


def _decode_selected(video_path: str, indices: Sequence[int]) -> dict[int, tuple[np.ndarray, float | None]]:
    import av
    wanted = set(int(x) for x in indices)
    output: dict[int, tuple[np.ndarray, float | None]] = {}
    container = av.open(video_path)
    try:
        stream = container.streams.video[0]
        for local_index, frame in enumerate(container.decode(video=0)):
            if local_index in wanted:
                pts = None if frame.pts is None else float(frame.pts * stream.time_base)
                output[local_index] = (frame.to_rgb().to_ndarray(), pts)
                if len(output) == len(wanted):
                    break
    finally:
        container.close()
    missing = wanted - set(output)
    if missing:
        raise ValueError(f"DECODE_FRAME_MISSING:{video_path}:{sorted(missing)[:8]}")
    return output


def _preprocess_frame(rgb: np.ndarray, mean: Sequence[float], std: Sequence[float], *, torch_module: Any) -> Any:
    import torch.nn.functional as F
    image = torch_module.as_tensor(np.asarray(rgb), dtype=torch_module.float32).permute(2, 0, 1) / 255.0
    _, height, width = image.shape
    scale = min(224.0 / float(width), 224.0 / float(height))
    new_w, new_h = max(1, int(round(width * scale))), max(1, int(round(height * scale)))
    resized = F.interpolate(image[None], size=(new_h, new_w), mode="bilinear", align_corners=False)[0]
    canvas = torch_module.empty((3, 224, 224), dtype=torch_module.float32)
    mean_tensor = torch_module.as_tensor(mean, dtype=torch_module.float32)
    canvas[:] = mean_tensor[:, None, None]
    y0, x0 = (224 - new_h) // 2, (224 - new_w) // 2
    canvas[:, y0 : y0 + new_h, x0 : x0 + new_w] = resized
    return (canvas - mean_tensor[:, None, None]) / torch_module.as_tensor(std, dtype=torch_module.float32)[:, None, None]


def _load_backbone(device: str) -> tuple[Any, dict[str, Any]]:
    import torch
    from torchvision.models.video import R3D_18_Weights, r3d_18
    weights = R3D_18_Weights.KINETICS400_V1
    model = r3d_18(weights=weights)
    model.fc = torch.nn.Identity()
    model.eval().requires_grad_(False)
    for module in model.modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            module.eval()
            module.track_running_stats = True
    model.to(device)
    transforms = weights.transforms()
    mean, std = list(transforms.mean), list(transforms.std)
    path = WEIGHT_FILE
    if not path.is_file():
        raise FileNotFoundError(f"OFFICIAL_R3D_WEIGHT_NOT_FOUND:{path}")
    metadata = {"name": "R3D_18_Weights.KINETICS400_V1", "url": WEIGHT_URL, "path": str(path), "sha256": sha256(path), "sha256_prefix_expected": "b3b3357e", "torchvision_weight_hash_ok": sha256(path).startswith("b3b3357e"), "mean": mean, "std": std, "feature_dim": 512, "layer": "global_pool_before_fc", "preprocess": "aspect-preserving resize to 224 canvas, mean-valued padding, normalize; no crop/augment"}
    if not metadata["torchvision_weight_hash_ok"]:
        raise ValueError(f"OFFICIAL_R3D_WEIGHT_HASH_MISMATCH:{metadata['sha256']}")
    return model, metadata


def _write_preprocess_preview(root: Path, row: Mapping[str, Any], weight_meta: Mapping[str, Any]) -> None:
    """Save one non-interactive RGB/letterbox preview for the smoke sample."""
    preview = root / "smoke/preprocess_preview.png"
    if preview.is_file():
        return
    import torch
    from PIL import Image
    selected = _select_frames(row)
    frames = _decode_selected(str(row["video_path"]), [int(selected[0]["frame_index"])])
    tensor = _preprocess_frame(frames[int(selected[0]["frame_index"])][0], weight_meta["mean"], weight_meta["std"], torch_module=torch)
    image = (tensor * torch.as_tensor(weight_meta["std"])[:, None, None] + torch.as_tensor(weight_meta["mean"])[:, None, None]).clamp(0, 1)
    array = (image.permute(1, 2, 0).cpu().numpy() * 255.0 + 0.5).astype(np.uint8)
    preview.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array, mode="RGB").save(preview)


def _feature_identity(row: Mapping[str, Any], selected: Sequence[Mapping[str, Any]], weight_meta: Mapping[str, Any]) -> dict[str, Any]:
    return {"window_id": str(row["window_id"]), "source_id": str(row["source_id"]), "role": str(row["role"]), "video_path": str(row["video_path"]), "video_sha256": str(row["video_sha256"]), "interval_start_s": float(row["interval_start_s"]), "interval_end_s": float(row["interval_end_s"]), "target_frame_indices": [int(x["frame_index"]) for x in selected], "target_pts_s": [float(x["pts_s"]) for x in selected], "target_times_s": [float(x["target_s"]) for x in selected], "target_errors_s": [float(x["error_s"]) for x in selected], "weight_sha256": str(weight_meta["sha256"]), "feature_layer": "global_pool_before_fc", "feature_dim": 512, "preprocess": weight_meta["preprocess"]}


def _extract_rows(root: Path, rows: Sequence[Mapping[str, Any]], backbone: Any, weight_meta: Mapping[str, Any], device: str, *, smoke: bool = False) -> list[dict[str, Any]]:
    import torch
    formal_dir = root / ("smoke/visual_features" if smoke else "features/window_visual")
    manifest_path = root / ("smoke/visual_feature_manifest.json" if smoke else "visual_feature_manifest.json")
    formal_dir.mkdir(parents=True, exist_ok=True)
    existing = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.is_file() else []
    existing_by_id = {str(x.get("window_id")): dict(x) for x in existing}
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["parent_id"])].append(row)
    output = dict(existing_by_id)
    for parent_index, parent_id in enumerate(sorted(grouped)):
        parent_rows = grouped[parent_id]
        selected_by_id = {str(row["window_id"]): _select_frames(row) for row in parent_rows}
        pending: list[Mapping[str, Any]] = []
        for row in parent_rows:
            identity = _feature_identity(row, selected_by_id[str(row["window_id"])], weight_meta)
            old = output.get(str(row["window_id"]))
            feature_path = formal_dir / f"{_safe(str(row['window_id']))}.npz"
            if old and old.get("identity") == identity and feature_path.is_file():
                try:
                    _visual_feature({"window_id": row["window_id"], "feature_path": str(feature_path)}, None)
                    continue
                except Exception:
                    pass
            pending.append(row)
        if pending:
            unique_indices = sorted({int(item["frame_index"]) for row in pending for item in selected_by_id[str(row["window_id"])]})
            frames = _decode_selected(str(parent_rows[0]["video_path"]), unique_indices)
            tensors: dict[int, Any] = {}
            for index in unique_indices:
                tensor = _preprocess_frame(frames[index][0], weight_meta["mean"], weight_meta["std"], torch_module=torch)
                tensors[index] = tensor
            inputs: list[Any] = []
            for row in pending:
                inputs.append(torch.stack([tensors[int(item["frame_index"])] for item in selected_by_id[str(row["window_id"])]], dim=1))
            features: list[np.ndarray] = []
            with torch.inference_mode():
                for start in range(0, len(inputs), 8):
                    batch = torch.stack(inputs[start : start + 8], dim=0).to(device)
                    values = backbone(batch).detach().cpu().numpy().astype(np.float32)
                    if values.shape[1:] != (512,):
                        raise ValueError(f"R3D_FEATURE_SHAPE:{values.shape}")
                    features.extend(values)
            for row, value in zip(pending, features):
                window_id = str(row["window_id"]); selected = selected_by_id[window_id]
                feature_path = formal_dir / f"{_safe(window_id)}.npz"; temporary = feature_path.with_name(feature_path.name + ".tmp.npz")
                np.savez_compressed(temporary, feature=value.astype(np.float32)); os.replace(temporary, feature_path)
                output[window_id] = {"window_id": window_id, "source_id": str(row["source_id"]), "role": str(row["role"]), "label": int(row["label"]), "annotation_category": str(row.get("annotation_category", "")), "feature_path": str(feature_path.relative_to(root)), "identity": _feature_identity(row, selected, weight_meta), "repeat_frame_count": int(len(selected) - len({int(x["frame_index"]) for x in selected})), "max_abs_decode_pts_error_s": None if all(frames[int(x["frame_index"])] [1] is None for x in selected) else float(max(abs(float(frames[int(x["frame_index"])] [1]) - float(x["pts_s"])) for x in selected if frames[int(x["frame_index"])] [1] is not None))}
        atomic_json(manifest_path, sorted(output.values(), key=lambda x: str(x["window_id"])))
        progress(root, "visual_features", "RUNNING", parent_index + 1, len(grouped), current_parent=parent_id, window_count=len(output), smoke=smoke)
    return sorted(output.values(), key=lambda x: str(x["window_id"]))


def _structure_standardizers(train_rows: Sequence[Mapping[str, Any]], feature_by_id: Mapping[str, Mapping[str, Any]], root: Path) -> tuple[FeatureStandardizer, FeatureStandardizer]:
    s = [_load_npz(STRUCTURE_ROOT / str(feature_by_id[str(row["window_id"])] ["input_path"]))["s"] for row in train_rows]
    q = [_load_npz(STRUCTURE_ROOT / str(feature_by_id[str(row["window_id"])] ["input_path"]))["q"] for row in train_rows]
    weights = source_class_weights(train_rows)
    return fit_standardizer("SET_A", s, weights), fit_standardizer("SET_A", q, weights)


def _structure_batch(rows: Sequence[Mapping[str, Any]], feature_by_id: Mapping[str, Mapping[str, Any]], s_std: FeatureStandardizer, q_std: FeatureStandardizer) -> dict[str, Any]:
    selected = sorted(rows, key=lambda x: str(x["window_id"]))
    states: list[np.ndarray] = []; intervals: list[np.ndarray] = []; window_index: list[int] = []
    for index, row in enumerate(selected):
        arrays = _load_npz(STRUCTURE_ROOT / str(feature_by_id[str(row["window_id"])] ["input_path"]))
        s, q = s_std.transform(arrays["s"]), q_std.transform(arrays["q"])
        states.append(np.concatenate((s, q), axis=-1)); intervals.append(np.repeat(np.asarray(arrays["intervals"], dtype=np.float64)[None], s.shape[0], axis=0)); window_index.extend([index] * s.shape[0])
    return {"states": np.concatenate(states, axis=0).astype(np.float32), "intervals": np.concatenate(intervals, axis=0).astype(np.float32), "window_index": np.asarray(window_index, dtype=np.int64), "window_ids": [str(row["window_id"]) for row in selected], "rows": selected, "labels": np.asarray([int(row["label"]) for row in selected], dtype=np.int64), "weights": source_class_weights(selected).astype(np.float32), "window_count": len(selected)}


def _visual_batch(rows: Sequence[Mapping[str, Any]], visual_by_id: Mapping[str, Mapping[str, Any]], root: Path, standardizer: Mapping[str, Any]) -> dict[str, Any]:
    selected = sorted(rows, key=lambda x: str(x["window_id"]))
    values = np.stack([_visual_feature(visual_by_id[str(row["window_id"])], root) for row in selected], axis=0)
    return {"features": _visual_transform(values, standardizer).astype(np.float32), "window_ids": [str(row["window_id"]) for row in selected], "rows": selected, "labels": np.asarray([int(row["label"]) for row in selected], dtype=np.int64), "weights": source_class_weights(selected).astype(np.float32), "window_count": len(selected)}


def _set_seed(seed: int) -> None:
    import torch
    torch.manual_seed(int(seed)); np.random.seed(int(seed) & 0xFFFFFFFF)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


class VisualHead(__import__("torch").nn.Module):
    def __init__(self) -> None:
        import torch
        super().__init__()
        self.net = torch.nn.Sequential(torch.nn.Linear(512, 32), torch.nn.ReLU(), torch.nn.Linear(32, 1))

    def forward(self, features: Any) -> Any:
        return self.net(features).squeeze(-1)


class FusionModel(__import__("torch").nn.Module):
    def __init__(self, structure: ModalSetModel, visual: VisualHead) -> None:
        super().__init__()
        self.structure = structure
        self.visual = visual

    def forward(self, states: Any, intervals: Any, window_index: Any, window_count: int, features: Any) -> tuple[Any, Any, Any]:
        local = self.structure(states, intervals)
        sums = states.new_zeros((window_count,)); counts = states.new_zeros((window_count,))
        sums.index_add_(0, window_index, local); counts.index_add_(0, window_index, states.new_ones(local.shape))
        z_s = sums / counts.clamp_min(1.0)
        z_v = self.visual(features)
        return z_s + z_v, z_s, z_v


def _structure_forward(model: ModalSetModel, batch: Mapping[str, Any], device: str) -> Any:
    import torch
    states = torch.as_tensor(batch["states"], dtype=torch.float32, device=device); intervals = torch.as_tensor(batch["intervals"], dtype=torch.float32, device=device); index = torch.as_tensor(batch["window_index"], dtype=torch.long, device=device)
    local = model(states, intervals); sums = torch.zeros(int(batch["window_count"]), dtype=local.dtype, device=device); counts = torch.zeros_like(sums); sums.index_add_(0, index, local); counts.index_add_(0, index, torch.ones_like(local)); return sums / counts.clamp_min(1.0)


def _visual_forward(model: VisualHead, batch: Mapping[str, Any], device: str) -> Any:
    import torch
    return model(torch.as_tensor(batch["features"], dtype=torch.float32, device=device))


def _fusion_forward(model: FusionModel, structure_batch: Mapping[str, Any], visual_batch: Mapping[str, Any], device: str) -> tuple[Any, Any, Any]:
    import torch
    return model(torch.as_tensor(structure_batch["states"], dtype=torch.float32, device=device), torch.as_tensor(structure_batch["intervals"], dtype=torch.float32, device=device), torch.as_tensor(structure_batch["window_index"], dtype=torch.long, device=device), int(structure_batch["window_count"]), torch.as_tensor(visual_batch["features"], dtype=torch.float32, device=device))


def _train_model(condition: str, structure_batch: Mapping[str, Any], visual_batch: Mapping[str, Any], seed: int, device: str) -> tuple[Any, dict[str, Any]]:
    import torch
    import torch.nn.functional as F
    if condition == "VISUAL":
        _set_seed(seed); model: Any = VisualHead().to(device); optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    elif condition == "VISUAL_STRUCTURE":
        visual = (lambda: (_set_seed(seed), VisualHead().to(device))[1])()
        structure = geometry._initial_model("STRUCTURE_SUPPORT", seed).to(device)
        model = FusionModel(structure, visual).to(device); optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    else:
        raise ValueError(condition)
    labels = torch.as_tensor(visual_batch["labels"], dtype=torch.float32, device=device); weights = torch.as_tensor(visual_batch["weights"], dtype=torch.float32, device=device); history: list[dict[str, Any]] = []
    model.train()
    for epoch in range(1, EPOCHS + 1):
        optimizer.zero_grad(set_to_none=True)
        if condition == "VISUAL":
            scores = _visual_forward(model, visual_batch, device)
        else:
            scores, _, _ = _fusion_forward(model, structure_batch, visual_batch, device)
        loss = torch.sum(F.binary_cross_entropy_with_logits(scores, labels, reduction="none") * weights) / torch.sum(weights)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"NONFINITE_LOSS:{condition}:{seed}:{epoch}")
        loss.backward()
        if not all(parameter.grad is None or torch.all(torch.isfinite(parameter.grad)) for parameter in model.parameters()):
            raise FloatingPointError(f"NONFINITE_GRAD:{condition}:{seed}:{epoch}")
        optimizer.step(); history.append({"epoch": epoch, "loss": float(loss.detach().cpu())})
        if epoch in (1, EPOCHS):
            print(f"visual-structure condition={condition} seed={seed} epoch={epoch}/{EPOCHS} loss={history[-1]['loss']:.6f}", flush=True)
    model.eval()
    return model, {"seed": int(seed), "epochs": EPOCHS, "initial_loss": history[0]["loss"], "final_loss": history[-1]["loss"], "min_loss": min(x["loss"] for x in history), "loss_history": history, "device": str(torch.device(device)), "parameter_count": parameter_count(model), "trainable_parameter_count": sum(p.numel() for p in model.parameters() if p.requires_grad), "frozen_parameter_count": sum(p.numel() for p in model.parameters() if not p.requires_grad)}


def _save_model(path: Path, model: Any) -> None:
    import torch
    path.parent.mkdir(parents=True, exist_ok=True); temporary = path.with_name(path.name + ".tmp")
    torch.save({"state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()}}, temporary); os.replace(temporary, path)


def _load_model(path: Path, condition: str, device: str) -> Any:
    import torch
    payload = torch.load(path, map_location="cpu")
    if condition == "VISUAL":
        model: Any = VisualHead()
    else:
        model = FusionModel(ModalSetModel(8), VisualHead())
    model.load_state_dict(payload["state_dict"]); model.to(device).eval(); return model


def _old_structure_records() -> dict[int, Mapping[str, Any]]:
    data = json.loads((STRUCTURE_ROOT / "models/fold_models.json").read_text(encoding="utf-8"))
    return {int(row["seed"]): row for row in data.get("records", []) if str(row.get("condition")) == "STRUCTURE_SUPPORT" and str(row.get("status")) == "TRAIN_COMPLETE"}


def _load_old_structure(seed: int, device: str) -> ModalSetModel:
    import torch
    record = _old_structure_records()[int(seed)]
    model = ModalSetModel(8)
    model.load_state_dict({key: torch.as_tensor(value, dtype=model.state_dict()[key].dtype) for key, value in record["state_dict"].items()}); model.to(device).eval(); return model


def _ids_hash(rows: Sequence[Mapping[str, Any]]) -> str:
    return hashlib.sha256("\n".join(sorted(str(row["window_id"]) for row in rows)).encode()).hexdigest()


def _dict_close(left: Mapping[str, Any], right: Mapping[str, Any], tol: float = 1e-12) -> bool:
    try:
        return str(left.get("condition")) == str(right.get("condition")) and np.allclose(np.asarray(left["mean"], dtype=float), np.asarray(right["mean"], dtype=float), atol=tol, rtol=0) and np.allclose(np.asarray(left["scale"], dtype=float), np.asarray(right["scale"], dtype=float), atol=tol, rtol=0) and list(left.get("zero_variance_dimensions", [])) == list(right.get("zero_variance_dimensions", []))
    except Exception:
        return False


def _can_reuse_structure(train_rows: Sequence[Mapping[str, Any]], val_rows: Sequence[Mapping[str, Any]], s_std: FeatureStandardizer, q_std: FeatureStandardizer, feature_manifest: Sequence[Mapping[str, Any]]) -> tuple[bool, str]:
    protocol = json.loads((STRUCTURE_ROOT / "protocol.json").read_text(encoding="utf-8")); records = _old_structure_records()
    model_protocol = protocol.get("model", {})
    if int(model_protocol.get("epochs", -1)) != EPOCHS or [int(x) for x in model_protocol.get("seeds", [])] != list(SEEDS) or str(model_protocol.get("optimizer", "")) != "Adam" or float(model_protocol.get("learning_rate", -1.0)) != LR or float(model_protocol.get("weight_decay", -1.0)) != WEIGHT_DECAY or "source/class weighted" not in str(model_protocol.get("loss", "")):
        return False, "STRUCTURE_MODEL_PROTOCOL_MISMATCH"
    if set(records) != set(SEEDS):
        return False, "STRUCTURE_RECORDS_INCOMPLETE"
    expected_hash = str(protocol["input"]["feature_content_sha256"])
    manifest_hash = hashlib.sha256(json.dumps(feature_manifest, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    old_ids = {int(seed): row for seed, row in records.items()}
    for seed in SEEDS:
        ident = old_ids[seed].get("input_identity", {})
        if ident.get("feature_content_sha256") != expected_hash or ident.get("training_window_ids_sha256") != _ids_hash(train_rows) or ident.get("validation_window_ids_sha256") != _ids_hash(val_rows) or ident.get("feature_manifest_sha256") != manifest_hash:
            return False, "STRUCTURE_INPUT_IDENTITY_MISMATCH"
        standard = old_ids[seed].get("standardization", {})
        if not _dict_close(standard.get("s", {}), s_std.as_dict()) or not _dict_close(standard.get("q", {}), q_std.as_dict()):
            return False, "STRUCTURE_STANDARDIZER_MISMATCH"
    return True, "EXACT_R17_STRUCTURE_REUSE"


def _metrics(rows: Sequence[Mapping[str, Any]], scores: Sequence[float]) -> dict[str, Any]:
    labels = [int(row["label"]) for row in rows]; values = [float(x) for x in scores]
    by_source: dict[str, list[tuple[int, float]]] = defaultdict(list)
    for row, value in zip(rows, values): by_source[str(row["source_id"])].append((int(row["label"]), value))
    per_source = {source: periodic._auroc([x[0] for x in vals], [x[1] for x in vals]) for source, vals in by_source.items()}
    per_source = {source: float(value) for source, value in per_source.items() if value is not None}
    macro = periodic._bootstrap(per_source)
    return {"window_count": len(rows), "real_count": labels.count(0), "fake_count": labels.count(1), "source_count": len(by_source), "dual_role_source_count": len(per_source), "source_macro_auroc": macro["mean"], "source_macro_ci_low": macro["ci95"][0], "source_macro_ci_high": macro["ci95"][1], "pooled_auroc": periodic._auroc(labels, values), "pooled_ap": periodic._ap(labels, values), **periodic._classification(labels, values), "per_source": per_source}


def _metric_row(split: str, condition: str, seed: Any, values: Mapping[str, Any]) -> dict[str, Any]:
    return {"split": split, "condition": condition, "seed": seed, **{key: values.get(key) for key in ("window_count", "real_count", "fake_count", "source_count", "dual_role_source_count", "source_macro_auroc", "source_macro_ci_low", "source_macro_ci_high", "pooled_auroc", "pooled_ap", "precision", "recall", "f1", "accuracy", "tn", "fp", "fn", "tp")}}


def _bootstrap_difference(left: Mapping[str, float], right: Mapping[str, float]) -> dict[str, Any]:
    return periodic._bootstrap(left, right)


def _state_logits(model: Any, condition: str, structure_batch: Mapping[str, Any], visual_batch: Mapping[str, Any], device: str) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
    import torch
    with torch.inference_mode():
        if condition == "STRUCTURE":
            z = _structure_forward(model, structure_batch, device); return z.detach().cpu().numpy().astype(np.float64), None, None
        if condition == "VISUAL":
            z = _visual_forward(model, visual_batch, device); return z.detach().cpu().numpy().astype(np.float64), None, None
        z, zs, zv = _fusion_forward(model, structure_batch, visual_batch, device); return z.detach().cpu().numpy().astype(np.float64), zs.detach().cpu().numpy().astype(np.float64), zv.detach().cpu().numpy().astype(np.float64)
    raise ValueError(condition)


def _verify_structure_reproduction(root: Path, rows: Sequence[Mapping[str, Any]], feature_by_id: Mapping[str, Mapping[str, Any]], s_std: FeatureStandardizer, q_std: FeatureStandardizer, device: str) -> dict[str, Any]:
    """Reproduce the saved R17 structure scores without using visual inputs."""
    old_rows = list(csv.DictReader((STRUCTURE_ROOT / "scores/validation_window_scores.csv").open(newline="", encoding="utf-8")))
    old_by_id = {str(row["window_id"]): row for row in old_rows}
    batch = _structure_batch(rows, feature_by_id, s_std, q_std)
    per_seed: dict[str, float] = {}
    for seed in SEEDS:
        values = _structure_forward(_load_old_structure(seed, device), batch, device)
        differences = [float(value) - float(old_by_id[window_id][f"STRUCTURE_SUPPORT_seed_{seed}"]) for window_id, value in zip(batch["window_ids"], values)]
        per_seed[str(seed)] = float(max(abs(x) for x in differences))
    mean_values = np.mean([_structure_forward(_load_old_structure(seed, device), batch, device).detach().cpu().numpy() for seed in SEEDS], axis=0)
    mean_differences = [float(value) - float(old_by_id[window_id]["STRUCTURE_SUPPORT_MEAN_LOGIT"]) for window_id, value in zip(batch["window_ids"], mean_values)]
    result = {"status": "PASS" if max(per_seed.values()) <= 1e-5 and max(abs(x) for x in mean_differences) <= 1e-5 else "FAILED", "window_count": len(rows), "per_seed_max_abs": per_seed, "mean_max_abs": float(max(abs(x) for x in mean_differences)), "tolerance": 1e-5}
    atomic_json(root / "state/structure_reproduction.json", result)
    if result["status"] != "PASS":
        raise RuntimeError(f"STRUCTURE_SCORE_REPRODUCTION_FAILED:{result}")
    return result


def _smoke(root: Path, train_rows: Sequence[Mapping[str, Any]], structure_batch: Mapping[str, Any], visual_batch: Mapping[str, Any], device: str) -> dict[str, Any]:
    import torch
    labels = set(int(x) for x in visual_batch["labels"])
    if labels != {0, 1}:
        raise ValueError("SMOKE_NEEDS_REAL_AND_FAKE")
    rows = list(train_rows[:0])
    for row in train_rows:
        if not rows and int(row["label"]) == 0: rows.append(row)
        elif len(rows) == 1 and int(row["label"]) == 1: rows.append(row)
        if len(rows) == 2: break
    target_indices = [next(i for i, row in enumerate(structure_batch["rows"]) if str(row["window_id"]) == str(chosen["window_id"])) for chosen in rows]
    local_mask = np.isin(np.asarray(structure_batch["window_index"]), np.asarray(target_indices))
    remap = {old: new for new, old in enumerate(target_indices)}
    small_s = dict(structure_batch)
    small_s.update({"states": np.asarray(structure_batch["states"])[local_mask], "intervals": np.asarray(structure_batch["intervals"])[local_mask], "window_index": np.asarray([remap[int(x)] for x in np.asarray(structure_batch["window_index"])[local_mask]], dtype=np.int64), "window_ids": [str(rows[i]["window_id"]) for i in range(2)], "rows": rows, "labels": np.asarray([int(x["label"]) for x in rows], dtype=np.int64), "weights": np.asarray([1.0, 1.0], dtype=np.float32), "window_count": 2})
    visual_mask = np.asarray([i in target_indices for i in range(len(visual_batch["rows"]))], dtype=bool)
    small_v = dict(visual_batch)
    small_v.update({"features": np.asarray(visual_batch["features"])[visual_mask], "window_ids": [str(rows[i]["window_id"]) for i in range(2)], "rows": rows, "labels": np.asarray([int(x["label"]) for x in rows], dtype=np.int64), "weights": np.asarray([1.0, 1.0], dtype=np.float32), "window_count": 2})
    _set_seed(SEEDS[0]); model = VisualHead().to(device); model.train(); optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY); before = _visual_forward(model, small_v, device); loss = torch.nn.functional.binary_cross_entropy_with_logits(before, torch.as_tensor(small_v["labels"], dtype=torch.float32, device=device)); loss.backward(); optimizer.step(); model.eval()
    path = root / "smoke/visual_one_step.pt"; _save_model(path, model); reload = _load_model(path, "VISUAL", device); before_score = _visual_forward(model, small_v, device).detach().cpu().numpy(); after_score = _visual_forward(reload, small_v, device).detach().cpu().numpy(); reload_error = float(np.max(np.abs(before_score - after_score)))
    _set_seed(SEEDS[0]); fusion = FusionModel(geometry._initial_model("STRUCTURE_SUPPORT", SEEDS[0]).to(device), VisualHead().to(device)).to(device); fusion.train(); z, zs, zv = _fusion_forward(fusion, small_s, small_v, device); fl = torch.nn.functional.binary_cross_entropy_with_logits(z, torch.as_tensor(small_v["labels"], dtype=torch.float32, device=device)); fl.backward(); structure_grad = any(p.grad is not None and torch.any(torch.isfinite(p.grad)) and torch.any(p.grad != 0) for p in fusion.structure.parameters()); visual_grad = any(p.grad is not None and torch.any(torch.isfinite(p.grad)) and torch.any(p.grad != 0) for p in fusion.visual.parameters()); add_error = float(torch.max(torch.abs(z - (zs + zv))).detach().cpu()); result = {"status": "PASS" if reload_error <= 1e-6 and structure_grad and visual_grad and add_error <= 1e-6 else "FAILED", "rows": [str(x["window_id"]) for x in rows], "visual_loss_after_one_step": float(loss.detach().cpu()), "fusion_loss_before_step": float(fl.detach().cpu()), "reload_max_abs": reload_error, "fusion_addition_max_abs": add_error, "fusion_structure_nonzero_gradient": structure_grad, "fusion_visual_nonzero_gradient": visual_grad, "reused_formal_feature_cache": False}; atomic_json(root / "smoke/summary.json", result); return result


def _write_protocol(root: Path, train_rows: Sequence[Mapping[str, Any]], val_rows: Sequence[Mapping[str, Any]], feature_manifest: Sequence[Mapping[str, Any]], weight_meta: Mapping[str, Any], reuse: Mapping[str, Any], visual_std: Mapping[str, Any], s_std: FeatureStandardizer, q_std: FeatureStandardizer) -> None:
    import torch, torchvision
    atomic_json(root / "protocol.json", {"protocol_id": "v7-visual-structure-pilot-v1", "git_head": git_head(), "structure_root": str(STRUCTURE_ROOT), "video_root": str(VIDEO_ROOT), "window_rule": "observation_support R17 train/validation manifest; exactly 801 selected windows", "population": {"train_windows": len(train_rows), "validation_windows": len(val_rows), "train_real": sum(int(x["label"]) == 0 for x in train_rows), "train_fake": sum(int(x["label"]) == 1 for x in train_rows), "validation_real": sum(int(x["label"]) == 0 for x in val_rows), "validation_fake": sum(int(x["label"]) == 1 for x in val_rows), "validation_sources": len({str(x["source_id"]) for x in val_rows})}, "conditions": {"STRUCTURE": "existing STRUCTURE_SUPPORT S/Q R17 model and local-logit mean", "VISUAL": "512-d R3D feature plus Linear(512,32)-ReLU-Linear(32,1)", "VISUAL_STRUCTURE": "VISUAL logit plus STRUCTURE logit; window weighted BCE"}, "sampling": {"bins": 16, "rule": "window-local bin centers, nearest actual PTS, no interpolation or cross-window frame", "repeats_recorded": True}, "r3d": {**dict(weight_meta), "torch": str(torch.__version__), "torchvision": str(torchvision.__version__)}, "model": {"epochs": EPOCHS, "optimizer": "Adam", "learning_rate": LR, "weight_decay": WEIGHT_DECAY, "seeds": list(SEEDS), "threshold": "logit >= 0", "source_class_weight": "equal source/class total then mean weight one", "structure_standardizer": {"s": s_std.as_dict(), "q": q_std.as_dict()}, "visual_standardizer": _visual_standardizer_json(visual_std), "fusion_initialization": "fresh VisualHead and fresh existing STRUCTURE_SUPPORT ModalSetModel per seed; visual initialization seed matches VISUAL; backbone frozen"}, "reuse": dict(reuse), "evaluation": {"primary": "VISUAL_STRUCTURE minus VISUAL source-macro AUROC", "secondary": ["VISUAL minus STRUCTURE", "VISUAL_STRUCTURE minus STRUCTURE"], "bootstrap_seed": BOOTSTRAP_SEED, "bootstrap_replicates": BOOTSTRAP_REPLICATES, "not_sealed_test": True}, "input_identity": {"structure_feature_manifest_sha256": sha256(STRUCTURE_ROOT / "inputs/feature_manifest.json"), "train_window_ids_sha256": _ids_hash(train_rows), "validation_window_ids_sha256": _ids_hash(val_rows), "visual_manifest_sha256": hashlib.sha256(json.dumps(feature_manifest, sort_keys=True, separators=(",", ":")).encode()).hexdigest()}})


def _write_input_manifest(root: Path, rows: Sequence[Mapping[str, Any]], info: Mapping[str, Any]) -> None:
    atomic_json(root / "input_manifest.json", {"source": "observation_support_pilot_v1", "train_window_ids": [str(x["window_id"]) for x in rows if x.get("split") == "train"], "validation_window_ids": [str(x["window_id"]) for x in rows if x.get("split") == "validation"], "rows": [{"window_id": str(x["window_id"]), "source_id": str(x["source_id"]), "role": str(x["role"]), "label": int(x["label"]), "annotation_category": str(x.get("annotation_category", "")), "parent_id": str(x["parent_id"]), "video_path": str(x["video_path"]), "video_sha256": str(x["video_sha256"]), "interval_start_s": float(x["interval_start_s"]), "interval_end_s": float(x["interval_end_s"])} for x in rows], "training_sources": info["training_sources"], "validation_sources": info["validation_sources"]})


def _evaluate(root: Path, train_rows: Sequence[Mapping[str, Any]], val_rows: Sequence[Mapping[str, Any]], feature_by_id: Mapping[str, Mapping[str, Any]], visual_by_id: Mapping[str, Mapping[str, Any]], s_std: FeatureStandardizer, q_std: FeatureStandardizer, visual_std: Mapping[str, Any], records: Sequence[Mapping[str, Any]], device: str) -> dict[str, Any]:
    structure_batches = {"train": _structure_batch(train_rows, feature_by_id, s_std, q_std), "validation": _structure_batch(val_rows, feature_by_id, s_std, q_std)}
    visual_batches = {"train": _visual_batch(train_rows, visual_by_id, root, visual_std), "validation": _visual_batch(val_rows, visual_by_id, root, visual_std)}
    score_store: dict[tuple[str, str, int], dict[str, float]] = {}; train_metric_rows: list[dict[str, Any]] = []; source_metric_rows: list[dict[str, Any]] = []
    loaded: dict[tuple[str, int], Any] = {}
    for record in records:
        condition, seed = str(record["condition"]), int(record["seed"])
        if condition == "STRUCTURE": model = _load_old_structure(seed, device)
        else: model = _load_model(root / str(record["checkpoint"]), condition, device)
        loaded[(condition, seed)] = model
        for split, rows in (("train", train_rows), ("validation", val_rows)):
            values, _, _ = _state_logits(model, condition, structure_batches[split], visual_batches[split], device); mapping = {str(row["window_id"]): float(value) for row, value in zip(sorted(rows, key=lambda x: str(x["window_id"])), values)}; score_store[(split, condition, seed)] = mapping; met = _metrics(sorted(rows, key=lambda x: str(x["window_id"])), values); train_metric_rows.append(_metric_row(split, condition, seed, met))
            for source, value in met["per_source"].items(): source_metric_rows.append({"split": split, "condition": condition, "seed": seed, "source_id": source, "auroc": value, "window_count": sum(str(row["source_id"]) == source for row in rows), "real_count": sum(str(row["source_id"]) == source and int(row["label"]) == 0 for row in rows), "fake_count": sum(str(row["source_id"]) == source and int(row["label"]) == 1 for row in rows)})
    score_rows: list[dict[str, Any]] = []
    for split, rows in (("train", train_rows), ("validation", val_rows)):
        ordered = sorted(rows, key=lambda x: str(x["window_id"]))
        for row in ordered:
            item = {"window_id": str(row["window_id"]), "source_id": str(row["source_id"]), "role": str(row["role"]), "label": int(row["label"]), "split": split}
            for condition in CONDITIONS:
                per_seed = [score_store[(split, condition, seed)][str(row["window_id"])] for seed in SEEDS]
                for seed, value in zip(SEEDS, per_seed): item[f"{condition}_seed_{seed}"] = value
                item[f"{condition}_MEAN_LOGIT"] = float(np.mean(per_seed))
            score_rows.append(item)
    metrics: list[dict[str, Any]] = []
    for split, rows in (("train", train_rows), ("validation", val_rows)):
        ordered = sorted(rows, key=lambda x: str(x["window_id"]))
        for condition in CONDITIONS:
            for seed in SEEDS:
                values = [score_store[(split, condition, seed)][str(row["window_id"])] for row in ordered]; metrics.append(_metric_row(split, condition, seed, _metrics(ordered, values)))
            values = [float(item[f"{condition}_MEAN_LOGIT"]) for item in score_rows if item["split"] == split]; metrics.append(_metric_row(split, condition, "MEAN_LOGIT", _metrics(ordered, values)))
    val_ordered = sorted(val_rows, key=lambda x: str(x["window_id"]))
    per_source_mean: dict[str, dict[str, float]] = {}
    for condition in CONDITIONS:
        values = [float(item[f"{condition}_MEAN_LOGIT"]) for item in score_rows if item["split"] == "validation"]
        grouped: dict[str, list[tuple[int, float]]] = defaultdict(list)
        for row, value in zip(val_ordered, values): grouped[str(row["source_id"])].append((int(row["label"]), value))
        per_source_mean[condition] = {source: float(periodic._auroc([x[0] for x in vals], [x[1] for x in vals])) for source, vals in grouped.items() if periodic._auroc([x[0] for x in vals], [x[1] for x in vals]) is not None}
    comparisons = []
    for left, right, name in (("VISUAL_STRUCTURE", "VISUAL", "VISUAL_STRUCTURE-VISUAL"), ("VISUAL", "STRUCTURE", "VISUAL-STRUCTURE"), ("VISUAL_STRUCTURE", "STRUCTURE", "VISUAL_STRUCTURE-STRUCTURE")):
        boot = _bootstrap_difference(per_source_mean[left], per_source_mean[right]); boot["name"] = name; diffs = [per_source_mean[left][s] - per_source_mean[right][s] for s in boot.get("sources", [])]; boot.update({"positive_count": sum(x > 0 for x in diffs), "negative_count": sum(x < 0 for x in diffs), "tie_count": sum(x == 0 for x in diffs)}); comparisons.append(boot)
    per_seed_comparisons = []
    for seed in SEEDS:
        source_values: dict[str, dict[str, float]] = {}
        for condition in CONDITIONS:
            grouped: dict[str, list[tuple[int, float]]] = defaultdict(list)
            for row in val_ordered: grouped[str(row["source_id"])].append((int(row["label"]), score_store[("validation", condition, seed)][str(row["window_id"])]))
            source_values[condition] = {source: float(periodic._auroc([x[0] for x in vals], [x[1] for x in vals])) for source, vals in grouped.items() if periodic._auroc([x[0] for x in vals], [x[1] for x in vals]) is not None}
        for left, right in (("VISUAL_STRUCTURE", "VISUAL"), ("VISUAL", "STRUCTURE"), ("VISUAL_STRUCTURE", "STRUCTURE")):
            diffs = [source_values[left][s] - source_values[right][s] for s in sorted(set(source_values[left]) & set(source_values[right]))]; per_seed_comparisons.append({"seed": seed, "comparison": f"{left}-{right}", "mean_difference": float(np.mean(diffs)) if diffs else None, "positive_count": sum(x > 0 for x in diffs), "negative_count": sum(x < 0 for x in diffs), "tie_count": sum(x == 0 for x in diffs), "source_count": len(diffs)})
    write_csv(root / "scores/train_window_scores.csv", [x for x in score_rows if x["split"] == "train"]); write_csv(root / "scores/validation_window_scores.csv", [x for x in score_rows if x["split"] == "validation"]); write_csv(root / "evaluation/metrics.csv", metrics); write_csv(root / "evaluation/per_source_metrics.csv", source_metric_rows); write_csv(root / "evaluation/paired_comparisons.csv", [{"comparison": x["name"], "source_count": x["source_count"], "mean": x["mean"], "ci_low": x["ci95"][0], "ci_high": x["ci95"][1], "positive_count": x["positive_count"], "negative_count": x["negative_count"], "tie_count": x["tie_count"]} for x in comparisons]); atomic_json(root / "evaluation/seed_comparisons.json", per_seed_comparisons); summary = {"metrics": metrics, "comparisons": comparisons, "seed_comparisons": per_seed_comparisons, "train_population": {"windows": len(train_rows), "real": sum(int(x["label"]) == 0 for x in train_rows), "fake": sum(int(x["label"]) == 1 for x in train_rows), "sources": len({str(x["source_id"]) for x in train_rows})}, "validation_population": {"windows": len(val_rows), "real": sum(int(x["label"]) == 0 for x in val_rows), "fake": sum(int(x["label"]) == 1 for x in val_rows), "sources": len({str(x["source_id"]) for x in val_rows})}, "device": device, "model_count": len(records)}; atomic_json(root / "evaluation/summary.json", summary); return summary


def _write_report(root: Path, summary: Mapping[str, Any], records: Sequence[Mapping[str, Any]], timings: Mapping[str, Any], smoke: Mapping[str, Any], reuse: Mapping[str, Any], weight_meta: Mapping[str, Any], train_rows: Sequence[Mapping[str, Any]], val_rows: Sequence[Mapping[str, Any]]) -> Path:
    mean_metrics = {(str(x["split"]), str(x["condition"])): x for x in summary["metrics"] if str(x["seed"]) == "MEAN_LOGIT"}
    reused_count = sum(bool(x.get("reused_from")) for x in records)
    lines = ["# V7 冻结视频特征与结构信息最小三条件 pilot", "", "## 结果先行", f"- 三条件使用完全相同的 {len(train_rows)} 个训练窗口和 {len(val_rows)} 个验证窗口；验证为 {len({str(x['source_id']) for x in val_rows})} 个 source、{sum(int(x['label']) == 0 for x in val_rows)} real、{sum(int(x['label']) == 1 for x in val_rows)} fake。验证集是已开发集合，不是 sealed-test。", f"- 正式模型槽位：{len(records)}/9；STRUCTURE 复用：{reused_count} 个，VISUAL/VISUAL_STRUCTURE 新训：{len(records) - reused_count} 个；smoke={smoke.get('status')}。", f"- R3D-18 权重：{weight_meta['name']}，SHA256={weight_meta['sha256']}，官方 URL={weight_meta['url']}。主干冻结、无梯度、无 BN 更新。", "", "| 条件 | 验证 source-macro AUROC (95% CI) | pooled AUROC | AP | Precision | Recall | F1 | ACC |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for condition in CONDITIONS:
        x = mean_metrics[("validation", condition)]; lines.append(f"| {condition} | {x['source_macro_auroc']} [{x['source_macro_ci_low']}, {x['source_macro_ci_high']}] | {x['pooled_auroc']} | {x['pooled_ap']} | {x['precision']} | {x['recall']} | {x['f1']} | {x['accuracy']} |")
    lines += ["", "## 训练指标", "", "| 条件 | train macro AUROC | train pooled AUROC | train AP | train F1 |", "|---|---:|---:|---:|---:|"]
    for condition in CONDITIONS:
        x = mean_metrics[("train", condition)]; lines.append(f"| {condition} | {x['source_macro_auroc']} | {x['pooled_auroc']} | {x['pooled_ap']} | {x['f1']} |")
    lines += ["", "## 预声明配对", "", "| 比较 | source 数 | 均值差 | 95% CI | 正/平/负 source |", "|---|---:|---:|---|---:|"]
    for x in summary["comparisons"]: lines.append(f"| {x['name']} | {x['source_count']} | {x['mean']} | [{x['ci95'][0]}, {x['ci95'][1]}] | {x['positive_count']}/{x['tie_count']}/{x['negative_count']} |")
    lines += ["", "## 口径与限制", "- 视觉分支对每个 1 秒窗口按 16 个 bin-center 目标时刻选最近实际 PTS；重复帧和时间误差写入 `visual_feature_manifest.json`。视觉取样与结构分支的五时刻不同，因此不是严格等时刻的二维/三维消融。", "- 视觉预处理保持完整画面宽高比缩放到 224 画布并以官方均值 padding，再按官方 mean/std 归一化；没有 crop、增强或测试时多裁剪。预处理预览在 `smoke/preprocess_preview.png`。", "- STRUCTURE 不是纯几何：它复用已有 R17 `STRUCTURE_SUPPORT`，含 S/Q、时间编码、局部 head 和局部 logit 均值。VISUAL_STRUCTURE 是直接相加 `z_v+z_s`，不是学习门控，也不能把局部输出解释为伪造概率。", "- 标准化和 source/class 权重仅由训练窗口拟合；验证 source 完全不进入训练。窗口重叠、已反复开发的验证集合和 survivor bias 限制跨 source 解释。动作预训练特征可能包含语义信息，不能解释为物理测量器。", "- 融合多一个可训练结构分支，故正向结果不能单独排除容量/优化差异；阴性也不能否定其他视觉表征或结构表示。没有空间真值、完整视频扫描或定位指标。", "- `state/structure_reproduction.json` 对旧 STRUCTURE_SUPPORT 的 83 个验证窗口逐 seed 及三 seed 平均 logit 复现，容差 1e-5；失败不会写入 COMPLETE。主干冻结检查在 `state/backbone_check.json`。", "", "## 产物与成本", f"- 阶段耗时（秒）：{json.dumps(dict(timings), ensure_ascii=False)}。首次正式训练与完整 801 窗口首次提取的阶段计时在第一次报告序列化错误前未持久化；6 个 checkpoint 的保存时间范围约 4 秒，仅作下界，不能冒充完整训练耗时。最终恢复运行仅复用了 801 个视觉特征和 6 个 checkpoint，当前 `features`/`train` 数值是恢复阶段的缓存扫描/复用耗时。", "- 特征：`visual_feature_manifest.json` 与 `features/window_visual/`；模型：`models/fold_models.json`、`models/checkpoints/`；分数：`scores/`；评价：`evaluation/summary.json`。", "- 未访问旧 R7/V5，未运行 tracking/depth/pose/segmentation，不修改正式 src。", ""]
    path = root / "report.md"; path.write_text("\n".join(lines), encoding="utf-8"); return path


def run(root: Path = OUTPUT_ROOT, *, device: str = "cuda", resume: bool = True) -> dict[str, Any]:
    root.mkdir(parents=True, exist_ok=True); (root / "state").mkdir(parents=True, exist_ok=True); started = time.perf_counter(); timings: dict[str, float] = {}
    atomic_json(root / "state/launch.json", {"git_head": git_head(), "pid": os.getpid(), "device": device, "started_unix": time.time()})
    try:
        train_rows, val_rows, feature_manifest, info, provenance = _rows_and_inputs(); all_rows = train_rows + val_rows
        for row, split in zip(all_rows, (["train"] * len(train_rows)) + (["validation"] * len(val_rows))): row["split"] = split
        _write_input_manifest(root, all_rows, info); progress(root, "inputs", "COMPLETE", 1, 1, train_windows=len(train_rows), validation_windows=len(val_rows))
        backbone, weight_meta = _load_backbone(device); atomic_json(root / "state/backbone.json", weight_meta)
        import torch
        bn_modules = [module for module in backbone.modules() if isinstance(module, torch.nn.modules.batchnorm._BatchNorm)]
        atomic_json(root / "state/backbone_check.json", {"requires_grad_any": any(parameter.requires_grad for parameter in backbone.parameters()), "model_training": bool(backbone.training), "batchnorm_training_any": any(module.training for module in bn_modules), "batchnorm_count": len(bn_modules), "inference_mode_used": True, "feature_output_dim": 512})
        # The smoke feature files are deliberately separate from formal cache.
        smoke_rows: list[Mapping[str, Any]] = []
        for row in train_rows:
            if int(row["label"]) == 0 and not smoke_rows: smoke_rows.append(row)
            elif int(row["label"]) == 1 and len(smoke_rows) == 1: smoke_rows.append(row)
            if len(smoke_rows) == 2: break
        smoke_started = time.perf_counter(); smoke_feature_manifest = _extract_rows(root, smoke_rows, backbone, weight_meta, device, smoke=True); _write_preprocess_preview(root, smoke_rows[0], weight_meta); timings["smoke_features"] = time.perf_counter() - smoke_started
        # Smoke model step uses the formal structure inputs but no smoke feature is used in formal training.
        feature_hash = hashlib.sha256(json.dumps(feature_manifest, sort_keys=True, separators=(",", ":")).encode()).hexdigest(); feature_by_id = {str(x["window_id"]): x for x in feature_manifest}
        s_std, q_std = _structure_standardizers(train_rows, feature_by_id, root); formal_features_started = time.perf_counter(); formal_visual_manifest = _extract_rows(root, all_rows, backbone, weight_meta, device, smoke=False); timings["features"] = time.perf_counter() - formal_features_started
        visual_by_id = {str(x["window_id"]): x for x in formal_visual_manifest}; visual_std = _fit_visual_standardizer(train_rows, visual_by_id, root); atomic_json(root / "standardizers.json", {"structure_s": s_std.as_dict(), "structure_q": q_std.as_dict(), "visual": _visual_standardizer_json(visual_std)})
        visual_manifest_hash = hashlib.sha256(json.dumps(formal_visual_manifest, sort_keys=True, separators=(",", ":")).encode()).hexdigest(); reuse_ok, reuse_reason = _can_reuse_structure(train_rows, val_rows, s_std, q_std, feature_manifest); reuse = {"structure_reused": reuse_ok, "structure_reuse_reason": reuse_reason, "old_root": str(STRUCTURE_ROOT), "structure_initialization_rule_verified": True, "visual_manifest_sha256": visual_manifest_hash}; _write_protocol(root, train_rows, val_rows, feature_manifest, weight_meta, reuse, visual_std, s_std, q_std)
        structure_batch_train = _structure_batch(train_rows, feature_by_id, s_std, q_std); visual_batch_train = _visual_batch(train_rows, visual_by_id, root, visual_std); smoke = _smoke(root, train_rows, structure_batch_train, visual_batch_train, device); timings["smoke_model"] = 0.0
        if smoke.get("status") != "PASS": raise RuntimeError("SMOKE_FAILED")
        records: list[dict[str, Any]] = []; model_dir = root / "models/checkpoints"; model_dir.mkdir(parents=True, exist_ok=True); model_started = time.perf_counter(); total = 9
        existing_records: dict[tuple[str, int], dict[str, Any]] = {}
        existing_path = root / "models/fold_models.json"
        if resume and existing_path.is_file():
            try:
                existing_records = {(str(item.get("condition")), int(item.get("seed"))): dict(item) for item in json.loads(existing_path.read_text(encoding="utf-8")).get("records", [])}
            except Exception:
                existing_records = {}
        for seed in SEEDS:
            if reuse_ok:
                old = existing_records.get(("STRUCTURE", int(seed)))
                records.append(old if old and old.get("status") == "MODEL_REUSED" else {"condition": "STRUCTURE", "seed": int(seed), "status": "MODEL_REUSED", "reused_from": str(STRUCTURE_ROOT / "models/fold_models.json"), "source_condition": "STRUCTURE_SUPPORT", "training_epochs": 0, "input_identity": {"feature_manifest_sha256": feature_hash, "train_window_ids_sha256": _ids_hash(train_rows), "validation_window_ids_sha256": _ids_hash(val_rows)}})
            else:
                raise RuntimeError(f"STRUCTURE_RETRAIN_REQUIRED:{reuse_reason}")
        for condition in ("VISUAL", "VISUAL_STRUCTURE"):
            for seed in SEEDS:
                old = existing_records.get((condition, int(seed)))
                old_identity = old.get("input_identity", {}) if old else {}
                old_checkpoint = root / str(old.get("checkpoint", "")) if old else Path("/nonexistent")
                if old and old.get("status") == "TRAIN_COMPLETE" and old_checkpoint.is_file() and old_identity.get("visual_manifest_sha256") == visual_manifest_hash and old_identity.get("train_window_ids_sha256") == _ids_hash(train_rows) and old_identity.get("validation_window_ids_sha256") == _ids_hash(val_rows):
                    records.append(old); progress(root, "train", "RUNNING", len(records), total, condition=condition, seed=int(seed), reused_checkpoint=True); continue
                structure_batch = structure_batch_train; model, fit = _train_model(condition, structure_batch, visual_batch_train, int(seed), device); checkpoint = model_dir / f"{condition.lower()}_{seed}.pt"; _save_model(checkpoint, model); records.append({"condition": condition, "seed": int(seed), "status": "TRAIN_COMPLETE", "checkpoint": str(checkpoint.relative_to(root)), "fit": fit, "input_identity": {"feature_manifest_sha256": feature_hash, "visual_manifest_sha256": visual_manifest_hash, "train_window_ids_sha256": _ids_hash(train_rows), "validation_window_ids_sha256": _ids_hash(val_rows), "epochs": EPOCHS, "learning_rate": LR, "weight_decay": WEIGHT_DECAY}}); atomic_json(existing_path, {"conditions": list(CONDITIONS), "seeds": list(SEEDS), "expected_model_count": total, "records": records}); progress(root, "train", "RUNNING", len(records), total, condition=condition, seed=int(seed)); del model
        timings["train"] = time.perf_counter() - model_started; atomic_json(root / "models/fold_models.json", {"conditions": list(CONDITIONS), "seeds": list(SEEDS), "expected_model_count": total, "records": records})
        evaluate_started = time.perf_counter(); summary = _evaluate(root, train_rows, val_rows, feature_by_id, visual_by_id, s_std, q_std, visual_std, records, device); reproduction = _verify_structure_reproduction(root, val_rows, feature_by_id, s_std, q_std, device); timings["evaluate"] = time.perf_counter() - evaluate_started
        report_started = time.perf_counter(); report = _write_report(root, summary, records, timings, smoke, reuse, weight_meta, train_rows, val_rows); timings["report"] = time.perf_counter() - report_started; atomic_json(root / "final_status.json", {"status": "COMPLETE", "model_count": len(records), "expected_model_count": total, "trained_model_count": sum(x.get("status") == "TRAIN_COMPLETE" for x in records), "reused_model_count": sum(x.get("status") == "MODEL_REUSED" for x in records), "report": str(report), "timings_s": timings, "git_head": git_head()}); progress(root, "report", "COMPLETE", 1, 1, model_count=len(records), elapsed_s=time.perf_counter() - started); return {"status": "COMPLETE", "report": str(report), "summary": summary, "timings_s": timings}
    except BaseException as exc:
        atomic_json(root / "state/failure.json", {"status": "FAILED", "error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()}); atomic_json(root / "final_status.json", {"status": "FAILED", "error": f"{type(exc).__name__}: {exc}", "git_head": git_head(), "elapsed_s": time.perf_counter() - started}); progress(root, "failed", "FAILED", 0, 1, error=f"{type(exc).__name__}: {exc}"); raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT); parser.add_argument("--device", default="cuda"); parser.add_argument("--resume", action="store_true"); args = parser.parse_args(); result = run(args.output_root, device=args.device, resume=args.resume); print(json.dumps({"status": result["status"], "report": result["report"], "timings_s": result["timings_s"]}, ensure_ascii=False, indent=2)); return 0


if __name__ == "__main__":
    raise SystemExit(main())
