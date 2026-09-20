"""Run the bounded V7 raw component-geometry time-order pilot.

This module reads the already materialized component-geometry feature cache.
It does not open videos or invoke tracking, depth, pose, segmentation, or
feature extraction.  A, B and C share the exact retained units, fixed edges,
Q values, labels, source split and window aggregation.  Only the time
aggregation of the raw 3-D edge state is changed.
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

from research_tools.v7.component_geometry_pilot import runner as base
from research_tools.v7.multi_order_sequence_probe.model import FeatureStandardizer, fit_standardizer, source_class_weights
from research_tools.v7.periodic_requery_probe import runner as periodic

from .model import OrderedComponentGeometryModel, parameter_count


REPO_ROOT = Path(__file__).resolve().parents[3]
DATA_ROOT = Path("/root/autodl-tmp/data/sparse_3d_forgery_detection")
INPUT_ROOT = DATA_ROOT / "derived/v7_activityforensics_component_geometry_pilot_v1"
OUTPUT_ROOT = DATA_ROOT / "derived/v7_activityforensics_component_geometry_time_order_pilot_v1"
CONDITIONS = ("G3D_SET", "G3D_ORDERED", "G3D_ORDERED_SHUFFLED")
SEEDS = (20260909, 20260910, 20260911)
EPOCHS = 200
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
BOOTSTRAP_SEED = 20260909
BOOTSTRAP_REPLICATES = 10_000
PERMUTATION_SEED = 20260909
UNIT_CHUNK_SIZE = 256
WINDOW_TRAIN_CHUNK_SIZE = 64
MATCH_TOLERANCE = 1e-5


def _jsonable(value: Any, path: str = "$") -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return [_jsonable(v, f"{path}[{i}]") for i, v in enumerate(value)]
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return _jsonable(value.item(), path)
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"NONFINITE_JSON:{path}:{value!r}")
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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_head() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()


def _progress(root: Path, stage: str, status: str, completed: int, total: int, **extra: Any) -> None:
    atomic_json(root / "progress.json", {"stage": stage, "status": status, "completed": int(completed), "total": int(total), "updated_unix": time.time(), **extra})


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {key: np.asarray(archive[key]) for key in archive.files}


def _load_inputs() -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    train, validation, _, info = base._load_selection()
    manifest_path = INPUT_ROOT / "inputs/feature_manifest.json"
    old_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    by_id = {str(item["window_id"]): dict(item) for item in old_manifest}
    expected = {str(row["window_id"]) for row in train + validation}
    if set(by_id) != expected:
        raise ValueError(f"INPUT_WINDOW_SET_MISMATCH:{len(by_id)}:{len(expected)}")
    selected: list[dict[str, Any]] = []
    compact: list[dict[str, Any]] = []
    for row in train + validation:
        item = by_id[str(row["window_id"])]
        path = INPUT_ROOT / str(item["input_path"])
        if not path.is_file():
            raise FileNotFoundError(f"MISSING_INPUT_FEATURE:{path}")
        merged = {**dict(row), "matched_unit_count": int(item["matched_unit_count"]), "input_item": item}
        selected.append(merged)
        compact.append({
            "window_id": str(item["window_id"]),
            "source_id": str(item["source_id"]),
            "role": str(item["role"]),
            "label": int(item["label"]),
            "annotation_category": str(item.get("annotation_category", "")),
            "offset_s": float(item.get("offset_s", 0.0)),
            "split": "train" if len(compact) < len(train) else "validation",
            "source_feature_root": str(INPUT_ROOT),
            "source_feature_path": str(path),
            "source_feature_sha256": _sha256(path),
            "matched_unit_count": int(item["matched_unit_count"]),
            "edge3d_shape": list(item["edge3d_shape"]),
            "edge_mask_shape": list(item["edge_mask_shape"]),
            "q_shape": list(item["q_shape"]),
            "interval_shape": [4],
        })
    train_rows = selected[: len(train)]
    validation_rows = selected[len(train) :]
    train_sources = {str(row["source_id"]) for row in train_rows}
    validation_sources = {str(row["source_id"]) for row in validation_rows}
    if train_sources & validation_sources:
        raise ValueError("TRAIN_VALIDATION_SOURCE_OVERLAP")
    # The source cache stores the four adjacent PTS intervals in each NPZ and
    # the per-unit target timestamps in the identity manifest.  Check that the
    # ordered pilot is really using the same physical target times rather than
    # merely trusting an array-position convention.
    time_checked = 0
    nonuniform_intervals = 0
    for item in old_manifest:
        arrays = _load_npz(INPUT_ROOT / str(item["input_path"]))
        intervals = np.asarray(arrays["intervals"], dtype=np.float64)
        if intervals.shape != (4,) or not np.all(np.isfinite(intervals)) or np.any(intervals <= 0):
            raise ValueError(f"INVALID_STORED_INTERVALS:{item['window_id']}")
        if np.ptp(intervals) > 1e-9:
            nonuniform_intervals += 1
        identities = list(item.get("unit_identities", []))
        for identity in identities[:1]:
            timestamps = np.asarray(identity.get("target_timestamps_s", []), dtype=np.float64)
            if timestamps.shape != (5,) or not np.all(np.isfinite(timestamps)) or np.any(np.diff(timestamps) <= 0):
                raise ValueError(f"INVALID_TARGET_PTS:{item['window_id']}")
            if not np.allclose(np.diff(timestamps), intervals, atol=MATCH_TOLERANCE, rtol=0):
                raise ValueError(f"PTS_INTERVAL_MISMATCH:{item['window_id']}")
            time_checked += 1
    info = {**info, "input_manifest_sha256": _sha256(manifest_path), "input_content_sha256": base._content_hash(INPUT_ROOT, old_manifest), "training_windows": len(train_rows), "validation_windows": len(validation_rows), "training_sources": sorted(train_sources), "validation_sources": sorted(validation_sources), "training_real": sum(int(row["label"]) == 0 for row in train_rows), "training_fake": sum(int(row["label"]) == 1 for row in train_rows), "validation_real": sum(int(row["label"]) == 0 for row in validation_rows), "validation_fake": sum(int(row["label"]) == 1 for row in validation_rows), "matched_train_units": sum(int(row["matched_unit_count"]) for row in train_rows), "matched_validation_units": sum(int(row["matched_unit_count"]) for row in validation_rows), "time_contract_checked_windows": time_checked, "nonuniform_interval_windows": nonuniform_intervals}
    return train_rows, validation_rows, compact, info


def _stable_permutation(window_id: str, seed: int = PERMUTATION_SEED) -> list[int]:
    digest = hashlib.sha256(f"{seed}\0{window_id}".encode("utf-8")).digest()
    local_seed = int.from_bytes(digest[:8], "little", signed=False)
    permutation = np.random.default_rng(local_seed).permutation(5).astype(int).tolist()
    if permutation == list(range(5)):
        permutation[-2], permutation[-1] = permutation[-1], permutation[-2]
    if permutation == list(range(5)):
        raise AssertionError("NON_IDENTITY_PERMUTATION_FAILED")
    return permutation


def _make_permutations(root: Path, train_rows: Sequence[Mapping[str, Any]], validation_rows: Sequence[Mapping[str, Any]], *, resume: bool) -> dict[str, list[int]]:
    path = root / "inputs/evaluation_permutation_manifest.json"
    selected = list(train_rows) + list(validation_rows)
    expected = {str(row["window_id"]) for row in selected}
    if resume and path.is_file():
        old = json.loads(path.read_text(encoding="utf-8"))
        if int(old.get("seed", -1)) == PERMUTATION_SEED and {str(item["window_id"]) for item in old.get("windows", [])} == expected:
            result = {str(item["window_id"]): [int(x) for x in item["permutation"]] for item in old["windows"]}
            if all(sorted(value) == list(range(5)) and value != list(range(5)) for value in result.values()):
                return result
    windows = []
    result: dict[str, list[int]] = {}
    train_ids = {str(row["window_id"]) for row in train_rows}
    for row in selected:
        window_id = str(row["window_id"])
        permutation = _stable_permutation(window_id)
        result[window_id] = permutation
        windows.append({"window_id": window_id, "source_id": str(row["source_id"]), "role": str(row["role"]), "split": "train" if window_id in train_ids else "validation", "permutation": permutation, "is_non_identity": True})
    atomic_json(path, {"seed": PERMUTATION_SEED, "definition": "state contents move to fixed increasing time slots; timestamps/interval side-channel are not permuted", "windows": windows})
    return result


def _std_json(value: Mapping[str, Any]) -> dict[str, Any]:
    return {key: (base._jsonable(item) if key in ("mean", "scale") else item) for key, item in value.items()}


def _load_standards(train_rows: Sequence[Mapping[str, Any]], old_manifest: Sequence[Mapping[str, Any]]) -> tuple[FeatureStandardizer, FeatureStandardizer, dict[str, Any]]:
    manifest_by_id = {str(item["window_id"]): item for item in old_manifest}
    arrays = [_load_npz(INPUT_ROOT / str(manifest_by_id[str(row["window_id"])] ["input_path"])) for row in train_rows]
    weights = source_class_weights(train_rows)
    standard_s = fit_standardizer("SET_A", [np.asarray(item["s"], dtype=np.float64) for item in arrays], weights)
    standard_q = fit_standardizer("SET_A", [np.asarray(item["q"], dtype=np.float64) for item in arrays], weights)
    standard_edge = base._fit_edge_standardizer(INPUT_ROOT, old_manifest, train_rows, "edge3d")
    return standard_s, standard_q, standard_edge


def _input_manifest(root: Path, compact: Sequence[Mapping[str, Any]]) -> None:
    atomic_json(root / "inputs/input_manifest.json", {"source_root": str(INPUT_ROOT), "source_feature_manifest": str(INPUT_ROOT / "inputs/feature_manifest.json"), "windows": list(compact)})


def _build_batch(
    rows: Sequence[Mapping[str, Any]],
    old_manifest: Sequence[Mapping[str, Any]],
    standard_s: FeatureStandardizer,
    standard_q: FeatureStandardizer,
    standard_edge: Mapping[str, Any],
    permutations: Mapping[str, Sequence[int]],
    condition: str,
) -> dict[str, Any]:
    by_id = {str(item["window_id"]): item for item in old_manifest}
    s_parts: list[np.ndarray] = []
    q_parts: list[np.ndarray] = []
    edge_parts: list[np.ndarray] = []
    mask_parts: list[np.ndarray] = []
    interval_parts: list[np.ndarray] = []
    time_parts: list[np.ndarray] = []
    window_index: list[int] = []
    labels: list[int] = []
    window_ids: list[str] = []
    for window_index_value, row in enumerate(rows):
        item = by_id[str(row["window_id"])]
        arrays = _load_npz(INPUT_ROOT / str(item["input_path"]))
        s = standard_s.transform(np.asarray(arrays["s"], dtype=np.float64)).astype(np.float32)
        q = standard_q.transform(np.asarray(arrays["q"], dtype=np.float64)).astype(np.float32)
        intervals_one = np.asarray(arrays["intervals"], dtype=np.float32)
        if intervals_one.shape != (4,) or not np.all(np.isfinite(intervals_one)) or np.any(intervals_one <= 0):
            raise ValueError(f"INVALID_INTERVALS:{row['window_id']}")
        tau_one = np.concatenate(([0.0], np.cumsum(intervals_one, dtype=np.float64))).astype(np.float32)
        edge = base._std_array(np.asarray(arrays["edge3d"], dtype=np.float32), standard_edge)
        mask = np.asarray(arrays["edge_mask"], dtype=bool)
        if edge.shape[:3] != mask.shape or edge.shape[0] != s.shape[0] or edge.shape[1] != 5 or q.shape != (s.shape[0], 5, 4):
            raise ValueError(f"BATCH_SHAPE_MISMATCH:{row['window_id']}")
        if condition == "G3D_ORDERED_SHUFFLED":
            permutation = np.asarray(permutations[str(row["window_id"])], dtype=np.int64)
            edge = edge[:, permutation, :, :]
            mask = mask[:, permutation, :]
            q = q[:, permutation, :]
            # tau_one stays in fixed ascending slots by contract.
        s_parts.append(s)
        q_parts.append(q)
        interval_parts.append(np.repeat(intervals_one[None, :], s.shape[0], axis=0))
        time_parts.append(np.repeat(tau_one[None, :], s.shape[0], axis=0))
        edge_parts.append(edge)
        mask_parts.append(mask)
        window_index.extend([window_index_value] * s.shape[0])
        labels.append(int(row["label"]))
        window_ids.append(str(row["window_id"]))
    if not window_ids:
        raise ValueError(f"EMPTY_BATCH:{condition}")
    max_edges = max(int(value.shape[2]) for value in edge_parts)
    padded_edges: list[np.ndarray] = []
    padded_masks: list[np.ndarray] = []
    for edge, mask in zip(edge_parts, mask_parts):
        if edge.shape[2] == max_edges:
            padded_edges.append(edge); padded_masks.append(mask)
        else:
            edge_pad = np.zeros((edge.shape[0], 5, max_edges, edge.shape[3]), dtype=np.float32)
            mask_pad = np.zeros((mask.shape[0], 5, max_edges), dtype=bool)
            edge_pad[:, :, : edge.shape[2]] = edge; mask_pad[:, :, : mask.shape[2]] = mask
            padded_edges.append(edge_pad); padded_masks.append(mask_pad)
    return {
        "edge_values": np.concatenate(padded_edges),
        "edge_mask": np.concatenate(padded_masks),
        "q": np.concatenate(q_parts),
        "intervals": np.concatenate(interval_parts),
        "time_offsets": np.concatenate(time_parts),
        "window_index": np.asarray(window_index, dtype=np.int64),
        "labels": np.asarray(labels, dtype=np.int64),
        "window_ids": window_ids,
        "window_rows": list(rows),
        "window_count": len(rows),
        "window_weights": source_class_weights(rows),
    }


def _slice_batch(batch: Mapping[str, Any], start: int, stop: int) -> dict[str, Any]:
    indices = np.asarray(batch["window_index"], dtype=np.int64)
    keep = (indices >= start) & (indices < stop)
    if not np.any(keep):
        raise ValueError(f"EMPTY_WINDOW_BLOCK:{start}:{stop}")
    result = dict(batch)
    for key in ("edge_values", "edge_mask", "q", "intervals", "time_offsets", "window_index"):
        result[key] = np.asarray(batch[key])[keep].copy()
    result["window_index"] = indices[keep] - start
    result["window_rows"] = list(batch["window_rows"])[start:stop]
    result["window_ids"] = list(batch["window_ids"])[start:stop]
    result["labels"] = np.asarray(batch["labels"])[start:stop].copy()
    result["window_weights"] = np.asarray(batch["window_weights"])[start:stop].copy()
    result["window_count"] = stop - start
    return result


def _initial_model(condition: str, seed: int) -> Any:
    import torch
    torch.manual_seed(int(seed))
    np.random.seed(int(seed) & 0xFFFFFFFF)
    if condition == "G3D_SET":
        from research_tools.v7.component_geometry_pilot.model import ComponentGeometryModel
        return ComponentGeometryModel(7)
    if condition in ("G3D_ORDERED", "G3D_ORDERED_SHUFFLED"):
        return OrderedComponentGeometryModel(7)
    raise ValueError(condition)


def _forward(model: Any, batch: Mapping[str, Any], device: str) -> Any:
    import torch
    outputs: list[torch.Tensor] = []
    indexes: list[torch.Tensor] = []
    total_units = int(np.asarray(batch["window_index"]).shape[0])
    for start in range(0, total_units, UNIT_CHUNK_SIZE):
        stop = min(start + UNIT_CHUNK_SIZE, total_units)
        index = torch.as_tensor(np.asarray(batch["window_index"])[start:stop], dtype=torch.long, device=device)
        edge = torch.as_tensor(np.asarray(batch["edge_values"])[start:stop], dtype=torch.float32, device=device)
        mask = torch.as_tensor(np.asarray(batch["edge_mask"])[start:stop], dtype=torch.bool, device=device)
        q = torch.as_tensor(np.asarray(batch["q"])[start:stop], dtype=torch.float32, device=device)
        intervals = torch.as_tensor(np.asarray(batch["intervals"])[start:stop], dtype=torch.float32, device=device)
        if isinstance(model, OrderedComponentGeometryModel):
            tau = torch.as_tensor(np.asarray(batch["time_offsets"])[start:stop], dtype=torch.float32, device=device)
            local = model(edge, mask, q, tau, intervals)
        else:
            local = model(edge, mask, q, intervals)
        outputs.append(local); indexes.append(index)
    local = torch.cat(outputs)
    index = torch.cat(indexes)
    sums = torch.zeros(int(batch["window_count"]), dtype=local.dtype, device=device)
    counts = torch.zeros_like(sums)
    sums.index_add_(0, index, local)
    counts.index_add_(0, index, torch.ones_like(local))
    return sums / counts.clamp_min(1.0)


def _state_dict_numpy(model: Any) -> dict[str, Any]:
    return {name: value.detach().cpu().numpy().tolist() for name, value in model.state_dict().items()}


def _restore_state(model: Any, state: Mapping[str, Any], device: str) -> Any:
    import torch
    expected = model.state_dict()
    tensor_state = {name: torch.as_tensor(value, dtype=expected[name].dtype) for name, value in state.items()}
    model.load_state_dict(tensor_state)
    model.to(device)
    return model


def _checkpoint_path(root: Path, condition: str, seed: int) -> Path:
    return root / "models/checkpoints" / f"{condition}__seed_{seed}.pt"


def _train_one(root: Path, condition: str, batch: Mapping[str, Any], seed: int, device: str, *, resume: bool) -> tuple[Any, dict[str, Any]]:
    import torch
    import torch.nn.functional as F
    checkpoint = _checkpoint_path(root, condition, seed)
    model = _initial_model(condition, seed).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    history: list[dict[str, Any]] = []
    start_epoch = 1
    resumed_from = 0
    if resume and checkpoint.is_file():
        saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if saved.get("condition") == condition and int(saved.get("seed", -1)) == int(seed) and int(saved.get("epochs_configured", -1)) == EPOCHS:
            model.load_state_dict(saved["model_state"]); optimizer.load_state_dict(saved["optimizer_state"])
            history = list(saved.get("history", [])); resumed_from = int(saved.get("epoch", 0)); start_epoch = resumed_from + 1
            if saved.get("torch_rng_state") is not None: torch.set_rng_state(saved["torch_rng_state"])
            if saved.get("numpy_rng_state") is not None:
                state = saved["numpy_rng_state"]; np.random.set_state((state[0], np.asarray(state[1], dtype=np.uint32), int(state[2]), int(state[3]), float(state[4])))
            if torch.cuda.is_available() and saved.get("cuda_rng_state") is not None: torch.cuda.set_rng_state_all(saved["cuda_rng_state"])
            model.to(device); optimizer_to_device(optimizer, device)
    model.train()
    labels = torch.as_tensor(batch["labels"], dtype=torch.float32, device=device)
    weights = torch.as_tensor(batch["window_weights"], dtype=torch.float32, device=device)
    total_weight = torch.sum(weights)
    for epoch in range(start_epoch, EPOCHS + 1):
        optimizer.zero_grad(set_to_none=True)
        loss_value = 0.0
        for start in range(0, int(batch["window_count"]), WINDOW_TRAIN_CHUNK_SIZE):
            stop = min(start + WINDOW_TRAIN_CHUNK_SIZE, int(batch["window_count"]))
            block = _slice_batch(batch, start, stop)
            scores = _forward(model, block, device)
            loss = torch.sum(F.binary_cross_entropy_with_logits(scores, labels[start:stop], reduction="none") * weights[start:stop]) / total_weight
            if not torch.isfinite(loss):
                raise FloatingPointError(f"NONFINITE_LOSS:{condition}:{seed}:{epoch}")
            loss.backward(); loss_value += float(loss.detach().cpu())
        if not all(parameter.grad is None or torch.all(torch.isfinite(parameter.grad)) for parameter in model.parameters()):
            raise FloatingPointError(f"NONFINITE_GRADIENT:{condition}:{seed}:{epoch}")
        optimizer.step()
        history.append({"epoch": epoch, "loss": loss_value})
        state = {
            "condition": condition, "seed": int(seed), "epoch": epoch, "global_step": epoch,
            "epochs_configured": EPOCHS, "model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(),
            "history": history, "torch_rng_state": torch.get_rng_state(), "numpy_rng_state": np.random.get_state(),
            "cuda_rng_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        }
        checkpoint.parent.mkdir(parents=True, exist_ok=True); temporary = checkpoint.with_suffix(".tmp.pt"); torch.save(state, temporary); os.replace(temporary, checkpoint)
        if epoch in (1, EPOCHS) or epoch % 25 == 0:
            print(f"component-time-order condition={condition} seed={seed} epoch={epoch}/{EPOCHS} loss={loss_value:.6f}", flush=True)
    model.eval()
    if not history:
        raise RuntimeError(f"EMPTY_TRAIN_HISTORY:{condition}:{seed}")
    return model, {"seed": int(seed), "epochs": len(history), "start_epoch": start_epoch, "resumed_from_epoch": resumed_from, "global_step": len(history), "initial_loss": history[0]["loss"], "final_loss": history[-1]["loss"], "min_loss": min(item["loss"] for item in history), "loss_history": history, "device": str(torch.device(device)), "parameter_count": parameter_count(model), "checkpoint": str(checkpoint)}


def optimizer_to_device(optimizer: Any, device: str) -> None:
    for state in optimizer.state.values():
        for key, value in list(state.items()):
            if hasattr(value, "to"):
                state[key] = value.to(device)


def _model_from_record(condition: str, record: Mapping[str, Any], device: str) -> Any:
    return _restore_state(_initial_model(condition, int(record["seed"])), record["state_dict"], device).eval()


def _metrics(rows: Sequence[Mapping[str, Any]], scores: Sequence[float]) -> tuple[dict[str, Any], dict[str, float]]:
    labels = [int(row["label"]) for row in rows]; values = [float(value) for value in scores]
    if not all(math.isfinite(value) for value in values):
        raise ValueError("NONFINITE_SCORE")
    by_source: dict[str, list[tuple[int, float]]] = defaultdict(list)
    for row, value in zip(rows, values): by_source[str(row["source_id"])].append((int(row["label"]), value))
    source_values = {source: periodic._auroc([x[0] for x in pairs], [x[1] for x in pairs]) for source, pairs in by_source.items()}
    source_values = {source: float(value) for source, value in source_values.items() if value is not None}
    return {"window_count": len(rows), "real_count": labels.count(0), "fake_count": labels.count(1), "source_count": len(by_source), "dual_role_source_count": len(source_values), "source_macro": base_source_bootstrap(source_values), "pooled_auroc": periodic._auroc(labels, values), "pooled_ap": periodic._ap(labels, values), **periodic._classification(labels, values)}, source_values


def base_source_bootstrap(values: Mapping[str, float]) -> dict[str, Any]:
    return base.source128._bootstrap(values)


def _metric_row(split: str, condition: str, seed: Any, metrics: Mapping[str, Any]) -> dict[str, Any]:
    ci = metrics.get("source_macro", {}).get("ci95") or [None, None]
    return {"split": split, "condition": condition, "seed": seed, "window_count": metrics["window_count"], "real_count": metrics["real_count"], "fake_count": metrics["fake_count"], "source_count": metrics["source_count"], "dual_role_source_count": metrics["dual_role_source_count"], "source_macro_auroc": metrics["source_macro"].get("mean"), "source_macro_ci_low": ci[0], "source_macro_ci_high": ci[1], "pooled_auroc": metrics.get("pooled_auroc"), "pooled_ap": metrics.get("pooled_ap"), "precision": metrics.get("precision"), "recall": metrics.get("recall"), "f1": metrics.get("f1"), "accuracy": metrics.get("accuracy"), "tn": metrics.get("tn"), "fp": metrics.get("fp"), "fn": metrics.get("fn"), "tp": metrics.get("tp")}


def _identity(info: Mapping[str, Any], old_manifest: Sequence[Mapping[str, Any]], train_rows: Sequence[Mapping[str, Any]], validation_rows: Sequence[Mapping[str, Any]], condition: str, seed: int, standard_s: FeatureStandardizer, standard_q: FeatureStandardizer, standard_edge: Mapping[str, Any], permutations: Mapping[str, Sequence[int]]) -> dict[str, Any]:
    perm_hash = hashlib.sha256(json.dumps({k: list(permutations[k]) for k in sorted(permutations)}, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return {"condition": condition, "seed": int(seed), "input_content_sha256": info["input_content_sha256"], "input_manifest_sha256": info["input_manifest_sha256"], "training_window_ids_sha256": hashlib.sha256("\n".join(sorted(str(row["window_id"]) for row in train_rows)).encode()).hexdigest(), "validation_window_ids_sha256": hashlib.sha256("\n".join(sorted(str(row["window_id"]) for row in validation_rows)).encode()).hexdigest(), "permutation_manifest_sha256": perm_hash, "standard_s": standard_s.as_dict(), "standard_q": standard_q.as_dict(), "standard_edge3d": _std_json(standard_edge), "model_config": {"condition_set": list(CONDITIONS), "epochs": EPOCHS, "learning_rate": LEARNING_RATE, "weight_decay": WEIGHT_DECAY, "window_train_chunk_size": WINDOW_TRAIN_CHUNK_SIZE, "unit_chunk_size": UNIT_CHUNK_SIZE}}


def _record_matches(record: Mapping[str, Any], expected: Mapping[str, Any]) -> bool:
    actual = record.get("input_identity")
    return isinstance(actual, Mapping) and all(actual.get(key) == value for key, value in expected.items())


def _score_rows(rows: Sequence[Mapping[str, Any]], store: Mapping[tuple[str, str, int], Mapping[str, float]], split: str) -> list[dict[str, Any]]:
    output = []
    for row in rows:
        item = {"window_id": str(row["window_id"]), "source_id": str(row["source_id"]), "role": str(row["role"]), "label": int(row["label"]), "matched_unit_count": int(row.get("matched_unit_count", 0))}
        for condition in CONDITIONS:
            for seed in SEEDS: item[f"{condition}_seed_{seed}"] = store[(split, condition, seed)][str(row["window_id"])]
            item[f"{condition}_MEAN_LOGIT"] = float(np.mean([item[f"{condition}_seed_{seed}"] for seed in SEEDS]))
        output.append(item)
    return output


def _evaluate(root: Path, old_manifest: Sequence[Mapping[str, Any]], train_rows: Sequence[Mapping[str, Any]], validation_rows: Sequence[Mapping[str, Any]], records: Sequence[Mapping[str, Any]], standard_s: FeatureStandardizer, standard_q: FeatureStandardizer, standard_edge: Mapping[str, Any], permutations: Mapping[str, Sequence[int]], device: str) -> dict[str, Any]:
    import torch
    store: dict[tuple[str, str, int], dict[str, float]] = {}
    metric_rows: list[dict[str, Any]] = []
    per_source_rows: list[dict[str, Any]] = []
    for condition in CONDITIONS:
        batch_train = _build_batch(train_rows, old_manifest, standard_s, standard_q, standard_edge, permutations, condition)
        batch_val = _build_batch(validation_rows, old_manifest, standard_s, standard_q, standard_edge, permutations, condition)
        for record in [item for item in records if str(item["condition"]) == condition]:
            seed = int(record["seed"]); model = _model_from_record(condition, record, device)
            for split, rows, batch in (("train", train_rows, batch_train), ("validation", validation_rows, batch_val)):
                with torch.no_grad(): scores = _forward(model, batch, device).detach().cpu().numpy().astype(np.float64)
                mapping = {window_id: float(value) for window_id, value in zip(batch["window_ids"], scores)}
                store[(split, condition, seed)] = mapping
                metrics, _ = _metrics(rows, scores); metric_rows.append(_metric_row(split, condition, seed, metrics))
            del model
    mean_store: dict[tuple[str, str], dict[str, float]] = {}
    for split, rows in (("train", train_rows), ("validation", validation_rows)):
        for condition in CONDITIONS:
            mapping = {str(row["window_id"]): float(np.mean([store[(split, condition, seed)][str(row["window_id"])] for seed in SEEDS])) for row in rows}
            mean_store[(split, condition)] = mapping
            metrics, _ = _metrics(rows, [mapping[str(row["window_id"])] for row in rows]); metric_rows.append(_metric_row(split, condition, "MEAN_LOGIT", metrics))
    source_values: dict[str, dict[str, float]] = {}
    for condition in CONDITIONS:
        source_values[condition] = {}
        mapping = mean_store[("validation", condition)]
        for source in sorted({str(row["source_id"]) for row in validation_rows}):
            subset = [row for row in validation_rows if str(row["source_id"]) == source]
            value = periodic._auroc([int(row["label"]) for row in subset], [mapping[str(row["window_id"])] for row in subset])
            if value is not None: source_values[condition][source] = float(value)
    comparisons = []
    for left, right in (("G3D_ORDERED", "G3D_SET"), ("G3D_ORDERED", "G3D_ORDERED_SHUFFLED")):
        result = base.source128._bootstrap(source_values[left], source_values[right])
        differences = [source_values[left][source] - source_values[right][source] for source in result["sources"]]
        result.update({"name": f"{left}-{right}", "left": left, "right": right, "positive_count": sum(v > 0 for v in differences), "negative_count": sum(v < 0 for v in differences), "tie_count": sum(v == 0 for v in differences)})
        comparisons.append(result)
    per_seed_comparisons = []
    for seed in SEEDS:
        seed_values: dict[str, dict[str, float]] = {}
        for condition in CONDITIONS:
            seed_values[condition] = {}
            mapping = store[("validation", condition, seed)]
            for source in sorted({str(row["source_id"]) for row in validation_rows}):
                subset = [row for row in validation_rows if str(row["source_id"]) == source]
                value = periodic._auroc([int(row["label"]) for row in subset], [mapping[str(row["window_id"])] for row in subset])
                if value is not None: seed_values[condition][source] = float(value)
        for left, right in (("G3D_ORDERED", "G3D_SET"), ("G3D_ORDERED", "G3D_ORDERED_SHUFFLED")):
            keys = sorted(set(seed_values[left]) & set(seed_values[right])); diffs = [seed_values[left][source] - seed_values[right][source] for source in keys]
            per_seed_comparisons.append({"seed": int(seed), "comparison": f"{left}-{right}", "source_count": len(diffs), "mean_source_difference": float(np.mean(diffs)) if diffs else None, "positive_count": sum(v > 0 for v in diffs), "negative_count": sum(v < 0 for v in diffs), "tie_count": sum(v == 0 for v in diffs)})
    for split, rows in (("train", train_rows), ("validation", validation_rows)):
        for source in sorted({str(row["source_id"]) for row in rows}):
            source_rows = [row for row in rows if str(row["source_id"]) == source]
            for condition in CONDITIONS:
                for seed in (*SEEDS, "MEAN_LOGIT"):
                    mapping = mean_store[(split, condition)] if seed == "MEAN_LOGIT" else store[(split, condition, int(seed))]
                    values = [mapping[str(row["window_id"])] for row in source_rows]
                    per_source_rows.append({"split": split, "source_id": source, "condition": condition, "seed": seed, "window_count": len(values), "real_count": sum(int(row["label"]) == 0 for row in source_rows), "fake_count": sum(int(row["label"]) == 1 for row in source_rows), "auroc": periodic._auroc([int(row["label"]) for row in source_rows], values)})
    write_csv(root / "scores/train_window_scores.csv", _score_rows(train_rows, store, "train")); write_csv(root / "scores/validation_window_scores.csv", _score_rows(validation_rows, store, "validation"))
    write_csv(root / "evaluation/metrics.csv", metric_rows); write_csv(root / "evaluation/per_source_metrics.csv", per_source_rows); write_csv(root / "evaluation/paired_comparisons.csv", [{"comparison": item["name"], "source_count": item["source_count"], "mean": item["mean"], "ci_low": item["ci95"][0], "ci_high": item["ci95"][1], "positive_count": item["positive_count"], "negative_count": item["negative_count"], "tie_count": item["tie_count"]} for item in comparisons]); write_csv(root / "evaluation/per_seed_comparisons.csv", per_seed_comparisons)
    summary = {"train_population": {"windows": len(train_rows), "real": sum(int(row["label"]) == 0 for row in train_rows), "fake": sum(int(row["label"]) == 1 for row in train_rows), "sources": len({str(row["source_id"]) for row in train_rows})}, "validation_population": {"windows": len(validation_rows), "real": sum(int(row["label"]) == 0 for row in validation_rows), "fake": sum(int(row["label"]) == 1 for row in validation_rows), "sources": len({str(row["source_id"]) for row in validation_rows})}, "metrics": metric_rows, "comparisons": comparisons, "per_seed_comparisons": per_seed_comparisons, "device": str(device), "model_count": len(records), "permutation_seed": PERMUTATION_SEED}
    atomic_json(root / "evaluation/summary.json", summary)
    return summary


def _smoke(root: Path, old_manifest: Sequence[Mapping[str, Any]], rows: Sequence[Mapping[str, Any]], standard_s: FeatureStandardizer, standard_q: FeatureStandardizer, standard_edge: Mapping[str, Any], permutations: Mapping[str, Sequence[int]], device: str) -> dict[str, Any]:
    import torch
    selected = list(rows[: min(8, len(rows))])
    if {int(row["label"]) for row in selected} != {0, 1}:
        raise ValueError("SMOKE_NEEDS_BOTH_CLASSES")
    results = []
    for condition in CONDITIONS:
        batch = _build_batch(selected, old_manifest, standard_s, standard_q, standard_edge, permutations, condition)
        model = _initial_model(condition, SEEDS[0]).to(device); optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY); optimizer.zero_grad(set_to_none=True)
        scores = _forward(model, batch, device); labels = torch.as_tensor(batch["labels"], dtype=torch.float32, device=device); weights = torch.as_tensor(batch["window_weights"], dtype=torch.float32, device=device); loss = torch.sum(torch.nn.functional.binary_cross_entropy_with_logits(scores, labels, reduction="none") * weights) / torch.sum(weights); loss.backward()
        finite_gradients = all(parameter.grad is None or torch.all(torch.isfinite(parameter.grad)) for parameter in model.parameters()); optimizer.step(); model.eval()
        with torch.no_grad(): before = _forward(model, batch, device).detach().cpu().numpy()
        reload = _restore_state(_initial_model(condition, SEEDS[0]), {name: value.detach().cpu().numpy().tolist() for name, value in model.state_dict().items()}, device).eval()
        with torch.no_grad(): after = _forward(reload, batch, device).detach().cpu().numpy()
        results.append({"condition": condition, "parameter_count": parameter_count(model), "loss_finite": bool(torch.isfinite(loss).item()), "gradient_finite": bool(finite_gradients), "reload_max_abs": float(np.max(np.abs(before - after))), "passed": bool(torch.isfinite(loss).item() and finite_gradients and np.max(np.abs(before - after)) <= 1e-5)})
    # Structural order check uses a deterministic asymmetric synthetic state.
    synthetic_edge = np.zeros((1, 5, 1, 7), dtype=np.float32); synthetic_q = np.zeros((1, 5, 4), dtype=np.float32); synthetic_mask = np.ones((1, 5, 1), dtype=bool); synthetic_tau = np.arange(5, dtype=np.float32)[None, :]; synthetic_intervals = np.ones((1, 4), dtype=np.float32)
    for t in range(5): synthetic_edge[0, t, 0, 0] = float(t + 1); synthetic_q[0, t, 0] = float((t + 1) * 0.25)
    ordered = _initial_model("G3D_ORDERED", SEEDS[0]).to(device).eval()
    with torch.no_grad():
        forward_a = ordered(torch.as_tensor(synthetic_edge, device=device), torch.as_tensor(synthetic_mask, device=device), torch.as_tensor(synthetic_q, device=device), torch.as_tensor(synthetic_tau, device=device), torch.as_tensor(synthetic_intervals, device=device)).item(); reverse = ordered(torch.as_tensor(np.ascontiguousarray(synthetic_edge[:, ::-1]), device=device), torch.as_tensor(np.ascontiguousarray(synthetic_mask[:, ::-1]), device=device), torch.as_tensor(np.ascontiguousarray(synthetic_q[:, ::-1]), device=device), torch.as_tensor(synthetic_tau, device=device), torch.as_tensor(synthetic_intervals, device=device)).item()
    results.append({"condition": "ORDER_SENSITIVITY_SYNTHETIC", "forward": forward_a, "reversed": reverse, "absolute_difference": abs(forward_a - reverse), "passed": bool(abs(forward_a - reverse) > 1e-9)})
    result = {"status": "PASS" if all(item["passed"] for item in results) else "FAILED", "rows": len(selected), "conditions": results, "device": str(device), "note": "one-step finite gradient and reload plus deterministic order-sensitive synthetic check"}
    atomic_json(root / "smoke/summary.json", result)
    return result


def _write_protocol(root: Path, info: Mapping[str, Any], compact: Sequence[Mapping[str, Any]], permutations: Mapping[str, Sequence[int]], standard_s: FeatureStandardizer, standard_q: FeatureStandardizer, standard_edge: Mapping[str, Any], device: str) -> None:
    import torch
    atomic_json(root / "protocol.json", {"protocol_id": "v7-component-geometry-time-order-pilot-v1", "git_head": _git_head(), "input_root": str(INPUT_ROOT), "input": {"train_windows": info["training_windows"], "validation_windows": info["validation_windows"], "training_sources": info["training_sources"], "validation_sources": info["validation_sources"], "train_real": info["training_real"], "train_fake": info["training_fake"], "validation_real": info["validation_real"], "validation_fake": info["validation_fake"], "matched_train_units": info["matched_train_units"], "matched_validation_units": info["matched_validation_units"], "input_manifest_sha256": info["input_manifest_sha256"], "input_content_sha256": info["input_content_sha256"], "time_contract_checked_windows": info["time_contract_checked_windows"], "nonuniform_interval_windows": info["nonuniform_interval_windows"]}, "conditions": {"G3D_SET": "existing matched 3-D fixed-edge encoder plus Q; per-time encoded states mean-pooled over five time positions; real PTS intervals remain side-channel", "G3D_ORDERED": "same fixed-edge spatial encoder and Q; per-time state a_t concatenated with tau_t=t_t-t_0, flattened in ascending target PTS order, Linear(45,8)-ReLU projection, same interval side-channel/head", "G3D_ORDERED_SHUFFLED": "same ordered model; edge values, edge masks and Q synchronously permuted per window into fixed ascending slots; tau/interval side-channel is not permuted; deterministic non-identity permutation"}, "support_contract": {"fixed_edges": "existing directed history edges and masks from matched feature cache", "q": "existing four-channel Q, identical for all conditions except synchronized state permutation in C", "unit_window_aggregation": "equal mean of local unit logits; no local labels or unit weighting", "missing": "explicit edge mask; no zero-filled observation participates"}, "time": {"target_slots": 5, "tau_definition": "[0, cumulative sum of four stored positive PTS intervals]", "tau_unit": "seconds", "ordered_slot_definition": "fixed ascending target PTS slots", "shuffle_definition": "state contents move to fixed slots; timestamps/intervals remain fixed", "permutation_seed": PERMUTATION_SEED, "permutation_manifest": str(root / "inputs/evaluation_permutation_manifest.json")}, "model": {"spatial_encoder": "Linear(7,32)-ReLU-Linear(32,64)-ReLU; masked edge mean; concat Q; Linear(68,16)-ReLU-Linear(16,8)-ReLU", "set_temporal": "mean over five 8-D states", "ordered_temporal": "concat five (8-D state + 1-D tau) slots -> Linear(45,8)-ReLU", "head": "Linear(12,16)-ReLU-Linear(16,8)-ReLU-Linear(8,1), intervals concatenated", "parameter_counts": {"G3D_SET": parameter_count(_initial_model("G3D_SET", SEEDS[0])), "G3D_ORDERED": parameter_count(_initial_model("G3D_ORDERED", SEEDS[0])), "G3D_ORDERED_SHUFFLED": parameter_count(_initial_model("G3D_ORDERED_SHUFFLED", SEEDS[0]))}, "epochs": EPOCHS, "optimizer": "Adam", "learning_rate": LEARNING_RATE, "weight_decay": WEIGHT_DECAY, "seeds": list(SEEDS), "threshold": "logit >= 0", "loss": "source/class weighted BCE at window level", "unit_chunk_size": UNIT_CHUNK_SIZE, "window_train_chunk_size": WINDOW_TRAIN_CHUNK_SIZE, "optimizer_step": "one step per epoch after exact window-block gradient accumulation", "checkpoint": {"optimizer_state": True, "scheduler_state": False, "rng_state": True, "epoch_boundary": True}}, "standardization": {"summary_s": standard_s.as_dict(), "q": standard_q.as_dict(), "edge3d": _std_json(standard_edge), "time_offsets": "raw seconds; no validation-derived scaling"}, "evaluation": {"primary": "G3D_ORDERED - G3D_SET source-macro AUROC", "secondary": "G3D_ORDERED - G3D_ORDERED_SHUFFLED source-macro AUROC", "bootstrap_seed": BOOTSTRAP_SEED, "bootstrap_replicates": BOOTSTRAP_REPLICATES, "mean_logits_before_metrics": True, "not_sealed_test": True}, "runtime": {"torch_version": str(torch.__version__), "cuda_available": bool(torch.cuda.is_available()), "cuda_version": str(torch.version.cuda), "device": str(device)}, "source_input_rows": len(compact)})


def _write_report(root: Path, info: Mapping[str, Any], summary: Mapping[str, Any], records: Sequence[Mapping[str, Any]], timings: Mapping[str, Any], smoke: Mapping[str, Any]) -> None:
    metrics = {(str(item["split"]), str(item["condition"])): item for item in summary.get("metrics", []) if str(item["seed"]) == "MEAN_LOGIT"}
    lines = ["# V7 component 可学习几何表示最小时间顺序对照", "", "本 pilot 固定已有 3-D component、历史固定边、五时刻共同支撑、Q、source-disjoint 窗口、窗口级 weighted BCE 和 unit 等权聚合；只比较 raw component geometry 的五时刻无序集合（G3D_SET）、真实 PTS 槽位有序编码（G3D_ORDERED）及同架构状态置换控制（G3D_ORDERED_SHUFFLED）。它不是首次时序研究，也不等价于固定单条 edge 的跨时刻轨迹模型。", "", "## 结论先行"]
    lines.append(f"- 正式模型 {len(records)}/9；smoke={smoke.get('status')}；设备={summary.get('device')}。")
    lines.append(f"- 训练 {info['training_windows']} 窗口/{len(info['training_sources'])} source（{info['training_real']} real、{info['training_fake']} fake，{info['matched_train_units']} matched units）；验证 {info['validation_windows']} 窗口/{len(info['validation_sources'])} source（{info['validation_real']} real、{info['validation_fake']} fake，{info['matched_validation_units']} matched units）。输入时间合同核对 {info['time_contract_checked_windows']} 窗口，其中 {info['nonuniform_interval_windows']} 个窗口的相邻 PTS 间隔不等。")
    lines.extend(["", "| split/condition | source-macro AUROC (95% CI) | pooled AUROC | AP | P | R | F1 | ACC | TN/FP/FN/TP |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|"])
    for split in ("train", "validation"):
        for condition in CONDITIONS:
            item = metrics.get((split, condition), {}); ci = [item.get("source_macro_ci_low"), item.get("source_macro_ci_high")]
            lines.append(f"| {split}/{condition} | {item.get('source_macro_auroc')} [{ci[0]}, {ci[1]}] | {item.get('pooled_auroc')} | {item.get('pooled_ap')} | {item.get('precision')} | {item.get('recall')} | {item.get('f1')} | {item.get('accuracy')} | {item.get('tn')}/{item.get('fp')}/{item.get('fn')}/{item.get('tp')} |")
    lines.extend(["", "## 主要配对", "", "| 比较 | source 数 | 均值差 | 95% CI | 正/平/负 source |", "|---|---:|---:|---|---:|"])
    for item in summary.get("comparisons", []):
        lines.append(f"| {item['name']} | {item['source_count']} | {item['mean']} | [{item['ci95'][0]}, {item['ci95'][1]}] | {item['positive_count']}/{item['tie_count']}/{item['negative_count']} |")
    lines.extend(["", "## 与当前分支已有实验的差别", "", "| 既有实验 | 主要输入/时间机制 | 与本 pilot 的关系 |", "|---|---|---|", "| `multi_order_sequence` | 展平的摘要序列，显式有序/打乱条件 | 有序摘要对照，不是 raw fixed-edge geometry。", "| `pair_trajectory` | 固定 pair 五时刻距离及 ID/time shuffle | 固定关系轨迹对照，不是本 pilot 的可学习逐时刻 edge 集合。", "| `relation_first` | 关系聚合后再做差分/表示 | 关系优先统计，不是本 pilot 的时间槽投影。", "| `local_context` | 邻域/重连上下文与时间控制 | 改变局部上下文或关系规则，非本 pilot 的固定边匹配。", "", "## 时间实现与解释边界", "", "- A 的 raw edge/Q 状态在五个时间位置做均值，因此对同步时间置换保持不变；四个原始 PTS 间隔仍作为独立有序旁路。", "- B 使用 `tau_t=t_t-t_0`（秒）与每个时刻的空间状态拼接，再按真实递增 PTS 槽位固定拼接并投影。C 使用完全相同的有序模型和参数量，但将 edge、mask、Q 的状态内容按窗口级冻结非恒等排列放入固定时间槽；tau/intervals 不被重排。", "- B−A 同时包含有序投影带来的容量/归纳偏置变化；B−C 是主要的时间对应对照，但 C 也制造了不自然的内容—时间槽过渡，且 Q 的值仍保留。", "- 先沿固定 K 边聚合为每时刻 component 表示，再做时间编码；本 pilot 没有保留同一条 edge 的五时刻轨迹，也没有组间关系、中心轨迹、视觉输入或像素真值。", "", "## 训练与恢复", f"- 参数量：G3D_SET=3961，G3D_ORDERED=4329，G3D_ORDERED_SHUFFLED=4329。三条件均从头训练 3 seeds × {EPOCHS} epochs，Adam lr={LEARNING_RATE}、weight_decay={WEIGHT_DECAY}；unit chunk={UNIT_CHUNK_SIZE}，window block={WINDOW_TRAIN_CHUNK_SIZE}，每 epoch 一次 optimizer step。每个 model checkpoint 保存 optimizer、epoch/global_step、Python/NumPy/PyTorch/CUDA RNG（若可用）和连续 loss。", "- 训练前冻结唯一 permutation manifest；训练和评价共享同一窗口级置换。标准化仅使用训练 source。", f"- 当前报告由最后一次 resume 调用生成；本次调用复用 9/9 完整模型，因而 `train={timings.get('train')}s` 不是首次完整训练成本。首次正式 run 的逐 epoch日志和 checkpoint 保留在 `run.log` 与 `models/checkpoints/`；resume 阶段只做身份核对、评分和报告重写。阶段耗时（当前调用，秒）：inputs={timings.get('inputs')}, smoke={timings.get('smoke')}, train={timings.get('train')}, evaluate={timings.get('evaluate')}, report={timings.get('report')}。", "", "## 结论边界", "- 若 B 同时高于 A 和 C，可支持在当前匹配边界继续研究 raw geometry 的有序时间编码；不能称为物理机制、固定关系对应、空间定位或未知 source 泛化证明。若只高于 A，不能把收益归因于正确顺序；若不高于 C，当前设置未建立顺序增益。", "- 本实验为多轮开发集上的小规模对照；source bootstrap CI 只描述不确定性，不把重叠窗口或 pair 数当独立样本。", "- 未重跑 tracking/depth/pose/segmentation，旧 component geometry 缓存作为输入；G3D_SET 是本目录新训匹配基线，旧 component pilot 仅作历史参考。"])
    (root / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(root: Path = OUTPUT_ROOT, *, device: str = "cuda", resume: bool = True) -> dict[str, Any]:
    root.mkdir(parents=True, exist_ok=True); (root / "state").mkdir(parents=True, exist_ok=True); started = time.perf_counter(); timings: dict[str, Any] = {}
    atomic_json(root / "state/launch.json", {"git_head": _git_head(), "pid": os.getpid(), "device": device, "started_unix": time.time(), "status": "RUNNING"})
    _progress(root, "inputs", "RUNNING", 0, 1)
    try:
        input_started = time.perf_counter(); train_rows, validation_rows, compact, info = _load_inputs(); old_manifest = json.loads((INPUT_ROOT / "inputs/feature_manifest.json").read_text(encoding="utf-8")); _input_manifest(root, compact); permutations = _make_permutations(root, train_rows, validation_rows, resume=resume); standard_s, standard_q, standard_edge = _load_standards(train_rows, old_manifest); _write_protocol(root, info, compact, permutations, standard_s, standard_q, standard_edge, device); timings["inputs"] = time.perf_counter() - input_started; _progress(root, "inputs", "COMPLETE", 1, 1, train_windows=len(train_rows), validation_windows=len(validation_rows))
        smoke_started = time.perf_counter(); smoke_path = root / "smoke/summary.json"; smoke = json.loads(smoke_path.read_text(encoding="utf-8")) if resume and smoke_path.is_file() else _smoke(root, old_manifest, train_rows, standard_s, standard_q, standard_edge, permutations, device); timings["smoke"] = time.perf_counter() - smoke_started
        if smoke.get("status") != "PASS": raise RuntimeError("SMOKE_FAILED")
        model_path = root / "models/fold_models.json"; existing = json.loads(model_path.read_text(encoding="utf-8")).get("records", []) if resume and model_path.is_file() else []; records: list[dict[str, Any]] = []; complete: set[tuple[str, int]] = set(); total = len(CONDITIONS) * len(SEEDS)
        for old in existing:
            condition, seed = str(old.get("condition", "")), int(old.get("seed", -1))
            if condition not in CONDITIONS or seed not in SEEDS: continue
            expected = _identity(info, old_manifest, train_rows, validation_rows, condition, seed, standard_s, standard_q, standard_edge, permutations)
            if old.get("status") == "TRAIN_COMPLETE" and _record_matches(old, expected) and int(old.get("fit", {}).get("epochs", -1)) == EPOCHS:
                records.append(old); complete.add((condition, seed))
        train_started = time.perf_counter()
        for condition in CONDITIONS:
            batch = _build_batch(train_rows, old_manifest, standard_s, standard_q, standard_edge, permutations, condition)
            for seed in SEEDS:
                if (condition, int(seed)) in complete: continue
                model, fit = _train_one(root, condition, batch, int(seed), device, resume=resume)
                record = {"condition": condition, "seed": int(seed), "status": "TRAIN_COMPLETE", "parameter_count": parameter_count(model), "fit": fit, "input_identity": _identity(info, old_manifest, train_rows, validation_rows, condition, int(seed), standard_s, standard_q, standard_edge, permutations), "state_dict": _state_dict_numpy(model)}
                records.append(record); complete.add((condition, int(seed))); atomic_json(model_path, {"conditions": list(CONDITIONS), "seeds": list(SEEDS), "epochs": EPOCHS, "records": records}); _progress(root, "train", "RUNNING", len(complete), total, condition=condition, seed=seed); del model
        timings["train"] = time.perf_counter() - train_started
        if len(complete) != total: raise RuntimeError(f"MODEL_COUNT_INCOMPLETE:{len(complete)}/{total}")
        evaluate_started = time.perf_counter(); summary = _evaluate(root, old_manifest, train_rows, validation_rows, records, standard_s, standard_q, standard_edge, permutations, device); timings["evaluate"] = time.perf_counter() - evaluate_started
        report_started = time.perf_counter(); _write_report(root, info, summary, records, timings, smoke); timings["report"] = time.perf_counter() - report_started; _write_report(root, info, summary, records, timings, smoke)
        atomic_json(root / "final_status.json", {"status": "COMPLETE", "model_count": len(records), "expected_model_count": total, "train_windows": len(train_rows), "validation_windows": len(validation_rows), "report": str(root / "report.md"), "timings_s": timings, "git_head": _git_head()}); _progress(root, "report", "COMPLETE", 1, 1, model_count=len(records), elapsed_s=time.perf_counter() - started); atomic_json(root / "state/launch.json", {"git_head": _git_head(), "pid": os.getpid(), "device": device, "started_unix": json.loads((root / "state/launch.json").read_text())["started_unix"], "status": "COMPLETE", "ended_unix": time.time()}); return {"status": "COMPLETE", "summary": summary, "timings_s": timings}
    except BaseException as exc:
        atomic_json(root / "state/failure.json", {"status": "FAILED", "error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc(), "elapsed_s": time.perf_counter() - started}); atomic_json(root / "final_status.json", {"status": "FAILED", "error": f"{type(exc).__name__}: {exc}", "git_head": _git_head(), "elapsed_s": time.perf_counter() - started}); _progress(root, "failed", "FAILED", 0, 1, error=f"{type(exc).__name__}: {exc}"); raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT); parser.add_argument("--device", default="cuda"); parser.add_argument("--resume", action="store_true"); args = parser.parse_args(); result = run(args.output_root, device=args.device, resume=args.resume); print(json.dumps({"status": result["status"], "timings_s": result["timings_s"]}, ensure_ascii=False, indent=2)); return 0


if __name__ == "__main__":
    raise SystemExit(main())
