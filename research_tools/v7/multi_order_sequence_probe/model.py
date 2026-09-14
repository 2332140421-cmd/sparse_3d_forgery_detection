"""Small fixed-budget models for the five-time sequence pilot."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import nn

from .representation import CONDITIONS


SEEDS = (20260909, 20260910, 20260911)
EPOCHS = 200
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4


def source_class_weights(rows: Sequence[Mapping[str, Any]]) -> np.ndarray:
    """Equal total weight per source/class, then normalize mean to one."""

    if not rows:
        raise ValueError("empty training rows")
    keys = [(str(row["source_id"]), int(row["label"])) for row in rows]
    if any(label not in (0, 1) for _, label in keys):
        raise ValueError("training labels must be 0/1")
    sources = sorted({source for source, _ in keys})
    counts: dict[tuple[str, int], int] = {}
    for key in keys:
        counts[key] = counts.get(key, 0) + 1
    values = np.asarray([1.0 / (2 * len(sources) * counts[key]) for key in keys], dtype=np.float64)
    return values * (values.size / values.sum())


@dataclass(frozen=True)
class FeatureStandardizer:
    condition: str
    mean: np.ndarray
    scale: np.ndarray
    zero_variance_dimensions: tuple[int, ...]

    def transform(self, values: np.ndarray) -> np.ndarray:
        array = np.asarray(values, dtype=np.float64)
        if self.condition == "SET_A":
            if array.ndim != 3 or array.shape[1:] != (5, 4):
                raise ValueError("SET_A values must have shape [U,5,4]")
            return (array - self.mean[None, None, :]) / self.scale[None, None, :]
        if array.ndim != 2 or array.shape[1] != 40:
            raise ValueError("sequence values must have shape [U,40]")
        output = array.copy()
        if self.condition == "RAW_SEQ":
            output[:, :36] = ((output[:, :36].reshape(-1, 9, 4) - self.mean[:4]) / self.scale[:4]).reshape(-1, 36)
        else:
            output[:, :12] = ((output[:, :12].reshape(-1, 3, 4) - self.mean[:4]) / self.scale[:4]).reshape(-1, 12)
            output[:, 12:24] = ((output[:, 12:24].reshape(-1, 3, 4) - self.mean[4:8]) / self.scale[4:8]).reshape(-1, 12)
            output[:, 24:36] = ((output[:, 24:36].reshape(-1, 3, 4) - self.mean[8:12]) / self.scale[8:12]).reshape(-1, 12)
        output[:, 36:40] = (output[:, 36:40] - self.mean[-4:]) / self.scale[-4:]
        return output

    def as_dict(self) -> dict[str, Any]:
        return {"condition": self.condition, "mean": self.mean.tolist(), "scale": self.scale.tolist(), "zero_variance_dimensions": list(self.zero_variance_dimensions)}


def fit_standardizer(condition: str, matrices: Sequence[np.ndarray], weights: np.ndarray) -> FeatureStandardizer:
    if not matrices:
        raise ValueError("no training features")
    values = np.concatenate([np.asarray(item, dtype=np.float64) for item in matrices], axis=0)
    unit_weights = np.repeat(np.asarray(weights, dtype=np.float64), [np.asarray(item).shape[0] for item in matrices])
    if condition == "SET_A":
        flat = values.reshape(-1, 4)
        row_weights = np.repeat(unit_weights, 5)
    elif condition == "RAW_SEQ":
        flat = values[:, :36].reshape(-1, 4)
        row_weights = np.repeat(unit_weights, 9)
    else:
        blocks = [values[:, :12].reshape(-1, 4), values[:, 12:24].reshape(-1, 4), values[:, 24:36].reshape(-1, 4)]
        flat = np.concatenate(blocks, axis=0)
        # The three channel blocks are concatenated block-wise (all S, then
        # all v, then all a), so repeat within each block and tile blocks.
        row_weights = np.tile(np.repeat(unit_weights, 3), 3)
    # h is included as four independently standardized seconds channels.
    if condition == "SET_A":
        h = np.zeros((values.shape[0], 4), dtype=np.float64)
    else:
        h = values[:, 36:40]
    # For SET_A h is appended by the feature batch and supplied separately;
    # its standardized state mean is still enough for the state encoder.
    if condition == "SET_A":
        raw_mean = np.average(flat, axis=0, weights=row_weights)
        raw_var = np.average((flat - raw_mean) ** 2, axis=0, weights=row_weights)
        mean = raw_mean
    elif condition == "RAW_SEQ":
        state_mean = np.average(flat, axis=0, weights=row_weights)
        state_var = np.average((flat - state_mean) ** 2, axis=0, weights=row_weights)
        h_weights = unit_weights
        h_mean = np.average(h, axis=0, weights=h_weights)
        h_var = np.average((h - h_mean) ** 2, axis=0, weights=h_weights)
        mean = np.concatenate((state_mean, h_mean)); raw_var = np.concatenate((state_var, h_var))
    else:
        block_weights = np.repeat(unit_weights, 3)
        block_means = [np.average(block, axis=0, weights=block_weights) for block in blocks]
        block_vars = [np.average((block - mean) ** 2, axis=0, weights=block_weights) for block, mean in zip(blocks, block_means)]
        state_mean = np.concatenate(block_means)
        state_var = np.concatenate(block_vars)
        h_weights = unit_weights
        h_mean = np.average(h, axis=0, weights=h_weights)
        h_var = np.average((h - h_mean) ** 2, axis=0, weights=h_weights)
        mean = np.concatenate((state_mean, h_mean)); raw_var = np.concatenate((state_var, h_var))
    zero = tuple(int(x) for x in np.flatnonzero(~np.isfinite(raw_var) | (raw_var <= 1e-12)))
    scale = np.sqrt(np.maximum(raw_var, 1e-12)); scale[list(zero)] = 1.0
    return FeatureStandardizer(condition, np.asarray(mean, dtype=np.float64), scale, zero)


class SequenceMLP(nn.Module):
    """40->16->8->1 network shared by all three sequence conditions."""

    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(40, 16), nn.ReLU(), nn.Linear(16, 8), nn.ReLU(), nn.Linear(8, 1))

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.net(values).squeeze(-1)


class SetAModel(nn.Module):
    """Shared state encoder, time mean, then interval-aware head."""

    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(4, 16), nn.ReLU(), nn.Linear(16, 8), nn.ReLU())
        self.head = nn.Sequential(nn.Linear(12, 16), nn.ReLU(), nn.Linear(16, 8), nn.ReLU(), nn.Linear(8, 1))

    def forward(self, states: torch.Tensor, intervals: torch.Tensor) -> torch.Tensor:
        encoded = self.encoder(states).mean(dim=1)
        return self.head(torch.cat((encoded, intervals), dim=-1)).squeeze(-1)


def parameter_count(condition: str) -> int:
    model: nn.Module = SetAModel() if condition == "SET_A" else SequenceMLP()
    return int(sum(parameter.numel() for parameter in model.parameters()))


def _window_aggregate(logits: torch.Tensor, unit_window_index: torch.Tensor, window_count: int) -> torch.Tensor:
    sums = torch.zeros(window_count, dtype=logits.dtype, device=logits.device)
    counts = torch.zeros(window_count, dtype=logits.dtype, device=logits.device)
    sums.index_add_(0, unit_window_index, logits)
    counts.index_add_(0, unit_window_index, torch.ones_like(logits))
    return sums / counts.clamp_min(1.0)


def _epoch_metrics(labels: np.ndarray, scores: np.ndarray) -> dict[str, float | None]:
    positives = scores[labels == 1]; negatives = scores[labels == 0]
    auc = float(np.mean((positives[:, None] > negatives[None, :]) + 0.5 * (positives[:, None] == negatives[None, :]))) if positives.size and negatives.size else None
    if positives.size:
        order = np.argsort(-scores, kind="mergesort"); ordered_labels = labels[order]; tp_curve = np.cumsum(ordered_labels == 1); precision_curve = tp_curve / np.arange(1, len(ordered_labels) + 1); average_precision = float(np.sum(precision_curve[ordered_labels == 1]) / positives.size)
    else:
        average_precision = None
    pred = scores >= 0.0; tp = int(np.sum((labels == 1) & pred)); fp = int(np.sum((labels == 0) & pred)); fn = int(np.sum((labels == 1) & ~pred))
    precision = tp / (tp + fp) if tp + fp else None; recall = tp / (tp + fn) if tp + fn else None
    f1 = 2 * precision * recall / (precision + recall) if precision is not None and recall is not None and precision + recall else None
    return {"train_auroc": auc, "train_ap": average_precision, "train_f1": f1}


def make_batch(condition: str, windows: Sequence[Mapping[str, Any]], standardizer: FeatureStandardizer | None = None, *, require_labels: bool = True) -> dict[str, Any]:
    if condition not in CONDITIONS:
        raise ValueError(condition)
    matrices: list[np.ndarray] = []
    indices: list[int] = []
    labels: list[int] = []
    window_ids: list[str] = []
    for window_index, row in enumerate(windows):
        matrix = np.asarray(row["features"][condition], dtype=np.float64)
        if matrix.ndim not in (2, 3) or matrix.shape[0] == 0:
            continue
        if not np.all(np.isfinite(matrix)):
            raise ValueError(f"nonfinite feature: {row.get('window_id')}")
        matrices.append(matrix)
        indices.extend([len(labels)] * matrix.shape[0])
        if row.get("label") is None:
            if require_labels:
                raise ValueError("labels required for training batch")
            labels.append(-1)
        else:
            labels.append(int(row["label"]))
        window_ids.append(str(row["window_id"]))
    if not matrices:
        raise ValueError("no valid windows")
    matrix = np.concatenate(matrices, axis=0)
    if standardizer is not None:
        matrix = standardizer.transform(matrix)
    intervals = np.concatenate(
        [np.repeat(np.asarray(row["intervals_s"], dtype=np.float64)[None, :], np.asarray(row["features"][condition]).shape[0], axis=0) for row in windows if np.asarray(row["features"][condition]).ndim in (2, 3) and np.asarray(row["features"][condition]).shape[0] > 0],
        axis=0,
    ) if condition == "SET_A" else None
    return {"features": matrix, "intervals": intervals, "unit_window_index": np.asarray(indices, dtype=np.int64), "labels": np.asarray(labels, dtype=np.int64), "window_ids": window_ids, "window_count": len(labels), "unit_count": int(matrix.shape[0])}


def train_one(condition: str, batch: Mapping[str, Any], *, seed: int, device: str, epochs: int = EPOCHS, epoch_callback: Any | None = None) -> tuple[nn.Module, dict[str, Any]]:
    torch.manual_seed(int(seed)); np.random.seed(int(seed) & 0xFFFFFFFF)
    target_device = torch.device(device)
    if target_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA_UNAVAILABLE")
    model: nn.Module = SetAModel() if condition == "SET_A" else SequenceMLP()
    model.to(target_device); model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    values = torch.as_tensor(batch["features"], dtype=torch.float32, device=target_device)
    indices = torch.as_tensor(batch["unit_window_index"], dtype=torch.long, device=target_device)
    labels = torch.as_tensor(batch["labels"], dtype=torch.float32, device=target_device)
    weights = torch.as_tensor(batch["window_weights"], dtype=torch.float32, device=target_device)
    if condition == "SET_A":
        states, intervals = values, torch.zeros((values.shape[0], 4), dtype=values.dtype, device=target_device)
        # intervals are supplied as a separate tensor by make_training_batch.
        intervals = torch.as_tensor(batch["intervals"], dtype=torch.float32, device=target_device)
    history: list[dict[str, float]] = []
    for epoch in range(1, int(epochs) + 1):
        optimizer.zero_grad(set_to_none=True)
        if condition == "SET_A":
            unit_logits = model(states, intervals)
        else:
            unit_logits = model(values)
        window_logits = _window_aggregate(unit_logits, indices, int(batch["window_count"]))
        loss = torch.sum(nn.functional.binary_cross_entropy_with_logits(window_logits, labels, reduction="none") * weights) / torch.sum(weights)
        loss.backward(); optimizer.step()
        loss_value = float(loss.detach().cpu())
        metric_row = _epoch_metrics(batch["labels"], window_logits.detach().cpu().numpy())
        history.append({"epoch": epoch, "loss": loss_value, **metric_row})
        if epoch_callback is not None:
            epoch_callback(epoch, history[-1])
    model.eval()
    return model, {"seed": int(seed), "epochs": int(epochs), "initial_loss": history[0]["loss"], "final_loss": history[-1]["loss"], "min_loss": min(item["loss"] for item in history), "loss_history": history, "device": str(target_device), "parameter_count": parameter_count(condition)}


def score_batch(condition: str, model: nn.Module, batch: Mapping[str, Any], device: str) -> np.ndarray:
    target_device = next(model.parameters()).device
    values = torch.as_tensor(batch["features"], dtype=torch.float32, device=target_device)
    indices = torch.as_tensor(batch["unit_window_index"], dtype=torch.long, device=target_device)
    with torch.no_grad():
        if condition == "SET_A":
            intervals = torch.as_tensor(batch["intervals"], dtype=torch.float32, device=target_device)
            unit_logits = model(values, intervals)
        else:
            unit_logits = model(values)
        return _window_aggregate(unit_logits, indices, int(batch["window_count"])).detach().cpu().numpy().astype(np.float64)


def model_state(model: nn.Module) -> dict[str, Any]:
    return {name: value.detach().cpu().numpy().tolist() for name, value in model.state_dict().items()}
