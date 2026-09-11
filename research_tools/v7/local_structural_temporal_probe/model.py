"""Small MLP and source-disjoint evaluation helpers for the V7 pilot."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
from torch import nn

from .representation import ARM_NAMES, arm_inputs_for_triplet


MODEL_CONFIG = {
    "architecture": "shared_mlp_12_16_8_plus_linear_head",
    "input_dim": 12,
    "hidden_dim": 16,
    "representation_dim": 8,
    "activation": "ReLU",
    "optimizer": "Adam",
    "learning_rate": 1e-3,
    "weight_decay": 1e-4,
    "epochs": 200,
    "full_batch": True,
}
STANDARDIZATION_VARIANCE_FLOOR = 1e-12


def window_label(example: Mapping[str, Any]) -> int | None:
    """Map only the preselected window role to the pilot's training label."""

    if str(example.get("kind")) != "MANIP":
        return None
    role = str(example.get("role"))
    if role == "real":
        return 0
    if role == "fake":
        return 1
    return None


def validate_training_examples(examples: Sequence[Mapping[str, Any]]) -> None:
    if not examples:
        raise ValueError("training examples are empty")
    source_classes: dict[str, set[int]] = {}
    for example in examples:
        label = window_label(example)
        if label is None:
            raise ValueError("training examples must be real/fake MANIP windows")
        source_classes.setdefault(str(example["source_id"]), set()).add(int(label))
    if any(classes != {0, 1} for classes in source_classes.values()):
        raise ValueError("every training source must have both real and fake MANIP windows")


def source_class_window_weights(examples: Sequence[Mapping[str, Any]]) -> np.ndarray:
    """Give every source/class group equal weight, then normalize mean to one."""

    validate_training_examples(examples)
    sources = sorted({str(example["source_id"]) for example in examples})
    counts: dict[tuple[str, int], int] = {}
    for example in examples:
        key = (str(example["source_id"]), int(window_label(example)))
        counts[key] = counts.get(key, 0) + 1
    values = np.asarray(
        [
            1.0 / (2.0 * len(sources) * counts[(str(example["source_id"]), int(window_label(example)))])
            for example in examples
        ],
        dtype=np.float64,
    )
    return values * (values.size / np.sum(values))


@dataclass(frozen=True)
class WeightedStandardizer:
    mean: np.ndarray
    scale: np.ndarray
    zero_variance_dimensions: tuple[int, ...]

    def transform(self, values: Iterable[Iterable[float]]) -> np.ndarray:
        array = np.asarray(values, dtype=np.float64)
        if array.ndim != 2 or array.shape[1] != 12:
            raise ValueError("pilot input matrix must have shape [N,12]")
        if not np.all(np.isfinite(array)):
            raise ValueError("pilot input matrix must be finite")
        return (array - self.mean) / self.scale

    def as_dict(self) -> dict[str, Any]:
        return {
            "mean": self.mean.tolist(),
            "scale": self.scale.tolist(),
            "zero_variance_dimensions": list(self.zero_variance_dimensions),
            "variance_floor": STANDARDIZATION_VARIANCE_FLOOR,
        }


def fit_weighted_standardizer(values: np.ndarray, weights: np.ndarray) -> WeightedStandardizer:
    values = np.asarray(values, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 12 or values.shape[0] == 0 or not np.all(np.isfinite(values)):
        raise ValueError("finite pilot inputs with shape [N,12] are required")
    if weights.shape != (values.shape[0],) or not np.all(np.isfinite(weights)) or np.any(weights < 0) or np.sum(weights) <= 0:
        raise ValueError("finite non-negative weights matching inputs are required")
    total = float(np.sum(weights))
    mean = np.sum(values * weights[:, None], axis=0) / total
    variance = np.sum(((values - mean) ** 2) * weights[:, None], axis=0) / total
    zero = tuple(int(index) for index in np.flatnonzero(~np.isfinite(variance) | (variance <= STANDARDIZATION_VARIANCE_FLOOR)))
    scale = np.sqrt(np.maximum(variance, STANDARDIZATION_VARIANCE_FLOOR))
    if zero:
        scale[list(zero)] = 1.0
    return WeightedStandardizer(mean=mean, scale=scale, zero_variance_dimensions=zero)


@dataclass
class Batch:
    inputs: np.ndarray
    row_triplet_ids: np.ndarray
    row_component_ids: np.ndarray
    triplet_component_ids: np.ndarray
    row_window_ids: np.ndarray
    labels: np.ndarray
    window_weights: np.ndarray
    window_ids: list[str]
    n_triplets: int
    n_components: int
    n_windows: int


@dataclass(frozen=True)
class PreparedBatch:
    """GPU-resident tensors prepared once outside the optimization loop."""

    inputs: torch.Tensor
    row_triplet_ids: torch.Tensor
    row_component_ids: torch.Tensor
    triplet_component_ids: torch.Tensor
    row_window_ids: torch.Tensor
    component_window_ids: torch.Tensor


def prepare_batch_tensors(batch: Batch, device: str | torch.device) -> PreparedBatch:
    """Move the fixed aggregation index and input tensors to one device once."""

    device = torch.device(device)
    component_windows = [
        int(batch.row_window_ids[np.flatnonzero(batch.row_component_ids == component_id)[0]])
        for component_id in range(batch.n_components)
    ]
    return PreparedBatch(
        inputs=torch.as_tensor(batch.inputs, dtype=torch.float32, device=device),
        row_triplet_ids=torch.as_tensor(batch.row_triplet_ids, dtype=torch.long, device=device),
        row_component_ids=torch.as_tensor(batch.row_component_ids, dtype=torch.long, device=device),
        triplet_component_ids=torch.as_tensor(batch.triplet_component_ids, dtype=torch.long, device=device),
        row_window_ids=torch.as_tensor(batch.row_window_ids, dtype=torch.long, device=device),
        component_window_ids=torch.as_tensor(component_windows, dtype=torch.long, device=device),
    )


def weighted_window_bce(logits: torch.Tensor, labels: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """Compute the predeclared mean-normalized weighted window BCE."""

    losses = nn.functional.binary_cross_entropy_with_logits(logits, labels, reduction="none")
    return torch.sum(losses * weights) / torch.sum(weights)


def _triplets(example: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    values = example.get("triplets")
    if values is None:
        values = example.get("support", {}).get("triplets", [])
    if not values:
        raise ValueError(f"window has no valid triplets: {example.get('window_id')}")
    return list(values)


def build_batch(
    examples: Sequence[Mapping[str, Any]],
    arm: str,
    *,
    standardizer: WeightedStandardizer | None = None,
    observation_weights: bool = False,
) -> tuple[Batch, np.ndarray | None]:
    """Flatten variable component/triplet support while retaining aggregation IDs."""

    if arm not in ARM_NAMES:
        raise ValueError(f"unknown arm: {arm}")
    if not examples:
        raise ValueError("examples are empty")
    window_weights = source_class_window_weights(examples) if observation_weights else None
    inputs: list[np.ndarray] = []
    row_triplet_ids: list[int] = []
    row_component_ids: list[int] = []
    triplet_component_ids: list[int] = []
    row_window_ids: list[int] = []
    labels: list[int] = []
    row_observation_weights: list[float] = []
    window_ids: list[str] = []
    triplet_counter = component_counter = 0
    for window_counter, example in enumerate(examples):
        window_ids.append(str(example["window_id"]))
        label = window_label(example)
        labels.append(-1 if label is None else int(label))
        triplets = _triplets(example)
        component_to_triplets: dict[int, list[Mapping[str, Any]]] = {}
        for triplet in triplets:
            component_to_triplets.setdefault(int(triplet["component_index"]), []).append(triplet)
        component_index_map = {key: index + component_counter for index, key in enumerate(sorted(component_to_triplets))}
        for component_key in sorted(component_to_triplets):
            component_triplets = component_to_triplets[component_key]
            component_id = component_index_map[component_key]
            for triplet in component_triplets:
                arm_rows = arm_inputs_for_triplet(triplet)[arm]
                if arm_rows.ndim != 2 or arm_rows.shape[1] != 12 or not np.all(np.isfinite(arm_rows)):
                    raise ValueError("arm input must be finite with shape [N,12]")
                row_count = int(arm_rows.shape[0])
                inputs.append(arm_rows)
                row_triplet_ids.extend([triplet_counter] * row_count)
                row_component_ids.extend([component_id] * row_count)
                row_window_ids.extend([window_counter] * row_count)
                triplet_component_ids.append(component_id)
                if observation_weights and window_weights is not None:
                    base = float(window_weights[window_counter]) / len(component_to_triplets) / len(component_triplets) / row_count
                    row_observation_weights.extend([base] * row_count)
                triplet_counter += 1
        component_counter += len(component_to_triplets)
    matrix = np.concatenate(inputs, axis=0) if inputs else np.empty((0, 12), dtype=np.float64)
    if standardizer is not None:
        matrix = standardizer.transform(matrix)
    batch = Batch(
        inputs=matrix,
        row_triplet_ids=np.asarray(row_triplet_ids, dtype=np.int64),
        row_component_ids=np.asarray(row_component_ids, dtype=np.int64),
        triplet_component_ids=np.asarray(triplet_component_ids, dtype=np.int64),
        row_window_ids=np.asarray(row_window_ids, dtype=np.int64),
        labels=np.asarray(labels, dtype=np.int64),
        window_weights=window_weights if window_weights is not None else np.ones(len(examples), dtype=np.float64),
        window_ids=window_ids,
        n_triplets=triplet_counter,
        n_components=component_counter,
        n_windows=len(examples),
    )
    if observation_weights:
        observation = np.asarray(row_observation_weights, dtype=np.float64)
        if observation.size != matrix.shape[0]:
            raise RuntimeError("observation weight bookkeeping mismatch")
        observation *= observation.size / np.sum(observation)
        return batch, observation
    return batch, None


class WindowMLP(nn.Module):
    """Shared triplet encoder followed by component/window mean aggregation."""

    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(MODEL_CONFIG["input_dim"], MODEL_CONFIG["hidden_dim"]),
            nn.ReLU(),
            nn.Linear(MODEL_CONFIG["hidden_dim"], MODEL_CONFIG["representation_dim"]),
        )
        self.head = nn.Linear(MODEL_CONFIG["representation_dim"], 1)

    def _forward_prepared(self, batch: Batch, prepared: PreparedBatch) -> torch.Tensor:
        device = prepared.inputs.device
        hidden = self.encoder(prepared.inputs)
        triplet_ids = prepared.row_triplet_ids
        triplet_sum = torch.zeros((batch.n_triplets, hidden.shape[1]), dtype=hidden.dtype, device=device)
        triplet_count = torch.zeros(batch.n_triplets, dtype=hidden.dtype, device=device)
        triplet_sum.index_add_(0, triplet_ids, hidden)
        triplet_count.index_add_(0, triplet_ids, torch.ones_like(triplet_ids, dtype=hidden.dtype))
        triplet_values = triplet_sum / triplet_count[:, None]
        component_for_triplet = prepared.triplet_component_ids
        component_sum = torch.zeros((batch.n_components, hidden.shape[1]), dtype=hidden.dtype, device=device)
        component_count = torch.zeros(batch.n_components, dtype=hidden.dtype, device=device)
        component_sum.index_add_(0, component_for_triplet, triplet_values)
        component_count.index_add_(0, component_for_triplet, torch.ones_like(component_for_triplet, dtype=hidden.dtype))
        component_values = component_sum / component_count[:, None]
        window_sum = torch.zeros((batch.n_windows, hidden.shape[1]), dtype=hidden.dtype, device=device)
        window_count = torch.zeros(batch.n_windows, dtype=hidden.dtype, device=device)
        component_for_window = prepared.component_window_ids
        window_sum.index_add_(0, component_for_window, component_values)
        window_count.index_add_(0, component_for_window, torch.ones_like(component_for_window, dtype=hidden.dtype))
        window_values = window_sum / window_count[:, None]
        return self.head(window_values).squeeze(-1)

    def forward(self, batch: Batch) -> torch.Tensor:
        device = next(self.parameters()).device
        return self._forward_prepared(batch, prepare_batch_tensors(batch, device))


def train_model(
    batch: Batch,
    *,
    seed: int,
    epochs: int = int(MODEL_CONFIG["epochs"]),
    device: str | torch.device = "cpu",
) -> tuple[WindowMLP, dict[str, Any]]:
    """Train exactly one full-batch model with fixed tensors on ``device``."""

    if np.any(batch.labels < 0):
        raise ValueError("training labels must be present")
    if batch.n_windows < 2 or len(np.unique(batch.labels)) != 2:
        raise ValueError("training batch must contain both classes")
    torch.manual_seed(int(seed))
    device = torch.device(device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("requested CUDA training but CUDA is unavailable")
    model = WindowMLP().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=MODEL_CONFIG["learning_rate"], weight_decay=MODEL_CONFIG["weight_decay"])
    prepared = prepare_batch_tensors(batch, device)
    labels = torch.as_tensor(batch.labels, dtype=torch.float32, device=device)
    weights = torch.as_tensor(batch.window_weights, dtype=torch.float32, device=device)
    loss_history: list[float] = []
    for _ in range(int(epochs)):
        optimizer.zero_grad(set_to_none=True)
        logits = model._forward_prepared(batch, prepared)
        loss = weighted_window_bce(logits, labels, weights)
        loss.backward()
        optimizer.step()
        loss_history.append(float(loss.detach().cpu()))
    return model, {
        "seed": int(seed),
        "epochs": int(epochs),
        "initial_loss": loss_history[0],
        "final_loss": loss_history[-1],
        "min_loss": float(np.min(loss_history)),
        "loss_history": loss_history,
        "device": str(device),
        "parameter_device": str(next(model.parameters()).device),
        "input_device": str(prepared.inputs.device),
        "input_dtype": str(prepared.inputs.dtype),
        "output_device": str(device),
    }


def score_model(model: WindowMLP, batch: Batch) -> np.ndarray:
    model.eval()
    device = next(model.parameters()).device
    prepared = prepare_batch_tensors(batch, device)
    with torch.no_grad():
        return model._forward_prepared(batch, prepared).detach().cpu().numpy().astype(np.float64)


def serialize_model(model: WindowMLP, standardizer: WeightedStandardizer) -> dict[str, Any]:
    return {
        "standardization": standardizer.as_dict(),
        "state_dict": {name: value.detach().cpu().numpy().tolist() for name, value in model.state_dict().items()},
        "model_config": copy.deepcopy(MODEL_CONFIG),
    }
