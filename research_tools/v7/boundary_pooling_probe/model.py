"""The two pooling variants for the bounded boundary pilot.

The encoder, activation, head and optimizer are the existing V7 pilot model.
Only the final local-group-to-window reduction changes from mean to max.
"""

from __future__ import annotations

import copy
from typing import Any

import numpy as np
import torch
from torch import nn

from research_tools.v7.local_structural_temporal_probe.model import (
    MODEL_CONFIG,
    Batch,
    PreparedBatch,
    build_batch,
    fit_weighted_standardizer,
    prepare_batch_tensors,
    source_class_window_weights,
    validate_training_examples,
    weighted_window_bce,
    window_label,
)


class PoolingWindowMLP(nn.Module):
    """Existing 12→16→8 encoder with explicit MEAN or MAX window pooling."""

    def __init__(self, pooling: str = "mean") -> None:
        super().__init__()
        if pooling not in {"mean", "max"}:
            raise ValueError("pooling must be 'mean' or 'max'")
        self.pooling = str(pooling)
        self.encoder = nn.Sequential(
            nn.Linear(MODEL_CONFIG["input_dim"], MODEL_CONFIG["hidden_dim"]),
            nn.ReLU(),
            nn.Linear(MODEL_CONFIG["hidden_dim"], MODEL_CONFIG["representation_dim"]),
        )
        self.head = nn.Linear(MODEL_CONFIG["representation_dim"], 1)

    def _component_values(self, batch: Batch, prepared: PreparedBatch) -> torch.Tensor:
        hidden = self.encoder(prepared.inputs)
        triplet_sum = torch.zeros((batch.n_triplets, hidden.shape[1]), dtype=hidden.dtype, device=hidden.device)
        triplet_count = torch.zeros(batch.n_triplets, dtype=hidden.dtype, device=hidden.device)
        triplet_sum.index_add_(0, prepared.row_triplet_ids, hidden)
        triplet_count.index_add_(0, prepared.row_triplet_ids, torch.ones_like(prepared.row_triplet_ids, dtype=hidden.dtype))
        triplet_values = triplet_sum / triplet_count[:, None]
        component_sum = torch.zeros((batch.n_components, hidden.shape[1]), dtype=hidden.dtype, device=hidden.device)
        component_count = torch.zeros(batch.n_components, dtype=hidden.dtype, device=hidden.device)
        component_sum.index_add_(0, prepared.triplet_component_ids, triplet_values)
        component_count.index_add_(0, prepared.triplet_component_ids, torch.ones_like(prepared.triplet_component_ids, dtype=hidden.dtype))
        return component_sum / component_count[:, None]

    def _forward_prepared(self, batch: Batch, prepared: PreparedBatch) -> torch.Tensor:
        component_values = self._component_values(batch, prepared)
        component_ids = prepared.component_window_ids
        if self.pooling == "mean":
            window_sum = torch.zeros((batch.n_windows, component_values.shape[1]), dtype=component_values.dtype, device=component_values.device)
            window_count = torch.zeros(batch.n_windows, dtype=component_values.dtype, device=component_values.device)
            window_sum.index_add_(0, component_ids, component_values)
            window_count.index_add_(0, component_ids, torch.ones_like(component_ids, dtype=component_values.dtype))
            return self.head(window_sum / window_count[:, None]).squeeze(-1)
        group_logits = self.head(component_values).squeeze(-1)
        values: list[torch.Tensor] = []
        for window_id in range(batch.n_windows):
            selected = group_logits[component_ids == int(window_id)]
            if selected.numel() == 0:
                raise ValueError("every window must have at least one local group")
            # torch.max is deterministic for ties on the fixed input order and
            # retains the correct gradient to the selected local group.
            values.append(torch.max(selected))
        return torch.stack(values)

    def forward(self, batch: Batch) -> torch.Tensor:
        device = next(self.parameters()).device
        return self._forward_prepared(batch, prepare_batch_tensors(batch, device))

    def group_logits(self, batch: Batch) -> torch.Tensor:
        """Return q_g for reporting local contributions after a fit."""

        device = next(self.parameters()).device
        prepared = prepare_batch_tensors(batch, device)
        with torch.no_grad():
            return self.head(self._component_values(batch, prepared)).squeeze(-1).detach().cpu()


def train_pooling_model(
    batch: Batch,
    *,
    pooling: str,
    seed: int,
    epochs: int = int(MODEL_CONFIG["epochs"]),
    device: str | torch.device = "cuda",
) -> tuple[PoolingWindowMLP, dict[str, Any]]:
    """Fit one full-batch model with the existing weighted BCE protocol."""

    if np.any(batch.labels < 0) or batch.n_windows < 2 or len(np.unique(batch.labels)) != 2:
        raise ValueError("training batch must contain both real and fake labels")
    torch.manual_seed(int(seed))
    device_obj = torch.device(device)
    if device_obj.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("boundary pilot requires CUDA for model training")
    model = PoolingWindowMLP(pooling).to(device_obj)
    optimizer = torch.optim.Adam(model.parameters(), lr=MODEL_CONFIG["learning_rate"], weight_decay=MODEL_CONFIG["weight_decay"])
    prepared = prepare_batch_tensors(batch, device_obj)
    labels = torch.as_tensor(batch.labels, dtype=torch.float32, device=device_obj)
    weights = torch.as_tensor(batch.window_weights, dtype=torch.float32, device=device_obj)
    losses: list[float] = []
    for _ in range(int(epochs)):
        optimizer.zero_grad(set_to_none=True)
        logits = model._forward_prepared(batch, prepared)
        loss = weighted_window_bce(logits, labels, weights)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    return model, {
        "seed": int(seed),
        "pooling": str(pooling),
        "epochs": int(epochs),
        "initial_loss": losses[0],
        "final_loss": losses[-1],
        "min_loss": float(np.min(losses)),
        "loss_history": losses,
        "device": str(device_obj),
        "parameter_device": str(next(model.parameters()).device),
        "input_device": str(prepared.inputs.device),
        "output_device": str(device_obj),
    }


def score_pooling_model(model: PoolingWindowMLP, batch: Batch) -> np.ndarray:
    model.eval()
    device = next(model.parameters()).device
    prepared = prepare_batch_tensors(batch, device)
    with torch.no_grad():
        return model._forward_prepared(batch, prepared).detach().cpu().numpy().astype(np.float64)


def serialize_pooling_model(model: PoolingWindowMLP, standardizer: Any) -> dict[str, Any]:
    return {
        "pooling": str(model.pooling),
        "standardization": standardizer.as_dict(),
        "state_dict": {name: value.detach().cpu().numpy().tolist() for name, value in model.state_dict().items()},
        "model_config": copy.deepcopy(MODEL_CONFIG),
    }


__all__ = [
    "PoolingWindowMLP",
    "build_batch",
    "fit_weighted_standardizer",
    "source_class_window_weights",
    "train_pooling_model",
    "score_pooling_model",
    "serialize_pooling_model",
    "validate_training_examples",
    "window_label",
]
