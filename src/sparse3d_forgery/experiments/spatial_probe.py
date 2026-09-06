"""Dense-no-self learned spatial dependency falsification probe."""

import numpy as np
import torch
from torch import nn

from .temporal_probe import HORIZONS, rank_auroc


def dense_relations(xyz: torch.Tensor, validity: torch.Tensor):
    """Return valid directed non-self candidates and exact X_j - X_i inputs."""

    indices = torch.nonzero(validity, as_tuple=False).flatten()
    if indices.numel() < 2:
        empty_edges = torch.empty((0, 2), dtype=torch.long, device=xyz.device)
        return empty_edges, xyz.new_empty((0, 3))
    centers = indices.repeat_interleave(indices.numel())
    neighbors = indices.repeat(indices.numel())
    keep = centers != neighbors
    edges = torch.stack((centers[keep], neighbors[keep]), dim=-1)
    return edges, xyz[edges[:, 1]] - xyz[edges[:, 0]]


class SelfEncoder(nn.Module):
    def __init__(self, spatial_dim: int = 32) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(3, spatial_dim), nn.GELU(), nn.Linear(spatial_dim, spatial_dim)
        )
        self.norm = nn.LayerNorm(spatial_dim)

    def forward(self, xyz: torch.Tensor) -> torch.Tensor:
        return self.norm(self.mlp(xyz))


class DenseSpatialEncoder(nn.Module):
    """One learned soft relation layer over valid directed non-self candidates."""

    def __init__(self, spatial_dim: int = 32) -> None:
        super().__init__()
        self.self_mlp = nn.Sequential(
            nn.Linear(3, spatial_dim), nn.GELU(), nn.Linear(spatial_dim, spatial_dim)
        )
        self.relation_mlp = nn.Sequential(
            nn.Linear(3, spatial_dim), nn.GELU(), nn.Linear(spatial_dim, spatial_dim)
        )
        self.attention = nn.Linear(spatial_dim, 1)
        self.norm = nn.LayerNorm(spatial_dim)

    def encode_frame(
        self, xyz: torch.Tensor, validity: torch.Tensor, return_attention: bool = False
    ):
        batch, particles, _ = xyz.shape
        states = xyz.new_zeros((batch, particles, self.norm.normalized_shape[0]))
        attention_records = []
        for batch_index in range(batch):
            indices = torch.nonzero(validity[batch_index], as_tuple=False).flatten()
            if indices.numel() == 0:
                attention_records.append((indices, xyz.new_empty((0, 0))))
                continue
            points = xyz[batch_index, indices]
            self_state = self.self_mlp(points)
            message = torch.zeros_like(self_state)
            weights = xyz.new_empty((indices.numel(), 0))
            if indices.numel() > 1:
                # delta[center, neighbor] is exactly X_j - X_i.
                delta = points.unsqueeze(0) - points.unsqueeze(1)
                relation = self.relation_mlp(delta)
                logits = self.attention(relation).squeeze(-1)
                diagonal = torch.eye(indices.numel(), dtype=torch.bool, device=xyz.device)
                weights = torch.softmax(logits.masked_fill(diagonal, -torch.inf), dim=-1)
                message = torch.sum(weights.unsqueeze(-1) * relation, dim=1)
            # Zero message above denotes an empty aggregation, never a missing XYZ value.
            states[batch_index, indices] = self.norm(self_state + message)
            attention_records.append((indices, weights))
        return (states, attention_records) if return_attention else states


class _SpatialTemporalBase(nn.Module):
    def __init__(self, hidden_dim: int = 64, spatial_dim: int = 32) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.cell = nn.GRUCell(spatial_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, len(HORIZONS) * 3)

    def spatial_state(self, xyz: torch.Tensor, validity: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def encode(self, history_xyz: torch.Tensor, history_validity: torch.Tensor) -> torch.Tensor:
        batch, _, particles, _ = history_xyz.shape
        hidden = history_xyz.new_zeros((batch, particles, self.hidden_dim))
        for step in range(history_xyz.shape[1]):
            valid = history_validity[:, step]
            spatial = self.spatial_state(history_xyz[:, step], valid)
            if valid.any():
                updated = self.cell(spatial[valid], hidden[valid])
                hidden = hidden.clone()
                hidden[valid] = updated
        return hidden

    def forward(self, history_xyz: torch.Tensor, history_validity: torch.Tensor) -> torch.Tensor:
        hidden = self.encode(history_xyz, history_validity)
        return self.output(hidden).reshape(
            history_xyz.shape[0], history_xyz.shape[2], len(HORIZONS), 3
        )


class SelfTemporalModel(_SpatialTemporalBase):
    """Capacity control that processes each particle independently."""

    def __init__(self, hidden_dim: int = 64, spatial_dim: int = 32) -> None:
        super().__init__(hidden_dim, spatial_dim)
        self.self_encoder = SelfEncoder(spatial_dim)

    def spatial_state(self, xyz: torch.Tensor, validity: torch.Tensor) -> torch.Tensor:
        states = xyz.new_zeros((*xyz.shape[:2], self.cell.input_size))
        if validity.any():
            states[validity] = self.self_encoder(xyz[validity])
        return states


class SpatialTemporalModel(_SpatialTemporalBase):
    def __init__(self, hidden_dim: int = 64, spatial_dim: int = 32) -> None:
        super().__init__(hidden_dim, spatial_dim)
        self.spatial_encoder = DenseSpatialEncoder(spatial_dim)

    def spatial_state(self, xyz: torch.Tensor, validity: torch.Tensor) -> torch.Tensor:
        return self.spatial_encoder.encode_frame(xyz, validity)


def trainable_parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def assert_real_only_training(roles: list[str]) -> None:
    if not roles or any(role != "real_train" for role in roles):
        raise ValueError("optimizer and checkpoint selection require real_train views only")


def paired_mean_bootstrap(
    candidate: np.ndarray, baseline: np.ndarray, seed: int, replicates: int = 10_000
) -> dict:
    difference = np.asarray(candidate, dtype=np.float64) - np.asarray(baseline, dtype=np.float64)
    if difference.ndim != 1 or not difference.size:
        raise ValueError("paired scores must be non-empty one-dimensional arrays")
    generator = np.random.default_rng(seed)
    samples = generator.integers(0, difference.size, size=(replicates, difference.size))
    means = difference[samples].mean(axis=1)
    return {
        "mean_difference": float(difference.mean()),
        "median_difference": float(np.median(difference)),
        "fraction_candidate_better": float(np.mean(difference < 0)),
        "mean_difference_ci95": [float(x) for x in np.percentile(means, [2.5, 97.5])],
    }


def auroc_delta_bootstrap(
    real_candidate: np.ndarray,
    fake_candidate: np.ndarray,
    real_baseline: np.ndarray,
    fake_baseline: np.ndarray,
    seed: int,
    replicates: int = 10_000,
) -> dict:
    arrays = [np.asarray(x, dtype=np.float64) for x in (real_candidate, fake_candidate, real_baseline, fake_baseline)]
    if arrays[0].shape != arrays[2].shape or arrays[1].shape != arrays[3].shape:
        raise ValueError("candidate and baseline clip scores must be paired")
    point = rank_auroc(arrays[0], arrays[1]) - rank_auroc(arrays[2], arrays[3])
    generator = np.random.default_rng(seed)
    deltas = np.empty(replicates, dtype=np.float64)
    for index in range(replicates):
        real_indices = generator.integers(0, arrays[0].size, arrays[0].size)
        fake_indices = generator.integers(0, arrays[1].size, arrays[1].size)
        deltas[index] = rank_auroc(arrays[0][real_indices], arrays[1][fake_indices]) - rank_auroc(
            arrays[2][real_indices], arrays[3][fake_indices]
        )
    return {
        "point_estimate": float(point),
        "ci95": [float(x) for x in np.percentile(deltas, [2.5, 97.5])],
    }
