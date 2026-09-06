"""Temporal-only real-normal learnability probe primitives."""

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from sparse3d_forgery.particle_sequence import ParticleSequence, validate_particle_sequence


HORIZONS = (1, 2, 4, 8)


@dataclass(frozen=True, slots=True)
class NormalizedWindow:
    xyz: np.ndarray
    validity: np.ndarray
    centroid: np.ndarray
    rms_radius: float


def history_normalize(sequence: ParticleSequence, history_count: int) -> NormalizedWindow:
    """Normalize all valid rows using only the valid history point cloud."""

    validate_particle_sequence(sequence)
    if history_count <= 0 or history_count >= sequence.num_frames:
        raise ValueError("history_count must leave non-empty history and future rows")
    history_points = sequence.xyz[:history_count][sequence.geometry_validity[:history_count]]
    if history_points.size == 0:
        raise ValueError("history has no geometry-valid observations")
    centroid = history_points.astype(np.float64).mean(axis=0)
    radius = float(
        np.sqrt(np.mean(np.sum((history_points.astype(np.float64) - centroid) ** 2, axis=1)))
    )
    if not np.isfinite(radius) or radius <= np.finfo(np.float32).eps:
        raise ValueError("history RMS radius must be finite and non-zero")
    normalized = np.full(sequence.xyz.shape, np.nan, dtype=np.float32)
    valid = sequence.geometry_validity.copy()
    normalized[valid] = ((sequence.xyz[valid].astype(np.float64) - centroid) / radius).astype(
        np.float32
    )
    return NormalizedWindow(
        xyz=normalized,
        validity=valid,
        centroid=centroid,
        rms_radius=radius,
    )


def build_targets(
    xyz: np.ndarray,
    validity: np.ndarray,
    history_count: int,
    horizons: tuple[int, ...] = HORIZONS,
) -> tuple[np.ndarray, np.ndarray]:
    """Build direct cutoff-relative displacements and their validity mask."""

    cutoff = history_count - 1
    indices = [cutoff + horizon for horizon in horizons]
    if min(horizons) <= 0 or max(indices) >= xyz.shape[0]:
        raise ValueError("horizons must refer to available future observations")
    targets = np.full((xyz.shape[1], len(horizons), 3), np.nan, dtype=np.float32)
    target_validity = np.zeros((xyz.shape[1], len(horizons)), dtype=np.bool_)
    for offset, target_index in enumerate(indices):
        valid = validity[cutoff] & validity[target_index]
        target_validity[:, offset] = valid
        targets[valid, offset] = xyz[target_index, valid] - xyz[cutoff, valid]
    return targets, target_validity


class MissingAwareGRU(nn.Module):
    """One-layer particle-wise GRU that never consumes invalid coordinates."""

    def __init__(self, hidden_dim: int = 64, horizon_count: int = len(HORIZONS)) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.horizon_count = horizon_count
        self.cell = nn.GRUCell(3, hidden_dim)
        self.output = nn.Linear(hidden_dim, horizon_count * 3)

    def encode(self, history_xyz: torch.Tensor, history_validity: torch.Tensor) -> torch.Tensor:
        """Update only valid particle rows and carry missing rows unchanged."""

        batch, _, particles, _ = history_xyz.shape
        hidden = history_xyz.new_zeros((batch, particles, self.hidden_dim))
        for step in range(history_xyz.shape[1]):
            valid = history_validity[:, step]
            if valid.any():
                updated = self.cell(history_xyz[:, step][valid], hidden[valid])
                hidden = hidden.clone()
                hidden[valid] = updated
        return hidden

    def forward(self, history_xyz: torch.Tensor, history_validity: torch.Tensor) -> torch.Tensor:
        hidden = self.encode(history_xyz, history_validity)
        return self.output(hidden).reshape(
            history_xyz.shape[0], history_xyz.shape[2], self.horizon_count, 3
        )


def zero_displacement(particle_count: int, horizon_count: int = len(HORIZONS)) -> np.ndarray:
    return np.zeros((particle_count, horizon_count, 3), dtype=np.float32)


def linear_extrapolation(
    xyz: np.ndarray,
    validity: np.ndarray,
    timestamps_s: np.ndarray,
    history_count: int,
    horizons: tuple[int, ...] = HORIZONS,
) -> tuple[np.ndarray, np.ndarray]:
    """Diagnostic timestamp-aware linear continuation from two latest valid history rows."""

    cutoff = history_count - 1
    predictions = np.full((xyz.shape[1], len(horizons), 3), np.nan, dtype=np.float32)
    coverage = np.zeros((xyz.shape[1], len(horizons)), dtype=np.bool_)
    for particle in range(xyz.shape[1]):
        valid_history = np.flatnonzero(validity[:history_count, particle])
        if valid_history.size < 2 or not validity[cutoff, particle]:
            continue
        previous, latest = valid_history[-2:]
        delta_t = timestamps_s[latest] - timestamps_s[previous]
        if not np.isfinite(delta_t) or delta_t <= 0:
            continue
        slope = (xyz[latest, particle] - xyz[previous, particle]) / delta_t
        for offset, horizon in enumerate(horizons):
            target_index = cutoff + horizon
            if target_index >= xyz.shape[0] or not validity[target_index, particle]:
                continue
            predicted_position = xyz[latest, particle] + slope * (
                timestamps_s[target_index] - timestamps_s[latest]
            )
            predictions[particle, offset] = predicted_position - xyz[cutoff, particle]
            coverage[particle, offset] = True
    return predictions, coverage


def masked_mse(prediction: torch.Tensor, target: torch.Tensor, validity: torch.Tensor) -> torch.Tensor:
    if not validity.any():
        raise ValueError("batch contains no valid prediction targets")
    return ((prediction - target) ** 2)[validity.unsqueeze(-1).expand_as(prediction)].mean()


def rank_auroc(real_scores: np.ndarray, fake_scores: np.ndarray) -> float:
    """Binary AUROC by pairwise ranks, awarding half credit to ties."""

    real = np.asarray(real_scores, dtype=np.float64)
    fake = np.asarray(fake_scores, dtype=np.float64)
    if real.ndim != 1 or fake.ndim != 1 or not len(real) or not len(fake):
        raise ValueError("real and fake scores must be non-empty one-dimensional arrays")
    if not np.isfinite(real).all() or not np.isfinite(fake).all():
        raise ValueError("AUROC scores must be finite")
    comparisons = fake[:, None] - real[None, :]
    return float((np.count_nonzero(comparisons > 0) + 0.5 * np.count_nonzero(comparisons == 0)) / comparisons.size)


def assert_disjoint_source_ids(train_ids: list[str], validation_ids: list[str]) -> None:
    if set(train_ids) & set(validation_ids):
        raise ValueError("real train and validation source_video_id must be disjoint")
