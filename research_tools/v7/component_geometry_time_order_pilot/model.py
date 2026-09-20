"""Small models for the matched raw-geometry time-order pilot.

The set baseline is the already frozen component-geometry encoder with a
mean over the five state slots.  The ordered model keeps the same spatial
encoder and replaces only that time mean with a fixed-slot projection.  The
shuffle condition uses the same ordered model, while moving state contents
to deterministic non-identity time slots.
"""

from __future__ import annotations

import torch
from torch import nn

from research_tools.v7.component_geometry_pilot.model import ComponentGeometryModel


class OrderedComponentGeometryModel(nn.Module):
    """Fixed-slot, PTS-aware time encoder for five component states.

    ``edge_values`` is ``[U,5,K,E]``; ``q_states`` is ``[U,5,4]``;
    ``time_offsets`` is ``[U,5]`` and contains ``t_t - t_0`` in seconds.
    The spatial edge encoder and per-time encoder are intentionally identical
    to :class:`ComponentGeometryModel`.  Only the five-state aggregation is
    order-sensitive.
    """

    def __init__(self, edge_dim: int, q_dim: int = 4, embedding_dim: int = 64) -> None:
        if edge_dim not in (5, 7):
            raise ValueError("edge_dim must be 5 or 7")
        super().__init__()
        self.edge_dim = int(edge_dim)
        self.embedding_dim = int(embedding_dim)
        self.edge_encoder = nn.Sequential(
            nn.Linear(edge_dim, 32), nn.ReLU(), nn.Linear(32, embedding_dim), nn.ReLU()
        )
        self.temporal = nn.Sequential(
            nn.Linear(embedding_dim + q_dim, 16), nn.ReLU(), nn.Linear(16, 8), nn.ReLU()
        )
        # Five (8-dimensional state + one real-time offset) slots.  This is
        # fixed by the protocol, not searched.
        self.order_projection = nn.Sequential(nn.Linear(5 * (8 + 1), 8), nn.ReLU())
        self.head = nn.Sequential(
            nn.Linear(12, 16), nn.ReLU(), nn.Linear(16, 8), nn.ReLU(), nn.Linear(8, 1)
        )

    def encode_spatial(
        self,
        edge_values: torch.Tensor,
        edge_mask: torch.Tensor,
        q_states: torch.Tensor,
    ) -> torch.Tensor:
        if edge_values.ndim != 4 or edge_values.shape[1] != 5 or edge_values.shape[-1] != self.edge_dim:
            raise ValueError("edge_values must have shape [U,5,K,E]")
        if edge_mask.shape != edge_values.shape[:3]:
            raise ValueError("edge_mask must have shape [U,5,K]")
        if q_states.shape != (edge_values.shape[0], 5, 4):
            raise ValueError("q_states must have shape [U,5,4]")
        encoded = self.edge_encoder(edge_values)
        mask = edge_mask.to(dtype=encoded.dtype).unsqueeze(-1)
        component = (encoded * mask).sum(dim=2) / mask.sum(dim=2).clamp_min(1.0)
        return self.temporal(torch.cat((component, q_states), dim=-1))

    def forward(
        self,
        edge_values: torch.Tensor,
        edge_mask: torch.Tensor,
        q_states: torch.Tensor,
        time_offsets: torch.Tensor,
        intervals: torch.Tensor,
    ) -> torch.Tensor:
        if time_offsets.shape != edge_values.shape[:2]:
            raise ValueError("time_offsets must have shape [U,5]")
        if intervals.shape != (edge_values.shape[0], 4):
            raise ValueError("intervals must have shape [U,4]")
        per_time = self.encode_spatial(edge_values, edge_mask, q_states)
        slots = torch.cat((per_time, time_offsets.unsqueeze(-1)), dim=-1)
        ordered = self.order_projection(slots.reshape(slots.shape[0], -1))
        return self.head(torch.cat((ordered, intervals), dim=-1)).squeeze(-1)


def parameter_count(model: nn.Module) -> int:
    return int(sum(parameter.numel() for parameter in model.parameters()))
