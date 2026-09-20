"""Small masked local-relation models for the component-geometry pilot.

The geometry encoder is intentionally independent of source IDs, labels and
point IDs.  It consumes fixed historical directed edges and a per-edge mask;
invalid padded edges never enter the component mean.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn


class SummaryModel(nn.Module):
    """A matched four-dimensional S+Q summary model."""

    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 8), nn.ReLU())
        self.head = nn.Sequential(nn.Linear(12, 16), nn.ReLU(), nn.Linear(16, 8), nn.ReLU(), nn.Linear(8, 1))

    def forward(self, states: torch.Tensor, intervals: torch.Tensor) -> torch.Tensor:
        if states.ndim != 3 or states.shape[1:] != (5, 8):
            raise ValueError("summary states must have shape [U,5,8]")
        if intervals.ndim != 2 or intervals.shape != (states.shape[0], 4):
            raise ValueError("intervals must have shape [U,4]")
        encoded = self.encoder(states).mean(dim=1)
        return self.head(torch.cat((encoded, intervals), dim=-1)).squeeze(-1)


class ComponentGeometryModel(nn.Module):
    """Encode fixed local edges, then use the same five-time/head layout."""

    def __init__(self, edge_dim: int, q_dim: int = 4, embedding_dim: int = 64) -> None:
        if edge_dim not in (5, 7):
            raise ValueError("edge_dim must be 5 (2D) or 7 (3D)")
        super().__init__()
        self.edge_dim = int(edge_dim)
        self.embedding_dim = int(embedding_dim)
        self.edge_encoder = nn.Sequential(nn.Linear(edge_dim, 32), nn.ReLU(), nn.Linear(32, embedding_dim), nn.ReLU())
        self.temporal = nn.Sequential(nn.Linear(embedding_dim + q_dim, 16), nn.ReLU(), nn.Linear(16, 8), nn.ReLU())
        self.head = nn.Sequential(nn.Linear(12, 16), nn.ReLU(), nn.Linear(16, 8), nn.ReLU(), nn.Linear(8, 1))

    def forward(
        self,
        edge_values: torch.Tensor,
        edge_mask: torch.Tensor,
        q_states: torch.Tensor,
        intervals: torch.Tensor,
    ) -> torch.Tensor:
        if edge_values.ndim != 4 or edge_values.shape[-1] != self.edge_dim or edge_values.shape[1] != 5:
            raise ValueError("edge_values must have shape [U,5,K,edge_dim]")
        if edge_mask.shape != edge_values.shape[:3]:
            raise ValueError("edge_mask must have shape [U,5,K]")
        if q_states.ndim != 3 or q_states.shape[:2] != edge_values.shape[:2] or q_states.shape[-1] != 4:
            raise ValueError("q_states must have shape [U,5,4]")
        encoded = self.edge_encoder(edge_values)
        mask = edge_mask.to(dtype=encoded.dtype).unsqueeze(-1)
        component = (encoded * mask).sum(dim=2) / mask.sum(dim=2).clamp_min(1.0)
        temporal = self.temporal(torch.cat((component, q_states), dim=-1)).mean(dim=1)
        return self.head(torch.cat((temporal, intervals), dim=-1)).squeeze(-1)


def parameter_count(model: nn.Module) -> int:
    return int(sum(parameter.numel() for parameter in model.parameters()))


def state_dict_numpy(model: nn.Module) -> dict[str, Any]:
    return {name: value.detach().cpu().numpy().tolist() for name, value in model.state_dict().items()}
