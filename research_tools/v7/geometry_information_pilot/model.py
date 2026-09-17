"""The fixed summary models used by the same-support geometry pilot."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn


class ModalSetModel(nn.Module):
    """SetA-style five-time encoder for either four or eight channels."""

    def __init__(self, input_dim: int) -> None:
        if input_dim not in (4, 8):
            raise ValueError("input_dim must be 4 or 8")
        super().__init__()
        self.input_dim = int(input_dim)
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
        )
        self.head = nn.Sequential(
            nn.Linear(12, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
        )

    def forward(self, states: torch.Tensor, intervals: torch.Tensor) -> torch.Tensor:
        if states.ndim != 3 or states.shape[-1] != self.input_dim or states.shape[1] != 5:
            raise ValueError(f"states must have shape [U,5,{self.input_dim}]")
        if intervals.ndim != 2 or intervals.shape != (states.shape[0], 4):
            raise ValueError("intervals must have shape [U,4]")
        encoded = self.encoder(states).mean(dim=1)
        return self.head(torch.cat((encoded, intervals), dim=-1)).squeeze(-1)


def parameter_count(model: nn.Module) -> int:
    return int(sum(parameter.numel() for parameter in model.parameters()))


def state_dict_numpy(model: nn.Module) -> dict[str, Any]:
    return {name: value.detach().cpu().numpy().tolist() for name, value in model.state_dict().items()}
