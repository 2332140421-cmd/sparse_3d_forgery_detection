"""Small fixed models for the V7 history-neighborhood pilot.

The context model is intentionally a single relation layer followed by a
short temporal convolution.  It receives only the already-frozen five-time
SET_A states, a frozen directed local-group graph, and the four measured time
intervals.  No labels or provenance enter its numeric input.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

CONTEXT_CONDITIONS = (
    "SELF_TEMPORAL",
    "NEIGHBOR_TEMPORAL",
    "NEIGHBOR_REWIRED",
    "NEIGHBOR_TIME_SHUFFLED",
)


class LocalContextModel(nn.Module):
    """Encode each local state and one masked directed relation layer."""

    def __init__(self) -> None:
        super().__init__()
        self.state_encoder = nn.Sequential(
            nn.Linear(4, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
        )
        self.relation_encoder = nn.Sequential(
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
        )
        self.temporal = nn.Sequential(
            nn.Conv1d(16, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(16, 8, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.head = nn.Sequential(
            nn.Linear(44, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(
        self,
        states: torch.Tensor,
        intervals: torch.Tensor,
        node_window_index: torch.Tensor,
        edge_src: torch.Tensor,
        edge_dst: torch.Tensor,
        edge_time_index: torch.Tensor | None,
        window_count: int,
    ) -> torch.Tensor:
        """Return one logit per window.

        ``edge_time_index`` is ``None`` for correctly aligned neighbors.  For
        the time-shuffled arm it has shape ``[E,5]`` and is applied only to
        the neighbor branch.  An empty edge set produces a zero message via
        the explicit count mask; it never changes the state tensor.
        """

        if states.ndim != 3 or states.shape[-2:] != (5, 4):
            raise ValueError("states must have shape [N,5,4]")
        encoded = self.state_encoder(states)
        if edge_src.numel():
            own = encoded.index_select(0, edge_src)
            neighbor = encoded.index_select(0, edge_dst)
            if edge_time_index is not None:
                if edge_time_index.shape != (edge_src.shape[0], 5):
                    raise ValueError("edge_time_index must have shape [E,5]")
                edge_rows = torch.arange(edge_dst.shape[0], device=states.device)[:, None]
                neighbor = neighbor[edge_rows, edge_time_index]
            relation_input = torch.cat((own, neighbor, neighbor - own, own * neighbor), dim=-1)
            messages = self.relation_encoder(relation_input)
            aggregate = torch.zeros((states.shape[0], 5, 8), dtype=messages.dtype, device=states.device)
            aggregate.index_add_(0, edge_src, messages)
            counts = torch.zeros(states.shape[0], dtype=messages.dtype, device=states.device)
            counts.index_add_(0, edge_src, torch.ones_like(edge_src, dtype=messages.dtype))
            aggregate = aggregate / counts.clamp_min(1.0)[:, None, None]
        else:
            aggregate = torch.zeros((states.shape[0], 5, 8), dtype=encoded.dtype, device=states.device)
        temporal_input = torch.cat((encoded, aggregate), dim=-1).transpose(1, 2)
        temporal_output = self.temporal(temporal_input).transpose(1, 2).reshape(states.shape[0], 40)
        node_intervals = intervals.index_select(0, node_window_index)
        local_logits = self.head(torch.cat((temporal_output, node_intervals), dim=-1)).squeeze(-1)
        sums = torch.zeros(window_count, dtype=local_logits.dtype, device=states.device)
        counts = torch.zeros(window_count, dtype=local_logits.dtype, device=states.device)
        sums.index_add_(0, node_window_index, local_logits)
        counts.index_add_(0, node_window_index, torch.ones_like(local_logits))
        return sums / counts.clamp_min(1.0)


def parameter_count(model: nn.Module) -> int:
    return int(sum(parameter.numel() for parameter in model.parameters()))


def state_dict_numpy(model: nn.Module) -> dict[str, Any]:
    return {name: value.detach().cpu().numpy().tolist() for name, value in model.state_dict().items()}
