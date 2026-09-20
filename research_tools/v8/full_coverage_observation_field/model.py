"""Small V8 model with explicit spatial and causal component stages."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from .common import CELL_COUNT, cell_neighbors


class FullCoverageModel(nn.Module):
    """FULL or RGB_2D model.

    Inputs are fixed-size base-cell sequences.  Component maps and predecessor
    maps are data, not learned IDs; they only control masked pooling/gathering.
    Every non-padding cell remains in the final area-weighted aggregation.
    """

    def __init__(self, condition: str, input_dim: int, hidden: int = 128):
        super().__init__()
        if condition not in {"FULL", "RGB_2D"}:
            raise ValueError(condition)
        self.condition = condition
        self.input_dim = int(input_dim)
        self.hidden = int(hidden)
        self.cell_encoder = nn.Sequential(nn.Linear(input_dim, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU())
        self.edge_mlp = nn.Sequential(nn.Linear(hidden, hidden // 2), nn.ReLU(), nn.Linear(hidden // 2, hidden))
        self.component_residual = nn.Sequential(nn.Linear(hidden * 2, hidden), nn.ReLU(), nn.Linear(hidden, hidden))
        self.temporal_in = nn.Sequential(nn.Linear(hidden + 4, hidden), nn.ReLU())
        self.gru = nn.GRUCell(hidden, hidden)
        self.cell_head = nn.Sequential(nn.Linear(hidden, hidden // 2), nn.ReLU(), nn.Linear(hidden // 2, 1))
        self.register_buffer("edge_index", torch.from_numpy(cell_neighbors()), persistent=False)

    def _pool_components(self, h: torch.Tensor, comp: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Mean pool by per-frame component labels and broadcast back."""
        b, n, d = h.shape
        pooled = torch.zeros_like(h)
        comp_counts = []
        for bi in range(b):
            ids = comp[bi].long()
            unique = torch.unique(ids, sorted=True)
            means = []
            for uid in unique:
                m = ids == uid
                means.append(h[bi, m].mean(dim=0))
                pooled[bi, m] = means[-1]
            comp_counts.append(unique.numel())
        return pooled, torch.as_tensor(comp_counts, device=h.device, dtype=torch.float32)

    def _spatial(self, h: torch.Tensor, comp: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        src, dst = self.edge_index[:, 0], self.edge_index[:, 1]
        rel = h[:, dst] - h[:, src]
        msg = self.edge_mlp(rel)
        agg = torch.zeros_like(h)
        deg = torch.zeros((h.shape[0], h.shape[1], 1), device=h.device, dtype=h.dtype)
        agg.index_add_(1, src, msg)
        agg.index_add_(1, dst, -msg)
        deg.index_add_(1, src, torch.ones_like(deg[:, src]))
        deg.index_add_(1, dst, torch.ones_like(deg[:, dst]))
        h = h + agg / deg.clamp_min(1.0)
        comp_h, counts = self._pool_components(h, comp)
        h = h + self.component_residual(torch.cat([h, comp_h], dim=-1))
        return h, counts

    def forward(
        self,
        x: torch.Tensor,
        component_ids: torch.Tensor,
        predecessor: torch.Tensor,
        association_weight: torch.Tensor,
        history_available: torch.Tensor,
        delta_t: torch.Tensor,
        area: torch.Tensor,
        return_evidence: bool = False,
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        if x.ndim != 4 or x.shape[2] != CELL_COUNT:
            raise ValueError(f"x must be [B,T,{CELL_COUNT},D], got {tuple(x.shape)}")
        b, t, n, _ = x.shape
        h_prev = torch.zeros((b, n, self.hidden), device=x.device, dtype=x.dtype)
        logits: list[torch.Tensor] = []
        comp_counts: list[torch.Tensor] = []
        for ti in range(t):
            h = self.cell_encoder(x[:, ti])
            h, cc = self._spatial(h, component_ids[:, ti])
            if ti > 0:
                # predecessor stores the source component's representative cell;
                # association_weight is an explicit coverage, not a label feature.
                pi = predecessor[:, ti].long().clamp(0, n - 1)
                prev = h_prev.gather(1, pi[..., None].expand(-1, -1, self.hidden))
                dt = delta_t[:, ti].unsqueeze(-1).unsqueeze(-1).expand(-1, n, 1)
                aux = torch.cat([dt, association_weight[:, ti, :, None], history_available[:, ti, :, None].float(), (1.0 - association_weight[:, ti, :, None])], dim=-1)
                temporal_input = torch.cat([h, aux], dim=-1).reshape(b * n, self.hidden + 4)
                h = self.gru(self.temporal_in(temporal_input), prev.reshape(b * n, self.hidden)).reshape(b, n, self.hidden)
            else:
                temporal_input = torch.cat([h, torch.zeros((b, n, 4), device=x.device, dtype=x.dtype)], dim=-1).reshape(b * n, self.hidden + 4)
                h = self.gru(self.temporal_in(temporal_input), h_prev.reshape(b * n, self.hidden)).reshape(b, n, self.hidden)
            h_prev = h
            logits.append(self.cell_head(h).squeeze(-1))
            comp_counts.append(cc)
        cell_logits = torch.stack(logits, dim=1)
        weights = area.to(cell_logits.dtype).clamp_min(0)
        weights = weights / weights.sum(dim=(1, 2), keepdim=True).clamp_min(1e-8)
        window_logit = torch.logsumexp(cell_logits + torch.log(weights.clamp_min(1e-12)), dim=(1, 2))
        if not return_evidence:
            return window_logit
        return {
            "window_logit": window_logit,
            "cell_logits": cell_logits,
            "weights": weights,
            "component_counts": torch.stack(comp_counts, dim=1),
        }


def parameter_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
