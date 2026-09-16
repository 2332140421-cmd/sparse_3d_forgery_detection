"""The two window-pooling variants used by the attention ablation.

The shared encoder and local classification head are the existing SET_A
architecture.  Only the local-unit-to-window reduction is different.
"""

from __future__ import annotations

from typing import Optional

import torch
from torch import nn

from research_tools.v7.multi_order_sequence_probe.model import SetAModel


def _segment_indices(unit_window_index: torch.Tensor, window_count: int) -> torch.Tensor:
    if unit_window_index.ndim != 1 or unit_window_index.dtype != torch.long:
        raise ValueError("unit_window_index must be a one-dimensional int64 tensor")
    if window_count < 1:
        raise ValueError("window_count must be positive")
    if unit_window_index.numel() and (
        int(unit_window_index.min()) < 0 or int(unit_window_index.max()) >= window_count
    ):
        raise ValueError("unit_window_index is outside window range")
    return unit_window_index


def _valid_mask(values: torch.Tensor, unit_mask: Optional[torch.Tensor]) -> torch.Tensor:
    if unit_mask is None:
        return torch.ones(values.shape[0], dtype=torch.bool, device=values.device)
    mask = unit_mask.to(device=values.device, dtype=torch.bool)
    if mask.ndim != 1 or mask.shape[0] != values.shape[0]:
        raise ValueError("unit_mask must have one value per local unit")
    return mask


def pool_mean(
    local_logits: torch.Tensor,
    unit_window_index: torch.Tensor,
    window_count: int,
    unit_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Mean local logits per window, with optional explicit padding mask."""

    indices = _segment_indices(unit_window_index, window_count)
    mask = _valid_mask(local_logits, unit_mask)
    values = torch.where(mask, local_logits, torch.zeros_like(local_logits))
    sums = torch.zeros(window_count, dtype=local_logits.dtype, device=local_logits.device)
    counts = torch.zeros(window_count, dtype=local_logits.dtype, device=local_logits.device)
    sums.scatter_add_(0, indices, values)
    counts.scatter_add_(0, indices, mask.to(local_logits.dtype))
    return sums / counts.clamp_min(1.0)


def _segment_softmax(
    attention_scores: torch.Tensor,
    unit_window_index: torch.Tensor,
    window_count: int,
    unit_mask: torch.Tensor,
) -> torch.Tensor:
    """Softmax scores independently inside each window segment."""

    indices = _segment_indices(unit_window_index, window_count)
    valid_scores = torch.where(unit_mask, attention_scores, torch.full_like(attention_scores, -torch.inf))
    maximum = torch.full(
        (window_count,), -torch.inf, dtype=attention_scores.dtype, device=attention_scores.device
    )
    maximum.scatter_reduce_(0, indices, valid_scores, reduce="amax", include_self=True)
    centered = torch.where(
        unit_mask,
        attention_scores - maximum[indices],
        torch.zeros_like(attention_scores),
    )
    exponent = torch.where(unit_mask, centered.exp(), torch.zeros_like(attention_scores))
    denominator = torch.zeros(window_count, dtype=attention_scores.dtype, device=attention_scores.device)
    denominator.scatter_add_(0, indices, exponent)
    return exponent / denominator[indices].clamp_min(torch.finfo(attention_scores.dtype).tiny)


def pool_attention(
    local_logits: torch.Tensor,
    local_repr: torch.Tensor,
    unit_window_index: torch.Tensor,
    window_count: int,
    unit_mask: Optional[torch.Tensor] = None,
    attention_v: Optional[torch.Tensor] = None,
    attention_b: Optional[torch.Tensor] = None,
    attention_w: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return attention-pooled window logits and per-unit weights.

    ``local_repr`` is the eight-dimensional SET_A representation immediately
    before the local classification head.  The optional parameter arguments
    make the pure pooling contract directly testable.
    """

    if local_repr.ndim != 2 or local_repr.shape[0] != local_logits.shape[0]:
        raise ValueError("local_repr must be [units, hidden]")
    hidden = local_repr.shape[1]
    value = attention_v if attention_v is not None else torch.eye(hidden, dtype=local_repr.dtype, device=local_repr.device)
    bias = attention_b if attention_b is not None else torch.zeros(hidden, dtype=local_repr.dtype, device=local_repr.device)
    weight = attention_w if attention_w is not None else torch.zeros(hidden, dtype=local_repr.dtype, device=local_repr.device)
    if value.shape != (hidden, hidden) or bias.shape != (hidden,) or weight.shape != (hidden,):
        raise ValueError("attention parameters have incompatible dimensions")
    scores = torch.tanh(local_repr @ value.T + bias) @ weight
    mask = _valid_mask(local_logits, unit_mask)
    alpha = _segment_softmax(scores, unit_window_index, window_count, mask)
    pooled = torch.zeros(window_count, dtype=local_logits.dtype, device=local_logits.device)
    pooled.scatter_add_(0, unit_window_index, torch.where(mask, alpha * local_logits, torch.zeros_like(local_logits)))
    return pooled, alpha


class MeanBaselineModel(SetAModel):
    """Existing SET_A encoder/head with the original equal-weight pooling."""

    def local_outputs(self, states: torch.Tensor, intervals: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        representation = self.encoder(states).mean(dim=1)
        logits = self.head(torch.cat((representation, intervals), dim=-1)).squeeze(-1)
        return representation, logits

    def window_outputs(
        self,
        states: torch.Tensor,
        intervals: torch.Tensor,
        unit_window_index: torch.Tensor,
        window_count: int,
        unit_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        representation, logits = self.local_outputs(states, intervals)
        return pool_mean(logits, unit_window_index, window_count, unit_mask), representation


class AttentionPoolModel(MeanBaselineModel):
    """SET_A with the one-head, eight-dimensional local attention reducer."""

    ATTENTION_HIDDEN = 8
    ATTENTION_INIT_SEED = 20260912

    def __init__(self) -> None:
        super().__init__()
        if self.encoder[2].out_features != self.ATTENTION_HIDDEN:
            raise ValueError("attention hidden dimension must match SET_A representation")
        self.attention_v = nn.Parameter(torch.empty(self.ATTENTION_HIDDEN, self.ATTENTION_HIDDEN))
        self.attention_b = nn.Parameter(torch.zeros(self.ATTENTION_HIDDEN))
        self.attention_w = nn.Parameter(torch.zeros(self.ATTENTION_HIDDEN))
        # Preserve the caller's RNG state: this seed is only for V, while the
        # shared encoder/head are copied from the paired baseline.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(self.ATTENTION_INIT_SEED)
            nn.init.xavier_uniform_(self.attention_v)

    def window_outputs(
        self,
        states: torch.Tensor,
        intervals: torch.Tensor,
        unit_window_index: torch.Tensor,
        window_count: int,
        unit_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        representation, logits = self.local_outputs(states, intervals)
        pooled, alpha = pool_attention(
            logits,
            representation,
            unit_window_index,
            window_count,
            unit_mask,
            self.attention_v,
            self.attention_b,
            self.attention_w,
        )
        return pooled, alpha
