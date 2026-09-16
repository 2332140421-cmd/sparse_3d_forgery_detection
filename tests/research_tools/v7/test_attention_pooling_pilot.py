"""Research-critical contracts for the local attention pooling ablation."""

import torch

from research_tools.v7.attention_pooling_pilot.model import (
    AttentionPoolModel,
    MeanBaselineModel,
    pool_attention,
    pool_mean,
)


def test_zero_attention_weights_reduce_to_mean_and_are_permutation_invariant() -> None:
    logits = torch.tensor([1.0, 3.0, -2.0, 4.0])
    representation = torch.tensor([[1.0, 0.0], [0.0, 1.0], [2.0, 1.0], [-1.0, 2.0]])
    indices = torch.tensor([0, 0, 1, 1], dtype=torch.long)
    value = torch.eye(2)
    bias = torch.zeros(2)
    weight = torch.zeros(2)
    attention, alpha = pool_attention(logits, representation, indices, 2, attention_v=value, attention_b=bias, attention_w=weight)
    expected = pool_mean(logits, indices, 2)
    assert torch.allclose(attention, expected, atol=1e-7)
    assert torch.allclose(alpha, torch.tensor([0.5, 0.5, 0.5, 0.5]), atol=1e-7)
    permutation = torch.tensor([3, 1, 0, 2])
    shuffled, _ = pool_attention(logits[permutation], representation[permutation], indices[permutation], 2, attention_v=value, attention_b=bias, attention_w=weight)
    assert torch.allclose(shuffled, attention, atol=1e-7)


def test_mask_and_single_unit_attention_contract() -> None:
    logits = torch.tensor([1.0, 3.0, 9.0])
    representation = torch.ones(3, 2)
    indices = torch.tensor([0, 0, 1], dtype=torch.long)
    mask = torch.tensor([True, False, True])
    pooled, alpha = pool_attention(logits, representation, indices, 2, mask, torch.eye(2), torch.zeros(2), torch.zeros(2))
    assert torch.allclose(pooled, torch.tensor([1.0, 9.0]))
    assert torch.allclose(alpha, torch.tensor([1.0, 0.0, 1.0]))


def test_attention_model_has_shared_set_a_parameters_and_finite_gradient() -> None:
    baseline = MeanBaselineModel()
    attention = AttentionPoolModel()
    attention.encoder.load_state_dict(baseline.encoder.state_dict())
    attention.head.load_state_dict(baseline.head.state_dict())
    assert sum(parameter.numel() for parameter in baseline.parameters()) == 569
    assert sum(parameter.numel() for parameter in attention.parameters()) == 649
    states = torch.randn(3, 5, 4)
    intervals = torch.ones(3, 4)
    indices = torch.tensor([0, 0, 1], dtype=torch.long)
    pooled, _ = attention.window_outputs(states, intervals, indices, 2)
    loss = pooled.square().mean()
    loss.backward()
    assert torch.isfinite(loss)
    assert attention.attention_w.grad is not None
    assert torch.all(torch.isfinite(attention.attention_w.grad))
