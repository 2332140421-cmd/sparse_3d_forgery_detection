import numpy as np
import pytest
import torch

from sparse3d_forgery.experiments.spatial_probe import (
    DenseSpatialEncoder,
    SelfTemporalModel,
    SpatialTemporalModel,
    assert_real_only_training,
    auroc_delta_bootstrap,
    dense_relations,
    paired_mean_bootstrap,
)
from sparse3d_forgery.experiments.temporal_probe import build_targets, history_normalize
from test_temporal_probe import _sequence


def test_dense_candidates_are_valid_directed_nonself_and_exact_delta():
    xyz = torch.tensor([[1.0, 2.0, 3.0], [9.0, 9.0, 9.0], [4.0, 6.0, 8.0]])
    edges, delta = dense_relations(xyz, torch.tensor([True, False, True]))

    assert edges.tolist() == [[0, 2], [2, 0]]
    torch.testing.assert_close(delta, torch.stack((xyz[2] - xyz[0], xyz[0] - xyz[2])))
    assert not torch.any(edges[:, 0] == edges[:, 1])
    assert 1 not in edges


def test_attention_normalizes_over_neighbors_and_empty_aggregate_is_internal_zero():
    torch.manual_seed(1)
    encoder = DenseSpatialEncoder(spatial_dim=4)
    xyz = torch.randn(1, 3, 3)
    _, records = encoder.encode_frame(xyz, torch.tensor([[True, True, True]]), True)
    _, weights = records[0]
    torch.testing.assert_close(weights.sum(dim=-1), torch.ones(3))
    assert torch.all(torch.diagonal(weights) == 0)

    state, single = encoder.encode_frame(xyz, torch.tensor([[True, False, False]]), True)
    expected = encoder.norm(encoder.self_mlp(xyz[0, :1]))
    torch.testing.assert_close(state[0, :1], expected)
    assert single[0][1].shape == (1, 0)


def test_invalid_particle_does_not_update_spatial_gru_hidden():
    torch.manual_seed(2)
    model = SpatialTemporalModel(hidden_dim=5, spatial_dim=4)
    xyz = torch.randn(1, 3, 3, 3)
    validity = torch.ones(1, 3, 3, dtype=torch.bool)
    validity[0, 1, 2] = False
    changed = xyz.clone()
    changed[0, 1, 2] = torch.tensor([float("nan")] * 3)

    first = model.encode(xyz, validity)
    second = model.encode(changed, validity)

    torch.testing.assert_close(first, second)


def test_self_control_does_not_read_other_particles():
    torch.manual_seed(3)
    model = SelfTemporalModel(hidden_dim=5, spatial_dim=4)
    xyz = torch.randn(1, 3, 2, 3)
    changed = xyz.clone()
    changed[:, :, 1] += 1000
    validity = torch.ones(1, 3, 2, dtype=torch.bool)

    first = model(xyz, validity)
    second = model(changed, validity)

    torch.testing.assert_close(first[:, 0], second[:, 0])


@pytest.mark.parametrize("model_type", [SelfTemporalModel, SpatialTemporalModel])
def test_direct_output_shape(model_type):
    model = model_type(hidden_dim=5, spatial_dim=4)
    output = model(torch.randn(2, 8, 3, 3), torch.ones(2, 8, 3, dtype=torch.bool))
    assert output.shape == (2, 3, 4, 3)


def test_target_mask_is_identical_to_temporal_probe():
    normalized = history_normalize(_sequence(), 8)
    _, mask = build_targets(normalized.xyz, normalized.validity, 8)
    expected = np.stack(
        [normalized.validity[7] & normalized.validity[7 + horizon] for horizon in (1, 2, 4, 8)],
        axis=1,
    )
    np.testing.assert_equal(mask, expected)


def test_fake_roles_are_rejected_from_optimizer_boundary():
    assert_real_only_training(["real_train", "real_train"])
    with pytest.raises(ValueError, match="real_train"):
        assert_real_only_training(["real_train", "fake_probe"])


def test_bootstraps_are_reproducible():
    candidate = np.array([0.1, 0.2, 0.3, 0.4])
    baseline = np.array([0.2, 0.1, 0.5, 0.6])
    first = paired_mean_bootstrap(candidate, baseline, 20260906, 100)
    second = paired_mean_bootstrap(candidate, baseline, 20260906, 100)
    assert first == second

    real_s, fake_s = np.array([0.1, 0.2]), np.array([0.8, 0.9])
    real_b, fake_b = np.array([0.2, 0.3]), np.array([0.4, 0.5])
    first_delta = auroc_delta_bootstrap(real_s, fake_s, real_b, fake_b, 20260906, 100)
    second_delta = auroc_delta_bootstrap(real_s, fake_s, real_b, fake_b, 20260906, 100)
    assert first_delta == second_delta
