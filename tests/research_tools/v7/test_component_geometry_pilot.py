import numpy as np
import torch

from research_tools.v7.component_geometry_pilot.model import ComponentGeometryModel, SummaryModel


def _inputs(edge_dim: int = 7):
    torch.manual_seed(4)
    values = torch.randn(2, 5, 6, edge_dim)
    mask = torch.tensor([[[1, 1, 0, 0, 0, 0]] * 5, [[1, 1, 1, 0, 0, 0]] * 5], dtype=torch.bool)
    q = torch.randn(2, 5, 4)
    intervals = torch.ones(2, 4) * 0.1
    return values, mask, q, intervals


def test_geometry_edge_row_permutation_and_mask_invariance():
    values, mask, q, intervals = _inputs()
    model = ComponentGeometryModel(7).eval()
    with torch.no_grad():
        reference = model(values, mask, q, intervals)
        order = torch.tensor([2, 0, 5, 1, 4, 3])
        permuted = model(values[:, :, order], mask[:, :, order], q, intervals)
        ignored = values.clone()
        ignored[~mask] = 1e9
        masked = model(ignored, mask, q, intervals)
    assert torch.allclose(reference, permuted, atol=1e-6, rtol=0)
    assert torch.allclose(reference, masked, atol=1e-6, rtol=0)


def test_geometry_and_time_paths_have_finite_gradients():
    values, mask, q, intervals = _inputs()
    model = ComponentGeometryModel(7)
    loss = model(values, mask, q, intervals).sum()
    loss.backward()
    assert all(parameter.grad is None or torch.isfinite(parameter.grad).all() for parameter in model.parameters())
    assert model.edge_encoder[0].weight.grad is not None
    assert model.temporal[0].weight.grad is not None


def test_summary_input_contract_and_numpy_inputs_are_not_used():
    model = SummaryModel()
    states = torch.zeros(3, 5, 8)
    intervals = torch.ones(3, 4)
    result = model(states, intervals)
    assert result.shape == (3,)
    assert np.isfinite(result.detach().numpy()).all()
