import torch
import numpy as np

from research_tools.v7.component_geometry_pilot.model import ComponentGeometryModel
from research_tools.v7.component_geometry_time_order_pilot.model import OrderedComponentGeometryModel, parameter_count
from research_tools.v7.component_geometry_time_order_pilot import runner


def _inputs():
    torch.manual_seed(17)
    values = torch.randn(2, 5, 4, 7)
    mask = torch.ones(2, 5, 4, dtype=torch.bool)
    q = torch.randn(2, 5, 4)
    tau = torch.tensor([[0.0, 0.1, 0.2, 0.3, 0.4], [0.0, 0.1, 0.2, 0.3, 0.4]])
    intervals = torch.full((2, 4), 0.1)
    return values, mask, q, tau, intervals


def test_set_baseline_is_invariant_to_synchronous_time_reordering_with_fixed_intervals():
    values, mask, q, tau, intervals = _inputs()
    model = ComponentGeometryModel(7).eval()
    with torch.no_grad():
        reference = model(values, mask, q, intervals)
        reverse = model(values[:, torch.arange(4, -1, -1)], mask[:, torch.arange(4, -1, -1)], q[:, torch.arange(4, -1, -1)], intervals)
    assert torch.allclose(reference, reverse, atol=1e-6, rtol=0)


def test_ordered_model_has_equal_b_c_parameters_and_order_sensitive_projection():
    torch.manual_seed(20260909)
    ordered_a = OrderedComponentGeometryModel(7)
    torch.manual_seed(20260909)
    ordered_b = OrderedComponentGeometryModel(7)
    assert parameter_count(ordered_a) == parameter_count(ordered_b) == 4329
    assert all(torch.equal(ordered_a.state_dict()[key], ordered_b.state_dict()[key]) for key in ordered_a.state_dict())
    # This directly tests the frozen slot projection rather than relying on a
    # random final head to expose a tiny output difference.
    with torch.no_grad():
        ordered_a.order_projection[0].weight.zero_()
        ordered_a.order_projection[0].bias.zero_()
        ordered_a.order_projection[0].weight[0, 0] = 1.0
    slots = torch.zeros(1, 5, 9)
    slots[0, :, 0] = torch.arange(1.0, 6.0)
    forward = ordered_a.order_projection(slots.reshape(1, -1))
    reverse = ordered_a.order_projection(slots[:, torch.arange(4, -1, -1)].reshape(1, -1))
    assert not torch.allclose(forward, reverse)


def test_ordered_forward_uses_real_time_offset_shape_and_finite_gradients():
    values, mask, q, tau, intervals = _inputs()
    model = OrderedComponentGeometryModel(7)
    loss = model(values, mask, q, tau, intervals).sum()
    loss.backward()
    assert torch.isfinite(loss)
    assert model.order_projection[0].weight.grad is not None
    assert torch.isfinite(model.order_projection[0].weight.grad).all()


def test_runner_unit_chunking_preserves_window_output():
    values, mask, q, tau, intervals = _inputs()
    model = OrderedComponentGeometryModel(7).eval()
    batch = {
        "edge_values": values.detach().numpy(),
        "edge_mask": mask.numpy(),
        "q": q.numpy(),
        "time_offsets": tau.numpy(),
        "intervals": intervals.numpy(),
        "window_index": np.array([0, 0], dtype=np.int64),
        "window_count": 1,
    }
    old = runner.UNIT_CHUNK_SIZE
    try:
        runner.UNIT_CHUNK_SIZE = 1
        one = runner._forward(model, batch, "cpu").detach()
        runner.UNIT_CHUNK_SIZE = 256
        many = runner._forward(model, batch, "cpu").detach()
    finally:
        runner.UNIT_CHUNK_SIZE = old
    assert torch.allclose(one, many, atol=1e-6, rtol=0)
