import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research_tools.v8.full_coverage_observation_field.common import CELL_COUNT, GRID_H, GRID_W, cell_neighbors
from research_tools.v8.full_coverage_observation_field.model import FullCoverageModel
from research_tools.v8.full_coverage_observation_field.pipeline import _predict


def _batch(dim=13, b=2):
    return {
        "x": torch.randn(b, 16, CELL_COUNT, dim),
        "component_ids": torch.arange(CELL_COUNT).view(1, 1, -1).expand(b, 16, -1).clone(),
        "predecessor": torch.arange(CELL_COUNT).view(1, 1, -1).expand(b, 16, -1).clone(),
        "association_weight": torch.ones(b, 16, CELL_COUNT),
        "history_available": torch.ones(b, 16, CELL_COUNT, dtype=torch.bool),
        "delta_t": torch.full((b, 16), 1 / 30),
        "area": torch.ones(b, 16, CELL_COUNT),
    }


def test_full_coverage_cell_grid_and_edges():
    assert CELL_COUNT == GRID_H * GRID_W == 256
    e = cell_neighbors()
    assert e.shape == (480, 2)
    assert np.unique(e).size == CELL_COUNT


def test_missing_geometry_still_has_finite_output():
    m = FullCoverageModel("FULL", 13)
    b = _batch()
    b["x"][:, :, :, 3:6] = 0
    b["x"][:, :, :, 6] = 0
    y = m(**b)
    assert y.shape == (2,)
    assert torch.isfinite(y).all()


def test_component_and_predecessor_permutation_is_explicitly_synchronized():
    torch.manual_seed(7)
    m = FullCoverageModel("RGB_2D", 7).eval()
    b = _batch(7, 1)
    b2 = {k: v.clone() for k, v in b.items()}
    # Relabel components without changing cell positions or predecessor cells.
    b2["component_ids"] = (CELL_COUNT - 1) - b["component_ids"]
    # This test synchronizes all cell-keyed fields but does not claim arbitrary
    # component IDs are semantic model inputs.
    with torch.no_grad():
        assert torch.allclose(m(**b), m(**b2), atol=1e-5, rtol=1e-5)


def test_rgb_control_does_not_consume_geometry_component_labels():
    torch.manual_seed(11)
    m = FullCoverageModel("RGB_2D", 7).eval()
    b = _batch(7, 1)
    b2 = {k: v.clone() for k, v in b.items()}
    b2["component_ids"].fill_(0)
    with torch.no_grad():
        assert torch.allclose(m(**b), m(**b2), atol=1e-5, rtol=1e-5)


def test_rgb_control_has_no_geometry_input_dimension():
    assert FullCoverageModel("RGB_2D", 7).input_dim == 7
    assert FullCoverageModel("FULL", 13).input_dim == 13


def test_multi_to_multi_association_and_no_history_are_finite():
    torch.manual_seed(13)
    m = FullCoverageModel("FULL", 13)
    b = _batch(13, 1)
    b["predecessor"] = b["predecessor"].unsqueeze(-1).expand(-1, -1, -1, 4).clone()
    b["association_weight"] = torch.zeros(1, 16, CELL_COUNT, 4)
    b["history_available"].zero_()
    y = m(**b)
    assert torch.isfinite(y).all()


def test_vectorized_component_pool_matches_reference():
    torch.manual_seed(17)
    m = FullCoverageModel("FULL", 13)
    h = torch.randn(2, CELL_COUNT, 16)
    comp = torch.randint(0, 32, (2, CELL_COUNT))
    pooled, counts = m._pool_components(h, comp)
    reference = torch.zeros_like(h)
    ref_counts = []
    for bi in range(h.shape[0]):
        ids = comp[bi]
        unique = torch.unique(ids, sorted=True)
        ref_counts.append(unique.numel())
        for uid in unique:
            mask = ids == uid
            reference[bi, mask] = h[bi, mask].mean(dim=0)
    assert torch.allclose(pooled, reference, atol=1e-6, rtol=1e-6)
    assert torch.equal(counts, torch.tensor(ref_counts, dtype=torch.float32))


def test_predict_moves_metric_labels_back_to_cpu():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = FullCoverageModel("RGB_2D", 7).to(device).eval()
    batch = _batch(7, 1)
    batch.update({"label": torch.tensor([1.0]), "source": ["S"], "window_id": ["W"]})
    scores, ids, sources, labels = _predict(model, [batch], device)
    assert scores.shape == (1,)
    assert ids == ["W"] and sources == ["S"]
    assert labels.tolist() == [1]
