"""Contracts for the finite history-neighborhood pilot."""

import numpy as np
import torch

from research_tools.v7.local_context_pilot.model import LocalContextModel
from research_tools.v7.local_context_pilot.runner import _input_content_sha256, _rewire_edges, _time_permutation


def _forward(model, states, edges, *, edge_time_index=None):
    intervals = torch.full((1, 4), 0.1, dtype=torch.float32)
    return model(
        torch.as_tensor(states, dtype=torch.float32),
        intervals,
        torch.zeros(states.shape[0], dtype=torch.long),
        torch.as_tensor(edges[0], dtype=torch.long),
        torch.as_tensor(edges[1], dtype=torch.long),
        None if edge_time_index is None else torch.as_tensor(edge_time_index, dtype=torch.long),
        1,
    )


def test_context_model_is_invariant_to_node_and_edge_order() -> None:
    torch.manual_seed(7)
    model = LocalContextModel().eval()
    states = np.arange(3 * 5 * 4, dtype=np.float32).reshape(3, 5, 4) / 10.0
    src = np.asarray([0, 1, 2, 0], dtype=np.int64)
    dst = np.asarray([1, 2, 0, 2], dtype=np.int64)
    base = _forward(model, states, (src, dst))
    order = np.asarray([2, 0, 1], dtype=np.int64)
    inverse = np.empty(3, dtype=np.int64); inverse[order] = np.arange(3)
    permuted = _forward(model, states[order], (inverse[src[::-1]], inverse[dst[::-1]]))
    assert torch.allclose(base, permuted, atol=1e-6, rtol=0)


def test_self_temporal_does_not_read_other_node_state() -> None:
    torch.manual_seed(8)
    model = LocalContextModel().eval()
    states = np.ones((2, 5, 4), dtype=np.float32)
    states[1] *= 7.0
    edges = (np.asarray([0]), np.asarray([0]))  # SELF replaces neighbor with source
    intervals = torch.full((2, 4), 0.1, dtype=torch.float32)
    node_windows = torch.arange(2, dtype=torch.long)
    first = model(torch.as_tensor(states), intervals, node_windows, torch.as_tensor(edges[0]), torch.as_tensor(edges[1]), None, 2)
    changed = states.copy(); changed[1] = -13.0
    second = model(torch.as_tensor(changed), intervals, node_windows, torch.as_tensor(edges[0]), torch.as_tensor(edges[1]), None, 2)
    assert torch.allclose(first[0], second[0], atol=1e-6, rtol=0)


def test_rewire_preserves_degrees_and_forbids_duplicate_or_self_edges() -> None:
    src = np.asarray([0, 0, 1, 1, 2, 2], dtype=np.int64)
    dst = np.asarray([1, 2, 0, 2, 0, 1], dtype=np.int64)
    rewired, summary = _rewire_edges(src, dst, 3, "synthetic")
    assert len(rewired) == len(dst)
    assert np.bincount(dst, minlength=3).tolist() == np.bincount(rewired, minlength=3).tolist()
    assert all(int(left) != int(right) for left, right in zip(src, rewired))
    assert len(set(zip(src.tolist(), rewired.tolist()))) == len(src)
    assert 0.0 <= float(summary["changed_edge_ratio"]) <= 1.0


def test_time_permutation_is_nonidentity_and_preserves_time_slots() -> None:
    permutation = _time_permutation("synthetic-window")
    assert sorted(permutation.tolist()) == [0, 1, 2, 3, 4]
    assert permutation.tolist() != [0, 1, 2, 3, 4]
    values = np.arange(5 * 4).reshape(5, 4)
    assert sorted(map(tuple, values[permutation])) == sorted(map(tuple, values))


def test_empty_graph_has_finite_output() -> None:
    torch.manual_seed(9)
    model = LocalContextModel().eval()
    states = np.zeros((1, 5, 4), dtype=np.float32)
    output = _forward(model, states, (np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)))
    assert output.shape == (1,)
    assert torch.isfinite(output).all()


def test_input_content_identity_changes_when_numeric_cache_changes(tmp_path) -> None:
    relative = "inputs/window_inputs/example.npz"
    path = tmp_path / relative
    path.parent.mkdir(parents=True)
    path.write_bytes(b"first")
    manifest = [{"window_id": "example", "input_path": relative}]
    first = _input_content_sha256(tmp_path, manifest)
    path.write_bytes(b"second")
    second = _input_content_sha256(tmp_path, manifest)
    assert first != second
