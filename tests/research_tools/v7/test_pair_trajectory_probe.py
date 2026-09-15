from __future__ import annotations

import numpy as np
import pytest

from research_tools.v7.pair_trajectory_probe.runner import (
    _aggregate_window_from_relation,
    _atomic_json,
    _condition_features,
    _summary_from_distances,
    _stable_permutation,
    Budget,
    _window_weights,
)


def _toy_data() -> dict[str, object]:
    distances = np.asarray([[1.0, 2.0, 3.0, 4.0, 5.0], [2.0, 4.0, 6.0, 8.0, 10.0]], dtype=np.float64)
    intervals = np.asarray([[0.1, 0.2, 0.3, 0.4], [0.1, 0.2, 0.3, 0.4]], dtype=np.float64)
    return {
        "distances": distances,
        "intervals": intervals,
        "relation_window": np.asarray([0, 0], dtype=np.int64),
        "relation_unit": np.asarray([0, 0], dtype=np.int64),
        "units": [{"unit_id": "W::R::local_0", "window_id": "W", "local_group_id": 0, "relation_start": 0, "relation_end": 2}],
        "windows": [{"window_id": "W", "relation_start": 0, "relation_end": 2}],
    }


def test_id_shuffle_preserves_per_time_multiset_and_summary() -> None:
    data = _toy_data()
    original = data["distances"]
    features, _ = _condition_features(data, "PAIR_ID_SHUFFLE")
    shuffled = features[:, :5]
    for time_index in range(5):
        assert np.array_equal(np.sort(original[:, time_index]), np.sort(shuffled[:, time_index]))
    assert np.allclose(_summary_from_distances(original), _summary_from_distances(shuffled), atol=1e-12)
    assert not np.array_equal(original, shuffled)


def test_time_shuffle_preserves_each_relation_distance_multiset() -> None:
    data = _toy_data()
    features, log = _condition_features(data, "PAIR_TIME_SHUFFLE")
    assert log["intervals_unchanged"] is True
    assert np.all(np.sort(data["distances"], axis=1) == np.sort(features[:, :5], axis=1))
    assert not np.array_equal(_stable_permutation(5, "W", "time", seed=20260909), np.arange(5))


def test_source_class_weights_equalize_source_class_totals() -> None:
    weights = _window_weights([
        {"source_id": "A", "label": 0}, {"source_id": "A", "label": 1},
        {"source_id": "B", "label": 0}, {"source_id": "B", "label": 1},
    ])
    assert np.isclose(weights.mean(), 1.0)
    assert np.isclose(weights[0] + weights[1], weights[2] + weights[3])


def test_pair_row_order_does_not_change_window_aggregation() -> None:
    import torch

    relation_logits = torch.tensor([0.2, 0.6, -0.1, 0.4], dtype=torch.float64)
    relation_units = torch.tensor([0, 0, 1, 1], dtype=torch.long)
    unit_windows = torch.tensor([0, 0], dtype=torch.long)
    expected = _aggregate_window_from_relation(relation_logits, relation_units, unit_windows, 1, 2)
    order = torch.tensor([2, 0, 3, 1], dtype=torch.long)
    permuted = _aggregate_window_from_relation(relation_logits[order], relation_units[order], unit_windows, 1, 2)
    assert torch.allclose(expected, permuted, atol=0.0, rtol=0.0)


def test_budget_save_carries_cumulative_baseline_across_processes(tmp_path) -> None:
    state = tmp_path / "state"
    state.mkdir()
    (state / "budget.json").write_text(
        '{"budget_s": 100.0, "elapsed_before_this_process_s": 12.5}\n',
        encoding="utf-8",
    )
    first = Budget.load(tmp_path, 100.0)
    first.save("checkpoint")
    saved = __import__("json").loads((state / "budget.json").read_text(encoding="utf-8"))
    assert saved["elapsed_before_this_process_s"] >= 12.5
    second = Budget.load(tmp_path, 100.0)
    assert second.elapsed() >= saved["cumulative_s"]


def test_strict_json_rejects_non_finite_values(tmp_path) -> None:
    with pytest.raises(ValueError, match="non-finite JSON value"):
        _atomic_json(tmp_path / "summary.json", {"score": float("nan")})
