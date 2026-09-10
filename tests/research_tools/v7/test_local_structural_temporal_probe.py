"""Contract tests for the bounded local structural temporal pilot."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

import research_tools.v7.local_structural_temporal_probe.run_pilot as run_pilot
from research_tools.v7.local_structural_temporal_probe.model import (
    build_batch,
    fit_weighted_standardizer,
    source_class_window_weights,
    weighted_window_bce,
    window_label,
)
from research_tools.v7.local_structural_temporal_probe.representation import (
    ARM_NAMES,
    COMPONENT_CONFIG,
    arm_inputs_for_triplet,
    build_window_support,
    compute_local_derivatives,
    support_arrays,
)
from research_tools.v7.local_structural_temporal_probe.run_pilot import paired_source_bootstrap


def _sequence(*, scale_after: float = 1.0, timestamps: np.ndarray | None = None, track_count: int = 4):
    if timestamps is None:
        timestamps = np.arange(15, dtype=np.float64) * 0.1
    base = np.asarray([[0.0, 0.0, 1.0], [0.8, 0.0, 1.0], [0.0, 0.8, 1.0], [5.0, 5.0, 1.0]], dtype=np.float64)[:track_count]
    xyz = []
    for timestamp in timestamps:
        factor = scale_after if timestamp >= 0.5 else 1.0
        xyz.append(base * factor + np.asarray([0.02 * timestamp, 0.0, 0.0]))
    xyz = np.asarray(xyz, dtype=np.float32)
    valid = np.ones(xyz.shape[:2], dtype=bool)
    return SimpleNamespace(
        frame_indices=np.arange(len(timestamps), dtype=np.int64) + 100,
        timestamps_s=np.asarray(timestamps, dtype=np.float64),
        xyz=xyz,
        geometry_validity=valid,
        track_ids=np.arange(track_count, dtype=np.int64) + 10,
    )


def _config():
    return type(COMPONENT_CONFIG)(1.5, 0.2, minimum_size=3, minimum_overlap=4)


def test_triplets_keep_the_same_pair_identity_and_common_members():
    support = build_window_support(_sequence(track_count=3), window_start_s=0.0, component_config=_config())
    assert support["valid_triplet_count"] == 3
    pair_sets = {tuple(tuple(pair) for pair in row["pair_ids"]) for row in support["triplets"]}
    member_sets = {tuple(row["common_track_ids"]) for row in support["triplets"]}
    assert len(pair_sets) == 1
    assert member_sets == {(10, 11, 12)}


def test_missing_target_does_not_bridge_noncontiguous_slots():
    timestamps = np.asarray([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 0.8, 0.9, 1.0])
    support = build_window_support(_sequence(timestamps=timestamps, track_count=3), window_start_s=0.0, component_config=_config())
    assert support["target_matches"][1]["status"] == "MISSING_TARGET_FRAME"
    assert [row["target_slots"] for row in support["triplets"]] == [[2, 3, 4]]
    assert any(item["reason"] == "TARGET_FRAME_MISSING" for item in support["invalid_reasons"])


def test_invalid_member_is_not_converted_to_a_distance_change():
    sequence = _sequence(track_count=3)
    sequence.geometry_validity[7, 0] = False
    support = build_window_support(sequence, window_start_s=0.0, component_config=_config())
    assert support["valid_triplet_count"] == 0
    assert any(item["reason"] == "COMMON_VALID_MEMBERS_LT3" for item in support["invalid_reasons"])


def test_rigid_translation_and_rotation_leave_states_unchanged():
    sequence = _sequence(track_count=3)
    reference = build_window_support(sequence, window_start_s=0.0, component_config=_config())
    theta = 0.47
    rotation = np.asarray([[np.cos(theta), -np.sin(theta), 0.0], [np.sin(theta), np.cos(theta), 0.0], [0.0, 0.0, 1.0]])
    transformed = _sequence(track_count=3)
    transformed.xyz = np.einsum("ab,tib->tia", rotation, sequence.xyz) + np.asarray([2.0, -1.0, 0.5], dtype=np.float32)
    shifted = build_window_support(transformed, window_start_s=0.0, component_config=_config())
    np.testing.assert_allclose(
        np.asarray([row["states"] for row in support_arrays(reference)["triplets"]]),
        np.asarray([row["states"] for row in support_arrays(shifted)["triplets"]]),
        rtol=1e-6,
        atol=1e-6,
    )


def test_fixed_history_scale_retains_uniform_evaluation_contraction():
    support = build_window_support(_sequence(scale_after=0.5, track_count=3), window_start_s=0.0, component_config=_config())
    states = np.asarray([row["states"] for row in support_arrays(support)["triplets"]])
    assert states.shape[0] == 3
    assert np.all(states[:, 0, 0] < 0.8)


def test_nonuniform_triplet_derivatives_use_actual_intervals():
    states = np.asarray([[0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0], [3.5, 3.5, 3.5, 3.5]])
    center, first, second = compute_local_derivatives(states, [0.0, 0.2, 0.7])
    np.testing.assert_allclose(center, 1.0)
    np.testing.assert_allclose(first, 5.0)
    np.testing.assert_allclose(second, 0.0, atol=1e-12)


def test_components_are_constructed_from_history_not_evaluation_positions():
    sequence = _sequence(track_count=4)
    sequence.xyz[7, 3] = sequence.xyz[7, 0]
    sequence.xyz[8, 3] = sequence.xyz[8, 0]
    sequence.xyz[9, 3] = sequence.xyz[9, 0]
    support = build_window_support(sequence, window_start_s=0.0, component_config=_config())
    assert all(13 not in row["track_ids"] for row in support["components"])


def test_arm_shapes_and_six_joint_permutations_are_fixed():
    triplet = {"states": np.asarray([[1.0, 2.0, 3.0, 4.0], [2.0, 3.0, 4.0, 5.0], [4.0, 5.0, 6.0, 7.0]]), "timestamps_s": [0.0, 0.2, 0.7]}
    arms = arm_inputs_for_triplet(triplet)
    assert set(arms) == set(ARM_NAMES)
    assert arms["UNORDERED_STATE"].shape == (3, 12)
    assert arms["ORDERED_FIRST"].shape == (1, 12)
    assert arms["ORDERED_SECOND"].shape == (1, 12)
    assert arms["PERMUTED_SECOND"].shape == (6, 12)
    np.testing.assert_allclose(arms["ORDERED_FIRST"][0, 8:], 0.0)


def test_window_label_does_not_create_local_component_labels():
    assert window_label({"kind": "MANIP", "role": "real"}) == 0
    assert window_label({"kind": "MANIP", "role": "fake"}) == 1
    assert window_label({"kind": "CTRL", "role": "fake"}) is None


def _example(source: str, role: str, window_id: str):
    triplet = {"states": np.asarray([[1.0, 2.0, 3.0, 4.0], [1.1, 2.1, 3.1, 4.1], [1.3, 2.3, 3.3, 4.3]]), "timestamps_s": [0.0, 0.1, 0.25], "component_index": 0, "pair_ids": [[1, 2]], "common_track_ids": [1, 2, 3]}
    return {"source_id": source, "role": role, "kind": "MANIP", "window_id": window_id, "triplets": [triplet]}


def test_source_class_weights_equalize_each_source_and_class():
    rows = [_example("a", "real", "a-r"), _example("a", "fake", "a-f"), _example("b", "real", "b-r"), _example("b", "fake", "b-f")]
    weights = source_class_window_weights(rows)
    assert np.allclose(weights, 1.0)


def test_standardizer_uses_only_the_values_given_by_training_fold():
    values = np.zeros((2, 12), dtype=np.float64)
    values[1] = 2.0
    standardizer = fit_weighted_standardizer(values, np.ones(2))
    assert standardizer.mean[0] == pytest.approx(1.0)
    assert np.all(standardizer.transform(values)[0] < 0)
    assert np.all(standardizer.transform(values)[1] > 0)


def test_training_batch_and_standardizer_can_exclude_held_out_source():
    training = [_example("a", "real", "a-r"), _example("a", "fake", "a-f"), _example("b", "real", "b-r"), _example("b", "fake", "b-f")]
    held_out = _example("held-out", "fake", "held-f")
    held_out["triplets"][0]["states"] = held_out["triplets"][0]["states"] + 100.0
    raw, observation_weights = build_batch(training, "ORDERED_SECOND", observation_weights=True)
    standardizer = fit_weighted_standardizer(raw.inputs, observation_weights)
    assert "held-out" not in {row["source_id"] for row in training}
    assert standardizer.mean[0] < 2.0
    held_batch, _ = build_batch([held_out], "ORDERED_SECOND", standardizer=standardizer)
    assert held_batch.inputs.shape[0] == 1


def _imbalanced_training_examples():
    rows = []
    for source, role, count in (("a", "real", 1), ("a", "fake", 3), ("b", "real", 2), ("b", "fake", 1)):
        for index in range(count):
            row = _example(source, role, f"{source}-{role}-{index}")
            row["pair_id"] = f"pair-{source}"
            rows.append(row)
    return rows


def test_actual_training_batch_carries_source_class_window_weights():
    training = _imbalanced_training_examples()
    raw_batch, observation_weights = build_batch(training, "ORDERED_SECOND", observation_weights=True)
    standardizer = fit_weighted_standardizer(raw_batch.inputs, observation_weights)
    train_batch, _ = build_batch(training, "ORDERED_SECOND", standardizer=standardizer, observation_weights=True)
    expected = {"a-real-0": 7.0 / 4.0, "a-fake-0": 7.0 / 12.0, "a-fake-1": 7.0 / 12.0, "a-fake-2": 7.0 / 12.0, "b-real-0": 7.0 / 8.0, "b-real-1": 7.0 / 8.0, "b-fake-0": 7.0 / 4.0}
    np.testing.assert_allclose(train_batch.window_weights, [expected[row["window_id"]] for row in training])
    assert not np.allclose(train_batch.window_weights, 1.0)
    for source, role in (("a", "real"), ("a", "fake"), ("b", "real"), ("b", "fake")):
        indices = [index for index, row in enumerate(training) if row["source_id"] == source and row["role"] == role]
        assert float(np.sum(train_batch.window_weights[indices])) == pytest.approx(7.0 / 4.0)


def test_weighted_window_bce_is_not_window_equivalent():
    logits = torch.tensor([-2.0, -0.5, 0.75, 2.0], dtype=torch.float32)
    labels = torch.tensor([0.0, 1.0, 0.0, 1.0], dtype=torch.float32)
    weights = torch.tensor([2.0, 0.5, 0.5, 1.0], dtype=torch.float32)
    actual = weighted_window_bce(logits, labels, weights)
    per_window = nn.functional.binary_cross_entropy_with_logits(logits, labels, reduction="none")
    expected = torch.sum(per_window * weights) / torch.sum(weights)
    assert actual.item() == pytest.approx(expected.item())
    assert actual.item() != pytest.approx(float(torch.mean(per_window)))


def test_run_passes_weighted_training_batch_to_train_model(monkeypatch, tmp_path: Path):
    examples = _imbalanced_training_examples()
    input_rows = [
        {
            "window_id": row["window_id"],
            "pair_id": row["window_id"],
            "source_id": row["source_id"],
            "role": row["role"],
            "kind": row["kind"],
            "label": window_label(row),
            "support_status": "VALID",
            "invalid_reasons": [],
            "match_error_summary_s": {"max": 0.0},
            "actual_interval_summary_s": {"median": 0.1},
        }
        for row in examples
    ]
    details = {
        "population": {
            "all_source_ids": ["a", "b"],
            "invalid_support_sources": [],
            "selected_pairs": 2,
            "frozen_windows": len(examples),
            "completed_frontend_windows": len(examples),
            "valid_support_windows": len(examples),
            "support_source_ids": ["a", "b"],
            "source_artifact_root": "fixture",
            "frontend_window_results_sha256": "fixture",
            "selected_pairs_sha256": "fixture",
            "window_manifest_sha256": "fixture",
        },
        "input_rows": input_rows,
    }
    captured = []

    def fake_train_model(batch, *, seed, epochs=200):
        captured.append(np.array(batch.window_weights, copy=True))
        return object(), {"seed": seed, "epochs": epochs, "initial_loss": 0.0, "final_loss": 0.0, "min_loss": 0.0, "loss_history": [0.0]}

    monkeypatch.setattr(run_pilot, "load_frozen_examples", lambda source_root: (examples, [], details))
    monkeypatch.setattr(run_pilot, "train_model", fake_train_model)
    monkeypatch.setattr(run_pilot, "score_model", lambda model, batch: np.zeros(batch.n_windows, dtype=np.float64))
    monkeypatch.setattr(run_pilot, "serialize_model", lambda model, standardizer: {})
    monkeypatch.setattr(run_pilot, "SEEDS", (20260909,))
    run_pilot.run(source_root=tmp_path / "source", output_root=tmp_path / "output")
    assert captured
    assert all(not np.allclose(weights, 1.0) for weights in captured)
    assert any(np.isclose(weights.max(), 2.0) for weights in captured)


def test_source_bootstrap_is_reproducible_and_paired():
    rows = [
        {"source_id": "s0", "UNORDERED_STATE_auroc": 0.2, "ORDERED_FIRST_auroc": 0.5, "ORDERED_SECOND_auroc": 0.6, "PERMUTED_SECOND_auroc": 0.3},
        {"source_id": "s1", "UNORDERED_STATE_auroc": 0.7, "ORDERED_FIRST_auroc": 0.1, "ORDERED_SECOND_auroc": 0.9, "PERMUTED_SECOND_auroc": 0.8},
        {"source_id": "s2", "UNORDERED_STATE_auroc": 0.4, "ORDERED_FIRST_auroc": 0.9, "ORDERED_SECOND_auroc": 0.2, "PERMUTED_SECOND_auroc": 0.1},
        {"source_id": "s3", "UNORDERED_STATE_auroc": 0.8, "ORDERED_FIRST_auroc": 0.3, "ORDERED_SECOND_auroc": 0.4, "PERMUTED_SECOND_auroc": 0.9},
    ]
    left = paired_source_bootstrap(rows)
    right = paired_source_bootstrap(rows)
    assert left == right
    assert left["N"] == 4
    for name, other in (("C-A", "UNORDERED_STATE"), ("C-D", "PERMUTED_SECOND"), ("C-B", "ORDERED_FIRST")):
        raw_delta = np.asarray([row["ORDERED_SECOND_auroc"] - row[f"{other}_auroc"] for row in rows], dtype=np.float64)
        assert left["gains"][name]["mean"] == pytest.approx(float(np.mean(raw_delta)))
        assert left["gains"][name]["mean"] == pytest.approx(left["arms"]["ORDERED_SECOND"]["mean"] - left["arms"][other]["mean"])
        rng = np.random.default_rng(20260909)
        indices = rng.integers(0, len(rows), size=(10000, len(rows)))
        c = np.asarray([row["ORDERED_SECOND_auroc"] for row in rows], dtype=np.float64)
        other_values = np.asarray([row[f"{other}_auroc"] for row in rows], dtype=np.float64)
        expected_ci = np.percentile(np.mean(c[indices] - other_values[indices], axis=1), [2.5, 97.5])
        np.testing.assert_allclose(left["gains"][name]["ci95"], expected_ci)
