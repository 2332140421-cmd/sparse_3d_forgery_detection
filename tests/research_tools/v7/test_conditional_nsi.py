"""Contract tests for the frozen conditional-NSI diagnostic."""

from __future__ import annotations

import inspect
import json
from pathlib import Path

import numpy as np
import pytest

from research_tools.v7.conditional_nsi.activity_condition import (
    activity_bin,
    activity_condition,
    anomaly_transform,
    ks_distance_to_uniform,
    tertile_cutpoints,
)
from research_tools.v7.conditional_nsi.analyze_conditional_nsi import (
    _fake_evaluation,
    _real_dependence,
    _score_loso,
    aggregate_window_scores,
)
from research_tools.v7.conditional_nsi.artifact_reconstruction import reconstruct_triplet
from research_tools.v7.conditional_nsi.source_balanced_ecdf import source_balanced_ecdf, source_balanced_support


def _particle_npz(tmp_path: Path, *, invalid_frame: int | None = None, frame_indices: np.ndarray | None = None) -> tuple[Path, np.ndarray, np.ndarray]:
    timestamps = np.asarray([0.0, 0.1, 0.3, 0.6, 1.0, 1.5, 2.1, 2.8, 3.6, 4.5], dtype=np.float64)
    indices = np.arange(timestamps.size, dtype=np.int64) if frame_indices is None else np.asarray(frame_indices, dtype=np.int64)
    xyz = np.asarray([[[0.0, 0.0, 1.0], [1.0, 0.0, 1.0], [0.0, 1.0, 1.0]]] * timestamps.size, dtype=np.float32)
    validity = np.ones((timestamps.size, 3), dtype=bool)
    if invalid_frame is not None:
        validity[invalid_frame, :] = False
    path = tmp_path / "particle.npz"
    np.savez(path, xyz=xyz, geometry_validity=validity, timestamps_s=timestamps, frame_indices=indices)
    return path, timestamps, indices


def _synthetic_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for source_index, source in enumerate(("A", "B", "C")):
        for index, condition in enumerate(np.linspace(0.1, 0.9, 9)):
            for role in ("real", "fake"):
                rows.append(
                    {
                        "source_id": source,
                        "role": role,
                        "kind": "MANIP" if index % 2 else "CTRL",
                        "label": f"label-{source}-{index}",
                        "pair_id": source,
                        "window_id": f"{source}-{role}-{index}",
                        "anchor_fraction": 0.25,
                        "component_index": 0,
                        "center_index": index + 1,
                        "I_stored": float(index) / 10.0 + source_index / 100.0,
                        "C": float(condition),
                    }
                )
    return rows


def test_condition_is_exactly_lagged_norm_over_sqrt_pair_count():
    assert activity_condition(6.0, 9) == pytest.approx(2.0)


def test_condition_is_stable_under_duplicate_relation_coordinates():
    assert activity_condition(2.0, 4) == pytest.approx(activity_condition(2.0 * np.sqrt(2.0), 8))


def test_condition_rejects_nonpositive_pair_count():
    with pytest.raises(ValueError):
        activity_condition(1.0, 0)


def test_tertile_cutpoints_and_bins_are_deterministic():
    cuts = tertile_cutpoints([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
    assert cuts == pytest.approx((5.0 / 3.0, 10.0 / 3.0))
    assert [activity_bin(value, cuts) for value in (0.0, 2.0, 5.0)] == ["LOW", "MID", "HIGH"]


def test_source_balanced_ecdf_equalizes_sources_not_triplets():
    q, support = source_balanced_ecdf(0.5, {"long": [0.0] * 100, "short": [1.0]})
    assert q == pytest.approx(0.5)
    assert support["distinct_source_count"] == 2
    assert support["observation_count"] == 101
    assert support["effective_n"] == pytest.approx(3.9603960396)


def test_source_balanced_ecdf_uses_half_rank_smoothing_and_no_endpoints():
    q_low, _ = source_balanced_ecdf(0.0, {"A": [1.0, 2.0]})
    q_high, _ = source_balanced_ecdf(2.0, {"A": [1.0, 2.0]})
    assert 0.0 < q_low < 1.0
    assert 0.0 < q_high < 1.0


def test_source_balanced_support_reports_equal_source_effective_n():
    support = source_balanced_support({"A": [1.0, 2.0], "B": [3.0]})
    assert support == {"distinct_source_count": 2, "observation_count": 3, "effective_n": pytest.approx(2.6666666667)}


def test_anomaly_transform_is_the_frozen_two_sided_transform():
    assert anomaly_transform(0.1) == pytest.approx(-np.log(0.2))
    assert anomaly_transform(0.9) == pytest.approx(anomaly_transform(0.1))


def test_ks_distance_is_descriptive_uniform_distance():
    assert ks_distance_to_uniform([0.25, 0.5, 0.75]) == pytest.approx(0.25)


def test_reconstruction_uses_sorted_component_pair_identity_and_pts(tmp_path):
    path, timestamps, indices = _particle_npz(tmp_path)
    forward = reconstruct_triplet(path, [0, 1, 2], 4, timestamps, indices)
    reverse = reconstruct_triplet(path, [2, 0, 1], 4, timestamps, indices)
    assert forward == reverse
    assert forward["M"] == 3
    assert forward["h0_s"] == pytest.approx(0.4)
    assert forward["h1_s"] == pytest.approx(0.5)
    assert forward["I_reconstructed"] == pytest.approx(0.0)
    assert np.isfinite(forward["C"])


def test_reconstruction_does_not_create_derivative_across_missing_observation(tmp_path):
    path, timestamps, indices = _particle_npz(tmp_path, invalid_frame=4)
    from research_tools.v7.conditional_nsi.artifact_reconstruction import _reconstruct_component

    centers = _reconstruct_component(path, [0, 1, 2], timestamps, indices)
    assert 3 not in centers
    assert 4 not in centers
    assert 5 not in centers


def test_reconstruction_does_not_create_derivative_across_source_frame_gap(tmp_path):
    gap_indices = np.asarray([0, 1, 2, 4, 5, 6, 7, 8, 9, 10], dtype=np.int64)
    path, timestamps, _ = _particle_npz(tmp_path, frame_indices=gap_indices)
    from research_tools.v7.conditional_nsi.artifact_reconstruction import _reconstruct_component

    centers = _reconstruct_component(path, [0, 1, 2], timestamps, gap_indices)
    assert 2 not in centers


def test_loso_training_references_exclude_held_out_source(tmp_path):
    rows = _synthetic_rows()
    scored, support, folds, _ = _score_loso(rows, ["A", "B", "C"], tmp_path)
    assert len(scored) == 54
    assert all("A" not in fold["training_source_ids"] for fold in folds if fold["held_out_source"] == "A")
    assert all(row["held_out_source"] == row["source_id"] for row in scored)
    assert len(support) == 9


def test_loso_records_zero_fake_and_heldout_real_calibration_counts(tmp_path):
    _, _, folds, _ = _score_loso(_synthetic_rows(), ["A", "B", "C"], tmp_path)
    assert all(fold["fake_calibration_count"] == 0 for fold in folds)
    assert all(fold["heldout_real_calibration_count"] == 0 for fold in folds)


def test_loso_cutpoints_use_training_real_only(tmp_path):
    rows = _synthetic_rows()
    _, _, folds, _ = _score_loso(rows, ["A", "B", "C"], tmp_path)
    fold_a = next(fold for fold in folds if fold["held_out_source"] == "A")
    expected = tertile_cutpoints([row["C"] for row in rows if row["role"] == "real" and row["source_id"] != "A"])
    assert (fold_a["cut_low"], fold_a["cut_high"]) == pytest.approx(expected)


def test_scoring_is_invariant_to_labels(tmp_path):
    rows = _synthetic_rows()
    first, _, _, _ = _score_loso(rows, ["A", "B", "C"], tmp_path / "first")
    changed = [dict(row, label="changed", kind="UNUSED_FOR_SCORING") for row in rows]
    second, _, _, _ = _score_loso(changed, ["A", "B", "C"], tmp_path / "second")
    first_values = [(row["window_id"], float(row["q_U0"]), float(row["q_U1"])) for row in first]
    second_values = [(row["window_id"], float(row["q_U0"]), float(row["q_U1"])) for row in second]
    assert [row[0] for row in first_values] == [row[0] for row in second_values]
    assert np.asarray([row[1:] for row in first_values]) == pytest.approx(np.asarray([row[1:] for row in second_values]))


def test_component_q90_then_equal_component_window_median():
    rows = []
    for component, values in ((0, (0.0, 1.0)), (1, (2.0, 3.0))):
        for center, value in enumerate(values):
            rows.append(
                {
                    "held_out_source": "A",
                    "source_id": "A",
                    "role": "real",
                    "kind": "CTRL",
                    "pair_id": "A",
                    "label": "x",
                    "anchor_fraction": 0.25,
                    "window_id": "w",
                    "component_index": component,
                    "center_index": center,
                    "A_U0": value,
                    "A_U1": value,
                }
            )
    output = aggregate_window_scores(rows)
    assert len(output) == 1
    assert output[0]["A_U0"] == pytest.approx(1.9)
    assert output[0]["A_U1"] == pytest.approx(1.9)


def test_real_dependence_is_source_level_and_real_only(tmp_path):
    rows = _synthetic_rows()
    result = _real_dependence(rows, ["A", "B", "C"], tmp_path)
    assert result["N_source"] == 3
    assert len(result["per_source"]) == 3


def test_fake_evaluation_changes_only_after_scores_are_available(tmp_path):
    rows = _synthetic_rows()
    scored, _, _, _ = _score_loso(rows, ["A", "B", "C"], tmp_path / "score")
    windows = aggregate_window_scores(scored)
    result = _fake_evaluation(windows, ["A", "B", "C"], tmp_path / "eval")
    assert result["fake_fitting_count"] == 0
    assert "NOT_FULL_VIDEO" in result["scope"]


def test_no_frontend_or_formal_src_dependency_in_reconstruction_module():
    source = inspect.getsource(reconstruct_triplet)
    assert "frontend" not in source.lower()
    assert not Path("src/sparse3d_forgery").joinpath("v7").exists()


def test_frozen_population_and_window_counts_are_unchanged():
    root = Path("/root/autodl-tmp/data/sparse_3d_forgery_detection/derived/v7_activityforensics_paired_second_order_pilot_v1")
    if not root.exists():
        pytest.skip("frozen V7 pilot artifacts are not mounted")
    assert len(json.loads((root / "manifests/selected_pairs.json").read_text())) == 16
    assert len(json.loads((root / "manifests/window_manifest.json").read_text())) == 192


def test_condition_contract_excludes_second_condition_and_future_velocity():
    source = Path("research_tools/v7/conditional_nsi/analyze_conditional_nsi.py").read_text()
    assert "C = ||v_minus|| / sqrt(M)" in source
    assert "v_plus_norm" not in source[source.index("def _score_loso"):source.index("def aggregate_window_scores")]
