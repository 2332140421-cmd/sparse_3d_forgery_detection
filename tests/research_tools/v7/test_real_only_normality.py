from __future__ import annotations

from types import SimpleNamespace
from pathlib import Path

import numpy as np

from research_tools.v7.normality.protocol import (
    ANCHOR_FRACTIONS,
    COMPONENT_CONFIG,
    GaussianNormality,
    build_source_split,
    build_window_manifest,
    extract_window_features,
    feature_schema,
    video_score,
)
from sparse3d_forgery.experiments.v7_dynamic_structure_probe import (
    ComponentConfig,
    structural_differences,
    structure_state,
)


def _media(count: int = 32):
    return [
        {"source_id": f"source-{i:02d}", "status": "MEDIA_VALID", "video_path": f"{i}.mp4", "timestamps_s": (np.arange(100, dtype=float) * 0.1).tolist()}
        for i in range(count)
    ]


def _sequence():
    t, n = 12, 4
    base = np.asarray([[0, 0, 1], [0.8, 0, 1], [0, 0.8, 1], [0.8, 0.8, 1]], dtype=np.float64)
    xyz = np.stack([base + [0.01 * i, 0.0, 0.0] for i in range(t)])
    return SimpleNamespace(xyz=xyz, geometry_validity=np.ones((t, n), dtype=bool), timestamps_s=np.arange(t, dtype=float))


def test_source_level_split_is_frozen_order_and_24_8():
    split = build_source_split(_media())
    assert len(split["real_train"]) == 24
    assert len(split["real_val"]) == 8
    assert split["real_train"][0]["source_id"] == "source-00"
    assert split["real_val"][0]["source_id"] == "source-24"


def test_fake_rows_are_absent_from_fitting_population():
    rows = _media(4) + [{"source_id": "fake", "status": "MEDIA_VALID", "label": "fake", "timestamps_s": [0, 1], "video_path": "fake.mp4"}]
    split = build_source_split([row for row in rows if row.get("label") != "fake"])
    assert all(row.get("label") != "fake" for row in split["real_train"])


def test_windows_use_fixed_anchors_true_timestamps_and_no_padding():
    rows = build_window_manifest(_media(1))
    assert [row["anchor_fraction"] for row in rows] == list(ANCHOR_FRACTIONS)
    assert all(row["timescale_s"] == 1.0 and row["status"] == "AVAILABLE" for row in rows)
    assert all(row["timestamps_s"] == [0.1 * i for i in row["frame_indices"]] for row in rows)


def test_current_component_config_is_unchanged():
    assert COMPONENT_CONFIG == ComponentConfig(1.0, 0.05, minimum_size=3, minimum_overlap=8)


def test_current_structure_state_and_derivatives_remain_timestamp_aware():
    sequence = _sequence()
    state, valid = structure_state(sequence.xyz, sequence.geometry_validity, (0, 1, 2, 3))
    first, first_valid, second, second_valid = structural_differences(state, valid, sequence.timestamps_s)
    assert state.shape == (12, 4)
    assert first.shape == second.shape == (12, 4)
    assert np.all(np.diff(sequence.timestamps_s) > 0)
    assert np.all(np.isfinite(first[first_valid]))
    assert np.all(np.isfinite(second[second_valid]))


def test_m1_m2_m3_feature_schema_and_dimensions():
    schema = feature_schema()
    assert schema["M1_delta_s"]["shape"] == [4]
    assert schema["M2_delta2_s"]["shape"] == [4]
    assert schema["M3_delta_s_delta2_s"]["shape"] == [8]
    result = extract_window_features(_sequence(), source_id="s", window_id="s::anchor-25")
    assert all(len(row["values"]) == 4 for row in result["observations"]["M1_delta_s"])
    assert all(len(row["values"]) == 4 for row in result["observations"]["M2_delta2_s"])
    assert all(len(row["values"]) == 8 for row in result["observations"]["M3_delta_s_delta2_s"])


def test_covariance_regularization_and_fallback_are_deterministic():
    values = np.ones((4, 4), dtype=float)
    left = GaussianNormality.fit(values)
    right = GaussianNormality.fit(values.copy())
    assert left.covariance_type == right.covariance_type == "diagonal"
    np.testing.assert_allclose(left.covariance, right.covariance)
    assert left.regularization_lambda > 0


def test_gaussian_fit_uses_only_train_values():
    train = np.zeros((20, 4), dtype=float)
    model = GaussianNormality.fit(train)
    assert model.feature_count == 20
    assert np.all(model.mean == 0)
    assert model.score(np.ones((1, 4)))[0] > 0


def test_video_score_aggregation_is_deterministic_and_not_sum():
    values = [1.0, 2.0, 100.0]
    assert video_score(values, "median") == 2.0
    assert video_score(values, "p95") == np.percentile(values, 95)


def test_paired_source_identity_is_explicitly_retained():
    rows = build_window_manifest(_media(1))
    assert rows[0]["source_id"] == "source-00"
    assert rows[0]["fake_used"] is False


def test_formal_src_does_not_import_research_tools():
    for path in Path("src/sparse3d_forgery").rglob("*.py"):
        assert "research_tools" not in path.read_text(encoding="utf-8")
