from __future__ import annotations

from pathlib import Path

import numpy as np

from research_tools.v7.detection.evaluate_pair2_detection import evaluate_video_scores
from research_tools.v7.detection.protocol import (
    ANCHOR_FRACTIONS,
    GENERATORS,
    MODEL_NAMES,
    auprc,
    auroc,
    build_pair2_window_manifest,
    frozen_pair2_population,
    load_frozen_models,
    paired_deltas,
    bootstrap_source_cluster,
    score_frontend_results,
)


def _row(source: str, role: str, generator: str | None, values: list[float]) -> dict[str, object]:
    return {
        "window": {
            "window_id": f"{source}::{role}::{generator or 'real'}::anchor-25",
            "video_id": f"{source}::{role}::{generator or 'real'}",
            "source_id": source,
            "role": role,
            "generator": generator,
        },
        "status": "COMPLETE",
        "observations": {name: [{"values": values[: (4 if name != "M3_delta_s_delta2_s" else 8)]}] for name in MODEL_NAMES},
    }


def test_pair2_population_is_first_frozen_sources_and_exact_generators():
    sources, pairs = frozen_pair2_population()
    assert len(sources) == 8
    assert len(pairs) == 32
    assert [row["ordinal"] for row in sources] == ["00015", "00018", "00033", "00042", "00044", "00079", "00082", "00087"]
    assert {row["generator"] for row in pairs} == set(GENERATORS)


def test_pair2_windows_keep_fixed_anchors_and_no_fake_selection():
    rows = [{
        "source_id": "s",
        "video_id": "s::real",
        "role": "real",
        "generator": None,
        "status": "MEDIA_VALID",
        "timestamps_s": (np.arange(100, dtype=float) * 0.1).tolist(),
    }]
    windows = build_pair2_window_manifest(rows)
    assert [row["anchor_fraction"] for row in windows] == list(ANCHOR_FRACTIONS)
    assert all(row["timescale_s"] == 1.0 for row in windows)
    assert all(row["fake_used"] is False for row in windows)


def test_frozen_models_load_without_refit_and_keep_identity():
    models, identity = load_frozen_models()
    assert tuple(models) == MODEL_NAMES
    assert identity["fitting_population"] == "real_train_only"
    assert identity["fake_count"] == 0
    assert identity["refit"] is False
    assert all(model.feature_count > 0 for model in models.values())


def test_pair2_scoring_uses_frozen_model_and_video_p95():
    models, _ = load_frozen_models()
    result = score_frontend_results([_row("s", "real", None, [0.0] * 8), _row("s", "fake", "musev", [1.0] * 8)], models)
    assert len(result["video_scores"]) == 2
    assert result["aggregation"] == {"primary": "p95", "sensitivity": "median", "sum_used": False}
    assert all(row["M1_delta_s"]["video_p95"] is not None for row in result["video_scores"])


def test_auroc_and_auprc_are_label_order_independent():
    assert auroc([0, 1], [0.1, 0.9]) == 1.0
    assert auroc([1, 0], [0.9, 0.1]) == 1.0
    assert np.isclose(auprc([0, 1], [0.1, 0.9]), 1.0)


def test_paired_delta_is_fake_minus_same_source_real():
    rows = [
        {"source_id": "s", "role": "real", "generator": None, "M1_delta_s": {"video_p95": 2.0}},
        {"source_id": "s", "role": "fake", "generator": "musev", "M1_delta_s": {"video_p95": 5.0}},
    ]
    paired = paired_deltas(rows, "M1_delta_s")
    assert paired["count"] == 1
    assert paired["median"] == 3.0
    assert paired["positive_fraction"] == 1.0
    assert paired["by_generator"]["musev"]["median"] == 3.0


def test_detection_evaluation_keeps_real_only_threshold_source():
    rows = []
    for index in range(4):
        source = f"s{index}"
        rows.append({"source_id": source, "role": "real", "generator": None, **{name: {"video_p95": 0.1} for name in MODEL_NAMES}})
        rows.append({"source_id": source, "role": "fake", "generator": "musev", **{name: {"video_p95": 1.0e9} for name in MODEL_NAMES}})
    result = evaluate_video_scores(rows)
    assert result["thresholds"]["label"] == "REAL-ONLY CALIBRATED THRESHOLD"
    assert result["models"]["M1_delta_s"]["auroc"] == 1.0
    assert result["models"]["M1_delta_s"]["paired"]["positive_fraction"] == 1.0
    assert result["models"]["M1_delta_s"]["operating_point"]["fake_tpr"] == 1.0


def test_source_cluster_bootstrap_uses_source_video_units():
    rows = []
    for index in range(4):
        source = f"s{index}"
        rows.append({"source_id": source, "role": "real", "generator": None, "M1_delta_s": {"video_p95": 0.1}})
        rows.append({"source_id": source, "role": "fake", "generator": "musev", "M1_delta_s": {"video_p95": 2.0}})
    result = bootstrap_source_cluster(rows, "M1_delta_s", seed=3, repeats=20)
    assert result["status"] == "OK"
    assert result["unit"] == "source_video"
    assert result["source_count"] == 4


def test_pair2_scoring_does_not_fit_or_accept_fake_features():
    models, identity = load_frozen_models()
    assert identity["fake_count"] == 0
    assert identity["refit"] is False
    scored = score_frontend_results([_row("s", "fake", "svd", [1.0] * 8)], models)
    assert scored["video_scores"][0]["role"] == "fake"
    assert all(name in scored["video_scores"][0] for name in MODEL_NAMES)


def test_formal_src_does_not_import_research_tools():
    for path in Path("src/sparse3d_forgery").rglob("*.py"):
        assert "research_tools" not in path.read_text(encoding="utf-8")
