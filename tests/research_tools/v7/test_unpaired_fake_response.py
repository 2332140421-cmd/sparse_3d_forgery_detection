from __future__ import annotations

from pathlib import Path

import numpy as np

from research_tools.v7.detection.evaluate_unpaired_fake_response import (
    _select_real_val_frontend,
    _tail_metrics,
    aggregate_window_scores,
    cluster_bootstrap,
    spearman,
)
from research_tools.v7.detection.protocol import (
    ANCHOR_FRACTIONS,
    MODEL_NAMES,
    auroc,
    load_frozen_models,
    build_pair2_window_manifest,
)
from research_tools.v7.normality.protocol import COMPONENT_CONFIG
from research_tools.v7.detection.run_unpaired_fake_response import (
    FROZEN_MODEL_SHA256,
    FAKE_MANIFEST,
    validate_fake_identity,
    verify_frozen_model,
)


def _score_row(video: str, role: str, score: float, *, generator: str | None = None, ordinal: str | None = None, window: str = "25"):
    return {
        "video_id": video,
        "source_id": video.split("::")[0],
        "role": role,
        "generator": generator,
        "pair2_ordinal": ordinal,
        "window_id": f"{video}::anchor-{window}",
        **{name: {"video_p95": score, "video_median": score / 2, "feature_count": 4} for name in MODEL_NAMES},
    }


def test_exact_frozen_fake_identity_is_ten_media_valid_rows():
    import json
    rows = json.loads(FAKE_MANIFEST.read_text(encoding="utf-8"))
    checked = validate_fake_identity(rows)
    assert len(checked) == 10
    assert {row["generator"] for row in checked} == {"svd", "cogvideo"}
    assert {str(row["pair_key"]) for row in checked} == {"00015", "00018", "00033", "00042", "00044"}


def test_frozen_model_sha_and_no_refit_identity():
    identity = verify_frozen_model()
    assert identity["sha256"] == FROZEN_MODEL_SHA256
    assert identity["fitting_population"] == "real_train_only"
    assert identity["fake_count"] == 0
    assert identity["refit"] is False
    models, loaded = load_frozen_models()
    assert tuple(models) == MODEL_NAMES
    assert loaded["fake_count"] == 0


def test_fixed_one_second_anchors_and_no_fake_selection():
    rows = [{
        "source_id": "source",
        "video_id": "source::fake::svd",
        "role": "fake",
        "generator": "svd",
        "pair_key": "00015",
        "pair_lineage": "shared Pair2 ordinal and official HDVG semantic record",
        "status": "MEDIA_VALID",
        "timestamps_s": (np.arange(40, dtype=float) * 0.1).tolist(),
    }]
    windows = build_pair2_window_manifest(rows)
    assert [row["anchor_fraction"] for row in windows] == list(ANCHOR_FRACTIONS)
    assert all(row["timescale_s"] == 1.0 for row in windows)
    assert all(row["fake_used"] is True for row in windows)
    assert all(row["status"] == "AVAILABLE" for row in windows)


def test_component_config_and_model_schema_are_frozen():
    assert COMPONENT_CONFIG.max_initial_distance == 1.0
    assert COMPONENT_CONFIG.max_relative_change == 0.05
    assert COMPONENT_CONFIG.minimum_overlap == 8
    assert COMPONENT_CONFIG.minimum_size == 3
    models, _ = load_frozen_models()
    assert models["M1_delta_s"].feature_count == 3200
    assert models["M2_delta2_s"].feature_count == 3061
    assert models["M3_delta_s_delta2_s"].feature_count == 3061
    assert models["M1_delta_s"].mean.size == 4
    assert models["M2_delta2_s"].mean.size == 4
    assert models["M3_delta_s_delta2_s"].mean.size == 8


def test_source_primary_aggregation_is_median_of_window_p95():
    rows = [_score_row("r", "real", 1.0, window="25"), _score_row("r", "real", 5.0, window="50"), _score_row("r", "real", 3.0, window="75")]
    result = aggregate_window_scores(rows)
    assert len(result) == 1
    assert result[0]["M1_delta_s"]["video_p95"] == 3.0
    assert result[0]["M1_delta_s"]["video_median"] == 1.5


def test_real_val_quality_excludes_train_windows():
    rows = [
        {"window": {"role": "real_train", "source_id": "train"}},
        *[
            {"window": {"role": "real_val", "source_id": f"val-{index}"}}
            for index in range(8)
            for _ in range(3)
        ],
    ]
    selected = _select_real_val_frontend(rows)
    assert len(selected) == 24
    assert {row["window"]["role"] for row in selected} == {"real_val"}


def test_real_only_tail_and_unscored_video_are_not_dropped():
    rows = [
        _score_row("real-1", "real", 1.0),
        _score_row("real-2", "real", 2.0),
        _score_row("fake-scored", "fake", 100.0, generator="svd", ordinal="00015"),
    ]
    missing = {
        "video_id": "fake-unscored",
        "source_id": "fake-unscored",
        "role": "fake",
        "generator": "cogvideo",
        "pair2_ordinal": "00015",
        "window_id": "fake-unscored::anchor-25",
        **{name: {"video_p95": None, "video_median": None, "feature_count": 0} for name in MODEL_NAMES},
    }
    rows.append(missing)
    tails = _tail_metrics(rows)
    assert tails["M1_delta_s"]["tau95_real_only"] == np.percentile([1.0, 2.0], 95)
    assert tails["M1_delta_s"]["overall_fake_gt_tau95_fraction"] == 1.0
    aggregated = aggregate_window_scores([missing])
    assert len(aggregated) == 1
    assert aggregated[0]["M1_delta_s"]["video_p95"] is None


def test_auroc_direction_and_cliffs_delta():
    value = auroc([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9])
    assert value == 1.0
    assert 2 * value - 1 == 1.0


def test_cluster_bootstrap_keeps_fake_generator_variants_together():
    rows = []
    for index in range(8):
        rows.append(_score_row(f"real-{index}", "real", 0.1))
    for ordinal in range(5):
        for generator in ("svd", "cogvideo"):
            rows.append(_score_row(f"fake-{ordinal}-{generator}", "fake", 1.0, generator=generator, ordinal=f"{ordinal:05d}"))
    result = cluster_bootstrap(rows, "M1_delta_s", seed=3, repeats=50)
    assert result["status"] == "OK"
    assert result["real_clusters"] == 8
    assert result["fake_clusters"] == 5
    assert result["fake_cluster_unit"].startswith("Pair2 source ordinal")


def test_spearman_and_quality_direction_are_deterministic():
    assert spearman([1, 2, 3], [3, 2, 1]) == -1.0
    assert spearman([1, 1], [2, 3]) is None


def test_formal_src_does_not_import_research_tools():
    for path in Path("src/sparse3d_forgery").rglob("*.py"):
        assert "research_tools" not in path.read_text(encoding="utf-8")
