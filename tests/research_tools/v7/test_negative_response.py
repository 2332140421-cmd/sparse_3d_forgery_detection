from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from research_tools.v7.analysis.analyze_negative_response import (
    BOOTSTRAP_REPEATS,
    _cluster_bootstrap,
    _dimension_summary,
    _effect,
    _metadata,
    _train_statistics,
)


DATA_ROOT = Path("/root/autodl-tmp/data/sparse_3d_forgery_detection/derived")
OUT = DATA_ROOT / "v7_negative_response_mechanism_v1"


def _row(video: str, role: str, value: float, *, ordinal: str | None = None, generator: str | None = None):
    return {"video_id": video, "role": role, "pair2_ordinal": ordinal, "generator": generator, "metric": value}


def test_population_identity_model_sha_and_no_refit():
    summary = json.loads((OUT / "mechanism_summary.json").read_text(encoding="utf-8"))
    assert summary["population"]["real_train_sources"] == 24
    assert summary["population"]["real_val_sources"] == 8
    assert summary["population"]["fake_videos"] == 10
    assert summary["frozen_model"]["sha256"] == "4e02e45bd56be8a7081c2c2af5046fbe7697434b110d016313b4d502b3d32847"
    assert summary["frozen_model"]["fake_count"] == 0
    assert summary["frozen_model"]["refit"] is False


def test_real_val_role_is_normalized_without_changing_source_role():
    row = _metadata({"role": "real_val", "source_id": "Vript:val"})
    assert row["role"] == "real"
    assert row["source_role"] == "real_val"
    assert row["generator"] is None


def test_train_statistics_are_reference_only_and_deterministic():
    train = {
        "r1": {"observations": {"M1_delta_s": [np.array([1.0, 2.0, 3.0, 4.0])], "M2_delta2_s": [np.ones(4)], "M3_delta_s_delta2_s": [np.ones(8)]}},
        "r2": {"observations": {"M1_delta_s": [np.array([3.0, 4.0, 5.0, 6.0])], "M2_delta2_s": [np.ones(4) * 2], "M3_delta_s_delta2_s": [np.ones(8) * 2]}},
    }
    stats = _train_statistics(train)
    assert np.allclose(stats["M1_delta_s"]["mean"], [2.0, 3.0, 4.0, 5.0])
    assert np.allclose(np.mean(stats["M1_delta_s"]["cloud"], axis=0), 0.0)
    assert int(stats["M1_delta_s"]["count"][0]) == 2


def test_effect_size_uses_video_rows_not_component_rows():
    rows = [_row("r1", "real", 0.1), _row("r2", "real", 0.2), _row("f1", "fake", 0.9, ordinal="00015", generator="svd"), _row("f2", "fake", 0.8, ordinal="00018", generator="cogvideo")]
    effect = _effect(rows, "metric")
    assert effect["real"]["N"] == 2
    assert effect["fake"]["N"] == 2
    assert effect["AUROC_fake_higher"] == 1.0
    assert effect["Cliffs_delta_fake_higher"] == 1.0


def test_fake_bootstrap_clusters_pair2_ordinals_and_keeps_variants():
    rows = [_row(f"r{i}", "real", 0.1) for i in range(8)]
    rows.extend(_row(f"f-{ordinal}-{generator}", "fake", 1.0, ordinal=f"{ordinal:05d}", generator=generator) for ordinal in range(5) for generator in ("svd", "cogvideo"))
    result = _cluster_bootstrap(rows, "metric")
    assert result["status"] == "OK"
    assert result["fake_clusters"] == 5
    assert result["fake_cluster_unit"].startswith("Pair2 ordinal")
    assert result["repeats"] == BOOTSTRAP_REPEATS


def test_relative_roughness_and_missing_fake_are_preserved():
    rows = json.loads((OUT / "per_video_mechanism.json").read_text(encoding="utf-8"))
    missing = [row for row in rows if row["video_id"].endswith("00042:2na5Sqit03s::fake::cogvideo")]
    assert len(missing) == 1
    assert missing[0]["M1_score"] is None
    assert missing[0]["details"]["M1_delta_s"]["N"] == 0
    assert missing[0]["relative_roughness"] is None
    assert all(row["relative_roughness"] is not None for row in rows if row["role"] == "real")


def test_per_dimension_summary_has_exact_four_frozen_coordinates():
    summary = json.loads((OUT / "per_dimension_summary.json").read_text(encoding="utf-8"))
    for model in ("M1_delta_s", "M2_delta2_s"):
        assert [row["dimension"] for row in summary[model]["dimensions"]] == [0, 1, 2, 3]
        assert [row["semantic"] for row in summary[model]["dimensions"]] == ["mean", "std", "p25", "p75"]


def test_per_dimension_csv_mirrors_json_contract():
    rows = list(csv.DictReader((OUT / "per_dimension_summary.csv").open(encoding="utf-8", newline="")))
    assert len(rows) == 2 * 4 * 4
    assert {row["model"] for row in rows} == {"M1_delta_s", "M2_delta2_s"}
    assert {row["group"] for row in rows} == {"real", "fake", "svd", "cogvideo"}
    assert {row["semantic"] for row in rows} == {"mean", "std", "p25", "p75"}


def test_per_video_csv_has_video_level_contract_fields():
    with (OUT / "per_video_mechanism.csv").open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        fields = set(reader.fieldnames or ())
    assert len(rows) == 18
    assert {"video_id", "role", "generator", "pair2_ordinal", "component_success", "M1_center_distance_median", "M1_knn1_distance", "M2_knn5_distance", "M3_knn5_distance"} <= fields


def test_relative_roughness_is_recomputed_from_video_medians():
    rows = json.loads((OUT / "per_video_mechanism.json").read_text(encoding="utf-8"))
    for row in rows:
        m1 = row["details"]["M1_delta_s"]["norm"]["median"]
        m2 = row["details"]["M2_delta2_s"]["norm"]["median"]
        if m1 is None or m2 is None:
            assert row["relative_roughness"] is None
        else:
            assert np.isclose(row["relative_roughness"], m2 / (m1 + 1e-12))


def test_no_posthoc_score_inversion_and_compression_status():
    summary = json.loads((OUT / "mechanism_summary.json").read_text(encoding="utf-8"))
    response = json.loads((DATA_ROOT / "v7_unpaired_fake_response_v1/metrics/unpaired_response_metrics.json").read_text(encoding="utf-8"))
    assert all(response["models"][name]["AUROC"] < 0.5 for name in response["models"])
    assert summary["compression"]["status"] == "IDENTIFIABLE_WITH_CURRENT_ARTIFACTS"
    assert summary["mechanism_interpretation"]["MECHANISM_UNRESOLVED"] is True


def test_formal_src_does_not_import_analysis_tools():
    for path in Path("src/sparse3d_forgery").rglob("*.py"):
        assert "research_tools.v7.analysis" not in path.read_text(encoding="utf-8")
