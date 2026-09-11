from __future__ import annotations

from research_tools.v7.boundary_pooling_probe.summarize_results import (
    _classification_row,
    _seed_stability,
    _source_seed_metrics,
)


def _rows() -> list[dict[str, object]]:
    return [
        {
            "kind": "MANIP",
            "role": "real",
            "source_id": "s0",
            "TEST_score": -1.0,
            "TEST_seed_20260909": -1.0,
            "TEST_seed_20260910": -0.5,
            "TEST_seed_20260911": -0.8,
        },
        {
            "kind": "MANIP",
            "role": "fake",
            "source_id": "s0",
            "TEST_score": 1.0,
            "TEST_seed_20260909": 1.0,
            "TEST_seed_20260910": 0.5,
            "TEST_seed_20260911": 0.8,
        },
        {
            "kind": "CTRL",
            "role": "fake",
            "source_id": "s0",
            "TEST_score": -100.0,
        },
    ]


def test_classification_summary_uses_manip_only_and_zero_logit_threshold() -> None:
    result = _classification_row(_rows(), "TEST")
    assert result["n_total"] == 2
    assert result["n_fake"] == 1
    assert result["n_real"] == 1
    assert result["ctrl_included"] is False
    assert result["pooled_roc_auc"] == 1.0
    assert result["tp"] == 1 and result["tn"] == 1


def test_source_seed_summary_keeps_three_saved_seed_scores() -> None:
    detailed = _source_seed_metrics(_rows(), "TEST")
    assert len(detailed) == 3
    assert all(item["auroc"] == 1.0 for item in detailed)
    summary = _seed_stability(_rows(), "TEST")
    assert summary["source_count"] == 1
    assert summary["seed_count_per_source"] == 3
    assert summary["missing_seed_rows"] == 0
