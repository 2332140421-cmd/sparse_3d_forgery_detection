import json
from types import SimpleNamespace

import numpy as np
import pytest

from research_tools.v7.boundary_pooling_probe.review_and_forward import (
    _aggregate_forward_scores,
    _boundary_relation_stats,
    _classification_metrics,
    _html_page,
    materialize_windows,
    _source_frames_for_timestamps,
)


def test_forward_review_classification_metrics_use_fixed_logit_threshold() -> None:
    metrics = _classification_metrics([0, 0, 1, 1], [-1.0, 0.2, -0.1, 0.8])

    assert metrics["threshold"] == "logit >= 0"
    assert metrics["tn"] == 1
    assert metrics["fp"] == 1
    assert metrics["fn"] == 1
    assert metrics["tp"] == 1
    assert metrics["n_real"] == 2
    assert metrics["n_fake"] == 2


def test_boundary_review_counts_retained_cut_and_unassigned_pairs() -> None:
    parent = {"local_group_id": 4, "member_slots": [0, 1, 2, 3]}
    children = [
        {"local_group_id": 10, "parent_local_group_id": 4, "member_slots": [0, 1]},
        {"local_group_id": 11, "parent_local_group_id": 4, "member_slots": [2]},
    ]

    result = _boundary_relation_stats(parent, children)

    assert result["h_pair_total"] == 6
    assert result["pairs_retained_within_one_b_child"] == 1
    assert result["pairs_cut_by_boundary"] == 5
    assert result["pairs_with_unassigned_endpoint"] == 3
    assert result["unassigned_pair_examples"] == [[0, 3], [1, 3], [2, 3]]


def test_triplet_pts_map_to_source_frames_without_using_local_target_slots() -> None:
    sequence = SimpleNamespace(
        timestamps_s=np.asarray([10.0, 10.1, 10.2, 10.3]),
        frame_indices=np.asarray([740, 741, 742, 743]),
    )

    assert _source_frames_for_timestamps(sequence, [10.1, 10.3]) == [741, 743]

    with pytest.raises(ValueError, match="not present"):
        _source_frames_for_timestamps(sequence, [10.15])


def test_forward_aggregate_keeps_pooled_and_source_macro_distinct() -> None:
    rows = [
        {"condition": "H_MEAN_A", "split": "heldout", "held_out_source": "s1", "seed": 1, "source_id": "s1", "window_id": "s1-r", "label": 0, "logit": -1.0},
        {"condition": "H_MEAN_A", "split": "heldout", "held_out_source": "s1", "seed": 1, "source_id": "s1", "window_id": "s1-f", "label": 1, "logit": 1.0},
        {"condition": "H_MEAN_A", "split": "heldout", "held_out_source": "s2", "seed": 1, "source_id": "s2", "window_id": "s2-r", "label": 0, "logit": 1.0},
        {"condition": "H_MEAN_A", "split": "heldout", "held_out_source": "s2", "seed": 1, "source_id": "s2", "window_id": "s2-f", "label": 1, "logit": -1.0},
    ]

    result = _aggregate_forward_scores(rows)
    pooled = next(x for x in result if x["condition"] == "H_MEAN_A" and x["split"] == "heldout" and x["aggregation"] == "pooled")
    macro = next(x for x in result if x["condition"] == "H_MEAN_A" and x["split"] == "heldout" and x["aggregation"] == "source_macro")
    assert pooled["score_rows"] == 4
    assert pooled["roc_auc"] == 0.5
    assert macro["source_count"] == 2
    assert macro["roc_auc"] == 0.5


def test_materialize_windows_is_explicit_and_uses_symlink(tmp_path) -> None:
    root = tmp_path / "artifact"
    (root / "manifests").mkdir(parents=True)
    (root / "review/details").mkdir(parents=True)
    source = tmp_path / "source.mp4"
    source.write_bytes(b"not decoded by this unit test")
    window_id = "w1::real"
    (root / "manifests/input_manifest.json").write_text(json.dumps({"rows": [{"window_id": window_id}]}))
    (root / "review/details/w1__real.json").write_text(json.dumps({"identity": {"role": "real"}, "video": {"path": str(source), "status": "AVAILABLE", "relative_media": None}}))
    (root / "review/index_data.json").write_text(json.dumps({"windows": [{"window_id": window_id, "detail_path": "details/w1__real.json"}]}))

    result = materialize_windows(root, [window_id])

    assert result[0]["status"] == "AVAILABLE"
    link = root / "review/media/w1__real__real.mp4"
    assert link.is_symlink()
    detail = json.loads((root / "review/details/w1__real.json").read_text())
    assert detail["video"]["materialization_status"] == "AVAILABLE"


def test_review_page_preserves_source_selection_when_refilling_windows() -> None:
    html = _html_page()

    assert "previous=s.value" in html
    assert "sources.includes(previous)" in html
