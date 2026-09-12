import json
from types import SimpleNamespace

import numpy as np
import pytest

from research_tools.v7.boundary_pooling_probe.review_and_forward import (
    _aggregate_forward_scores,
    _boundary_relation_stats,
    _classification_metrics,
    _fixed_review_html,
    _group_owner_ids,
    _html_page,
    materialize_windows,
    _observation_frame_match,
    _source_frames_for_timestamps,
    _triplet_display,
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

    assert "old=s.value" in html
    assert "list.includes(old)" in html


def test_fixed_review_page_keeps_window_selection_and_observation_boundaries() -> None:
    html = _fixed_review_html()

    assert "function fillSources()" in html
    assert "loadWindow(role,$(role+'Window').value)" in html
    assert "OUTSIDE_OBSERVATION" in html
    assert "NO_VALID_TRIPLET" in html
    assert "common_member_slots" in html
    assert "track ID" in html
    assert "requestVideoFrameCallback" in html


def test_observation_match_rejects_outside_range_and_accepts_pts_tolerance() -> None:
    frames = [
        {"timestamp_s": 10.0, "source_frame_index": 100},
        {"timestamp_s": 10.1, "source_frame_index": 101},
        {"timestamp_s": 10.2, "source_frame_index": 102},
    ]

    assert _observation_frame_match(frames, 9.9)["status"] == "OUTSIDE_OBSERVATION"
    matched = _observation_frame_match(frames, 10.04)
    assert matched["status"] == "MATCHED"
    assert matched["record"]["source_frame_index"] == 100
    assert _observation_frame_match(frames, 10.051, tolerance_s=0.01)["status"] == "NO_OBSERVATION"


def test_observation_match_does_not_interpolate_across_saved_gap() -> None:
    frames = [
        {"timestamp_s": 1.0, "source_frame_index": 1},
        {"timestamp_s": 1.1, "source_frame_index": 2},
        {"timestamp_s": 2.0, "source_frame_index": 3},
    ]

    result = _observation_frame_match(frames, 1.5, tolerance_s=0.2)
    assert result["status"] == "NO_OBSERVATION"
    assert result["reason"] == "saved_observation_gap"


def test_triplet_mapping_uses_track_ids_not_array_slots() -> None:
    triplet = {
        "triplet_id": 2,
        "component_index": 7,
        "common_track_ids": [101, 303],
        "pair_ids": [[101, 303]],
        "states": [[1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4]],
        "timestamps_s": [1.0, 1.1, 1.2],
        "target_slots": [0, 1, 2],
    }
    mapping = {101: 4, 303: 1}
    display = _triplet_display(triplet, track_to_slot=mapping, owner_group_ids=[9])

    assert display["common_member_slots"] == [4, 1]
    assert display["pair_member_slots"] == [[4, 1]]
    assert display["owner_group_ids"] == [9]


def test_triplet_is_owned_only_by_group_containing_common_members() -> None:
    triplet = {"component_index": 3, "members_considered": [0, 1], "common_track_ids": [10, 20]}
    groups = [
        {"local_group_id": 1, "parent_component_id": 3, "member_slots": [0, 1]},
        {"local_group_id": 2, "parent_component_id": 3, "member_slots": [0]},
        {"local_group_id": 8, "parent_component_id": 4, "member_slots": [2, 3]},
    ]

    assert _group_owner_ids(triplet, groups, {10: 0, 20: 1}) == [1]
