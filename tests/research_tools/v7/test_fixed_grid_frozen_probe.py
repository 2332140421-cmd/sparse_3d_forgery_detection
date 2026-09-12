from research_tools.v7.fixed_grid_frozen_probe.grid import (
    classify_fake_window,
    generate_grid_windows,
    interval_union_overlap,
    map_target_frames_to_intervals,
)
from research_tools.v7.fixed_grid_frozen_probe import runner

import json


def test_grid_adds_tail_without_duplicate_and_keeps_pts():
    rows = generate_grid_windows([0.0, 0.4, 0.8, 1.2, 1.6, 2.0, 2.4], window_length_s=1.0, stride_s=0.5)
    assert [row["nominal_start_s"] for row in rows] == [0.0, 0.5, 1.0, 1.4]
    assert rows[-1]["tail_window"] is True
    assert rows[0]["frame_pts_s"] == [0.0, 0.4, 0.8]


def test_short_video_is_not_padded():
    rows = generate_grid_windows([2.0, 2.2, 2.4], window_length_s=1.0)
    assert rows[0]["status"] == "SHORT_VIDEO"
    assert abs(rows[0]["relative_duration_s"] - 0.4) < 1e-9
    assert rows[0]["timestamp_origin_s"] == 2.0


def test_annotation_union_and_categories():
    intervals = [{"start_s": 0.0, "end_s": 0.4}, {"start_s": 0.3, "end_s": 1.0}]
    assert interval_union_overlap(0.0, 1.0, intervals) == 1.0
    assert classify_fake_window(0.0, 1.0, intervals)["annotation_category"] == "FAKE_MANIPULATION"
    assert classify_fake_window(0.8, 1.8, intervals)["annotation_category"] == "BOUNDARY_MIXED"
    assert classify_fake_window(2.0, 3.0, intervals)["annotation_category"] == "OUTSIDE_ANNOTATED_MANIPULATION"


def test_target_frame_mapping_is_timestamp_based():
    result = map_target_frames_to_intervals([0.1, 0.5, 1.1], [{"start_s": 0.4, "end_s": 1.0}])
    assert result["target_frames_in_annotation_count"] == 1


def test_metadata_complete_video_generates_grid_in_new_plan(tmp_path, monkeypatch):
    source_root = tmp_path / "source"
    paired_root = tmp_path / "paired"
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"placeholder")
    (source_root / "manifests").mkdir(parents=True)
    (paired_root / "manifests").mkdir(parents=True)
    (source_root / "manifests" / "input_manifest.json").write_text(json.dumps({"rows": [{"source_id": "S1", "role": "real", "pair_id": "P1", "video_path": str(video)}]}))
    (paired_root / "manifests" / "selected_pairs.json").write_text(json.dumps([{"source_id": "S1", "all_manipulation_segments": []}]))
    monkeypatch.setattr(runner, "SOURCE_ROOT", source_root)
    monkeypatch.setattr(runner, "PAIRED_ROOT", paired_root)
    monkeypatch.setattr(runner, "_probe_video", lambda path: {"path": str(path), "bytes": 1, "sha256": "x", "frame_count": 7, "frame_indices": list(range(7)), "timestamps_s": [0.0, 0.4, 0.8, 1.2, 1.6, 2.0, 2.4], "width": 10, "height": 10, "codec": "test", "origin_pts_s": 0.0, "duration_relative_s": 2.4})
    plan = runner.prepare_plan(tmp_path / "out")
    assert plan["grid_window_count"] > 0
    assert plan["videos"][0]["status"] == "METADATA_COMPLETE"


def test_plan_identity_reports_frame_or_window_changes():
    row = {"window_id": "S::real::grid0000", "source_video_id": "S::real", "source_id": "S", "role": "real", "grid_index": 0, "frame_indices": [0, 1], "timestamps_s": [0.0, 0.1], "interval_start_s": 0.0, "interval_end_s": 1.0}
    changed = {**row, "frame_indices": [0, 2]}
    assert runner.compare_plan_identity([row], [row]) == []
    assert runner.compare_plan_identity([row], [changed]) == ["mismatch:S::real::grid0000"]


def test_cached_record_without_artifact_is_not_reusable(tmp_path):
    row = {"window_id": "S::real::grid0000", "source_video_id": "S::real", "frame_indices": [0], "timestamps_s": [0.0]}
    ok, reason = runner._result_reuse_check(tmp_path, row, {"window_id": row["window_id"], "source_video_id": row["source_video_id"], "sequence_prefix": str(tmp_path / "missing")})
    assert not ok
    assert reason == "SEQUENCE_ARTIFACT_MISSING"


def test_budget_resume_does_not_reset_consumed_time(tmp_path):
    path = tmp_path / "manifests"
    path.mkdir()
    (path / runner.BUDGET_STATE_NAME).write_text(json.dumps({"budget_s": 7200.0, "consumed_s": 321.5, "accounting_status": "RUNNING"}))
    state = runner._load_budget_state(tmp_path, 7200.0, {})
    assert state["consumed_s"] == 321.5


def test_budget_clock_adds_process_elapsed_once():
    state = {"consumed_s": 100.0}
    clock = runner._start_budget_clock(state, monotonic_start=10.0, wall_start=20.0)
    cumulative, process_elapsed = runner._budget_snapshot(clock, monotonic_now=35.0)
    assert process_elapsed == 25.0
    assert cumulative == 125.0


def test_score_window_sets_are_unique_and_label_filtered():
    def row(window_id, source_id, role, category, h, b_a, b_c):
        return {
            "window_id": window_id,
            "source_id": source_id,
            "role": role,
            "annotation_category": category,
            "H_MEAN_A": h,
            "H_MEAN_A_status": "SCORED" if h is not None else "NO_VALID_TRIPLET",
            "B_MEAN_A": b_a,
            "B_MEAN_A_status": "SCORED" if b_a is not None else "NO_VALID_TRIPLET",
            "B_MEAN_C": b_c,
            "B_MEAN_C_status": "SCORED" if b_c is not None else "NO_VALID_TRIPLET",
        }

    rows = [
        row("real", "S", "real", "REAL_NEGATIVE", -1.0, -1.0, -1.0),
        row("fake", "S", "fake", "FAKE_MANIPULATION", 1.0, 1.0, 1.0),
        row("boundary", "S", "fake", "BOUNDARY_MIXED", 2.0, 2.0, 2.0),
        row("partial", "P", "real", "REAL_NEGATIVE", -2.0, -2.0, -2.0),
        row("h_only", "S", "real", "REAL_NEGATIVE", -3.0, None, -3.0),
        row("fake", "S", "fake", "FAKE_MANIPULATION", 1.0, 1.0, 1.0),
    ]
    sets = runner._score_window_sets(rows, {"S"})
    assert len(sets["unique_rows"]) == 5
    assert sets["duplicate_window_ids"] == ["fake"]
    assert {condition: len(values) for condition, values in sets["condition_ids"].items()} == {"H_MEAN_A": 5, "B_MEAN_A": 4, "B_MEAN_C": 5}
    assert len(sets["raw_three_condition_ids"]) == 4
    assert len(sets["complete_source_three_condition_ids"]) == 3
    assert sets["main_label_filtered_ids"] == {"real", "fake"}


def test_resume_keeps_persisted_source_prefix(tmp_path):
    (tmp_path / "manifests").mkdir()
    (tmp_path / "manifests" / "execution_plan.json").write_text(json.dumps({"source_order": ["A", "B"], "selected_source_prefix": ["A"]}))
    rows = [
        {"window_id": "A::0", "source_id": "A", "role": "real", "grid_index": 0},
        {"window_id": "B::0", "source_id": "B", "role": "real", "grid_index": 0},
    ]
    selected, selected_rows, error = runner._select_source_prefix(tmp_path, rows, {"A::0": {}}, {"consumed_s": 0.0, "budget_s": 7200.0})
    assert error is None
    assert selected == ["A"]
    assert [row["source_id"] for row in selected_rows] == ["A"]


def test_missing_score_placeholder_is_not_scored():
    row = {"H_MEAN_A": "", "H_MEAN_A_status": "MISSING_FEATURE"}
    assert runner._condition_scored(row, "H_MEAN_A") is False
    assert runner._condition_scored({"H_MEAN_A": "0.25", "H_MEAN_A_status": "SCORED"}, "H_MEAN_A") is True


def test_model_used_frames_are_triplet_supported_not_all_frontend_frames(tmp_path):
    feature_path = tmp_path / "features" / "S__real__grid0000.json"
    feature_path.parent.mkdir(parents=True)
    feature_path.write_text(json.dumps({"identity": {"frame_indices": [10, 11, 12], "timestamps_s": [1.0, 1.1, 1.2]}, "h_support": {"triplets": [{"timestamps_s": [1.1, 1.2]}]}, "b_support": {"triplets": [{"timestamps_s": [1.2]}]}}))
    rows = [{"window_id": "S::real::grid0000", "source_id": "S"}]
    values = runner._model_used_frame_rows(tmp_path, rows)
    assert {(row["condition"], row["frame_index"]) for row in values if row["status"] == "MODEL_USED"} == {("H_MEAN_A", 11), ("H_MEAN_A", 12), ("B_MEAN_A", 12), ("B_MEAN_C", 12)}


def test_missing_seed_is_not_silently_complete():
    models = {("H_MEAN_A", "S", 1): object()}
    assert runner._missing_model_seeds(models, "H_MEAN_A", "S", [1, 2, 3]) == [2, 3]


def test_partial_source_is_excluded_from_main_evaluation(tmp_path):
    (tmp_path / "scores").mkdir(parents=True)
    (tmp_path / "coverage").mkdir(parents=True)
    (tmp_path / "scores" / "window_scores.csv").write_text("window_id,source_id,role,annotation_category,H_MEAN_A,H_MEAN_A_status,B_MEAN_A,B_MEAN_A_status,B_MEAN_C,B_MEAN_C_status\nS::real::0,S,real,REAL_NEGATIVE,-1,SCORED,-1,SCORED,-1,SCORED\nS::fake::0,S,fake,FAKE_MANIPULATION,1,SCORED,1,SCORED,1,SCORED\n")
    (tmp_path / "coverage" / "source_coverage.csv").write_text("source_id,complete_source,frontend_windows,planned_windows\nS,false,2,3\n")
    (tmp_path / "manifests").mkdir()
    (tmp_path / "manifests" / "grid_windows.json").write_text("[]")
    summary = runner.evaluate(tmp_path)
    assert summary["conditions"]["H_MEAN_A"]["source_count"] == 0
    assert summary["conditions"]["H_MEAN_A"]["source_mean_auroc"] is None
