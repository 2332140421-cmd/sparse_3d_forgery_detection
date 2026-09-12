from research_tools.v7.fixed_grid_frozen_probe.grid import (
    classify_fake_window,
    generate_grid_windows,
    interval_union_overlap,
    map_target_frames_to_intervals,
)


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
