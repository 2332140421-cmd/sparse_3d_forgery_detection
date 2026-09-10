"""Small contract tests for the frozen joint diagnostic helpers."""

from research_tools.v7.local_structural_temporal_diagnostic.diagnose import (
    aggregate_window_logit,
    build_review_html,
    nearest_frame_index,
)


def test_exact_mean_aggregation_matches_frozen_equation():
    assert aggregate_window_logit([[1.0, 3.0], [5.0]]) == 3.5


def test_nearest_frame_uses_relative_pts_without_interpolation():
    frames = [
        {"array_index": 4, "relative_time_s": 0.0},
        {"array_index": 5, "relative_time_s": 0.11},
        {"array_index": 6, "relative_time_s": 0.21},
    ]
    assert nearest_frame_index(frames, 0.18) == 6


def test_review_page_is_inline_and_has_annotation_and_layer_contracts():
    html = build_review_html({"windows": [], "groups": {}, "details": {}, "sample_cases": {}})
    assert "https://" not in html
    assert "<script id=\"payload\"" in html
    assert "导出 JSON" in html
    assert "观测覆盖" in html and "结构变化" in html and "模型响应" in html
    assert "table.style.display='none'" in html
