"""Small contract tests for the frozen joint diagnostic helpers."""

from research_tools.v7.local_structural_temporal_diagnostic.diagnose import (
    _build_index,
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
    assert "当前可定位跟踪点" in html and "geometry-valid 点" in html and "模型响应" in html
    assert "table.style.display='none'" in html
    assert "realComponent" in html and "fakeComponent" in html
    assert "相对当前 triplet 首时刻的归一化距离变化" in html
    assert "source_frame_index" in html and "timestamp_s" in html
    assert "parseCsv" in html and "numberOrNull" in html
    assert "NOT_MATERIALIZED" in html and "SOURCE_MISSING" in html
    assert "layerTracks" in html and "layerGeometry" in html and "layerTriplet" in html and "layerPair" in html
    assert "stepToAdjacentSourceFrame" in html
    assert "component_total_members" in html and "model_sample_timestamps_s" in html
    assert "canvasPoint" in html and "x>=regionSelection.x" in html
    assert "source_frame_index" in html and "frame_size_hw" in html
    assert "common_member_indices" in html
    assert "clamp=(x,lo,hi)" in html
    assert "regionSelection.role!==$('annRole').value" in html
    assert "loadGroupDetails" in html and "DATA.detail_paths" in html
    assert "不要直接打开 file:// 文件" in html
    fallback = build_review_html({"windows": [], "groups": {"g": {"source_id": "S01"}}, "details": {}, "sample_cases": {}})
    assert '<option value="S01">S01</option>' in fallback
    assert 'id="initError"' in fallback
    # Python's outer template must preserve JavaScript escape sequences.  A
    # literal newline inside the single-quoted CSV parser strings makes the
    # browser reject the whole page before the videos can be initialised.
    assert "ch==='\\n'" in fallback
    assert "lines.join('\\n')" in fallback
    assert "ch==='\n'" not in fallback


def test_index_distinguishes_unmaterialized_from_missing_source(tmp_path):
    rows = [
        {"window_id": "w1::real", "source_id": "s", "pair_id": "p", "role": "real", "kind": "MANIP", "anchor_fraction": 0.25},
        {"window_id": "w2::real", "source_id": "s", "pair_id": "p", "role": "real", "kind": "MANIP", "anchor_fraction": 0.25},
    ]
    manifest = {
        "w1::real": {"label": "MANIP_25", "anchor_fraction": 0.25, "interval_start_s": 0.0, "interval_end_s": 1.0, "video_path": str(tmp_path / "present.mp4")},
        "w2::real": {"label": "MANIP_25", "anchor_fraction": 0.25, "interval_start_s": 0.0, "interval_end_s": 1.0, "video_path": str(tmp_path / "absent.mp4")},
    }
    (tmp_path / "present.mp4").write_bytes(b"not decoded in this contract test")
    coverage = {"w1::real": {"support_status": "NO_VALID_TRIPLET", "invalid_reasons": []}, "w2::real": {"support_status": "NO_VALID_TRIPLET", "invalid_reasons": []}}
    result = _build_index(rows, coverage, {}, manifest, {})
    assert result[0]["media"]["status"] == "NOT_MATERIALIZED"
    assert result[1]["media"]["status"] == "SOURCE_MISSING"
