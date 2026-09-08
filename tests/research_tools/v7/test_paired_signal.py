"""Small contract tests for the paired ActivityForensics pilot protocol."""

from __future__ import annotations

import numpy as np
import pytest

from research_tools.v7.paired_signal.protocol import (
    anchor_interval,
    assert_original_video_path,
    choose_control_block,
    indices_for_interval,
    primary_segment,
    robust_scale,
)


def test_preview_input_is_rejected(tmp_path):
    path = tmp_path / "PREVIEW_FAKE_H264.mp4"
    path.write_bytes(b"not media")
    with pytest.raises(ValueError, match="preview"):
        assert_original_video_path(path)


def test_primary_segment_uses_longest_then_earliest():
    assert primary_segment([{"start_s": 4, "end_s": 6}, {"start_s": 1, "end_s": 3}])["start_s"] == 1
    assert primary_segment([{"start_s": 4, "end_s": 7}, {"start_s": 1, "end_s": 4}])["start_s"] == 1


def test_anchor_interval_is_one_second_without_padding():
    left, right, center = anchor_interval((2.0, 8.0), 0.5)
    assert right - left == pytest.approx(1.0)
    assert left >= 2.0 and right <= 8.0
    assert center == pytest.approx(5.0)


def test_indices_for_interval_uses_only_existing_frames():
    timeline = {"timestamps_s": [0.0, 0.4, 0.8, 1.2, 1.6, 2.0]}
    indices, timestamps = indices_for_interval(timeline, (0.4, 1.6))
    assert indices == [1, 2, 3, 4]
    assert timestamps == [0.4, 0.8, 1.2, 1.6]


def test_short_interval_is_not_padded():
    with pytest.raises(ValueError, match="fewer than two"):
        indices_for_interval({"timestamps_s": [0.0, 2.0]}, (0.1, 0.9))


def test_control_block_is_outside_official_segment():
    block = choose_control_block(0.0, 10.0, {"start_s": 5.0, "end_s": 7.0}, [{"start_s": 5.0, "end_s": 7.0}])
    assert block in ((0.0, 5.0), (7.0, 10.0))


def test_robust_scale_uses_real_control_only_and_fallbacks():
    scale, details = robust_scale(np.asarray([[1, 2, 3, 4], [1, 2, 3, 4], [2, 3, 4, 5]], dtype=np.float64))
    assert np.all(np.isfinite(scale))
    assert details["source"] == "real_control_windows_only"
    assert details["fallback_count"] >= 1


def test_spearman_aligns_missing_values():
    from research_tools.v7.paired_signal.analyze import _spearman

    value = _spearman([1.0, None, 3.0, 4.0], [2.0, 99.0, 6.0, 8.0])
    assert value == pytest.approx(1.0)
