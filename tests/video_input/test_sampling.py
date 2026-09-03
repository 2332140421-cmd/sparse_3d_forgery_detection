from pathlib import Path

import numpy as np
import pytest

from sparse3d_forgery.video_input import (
    DecodedFrame,
    DecodedVideoSample,
    VideoSource,
    explicit_frame_indices,
    fixed_stride_frame_indices,
)


def frame(index: int, timestamp: float, shape: tuple[int, int] = (4, 6)) -> DecodedFrame:
    return DecodedFrame(index, timestamp, np.zeros((*shape, 3), dtype=np.uint8))


def test_valid_source_and_single_frame_sample_preserve_caller_identity() -> None:
    source = VideoSource("chosen-sample", "chosen-video", Path("different-name.mp4"))
    decoded = DecodedVideoSample(source.sample_id, source.source_video_id, (frame(3, 0.25),))

    assert decoded.sample_id == "chosen-sample"
    assert decoded.source_video_id == "chosen-video"
    np.testing.assert_array_equal(decoded.frame_indices, np.array([3], dtype=np.int64))


def test_multi_frame_sample_derives_arrays_from_frames() -> None:
    decoded = DecodedVideoSample("sample", "video", (frame(1, 0.1), frame(4, 0.4, (8, 9))))

    np.testing.assert_array_equal(decoded.frame_indices, np.array([1, 4], dtype=np.int64))
    np.testing.assert_array_equal(decoded.timestamps_s, np.array([0.1, 0.4], dtype=np.float64))
    np.testing.assert_array_equal(decoded.frame_sizes_hw, np.array([[4, 6], [8, 9]], dtype=np.int64))


def test_nonuniform_vfr_timestamps_are_valid() -> None:
    decoded = DecodedVideoSample("sample", "video", (frame(0, 0.0), frame(1, 0.04), frame(2, 0.13)))
    np.testing.assert_array_equal(decoded.timestamps_s, [0.0, 0.04, 0.13])


def test_frame_rejects_invalid_rgb() -> None:
    invalid = (
        (np.zeros((2, 3, 3), dtype=np.float32), "rgb dtype"),
        (np.zeros((2, 3), dtype=np.uint8), "rgb shape"),
        (np.zeros((0, 3, 3), dtype=np.uint8), "rgb height"),
    )
    for rgb, message in invalid:
        with pytest.raises(ValueError, match=message):
            DecodedFrame(0, 0.0, rgb)


def test_sample_rejects_duplicate_or_decreasing_frame_indices() -> None:
    for left, right in ((1, 1), (2, 1)):
        with pytest.raises(ValueError, match="source_frame_index"):
            DecodedVideoSample("sample", "video", (frame(left, 0.0), frame(right, 0.1)))


def test_sample_rejects_nonfinite_duplicate_or_decreasing_timestamps() -> None:
    with pytest.raises(ValueError, match="timestamp_s must be a float"):
        DecodedFrame(1, 0, np.zeros((2, 3, 3), dtype=np.uint8))
    with pytest.raises(ValueError, match="timestamp_s must be finite"):
        frame(1, np.nan)
    for left, right in ((0.0, 0.0), (0.2, 0.1)):
        with pytest.raises(ValueError, match="frames.timestamp_s"):
            DecodedVideoSample("sample", "video", (frame(0, left), frame(1, right)))


def test_schema_does_not_modify_rgb() -> None:
    rgb = np.arange(18, dtype=np.uint8).reshape(2, 3, 3)
    before = rgb.copy()
    decoded_frame = DecodedFrame(0, 0.0, rgb)

    assert decoded_frame.rgb is rgb
    np.testing.assert_array_equal(rgb, before)


def test_explicit_indices_are_int64_and_unchanged() -> None:
    requested = [0, 3, 9]
    before = requested.copy()
    actual = explicit_frame_indices(requested)

    np.testing.assert_array_equal(actual, np.array(requested, dtype=np.int64))
    assert actual.dtype == np.int64
    assert requested == before


def test_explicit_indices_reject_invalid_requests() -> None:
    for indices in ([], [-1, 2], [1, 1], [2, 1]):
        with pytest.raises(ValueError, match="indices"):
            explicit_frame_indices(indices)


def test_fixed_stride_supports_stop_and_max_truncation_deterministically() -> None:
    expected = np.array([2, 5, 8], dtype=np.int64)
    first = fixed_stride_frame_indices(2, stop_index_exclusive=20, stride=3, max_frames=3)
    second = fixed_stride_frame_indices(2, stop_index_exclusive=20, stride=3, max_frames=3)

    np.testing.assert_array_equal(first, expected)
    np.testing.assert_array_equal(second, expected)


def test_fixed_stride_supports_max_without_stop() -> None:
    actual = fixed_stride_frame_indices(4, stride=2, max_frames=4)
    np.testing.assert_array_equal(actual, np.array([4, 6, 8, 10], dtype=np.int64))


def test_fixed_stride_rejects_invalid_parameters() -> None:
    invalid = (
        {"start_index": -1, "max_frames": 1},
        {"start_index": 0, "stride": 0, "max_frames": 1},
        {"start_index": 2, "stop_index_exclusive": 2},
        {"start_index": 0, "max_frames": 0},
        {"start_index": 0},
    )
    for kwargs in invalid:
        with pytest.raises(ValueError):
            fixed_stride_frame_indices(**kwargs)
