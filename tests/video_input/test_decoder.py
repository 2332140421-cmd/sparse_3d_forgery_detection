from fractions import Fraction
from pathlib import Path

import av
import numpy as np
import pytest

from sparse3d_forgery.video_input import VideoDecodingError, VideoSource, decode_video


def make_video(path: Path, frame_count: int = 6) -> None:
    with av.open(str(path), mode="w") as container:
        stream = container.add_stream("mpeg4", rate=10)
        stream.width = 8
        stream.height = 6
        stream.pix_fmt = "yuv420p"
        stream.time_base = Fraction(1, 10)
        for index in range(frame_count):
            rgb = np.full((6, 8, 3), index * 20, dtype=np.uint8)
            frame = av.VideoFrame.from_ndarray(rgb, format="rgb24")
            frame.pts = index
            frame.time_base = Fraction(1, 10)
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


def test_decode_video_preserves_requested_frames_pts_and_input(tmp_path: Path) -> None:
    path = tmp_path / "tiny.mp4"
    make_video(path)
    requested = [0, 2, 4]
    before = requested.copy()

    decoded = decode_video(VideoSource("sample", "source", path), requested)

    assert requested == before
    np.testing.assert_array_equal(decoded.frame_indices, [0, 2, 4])
    np.testing.assert_allclose(decoded.timestamps_s, [0.0, 0.2, 0.4])
    assert all(frame.rgb.dtype == np.uint8 for frame in decoded.frames)
    assert all(frame.rgb.shape == (6, 8, 3) for frame in decoded.frames)


def test_decode_video_does_not_repair_invalid_indices(tmp_path: Path) -> None:
    path = tmp_path / "tiny.mp4"
    make_video(path)

    for requested in ([1, 1], [2, 1], [-1, 2]):
        with pytest.raises(ValueError, match="indices"):
            decode_video(VideoSource("sample", "source", path), requested)


def test_decode_video_reports_missing_requested_frame(tmp_path: Path) -> None:
    path = tmp_path / "tiny.mp4"
    make_video(path, frame_count=3)

    with pytest.raises(VideoDecodingError, match=r"do not exist: \[4\]"):
        decode_video(VideoSource("sample", "source", path), [0, 4])
