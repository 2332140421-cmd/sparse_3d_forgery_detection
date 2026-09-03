"""Minimal in-memory video input objects."""

from dataclasses import dataclass
import math
from pathlib import Path

import numpy as np


def _require_identity(value: object, field: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")


@dataclass(frozen=True, slots=True)
class VideoSource:
    """Caller-supplied video identity and its I/O locator."""

    sample_id: str
    source_video_id: str
    source_locator: str | Path

    def __post_init__(self) -> None:
        _require_identity(self.sample_id, "sample_id")
        _require_identity(self.source_video_id, "source_video_id")
        if not isinstance(self.source_locator, (str, Path)):
            raise ValueError("source_locator must be a string or pathlib.Path")
        if isinstance(self.source_locator, str) and not self.source_locator.strip():
            raise ValueError("source_locator must not be empty")


@dataclass(frozen=True, slots=True)
class DecodedFrame:
    """One decoder-produced RGB frame on the source-video timeline."""

    source_frame_index: int
    timestamp_s: float
    rgb: np.ndarray

    def __post_init__(self) -> None:
        if isinstance(self.source_frame_index, bool) or not isinstance(
            self.source_frame_index, (int, np.integer)
        ):
            raise ValueError("source_frame_index must be an integer")
        if self.source_frame_index < 0:
            raise ValueError("source_frame_index must be non-negative")
        if not isinstance(self.timestamp_s, (float, np.floating)):
            raise ValueError("timestamp_s must be a float")
        if not math.isfinite(self.timestamp_s):
            raise ValueError("timestamp_s must be finite")
        if not isinstance(self.rgb, np.ndarray):
            raise ValueError("rgb must be a numpy.ndarray")
        if self.rgb.dtype != np.uint8:
            raise ValueError("rgb dtype must be uint8")
        if self.rgb.ndim != 3 or self.rgb.shape[2] != 3:
            raise ValueError("rgb shape must be [H, W, 3]")
        if self.rgb.shape[0] <= 0 or self.rgb.shape[1] <= 0:
            raise ValueError("rgb height and width must be positive")


@dataclass(frozen=True, slots=True)
class DecodedVideoSample:
    """A non-empty, ordered sequence of decoded source frames."""

    sample_id: str
    source_video_id: str
    frames: tuple[DecodedFrame, ...]

    def __post_init__(self) -> None:
        _require_identity(self.sample_id, "sample_id")
        _require_identity(self.source_video_id, "source_video_id")
        if not isinstance(self.frames, tuple):
            raise ValueError("frames must be a tuple")
        if not self.frames:
            raise ValueError("frames must be non-empty")
        if any(not isinstance(frame, DecodedFrame) for frame in self.frames):
            raise ValueError("frames must contain only DecodedFrame values")

        indices = self.frame_indices
        if indices.size > 1 and not np.all(np.diff(indices) > 0):
            raise ValueError("frames.source_frame_index must be strictly increasing")
        timestamps = self.timestamps_s
        if not np.all(np.isfinite(timestamps)):
            raise ValueError("frames.timestamp_s must be finite")
        if timestamps.size > 1 and not np.all(np.diff(timestamps) > 0):
            raise ValueError("frames.timestamp_s must be strictly increasing")

    @property
    def frame_indices(self) -> np.ndarray:
        """Return source indices derived from frames as int64 [T]."""

        return np.fromiter(
            (frame.source_frame_index for frame in self.frames),
            dtype=np.int64,
            count=len(self.frames),
        )

    @property
    def timestamps_s(self) -> np.ndarray:
        """Return decoder timestamps derived from frames as float64 [T]."""

        return np.fromiter(
            (frame.timestamp_s for frame in self.frames),
            dtype=np.float64,
            count=len(self.frames),
        )

    @property
    def frame_sizes_hw(self) -> np.ndarray:
        """Return raster sizes derived from frames as int64 [T, 2]."""

        return np.asarray(
            [(frame.rgb.shape[0], frame.rgb.shape[1]) for frame in self.frames],
            dtype=np.int64,
        )
