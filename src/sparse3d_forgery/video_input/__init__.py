"""Provider-neutral video input and deterministic sampling."""

from .sampling import explicit_frame_indices, fixed_stride_frame_indices
from .schema import DecodedFrame, DecodedVideoSample, VideoSource

__all__ = [
    "DecodedFrame",
    "DecodedVideoSample",
    "VideoSource",
    "explicit_frame_indices",
    "fixed_stride_frame_indices",
]
