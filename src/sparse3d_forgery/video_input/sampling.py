"""Deterministic source-frame index generation."""

from collections.abc import Sequence

import numpy as np


def _require_integer(value: object, field: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{field} must be an integer")


def explicit_frame_indices(indices: Sequence[int]) -> np.ndarray:
    """Validate caller-selected indices without sorting or deduplicating them."""

    if isinstance(indices, (str, bytes)) or not isinstance(indices, Sequence):
        raise ValueError("indices must be a sequence of integers")
    if len(indices) == 0:
        raise ValueError("indices must be non-empty")
    for index in indices:
        _require_integer(index, "indices")
        if index < 0:
            raise ValueError("indices must be non-negative")
    if any(right <= left for left, right in zip(indices, indices[1:])):
        raise ValueError("indices must be strictly increasing without duplicates")
    try:
        return np.asarray(indices, dtype=np.int64)
    except OverflowError as exc:
        raise ValueError("indices must fit int64") from exc


def fixed_stride_frame_indices(
    start_index: int,
    stop_index_exclusive: int | None = None,
    stride: int = 1,
    max_frames: int | None = None,
) -> np.ndarray:
    """Generate deterministic fixed-stride source indices."""

    _require_integer(start_index, "start_index")
    _require_integer(stride, "stride")
    if start_index < 0:
        raise ValueError("start_index must be non-negative")
    if stride <= 0:
        raise ValueError("stride must be positive")
    if stop_index_exclusive is None and max_frames is None:
        raise ValueError("stop_index_exclusive or max_frames must be provided")
    if stop_index_exclusive is not None:
        _require_integer(stop_index_exclusive, "stop_index_exclusive")
        if stop_index_exclusive <= start_index:
            raise ValueError("stop_index_exclusive must be greater than start_index")
    if max_frames is not None:
        _require_integer(max_frames, "max_frames")
        if max_frames <= 0:
            raise ValueError("max_frames must be positive")

    if stop_index_exclusive is None:
        stop_index_exclusive = start_index + stride * max_frames
    indices = np.arange(start_index, stop_index_exclusive, stride, dtype=np.int64)
    if max_frames is not None:
        indices = indices[:max_frames]
    return indices
