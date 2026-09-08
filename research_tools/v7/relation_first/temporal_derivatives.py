"""Timestamp-aware derivatives for a persistent relation trajectory."""

from __future__ import annotations

import numpy as np


def timestamp_derivatives(
    values: np.ndarray,
    valid: np.ndarray,
    timestamps_s: np.ndarray,
    frame_indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute first/second derivatives without crossing invalid or frame gaps."""

    values = np.asarray(values, dtype=np.float64)
    valid = np.asarray(valid, dtype=bool)
    timestamps_s = np.asarray(timestamps_s, dtype=np.float64)
    frame_indices = np.asarray(frame_indices, dtype=np.int64)
    if values.ndim != 1 or valid.shape != values.shape or timestamps_s.shape != values.shape or frame_indices.shape != values.shape:
        raise ValueError("values, validity, timestamps, and frame indices must be one-dimensional and aligned")
    if values.size > 1 and not np.all(np.diff(timestamps_s) > 0):
        raise ValueError("timestamps must be strictly increasing")
    first = np.full(values.shape, np.nan, dtype=np.float64)
    first_valid = np.zeros(values.shape, dtype=bool)
    for index in range(1, values.size):
        dt = timestamps_s[index] - timestamps_s[index - 1]
        adjacent = frame_indices[index] == frame_indices[index - 1] + 1
        if valid[index - 1] and valid[index] and adjacent and np.isfinite(dt) and dt > 0:
            candidate = (values[index] - values[index - 1]) / dt
            if np.isfinite(candidate):
                first[index] = candidate
                first_valid[index] = True
    second = np.full(values.shape, np.nan, dtype=np.float64)
    second_valid = np.zeros(values.shape, dtype=bool)
    for index in range(2, values.size):
        h0 = timestamps_s[index - 1] - timestamps_s[index - 2]
        h1 = timestamps_s[index] - timestamps_s[index - 1]
        adjacent = frame_indices[index - 1] == frame_indices[index - 2] + 1 and frame_indices[index] == frame_indices[index - 1] + 1
        if not (first_valid[index - 1] and first_valid[index] and adjacent and np.isfinite(h0) and np.isfinite(h1) and h0 > 0 and h1 > 0):
            continue
        candidate = 2.0 * (first[index] - first[index - 1]) / (h0 + h1)
        if np.isfinite(candidate):
            second[index] = candidate
            second_valid[index] = True
    return first, first_valid, second, second_valid
