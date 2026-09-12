"""Fixed-time-grid observation and frozen-model scoring pilot for V7."""

from .grid import (
    classify_fake_window,
    generate_grid_windows,
    interval_union_overlap,
    map_target_frames_to_intervals,
)

__all__ = [
    "classify_fake_window",
    "generate_grid_windows",
    "interval_union_overlap",
    "map_target_frames_to_intervals",
]
