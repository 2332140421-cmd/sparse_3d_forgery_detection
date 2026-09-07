"""V7 real-only structural-normality experiment tooling."""

from .protocol import (
    ANCHOR_FRACTIONS,
    COMPONENT_CONFIG,
    TIMESCALE_S,
    GaussianNormality,
    build_source_split,
    build_window_manifest,
    extract_window_features,
    feature_schema,
    summarize_scores,
    video_score,
)

__all__ = [
    "ANCHOR_FRACTIONS",
    "COMPONENT_CONFIG",
    "TIMESCALE_S",
    "GaussianNormality",
    "build_source_split",
    "build_window_manifest",
    "extract_window_features",
    "feature_schema",
    "summarize_scores",
    "video_score",
]
