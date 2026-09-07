"""Fit and score the frozen real-only Gaussian normality protocol."""

from __future__ import annotations

from collections import defaultdict
from typing import Mapping, Sequence

import numpy as np

from .protocol import GaussianNormality, MODEL_NAMES, summarize_scores, video_score


def _rows_by_model(window_rows: Sequence[Mapping[str, object]], model: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for window in window_rows:
        observations = window.get("observations", {})
        rows.extend(dict(row) for row in observations.get(model, []))
    return rows


def fit_models(train_windows: Sequence[Mapping[str, object]]) -> dict[str, GaussianNormality]:
    models: dict[str, GaussianNormality] = {}
    for model in MODEL_NAMES:
        rows = _rows_by_model(train_windows, model)
        values = np.asarray([row["values"] for row in rows], dtype=np.float64)
        models[model] = GaussianNormality.fit(values)
    return models


def score_windows(
    models: Mapping[str, GaussianNormality],
    windows: Sequence[Mapping[str, object]],
) -> tuple[list[dict[str, object]], dict[str, dict[str, list[float]]]]:
    scored: list[dict[str, object]] = []
    per_source: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for window in windows:
        metadata = window.get("window", window)
        output = {"window_id": metadata["window_id"], "source_id": metadata["source_id"], "role": metadata["role"]}
        for model, normality in models.items():
            values = np.asarray([row["values"] for row in window.get("observations", {}).get(model, [])], dtype=np.float64)
            scores = normality.score(values) if values.size else np.asarray([], dtype=np.float64)
            output[model] = {
                "feature_count": int(values.shape[0]) if values.ndim == 2 else 0,
                "score_summary": summarize_scores(scores),
                "video_median": video_score(scores, "median"),
                "video_p95": video_score(scores, "p95"),
            }
            per_source[str(metadata["source_id"])][model].extend(scores.tolist())
        scored.append(output)
    return scored, {source: dict(values) for source, values in per_source.items()}


def source_summaries(
    models: Mapping[str, GaussianNormality],
    windows: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    scored, _ = score_windows(models, windows)
    by_source: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    feature_counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for row in scored:
        source = str(row["source_id"])
        for model in models:
            feature_counts[source][model] += int(row[model]["feature_count"])
            value = row[model]["video_median"]
            if value is not None:
                by_source[source][model].append(float(value))
    result: dict[str, object] = {}
    for source in sorted(by_source):
        result[source] = {
            model: {
                "window_median_scores": values,
                "source_median_score": float(np.median(values)) if values else None,
                "feature_count": int(feature_counts[source][model]),
            }
            for model, values in by_source[source].items()
        }
    return result
