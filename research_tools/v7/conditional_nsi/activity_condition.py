"""The single frozen lagged-activity condition used by conditional NSI."""

from __future__ import annotations

from typing import Iterable

import numpy as np


CONDITION_NAME = "lagged_first_order_structural_activity"


def activity_condition(v_minus_norm: float, pair_count: int | float) -> float:
    """Return C = ||v_minus|| / sqrt(M), without adding a second signal."""

    norm = float(v_minus_norm)
    count = float(pair_count)
    if not np.isfinite(norm) or not np.isfinite(count) or count <= 0:
        raise ValueError("activity condition requires finite norm and positive pair count")
    return float(norm / np.sqrt(count))


def tertile_cutpoints(values: Iterable[float]) -> tuple[float, float]:
    """Compute deterministic 1/3 and 2/3 cutpoints from training real C only."""

    array = np.asarray(list(values), dtype=np.float64)
    array = array[np.isfinite(array)]
    if array.size == 0:
        raise ValueError("at least one finite training-real condition is required")
    cuts = np.quantile(array, (1.0 / 3.0, 2.0 / 3.0), method="linear")
    return float(cuts[0]), float(cuts[1])


def activity_bin(value: float, cuts: tuple[float, float]) -> str:
    """Assign LOW/MID/HIGH using fixed fold-local training cutpoints."""

    query = float(value)
    if not np.isfinite(query):
        raise ValueError("activity value must be finite")
    first, second = (float(cuts[0]), float(cuts[1]))
    if not (np.isfinite(first) and np.isfinite(second) and first <= second):
        raise ValueError("tertile cutpoints must be finite and ordered")
    return ("LOW", "MID", "HIGH")[int(np.searchsorted(np.asarray((first, second)), query, side="right"))]


def anomaly_transform(q: float) -> float:
    """Apply the frozen two-sided no-direction anomaly transform."""

    percentile = float(q)
    if not np.isfinite(percentile) or not 0.0 < percentile < 1.0:
        raise ValueError("smoothed ECDF percentile must lie strictly between zero and one")
    return float(-np.log(2.0 * min(percentile, 1.0 - percentile)))


def ks_distance_to_uniform(values: Iterable[float]) -> float | None:
    """Descriptive one-sample KS distance for percentiles versus Uniform(0,1)."""

    array = np.asarray(list(values), dtype=np.float64)
    array = np.sort(array[np.isfinite(array)])
    if array.size == 0:
        return None
    n = array.size
    upper = np.max(np.abs(np.arange(1, n + 1, dtype=np.float64) / n - array))
    lower = np.max(np.abs(array - np.arange(0, n, dtype=np.float64) / n))
    return float(max(upper, lower))


def median_absolute_deviation(values: Iterable[float]) -> float | None:
    """Median absolute deviation of finite values."""

    array = np.asarray(list(values), dtype=np.float64)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return None
    median = float(np.median(array))
    return float(np.median(np.abs(array - median)))
