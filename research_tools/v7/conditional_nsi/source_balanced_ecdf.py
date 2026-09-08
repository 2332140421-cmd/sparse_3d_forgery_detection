"""Small source-balanced empirical-CDF utilities for the V7 diagnostic."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np


def _finite(values: Sequence[float]) -> np.ndarray:
    array = np.asarray(list(values), dtype=np.float64)
    return array[np.isfinite(array)]


def _midrank_cdf(value: float, values: np.ndarray) -> float:
    """Finite-sample smoothed CDF, with half a tie's mass at the query."""

    if values.size == 0:
        raise ValueError("an ECDF source must contain at least one finite value")
    less = np.count_nonzero(values < value)
    equal = np.count_nonzero(values == value)
    return float((less + 0.5 * equal) / values.size)


def source_balanced_support(source_values: Mapping[str, Sequence[float]]) -> dict[str, float | int]:
    """Return support counts and equal-source effective sample size."""

    finite_by_source = {str(source): _finite(values) for source, values in source_values.items()}
    finite_by_source = {source: values for source, values in finite_by_source.items() if values.size}
    source_count = len(finite_by_source)
    observation_count = int(sum(values.size for values in finite_by_source.values()))
    if source_count == 0:
        return {"distinct_source_count": 0, "observation_count": 0, "effective_n": 0.0}
    weights = np.concatenate([np.full(values.size, 1.0 / (source_count * values.size)) for values in finite_by_source.values()])
    effective_n = float(1.0 / np.sum(weights**2))
    return {
        "distinct_source_count": source_count,
        "observation_count": observation_count,
        "effective_n": effective_n,
    }


def source_balanced_ecdf(value: float, source_values: Mapping[str, Sequence[float]]) -> tuple[float, dict[str, float | int]]:
    """Evaluate an equally weighted per-source smoothed ECDF.

    The return value is ``(q, support)``.  Sources with no finite observations
    are excluded from the reference and reported through the support counts.
    """

    query = float(value)
    if not np.isfinite(query):
        raise ValueError("ECDF query must be finite")
    finite_by_source = {str(source): _finite(values) for source, values in source_values.items()}
    finite_by_source = {source: values for source, values in finite_by_source.items() if values.size}
    support = source_balanced_support(finite_by_source)
    if not finite_by_source:
        raise ValueError("source-balanced ECDF has no finite reference observations")
    q = float(np.mean([_midrank_cdf(query, values) for values in finite_by_source.values()]))
    # Half-rank smoothing keeps q strictly inside (0, 1) for non-empty refs.
    # Clamp only a possible floating-point endpoint after averaging many sources.
    q = float(np.clip(q, np.nextafter(0.0, 1.0), np.nextafter(1.0, 0.0)))
    return q, support
