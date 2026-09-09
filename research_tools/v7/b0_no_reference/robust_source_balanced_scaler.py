"""The single pre-declared robust diagonal B0 no-reference scorer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping

import numpy as np


MAD_MULTIPLIER = 1.4826
IQR_DIVISOR = 1.349
NUMERICAL_FLOOR = 1e-12
SCALE_FLOOR = 1e-6
SCORE_EPSILON = 1e-8


def _arrays_by_source(values_by_source: Mapping[str, Iterable[Iterable[float]]]) -> dict[str, np.ndarray]:
    arrays: dict[str, np.ndarray] = {}
    for source_id in sorted(values_by_source):
        array = np.asarray(list(values_by_source[source_id]), dtype=np.float64)
        if array.size == 0:
            continue
        if array.ndim != 2 or array.shape[1] != 4:
            raise ValueError("B0 values must have shape [N,4]")
        arrays[str(source_id)] = array
    if not arrays:
        raise ValueError("at least one source with B0 observations is required")
    return arrays


def source_balanced_quantile(values_by_source: Mapping[str, Iterable[Iterable[float]]], q: float) -> np.ndarray:
    """Return a left-continuous weighted empirical quantile.

    Every source receives total mass ``1 / source_count``; observations within
    a source share that mass equally.  The first value whose cumulative mass
    reaches ``q`` is selected.  This convention is fixed and deterministic.
    """

    if not 0.0 <= q <= 1.0:
        raise ValueError("q must be in [0,1]")
    arrays = _arrays_by_source(values_by_source)
    output = np.empty(4, dtype=np.float64)
    for dimension in range(4):
        entries: list[tuple[float, float]] = []
        eligible: list[tuple[str, np.ndarray]] = []
        for source_id in sorted(arrays):
            values = arrays[source_id][:, dimension]
            values = values[np.isfinite(values)]
            if values.size == 0:
                continue
            eligible.append((source_id, values))
        source_count = len(eligible)
        for _, values in eligible:
            weight = 1.0 / (source_count * values.size)
            entries.extend((float(value), weight) for value in values)
        if not entries:
            output[dimension] = np.nan
            continue
        entries.sort(key=lambda item: item[0])
        target = q
        cumulative = 0.0
        selected = entries[-1][0]
        for value, weight in entries:
            cumulative += weight
            if cumulative + NUMERICAL_FLOOR >= target:
                selected = value
                break
        output[dimension] = selected
    return output


@dataclass(frozen=True)
class RobustDiagonalB0Scaler:
    center: np.ndarray
    scale: np.ndarray
    mad: np.ndarray
    iqr: np.ndarray
    iqr_fallback_dimensions: tuple[int, ...]
    floor_fallback_dimensions: tuple[int, ...]
    training_sources: tuple[str, ...]

    def score(self, values: Iterable[float]) -> float:
        vector = np.asarray(list(values), dtype=np.float64)
        if vector.shape != (4,) or not np.all(np.isfinite(vector)):
            raise ValueError("B0 score input must be finite shape [4]")
        z = (vector - self.center) / (self.scale + SCORE_EPSILON)
        return float(np.sqrt(np.mean(z**2)))

    def as_dict(self) -> dict[str, object]:
        return {
            "center_median": self.center.tolist(),
            "scale": self.scale.tolist(),
            "mad": self.mad.tolist(),
            "iqr": self.iqr.tolist(),
            "mad_multiplier": MAD_MULTIPLIER,
            "iqr_divisor": IQR_DIVISOR,
            "numerical_floor": NUMERICAL_FLOOR,
            "scale_floor": SCALE_FLOOR,
            "score_epsilon": SCORE_EPSILON,
            "iqr_fallback_dimensions": list(self.iqr_fallback_dimensions),
            "floor_fallback_dimensions": list(self.floor_fallback_dimensions),
            "training_sources": list(self.training_sources),
            "source_balance": "equal total weight per source; equal weight within source",
            "quantile_convention": "left-continuous weighted empirical quantile",
            "score": "sqrt(mean(((S-center)/(scale+1e-8))**2))",
        }


def fit_source_balanced_scaler(values_by_source: Mapping[str, Iterable[Iterable[float]]]) -> RobustDiagonalB0Scaler:
    arrays = _arrays_by_source(values_by_source)
    center = source_balanced_quantile(arrays, 0.5)
    deviations = {source: np.abs(values - center) for source, values in arrays.items()}
    mad = source_balanced_quantile(deviations, 0.5)
    q25 = source_balanced_quantile(arrays, 0.25)
    q75 = source_balanced_quantile(arrays, 0.75)
    iqr = q75 - q25
    scale = MAD_MULTIPLIER * mad
    iqr_fallback: list[int] = []
    floor_fallback: list[int] = []
    for dimension in range(4):
        if not np.isfinite(scale[dimension]) or scale[dimension] <= NUMERICAL_FLOOR:
            candidate = iqr[dimension] / IQR_DIVISOR
            if np.isfinite(candidate) and candidate > NUMERICAL_FLOOR:
                scale[dimension] = candidate
                iqr_fallback.append(dimension)
            else:
                scale[dimension] = SCALE_FLOOR
                floor_fallback.append(dimension)
    if not np.all(np.isfinite(center)) or not np.all(np.isfinite(scale)):
        raise ValueError("non-finite robust scaler")
    return RobustDiagonalB0Scaler(
        center=center,
        scale=scale,
        mad=mad,
        iqr=iqr,
        iqr_fallback_dimensions=tuple(iqr_fallback),
        floor_fallback_dimensions=tuple(floor_fallback),
        training_sources=tuple(sorted(arrays)),
    )
