"""Small, deterministic protocol and Gaussian baseline for V7 H3."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

import numpy as np

from sparse3d_forgery.experiments.v7_dynamic_structure_probe import (
    ComponentConfig,
    motion_coherent_components,
    structural_differences,
    structure_state,
)


TIMESCALE_S = 1.0
ANCHOR_FRACTIONS = (0.25, 0.50, 0.75)
COMPONENT_CONFIG = ComponentConfig(
    max_initial_distance=1.0,
    max_relative_change=0.05,
    minimum_overlap=8,
    minimum_size=3,
)
MODEL_NAMES = ("M1_delta_s", "M2_delta2_s", "M3_delta_s_delta2_s")


def _finite_distribution(values: Iterable[float]) -> dict[str, float | int | None]:
    array = np.asarray(list(values), dtype=np.float64)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return {"count": 0, "median": None, "iqr": None, "p10": None, "p90": None}
    return {
        "count": int(array.size),
        "median": float(np.median(array)),
        "iqr": float(np.percentile(array, 75) - np.percentile(array, 25)),
        "p10": float(np.percentile(array, 10)),
        "p90": float(np.percentile(array, 90)),
    }


def build_source_split(media_rows: Sequence[Mapping[str, object]]) -> dict[str, list[dict[str, object]]]:
    """Split valid frozen sources by order; no random selection is permitted."""

    valid = [dict(row) for row in media_rows if row.get("status") == "MEDIA_VALID"]
    train_count = int(np.floor(len(valid) * 0.75))
    if len(valid) >= 2:
        train_count = min(max(train_count, 1), len(valid) - 1)
    return {"real_train": valid[:train_count], "real_val": valid[train_count:]}


def _window_indices(timestamps_s: Sequence[float], duration_s: float, anchor_fraction: float) -> list[int]:
    values = np.asarray(timestamps_s, dtype=np.float64)
    if values.ndim != 1 or values.size < 2 or not np.all(np.isfinite(values)) or not np.all(np.diff(values) > 0):
        raise ValueError("timestamps must be finite and strictly increasing")
    if duration_s <= 0 or not 0.0 <= anchor_fraction <= 1.0:
        raise ValueError("invalid window protocol")
    start, end = float(values[0]), float(values[-1])
    center = start + (end - start) * anchor_fraction
    if end - start < duration_s:
        return []
    left, right = center - duration_s / 2.0, center + duration_s / 2.0
    return [int(i) for i, value in enumerate(values) if left <= value <= right]


def build_window_manifest(media_rows: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    """Make three true-time, no-padding windows per valid source in frozen order."""

    rows: list[dict[str, object]] = []
    for source in media_rows:
        if source.get("status") != "MEDIA_VALID":
            continue
        source_id = str(source["source_id"])
        for anchor_fraction in ANCHOR_FRACTIONS:
            indices = _window_indices(source["timestamps_s"], TIMESCALE_S, anchor_fraction)
            timestamps = [float(source["timestamps_s"][i]) for i in indices]
            window_id = f"{source_id}::anchor-{int(anchor_fraction * 100):02d}"
            rows.append(
                {
                    "window_id": window_id,
                    "source_id": source_id,
                    "video_path": str(source["video_path"]),
                    "role": str(source.get("role", "real")),
                    "anchor_fraction": anchor_fraction,
                    "timescale_s": TIMESCALE_S,
                    "frame_indices": indices,
                    "timestamps_s": timestamps,
                    "status": "AVAILABLE" if len(indices) >= 2 else "WINDOW_DURATION_UNAVAILABLE",
                    "fake_used": False,
                }
            )
    return rows


def feature_schema() -> dict[str, object]:
    return {
        "M1_delta_s": {"source": "first_order_structural_change", "shape": [4], "dtype": "float64"},
        "M2_delta2_s": {"source": "second_order_structural_evolution", "shape": [4], "dtype": "float64"},
        "M3_delta_s_delta2_s": {"source": "concatenate(M1,M2)", "shape": [8], "dtype": "float64"},
        "timestamp_derivative": True,
        "fake_used_for_features": False,
    }


def extract_window_features(sequence, *, source_id: str, window_id: str) -> dict[str, object]:
    """Run the existing component/state code and retain only M1/M2/M3 rows."""

    components = motion_coherent_components(sequence.xyz, sequence.geometry_validity, COMPONENT_CONFIG)
    observations: dict[str, list[dict[str, object]]] = {name: [] for name in MODEL_NAMES}
    component_rows: list[dict[str, object]] = []
    valid_state_count = valid_delta_count = valid_delta2_count = 0
    for component_index, component in enumerate(components):
        state, state_valid = structure_state(sequence.xyz, sequence.geometry_validity, component)
        first, first_valid, second, second_valid = structural_differences(
            state, state_valid, sequence.timestamps_s
        )
        valid_state_count += int(np.sum(state_valid))
        valid_delta_count += int(np.sum(first_valid))
        valid_delta2_count += int(np.sum(second_valid))
        component_rows.append(
            {
                "component_index": component_index,
                "members": list(component),
                "size": len(component),
                "valid_state_count": int(np.sum(state_valid)),
            }
        )
        for time_index in np.flatnonzero(first_valid):
            values = np.asarray(first[time_index], dtype=np.float64)
            row = {
                "source_id": source_id,
                "window_id": window_id,
                "component_index": component_index,
                "time_index": int(time_index),
                "timestamp_s": float(sequence.timestamps_s[time_index]),
                "values": values.tolist(),
            }
            observations["M1_delta_s"].append(row)
        for time_index in np.flatnonzero(second_valid):
            values = np.asarray(second[time_index], dtype=np.float64)
            row = {
                "source_id": source_id,
                "window_id": window_id,
                "component_index": component_index,
                "time_index": int(time_index),
                "timestamp_s": float(sequence.timestamps_s[time_index]),
                "values": values.tolist(),
            }
            observations["M2_delta2_s"].append(row)
            if first_valid[time_index]:
                first_values = np.asarray(first[time_index], dtype=np.float64)
                observations["M3_delta_s_delta2_s"].append(
                    {**row, "values": np.concatenate((first_values, values)).tolist()}
                )
    return {
        "source_id": source_id,
        "window_id": window_id,
        "component_count": len(components),
        "component_rows": component_rows,
        "valid_s_states": valid_state_count,
        "valid_delta_s": valid_delta_count,
        "valid_delta2_s": valid_delta2_count,
        "observations": observations,
        "fake_used": False,
    }


@dataclass(frozen=True)
class GaussianNormality:
    mean: np.ndarray
    covariance: np.ndarray
    covariance_type: str
    regularization_lambda: float
    condition_number: float
    feature_count: int

    @classmethod
    def fit(cls, values: np.ndarray) -> "GaussianNormality":
        values = np.asarray(values, dtype=np.float64)
        if values.ndim != 2 or values.shape[0] == 0 or not np.all(np.isfinite(values)):
            raise ValueError("Gaussian fitting requires finite [count, dimension] values")
        mean = np.mean(values, axis=0)
        raw = np.cov(values, rowvar=False, ddof=1) if values.shape[0] > 1 else np.zeros((values.shape[1], values.shape[1]))
        raw = np.atleast_2d(np.asarray(raw, dtype=np.float64))
        diag_mean = float(np.mean(np.diag(raw)))
        regularization = 1e-5 * max(diag_mean, 1e-12)
        identity = np.eye(values.shape[1], dtype=np.float64)
        full = raw + regularization * identity
        condition = float(np.linalg.cond(full)) if np.all(np.isfinite(full)) else float("inf")
        if values.shape[0] <= values.shape[1] or not np.isfinite(condition) or condition > 1e12:
            covariance_type = "diagonal"
            diagonal = np.maximum(np.diag(raw), regularization)
            covariance = np.diag(diagonal + regularization)
        else:
            covariance_type = "full"
            covariance = full
        condition = float(np.linalg.cond(covariance))
        return cls(mean, covariance, covariance_type, regularization, condition, int(values.shape[0]))

    def score(self, values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != self.mean.size:
            raise ValueError("feature dimension mismatch")
        delta = values - self.mean
        if self.covariance_type == "diagonal":
            return np.sum((delta * delta) / np.diag(self.covariance), axis=1)
        solved = np.linalg.solve(self.covariance, delta.T).T
        return np.sum(delta * solved, axis=1)

    def as_dict(self) -> dict[str, object]:
        return {
            "mean": self.mean.tolist(),
            "covariance": self.covariance.tolist(),
            "covariance_type": self.covariance_type,
            "regularization_lambda": self.regularization_lambda,
            "condition_number": self.condition_number,
            "feature_count": self.feature_count,
        }


def summarize_scores(values: Sequence[float]) -> dict[str, float | int | None]:
    return _finite_distribution(values)


def video_score(values: Sequence[float], aggregation: str) -> float | None:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return None
    if aggregation == "median":
        return float(np.median(finite))
    if aggregation == "p95":
        return float(np.percentile(finite, 95))
    raise ValueError(f"unsupported score aggregation: {aggregation}")
