"""Recover the frozen B0 observations without rerunning the frontend.

The paired pilot persisted the component/time observations that define the
frozen B0 state.  This module only reads those observations and checks them
against the historical per-window CSV.  It never reads labels while scoring
and never opens video, depth, tracking, or particle artifacts.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np


B0_DIMENSIONS = ("mean", "std", "p25", "p75")
B0_KEY = "K0_S"


def _finite_vector(values: Any) -> np.ndarray | None:
    array = np.asarray(values, dtype=np.float64)
    if array.shape != (4,) or not np.all(np.isfinite(array)):
        return None
    return array


def b0_observations(result: dict[str, Any]) -> list[dict[str, Any]]:
    """Return the persisted B0 observations in their original order."""

    window = result["window"]
    features = result.get("features", {})
    observations = features.get("observations", {}).get(B0_KEY, [])
    output: list[dict[str, Any]] = []
    for ordinal, item in enumerate(observations):
        values = _finite_vector(item.get("values"))
        if values is None:
            continue
        output.append(
            {
                "source_id": str(window["source_id"]),
                "pair_id": str(window["pair_id"]),
                "window_id": str(window["window_id"]),
                "role": str(window["role"]),
                "kind": str(window["kind"]),
                "anchor_fraction": float(window["anchor_fraction"]),
                "component_index": int(item.get("component_index", -1)),
                "time_index": int(item.get("time_index", -1)),
                "timestamp_s": float(item["timestamp_s"]),
                "observation_ordinal": ordinal,
                "values": values.tolist(),
            }
        )
    return output


def b0_window_median(result: dict[str, Any]) -> np.ndarray | None:
    """Use the historical component/time median aggregation exactly."""

    observations = b0_observations(result)
    if not observations:
        return None
    return np.median(np.asarray([row["values"] for row in observations], dtype=np.float64), axis=0)


def _historical_medians(path: Path) -> dict[str, np.ndarray | None]:
    rows: dict[str, np.ndarray | None] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            key = str(row["window_id"])
            if key in rows:
                raise ValueError(f"duplicate historical B0 window: {key}")
            value = row.get("K0_S_median", "")
            if not value:
                rows[key] = None
                continue
            parsed = _finite_vector(json.loads(value))
            rows[key] = parsed
    return rows


def load_and_reproduce(
    artifact_root: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Load frozen results and prove exact historical B0 reproduction.

    Returns observation rows, window rows, and a small reproduction record.
    A mismatch raises ``ValueError`` before any no-reference score is made.
    """

    results = json.loads((artifact_root / "frontend/window_results.json").read_text(encoding="utf-8"))
    historical = _historical_medians(artifact_root / "metrics/per_window_signal.csv")
    observations: list[dict[str, Any]] = []
    windows: list[dict[str, Any]] = []
    result_window_ids = {str(result["window"]["window_id"]) for result in results}
    if set(historical) != result_window_ids:
        raise ValueError("historical B0 window identities do not match frozen results")
    max_error = 0.0
    mismatches: list[str] = []
    for result in results:
        if result.get("status") != "COMPLETE":
            raise ValueError(f"incomplete frozen result: {result.get('window', {}).get('window_id')}")
        window = result["window"]
        window_id = str(window["window_id"])
        median = b0_window_median(result)
        expected = historical.get(window_id)
        if (median is None) != (expected is None):
            mismatches.append(window_id)
        elif median is not None and expected is not None:
            error = float(np.max(np.abs(median - expected)))
            max_error = max(max_error, error)
            if error > 1e-12:
                mismatches.append(window_id)
        rows = b0_observations(result)
        observations.extend(rows)
        windows.append(
            {
                "window_id": window_id,
                "pair_id": str(window["pair_id"]),
                "source_id": str(window["source_id"]),
                "role": str(window["role"]),
                "kind": str(window["kind"]),
                "anchor_fraction": float(window["anchor_fraction"]),
                "frame_indices": list(window["frame_indices"]),
                "timestamps_s": list(window["timestamps_s"]),
                "b0_observation_count": len(rows),
                "b0_median": median.tolist() if median is not None else None,
            }
        )
    if mismatches:
        raise ValueError(f"B0_RECONSTRUCTION_MISMATCH:{mismatches[:5]}")
    return observations, windows, {
        "historical_csv": str(artifact_root / "metrics/per_window_signal.csv"),
        "window_count": len(windows),
        "observation_count": len(observations),
        "matched_window_count": len(windows),
        "mismatch_count": 0,
        "max_abs_error": max_error,
        "aggregation": "median over persisted K0_S component/time observations per window",
        "definition": "S_t=[mean,std,p25,p75]",
    }
