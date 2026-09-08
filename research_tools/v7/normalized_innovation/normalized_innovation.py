"""Scalar NSI definition and small statistical helpers for the V7 pilot."""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Iterable, Sequence

import numpy as np

from research_tools.v7.paired_signal.analyze import (
    BOOTSTRAP_REPLICATES,
    BOOTSTRAP_SEED,
    _auroc,
    _bootstrap_median,
    _sign_test,
    _spearman,
    finite_summary,
)


EPSILON = 1e-8


def normalized_structural_innovation(v_minus: Sequence[float], v_plus: Sequence[float], epsilon: float = EPSILON) -> float:
    """Return the frozen dimensionless normalized local structural innovation."""

    left = np.asarray(v_minus, dtype=np.float64)
    right = np.asarray(v_plus, dtype=np.float64)
    if left.ndim != 1 or right.shape != left.shape or not np.all(np.isfinite(left)) or not np.all(np.isfinite(right)):
        raise ValueError("NSI velocities must be finite one-dimensional vectors of equal shape")
    numerator = float(np.linalg.norm(right - left))
    denominator = float(np.linalg.norm(right) + np.linalg.norm(left))
    return numerator / (denominator + float(epsilon))


def component_q90(values: Iterable[float]) -> float:
    array = np.asarray(list(values), dtype=np.float64)
    if array.size == 0 or not np.all(np.isfinite(array)):
        raise ValueError("component Q90 requires finite values")
    return float(np.percentile(array, 90))


def robust_real_scale(values: Iterable[float]) -> tuple[float, dict[str, Any]]:
    array = np.asarray(list(values), dtype=np.float64)
    array = array[np.isfinite(array)]
    if array.size == 0:
        raise ValueError("real-only calibration requires at least one value")
    center = float(np.median(array))
    mad = float(np.median(np.abs(array - center)))
    scale = 1.4826 * mad
    fallback = None
    if not np.isfinite(scale) or scale == 0:
        fallback = float((np.percentile(array, 75) - np.percentile(array, 25)) / 1.349)
        scale = fallback
    if not np.isfinite(scale) or scale == 0:
        scale = 1e-6
        fallback = "floor"
    return float(scale), {
        "N_real_calibration": int(array.size),
        "median": center,
        "mad": mad,
        "scale": float(scale),
        "fallback": fallback,
        "source": "other_sources_real_windows_only",
    }


def leave_one_source_out(rows: Sequence[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Calibrate Z_I from real windows belonging to other sources only."""

    sources = sorted({str(row["source_id"]) for row in rows})
    calibrated: list[dict[str, Any]] = []
    calibration_meta: dict[str, Any] = {}
    for held_out in sources:
        train = [row["I_window"] for row in rows if row["role"] == "real" and str(row["source_id"]) != held_out and row["I_window"] is not None]
        scale, details = robust_real_scale(train)
        calibration_meta[held_out] = {**details, "held_out_source": held_out, "fake_count": 0}
        for row in rows:
            if str(row["source_id"]) != held_out:
                continue
            copy = dict(row)
            copy["Z_I"] = float(abs(float(row["I_window"]) - details["median"]) / (scale + EPSILON)) if row["I_window"] is not None else None
            copy["calibration_source_count"] = len(train)
            calibrated.append(copy)
    calibrated.sort(key=lambda row: row["window_id"])
    return calibrated, calibration_meta


def metric_summary(values: Iterable[float], include_positive: bool = False) -> dict[str, Any]:
    array = np.asarray(list(values), dtype=np.float64)
    array = array[np.isfinite(array)]
    result: dict[str, Any] = finite_summary(array)
    if include_positive:
        result["positive_fraction"] = float(np.mean(array > 0)) if array.size else None
    return result


def signed_summary(values: Iterable[float]) -> dict[str, Any]:
    array = np.asarray(list(values), dtype=np.float64)
    array = array[np.isfinite(array)]
    return {
        **finite_summary(array),
        "positive_fraction": float(np.mean(array > 0)) if array.size else None,
        "sign_test": _sign_test(array),
        "bootstrap_median": _bootstrap_median(array),
    }


def paired_nsi_gain(rows: Sequence[dict[str, Any]], selected_pairs: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Compute frozen manipulation/control paired discrepancy for each source."""

    grouped: dict[tuple[str, str, str, float], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        grouped[(row["pair_id"], row["kind"], row["label"], float(row["anchor_fraction"]))][row["role"]] = row
    by_pair: dict[str, dict[str, Any]] = {
        str(pair["pair_id"]): {
            "pair_id": str(pair["pair_id"]),
            "source_id": str(pair["source_id"]),
            "D_manip": [],
            "D_ctrl": [],
        }
        for pair in selected_pairs
    }
    for (pair_id, kind, _label, _anchor), sides in grouped.items():
        if pair_id not in by_pair or "real" not in sides or "fake" not in sides:
            continue
        if sides["real"].get("I_window") is None or sides["fake"].get("I_window") is None:
            continue
        by_pair[pair_id][f"D_{kind.lower()}"] .append(abs(float(sides["fake"]["I_window"]) - float(sides["real"]["I_window"])))
    output: list[dict[str, Any]] = []
    for item in by_pair.values():
        manip = np.asarray(item["D_manip"], dtype=np.float64)
        ctrl = np.asarray(item["D_ctrl"], dtype=np.float64)
        output.append(
            {
                "pair_id": item["pair_id"],
                "source_id": item["source_id"],
                "D_manip": float(np.median(manip)) if manip.size else None,
                "D_ctrl": float(np.median(ctrl)) if ctrl.size else None,
                "G_I": float(np.median(manip) - np.median(ctrl)) if manip.size and ctrl.size else None,
            }
        )
    return output


def group_medians(rows: Sequence[dict[str, Any]], value_key: str) -> dict[tuple[str, str, str], float | None]:
    grouped: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for row in rows:
        value = row.get(value_key)
        if value is not None and np.isfinite(float(value)):
            grouped[(str(row["source_id"]), str(row["role"]), str(row["kind"]))].append(float(value))
    return {key: (float(np.median(values)) if values else None) for key, values in grouped.items()}


def cross_source_summary(rows: Sequence[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Return source-level ruler gains and group summaries after LOSO calibration."""

    groups = group_medians(rows, "Z_I")
    all_sources = sorted({str(row["source_id"]) for row in rows})
    source_rows: list[dict[str, Any]] = []
    for source in all_sources:
        values = {f"Z_{role}_{kind.lower()}": groups.get((source, role, kind)) for role in ("real", "fake") for kind in ("MANIP", "CTRL")}
        if all(values[key] is not None for key in ("Z_real_manip", "Z_real_ctrl", "Z_fake_manip", "Z_fake_ctrl")):
            values["H_I"] = (values["Z_fake_manip"] - values["Z_real_manip"]) - (values["Z_fake_ctrl"] - values["Z_real_ctrl"])
        else:
            values["H_I"] = None
        values["source_id"] = source
        source_rows.append(values)
    summary: dict[str, Any] = {
        key: metric_summary([row[key] for row in source_rows if row.get(key) is not None])
        for key in ("Z_real_manip", "Z_real_ctrl", "Z_fake_manip", "Z_fake_ctrl")
    }
    h = [row["H_I"] for row in source_rows if row.get("H_I") is not None]
    summary["H_I"] = signed_summary(h)
    summary["real_manip_minus_ctrl"] = signed_summary([row["Z_real_manip"] - row["Z_real_ctrl"] for row in source_rows if row.get("Z_real_manip") is not None and row.get("Z_real_ctrl") is not None])
    summary["fake_manip_minus_ctrl"] = signed_summary([row["Z_fake_manip"] - row["Z_fake_ctrl"] for row in source_rows if row.get("Z_fake_manip") is not None and row.get("Z_fake_ctrl") is not None])
    summary["exploratory_auroc_fake_manip_vs_all_real"] = _auroc(
        [row["Z_I"] for row in rows if row["role"] == "fake" and row["kind"] == "MANIP" and row.get("Z_I") is not None],
        [row["Z_I"] for row in rows if row["role"] == "real" and row.get("Z_I") is not None],
    )
    return source_rows, summary


def activity_correlations(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    metrics = ("speed_sum_median", "raw_delta_r_median", "raw_second_difference_median")
    output: dict[str, Any] = {"window_level": {}, "real_window_level": {}}
    for value_key in ("I_window", "Z_I"):
        output["window_level"][value_key] = {metric: _spearman([row.get(value_key) for row in rows], [row.get(metric) for row in rows]) for metric in metrics}
        output["real_window_level"][value_key] = {metric: _spearman([row.get(value_key) for row in rows if row["role"] == "real"], [row.get(metric) for row in rows if row["role"] == "real"]) for metric in metrics}
    output["strong_threshold"] = 0.7
    return output


def pair_dimension_diagnostics(rows: Sequence[dict[str, Any]], pair_rows: Sequence[dict[str, Any]], source_rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    output = {
        "window_level_I_vs_common_pair_dimension": _spearman([row.get("I_window") for row in rows], [row.get("common_pair_dimension_median") for row in rows]),
        "window_level_Z_vs_common_pair_dimension": _spearman([row.get("Z_I") for row in rows], [row.get("common_pair_dimension_median") for row in rows]),
    }
    by_source = {row["source_id"]: row for row in source_rows}
    gaps: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        if row["role"] not in ("real", "fake") or row["kind"] not in ("MANIP", "CTRL"):
            continue
        key = (str(row["source_id"]), str(row["role"]), str(row["kind"]))
        gaps[key].append(float(row["common_pair_dimension_median"])) if row.get("common_pair_dimension_median") is not None else None
    source_medians = {key: (float(np.median(value)) if value else None) for key, value in gaps.items()}
    dims_gap = []
    g_values = []
    h_values = []
    for row in pair_rows:
        source = str(row["source_id"])
        dim_fake = source_medians.get((source, "fake", "MANIP"))
        dim_real = source_medians.get((source, "real", "MANIP"))
        if dim_fake is not None and dim_real is not None:
            dims_gap.append(dim_fake - dim_real)
            g_values.append(row.get("G_I"))
            h_values.append(by_source.get(source, {}).get("H_I"))
    output["source_level_G_I_vs_manipulation_dimension_gap"] = _spearman(g_values, dims_gap)
    output["source_level_H_I_vs_manipulation_dimension_gap"] = _spearman(h_values, dims_gap)
    output["strong_threshold"] = 0.7
    output["flag"] = any(
        abs(value) >= 0.7
        for key, value in output.items()
        if key != "strong_threshold" and isinstance(value, (int, float))
    )
    return output
