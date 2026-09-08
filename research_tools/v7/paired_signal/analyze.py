"""Analyze the frozen paired second-order pilot without fitting a detector.

The analysis uses only matched real/fake windows.  Robust scales are estimated
from real control windows; fake observations never enter that fitting step.
"""

from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .protocol import ORDER_NAMES, finite_summary, robust_scale, window_observation_median


BOOTSTRAP_SEED = 20260909
BOOTSTRAP_REPLICATES = 10_000


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False, ensure_ascii=False) + "\n", encoding="utf-8")


def _finite(values: Iterable[float]) -> np.ndarray:
    array = np.asarray(list(values), dtype=np.float64)
    return array[np.isfinite(array)]


def _positive_fraction(values: Iterable[float]) -> float | None:
    array = _finite(values)
    return float(np.mean(array > 0)) if array.size else None


def _sign_test(values: Iterable[float]) -> dict[str, Any]:
    array = _finite(values)
    signs = np.sign(array[array != 0]).astype(np.int64)
    n = int(signs.size)
    if n == 0:
        return {"N_nonzero": 0, "positive": 0, "negative": 0, "p_two_sided": None}
    positive = int(np.sum(signs > 0))
    negative = n - positive
    tail = sum(math.comb(n, k) for k in range(min(positive, negative) + 1)) / (2.0**n)
    return {"N_nonzero": n, "positive": positive, "negative": negative, "p_two_sided": float(min(1.0, 2.0 * tail))}


def _rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=np.float64)
    sorted_values = values[order]
    start = 0
    while start < values.size:
        end = start + 1
        while end < values.size and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2.0 + 1.0
        start = end
    return ranks


def _spearman(x: Iterable[float], y: Iterable[float]) -> float | None:
    pairs = []
    for left_value, right_value in zip(x, y):
        try:
            left_float, right_float = float(left_value), float(right_value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(left_float) and math.isfinite(right_float):
            pairs.append((left_float, right_float))
    if len(pairs) < 3:
        return None
    left = np.asarray([item[0] for item in pairs], dtype=np.float64)
    right = np.asarray([item[1] for item in pairs], dtype=np.float64)
    rx, ry = _rankdata(left), _rankdata(right)
    if np.std(rx) == 0 or np.std(ry) == 0:
        return None
    return float(np.corrcoef(rx, ry)[0, 1])


def _bootstrap_median(values: Iterable[float]) -> dict[str, Any]:
    array = _finite(values)
    if array.size == 0:
        return {"N": 0, "seed": BOOTSTRAP_SEED, "replicates": BOOTSTRAP_REPLICATES, "median": None, "ci95": None}
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    samples = rng.integers(0, array.size, size=(BOOTSTRAP_REPLICATES, array.size))
    medians = np.median(array[samples], axis=1)
    return {
        "N": int(array.size),
        "seed": BOOTSTRAP_SEED,
        "replicates": BOOTSTRAP_REPLICATES,
        "median": float(np.median(array)),
        "ci95": [float(np.percentile(medians, 2.5)), float(np.percentile(medians, 97.5))],
    }


def _auroc(manip: Iterable[float], control: Iterable[float]) -> float | None:
    positive = _finite(manip)
    negative = _finite(control)
    if positive.size == 0 or negative.size == 0:
        return None
    comparisons = (positive[:, None] > negative[None, :]).astype(np.float64)
    comparisons += 0.5 * (positive[:, None] == negative[None, :])
    return float(np.mean(comparisons))


def _quality_summary(results: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for result in results:
        window = result.get("window", {})
        grouped[f"{window.get('role')}::{window.get('kind')}"].append(result)
    output: dict[str, Any] = {}
    for key, rows in sorted(grouped.items()):
        quality_keys = ("geometry_coverage", "tracking_persistence", "pose_success", "pose_valid_fraction", "s_valid_fraction")
        output[key] = {
            "N": len(rows),
            "complete": sum(row.get("status") == "COMPLETE" for row in rows),
            "failures": sum(row.get("status") != "COMPLETE" for row in rows),
            **{name: finite_summary(row.get("quality", {}).get(name) for row in rows) for name in quality_keys},
            "valid_delta_s": finite_summary(row.get("quality", {}).get("valid_delta_s", 0) for row in rows),
            "valid_delta2_s": finite_summary(row.get("quality", {}).get("valid_delta2_s", 0) for row in rows),
            "component_count": finite_summary(row.get("quality", {}).get("component_count", 0) for row in rows),
            "component_success_fraction": finite_summary(float(bool(row.get("quality", {}).get("component_success", False))) for row in rows),
        }
    return output


def _window_rows(results: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, list[list[float]]]]:
    rows: list[dict[str, Any]] = []
    medians: dict[str, list[list[float]]] = defaultdict(list)
    for result in results:
        if result.get("status") != "COMPLETE":
            continue
        window = result["window"]
        row: dict[str, Any] = {
            "pair_id": window["pair_id"],
            "source_id": window["source_id"],
            "window_id": window["window_id"],
            "role": window["role"],
            "kind": window["kind"],
            "anchor_fraction": window["anchor_fraction"],
            "generator": None,
            "manipulation_operation": None,
            "elapsed_s": result.get("elapsed_s"),
            "peak_gpu_memory_bytes": result.get("peak_gpu_memory_bytes"),
            "causal_training_eligible": result.get("causal_training_eligible"),
            "preview_used": result.get("preview_used"),
            **result.get("quality", {}),
        }
        for key in ORDER_NAMES:
            median, details = window_observation_median(result.get("features", {}), key)
            row[f"{key}_valid"] = median is not None
            row[f"{key}_norm"] = float(np.linalg.norm(median)) if median is not None else None
            row[f"{key}_median"] = json.dumps(median.tolist()) if median is not None else None
            if median is not None:
                medians[f"{window['pair_id']}::{window['role']}::{window['kind']}::{window['anchor_fraction']}::{key}"].append(median.tolist())
            row[f"{key}_observation_count"] = details.get("observation_count", 0)
        pairwise = result.get("features", {}).get("pairwise", {})
        row["pairwise_distance_count"] = pairwise.get("pair_count", 0)
        row["pairwise_temporal_mad"] = pairwise.get("temporal_mad")
        row["pairwise_temporal_iqr"] = pairwise.get("temporal_iqr")
        rows.append(row)
    return rows, medians


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _pair_rows(selected: list[dict[str, Any]], results: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    complete = [row for row in results if row.get("status") == "COMPLETE"]
    by_key: dict[tuple[str, str, str, float], dict[str, dict[str, Any]]] = defaultdict(dict)
    for result in complete:
        window = result["window"]
        by_key[(window["pair_id"], window["kind"], window["label"], float(window["anchor_fraction"]))][window["role"]] = result
    scales: dict[str, Any] = {}
    for key in ORDER_NAMES:
        controls: list[list[float]] = []
        for (pair_id, kind, label, anchor), sides in by_key.items():
            if kind != "CTRL" or "real" not in sides:
                continue
            median, _ = window_observation_median(sides["real"].get("features", {}), key)
            if median is not None:
                controls.append(median.tolist())
        if controls:
            scale, details = robust_scale(controls)
            scales[key] = {"details": details, "N_real_control_windows": len(controls)}
        else:
            scales[key] = {"details": None, "N_real_control_windows": 0}
    values_by_pair: dict[str, dict[str, Any]] = {}
    for pair in selected:
        values_by_pair[pair["pair_id"]] = {
            "pair_id": pair["pair_id"],
            "source_id": pair["source_id"],
            "generator": pair["generator"],
            "manipulation_operation": pair["manipulation_operation"],
            "D": {key: {"MANIP": [], "CTRL": []} for key in ORDER_NAMES},
            "quality": {kind: {metric: {role: [] for role in ("real", "fake")} for metric in ("geometry_coverage", "tracking_persistence", "pose_success", "valid_delta2_s")} for kind in ("MANIP", "CTRL")},
        }
    for (pair_id, kind, label, anchor), sides in by_key.items():
        if pair_id not in values_by_pair or "real" not in sides or "fake" not in sides:
            continue
        target = values_by_pair[pair_id]
        for quality_kind in ("geometry_coverage", "tracking_persistence", "pose_success", "valid_delta2_s"):
            for role in ("real", "fake"):
                target["quality"][kind][quality_kind][role].append(sides[role].get("quality", {}).get(quality_kind))
        for key in ORDER_NAMES:
            if scales[key]["details"] is None:
                continue
            real_median, _ = window_observation_median(sides["real"].get("features", {}), key)
            fake_median, _ = window_observation_median(sides["fake"].get("features", {}), key)
            if real_median is None or fake_median is None:
                continue
            center = np.asarray(scales[key]["details"]["center_median"], dtype=np.float64)
            scale = np.asarray(scales[key]["details"]["scale"], dtype=np.float64)
            distance = float(np.linalg.norm((fake_median - real_median) / scale))
            target["D"][key][kind].append(distance)
    output: list[dict[str, Any]] = []
    for target in values_by_pair.values():
        row = {key: value for key, value in target.items() if key not in ("D", "quality")}
        for key in ORDER_NAMES:
            dm = _finite(target["D"][key]["MANIP"])
            dc = _finite(target["D"][key]["CTRL"])
            row[f"Dmanip_{key}"] = float(np.median(dm)) if dm.size else None
            row[f"Dctrl_{key}"] = float(np.median(dc)) if dc.size else None
            row[f"G_{key}"] = float(np.median(dm) - np.median(dc)) if dm.size and dc.size else None
        for kind in ("MANIP", "CTRL"):
            for metric in ("geometry_coverage", "tracking_persistence", "pose_success", "valid_delta2_s"):
                real = _finite(target["quality"][kind][metric]["real"])
                fake = _finite(target["quality"][kind][metric]["fake"])
                row[f"quality_gap_{kind}_{metric}"] = float(np.median(fake) - np.median(real)) if real.size and fake.size else None
        output.append(row)
    return output, scales


def _pairwise_rows(selected: list[dict[str, Any]], results: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    metrics = ("temporal_mad", "temporal_iqr", "first_difference_magnitude", "second_difference_magnitude")
    by_key: dict[tuple[str, str, str, float], dict[str, dict[str, Any]]] = defaultdict(dict)
    for result in results:
        if result.get("status") != "COMPLETE":
            continue
        window = result["window"]
        by_key[(window["pair_id"], window["kind"], window["label"], float(window["anchor_fraction"]))][window["role"]] = result
    values: dict[str, dict[str, Any]] = {pair["pair_id"]: {"pair_id": pair["pair_id"], "source_id": pair["source_id"], "generator": pair["generator"], "manipulation_operation": pair["manipulation_operation"], "D": {metric: {"MANIP": [], "CTRL": []} for metric in metrics}} for pair in selected}
    for (pair_id, kind, label, anchor), sides in by_key.items():
        if pair_id not in values or "real" not in sides or "fake" not in sides:
            continue
        for metric in metrics:
            def metric_value(result: dict[str, Any]) -> float | None:
                pairwise = result.get("features", {}).get("pairwise", {})
                value = pairwise.get(metric) if metric in ("temporal_mad", "temporal_iqr") else pairwise.get(metric, {}).get("median")
                try:
                    value = float(value)
                except (TypeError, ValueError):
                    return None
                return value if np.isfinite(value) else None
            real, fake = metric_value(sides["real"]), metric_value(sides["fake"])
            if real is not None and fake is not None:
                values[pair_id]["D"][metric][kind].append(abs(fake - real))
    rows: list[dict[str, Any]] = []
    summaries: dict[str, Any] = {}
    for value in values.values():
        row = {key: item for key, item in value.items() if key != "D"}
        for metric in metrics:
            dm, dc = _finite(value["D"][metric]["MANIP"]), _finite(value["D"][metric]["CTRL"])
            row[f"Dmanip_{metric}"] = float(np.median(dm)) if dm.size else None
            row[f"Dctrl_{metric}"] = float(np.median(dc)) if dc.size else None
            row[f"G_{metric}"] = float(np.median(dm) - np.median(dc)) if dm.size and dc.size else None
        rows.append(row)
    for metric in metrics:
        gaps = [row[f"G_{metric}"] for row in rows]
        summaries[metric] = {"D_manip": finite_summary(row[f"Dmanip_{metric}"] for row in rows), "D_ctrl": finite_summary(row[f"Dctrl_{metric}"] for row in rows), "G": {**finite_summary(gaps), "positive_fraction": _positive_fraction(gaps), "bootstrap_median": _bootstrap_median(gaps), "sign_test": _sign_test(gaps)}}
    return rows, summaries


def _summary_for(pair_rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
    manip = [row[f"Dmanip_{key}"] for row in pair_rows]
    control = [row[f"Dctrl_{key}"] for row in pair_rows]
    gap = [row[f"G_{key}"] for row in pair_rows]
    return {
        "D_manip": finite_summary(manip),
        "D_ctrl": finite_summary(control),
        "G": {**finite_summary(gap), "positive_fraction": _positive_fraction(gap), "sign_test": _sign_test(gap), "paired_rank_biserial": (lambda s: (s["positive"] - s["negative"]) / s["N_nonzero"] if s["N_nonzero"] else None)(_sign_test(gap)), "bootstrap_median": _bootstrap_median(gap)},
        "exploratory_auroc_Dmanip_vs_Dctrl": _auroc(manip, control),
    }


def analyze(output: Path) -> dict[str, Any]:
    selected = json.loads((output / "manifests/selected_pairs.json").read_text(encoding="utf-8"))
    summary_path = output / "run_summary.json"
    prior = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
    results_path = output / "frontend/window_results.json"
    if not results_path.exists():
        raise FileNotFoundError(results_path)
    results = json.loads(results_path.read_text(encoding="utf-8"))
    if prior.get("status") == "FRONTEND_FEASIBILITY_BLOCKED_ON_NEW_CORE_DATA":
        return {**prior, "analysis_status": "FRONTEND_FEASIBILITY_BLOCKED_ON_NEW_CORE_DATA"}
    quality = _quality_summary(results)
    _write_json(output / "metrics/frontend_quality.json", quality)
    window_rows, _ = _window_rows(results)
    _write_csv(output / "metrics/per_window_signal.csv", window_rows)
    pair_rows, scales = _pair_rows(selected, results)
    _write_csv(output / "metrics/per_pair_signal.csv", pair_rows)
    summaries = {key: _summary_for(pair_rows, key) for key in ORDER_NAMES}
    a21 = [row.get("G_K2_delta2_s") - row.get("G_K1_delta_s") for row in pair_rows if row.get("G_K2_delta2_s") is not None and row.get("G_K1_delta_s") is not None]
    a20 = [row.get("G_K2_delta2_s") - row.get("G_K0_S") for row in pair_rows if row.get("G_K2_delta2_s") is not None and row.get("G_K0_S") is not None]
    order_summary = {"bootstrap_seed": BOOTSTRAP_SEED, "bootstrap_replicates": BOOTSTRAP_REPLICATES, "fake_count_in_fitting": 0, "normality_model": "NONE", "scales": scales, "orders": summaries, "A21": {**finite_summary(a21), "bootstrap_median": _bootstrap_median(a21), "positive_fraction": _positive_fraction(a21), "sign_test": _sign_test(a21)}, "A20": {**finite_summary(a20), "bootstrap_median": _bootstrap_median(a20), "positive_fraction": _positive_fraction(a20), "sign_test": _sign_test(a20)}}
    _write_json(output / "metrics/structural_order_summary.json", order_summary)
    pairwise = [row for row in window_rows if row.get("pairwise_distance_count", 0)]
    pairwise_rows, pairwise_summary = _pairwise_rows(selected, results)
    _write_json(output / "metrics/pairwise_diagnostic.json", {"N_windows": len(pairwise), "temporal_mad": finite_summary(row.get("pairwise_temporal_mad") for row in pairwise), "temporal_iqr": finite_summary(row.get("pairwise_temporal_iqr") for row in pairwise), "paired": pairwise_summary, "per_pair": pairwise_rows})
    raw_pairwise_signal = any(
        item["G"]["N"] >= 12
        and item["G"]["median"] is not None
        and item["G"]["median"] > 0
        and item["G"]["positive_fraction"] is not None
        and item["G"]["positive_fraction"] >= 0.70
        and item["G"]["bootstrap_median"]["ci95"][0] > 0
        for item in pairwise_summary.values()
    )
    confounds = {}
    for metric in ("geometry_coverage", "tracking_persistence", "pose_success", "valid_delta2_s"):
        confounds[metric] = _spearman([row.get("G_K2_delta2_s") for row in pair_rows], [row.get("quality_gap_MANIP_" + metric) for row in pair_rows])
    group_summary: dict[str, Any] = {}
    for row in pair_rows:
        group = f"{row['generator']}::{row['manipulation_operation']}"
        group_summary.setdefault(group, []).append(row.get("G_K2_delta2_s"))
    groups = {key: {"N": int(np.sum(np.isfinite(_finite(values)))), "G_K2": finite_summary(values)} for key, values in sorted(group_summary.items())}
    _write_json(output / "metrics/frontend_quality.json", {"window_quality": quality, "quality_gap_spearman_with_G2": confounds})
    status = "PILOT_INCONCLUSIVE"
    g2 = summaries["K2_delta2_s"]["G"]
    a21_summary = order_summary["A21"]
    if g2["N"] >= 12 and g2["bootstrap_median"]["ci95"] and g2["bootstrap_median"]["ci95"][0] > 0:
        status = "PAIRED_SECOND_ORDER_SIGNAL_PRESENT"
        if a21_summary["N"] >= 12 and a21_summary["bootstrap_median"]["ci95"] and a21_summary["bootstrap_median"]["ci95"][0] > 0:
            status = "PAIRED_SECOND_ORDER_ADVANTAGE_PRESENT"
    elif raw_pairwise_signal:
        status = "REPRESENTATION_REVISION_NEEDED_BEFORE_FORMAL_TRAINING"
    elif g2["N"] >= 12:
        status = "PAIRED_SECOND_ORDER_SIGNAL_NOT_SUPPORTED_IN_CURRENT_BASELINE"
    activity: dict[str, Any] = {}
    for key in ORDER_NAMES:
        activity[key] = {}
        for role in ("real", "fake"):
            for kind in ("MANIP", "CTRL"):
                activity[key][f"{role}_{kind}"] = finite_summary(row.get(f"{key}_norm") for row in window_rows if row.get("role") == role and row.get("kind") == kind)
    final = {**prior, "status": status, "analysis_status": status, "complete_window_results": len([row for row in results if row.get("status") == "COMPLETE"]), "failed_window_results": len([row for row in results if row.get("status") != "COMPLETE"]), "metrics": {"per_window": "metrics/per_window_signal.csv", "per_pair": "metrics/per_pair_signal.csv", "structural_orders": "metrics/structural_order_summary.json", "frontend_quality": "metrics/frontend_quality.json", "pairwise": "metrics/pairwise_diagnostic.json"}, "raw_activity": activity, "generator_operation": groups, "quality_gap_spearman_with_G2": confounds, "raw_pairwise_signal": raw_pairwise_signal, "normality_model_fitting": "NONE", "fake_count_in_fitting": 0, "selection_model_result_independent": True}
    _write_json(summary_path, final)
    return final


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(analyze(args.output), indent=2, sort_keys=True, ensure_ascii=False))


if __name__ == "__main__":
    main()
