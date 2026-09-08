"""Real-only LOSO calibration diagnostic for the frozen NSI artifact.

This module deliberately keeps the diagnostic scalar: the only condition is
the lagged first-order activity ``C = ||v_minus|| / sqrt(M)``.  No detector is
fit and no label is consulted while producing the reference ECDFs.
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

from research_tools.v7.paired_signal.analyze import (
    _auroc,
    _spearman,
    finite_summary,
)
from research_tools.v7.normalized_innovation.normalized_innovation import signed_summary

from .activity_condition import (
    activity_bin,
    anomaly_transform,
    ks_distance_to_uniform,
    median_absolute_deviation,
    tertile_cutpoints,
)
from .source_balanced_ecdf import source_balanced_ecdf, source_balanced_support


PREVIOUS_ROOT = Path("/root/autodl-tmp/data/sparse_3d_forgery_detection/derived/v7_activityforensics_paired_second_order_pilot_v1")
OUTPUT_ROOT = Path("/root/autodl-tmp/data/sparse_3d_forgery_detection/derived/v7_activityforensics_conditional_nsi_pilot_v1")
EXPECTED_SOURCES = 16
EXPECTED_WINDOWS = 192
WEAK_SUPPORT_THRESHOLD = 5


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False, ensure_ascii=False) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _read_rows(path: Path) -> list[dict[str, Any]]:
    numeric_fields = {
        "I_stored",
        "I_reconstructed",
        "abs_error",
        "v_minus_norm",
        "v_plus_norm",
        "M",
        "C",
        "h0_s",
        "h1_s",
        "anchor_fraction",
        "center_index",
        "component_index",
        "frame_index_prev",
        "frame_index",
        "frame_index_next",
        "timestamp_prev",
        "timestamp",
        "timestamp_next",
    }
    rows: list[dict[str, Any]] = []
    with path.open(newline="", encoding="utf-8") as handle:
        for raw in csv.DictReader(handle):
            row: dict[str, Any] = dict(raw)
            for key in numeric_fields:
                value = row.get(key)
                if value in (None, ""):
                    row[key] = None
                else:
                    row[key] = float(value)
            rows.append(row)
    return rows


def _finite(values: Iterable[float]) -> np.ndarray:
    array = np.asarray(list(values), dtype=np.float64)
    return array[np.isfinite(array)]


def _fraction(values: Iterable[float], predicate: Any) -> float | None:
    array = _finite(values)
    return float(np.mean(predicate(array))) if array.size else None


def _signed_with_fractions(values: Iterable[float]) -> dict[str, Any]:
    array = _finite(values)
    result = signed_summary(array)
    result["fraction_negative"] = float(np.mean(array < 0)) if array.size else None
    result["fraction_positive"] = float(np.mean(array > 0)) if array.size else None
    return result


def _source_values(rows: Iterable[Mapping[str, Any]], value_key: str) -> dict[str, list[float]]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        value = row.get(value_key)
        try:
            value_float = float(value)
        except (TypeError, ValueError):
            continue
        if np.isfinite(value_float):
            grouped[str(row["source_id"])].append(value_float)
    return dict(grouped)


def _copy_frozen_manifests(output: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    selected_path = PREVIOUS_ROOT / "manifests/selected_pairs.json"
    windows_path = PREVIOUS_ROOT / "manifests/window_manifest.json"
    selected = json.loads(selected_path.read_text(encoding="utf-8"))
    windows = json.loads(windows_path.read_text(encoding="utf-8"))
    if len(selected) != EXPECTED_SOURCES or len(windows) != EXPECTED_WINDOWS:
        raise RuntimeError("CONDITIONAL_NSI_ARTIFACT_INSUFFICIENT: frozen population/window count mismatch")
    # These are small JSON provenance copies, not new media or particle data.
    _write_json(output / "manifests/frozen_population.json", selected)
    _write_json(output / "manifests/frozen_windows.json", windows)
    return selected, windows


def _real_dependence(rows: list[dict[str, Any]], sources: list[str], output: Path) -> dict[str, Any]:
    per_source: list[dict[str, Any]] = []
    for source in sources:
        real = [row for row in rows if str(row["source_id"]) == source and row["role"] == "real"]
        rho = _spearman([row.get("I_stored") for row in real], [row.get("C") for row in real])
        per_source.append({"source_id": source, "N_real_triplets": len(real), "rho_I_C": rho})
    _write_csv(output / "real_condition/per_source_i_c_dependence.csv", per_source)
    rho_values = [row["rho_I_C"] for row in per_source if row["rho_I_C"] is not None]
    return {
        "N_source": len(per_source),
        "rho_summary": finite_summary(rho_values),
        "positive_fraction": _fraction(rho_values, lambda x: x > 0),
        "negative_fraction": _fraction(rho_values, lambda x: x < 0),
        "per_source": per_source,
    }


def _score_loso(
    rows: list[dict[str, Any]],
    sources: list[str],
    output: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Score held-out source rows using real-only source-balanced references."""

    scored: list[dict[str, Any]] = []
    support_rows: list[dict[str, Any]] = []
    fold_rows: list[dict[str, Any]] = []
    bin_distribution_rows: list[dict[str, Any]] = []
    for held_out in sources:
        training_real = [row for row in rows if row["role"] == "real" and str(row["source_id"]) != held_out]
        train_cuts = tertile_cutpoints([row["C"] for row in training_real])
        training_source_ids = sorted({str(row["source_id"]) for row in training_real})
        refs0 = _source_values(training_real, "I_stored")
        refs1: dict[str, dict[str, list[float]]] = {"LOW": {}, "MID": {}, "HIGH": {}}
        for source in training_source_ids:
            source_rows = [row for row in training_real if str(row["source_id"]) == source]
            for bin_name in refs1:
                refs1[bin_name][source] = [
                    float(row["I_stored"])
                    for row in source_rows
                    if activity_bin(float(row["C"]), train_cuts) == bin_name and np.isfinite(float(row["I_stored"]))
                ]
                if not refs1[bin_name][source]:
                    del refs1[bin_name][source]
        for bin_name in ("LOW", "MID", "HIGH"):
            support = source_balanced_support(refs1[bin_name])
            support_rows.append(
                {
                    "held_out_source": held_out,
                    "activity_bin": bin_name,
                    "training_source_count": support["distinct_source_count"],
                    "training_observation_count": support["observation_count"],
                    "effective_n": support["effective_n"],
                    "weak_support": bool(support["distinct_source_count"] < WEAK_SUPPORT_THRESHOLD),
                }
            )
            bin_values = [
                float(row["I_stored"])
                for row in training_real
                if activity_bin(float(row["C"]), train_cuts) == bin_name and np.isfinite(float(row["I_stored"]))
            ]
            distribution = finite_summary(bin_values)
            distribution.update(
                {
                    "held_out_source": held_out,
                    "activity_bin": bin_name,
                    "training_source_count": support["distinct_source_count"],
                    "effective_n": support["effective_n"],
                    "weak_support": bool(support["distinct_source_count"] < WEAK_SUPPORT_THRESHOLD),
                }
            )
            bin_distribution_rows.append(distribution)
        for row in rows:
            if str(row["source_id"]) != held_out:
                continue
            scored_row = dict(row)
            scored_row["held_out_source"] = held_out
            scored_row["activity_bin"] = activity_bin(float(row["C"]), train_cuts)
            scored_row["cut_low"] = train_cuts[0]
            scored_row["cut_high"] = train_cuts[1]
            q0, support0 = source_balanced_ecdf(float(row["I_stored"]), refs0)
            q1, support1 = source_balanced_ecdf(float(row["I_stored"]), refs1[scored_row["activity_bin"]])
            scored_row["q_U0"] = q0
            scored_row["q_U1"] = q1
            scored_row["A_U0"] = anomaly_transform(q0)
            scored_row["A_U1"] = anomaly_transform(q1)
            scored_row["U0_reference_source_count"] = support0["distinct_source_count"]
            scored_row["U1_reference_source_count"] = support1["distinct_source_count"]
            scored_row["U0_reference_effective_n"] = support0["effective_n"]
            scored_row["U1_reference_effective_n"] = support1["effective_n"]
            scored.append(scored_row)
        fold_rows.append(
            {
                "held_out_source": held_out,
                "training_source_count": len(training_source_ids),
                "training_source_ids": training_source_ids,
                "training_real_triplet_count": len(training_real),
                "cut_low": train_cuts[0],
                "cut_high": train_cuts[1],
                "fake_calibration_count": 0,
                "heldout_real_calibration_count": 0,
            }
        )
    scored.sort(key=lambda row: (str(row["window_id"]), int(row["component_index"]), int(row["center_index"])))
    _write_csv(output / "real_condition/conditional_support.csv", support_rows)
    _write_csv(output / "real_condition/i_by_activity_bin.csv", bin_distribution_rows)
    _write_json(
        output / "real_condition/loso_activity_bins.json",
        {
            "folds": fold_rows,
            "cutpoint_source": "training_real_only_other_15_sources",
            "fake_calibration_count": 0,
            "heldout_real_calibration_count": 0,
            "bin_names": ["LOW", "MID", "HIGH"],
        },
    )
    return scored, support_rows, fold_rows, bin_distribution_rows


def aggregate_window_scores(rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Apply frozen component-Q90 then equal-component window aggregation."""

    grouped: dict[tuple[str, str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["held_out_source"]), str(row["window_id"]), str(row["component_index"]))].append(row)
    component_scores: dict[tuple[str, str], dict[str, list[float]]] = defaultdict(lambda: {"U0": [], "U1": []})
    component_meta: dict[tuple[str, str], Mapping[str, Any]] = {}
    for (held_source, window_id, component_index), component_rows in grouped.items():
        component_meta[(held_source, window_id)] = component_rows[0]
        for key in ("U0", "U1"):
            values = _finite(float(row[f"A_{key}"]) for row in component_rows)
            if values.size:
                component_scores[(held_source, window_id)][key].append(float(np.percentile(values, 90)))
    output: list[dict[str, Any]] = []
    for key in sorted(component_scores):
        held_source, window_id = key
        meta = component_meta[key]
        values = component_scores[key]
        output.append(
            {
                "held_out_source": held_source,
                "source_id": str(meta["source_id"]),
                "role": str(meta["role"]),
                "kind": str(meta["kind"]),
                "pair_id": str(meta["pair_id"]),
                "label": str(meta["label"]),
                "anchor_fraction": float(meta["anchor_fraction"]),
                "window_id": window_id,
                "A_U0": float(np.median(values["U0"])) if values["U0"] else None,
                "A_U1": float(np.median(values["U1"])) if values["U1"] else None,
                "component_q90_count": len(values["U0"]),
                "triplet_count": len(grouped[(held_source, window_id, str(meta["component_index"]))]),
            }
        )
    return output


def _real_calibration(scored: list[dict[str, Any]], window_rows: list[dict[str, Any]], sources: list[str], output: Path) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for source in sources:
        triplets = [row for row in scored if str(row["source_id"]) == source and row["role"] == "real"]
        q0 = [row["q_U0"] for row in triplets]
        q1 = [row["q_U1"] for row in triplets]
        ks0 = ks_distance_to_uniform(q0)
        ks1 = ks_distance_to_uniform(q1)
        rho0 = _spearman(q0, [row["C"] for row in triplets])
        rho1 = _spearman(q1, [row["C"] for row in triplets])
        source_windows = [row for row in window_rows if str(row["source_id"]) == source and row["role"] == "real"]
        a0 = [row["A_U0"] for row in source_windows]
        a1 = [row["A_U1"] for row in source_windows]
        rows.append(
            {
                "source_id": source,
                "N_real_triplets": len(triplets),
                "N_real_windows": len(source_windows),
                "KS_U0": ks0,
                "KS_U1": ks1,
                "delta_KS": (ks1 - ks0) if ks0 is not None and ks1 is not None else None,
                "rho_U0_C": rho0,
                "rho_U1_C": rho1,
                "delta_rho": (abs(rho1) - abs(rho0)) if rho0 is not None and rho1 is not None else None,
                "real_window_median_A_U0": float(np.median(_finite(a0))) if _finite(a0).size else None,
                "real_window_median_A_U1": float(np.median(_finite(a1))) if _finite(a1).size else None,
            }
        )
    _write_csv(output / "calibration/per_source_real_calibration.csv", rows)
    delta_ks = [row["delta_KS"] for row in rows]
    delta_rho = [row["delta_rho"] for row in rows]
    dispersion_u0 = median_absolute_deviation(row["real_window_median_A_U0"] for row in rows)
    dispersion_u1 = median_absolute_deviation(row["real_window_median_A_U1"] for row in rows)
    result = {
        "per_source": rows,
        "uniformity": {
            "KS_U0": finite_summary(row["KS_U0"] for row in rows),
            "KS_U1": finite_summary(row["KS_U1"] for row in rows),
            "delta_KS": _signed_with_fractions(delta_ks),
            "fraction_improved": _fraction(delta_ks, lambda x: x < 0),
        },
        "residual_activity": {
            "rho_U0_C": finite_summary(row["rho_U0_C"] for row in rows),
            "rho_U1_C": finite_summary(row["rho_U1_C"] for row in rows),
            "delta_rho": _signed_with_fractions(delta_rho),
            "fraction_reduced": _fraction(delta_rho, lambda x: x < 0),
        },
        "real_source_score_dispersion": {
            "MAD_U0": dispersion_u0,
            "MAD_U1": dispersion_u1,
            "delta_MAD": (dispersion_u1 - dispersion_u0) if dispersion_u0 is not None and dispersion_u1 is not None else None,
        },
        "bootstrap_seed": 20260909,
        "bootstrap_replicates": 10000,
    }
    _write_json(output / "calibration/real_uniformity_summary.json", result)
    _write_json(output / "calibration/real_residual_activity.json", result["residual_activity"])
    _write_json(output / "calibration/real_source_score_dispersion.json", result["real_source_score_dispersion"])
    return result


def _fake_evaluation(window_rows: list[dict[str, Any]], sources: list[str], output: Path) -> dict[str, Any]:
    per_source: list[dict[str, Any]] = []
    for source in sources:
        groups: dict[tuple[str, str], list[float]] = defaultdict(list)
        for row in window_rows:
            if str(row["source_id"]) == source and np.isfinite(float(row["A_U0"])) and np.isfinite(float(row["A_U1"])):
                groups[(str(row["role"]), str(row["kind"]))].append(float(row["A_U0"]))
        groups_u1: dict[tuple[str, str], list[float]] = defaultdict(list)
        for row in window_rows:
            if str(row["source_id"]) == source and np.isfinite(float(row["A_U0"])) and np.isfinite(float(row["A_U1"])):
                groups_u1[(str(row["role"]), str(row["kind"]))].append(float(row["A_U1"]))
        def median_group(group: Mapping[tuple[str, str], list[float]], role: str, kind: str) -> float | None:
            values = _finite(group.get((role, kind), []))
            return float(np.median(values)) if values.size else None
        rm0, fm0, rc0, fc0 = (median_group(groups, "real", "MANIP"), median_group(groups, "fake", "MANIP"), median_group(groups, "real", "CTRL"), median_group(groups, "fake", "CTRL"))
        rm1, fm1, rc1, fc1 = (median_group(groups_u1, "real", "MANIP"), median_group(groups_u1, "fake", "MANIP"), median_group(groups_u1, "real", "CTRL"), median_group(groups_u1, "fake", "CTRL"))
        j0 = (fm0 - rm0) - (fc0 - rc0) if None not in (rm0, fm0, rc0, fc0) else None
        j1 = (fm1 - rm1) - (fc1 - rc1) if None not in (rm1, fm1, rc1, fc1) else None
        per_source.append(
            {
                "source_id": source,
                "A_RM_U0": rm0,
                "A_FM_U0": fm0,
                "A_RC_U0": rc0,
                "A_FC_U0": fc0,
                "J_U0": j0,
                "A_RM_U1": rm1,
                "A_FM_U1": fm1,
                "A_RC_U1": rc1,
                "A_FC_U1": fc1,
                "J_U1": j1,
                "delta_J": (j1 - j0) if j0 is not None and j1 is not None else None,
            }
        )
    _write_csv(output / "evaluation/per_source_j.csv", per_source)
    j0_values = [row["J_U0"] for row in per_source]
    j1_values = [row["J_U1"] for row in per_source]
    delta_values = [row["delta_J"] for row in per_source]
    fake_manip_u0 = [row["A_U0"] for row in window_rows if row["role"] == "fake" and row["kind"] == "MANIP"]
    real_manip_u0 = [row["A_U0"] for row in window_rows if row["role"] == "real" and row["kind"] == "MANIP"]
    all_real_u0 = [row["A_U0"] for row in window_rows if row["role"] == "real"]
    fake_manip_u1 = [row["A_U1"] for row in window_rows if row["role"] == "fake" and row["kind"] == "MANIP"]
    real_manip_u1 = [row["A_U1"] for row in window_rows if row["role"] == "real" and row["kind"] == "MANIP"]
    all_real_u1 = [row["A_U1"] for row in window_rows if row["role"] == "real"]
    result = {
        "per_source": per_source,
        "J_U0": _signed_with_fractions(j0_values),
        "J_U1": _signed_with_fractions(j1_values),
        "delta_J": _signed_with_fractions(delta_values),
        "exploratory_auroc_fake_manip_vs_real_manip": {"U0": _auroc(fake_manip_u0, real_manip_u0), "U1": _auroc(fake_manip_u1, real_manip_u1)},
        "exploratory_auroc_fake_manip_vs_all_heldout_real": {"U0": _auroc(fake_manip_u0, all_real_u0), "U1": _auroc(fake_manip_u1, all_real_u1)},
        "scope": "DEVELOPMENT_DIAGNOSTIC_ONLY; NOT_FULL_VIDEO; NOT_SEALED_TEST",
        "fake_fitting_count": 0,
    }
    _write_json(output / "evaluation/conditional_vs_unconditional.json", result)
    return result


def analyze(output: Path = OUTPUT_ROOT) -> dict[str, Any]:
    reconstruction_summary = json.loads((output / "reconstruction/reconstruction_summary.json").read_text(encoding="utf-8"))
    if reconstruction_summary.get("status") != "NSI_RECONSTRUCTION_VERIFIED":
        raise RuntimeError("NSI_RECONSTRUCTION_MISMATCH")
    rows = _read_rows(output / "reconstruction/per_triplet_reconstruction.csv")
    if len(rows) != 9101:
        raise RuntimeError("CONDITIONAL_NSI_ARTIFACT_INSUFFICIENT: expected 9101 reconstructed triplets")
    selected, windows = _copy_frozen_manifests(output)
    sources = sorted(str(item["source_id"]) for item in selected)
    dependence = _real_dependence(rows, sources, output)
    scored, support_rows, fold_rows, _ = _score_loso(rows, sources, output)
    _write_csv(output / "metrics/per_triplet_nsi_with_condition.csv", scored)
    window_rows = aggregate_window_scores(scored)
    _write_csv(output / "evaluation/per_window_scores.csv", window_rows)
    calibration = _real_calibration(scored, window_rows, sources, output)
    evaluation = _fake_evaluation(window_rows, sources, output)
    weak_count = int(sum(bool(row["weak_support"]) for row in support_rows))
    delta_ks = _finite(row["delta_KS"] for row in calibration["per_source"])
    delta_rho = _finite(row["delta_rho"] for row in calibration["per_source"])
    j1 = _finite(row["J_U1"] for row in evaluation["per_source"])
    delta_j = _finite(row["delta_J"] for row in evaluation["per_source"])
    condition_values = _finite(row["C"] for row in rows)
    condition_summary = finite_summary(condition_values)
    condition_summary["min"] = float(np.min(condition_values)) if condition_values.size else None
    condition_summary["max"] = float(np.max(condition_values)) if condition_values.size else None
    real_calibration_gate = bool(delta_ks.size) and float(np.median(delta_ks)) < 0 and float(np.mean(delta_ks < 0)) >= 0.70
    activity_gate = bool(delta_rho.size) and float(np.median(delta_rho)) < 0 and float(np.mean(delta_rho < 0)) >= 0.70
    fake_gate = bool(j1.size) and float(np.median(j1)) > 0 and float(np.mean(j1 > 0)) >= 0.70 and bool(delta_j.size) and float(np.median(delta_j)) > 0
    rho_values = _finite(item["rho_I_C"] for item in dependence["per_source"])
    lagged_dependence = bool(rho_values.size) and float(np.median(np.abs(rho_values))) >= 0.30 and float(np.mean(rho_values > 0)) >= 0.70
    if weak_count > len(support_rows) / 2:
        status = "CONDITIONAL_SUPPORT_INSUFFICIENT"
    elif real_calibration_gate and activity_gate and fake_gate:
        status = "MINIMAL_CONDITIONAL_NSI_SUPPORTED"
    elif real_calibration_gate and activity_gate and not fake_gate:
        status = "REAL_CALIBRATION_IMPROVED_BUT_NSI_NONDISCRIMINATIVE"
    elif not real_calibration_gate and not activity_gate and not fake_gate and not lagged_dependence:
        status = "LAGGED_ACTIVITY_CONDITION_NOT_SUPPORTED"
    elif fake_gate and not (real_calibration_gate and activity_gate):
        status = "CONDITIONAL_EFFECT_UNSTABLE"
    else:
        status = "PILOT_INCONCLUSIVE"
    summary = {
        "status": status,
        "reconstruction_status": reconstruction_summary["status"],
        "selected_source_pairs": len(selected),
        "frozen_windows": len(windows),
        "reconstructed_triplets": len(rows),
        "condition": "C = ||v_minus|| / sqrt(M)",
        "condition_uses": ["v_minus_norm", "M"],
        "condition_summary": condition_summary,
        "condition_excludes": ["v_plus", "speed_sum", "future_average", "B0", "tracking", "geometry", "scene", "action", "semantic", "generator", "operation"],
        "real_only_i_c_dependence": dependence,
        "conditional_support": {
            "records": len(support_rows),
            "weak_records": weak_count,
            "weak_support_threshold_distinct_training_sources": WEAK_SUPPORT_THRESHOLD,
        },
        "heldout_calibration": calibration,
        "fake_evaluation": evaluation,
        "loso": {
            "fold_count": len(fold_rows),
            "fake_calibration_count": 0,
            "heldout_real_calibration_count": 0,
            "reference": "other real sources only; source-balanced ECDF",
        },
        "frozen_population": True,
        "annotation_selected_windows": True,
        "frontend_rerun": False,
        "gpu_frontend": False,
        "formal_src_modified": False,
        "not_full_video": True,
        "not_sealed_test": True,
        "normality_model": "NONE",
        "metrics": {
            "reconstruction": "reconstruction/per_triplet_reconstruction.csv",
            "per_triplet": "metrics/per_triplet_nsi_with_condition.csv",
            "per_window": "evaluation/per_window_scores.csv",
            "support": "real_condition/conditional_support.csv",
            "real_calibration": "calibration/per_source_real_calibration.csv",
            "fake_evaluation": "evaluation/per_source_j.csv",
        },
    }
    _write_json(output / "run_summary.json", summary)
    return summary


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUTPUT_ROOT)
    args = parser.parse_args()
    print(json.dumps(analyze(args.output), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
