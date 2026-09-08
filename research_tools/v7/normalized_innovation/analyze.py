"""Analyze the frozen NSI pilot without fitting a detector."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from research_tools.v7.paired_signal.analyze import _auroc, _spearman, finite_summary

from .normalized_innovation import (
    activity_correlations,
    cross_source_summary,
    leave_one_source_out,
    metric_summary,
    paired_nsi_gain,
    pair_dimension_diagnostics,
    signed_summary,
)


OUTPUT_ROOT = Path("/root/autodl-tmp/data/sparse_3d_forgery_detection/derived/v7_activityforensics_normalized_structural_innovation_pilot_v1")
PREVIOUS_ROOT = Path("/root/autodl-tmp/data/sparse_3d_forgery_detection/derived/v7_activityforensics_paired_second_order_pilot_v1")
RELATION_ROOT = Path("/root/autodl-tmp/data/sparse_3d_forgery_detection/derived/v7_activityforensics_relation_first_pilot_v1")


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
    rows: list[dict[str, Any]] = []
    with path.open(newline="", encoding="utf-8") as handle:
        for raw in csv.DictReader(handle):
            row = dict(raw)
            for key, value in list(row.items()):
                if key in {"window_id", "pair_id", "source_id", "role", "kind", "label", "source_sha256", "particle_artifact"}:
                    continue
                if value in ("", "None"):
                    row[key] = None
                else:
                    try:
                        row[key] = float(value)
                    except ValueError:
                        pass
            row["anchor_fraction"] = float(row["anchor_fraction"])
            rows.append(row)
    return rows


def _raw_distributions(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for role in ("real", "fake"):
        for kind in ("MANIP", "CTRL"):
            values = [row[key] for row in rows if row["role"] == role and row["kind"] == kind and row.get(key) is not None]
            output[f"{role}_{kind.lower()}"] = metric_summary(values)
    return output


def _source_activity(rows: list[dict[str, Any]], key: str) -> dict[str, float | None]:
    grouped: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for row in rows:
        if row.get(key) is not None:
            grouped[(str(row["source_id"]), str(row["role"]), str(row["kind"]))].append(float(row[key]))
    return {"::".join(group): (float(np.median(values)) if values else None) for group, values in grouped.items()}


def _aligned_spearman(rows: list[dict[str, Any]], left_key: str, right_key: str) -> float | None:
    left: list[float] = []
    right: list[float] = []
    for row in rows:
        if row.get(left_key) is None or row.get(right_key) is None:
            continue
        left.append(float(row[left_key]))
        right.append(float(row[right_key]))
    return _spearman(left, right)


def _source_level_correlation(rows: list[dict[str, Any]], left_key: str, right_key: str) -> float | None:
    left = _source_activity(rows, left_key)
    right = _source_activity(rows, right_key)
    keys = sorted(set(left) & set(right))
    return _spearman([left[key] for key in keys if left[key] is not None and right[key] is not None], [right[key] for key in keys if left[key] is not None and right[key] is not None])


def _source_gap_correlations(rows: list[dict[str, Any]], paired_rows: list[dict[str, Any]], source_rows: list[dict[str, Any]]) -> dict[str, Any]:
    grouped = _source_activity(rows, "I_window")
    output: dict[str, Any] = {}
    for metric in ("speed_sum_median", "raw_delta_r_median", "raw_second_difference_median"):
        metric_values = _source_activity(rows, metric)
        i_gaps: list[float] = []
        metric_gaps: list[float] = []
        for pair in paired_rows:
            source = str(pair["source_id"])
            left = metric_values.get(f"{source}::fake::MANIP")
            right = metric_values.get(f"{source}::real::MANIP")
            if pair.get("G_I") is not None and left is not None and right is not None:
                i_gaps.append(float(pair["G_I"]))
                metric_gaps.append(float(left - right))
        output[f"G_I_vs_{metric}_manipulation_fake_real_gap"] = _spearman(i_gaps, metric_gaps)
    h_by_source = {str(row["source_id"]): row.get("H_I") for row in source_rows}
    for metric in ("speed_sum_median", "raw_delta_r_median", "raw_second_difference_median"):
        metric_values = _source_activity(rows, metric)
        h_values: list[float] = []
        metric_gaps = []
        for source, h_value in h_by_source.items():
            left = metric_values.get(f"{source}::fake::MANIP")
            right = metric_values.get(f"{source}::real::MANIP")
            if h_value is not None and left is not None and right is not None:
                h_values.append(float(h_value))
                metric_gaps.append(float(left - right))
        output[f"H_I_vs_{metric}_manipulation_fake_real_gap"] = _spearman(h_values, metric_gaps)
    return output


def analyze(output: Path = OUTPUT_ROOT) -> dict[str, Any]:
    rows = _read_rows(output / "metrics/per_window_nsi.csv")
    if len(rows) != 192:
        raise RuntimeError("NSI_ARTIFACT_INSUFFICIENT: expected 192 window rows")
    selected = json.loads((PREVIOUS_ROOT / "manifests/selected_pairs.json").read_text(encoding="utf-8"))
    paired_rows = paired_nsi_gain(rows, selected)
    _write_csv(output / "metrics/per_source_paired_gain.csv", paired_rows)
    gaps = [row["G_I"] for row in paired_rows if row.get("G_I") is not None]
    manip = [row["D_manip"] for row in paired_rows if row.get("D_manip") is not None]
    ctrl = [row["D_ctrl"] for row in paired_rows if row.get("D_ctrl") is not None]
    paired_summary = {
        "D_manip": metric_summary(manip),
        "D_ctrl": metric_summary(ctrl),
        "G_I": signed_summary(gaps),
        "exploratory_auroc_Dmanip_vs_Dctrl": _auroc(manip, ctrl),
        "N_source_pairs": len(paired_rows),
    }
    raw_i_distributions = _raw_distributions(rows, "I_window")
    calibrated, calibration_meta = leave_one_source_out(rows)
    _write_csv(output / "metrics/loso_real_calibration.csv", [{"source_id": key, **value} for key, value in calibration_meta.items()])
    source_rows, ruler_summary = cross_source_summary(calibrated)
    _write_csv(output / "metrics/per_source_ruler_gain.csv", source_rows)
    activity = activity_correlations(calibrated)
    dimensions = pair_dimension_diagnostics(calibrated, paired_rows, source_rows)
    confounds = {
        "activity_correlations": activity,
        "pair_dimension": dimensions,
        "source_level_activity": {
            "I_window": {metric: _source_level_correlation(rows, "I_window", metric) for metric in ("speed_sum_median", "raw_delta_r_median", "raw_second_difference_median")},
            "Z_I": {metric: _source_level_correlation(calibrated, "Z_I", metric) for metric in ("speed_sum_median", "raw_delta_r_median", "raw_second_difference_median")},
        },
        "source_level_gain_activity": _source_gap_correlations(rows, paired_rows, source_rows),
        "strong_threshold": 0.7,
    }
    _write_json(output / "metrics/confound_diagnostics.json", confounds)
    previous = json.loads((RELATION_ROOT / "metrics/representation_comparison.json").read_text(encoding="utf-8"))
    raw = json.loads((PREVIOUS_ROOT / "metrics/pairwise_diagnostic.json").read_text(encoding="utf-8"))["paired"]["second_difference_magnitude"]["G"]
    comparison = {
        "NSI": paired_summary,
        "raw_I_window": raw_i_distributions,
        "B0": previous["representations"]["B0"]["G"],
        "P2": previous["representations"]["P2"]["G"],
        "R2": previous["representations"]["R2"]["G"],
        "raw_pairwise_second_difference": raw,
        "same_frozen_population": True,
        "frontend_rerun": False,
    }
    _write_json(output / "metrics/comparison_summary.json", comparison)
    real_activity = ruler_summary["real_manip_minus_ctrl"]
    fake_activity = ruler_summary["fake_manip_minus_ctrl"]
    activity_flag = any(abs(value) >= 0.7 for section in activity.values() if isinstance(section, dict) for value in section.values() if isinstance(value, (int, float)))
    dimension_flag = bool(dimensions.get("flag"))
    paired_gate = bool(gaps) and float(np.median(gaps)) > 0 and float(np.mean(np.asarray(gaps) > 0)) >= 0.70
    paired_ci = paired_summary["G_I"]["bootstrap_median"]["ci95"]
    paired_gate = paired_gate and paired_ci is not None and paired_ci[0] >= 0
    h = [row["H_I"] for row in source_rows if row.get("H_I") is not None]
    h_gate = bool(h) and float(np.median(h)) > 0 and float(np.mean(np.asarray(h) > 0)) >= 0.70
    real_not_same_level = real_activity["median"] is None or fake_activity["median"] is None or real_activity["median"] < fake_activity["median"] or (real_activity["positive_fraction"] or 0) < 0.70
    h_gate = h_gate and real_not_same_level
    if len(gaps) < 12 or len(h) < 12:
        status = "PILOT_INCONCLUSIVE"
    elif activity_flag:
        status = "ACTIVITY_NORMALIZATION_FAILED"
    elif dimension_flag:
        status = "PAIR_DIMENSION_CONFOUND"
    elif paired_gate and h_gate:
        status = "NORMALIZED_STRUCTURAL_INNOVATION_SUPPORTED"
    elif paired_gate:
        status = "NSI_SENSITIVE_BUT_NOT_NORMALIZED"
    elif h_gate:
        status = "NSI_CROSS_SOURCE_CALIBRATION_PROMISING"
    else:
        status = "NSI_MECHANISM_NOT_SUPPORTED"
    summary = {
        "status": status,
        "secondary_tags": (["ACTIVITY_NORMALIZATION_FAILED"] if activity_flag and status != "ACTIVITY_NORMALIZATION_FAILED" else []) + (["PAIR_DIMENSION_CONFOUND"] if dimension_flag and status != "PAIR_DIMENSION_CONFOUND" else []),
        "selected_pairs": 16,
        "windows": len(rows),
        "triplets": sum(int(row.get("triplet_count") or 0) for row in rows),
        "frontend_rerun": False,
        "formal_src_modified": False,
        "fake_count_in_calibration": 0,
        "normality_model": "NONE",
        "paired": paired_summary,
        "raw_I_window": raw_i_distributions,
        "ruler": ruler_summary,
        "activity": activity,
        "pair_dimension": dimensions,
        "metrics": {
            "per_triplet": "metrics/per_triplet_nsi.csv",
            "per_window": "metrics/per_window_nsi.csv",
            "paired": "metrics/per_source_paired_gain.csv",
            "calibration": "metrics/loso_real_calibration.csv",
            "ruler": "metrics/per_source_ruler_gain.csv",
            "comparison": "metrics/comparison_summary.json",
            "confounds": "metrics/confound_diagnostics.json",
        },
        "calibration_meta": calibration_meta,
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
