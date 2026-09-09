"""Run the frozen B0 supervised source-disjoint linear probe."""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from research_tools.v7.b0_no_reference.b0_reconstruction import load_and_reproduce
from research_tools.v7.b0_no_reference.analyze_b0_no_reference import auroc, finite_summary

from .probe import (
    BOOTSTRAP_REPLICATES,
    BOOTSTRAP_SEED,
    MODEL_CONFIG,
    fit_fold,
    main_training_label,
    paired_source_bootstrap,
    primary_status,
    source_auc,
    source_group_medians,
    summary_for_deltas,
)


DATA_ROOT = Path("/root/autodl-tmp/data/sparse_3d_forgery_detection")
SOURCE_ARTIFACT_ROOT = DATA_ROOT / "derived/v7_activityforensics_paired_second_order_pilot_v1"
BASELINE_ARTIFACT_ROOT = DATA_ROOT / "derived/v7_activityforensics_b0_no_reference_pilot_v1"
OUTPUT_ROOT = DATA_ROOT / "derived/v7_activityforensics_b0_supervised_linear_probe_v1"


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _read_baseline(path: Path) -> dict[str, float]:
    values: dict[str, float] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            window_id = str(row["window_id"])
            if window_id in values:
                raise ValueError(f"duplicate baseline window: {window_id}")
            values[window_id] = float(row["score"])
    return values


def _load_samples(source_root: Path, baseline_root: Path) -> tuple[list[dict[str, Any]], list[str], dict[str, Any]]:
    selected = json.loads((source_root / "manifests/selected_pairs.json").read_text(encoding="utf-8"))
    frozen_windows = json.loads((source_root / "manifests/window_manifest.json").read_text(encoding="utf-8"))
    if len(selected) != 16 or len(frozen_windows) != 192:
        raise ValueError("frozen population is not 16 pairs and 192 windows")
    sources = sorted({str(row["source_id"]) for row in frozen_windows})
    if len(sources) != 16:
        raise ValueError("frozen population does not contain 16 source identities")
    _, windows, reproduction = load_and_reproduce(source_root)
    baseline = _read_baseline(baseline_root / "scores/per_window_b0_scores.csv")
    valid_windows = [row for row in windows if row["b0_median"] is not None]
    valid_ids = {str(row["window_id"]) for row in valid_windows}
    if set(baseline) != valid_ids:
        raise ValueError("baseline and supervised probe do not use identical valid windows")
    samples = []
    for row in valid_windows:
        samples.append({
            "window_id": str(row["window_id"]),
            "pair_id": str(row["pair_id"]),
            "source_id": str(row["source_id"]),
            "role": str(row["role"]),
            "kind": str(row["kind"]),
            "anchor_fraction": float(row["anchor_fraction"]),
            "b0": list(row["b0_median"]),
            "baseline_score": baseline[str(row["window_id"])],
        })
    return samples, sources, {
        "selected_pairs": len(selected),
        "frozen_windows": len(frozen_windows),
        "valid_windows": len(samples),
        "valid_sources": len({row["source_id"] for row in samples}),
        "invalid_sources": sorted(set(sources) - {row["source_id"] for row in samples}),
        "b0_reconstruction": reproduction,
        "baseline_artifact": str(baseline_root),
    }


def _protocol(population: dict[str, Any]) -> dict[str, Any]:
    return {
        "experiment": "V7 Frozen B0 Supervised Linear Probe",
        "input": "window-level S=[mean,std,p25,p75] using frozen B0 component/time median",
        "main_training_labels": {"real_MANIP": 0, "fake_MANIP": 1},
        "control_training": "excluded from main fitting; evaluated descriptively only",
        "label_source": "preselected frozen manipulation-window annotation; not a paired input",
        "source_disjoint": "leave-one-source-out; all held-out source windows excluded from standardization, weights, and fitting",
        "population": population,
        "missing": "windows without finite B0 are unavailable; source 0HV07 retained in coverage accounting; no fill or repair",
        "sample_weight": "each training source total equal; within source real/fake each half; within source/class windows equal; mean weight normalized to 1",
        "standardization": "weighted training real+fake mean/variance only; variance <=1e-12 uses fixed scale 1.0 and is recorded",
        "model": {**MODEL_CONFIG, "score": "decision_function; higher means more fake-like", "probability_claim": "none; sigmoid output is not calibrated probability"},
        "primary_comparison": "source-equal mean AUROC on fake MANIP vs real MANIP and paired difference from frozen B0 real-only scores",
        "bootstrap": {"seed": BOOTSTRAP_SEED, "replicates": BOOTSTRAP_REPLICATES, "unit": "complete source resampling; same indices for model and baseline"},
        "status_rules": {
            "blocked": "input, labels, or source isolation invalid",
            "supported": "model source-mean AUROC CI lower > 0.5 and model-baseline delta CI lower > 0",
            "signal_without_gain": "model source-mean AUROC CI lower > 0.5 but gain CI lower <= 0",
            "not_established": "otherwise",
        },
        "not_authorized": ["full-video", "sealed-test", "probability calibration", "hyperparameter search", "multi-order", "NSI", "dynamic fusion", "feature search", "frontend rerun"],
    }


def _attach_model_scores(samples: list[dict[str, Any]], sources: list[str]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    oof: list[dict[str, Any]] = []
    model_records: list[dict[str, Any]] = []
    support: list[dict[str, Any]] = []
    main_samples = [row for row in samples if main_training_label(row) is not None]
    for held_out in sources:
        training = [row for row in main_samples if row["source_id"] != held_out]
        held = [row for row in samples if row["source_id"] == held_out]
        if not held:
            model_records.append({"held_out_source": held_out, "status": "NO_HELDOUT_B0"})
            support.append({"held_out_source": held_out, "training_real_count": sum(main_training_label(row) == 0 for row in training), "training_fake_count": sum(main_training_label(row) == 1 for row in training), "held_out_window_count": 0, "status": "NO_HELDOUT_B0"})
            continue
        model = fit_fold(training, held_out)
        model_records.append(model.as_dict())
        support.append({"held_out_source": held_out, "training_real_count": model.training_real_count, "training_fake_count": model.training_fake_count, "training_source_count": model.training_source_count, "held_out_window_count": len(held), "sample_weight_mean": model.sample_weight_mean, "n_iter": model.n_iter, "convergence_status": model.convergence_status, "status": "SCORED"})
        scores = model.score([row["b0"] for row in held])
        for row, score in zip(held, scores):
            oof.append({**row, "label": main_training_label(row), "score": float(score), "fold_held_out_source": held_out})
    return oof, model_records, support


def _pooled_scores(rows: list[dict[str, Any]], score_key: str, *, positive_role: str = "fake", positive_kind: str = "MANIP", negative_kind: str | None = "MANIP") -> float | None:
    positive = [row[score_key] for row in rows if row.get("role") == positive_role and row.get("kind") == positive_kind and row.get(score_key) is not None]
    negative = [row[score_key] for row in rows if row.get("role") == "real" and (negative_kind is None or row.get("kind") == negative_kind) and row.get(score_key) is not None]
    return auroc(positive, negative)


def run(source_root: Path = SOURCE_ARTIFACT_ROOT, baseline_root: Path = BASELINE_ARTIFACT_ROOT, output_root: Path = OUTPUT_ROOT) -> dict[str, Any]:
    started = time.perf_counter()
    samples, sources, population = _load_samples(source_root, baseline_root)
    output_root.mkdir(parents=True, exist_ok=True)
    protocol = _protocol(population)
    _write_json(output_root / "protocol.json", protocol)
    _write_json(output_root / "manifests/frozen_population.json", {"source_ids": sources, **population, "main_training_windows": "real/fake MANIP only", "control_windows": "evaluation only"})
    _write_json(output_root / "manifests/source_artifact_links.json", {
        "source_artifact_root": str(source_root),
        "baseline_artifact_root": str(baseline_root),
        "selected_pairs": str(source_root / "manifests/selected_pairs.json"),
        "window_manifest": str(source_root / "manifests/window_manifest.json"),
        "frontend_window_results": str(source_root / "frontend/window_results.json"),
        "historical_b0_scores": str(baseline_root / "scores/per_window_b0_scores.csv"),
        "frontend_rerun": False,
        "large_inputs_copied": False,
    })
    _write_json(output_root / "manifests/frozen_windows.json", samples)

    oof, model_records, support = _attach_model_scores(samples, sources)
    if not model_records or len(oof) != len(samples):
        raise ValueError("B0_LINEAR_PROBE_BLOCKED: incomplete source-disjoint OOF scoring")
    _write_json(output_root / "models/per_fold_models.json", {"model_config": MODEL_CONFIG, "folds": model_records})
    _write_csv(output_root / "calibration/per_fold_support.csv", support)
    _write_csv(output_root / "scores/oof_window_scores.csv", [{key: value for key, value in row.items() if key != "b0"} | {f"b0_{index}": row["b0"][index] for index in range(4)} for row in oof])

    model_auc = source_auc(oof, "score")
    baseline_auc = source_auc(oof, "baseline_score")
    baseline_by_source = {row["source_id"]: row["auroc"] for row in baseline_auc}
    source_metrics: list[dict[str, Any]] = []
    for row in model_auc:
        source_metrics.append({**row, "model_auroc": row["auroc"], "baseline_auroc": baseline_by_source.get(row["source_id"]), "auroc_delta": row["auroc"] - baseline_by_source[row["source_id"]] if row["auroc"] is not None and baseline_by_source.get(row["source_id"]) is not None else None})
        source_metrics[-1].pop("auroc", None)
    bootstrap = paired_source_bootstrap(source_metrics, "model_auroc", "baseline_auroc")
    source_group_model = source_group_medians(oof, "score")
    source_group_baseline = source_group_medians(oof, "baseline_score")
    baseline_groups = {row["source_id"]: row for row in source_group_baseline}
    delta_rows: list[dict[str, Any]] = []
    for row in source_group_model:
        baseline_row = baseline_groups.get(row["source_id"], {})
        delta_rows.append({**row, **{f"baseline_{key}": baseline_row.get(key) for key in ("A_RM", "A_FM", "A_RC", "A_FC", "Delta_manip", "Delta_control", "J_B0")}})
    _write_csv(output_root / "evaluation/per_source_linear_probe.csv", [{**metrics, **next((row for row in delta_rows if row["source_id"] == metrics["source_id"]), {})} for metrics in source_metrics])

    delta_summary = summary_for_deltas(delta_rows)
    control_summary = {
        "model_real_control": finite_summary([row["A_RC"] for row in delta_rows if row.get("A_RC") is not None]),
        "model_fake_control": finite_summary([row["A_FC"] for row in delta_rows if row.get("A_FC") is not None]),
        "baseline_real_control": finite_summary([row["baseline_A_RC"] for row in delta_rows if row.get("baseline_A_RC") is not None]),
        "baseline_fake_control": finite_summary([row["baseline_A_FC"] for row in delta_rows if row.get("baseline_A_FC") is not None]),
    }
    primary_pooled = _pooled_scores(oof, "score")
    baseline_pooled = _pooled_scores(oof, "baseline_score")
    fake_vs_all_real = _pooled_scores(oof, "score", negative_kind=None)
    baseline_fake_vs_all_real = _pooled_scores(oof, "baseline_score", negative_kind=None)
    primary_status_value = primary_status(bootstrap)
    summary = {
        "status": primary_status_value,
        "experiment": "V7 Frozen B0 Supervised Linear Probe",
        "population": population,
        "sources": sources,
        "invalid_sources": population["invalid_sources"],
        "training_protocol": protocol,
        "training_counts": {"main_real_windows": sum(main_training_label(row) == 0 for row in samples), "main_fake_windows": sum(main_training_label(row) == 1 for row in samples), "control_windows_excluded_from_training": sum(row["kind"] == "CTRL" for row in samples)},
        "oof_window_count": len(oof),
        "source_level_metrics": source_metrics,
        "source_bootstrap": bootstrap,
        "pooled_auroc_exploratory": {"model_fake_manip_vs_real_manip": primary_pooled, "baseline_fake_manip_vs_real_manip": baseline_pooled, "model_fake_manip_vs_all_real": fake_vs_all_real, "baseline_fake_manip_vs_all_real": baseline_fake_vs_all_real},
        "control_behavior": control_summary,
        "delta_definition": {"Delta_manip": "A_FM-A_RM", "Delta_control": "A_FC-A_RC", "J_B0": "Delta_manip-Delta_control", "interpretation": "evaluation contrasts, not inference score"},
        "model_delta_summary": delta_summary,
        "convergence": support,
        "paired_reference_used": False,
        "frontend_rerun": False,
        "full_video": False,
        "sealed_test": False,
        "formal_src_modified": False,
        "elapsed_s": time.perf_counter() - started,
    }
    _write_json(output_root / "evaluation/summary.json", summary)
    _write_json(output_root / "run_summary.json", {"status": primary_status_value, "artifact_root": str(output_root), "summary": "evaluation/summary.json", "protocol": "protocol.json", "oof_window_count": len(oof), "formal_src_modified": False, "frontend_rerun": False})
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, default=SOURCE_ARTIFACT_ROOT)
    parser.add_argument("--baseline-root", type=Path, default=BASELINE_ARTIFACT_ROOT)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    args = parser.parse_args()
    print(json.dumps(run(args.source_root, args.baseline_root, args.output_root), indent=2, sort_keys=True, ensure_ascii=False))


if __name__ == "__main__":
    main()
