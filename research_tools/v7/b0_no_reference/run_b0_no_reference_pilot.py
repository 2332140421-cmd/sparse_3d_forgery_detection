"""Run the frozen B0 source-disjoint no-reference development pilot."""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from .analyze_b0_no_reference import choose_status, evaluation, real_behavior, source_score_rows, window_score_rows
from .b0_reconstruction import load_and_reproduce
from .robust_source_balanced_scaler import fit_source_balanced_scaler


DATA_ROOT = Path("/root/autodl-tmp/data/sparse_3d_forgery_detection")
SOURCE_ARTIFACT_ROOT = DATA_ROOT / "derived/v7_activityforensics_paired_second_order_pilot_v1"
OUTPUT_ROOT = DATA_ROOT / "derived/v7_activityforensics_b0_no_reference_pilot_v1"
BOOTSTRAP_SEED = 20260909
BOOTSTRAP_REPLICATES = 10_000


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


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _source_ids(selected_pairs: list[dict[str, Any]], frozen_windows: list[dict[str, Any]]) -> list[str]:
    pair_sources = [str(row["source_id"]) for row in selected_pairs]
    window_sources = sorted({str(row["source_id"]) for row in frozen_windows})
    if len(selected_pairs) != 16 or len(frozen_windows) != 192 or sorted(pair_sources) != window_sources:
        raise ValueError("frozen population must remain 16 source pairs and 192 windows")
    if len(set(pair_sources)) != 16:
        raise ValueError("source identities are not one-to-one in frozen population")
    return sorted(pair_sources)


def _historical_paired_positive(artifact_root: Path) -> bool:
    summary = _load_json(artifact_root / "metrics/structural_order_summary.json")
    b0 = summary.get("orders", {}).get("K0_S", {}).get("G", {})
    return bool(
        b0.get("median") is not None
        and float(b0["median"]) > 0
        and float(b0.get("positive_fraction", 0)) >= 0.70
    )


def run(source_artifact_root: Path = SOURCE_ARTIFACT_ROOT, output_root: Path = OUTPUT_ROOT) -> dict[str, Any]:
    started = time.perf_counter()
    selected_pairs = _load_json(source_artifact_root / "manifests/selected_pairs.json")
    frozen_windows = _load_json(source_artifact_root / "manifests/window_manifest.json")
    results = _load_json(source_artifact_root / "frontend/window_results.json")
    sources = _source_ids(selected_pairs, frozen_windows)
    if len(results) != 192:
        raise ValueError(f"frozen frontend result count changed: {len(results)}")

    observations, b0_windows, reproduction = load_and_reproduce(source_artifact_root)
    if len(b0_windows) != len(frozen_windows):
        raise ValueError("B0 window reconstruction count does not match frozen manifest")
    manifest_ids = [str(row["window_id"]) for row in frozen_windows]
    reconstructed_ids = [str(row["window_id"]) for row in b0_windows]
    if manifest_ids != reconstructed_ids:
        raise ValueError("frozen window order or identity changed")

    # Only small manifests and scalar/vector diagnostics are copied.  No video,
    # NPZ, frontend cache, RGB, or dense depth is copied into the new root.
    output_root.mkdir(parents=True, exist_ok=True)
    _write_json(output_root / "manifests/frozen_population.json", {
        "source_pair_count": len(selected_pairs),
        "window_count": len(frozen_windows),
        "source_ids": sources,
        "pair_ids": [str(row["pair_id"]) for row in selected_pairs],
        "selected_pairs": [
            {"pair_id": str(row["pair_id"]), "source_id": str(row["source_id"]), "generator": row.get("generator"), "manipulation_operation": row.get("manipulation_operation"), "official_split": row.get("official_split"), "lineage_status": row.get("lineage_status")}
            for row in selected_pairs
        ],
        "invalid_b0_sources": [],
        "population_role": "16-source development diagnostic; annotation-selected frozen windows; not sealed test",
        "source_artifact_root": str(source_artifact_root),
    })
    _write_json(output_root / "manifests/frozen_windows.json", frozen_windows)
    _write_json(output_root / "manifests/source_artifact_links.json", {
        "source_root": str(source_artifact_root),
        "links": {
            "selected_pairs": str(source_artifact_root / "manifests/selected_pairs.json"),
            "window_manifest": str(source_artifact_root / "manifests/window_manifest.json"),
            "frontend_window_results": str(source_artifact_root / "frontend/window_results.json"),
            "historical_b0_window_metrics": str(source_artifact_root / "metrics/per_window_signal.csv"),
            "historical_b0_summary": str(source_artifact_root / "metrics/structural_order_summary.json"),
        },
        "frontend_rerun": False,
        "copied_large_arrays": False,
    })

    # Every source is retained in accounting.  A source with no real B0 values
    # is unavailable for fitting but is never silently dropped.
    real_by_source: dict[str, list[list[float]]] = {source: [] for source in sources}
    for row in observations:
        if row["role"] == "real":
            real_by_source[row["source_id"]].append(row["values"])
    invalid_sources = sorted(source for source, values in real_by_source.items() if not values)
    _write_json(output_root / "manifests/frozen_population.json", {
        **_load_json(output_root / "manifests/frozen_population.json"),
        "invalid_b0_sources": invalid_sources,
        "valid_b0_source_count": len(sources) - len(invalid_sources),
        "b0_reconstruction": reproduction,
    })

    scored_observations: list[dict[str, Any]] = []
    scaler_records: list[dict[str, Any]] = []
    support_records: list[dict[str, Any]] = []
    for held_out in sources:
        training_sources = [source for source in sources if source != held_out and real_by_source[source]]
        training_values = {source: real_by_source[source] for source in training_sources}
        if not training_values:
            scaler_records.append({"held_out_source": held_out, "status": "NO_TRAINING_REAL_B0"})
            support_records.append({"held_out_source": held_out, "training_real_source_count": 0, "training_real_observation_count": 0, "fake_fitting_count": 0, "held_out_real_fitting_count": 0, "status": "NO_TRAINING_REAL_B0"})
            continue
        scaler = fit_source_balanced_scaler(training_values)
        training_count = sum(len(values) for values in training_values.values())
        held_rows = [row for row in observations if row["source_id"] == held_out]
        scaler_records.append({"held_out_source": held_out, "status": "NO_HELDOUT_B0" if not held_rows else "CALIBRATED", "held_out_observation_count": len(held_rows), **scaler.as_dict()})
        support_records.append({
            "held_out_source": held_out,
            "training_real_source_count": len(training_sources),
            "training_real_observation_count": training_count,
            "fake_fitting_count": 0,
            "held_out_real_fitting_count": 0,
            "held_out_observation_count": len(held_rows),
            "status": "NO_HELDOUT_B0" if not held_rows else "CALIBRATED",
        })
        for row in held_rows:
            scored_observations.append({**row, "score": scaler.score(row["values"]), "calibration_held_out_source": held_out})

    window_rows = window_score_rows(scored_observations)
    source_rows = source_score_rows(window_rows, sources)
    _write_json(output_root / "calibration/loso_scalers.json", {
        "protocol": "source-disjoint leave-one-source-out; real B0 observations only",
        "fake_fitting_count": 0,
        "held_out_real_fitting_count": 0,
        "folds": scaler_records,
    })
    _write_csv(output_root / "calibration/per_fold_support.csv", support_records)
    _write_csv(output_root / "scores/per_observation_b0_scores.csv", scored_observations)
    _write_csv(output_root / "scores/per_window_b0_scores.csv", window_rows)
    _write_csv(output_root / "scores/per_source_b0_scores.csv", source_rows)

    real_summary = real_behavior(window_rows, sources)
    eval_result = evaluation(source_rows, scored_observations, window_rows)
    _write_csv(output_root / "evaluation/per_source_b0_deltas.csv", eval_result["per_source_deltas"])
    historical_summary = _load_json(source_artifact_root / "metrics/structural_order_summary.json")
    historical_b0 = historical_summary.get("orders", {}).get("K0_S", {})
    status = choose_status(
        eval_result["Delta_manip"],
        eval_result["J_B0"],
        len(eval_result["per_source_deltas"]),
        _historical_paired_positive(source_artifact_root),
        eval_result["exploratory_auroc"].get("fake_manip_vs_all_heldout_real"),
    )
    summary = {
        "status": status,
        "experiment": "V7 B0 Minimal No-Reference Development Baseline",
        "b0_definition": "S_t=[mean,std,p25,p75] from frozen K0_S; window median over component/time observations",
        "normalization": "frozen component pairwise distance median normalization from prior B0 implementation; not recomputed in this run",
        "population": {"source_pairs": len(selected_pairs), "windows": len(frozen_windows), "sources": len(sources), "valid_b0_sources": len(sources) - len(invalid_sources), "invalid_b0_sources": invalid_sources, "annotation_selected": True, "sealed_test": False, "full_video": False},
        "frontend_rerun": False,
        "no_reference_inference": True,
        "paired_distance_used_for_scoring": False,
        "fake_fitting_count": 0,
        "held_out_real_fitting_count": 0,
        "bootstrap": {"seed": BOOTSTRAP_SEED, "replicates": BOOTSTRAP_REPLICATES, "statistical_unit": "source"},
        "reconstruction": reproduction,
        "held_out_real_behavior": real_summary,
        "evaluation": {key: value for key, value in eval_result.items() if key != "per_source_deltas"},
        "historical_paired_b0": {"source": str(source_artifact_root / "metrics/structural_order_summary.json"), "summary": historical_b0, "comparison_rule": "direction and source positive fraction only; score magnitudes not compared"},
        "scoring_excludes": ["paired real reference", "fake fitting", "held-out real fitting", "MANIP/CTRL labels", "generator", "operation", "anomaly score"],
        "status_rule": "predeclared Delta_manip and J_B0 positive gates; pooled AUROC >=0.70 with unstable source outcomes flags heterogeneity",
        "elapsed_s": time.perf_counter() - started,
    }
    _write_json(output_root / "evaluation/b0_no_reference_summary.json", summary)
    _write_json(output_root / "run_summary.json", {
        "status": status,
        "artifact_root": str(output_root),
        "source_artifact_root": str(source_artifact_root),
        "source_pairs": 16,
        "windows": 192,
        "frontend_rerun": False,
        "large_inputs_copied": False,
        "scored_observations": len(scored_observations),
        "scored_windows": len(window_rows),
        "valid_b0_sources": len(sources) - len(invalid_sources),
        "invalid_b0_sources": invalid_sources,
        "fake_fitting_count": 0,
        "held_out_real_fitting_count": 0,
        "formal_src_modified": False,
        "no_reference_inference": True,
        "summary": "evaluation/b0_no_reference_summary.json",
    })
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-artifact-root", type=Path, default=SOURCE_ARTIFACT_ROOT)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    args = parser.parse_args()
    print(json.dumps(run(args.source_artifact_root, args.output_root), indent=2, sort_keys=True, ensure_ascii=False))


if __name__ == "__main__":
    main()
