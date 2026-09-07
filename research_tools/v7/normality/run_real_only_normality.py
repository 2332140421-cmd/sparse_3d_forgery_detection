"""Run the frozen V7 real-only normality pilot and bounded pair check."""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import platform
import time

import numpy as np

from research_tools.v7.feasibility.materialize_vript_frozen_targets import write_json

from .fit import fit_models, score_windows, source_summaries
from .frontend import DepthProRunner, OnlineBootsTapir, run_frontend_window
from .protocol import (
    MODEL_NAMES,
    build_source_split,
    build_window_manifest,
    feature_schema,
)


DATA_ROOT = Path("/root/autodl-tmp/data/sparse_3d_forgery_detection")
OUTPUT_ROOT = DATA_ROOT / "derived/v7_real_only_normality_pilot_v1"
FROZEN_MANIFEST = DATA_ROOT / "derived/v7_genvidbench_core_pilot_v1/manifests/train_real_pilot.json"
MEDIA_MANIFEST = OUTPUT_ROOT / "manifests/media_validation.json"
PAIR_PLAN = DATA_ROOT / "derived/v7_genvidbench_core_pilot_v1/materialization/materialization_plan.json"


def _read_json(path: Path) -> object:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _prepare_population() -> tuple[dict[str, list[dict[str, object]]], list[dict[str, object]]]:
    media = _read_json(MEDIA_MANIFEST)
    frozen = _read_json(FROZEN_MANIFEST)
    if not isinstance(media, list) or not isinstance(frozen, list):
        raise ValueError("population manifests must be lists")
    frozen_by_id = {str(row["source_id"]): row for row in frozen}
    merged: list[dict[str, object]] = []
    for row in media:
        source_id = str(row["source_id"])
        if source_id not in frozen_by_id:
            raise ValueError(f"media row is not in frozen population: {source_id}")
        merged.append({**dict(row), "role": "real", "frozen_ordinal": frozen_by_id[source_id]["ordinal"]})
    split = build_source_split(merged)
    train_ids = {str(row["source_id"]) for row in split["real_train"]}
    val_ids = {str(row["source_id"]) for row in split["real_val"]}
    ordered = []
    for row in merged:
        source_id = str(row["source_id"])
        if source_id in train_ids:
            row = {**row, "role": "real_train"}
        elif source_id in val_ids:
            row = {**row, "role": "real_val"}
        else:
            raise ValueError(f"source split lost frozen source: {source_id}")
        ordered.append(row)
    split = {"real_train": [row for row in ordered if row["role"] == "real_train"], "real_val": [row for row in ordered if row["role"] == "real_val"]}
    return split, ordered


def _write_population_manifests(output: Path, split: dict[str, list[dict[str, object]]], ordered: list[dict[str, object]]) -> list[dict[str, object]]:
    windows = build_window_manifest(ordered)
    train_ids = {str(row["source_id"]) for row in split["real_train"]}
    windows = [{**row, "role": "real_train" if str(row["source_id"]) in train_ids else "real_val"} for row in windows]
    write_json(output / "manifests" / "population_split.json", {
        "frozen_source_order": [row["source_id"] for row in ordered],
        "real_train": [row["source_id"] for row in split["real_train"]],
        "real_val": [row["source_id"] for row in split["real_val"]],
        "fake_count_for_fitting": 0,
        "media_valid": len(ordered),
        "frozen_population_requested": 32,
    })
    write_json(output / "manifests" / "window_manifest.json", windows)
    return windows


def _run_frontend(windows: list[dict[str, object]], output: Path, args: argparse.Namespace) -> list[dict[str, object]]:
    if any(row["status"] != "AVAILABLE" for row in windows):
        raise RuntimeError("one or more frozen windows are unavailable; no padding is allowed")
    tracker = OnlineBootsTapir(args.tapnet_source, args.tapnet_checkpoint)
    depth_runner = DepthProRunner(args.depth_source, args.depth_checkpoint)
    results: list[dict[str, object]] = []
    started = time.perf_counter()
    for index, item in enumerate(windows, 1):
        print(f"frontend {index}/{len(windows)} {item['window_id']}", flush=True)
        result = run_frontend_window(item, tracker, depth_runner, output)
        results.append(result)
        write_json(output / "frontend" / "window_results_progress.json", results)
        print(
            f"  geometry={result['coverage']['final_geometry_valid_fraction']:.3f} "
            f"components={result['component_count']} elapsed={result['elapsed_s']:.1f}s "
            f"peak_gpu={result['peak_gpu_memory_bytes']}",
            flush=True,
        )
    del tracker, depth_runner
    gc.collect()
    write_json(output / "frontend" / "window_results.json", results)
    write_json(output / "frontend" / "run_meta.json", {
        "windows": len(results),
        "elapsed_s": time.perf_counter() - started,
        "causal_execution": True,
        "frontend_unchanged": True,
        "fake_used": False,
        "python": platform.python_version(),
    })
    return results


def _load_frontend_results(output: Path) -> list[dict[str, object]]:
    path = output / "frontend" / "window_results.json"
    rows = _read_json(path)
    if not isinstance(rows, list):
        raise ValueError("frontend window results must be a list")
    return rows


def _frontend_summary(output: Path, results: list[dict[str, object]]) -> dict[str, object]:
    geometry = [float(row["coverage"]["final_geometry_valid_fraction"]) for row in results]
    component = [int(row["component_count"]) for row in results]
    summary = {
        "windows": len(results),
        "complete_windows": sum(row.get("status") == "COMPLETE" for row in results),
        "windows_with_component": sum(value > 0 for value in component),
        "no_component_windows": sum(value == 0 for value in component),
        "component_success_rate": float(np.mean(np.asarray(component) > 0)),
        "geometry_coverage_median": float(np.median(geometry)),
        "geometry_coverage_iqr": float(np.percentile(geometry, 75) - np.percentile(geometry, 25)),
        "geometry_coverage_p10": float(np.percentile(geometry, 10)),
        "geometry_coverage_p90": float(np.percentile(geometry, 90)),
        "s_t_valid_states": int(sum(int(row["valid_s_states"]) for row in results)),
        "s_t_total_state_slots": int(sum(len(row["timestamps_s"]) * int(row["component_count"]) for row in results)),
        "s_t_valid_rate": float(sum(int(row["valid_s_states"]) for row in results) / max(1, sum(len(row["timestamps_s"]) * int(row["component_count"]) for row in results))),
        "delta_s_observations": int(sum(int(row["valid_delta_s"]) for row in results)),
        "delta2_s_observations": int(sum(int(row["valid_delta2_s"]) for row in results)),
        "elapsed_s_total": float(sum(float(row["elapsed_s"]) for row in results)),
        "peak_gpu_memory_bytes": int(max(row["peak_gpu_memory_bytes"] for row in results if row["peak_gpu_memory_bytes"] is not None)),
        "frontend_unchanged": True,
        "causal_execution": True,
    }
    write_json(output / "metrics" / "frontend_summary.json", summary)
    return summary


def _fit_and_score(output: Path, windows: list[dict[str, object]], results: list[dict[str, object]]) -> dict[str, object]:
    by_id = {str(row["window"]["window_id"]): row for row in results}
    missing = [str(row["window_id"]) for row in windows if str(row["window_id"]) not in by_id]
    if missing:
        raise ValueError(f"frontend results missing windows: {missing[:3]}")
    train_windows = [by_id[str(row["window_id"])] for row in windows if row["role"] == "real_train"]
    val_windows = [by_id[str(row["window_id"])] for row in windows if row["role"] == "real_val"]
    if not train_windows or not val_windows:
        raise ValueError("source-level train/validation split is empty")
    models = fit_models(train_windows)
    write_json(output / "models" / "normality_models.json", {
        "feature_schema": feature_schema(),
        "fitting_population": "real_train_only",
        "fake_count": 0,
        "models": {name: model.as_dict() for name, model in models.items()},
    })
    train_scores, _ = score_windows(models, train_windows)
    val_scores, _ = score_windows(models, val_windows)
    write_json(output / "scores" / "train_window_scores.json", train_scores)
    write_json(output / "scores" / "val_window_scores.json", val_scores)
    source_train = source_summaries(models, train_windows)
    source_val = source_summaries(models, val_windows)

    metrics: dict[str, object] = {
        "models": {},
        "aggregation": {"primary": "p95", "sensitivity": "median", "sum_used": False},
        "train_source_count": len({row["window"]["source_id"] for row in train_windows}),
        "val_source_count": len({row["window"]["source_id"] for row in val_windows}),
        "fake_count_for_fitting": 0,
        "source_level_train": source_train,
        "source_level_val": source_val,
    }
    for name, model in models.items():
        train_rows = [row[name] for row in train_scores]
        val_rows = [row[name] for row in val_scores]
        train_values = [value for row in train_rows for value in [row["video_p95"]] if value is not None]
        val_values = [value for row in val_rows for value in [row["video_p95"]] if value is not None]
        train_feature_count = sum(int(row[name]["feature_count"]) for row in train_scores)
        val_feature_count = sum(int(row[name]["feature_count"]) for row in val_scores)
        metrics["models"][name] = {
            "feature_dimension": int(model.mean.size),
            "train_feature_count": train_feature_count,
            "val_feature_count": val_feature_count,
            "train_score_distribution_p95_aggregation": {
                "video_count": len(train_values),
                **_distribution(train_values),
            },
            "val_score_distribution_p95_aggregation": {
                "video_count": len(val_values),
                **_distribution(val_values),
            },
            "covariance_type": model.covariance_type,
            "covariance_condition_number": model.condition_number,
            "regularization_lambda": model.regularization_lambda,
            "train_window_missing_rate": float(sum(not row[name]["feature_count"] for row in train_scores) / len(train_scores)),
            "val_window_missing_rate": float(sum(not row[name]["feature_count"] for row in val_scores) / len(val_scores)),
        }
    write_json(output / "metrics" / "normality_metrics.json", metrics)
    return metrics


def _distribution(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"median": None, "iqr": None, "p10": None, "p90": None}
    array = np.asarray(values, dtype=np.float64)
    return {
        "median": float(np.median(array)),
        "iqr": float(np.percentile(array, 75) - np.percentile(array, 25)),
        "p10": float(np.percentile(array, 10)),
        "p90": float(np.percentile(array, 90)),
    }


def _paired_status(output: Path) -> dict[str, object]:
    # The prior bounded materialization plan is the frozen Pair2 access result;
    # it requires multi-volume/full archives beyond the 20 GiB pilot cap.
    plan = _read_json(PAIR_PLAN)
    status = "PAIRED_TEST_MEDIA_BLOCKED"
    payload = {
        "status": status,
        "reason": plan.get("reason") if isinstance(plan, dict) else "frozen Pair2 media source is unavailable within the bounded cap",
        "download_gate_bytes": 20 * 1024**3,
        "fake_used_for_fitting": False,
        "real_only_training_unaffected": True,
        "frozen_manifest": str(DATA_ROOT / "derived/v7_genvidbench_core_pilot_v1/paired_test/paired_test_manifest.json"),
    }
    write_json(output / "paired_test" / "status.json", payload)
    return payload


def run(args: argparse.Namespace) -> dict[str, object]:
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    split, ordered = _prepare_population()
    if len(ordered) < 24:
        raise RuntimeError("REAL_TRAIN_POPULATION_INSUFFICIENT")
    windows = _write_population_manifests(output, split, ordered)
    if args.run_frontend:
        frontend_rows = _run_frontend(windows, output, args)
    else:
        frontend_rows = _load_frontend_results(output)
    frontend_metrics = _frontend_summary(output, frontend_rows)
    metrics = _fit_and_score(output, windows, frontend_rows)
    paired = _paired_status(output)
    summary = {
        "status": "REAL_ONLY_NORMALITY_TRAINED_PAIRED_TEST_BLOCKED",
        "h3_conclusion": "REAL_ONLY_NORMALITY_TRAINED_PAIRED_TEST_BLOCKED",
        "frozen_timescale_s": 1.0,
        "anchors": [0.25, 0.50, 0.75],
        "frozen_population_requested": 32,
        "media_valid": len(ordered),
        "real_train_sources": len(split["real_train"]),
        "real_val_sources": len(split["real_val"]),
        "train_windows": len([row for row in windows if row["role"] == "real_train"]),
        "val_windows": len([row for row in windows if row["role"] == "real_val"]),
        "feature_models": list(MODEL_NAMES),
        "frontend_metrics": frontend_metrics,
        "metrics": metrics,
        "paired_test": paired,
        "component_config_unchanged": True,
        "structure_state_unchanged": True,
        "delta_s_unchanged": True,
        "delta2_s_unchanged": True,
        "fake_count_for_fitting": 0,
        "synthetic_perturbation": False,
        "spatial_localization_ground_truth": "SPATIAL_LOCALIZATION_GT_NOT_AVAILABLE",
    }
    write_json(output / "run_summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--run-frontend", action="store_true")
    parser.add_argument("--tapnet-source", type=Path, required=False)
    parser.add_argument("--tapnet-checkpoint", type=Path, required=False)
    parser.add_argument("--depth-source", type=Path, required=False)
    parser.add_argument("--depth-checkpoint", type=Path, required=False)
    args = parser.parse_args()
    if args.run_frontend and not all((args.tapnet_source, args.tapnet_checkpoint, args.depth_source, args.depth_checkpoint)):
        parser.error("--run-frontend requires all provider paths")
    result = run(args)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
