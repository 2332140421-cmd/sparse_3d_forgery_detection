"""Evaluate frozen real-only models on Pair2 frontend results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from .protocol import (
    FROZEN_MODEL_PATH,
    FROZEN_VAL_SCORE_PATH,
    GENERATORS,
    MODEL_NAMES,
    calibrate_real_validation_threshold,
    auprc,
    auroc,
    bootstrap_source_cluster,
    distribution,
    load_frozen_models,
    paired_deltas,
    read_json,
    score_frontend_results,
    write_json,
)


def evaluate_video_scores(video_scores: Sequence[Mapping[str, object]], *, threshold_source: Path = FROZEN_VAL_SCORE_PATH) -> dict[str, object]:
    thresholds = calibrate_real_validation_threshold(threshold_source)
    result: dict[str, object] = {
        "unit": "video",
        "primary_aggregation": "p95",
        "thresholds": thresholds,
        "models": {},
        "spatial_localization_ground_truth": "SPATIAL_LOCALIZATION_GT_NOT_AVAILABLE",
    }
    rows = list(video_scores)
    for model in MODEL_NAMES:
        valid = [row for row in rows if row.get(model, {}).get("video_p95") is not None]
        labels = [int(str(row["role"]) == "fake") for row in valid]
        scores = [float(row[model]["video_p95"]) for row in valid]
        real = [score for row, score in zip(valid, scores) if str(row["role"]) == "real"]
        fake = [score for row, score in zip(valid, scores) if str(row["role"]) == "fake"]
        threshold = thresholds["models"][model]["threshold"]
        operating = {}
        if threshold is not None:
            real_rows = [row for row in valid if str(row["role"]) == "real"]
            fake_rows = [row for row in valid if str(row["role"]) == "fake"]
            operating = {
                "test_real_fpr": float(np.mean([float(row[model]["video_p95"]) >= threshold for row in real_rows])) if real_rows else None,
                "fake_tpr": float(np.mean([float(row[model]["video_p95"]) >= threshold for row in fake_rows])) if fake_rows else None,
                "overall_tpr": float(np.mean([float(row[model]["video_p95"]) >= threshold for row in fake_rows])) if fake_rows else None,
                "per_generator_tpr": {
                    generator: (
                        float(np.mean([float(row[model]["video_p95"]) >= threshold for row in valid if str(row.get("generator")) == generator]))
                        if any(str(row.get("generator")) == generator for row in valid)
                        else None
                    )
                    for generator in GENERATORS
                },
            }
        generator_metrics = {}
        for generator in GENERATORS:
            generator_sources = {
                str(row["source_id"])
                for row in valid
                if str(row.get("generator")) == generator and str(row["role"]) == "fake"
            }
            subset = [
                row
                for row in valid
                if str(row["source_id"]) in generator_sources
                and (str(row["role"]) == "real" or str(row.get("generator")) == generator)
            ]
            gen_labels = [int(str(row["role"]) == "fake") for row in subset]
            gen_scores = [float(row[model]["video_p95"]) for row in subset]
            generator_metrics[generator] = {
                "source_count": len({str(row["source_id"]) for row in subset if str(row["role"]) == "fake"}),
                "video_count": len(subset),
                "auroc": auroc(gen_labels, gen_scores),
                "auprc": auprc(gen_labels, gen_scores),
                "paired": paired_deltas(subset, model),
                "status": "SMALL_N" if len({str(row["source_id"]) for row in subset if str(row["role"]) == "fake"}) < 4 else "OK",
            }
        macro_auroc = [row.get("auroc") for row in generator_metrics.values() if row.get("auroc") is not None]
        macro_auprc = [row.get("auprc") for row in generator_metrics.values() if row.get("auprc") is not None]
        result["models"][model] = {
            "video_count": len(valid),
            "real_video_count": len(real),
            "fake_video_count": len(fake),
            "auroc": auroc(labels, scores),
            "auprc": auprc(labels, scores),
            "bootstrap_95": bootstrap_source_cluster(valid, model),
            "real_distribution": distribution(real),
            "fake_distribution": distribution(fake),
            "paired": paired_deltas(valid, model),
            "operating_point": operating,
            "per_generator": generator_metrics,
            "macro_average": {
                "auroc": float(np.mean(macro_auroc)) if macro_auroc else None,
                "auprc": float(np.mean(macro_auprc)) if macro_auprc else None,
                "generator_count": len(macro_auroc),
            },
        }
    return result


def evaluate_from_frontend(frontend_path: Path, output: Path) -> dict[str, object]:
    raw = read_json(frontend_path)
    if not isinstance(raw, list):
        raise ValueError("frontend results must be a list")
    models, identity = load_frozen_models()
    scored = score_frontend_results(raw, models)
    metrics = evaluate_video_scores(scored["video_scores"])
    result = {
        "frozen_models": identity,
        "scores": scored,
        "metrics": metrics,
        "fake_used_for_fitting": False,
        "normality_refit": False,
    }
    write_json(output / "scores" / "pair2_scores.json", scored)
    write_json(output / "metrics" / "detection_metrics.json", metrics)
    write_json(output / "models_frozen" / "frozen_model_identity.json", identity)
    write_json(output / "models_frozen" / "normality_models.json", read_json(FROZEN_MODEL_PATH))
    write_json(output / "run_evaluation.json", {key: value for key, value in result.items() if key != "scores"})
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frontend", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(evaluate_from_frontend(args.frontend, args.output), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
