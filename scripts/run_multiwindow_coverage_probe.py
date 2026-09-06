#!/usr/bin/env python3
"""Run the frozen-model, label-blind five-window coverage probe."""

import json
import os
from pathlib import Path
import time

import numpy as np
import torch

import run_temporal_learnability_probe as temporal_run
from sparse3d_forgery.experiments.multiwindow_probe import (
    actual_time_spans,
    aggregate_video_scores,
    arithmetic_window_score,
    bootstrap_auroc_difference,
    classify_probe,
    eligible_window_records,
    exact_artifact_match,
    fixed_window_starts,
)
from sparse3d_forgery.experiments.spatial_probe import SelfTemporalModel, SpatialTemporalModel
from sparse3d_forgery.experiments.temporal_probe import (
    MissingAwareGRU,
    build_targets,
    history_normalize,
    linear_extrapolation,
    rank_auroc,
    zero_displacement,
)
from sparse3d_forgery.frontend import VggtFrontend, VggtFrontendConfig
from sparse3d_forgery.particle_sequence import load_particle_sequence, save_particle_sequence
from sparse3d_forgery.video_input import VideoSource, decode_video


SOURCE_HEAD = "9e67ca5fdc33398a29dab0e2f2369b8a3e35d399"
SEED = 20260906
OUTPUT_ROOT = temporal_run.DATA_ROOT / "derived/multiwindow_coverage_probe_v1"
TEMPORAL_ROOT = temporal_run.OUTPUT_ROOT
SPATIAL_ROOT = temporal_run.DATA_ROOT / "derived/spatial_dependency_probe_v1"
PREDICTORS = ("zero", "linear", "t0", "t1", "spatial")


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def prior_items() -> list[dict]:
    with (TEMPORAL_ROOT / "pilot_manifest.json").open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    selected = [item for item in manifest["items"] if item["role"] in ("real_val", "fake_probe")]
    if sum(item["role"] == "real_val" for item in selected) != 32 or sum(
        item["role"] == "fake_probe" for item in selected
    ) != 32:
        raise RuntimeError("frozen 32/32 video identity gate failed")
    return selected


def make_jobs(items: list[dict]) -> list[dict]:
    jobs = []
    for item in items:
        starts = fixed_window_starts(item["declared_frame_count"])
        old_prefix = TEMPORAL_ROOT / item["artifact_prefix"]
        old = load_particle_sequence(old_prefix)
        for position, start in enumerate(starts):
            indices = tuple(range(start, start + 16))
            reuse = exact_artifact_match(old.frame_indices, indices)
            prefix = (
                old_prefix
                if reuse
                else OUTPUT_ROOT
                / "causal_artifacts"
                / item["role"]
                / item["source_video_id"]
                / f"w{position}"
            )
            jobs.append(
                {
                    "role": item["role"],
                    "source_video_id": item["source_video_id"],
                    "video_path": item["video_path"],
                    "declared_frame_count": item["declared_frame_count"],
                    "window_position": position,
                    "window_start": start,
                    "frame_indices": list(indices),
                    "artifact_prefix": str(prefix),
                    "reused_old_center": reuse,
                    "status": "pending",
                }
            )
    return jobs


def preprocess(jobs: list[dict]) -> None:
    frontend = None
    for number, job in enumerate(jobs, 1):
        prefix = Path(job["artifact_prefix"])
        try:
            if prefix.with_suffix(".npz").is_file() and prefix.with_suffix(".json").is_file():
                sequence = load_particle_sequence(prefix)
                if sequence.source_video_id != job["source_video_id"] or not exact_artifact_match(
                    sequence.frame_indices, tuple(job["frame_indices"])
                ):
                    raise RuntimeError("existing artifact does not exactly match requested window")
                if job.get("status") != "eligible_generated":
                    job["status"] = (
                        "eligible_reused_center"
                        if job["reused_old_center"]
                        else "eligible_resumed"
                    )
            else:
                if job["reused_old_center"]:
                    raise FileNotFoundError("frozen center artifact is missing")
                if frontend is None:
                    frontend = VggtFrontend(
                        temporal_run.WEIGHT_PATH,
                        VggtFrontendConfig(num_tracks=128, target_size=518),
                    )
                started = time.perf_counter()
                decoded = decode_video(
                    VideoSource(
                        sample_id=f"multiwindow-{job['source_video_id']}-w{job['window_position']}",
                        source_video_id=job["source_video_id"],
                        source_locator=temporal_run.EXTRACTED_ROOT / job["video_path"],
                    ),
                    job["frame_indices"],
                )
                sequence = frontend.extract_causal_window(decoded, temporal_run.HISTORY_COUNT)
                job["frontend_elapsed_s"] = time.perf_counter() - started
                if sequence.provenance.get("causal_training_eligible") is not True:
                    job["status"] = "causal_ineligible"
                    job["failure"] = sequence.provenance.get("alignment_failure")
                else:
                    save_particle_sequence(sequence, prefix)
                    job["status"] = "eligible_generated"
            if job["status"].startswith("eligible"):
                sequence = load_particle_sequence(prefix)
                job["frontend_diagnostics"] = {
                    "common_historical_correspondence_count": sequence.provenance.get(
                        "common_historical_correspondence_count"
                    ),
                    "estimated_scale": sequence.provenance.get("estimated_scale"),
                    "rotation_determinant": sequence.provenance.get("rotation_determinant"),
                    **(sequence.provenance.get("alignment_diagnostics") or {}),
                }
        except Exception as exc:
            job["status"] = "frontend_failure"
            job["failure"] = f"{type(exc).__name__}: {exc}"
        write_json(OUTPUT_ROOT / "window_manifest.json", {"source_head": SOURCE_HEAD, "jobs": jobs})
        print(number, len(jobs), job["role"], job["source_video_id"], job["window_position"], job["status"], flush=True)


def load_frozen_models(device: torch.device) -> dict[str, torch.nn.Module]:
    t0_checkpoint = torch.load(TEMPORAL_ROOT / "checkpoint.pt", map_location=device, weights_only=False)
    if (
        tuple(t0_checkpoint["config"]["horizons_observation_offset"]) != (1, 2, 4, 8)
        or t0_checkpoint["config"]["model"] != {
            "type": "single-layer GRUCell",
            "input_dim": 3,
            "hidden_dim": 64,
        }
    ):
        raise RuntimeError("T0 checkpoint horizon provenance mismatch")
    t0 = MissingAwareGRU(hidden_dim=64).to(device)
    t0.load_state_dict(t0_checkpoint["model_state_dict"])
    t1_checkpoint = torch.load(SPATIAL_ROOT / "model_t1.pt", map_location=device, weights_only=False)
    spatial_checkpoint = torch.load(
        SPATIAL_ROOT / "model_spatial.pt", map_location=device, weights_only=False
    )
    with (SPATIAL_ROOT / "probe_config.json").open(encoding="utf-8") as handle:
        spatial_config = json.load(handle)
    if (
        spatial_config.get("topology") != "probe_dense_no_self"
        or spatial_config.get("spatial_dim") != 32
        or spatial_config.get("hidden_dim") != 64
        or tuple(spatial_config.get("horizons", ())) != (1, 2, 4, 8)
    ):
        raise RuntimeError("T1/S architecture configuration mismatch")
    if t1_checkpoint.get("source_head") != "f24ca1566b010fecd39ebda8714c50e5121563e6" or spatial_checkpoint.get(
        "source_head"
    ) != "f24ca1566b010fecd39ebda8714c50e5121563e6":
        raise RuntimeError("T1/S checkpoint provenance mismatch")
    t1 = SelfTemporalModel(64, 32).to(device)
    spatial = SpatialTemporalModel(64, 32).to(device)
    t1.load_state_dict(t1_checkpoint["model_state_dict"])
    spatial.load_state_dict(spatial_checkpoint["model_state_dict"])
    models = {"t0": t0, "t1": t1, "spatial": spatial}
    for model in models.values():
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
    return models


def score_job(job: dict, models: dict, device: torch.device) -> dict:
    sequence = load_particle_sequence(job["artifact_prefix"])
    normalized = history_normalize(sequence, 8)
    targets, validity = build_targets(normalized.xyz, normalized.validity, 8)
    if not validity.any():
        raise ValueError("no valid prediction targets")
    history_xyz = torch.from_numpy(normalized.xyz[:8][None]).to(device)
    history_validity = torch.from_numpy(normalized.validity[:8][None]).to(device)
    predictions = {"zero": zero_displacement(normalized.xyz.shape[1])}
    with torch.inference_mode():
        for name, model in models.items():
            predictions[name] = model(history_xyz, history_validity)[0].cpu().numpy()
    linear, linear_coverage = linear_extrapolation(
        normalized.xyz, normalized.validity, sequence.timestamps_s, 8
    )
    predictions["linear"] = linear
    scores = {}
    valid_counts = {}
    for name, prediction in predictions.items():
        mask = validity & linear_coverage if name == "linear" else validity
        errors = np.linalg.norm(prediction - targets, axis=-1)
        scores[name] = arithmetic_window_score(errors, mask)
        valid_counts[name] = int(mask.sum())
    return {
        "role": job["role"],
        "source_video_id": job["source_video_id"],
        "window_position": job["window_position"],
        "window_start": job["window_start"],
        "reused_old_center": job["reused_old_center"],
        "scores": scores,
        "valid_counts": valid_counts,
        "time_spans": actual_time_spans(sequence.timestamps_s),
    }


def distribution(values: list[float]) -> dict:
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(array.size),
        "min": float(array.min()),
        "median": float(np.median(array)),
        "mean": float(array.mean()),
        "p75": float(np.percentile(array, 75)),
        "p90": float(np.percentile(array, 90)),
        "max": float(array.max()),
    }


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    manifest_path = OUTPUT_ROOT / "window_manifest.json"
    if manifest_path.exists():
        with manifest_path.open(encoding="utf-8") as handle:
            saved = json.load(handle)
        if saved.get("source_head") != SOURCE_HEAD:
            raise RuntimeError("existing run source HEAD differs")
        jobs = saved["jobs"]
    else:
        jobs = make_jobs(prior_items())
        write_json(manifest_path, {"source_head": SOURCE_HEAD, "jobs": jobs})
    preprocess(jobs)

    eligible = eligible_window_records(jobs)
    device = torch.device("cuda")
    models = load_frozen_models(device)
    scores = []
    for job in eligible:
        try:
            scores.append(score_job(job, models, device))
        except ValueError as exc:
            job["score_failure"] = str(exc)
    write_json(OUTPUT_ROOT / "window_scores.json", scores)
    write_json(
        OUTPUT_ROOT / "frontend_diagnostics.json",
        [
            {key: job.get(key) for key in ("role", "source_video_id", "window_position", "status", "frontend_diagnostics", "failure") if key in job}
            for job in jobs
        ],
    )

    grouped = {}
    for item in scores:
        grouped.setdefault((item["role"], item["source_video_id"]), []).append(item)
    videos = []
    for (role, video_id), windows in sorted(grouped.items()):
        windows.sort(key=lambda value: value["window_position"])
        result = {"role": role, "source_video_id": video_id, "unique_eligible_windows": len(windows)}
        result["predictors"] = {}
        for predictor in PREDICTORS:
            values = [window["scores"][predictor] for window in windows]
            aggregate = aggregate_video_scores(values)
            centers = [window["scores"][predictor] for window in windows if window["reused_old_center"]]
            if len(centers) != 1:
                raise RuntimeError("each video must preserve exactly one previous center score")
            aggregate["previous_center_score"] = centers[0]
            aggregate["max_to_center_ratio"] = aggregate["score_max"] / centers[0]
            aggregate["max_window_position"] = windows[aggregate["max_window_offset"]]["window_position"]
            result["predictors"][predictor] = aggregate
        videos.append(result)
    write_json(OUTPUT_ROOT / "video_scores.json", videos)

    by_role = {role: [video for video in videos if video["role"] == role] for role in ("real_val", "fake_probe")}
    aurocs = {}
    arrays = {}
    for predictor in PREDICTORS:
        real_center = np.array([v["predictors"][predictor]["previous_center_score"] for v in by_role["real_val"]])
        fake_center = np.array([v["predictors"][predictor]["previous_center_score"] for v in by_role["fake_probe"]])
        real_max = np.array([v["predictors"][predictor]["score_max"] for v in by_role["real_val"]])
        fake_max = np.array([v["predictors"][predictor]["score_max"] for v in by_role["fake_probe"]])
        arrays[predictor] = {"real_center": real_center, "fake_center": fake_center, "real_max": real_max, "fake_max": fake_max}
        aurocs[predictor] = {"center": rank_auroc(real_center, fake_center), "max": rank_auroc(real_max, fake_max)}
        aurocs[predictor]["delta"] = aurocs[predictor]["max"] - aurocs[predictor]["center"]
    bootstrap = {
        "t0_max_minus_center": bootstrap_auroc_difference(arrays["t0"]["real_max"], arrays["t0"]["fake_max"], arrays["t0"]["real_center"], arrays["t0"]["fake_center"]),
        "spatial_max_minus_center": bootstrap_auroc_difference(arrays["spatial"]["real_max"], arrays["spatial"]["fake_max"], arrays["spatial"]["real_center"], arrays["spatial"]["fake_center"]),
        "t0_max_minus_zero_max": bootstrap_auroc_difference(arrays["t0"]["real_max"], arrays["t0"]["fake_max"], arrays["zero"]["real_max"], arrays["zero"]["fake_max"]),
        "spatial_max_minus_zero_max": bootstrap_auroc_difference(arrays["spatial"]["real_max"], arrays["spatial"]["fake_max"], arrays["zero"]["real_max"], arrays["zero"]["fake_max"]),
    }
    write_json(OUTPUT_ROOT / "bootstrap_metrics.json", bootstrap)

    ratios = {
        role: {
            predictor: distribution([v["predictors"][predictor]["max_to_center_ratio"] for v in values])
            for predictor in PREDICTORS
        }
        for role, values in by_role.items()
    }
    positions = {
        role: {
            predictor: {
                str(position): sum(v["predictors"][predictor]["max_window_position"] == position for v in values)
                for position in range(5)
            }
            for predictor in PREDICTORS
        }
        for role, values in by_role.items()
    }
    fake_ratio_higher = ratios["fake_probe"]["t0"]["median"] > ratios["real_val"]["t0"]["median"]
    time_values = {
        "window_duration_s": distribution([x["time_spans"]["window_duration_s"] for x in scores]),
        "history_duration_s": distribution([x["time_spans"]["history_duration_s"] for x in scores]),
        "horizon_delta_s": {
            str(h): distribution([x["time_spans"]["horizon_delta_s"][str(h)] for x in scores])
            for h in (1, 2, 4, 8)
        },
    }
    status_counts = {status: sum(job["status"] == status for job in jobs) for status in sorted({job["status"] for job in jobs})}
    diagnostics_by_window = {
        (job["role"], job["source_video_id"], job["window_position"]): job[
            "frontend_diagnostics"
        ]["normalized_aligned_rmse"]
        for job in eligible
    }
    score_residual_correlations = {
        role: {
            predictor: float(
                np.corrcoef(
                    [item["scores"][predictor] for item in scores if item["role"] == role],
                    [
                        diagnostics_by_window[
                            (item["role"], item["source_video_id"], item["window_position"])
                        ]
                        for item in scores
                        if item["role"] == role
                    ],
                )[0, 1]
            )
            for predictor in ("zero", "t0", "spatial")
        }
        for role in ("real_val", "fake_probe")
    }
    config = {
        "source_head": SOURCE_HEAD,
        "selection": "same frozen 32 real-validation and 32 fake-probe source_video_id",
        "window_rule": "s_k=floor(k*(F-16)/4), k=0..4, deterministic deduplication",
        "window_length": 16,
        "history_count": 8,
        "horizons": [1, 2, 4, 8],
        "primary_video_aggregation": "maximum eligible window arithmetic-mean score",
        "annotation_used": False,
        "models_frozen": True,
    }
    write_json(OUTPUT_ROOT / "probe_config.json", config)
    summary = {
        "classification": classify_probe(aurocs, fake_ratio_higher),
        "short_temporal_scale_limitation": "undetermined",
        "video_counts": {role: len(values) for role, values in by_role.items()},
        "window_status_counts": status_counts,
        "exactly_five_window_video_count": sum(v["unique_eligible_windows"] == 5 for v in videos),
        "aurocs": aurocs,
        "learned_contrasts": {
            "t0_max_minus_zero_max": aurocs["t0"]["max"] - aurocs["zero"]["max"],
            "spatial_max_minus_zero_max": aurocs["spatial"]["max"] - aurocs["zero"]["max"],
        },
        "bootstrap": bootstrap,
        "time_spans": time_values,
        "max_to_center_ratios": ratios,
        "max_window_position_counts": positions,
        "score_normalized_alignment_residual_correlations": score_residual_correlations,
        "frontend_edge_instability": "no edge-only increase; diagnostics were not thresholded or filtered",
    }
    write_json(OUTPUT_ROOT / "summary.json", summary)


if __name__ == "__main__":
    main()
