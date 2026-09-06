#!/usr/bin/env python3
"""Audit per-horizon VGGT target context sensitivity without training."""

import json
import os
from pathlib import Path
import time

import numpy as np
import torch

import run_temporal_learnability_probe as temporal_run
from sparse3d_forgery.experiments.prefix_target_audit import (
    HORIZONS,
    PREFIX_INFERENCE_HORIZONS,
    classify_real,
    common_target_measurements,
    compatibility_signature,
    diagnostic_ratio,
    fixed_history_anchor,
    prefix_frame_indices,
    summarize,
    summarize_ratios,
)
from sparse3d_forgery.experiments.temporal_probe import MissingAwareGRU, history_normalize
from sparse3d_forgery.frontend import VggtFrontend, VggtFrontendConfig
from sparse3d_forgery.frontend.vggt import (
    VGGT_CODE_REVISION,
    VGGT_WEIGHT_REVISION,
    VGGT_WEIGHT_SHA256,
    _construct_history_anchored_window,
)
from sparse3d_forgery.particle_sequence import load_particle_sequence
from sparse3d_forgery.video_input import VideoSource, decode_video


SOURCE_HEAD = "8c21aa6f737e815fdf523555f77d0bc05b6e8998"
OUTPUT_ROOT = temporal_run.DATA_ROOT / "derived/prefix_target_frontend_audit_v1"
MULTIWINDOW_ROOT = temporal_run.DATA_ROOT / "derived/multiwindow_coverage_probe_v1"


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def target_prefix(job: dict, horizon: int) -> Path:
    return (
        OUTPUT_ROOT
        / "prefix_targets"
        / job["role"]
        / job["source_video_id"]
        / f"w{job['window_position']}-h{horizon}"
    )


def check_compatibility(sequence) -> None:
    signature = compatibility_signature(sequence)
    expected = {
        "provider": "facebookresearch/vggt",
        "code_revision": VGGT_CODE_REVISION,
        "weight_revision": VGGT_WEIGHT_REVISION,
        "weight_sha256": VGGT_WEIGHT_SHA256,
        "query_initialization": "deterministic uniform grid in first source frame",
        "history_frame_indices": tuple(sequence.frame_indices[:8]),
        "history_count": 8,
        "track_count": 128,
    }
    if signature != expected:
        raise RuntimeError(f"incompatible fixed Pass-A lineage: {signature}")


def save_target(prefix: Path, causal, horizon: int, elapsed: float) -> dict:
    target_index = 7 + horizon
    prefix.parent.mkdir(parents=True, exist_ok=True)
    npz = prefix.with_suffix(".npz")
    metadata = prefix.with_suffix(".json")
    if npz.exists() or metadata.exists():
        raise FileExistsError(prefix)
    with npz.open("xb") as handle:
        np.savez_compressed(
            handle,
            xyz=causal.xyz[target_index],
            validity=causal.geometry_validity[target_index],
        )
    result = {
        "source_video_id": causal.source_video_id,
        "horizon": horizon,
        "target_frame_index": int(causal.frame_indices[target_index]),
        "target_timestamp_s": float(causal.timestamps_s[target_index]),
        "diagnostics": causal.provenance,
        "elapsed_s": elapsed,
    }
    write_json(metadata, result)
    return result


def load_target(prefix: Path) -> tuple[np.ndarray, np.ndarray, dict]:
    with np.load(prefix.with_suffix(".npz"), allow_pickle=False) as arrays:
        xyz = arrays["xyz"].copy()
        validity = arrays["validity"].copy()
    with prefix.with_suffix(".json").open(encoding="utf-8") as handle:
        metadata = json.load(handle)
    return xyz, validity, metadata


def run_prefix(frontend, job: dict, full, horizon: int) -> dict:
    prefix = target_prefix(job, horizon)
    if prefix.with_suffix(".npz").is_file() and prefix.with_suffix(".json").is_file():
        _, _, metadata = load_target(prefix)
        return {"status": "eligible_resumed", "artifact_prefix": str(prefix), **metadata}
    anchor = fixed_history_anchor(full)
    indices = prefix_frame_indices(full.frame_indices, 8, horizon)
    started = time.perf_counter()
    decoded = decode_video(
        VideoSource(
            sample_id=f"prefix-audit-{job['source_video_id']}-w{job['window_position']}-h{horizon}",
            source_video_id=job["source_video_id"],
            source_locator=temporal_run.EXTRACTED_ROOT / job["video_path"],
        ),
        indices,
    )
    extension = frontend.extract(decoded)
    causal = _construct_history_anchored_window(anchor, extension, 8)
    elapsed = time.perf_counter() - started
    if causal.provenance.get("causal_training_eligible") is not True:
        return {
            "status": "causal_ineligible",
            "failure": causal.provenance.get("alignment_failure"),
            "elapsed_s": elapsed,
        }
    metadata = save_target(prefix, causal, horizon, elapsed)
    return {"status": "eligible_generated", "artifact_prefix": str(prefix), **metadata}


def process(jobs: list[dict]) -> None:
    frontend = VggtFrontend(
        temporal_run.WEIGHT_PATH, VggtFrontendConfig(num_tracks=128, target_size=518)
    )
    total = len(jobs) * 3
    number = 0
    for job in jobs:
        full = load_particle_sequence(job["artifact_prefix"])
        check_compatibility(full)
        job.setdefault("prefix_horizons", {})
        for horizon in PREFIX_INFERENCE_HORIZONS:
            number += 1
            try:
                result = run_prefix(frontend, job, full, horizon)
            except Exception as exc:
                result = {"status": "frontend_failure", "failure": f"{type(exc).__name__}: {exc}"}
            job["prefix_horizons"][str(horizon)] = result
            write_json(OUTPUT_ROOT / "window_manifest.json", {"source_head": SOURCE_HEAD, "jobs": jobs})
            print(number, total, job["role"], job["source_video_id"], job["window_position"], horizon, result["status"], flush=True)


def load_t0(device):
    checkpoint = torch.load(temporal_run.OUTPUT_ROOT / "checkpoint.pt", map_location=device, weights_only=False)
    model = MissingAwareGRU(64).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def pearson(x: list[float], y: list[float]) -> float:
    return float(np.corrcoef(np.asarray(x), np.asarray(y))[0, 1])


def evaluate_role(jobs: list[dict], role: str, model, device) -> tuple[dict, dict, dict]:
    role_jobs = [job for job in jobs if job["role"] == role]
    rows = []
    for job in role_jobs:
        full = load_particle_sequence(job["artifact_prefix"])
        normalized = history_normalize(full, 8)
        history = torch.from_numpy(normalized.xyz[:8][None]).to(device)
        history_valid = torch.from_numpy(normalized.validity[:8][None]).to(device)
        with torch.inference_mode():
            prediction = model(history, history_valid)[0].cpu().numpy()
        full_n8 = full.provenance["alignment_diagnostics"]["normalized_aligned_rmse"]
        for offset, horizon in enumerate(HORIZONS):
            full_index = 7 + horizon
            if horizon == 8:
                prefix_xyz = full.xyz[full_index]
                prefix_valid = full.geometry_validity[full_index]
                diagnostics = full.provenance
            else:
                result = job["prefix_horizons"][str(horizon)]
                if not result["status"].startswith("eligible"):
                    continue
                prefix_xyz, prefix_valid, metadata = load_target(Path(result["artifact_prefix"]))
                diagnostics = metadata["diagnostics"]
            compared = common_target_measurements(
                prefix_xyz,
                prefix_valid,
                full.xyz[full_index],
                full.geometry_validity[full_index],
                full.xyz[7],
                full.geometry_validity[7],
            )
            common = compared["motion_validity"]
            if not common.any():
                continue
            prefix_displacement = (prefix_xyz[common] - full.xyz[7, common]) / normalized.rms_radius
            full_displacement = (full.xyz[full_index, common] - full.xyz[7, common]) / normalized.rms_radius
            t0 = prediction[common, offset]
            t0_prefix_error = np.linalg.norm(t0 - prefix_displacement, axis=-1)
            t0_full_error = np.linalg.norm(t0 - full_displacement, axis=-1)
            zero_prefix_error = np.linalg.norm(prefix_displacement, axis=-1)
            zero_full_error = np.linalg.norm(full_displacement, axis=-1)
            motion_rms = float(np.sqrt(np.mean(compared["motion"] ** 2)))
            motion_norm_rms = motion_rms / normalized.rms_radius
            n_h = diagnostics["alignment_diagnostics"]["normalized_aligned_rmse"]
            rows.append(
                {
                    "source_video_id": job["source_video_id"],
                    "window_position": job["window_position"],
                    "horizon": horizon,
                    "common_target_count": int(compared["validity"].sum()),
                    "common_motion_count": int(common.sum()),
                    "n_h": n_h,
                    "n_8": full_n8,
                    "disagreement": compared["disagreement"].tolist(),
                    "disagreement_normalized": (
                        compared["disagreement"] / normalized.rms_radius
                    ).tolist(),
                    "motion": compared["motion"].tolist(),
                    "motion_normalized": (compared["motion"] / normalized.rms_radius).tolist(),
                    "q_target": compared["q_target"],
                    "q_history": diagnostic_ratio(n_h, motion_norm_rms),
                    "t0_error_full_mean": float(t0_full_error.mean()),
                    "t0_error_prefix_mean": float(t0_prefix_error.mean()),
                    "zero_error_full_mean": float(zero_full_error.mean()),
                    "zero_error_prefix_mean": float(zero_prefix_error.mean()),
                    "alignment": diagnostics,
                }
            )
    geometry = {}
    context = {}
    model_metrics = {"t0": {}, "zero": {}}
    for horizon in HORIZONS:
        selected = [row for row in rows if row["horizon"] == horizon]
        key = str(horizon)
        geometry[key] = {
            "window_count": len(selected),
            "normalized_aligned_rmse": summarize([row["n_h"] for row in selected]),
            "relative_median_reduction_vs_n8": 1.0
            - float(np.median([row["n_h"] for row in selected]))
            / float(np.median([row["n_8"] for row in selected])),
            "common_history_count": summarize(
                [row["alignment"]["common_historical_correspondence_count"] for row in selected]
            ),
            "sim3_scale": summarize([row["alignment"]["estimated_scale"] for row in selected]),
            "rotation_determinant": summarize(
                [row["alignment"]["rotation_determinant"] for row in selected]
            ),
            "raw_history_residual_mean": summarize(
                [row["alignment"]["alignment_diagnostics"]["raw_residual_mean"] for row in selected]
            ),
            "raw_history_residual_rmse": {
                "status": "not_recorded_by_existing_accepted_alignment_diagnostics"
            },
            "raw_history_residual_max": summarize(
                [row["alignment"]["alignment_diagnostics"]["raw_residual_max"] for row in selected]
            ),
            "aligned_history_residual_mean": summarize(
                [row["alignment"]["alignment_diagnostics"]["aligned_residual_mean"] for row in selected]
            ),
            "aligned_history_residual_rmse": summarize(
                [row["alignment"]["alignment_diagnostics"]["aligned_residual_rmse"] for row in selected]
            ),
            "aligned_history_residual_max": summarize(
                [row["alignment"]["alignment_diagnostics"]["aligned_residual_max"] for row in selected]
            ),
        }
        raw_disagreements = [value for row in selected for value in row["disagreement"]]
        disagreements = [value for row in selected for value in row["disagreement_normalized"]]
        raw_motions = [value for row in selected for value in row["motion"]]
        motions = [value for row in selected for value in row["motion_normalized"]]
        context[key] = {
            "common_target_count": sum(row["common_target_count"] for row in selected),
            "common_motion_count": sum(row["common_motion_count"] for row in selected),
            "target_disagreement": summarize(raw_disagreements),
            "target_disagreement_normalized": summarize(disagreements),
            "observed_motion": summarize(raw_motions),
            "observed_motion_normalized": summarize(motions),
            "q_target": summarize_ratios([row["q_target"] for row in selected if row["q_target"] is not None]),
            "q_history": summarize_ratios([row["q_history"] for row in selected if row["q_history"] is not None]),
        }
        for predictor in ("t0", "zero"):
            model_metrics[predictor][key] = {
                "full_error_residual_pearson": pearson(
                    [row[f"{predictor}_error_full_mean"] for row in selected],
                    [row["n_8"] for row in selected],
                ),
                "prefix_error_residual_pearson": pearson(
                    [row[f"{predictor}_error_prefix_mean"] for row in selected],
                    [row["n_h"] for row in selected],
                ),
            }
    return geometry, context, model_metrics


def repeatability(frontend, jobs: list[dict]) -> dict:
    records = []
    selected = []
    for role in ("real_val", "fake_probe"):
        centers = sorted(
            [job for job in jobs if job["role"] == role and job["window_position"] == 2],
            key=lambda job: job["source_video_id"],
        )[:4]
        selected.extend(centers)
    for job in selected:
        full = load_particle_sequence(job["artifact_prefix"])
        first_xyz, first_valid, _ = load_target(target_prefix(job, 1))
        anchor = fixed_history_anchor(full)
        indices = prefix_frame_indices(full.frame_indices, 8, 1)
        decoded = decode_video(
            VideoSource(
                sample_id=f"repeat-{job['source_video_id']}",
                source_video_id=job["source_video_id"],
                source_locator=temporal_run.EXTRACTED_ROOT / job["video_path"],
            ),
            indices,
        )
        repeated = _construct_history_anchored_window(anchor, frontend.extract(decoded), 8)
        repeated_xyz = repeated.xyz[8]
        repeated_valid = repeated.geometry_validity[8]
        common_repeat = first_valid & repeated_valid
        repeat_error = np.linalg.norm(first_xyz[common_repeat] - repeated_xyz[common_repeat], axis=-1)
        compared = common_target_measurements(
            first_xyz,
            first_valid,
            full.xyz[8],
            full.geometry_validity[8],
            full.xyz[7],
            full.geometry_validity[7],
        )
        records.append(
            {
                "role": job["role"],
                "source_video_id": job["source_video_id"],
                "repeat_common_count": int(common_repeat.sum()),
                "repeat_disagreement_rmse": float(np.sqrt(np.mean(repeat_error**2))),
                "context_disagreement_rmse": float(
                    np.sqrt(np.mean(compared["disagreement"] ** 2))
                ),
            }
        )
    ratios = [
        row["repeat_disagreement_rmse"] / row["context_disagreement_rmse"]
        for row in records
        if row["context_disagreement_rmse"] > 1e-12
    ]
    # A repeat/context ratio near one would prevent a context-only interpretation.
    blocker = bool(ratios and np.median(ratios) >= 0.5)
    return {"records": records, "repeat_to_context_ratio": summarize(ratios), "blocker": blocker}


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    manifest_path = OUTPUT_ROOT / "window_manifest.json"
    if manifest_path.exists():
        with manifest_path.open(encoding="utf-8") as handle:
            saved = json.load(handle)
        if saved.get("source_head") != SOURCE_HEAD:
            raise RuntimeError("existing audit source HEAD differs")
        jobs = saved["jobs"]
    else:
        with (MULTIWINDOW_ROOT / "window_manifest.json").open(encoding="utf-8") as handle:
            prior = json.load(handle)
        jobs = [job for job in prior["jobs"] if job["status"].startswith("eligible")]
        if len(jobs) != 320:
            raise RuntimeError("expected exactly 320 fixed eligible windows")
        write_json(manifest_path, {"source_head": SOURCE_HEAD, "jobs": jobs})

    started = time.perf_counter()
    process(jobs)
    device = torch.device("cuda")
    model = load_t0(device)
    real_geometry, real_context, real_model = evaluate_role(jobs, "real_val", model, device)
    write_json(OUTPUT_ROOT / "real_geometry_metrics.json", real_geometry)
    write_json(OUTPUT_ROOT / "real_target_context_metrics.json", real_context)
    write_json(OUTPUT_ROOT / "real_model_diagnostics.json", real_model)

    fake_geometry, fake_context, fake_model = evaluate_role(jobs, "fake_probe", model, device)
    write_json(OUTPUT_ROOT / "fake_geometry_metrics.json", fake_geometry)
    write_json(OUTPUT_ROOT / "fake_target_context_metrics.json", fake_context)
    write_json(OUTPUT_ROOT / "fake_model_diagnostics.json", fake_model)

    frontend = VggtFrontend(
        temporal_run.WEIGHT_PATH, VggtFrontendConfig(num_tracks=128, target_size=518)
    )
    repeat = repeatability(frontend, jobs)
    write_json(OUTPUT_ROOT / "repeatability_metrics.json", repeat)
    classification_input = {
        "alignment": {key: value["normalized_aligned_rmse"] for key, value in real_geometry.items()},
        "q_target": {key: value["q_target"] for key, value in real_context.items()},
        "correlation": {
            key: {
                "full": real_model["t0"][key]["full_error_residual_pearson"],
                "prefix": real_model["t0"][key]["prefix_error_residual_pearson"],
            }
            for key in ("1", "2", "4")
        },
    }
    statuses = [result["status"] for job in jobs for result in job["prefix_horizons"].values()]
    prefix_inference_runtime = sum(
        result.get("elapsed_s", 0.0)
        for job in jobs
        for result in job["prefix_horizons"].values()
        if result["status"].startswith("eligible")
    )
    elapsed = time.perf_counter() - started
    peak = max(
        result.get("diagnostics", {}).get("peak_gpu_memory_bytes", 0)
        for job in jobs
        for result in job["prefix_horizons"].values()
    )
    config = {
        "source_head": SOURCE_HEAD,
        "dataset_revision": temporal_run.DATASET_REVISION,
        "reused_multiwindow_manifest": str(MULTIWINDOW_ROOT / "window_manifest.json"),
        "real_primary_windows": 160,
        "fake_supplementary_windows": 160,
        "horizons": list(HORIZONS),
        "pass_a_rerun": False,
        "b8_rerun": False,
        "annotation_used": False,
        "model_training": False,
        "new_b1_b2_b4_inference_count": len(jobs) * len(PREFIX_INFERENCE_HORIZONS),
    }
    write_json(OUTPUT_ROOT / "audit_config.json", config)
    write_json(
        OUTPUT_ROOT / "summary.json",
        {
            "classification": classify_real(classification_input, repeat["blocker"]),
            "repeatability_blocker": repeat["blocker"],
            "prefix_status_counts": {status: statuses.count(status) for status in sorted(set(statuses))},
            "b8_reused_count": len(jobs),
            "runtime_s": elapsed,
            "prefix_inference_runtime_s": prefix_inference_runtime,
            "peak_gpu_memory_bytes": peak,
            "real_primary": classification_input,
            "fake_supplementary_same_direction": {
                "alignment": {key: value["normalized_aligned_rmse"] for key, value in fake_geometry.items()},
                "q_target": {key: value["q_target"] for key, value in fake_context.items()},
                "correlation": fake_model,
            },
        },
    )


if __name__ == "__main__":
    main()
