#!/usr/bin/env python3
"""Run the fixed, bounded temporal-only real-normal learnability probe."""

from copy import deepcopy
import json
from pathlib import Path
import random
import time

import numpy as np
import torch

from sparse3d_forgery.experiments.temporal_probe import (
    HORIZONS,
    MissingAwareGRU,
    assert_disjoint_source_ids,
    build_targets,
    history_normalize,
    linear_extrapolation,
    masked_mse,
    rank_auroc,
    zero_displacement,
)
from sparse3d_forgery.frontend import VggtFrontend, VggtFrontendConfig
from sparse3d_forgery.particle_sequence import load_particle_sequence, save_particle_sequence
from sparse3d_forgery.video_input import VideoSource, decode_video


SEED = 20260906
HISTORY_COUNT = 8
WINDOW_FRAMES = 16
DATASET_REVISION = "92e76e78e8c90a1ff7ec9354bee44eb024265e79"
GIT_BASELINE = "f9fbed0ad554732a5d6e249e30afb068ad2a973e"
DATA_ROOT = Path("/root/autodl-tmp/data/sparse_3d_forgery_detection")
EXTRACTED_ROOT = DATA_ROOT / "extracted/deeptrace_reward" / DATASET_REVISION
METADATA_PATH = DATA_ROOT / "raw/deeptrace_reward" / DATASET_REVISION / "all_data.json"
WEIGHT_PATH = DATA_ROOT / "external/vggt-weights/model.pt"
OUTPUT_ROOT = DATA_ROOT / "derived/temporal_learnability_probe_v1"


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        with path.open(encoding="utf-8") as handle:
            if json.load(handle) == value:
                return
        raise FileExistsError(f"existing result differs: {path}")
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def select_rows(
    rows: list[dict], split: str, video_type: str, limit: int
) -> tuple[list[dict], int, int, int]:
    candidates = {}
    short = 0
    missing = 0
    for row in rows:
        if row.get("split") != split or row.get("video_type") != video_type:
            continue
        if int(row.get("video_frame_count", 0)) < WINDOW_FRAMES:
            short += 1
            continue
        path = EXTRACTED_ROOT / row["video_path"]
        if not path.is_file():
            missing += 1
            continue
        candidates.setdefault(row["video_id"], row)
    return [candidates[key] for key in sorted(candidates)[:limit]], len(candidates), short, missing


def manifest_item(row: dict, role: str) -> dict:
    frame_count = int(row["video_frame_count"])
    start = (frame_count - WINDOW_FRAMES) // 2
    return {
        "role": role,
        "source_video_id": row["video_id"],
        "video_path": row["video_path"],
        "official_split": row["split"],
        "declared_frame_count": frame_count,
        "frame_indices": list(range(start, start + WINDOW_FRAMES)),
        "status": "selected",
    }


def artifact_prefix(item: dict) -> Path:
    return OUTPUT_ROOT / "causal_artifacts" / item["role"] / item["source_video_id"]


def preprocess(frontend: VggtFrontend, items: list[dict]) -> dict:
    counts = {"selected": len(items), "eligible": 0, "ineligible": 0, "frontend_failure": 0}
    for number, item in enumerate(items, start=1):
        started = time.perf_counter()
        try:
            prefix = artifact_prefix(item)
            if prefix.with_suffix(".npz").is_file() and prefix.with_suffix(".json").is_file():
                existing = load_particle_sequence(prefix)
                if (
                    existing.source_video_id != item["source_video_id"]
                    or not np.array_equal(existing.frame_indices, item["frame_indices"])
                    or existing.provenance.get("causal_training_eligible") is not True
                ):
                    raise RuntimeError("existing causal artifact does not match the selected window")
                item["status"] = "eligible"
                item["artifact_prefix"] = str(prefix.relative_to(OUTPUT_ROOT))
                item["frontend_elapsed_s"] = 0.0
                item["reused_from_interrupted_probe"] = True
                counts["eligible"] += 1
                print(
                    item["role"], number, len(items), item["source_video_id"], "eligible-reused", flush=True
                )
                continue
            decoded = decode_video(
                VideoSource(
                    sample_id=f"temporal-probe-{item['source_video_id']}",
                    source_video_id=item["source_video_id"],
                    source_locator=EXTRACTED_ROOT / item["video_path"],
                ),
                item["frame_indices"],
            )
            sequence = frontend.extract_causal_window(decoded, HISTORY_COUNT)
            if sequence.provenance["causal_training_eligible"] is not True:
                item["status"] = "ineligible"
                item["failure"] = sequence.provenance.get("alignment_failure")
                counts["ineligible"] += 1
            else:
                save_particle_sequence(sequence, artifact_prefix(item))
                item["status"] = "eligible"
                item["artifact_prefix"] = str(artifact_prefix(item).relative_to(OUTPUT_ROOT))
                item["frontend_elapsed_s"] = time.perf_counter() - started
                counts["eligible"] += 1
        except Exception as exc:
            item["status"] = "frontend_failure"
            item["failure"] = f"{type(exc).__name__}: {exc}"
            counts["frontend_failure"] += 1
        print(
            item["role"],
            number,
            len(items),
            item["source_video_id"],
            item["status"],
            item.get("failure", ""),
            flush=True,
        )
    return counts


def load_views(items: list[dict]) -> list[dict]:
    views = []
    for item in items:
        if item["status"] != "eligible":
            continue
        sequence = load_particle_sequence(OUTPUT_ROOT / item["artifact_prefix"])
        try:
            normalized = history_normalize(sequence, HISTORY_COUNT)
            targets, target_validity = build_targets(
                normalized.xyz, normalized.validity, HISTORY_COUNT, HORIZONS
            )
        except ValueError as exc:
            item["probe_view_status"] = "normalization_failure"
            item["probe_view_failure"] = str(exc)
            continue
        if not target_validity.any():
            item["probe_view_status"] = "no_valid_prediction_targets"
            continue
        item["probe_view_status"] = "usable"
        views.append(
            {
                "source_video_id": sequence.source_video_id,
                "xyz": normalized.xyz,
                "validity": normalized.validity,
                "timestamps_s": sequence.timestamps_s,
                "targets": targets,
                "target_validity": target_validity,
                "centroid": normalized.centroid,
                "rms_radius": normalized.rms_radius,
            }
        )
    return views


def batch_tensors(batch: list[dict], device: torch.device):
    history_xyz = torch.from_numpy(np.stack([x["xyz"][:HISTORY_COUNT] for x in batch])).to(device)
    history_validity = torch.from_numpy(
        np.stack([x["validity"][:HISTORY_COUNT] for x in batch])
    ).to(device)
    target_validity_np = np.stack([x["target_validity"] for x in batch])
    targets_np = np.stack([x["targets"] for x in batch]).copy()
    targets_np[~target_validity_np] = 0.0
    targets = torch.from_numpy(targets_np).to(device)
    target_validity = torch.from_numpy(target_validity_np).to(device)
    return history_xyz, history_validity, targets, target_validity


def predict_gru(model: MissingAwareGRU, views: list[dict], device: torch.device) -> list[np.ndarray]:
    predictions = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(views), 8):
            batch = views[start : start + 8]
            history_xyz, history_validity, _, _ = batch_tensors(batch, device)
            predictions.extend(model(history_xyz, history_validity).cpu().numpy())
    return predictions


def predictor_metrics(views: list[dict], predictions: list[np.ndarray], kind: str) -> tuple[dict, list[dict]]:
    errors_by_horizon = [[] for _ in HORIZONS]
    squared_by_horizon = [[] for _ in HORIZONS]
    clip_metrics = []
    for view, prediction in zip(views, predictions):
        mask = view["target_validity"].copy()
        if kind == "linear":
            prediction, coverage = linear_extrapolation(
                view["xyz"], view["validity"], view["timestamps_s"], HISTORY_COUNT, HORIZONS
            )
            mask &= coverage
        difference = prediction - view["targets"]
        clip_errors = []
        for offset in range(len(HORIZONS)):
            valid = mask[:, offset]
            error = np.linalg.norm(difference[valid, offset], axis=-1).astype(np.float64)
            squared = (difference[valid, offset].astype(np.float64) ** 2).reshape(-1)
            errors_by_horizon[offset].append(error)
            squared_by_horizon[offset].append(squared)
            clip_errors.append(error)
        nonempty = [x for x in clip_errors if x.size]
        if not nonempty:
            continue
        combined = np.concatenate(nonempty)
        clip_metrics.append(
            {
                "source_video_id": view["source_video_id"],
                "score_mean": float(combined.mean()),
                "score_median": float(np.median(combined)),
                "score_p90": float(np.percentile(combined, 90)),
                "valid_particle_horizon_count": int(combined.size),
            }
        )
    per_horizon = {}
    all_errors = []
    all_squared = []
    for offset, horizon in enumerate(HORIZONS):
        errors = np.concatenate(errors_by_horizon[offset])
        squared = np.concatenate(squared_by_horizon[offset])
        all_errors.append(errors)
        all_squared.append(squared)
        per_horizon[str(horizon)] = {
            "valid_particle_target_count": int(errors.size),
            "masked_mse": float(squared.mean()),
            "particle_l2_mean": float(errors.mean()),
            "particle_l2_median": float(np.median(errors)),
            "particle_l2_rmse": float(np.sqrt(np.mean(errors**2))),
            "particle_l2_p90": float(np.percentile(errors, 90)),
        }
    errors = np.concatenate(all_errors)
    squared = np.concatenate(all_squared)
    metrics = {
        "per_horizon": per_horizon,
        "overall": {
            "valid_particle_target_count": int(errors.size),
            "masked_mse": float(squared.mean()),
            "particle_l2_mean": float(errors.mean()),
            "particle_l2_median": float(np.median(errors)),
            "particle_l2_rmse": float(np.sqrt(np.mean(errors**2))),
            "particle_l2_p90": float(np.percentile(errors, 90)),
        },
    }
    return metrics, clip_metrics


def evaluate(model: MissingAwareGRU, views: list[dict], device: torch.device) -> tuple[dict, dict]:
    gru_predictions = predict_gru(model, views, device)
    zero_predictions = [zero_displacement(view["xyz"].shape[1]) for view in views]
    zero_metrics, zero_clips = predictor_metrics(views, zero_predictions, "zero")
    linear_metrics, linear_clips = predictor_metrics(views, zero_predictions, "linear")
    gru_metrics, gru_clips = predictor_metrics(views, gru_predictions, "gru")
    return (
        {"zero": zero_metrics, "linear": linear_metrics, "gru": gru_metrics},
        {"zero": zero_clips, "linear": linear_clips, "gru": gru_clips},
    )


def train(real_train: list[dict], real_val: list[dict], device: torch.device):
    model = MissingAwareGRU(hidden_dim=64).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    best_state = None
    best_epoch = 0
    best_val_mse = float("inf")
    epochs_without_improvement = 0
    epoch_history = []
    generator = torch.Generator().manual_seed(SEED)
    for epoch in range(1, 31):
        model.train()
        permutation = torch.randperm(len(real_train), generator=generator).tolist()
        train_losses = []
        for start in range(0, len(permutation), 8):
            batch = [real_train[index] for index in permutation[start : start + 8]]
            history_xyz, history_validity, targets, target_validity = batch_tensors(batch, device)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(history_xyz, history_validity)
            loss = masked_mse(prediction, targets, target_validity)
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.detach().cpu()))
        validation, _ = evaluate(model, real_val, device)
        val_mse = validation["gru"]["overall"]["masked_mse"]
        epoch_history.append(
            {"epoch": epoch, "train_batch_mse": float(np.mean(train_losses)), "real_val_mse": val_mse}
        )
        print("epoch", epoch, "train", np.mean(train_losses), "val", val_mse, flush=True)
        if val_mse < best_val_mse:
            best_val_mse = val_mse
            best_epoch = epoch
            best_state = deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= 5:
                break
    model.load_state_dict(best_state)
    return model, {"best_epoch": best_epoch, "best_real_val_mse": best_val_mse, "epochs": epoch_history}


def fake_comparison(real_clips: dict, fake_clips: dict) -> dict:
    result = {}
    for predictor in ("zero", "linear", "gru"):
        real = np.array([x["score_mean"] for x in real_clips[predictor]])
        fake = np.array([x["score_mean"] for x in fake_clips[predictor]])
        threshold = float(np.percentile(real, 95))
        result[predictor] = {
            "auroc": rank_auroc(real, fake),
            "median_fake_to_real_ratio": float(np.median(fake) / np.median(real)),
            "real_val_p95": threshold,
            "fake_exceedance_rate": float(np.mean(fake > threshold)),
        }
    return result


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(False)
    device = torch.device("cuda")

    with METADATA_PATH.open(encoding="utf-8") as handle:
        rows = json.load(handle)
    train_rows, train_candidates, train_short, train_missing = select_rows(
        rows, "train", "real_video", 128
    )
    val_rows, val_candidates, val_short, val_missing = select_rows(rows, "val", "real_video", 32)
    fake_rows, fake_candidates, fake_short, fake_missing = select_rows(rows, "val", "fake_video", 32)
    train_items = [manifest_item(row, "real_train") for row in train_rows]
    val_items = [manifest_item(row, "real_val") for row in val_rows]
    fake_items = [manifest_item(row, "fake_probe") for row in fake_rows]
    assert_disjoint_source_ids(
        [x["source_video_id"] for x in train_items],
        [x["source_video_id"] for x in val_items],
    )

    config = {
        "seed": SEED,
        "git_baseline": GIT_BASELINE,
        "dataset_revision": DATASET_REVISION,
        "selection": "lexicographically first unique source_video_id within official split/type",
        "window": "center 16 declared source frames",
        "history_count": HISTORY_COUNT,
        "horizons_observation_offset": list(HORIZONS),
        "num_tracks": 128,
        "provider_input_size": 518,
        "normalization": "history-valid centroid and RMS radius; (X-c_H)/r_H",
        "model": {"type": "single-layer GRUCell", "input_dim": 3, "hidden_dim": 64},
        "optimizer": "AdamW",
        "learning_rate": 1e-3,
        "weight_decay": 1e-4,
        "batch_size_windows": 8,
        "max_epochs": 30,
        "early_stopping_patience": 5,
        "early_stopping_metric": "real validation masked MSE",
        "seed_settings": "Python, NumPy, Torch CPU/CUDA",
        "torch_deterministic_algorithms": False,
        "determinism_note": "strict CUDA algorithms disabled because VGGT cuBLAS operations require external workspace configuration",
    }
    write_json(OUTPUT_ROOT / "probe_config.json", config)
    manifest = {
        "selection_counts": {
            "real_train_candidates": train_candidates,
            "real_val_candidates": val_candidates,
            "fake_probe_candidates": fake_candidates,
            "real_train_selected": len(train_rows),
            "real_val_selected": len(val_rows),
            "fake_probe_selected": len(fake_rows),
            "excluded_short": {"real_train": train_short, "real_val": val_short, "fake_probe": fake_short},
            "missing_media": {"real_train": train_missing, "real_val": val_missing, "fake_probe": fake_missing},
        },
        "items": train_items + val_items + fake_items,
    }

    frontend = VggtFrontend(WEIGHT_PATH, VggtFrontendConfig(num_tracks=128, target_size=518))
    manifest["real_train_preprocessing"] = preprocess(frontend, train_items)
    manifest["real_val_preprocessing"] = preprocess(frontend, val_items)
    train_views = load_views(train_items)
    val_views = load_views(val_items)
    if not train_views or not val_views:
        raise RuntimeError("eligible real train and validation windows are required")

    model, training = train(train_views, val_views, device)
    train_metrics, train_clips = evaluate(model, train_views, device)
    val_metrics, val_clips = evaluate(model, val_views, device)
    write_json(OUTPUT_ROOT / "real_train_metrics.json", {"metrics": train_metrics, "clips": train_clips})
    write_json(OUTPUT_ROOT / "real_val_metrics.json", {"metrics": val_metrics, "clips": val_clips})
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": config,
            "best_epoch": training["best_epoch"],
            "best_real_val_mse": training["best_real_val_mse"],
        },
        OUTPUT_ROOT / "checkpoint.pt",
    )
    train_rmse = train_metrics["gru"]["overall"]["particle_l2_rmse"]
    val_rmse = val_metrics["gru"]["overall"]["particle_l2_rmse"]
    zero_val_rmse = val_metrics["zero"]["overall"]["particle_l2_rmse"]
    gate = val_rmse < zero_val_rmse and val_rmse / train_rmse <= 1.5
    summary = {
        "training": training,
        "real_train_count": len(train_views),
        "real_val_count": len(val_views),
        "gru_real_train_rmse": train_rmse,
        "gru_real_val_rmse": val_rmse,
        "val_train_ratio": val_rmse / train_rmse,
        "gru_improvement_vs_zero_real_val": 1.0 - val_rmse / zero_val_rmse,
        "gru_improvement_vs_linear_real_val": 1.0
        - val_rmse / val_metrics["linear"]["overall"]["particle_l2_rmse"],
        "stage_d_gate_passed": gate,
    }

    if gate:
        manifest["fake_probe_preprocessing"] = preprocess(frontend, fake_items)
        fake_views = load_views(fake_items)
        fake_metrics, fake_clips = evaluate(model, fake_views, device)
        comparison = fake_comparison(val_clips, fake_clips)
        write_json(
            OUTPUT_ROOT / "fake_probe_metrics.json",
            {"metrics": fake_metrics, "clips": fake_clips, "real_val_comparison": comparison},
        )
        summary["fake_probe_count"] = len(fake_views)
        summary["fake_comparison"] = comparison
    else:
        summary["fake_probe_count"] = 0
        summary["fake_probe_not_run_reason"] = "pre-registered real-only learnability gate failed"
    write_json(OUTPUT_ROOT / "pilot_manifest.json", manifest)
    write_json(OUTPUT_ROOT / "summary.json", summary)


if __name__ == "__main__":
    main()
