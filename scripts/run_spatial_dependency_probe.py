#!/usr/bin/env python3
"""Run the fixed dense spatial-dependency falsification probe."""

from copy import deepcopy
import json
from pathlib import Path
import random

import numpy as np
import torch

import run_temporal_learnability_probe as temporal_run
from sparse3d_forgery.experiments.spatial_probe import (
    SelfTemporalModel,
    SpatialTemporalModel,
    assert_real_only_training,
    auroc_delta_bootstrap,
    paired_mean_bootstrap,
    trainable_parameter_count,
)
from sparse3d_forgery.experiments.temporal_probe import MissingAwareGRU, masked_mse, rank_auroc


SEED = 20260906
SOURCE_HEAD = "f24ca1566b010fecd39ebda8714c50e5121563e6"
TEMPORAL_ROOT = temporal_run.OUTPUT_ROOT
OUTPUT_ROOT = temporal_run.DATA_ROOT / "derived/spatial_dependency_probe_v1"


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def load_role_views(manifest: dict, role: str) -> list[dict]:
    items = [item.copy() for item in manifest["items"] if item["role"] == role]
    views = temporal_run.load_views(items)
    for view in views:
        view["role"] = role
    return views


def predict(model: torch.nn.Module, views: list[dict], device: torch.device) -> list[np.ndarray]:
    predictions = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(views), 8):
            history_xyz, history_validity, _, _ = temporal_run.batch_tensors(
                views[start : start + 8], device
            )
            predictions.extend(model(history_xyz, history_validity).cpu().numpy())
    return predictions


def train_model(model: torch.nn.Module, train_views: list[dict], val_views: list[dict], device):
    assert_real_only_training([view["role"] for view in train_views])
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    generator = torch.Generator().manual_seed(SEED)
    best_state = None
    best_epoch = 0
    best_val_mse = float("inf")
    stale = 0
    epochs = []
    for epoch in range(1, 31):
        model.train()
        order = torch.randperm(len(train_views), generator=generator).tolist()
        losses = []
        for start in range(0, len(order), 8):
            batch = [train_views[index] for index in order[start : start + 8]]
            history_xyz, history_validity, targets, target_validity = temporal_run.batch_tensors(
                batch, device
            )
            optimizer.zero_grad(set_to_none=True)
            loss = masked_mse(model(history_xyz, history_validity), targets, target_validity)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        val_predictions = predict(model, val_views, device)
        val_metrics, _ = temporal_run.predictor_metrics(val_views, val_predictions, "learned")
        val_mse = val_metrics["overall"]["masked_mse"]
        epochs.append(
            {"epoch": epoch, "train_batch_mse": float(np.mean(losses)), "real_val_mse": val_mse}
        )
        print(type(model).__name__, epoch, np.mean(losses), val_mse, flush=True)
        if val_mse < best_val_mse:
            best_val_mse = val_mse
            best_epoch = epoch
            best_state = deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
            if stale >= 5:
                break
    model.load_state_dict(best_state)
    return {"best_epoch": best_epoch, "best_real_val_mse": best_val_mse, "epochs": epochs}


def all_metrics(models: dict, views: list[dict], device):
    predictions = {
        "zero": [temporal_run.zero_displacement(view["xyz"].shape[1]) for view in views],
        "t0": predict(models["t0"], views, device),
        "t1": predict(models["t1"], views, device),
        "spatial": predict(models["spatial"], views, device),
    }
    metrics = {}
    clips = {}
    metrics["zero"], clips["zero"] = temporal_run.predictor_metrics(
        views, predictions["zero"], "zero"
    )
    metrics["linear"], clips["linear"] = temporal_run.predictor_metrics(
        views, predictions["zero"], "linear"
    )
    for name in ("t0", "t1", "spatial"):
        metrics[name], clips[name] = temporal_run.predictor_metrics(
            views, predictions[name], "learned"
        )
    return metrics, clips


def clip_score_array(clips: dict, predictor: str, ids: list[str]) -> np.ndarray:
    by_id = {item["source_video_id"]: item["score_mean"] for item in clips[predictor]}
    return np.array([by_id[source_id] for source_id in ids], dtype=np.float64)


def fake_summary(real_clips: dict, fake_clips: dict, real_ids: list[str], fake_ids: list[str]):
    result = {}
    arrays = {}
    for predictor in ("zero", "linear", "t0", "t1", "spatial"):
        real = clip_score_array(real_clips, predictor, real_ids)
        fake = clip_score_array(fake_clips, predictor, fake_ids)
        arrays[predictor] = (real, fake)
        threshold = float(np.percentile(real, 95))
        result[predictor] = {
            "auroc": rank_auroc(real, fake),
            "median_fake_to_real_ratio": float(np.median(fake) / np.median(real)),
            "real_val_p95": threshold,
            "fake_exceedance_rate": float(np.mean(fake > threshold)),
        }
    return result, arrays


def main() -> None:
    if OUTPUT_ROOT.exists():
        raise FileExistsError(OUTPUT_ROOT)
    OUTPUT_ROOT.mkdir(parents=True)
    with (TEMPORAL_ROOT / "pilot_manifest.json").open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    train_views = load_role_views(manifest, "real_train")
    val_views = load_role_views(manifest, "real_val")
    if len(train_views) != 126 or len(val_views) != 32:
        raise RuntimeError("reused usable real window counts do not match the frozen probe")
    assert_real_only_training([view["role"] for view in train_views])

    device = torch.device("cuda")
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(False)
    t0 = MissingAwareGRU(hidden_dim=64).to(device)
    checkpoint = torch.load(TEMPORAL_ROOT / "checkpoint.pt", map_location=device, weights_only=False)
    t0.load_state_dict(checkpoint["model_state_dict"])
    t0.eval()

    torch.manual_seed(SEED)
    t1 = SelfTemporalModel(hidden_dim=64, spatial_dim=32)
    t1_training = train_model(t1, train_views, val_views, device)
    torch.manual_seed(SEED)
    spatial = SpatialTemporalModel(hidden_dim=64, spatial_dim=32)
    spatial_training = train_model(spatial, train_views, val_views, device)
    models = {"t0": t0, "t1": t1, "spatial": spatial}
    parameters = {name: trainable_parameter_count(model) for name, model in models.items()}

    torch.save(
        {"model_state_dict": t1.state_dict(), "training": t1_training, "source_head": SOURCE_HEAD},
        OUTPUT_ROOT / "model_t1.pt",
    )
    torch.save(
        {
            "model_state_dict": spatial.state_dict(),
            "training": spatial_training,
            "source_head": SOURCE_HEAD,
        },
        OUTPUT_ROOT / "model_spatial.pt",
    )
    real_train_metrics, _ = all_metrics(models, train_views, device)
    real_val_metrics, real_val_clips = all_metrics(models, val_views, device)
    write_json(
        OUTPUT_ROOT / "real_metrics.json",
        {"train": real_train_metrics, "validation": real_val_metrics},
    )

    val_ids = [view["source_video_id"] for view in val_views]
    paired = {
        "spatial_minus_t0": paired_mean_bootstrap(
            clip_score_array(real_val_clips, "spatial", val_ids),
            clip_score_array(real_val_clips, "t0", val_ids),
            SEED,
        ),
        "spatial_minus_t1": paired_mean_bootstrap(
            clip_score_array(real_val_clips, "spatial", val_ids),
            clip_score_array(real_val_clips, "t1", val_ids),
            SEED,
        ),
    }

    # Fake artifacts are read only after both real-only checkpoints have been selected and saved.
    fake_views = load_role_views(manifest, "fake_probe")
    if len(fake_views) != 32:
        raise RuntimeError("reused fake window count does not match the frozen probe")
    fake_metrics, fake_clips = all_metrics(models, fake_views, device)
    fake_ids = [view["source_video_id"] for view in fake_views]
    fake_comparison, score_arrays = fake_summary(
        real_val_clips, fake_clips, val_ids, fake_ids
    )
    write_json(
        OUTPUT_ROOT / "fake_metrics.json",
        {"metrics": fake_metrics, "real_val_comparison": fake_comparison},
    )

    bootstrap = {"paired_real_validation": paired, "auroc_delta": {}}
    for baseline in ("t0", "t1", "zero"):
        bootstrap["auroc_delta"][f"spatial_minus_{baseline}"] = auroc_delta_bootstrap(
            score_arrays["spatial"][0],
            score_arrays["spatial"][1],
            score_arrays[baseline][0],
            score_arrays[baseline][1],
            SEED,
        )
    write_json(OUTPUT_ROOT / "bootstrap_metrics.json", bootstrap)

    t0_rmse = real_val_metrics["t0"]["overall"]["particle_l2_rmse"]
    t1_rmse = real_val_metrics["t1"]["overall"]["particle_l2_rmse"]
    spatial_train_rmse = real_train_metrics["spatial"]["overall"]["particle_l2_rmse"]
    spatial_rmse = real_val_metrics["spatial"]["overall"]["particle_l2_rmse"]
    spatial_fake = fake_comparison["spatial"]
    pass_conditions = {
        "real_val_within_t0_2_percent": spatial_rmse <= 1.02 * t0_rmse,
        "paired_fraction_better_at_least_55_percent": paired["spatial_minus_t0"][
            "fraction_candidate_better"
        ]
        >= 0.55,
        "spatial_auroc_at_least_0_60": spatial_fake["auroc"] >= 0.60,
        "auroc_margin_at_least_0_05": spatial_fake["auroc"]
        - max(fake_comparison["t0"]["auroc"], fake_comparison["zero"]["auroc"])
        >= 0.05,
        "median_fake_to_real_above_one": spatial_fake["median_fake_to_real_ratio"] > 1.0,
    }
    if all(pass_conditions.values()):
        decision = "PASS"
    elif (
        spatial_rmse >= t0_rmse
        and spatial_rmse >= t1_rmse
        and spatial_fake["auroc"] <= 0.50
        and spatial_fake["median_fake_to_real_ratio"] <= 1.0
    ):
        decision = "FAIL"
    else:
        decision = "INCONCLUSIVE"
    if t1_rmse < t0_rmse and abs(spatial_rmse / t1_rmse - 1.0) <= 0.02:
        case = "A: added nonlinear self encoding explains the real-side gain"
    elif spatial_rmse < t1_rmse < t0_rmse:
        case = "B: spatial relation improves beyond self capacity control"
    elif spatial_rmse < t0_rmse and spatial_fake["auroc"] <= 0.55:
        case = "C: spatial relation helps real prediction but not fake separation"
    else:
        case = "none of the predefined A/B/C patterns"
    config = {
        "source_head": SOURCE_HEAD,
        "reused_manifest": str(TEMPORAL_ROOT / "pilot_manifest.json"),
        "topology": "probe_dense_no_self",
        "spatial_dim": 32,
        "hidden_dim": 64,
        "horizons": list(temporal_run.HORIZONS),
        "seed": SEED,
        "training_protocol": {
            "optimizer": "AdamW",
            "learning_rate": 1e-3,
            "weight_decay": 1e-4,
            "batch_size": 8,
            "max_epochs": 30,
            "patience": 5,
            "checkpoint_metric": "real validation masked MSE",
        },
        "parameters": parameters,
    }
    summary = {
        "decision": decision,
        "case": case,
        "parameters": parameters,
        "real_train_count": len(train_views),
        "real_val_count": len(val_views),
        "fake_count": len(fake_views),
        "t1_training": t1_training,
        "spatial_training": spatial_training,
        "spatial_real_val_improvement_vs_t0": 1.0 - spatial_rmse / t0_rmse,
        "spatial_real_val_improvement_vs_t1": 1.0 - spatial_rmse / t1_rmse,
        "spatial_val_train_rmse_ratio": spatial_rmse / spatial_train_rmse,
        "pass_conditions": pass_conditions,
        "paired_real_validation": paired,
        "fake_comparison": fake_comparison,
        "auroc_delta_bootstrap": bootstrap["auroc_delta"],
    }
    write_json(OUTPUT_ROOT / "probe_config.json", config)
    write_json(OUTPUT_ROOT / "summary.json", summary)


if __name__ == "__main__":
    main()
