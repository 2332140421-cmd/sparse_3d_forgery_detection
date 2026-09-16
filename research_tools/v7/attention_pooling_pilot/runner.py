"""Run the finite V7 learned local-unit pooling ablation.

Only the reduction from local SET_A logits to a window logit changes.  The
frontend, support rows, standardization, labels, source split, and optimizer
are all reused from the completed source-learning-curve pilot.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from research_tools.v7.multi_order_sequence_probe import model as sequence_model
from research_tools.v7.periodic_requery_probe import runner as periodic
from research_tools.v7.source_learning_curve import runner as curve

from .model import AttentionPoolModel, MeanBaselineModel, pool_mean


REPO_ROOT = Path(__file__).resolve().parents[3]
SOURCE_ROOT = curve.OUTPUT_ROOT
OUTPUT_ROOT = curve.DATA_ROOT / "derived/v7_activityforensics_attention_pooling_pilot_v1"
ORDERING_SEED = 20260909
MODEL_SEEDS = (20260909, 20260910, 20260911)
CONDITIONS = ("MEAN_BASELINE", "ATTENTION_POOL")
EPOCHS = 200
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
BOOTSTRAP_SEED = 20260909
BOOTSTRAP_REPLICATES = 10_000
BUDGET_S = 900.0


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def _atomic_json(path: Path, value: Any) -> None:
    curve.atomic_json(path, value)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_head() -> str:
    return curve.git_head()


def _progress(root: Path, stage: str, status: str, completed: int, total: int, **extra: Any) -> None:
    _atomic_json(root / "progress.json", {"stage": stage, "status": status, "completed": int(completed), "total": int(total), "updated_unix": time.time(), **extra})


def _budget_state(root: Path) -> dict[str, Any]:
    path = root / "state/budget.json"
    history_path = root / "state/retry_history.json"
    if path.is_file():
        old = json.loads(path.read_text(encoding="utf-8"))
        previous = float(old.get("cumulative_s", 0.0))
    else:
        previous = 0.0
    history = json.loads(history_path.read_text(encoding="utf-8")) if history_path.is_file() else []
    return {"budget_s": BUDGET_S, "elapsed_before_this_process_s": previous, "process_started_monotonic": time.monotonic(), "process_started_unix": time.time(), "retry_history": history}


def _elapsed(state: Mapping[str, Any]) -> float:
    return float(state["elapsed_before_this_process_s"] + time.monotonic() - state["process_started_monotonic"])


def _save_budget(root: Path, state: Mapping[str, Any], reason: str | None, **extra: Any) -> None:
    elapsed = _elapsed(state)
    _atomic_json(root / "state/budget.json", {"budget_s": BUDGET_S, "elapsed_before_this_process_s": elapsed, "process_started_unix": state["process_started_unix"], "process_elapsed_s": elapsed - float(state["elapsed_before_this_process_s"]), "cumulative_s": elapsed, "stop_reason": reason, "retry_history": state.get("retry_history", []), **extra})


def _values(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [{**dict(row), "features": {"SET_A": np.asarray(row["features"]["SET_A"], dtype=np.float64)}} for row in rows]


def _batch(rows: Sequence[Mapping[str, Any]], standardizer: Any, *, weighted: bool) -> dict[str, Any]:
    values = _values(rows)
    batch = periodic.make_batch("SET_A", values, standardizer, require_labels=False if not weighted else True)
    if weighted:
        batch["window_weights"] = periodic.source_class_weights(values)
    return batch


def _model_type(condition: str) -> type[MeanBaselineModel]:
    return MeanBaselineModel if condition == "MEAN_BASELINE" else AttentionPoolModel


def _state_hash(model: Any) -> str:
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        digest.update(name.encode("utf-8")); digest.update(np.asarray(value.detach().cpu()).tobytes(order="C"))
    return digest.hexdigest()


def _model_state(model: Any) -> dict[str, Any]:
    return {name: value.detach().cpu().numpy().tolist() for name, value in model.state_dict().items()}


def _load_model(record: Mapping[str, Any], device: str) -> Any:
    model = _model_type(str(record["condition"]))()
    state = {name: __import__("torch").as_tensor(value, dtype=model.state_dict()[name].dtype) for name, value in record["state_dict"].items()}
    model.load_state_dict(state)
    model.to(device); model.eval()
    return model


def _shared_pair(seed: int) -> tuple[MeanBaselineModel, AttentionPoolModel, str, str]:
    import torch

    torch.manual_seed(int(seed)); np.random.seed(int(seed) & 0xFFFFFFFF)
    baseline = MeanBaselineModel()
    baseline_hash = _state_hash(baseline)
    attention = AttentionPoolModel()
    attention.encoder.load_state_dict(baseline.encoder.state_dict())
    attention.head.load_state_dict(baseline.head.state_dict())
    shared_hash = hashlib.sha256()
    for name, value in baseline.state_dict().items():
        shared_hash.update(name.encode("utf-8")); shared_hash.update(value.detach().cpu().numpy().tobytes(order="C"))
    return baseline, attention, baseline_hash, shared_hash.hexdigest()


def _window_outputs(model: Any, batch: Mapping[str, Any], device: str) -> tuple[Any, Any]:
    import torch

    target = next(model.parameters()).device
    values = torch.as_tensor(batch["features"], dtype=torch.float32, device=target)
    intervals = torch.as_tensor(batch["intervals"], dtype=torch.float32, device=target)
    indices = torch.as_tensor(batch["unit_window_index"], dtype=torch.long, device=target)
    with torch.no_grad():
        pooled, auxiliary = model.window_outputs(values, intervals, indices, int(batch["window_count"]))
    return pooled.detach().cpu().numpy().astype(np.float64), auxiliary.detach().cpu().numpy().astype(np.float64)


def _score_attention_weights(model: AttentionPoolModel, batch: Mapping[str, Any], device: str) -> tuple[np.ndarray, np.ndarray]:
    import torch

    target = next(model.parameters()).device
    values = torch.as_tensor(batch["features"], dtype=torch.float32, device=target)
    intervals = torch.as_tensor(batch["intervals"], dtype=torch.float32, device=target)
    indices = torch.as_tensor(batch["unit_window_index"], dtype=torch.long, device=target)
    with torch.no_grad():
        pooled, alpha = model.window_outputs(values, intervals, indices, int(batch["window_count"]))
    maximum = torch.full((int(batch["window_count"]),), -torch.inf, dtype=alpha.dtype, device=alpha.device)
    maximum.scatter_reduce_(0, indices, alpha, reduce="amax", include_self=True)
    return pooled.cpu().numpy().astype(np.float64), maximum.cpu().numpy().astype(np.float64)


def _train_one(model: Any, batch: Mapping[str, Any], *, seed: int, condition: str, device: str, epoch_callback: Any | None = None) -> dict[str, Any]:
    import torch
    import torch.nn.functional as F

    torch.manual_seed(int(seed)); np.random.seed(int(seed) & 0xFFFFFFFF)
    target = torch.device(device)
    model.to(target); model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    values = torch.as_tensor(batch["features"], dtype=torch.float32, device=target)
    intervals = torch.as_tensor(batch["intervals"], dtype=torch.float32, device=target)
    indices = torch.as_tensor(batch["unit_window_index"], dtype=torch.long, device=target)
    labels = torch.as_tensor(batch["labels"], dtype=torch.float32, device=target)
    weights = torch.as_tensor(batch["window_weights"], dtype=torch.float32, device=target)
    history: list[dict[str, float]] = []
    for epoch in range(1, EPOCHS + 1):
        optimizer.zero_grad(set_to_none=True)
        pooled, _ = model.window_outputs(values, intervals, indices, int(batch["window_count"]))
        loss = torch.sum(F.binary_cross_entropy_with_logits(pooled, labels, reduction="none") * weights) / torch.sum(weights)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite loss at {condition} seed={seed} epoch={epoch}")
        loss.backward()
        if not all(parameter.grad is None or torch.all(torch.isfinite(parameter.grad)) for parameter in model.parameters()):
            raise FloatingPointError(f"non-finite gradient at {condition} seed={seed} epoch={epoch}")
        optimizer.step()
        row = {"epoch": epoch, "loss": float(loss.detach().cpu())}
        history.append(row)
        if epoch_callback is not None and epoch in (1, EPOCHS):
            epoch_callback(epoch, row)
    model.eval()
    return {"seed": int(seed), "epochs": EPOCHS, "initial_loss": history[0]["loss"], "final_loss": history[-1]["loss"], "min_loss": min(row["loss"] for row in history), "loss_history": history, "device": str(target), "parameter_count": int(sum(parameter.numel() for parameter in model.parameters()))}


def _gradient_check(model: AttentionPoolModel, batch: Mapping[str, Any], device: str) -> dict[str, Any]:
    import torch
    import torch.nn.functional as F

    target = torch.device(device)
    model.to(target); model.train()
    values = torch.as_tensor(batch["features"], dtype=torch.float32, device=target)
    intervals = torch.as_tensor(batch["intervals"], dtype=torch.float32, device=target)
    indices = torch.as_tensor(batch["unit_window_index"], dtype=torch.long, device=target)
    labels = torch.as_tensor(batch["labels"], dtype=torch.float32, device=target)
    weights = torch.as_tensor(batch["window_weights"], dtype=torch.float32, device=target)
    model.zero_grad(set_to_none=True)
    pooled, _ = model.window_outputs(values, intervals, indices, int(batch["window_count"]))
    loss = torch.sum(F.binary_cross_entropy_with_logits(pooled, labels, reduction="none") * weights) / torch.sum(weights)
    loss.backward()
    attention_gradient = model.attention_w.grad
    finite = all(parameter.grad is None or torch.all(torch.isfinite(parameter.grad)) for parameter in model.parameters())
    nonzero_attention = bool(attention_gradient is not None and torch.any(torch.abs(attention_gradient) > 0))
    return {"loss_finite": bool(torch.isfinite(loss)), "all_gradients_finite": bool(finite), "attention_gradient_nonzero": nonzero_attention, "loss": float(loss.detach().cpu())}


def _metrics(rows: Sequence[Mapping[str, Any]], scores: Sequence[float]) -> dict[str, Any]:
    labels = [int(row["label"]) for row in rows]
    values = [float(value) for value in scores]
    if not all(math.isfinite(value) for value in values):
        raise ValueError("non-finite score")
    by_source: dict[str, tuple[list[int], list[float]]] = {}
    for row, score in zip(rows, values):
        labels_for_source, scores_for_source = by_source.setdefault(str(row["source_id"]), ([], []))
        labels_for_source.append(int(row["label"])); scores_for_source.append(score)
    source_values = {source: value for source, (ys, ss) in by_source.items() if (value := periodic._auroc(ys, ss)) is not None}
    macro = curve._source_bootstrap(source_values)
    classification = periodic._classification(labels, values)
    return {"window_count": len(rows), "real_count": labels.count(0), "fake_count": labels.count(1), "source_count": len(by_source), "dual_role_source_count": len(source_values), "source_macro": macro, "pooled_auroc": periodic._auroc(labels, values), "pooled_ap": periodic._ap(labels, values), "classification": classification, "source_values": source_values}


def _diff_bootstrap(left: Mapping[str, float], right: Mapping[str, float]) -> dict[str, Any]:
    keys = sorted(set(left) & set(right))
    raw = np.asarray([float(left[key]) - float(right[key]) for key in keys], dtype=np.float64)
    if raw.size == 0:
        return {"source_count": 0, "mean": None, "ci95": [None, None], "sources": []}
    draw_indices = np.random.default_rng(BOOTSTRAP_SEED).integers(0, raw.size, size=(BOOTSTRAP_REPLICATES, raw.size))
    draws = raw[draw_indices].mean(axis=1)
    return {"source_count": len(keys), "sources": keys, "mean": float(raw.mean()), "ci95": [float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))], "seed": BOOTSTRAP_SEED, "replicates": BOOTSTRAP_REPLICATES, "positive_count": int(np.sum(raw > 0)), "zero_count": int(np.sum(raw == 0)), "negative_count": int(np.sum(raw < 0))}


def _metric_row(split: str, condition: str, seed: str | int, metrics: Mapping[str, Any]) -> dict[str, Any]:
    macro = metrics["source_macro"]; classification = metrics["classification"]
    return {"split": split, "condition": condition, "seed": seed, "window_count": metrics["window_count"], "real_count": metrics["real_count"], "fake_count": metrics["fake_count"], "source_count": metrics["source_count"], "dual_role_source_count": metrics["dual_role_source_count"], "source_macro_auroc": macro.get("mean"), "source_macro_ci_low": (macro.get("ci95") or [None, None])[0], "source_macro_ci_high": (macro.get("ci95") or [None, None])[1], "pooled_auroc": metrics["pooled_auroc"], "pooled_ap": metrics["pooled_ap"], "precision": classification["precision"], "recall": classification["recall"], "f1": classification["f1"], "accuracy": classification["accuracy"], "tn": classification["tn"], "fp": classification["fp"], "fn": classification["fn"], "tp": classification["tp"]}


def _attention_weight_summary(rows: Sequence[Mapping[str, Any]], weights: Sequence[float], split: str, seed: str | int) -> dict[str, Any]:
    values = np.asarray(weights, dtype=np.float64)
    if values.size == 0 or not np.all(np.isfinite(values)):
        raise ValueError("invalid attention max-weight summary")
    return {"split": split, "condition": "ATTENTION_POOL", "seed": seed, "window_count": int(values.size), "max_weight_mean": float(values.mean()), "max_weight_median": float(np.median(values)), "max_weight_p25": float(np.percentile(values, 25)), "max_weight_p75": float(np.percentile(values, 75)), "max_weight_p95": float(np.percentile(values, 95)), "max_weight_max": float(values.max()), "max_weight_min": float(values.min())}


def _plot_loss(path: Path, records: Sequence[Mapping[str, Any]]) -> str:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover
        return f"unavailable: {type(exc).__name__}: {exc}"
    figure, axis = plt.subplots(figsize=(7.0, 4.2), constrained_layout=True)
    colors = {"MEAN_BASELINE": "#1f77b4", "ATTENTION_POOL": "#d62728"}
    for record in records:
        history = record["fit"]["loss_history"]
        axis.plot([int(item["epoch"]) for item in history], [float(item["loss"]) for item in history], alpha=0.55, color=colors[str(record["condition"])], label=f"{record['condition']} seed {record['seed']}")
    axis.set_xlabel("epoch"); axis.set_ylabel("weighted BCE"); axis.set_title("V7 local pooling pilot training loss"); axis.grid(alpha=0.2); axis.legend(fontsize=7, ncol=2)
    figure.savefig(path, format="svg"); plt.close(figure)
    return "written"


def _load_inputs() -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    _, subsets = curve._load_split(SOURCE_ROOT)
    rows = curve._load_r_rows(SOURCE_ROOT)
    train_sources = [str(item) for item in subsets["subsets"][str(ORDERING_SEED)]["64"]]
    train_rows = [row for row in rows if str(row["source_id"]) in set(train_sources)]
    validation_sources = {str(item) for item in subsets["validation_sources"]}
    validation_candidates = [row for row in rows if str(row["source_id"]) in validation_sources]
    dual_sources = {source for source in validation_sources if {int(row["label"]) for row in validation_candidates if str(row["source_id"]) == source} == {0, 1}}
    validation_rows = [row for row in validation_candidates if str(row["source_id"]) in dual_sources]
    expected = {"train_sources": train_sources, "validation_sources": sorted(dual_sources), "train_rows": len(train_rows), "train_real": sum(int(row["label"]) == 0 for row in train_rows), "train_fake": sum(int(row["label"]) == 1 for row in train_rows), "validation_rows": len(validation_rows), "validation_real": sum(int(row["label"]) == 0 for row in validation_rows), "validation_fake": sum(int(row["label"]) == 1 for row in validation_rows)}
    if expected["train_rows"] != 358 or expected["train_real"] != 181 or expected["train_fake"] != 177:
        raise ValueError(f"unexpected train set: {expected}")
    if expected["validation_rows"] != 83 or expected["validation_real"] != 42 or expected["validation_fake"] != 41 or len(dual_sources) != 14:
        raise ValueError(f"unexpected validation set: {expected}")
    old_models = json.loads((SOURCE_ROOT / "models/fold_models.json").read_text(encoding="utf-8"))
    old_record = next(item for item in old_models["records"] if int(item["ordering_seed"]) == ORDERING_SEED and int(item["source_count"]) == 64 and int(item["seed"]) == MODEL_SEEDS[0])
    return train_rows, validation_rows, subsets, {"expected": expected, "old_record": old_record}


def _old_loader_check(validation_rows: Sequence[Mapping[str, Any]], old_record: Mapping[str, Any]) -> dict[str, Any]:
    old_model = curve._model_from_record(old_record, "cpu")
    standardizer = periodic._standardizer(old_record)
    batch = periodic.make_batch("SET_A", _values(validation_rows), standardizer, require_labels=False)
    scores = periodic.score_batch("SET_A", old_model, batch, "cpu")
    score_path = SOURCE_ROOT / "scores/validation_window_scores.csv"
    with score_path.open(newline="", encoding="utf-8") as handle:
        existing = {str(row["window_id"]): float(row[_old_score_column()]) for row in csv.DictReader(handle)}
    differences = [abs(float(score) - existing[str(window_id)]) for window_id, score in zip(batch["window_ids"], scores)]
    return {"old_condition": "SUMMARY_SET/SET_A", "old_model_key": f"{ORDERING_SEED}/64/{MODEL_SEEDS[0]}", "max_abs_score_difference": float(max(differences)) if differences else None, "scored_windows": len(differences), "tolerance": 1e-5, "passed": bool(differences and max(differences) <= 1e-5)}


def _old_score_column() -> str:
    return f"ordering_{ORDERING_SEED}_size_64_seed_{MODEL_SEEDS[0]}"


def _initial_checks(train_batch: Mapping[str, Any], validation_batch: Mapping[str, Any], device: str) -> dict[str, Any]:
    import torch

    baseline, attention, baseline_hash, shared_hash = _shared_pair(MODEL_SEEDS[0])
    baseline.eval(); attention.eval()
    baseline.to(device); attention.to(device)
    base_train, _ = _window_outputs(baseline, train_batch, device)
    attention_train, alpha = _score_attention_weights(attention, train_batch, device)
    base_validation, _ = _window_outputs(baseline, validation_batch, device)
    attention_validation, _ = _score_attention_weights(attention, validation_batch, device)
    uniform_diff = max(float(np.max(np.abs(base_train - attention_train))), float(np.max(np.abs(base_validation - attention_validation))))
    # Permuting units while retaining each unit's window ID must not alter output.
    values = torch.as_tensor(validation_batch["features"], dtype=torch.float32, device=device)
    intervals = torch.as_tensor(validation_batch["intervals"], dtype=torch.float32, device=device)
    indices = torch.as_tensor(validation_batch["unit_window_index"], dtype=torch.long, device=device)
    with torch.no_grad():
        representation, local_logits = attention.local_outputs(values, intervals)
        permutation = torch.arange(values.shape[0] - 1, -1, -1, device=values.device)
        permuted, _ = attention.window_outputs(values[permutation], intervals[permutation], indices[permutation], int(validation_batch["window_count"]))
        original, _ = attention.window_outputs(values, intervals, indices, int(validation_batch["window_count"]))
    gradient = _gradient_check(attention, train_batch, device)
    return {"uniform_attention_max_abs_difference": uniform_diff, "uniform_attention_tolerance": 1e-5, "uniform_attention_passed": bool(uniform_diff <= 1e-5), "permutation_max_abs_difference": float(torch.max(torch.abs(original - permuted)).detach().cpu()), "permutation_invariant": bool(torch.max(torch.abs(original - permuted)) <= 1e-5), "baseline_initial_state_hash": baseline_hash, "shared_initial_state_hash": shared_hash, "attention_parameter_count": int(sum(parameter.numel() for parameter in attention.parameters())), "baseline_parameter_count": int(sum(parameter.numel() for parameter in baseline.parameters())), "attention_max_weight_initial_mean": float(np.mean(alpha)), "gradient_check": gradient}


def _records_save(path: Path, records: Sequence[Mapping[str, Any]]) -> None:
    _atomic_json(path, {"condition": "MEAN_BASELINE + ATTENTION_POOL", "ordering_seed": ORDERING_SEED, "seeds": list(MODEL_SEEDS), "epochs": EPOCHS, "learning_rate": LEARNING_RATE, "weight_decay": WEIGHT_DECAY, "records": list(records)})


def _score_columns(split: str, score_rows: Sequence[Mapping[str, Any]], score_store: Mapping[tuple[str, str, int], Mapping[str, float]], weight_store: Mapping[tuple[str, str, int], Mapping[str, float]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for row in score_rows:
        item = {key: row.get(key) for key in ("window_id", "source_id", "role", "label", "annotation_category", "offset_s", "valid_unit_count")}
        for condition in CONDITIONS:
            for seed in MODEL_SEEDS:
                item[f"{condition}_seed_{seed}"] = score_store[(split, condition, seed)][str(row["window_id"])]
            item[f"{condition}_MEAN_LOGIT"] = float(np.mean([item[f"{condition}_seed_{seed}"] for seed in MODEL_SEEDS]))
            if condition == "ATTENTION_POOL":
                for seed in MODEL_SEEDS:
                    item[f"ATTENTION_MAX_WEIGHT_seed_{seed}"] = weight_store[(split, condition, seed)][str(row["window_id"])]
                item["ATTENTION_MAX_WEIGHT_MEAN"] = float(np.mean([item[f"ATTENTION_MAX_WEIGHT_seed_{seed}"] for seed in MODEL_SEEDS]))
        output.append(item)
    return output


def _metric_and_sources(rows: Sequence[Mapping[str, Any]], score_store: Mapping[tuple[str, str, int], Mapping[str, float]], split: str, condition: str, seed: int | str) -> tuple[dict[str, Any], dict[str, float]]:
    source_scores = score_store[(split, condition, int(seed))] if seed != "MEAN_LOGIT" else {str(row["window_id"]): float(np.mean([score_store[(split, condition, model_seed)][str(row["window_id"])] for model_seed in MODEL_SEEDS])) for row in rows}
    scores = [float(source_scores[str(row["window_id"])]) for row in rows]
    metrics = _metrics(rows, scores)
    return metrics, metrics["source_values"]


def evaluate(root: Path, train_rows: Sequence[Mapping[str, Any]], validation_rows: Sequence[Mapping[str, Any]], score_store: Mapping[tuple[str, str, int], Mapping[str, float]], weight_store: Mapping[tuple[str, str, int], Mapping[str, float]]) -> dict[str, Any]:
    metric_rows: list[dict[str, Any]] = []
    weight_rows: list[dict[str, Any]] = []
    for split, rows in (("train", train_rows), ("validation", validation_rows)):
        for condition in CONDITIONS:
            for seed in MODEL_SEEDS:
                metrics, _ = _metric_and_sources(rows, score_store, split, condition, seed)
                metric_rows.append(_metric_row(split, condition, seed, metrics))
            mean_metrics, _ = _metric_and_sources(rows, score_store, split, condition, "MEAN_LOGIT")
            metric_rows.append(_metric_row(split, condition, "MEAN_LOGIT", mean_metrics))
            if condition == "ATTENTION_POOL":
                for seed in MODEL_SEEDS:
                    weights = [weight_store[(split, condition, seed)][str(row["window_id"])] for row in rows]
                    weight_rows.append(_attention_weight_summary(rows, weights, split, seed))
                mean_weights = [float(np.mean([weight_store[(split, condition, seed)][str(row["window_id"])] for seed in MODEL_SEEDS])) for row in rows]
                weight_rows.append(_attention_weight_summary(rows, mean_weights, split, "MEAN_LOGIT"))
    baseline_values = _metric_and_sources(validation_rows, score_store, "validation", "MEAN_BASELINE", "MEAN_LOGIT")[1]
    attention_values = _metric_and_sources(validation_rows, score_store, "validation", "ATTENTION_POOL", "MEAN_LOGIT")[1]
    primary = _diff_bootstrap(attention_values, baseline_values)
    per_source: list[dict[str, Any]] = []
    for condition in CONDITIONS:
        for seed in (*MODEL_SEEDS, "MEAN_LOGIT"):
            values = _metric_and_sources(validation_rows, score_store, "validation", condition, seed)[1]
            for source in sorted(values):
                source_rows = [row for row in validation_rows if str(row["source_id"]) == source]
                per_source.append({"split": "validation", "condition": condition, "seed": seed, "source_id": source, "source_auroc": values[source], "real_count": sum(int(row["label"]) == 0 for row in source_rows), "fake_count": sum(int(row["label"]) == 1 for row in source_rows), "valid_unit_count": sum(int(row["valid_unit_count"]) for row in source_rows)})
    for seed in (*MODEL_SEEDS, "MEAN_LOGIT"):
        baseline_seed_values = _metric_and_sources(validation_rows, score_store, "validation", "MEAN_BASELINE", seed)[1]
        attention_seed_values = _metric_and_sources(validation_rows, score_store, "validation", "ATTENTION_POOL", seed)[1]
        for source in sorted(set(baseline_seed_values) & set(attention_seed_values)):
            per_source.append({"split": "validation", "condition": "ATTENTION_MINUS_BASELINE", "seed": seed, "source_id": source, "source_auroc": attention_seed_values[source] - baseline_seed_values[source], "real_count": sum(int(row["label"]) == 0 for row in validation_rows if str(row["source_id"]) == source), "fake_count": sum(int(row["label"]) == 1 for row in validation_rows if str(row["source_id"]) == source)})
    _write_csv(root / "evaluation/metrics.csv", metric_rows)
    _write_csv(root / "evaluation/per_source_metrics.csv", per_source)
    _write_csv(root / "evaluation/attention_weight_summary.csv", weight_rows)
    summary = {"validation_population": {"source_count": len({str(row["source_id"]) for row in validation_rows}), "window_count": len(validation_rows), "real_count": sum(int(row["label"]) == 0 for row in validation_rows), "fake_count": sum(int(row["label"]) == 1 for row in validation_rows)}, "train_population": {"source_count": len({str(row["source_id"]) for row in train_rows}), "window_count": len(train_rows), "real_count": sum(int(row["label"]) == 0 for row in train_rows), "fake_count": sum(int(row["label"]) == 1 for row in train_rows)}, "primary_comparison": {"attention_minus_mean_baseline": primary}, "attention_weight_summary": weight_rows, "metrics": metric_rows}
    _atomic_json(root / "evaluation/summary.json", summary)
    return summary


def _report(root: Path, protocol: Mapping[str, Any], checks: Mapping[str, Any], summary: Mapping[str, Any], records: Sequence[Mapping[str, Any]], plot_status: str) -> Path:
    metrics = summary["metrics"]
    mean = {(str(row["split"]), str(row["condition"])): row for row in metrics if str(row["seed"]) == "MEAN_LOGIT"}
    primary = summary["primary_comparison"]["attention_minus_mean_baseline"]
    train_b = mean[("train", "MEAN_BASELINE")]; train_a = mean[("train", "ATTENTION_POOL")]
    val_b = mean[("validation", "MEAN_BASELINE")]; val_a = mean[("validation", "ATTENTION_POOL")]
    train_diff = float(train_a["source_macro_auroc"]) - float(train_b["source_macro_auroc"])
    val_diff = float(val_a["source_macro_auroc"]) - float(val_b["source_macro_auroc"])
    lines = [
        "# V7 局部注意力聚合最小对照 pilot",
        "",
        "本实验固定 ordering=20260909 的 R 观测、SET_A 局部编码、训练窗口、标准化、source/class weighted BCE、Adam、200 epochs 和 14-source/83-window 验证集，只将局部单元到窗口的等权平均替换为单头八维 softmax 加权。",
        "",
        "## 结果先行",
        "",
        f"- 完成模型：{len(records)}/{len(CONDITIONS) * len(MODEL_SEEDS)}；训练样本：{summary['train_population']['window_count']} 窗口（{summary['train_population']['real_count']} real / {summary['train_population']['fake_count']} fake）；验证样本：{summary['validation_population']['window_count']} 窗口（{summary['validation_population']['real_count']} real / {summary['validation_population']['fake_count']} fake）。",
        f"- 训练 source-macro AUROC：MEAN_BASELINE={train_b['source_macro_auroc']:.6f}，ATTENTION_POOL={train_a['source_macro_auroc']:.6f}，差={train_diff:+.6f}。",
        f"- 验证 source-macro AUROC：MEAN_BASELINE={val_b['source_macro_auroc']:.6f}，ATTENTION_POOL={val_a['source_macro_auroc']:.6f}，差={val_diff:+.6f}；source bootstrap 95% CI=[{primary['ci95'][0]:.6f}, {primary['ci95'][1]:.6f}]。",
        f"- source 方向：提高 {primary['positive_count']}、持平 {primary['zero_count']}、下降 {primary['negative_count']}（共 {primary['source_count']} 个 source）。",
        "- 按预声明规则，只有训练和验证都改善才支持继续验证学习式聚合；本报告依据实际数值给出结论，不因单个 seed 或点估计选择模型。",
        "",
        "## 固定身份与必要验证",
        "",
        f"- Git HEAD：`{protocol['git_head']}`；本轮运行 HEAD：`{checks['git_head']}`。输入来自既有 source-learning-curve 目录，未重跑前端。",
        f"- 训练 ordering：{ORDERING_SEED}；有效训练窗口：{summary['train_population']['window_count']}；验证双类别 source：{summary['validation_population']['source_count']}。",
        f"- 旧 SUMMARY_SET 加载复现：{checks['old_loader_check']['scored_windows']} 个验证窗口，最大绝对差={checks['old_loader_check']['max_abs_score_difference']:.3g}，通过={checks['old_loader_check']['passed']}。",
        f"- 初始均匀注意力退化误差：{checks['initial']['uniform_attention_max_abs_difference']:.3g}（容差 1e-5，通过={checks['initial']['uniform_attention_passed']}）；局部排列不变性误差={checks['initial']['permutation_max_abs_difference']:.3g}；梯度检查={checks['initial']['gradient_check']}。",
        f"- 参数量：MEAN_BASELINE={checks['initial']['baseline_parameter_count']}，ATTENTION_POOL={checks['initial']['attention_parameter_count']}；attention V 固定初始化 seed={AttentionPoolModel.ATTENTION_INIT_SEED}，b/w 为零初始化。",
        "",
        "## seed 平均 logit 指标",
        "",
        "| split | condition | macro AUROC | pooled AUROC | AP | F1 | ACC | TN/FP/FN/TP |",
        "|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in (train_b, train_a, val_b, val_a):
        cls = row
        lines.append(f"| {row['split']} | {row['condition']} | {row['source_macro_auroc']:.6f} | {row['pooled_auroc']:.6f} | {row['pooled_ap']:.6f} | {row['f1']:.6f} | {row['accuracy']:.6f} | {row['tn']}/{row['fp']}/{row['fn']}/{row['tp']} |")
    lines += [
        "",
        "训练/验证各 seed、窗口数、P/R/F1/ACC 和完整混淆矩阵见 `evaluation/metrics.csv`；逐 source 与逐 seed 差值见 `evaluation/per_source_metrics.csv`。",
        "",
        "## 注意力权重",
        "",
        "`evaluation/attention_weight_summary.csv` 仅描述每个窗口的最大局部权重；权重不是定位真值或伪造概率。单局部窗口和初始 w=0 时权重为均匀分配，padding/无效单元不进入 softmax。",
        "",
        "| split | seed | max-weight median | max-weight P95 | max-weight max |",
        "|---|---:|---:|---:|---:|",
    ]
    for weight_row in summary.get("attention_weight_summary", []):
        if str(weight_row["seed"]) == "MEAN_LOGIT":
            lines.append(f"| {weight_row['split']} | MEAN_LOGIT | {weight_row['max_weight_median']:.6f} | {weight_row['max_weight_p95']:.6f} | {weight_row['max_weight_max']:.6f} |")
    lines += [
        "",
        "## 解释边界",
        "",
        "这是固定数据与固定 SET_A 编码器上的学习式聚合对照，不是完整 Set Transformer，不改变正式检测链，也不证明注意力权重对应伪造区域。没有 AP@IoU、空间真值或未知 source 泛化证据。",
        "",
        "## 判定",
        "",
        f"训练差={train_diff:+.6f}，验证差={val_diff:+.6f}，因此本轮按协议判断为 {'训练与验证均改善，值得在独立验证上继续观察' if train_diff > 0 and val_diff > 0 else '未形成训练与验证同时改善的证据，结束该候选'}。若区间跨 0，则不能写成稳定有效。",
        "",
        f"预算累计：{checks['elapsed_s']:.3f}s / {BUDGET_S:.3f}s；loss 图：`loss_curve.svg`（{plot_status}）。",
        f"本轮重试记录：{len(checks.get('retry_history', []))} 条；历史失败不会计为模型完成。",
        "未访问旧 R7/V5；未下载数据；未重跑 frontend、tracking、depth、pose 或 segmentation；未提交模型、视频或大数组。",
        "",
    ]
    report = root / "report.md"; report.write_text("\n".join(lines), encoding="utf-8"); return report


def run(root: Path = OUTPUT_ROOT, *, device: str = "cuda", resume: bool = True) -> dict[str, Any]:
    started = time.time(); root.mkdir(parents=True, exist_ok=True); (root / "state").mkdir(parents=True, exist_ok=True)
    state = _budget_state(root)
    launch = {"git_head": _git_head(), "device": device, "pid": os.getpid(), "started_unix": time.time(), "ordering_seed": ORDERING_SEED}
    _atomic_json(root / "state/launch.json", launch)
    train_rows, validation_rows, subsets, input_info = _load_inputs()
    old_check = _old_loader_check(validation_rows, input_info["old_record"])
    standardizer = periodic.fit_standardizer("SET_A", [row["features"]["SET_A"] for row in _values(train_rows)], periodic.source_class_weights(_values(train_rows)))
    train_batch = _batch(train_rows, standardizer, weighted=True); validation_batch = _batch(validation_rows, standardizer, weighted=False)
    checks = {"git_head": launch["git_head"], "old_loader_check": old_check, "elapsed_s": 0.0, "retry_history": state.get("retry_history", [])}
    checks["initial"] = _initial_checks(train_batch, validation_batch, device)
    if not checks["initial"]["uniform_attention_passed"] or not checks["initial"]["permutation_invariant"] or not checks["old_loader_check"]["passed"] or not checks["initial"]["gradient_check"]["all_gradients_finite"] or not checks["initial"]["gradient_check"]["attention_gradient_nonzero"]:
        raise RuntimeError(f"preflight contract failed: {checks}")
    expected_records = {("MEAN_BASELINE", seed) for seed in MODEL_SEEDS} | {("ATTENTION_POOL", seed) for seed in MODEL_SEEDS}
    record_path = root / "models/fold_models.json"
    old_records = json.loads(record_path.read_text(encoding="utf-8")).get("records", []) if resume and record_path.is_file() else []
    records = [record for record in old_records if (str(record.get("condition")), int(record.get("seed", -1))) in expected_records and record.get("status") == "TRAIN_COMPLETE"]
    complete = {(str(record["condition"]), int(record["seed"])) for record in records}
    _atomic_json(root / "protocol.json", {"protocol_id": "v7-attention-pooling-pilot-v1", "source_learning_curve_root": str(SOURCE_ROOT), "source_learning_curve_git_head": "c814abb23f367adc82499fa262948c444fbad024", "git_head": launch["git_head"], "ordering_seed": ORDERING_SEED, "training_window_count": len(train_rows), "training_real_count": sum(int(row["label"]) == 0 for row in train_rows), "training_fake_count": sum(int(row["label"]) == 1 for row in train_rows), "validation_source_count": len({str(row["source_id"]) for row in validation_rows}), "validation_window_count": len(validation_rows), "validation_real_count": sum(int(row["label"]) == 0 for row in validation_rows), "validation_fake_count": sum(int(row["label"]) == 1 for row in validation_rows), "conditions": {"MEAN_BASELINE": "existing SET_A local logit then equal mean over local units", "ATTENTION_POOL": "same local representation/logit, one-head 8D softmax local pooling, temperature=1"}, "attention": {"hidden_dim": 8, "v_init_seed": AttentionPoolModel.ATTENTION_INIT_SEED, "bias_init": "zeros", "weight_init": "zeros", "padding_masked": True}, "model": {"epochs": EPOCHS, "optimizer": "Adam", "learning_rate": LEARNING_RATE, "weight_decay": WEIGHT_DECAY, "seeds": list(MODEL_SEEDS), "weighted_bce": "existing source/class window weights", "standardization": "training rows only"}, "evaluation": {"threshold": "logit >= 0", "bootstrap_seed": BOOTSTRAP_SEED, "bootstrap_replicates": BOOTSTRAP_REPLICATES, "main": "source-macro AUROC on fixed validation rows"}, "boundaries": ["no frontend rerun", "no formal src changes", "no old R7/V5", "attention weights are not localization truth"]})
    _atomic_json(root / "input_manifest.json", {"git_head": launch["git_head"], "source_protocol_sha256": _sha256(SOURCE_ROOT / "protocol.json"), "source_subsets_sha256": _sha256(SOURCE_ROOT / "manifests/training_subsets.json"), "source_support_sha256": _sha256(SOURCE_ROOT / "support/window_support.json"), "source_model_sha256": _sha256(SOURCE_ROOT / "models/fold_models.json"), "source_validation_score_sha256": _sha256(SOURCE_ROOT / "scores/validation_window_scores.csv"), "train_source_ids": input_info["expected"]["train_sources"], "validation_source_ids": input_info["expected"]["validation_sources"], "train_window_ids": [str(row["window_id"]) for row in train_rows], "validation_window_ids": [str(row["window_id"]) for row in validation_rows], "expected_counts": input_info["expected"]})
    _atomic_json(root / "checks/preflight.json", checks)
    _progress(root, "train", "RUNNING", len(complete), len(expected_records), elapsed_s=_elapsed(state))
    score_store: dict[tuple[str, str, int], dict[str, float]] = {}
    weight_store: dict[tuple[str, str, int], dict[str, float]] = {}
    for record in records:
        model = _load_model(record, device)
        train_scores, train_aux = _window_outputs(model, train_batch, device) if record["condition"] == "MEAN_BASELINE" else _score_attention_weights(model, train_batch, device)
        validation_scores, validation_aux = _window_outputs(model, validation_batch, device) if record["condition"] == "MEAN_BASELINE" else _score_attention_weights(model, validation_batch, device)
        score_store[("train", str(record["condition"]), int(record["seed"]))] = {str(row["window_id"]): float(score) for row, score in zip(train_rows, train_scores)}
        score_store[("validation", str(record["condition"]), int(record["seed"]))] = {str(row["window_id"]): float(score) for row, score in zip(validation_rows, validation_scores)}
        if record["condition"] == "ATTENTION_POOL":
            weight_store[("train", str(record["condition"]), int(record["seed"]))] = {str(row["window_id"]): float(value) for row, value in zip(train_rows, train_aux)}
            weight_store[("validation", str(record["condition"]), int(record["seed"]))] = {str(row["window_id"]): float(value) for row, value in zip(validation_rows, validation_aux)}
        del model
    for condition, seed in sorted(expected_records):
        if (condition, seed) in complete:
            continue
        if _elapsed(state) >= BUDGET_S:
            _save_budget(root, state, "BUDGET_EXHAUSTED", completed_models=len(complete), total_models=len(expected_records)); _progress(root, "train", "BUDGET_EXHAUSTED", len(complete), len(expected_records)); raise RuntimeError("BUDGET_EXHAUSTED")
        baseline, attention, baseline_hash, shared_hash = _shared_pair(seed)
        model = baseline if condition == "MEAN_BASELINE" else attention
        batch = train_batch
        fit = _train_one(model, batch, seed=seed, condition=condition, device=device, epoch_callback=lambda epoch, metrics, c=condition, s=seed: print(f"attention-pooling condition={c} seed={s} epoch={epoch}/{EPOCHS} loss={metrics['loss']:.6f} elapsed={_elapsed(state):.1f}s", flush=True))
        record = {"condition": condition, "seed": int(seed), "status": "TRAIN_COMPLETE", "base_condition": "SET_A", "ordering_seed": ORDERING_SEED, "parameter_count": fit["parameter_count"], "training_window_count": len(train_rows), "training_real_count": sum(int(row["label"]) == 0 for row in train_rows), "training_fake_count": sum(int(row["label"]) == 1 for row in train_rows), "training_source_count": len(input_info["expected"]["train_sources"]), "validation_source_count": len(input_info["expected"]["validation_sources"]), "shared_initial_state_hash": shared_hash, "baseline_initial_state_hash": baseline_hash, "attention_v_init_seed": AttentionPoolModel.ATTENTION_INIT_SEED, "fit": fit, "standardization": standardizer.as_dict(), "state_dict": _model_state(model)}
        records.append(record); complete.add((condition, seed)); _records_save(record_path, records)
        train_scores, train_aux = _window_outputs(model, train_batch, device) if condition == "MEAN_BASELINE" else _score_attention_weights(model, train_batch, device)
        validation_scores, validation_aux = _window_outputs(model, validation_batch, device) if condition == "MEAN_BASELINE" else _score_attention_weights(model, validation_batch, device)
        score_store[("train", condition, seed)] = {str(row["window_id"]): float(score) for row, score in zip(train_rows, train_scores)}
        score_store[("validation", condition, seed)] = {str(row["window_id"]): float(score) for row, score in zip(validation_rows, validation_scores)}
        if condition == "ATTENTION_POOL":
            weight_store[("train", condition, seed)] = {str(row["window_id"]): float(value) for row, value in zip(train_rows, train_aux)}
            weight_store[("validation", condition, seed)] = {str(row["window_id"]): float(value) for row, value in zip(validation_rows, validation_aux)}
        del model; _progress(root, "train", "RUNNING", len(complete), len(expected_records), current_condition=condition, current_seed=seed, elapsed_s=_elapsed(state))
    if len(complete) != len(expected_records):
        raise RuntimeError("formal model records incomplete")
    train_score_rows = _score_columns("train", train_rows, score_store, weight_store); validation_score_rows = _score_columns("validation", validation_rows, score_store, weight_store)
    _write_csv(root / "scores/train_window_scores.csv", train_score_rows); _write_csv(root / "scores/validation_window_scores.csv", validation_score_rows)
    summary = evaluate(root, train_rows, validation_rows, score_store, weight_store)
    plot_status = _plot_loss(root / "loss_curve.svg", records)
    checks["elapsed_s"] = _elapsed(state); report = _report(root, json.loads((root / "protocol.json").read_text(encoding="utf-8")), checks, summary, records, plot_status)
    _atomic_json(root / "checks/preflight.json", checks); _save_budget(root, state, None, completed_models=len(complete), total_models=len(expected_records)); _atomic_json(root / "final_status.json", {"status": "COMPLETE", "completed_models": len(complete), "total_models": len(expected_records), "report": str(report), "elapsed_s": checks["elapsed_s"], "git_head": launch["git_head"], "device": device})
    _progress(root, "report", "COMPLETE", 1, 1, completed_models=len(complete), elapsed_s=checks["elapsed_s"])
    return {"status": "COMPLETE", "report": str(report), "elapsed_s": checks["elapsed_s"], "completed_models": len(complete), "checks": checks, "summary": summary}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    result = run(args.output_root, device=args.device, resume=args.resume)
    print(json.dumps({"status": result["status"], "report": result["report"], "elapsed_s": result["elapsed_s"], "completed_models": result["completed_models"]}, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
