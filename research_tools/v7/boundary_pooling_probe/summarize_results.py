"""Post-hoc classification and optimization summary for a completed pilot.

This module reads only the saved OOF scores and fold model records.  It never
reruns a frontend, fits a model, changes a threshold, or changes the primary
source-level AUROC calculation.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_ROOT = Path("/root/autodl-tmp/data/sparse_3d_forgery_detection/derived/v7_activityforensics_boundary_pooling_pilot_v1")
CONDITIONS = ("H_MEAN_A", "H_MEAN_C", "H_MEAN_D", "H_MAX_A", "H_MAX_C", "H_MAX_D", "B_MEAN_A", "B_MEAN_C", "B_MEAN_D", "B_MAX_A", "B_MAX_C", "B_MAX_D")
SEEDS = (20260909, 20260910, 20260911)
REPRESENTATIVE_CONDITIONS = ("H_MEAN_A", "H_MAX_A", "B_MEAN_A", "B_MAX_A")


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp, path)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _finite(value: Any) -> float | None:
    if value in (None, "", "NA", "nan", "NaN"):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def _roc_auc(labels: np.ndarray, scores: np.ndarray) -> float | None:
    positives = scores[labels == 1]
    negatives = scores[labels == 0]
    if positives.size == 0 or negatives.size == 0:
        return None
    comparisons = (positives[:, None] > negatives[None, :]).astype(np.float64)
    comparisons += 0.5 * (positives[:, None] == negatives[None, :])
    return float(np.mean(comparisons))


def _average_precision(labels: np.ndarray, scores: np.ndarray) -> float | None:
    positive_count = int(np.sum(labels == 1))
    if positive_count == 0 or int(np.sum(labels == 0)) == 0:
        return None
    order = np.argsort(-scores, kind="mergesort")
    sorted_labels = labels[order]
    cumulative = np.cumsum(sorted_labels == 1)
    ranks = np.arange(1, sorted_labels.size + 1, dtype=np.float64)
    return float(np.sum((cumulative / ranks) * (sorted_labels == 1)) / positive_count)


def _classification_row(rows: Sequence[Mapping[str, Any]], condition: str) -> dict[str, Any]:
    selected: list[tuple[int, float]] = []
    missing = 0
    for row in rows:
        if str(row.get("kind")) != "MANIP":
            continue
        value = _finite(row.get(f"{condition}_score"))
        if value is None:
            missing += 1
            continue
        selected.append((1 if str(row.get("role")) == "fake" else 0, value))
    labels = np.asarray([item[0] for item in selected], dtype=np.int64)
    scores = np.asarray([item[1] for item in selected], dtype=np.float64)
    predicted = scores >= 0.0
    tp = int(np.sum((labels == 1) & predicted))
    fp = int(np.sum((labels == 0) & predicted))
    tn = int(np.sum((labels == 0) & ~predicted))
    fn = int(np.sum((labels == 1) & ~predicted))
    precision = float(tp / (tp + fp)) if tp + fp else None
    recall = float(tp / (tp + fn)) if tp + fn else None
    f1 = float(2 * precision * recall / (precision + recall)) if precision is not None and recall is not None and precision + recall else None
    accuracy = float((tp + tn) / labels.size) if labels.size else None
    return {
        "condition": condition,
        "sample_unit": "MANIP window; fake=positive, real=negative",
        "score": "mean of the three saved source-disjoint OOF logits",
        "threshold": "logit >= 0",
        "n_total": int(labels.size),
        "n_fake": int(np.sum(labels == 1)),
        "n_real": int(np.sum(labels == 0)),
        "missing_score_windows": int(missing),
        "pooled_roc_auc": _roc_auc(labels, scores),
        "average_precision": _average_precision(labels, scores),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "ctrl_included": False,
        "missing_values": "rows without all three saved seed logits are excluded; no zero fill",
    }


def _source_seed_metrics(rows: Sequence[Mapping[str, Any]], condition: str) -> list[dict[str, Any]]:
    """Compute descriptive MANIP AUROC for each saved source/seed score."""

    grouped: dict[tuple[str, int], list[tuple[int, float]]] = {}
    for row in rows:
        if str(row.get("kind")) != "MANIP":
            continue
        source = str(row.get("source_id"))
        for seed in SEEDS:
            value = _finite(row.get(f"{condition}_seed_{seed}"))
            if value is None:
                continue
            grouped.setdefault((source, seed), []).append((1 if str(row.get("role")) == "fake" else 0, value))
    output: list[dict[str, Any]] = []
    for (source, seed), values in sorted(grouped.items()):
        labels = np.asarray([item[0] for item in values], dtype=np.int64)
        scores = np.asarray([item[1] for item in values], dtype=np.float64)
        output.append({
            "condition": condition,
            "source_id": source,
            "seed": int(seed),
            "sample_unit": "MANIP window within one source; fake=positive",
            "n_total": int(labels.size),
            "n_fake": int(np.sum(labels == 1)),
            "n_real": int(np.sum(labels == 0)),
            "auroc": _roc_auc(labels, scores),
            "missing_score_windows": 0,
        })
    return output


def _seed_stability(rows: Sequence[Mapping[str, Any]], condition: str) -> dict[str, Any]:
    detailed = _source_seed_metrics(rows, condition)
    by_source: dict[str, list[float]] = {}
    for item in detailed:
        if item["auroc"] is not None:
            by_source.setdefault(str(item["source_id"]), []).append(float(item["auroc"]))
    source_stds = [float(np.std(values)) for values in by_source.values() if len(values) == len(SEEDS)]
    aurocs = [float(item["auroc"]) for item in detailed if item["auroc"] is not None]
    return {
        "condition": condition,
        "source_count": len(by_source),
        "seed_count_per_source": len(SEEDS),
        "source_seed_rows": len(detailed),
        "source_seed_auroc_mean": float(np.mean(aurocs)) if aurocs else None,
        "source_seed_auroc_std": float(np.std(aurocs)) if aurocs else None,
        "source_seed_auroc_min": float(np.min(aurocs)) if aurocs else None,
        "source_seed_auroc_max": float(np.max(aurocs)) if aurocs else None,
        "mean_within_source_seed_auroc_std": float(np.mean(source_stds)) if source_stds else None,
        "max_within_source_seed_auroc_std": float(np.max(source_stds)) if source_stds else None,
        "missing_seed_rows": int(len(by_source) * len(SEEDS) - len(detailed)),
    }


def _completion(root: Path, records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    manifest = json.loads((root / "manifests/input_manifest.json").read_text(encoding="utf-8"))
    sources = sorted({str(row["source_id"]) for row in manifest["rows"]})
    expected = {(condition, source, seed) for condition in CONDITIONS for source in sources for seed in SEEDS}
    present = {(str(row["condition"]), str(row["held_out_source"]), int(row["seed"])) for row in records}
    rows: list[dict[str, Any]] = []
    for condition in CONDITIONS:
        condition_missing = sorted((source, seed) for cond, source, seed in expected - present if cond == condition)
        rows.append({
            "condition": condition,
            "expected_slots": len(sources) * len(SEEDS),
            "completed_slots": sum(1 for cond, _source, _seed in present if cond == condition),
            "missing_slots": len(condition_missing),
            "skipped_sources": ";".join(sorted({source for source, _seed in condition_missing})),
            "missing_seeds": ";".join(f"{source}:{seed}" for source, seed in condition_missing),
            "duplicate_keys": len(records) - len(present),
        })
    return rows


def _loss_summary(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for condition in CONDITIONS:
        histories = [np.asarray(row["fit"]["loss_history"], dtype=np.float64) for row in records if str(row["condition"]) == condition]
        finite_histories = [history for history in histories if history.size and np.all(np.isfinite(history))]
        converged = [history for history in finite_histories if float(history[-1]) < float(history[0])]
        output.append({
            "condition": condition,
            "record_count": len(histories),
            "finite_history_count": len(finite_histories),
            "converged_final_below_initial_count": len(converged),
            "initial_loss_mean": float(np.mean([history[0] for history in finite_histories])) if finite_histories else None,
            "final_loss_mean": float(np.mean([history[-1] for history in finite_histories])) if finite_histories else None,
            "final_minus_initial_mean": float(np.mean([history[-1] - history[0] for history in finite_histories])) if finite_histories else None,
            "min_loss_mean": float(np.mean([np.min(history) for history in finite_histories])) if finite_histories else None,
            "epochs_recorded": int(max((history.size for history in finite_histories), default=0)),
            "criterion": "descriptive only: finite loss and final loss below initial loss; no per-epoch classification metric was saved",
        })
    return output


def _plot_losses(root: Path, records: Sequence[Mapping[str, Any]]) -> Path:
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True)
    for axis, condition in zip(axes.flat, REPRESENTATIVE_CONDITIONS):
        histories = [np.asarray(row["fit"]["loss_history"], dtype=np.float64) for row in records if str(row["condition"]) == condition]
        matrix = np.stack([history for history in histories if history.size and np.all(np.isfinite(history))])
        x = np.arange(1, matrix.shape[1] + 1)
        median = np.median(matrix, axis=0); low = np.percentile(matrix, 10, axis=0); high = np.percentile(matrix, 90, axis=0)
        axis.plot(x, median, linewidth=2, label="median")
        axis.fill_between(x, low, high, alpha=0.2, label="10–90 percentile")
        axis.set_title(condition); axis.set_xlabel("epoch"); axis.set_ylabel("weighted BCE"); axis.grid(alpha=0.25)
    axes.flat[0].legend(loc="best", fontsize=8)
    figure.suptitle("V7 boundary/pooling pilot: representative saved training loss")
    figure.tight_layout()
    output = root / "visualizations/representative_loss_curves.png"; output.parent.mkdir(parents=True, exist_ok=True); figure.savefig(output, dpi=150); plt.close(figure)
    docs_output = REPO_ROOT / "docs/experiments/assets/v7-boundary-pooling-loss-curves.png"; docs_output.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(output, docs_output)
    return docs_output


def summarize(root: Path) -> dict[str, Any]:
    rows = _read_csv(root / "scores/oof_window_scores.csv")
    model_data = json.loads((root / "models/fold_models.json").read_text(encoding="utf-8"))
    records = model_data["records"]
    classification = [_classification_row(rows, condition) for condition in CONDITIONS]
    source_seed = [item for condition in CONDITIONS for item in _source_seed_metrics(rows, condition)]
    seed_stability = [_seed_stability(rows, condition) for condition in CONDITIONS]
    completion = _completion(root, records)
    losses = _loss_summary(records)
    _write_csv(root / "evaluation/classification_metrics.csv", classification)
    _atomic_json(root / "evaluation/classification_metrics.json", {"metrics": classification, "protocol": {"sample_unit": "MANIP window", "fake_positive": True, "threshold": "logit>=0", "ctrl_included": False, "missing": "NA/excluded, never zero-filled"}})
    _write_csv(root / "evaluation/completion_audit.csv", completion)
    _atomic_json(root / "evaluation/completion_audit.json", {"total_plan_slots": len(CONDITIONS) * len({str(row['source_id']) for row in json.loads((root / 'manifests/input_manifest.json').read_text())['rows']}) * len(SEEDS), "completed_records": len(records), "unique_keys": len({(row['condition'], row['held_out_source'], int(row['seed'])) for row in records}), "conditions": completion})
    _write_csv(root / "evaluation/loss_summary.csv", losses)
    _atomic_json(root / "evaluation/loss_summary.json", {"conditions": losses, "records": len(records), "per_epoch_auc_saved": False})
    _write_csv(root / "evaluation/per_source_seed_metrics.csv", source_seed)
    _write_csv(root / "evaluation/seed_stability.csv", seed_stability)
    _atomic_json(root / "evaluation/seed_stability.json", {"conditions": seed_stability, "sample_unit": "MANIP window, source-seed AUROC"})
    loss_plot = _plot_losses(root, records)
    for source_name, destination in (("classification_metrics.csv", REPO_ROOT / "docs/experiments/v7-boundary-pooling-classification-metrics.csv"), ("completion_audit.csv", REPO_ROOT / "docs/experiments/v7-boundary-pooling-completion-audit.csv"), ("loss_summary.csv", REPO_ROOT / "docs/experiments/v7-boundary-pooling-loss-summary.csv"), ("per_source_metrics.csv", REPO_ROOT / "docs/experiments/v7-boundary-pooling-per-source-metrics.csv"), ("paired_gains.csv", REPO_ROOT / "docs/experiments/v7-boundary-pooling-paired-gains.csv"), ("seed_stability.csv", REPO_ROOT / "docs/experiments/v7-boundary-pooling-seed-stability.csv")):
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text((root / "evaluation" / source_name).read_text(encoding="utf-8"), encoding="utf-8")
    return {"completed_records": len(records), "planned_records": len(CONDITIONS) * 16 * len(SEEDS), "classification": classification, "completion": completion, "losses": losses, "seed_stability": seed_stability, "loss_plot": str(loss_plot)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    args = parser.parse_args()
    print(json.dumps(summarize(args.root), indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
