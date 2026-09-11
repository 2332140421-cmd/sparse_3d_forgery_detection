"""Reuse the frozen V7 MLP/LOSO training path for dense features."""

from __future__ import annotations

import csv
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from research_tools.v7.local_structural_temporal_probe.model import (
    MODEL_CONFIG,
    build_batch,
    fit_weighted_standardizer,
    score_model,
    serialize_model,
    train_model,
    validate_training_examples,
    window_label,
)


ARMS = ("UNORDERED_STATE", "ORDERED_SECOND", "PERMUTED_SECOND")
SEEDS = (20260909, 20260910, 20260911)
BOOTSTRAP_SEED = 20260909
BOOTSTRAP_REPLICATES = 10_000


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row}) or ["status"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def load_feature_examples(feature_root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    examples: list[dict[str, Any]] = []
    coverage: list[dict[str, Any]] = []
    for meta_path in sorted(feature_root.glob("*.json")):
        if meta_path.name == "index.json":
            continue
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        coverage.append({key: value for key, value in meta.items() if key not in {"triplets"}})
        if meta.get("status") != "VALID":
            continue
        npz_path = meta_path.with_suffix(".npz")
        arrays = np.load(npz_path, allow_pickle=False)
        states = np.asarray(arrays["states"], dtype=np.float64)
        timestamps = np.asarray(arrays["timestamps_s"], dtype=np.float64)
        component_ids = np.asarray(arrays["component_ids"], dtype=np.int64)
        triplets = []
        for index, triplet_meta in enumerate(meta.get("triplets", [])):
            triplets.append(
                {
                    "triplet_id": int(index),
                    "component_index": int(component_ids[index]),
                    "states": states[index],
                    "timestamps_s": timestamps[index],
                }
            )
        examples.append({
            "window_id": str(meta["window_id"]),
            "pair_id": str(meta["pair_id"]),
            "source_id": str(meta["source_id"]),
            "role": str(meta["role"]),
            "kind": str(meta["kind"]),
            "triplets": triplets,
        })
    return examples, coverage


def _auroc(positive: Sequence[float], negative: Sequence[float]) -> float | None:
    pos = np.asarray([float(value) for value in positive if np.isfinite(value)], dtype=np.float64)
    neg = np.asarray([float(value) for value in negative if np.isfinite(value)], dtype=np.float64)
    if pos.size == 0 or neg.size == 0:
        return None
    comparison = (pos[:, None] > neg[None, :]).astype(np.float64)
    comparison += 0.5 * (pos[:, None] == neg[None, :])
    return float(np.mean(comparison))


def _source_metrics(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        if row.get("kind") == "MANIP" and row.get("role") in {"real", "fake"}:
            for arm in ARMS:
                score = row.get(f"{arm}_score")
                if score is not None:
                    grouped[str(row["source_id"])][f"{arm}_{row['role']}"].append(float(score))
    output = []
    for source_id in sorted(grouped):
        row: dict[str, Any] = {"source_id": source_id}
        for arm in ARMS:
            row[f"{arm}_real_count"] = len(grouped[source_id].get(f"{arm}_real", []))
            row[f"{arm}_fake_count"] = len(grouped[source_id].get(f"{arm}_fake", []))
            row[f"{arm}_auroc"] = _auroc(grouped[source_id].get(f"{arm}_fake", []), grouped[source_id].get(f"{arm}_real", []))
        c, a, d = row.get("ORDERED_SECOND_auroc"), row.get("UNORDERED_STATE_auroc"), row.get("PERMUTED_SECOND_auroc")
        row["C-A"] = c - a if c is not None and a is not None else None
        row["C-D"] = c - d if c is not None and d is not None else None
        output.append(row)
    return output


def _bootstrap(source_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    complete = [row for row in source_rows if all(row.get(f"{arm}_auroc") is not None for arm in ARMS)]
    if not complete:
        return {"N": 0, "seed": BOOTSTRAP_SEED, "replicates": BOOTSTRAP_REPLICATES, "arms": {}, "gains": {}}
    values = {arm: np.asarray([float(row[f"{arm}_auroc"]) for row in complete]) for arm in ARMS}
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    indices = rng.integers(0, len(complete), size=(BOOTSTRAP_REPLICATES, len(complete)))
    means = {arm: np.mean(value[indices], axis=1) for arm, value in values.items()}
    gains = {"C-A": means["ORDERED_SECOND"] - means["UNORDERED_STATE"], "C-D": means["ORDERED_SECOND"] - means["PERMUTED_SECOND"]}
    return {
        "N": len(complete),
        "source_ids": [str(row["source_id"]) for row in complete],
        "seed": BOOTSTRAP_SEED,
        "replicates": BOOTSTRAP_REPLICATES,
        "arms": {arm: {"mean": float(np.mean(values[arm])), "ci95": [float(np.percentile(means[arm], 2.5)), float(np.percentile(means[arm], 97.5))]} for arm in ARMS},
        "gains": {name: {"mean": float(np.mean(array)), "ci95": [float(np.percentile(array, 2.5)), float(np.percentile(array, 97.5))]} for name, array in gains.items()},
    }


def train_and_evaluate(
    examples: Sequence[Mapping[str, Any]],
    output_root: Path,
    *,
    device: str = "cuda",
    time_limit_s: float | None = None,
) -> dict[str, Any]:
    started = time.perf_counter()
    examples = list(examples)
    sources = sorted({str(example["source_id"]) for example in examples})
    labeled = [example for example in examples if window_label(example) is not None]
    oof: dict[str, dict[str, Any]] = {
        str(example["window_id"]): {key: example[key] for key in ("window_id", "pair_id", "source_id", "role", "kind")}
        for example in examples
    }
    fold_rows: list[dict[str, Any]] = []
    model_rows: list[dict[str, Any]] = []
    for held_out in sources:
        if time_limit_s is not None and time.perf_counter() - started >= float(time_limit_s):
            fold_rows.append({"held_out_source": held_out, "status": "TRAINING_BUDGET_EXHAUSTED"})
            break
        heldout = [example for example in examples if str(example["source_id"]) == held_out]
        training = [example for example in labeled if str(example["source_id"]) != held_out]
        if not heldout:
            fold_rows.append({"held_out_source": held_out, "status": "NO_VALID_SUPPORT_WINDOWS"})
            continue
        try:
            validate_training_examples(training)
        except ValueError as exc:
            fold_rows.append({"held_out_source": held_out, "status": "TRAINING_SUPPORT_INVALID", "reason": str(exc), "training_real_count": sum(window_label(item) == 0 for item in training), "training_fake_count": sum(window_label(item) == 1 for item in training)})
            continue
        fold_rows.append({"held_out_source": held_out, "status": "SCORED", "training_real_count": sum(window_label(item) == 0 for item in training), "training_fake_count": sum(window_label(item) == 1 for item in training), "held_out_window_count": len(heldout)})
        for arm in ARMS:
            raw_batch, observation_weights = build_batch(training, arm, observation_weights=True)
            if observation_weights is None:
                raise RuntimeError("source/class observation weights missing")
            standardizer = fit_weighted_standardizer(raw_batch.inputs, observation_weights)
            train_batch, _ = build_batch(training, arm, standardizer=standardizer, observation_weights=True)
            held_batch, _ = build_batch(heldout, arm, standardizer=standardizer)
            for seed in SEEDS:
                model, fit = train_model(train_batch, seed=seed, device=device)
                scores = score_model(model, held_batch)
                for example, score in zip(heldout, scores):
                    oof[str(example["window_id"])][f"{arm}_score_seed_{seed}"] = float(score)
                model_rows.append({"held_out_source": held_out, "arm": arm, "seed": seed, "training_real_count": sum(window_label(item) == 0 for item in training), "training_fake_count": sum(window_label(item) == 1 for item in training), "training_triplet_count": train_batch.n_triplets, "held_out_triplet_count": held_batch.n_triplets, "zero_variance_dimensions": list(standardizer.zero_variance_dimensions), "fit": fit, "model": serialize_model(model, standardizer)})
                del model
    oof_rows = []
    for row in oof.values():
        for arm in ARMS:
            values = [row.get(f"{arm}_score_seed_{seed}") for seed in SEEDS]
            finite = [float(value) for value in values if value is not None]
            row[f"{arm}_score"] = float(np.mean(finite)) if len(finite) == len(SEEDS) else None
            row[f"{arm}_seed_std"] = float(np.std(finite)) if len(finite) == len(SEEDS) else None
        oof_rows.append(row)
    source_rows = _source_metrics(oof_rows)
    bootstrap = _bootstrap(source_rows)
    complete_sources = bootstrap.get("N", 0)
    c_ci = bootstrap.get("arms", {}).get("ORDERED_SECOND", {}).get("ci95")
    ca_ci = bootstrap.get("gains", {}).get("C-A", {}).get("ci95")
    cd_ci = bootstrap.get("gains", {}).get("C-D", {}).get("ci95")
    if complete_sources < 12:
        status = "MEASUREMENT_SUPPORT_INSUFFICIENT"
    elif c_ci and ca_ci and cd_ci and c_ci[0] > 0.5 and ca_ci[0] > 0 and cd_ci[0] > 0:
        status = "LOCAL_TEMPORAL_DISCRIMINATION_SUPPORTED_IN_DEVELOPMENT"
    elif c_ci and c_ci[0] > 0.5:
        status = "DISCRIMINATION_WITHOUT_ESTABLISHED_TEMPORAL_GAIN"
    else:
        status = "LOCAL_TEMPORAL_DISCRIMINATION_NOT_ESTABLISHED"
    _write_csv(output_root / "oof_window_scores.csv", oof_rows)
    _write_csv(output_root / "per_source_metrics.csv", source_rows)
    _write_csv(output_root / "fold_support.csv", fold_rows)
    _write_json(output_root / "model_records.json", {"model_config": MODEL_CONFIG, "seeds": list(SEEDS), "folds": model_rows})
    summary = {"status": status, "arms": list(ARMS), "sources": sources, "oof_window_count": len(oof_rows), "source_metrics": source_rows, "bootstrap": bootstrap, "fold_support": fold_rows, "training_elapsed_s": time.perf_counter() - started, "training_budget_s": time_limit_s, "ctrl_excluded_from_training": True, "formal_src_modified": False, "training_device": device}
    _write_json(output_root / "training_summary.json", summary)
    return summary
