"""Independent runner for the V7 five-time sequence representation pilot."""

from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import json
import os
import signal
import subprocess
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from sparse3d_forgery.particle_sequence import load_particle_sequence

from .model import (
    EPOCHS,
    LEARNING_RATE,
    SEEDS,
    WEIGHT_DECAY,
    FeatureStandardizer,
    SequenceMLP,
    SetAModel,
    fit_standardizer,
    make_batch,
    model_state,
    parameter_count,
    score_batch,
    source_class_weights,
    train_one,
)
from .representation import CONDITIONS, condition_feature_matrix, build_five_time_unit, deterministic_permutation, json_ready


REPO_ROOT = Path(__file__).resolve().parents[3]
DATA_ROOT = Path("/root/autodl-tmp/data/sparse_3d_forgery_detection")
INPUT_ROOT = DATA_ROOT / "derived/v7_activityforensics_local_organization_pilot_v1"
OUTPUT_ROOT = DATA_ROOT / "derived/v7_activityforensics_multi_order_sequence_pilot_v1"
BUDGET_S = 3600.0
BOOTSTRAP_SEED = 20260909
BOOTSTRAP_REPLICATES = 10_000
EXPECTED_SOURCES = 16
EXPECTED_WINDOWS = 192
STOP_REQUESTED = False


def _safe(value: str) -> str:
    return "".join(c if c.isalnum() or c in "._-" else "_" for c in value)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(json_ready(value), indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
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
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader(); writer.writerows(rows)
    os.replace(tmp, path)


def _git_head() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()


class Budget:
    """Persistent additive budget; historical time is not counted twice."""

    def __init__(self, root: Path, budget_s: float) -> None:
        self.path = root / "state" / "budget.json"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.is_file():
            old = json.loads(self.path.read_text(encoding="utf-8"))
            if abs(float(old.get("budget_s", budget_s)) - float(budget_s)) > 1e-9:
                raise ValueError("BUDGET_ARGUMENT_MISMATCH")
            self.before = float(old.get("cumulative_s", 0.0))
        else:
            self.before = 0.0
        self.budget_s = float(budget_s)
        self.started = time.monotonic()

    def elapsed(self) -> float:
        return self.before + (time.monotonic() - self.started)

    def remaining(self) -> float:
        return max(0.0, self.budget_s - self.elapsed())

    def save(self, *, stop_reason: str | None = None) -> None:
        _write_json(self.path, {"budget_s": self.budget_s, "elapsed_before_this_process_s": self.before, "process_start_unix": time.time() - (time.monotonic() - self.started), "cumulative_s": self.elapsed(), "stop_reason": stop_reason})


def _progress(root: Path, *, stage: str, status: str, completed: int = 0, total: int = 0, budget: Budget | None = None, **extra: Any) -> None:
    row = {"stage": stage, "status": status, "completed": int(completed), "total": int(total), "updated_unix": time.time(), "cumulative_elapsed_s": budget.elapsed() if budget else None, **extra}
    _write_json(root / "progress.json", row)


def _load_input() -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    manifest = json.loads((INPUT_ROOT / "manifests/input_manifest.json").read_text(encoding="utf-8"))
    rows = list(manifest["rows"])
    if len(rows) != EXPECTED_WINDOWS:
        raise ValueError(f"EXPECTED_192_WINDOWS:{len(rows)}")
    if len({str(row["source_id"]) for row in rows}) != EXPECTED_SOURCES:
        raise ValueError("EXPECTED_16_SOURCES")
    mapping: dict[str, list[dict[str, Any]]] = defaultdict(list)
    with (INPUT_ROOT / "manifests/local_group_mapping.csv").open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if str(row.get("retained", "")).lower() == "true" and str(row.get("status")) == "RETAINED":
                row["member_slots"] = json.loads(row["member_slots"])
                mapping[str(row["window_id"])].append(row)
    return rows, mapping


def _build_support_row(row: Mapping[str, Any], groups: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    prefix = Path(str(row["particle_prefix"]))
    sequence = load_particle_sequence(prefix)
    if not np.array_equal(sequence.frame_indices, np.asarray(row["frame_indices"], dtype=np.int64)):
        raise ValueError(f"FRAME_IDENTITY_MISMATCH:{row['window_id']}")
    if np.max(np.abs(sequence.timestamps_s - np.asarray(row["timestamps_s"], dtype=np.float64))) > 1e-7:
        raise ValueError(f"PTS_IDENTITY_MISMATCH:{row['window_id']}")
    units = [build_five_time_unit(sequence, window_id=str(row["window_id"]), window_start_s=float(row["interval_start_s"]), member_slots=group["member_slots"], local_group_id=int(group["local_group_id"])) for group in groups]
    valid_units = [unit for unit in units if unit.get("status") == "VALID"]
    target = valid_units[0]["timestamps_s"] if valid_units else next((item.get("timestamp_s") for item in []), None)
    if target is None and units:
        matches = units[0].get("target_matches", [])
        if matches and all(item.get("timestamp_s") is not None for item in matches):
            target = [float(item["timestamp_s"]) for item in matches]
    return {
        "window_id": str(row["window_id"]), "source_id": str(row["source_id"]), "pair_id": str(row["pair_id"]), "role": str(row["role"]), "kind": str(row["kind"]), "label": 0 if str(row["kind"]) == "MANIP" and str(row["role"]) == "real" else (1 if str(row["kind"]) == "MANIP" and str(row["role"]) == "fake" else None),
        "interval_start_s": float(row["interval_start_s"]), "interval_end_s": float(row["interval_end_s"]), "particle_prefix": str(prefix), "particle_npz_sha256": _sha256(prefix.with_suffix(".npz")), "particle_json_sha256": _sha256(prefix.with_suffix(".json")), "group_count": len(groups), "units": units, "valid_unit_count": len(valid_units), "target_timestamps_s": target, "permutation": list(deterministic_permutation(str(row["window_id"]))),
    }


def prepare(root: Path, budget: Budget) -> dict[str, Any]:
    existing = root / "support/window_support.json"
    if existing.is_file():
        rows = json.loads(existing.read_text(encoding="utf-8"))
        _progress(root, stage="prepare", status="COMPLETE", completed=len(rows), total=EXPECTED_WINDOWS, budget=budget, reused=True)
        return {"windows": len(rows), "valid_units": sum(int(row.get("valid_unit_count", 0)) for row in rows), "reused": True}
    rows, mapping = _load_input()
    support_rows = []
    for index, row in enumerate(rows, 1):
        support_rows.append(_build_support_row(row, mapping.get(str(row["window_id"]), [])))
        if index % 8 == 0:
            _progress(root, stage="prepare", status="RUNNING", completed=index, total=len(rows), budget=budget)
    _write_json(root / "input_manifest.json", {"source_manifest": str(INPUT_ROOT / "manifests/input_manifest.json"), "source_manifest_sha256": _sha256(INPUT_ROOT / "manifests/input_manifest.json"), "rows": rows, "git_head": _git_head()})
    _write_json(root / "execution_plan.json", {"source_order": sorted({str(row["source_id"]) for row in rows}), "window_count": len(rows), "window_ids": [str(row["window_id"]) for row in rows], "conditions": list(CONDITIONS), "training_seeds": list(SEEDS), "epochs": EPOCHS, "budget_s": budget.budget_s, "selection": "all frozen 192 windows in manifest order; no result-dependent filtering"})
    _write_json(root / "support/window_support.json", support_rows)
    summary = {"frozen_sources": EXPECTED_SOURCES, "frozen_windows": EXPECTED_WINDOWS, "valid_five_time_units": sum(int(row["valid_unit_count"]) for row in support_rows), "windows_with_valid_five_time_support": sum(bool(row["valid_unit_count"]) for row in support_rows), "windows_without_support": sum(not bool(row["valid_unit_count"]) for row in support_rows), "target_offsets_s": [float(x) for x in (0.5, 0.6, 0.7, 0.8, 0.9)], "source": str(INPUT_ROOT)}
    _write_json(root / "support/support_summary.json", summary)
    _progress(root, stage="prepare", status="COMPLETE", completed=len(rows), total=len(rows), budget=budget, **summary)
    return summary


def _write_npz_atomic(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    np.savez_compressed(temporary, **arrays)
    generated = temporary if temporary.exists() else temporary.with_suffix(temporary.suffix + ".npz")
    os.replace(generated, path)


def features(root: Path, budget: Budget) -> dict[str, Any]:
    support_rows = json.loads((root / "support/window_support.json").read_text(encoding="utf-8"))
    records = []
    for index, row in enumerate(support_rows, 1):
        path = root / "features" / f"{_safe(row['window_id'])}.npz"
        if not path.is_file():
            built = condition_feature_matrix(row, str(row["window_id"]))
            arrays = {condition: np.asarray(built["features"][condition], dtype=np.float64) if built["features"][condition] is not None else np.empty((0, 5, 4) if condition == "SET_A" else (0, 40), dtype=np.float64) for condition in CONDITIONS}
            _write_npz_atomic(path, **arrays)
        with np.load(path, allow_pickle=False) as archive:
            shapes = {condition: list(archive[condition].shape) for condition in CONDITIONS}
        records.append({"window_id": str(row["window_id"]), "path": str(path), "sha256": _sha256(path), "unit_count": int(row["valid_unit_count"]), "shapes": shapes, "permutation": list(row.get("permutation", []))})
        if index % 8 == 0:
            _progress(root, stage="features", status="RUNNING", completed=index, total=len(support_rows), budget=budget)
    _write_json(root / "features_manifest.json", {"conditions": list(CONDITIONS), "rows": records, "source_support_sha256": _sha256(root / "support/window_support.json")})
    _progress(root, stage="features", status="COMPLETE", completed=len(records), total=len(records), budget=budget)
    return {"windows": len(records), "unit_count": sum(int(row["unit_count"]) for row in records)}


def _load_feature_rows(root: Path, *, include_control: bool = True) -> list[dict[str, Any]]:
    supports = json.loads((root / "support/window_support.json").read_text(encoding="utf-8"))
    rows = []
    for row in supports:
        if not include_control and row.get("label") is None:
            continue
        path = root / "features" / f"{_safe(row['window_id'])}.npz"
        if not path.is_file() or int(row.get("valid_unit_count", 0)) <= 0:
            continue
        with np.load(path, allow_pickle=False) as archive:
            feature = {condition: archive[condition].astype(np.float64, copy=True) for condition in CONDITIONS}
        rows.append({**row, "features": feature, "intervals_s": np.diff(np.asarray(row["target_timestamps_s"], dtype=np.float64)) if row.get("target_timestamps_s") else np.empty(4, dtype=np.float64)})
    return rows


def _standardizer_record(value: Mapping[str, Any]) -> FeatureStandardizer:
    return FeatureStandardizer(str(value["condition"]), np.asarray(value["mean"], dtype=np.float64), np.asarray(value["scale"], dtype=np.float64), tuple(int(x) for x in value.get("zero_variance_dimensions", [])))


def _model_from_record(condition: str, record: Mapping[str, Any], device: str) -> Any:
    import torch
    model = SetAModel() if condition == "SET_A" else SequenceMLP()
    model.load_state_dict({name: torch.as_tensor(value, dtype=model.state_dict()[name].dtype) for name, value in record["state_dict"].items()})
    model.to(device); model.eval(); return model


def _save_model_records(root: Path, records: Sequence[Mapping[str, Any]]) -> None:
    _write_json(root / "models/fold_models.json", {"conditions": list(CONDITIONS), "seeds": list(SEEDS), "epochs": EPOCHS, "learning_rate": LEARNING_RATE, "weight_decay": WEIGHT_DECAY, "records": list(records)})


def train(root: Path, budget: Budget, *, device: str, resume: bool) -> dict[str, Any]:
    rows = _load_feature_rows(root, include_control=False)
    sources = sorted({str(row["source_id"]) for row in rows})
    model_path = root / "models/fold_models.json"
    records = json.loads(model_path.read_text(encoding="utf-8")).get("records", []) if resume and model_path.is_file() else []
    completed = {(str(item["condition"]), str(item["held_out_source"]), int(item["seed"])) for item in records}
    total = len(sources) * len(CONDITIONS) * len(SEEDS)
    _progress(root, stage="train", status="RUNNING", completed=len(completed), total=total, budget=budget, device=device)
    for held_out in sources:
        training = [row for row in rows if str(row["source_id"]) != held_out]
        if not training or {int(row["label"]) for row in training} != {0, 1}:
            continue
        weights = source_class_weights(training)
        for condition in CONDITIONS:
            if STOP_REQUESTED or budget.remaining() <= 0:
                budget.save(stop_reason="BUDGET_EXHAUSTED" if budget.remaining() <= 0 else "STOP_REQUESTED")
                _progress(root, stage="train", status="STOPPED", completed=len(completed), total=total, budget=budget, stop_reason="BUDGET_EXHAUSTED" if budget.remaining() <= 0 else "STOP_REQUESTED")
                return {"status": "STOPPED", "completed_models": len(completed), "total_models": total}
            matrices = [row["features"][condition] for row in training]
            standardizer = fit_standardizer(condition, matrices, weights)
            batch = make_batch(condition, training, standardizer)
            batch["window_weights"] = source_class_weights([{**row, "label": int(row["label"])} for row in training])
            # The standardizer receives one matrix per window; make_batch retains the same window order.
            for seed in SEEDS:
                key = (condition, held_out, int(seed))
                if key in completed:
                    continue
                if STOP_REQUESTED or budget.remaining() <= 0:
                    budget.save(stop_reason="BUDGET_EXHAUSTED" if budget.remaining() <= 0 else "STOP_REQUESTED")
                    _progress(root, stage="train", status="STOPPED", completed=len(completed), total=total, budget=budget, stop_reason="BUDGET_EXHAUSTED" if budget.remaining() <= 0 else "STOP_REQUESTED")
                    _save_model_records(root, records)
                    return {"status": "STOPPED", "completed_models": len(completed), "total_models": total}
                started = time.perf_counter()
                def _epoch_line(epoch: int, metrics: Mapping[str, Any]) -> None:
                    print(f"multi-order condition={condition} held_out={held_out} seed={seed} epoch={epoch}/{EPOCHS} weighted_bce={float(metrics['loss']):.6f} train_auroc={metrics.get('train_auroc')} train_ap={metrics.get('train_ap')} train_f1={metrics.get('train_f1')} elapsed={budget.elapsed():.1f}s remaining={budget.remaining():.1f}s", flush=True)
                model, fit = train_one(condition, batch, seed=int(seed), device=device, epochs=EPOCHS, epoch_callback=_epoch_line)
                record = {"condition": condition, "held_out_source": held_out, "seed": int(seed), "parameter_count": parameter_count(condition), "training_source_count": len({str(row["source_id"]) for row in training}), "training_real_count": sum(int(row["label"]) == 0 for row in training), "training_fake_count": sum(int(row["label"]) == 1 for row in training), "training_window_count": len(training), "training_unit_count": int(batch["unit_count"]), "standardization": standardizer.as_dict(), "fit": fit, "elapsed_s": time.perf_counter() - started, "state_dict": model_state(model)}
                records.append(record); completed.add(key); _save_model_records(root, records)
                with (root / "models/loss_history.jsonl").open("a", encoding="utf-8") as handle:
                    for epoch_row in fit["loss_history"]:
                        handle.write(json.dumps({"condition": condition, "held_out_source": held_out, "seed": int(seed), **epoch_row}, allow_nan=False) + "\n")
                print(f"multi-order model condition={condition} held_out={held_out} seed={seed} final_loss={fit['final_loss']:.6f} elapsed={fit['epochs']} epochs", flush=True)
                _progress(root, stage="train", status="RUNNING", completed=len(completed), total=total, budget=budget, current_condition=condition, current_held_out_source=held_out, current_seed=int(seed))
    status = "COMPLETE" if len(completed) == total else "PARTIAL"
    budget.save(stop_reason=None if status == "COMPLETE" else "INCOMPLETE")
    _progress(root, stage="train", status=status, completed=len(completed), total=total, budget=budget)
    return {"status": status, "completed_models": len(completed), "total_models": total}


def _auroc(labels: Sequence[int], scores: Sequence[float]) -> float | None:
    y = np.asarray(labels, dtype=np.int64); s = np.asarray(scores, dtype=np.float64)
    if np.sum(y == 0) == 0 or np.sum(y == 1) == 0:
        return None
    pos, neg = s[y == 1], s[y == 0]
    return float(np.mean((pos[:, None] > neg[None, :]) + 0.5 * (pos[:, None] == neg[None, :])))


def _ap(labels: Sequence[int], scores: Sequence[float]) -> float | None:
    y = np.asarray(labels, dtype=np.int64); s = np.asarray(scores, dtype=np.float64)
    if not np.any(y == 1) or not np.any(y == 0): return None
    order = np.argsort(-s, kind="mergesort"); yy = y[order]; tp = np.cumsum(yy == 1); precision = tp / np.arange(1, len(yy) + 1)
    return float(np.sum(precision[yy == 1]) / np.sum(yy == 1))


def _classification(labels: Sequence[int], scores: Sequence[float]) -> dict[str, Any]:
    y = np.asarray(labels, dtype=np.int64); pred = np.asarray(scores, dtype=np.float64) >= 0.0
    tn = int(np.sum((y == 0) & ~pred)); fp = int(np.sum((y == 0) & pred)); fn = int(np.sum((y == 1) & ~pred)); tp = int(np.sum((y == 1) & pred)); precision = tp / (tp + fp) if tp + fp else None; recall = tp / (tp + fn) if tp + fn else None; f1 = 2 * precision * recall / (precision + recall) if precision is not None and recall is not None and precision + recall else None
    return {"precision": precision, "recall": recall, "f1": f1, "accuracy": (tn + tp) / len(y) if len(y) else None, "tn": tn, "fp": fp, "fn": fn, "tp": tp}


def _bootstrap(source_values: Mapping[str, Mapping[str, float]], left: str, right: str | None = None) -> dict[str, Any]:
    sources = sorted(
        set(source_values.get(left, {}))
        & (set(source_values.get(right, {})) if right else set(source_values.get(left, {})))
    )
    if not sources:
        return {"source_count": 0, "mean": None, "ci95": None, "seed": BOOTSTRAP_SEED, "replicates": BOOTSTRAP_REPLICATES}
    values = np.asarray([source_values[left][source] - source_values[right][source] if right else source_values[left][source] for source in sources], dtype=np.float64)
    rng = np.random.default_rng(BOOTSTRAP_SEED); draws = values[rng.integers(0, len(values), size=(BOOTSTRAP_REPLICATES, len(values)))].mean(axis=1)
    return {"source_count": len(sources), "sources": sources, "mean": float(np.mean(values)), "ci95": [float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))], "seed": BOOTSTRAP_SEED, "replicates": BOOTSTRAP_REPLICATES}


def evaluate(root: Path, budget: Budget, *, device: str) -> dict[str, Any]:
    rows = _load_feature_rows(root, include_control=True)
    all_support_rows = json.loads((root / "support/window_support.json").read_text(encoding="utf-8"))
    data = json.loads((root / "models/fold_models.json").read_text(encoding="utf-8")) if (root / "models/fold_models.json").is_file() else {"records": []}
    records = {(str(item["condition"]), str(item["held_out_source"]), int(item["seed"])): item for item in data.get("records", [])}
    output_rows: dict[str, dict[str, Any]] = {str(row["window_id"]): {k: row.get(k) for k in ("window_id", "source_id", "pair_id", "role", "kind", "label", "interval_start_s", "interval_end_s", "valid_unit_count")} for row in all_support_rows}
    for source in sorted({str(row["source_id"]) for row in rows}):
        held = [row for row in rows if str(row["source_id"]) == source]
        for condition in CONDITIONS:
            for seed in SEEDS:
                record = records.get((condition, source, int(seed)))
                if record is None:
                    continue
                model = _model_from_record(condition, record, device)
                standardizer = _standardizer_record(record["standardization"])
                batch = make_batch(condition, held, standardizer, require_labels=False)
                scores = score_batch(condition, model, batch, device)
                for window_id, score in zip(batch["window_ids"], scores):
                    output_rows[window_id][f"{condition}_seed_{seed}"] = float(score)
                del model
    final_rows = []
    for row in output_rows.values():
        for condition in CONDITIONS:
            values = [row.get(f"{condition}_seed_{seed}") for seed in SEEDS]
            if all(value is not None and np.isfinite(float(value)) for value in values):
                row[condition] = float(np.mean(np.asarray(values, dtype=np.float64))); row[f"{condition}_seed_std"] = float(np.std(np.asarray(values, dtype=np.float64))); row[f"{condition}_status"] = "SCORED"
            else:
                row[condition] = ""; row[f"{condition}_status"] = "MISSING_SEED_MODEL"
        final_rows.append(row)
    _write_csv(root / "scores/oof_window_scores.csv", final_rows)
    source_values: dict[str, dict[str, float]] = {condition: {} for condition in CONDITIONS}
    per_source: list[dict[str, Any]] = []
    for source in sorted({str(row["source_id"]) for row in final_rows}):
        source_rows = [row for row in final_rows if str(row["source_id"]) == source and row.get("kind") == "MANIP" and row.get("label") in (0, 1)]
        source_out = {"source_id": source}
        for condition in CONDITIONS:
            eligible = [row for row in source_rows if row.get(f"{condition}_status") == "SCORED"]
            labels = [int(row["label"]) for row in eligible]; scores = [float(row[condition]) for row in eligible]
            value = _auroc(labels, scores)
            source_out[f"{condition}_auroc"] = value; source_out[f"{condition}_real_count"] = sum(label == 0 for label in labels); source_out[f"{condition}_fake_count"] = sum(label == 1 for label in labels)
            if value is not None: source_values[condition][source] = value
        for left, right, name in (("MULTI_ORDER_SEQ", "RAW_SEQ", "MULTI_MINUS_RAW"), ("MULTI_ORDER_SEQ", "SHUFFLED_MULTI_ORDER", "MULTI_MINUS_SHUFFLED"), ("RAW_SEQ", "SET_A", "RAW_MINUS_SET_A"), ("MULTI_ORDER_SEQ", "SET_A", "MULTI_MINUS_SET_A")):
            source_out[name] = source_values[left].get(source, np.nan) - source_values[right].get(source, np.nan) if source in source_values[left] and source in source_values[right] else ""
        per_source.append(source_out)
    _write_csv(root / "evaluation/per_source_metrics.csv", per_source)
    summary: dict[str, Any] = {"experiment": "V7 five-time raw-state and multi-order sequence matched pilot", "population": {"source_count": len({str(row['source_id']) for row in final_rows}), "window_count": len(final_rows), "valid_feature_windows": sum(int(row.get("valid_unit_count") or 0) > 0 for row in final_rows)}, "conditions": {}, "paired_comparisons": {}, "limitations": ["five-time support is an offline survivor filter", "overlapping windows and source correlation remain", "development LOSO is not a sealed test", "no spatial ground truth or physical velocity claim"]}
    for condition in CONDITIONS:
        values = source_values[condition]
        eligible = [row for row in final_rows if row.get("kind") == "MANIP" and row.get(f"{condition}_status") == "SCORED" and row.get("label") in (0, 1)]
        labels = [int(row["label"]) for row in eligible]; scores = [float(row[condition]) for row in eligible]
        summary["conditions"][condition] = {"source_auroc": _bootstrap(source_values, condition), "pooled_auroc": _auroc(labels, scores), "pooled_ap": _ap(labels, scores), "classification_logit_ge_0": _classification(labels, scores), "window_count": len(eligible), "source_count": len(values), "parameter_count": parameter_count(condition)}
    for left, right, name in (("MULTI_ORDER_SEQ", "RAW_SEQ", "MULTI_MINUS_RAW"), ("MULTI_ORDER_SEQ", "SHUFFLED_MULTI_ORDER", "MULTI_MINUS_SHUFFLED"), ("RAW_SEQ", "SET_A", "RAW_MINUS_SET_A"), ("MULTI_ORDER_SEQ", "SET_A", "MULTI_MINUS_SET_A")):
        summary["paired_comparisons"][name] = _bootstrap(source_values, left, right)
    _write_json(root / "evaluation/summary.json", summary)
    _progress(root, stage="evaluate", status="COMPLETE", completed=len(final_rows), total=len(final_rows), budget=budget, scored_windows=len(final_rows))
    return summary


def report(root: Path, budget: Budget) -> dict[str, Any]:
    summary_path = root / "evaluation/summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.is_file() else {}
    support = json.loads((root / "support/support_summary.json").read_text(encoding="utf-8")) if (root / "support/support_summary.json").is_file() else {}
    models = json.loads((root / "models/fold_models.json").read_text(encoding="utf-8")) if (root / "models/fold_models.json").is_file() else {"records": []}
    lines = ["# V7 五时刻原始状态序列与多阶状态序列匹配对照 pilot", "", "本报告由当前产物生成；它不是正式检测链、不是 sealed-test，也不声称恢复物理点对应。", "", "## 直接回答", "", f"- 实际完成 source/window/model：{summary.get('population', {}).get('source_count', 'NA')} source、{summary.get('population', {}).get('window_count', 'NA')} 窗口、{len(models.get('records', []))} 个 fold/condition/seed 模型。", f"- 五时刻共同支撑：{support.get('windows_with_valid_five_time_support', 'NA')}/{support.get('frozen_windows', 'NA')} 窗口，valid unit={support.get('valid_five_time_units', 'NA')}；无效支撑不填零。", f"- MULTI_ORDER_SEQ−RAW_SEQ：{summary.get('paired_comparisons', {}).get('MULTI_MINUS_RAW', {})}", f"- MULTI_ORDER_SEQ−SHUFFLED_MULTI_ORDER：{summary.get('paired_comparisons', {}).get('MULTI_MINUS_SHUFFLED', {})}", "- 少数 source 的方向不能替代 source 等权总体；逐 source 结果见 evaluation/per_source_metrics.csv。", "- 支持的是当前五时刻共同可测关系上的表示对照；不支持物理速度/加速度、像素定位、跨生成器泛化或严格在线因果训练。", "", "## 条件摘要", "", "| condition | source AUROC | pooled AUROC | pooled AP | source count | parameter count |", "|---|---:|---:|---:|---:|---:|"]
    for condition in CONDITIONS:
        item = summary.get("conditions", {}).get(condition, {}); macro = item.get("source_auroc", {}); lines.append(f"| {condition} | {macro.get('mean', 'NA')} [{macro.get('ci95', 'NA')}] | {item.get('pooled_auroc', 'NA')} | {item.get('pooled_ap', 'NA')} | {item.get('source_count', 'NA')} | {item.get('parameter_count', 'NA')} |")
    lines += ["", "## 冻结协议与限制", "", "- 输入来自既有 16 source/192 window 的 289-query缓存及历史 H 局部组；未重跑 tracking、depth、pose 或 segmentation。", "- 五个实际 PTS 必须严格递增；相同五时刻共同成员、pair 和历史尺度用于一个 local unit。五时刻支持是事后筛选，存在 survivor bias。", "- SET_A 对五个 S 做共享编码和时间平均；三个序列条件使用相同 40→16→8→1 MLP。SHUFFLED 在 S 上按窗口冻结的标签无关非恒等排列后重新计算 v/a。", "- MANIP real=0、fake=1；CTRL 不进训练，若有评分仅作描述。LOSO 标准化和 source/class weighted BCE 只使用训练 source。", f"- 预算累计：{budget.elapsed():.3f}s / {budget.budget_s:.3f}s；设备由模型记录；详细 loss 在 models/loss_history.jsonl。", "", "## 产物", "", "`protocol.json`、`input_manifest.json`、`support/`、`features_manifest.json`、`models/`、`scores/oof_window_scores.csv`、`evaluation/summary.json` 和 `evaluation/per_source_metrics.csv`。", ""]
    (root / "report.md").write_text("\n".join(lines), encoding="utf-8")
    _progress(root, stage="report", status="COMPLETE", completed=1, total=1, budget=budget)
    return {"report": str(root / "report.md"), "model_count": len(models.get("records", []))}


def protocol(root: Path) -> None:
    _write_json(root / "protocol.json", {"experiment": "V7 five-time raw-state and multi-order sequence matched pilot", "git_head": _git_head(), "input_root": str(INPUT_ROOT), "output_root": str(root), "population": "16 frozen source pairs, 192 windows, 289 queries", "target_offsets_s": [0.5, 0.6, 0.7, 0.8, 0.9], "conditions": list(CONDITIONS), "model": {"sequence": "40->16->8->1", "set_a": "shared 4->16->8 encoder, mean over five states, interval-aware 12->16->8->1 head", "epochs": EPOCHS, "optimizer": "Adam", "learning_rate": LEARNING_RATE, "weight_decay": WEIGHT_DECAY, "seeds": list(SEEDS), "budget_s": BUDGET_S}, "evaluation": {"primary": ["MULTI_ORDER_SEQ-RAW_SEQ", "MULTI_ORDER_SEQ-SHUFFLED_MULTI_ORDER"], "bootstrap": {"unit": "source", "seed": BOOTSTRAP_SEED, "replicates": BOOTSTRAP_REPLICATES}}, "boundaries": ["research-only pilot", "no formal src changes", "no frontend rerun", "not spatial localization or physical motion estimation"]})


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("prepare", "features", "train", "evaluate", "report", "all"))
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--budget-s", type=float, default=BUDGET_S)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    root = args.output_root
    root.mkdir(parents=True, exist_ok=True)
    lock_path = root / ".run.lock"; lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("w") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit("RUN_LOCK_HELD")
        budget = Budget(root, args.budget_s)
        def stop_handler(signum: int, frame: Any) -> None:
            nonlocal budget
            global STOP_REQUESTED
            STOP_REQUESTED = True
            budget.save(stop_reason="STOP_REQUESTED")
        signal.signal(signal.SIGTERM, stop_handler); signal.signal(signal.SIGINT, stop_handler)
        _write_json(root / "resolved_config.json", {"git_head": _git_head(), "device": args.device, "budget_s": args.budget_s, "input_root": str(INPUT_ROOT), "output_root": str(root), "resume": bool(args.resume)})
        if args.stage in ("prepare", "all"):
            protocol(root); prepare(root, budget)
        if args.stage in ("features", "all"):
            features(root, budget)
        if args.stage in ("train", "all"):
            train(root, budget, device=args.device, resume=args.resume)
        if args.stage in ("evaluate", "all") and (root / "models/fold_models.json").is_file():
            evaluate(root, budget, device=args.device)
        if args.stage in ("report", "all"):
            report(root, budget)
        status = "COMPLETE" if args.stage == "all" and (root / "evaluation/summary.json").is_file() else "STAGE_COMPLETE"
        if STOP_REQUESTED: status = "STOP_REQUESTED"
        elif budget.remaining() <= 0: status = "BUDGET_EXHAUSTED"
        budget.save(stop_reason=None if status in ("COMPLETE", "STAGE_COMPLETE") else status)
        _write_json(root / "final_status.json", {"status": status, "updated_unix": time.time(), "cumulative_elapsed_s": budget.elapsed(), "budget_s": budget.budget_s, "stop_reason": None if status in ("COMPLETE", "STAGE_COMPLETE") else status})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
