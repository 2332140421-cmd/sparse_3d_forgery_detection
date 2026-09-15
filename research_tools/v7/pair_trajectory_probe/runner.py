"""V7 fixed-pair distance trajectory matched pilot.

This module reads only the already materialized periodic re-query R sequences
and their frozen five-time support.  It deliberately lives under research
tools rather than the formal ``src`` detection chain.
"""

from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import json
import math
import os
import signal
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[3]
DATA_ROOT = Path("/root/autodl-tmp/data/sparse_3d_forgery_detection")
INPUT_ROOT = DATA_ROOT / "derived/v7_activityforensics_periodic_requery_pilot_v1"
SUMMARY_ROOT = DATA_ROOT / "derived/v7_activityforensics_requery_training_support_pilot_v1"
OUTPUT_ROOT = DATA_ROOT / "derived/v7_activityforensics_pair_trajectory_pilot_v1"
SEEDS = (20260909, 20260910, 20260911)
ID_SHUFFLE_SEED = 20260909
TIME_SHUFFLE_SEED = 20260909
EPOCHS = 200
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
BUDGET_S = 1800.0
BOOTSTRAP_SEED = 20260909
BOOTSTRAP_REPLICATES = 10_000
CONDITIONS = ("PAIR_SEQ", "PAIR_ID_SHUFFLE", "PAIR_TIME_SHUFFLE")
STOP_REQUESTED = False


def _safe(value: str) -> str:
    return "".join(c if c.isalnum() or c in "._-" else "_" for c in value)


def _jsonable(value: Any, path: str = "$") -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return [_jsonable(item, f"{path}[{index}]") for index, item in enumerate(value)]
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return _jsonable(value.item(), path)
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"non-finite JSON value at {path}: {value!r}")
        return value
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    if isinstance(value, tuple):
        return [_jsonable(item, f"{path}[{index}]") for index, item in enumerate(value)]
    if isinstance(value, list):
        return [_jsonable(item, f"{path}[{index}]") for index, item in enumerate(value)]
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item, f"{path}.{key}") for key, item in value.items()}
    raise TypeError(f"unsupported JSON value at {path}: {type(value).__name__}")


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(_jsonable(value), indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
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
        writer.writeheader()
        for row in rows:
            writer.writerow({key: ("" if value is None else value) for key, value in row.items()})
    os.replace(tmp, path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_head() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()


def _progress(root: Path, stage: str, status: str, completed: int, total: int, **extra: Any) -> None:
    _atomic_json(root / "progress.json", {"stage": stage, "status": status, "completed": int(completed), "total": int(total), "updated_unix": time.time(), **extra})


@dataclass
class Budget:
    root: Path
    budget_s: float
    started: float

    @classmethod
    def load(cls, root: Path, budget_s: float) -> "Budget":
        path = root / "state/budget.json"
        previous = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
        before = float(previous.get("elapsed_before_this_process_s", 0.0))
        item = cls(root, float(previous.get("budget_s", budget_s)), time.monotonic())
        item._before = before  # type: ignore[attr-defined]
        return item

    def elapsed(self) -> float:
        return float(getattr(self, "_before", 0.0) + time.monotonic() - self.started)

    def remaining(self) -> float:
        return max(0.0, self.budget_s - self.elapsed())

    def save(self, stop_reason: str | None = None, **extra: Any) -> None:
        # Persist the cumulative value as the next process' baseline.  The
        # baseline itself remains fixed in memory for this process, so
        # repeated saves do not double-count the current process duration.
        cumulative = self.elapsed()
        _atomic_json(self.root / "state/budget.json", {"budget_s": self.budget_s, "elapsed_before_this_process_s": cumulative, "process_elapsed_s": float(time.monotonic() - self.started), "cumulative_s": cumulative, "stop_reason": stop_reason, **extra})


def _load_support_rows() -> list[dict[str, Any]]:
    path = INPUT_ROOT / "support/window_support.json"
    rows = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise ValueError("window_support.json must be a list")
    return [dict(row) for row in rows if str(row.get("mode")) == "R" and row.get("support_status") == "VALID" and int(row.get("valid_unit_count", 0)) > 0 and row.get("label") in (0, 1)]


def _load_baseline_scores() -> dict[str, float]:
    path = SUMMARY_ROOT / "scores/r_set_support_comparison.csv"
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    out: dict[str, float] = {}
    for row in rows:
        value = row.get("new_R_SET_ALL_VALID_TRAIN")
        if value not in (None, "") and math.isfinite(float(value)):
            out[str(row["window_id"])] = float(value)
    return out


def _load_sequence(prefix: str) -> Any:
    from sparse3d_forgery.particle_sequence import load_particle_sequence

    return load_particle_sequence(prefix)


def _stable_permutation(length: int, *parts: Any, seed: int) -> np.ndarray:
    token = "v7-pair-trajectory|" + "|".join(str(item) for item in parts) + f"|seed={seed}"
    digest = hashlib.sha256(token.encode("utf-8")).digest()
    value = int.from_bytes(digest[:8], "little", signed=False)
    permutation = np.random.default_rng(value).permutation(length).astype(np.int64)
    if length > 1 and np.array_equal(permutation, np.arange(length, dtype=np.int64)):
        permutation[:2] = permutation[1], permutation[0]
    return permutation


def _summary_from_distances(distances: np.ndarray) -> np.ndarray:
    if distances.ndim != 2 or distances.shape[1] != 5:
        raise ValueError("distance matrix must have shape [pair,5]")
    return np.stack((distances.mean(axis=0), distances.std(axis=0), np.percentile(distances, 25, axis=0), np.percentile(distances, 75, axis=0)), axis=1)


def _reconstruct_unit(row: Mapping[str, Any], unit: Mapping[str, Any], sequence: Any, stored_state: np.ndarray | None) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    frame_indices = np.asarray(unit["array_indices"], dtype=np.int64)
    pair_indices = np.asarray(unit["pair_indices"], dtype=np.int64)
    if frame_indices.shape != (5,) or pair_indices.ndim != 2 or pair_indices.shape[1] != 2:
        raise ValueError(f"invalid stored support shape for {row['window_id']} unit={unit.get('local_group_id')}")
    if np.any(pair_indices[:, 0] >= pair_indices[:, 1]):
        raise ValueError(f"non-canonical pair indices for {row['window_id']} unit={unit.get('local_group_id')}")
    xyz = np.asarray(sequence.xyz, dtype=np.float64)
    valid = np.asarray(sequence.geometry_validity, dtype=bool)
    if np.any(frame_indices < 0) or np.any(frame_indices >= xyz.shape[0]):
        raise ValueError(f"frame index outside sequence for {row['window_id']}")
    left_valid = valid[frame_indices[:, None], pair_indices[:, 0][None, :]]
    right_valid = valid[frame_indices[:, None], pair_indices[:, 1][None, :]]
    if not np.all(left_valid & right_valid):
        raise ValueError(f"stored valid pair has invalid geometry for {row['window_id']}")
    scale = float(unit["history_scale"])
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError(f"invalid history scale for {row['window_id']}")
    distances = np.stack([np.linalg.norm(xyz[frame_indices, right] - xyz[frame_indices, left], axis=1) / scale for left, right in pair_indices], axis=0)
    if not np.all(np.isfinite(distances)):
        raise ValueError(f"non-finite reconstructed distance for {row['window_id']}")
    timestamps = np.asarray(sequence.timestamps_s[frame_indices], dtype=np.float64)
    intervals = np.diff(timestamps)
    if intervals.shape != (4,) or not np.all(np.isfinite(intervals)) or not np.all(intervals > 0):
        raise ValueError(f"invalid five-time intervals for {row['window_id']}")
    states = _summary_from_distances(distances)
    max_diff = None
    if stored_state is not None:
        stored = np.asarray(stored_state, dtype=np.float64)
        if stored.shape != (5, 4):
            raise ValueError(f"stored SET_A state shape mismatch for {row['window_id']}")
        max_diff = float(np.max(np.abs(states - stored)))
        if max_diff > 1e-5:
            raise ValueError(f"stored S(t) mismatch for {row['window_id']} unit={unit.get('local_group_id')} max_diff={max_diff}")
    track_ids = np.asarray(sequence.track_ids, dtype=np.int64)
    pair_ids = [tuple(sorted((int(track_ids[left]), int(track_ids[right])))) for left, right in pair_indices]
    stored_pair_ids = [tuple(sorted((int(pair[0]), int(pair[1])))) for pair in unit.get("pair_ids", [])]
    if pair_ids != stored_pair_ids:
        raise ValueError(f"stored canonical pair identity mismatch for {row['window_id']} unit={unit.get('local_group_id')}")
    identity = {
        "unit_id": f"{row['window_id']}::R::local_{int(unit['local_group_id'])}",
        "window_id": str(row["window_id"]),
        "source_id": str(row["source_id"]),
        "parent_id": str(row.get("pair_id", "")),
        "query_cohort": "R",
        "local_group_id": int(unit["local_group_id"]),
        "member_slots": [int(x) for x in unit.get("member_slots", [])],
        "track_ids": [int(x) for x in unit.get("track_ids", [])],
        "pair_ids": [list(pair) for pair in pair_ids],
        "frame_indices": [int(x) for x in sequence.frame_indices[frame_indices]],
        "timestamps_s": [float(x) for x in timestamps],
        "history_scale": scale,
        "history_scale_pair_time_count": int(unit.get("history_scale_pair_time_count", 0)),
        "s_reconstruction_max_abs": max_diff,
    }
    return distances, np.repeat(intervals[None, :], distances.shape[0], axis=0), identity


def _build_relation_dataset(root: Path) -> dict[str, Any]:
    rows = _load_support_rows()
    baseline = _load_baseline_scores()
    if not rows:
        raise ValueError("no valid R windows")
    distances: list[np.ndarray] = []
    intervals: list[np.ndarray] = []
    relation_window: list[int] = []
    relation_unit: list[int] = []
    pair_ids: list[list[int]] = []
    units: list[dict[str, Any]] = []
    windows: list[dict[str, Any]] = []
    reconstruction_diffs: list[float] = []
    for window_index, row in enumerate(sorted(rows, key=lambda item: str(item["window_id"]))):
        sequence = _load_sequence(str(row["particle_prefix"]))
        valid_units = [unit for unit in row.get("support", {}).get("units", []) if unit.get("status") == "VALID"]
        stored_states = np.asarray(row["features"]["SET_A"], dtype=np.float64)
        if stored_states.shape[0] != len(valid_units):
            raise ValueError(f"valid unit count mismatch for {row['window_id']}")
        window_unit_start = len(units)
        window_relation_start = sum(item.shape[0] for item in distances)
        for unit_index_local, (unit, state) in enumerate(zip(valid_units, stored_states)):
            dist, ints, identity = _reconstruct_unit(row, unit, sequence, state)
            global_unit = len(units)
            identity["window_unit_index"] = int(unit_index_local)
            identity["relation_start"] = int(window_relation_start + sum(item.shape[0] for item in distances[window_unit_start:]))
            identity["relation_end"] = int(identity["relation_start"] + dist.shape[0])
            units.append(identity)
            distances.append(dist)
            intervals.append(ints)
            relation_window.extend([window_index] * dist.shape[0])
            relation_unit.extend([global_unit] * dist.shape[0])
            pair_ids.extend([list(pair) for pair in identity["pair_ids"]])
            if identity["s_reconstruction_max_abs"] is not None:
                reconstruction_diffs.append(float(identity["s_reconstruction_max_abs"]))
        window_relation_end = sum(item.shape[0] for item in distances)
        windows.append({
            "window_id": str(row["window_id"]), "source_id": str(row["source_id"]), "role": str(row["role"]), "label": int(row["label"]),
            "kind": str(row.get("kind", "MANIP")), "annotation_category": str(row.get("annotation_category", "")), "pair_id": str(row.get("pair_id", "")),
            "parent_id": str(row.get("parent_id", row.get("pair_id", ""))), "offset_s": float(row["offset_s"]), "interval_start_s": float(row["interval_start_s"]),
            "interval_end_s": float(row["interval_end_s"]), "target_timestamps_s": [float(x) for x in row.get("timestamps_s", [])],
            "particle_prefix": str(row["particle_prefix"]), "unit_start": int(window_unit_start), "unit_end": int(len(units)),
            "relation_start": int(window_relation_start), "relation_end": int(window_relation_end), "unit_count": int(len(units) - window_unit_start),
            "relation_count": int(window_relation_end - window_relation_start), "baseline_summary_set": baseline.get(str(row["window_id"])),
        })
    dist_array = np.concatenate(distances, axis=0).astype(np.float64)
    interval_array = np.concatenate(intervals, axis=0).astype(np.float64)
    if not np.all(np.isfinite(dist_array)) or not np.all(np.isfinite(interval_array)):
        raise ValueError("non-finite relation input")
    root.joinpath("relations").mkdir(parents=True, exist_ok=True)
    np.savez_compressed(root / "relations/relation_features.npz", distances=dist_array, intervals=interval_array, relation_window=np.asarray(relation_window, dtype=np.int64), relation_unit=np.asarray(relation_unit, dtype=np.int64), pair_track_ids=np.asarray(pair_ids, dtype=np.int64))
    _atomic_json(root / "relations/windows.json", windows)
    _atomic_json(root / "relations/units.json", units)
    _atomic_json(root / "relations/support_summary.json", {
        "valid_windows": len(windows), "valid_units": len(units), "relation_count": int(dist_array.shape[0]),
        "reconstruction_max_abs": float(max(reconstruction_diffs, default=0.0)), "reconstruction_tolerance": 1e-5,
        "reconstruction_checked_units": len(reconstruction_diffs), "same_five_time_support": True,
    })
    manifest_rows = []
    for window in windows:
        manifest_rows.append({key: window.get(key) for key in ("window_id", "source_id", "role", "label", "offset_s", "unit_count", "relation_count", "particle_prefix", "baseline_summary_set")})
    _write_csv(root / "relations/window_manifest.csv", manifest_rows)
    return {"windows": windows, "units": units, "distances": dist_array, "intervals": interval_array, "relation_window": np.asarray(relation_window, dtype=np.int64), "relation_unit": np.asarray(relation_unit, dtype=np.int64), "pair_track_ids": np.asarray(pair_ids, dtype=np.int64)}


def _load_relation_dataset(root: Path) -> dict[str, Any]:
    arrays = np.load(root / "relations/relation_features.npz", allow_pickle=False)
    return {"windows": json.loads((root / "relations/windows.json").read_text(encoding="utf-8")), "units": json.loads((root / "relations/units.json").read_text(encoding="utf-8")), **{key: arrays[key] for key in arrays.files}}


def _condition_features(data: Mapping[str, Any], condition: str) -> tuple[np.ndarray, dict[str, Any]]:
    distances = np.asarray(data["distances"], dtype=np.float64)
    intervals = np.asarray(data["intervals"], dtype=np.float64)
    window_index = np.asarray(data["relation_window"], dtype=np.int64)
    unit_index = np.asarray(data["relation_unit"], dtype=np.int64)
    output = distances.copy()
    permutation_log: dict[str, Any] = {"condition": condition, "fixed_seed": ID_SHUFFLE_SEED if condition == "PAIR_ID_SHUFFLE" else TIME_SHUFFLE_SEED if condition == "PAIR_TIME_SHUFFLE" else None, "window_permutations": {}, "unit_permutations": {}}
    if condition == "PAIR_ID_SHUFFLE":
        changed_entries = 0
        changed_trajectories = 0
        total_trajectories = 0
        for unit_idx, unit in enumerate(data["units"]):
            start, end = int(unit["relation_start"]), int(unit["relation_end"])
            original = output[start:end].copy()
            perms: list[list[int]] = []
            for time_index in range(1, 5):
                perm = _stable_permutation(end - start, unit["window_id"], unit["local_group_id"], time_index, seed=ID_SHUFFLE_SEED)
                output[start:end, time_index] = original[perm, time_index]
                perms.append([int(x) for x in perm])
            changed = np.any(np.abs(output[start:end, 1:] - original[:, 1:]) > 0.0, axis=1)
            changed_entries += int(np.sum(np.abs(output[start:end, 1:] - original[:, 1:]) > 0.0))
            changed_trajectories += int(np.sum(changed))
            total_trajectories += int(end - start)
            permutation_log["unit_permutations"][unit["unit_id"]] = perms
        permutation_log["changed_entry_ratio"] = float(changed_entries / max(1, sum(int(u["relation_end"]) - int(u["relation_start"]) for u in data["units"]) * 4))
        permutation_log["changed_trajectory_ratio"] = float(changed_trajectories / max(1, total_trajectories))
    elif condition == "PAIR_TIME_SHUFFLE":
        changed_windows = 0
        for window in data["windows"]:
            permutation = _stable_permutation(5, window["window_id"], "time", seed=TIME_SHUFFLE_SEED)
            permutation_log["window_permutations"][window["window_id"]] = [int(x) for x in permutation]
            if not np.array_equal(permutation, np.arange(5, dtype=np.int64)):
                changed_windows += 1
                start, end = int(window["relation_start"]), int(window["relation_end"])
                output[start:end] = output[start:end][:, permutation]
        permutation_log["changed_window_ratio"] = float(changed_windows / max(1, len(data["windows"])))
        permutation_log["intervals_unchanged"] = True
    elif condition != "PAIR_SEQ":
        raise ValueError(condition)
    features = np.concatenate((output, intervals), axis=1)
    if features.shape[1] != 9 or not np.all(np.isfinite(features)):
        raise ValueError(f"invalid {condition} relation features")
    return features, permutation_log


def _validate_transforms(data: Mapping[str, Any]) -> dict[str, Any]:
    original = np.asarray(data["distances"], dtype=np.float64)
    summary_original = [_summary_from_distances(original[int(w["relation_start"]):int(w["relation_end"]), :]) for w in data["units"]]
    id_features, id_log = _condition_features(data, "PAIR_ID_SHUFFLE")
    time_features, time_log = _condition_features(data, "PAIR_TIME_SHUFFLE")
    id_distance = id_features[:, :5]
    time_distance = time_features[:, :5]
    unit_checks = []
    for unit_index, unit in enumerate(data["units"]):
        start, end = int(unit["relation_start"]), int(unit["relation_end"])
        for t in range(5):
            if not np.array_equal(np.sort(original[start:end, t]), np.sort(id_distance[start:end, t])):
                raise ValueError(f"ID shuffle changed distance multiset at unit={unit['unit_id']} t={t}")
        if not np.allclose(_summary_from_distances(id_distance[start:end]), summary_original[unit_index], rtol=0, atol=1e-12):
            raise ValueError(f"ID shuffle changed S(t) at unit={unit['unit_id']}")
        for relation in range(end - start):
            if not np.array_equal(np.sort(original[start + relation]), np.sort(time_distance[start + relation])):
                raise ValueError(f"TIME shuffle changed relation distance multiset at unit={unit['unit_id']}")
        unit_checks.append({"unit_id": unit["unit_id"], "id_multiset_preserved": True, "id_summary_preserved": True, "time_relation_multiset_preserved": True})
    return {"id_shuffle": id_log, "time_shuffle": time_log, "unit_checks": unit_checks, "all_checks_passed": True}


@dataclass(frozen=True)
class DistanceStandardizer:
    mean: np.ndarray
    scale: np.ndarray
    zero_variance_dimensions: tuple[int, ...]
    weighting: dict[str, Any]

    def transform(self, features: np.ndarray) -> np.ndarray:
        result = np.asarray(features, dtype=np.float64).copy()
        result[:, :5] = (result[:, :5] - self.mean[None, :]) / self.scale[None, :]
        return result

    def as_dict(self) -> dict[str, Any]:
        return {"mean": self.mean.tolist(), "scale": self.scale.tolist(), "zero_variance_dimensions": list(self.zero_variance_dimensions), "weighting": self.weighting}


def _window_weights(windows: Sequence[Mapping[str, Any]]) -> np.ndarray:
    if not windows:
        raise ValueError("no windows")
    source_class = [(str(row["source_id"]), int(row["label"])) for row in windows]
    sources = sorted({source for source, _ in source_class})
    counts: dict[tuple[str, int], int] = {}
    for key in source_class:
        counts[key] = counts.get(key, 0) + 1
    raw = np.asarray([1.0 / (2 * len(sources) * counts[key]) for key in source_class], dtype=np.float64)
    return raw * (len(raw) / raw.sum())


def _fit_distance_standardizer(data: Mapping[str, Any], window_indices: Sequence[int]) -> DistanceStandardizer:
    windows = [data["windows"][int(index)] for index in window_indices]
    weights = _window_weights(windows)
    sums = np.zeros(5, dtype=np.float64)
    sums2 = np.zeros(5, dtype=np.float64)
    total = np.zeros(5, dtype=np.float64)
    for row_weight, window_index in zip(weights, window_indices):
        window = data["windows"][int(window_index)]
        unit_count = int(window["unit_count"])
        for unit_index in range(int(window["unit_start"]), int(window["unit_end"])):
            unit = data["units"][unit_index]
            start, end = int(unit["relation_start"]), int(unit["relation_end"])
            relation_weight = float(row_weight) / max(1, unit_count) / max(1, end - start)
            values = np.asarray(data["distances"][start:end], dtype=np.float64)
            sums += relation_weight * values.sum(axis=0)
            sums2 += relation_weight * np.square(values).sum(axis=0)
            total += relation_weight * float(end - start)
    mean = sums / total
    variance = np.maximum(sums2 / total - mean * mean, 0.0)
    zero = tuple(int(x) for x in np.flatnonzero(~np.isfinite(variance) | (variance <= 1e-12)))
    scale = np.sqrt(np.maximum(variance, 1e-12)); scale[list(zero)] = 1.0
    return DistanceStandardizer(mean, scale, zero, {"window": "equal source/class weighted", "unit": "equal within window", "pair": "equal within unit", "time": "channel-wise equal", "training_window_count": len(windows)})


def _batch(data: Mapping[str, Any], window_indices: Sequence[int], features: np.ndarray, standardizer: DistanceStandardizer | None, *, require_labels: bool = True) -> dict[str, Any]:
    selected = [int(index) for index in window_indices]
    relation_rows: list[int] = []
    relation_unit: list[int] = []
    unit_window: list[int] = []
    labels: list[int] = []
    window_ids: list[str] = []
    for local_window, window_index in enumerate(selected):
        window = data["windows"][window_index]
        label = window.get("label")
        if label is None and require_labels:
            raise ValueError("training label missing")
        labels.append(-1 if label is None else int(label))
        window_ids.append(str(window["window_id"]))
        for unit_index in range(int(window["unit_start"]), int(window["unit_end"])):
            unit = data["units"][unit_index]
            start, end = int(unit["relation_start"]), int(unit["relation_end"])
            relation_rows.extend(range(start, end))
            relation_unit.extend([len(unit_window)] * (end - start))
            unit_window.append(local_window)
    matrix = np.asarray(features[relation_rows], dtype=np.float64)
    if standardizer is not None:
        matrix = standardizer.transform(matrix)
    if not np.all(np.isfinite(matrix)):
        raise ValueError("non-finite relation batch")
    return {"features": matrix, "relation_unit_index": np.asarray(relation_unit, dtype=np.int64), "unit_window_index": np.repeat(np.arange(len(unit_window), dtype=np.int64), 1) if False else np.asarray(unit_window, dtype=np.int64), "labels": np.asarray(labels, dtype=np.int64), "window_ids": window_ids, "window_count": len(selected), "unit_count": len(unit_window)}


def _aggregate(values: Any, index: Any, count: int) -> Any:
    import torch

    sums = torch.zeros(count, dtype=values.dtype, device=values.device)
    counts = torch.zeros(count, dtype=values.dtype, device=values.device)
    sums.index_add_(0, index, values)
    counts.index_add_(0, index, torch.ones_like(values))
    return sums / counts.clamp_min(1.0)


def _aggregate_unit_window(unit_logits: Any, unit_window_index: Any, count: int) -> Any:
    return _aggregate(unit_logits, unit_window_index, count)


def _aggregate_window_from_relation(relation_logits: Any, relation_unit_index: Any, unit_window_index: Any, window_count: int, unit_count: int) -> Any:
    return _aggregate(_aggregate(relation_logits, relation_unit_index, unit_count), unit_window_index, window_count)


def _model_state(model: Any) -> dict[str, Any]:
    return {name: value.detach().cpu().numpy().tolist() for name, value in model.state_dict().items()}


def _classification(labels: Sequence[int], scores: Sequence[float]) -> dict[str, Any]:
    y = np.asarray(labels, dtype=np.int64); s = np.asarray(scores, dtype=np.float64); pred = s >= 0.0
    tn = int(np.sum((y == 0) & ~pred)); fp = int(np.sum((y == 0) & pred)); fn = int(np.sum((y == 1) & ~pred)); tp = int(np.sum((y == 1) & pred))
    precision = tp / (tp + fp) if tp + fp else None; recall = tp / (tp + fn) if tp + fn else None
    f1 = 2 * precision * recall / (precision + recall) if precision is not None and recall is not None and precision + recall else None
    return {"precision": precision, "recall": recall, "f1": f1, "accuracy": (tn + tp) / len(y) if len(y) else None, "tn": tn, "fp": fp, "fn": fn, "tp": tp}


def _auroc(labels: Sequence[int], scores: Sequence[float]) -> float | None:
    y = np.asarray(labels, dtype=np.int64); s = np.asarray(scores, dtype=np.float64)
    if np.sum(y == 0) == 0 or np.sum(y == 1) == 0:
        return None
    pos, neg = s[y == 1], s[y == 0]
    return float(np.mean((pos[:, None] > neg[None, :]) + 0.5 * (pos[:, None] == neg[None, :])))


def _average_precision(labels: Sequence[int], scores: Sequence[float]) -> float | None:
    y = np.asarray(labels, dtype=np.int64); s = np.asarray(scores, dtype=np.float64)
    if not np.any(y == 1) or not np.any(y == 0):
        return None
    order = np.argsort(-s, kind="mergesort"); ordered = y[order]; tp = np.cumsum(ordered == 1); precision = tp / np.arange(1, len(ordered) + 1)
    return float(np.sum(precision[ordered == 1]) / np.sum(ordered == 1))


def _metric_summary(rows: Sequence[Mapping[str, Any]], condition: str, score_key: str | None = None, sources: Sequence[str] | None = None) -> tuple[dict[str, Any], dict[str, float]]:
    selected = [row for row in rows if sources is None or str(row["source_id"]) in set(sources)]
    key = score_key or condition
    source_values: dict[str, float] = {}
    for source in sorted({str(row["source_id"]) for row in selected}):
        subset = [row for row in selected if str(row["source_id"]) == source and row.get(key) is not None]
        labels = [int(row["label"]) for row in subset]; scores = [float(row[key]) for row in subset]
        value = _auroc(labels, scores)
        if value is not None:
            source_values[source] = value
    eligible = [row for row in selected if row.get(key) is not None]
    labels = [int(row["label"]) for row in eligible]; scores = [float(row[key]) for row in eligible]
    source_count = len(source_values)
    return {"source_count": source_count, "window_count": len(eligible), "real_count": sum(x == 0 for x in labels), "fake_count": sum(x == 1 for x in labels), "source_auroc": float(np.mean(list(source_values.values()))) if source_values else None, "pooled_auroc": _auroc(labels, scores), "pooled_ap": _average_precision(labels, scores), **_classification(labels, scores)}, source_values


def _paired_bootstrap(left: Mapping[str, float], right: Mapping[str, float] | None = None) -> dict[str, Any]:
    sources = sorted(set(left) & (set(right) if right is not None else set(left)))
    if not sources:
        return {"source_count": 0, "mean": None, "ci95": None, "seed": BOOTSTRAP_SEED, "replicates": BOOTSTRAP_REPLICATES}
    raw = np.asarray([left[source] - right[source] if right is not None else left[source] for source in sources], dtype=np.float64)
    rng = np.random.default_rng(BOOTSTRAP_SEED); draws = raw[rng.integers(0, len(raw), size=(BOOTSTRAP_REPLICATES, len(raw)))].mean(axis=1)
    return {"source_count": len(sources), "sources": sources, "mean": float(np.mean(raw)), "ci95": [float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))], "seed": BOOTSTRAP_SEED, "replicates": BOOTSTRAP_REPLICATES}


def _model_class() -> Any:
    import torch.nn as nn

    class PairTrajectoryMLP(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.net = nn.Sequential(nn.Linear(9, 16), nn.ReLU(), nn.Linear(16, 8), nn.ReLU(), nn.Linear(8, 1))

        def forward(self, values: Any) -> Any:
            return self.net(values).squeeze(-1)

    return PairTrajectoryMLP


def _parameter_count() -> int:
    model = _model_class()()
    return int(sum(parameter.numel() for parameter in model.parameters()))


def _train_one(batch: Mapping[str, Any], seed: int, device: str, epochs: int, callback: Any | None = None) -> tuple[Any, dict[str, Any]]:
    import torch
    import torch.nn as nn

    torch.manual_seed(int(seed)); np.random.seed(int(seed) & 0xFFFFFFFF)
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA_UNAVAILABLE")
    model = _model_class()().to(torch.device(device)); model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    values = torch.as_tensor(batch["features"], dtype=torch.float32, device=device)
    relation_unit = torch.as_tensor(batch["relation_unit_index"], dtype=torch.long, device=device)
    unit_window = torch.as_tensor(batch["unit_window_index"], dtype=torch.long, device=device)
    labels = torch.as_tensor(batch["labels"], dtype=torch.float32, device=device)
    weights = torch.as_tensor(batch["window_weights"], dtype=torch.float32, device=device)
    history: list[dict[str, float]] = []
    for epoch in range(1, int(epochs) + 1):
        optimizer.zero_grad(set_to_none=True)
        relation_logits = model(values)
        window_logits = _aggregate_window_from_relation(relation_logits, relation_unit, unit_window, int(batch["window_count"]), int(batch["unit_count"]))
        loss = torch.sum(nn.functional.binary_cross_entropy_with_logits(window_logits, labels, reduction="none") * weights) / torch.sum(weights)
        loss.backward(); optimizer.step()
        record = {"epoch": epoch, "loss": float(loss.detach().cpu())}
        history.append(record)
        if callback is not None:
            callback(epoch, record)
    model.eval()
    return model, {"seed": int(seed), "epochs": int(epochs), "initial_loss": history[0]["loss"], "final_loss": history[-1]["loss"], "min_loss": min(row["loss"] for row in history), "loss_history": history, "device": str(device), "parameter_count": _parameter_count()}


def _score(model: Any, batch: Mapping[str, Any], device: str) -> np.ndarray:
    import torch

    values = torch.as_tensor(batch["features"], dtype=torch.float32, device=device)
    relation_unit = torch.as_tensor(batch["relation_unit_index"], dtype=torch.long, device=device)
    unit_window = torch.as_tensor(batch["unit_window_index"], dtype=torch.long, device=device)
    with torch.no_grad():
        logits = model(values)
        return _aggregate_window_from_relation(logits, relation_unit, unit_window, int(batch["window_count"]), int(batch["unit_count"])).detach().cpu().numpy().astype(np.float64)


def _model_from_record(record: Mapping[str, Any], device: str) -> Any:
    import torch

    model = _model_class()()
    state = {name: torch.as_tensor(value, dtype=model.state_dict()[name].dtype) for name, value in record["state_dict"].items()}
    model.load_state_dict(state); model.to(device); model.eval()
    return model


def _prepare(root: Path, budget: Budget) -> dict[str, Any]:
    root.mkdir(parents=True, exist_ok=True)
    _progress(root, "prepare", "RUNNING", 0, 1)
    data = _build_relation_dataset(root)
    transform_checks = _validate_transforms(data)
    _atomic_json(root / "transforms.json", transform_checks)
    windows = data["windows"]
    source_roles: dict[str, set[str]] = {}
    for row in windows:
        source_roles.setdefault(str(row["source_id"]), set()).add(str(row["role"]))
    main_sources = sorted(source for source, roles in source_roles.items() if roles == {"real", "fake"})
    main_indices = [index for index, row in enumerate(windows) if str(row["source_id"]) in main_sources]
    old_support = {str(row["window_id"]): row for row in json.loads((INPUT_ROOT / "support/window_support.json").read_text(encoding="utf-8")) if row.get("mode") == "O"}
    common_indices = [index for index in main_indices if int(old_support.get(windows[index]["window_id"], {}).get("valid_unit_count", 0)) > 0]
    input_manifest = {
        "git_head": _git_head(), "input_root": str(INPUT_ROOT), "summary_root": str(SUMMARY_ROOT),
        "files": {"support/window_support.json": _sha256(INPUT_ROOT / "support/window_support.json"), "summary/scores/r_set_support_comparison.csv": _sha256(SUMMARY_ROOT / "scores/r_set_support_comparison.csv"), "summary/models/fold_models.json": _sha256(SUMMARY_ROOT / "models/fold_models.json")},
        "query_cohort": "R", "sequence_identity": "video/source/parent/query-cohort/local-unit; no cross-cohort concatenation", "reconstruction_tolerance": 1e-5,
    }
    _atomic_json(root / "input_manifest.json", input_manifest)
    _atomic_json(root / "protocol.json", {
        "experiment": "V7 R fixed pair-distance trajectory matched pilot", "git_head": _git_head(), "input_root": str(INPUT_ROOT), "summary_baseline": "R_SET_ALL_VALID_TRAIN scores from existing supplement", "valid_r_windows": len(windows), "valid_r_units": len(data["units"]), "relation_count": int(data["distances"].shape[0]), "main_sources": main_sources, "main_window_count": len(main_indices), "main_real_count": sum(windows[i]["label"] == 0 for i in main_indices), "main_fake_count": sum(windows[i]["label"] == 1 for i in main_indices), "secondary_common_window_count": len(common_indices), "secondary_common_sources": sorted({str(windows[i]["source_id"]) for i in common_indices}),
        "conditions": {"SUMMARY_SET": "existing R_SET_ALL_VALID_TRAIN OOF logits", "PAIR_SEQ": "original canonical relation trajectories", "PAIR_ID_SHUFFLE": "first distance column fixed; later columns independently pair-row permuted per unit/time", "PAIR_TIME_SHUFFLE": "one stable non-identity time permutation per window; intervals unchanged"},
        "relation_input": "5 history-scaled distances + 4 actual adjacent PTS intervals", "model": "shared relation MLP 9->16->8->1; mean relation logit per unit; mean unit logit per window", "distance_standardization": "training-fold source/class weighted, equal window/unit/pair/time channel; shared across three new conditions", "interval_input": "actual seconds, not standardized", "seeds": list(SEEDS), "epochs": EPOCHS, "optimizer": "Adam", "learning_rate": LEARNING_RATE, "weight_decay": WEIGHT_DECAY, "source_disjoint": True, "bootstrap": {"unit": "source", "seed": BOOTSTRAP_SEED, "replicates": BOOTSTRAP_REPLICATES}, "budget_s": BUDGET_S, "boundaries": ["no frontend rerun", "no new detector chain", "no spatial ground truth claim", "no cross-query physical identity claim"],
    })
    _progress(root, "prepare", "COMPLETE", 1, 1, valid_windows=len(windows), valid_units=len(data["units"]), relation_count=int(data["distances"].shape[0]), main_sources=len(main_sources), main_windows=len(main_indices), common_windows=len(common_indices))
    budget.save(None, stage="prepare")
    return {**data, "main_sources": main_sources, "main_indices": main_indices, "common_indices": common_indices, "transform_checks": transform_checks}


def _run_smoke(root: Path, data: Mapping[str, Any], device: str) -> dict[str, Any]:
    features = {condition: _condition_features(data, condition)[0] for condition in CONDITIONS}
    heldout = sorted({str(data["windows"][index]["source_id"]) for index in range(len(data["windows"]))})[0]
    train_indices = [index for index, row in enumerate(data["windows"]) if str(row["source_id"]) != heldout]
    test_indices = [index for index, row in enumerate(data["windows"]) if str(row["source_id"]) == heldout]
    results = []
    for condition in CONDITIONS:
        standardizer = _fit_distance_standardizer(data, train_indices)
        batch = _batch(data, train_indices, features[condition], standardizer)
        batch["window_weights"] = _window_weights([data["windows"][index] for index in train_indices])
        model, fit = _train_one(batch, SEEDS[0], device, 1)
        test_batch = _batch(data, test_indices, features[condition], standardizer, require_labels=False)
        scores = _score(model, test_batch, device)
        results.append({"condition": condition, "held_out_source": heldout, "seed": SEEDS[0], "epochs": fit["epochs"], "training_windows": len(train_indices), "scored_windows": len(scores), "device": device, "scores_finite": bool(np.all(np.isfinite(scores)))})
        del model
    output = {"status": "PASS", "held_out_source": heldout, "results": results, "formal_model_persisted": False}
    _atomic_json(root / "smoke/summary.json", output)
    _atomic_json(root / "smoke/report.json", {"status": "PASS", "description": "isolated one-fold/one-seed/three-condition one-epoch path validation", "results": results})
    return output


def _train_formal(root: Path, data: Mapping[str, Any], budget: Budget, device: str, resume: bool) -> dict[str, Any]:
    import torch

    features = {condition: _condition_features(data, condition)[0] for condition in CONDITIONS}
    condition_logs = {condition: _condition_features(data, condition)[1] for condition in CONDITIONS}
    _atomic_json(root / "transform_manifest.json", condition_logs)
    source_list = sorted({str(row["source_id"]) for row in data["windows"]})
    records_path = root / "models/fold_models.json"
    existing = json.loads(records_path.read_text(encoding="utf-8")).get("records", []) if resume and records_path.is_file() else []
    records = list(existing)
    completed = {(str(row["condition"]), str(row["held_out_source"]), int(row["seed"])) for row in records}
    total = len(source_list) * len(CONDITIONS) * len(SEEDS)
    root.joinpath("models").mkdir(parents=True, exist_ok=True)
    _progress(root, "train", "RUNNING", len(completed), total, model_total=total, device=device)
    loss_path = root / "models/loss_history.jsonl"
    loss_handle = loss_path.open("a", encoding="utf-8")
    try:
        for heldout in source_list:
            train_indices = [index for index, row in enumerate(data["windows"]) if str(row["source_id"]) != heldout]
            if {int(data["windows"][index]["label"]) for index in train_indices} != {0, 1}:
                continue
            standardizer = _fit_distance_standardizer(data, train_indices)
            weights = _window_weights([data["windows"][index] for index in train_indices])
            for condition in CONDITIONS:
                for seed in SEEDS:
                    key = (condition, heldout, int(seed))
                    if key in completed:
                        continue
                    if STOP_REQUESTED or budget.remaining() <= 0:
                        reason = "STOP_REQUESTED" if STOP_REQUESTED else "BUDGET_EXHAUSTED"
                        budget.save(reason, stage="train", completed_models=len(completed), total_models=total)
                        _progress(root, "train", reason, len(completed), total, stop_reason=reason)
                        return {"status": reason, "completed_models": len(completed), "total_models": total}
                    batch = _batch(data, train_indices, features[condition], standardizer)
                    batch["window_weights"] = weights
                    started = time.perf_counter()
                    model, fit = _train_one(batch, int(seed), device, EPOCHS, callback=(lambda epoch, metrics, c=condition, h=heldout, s=seed: print(f"pair condition={c} held_out={h} seed={s} epoch={epoch}/{EPOCHS} loss={metrics['loss']:.6f} elapsed={budget.elapsed():.1f}s remaining={budget.remaining():.1f}s", flush=True)))
                    record = {"condition": condition, "held_out_source": heldout, "seed": int(seed), "parameter_count": _parameter_count(), "training_source_count": len({str(data["windows"][index]["source_id"]) for index in train_indices}), "training_window_count": len(train_indices), "training_real_count": sum(int(data["windows"][index]["label"]) == 0 for index in train_indices), "training_fake_count": sum(int(data["windows"][index]["label"]) == 1 for index in train_indices), "training_window_ids": [str(data["windows"][index]["window_id"]) for index in train_indices], "standardization": standardizer.as_dict(), "fit": fit, "elapsed_s": time.perf_counter() - started, "state_dict": _model_state(model), "device": device}
                    records.append(record); completed.add(key)
                    _atomic_json(records_path, {"conditions": list(CONDITIONS), "seeds": list(SEEDS), "epochs": EPOCHS, "learning_rate": LEARNING_RATE, "weight_decay": WEIGHT_DECAY, "parameter_count": _parameter_count(), "records": records})
                    for history in fit["loss_history"]:
                        loss_handle.write(json.dumps({"condition": condition, "held_out_source": heldout, "seed": int(seed), **history}, allow_nan=False) + "\n")
                    loss_handle.flush(); del model
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    _progress(root, "train", "RUNNING", len(completed), total, current_condition=condition, held_out_source=heldout, seed=int(seed), device=device)
    finally:
        loss_handle.close()
    # Keep an explicit, small training-window manifest independent of the
    # serialized model state.  This makes the source-disjoint split auditable
    # without loading any model tensors.
    windows_by_id = {str(row["window_id"]): row for row in data["windows"]}
    training_manifest: list[dict[str, Any]] = []
    for record in records:
        for window_id in record.get("training_window_ids", []):
            window = windows_by_id.get(str(window_id))
            if window is None:
                raise ValueError(f"training manifest references unknown window_id={window_id}")
            training_manifest.append({
                "condition": str(record["condition"]),
                "held_out_source": str(record["held_out_source"]),
                "seed": int(record["seed"]),
                "window_id": str(window_id),
                "source_id": str(window["source_id"]),
                "role": str(window["role"]),
                "label": int(window["label"]),
                "unit_count": int(window["unit_count"]),
                "relation_count": int(window["relation_count"]),
            })
    _write_csv(root / "models/training_window_manifest.csv", training_manifest)
    status = "COMPLETE" if len(completed) == total else "PARTIAL"
    budget.save(None if status == "COMPLETE" else status, stage="train", completed_models=len(completed), total_models=total)
    _progress(root, "train", status, len(completed), total, device=device)
    return {"status": status, "completed_models": len(completed), "total_models": total, "cumulative_s": budget.elapsed()}


def _evaluate(root: Path, data: Mapping[str, Any], budget: Budget, device: str) -> dict[str, Any]:
    model_data = json.loads((root / "models/fold_models.json").read_text(encoding="utf-8"))
    records = {(str(row["condition"]), str(row["held_out_source"]), int(row["seed"])): row for row in model_data.get("records", [])}
    features = {condition: _condition_features(data, condition)[0] for condition in CONDITIONS}
    baseline = _load_baseline_scores()
    score_rows = []
    for window_index, window in enumerate(data["windows"]):
        row = {key: window.get(key) for key in ("window_id", "source_id", "role", "label", "annotation_category", "pair_id", "parent_id", "offset_s", "interval_start_s", "interval_end_s", "unit_count", "relation_count")}
        row["SUMMARY_SET"] = baseline.get(str(window["window_id"]))
        for condition in CONDITIONS:
            for seed in SEEDS:
                record = records.get((condition, str(window["source_id"]), int(seed)))
                if record is None:
                    continue
                standardizer = DistanceStandardizer(np.asarray(record["standardization"]["mean"], dtype=np.float64), np.asarray(record["standardization"]["scale"], dtype=np.float64), tuple(int(x) for x in record["standardization"].get("zero_variance_dimensions", [])), dict(record["standardization"].get("weighting", {})))
                model = _model_from_record(record, device)
                batch = _batch(data, [window_index], features[condition], standardizer, require_labels=False)
                row[f"{condition}_seed_{seed}"] = float(_score(model, batch, device)[0])
                del model
            values = [row.get(f"{condition}_seed_{seed}") for seed in SEEDS]
            if all(value is not None and math.isfinite(float(value)) for value in values):
                row[condition] = float(np.mean(np.asarray(values, dtype=np.float64))); row[f"{condition}_seed_std"] = float(np.std(np.asarray(values, dtype=np.float64))); row[f"{condition}_status"] = "SCORED"
            else:
                row[condition] = None; row[f"{condition}_status"] = "MISSING_SEED_MODEL"
        score_rows.append(row)
        _progress(root, "evaluate", "RUNNING", window_index + 1, len(data["windows"]), scored_windows=window_index + 1)
    _write_csv(root / "scores/oof_window_scores.csv", score_rows)
    source_roles: dict[str, set[str]] = {}
    for row in data["windows"]:
        source_roles.setdefault(str(row["source_id"]), set()).add(str(row["role"]))
    main_sources = sorted(source for source, roles in source_roles.items() if roles == {"real", "fake"})
    main_rows = [row for row in score_rows if str(row["source_id"]) in main_sources]
    common_ids = {data["windows"][index]["window_id"] for index in data.get("common_indices", [])} if data.get("common_indices") is not None else set()
    secondary_rows = [row for row in main_rows if str(row["window_id"]) in common_ids]
    summaries: dict[str, Any] = {}
    source_values: dict[str, dict[str, float]] = {}
    conditions_all = ("SUMMARY_SET",) + CONDITIONS
    for condition in conditions_all:
        summary, values = _metric_summary(main_rows, condition, condition, main_sources)
        summary["evaluation_set"] = "R-valid label-qualified dual-role sources"; summaries[condition] = summary; source_values[condition] = values
    secondary_summaries = {}
    for condition in conditions_all:
        secondary_summaries[condition], _ = _metric_summary(secondary_rows, condition, condition, sorted({str(row["source_id"]) for row in secondary_rows}))
    per_source: list[dict[str, Any]] = []
    for source in main_sources:
        rows = [row for row in main_rows if str(row["source_id"]) == source]
        output = {"source_id": source, "window_count": len(rows), "real_count": sum(int(row["label"]) == 0 for row in rows), "fake_count": sum(int(row["label"]) == 1 for row in rows)}
        for condition in conditions_all:
            labels = [int(row["label"]) for row in rows if row.get(condition) is not None]; scores = [float(row[condition]) for row in rows if row.get(condition) is not None]
            output[f"{condition}_auroc"] = _auroc(labels, scores)
            output[f"{condition}_real_count"] = sum(label == 0 for label in labels); output[f"{condition}_fake_count"] = sum(label == 1 for label in labels)
        for left, right, name in (("PAIR_SEQ", "PAIR_ID_SHUFFLE", "PAIR_SEQ_MINUS_PAIR_ID_SHUFFLE"), ("PAIR_SEQ", "PAIR_TIME_SHUFFLE", "PAIR_SEQ_MINUS_PAIR_TIME_SHUFFLE"), ("PAIR_SEQ", "SUMMARY_SET", "PAIR_SEQ_MINUS_SUMMARY_SET")):
            left_value = output.get(f"{left}_auroc"); right_value = output.get(f"{right}_auroc"); output[name] = left_value - right_value if left_value is not None and right_value is not None else None
        per_source.append(output)
    per_seed: list[dict[str, Any]] = []
    for condition in CONDITIONS:
        for seed in SEEDS:
            key = f"{condition}_seed_{seed}"; summary, _ = _metric_summary(main_rows, key, key, main_sources)
            per_seed.append({"condition": condition, "seed": int(seed), **summary})
    paired = {
        "PAIR_SEQ_MINUS_PAIR_ID_SHUFFLE": _paired_bootstrap(source_values["PAIR_SEQ"], source_values["PAIR_ID_SHUFFLE"]),
        "PAIR_SEQ_MINUS_PAIR_TIME_SHUFFLE": _paired_bootstrap(source_values["PAIR_SEQ"], source_values["PAIR_TIME_SHUFFLE"]),
        "PAIR_SEQ_MINUS_SUMMARY_SET": _paired_bootstrap(source_values["PAIR_SEQ"], source_values["SUMMARY_SET"]),
    }
    _write_csv(root / "evaluation/per_source_metrics.csv", per_source)
    _write_csv(root / "evaluation/per_seed_metrics.csv", per_seed)
    _write_csv(root / "evaluation/paired_comparisons.csv", [{"comparison": name, **value} for name, value in paired.items()])
    summary = {"experiment": "V7 R fixed pair-distance trajectory matched pilot", "main_evaluation": {"source_count": len(main_sources), "window_count": len(main_rows), "real_count": sum(int(row["label"]) == 0 for row in main_rows), "fake_count": sum(int(row["label"]) == 1 for row in main_rows), "sources": main_sources}, "conditions": summaries, "secondary_common_14_source_76_window": secondary_summaries, "paired_comparisons": paired, "model_count": len(records), "all_formal_models_200_epochs": all(bool(row.get("fit", {}).get("epochs") == EPOCHS) for row in model_data.get("records", [])), "relation_support": {"valid_r_windows": len(data["windows"]), "valid_units": len(data["units"]), "relation_count": int(data["distances"].shape[0])}, "limitations": ["five-time support is an offline survivor filter", "overlapping windows are correlated", "small development source set", "SUMMARY_SET is a practical baseline with different model/input", "no spatial ground truth or cross-query physical correspondence claim"]}
    _atomic_json(root / "evaluation/summary.json", summary)
    _progress(root, "evaluate", "COMPLETE", len(score_rows), len(score_rows), scored_windows=len(score_rows), main_windows=len(main_rows))
    budget.save(None, stage="evaluate", scored_windows=len(score_rows))
    return summary


def _report(root: Path, summary: Mapping[str, Any], data: Mapping[str, Any], budget: Budget) -> dict[str, Any]:
    models = json.loads((root / "models/fold_models.json").read_text(encoding="utf-8"))
    historical_budget_note = json.loads((root / "state/historical_budget_note.json").read_text(encoding="utf-8")) if (root / "state/historical_budget_note.json").is_file() else None
    per_source_rows = list(csv.DictReader((root / "evaluation/per_source_metrics.csv").open(newline="", encoding="utf-8"))) if (root / "evaluation/per_source_metrics.csv").is_file() else []
    per_seed_rows = list(csv.DictReader((root / "evaluation/per_seed_metrics.csv").open(newline="", encoding="utf-8"))) if (root / "evaluation/per_seed_metrics.csv").is_file() else []
    lines = ["# V7 R 固定点对距离轨迹匹配对照 pilot", "", "本实验只读取已完成 periodic re-query R 前端与五时刻结构缓存；不是正式 src 检测链、sealed-test、空间定位或跨查询物理对应实验。", "", "## 直接回答", "", "- `PAIR_SEQ` 保留了 `SUMMARY_SET` 的逐时刻分布之外的固定 pair 轨迹信息；关系重建逐 unit 与既有 `SET_A` 的最大绝对误差见 `relations/support_summary.json`。", "- `PAIR_ID_SHUFFLE` 保留每个时刻的距离多重集合和旧 `S(t)`，但破坏后四个时刻的 pair 行对应。", "- `PAIR_TIME_SHUFFLE` 破坏时间顺序；所有窗口使用同一窗口级固定非恒等排列，间隔输入保持真实值。", "- 主要比较是 14 个双角色 source、82 个 R-valid 窗口上的 `PAIR_SEQ−PAIR_ID_SHUFFLE`；不能把它解释为真实物理对应恢复。", "", "## 数据与支撑", "", f"- R 有效窗口：{len(data['windows'])}；有效 local unit：{len(data['units'])}；canonical pair 关系行：{int(data['distances'].shape[0])}。", f"- 主评价集合：{summary['main_evaluation']['source_count']} source、{summary['main_evaluation']['window_count']} window、real/fake={summary['main_evaluation']['real_count']}/{summary['main_evaluation']['fake_count']}。", f"- 原 O/R 共同 14-source/76-window 集合仅作为次要复现，不替换本轮主集合。", "- 每个 relation identity 包含 video/source、parent、R query cohort、local unit；没有跨 query cohort 拼接。", "", "## 主条件结果", "", "| condition | source-macro AUROC | pooled AUROC | AP | P/R/F1/ACC | source/window |", "|---|---:|---:|---:|---|---:|"]
    for condition in ("SUMMARY_SET",) + CONDITIONS:
        item = summary["conditions"][condition]; lines.append(f"| {condition} | {item['source_auroc']} | {item['pooled_auroc']} | {item['pooled_ap']} | {item['precision']}/{item['recall']}/{item['f1']}/{item['accuracy']} | {item['source_count']}/{item['window_count']} |")
    lines += ["", "## 配对比较", "", "| comparison | mean AUROC difference | source bootstrap 95% CI |", "|---|---:|---|"]
    for name, value in summary["paired_comparisons"].items():
        lines.append(f"| {name} | {value.get('mean')} | {value.get('ci95')} |")
    lines += ["", "## 次要共同 14-source/76-window 复现", "", "| condition | source-macro AUROC | pooled AUROC | AP | P/R/F1/ACC |", "|---|---:|---:|---:|---|"]
    for condition in ("SUMMARY_SET",) + CONDITIONS:
        item = summary["secondary_common_14_source_76_window"][condition]
        lines.append(f"| {condition} | {item.get('source_auroc')} | {item.get('pooled_auroc')} | {item.get('pooled_ap')} | {item.get('precision')}/{item.get('recall')}/{item.get('f1')}/{item.get('accuracy')} |")
    lines += ["", "## 逐 source 差异", "", "| source | n(real/fake) | PAIR_SEQ | ID shuffle | TIME shuffle | SEQ−ID | SEQ−TIME | SEQ−SUMMARY |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for row in per_source_rows:
        lines.append(f"| {row.get('source_id')} | {row.get('real_count')}/{row.get('fake_count')} | {row.get('PAIR_SEQ_auroc')} | {row.get('PAIR_ID_SHUFFLE_auroc')} | {row.get('PAIR_TIME_SHUFFLE_auroc')} | {row.get('PAIR_SEQ_MINUS_PAIR_ID_SHUFFLE')} | {row.get('PAIR_SEQ_MINUS_PAIR_TIME_SHUFFLE')} | {row.get('PAIR_SEQ_MINUS_SUMMARY_SET')} |")
    lines += ["", "## 逐 seed 稳定性", "", "`per_seed_metrics.csv` 保存每个新条件、seed 的 source-macro、pooled、AP 和固定零 logit 指标；主结果先对三个 seed 的 OOF logit 求平均，再评价。"]
    for row in per_seed_rows:
        lines.append(f"- {row.get('condition')} seed={row.get('seed')}: source-AUROC={row.get('source_auroc')}, pooled-AUROC={row.get('pooled_auroc')}, AP={row.get('pooled_ap')}, P/R/F1/ACC={row.get('precision')}/{row.get('recall')}/{row.get('f1')}/{row.get('accuracy')}")
    budget_line = f"- 训练/评价预算累计：{budget.elapsed():.3f}s / {budget.budget_s:.3f}s。"
    if historical_budget_note is not None:
        budget_line += f" 历史说明：{historical_budget_note.get('note', '')}"
    lines += ["", "## 训练与隔离", "", f"- 新模型：{len(models.get('records', []))}（15 held-out source × 3 condition × 3 seed），全部 200 epochs={summary.get('all_formal_models_200_epochs')}。", "- `models/training_window_manifest.csv` 显式列出每个 fold-condition-seed 的训练 window；held-out source 全部排除。", "- 距离标准化只由训练 fold 拟合，三种新条件共享同一 fold 参数；四个实际秒间隔作为第 6–9 个输入通道，未用标签或 source ID。", "- loss 在窗口级 weighted BCE；relation logit 先按 unit 平均，再按 window 平均，关系多的 unit 不额外增权。", budget_line, "", "## 限制", "", "- 五时刻共同支撑是离线 survivor filter；无效关系保留为缺失原因，不填零。", "- 窗口重叠、source 数量有限，且这是开发性 pilot。", "- `SUMMARY_SET` 是已验证的 R_SET_ALL_VALID_TRAIN 实用基准，模型结构与新 relation MLP 不匹配，因此只能作实用比较。", "- 即使 `PAIR_SEQ` 超过 ID 或 TIME 对照，也不证明未知生成器泛化、空间伪造真值、完整拓扑建模或跨查询物理对应。", "", "## 产物", "", "`protocol.json`, `input_manifest.json`, `relations/`, `transforms.json`, `transform_manifest.json`, `models/fold_models.json`, `models/training_window_manifest.csv`, `scores/oof_window_scores.csv`, `evaluation/summary.json`, `evaluation/per_source_metrics.csv`, `evaluation/per_seed_metrics.csv`, `evaluation/paired_comparisons.csv`. "]
    (root / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    _progress(root, "report", "COMPLETE", 1, 1, report=str(root / "report.md"))
    budget.save(None, stage="report")
    return {"status": "COMPLETE", "report": str(root / "report.md")}


def _run(root: Path, stage: str, device: str, budget_s: float, resume: bool) -> int:
    global STOP_REQUESTED
    root.mkdir(parents=True, exist_ok=True)
    lock_path = root / ".run.lock"
    with lock_path.open("w") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit("RUN_LOCK_HELD")
        budget = Budget.load(root, budget_s)
        def stop_handler(signum: int, frame: Any) -> None:
            nonlocal budget
            global STOP_REQUESTED
            STOP_REQUESTED = True
            budget.save("STOP_REQUESTED", stage=stage)
        signal.signal(signal.SIGTERM, stop_handler); signal.signal(signal.SIGINT, stop_handler)
        _atomic_json(root / "resolved_config.json", {"git_head": _git_head(), "device": device, "budget_s": budget.budget_s, "input_root": str(INPUT_ROOT), "output_root": str(root), "stage": stage, "resume": bool(resume)})
        try:
            if stage == "prepare":
                _prepare(root, budget)
            elif stage == "smoke":
                data = _load_relation_dataset(root); result = _run_smoke(root, data, device); _progress(root, "smoke", "COMPLETE", 1, 1, smoke_result=result)
            elif stage in ("train", "evaluate", "report", "all"):
                if stage == "all" and not (root / "relations/relation_features.npz").is_file():
                    _prepare(root, budget)
                data = _load_relation_dataset(root)
                data["common_indices"] = [index for index, window in enumerate(data["windows"]) if any(other["window_id"] == window["window_id"] and other.get("mode") == "O" and other.get("support_status") == "VALID" for other in json.loads((INPUT_ROOT / "support/window_support.json").read_text(encoding="utf-8")))]
                if stage in ("train", "all"):
                    _train_formal(root, data, budget, device, resume)
                if stage in ("evaluate", "all") and (root / "models/fold_models.json").is_file():
                    summary = _evaluate(root, data, budget, device)
                else:
                    summary = json.loads((root / "evaluation/summary.json").read_text(encoding="utf-8")) if (root / "evaluation/summary.json").is_file() else None
                if stage in ("report", "all") and summary is not None:
                    _report(root, summary, data, budget)
            status = "COMPLETE" if (root / "evaluation/summary.json").is_file() and (root / "report.md").is_file() else "STAGE_COMPLETE"
            if STOP_REQUESTED:
                status = "STOP_REQUESTED"
            elif budget.remaining() <= 0:
                status = "BUDGET_EXHAUSTED"
            final_payload: dict[str, Any] = {"status": status, "updated_unix": time.time(), "cumulative_elapsed_s": budget.elapsed(), "budget_s": budget.budget_s, "stop_reason": None if status in ("COMPLETE", "STAGE_COMPLETE") else status}
            model_path = root / "models/fold_models.json"
            if model_path.is_file():
                model_rows = json.loads(model_path.read_text(encoding="utf-8")).get("records", [])
                final_payload["models"] = {
                    "planned": len({str(row["held_out_source"]) for row in model_rows}) * len(CONDITIONS) * len(SEEDS),
                    "completed": len(model_rows),
                    "epochs_complete": sum(bool(row.get("fit", {}).get("epochs") == EPOCHS) for row in model_rows),
                    "failed": 0,
                }
            score_path = root / "scores/oof_window_scores.csv"
            if score_path.is_file():
                score_rows = list(csv.DictReader(score_path.open(newline="", encoding="utf-8")))
                final_payload["scores"] = {
                    "rows": len(score_rows),
                    "condition_scored": {condition: sum(row.get(f"{condition}_status") == "SCORED" for row in score_rows) for condition in CONDITIONS},
                }
            relation_path = root / "relations/windows.json"
            if relation_path.is_file():
                final_payload["feature_input"] = {"valid_windows": len(json.loads(relation_path.read_text(encoding="utf-8"))), "valid_units": len(json.loads((root / "relations/units.json").read_text(encoding="utf-8")))}
            final_payload["report_complete"] = bool((root / "report.md").is_file() and (root / "evaluation/summary.json").is_file())
            _atomic_json(root / "final_status.json", final_payload)
            return 0
        except Exception as exc:
            _atomic_json(root / "final_status.json", {"status": "FAILED", "failed_stage": stage, "failure_reason": f"{type(exc).__name__}: {exc}", "traceback": __import__("traceback").format_exc(), "updated_unix": time.time()})
            _progress(root, stage, "FAILED", 0, 1, failure_reason=f"{type(exc).__name__}: {exc}")
            raise


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("prepare", "smoke", "train", "evaluate", "report", "all"))
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--budget-s", type=float, default=BUDGET_S)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    return _run(args.output_root, args.stage, args.device, args.budget_s, args.resume)


if __name__ == "__main__":
    raise SystemExit(main())
