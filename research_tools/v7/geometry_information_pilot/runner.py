"""Run the bounded same-support 2D/3D geometry information pilot.

The runner consumes only the completed source128 R particle/support cache.  It
does not open video files or invoke a frontend.  The four conditions share
the exact matched local units, pair identities, target frames and labels.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import subprocess
import time
import traceback
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from sparse3d_forgery.particle_sequence import load_particle_sequence
from research_tools.v7.multi_order_sequence_probe.model import FeatureStandardizer
from research_tools.v7.periodic_requery_probe import runner as periodic
from research_tools.v7.source128_extension import runner as source128

from .model import ModalSetModel, parameter_count, state_dict_numpy


REPO_ROOT = Path(__file__).resolve().parents[3]
DATA_ROOT = Path("/root/autodl-tmp/data/sparse_3d_forgery_detection")
SOURCE_ROOT = DATA_ROOT / "derived/v7_activityforensics_source128_extension_v1"
OUTPUT_ROOT = DATA_ROOT / "derived/v7_activityforensics_2d3d_information_pilot_v1"
CONDITIONS = ("UV_2D", "XYZ_3D", "UV_XYZ", "UV_UV_CONTROL")
SEEDS = (20260909, 20260910, 20260911)
EPOCHS = 200
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
BOOTSTRAP_SEED = 20260909
BOOTSTRAP_REPLICATES = 10_000
MATCH_TOLERANCE = 1e-5


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
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item, f"{path}.{key}") for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item, f"{path}[{index}]") for index, item in enumerate(value)]
    return value


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(_jsonable(value), indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
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


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_head() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()


def progress(root: Path, stage: str, status: str, completed: int, total: int, **extra: Any) -> None:
    atomic_json(root / "progress.json", {"stage": stage, "status": status, "completed": int(completed), "total": int(total), "updated_unix": time.time(), **extra})


def _safe(value: str) -> str:
    return "".join(char if char.isalnum() or char in "._-" else "_" for char in value)


def _load_selection() -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    support_rows = json.loads((SOURCE_ROOT / "support/window_support.json").read_text(encoding="utf-8"))
    r_rows = [dict(row) for row in support_rows if str(row.get("mode")) == "R"]
    by_id = {str(row["window_id"]): row for row in r_rows}
    if len(by_id) != len(r_rows):
        raise ValueError("DUPLICATE_R_WINDOW_ID")
    with (SOURCE_ROOT / "models/training_window_manifest.csv").open(newline="", encoding="utf-8") as handle:
        train_ids = [str(row["window_id"]) for row in csv.DictReader(handle)]
    with (SOURCE_ROOT / "manifests/validation_window_manifest.csv").open(newline="", encoding="utf-8") as handle:
        validation_ids = [str(row["window_id"]) for row in csv.DictReader(handle)]
    if len(set(train_ids)) != len(train_ids) or len(set(validation_ids)) != len(validation_ids):
        raise ValueError("DUPLICATE_WINDOW_ID_IN_FREEZE")
    if set(train_ids) & set(validation_ids):
        raise ValueError("TRAIN_VALIDATION_WINDOW_OVERLAP")
    if set(train_ids) - set(by_id) or set(validation_ids) - set(by_id):
        raise ValueError("FROZEN_WINDOW_MISSING_FROM_R_SUPPORT")
    train = [by_id[item] for item in train_ids]
    validation = [by_id[item] for item in validation_ids]
    train_sources = {str(row["source_id"]) for row in train}
    validation_sources = {str(row["source_id"]) for row in validation}
    if train_sources & validation_sources:
        raise ValueError("TRAIN_VALIDATION_SOURCE_OVERLAP")
    info = {
        "all_r_rows": len(r_rows),
        "all_r_valid_rows": sum(str(row.get("support_status")) == "VALID" and int(row.get("valid_unit_count", 0)) > 0 for row in r_rows),
        "original_train_windows": len(train),
        "original_validation_windows": len(validation),
        "training_sources": sorted(train_sources),
        "validation_sources": sorted(validation_sources),
        "original_train_real": sum(int(row["label"]) == 0 for row in train),
        "original_train_fake": sum(int(row["label"]) == 1 for row in train),
        "original_validation_real": sum(int(row["label"]) == 0 for row in validation),
        "original_validation_fake": sum(int(row["label"]) == 1 for row in validation),
    }
    return train, validation, r_rows, info


def _sequence_for_row(row: Mapping[str, Any]) -> Any:
    prefix = str(row.get("particle_prefix") or "")
    if not prefix:
        raise ValueError(f"MISSING_PARTICLE_PREFIX:{row['window_id']}")
    sequence = load_particle_sequence(prefix)
    expected_video = f"{row['source_id']}::{row['role']}"
    if str(sequence.source_video_id) != expected_video or int(sequence.num_tracks) != 289:
        raise ValueError(f"SEQUENCE_IDENTITY_MISMATCH:{row['window_id']}")
    lineage = sequence.lineage
    for key in ("source_id", "role", "pair_id"):
        if str(lineage.get(key, "")) != str(row.get(key, "")):
            raise ValueError(f"SEQUENCE_LINEAGE_MISMATCH:{row['window_id']}:{key}")
    parent_id = str(row["window_id"]).rsplit("::b", 1)[0]
    if str(lineage.get("parent_id", "")) != parent_id:
        raise ValueError(f"SEQUENCE_PARENT_MISMATCH:{row['window_id']}")
    offset = float(row.get("offset_s", 0.0))
    expected_cohort = "O" if abs(offset) < 1e-9 else "R"
    if str(sequence.provenance.get("query_cohort", "")) != expected_cohort:
        raise ValueError(f"SEQUENCE_COHORT_MISMATCH:{row['window_id']}")
    expected_window = parent_id if expected_cohort == "O" else str(row["window_id"])
    if str(lineage.get("window_id", "")) != expected_window:
        raise ValueError(f"SEQUENCE_WINDOW_MISMATCH:{row['window_id']}")
    uv = np.asarray(sequence.uv)
    if uv.ndim != 3 or uv.shape[-1] != 2 or uv.shape[:2] != np.asarray(sequence.xyz).shape[:2]:
        raise ValueError(f"UV_XYZ_SHAPE_MISMATCH:{row['window_id']}")
    if str(sequence.coordinate_system.length_unit.value) != "meter":
        raise ValueError(f"UNDECLARED_XYZ_UNIT:{row['window_id']}")
    return sequence


def _summary_from_distances(distances: np.ndarray) -> np.ndarray:
    values = np.asarray(distances, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 5 or values.shape[0] <= 0 or not np.all(np.isfinite(values)):
        raise ValueError("distance matrix must be finite [pair,5]")
    return np.stack((values.mean(axis=0), values.std(axis=0), np.percentile(values, 25, axis=0), np.percentile(values, 75, axis=0)), axis=1)


def _history_distances(sequence: Any, history_indices: np.ndarray, pair_indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    uv = np.asarray(sequence.uv, dtype=np.float64)
    xyz = np.asarray(sequence.xyz, dtype=np.float64)
    visible = np.asarray(sequence.visibility, dtype=bool) & np.all(np.isfinite(uv), axis=-1)
    geometry = np.asarray(sequence.geometry_validity, dtype=bool) & np.all(np.isfinite(xyz), axis=-1)
    distances2: list[float] = []
    distances3: list[float] = []
    for left, right in pair_indices:
        both = visible[history_indices, left] & visible[history_indices, right] & geometry[history_indices, left] & geometry[history_indices, right]
        if not np.any(both):
            continue
        frames = history_indices[both]
        d2 = np.linalg.norm(uv[frames, right] - uv[frames, left], axis=1)
        d3 = np.linalg.norm(xyz[frames, right] - xyz[frames, left], axis=1)
        distances2.extend(float(value) for value in d2 if math.isfinite(float(value)))
        distances3.extend(float(value) for value in d3 if math.isfinite(float(value)))
    return np.asarray(distances2, dtype=np.float64), np.asarray(distances3, dtype=np.float64)


def _match_unit(row: Mapping[str, Any], unit: Mapping[str, Any], stored_state: np.ndarray, sequence: Any, unit_index: int) -> tuple[dict[str, Any] | None, str | None]:
    array_indices = np.asarray(unit.get("array_indices", []), dtype=np.int64)
    pair_indices = np.asarray(unit.get("pair_indices", []), dtype=np.int64)
    member_slots = np.asarray(unit.get("member_slots", []), dtype=np.int64)
    if array_indices.shape != (5,) or pair_indices.ndim != 2 or pair_indices.shape[1] != 2 or member_slots.ndim != 1:
        return None, "SUPPORT_SHAPE_INVALID"
    if np.any(pair_indices[:, 0] >= pair_indices[:, 1]) or np.any(array_indices < 0) or np.any(array_indices >= len(sequence.frame_indices)):
        return None, "PAIR_OR_FRAME_INDEX_INVALID"
    if pair_indices.shape[0] != member_slots.size * (member_slots.size - 1) // 2:
        return None, "PAIR_MEMBER_COUNT_MISMATCH"
    track_ids = np.asarray(sequence.track_ids, dtype=np.int64)
    expected_pairs = [tuple(sorted(pair)) for pair in track_ids[pair_indices].tolist()]
    stored_pairs = [tuple(sorted(map(int, pair))) for pair in unit.get("pair_ids", [])]
    if expected_pairs != stored_pairs:
        return None, "PAIR_IDENTITY_MISMATCH"
    uv = np.asarray(sequence.uv, dtype=np.float64)
    xyz = np.asarray(sequence.xyz, dtype=np.float64)
    visible = np.asarray(sequence.visibility, dtype=bool) & np.all(np.isfinite(uv), axis=-1)
    geometry = np.asarray(sequence.geometry_validity, dtype=bool) & np.all(np.isfinite(xyz), axis=-1)
    target_ok = visible[array_indices][:, member_slots] & geometry[array_indices][:, member_slots]
    if not np.all(target_ok):
        return None, "TARGET_UV_XYZ_COMMON_INVALID"
    history_indices = np.asarray(row.get("grouping", {}).get("history_array_indices", []), dtype=np.int64)
    if history_indices.ndim != 1 or history_indices.size == 0:
        return None, "HISTORY_INDICES_MISSING"
    history2, history3 = _history_distances(sequence, history_indices, pair_indices)
    if history2.size == 0 or history3.size == 0:
        return None, "SHARED_HISTORY_PAIR_TIME_EMPTY"
    scale2 = float(np.median(history2))
    scale3 = float(np.median(history3))
    if not math.isfinite(scale2) or scale2 <= 0:
        return None, "UV_HISTORY_SCALE_INVALID"
    if not math.isfinite(scale3) or scale3 <= 0:
        return None, "XYZ_HISTORY_SCALE_INVALID"
    stored_scale3 = float(unit.get("history_scale", float("nan")))
    if not math.isfinite(stored_scale3) or stored_scale3 <= 0:
        return None, "STORED_XYZ_HISTORY_SCALE_INVALID"
    # Reconstruct the old SET_A from the stored 3D scale before applying the
    # common UV/XYZ history mask.  A mismatch is recorded, never hidden.
    target_xyz_distances = np.stack([np.linalg.norm(xyz[array_indices, right] - xyz[array_indices, left], axis=1) for left, right in pair_indices], axis=0)
    original_s3 = _summary_from_distances(target_xyz_distances / stored_scale3)
    stored = np.asarray(stored_state, dtype=np.float64)
    if stored.shape != (5, 4) or not np.all(np.isfinite(stored)):
        return None, "STORED_SET_A_INVALID"
    original_diff = float(np.max(np.abs(original_s3 - stored)))
    if original_diff > MATCH_TOLERANCE:
        raise ValueError(f"ORIGINAL_S3_RECONSTRUCTION_MISMATCH:{row['window_id']}:unit={unit.get('local_group_id')}:max={original_diff}")
    target_uv_distances = np.stack([np.linalg.norm(uv[array_indices, right] - uv[array_indices, left], axis=1) for left, right in pair_indices], axis=0)
    s2 = _summary_from_distances(target_uv_distances / scale2)
    s3 = _summary_from_distances(target_xyz_distances / scale3)
    timestamps = np.asarray(sequence.timestamps_s[array_indices], dtype=np.float64)
    intervals = np.diff(timestamps)
    if intervals.shape != (4,) or not np.all(np.isfinite(intervals)) or not np.all(intervals > 0):
        return None, "TARGET_PTS_INTERVAL_INVALID"
    identity = {
        "unit_id": f"{row['window_id']}::local_{int(unit['local_group_id'])}",
        "window_id": str(row["window_id"]),
        "source_id": str(row["source_id"]),
        "role": str(row["role"]),
        "parent_id": str(row["window_id"]).rsplit("::b", 1)[0],
        "query_cohort": "O" if abs(float(row.get("offset_s", 0.0))) < 1e-9 else "R",
        "local_group_id": int(unit["local_group_id"]),
        "member_slots": [int(value) for value in member_slots],
        "track_ids": [int(value) for value in track_ids[member_slots]],
        "pair_ids": [list(pair) for pair in expected_pairs],
        "array_indices": [int(value) for value in array_indices],
        "frame_indices": [int(value) for value in sequence.frame_indices[array_indices]],
        "timestamps_s": [float(value) for value in timestamps],
        "history_scale2": scale2,
        "history_scale3_shared": scale3,
        "stored_history_scale3": stored_scale3,
        "history_pair_time_count_shared": int(history2.size),
        "stored_history_pair_time_count": int(unit.get("history_scale_pair_time_count", 0)),
        "original_s3_max_abs": original_diff,
        "matched_s3_vs_original_max_abs": float(np.max(np.abs(s3 - original_s3))),
        "unit_index": int(unit_index),
    }
    return {"identity": identity, "s2": s2, "s3": s3, "intervals": intervals}, None


def _input_content_sha256(root: Path, manifest: Sequence[Mapping[str, Any]]) -> str:
    digest = hashlib.sha256()
    for item in sorted(manifest, key=lambda value: str(value["window_id"])):
        relative = str(item["input_path"])
        path = root / relative
        if not path.is_file():
            raise FileNotFoundError(f"MISSING_FEATURE_NPZ:{relative}")
        digest.update(str(item["window_id"]).encode("utf-8")); digest.update(b"\0")
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                digest.update(block)
    return digest.hexdigest()


def _save_features(root: Path, train_rows: Sequence[Mapping[str, Any]], validation_rows: Sequence[Mapping[str, Any]], all_r_rows: Sequence[Mapping[str, Any]], *, resume: bool) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    selected = list(train_rows) + list(validation_rows)
    selected_ids = {str(row["window_id"]) for row in selected}
    manifest_path = root / "inputs/feature_manifest.json"
    if resume and manifest_path.is_file():
        old = json.loads(manifest_path.read_text(encoding="utf-8"))
        old_ids = {str(item["window_id"]) for item in old}
        if old_ids == selected_ids and all((root / str(item["input_path"])).is_file() for item in old):
            by_window = {str(item["window_id"]): item for item in old}
            train_kept = [
                {**dict(row), "matched_unit_count": int(by_window[str(row["window_id"])] ["matched_unit_count"])}
                for row in train_rows
                if str(row["window_id"]) in by_window
            ]
            val_kept = [
                {**dict(row), "matched_unit_count": int(by_window[str(row["window_id"])] ["matched_unit_count"])}
                for row in validation_rows
                if str(row["window_id"]) in by_window
            ]
            return train_kept, val_kept, {"manifest": old, "reused": True}
    feature_dir = root / "features/window_inputs"
    manifest: list[dict[str, Any]] = []
    coverage: list[dict[str, Any]] = []
    matched_train: list[dict[str, Any]] = []
    matched_validation: list[dict[str, Any]] = []
    reason_counts: Counter[str] = Counter()
    lost_unit_count = 0
    original_unit_count = 0
    selected_by_id = {str(row["window_id"]): row for row in selected}
    for index, window_id in enumerate(sorted(selected_by_id)):
        row = selected_by_id[window_id]
        progress(root, "matching", "RUNNING", index, len(selected), current_window=window_id)
        if str(row.get("support_status")) != "VALID":
            coverage.append({"window_id": window_id, "split": "train" if row in train_rows else "validation", "status": "NOT_VALID_R_SUPPORT", "original_unit_count": int(row.get("valid_unit_count", 0)), "matched_unit_count": 0, "loss_reason": str(row.get("support_status"))})
            reason_counts[str(row.get("support_status"))] += 1
            continue
        sequence = _sequence_for_row(row)
        valid_units = [unit for unit in row.get("support", {}).get("units", []) if unit.get("status") == "VALID"]
        stored_states = np.asarray(row.get("features", {}).get("SET_A"), dtype=np.float64)
        if stored_states.shape != (len(valid_units), 5, 4):
            raise ValueError(f"SET_A_UNIT_SHAPE_MISMATCH:{window_id}")
        original_unit_count += len(valid_units)
        matched: list[dict[str, Any]] = []
        loss_reasons: Counter[str] = Counter()
        for unit_index, (unit, stored_state) in enumerate(zip(valid_units, stored_states)):
            result, reason = _match_unit(row, unit, stored_state, sequence, unit_index)
            if result is None:
                loss_reasons[str(reason)] += 1; reason_counts[str(reason)] += 1; lost_unit_count += 1
                continue
            matched.append(result)
        split = "train" if window_id in {str(item["window_id"]) for item in train_rows} else "validation"
        if not matched:
            coverage.append({"window_id": window_id, "split": split, "status": "NO_COMMON_2D3D_UNIT", "original_unit_count": len(valid_units), "matched_unit_count": 0, "loss_reason": json.dumps(dict(loss_reasons), sort_keys=True)})
            continue
        s2 = np.stack([item["s2"] for item in matched], axis=0).astype(np.float64)
        s3 = np.stack([item["s3"] for item in matched], axis=0).astype(np.float64)
        intervals = np.asarray(matched[0]["intervals"], dtype=np.float64)
        if any(np.max(np.abs(item["intervals"] - intervals)) > MATCH_TOLERANCE for item in matched):
            raise ValueError(f"PTS_INTERVAL_MISMATCH_WITHIN_WINDOW:{window_id}")
        relative = Path("features/window_inputs") / f"{_safe(window_id)}.npz"
        temporary = (root / relative).with_name((root / relative).name + ".tmp.npz")
        temporary.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(temporary, s2=s2, s3=s3, intervals=intervals)
        os.replace(temporary, root / relative)
        item = {"window_id": window_id, "source_id": str(row["source_id"]), "role": str(row["role"]), "label": int(row["label"]), "annotation_category": str(row.get("annotation_category", "")), "offset_s": float(row.get("offset_s", 0.0)), "input_path": str(relative), "s2_shape": list(s2.shape), "s3_shape": list(s3.shape), "matched_unit_count": len(matched), "original_unit_count": len(valid_units), "unit_identities": [unit["identity"] for unit in matched], "loss_reasons": dict(loss_reasons), "intervals_s": intervals.tolist()}
        manifest.append(item)
        coverage.append({"window_id": window_id, "split": split, "status": "MATCHED", "original_unit_count": len(valid_units), "matched_unit_count": len(matched), "loss_reason": json.dumps(dict(loss_reasons), sort_keys=True)})
        matched_row = {**dict(row), "matched_unit_count": len(matched)}
        (matched_train if split == "train" else matched_validation).append(matched_row)
        del sequence
    manifest.sort(key=lambda item: str(item["window_id"]))
    matched_train.sort(key=lambda row: str(row["window_id"])); matched_validation.sort(key=lambda row: str(row["window_id"]))
    write_csv(root / "support/matching_coverage.csv", coverage)
    atomic_json(manifest_path, manifest)
    content_hash = _input_content_sha256(root, manifest)
    atomic_json(root / "inputs/input_manifest.json", {"selected_window_count": len(manifest), "train_window_count": len(matched_train), "validation_window_count": len(matched_validation), "feature_content_sha256": content_hash, "source_support_sha256": sha256(SOURCE_ROOT / "support/window_support.json"), "training_manifest_sha256": sha256(SOURCE_ROOT / "models/training_window_manifest.csv"), "validation_manifest_sha256": sha256(SOURCE_ROOT / "manifests/validation_window_manifest.csv")})
    summary = {"original_train_windows": len(train_rows), "original_validation_windows": len(validation_rows), "matched_train_windows": len(matched_train), "matched_validation_windows": len(matched_validation), "original_train_units": sum(int(row.get("valid_unit_count", 0)) for row in train_rows), "original_validation_units": sum(int(row.get("valid_unit_count", 0)) for row in validation_rows), "matched_train_units": sum(int(item["matched_unit_count"]) for item in manifest if item["window_id"] in {str(row["window_id"]) for row in matched_train}), "matched_validation_units": sum(int(item["matched_unit_count"]) for item in manifest if item["window_id"] in {str(row["window_id"]) for row in matched_validation}), "lost_unit_count": int(lost_unit_count), "reason_counts": dict(reason_counts), "feature_content_sha256": content_hash}
    atomic_json(root / "support/matching_summary.json", summary)
    progress(root, "matching", "COMPLETE", len(manifest), len(selected), **summary)
    return matched_train, matched_validation, {"manifest": manifest, "summary": summary, "reused": False}


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {key: np.asarray(archive[key]) for key in archive.files}


def _batch(root: Path, manifest: Sequence[Mapping[str, Any]], rows: Sequence[Mapping[str, Any]], standardizer2: FeatureStandardizer, standardizer3: FeatureStandardizer, condition: str) -> dict[str, Any]:
    row_by_id = {str(row["window_id"]): row for row in rows}
    selected = sorted((item for item in manifest if str(item["window_id"]) in row_by_id), key=lambda item: str(item["window_id"]))
    s2_parts: list[np.ndarray] = []; s3_parts: list[np.ndarray] = []; interval_parts: list[np.ndarray] = []; indices: list[int] = []; labels: list[int] = []; ids: list[str] = []; node = 0
    for window_index, item in enumerate(selected):
        arrays = _load_npz(root / str(item["input_path"]))
        s2 = standardizer2.transform(np.asarray(arrays["s2"], dtype=np.float64)); s3 = standardizer3.transform(np.asarray(arrays["s3"], dtype=np.float64))
        count = int(s2.shape[0]); s2_parts.append(s2); s3_parts.append(s3); interval_parts.append(np.repeat(np.asarray(arrays["intervals"], dtype=np.float64)[None, :], count, axis=0)); indices.extend([window_index] * count); labels.append(int(row_by_id[str(item["window_id"])] ["label"])); ids.append(str(item["window_id"])); node += count
    if not ids:
        raise ValueError(f"NO_BATCH_WINDOWS:{condition}")
    s2_all = np.concatenate(s2_parts, axis=0); s3_all = np.concatenate(s3_parts, axis=0)
    if condition == "UV_2D":
        states = s2_all
    elif condition == "XYZ_3D":
        states = s3_all
    elif condition == "UV_XYZ":
        states = np.concatenate((s2_all, s3_all), axis=-1)
    elif condition == "UV_UV_CONTROL":
        states = np.concatenate((s2_all, s2_all), axis=-1)
    else:
        raise ValueError(condition)
    window_rows = [row_by_id[item] for item in ids]
    return {"states": states, "intervals": np.concatenate(interval_parts, axis=0), "window_index": np.asarray(indices, dtype=np.int64), "labels": np.asarray(labels, dtype=np.int64), "window_ids": ids, "window_rows": window_rows, "window_count": len(ids), "window_weights": periodic.source_class_weights(window_rows)}


def _initial_model(condition: str, seed: int) -> ModalSetModel:
    import torch
    torch.manual_seed(int(seed)); np.random.seed(int(seed) & 0xFFFFFFFF)
    dimension = 4 if condition in ("UV_2D", "XYZ_3D") else 8
    if dimension == 4:
        return ModalSetModel(4)
    base = ModalSetModel(4)
    fused = ModalSetModel(8)
    with torch.no_grad():
        fused.encoder[0].weight.zero_(); fused.encoder[0].weight[:, :4].copy_(base.encoder[0].weight); fused.encoder[0].bias.copy_(base.encoder[0].bias)
        for target, source in ((fused.encoder[2], base.encoder[2]), (fused.head[0], base.head[0]), (fused.head[2], base.head[2]), (fused.head[4], base.head[4])):
            target.weight.copy_(source.weight); target.bias.copy_(source.bias)
    return fused


def _forward(model: ModalSetModel, batch: Mapping[str, Any], device: str) -> Any:
    import torch
    states = torch.as_tensor(batch["states"], dtype=torch.float32, device=device); intervals = torch.as_tensor(batch["intervals"], dtype=torch.float32, device=device); indices = torch.as_tensor(batch["window_index"], dtype=torch.long, device=device)
    local = model(states, intervals); sums = torch.zeros(int(batch["window_count"]), dtype=local.dtype, device=device); counts = torch.zeros_like(sums); sums.index_add_(0, indices, local); counts.index_add_(0, indices, torch.ones_like(local)); return sums / counts.clamp_min(1.0)


def _train_one(condition: str, batch: Mapping[str, Any], seed: int, device: str, epochs: int = EPOCHS) -> tuple[ModalSetModel, dict[str, Any]]:
    import torch
    import torch.nn.functional as F
    model = _initial_model(condition, seed).to(device); model.train(); optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    labels = torch.as_tensor(batch["labels"], dtype=torch.float32, device=device); weights = torch.as_tensor(batch["window_weights"], dtype=torch.float32, device=device); history: list[dict[str, Any]] = []
    for epoch in range(1, epochs + 1):
        optimizer.zero_grad(set_to_none=True); scores = _forward(model, batch, device); loss = torch.sum(F.binary_cross_entropy_with_logits(scores, labels, reduction="none") * weights) / torch.sum(weights)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"NONFINITE_LOSS:{condition}:{seed}:{epoch}")
        loss.backward()
        if not all(parameter.grad is None or torch.all(torch.isfinite(parameter.grad)) for parameter in model.parameters()):
            raise FloatingPointError(f"NONFINITE_GRADIENT:{condition}:{seed}:{epoch}")
        optimizer.step(); history.append({"epoch": epoch, "loss": float(loss.detach().cpu())})
        if epoch in (1, epochs):
            print(f"geometry-information condition={condition} seed={seed} epoch={epoch}/{epochs} loss={history[-1]['loss']:.6f}", flush=True)
    model.eval(); return model, {"seed": int(seed), "epochs": int(epochs), "initial_loss": history[0]["loss"], "final_loss": history[-1]["loss"], "min_loss": min(item["loss"] for item in history), "loss_history": history, "device": str(torch.device(device)), "parameter_count": parameter_count(model)}


def _model_identity(root: Path, feature_manifest: Sequence[Mapping[str, Any]], train_rows: Sequence[Mapping[str, Any]], validation_rows: Sequence[Mapping[str, Any]], condition: str, seed: int, standardizer2: FeatureStandardizer, standardizer3: FeatureStandardizer, feature_hash: str) -> dict[str, Any]:
    return {"condition": str(condition), "seed": int(seed), "feature_content_sha256": feature_hash, "feature_manifest_sha256": hashlib.sha256(json.dumps(feature_manifest, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest(), "training_window_ids_sha256": hashlib.sha256("\n".join(sorted(str(row["window_id"]) for row in train_rows)).encode("utf-8")).hexdigest(), "validation_window_ids_sha256": hashlib.sha256("\n".join(sorted(str(row["window_id"]) for row in validation_rows)).encode("utf-8")).hexdigest(), "standardizer2": standardizer2.as_dict(), "standardizer3": standardizer3.as_dict(), "model_config": {"conditions": list(CONDITIONS), "epochs": EPOCHS, "learning_rate": LEARNING_RATE, "weight_decay": WEIGHT_DECAY}}


def _record_matches(record: Mapping[str, Any], expected: Mapping[str, Any]) -> bool:
    actual = record.get("input_identity")
    return isinstance(actual, Mapping) and all(actual.get(key) == value for key, value in expected.items())


def _model_load(condition: str, record: Mapping[str, Any], device: str) -> ModalSetModel:
    import torch
    model = ModalSetModel(4 if condition in ("UV_2D", "XYZ_3D") else 8)
    model.load_state_dict({name: torch.as_tensor(value, dtype=model.state_dict()[name].dtype) for name, value in record["state_dict"].items()}); model.to(device); model.eval(); return model


def _metrics(rows: Sequence[Mapping[str, Any]], scores: Sequence[float]) -> tuple[dict[str, Any], dict[str, float]]:
    labels = [int(row["label"]) for row in rows]; values = [float(item) for item in scores]
    if not all(math.isfinite(item) for item in values):
        raise ValueError("NONFINITE_SCORE")
    by_source: dict[str, list[tuple[int, float]]] = defaultdict(list)
    for row, score in zip(rows, values): by_source[str(row["source_id"])].append((int(row["label"]), score))
    source_values = {source: periodic._auroc([x[0] for x in pairs], [x[1] for x in pairs]) for source, pairs in by_source.items()}
    source_values = {source: float(value) for source, value in source_values.items() if value is not None}
    return {"window_count": len(rows), "real_count": labels.count(0), "fake_count": labels.count(1), "source_count": len(by_source), "dual_role_source_count": len(source_values), "source_macro": source128._bootstrap(source_values), "pooled_auroc": periodic._auroc(labels, values), "pooled_ap": periodic._ap(labels, values), **periodic._classification(labels, values)}, source_values


def _metric_row(split: str, condition: str, seed: Any, metrics: Mapping[str, Any]) -> dict[str, Any]:
    ci = metrics.get("source_macro", {}).get("ci95") or [None, None]
    return {"split": split, "condition": condition, "seed": seed, "window_count": metrics["window_count"], "real_count": metrics["real_count"], "fake_count": metrics["fake_count"], "source_count": metrics["source_count"], "dual_role_source_count": metrics["dual_role_source_count"], "source_macro_auroc": metrics["source_macro"].get("mean"), "source_macro_ci_low": ci[0], "source_macro_ci_high": ci[1], "pooled_auroc": metrics.get("pooled_auroc"), "pooled_ap": metrics.get("pooled_ap"), "precision": metrics.get("precision"), "recall": metrics.get("recall"), "f1": metrics.get("f1"), "accuracy": metrics.get("accuracy"), "tn": metrics.get("tn"), "fp": metrics.get("fp"), "fn": metrics.get("fn"), "tp": metrics.get("tp")}


def _score_rows(rows: Sequence[Mapping[str, Any]], store: Mapping[tuple[str, str, Any], Mapping[str, float]], split: str) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for row in rows:
        item: dict[str, Any] = {"window_id": str(row["window_id"]), "source_id": str(row["source_id"]), "role": str(row["role"]), "label": int(row["label"]), "annotation_category": str(row.get("annotation_category", "")), "matched_unit_count": int(row.get("matched_unit_count", 0))}
        for condition in CONDITIONS:
            for seed in SEEDS: item[f"{condition}_seed_{seed}"] = store[(split, condition, seed)][str(row["window_id"])]
            item[f"{condition}_MEAN_LOGIT"] = float(np.mean([item[f"{condition}_seed_{seed}"] for seed in SEEDS]))
        output.append(item)
    return output


def evaluate(root: Path, feature_manifest: Sequence[Mapping[str, Any]], train_rows: Sequence[Mapping[str, Any]], validation_rows: Sequence[Mapping[str, Any]], records: Sequence[Mapping[str, Any]], standardizer2: FeatureStandardizer, standardizer3: FeatureStandardizer, device: str) -> dict[str, Any]:
    store: dict[tuple[str, str, Any], dict[str, float]] = {}; metric_rows: list[dict[str, Any]] = []; per_seed_rows: list[dict[str, Any]] = []
    for condition in CONDITIONS:
        for record in [item for item in records if str(item["condition"]) == condition]:
            seed = int(record["seed"]); model = _model_load(condition, record, device)
            for split, rows in (("train", train_rows), ("validation", validation_rows)):
                batch = _batch(root, feature_manifest, rows, standardizer2, standardizer3, condition); scores = _forward(model, batch, device).detach().cpu().numpy().astype(np.float64); store[(split, condition, seed)] = {window_id: float(value) for window_id, value in zip(batch["window_ids"], scores)}; metrics, _ = _metrics(batch["window_rows"], scores); row = _metric_row(split, condition, seed, metrics); metric_rows.append(row); per_seed_rows.append(row)
            del model
    mean_store: dict[tuple[str, str, int], dict[str, float]] = {}
    for split, rows in (("train", train_rows), ("validation", validation_rows)):
        for condition in CONDITIONS:
            mean_scores = {str(row["window_id"]): float(np.mean([store[(split, condition, seed)][str(row["window_id"])] for seed in SEEDS])) for row in rows}; mean_store[(split, condition, 0)] = mean_scores; metrics, _ = _metrics(rows, [mean_scores[str(row["window_id"])] for row in rows]); metric_rows.append(_metric_row(split, condition, "MEAN_LOGIT", metrics))
    per_source_rows: list[dict[str, Any]] = []
    for split, rows in (("train", train_rows), ("validation", validation_rows)):
        for source in sorted({str(row["source_id"]) for row in rows}):
            source_rows = [row for row in rows if str(row["source_id"]) == source]
            for condition in CONDITIONS:
                for seed in (*SEEDS, "MEAN_LOGIT"):
                    mapping = mean_store[(split, condition, 0)] if seed == "MEAN_LOGIT" else store[(split, condition, int(seed))]; values = [mapping[str(row["window_id"])] for row in source_rows]; per_source_rows.append({"split": split, "source_id": source, "condition": condition, "seed": seed, "window_count": len(values), "real_count": sum(int(row["label"]) == 0 for row in source_rows), "fake_count": sum(int(row["label"]) == 1 for row in source_rows), "auroc": periodic._auroc([int(row["label"]) for row in source_rows], values)})
    validation_values: dict[str, dict[str, float]] = {}
    for condition in CONDITIONS:
        mapping = mean_store[("validation", condition, 0)]; source_values: dict[str, float] = {}
        for source in sorted({str(row["source_id"]) for row in validation_rows}):
            subset = [row for row in validation_rows if str(row["source_id"]) == source]; value = periodic._auroc([int(row["label"]) for row in subset], [mapping[str(row["window_id"])] for row in subset]);
            if value is not None: source_values[source] = float(value)
        validation_values[condition] = source_values
    comparisons: list[dict[str, Any]] = []
    for left, right in (("UV_XYZ", "UV_UV_CONTROL"), ("XYZ_3D", "UV_2D"), ("UV_XYZ", "UV_2D"), ("UV_UV_CONTROL", "UV_2D")):
        result = source128._bootstrap(validation_values[left], validation_values[right]); result.update({"name": f"{left}-{right}", "left": left, "right": right}); result["positive_count"] = sum(value > 0 for value in (validation_values[left][source] - validation_values[right][source] for source in result["sources"])); result["negative_count"] = sum(value < 0 for value in (validation_values[left][source] - validation_values[right][source] for source in result["sources"])); result["tie_count"] = result["source_count"] - result["positive_count"] - result["negative_count"]; comparisons.append(result)
    write_csv(root / "scores/train_window_scores.csv", _score_rows(train_rows, store, "train")); write_csv(root / "scores/validation_window_scores.csv", _score_rows(validation_rows, store, "validation")); write_csv(root / "evaluation/metrics.csv", metric_rows); write_csv(root / "evaluation/per_seed_metrics.csv", per_seed_rows); write_csv(root / "evaluation/per_source_metrics.csv", per_source_rows); write_csv(root / "evaluation/paired_comparisons.csv", [{"comparison": item["name"], "source_count": item["source_count"], "mean": item["mean"], "ci_low": item["ci95"][0], "ci_high": item["ci95"][1], "positive_count": item["positive_count"], "negative_count": item["negative_count"], "tie_count": item["tie_count"]} for item in comparisons])
    summary = {"train_population": {"windows": len(train_rows), "real": sum(int(row["label"]) == 0 for row in train_rows), "fake": sum(int(row["label"]) == 1 for row in train_rows), "sources": len({str(row["source_id"]) for row in train_rows})}, "validation_population": {"windows": len(validation_rows), "real": sum(int(row["label"]) == 0 for row in validation_rows), "fake": sum(int(row["label"]) == 1 for row in validation_rows), "sources": len({str(row["source_id"]) for row in validation_rows})}, "metrics": metric_rows, "comparisons": comparisons, "device": str(device), "model_count": len(records)}
    atomic_json(root / "evaluation/summary.json", summary); return summary


def _smoke(root: Path, feature_manifest: Sequence[Mapping[str, Any]], train_rows: Sequence[Mapping[str, Any]], standardizer2: FeatureStandardizer, standardizer3: FeatureStandardizer, device: str) -> dict[str, Any]:
    import torch
    rows = list(train_rows[:8]); conditions: list[dict[str, Any]] = []
    if {int(row["label"]) for row in rows} != {0, 1}:
        raise ValueError("SMOKE_NEEDS_BOTH_CLASSES")
    base_batch = _batch(root, feature_manifest, rows, standardizer2, standardizer3, "UV_2D")
    base = _initial_model("UV_2D", SEEDS[0]).to(device).eval(); control = _initial_model("UV_UV_CONTROL", SEEDS[0]).to(device).eval(); control_batch = _batch(root, feature_manifest, rows, standardizer2, standardizer3, "UV_UV_CONTROL")
    with torch.no_grad():
        degeneration = float(torch.max(torch.abs(_forward(base, base_batch, device) - _forward(control, control_batch, device))).detach().cpu())
    if degeneration > 1e-5:
        raise ValueError(f"FUSED_INITIALIZATION_NOT_DEGENERATE:{degeneration}")
    for condition in CONDITIONS:
        batch = _batch(root, feature_manifest, rows, standardizer2, standardizer3, condition); model, fit = _train_one(condition, batch, SEEDS[0], device, epochs=1); before = _forward(model, batch, device).detach().cpu().numpy(); reloaded = _model_load(condition, {"state_dict": state_dict_numpy(model)}, device); after = _forward(reloaded, batch, device).detach().cpu().numpy(); conditions.append({"condition": condition, "parameter_count": parameter_count(model), "epochs": fit["epochs"], "scores_finite": bool(np.all(np.isfinite(before))), "reload_max_abs": float(np.max(np.abs(before - after))), "passed": bool(np.all(np.isfinite(before)) and np.max(np.abs(before - after)) <= 1e-5)}); del model, reloaded
    result = {"status": "PASS" if all(item["passed"] for item in conditions) and degeneration <= 1e-5 else "FAILED", "rows": len(rows), "initial_fused_control_max_abs": degeneration, "conditions": conditions, "device": device}; atomic_json(root / "smoke/summary.json", result); return result


def _write_protocol(root: Path, info: Mapping[str, Any], matching: Mapping[str, Any], feature_manifest: Sequence[Mapping[str, Any]]) -> None:
    import torch
    atomic_json(root / "protocol.json", {"protocol_id": "v7-2d3d-information-pilot-v1", "git_head": git_head(), "source_root": str(SOURCE_ROOT), "input": {"original_train_windows": info["original_train_windows"], "original_validation_windows": info["original_validation_windows"], "matched_train_windows": matching["matched_train_windows"], "matched_validation_windows": matching["matched_validation_windows"], "training_sources": info["training_sources"], "validation_sources": info["validation_sources"], "source_support_sha256": sha256(SOURCE_ROOT / "support/window_support.json"), "training_manifest_sha256": sha256(SOURCE_ROOT / "models/training_window_manifest.csv"), "validation_manifest_sha256": sha256(SOURCE_ROOT / "manifests/validation_window_manifest.csv"), "feature_content_sha256": matching.get("feature_content_sha256")}, "coordinate_contract": {"uv": "ParticleSequence source-pixel coordinates, x/y in frame width/height convention", "xyz": "ParticleSequence camera-motion-compensated meter coordinates", "distance": "Euclidean pair distance", "normalization": "one shared history pair-time median per modality and unit; no per-frame scale", "support": "same target members/pairs/frames require UV visibility and XYZ geometry validity"}, "conditions": {"UV_2D": "S2 only", "XYZ_3D": "S3 only", "UV_XYZ": "standardized S2 concatenated with standardized S3", "UV_UV_CONTROL": "standardized S2 concatenated with itself"}, "model": {"architecture": "Linear(D,16)-ReLU-Linear(16,8), five-time mean, four PTS intervals, 12-16-8-1 head", "dimensions": {"UV_2D": 4, "XYZ_3D": 4, "UV_XYZ": 8, "UV_UV_CONTROL": 8}, "parameter_counts": {"four_dimensional": parameter_count(ModalSetModel(4)), "eight_dimensional": parameter_count(ModalSetModel(8))}, "epochs": EPOCHS, "optimizer": "Adam", "learning_rate": LEARNING_RATE, "weight_decay": WEIGHT_DECAY, "seeds": list(SEEDS), "threshold": "logit >= 0", "loss": "source/class weighted BCE at window level", "eight_dim_initialization": "first four input columns copied from four-dimensional initialization; extra columns zero then trainable"}, "evaluation": {"primary": "UV_XYZ minus UV_UV_CONTROL source-macro AUROC", "bootstrap_seed": BOOTSTRAP_SEED, "bootstrap_replicates": BOOTSTRAP_REPLICATES, "validation": "matched source128 validation rows", "survivor_bias": True, "not_sealed_test": True}, "runtime": {"torch_version": str(torch.__version__), "cuda_available": bool(torch.cuda.is_available()), "cuda_version": str(torch.version.cuda), "deterministic_algorithms": bool(torch.are_deterministic_algorithms_enabled()), "cudnn_deterministic": bool(torch.backends.cudnn.deterministic), "cudnn_benchmark": bool(torch.backends.cudnn.benchmark)}, "manifest_count": len(feature_manifest)})


def _write_report(root: Path, info: Mapping[str, Any], matching: Mapping[str, Any], summary: Mapping[str, Any], records: Sequence[Mapping[str, Any]], timings: Mapping[str, float], smoke: Mapping[str, Any]) -> Path:
    metrics = {(str(item["split"]), str(item["condition"])): item for item in summary.get("metrics", []) if str(item["seed"]) == "MEAN_LOGIT"}
    lines = ["# V7 同支撑二维／三维信息增量 pilot", "", "本实验在相同 R query cohort、局部组、成员、pair、五个目标时刻和共同有效性上比较二维 UV 摘要与三维 XYZ 摘要；二维分支复用了三维流程产生的分组和支撑，不是独立纯二维前端。", "", "## 结果先行", f"- 正式模型：{len(records)}/12；smoke：{smoke.get('status')}；设备：{summary.get('device')}。", f"- 原冻结集合：训练 {info['original_train_windows']} 窗口、验证 {info['original_validation_windows']} 窗口；匹配后训练 {matching['matched_train_windows']}、验证 {matching['matched_validation_windows']}。训练 source={len(info['training_sources'])}，验证 source={len(info['validation_sources'])}，source 不交叉。", f"- 匹配后 unit 损失：{matching.get('lost_unit_count', 0)}；窗口损失见 `support/matching_coverage.csv`，没有用零或其他窗口替代。", "", "| 条件 | source-macro AUROC (95% CI) | pooled AUROC | AP | Precision | Recall | F1 | ACC |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for condition in CONDITIONS:
        item = metrics.get(("validation", condition), {}); lines.append(f"| {condition} | {item.get('source_macro_auroc')} [{item.get('source_macro_ci_low')}, {item.get('source_macro_ci_high')}] | {item.get('pooled_auroc')} | {item.get('pooled_ap')} | {item.get('precision')} | {item.get('recall')} | {item.get('f1')} | {item.get('accuracy')} |")
    lines += ["", "训练集三 seed 平均 logit（拟合诊断，不是泛化证据）：", "", "| 条件 | source-macro AUROC | pooled AUROC | AP | F1 | ACC |", "|---|---:|---:|---:|---:|---:|"]
    for condition in CONDITIONS:
        item = metrics.get(("train", condition), {}); lines.append(f"| {condition} | {item.get('source_macro_auroc')} | {item.get('pooled_auroc')} | {item.get('pooled_ap')} | {item.get('f1')} | {item.get('accuracy')} |")
    lines += ["", "## 预声明配对", "", "| 比较 | source 数 | 均值差 | 95% CI | 正/负/平 source |", "|---|---:|---:|---|---:|"]
    for item in summary.get("comparisons", []): lines.append(f"| {item['name']} | {item['source_count']} | {item['mean']} | [{item['ci95'][0]}, {item['ci95'][1]}] | {item['positive_count']}/{item['negative_count']}/{item['tie_count']} |")
    lines += ["", "## 解释边界", "", "- `UV_XYZ − UV_UV_CONTROL` 是本轮三维相对二维重复输入的主要比较；它没有稳定正增益时，不能声称三维摘要提供额外跨 source 信息。`XYZ_3D − UV_2D` 只回答单模态差异，不能否定其他三维表示。", "- 共同有效性要求 UV visibility、有限 UV、XYZ geometry validity 和有限 XYZ 同时成立；因此结果带有 survivor bias，也不能回答去掉三维前端后纯二维系统的表现。", "- 归一化使用每个 unit 的历史 pair-time 中位数；不使用目标时刻重定尺度、逐帧尺度、标签或模型分数。S3 重建与原 SET_A 的误差、unit/pair/PTS 身份均写入特征 manifest。", "- 验证集已被既有开发流程使用，不是 sealed test；本轮不检验因果时间顺序、不提供空间定位真值。", "", "## 产物与成本", f"- 阶段耗时（秒）：matching={timings.get('matching')}, smoke={timings.get('smoke')}, train={timings.get('train')}, evaluate={timings.get('evaluate')}, report={timings.get('report')}。", "- 逐 seed/逐 source：`evaluation/per_seed_metrics.csv`、`evaluation/per_source_metrics.csv`；窗口分数：`scores/train_window_scores.csv`、`scores/validation_window_scores.csv`；配对 bootstrap：`evaluation/paired_comparisons.csv`。", "- 主要产物：`protocol.json`、`inputs/input_manifest.json`、`inputs/feature_manifest.json`、`support/matching_summary.json`、`models/fold_models.json`、`evaluation/summary.json`、`final_status.json`。", "- 未访问旧 R7/V5；未修改正式 src；未下载数据；未重跑 tracking、depth、pose、segmentation 或前端。", ""]
    path = root / "report.md"; path.write_text("\n".join(lines), encoding="utf-8"); return path


def run(root: Path = OUTPUT_ROOT, *, device: str = "cuda", resume: bool = True) -> dict[str, Any]:
    root.mkdir(parents=True, exist_ok=True); (root / "state").mkdir(parents=True, exist_ok=True); started = time.perf_counter(); timings: dict[str, float] = {}
    atomic_json(root / "state/launch.json", {"git_head": git_head(), "pid": os.getpid(), "device": device, "started_unix": time.time()})
    try:
        train_original, validation_original, all_r_rows, info = _load_selection()
        matching_started = time.perf_counter(); train_rows, validation_rows, built = _save_features(root, train_original, validation_original, all_r_rows, resume=resume); timings["matching"] = time.perf_counter() - matching_started; feature_manifest = built["manifest"]; matching = built.get("summary") or json.loads((root / "support/matching_summary.json").read_text(encoding="utf-8"));
        if not train_rows or not validation_rows:
            raise RuntimeError("COMMON_SUPPORT_REMOVED_ALL_TRAIN_OR_VALIDATION")
        standardizer2 = periodic.fit_standardizer("SET_A", [np.asarray(_load_npz(root / str(item["input_path"]))["s2"], dtype=np.float64) for item in feature_manifest if item["window_id"] in {str(row["window_id"]) for row in train_rows}], periodic.source_class_weights(train_rows)); standardizer3 = periodic.fit_standardizer("SET_A", [np.asarray(_load_npz(root / str(item["input_path"]))["s3"], dtype=np.float64) for item in feature_manifest if item["window_id"] in {str(row["window_id"]) for row in train_rows}], periodic.source_class_weights(train_rows));
        _write_protocol(root, info, matching, feature_manifest)
        smoke_started = time.perf_counter(); smoke = json.loads((root / "smoke/summary.json").read_text(encoding="utf-8")) if resume and (root / "smoke/summary.json").is_file() else _smoke(root, feature_manifest, train_rows, standardizer2, standardizer3, device); timings["smoke"] = time.perf_counter() - smoke_started
        if smoke.get("status") != "PASS": raise RuntimeError("SMOKE_FAILED")
        model_path = root / "models/fold_models.json"; existing = json.loads(model_path.read_text(encoding="utf-8")).get("records", []) if resume and model_path.is_file() else []; feature_hash = str(matching["feature_content_sha256"]); records: list[dict[str, Any]] = []; complete: set[tuple[str, int]] = set(); unverified: list[dict[str, Any]] = []
        for old in existing:
            condition, seed = str(old.get("condition", "")), int(old.get("seed", -1)); expected = _model_identity(root, feature_manifest, train_rows, validation_rows, condition, seed, standardizer2, standardizer3, feature_hash) if condition in CONDITIONS and seed in SEEDS else None
            if old.get("status") == "TRAIN_COMPLETE" and expected and _record_matches(old, expected): records.append(old); complete.add((condition, seed))
            elif old: unverified.append({"condition": condition, "seed": seed, "status": old.get("status")})
        if unverified: atomic_json(root / "state/unverified_model_records.json", {"reason": "missing_or_mismatched_input_identity", "records": unverified})
        train_started = time.perf_counter(); total = len(CONDITIONS) * len(SEEDS)
        for condition in CONDITIONS:
            batch = _batch(root, feature_manifest, train_rows, standardizer2, standardizer3, condition)
            for seed in SEEDS:
                if (condition, int(seed)) in complete: continue
                model, fit = _train_one(condition, batch, seed, device); record = {"condition": condition, "seed": int(seed), "status": "TRAIN_COMPLETE", "parameter_count": parameter_count(model), "fit": fit, "standardization": {"s2": standardizer2.as_dict(), "s3": standardizer3.as_dict()}, "input_identity": _model_identity(root, feature_manifest, train_rows, validation_rows, condition, seed, standardizer2, standardizer3, feature_hash), "state_dict": state_dict_numpy(model)}; records.append(record); complete.add((condition, int(seed))); atomic_json(model_path, {"conditions": list(CONDITIONS), "seeds": list(SEEDS), "epochs": EPOCHS, "records": records}); progress(root, "train", "RUNNING", len(complete), total, condition=condition, seed=seed); del model
        timings["train"] = time.perf_counter() - train_started
        if len(complete) != total: raise RuntimeError(f"MODEL_COUNT_INCOMPLETE:{len(complete)}/{total}")
        evaluate_started = time.perf_counter(); summary = evaluate(root, feature_manifest, train_rows, validation_rows, records, standardizer2, standardizer3, device); timings["evaluate"] = time.perf_counter() - evaluate_started
        report_started = time.perf_counter(); timings["report"] = 0.0; report_path = _write_report(root, info, matching, summary, records, timings, smoke); timings["report"] = time.perf_counter() - report_started; _write_report(root, info, matching, summary, records, timings, smoke)
        atomic_json(root / "final_status.json", {"status": "COMPLETE", "model_count": len(records), "expected_model_count": total, "original_train_windows": len(train_original), "original_validation_windows": len(validation_original), "matched_train_windows": len(train_rows), "matched_validation_windows": len(validation_rows), "report": str(report_path), "timings_s": timings, "git_head": git_head()}); progress(root, "report", "COMPLETE", 1, 1, model_count=len(records), elapsed_s=time.perf_counter() - started); return {"status": "COMPLETE", "report": str(report_path), "summary": summary, "timings_s": timings}
    except BaseException as exc:
        atomic_json(root / "state/failure.json", {"status": "FAILED", "error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()}); atomic_json(root / "final_status.json", {"status": "FAILED", "error": f"{type(exc).__name__}: {exc}", "git_head": git_head(), "elapsed_s": time.perf_counter() - started}); progress(root, "failed", "FAILED", 0, 1, error=f"{type(exc).__name__}: {exc}"); raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT); parser.add_argument("--device", default="cuda"); parser.add_argument("--resume", action="store_true"); args = parser.parse_args(); result = run(args.output_root, device=args.device, resume=args.resume); print(json.dumps({"status": result["status"], "report": result["report"], "timings_s": result["timings_s"]}, ensure_ascii=False, indent=2)); return 0


if __name__ == "__main__":
    raise SystemExit(main())
