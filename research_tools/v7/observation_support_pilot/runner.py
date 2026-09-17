"""Run the bounded structure/support information pilot from cached R data."""

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

from research_tools.v7.geometry_information_pilot import runner as geometry
from research_tools.v7.geometry_information_pilot.model import ModalSetModel, parameter_count, state_dict_numpy
from research_tools.v7.multi_order_sequence_probe.model import FeatureStandardizer, fit_standardizer, source_class_weights
from research_tools.v7.periodic_requery_probe import runner as periodic


REPO_ROOT = Path(__file__).resolve().parents[3]
DATA_ROOT = Path("/root/autodl-tmp/data/sparse_3d_forgery_detection")
SOURCE_ROOT = DATA_ROOT / "derived/v7_activityforensics_source128_extension_v1"
GEOMETRY_ROOT = DATA_ROOT / "derived/v7_activityforensics_2d3d_information_pilot_v1"
OUTPUT_ROOT = DATA_ROOT / "derived/v7_activityforensics_observation_support_pilot_v1"
CONDITIONS = ("STRUCTURE_ONLY", "SUPPORT_ONLY", "STRUCTURE_SUPPORT", "STRUCTURE_DUP_CONTROL")
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


def _load_reference() -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    train, validation, all_r, info = geometry._load_selection()
    geometry_manifest = json.loads((GEOMETRY_ROOT / "inputs/feature_manifest.json").read_text(encoding="utf-8"))
    selected = {str(item["window_id"]): item for item in geometry_manifest}
    expected = {str(row["window_id"]) for row in train + validation}
    if set(selected) != expected:
        raise ValueError(f"REFERENCE_WINDOW_SET_MISMATCH:{len(selected)}:{len(expected)}")
    support_by_id = {str(row["window_id"]): row for row in all_r}
    if not expected.issubset(support_by_id):
        raise ValueError("REFERENCE_SUPPORT_ROW_MISSING")
    if not all((GEOMETRY_ROOT / str(item["input_path"])).is_file() for item in geometry_manifest):
        raise FileNotFoundError("REFERENCE_GEOMETRY_FEATURE_MISSING")
    if len(train) != 718 or len(validation) != 83:
        print(f"observation-support reference counts train={len(train)} validation={len(validation)}", flush=True)
    return train, validation, selected, support_by_id, {**info, "geometry_manifest": geometry_manifest}


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {key: np.asarray(archive[key]) for key in archive.files}


def _q_for_unit(row: Mapping[str, Any], identity: Mapping[str, Any], support_unit: Mapping[str, Any], sequence: Any, group: Mapping[str, Any]) -> tuple[np.ndarray, dict[str, Any], int]:
    raw_members = np.asarray(group.get("member_slots", []), dtype=np.int64)
    common_members = np.asarray(support_unit.get("member_slots", []), dtype=np.int64)
    array_indices = np.asarray(identity.get("array_indices", []), dtype=np.int64)
    history_indices = np.asarray(row.get("grouping", {}).get("history_array_indices", []), dtype=np.int64)
    if raw_members.ndim != 1 or raw_members.size == 0 or common_members.ndim != 1 or common_members.size == 0 or array_indices.shape != (5,) or history_indices.ndim != 1 or history_indices.size == 0:
        raise ValueError(f"Q_SUPPORT_IDENTITY_INVALID:{row['window_id']}:{identity.get('local_group_id')}")
    timestamps = np.asarray(sequence.timestamps_s, dtype=np.float64)
    target_timestamps = timestamps[array_indices]
    prior = history_indices[timestamps[history_indices] < target_timestamps[0]]
    if prior.size == 0:
        raise ValueError(f"Q_HISTORY_REFERENCE_MISSING:{row['window_id']}:{identity.get('local_group_id')}")
    reference_index = int(prior[np.argmax(timestamps[prior])])
    visibility = np.asarray(sequence.visibility, dtype=bool)
    geometry_validity = np.asarray(sequence.geometry_validity, dtype=bool)
    uv_finite = np.all(np.isfinite(np.asarray(sequence.uv)), axis=-1)
    xyz_finite = np.all(np.isfinite(np.asarray(sequence.xyz)), axis=-1)
    if np.any(visibility & ~uv_finite) or np.any(geometry_validity & ~xyz_finite):
        raise ValueError(f"MASK_FINITE_CONTRACT_CONFLICT:{row['window_id']}")
    visibility_fraction = np.mean(visibility[array_indices][:, raw_members], axis=1)
    geometry_fraction = np.mean(geometry_validity[array_indices][:, raw_members], axis=1)
    history_visibility = float(np.mean(visibility[reference_index, raw_members]))
    history_geometry = float(np.mean(geometry_validity[reference_index, raw_members]))
    q = np.stack((visibility_fraction, geometry_fraction, visibility_fraction - history_visibility, geometry_fraction - history_geometry), axis=1).astype(np.float64)
    if not np.all(np.isfinite(q)):
        raise ValueError(f"Q_NONFINITE:{row['window_id']}:{identity.get('local_group_id')}")
    metadata = {
        "local_group_id": int(identity["local_group_id"]),
        "raw_member_slots": [int(value) for value in raw_members],
        "raw_track_ids": [int(value) for value in np.asarray(sequence.track_ids, dtype=np.int64)[raw_members]],
        "common_member_slots": [int(value) for value in common_members],
        "common_track_ids": [int(value) for value in identity.get("track_ids", [])],
        "target_array_indices": [int(value) for value in array_indices],
        "target_frame_indices": [int(value) for value in np.asarray(sequence.frame_indices)[array_indices]],
        "target_timestamps_s": [float(value) for value in target_timestamps],
        "history_reference_array_index": reference_index,
        "history_reference_frame_index": int(np.asarray(sequence.frame_indices)[reference_index]),
        "history_reference_timestamp_s": float(timestamps[reference_index]),
        "query_initialization_timestamp_s": float(sequence.provenance["cohort_start_s"]),
        "query_age_s": [float(value - sequence.provenance["cohort_start_s"]) for value in target_timestamps],
        "history_reference_age_s": float(timestamps[reference_index] - sequence.provenance["cohort_start_s"]),
        "raw_member_count": int(raw_members.size),
        "common_member_count": int(common_members.size),
    }
    changed = int(np.any(np.abs(q[:, 2:]) > 1e-12))
    return q, metadata, changed


def _content_hash(root: Path, manifest: Sequence[Mapping[str, Any]]) -> str:
    digest = hashlib.sha256()
    for item in sorted(manifest, key=lambda value: str(value["window_id"])):
        digest.update(str(item["window_id"]).encode()); digest.update(b"\0")
        path = root / str(item["input_path"])
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                digest.update(block)
    return digest.hexdigest()


def _build_features(root: Path, train_rows: Sequence[Mapping[str, Any]], validation_rows: Sequence[Mapping[str, Any]], geometry_items: Mapping[str, Mapping[str, Any]], support_by_id: Mapping[str, Mapping[str, Any]], *, resume: bool) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    selected = list(train_rows) + list(validation_rows)
    selected_ids = {str(row["window_id"]) for row in selected}
    manifest_path = root / "inputs/feature_manifest.json"
    summary_path = root / "support/summary.json"
    if resume and manifest_path.is_file() and summary_path.is_file():
        old = json.loads(manifest_path.read_text(encoding="utf-8")); old_by_id = {str(item["window_id"]): item for item in old}; old_summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if old_summary.get("feature_version") == "v1" and "raw_common_member_difference_unit_count" in old_summary and set(old_by_id) == selected_ids and all((root / str(item["input_path"])).is_file() for item in old):
            by_window = {str(item["window_id"]): item for item in old}
            train_kept = [{**dict(row), "matched_unit_count": int(by_window[str(row["window_id"])] ["matched_unit_count"])} for row in train_rows]
            val_kept = [{**dict(row), "matched_unit_count": int(by_window[str(row["window_id"])] ["matched_unit_count"])} for row in validation_rows]
            return train_kept, val_kept, json.loads(summary_path.read_text(encoding="utf-8")), {"manifest": old, "reused": True}
    manifest: list[dict[str, Any]] = []
    coverage: list[dict[str, Any]] = []
    coarse_coverage: list[dict[str, Any]] = []
    reason_counts: Counter[str] = Counter()
    loaded_sequences: dict[str, Any] = {}
    changed_units = 0
    visibility_geometry_equal = 0
    visibility_geometry_total = 0
    q_values: list[np.ndarray] = []
    original_unit_total = 0
    matched_unit_total = 0
    unformed_group_total = 0
    train_ids = {str(row["window_id"]) for row in train_rows}
    for index, item in enumerate(sorted((geometry_items[str(row["window_id"])] for row in selected), key=lambda value: str(value["window_id"]))):
        window_id = str(item["window_id"]); row = support_by_id[window_id]
        progress(root, "features", "RUNNING", index, len(selected), current_window=window_id)
        prefix = str(row.get("particle_prefix", ""))
        if prefix not in loaded_sequences:
            loaded_sequences[prefix] = geometry._sequence_for_row(row)
        sequence = loaded_sequences[prefix]
        arrays = _load_npz(GEOMETRY_ROOT / str(item["input_path"]))
        s = np.asarray(arrays["s3"], dtype=np.float64)
        intervals = np.asarray(arrays["intervals"], dtype=np.float64)
        valid_units = [unit for unit in row.get("support", {}).get("units", []) if unit.get("status") == "VALID"]
        stored_states = np.asarray(row.get("features", {}).get("SET_A"), dtype=np.float64)
        if stored_states.shape != (len(valid_units), 5, 4) or s.shape != stored_states.shape:
            raise ValueError(f"STRUCTURE_SHAPE_MISMATCH:{window_id}")
        unit_identities = item.get("unit_identities", [])
        if len(unit_identities) != len(valid_units):
            raise ValueError(f"UNIT_IDENTITY_COUNT_MISMATCH:{window_id}")
        groups = {int(group["local_group_id"]): group for group in row.get("grouping", {}).get("groups", [])}
        unformed_group_total += sum(1 for group in groups.values() if not bool(group.get("retained")))
        q_parts: list[np.ndarray] = []; metadata_parts: list[dict[str, Any]] = []
        for unit_index, identity in enumerate(unit_identities):
            if int(identity.get("unit_index", -1)) != unit_index:
                raise ValueError(f"UNIT_INDEX_ORDER_MISMATCH:{window_id}")
            unit = valid_units[unit_index]; local_group_id = int(identity["local_group_id"]); group = groups.get(local_group_id)
            if group is None or not bool(group.get("retained")):
                raise ValueError(f"RAW_GROUP_NOT_RETAINED:{window_id}:{local_group_id}")
            if list(identity.get("member_slots", [])) != list(unit.get("member_slots", [])):
                raise ValueError(f"COMMON_MEMBER_IDENTITY_MISMATCH:{window_id}:{local_group_id}")
            common_s3_diff = float(np.max(np.abs(s[unit_index] - stored_states[unit_index])))
            if common_s3_diff > MATCH_TOLERANCE:
                raise ValueError(f"STRUCTURE_REUSE_MISMATCH:{window_id}:{local_group_id}:{common_s3_diff}")
            q, metadata, changed = _q_for_unit(row, identity, unit, sequence, group)
            metadata["window_id"] = window_id; metadata["parent_id"] = str(row["window_id"]).rsplit("::b", 1)[0]; metadata["query_cohort"] = str(sequence.provenance.get("query_cohort")); metadata["b_offset_s"] = float(row.get("offset_s", 0.0)); metadata["structure_reuse_max_abs"] = common_s3_diff
            q_parts.append(q); metadata_parts.append(metadata); q_values.append(q); changed_units += changed
            raw = np.asarray(metadata["raw_member_slots"], dtype=np.int64); target = np.asarray(metadata["target_array_indices"], dtype=np.int64)
            visibility = np.asarray(sequence.visibility, dtype=bool)[target][:, raw]; geom_valid = np.asarray(sequence.geometry_validity, dtype=bool)[target][:, raw]
            visibility_geometry_total += int(visibility.size); visibility_geometry_equal += int(np.sum(visibility == geom_valid))
        q_array = np.stack(q_parts, axis=0).astype(np.float64)
        relative = Path("features/window_inputs") / f"{_safe(window_id)}.npz"; path = root / relative; temporary = path.with_name(path.name + ".tmp.npz"); path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(temporary, s=s.astype(np.float32), q=q_array.astype(np.float32), intervals=intervals.astype(np.float64)); os.replace(temporary, path)
        matched_count = int(q_array.shape[0]); original_count = len(valid_units); original_unit_total += original_count; matched_unit_total += matched_count
        feature_row = {"window_id": window_id, "source_id": str(row["source_id"]), "role": str(row["role"]), "label": int(row["label"]), "annotation_category": str(row.get("annotation_category", "")), "offset_s": float(row.get("offset_s", 0.0)), "input_path": str(relative), "s_shape": list(s.shape), "q_shape": list(q_array.shape), "matched_unit_count": matched_count, "original_unit_count": original_count, "unit_identities": metadata_parts, "intervals_s": intervals.tolist()}
        window_changed = int(sum(bool(np.any(np.abs(part[:, 2:]) > 1e-12)) for part in q_parts))
        coarse_coverage.append({"source_id": str(row["source_id"]), "role": str(row["role"]), "offset_s": float(row.get("offset_s", 0.0)), "window_count": 1, "matched_unit_count": matched_count, "q_changed_unit_count": window_changed, "raw_member_min": min(metadata["raw_member_count"] for metadata in metadata_parts), "raw_member_mean": float(np.mean([metadata["raw_member_count"] for metadata in metadata_parts])), "raw_member_max": max(metadata["raw_member_count"] for metadata in metadata_parts), "common_member_mean": float(np.mean([metadata["common_member_count"] for metadata in metadata_parts])), "query_age_min_s": min(metadata["query_age_s"][0] for metadata in metadata_parts), "query_age_max_s": max(metadata["query_age_s"][-1] for metadata in metadata_parts)})
        manifest.append(feature_row)
        coverage.append({"window_id": window_id, "source_id": str(row["source_id"]), "role": str(row["role"]), "offset_s": float(row.get("offset_s", 0.0)), "split": "train" if window_id in train_ids else "validation", "status": "MATCHED", "original_unit_count": original_count, "matched_unit_count": matched_count, "q_changed_unit_count": window_changed, "unformed_group_count": int(sum(1 for group in groups.values() if not bool(group.get("retained")))), "loss_reason": ""})
    manifest.sort(key=lambda value: str(value["window_id"])); write_csv(root / "support/coverage.csv", coverage); atomic_json(manifest_path, manifest)
    coarse_keys = ("source_id", "role", "offset_s")
    coarse_aggregated: dict[tuple[str, str, float], dict[str, Any]] = {}
    for row in coarse_coverage:
        key = (str(row["source_id"]), str(row["role"]), float(row["offset_s"])); current = coarse_aggregated.setdefault(key, {"source_id": key[0], "role": key[1], "offset_s": key[2], "window_count": 0, "matched_unit_count": 0, "q_changed_unit_count": 0, "raw_member_min": row["raw_member_min"], "raw_member_max": row["raw_member_max"], "raw_member_sum": 0.0, "common_member_sum": 0.0, "query_age_min_s": row["query_age_min_s"], "query_age_max_s": row["query_age_max_s"]}); current["window_count"] += 1; current["matched_unit_count"] += int(row["matched_unit_count"]); current["q_changed_unit_count"] += int(row["q_changed_unit_count"]); current["raw_member_min"] = min(current["raw_member_min"], row["raw_member_min"]); current["raw_member_max"] = max(current["raw_member_max"], row["raw_member_max"]); current["raw_member_sum"] += float(row["raw_member_mean"]) * int(row["matched_unit_count"]); current["common_member_sum"] += float(row["common_member_mean"]) * int(row["matched_unit_count"]); current["query_age_min_s"] = min(current["query_age_min_s"], row["query_age_min_s"]); current["query_age_max_s"] = max(current["query_age_max_s"], row["query_age_max_s"])
    coarse_rows = []
    for row in coarse_aggregated.values():
        count = max(int(row["matched_unit_count"]), 1); coarse_rows.append({"source_id": row["source_id"], "role": row["role"], "offset_s": row["offset_s"], "window_count": row["window_count"], "matched_unit_count": row["matched_unit_count"], "q_changed_unit_count": row["q_changed_unit_count"], "raw_member_min": row["raw_member_min"], "raw_member_mean": row["raw_member_sum"] / count, "raw_member_max": row["raw_member_max"], "common_member_mean": row["common_member_sum"] / count, "query_age_min_s": row["query_age_min_s"], "query_age_max_s": row["query_age_max_s"]})
    write_csv(root / "support/q_coverage.csv", sorted(coarse_rows, key=lambda item: (str(item["source_id"]), str(item["role"]), float(item["offset_s"]))))
    content_hash = _content_hash(root, manifest)
    q_all = np.stack(q_values, axis=0) if q_values else np.empty((0, 5, 4), dtype=np.float64)
    q_flat = q_all.reshape(-1, 4) if q_all.size else np.empty((0, 4), dtype=np.float64)
    q_stats = {"channels": [{"name": name, "min": float(np.min(q_flat[:, idx])), "max": float(np.max(q_flat[:, idx])), "mean": float(np.mean(q_flat[:, idx])), "std": float(np.std(q_flat[:, idx])), "constant_fraction": float(np.mean(np.isclose(q_flat[:, idx], q_flat[0, idx]))) } for idx, name in enumerate(("visibility_fraction", "geometry_fraction", "visibility_change", "geometry_change"))]}
    summary = {"feature_version": "v1", "original_train_units": sum(int(row.get("valid_unit_count", 0)) for row in train_rows), "original_validation_units": sum(int(row.get("valid_unit_count", 0)) for row in validation_rows), "matched_train_units": sum(int(item["matched_unit_count"]) for item in manifest if item["window_id"] in train_ids), "matched_validation_units": sum(int(item["matched_unit_count"]) for item in manifest if item["window_id"] not in train_ids), "matched_train_windows": len(train_rows), "matched_validation_windows": len(validation_rows), "q_changed_unit_count": int(changed_units), "q_unit_count": int(q_all.shape[0]), "raw_common_member_difference_unit_count": int(sum(metadata["raw_member_count"] != metadata["common_member_count"] for item in manifest for metadata in item["unit_identities"])), "raw_member_count_min": int(min(metadata["raw_member_count"] for item in manifest for metadata in item["unit_identities"])), "raw_member_count_mean": float(np.mean([metadata["raw_member_count"] for item in manifest for metadata in item["unit_identities"]])), "raw_member_count_max": int(max(metadata["raw_member_count"] for item in manifest for metadata in item["unit_identities"])), "common_member_count_min": int(min(metadata["common_member_count"] for item in manifest for metadata in item["unit_identities"])), "common_member_count_mean": float(np.mean([metadata["common_member_count"] for item in manifest for metadata in item["unit_identities"]])), "common_member_count_max": int(max(metadata["common_member_count"] for item in manifest for metadata in item["unit_identities"])), "query_age_min_s": float(min(metadata["query_age_s"][0] for item in manifest for metadata in item["unit_identities"])), "query_age_max_s": float(max(metadata["query_age_s"][-1] for item in manifest for metadata in item["unit_identities"])), "history_reference_age_min_s": float(min(metadata["history_reference_age_s"] for item in manifest for metadata in item["unit_identities"])), "history_reference_age_max_s": float(max(metadata["history_reference_age_s"] for item in manifest for metadata in item["unit_identities"])), "raw_unformed_group_count": int(unformed_group_total), "visibility_geometry_equal_fraction": float(visibility_geometry_equal / visibility_geometry_total) if visibility_geometry_total else None, "q_stats": q_stats, "feature_content_sha256": content_hash, "source_support_sha256": sha256(SOURCE_ROOT / "support/window_support.json"), "reference_feature_manifest_sha256": sha256(GEOMETRY_ROOT / "inputs/feature_manifest.json"), "reason_counts": dict(reason_counts)}
    atomic_json(root / "support/summary.json", summary); atomic_json(root / "inputs/input_manifest.json", {"selected_window_count": len(manifest), "train_window_count": len(train_rows), "validation_window_count": len(validation_rows), "feature_content_sha256": content_hash, "source_support_sha256": summary["source_support_sha256"], "reference_feature_manifest_sha256": summary["reference_feature_manifest_sha256"]}); progress(root, "features", "COMPLETE", len(manifest), len(selected), **summary)
    train_kept = [{**dict(row), "matched_unit_count": int(next(item["matched_unit_count"] for item in manifest if item["window_id"] == str(row["window_id"])))} for row in train_rows]; val_kept = [{**dict(row), "matched_unit_count": int(next(item["matched_unit_count"] for item in manifest if item["window_id"] == str(row["window_id"])))} for row in validation_rows]
    return train_kept, val_kept, summary, {"manifest": manifest, "reused": False}


def _batch(root: Path, manifest: Sequence[Mapping[str, Any]], rows: Sequence[Mapping[str, Any]], standardizer_s: FeatureStandardizer, standardizer_q: FeatureStandardizer, condition: str) -> dict[str, Any]:
    row_by_id = {str(row["window_id"]): row for row in rows}; selected = [item for item in manifest if str(item["window_id"]) in row_by_id]; selected.sort(key=lambda item: str(item["window_id"]))
    s_parts: list[np.ndarray] = []; q_parts: list[np.ndarray] = []; intervals: list[np.ndarray] = []; indices: list[int] = []; ids: list[str] = []; labels: list[int] = []
    for window_index, item in enumerate(selected):
        arrays = _load_npz(root / str(item["input_path"])); s = standardizer_s.transform(np.asarray(arrays["s"], dtype=np.float64)); q = standardizer_q.transform(np.asarray(arrays["q"], dtype=np.float64)); count = int(s.shape[0]); s_parts.append(s); q_parts.append(q); intervals.append(np.repeat(np.asarray(arrays["intervals"], dtype=np.float64)[None, :], count, axis=0)); indices.extend([window_index] * count); ids.append(str(item["window_id"])); labels.append(int(row_by_id[str(item["window_id"])] ["label"]))
    if not ids:
        raise ValueError(f"NO_BATCH_WINDOWS:{condition}")
    s_all = np.concatenate(s_parts, axis=0); q_all = np.concatenate(q_parts, axis=0)
    if condition == "STRUCTURE_ONLY": states = s_all
    elif condition == "SUPPORT_ONLY": states = q_all
    elif condition == "STRUCTURE_SUPPORT": states = np.concatenate((s_all, q_all), axis=-1)
    elif condition == "STRUCTURE_DUP_CONTROL": states = np.concatenate((s_all, s_all), axis=-1)
    else: raise ValueError(condition)
    return {"states": states, "intervals": np.concatenate(intervals, axis=0), "window_index": np.asarray(indices, dtype=np.int64), "labels": np.asarray(labels, dtype=np.int64), "window_ids": ids, "window_rows": [row_by_id[item] for item in ids], "window_count": len(ids), "window_weights": source_class_weights([row_by_id[item] for item in ids])}


def _initial_model(condition: str, seed: int) -> ModalSetModel:
    import torch
    torch.manual_seed(int(seed)); np.random.seed(int(seed) & 0xFFFFFFFF)
    if condition in ("STRUCTURE_ONLY", "SUPPORT_ONLY"):
        return ModalSetModel(4)
    base = ModalSetModel(4); fused = ModalSetModel(8)
    with torch.no_grad():
        fused.encoder[0].weight.zero_(); fused.encoder[0].weight[:, :4].copy_(base.encoder[0].weight); fused.encoder[0].bias.copy_(base.encoder[0].bias)
        for target, source in ((fused.encoder[2], base.encoder[2]), (fused.head[0], base.head[0]), (fused.head[2], base.head[2]), (fused.head[4], base.head[4])):
            target.weight.copy_(source.weight); target.bias.copy_(source.bias)
    return fused


def _forward(model: ModalSetModel, batch: Mapping[str, Any], device: str) -> Any:
    import torch
    states = torch.as_tensor(batch["states"], dtype=torch.float32, device=device); intervals = torch.as_tensor(batch["intervals"], dtype=torch.float32, device=device); indices = torch.as_tensor(batch["window_index"], dtype=torch.long, device=device); local = model(states, intervals); sums = torch.zeros(int(batch["window_count"]), dtype=local.dtype, device=device); counts = torch.zeros_like(sums); sums.index_add_(0, indices, local); counts.index_add_(0, indices, torch.ones_like(local)); return sums / counts.clamp_min(1.0)


def _train_one(condition: str, batch: Mapping[str, Any], seed: int, device: str, epochs: int = EPOCHS) -> tuple[ModalSetModel, dict[str, Any]]:
    import torch
    import torch.nn.functional as F
    model = _initial_model(condition, seed).to(device); model.train(); optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY); labels = torch.as_tensor(batch["labels"], dtype=torch.float32, device=device); weights = torch.as_tensor(batch["window_weights"], dtype=torch.float32, device=device); history: list[dict[str, Any]] = []
    for epoch in range(1, epochs + 1):
        optimizer.zero_grad(set_to_none=True); scores = _forward(model, batch, device); loss = torch.sum(F.binary_cross_entropy_with_logits(scores, labels, reduction="none") * weights) / torch.sum(weights)
        if not torch.isfinite(loss): raise FloatingPointError(f"NONFINITE_LOSS:{condition}:{seed}:{epoch}")
        loss.backward()
        if not all(parameter.grad is None or torch.all(torch.isfinite(parameter.grad)) for parameter in model.parameters()): raise FloatingPointError(f"NONFINITE_GRADIENT:{condition}:{seed}:{epoch}")
        optimizer.step(); history.append({"epoch": epoch, "loss": float(loss.detach().cpu())})
        if epoch in (1, epochs): print(f"observation-support condition={condition} seed={seed} epoch={epoch}/{epochs} loss={history[-1]['loss']:.6f}", flush=True)
    model.eval(); return model, {"seed": int(seed), "epochs": int(epochs), "initial_loss": history[0]["loss"], "final_loss": history[-1]["loss"], "min_loss": min(item["loss"] for item in history), "loss_history": history, "device": str(torch.device(device)), "parameter_count": parameter_count(model)}


def _model_identity(feature_manifest: Sequence[Mapping[str, Any]], train_rows: Sequence[Mapping[str, Any]], validation_rows: Sequence[Mapping[str, Any]], condition: str, seed: int, standardizer_s: FeatureStandardizer, standardizer_q: FeatureStandardizer, feature_hash: str) -> dict[str, Any]:
    return {"condition": str(condition), "seed": int(seed), "feature_content_sha256": feature_hash, "feature_manifest_sha256": hashlib.sha256(json.dumps(feature_manifest, sort_keys=True, separators=(",", ":")).encode()).hexdigest(), "training_window_ids_sha256": hashlib.sha256("\n".join(sorted(str(row["window_id"]) for row in train_rows)).encode()).hexdigest(), "validation_window_ids_sha256": hashlib.sha256("\n".join(sorted(str(row["window_id"]) for row in validation_rows)).encode()).hexdigest(), "standardizer_s": standardizer_s.as_dict(), "standardizer_q": standardizer_q.as_dict(), "model_config": {"conditions": list(CONDITIONS), "epochs": EPOCHS, "learning_rate": LEARNING_RATE, "weight_decay": WEIGHT_DECAY}}


def _record_matches(record: Mapping[str, Any], expected: Mapping[str, Any]) -> bool:
    actual = record.get("input_identity"); return isinstance(actual, Mapping) and all(actual.get(key) == value for key, value in expected.items())


def _model_load(condition: str, record: Mapping[str, Any], device: str) -> ModalSetModel:
    import torch
    model = ModalSetModel(4 if condition in ("STRUCTURE_ONLY", "SUPPORT_ONLY") else 8); model.load_state_dict({name: torch.as_tensor(value, dtype=model.state_dict()[name].dtype) for name, value in record["state_dict"].items()}); model.to(device); model.eval(); return model


def _metrics(rows: Sequence[Mapping[str, Any]], scores: Sequence[float]) -> tuple[dict[str, Any], dict[str, float]]:
    labels = [int(row["label"]) for row in rows]; values = [float(item) for item in scores]
    if not all(math.isfinite(value) for value in values): raise ValueError("NONFINITE_SCORE")
    by_source: dict[str, list[tuple[int, float]]] = defaultdict(list)
    for row, score in zip(rows, values): by_source[str(row["source_id"])].append((int(row["label"]), score))
    source_values = {source: periodic._auroc([pair[0] for pair in pairs], [pair[1] for pair in pairs]) for source, pairs in by_source.items()}; source_values = {source: float(value) for source, value in source_values.items() if value is not None}
    return {"window_count": len(rows), "real_count": labels.count(0), "fake_count": labels.count(1), "source_count": len(by_source), "dual_role_source_count": len(source_values), "source_macro": geometry.source128._bootstrap(source_values), "pooled_auroc": periodic._auroc(labels, values), "pooled_ap": periodic._ap(labels, values), **periodic._classification(labels, values)}, source_values


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


def evaluate(root: Path, feature_manifest: Sequence[Mapping[str, Any]], train_rows: Sequence[Mapping[str, Any]], validation_rows: Sequence[Mapping[str, Any]], records: Sequence[Mapping[str, Any]], standardizer_s: FeatureStandardizer, standardizer_q: FeatureStandardizer, device: str) -> dict[str, Any]:
    store: dict[tuple[str, str, Any], dict[str, float]] = {}; metric_rows: list[dict[str, Any]] = []; per_seed_rows: list[dict[str, Any]] = []
    for condition in CONDITIONS:
        for record in [item for item in records if str(item["condition"]) == condition]:
            seed = int(record["seed"]); model = _model_load(condition, record, device)
            for split, rows in (("train", train_rows), ("validation", validation_rows)):
                batch = _batch(root, feature_manifest, rows, standardizer_s, standardizer_q, condition); scores = _forward(model, batch, device).detach().cpu().numpy().astype(np.float64); store[(split, condition, seed)] = {window_id: float(value) for window_id, value in zip(batch["window_ids"], scores)}; metrics, _ = _metrics(batch["window_rows"], scores); row = _metric_row(split, condition, seed, metrics); metric_rows.append(row); per_seed_rows.append(row)
            del model
    mean_store: dict[tuple[str, str], dict[str, float]] = {}
    for split, rows in (("train", train_rows), ("validation", validation_rows)):
        for condition in CONDITIONS:
            scores = {str(row["window_id"]): float(np.mean([store[(split, condition, seed)][str(row["window_id"])] for seed in SEEDS])) for row in rows}; mean_store[(split, condition)] = scores; metrics, _ = _metrics(rows, [scores[str(row["window_id"])] for row in rows]); metric_rows.append(_metric_row(split, condition, "MEAN_LOGIT", metrics))
    per_source_rows: list[dict[str, Any]] = []
    for split, rows in (("train", train_rows), ("validation", validation_rows)):
        for source in sorted({str(row["source_id"]) for row in rows}):
            source_rows = [row for row in rows if str(row["source_id"]) == source]
            for condition in CONDITIONS:
                for seed in (*SEEDS, "MEAN_LOGIT"):
                    mapping = mean_store[(split, condition)] if seed == "MEAN_LOGIT" else store[(split, condition, int(seed))]; values = [mapping[str(row["window_id"])] for row in source_rows]; per_source_rows.append({"split": split, "source_id": source, "condition": condition, "seed": seed, "window_count": len(values), "real_count": sum(int(row["label"]) == 0 for row in source_rows), "fake_count": sum(int(row["label"]) == 1 for row in source_rows), "auroc": periodic._auroc([int(row["label"]) for row in source_rows], values)})
    source_values: dict[str, dict[str, float]] = {}
    for condition in CONDITIONS:
        values = mean_store[("validation", condition)]; source_values[condition] = {}
        for source in sorted({str(row["source_id"]) for row in validation_rows}):
            subset = [row for row in validation_rows if str(row["source_id"]) == source]; value = periodic._auroc([int(row["label"]) for row in subset], [values[str(row["window_id"])] for row in subset]);
            if value is not None: source_values[condition][source] = float(value)
    comparisons: list[dict[str, Any]] = []
    for left, right in (("STRUCTURE_SUPPORT", "STRUCTURE_DUP_CONTROL"), ("STRUCTURE_SUPPORT", "STRUCTURE_ONLY"), ("SUPPORT_ONLY", "STRUCTURE_ONLY"), ("STRUCTURE_DUP_CONTROL", "STRUCTURE_ONLY")):
        result = geometry.source128._bootstrap(source_values[left], source_values[right]); result.update({"name": f"{left}-{right}", "left": left, "right": right}); differences = [source_values[left][source] - source_values[right][source] for source in result["sources"]]; result.update({"positive_count": sum(value > 0 for value in differences), "negative_count": sum(value < 0 for value in differences), "tie_count": sum(value == 0 for value in differences)}); comparisons.append(result)
    seed_comparisons: list[dict[str, Any]] = []
    for seed in SEEDS:
        per_seed_source: dict[str, dict[str, float]] = {}
        for condition in CONDITIONS:
            per_seed_source[condition] = {}
            for source in sorted({str(row["source_id"]) for row in validation_rows}):
                subset = [row for row in validation_rows if str(row["source_id"]) == source]; mapping = store[("validation", condition, seed)]; value = periodic._auroc([int(row["label"]) for row in subset], [mapping[str(row["window_id"])] for row in subset]);
                if value is not None: per_seed_source[condition][source] = float(value)
        for left, right in (("STRUCTURE_SUPPORT", "STRUCTURE_DUP_CONTROL"), ("STRUCTURE_SUPPORT", "STRUCTURE_ONLY")):
            diffs = [per_seed_source[left][source] - per_seed_source[right][source] for source in sorted(set(per_seed_source[left]) & set(per_seed_source[right]))]; seed_comparisons.append({"seed": int(seed), "comparison": f"{left}-{right}", "mean_source_difference": float(np.mean(diffs)) if diffs else None, "positive_count": sum(value > 0 for value in diffs), "negative_count": sum(value < 0 for value in diffs), "tie_count": sum(value == 0 for value in diffs), "source_count": len(diffs)})
    write_csv(root / "scores/train_window_scores.csv", _score_rows(train_rows, store, "train")); write_csv(root / "scores/validation_window_scores.csv", _score_rows(validation_rows, store, "validation")); write_csv(root / "evaluation/metrics.csv", metric_rows); write_csv(root / "evaluation/per_seed_metrics.csv", per_seed_rows); write_csv(root / "evaluation/per_source_metrics.csv", per_source_rows); write_csv(root / "evaluation/paired_comparisons.csv", [{"comparison": item["name"], "source_count": item["source_count"], "mean": item["mean"], "ci_low": item["ci95"][0], "ci_high": item["ci95"][1], "positive_count": item["positive_count"], "negative_count": item["negative_count"], "tie_count": item["tie_count"]} for item in comparisons]); atomic_json(root / "evaluation/seed_comparisons.json", seed_comparisons)
    summary = {"train_population": {"windows": len(train_rows), "real": sum(int(row["label"]) == 0 for row in train_rows), "fake": sum(int(row["label"]) == 1 for row in train_rows), "sources": len({str(row["source_id"]) for row in train_rows})}, "validation_population": {"windows": len(validation_rows), "real": sum(int(row["label"]) == 0 for row in validation_rows), "fake": sum(int(row["label"]) == 1 for row in validation_rows), "sources": len({str(row["source_id"]) for row in validation_rows})}, "metrics": metric_rows, "comparisons": comparisons, "seed_comparisons": seed_comparisons, "device": str(device), "model_count": len(records)}; atomic_json(root / "evaluation/summary.json", summary); return summary


def _smoke(root: Path, feature_manifest: Sequence[Mapping[str, Any]], train_rows: Sequence[Mapping[str, Any]], standardizer_s: FeatureStandardizer, standardizer_q: FeatureStandardizer, device: str) -> dict[str, Any]:
    import torch
    rows = list(train_rows[:8]); if_labels = {int(row["label"]) for row in rows}
    if if_labels != {0, 1}: raise ValueError("SMOKE_NEEDS_BOTH_CLASSES")
    base = _initial_model("STRUCTURE_ONLY", SEEDS[0]).to(device).eval(); control = _initial_model("STRUCTURE_DUP_CONTROL", SEEDS[0]).to(device).eval(); base_batch = _batch(root, feature_manifest, rows, standardizer_s, standardizer_q, "STRUCTURE_ONLY"); control_batch = _batch(root, feature_manifest, rows, standardizer_s, standardizer_q, "STRUCTURE_DUP_CONTROL")
    with torch.no_grad(): degeneration = float(torch.max(torch.abs(_forward(base, base_batch, device) - _forward(control, control_batch, device))).detach().cpu())
    conditions: list[dict[str, Any]] = []
    for condition in CONDITIONS:
        batch = _batch(root, feature_manifest, rows, standardizer_s, standardizer_q, condition); model, fit = _train_one(condition, batch, SEEDS[0], device, epochs=1); before = _forward(model, batch, device).detach().cpu().numpy(); reloaded = _model_load(condition, {"state_dict": state_dict_numpy(model)}, device); after = _forward(reloaded, batch, device).detach().cpu().numpy(); conditions.append({"condition": condition, "parameter_count": parameter_count(model), "epochs": fit["epochs"], "scores_finite": bool(np.all(np.isfinite(before))), "reload_max_abs": float(np.max(np.abs(before - after))), "passed": bool(np.all(np.isfinite(before)) and np.max(np.abs(before - after)) <= 1e-5)}); del model, reloaded
    result = {"status": "PASS" if all(item["passed"] for item in conditions) and degeneration <= 1e-5 else "FAILED", "rows": len(rows), "initial_fused_control_max_abs": degeneration, "conditions": conditions, "device": device}; atomic_json(root / "smoke/summary.json", result); return result


def _write_protocol(root: Path, info: Mapping[str, Any], feature_summary: Mapping[str, Any], feature_manifest: Sequence[Mapping[str, Any]]) -> None:
    import torch
    atomic_json(root / "protocol.json", {"protocol_id": "v7-observation-support-pilot-v1", "git_head": git_head(), "source_root": str(SOURCE_ROOT), "reference_geometry_root": str(GEOMETRY_ROOT), "input": {"train_windows": info["original_train_windows"], "validation_windows": info["original_validation_windows"], "matched_train_windows": feature_summary["matched_train_windows"], "matched_validation_windows": feature_summary["matched_validation_windows"], "training_sources": info["training_sources"], "validation_sources": info["validation_sources"], "source_support_sha256": feature_summary["source_support_sha256"], "reference_feature_manifest_sha256": feature_summary["reference_feature_manifest_sha256"], "feature_content_sha256": feature_summary["feature_content_sha256"]}, "definition": {"raw_history_members": "grouping.groups member_slots/track_ids", "structure_members": "support.units member_slots/track_ids", "q": "[visibility_fraction, geometry_fraction, visibility_minus_history, geometry_minus_history]", "history_reference": "last history_array_indices frame strictly before first actual target PTS", "visibility_semantics": "saved frontend visibility mask; not occlusion truth", "geometry_semantics": "saved frontend geometry_validity mask; not geometric truth"}, "conditions": {"STRUCTURE_ONLY": "S", "SUPPORT_ONLY": "Q", "STRUCTURE_SUPPORT": "standardized S concatenated with standardized Q", "STRUCTURE_DUP_CONTROL": "standardized S concatenated with itself"}, "model": {"architecture": "Linear(D,16)-ReLU-Linear(16,8), five-time mean, four PTS intervals, 12-16-8-1 head", "parameter_counts": {"four_dimensional": parameter_count(ModalSetModel(4)), "eight_dimensional": parameter_count(ModalSetModel(8))}, "epochs": EPOCHS, "optimizer": "Adam", "learning_rate": LEARNING_RATE, "weight_decay": WEIGHT_DECAY, "seeds": list(SEEDS), "threshold": "logit >= 0", "loss": "source/class weighted BCE at window level"}, "evaluation": {"primary": "STRUCTURE_SUPPORT minus STRUCTURE_DUP_CONTROL source-macro AUROC", "secondary": "STRUCTURE_SUPPORT minus STRUCTURE_ONLY", "bootstrap_seed": BOOTSTRAP_SEED, "bootstrap_replicates": BOOTSTRAP_REPLICATES, "not_sealed_test": True}, "runtime": {"torch_version": str(torch.__version__), "cuda_available": bool(torch.cuda.is_available()), "cuda_version": str(torch.version.cuda), "deterministic_algorithms": bool(torch.are_deterministic_algorithms_enabled())}, "manifest_count": len(feature_manifest)})


def _write_report(root: Path, info: Mapping[str, Any], feature_summary: Mapping[str, Any], q_summary: Mapping[str, Any], summary: Mapping[str, Any], records: Sequence[Mapping[str, Any]], timings: Mapping[str, float], smoke: Mapping[str, Any]) -> Path:
    metrics = {(str(item["split"]), str(item["condition"])): item for item in summary.get("metrics", []) if str(item["seed"]) == "MEAN_LOGIT"}
    lines = ["# V7 结构状态与观测支撑信息增量 pilot", "", "本实验复用同一 R query cohort、局部组、五时刻、结构 unit 和标签，比较结构状态 S、观测支撑 Q、S+Q 与重复 S 控制。Q 使用历史原始组成员；S 使用五时刻共同有效成员。", "", "## 结果先行", f"- 正式模型：{len(records)}/12；smoke：{smoke.get('status')}；设备：{summary.get('device')}。", f"- 训练 {info['original_train_windows']} 窗口/{len(info['training_sources'])} source，验证 {info['original_validation_windows']} 窗口/{len(info['validation_sources'])} source（{info['original_validation_real']} real、{info['original_validation_fake']} fake）。共同支撑窗口和 unit 未改变：{feature_summary['matched_train_windows']}/{feature_summary['matched_validation_windows']} 窗口，{feature_summary['matched_train_units']}/{feature_summary['matched_validation_units']} unit。", f"- 未形成结构的原始组数量仅作盲区记录：{feature_summary.get('raw_unformed_group_count', 0)}；没有把它们作为分类样本。", "", "| 条件 | source-macro AUROC (95% CI) | pooled AUROC | AP | Precision | Recall | F1 | ACC |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for condition in CONDITIONS:
        item = metrics.get(("validation", condition), {}); lines.append(f"| {condition} | {item.get('source_macro_auroc')} [{item.get('source_macro_ci_low')}, {item.get('source_macro_ci_high')}] | {item.get('pooled_auroc')} | {item.get('pooled_ap')} | {item.get('precision')} | {item.get('recall')} | {item.get('f1')} | {item.get('accuracy')} |")
    lines += ["", "## 主要配对", "", "| 比较 | source 数 | 均值差 | 95% CI | 正/负/平 source |", "|---|---:|---:|---|---:|"]
    for item in summary.get("comparisons", []): lines.append(f"| {item['name']} | {item['source_count']} | {item['mean']} | [{item['ci95'][0]}, {item['ci95'][1]}] | {item['positive_count']}/{item['negative_count']}/{item['tie_count']} |")
    lines += ["", "## Q 可用性与混淆核对", f"- Q 单元数：{q_summary.get('q_unit_count')}；存在目标阶段支撑变化的 unit：{q_summary.get('q_changed_unit_count')}。", f"- 原始成员数范围/均值：{q_summary.get('raw_member_count_min')}/{q_summary.get('raw_member_count_mean')}/{q_summary.get('raw_member_count_max')}；共同成员数范围/均值：{q_summary.get('common_member_count_min')}/{q_summary.get('common_member_count_mean')}/{q_summary.get('common_member_count_max')}；两者不同的 unit：{q_summary.get('raw_common_member_difference_unit_count')}。", f"- 查询年龄范围：{q_summary.get('query_age_min_s')}–{q_summary.get('query_age_max_s')} s；历史参照年龄范围：{q_summary.get('history_reference_age_min_s')}–{q_summary.get('history_reference_age_max_s')} s。source/role/b 粗覆盖见 `support/q_coverage.csv`。", f"- visibility 与 geometry_validity 在原始组成员/目标时刻上的相等比例：{q_summary.get('visibility_geometry_equal_fraction')}；两者是保存的前端判断，不是真实遮挡或几何真值。", "- Q 四通道范围、标准差和恒定比例见 `support/summary.json`；Q 的变化量是相对历史参照比例差，不是速度、加速度或物理形变。", "- 原始成员集合与共同成员集合分开保存在每个 unit identity 中；本轮没有把共同成员数量当作 Q 分母。", "", "## 解释边界", "- 主要比较 `STRUCTURE_SUPPORT - STRUCTURE_DUP_CONTROL` 若区间跨 0，不能宣称支撑状态提供稳定增量；`STRUCTURE_SUPPORT - STRUCTURE_ONLY` 是实用辅助比较。", "- SUPPORT_ONLY 较高只能说明保存的观测状态与标签存在可预测关系，不能说明发现了物理伪造规律。", "- 共同有效结构窗口造成 survivor bias；验证集是已开发的固定验证集，不是 sealed-test。结果不回答像素/空间定位、因果预测、纯前端二维系统或真实遮挡真值。", "- 未形成结构的历史组和无有效 unit 的区域不进入分类器，仍是观测盲区；没有填 0 或伪造分数。", "", "## 产物与成本", f"- 阶段耗时（秒）：features={timings.get('features')}, smoke={timings.get('smoke')}, train={timings.get('train')}, evaluate={timings.get('evaluate')}, report={timings.get('report')}。", "- 逐窗口分数：`scores/train_window_scores.csv`、`scores/validation_window_scores.csv`；逐 source/seed：`evaluation/per_source_metrics.csv`、`evaluation/per_seed_metrics.csv`；配对：`evaluation/paired_comparisons.csv`。", "- 未访问旧 R7/V5；未重跑前端、tracking、depth、pose 或 segmentation。", ""]
    path = root / "report.md"; path.write_text("\n".join(lines), encoding="utf-8"); return path


def run(root: Path = OUTPUT_ROOT, *, device: str = "cuda", resume: bool = True) -> dict[str, Any]:
    root.mkdir(parents=True, exist_ok=True); (root / "state").mkdir(parents=True, exist_ok=True); started = time.perf_counter(); timings: dict[str, float] = {}
    atomic_json(root / "state/launch.json", {"git_head": git_head(), "pid": os.getpid(), "device": device, "started_unix": time.time()})
    try:
        train_original, validation_original, geometry_items, support_by_id, info = _load_reference(); feature_started = time.perf_counter(); train_rows, validation_rows, feature_summary, built = _build_features(root, train_original, validation_original, geometry_items, support_by_id, resume=resume); timings["features"] = time.perf_counter() - feature_started; feature_manifest = built["manifest"]
        train_ids = {str(row["window_id"]) for row in train_rows}; by_id = {str(item["window_id"]): item for item in feature_manifest}; ordered_train = [by_id[str(row["window_id"])] for row in train_rows]; ordered_val = [by_id[str(row["window_id"])] for row in validation_rows]; standardizer_s = fit_standardizer("SET_A", [_load_npz(root / str(item["input_path"]))["s"] for item in ordered_train], source_class_weights(train_rows)); standardizer_q = fit_standardizer("SET_A", [_load_npz(root / str(item["input_path"]))["q"] for item in ordered_train], source_class_weights(train_rows)); _write_protocol(root, info, feature_summary, feature_manifest)
        smoke_started = time.perf_counter(); smoke = json.loads((root / "smoke/summary.json").read_text()) if resume and (root / "smoke/summary.json").is_file() else _smoke(root, feature_manifest, train_rows, standardizer_s, standardizer_q, device); timings["smoke"] = time.perf_counter() - smoke_started
        if smoke.get("status") != "PASS": raise RuntimeError("SMOKE_FAILED")
        model_path = root / "models/fold_models.json"; existing = json.loads(model_path.read_text()).get("records", []) if resume and model_path.is_file() else []; feature_hash = str(feature_summary["feature_content_sha256"]); records: list[dict[str, Any]] = []; complete: set[tuple[str, int]] = set(); unverified: list[dict[str, Any]] = []
        for old in existing:
            condition, seed = str(old.get("condition", "")), int(old.get("seed", -1)); expected = _model_identity(feature_manifest, train_rows, validation_rows, condition, seed, standardizer_s, standardizer_q, feature_hash) if condition in CONDITIONS and seed in SEEDS else None
            if old.get("status") == "TRAIN_COMPLETE" and expected and _record_matches(old, expected): records.append(old); complete.add((condition, seed))
            elif old: unverified.append({"condition": condition, "seed": seed, "status": old.get("status")})
        if unverified: atomic_json(root / "state/unverified_model_records.json", {"reason": "missing_or_mismatched_input_identity", "records": unverified})
        train_started = time.perf_counter(); total = len(CONDITIONS) * len(SEEDS)
        for condition in CONDITIONS:
            batch = _batch(root, feature_manifest, train_rows, standardizer_s, standardizer_q, condition)
            for seed in SEEDS:
                if (condition, int(seed)) in complete: continue
                model, fit = _train_one(condition, batch, seed, device); record = {"condition": condition, "seed": int(seed), "status": "TRAIN_COMPLETE", "parameter_count": parameter_count(model), "fit": fit, "standardization": {"s": standardizer_s.as_dict(), "q": standardizer_q.as_dict()}, "input_identity": _model_identity(feature_manifest, train_rows, validation_rows, condition, seed, standardizer_s, standardizer_q, feature_hash), "state_dict": state_dict_numpy(model)}; records.append(record); complete.add((condition, int(seed))); atomic_json(model_path, {"conditions": list(CONDITIONS), "seeds": list(SEEDS), "epochs": EPOCHS, "records": records}); progress(root, "train", "RUNNING", len(complete), total, condition=condition, seed=seed); del model
        timings["train"] = time.perf_counter() - train_started
        if len(complete) != total: raise RuntimeError(f"MODEL_COUNT_INCOMPLETE:{len(complete)}/{total}")
        evaluate_started = time.perf_counter(); summary = evaluate(root, feature_manifest, train_rows, validation_rows, records, standardizer_s, standardizer_q, device); timings["evaluate"] = time.perf_counter() - evaluate_started; report_started = time.perf_counter(); report_path = _write_report(root, info, feature_summary, feature_summary, summary, records, timings, smoke); timings["report"] = time.perf_counter() - report_started; _write_report(root, info, feature_summary, feature_summary, summary, records, timings, smoke); atomic_json(root / "final_status.json", {"status": "COMPLETE", "model_count": len(records), "expected_model_count": total, "train_windows": len(train_rows), "validation_windows": len(validation_rows), "report": str(report_path), "timings_s": timings, "git_head": git_head()}); progress(root, "report", "COMPLETE", 1, 1, model_count=len(records), elapsed_s=time.perf_counter() - started); return {"status": "COMPLETE", "report": str(report_path), "summary": summary, "timings_s": timings}
    except BaseException as exc:
        atomic_json(root / "state/failure.json", {"status": "FAILED", "error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()}); atomic_json(root / "final_status.json", {"status": "FAILED", "error": f"{type(exc).__name__}: {exc}", "git_head": git_head(), "elapsed_s": time.perf_counter() - started}); progress(root, "failed", "FAILED", 0, 1, error=f"{type(exc).__name__}: {exc}"); raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT); parser.add_argument("--device", default="cuda"); parser.add_argument("--resume", action="store_true"); args = parser.parse_args(); result = run(args.output_root, device=args.device, resume=args.resume); print(json.dumps({"status": result["status"], "report": result["report"], "timings_s": result["timings_s"]}, ensure_ascii=False, indent=2)); return 0


if __name__ == "__main__":
    raise SystemExit(main())
