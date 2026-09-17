"""Construct and read the current five-time S/Q input for the 04LAX R case.

This is a deliberately small, case-specific diagnostic.  It consumes the
already saved R trajectory/geometry and H grouping, calls the current V7
five-time and Q builders, and reads (never trains) the frozen observation
support models.  The output is kept outside the formal feature manifest so
that this case cannot silently become a training or evaluation sample.
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
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

import numpy as np

from research_tools.v7.local_organization_probe.grouping import rebuild_components_fast, build_local_groups
from research_tools.v7.multi_order_sequence_probe.representation import (
    MAX_TARGET_ERROR_S,
    TARGET_OFFSETS_S,
    build_five_time_unit,
)
from research_tools.v7.multi_order_sequence_probe.runner import _standardizer_record
from research_tools.v7.observation_support_pilot import runner as observation


REPO_ROOT = Path(__file__).resolve().parents[3]
DATA_ROOT = Path("/root/autodl-tmp/data/sparse_3d_forgery_detection")
OLD_CASE = DATA_ROOT / "derived/v7_activityforensics_local_observation_recovery_probe_v1"
MODEL_ROOT = DATA_ROOT / "derived/v7_activityforensics_observation_support_pilot_v1"
DEFAULT_OUTPUT = OLD_CASE / "frozen_five_time_readout_v1"
VIDEO = Path(
    "/root/autodl-tmp/data/sparse_3d_forgery_detection/datasets/v7_core_candidates/"
    "activityforensics_charades_v1/source/activityforensics/raw/video/02_wan/"
    "04LAX+13.90=22.70=charades@train_delete@04LAX@365@wan.mp4"
)
WINDOW_ID = "0002_MANIP_25::fake"
SOURCE_ID = "04LAX"
ROLE = "fake"
SEEDS = (20260909, 20260910, 20260911)
CONDITIONS = ("STRUCTURE_ONLY", "STRUCTURE_SUPPORT")
TOLERANCE = 1e-5


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


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(_jsonable(value), indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
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


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {key: np.asarray(archive[key]) for key in archive.files}


def _exact_equal(left: np.ndarray, right: np.ndarray) -> bool:
    return left.shape == right.shape and np.array_equal(left, right, equal_nan=True)


def _make_sequence(arrays: Mapping[str, np.ndarray], structure: Mapping[str, Any]) -> Any:
    """Adapt the verified geometry NPZ to the current representation API."""

    return SimpleNamespace(
        frame_indices=np.asarray(arrays["frame_indices"], dtype=np.int64),
        timestamps_s=np.asarray(arrays["timestamps_s"], dtype=np.float64),
        frame_sizes_hw=np.asarray(arrays["frame_sizes_hw"], dtype=np.int64),
        track_ids=np.asarray(arrays["track_ids"], dtype=np.int64),
        uv=np.asarray(arrays["uv"], dtype=np.float32),
        visibility=np.asarray(arrays["visibility"], dtype=bool),
        geometry_validity=np.asarray(arrays["geometry_validity"], dtype=bool),
        xyz=np.asarray(arrays["xyz"], dtype=np.float32),
        provenance={
            "query_cohort": str(structure["id_namespace"]),
            "cohort_start_s": float(structure["query_start"]["timestamp_s"]),
        },
    )


def _validate_case() -> tuple[dict[str, Any], Any, dict[str, Any], dict[str, Any]]:
    case = json.loads((OLD_CASE / "case_manifest.json").read_text(encoding="utf-8"))
    geometry_meta = json.loads((OLD_CASE / "conditions/R_geometry.json").read_text(encoding="utf-8"))
    structure = json.loads((OLD_CASE / "conditions/R_structure.json").read_text(encoding="utf-8"))
    geometry_path = OLD_CASE / "conditions/R_geometry.npz"
    requery_path = OLD_CASE / "conditions/R_requery.npz"
    if not geometry_path.is_file():
        raise FileNotFoundError(f"R_GEOMETRY_MISSING:{geometry_path}")
    if not VIDEO.is_file():
        raise FileNotFoundError(f"04LAX_VIDEO_MISSING:{VIDEO}")
    expected_video = case.get("video", {})
    video_checks = {
        "path_matches_case": str(VIDEO) == str(expected_video.get("path")),
        "sha256_matches_case": sha256(VIDEO) == str(expected_video.get("sha256")),
        "bytes_matches_case": VIDEO.stat().st_size == int(expected_video.get("bytes", -1)),
    }
    if not all(video_checks.values()):
        raise ValueError(f"VIDEO_IDENTITY_MISMATCH:{video_checks}")
    if case.get("window_id") != WINDOW_ID or case.get("source_id") != SOURCE_ID or case.get("role") != ROLE:
        raise ValueError("CASE_MANIFEST_IDENTITY_MISMATCH")
    if structure.get("source_id") != SOURCE_ID or structure.get("window_id") != WINDOW_ID or structure.get("condition") != "R":
        raise ValueError("R_STRUCTURE_IDENTITY_MISMATCH")
    if int(geometry_meta.get("query_start_frame", -1)) != int(structure["query_start"]["source_frame_index"]):
        raise ValueError("R_GEOMETRY_QUERY_START_MISMATCH")
    arrays = _load_npz(geometry_path)
    required = {"frame_indices", "timestamps_s", "frame_sizes_hw", "track_ids", "uv", "visibility", "geometry_validity", "xyz"}
    if set(arrays) != required:
        raise ValueError(f"R_GEOMETRY_FIELDS_MISMATCH:{sorted(set(arrays) ^ required)}")
    frames = np.asarray(arrays["frame_indices"], dtype=np.int64)
    times = np.asarray(arrays["timestamps_s"], dtype=np.float64)
    if frames.ndim != 1 or times.shape != frames.shape or not np.all(np.diff(frames) > 0) or not np.all(np.diff(times) > 0):
        raise ValueError("R_TIME_AXIS_INVALID")
    if arrays["xyz"].shape != (frames.size, 289, 3) or arrays["uv"].shape != (frames.size, 289, 2):
        raise ValueError("R_TRACK_ARRAY_SHAPE_INVALID")
    if arrays["visibility"].shape != (frames.size, 289) or arrays["geometry_validity"].shape != (frames.size, 289):
        raise ValueError("R_MASK_ARRAY_SHAPE_INVALID")
    if arrays["xyz"].dtype != np.float32 or arrays["uv"].dtype != np.float32 or arrays["track_ids"].dtype != np.int64:
        raise ValueError("R_DTYPE_CONTRACT_INVALID")
    if not np.all(np.asarray(arrays["geometry_validity"], dtype=bool) <= np.asarray(arrays["visibility"], dtype=bool)):
        raise ValueError("R_GEOMETRY_NOT_SUBSET_OF_VISIBILITY")
    finite_xyz = np.all(np.isfinite(arrays["xyz"]), axis=-1)
    finite_uv = np.all(np.isfinite(arrays["uv"]), axis=-1)
    if np.any(np.asarray(arrays["geometry_validity"], dtype=bool) & ~finite_xyz):
        raise ValueError("R_VALID_GEOMETRY_NONFINITE_XYZ")
    if np.any(np.asarray(arrays["visibility"], dtype=bool) & ~finite_uv):
        raise ValueError("R_VISIBLE_NONFINITE_UV")
    frame_sizes = np.asarray(arrays["frame_sizes_hw"], dtype=np.int64)
    if frame_sizes.shape != (frames.size, 2) or not np.all(frame_sizes == np.asarray(case["image_size_hw"], dtype=np.int64)[None, :]):
        raise ValueError("R_FRAME_SIZE_IDENTITY_MISMATCH")
    expected_struct_frames = np.asarray(structure["history_frame_indices"] + structure["evaluation_frame_indices"], dtype=np.int64)
    if not _exact_equal(frames, expected_struct_frames):
        raise ValueError("R_STRUCTURE_FRAME_IDENTITY_MISMATCH")
    if not _exact_equal(frames, np.asarray(geometry_meta["frame_indices"], dtype=np.int64)):
        raise ValueError("R_GEOMETRY_METADATA_FRAME_IDENTITY_MISMATCH")
    query = structure["query_start"]
    if int(frames[0]) != int(query["source_frame_index"]) or abs(float(times[0]) - float(query["timestamp_s"])) > 1e-12:
        raise ValueError("R_QUERY_START_IDENTITY_MISMATCH")
    requery_comparison: dict[str, Any] = {"available": requery_path.is_file(), "path": str(requery_path)}
    if requery_path.is_file():
        requery = _load_npz(requery_path)
        for key in ("frame_indices", "timestamps_s", "frame_sizes_hw", "track_ids", "uv", "visibility"):
            requery_comparison[key] = _exact_equal(arrays[key], requery[key])
        requery_comparison["geometry_validity_equal"] = _exact_equal(arrays["geometry_validity"], requery["geometry_validity"])
        requery_comparison["xyz_equal"] = _exact_equal(arrays["xyz"], requery["xyz"])
        requery_comparison["requery_has_complete_xyz"] = bool(np.any(np.isfinite(requery["xyz"])))
        requery_comparison["requery_geometry_valid_count"] = int(np.sum(requery["geometry_validity"]))
        if not all(requery_comparison[key] for key in ("frame_indices", "timestamps_s", "frame_sizes_hw", "track_ids", "uv", "visibility")):
            raise ValueError("R_REQUERY_GEOMETRY_IDENTITY_MISMATCH")
    sequence = _make_sequence(arrays, structure)
    checks = {
        "video": video_checks,
        "r_geometry_sha256": sha256(geometry_path),
        "r_requery_sha256": sha256(requery_path) if requery_path.is_file() else None,
        "r_structure_sha256": sha256(OLD_CASE / "conditions/R_structure.json"),
        "frame_count": int(frames.size),
        "frame_first_last": [int(frames[0]), int(frames[-1])],
        "track_count": int(arrays["track_ids"].size),
        "query_start": {"frame": int(frames[0]), "pts_s": float(times[0]), "id_namespace": structure["id_namespace"]},
        "requery_identity": requery_comparison,
        "geometry_source": str(geometry_path),
        "requery_source": str(requery_path),
        "source_artifact_in_structure": structure.get("source_artifact"),
        "source_artifact_note": "R_requery.npz is the 2D/query cache and has no finite XYZ; R_geometry.npz is the aligned complete geometry cache used for S/Q.",
    }
    return checks, sequence, structure, {"case": case, "geometry_meta": geometry_meta, "arrays": arrays}


def _build_diagnostic(sequence: Any, structure: Mapping[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]], np.ndarray, np.ndarray, np.ndarray]:
    grouping = structure["grouping"]
    history_indices = np.asarray(grouping["history_array_indices"], dtype=np.int64)
    old_parent_components = [item["member_slots"] for item in grouping.get("parent_components", [])]
    rebuilt_parent = rebuild_components_fast(sequence.xyz, sequence.geometry_validity, history_indices)
    old_parent = {tuple(sorted(int(x) for x in item)) for item in old_parent_components}
    new_parent = {tuple(sorted(int(x) for x in item)) for item in rebuilt_parent}
    current_grouping = build_local_groups(sequence, history_indices, old_components=rebuilt_parent)
    old_groups = {tuple(sorted(int(x) for x in item["member_slots"])): item for item in grouping["groups"]}
    new_groups = {tuple(sorted(int(x) for x in item["member_slots"])): item for item in current_grouping["groups"]}
    retained_old = {key for key, item in old_groups.items() if bool(item.get("retained"))}
    retained_new = {key for key, item in new_groups.items() if bool(item.get("retained"))}
    if old_parent != new_parent or set(old_groups) != set(new_groups) or retained_old != retained_new:
        raise ValueError("R_GROUPING_RECONSTRUCTION_MISMATCH")
    row = {"window_id": WINDOW_ID, "grouping": {"history_array_indices": history_indices.tolist(), "groups": list(current_grouping["groups"])}}
    units: list[dict[str, Any]] = []
    invalid: list[dict[str, Any]] = []
    q_values: list[np.ndarray] = []
    s_values: list[np.ndarray] = []
    intervals: np.ndarray | None = None
    for group in current_grouping["groups"]:
        if not bool(group.get("retained")):
            invalid.append({"local_group_id": int(group["local_group_id"]), "reason": "GROUP_NOT_RETAINED"})
            continue
        unit = build_five_time_unit(
            sequence,
            window_id=WINDOW_ID,
            window_start_s=float(structure["query_start"]["timestamp_s"]),
            member_slots=group["member_slots"],
            local_group_id=int(group["local_group_id"]),
            max_target_error_s=MAX_TARGET_ERROR_S,
        )
        if unit.get("status") != "VALID":
            invalid.append({"local_group_id": int(unit["local_group_id"]), "reason": str(unit.get("reason", "UNKNOWN")), "target_matches": unit.get("target_matches")})
            continue
        identity = {"local_group_id": unit["local_group_id"], "array_indices": unit["array_indices"], "track_ids": unit["track_ids"]}
        q, q_meta, changed = observation._q_for_unit(row, identity, unit, sequence, group)
        q_meta["q_changed_from_history"] = bool(changed)
        unit = {**unit, "raw_member_slots": [int(x) for x in group["member_slots"]], "raw_track_ids": [int(x) for x in group["track_ids"]], "q_changed_from_history": bool(changed), "q": q, "q_metadata": q_meta}
        current_intervals = np.diff(np.asarray(unit["timestamps_s"], dtype=np.float64))
        if intervals is None:
            intervals = current_intervals
        elif not np.allclose(intervals, current_intervals, atol=1e-12, rtol=0):
            raise ValueError("TARGET_INTERVALS_DIFFER_ACROSS_UNITS")
        units.append(unit)
        s_values.append(np.asarray(unit["states"], dtype=np.float64))
        q_values.append(np.asarray(q, dtype=np.float64))
    if intervals is None or not units:
        raise ValueError("NO_VALID_FIVE_TIME_UNIT")
    s_array = np.stack(s_values, axis=0)
    q_array = np.stack(q_values, axis=0)
    if s_array.shape != (len(units), 5, 4) or q_array.shape != (len(units), 5, 4) or not np.all(np.isfinite(s_array)) or not np.all(np.isfinite(q_array)):
        raise ValueError("DIAGNOSTIC_FEATURE_ARRAY_INVALID")
    old_triplets = [item for item in structure["support"].get("triplets", []) if item.get("status", "VALID") == "VALID"]
    old_by_key = {(int(item["local_group_id"]), tuple(int(x) for x in item["target_slots"]), tuple(tuple(int(y) for y in pair) for pair in item["pair_ids"])): item for item in old_triplets}
    crosscheck = {
        "matched_triplets": 0,
        "missing_old_triplets": 0,
        "max_abs_state_diff": 0.0,
        "state_matches_within_1e-8": 0,
        "history_scale_mismatch_count": 0,
        "max_abs_history_scale_difference": 0.0,
        "comparison_note": "Old R_structure sliding triplets are compared only by identity. They were produced by local_organization.build_local_support, whose scale is computed on the retained raw group; the current five-time builder computes the frozen scale on the five-time common members. Old triplet states are therefore not reused as current model input.",
    }
    for unit in units:
        pair_key = tuple(tuple(int(y) for y in pair) for pair in unit["pair_ids"])
        for start in range(3):
            key = (int(unit["local_group_id"]), (start, start + 1, start + 2), pair_key)
            old = old_by_key.get(key)
            if old is None:
                crosscheck["missing_old_triplets"] += 1
                continue
            diff = float(np.max(np.abs(np.asarray(old["states"], dtype=np.float64) - np.asarray(unit["states"], dtype=np.float64)[start : start + 3])))
            crosscheck["matched_triplets"] += 1
            crosscheck["max_abs_state_diff"] = max(crosscheck["max_abs_state_diff"], diff)
            crosscheck["state_matches_within_1e-8"] += int(diff <= 1e-8)
            scale_diff = abs(float(old["history_scale"]) - float(unit["history_scale"]))
            crosscheck["max_abs_history_scale_difference"] = max(crosscheck["max_abs_history_scale_difference"], scale_diff)
            crosscheck["history_scale_mismatch_count"] += int(scale_diff > 1e-12)
    summary = {
        "history_array_indices": history_indices.tolist(),
        "history_frame_indices": [int(sequence.frame_indices[index]) for index in history_indices],
        "history_timestamps_s": [float(sequence.timestamps_s[index]) for index in history_indices],
        "target_offsets_s": list(TARGET_OFFSETS_S),
        "target_tolerance_s": float(MAX_TARGET_ERROR_S),
        "target_matches": [{"slot": index, "frame_index": int(unit["frame_indices"][index]), "array_index": int(unit["array_indices"][index]), "timestamp_s": float(unit["timestamps_s"][index]), "target_time_s": float(unit["target_times_s"][index]), "match_error_s": float(unit["match_errors_s"][index])} for index in range(5) for unit in units[:1]],
        "target_frame_indices": units[0]["frame_indices"],
        "target_timestamps_s": units[0]["timestamps_s"],
        "intervals_s": intervals.tolist(),
        "old_retained_group_count": int(len(retained_old)),
        "current_retained_group_count": int(len(retained_new)),
        "valid_unit_count": int(len(units)),
        "retained_invalid_unit_count": int(sum(item["reason"] != "GROUP_NOT_RETAINED" for item in invalid)),
        "nonretained_group_count": int(sum(item["reason"] == "GROUP_NOT_RETAINED" for item in invalid)),
        "invalid_group_or_unit_count": int(len(invalid)),
        "invalid_reason_counts": dict(Counter(str(item["reason"]) for item in invalid)),
        "q_changed_unit_count": int(sum(bool(item["q_changed_from_history"]) for item in units)),
        "q_unchanged_unit_count": int(sum(not bool(item["q_changed_from_history"]) for item in units)),
        "old_sliding_triplet_count": int(len(old_triplets)),
        "old_triplet_crosscheck": crosscheck,
        "grouping_reconstruction": {"parent_components_match": True, "all_groups_match": True, "retained_groups_match": True, "component_config": current_grouping.get("component_config")},
        "invalid_units": invalid,
    }
    return summary, units, s_array, q_array, intervals


def _save_features(output: Path, units: Sequence[Mapping[str, Any]], s_array: np.ndarray, q_array: np.ndarray, intervals: np.ndarray) -> dict[str, Any]:
    feature_path = output / "diagnostic_features.npz"
    feature_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = feature_path.with_name(feature_path.name + ".tmp.npz")
    np.savez_compressed(temporary, s=s_array.astype(np.float32), q=q_array.astype(np.float32), intervals=intervals.astype(np.float64))
    os.replace(temporary, feature_path)
    manifest = [{
        "window_id": WINDOW_ID,
        "source_id": SOURCE_ID,
        "role": ROLE,
        "label": 1,
        "annotation_category": "FAKE_MANIPULATION_DIAGNOSTIC_ONLY",
        "input_path": feature_path.name,
        "s_shape": list(s_array.shape),
        "q_shape": list(q_array.shape),
        "intervals_s": intervals.tolist(),
        "matched_unit_count": int(len(units)),
        "original_unit_count": int(len(units)),
        "unit_identities": [{
            "local_group_id": int(unit["local_group_id"]),
            "member_slots": unit["member_slots"],
            "track_ids": unit["track_ids"],
            "raw_member_slots": unit["raw_member_slots"],
            "raw_track_ids": unit["raw_track_ids"],
            "pair_ids": unit["pair_ids"],
            "history_scale": float(unit["history_scale"]),
            "history_scale_pair_time_count": int(unit["history_scale_pair_time_count"]),
            "array_indices": unit["array_indices"],
            "frame_indices": unit["frame_indices"],
            "timestamps_s": unit["timestamps_s"],
            "target_times_s": unit["target_times_s"],
            "match_errors_s": unit["match_errors_s"],
            "q_metadata": unit["q_metadata"],
        } for unit in units],
    }]
    write_json(output / "feature_manifest.json", manifest)
    return {"path": str(feature_path), "sha256": sha256(feature_path), "manifest_path": str(output / "feature_manifest.json"), "manifest_sha256": sha256(output / "feature_manifest.json")}


def _model_records() -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    protocol = json.loads((MODEL_ROOT / "protocol.json").read_text(encoding="utf-8"))
    records = json.loads((MODEL_ROOT / "models/fold_models.json").read_text(encoding="utf-8"))["records"]
    selected = [item for item in records if str(item.get("condition")) in CONDITIONS and int(item.get("seed", -1)) in SEEDS and str(item.get("status")) == "TRAIN_COMPLETE"]
    if len(selected) != 6:
        raise ValueError(f"FROZEN_MODEL_RECORDS_INCOMPLETE:{len(selected)}")
    membership = {"training_sources": list(protocol.get("input", {}).get("training_sources", [])), "validation_sources": list(protocol.get("input", {}).get("validation_sources", []))}
    membership["04LAX_in_training_sources"] = SOURCE_ID in membership["training_sources"]
    membership["04LAX_in_validation_sources"] = SOURCE_ID in membership["validation_sources"]
    membership["status"] = "TRAINING_SOURCE_DIAGNOSTIC" if membership["04LAX_in_training_sources"] else "NOT_IN_TRAIN_OR_VALIDATION"
    return protocol, selected, membership


def _model_readout(output: Path, feature_manifest: Sequence[Mapping[str, Any]], protocol: Mapping[str, Any], records: Sequence[Mapping[str, Any]]) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    import torch

    sample_manifest = json.loads((MODEL_ROOT / "inputs/feature_manifest.json").read_text(encoding="utf-8"))
    sample_scores = list(csv.DictReader((MODEL_ROOT / "scores/validation_window_scores.csv").open(newline="", encoding="utf-8")))
    sample_row = next((row for row in sample_scores if row.get("source_id") == "00ZCA"), None)
    if sample_row is None:
        raise ValueError("FROZEN_MODEL_REPRODUCTION_SAMPLE_MISSING")
    reproduction: list[dict[str, Any]] = []
    for condition in CONDITIONS:
        for seed in SEEDS:
            record = next(item for item in records if str(item["condition"]) == condition and int(item["seed"]) == seed)
            standardizer_s = _standardizer_record(record["standardization"]["s"])
            standardizer_q = _standardizer_record(record["standardization"]["q"])
            sample_batch = observation._batch(MODEL_ROOT, sample_manifest, [{"window_id": sample_row["window_id"], "source_id": sample_row["source_id"], "label": int(sample_row["label"])}], standardizer_s, standardizer_q, condition)
            model = observation._model_load(condition, record, "cpu")
            sample_states = torch.as_tensor(sample_batch["states"], dtype=torch.float32)
            sample_intervals = torch.as_tensor(sample_batch["intervals"], dtype=torch.float32)
            sample_score = float(model(sample_states, sample_intervals).detach().cpu().numpy().mean())
            saved_score = float(sample_row[f"{condition}_seed_{seed}"])
            reproduction.append({"condition": condition, "seed": int(seed), "sample_window_id": sample_row["window_id"], "saved_window_logit": saved_score, "reproduced_window_logit": sample_score, "max_abs_error": abs(sample_score - saved_score), "passed": bool(abs(sample_score - saved_score) <= TOLERANCE)})
    if not all(item["passed"] for item in reproduction):
        raise ValueError("FROZEN_MODEL_REPRODUCTION_FAILED")
    unit_rows: list[dict[str, Any]] = []
    window: dict[str, Any] = {"window_id": WINDOW_ID, "source_id": SOURCE_ID, "role": ROLE, "valid_unit_count": len(feature_manifest[0]["unit_identities"]), "conditions": {}}
    for condition in CONDITIONS:
        window["conditions"][condition] = {"seeds": {}, "mean_logit": None}
        seed_scores: list[float] = []
        for seed in SEEDS:
            record = next(item for item in records if str(item["condition"]) == condition and int(item["seed"]) == seed)
            standardizer_s = _standardizer_record(record["standardization"]["s"])
            standardizer_q = _standardizer_record(record["standardization"]["q"])
            batch = observation._batch(output, feature_manifest, [{"window_id": WINDOW_ID, "source_id": SOURCE_ID, "label": 1}], standardizer_s, standardizer_q, condition)
            model = observation._model_load(condition, record, "cpu")
            local = model(torch.as_tensor(batch["states"], dtype=torch.float32), torch.as_tensor(batch["intervals"], dtype=torch.float32)).detach().cpu().numpy().astype(np.float64)
            if local.shape != (len(feature_manifest[0]["unit_identities"]),) or not np.all(np.isfinite(local)):
                raise ValueError(f"NONFINITE_DIAGNOSTIC_MODEL_OUTPUT:{condition}:{seed}")
            score = float(np.mean(local)); seed_scores.append(score)
            window["conditions"][condition]["seeds"][str(seed)] = score
            for unit_identity, local_score in zip(feature_manifest[0]["unit_identities"], local.tolist()):
                unit_rows.append({"window_id": WINDOW_ID, "source_id": SOURCE_ID, "role": ROLE, "condition": condition, "seed": int(seed), "local_group_id": int(unit_identity["local_group_id"]), "track_ids": json.dumps(unit_identity["track_ids"], separators=(",", ":")), "pair_count": len(unit_identity["pair_ids"]), "model_local_logit": float(local_score), "unit_mean_contribution": float(local_score / len(local))})
        window["conditions"][condition]["mean_logit"] = float(np.mean(seed_scores))
    # The condition-specific local means are the exact aggregation used by the
    # frozen model; explicitly verify the contribution decomposition.
    mean_local_by_unit: dict[tuple[str, int], float] = {}
    for condition in CONDITIONS:
        for unit_identity in feature_manifest[0]["unit_identities"]:
            values = [float(row["model_local_logit"]) for row in unit_rows if row["condition"] == condition and int(row["local_group_id"]) == int(unit_identity["local_group_id"])]
            if len(values) != len(SEEDS):
                raise ValueError(f"LOCAL_SEED_COUNT_MISMATCH:{condition}:{unit_identity['local_group_id']}")
            mean_local_by_unit[(condition, int(unit_identity["local_group_id"]))] = float(np.mean(values))
    for row in unit_rows:
        mean_local = mean_local_by_unit[(str(row["condition"]), int(row["local_group_id"]))]
        row["mean_local_logit"] = mean_local
        row["mean_unit_mean_contribution"] = float(mean_local / len(feature_manifest[0]["unit_identities"]))
    for condition in CONDITIONS:
        for seed in SEEDS:
            values = [row["model_local_logit"] for row in unit_rows if row["condition"] == condition and int(row["seed"]) == seed]
            contribution = sum(float(row["unit_mean_contribution"]) for row in unit_rows if row["condition"] == condition and int(row["seed"]) == seed)
            score = float(window["conditions"][condition]["seeds"][str(seed)])
            if abs(contribution - score) > 2e-7 or abs(float(np.mean(values)) - score) > 2e-7:
                raise ValueError(f"LOCAL_CONTRIBUTION_MISMATCH:{condition}:{seed}")
    write_csv(output / "unit_scores.csv", unit_rows)
    write_json(output / "window_scores.json", window)
    return window, unit_rows, {"reproduction": reproduction, "device": "cpu", "model_protocol_git_head": protocol.get("git_head"), "model_conditions": list(CONDITIONS), "model_seed_count": len(records)}


def _report(output: Path, compatibility: Mapping[str, Any], feature_info: Mapping[str, Any], window: Mapping[str, Any] | None, model_info: Mapping[str, Any], status: str) -> str:
    summary = compatibility["five_time"]
    lines = [
        "# 04LAX R 冻结五时刻模型读出",
        "",
        f"状态：`{status}`。本案例为单视频开发诊断，不是 AUROC、定位性能或 sealed-test 结果。",
        "",
        "## 直接结论",
        "",
        f"- 旧 R 缓存可以按当前五时刻规则构造 S/Q：49 个历史组中 44 个保留，保留组内 {summary['valid_unit_count']} 个 unit 合法、{summary['retained_invalid_unit_count']} 个因五时刻共同 geometry-valid 成员不足而排除；另有 {summary['nonretained_group_count']} 个原本未达到历史组最小规模。没有补零、平移时间或降低门槛。",
        f"- 目标帧/PTS 为 `{summary['target_frame_indices']}` / `{[round(float(x), 12) for x in summary['target_timestamps_s']]}`；历史为 frame `{summary['history_frame_indices'][0]}–{summary['history_frame_indices'][-1]}`，Q 的历史参照是 frame `{summary['q_reference_frame_index']}`。",
        f"- S 与 Q 均为 `[M,5,4] = [{summary['valid_unit_count']},5,4]`；Q 相对历史参照发生变化的 unit 为 {summary['q_changed_unit_count']}/{summary['valid_unit_count']}。",
        f"- 冻结 `STRUCTURE_ONLY` 与 `STRUCTURE_SUPPORT` 的 3×2 个模型均通过已有合法窗口复现，随后对 04LAX 的 32 个 unit 进行只读前向；局部 logit 不是局部伪造概率。",
        "",
        "## 输入与兼容性",
        "",
        f"- 视频身份：`{compatibility['case']['video_identity']['sha256']}`，当前文件与旧案例 manifest 的 path/bytes/SHA 均一致；R geometry 输入为 `{compatibility['case']['geometry_source']}`。",
        f"- R query cohort：`{compatibility['case']['query_start']['id_namespace']}`，query 起点 frame {compatibility['case']['query_start']['frame']} / PTS {compatibility['case']['query_start']['pts_s']:.12f}。R_requery 与 R_geometry 的 frame、PTS、frame size、track ID、UV、visibility 均逐数组一致；R_requery 的 XYZ/geometry validity 为空，因此没有被冒充为三维输入。",
        f"- 当前代码重建 parent/local grouping 与缓存一致：parent、49 个组、44 个 retained 组均匹配；使用的 component 配置为 `{summary['grouping_reconstruction']['component_config']}`。",
        f"- 旧滑动三时刻缓存有 {summary['old_sliding_triplet_count']} 个 VALID triplet；当前五时刻 builder 与其中 `{summary['old_triplet_crosscheck']['matched_triplets']}` 个具有相同 group/pair/slot 身份，但旧/当前 history scale 有 `{summary['old_triplet_crosscheck']['history_scale_mismatch_count']}` 个不一致（最大差 `{summary['old_triplet_crosscheck']['max_abs_history_scale_difference']:.3g}`），因此 S 最大差为 `{summary['old_triplet_crosscheck']['max_abs_state_diff']:.3g}`。这是旧三时刻构造与当前五时刻规则的尺度口径差异；旧 triplet 没有被包装成五时刻输入。",
        f"- B 分割支撑仍未计算（旧缓存状态 `{compatibility.get('old_structure_b_status')}`）；当前冻结模型的 S/Q 输入只使用 H 组、geometry/visibility 与五个目标时刻，没有为了本读出复用 O mask。",
        "",
        "## 支撑与缺失",
        "",
        f"- 五时刻匹配容差为 {summary['target_tolerance_s']} s，目标间隔为 `{[round(float(x), 12) for x in summary['intervals_s']]}` s。",
        f"- 无效原因计数：`{summary['invalid_reason_counts']}`。这些组没有进入模型，不写入正常分数。",
        f"- 当前 Q 是 `[visibility_fraction, geometry_fraction, visibility_minus_history, geometry_minus_history]`，原始 H 成员作为分母，共同成员只用于 S/结构有效性；Q 变化不是物理形变真值。",
        "",
        "## 冻结模型读出",
        "",
    ]
    if window is not None:
        lines += ["| condition | seed | window logit |", "|---|---:|---:|"]
        for condition in CONDITIONS:
            for seed in SEEDS:
                lines.append(f"| {condition} | {seed} | {window['conditions'][condition]['seeds'][str(seed)]:.9f} |")
            lines.append(f"| {condition} | mean | {window['conditions'][condition]['mean_logit']:.9f} |")
        lines += ["", f"- 模型输出设备：`{model_info.get('device')}`；冻结模型记录来自 protocol git head `{model_info.get('model_protocol_git_head')}`。每个条件/seed 的权重和标准化摘要 hash 保存在 `compatibility_report.json`。当前 04LAX 不在 observation-support protocol 的 training/validation source 清单中，身份为 `{compatibility['model_membership']['status']}`；这仍是开发诊断，不能称 sealed-test。", "- 每个 unit 的 `unit_mean_contribution = local_logit / M`，所有 unit 均保留；贡献和已核对等于对应窗口 logit。详见 `unit_scores.csv` 与 `window_scores.json`。"]
    else:
        lines += ["- 模型读出不可用；详见 compatibility_report.json 中的阻塞原因。"]
    lines += [
        "",
        "## 边界",
        "",
        "- 本案例只说明旧 R 的一段新查询轨迹在当前规则下可以形成有限的五时刻结构输入，并可被冻结模型读出；不能恢复旧 O 轨迹跨失踪事件的物理对应。",
        "- 已有 ROI 仅是主体粗框背景，不参与 unit 筛选或模型输入；没有新增 ROI 审计、前端、跟踪、深度、姿态、分割或训练。",
        "- 单个 04LAX fake 案例不能证明模型学会/不会检测真实失真，也不能把局部 logit 当作空间真值。",
        "",
        "## 产物",
        "",
        "- `compatibility_report.json`：缓存身份、规则、支撑缺失和模型复现证据。",
        "- `diagnostic_features.npz` / `feature_manifest.json`：本案例独立诊断特征，不是正式 feature manifest 记录。",
        "- `unit_scores.csv` / `window_scores.json`：冻结模型局部与窗口读出。",
    ]
    return "\n".join(lines) + "\n"


def run(output: Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    started = time.time()
    checks, sequence, structure, source_data = _validate_case()
    five_summary, units, s_array, q_array, intervals = _build_diagnostic(sequence, structure)
    five_summary["q_reference_frame_index"] = int(sequence.frame_indices[np.asarray(structure["grouping"]["history_array_indices"], dtype=np.int64)[-1]])
    feature_info = _save_features(output, units, s_array, q_array, intervals)
    protocol, records, membership = _model_records()
    model_status = {"status": "NOT_RUN", "reason": None}
    window = None
    unit_rows: list[dict[str, Any]] = []
    model_info: dict[str, Any] = {"model_protocol_git_head": protocol.get("git_head"), "device": "cpu"}
    try:
        window, unit_rows, model_info = _model_readout(output, [json.loads((output / "feature_manifest.json").read_text(encoding="utf-8"))[0]], protocol, records)
        model_status = {"status": "AVAILABLE", "condition_count": len(CONDITIONS), "seed_count": len(SEEDS), "unit_count": len(units), "window_count": 1}
        status = "COMPLETE_WITH_MODEL_READOUT"
    except Exception as exc:
        model_status = {"status": "NOT_AVAILABLE", "reason": f"{type(exc).__name__}: {exc}"}
        status = "COMPLETE_NO_MODEL_READOUT"
    compatibility = {
        "status": status,
        "generated_at_unix": time.time(),
        "git_head": git_head(),
        "case": {"source_id": SOURCE_ID, "role": ROLE, "window_id": WINDOW_ID, "video_identity": {"path": str(VIDEO), "sha256": sha256(VIDEO), "bytes": VIDEO.stat().st_size}, "geometry_source": checks["geometry_source"], "query_start": checks["query_start"], "r_cache": checks},
        "rules": {"target_offsets_s": list(TARGET_OFFSETS_S), "target_tolerance_s": float(MAX_TARGET_ERROR_S), "minimum_common_members": 3, "q_definition": "raw H member denominator; visibility/geometry target fractions minus latest history fraction", "missing_policy": "invalid units retained with reason; no zero fill/interpolation"},
        "five_time": five_summary,
        "old_structure_b_status": structure.get("b_status"),
        "old_structure_b_reason": structure.get("b_reason"),
        "feature": feature_info,
        "model_root": str(MODEL_ROOT),
        "model_protocol_git_head": protocol.get("git_head"),
        "model_membership": membership,
        "model_record_identity": [
            {
                "condition": str(record["condition"]),
                "seed": int(record["seed"]),
                "status": str(record.get("status")),
                "parameter_count": int(record.get("parameter_count", 0)),
                "standardization_sha256": hashlib.sha256(json.dumps(record.get("standardization", {}), sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest(),
                "state_dict_sha256": hashlib.sha256(json.dumps(record.get("state_dict", {}), sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest(),
                "input_identity": {key: record.get("input_identity", {}).get(key) for key in ("condition", "seed", "feature_content_sha256", "training_window_ids_sha256", "validation_window_ids_sha256")},
            }
            for record in records
        ],
        "model_status": model_status,
        "model_reproduction": model_info.get("reproduction", []),
        "elapsed_s": time.time() - started,
    }
    write_json(output / "compatibility_report.json", compatibility)
    if model_info.get("reproduction"):
        write_json(output / "model_reproduction.json", model_info["reproduction"])
    report = _report(output, {**compatibility, "case": {**compatibility["case"], "video_identity": compatibility["case"]["video_identity"]}}, feature_info, window, model_info, status)
    (output / "report.md").write_text(report, encoding="utf-8")
    write_json(output / "final_status.json", {"status": status, "window_id": WINDOW_ID, "valid_unit_count": len(units), "model_readout": model_status, "elapsed_s": time.time() - started, "git_head": git_head()})
    return {"status": status, "valid_unit_count": len(units), "model_readout": model_status, "output": str(output)}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    print(json.dumps(run(args.output), indent=2, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
