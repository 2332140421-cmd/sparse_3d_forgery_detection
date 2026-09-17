"""Run the bounded 04LAX R17 versus nested R33 query-density diagnostic.

This module is intentionally case-specific.  It reuses the audited R17 2-D
trajectory, runs one nested 33x33 BootsTAPIR query on the same short clip, and
uses one shared Depth Pro/Open3D geometry result for both densities.  It does
not train a model or modify the formal V7 detection chain.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import time
from collections import Counter
from typing import Any, Mapping, Sequence

import numpy as np

from sparse3d_forgery.experiments.v7_explicit_geometry_frontend import (
    RasterScale,
    accumulate_world_from_camera,
    causal_first_frame_intrinsics,
    sample_depth_at_uv,
    world_xyz,
)
from sparse3d_forgery.particle_sequence import (
    CoordinateSystem,
    Handedness,
    LengthUnit,
    build_particle_sequence,
)
from sparse3d_forgery.video_input import VideoSource, decode_video

from research_tools.v7.local_organization_probe.grouping import build_local_groups, rebuild_components_fast
from research_tools.v7.local_structural_temporal_probe.representation import COMPONENT_CONFIG
from research_tools.v7.multi_order_sequence_probe.representation import (
    MAX_TARGET_ERROR_S,
    TARGET_OFFSETS_S,
    build_five_time_unit,
)
from research_tools.v7.multi_order_sequence_probe.runner import _standardizer_record
from research_tools.v7.observation_support_pilot import runner as observation
from research_tools.v7.observation_density_diagnostic import run_diagnostic as density


REPO_ROOT = Path(__file__).resolve().parents[3]
DATA_ROOT = Path("/root/autodl-tmp/data/sparse_3d_forgery_detection")
OLD_CASE = DATA_ROOT / "derived/v7_activityforensics_local_observation_recovery_probe_v1"
OUTPUT_ROOT = DATA_ROOT / "derived/v7_activityforensics_query_density_probe_v1"
MODEL_ROOT = DATA_ROOT / "derived/v7_activityforensics_observation_support_pilot_v1"
VIDEO = Path(
    "/root/autodl-tmp/data/sparse_3d_forgery_detection/datasets/v7_core_candidates/"
    "activityforensics_charades_v1/source/activityforensics/raw/video/02_wan/"
    "04LAX+13.90=22.70=charades@train_delete@04LAX@365@wan.mp4"
)
WINDOW_ID = "0002_MANIP_25::fake"
SOURCE_ID = "04LAX"
ROLE = "fake"
QUERY_FRAME = 488
HISTORY_LAST_FRAME = 502
TARGET_FRAMES = (503, 506, 509, 512, 515)
FRAME_RANGE = tuple(range(488, 519))
PROCESS_SIZE = 256
BASE_GRID_SIZE = 17
DENSE_GRID_SIZE = 33
SEEDS = (20260909, 20260910, 20260911)
MODEL_CONDITIONS = ("STRUCTURE_ONLY", "STRUCTURE_SUPPORT")
TOLERANCE = 1e-5


def _jsonable(value: Any, path: str = "$") -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return [_jsonable(item, f"{path}[{i}]") for i, item in enumerate(value)]
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return _jsonable(value.item(), path)
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"non-finite JSON value at {path}: {value!r}")
        return value
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v, f"{path}.{k}") for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v, f"{path}[{i}]") for i, v in enumerate(value)]
    return value


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(_jsonable(value), indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
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
        writer.writerows(rows)
    os.replace(tmp, path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_head() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()


def _grid_axis(process_size: int, grid_size: int) -> np.ndarray:
    if grid_size <= 0 or process_size <= 0:
        raise ValueError("grid_size and process_size must be positive")
    return np.linspace(0, process_size - 1, grid_size + 2, dtype=np.float32)[1:-1]


def nested_axes(process_size: int = PROCESS_SIZE) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the original 17 axis, midpoint-inserted 33 axis, and map."""

    original = _grid_axis(process_size, BASE_GRID_SIZE)
    dense = np.empty(DENSE_GRID_SIZE, dtype=np.float32)
    dense[0::2] = original
    dense[1::2] = (original[:-1] + original[1:]) / np.float32(2.0)
    if not np.all(np.diff(dense) > 0):
        raise AssertionError("nested axis is not strictly increasing")
    return original, dense, np.arange(BASE_GRID_SIZE, dtype=np.int64) * 2


def query_manifest(source_hw: tuple[int, int]) -> list[dict[str, Any]]:
    original, dense, old_indices = nested_axes()
    mapping = RasterScale(source_hw, (PROCESS_SIZE, PROCESS_SIZE))
    rows: list[dict[str, Any]] = []
    for row in range(DENSE_GRID_SIZE):
        for col in range(DENSE_GRID_SIZE):
            index = row * DENSE_GRID_SIZE + col
            is_original = row % 2 == 0 and col % 2 == 0
            old_id = (row // 2) * BASE_GRID_SIZE + (col // 2) if is_original else None
            process_uv = np.asarray([[dense[col], dense[row]]], dtype=np.float32)
            source_uv = mapping.process_to_source(process_uv)[0]
            rows.append({
                "query_id": int(index),
                "grid_row": int(row),
                "grid_col": int(col),
                "process_u": float(process_uv[0, 0]),
                "process_v": float(process_uv[0, 1]),
                "source_u": float(source_uv[0]),
                "source_v": float(source_uv[1]),
                "membership": "original" if is_original else "added",
                "original_query_id": int(old_id) if old_id is not None else None,
                "query_frame_index": QUERY_FRAME,
            })
    if len(rows) != 1089 or sum(row["membership"] == "original" for row in rows) != 289:
        raise AssertionError("unexpected query manifest size")
    original_rows = [row for row in rows if row["membership"] == "original"]
    for row in original_rows:
        expected = (row["grid_row"] // 2) * BASE_GRID_SIZE + row["grid_col"] // 2
        if row["original_query_id"] != expected:
            raise AssertionError("original query id mapping mismatch")
    return rows


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {key: np.asarray(archive[key]) for key in archive.files}


def _validate_case() -> tuple[dict[str, Any], dict[str, np.ndarray], dict[str, Any], dict[str, Any]]:
    case = json.loads((OLD_CASE / "case_manifest.json").read_text(encoding="utf-8"))
    structure = json.loads((OLD_CASE / "conditions/R_structure.json").read_text(encoding="utf-8"))
    geometry_meta = json.loads((OLD_CASE / "conditions/R_geometry.json").read_text(encoding="utf-8"))
    geometry_path = OLD_CASE / "conditions/R_geometry.npz"
    requery_path = OLD_CASE / "conditions/R_requery.npz"
    if not VIDEO.is_file() or sha256(VIDEO) != str(case.get("video", {}).get("sha256")) or VIDEO.stat().st_size != int(case.get("video", {}).get("bytes", -1)):
        raise ValueError("VIDEO_IDENTITY_MISMATCH")
    if case.get("source_id") != SOURCE_ID or case.get("role") != ROLE or case.get("window_id") != WINDOW_ID:
        raise ValueError("CASE_IDENTITY_MISMATCH")
    arrays = _load_npz(geometry_path)
    required = {"frame_indices", "timestamps_s", "frame_sizes_hw", "track_ids", "uv", "visibility", "geometry_validity", "xyz"}
    if set(arrays) != required:
        raise ValueError("R_GEOMETRY_FIELDS_MISMATCH")
    frames = np.asarray(arrays["frame_indices"], dtype=np.int64)
    if tuple(frames.tolist()) != FRAME_RANGE or arrays["uv"].shape != (31, 289, 2) or arrays["xyz"].shape != (31, 289, 3):
        raise ValueError("R_GEOMETRY_FRAME_OR_SHAPE_MISMATCH")
    if not np.array_equal(frames, np.asarray(geometry_meta["frame_indices"], dtype=np.int64)):
        raise ValueError("R_GEOMETRY_METADATA_FRAME_MISMATCH")
    if int(geometry_meta.get("query_start_frame", -1)) != QUERY_FRAME:
        raise ValueError("R_QUERY_FRAME_MISMATCH")
    if requery_path.is_file():
        requery = _load_npz(requery_path)
        for key in ("frame_indices", "timestamps_s", "frame_sizes_hw", "track_ids", "uv", "visibility"):
            if not np.array_equal(arrays[key], requery[key], equal_nan=True):
                raise ValueError(f"R_REQUERY_IDENTITY_MISMATCH:{key}")
    if not np.all(np.asarray(arrays["geometry_validity"], dtype=bool) <= np.asarray(arrays["visibility"], dtype=bool)):
        raise ValueError("R_GEOMETRY_MASK_CONTRACT")
    if np.any(np.asarray(arrays["geometry_validity"], dtype=bool) & ~np.all(np.isfinite(arrays["xyz"]), axis=-1)):
        raise ValueError("R_GEOMETRY_FINITE_CONTRACT")
    source_hw = tuple(int(x) for x in np.asarray(arrays["frame_sizes_hw"])[0])
    manifest = query_manifest((source_hw[0], source_hw[1]))
    query = {
        "process_size": PROCESS_SIZE,
        "original_grid_size": BASE_GRID_SIZE,
        "dense_grid_size": DENSE_GRID_SIZE,
        "original_query_count": 289,
        "added_query_count": 800,
        "dense_query_count": 1089,
        "axis_definition": "17-axis from the audited 17x17 tracker formula; insert midpoint between every adjacent original axis value; no outer linspace points",
        "original_axis_process": nested_axes()[0].tolist(),
        "dense_axis_process": nested_axes()[1].tolist(),
        "original_ids_in_dense": [row["query_id"] for row in manifest if row["membership"] == "original"],
        "query_frame_index": QUERY_FRAME,
        "query_pts_s": float(np.asarray(arrays["timestamps_s"])[0]),
        "coordinate_system": "process raster (u,v), mapped to source pixels by RasterScale pixel-center mapping",
    }
    return case, arrays, structure, {"geometry_meta": geometry_meta, "query": query, "query_manifest": manifest}


def _track_with_queries(tracker: Any, decoded: Any, query_process_uv: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Use the audited tracker internals with a frozen custom query array."""

    torch = tracker.torch
    source_h, source_w = decoded.frames[0].rgb.shape[:2]
    if any(frame.rgb.shape[:2] != (source_h, source_w) for frame in decoded.frames):
        raise RuntimeError("VARIABLE_RASTER_SIZE")
    if query_process_uv.shape != (1089, 2) or not np.all(np.isfinite(query_process_uv)):
        raise ValueError("DENSE_QUERY_SHAPE")
    query = np.stack((np.zeros(query_process_uv.shape[0], np.float32), query_process_uv[:, 1], query_process_uv[:, 0]), axis=1)
    frames = np.stack([frame.rgb for frame in decoded.frames])
    tensor = torch.from_numpy(frames).to(tracker.device)
    tensor = tensor.permute(0, 3, 1, 2).float()
    tensor = tracker.functional.interpolate(tensor, size=(tracker.process_size, tracker.process_size), mode="bilinear", align_corners=False).permute(0, 2, 3, 1)
    tensor = tensor / 255 * 2 - 1
    query_tensor = torch.from_numpy(query).to(tracker.device)
    with torch.inference_mode():
        first = tensor[None, 0:1]
        feature_grids = tracker.model.get_feature_grids(first, is_training=False)
        query_features = tracker.model.get_query_features(first, is_training=False, query_points=query_tensor[None], feature_grids=feature_grids)
        causal_state = tracker.model.construct_initial_causal_state(query.shape[0], len(query_features.resolutions) - 1)
        causal_state = tracker.tree.map_structure(lambda value: value.to(tracker.device), causal_state)
        tracks: list[np.ndarray] = []
        visibility: list[np.ndarray] = []
        for frame in tensor:
            single = frame[None, None]
            grids = tracker.model.get_feature_grids(single, is_training=False)
            result = tracker.model.estimate_trajectories(single.shape[-3:-1], is_training=False, feature_grids=grids, query_features=query_features, query_points_in_video=None, query_chunk_size=64, causal_context=causal_state, get_causal_context=True)
            causal_state = result["causal_context"]
            tracks.append(result["tracks"][-1][0, :, 0].detach().cpu().numpy())
            occlusion = result["occlusion"][-1][0, :, 0]
            expected = result["expected_dist"][-1][0, :, 0]
            visible = ((1 - torch.sigmoid(occlusion)) * (1 - torch.sigmoid(expected)) > 0.5)
            visibility.append(visible.detach().cpu().numpy())
    mapping = RasterScale((source_h, source_w), (tracker.process_size, tracker.process_size))
    uv_source = mapping.process_to_source(np.asarray(tracks, dtype=np.float32))
    visible = np.asarray(visibility, dtype=np.bool_)
    inside = np.all(np.isfinite(uv_source), axis=-1) & (uv_source[..., 0] >= 0) & (uv_source[..., 0] < source_w) & (uv_source[..., 1] >= 0) & (uv_source[..., 1] < source_h)
    visible &= inside
    uv_source[~visible] = np.nan
    return uv_source.astype(np.float32), visible


def _build_sequence(decoded: Any, *, uv: np.ndarray, visibility: np.ndarray, xyz: np.ndarray, geometry_valid: np.ndarray, grid_size: int, cohort: str, query_pts_s: float) -> Any:
    return build_particle_sequence(
        decoded,
        track_ids=np.arange(uv.shape[1], dtype=np.int64),
        xyz=np.asarray(xyz, dtype=np.float32),
        uv=np.asarray(uv, dtype=np.float32),
        visibility=np.asarray(visibility, dtype=bool),
        geometry_validity=np.asarray(geometry_valid, dtype=bool),
        coordinate_system=CoordinateSystem(frame_name="first_camera_world", handedness=Handedness.RIGHT, axis_directions=("right", "down", "forward"), length_unit=LengthUnit.METER, camera_motion_compensated=True, normalization={}),
        lineage={"dataset": "ActivityForensics+Charades", "official_split": "train", "source_id": SOURCE_ID, "role": ROLE, "window_id": WINDOW_ID},
        provenance={"tracker": "online_bootstapir", "tracker_source_sha": density.TRACKER_SHA, "depth": "apple_depth_pro", "depth_source_sha": density.DEPTH_SHA, "pose": "open3d_rgbd_odometry", "process_size": PROCESS_SIZE, "query_grid_size": grid_size, "query_count": int(uv.shape[1]), "query_cohort": cohort, "cohort_start_s": float(query_pts_s), "causal_execution": True, "density_probe": "v7-04LAX-query-density-probe-v1"},
    )


def _support(sequence: Any, *, query_pts_s: float) -> tuple[dict[str, Any], list[dict[str, Any]], np.ndarray, np.ndarray, np.ndarray]:
    timestamps = np.asarray(sequence.timestamps_s, dtype=np.float64)
    history_indices = np.flatnonzero(timestamps < float(query_pts_s) + 0.5).astype(np.int64)
    components = rebuild_components_fast(sequence.xyz, sequence.geometry_validity, history_indices, config=COMPONENT_CONFIG)
    grouping = build_local_groups(sequence, history_indices, old_components=components, component_config=COMPONENT_CONFIG)
    row = {"window_id": WINDOW_ID, "grouping": {"history_array_indices": history_indices.tolist()}}
    units: list[dict[str, Any]] = []
    invalid: list[dict[str, Any]] = []
    s_values: list[np.ndarray] = []
    q_values: list[np.ndarray] = []
    intervals: np.ndarray | None = None
    for group in grouping["groups"]:
        group_id = int(group["local_group_id"])
        if not bool(group.get("retained")):
            invalid.append({"local_group_id": group_id, "reason": "GROUP_NOT_RETAINED"})
            continue
        unit = build_five_time_unit(sequence, window_id=WINDOW_ID, window_start_s=float(query_pts_s), member_slots=group["member_slots"], local_group_id=group_id, max_target_error_s=MAX_TARGET_ERROR_S)
        if unit.get("status") != "VALID":
            invalid.append({"local_group_id": group_id, "reason": str(unit.get("reason", "UNKNOWN")), "target_matches": unit.get("target_matches")})
            continue
        identity = {"local_group_id": group_id, "array_indices": unit["array_indices"], "track_ids": unit["track_ids"]}
        q, q_meta, changed = observation._q_for_unit(row, identity, unit, sequence, group)
        unit = {**unit, "raw_member_slots": [int(x) for x in group["member_slots"]], "raw_track_ids": [int(x) for x in group["track_ids"]], "q": q, "q_metadata": q_meta, "q_changed_from_history": bool(changed)}
        current_intervals = np.diff(np.asarray(unit["timestamps_s"], dtype=np.float64))
        if intervals is None:
            intervals = current_intervals
        elif not np.allclose(intervals, current_intervals, atol=1e-12, rtol=0):
            raise ValueError("TARGET_INTERVALS_DIFFER")
        units.append(unit)
        s_values.append(np.asarray(unit["states"], dtype=np.float64))
        q_values.append(np.asarray(q, dtype=np.float64))
    if intervals is None:
        intervals = np.empty((4,), dtype=np.float64)
    s_array = np.stack(s_values, axis=0) if s_values else np.empty((0, 5, 4), dtype=np.float64)
    q_array = np.stack(q_values, axis=0) if q_values else np.empty((0, 5, 4), dtype=np.float64)
    summary = {"history_array_indices": history_indices.tolist(), "history_frame_indices": [int(sequence.frame_indices[i]) for i in history_indices], "target_frame_indices": list(TARGET_FRAMES), "target_timestamps_s": [float(sequence.timestamps_s[list(sequence.frame_indices).index(frame)]) for frame in TARGET_FRAMES if frame in sequence.frame_indices], "intervals_s": intervals.tolist(), "component_count": len(components), "history_group_count": len(grouping["groups"]), "retained_group_count": int(grouping["retained_group_count"]), "valid_unit_count": len(units), "invalid_group_or_unit_count": len(invalid), "invalid_reason_counts": dict(Counter(str(item["reason"]) for item in invalid)), "q_changed_unit_count": int(sum(item["q_changed_from_history"] for item in units)), "q_unchanged_unit_count": int(sum(not item["q_changed_from_history"] for item in units)), "grouping": grouping}
    return summary, units, s_array, q_array, intervals


def _feature_manifest(output: Path, condition: str, summary: Mapping[str, Any], units: Sequence[Mapping[str, Any]], s: np.ndarray, q: np.ndarray, intervals: np.ndarray) -> list[dict[str, Any]]:
    relative = Path("features") / f"{condition}.npz"
    path = output / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, s=np.asarray(s, dtype=np.float32), q=np.asarray(q, dtype=np.float32), intervals=np.asarray(intervals, dtype=np.float64))
    identities = []
    for index, unit in enumerate(units):
        identities.append({"unit_index": index, "local_group_id": int(unit["local_group_id"]), "member_slots": unit["member_slots"], "track_ids": unit["track_ids"], "raw_member_slots": unit["raw_member_slots"], "raw_track_ids": unit["raw_track_ids"], "pair_ids": unit["pair_ids"], "history_scale": float(unit["history_scale"]), "history_scale_pair_time_count": int(unit["history_scale_pair_time_count"]), "array_indices": unit["array_indices"], "frame_indices": unit["frame_indices"], "timestamps_s": unit["timestamps_s"], "target_times_s": unit["target_times_s"], "match_errors_s": unit["match_errors_s"], "q_metadata": unit["q_metadata"]})
    manifest = [{"window_id": WINDOW_ID, "source_id": SOURCE_ID, "role": ROLE, "label": 1, "annotation_category": "FAKE_MANIPULATION_DIAGNOSTIC_ONLY", "condition": condition, "input_path": str(relative), "s_shape": list(s.shape), "q_shape": list(q.shape), "intervals_s": intervals.tolist(), "matched_unit_count": len(units), "original_unit_count": len(units), "unit_identities": identities}]
    return manifest


def _model_readout(output: Path, manifests: Mapping[str, Sequence[Mapping[str, Any]]]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    import torch
    records = json.loads((MODEL_ROOT / "models/fold_models.json").read_text(encoding="utf-8"))["records"]
    records = [record for record in records if record.get("condition") in MODEL_CONDITIONS and int(record.get("seed", -1)) in SEEDS and record.get("status") == "TRAIN_COMPLETE"]
    if len(records) != 6:
        raise ValueError(f"FROZEN_MODEL_RECORDS_INCOMPLETE:{len(records)}")
    sample_manifest = json.loads((MODEL_ROOT / "inputs/feature_manifest.json").read_text(encoding="utf-8"))
    sample_scores = list(csv.DictReader((MODEL_ROOT / "scores/validation_window_scores.csv").open(newline="", encoding="utf-8")))
    sample_row = next((row for row in sample_scores if row.get("source_id") == "00ZCA"), None)
    if sample_row is None:
        raise ValueError("FROZEN_MODEL_SAMPLE_MISSING")
    reproduction: list[dict[str, Any]] = []
    for condition in MODEL_CONDITIONS:
        for seed in SEEDS:
            record = next(item for item in records if item["condition"] == condition and int(item["seed"]) == seed)
            ss = _standardizer_record(record["standardization"]["s"])
            sq = _standardizer_record(record["standardization"]["q"])
            batch = observation._batch(MODEL_ROOT, sample_manifest, [{"window_id": sample_row["window_id"], "source_id": sample_row["source_id"], "label": int(sample_row["label"])}], ss, sq, condition)
            model = observation._model_load(condition, record, "cpu")
            score = float(model(torch.as_tensor(batch["states"], dtype=torch.float32), torch.as_tensor(batch["intervals"], dtype=torch.float32)).detach().cpu().numpy().mean())
            saved = float(sample_row[f"{condition}_seed_{seed}"])
            reproduction.append({"condition": condition, "seed": seed, "sample_window_id": sample_row["window_id"], "saved_window_logit": saved, "reproduced_window_logit": score, "max_abs_error": abs(score - saved), "passed": abs(score - saved) <= TOLERANCE})
    if not all(item["passed"] for item in reproduction):
        raise ValueError("FROZEN_MODEL_REPRODUCTION_FAILED")
    rows: list[dict[str, Any]] = []
    window: dict[str, Any] = {"window_id": WINDOW_ID, "source_id": SOURCE_ID, "role": ROLE, "conditions": {}}
    for density_name, manifest in manifests.items():
        window["conditions"][density_name] = {}
        for condition in MODEL_CONDITIONS:
            for seed in SEEDS:
                record = next(item for item in records if item["condition"] == condition and int(item["seed"]) == seed)
                ss = _standardizer_record(record["standardization"]["s"])
                sq = _standardizer_record(record["standardization"]["q"])
                batch = observation._batch(output, manifest, [{"window_id": WINDOW_ID, "source_id": SOURCE_ID, "label": 1}], ss, sq, condition)
                model = observation._model_load(condition, record, "cpu")
                local = model(torch.as_tensor(batch["states"], dtype=torch.float32), torch.as_tensor(batch["intervals"], dtype=torch.float32)).detach().cpu().numpy().astype(np.float64)
                if local.shape != (int(manifest[0]["matched_unit_count"]),) or not np.all(np.isfinite(local)):
                    raise ValueError(f"NONFINITE_MODEL_OUTPUT:{density_name}:{condition}:{seed}")
                window["conditions"][density_name].setdefault(condition, {})[str(seed)] = float(np.mean(local))
                for identity, score in zip(manifest[0]["unit_identities"], local.tolist()):
                    rows.append({"density": density_name, "condition": condition, "seed": seed, "local_group_id": int(identity["local_group_id"]), "track_ids": json.dumps(identity["track_ids"], separators=(",", ":")), "pair_count": len(identity["pair_ids"]), "model_local_logit": float(score), "unit_mean_contribution": float(score / len(local))})
    window["threshold_logit"] = 0.0
    window["mean_logit"] = {
        density_name: {
            condition: float(np.mean([window["conditions"][density_name][condition][str(seed)] for seed in SEEDS]))
            for condition in MODEL_CONDITIONS
        }
        for density_name in manifests
    }
    window["threshold_ge_zero"] = {
        density_name: {
            condition: {str(seed): bool(window["conditions"][density_name][condition][str(seed)] >= 0.0) for seed in SEEDS}
            for condition in MODEL_CONDITIONS
        }
        for density_name in manifests
    }
    atomic_json(output / "model_reproduction.json", reproduction)
    write_csv(output / "unit_scores.csv", rows)
    atomic_json(output / "window_scores.json", window)
    return window, rows


def _roi_annotations() -> dict[int, dict[str, Any]]:
    path = OLD_CASE / "roi_evidence_review_v1/roi_annotations.json"
    if not path.is_file():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    # The review package keeps the validated envelope around the user input;
    # older versions put entries at the top level.  Read only the validated
    # input and never treat SKIP/UNCERTAIN/PENDING as confirmed geometry.
    rows = payload.get("entries")
    if rows is None and isinstance(payload.get("input"), Mapping):
        rows = payload["input"].get("entries", [])
    if rows is None and isinstance(payload, list):
        rows = payload
    if not isinstance(rows, list):
        rows = []
    return {int(item["source_frame_index"]): item for item in rows if str(item.get("human_confirmation_status", "")).upper() == "CONFIRMED"}


def _roi_counts(sequence: Any, summary: Mapping[str, Any], units: Sequence[Mapping[str, Any]], manifest: Sequence[Mapping[str, Any]], density_name: str, frame: int, rect: Mapping[str, Any]) -> dict[str, Any]:
    frames = np.asarray(sequence.frame_indices, dtype=np.int64)
    if frame not in frames:
        return {"status": "FRAME_MISSING", "density": density_name, "frame_index": frame}
    t = int(np.flatnonzero(frames == frame)[0])
    x0, y0, x1, y1 = (float(rect[key]) for key in ("x_min", "y_min", "x_max", "y_max"))
    uv = np.asarray(sequence.uv[t], dtype=np.float64)
    finite = np.all(np.isfinite(uv), axis=1)
    inside = finite & (uv[:, 0] >= x0) & (uv[:, 0] <= x1) & (uv[:, 1] >= y0) & (uv[:, 1] <= y1)
    vis = inside & np.asarray(sequence.visibility[t], dtype=bool)
    geo = inside & np.asarray(sequence.geometry_validity[t], dtype=bool)
    original_ids = {int(row["query_id"]) for row in manifest if row["membership"] == "original"}
    original_mask = np.asarray([index in original_ids for index in range(uv.shape[0])], dtype=bool)
    unit_members = sorted({int(member) for unit in units for member in unit["member_slots"]})
    common = sorted(set(unit_members) & set(np.flatnonzero(inside).tolist()))
    pairs_both = 0
    pairs_one = 0
    pair_total = 0
    pair_both_original = 0
    pair_mixed = 0
    pair_both_added = 0
    for unit in units:
        for left, right in unit["pair_indices"]:
            pair_total += 1
            left_in = bool(inside[int(left)]); right_in = bool(inside[int(right)])
            if left_in and right_in:
                pairs_both += 1
            elif left_in or right_in:
                pairs_one += 1
            left_original = int(left) in original_ids; right_original = int(right) in original_ids
            if left_original and right_original: pair_both_original += 1
            elif left_original or right_original: pair_mixed += 1
            else: pair_both_added += 1
    return {"status": "CONFIRMED_ROI", "density": density_name, "frame_index": frame, "pts_s": float(sequence.timestamps_s[t]), "roi": [x0, y0, x1, y1], "visible_points_in_roi": int(np.sum(vis)), "visible_original_points_in_roi": int(np.sum(vis & original_mask)), "visible_added_points_in_roi": int(np.sum(vis & ~original_mask)), "geometry_valid_points_in_roi": int(np.sum(geo)), "geometry_valid_original_points_in_roi": int(np.sum(geo & original_mask)), "geometry_valid_added_points_in_roi": int(np.sum(geo & ~original_mask)), "common_member_points_in_roi": len(common), "common_member_point_ids": common, "valid_pair_total": pair_total, "valid_pairs_both_endpoints_in_roi": pairs_both, "valid_pairs_one_endpoint_in_roi": pairs_one, "valid_pairs_both_original": pair_both_original, "valid_pairs_mixed_original_added": pair_mixed, "valid_pairs_both_added": pair_both_added, "visible_points_outside_roi": int(np.sum(np.asarray(sequence.visibility[t], dtype=bool) & ~inside)), "geometry_valid_points_outside_roi": int(np.sum(np.asarray(sequence.geometry_validity[t], dtype=bool) & ~inside))}


def _draw_overlays(output: Path, decoded: Any, sequences: Mapping[str, Any], manifests: Mapping[str, Sequence[Mapping[str, Any]]], units: Mapping[str, Sequence[Mapping[str, Any]]]) -> list[str]:
    from PIL import Image, ImageDraw
    targets = list(TARGET_FRAMES)
    written: list[str] = []
    frame_to_pos = {int(frame): i for i, frame in enumerate(decoded.frame_indices)}
    for frame in targets:
        pos = frame_to_pos[frame]
        base = Image.fromarray(np.asarray(decoded.frames[pos].rgb, dtype=np.uint8), mode="RGB")
        clean = output / "overlays" / f"frame_{frame}_clean.png"; clean.parent.mkdir(parents=True, exist_ok=True); base.save(clean); written.append(str(clean))
        for name in ("R17", "R33"):
            image = base.copy(); draw = ImageDraw.Draw(image)
            seq = sequences[name]; uv = np.asarray(seq.uv[pos]); visible = np.asarray(seq.visibility[pos], dtype=bool); geom = np.asarray(seq.geometry_validity[pos], dtype=bool)
            original_ids = {int(item["query_id"]) for item in manifests["R33"] if item["membership"] == "original"}
            common_ids = {int(member) for unit in units[name] for member in unit["member_slots"]}
            for idx, point in enumerate(uv):
                if not visible[idx] or not np.all(np.isfinite(point)): continue
                radius = 2 if name == "R17" else 1
                if name == "R33": color = (0, 210, 255) if idx in original_ids else (255, 155, 0)
                else: color = (0, 210, 255)
                if idx in common_ids and geom[idx]: color = (0, 255, 80) if name == "R17" else ((0, 255, 80) if idx in original_ids else (255, 0, 220))
                x, y = float(point[0]), float(point[1]); draw.ellipse((x-radius, y-radius, x+radius, y+radius), fill=color)
            path = output / "overlays" / f"frame_{frame}_{name}.png"; image.save(path); written.append(str(path))
    return written


def _compare_original_queries(r17_uv: np.ndarray, r17_visibility: np.ndarray, r17_geometry: np.ndarray, r33_uv: np.ndarray, r33_visibility: np.ndarray, r33_geometry: np.ndarray, manifest: Sequence[Mapping[str, Any]], frame_indices: Sequence[int]) -> list[dict[str, Any]]:
    original = [int(item["query_id"]) for item in manifest if item["membership"] == "original"]
    old = [int(item["original_query_id"]) for item in manifest if item["membership"] == "original"]
    rows: list[dict[str, Any]] = []
    for t, frame in enumerate(frame_indices):
        both = np.asarray(r17_visibility[t, old], dtype=bool) & np.asarray(r33_visibility[t, original], dtype=bool) & np.all(np.isfinite(r17_uv[t, old]), axis=1) & np.all(np.isfinite(r33_uv[t, original]), axis=1)
        diff = np.linalg.norm(np.asarray(r17_uv[t, old], dtype=np.float64) - np.asarray(r33_uv[t, original], dtype=np.float64), axis=1)
        values = diff[both]
        rows.append({"frame_index": int(frame), "r17_original_visible": int(np.sum(r17_visibility[t, old])), "r33_mapped_original_visible": int(np.sum(r33_visibility[t, original])), "visibility_agreement": float(np.mean(np.asarray(r17_visibility[t, old]) == np.asarray(r33_visibility[t, original]))), "both_visible_count": int(np.sum(both)), "uv_displacement_median_px": float(np.median(values)) if values.size else None, "uv_displacement_p95_px": float(np.percentile(values, 95)) if values.size else None, "uv_displacement_max_px": float(np.max(values)) if values.size else None, "r17_geometry_valid": int(np.sum(r17_geometry[t, old])), "r33_mapped_original_geometry_valid": int(np.sum(r33_geometry[t, original])), "geometry_validity_delta": int(np.sum(r33_geometry[t, original])) - int(np.sum(r17_geometry[t, old]))})
    return rows


def _roi_rows_from_existing(output: Path, manifests: Mapping[str, Sequence[Mapping[str, Any]]]) -> list[dict[str, Any]]:
    """Recreate the small ROI table without rerunning video or geometry work."""
    from types import SimpleNamespace

    annotations = _roi_annotations()
    query_rows = json.loads((output / "query_manifest.json").read_text(encoding="utf-8"))["rows"]
    r17_rows = [{"query_id": index, "membership": "original", "original_query_id": index} for index in range(289)]
    rows: list[dict[str, Any]] = []
    for density_name, geometry_name in (("R17", "R17_shared_geometry.npz"), ("R33", "R33_geometry.npz")):
        with np.load(output / geometry_name, allow_pickle=False) as arrays:
            sequence = SimpleNamespace(
                frame_indices=np.asarray(arrays["frame_indices"], dtype=np.int64),
                timestamps_s=np.asarray(arrays["timestamps_s"], dtype=np.float64),
                uv=np.asarray(arrays["uv"], dtype=np.float32),
                visibility=np.asarray(arrays["visibility"], dtype=bool),
                geometry_validity=np.asarray(arrays["geometry_validity"], dtype=bool),
            )
            units = []
            for identity in manifests[density_name][0]["unit_identities"]:
                units.append({
                    "member_slots": [int(value) for value in identity["member_slots"]],
                    "pair_indices": [[int(pair[0]), int(pair[1])] for pair in identity["pair_ids"]],
                })
            for frame, annotation in sorted(annotations.items()):
                rect = {key: annotation[key] for key in ("x_min", "y_min", "x_max", "y_max")}
                rows.append(_roi_counts(sequence, {}, units, r17_rows if density_name == "R17" else query_rows, density_name, frame, rect))
    return rows


def _frame_rows_from_existing(output: Path, manifests: Mapping[str, Sequence[Mapping[str, Any]]]) -> list[dict[str, Any]]:
    """Return target-frame visibility/geometry counts for both densities."""
    query_rows = json.loads((output / "query_manifest.json").read_text(encoding="utf-8"))["rows"]
    original_ids = {int(row["query_id"]) for row in query_rows if row["membership"] == "original"}
    rows: list[dict[str, Any]] = []
    for density_name, geometry_name, count in (("R17", "R17_shared_geometry.npz", 289), ("R33", "R33_geometry.npz", 1089)):
        with np.load(output / geometry_name, allow_pickle=False) as arrays:
            frames = np.asarray(arrays["frame_indices"], dtype=np.int64)
            timestamps = np.asarray(arrays["timestamps_s"], dtype=np.float64)
            visibility = np.asarray(arrays["visibility"], dtype=bool)
            geometry = np.asarray(arrays["geometry_validity"], dtype=bool)
            for frame in TARGET_FRAMES:
                positions = np.flatnonzero(frames == int(frame))
                if positions.size != 1:
                    rows.append({"density": density_name, "frame_index": int(frame), "status": "FRAME_MISSING", "query_count": count})
                    continue
                t = int(positions[0])
                original = np.ones((count,), dtype=bool) if density_name == "R17" else np.asarray([idx in original_ids for idx in range(count)], dtype=bool)
                rows.append({
                    "density": density_name,
                    "frame_index": int(frame),
                    "pts_s": float(timestamps[t]),
                    "status": "AVAILABLE",
                    "query_count": count,
                    "visible_count": int(np.sum(visibility[t])),
                    "visible_fraction": float(np.mean(visibility[t])),
                    "geometry_valid_count": int(np.sum(geometry[t])),
                    "geometry_valid_fraction": float(np.mean(geometry[t])),
                    "original_query_visible_count": int(np.sum(visibility[t] & original)),
                    "added_query_visible_count": int(np.sum(visibility[t] & ~original)),
                    "original_query_geometry_valid_count": int(np.sum(geometry[t] & original)),
                    "added_query_geometry_valid_count": int(np.sum(geometry[t] & ~original)),
                })
    return rows


def _support_density_stats(density_name: str, manifest: Sequence[Mapping[str, Any]], original_ids: set[int] | None = None) -> dict[str, Any]:
    identities = list(manifest[0]["unit_identities"])
    if original_ids is None:
        original_ids = set(range(289))
    pairs = [tuple(map(int, pair)) for identity in identities for pair in identity["pair_ids"]]
    return {
        "unique_common_member_count": len({int(member) for identity in identities for member in identity["member_slots"]}),
        "member_count_min": min((len(identity["member_slots"]) for identity in identities), default=0),
        "member_count_median": float(np.median([len(identity["member_slots"]) for identity in identities])) if identities else None,
        "member_count_max": max((len(identity["member_slots"]) for identity in identities), default=0),
        "valid_pair_total": len(pairs),
        "valid_pairs_both_original": sum(int(left in original_ids and right in original_ids) for left, right in pairs),
        "valid_pairs_mixed_original_added": sum(int((left in original_ids) != (right in original_ids)) for left, right in pairs),
        "valid_pairs_both_added": sum(int(left not in original_ids and right not in original_ids) for left, right in pairs),
    }


def _finish_existing(output: Path) -> dict[str, Any]:
    """Finish model readout/report from verified artifacts left by a prior run.

    The first invocation completed R33 tracking, shared geometry, and support,
    then failed only at frozen-model loading.  This path validates those
    artifacts and resumes at readout, so a model API fix cannot accidentally
    rerun the expensive frontend.
    """
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    previous_status = None
    final_path = output / "final_status.json"
    if final_path.is_file():
        previous_status = json.loads(final_path.read_text(encoding="utf-8"))
        if previous_status.get("status") == "FAILED":
            atomic_json(output / "previous_failure.json", previous_status)
    atomic_json(output / "state.json", {"status": "RUNNING", "stage": "resume_model_readout", "pid": os.getpid(), "updated_unix": time.time(), "git_head": git_head()})
    try:
        manifests = json.loads((output / "feature_manifest.json").read_text(encoding="utf-8"))
        protocol = json.loads((output / "protocol.json").read_text(encoding="utf-8"))
        expected_frames = np.asarray(protocol["case"]["frame_indices"], dtype=np.int64)
        required = ("R33_requery.npz", "R17_shared_geometry.npz", "R33_geometry.npz", "R17_support.json", "R33_support.json", "feature_manifest.json")
        missing = [name for name in required if not (output / name).is_file()]
        if missing:
            raise FileNotFoundError("RESUME_ARTIFACTS_MISSING:" + ",".join(missing))
        with np.load(output / "R17_shared_geometry.npz", allow_pickle=False) as r17, np.load(output / "R33_requery.npz", allow_pickle=False) as r33q, np.load(output / "R33_geometry.npz", allow_pickle=False) as r33:
            for label, arrays, count in (("R17", r17, 289), ("R33_REQUERY", r33q, 1089), ("R33", r33, 1089)):
                if not np.array_equal(np.asarray(arrays["frame_indices"], dtype=np.int64), expected_frames):
                    raise ValueError(f"RESUME_FRAME_IDENTITY_MISMATCH:{label}")
                if np.asarray(arrays["uv"]).shape[1:] != (count, 2):
                    raise ValueError(f"RESUME_QUERY_SHAPE_MISMATCH:{label}")
            if not np.array_equal(r33q["frame_indices"], r33["frame_indices"]) or not np.allclose(r33q["timestamps_s"], r33["timestamps_s"], rtol=0, atol=1e-12):
                raise ValueError("RESUME_R33_TRACK_GEOMETRY_TIME_MISMATCH")
            if not np.allclose(r33q["uv"], r33["uv"], rtol=0, atol=0, equal_nan=True) or not np.array_equal(r33q["visibility"], r33["visibility"]):
                raise ValueError("RESUME_R33_TRACK_GEOMETRY_UV_MISMATCH")
            timestamps = np.asarray(r17["timestamps_s"], dtype=np.float64)
        for density_name in ("R17", "R33"):
            manifest = manifests[density_name][0]
            if list(manifest["s_shape"])[1:] != [5, 4] or list(manifest["q_shape"])[1:] != [5, 4]:
                raise ValueError(f"RESUME_FEATURE_SHAPE_MISMATCH:{density_name}")
            if int(manifest["matched_unit_count"]) != int(manifest["s_shape"][0]):
                raise ValueError(f"RESUME_FEATURE_UNIT_COUNT_MISMATCH:{density_name}")

        window_scores, unit_scores = _model_readout(output, manifests)
        supports = {name: json.loads((output / f"{name}_support.json").read_text(encoding="utf-8")) for name in ("R17", "R33")}
        original_compare = list(csv.DictReader((output / "original_query_consistency.csv").open(newline="", encoding="utf-8")))
        roi_rows = _roi_rows_from_existing(output, manifests)
        frame_rows = _frame_rows_from_existing(output, manifests)
        query_rows = json.loads((output / "query_manifest.json").read_text(encoding="utf-8"))["rows"]
        original_ids = {int(row["query_id"]) for row in query_rows if row["membership"] == "original"}
        stats_by_density = {name: _support_density_stats(name, manifests[name], set(range(289)) if name == "R17" else original_ids) for name in ("R17", "R33")}
        write_csv(output / "roi_support_comparison.csv", roi_rows)
        support_rows = []
        for name in ("R17", "R33"):
            support_rows.append({
                "density": name,
                "query_count": 289 if name == "R17" else 1089,
                "added_query_count": 0 if name == "R17" else 800,
                "history_group_count": supports[name]["history_group_count"],
                "retained_group_count": supports[name]["retained_group_count"],
                "valid_unit_count": supports[name]["valid_unit_count"],
                "invalid_group_or_unit_count": supports[name]["invalid_group_or_unit_count"],
                "invalid_reason_counts": json.dumps(supports[name]["invalid_reason_counts"], sort_keys=True),
                "q_changed_unit_count": supports[name]["q_changed_unit_count"],
                "component_count": supports[name]["component_count"],
                **stats_by_density[name],
            })
        write_csv(output / "support_comparison.csv", support_rows)
        write_csv(output / "frame_support.csv", frame_rows)
        geometry_metadata = json.loads((output / "geometry_metadata.json").read_text(encoding="utf-8"))
        fixed_focal = float(geometry_metadata["fixed_focal_px"])
        report_lines = [
            "# 04LAX R 查询密度匹配对照",
            "",
            "状态：`COMPLETE_WITH_SHARED_GEOMETRY`。这是单案例观测支撑诊断，不是新的检测器训练、AUROC 或空间真值评价。",
            "",
            "## 结果先行",
            f"- R17 使用旧 R_requery 的 289 点 UV/visibility；R33 在同一源视频 frame {QUERY_FRAME} / PTS {timestamps[0]:.12f} 重新查询 1089 点，其中新增 800 点。33 轴由实际 17 轴逐邻接插入中点构造，未使用另一套 linspace。",
            f"- R17/R33 共用一次 Depth Pro、固定首帧 focal={fixed_focal:.6f}、Open3D pose 与世界坐标反投影；R17 的 `R17_shared_geometry.npz` 是公平比较输入，旧 R_geometry 仅作历史参照。",
            "- 本次从已完成的跟踪、几何和结构缓存恢复，首轮失败仅发生在冻结模型标准化读取接口；该失败证据保存在 `previous_failure.json`。",
            "",
            "| condition | history groups | retained groups | valid five-time units | invalid reasons | Q changed units |",
            "|---|---:|---:|---:|---|---:|",
        ]
        for name in ("R17", "R33"):
            report_lines.append(f"| {name} | {supports[name]['history_group_count']} | {supports[name]['retained_group_count']} | {supports[name]['valid_unit_count']} | `{supports[name]['invalid_reason_counts']}` | {supports[name]['q_changed_unit_count']} |")
        report_lines += ["", "## 有效 unit 与 pair 分布", "", "| density | unique common members | unit members min/median/max | valid pairs | original/original | mixed | added/added |", "|---|---:|---|---:|---:|---:|---:|"]
        for name in ("R17", "R33"):
            stats = stats_by_density[name]
            report_lines.append(f"| {name} | {stats['unique_common_member_count']} | {stats['member_count_min']}/{stats['member_count_median']:.1f}/{stats['member_count_max']} | {stats['valid_pair_total']} | {stats['valid_pairs_both_original']} | {stats['valid_pairs_mixed_original_added']} | {stats['valid_pairs_both_added']} |")
        r17_so = window_scores["conditions"]["R17"]["STRUCTURE_ONLY"]
        r33_so = window_scores["conditions"]["R33"]["STRUCTURE_ONLY"]
        r17_ss = window_scores["conditions"]["R17"]["STRUCTURE_SUPPORT"]
        r33_ss = window_scores["conditions"]["R33"]["STRUCTURE_SUPPORT"]
        report_lines += ["", "## 结论与边界", "", f"- 原 289 个查询在 R33 的 1089 个查询中逐坐标嵌套；联合跟踪下原点的 UV 位移最大为 {max(float(row['uv_displacement_max_px']) for row in original_compare):.6f} px，visibility 一致率仅在 frame 497 为 0.9965，其余记录为 1.0。该变化是联合跟踪事实，不是正确性证明。", f"- R33 使有效五时刻 unit 从 {supports['R17']['valid_unit_count']} 增至 {supports['R33']['valid_unit_count']}，Q 发生变化的 unit 从 {supports['R17']['q_changed_unit_count']} 增至 {supports['R33']['q_changed_unit_count']}；新增 pair 中 {stats_by_density['R33']['valid_pairs_mixed_original_added']} 条为一新一旧、{stats_by_density['R33']['valid_pairs_both_added']} 条为两端新增，因此不能把全部 pair 增长视为独立新信息。", f"- 已确认粗 ROI 的目标帧中，R33 的可见/几何有效点数为 824–1045/帧，R17 为 221–276/帧；这是同一矩形下的观测计数，不是伪造区域召回。", f"- 冻结读出（R33−R17）按 seed：STRUCTURE_ONLY = {r33_so['20260909'] - r17_so['20260909']:.9f}, {r33_so['20260910'] - r17_so['20260910']:.9f}, {r33_so['20260911'] - r17_so['20260911']:.9f}；STRUCTURE_SUPPORT = {r33_ss['20260909'] - r17_ss['20260909']:.9f}, {r33_ss['20260910'] - r17_ss['20260910']:.9f}, {r33_ss['20260911'] - r17_ss['20260911']:.9f}。这是同一冻结模型上的单案例读出，不是检测性能比较。", "- 因只有一个 fake 案例、R33 改变了分组和聚合组成、且没有像素级空间真值，本轮只能判定为 `DENSITY_ADDS_MEASURABLE_SUPPORT_WITH_COMPOSITION_CHANGE` 的观测诊断；不能宣称空间定位或检测能力提升。"]
        report_lines += ["", "## 目标帧覆盖", "", "| density | frame | PTS | query count | visible | geometry-valid | original visible | added visible | original geometry | added geometry |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for row in frame_rows:
            if row.get("status") != "AVAILABLE":
                report_lines.append(f"| {row['density']} | {row['frame_index']} | — | {row['query_count']} | — | — | — | — | — | — |")
            else:
                report_lines.append(f"| {row['density']} | {row['frame_index']} | {row['pts_s']:.12f} | {row['query_count']} | {row['visible_count']} ({row['visible_fraction']:.3f}) | {row['geometry_valid_count']} ({row['geometry_valid_fraction']:.3f}) | {row['original_query_visible_count']} | {row['added_query_visible_count']} | {row['original_query_geometry_valid_count']} | {row['added_query_geometry_valid_count']} |")
        report_lines += ["", "## 原查询一致性", "", "| frame | mapped original visible R17/R33 | visibility agreement | both visible | UV median / p95 / max px | geometry valid delta R33-R17 |", "|---:|---:|---:|---:|---|---:|"]
        for row in original_compare:
            report_lines.append(f"| {row['frame_index']} | {row['r17_original_visible']}/{row['r33_mapped_original_visible']} | {float(row['visibility_agreement']):.4f} | {row['both_visible_count']} | {row['uv_displacement_median_px']}/{row['uv_displacement_p95_px']}/{row['uv_displacement_max_px']} | {row['geometry_validity_delta']} |")
        report_lines += ["", "## ROI 对照", "", "仅使用 `roi_evidence_review_v1/roi_annotations.json` 中 `CONFIRMED` 的逐帧矩形；frame 488 的 `SKIP` 未使用，frame 487 在 R 缓存范围外并明确记为 FRAME_MISSING。矩形是主体粗框，不是精确伪造 mask。", "", "| frame | density | visible ROI (orig/add) | geometry ROI (orig/add) | common-member points | pair both / one endpoint | original/mixed/added pairs |", "|---:|---|---:|---:|---:|---:|---|"]
        for row in roi_rows:
            if row.get("status") == "FRAME_MISSING":
                report_lines.append(f"| {row['frame_index']} | {row['density']} | — | — | — | — | FRAME_MISSING |")
            else:
                report_lines.append(f"| {row['frame_index']} | {row['density']} | {row['visible_points_in_roi']} ({row['visible_original_points_in_roi']}/{row['visible_added_points_in_roi']}) | {row['geometry_valid_points_in_roi']} ({row['geometry_valid_original_points_in_roi']}/{row['geometry_valid_added_points_in_roi']}) | {row['common_member_points_in_roi']} | {row['valid_pairs_both_endpoints_in_roi']} / {row['valid_pairs_one_endpoint_in_roi']} | {row['valid_pairs_both_original']} / {row['valid_pairs_mixed_original_added']} / {row['valid_pairs_both_added']} |")
        report_lines += ["", "## Frozen-model readout", "", "模型在原 R17 分布上训练；本读出不重新拟合标准化、不训练、不校准。R17/R33 的局部组不是物理一一对应，分数变化不能单独证明检测收益。局部 logit 是窗口监督下的中间读出，不是局部伪造概率；固定诊断阈值为 logit ≥ 0，未搜索阈值。", "", "| density | condition | seed 20260909 | seed 20260910 | seed 20260911 | mean |", "|---|---|---:|---:|---:|---:|"]
        for density_name in ("R17", "R33"):
            for condition in MODEL_CONDITIONS:
                values = window_scores["conditions"][density_name][condition]
                mean = float(np.mean([values[str(seed)] for seed in SEEDS]))
                report_lines.append(f"| {density_name} | {condition} | {values[str(SEEDS[0])]:.9f} | {values[str(SEEDS[1])]:.9f} | {values[str(SEEDS[2])]:.9f} | {mean:.9f} |")
        report_lines += ["", "## 限制", "", "- R33 的原 289 查询在联合 1089 查询中可能改变跟踪输出；mapped-original 差异只是事实记录，不能将其全部解释成 800 条独立轨迹。", "- 点数增加会自然增加候选 pair；有效 unit/关系增加不等于同等比例的新独立信息，也不等于空间定位真值。", "- 只分析一个 04LAX fake 案例；04LAX 的冻结模型读出属于开发诊断，不是 sealed-test。", "", "## 产物", "", "`protocol.json`、`query_manifest.json/csv`、`R33_requery.npz`、`R17_shared_geometry.npz`、`R33_geometry.npz`、`support_comparison.csv`、`frame_support.csv`、`original_query_consistency.csv`、`roi_support_comparison.csv`、`window_scores.json`、`unit_scores.csv`、`model_reproduction.json`、`overlays/`。"]
        (output / "report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")
        result = {"status": "COMPLETE_WITH_SHARED_GEOMETRY", "window_id": WINDOW_ID, "conditions": ["R17", "R33"], "query_counts": {"R17": 289, "R33": 1089, "R33_added": 800}, "support": {name: {key: supports[name][key] for key in ("history_group_count", "retained_group_count", "valid_unit_count", "invalid_group_or_unit_count", "q_changed_unit_count", "component_count")} for name in supports}, "model_readout": "AVAILABLE", "resumed_from_existing": True, "elapsed_s": time.perf_counter() - started, "git_head": git_head()}
        atomic_json(output / "final_status.json", result)
        atomic_json(output / "state.json", {"status": "COMPLETE", "stage": "report", "pid": os.getpid(), "updated_unix": time.time(), "git_head": git_head(), "elapsed_s": time.perf_counter() - started})
        return result
    except Exception as exc:
        error = {"type": type(exc).__name__, "message": str(exc)}
        atomic_json(output / "final_status.json", {"status": "FAILED", "error": error, "traceback": __import__("traceback").format_exc(), "elapsed_s": time.perf_counter() - started, "git_head": git_head()})
        atomic_json(output / "state.json", {"status": "FAILED", "stage": "resume_error", "error": error, "updated_unix": time.time(), "git_head": git_head()})
        raise


def _write_protocol(output: Path, case: Mapping[str, Any], query: Mapping[str, Any], frame_indices: Sequence[int], timestamps: Sequence[float]) -> None:
    atomic_json(output / "protocol.json", {"protocol_id": "v7-04LAX-query-density-probe-v1", "git_head": git_head(), "question": "Whether nested 33x33 queries add sustained local structural support relative to the audited 17x17 R query", "case": {"source_id": SOURCE_ID, "role": ROLE, "window_id": WINDOW_ID, "video": case.get("video"), "query_start_frame": QUERY_FRAME, "query_start_pts_s": float(timestamps[0]), "frame_indices": list(map(int, frame_indices)), "history_frame_indices": list(map(int, frame_indices[:15])), "target_frame_indices": list(map(int, TARGET_FRAMES))}, "conditions": {"R17": {"query_grid": "audited 17x17 tracker grid", "query_count": 289, "source": "existing R_requery UV/visibility"}, "R33": {"query_grid": "midpoint-inserted nested axis from actual R17 process axis", "query_count": 1089, "added_query_count": 800, "tracker_run": True}}, "fixed": {"tracker": "online_bootstapir", "tracker_source_sha": density.TRACKER_SHA, "checkpoint": str(density.TAPNET_CHECKPOINT), "process_size": PROCESS_SIZE, "query_chunk_size": 64, "depth": "apple_depth_pro", "depth_source_sha": density.DEPTH_SHA, "pose": "open3d_rgbd_odometry", "same_shared_depth_pose_for_R17_R33": True, "target_offsets_s": list(TARGET_OFFSETS_S), "target_tolerance_s": float(MAX_TARGET_ERROR_S), "component_config": {"max_initial_distance": COMPONENT_CONFIG.max_initial_distance, "max_relative_change": COMPONENT_CONFIG.max_relative_change, "minimum_overlap": COMPONENT_CONFIG.minimum_overlap, "minimum_size": COMPONENT_CONFIG.minimum_size}, "minimum_common_members": 3, "q_definition": "raw historical group member denominator; target visibility/geometry fractions minus last history reference", "missing_policy": "explicit invalid reason; no zero fill/interpolation"}, "query_manifest": "query_manifest.json", "bounds": ["single 04LAX fake case", "no training", "no AUROC/AP", "not spatial ground truth", "not sealed test"]})


def run(output: Path = OUTPUT_ROOT) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    atomic_json(output / "state.json", {"status": "RUNNING", "stage": "validate", "pid": os.getpid(), "started_unix": time.time(), "git_head": git_head()})
    try:
        case, old_arrays, old_structure, case_info = _validate_case()
        frame_indices = np.asarray(old_arrays["frame_indices"], dtype=np.int64)
        timestamps = np.asarray(old_arrays["timestamps_s"], dtype=np.float64)
        query_manifest_rows = case_info["query_manifest"]
        atomic_json(output / "query_manifest.json", {"query": case_info["query"], "rows": query_manifest_rows})
        _write_protocol(output, case, case_info["query"], frame_indices, timestamps)
        write_csv(output / "query_manifest.csv", query_manifest_rows)
        atomic_json(output / "state.json", {"status": "RUNNING", "stage": "decode_track_R33", "pid": os.getpid(), "updated_unix": time.time(), "git_head": git_head()})
        decoded = decode_video(VideoSource(sample_id="v7-query-density-04LAX", source_video_id="04LAX::fake", source_locator=VIDEO), frame_indices.tolist())
        from scripts.run_v7_explicit_geometry_frontend import OnlineBootsTapir, DepthProRunner, rgbd_odometry
        tracker = OnlineBootsTapir(density.TAPNET_SOURCE, density.TAPNET_CHECKPOINT, process_size=PROCESS_SIZE, grid_size=BASE_GRID_SIZE)
        original_axis, dense_axis, _ = nested_axes()
        uu, vv = np.meshgrid(dense_axis, dense_axis)
        r33_process = np.stack((uu.ravel(), vv.ravel()), axis=1).astype(np.float32)
        r33_uv, r33_visibility = _track_with_queries(tracker, decoded, r33_process)
        np.savez_compressed(output / "R33_requery.npz", frame_indices=frame_indices, timestamps_s=timestamps, frame_sizes_hw=np.asarray(old_arrays["frame_sizes_hw"], dtype=np.int64), track_ids=np.arange(1089, dtype=np.int64), uv=r33_uv, visibility=r33_visibility)
        del tracker
        import torch
        torch.cuda.empty_cache()
        atomic_json(output / "state.json", {"status": "RUNNING", "stage": "shared_geometry", "pid": os.getpid(), "updated_unix": time.time(), "git_head": git_head()})
        depth_runner = DepthProRunner(density.DEPTH_SOURCE, density.DEPTH_CHECKPOINT)
        depths, focals, depth_valid = depth_runner.infer(decoded)
        intrinsics, fixed_focal = causal_first_frame_intrinsics(depths, focals)
        adjacent, pair_valid, pose_info = rgbd_odometry(decoded, depths, intrinsics)
        world_from_camera, pose_valid = accumulate_world_from_camera(adjacent, pair_valid)
        sampled17, sampled17_valid = sample_depth_at_uv(depths, np.asarray(old_arrays["uv"], dtype=np.float32))
        observed17 = np.asarray(old_arrays["visibility"], dtype=bool) & sampled17_valid & depth_valid[:, None]
        xyz17, geo17 = world_xyz(np.asarray(old_arrays["uv"], dtype=np.float32), sampled17, intrinsics, world_from_camera, observed17, pose_valid)
        sampled33, sampled33_valid = sample_depth_at_uv(depths, r33_uv)
        observed33 = r33_visibility & sampled33_valid & depth_valid[:, None]
        xyz33, geo33 = world_xyz(r33_uv, sampled33, intrinsics, world_from_camera, observed33, pose_valid)
        np.savez_compressed(output / "shared_geometry.npz", depths=depths.astype(np.float32), intrinsics=intrinsics.astype(np.float64), world_from_camera=world_from_camera.astype(np.float64), pose_valid=pose_valid, depth_valid=depth_valid, pair_valid=pair_valid)
        np.savez_compressed(output / "R17_shared_geometry.npz", frame_indices=frame_indices, timestamps_s=timestamps, frame_sizes_hw=np.asarray(old_arrays["frame_sizes_hw"], dtype=np.int64), track_ids=np.arange(289, dtype=np.int64), uv=np.asarray(old_arrays["uv"], dtype=np.float32), visibility=np.asarray(old_arrays["visibility"], dtype=bool), geometry_validity=geo17, xyz=xyz17)
        np.savez_compressed(output / "R33_geometry.npz", frame_indices=frame_indices, timestamps_s=timestamps, frame_sizes_hw=np.asarray(old_arrays["frame_sizes_hw"], dtype=np.int64), track_ids=np.arange(1089, dtype=np.int64), uv=r33_uv, visibility=r33_visibility, geometry_validity=geo33, xyz=xyz33)
        atomic_json(output / "geometry_metadata.json", {"depth_provider": "apple_depth_pro", "depth_source_sha": density.DEPTH_SHA, "fixed_focal_px": fixed_focal, "focals_px": focals.tolist(), "depth_valid": depth_valid.tolist(), "pose_valid": pose_valid.tolist(), "pair_valid": pair_valid.tolist(), "same_geometry_for_R17_R33": True, "source_frame_indices": frame_indices.tolist(), "source_pts_s": timestamps.tolist()})
        sequences = {
            "R17": _build_sequence(decoded, uv=np.asarray(old_arrays["uv"], dtype=np.float32), visibility=np.asarray(old_arrays["visibility"], dtype=bool), xyz=xyz17, geometry_valid=geo17, grid_size=17, cohort="R::independent_query_frame_488", query_pts_s=float(timestamps[0])),
            "R33": _build_sequence(decoded, uv=r33_uv, visibility=r33_visibility, xyz=xyz33, geometry_valid=geo33, grid_size=33, cohort="R33::independent_query_frame_488", query_pts_s=float(timestamps[0])),
        }
        atomic_json(output / "state.json", {"status": "RUNNING", "stage": "support", "pid": os.getpid(), "updated_unix": time.time(), "git_head": git_head()})
        supports: dict[str, dict[str, Any]] = {}; units_by_density: dict[str, list[dict[str, Any]]] = {}; s_by_density: dict[str, np.ndarray] = {}; q_by_density: dict[str, np.ndarray] = {}; intervals_by_density: dict[str, np.ndarray] = {}
        for name, sequence in sequences.items():
            summary, units, s_array, q_array, intervals = _support(sequence, query_pts_s=float(timestamps[0]))
            supports[name] = summary; units_by_density[name] = units; s_by_density[name] = s_array; q_by_density[name] = q_array; intervals_by_density[name] = intervals
            atomic_json(output / f"{name}_support.json", summary)
        manifests: dict[str, list[dict[str, Any]]] = {}
        for name in ("R17", "R33"):
            manifests[name] = _feature_manifest(output, name, supports[name], units_by_density[name], s_by_density[name], q_by_density[name], intervals_by_density[name])
        atomic_json(output / "feature_manifest.json", manifests)
        atomic_json(output / "support_summary.json", supports)
        original_compare = _compare_original_queries(np.asarray(old_arrays["uv"]), np.asarray(old_arrays["visibility"]), geo17, r33_uv, r33_visibility, geo33, query_manifest_rows, frame_indices)
        write_csv(output / "original_query_consistency.csv", original_compare)
        atomic_json(output / "original_query_consistency.json", {"rows": original_compare, "note": "R17 uses the audited saved UV; R33 rows at original query positions are compared by the frozen query_manifest mapping. Displacement is not a correctness proof."})
        roi_rows: list[dict[str, Any]] = []
        roi_by_frame = _roi_annotations()
        for frame, annotation in sorted(roi_by_frame.items()):
            rect = {"x_min": annotation["x_min"], "y_min": annotation["y_min"], "x_max": annotation["x_max"], "y_max": annotation["y_max"]}
            for name in ("R17", "R33"):
                roi_manifest = query_manifest_rows if name == "R33" else [{"query_id": i, "membership": "original", "original_query_id": i} for i in range(289)]
                roi_rows.append(_roi_counts(sequences[name], supports[name], units_by_density[name], roi_manifest, name, frame, rect))
        write_csv(output / "roi_support_comparison.csv", roi_rows)
        overlay_paths = _draw_overlays(output, decoded, sequences, {"R33": query_manifest_rows, "R17": [{"query_id": i, "membership": "original", "original_query_id": i} for i in range(289)]}, units_by_density)
        atomic_json(output / "overlay_index.json", {"paths": overlay_paths, "target_frames": list(TARGET_FRAMES), "note": "green marks geometry-valid points in a valid unit; R33 cyan=original query and orange=added query; colors do not indicate authenticity"})
        atomic_json(output / "state.json", {"status": "RUNNING", "stage": "model_readout", "pid": os.getpid(), "updated_unix": time.time(), "git_head": git_head()})
        window_scores, unit_scores = _model_readout(output, manifests)
        atomic_json(output / "final_status.json", {"status": "COMPLETE_WITH_SHARED_GEOMETRY", "window_id": WINDOW_ID, "conditions": ["R17", "R33"], "query_counts": {"R17": 289, "R33": 1089, "R33_added": 800}, "support": {name: {key: supports[name][key] for key in ("history_group_count", "retained_group_count", "valid_unit_count", "invalid_group_or_unit_count", "q_changed_unit_count", "component_count")} for name in supports}, "model_readout": "AVAILABLE", "elapsed_s": time.perf_counter() - started, "git_head": git_head()})
        report_lines = ["# 04LAX R 查询密度匹配对照", "", "状态：`COMPLETE_WITH_SHARED_GEOMETRY`。这是单案例观测支撑诊断，不是新的检测器训练、AUROC 或空间真值评价。", "", "## 结果先行", f"- R17 使用旧 R_requery 的 289 点 UV/visibility；R33 在同一源视频 frame {QUERY_FRAME} / PTS {timestamps[0]:.12f} 重新查询 1089 点，其中新增 800 点。33 轴由实际 17 轴逐邻接插入中点构造，未使用另一套 linspace。", f"- R17/R33 共用一次 Depth Pro、固定首帧 focal={fixed_focal:.6f}、Open3D pose 与世界坐标反投影；R17 的 `R17_shared_geometry.npz` 是公平比较输入，旧 R_geometry 仅作历史参照。", "", "| condition | history groups | retained groups | valid five-time units | invalid reasons | Q changed units |", "|---|---:|---:|---:|---|---:|"]
        for name in ("R17", "R33"):
            report_lines.append(f"| {name} | {supports[name]['history_group_count']} | {supports[name]['retained_group_count']} | {supports[name]['valid_unit_count']} | `{supports[name]['invalid_reason_counts']}` | {supports[name]['q_changed_unit_count']} |")
        report_lines += ["", "## 原查询一致性", "", "| frame | mapped original visible R17/R33 | visibility agreement | both visible | UV median / p95 / max px | geometry valid delta R33-R17 |", "|---:|---:|---:|---:|---|---:|"]
        for row in original_compare:
            report_lines.append(f"| {row['frame_index']} | {row['r17_original_visible']}/{row['r33_mapped_original_visible']} | {row['visibility_agreement']:.4f} | {row['both_visible_count']} | {row['uv_displacement_median_px']}/{row['uv_displacement_p95_px']}/{row['uv_displacement_max_px']} | {row['geometry_validity_delta']} |")
        report_lines += ["", "## ROI 对照", "", "已确认 ROI 只使用 `roi_annotations.json` 中 status=CONFIRMED 的逐帧矩形；frame 488 的 SKIP 未使用。矩形是主体粗框，不是精确伪造 mask。", "", "| frame | density | visible in ROI | geometry-valid in ROI | common-member points | pair both / one endpoint | original/mixed/added pairs |", "|---:|---|---:|---:|---:|---:|---|"]
        for row in roi_rows:
            report_lines.append(f"| {row['frame_index']} | {row['density']} | {row['visible_points_in_roi']} | {row['geometry_valid_points_in_roi']} | {row['common_member_points_in_roi']} | {row['valid_pairs_both_endpoints_in_roi']} / {row['valid_pairs_one_endpoint_in_roi']} | {row['valid_pairs_both_original']} / {row['valid_pairs_mixed_original_added']} / {row['valid_pairs_both_added']} |")
        report_lines += ["", "## Frozen-model readout", "", "模型在原 R17 分布上训练；本读出不重新拟合标准化、不训练、不校准。R17/R33 的局部组不是物理一一对应，分数变化不能单独证明检测收益。", "", "| density | condition | seed 20260909 | seed 20260910 | seed 20260911 | mean |", "|---|---|---:|---:|---:|---:|"]
        for density_name in ("R17", "R33"):
            for condition in MODEL_CONDITIONS:
                values = window_scores["conditions"][density_name][condition]
                mean = float(np.mean([values[str(seed)] for seed in SEEDS]))
                report_lines.append(f"| {density_name} | {condition} | {values[str(SEEDS[0])]:.9f} | {values[str(SEEDS[1])]:.9f} | {values[str(SEEDS[2])]:.9f} | {mean:.9f} |")
        report_lines += ["", "## 限制", "", "- R33 的原 289 查询在联合 1089 查询中可能改变跟踪输出；这次只把 mapped-original 差异作为事实记录，不能将其全部解释成 800 条独立轨迹。", "- 点数增加会自然增加候选 pair；有效 unit/关系增加不等于同等比例的新独立信息，也不等于空间定位真值。", "- 只分析一个 04LAX fake 案例；冻结模型读出是开发诊断，不是 sealed-test。", "", "## 产物", "", "`protocol.json`、`query_manifest.json/csv`、`R33_requery.npz`、`R17_shared_geometry.npz`、`R33_geometry.npz`、`support_comparison.csv`、`original_query_consistency.csv`、`roi_support_comparison.csv`、`window_scores.json`、`overlays/`。"]
        (output / "report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")
        write_csv(output / "support_comparison.csv", [{"density": name, "query_count": 289 if name == "R17" else 1089, "added_query_count": 0 if name == "R17" else 800, "history_group_count": supports[name]["history_group_count"], "retained_group_count": supports[name]["retained_group_count"], "valid_unit_count": supports[name]["valid_unit_count"], "invalid_group_or_unit_count": supports[name]["invalid_group_or_unit_count"], "invalid_reason_counts": json.dumps(supports[name]["invalid_reason_counts"], sort_keys=True), "q_changed_unit_count": supports[name]["q_changed_unit_count"], "component_count": supports[name]["component_count"]} for name in ("R17", "R33")])
        atomic_json(output / "state.json", {"status": "COMPLETE", "stage": "report", "pid": os.getpid(), "updated_unix": time.time(), "git_head": git_head(), "elapsed_s": time.perf_counter() - started})
        return {"status": "COMPLETE_WITH_SHARED_GEOMETRY", "output": str(output), "elapsed_s": time.perf_counter() - started}
    except Exception as exc:
        error = {"type": type(exc).__name__, "message": str(exc)}
        atomic_json(output / "final_status.json", {"status": "FAILED", "error": error, "traceback": __import__("traceback").format_exc(), "elapsed_s": time.perf_counter() - started, "git_head": git_head()})
        atomic_json(output / "state.json", {"status": "FAILED", "stage": "error", "error": error, "updated_unix": time.time(), "git_head": git_head()})
        raise


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--resume-readout", action="store_true", help="resume from verified frontend/geometry/support artifacts")
    args = parser.parse_args(argv)
    result = _finish_existing(args.output) if args.resume_readout else run(args.output)
    print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
