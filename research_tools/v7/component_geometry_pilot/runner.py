"""Run the bounded V7 learnable component-geometry comparison.

Only completed R particle/support caches are read.  The runner deliberately
does not open video files or invoke tracking, depth, pose or segmentation.
Three conditions share the same windows, labels, components, members and
five target times: a matched S+Q summary, a masked 3-D local-edge encoder and
a matched 2-D local-edge control built on the same frozen 3-D grouping.
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
from research_tools.v7.geometry_information_pilot import runner as geometry
from research_tools.v7.multi_order_sequence_probe.model import FeatureStandardizer, fit_standardizer, source_class_weights
from research_tools.v7.periodic_requery_probe import runner as periodic
from research_tools.v7.source128_extension import runner as source128

from .model import ComponentGeometryModel, SummaryModel, parameter_count, state_dict_numpy


REPO_ROOT = Path(__file__).resolve().parents[3]
DATA_ROOT = Path("/root/autodl-tmp/data/sparse_3d_forgery_detection")
STRUCTURE_ROOT = DATA_ROOT / "derived/v7_activityforensics_observation_support_pilot_v1"
SOURCE_ROOT = DATA_ROOT / "derived/v7_activityforensics_source128_extension_v1"
OUTPUT_ROOT = DATA_ROOT / "derived/v7_activityforensics_component_geometry_pilot_v1"
CONDITIONS = ("SUMMARY_SET", "GEOMETRY_3D", "GEOMETRY_2D")
SEEDS = (20260909, 20260910, 20260911)
EPOCHS = 200
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
BOOTSTRAP_SEED = 20260909
BOOTSTRAP_REPLICATES = 10_000
MAX_NEIGHBORS = 8
MATCH_TOLERANCE = 1e-5
UNIT_CHUNK_SIZE = 256
WINDOW_TRAIN_CHUNK_SIZE = 64


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


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {key: np.asarray(archive[key]) for key in archive.files}


def _load_selection() -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    train, validation, support_rows, info = geometry._load_selection()
    feature_manifest = json.loads((STRUCTURE_ROOT / "inputs/feature_manifest.json").read_text(encoding="utf-8"))
    feature_by_id = {str(item["window_id"]): dict(item) for item in feature_manifest}
    expected = {str(row["window_id"]) for row in train + validation}
    if set(feature_by_id) != expected:
        raise ValueError(f"STRUCTURE_WINDOW_SET_MISMATCH:{len(feature_by_id)}:{len(expected)}")
    support_by_id = {str(row["window_id"]): dict(row) for row in support_rows if str(row.get("mode")) == "R"}
    if len(support_by_id) != len([row for row in support_rows if str(row.get("mode")) == "R"]):
        raise ValueError("DUPLICATE_R_SUPPORT_WINDOW_ID")
    if not expected.issubset(support_by_id):
        raise ValueError("R_SUPPORT_WINDOW_MISSING")
    rows: list[dict[str, Any]] = []
    for row in train + validation:
        item = feature_by_id[str(row["window_id"])]
        path = STRUCTURE_ROOT / str(item["input_path"])
        if not path.is_file():
            raise FileNotFoundError(f"MISSING_STRUCTURE_FEATURE:{path}")
        merged = dict(support_by_id[str(row["window_id"])])
        merged["split"] = "train" if row in train else "validation"
        merged["feature_item"] = item
        rows.append(merged)
    train_rows = rows[: len(train)]
    validation_rows = rows[len(train) :]
    train_sources = {str(row["source_id"]) for row in train_rows}
    validation_sources = {str(row["source_id"]) for row in validation_rows}
    if train_sources & validation_sources:
        raise ValueError("TRAIN_VALIDATION_SOURCE_OVERLAP")
    info = {**info, "structure_feature_manifest": feature_manifest, "training_windows": len(train_rows), "validation_windows": len(validation_rows), "training_sources": sorted(train_sources), "validation_sources": sorted(validation_sources), "training_real": sum(int(row["label"]) == 0 for row in train_rows), "training_fake": sum(int(row["label"]) == 1 for row in train_rows), "validation_real": sum(int(row["label"]) == 0 for row in validation_rows), "validation_fake": sum(int(row["label"]) == 1 for row in validation_rows)}
    return train_rows, validation_rows, support_by_id, info


def _sequence_for_row(row: Mapping[str, Any]) -> Any:
    """Load and validate one cached R ParticleSequence without frontend work."""

    prefix = str(row.get("particle_prefix") or "")
    if not prefix:
        raise ValueError(f"MISSING_PARTICLE_PREFIX:{row['window_id']}")
    sequence = load_particle_sequence(prefix)
    expected_video = f"{row['source_id']}::{row['role']}"
    if str(sequence.source_video_id) != expected_video or int(sequence.num_tracks) != 289:
        raise ValueError(f"SEQUENCE_IDENTITY_MISMATCH:{row['window_id']}")
    if str(sequence.coordinate_system.length_unit.value) != "meter" or not sequence.coordinate_system.camera_motion_compensated:
        raise ValueError(f"SEQUENCE_COORDINATE_CONTRACT_MISMATCH:{row['window_id']}")
    lineage = sequence.lineage
    for key in ("source_id", "role", "pair_id"):
        if str(lineage.get(key, "")) != str(row.get(key, "")):
            raise ValueError(f"SEQUENCE_LINEAGE_MISMATCH:{row['window_id']}:{key}")
    offset = float(row.get("offset_s", 0.0))
    expected_cohort = "O" if abs(offset) < 1e-9 else "R"
    if str(sequence.provenance.get("query_cohort", "")) != expected_cohort:
        raise ValueError(f"SEQUENCE_COHORT_MISMATCH:{row['window_id']}")
    parent_id = str(row["window_id"]).rsplit("::b", 1)[0]
    expected_window = parent_id if expected_cohort == "O" else str(row["window_id"])
    if str(lineage.get("window_id", "")) != expected_window:
        raise ValueError(f"SEQUENCE_WINDOW_MISMATCH:{row['window_id']}")
    uv = np.asarray(sequence.uv)
    xyz = np.asarray(sequence.xyz)
    if uv.shape[:2] != xyz.shape[:2] or uv.shape[-1] != 2 or xyz.shape[-1] != 3:
        raise ValueError(f"SEQUENCE_ARRAY_SHAPE_MISMATCH:{row['window_id']}")
    return sequence


def _fixed_edges(xyz: np.ndarray, slots: np.ndarray, history_index: int) -> list[tuple[int, int]]:
    """Build canonical directed edges once from the frozen history geometry."""

    history = np.asarray(xyz[history_index], dtype=np.float64)
    if slots.size < 2 or not np.all(np.isfinite(history[slots])):
        return []
    edges: list[tuple[int, int]] = []
    for p in slots.tolist():
        candidates = []
        for q in slots.tolist():
            if int(q) == int(p):
                continue
            distance = float(np.linalg.norm(history[int(q)] - history[int(p)]))
            if math.isfinite(distance):
                candidates.append((distance, int(q)))
        candidates.sort(key=lambda value: (value[0], value[1]))
        for _, q in candidates[:MAX_NEIGHBORS]:
            edges.append((int(p), int(q)))
    return edges


def _edge_values(sequence: Any, slots: np.ndarray, edges: Sequence[tuple[int, int]], array_indices: np.ndarray, *, dimension: int) -> tuple[np.ndarray, np.ndarray]:
    uv = np.asarray(sequence.uv, dtype=np.float64)
    xyz = np.asarray(sequence.xyz, dtype=np.float64)
    visibility = np.asarray(sequence.visibility, dtype=bool)
    geometry_validity = np.asarray(sequence.geometry_validity, dtype=bool)
    if dimension == 3:
        coordinates = xyz
        coordinate_valid = geometry_validity & visibility & np.all(np.isfinite(xyz), axis=-1)
    elif dimension == 2:
        coordinates = uv
        coordinate_valid = visibility & np.all(np.isfinite(uv), axis=-1)
    else:
        raise ValueError(dimension)
    values = np.zeros((5, len(edges), 2 * dimension + 1), dtype=np.float32)
    mask = np.zeros((5, len(edges)), dtype=bool)
    for time_index, array_index in enumerate(array_indices.tolist()):
        valid_slots = slots[coordinate_valid[int(array_index), slots]]
        if valid_slots.size == 0:
            continue
        center = np.mean(coordinates[int(array_index), valid_slots], axis=0)
        for edge_index, (p, q) in enumerate(edges):
            if not coordinate_valid[int(array_index), p] or not coordinate_valid[int(array_index), q]:
                continue
            left = coordinates[int(array_index), p] - center
            relative = coordinates[int(array_index), q] - coordinates[int(array_index), p]
            norm = np.asarray([np.linalg.norm(relative)], dtype=np.float64)
            feature = np.concatenate((left, relative, norm))
            if not np.all(np.isfinite(feature)):
                continue
            values[time_index, edge_index] = feature.astype(np.float32)
            mask[time_index, edge_index] = True
    return values, mask


def _build_features(root: Path, train_rows: Sequence[Mapping[str, Any]], validation_rows: Sequence[Mapping[str, Any]], *, resume: bool) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    selected = list(train_rows) + list(validation_rows)
    selected_ids = {str(row["window_id"]) for row in selected}
    manifest_path = root / "inputs/feature_manifest.json"
    summary_path = root / "support/matching_summary.json"
    if resume and manifest_path.is_file() and summary_path.is_file():
        old = json.loads(manifest_path.read_text(encoding="utf-8"))
        old_ids = {str(item["window_id"]) for item in old}
        if old_ids == selected_ids and all((root / str(item["input_path"])).is_file() for item in old):
            by_id = {str(item["window_id"]): item for item in old}
            train = [{**dict(row), "matched_unit_count": int(by_id[str(row["window_id"])] ["matched_unit_count"])} for row in train_rows if str(row["window_id"]) in by_id]
            validation = [{**dict(row), "matched_unit_count": int(by_id[str(row["window_id"])] ["matched_unit_count"])} for row in validation_rows if str(row["window_id"]) in by_id]
            return train, validation, json.loads(summary_path.read_text(encoding="utf-8")), old
    root.joinpath("features/window_inputs").mkdir(parents=True, exist_ok=True)
    manifests: list[dict[str, Any]] = []
    coverage: list[dict[str, Any]] = []
    reason_counts: Counter[str] = Counter()
    sequence_cache: dict[str, Any] = {}
    structure_by_id = {str(item["window_id"]): item for item in json.loads((STRUCTURE_ROOT / "inputs/feature_manifest.json").read_text(encoding="utf-8"))}
    for index, row in enumerate(sorted(selected, key=lambda item: str(item["window_id"]))):
        window_id = str(row["window_id"])
        progress(root, "features", "RUNNING", index, len(selected), current_window=window_id)
        structure_item = structure_by_id[window_id]
        arrays = _load_npz(STRUCTURE_ROOT / str(structure_item["input_path"]))
        s = np.asarray(arrays["s"], dtype=np.float64)
        q = np.asarray(arrays["q"], dtype=np.float64)
        intervals = np.asarray(arrays["intervals"], dtype=np.float64)
        valid_units = [unit for unit in row.get("support", {}).get("units", []) if str(unit.get("status")) == "VALID"]
        identities = list(structure_item.get("unit_identities", []))
        if s.shape != (len(valid_units), 5, 4) or q.shape != s.shape or len(identities) != len(valid_units) or intervals.shape != (4,):
            raise ValueError(f"STRUCTURE_INPUT_SHAPE_MISMATCH:{window_id}")
        prefix = str(row["particle_prefix"])
        if prefix not in sequence_cache:
            sequence_cache[prefix] = _sequence_for_row(row)
        sequence = sequence_cache[prefix]
        kept_indices: list[int] = []
        edge3_list: list[np.ndarray] = []
        edge2_list: list[np.ndarray] = []
        mask_list: list[np.ndarray] = []
        kept_identities: list[dict[str, Any]] = []
        dropped: Counter[str] = Counter()
        xyz = np.asarray(sequence.xyz, dtype=np.float64)
        for unit_index, identity in enumerate(identities):
            unit = valid_units[unit_index]
            slots = np.asarray(identity.get("common_member_slots", []), dtype=np.int64)
            array_indices = np.asarray(identity.get("target_array_indices", []), dtype=np.int64)
            history_index = int(identity.get("history_reference_array_index", -1))
            if slots.ndim != 1 or slots.size < 2 or array_indices.shape != (5,) or history_index < 0 or history_index >= xyz.shape[0]:
                dropped["IDENTITY_SHAPE_INVALID"] += 1
                continue
            if list(map(int, slots.tolist())) != list(map(int, unit.get("member_slots", []))):
                raise ValueError(f"COMMON_MEMBER_MISMATCH:{window_id}:{unit_index}")
            edges = _fixed_edges(xyz, slots, history_index)
            if not edges:
                dropped["HISTORY_EDGE_UNAVAILABLE"] += 1
                continue
            edge3, mask3 = _edge_values(sequence, slots, edges, array_indices, dimension=3)
            edge2, mask2 = _edge_values(sequence, slots, edges, array_indices, dimension=2)
            # A/B/C use one identical unit support.  A unit is retained only
            # when both modalities have at least one valid fixed edge at all
            # five target times; no missing value is turned into a real point.
            common_mask = mask3 & mask2
            if not np.all(np.any(common_mask, axis=1)):
                dropped["TARGET_FIXED_EDGE_SUPPORT_INCOMPLETE"] += 1
                continue
            if not np.all(np.isfinite(s[unit_index])) or not np.all(np.isfinite(q[unit_index])):
                dropped["SUMMARY_NONFINITE"] += 1
                continue
            kept_indices.append(unit_index)
            edge3_list.append(edge3)
            edge2_list.append(edge2)
            mask_list.append(common_mask)
            kept_identities.append({**dict(identity), "unit_index": len(kept_indices) - 1, "fixed_directed_edges": [[int(left), int(right)] for left, right in edges], "fixed_edge_count": len(edges), "history_reference_frame_index": int(sequence.frame_indices[history_index]), "history_reference_timestamp_s": float(sequence.timestamps_s[history_index]), "target_frame_indices": [int(x) for x in sequence.frame_indices[array_indices]], "target_timestamps_s": [float(x) for x in sequence.timestamps_s[array_indices]], "geometry_coordinate": "camera_motion_compensated_meter", "uv_coordinate": "source_pixel_xy", "center_definition": "mean of finite visible members at each target time"})
        if not kept_indices:
            reason_counts.update(dropped)
            coverage.append({"window_id": window_id, "split": str(row["split"]), "status": "NO_COMMON_GEOMETRY_UNIT", "original_unit_count": len(valid_units), "matched_unit_count": 0, "dropped_unit_reasons": json.dumps(dict(dropped), sort_keys=True)})
            continue
        max_edges = max(int(item.shape[1]) for item in edge3_list)
        edge3 = np.zeros((len(kept_indices), 5, max_edges, 7), dtype=np.float32)
        edge2 = np.zeros((len(kept_indices), 5, max_edges, 5), dtype=np.float32)
        mask = np.zeros((len(kept_indices), 5, max_edges), dtype=bool)
        for output_index, (value3, value2, value_mask) in enumerate(zip(edge3_list, edge2_list, mask_list)):
            edge3[output_index, :, : value3.shape[1]] = value3
            edge2[output_index, :, : value2.shape[1]] = value2
            mask[output_index, :, : value_mask.shape[1]] = value_mask
        selected_s = s[np.asarray(kept_indices, dtype=np.int64)].astype(np.float32)
        selected_q = q[np.asarray(kept_indices, dtype=np.int64)].astype(np.float32)
        relative = Path("features/window_inputs") / f"{_safe(window_id)}.npz"
        target = root / relative
        temporary = target.with_name(target.name + ".tmp.npz")
        np.savez_compressed(temporary, s=selected_s, q=selected_q, intervals=intervals.astype(np.float64), edge3d=edge3, edge2d=edge2, edge_mask=mask)
        os.replace(temporary, target)
        manifests.append({"window_id": window_id, "source_id": str(row["source_id"]), "role": str(row["role"]), "label": int(row["label"]), "annotation_category": str(row.get("annotation_category", "")), "offset_s": float(row.get("offset_s", 0.0)), "input_path": str(relative), "s_shape": list(selected_s.shape), "q_shape": list(selected_q.shape), "edge3d_shape": list(edge3.shape), "edge2d_shape": list(edge2.shape), "edge_mask_shape": list(mask.shape), "original_unit_count": len(valid_units), "matched_unit_count": len(kept_indices), "unit_identities": kept_identities, "dropped_unit_reasons": dict(dropped), "particle_prefix": prefix, "sequence_source_video_id": str(sequence.source_video_id), "sequence_query_cohort": str(sequence.provenance.get("query_cohort"))})
        reason_counts.update(dropped)
        coverage.append({"window_id": window_id, "split": str(row["split"]), "status": "MATCHED", "original_unit_count": len(valid_units), "matched_unit_count": len(kept_indices), "dropped_unit_reasons": json.dumps(dict(dropped), sort_keys=True)})
        if index % 25 == 0:
            print(f"component-geometry features {index + 1}/{len(selected)} window={window_id} units={len(kept_indices)}", flush=True)
    manifests.sort(key=lambda item: str(item["window_id"]))
    write_csv(root / "support/matching_coverage.csv", coverage)
    atomic_json(root / "inputs/feature_manifest.json", manifests)
    train_ids = {str(row["window_id"]) for row in train_rows}
    content_hash = _content_hash(root, manifests)
    summary = {"feature_version": "component-geometry-v1", "selected_windows": len(selected), "matched_windows": len(manifests), "matched_train_windows": sum(str(item["window_id"]) in train_ids for item in manifests), "matched_validation_windows": sum(str(item["window_id"]) not in train_ids for item in manifests), "original_train_units": sum(int(row.get("valid_unit_count", 0)) for row in train_rows), "original_validation_units": sum(int(row.get("valid_unit_count", 0)) for row in validation_rows), "matched_train_units": sum(int(item["matched_unit_count"]) for item in manifests if str(item["window_id"]) in train_ids), "matched_validation_units": sum(int(item["matched_unit_count"]) for item in manifests if str(item["window_id"]) not in train_ids), "reason_counts": dict(reason_counts), "max_neighbors": MAX_NEIGHBORS, "feature_content_sha256": content_hash, "structure_feature_manifest_sha256": sha256(STRUCTURE_ROOT / "inputs/feature_manifest.json"), "source_support_sha256": sha256(SOURCE_ROOT / "support/window_support.json")}
    atomic_json(summary_path, summary)
    progress(root, "features", "COMPLETE", len(manifests), len(selected), **summary)
    by_id = {str(item["window_id"]): item for item in manifests}
    train_kept = [{**dict(row), "matched_unit_count": int(by_id[str(row["window_id"])] ["matched_unit_count"])} for row in train_rows if str(row["window_id"]) in by_id]
    validation_kept = [{**dict(row), "matched_unit_count": int(by_id[str(row["window_id"])] ["matched_unit_count"])} for row in validation_rows if str(row["window_id"]) in by_id]
    return train_kept, validation_kept, summary, manifests


def _content_hash(root: Path, manifest: Sequence[Mapping[str, Any]]) -> str:
    digest = hashlib.sha256()
    for item in sorted(manifest, key=lambda value: str(value["window_id"])):
        digest.update(str(item["window_id"]).encode()); digest.update(b"\0")
        path = root / str(item["input_path"])
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                digest.update(block)
    return digest.hexdigest()


def _fit_edge_standardizer(root: Path, manifests: Sequence[Mapping[str, Any]], rows: Sequence[Mapping[str, Any]], key: str) -> dict[str, Any]:
    weights = source_class_weights(rows)
    weight_by_id = {str(row["window_id"]): float(weight) for row, weight in zip(rows, weights)}
    values: list[np.ndarray] = []
    value_weights: list[np.ndarray] = []
    for manifest in manifests:
        window_id = str(manifest["window_id"])
        if window_id not in weight_by_id:
            continue
        weight = weight_by_id[window_id]
        arrays = _load_npz(root / str(manifest["input_path"]))
        value = np.asarray(arrays[key], dtype=np.float64)
        mask = np.asarray(arrays["edge_mask"], dtype=bool)
        selected = value[mask]
        if selected.size:
            values.append(selected)
            value_weights.append(np.full(selected.shape[0], float(weight) / max(selected.shape[0], 1), dtype=np.float64))
    if not values:
        raise ValueError(f"NO_VALID_EDGE_VALUES:{key}")
    matrix = np.concatenate(values, axis=0)
    row_weights = np.concatenate(value_weights, axis=0)
    mean = np.average(matrix, axis=0, weights=row_weights)
    variance = np.average((matrix - mean) ** 2, axis=0, weights=row_weights)
    zero = np.flatnonzero(~np.isfinite(variance) | (variance <= 1e-12)).astype(int).tolist()
    scale = np.sqrt(np.maximum(variance, 1e-12)); scale[zero] = 1.0
    return {"mean": mean, "scale": scale, "zero_variance_dimensions": zero, "condition": key, "training_edge_rows": int(matrix.shape[0]), "training_window_count": len(rows)}


def _std_array(values: np.ndarray, standardizer: Mapping[str, Any]) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    output = (array - np.asarray(standardizer["mean"], dtype=np.float64)) / np.asarray(standardizer["scale"], dtype=np.float64)
    return output.astype(np.float32)


def _batch(root: Path, manifests: Sequence[Mapping[str, Any]], rows: Sequence[Mapping[str, Any]], standard_s: FeatureStandardizer, standard_q: FeatureStandardizer, standard3: Mapping[str, Any], standard2: Mapping[str, Any], condition: str) -> dict[str, Any]:
    by_id = {str(item["window_id"]): item for item in manifests}
    values_s: list[np.ndarray] = []; values_q: list[np.ndarray] = []; edge_values: list[np.ndarray] = []; masks: list[np.ndarray] = []; intervals: list[np.ndarray] = []; indices: list[int] = []; labels: list[int] = []; ids: list[str] = []
    for window_index, row in enumerate(rows):
        item = by_id[str(row["window_id"])]
        arrays = _load_npz(root / str(item["input_path"]))
        s = standard_s.transform(np.asarray(arrays["s"], dtype=np.float64)).astype(np.float32)
        q = standard_q.transform(np.asarray(arrays["q"], dtype=np.float64)).astype(np.float32)
        values_s.append(s); values_q.append(q); intervals.append(np.repeat(np.asarray(arrays["intervals"], dtype=np.float32)[None, :], s.shape[0], axis=0)); indices.extend([window_index] * s.shape[0]); labels.append(int(row["label"])); ids.append(str(row["window_id"]))
        if condition != "SUMMARY_SET":
            raw_key = "edge3d" if condition == "GEOMETRY_3D" else "edge2d"
            standard = standard3 if condition == "GEOMETRY_3D" else standard2
            edge_values.append(_std_array(np.asarray(arrays[raw_key], dtype=np.float32), standard))
            masks.append(np.asarray(arrays["edge_mask"], dtype=bool))
    if not ids:
        raise ValueError(f"EMPTY_BATCH:{condition}")
    batch: dict[str, Any] = {"s": np.concatenate(values_s), "q": np.concatenate(values_q), "intervals": np.concatenate(intervals), "window_index": np.asarray(indices, dtype=np.int64), "labels": np.asarray(labels, dtype=np.int64), "window_ids": ids, "window_rows": list(rows), "window_count": len(rows), "window_weights": source_class_weights(rows)}
    if condition != "SUMMARY_SET":
        # Each window has its own number of fixed directed edges.  Pad only
        # the edge axis here; the explicit mask keeps these entries out of
        # the encoder and component mean.
        max_edges = max(int(value.shape[2]) for value in edge_values)
        padded_values: list[np.ndarray] = []
        padded_masks: list[np.ndarray] = []
        for value, value_mask in zip(edge_values, masks):
            if value.shape[2] == max_edges:
                padded_values.append(value); padded_masks.append(value_mask)
                continue
            value_pad = np.zeros((value.shape[0], value.shape[1], max_edges, value.shape[3]), dtype=np.float32)
            mask_pad = np.zeros((value_mask.shape[0], value_mask.shape[1], max_edges), dtype=bool)
            value_pad[:, :, : value.shape[2]] = value
            mask_pad[:, :, : value_mask.shape[2]] = value_mask
            padded_values.append(value_pad); padded_masks.append(mask_pad)
        batch["edge_values"] = np.concatenate(padded_values)
        batch["edge_mask"] = np.concatenate(padded_masks)
    batch["states"] = np.concatenate((batch["s"], batch["q"]), axis=-1)
    return batch


def _initial_model(condition: str, seed: int) -> Any:
    import torch
    torch.manual_seed(int(seed)); np.random.seed(int(seed) & 0xFFFFFFFF)
    if condition == "SUMMARY_SET":
        return SummaryModel()
    return ComponentGeometryModel(7 if condition == "GEOMETRY_3D" else 5)


def _forward(model: Any, batch: Mapping[str, Any], device: str) -> Any:
    import torch
    # A full 718-window edge tensor can contain tens of millions of padded
    # values.  Chunking units here preserves the exact window mean and loss,
    # while avoiding a large intermediate edge-encoder activation on CUDA.
    local_parts: list[torch.Tensor] = []
    index_parts: list[torch.Tensor] = []
    unit_count = int(np.asarray(batch["window_index"]).shape[0])
    for start in range(0, unit_count, UNIT_CHUNK_SIZE):
        stop = min(start + UNIT_CHUNK_SIZE, unit_count)
        index = torch.as_tensor(np.asarray(batch["window_index"])[start:stop], dtype=torch.long, device=device)
        intervals = torch.as_tensor(np.asarray(batch["intervals"])[start:stop], dtype=torch.float32, device=device)
        q = torch.as_tensor(np.asarray(batch["q"])[start:stop], dtype=torch.float32, device=device)
        if isinstance(model, SummaryModel):
            local = model(torch.as_tensor(np.asarray(batch["states"])[start:stop], dtype=torch.float32, device=device), intervals)
        else:
            local = model(torch.as_tensor(np.asarray(batch["edge_values"])[start:stop], dtype=torch.float32, device=device), torch.as_tensor(np.asarray(batch["edge_mask"])[start:stop], dtype=torch.bool, device=device), q, intervals)
        local_parts.append(local); index_parts.append(index)
    if not local_parts:
        raise ValueError("EMPTY_UNIT_BATCH")
    local = torch.cat(local_parts, dim=0)
    index = torch.cat(index_parts, dim=0)
    sums = torch.zeros(int(batch["window_count"]), dtype=local.dtype, device=device)
    counts = torch.zeros_like(sums)
    sums.index_add_(0, index, local); counts.index_add_(0, index, torch.ones_like(local))
    return sums / counts.clamp_min(1.0)


def _window_slice(batch: Mapping[str, Any], start: int, stop: int) -> dict[str, Any]:
    """Select a consecutive window block for gradient accumulation."""

    window_index = np.asarray(batch["window_index"], dtype=np.int64)
    keep = (window_index >= start) & (window_index < stop)
    if not np.any(keep):
        raise ValueError(f"EMPTY_WINDOW_BLOCK:{start}:{stop}")
    result = dict(batch)
    for key in ("s", "q", "states", "intervals", "edge_values", "edge_mask", "window_index"):
        if key in batch:
            result[key] = np.asarray(batch[key])[keep].copy()
    result["window_index"] = window_index[keep] - start
    result["window_rows"] = list(batch["window_rows"])[start:stop]
    result["window_ids"] = list(batch["window_ids"])[start:stop]
    result["labels"] = np.asarray(batch["labels"])[start:stop].copy()
    result["window_weights"] = np.asarray(batch["window_weights"])[start:stop].copy()
    result["window_count"] = stop - start
    return result


def _train_one(condition: str, batch: Mapping[str, Any], seed: int, device: str, epochs: int = EPOCHS) -> tuple[Any, dict[str, Any]]:
    import torch
    import torch.nn.functional as F
    model = _initial_model(condition, seed).to(device); model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    labels = torch.as_tensor(batch["labels"], dtype=torch.float32, device=device)
    weights = torch.as_tensor(batch["window_weights"], dtype=torch.float32, device=device)
    history: list[dict[str, Any]] = []
    for epoch in range(1, int(epochs) + 1):
        optimizer.zero_grad(set_to_none=True)
        loss_value = 0.0
        for start in range(0, int(batch["window_count"]), WINDOW_TRAIN_CHUNK_SIZE):
            stop = min(start + WINDOW_TRAIN_CHUNK_SIZE, int(batch["window_count"]))
            block = _window_slice(batch, start, stop)
            scores = _forward(model, block, device)
            block_labels = labels[start:stop]
            block_weights = weights[start:stop]
            # Accumulate the exact full-batch weighted BCE gradient over
            # window blocks; optimizer.step remains once per epoch.
            loss = torch.sum(F.binary_cross_entropy_with_logits(scores, block_labels, reduction="none") * block_weights) / torch.sum(weights)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"NONFINITE_LOSS:{condition}:{seed}:{epoch}")
            loss.backward()
            loss_value += float(loss.detach().cpu())
        if not all(parameter.grad is None or torch.all(torch.isfinite(parameter.grad)) for parameter in model.parameters()):
            raise FloatingPointError(f"NONFINITE_GRADIENT:{condition}:{seed}:{epoch}")
        optimizer.step()
        history.append({"epoch": epoch, "loss": loss_value})
        if epoch in (1, epochs):
            print(f"component-geometry condition={condition} seed={seed} epoch={epoch}/{epochs} loss={history[-1]['loss']:.6f}", flush=True)
    model.eval()
    return model, {"seed": int(seed), "epochs": int(epochs), "initial_loss": history[0]["loss"], "final_loss": history[-1]["loss"], "min_loss": min(item["loss"] for item in history), "loss_history": history, "device": str(torch.device(device)), "parameter_count": parameter_count(model)}


def _model_from_record(condition: str, record: Mapping[str, Any], device: str) -> Any:
    import torch
    model = _initial_model(condition, int(record["seed"]))
    state = {name: torch.as_tensor(value, dtype=model.state_dict()[name].dtype) for name, value in record["state_dict"].items()}
    model.load_state_dict(state); model.to(device); model.eval(); return model


def _metrics(rows: Sequence[Mapping[str, Any]], scores: Sequence[float]) -> tuple[dict[str, Any], dict[str, float]]:
    labels = [int(row["label"]) for row in rows]; values = [float(value) for value in scores]
    if not all(math.isfinite(value) for value in values):
        raise ValueError("NONFINITE_SCORE")
    by_source: dict[str, list[tuple[int, float]]] = defaultdict(list)
    for row, score in zip(rows, values):
        by_source[str(row["source_id"])].append((int(row["label"]), score))
    source_values = {source: periodic._auroc([item[0] for item in pairs], [item[1] for item in pairs]) for source, pairs in by_source.items()}
    source_values = {source: float(value) for source, value in source_values.items() if value is not None}
    return {"window_count": len(rows), "real_count": labels.count(0), "fake_count": labels.count(1), "source_count": len(by_source), "dual_role_source_count": len(source_values), "source_macro": source128._bootstrap(source_values), "pooled_auroc": periodic._auroc(labels, values), "pooled_ap": periodic._ap(labels, values), **periodic._classification(labels, values)}, source_values


def _metric_row(split: str, condition: str, seed: Any, metrics: Mapping[str, Any]) -> dict[str, Any]:
    ci = metrics.get("source_macro", {}).get("ci95") or [None, None]
    return {"split": split, "condition": condition, "seed": seed, "window_count": metrics["window_count"], "real_count": metrics["real_count"], "fake_count": metrics["fake_count"], "source_count": metrics["source_count"], "dual_role_source_count": metrics["dual_role_source_count"], "source_macro_auroc": metrics["source_macro"].get("mean"), "source_macro_ci_low": ci[0], "source_macro_ci_high": ci[1], "pooled_auroc": metrics.get("pooled_auroc"), "pooled_ap": metrics.get("pooled_ap"), "precision": metrics.get("precision"), "recall": metrics.get("recall"), "f1": metrics.get("f1"), "accuracy": metrics.get("accuracy"), "tn": metrics.get("tn"), "fp": metrics.get("fp"), "fn": metrics.get("fn"), "tp": metrics.get("tp")}


def _identity(manifests: Sequence[Mapping[str, Any]], train_rows: Sequence[Mapping[str, Any]], validation_rows: Sequence[Mapping[str, Any]], condition: str, seed: int, standard_s: FeatureStandardizer, standard_q: FeatureStandardizer, standard3: Mapping[str, Any], standard2: Mapping[str, Any], feature_hash: str) -> dict[str, Any]:
    return {"condition": condition, "seed": int(seed), "feature_content_sha256": feature_hash, "feature_manifest_sha256": hashlib.sha256(json.dumps(manifests, sort_keys=True, separators=(",", ":")).encode()).hexdigest(), "training_window_ids_sha256": hashlib.sha256("\n".join(sorted(str(row["window_id"]) for row in train_rows)).encode()).hexdigest(), "validation_window_ids_sha256": hashlib.sha256("\n".join(sorted(str(row["window_id"]) for row in validation_rows)).encode()).hexdigest(), "standard_s": standard_s.as_dict(), "standard_q": standard_q.as_dict(), "standard_edge3": _std_json(standard3), "standard_edge2": _std_json(standard2), "model_config": {"conditions": list(CONDITIONS), "epochs": EPOCHS, "learning_rate": LEARNING_RATE, "weight_decay": WEIGHT_DECAY}}


def _std_json(value: Mapping[str, Any]) -> dict[str, Any]:
    return {key: (_jsonable(item) if key in ("mean", "scale") else item) for key, item in value.items()}


def _record_matches(record: Mapping[str, Any], expected: Mapping[str, Any]) -> bool:
    actual = record.get("input_identity")
    return isinstance(actual, Mapping) and all(actual.get(key) == value for key, value in expected.items())


def _score_rows(rows: Sequence[Mapping[str, Any]], store: Mapping[tuple[str, str, int], Mapping[str, float]], split: str) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for row in rows:
        item: dict[str, Any] = {"window_id": str(row["window_id"]), "source_id": str(row["source_id"]), "role": str(row["role"]), "label": int(row["label"]), "matched_unit_count": int(row.get("matched_unit_count", 0))}
        for condition in CONDITIONS:
            for seed in SEEDS:
                item[f"{condition}_seed_{seed}"] = store[(split, condition, seed)][str(row["window_id"])]
            item[f"{condition}_MEAN_LOGIT"] = float(np.mean([item[f"{condition}_seed_{seed}"] for seed in SEEDS]))
        output.append(item)
    return output


def evaluate(root: Path, manifests: Sequence[Mapping[str, Any]], train_rows: Sequence[Mapping[str, Any]], validation_rows: Sequence[Mapping[str, Any]], records: Sequence[Mapping[str, Any]], standard_s: FeatureStandardizer, standard_q: FeatureStandardizer, standard3: Mapping[str, Any], standard2: Mapping[str, Any], device: str) -> dict[str, Any]:
    store: dict[tuple[str, str, int], dict[str, float]] = {}
    metric_rows: list[dict[str, Any]] = []
    per_source_rows: list[dict[str, Any]] = []
    for condition in CONDITIONS:
        for record in [item for item in records if str(item["condition"]) == condition]:
            seed = int(record["seed"]); model = _model_from_record(condition, record, device)
            for split, rows in (("train", train_rows), ("validation", validation_rows)):
                batch = _batch(root, manifests, rows, standard_s, standard_q, standard3, standard2, condition)
                with __import__("torch").no_grad():
                    scores = _forward(model, batch, device).detach().cpu().numpy().astype(np.float64)
                mapping = {window_id: float(value) for window_id, value in zip(batch["window_ids"], scores)}
                store[(split, condition, seed)] = mapping
                metrics, _ = _metrics(rows, scores); metric_rows.append(_metric_row(split, condition, seed, metrics))
            del model
    mean_store: dict[tuple[str, str], dict[str, float]] = {}
    for split, rows in (("train", train_rows), ("validation", validation_rows)):
        for condition in CONDITIONS:
            mapping = {str(row["window_id"]): float(np.mean([store[(split, condition, seed)][str(row["window_id"])] for seed in SEEDS])) for row in rows}
            mean_store[(split, condition)] = mapping
            metrics, _ = _metrics(rows, [mapping[str(row["window_id"])] for row in rows]); metric_rows.append(_metric_row(split, condition, "MEAN_LOGIT", metrics))
    source_values: dict[str, dict[str, float]] = {}
    for condition in CONDITIONS:
        source_values[condition] = {}
        mapping = mean_store[("validation", condition)]
        for source in sorted({str(row["source_id"]) for row in validation_rows}):
            subset = [row for row in validation_rows if str(row["source_id"]) == source]
            value = periodic._auroc([int(row["label"]) for row in subset], [mapping[str(row["window_id"])] for row in subset])
            if value is not None:
                source_values[condition][source] = float(value)
    comparisons: list[dict[str, Any]] = []
    for left, right in (("GEOMETRY_3D", "SUMMARY_SET"), ("GEOMETRY_3D", "GEOMETRY_2D")):
        result = source128._bootstrap(source_values[left], source_values[right]); differences = [source_values[left][source] - source_values[right][source] for source in result["sources"]]; result.update({"name": f"{left}-{right}", "left": left, "right": right, "positive_count": sum(value > 0 for value in differences), "negative_count": sum(value < 0 for value in differences), "tie_count": sum(value == 0 for value in differences)})
        comparisons.append(result)
    per_seed_comparisons: list[dict[str, Any]] = []
    for seed in SEEDS:
        seed_values: dict[str, dict[str, float]] = {}
        for condition in CONDITIONS:
            seed_values[condition] = {}
            mapping = store[("validation", condition, seed)]
            for source in sorted({str(row["source_id"]) for row in validation_rows}):
                subset = [row for row in validation_rows if str(row["source_id"]) == source]
                value = periodic._auroc([int(row["label"]) for row in subset], [mapping[str(row["window_id"])] for row in subset])
                if value is not None:
                    seed_values[condition][source] = float(value)
        for left, right in (("GEOMETRY_3D", "SUMMARY_SET"), ("GEOMETRY_3D", "GEOMETRY_2D")):
            keys = sorted(set(seed_values[left]) & set(seed_values[right])); diffs = [seed_values[left][source] - seed_values[right][source] for source in keys]
            per_seed_comparisons.append({"seed": int(seed), "comparison": f"{left}-{right}", "source_count": len(diffs), "mean_source_difference": float(np.mean(diffs)) if diffs else None, "positive_count": sum(value > 0 for value in diffs), "negative_count": sum(value < 0 for value in diffs), "tie_count": sum(value == 0 for value in diffs)})
    for split, rows in (("train", train_rows), ("validation", validation_rows)):
        for source in sorted({str(row["source_id"]) for row in rows}):
            source_rows = [row for row in rows if str(row["source_id"]) == source]
            for condition in CONDITIONS:
                for seed in (*SEEDS, "MEAN_LOGIT"):
                    mapping = mean_store[(split, condition)] if seed == "MEAN_LOGIT" else store[(split, condition, int(seed))]
                    values = [mapping[str(row["window_id"])] for row in source_rows]
                    per_source_rows.append({"split": split, "source_id": source, "condition": condition, "seed": seed, "window_count": len(values), "real_count": sum(int(row["label"]) == 0 for row in source_rows), "fake_count": sum(int(row["label"]) == 1 for row in source_rows), "auroc": periodic._auroc([int(row["label"]) for row in source_rows], values)})
    write_csv(root / "scores/train_window_scores.csv", _score_rows(train_rows, store, "train")); write_csv(root / "scores/validation_window_scores.csv", _score_rows(validation_rows, store, "validation")); write_csv(root / "evaluation/metrics.csv", metric_rows); write_csv(root / "evaluation/per_source_metrics.csv", per_source_rows); write_csv(root / "evaluation/paired_comparisons.csv", [{"comparison": item["name"], "source_count": item["source_count"], "mean": item["mean"], "ci_low": item["ci95"][0], "ci_high": item["ci95"][1], "positive_count": item["positive_count"], "negative_count": item["negative_count"], "tie_count": item["tie_count"]} for item in comparisons]); write_csv(root / "evaluation/per_seed_comparisons.csv", per_seed_comparisons)
    summary = {"train_population": {"windows": len(train_rows), "real": sum(int(row["label"]) == 0 for row in train_rows), "fake": sum(int(row["label"]) == 1 for row in train_rows), "sources": len({str(row["source_id"]) for row in train_rows})}, "validation_population": {"windows": len(validation_rows), "real": sum(int(row["label"]) == 0 for row in validation_rows), "fake": sum(int(row["label"]) == 1 for row in validation_rows), "sources": len({str(row["source_id"]) for row in validation_rows})}, "metrics": metric_rows, "comparisons": comparisons, "per_seed_comparisons": per_seed_comparisons, "device": str(device), "model_count": len(records)}
    atomic_json(root / "evaluation/summary.json", summary)
    return summary


def _smoke(root: Path, manifests: Sequence[Mapping[str, Any]], train_rows: Sequence[Mapping[str, Any]], standard_s: FeatureStandardizer, standard_q: FeatureStandardizer, standard3: Mapping[str, Any], standard2: Mapping[str, Any], device: str) -> dict[str, Any]:
    import torch
    rows = list(train_rows[: min(8, len(train_rows))])
    if {int(row["label"]) for row in rows} != {0, 1}:
        raise ValueError("SMOKE_NEEDS_BOTH_CLASSES")
    results: list[dict[str, Any]] = []
    for condition in CONDITIONS:
        batch = _batch(root, manifests, rows, standard_s, standard_q, standard3, standard2, condition)
        model = _initial_model(condition, SEEDS[0]).to(device)
        model.train(); optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        optimizer.zero_grad(set_to_none=True); scores = _forward(model, batch, device); labels = torch.as_tensor(batch["labels"], dtype=torch.float32, device=device); weights = torch.as_tensor(batch["window_weights"], dtype=torch.float32, device=device); loss = torch.sum(torch.nn.functional.binary_cross_entropy_with_logits(scores, labels, reduction="none") * weights) / torch.sum(weights); loss.backward()
        finite_grads = all(parameter.grad is None or torch.all(torch.isfinite(parameter.grad)) for parameter in model.parameters())
        optimizer.step(); model.eval(); before = _forward(model, batch, device).detach().cpu().numpy()
        reloaded = _model_from_record(condition, {"seed": SEEDS[0], "state_dict": state_dict_numpy(model)}, device); after = _forward(reloaded, batch, device).detach().cpu().numpy()
        permutation_gap = 0.0
        if condition != "SUMMARY_SET":
            permuted = {key: np.array(value, copy=True) for key, value in batch.items()}
            rng = np.random.default_rng(20260909)
            for index in range(permuted["edge_values"].shape[0]):
                for time_index in range(5):
                    order = rng.permutation(permuted["edge_values"].shape[2]); permuted["edge_values"][index, time_index] = permuted["edge_values"][index, time_index, order]; permuted["edge_mask"][index, time_index] = permuted["edge_mask"][index, time_index, order]
            permutation_gap = float(np.max(np.abs(before - _forward(reloaded, permuted, device).detach().cpu().numpy())))
        results.append({"condition": condition, "parameter_count": parameter_count(model), "loss_finite": bool(torch.isfinite(loss).item()), "gradient_finite": bool(finite_grads), "reload_max_abs": float(np.max(np.abs(before - after))), "edge_row_permutation_max_abs": permutation_gap, "passed": bool(torch.isfinite(loss).item() and finite_grads and np.max(np.abs(before - after)) <= 1e-5 and permutation_gap <= 1e-5)})
        del model, reloaded
    result = {"status": "PASS" if all(item["passed"] for item in results) else "FAILED", "rows": len(rows), "conditions": results, "device": str(device)}
    atomic_json(root / "smoke/summary.json", result)
    return result


def _write_protocol(root: Path, info: Mapping[str, Any], matching: Mapping[str, Any], manifests: Sequence[Mapping[str, Any]], standard_s: FeatureStandardizer, standard_q: FeatureStandardizer, standard3: Mapping[str, Any], standard2: Mapping[str, Any]) -> None:
    import torch
    atomic_json(root / "protocol.json", {"protocol_id": "v7-component-learnable-geometry-pilot-v1", "git_head": git_head(), "source_root": str(SOURCE_ROOT), "structure_root": str(STRUCTURE_ROOT), "input": {"train_windows": info["training_windows"], "validation_windows": info["validation_windows"], "matched_train_windows": matching["matched_train_windows"], "matched_validation_windows": matching["matched_validation_windows"], "training_sources": info["training_sources"], "validation_sources": info["validation_sources"], "structure_feature_manifest_sha256": matching["structure_feature_manifest_sha256"], "source_support_sha256": matching["source_support_sha256"], "feature_content_sha256": matching["feature_content_sha256"]}, "support_contract": {"component_members": "existing retained support units/common_member_slots", "same_support": "A/B/C use identical retained units and labels", "fixed_edges": "directed nearest-neighbor edges selected at stored history_reference_array_index; max 8 neighbors per member; distance then slot-ID tie break", "edge_mask": "visibility and geometry_validity plus finite coordinate; invalid padded edges are excluded from means", "center": "mean of finite valid members at each target time; no per-frame scale or registration", "q": "existing four-channel observation-support Q, unchanged and standardized from train only"}, "coordinates": {"xyz": "ParticleSequence camera-motion-compensated meter coordinates", "uv": "ParticleSequence source-pixel x/y coordinates", "geometry_edge": "[x_p-c_i, x_q-x_p, norm]", "image_edge": "[u_p-c_i, u_q-u_p, norm]", "2d_control": "fixed 3-D grouping/edges, not a complete independent 2-D frontend"}, "conditions": {"SUMMARY_SET": "standardized existing S+Q summary with common support", "GEOMETRY_3D": "masked learnable 3-D fixed-edge embedding plus Q", "GEOMETRY_2D": "same encoder dimensions using UV edges plus Q"}, "model": {"geometry_encoder": "shared Linear(E,32)-ReLU-Linear(32,64), masked edge mean, concatenate Q, Linear(68,16)-ReLU-Linear(16,8)-ReLU, five-time mean, 12-16-8-1 head", "summary_encoder": "Linear(8,16)-ReLU-Linear(16,8)-ReLU, five-time mean, 12-16-8-1 head", "embedding_dim": 64, "max_neighbors": MAX_NEIGHBORS, "epochs": EPOCHS, "optimizer": "Adam", "learning_rate": LEARNING_RATE, "weight_decay": WEIGHT_DECAY, "seeds": list(SEEDS), "threshold": "logit >= 0", "loss": "source/class weighted BCE at window level", "parameter_counts": {condition: parameter_count(_initial_model(condition, SEEDS[0])) for condition in CONDITIONS}}, "standardization": {"summary_s": standard_s.as_dict(), "q": standard_q.as_dict(), "edge3d": _std_json(standard3), "edge2d": _std_json(standard2)}, "evaluation": {"primary": "GEOMETRY_3D minus SUMMARY_SET source-macro AUROC", "secondary": "GEOMETRY_3D minus GEOMETRY_2D source-macro AUROC", "bootstrap_seed": BOOTSTRAP_SEED, "bootstrap_replicates": BOOTSTRAP_REPLICATES, "same_support": True, "not_sealed_test": True}, "runtime": {"torch_version": str(torch.__version__), "cuda_available": bool(torch.cuda.is_available()), "cuda_version": str(torch.version.cuda), "device": "cuda" if torch.cuda.is_available() else "cpu"}, "manifest_count": len(manifests)})


def _write_feasibility(root: Path, info: Mapping[str, Any], matching: Mapping[str, Any]) -> None:
    atomic_json(root / "no_tracking_feasibility.json", {"status": "READ_ONLY_AUDIT", "current_input": {"sparse_tracks": 289, "per_frame_uv_xyz": True, "visibility_and_geometry_masks": True, "camera_motion_compensated_xyz": True, "five_time_common_member_support": True}, "whole_frame_dense_point_set": {"available": False, "reason": "current caches contain sparse tracked query slots, not a dense per-pixel depth/intrinsics/pose map with a verified cross-frame scale contract"}, "frontend_dependency": {"tracking_required_now": True, "model_without_ids_possible_in_principle": "only after a new identity-free local spatiotemporal correspondence/anchor contract; current pipeline still uses track IDs/query cohort to form components and five-time support"}, "common_coordinates": {"sparse_particles": "available in camera-motion-compensated meter coordinates", "dense_spatial_anchor": "not available for the full image"}, "labels": {"window_supervision": True, "frame_or_pixel_ground_truth": False, "consequence": "a future frame-output route would be weakly window supervised unless new time/space labels are supplied"}, "minimum_future_interface": ["dense or regular sampled depth with intrinsics and pose per frame", "validity/quality mask and cross-frame scale provenance", "identity-free local spatiotemporal neighborhood builder", "frozen coordinate/anchor contract", "explicit mapping from window labels to frame outputs"], "not_implemented": True, "training_windows": info["training_windows"], "matching_summary": matching})
    (root / "no_tracking_feasibility.md").write_text("""# No-explicit-tracking input feasibility audit\n\n当前缓存能够提供 289 个稀疏查询槽位的逐帧 UV/XYZ、visibility、geometry_validity、相机运动补偿米制坐标，以及由 query cohort/track ID 定义的五时刻共同成员。它不能构成覆盖整幅画面的逐帧稠密有效点集：没有统一保存的逐像素深度、内参、位姿、质量 mask 和跨帧尺度合同。\n\n当前检测模型即使不把 ID 作为数值输入，前端仍依赖跟踪 ID/cohort 来形成 component、固定成员和五时刻支撑。因此“模型不读取 ID”不等于“全系统无跟踪”。在原则上可以做局部时空邻域编码，但需要新的无身份对应/空间锚点接口；本轮没有重建。现有共同坐标只对稀疏 ParticleSequence 有效，不能直接给全画面像素锚定。\n\n现有标签是窗口级 real/fake，不能直接复制成可信帧级或像素级真值。若未来实现逐帧输出，当前数据最多支持弱监督的窗口约束，仍需要明确的时间映射或独立标注。\n\n最小接口草案：逐帧稠密/规则采样深度+内参+位姿；有效性/质量与跨帧尺度 provenance；不依赖持久 track ID 的局部时空邻域构造；冻结坐标/锚点合同；以及窗口标签到帧输出的显式映射。\n\n本审计只读现有缓存，未移除跟踪、未重跑前端、未实现该路线。\n""", encoding="utf-8")


def _write_report(root: Path, info: Mapping[str, Any], matching: Mapping[str, Any], summary: Mapping[str, Any], records: Sequence[Mapping[str, Any]], timings: Mapping[str, float], smoke: Mapping[str, Any]) -> Path:
    metrics = {(str(item["split"]), str(item["condition"])): item for item in summary.get("metrics", []) if str(item["seed"]) == "MEAN_LOGIT"}
    lines = ["# V7 component 可学习几何表示最小验证", "", "本 pilot 固定已有 R query cohort、component/common members、五个目标时刻、Q、标签、source 隔离和窗口平均聚合；只比较现有四维 S+Q 摘要与共享小型局部关系编码。GEOMETRY_2D 使用同一三维分组/固定边，是二维输入对照，不是完整纯二维前端。", "", "## 结果先行", f"- 正式模型：{len(records)}/9；smoke：{smoke.get('status')}；设备：{summary.get('device')}。", f"- 训练窗口 {len(info['training_sources'])} source/{info['training_windows']}（{info['training_real']} real、{info['training_fake']} fake），验证 {len(info['validation_sources'])} source/{info['validation_windows']}（{info['validation_real']} real、{info['validation_fake']} fake）。共同支撑筛选后训练 {matching['matched_train_windows']}、验证 {matching['matched_validation_windows']} 窗口；unit {matching['matched_train_units']}/{matching['matched_validation_units']}。", "", "| 条件 | source-macro AUROC (95% CI) | pooled AUROC | AP | Precision | Recall | F1 | ACC | TN/FP/FN/TP |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for condition in CONDITIONS:
        item = metrics.get(("validation", condition), {}); lines.append(f"| {condition} | {item.get('source_macro_auroc')} [{item.get('source_macro_ci_low')}, {item.get('source_macro_ci_high')}] | {item.get('pooled_auroc')} | {item.get('pooled_ap')} | {item.get('precision')} | {item.get('recall')} | {item.get('f1')} | {item.get('accuracy')} | {item.get('tn')}/{item.get('fp')}/{item.get('fn')}/{item.get('tp')} |")
    lines += ["", "## 预声明配对", "", "| 比较 | source 数 | 均值差 | 95% CI | 正/负/平 source |", "|---|---:|---:|---|---:|"]
    for item in summary.get("comparisons", []):
        lines.append(f"| {item['name']} | {item['source_count']} | {item['mean']} | [{item['ci95'][0]}, {item['ci95'][1]}] | {item['positive_count']}/{item['negative_count']}/{item['tie_count']} |")
    lines += ["", "## 解释", "", "- GEOMETRY_3D−SUMMARY_SET 检验可学习局部三维边关系是否在本匹配边界提供额外信息；B>A 仍可能包含编码器容量/归纳偏置，不能单独证明四维摘要是唯一瓶颈。", "- GEOMETRY_3D−GEOMETRY_2D 只是在固定三维分组和相同边身份下的二维/三维输入对照。它不是完整二维前端，也不能据此声明三维系统整体优于二维。", "- 中心化移除整体平移；原始向量保留方向，未宣称旋转不变。没有逐帧独立缩放或刚体配准。无效边由 mask 排除，未用零坐标充当观测。", "- 训练与验证指标是窗口级开发集结果；不能推出组间动态、时序机制、视觉互补、独立泛化或像素定位。", "", "## 覆盖与成本", f"- matching={timings.get('matching')}s, smoke={timings.get('smoke')}s, train={timings.get('train')}s, evaluate={timings.get('evaluate')}s, report={timings.get('report')}s。这里的阶段计时是最后一次 runner 调用；`models/fold_models.json` 才是正式训练完成证据（9/9、每个 200 epochs）。前面为修复显存峰值发生过 OOM/中断，精确累计墙钟未持久化，不能把本次 resume 的 train 秒数冒充完整训练成本。", "- 详细训练 loss 与模型记录见 `models/fold_models.json`；逐窗口分数见 `scores/`; 指标与 source/seed 比较见 `evaluation/`。", "- 固定边和 unit 丢失原因见 `support/matching_summary.json`、`support/matching_coverage.csv`；输入 hash 和协议见 `protocol.json`。", "- 未重跑 tracking/depth/pose/segmentation，未访问旧 R7/V5；前端仍是稀疏跟踪依赖。", "", "## 无显式跟踪路线", "详见 `no_tracking_feasibility.md`。当前没有覆盖全画面的稠密逐帧三维输入、无统一 dense anchor/scale contract，不能把本 pilot 当作无跟踪路线验证。", ""]
    path = root / "report.md"; path.write_text("\n".join(lines), encoding="utf-8"); return path


def run(root: Path = OUTPUT_ROOT, *, device: str = "cuda", resume: bool = True) -> dict[str, Any]:
    root.mkdir(parents=True, exist_ok=True); (root / "state").mkdir(parents=True, exist_ok=True); started = time.perf_counter(); timings: dict[str, float] = {}
    atomic_json(root / "state/launch.json", {"git_head": git_head(), "pid": os.getpid(), "device": device, "started_unix": time.time()})
    try:
        train_original, validation_original, _, info = _load_selection()
        matching_started = time.perf_counter(); train_rows, validation_rows, matching, manifests = _build_features(root, train_original, validation_original, resume=resume); timings["matching"] = time.perf_counter() - matching_started
        if not train_rows or not validation_rows:
            raise RuntimeError("MATCHED_TRAIN_OR_VALIDATION_EMPTY")
        manifest_by_id = {str(item["window_id"]): item for item in manifests}
        train_manifests = [manifest_by_id[str(row["window_id"])] for row in train_rows]
        standard_s = fit_standardizer("SET_A", [_load_npz(root / str(item["input_path"]))["s"] for item in train_manifests], source_class_weights(train_rows))
        standard_q = fit_standardizer("SET_A", [_load_npz(root / str(item["input_path"]))["q"] for item in train_manifests], source_class_weights(train_rows))
        standard3 = _fit_edge_standardizer(root, manifests, train_rows, "edge3d"); standard2 = _fit_edge_standardizer(root, manifests, train_rows, "edge2d")
        _write_protocol(root, info, matching, manifests, standard_s, standard_q, standard3, standard2); _write_feasibility(root, info, matching)
        smoke_started = time.perf_counter(); smoke = json.loads((root / "smoke/summary.json").read_text(encoding="utf-8")) if resume and (root / "smoke/summary.json").is_file() else _smoke(root, manifests, train_rows, standard_s, standard_q, standard3, standard2, device); timings["smoke"] = time.perf_counter() - smoke_started
        if smoke.get("status") != "PASS":
            raise RuntimeError("SMOKE_FAILED")
        model_path = root / "models/fold_models.json"; existing = json.loads(model_path.read_text(encoding="utf-8")).get("records", []) if resume and model_path.is_file() else []; records: list[dict[str, Any]] = []; complete: set[tuple[str, int]] = set(); feature_hash = _content_hash(root, manifests)
        for old in existing:
            condition, seed = str(old.get("condition", "")), int(old.get("seed", -1))
            if condition not in CONDITIONS or seed not in SEEDS:
                continue
            expected = _identity(manifests, train_rows, validation_rows, condition, seed, standard_s, standard_q, standard3, standard2, feature_hash)
            if old.get("status") == "TRAIN_COMPLETE" and _record_matches(old, expected):
                records.append(old); complete.add((condition, seed))
        train_started = time.perf_counter(); total = len(CONDITIONS) * len(SEEDS)
        for condition in CONDITIONS:
            batch = _batch(root, manifests, train_rows, standard_s, standard_q, standard3, standard2, condition)
            for seed in SEEDS:
                if (condition, int(seed)) in complete:
                    continue
                model, fit = _train_one(condition, batch, int(seed), device)
                record = {"condition": condition, "seed": int(seed), "status": "TRAIN_COMPLETE", "parameter_count": parameter_count(model), "fit": fit, "input_identity": _identity(manifests, train_rows, validation_rows, condition, int(seed), standard_s, standard_q, standard3, standard2, feature_hash), "state_dict": state_dict_numpy(model)}
                records.append(record); complete.add((condition, int(seed))); atomic_json(model_path, {"conditions": list(CONDITIONS), "seeds": list(SEEDS), "epochs": EPOCHS, "records": records}); progress(root, "train", "RUNNING", len(complete), total, condition=condition, seed=seed); del model
        timings["train"] = time.perf_counter() - train_started
        if len(complete) != total:
            raise RuntimeError(f"MODEL_COUNT_INCOMPLETE:{len(complete)}/{total}")
        evaluate_started = time.perf_counter(); summary = evaluate(root, manifests, train_rows, validation_rows, records, standard_s, standard_q, standard3, standard2, device); timings["evaluate"] = time.perf_counter() - evaluate_started
        report_started = time.perf_counter(); report_path = _write_report(root, info, matching, summary, records, timings, smoke); timings["report"] = time.perf_counter() - report_started; _write_report(root, info, matching, summary, records, timings, smoke)
        atomic_json(root / "final_status.json", {"status": "COMPLETE", "model_count": len(records), "expected_model_count": total, "train_windows": len(train_rows), "validation_windows": len(validation_rows), "report": str(report_path), "timings_s": timings, "git_head": git_head()}); progress(root, "report", "COMPLETE", 1, 1, model_count=len(records), elapsed_s=time.perf_counter() - started); return {"status": "COMPLETE", "report": str(report_path), "summary": summary, "timings_s": timings}
    except BaseException as exc:
        atomic_json(root / "state/failure.json", {"status": "FAILED", "error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()}); atomic_json(root / "final_status.json", {"status": "FAILED", "error": f"{type(exc).__name__}: {exc}", "git_head": git_head(), "elapsed_s": time.perf_counter() - started}); progress(root, "failed", "FAILED", 0, 1, error=f"{type(exc).__name__}: {exc}"); raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT); parser.add_argument("--device", default="cuda"); parser.add_argument("--resume", action="store_true"); args = parser.parse_args(); result = run(args.output_root, device=args.device, resume=args.resume); print(json.dumps({"status": result["status"], "report": result["report"], "timings_s": result["timings_s"]}, ensure_ascii=False, indent=2)); return 0


if __name__ == "__main__":
    raise SystemExit(main())
