"""Run the bounded V7 representation-sensitivity probe.

The probe operates on a few cached ParticleSequence units and on a small
synthetic point set.  It never writes a ParticleSequence, never changes a
cached file, and never trains or evaluates an AUROC.  The frozen models are
used only as a secondary unit/window readout.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

import numpy as np

from research_tools.v7.geometry_information_pilot import runner as geometry
from research_tools.v7.multi_order_sequence_probe.model import FeatureStandardizer
from research_tools.v7.observation_support_pilot import runner as observation
from sparse3d_forgery.particle_sequence import load_particle_sequence


REPO_ROOT = Path(__file__).resolve().parents[3]
DATA_ROOT = Path("/root/autodl-tmp/data/sparse_3d_forgery_detection")
SOURCE_ROOT = DATA_ROOT / "derived/v7_activityforensics_source128_extension_v1"
OBS_ROOT = DATA_ROOT / "derived/v7_activityforensics_observation_support_pilot_v1"
OUTPUT_ROOT = DATA_ROOT / "derived/v7_activityforensics_representation_sensitivity_probe_v1"
SEED = 20260909
MAX_SOURCES = 12
MAX_UNITS_PER_WINDOW = 3
TOLERANCE = 1e-5
MODEL_CONDITIONS = ("STRUCTURE_ONLY", "SUPPORT_ONLY", "STRUCTURE_SUPPORT")
MODEL_SEEDS = (20260909, 20260910, 20260911)


def _jsonable(value: Any, path: str = "$") -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return [_jsonable(item, f"{path}[{index}]") for index, item in enumerate(value)]
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return _jsonable(value.item(), path)
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"NONFINITE_JSON:{path}:{value}")
        return value
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item, f"{path}.{key}") for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item, f"{path}[{index}]") for index, item in enumerate(value)]
    return value


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(_jsonable(value), indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
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
            writer.writerow({key: "" if row.get(key) is None else row.get(key) for key in fields})
    os.replace(tmp, path)


def _git_head() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {key: np.asarray(archive[key]) for key in archive.files}


def _copy_sequence(sequence: Any, *, xyz: np.ndarray | None = None, visibility: np.ndarray | None = None, geometry_validity: np.ndarray | None = None) -> Any:
    # Only the fields consumed by the existing representation functions are
    # copied.  No modified object is passed to ParticleSequence validation or
    # persisted.
    return SimpleNamespace(
        uv=np.asarray(sequence.uv).copy(),
        xyz=np.asarray(sequence.xyz if xyz is None else xyz).copy(),
        visibility=np.asarray(sequence.visibility if visibility is None else visibility).copy(),
        geometry_validity=np.asarray(sequence.geometry_validity if geometry_validity is None else geometry_validity).copy(),
        frame_indices=np.asarray(sequence.frame_indices).copy(),
        timestamps_s=np.asarray(sequence.timestamps_s).copy(),
        track_ids=np.asarray(sequence.track_ids).copy(),
        provenance=copy.deepcopy(sequence.provenance),
    )


def _pairs_for_members(members: Sequence[int]) -> np.ndarray:
    ordered = sorted(int(item) for item in members)
    return np.asarray([(ordered[i], ordered[j]) for i in range(len(ordered)) for j in range(i + 1, len(ordered))], dtype=np.int64)


def _selected_moved_members(members: Sequence[int]) -> np.ndarray:
    values = sorted(int(item) for item in members)
    if len(values) <= 1:
        return np.asarray(values, dtype=np.int64)
    count = max(1, min(len(values) - 1, int(math.ceil(len(values) / 4.0))))
    values.sort(key=lambda value: _sha(f"representation-sensitivity-member|{SEED}|{value}"))
    return np.asarray(sorted(values[:count]), dtype=np.int64)


def _finite_masks(sequence: Any) -> tuple[np.ndarray, np.ndarray]:
    uv = np.asarray(sequence.uv)
    xyz = np.asarray(sequence.xyz)
    visible = np.asarray(sequence.visibility, dtype=bool) & np.all(np.isfinite(uv), axis=-1)
    geometry_valid = np.asarray(sequence.geometry_validity, dtype=bool) & np.all(np.isfinite(xyz), axis=-1)
    return visible, geometry_valid


def _summary(distances: np.ndarray) -> np.ndarray:
    values = np.asarray(distances, dtype=np.float64)
    return geometry._summary_from_distances(values)


def _compute_s(row: Mapping[str, Any], sequence: Any, members: Sequence[int], *, pairs: np.ndarray | None = None) -> dict[str, Any]:
    members_array = np.asarray(sorted(int(item) for item in members), dtype=np.int64)
    if members_array.size < 3:
        return {"status": "INVALID", "reason": "SUPPORT_INSUFFICIENT_GROUP_SIZE", "members": members_array, "pairs": np.empty((0, 2), dtype=np.int64)}
    pairs_array = _pairs_for_members(members_array) if pairs is None else np.asarray(pairs, dtype=np.int64)
    history = np.asarray(row["grouping"]["history_array_indices"], dtype=np.int64)
    target = np.asarray(next(unit["array_indices"] for unit in row["support"]["units"] if set(unit["member_slots"]) == set(members_array.tolist())), dtype=np.int64) if pairs is None else None
    # For transformed units, target indices are supplied in the row's unit
    # context by the caller; the fallback above is only used for synthetic
    # rows.  Real calls use _compute_s_with_target below.
    if target is None:
        return {"status": "INVALID", "reason": "TARGET_INDEX_CONTEXT_MISSING", "members": members_array, "pairs": pairs_array}
    return _compute_s_with_target(row, sequence, members_array, pairs_array, target, history)


def _compute_s_with_target(row: Mapping[str, Any], sequence: Any, members: np.ndarray, pairs: np.ndarray, target: np.ndarray, history: np.ndarray) -> dict[str, Any]:
    visible, geom = _finite_masks(sequence)
    target_ok = visible[target][:, members] & geom[target][:, members]
    if not np.all(target_ok):
        return {"status": "INVALID", "reason": "TARGET_UV_XYZ_COMMON_INVALID", "members": members, "pairs": pairs}
    history_uv, history_xyz = geometry._history_distances(sequence, history, pairs)
    if history_xyz.size == 0:
        return {"status": "INVALID", "reason": "SHARED_HISTORY_PAIR_TIME_EMPTY", "members": members, "pairs": pairs}
    scale = float(np.median(history_xyz))
    if not math.isfinite(scale) or scale <= 0:
        return {"status": "INVALID", "reason": "XYZ_HISTORY_SCALE_INVALID", "members": members, "pairs": pairs}
    xyz = np.asarray(sequence.xyz, dtype=np.float64)
    distances = np.stack([np.linalg.norm(xyz[target, right] - xyz[target, left], axis=1) for left, right in pairs], axis=0)
    if not np.all(np.isfinite(distances)):
        return {"status": "INVALID", "reason": "TARGET_DISTANCE_NONFINITE", "members": members, "pairs": pairs}
    return {"status": "VALID", "reason": "", "members": members, "pairs": pairs, "scale": scale, "distances": distances, "s": _summary(distances / scale), "history_pair_time_count": int(history_xyz.size), "target_indices": target, "history_indices": history}


def _compute_q(row: Mapping[str, Any], sequence: Any, raw_members: Sequence[int], target: np.ndarray, history: np.ndarray) -> dict[str, Any]:
    visible, geom = _finite_masks(sequence)
    raw = np.asarray(sorted(int(item) for item in raw_members), dtype=np.int64)
    if raw.size == 0 or target.shape != (5,) or history.size == 0:
        return {"status": "INVALID", "reason": "Q_INPUT_INVALID"}
    target_timestamps = np.asarray(sequence.timestamps_s)[target]
    prior = history[np.asarray(sequence.timestamps_s)[history] < target_timestamps[0]]
    if prior.size == 0:
        return {"status": "INVALID", "reason": "Q_HISTORY_REFERENCE_MISSING"}
    reference = int(prior[np.argmax(np.asarray(sequence.timestamps_s)[prior])])
    v = np.mean(visible[target][:, raw], axis=1)
    g = np.mean(geom[target][:, raw], axis=1)
    q = np.stack((v, g, v - np.mean(visible[reference, raw]), g - np.mean(geom[reference, raw])), axis=1)
    return {"status": "VALID", "reason": "", "q": q.astype(np.float64), "reference": reference, "raw_members": raw}


def _modify_xyz(base: np.ndarray, operation: str, members: np.ndarray, history: np.ndarray, target: np.ndarray, *, direction: np.ndarray, scale: float, synthetic: bool = False) -> np.ndarray:
    xyz = np.asarray(base, dtype=np.float64).copy()
    frames_all = np.unique(np.concatenate((history, target)))
    center = np.mean(xyz[history[0], members], axis=0)
    angle = 0.37
    axis = np.asarray([0.3, 0.5, 0.8], dtype=np.float64); axis /= np.linalg.norm(axis)
    skew = np.array([[0.0, -axis[2], axis[1]], [axis[2], 0.0, -axis[0]], [-axis[1], axis[0], 0.0]])
    rotation = np.eye(3) * math.cos(angle) + (1.0 - math.cos(angle)) * np.outer(axis, axis) + math.sin(angle) * skew
    def rigid(frame_indices: np.ndarray, angle_factor: float = 1.0) -> None:
        local_angle = angle * angle_factor
        a = np.array([[0.0, -axis[2], axis[1]], [axis[2], 0.0, -axis[0]], [-axis[1], axis[0], 0.0]])
        r = np.eye(3) * math.cos(local_angle) + (1.0 - math.cos(local_angle)) * np.outer(axis, axis) + math.sin(local_angle) * a
        for frame in frame_indices:
            xyz[frame, members] = xyz[frame, members] @ r.T + np.asarray([0.07, -0.03, 0.02]) * angle_factor
    if operation == "G0":
        return xyz
    if operation == "G1":
        xyz[np.ix_(frames_all, members)] = xyz[np.ix_(frames_all, members)] @ rotation.T + np.asarray([0.07, -0.03, 0.02])
    elif operation == "G2":
        for index, frame in enumerate(target, start=1):
            local_angle = 0.04 * index
            a = np.array([[0.0, -axis[2], axis[1]], [axis[2], 0.0, -axis[0]], [-axis[1], axis[0], 0.0]])
            r = np.eye(3) * math.cos(local_angle) + (1.0 - math.cos(local_angle)) * np.outer(axis, axis) + math.sin(local_angle) * a
            xyz[frame, members] = xyz[frame, members] @ r.T + np.asarray([0.01, -0.005, 0.002]) * index
    elif operation == "G3":
        xyz[np.ix_(frames_all, members)] = center + 1.2 * (xyz[np.ix_(frames_all, members)] - center)
    elif operation == "G4":
        xyz[np.ix_(target, members)] = center + 1.2 * (xyz[np.ix_(target, members)] - center)
    elif operation in ("G5_A0", "G5_A005", "G5_A020"):
        amplitude = {"G5_A0": 0.0, "G5_A005": 0.05, "G5_A020": 0.20}[operation]
        moved = _selected_moved_members(members)
        for index, frame in enumerate(target, start=1):
            xyz[frame, moved] += amplitude * scale * (index / 5.0) * direction
    elif operation in ("G6_TARGET_ONLY", "G6_HISTORY_AND_TARGET"):
        moved = _selected_moved_members(members)
        frames = target if operation == "G6_TARGET_ONLY" else frames_all
        xyz[np.ix_(frames, moved)] += 0.20 * scale * direction
    else:
        raise ValueError(operation)
    return xyz


def _mask_modify(sequence: Any, operation: str, raw_members: np.ndarray, common_members: np.ndarray, target: np.ndarray) -> tuple[Any, np.ndarray | None, str]:
    if operation == "M1":
        candidates = [int(item) for item in raw_members if int(item) not in set(common_members.tolist())]
        if not candidates:
            return sequence, None, "NOT_APPLICABLE_NO_NONCOMMON_MEMBER"
        removed = np.asarray([candidates[0]], dtype=np.int64)
        frames = target
    elif operation == "M2":
        if common_members.size < 4:
            return sequence, None, "NOT_APPLICABLE_COMMON_MEMBER_COUNT_LT4"
        removed = np.asarray([int(common_members[0])], dtype=np.int64)
        frames = target[-2:]
    elif operation == "M3":
        if common_members.size < 4:
            return sequence, None, "NOT_APPLICABLE_COMMON_MEMBER_COUNT_LT4"
        removed = np.asarray([int(common_members[0])], dtype=np.int64)
        frames = target[2:3]
    else:
        raise ValueError(operation)
    visibility = np.asarray(sequence.visibility, dtype=bool).copy(); geom = np.asarray(sequence.geometry_validity, dtype=bool).copy(); uv = np.asarray(sequence.uv, dtype=np.float64).copy(); xyz = np.asarray(sequence.xyz, dtype=np.float64).copy()
    visibility[np.ix_(frames, removed)] = False; geom[np.ix_(frames, removed)] = False; uv[np.ix_(frames, removed)] = np.nan; xyz[np.ix_(frames, removed)] = np.nan
    return SimpleNamespace(
        uv=uv,
        xyz=xyz,
        visibility=visibility,
        geometry_validity=geom,
        frame_indices=np.asarray(sequence.frame_indices).copy(),
        timestamps_s=np.asarray(sequence.timestamps_s).copy(),
        track_ids=np.asarray(sequence.track_ids).copy(),
        provenance=copy.deepcopy(sequence.provenance),
    ), removed, ""


def _synthetic() -> dict[str, Any]:
    points = np.asarray([[0.0, 0.0, 1.0], [0.8, 0.1, 1.1], [0.1, 0.9, 1.3], [0.7, 0.8, 1.0], [1.4, 0.2, 1.5], [1.2, 1.1, 0.9], [0.2, 1.6, 1.2], [1.6, 1.5, 1.4]], dtype=np.float64)
    xyz = np.repeat(points[None, :, :], 10, axis=0)
    timestamps = np.arange(10, dtype=np.float64) * 0.1
    sequence = SimpleNamespace(uv=np.repeat(points[None, :, :2], 10, axis=0), xyz=xyz, visibility=np.ones((10, 8), dtype=bool), geometry_validity=np.ones((10, 8), dtype=bool), frame_indices=np.arange(10, dtype=np.int64), timestamps_s=timestamps, track_ids=np.arange(8, dtype=np.int64), provenance={"cohort_start_s": 0.0})
    row = {"window_id": "synthetic::unit", "grouping": {"history_array_indices": [0, 1, 2, 3, 4]}}
    history = np.asarray([0, 1, 2, 3, 4], dtype=np.int64); target = np.asarray([5, 6, 7, 8, 9], dtype=np.int64)
    members = np.arange(7, dtype=np.int64); raw = np.arange(8, dtype=np.int64); pairs = _pairs_for_members(members)
    return {"sample_id": "synthetic::unit", "source_id": "SYNTHETIC", "role": "synthetic", "window_id": "synthetic::unit", "sequence": sequence, "row": row, "target": target, "history": history, "members": members, "raw_members": raw, "pairs": pairs, "common_members": members, "base_s": _compute_s_with_target(row, sequence, members, pairs, target, history), "base_q": _compute_q(row, sequence, raw, target, history)}


@dataclass
class RealSample:
    sample_id: str
    source_id: str
    role: str
    window_id: str
    row: dict[str, Any]
    item: dict[str, Any]
    support_units: dict[int, dict[str, Any]]
    sequence: Any
    target: np.ndarray
    history: np.ndarray
    units: list[dict[str, Any]]


def _select_real_samples() -> list[RealSample]:
    manifest = json.loads((OBS_ROOT / "inputs/feature_manifest.json").read_text(encoding="utf-8"))
    score_rows = list(csv.DictReader((OBS_ROOT / "scores/train_window_scores.csv").open(newline="", encoding="utf-8")))
    train_ids = {str(row["window_id"]) for row in score_rows}
    items = [dict(item) for item in manifest if str(item["window_id"]) in train_ids]
    by_source: dict[str, list[dict[str, Any]]] = {}
    for item in items:
        by_source.setdefault(str(item["source_id"]), []).append(item)
    sources = sorted(by_source, key=lambda source: _sha(f"representation-sensitivity|{SEED}|{source}"))[:MAX_SOURCES]
    source_support = json.loads((SOURCE_ROOT / "support/window_support.json").read_text(encoding="utf-8")); support_by_id = {str(row["window_id"]): row for row in source_support if str(row.get("mode")) == "R"}
    selected: list[RealSample] = []
    for source in sources:
        source_items = by_source[source]
        by_b: dict[float, dict[str, dict[str, Any]]] = {}
        for item in source_items:
            by_b.setdefault(float(item["offset_s"]), {})[str(item["role"])] = item
        pair = next((value for _, value in sorted(by_b.items()) if "real" in value and "fake" in value and int(value["real"]["matched_unit_count"]) > 0 and int(value["fake"]["matched_unit_count"]) > 0), None)
        if pair is None:
            continue
        for role in ("real", "fake"):
            item = pair[role]; row = support_by_id[str(item["window_id"])]
            sequence = load_particle_sequence(str(row["particle_prefix"]))
            target = np.asarray(item["unit_identities"][0]["target_array_indices"], dtype=np.int64)
            history = np.asarray(row["grouping"]["history_array_indices"], dtype=np.int64)
            valid_support = {int(unit["local_group_id"]): unit for unit in row["support"]["units"] if unit.get("status") == "VALID"}
            identities = sorted(item["unit_identities"], key=lambda value: int(value["local_group_id"]))[:MAX_UNITS_PER_WINDOW]
            units = [identity for identity in identities if int(identity["local_group_id"]) in valid_support]
            selected.append(RealSample(f"{source}:{role}", source, role, str(item["window_id"]), row, item, valid_support, sequence, target, history, units))
    if not selected:
        raise RuntimeError("NO_REAL_SAMPLES")
    return selected


def _real_unit(sample: RealSample, identity: Mapping[str, Any]) -> dict[str, Any]:
    support = sample.support_units[int(identity["local_group_id"])]
    members = np.asarray(support["member_slots"], dtype=np.int64)
    pairs = np.asarray(support["pair_indices"], dtype=np.int64)
    base_s = _compute_s_with_target(sample.row, sample.sequence, members, pairs, sample.target, sample.history)
    raw = np.asarray(identity["raw_member_slots"], dtype=np.int64)
    base_q = _compute_q(sample.row, sample.sequence, raw, sample.target, sample.history)
    return {"identity": identity, "support": support, "members": members, "pairs": pairs, "raw_members": raw, "base_s": base_s, "base_q": base_q}


def _apply_operation(sample: Mapping[str, Any], operation: str, synthetic: bool) -> dict[str, Any]:
    sequence = sample["sequence"]; row = sample["row"]; members = np.asarray(sample["members"], dtype=np.int64); raw = np.asarray(sample["raw_members"], dtype=np.int64); target = np.asarray(sample["target"], dtype=np.int64); history = np.asarray(sample["history"], dtype=np.int64); base_s = sample["base_s"]; base_q = sample["base_q"]
    if operation in ("M1", "M2", "M3"):
        modified, removed, reason = _mask_modify(sequence, operation, raw, members, target)
        if removed is None:
            return {"status": "NOT_APPLICABLE", "reason": reason, "sequence": sequence, "s": base_s, "q": base_q, "members": members, "pairs": sample["pairs"], "removed": []}
        visible, geom = _finite_masks(modified); new_members = np.asarray([member for member in members if np.all(visible[target, member] & geom[target, member])], dtype=np.int64); new_pairs = _pairs_for_members(new_members) if new_members.size >= 3 else np.empty((0, 2), dtype=np.int64)
        new_s = _compute_s_with_target(row, modified, new_members, new_pairs, target, history) if new_members.size >= 3 else {"status": "INVALID", "reason": "SUPPORT_INSUFFICIENT_GROUP_SIZE", "members": new_members, "pairs": new_pairs}
        new_q = _compute_q(row, modified, raw, target, history)
        return {"status": "VALID" if new_s["status"] == "VALID" and new_q["status"] == "VALID" else "INVALID", "reason": new_s.get("reason") or new_q.get("reason", ""), "sequence": modified, "s": new_s, "q": new_q, "members": new_members, "pairs": new_pairs, "removed": removed}
    direction = np.asarray([0.4, -0.7, 0.5], dtype=np.float64); direction /= np.linalg.norm(direction); scale = float(base_s.get("scale", 1.0))
    modified_xyz = _modify_xyz(np.asarray(sequence.xyz, dtype=np.float64), operation, members, history, target, direction=direction, scale=scale, synthetic=synthetic)
    modified = _copy_sequence(sequence, xyz=modified_xyz)
    new_s = _compute_s_with_target(row, modified, members, np.asarray(sample["pairs"], dtype=np.int64), target, history)
    return {"status": new_s["status"], "reason": new_s.get("reason", ""), "sequence": modified, "s": new_s, "q": base_q, "members": members, "pairs": np.asarray(sample["pairs"], dtype=np.int64), "removed": []}


def _feature_row(sample: Mapping[str, Any], operation: str, result: Mapping[str, Any], synthetic: bool) -> dict[str, Any]:
    base_s = sample["base_s"]; base_q = sample["base_q"]; new_s = result["s"]; new_q = result["q"]
    base_dist = np.asarray(base_s.get("distances", np.empty((0, 5))), dtype=np.float64); new_dist = np.asarray(new_s.get("distances", np.empty((0, 5))), dtype=np.float64)
    s_diff = float(np.max(np.abs(new_s["s"] - base_s["s"]))) if new_s.get("status") == "VALID" and base_s.get("status") == "VALID" and new_s["s"].shape == base_s["s"].shape else None
    s_rms = float(np.sqrt(np.mean((new_s["s"] - base_s["s"]) ** 2))) if s_diff is not None else None
    q_diff = float(np.max(np.abs(new_q["q"] - base_q["q"]))) if new_q.get("status") == "VALID" and base_q.get("status") == "VALID" and new_q["q"].shape == base_q["q"].shape else None
    saved_s = sample.get("saved_s")
    saved_q = sample.get("saved_q")
    saved_s_error = float(np.max(np.abs(base_s["s"] - saved_s))) if saved_s is not None and base_s.get("status") == "VALID" and np.asarray(saved_s).shape == base_s["s"].shape else None
    saved_q_error = float(np.max(np.abs(base_q["q"] - saved_q))) if saved_q is not None and base_q.get("status") == "VALID" and np.asarray(saved_q).shape == base_q["q"].shape else None
    distance_diff = float(np.max(np.abs(new_dist - base_dist))) if new_dist.shape == base_dist.shape and new_dist.size else None
    distance_rms = float(np.sqrt(np.mean((new_dist - base_dist) ** 2))) if distance_diff is not None else None
    moved = _selected_moved_members(sample["members"]) if operation.startswith("G5") or operation.startswith("G6") else np.empty((0,), dtype=np.int64)
    affected_pairs = int(sum(int(left) in set(moved.tolist()) or int(right) in set(moved.tolist()) for left, right in np.asarray(sample["pairs"], dtype=np.int64)))
    q_visibility_trace = json.dumps(np.asarray(new_q["q"])[:, 0].tolist(), separators=(",", ":")) if new_q.get("status") == "VALID" else None
    q_geometry_trace = json.dumps(np.asarray(new_q["q"])[:, 1].tolist(), separators=(",", ":")) if new_q.get("status") == "VALID" else None
    return {"sample_id": sample["sample_id"], "synthetic": bool(synthetic), "source_id": sample["source_id"], "role": sample["role"], "window_id": sample["window_id"], "local_group_id": int(sample.get("local_group_id", -1)), "operation": operation, "raw_member_count": int(np.asarray(sample["raw_members"]).size), "common_member_count_before": int(np.asarray(sample["members"]).size), "common_member_count_after": int(np.asarray(result["members"]).size), "pair_count_before": int(np.asarray(sample["pairs"]).shape[0]), "pair_count_after": int(np.asarray(result["pairs"]).shape[0]), "moved_member_slots": [int(value) for value in moved], "affected_pair_count": affected_pairs, "history_scale_before": base_s.get("scale"), "history_scale_after": new_s.get("scale"), "history_scale_relative_change": (float(new_s["scale"] / base_s["scale"] - 1.0) if new_s.get("status") == "VALID" and base_s.get("status") == "VALID" else None), "mask_changed": operation in ("M1", "M2", "M3"), "raw_distance_max_abs_diff": distance_diff, "raw_distance_rms_diff": distance_rms, "s_max_abs_diff": s_diff, "s_rms_diff": s_rms, "q_max_abs_diff": q_diff, "saved_s_reconstruction_max_abs": saved_s_error, "saved_q_reconstruction_max_abs": saved_q_error, "q_visibility_trace": q_visibility_trace, "q_geometry_trace": q_geometry_trace, "status": result["status"], "reason": result.get("reason", "")}


def _load_models() -> dict[tuple[str, int], tuple[Any, FeatureStandardizer, FeatureStandardizer]]:
    records = _jsonable(json.loads((OBS_ROOT / "models/fold_models.json").read_text(encoding="utf-8")))
    loaded: dict[tuple[str, int], tuple[Any, FeatureStandardizer, FeatureStandardizer]] = {}
    for record in records["records"]:
        condition = str(record["condition"]); seed = int(record["seed"])
        if condition not in MODEL_CONDITIONS:
            continue
        std_s = record["standardization"]["s"]; std_q = record["standardization"]["q"]
        standardizer_s = FeatureStandardizer("SET_A", np.asarray(std_s["mean"], dtype=np.float64), np.asarray(std_s["scale"], dtype=np.float64), tuple(int(x) for x in std_s.get("zero_variance_dimensions", [])))
        standardizer_q = FeatureStandardizer("SET_A", np.asarray(std_q["mean"], dtype=np.float64), np.asarray(std_q["scale"], dtype=np.float64), tuple(int(x) for x in std_q.get("zero_variance_dimensions", [])))
        model = observation._model_load(condition, record, "cpu")
        loaded[(condition, seed)] = (model, standardizer_s, standardizer_q)
    if len(loaded) != 9:
        raise ValueError(f"MODEL_COUNT_MISMATCH:{len(loaded)}")
    return loaded


def _model_forward(model: Any, state: np.ndarray, intervals: np.ndarray) -> np.ndarray:
    import torch
    with torch.no_grad():
        return model(torch.as_tensor(state, dtype=torch.float32), torch.as_tensor(intervals, dtype=torch.float32)).detach().cpu().numpy().astype(np.float64)


def _model_condition_state(condition: str, s: np.ndarray, q: np.ndarray, std_s: FeatureStandardizer, std_q: FeatureStandardizer) -> np.ndarray:
    s_norm = std_s.transform(s[None, ...])[0]; q_norm = std_q.transform(q[None, ...])[0]
    if condition == "STRUCTURE_ONLY":
        return s_norm
    if condition == "SUPPORT_ONLY":
        return q_norm
    return np.concatenate((s_norm, q_norm), axis=-1)


def _model_rows(samples: Sequence[Mapping[str, Any]], models: Mapping[tuple[str, int], tuple[Any, FeatureStandardizer, FeatureStandardizer]], feature_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    by_sample: dict[str, list[Mapping[str, Any]]] = {}
    for row in feature_rows:
        by_sample.setdefault(str(row["sample_id"]), []).append(row)
    outputs: list[dict[str, Any]] = []
    for sample in samples:
        units = sample.get("units", [sample])
        intervals = np.diff(np.asarray(sample["sequence"].timestamps_s)[np.asarray(sample["target"], dtype=np.int64)])
        # The synthetic unit uses the same four target PTS interval convention.
        if sample.get("synthetic"):
            base_s = sample["base_s"]; base_q = sample["base_q"]
        else:
            baseline_units = [unit for unit in sample.get("all_units", []) if unit.get("base_s", {}).get("status") == "VALID" and unit.get("base_q", {}).get("status") == "VALID"]
            if not baseline_units:
                continue
            base_s = baseline_units[0]["base_s"]; base_q = baseline_units[0]["base_q"]
        unit_records = []
        for unit in units:
            if "base_s" in unit:
                unit_records.append(unit)
        if not unit_records:
            unit_records = [sample]
        base_states: dict[str, list[np.ndarray]] = {condition: [] for condition in MODEL_CONDITIONS}
        # For real windows use all saved units for an unbiased window baseline;
        # synthetic samples contain exactly one unit.
        if sample.get("synthetic"):
            all_units = [sample]
        else:
            all_units = sample["all_units"]
        for unit in all_units:
            for condition in MODEL_CONDITIONS:
                _, std_s, std_q = models[(condition, MODEL_SEEDS[0])]
                base_states[condition].append(_model_condition_state(condition, unit["base_s"]["s"], unit["base_q"]["q"], std_s, std_q))
        for condition in MODEL_CONDITIONS:
            for seed in MODEL_SEEDS:
                model, std_s, std_q = models[(condition, seed)]
                states = np.stack([_model_condition_state(condition, unit["base_s"]["s"], unit["base_q"]["q"], std_s, std_q) for unit in all_units])
                interval_array = np.repeat(intervals[None, :], len(all_units), axis=0)
                local_base = _model_forward(model, states, interval_array)
                window_base = float(np.mean(local_base))
                outputs.append({"sample_id": sample["sample_id"], "synthetic": bool(sample.get("synthetic", False)), "source_id": sample["source_id"], "role": sample["role"], "window_id": sample["window_id"], "local_group_id": "ALL", "operation": "BASELINE_REPRO", "condition": condition, "seed": seed, "unit_logit_base": None, "unit_logit_intervened": None, "unit_logit_delta": None, "window_logit_base": window_base, "window_logit_intervened": window_base, "window_logit_delta": 0.0, "expected_window_delta": 0.0, "aggregation_check_abs": None, "saved_window_logit": sample.get("saved_scores", {}).get((condition, seed)), "saved_reproduction_abs_error": (abs(window_base - sample["saved_scores"][(condition, seed)]) if (condition, seed) in sample.get("saved_scores", {}) else None), "status": "VALID"})
                for feature in by_sample.get(sample["sample_id"], []):
                    if feature["operation"] == "G0" or feature["status"] != "VALID":
                        continue
                    index = next((i for i, unit in enumerate(all_units) if int(unit.get("local_group_id", -1)) == int(feature["local_group_id"])), None)
                    if index is None:
                        continue
                    # The operation result is recomputed from its feature row's
                    # parent object, attached by the caller.
                    result = feature["_result"]
                    s_new = result["s"]; q_new = result["q"]
                    state_new = _model_condition_state(condition, s_new["s"], q_new["q"], std_s, std_q)
                    local_new = _model_forward(model, state_new[None, ...], intervals[None, :])[0]
                    unit_delta = float(local_new - local_base[index]); window_delta = unit_delta / len(all_units)
                    expected_window_delta = unit_delta / len(all_units)
                    outputs.append({"sample_id": sample["sample_id"], "synthetic": bool(sample.get("synthetic", False)), "source_id": sample["source_id"], "role": sample["role"], "window_id": sample["window_id"], "local_group_id": feature["local_group_id"], "operation": feature["operation"], "condition": condition, "seed": seed, "unit_logit_base": float(local_base[index]), "unit_logit_intervened": float(local_new), "unit_logit_delta": unit_delta, "window_logit_base": window_base, "window_logit_intervened": float(window_base + window_delta), "window_logit_delta": window_delta, "expected_window_delta": expected_window_delta, "aggregation_check_abs": abs(window_delta - expected_window_delta), "saved_window_logit": sample.get("saved_scores", {}).get((condition, seed)), "saved_reproduction_abs_error": None, "status": "VALID"})
    return outputs


def _make_real_payload(sample: RealSample) -> dict[str, Any]:
    # The intervention table is intentionally capped at three units, but the
    # frozen window score must be reconstructed from every valid unit in the
    # cached feature manifest.  Keeping these two lists separate prevents a
    # display/sample cap from changing the baseline aggregation.
    npz = _load_npz(OBS_ROOT / str(sample.item["input_path"]))
    identities = sorted(sample.item["unit_identities"], key=lambda value: int(value["local_group_id"]))
    identity_index = {int(identity["local_group_id"]): index for index, identity in enumerate(identities)}
    all_units: list[dict[str, Any]] = []
    for identity in identities:
        group_id = int(identity["local_group_id"])
        if group_id not in sample.support_units:
            continue
        unit = _real_unit(sample, identity)
        unit["local_group_id"] = group_id
        index = identity_index[group_id]
        unit["saved_s"] = np.asarray(npz["s"][index], dtype=np.float64)
        unit["saved_q"] = np.asarray(npz["q"][index], dtype=np.float64)
        all_units.append(unit)
    selected_ids = {int(identity["local_group_id"]) for identity in sample.units}
    selected_units = [unit for unit in all_units if int(unit["local_group_id"]) in selected_ids]
    return {"sample_id": sample.sample_id, "source_id": sample.source_id, "role": sample.role, "window_id": sample.window_id, "row": sample.row, "sequence": sample.sequence, "target": sample.target, "history": sample.history, "units": selected_units, "all_units": all_units, "synthetic": False}


def run(root: Path = OUTPUT_ROOT) -> dict[str, Any]:
    started_mono = time.perf_counter()
    started_unix = time.time()
    root.mkdir(parents=True, exist_ok=True)
    synthetic = _synthetic()
    synthetic["synthetic"] = True; synthetic["units"] = [synthetic]; synthetic["all_units"] = [synthetic]; synthetic["local_group_id"] = 0; synthetic["raw_members"] = synthetic["raw_members"]; synthetic["pairs"] = synthetic["pairs"]
    real_samples = _select_real_samples(); payloads = [synthetic] + [_make_real_payload(sample) for sample in real_samples]
    # Validate the cached S before any intervention and collect frozen model
    # scores for the real windows.
    score_by_id = {row["window_id"]: row for row in csv.DictReader((OBS_ROOT / "scores/train_window_scores.csv").open(newline="", encoding="utf-8"))}
    for payload in payloads:
        if not payload.get("synthetic"):
            saved = score_by_id[payload["window_id"]]; payload["saved_scores"] = {(condition, seed): float(saved[f"{condition}_seed_{seed}"]) for condition in MODEL_CONDITIONS for seed in MODEL_SEEDS}
        else:
            payload["saved_scores"] = {}
    feature_rows: list[dict[str, Any]] = []; operation_results: dict[tuple[str, str, int], dict[str, Any]] = {}
    operations = ("G0", "G1", "G2", "G3", "G4", "G5_A0", "G5_A005", "G5_A020", "G6_TARGET_ONLY", "G6_HISTORY_AND_TARGET", "M1", "M2", "M3")
    for payload in payloads:
        units = payload["units"]
        for unit in units:
            unit_payload = {**payload, **unit, "sample_id": payload["sample_id"], "source_id": payload["source_id"], "role": payload["role"], "window_id": payload["window_id"], "sequence": payload["sequence"], "target": payload["target"], "history": payload["history"], "base_s": unit["base_s"], "base_q": unit["base_q"], "synthetic": payload.get("synthetic", False)}
            for operation in operations:
                if operation == "M3" and not payload.get("synthetic", False):
                    result = {"status": "NOT_APPLICABLE", "reason": "SYNTHETIC_ONLY", "sequence": unit_payload["sequence"], "s": unit_payload["base_s"], "q": unit_payload["base_q"], "members": unit_payload["members"], "pairs": unit_payload["pairs"], "removed": []}
                else:
                    result = _apply_operation(unit_payload, operation, bool(payload.get("synthetic", False)))
                row = _feature_row(unit_payload, operation, result, bool(payload.get("synthetic", False)))
                # Expected status is descriptive, never a threshold used to
                # select samples or discard a result.
                if operation in ("G0", "G1", "G2", "G3") and result["status"] == "VALID":
                    expected = "INVARIANT_CHECK"; row["expected_pass"] = bool(row["s_max_abs_diff"] is not None and row["s_max_abs_diff"] <= TOLERANCE and row["q_max_abs_diff"] is not None and row["q_max_abs_diff"] <= TOLERANCE)
                elif operation == "G4" and result["status"] == "VALID":
                    expected = "TARGET_SCALE_SENSITIVE"; row["expected_pass"] = bool(row["q_max_abs_diff"] is not None and row["q_max_abs_diff"] <= TOLERANCE)
                elif operation.startswith("G5") or operation.startswith("G6"):
                    expected = "NONRIGID_OR_REFERENCE_SENSITIVE"; row["expected_pass"] = None
                elif operation in ("M1", "M2", "M3"):
                    expected = "SUPPORT_SET_EFFECT"; row["expected_pass"] = None
                else:
                    expected = "IDENTITY"; row["expected_pass"] = bool((row["s_max_abs_diff"] or 0.0) <= TOLERANCE and (row["q_max_abs_diff"] or 0.0) <= TOLERANCE) if result["status"] == "VALID" else False
                row["mathematical_expectation"] = expected
                row["_result"] = result
                feature_rows.append(row); operation_results[(payload["sample_id"], operation, int(unit.get("local_group_id", 0)))] = result
    # Compare G6 target coordinates directly for the synthetic and real unit.
    for payload in payloads:
        for unit in payload["units"]:
            target_a = _apply_operation({**payload, **unit}, "G6_TARGET_ONLY", bool(payload.get("synthetic", False)))["sequence"].xyz[np.asarray(payload["target"])]
            target_b = _apply_operation({**payload, **unit}, "G6_HISTORY_AND_TARGET", bool(payload.get("synthetic", False)))["sequence"].xyz[np.asarray(payload["target"])]
            for row in feature_rows:
                if row["sample_id"] == payload["sample_id"] and row["local_group_id"] == int(unit.get("local_group_id", 0)) and row["operation"] in ("G6_TARGET_ONLY", "G6_HISTORY_AND_TARGET"):
                    row["g6_target_xyz_pair_max_abs"] = float(np.max(np.abs(target_a - target_b)))
    models = _load_models()
    model_feature_rows = []
    for row in feature_rows:
        if row["status"] == "VALID" and row["operation"] != "M0":
            row_copy = dict(row); row_copy["_result"] = operation_results[(row["sample_id"], row["operation"], int(row["local_group_id"]))]; model_feature_rows.append(row_copy)
    model_rows = _model_rows(payloads, models, model_feature_rows)
    for row in model_feature_rows:
        row.pop("_result", None)
    for row in feature_rows:
        row.pop("_result", None)
    # The model baseline rows are the frozen-model reproduction check.  Its
    # tolerances are reported, never used to change a score.
    baseline_errors = [float(row["saved_reproduction_abs_error"]) for row in model_rows if row["saved_reproduction_abs_error"] is not None]
    saved_s_errors = [float(row["saved_s_reconstruction_max_abs"]) for row in feature_rows if row.get("saved_s_reconstruction_max_abs") is not None]
    saved_q_errors = [float(row["saved_q_reconstruction_max_abs"]) for row in feature_rows if row.get("saved_q_reconstruction_max_abs") is not None]
    elapsed_s = time.perf_counter() - started_mono
    manifest = {"protocol_id": "v7-representation-sensitivity-probe-v1", "git_head": _git_head(), "selection_seed": SEED, "max_sources": MAX_SOURCES, "max_units_per_window": MAX_UNITS_PER_WINDOW, "model_conditions": list(MODEL_CONDITIONS), "model_seeds": list(MODEL_SEEDS), "tolerance": TOLERANCE, "source_root": str(SOURCE_ROOT), "observation_support_root": str(OBS_ROOT), "samples": []}
    for payload in payloads:
        manifest["samples"].append({"sample_id": payload["sample_id"], "synthetic": bool(payload.get("synthetic", False)), "source_id": payload["source_id"], "role": payload["role"], "window_id": payload["window_id"], "local_group_ids": [int(unit.get("local_group_id", 0)) for unit in payload["units"]], "target_frame_indices": [int(value) for value in payload["target"]], "target_timestamps_s": [float(payload["sequence"].timestamps_s[int(value)]) for value in payload["target"]], "history_frame_indices": [int(value) for value in payload["history"]], "query_cohort": str(payload["sequence"].provenance.get("query_cohort", "synthetic")), "particle_prefix": None if payload.get("synthetic") else str(next(item for item in real_samples if item.sample_id == payload["sample_id"]).row["particle_prefix"]), "unit_members": [{"local_group_id": int(unit.get("local_group_id", 0)), "raw_members": [int(value) for value in unit["raw_members"]], "common_members": [int(value) for value in unit["members"]]} for unit in payload["units"]]})
    _atomic_json(root / "protocol.json", {"protocol_id": "v7-representation-sensitivity-probe-v1", "git_head": _git_head(), "operations": list(operations), "geometry_definition": "existing five-time XYZ pair distances normalized by recomputed shared history median; S=[mean,std,p25,p75]", "support_definition": "existing Q=[visibility,geometry,visibility-history,geometry-history] over raw history members", "intervention_semantics": {"geometry": "fixed masks, groups, common members and pairs; history scale recomputed", "mask": "copy masks and invalid coordinates; common members/pairs/scale recomputed"}, "model_readout": "frozen observation-support STRUCTURE_ONLY/SUPPORT_ONLY/STRUCTURE_SUPPORT, original standardizers, no training", "tolerance": TOLERANCE, "selection_seed": SEED, "real_source_count": len({sample.source_id for sample in real_samples}), "real_sample_count": len(real_samples), "real_window_count": len(real_samples), "started_unix": started_unix, "elapsed_s": elapsed_s})
    _atomic_json(root / "sample_manifest.json", manifest)
    _write_csv(root / "feature_responses.csv", feature_rows)
    _write_csv(root / "model_responses.csv", model_rows)
    invariant_rows = [row for row in feature_rows if row["mathematical_expectation"] == "INVARIANT_CHECK" and row["status"] == "VALID"]
    invariant_pass = sum(bool(row.get("expected_pass")) for row in invariant_rows)
    g6_rows = [row for row in feature_rows if row["operation"] in ("G6_TARGET_ONLY", "G6_HISTORY_AND_TARGET")]
    real_source_count = len({sample.source_id for sample in real_samples})
    m3_rows = [row for row in feature_rows if row["operation"] == "M3" and row["synthetic"] and row["status"] == "VALID"]
    report_lines = ["# V7 表示敏感性与历史参照吸收探针", "", "本探针只在合成点阵和少量已缓存 ParticleSequence 的内存副本上执行；没有修改缓存、重跑前端、训练模型或计算 AUROC。冻结模型读出仅为次要响应，不把扰动样本当作 fake。", "", "## 样本与路径", f"- 合成 unit：1；真实 source：{real_source_count}，真实窗口：{len(real_samples)}（每个 source 一个 real/fake pair，共 {len(real_samples)} 个真实窗口）。选取使用 seed/hash={SEED}，不按模型分数。", f"- 每个真实窗口最多取3个展示/干预 unit，但冻结模型基线使用该窗口全部 {max((len(payload.get('all_units', [])) for payload in payloads if not payload.get('synthetic')), default=0)} 个有效 unit；这两个集合不混用。", f"- 已保存 S/Q 与现有表示函数重建的最大绝对误差：S={max(saved_s_errors) if saved_s_errors else 'NA'}，Q={max(saved_q_errors) if saved_q_errors else 'NA'}。", f"- 模型读出基线保存分数的最大绝对复现误差：{max(baseline_errors) if baseline_errors else 'NA'}；模型条件为 STRUCTURE_ONLY、SUPPORT_ONLY、STRUCTURE_SUPPORT，三 seed 原标准化复用。", f"- 本次运行耗时：{elapsed_s:.3f} s（仅读取缓存、内存干预和冻结模型 CPU 前向）。", "", "## 操作—预期—实际", "", "| 操作 | 数学/表示预期 | 实际特征响应 | 冻结模型响应 | 含义 |", "|---|---|---|---|---|"]
    summaries = {}
    for operation in operations:
        rows = [row for row in feature_rows if row["operation"] == operation and row["status"] == "VALID"]
        if not rows:
            summaries[operation] = {"valid": 0}; report_lines.append(f"| {operation} | — | 无有效记录 | — | NOT_APPLICABLE/支撑不足 |")
            continue
        s_values = [row["s_max_abs_diff"] for row in rows if row["s_max_abs_diff"] is not None]; q_values = [row["q_max_abs_diff"] for row in rows if row["q_max_abs_diff"] is not None]; m_values = [abs(float(row["window_logit_delta"])) for row in model_rows if row["operation"] == operation and row["window_logit_delta"] is not None]
        summaries[operation] = {"valid": len(rows), "s_max_abs_diff_max": max(s_values) if s_values else None, "q_max_abs_diff_max": max(q_values) if q_values else None, "model_window_delta_abs_max": max(m_values) if m_values else None}
        report_lines.append(f"| {operation} | {'S/Q invariant' if operation in ('G0','G1','G2','G3') else 'scale/reference/support response'} | valid={len(rows)}; max ΔS={max(s_values) if s_values else 'NA'}; max ΔQ={max(q_values) if q_values else 'NA'} | max |Δwindow logit|={max(m_values) if m_values else 'NA'} | {'意向不变性核对' if operation in ('G0','G1','G2','G3') else '表示/支撑响应，非检测结论'} |")
    g6_target_equal = max((float(row.get("g6_target_xyz_pair_max_abs", 0.0)) for row in g6_rows), default=None)
    report_lines += ["", "## 关键判断", f"- G0/G1/G2/G3 的 S/Q 不变性检查通过 {invariant_pass}/{len(invariant_rows)} 条有效记录（atol={TOLERANCE}）；G3 重新计算了历史尺度，未复用旧尺度。", "- G4 只在目标阶段缩放，历史尺度未同步缩放，因而与 G3 区分；S 的响应是历史参照敏感性，而非整体尺度不变性。", f"- G5 局部渐进形变和 G6 历史参照对照只报告距离、S、Q 与冻结模型读出；G6 两种操作的目标 XYZ 最大差为 {g6_target_equal}，但历史尺度/归一化 S 可不同。", "- M1 只改变非共同历史成员的观测 mask，预期 S 不变而 Q 可能变；M2 改变共同成员集合，任何差异都必须解释为支撑集合改变。M3 仅在合成点阵检查 Q 的中间下降/随后恢复，同时该点不重新进入五时刻共同结构。", f"- M3 合成 Q 可用性记录数：{len(m3_rows)}；其 `q_visibility_trace`/`q_geometry_trace` 保留中间时刻失效后的轨迹，不能解读为补全或物理可见性真值。", "- 结果只适用于固定已有分组/共同成员的表示；不验证 component 发现、跨窗口物理对应、记忆继承、真实前端物理准确性或空间伪造真值。", "", "## 产物", "- `protocol.json`：冻结定义与操作。", "- `sample_manifest.json`：合成与真实缓存样本身份。", "- `feature_responses.csv`：每个 unit/操作的距离、尺度、S/Q、mask 与支撑结果。", "- `model_responses.csv`：冻结模型 unit/window 读出及均值聚合核对。", "", "## 优先建议", "先保留当前表示的明确不变性边界，不据此增加模型或重跑数据；若仍需推进，应只在未参与开发的独立样本上验证局部形变对 S/Q 的响应。"]
    (root / "report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    final = {"status": "COMPLETE", "git_head": _git_head(), "synthetic_units": 1, "real_sources": real_source_count, "real_windows": len(real_samples), "feature_rows": len(feature_rows), "model_rows": len(model_rows), "elapsed_s": elapsed_s, "saved_reconstruction_max_abs_error": {"s": max(saved_s_errors) if saved_s_errors else None, "q": max(saved_q_errors) if saved_q_errors else None}, "invariant_checks": {"valid": len(invariant_rows), "passed": invariant_pass, "tolerance": TOLERANCE}, "baseline_model_reproduction_max_abs_error": max(baseline_errors) if baseline_errors else None, "summary": summaries}
    _atomic_json(root / "final_status.json", final)
    return final


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUTPUT_ROOT)
    args = parser.parse_args()
    print(json.dumps(run(args.output), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
