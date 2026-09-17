"""Run the bounded V7 history-neighborhood local-context pilot.

This entry point consumes only the completed source128 R support cache.  It
constructs a history-only graph for each retained local group, writes a small
numeric input cache, runs one smoke path, and then trains/evaluates the five
frozen conditions.  It is deliberately a single experiment entry point, not
a graph-learning framework.
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
from research_tools.v7.multi_order_sequence_probe.model import SetAModel
from research_tools.v7.periodic_requery_probe import runner as periodic
from research_tools.v7.source128_extension import runner as source128

from .model import CONTEXT_CONDITIONS, LocalContextModel, parameter_count, state_dict_numpy


REPO_ROOT = Path(__file__).resolve().parents[3]
DATA_ROOT = Path("/root/autodl-tmp/data/sparse_3d_forgery_detection")
SOURCE_ROOT = DATA_ROOT / "derived/v7_activityforensics_source128_extension_v1"
OUTPUT_ROOT = DATA_ROOT / "derived/v7_activityforensics_local_context_pilot_v1"
CONDITIONS = ("SUMMARY_BASELINE",) + CONTEXT_CONDITIONS
SEEDS = (20260909, 20260910, 20260911)
EPOCHS = 200
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
BOOTSTRAP_SEED = 20260909
BOOTSTRAP_REPLICATES = 10_000
K_NEIGHBORS = 3
DISTANCE_MULTIPLIER = 2.0
MINIMUM_HISTORY_OVERLAP = 8
MAX_REWIRE_ATTEMPTS_MULTIPLIER = 20


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
        writer.writeheader(); writer.writerows(rows)
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


def _load_selection() -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    all_support = json.loads((SOURCE_ROOT / "support/window_support.json").read_text(encoding="utf-8"))
    r_support = [dict(row) for row in all_support if str(row.get("mode")) == "R"]
    support_by_id = {str(row["window_id"]): row for row in r_support}
    if len(support_by_id) != len(r_support):
        raise ValueError("DUPLICATE_R_WINDOW_ID")
    with (SOURCE_ROOT / "models/training_window_manifest.csv").open(newline="", encoding="utf-8") as handle:
        train_ids = [str(row["window_id"]) for row in csv.DictReader(handle)]
    with (SOURCE_ROOT / "manifests/validation_window_manifest.csv").open(newline="", encoding="utf-8") as handle:
        validation_ids = [str(row["window_id"]) for row in csv.DictReader(handle)]
    if len(set(train_ids)) != len(train_ids) or len(set(validation_ids)) != len(validation_ids):
        raise ValueError("DUPLICATE_TRAIN_OR_VALIDATION_WINDOW_ID")
    if set(train_ids) & set(validation_ids):
        raise ValueError("TRAIN_VALIDATION_WINDOW_OVERLAP")
    missing_train = sorted(set(train_ids) - set(support_by_id))
    missing_validation = sorted(set(validation_ids) - set(support_by_id))
    if missing_train or missing_validation:
        raise ValueError(f"SUPPORT_WINDOW_MISSING:train={missing_train[:3]} validation={missing_validation[:3]}")
    train_rows = [support_by_id[item] for item in train_ids]
    validation_rows = [support_by_id[item] for item in validation_ids]
    train_sources = {str(row["source_id"]) for row in train_rows}
    validation_sources = {str(row["source_id"]) for row in validation_rows}
    if train_sources & validation_sources:
        raise ValueError("TRAIN_VALIDATION_SOURCE_OVERLAP")
    for row in train_rows + validation_rows:
        feature = np.asarray(row.get("features", {}).get("SET_A"), dtype=np.float64)
        if feature.ndim != 3 or feature.shape[1:] != (5, 4) or feature.shape[0] <= 0 or not np.all(np.isfinite(feature)):
            raise ValueError(f"INVALID_SET_A_STATE:{row['window_id']}")
        intervals = np.asarray(row.get("intervals_s"), dtype=np.float64)
        if intervals.shape != (4,) or not np.all(np.isfinite(intervals)) or not np.all(intervals > 0):
            raise ValueError(f"INVALID_INTERVALS:{row['window_id']}")
    info = {
        "all_r_support_rows": len(r_support),
        "all_r_valid_rows": sum(str(row.get("support_status")) == "VALID" and int(row.get("valid_unit_count", 0)) > 0 for row in r_support),
        "training_windows": len(train_rows),
        "validation_windows": len(validation_rows),
        "training_sources": sorted(train_sources),
        "validation_sources": sorted(validation_sources),
        "training_real": sum(int(row["label"]) == 0 for row in train_rows),
        "training_fake": sum(int(row["label"]) == 1 for row in train_rows),
        "validation_real": sum(int(row["label"]) == 0 for row in validation_rows),
        "validation_fake": sum(int(row["label"]) == 1 for row in validation_rows),
    }
    return train_rows, validation_rows, {"r_support": r_support, "info": info}


def _sequence_for_row(row: Mapping[str, Any]) -> Any:
    prefix = str(row.get("particle_prefix") or "")
    if not prefix:
        raise ValueError(f"MISSING_PARTICLE_PREFIX:{row['window_id']}")
    sequence = load_particle_sequence(prefix)
    expected_video = f"{row['source_id']}::{row['role']}"
    if str(sequence.source_video_id) != expected_video or int(sequence.num_tracks) != 289:
        raise ValueError(f"SEQUENCE_IDENTITY_MISMATCH:{row['window_id']}")
    lineage = sequence.lineage
    if any(str(lineage.get(key, "")) != str(row.get(key, "")) for key in ("source_id", "role", "pair_id")):
        raise ValueError(f"SEQUENCE_LINEAGE_MISMATCH:{row['window_id']}")
    offset = float(row.get("offset_s", 0.0))
    expected_cohort = "O" if abs(offset) < 1e-9 else "R"
    if str(sequence.provenance.get("query_cohort", "")) != expected_cohort:
        raise ValueError(f"SEQUENCE_COHORT_MISMATCH:{row['window_id']}")
    parent_id = str(row["window_id"]).rsplit("::b", 1)[0]
    expected_window = parent_id if expected_cohort == "O" else str(row["window_id"])
    if str(lineage.get("window_id", "")) != expected_window:
        raise ValueError(f"SEQUENCE_WINDOW_MISMATCH:{row['window_id']}")
    return sequence


def _stable_rng(window_id: str, suffix: str) -> np.random.Generator:
    digest = hashlib.sha256(f"v7-local-context|{suffix}|{window_id}|seed=20260909".encode("utf-8")).digest()
    return np.random.default_rng(int.from_bytes(digest[:8], "little", signed=False))


def _time_permutation(window_id: str) -> np.ndarray:
    permutation = _stable_rng(window_id, "time").permutation(5).astype(np.int64)
    if np.array_equal(permutation, np.arange(5, dtype=np.int64)):
        permutation[:2] = permutation[1], permutation[0]
    return permutation


def _rewire_edges(edge_src: np.ndarray, edge_dst: np.ndarray, node_count: int, window_id: str) -> tuple[np.ndarray, dict[str, Any]]:
    original = [(int(src), int(dst)) for src, dst in zip(edge_src, edge_dst)]
    if len(original) < 2:
        return edge_dst.copy(), {"attempts": 0, "max_attempts": 0, "changed_edge_ratio": 0.0, "status": "NO_TWO_EDGES"}
    current = list(original)
    edge_set = set(current)
    rng = _stable_rng(window_id, "rewire")
    max_attempts = MAX_REWIRE_ATTEMPTS_MULTIPLIER * len(current)
    accepted = 0
    for attempt in range(max_attempts):
        left, right = rng.choice(len(current), size=2, replace=False)
        i, j = current[int(left)]
        p, q = current[int(right)]
        if i == p or j == q:
            continue
        proposals = ((i, q), (p, j))
        if any(src == dst for src, dst in proposals):
            continue
        old_left, old_right = current[int(left)], current[int(right)]
        edge_set_without = edge_set - {old_left, old_right}
        if proposals[0] in edge_set_without or proposals[1] in edge_set_without or proposals[0] == proposals[1]:
            continue
        current[int(left)], current[int(right)] = proposals
        edge_set = edge_set_without | set(proposals)
        accepted += 1
    rewired = np.asarray([dst for _, dst in current], dtype=np.int64)
    changed = float(np.mean(rewired != edge_dst)) if edge_dst.size else 0.0
    return rewired, {"attempts": max_attempts, "max_attempts": max_attempts, "accepted_swaps": accepted, "changed_edge_ratio": changed, "status": "CHANGED" if accepted else "UNCHANGED"}


def _history_graph(row: Mapping[str, Any], sequence: Any) -> tuple[dict[str, Any], np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    grouping = row.get("grouping") or {}
    groups = [dict(group) for group in grouping.get("groups", []) if bool(group.get("retained"))]
    groups.sort(key=lambda group: int(group["local_group_id"]))
    if not groups:
        raise ValueError(f"NO_RETAINED_GROUPS:{row['window_id']}")
    history_indices = np.asarray(grouping.get("history_array_indices", []), dtype=np.int64)
    if history_indices.ndim != 1 or history_indices.size == 0:
        raise ValueError(f"HISTORY_INDICES_MISSING:{row['window_id']}")
    xyz = np.asarray(sequence.xyz, dtype=np.float64)
    valid = np.asarray(sequence.geometry_validity, dtype=bool)
    if np.any(history_indices < 0) or np.any(history_indices >= xyz.shape[0]):
        raise ValueError(f"HISTORY_INDEX_OUT_OF_RANGE:{row['window_id']}")
    centers: list[np.ndarray] = []
    center_valid: list[np.ndarray] = []
    for group in groups:
        members = np.asarray(group.get("member_slots", []), dtype=np.int64)
        if members.size < 3:
            centers.append(np.full((history_indices.size, 3), np.nan, dtype=np.float64)); center_valid.append(np.zeros(history_indices.size, dtype=bool)); continue
        points = xyz[history_indices][:, members]
        observed = valid[history_indices][:, members] & np.all(np.isfinite(points), axis=-1)
        enough = np.sum(observed, axis=1) >= 3
        center = np.full((history_indices.size, 3), np.nan, dtype=np.float64)
        for frame_index in np.flatnonzero(enough):
            center[frame_index] = np.mean(points[frame_index][observed[frame_index]], axis=0)
        centers.append(center); center_valid.append(enough)
    distances: dict[tuple[int, int], float] = {}
    nearest: list[float] = []
    for left in range(len(groups)):
        candidates: list[tuple[float, int]] = []
        for right in range(len(groups)):
            if left == right:
                continue
            overlap = center_valid[left] & center_valid[right]
            if int(np.sum(overlap)) < MINIMUM_HISTORY_OVERLAP:
                continue
            values = np.linalg.norm(centers[left][overlap] - centers[right][overlap], axis=1)
            if values.size and np.all(np.isfinite(values)):
                distance = float(np.median(values)); distances[(left, right)] = distance; candidates.append((distance, right))
        if candidates:
            nearest.append(min(candidates, key=lambda item: (item[0], item[1]))[0])
    d0 = float(np.median(np.asarray(nearest, dtype=np.float64))) if nearest else None
    edge_pairs: list[tuple[int, int]] = []
    if d0 is not None and math.isfinite(d0) and d0 > 0:
        threshold = DISTANCE_MULTIPLIER * d0
        for left in range(len(groups)):
            candidates = sorted((distance, right) for (source, right), distance in distances.items() if source == left and distance <= threshold)
            edge_pairs.extend((left, right) for _, right in candidates[:K_NEIGHBORS])
    group_to_node = {int(group["local_group_id"]): index for index, group in enumerate(groups)}
    valid_units = [unit for unit in (row.get("support", {}).get("units", [])) if unit.get("status") == "VALID"]
    states = np.asarray(row.get("features", {}).get("SET_A"), dtype=np.float64)
    if states.shape != (len(valid_units), 5, 4) or not np.all(np.isfinite(states)):
        raise ValueError(f"STATE_UNIT_ALIGNMENT_MISMATCH:{row['window_id']}")
    unit_by_group = {int(unit["local_group_id"]): (index, unit) for index, unit in enumerate(valid_units)}
    valid_group_ids = sorted(set(unit_by_group) & set(group_to_node))
    if not valid_group_ids:
        raise ValueError(f"NO_VALID_GRAPH_NODES:{row['window_id']}")
    states_ordered = np.stack([states[unit_by_group[group_id][0]] for group_id in valid_group_ids], axis=0).astype(np.float64)
    node_map = {group_to_node[group_id]: index for index, group_id in enumerate(valid_group_ids)}
    induced = [(node_map[left], node_map[right]) for left, right in edge_pairs if left in node_map and right in node_map]
    induced.sort()
    edge_src = np.asarray([left for left, _ in induced], dtype=np.int64)
    edge_dst = np.asarray([right for _, right in induced], dtype=np.int64)
    rewired_dst, rewire = _rewire_edges(edge_src, edge_dst, len(valid_group_ids), str(row["window_id"]))
    permutation = _time_permutation(str(row["window_id"]))
    parent_by_node = [int(groups[group_to_node[group_id]].get("parent_component_id", -1)) for group_id in valid_group_ids]
    same_parent = sum(parent_by_node[int(src)] == parent_by_node[int(dst)] for src, dst in zip(edge_src, edge_dst))
    graph = {
        "window_id": str(row["window_id"]),
        "source_id": str(row["source_id"]),
        "role": str(row["role"]),
        "offset_s": float(row.get("offset_s", 0.0)),
        "label": int(row["label"]),
        "annotation_category": str(row.get("annotation_category", "")),
        "query_cohort": "R" if float(row.get("offset_s", 0.0)) else "O_REUSED_AT_B0",
        "sequence_prefix": str(row["particle_prefix"]),
        "local_group_ids": valid_group_ids,
        "parent_component_ids": parent_by_node,
        "history_node_count": len(groups),
        "valid_node_count": len(valid_group_ids),
        "history_edge_count": len(edge_pairs),
        "induced_edge_count": int(edge_src.size),
        "d0": d0,
        "distance_threshold": None if d0 is None or not math.isfinite(d0) or d0 <= 0 else DISTANCE_MULTIPLIER * d0,
        "minimum_history_overlap": MINIMUM_HISTORY_OVERLAP,
        "same_parent_edge_count": int(same_parent),
        "cross_parent_edge_count": int(edge_src.size - same_parent),
        "rewire": rewire,
        "time_permutation": permutation.tolist(),
        "empty_neighborhood_node_count": int(len(valid_group_ids) - len(set(int(x) for x in edge_src))),
    }
    return graph, states_ordered, np.asarray(row["intervals_s"], dtype=np.float64), edge_src, edge_dst, rewired_dst, permutation


def _save_npz(path: Path, *, states: np.ndarray, intervals: np.ndarray, edge_src: np.ndarray, edge_dst: np.ndarray, rewired_dst: np.ndarray, time_permutation: np.ndarray, local_group_ids: np.ndarray, parent_component_ids: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp.npz")
    np.savez_compressed(temporary, states=states, intervals=intervals, edge_src=edge_src, edge_dst=edge_dst, rewired_dst=rewired_dst, time_permutation=time_permutation, local_group_ids=local_group_ids, parent_component_ids=parent_component_ids)
    os.replace(temporary, path)


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {key: np.asarray(archive[key]) for key in archive.files}


def _input_content_sha256(root: Path, manifest: Sequence[Mapping[str, Any]]) -> str:
    """Hash the actual numeric input files used by the pilot.

    The graph manifest identifies the construction, while this second digest
    protects model resume from an NPZ being replaced without a manifest edit.
    It is computed once per run and is deliberately not a cache registry.
    """

    digest = hashlib.sha256()
    for item in sorted(manifest, key=lambda value: str(value["window_id"])):
        relative = str(item["input_path"])
        path = root / relative
        if not path.is_file():
            raise FileNotFoundError(f"MISSING_INPUT_NPZ:{relative}")
        digest.update(str(item["window_id"]).encode("utf-8"))
        digest.update(b"\0")
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                digest.update(block)
    return digest.hexdigest()


def build_inputs(root: Path, train_rows: Sequence[Mapping[str, Any]], validation_rows: Sequence[Mapping[str, Any]], all_r_rows: Sequence[Mapping[str, Any]], *, resume: bool = True) -> dict[str, Any]:
    input_dir = root / "inputs/window_inputs"
    manifest_path = root / "inputs/graph_manifest.json"
    selected = list(train_rows) + list(validation_rows)
    selected_ids = {str(row["window_id"]) for row in selected}
    if resume and manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if {str(row["window_id"]) for row in manifest} == selected_ids and all((root / str(row["input_path"])).is_file() for row in manifest):
            return {"manifest": manifest, "reused": True}
    manifest: list[dict[str, Any]] = []
    coverage_rows = []
    for row in all_r_rows:
        coverage_rows.append({"window_id": str(row["window_id"]), "source_id": str(row["source_id"]), "role": str(row["role"]), "label": row.get("label"), "offset_s": row.get("offset_s"), "support_status": row.get("support_status"), "valid_unit_count": int(row.get("valid_unit_count", 0)), "support_reasons": json.dumps(row.get("support_reasons", row.get("support", {}).get("invalid_reasons", [])), sort_keys=True)})
    write_csv(root / "coverage/r_support_coverage.csv", coverage_rows)
    selected_by_id = {str(row["window_id"]): row for row in selected}
    for index, window_id in enumerate(sorted(selected_by_id)):
        row = selected_by_id[window_id]
        progress(root, "graph", "RUNNING", index, len(selected), current_window=window_id)
        sequence = _sequence_for_row(row)
        graph, states, intervals, edge_src, edge_dst, rewired_dst, permutation = _history_graph(row, sequence)
        relative = Path("inputs/window_inputs") / f"{_safe(window_id)}.npz"
        _save_npz(root / relative, states=states, intervals=intervals, edge_src=edge_src, edge_dst=edge_dst, rewired_dst=rewired_dst, time_permutation=permutation, local_group_ids=np.asarray(graph["local_group_ids"], dtype=np.int64), parent_component_ids=np.asarray(graph["parent_component_ids"], dtype=np.int64))
        graph["input_path"] = str(relative)
        graph["states_shape"] = list(states.shape)
        graph["intervals"] = intervals.tolist()
        graph["train_or_validation"] = "train" if window_id in {str(item["window_id"]) for item in train_rows} else "validation"
        manifest.append(graph)
        del sequence
    manifest.sort(key=lambda item: str(item["window_id"]))
    atomic_json(manifest_path, manifest)
    atomic_json(root / "inputs/input_manifest.json", {"selected_window_count": len(manifest), "train_window_count": len(train_rows), "validation_window_count": len(validation_rows), "graph_manifest_sha256": sha256(manifest_path), "input_content_sha256": _input_content_sha256(root, manifest), "coverage_path": str(root / "coverage/r_support_coverage.csv")})
    progress(root, "graph", "COMPLETE", len(manifest), len(manifest), graph_manifest_sha256=sha256(manifest_path))
    return {"manifest": manifest, "reused": False}


def _load_batches(root: Path, manifest: Sequence[Mapping[str, Any]], rows: Sequence[Mapping[str, Any]], standardizer: Any, condition: str) -> dict[str, Any]:
    row_by_id = {str(row["window_id"]): row for row in rows}
    selected = [item for item in manifest if str(item["window_id"]) in row_by_id]
    selected.sort(key=lambda item: str(item["window_id"]))
    states_parts: list[np.ndarray] = []
    intervals: list[np.ndarray] = []
    node_window: list[int] = []
    edge_src_parts: list[np.ndarray] = []
    edge_dst_parts: list[np.ndarray] = []
    edge_perm_parts: list[np.ndarray] = []
    labels: list[int] = []
    window_ids: list[str] = []
    node_offset = 0
    for window_index, item in enumerate(selected):
        arrays = _load_npz(root / str(item["input_path"]))
        states = np.asarray(arrays["states"], dtype=np.float64)
        states_parts.append(standardizer.transform(states)); intervals.append(np.asarray(arrays["intervals"], dtype=np.float64)); node_window.extend([window_index] * states.shape[0])
        src = np.asarray(arrays["edge_src"], dtype=np.int64) + node_offset
        if condition == "SELF_TEMPORAL":
            dst = src.copy()
        elif condition == "NEIGHBOR_REWIRED":
            dst = np.asarray(arrays["rewired_dst"], dtype=np.int64) + node_offset
        else:
            dst = np.asarray(arrays["edge_dst"], dtype=np.int64) + node_offset
        edge_src_parts.append(src); edge_dst_parts.append(dst)
        if condition == "NEIGHBOR_TIME_SHUFFLED":
            permutation = np.asarray(arrays["time_permutation"], dtype=np.int64)
            edge_perm_parts.append(np.repeat(permutation[None, :], src.size, axis=0))
        labels.append(int(row_by_id[str(item["window_id"])]["label"])); window_ids.append(str(item["window_id"]))
        node_offset += states.shape[0]
    edge_src = np.concatenate(edge_src_parts) if edge_src_parts else np.empty(0, dtype=np.int64)
    edge_dst = np.concatenate(edge_dst_parts) if edge_dst_parts else np.empty(0, dtype=np.int64)
    edge_perm = np.concatenate(edge_perm_parts) if edge_perm_parts else None
    return {"states": np.concatenate(states_parts), "intervals": np.stack(intervals), "node_window_index": np.asarray(node_window, dtype=np.int64), "edge_src": edge_src, "edge_dst": edge_dst, "edge_time_index": edge_perm, "labels": np.asarray(labels, dtype=np.int64), "window_ids": window_ids, "window_count": len(labels), "window_rows": [row_by_id[item] for item in window_ids]}


def _state_hash(model: Any) -> str:
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        digest.update(name.encode("utf-8")); digest.update(value.detach().cpu().numpy().tobytes(order="C"))
    return digest.hexdigest()


def _new_model(condition: str, seed: int, device: str) -> Any:
    import torch
    torch.manual_seed(int(seed)); np.random.seed(int(seed) & 0xFFFFFFFF)
    model = SetAModel() if condition == "SUMMARY_BASELINE" else LocalContextModel()
    model.to(device)
    return model


def _forward(model: Any, condition: str, batch: Mapping[str, Any], device: str) -> Any:
    import torch
    target = torch.device(device)
    states = torch.as_tensor(batch["states"], dtype=torch.float32, device=target)
    intervals = torch.as_tensor(batch["intervals"], dtype=torch.float32, device=target)
    indices = torch.as_tensor(batch["node_window_index"], dtype=torch.long, device=target)
    if condition == "SUMMARY_BASELINE":
        unit_logits = model(states, intervals.index_select(0, indices))
        sums = torch.zeros(batch["window_count"], dtype=unit_logits.dtype, device=target); counts = torch.zeros(batch["window_count"], dtype=unit_logits.dtype, device=target)
        sums.index_add_(0, indices, unit_logits); counts.index_add_(0, indices, torch.ones_like(unit_logits))
        return sums / counts.clamp_min(1.0)
    return model(states, intervals, indices, torch.as_tensor(batch["edge_src"], dtype=torch.long, device=target), torch.as_tensor(batch["edge_dst"], dtype=torch.long, device=target), None if batch["edge_time_index"] is None else torch.as_tensor(batch["edge_time_index"], dtype=torch.long, device=target), int(batch["window_count"]))


def _train_one(condition: str, batch: Mapping[str, Any], seed: int, device: str, epochs: int = EPOCHS) -> tuple[Any, dict[str, Any]]:
    import torch
    import torch.nn.functional as F
    model = _new_model(condition, seed, device); initial_hash = _state_hash(model); model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    labels = torch.as_tensor(batch["labels"], dtype=torch.float32, device=device); weights = torch.as_tensor(batch["window_weights"], dtype=torch.float32, device=device)
    history: list[dict[str, Any]] = []
    for epoch in range(1, epochs + 1):
        optimizer.zero_grad(set_to_none=True)
        scores = _forward(model, condition, batch, device)
        loss = torch.sum(F.binary_cross_entropy_with_logits(scores, labels, reduction="none") * weights) / torch.sum(weights)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"NONFINITE_LOSS:{condition}:{seed}:{epoch}")
        loss.backward()
        if not all(parameter.grad is None or torch.all(torch.isfinite(parameter.grad)) for parameter in model.parameters()):
            raise FloatingPointError(f"NONFINITE_GRADIENT:{condition}:{seed}:{epoch}")
        optimizer.step()
        history.append({"epoch": epoch, "loss": float(loss.detach().cpu())})
        if epoch in (1, epochs):
            print(f"local-context condition={condition} seed={seed} epoch={epoch}/{epochs} loss={history[-1]['loss']:.6f}", flush=True)
    model.eval()
    return model, {"seed": int(seed), "epochs": int(epochs), "initial_loss": history[0]["loss"], "final_loss": history[-1]["loss"], "min_loss": min(item["loss"] for item in history), "loss_history": history, "device": str(torch.device(device)), "parameter_count": parameter_count(model), "initial_state_hash": initial_hash}


def _model_record_identity(root: Path, manifest: Sequence[Mapping[str, Any]], train_rows: Sequence[Mapping[str, Any]], validation_rows: Sequence[Mapping[str, Any]], condition: str, seed: int, standardizer: Any, input_content_sha256: str | None = None) -> dict[str, Any]:
    payload = json.dumps([{key: value for key, value in item.items() if key not in {"intervals"}} for item in manifest], sort_keys=True, separators=(",", ":")).encode("utf-8")
    return {"condition": str(condition), "seed": int(seed), "graph_manifest_sha256": hashlib.sha256(payload).hexdigest(), "input_content_sha256": input_content_sha256 or _input_content_sha256(root, manifest), "training_window_ids_sha256": hashlib.sha256("\n".join(sorted(str(row["window_id"]) for row in train_rows)).encode("utf-8")).hexdigest(), "validation_window_ids_sha256": hashlib.sha256("\n".join(sorted(str(row["window_id"]) for row in validation_rows)).encode("utf-8")).hexdigest(), "standardization": standardizer.as_dict(), "model_config": {"context_conditions": list(CONTEXT_CONDITIONS), "epochs": EPOCHS, "learning_rate": LEARNING_RATE, "weight_decay": WEIGHT_DECAY}}


def _record_matches(record: Mapping[str, Any], expected: Mapping[str, Any]) -> bool:
    actual = record.get("input_identity")
    return isinstance(actual, Mapping) and all(actual.get(key) == value for key, value in expected.items())


def _model_state_load(condition: str, record: Mapping[str, Any], device: str) -> Any:
    import torch
    model = SetAModel() if condition == "SUMMARY_BASELINE" else LocalContextModel()
    model.load_state_dict({name: torch.as_tensor(value, dtype=model.state_dict()[name].dtype) for name, value in record["state_dict"].items()})
    model.to(device); model.eval(); return model


def _metrics(rows: Sequence[Mapping[str, Any]], scores: Sequence[float]) -> tuple[dict[str, Any], dict[str, float]]:
    labels = [int(row["label"]) for row in rows]; values = [float(item) for item in scores]
    if not all(math.isfinite(item) for item in values):
        raise ValueError("NONFINITE_SCORE")
    by_source: dict[str, list[tuple[int, float]]] = defaultdict(list)
    for row, score in zip(rows, values): by_source[str(row["source_id"])].append((int(row["label"]), score))
    source_values = {source: periodic._auroc([item[0] for item in values_], [item[1] for item in values_]) for source, values_ in by_source.items()}
    source_values = {source: float(value) for source, value in source_values.items() if value is not None}
    macro = source128._bootstrap(source_values)
    classification = periodic._classification(labels, values)
    return {"window_count": len(rows), "real_count": labels.count(0), "fake_count": labels.count(1), "source_count": len(by_source), "dual_role_source_count": len(source_values), "source_macro_auroc": macro, "pooled_auroc": periodic._auroc(labels, values), "pooled_ap": periodic._ap(labels, values), **classification}, source_values


def _metric_row(split: str, condition: str, seed: int | str, metrics: Mapping[str, Any]) -> dict[str, Any]:
    macro = metrics["source_macro_auroc"]; return {"split": split, "condition": condition, "seed": seed, "window_count": metrics["window_count"], "real_count": metrics["real_count"], "fake_count": metrics["fake_count"], "source_count": metrics["source_count"], "dual_role_source_count": metrics["dual_role_source_count"], "source_macro_auroc": macro.get("mean"), "source_macro_ci_low": (macro.get("ci95") or [None, None])[0], "source_macro_ci_high": (macro.get("ci95") or [None, None])[1], "pooled_auroc": metrics["pooled_auroc"], "pooled_ap": metrics["pooled_ap"], "precision": metrics["precision"], "recall": metrics["recall"], "f1": metrics["f1"], "accuracy": metrics["accuracy"], "tn": metrics["tn"], "fp": metrics["fp"], "fn": metrics["fn"], "tp": metrics["tp"]}


def evaluate(root: Path, manifest: Sequence[Mapping[str, Any]], train_rows: Sequence[Mapping[str, Any]], validation_rows: Sequence[Mapping[str, Any]], records: Sequence[Mapping[str, Any]], standardizer: Any, device: str) -> dict[str, Any]:
    score_store: dict[tuple[str, str, int], dict[str, float]] = {}
    metric_rows: list[dict[str, Any]] = []
    for condition in CONDITIONS:
        for record in [item for item in records if str(item["condition"]) == condition]:
            seed = int(record["seed"]); model = _model_state_load(condition, record, device)
            for split, rows in (("train", train_rows), ("validation", validation_rows)):
                batch = _load_batches(root, manifest, rows, standardizer, "SELF_TEMPORAL" if condition == "SUMMARY_BASELINE" else condition)
                scores = _forward(model, condition, batch, device).detach().cpu().numpy().astype(np.float64)
                score_store[(split, condition, seed)] = {window_id: float(value) for window_id, value in zip(batch["window_ids"], scores)}
                metrics, _ = _metrics(batch["window_rows"], scores); metric_rows.append(_metric_row(split, condition, seed, metrics))
            del model
    for split, rows in (("train", train_rows), ("validation", validation_rows)):
        for condition in CONDITIONS:
            mean_scores = {str(row["window_id"]): float(np.mean([score_store[(split, condition, seed)][str(row["window_id"])] for seed in SEEDS])) for row in rows}
            ordered = [mean_scores[str(row["window_id"])] for row in rows]; metrics, source_values = _metrics(rows, ordered); metric_rows.append(_metric_row(split, condition, "MEAN_LOGIT", metrics))
            if split == "validation":
                score_store[(split, condition, 0)] = mean_scores
    write_csv(root / "scores/train_window_scores.csv", _score_rows(train_rows, score_store, "train")); write_csv(root / "scores/validation_window_scores.csv", _score_rows(validation_rows, score_store, "validation"))
    validation_source_values: dict[str, dict[str, float]] = {}
    for condition in CONDITIONS:
        rows = validation_rows; mean_scores = score_store[("validation", condition, 0)]
        validation_source_values[condition] = _source_values_from_mapping(rows, mean_scores)
    comparisons: list[dict[str, Any]] = []
    for left, right in (("NEIGHBOR_TEMPORAL", "SELF_TEMPORAL"), ("NEIGHBOR_TEMPORAL", "NEIGHBOR_REWIRED"), ("NEIGHBOR_TEMPORAL", "NEIGHBOR_TIME_SHUFFLED"), ("NEIGHBOR_TEMPORAL", "SUMMARY_BASELINE")):
        result = source128._bootstrap(validation_source_values[left], validation_source_values[right]); result.update({"left": left, "right": right, "name": f"{left}-{right}"}); comparisons.append(result)
    per_source: list[dict[str, Any]] = []
    for source in sorted({str(row["source_id"]) for row in validation_rows}):
        source_rows = [row for row in validation_rows if str(row["source_id"]) == source]
        for condition in CONDITIONS:
            for seed in (*SEEDS, "MEAN_LOGIT"):
                mapping = score_store[("validation", condition, 0 if seed == "MEAN_LOGIT" else int(seed))]
                values = [mapping[str(row["window_id"])] for row in source_rows]
                per_source.append({"source_id": source, "condition": condition, "seed": seed, "window_count": len(values), "real_count": sum(int(row["label"]) == 0 for row in source_rows), "fake_count": sum(int(row["label"]) == 1 for row in source_rows), "auroc": periodic._auroc([int(row["label"]) for row in source_rows], values)})
    write_csv(root / "evaluation/metrics.csv", metric_rows); write_csv(root / "evaluation/per_source_metrics.csv", per_source); write_csv(root / "evaluation/paired_comparisons.csv", [{"comparison": item["name"], "source_count": item["source_count"], "mean": item["mean"], "ci_low": item["ci95"][0], "ci_high": item["ci95"][1], "positive_count": sum(float(value) > 0 for value in [validation_source_values[item["left"]][source] - validation_source_values[item["right"]][source] for source in item["sources"]]), "negative_count": sum(float(value) < 0 for value in [validation_source_values[item["left"]][source] - validation_source_values[item["right"]][source] for source in item["sources"]])} for item in comparisons])
    summary = {"train_population": {"windows": len(train_rows), "real": sum(int(row["label"]) == 0 for row in train_rows), "fake": sum(int(row["label"]) == 1 for row in train_rows), "sources": len({str(row["source_id"]) for row in train_rows})}, "validation_population": {"windows": len(validation_rows), "real": sum(int(row["label"]) == 0 for row in validation_rows), "fake": sum(int(row["label"]) == 1 for row in validation_rows), "sources": len({str(row["source_id"]) for row in validation_rows})}, "metrics": metric_rows, "comparisons": comparisons, "device": str(device), "model_count": len(records)}
    atomic_json(root / "evaluation/summary.json", summary)
    return summary


def _source_values_from_mapping(rows: Sequence[Mapping[str, Any]], mapping: Mapping[str, float]) -> dict[str, float]:
    result: dict[str, float] = {}
    for source in sorted({str(row["source_id"]) for row in rows}):
        subset = [row for row in rows if str(row["source_id"]) == source]
        value = periodic._auroc([int(row["label"]) for row in subset], [float(mapping[str(row["window_id"])]) for row in subset])
        if value is not None: result[source] = float(value)
    return result


def _score_rows(rows: Sequence[Mapping[str, Any]], score_store: Mapping[tuple[str, str, int], Mapping[str, float]], split: str) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for row in rows:
        item = {"window_id": str(row["window_id"]), "source_id": str(row["source_id"]), "role": str(row["role"]), "label": int(row["label"]), "annotation_category": str(row.get("annotation_category", "")), "valid_unit_count": int(row.get("valid_unit_count", 0))}
        for condition in CONDITIONS:
            for seed in SEEDS: item[f"{condition}_seed_{seed}"] = score_store[(split, condition, seed)][str(row["window_id"])]
            item[f"{condition}_MEAN_LOGIT"] = float(np.mean([item[f"{condition}_seed_{seed}"] for seed in SEEDS]))
        output.append(item)
    return output


def _write_protocol(root: Path, info: Mapping[str, Any], manifest: Sequence[Mapping[str, Any]]) -> None:
    try:
        import torch
        runtime = {"torch_version": str(torch.__version__), "cuda_available": bool(torch.cuda.is_available()), "cuda_version": str(getattr(torch.version, "cuda", None)), "deterministic_algorithms": bool(torch.are_deterministic_algorithms_enabled()), "cudnn_deterministic": bool(torch.backends.cudnn.deterministic), "cudnn_benchmark": bool(torch.backends.cudnn.benchmark)}
    except Exception:
        runtime = {"torch_version": "UNAVAILABLE", "cuda_available": False, "cuda_version": None, "deterministic_algorithms": None, "cudnn_deterministic": None, "cudnn_benchmark": None}
    input_manifest = root / "inputs/input_manifest.json"
    input_content = None
    if input_manifest.is_file():
        input_content = json.loads(input_manifest.read_text(encoding="utf-8")).get("input_content_sha256")
    atomic_json(root / "protocol.json", {"protocol_id": "v7-local-context-pilot-v1", "git_head": git_head(), "source_root": str(SOURCE_ROOT), "input": {"train_windows": info["training_windows"], "validation_windows": info["validation_windows"], "train_sources": info["training_sources"], "validation_sources": info["validation_sources"], "train_real": info["training_real"], "train_fake": info["training_fake"], "validation_real": info["validation_real"], "validation_fake": info["validation_fake"], "source_support_sha256": sha256(SOURCE_ROOT / "support/window_support.json"), "training_manifest_sha256": sha256(SOURCE_ROOT / "models/training_window_manifest.csv"), "validation_manifest_sha256": sha256(SOURCE_ROOT / "manifests/validation_window_manifest.csv"), "selected_input_content_sha256": input_content}, "graph": {"history_only": True, "minimum_history_overlap": MINIMUM_HISTORY_OVERLAP, "k_neighbors": K_NEIGHBORS, "distance_multiplier": DISTANCE_MULTIPLIER, "rewire_max_attempts_multiplier": MAX_REWIRE_ATTEMPTS_MULTIPLIER, "nodes": "retained history-local groups", "state_shape": [5, 4], "empty_message": "masked zero message only when no induced edge"}, "conditions": {"SUMMARY_BASELINE": "existing SetAModel/SUMMARY_SET trained independently in this output", "SELF_TEMPORAL": "context model with every neighbor slot replaced by source node", "NEIGHBOR_TEMPORAL": "frozen history graph and aligned neighbor states", "NEIGHBOR_REWIRED": "frozen degree-preserving directed edge swaps", "NEIGHBOR_TIME_SHUFFLED": "frozen non-identity permutation on neighbor time axis"}, "model": {"context_architecture": "4-16-8 state; 32-16-8 relation; Conv1d 16-16-8; 44-16-1 head", "epochs": EPOCHS, "optimizer": "Adam", "learning_rate": LEARNING_RATE, "weight_decay": WEIGHT_DECAY, "seeds": list(SEEDS), "threshold": "logit >= 0", "loss": "source/class weighted BCE at window level"}, "runtime": runtime, "evaluation": {"primary": "NEIGHBOR_TEMPORAL minus SELF_TEMPORAL source-macro AUROC", "bootstrap_seed": BOOTSTRAP_SEED, "bootstrap_replicates": BOOTSTRAP_REPLICATES, "validation": "frozen source128 validation manifest", "offline_support_limitation": True}, "manifest_count": len(manifest)})


def _smoke(root: Path, manifest: Sequence[Mapping[str, Any]], train_rows: Sequence[Mapping[str, Any]], device: str) -> dict[str, Any]:
    smoke_rows = list(train_rows[:4])
    if {int(row["label"]) for row in smoke_rows} != {0, 1}:
        smoke_rows = list(train_rows[:8])
    weights = periodic.source_class_weights(smoke_rows)
    standardizer = periodic.fit_standardizer("SET_A", [np.asarray(row["features"]["SET_A"], dtype=np.float64) for row in smoke_rows], weights)
    records = []
    for condition in CONDITIONS:
        batch = _load_batches(root, manifest, smoke_rows, standardizer, "SELF_TEMPORAL" if condition == "SUMMARY_BASELINE" else condition); batch["window_weights"] = periodic.source_class_weights(batch["window_rows"])
        model, fit = _train_one(condition, batch, SEEDS[0], device, epochs=1)
        before = _forward(model, condition, batch, device).detach().cpu().numpy()
        record = {"condition": condition, "parameter_count": parameter_count(model), "scores_finite": bool(np.all(np.isfinite(before))), "epochs": fit["epochs"], "initial_state_hash": fit["initial_state_hash"]}
        reloaded = _model_state_load(condition, {"state_dict": state_dict_numpy(model)}, device)
        after = _forward(reloaded, condition, batch, device).detach().cpu().numpy(); record["reload_max_abs"] = float(np.max(np.abs(before - after))); record["passed"] = bool(record["scores_finite"] and record["reload_max_abs"] <= 1e-5); records.append(record)
        del model, reloaded
    result = {"status": "PASS" if all(item["passed"] for item in records) else "FAILED", "rows": len(smoke_rows), "conditions": records, "device": device}
    atomic_json(root / "smoke/summary.json", result); return result


def _write_report(root: Path, info: Mapping[str, Any], summary: Mapping[str, Any], records: Sequence[Mapping[str, Any]], timings: Mapping[str, float], smoke: Mapping[str, Any]) -> Path:
    metrics = summary.get("metrics", [])
    mean = {(str(item["split"]), str(item["condition"])): item for item in metrics if str(item["seed"]) == "MEAN_LOGIT"}
    lines = ["# V7 历史邻域局部时空关系 pilot", "", "本 pilot 固定 source128 的 R 五时刻 SET_A 状态、训练/验证窗口、标签、时间间隔和源隔离，只在历史半段构建 retained local-group 邻域；没有重新运行前端或使用目标时刻选择邻居。", "", "## 结果先行", "", f"- 实际模型：{len(records)}/{len(CONDITIONS) * len(SEEDS)}；训练窗口 {info['training_windows']}（{info['training_real']} real / {info['training_fake']} fake），验证窗口 {info['validation_windows']}（{info['validation_real']} real / {info['validation_fake']} fake），训练/验证 source 无交集。", f"- 有效 R 支持：{info['all_r_valid_rows']}/{info['all_r_support_rows']} 行；无支撑仍作为缺失保留，没有填零或因无邻居删除。", f"- smoke：{smoke.get('status')}；设备：{summary.get('device')}。", "", "| 条件 | 验证 source-macro AUROC | pooled AUROC | AP | Precision | Recall | F1 | ACC | TN/FP/FN/TP |", "|---|---:|---:|---:|---:|---:|---:|---:|---|"]
    for condition in CONDITIONS:
        item = mean.get(("validation", condition), {}); lines.append(f"| {condition} | {item.get('source_macro_auroc')} | {item.get('pooled_auroc')} | {item.get('pooled_ap')} | {item.get('precision')} | {item.get('recall')} | {item.get('f1')} | {item.get('accuracy')} | {item.get('tn')}/{item.get('fp')}/{item.get('fn')}/{item.get('tp')} |")
    lines += ["", "训练集三 seed 平均 logit 指标（仅拟合诊断，不是泛化证据）：", "", "| 条件 | source-macro AUROC | pooled AUROC | AP | F1 | ACC |", "|---|---:|---:|---:|---:|---:|"]
    for condition in CONDITIONS:
        item = mean.get(("train", condition), {}); lines.append(f"| {condition} | {item.get('source_macro_auroc')} | {item.get('pooled_auroc')} | {item.get('pooled_ap')} | {item.get('f1')} | {item.get('accuracy')} |")
    lines += ["", "正确邻域相对自身、重连、时间打乱和摘要基线的主差值均未得到稳定正向证据；完整逐 seed、逐 source 指标见评价 CSV。"]
    lines += ["", "## 预声明配对", "", "| 比较 | source 数 | 均值差 | 95% CI |", "|---|---:|---:|---|"]
    for item in summary.get("comparisons", []): lines.append(f"| {item['name']} | {item['source_count']} | {item['mean']} | [{item['ci95'][0]}, {item['ci95'][1]}] |")
    parameter_counts = {str(item["condition"]): int(item.get("parameter_count", -1)) for item in records}
    graph_manifest = root / "inputs/graph_manifest.json"
    graph_rows = json.loads(graph_manifest.read_text(encoding="utf-8")) if graph_manifest.is_file() else []
    changed_rewire = sum(float(item.get("rewire", {}).get("changed_edge_ratio", 0.0)) > 0 for item in graph_rows)
    nonidentity_time = sum(item.get("time_permutation") != [0, 1, 2, 3, 4] for item in graph_rows)
    lines += ["", "## 图覆盖", "", f"- 历史节点是同一窗口、同一 R cohort、同一坐标系中的 retained local group；图构建只使用历史帧质心。五时刻有效节点数、原始/诱导边数、d0、重连变化率见 `inputs/graph_manifest.json`。", f"- 训练 source 数：{len(info['training_sources'])}；验证 source 数：{len(info['validation_sources'])}；图没有邻居的节点仍进入评分，空消息只由 mask 产生。", f"- 选定图窗口 {len(graph_rows)}；实际发生重连的窗口 {changed_rewire}；非恒等时间排列窗口 {nonidentity_time}。", "- `NEIGHBOR_REWIRED` 的有向双边交换只在合法小图上执行，不能重连时原图保留并在 manifest 中标记；`NEIGHBOR_TIME_SHUFFLED` 保留邻居状态多重集合而改变共同时间对应。", "", "## 边界", "", "- 五时刻支撑是离线事后有效性筛选，不是严格在线因果输入。", "- 节点是稀疏局部组，邻居距离仅用于历史图选边，不是伪造判定或精确物理米制证据。", "- 这不是完整图网络、空间定位或物理规律验证；验证集经过此前开发比较，只是开发性结果。", "- 分数是窗口级 logit，消息/局部分数不解释为伪造概率或空间真值。", "", "## 成本与产物", "", f"- 参数量：{parameter_counts}。阶段耗时（秒）：graph={timings.get('graph')}, smoke={timings.get('smoke')}, train={timings.get('train')}, evaluate={timings.get('evaluate')}, report={timings.get('report')}。", "- 逐 seed 与逐 source 结果：`evaluation/metrics.csv`、`evaluation/per_source_metrics.csv`；配对 bootstrap：`evaluation/paired_comparisons.csv`。", "- 主要产物：`protocol.json`、`inputs/input_manifest.json`、`inputs/graph_manifest.json`、`inputs/window_inputs/`、`coverage/r_support_coverage.csv`、`models/fold_models.json`、`scores/`、`evaluation/`、`final_status.json`。", "- 未访问旧 R7/V5；未修改正式 src；未下载数据；未重跑 tracking、depth、pose、segmentation 或前端。", ""]
    path = root / "report.md"; path.write_text("\n".join(lines), encoding="utf-8"); return path


def run(root: Path = OUTPUT_ROOT, *, device: str = "cuda", resume: bool = True) -> dict[str, Any]:
    root.mkdir(parents=True, exist_ok=True); (root / "state").mkdir(parents=True, exist_ok=True); started = time.perf_counter(); timings: dict[str, float] = {}
    atomic_json(root / "state/launch.json", {"git_head": git_head(), "pid": os.getpid(), "device": device, "started_unix": time.time()})
    try:
        train_rows, validation_rows, loaded = _load_selection(); info = loaded["info"]
        progress(root, "graph", "RUNNING", 0, len(train_rows) + len(validation_rows))
        graph_started = time.perf_counter(); built = build_inputs(root, train_rows, validation_rows, loaded["r_support"], resume=resume); timings["graph"] = time.perf_counter() - graph_started
        manifest = built["manifest"]
        input_content_sha256 = _input_content_sha256(root, manifest)
        input_manifest_path = root / "inputs/input_manifest.json"
        input_manifest = json.loads(input_manifest_path.read_text(encoding="utf-8")) if input_manifest_path.is_file() else {}
        input_manifest["input_content_sha256"] = input_content_sha256
        atomic_json(input_manifest_path, input_manifest)
        _write_protocol(root, info, manifest)
        smoke_started = time.perf_counter(); smoke = json.loads((root / "smoke/summary.json").read_text(encoding="utf-8")) if resume and (root / "smoke/summary.json").is_file() else _smoke(root, manifest, train_rows, device); timings["smoke"] = time.perf_counter() - smoke_started
        if smoke.get("status") != "PASS": raise RuntimeError("SMOKE_FAILED")
        standardizer = periodic.fit_standardizer("SET_A", [np.asarray(row["features"]["SET_A"], dtype=np.float64) for row in train_rows], periodic.source_class_weights(train_rows))
        model_path = root / "models/fold_models.json"; existing = json.loads(model_path.read_text(encoding="utf-8")).get("records", []) if resume and model_path.is_file() else []; records: list[dict[str, Any]] = []; unverified = []
        for old in existing:
            condition, seed = str(old.get("condition", "")), int(old.get("seed", -1)); expected = _model_record_identity(root, manifest, train_rows, validation_rows, condition, seed, standardizer, input_content_sha256) if condition in CONDITIONS and seed in SEEDS else None
            if old.get("status") == "TRAIN_COMPLETE" and expected and _record_matches(old, expected): records.append(old)
            elif old: unverified.append({key: old.get(key) for key in ("condition", "seed", "status")})
        if unverified: atomic_json(root / "state/unverified_model_records.json", {"reason": "missing_or_mismatched_input_identity", "records": unverified})
        complete = {(str(item["condition"]), int(item["seed"])) for item in records}; total = len(CONDITIONS) * len(SEEDS); train_started = time.perf_counter()
        for condition in CONDITIONS:
            batch = _load_batches(root, manifest, train_rows, standardizer, "SELF_TEMPORAL" if condition == "SUMMARY_BASELINE" else condition); batch["window_weights"] = periodic.source_class_weights(batch["window_rows"])
            for seed in SEEDS:
                key = (condition, int(seed))
                if key in complete: continue
                model, fit = _train_one(condition, batch, seed, device)
                record = {"condition": condition, "seed": int(seed), "status": "TRAIN_COMPLETE", "parameter_count": parameter_count(model), "fit": fit, "standardization": standardizer.as_dict(), "input_identity": _model_record_identity(root, manifest, train_rows, validation_rows, condition, seed, standardizer, input_content_sha256), "state_dict": state_dict_numpy(model)}
                records.append(record); complete.add(key); root.joinpath("models").mkdir(parents=True, exist_ok=True); atomic_json(model_path, {"conditions": list(CONDITIONS), "seeds": list(SEEDS), "epochs": EPOCHS, "records": records}); progress(root, "train", "RUNNING", len(complete), total, condition=condition, seed=seed)
                del model
        timings["train"] = time.perf_counter() - train_started
        if len(complete) != total: raise RuntimeError(f"MODEL_COUNT_INCOMPLETE:{len(complete)}/{total}")
        evaluate_started = time.perf_counter(); summary = evaluate(root, manifest, train_rows, validation_rows, records, standardizer, device); timings["evaluate"] = time.perf_counter() - evaluate_started
        report_started = time.perf_counter(); timings["report"] = 0.0; report_path = _write_report(root, info, summary, records, timings, smoke); timings["report"] = time.perf_counter() - report_started; _write_report(root, info, summary, records, timings, smoke)
        atomic_json(root / "final_status.json", {"status": "COMPLETE", "model_count": len(records), "expected_model_count": total, "train_windows": len(train_rows), "validation_windows": len(validation_rows), "report": str(report_path), "timings_s": timings, "git_head": git_head()}); progress(root, "report", "COMPLETE", 1, 1, model_count=len(records), elapsed_s=time.perf_counter() - started); return {"status": "COMPLETE", "report": str(report_path), "summary": summary, "timings_s": timings}
    except BaseException as exc:
        atomic_json(root / "state/failure.json", {"status": "FAILED", "error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()}); atomic_json(root / "final_status.json", {"status": "FAILED", "error": f"{type(exc).__name__}: {exc}", "git_head": git_head(), "elapsed_s": time.perf_counter() - started}); progress(root, "failed", "FAILED", 0, 1, error=f"{type(exc).__name__}: {exc}"); raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT); parser.add_argument("--device", default="cuda"); parser.add_argument("--resume", action="store_true"); args = parser.parse_args(); result = run(args.output_root, device=args.device, resume=args.resume); print(json.dumps({"status": result["status"], "report": result["report"], "timings_s": result["timings_s"]}, ensure_ascii=False, indent=2)); return 0


if __name__ == "__main__":
    raise SystemExit(main())
