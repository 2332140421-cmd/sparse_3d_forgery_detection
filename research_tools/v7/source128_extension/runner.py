"""Run the fixed V7 64-to-128 source training extension.

The extension deliberately reuses the audited source-learning-curve and
periodic re-query implementations.  It freezes one source ordering, links
validated old particle caches, and only computes the missing source inputs.
Large media, particle arrays and model records stay on the data disk.
"""

from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import json
import math
import os
import shutil
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter, defaultdict
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from research_tools.v7.observation_density_diagnostic import run_diagnostic as diagnostic


REPO_ROOT = Path(__file__).resolve().parents[3]
DATA_ROOT = Path("/root/autodl-tmp/data/sparse_3d_forgery_detection")
SOURCE_ROOT = DATA_ROOT / "derived/v7_activityforensics_source_learning_curve_v1"
BASELINE_ROOT = DATA_ROOT / "derived/v7_activityforensics_attention_pooling_pilot_v1"
OUTPUT_ROOT = DATA_ROOT / "derived/v7_activityforensics_source128_extension_v1"
DATASET_ROOT = DATA_ROOT / "datasets/v7_core_candidates/activityforensics_charades_v1"
STAGING_NAME = "staging_base_manifest"
SELECTION_SEED = 20260909
ORDERING_SEED = 20260909
BASE_SOURCE_COUNT = 64
ADDED_SOURCE_COUNT = 64
VALIDATION_SOURCE_COUNT = 16
TOTAL_SOURCE_COUNT = 128
EPOCHS = 200
MODEL_SEEDS = (20260909, 20260910, 20260911)
FRONTEND_BUDGET_S = 7200.0
TRAINING_BUDGET_S = 900.0
BOOTSTRAP_SEED = 20260909
BOOTSTRAP_REPLICATES = 10_000
STOP_REQUESTED = False


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


def _safe(value: str) -> str:
    return "".join(character if character.isalnum() or character in "._-" else "_" for character in value)


def progress(root: Path, stage: str, status: str, completed: int, total: int, **extra: Any) -> None:
    atomic_json(root / "progress.json", {"stage": stage, "status": status, "completed": int(completed), "total": int(total), "updated_unix": time.time(), **extra})


class Budget:
    def __init__(self, root: Path, name: str, budget_s: float) -> None:
        self.path = root / "state" / f"{name}_budget.json"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        old = json.loads(self.path.read_text(encoding="utf-8")) if self.path.is_file() else {}
        if old and abs(float(old.get("budget_s", budget_s)) - float(budget_s)) > 1e-9:
            raise ValueError(f"{name.upper()}_BUDGET_ARGUMENT_MISMATCH")
        self.recovered_interrupted_process: dict[str, Any] | None = None
        self.before = float(old.get("cumulative_s", 0.0))
        self.estimated_reserved_before = float(old.get("estimated_reserved_s", 0.0))
        if old.get("process_state") == "RUNNING":
            try:
                previous_before = float(old["elapsed_before_this_process_s"])
                previous_start = float(old["process_start_unix"])
                previous_checkpoint = float(old.get("process_elapsed_s", 0.0))
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"{name.upper()}_BUDGET_ACTIVE_PROCESS_RECORD_INCOMPLETE") from exc
            if not all(math.isfinite(value) for value in (previous_before, previous_start, previous_checkpoint)):
                raise ValueError(f"{name.upper()}_BUDGET_ACTIVE_PROCESS_RECORD_NONFINITE")
            now_unix = time.time()
            wall_upper = max(0.0, now_unix - previous_start)
            recovered_elapsed = max(wall_upper, previous_checkpoint)
            expected_checkpoint = previous_before + previous_checkpoint
            old_cumulative = float(old.get("cumulative_s", expected_checkpoint))
            if not math.isclose(old_cumulative, expected_checkpoint, rel_tol=0.0, abs_tol=1e-3):
                raise ValueError(f"{name.upper()}_BUDGET_RUNNING_CHECKPOINT_INCONSISTENT")
            self.before = expected_checkpoint
            newly_reserved = max(0.0, recovered_elapsed - previous_checkpoint)
            self.estimated_reserved_before += newly_reserved
            self.recovered_interrupted_process = {
                "previous_process_start_unix": previous_start,
                "detected_unix": now_unix,
                "wall_elapsed_upper_s": wall_upper,
                "last_checkpoint_process_elapsed_s": previous_checkpoint,
                "charged_process_elapsed_s": recovered_elapsed,
                "new_estimated_reservation_s": newly_reserved,
                "accounting": "elapsed_before_this_process + last checkpoint; reserve only wall/checkpoint delta",
            }
        self.budget_s = float(budget_s)
        self.started_monotonic = time.monotonic()
        self.started_unix = time.time()
        self.save(None, process_state="RUNNING")

    def elapsed(self) -> float:
        return self.before + max(0.0, time.monotonic() - self.started_monotonic)

    def remaining(self) -> float:
        return max(0.0, self.budget_s - self.accounted_upper())

    def accounted_upper(self) -> float:
        return self.before + self.estimated_reserved_before + max(0.0, time.monotonic() - self.started_monotonic)

    def save(self, reason: str | None = None, *, process_state: str | None = None, **extra: Any) -> None:
        if process_state is None:
            process_state = "RUNNING" if reason is None else "STOPPED"
        atomic_json(self.path, {
            "budget_s": self.budget_s,
            "elapsed_before_this_process_s": self.before,
            "process_start_unix": self.started_unix,
            "process_elapsed_s": max(0.0, time.monotonic() - self.started_monotonic),
            "cumulative_s": self.elapsed(),
            "estimated_reserved_s": self.estimated_reserved_before,
            "budget_accounted_upper_s": self.accounted_upper(),
            "stop_reason": reason,
            "process_state": process_state,
            "recovered_interrupted_process": self.recovered_interrupted_process,
            **extra,
        })


def record_supervisor_exit(
    root: Path,
    *,
    runner_pid: int,
    wrapper_pid: int,
    exit_code: int,
    started_unix: float,
    log_path: str,
    screen_session: str,
    command: str,
) -> dict[str, Any]:
    ended_unix = time.time()
    record = {
        "status": "CAPTURED",
        "runner_pid": int(runner_pid),
        "wrapper_pid": int(wrapper_pid),
        "exit_code": int(exit_code),
        "started_unix": float(started_unix),
        "ended_unix": ended_unix,
        "elapsed_wall_s": max(0.0, ended_unix - float(started_unix)),
        "log_path": str(log_path),
        "screen_session": str(screen_session),
        "command": str(command),
    }
    atomic_json(root / "state/supervisor_exit.json", record)

    launch_path = root / "state/launch.json"
    final_path = root / "final_status.json"
    try:
        launch = json.loads(launch_path.read_text(encoding="utf-8"))
        final = json.loads(final_path.read_text(encoding="utf-8")) if final_path.is_file() else {}
    except (OSError, ValueError, TypeError):
        return record
    is_our_launch = int(launch.get("pid", -1)) == int(runner_pid) and float(launch.get("started_unix", 0.0)) >= float(started_unix) - 1.0
    if is_our_launch:
        for name in ("frontend", "training"):
            budget_path = root / "state" / f"{name}_budget.json"
            if not budget_path.is_file():
                continue
            try:
                budget = json.loads(budget_path.read_text(encoding="utf-8"))
                if budget.get("process_state") != "RUNNING" or float(budget.get("process_start_unix", 0.0)) < float(started_unix) - 1.0:
                    continue
                before = float(budget["elapsed_before_this_process_s"])
                checkpoint = float(budget.get("process_elapsed_s", 0.0))
                previous_cumulative = float(budget.get("cumulative_s", before + checkpoint))
                if not math.isclose(previous_cumulative, before + checkpoint, rel_tol=0.0, abs_tol=1e-3):
                    continue
                wall_upper = max(0.0, ended_unix - float(budget["process_start_unix"]))
                settled_elapsed = max(wall_upper, checkpoint)
                additional_reservation = max(0.0, settled_elapsed - checkpoint)
                budget["estimated_reserved_s"] = float(budget.get("estimated_reserved_s", 0.0)) + additional_reservation
                budget["budget_accounted_upper_s"] = previous_cumulative + float(budget["estimated_reserved_s"])
                budget["process_state"] = "INTERRUPTED"
                budget["stop_reason"] = "SUPERVISOR_CAPTURED_RUNNER_EXIT"
                budget["supervisor_exit_code"] = int(exit_code)
                budget["supervisor_exit_unix"] = ended_unix
                budget["settled_process_elapsed_upper_s"] = settled_elapsed
                budget["new_estimated_reservation_s"] = additional_reservation
                atomic_json(budget_path, budget)
            except (OSError, ValueError, TypeError, KeyError):
                continue
        if final.get("status") == "RUNNING":
            final.update({
                "status": "INTERRUPTED",
                "stop_reason": "RUNNER_EXITED_BEFORE_FINAL_STATUS",
                "wrapper_exit_code": int(exit_code),
                "wrapper_exit_unix": ended_unix,
                "updated_unix": ended_unix,
            })
            atomic_json(final_path, final)
    return record


@contextmanager
def run_lock(root: Path):
    path = root / "state" / "run.lock"
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("w", encoding="utf-8")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        handle.close()
        raise RuntimeError("SOURCE128_RUN_ALREADY_ACTIVE") from exc
    handle.write(f"pid={os.getpid()}\n")
    handle.flush()
    try:
        yield
    finally:
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        handle.close()


def _curve_module():
    from research_tools.v7.source_learning_curve import runner as curve
    return curve


def _base_ids() -> tuple[list[str], list[str], dict[str, Any]]:
    payload = json.loads((SOURCE_ROOT / "manifests/training_subsets.json").read_text(encoding="utf-8"))
    base = [str(item) for item in payload["subsets"][str(ORDERING_SEED)][str(BASE_SOURCE_COUNT)]]
    validation = [str(item) for item in payload["validation_sources"]]
    if len(base) != BASE_SOURCE_COUNT or len(validation) != VALIDATION_SOURCE_COUNT:
        raise ValueError(f"BASE_SPLIT_COUNT_MISMATCH:{len(base)}:{len(validation)}")
    if set(base) & set(validation):
        raise ValueError("BASE_VALIDATION_OVERLAP")
    return base, validation, payload


def _candidate_rows() -> list[dict[str, Any]]:
    curve = _curve_module()
    base, validation, _ = _base_ids()
    excluded = set(base) | set(validation)
    rows: list[dict[str, Any]] = []
    for row in curve._candidate_rows():
        source = str(row["charades_source_id"])
        if source in excluded:
            continue
        variants = [dict(item) for item in row.get("fake_variants", []) if str(item.get("official_split")) == "train" and str(item.get("lineage_status")) == "EXACT"]
        if variants:
            rows.append({"charades_source_id": source, "fake_variants": variants, "available_cells": row.get("available_cells", [])})
    return rows


def _stable_key(source: str) -> str:
    return hashlib.sha256(f"v7-source128-extension|{SELECTION_SEED}|{source}".encode("utf-8")).hexdigest()


def _choose_variant(row: Mapping[str, Any]) -> dict[str, Any]:
    variants = sorted((dict(item) for item in row["fake_variants"] if str(item.get("official_split")) == "train" and str(item.get("lineage_status")) == "EXACT"), key=lambda item: (str(item.get("generator")), str(item.get("manipulation_operation")), str(item.get("activityforensics_file"))))
    if not variants:
        raise ValueError(f"NO_EXACT_TRAIN_VARIANT:{row['charades_source_id']}")
    return variants[0]


def freeze_plan(root: Path, *, overwrite: bool = False) -> dict[str, Any]:
    """Freeze the original 64, validation 16, and exactly 64 added sources."""
    required = (
        root / "protocol.json",
        root / "manifests/source_split_manifest.csv",
        root / "manifests/training_subsets.json",
        root / "manifests/training_source_manifest.csv",
        root / "manifests/added_source_manifest.csv",
        root / "manifests/validation_window_manifest.csv",
    )
    if all(path.is_file() for path in required) and not overwrite:
        return json.loads((root / "protocol.json").read_text(encoding="utf-8"))
    base, validation, old_subsets = _base_ids()
    candidates = sorted(_candidate_rows(), key=lambda row: _stable_key(str(row["charades_source_id"])))
    if len(candidates) < ADDED_SOURCE_COUNT:
        raise RuntimeError(f"LEGAL_TRAIN_CANDIDATES_INSUFFICIENT:{len(candidates)}<{ADDED_SOURCE_COUNT}")
    added = [{**dict(row), "fake_variant": _choose_variant(row), "selection_rank": index + 1, "selection_hash": _stable_key(str(row["charades_source_id"]))} for index, row in enumerate(candidates[:ADDED_SOURCE_COUNT])]
    added_ids = [str(row["charades_source_id"]) for row in added]
    if set(added_ids) & (set(base) | set(validation)):
        raise ValueError("ADDED_SOURCE_SPLIT_OVERLAP")
    all_train = base + added_ids
    source_rows: list[dict[str, Any]] = []
    old_rows = []
    with (SOURCE_ROOT / "manifests/source_split_manifest.csv").open(newline="", encoding="utf-8") as handle:
        old_rows = list(csv.DictReader(handle))
    old_by_source = {str(row["source_id"]): row for row in old_rows}
    for index, source in enumerate(base, 1):
        row = dict(old_by_source[source]); row.update(source_origin="BASE_64", extension_order=index)
        source_rows.append(row)
    for index, source in enumerate(validation):
        row = dict(old_by_source[source]); row.update(source_origin="VALIDATION", extension_order=index)
        source_rows.append(row)
    added_manifest: list[dict[str, Any]] = []
    for index, item in enumerate(added, 1):
        variant = item["fake_variant"]
        row = {
            "source_id": str(item["charades_source_id"]),
            "split_role": "TRAIN_POOL_ADDED",
            "pool_index": str(BASE_SOURCE_COUNT + index),
            "validation_index": "",
            "selected_cell": json.dumps([variant["generator"], variant["manipulation_operation"]], ensure_ascii=False),
            "generator": str(variant["generator"]),
            "manipulation_operation": json.dumps(str(variant["manipulation_operation"])),
            "activityforensics_file": str(variant["activityforensics_file"]),
            "official_split": str(variant["official_split"]),
            "lineage_status": str(variant["lineage_status"]),
            "source_origin": "ADDED_64",
            "extension_order": str(index),
        }
        source_rows.append(row)
        added_manifest.append({**row, "selection_seed": SELECTION_SEED, "selection_hash": str(item["selection_hash"]), "selection_rank": index})
    root.joinpath("manifests").mkdir(parents=True, exist_ok=True)
    write_csv(root / "manifests/source_split_manifest.csv", source_rows)
    # The audited media/window helpers use the source-learning-curve split
    # loader.  Keep a local, single-ordering subset manifest rather than
    # reaching back into the old experiment at runtime.
    atomic_json(root / "manifests/training_subsets.json", {
        "selection_seed": SELECTION_SEED,
        "ordering_seeds": [ORDERING_SEED],
        "validation_sources": validation,
        "max_training_pool": all_train,
        "subsets": {str(ORDERING_SEED): {str(TOTAL_SOURCE_COUNT): all_train}},
        "nested": True,
        "label_independent": True,
        "base_training_sources": base,
        "added_training_sources": added_ids,
    })
    write_csv(root / "manifests/training_source_manifest.csv", [
        {"source_id": source, "split_role": "TRAIN_POOL", "source_origin": "BASE_64", "training_order": index + 1, "selection_seed": ""}
        for index, source in enumerate(base)
    ] + [
        {"source_id": str(row["source_id"]), "split_role": "TRAIN_POOL", "source_origin": "ADDED_64", "training_order": BASE_SOURCE_COUNT + index + 1, "selection_seed": SELECTION_SEED, "selection_rank": row["selection_rank"]}
        for index, row in enumerate(added_manifest)
    ])
    write_csv(root / "manifests/added_source_manifest.csv", added_manifest)
    validation_ids = set(validation)
    validation_rows = []
    old_validation_ids = set(json.loads((BASELINE_ROOT / "input_manifest.json").read_text(encoding="utf-8"))["validation_window_ids"])
    old_support = json.loads((SOURCE_ROOT / "support/window_support.json").read_text(encoding="utf-8"))
    for row in old_support:
        if str(row.get("mode")) == "R" and str(row.get("window_id")) in old_validation_ids and str(row.get("source_id")) in validation_ids:
            validation_rows.append({key: row.get(key) for key in ("window_id", "source_id", "role", "label", "annotation_category", "offset_s", "interval_start_s", "interval_end_s", "frame_indices", "timestamps_s", "valid_unit_count", "support_status")})
    write_csv(root / "manifests/validation_window_manifest.csv", validation_rows)
    if len(validation_rows) != 83:
        raise ValueError(f"VALIDATION_WINDOW_COUNT_MISMATCH:{len(validation_rows)}")
    protocol = {
        "protocol_id": "v7-source128-extension-v1",
        "git_head": git_head(),
        "source_learning_curve_root": str(SOURCE_ROOT),
        "baseline_root": str(BASELINE_ROOT),
        "dataset": {"name": "ActivityForensics/ActivityForensics + Charades", "revision": "a34d4b7b04b0f3f3e26ba900adc367218667c581", "official_split": "train_only"},
        "selection": {"base_source_count": BASE_SOURCE_COUNT, "added_source_count": ADDED_SOURCE_COUNT, "total_training_source_count": TOTAL_SOURCE_COUNT, "validation_source_count": VALIDATION_SOURCE_COUNT, "selection_seed": SELECTION_SEED, "algorithm": "candidate exact-lineage train sources excluding original train/validation, stable SHA-256 order", "label_independent": True, "ordering_seed": ORDERING_SEED, "base_sources": base, "added_sources": added_ids, "validation_sources": validation},
        "windows": {"protocol": "reuse source-learning-curve MANIP50, b=0/0.5/1.0, periodic R support", "validation_window_count": len(validation_rows), "validation_window_ids_sha256": sha256(root / "manifests/validation_window_manifest.csv")},
        "model": {"condition": "SUMMARY_SET / SET_A", "implementation": "research_tools.v7.multi_order_sequence_probe.model.SetAModel", "epochs": EPOCHS, "seeds": list(MODEL_SEEDS), "optimizer": "Adam", "learning_rate": 1e-3, "weight_decay": 1e-4, "weighted_bce": "equal source and class total weight", "standardization": "128-source training rows only", "threshold": "logit >= 0", "parameter_count": 569},
        "evaluation": {"primary": "128-source validation source-macro AUROC minus fixed 64-source MEAN_BASELINE", "validation_population": "same 14 dual-role sources / 83 windows as baseline", "bootstrap": {"unit": "source", "replicates": BOOTSTRAP_REPLICATES, "seed": BOOTSTRAP_SEED}, "pooled": "auxiliary"},
        "budgets": {"frontend_s": FRONTEND_BUDGET_S, "training_s": TRAINING_BUDGET_S, "download_separate": True},
        "boundaries": ["development validation, not sealed test", "no old R7/V5", "no formal src changes", "no frontend method changes", "large media, particles and models remain on data disk"],
        "source_learning_curve_reference": {"training_subsets_sha256": sha256(SOURCE_ROOT / "manifests/training_subsets.json"), "source_split_sha256": sha256(SOURCE_ROOT / "manifests/source_split_manifest.csv"), "baseline_input_manifest_sha256": sha256(BASELINE_ROOT / "input_manifest.json")},
    }
    atomic_json(root / "protocol.json", protocol)
    atomic_json(root / "manifests/freeze_identity.json", {"protocol_sha256": sha256(root / "protocol.json"), "source_split_sha256": sha256(root / "manifests/source_split_manifest.csv"), "training_source_sha256": sha256(root / "manifests/training_source_manifest.csv"), "added_source_sha256": sha256(root / "manifests/added_source_manifest.csv"), "validation_window_sha256": sha256(root / "manifests/validation_window_manifest.csv"), "git_head": git_head()})
    atomic_json(root / "state/selection_summary.json", {"candidate_count": len(candidates), "base_count": len(base), "added_count": len(added), "validation_count": len(validation), "old_subsets_sha256": sha256(SOURCE_ROOT / "manifests/training_subsets.json"), "old_subsets_ordering_20260909_64": old_subsets["subsets"][str(ORDERING_SEED)][str(BASE_SOURCE_COUNT)]})
    return protocol


def _set_curve_roots(root: Path) -> Any:
    curve = _curve_module()
    curve.STAGING_ROOT = root / STAGING_NAME
    return curve


def download(root: Path, workers: int = 4) -> dict[str, Any]:
    """Download the frozen 144-source media with bounded, resumable requests.

    The older helper is intentionally not used here because its HF streaming
    call has no useful per-file read timeout.  This adapter keeps the same
    official URLs and checksum/size checks, writes each result incrementally,
    and leaves a failed file as a ``.part`` for a later explicit resume.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from huggingface_hub import HfApi
    from research_tools.v7.data.remote_zip64 import RangeClient, exact_basename, extract_member, read_central_directory

    index_path = DATASET_ROOT / "acquisition/targeted_review_v1/charades_range_index.json"
    archive = json.loads(index_path.read_text(encoding="utf-8"))
    entries = {str(item["name"]).split("/")[-1]: item for item in archive["entries"]}
    metadata_errors: list[str] = []
    info = None
    for attempt in range(1, 4):
        try:
            info = HfApi().dataset_info("ActivityForensics/ActivityForensics", revision="a34d4b7b04b0f3f3e26ba900adc367218667c581", files_metadata=True)
            break
        except Exception as exc:
            metadata_errors.append(f"attempt={attempt}:{type(exc).__name__}: {exc}")
            if attempt < 3:
                time.sleep(2.0)
    if info is None:
        error = "HF_METADATA_UNAVAILABLE; " + " | ".join(metadata_errors)
        atomic_json(root / "state/download_error.json", {"status": "DOWNLOAD_BLOCKED", "error": error, "updated_unix": time.time()})
        progress(root, "download", "DOWNLOAD_BLOCKED", 0, 0, error=error)
        raise RuntimeError(error)
    metadata = {str(item.rfilename): item for item in info.siblings}
    split_rows, _ = _load_split_for_extension(root)
    rows: list[dict[str, Any]] = []
    for split in split_rows:
        sid = str(split["source_id"])
        fake_rel = str(split["activityforensics_file"])
        item = metadata.get(fake_rel)
        rows.append({"source_id": sid, "role": "real", "path": str(DATASET_ROOT / "source/charades/videos" / f"{sid}.mp4"), "remote_member": f"Charades_v1/{sid}.mp4", "expected_bytes": entries.get(f"{sid}.mp4", {}).get("uncompressed_size"), "expected_sha256": None, "status": "PLANNED"})
        rows.append({"source_id": sid, "role": "fake", "path": str(DATASET_ROOT / "source/activityforensics/raw" / fake_rel), "remote_member": fake_rel, "expected_bytes": int(item.size) if item else None, "expected_sha256": item.lfs.sha256 if item and item.lfs else None, "status": "PLANNED"})
    rows.sort(key=lambda row: (str(row["source_id"]), str(row["role"])))
    result_path = root / "acquisition/media_manifest.json"
    previous = {}
    if result_path.is_file():
        try:
            previous = {str(item["source_id"]) + "::" + str(item["role"]): dict(item) for item in json.loads(result_path.read_text(encoding="utf-8")).get("results", [])}
        except (OSError, ValueError, KeyError, TypeError):
            previous = {}
    results: dict[str, dict[str, Any]] = {}
    for row in rows:
        key = str(row["source_id"]) + "::" + str(row["role"])
        old = previous.get(key, {})
        if old.get("status") in {"DOWNLOADED", "MATERIALIZED", "REUSED_EXISTING"} and Path(str(row["path"])).is_file():
            results[key] = {**dict(row), **old}
        else:
            results[key] = dict(row)
    progress(root, "download", "RUNNING", sum(item.get("status") in {"DOWNLOADED", "MATERIALIZED", "REUSED_EXISTING"} for item in results.values()), len(rows))

    def _download_hf(row: Mapping[str, Any]) -> dict[str, Any]:
        output = dict(row); path = Path(str(row["path"])); expected_size = row.get("expected_bytes"); expected_hash = row.get("expected_sha256")
        if expected_size is None or not expected_hash:
            output.update(status="REMOTE_METADATA_MISSING")
            return output
        if path.is_file() and path.stat().st_size == int(expected_size) and sha256(path) == str(expected_hash):
            output.update(status="REUSED_EXISTING", local_bytes=path.stat().st_size, local_sha256=sha256(path)); return output
        path.parent.mkdir(parents=True, exist_ok=True); part = path.with_suffix(path.suffix + ".part")
        url = "https://huggingface.co/datasets/ActivityForensics/ActivityForensics/resolve/a34d4b7b04b0f3f3e26ba900adc367218667c581/" + urllib.parse.quote(str(row["remote_member"]), safe="/") + "?download=true"
        try:
            start = part.stat().st_size if part.is_file() else 0
            request = urllib.request.Request(url, headers={"Range": f"bytes={start}-"} if start else {})
            with urllib.request.urlopen(request, timeout=30.0) as response, part.open("ab" if start and getattr(response, "status", 200) == 206 else "wb") as handle:
                if start and getattr(response, "status", 200) != 206:
                    start = 0
                for block in iter(lambda: response.read(1024 * 1024), b""):
                    handle.write(block)
            if part.stat().st_size != int(expected_size) or sha256(part) != str(expected_hash):
                output.update(status="CHECKSUM_FAILURE", local_bytes=part.stat().st_size, local_sha256=sha256(part)); return output
            os.replace(part, path); output.update(status="DOWNLOADED", local_bytes=path.stat().st_size, local_sha256=sha256(path)); return output
        except (OSError, urllib.error.URLError, TimeoutError) as exc:
            output.update(status="DOWNLOAD_FAILURE", error=f"{type(exc).__name__}: {exc}", partial_bytes=part.stat().st_size if part.is_file() else 0); return output

    central_url = "https://ai2-public-datasets.s3-us-west-2.amazonaws.com/charades/Charades_v1.zip"
    central_client = None
    central = None
    central_errors: list[str] = []
    for attempt in range(1, 4):
        try:
            central_client = RangeClient(central_url, timeout=60.0)
            central, _ = read_central_directory(central_client)
            break
        except Exception as exc:
            central_errors.append(f"attempt={attempt}:{type(exc).__name__}: {exc}")
            central_client = None
            central = None
            if attempt < 3:
                time.sleep(2.0)
    if central_client is None or central is None:
        error = "CHARADES_CENTRAL_DIRECTORY_UNAVAILABLE; " + " | ".join(central_errors)
        atomic_json(root / "state/download_error.json", {"status": "DOWNLOAD_BLOCKED", "error": error, "updated_unix": time.time()})
        progress(root, "download", "DOWNLOAD_BLOCKED", sum(item.get("status") in {"DOWNLOADED", "MATERIALIZED", "REUSED_EXISTING"} for item in results.values()), len(rows), error=error)
        raise RuntimeError(error)
    def _download_real(row: Mapping[str, Any]) -> dict[str, Any]:
        nonlocal central
        output = dict(row); path = Path(str(row["path"])); expected = row.get("expected_bytes")
        if expected is None:
            output.update(status="SOURCE_MEMBER_MISSING"); return output
        if path.is_file() and path.stat().st_size == int(expected):
            output.update(status="REUSED_EXISTING", local_bytes=path.stat().st_size, local_sha256=sha256(path)); return output
        try:
            # The ZIP central directory is read once per process; member
            # ranges remain bounded by RangeClient's existing timeout.
            part = path.with_suffix(path.suffix + ".part"); path.parent.mkdir(parents=True, exist_ok=True)
            member = exact_basename(central, f"{row['source_id']}.mp4")
            # Give each member its own HTTP session so independent ranges can
            # proceed concurrently without sharing mutable requests state.
            member_client = RangeClient("https://ai2-public-datasets.s3-us-west-2.amazonaws.com/charades/Charades_v1.zip", timeout=60.0)
            # ``extract_member`` writes the bounded temporary payload and
            # atomically renames ``*.part`` to the requested target itself.
            # Check the final target, rather than the already-renamed
            # temporary path (the latter used to turn successful extracts
            # into spurious FileNotFoundError failures).
            extract_member(member_client, member, part)
            if not path.is_file() or path.stat().st_size != int(expected):
                output.update(status="CHECKSUM_FAILURE", local_bytes=path.stat().st_size if path.is_file() else 0); return output
            output.update(status="MATERIALIZED", local_bytes=path.stat().st_size, local_sha256=sha256(path)); return output
        except Exception as exc:
            output.update(status="DOWNLOAD_FAILURE", error=f"{type(exc).__name__}: {exc}"); return output

    def one(row: Mapping[str, Any]) -> dict[str, Any]:
        return _download_real(row) if row["role"] == "real" else _download_hf(row)

    completed = sum(item.get("status") in {"DOWNLOADED", "MATERIALIZED", "REUSED_EXISTING"} for item in results.values())
    with ThreadPoolExecutor(max_workers=max(1, int(workers))) as pool:
        futures = {pool.submit(one, row): key for key, row in results.items() if results[key].get("status") not in {"DOWNLOADED", "MATERIALIZED", "REUSED_EXISTING"}}
        for future in as_completed(futures):
            key = futures[future]; value = future.result(); results[key] = value; completed += int(value.get("status") in {"DOWNLOADED", "MATERIALIZED", "REUSED_EXISTING"})
            atomic_json(result_path, {"revision": "a34d4b7b04b0f3f3e26ba900adc367218667c581", "results": sorted(results.values(), key=lambda item: (str(item["source_id"]), str(item["role"])))})
            progress(root, "download", "RUNNING", completed, len(rows), last_source=value.get("source_id"), last_role=value.get("role"), last_status=value.get("status"))
    final_rows = sorted(results.values(), key=lambda item: (str(item["source_id"]), str(item["role"])))
    atomic_json(result_path, {"revision": "a34d4b7b04b0f3f3e26ba900adc367218667c581", "results": final_rows})
    status_counts = dict(Counter(str(item.get("status")) for item in final_rows))
    status = "COMPLETE" if all(item.get("status") in {"DOWNLOADED", "MATERIALIZED", "REUSED_EXISTING"} for item in final_rows) else "PARTIAL"
    summary = {"status": status, "count": len(final_rows), "completed": sum(item.get("status") in {"DOWNLOADED", "MATERIALIZED", "REUSED_EXISTING"} for item in final_rows), "status_counts": status_counts}
    atomic_json(root / "state/download_summary.json", summary)
    progress(root, "download", status, summary["completed"], len(final_rows), status_counts=status_counts)
    return summary


def _load_split_for_extension(root: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    with (root / "manifests/source_split_manifest.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    subsets = json.loads((root / "manifests/training_subsets.json").read_text(encoding="utf-8"))
    return rows, subsets


def build_and_prepare(root: Path) -> dict[str, Any]:
    curve = _set_curve_roots(root)
    summary = curve.build_windows(root)
    # The extension adds sources to the lexicographic population, which can
    # otherwise renumber the legacy pair IDs even though the source/video,
    # frame and PTS identities are unchanged.  Preserve the old pair IDs for
    # the 80 legacy sources so their validated particle caches remain
    # reusable; new source IDs keep the extension-generated IDs.
    legacy_pairs_path = SOURCE_ROOT / STAGING_NAME / "manifests/selected_pairs.json"
    current_pairs_path = curve.STAGING_ROOT / "manifests/selected_pairs.json"
    if legacy_pairs_path.is_file() and current_pairs_path.is_file():
        legacy_pairs = {str(item["source_id"]): str(item["pair_id"]) for item in json.loads(legacy_pairs_path.read_text(encoding="utf-8"))}
        current_pairs = json.loads(current_pairs_path.read_text(encoding="utf-8"))
        pair_aliases = {str(item["source_id"]): legacy_pairs[str(item["source_id"])] for item in current_pairs if str(item["source_id"]) in legacy_pairs}
        for item in current_pairs:
            if str(item["source_id"]) in pair_aliases:
                item["pair_id"] = pair_aliases[str(item["source_id"])]
        atomic_json(current_pairs_path, current_pairs)
        current_windows_path = curve.STAGING_ROOT / "manifests/window_manifest.json"
        current_windows = json.loads(current_windows_path.read_text(encoding="utf-8"))
        for item in current_windows:
            if str(item["source_id"]) in pair_aliases:
                item["pair_id"] = pair_aliases[str(item["source_id"])]
        atomic_json(current_windows_path, current_windows)
        atomic_json(root / "manifests/input_pairs.json", current_pairs)
        atomic_json(root / "state/legacy_pair_id_aliases.json", pair_aliases)
    prepared = curve.prepare_periodic(root)
    # periodic.prepare writes its own generic protocol; restore the extension
    # protocol as the authoritative identity after the manifest is expanded.
    saved = root / "state/extension_protocol.json"
    if not saved.is_file():
        raise RuntimeError("EXTENSION_PROTOCOL_SNAPSHOT_MISSING")
    atomic_json(root / "protocol.json", json.loads(saved.read_text(encoding="utf-8")))
    frozen = json.loads((root / "state/selection_summary.json").read_text(encoding="utf-8"))
    atomic_json(root / "state/window_plan_summary.json", {"source_window_summary": summary, "periodic_prepare": prepared, "source_count": len({str(x["source_id"]) for x in json.loads((root / "manifests/parents.json").read_text(encoding="utf-8"))}), "subwindow_count": len(json.loads((root / "manifests/subwindows.json").read_text(encoding="utf-8"))), "selection_summary": frozen})
    return {"build": summary, "prepare": prepared}


def _save_extension_protocol(root: Path) -> None:
    path = root / "protocol.json"
    saved = root / "state/extension_protocol.json"
    if path.is_file() and not saved.is_file():
        atomic_json(saved, json.loads(path.read_text(encoding="utf-8")))


def verify_legacy_cache_identity(root: Path) -> dict[str, Any]:
    """Validate old 80-source particles before allowing frontend reuse."""
    curve = _set_curve_roots(root)
    from research_tools.v7.periodic_requery_probe import runner as periodic
    old_base, old_validation, _ = _base_ids()
    old_sources = set(old_base) | set(old_validation)
    parents = json.loads((root / "manifests/parents.json").read_text(encoding="utf-8"))
    subwindows = json.loads((root / "manifests/subwindows.json").read_text(encoding="utf-8"))
    old_particles = SOURCE_ROOT / "particles"
    root_particles = root / "particles"
    root_particles.mkdir(parents=True, exist_ok=True)
    linked: list[str] = []
    invalid: list[str] = []
    for source in sorted(old_sources):
        for role in ("real", "fake"):
            parent_id = f"{source}::{role}::MANIP50"
            source_dir = old_particles / _safe(parent_id)
            target = root_particles / _safe(parent_id)
            if not source_dir.is_dir():
                invalid.append(f"{parent_id}:MISSING_DIRECTORY")
                continue
            if not target.exists():
                target.symlink_to(source_dir, target_is_directory=True)
            elif not target.is_symlink() and target.resolve() != source_dir.resolve():
                invalid.append(f"{parent_id}:TARGET_CONFLICT")
                continue
            linked.append(parent_id)
    by_parent = defaultdict(list)
    for row in subwindows:
        by_parent[str(row["parent_id"])].append(row)
    valid_parents = 0
    for parent in parents:
        if str(parent["source_id"]) not in old_sources:
            continue
        generated = periodic._cached_parent_outputs(parent, by_parent[str(parent["parent_id"])], root)
        if generated is None:
            invalid.append(f"{parent['parent_id']}:IDENTITY_OR_SEQUENCE_MISMATCH")
        else:
            valid_parents += 1
    result = {"old_source_count": len(old_sources), "expected_parent_count": len(old_sources) * 2, "linked_parent_count": len(linked), "valid_parent_count": valid_parents, "invalid": invalid, "status": "PASS" if not invalid and valid_parents == len(old_sources) * 2 else "FAIL"}
    atomic_json(root / "checks/legacy_cache_identity.json", result)
    if result["status"] != "PASS":
        raise RuntimeError("LEGACY_CACHE_IDENTITY_FAILED")
    return result


def frontend(root: Path, budget_s: float = FRONTEND_BUDGET_S, resume: bool = True, budget_name: str = "frontend") -> dict[str, Any]:
    curve = _set_curve_roots(root)
    return curve.run_frontend(root, budget_s=budget_s, resume=resume, budget_name=budget_name)


def features(root: Path) -> dict[str, Any]:
    curve = _set_curve_roots(root)
    return curve.run_features(root)


def _feature_artifacts_complete(root: Path) -> bool:
    """Only reuse features when they cover the current, complete window plan."""
    try:
        subwindows = _load_subwindows(root)
        expected = {str(row["window_id"]): row for row in subwindows}
        if not expected:
            return False
        frontend_rows = json.loads((root / "frontend/results.json").read_text(encoding="utf-8"))
        frontend_by_id = {str(row["window_id"]): row for row in frontend_rows}
        if len(frontend_by_id) != len(expected) or set(frontend_by_id) != set(expected):
            return False
        identity_fields = (
            "source_id", "role", "parent_id", "pair_id", "kind", "offset_s",
            "frame_indices", "timestamps_s", "interval_start_s", "interval_end_s",
            "label", "annotation_category", "video_path",
        )
        for window_id, row in frontend_by_id.items():
            if str(row.get("status")) != "FRONTEND_COMPLETE":
                return False
            if any(row.get(field) != expected[window_id].get(field) for field in identity_fields):
                return False
            if not row.get("o_sequence_prefix") or not row.get("r_sequence_prefix"):
                return False
        support_rows = json.loads((root / "support/window_support.json").read_text(encoding="utf-8"))
        keys = [(str(row["window_id"]), str(row["mode"])) for row in support_rows]
        expected_keys = {(window_id, mode) for window_id in expected for mode in ("O", "R")}
        if len(keys) != len(expected_keys) or set(keys) != expected_keys:
            return False
        for row in support_rows:
            window_id = str(row["window_id"])
            source = expected[window_id]
            frontend = frontend_by_id[window_id]
            for field in ("source_id", "role", "pair_id", "kind", "offset_s", "interval_start_s", "interval_end_s", "label", "annotation_category"):
                if row.get(field) != source.get(field):
                    return False
            if str(row.get("support_status")) in {"MISSING_FRONTEND", "FEATURE_FAILED"}:
                return False
            prefix_key = "o_sequence_prefix" if str(row["mode"]) == "O" else "r_sequence_prefix"
            if str(row.get("particle_prefix")) != str(frontend.get(prefix_key)):
                return False
            frame_indices = [int(value) for value in row.get("frame_indices", [])]
            timestamps = [float(value) for value in row.get("timestamps_s", [])]
            if not frame_indices or len(frame_indices) != len(timestamps) or len(frame_indices) != int(frontend.get("o_frame_count" if row["mode"] == "O" else "r_frame_count", -1)):
                return False
            if any(right <= left for left, right in zip(frame_indices, frame_indices[1:])) or any(right <= left for left, right in zip(timestamps, timestamps[1:])):
                return False
            frame_to_time = dict(zip(frame_indices, timestamps))
            for frame, timestamp in zip(source["frame_indices"], source["timestamps_s"]):
                if int(frame) not in frame_to_time or not math.isclose(frame_to_time[int(frame)], float(timestamp), rel_tol=0.0, abs_tol=1e-7):
                    return False
            if not isinstance(row.get("features"), Mapping) or "SET_A" not in row["features"]:
                return False
            valid_unit_count = int(row.get("valid_unit_count", 0))
            feature = row["features"]["SET_A"]
            if valid_unit_count > 0:
                if str(row.get("support_status")) != "VALID" or feature is None:
                    return False
                values = np.asarray(feature, dtype=np.float64)
                if values.size == 0 or not np.all(np.isfinite(values)):
                    return False
            elif str(row.get("support_status")) == "VALID":
                return False
        summary = json.loads((root / "support/support_summary.json").read_text(encoding="utf-8"))
        return int(summary.get("subwindow_count", -1)) == len(expected) and int(summary.get("mode_rows", -1)) == len(expected_keys)
    except (OSError, ValueError, TypeError, KeyError):
        return False


def _features_with_budget(root: Path, budget: Budget) -> dict[str, Any]:
    """Charge feature extraction to the frozen combined feature/training budget."""
    started = time.monotonic()
    state_path = root / "state/feature_runtime.json"
    old = json.loads(state_path.read_text(encoding="utf-8")) if state_path.is_file() else {"attempts": []}
    try:
        result = features(root)
        status = "COMPLETE" if _feature_artifacts_complete(root) else "INCOMPLETE"
        reason = None if status == "COMPLETE" else "FEATURE_ARTIFACT_IDENTITY_OR_COMPLETENESS_FAILURE"
    except BaseException as exc:
        status = "FAILED"
        reason = f"{type(exc).__name__}: {exc}"
        elapsed = max(0.0, time.monotonic() - started)
        attempts = list(old.get("attempts", []))
        attempts.append({"status": status, "elapsed_s": elapsed, "error": reason, "updated_unix": time.time()})
        atomic_json(state_path, {"attempts": attempts, "cumulative_feature_elapsed_s": sum(float(item.get("elapsed_s", 0.0)) for item in attempts)})
        budget.save("FEATURE_EXTRACTION_FAILED", process_state="FAILED", stage="features", feature_elapsed_s=elapsed, error=reason)
        raise
    elapsed = max(0.0, time.monotonic() - started)
    attempts = list(old.get("attempts", []))
    attempts.append({"status": status, "elapsed_s": elapsed, "updated_unix": time.time()})
    atomic_json(state_path, {"attempts": attempts, "cumulative_feature_elapsed_s": sum(float(item.get("elapsed_s", 0.0)) for item in attempts)})
    budget.save(reason, process_state=status, stage="features", feature_elapsed_s=elapsed, feature_status=status)
    return {"status": status, "summary": result, "elapsed_s": elapsed, "error": reason}


def _load_rows(root: Path) -> list[dict[str, Any]]:
    rows = json.loads((root / "support/window_support.json").read_text(encoding="utf-8"))
    return [dict(row) for row in rows if str(row.get("mode")) == "R" and int(row.get("valid_unit_count", 0)) > 0 and row.get("label") in (0, 1) and row.get("features", {}).get("SET_A") is not None]


def _effective_training_rows(rows: Sequence[Mapping[str, Any]], planned_sources: Sequence[str]) -> tuple[list[dict[str, Any]], set[str], list[str]]:
    planned = {str(source) for source in planned_sources}
    selected = [dict(row) for row in rows if str(row.get("source_id")) in planned]
    effective = {str(row["source_id"]) for row in selected}
    return selected, effective, sorted(planned - effective)


def _load_parents(root: Path) -> list[dict[str, Any]]:
    return [dict(row) for row in json.loads((root / "manifests/parents.json").read_text(encoding="utf-8"))]


def _load_subwindows(root: Path) -> list[dict[str, Any]]:
    return [dict(row) for row in json.loads((root / "manifests/subwindows.json").read_text(encoding="utf-8"))]


def _load_validation_baseline() -> list[dict[str, Any]]:
    path = BASELINE_ROOT / "scores/validation_window_scores.csv"
    with path.open(newline="", encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _budget(root: Path, budget_s: float, budget_name: str = "training") -> Budget:
    return Budget(root, budget_name, budget_s)


def train(root: Path, budget_s: float = TRAINING_BUDGET_S, device: str = "cuda", resume: bool = True, budget_name: str = "training") -> dict[str, Any]:
    import torch
    from research_tools.v7.periodic_requery_probe import runner as periodic

    budget = _budget(root, budget_s, budget_name)
    rows = _load_rows(root)
    protocol = json.loads((root / "protocol.json").read_text(encoding="utf-8"))
    train_sources = [str(x) for x in protocol["selection"]["base_sources"] + protocol["selection"]["added_sources"]]
    train_rows, observed_sources, missing_sources = _effective_training_rows(rows, train_sources)
    if not train_rows:
        raise RuntimeError("NO_VALID_128_TRAINING_ROWS")
    counts = Counter(int(row["label"]) for row in train_rows)
    if set(counts) != {0, 1}:
        raise RuntimeError(f"TRAINING_LABEL_CLASS_MISSING:{dict(counts)}")
    values = [{**dict(row), "features": {"SET_A": np.asarray(row["features"]["SET_A"], dtype=np.float64)}} for row in train_rows]
    weights = periodic.source_class_weights(values)
    standardizer = periodic.fit_standardizer("SET_A", [row["features"]["SET_A"] for row in values], weights)
    batch = periodic.make_batch("SET_A", values, standardizer)
    batch["window_weights"] = weights
    write_csv(root / "models/training_window_manifest.csv", [
        {"window_id": row["window_id"], "source_id": row["source_id"], "role": row["role"], "label": row["label"], "valid_unit_count": row.get("valid_unit_count", 0), "included": True}
        for row in train_rows
    ])
    coverage_by_source = Counter(str(row["source_id"]) for row in train_rows)
    write_csv(root / "models/training_source_coverage.csv", [
        {"source_id": source, "selected_for_training": True, "valid_label_eligible_window_count": int(coverage_by_source.get(source, 0)), "effective_training_source": source in observed_sources}
        for source in train_sources
    ])
    model_path = root / "models/fold_models.json"
    records = json.loads(model_path.read_text(encoding="utf-8")).get("records", []) if resume and model_path.is_file() else []
    done = {(int(item["seed"]), int(item.get("source_count", 0))) for item in records if item.get("status") == "TRAIN_COMPLETE"}
    root.joinpath("models").mkdir(parents=True, exist_ok=True)
    total = len(MODEL_SEEDS)
    progress(root, "train", "RUNNING", len(done), total, planned_source_count=TOTAL_SOURCE_COUNT, effective_training_source_count=len(observed_sources), missing_training_source_count=len(missing_sources), missing_training_sources=missing_sources, training_window_count=len(train_rows), training_real_count=counts[0], training_fake_count=counts[1], cumulative_elapsed_s=budget.elapsed())
    for seed in MODEL_SEEDS:
        if (int(seed), TOTAL_SOURCE_COUNT) in done:
            continue
        if budget.remaining() <= 0:
            budget.save("TRAINING_BUDGET_EXHAUSTED", process_state="TRAINING_BUDGET_EXHAUSTED", completed_models=len(done), total_models=total)
            progress(root, "train", "TRAINING_BUDGET_EXHAUSTED", len(done), total, cumulative_elapsed_s=budget.elapsed())
            return {"status": "TRAINING_BUDGET_EXHAUSTED", "completed_models": len(done), "total_models": total}
        started = time.perf_counter()
        model, fit = periodic.train_one("SET_A", batch, seed=int(seed), device=device, epochs=EPOCHS)
        record = {"condition": "SUMMARY_SET", "base_condition": "SET_A", "ordering_seed": ORDERING_SEED, "source_count": TOTAL_SOURCE_COUNT, "planned_source_count": TOTAL_SOURCE_COUNT, "effective_training_source_count": len(observed_sources), "missing_training_source_count": len(missing_sources), "missing_training_sources": missing_sources, "seed": int(seed), "status": "TRAIN_COMPLETE", "training_sources": train_sources, "effective_training_sources": sorted(observed_sources), "training_window_count": len(train_rows), "training_real_count": counts[0], "training_fake_count": counts[1], "parameter_count": periodic.parameter_count("SET_A"), "standardization": standardizer.as_dict(), "fit": fit, "elapsed_s": time.perf_counter() - started, "device": str(torch.device(device)), "state_dict": periodic.model_state(model)}
        records.append(record)
        atomic_json(model_path, {"condition": "SUMMARY_SET", "base_condition": "SET_A", "ordering_seed": ORDERING_SEED, "source_count": TOTAL_SOURCE_COUNT, "epochs": EPOCHS, "seeds": list(MODEL_SEEDS), "records": records})
        done.add((int(seed), TOTAL_SOURCE_COUNT))
        budget.save(None, process_state="RUNNING", last_seed=int(seed), completed_models=len(done), total_models=total)
        progress(root, "train", "RUNNING", len(done), total, last_seed=int(seed), cumulative_elapsed_s=budget.elapsed())
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    budget.save(None, process_state="COMPLETE", completed_models=len(done), total_models=total)
    progress(root, "train", "COMPLETE", len(done), total, cumulative_elapsed_s=budget.elapsed())
    return {"status": "COMPLETE", "completed_models": len(done), "total_models": total, "planned_source_count": TOTAL_SOURCE_COUNT, "effective_training_source_count": len(observed_sources), "missing_training_source_count": len(missing_sources), "missing_training_sources": missing_sources, "training_window_count": len(train_rows), "training_real_count": counts[0], "training_fake_count": counts[1]}


def _metrics(labels: Sequence[int], scores: Sequence[float]) -> dict[str, Any]:
    from research_tools.v7.periodic_requery_probe.runner import _ap, _auroc, _classification
    return {"window_count": len(labels), "real_count": int(sum(int(x) == 0 for x in labels)), "fake_count": int(sum(int(x) == 1 for x in labels)), "pooled_auroc": _auroc(labels, scores), "pooled_ap": _ap(labels, scores), **_classification(labels, scores)}


def _source_values(rows: Sequence[Mapping[str, Any]], score_key: str) -> dict[str, float]:
    from research_tools.v7.periodic_requery_probe.runner import _auroc
    output: dict[str, float] = {}
    for source in sorted({str(row["source_id"]) for row in rows}):
        subset = [row for row in rows if str(row["source_id"]) == source]
        value = _auroc([int(row["label"]) for row in subset], [float(row[score_key]) for row in subset])
        if value is not None:
            output[source] = float(value)
    return output


def _bootstrap(values: Mapping[str, float], other: Mapping[str, float] | None = None) -> dict[str, Any]:
    keys = sorted(set(values) & (set(other) if other is not None else set(values)))
    if not keys:
        return {"source_count": 0, "mean": None, "ci95": [None, None], "seed": BOOTSTRAP_SEED, "replicates": BOOTSTRAP_REPLICATES, "sources": []}
    left = np.asarray([values[key] for key in keys], dtype=np.float64)
    if other is None:
        diff = left
    else:
        diff = left - np.asarray([other[key] for key in keys], dtype=np.float64)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    indices = rng.integers(0, len(keys), size=(BOOTSTRAP_REPLICATES, len(keys)))
    draws = diff[indices].mean(axis=1)
    return {"source_count": len(keys), "sources": keys, "mean": float(diff.mean()), "ci95": [float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))], "seed": BOOTSTRAP_SEED, "replicates": BOOTSTRAP_REPLICATES}


def evaluate(root: Path, device: str = "cuda") -> dict[str, Any]:
    import torch
    from research_tools.v7.periodic_requery_probe import runner as periodic

    all_rows = _load_rows(root)
    protocol = json.loads((root / "protocol.json").read_text(encoding="utf-8"))
    planned_train_sources = set(str(x) for x in protocol["selection"]["base_sources"] + protocol["selection"]["added_sources"])
    train_rows = [row for row in all_rows if str(row["source_id"]) in planned_train_sources]
    training_manifest = root / "models/training_window_manifest.csv"
    if training_manifest.is_file():
        with training_manifest.open(newline="", encoding="utf-8") as handle:
            expected_training_ids = {str(item["window_id"]) for item in csv.DictReader(handle)}
        if expected_training_ids != {str(row["window_id"]) for row in train_rows}:
            raise RuntimeError("TRAINING_EVALUATION_WINDOW_SET_MISMATCH")
    rows = all_rows
    baseline_rows = _load_validation_baseline()
    baseline_by_id = {str(row["window_id"]): row for row in baseline_rows}
    validation_ids = set(baseline_by_id)
    rows = [row for row in rows if str(row["window_id"]) in validation_ids and str(row["source_id"]) in set(protocol["selection"]["validation_sources"])]
    if len(rows) != len(validation_ids):
        raise RuntimeError(f"VALIDATION_FEATURE_SET_MISMATCH:{len(rows)}:{len(validation_ids)}")
    records = json.loads((root / "models/fold_models.json").read_text(encoding="utf-8"))["records"]
    score_rows = [{"window_id": str(row["window_id"]), "source_id": str(row["source_id"]), "role": str(row["role"]), "label": int(row["label"]), "valid_unit_count": int(row.get("valid_unit_count", 0)), "baseline_seed_20260909": float(baseline_by_id[str(row["window_id"])]["MEAN_BASELINE_seed_20260909"]), "baseline_seed_20260910": float(baseline_by_id[str(row["window_id"])]["MEAN_BASELINE_seed_20260910"]), "baseline_seed_20260911": float(baseline_by_id[str(row["window_id"])]["MEAN_BASELINE_seed_20260911"]), "baseline_mean_logit": float(baseline_by_id[str(row["window_id"])]["MEAN_BASELINE_MEAN_LOGIT"])} for row in rows]
    training_score_rows = [{"window_id": str(row["window_id"]), "source_id": str(row["source_id"]), "role": str(row["role"]), "label": int(row["label"]), "valid_unit_count": int(row.get("valid_unit_count", 0))} for row in train_rows]
    standardizers: dict[int, Any] = {}
    for record in records:
        seed = int(record["seed"])
        standardizers[seed] = periodic.FeatureStandardizer("SET_A", np.asarray(record["standardization"]["mean"], dtype=np.float64), np.asarray(record["standardization"]["scale"], dtype=np.float64), tuple(int(x) for x in record["standardization"].get("zero_variance_dimensions", [])))
        model = periodic._model_from_record("SET_A", record, device)
        batch_rows = [{**dict(row), "features": {"SET_A": np.asarray(row["features"]["SET_A"], dtype=np.float64)}} for row in rows]
        batch = periodic.make_batch("SET_A", batch_rows, standardizers[seed], require_labels=False)
        scores = periodic.score_batch("SET_A", model, batch, device)
        for item, value in zip(score_rows, scores):
            item[f"source128_seed_{seed}"] = float(value)
        train_batch_rows = [{**dict(row), "features": {"SET_A": np.asarray(row["features"]["SET_A"], dtype=np.float64)}} for row in train_rows]
        train_batch = periodic.make_batch("SET_A", train_batch_rows, standardizers[seed], require_labels=False)
        train_scores = periodic.score_batch("SET_A", model, train_batch, device)
        for item, value in zip(training_score_rows, train_scores):
            item[f"source128_seed_{seed}"] = float(value)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    seed_names = [f"source128_seed_{seed}" for seed in MODEL_SEEDS]
    for item in score_rows:
        if not all(name in item for name in seed_names):
            raise RuntimeError(f"MISSING_128_SEED_SCORE:{item['window_id']}")
        item["source128_mean_logit"] = float(np.mean([item[name] for name in seed_names]))
        item["baseline_mean_logit"] = float(np.mean([item[f"baseline_seed_{seed}"] for seed in MODEL_SEEDS]))
    for item in training_score_rows:
        if not all(name in item for name in seed_names):
            raise RuntimeError(f"MISSING_128_TRAIN_SEED_SCORE:{item['window_id']}")
        item["source128_mean_logit"] = float(np.mean([item[name] for name in seed_names]))
    root.joinpath("scores").mkdir(parents=True, exist_ok=True)
    write_csv(root / "scores/validation_window_scores.csv", score_rows)
    write_csv(root / "scores/train_window_scores.csv", training_score_rows)
    labels = [int(row["label"]) for row in score_rows]
    metrics_rows: list[dict[str, Any]] = []
    for condition, key in (("MEAN_BASELINE_64", "baseline_mean_logit"), ("SUMMARY_SET_128", "source128_mean_logit")):
        source = _source_values(score_rows, key)
        metrics_rows.append({"condition": condition, "seed": "MEAN_LOGIT", "source_count": len(source), "source_macro_auroc": _bootstrap(source), **_metrics(labels, [float(row[key]) for row in score_rows])})
        for seed in MODEL_SEEDS:
            skey = f"baseline_seed_{seed}" if condition == "MEAN_BASELINE_64" else f"source128_seed_{seed}"
            source_seed = _source_values(score_rows, skey)
            metrics_rows.append({"condition": condition, "seed": int(seed), "source_count": len(source_seed), "source_macro_auroc": _bootstrap(source_seed), **_metrics(labels, [float(row[skey]) for row in score_rows])})
    write_csv(root / "evaluation/metrics.csv", metrics_rows)
    training_metrics: list[dict[str, Any]] = []
    for seed in MODEL_SEEDS:
        key = f"source128_seed_{seed}"
        by_source = _source_values(training_score_rows, key)
        training_metrics.append({"condition": "SUMMARY_SET_128_TRAIN", "seed": int(seed), "source_count": len(by_source), **_metrics([int(row["label"]) for row in training_score_rows], [float(row[key]) for row in training_score_rows]), "source_macro_auroc_point_estimate": float(np.mean(list(by_source.values()))) if by_source else None})
    train_source_mean = _source_values(training_score_rows, "source128_mean_logit")
    training_metrics.append({"condition": "SUMMARY_SET_128_TRAIN", "seed": "MEAN_LOGIT", "source_count": len(train_source_mean), **_metrics([int(row["label"]) for row in training_score_rows], [float(row["source128_mean_logit"]) for row in training_score_rows]), "source_macro_auroc_point_estimate": float(np.mean(list(train_source_mean.values()))) if train_source_mean else None})
    baseline_summary = json.loads((BASELINE_ROOT / "evaluation/summary.json").read_text(encoding="utf-8"))
    baseline_training_metrics = [dict(row) for row in baseline_summary.get("metrics", []) if row.get("condition") == "MEAN_BASELINE" and row.get("split") == "train"]
    write_csv(root / "evaluation/train_metrics.csv", [
        {"condition": "MEAN_BASELINE_64_TRAIN", "seed": row.get("seed"), "source_count": row.get("source_count"), "window_count": row.get("window_count"), "real_count": row.get("real_count"), "fake_count": row.get("fake_count"), "source_macro_auroc": row.get("source_macro_auroc"), "pooled_auroc": row.get("pooled_auroc"), "pooled_ap": row.get("pooled_ap"), "precision": row.get("precision"), "recall": row.get("recall"), "f1": row.get("f1"), "accuracy": row.get("accuracy"), "tn": row.get("tn"), "fp": row.get("fp"), "fn": row.get("fn"), "tp": row.get("tp"), "origin": "saved attention-pooling pilot evaluation"}
        for row in baseline_training_metrics
    ] + [
        {"condition": row["condition"], "seed": row["seed"], "source_count": row["source_count"], "window_count": row["window_count"], "real_count": row["real_count"], "fake_count": row["fake_count"], "source_macro_auroc": row["source_macro_auroc_point_estimate"], "pooled_auroc": row["pooled_auroc"], "pooled_ap": row["pooled_ap"], "precision": row["precision"], "recall": row["recall"], "f1": row["f1"], "accuracy": row["accuracy"], "tn": row["tn"], "fp": row["fp"], "fn": row["fn"], "tp": row["tp"], "origin": "forward pass on training windows using saved 128-source model"}
        for row in training_metrics
    ])
    baseline_source = _source_values(score_rows, "baseline_mean_logit")
    new_source = _source_values(score_rows, "source128_mean_logit")
    per_source = []
    for source in sorted(set(baseline_source) | set(new_source)):
        subset = [row for row in score_rows if str(row["source_id"]) == source]
        item = {"source_id": source, "window_count": len(subset), "real_count": sum(int(row["label"]) == 0 for row in subset), "fake_count": sum(int(row["label"]) == 1 for row in subset), "baseline_auroc": baseline_source.get(source), "source128_auroc": new_source.get(source), "delta": (new_source[source] - baseline_source[source]) if source in baseline_source and source in new_source else None}
        for seed in MODEL_SEEDS:
            b = _source_values(score_rows, f"baseline_seed_{seed}").get(source); n = _source_values(score_rows, f"source128_seed_{seed}").get(source)
            item[f"delta_seed_{seed}"] = (n - b) if n is not None and b is not None else None
        per_source.append(item)
    write_csv(root / "evaluation/per_source_metrics.csv", per_source)
    paired = _bootstrap(new_source, baseline_source)
    summary = {"validation_population": {"source_count": len(new_source), "window_count": len(score_rows), "real_count": int(sum(x == 0 for x in labels)), "fake_count": int(sum(x == 1 for x in labels)), "window_ids": sorted(validation_ids)}, "training_population": {"planned_source_count": len(planned_train_sources), "effective_source_count": len({str(row['source_id']) for row in train_rows}), "window_count": len(train_rows), "real_count": sum(int(row["label"]) == 0 for row in train_rows), "fake_count": sum(int(row["label"]) == 1 for row in train_rows), "window_ids_file": str(root / "models/training_window_manifest.csv")}, "training_metrics": training_metrics, "baseline_training_metrics": baseline_training_metrics, "conditions": metrics_rows, "primary_comparison": {"name": "SUMMARY_SET_128_MINUS_MEAN_BASELINE_64", **paired}, "per_source": per_source, "baseline_source": baseline_source, "source128_source": new_source, "device": str(torch.device(device)), "model_count": len(records), "baseline_artifact": str(BASELINE_ROOT), "score_identity": "baseline MEAN_BASELINE seed scores reused; 128 scores from newly trained models"}
    atomic_json(root / "evaluation/summary.json", summary)
    return summary


def report(root: Path) -> dict[str, Any]:
    protocol = json.loads((root / "protocol.json").read_text(encoding="utf-8"))
    summary = json.loads((root / "evaluation/summary.json").read_text(encoding="utf-8")) if (root / "evaluation/summary.json").is_file() else {}
    frontend_rows = json.loads((root / "frontend/results.json").read_text(encoding="utf-8")) if (root / "frontend/results.json").is_file() else []
    support_rows = json.loads((root / "support/window_support.json").read_text(encoding="utf-8")) if (root / "support/window_support.json").is_file() else []
    model_records = json.loads((root / "models/fold_models.json").read_text(encoding="utf-8")).get("records", []) if (root / "models/fold_models.json").is_file() else []
    subwindows = _load_subwindows(root) if (root / "manifests/subwindows.json").is_file() else []
    counts = Counter(str(row.get("status")) for row in frontend_rows)
    support_valid = Counter(str(row.get("mode")) for row in support_rows if int(row.get("valid_unit_count", 0)) > 0)
    support_status_counts = Counter(f"{row.get('mode')}:{row.get('support_status')}" for row in support_rows)
    feature_errors = sum(str(row.get("support_status")) == "FEATURE_FAILED" for row in support_rows)
    r_train_rows = [row for row in support_rows if str(row.get("mode")) == "R" and int(row.get("valid_unit_count", 0)) > 0 and row.get("label") in (0, 1) and row.get("features", {}).get("SET_A") is not None]
    r_train_windows = {str(row["window_id"]) for row in r_train_rows}
    r_train_sources = {str(row["source_id"]) for row in r_train_rows}
    frontend_complete_ids = {str(row.get("window_id")) for row in frontend_rows if str(row.get("status")) == "FRONTEND_COMPLETE"}
    frontend_complete_parents = {str(row.get("parent_id")) for row in frontend_rows if str(row.get("status")) == "FRONTEND_COMPLETE"}
    feature_ready = _feature_artifacts_complete(root)
    media_rows = json.loads((root / "acquisition/media_manifest.json").read_text(encoding="utf-8")).get("results", []) if (root / "acquisition/media_manifest.json").is_file() else []
    media_counts = Counter(str(row.get("status")) for row in media_rows)
    download_error = json.loads((root / "state/download_error.json").read_text(encoding="utf-8")) if (root / "state/download_error.json").is_file() else {}
    smoke_frontend = json.loads((root / "smoke/frontend_summary.json").read_text(encoding="utf-8")) if (root / "smoke/frontend_summary.json").is_file() else {}
    smoke_train = json.loads((root / "smoke/summary.json").read_text(encoding="utf-8")) if (root / "smoke/summary.json").is_file() else {}
    selection = protocol["selection"]
    planned_source_count = len(selection["base_sources"]) + len(selection["added_sources"]) + len(selection["validation_sources"])
    expected_media_count = planned_source_count * 2
    media_success = {"DOWNLOADED", "MATERIALIZED", "REUSED_EXISTING"}
    media_complete = len(media_rows) == expected_media_count and expected_media_count > 0 and all(str(row.get("status")) in media_success for row in media_rows)
    media_completed_count = sum(str(row.get("status")) in media_success for row in media_rows)
    lines = ["# V7 64→128 source 训练扩容对照", "", "本报告为当前数据集内的开发性 source 扩容 pilot；固定 R、SET_A/SUMMARY_SET、平均局部聚合和原验证总体，不是 sealed test 或最终泛化结论。", "", "## 结论先行", ""]
    source_pairs = defaultdict(set)
    for item in media_rows:
        if str(item.get("status")) in {"DOWNLOADED", "MATERIALIZED", "REUSED_EXISTING"}:
            source_pairs[str(item.get("source_id"))].add(str(item.get("role")))
    complete_source_count = sum(roles == {"real", "fake"} for roles in source_pairs.values())
    lines += [f"- 冻结训练池：原 64 + 新增 {len(protocol['selection']['added_sources'])} = {len(protocol['selection']['base_sources']) + len(protocol['selection']['added_sources'])} source；验证：{len(protocol['selection']['validation_sources'])} source。", f"- 下载状态：{'COMPLETE' if media_complete else 'INCOMPLETE'}（{media_completed_count}/{expected_media_count} 项）；媒体状态：{dict(media_counts)}；成对可用 source：{complete_source_count}（计划 {planned_source_count}）；前端结果：{len(frontend_complete_ids)}/{len(subwindows)} 个子窗口、{len(frontend_complete_parents)} 个父片段完成；状态分布：{dict(counts)}。", f"- 支撑处理：R 有效几何/结构行 {support_valid.get('R', 0)}；R 标签合格且 SET_A 有效窗口 {len(r_train_windows)}，覆盖 {len(r_train_sources)}/{len(selection['base_sources']) + len(selection['added_sources'])} 个冻结训练 source；O/R 支撑状态分布：{dict(support_status_counts)}；特征执行错误 {feature_errors}；特征清单可复用：{feature_ready}。", f"- 正式模型：{sum(row.get('status') == 'TRAIN_COMPLETE' for row in model_records)}/3；端到端 smoke 仅作验收，不进入正式比较。"]
    if smoke_frontend or smoke_train:
        lines.append(f"- 端到端 smoke：frontend={smoke_frontend.get('status', '未记录')} source={smoke_frontend.get('source_id', '未记录')}；训练={smoke_train.get('status', '未记录')} epochs={smoke_train.get('epochs', '未记录')}；smoke 不进入正式比较。")
    if summary:
        primary = summary.get("primary_comparison", {})
        lines += [f"- 主比较 128−64 source-macro AUROC：{primary.get('mean')}，95% CI={primary.get('ci95')}。", "- 该区间若跨 0，不解释为稳定扩容收益。"]
        per_source = [row for row in summary.get("per_source", []) if row.get("delta") is not None]
        directions = {
            "higher": sum(float(row["delta"]) > 1e-12 for row in per_source),
            "tie": sum(abs(float(row["delta"])) <= 1e-12 for row in per_source),
            "lower": sum(float(row["delta"]) < -1e-12 for row in per_source),
        }
        lines.append(f"- 逐 source 主比较方向（128 相对 64）：提高 {directions['higher']}、持平 {directions['tie']}、下降 {directions['lower']}；此处单位为验证 source，不是窗口。")
        conditions = summary.get("conditions", [])
        for seed in MODEL_SEEDS:
            baseline_seed = next((row for row in conditions if row.get("condition") == "MEAN_BASELINE_64" and row.get("seed") == seed), None)
            extended_seed = next((row for row in conditions if row.get("condition") == "SUMMARY_SET_128" and row.get("seed") == seed), None)
            if baseline_seed is not None and extended_seed is not None:
                left = baseline_seed.get("source_macro_auroc", {}).get("mean")
                right = extended_seed.get("source_macro_auroc", {}).get("mean")
                if left is not None and right is not None:
                    seed_deltas = [float(row[f"delta_seed_{seed}"]) for row in per_source if row.get(f"delta_seed_{seed}") is not None]
                    seed_directions = {
                        "higher": sum(value > 1e-12 for value in seed_deltas),
                        "tie": sum(abs(value) <= 1e-12 for value in seed_deltas),
                        "lower": sum(value < -1e-12 for value in seed_deltas),
                    }
                    lines.append(f"- Seed {seed}: source-macro AUROC 64={left:.6f}, 128={right:.6f}, Δ={right-left:+.6f}; 逐 source 提高/持平/下降={seed_directions['higher']}/{seed_directions['tie']}/{seed_directions['lower']}。")
    lines += ["", "## 主验证指标（固定 14 source / 83 window；logit≥0）", "", "| condition | seed | windows | real | fake | source-macro AUROC | CI | pooled AUROC | AP | Precision | Recall | F1 | ACC |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in summary.get("conditions", []):
        macro = row.get("source_macro_auroc", {})
        cls = row.get("classification", row)
        lines.append(f"| {row.get('condition')} | {row.get('seed')} | {row.get('window_count')} | {row.get('real_count')} | {row.get('fake_count')} | {macro.get('mean')} | {macro.get('ci95')} | {row.get('pooled_auroc')} | {row.get('pooled_ap')} | {cls.get('precision')} | {cls.get('recall')} | {cls.get('f1')} | {cls.get('accuracy')} |")
    training_metrics = summary.get("training_metrics", [])
    baseline_training_metrics = summary.get("baseline_training_metrics", [])
    if training_metrics or baseline_training_metrics:
        lines += ["", "## 训练集拟合表现（样本总体不同，不作配对收益解释）", "", "| condition | seed | macro AUROC | macro source n | pooled AUROC | AP | Precision | Recall | F1 | ACC | windows (real/fake) |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for row in baseline_training_metrics:
            lines.append(f"| MEAN_BASELINE_64_TRAIN | {row.get('seed')} | {row.get('source_macro_auroc')} | {row.get('source_count')} | {row.get('pooled_auroc')} | {row.get('pooled_ap')} | {row.get('precision')} | {row.get('recall')} | {row.get('f1')} | {row.get('accuracy')} | {row.get('window_count')} ({row.get('real_count')}/{row.get('fake_count')}) |")
        for row in training_metrics:
            lines.append(f"| SUMMARY_SET_128_TRAIN | {row.get('seed')} | {row.get('source_macro_auroc_point_estimate')} | {row.get('source_count')} | {row.get('pooled_auroc')} | {row.get('pooled_ap')} | {row.get('precision')} | {row.get('recall')} | {row.get('f1')} | {row.get('accuracy')} | {row.get('window_count')} ({row.get('real_count')}/{row.get('fake_count')}) |")
    frontend_budget = json.loads((root / "state/frontend_budget.json").read_text(encoding="utf-8")) if (root / "state/frontend_budget.json").is_file() else {}
    recovery_budget_path = root / "state/frontend_recovery_budget.json"
    recovery_budget = json.loads(recovery_budget_path.read_text(encoding="utf-8")) if recovery_budget_path.is_file() else {}
    training_budget = json.loads((root / "state/training_budget.json").read_text(encoding="utf-8")) if (root / "state/training_budget.json").is_file() else {}
    recovery_training_path = root / "state/training_recovery_budget.json"
    recovery_training_budget = json.loads(recovery_training_path.read_text(encoding="utf-8")) if recovery_training_path.is_file() else {}
    uncertain = frontend_budget.get("uncertain_attempt") or {}
    historic_lower = uncertain.get("lower_bound_s_approx")
    historic_upper = uncertain.get("upper_bound_s")
    historic_interval = "未知耗时区间未记录"
    if historic_lower is not None and historic_upper is not None:
        historic_interval = f"未知实耗边界约 {float(historic_lower):.1f}–{float(historic_upper):.1f}s（不是实测值）"
    recovery_used = float(recovery_budget.get("cumulative_s", 0.0))
    recovery_accounted = float(recovery_budget.get("budget_accounted_upper_s", recovery_used))
    recovery_allowance = float(recovery_budget.get("budget_s", 0.0))
    recovery_remaining = max(0.0, recovery_allowance - recovery_accounted)
    training_used = float(training_budget.get("cumulative_s", 0.0))
    recovery_training_used = float(recovery_training_budget.get("cumulative_s", 0.0))
    recovery_training_allowance = float(recovery_training_budget.get("budget_s", 0.0))
    recovery_training_accounted = float(recovery_training_budget.get("budget_accounted_upper_s", recovery_training_used))
    lines += ["", "## 冻结与覆盖", "", f"- 新增 source 选择：stable SHA-256，seed={SELECTION_SEED}；不使用分数、支撑或人工观察。", f"- 训练池计划 {len(selection['base_sources']) + len(selection['added_sources'])} 个 source；实际仅纳入标签为 0/1 且 R SET_A 特征有效的窗口。无支撑窗口保留缺失原因，不填零；未以其他 source 替换。", "- 验证标准化、权重与模型训练完全排除；旧 64-source MEAN_BASELINE 分数来自 attention-pooling pilot，不使用 ATTENTION_POOL。", f"- 原7200秒前端额度历史：已保存实测累计 {float(frontend_budget.get('cumulative_s', 0.0)):.1f}s；{historic_interval}；旧记录的估算预留 {float(frontend_budget.get('estimated_reserved_s', 0.0)):.1f}s。历史未知区间与预留保持独立，不改写为实测，也不抵扣本次新增额度。", f"- 本次独立恢复前端额度：实际累计 {recovery_used:.1f}/{recovery_allowance:.0f}s；按本次额度账面上界剩余 {recovery_remaining:.1f}s；估算预留 {float(recovery_budget.get('estimated_reserved_s', 0.0)):.1f}s。额度记录：`{recovery_budget_path}`。", f"- 历史特征+训练账本保持：{training_used:.1f}/{TRAINING_BUDGET_S:.0f}s，状态={training_budget.get('process_state', '未记录')}；其已完成特征耗时不计入本次新增正式训练额度。", f"- 本次新增正式训练额度：{recovery_training_used:.1f}/{recovery_training_allowance:.0f}s；账面剩余 {max(0.0, recovery_training_allowance - recovery_training_accounted):.1f}s；额度记录：`{recovery_training_path}`。", "", "## 边界", "", "未访问旧 R7/V5；未修改正式 src 检测链；未改变点数、窗口、R 前端、分组、表示或聚合方法；未提交视频、权重、粒子数组或大缓存；不声称空间定位或未知生成器泛化。", ""]
    path = root / "report.md"; path.write_text("\n".join(lines), encoding="utf-8")
    expected_window_ids = {str(row["window_id"]) for row in subwindows}
    model_seeds = {int(row["seed"]) for row in model_records if row.get("status") == "TRAIN_COMPLETE"}
    model_complete = model_seeds == set(MODEL_SEEDS) and len(model_records) == len(MODEL_SEEDS)
    evaluation_complete = bool(summary) and len(summary.get("conditions", [])) >= 8 and int(summary.get("validation_population", {}).get("window_count", 0)) == 83
    frontend_complete = bool(expected_window_ids) and frontend_complete_ids == expected_window_ids and len(frontend_rows) == len(expected_window_ids)
    status = "COMPLETE" if media_complete and frontend_complete and feature_ready and feature_errors == 0 and model_complete and evaluation_complete else ("PARTIAL" if model_records or frontend_rows or support_rows else "PLANNED")
    existing_final_path = root / "final_status.json"
    if existing_final_path.is_file():
        try:
            existing_status = str(json.loads(existing_final_path.read_text(encoding="utf-8")).get("status", ""))
            if existing_status == "MEDIA_INCOMPLETE" and not media_complete:
                status = existing_status
        except (OSError, ValueError, TypeError):
            pass
    atomic_json(root / "final_status.json", {"status": status, "model_count": sum(row.get("status") == "TRAIN_COMPLETE" for row in model_records), "expected_model_count": 3, "model_seed_set_complete": sorted(model_seeds) == list(MODEL_SEEDS), "frontend_result_rows": len(frontend_rows), "frontend_expected_subwindows": len(subwindows), "frontend_complete_subwindows": len(frontend_complete_ids), "frontend_complete_parents": len(frontend_complete_parents), "frontend_status_counts": dict(counts), "feature_artifacts_complete": feature_ready, "feature_error_rows": feature_errors, "support_rows": len(support_rows), "r_valid_labeled_feature_windows": len(r_train_windows), "r_effective_training_sources": len(r_train_sources), "evaluation_present": bool(summary), "media_status_counts": dict(media_counts), "complete_source_count": int(complete_source_count), "planned_source_count": int(len(protocol["selection"]["base_sources"]) + len(protocol["selection"]["added_sources"]) + len(protocol["selection"]["validation_sources"])), "download_error": download_error, "smoke_frontend": smoke_frontend, "smoke_train": smoke_train, "report": str(path), "git_head": git_head(), "updated_unix": time.time()})
    final_status = json.loads((root / "final_status.json").read_text(encoding="utf-8"))
    final_status.update(
        media_download_complete=media_complete,
        media_manifest_count=len(media_rows),
        expected_media_count=expected_media_count,
        download_error_resolved=bool(media_complete and download_error),
    )
    atomic_json(root / "final_status.json", final_status)
    progress(root, "report", status, 1 if status == "COMPLETE" else 0, 1)
    return {"status": status, "report": str(path)}


def run_smoke(root: Path, device: str = "cuda") -> dict[str, Any]:
    """Run one 200-epoch endpoint on a tiny, deterministic training subset.

    The subset deliberately includes the first available added source so the
    smoke path exercises the newly materialized source rather than only the
    legacy cache.  This model is written only under ``smoke/`` and is never
    used by the formal 128-source comparison.
    """
    import torch
    from research_tools.v7.periodic_requery_probe import runner as periodic
    rows = _load_rows(root)
    protocol = json.loads((root / "protocol.json").read_text(encoding="utf-8"))
    available = {str(row["source_id"]) for row in rows}
    added = [str(source) for source in protocol["selection"]["added_sources"] if str(source) in available]
    base = [str(source) for source in protocol["selection"]["base_sources"] if str(source) in available]
    first_sources = set((added[:1] + base[:7]) or base[:8])
    train_rows = [row for row in rows if str(row["source_id"]) in first_sources]
    if not train_rows:
        raise RuntimeError("SMOKE_NO_ROWS")
    weights = periodic.source_class_weights(train_rows)
    standardizer = periodic.fit_standardizer("SET_A", [row["features"]["SET_A"] for row in train_rows], weights)
    values = [{**dict(row), "features": {"SET_A": np.asarray(row["features"]["SET_A"], dtype=np.float64)}} for row in train_rows]
    batch = periodic.make_batch("SET_A", values, standardizer); batch["window_weights"] = weights
    model, fit = periodic.train_one("SET_A", batch, seed=MODEL_SEEDS[0], device=device, epochs=EPOCHS)
    output = {"status": "PASS", "source_count": len(first_sources), "source_ids": sorted(first_sources), "added_source_ids": sorted(set(added) & first_sources), "window_count": len(train_rows), "epochs": fit["epochs"], "device": str(torch.device(device)), "formal_records_untouched": True}
    atomic_json(root / "smoke/summary.json", output)
    del model
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    return output


def run_frontend_smoke(root: Path, *, budget_s: float = FRONTEND_BUDGET_S, resume: bool = True) -> dict[str, Any]:
    """Materialize one complete added source (real and fake) through R.

    This is an endpoint check before the long formal run.  It uses the same
    frozen parent/window configuration and the same frontend budget clock;
    completed parents are atomically persisted and can be reused by the
    subsequent ``--resume`` run.
    """
    import torch
    from research_tools.v7.periodic_requery_probe import runner as periodic

    protocol = json.loads((root / "protocol.json").read_text(encoding="utf-8"))
    added = [str(source) for source in protocol["selection"]["added_sources"]]
    parents = _load_parents(root)
    subwindows = _load_subwindows(root)
    result_path = root / "frontend/results.json"
    results = {str(item["window_id"]): dict(item) for item in (json.loads(result_path.read_text(encoding="utf-8")) if result_path.is_file() else [])} if resume else {}
    by_parent: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in subwindows:
        by_parent[str(row["parent_id"])].append(dict(row))
    by_source_role = {(str(parent["source_id"]), str(parent["role"])): parent for parent in parents}
    selected_source = None
    selected_parents: list[dict[str, Any]] = []
    for source in added:
        candidate = [by_source_role.get((source, role)) for role in ("real", "fake")]
        if all(item is not None for item in candidate):
            selected_source = source
            selected_parents = [item for item in candidate if item is not None]
            break
    if selected_source is None:
        raise RuntimeError("SMOKE_NO_ADDED_SOURCE_IN_WINDOW_PLAN")
    pending = [parent for parent in selected_parents if not all(str(row["window_id"]) in results and str(results[str(row["window_id"])].get("status")) == "FRONTEND_COMPLETE" for row in by_parent[str(parent["parent_id"])] )]
    budget = Budget(root, "frontend", budget_s)
    if not pending:
        output = {"status": "PASS", "source_id": selected_source, "parents": 2, "reused": True, "budget_elapsed_s": budget.elapsed()}
        atomic_json(root / "smoke/frontend_summary.json", output)
        budget.save(None, process_state="COMPLETE", smoke_source=selected_source, completed_parents=0, total_parents=0)
        return output
    if budget.remaining() <= 0:
        budget.save("SMOKE_FRONTEND_BUDGET_EXHAUSTED", smoke_source=selected_source)
        output = {"status": "BUDGET_EXHAUSTED", "source_id": selected_source, "parents_completed": 0, "parents_total": len(pending), "budget_elapsed_s": budget.elapsed()}
        atomic_json(root / "smoke/frontend_summary.json", output)
        return output
    torch.set_num_threads(4)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA_UNAVAILABLE_FRONTEND_REQUIRES_GPU")
    from scripts.run_v7_explicit_geometry_frontend import DepthProRunner, OnlineBootsTapir
    tracker = OnlineBootsTapir(diagnostic.TAPNET_SOURCE, diagnostic.TAPNET_CHECKPOINT, process_size=256, grid_size=17)
    depth_runner = DepthProRunner(diagnostic.DEPTH_SOURCE, diagnostic.DEPTH_CHECKPOINT)
    completed_parents = 0
    errors: list[str] = []
    started_total = time.perf_counter()
    try:
        for parent in pending:
            if budget.remaining() <= 0:
                budget.save("SMOKE_FRONTEND_BUDGET_EXHAUSTED", smoke_source=selected_source, completed_parents=completed_parents)
                break
            parent_rows = by_parent[str(parent["parent_id"])]
            started = time.perf_counter()
            parent_started_unix = time.time()
            atomic_json(root / "state/current_parent.json", {"status": "RUNNING", "parent_id": str(parent["parent_id"]), "source_id": str(parent["source_id"]), "role": str(parent["role"]), "window_ids": [str(row["window_id"]) for row in parent_rows], "started_unix": parent_started_unix, "started_parent_count": completed_parents + 1, "total_parent_count": len(pending), "budget_elapsed_s": budget.elapsed(), "budget_remaining_s": budget.remaining(), "smoke": True})
            try:
                generated = periodic._frontend_parent(parent, parent_rows, tracker, depth_runner, root)
                elapsed = time.perf_counter() - started
                periodic._persist_parent_completion(root, result_path, results, generated, parent_id=str(parent["parent_id"]), parent_elapsed_s=elapsed, budget=budget, parent_completed=completed_parents + 1, total_windows=len(subwindows), parent_started_unix=parent_started_unix, total_parents=len(pending))
                completed_parents += 1
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
                errors.append(f"{parent['parent_id']}:{error}")
                atomic_json(root / "state/current_parent.json", {"status": "FAILED", "parent_id": str(parent["parent_id"]), "source_id": str(parent["source_id"]), "role": str(parent["role"]), "window_ids": [str(row["window_id"]) for row in parent_rows], "started_unix": parent_started_unix, "finished_unix": time.time(), "elapsed_s": time.perf_counter() - started, "error": error, "smoke": True})
                break
    finally:
        del tracker, depth_runner
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    status = "PASS" if completed_parents == len(pending) and not errors else ("BUDGET_EXHAUSTED" if budget.remaining() <= 0 else "FAILED")
    budget.save(None if status == "PASS" else status, process_state="COMPLETE" if status == "PASS" else status, smoke_source=selected_source, completed_parents=completed_parents, total_parents=len(pending), errors=errors)
    output = {"status": status, "source_id": selected_source, "parents_completed": completed_parents, "parents_total": len(pending), "elapsed_s": time.perf_counter() - started_total, "budget_elapsed_s": budget.elapsed(), "errors": errors}
    atomic_json(root / "smoke/frontend_summary.json", output)
    return output


def _ensure_plan(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    if not (root / "protocol.json").is_file():
        freeze_plan(root)
    _save_extension_protocol(root)


def run_all(root: Path, *, device: str, resume: bool, workers: int, frontend_budget_s: float, train_budget_s: float, frontend_budget_name: str = "frontend", train_budget_name: str = "training") -> dict[str, Any]:
    global STOP_REQUESTED
    _ensure_plan(root)
    atomic_json(root / "final_status.json", {"status": "RUNNING", "stage": "frontend", "git_head": git_head(), "pid": os.getpid(), "updated_unix": time.time()})
    media_ready = False
    media_path = root / "acquisition/media_manifest.json"
    if media_path.is_file():
        try:
            media_rows = json.loads(media_path.read_text(encoding="utf-8")).get("results", [])
            media_ready = bool(media_rows) and all(str(item.get("status")) in {"DOWNLOADED", "MATERIALIZED", "REUSED_EXISTING"} for item in media_rows)
        except (OSError, ValueError, TypeError):
            media_ready = False
    if not media_ready:
        try:
            download_result = download(root, workers=workers)
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            final = {"status": "MEDIA_INCOMPLETE", "download": {"status": "DOWNLOAD_BLOCKED", "error": error}, "git_head": git_head(), "updated_unix": time.time()}
            atomic_json(root / "final_status.json", final)
            try:
                current_rows = json.loads(media_path.read_text(encoding="utf-8")).get("results", []) if media_path.is_file() else []
                current_completed = sum(str(item.get("status")) in {"DOWNLOADED", "MATERIALIZED", "REUSED_EXISTING"} for item in current_rows)
                current_total = len(current_rows)
                current_counts = dict(Counter(str(item.get("status")) for item in current_rows))
            except (OSError, ValueError, TypeError):
                current_completed, current_total, current_counts = 0, 0, {}
            progress(root, "download", "MEDIA_INCOMPLETE", current_completed, current_total, error=error, status_counts=current_counts)
            return final
        if str(download_result.get("status")) != "COMPLETE":
            final = {"status": "MEDIA_INCOMPLETE", "download": download_result, "git_head": git_head(), "updated_unix": time.time()}
            atomic_json(root / "final_status.json", final)
            progress(root, "download", "MEDIA_INCOMPLETE", int(download_result.get("completed", 0)), int(download_result.get("count", 0)), status_counts=download_result.get("status_counts", {}))
            return final
    if not (root / "manifests/parents.json").is_file():
        build_and_prepare(root)
    _save_extension_protocol(root)
    if not (root / "checks/legacy_cache_identity.json").is_file():
        verify_legacy_cache_identity(root)
    frontend_result = frontend(root, frontend_budget_s, resume=resume, budget_name=frontend_budget_name)
    if frontend_result.get("status") not in {"COMPLETE", "FRONTEND_INCOMPLETE", "BUDGET_EXHAUSTED", "STOPPED_SAFE"}:
        report_result = report(root)
        final = {"status": "BLOCKED", "frontend": frontend_result, "report": report_result, "git_head": git_head(), "updated_unix": time.time()}
        atomic_json(root / "final_status.json", final)
        return final
    features_result: dict[str, Any] = {"status": "DEFERRED_FRONTEND_INCOMPLETE"}
    smoke_result = json.loads((root / "smoke/summary.json").read_text(encoding="utf-8")) if (root / "smoke/summary.json").is_file() else None
    training_result: dict[str, Any] = {"status": "NOT_RUN", "reason": "FRONTEND_NOT_COMPLETE"}
    if frontend_result.get("status") == "COMPLETE":
        if _feature_artifacts_complete(root):
            features_result = {"status": "COMPLETE", "reused": True, "summary": json.loads((root / "support/support_summary.json").read_text(encoding="utf-8"))}
        else:
            combined_budget = Budget(root, train_budget_name, train_budget_s)
            if combined_budget.remaining() <= 0:
                combined_budget.save("FEATURE_TRAIN_BUDGET_EXHAUSTED", stage="features")
                features_result = {"status": "BUDGET_EXHAUSTED", "elapsed_s": combined_budget.elapsed()}
            else:
                features_result = _features_with_budget(root, combined_budget)
        if features_result.get("status") == "COMPLETE":
            if smoke_result is None or smoke_result.get("status") != "PASS":
                smoke_result = run_smoke(root, device=device)
            training_result = train(root, train_budget_s, device, resume=resume, budget_name=train_budget_name)
        else:
            training_result = {"status": "NOT_RUN", "reason": "FEATURE_EXTRACTION_INCOMPLETE"}
    evaluation = evaluate(root, device) if training_result.get("status") == "COMPLETE" else {}
    report_result = report(root)
    status = "COMPLETE" if frontend_result.get("status") == "COMPLETE" and training_result.get("status") == "COMPLETE" and bool(evaluation) and report_result.get("status") == "COMPLETE" else ("PARTIAL" if frontend_result.get("completed", 0) or training_result.get("completed_models", 0) else "BLOCKED")
    final = {"status": status, "frontend": frontend_result, "features": features_result, "smoke": smoke_result, "training": training_result, "evaluation_present": bool(evaluation), "report": report_result, "git_head": git_head(), "updated_unix": time.time()}
    atomic_json(root / "final_status.json", final)
    return final


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("plan", "download", "windows", "verify-cache", "frontend", "smoke-frontend", "features", "smoke", "train", "evaluate", "report", "all", "record-wrapper-exit"))
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--frontend-budget-s", type=float, default=FRONTEND_BUDGET_S)
    parser.add_argument("--frontend-budget-name", choices=("frontend", "frontend_recovery"), default="frontend")
    parser.add_argument("--train-budget-s", type=float, default=TRAINING_BUDGET_S)
    parser.add_argument("--train-budget-name", choices=("training", "training_recovery"), default="training")
    parser.add_argument("--log-path")
    parser.add_argument("--wrapper-pid", type=int)
    parser.add_argument("--screen-session")
    parser.add_argument("--runner-pid", type=int)
    parser.add_argument("--exit-code", type=int)
    parser.add_argument("--started-unix", type=float)
    parser.add_argument("--command-line")
    args = parser.parse_args()
    root = args.output_root
    root.mkdir(parents=True, exist_ok=True)
    if args.stage == "record-wrapper-exit":
        required = (args.runner_pid, args.wrapper_pid, args.exit_code, args.started_unix, args.log_path, args.screen_session, args.command_line)
        if any(value is None for value in required):
            parser.error("record-wrapper-exit requires runner/wrapper PID, exit code, start time, log path, screen session, and command line")
        record_supervisor_exit(
            root,
            runner_pid=args.runner_pid,
            wrapper_pid=args.wrapper_pid,
            exit_code=args.exit_code,
            started_unix=args.started_unix,
            log_path=args.log_path,
            screen_session=args.screen_session,
            command=args.command_line,
        )
        return 0
    with run_lock(root):
        started_unix = time.time()
        command = [sys.executable, "-u", "-m", "research_tools.v7.source128_extension.runner", *sys.argv[1:]]
        atomic_json(root / "state/launch.json", {"stage": args.stage, "pid": os.getpid(), "wrapper_pid": args.wrapper_pid, "screen_session": args.screen_session, "command": command, "log_path": args.log_path, "git_head": git_head(), "started_unix": started_unix, "device": args.device})
        try:
            _ensure_plan(root)
            if args.stage == "plan":
                freeze_plan(root); result = {"status": "PLANNED"}
            elif args.stage == "download": result = download(root, args.workers)
            elif args.stage == "windows": result = build_and_prepare(root)
            elif args.stage == "verify-cache": result = verify_legacy_cache_identity(root)
            elif args.stage == "frontend": result = frontend(root, args.frontend_budget_s, args.resume, budget_name=args.frontend_budget_name)
            elif args.stage == "smoke-frontend": result = run_frontend_smoke(root, budget_s=args.frontend_budget_s, resume=args.resume)
            elif args.stage == "features": result = features(root)
            elif args.stage == "smoke": result = run_smoke(root, args.device)
            elif args.stage == "train": result = train(root, args.train_budget_s, args.device, args.resume, budget_name=args.train_budget_name)
            elif args.stage == "evaluate": result = evaluate(root, args.device)
            elif args.stage == "report": result = report(root)
            else: result = run_all(root, device=args.device, resume=args.resume, workers=args.workers, frontend_budget_s=args.frontend_budget_s, train_budget_s=args.train_budget_s, frontend_budget_name=args.frontend_budget_name, train_budget_name=args.train_budget_name)
            atomic_json(root / "state/last_exit.json", {"status": "OK", "stage": args.stage, "result": result, "updated_unix": time.time()})
            return 0
        except BaseException as exc:
            error = f"{type(exc).__name__}: {exc}"
            traceback_text = __import__("traceback").format_exc()
            atomic_json(root / "state/last_exit.json", {"status": "FAILED", "stage": args.stage, "error": error, "traceback": traceback_text, "updated_unix": time.time()})
            atomic_json(root / "final_status.json", {"status": "FAILED", "failed_stage": args.stage, "error": error, "git_head": git_head(), "updated_unix": time.time()})
            raise


if __name__ == "__main__":
    raise SystemExit(main())
