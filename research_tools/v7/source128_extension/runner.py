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
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter, defaultdict
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


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
        self.before = float(old.get("cumulative_s", 0.0))
        self.budget_s = float(budget_s)
        self.started_monotonic = time.monotonic()
        self.started_unix = time.time()

    def elapsed(self) -> float:
        return self.before + max(0.0, time.monotonic() - self.started_monotonic)

    def remaining(self) -> float:
        return max(0.0, self.budget_s - self.elapsed())

    def save(self, reason: str | None = None, **extra: Any) -> None:
        atomic_json(self.path, {
            "budget_s": self.budget_s,
            "elapsed_before_this_process_s": self.before,
            "process_start_unix": self.started_unix,
            "process_elapsed_s": max(0.0, time.monotonic() - self.started_monotonic),
            "cumulative_s": self.elapsed(),
            "stop_reason": reason,
            **extra,
        })


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


def frontend(root: Path, budget_s: float = FRONTEND_BUDGET_S, resume: bool = True) -> dict[str, Any]:
    curve = _set_curve_roots(root)
    return curve.run_frontend(root, budget_s=budget_s, resume=resume)


def features(root: Path) -> dict[str, Any]:
    curve = _set_curve_roots(root)
    return curve.run_features(root)


def _load_rows(root: Path) -> list[dict[str, Any]]:
    rows = json.loads((root / "support/window_support.json").read_text(encoding="utf-8"))
    return [dict(row) for row in rows if str(row.get("mode")) == "R" and int(row.get("valid_unit_count", 0)) > 0 and row.get("label") in (0, 1) and row.get("features", {}).get("SET_A") is not None]


def _load_validation_baseline() -> list[dict[str, Any]]:
    path = BASELINE_ROOT / "scores/validation_window_scores.csv"
    with path.open(newline="", encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _budget(root: Path, budget_s: float) -> Budget:
    return Budget(root, "training", budget_s)


def train(root: Path, budget_s: float = TRAINING_BUDGET_S, device: str = "cuda", resume: bool = True) -> dict[str, Any]:
    import torch
    from research_tools.v7.periodic_requery_probe import runner as periodic

    rows = _load_rows(root)
    protocol = json.loads((root / "protocol.json").read_text(encoding="utf-8"))
    train_sources = [str(x) for x in protocol["selection"]["base_sources"] + protocol["selection"]["added_sources"]]
    train_set = set(train_sources)
    train_rows = [row for row in rows if str(row["source_id"]) in train_set]
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
    model_path = root / "models/fold_models.json"
    records = json.loads(model_path.read_text(encoding="utf-8")).get("records", []) if resume and model_path.is_file() else []
    done = {(int(item["seed"]), int(item.get("source_count", 0))) for item in records if item.get("status") == "TRAIN_COMPLETE"}
    budget = _budget(root, budget_s)
    root.joinpath("models").mkdir(parents=True, exist_ok=True)
    total = len(MODEL_SEEDS)
    progress(root, "train", "RUNNING", len(done), total, source_count=TOTAL_SOURCE_COUNT, training_window_count=len(train_rows), training_real_count=counts[0], training_fake_count=counts[1], cumulative_elapsed_s=budget.elapsed())
    for seed in MODEL_SEEDS:
        if (int(seed), TOTAL_SOURCE_COUNT) in done:
            continue
        if budget.remaining() <= 0:
            budget.save("TRAINING_BUDGET_EXHAUSTED", completed_models=len(done), total_models=total)
            progress(root, "train", "TRAINING_BUDGET_EXHAUSTED", len(done), total, cumulative_elapsed_s=budget.elapsed())
            return {"status": "TRAINING_BUDGET_EXHAUSTED", "completed_models": len(done), "total_models": total}
        started = time.perf_counter()
        model, fit = periodic.train_one("SET_A", batch, seed=int(seed), device=device, epochs=EPOCHS)
        record = {"condition": "SUMMARY_SET", "base_condition": "SET_A", "ordering_seed": ORDERING_SEED, "source_count": TOTAL_SOURCE_COUNT, "seed": int(seed), "status": "TRAIN_COMPLETE", "training_sources": train_sources, "training_window_count": len(train_rows), "training_real_count": counts[0], "training_fake_count": counts[1], "parameter_count": periodic.parameter_count("SET_A"), "standardization": standardizer.as_dict(), "fit": fit, "elapsed_s": time.perf_counter() - started, "device": str(torch.device(device)), "state_dict": periodic.model_state(model)}
        records.append(record)
        atomic_json(model_path, {"condition": "SUMMARY_SET", "base_condition": "SET_A", "ordering_seed": ORDERING_SEED, "source_count": TOTAL_SOURCE_COUNT, "epochs": EPOCHS, "seeds": list(MODEL_SEEDS), "records": records})
        done.add((int(seed), TOTAL_SOURCE_COUNT))
        budget.save(None, last_seed=int(seed), completed_models=len(done), total_models=total)
        progress(root, "train", "RUNNING", len(done), total, last_seed=int(seed), cumulative_elapsed_s=budget.elapsed())
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    budget.save(None, completed_models=len(done), total_models=total)
    progress(root, "train", "COMPLETE", len(done), total, cumulative_elapsed_s=budget.elapsed())
    return {"status": "COMPLETE", "completed_models": len(done), "total_models": total, "elapsed_s": budget.elapsed(), "training_window_count": len(train_rows), "training_real_count": counts[0], "training_fake_count": counts[1]}


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

    rows = _load_rows(root)
    baseline_rows = _load_validation_baseline()
    baseline_by_id = {str(row["window_id"]): row for row in baseline_rows}
    validation_ids = set(baseline_by_id)
    rows = [row for row in rows if str(row["window_id"]) in validation_ids and str(row["source_id"]) in set(json.loads((root / "protocol.json").read_text(encoding="utf-8"))["selection"]["validation_sources"])]
    if len(rows) != len(validation_ids):
        raise RuntimeError(f"VALIDATION_FEATURE_SET_MISMATCH:{len(rows)}:{len(validation_ids)}")
    records = json.loads((root / "models/fold_models.json").read_text(encoding="utf-8"))["records"]
    score_rows = [{"window_id": str(row["window_id"]), "source_id": str(row["source_id"]), "role": str(row["role"]), "label": int(row["label"]), "valid_unit_count": int(row.get("valid_unit_count", 0)), "baseline_seed_20260909": float(baseline_by_id[str(row["window_id"])]["MEAN_BASELINE_seed_20260909"]), "baseline_seed_20260910": float(baseline_by_id[str(row["window_id"])]["MEAN_BASELINE_seed_20260910"]), "baseline_seed_20260911": float(baseline_by_id[str(row["window_id"])]["MEAN_BASELINE_seed_20260911"]), "baseline_mean_logit": float(baseline_by_id[str(row["window_id"])]["MEAN_BASELINE_MEAN_LOGIT"])} for row in rows]
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
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    seed_names = [f"source128_seed_{seed}" for seed in MODEL_SEEDS]
    for item in score_rows:
        if not all(name in item for name in seed_names):
            raise RuntimeError(f"MISSING_128_SEED_SCORE:{item['window_id']}")
        item["source128_mean_logit"] = float(np.mean([item[name] for name in seed_names]))
        item["baseline_mean_logit"] = float(np.mean([item[f"baseline_seed_{seed}"] for seed in MODEL_SEEDS]))
    root.joinpath("scores").mkdir(parents=True, exist_ok=True)
    write_csv(root / "scores/validation_window_scores.csv", score_rows)
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
    summary = {"validation_population": {"source_count": len(new_source), "window_count": len(score_rows), "real_count": int(sum(x == 0 for x in labels)), "fake_count": int(sum(x == 1 for x in labels)), "window_ids": sorted(validation_ids)}, "conditions": metrics_rows, "primary_comparison": {"name": "SUMMARY_SET_128_MINUS_MEAN_BASELINE_64", **paired}, "per_source": per_source, "baseline_source": baseline_source, "source128_source": new_source, "device": str(torch.device(device)), "model_count": len(records), "baseline_artifact": str(BASELINE_ROOT), "score_identity": "baseline MEAN_BASELINE seed scores reused; 128 scores from newly trained models"}
    atomic_json(root / "evaluation/summary.json", summary)
    return summary


def report(root: Path) -> dict[str, Any]:
    protocol = json.loads((root / "protocol.json").read_text(encoding="utf-8"))
    summary = json.loads((root / "evaluation/summary.json").read_text(encoding="utf-8")) if (root / "evaluation/summary.json").is_file() else {}
    frontend_rows = json.loads((root / "frontend/results.json").read_text(encoding="utf-8")) if (root / "frontend/results.json").is_file() else []
    support_rows = json.loads((root / "support/window_support.json").read_text(encoding="utf-8")) if (root / "support/window_support.json").is_file() else []
    model_records = json.loads((root / "models/fold_models.json").read_text(encoding="utf-8")).get("records", []) if (root / "models/fold_models.json").is_file() else []
    counts = Counter(str(row.get("status")) for row in frontend_rows)
    support_valid = Counter(str(row.get("mode")) for row in support_rows if int(row.get("valid_unit_count", 0)) > 0)
    lines = ["# V7 64→128 source 训练扩容对照", "", "本报告为当前数据集内的开发性 source 扩容 pilot；固定 R、SET_A/SUMMARY_SET、平均局部聚合和原验证总体，不是 sealed test 或最终泛化结论。", "", "## 结论先行", ""]
    lines += [f"- 冻结训练池：原 64 + 新增 {len(protocol['selection']['added_sources'])} = {len(protocol['selection']['base_sources']) + len(protocol['selection']['added_sources'])} source；验证：{len(protocol['selection']['validation_sources'])} source。", f"- 前端窗口状态：{dict(counts)}；R 有效支撑行：{support_valid.get('R', 0)}；模型完成：{len(model_records)}/3。"]
    if summary:
        primary = summary.get("primary_comparison", {})
        lines += [f"- 主比较 128−64 source-macro AUROC：{primary.get('mean')}，95% CI={primary.get('ci95')}。", "- 该区间若跨 0，不解释为稳定扩容收益。"]
    lines += ["", "## 主指标", "", "| condition | seed | windows | real | fake | source-macro AUROC | CI | pooled AUROC | AP | F1 | ACC |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in summary.get("conditions", []):
        macro = row.get("source_macro_auroc", {})
        cls = row.get("classification", {})
        lines.append(f"| {row.get('condition')} | {row.get('seed')} | {row.get('window_count')} | {row.get('real_count')} | {row.get('fake_count')} | {macro.get('mean')} | {macro.get('ci95')} | {row.get('pooled_auroc')} | {row.get('pooled_ap')} | {cls.get('f1')} | {cls.get('accuracy')} |")
    lines += ["", "## 冻结与覆盖", "", f"- 新增 source 选择：stable SHA-256，seed={SELECTION_SEED}；不使用分数、支撑或人工观察。", "- 训练仅纳入标签为 0/1 且 R SET_A 特征有效的窗口；验证标准化、权重和模型训练完全排除。", "- 旧 64-source MEAN_BASELINE 分数来自 attention-pooling pilot；不使用 ATTENTION_POOL。", f"- 前端预算={FRONTEND_BUDGET_S}s，训练预算={TRAINING_BUDGET_S}s；实际预算文件见 `state/`。", "", "## 边界", "", "未访问旧 R7/V5；未修改正式 src 检测链；未改变点数、窗口、R 前端、分组、表示或聚合方法；未提交视频、权重、粒子数组或大缓存；不声称空间定位或未知生成器泛化。", ""]
    path = root / "report.md"; path.write_text("\n".join(lines), encoding="utf-8")
    status = "COMPLETE" if len(model_records) == 3 and summary else ("PARTIAL" if model_records or frontend_rows else "PLANNED")
    atomic_json(root / "final_status.json", {"status": status, "model_count": len(model_records), "expected_model_count": 3, "frontend_result_rows": len(frontend_rows), "frontend_status_counts": dict(counts), "support_rows": len(support_rows), "evaluation_present": bool(summary), "report": str(path), "git_head": git_head(), "updated_unix": time.time()})
    progress(root, "report", status, 1 if status == "COMPLETE" else 0, 1)
    return {"status": status, "report": str(path)}


def run_smoke(root: Path, device: str = "cuda") -> dict[str, Any]:
    """Run one 200-epoch endpoint using the first 8 frozen training sources."""
    import torch
    from research_tools.v7.periodic_requery_probe import runner as periodic
    rows = _load_rows(root)
    protocol = json.loads((root / "protocol.json").read_text(encoding="utf-8"))
    first_sources = set(protocol["selection"]["base_sources"][:8])
    train_rows = [row for row in rows if str(row["source_id"]) in first_sources]
    if not train_rows:
        raise RuntimeError("SMOKE_NO_ROWS")
    weights = periodic.source_class_weights(train_rows)
    standardizer = periodic.fit_standardizer("SET_A", [row["features"]["SET_A"] for row in train_rows], weights)
    values = [{**dict(row), "features": {"SET_A": np.asarray(row["features"]["SET_A"], dtype=np.float64)}} for row in train_rows]
    batch = periodic.make_batch("SET_A", values, standardizer); batch["window_weights"] = weights
    model, fit = periodic.train_one("SET_A", batch, seed=MODEL_SEEDS[0], device=device, epochs=EPOCHS)
    output = {"status": "PASS", "source_count": len(first_sources), "window_count": len(train_rows), "epochs": fit["epochs"], "device": str(torch.device(device)), "formal_records_untouched": True}
    atomic_json(root / "smoke/summary.json", output)
    del model
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    return output


def _ensure_plan(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    if not (root / "protocol.json").is_file():
        freeze_plan(root)
    _save_extension_protocol(root)


def run_all(root: Path, *, device: str, resume: bool, workers: int, frontend_budget_s: float, train_budget_s: float) -> dict[str, Any]:
    global STOP_REQUESTED
    _ensure_plan(root)
    media_ready = False
    media_path = root / "acquisition/media_manifest.json"
    if media_path.is_file():
        try:
            media_rows = json.loads(media_path.read_text(encoding="utf-8")).get("results", [])
            media_ready = bool(media_rows) and all(str(item.get("status")) in {"DOWNLOADED", "MATERIALIZED", "REUSED_EXISTING"} for item in media_rows)
        except (OSError, ValueError, TypeError):
            media_ready = False
    if not media_ready:
        download(root, workers=workers)
    if not (root / "manifests/parents.json").is_file():
        build_and_prepare(root)
    _save_extension_protocol(root)
    if not (root / "checks/legacy_cache_identity.json").is_file():
        verify_legacy_cache_identity(root)
    frontend_result = frontend(root, frontend_budget_s, resume=resume)
    if frontend_result.get("status") not in {"COMPLETE", "FRONTEND_INCOMPLETE", "BUDGET_EXHAUSTED", "STOPPED_SAFE"}:
        return {"status": frontend_result.get("status", "FRONTEND_FAILED"), "frontend": frontend_result}
    features_result = features(root)
    smoke_result = None
    training_result: dict[str, Any] = {"status": "NOT_RUN", "reason": "FRONTEND_NOT_COMPLETE"}
    if frontend_result.get("status") == "COMPLETE":
        smoke_result = run_smoke(root, device=device) if not (root / "smoke/summary.json").is_file() else json.loads((root / "smoke/summary.json").read_text(encoding="utf-8"))
        training_result = train(root, train_budget_s, device, resume=resume)
    evaluation = evaluate(root, device) if training_result.get("status") == "COMPLETE" else {}
    report_result = report(root)
    status = "COMPLETE" if frontend_result.get("status") == "COMPLETE" and training_result.get("status") == "COMPLETE" and bool(evaluation) and report_result.get("status") == "COMPLETE" else ("PARTIAL" if frontend_result.get("completed", 0) or training_result.get("completed_models", 0) else "BLOCKED")
    final = {"status": status, "frontend": frontend_result, "features": features_result, "smoke": smoke_result, "training": training_result, "evaluation_present": bool(evaluation), "report": report_result, "git_head": git_head(), "updated_unix": time.time()}
    atomic_json(root / "final_status.json", final)
    return final


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("plan", "download", "windows", "verify-cache", "frontend", "features", "smoke", "train", "evaluate", "report", "all"))
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--frontend-budget-s", type=float, default=FRONTEND_BUDGET_S)
    parser.add_argument("--train-budget-s", type=float, default=TRAINING_BUDGET_S)
    args = parser.parse_args()
    root = args.output_root
    root.mkdir(parents=True, exist_ok=True)
    with run_lock(root):
        atomic_json(root / "state/launch.json", {"stage": args.stage, "pid": os.getpid(), "git_head": git_head(), "started_unix": time.time(), "device": args.device})
        try:
            _ensure_plan(root)
            if args.stage == "plan":
                freeze_plan(root); result = {"status": "PLANNED"}
            elif args.stage == "download": result = download(root, args.workers)
            elif args.stage == "windows": result = build_and_prepare(root)
            elif args.stage == "verify-cache": result = verify_legacy_cache_identity(root)
            elif args.stage == "frontend": result = frontend(root, args.frontend_budget_s, args.resume)
            elif args.stage == "features": result = features(root)
            elif args.stage == "smoke": result = run_smoke(root, args.device)
            elif args.stage == "train": result = train(root, args.train_budget_s, args.device, args.resume)
            elif args.stage == "evaluate": result = evaluate(root, args.device)
            elif args.stage == "report": result = report(root)
            else: result = run_all(root, device=args.device, resume=args.resume, workers=args.workers, frontend_budget_s=args.frontend_budget_s, train_budget_s=args.train_budget_s)
            atomic_json(root / "state/last_exit.json", {"status": "OK", "stage": args.stage, "result": result, "updated_unix": time.time()})
            return 0
        except BaseException as exc:
            atomic_json(root / "state/last_exit.json", {"status": "FAILED", "stage": args.stage, "error": f"{type(exc).__name__}: {exc}", "updated_unix": time.time()})
            raise


if __name__ == "__main__":
    raise SystemExit(main())
