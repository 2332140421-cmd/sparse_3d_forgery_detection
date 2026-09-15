"""Run the V7 cross-source training-scale learning-curve pilot.

This tool is intentionally a small experiment wrapper.  It freezes source and
window identities from the ActivityForensics/Charades manifests, reuses the
audited periodic re-query frontend and SUMMARY_SET model, and keeps all large
media/model artifacts on the data disk.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[3]
DATA_ROOT = Path("/root/autodl-tmp/data/sparse_3d_forgery_detection")
DATASET_ROOT = DATA_ROOT / "datasets/v7_core_candidates/activityforensics_charades_v1"
OUTPUT_ROOT = DATA_ROOT / "derived/v7_activityforensics_source_learning_curve_v1"
STAGING_ROOT = OUTPUT_ROOT / "staging_base_manifest"
HISTORICAL_SOURCES = (
    "01KML", "04LAX", "0AGCS", "0BX9N", "0CG15", "0CGMQ", "0DVVD",
    "0ET8W", "0FM93", "0FO58", "0G2SC", "0HV07", "0KTWY", "0LNLR",
    "0PU21", "0QA8P",
)
POOL_SOURCE_COUNT = 64
VALIDATION_SOURCE_COUNT = 16
SOURCE_POOL_SEED = 20260908
ORDER_SEEDS = (20260909, 20260910)
TRAINING_SIZES = (8, 16, 32, 64)
MODEL_SEEDS = (20260909, 20260910, 20260911)
EPOCHS = 200
TRAINING_BUDGET_S = 1800.0
FRONTEND_BUDGET_S = 7200.0
BOOTSTRAP_SEED = 20260909
BOOTSTRAP_REPLICATES = 10_000
OFFSETS_S = (0.0, 0.5, 1.0)
STOP_REQUESTED = False


def _jsonable(value: Any, path: str = "$") -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return [_jsonable(item, f"{path}[{idx}]") for idx, item in enumerate(value)]
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return _jsonable(value.item(), path)
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"non-finite JSON value at {path}: {value!r}")
        return value
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item, f"{path}.{key}") for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item, f"{path}[{idx}]") for idx, item in enumerate(value)]
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


def progress(root: Path, stage: str, status: str, completed: int, total: int, **extra: Any) -> None:
    atomic_json(root / "progress.json", {"stage": stage, "status": status, "completed": int(completed), "total": int(total), "updated_unix": time.time(), **extra})


def _cells(variant: Mapping[str, Any]) -> set[tuple[str, str]]:
    operations = variant.get("manipulation_operation")
    values = operations if isinstance(operations, list) else [operations]
    return {(str(variant["generator"]), str(value)) for value in values}


def _candidate_rows() -> list[dict[str, Any]]:
    path = DATASET_ROOT / "acquisition/targeted_review_v1/candidate_sources.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows: list[dict[str, Any]] = []
    historical = set(HISTORICAL_SOURCES)
    for source in payload["sources"]:
        source_id = str(source["charades_source_id"])
        variants = [dict(item) for item in source.get("fake_variants", []) if str(item.get("official_split")) == "train" and str(item.get("lineage_status")) == "EXACT"]
        if not variants or source_id in historical:
            continue
        rows.append({"charades_source_id": source_id, "fake_variants": variants, "available_cells": sorted([list(cell) for variant in variants for cell in _cells(variant)])})
    return rows


def _balanced_selection(rows: Sequence[Mapping[str, Any]], count: int) -> list[dict[str, Any]]:
    """Same label-blind round-robin cell selection used by the dataset census."""

    by_source = {str(row["charades_source_id"]): row for row in rows}
    cells = sorted({tuple(cell) for row in rows for cell in row["available_cells"]})
    remaining = set(by_source)
    selected: list[dict[str, Any]] = []
    index = 0
    while remaining and len(selected) < count:
        cell = cells[index % len(cells)]
        index += 1
        candidates = sorted(source for source in remaining if list(cell) in by_source[source]["available_cells"])
        if not candidates:
            continue
        source_id = candidates[0]
        row = by_source[source_id]
        variants = sorted(row["fake_variants"], key=lambda item: (str(item["generator"]), str(item["manipulation_operation"]), str(item["activityforensics_file"])))
        variant = next(item for item in variants if list(cell) in [list(value) for value in _cells(item)])
        selected.append({"charades_source_id": source_id, "selected_cell": list(cell), "fake_variant": variant})
        remaining.remove(source_id)
    if len(selected) < count:
        raise ValueError(f"candidate source pool too small: requested={count} selected={len(selected)}")
    return selected


def _source_order(source_ids: Sequence[str], seed: int) -> list[str]:
    return sorted((str(x) for x in source_ids), key=lambda source: hashlib.sha256(f"v7-source-learning-order|{seed}|{source}".encode("utf-8")).hexdigest())


def _source_split_rows() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    selected = _balanced_selection(_candidate_rows(), POOL_SOURCE_COUNT + VALIDATION_SOURCE_COUNT)
    validation = selected[:VALIDATION_SOURCE_COUNT]
    pool = selected[VALIDATION_SOURCE_COUNT:]
    rows: list[dict[str, Any]] = []
    for index, item in enumerate(validation):
        variant = item["fake_variant"]
        rows.append({"source_id": item["charades_source_id"], "split_role": "VALIDATION", "pool_index": "", "validation_index": index, "selected_cell": json.dumps(item["selected_cell"]), "generator": variant["generator"], "manipulation_operation": json.dumps(variant["manipulation_operation"]), "activityforensics_file": variant["activityforensics_file"], "official_split": variant["official_split"], "lineage_status": variant["lineage_status"]})
    for index, item in enumerate(pool):
        variant = item["fake_variant"]
        rows.append({"source_id": item["charades_source_id"], "split_role": "TRAIN_POOL", "pool_index": index, "validation_index": "", "selected_cell": json.dumps(item["selected_cell"]), "generator": variant["generator"], "manipulation_operation": json.dumps(variant["manipulation_operation"]), "activityforensics_file": variant["activityforensics_file"], "official_split": variant["official_split"], "lineage_status": variant["lineage_status"]})
    return rows, selected


def freeze_plan(root: Path, *, overwrite: bool = False) -> dict[str, Any]:
    """Freeze source splits and protocol before any model result exists."""

    if (root / "protocol.json").is_file() and not overwrite:
        return json.loads((root / "protocol.json").read_text(encoding="utf-8"))
    rows, selected = _source_split_rows()
    validation_ids = [str(item["charades_source_id"]) for item in selected[:VALIDATION_SOURCE_COUNT]]
    pool_ids = [str(item["charades_source_id"]) for item in selected[VALIDATION_SOURCE_COUNT:]]
    ordering = {str(seed): _source_order(pool_ids, seed) for seed in ORDER_SEEDS}
    subsets = {str(seed): {str(size): ordering[str(seed)][:size] for size in TRAINING_SIZES} for seed in ORDER_SEEDS}
    root.joinpath("manifests").mkdir(parents=True, exist_ok=True)
    write_csv(root / "manifests/source_split_manifest.csv", rows)
    atomic_json(root / "manifests/training_subsets.json", {"selection_seed": SOURCE_POOL_SEED, "ordering_seeds": list(ORDER_SEEDS), "validation_sources": validation_ids, "max_training_pool": pool_ids, "subsets": subsets, "nested": True, "label_independent": True})
    protocol = {
        "protocol_id": "v7-source-learning-curve-v1",
        "git_head": git_head(),
        "dataset": {"name": "ActivityForensics/ActivityForensics + Charades", "revision": "a34d4b7b04b0f3f3e26ba900adc367218667c581", "official_split": "train_only", "candidate_manifest": str(DATASET_ROOT / "acquisition/targeted_review_v1/candidate_sources.json"), "candidate_manifest_sha256": sha256(DATASET_ROOT / "acquisition/targeted_review_v1/candidate_sources.json")},
        "historical_development_sources_excluded": list(HISTORICAL_SOURCES),
        "source_selection": {"algorithm": "exact-lineage train candidates, historical IDs excluded, generator/operation cell round-robin", "selection_seed": SOURCE_POOL_SEED, "ordering_seeds": list(ORDER_SEEDS), "validation_count": VALIDATION_SOURCE_COUNT, "training_pool_count": POOL_SOURCE_COUNT, "same_max_pool_for_orderings": True, "model_result_independent": True},
        "windows": {"rule": "periodic re-query MANIP50 parent", "offsets_s": list(OFFSETS_S), "window_length_s": 1.0, "parent_length_s": 2.0, "query_count": 289, "label_rule": "real=0; fake=1 only if whole 1s window is contained in merged official manipulation interval; boundary/outside excluded", "labels_applied_after_sampling": True, "support": "same audited H grouping, history-local scale and five-time common support"},
        "model": {"condition": "SUMMARY_SET / SET_A", "implementation": "research_tools.v7.multi_order_sequence_probe.model.SetAModel", "epochs": EPOCHS, "optimizer": "Adam", "learning_rate": 1e-3, "weight_decay": 1e-4, "seeds": list(MODEL_SEEDS), "weighted_bce": "equal source and class total weight", "standardization": "training subset only"},
        "evaluation": {"validation_sources": validation_ids, "main_population": "label-qualified R-valid windows from validation sources with both real and fake", "threshold": "logit >= 0", "bootstrap": {"unit": "validation source", "replicates": BOOTSTRAP_REPLICATES, "seed": BOOTSTRAP_SEED}, "ap_tiou": "NOT_EVALUATED"},
        "budgets": {"frontend_s": FRONTEND_BUDGET_S, "training_s": TRAINING_BUDGET_S, "download_separate": True},
        "boundaries": ["development validation, not sealed test", "no old R7/V5", "no formal src changes", "no new features or architecture", "no post-hoc source selection", "large artifacts remain on data disk"],
    }
    atomic_json(root / "protocol.json", protocol)
    atomic_json(root / "manifests/freeze_identity.json", {"source_split_sha256": sha256(root / "manifests/source_split_manifest.csv"), "training_subsets_sha256": sha256(root / "manifests/training_subsets.json"), "protocol_sha256": sha256(root / "protocol.json"), "git_head": git_head()})
    return protocol


def _load_split(root: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    with (root / "manifests/source_split_manifest.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    subsets = json.loads((root / "manifests/training_subsets.json").read_text(encoding="utf-8"))
    return rows, subsets


def download_media(root: Path, workers: int = 4) -> dict[str, Any]:
    """Download only the frozen 80-source media, with per-file checksums."""

    from concurrent.futures import ThreadPoolExecutor, as_completed
    from huggingface_hub import HfApi, hf_hub_download
    from research_tools.v7.data.remote_zip64 import RangeClient, exact_basename, extract_member, read_central_directory

    split_rows, _ = _load_split(root)
    index_path = DATASET_ROOT / "acquisition/targeted_review_v1/charades_range_index.json"
    archive = json.loads(index_path.read_text(encoding="utf-8"))
    entries = {str(item["name"]).split("/")[-1]: item for item in archive["entries"]}
    api = HfApi()
    info = api.dataset_info("ActivityForensics/ActivityForensics", revision="a34d4b7b04b0f3f3e26ba900adc367218667c581", files_metadata=True)
    metadata = {str(item.rfilename): item for item in info.siblings}
    rows: list[dict[str, Any]] = []
    for split in split_rows:
        sid = str(split["source_id"])
        fake_rel = str(split["activityforensics_file"])
        rows.append({"source_id": sid, "role": "real", "path": str(DATASET_ROOT / "source/charades/videos" / f"{sid}.mp4"), "remote_member": f"Charades_v1/{sid}.mp4", "expected_bytes": entries.get(f"{sid}.mp4", {}).get("uncompressed_size"), "expected_sha256": None, "status": "PLANNED"})
        item = metadata.get(fake_rel)
        rows.append({"source_id": sid, "role": "fake", "path": str(DATASET_ROOT / "source/activityforensics/raw" / fake_rel), "remote_member": fake_rel, "expected_bytes": int(item.size) if item else None, "expected_sha256": item.lfs.sha256 if item and item.lfs else None, "status": "PLANNED"})
    rows.sort(key=lambda row: (str(row["source_id"]), str(row["role"])))
    progress(root, "download", "RUNNING", 0, len(rows))

    def one(row: Mapping[str, Any]) -> dict[str, Any]:
        result = dict(row)
        path = Path(str(row["path"]))
        try:
            if row["role"] == "real":
                entry = entries.get(f"{row['source_id']}.mp4")
                if entry is None:
                    result.update(status="SOURCE_MEMBER_MISSING")
                elif path.is_file() and path.stat().st_size == int(entry["uncompressed_size"]):
                    result.update(status="REUSED_EXISTING", local_bytes=path.stat().st_size, local_sha256=sha256(path))
                else:
                    client = RangeClient("https://ai2-public-datasets.s3-us-west-2.amazonaws.com/charades/Charades_v1.zip", timeout=180.0)
                    central, _ = read_central_directory(client)
                    member = exact_basename(central, f"{row['source_id']}.mp4")
                    extract_member(client, member, path.with_suffix(path.suffix + ".part"))
                    result.update(status="MATERIALIZED", local_bytes=path.stat().st_size, local_sha256=sha256(path))
            else:
                expected = metadata.get(str(row["remote_member"]))
                if expected is None:
                    result.update(status="REMOTE_MEMBER_MISSING")
                elif path.is_file() and path.stat().st_size == int(expected.size) and sha256(path) == str(expected.lfs.sha256):
                    result.update(status="REUSED_EXISTING", local_bytes=path.stat().st_size, local_sha256=sha256(path))
                else:
                    downloaded = Path(hf_hub_download(repo_id="ActivityForensics/ActivityForensics", filename=str(row["remote_member"]), revision="a34d4b7b04b0f3f3e26ba900adc367218667c581", repo_type="dataset", local_dir=str(DATASET_ROOT / "source/activityforensics/raw")))
                    result.update(status="DOWNLOADED" if downloaded.stat().st_size == int(expected.size) and sha256(downloaded) == str(expected.lfs.sha256) else "CHECKSUM_FAILURE", local_bytes=downloaded.stat().st_size, local_sha256=sha256(downloaded))
        except Exception as exc:  # preserve exact media failure in the manifest
            result.update(status="DOWNLOAD_FAILURE", error=f"{type(exc).__name__}: {exc}")
        return result

    results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max(1, int(workers))) as pool:
        futures = [pool.submit(one, row) for row in rows]
        for index, future in enumerate(as_completed(futures), 1):
            result = future.result()
            results.append(result)
            progress(root, "download", "RUNNING", index, len(rows), last_source=result.get("source_id"), last_role=result.get("role"))
    results.sort(key=lambda row: (str(row["source_id"]), str(row["role"])))
    atomic_json(root / "acquisition/media_manifest.json", {"revision": "a34d4b7b04b0f3f3e26ba900adc367218667c581", "results": results})
    counts = defaultdict(int)
    for row in results:
        counts[str(row["status"])] += 1
    status = "COMPLETE" if all(row["status"] in {"DOWNLOADED", "MATERIALIZED", "REUSED_EXISTING"} for row in results) else "PARTIAL"
    progress(root, "download", status, len(results), len(rows), status_counts=dict(counts))
    return {"status": status, "count": len(results), "status_counts": dict(counts)}


def _timeline(path: Path) -> dict[str, Any]:
    """Read presentation-order PTS without decoding RGB frames.

    The learning-curve planner only needs frame identity and timestamps.  The
    audited periodic frontend performs the actual frame decode later.  Reading
    packet PTS here avoids decoding every source twice while retaining the
    exact time-base conversion used by the existing timeline reader.  The
    candidate MP4s are one-video-frame-per-PTS-packet; duplicate/missing PTS
    values are rejected rather than silently falling back to FPS.
    """

    import av
    from research_tools.v7.paired_signal.protocol import assert_original_video_path, file_sha256

    source = assert_original_video_path(path)
    timestamps: list[float] = []
    with av.open(str(source)) as container:
        stream = next((item for item in container.streams if item.type == "video"), None)
        if stream is None:
            raise ValueError(f"no video stream: {source}")
        if stream.time_base is None:
            raise ValueError(f"missing video time base: {source}")
        for packet in container.demux(stream):
            if packet.pts is None:
                continue
            timestamp = float(packet.pts * stream.time_base)
            if not math.isfinite(timestamp):
                raise ValueError(f"non-finite packet PTS: {source}")
            timestamps.append(timestamp)
        if len(timestamps) < 2:
            raise ValueError(f"not enough timestamped video packets: {source}")
        if len(set(timestamps)) != len(timestamps):
            raise ValueError(f"duplicate packet PTS: {source}")
        timestamps.sort()
        periods = np.diff(np.asarray(timestamps, dtype=np.float64))
        if np.any(periods <= 0):
            raise ValueError(f"non-increasing packet PTS: {source}")
        values = np.asarray(timestamps, dtype=np.float64)
        return {
            "path": str(source),
            "sha256": file_sha256(source),
            "frame_indices": list(range(int(values.size))),
            "timestamps_s": values.tolist(),
            "frame_count": int(values.size),
            "start_s": float(values[0]),
            "end_s": float(values[-1]),
            "duration_s": float(values[-1] - values[0]),
            "median_frame_period_s": float(np.median(periods)),
            "width": int(stream.width),
            "height": int(stream.height),
            "codec_name": str(stream.codec_context.name),
            "timestamp_source": "video packet PTS sorted by presentation timestamp",
        }


def _merge_segments(segments: Sequence[Mapping[str, Any]]) -> list[dict[str, float]]:
    merged: list[list[float]] = []
    for item in sorted((float(x["start_s"]), float(x["end_s"])) for x in segments):
        if item[1] <= item[0]:
            continue
        if not merged or item[0] > merged[-1][1]:
            merged.append([item[0], item[1]])
        else:
            merged[-1][1] = max(merged[-1][1], item[1])
    return [{"start_s": left, "end_s": right} for left, right in merged]


def _primary_segment(segments: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return dict(sorted((dict(item) for item in segments), key=lambda item: (-(float(item["end_s"]) - float(item["start_s"])), float(item["start_s"]))) [0])


def _frame_interval(timeline: Mapping[str, Any], start: float, end: float) -> tuple[list[int], list[float]]:
    times = np.asarray(timeline["timestamps_s"], dtype=np.float64)
    indices = np.flatnonzero((times >= float(start) - 1e-9) & (times <= float(end) + 1e-9)).astype(int)
    return indices.tolist(), times[indices].tolist()


def _label(role: str, start: float, end: float, segments: Sequence[Mapping[str, Any]]) -> tuple[int | None, str]:
    if role == "real":
        return 0, "REAL_NEGATIVE"
    merged = _merge_segments(segments)
    if any(start >= item["start_s"] - 1e-9 and end <= item["end_s"] + 1e-9 for item in merged):
        return 1, "FAKE_MANIPULATION"
    if any(min(end, item["end_s"]) > max(start, item["start_s"]) for item in merged):
        return None, "BOUNDARY_MIXED"
    return None, "OUTSIDE_ANNOTATED_MANIPULATION"


def build_windows(root: Path) -> dict[str, Any]:
    split_rows, _ = _load_split(root)
    media = json.loads((root / "acquisition/media_manifest.json").read_text(encoding="utf-8"))["results"]
    media_by_key = {(str(row["source_id"]), str(row["role"])): row for row in media}
    selected_by_source = {str(row["source_id"]): row for row in split_rows}
    pairs: list[dict[str, Any]] = []
    windows: list[dict[str, Any]] = []
    exclusions: list[dict[str, Any]] = []
    for index, source_id in enumerate(sorted(selected_by_source), 1):
        split = selected_by_source[source_id]
        variant = json.loads(split["manipulation_operation"])  # only used for a stable manifest field
        real = media_by_key[(source_id, "real")]
        fake = media_by_key[(source_id, "fake")]
        if real.get("status") not in {"DOWNLOADED", "MATERIALIZED", "REUSED_EXISTING"} or fake.get("status") not in {"DOWNLOADED", "MATERIALIZED", "REUSED_EXISTING"}:
            exclusions.append({"source_id": source_id, "reason": "MEDIA_NOT_AVAILABLE"})
            continue
        try:
            real_timeline = _timeline(Path(str(real["path"])))
            fake_timeline = _timeline(Path(str(fake["path"])))
            source_item = next(item for item in _candidate_rows() if str(item["charades_source_id"]) == source_id)
            fake_variant = next(item for item in source_item["fake_variants"] if str(item["activityforensics_file"]) == str(split["activityforensics_file"]))
            segments = _merge_segments(fake_variant["manipulation_segments"])
            primary = _primary_segment(segments)
            if primary["end_s"] - primary["start_s"] < 1.0:
                raise ValueError("MANIPULATION_SEGMENT_TOO_SHORT")
            # The audited paired protocol uses the middle anchor of the
            # longest official segment, then the periodic 2-second parent.
            center = primary["start_s"] + 0.5 * (primary["end_s"] - primary["start_s"])
            interval_start = center - 0.5
            interval_end = center + 0.5
            parent_id = f"{source_id}::MANIP50"
            pair = {"pair_id": f"SL{index:04d}", "source_id": source_id, "generator": fake_variant["generator"], "manipulation_operation": fake_variant["manipulation_operation"], "official_split": "train", "lineage_status": "EXACT", "real_video_path": str(real["path"]), "fake_video_path": str(fake["path"]), "real_timeline": real_timeline, "fake_timeline": fake_timeline, "all_manipulation_segments": fake_variant["manipulation_segments"], "primary_manipulation_segment": primary, "control_block": None}
            pairs.append(pair)
            for role, timeline in (("real", real_timeline), ("fake", fake_timeline)):
                frame_indices, timestamps = _frame_interval(timeline, interval_start, interval_end)
                label, category = _label(role, interval_start, interval_start + 1.0, fake_variant["manipulation_segments"])
                windows.append({"window_id": f"{parent_id}::b00::{role}", "pair_id": pair["pair_id"], "source_id": source_id, "role": role, "kind": "MANIP", "anchor_fraction": 0.5, "interval_start_s": interval_start, "interval_end_s": interval_end, "center_s": center, "frame_indices": frame_indices, "timestamps_s": timestamps, "label": label, "annotation_category": category, "video_path": str(timeline["path"])})
        except Exception as exc:
            exclusions.append({"source_id": source_id, "reason": f"TIMELINE_OR_WINDOW_FAILURE:{type(exc).__name__}:{exc}"})
    # The periodic runner expects one MANIP50 source row per role and builds
    # the b=0/0.5/1.0 children itself from the full timeline.
    STAGING_ROOT.joinpath("manifests").mkdir(parents=True, exist_ok=True)
    atomic_json(STAGING_ROOT / "manifests/selected_pairs.json", pairs)
    atomic_json(STAGING_ROOT / "manifests/window_manifest.json", windows)
    atomic_json(root / "manifests/window_manifest.json", {"source_manifest": str(STAGING_ROOT / "manifests/window_manifest.json"), "rows": windows, "exclusions": exclusions, "frozen_before_model_results": True})
    atomic_json(root / "manifests/input_pairs.json", pairs)
    summary = {"status": "COMPLETE" if not exclusions else "PARTIAL", "source_count": len(pairs), "window_seed_rows": len(windows), "exclusions": exclusions}
    atomic_json(root / "manifests/window_plan_summary.json", summary)
    return summary


def prepare_periodic(root: Path) -> dict[str, Any]:
    from research_tools.v7.periodic_requery_probe import runner as periodic
    frozen_protocol = json.loads((root / "protocol.json").read_text(encoding="utf-8"))
    periodic.BASE_ROOT = STAGING_ROOT
    result = periodic.prepare(root)
    # The periodic adapter expands each frozen MANIP50 row into the actual
    # b=0/0.5/1.0 children.  Keep that expanded list as this experiment's
    # frozen window manifest, rather than leaving only the staging rows.
    expanded = json.loads((root / "manifests/subwindows.json").read_text(encoding="utf-8"))
    atomic_json(root / "manifests/window_manifest.json", {"source_manifest": str(STAGING_ROOT / "manifests/window_manifest.json"), "rows": expanded, "frozen_before_model_results": True})
    # Preserve the learning-curve protocol as the authoritative top-level
    # protocol; periodic.prepare writes its implementation details separately.
    atomic_json(root / "manifests/periodic_prepare_summary.json", result)
    atomic_json(root / "protocol.json", frozen_protocol)
    return result


def run_frontend(root: Path, budget_s: float, resume: bool) -> dict[str, Any]:
    from research_tools.v7.periodic_requery_probe import runner as periodic
    periodic.BASE_ROOT = STAGING_ROOT
    return periodic.run_frontend(root, budget_s=budget_s, resume=resume)


def run_features(root: Path) -> dict[str, Any]:
    from research_tools.v7.periodic_requery_probe import runner as periodic
    periodic.BASE_ROOT = STAGING_ROOT
    return periodic.features(root)


def _load_r_rows(root: Path) -> list[dict[str, Any]]:
    rows = json.loads((root / "support/window_support.json").read_text(encoding="utf-8"))
    return [dict(row) for row in rows if str(row.get("mode")) == "R" and int(row.get("valid_unit_count", 0)) > 0 and row.get("label") in (0, 1) and row.get("features", {}).get("SET_A") is not None]


def _budget_path(root: Path) -> Path:
    return root / "state/training_budget.json"


def _load_budget(root: Path, budget_s: float) -> dict[str, Any]:
    old = json.loads(_budget_path(root).read_text(encoding="utf-8")) if _budget_path(root).is_file() else {}
    previous = float(old.get("cumulative_s", 0.0))
    if old and abs(float(old.get("budget_s", budget_s)) - float(budget_s)) > 1e-9:
        raise ValueError("TRAINING_BUDGET_ARGUMENT_MISMATCH")
    return {"budget_s": float(budget_s), "elapsed_before_this_process_s": previous, "process_start_unix": time.time(), "process_started_monotonic": time.monotonic(), "stop_reason": None}


def _budget_elapsed(state: Mapping[str, Any]) -> float:
    return float(state["elapsed_before_this_process_s"] + time.monotonic() - state["process_started_monotonic"])


def _save_budget(root: Path, state: Mapping[str, Any], reason: str | None = None, **extra: Any) -> None:
    elapsed = _budget_elapsed(state)
    atomic_json(_budget_path(root), {"budget_s": float(state["budget_s"]), "elapsed_before_this_process_s": elapsed, "process_start_unix": state["process_start_unix"], "process_elapsed_s": elapsed - float(state["elapsed_before_this_process_s"]), "cumulative_s": elapsed, "stop_reason": reason, **extra})


def _model_batch(rows: Sequence[Mapping[str, Any]], standardizer: Any) -> tuple[dict[str, Any], np.ndarray]:
    from research_tools.v7.periodic_requery_probe import runner as periodic
    values = [{**dict(row), "features": {"SET_A": np.asarray(row["features"]["SET_A"], dtype=np.float64)}} for row in rows]
    weights = periodic.source_class_weights(values)
    batch = periodic.make_batch("SET_A", values, standardizer)
    batch["window_weights"] = weights
    return batch, weights


def _records_save(path: Path, records: Sequence[Mapping[str, Any]]) -> None:
    atomic_json(path, {"condition": "SUMMARY_SET", "base_condition": "SET_A", "seeds": list(MODEL_SEEDS), "epochs": EPOCHS, "records": list(records)})


def train(root: Path, budget_s: float = TRAINING_BUDGET_S, device: str = "cuda", resume: bool = True) -> dict[str, Any]:
    global STOP_REQUESTED
    import torch
    from research_tools.v7.periodic_requery_probe import runner as periodic
    split_rows, subsets = _load_split(root)
    rows = _load_r_rows(root)
    validation_sources = [str(x) for x in subsets["validation_sources"]]
    models_path = root / "models/fold_models.json"
    records = json.loads(models_path.read_text(encoding="utf-8")).get("records", []) if resume and models_path.is_file() else []
    complete = {(str(x["ordering_seed"]), int(x["source_count"]), int(x["seed"])) for x in records}
    jobs = [(str(order_seed), int(size), int(seed)) for order_seed in ORDER_SEEDS for size in TRAINING_SIZES for seed in MODEL_SEEDS]
    state = _load_budget(root, budget_s)
    root.joinpath("models").mkdir(parents=True, exist_ok=True)
    train_manifest: list[dict[str, Any]] = []
    subsets_map = subsets["subsets"]
    for order_seed in ORDER_SEEDS:
        for size in TRAINING_SIZES:
            train_sources = [str(x) for x in subsets_map[str(order_seed)][str(size)]]
            train_rows = [row for row in rows if str(row["source_id"]) in set(train_sources)]
            classes = {int(row["label"]) for row in train_rows}
            for row in train_rows:
                train_manifest.append({"ordering_seed": int(order_seed), "source_count": int(size), "window_id": row["window_id"], "source_id": row["source_id"], "role": row["role"], "label": row["label"], "valid_unit_count": row.get("valid_unit_count", 0), "included": bool(classes == {0, 1})})
    write_csv(root / "models/training_window_manifest.csv", train_manifest)
    total = len(jobs)
    progress(root, "train", "RUNNING", len(complete), total, training_source_pool=POOL_SOURCE_COUNT, validation_source_count=len(validation_sources))
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA_UNAVAILABLE")
    for order_seed, size, seed in jobs:
        if (order_seed, size, seed) in complete:
            continue
        if STOP_REQUESTED or _budget_elapsed(state) >= float(state["budget_s"]):
            reason = "STOPPED_SAFE" if STOP_REQUESTED else "TRAINING_BUDGET_EXHAUSTED"
            _save_budget(root, state, reason, completed_jobs=len(complete), total_jobs=total)
            _records_save(models_path, records)
            progress(root, "train", reason, len(complete), total)
            return {"status": reason, "completed_jobs": len(complete), "total_jobs": total}
        train_sources = [str(x) for x in subsets_map[str(order_seed)][str(size)]]
        train_rows = [row for row in rows if str(row["source_id"]) in set(train_sources)]
        if {int(row["label"]) for row in train_rows} != {0, 1}:
            records.append({"ordering_seed": int(order_seed), "source_count": int(size), "seed": int(seed), "status": "SKIPPED_NO_BOTH_CLASSES", "training_sources": train_sources})
            complete.add((order_seed, size, seed)); _records_save(models_path, records); continue
        values = [{**dict(row), "features": {"SET_A": np.asarray(row["features"]["SET_A"], dtype=np.float64)}} for row in train_rows]
        weights = periodic.source_class_weights(values)
        standardizer = periodic.fit_standardizer("SET_A", [row["features"]["SET_A"] for row in values], weights)
        batch, weights = _model_batch(train_rows, standardizer)
        started = time.perf_counter()
        model, fit = periodic.train_one("SET_A", batch, seed=seed, device=device, epochs=EPOCHS, epoch_callback=lambda epoch, metrics: print(f"source-learning order={order_seed} size={size} seed={seed} epoch={epoch}/{EPOCHS} loss={float(metrics['loss']):.6f} elapsed={_budget_elapsed(state):.1f}s", flush=True))
        record = {"ordering_seed": int(order_seed), "source_count": int(size), "seed": int(seed), "status": "TRAIN_COMPLETE", "condition": "SUMMARY_SET", "base_condition": "SET_A", "training_sources": train_sources, "validation_sources": validation_sources, "training_window_count": len(train_rows), "training_real_count": sum(int(row["label"]) == 0 for row in train_rows), "training_fake_count": sum(int(row["label"]) == 1 for row in train_rows), "parameter_count": periodic.parameter_count("SET_A"), "fit": fit, "standardization": standardizer.as_dict(), "elapsed_s": time.perf_counter() - started, "state_dict": periodic.model_state(model)}
        records.append(record); complete.add((order_seed, size, seed)); _records_save(models_path, records)
        progress(root, "train", "RUNNING", len(complete), total, current_ordering_seed=order_seed, current_source_count=size, current_seed=seed)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    _save_budget(root, state, None, completed_jobs=len(complete), total_jobs=total)
    progress(root, "train", "COMPLETE", len(complete), total)
    return {"status": "COMPLETE", "completed_jobs": len(complete), "total_jobs": total, "elapsed_s": _budget_elapsed(state)}


def _model_from_record(record: Mapping[str, Any], device: str) -> Any:
    from research_tools.v7.periodic_requery_probe import runner as periodic
    # Reuse the audited SET_A loader, whose model class is defined in the
    # existing multi-order implementation rather than exported by the
    # periodic runner module.
    return periodic._model_from_record("SET_A", record, device)


def _metrics(labels: Sequence[int], scores: Sequence[float]) -> dict[str, Any]:
    from research_tools.v7.periodic_requery_probe.runner import _ap, _auroc, _classification
    return {"window_count": len(labels), "real_count": sum(int(x) == 0 for x in labels), "fake_count": sum(int(x) == 1 for x in labels), "pooled_auroc": _auroc(labels, scores), "pooled_ap": _ap(labels, scores), "classification": _classification(labels, scores)}


def _source_bootstrap(values: Mapping[str, float]) -> dict[str, Any]:
    keys = sorted(values)
    if not keys:
        return {"source_count": 0, "mean": None, "ci95": [None, None], "seed": BOOTSTRAP_SEED, "replicates": BOOTSTRAP_REPLICATES}
    raw = np.asarray([values[key] for key in keys], dtype=np.float64)
    draws = raw[np.random.default_rng(BOOTSTRAP_SEED).integers(0, len(raw), size=(BOOTSTRAP_REPLICATES, len(raw)))].mean(axis=1)
    return {"source_count": len(keys), "sources": keys, "mean": float(raw.mean()), "ci95": [float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))], "seed": BOOTSTRAP_SEED, "replicates": BOOTSTRAP_REPLICATES}


def evaluate(root: Path, device: str = "cuda") -> dict[str, Any]:
    import torch
    from research_tools.v7.periodic_requery_probe import runner as periodic
    rows = _load_r_rows(root)
    _, subsets = _load_split(root)
    validation_sources = set(str(x) for x in subsets["validation_sources"])
    rows = [row for row in rows if str(row["source_id"]) in validation_sources]
    dual = {source for source in validation_sources if {int(row["label"]) for row in rows if str(row["source_id"]) == source} == {0, 1}}
    rows = [row for row in rows if str(row["source_id"]) in dual]
    records = json.loads((root / "models/fold_models.json").read_text(encoding="utf-8"))["records"]
    score_rows: dict[str, dict[str, Any]] = {str(row["window_id"]): {"window_id": row["window_id"], "source_id": row["source_id"], "role": row["role"], "label": row["label"], "annotation_category": row.get("annotation_category"), "offset_s": row.get("offset_s"), "valid_unit_count": row.get("valid_unit_count", 0)} for row in rows}
    metrics: list[dict[str, Any]] = []
    for record in records:
        if record.get("status") != "TRAIN_COMPLETE":
            continue
        model = _model_from_record(record, device)
        standardizer = periodic.FeatureStandardizer(str(record["standardization"]["condition"]), np.asarray(record["standardization"]["mean"], dtype=np.float64), np.asarray(record["standardization"]["scale"], dtype=np.float64), tuple(int(x) for x in record["standardization"].get("zero_variance_dimensions", [])))
        batch = periodic.make_batch("SET_A", [{**dict(row), "features": {"SET_A": np.asarray(row["features"]["SET_A"], dtype=np.float64)}} for row in rows if str(row["source_id"]) in dual], standardizer, require_labels=False)
        scores = periodic.score_batch("SET_A", model, batch, device)
        for window_id, score in zip(batch["window_ids"], scores):
            score_rows[str(window_id)][f"ordering_{record['ordering_seed']}_size_{record['source_count']}_seed_{record['seed']}"] = float(score)
        del model
        if torch.cuda.is_available(): torch.cuda.empty_cache()
    # Average seed logits only after all three scores exist.
    summaries: list[dict[str, Any]] = []
    for order_seed in ORDER_SEEDS:
        for size in TRAINING_SIZES:
            condition = f"ordering_{order_seed}_size_{size}"
            seed_names = [f"{condition}_seed_{seed}" for seed in MODEL_SEEDS]
            eligible = [row for row in score_rows.values() if all(name in row for name in seed_names)]
            for row in eligible:
                row[f"{condition}_logit"] = float(np.mean([float(row[name]) for name in seed_names]))
            by_source: dict[str, float] = {}
            for source in sorted(dual):
                source_rows = [row for row in eligible if str(row["source_id"]) == source]
                labels = [int(row["label"]) for row in source_rows]; scores = [float(row[f"{condition}_logit"]) for row in source_rows]
                value = periodic._auroc(labels, scores)
                if value is not None: by_source[source] = value
            labels = [int(row["label"]) for row in eligible]; scores = [float(row[f"{condition}_logit"]) for row in eligible]
            summaries.append({"ordering_seed": order_seed, "source_count": size, "seed": "MEAN_LOGIT", "source_macro": _source_bootstrap(by_source), **_metrics(labels, scores), "validation_source_count": len(dual)})
            for seed in MODEL_SEEDS:
                seed_name = f"{condition}_seed_{seed}"
                seed_rows = [row for row in score_rows.values() if seed_name in row]
                per_source = {source: periodic._auroc([int(r["label"]) for r in seed_rows if str(r["source_id"]) == source], [float(r[seed_name]) for r in seed_rows if str(r["source_id"]) == source]) for source in dual}
                per_source = {k: v for k, v in per_source.items() if v is not None}
                summaries.append({"ordering_seed": order_seed, "source_count": size, "seed": seed, "source_macro": _source_bootstrap(per_source), **_metrics([int(r["label"]) for r in seed_rows], [float(r[seed_name]) for r in seed_rows]), "validation_source_count": len(dual)})
    write_csv(root / "evaluation/learning_curve_metrics.csv", summaries)
    write_csv(root / "scores/validation_window_scores.csv", list(score_rows.values()))
    summary = {"validation_sources": sorted(dual), "validation_source_count": len(dual), "validation_window_count": len(score_rows), "conditions": summaries, "main_metric": "source-macro AUROC and pooled AUROC on the same label-qualified R-valid dual-role validation windows", "ap_tiou": "NOT_EVALUATED"}
    atomic_json(root / "evaluation/summary.json", summary)
    progress(root, "evaluate", "COMPLETE", len(summaries), len(ORDER_SEEDS) * len(TRAINING_SIZES) * (len(MODEL_SEEDS) + 1))
    return summary


def report(root: Path) -> dict[str, Any]:
    protocol = json.loads((root / "protocol.json").read_text(encoding="utf-8"))
    split_rows, subsets = _load_split(root)
    final = json.loads((root / "evaluation/summary.json").read_text(encoding="utf-8")) if (root / "evaluation/summary.json").is_file() else {}
    media = json.loads((root / "acquisition/media_manifest.json").read_text(encoding="utf-8")) if (root / "acquisition/media_manifest.json").is_file() else {"results": []}
    frontend = json.loads((root / "frontend/results.json").read_text(encoding="utf-8")) if (root / "frontend/results.json").is_file() else []
    support = json.loads((root / "support/window_support.json").read_text(encoding="utf-8")) if (root / "support/window_support.json").is_file() else []
    models = json.loads((root / "models/fold_models.json").read_text(encoding="utf-8")) if (root / "models/fold_models.json").is_file() else {"records": []}
    lines = ["# V7 当前数据集内跨 source 训练规模学习曲线", "", "本报告是开发性 validation pilot，不是 sealed test、full-video 检测或最终泛化结论。", "", "## 先看结论", "", f"- 冻结 validation source：{len(subsets['validation_sources'])}；最大训练 source pool：{len(subsets['max_training_pool'])}；两种 ordering 共用该 pool。", f"- 媒体状态：{len(media.get('results', []))} 个文件；前端结果行：{len(frontend)}；R 支撑行：{sum(str(row.get('mode')) == 'R' for row in support)}；训练模型记录：{len(models.get('records', []))}。", "- 主评价只使用 validation 中标签合格、R-valid 且同时有 real/fake 的 source，并让 pooled 与 source-macro 使用同一窗口集合。", "- 本轮不把 AP 当时间定位 AP；`AP@tIoU = NOT_EVALUATED`。", "", "## 冻结身份", "", f"- protocol：`protocol.json`；source split：`manifests/source_split_manifest.csv`；subsets：`manifests/training_subsets.json`；window manifest：`manifests/window_manifest.json`。", f"- 历史开发 source 排除数：{len(protocol['historical_development_sources_excluded'])}；官方 split：`train`。", "", "## 学习曲线（source-macro AUROC；MEAN_LOGIT 行）", "", "| ordering seed | training sources | validation sources | macro AUROC | 95% CI | pooled AUROC | AP | F1 | windows |", "|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in final.get("conditions", []):
        if row.get("seed") != "MEAN_LOGIT": continue
        macro = row.get("source_macro", {})
        cls = row.get("classification", {})
        lines.append(f"| {row.get('ordering_seed')} | {row.get('source_count')} | {row.get('validation_source_count')} | {macro.get('mean')} | {macro.get('ci95')} | {row.get('pooled_auroc')} | {row.get('pooled_ap')} | {cls.get('f1')} | {row.get('window_count')} |")
    lines += ["", "## 训练与覆盖限制", "", "- 训练使用 `SetAModel`/SET_A SUMMARY_SET 实现、source/class weighted BCE、subset-only standardization、Adam、200 epochs 与三个模型 seed；不按 validation 指标选 seed、epoch 或阈值。", "- 新 source 的前端沿用已有周期 re-query 实现；其 O/R 粒子产物均保留以复用已审计几何路径，但本学习曲线只使用 R 特征。", "- 窗口、source 和真实/伪造标签在模型结果前冻结；boundary/outside 不进入主监督样本，缺失不填 0。", "- 两种 source ordering、四个名义规模和三个模型 seed 的模型记录、逐窗口分数及训练 loss 位于数据目录；大数组、视频和权重不进入 Git。", "", "## 边界声明", "", "未访问旧 R7/V5；未修改正式 `src` 检测链；未运行官方测试 split；没有空间真值、像素定位或未知生成器泛化证据；本报告只支持当前开发 source 学习曲线的描述。", ""]
    path = root / "report.md"; path.write_text("\n".join(lines), encoding="utf-8")
    status = "COMPLETE" if len(models.get("records", [])) == len(ORDER_SEEDS) * len(TRAINING_SIZES) * len(MODEL_SEEDS) and bool(final.get("conditions")) else "PARTIAL"
    atomic_json(root / "final_status.json", {"status": status, "frontend_result_rows": len(frontend), "r_support_rows": sum(str(row.get('mode')) == 'R' for row in support), "model_count": len(models.get('records', [])), "expected_model_count": len(ORDER_SEEDS) * len(TRAINING_SIZES) * len(MODEL_SEEDS), "evaluation_present": bool(final.get('conditions')), "report": str(path), "git_head": git_head(), "updated_unix": time.time()})
    progress(root, "report", status, 1 if status == "COMPLETE" else 0, 1)
    return {"status": status, "report": str(path)}


def run_smoke(root: Path, device: str = "cuda") -> dict[str, Any]:
    """One complete 200-epoch smallest-subset endpoint, kept outside formal records."""

    import torch
    from research_tools.v7.periodic_requery_probe import runner as periodic
    _, subsets = _load_split(root); rows = _load_r_rows(root); source_ids = set(subsets["subsets"][str(ORDER_SEEDS[0])][str(TRAINING_SIZES[0])]); val_ids = set(subsets["validation_sources"])
    train_rows = [row for row in rows if str(row["source_id"]) in source_ids]; val_rows = [row for row in rows if str(row["source_id"]) in val_ids]
    weights = periodic.source_class_weights(train_rows); standardizer = periodic.fit_standardizer("SET_A", [row["features"]["SET_A"] for row in train_rows], weights); batch, _ = _model_batch(train_rows, standardizer); model, fit = periodic.train_one("SET_A", batch, seed=MODEL_SEEDS[0], device=device, epochs=EPOCHS); val_batch = periodic.make_batch("SET_A", val_rows, standardizer, require_labels=False); scores = periodic.score_batch("SET_A", model, val_batch, device); output = {"status": "PASS", "ordering_seed": ORDER_SEEDS[0], "source_count": TRAINING_SIZES[0], "seed": MODEL_SEEDS[0], "training_window_count": len(train_rows), "validation_window_count": len(val_rows), "scored_count": len(scores), "epochs": fit["epochs"], "device": str(torch.device(device)), "formal_records_untouched": True}; atomic_json(root / "smoke/summary.json", output); del model; return output


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("plan", "download", "windows", "prepare", "frontend", "features", "smoke", "train", "evaluate", "report", "all"))
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--frontend-budget-s", type=float, default=FRONTEND_BUDGET_S)
    parser.add_argument("--train-budget-s", type=float, default=TRAINING_BUDGET_S)
    args = parser.parse_args()
    root = args.output_root; root.mkdir(parents=True, exist_ok=True)
    atomic_json(root / "state/launch.json", {"stage": args.stage, "pid": os.getpid(), "git_head": git_head(), "started_unix": time.time(), "device": args.device})
    try:
        if args.stage in {"plan", "all"}: freeze_plan(root)
        if args.stage in {"download", "all"}: download_media(root, workers=args.workers)
        if args.stage in {"windows", "all"}: build_windows(root)
        if args.stage in {"prepare", "all"}: prepare_periodic(root)
        if args.stage in {"frontend", "all"}: run_frontend(root, args.frontend_budget_s, args.resume)
        if args.stage in {"features", "all"}: run_features(root)
        if args.stage in {"smoke", "all"}: run_smoke(root, args.device)
        if args.stage in {"train", "all"}: train(root, args.train_budget_s, args.device, args.resume)
        if args.stage in {"evaluate", "all"}: evaluate(root, args.device)
        if args.stage in {"report", "all"}: report(root)
        atomic_json(root / "state/last_exit.json", {"status": "OK", "stage": args.stage, "updated_unix": time.time()})
        return 0
    except BaseException as exc:
        atomic_json(root / "state/last_exit.json", {"status": "FAILED", "stage": args.stage, "error": f"{type(exc).__name__}: {exc}", "updated_unix": time.time()})
        raise


if __name__ == "__main__":
    raise SystemExit(main())
