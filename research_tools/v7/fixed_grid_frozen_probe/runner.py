"""Independent execution for the V7 fixed-grid frozen-model pilot.

This runner intentionally reuses the already audited 289-point frontend and
local-organization implementation.  It writes only to the data disk and can
be resumed after a process stop; it never changes the formal ``src`` chain or
the historical pilot outputs.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import os
import random
import signal
import subprocess
import time
import traceback
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence
from xml.sax.saxutils import escape

import numpy as np

from sparse3d_forgery.particle_sequence import (
    CoordinateSystem,
    Handedness,
    LengthUnit,
    build_particle_sequence,
    load_particle_sequence,
    save_particle_sequence,
)
from sparse3d_forgery.video_input import VideoSource, decode_video
from research_tools.v7.boundary_pooling_probe.grouping import (
    assign_uv_to_masks,
    split_local_groups_by_assignment,
    validate_boundary_partition,
)
from research_tools.v7.boundary_pooling_probe.model import PoolingWindowMLP, score_pooling_model
from research_tools.v7.boundary_pooling_probe.pipeline import (
    SEG_WEIGHT,
    _compact_grouping,
    _compact_support,
    _mask_cache_key,
    _polygons_to_masks,
    _predict_masks,
    _sha256 as file_sha256,
)
from research_tools.v7.local_organization_probe.grouping import (
    build_local_groups,
    build_local_support,
    rebuild_components_fast,
)
from research_tools.v7.local_structural_temporal_probe.model import (
    ARM_NAMES,
    WeightedStandardizer,
    build_batch,
)
from research_tools.v7.local_structural_temporal_probe.representation import COMPONENT_CONFIG
from research_tools.v7.observation_density_diagnostic import run_diagnostic as diagnostic
from .grid import classify_fake_window, generate_grid_windows, map_target_frames_to_intervals


REPO_ROOT = Path(__file__).resolve().parents[3]
DATA_ROOT = Path("/root/autodl-tmp/data/sparse_3d_forgery_detection")
SOURCE_ROOT = DATA_ROOT / "derived/v7_activityforensics_boundary_pooling_pilot_v1"
PAIRED_ROOT = DATA_ROOT / "derived/v7_activityforensics_paired_second_order_pilot_v1"
MODEL_ROOT = SOURCE_ROOT / "models/fold_models.json"
OUTPUT_ROOT = DATA_ROOT / "derived/v7_activityforensics_fixed_grid_frozen_probe_v1"
WINDOW_LENGTH_S = 1.0
STRIDE_S = 0.5
SEED = 20260909
CONDITIONS = ("H_MEAN_A", "B_MEAN_A", "B_MEAN_C")
ARM_BY_CONDITION = {"H_MEAN_A": ("H", "UNORDERED_STATE"), "B_MEAN_A": ("B", "UNORDERED_STATE"), "B_MEAN_C": ("B", "ORDERED_SECOND")}
EXPECTED_QUERY_COUNT = 289
MAX_FRONTEND_ATTEMPTS = 3
PHASE_RESERVE_S = 1200.0
BUDGET_STATE_NAME = "budget_state.json"
RUN_LOCK_NAME = "run.lock"
BUDGET_SCOPE = "frontend_end_to_end_until_selected_source_prefix_or_budget; postprocess_excluded"
TAPNET_SOURCE = diagnostic.TAPNET_SOURCE
TAPNET_CHECKPOINT = diagnostic.TAPNET_CHECKPOINT
DEPTH_SOURCE = diagnostic.DEPTH_SOURCE
DEPTH_CHECKPOINT = diagnostic.DEPTH_CHECKPOINT
TRACKER_SHA = diagnostic.TRACKER_SHA
DEPTH_SHA = diagnostic.DEPTH_SHA


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    raise TypeError(type(value).__name__)


def _safe(value: str) -> str:
    return "".join(char if char.isalnum() or char in "._-" else "_" for char in value)


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False, default=_json_default) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    names: list[str] = []
    for row in rows:
        for name in row:
            if name not in names:
                names.append(name)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=names, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def _progress(root: Path, phase: str, completed: int, total: int, status: str = "RUNNING", **extra: Any) -> None:
    _atomic_json(root / "progress.json", {"phase": phase, "completed": int(completed), "total": int(total), "status": status, "updated_unix": time.time(), **extra})


def _acquire_run_lock(root: Path) -> Path:
    """Prevent two independent runners from writing the same experiment."""

    path = root / RUN_LOCK_NAME
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    except FileExistsError:
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
            pid = int(record.get("pid", -1))
        except Exception:
            pid = -1
        if pid > 0:
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                stale = path.with_name(f"{path.name}.stale.{int(time.time())}")
                os.replace(path, stale)
                fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
            except PermissionError:
                raise RuntimeError(f"RUN_LOCK_ACTIVE:{path}:pid={pid}") from None
            else:
                raise RuntimeError(f"RUN_LOCK_ACTIVE:{path}:pid={pid}") from None
        else:
            raise RuntimeError(f"RUN_LOCK_ACTIVE:{path}") from None
    os.write(fd, (json.dumps({"pid": os.getpid(), "started_unix": time.time()}) + "\n").encode("utf-8"))
    os.close(fd)
    return path


def _release_run_lock(path: Path) -> None:
    try:
        path.unlink()
    except FileNotFoundError:
        pass


def _load_budget_state(root: Path, budget_s: float, existing_results: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """Load persistent budget accounting without resetting it on resume.

    The two historical benchmark windows are explicitly outside the formal
    7200-second run.  If there is no such bounded-benchmark evidence and no
    persisted state, execution is refused rather than silently resetting an
    unknown budget.
    """

    path = root / "manifests" / BUDGET_STATE_NAME
    if path.is_file():
        state = json.loads(path.read_text(encoding="utf-8"))
        if abs(float(state.get("budget_s", budget_s)) - float(budget_s)) > 1e-6:
            raise RuntimeError("BUDGET_ARGUMENT_MISMATCH: persisted budget differs from requested budget")
        state.setdefault("budget_scope", BUDGET_SCOPE)
        state.setdefault("elapsed_before_this_process_s", float(state.get("consumed_s", 0.0)))
        return state
    benchmark = root / "evaluation" / "benchmark.json"
    if benchmark.is_file() and existing_results:
        elapsed = sum(float(item.get("elapsed_s", 0.0)) for item in existing_results.values() if item.get("sequence_prefix"))
        state = {
            "budget_s": float(budget_s),
            "consumed_s": 0.0,
            "prior_benchmark_elapsed_s": elapsed,
            "accounting_status": "FORMAL_RUN_NOT_STARTED_BOUNDED_BENCHMARK",
            "budget_scope": BUDGET_SCOPE,
            "elapsed_before_this_process_s": 0.0,
            "updated_unix": time.time(),
        }
        _atomic_json(path, state)
        return state
    raise RuntimeError("BUDGET_ACCOUNTING_UNKNOWN: no persisted runtime state or bounded benchmark evidence")


def _start_budget_clock(state: Mapping[str, Any], *, monotonic_start: float | None = None, wall_start: float | None = None) -> dict[str, float]:
    """Start one process clock without folding its elapsed time into itself twice."""

    before = float(state.get("consumed_s", 0.0))
    return {
        "elapsed_before_this_process_s": before,
        "process_start_monotonic": float(time.monotonic() if monotonic_start is None else monotonic_start),
        "process_start_unix": float(time.time() if wall_start is None else wall_start),
    }


def _budget_snapshot(clock: Mapping[str, float], *, monotonic_now: float | None = None) -> tuple[float, float]:
    """Return (cumulative_budget_seconds, this_process_seconds)."""

    process_elapsed = max(
        0.0,
        float(time.monotonic() if monotonic_now is None else monotonic_now) - float(clock["process_start_monotonic"]),
    )
    return float(clock["elapsed_before_this_process_s"]) + process_elapsed, process_elapsed


def _persist_budget_state(
    root: Path,
    state: dict[str, Any],
    clock: Mapping[str, float],
    *,
    stop_reason: str | None = None,
    **extra: Any,
) -> None:
    cumulative, process_elapsed = _budget_snapshot(clock)
    state["budget_scope"] = BUDGET_SCOPE
    state["elapsed_before_this_process_s"] = float(clock["elapsed_before_this_process_s"])
    state["process_start_unix"] = float(clock["process_start_unix"])
    state["process_elapsed_s"] = process_elapsed
    state["consumed_s"] = cumulative
    if stop_reason is not None:
        state["last_stop_reason"] = str(stop_reason)
        if not state.get("stop_recorded_this_process", False):
            state["stop_cumulative_s"] = cumulative
            state["stop_budget_s"] = float(state.get("budget_s", 0.0))
            state["stop_recorded_this_process"] = True
        state["accounting_status"] = str(stop_reason)
    if state.get("accounting_status") == "FORMAL_RUN_NOT_STARTED_BOUNDED_BENCHMARK":
        state["accounting_status"] = "RUNNING"
    state["updated_unix"] = time.time()
    state.update(extra)
    _atomic_json(root / "manifests" / BUDGET_STATE_NAME, state)


def _update_budget_state(
    root: Path,
    state: dict[str, Any],
    elapsed_s: float,
    *,
    clock: Mapping[str, float] | None = None,
    **extra: Any,
) -> None:
    """Persist budget progress using one cumulative process clock.

    ``elapsed_s`` is retained for callers outside the runner, but the live
    frontend always supplies ``clock``.  That path never adds a window
    duration to a value which already includes the current process.
    """

    if clock is None:
        state["consumed_s"] = float(state.get("consumed_s", 0.0)) + max(0.0, float(elapsed_s))
        state["elapsed_before_this_process_s"] = float(state.get("consumed_s", 0.0))
        state["process_elapsed_s"] = 0.0
        state["budget_scope"] = BUDGET_SCOPE
        state["updated_unix"] = time.time()
        state.update(extra)
        _atomic_json(root / "manifests" / BUDGET_STATE_NAME, state)
        return
    _persist_budget_state(root, state, clock, **extra)


def _plan_identity(row: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        str(row.get("window_id")),
        str(row.get("source_video_id")),
        str(row.get("source_id")),
        str(row.get("role")),
        int(row.get("grid_index")),
        tuple(int(value) for value in row.get("frame_indices", [])),
        tuple(float(value) for value in row.get("timestamps_s", [])),
        float(row.get("interval_start_s", 0.0)),
        float(row.get("interval_end_s", 0.0)),
    )


def compare_plan_identity(expected: Sequence[Mapping[str, Any]], actual: Sequence[Mapping[str, Any]]) -> list[str]:
    """Return deterministic differences without replacing a frozen plan."""

    left = {str(row.get("window_id")): _plan_identity(row) for row in expected}
    right = {str(row.get("window_id")): _plan_identity(row) for row in actual}
    differences: list[str] = []
    for window_id in sorted(set(left) | set(right)):
        if window_id not in left:
            differences.append(f"unexpected:{window_id}")
        elif window_id not in right:
            differences.append(f"missing:{window_id}")
        elif left[window_id] != right[window_id]:
            differences.append(f"mismatch:{window_id}")
    return differences


def _git_head() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()


def _probe_video(path: Path) -> dict[str, Any]:
    import av

    pts: list[float] = []
    width = height = None
    codec = None
    with av.open(str(path)) as container:
        stream = next(stream for stream in container.streams if stream.type == "video")
        codec = str(stream.codec_context.name)
        for frame in container.decode(stream):
            if frame.pts is None or frame.time_base is None:
                raise RuntimeError(f"PTS_MISSING:{path}")
            stamp = float(frame.pts * frame.time_base)
            if not np.isfinite(stamp) or (pts and stamp <= pts[-1]):
                raise RuntimeError(f"PTS_NOT_STRICT:{path}")
            pts.append(stamp)
            height, width = int(frame.height), int(frame.width)
    if not pts:
        raise RuntimeError(f"NO_VIDEO_FRAMES:{path}")
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": file_sha256(path), "frame_count": len(pts), "frame_indices": list(range(len(pts))), "timestamps_s": pts, "width": width, "height": height, "codec": codec, "origin_pts_s": pts[0], "duration_relative_s": pts[-1] - pts[0]}


def _selected_pairs() -> dict[str, dict[str, Any]]:
    path = PAIRED_ROOT / "manifests/selected_pairs.json"
    values = json.loads(path.read_text(encoding="utf-8"))
    return {str(row["source_id"]): row for row in values}


def prepare_plan(root: Path) -> dict[str, Any]:
    input_rows = json.loads((SOURCE_ROOT / "manifests/input_manifest.json").read_text(encoding="utf-8"))["rows"]
    by_video: dict[tuple[str, str], dict[str, Any]] = {}
    for row in input_rows:
        key = (str(row["source_id"]), str(row["role"]))
        by_video.setdefault(key, {"source_id": key[0], "role": key[1], "pair_id": str(row["pair_id"]), "video_path": str(row["video_path"])})
        if by_video[key]["video_path"] != str(row["video_path"]):
            raise RuntimeError(f"multiple videos for frozen source/role: {key}")
    pairs = _selected_pairs()
    videos: list[dict[str, Any]] = []
    grids: list[dict[str, Any]] = []
    for (source_id, role), item in sorted(by_video.items()):
        path = Path(item["video_path"])
        record: dict[str, Any] = {**item, "video_id": f"{source_id}::{role}", "status": "MISSING" if not path.is_file() else "PENDING"}
        if path.is_file():
            try:
                record.update(_probe_video(path))
                record["status"] = "METADATA_COMPLETE"
            except Exception as exc:
                record.update({"status": "PTS_OR_DECODE_FAILED", "error": f"{type(exc).__name__}: {exc}"})
        # A successful probe transitions the record to METADATA_COMPLETE.
        # The previous PENDING check silently produced videos without grid
        # rows on a fresh plan; keep planning tied to the successful state.
        if record["status"] == "METADATA_COMPLETE":
            rows = generate_grid_windows(record["timestamps_s"], window_length_s=WINDOW_LENGTH_S, stride_s=STRIDE_S)
            if rows and rows[0].get("status") == "SHORT_VIDEO":
                record["status"] = "SHORT_VIDEO"
            annotations = [] if role == "real" else list(pairs.get(source_id, {}).get("all_manipulation_segments", []))
            record["annotation_intervals_relative_s"] = annotations
            for planned in rows:
                if planned.get("status") != "PLANNED":
                    continue
                grid_id = f"{source_id}::{role}::grid{int(planned['grid_index']):04d}"
                start_pts = float(record["origin_pts_s"] + planned["nominal_start_s"])
                end_pts = float(record["origin_pts_s"] + planned["nominal_end_s"])
                annotation = {"annotation_category": "REAL_NEGATIVE"} if role == "real" else classify_fake_window(start_pts, end_pts, annotations, origin_pts_s=float(record["origin_pts_s"]))
                target = {"target_frame_count": 0, "target_frames_in_annotation_count": 0, "target_pts_in_annotation_s": []} if role == "real" else map_target_frames_to_intervals(planned["frame_pts_s"], annotations, origin_pts_s=float(record["origin_pts_s"]))
                grids.append({
                    "grid_window_id": grid_id, "window_id": grid_id, "source_video_id": record["video_id"], "source_id": source_id, "pair_id": record["pair_id"], "role": role, "kind": "GRID",
                    "video_path": str(path), "grid_index": int(planned["grid_index"]), "nominal_start_s": float(planned["nominal_start_s"]), "nominal_end_s": float(planned["nominal_end_s"]),
                    "interval_start_s": start_pts, "interval_end_s": end_pts, "actual_start_pts_s": planned["actual_start_pts_s"], "actual_end_pts_s": planned["actual_end_pts_s"],
                    "frame_indices": list(planned["frame_indices"]), "timestamps_s": list(planned["frame_pts_s"]), "tail_window": bool(planned["tail_window"]), "timestamp_origin_s": float(record["origin_pts_s"]),
                    **annotation, **target, "grid_frozen_before_labels": True,
                })
        videos.append(record)
    source_ids = sorted({str(row["source_id"]) for row in videos})
    random.Random(SEED).shuffle(source_ids)
    plan = {"seed": SEED, "source_order": source_ids, "window_length_s": WINDOW_LENGTH_S, "stride_s": STRIDE_S, "video_count": len(videos), "grid_window_count": len(grids), "videos": videos}
    _atomic_json(root / "manifests/videos.json", videos)
    _atomic_json(root / "manifests/grid_windows.json", grids)
    _atomic_json(root / "manifests/execution_plan.json", {"source_order": source_ids, "seed": SEED, "window_count": len(grids), "rule": "complete source prefix in fixed randomized order; no label or score selection"})
    return plan


def write_protocol(root: Path) -> None:
    records = json.loads(MODEL_ROOT.read_text(encoding="utf-8"))
    model_sha = file_sha256(MODEL_ROOT)
    _atomic_json(root / "protocol.json", {
        "protocol_id": "v7-fixed-grid-frozen-probe-v1", "git_head": _git_head(), "question": "fixed label-blind time coverage versus observation support and frozen response",
        "window_grid": {"length_s": WINDOW_LENGTH_S, "stride_s": STRIDE_S, "time_origin": "first actual decoded PTS per video", "tail_policy": "append one ending at last PTS when needed", "short_video": "record SHORT_VIDEO; no padding"},
        "frontend": {"provider_tracker": "online_bootstapir", "tracker_source_sha": TRACKER_SHA, "tracker_checkpoint": str(TAPNET_CHECKPOINT), "depth": "apple_depth_pro", "depth_source_sha": DEPTH_SHA, "depth_checkpoint": str(DEPTH_CHECKPOINT), "process_size": 256, "query_grid": "17x17 interior linspace on process raster = 289", "visibility": "tracker visibility plus finite depth/pose geometry masks", "depth_semantics": "optical-axis z-depth", "pose": "Open3D RGB-D odometry, target camera-from-source inverted during world accumulation", "coordinates": "first_camera_world, right-handed right/down/forward, meters"},
        "window_initialization": "each grid window decodes its own frames and initializes 289 queries and causal state; no cross-window track identity",
        "representation": {"history_s": 0.5, "target_offsets_s": [0.5, 0.6, 0.7, 0.8, 0.9], "target_tolerance_s": 0.05, "component_config": {"max_initial_distance_m": float(COMPONENT_CONFIG.max_initial_distance), "max_relative_change_m": float(COMPONENT_CONFIG.max_relative_change), "minimum_size": int(COMPONENT_CONFIG.minimum_size), "minimum_overlap": int(COMPONENT_CONFIG.minimum_overlap)}, "H": "existing history local groups", "B": "first-frame segmentation partition of H; no later reassignment"},
        "frozen_scoring": {"conditions": list(CONDITIONS), "models": str(MODEL_ROOT), "models_sha256": model_sha, "available_conditions": records.get("conditions"), "seeds": records.get("seeds"), "held_out_rule": "only held_out_source=s records", "standardization": "saved fold standardizer; no refit", "score": "arithmetic mean of three seed logits; missing support is missing, never zero"},
        "labels": "applied only after grid freeze; real windows negative, fake windows fully within annotation union positive; boundary/outside descriptive only",
        "bootstrap": {"unit": "source", "seed": SEED, "replicates": 10000},
        "budget_s": 7200,
        "boundaries": ["development pilot, not sealed test", "no dense branch", "no training or calibration", "no spatial ground truth claim"],
    })


def _load_grid(root: Path) -> list[dict[str, Any]]:
    return json.loads((root / "manifests/grid_windows.json").read_text(encoding="utf-8"))


def _load_results(root: Path) -> dict[str, dict[str, Any]]:
    path = root / "frontend_results.json"
    if not path.is_file():
        return {}
    return {str(row["window_id"]): row for row in json.loads(path.read_text(encoding="utf-8")).get("results", [])}


def _save_results(root: Path, results: Mapping[str, Mapping[str, Any]]) -> None:
    _atomic_json(root / "frontend_results.json", {"results": list(sorted(results.values(), key=lambda row: str(row["window_id"])))})


def _result_reuse_check(root: Path, row: Mapping[str, Any], result: Mapping[str, Any]) -> tuple[bool, str]:
    """Validate a cached frontend result before allowing resume to skip it."""

    window_id = str(row["window_id"])
    if str(result.get("window_id")) != window_id:
        return False, "WINDOW_ID_MISMATCH"
    if str(result.get("source_video_id")) != str(row["source_video_id"]):
        return False, "SOURCE_VIDEO_ID_MISMATCH"
    prefix_value = result.get("sequence_prefix")
    if not prefix_value:
        return False, "NO_SEQUENCE_PREFIX"
    prefix = Path(str(prefix_value))
    if not prefix.with_suffix(".npz").is_file() or not prefix.with_suffix(".json").is_file():
        return False, "SEQUENCE_ARTIFACT_MISSING"
    try:
        sequence = load_particle_sequence(prefix)
    except Exception as exc:
        return False, f"SEQUENCE_LOAD_FAILED:{type(exc).__name__}"
    if sequence.source_video_id != str(row["source_video_id"]):
        return False, "SEQUENCE_SOURCE_VIDEO_ID_MISMATCH"
    if sequence.num_tracks != EXPECTED_QUERY_COUNT:
        return False, f"QUERY_COUNT_MISMATCH:{sequence.num_tracks}"
    provenance = sequence.provenance
    try:
        query_count = int(provenance.get("query_count", -1)) if isinstance(provenance, Mapping) else -1
        query_grid_size = int(provenance.get("query_grid_size", -1)) if isinstance(provenance, Mapping) else -1
    except (TypeError, ValueError):
        query_count = query_grid_size = -1
    if query_count != EXPECTED_QUERY_COUNT or query_grid_size != 17:
        return False, "QUERY_CONFIGURATION_MISSING_OR_MISMATCH"
    if not np.array_equal(np.asarray(sequence.frame_indices), np.asarray(row.get("frame_indices", []), dtype=np.int64)):
        return False, "FRAME_INDICES_MISMATCH"
    expected_timestamps = np.asarray(row.get("timestamps_s", []), dtype=np.float64)
    if not np.array_equal(np.asarray(sequence.timestamps_s, dtype=np.float64), expected_timestamps):
        if not np.allclose(np.asarray(sequence.timestamps_s, dtype=np.float64), expected_timestamps, rtol=0.0, atol=1e-9):
            return False, "PTS_MISMATCH"
    lineage = sequence.lineage
    if not isinstance(lineage, Mapping) or str(lineage.get("grid_window_id")) != window_id:
        return False, "LINEAGE_WINDOW_ID_MISSING_OR_MISMATCH"
    return True, "VALID"


def _validated_results(root: Path, rows: Sequence[Mapping[str, Any]]) -> tuple[dict[str, dict[str, Any]], dict[str, str]]:
    raw = _load_results(root)
    valid: dict[str, dict[str, Any]] = {}
    invalid: dict[str, str] = {}
    for row in rows:
        window_id = str(row["window_id"])
        result = raw.get(window_id)
        if result is None:
            continue
        ok, reason = _result_reuse_check(root, row, result)
        if ok:
            valid[window_id] = result
        else:
            invalid[window_id] = reason
    return valid, invalid


def _frontend_status_counts(rows: Sequence[Mapping[str, Any]], results: Mapping[str, Mapping[str, Any]], valid_ids: set[str]) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for row in rows:
        window_id = str(row["window_id"])
        if window_id in valid_ids:
            counts["FRONTEND_COMPLETE"] += 1
        elif window_id not in results:
            counts["NOT_RUN"] += 1
        else:
            counts[str(results[window_id].get("status", "INVALID_CACHE"))] += 1
    return dict(counts)


def _estimate_window_seconds(results: Mapping[str, Mapping[str, Any]]) -> float:
    values = [float(item["elapsed_s"]) for item in results.values() if item.get("sequence_prefix") and item.get("elapsed_s") is not None and float(item["elapsed_s"]) > 0]
    return float(np.median(np.asarray(values, dtype=np.float64))) if values else 180.0


def _select_source_prefix(root: Path, rows: Sequence[Mapping[str, Any]], valid: Mapping[str, Mapping[str, Any]], budget_state: dict[str, Any]) -> tuple[list[str], list[dict[str, Any]], str | None]:
    """Select a complete source prefix once, without label/score selection."""

    execution_path = root / "manifests" / "execution_plan.json"
    execution = json.loads(execution_path.read_text(encoding="utf-8"))
    order = [str(value) for value in execution.get("source_order", [])]
    by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_source[str(row["source_id"])].append(dict(row))
    if len(valid) == len(rows):
        selected_rows = [dict(row) for row in rows]
        return order, selected_rows, None
    old_prefix = [str(value) for value in execution.get("selected_source_prefix", [])]
    if old_prefix:
        # A persisted prefix is a frozen execution decision.  Resume may
        # finish it, but must not silently add a new source in this run.
        source_set = set(old_prefix)
        selected_rows = [dict(row) for row in rows if str(row["source_id"]) in source_set]
        selected_rows.sort(key=lambda row: (old_prefix.index(str(row["source_id"])), str(row["role"]), int(row["grid_index"])))
        if not selected_rows:
            return [], [], "FROZEN_SOURCE_PREFIX_EMPTY"
        return old_prefix, selected_rows, None
    start = 0
    prefix: list[str] = []
    estimate_per_window = _estimate_window_seconds(_load_results(root))
    consumed = float(budget_state.get("consumed_s", 0.0))
    remaining = float(budget_state.get("budget_s", 0.0)) - consumed
    if not prefix:
        estimate = 0.0
        for source in order[start:]:
            pending = sum(str(row["window_id"]) not in valid for row in by_source.get(source, []))
            source_estimate = pending * estimate_per_window
            if pending and estimate + source_estimate + PHASE_RESERVE_S > remaining:
                break
            prefix.append(source)
            estimate += source_estimate
        if prefix:
            execution.update({
                "selected_source_prefix": prefix,
                "selected_window_count": sum(len(by_source[source]) for source in prefix),
                "selection_estimate_frontend_s": estimate,
                "selection_remaining_budget_s": remaining,
                "selection_updated_unix": time.time(),
            })
            _atomic_json(execution_path, execution)
    selected_rows = [row for row in rows if str(row["source_id"]) in set(prefix)]
    selected_rows.sort(key=lambda row: (order.index(str(row["source_id"])) if str(row["source_id"]) in order else 10_000, str(row["role"]), int(row["grid_index"])))
    if prefix:
        return prefix, selected_rows, None
    return [], [], "BUDGET_INSUFFICIENT_FOR_COMPLETE_SOURCE"


def run_frontend(root: Path, budget_s: float, resume: bool = True, max_windows: int | None = None) -> dict[str, Any]:
    """Run the reused 289-point frontend in fixed source/window order."""

    rows = _load_grid(root)
    stored_results = _load_results(root) if resume else {}
    valid_results, invalid_cache = _validated_results(root, rows)
    budget_state = _load_budget_state(root, budget_s, stored_results)
    budget_clock = _start_budget_clock(budget_state)
    budget_state["accounting_status"] = "RUNNING"
    budget_state["current_process_started_unix"] = budget_clock["process_start_unix"]
    budget_state["current_process_elapsed_s"] = 0.0
    budget_state["stop_recorded_this_process"] = False
    _atomic_json(root / "manifests" / BUDGET_STATE_NAME, budget_state)
    source_prefix, selected_rows, selection_error = _select_source_prefix(root, rows, valid_results, budget_state)
    if selection_error is not None:
        _persist_budget_state(root, budget_state, budget_clock, stop_reason=selection_error)
        cumulative, process_elapsed = _budget_snapshot(budget_clock)
        _progress(root, "frontend", len(valid_results), len(rows), selection_error, planned_windows=len(rows), selected_windows=0, invalid_cache=invalid_cache, elapsed_s=process_elapsed, consumed_s=cumulative, stop_reason=selection_error)
        return {"status": selection_error, "completed": len(valid_results), "total": len(rows), "selected_windows": 0, "selected_source_prefix": []}
    if max_windows is not None:
        selected_rows = selected_rows[: int(max_windows)]
    pending = [row for row in selected_rows if str(row["window_id"]) not in valid_results]
    retry_limited = []
    for row in pending[:]:
        previous = stored_results.get(str(row["window_id"]), {})
        if int(previous.get("attempt_count", 0)) >= MAX_FRONTEND_ATTEMPTS:
            pending.remove(row)
            retry_limited.append(str(row["window_id"]))
    if not pending:
        selected_complete = all(str(row["window_id"]) in valid_results for row in selected_rows)
        status = "COMPLETE" if len(valid_results) == len(rows) else ("SOURCE_PREFIX_COMPLETE" if selected_complete else "RETRY_LIMIT_REACHED")
        _persist_budget_state(root, budget_state, budget_clock, stop_reason=status)
        cumulative, process_elapsed = _budget_snapshot(budget_clock)
        _progress(root, "frontend", len(valid_results), len(rows), status, planned_windows=len(rows), selected_windows=len(selected_rows), selected_source_prefix=source_prefix, invalid_cache=invalid_cache, retry_limited=retry_limited, elapsed_s=process_elapsed, consumed_s=cumulative, stop_reason=status)
        return {"status": status, "completed": len(valid_results), "total": len(rows), "selected_windows": len(selected_rows), "selected_source_prefix": source_prefix, "invalid_cache": invalid_cache, "retry_limited": retry_limited, "budget_state": budget_state}

    import torch
    from scripts.run_v7_explicit_geometry_frontend import (
        DepthProRunner,
        OnlineBootsTapir,
        accumulate_world_from_camera,
        causal_first_frame_intrinsics,
        persistent_track_ids,
        rgbd_odometry,
        sample_depth_at_uv,
        world_xyz,
    )

    print(f"frontend device={torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'} selected_sources={source_prefix} selected_windows={len(selected_rows)}", flush=True)
    tracker = OnlineBootsTapir(TAPNET_SOURCE, TAPNET_CHECKPOINT, process_size=256, grid_size=17)
    depth_runner = DepthProRunner(DEPTH_SOURCE, DEPTH_CHECKPOINT)
    stop_requested = False
    stop_reason: str | None = None
    results = dict(stored_results)

    def stop(_signum: int, _frame: Any) -> None:
        nonlocal stop_requested
        stop_requested = True

    old_handler = signal.signal(signal.SIGTERM, stop)
    old_int = signal.signal(signal.SIGINT, stop)
    try:
        for number, row in enumerate(pending, 1):
            cumulative, process_elapsed = _budget_snapshot(budget_clock)
            if stop_requested or cumulative >= float(budget_state.get("budget_s", budget_s)):
                stop_reason = "STOPPED_SAFE" if stop_requested else "BUDGET_EXHAUSTED"
                _persist_budget_state(root, budget_state, budget_clock, stop_reason=stop_reason, last_window_status="STOPPED_BEFORE_NEXT_WINDOW")
                _progress(root, "frontend", len(valid_results), len(rows), stop_reason, planned_windows=len(rows), selected_windows=len(selected_rows), selected_source_prefix=source_prefix, elapsed_s=process_elapsed, consumed_s=cumulative, stop_reason=stop_reason, invalid_cache=invalid_cache)
                break
            window_started = time.perf_counter()
            window_id = str(row["window_id"])
            previous = results.get(window_id, {})
            attempt_count = int(previous.get("attempt_count", 0)) + 1
            try:
                decoded = decode_video(VideoSource(sample_id=f"v7-fixed-grid-{row['window_id']}", source_video_id=str(row["source_video_id"]), source_locator=row["video_path"]), row["frame_indices"])
                depths, focals, frame_depth_valid = depth_runner.infer(decoded)
                intrinsics, fixed_focal_px = causal_first_frame_intrinsics(depths, focals)
                adjacent, pair_valid, information = rgbd_odometry(decoded, depths, intrinsics)
                world_from_camera, pose_valid = accumulate_world_from_camera(adjacent, pair_valid)
                tracker.grid_size = 17
                uv, visibility = tracker.track(decoded)
                sampled_depth, sampled_valid = sample_depth_at_uv(depths, uv)
                observation_valid = visibility & sampled_valid & frame_depth_valid[:, None]
                xyz, geometry_valid = world_xyz(uv, sampled_depth, intrinsics, world_from_camera, observation_valid, pose_valid)
                sequence = build_particle_sequence(
                    decoded, track_ids=persistent_track_ids(uv.shape[1]), xyz=xyz, uv=uv, visibility=visibility, geometry_validity=geometry_valid,
                    coordinate_system=CoordinateSystem(frame_name="first_camera_world", handedness=Handedness.RIGHT, axis_directions=("right", "down", "forward"), length_unit=LengthUnit.METER, camera_motion_compensated=True, normalization={}),
                    lineage={"dataset": "ActivityForensics+Charades", "official_split": "train", "source_id": row["source_id"], "pair_id": row["pair_id"], "role": row["role"], "grid_window_id": row["window_id"]},
                    provenance={"tracker": "online_bootstapir", "tracker_source_sha": TRACKER_SHA, "depth": "apple_depth_pro", "depth_source_sha": DEPTH_SHA, "depth_semantics": "optical_axis_z_depth", "pose": "open3d_rgbd_odometry", "pose_convention": "target_camera_from_source_camera_inverted", "process_size": 256, "query_grid_size": 17, "query_count": 289, "window_initialization": "independent", "causal_execution": True, "fixed_focal_px": fixed_focal_px},
                )
                prefix = root / "particles" / f"{_safe(str(row['window_id']))}__density289"
                existing_prefix = {prefix.with_suffix(".npz").is_file(), prefix.with_suffix(".json").is_file()}
                if existing_prefix != {False}:
                    prefix = root / "particles" / f"{_safe(str(row['window_id']))}__density289__attempt{attempt_count}"
                if not prefix.with_suffix(".npz").is_file() and not prefix.with_suffix(".json").is_file():
                    save_particle_sequence(sequence, prefix)
                diag = diagnostic.component_diagnostic(sequence, window_start_s=float(row["interval_start_s"]), label="density289")
                result = {**row, "status": "FRONTEND_COMPLETE", "attempt_count": attempt_count, "sequence_prefix": str(prefix), "result_key": f"{row['window_id']}::density289", "frame_indices": [int(x) for x in sequence.frame_indices], "timestamps_s": [float(x) for x in sequence.timestamps_s], "frame_sizes_hw": np.asarray(sequence.frame_sizes_hw).tolist(), "query_count": int(sequence.num_tracks), "tracking_visible_fraction": float(np.mean(visibility)), "geometry_valid_fraction": float(np.mean(geometry_valid)), "valid_points_per_frame": np.sum(geometry_valid, axis=1).astype(int).tolist(), "pose_pair_valid": np.asarray(pair_valid).tolist(), "pose_valid": np.asarray(pose_valid).tolist(), "support_status": str(diag["support_status"]), "valid_triplet_count": int(diag["valid_triplet_count"]), "component_count": int(diag["component_count"]), "max_component_members": int(diag["max_component_members"]), "selected_triplet_common_members": int(diag["selected_triplet_common_members"]), "causal_training_eligible": False, "causal_training_reason": "frozen frontend window observation; no target construction", "elapsed_s": time.perf_counter() - window_started}
                results[str(row["window_id"])] = result
                _save_results(root, results)
                valid_results[str(row["window_id"])] = result
                window_elapsed = float(result["elapsed_s"])
                _update_budget_state(root, budget_state, window_elapsed, clock=budget_clock, last_window=window_id, last_window_status="FRONTEND_COMPLETE")
                cumulative, process_elapsed = _budget_snapshot(budget_clock)
                _progress(root, "frontend", len(valid_results), len(rows), "RUNNING", planned_windows=len(rows), selected_windows=len(selected_rows), selected_source_prefix=source_prefix, attempted=number, succeeded=len(valid_results), failed=sum(item.get("status") == "FRONTEND_FAILED" for item in results.values()), last_window=window_id, elapsed_s=process_elapsed, consumed_s=cumulative, invalid_cache=invalid_cache)
                print(f"frontend {len(results)}/{len(rows)} source={row['source_id']} role={row['role']} grid={row['grid_index']} new {result['elapsed_s']:.1f}s", flush=True)
                del decoded, depths, uv, visibility, xyz, geometry_valid, sequence
                gc.collect()
                torch.cuda.empty_cache()
            except Exception as exc:
                window_elapsed = time.perf_counter() - window_started
                error = {**row, "result_key": f"{row['window_id']}::density289", "status": "FRONTEND_FAILED", "attempt_count": attempt_count, "error": f"{type(exc).__name__}: {exc}", "elapsed_s": window_elapsed}
                results[str(row["window_id"])] = error
                _save_results(root, results)
                _update_budget_state(root, budget_state, window_elapsed, clock=budget_clock, last_window=window_id, last_window_status="FRONTEND_FAILED")
                cumulative, process_elapsed = _budget_snapshot(budget_clock)
                _progress(root, "frontend", len(valid_results), len(rows), "RUNNING", planned_windows=len(rows), selected_windows=len(selected_rows), selected_source_prefix=source_prefix, attempted=number, succeeded=len(valid_results), failed=sum(item.get("status") == "FRONTEND_FAILED" for item in results.values()), last_error=error["error"], elapsed_s=process_elapsed, consumed_s=cumulative, invalid_cache=invalid_cache)
                print(f"frontend FAILED {row['window_id']}: {error['error']}", flush=True)
    finally:
        signal.signal(signal.SIGTERM, old_handler)
        signal.signal(signal.SIGINT, old_int)
        del tracker, depth_runner
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    valid_results, invalid_cache = _validated_results(root, rows)
    selected_complete = all(str(row["window_id"]) in valid_results for row in selected_rows)
    cumulative, process_elapsed = _budget_snapshot(budget_clock)
    if len(valid_results) == len(rows):
        status = "COMPLETE"
        stop_reason = stop_reason or "COMPLETE"
    elif stop_requested:
        status = "STOPPED_SAFE"
        stop_reason = stop_reason or "STOPPED_SAFE"
    elif cumulative >= float(budget_state.get("budget_s", budget_s)):
        status = "BUDGET_EXHAUSTED"
        stop_reason = stop_reason or "BUDGET_EXHAUSTED"
    elif selected_complete and max_windows is None:
        status = "SOURCE_PREFIX_COMPLETE"
        stop_reason = stop_reason or "SOURCE_PREFIX_COMPLETE"
    elif max_windows is not None:
        status = "BENCHMARK_PARTIAL"
        stop_reason = stop_reason or "BENCHMARK_PARTIAL"
    else:
        status = "FRONTEND_INCOMPLETE"
        stop_reason = stop_reason or "FRONTEND_INCOMPLETE"
    _persist_budget_state(root, budget_state, budget_clock, stop_reason=stop_reason, last_window_status=stop_reason)
    _progress(root, "frontend", len(valid_results), len(rows), status, planned_windows=len(rows), selected_windows=len(selected_rows), selected_source_prefix=source_prefix, elapsed_s=process_elapsed, consumed_s=cumulative, stop_reason=stop_reason, invalid_cache=invalid_cache, retry_limited=retry_limited)
    if max_windows is not None:
        benchmark_ids = [str(row["window_id"]) for row in selected_rows]
        _atomic_json(root / "evaluation" / "benchmark.json", {"source_id": source_prefix[0] if source_prefix else None, "window_ids": benchmark_ids, "completed": [window_id for window_id in benchmark_ids if window_id in valid_results], "note": "bounded frontend validation; no label/score selection"})
    return {"status": status, "completed": len(valid_results), "total": len(rows), "selected_windows": len(selected_rows), "selected_source_prefix": source_prefix, "invalid_cache": invalid_cache, "retry_limited": retry_limited, "budget_state": budget_state, "process_elapsed_s": process_elapsed, "stop_reason": stop_reason}


def _feature_path(root: Path, window_id: str) -> Path:
    return root / "features" / f"{_safe(window_id)}.json"


def build_features(root: Path, resume: bool = True) -> dict[str, Any]:
    """Build the existing H and first-frame segmentation B supports."""

    rows = _load_grid(root)
    results = _load_results(root)
    valid_results, invalid_cache = _validated_results(root, rows)
    usable = [row for row in rows if str(row["window_id"]) in valid_results]
    mask_dir = root / "masks"
    mask_dir.mkdir(parents=True, exist_ok=True)
    model = None
    mask_cache: dict[str, dict[str, Any]] = {}
    coverage: list[dict[str, Any]] = []
    for number, row in enumerate(usable, 1):
        path = _feature_path(root, str(row["window_id"]))
        if resume and path.is_file():
            feature = json.loads(path.read_text(encoding="utf-8"))
        else:
            result = valid_results[str(row["window_id"])]
            sequence = load_particle_sequence(Path(str(result["sequence_prefix"])))
            history = np.flatnonzero(np.asarray(sequence.timestamps_s) < float(row["interval_start_s"]) + 0.5).astype(np.int64)
            components = rebuild_components_fast(sequence.xyz, sequence.geometry_validity, history, COMPONENT_CONFIG)
            h_grouping = build_local_groups(sequence, history, old_components=components)
            h_support = build_local_support(sequence, window_start_s=float(row["interval_start_s"]), grouping=h_grouping)
            feature = {"identity": {"window_id": str(row["window_id"]), "source_id": str(row["source_id"]), "role": str(row["role"]), "grid_index": int(row["grid_index"]), "particle_prefix": str(result["sequence_prefix"]), "particle_count": int(sequence.num_tracks), "frame_indices": [int(x) for x in sequence.frame_indices], "timestamps_s": [float(x) for x in sequence.timestamps_s]}, "h_grouping": _compact_grouping(h_grouping), "h_support": _compact_support(h_support), "segmentation": {"status": "PENDING"}}
            video = Path(str(row["video_path"]))
            key = _mask_cache_key(str(video), int(sequence.frame_indices[0]))
            mask_path = mask_dir / f"{key}.json"
            if key not in mask_cache:
                if resume and mask_path.is_file():
                    mask_cache[key] = json.loads(mask_path.read_text(encoding="utf-8"))
                elif not video.is_file():
                    mask_cache[key] = {"status": "SOURCE_MISSING", "source_video": str(video), "frame_index": int(sequence.frame_indices[0]), "polygons": [], "areas": []}
                else:
                    if model is None:
                        from ultralytics import YOLO
                        model = YOLO(str(SEG_WEIGHT))
                    try:
                        mask_cache[key] = {"status": "COMPLETE", **_predict_masks(video, int(sequence.frame_indices[0]), model)}
                    except Exception as exc:
                        mask_cache[key] = {"status": "DECODE_FAILED", "source_video": str(video), "frame_index": int(sequence.frame_indices[0]), "polygons": [], "areas": [], "error": f"{type(exc).__name__}: {exc}"}
                _atomic_json(mask_path, mask_cache[key])
            record = mask_cache[key]
            masks = _polygons_to_masks(record) if record.get("status") == "COMPLETE" else []
            assignments = assign_uv_to_masks(np.asarray(sequence.uv)[0], np.asarray(sequence.visibility)[0], masks, record.get("areas", []))
            b_grouping = split_local_groups_by_assignment(h_grouping, assignments, minimum_size=int(COMPONENT_CONFIG.minimum_size))
            for child in b_grouping["groups"]:
                child["track_ids"] = [int(sequence.track_ids[int(slot)]) for slot in child["member_slots"]]
            checks = validate_boundary_partition(h_grouping, b_grouping)
            b_support = build_local_support(sequence, window_start_s=float(row["interval_start_s"]), grouping=b_grouping)
            feature["segmentation"] = {"status": str(record.get("status")), "cache_key": key, "cache_path": str(mask_path), "assignment_counts": {str(int(k)): int(v) for k, v in zip(*np.unique(assignments, return_counts=True))}}
            feature["b_grouping"] = _compact_grouping(b_grouping)
            feature["b_support"] = _compact_support(b_support)
            feature["boundary_checks"] = checks
            _atomic_json(path, feature)
        coverage.append({"window_id": row["window_id"], "source_id": row["source_id"], "role": row["role"], "grid_index": row["grid_index"], "h_support": feature.get("h_support", {}).get("support_status"), "b_support": feature.get("b_support", {}).get("support_status"), "h_triplets": feature.get("h_support", {}).get("valid_triplet_count", 0), "b_triplets": feature.get("b_support", {}).get("valid_triplet_count", 0), "segmentation_status": feature.get("segmentation", {}).get("status"), "frontend_status": valid_results[str(row["window_id"])].get("status", "FRONTEND_COMPLETE")})
        if number % 10 == 0:
            _progress(root, "features", number, len(usable), "RUNNING")
    _write_csv(root / "coverage/window_support.csv", coverage)
    source_rows: list[dict[str, Any]] = []
    for source in sorted({str(row["source_id"]) for row in rows}):
        source_all = [row for row in rows if str(row["source_id"]) == source]
        source_done = [row for row in usable if str(row["source_id"]) == source]
        source_feature_ids = {str(item["window_id"]) for item in source_done}
        no_support = sum(
            str(row["window_id"]) in source_feature_ids
            and json.loads(_feature_path(root, str(row["window_id"])).read_text(encoding="utf-8")).get("h_support", {}).get("support_status") != "VALID"
            and json.loads(_feature_path(root, str(row["window_id"])).read_text(encoding="utf-8")).get("b_support", {}).get("support_status") != "VALID"
            for row in source_all
        )
        failed = sum(str(row["window_id"]) in results and str(results[str(row["window_id"])].get("status")) == "FRONTEND_FAILED" for row in source_all)
        invalid = sum(str(row["window_id"]) in invalid_cache for row in source_all)
        not_run = len(source_all) - len(source_done) - failed - invalid
        source_rows.append({"source_id": source, "planned_videos": len({str(row["role"]) for row in source_all}), "planned_windows": len(source_all), "frontend_windows": len(source_done), "feature_windows": len(source_done), "no_valid_support_windows": no_support, "failed_windows": failed, "invalid_cache_windows": invalid, "not_run_windows": max(0, not_run), "complete_source": len(source_done) == len(source_all) and failed == 0 and invalid == 0 and not_run == 0, "partial_reason": "" if len(source_done) == len(source_all) and failed == 0 and invalid == 0 and not_run == 0 else "budget_or_frontend_failure"})
    _write_csv(root / "coverage/source_coverage.csv", source_rows)
    _atomic_json(root / "manifests/feature_summary.json", {"windows": len(coverage), "h_valid": sum(row["h_support"] == "VALID" for row in coverage), "b_valid": sum(row["b_support"] == "VALID" for row in coverage), "invalid_cache": invalid_cache, "segmentation_status": {status: sum(row["segmentation_status"] == status for row in coverage) for status in sorted({str(row["segmentation_status"]) for row in coverage})}})
    _progress(root, "features", len(coverage), len(usable), "COMPLETE", planned_windows=len(rows), usable_windows=len(usable), invalid_cache=invalid_cache)
    return {"windows": len(coverage), "usable": len(usable), "invalid_cache": invalid_cache, "source_coverage": source_rows}


def _load_models() -> tuple[dict[tuple[str, str, int], tuple[Any, Any]], dict[str, Any]]:
    import torch

    data = json.loads(MODEL_ROOT.read_text(encoding="utf-8"))
    models: dict[tuple[str, str, int], tuple[Any, Any]] = {}
    for record in data.get("records", []):
        condition = str(record["condition"])
        if condition not in CONDITIONS:
            continue
        model_record = record["model"]
        model = PoolingWindowMLP(str(model_record["pooling"]))
        state = {name: torch.as_tensor(value, dtype=model.state_dict()[name].dtype) for name, value in model_record["state_dict"].items()}
        model.load_state_dict(state)
        model.eval()
        std = model_record["standardization"]
        standardizer = WeightedStandardizer(np.asarray(std["mean"], dtype=np.float64), np.asarray(std["scale"], dtype=np.float64), tuple(int(x) for x in std.get("zero_variance_dimensions", [])))
        models[(condition, str(record["held_out_source"]), int(record["seed"]))] = (model, standardizer)
    return models, data


def _support_timestamps(feature: Mapping[str, Any], condition: str) -> list[float]:
    organization, _ = ARM_BY_CONDITION[condition]
    support = feature.get("h_support" if organization == "H" else "b_support", {})
    values: set[float] = set()
    for triplet in support.get("triplets", []):
        for stamp in triplet.get("timestamps_s", []):
            value = float(stamp)
            if np.isfinite(value):
                values.add(value)
    return sorted(values)


def _missing_model_seeds(models: Mapping[tuple[str, str, int], Any], condition: str, source: str, seeds: Sequence[int]) -> list[int]:
    return [int(seed) for seed in seeds if (condition, str(source), int(seed)) not in models]


def _model_used_frame_rows(root: Path, rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Materialize actual triplet-supported source frame/PTS identities."""

    output: list[dict[str, Any]] = []
    for row in rows:
        path = _feature_path(root, str(row["window_id"]))
        if not path.is_file():
            continue
        feature = json.loads(path.read_text(encoding="utf-8"))
        identity = feature.get("identity", {})
        frame_indices = [int(value) for value in identity.get("frame_indices", [])]
        timestamps = np.asarray(identity.get("timestamps_s", []), dtype=np.float64)
        for condition in CONDITIONS:
            for stamp in _support_timestamps(feature, condition):
                if timestamps.size == 0:
                    output.append({"window_id": row["window_id"], "source_id": row["source_id"], "condition": condition, "frame_index": "UNKNOWN", "timestamp_s": stamp, "status": "UNKNOWN_IDENTITY"})
                    continue
                index = int(np.argmin(np.abs(timestamps - stamp)))
                if abs(float(timestamps[index]) - stamp) > 1e-8:
                    output.append({"window_id": row["window_id"], "source_id": row["source_id"], "condition": condition, "frame_index": "UNKNOWN", "timestamp_s": stamp, "status": "PTS_NOT_IN_SEQUENCE"})
                else:
                    output.append({"window_id": row["window_id"], "source_id": row["source_id"], "condition": condition, "frame_index": frame_indices[index], "timestamp_s": float(timestamps[index]), "status": "MODEL_USED"})
    return output


def score_windows(root: Path, resume: bool = True) -> dict[str, Any]:
    rows = _load_grid(root)
    results, invalid_cache = _validated_results(root, rows)
    models, model_data = _load_models()
    score_rows: list[dict[str, Any]] = []
    for row in rows:
        window_id = str(row["window_id"])
        result = results.get(window_id, {})
        feature_path = _feature_path(root, window_id)
        base: dict[str, Any] = {"window_id": window_id, "source_id": str(row["source_id"]), "source_video_id": str(row.get("source_video_id", "")), "role": str(row["role"]), "grid_index": int(row["grid_index"]), "interval_start_s": row.get("interval_start_s"), "interval_end_s": row.get("interval_end_s"), "annotation_category": row.get("annotation_category"), "annotation_overlap_s": row.get("annotation_overlap_s"), "annotation_overlap_fraction": row.get("annotation_overlap_fraction"), "frontend_status": result.get("status", "FRONTEND_COMPLETE" if result.get("sequence_prefix") else "MISSING"), "frontend_frame_count": len(result.get("frame_indices", [])), "model_used_pts_count": 0, "model_used_frame_indices": ""}
        if not result:
            base["score_status"] = "MISSING_FRONTEND"
            score_rows.append(base)
            continue
        if not feature_path.is_file():
            base["score_status"] = "MISSING_FEATURE"
            score_rows.append(base)
            continue
        feature = json.loads(feature_path.read_text(encoding="utf-8"))
        for condition, (organization, arm) in ARM_BY_CONDITION.items():
            support = feature.get("h_support" if organization == "H" else "b_support", {})
            triplets = support.get("triplets", [])
            support_stamps = _support_timestamps(feature, condition)
            base[f"{condition}_model_used_pts_count"] = len(support_stamps)
            base[f"{condition}_model_used_timestamps_s"] = json.dumps(support_stamps, separators=(",", ":"))
            if not triplets:
                base[f"{condition}_status"] = "NO_VALID_TRIPLET"
                continue
            example = {"window_id": window_id, "source_id": str(row["source_id"]), "role": str(row["role"]), "kind": "MANIP", "triplets": triplets}
            seed_values: list[float] = []
            for seed in model_data.get("seeds", []):
                item = models.get((condition, str(row["source_id"]), int(seed)))
                if item is None:
                    continue
                model, standardizer = item
                batch, _ = build_batch([example], arm, standardizer=standardizer)
                seed_values.append(float(score_pooling_model(model, batch)[0]))
                base[f"{condition}_seed_{seed}"] = seed_values[-1]
            expected_seeds = [int(seed) for seed in model_data.get("seeds", [])]
            missing_seeds = _missing_model_seeds(models, condition, str(row["source_id"]), expected_seeds)
            base[f"{condition}_expected_seed_count"] = len(expected_seeds)
            base[f"{condition}_seed_count"] = len(seed_values)
            base[f"{condition}_missing_seeds"] = json.dumps(missing_seeds, separators=(",", ":"))
            if seed_values and not missing_seeds and len(seed_values) == len(expected_seeds):
                base[condition] = float(np.mean(seed_values))
                base[f"{condition}_status"] = "SCORED"
            elif missing_seeds:
                base[f"{condition}_status"] = "MISSING_SEED_MODEL"
            else:
                base[f"{condition}_status"] = "MISSING_HELDOUT_MODEL"
        base["score_status"] = "SCORED" if any(base.get(f"{condition}_status") == "SCORED" for condition in CONDITIONS) else "NO_MODEL_SCORE"
        score_rows.append(base)
    _write_csv(root / "scores/window_scores.csv", score_rows)
    _write_csv(root / "evaluation/model_used_frames.csv", _model_used_frame_rows(root, rows))
    unique_score_rows, duplicate_score_ids = _unique_window_rows(score_rows)
    scored_by_condition = {condition: sum(_condition_scored(row, condition) for row in unique_score_rows) for condition in CONDITIONS}
    _progress(root, "score", sum(scored_by_condition.values()), len(unique_score_rows), "COMPLETE", processed_rows=len(score_rows), unique_window_count=len(unique_score_rows), scored_by_condition=scored_by_condition, duplicate_window_ids=duplicate_score_ids, invalid_cache=invalid_cache)
    return {"processed_windows": len(score_rows), "models": len(models), "scored_by_condition": scored_by_condition, "invalid_cache": invalid_cache}


def _auroc(labels: Sequence[int], scores: Sequence[float]) -> float | None:
    labels_array = np.asarray(labels, dtype=np.int64); values = np.asarray(scores, dtype=np.float64)
    positives = values[labels_array == 1]; negatives = values[labels_array == 0]
    if not positives.size or not negatives.size: return None
    return float(np.mean((positives[:, None] > negatives[None, :]).astype(float) + 0.5 * (positives[:, None] == negatives[None, :])))


def _average_precision(labels: Sequence[int], scores: Sequence[float]) -> float | None:
    y = np.asarray(labels, dtype=np.int64); s = np.asarray(scores, dtype=np.float64)
    if not np.any(y == 1) or not np.any(y == 0): return None
    order = np.argsort(-s, kind="mergesort"); yy = y[order]; tp = np.cumsum(yy == 1); precision = tp / np.arange(1, len(yy) + 1); return float(np.sum(precision[yy == 1]) / np.sum(yy == 1))


def _classification(labels: Sequence[int], scores: Sequence[float]) -> dict[str, Any]:
    y = np.asarray(labels, dtype=np.int64); pred = np.asarray(scores) >= 0.0
    tn = int(np.sum((y == 0) & ~pred)); fp = int(np.sum((y == 0) & pred)); fn = int(np.sum((y == 1) & ~pred)); tp = int(np.sum((y == 1) & pred)); precision = tp / (tp + fp) if tp + fp else None; recall = tp / (tp + fn) if tp + fn else None; f1 = 2 * precision * recall / (precision + recall) if precision is not None and recall is not None and precision + recall else None
    return {"precision": precision, "recall": recall, "f1": f1, "accuracy": (tn + tp) / len(y) if len(y) else None, "tn": tn, "fp": fp, "fn": fn, "tp": tp}


def _condition_scored(row: Mapping[str, Any], condition: str) -> bool:
    if row.get(f"{condition}_status") != "SCORED" or row.get(condition, "") in (None, ""):
        return False
    try:
        return bool(np.isfinite(float(row[condition])))
    except (TypeError, ValueError):
        return False


def _unique_window_rows(rows: Sequence[Mapping[str, Any]]) -> tuple[list[dict[str, Any]], list[str]]:
    """Keep one deterministic score row per frozen ``window_id``."""

    unique: dict[str, dict[str, Any]] = {}
    duplicate_ids: list[str] = []
    for row in rows:
        window_id = str(row.get("window_id", ""))
        if window_id in unique:
            if window_id not in duplicate_ids:
                duplicate_ids.append(window_id)
            continue
        unique[window_id] = dict(row)
    return list(unique.values()), sorted(duplicate_ids)


def _score_window_sets(rows: Sequence[Mapping[str, Any]], complete_sources: set[str]) -> dict[str, Any]:
    """Compute scoring populations from unique window identities only."""

    unique_rows, duplicate_ids = _unique_window_rows(rows)
    condition_ids = {
        condition: {str(row["window_id"]) for row in unique_rows if _condition_scored(row, condition)}
        for condition in CONDITIONS
    }
    raw_intersection = set.intersection(*(condition_ids[condition] for condition in CONDITIONS)) if CONDITIONS else set()
    complete_intersection = {
        str(row["window_id"])
        for row in unique_rows
        if str(row.get("window_id")) in raw_intersection and str(row.get("source_id")) in complete_sources
    }
    main_intersection = {
        str(row["window_id"])
        for row in unique_rows
        if str(row.get("window_id")) in complete_intersection
        and (
            row.get("role") == "real"
            or (row.get("role") == "fake" and row.get("annotation_category") == "FAKE_MANIPULATION")
        )
    }
    return {
        "unique_rows": unique_rows,
        "duplicate_window_ids": duplicate_ids,
        "condition_ids": condition_ids,
        "raw_three_condition_ids": raw_intersection,
        "complete_source_three_condition_ids": complete_intersection,
        "main_label_filtered_ids": main_intersection,
    }


def _unscored_reason_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Summarize missing scores without turning missing into numeric values."""

    unique_rows, _ = _unique_window_rows(rows)
    counts: Counter[tuple[str, str]] = Counter()
    for condition in CONDITIONS:
        status_name = f"{condition}_status"
        for row in unique_rows:
            if _condition_scored(row, condition):
                continue
            reason = str(row.get(status_name) or row.get("score_status") or row.get("frontend_status") or "UNKNOWN")
            counts[(condition, reason)] += 1
    return [{"condition": condition, "reason": reason, "count": count} for (condition, reason), count in sorted(counts.items())]


def _write_source_coverage_summary(
    root: Path,
    rows: Sequence[Mapping[str, Any]],
    complete_sources: set[str],
    window_sets: Mapping[str, Any],
) -> None:
    """Write source-level processing and scoring coverage in one small table."""

    coverage_path = root / "coverage" / "source_coverage.csv"
    base_rows = {
        str(row["source_id"]): dict(row)
        for row in csv.DictReader(coverage_path.open(newline="", encoding="utf-8"))
    } if coverage_path.is_file() else {}
    raw_ids = set(window_sets["raw_three_condition_ids"])
    complete_ids = set(window_sets["complete_source_three_condition_ids"])
    main_ids = set(window_sets["main_label_filtered_ids"])
    source_ids = sorted({str(row.get("source_id")) for row in rows} | set(base_rows))
    output: list[dict[str, Any]] = []
    for source in source_ids:
        source_rows = [row for row in rows if str(row.get("source_id")) == source]
        row: dict[str, Any] = {
            "source_id": source,
            "complete_source": source in complete_sources,
            "planned_windows": base_rows.get(source, {}).get("planned_windows", len(source_rows)),
            "frontend_windows": base_rows.get(source, {}).get("frontend_windows", ""),
            "feature_windows": base_rows.get(source, {}).get("feature_windows", ""),
            "no_valid_support_windows": base_rows.get(source, {}).get("no_valid_support_windows", ""),
            "failed_windows": base_rows.get(source, {}).get("failed_windows", ""),
            "not_run_windows": base_rows.get(source, {}).get("not_run_windows", ""),
            "raw_three_condition_windows": sum(str(item.get("window_id")) in raw_ids for item in source_rows),
            "complete_source_three_condition_windows": sum(str(item.get("window_id")) in complete_ids for item in source_rows),
            "main_label_filtered_windows": sum(str(item.get("window_id")) in main_ids for item in source_rows),
            "main_real_windows": sum(str(item.get("window_id")) in main_ids and item.get("role") == "real" for item in source_rows),
            "main_fake_windows": sum(str(item.get("window_id")) in main_ids and item.get("role") == "fake" for item in source_rows),
        }
        for condition in CONDITIONS:
            row[f"{condition}_scored_windows"] = sum(_condition_scored(item, condition) for item in source_rows)
        output.append(row)
    _write_csv(root / "evaluation/source_coverage_summary.csv", output)


def _write_time_curve_outputs(root: Path, rows: Sequence[Mapping[str, Any]], complete_sources: set[str]) -> None:
    """Write dependency-free SVG curves for complete sources; missing scores break lines."""

    unique_rows, _ = _unique_window_rows(rows)
    output_dir = root / "evaluation" / "time_curves"
    output_dir.mkdir(parents=True, exist_ok=True)
    index_rows: list[dict[str, Any]] = []
    colors = {
        ("real", "H_MEAN_A"): "#2563eb", ("fake", "H_MEAN_A"): "#ef4444",
        ("real", "B_MEAN_A"): "#16a34a", ("fake", "B_MEAN_A"): "#f97316",
        ("real", "B_MEAN_C"): "#7c3aed", ("fake", "B_MEAN_C"): "#a16207",
    }
    for source in sorted(complete_sources):
        source_rows = [row for row in unique_rows if str(row.get("source_id")) == source and row.get("role") in {"real", "fake"}]
        finite_values = [
            float(row[condition])
            for row in source_rows
            for condition in CONDITIONS
            if _condition_scored(row, condition)
        ]
        if finite_values:
            y_min, y_max = min(finite_values), max(finite_values)
            if y_max <= y_min:
                y_min, y_max = y_min - 1.0, y_max + 1.0
            margin = max(0.1, (y_max - y_min) * 0.08)
            y_min, y_max = y_min - margin, y_max + margin
        else:
            y_min, y_max = -1.0, 1.0
        x_values = []
        for row in source_rows:
            try:
                x_values.append(float(row["interval_start_s"]))
            except (KeyError, TypeError, ValueError):
                pass
        x_min = min(x_values) if x_values else 0.0
        x_max = max(x_values) if x_values else 1.0
        if x_max <= x_min:
            x_max = x_min + 1.0
        left, right, top, bottom = 80.0, 1060.0, 55.0, 560.0

        def px(value: float) -> float:
            return left + (value - x_min) / (x_max - x_min) * (right - left)

        def py(value: float) -> float:
            return bottom - (value - y_min) / (y_max - y_min) * (bottom - top)

        elements = [
            '<rect width="1120" height="640" fill="white"/>',
            f'<text x="80" y="25" font-family="sans-serif" font-size="16">source {escape(source)} frozen model logits (missing scores are gaps)</text>',
            f'<line x1="{left}" y1="{bottom}" x2="{right}" y2="{bottom}" stroke="#334155"/>',
            f'<line x1="{left}" y1="{top}" x2="{left}" y2="{bottom}" stroke="#334155"/>',
            f'<text x="{right - 90}" y="610" font-family="sans-serif" font-size="12">video time (s)</text>',
            f'<text x="12" y="{top + 20}" transform="rotate(-90 12 {top + 20})" font-family="sans-serif" font-size="12">logit</text>',
        ]
        # Dataset annotation windows are a separate background layer.
        annotation_intervals = []
        for row in source_rows:
            if row.get("role") != "fake" or row.get("annotation_category") != "FAKE_MANIPULATION":
                continue
            try:
                annotation_intervals.append((float(row["interval_start_s"]), float(row["interval_end_s"])))
            except (KeyError, TypeError, ValueError):
                continue
        for start, end in annotation_intervals:
            elements.append(f'<rect x="{px(start):.2f}" y="{top}" width="{max(1.0, px(end) - px(start)):.2f}" height="{bottom - top}" fill="#facc15" opacity="0.15"/>')
        for role in ("real", "fake"):
            role_rows = sorted((row for row in source_rows if row.get("role") == role), key=lambda item: (float(item.get("interval_start_s", 0.0)), int(item.get("grid_index", 0))))
            for condition in CONDITIONS:
                segments: list[list[tuple[float, float]]] = []
                segment: list[tuple[float, float]] = []
                for row in role_rows:
                    if not _condition_scored(row, condition):
                        if segment:
                            segments.append(segment); segment = []
                        continue
                    try:
                        point = (float(row["interval_start_s"]), float(row[condition]))
                    except (KeyError, TypeError, ValueError):
                        if segment:
                            segments.append(segment); segment = []
                        continue
                    segment.append(point)
                if segment:
                    segments.append(segment)
                color = colors[(role, condition)]
                for points in segments:
                    if len(points) >= 2:
                        path = " ".join(f"{px(x):.2f},{py(y):.2f}" for x, y in points)
                        elements.append(f'<polyline points="{path}" fill="none" stroke="{color}" stroke-width="2"/>')
                    for x, y in points:
                        elements.append(f'<circle cx="{px(x):.2f}" cy="{py(y):.2f}" r="3" fill="{color}"/>')
        legend_x, legend_y = 80.0, 595.0
        for index, ((role, condition), color) in enumerate(colors.items()):
            x = legend_x + (index % 3) * 300
            y = legend_y + (index // 3) * 18
            elements.append(f'<line x1="{x:.2f}" y1="{y - 4:.2f}" x2="{x + 18:.2f}" y2="{y - 4:.2f}" stroke="{color}" stroke-width="3"/>')
            elements.append(f'<text x="{x + 24:.2f}" y="{y:.2f}" font-family="sans-serif" font-size="12">{role} {condition}</text>')
        elements.append(f'<text x="{left + 8:.2f}" y="{top + 16:.2f}" font-family="sans-serif" font-size="11" fill="#a16207">yellow = dataset annotation window, not prediction</text>')
        svg_path = output_dir / f"source_{_safe(source)}.svg"
        svg_path.write_text('<svg xmlns="http://www.w3.org/2000/svg" width="1120" height="640">' + "".join(elements) + "</svg>\n", encoding="utf-8")
        index_rows.append({"source_id": source, "path": str(svg_path), "window_count": len(source_rows), "note": "real/fake curves; missing score rows are disconnected"})
    _write_csv(root / "evaluation/time_curve_index.csv", index_rows)


def evaluate(root: Path) -> dict[str, Any]:
    with (root / "scores/window_scores.csv").open(newline="", encoding="utf-8") as handle:
        rows, duplicate_window_ids = _unique_window_rows(list(csv.DictReader(handle)))
    coverage_path = root / "coverage" / "source_coverage.csv"
    coverage = {str(row["source_id"]): row for row in csv.DictReader(coverage_path.open(newline="", encoding="utf-8"))} if coverage_path.is_file() else {}
    complete_sources = {source for source, row in coverage.items() if str(row.get("complete_source", "")).lower() == "true"}
    window_sets = _score_window_sets(rows, complete_sources)
    per_source: list[dict[str, Any]] = []
    summary: dict[str, Any] = {
        "conditions": {},
        "protocol": {"positive": "fake windows fully inside annotation union", "negative": "all real windows", "boundary_and_outside": "descriptive only", "main_comparison": "complete sources and common windows across all conditions", "threshold": "logit >= 0", "bootstrap": {"unit": "source", "seed": SEED, "replicates": 10000}, "budget_scope": BUDGET_SCOPE},
        "complete_sources": sorted(complete_sources),
        "partial_sources": sorted(set(coverage) - complete_sources),
        "duplicate_window_ids": duplicate_window_ids,
    }
    source_ids = sorted({str(row["source_id"]) for row in rows})
    main_ids = window_sets["main_label_filtered_ids"]
    common_rows = [row for row in rows if str(row.get("window_id")) in main_ids]

    def ci(values: Sequence[float]) -> list[float] | None:
        if not values:
            return None
        array = np.asarray(values, dtype=np.float64)
        rng = np.random.default_rng(SEED)
        samples = np.asarray([np.mean(array[rng.integers(0, len(array), len(array))]) for _ in range(10000)], dtype=np.float64)
        return [float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5))]

    source_scores: dict[str, dict[str, float]] = {condition: {} for condition in CONDITIONS}
    for condition in CONDITIONS:
        source_values: dict[str, float] = {}; pooled_labels: list[int] = []; pooled_scores: list[float] = []
        for source in source_ids:
            selected = []
            for row in common_rows:
                if str(row["source_id"]) != source:
                    continue
                if row["role"] == "real": label = 0
                elif row.get("annotation_category") == "FAKE_MANIPULATION": label = 1
                else: continue
                selected.append((label, float(row[condition])))
            if selected and {label for label, _ in selected} == {0, 1}:
                value = _auroc([x[0] for x in selected], [x[1] for x in selected]); source_values[source] = value; source_scores[condition][source] = float(value)
                pooled_labels.extend(x[0] for x in selected); pooled_scores.extend(x[1] for x in selected)
            per_source.append({"condition": condition, "source_id": source, "source_complete": source in complete_sources, "common_window_count": len(selected), "real_count": sum(x[0] == 0 for x in selected), "fake_count": sum(x[0] == 1 for x in selected), "auroc": value if selected and {x[0] for x in selected} == {0, 1} else None, "eligibility": "ELIGIBLE" if source in source_values else ("PARTIAL_SOURCE" if source not in complete_sources else "NO_BOTH_CLASSES")})
        values = list(source_values.values()); mean = float(np.mean(values)) if values else None
        classification = _classification(pooled_labels, pooled_scores) if pooled_labels else {"precision": None, "recall": None, "f1": None, "accuracy": None, "tn": None, "fp": None, "fn": None, "tp": None}
        summary["conditions"][condition] = {"source_count": len(values), "source_mean_auroc": mean, "bootstrap_ci95": ci(values), "pooled_window_count": len(pooled_labels), "pooled_auroc": _auroc(pooled_labels, pooled_scores), "pooled_ap": _average_precision(pooled_labels, pooled_scores), **classification}
    comparisons: dict[str, Any] = {}
    for left, right, name in (("B_MEAN_A", "H_MEAN_A", "B_MEAN_A_minus_H_MEAN_A"), ("B_MEAN_C", "B_MEAN_A", "B_MEAN_C_minus_B_MEAN_A")):
        shared = sorted(set(source_scores[left]) & set(source_scores[right]))
        differences = [source_scores[left][source] - source_scores[right][source] for source in shared]
        comparisons[name] = {"source_count": len(differences), "sources": shared, "source_mean_difference": float(np.mean(differences)) if differences else None, "bootstrap_ci95": ci(differences)}
    summary["paired_comparisons"] = comparisons
    summary["window_sets"] = {
        "unique_window_count": len(rows),
        "condition_scored_windows": {condition: len(window_sets["condition_ids"][condition]) for condition in CONDITIONS},
        "raw_three_condition_intersection": len(window_sets["raw_three_condition_ids"]),
        "complete_source_three_condition_intersection": len(window_sets["complete_source_three_condition_ids"]),
        "main_label_filtered_intersection": len(window_sets["main_label_filtered_ids"]),
        "duplicate_window_count": len(duplicate_window_ids),
    }
    summary["unscored_reasons"] = _unscored_reason_rows(rows)
    summary["common_window_count"] = len(common_rows)
    _write_csv(root / "evaluation/per_source_metrics.csv", per_source)
    _write_csv(root / "evaluation/unscored_reasons.csv", summary["unscored_reasons"])
    _write_csv(root / "evaluation/window_set_counts.csv", [
        {"set_name": "unique_window_count", "count": len(rows)},
        *({"set_name": f"condition_scored:{condition}", "count": len(window_sets["condition_ids"][condition])} for condition in CONDITIONS),
        {"set_name": "raw_three_condition_intersection", "count": len(window_sets["raw_three_condition_ids"])},
        {"set_name": "complete_source_three_condition_intersection", "count": len(window_sets["complete_source_three_condition_ids"])},
        {"set_name": "main_label_filtered_intersection", "count": len(window_sets["main_label_filtered_ids"])},
    ])
    _write_source_coverage_summary(root, rows, complete_sources, window_sets)
    _write_time_curve_outputs(root, rows, complete_sources)
    _atomic_json(root / "evaluation/summary.json", summary)
    _progress(root, "evaluate", len(complete_sources), len(source_ids), "COMPLETE", common_windows=len(common_rows), complete_source_count=len(complete_sources), source_count=len(source_ids), condition_source_cells=sum(int(value["source_count"]) for value in summary["conditions"].values()), window_sets=summary["window_sets"], complete_sources=sorted(complete_sources), partial_sources=sorted(set(source_ids) - complete_sources))
    return summary


def build_review(root: Path) -> None:
    rows = list(csv.DictReader((root / "scores/window_scores.csv").open(newline="", encoding="utf-8")))
    payload = json.dumps(rows, ensure_ascii=False)
    html = """<!doctype html><meta charset='utf-8'><title>V7 fixed-grid frozen probe</title><style>body{font:14px sans-serif;background:#101826;color:#eee;margin:2em}table{border-collapse:collapse}td,th{border:1px solid #475569;padding:4px}tr:hover{background:#26354d}.missing{color:#fbbf24}</style><h1>V7 固定时间网格观测与冻结模型评分</h1><p>窗口标签是数据集时间标注；分数是冻结模型 logit，不是概率或空间真值。窗口之间不共享 track identity。</p><input id='q' placeholder='source/window filter'><table id='t'><thead><tr><th>window</th><th>role</th><th>annotation</th><th>H_MEAN_A</th><th>B_MEAN_A</th><th>B_MEAN_C</th><th>score status</th></tr></thead><tbody></tbody></table><script>const rows=__ROWS__;const body=document.querySelector('tbody');function draw(){const q=document.querySelector('#q').value.toLowerCase();body.innerHTML=rows.filter(r=>(r.window_id+' '+r.source_id).toLowerCase().includes(q)).map(r=>'<tr><td>'+r.window_id+'</td><td>'+r.role+'</td><td>'+r.annotation_category+'</td><td>'+fmt(r.H_MEAN_A)+'</td><td>'+fmt(r.B_MEAN_A)+'</td><td>'+fmt(r.B_MEAN_C)+'</td><td>'+r.score_status+'</td></tr>').join('')}function fmt(x){return x==null||x===''?'<span class="missing">—</span>':Number(x).toFixed(5)}document.querySelector('#q').oninput=draw;draw()</script>""".replace("__ROWS__", payload)
    (root / "review/index.html").parent.mkdir(parents=True, exist_ok=True); (root / "review/index.html").write_text(html, encoding="utf-8")


def _actual_inventory(root: Path) -> dict[str, Any]:
    rows = _load_grid(root)
    results = _load_results(root)
    valid_results, invalid_cache = _validated_results(root, rows)
    score_path = root / "scores/window_scores.csv"
    scores_raw = list(csv.DictReader(score_path.open(newline="", encoding="utf-8"))) if score_path.is_file() else []
    scores, duplicate_score_ids = _unique_window_rows(scores_raw)
    source_rows = list(csv.DictReader((root / "coverage/source_coverage.csv").open(newline="", encoding="utf-8"))) if (root / "coverage/source_coverage.csv").is_file() else []
    feature_summary_path = root / "manifests/feature_summary.json"
    feature_windows = int(json.loads(feature_summary_path.read_text(encoding="utf-8")).get("windows", 0)) if feature_summary_path.is_file() else 0
    success = list(valid_results.values())
    statuses = _frontend_status_counts(rows, results, set(valid_results))
    scored = {condition: sum(_condition_scored(row, condition) for row in scores) for condition in CONDITIONS}
    frontend_frames = sum(len(item.get("frame_indices", [])) for item in success)
    model_rows = list(csv.DictReader((root / "evaluation/model_used_frames.csv").open(newline="", encoding="utf-8"))) if (root / "evaluation/model_used_frames.csv").is_file() else []
    model_counts = {condition: len({(str(item.get("source_id")), str(item.get("window_id")), str(item.get("frame_index")), str(item.get("timestamp_s"))) for item in model_rows if item.get("condition") == condition and item.get("status") == "MODEL_USED"}) for condition in CONDITIONS}
    categories = defaultdict(int)
    for row in rows:
        categories[str(row.get("annotation_category", "UNLABELED"))] += 1
    summary_path = root / "evaluation/summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.is_file() else {}
    return {"planned_windows": len(rows), "planned_videos": len({str(row["source_video_id"]) for row in rows}), "frontend_results": len(success), "frontend_processed_frames": frontend_frames, "feature_windows": feature_windows, "window_statuses": dict(statuses), "scored_windows": scored, "duplicate_score_window_ids": duplicate_score_ids, "window_sets": summary.get("window_sets", {}), "unscored_reasons": summary.get("unscored_reasons", []), "model_used_frames_by_condition": model_counts, "model_used_frame_rows": len(model_rows), "annotation_categories": dict(categories), "invalid_cache": invalid_cache, "complete_sources": [row["source_id"] for row in source_rows if str(row.get("complete_source")).lower() == "true"], "partial_sources": [row["source_id"] for row in source_rows if int(row.get("frontend_windows", 0)) > 0 and str(row.get("complete_source")).lower() != "true"], "unrun_sources": [row["source_id"] for row in source_rows if int(row.get("frontend_windows", 0)) == 0]}


def write_report(root: Path, frontend: Mapping[str, Any], feature: Mapping[str, Any], summary: Mapping[str, Any]) -> None:
    inventory = _actual_inventory(root)
    lines = ["# V7 固定时间网格观测与冻结模型评分 pilot", "", f"- Git HEAD: `{_git_head()}`", f"- 计划：{inventory['planned_videos']} 个独立视频、{inventory['planned_windows']} 个固定网格窗口；该计数表示固定时间网格窗口总数，不是视频数、模型条件数或评分完成数。", f"- 前端完成：{inventory['frontend_results']} 个窗口；前端处理帧数：{inventory['frontend_processed_frames']}；H/B 特征：{inventory['feature_windows']} 个窗口；运行状态：`{frontend.get('status')}`。", f"- 窗口状态：{inventory['window_statuses']}；各条件有效评分：{inventory['scored_windows']}；评分循环的缺失行不计为评分。", f"- 模型实际使用帧（按 source/video/window/frame/PTS 去重）：{inventory['model_used_frames_by_condition']}；记录行数：{inventory['model_used_frame_rows']}。前端处理时刻不自动等同模型使用时刻。", f"- source 状态：COMPLETE={inventory['complete_sources']}；PARTIAL={inventory['partial_sources']}；未运行={inventory['unrun_sources']}。", f"- 缓存身份问题：{inventory['invalid_cache']}。", "- 时间网格先于标签冻结；每个窗口独立初始化 17×17=289 查询点。", "- real 窗口作负类；fake 仅在一秒窗口完整包含于标注区间并集时作主正类；BOUNDARY_MIXED/OUTSIDE 仅描述。", "- 冻结条件：H_MEAN_A、B_MEAN_A、B_MEAN_C；仅使用 held-out source 的旧 fold 模型与原标准化。", "", "## 条件摘要", ""]
    for condition, value in summary.get("conditions", {}).items():
        lines.append(f"- `{condition}`：source mean AUROC={value.get('source_mean_auroc')}, CI={value.get('bootstrap_ci95')}, pooled AUROC={value.get('pooled_auroc')}, AP={value.get('pooled_ap')}，source 数={value.get('source_count')}。")
    lines += ["", "## 覆盖与可判定性", "", f"固定网格标签类别计数：{inventory['annotation_categories']}。只有完整处理 source 且三条件共同有有效分数的窗口才进入主评价；部分 source、NO_VALID_TRIPLET、缺少 seed 模型、前端失败和未运行窗口保留为描述，不填 0。", "", "## 限制", "", "本 pilot 不是 sealed-test、不是新模型训练，也没有空间真值。模型实际使用帧从 H/B triplet 支撑恢复；若无法从支撑映射到序列 PTS，记录 UNKNOWN。窗口有分数不等于失真部位有观测；缺失不自动解释为伪造。"]
    (root / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_report_v2(root: Path, frontend: Mapping[str, Any], feature: Mapping[str, Any], summary: Mapping[str, Any]) -> None:
    """Write the post-recovery report with explicit budget and score populations."""

    inventory = _actual_inventory(root)
    budget_path = root / "manifests" / BUDGET_STATE_NAME
    budget = json.loads(budget_path.read_text(encoding="utf-8")) if budget_path.is_file() else {}
    lines = [
        "# V7 固定时间网格观测与冻结模型评分 pilot",
        "",
        f"- Git HEAD: `{_git_head()}`",
        f"- 计划：{inventory['planned_videos']} 个独立视频、{inventory['planned_windows']} 个固定网格窗口；该计数是窗口总数，不是视频数或评分完成数。",
        f"- 前端完成：{inventory['frontend_results']}；前端处理帧数：{inventory['frontend_processed_frames']}；H/B 特征：{inventory['feature_windows']}；前端状态：`{frontend.get('status')}`。",
        f"- source：COMPLETE={inventory['complete_sources']}；PARTIAL={inventory['partial_sources']}；未运行={inventory['unrun_sources']}。",
        f"- 预算范围：`{budget.get('budget_scope', BUDGET_SCOPE)}`；累计前端预算={budget.get('consumed_s')} / {budget.get('budget_s')} s；本次进程前累计={budget.get('elapsed_before_this_process_s')} s；停止原因=`{budget.get('last_stop_reason', frontend.get('stop_reason'))}`。",
        f"- 旧 bounded benchmark 耗时 `{budget.get('prior_benchmark_elapsed_s')}` s 不计入正式 7200 s；后处理不参与前端停止判断。",
        f"- 缓存身份问题：{inventory['invalid_cache']}；重复 score window_id：{inventory.get('duplicate_score_window_ids', [])}。",
        "- 时间网格先于标签冻结；每个窗口独立初始化 17×17=289 查询点。",
        "- real 窗口作负类；fake 仅在一秒窗口完整包含于标注区间并集时作主正类；BOUNDARY_MIXED/OUTSIDE 仅描述。",
        "- 冻结条件：H_MEAN_A、B_MEAN_A、B_MEAN_C；仅使用 held-out source 的旧 fold 模型与原标准化。",
        "",
        "## 评分集合口径",
        "",
    ]
    for name, count in summary.get("window_sets", {}).items():
        lines.append(f"- `{name}`：{count}")
    lines += ["", "## 条件摘要", ""]
    for condition, value in summary.get("conditions", {}).items():
        real_count = (value.get("tn") or 0) + (value.get("fp") or 0) if value.get("tn") is not None else None
        fake_count = (value.get("fn") or 0) + (value.get("tp") or 0) if value.get("fn") is not None else None
        lines.append(f"- `{condition}`：source mean AUROC={value.get('source_mean_auroc')}, CI={value.get('bootstrap_ci95')}, pooled AUROC={value.get('pooled_auroc')}, AP={value.get('pooled_ap')}，pooled n={value.get('pooled_window_count')}，real/fake={real_count}/{fake_count}，source 数={value.get('source_count')}。")
    for name, value in summary.get("paired_comparisons", {}).items():
        lines.append(f"- 配对 `{name}`：source mean difference={value.get('source_mean_difference')}, CI={value.get('bootstrap_ci95')}, source 数={value.get('source_count')}。")
    lines += ["", "## 未评分原因", "", "缺失分数保持为空；以下是唯一 window_id 上按条件统计的缺失原因："]
    for row in summary.get("unscored_reasons", []):
        lines.append(f"- `{row['condition']}` / `{row['reason']}`：{row['count']}")
    lines += ["", "## Source 覆盖", "", "详细表：`evaluation/source_coverage_summary.csv`。该表分别记录计划、前端、有效结构、各条件评分、三条件交集和主标签过滤交集；部分 source 不进入主 source-level 统计。", ""]
    coverage_summary_path = root / "evaluation/source_coverage_summary.csv"
    if coverage_summary_path.is_file():
        for row in csv.DictReader(coverage_summary_path.open(newline="", encoding="utf-8")):
            lines.append(f"- `{row.get('source_id')}`：planned={row.get('planned_windows')}, frontend={row.get('frontend_windows')}, H={row.get('H_MEAN_A_scored_windows')}, B-A={row.get('B_MEAN_A_scored_windows')}, B-C={row.get('B_MEAN_C_scored_windows')}, main real/fake={row.get('main_real_windows')}/{row.get('main_fake_windows')}。")
    lines += ["", "## 时间响应", "", "完整 source 的 real/fake 冻结 logit 曲线以 SVG 输出到 `evaluation/time_curves/`，缺失评分处断线；黄色背景是数据集标注窗口，不是模型预测。索引：`evaluation/time_curve_index.csv`。", "", "## 观测支撑与限制", "", f"各条件模型实际使用帧（按 source/video/window/frame/PTS 去重）：{inventory['model_used_frames_by_condition']}；记录行数：{inventory['model_used_frame_rows']}。前端处理帧不自动等同模型使用时刻。", f"固定网格标签类别计数：{inventory['annotation_categories']}。只有完整处理 source 且三条件共同有有效分数、并满足主标签过滤的窗口才进入主评价；部分 source、NO_VALID_TRIPLET、缺少 seed 模型、前端失败和未运行窗口均保留为缺失，不填 0。", "本 pilot 不是 sealed-test、不是新模型训练，也没有空间真值。窗口有分数不等于失真部位有观测；缺失不自动解释为伪造。"]
    (root / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def reproduce_old_score(root: Path) -> dict[str, Any]:
    """Check the deserializer/scorer against one previously saved OOF row."""

    old_feature = SOURCE_ROOT / "features/0001_MANIP_25__fake.json"
    old_oof = SOURCE_ROOT / "scores/oof_window_scores.csv"
    if not old_feature.is_file() or not old_oof.is_file():
        return {"status": "UNAVAILABLE"}
    feature = json.loads(old_feature.read_text(encoding="utf-8"))
    with old_oof.open(newline="", encoding="utf-8") as handle:
        expected_rows = {row["window_id"]: row for row in csv.DictReader(handle)}
    expected = expected_rows.get("0001_MANIP_25::fake", {}).get("H_MEAN_A_seed_20260909")
    models, _ = _load_models()
    item = models.get(("H_MEAN_A", "01KML", 20260909))
    if expected is None or item is None:
        return {"status": "UNAVAILABLE"}
    model, standardizer = item
    example = {"window_id": "0001_MANIP_25::fake", "source_id": "01KML", "role": "fake", "kind": "MANIP", "triplets": feature["h_support"]["triplets"]}
    batch, _ = build_batch([example], "UNORDERED_STATE", standardizer=standardizer)
    actual = float(score_pooling_model(model, batch)[0])
    result = {"status": "CHECKED", "window_id": example["window_id"], "condition": "H_MEAN_A", "seed": 20260909, "expected_oof_logit": float(expected), "recomputed_logit": actual, "absolute_error": abs(actual - float(expected)), "tolerance": 1e-6, "pass": abs(actual - float(expected)) <= 1e-6}
    _atomic_json(root / "evaluation/old_score_reproduction.json", result)
    return result


def run(root: Path, *, resume: bool, budget_s: float, max_frontend_windows: int | None = None) -> dict[str, Any]:
    root.mkdir(parents=True, exist_ok=True)
    lock_path = _acquire_run_lock(root)
    try:
        if not (root / "manifests/grid_windows.json").is_file():
            prepare_plan(root)
            write_protocol(root)
        elif not resume:
            raise RuntimeError("FROZEN_PLAN_EXISTS: use --resume; refusing to replace the existing grid plan")
        frontend = run_frontend(root, budget_s=budget_s, resume=resume, max_windows=max_frontend_windows)
        if max_frontend_windows is not None:
            return {"frontend": frontend, "status": frontend.get("status")}
        postprocess_started = time.monotonic()
        feature = build_features(root, resume=resume)
        scores = score_windows(root, resume=resume)
        reproduction = reproduce_old_score(root)
        summary = evaluate(root)
        build_review(root)
        _write_report_v2(root, frontend, feature, summary)
        postprocess_elapsed_s = time.monotonic() - postprocess_started
        inventory = _actual_inventory(root)
        final_status = frontend.get("status")
        _atomic_json(root / "final_status.json", {"status": final_status, "frontend": frontend, "feature": feature, "scores": scores, "inventory": inventory, "old_score_reproduction": reproduction, "postprocess_elapsed_s": postprocess_elapsed_s, "git_head": _git_head(), "planned_windows": inventory["planned_windows"], "frontend_results": inventory["frontend_results"], "updated_unix": time.time()})
        return {"frontend": frontend, "feature": feature, "scores": scores, "old_score_reproduction": reproduction, "inventory": inventory, "status": final_status}
    except Exception as exc:
        _atomic_json(root / "final_status.json", {"status": "FAILED", "error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc(), "git_head": _git_head(), "updated_unix": time.time()})
        raise
    finally:
        _release_run_lock(lock_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--budget-s", type=float, default=7200.0)
    parser.add_argument("--max-frontend-windows", type=int, default=None, help="bounded smoke run; do not use for the full pilot")
    parser.add_argument("--phase", choices=("plan", "run"), default="run")
    args = parser.parse_args()
    if args.phase == "plan":
        args.output_root.mkdir(parents=True, exist_ok=True)
        if (args.output_root / "manifests/grid_windows.json").is_file():
            raise RuntimeError("FROZEN_PLAN_EXISTS: refusing to overwrite an existing plan")
        prepare_plan(args.output_root); write_protocol(args.output_root); print(json.dumps({"status": "PLANNED", "output": str(args.output_root)}, indent=2)); return
    print(json.dumps(run(args.output_root, resume=args.resume, budget_s=args.budget_s, max_frontend_windows=args.max_frontend_windows), indent=2, default=_json_default), flush=True)


if __name__ == "__main__":
    main()
