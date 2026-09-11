"""Resumable dense local-structure V7 exploratory pilot.

The pilot deliberately keeps its data products outside the Git checkout.  A
window is complete only when its identity record and all numerical arrays are
present, so a failed or interrupted frontend attempt is retried rather than
being mistaken for a result.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import subprocess
import time
import traceback
import zipfile
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .frontend import (
    SUPPORTED_BATCH_SIZES,
    benchmark_tracking_batches,
    build_geometry,
    infer_depth_and_masks,
    sha256,
    track_dense_window,
)
from .grouping import assign_groups, fixed_local_edges
from .representation import fixed_edge_triplet
from .train_probe import load_feature_examples, train_and_evaluate


DATA_ROOT = Path("/root/autodl-tmp/data/sparse_3d_forgery_detection")
FIXED_SOURCE_ROOT = DATA_ROOT / "derived/v7_activityforensics_observation_density_diagnostic_v1"
FULL_SOURCE_ROOT = DATA_ROOT / "derived/v7_activityforensics_density_matched_frontend_v1"
OUTPUT_ROOT = DATA_ROOT / "derived/v7_activityforensics_dense_local_structure_pilot_gpu_resume_v1"
OLD_OUTPUT_ROOTS = (
    DATA_ROOT / "derived/v7_activityforensics_dense_local_structure_pilot_chunkfix_v1",
    DATA_ROOT / "derived/v7_activityforensics_dense_local_structure_pilot_v1",
)
TAPNET_SOURCE = DATA_ROOT / "external/v7_explicit_geometry/tapnet-c2cbab81cc06092b5f05bfe2da7bfec54e2079c9"
TAPNET_CHECKPOINT = DATA_ROOT / "external/v7_explicit_geometry/checkpoints/causal_bootstapir_checkpoint.pt"
DEPTH_WEIGHT = DATA_ROOT / "external/yolo26_depth/yolo26m-depth.pt"
SEG_WEIGHT = DATA_ROOT / "external/yolo26_depth/yolo26m-seg.pt"
IMPLEMENTATION_REVISION = "dense-gpu-resume-v1"
BATCH_SIZE = 128
PERFORMANCE_LIMIT_S = 15 * 60
FRONTEND_LIMIT_S = 8 * 3600
TRAINING_LIMIT_S = 2 * 3600
BOOTSTRAP_SEED = 20260909
BOOTSTRAP_REPLICATES = 10_000


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row}) or ["status"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _atomic_npz(path: Path, **arrays: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    np.savez_compressed(temporary, **arrays)
    generated = temporary if temporary.suffix == ".npz" else temporary.with_suffix(temporary.suffix + ".npz")
    generated.replace(path)


def _head() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def _read_manifest(path: Path, expected: int) -> list[dict[str, Any]]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(rows, list) or len(rows) != expected:
        raise RuntimeError(f"{path} must contain exactly {expected} rows")
    return [dict(row) for row in rows]


def _manifests() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    fixed = _read_manifest(FIXED_SOURCE_ROOT / "window_manifest.json", 16)
    full = _read_manifest(FULL_SOURCE_ROOT / "window_manifest.json", 192)
    fixed_ids = {str(row["window_id"]) for row in fixed}
    full_ids = {str(row["window_id"]) for row in full}
    if not fixed_ids <= full_ids:
        raise RuntimeError("fixed manifest is not a subset of the full manifest")
    return fixed, full


def _protocol(*, batch_size: int = BATCH_SIZE) -> dict[str, Any]:
    try:
        ultralytics_version = importlib.metadata.version("ultralytics")
    except importlib.metadata.PackageNotFoundError:
        ultralytics_version = None
    fixed, full = _manifests()
    return {
        "implementation_revision": IMPLEMENTATION_REVISION,
        "experiment": "V7 dense candidate points and boundary-constrained local structure exploratory pilot",
        "status_class": "EXPLORATORY_PILOT",
        "git_head": _head(),
        "population": {
            "fixed_manifest": str(FIXED_SOURCE_ROOT / "window_manifest.json"),
            "fixed_manifest_sha256": sha256(FIXED_SOURCE_ROOT / "window_manifest.json"),
            "fixed_windows": len(fixed),
            "full_manifest": str(FULL_SOURCE_ROOT / "window_manifest.json"),
            "full_manifest_sha256": sha256(FULL_SOURCE_ROOT / "window_manifest.json"),
            "full_windows": len(full),
            "initial_frontend_gate": "fixed 16 windows before full expansion",
        },
        "tracking": {
            "provider": "BootsTAPIR causal checkpoint",
            "checkpoint": str(TAPNET_CHECKPOINT),
            "checkpoint_sha256": sha256(TAPNET_CHECKPOINT),
            "analysis_size": [384, 384],
            "grid_spacing_px": 3,
            "grid_centers": "1.5,4.5,...",
            "query_count": 16384,
            "external_query_batch_size": int(batch_size),
            "internal_query_chunk_size": int(batch_size),
            "no_query_reduction_or_fallback": True,
            "feature_grids_reused_per_window": True,
        },
        "performance_check": {
            "status": "PENDING",
            "selected_external_query_batch_size": int(batch_size),
            "comparison_batch_sizes": list(SUPPORTED_BATCH_SIZES),
            "query_count": 1024,
            "time_limit_s": PERFORMANCE_LIMIT_S,
        },
        "depth": {
            "provider": "official Ultralytics YOLO26m-depth",
            "weight": str(DEPTH_WEIGHT),
            "weight_sha256": sha256(DEPTH_WEIGHT),
            "input_size": 768,
            "semantics": "result.depth.data positive optical-axis z-depth aligned to source RGB",
        },
        "segmentation": {
            "provider": "official Ultralytics YOLO26m-seg",
            "ultralytics_version": ultralytics_version,
            "weight": str(SEG_WEIGHT),
            "weight_sha256": sha256(SEG_WEIGHT),
            "frames": "window start only",
            "mask_coordinate_mapping": "polygon in original RGB raster, then nearest resize to 384",
        },
        "grouping": {
            "block_size_px": [24, 24],
            "overlap": "smaller mask area then mask index",
            "background": "separate group per block",
            "identity": "frozen at start grid coordinate",
        },
        "edges": {
            "within_group_only": True,
            "max_neighbors": 8,
            "selection": "start-coordinate distance, query id tie-break, undirected dedup",
            "identity": "frozen for window",
        },
        "geometry": {
            "backprojection": "X=d K^-1 [u,v,1]",
            "coordinate_frame": "camera coordinates per frame; no cross-frame camera XYZ subtraction",
            "intrinsics": "recorded Depth Pro-derived focal convention from matching density289 metadata",
            "camera_motion_compensation": False,
            "missing": "raw_uv retained; invalid XYZ NaN; no zero, clamp, interpolation or future repair",
        },
        "representation": {
            "history": "timestamps < window_start + 0.5 s",
            "targets_s": [0.5, 0.6, 0.7, 0.8, 0.9],
            "state": "S=[mean,std,p25,p75] over common fixed-edge distances / ordinary history median",
            "common_edge_rule": "same edge identities valid at all three target frames; at least 3 edges and 3 endpoints",
            "arms": ["UNORDERED_STATE", "ORDERED_SECOND", "PERMUTED_SECOND"],
            "no_extra_features": True,
        },
        "training": {
            "labels": "MANIP real=0, fake=1; CTRL excluded",
            "split": "source-disjoint LOSO",
            "seeds": [20260909, 20260910, 20260911],
            "epochs": 200,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        },
        "boundaries": ["No confidence or quality feature", "No old component filtering", "No formal src change", "No full-video or sealed-test claim", "No spatial ground-truth claim"],
        "official_references": ["https://docs.ultralytics.com/tasks/depth", "https://docs.ultralytics.com/tasks/segment"],
    }


def _identity(row: Mapping[str, Any], protocol: Mapping[str, Any]) -> dict[str, Any]:
    identity = {
        "implementation_revision": protocol["implementation_revision"],
        "window_id": str(row["window_id"]),
        "source_id": str(row["source_id"]),
        "role": str(row["role"]),
        "kind": str(row["kind"]),
        "video_path": str(row["video_path"]),
        "frame_indices": [int(value) for value in row["frame_indices"]],
        "timestamps_s": [float(value) for value in row["timestamps_s"]],
        "checkpoint_sha256": protocol["tracking"]["checkpoint_sha256"],
        "depth_weight_sha256": protocol["depth"]["weight_sha256"],
        "seg_weight_sha256": protocol["segmentation"]["weight_sha256"],
        "analysis_hw": protocol["tracking"]["analysis_size"],
        "query_count": protocol["tracking"]["query_count"],
        "external_query_batch_size": protocol["tracking"]["external_query_batch_size"],
    }
    encoded = json.dumps(identity, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    return {"identity": identity, "identity_sha256": hashlib.sha256(encoded).hexdigest()}


def _load_protocol(output: Path, *, resume: bool) -> dict[str, Any]:
    path = output / "protocol.json"
    if path.exists():
        if not resume:
            raise FileExistsError(f"existing pilot output requires --resume: {output}")
        previous = json.loads(path.read_text(encoding="utf-8"))
        selected_batch = int(previous.get("tracking", {}).get("external_query_batch_size", BATCH_SIZE))
        protocol = _protocol(batch_size=selected_batch)
        previous_identity = dict(previous)
        current_identity = dict(protocol)
        previous_identity.pop("git_head", None)
        current_identity.pop("git_head", None)
        previous_identity.pop("performance_check", None)
        current_identity.pop("performance_check", None)
        if previous_identity != current_identity:
            raise RuntimeError("existing pilot protocol identity does not match current protocol")
        return previous
    protocol = _protocol()
    _write_json(path, protocol)
    return protocol


def phase_prepare(output: Path, *, resume: bool = False) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    protocol = _load_protocol(output, resume=resume)
    fixed, full = _manifests()
    _write_json(output / "manifests/population.json", {"fixed": fixed, "full": full})
    reuse = _audit_reuse(output, protocol, full, resume=resume)
    result = {
        "stage": "prepare",
        "status": "PREPARED",
        "fixed_windows": len(fixed),
        "full_windows": len(full),
        "reuse_counts": reuse["counts"],
        "updated_unix": time.time(),
    }
    _write_json(output / "progress.json", result)
    _write_json(output / "run_summary.json", {"status": "PREPARED", "protocol": "protocol.json", **result})
    return result


def phase_benchmark(output: Path, *, resume: bool) -> dict[str, Any]:
    """Run the finite CUDA batch comparison before any new frontend work."""

    benchmark_path = output / "performance_benchmark.json"
    protocol = _load_protocol(output, resume=resume)
    if resume and benchmark_path.is_file():
        benchmark = json.loads(benchmark_path.read_text(encoding="utf-8"))
    else:
        fixed, _ = _manifests()
        try:
            benchmark = benchmark_tracking_batches(
                fixed[0],
                tapnet_source=TAPNET_SOURCE,
                checkpoint=TAPNET_CHECKPOINT,
                batch_sizes=SUPPORTED_BATCH_SIZES,
                query_count=1024,
                time_limit_s=PERFORMANCE_LIMIT_S,
            )
        except Exception as exc:
            benchmark = {
                "status": "PERFORMANCE_CHECK_FAILED",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(limit=30),
                "selected_external_query_batch_size": None,
            }
        _write_json(benchmark_path, benchmark)
    selected = benchmark.get("selected_external_query_batch_size")
    if benchmark.get("status") in {"PERFORMANCE_IMPROVED", "PERFORMANCE_NOT_IMPROVED"} and selected in SUPPORTED_BATCH_SIZES:
        protocol = dict(protocol)
        tracking = dict(protocol["tracking"])
        tracking["external_query_batch_size"] = int(selected)
        tracking["internal_query_chunk_size"] = int(selected)
        protocol["tracking"] = tracking
        protocol["performance_check"] = benchmark
        _write_json(output / "protocol.json", protocol)
        reuse_path = output / "reuse_manifest.json"
        reuse = json.loads(reuse_path.read_text(encoding="utf-8")) if reuse_path.is_file() else {"records": [], "counts": {}}
        baseline = next((row for row in benchmark.get("records", []) if row.get("batch_size") == 128 and row.get("status") == "SUCCESS"), None)
        reuse["compatibility_gate"] = "PASS" if baseline and baseline.get("numeric_compatible", False) else "FAIL"
        reuse["compatibility_basis"] = {
            "benchmark_path": str(benchmark_path),
            "selected_batch_size": int(selected),
            "baseline_128_numeric_compatible": bool(baseline and baseline.get("numeric_compatible", False)),
            "old_arrays_not_copied": True,
        }
        _write_json(reuse_path, reuse)
    else:
        reuse_path = output / "reuse_manifest.json"
        if reuse_path.is_file():
            reuse = json.loads(reuse_path.read_text(encoding="utf-8"))
            reuse["compatibility_gate"] = "FAIL"
            reuse["compatibility_basis"] = {"benchmark_path": str(benchmark_path), "reason": "no verified batch configuration"}
            _write_json(reuse_path, reuse)
    result = {
        "stage": "benchmark",
        "status": benchmark.get("status", "PERFORMANCE_CHECK_FAILED"),
        "selected_external_query_batch_size": selected,
        "records": benchmark.get("records", []),
        "time_limit_s": PERFORMANCE_LIMIT_S,
        "updated_unix": time.time(),
    }
    _write_json(output / "progress.json", result)
    return result


def _safe_window(window_id: str) -> str:
    return str(window_id).replace("::", "__").replace("/", "_").replace("\\", "_")


def _focal_for_window(row: Mapping[str, Any]) -> tuple[float, str]:
    safe = _safe_window(str(row["window_id"]))
    candidates = [FIXED_SOURCE_ROOT / "particles" / f"{safe}__density289.json", FULL_SOURCE_ROOT / "particles" / f"{safe}__density289.json"]
    for path in candidates:
        if path.is_file():
            meta = json.loads(path.read_text(encoding="utf-8"))
            focal = meta.get("provenance", {}).get("fixed_focal_px")
            if focal is not None and np.isfinite(float(focal)) and float(focal) > 0:
                return float(focal), str(path)
    raise FileNotFoundError(f"no fixed focal metadata for {row['window_id']}")


def _frontend_complete(meta_path: Path, expected: Mapping[str, Any]) -> bool:
    if not meta_path.is_file() or not meta_path.with_suffix(".npz").is_file():
        return False
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        return meta.get("status") == "COMPLETE" and meta.get("identity_sha256") == expected["identity_sha256"]
    except (OSError, ValueError, json.JSONDecodeError):
        return False


_REQUIRED_FRONTEND_ARRAYS = {
    "raw_uv",
    "visibility",
    "xyz",
    "geometry_validity",
    "frame_indices",
    "timestamps_s",
    "query_start_uv_analysis",
    "group_ids",
    "edges",
    "edge_groups",
}


def _source_frontend_paths(window_id: str) -> list[tuple[Path, Path]]:
    safe = _safe_window(window_id)
    return [
        (root / "frontend/windows" / f"{safe}.json", root / "frontend/windows" / f"{safe}.npz")
        for root in OLD_OUTPUT_ROOTS
    ]


def _row_identity_matches(meta: Mapping[str, Any], row: Mapping[str, Any]) -> bool:
    identity = meta.get("identity")
    if not isinstance(identity, Mapping):
        return False
    for key in ("window_id", "source_id", "role", "kind", "video_path"):
        if str(identity.get(key)) != str(row.get(key)):
            return False
    for key in ("frame_indices",):
        if [int(value) for value in identity.get(key, [])] != [int(value) for value in row.get(key, [])]:
            return False
    observed_times = np.asarray(identity.get("timestamps_s", []), dtype=np.float64)
    expected_times = np.asarray(row.get("timestamps_s", []), dtype=np.float64)
    return observed_times.shape == expected_times.shape and bool(np.allclose(observed_times, expected_times, rtol=0.0, atol=1e-6))


def _validate_frontend_arrays(npz_path: Path, row: Mapping[str, Any], query_count: int) -> dict[str, Any]:
    with np.load(npz_path, allow_pickle=False) as arrays:
        names = set(arrays.files)
        if not _REQUIRED_FRONTEND_ARRAYS <= names:
            return {"ok": False, "reason": f"missing_arrays:{sorted(_REQUIRED_FRONTEND_ARRAYS - names)}"}
        frames = len(row["frame_indices"])
        checks = {
            "raw_uv": (frames, query_count, 2),
            "visibility": (frames, query_count),
            "xyz": (frames, query_count, 3),
            "geometry_validity": (frames, query_count),
            "frame_indices": (frames,),
            "timestamps_s": (frames,),
            "query_start_uv_analysis": (query_count, 2),
            "group_ids": (query_count,),
            "edges": (None, 2),
            "edge_groups": (None,),
        }
        for name, shape in checks.items():
            observed = tuple(arrays[name].shape)
            if shape[0] is None:
                if len(observed) != len(shape) or observed[1:] != shape[1:]:
                    return {"ok": False, "reason": f"shape:{name}:{observed}"}
            elif observed != shape:
                return {"ok": False, "reason": f"shape:{name}:{observed}:expected:{shape}"}
        if not np.array_equal(np.asarray(arrays["frame_indices"], dtype=np.int64), np.asarray(row["frame_indices"], dtype=np.int64)):
            return {"ok": False, "reason": "frame_indices_mismatch"}
    return {"ok": True}


def _audit_reuse(output: Path, protocol: Mapping[str, Any], rows: Sequence[Mapping[str, Any]], *, resume: bool) -> dict[str, Any]:
    """Audit old artifacts without copying or modifying them."""

    path = output / "reuse_manifest.json"
    if resume and path.is_file():
        return json.loads(path.read_text(encoding="utf-8"))
    records: list[dict[str, Any]] = []
    query_count = int(protocol["tracking"]["query_count"])
    for row in rows:
        window_id = str(row["window_id"])
        record: dict[str, Any] = {
            "window_id": window_id,
            "source_id": str(row["source_id"]),
            "role": str(row["role"]),
            "kind": str(row["kind"]),
            "status": "MISSING",
            "reason": "no prior dense pilot artifact",
        }
        for meta_path, npz_path in _source_frontend_paths(window_id):
            if not meta_path.exists() and not npz_path.exists():
                continue
            if not meta_path.exists() or not npz_path.exists():
                record.update({"status": "INCOMPLETE", "reason": "metadata_or_npz_missing", "source_root": str(meta_path.parent.parent.parent)})
                continue
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                if meta.get("status") != "COMPLETE":
                    record.update({"status": "INCOMPATIBLE", "reason": f"source_status:{meta.get('status')}", "source_meta": str(meta_path), "source_npz": str(npz_path)})
                    continue
                if not _row_identity_matches(meta, row):
                    record.update({"status": "INCOMPATIBLE", "reason": "window_identity_mismatch", "source_meta": str(meta_path), "source_npz": str(npz_path)})
                    continue
                array_check = _validate_frontend_arrays(npz_path, row, query_count)
                if not array_check["ok"]:
                    record.update({"status": "INCOMPATIBLE", "reason": array_check["reason"], "source_meta": str(meta_path), "source_npz": str(npz_path)})
                    continue
                old_identity = dict(meta.get("identity", {}))
                structural_ok = (
                    old_identity.get("checkpoint_sha256") == protocol["tracking"]["checkpoint_sha256"]
                    and old_identity.get("depth_weight_sha256") == protocol["depth"]["weight_sha256"]
                    and old_identity.get("seg_weight_sha256") == protocol["segmentation"]["weight_sha256"]
                    and old_identity.get("analysis_hw") == protocol["tracking"]["analysis_size"]
                    and int(old_identity.get("query_count", -1)) == query_count
                )
                old_batch = int(old_identity.get("external_query_batch_size", -1))
                same_revision = old_identity.get("implementation_revision") == IMPLEMENTATION_REVISION
                status = "REUSABLE" if structural_ok and same_revision and old_batch == int(protocol["tracking"]["external_query_batch_size"]) else "PARTIAL_REUSABLE" if structural_ok else "INCOMPATIBLE"
                reason = "matching execution identity" if status == "REUSABLE" else "arrays and geometry identity match; old execution batch/revision requires benchmark gate" if status == "PARTIAL_REUSABLE" else "provider or analysis identity mismatch"
                if status == "INCOMPATIBLE":
                    record.update({"status": status, "reason": reason, "source_meta": str(meta_path), "source_npz": str(npz_path)})
                    continue
                record.update({
                    "status": status,
                    "reason": reason,
                    "source_meta": str(meta_path),
                    "source_npz": str(npz_path),
                    "source_meta_sha256": sha256(meta_path),
                    "source_npz_sha256": sha256(npz_path),
                    "old_implementation_revision": old_identity.get("implementation_revision"),
                    "old_external_query_batch_size": old_batch,
                    "frame_count": len(row["frame_indices"]),
                    "query_count": query_count,
                })
                break
            except Exception as exc:
                record.update({"status": "INCOMPATIBLE", "reason": f"audit_error:{type(exc).__name__}:{exc}", "source_meta": str(meta_path), "source_npz": str(npz_path)})
        records.append(record)
    summary = {
        "implementation_revision": IMPLEMENTATION_REVISION,
        "status": "AUDITED",
        "compatibility_gate": "PENDING_BENCHMARK",
        "old_roots": [str(root) for root in OLD_OUTPUT_ROOTS],
        "records": records,
        "counts": {status: sum(record["status"] == status for record in records) for status in ("REUSABLE", "PARTIAL_REUSABLE", "INCOMPATIBLE", "INCOMPLETE", "MISSING")},
        "updated_unix": time.time(),
    }
    _write_json(path, summary)
    _write_csv(output / "reuse_audit.csv", records)
    return summary


def _reusable_records(output: Path) -> dict[str, dict[str, Any]]:
    path = output / "reuse_manifest.json"
    if not path.is_file():
        return {}
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("compatibility_gate") != "PASS":
        return {}
    return {
        str(record["window_id"]): record
        for record in manifest.get("records", [])
        if record.get("status") in {"REUSABLE", "PARTIAL_REUSABLE"}
    }


def _frontend_meta_paths(output: Path, protocol: Mapping[str, Any]) -> list[Path]:
    """Return one validated new or reused metadata path per manifest window."""

    _, full = _manifests()
    reused = _reusable_records(output)
    paths: list[Path] = []
    for row in full:
        new_path = output / "frontend/windows" / f"{_safe_window(str(row['window_id']))}.json"
        if _frontend_complete(new_path, _identity(row, protocol)):
            paths.append(new_path)
        elif str(row["window_id"]) in reused:
            source = Path(str(reused[str(row["window_id"])] ["source_meta"]))
            if source.is_file():
                paths.append(source)
    return paths


def _run_one_frontend(row: Mapping[str, Any], protocol: Mapping[str, Any], output: Path) -> dict[str, Any]:
    started = time.perf_counter()
    identity = _identity(row, protocol)
    safe = _safe_window(str(row["window_id"]))
    window_dir = output / "frontend/windows"
    meta_path = window_dir / f"{safe}.json"
    npz_path = window_dir / f"{safe}.npz"
    try:
        track = track_dense_window(
            row,
            tapnet_source=TAPNET_SOURCE,
            checkpoint=TAPNET_CHECKPOINT,
            batch_size=int(protocol["tracking"]["external_query_batch_size"]),
            progress_prefix="dense",
        )
        depths, masks, measurement = infer_depth_and_masks(track["frames_rgb"], depth_weight=DEPTH_WEIGHT, seg_weight=SEG_WEIGHT)
        focal, focal_path = _focal_for_window(row)
        xyz, geometry_validity, geometry_meta = build_geometry(track["raw_uv"], depths, focal_px=focal)
        groups, group_summary = assign_groups(track["query_start_uv_analysis"], masks)
        edge_rows = fixed_local_edges(track["query_start_uv_analysis"], groups, max_neighbors=8)
        edges = np.asarray([[int(edge["left_query_id"]), int(edge["right_query_id"])] for edge in edge_rows], dtype=np.int32)
        edge_groups = np.asarray([str(edge["group_id"]) for edge in edge_rows], dtype="U64")
        _atomic_npz(npz_path, raw_uv=np.asarray(track["raw_uv"], dtype=np.float32), visibility=np.asarray(track["visibility"], dtype=bool), xyz=np.asarray(xyz, dtype=np.float32), geometry_validity=np.asarray(geometry_validity, dtype=bool), frame_indices=np.asarray(track["frame_indices"], dtype=np.int64), timestamps_s=np.asarray(track["timestamps_s"], dtype=np.float64), query_start_uv_analysis=np.asarray(track["query_start_uv_analysis"], dtype=np.float32), group_ids=groups.astype("U64"), edges=edges, edge_groups=edge_groups)
        result = {**identity, "status": "COMPLETE", "row": dict(row), "query_count": int(track["query_count"]), "frame_count": len(track["frame_indices"]), "raw_uv_shape": list(np.asarray(track["raw_uv"]).shape), "visibility_fraction": float(np.mean(track["visibility"])), "geometry_valid_fraction": float(np.mean(geometry_validity)), "group_count": len(group_summary), "group_summary": group_summary, "edge_count": len(edge_rows), "depth_focal_px": focal, "focal_metadata_path": focal_path, "measurement": measurement, "geometry": geometry_meta, "elapsed_s": float(time.perf_counter() - started), "peak_gpu_bytes": int(track["peak_gpu_bytes"]), "npz": str(npz_path)}
        _write_json(meta_path, result)
        return result
    except Exception as exc:
        result = {**identity, "status": "FAILED", "row": dict(row), "elapsed_s": float(time.perf_counter() - started), "error_type": type(exc).__name__, "error": str(exc), "traceback": traceback.format_exc(limit=20)}
        _write_json(meta_path, result)
        if npz_path.exists():
            npz_path.unlink()
        return result


def phase_frontend(output: Path, *, resume: bool, limit: int | None = None, expand_full: bool = False) -> dict[str, Any]:
    protocol = _load_protocol(output, resume=resume)
    fixed, full = _manifests()
    reused = _reusable_records(output)
    fixed_ids = {str(row["window_id"]) for row in fixed}
    ordered = fixed + [row for row in full if str(row["window_id"]) not in fixed_ids]
    def is_new_complete(row: Mapping[str, Any]) -> bool:
        return _frontend_complete(output / "frontend/windows" / f"{_safe_window(str(row['window_id']))}.json", _identity(row, protocol))

    def is_complete(row: Mapping[str, Any]) -> bool:
        return is_new_complete(row) or str(row["window_id"]) in reused

    complete_fixed_ids = {str(row["window_id"]) for row in fixed if is_complete(row)}
    complete_full_ids = {str(row["window_id"]) for row in full if is_complete(row)}
    projected_full_s = None
    if limit is not None:
        selected = [row for row in ordered[:limit] if not is_complete(row)]
    elif expand_full and len(complete_fixed_ids) < len(fixed):
        selected = [row for row in fixed if not is_complete(row)]
    elif expand_full:
        complete_elapsed: list[float] = []
        for row in full:
            meta_path = output / "frontend/windows" / f"{_safe_window(str(row['window_id']))}.json"
            source_meta = meta_path if is_new_complete(row) else Path(reused[str(row["window_id"])] ["source_meta"]) if str(row["window_id"]) in reused else None
            if source_meta is not None:
                try:
                    complete_elapsed.append(float(json.loads(source_meta.read_text(encoding="utf-8")).get("elapsed_s", 0.0)))
                except (OSError, ValueError, json.JSONDecodeError):
                    pass
        mean_existing = float(np.mean(complete_elapsed)) if complete_elapsed else float("inf")
        projected_full_s = mean_existing * len(full)
        if projected_full_s > FRONTEND_LIMIT_S:
            result = {"stage": "frontend", "status": "DENSE_RUNTIME_LIMITED", "selected_windows": 0, "complete_fixed_windows": len(complete_fixed_ids), "complete_full_windows": len(complete_full_ids), "fixed_total": len(fixed), "full_total": len(full), "elapsed_s": 0.0, "mean_selected_window_s": mean_existing, "projected_full_s": projected_full_s, "runtime_limit_s": FRONTEND_LIMIT_S, "expansion_skipped": "fixed-window throughput projects beyond the declared 8-hour budget", "retry_policy": "FAILED windows are retried; only COMPLETE with matching identity or benchmark-gated reuse is resumable"}
            _write_json(output / "frontend_summary.json", result)
            _write_json(output / "progress.json", {**result, "updated_unix": time.time()})
            return result
        selected = [row for row in full if not is_complete(row)]
    else:
        selected = [row for row in fixed if not is_complete(row)]
    if not selected:
        status = "FULL_FRONTEND_COMPLETE" if len(complete_full_ids) == len(full) else "FIXED_FRONTEND_COMPLETE" if len(complete_fixed_ids) == len(fixed) else "DENSE_FRONTEND_PARTIAL"
        result = {"stage": "frontend", "status": status, "selected_windows": 0, "complete_fixed_windows": len(complete_fixed_ids), "complete_full_windows": len(complete_full_ids), "fixed_total": len(fixed), "full_total": len(full), "elapsed_s": 0.0, "projected_full_s": projected_full_s, "runtime_limit_s": FRONTEND_LIMIT_S, "reuse_count": len(reused), "retry_policy": "FAILED windows are retried; only COMPLETE with matching identity or benchmark-gated reuse is resumable"}
        _write_json(output / "frontend_summary.json", result)
        _write_json(output / "progress.json", {**result, "updated_unix": time.time()})
        return result
    started = time.perf_counter()
    rows: list[dict[str, Any]] = []
    budget_exhausted = False
    for index, row in enumerate(selected, 1):
        if expand_full and limit is None and time.perf_counter() - started >= FRONTEND_LIMIT_S:
            budget_exhausted = True
            break
        expected = _identity(row, protocol)
        meta_path = output / "frontend/windows" / f"{_safe_window(str(row['window_id']))}.json"
        result = json.loads(meta_path.read_text(encoding="utf-8")) if resume and _frontend_complete(meta_path, expected) else _run_one_frontend(row, protocol, output)
        rows.append(result)
        _write_json(output / "progress.json", {"stage": "frontend", "status": "RUNNING", "completed": index, "total": len(selected), "last_window": row["window_id"], "updated_unix": time.time()})
        print(f"frontend {index}/{len(selected)} {row['window_id']} {result.get('status')} elapsed={result.get('elapsed_s', 0):.1f}s", flush=True)
    complete_fixed = sum(is_complete(row) for row in fixed)
    complete_full = sum(is_complete(row) for row in full)
    elapsed = time.perf_counter() - started
    mean_s = elapsed / len(selected)
    projected_full_s = mean_s * len(full)
    status = "FIXED_FRONTEND_COMPLETE" if complete_fixed == len(fixed) else "DENSE_FRONTEND_PARTIAL"
    if expand_full and complete_full == len(full):
        status = "FULL_FRONTEND_COMPLETE"
    elif expand_full and (budget_exhausted or (projected_full_s is not None and projected_full_s > FRONTEND_LIMIT_S)):
        status = "DENSE_RUNTIME_LIMITED"
    result = {"stage": "frontend", "status": status, "selected_windows": len(selected), "complete_fixed_windows": complete_fixed, "complete_full_windows": complete_full, "fixed_total": len(fixed), "full_total": len(full), "elapsed_s": elapsed, "mean_selected_window_s": mean_s, "projected_full_s": projected_full_s, "runtime_limit_s": FRONTEND_LIMIT_S, "reuse_count": len(reused), "last_results": rows[-5:], "retry_policy": "FAILED windows are retried; only COMPLETE with matching identity or benchmark-gated reuse is resumable"}
    _write_json(output / "frontend_summary.json", result)
    _write_json(output / "progress.json", {**result, "updated_unix": time.time()})
    return result


def _target_matches(timestamps: np.ndarray, start_s: float) -> list[dict[str, Any]]:
    used: set[int] = set()
    candidates = np.flatnonzero(timestamps >= float(start_s) + 0.5).astype(np.int64)
    output: list[dict[str, Any]] = []
    for slot, offset in enumerate((0.5, 0.6, 0.7, 0.8, 0.9)):
        target = float(start_s) + offset
        choices = [(abs(float(timestamps[index]) - target), int(index)) for index in candidates if int(index) not in used and abs(float(timestamps[index]) - target) <= 0.05]
        if not choices:
            output.append({"slot": slot, "target_time_s": target, "array_index": None, "timestamp_s": None, "status": "MISSING_TARGET_FRAME"})
            continue
        error, index = min(choices, key=lambda item: (item[0], item[1]))
        used.add(index)
        output.append({"slot": slot, "target_time_s": target, "array_index": index, "timestamp_s": float(timestamps[index]), "match_error_s": float(error), "status": "MATCHED"})
    return output


def _feature_one(meta_path: Path, output: Path) -> dict[str, Any]:
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    npz_path = meta_path.with_suffix(".npz")
    if meta.get("status") != "COMPLETE" or not npz_path.is_file():
        return {"window_id": meta.get("identity", {}).get("window_id", meta_path.stem), "status": "FRONTEND_NOT_COMPLETE"}
    arrays = np.load(npz_path, allow_pickle=False)
    xyz = np.asarray(arrays["xyz"], dtype=np.float64)
    timestamps = np.asarray(arrays["timestamps_s"], dtype=np.float64)
    groups = np.asarray(arrays["group_ids"], dtype="U64")
    edges_array = np.asarray(arrays["edges"], dtype=np.int64)
    edge_groups = np.asarray(arrays["edge_groups"], dtype="U64")
    edges = [{"edge_id": int(index), "left_query_id": int(edge[0]), "right_query_id": int(edge[1]), "group_id": str(edge_groups[index])} for index, edge in enumerate(edges_array)]
    matches = _target_matches(timestamps, float(meta["row"]["interval_start_s"]))
    history = np.flatnonzero(timestamps < float(meta["row"]["interval_start_s"]) + 0.5).astype(np.int64)
    triplet_states: list[np.ndarray] = []
    triplet_times: list[np.ndarray] = []
    component_ids: list[int] = []
    triplet_meta: list[dict[str, Any]] = []
    invalid_reasons: list[str] = []
    group_names = sorted(set(groups.tolist()))
    edge_by_group = {group: [edge for edge in edges if edge["group_id"] == group] for group in group_names}
    for component_index, group in enumerate(group_names):
        for start in range(3):
            target_rows = matches[start : start + 3]
            if any(row["array_index"] is None for row in target_rows):
                invalid_reasons.append(f"{group}:TARGET_FRAME_MISSING:{start}")
                continue
            target_indices = [int(row["array_index"]) for row in target_rows]
            triplet = fixed_edge_triplet(xyz, timestamps, edge_by_group[group], history, target_indices)
            if triplet is None:
                invalid_reasons.append(f"{group}:NO_COMMON_EDGE_SUPPORT:{start}")
                continue
            triplet_states.append(np.asarray(triplet["states"], dtype=np.float64))
            triplet_times.append(np.asarray(triplet["timestamps_s"], dtype=np.float64))
            component_ids.append(component_index)
            triplet_meta.append({"triplet_id": len(triplet_meta), "component_index": component_index, "group_id": group, "target_slots": [start, start + 1, start + 2], "edge_ids": triplet["edge_ids"], "query_ids": triplet["query_ids"], "history_scale": triplet["history_scale"], "target_frame_indices": target_indices})
    safe = meta_path.stem
    feature_dir = output / "features"
    feature_path = feature_dir / f"{safe}.npz"
    feature_meta_path = feature_dir / f"{safe}.json"
    status = "VALID" if triplet_states else "NO_VALID_SUPPORT"
    if triplet_states:
        _atomic_npz(feature_path, states=np.asarray(triplet_states, dtype=np.float64), timestamps_s=np.asarray(triplet_times, dtype=np.float64), component_ids=np.asarray(component_ids, dtype=np.int64))
    elif feature_path.exists():
        feature_path.unlink()
    result = {"window_id": meta["identity"]["window_id"], "pair_id": meta["row"]["pair_id"], "source_id": meta["row"]["source_id"], "role": meta["row"]["role"], "kind": meta["row"]["kind"], "status": status, "triplet_count": len(triplet_meta), "component_count": len(group_names), "history_frame_count": int(history.size), "target_matches": matches, "triplets": triplet_meta, "invalid_reasons": invalid_reasons[:200], "feature_npz": str(feature_path) if triplet_states else None}
    _write_json(feature_meta_path, result)
    return result


def phase_features(output: Path) -> dict[str, Any]:
    protocol = _load_protocol(output, resume=True)
    feature_dir = output / "features"
    feature_dir.mkdir(parents=True, exist_ok=True)
    results = [_feature_one(meta_path, output) for meta_path in _frontend_meta_paths(output, protocol)]
    _write_json(feature_dir / "index.json", {"count": len(results), "results": results})
    _write_csv(output / "coverage_by_window.csv", results)
    status = "FEATURES_COMPLETE" if results else "NO_FRONTEND_ARTIFACTS"
    result = {"stage": "features", "status": status, "windows_seen": len(results), "valid_windows": sum(row["status"] == "VALID" for row in results), "no_valid_support": sum(row["status"] == "NO_VALID_SUPPORT" for row in results), "frontend_not_complete": sum(row["status"] == "FRONTEND_NOT_COMPLETE" for row in results)}
    _write_json(output / "features_summary.json", result)
    return result


def phase_train(output: Path) -> dict[str, Any]:
    examples, coverage = load_feature_examples(output / "features")
    if not examples:
        result = {"status": "NOT_RUN_NO_VALID_FEATURES", "coverage_rows": len(coverage)}
        _write_json(output / "training_summary.json", result)
        return result
    summary = train_and_evaluate(examples, output, device="cuda", time_limit_s=TRAINING_LIMIT_S)
    summary["measurement_support_note"] = "Only sources represented by current complete frontend features are measured; fewer than 12 complete sources is not a cross-source population claim."
    _write_json(output / "training_summary.json", summary)
    return summary


def phase_export(output: Path) -> dict[str, Any]:
    protocol = json.loads((output / "protocol.json").read_text(encoding="utf-8"))
    frontend = json.loads((output / "frontend_summary.json").read_text(encoding="utf-8")) if (output / "frontend_summary.json").is_file() else {"status": "NOT_RUN"}
    features = json.loads((output / "features_summary.json").read_text(encoding="utf-8")) if (output / "features_summary.json").is_file() else {"status": "NOT_RUN"}
    training = json.loads((output / "training_summary.json").read_text(encoding="utf-8")) if (output / "training_summary.json").is_file() else {"status": "NOT_RUN"}
    benchmark = json.loads((output / "performance_benchmark.json").read_text(encoding="utf-8")) if (output / "performance_benchmark.json").is_file() else {"status": "NOT_RUN"}
    reuse = json.loads((output / "reuse_manifest.json").read_text(encoding="utf-8")) if (output / "reuse_manifest.json").is_file() else {"status": "NOT_RUN"}
    summary = {"status": training.get("status") if training.get("status") not in {None, "NOT_RUN"} else frontend.get("status", "UNKNOWN"), "experiment": protocol["experiment"], "implementation_revision": IMPLEMENTATION_REVISION, "benchmark": benchmark, "reuse": reuse, "frontend": frontend, "features": features, "training": training, "causal_training_eligible": False, "causal_training_reason": "This exploratory frontend uses per-frame camera coordinates and identity poses; no causal camera-motion-compensated training claim is made.", "limitations": ["no metric depth ground truth", "no pixel-level spatial ground truth", "no full-video or sealed-test claim", "segmentation is a pretrained grouping prior", "not a formal detector"]}
    _write_json(output / "summary.json", summary)
    _write_csv(output / "runtime_summary.csv", [{"stage": "frontend", "status": frontend.get("status"), "elapsed_s": frontend.get("elapsed_s", "")}, {"stage": "features", "status": features.get("status"), "elapsed_s": features.get("elapsed_s", "")}, {"stage": "training", "status": training.get("status"), "elapsed_s": training.get("training_elapsed_s", "")}])
    _write_csv(output / "case_manifest.csv", [])
    readme = f"""# V7 dense local-structure pilot review bundle

Implementation revision: `{IMPLEMENTATION_REVISION}`. The frontend retains all
16,384 initialized 384x384/3px query identities while processing causal
BootsTAPIR state in externally benchmark-selected batches of
{protocol['tracking'].get('external_query_batch_size', BATCH_SIZE)}. Depth is generated by official
YOLO26m-depth for every window frame and YOLO26m-seg is used only on the start
frame for deterministic grouping. Fixed local edges are shared by all three
target states.

Current status: **{summary['status']}**. This is an exploratory measurement,
not a formal detector, and no pixel-level localization claim is made.

Reproduce with the project interpreter:
`/root/autodl-tmp/projects/sparse_3d_forgery_detection/.venv/bin/python -m research_tools.v7.dense_local_structure_pilot.run_pipeline all --resume`
"""
    (output / "README.md").write_text(readme, encoding="utf-8")
    include = ["README.md", "protocol.json", "summary.json", "performance_benchmark.json", "reuse_manifest.json", "reuse_audit.csv", "frontend_summary.json", "features_summary.json", "training_summary.json", "coverage_by_window.csv", "per_source_metrics.csv", "oof_window_scores.csv", "fold_support.csv", "runtime_summary.csv", "case_manifest.csv", "model_records.json"]
    with zipfile.ZipFile(output / "review_bundle.zip", "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name in include:
            path = output / name
            if path.is_file():
                archive.write(path, arcname=name)
    return summary


def phase_all(output: Path, *, resume: bool) -> dict[str, Any]:
    """Advance phases independently; safe to run from nohup without a caller."""

    try:
        phase_prepare(output, resume=resume)
        benchmark = phase_benchmark(output, resume=resume)
        if benchmark.get("status") == "PERFORMANCE_CHECK_FAILED":
            raise RuntimeError("PERFORMANCE_CHECK_FAILED: no verified CUDA batch configuration")
        reuse_manifest = json.loads((output / "reuse_manifest.json").read_text(encoding="utf-8"))
        if reuse_manifest.get("compatibility_gate") != "PASS":
            raise RuntimeError("PERFORMANCE_NOT_IMPROVED: old frontend cache did not pass numerical compatibility gate")
        frontend = phase_frontend(output, resume=resume, expand_full=True)
        # A first pass may finish the remaining fixed gate.  Re-enter once so
        # the same process can apply the budget gate before any full expansion.
        if frontend.get("status") == "FIXED_FRONTEND_COMPLETE" and frontend.get("complete_full_windows", 0) < frontend.get("full_total", 0):
            frontend = phase_frontend(output, resume=True, expand_full=True)
        phase_features(output)
        phase_train(output)
        summary = phase_export(output)
        if frontend.get("status") == "DENSE_RUNTIME_LIMITED":
            final_status = "FINAL_BUDGET_EXHAUSTED"
        elif frontend.get("status") == "FULL_FRONTEND_COMPLETE":
            final_status = "FINAL_SUCCESS"
        else:
            final_status = "FINAL_PARTIAL"
        final = {"status": final_status, "updated_unix": time.time(), "frontend_status": frontend.get("status"), "summary_status": summary.get("status")}
        _write_json(output / "run_status.json", final)
        return final
    except KeyboardInterrupt:
        final = {"status": "FINAL_FAILED", "error_type": "KeyboardInterrupt", "error": "run interrupted; completed window artifacts remain resumable", "updated_unix": time.time()}
        _write_json(output / "run_status.json", final)
        raise
    except Exception as exc:
        final = {"status": "FINAL_FAILED", "error_type": type(exc).__name__, "error": str(exc), "traceback": traceback.format_exc(limit=30), "updated_unix": time.time()}
        _write_json(output / "run_status.json", final)
        raise


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=["prepare", "benchmark", "frontend", "features", "train", "export", "all"])
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--limit", type=int, default=None, help="number of ordered windows to process (fixed windows come first)")
    parser.add_argument("--expand-full", action="store_true", help="process the remaining full 192-window manifest")
    args = parser.parse_args(argv)
    output = args.output_root
    if args.phase == "all":
        final = phase_all(output, resume=args.resume)
        print(json.dumps(final, indent=2, sort_keys=True, ensure_ascii=False))
        return
    if args.phase == "prepare":
        phase_prepare(output, resume=args.resume)
    if args.phase == "benchmark":
        phase_benchmark(output, resume=args.resume)
    if args.phase == "frontend":
        phase_frontend(output, resume=args.resume, limit=args.limit, expand_full=args.expand_full)
    if args.phase == "features":
        phase_features(output)
    if args.phase == "train":
        phase_train(output)
    if args.phase == "export":
        print(json.dumps(phase_export(output), indent=2, sort_keys=True, ensure_ascii=False))


if __name__ == "__main__":
    main()
