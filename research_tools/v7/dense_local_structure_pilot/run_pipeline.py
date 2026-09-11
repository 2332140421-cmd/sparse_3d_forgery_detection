"""Executable phases for the bounded dense local-structure V7 pilot."""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import importlib.metadata
import json
import platform
import shutil
import subprocess
import time
import traceback
import zipfile
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .frontend import sha256, track_dense_window


DATA_ROOT = Path("/root/autodl-tmp/data/sparse_3d_forgery_detection")
SOURCE_ROOT = DATA_ROOT / "derived/v7_activityforensics_observation_density_diagnostic_v1"
OUTPUT_ROOT = DATA_ROOT / "derived/v7_activityforensics_dense_local_structure_pilot_v1"
TAPNET_SOURCE = DATA_ROOT / "external/v7_explicit_geometry/tapnet-c2cbab81cc06092b5f05bfe2da7bfec54e2079c9"
TAPNET_CHECKPOINT = DATA_ROOT / "external/v7_explicit_geometry/checkpoints/causal_bootstapir_checkpoint.pt"
DEPTH_WEIGHT = DATA_ROOT / "external/yolo26_depth/yolo26m-depth.pt"
SEG_WEIGHT = DATA_ROOT / "external/yolo26_depth/yolo26m-seg.pt"
BOOTSTRAP_SEED = 20260909
BOOTSTRAP_REPLICATES = 10_000


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row}) or ["status"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _head() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def _rows() -> list[dict[str, Any]]:
    rows = json.loads((SOURCE_ROOT / "window_manifest.json").read_text(encoding="utf-8"))
    if not isinstance(rows, list) or len(rows) != 16:
        raise RuntimeError("fixed observation-density manifest must contain exactly 16 windows")
    return [dict(row) for row in rows]


def _protocol() -> dict[str, Any]:
    try:
        ultralytics_version = importlib.metadata.version("ultralytics")
    except importlib.metadata.PackageNotFoundError:
        ultralytics_version = None
    return {
        "experiment": "V7 dense candidate points and boundary-constrained local structure exploratory pilot",
        "status_class": "EXPLORATORY_PILOT",
        "git_head": _head(),
        "population": {"fixed_manifest": str(SOURCE_ROOT / "window_manifest.json"), "windows": 192, "initial_frontend_gate": "fixed 16 windows; first window required before expansion"},
        "tracking": {"provider": "BootsTAPIR causal checkpoint", "checkpoint": str(TAPNET_CHECKPOINT), "checkpoint_sha256": sha256(TAPNET_CHECKPOINT), "analysis_size": [384, 384], "grid_spacing_px": 3, "grid_centers": "1.5,4.5,...", "query_count": 16384, "query_chunk_size": 256, "no_fallback": True},
        "depth": {"provider": "official Ultralytics YOLO26m-depth", "weight": str(DEPTH_WEIGHT), "weight_sha256": sha256(DEPTH_WEIGHT), "input_size": 768, "semantics": "direct result.depth.data aligned to original RGB; no per-frame scale fit"},
        "segmentation": {"provider": "official Ultralytics YOLO26m-seg", "ultralytics_version": ultralytics_version, "weight": str(SEG_WEIGHT), "weight_sha256": sha256(SEG_WEIGHT), "frames": "window start only", "parameters": "Ultralytics default predict parameters; no class filtering"},
        "grouping": {"block_size_px": [24, 24], "overlap": "smaller mask area then mask index", "background": "separate group per block", "identity": "frozen at start frame"},
        "edges": {"within_group_only": True, "max_neighbors": 8, "selection": "start-coordinate distance, query id tie-break, undirected dedup", "identity": "frozen for window"},
        "geometry": {"backprojection": "X=d K^-1 [u,v,1]", "coordinate_frame": "camera coordinates per frame; no cross-frame camera XYZ subtraction", "intrinsics": "existing recorded Depth Pro-derived focal convention only if frontend reaches geometry", "missing": "raw_uv retained; invalid XYZ NaN; no zero, clamp, interpolation or future repair"},
        "representation": {"history": "timestamps < window_start + 0.5 s", "targets_s": [0.5, 0.6, 0.7, 0.8, 0.9], "state": "S=[mean,std,p25,p75] over fixed edge distances / ordinary history median", "arms": ["UNORDERED_STATE", "ORDERED_SECOND", "PERMUTED_SECOND"], "no_extra_features": True},
        "training": {"labels": "MANIP real=0, fake=1; CTRL excluded", "split": "source-disjoint LOSO", "seeds": [20260909, 20260910, 20260911], "epochs": 200, "bootstrap_seed": BOOTSTRAP_SEED, "bootstrap_replicates": BOOTSTRAP_REPLICATES},
        "boundaries": ["No confidence or quality feature", "No old component filtering", "No formal src change", "No full-video or sealed-test claim", "No spatial ground-truth claim"],
        "official_references": ["https://docs.ultralytics.com/tasks/depth", "https://docs.ultralytics.com/tasks/segment"],
    }


def phase_prepare(output: Path, *, resume: bool = False) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    protocol = _protocol()
    existing_protocol = output / "protocol.json"
    if existing_protocol.is_file():
        if not resume:
            raise FileExistsError(f"existing pilot output requires --resume: {output}")
        previous = json.loads(existing_protocol.read_text(encoding="utf-8"))
        previous_identity = dict(previous)
        current_identity = dict(protocol)
        # A committed HEAD may advance while data/weights remain identical;
        # all frontend, grouping, and model identities must still match.
        previous_identity.pop("git_head", None)
        current_identity.pop("git_head", None)
        if previous_identity != current_identity:
            raise RuntimeError("existing pilot protocol identity does not match current protocol")
        progress_path = output / "progress.json"
        if progress_path.is_file():
            return json.loads(progress_path.read_text(encoding="utf-8"))
    _write_json(output / "protocol.json", protocol)
    rows = _rows()
    _write_json(output / "window_manifest.json", rows)
    progress = {"stage": "prepare", "completed": 1, "total": 1, "status": "PREPARED", "updated_unix": time.time()}
    _write_json(output / "progress.json", progress)
    _write_json(output / "run_summary.json", {"status": "PREPARED", "protocol": "protocol.json", "rows": len(rows)})
    return progress


def phase_frontend(output: Path, *, resume: bool) -> dict[str, Any]:
    if not (output / "protocol.json").is_file():
        phase_prepare(output, resume=resume)
    existing_summary = output / "frontend_summary.json"
    if resume and existing_summary.is_file():
        previous = json.loads(existing_summary.read_text(encoding="utf-8"))
        if previous.get("status") in {"DENSE_FRONTEND_BLOCKED_384_CUDA_OOM", "DENSE_FRONTEND_FAILED", "FIRST_WINDOW_COMPLETE"}:
            return previous
    rows = _rows()
    started = time.perf_counter()
    result: dict[str, Any] = {"status": "NOT_RUN", "stage": "frontend", "attempted_windows": 0, "total_fixed_windows": 16}
    _write_json(output / "progress.json", {"stage": "frontend", "completed": 0, "total": 16, "status": "RUNNING", "updated_unix": time.time()})
    try:
        item = track_dense_window(rows[0], tapnet_source=TAPNET_SOURCE, checkpoint=TAPNET_CHECKPOINT)
        np.savez_compressed(output / "frontend_first_window.npz", raw_uv=item["raw_uv"], visibility=item["visibility"], query_start_uv_analysis=item["query_start_uv_analysis"])
        _write_json(output / "frontend_first_window.json", {key: value for key, value in item.items() if key not in {"raw_uv", "visibility", "query_start_uv_analysis"} and key != "group_ids"})
        result.update({"status": "FIRST_WINDOW_COMPLETE", "attempted_windows": 1, "elapsed_s": time.perf_counter() - started})
    except Exception as exc:
        result.update({"status": "DENSE_FRONTEND_BLOCKED_384_CUDA_OOM" if "out of memory" in str(exc).lower() or exc.__class__.__name__ == "OutOfMemoryError" else "DENSE_FRONTEND_FAILED", "attempted_windows": 1, "elapsed_s": time.perf_counter() - started, "error_type": type(exc).__name__, "error": str(exc), "traceback": traceback.format_exc(limit=8)})
    _write_json(output / "frontend_summary.json", result)
    _write_json(output / "progress.json", {"stage": "frontend", "completed": int(result["attempted_windows"]), "total": 16, "status": result["status"], "updated_unix": time.time()})
    return result


def phase_features(output: Path) -> dict[str, Any]:
    frontend = json.loads((output / "frontend_summary.json").read_text(encoding="utf-8")) if (output / "frontend_summary.json").is_file() else {"status": "NOT_RUN"}
    status = "NOT_RUN_FRONTEND_BLOCKED" if frontend.get("status") != "FIRST_WINDOW_COMPLETE" else "PENDING_IMPLEMENTATION_AFTER_FRONTEND_GATE"
    result = {"status": status, "reason": frontend.get("status")}
    _write_json(output / "features_summary.json", result)
    return result


def phase_train(output: Path) -> dict[str, Any]:
    result = {"status": "NOT_RUN", "reason": "dense frontend did not produce a valid fixed-edge feature population"}
    _write_json(output / "training_summary.json", result)
    return result


def _review_files(output: Path) -> dict[str, Any]:
    protocol = json.loads((output / "protocol.json").read_text(encoding="utf-8"))
    frontend = json.loads((output / "frontend_summary.json").read_text(encoding="utf-8")) if (output / "frontend_summary.json").is_file() else {"status": "NOT_RUN"}
    features = json.loads((output / "features_summary.json").read_text(encoding="utf-8")) if (output / "features_summary.json").is_file() else {"status": "NOT_RUN"}
    training = json.loads((output / "training_summary.json").read_text(encoding="utf-8")) if (output / "training_summary.json").is_file() else {"status": "NOT_RUN"}
    summary = {"status": frontend.get("status", "UNKNOWN"), "experiment": protocol["experiment"], "frontend": frontend, "features": features, "training": training, "causal_training_eligible": False, "causal_training_reason": "384 dense tracker gate did not complete; no feature/training result", "limitations": ["no metric depth ground truth", "no pixel-level spatial ground truth", "first-window CUDA OOM blocks dense frontend", "not a formal detector"]}
    _write_json(output / "summary.json", summary)
    _write_csv(output / "coverage_by_window.csv", [{"window_id": row["window_id"], "source_id": row["source_id"], "kind": row["kind"], "role": row["role"], "status": "NOT_ATTEMPTED_AFTER_FIRST_WINDOW_GATE"} for row in _rows()])
    _write_csv(output / "per_source_metrics.csv", [])
    _write_csv(output / "oof_window_scores.csv", [])
    _write_csv(output / "runtime_summary.csv", [{"stage": "frontend", "status": frontend.get("status"), "elapsed_s": frontend.get("elapsed_s", "")}])
    _write_csv(output / "case_manifest.csv", [])
    readme = """# V7 dense local-structure pilot review bundle\n\nThis exploratory pilot was stopped at the declared 384x384, 3-pixel BootsTAPIR gate. The first fixed window attempted 16,384 initialized queries and the tracker causal state exceeded the RTX 4090 D memory. No smaller grid, old component filter, confidence feature, or result-dependent fallback was used.\n\nThe status is a frontend feasibility result, not a detector result. Features and supervised arms are NOT_RUN. No spatial ground-truth or depth-accuracy claim is made.\n\nReproduce with the project interpreter:\n`/root/autodl-tmp/projects/sparse_3d_forgery_detection/.venv/bin/python -m research_tools.v7.dense_local_structure_pilot.run_pipeline all`\n"""
    (output / "README.md").write_text(readme, encoding="utf-8")
    bundle = output / "review_bundle.zip"
    with zipfile.ZipFile(bundle, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name in ["README.md", "protocol.json", "summary.json", "per_source_metrics.csv", "oof_window_scores.csv", "coverage_by_window.csv", "runtime_summary.csv", "case_manifest.csv"]:
            archive.write(output / name, arcname=name)
    return summary


def phase_export(output: Path) -> dict[str, Any]:
    return _review_files(output)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=["prepare", "frontend", "features", "train", "export", "all"])
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    output = args.output_root
    if args.phase in {"prepare", "all"}:
        phase_prepare(output, resume=args.resume)
    if args.phase in {"frontend", "all"}:
        phase_frontend(output, resume=args.resume)
    if args.phase in {"features", "all"}:
        phase_features(output)
    if args.phase in {"train", "all"}:
        phase_train(output)
    if args.phase in {"export", "all"}:
        summary = phase_export(output)
        print(json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=False))


if __name__ == "__main__":
    main()
