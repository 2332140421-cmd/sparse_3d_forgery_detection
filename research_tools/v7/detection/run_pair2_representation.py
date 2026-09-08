"""Run the unchanged V7 frontend and components on frozen Pair2 media."""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import platform
import time

import numpy as np

from research_tools.v7.normality.frontend import (
    DepthProRunner,
    OnlineBootsTapir,
    run_frontend_window,
)
from sparse3d_forgery.particle_sequence import load_particle_sequence

from .protocol import (
    DATA_ROOT,
    build_pair2_window_manifest,
    file_identity,
    read_json,
    write_json,
)


DEFAULT_OUTPUT = DATA_ROOT / "derived/v7_pair2_detection_pilot_v1"
TAPNET_SHA = "c2cbab81cc06092b5f05bfe2da7bfec54e2079c9"
DEPTH_PRO_SHA = "9efe5c1def37a26c5367a71df664b18e1306c708"


def _patch_pair2_metadata(result: dict[str, object]) -> None:
    """Keep the frozen numeric artifact and correct only experiment lineage."""

    window = result["window"]
    prefix = Path(str(result["particle_prefix"]))
    metadata_path = prefix.with_suffix(".json")
    with metadata_path.open(encoding="utf-8") as handle:
        metadata = json.load(handle)
    metadata["lineage"] = {
        "dataset": "GenVidBench",
        "revision": str(window.get("revision", "701cafb6f999d7ea0cbf3c354df6177311a4d824")),
        "official_split": "Pair2",
        "population": "V7_PAIR2_REAL_FAKE_DETECTION_PILOT",
        "source_id": str(window["source_id"]),
        "source_identity": window.get("source_identity"),
        "pair_key": window.get("pair_key"),
        "role": str(window["role"]),
        "generator": window.get("generator"),
        "pair_lineage": window.get("pair_lineage"),
    }
    metadata["provenance"]["pair2_experiment"] = {
        "role": str(window["role"]),
        "generator": window.get("generator"),
        "official_relative_path": window.get("official_relative_path"),
        "causal_training_eligible": bool(result.get("causal_training_eligible", False)),
    }
    write_json(metadata_path, metadata)
    load_particle_sequence(prefix)


def _summary(rows: list[dict[str, object]]) -> dict[str, object]:
    if not rows:
        return {"videos": 0, "windows": 0}
    geometry = [float(row["coverage"]["final_geometry_valid_fraction"]) for row in rows]
    components = [int(row["component_count"]) for row in rows]
    total_slots = sum(len(row["timestamps_s"]) * int(row["component_count"]) for row in rows)
    valid_states = sum(int(row["valid_s_states"]) for row in rows)
    return {
        "windows": len(rows),
        "videos": len({str(row["window"]["video_id"]) for row in rows}),
        "complete_windows": sum(row.get("status") == "COMPLETE" for row in rows),
        "windows_with_component": sum(value > 0 for value in components),
        "component_success_rate": float(np.mean(np.asarray(components) > 0)),
        "geometry_coverage_median": float(np.median(geometry)),
        "geometry_coverage_iqr": float(np.percentile(geometry, 75) - np.percentile(geometry, 25)),
        "geometry_coverage_p10": float(np.percentile(geometry, 10)),
        "geometry_coverage_p90": float(np.percentile(geometry, 90)),
        "s_t_valid_states": valid_states,
        "s_t_total_state_slots": total_slots,
        "s_t_valid_rate": float(valid_states / max(1, total_slots)),
        "delta_s_observations": int(sum(int(row["valid_delta_s"]) for row in rows)),
        "delta2_s_observations": int(sum(int(row["valid_delta2_s"]) for row in rows)),
        "elapsed_s_total": float(sum(float(row["elapsed_s"]) for row in rows)),
        "peak_gpu_memory_bytes": max((row.get("peak_gpu_memory_bytes") or 0 for row in rows), default=0),
        "frontend_unchanged": True,
        "causal_execution": True,
    }


def run_representation(
    *,
    output: Path,
    tapnet_source: Path,
    tapnet_checkpoint: Path,
    depth_source: Path,
    depth_checkpoint: Path,
    media_manifest_path: Path | None = None,
) -> dict[str, object]:
    manifest = read_json(media_manifest_path or (output / "manifests" / "pair2_media_manifest.json"))
    if not isinstance(manifest, list):
        raise ValueError("Pair2 media manifest must be a list")
    windows = build_pair2_window_manifest(manifest)
    write_json(output / "manifests" / "pair2_window_manifest.json", windows)
    unavailable = [row for row in windows if row["status"] != "AVAILABLE"]
    runnable = [row for row in windows if row["status"] == "AVAILABLE"]
    if not runnable:
        raise RuntimeError("PAIR2_MEDIA_HAS_NO_AVAILABLE_WINDOWS")
    tracker = OnlineBootsTapir(tapnet_source, tapnet_checkpoint)
    depth_runner = DepthProRunner(depth_source, depth_checkpoint)
    rows: list[dict[str, object]] = []
    started = time.perf_counter()
    for index, item in enumerate(runnable, 1):
        print(f"frontend {index}/{len(runnable)} {item['window_id']}", flush=True)
        result = run_frontend_window(item, tracker, depth_runner, output)
        _patch_pair2_metadata(result)
        rows.append(result)
        write_json(output / "frontend" / "window_results_progress.json", rows)
        print(
            f"  role={item['role']} generator={item.get('generator')} "
            f"geometry={result['coverage']['final_geometry_valid_fraction']:.3f} "
            f"components={result['component_count']} elapsed={result['elapsed_s']:.1f}s "
            f"peak_gpu={result['peak_gpu_memory_bytes']}",
            flush=True,
        )
    del tracker, depth_runner
    gc.collect()
    write_json(output / "frontend" / "window_results.json", rows)
    summaries = {
        role: _summary([row for row in rows if str(row["window"]["role"]) == role])
        for role in ("real", "fake")
    }
    all_rows = _summary(rows)
    meta = {
        "dataset": "GenVidBench",
        "revision": "701cafb6f999d7ea0cbf3c354df6177311a4d824",
        "windows_requested": len(windows),
        "windows_run": len(rows),
        "windows_unavailable": len(unavailable),
        "elapsed_s": time.perf_counter() - started,
        "causal_execution": True,
        "frontend_unchanged": True,
        "fake_used_for_fitting": False,
        "python": platform.python_version(),
        "providers": {
            "tapnet_source": str(tapnet_source),
            "tapnet_source_sha": TAPNET_SHA,
            "tapnet_checkpoint": file_identity(tapnet_checkpoint),
            "depth_source": str(depth_source),
            "depth_source_sha": DEPTH_PRO_SHA,
            "depth_checkpoint": file_identity(depth_checkpoint),
        },
        "media_summary": summaries,
        "all_summary": all_rows,
    }
    write_json(output / "frontend" / "run_meta.json", meta)
    write_json(output / "metrics" / "frontend_summary.json", meta)
    return {"windows": rows, "unavailable": unavailable, "meta": meta}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--tapnet-source", type=Path, required=True)
    parser.add_argument("--tapnet-checkpoint", type=Path, required=True)
    parser.add_argument("--depth-source", type=Path, required=True)
    parser.add_argument("--depth-checkpoint", type=Path, required=True)
    parser.add_argument("--media-manifest", type=Path, required=False)
    args = parser.parse_args()
    result = run_representation(
        output=args.output,
        tapnet_source=args.tapnet_source,
        tapnet_checkpoint=args.tapnet_checkpoint,
        depth_source=args.depth_source,
        depth_checkpoint=args.depth_checkpoint,
        media_manifest_path=args.media_manifest,
    )
    print(json.dumps(result["meta"], indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
