"""Compare one official YOLO26 depth model with the frozen Depth Pro measurement.

This is a small, independent measurement probe.  It reuses the already saved
289-point UV/visibility observations and the existing Depth Pro intrinsics and
pose convention.  It does not train a detector, alter the V7 representation,
or treat either depth map as ground truth.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from sparse3d_forgery.experiments.v7_explicit_geometry_frontend import (
    accumulate_world_from_camera,
    causal_first_frame_intrinsics,
    sample_depth_at_uv,
    world_xyz,
)
from sparse3d_forgery.particle_sequence import (
    load_particle_sequence,
    validate_particle_sequence,
)
from sparse3d_forgery.video_input import VideoSource, decode_video
from research_tools.v7.local_structural_temporal_probe.representation import build_window_support
from research_tools.v7.observation_density_diagnostic.diagnostics import (
    component_diagnostic,
)


DATA_ROOT = Path("/root/autodl-tmp/data/sparse_3d_forgery_detection")
SOURCE_ROOT = DATA_ROOT / "derived/v7_activityforensics_observation_density_diagnostic_v1"
FRONTEND_ROOT = DATA_ROOT / "derived/v7_activityforensics_density_matched_frontend_v1"
OUTPUT_ROOT = DATA_ROOT / "derived/v7_activityforensics_yolo26_depth_comparison_v1"
YOLO_ROOT = DATA_ROOT / "external/yolo26_depth"
DEFAULT_WEIGHT = YOLO_ROOT / "yolo26m-depth.pt"
DEPTH_SOURCE = DATA_ROOT / "external/v7_explicit_geometry/ml-depth-pro-9efe5c1def37a26c5367a71df664b18e1306c708"
DEPTH_CHECKPOINT = DATA_ROOT / "external/v7_explicit_geometry/checkpoints/depth_pro.pt"
OFFICIAL_URL = "https://docs.ultralytics.com/tasks/depth/"
YOLO_MODEL = "yolo26m-depth"
INPUT_SIZE = 768


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    raise TypeError(type(value).__name__)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False, default=_json_default)
        + "\n",
        encoding="utf-8",
    )


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalize_yolo_depth(result: Any, expected_hw: tuple[int, int]) -> np.ndarray:
    """Return the official ``result.depth.data`` as finite-positive float32 pixels."""

    depth_object = getattr(result, "depth", None)
    if depth_object is None or not hasattr(depth_object, "data"):
        raise ValueError("YOLO result has no depth.data output")
    data = depth_object.data
    if hasattr(data, "detach"):
        data = data.detach().float().cpu().numpy()
    array = np.asarray(data)
    while array.ndim > 2 and array.shape[0] == 1:
        array = array[0]
    if array.ndim != 2 or tuple(array.shape) != tuple(expected_hw):
        raise ValueError(f"YOLO depth shape {array.shape} does not match source raster {expected_hw}")
    array = array.astype(np.float32, copy=False)
    if not np.all(np.isfinite(array[array > 0])):
        raise ValueError("YOLO depth contains non-finite positive output")
    return array


def fixed_depth_limits(depth_maps: Iterable[np.ndarray]) -> tuple[float, float] | None:
    """Choose one fixed display range over all frames and both depth methods."""

    values = [np.asarray(item, dtype=np.float64).ravel() for item in depth_maps]
    finite = np.concatenate([item[np.isfinite(item) & (item > 0)] for item in values]) if values else np.empty(0)
    if finite.size == 0:
        return None
    low, high = np.percentile(finite, [2.0, 98.0])
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        low, high = float(np.min(finite)), float(np.max(finite))
    if high <= low:
        high = low + 1.0
    return float(low), float(high)


def colorize_depth(depth: np.ndarray, limits: tuple[float, float] | None) -> np.ndarray:
    """Colorize with a fixed range; invalid pixels remain black."""

    import cv2

    if limits is None:
        return np.zeros((*depth.shape, 3), dtype=np.uint8)
    low, high = limits
    finite = np.isfinite(depth) & (depth > 0)
    normalized = np.zeros(depth.shape, dtype=np.uint8)
    normalized[finite] = np.clip((depth[finite] - low) / (high - low) * 255.0, 0, 255).astype(np.uint8)
    colored = cv2.applyColorMap(normalized, cv2.COLORMAP_TURBO)
    colored[~finite] = 0
    return colored


def _depth_stats(depth: np.ndarray, sampled: np.ndarray, sampled_valid: np.ndarray) -> dict[str, Any]:
    finite = np.isfinite(depth) & (depth > 0)
    frame_medians = [float(np.median(depth[index][finite[index]])) if np.any(finite[index]) else None for index in range(depth.shape[0])]
    sampled_values = sampled[sampled_valid & np.isfinite(sampled)]
    return {
        "map_shape": [int(x) for x in depth.shape[1:]],
        "finite_positive_fraction": float(np.mean(finite)),
        "finite_positive_count": int(np.sum(finite)),
        "min_m": float(np.min(depth[finite])) if np.any(finite) else None,
        "median_m": float(np.median(depth[finite])) if np.any(finite) else None,
        "p95_m": float(np.percentile(depth[finite], 95)) if np.any(finite) else None,
        "frame_median_m": frame_medians,
        "frame_median_ratio_max_min": (
            float(max(x for x in frame_medians if x is not None) / min(x for x in frame_medians if x is not None))
            if any(x is not None and x > 0 for x in frame_medians) and sum(x is not None and x > 0 for x in frame_medians) > 1
            else None
        ),
        "same_query_sample_count": int(sampled_values.size),
        "same_query_sample_median_m": float(np.median(sampled_values)) if sampled_values.size else None,
    }


def _pair_curve(
    sequence: Any,
    support: Mapping[str, Any],
    *,
    pair_reference: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Measure fixed triplet pair identities over the full window.

    ``pair_reference`` lets the depth replacement use exactly the same
    Depth-Pro-selected pair identities, so a changed component graph cannot be
    mistaken for a depth-only curve difference.
    """

    triplets = list((pair_reference or support).get("triplets", []))
    if not triplets:
        return {"triplet_id": None, "pair_count": 0, "values_m": [None] * int(sequence.xyz.shape[0]), "valid_count": 0}
    triplet = triplets[0]
    pairs = [tuple(int(x) for x in pair) for pair in triplet.get("pair_indices", [])]
    xyz = np.asarray(sequence.xyz, dtype=np.float64)
    values: list[float | None] = []
    for frame in range(xyz.shape[0]):
        distances = []
        for left, right in pairs:
            if np.all(np.isfinite(xyz[frame, left])) and np.all(np.isfinite(xyz[frame, right])):
                distances.append(float(np.linalg.norm(xyz[frame, right] - xyz[frame, left])))
        values.append(float(np.median(distances)) if distances else None)
    return {
        "triplet_id": int(triplet.get("triplet_id", 0)),
        "pair_count": len(pairs),
        "pair_track_ids": [[int(x) for x in pair] for pair in triplet.get("pair_ids", [])],
        "source_frame_indices": [int(x) for x in sequence.frame_indices],
        "timestamps_s": [float(x) for x in sequence.timestamps_s],
        "values_m": values,
        "valid_count": int(sum(value is not None for value in values)),
    }


def _component_stats(sequence: Any, window_start_s: float, label: str) -> tuple[dict[str, Any], dict[str, Any]]:
    diagnostic = component_diagnostic(sequence, window_start_s=window_start_s, label=label)
    summary = {
        "component_count": int(diagnostic["component_count"]),
        "max_component_members": int(diagnostic["max_component_members"]),
        "component_member_count_total": int(diagnostic["component_member_count_total"]),
        "qualified_graph_edge_count": int(diagnostic["qualified_graph_edge_count"]),
        "component_internal_pair_count": int(diagnostic["component_internal_pair_count"]),
        "valid_triplet_count": int(diagnostic["valid_triplet_count"]),
        "support_status": str(diagnostic["support_status"]),
        "component_members": diagnostic.get("component_members", []),
    }
    return summary, diagnostic


def _decode(row: Mapping[str, Any]) -> Any:
    return decode_video(
        VideoSource(
            sample_id=f"v7-yolo26-{row['window_id']}",
            source_video_id=f"{row['source_id']}-{row['role']}",
            source_locator=Path(str(row["video_path"])),
        ),
        [int(x) for x in row["frame_indices"]],
    )


def _draw_points(image: np.ndarray, sequence: Any, diagnostic: Mapping[str, Any], frame_index: int, *, label: str) -> np.ndarray:
    import cv2

    canvas = image.copy()
    component_by_slot: dict[int, int] = {}
    for component_index, members in enumerate(diagnostic.get("component_members", [])):
        for slot in members:
            component_by_slot[int(slot)] = component_index
    for slot, uv in enumerate(np.asarray(sequence.uv)[frame_index]):
        if not np.all(np.isfinite(uv)):
            continue
        if not bool(sequence.visibility[frame_index, slot]):
            continue
        if not bool(sequence.geometry_validity[frame_index, slot]):
            continue
        x, y = int(round(float(uv[0]))), int(round(float(uv[1])))
        component = component_by_slot.get(slot)
        color = (255, 255, 255) if component is None else ((0, 220, 255) if component % 2 == 0 else (255, 80, 180))
        cv2.circle(canvas, (x, y), 3, color, -1, lineType=cv2.LINE_AA)
    cv2.putText(canvas, label, (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    return canvas


def _write_media(
    output: Path,
    source_id: str,
    decoded: Any,
    depth_pro: np.ndarray,
    depth_yolo: np.ndarray,
    sequence: Any,
    yolo_sequence: Any,
    pro_diag: Mapping[str, Any],
    yolo_diag: Mapping[str, Any],
) -> dict[str, Any]:
    import cv2

    limits = fixed_depth_limits([depth_pro, depth_yolo])
    media_dir = output / "media"
    media_dir.mkdir(parents=True, exist_ok=True)
    path = media_dir / f"{source_id}_manip_real_comparison.mp4"
    first_rgb = np.asarray(decoded.frames[0].rgb)
    target_h = 256
    panel_w = max(1, round(first_rgb.shape[1] * target_h / first_rgb.shape[0]))
    canvas_size = (panel_w * 4, target_h)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 8.0, canvas_size)
    if not writer.isOpened():
        raise RuntimeError(f"cannot create comparison media: {path}")
    for frame_index, frame in enumerate(decoded.frames):
        rgb = cv2.cvtColor(np.asarray(frame.rgb), cv2.COLOR_RGB2BGR)
        panels = [
            _draw_points(rgb, sequence, pro_diag, frame_index, label="RGB + Depth Pro geometry"),
            _draw_points(colorize_depth(depth_pro[frame_index], limits), sequence, pro_diag, frame_index, label="Depth Pro (fixed range)"),
            _draw_points(colorize_depth(depth_yolo[frame_index], limits), yolo_sequence, yolo_diag, frame_index, label="YOLO26m-depth (fixed range)"),
            _draw_points(rgb, yolo_sequence, yolo_diag, frame_index, label="RGB + YOLO geometry"),
        ]
        resized = [cv2.resize(panel, (panel_w, target_h), interpolation=cv2.INTER_AREA) for panel in panels]
        writer.write(np.concatenate(resized, axis=1))
    writer.release()
    return {"path": str(path), "source_id": source_id, "frames": len(decoded.frames), "fixed_color_range_m": list(limits) if limits else None}


def _fixed_rows() -> list[dict[str, Any]]:
    rows = read_json(SOURCE_ROOT / "window_manifest.json")
    if not isinstance(rows, list) or len(rows) != 16:
        raise ValueError("fixed density diagnostic manifest must contain 16 windows")
    return [dict(row) for row in rows]


def _target_sequence_prefix(row: Mapping[str, Any]) -> Path:
    safe = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in str(row["window_id"]))
    return FRONTEND_ROOT / "particles" / f"{safe}__density289"


def run_probe(weight: Path = DEFAULT_WEIGHT, output: Path = OUTPUT_ROOT) -> dict[str, Any]:
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(output)
    if not weight.is_file():
        raise FileNotFoundError(weight)
    rows = _fixed_rows()
    output.mkdir(parents=True, exist_ok=True)
    protocol = {
        "experiment": "V7 YOLO26m-depth independent measurement comparison",
        "official_reference": OFFICIAL_URL,
        "model": YOLO_MODEL,
        "input_size": INPUT_SIZE,
        "population": "fixed four sources, earliest existing density-diagnostic MANIP and CTRL windows, both roles (16 windows)",
        "inputs": "saved nested 289-point UV and visibility; no re-tracking",
        "pose_and_intrinsics": "one existing Depth Pro-derived causal-first-frame intrinsics and Open3D pose per window, applied to both depth maps",
        "pose_reuse_caveat": "historical ParticleSequence artifacts do not persist pose matrices; the existing helper deterministically recomputes them once per window and shares that result",
        "depth": "official result.depth.data interpreted directly as positive metric depth; no per-frame alignment or fitting",
        "coordinate": "first_camera_world, right-handed right/down/forward, meter; existing pose convention",
        "limitations": ["no metric depth ground truth", "Depth Pro intrinsics/pose remain fixed dependencies", "measurement comparison, not detector training"],
    }
    write_json(output / "protocol.json", protocol)
    write_json(output / "weight.json", {"path": str(weight), "bytes": weight.stat().st_size, "sha256": sha256(weight)})

    import torch
    import ultralytics
    from ultralytics import YOLO
    from scripts.run_v7_explicit_geometry_frontend import DepthProRunner, rgbd_odometry

    model = YOLO(str(weight))
    depth_runner = DepthProRunner(DEPTH_SOURCE, DEPTH_CHECKPOINT)
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    results: list[dict[str, Any]] = []
    media_candidates: dict[str, dict[str, Any]] = {}
    media_selection = sorted(rows, key=lambda row: (str(row["source_id"]), 0 if row["kind"] == "MANIP" else 1, 0 if row["role"] == "real" else 1))
    selected_media_ids = {str(row["window_id"]) for row in media_selection[::4]} if len(media_selection) >= 4 else set()
    for number, row in enumerate(rows, 1):
        window_started = time.perf_counter()
        prefix = _target_sequence_prefix(row)
        if not prefix.with_suffix(".npz").is_file():
            raise FileNotFoundError(f"289-point matched sequence missing: {prefix}")
        sequence = load_particle_sequence(prefix)
        decoded = _decode(row)
        if not np.array_equal(sequence.frame_indices, decoded.frame_indices):
            raise ValueError(f"frame identity mismatch for {row['window_id']}")
        if not np.allclose(sequence.timestamps_s, decoded.timestamps_s, rtol=0, atol=1e-7):
            raise ValueError(f"PTS mismatch for {row['window_id']}")
        depth_pro, focals, frame_depth_valid = depth_runner.infer(decoded)
        intrinsics, fixed_focal = causal_first_frame_intrinsics(depth_pro, focals)
        adjacent, pose_pair_valid, _information = rgbd_odometry(decoded, depth_pro, intrinsics)
        world_from_camera, pose_valid = accumulate_world_from_camera(adjacent, pose_pair_valid)

        depth_yolo_frames: list[np.ndarray] = []
        for frame in decoded.frames:
            predictions = model.predict(source=np.asarray(frame.rgb), imgsz=INPUT_SIZE, device=0, verbose=False)
            if not predictions:
                raise RuntimeError(f"YOLO returned no result for {row['window_id']}")
            depth_yolo_frames.append(normalize_yolo_depth(predictions[0], tuple(frame.rgb.shape[:2])))
        depth_yolo = np.stack(depth_yolo_frames).astype(np.float32, copy=False)

        pro_sampled, pro_sample_valid = sample_depth_at_uv(depth_pro, sequence.uv)
        yolo_sampled, yolo_sample_valid = sample_depth_at_uv(depth_yolo, sequence.uv)
        same_query = np.asarray(sequence.visibility, dtype=bool) & np.all(np.isfinite(sequence.uv), axis=-1)
        pro_obs = same_query & pro_sample_valid & frame_depth_valid[:, None]
        yolo_obs = same_query & yolo_sample_valid
        yolo_xyz, yolo_geometry = world_xyz(
            sequence.uv,
            yolo_sampled,
            intrinsics,
            world_from_camera,
            yolo_obs,
            pose_valid,
        )
        yolo_sequence = replace(
            sequence,
            xyz=yolo_xyz,
            geometry_validity=yolo_geometry,
            provenance={**sequence.provenance, "depth_comparison": "yolo26m-depth", "yolo_depth_semantics": "official result.depth.data", "yolo_input_size": INPUT_SIZE},
        )
        validate_particle_sequence(yolo_sequence)
        pro_support = build_window_support(sequence, window_start_s=float(row["interval_start_s"]))
        yolo_support = build_window_support(yolo_sequence, window_start_s=float(row["interval_start_s"]))
        pro_component, pro_diag = _component_stats(sequence, float(row["interval_start_s"]), "depth_pro")
        yolo_component, yolo_diag = _component_stats(yolo_sequence, float(row["interval_start_s"]), "yolo26m_depth")
        result = {
            "window_id": str(row["window_id"]),
            "source_id": str(row["source_id"]),
            "kind": str(row["kind"]),
            "role": str(row["role"]),
            "source_video_path": str(row["video_path"]),
            "frame_indices": [int(x) for x in sequence.frame_indices],
            "timestamps_s": [float(x) for x in sequence.timestamps_s],
            "query_count": int(sequence.num_tracks),
            "same_query_visibility_fraction": float(np.mean(same_query)),
            "depth_pro_support_fraction_on_same_query": float(np.mean(pro_obs[same_query])) if np.any(same_query) else 0.0,
            "yolo_support_fraction_on_same_query": float(np.mean(yolo_obs[same_query])) if np.any(same_query) else 0.0,
            "depth_pro": _depth_stats(depth_pro, pro_sampled, pro_sample_valid),
            "yolo26m_depth": _depth_stats(depth_yolo, yolo_sampled, yolo_sample_valid),
            "pose_valid_fraction": float(np.mean(pose_valid)),
            "fixed_focal_px": float(fixed_focal),
            "intrinsics_policy": "causal_first_frame_depth_pro_focal_assumption",
            "pose_reused_for_both": True,
            "depth_pro_components": pro_component,
            "yolo_components": yolo_component,
            "pair_curve_depth_pro": _pair_curve(sequence, pro_support),
            "pair_curve_yolo26m_depth": _pair_curve(yolo_sequence, yolo_support, pair_reference=pro_support),
            "elapsed_s": time.perf_counter() - window_started,
            "causal_training_eligible": False,
            "causal_training_reason": "independent depth measurement comparison; no causal target construction",
        }
        results.append(result)
        if str(row["window_id"]) in selected_media_ids:
            media_candidates[str(row["source_id"])] = _write_media(
                output,
                str(row["source_id"]),
                decoded,
                depth_pro,
                depth_yolo,
                sequence,
                yolo_sequence,
                pro_diag,
                yolo_diag,
            )
        print(f"yolo26 {number}/{len(rows)} {row['window_id']} pro={result['depth_pro_support_fraction_on_same_query']:.3f} yolo={result['yolo_support_fraction_on_same_query']:.3f}", flush=True)

    summary = {
        "status": "YOLO26_DEPTH_COMPARISON_COMPLETE",
        "experiment": protocol["experiment"],
        "windows": len(results),
        "sources": sorted({str(row["source_id"]) for row in rows}),
        "results": results,
        "media": list(media_candidates.values()),
        "weight": {"path": str(weight), "sha256": sha256(weight), "model": YOLO_MODEL},
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "ultralytics": ultralytics.__version__,
            "input_size": INPUT_SIZE,
        },
        "elapsed_s": time.perf_counter() - started,
        "peak_gpu_bytes": int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else 0,
        "interpretation": {
            "absolute_accuracy": "not measured; no metric depth ground truth",
            "comparison": "reports finite support, direct depth scale behavior, pair-distance curves and component diagnostics",
            "depth_pro_is_ground_truth": False,
        },
    }
    write_json(output / "window_results.json", {"windows": results})
    write_csv(output / "window_metrics.csv", [
        {
            "window_id": row["window_id"],
            "source_id": row["source_id"],
            "kind": row["kind"],
            "role": row["role"],
            "pro_same_query_support": row["depth_pro_support_fraction_on_same_query"],
            "yolo_same_query_support": row["yolo_support_fraction_on_same_query"],
            "pro_component_count": row["depth_pro_components"]["component_count"],
            "yolo_component_count": row["yolo_components"]["component_count"],
            "pro_max_component_members": row["depth_pro_components"]["max_component_members"],
            "yolo_max_component_members": row["yolo_components"]["max_component_members"],
            "pose_valid_fraction": row["pose_valid_fraction"],
            "elapsed_s": row["elapsed_s"],
        }
        for row in results
    ])
    write_json(output / "media_manifest.json", {"media": list(media_candidates.values()), "fixed_color_range": "per-video range computed once over both methods, not per frame"})
    write_json(output / "summary.json", summary)
    write_json(output / "run_summary.json", {"status": summary["status"], "summary": "summary.json", "windows": len(results)})
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weight", type=Path, default=DEFAULT_WEIGHT)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    args = parser.parse_args()
    print(json.dumps(run_probe(args.weight, args.output_root), indent=2, sort_keys=True, ensure_ascii=False))


if __name__ == "__main__":
    main()
