"""Run the unchanged V7 explicit-geometry/component probe on eight Vript clips."""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import platform
import time

import numpy as np

from research_tools.v7.feasibility.materialize_vript_frozen_targets import (
    build_time_window,
    contact_sheet,
    read_json,
    write_json,
)


OUTPUT_ROOT = Path("/root/autodl-tmp/data/sparse_3d_forgery_detection/derived/v7_vript_core_feasibility_v1")
DATA_REVISION = "acc278efb0ee249d646eef8a6b023595ca7efb93"
TAPNET_SHA = "c2cbab81cc06092b5f05bfe2da7bfec54e2079c9"
DEPTH_PRO_SHA = "9efe5c1def37a26c5367a71df664b18e1306c708"
DURATION_S = (0.5, 1.0, 2.0)


def build_window_manifest(media_validation: list[dict[str, object]], output: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for media in media_validation:
        if media.get("status") != "MEDIA_VALID":
            continue
        source_id = str(media["source_id"])
        for duration in DURATION_S:
            window = build_time_window(media["timestamps_s"], duration)
            rows.append(
                {
                    "source_id": source_id,
                    "video_path": media["video_path"],
                    "timescale_s": duration,
                    "window": window,
                    "population_marker": "UNREVIEWED_METADATA_CORE_CANDIDATES",
                    "real_only": True,
                    "fake_used": False,
                }
            )
    write_json(output / "manifest" / "window_manifest.json", rows)
    return rows


def _distribution(values: np.ndarray) -> dict[str, float | int | None]:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {"count": 0, "median": None, "iqr": None, "p10": None, "p90": None}
    return {
        "count": int(values.size),
        "median": float(np.median(values)),
        "iqr": float(np.percentile(values, 75) - np.percentile(values, 25)),
        "p10": float(np.percentile(values, 10)),
        "p90": float(np.percentile(values, 90)),
    }


def _decode(path: Path, source_id: str, indices: list[int]):
    from sparse3d_forgery.video_input import VideoSource, decode_video

    return decode_video(
        VideoSource(sample_id=f"v7-vript-{source_id}", source_video_id=source_id, source_locator=path),
        indices,
    )


def _tracking_metrics(visibility: np.ndarray) -> dict[str, object]:
    lifetimes = np.sum(visibility, axis=0)
    return {
        "requested_tracks": int(visibility.shape[1]),
        "visible_tracks_per_frame": np.sum(visibility, axis=1).astype(int).tolist(),
        "persistent_fraction": float(np.mean(np.all(visibility, axis=0))),
        "lifetime": _distribution(lifetimes),
    }


def _frontend_window(item: dict[str, object], tracker, depth_runner, output: Path) -> dict[str, object]:
    """Use the same providers and geometry helpers as the formal V7 script."""

    from scripts.run_v7_explicit_geometry_frontend import (
        accumulate_world_from_camera,
        causal_first_frame_intrinsics,
        persistent_track_ids,
        rgbd_odometry,
        sample_depth_at_uv,
        summarize_distribution,
        tracking_metrics,
        world_xyz,
    )
    from sparse3d_forgery.experiments.v7_dynamic_structure_probe import (
        ComponentConfig,
        motion_coherent_components,
        structural_differences,
        structure_state,
    )
    from sparse3d_forgery.particle_sequence import (
        CoordinateSystem,
        Handedness,
        LengthUnit,
        build_particle_sequence,
        save_particle_sequence,
    )

    started = time.perf_counter()
    path = Path(str(item["video_path"]))
    source_id = str(item["source_id"])
    window = item["window"]
    decoded = _decode(path, source_id, list(window["frame_indices"]))
    uv, visibility = tracker.track(decoded)
    depths, focals, frame_depth_valid = depth_runner.infer(decoded)
    intrinsics, fixed_focal_px = causal_first_frame_intrinsics(depths, focals)
    adjacent, pair_valid, information = rgbd_odometry(decoded, depths, intrinsics)
    world_from_camera, pose_valid = accumulate_world_from_camera(adjacent, pair_valid)
    sampled_depth, sampled_valid = sample_depth_at_uv(depths, uv)
    observation_valid = visibility & sampled_valid & frame_depth_valid[:, None]
    xyz, geometry_valid = world_xyz(uv, sampled_depth, intrinsics, world_from_camera, observation_valid, pose_valid)
    sequence = build_particle_sequence(
        decoded,
        track_ids=persistent_track_ids(uv.shape[1]),
        xyz=xyz,
        uv=uv,
        visibility=visibility,
        geometry_validity=geometry_valid,
        coordinate_system=CoordinateSystem(
            frame_name="first_camera_world",
            handedness=Handedness.RIGHT,
            axis_directions=("right", "down", "forward"),
            length_unit=LengthUnit.METER,
            camera_motion_compensated=True,
            normalization={},
        ),
        lineage={"dataset": "Vript", "revision": DATA_REVISION, "split": "train_real", "population": "UNREVIEWED_METADATA_CORE_CANDIDATES"},
        provenance={
            "tracker": "online_bootstapir",
            "tracker_source_sha": TAPNET_SHA,
            "depth": "apple_depth_pro",
            "depth_source_sha": DEPTH_PRO_SHA,
            "depth_semantics": "optical_axis_z_depth_interpretation_of_official_depth_output",
            "principal_point": "EXPERIMENTAL_CENTER_PRINCIPAL_POINT_ASSUMPTION",
            "distortion": "EXPERIMENTAL_ZERO_DISTORTION_ASSUMPTION",
            "intrinsics_policy": "CAUSAL_FIRST_FRAME_DEPTH_PRO_FOCAL_ASSUMPTION",
            "fixed_focal_px": fixed_focal_px,
            "pose": "open3d_0.19_adjacent_rgbd_odometry",
            "pose_convention": "target_camera_from_source_camera; inverted during world accumulation",
            "causal_execution": True,
            "frontend_baseline_unchanged": True,
        },
    )
    sequence_prefix = output / "particles" / str(item["timescale_s"]).replace(".", "p") / source_id.replace("/", "_")
    save_particle_sequence(sequence, sequence_prefix)

    config = ComponentConfig(max_initial_distance=1.0, max_relative_change=0.05, minimum_size=3, minimum_overlap=8)
    components = motion_coherent_components(xyz, geometry_valid, config)
    st_valid_total = ds_total = d2_total = 0
    st_denominator = 0
    d2_values: list[float] = []
    rows: list[dict[str, object]] = []
    for component in components:
        state, state_valid = structure_state(xyz, geometry_valid, component)
        first, first_valid, second, second_valid = structural_differences(state, state_valid, decoded.timestamps_s)
        st_valid_total += int(np.sum(state_valid))
        st_denominator += int(state_valid.size)
        ds_total += int(np.sum(first_valid))
        d2_total += int(np.sum(second_valid))
        d2_values.extend(np.linalg.norm(second[second_valid], axis=1).tolist())
        rows.append({"members": list(component), "size": len(component), "valid_s_states": int(np.sum(state_valid))})
    valid_step = geometry_valid[1:] & geometry_valid[:-1]
    steps = np.linalg.norm(xyz[1:] - xyz[:-1], axis=-1)
    steps[~valid_step] = np.nan
    extreme_threshold_m = 5.0
    finite_steps = steps[np.isfinite(steps)]
    return {
        "source_id": source_id,
        "timescale_s": float(item["timescale_s"]),
        "frame_indices": decoded.frame_indices.tolist(),
        "timestamps_s": decoded.timestamps_s.tolist(),
        "geometry_coverage": float(np.mean(geometry_valid)),
        "pose_chain_complete": bool(np.all(pair_valid)) if pair_valid.size else True,
        "pose_pair_valid_fraction": float(np.mean(pair_valid)) if pair_valid.size else 1.0,
        "tracking": _tracking_metrics(visibility),
        "component_count": len(components),
        "component_sizes": [len(component) for component in components],
        "component_particle_fraction": float(len({i for component in components for i in component}) / xyz.shape[1]),
        "s_t_valid_frame_fraction": float(st_valid_total / st_denominator) if st_denominator else 0.0,
        "valid_s_states": st_valid_total,
        "valid_delta_s": ds_total,
        "valid_delta2_s": d2_total,
        "delta2_s_magnitude": _distribution(np.asarray(d2_values)),
        "same_track_xyz_step_m": _distribution(finite_steps),
        "extreme_xyz_step_threshold_m": extreme_threshold_m,
        "extreme_xyz_step_rate": float(np.mean(finite_steps > extreme_threshold_m)) if finite_steps.size else 0.0,
        "components": rows,
        "particle_prefix": str(sequence_prefix),
        "elapsed_s": time.perf_counter() - started,
    }


def summarize_timescales(rows: list[dict[str, object]]) -> dict[str, object]:
    summary: dict[str, object] = {}
    for duration in DURATION_S:
        group = [row for row in rows if row.get("timescale_s") == duration and row.get("status", "COMPLETE") == "COMPLETE"]
        if not group:
            summary[str(duration)] = {"windows": 0}
            continue
        coverage = np.asarray([row["geometry_coverage"] for row in group], dtype=float)
        persistence = np.asarray([row["tracking"]["persistent_fraction"] for row in group], dtype=float)
        d2 = np.asarray([row["valid_delta2_s"] for row in group], dtype=float)
        summary[str(duration)] = {
            "windows": len(group),
            "frontend_complete": len(group),
            "geometry_coverage": _distribution(coverage),
            "pose_chain_success_rate": float(np.mean([row["pose_chain_complete"] for row in group])),
            "tracking_persistence": _distribution(persistence),
            "windows_with_component": int(sum(row["component_count"] > 0 for row in group)),
            "no_component_rate": float(np.mean([row["component_count"] == 0 for row in group])),
            "component_count": _distribution(np.asarray([row["component_count"] for row in group])),
            "component_particle_fraction": _distribution(np.asarray([row["component_particle_fraction"] for row in group])),
            "component_size": _distribution(np.asarray([size for row in group for size in row["component_sizes"]], dtype=float)),
            "s_t_valid_frame_rate": _distribution(np.asarray([row["s_t_valid_frame_fraction"] for row in group])),
            "valid_delta_s": int(np.sum([row["valid_delta_s"] for row in group])),
            "valid_delta2_s": int(np.sum([row["valid_delta2_s"] for row in group])),
            "delta2_s_magnitude": _distribution(np.asarray([value for row in group for value in [row["delta2_s_magnitude"].get("median")] if value is not None])),
            "extreme_xyz_step_rate": _distribution(np.asarray([row["extreme_xyz_step_rate"] for row in group])),
        }
    return summary


def run_probe(output: Path, *, tapnet_source: Path, tapnet_checkpoint: Path, depth_source: Path, depth_checkpoint: Path) -> dict[str, object]:
    media_validation = read_json(output / "manifest" / "media_validation.json")
    if not isinstance(media_validation, list):
        raise ValueError("media_validation.json must contain a list")
    valid = [row for row in media_validation if row.get("status") == "MEDIA_VALID"][:8]
    if len(valid) < 8:
        raise RuntimeError(f"VRIPT_MEDIA_ACCESS_BLOCKED: only {len(valid)} MEDIA_VALID videos")
    review_rows = []
    for row in valid:
        sheet = output / "review" / f"{row['source_id'].replace(':', '_')}.png"
        row["contact_sheet"] = contact_sheet(Path(row["video_path"]), sheet)
        review_rows.append(row)
    write_json(output / "manifest" / "probe_population.json", {"marker": "UNREVIEWED_METADATA_CORE_CANDIDATES", "source_ids": [row["source_id"] for row in valid], "fake_count": 0})
    windows = build_window_manifest(valid, output)

    from scripts.run_v7_explicit_geometry_frontend import DepthProRunner, OnlineBootsTapir
    import torch

    tracker = OnlineBootsTapir(tapnet_source, tapnet_checkpoint)
    depth_runner = DepthProRunner(depth_source, depth_checkpoint)
    rows: list[dict[str, object]] = []
    for number, item in enumerate(windows, 1):
        if item["window"]["status"] != "AVAILABLE":
            rows.append({**item, "status": "WINDOW_DURATION_UNAVAILABLE"})
            continue
        print(f"frontend {number}/{len(windows)} {item['source_id']} {item['timescale_s']}s", flush=True)
        try:
            rows.append({**item, **_frontend_window(item, tracker, depth_runner, output), "status": "COMPLETE"})
        except Exception as exc:
            rows.append({**item, "status": "FRONTEND_FAILURE", "error": f"{type(exc).__name__}: {exc}"})
    del tracker, depth_runner
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    write_json(output / "metrics" / "window_metrics.json", rows)
    summary = summarize_timescales(rows)
    write_json(output / "metrics" / "timescale_summary.json", summary)
    paired = []
    for source_id in [row["source_id"] for row in valid]:
        paired.append({"source_id": source_id, "timescales": {str(d): next((row for row in rows if row["source_id"] == source_id and row["timescale_s"] == d), None) for d in DURATION_S}})
    write_json(output / "metrics" / "per_video_timescale_comparison.json", paired)
    completed = [row for row in rows if row.get("status") == "COMPLETE"]
    supported = False
    for duration in DURATION_S:
        group = [row for row in completed if row["timescale_s"] == duration]
        with_component = [row for row in group if row["component_count"] > 0]
        supported |= (
            len({row["source_id"] for row in with_component}) >= 6
            and np.median([row["s_t_valid_frame_fraction"] for row in with_component]) >= 0.75
            and sum(row["valid_delta2_s"] >= 3 for row in group) >= 6
            and np.median([row["geometry_coverage"] for row in group]) >= 0.80
            and np.median([row["extreme_xyz_step_rate"] for row in group]) < 0.5
        )
    status = "SUPPORTED" if supported else "INCONCLUSIVE"
    result = {"status": status, "timescales": summary, "population": [row["source_id"] for row in valid], "fake_count": 0, "normality_training": False}
    write_json(output / "run_summary.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--tapnet-source", type=Path, required=True)
    parser.add_argument("--tapnet-checkpoint", type=Path, required=True)
    parser.add_argument("--depth-source", type=Path, required=True)
    parser.add_argument("--depth-checkpoint", type=Path, required=True)
    args = parser.parse_args()
    result = run_probe(args.output, tapnet_source=args.tapnet_source, tapnet_checkpoint=args.tapnet_checkpoint, depth_source=args.depth_source, depth_checkpoint=args.depth_checkpoint)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
