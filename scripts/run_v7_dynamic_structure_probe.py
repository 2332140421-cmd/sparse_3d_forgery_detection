#!/usr/bin/env python3
"""Summarize V7 frontend runs and execute the minimal real-only structure probe."""

import argparse
import json
from pathlib import Path

import numpy as np

from sparse3d_forgery.experiments.v7_dynamic_structure_probe import (
    ComponentConfig,
    motion_coherent_components,
    structural_differences,
    structure_state,
)
from sparse3d_forgery.experiments.v7_explicit_geometry_frontend import summarize_distribution
from sparse3d_forgery.particle_sequence import load_particle_sequence


def write_json(path: Path, value: object) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def load_run(root: Path, name: str) -> dict:
    with (root / f"run_{name}.json").open(encoding="utf-8") as handle:
        return json.load(handle)


def compare_sequences(root: Path, left: str, right: str, ids: list[str], prefix: int | None = None) -> dict:
    distances = []
    uv_differences = []
    exact_masks = True
    exact_frames = True
    for video_id in ids:
        a = load_particle_sequence(root / "particle_sequences" / left / video_id)
        b = load_particle_sequence(root / "particle_sequences" / right / video_id)
        count = prefix or min(a.num_frames, b.num_frames)
        exact_frames &= np.array_equal(a.frame_indices[:count], b.frame_indices[:count])
        exact_masks &= np.array_equal(a.visibility[:count], b.visibility[:count])
        exact_masks &= np.array_equal(a.geometry_validity[:count], b.geometry_validity[:count])
        common = a.geometry_validity[:count] & b.geometry_validity[:count]
        if np.any(common):
            distances.extend(np.linalg.norm(a.xyz[:count][common] - b.xyz[:count][common], axis=-1))
        uv_common = a.visibility[:count] & b.visibility[:count]
        if np.any(uv_common):
            uv_differences.extend(np.linalg.norm(a.uv[:count][uv_common] - b.uv[:count][uv_common], axis=-1))
    result = summarize_distribution(np.asarray(distances))
    result.update({
        "exact_frame_indices": bool(exact_frames),
        "exact_masks": bool(exact_masks),
        "uv_difference": summarize_distribution(np.asarray(uv_differences)),
    })
    return result


def analyze_structure(sequence, config: ComponentConfig) -> dict:
    components = motion_coherent_components(sequence.xyz, sequence.geometry_validity, config)
    component_rows = []
    first_norms, second_norms, state_changes = [], [], []
    for component in components:
        state, valid = structure_state(sequence.xyz, sequence.geometry_validity, component)
        first, first_valid, second, second_valid = structural_differences(
            state, valid, sequence.timestamps_s
        )
        first_norms.extend(np.linalg.norm(first[first_valid], axis=1))
        second_norms.extend(np.linalg.norm(second[second_valid], axis=1))
        if np.sum(valid) > 1:
            state_changes.extend(np.linalg.norm(np.diff(state[valid], axis=0), axis=1))
        component_rows.append({
            "members": list(component),
            "size": len(component),
            "valid_frames": int(np.sum(valid)),
        })
    valid_step = sequence.geometry_validity[1:] & sequence.geometry_validity[:-1]
    particle_step = np.linalg.norm(sequence.xyz[1:] - sequence.xyz[:-1], axis=-1)
    particle_step[~valid_step] = np.nan
    return {
        "source_video_id": sequence.source_video_id,
        "components": component_rows,
        "component_count": len(components),
        "component_particle_fraction": float(len({i for c in components for i in c}) / sequence.num_tracks),
        "particle_step_m": summarize_distribution(particle_step),
        "structure_state_change": summarize_distribution(np.asarray(state_changes)),
        "first_order_structural_change": summarize_distribution(np.asarray(first_norms)),
        "second_order_structural_evolution": summarize_distribution(np.asarray(second_norms)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--pilot-a", default="pilot_a")
    parser.add_argument("--pilot-b", default="pilot_b")
    parser.add_argument("--prefix", default="prefix8")
    parser.add_argument("--expansion", default="expansion32")
    parser.add_argument("--part-a-only", action="store_true")
    args = parser.parse_args()
    root = args.root
    pilot = load_run(root, args.pilot_a)
    ids = [item["source_video_id"] for item in pilot["windows"]]
    repeatability = compare_sequences(root, args.pilot_a, args.pilot_b, ids)
    prefix_ids = ids[:2]
    history = compare_sequences(root, args.prefix, args.pilot_a, prefix_ids, prefix=8)
    write_json(root / "repeatability.json", repeatability)
    write_json(root / "history_immutability.json", history)

    expansion = load_run(root, args.expansion)
    coverages = [row["coverage"]["final_geometry_valid_fraction"] for row in expansion["windows"]]
    pair_success = [value for row in expansion["windows"] for value in row["pose_pair_valid"]]
    complete_pose = [all(row["pose_pair_valid"]) for row in expansion["windows"]]
    part_a_summary = {
        "repeatability": repeatability,
        "history_immutability": history,
        "windows": len(expansion["windows"]),
        "median_final_xyz_coverage": float(np.median(coverages)),
        "pose_pair_success_rate": float(np.mean(pair_success)),
        "complete_pose_window_rate": float(np.mean(complete_pose)),
    }
    if args.part_a_only:
        part_a_summary["classification"] = "EXPLICIT_FRONTEND_MIXED"
        write_json(root / "frontend_summary.json", part_a_summary)
        write_json(root / "real_normality_metrics.json", {
            "status": "NOT_RUN_PART_A_GATE_FAILED",
            "reason": "pilot median coverage was below 60% and only half of windows had complete pose",
        })
        print(json.dumps(part_a_summary, indent=2, allow_nan=False))
        return
    config = ComponentConfig(max_initial_distance=1.0, max_relative_change=0.05)
    structure = []
    for row in expansion["windows"]:
        sequence = load_particle_sequence(root / "particle_sequences" / args.expansion / row["source_video_id"])
        structure.append(analyze_structure(sequence, config))
    write_json(root / "component_config.json", {
        "method": "pairwise proximity and relative-displacement coherence connected components",
        "max_initial_distance_m": config.max_initial_distance,
        "max_relative_change_m": config.max_relative_change,
        "minimum_size": config.minimum_size,
        "minimum_temporal_overlap": config.minimum_overlap,
        "semantic_or_authenticity_input": False,
    })
    write_json(root / "component_metrics.json", {"windows": structure})
    write_json(root / "structure_state_metrics.json", {"representation": "median-scale-normalized pairwise-distance mean/std/p25/p75", "windows": structure})
    write_json(root / "first_order_metrics.json", {"windows": [{"id": x["source_video_id"], **x["first_order_structural_change"]} for x in structure]})
    write_json(root / "second_order_metrics.json", {"windows": [{"id": x["source_video_id"], **x["second_order_structural_evolution"]} for x in structure]})
    summary = {**part_a_summary, "windows_with_components": int(sum(row["component_count"] > 0 for row in structure))}
    write_json(root / "frontend_summary.json", summary)
    write_json(root / "real_normality_metrics.json", {
        "status": "NOT_RUN_REQUIRES_SEPARATE_OFFICIAL_REAL_TRAIN_POPULATION",
        "reason": "the fixed Phase 1B expansion is official validation only; source-video split integrity is preserved",
    })
    print(json.dumps(summary, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
