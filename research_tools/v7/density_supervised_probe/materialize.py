"""Materialize the matched 64/289 frontend population for the density pilot."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np

from research_tools.v7.observation_density_diagnostic.run_diagnostic import (
    BASE_ROOT,
    _front_result_key,
    _load_front_results,
    _old_prefix_without_suffix,
    _safe,
    run_frontend,
    write_json,
)


DEFAULT_OUTPUT = Path("/root/autodl-tmp/data/sparse_3d_forgery_detection/derived/v7_activityforensics_density_matched_frontend_v1")
SOURCE_DENSITY_ROOT = Path("/root/autodl-tmp/data/sparse_3d_forgery_detection/derived/v7_activityforensics_observation_density_diagnostic_v1")


def prepare_all_manifest(output: Path = DEFAULT_OUTPUT) -> list[dict[str, Any]]:
    """Create an all-192 manifest using the frozen density-diagnostic order."""

    rows = json.loads((BASE_ROOT / "manifests/window_manifest.json").read_text(encoding="utf-8"))
    output.mkdir(parents=True, exist_ok=True)
    write_json(
        output / "protocol.json",
        {
            "protocol_id": "v7-density-matched-frontend-v1",
            "population": "all 16 frozen source pairs and 192 existing windows",
            "densities": {"density64": 64, "density289": 289},
            "common_geometry": "depth, intrinsics and pose are computed once per window and reused by both query grids",
            "query_grids": "64=8x8 and 289=17x17 using the established process_size=256 interior linspace",
            "source_manifest": str(BASE_ROOT / "manifests/window_manifest.json"),
        },
    )
    write_json(output / "window_manifest.json", rows)
    return [dict(row) for row in rows]


def seed_existing_sample(output: Path = DEFAULT_OUTPUT) -> int:
    """Copy only the already-computed 16-window density sample, never videos."""

    source_results_path = SOURCE_DENSITY_ROOT / "frontend_results.json"
    if not source_results_path.is_file():
        return 0
    source_results = _load_front_results(SOURCE_DENSITY_ROOT)
    copied: dict[str, dict[str, Any]] = {}
    for key, item in source_results.items():
        window_id = str(item["window_id"])
        density = str(item["density"])
        old_prefix = Path(item["sequence_prefix"])
        if not old_prefix.with_suffix(".npz").is_file() or not old_prefix.with_suffix(".json").is_file():
            continue
        new_prefix = output / "particles" / f"{_safe(window_id)}__{density}"
        new_prefix.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(old_prefix.with_suffix(".npz"), new_prefix.with_suffix(".npz"))
        shutil.copy2(old_prefix.with_suffix(".json"), new_prefix.with_suffix(".json"))
        record = dict(item)
        record["sequence_prefix"] = str(new_prefix)
        record["seeded_from"] = str(old_prefix)
        copied[key] = record
    write_json(output / "frontend_results.json", {"results": list(sorted(copied.values(), key=lambda row: row["result_key"]))})
    return len(copied)


def materialize(output: Path = DEFAULT_OUTPUT, *, resume: bool = True) -> dict[str, Any]:
    if not (output / "window_manifest.json").is_file():
        prepare_all_manifest(output)
    if not (output / "frontend_results.json").is_file():
        seed_existing_sample(output)
    return run_frontend(output, resume=resume)


def materialize_nested(output: Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    """Run missing 289-point queries, then derive the validated nested 64 set.

    The density diagnostic already established that the odd-row/odd-column
    subset of the 17x17 grid is the exact 8x8 query lattice.  Reusing that
    subset keeps one depth/intrinsics/pose result and one tracker pass per
    window for the full 192-window population.
    """

    if not (output / "window_manifest.json").is_file():
        prepare_all_manifest(output)
    if not (output / "frontend_results.json").is_file():
        seed_existing_sample(output)

    import research_tools.v7.observation_density_diagnostic.run_diagnostic as diagnostic

    original_grid_config = diagnostic.GRID_CONFIG
    try:
        diagnostic.GRID_CONFIG = (("density289", 17),)
        frontend_result = run_frontend(output, resume=True)
    finally:
        diagnostic.GRID_CONFIG = original_grid_config

    from dataclasses import replace
    from types import SimpleNamespace
    from sparse3d_forgery.particle_sequence import load_particle_sequence, save_particle_sequence
    from research_tools.v7.observation_density_diagnostic.diagnostics import (
        component_diagnostic,
        compare_sequences,
        nested_query_mapping,
    )

    rows = json.loads((output / "window_manifest.json").read_text(encoding="utf-8"))
    results = _load_front_results(output)
    nested_indices = np.asarray([item["dense_index"] for item in nested_query_mapping()], dtype=np.int64)
    derived_count = 0
    for row in rows:
        window_id = str(row["window_id"])
        dense_key = _front_result_key(window_id, "density289")
        base_key = _front_result_key(window_id, "density64")
        dense_result = results.get(dense_key)
        if dense_result is None:
            raise RuntimeError(f"missing density289 result after frontend run: {window_id}")
        dense_prefix = Path(dense_result["sequence_prefix"])
        if not dense_prefix.with_suffix(".npz").is_file():
            raise FileNotFoundError(dense_prefix.with_suffix(".npz"))
        if base_key not in results:
            dense = load_particle_sequence(dense_prefix)
            subset = replace(
                dense,
                sample_id=f"{dense.sample_id}-nested64",
                track_ids=np.asarray(dense.track_ids)[nested_indices],
                xyz=np.asarray(dense.xyz)[:, nested_indices],
                uv=np.asarray(dense.uv)[:, nested_indices],
                visibility=np.asarray(dense.visibility)[:, nested_indices],
                geometry_validity=np.asarray(dense.geometry_validity)[:, nested_indices],
                lineage={**dense.lineage, "density_derivation": "validated_nested_17x17_odd_row_odd_column_subset"},
                provenance={**dense.provenance, "density_grid_size": 8, "query_count": 64, "derived_from_density289": True},
            )
            base_prefix = output / "particles" / f"{_safe(window_id)}__density64"
            save_particle_sequence(subset, base_prefix)
            base_diag = component_diagnostic(subset, window_start_s=float(row["interval_start_s"]), label="density64")
            base_result = {
                **dense_result,
                "result_key": base_key,
                "density": "density64",
                "query_count": 64,
                "grid_size": 8,
                "sequence_prefix": str(base_prefix),
                "density_derived_from": dense_result["result_key"],
                "diagnostic": {key: value for key, value in base_diag.items() if key != "support"},
                "valid_triplet_count": int(base_diag["valid_triplet_count"]),
                "component_count": int(base_diag["component_count"]),
                "max_component_members": int(base_diag["max_component_members"]),
                "component_internal_pair_count": int(base_diag["component_internal_pair_count"]),
                "qualified_graph_edge_count": int(base_diag["qualified_graph_edge_count"]),
                "selected_triplet_common_members": int(base_diag["selected_triplet_common_members"]),
                "support_status": base_diag["support_status"],
                "matched_historical_64": compare_sequences(load_particle_sequence(_old_prefix_without_suffix(window_id)), subset),
            }
            results[base_key] = base_result
            derived_count += 1
        dense = load_particle_sequence(dense_prefix)
        base = load_particle_sequence(Path(results[base_key]["sequence_prefix"]))
        dense_result["matched_nested_64_vs_289"] = compare_sequences(
            base,
            SimpleNamespace(
                frame_indices=dense.frame_indices,
                timestamps_s=dense.timestamps_s,
                frame_sizes_hw=dense.frame_sizes_hw,
                track_ids=np.asarray(dense.track_ids)[nested_indices],
                xyz=np.asarray(dense.xyz)[:, nested_indices],
                uv=np.asarray(dense.uv)[:, nested_indices],
                visibility=np.asarray(dense.visibility)[:, nested_indices],
                geometry_validity=np.asarray(dense.geometry_validity)[:, nested_indices],
            ),
        )
        results[dense_key] = dense_result
    write_json(output / "frontend_results.json", {"results": list(sorted(results.values(), key=lambda row: row["result_key"]))})
    return {**frontend_result, "derived_nested64": derived_count, "completed": len(results), "status": "COMPLETE" if len(results) == 384 else "PARTIAL"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--nested", action="store_true", help="run density289 and derive the nested density64 subset")
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()
    if args.prepare_only:
        prepare_all_manifest(args.output_root)
        print(json.dumps({"status": "PREPARED", "output": str(args.output_root)}, ensure_ascii=False))
        return
    result = materialize_nested(args.output_root) if args.nested else materialize(args.output_root, resume=not args.no_resume)
    print(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False))


if __name__ == "__main__":
    main()
