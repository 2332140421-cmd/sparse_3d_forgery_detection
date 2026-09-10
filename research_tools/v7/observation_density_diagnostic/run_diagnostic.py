#!/usr/bin/env python3
"""Run the bounded V7 observation-density and local-grouping diagnostic.

The script deliberately has a small phase-oriented surface.  It reuses the
frozen 64-point frontend and component implementation, writes research
artifacts outside the repository, and never fits a detector.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
from pathlib import Path
import platform
import time
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from sparse3d_forgery.experiments.v7_dynamic_structure_probe import ComponentConfig
from sparse3d_forgery.particle_sequence import (
    CoordinateSystem,
    Handedness,
    LengthUnit,
    build_particle_sequence,
    load_particle_sequence,
    save_particle_sequence,
)
from sparse3d_forgery.video_input import VideoSource, decode_video

from research_tools.v7.local_structural_temporal_probe.representation import (
    build_window_support,
    json_ready_support,
)
from research_tools.v7.observation_density_diagnostic.diagnostics import (
    COMPONENT_CONFIG,
    component_diagnostic,
    compare_sequences,
    density_summary,
    nested_query_mapping,
    review_frame_data,
    tracking_validity_rows,
)
from research_tools.v7.observation_density_diagnostic.roi import (
    OUTSIDE_DENSITY_POPULATION,
    ROI_SCHEMA_VERSION,
    TEMPORAL_ROI_PENDING,
    VALID,
    compare_density_details,
    validate_annotations,
)


DATA_ROOT = Path("/root/autodl-tmp/data/sparse_3d_forgery_detection")
BASE_ROOT = DATA_ROOT / "derived/v7_activityforensics_paired_second_order_pilot_v1"
JOINT_ROOT = DATA_ROOT / "derived/v7_activityforensics_local_structural_temporal_joint_diagnostic_v1"
OUTPUT_ROOT = DATA_ROOT / "derived/v7_activityforensics_observation_density_diagnostic_v1"
TAPNET_SOURCE = DATA_ROOT / "external/v7_explicit_geometry/tapnet-c2cbab81cc06092b5f05bfe2da7bfec54e2079c9"
TAPNET_CHECKPOINT = DATA_ROOT / "external/v7_explicit_geometry/checkpoints/causal_bootstapir_checkpoint.pt"
DEPTH_SOURCE = DATA_ROOT / "external/v7_explicit_geometry/ml-depth-pro-9efe5c1def37a26c5367a71df664b18e1306c708"
DEPTH_CHECKPOINT = DATA_ROOT / "external/v7_explicit_geometry/checkpoints/depth_pro.pt"
TRACKER_SHA = "c2cbab81cc06092b5f05bfe2da7bfec54e2079c9"
DEPTH_SHA = "9efe5c1def37a26c5367a71df664b18e1306c708"
GRID_CONFIG = (("density64", 8), ("density289", 17))


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


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False, default=_json_default)
        handle.write("\n")


def read_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def write_csv(path: Path, rows: Iterable[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    rows = list(rows)
    if fieldnames is None:
        fieldnames = []
        for row in rows:
            for name in row:
                if name not in fieldnames:
                    fieldnames.append(name)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _safe(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in value)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _old_prefix(window_id: str) -> Path:
    return BASE_ROOT / "particles" / f"{window_id.replace('::', '__')}.npz"


def _old_prefix_without_suffix(window_id: str) -> Path:
    return BASE_ROOT / "particles" / window_id.replace("::", "__")


def _protocol() -> dict[str, Any]:
    return {
        "protocol_id": "v7-observation-density-diagnostic-v1",
        "question": "Whether a nested 64-to-289 query grid adds reliable local relation support under the frozen frontend and component rule",
        "label_use": "label-blind measurement diagnostic; real/fake labels are not used by grouping or missing-state logic",
        "population_a": {"windows": 192, "source_ids": 16, "artifact": str(BASE_ROOT)},
        "population_b": {
            "selection": "first four source_id values with both source media and historical 64-point artifacts; earliest MANIP and CTRL per source, both real and fake",
            "maximum_windows": 16,
            "densities": {"density64": 64, "density289": 289},
            "nested_grid": "grid_size=8 is a subset of grid_size=17 under linspace(0,255,grid_size+2)[1:-1]",
        },
        "frontend_contract": {
            "process_size": 256,
            "query_chunk_size": 64,
            "tracker": "online_bootstapir",
            "tracker_source_sha": TRACKER_SHA,
            "depth": "apple_depth_pro",
            "depth_source_sha": DEPTH_SHA,
            "pose": "open3d RGB-D odometry; target_camera_from_source_camera, inverted during world accumulation",
            "depth_semantics": "optical-axis z-depth interpreted from official depth output",
            "intrinsics": "center principal point with first-frame focal length held fixed per window",
            "coordinate_system": "first_camera_world, right-handed, right/down/forward, meter",
            "causal_execution": True,
            "same_geometry_per_window": True,
            "random_seed": None,
            "raw_tracker_logits_or_expected_distance": "not exposed by the reused tracker.track API; canonical UV and thresholded visibility are retained without changing formal semantics",
        },
        "component_contract": {
            "max_initial_distance_m": COMPONENT_CONFIG.max_initial_distance,
            "max_relative_change_m": COMPONENT_CONFIG.max_relative_change,
            "minimum_size": COMPONENT_CONFIG.minimum_size,
            "minimum_overlap": COMPONENT_CONFIG.minimum_overlap,
            "history": "timestamps < interval_start_s + 0.5 s",
            "graph_vs_internal_pairs": "component graph edges connect members; model support may measure every internal pair",
        },
        "missing_state_policy": {
            "unknown_causes": "The canonical ParticleSequence masks do not separate true occlusion from tracker failure or depth from pose failure",
            "no_fake_evidence": "TRACK_NOT_VISIBLE and missing geometry are not treated as forgery evidence",
            "roi_status": "ROI_PENDING; local artifact has no mapped spatial ground truth",
        },
        "roi_contract": {
            "coordinates": "original decoded-video pixel coordinates; rectangle boundaries are inclusive",
            "identity": "source_id, window_id, role, source_frame_index and decoder PTS are required",
            "comparison": "density64 and density289 use the same window, source frame, PTS, frame size and rectangle",
            "counts": "unique actual triplet common members and normalized unordered track-ID pairs; not accuracy or anomaly scores",
            "temporal": "a one-frame annotation is TEMPORAL_ROI_PENDING and is never copied to other triplet frames",
            "current_status": "ROI_ANNOTATION_READY until a human exports a confirmed annotation",
        },
        "reuse": {
            "historical_64": str(BASE_ROOT / "particles"),
            "frozen_coverage": str(JOINT_ROOT / "coverage/coverage_windows.json"),
            "dense_depth_cache": False,
            "old_64_vs_matched_64": "reported when selected windows are reprocessed",
        },
    }


def prepare_manifest(output: Path) -> list[dict[str, Any]]:
    pairs = sorted(read_json(BASE_ROOT / "manifests/selected_pairs.json"), key=lambda row: row["source_id"])
    all_windows = read_json(BASE_ROOT / "manifests/window_manifest.json")
    by_source_kind_role = {}
    for row in all_windows:
        key = (row["source_id"], row["kind"], row["role"])
        by_source_kind_role.setdefault(key, []).append(row)
    selected: list[dict[str, Any]] = []
    selection_notes: list[dict[str, Any]] = []
    for pair in pairs:
        if len({row["source_id"] for row in selected}) >= 4:
            break
        source_id = pair["source_id"]
        candidates = []
        for kind in ("MANIP", "CTRL"):
            group = by_source_kind_role.get((source_id, kind, "real"), [])
            if not group:
                continue
            earliest = min(group, key=lambda row: (row["anchor_fraction"], row["window_id"]))
            fake_group = by_source_kind_role.get((source_id, kind, "fake"), [])
            base_window_id = earliest["window_id"].rsplit("::", 1)[0]
            fake = next((row for row in fake_group if row["window_id"].rsplit("::", 1)[0] == base_window_id), None)
            if fake is None:
                continue
            candidates.extend((earliest, fake))
        if len(candidates) != 4:
            continue
        if not all(Path(row["video_path"]).is_file() and _old_prefix_without_suffix(row["window_id"]).with_suffix(".npz").is_file() for row in candidates):
            continue
        selection_notes.append({"source_id": source_id, "reason": "fixed convenience sample: sorted source, earliest MANIP and CTRL, both roles"})
        for row in sorted(candidates, key=lambda item: (item["kind"], item["role"])):
            item = dict(row)
            item.update({
                "selection_reason": "no local spatial ROI mapping; fixed sorted-source convenience sample",
                "roi_status": "ROI_PENDING",
                "historical_particle_prefix": str(_old_prefix_without_suffix(row["window_id"])),
                "source_media_exists": Path(row["video_path"]).is_file(),
                "historical_64_exists": _old_prefix_without_suffix(row["window_id"]).with_suffix(".npz").is_file(),
            })
            selected.append(item)
    if len(selected) != 16:
        raise RuntimeError(f"fixed density sample requires 16 windows, selected {len(selected)}")
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "protocol.json", _protocol())
    write_json(output / "window_manifest.json", selected)
    fields = [
        "window_id", "source_id", "pair_id", "kind", "role", "label", "anchor_fraction",
        "interval_start_s", "interval_end_s", "video_path", "frame_indices", "timestamps_s",
        "selection_reason", "roi_status", "historical_particle_prefix", "source_media_exists", "historical_64_exists",
    ]
    csv_rows = []
    for row in selected:
        csv_rows.append({field: json.dumps(row[field], sort_keys=True) if isinstance(row[field], (list, dict)) else row[field] for field in fields})
    write_csv(output / "window_manifest.csv", csv_rows, fields)
    write_json(output / "selection_notes.json", {"notes": selection_notes, "selected_windows": len(selected)})
    write_csv(output / "query_mapping.csv", nested_query_mapping(), list(nested_query_mapping()[0]))
    return selected


def _compact_diagnostic(diag: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in diag.items() if key != "support"}


def _diag_row(row: dict[str, Any], diag: dict[str, Any], *, artifact_origin: str, frozen_match: bool | None = None) -> dict[str, Any]:
    return {
        "window_id": row["window_id"],
        "source_id": row["source_id"],
        "kind": row["kind"],
        "role": row["role"],
        "density": diag["density_label"],
        "artifact_origin": artifact_origin,
        "history_frame_count": diag["history_frame_count"],
        "history_candidate_points": diag["history_candidate_points"],
        "history_valid_point_observations": diag["history_valid_point_observations"],
        "qualified_graph_edge_count": diag["qualified_graph_edge_count"],
        "qualified_graph_edge_count_within_components": diag["qualified_graph_edge_count_within_components"],
        "component_count": diag["component_count"],
        "component_member_count_total": diag["component_member_count_total"],
        "max_component_members": diag["max_component_members"],
        "max_component_share_of_grouped": diag["max_component_share_of_grouped"],
        "component_internal_pair_count": diag["component_internal_pair_count"],
        "triplet_measured_pair_count": diag["triplet_measured_pair_count"],
        "qualified_edge_fraction_of_internal_pairs": diag["qualified_edge_fraction_of_internal_pairs"],
        "valid_triplet_count": diag["valid_triplet_count"],
        "selected_triplet_common_members": diag["selected_triplet_common_members"],
        "triplet_range_2d_per_frame": json.dumps(diag["triplet_range_2d_per_frame"], sort_keys=True),
        "triplet_range_3d_per_frame": json.dumps(diag["triplet_range_3d_per_frame"], sort_keys=True),
        "support_status": diag["support_status"],
        "frozen_component_match": frozen_match,
    }


def run_analysis(output: Path) -> dict[str, Any]:
    rows = read_json(output / "window_manifest.json") if (output / "window_manifest.json").is_file() else prepare_manifest(output)
    all_rows = read_json(BASE_ROOT / "manifests/window_manifest.json")
    frozen = read_json(JOINT_ROOT / "coverage/coverage_windows.json")
    diagnostics: list[dict[str, Any]] = []
    indirect: dict[str, Any] = {}
    tracking_path = output / "tracking_and_validity.csv"
    tracking_rows: list[dict[str, Any]] = []
    for number, row in enumerate(all_rows, 1):
        sequence = load_particle_sequence(_old_prefix_without_suffix(row["window_id"]))
        diag = component_diagnostic(sequence, window_start_s=float(row["interval_start_s"]), label="historical64")
        frozen_components = [tuple(int(x) for x in item["member_indices"]) for item in frozen[row["window_id"]].get("components", [])]
        rebuilt_components = [tuple(int(x) for x in item) for item in diag["component_members"]]
        frozen_match = sorted(frozen_components) == sorted(rebuilt_components)
        diagnostics.append(_diag_row(row, diag, artifact_origin="historical_64", frozen_match=frozen_match))
        indirect[row["window_id"]] = {
            "window_id": row["window_id"],
            "indirect_connectivity_examples": diag["indirect_connectivity_examples"],
            "graph_edges": diag["graph_edges"],
            "component_members": diag["component_members"],
            "frozen_component_match": frozen_match,
        }
        support = diag["support"]
        for validity_row in tracking_validity_rows(sequence, support, density_label="historical64"):
            tracking_rows.append({"window_id": row["window_id"], "source_id": row["source_id"], "kind": row["kind"], "role": row["role"], **validity_row})
        if number % 32 == 0:
            print(f"analysis {number}/{len(all_rows)}", flush=True)
    write_csv(output / "component_diagnostics.csv", diagnostics)
    write_csv(output / "tracking_and_validity.csv", tracking_rows)
    write_json(output / "indirect_connectivity_examples.json", indirect)
    summary = {
        "windows": len(all_rows),
        "frozen_component_matches": int(sum(bool(row["frozen_component_match"]) for row in diagnostics)),
        "frozen_component_mismatches": [row["window_id"] for row in diagnostics if row["frozen_component_match"] is not True],
        "support_status_counts": {status: sum(row["support_status"] == status for row in diagnostics) for status in sorted({row["support_status"] for row in diagnostics})},
        "historical_density": "64",
        "roi_status": "ROI_PENDING",
    }
    write_json(output / "analysis_summary.json", summary)
    return summary


def _decode_window(row: dict[str, Any]):
    return decode_video(
        VideoSource(
            sample_id=f"v7-density-{row['window_id']}",
            source_video_id=f"{row['source_id']}-{row['role']}",
            source_locator=row["video_path"],
        ),
        row["frame_indices"],
    )


def _sequence_subset(sequence: Any, indices: np.ndarray) -> Any:
    from types import SimpleNamespace
    return SimpleNamespace(
        frame_indices=np.asarray(sequence.frame_indices),
        timestamps_s=np.asarray(sequence.timestamps_s),
        frame_sizes_hw=np.asarray(sequence.frame_sizes_hw),
        track_ids=np.asarray(sequence.track_ids)[indices],
        xyz=np.asarray(sequence.xyz)[:, indices],
        uv=np.asarray(sequence.uv)[:, indices],
        visibility=np.asarray(sequence.visibility)[:, indices],
        geometry_validity=np.asarray(sequence.geometry_validity)[:, indices],
    )


def _front_result_key(window_id: str, density: str) -> str:
    return f"{window_id}::{density}"


def _load_front_results(output: Path) -> dict[str, dict[str, Any]]:
    path = output / "frontend_results.json"
    if not path.is_file():
        return {}
    return {str(item["result_key"]): item for item in read_json(path).get("results", [])}


def _load_roi_annotations(path: Path) -> list[dict[str, Any]]:
    """Read human input without replacing or normalizing the original file."""

    if not path.is_file():
        return []
    value = read_json(path)
    if isinstance(value, list):
        return [dict(item) for item in value if isinstance(item, dict)]
    if isinstance(value, dict) and isinstance(value.get("annotations"), list):
        return [dict(item) for item in value["annotations"] if isinstance(item, dict)]
    raise ValueError(f"ROI annotations must be a JSON list: {path}")


def _write_roi_outputs(
    output: Path,
    detail_by_density: Mapping[tuple[str, str], Mapping[str, Any]],
    manifest_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Create the small ROI hand-off and compute only explicitly supplied ROIs."""

    roi_root = output / "roi_review"
    roi_root.mkdir(parents=True, exist_ok=True)
    annotation_path = roi_root / "annotations.json"
    if not annotation_path.exists():
        write_json(annotation_path, [])
    template_path = roi_root / "annotation_template.json"
    if not template_path.exists():
        write_json(
            template_path,
            {
                "schema_version": ROI_SCHEMA_VERSION,
                "instructions": "Fill one row per manually confirmed source frame; coordinates are original pixels.",
                "required": ["source_id", "window_id", "role", "source_frame_index", "timestamp_s", "frame_size_hw", "rect"],
                "rect": {"x": 0, "y": 0, "w": 1, "h": 1},
                "observation": "uncertain",
                "tracking_observation": "uncertain",
                "notes": "",
            },
        )
    annotations = _load_roi_annotations(annotation_path)
    manifest_by_window = {str(row["window_id"]): row for row in manifest_rows}
    density64 = {window_id: detail for (window_id, density), detail in detail_by_density.items() if density == "density64"}
    validation_rows = validate_annotations(annotations, density64, manifest_by_window)
    write_csv(
        roi_root / "roi_validation.csv",
        validation_rows,
        ["status", "window_id", "source_id", "role", "source_frame_index", "timestamp_s", "reason", "frame_size_hw", "rect"],
    )
    frame_rows: list[dict[str, Any]] = []
    component_rows: list[dict[str, Any]] = []
    comparisons: list[dict[str, Any]] = []
    valid_count = 0
    for annotation, validation in zip(annotations, validation_rows):
        if validation.get("status") != VALID:
            continue
        valid_count += 1
        normalized = dict(validation)
        window_id = str(normalized["window_id"])
        detail64 = detail_by_density.get((window_id, "density64"))
        detail289 = detail_by_density.get((window_id, "density289"))
        if detail64 is None or detail289 is None:
            validation["status"] = OUTSIDE_DENSITY_POPULATION
            validation["reason"] = "both density details are required"
            continue
        comparison, per_frame, per_component = compare_density_details(detail64, detail289, normalized)
        comparisons.append(comparison)
        frame_rows.extend(per_frame)
        component_rows.extend(per_component)
    write_csv(
        roi_root / "per_frame_coverage.csv",
        frame_rows,
        [
            "window_id", "source_id", "role", "source_frame_index", "timestamp_s", "density",
            "roi_x", "roi_y", "roi_w", "roi_h", "visible_query_points", "geometry_valid_points",
            "valid_history_component_points", "valid_triplet_common_points", "internal_pairs_both_endpoints",
            "cross_roi_pairs_one_endpoint", "outside_pairs_both_endpoints", "component_ids", "triplet_ids",
            "model_sample_status",
        ],
    )
    write_csv(
        roi_root / "per_component_triplet_coverage.csv",
        component_rows,
        [
            "component_index", "triplet_id", "source_frame_index", "triplet_common_members_total",
            "roi_common_members", "actual_internal_pairs_total", "roi_internal_pairs", "cross_roi_pairs",
            "outside_pairs", "roi_internal_pair_fraction",
        ],
    )
    write_csv(
        roi_root / "density_comparison.csv",
        comparisons,
        [
            "window_id", "source_id", "role", "source_frame_index", "timestamp_s", "same_source_frame",
            "timestamp_abs_difference_s", "density64_visible_query_points", "density289_visible_query_points",
            "density64_geometry_valid_points", "density289_geometry_valid_points",
            "density64_valid_triplet_common_points", "density289_valid_triplet_common_points",
            "density64_internal_pairs_both_endpoints", "density289_internal_pairs_both_endpoints",
            "density64_cross_roi_pairs_one_endpoint", "density289_cross_roi_pairs_one_endpoint",
            "support_status", "interpretation",
        ],
    )
    valid_frames_by_window: dict[str, set[int]] = {}
    for row in validation_rows:
        if row.get("status") == VALID:
            valid_frames_by_window.setdefault(str(row["window_id"]), set()).add(int(row["source_frame_index"]))
    if not valid_frames_by_window:
        temporal_status = "NO_ROI"
    elif any(len(frames) < 3 for frames in valid_frames_by_window.values()):
        temporal_status = TEMPORAL_ROI_PENDING
    else:
        temporal_status = "READY_FOR_THREE_FRAME_REVIEW"
    if comparisons:
        status = "ROI_COVERAGE_COMPARISON_COMPLETE"
    else:
        status = "ROI_ANNOTATION_READY"
    summary = {
        "status": status,
        "schema_version": ROI_SCHEMA_VERSION,
        "annotation_count": len(annotations),
        "valid_annotation_count": valid_count,
        "comparison_count": len(comparisons),
        "temporal_status": temporal_status,
        "population": {"source_count": len({str(row["source_id"]) for row in manifest_rows}), "window_count": len(manifest_rows)},
        "interpretation": "ROI counts describe measurement support only; they are not spatial ground truth, accuracy, anomaly score, or detection probability.",
        "input": str(annotation_path),
        "pending_user_fields": ["source_id", "window_id", "role", "source_frame_index", "timestamp_s", "frame_size_hw", "rect", "observation", "tracking_observation"],
    }
    write_json(roi_root / "summary.json", summary)
    return {
        "status": status,
        "schema_version": ROI_SCHEMA_VERSION,
        "annotation_path": "../roi_review/annotations.json",
        "template_path": "../roi_review/annotation_template.json",
        "summary_path": "../roi_review/summary.json",
        "validation_path": "../roi_review/roi_validation.csv",
        "comparison_path": "../roi_review/density_comparison.csv",
        "annotation_count": len(annotations),
        "valid_annotation_count": valid_count,
        "comparison_count": len(comparisons),
        "temporal_status": temporal_status,
    }


def _save_front_results(output: Path, results: dict[str, dict[str, Any]]) -> None:
    write_json(output / "frontend_results.json", {"results": list(sorted(results.values(), key=lambda item: item["result_key"]))})


def run_frontend(output: Path, *, only_window: str | None = None, resume: bool = False) -> dict[str, Any]:
    rows = read_json(output / "window_manifest.json")
    if only_window is not None:
        rows = [row for row in rows if row["window_id"] == only_window]
        if not rows:
            raise ValueError(f"unknown selected window: {only_window}")
    results = _load_front_results(output) if resume else {}
    pending = []
    for row in rows:
        for density, grid in GRID_CONFIG:
            existing = results.get(_front_result_key(row["window_id"], density))
            # Older partial runs predated the nested 64-vs-289 comparison;
            # recompute only that bookkeeping, reusing the saved arrays.
            needs_nested_audit = density == "density289" and existing is not None and "matched_nested_64_vs_289" not in existing
            if existing is None or needs_nested_audit:
                pending.append((row, density, grid))
    bookkeeping_only = []
    for row, density, grid in pending:
        existing = results.get(_front_result_key(row["window_id"], density))
        if density == "density289" and existing is not None:
            dense_prefix = Path(existing["sequence_prefix"])
            matched64_prefix = output / "particles" / f"{_safe(row['window_id'])}__density64"
            if dense_prefix.with_suffix(".npz").is_file() and matched64_prefix.with_suffix(".npz").is_file():
                nested_indices = np.asarray([item["dense_index"] for item in nested_query_mapping()], dtype=np.int64)
                existing["matched_nested_64_vs_289"] = compare_sequences(
                    load_particle_sequence(matched64_prefix),
                    _sequence_subset(load_particle_sequence(dense_prefix), nested_indices),
                )
                results[existing["result_key"]] = existing
                bookkeeping_only.append((row, density, grid))
    if bookkeeping_only:
        for item in bookkeeping_only:
            pending.remove(item)
        _save_front_results(output, results)
    if not pending:
        return {"completed": len(results), "pending": 0, "status": "REUSED"}
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

    tracker = OnlineBootsTapir(TAPNET_SOURCE, TAPNET_CHECKPOINT, process_size=256, grid_size=8)
    depth_runner = DepthProRunner(DEPTH_SOURCE, DEPTH_CHECKPOINT)
    geometry_cache: dict[tuple[str, tuple[int, ...]], tuple[Any, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]] = {}
    for number, (row, density, grid_size) in enumerate(pending, 1):
        started = time.perf_counter()
        frame_key = (str(row["video_path"]), tuple(int(x) for x in row["frame_indices"]))
        if frame_key not in geometry_cache:
            decoded = _decode_window(row)
            depths, focals, frame_depth_valid = depth_runner.infer(decoded)
            intrinsics, fixed_focal_px = causal_first_frame_intrinsics(depths, focals)
            adjacent, pair_valid, information = rgbd_odometry(decoded, depths, intrinsics)
            world_from_camera, pose_valid = accumulate_world_from_camera(adjacent, pair_valid)
            geometry_cache[frame_key] = (decoded, depths, focals, frame_depth_valid, intrinsics, pair_valid, information, world_from_camera, pose_valid, fixed_focal_px)
        decoded, depths, focals, frame_depth_valid, intrinsics, pair_valid, information, world_from_camera, pose_valid, fixed_focal_px = geometry_cache[frame_key]
        tracker.grid_size = grid_size
        uv, visibility = tracker.track(decoded)
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
            lineage={
                "dataset": "ActivityForensics+Charades",
                "official_split": "train",
                "source_id": row["source_id"],
                "pair_id": row["pair_id"],
                "role": row["role"],
                "window_id": row["window_id"],
                "density_diagnostic": "v7-observation-density-diagnostic-v1",
            },
            provenance={
                "tracker": "online_bootstapir",
                "tracker_source_sha": TRACKER_SHA,
                "depth": "apple_depth_pro",
                "depth_source_sha": DEPTH_SHA,
                "depth_semantics": "optical_axis_z_depth_interpretation_of_official_depth_output",
                "principal_point": "EXPERIMENTAL_CENTER_PRINCIPAL_POINT_ASSUMPTION",
                "distortion": "EXPERIMENTAL_ZERO_DISTORTION_ASSUMPTION",
                "intrinsics_policy": "CAUSAL_FIRST_FRAME_DEPTH_PRO_FOCAL_ASSUMPTION",
                "fixed_focal_px": fixed_focal_px,
                "pose": "open3d_0.19_adjacent_rgbd_odometry",
                "pose_convention": "target_camera_from_source_camera; inverted during world accumulation",
                "causal_execution": True,
                "process_size": 256,
                "query_chunk_size": 64,
                "density_grid_size": grid_size,
                "query_count": int(uv.shape[1]),
                "shared_geometry_key": hashlib.sha256((str(row["video_path"]) + repr(row["frame_indices"])).encode()).hexdigest(),
                "shared_geometry_with_density_arms": True,
            },
        )
        prefix = output / "particles" / f"{_safe(row['window_id'])}__{density}"
        if not prefix.with_suffix(".npz").is_file():
            save_particle_sequence(sequence, prefix)
        support = build_window_support(sequence, window_start_s=float(row["interval_start_s"]), component_config=COMPONENT_CONFIG)
        diag = component_diagnostic(sequence, window_start_s=float(row["interval_start_s"]), label=density)
        old = load_particle_sequence(_old_prefix_without_suffix(row["window_id"]))
        matched_compare = compare_sequences(old, sequence) if density == "density64" else None
        nested_compare = None
        if density == "density289":
            matched64_prefix = output / "particles" / f"{_safe(row['window_id'])}__density64"
            if matched64_prefix.with_suffix(".npz").is_file():
                matched64 = load_particle_sequence(matched64_prefix)
                nested_indices = np.asarray([item["dense_index"] for item in nested_query_mapping()], dtype=np.int64)
                nested_compare = compare_sequences(matched64, _sequence_subset(sequence, nested_indices))
        result = {
            "result_key": _front_result_key(row["window_id"], density),
            "window_id": row["window_id"],
            "source_id": row["source_id"],
            "source_video_path": row["video_path"],
            "kind": row["kind"],
            "role": row["role"],
            "density": density,
            "query_count": int(uv.shape[1]),
            "grid_size": grid_size,
            "frame_indices": [int(x) for x in sequence.frame_indices],
            "timestamps_s": [float(x) for x in sequence.timestamps_s],
            "frame_sizes_hw": np.asarray(sequence.frame_sizes_hw).tolist(),
            "sequence_prefix": str(prefix),
            "geometry_key": hashlib.sha256((str(row["video_path"]) + repr(row["frame_indices"])).encode()).hexdigest(),
            "geometry_shared_with_other_density": True,
            "depth_pose_computed_once_for_window": True,
            "focal_px": np.asarray(focals).tolist(),
            "frame_depth_valid": np.asarray(frame_depth_valid).tolist(),
            "pose_pair_valid": np.asarray(pair_valid).tolist(),
            "pose_valid": np.asarray(pose_valid).tolist(),
            "tracking_visible_fraction": float(np.mean(visibility)),
            "geometry_valid_fraction": float(np.mean(geometry_valid)),
            "valid_points_per_frame": np.sum(geometry_valid, axis=1).astype(int).tolist(),
            "valid_triplet_count": int(diag["valid_triplet_count"]),
            "component_count": int(diag["component_count"]),
            "max_component_members": int(diag["max_component_members"]),
            "component_internal_pair_count": int(diag["component_internal_pair_count"]),
            "qualified_graph_edge_count": int(diag["qualified_graph_edge_count"]),
            "selected_triplet_common_members": int(diag["selected_triplet_common_members"]),
            "support_status": diag["support_status"],
            "diagnostic": _compact_diagnostic(diag),
            "matched_historical_64": matched_compare,
            "matched_nested_64_vs_289": nested_compare,
            "source_frame_contract_equal": np.array_equal(sequence.frame_indices, np.asarray(row["frame_indices"], dtype=np.int64)),
            "source_timestamp_max_abs_error_s": float(np.max(np.abs(sequence.timestamps_s - np.asarray(row["timestamps_s"], dtype=np.float64)))),
            "elapsed_s": time.perf_counter() - started,
            "causal_training_eligible": False,
            "causal_training_reason": "bounded frontend diagnostic; no formal causal target construction or training was performed",
        }
        results[result["result_key"]] = result
        _save_front_results(output, results)
        print(f"frontend {number}/{len(pending)} {result['result_key']} valid={result['geometry_valid_fraction']:.3f} support={result['support_status']}", flush=True)
        del uv, visibility, xyz, geometry_valid, sequence, support, diag
        gc.collect()
        torch.cuda.empty_cache()
        # A full 192-window run keeps both densities in one process.  Once the
        # second density for a window is emitted, release that window's shared
        # RGB/depth/pose arrays so memory usage does not grow with population.
        if density == "density289":
            geometry_cache.pop(frame_key, None)
    peak = int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else 0
    write_json(output / "frontend_resource_metrics.json", {
        "completed_results": len(results),
        "new_results_this_run": len(pending),
        "peak_gpu_bytes": peak,
        "environment": "/root/autodl-tmp/envs/v7-explicit-geometry",
        "python": platform.python_version(),
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    })
    return {"completed": len(results), "pending": 0, "status": "COMPLETE" if len(results) >= 32 else "PARTIAL"}


def _frozen_media_map() -> dict[str, dict[str, Any]]:
    path = JOINT_ROOT / "review/data.json"
    if not path.is_file():
        return {}
    return {row["window_id"]: row for row in read_json(path).get("windows", [])}


def build_review(output: Path) -> dict[str, Any]:
    results = _load_front_results(output)
    media_map = _frozen_media_map()
    review_root = output / "review"
    detail_root = review_root / "details"
    detail_root.mkdir(parents=True, exist_ok=True)
    index_rows: list[dict[str, Any]] = []
    detail_by_density: dict[tuple[str, str], dict[str, Any]] = {}
    for key, result in sorted(results.items()):
        prefix = Path(result["sequence_prefix"])
        sequence = load_particle_sequence(prefix)
        support = build_window_support(sequence, window_start_s=float(next(row["interval_start_s"] for row in read_json(output / "window_manifest.json") if row["window_id"] == result["window_id"])), component_config=COMPONENT_CONFIG)
        diag = result["diagnostic"]
        detail = review_frame_data(
            sequence,
            support,
            diag,
            nested_dense=result["density"] == "density289",
        )
        source_media = media_map.get(result["window_id"], {}).get("media", {})
        clip_path = source_media.get("clip_path")
        if clip_path and (JOINT_ROOT / "review" / clip_path).is_file():
            video_url = f"../../v7_activityforensics_local_structural_temporal_joint_diagnostic_v1/review/{clip_path}"
            media_status = "AVAILABLE"
        elif Path(result.get("source_video_path", "")).is_file():
            video_url = ""
            media_status = "NOT_MATERIALIZED"
        else:
            video_url = ""
            media_status = "SOURCE_MISSING"
        detail.update({
            "window_id": result["window_id"],
            "density": result["density"],
            "query_count": result["query_count"],
            "grid_size": result["grid_size"],
            "video_url": video_url,
            "media_status": media_status,
            "source_frame_contract": result["source_frame_contract_equal"],
            "exact_source_frame_indices": result["frame_indices"],
            "exact_source_timestamps_s": result["timestamps_s"],
            "component_summaries": [
                {"component_index": i, "member_count": len(members)}
                for i, members in enumerate(diag.get("component_members", []))
            ],
            "graph_edges_total": result["qualified_graph_edge_count"],
            "component_internal_pair_count": result["component_internal_pair_count"],
            "roi_status": "ROI_PENDING",
        })
        detail_by_density[(str(result["window_id"]), str(result["density"]))] = detail
        detail_path = detail_root / f"{_safe(key)}.json"
        write_json(detail_path, detail)
        index_rows.append({
            "result_key": key,
            "window_id": result["window_id"],
            "density": result["density"],
            "source_id": result["source_id"],
            "kind": result["kind"],
            "role": result["role"],
            "detail_path": f"details/{detail_path.name}",
            "video_url": video_url,
            "media_status": media_status,
        })
    roi_info = _write_roi_outputs(output, detail_by_density, read_json(output / "window_manifest.json"))
    write_json(review_root / "data.json", {
        "protocol": "v7-observation-density-diagnostic-v1",
        "browser_visual_acceptance": "not_run_on_server; static HTTP and JSON checks are provided",
        "windows": index_rows,
        "roi_review": roi_info,
        "legend": {
            "yellow": "not in a history component",
            "component": "component index only; not a real/fake or semantic label",
            "white": "selected triplet common member",
            "base_grid": "nested 64-point coordinates in the 289 grid",
        },
    })
    html = _review_html()
    (review_root / "index.html").write_text(html, encoding="utf-8")
    return {
        "details": len(index_rows),
        "available_media": sum(row["media_status"] == "AVAILABLE" for row in index_rows),
        "roi_status": roi_info["status"],
        "roi_annotation_count": roi_info["annotation_count"],
    }


def _review_html() -> str:
    return r'''<!doctype html>
<meta charset="utf-8">
<title>V7 observation density diagnostic</title>
<style>
body{font-family:system-ui,sans-serif;background:#101827;color:#e8edf5;margin:20px}
select,button,input{font-size:1rem;margin:4px;padding:5px} .note{background:#202d40;padding:10px;border-radius:8px}
.stage{position:relative;max-width:960px;background:#000} video{max-width:960px;width:100%;display:block;background:#000}
canvas{position:absolute;left:0;top:0;width:100%;height:100%;pointer-events:none} #stats,#roiStats{white-space:pre-wrap;background:#202d40;padding:10px;border-radius:8px}
table{border-collapse:collapse}td,th{border:1px solid #46566c;padding:4px}.pending{color:#fbbf24}
</style>
<h1>V7 观测密度与局部分组诊断</h1>
<div class="note">这是固定前端、无训练的观测支撑诊断。点消失不等于伪造；component 编号不是语义部位或真假标签。<span class="pending">ROI 仅在人工确认后计算，当前没有预填空间真值。</span>源帧与 PTS 列表是精确核对依据，播放器 seek 只是浏览便利。</div>
<p><label>source <select id="source"></select></label><label>kind <select id="kind"><option value="ALL">ALL</option><option value="MANIP">MANIP</option><option value="CTRL">CTRL</option></select></label><label>role <select id="role"><option value="ALL">ALL</option><option value="real">real</option><option value="fake">fake</option></select></label></p>
<p><label>window <select id="window"></select></label><label>density <select id="density"><option value="density64">64</option><option value="density289">289</option></select></label><label>component <select id="component"><option value="-1">全部</option></select></label> <button id="prev">上一帧</button><button id="next">下一帧</button></p>
<div class="stage"><video id="video" controls></video><canvas id="canvas"></canvas></div>
<p><label><input id="roiMode" type="checkbox"> 圈选原始像素 ROI</label><button id="clearRoi">清除 ROI</button><button id="saveRoi">保存当前人工观察</button><button id="exportRoi">导出 JSON</button><label>导入 JSON <input id="importRoi" type="file" accept="application/json"></label></p>
<p><label>观察<select id="observation"><option>uncertain</option><option>internal_deformation</option><option>overall_motion</option><option>appearance_change</option><option>no_obvious_distortion</option></select></label><label>跟踪<select id="tracking"><option>uncertain</option><option>stable_visible_surface</option><option>sliding</option><option>surface_visible_point_missing</option><option>occluded</option></select></label><input id="notes" placeholder="人工备注"></p>
<div id="roiStats"></div><div id="stats"></div><h2>精确源帧 / PTS</h2><table><thead><tr><th>序号</th><th>源帧</th><th>PTS(s)</th><th>模型实际使用</th></tr></thead><tbody id="frames"></tbody></table>
<script>
const state={windows:[],details:null,frame:0,sourceFrame:null,roi:null,annotations:[],roiStatus:'ROI_ANNOTATION_READY'};const $=id=>document.getElementById(id);let dragStart=null;
function unique(values){return [...new Set(values)].sort()}
function filtered(){const source=$('source').value,kind=$('kind').value,role=$('role').value;return state.windows.filter(w=>(source==='ALL'||w.source_id===source)&&(kind==='ALL'||w.kind===kind)&&(role==='ALL'||w.role===role))}
function rebuildWindows(){const rows=filtered(),seen=new Set(),select=$('window'),old=select.value;select.innerHTML='';for(const w of rows){if(seen.has(w.window_id))continue;seen.add(w.window_id);const o=document.createElement('option');o.value=w.window_id;o.textContent=w.source_id+' '+w.kind+' '+w.role+' '+w.window_id;select.append(o)}if([...select.options].some(o=>o.value===old))select.value=old}
function chosen(){const id=$('window').value,den=$('density').value;return state.windows.find(w=>w.window_id===id&&w.density===den)||state.windows.find(w=>w.window_id===id)}
function selectedWindow(){return state.windows.find(w=>w.window_id===$('window').value)||null}
async function init(){const data=await (await fetch('data.json')).json();state.windows=data.windows||[];state.roiStatus=data.roi_review?.status||'ROI_ANNOTATION_READY';const sources=unique(state.windows.map(w=>w.source_id));$('source').innerHTML='<option value="ALL">ALL</option>'+sources.map(x=>'<option>'+x+'</option>').join('');rebuildWindows();for(const id of ['source','kind','role'])$(id).onchange=()=>{rebuildWindows();load()};$('window').onchange=load;$('density').onchange=load;$('component').onchange=draw;$('prev').onclick=()=>move(-1);$('next').onclick=()=>move(1);$('roiMode').onchange=()=>{$('canvas').style.pointerEvents=$('roiMode').checked?'auto':'none';if(!$('roiMode').checked)dragStart=null};$('clearRoi').onclick=()=>{state.roi=null;draw();updateRoiStats()};$('saveRoi').onclick=saveAnnotation;$('exportRoi').onclick=exportAnnotations;$('importRoi').onchange=importAnnotations;await load();try{const r=await fetch(data.roi_review?.annotation_path||'../roi_review/annotations.json');const x=await r.json();state.annotations=Array.isArray(x)?x:(x.annotations||[])}catch(_){state.annotations=[]}updateRoiStats()}
async function load(){const w=chosen();if(!w)return;const oldFrame=state.details?.frames?.[state.frame]?.source_frame_index;state.details=await (await fetch(w.detail_path)).json();let index=state.details.frames.findIndex(f=>f.source_frame_index===oldFrame);state.frame=index>=0?index:0;let c=$('component');c.innerHTML='<option value="-1">全部</option>';for(const x of state.details.component_summaries){const o=document.createElement('option');o.value=x.component_index;o.textContent='component '+x.component_index+' ('+x.member_count+' 点)';c.append(o)}$('video').src=state.details.video_url||'';$('video').load();$('video').onloadedmetadata=()=>seekFrame();let tb=$('frames');tb.innerHTML='';state.details.frames.forEach((f,i)=>{const tr=document.createElement('tr');const model=(state.details.triplets||[]).some(t=>t.source_frame_indices.includes(f.source_frame_index));tr.innerHTML='<td>'+i+'</td><td>'+f.source_frame_index+'</td><td>'+f.timestamp_s.toFixed(6)+'</td><td>'+ (model?'yes':'—') +'</td>';tb.append(tr)});draw();updateRoiStats()}
function seekFrame(){if(!state.details)return;const f=state.details.frames[state.frame],first=state.details.frames[0];if(f&&first&&Number.isFinite($('video').duration))$('video').currentTime=Math.max(0,f.timestamp_s-first.timestamp_s)}
function move(delta){if(!state.details)return;state.frame=Math.max(0,Math.min(state.details.frames.length-1,state.frame+delta));seekFrame();draw();updateRoiStats()}
function canvasPoint(event){const f=state.details.frames[state.frame],r=$('canvas').getBoundingClientRect();return{x:(event.clientX-r.left)*f.width/r.width,y:(event.clientY-r.top)*f.height/r.height}}
function draw(){const d=state.details;if(!d)return;const f=d.frames[state.frame],c=$('canvas');c.width=f.width;c.height=f.height;const x=c.getContext('2d');x.clearRect(0,0,c.width,c.height);const selected=+$('component').value;for(const p of f.points){const [slot,u,v,vis,geo,comp,common,base]=p;if(selected>=0&&comp!==selected)continue;x.beginPath();x.arc(u,v,common?8:base?5:3,0,Math.PI*2);x.fillStyle=comp<0?'#ffd400':(base?'#55d6ff':'#b084ff');x.fill();if(common){x.lineWidth=2;x.strokeStyle='#fff';x.stroke()}}if(state.roi){x.strokeStyle='#facc15';x.lineWidth=3;x.setLineDash([8,5]);x.strokeRect(state.roi.x,state.roi.y,state.roi.w,state.roi.h);x.setLineDash([])}$('stats').textContent='window='+d.window_id+' density='+d.density+' frame='+f.source_frame_index+' PTS='+f.timestamp_s.toFixed(6)+'\nvisible/geometry points='+f.points.filter(p=>p[3]&&p[4]).length+'/'+f.points.length+'\ncomponent direct graph edges shown='+Math.min(d.graph_edges_total,200)+' / actual='+d.graph_edges_total+'; internal measured pairs='+d.component_internal_pair_count+'; common triplet members='+d.common_member_slots.length+'\nmedia='+d.media_status+'; source-frame contract='+d.source_frame_contract}
function updateRoiStats(){const f=state.details?.frames?.[state.frame];$('roiStats').textContent='ROI status='+state.roiStatus+'; current source frame='+(f?.source_frame_index??'—')+' PTS='+(f?.timestamp_s?.toFixed(6)??'—')+'; ROI='+ (state.roi?JSON.stringify(state.roi):'未圈选') +'\n64/289 必须在同一 window、同一源帧、同一原始像素矩形上比较；程序计数结果写入 roi_review/。单帧标注不会自动成为三帧持续 ROI。'}
function saveAnnotation(){const w=selectedWindow(),f=state.details?.frames?.[state.frame];if(!w||!f||!state.roi){$('roiStats').textContent='请先选择窗口、准确源帧并圈选 ROI';return}state.annotations.push({schema_version:'v1',source_id:w.source_id,window_id:w.window_id,role:w.role,source_frame_index:f.source_frame_index,timestamp_s:f.timestamp_s,frame_size_hw:[f.height,f.width],rect:{...state.roi},density_reference:'both',observation:$('observation').value,tracking_observation:$('tracking').value,notes:$('notes').value||''});$('roiStats').textContent='已保存浏览器内标注 '+state.annotations.length+' 条；请导出 JSON 后交给离线计数程序。'}
function download(name,text,type){const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([text],{type}));a.download=name;a.click();setTimeout(()=>URL.revokeObjectURL(a.href),1000)}
function exportAnnotations(){download('v7_density_roi_annotations.json',JSON.stringify(state.annotations,null,2),'application/json')}
async function importAnnotations(event){const file=event.target.files[0];if(!file)return;try{const x=JSON.parse(await file.text());state.annotations=Array.isArray(x)?x:(x.annotations||[]);const a=state.annotations.find(x=>x.window_id===$('window').value);if(a){state.frame=state.details.frames.findIndex(f=>f.source_frame_index===Number(a.source_frame_index));state.frame=Math.max(0,state.frame);state.roi=a.rect||a.region;seekFrame();draw()}$('roiStats').textContent='已导入 '+state.annotations.length+' 条；程序仍会重新校验源帧、PTS、尺寸和边界。'}catch(error){$('roiStats').textContent='导入失败：'+error.message}}
$('canvas').addEventListener('pointerdown',e=>{if(!$('roiMode').checked||!state.details)return;dragStart=canvasPoint(e);$('canvas').setPointerCapture(e.pointerId)});$('canvas').addEventListener('pointermove',e=>{if(!dragStart)return;const p=canvasPoint(e);state.roi={x:Math.min(dragStart.x,p.x),y:Math.min(dragStart.y,p.y),w:Math.abs(p.x-dragStart.x),h:Math.abs(p.y-dragStart.y)};draw();updateRoiStats()});$('canvas').addEventListener('pointerup',e=>{if(!dragStart)return;const p=canvasPoint(e);state.roi={x:Math.min(dragStart.x,p.x),y:Math.min(dragStart.y,p.y),w:Math.abs(dragStart.x-p.x),h:Math.abs(dragStart.y-p.y)};dragStart=null;draw();updateRoiStats()});$('video').addEventListener('timeupdate',()=>{if(!state.details)return;const first=state.details.frames[0];const target=first.timestamp_s+$('video').currentTime;const i=state.details.frames.reduce((best,f,index)=>Math.abs(f.timestamp_s-target)<Math.abs(state.details.frames[best].timestamp_s-target)?index:best,0);if(i!==state.frame){state.frame=i;draw();updateRoiStats()}});init().catch(e=>{$('stats').textContent='页面初始化错误: '+e});
</script>'''


def finalize(output: Path) -> dict[str, Any]:
    manifest = read_json(output / "window_manifest.json")
    results = _load_front_results(output)
    comp_rows = []
    tracking_rows = []
    if (output / "component_diagnostics.csv").is_file():
        with (output / "component_diagnostics.csv").open(encoding="utf-8") as handle:
            comp_rows.extend(csv.DictReader(handle))
        comp_rows = [row for row in comp_rows if row.get("artifact_origin") == "historical_64"]
    for row in manifest:
        for density, _grid in GRID_CONFIG:
            result = results.get(_front_result_key(row["window_id"], density))
            if result is None:
                continue
            diag = result["diagnostic"]
            comp_rows.append(_diag_row(row, diag, artifact_origin="frontend_rerun", frozen_match=None))
            sequence = load_particle_sequence(result["sequence_prefix"])
            support = build_window_support(sequence, window_start_s=float(row["interval_start_s"]), component_config=COMPONENT_CONFIG)
            for validity_row in tracking_validity_rows(sequence, support, density_label=density):
                tracking_rows.append({"window_id": row["window_id"], "source_id": row["source_id"], "kind": row["kind"], "role": row["role"], **validity_row})
    # Re-write the combined files so resume cannot duplicate rows.
    if comp_rows:
        write_csv(output / "component_diagnostics.csv", comp_rows)
    historical_rows = []
    if (output / "tracking_and_validity.csv").is_file():
        with (output / "tracking_and_validity.csv").open(encoding="utf-8") as handle:
            historical_rows = [row for row in csv.DictReader(handle) if row.get("density") == "historical64"]
    if historical_rows or tracking_rows:
        write_csv(output / "tracking_and_validity.csv", historical_rows + tracking_rows)
    density_rows = []
    for row in manifest:
        for density, grid in GRID_CONFIG:
            result = results.get(_front_result_key(row["window_id"], density))
            if result is None:
                continue
            matched = result.get("matched_historical_64") or {}
            nested = result.get("matched_nested_64_vs_289") or {}
            density_rows.append({
                "window_id": row["window_id"], "source_id": row["source_id"], "kind": row["kind"], "role": row["role"],
                "density": density, "query_count": result["query_count"], "grid_size": grid,
                "tracking_visible_fraction": result["tracking_visible_fraction"], "geometry_valid_fraction": result["geometry_valid_fraction"],
                "min_geometry_valid_points": min(result["valid_points_per_frame"]), "mean_geometry_valid_points": float(np.mean(result["valid_points_per_frame"])),
                "component_count": result["component_count"], "max_component_members": result["max_component_members"],
                "qualified_graph_edge_count": result["qualified_graph_edge_count"], "component_internal_pair_count": result["component_internal_pair_count"],
                "qualified_graph_edge_count_within_components": result["diagnostic"].get("qualified_graph_edge_count_within_components"),
                "selected_triplet_common_members": result["selected_triplet_common_members"], "valid_triplet_count": result["valid_triplet_count"],
                "support_status": result["support_status"], "historical64_visibility_equal_fraction": matched.get("visibility_equal_fraction"),
                "historical64_uv_median_error_px": matched.get("uv_abs_error_median_px"), "historical64_xyz_median_error_m": matched.get("xyz_abs_error_median_m"),
                "historical64_visibility_disagreement_frames": matched.get("visibility_disagreement_frame_count"),
                "matched64_vs_289_visibility_equal_fraction": nested.get("visibility_equal_fraction"),
                "matched64_vs_289_uv_median_error_px": nested.get("uv_abs_error_median_px"),
                "matched64_vs_289_xyz_median_error_m": nested.get("xyz_abs_error_median_m"),
                "matched64_vs_289_visibility_disagreement_frames": nested.get("visibility_disagreement_frame_count"),
                "causal_training_eligible": result["causal_training_eligible"],
            })
    write_csv(output / "density_comparison.csv", density_rows)
    review_summary = build_review(output)
    analysis_summary = read_json(output / "analysis_summary.json")
    status = "OBSERVATION_DENSITY_DIAGNOSTIC_COMPLETE" if len(results) >= 32 and analysis_summary["frozen_component_mismatches"] == [] else "OBSERVATION_DENSITY_DIAGNOSTIC_INCOMPLETE"
    summary = {
        "status": status,
        "part_a": analysis_summary,
        "part_b": {
            "selected_windows": len(manifest),
            "density_results": len(results),
            "expected_density_results": 32,
            "sum_arm_elapsed_s": float(sum(float(item["elapsed_s"]) for item in results.values())),
            "max_arm_elapsed_s": float(max((float(item["elapsed_s"]) for item in results.values()), default=0.0)),
            "review": review_summary,
        },
        "roi_status": review_summary.get("roi_status", "ROI_ANNOTATION_READY"),
        "supported": {
            "nested_query_mapping": True,
            "same_depth_pose_per_density_window": all(item.get("depth_pose_computed_once_for_window") for item in results.values()),
            "historical_component_reproduction": analysis_summary["frozen_component_mismatches"] == [],
        },
        "not_supported": {
            "pixel_level_spatial_ground_truth": True,
            "new_detector_improvement": True,
            "cause_of_each_missing_observation": True,
        },
        "causal_training_qualification": "not eligible; frontend-only diagnostic without target construction/training",
    }
    write_json(output / "summary.json", summary)
    report = _report_text(summary, density_rows)
    (output / "experiment_report.md").write_text(report, encoding="utf-8")
    return summary


def _report_text(summary: dict[str, Any], density_rows: list[dict[str, Any]]) -> str:
    rows64 = [row for row in density_rows if row["density"] == "density64"]
    rows289 = [row for row in density_rows if row["density"] == "density289"]
    def mean(rows: list[dict[str, Any]], key: str) -> float | None:
        values = [float(row[key]) for row in rows if row.get(key) not in (None, "")]
        return float(np.mean(values)) if values else None
    return f"""# V7 observation density and local grouping diagnostic v1

## 结论先行

- 状态：`{summary['status']}`。
- 更多点是否真正增加可靠局部关系：本实验只在固定四源、16 窗口上测量；64→289 的可见/几何支撑与共同成员数见 `density_comparison.csv`，不能仅由点数增加宣称可靠关系增加。
- 当前大 component 主要暴露：component 是按固定直接边连通后的集合，内部 pair 数可远大于直接边；间接连通示例见 `indirect_connectivity_examples.json`。这说明需区分连通性与直接测量，不自动说明语义混合。
- 哪些失踪仍无法解释：ParticleSequence 的 mask 不能把真实遮挡与可见表面跟踪失败分开，也不能把 depth 无效与 pose 无效分开；这些记录为 `UNKNOWN`。

## 实际范围

Part A 重建全部 {summary['part_a']['windows']} 个历史64点窗口；冻结 component 完全复现 {summary['part_a']['frozen_component_matches']} 个，未复现 {len(summary['part_a']['frozen_component_mismatches'])} 个。Part B 对 {summary['part_b']['selected_windows']} 个固定窗口分别运行64和289点，得到 {summary['part_b']['density_results']} 个密度结果。

历史64与本轮匹配64的差异只作为重算审计；289点共享同一窗口的 depth、intrinsics 和 pose 结果，不把历史64→289直接称作纯密度差异。

## 限制

本轮不训练检测器、不使用真假标签做分组、不改变 component 阈值、不修改正式检测链。当前空间 ROI 状态为 `{summary['roi_status']}`；没有人工确认矩形时只交付 `roi_review/` 的标注入口与离线计数协议，不能报告像素级或局部伪造覆盖率。输出不是训练就绪的正式数据集。

ROI 收尾产物位于 `roi_review/`：`annotations.json` 保留人工输入，`roi_validation.csv` 校验 source/frame/PTS/原图坐标，`per_frame_coverage.csv` 和 `per_component_triplet_coverage.csv` 只统计实际保存的共同成员与无序 pair，`density_comparison.csv` 仅在有效 ROI 存在时产生对照。单帧 ROI 不自动传播到其他 triplet 时刻；统计不是准确率或检测概率。

审查页面：`review/index.html`。可在数据派生目录启动 `python3 -m http.server 8765`，访问 `/v7_activityforensics_observation_density_diagnostic_v1/review/`；页面依赖既有短片副本，不使用外部 CDN。服务器未执行浏览器视觉验收，仅做静态 HTTP/JSON 检查。

## 后续唯一优先建议

若增密后共同成员仍低且直接边占比不改善，优先进行局部测量组织/追踪输入审查；否则才进入匹配的检测验证。不要把缺失率直接变成分类特征。
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("prepare", "analysis", "frontend", "review", "finalize", "all"), required=True)
    parser.add_argument("--output", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--only-window")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    output = args.output
    if args.phase in ("prepare", "analysis", "frontend", "all") and not (output / "window_manifest.json").is_file():
        prepare_manifest(output)
    if args.phase == "prepare":
        return
    if args.phase == "analysis":
        run_analysis(output)
        return
    if args.phase == "frontend":
        run_frontend(output, only_window=args.only_window, resume=args.resume)
        return
    if args.phase == "review":
        build_review(output)
        return
    if args.phase == "finalize":
        finalize(output)
        return
    run_analysis(output)
    run_frontend(output, resume=args.resume)
    finalize(output)


if __name__ == "__main__":
    main()
