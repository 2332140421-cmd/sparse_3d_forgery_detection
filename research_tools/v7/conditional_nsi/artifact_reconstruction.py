"""Deterministically reconstruct the scalars needed by conditional NSI.

The caller and output are the frozen NSI triplet artifact and a compact CSV
completion.  Reconstructing from saved XYZ/component artifacts is sufficient;
rerunning any frontend or component discovery would add unnecessary state.
"""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from research_tools.v7.normalized_innovation.normalized_innovation import normalized_structural_innovation
from research_tools.v7.normalized_innovation.structural_trajectory import _relation_trajectories


PREVIOUS_ROOT = Path("/root/autodl-tmp/data/sparse_3d_forgery_detection/derived/v7_activityforensics_paired_second_order_pilot_v1")
NSI_ROOT = Path("/root/autodl-tmp/data/sparse_3d_forgery_detection/derived/v7_activityforensics_normalized_structural_innovation_pilot_v1")
OUTPUT_ROOT = Path("/root/autodl-tmp/data/sparse_3d_forgery_detection/derived/v7_activityforensics_conditional_nsi_pilot_v1")


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False, ensure_ascii=False) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def triplet_identity(row: dict[str, Any]) -> tuple[Any, ...]:
    """Identity used for exact stored/reconstructed triplet matching."""

    return (
        str(row["source_id"]),
        str(row["role"]),
        str(row["window_id"]),
        int(row["component_index"]),
        int(row["frame_index_prev"]),
        int(row["frame_index"]),
        int(row["frame_index_next"]),
    )


def reconstruct_triplet(
    npz_path: str | Path,
    members: list[int],
    center_index: int,
    timestamps_s: np.ndarray,
    frame_indices: np.ndarray,
) -> dict[str, Any]:
    """Rebuild the exact relation coordinate set and NSI for one triplet."""

    reconstructed = _reconstruct_component(npz_path, members, timestamps_s, frame_indices).get(int(center_index))
    if reconstructed is None:
        raise ValueError("triplet has no common persistent pair coordinates")
    return reconstructed


def _reconstruct_component(
    npz_path: str | Path,
    members: list[int],
    timestamps_s: np.ndarray,
    frame_indices: np.ndarray,
) -> dict[int, dict[str, Any]]:
    """Rebuild all valid centers for one component in one artifact pass."""

    with np.load(npz_path, allow_pickle=False) as arrays:
        xyz = np.asarray(arrays["xyz"], dtype=np.float64)
        validity = np.asarray(arrays["geometry_validity"], dtype=bool)
        stored_timestamps = np.asarray(arrays["timestamps_s"], dtype=np.float64)
        stored_indices = np.asarray(arrays["frame_indices"], dtype=np.int64)
    if not np.array_equal(stored_timestamps, timestamps_s) or not np.array_equal(stored_indices, frame_indices):
        raise ValueError("reconstruction timestamps/frame indices do not match frozen particle artifact")
    trajectories, masks, _scales = _relation_trajectories(xyz, validity, timestamps_s, members)
    persistent = {pair: int(np.sum(mask)) >= 8 for pair, mask in masks.items()}
    result: dict[int, dict[str, Any]] = {}
    for center in range(1, len(timestamps_s) - 1):
        if frame_indices[center] - frame_indices[center - 1] != 1 or frame_indices[center + 1] - frame_indices[center] != 1:
            continue
        pairs = [
            pair
            for pair in sorted(trajectories)
            if persistent[pair]
            and masks[pair][center - 1]
            and masks[pair][center]
            and masks[pair][center + 1]
        ]
        if not pairs:
            continue
        h0 = float(timestamps_s[center] - timestamps_s[center - 1])
        h1 = float(timestamps_s[center + 1] - timestamps_s[center])
        if not (np.isfinite(h0) and np.isfinite(h1) and h0 > 0 and h1 > 0):
            continue
        r_minus = np.asarray([trajectories[pair][center - 1] for pair in pairs], dtype=np.float64)
        r_zero = np.asarray([trajectories[pair][center] for pair in pairs], dtype=np.float64)
        r_plus = np.asarray([trajectories[pair][center + 1] for pair in pairs], dtype=np.float64)
        v_minus = (r_zero - r_minus) / h0
        v_plus = (r_plus - r_zero) / h1
        if not (np.all(np.isfinite(v_minus)) and np.all(np.isfinite(v_plus))):
            continue
        v_minus_norm = float(np.linalg.norm(v_minus))
        v_plus_norm = float(np.linalg.norm(v_plus))
        innovation = normalized_structural_innovation(v_minus, v_plus, epsilon=1e-8)
        M = len(pairs)
        result[center] = {
            "I_reconstructed": innovation,
            "v_minus_norm": v_minus_norm,
            "v_plus_norm": v_plus_norm,
            "M": M,
            "C": float(v_minus_norm / np.sqrt(M)),
            "pair_identity_count": M,
            "h0_s": h0,
            "h1_s": h1,
        }
    return result


def _load_csv(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def reconstruct(output: Path = OUTPUT_ROOT) -> dict[str, Any]:
    """Complete a fresh conditional artifact root and enforce fidelity."""

    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty conditional artifact root: {output}")
    stored_path = NSI_ROOT / "metrics/per_triplet_nsi.csv"
    results_path = PREVIOUS_ROOT / "frontend/window_results.json"
    selected_path = PREVIOUS_ROOT / "manifests/selected_pairs.json"
    windows_path = PREVIOUS_ROOT / "manifests/window_manifest.json"
    for path in (stored_path, results_path, selected_path, windows_path):
        if not path.exists():
            raise RuntimeError(f"NSI_RECONSTRUCTION_INPUT_INSUFFICIENT: missing {path}")
    stored = _load_csv(stored_path)
    results = json.loads(results_path.read_text(encoding="utf-8"))
    selected = json.loads(selected_path.read_text(encoding="utf-8"))
    windows = json.loads(windows_path.read_text(encoding="utf-8"))
    if len(selected) != 16 or len(windows) != 192:
        raise RuntimeError("NSI_RECONSTRUCTION_INPUT_INSUFFICIENT: frozen population/window count mismatch")
    result_by_id = {row["window"]["window_id"]: row for row in results}
    if len(result_by_id) != len(results):
        raise RuntimeError("NSI_RECONSTRUCTION_INPUT_INSUFFICIENT: duplicate frontend window identity")
    component_by_window = {
        window_id: {int(component["component_index"]): component for component in row["features"]["component_rows"]}
        for window_id, row in result_by_id.items()
    }
    identity_counts = Counter(triplet_identity(row) for row in stored)
    duplicate_count = sum(count - 1 for count in identity_counts.values() if count > 1)
    output.mkdir(parents=True, exist_ok=True)
    reconstruction_rows: list[dict[str, Any]] = []
    identity_missing = 0
    nonfinite = 0
    component_cache: dict[tuple[str, int], dict[int, dict[str, Any]]] = {}
    for row in stored:
        result = result_by_id.get(str(row["window_id"]))
        component = component_by_window.get(str(row["window_id"]), {}).get(int(row["component_index"]))
        if result is None or component is None:
            identity_missing += 1
            continue
        try:
            timestamps = np.asarray(result["timestamps_s"], dtype=np.float64)
            frame_indices = np.asarray(result["frame_indices"], dtype=np.int64)
            if (
                int(frame_indices[int(row["center_index"]) - 1]) != int(row["frame_index_prev"])
                or int(frame_indices[int(row["center_index"])]) != int(row["frame_index"])
                or int(frame_indices[int(row["center_index"]) + 1]) != int(row["frame_index_next"])
                or not np.isclose(float(timestamps[int(row["center_index"]) - 1]), float(row["timestamp_prev"]), rtol=1e-12, atol=1e-12)
                or not np.isclose(float(timestamps[int(row["center_index"])]), float(row["timestamp"]), rtol=1e-12, atol=1e-12)
                or not np.isclose(float(timestamps[int(row["center_index"]) + 1]), float(row["timestamp_next"]), rtol=1e-12, atol=1e-12)
            ):
                raise ValueError("stored triplet frame identity does not match frozen window")
            cache_key = (str(row["particle_artifact"]), int(row["component_index"]))
            if cache_key not in component_cache:
                component_cache[cache_key] = _reconstruct_component(
                    row["particle_artifact"], list(component["members"]), timestamps, frame_indices
                )
            reconstructed = component_cache[cache_key].get(int(row["center_index"]))
            if reconstructed is None:
                raise ValueError("triplet has no common persistent pair coordinates")
        except (OSError, KeyError, ValueError, RuntimeError):
            identity_missing += 1
            continue
        stored_i = float(row["structural_innovation"])
        error = abs(reconstructed["I_reconstructed"] - stored_i)
        if not all(np.isfinite(value) for value in (stored_i, reconstructed["I_reconstructed"], error, reconstructed["v_minus_norm"], reconstructed["v_plus_norm"], reconstructed["C"])):
            nonfinite += 1
        reconstruction_rows.append(
            {
                "source_id": row["source_id"],
                "role": row["role"],
                "window_id": row["window_id"],
                "pair_id": row["pair_id"],
                "kind": row["kind"],
                "label": row["label"],
                "anchor_fraction": row["anchor_fraction"],
                "component_index": row["component_index"],
                "center_index": row["center_index"],
                "frame_index_prev": row["frame_index_prev"],
                "frame_index": row["frame_index"],
                "frame_index_next": row["frame_index_next"],
                "timestamp_prev": row["timestamp_prev"],
                "timestamp": row["timestamp"],
                "timestamp_next": row["timestamp_next"],
                "particle_artifact": row["particle_artifact"],
                "I_stored": stored_i,
                "I_reconstructed": reconstructed["I_reconstructed"],
                "abs_error": error,
                "v_minus_norm": reconstructed["v_minus_norm"],
                "v_plus_norm": reconstructed["v_plus_norm"],
                "M": reconstructed["M"],
                "C": reconstructed["C"],
                "h0_s": reconstructed["h0_s"],
                "h1_s": reconstructed["h1_s"],
            }
        )
    errors = np.asarray([row["abs_error"] for row in reconstruction_rows], dtype=np.float64)
    n_total = len(stored)
    n_reconstructed = len(reconstruction_rows)
    identity_coverage = float(n_reconstructed / n_total) if n_total else 0.0
    max_error = float(np.max(errors)) if errors.size else None
    median_error = float(np.median(errors)) if errors.size else None
    p99_error = float(np.percentile(errors, 99)) if errors.size else None
    rmse = float(np.sqrt(np.mean(errors**2))) if errors.size else None
    allclose = bool(errors.size == n_total and np.allclose([row["I_reconstructed"] for row in reconstruction_rows], [row["I_stored"] for row in reconstruction_rows], rtol=1e-8, atol=1e-10))
    verified = bool(identity_coverage == 1.0 and duplicate_count == 0 and nonfinite == 0 and allclose and max_error is not None and max_error <= 1e-8)
    _write_csv(output / "reconstruction/per_triplet_reconstruction.csv", reconstruction_rows)
    summary = {
        "status": "NSI_RECONSTRUCTION_VERIFIED" if verified else "NSI_RECONSTRUCTION_MISMATCH",
        "n_total_stored": n_total,
        "n_reconstructed": n_reconstructed,
        "identity_coverage": identity_coverage,
        "n_identity_missing": identity_missing,
        "n_duplicate_identity": duplicate_count,
        "n_nonfinite": nonfinite,
        "max_abs_error": max_error,
        "median_abs_error": median_error,
        "p99_abs_error": p99_error,
        "rmse": rmse,
        "allclose_rtol": 1e-8,
        "allclose_atol": 1e-10,
        "relation_epsilon": 1e-8,
        "nsi_epsilon": 1e-8,
        "population_pairs": len(selected),
        "frozen_windows": len(windows),
        "frontend_rerun": False,
        "formal_src_modified": False,
    }
    _write_json(output / "reconstruction/reconstruction_summary.json", summary)
    _write_json(
        output / "manifests/source_artifact_links.json",
        {
            "previous_pilot_root": str(PREVIOUS_ROOT),
            "nsi_root": str(NSI_ROOT),
            "frontend_results": str(results_path),
            "selected_pairs": str(selected_path),
            "window_manifest": str(windows_path),
            "source_sha256": {
                "stored_per_triplet_nsi.csv": _sha256(stored_path),
                "window_results.json": _sha256(results_path),
                "selected_pairs.json": _sha256(selected_path),
                "window_manifest.json": _sha256(windows_path),
            },
            "frontend_rerun": False,
        },
    )
    if not verified:
        raise RuntimeError("NSI_RECONSTRUCTION_MISMATCH")
    return summary
