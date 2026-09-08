"""Extract normalized structural innovation from frozen V7 artifacts."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any

from .structural_trajectory import window_nsi


PREVIOUS_ROOT = Path("/root/autodl-tmp/data/sparse_3d_forgery_detection/derived/v7_activityforensics_paired_second_order_pilot_v1")
OUTPUT_ROOT = Path("/root/autodl-tmp/data/sparse_3d_forgery_detection/derived/v7_activityforensics_normalized_structural_innovation_pilot_v1")


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


def run(output: Path = OUTPUT_ROOT) -> dict[str, Any]:
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty artifact root: {output}")
    selected_path = PREVIOUS_ROOT / "manifests/selected_pairs.json"
    windows_path = PREVIOUS_ROOT / "manifests/window_manifest.json"
    results_path = PREVIOUS_ROOT / "frontend/window_results.json"
    for path in (selected_path, windows_path, results_path):
        if not path.exists():
            raise FileNotFoundError(path)
    selected = json.loads(selected_path.read_text(encoding="utf-8"))
    frozen_windows = json.loads(windows_path.read_text(encoding="utf-8"))
    previous_results = json.loads(results_path.read_text(encoding="utf-8"))
    if len(selected) != 16 or len(frozen_windows) != 192 or len(previous_results) != 192:
        raise RuntimeError("NSI_ARTIFACT_INSUFFICIENT: frozen population/window artifacts are incomplete")
    result_ids = {row["window"]["window_id"] for row in previous_results}
    if result_ids != {row["window_id"] for row in frozen_windows}:
        raise RuntimeError("frozen window identity mismatch")
    output.mkdir(parents=True, exist_ok=False)
    source_links = {
        "previous_pilot_root": str(PREVIOUS_ROOT),
        "selected_pairs": str(selected_path),
        "window_manifest": str(windows_path),
        "frontend_results": str(results_path),
        "particle_artifacts": str(PREVIOUS_ROOT / "particles"),
        "frontend_rerun": False,
        "source_sha256": {
            "selected_pairs.json": _sha256(selected_path),
            "window_manifest.json": _sha256(windows_path),
            "window_results.json": _sha256(results_path),
        },
    }
    _write_json(output / "manifests/source_artifact_links.json", source_links)
    _write_json(output / "manifests/frozen_population.json", selected)
    _write_json(output / "manifests/frozen_windows.json", frozen_windows)
    triplet_rows: list[dict[str, Any]] = []
    window_rows: list[dict[str, Any]] = []
    range_violations = 0
    for result in previous_results:
        window = result["window"]
        particle = Path(result["particle_prefix"]).with_suffix(".npz")
        if not particle.exists():
            raise RuntimeError(f"NSI_ARTIFACT_INSUFFICIENT: missing particle artifact {particle}")
        nsi = window_nsi(
            particle,
            result["features"]["component_rows"],
            result["timestamps_s"],
            result["frame_indices"],
        )
        range_violations += nsi["range_violations"]
        base = {
            "window_id": window["window_id"],
            "pair_id": window["pair_id"],
            "source_id": window["source_id"],
            "role": window["role"],
            "kind": window["kind"],
            "label": window["label"],
            "anchor_fraction": window["anchor_fraction"],
            "source_sha256": result.get("source_sha256"),
            "geometry_coverage": result.get("quality", {}).get("geometry_coverage"),
            "tracking_persistence": result.get("quality", {}).get("tracking_persistence"),
            "particle_artifact": str(particle),
            "I_window": nsi["I_window"],
            "component_count": nsi["component_count"],
            "component_with_valid_nsi": nsi["component_with_valid_nsi"],
            "triplet_count": nsi["triplet_count"],
            "persistent_pair_count": nsi["persistent_pair_count"],
            "common_pair_dimension_median": nsi["common_pair_dimension_median"],
            "speed_sum_median": nsi["speed_sum_median"],
            "raw_delta_r_median": nsi["raw_delta_r_median"],
            "raw_second_difference_median": nsi["raw_second_difference_median"],
            "range_violations": nsi["range_violations"],
        }
        window_rows.append(base)
        for component in nsi["components"]:
            for triplet in component["triplets"]:
                triplet_rows.append({**base, **triplet})
    _write_csv(output / "metrics/per_triplet_nsi.csv", triplet_rows)
    _write_csv(output / "metrics/per_window_nsi.csv", window_rows)
    summary = {
        "status": "NSI_EXTRACTION_COMPLETE",
        "selected_pairs": len(selected),
        "windows": len(window_rows),
        "triplets": len(triplet_rows),
        "range_violations": range_violations,
        "frontend_rerun": False,
        "formal_src_modified": False,
        "fake_count_in_calibration": 0,
        "normality_model": "NONE",
        "metrics": {
            "per_triplet": "metrics/per_triplet_nsi.csv",
            "per_window": "metrics/per_window_nsi.csv",
        },
    }
    _write_json(output / "run_summary.json", summary)
    return summary


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUTPUT_ROOT)
    args = parser.parse_args()
    print(json.dumps(run(args.output), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
