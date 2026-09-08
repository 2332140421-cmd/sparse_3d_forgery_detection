"""Run relation-first representation extraction on frozen V7 artifacts."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any

from .representation import relation_first_window


PREVIOUS_ROOT = Path("/root/autodl-tmp/data/sparse_3d_forgery_detection/derived/v7_activityforensics_paired_second_order_pilot_v1")
OUTPUT_ROOT = Path("/root/autodl-tmp/data/sparse_3d_forgery_detection/derived/v7_activityforensics_relation_first_pilot_v1")


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False, ensure_ascii=False) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def run(output: Path = OUTPUT_ROOT) -> dict[str, Any]:
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(output)
    output.mkdir(parents=True, exist_ok=True)
    selected = json.loads((PREVIOUS_ROOT / "manifests/selected_pairs.json").read_text(encoding="utf-8"))
    windows = json.loads((PREVIOUS_ROOT / "manifests/window_manifest.json").read_text(encoding="utf-8"))
    results = json.loads((PREVIOUS_ROOT / "frontend/window_results.json").read_text(encoding="utf-8"))
    if len(selected) != 16 or len(windows) != 192 or len(results) != 192:
        raise RuntimeError("frozen population/window artifacts do not have the expected 16/192/192 sizes")
    if any(item.get("status") != "COMPLETE" for item in results):
        raise RuntimeError("relation-first requires complete frozen frontend results")
    result_by_window = {item["window"]["window_id"]: item for item in results}
    source_links = {
        "previous_artifact_root": str(PREVIOUS_ROOT),
        "selected_pairs": str(PREVIOUS_ROOT / "manifests/selected_pairs.json"),
        "window_manifest": str(PREVIOUS_ROOT / "manifests/window_manifest.json"),
        "frontend_results": str(PREVIOUS_ROOT / "frontend/window_results.json"),
        "particle_artifacts": str(PREVIOUS_ROOT / "particles"),
        "component_membership": "frontend/window_results.json::features.component_rows",
        "frontend_rerun": False,
        "source_sha256": {name: _sha256(PREVIOUS_ROOT / name) for name in ("manifests/selected_pairs.json", "manifests/window_manifest.json", "frontend/window_results.json")},
    }
    _write_json(output / "manifests/source_artifact_links.json", source_links)
    _write_json(output / "manifests/frozen_population.json", selected)
    _write_json(output / "manifests/frozen_windows.json", windows)
    relation_rows: list[dict[str, Any]] = []
    coverage_rows: list[dict[str, Any]] = []
    for window in windows:
        result = result_by_window[window["window_id"]]
        prefix = Path(result["particle_prefix"])
        relation = relation_first_window(prefix.with_suffix(".npz"), result["features"]["component_rows"], result["timestamps_s"], result["frame_indices"])
        row = {"window": window, "relation": relation, "source_sha256": result["source_sha256"]}
        relation_rows.append(row)
        coverage_rows.append({"window_id": window["window_id"], "pair_id": window["pair_id"], "source_id": window["source_id"], "role": window["role"], "kind": window["kind"], "anchor_fraction": window["anchor_fraction"], "component_count": relation["component_count"], "component_with_valid_r1": relation["component_with_valid_r1"], "component_with_valid_r2": relation["component_with_valid_r2"], "persistent_pair_count": relation["persistent_pair_count"], "r1_eligible_pair_count": relation["r1_eligible_pair_count"], "r2_eligible_pair_count": relation["r2_eligible_pair_count"], "r1_eligible_fraction": relation["r1_eligible_fraction"], "r2_eligible_fraction": relation["r2_eligible_fraction"]})
    _write_json(output / "relations/per_window_relation_summary.json", relation_rows)
    _write_csv(output / "relations/pair_coverage.csv", coverage_rows)
    summary = {"status": "RELATION_EXTRACTION_COMPLETE", "selected_pairs": len(selected), "windows": len(windows), "frontend_rerun": False, "formal_src_modified": False}
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
