"""Run the frozen V7 representation chain on the exact ten Pair2 fake videos.

This is an experiment adapter only. It validates the already frozen fake
lineage, runs the unchanged frontend/component chain, and never fits or
changes a normality model.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import platform
import shutil

from research_tools.v7.feasibility.materialize_vript_frozen_targets import validate_video

from .protocol import (
    DATA_ROOT,
    PAIR2_REVISION,
    file_identity,
    read_json,
    sha256_file,
    write_json,
)
from .run_pair2_representation import DEPTH_PRO_SHA, TAPNET_SHA, run_representation


FAKE_GENERATORS = ("svd", "cogvideo")
OUTPUT_ROOT = DATA_ROOT / "derived/v7_unpaired_fake_response_v1"
FAKE_MANIFEST = DATA_ROOT / "derived/v7_pair2_detection_pilot_v1/manifests/prefix_fake_manifest.json"
FROZEN_MODEL_PATH = DATA_ROOT / "derived/v7_real_only_normality_pilot_v1/models/normality_models.json"
FROZEN_MODEL_SHA256 = "4e02e45bd56be8a7081c2c2af5046fbe7697434b110d016313b4d502b3d32847"


def verify_frozen_model(path: Path = FROZEN_MODEL_PATH) -> dict[str, object]:
    actual = sha256_file(path) if path.is_file() else None
    if actual != FROZEN_MODEL_SHA256:
        raise RuntimeError("FROZEN_MODEL_IDENTITY_MISMATCH")
    return {"path": str(path), "sha256": actual, "fitting_population": "real_train_only", "fake_count": 0, "refit": False}


def validate_fake_identity(rows: object) -> list[dict[str, object]]:
    if not isinstance(rows, list) or len(rows) != 10:
        raise ValueError("exact frozen fake population must contain ten rows")
    result = [dict(row) for row in rows]
    if any(row.get("status") != "MEDIA_VALID" or row.get("role") != "fake" for row in result):
        raise ValueError("fake manifest contains a non-MEDIA_VALID or non-fake row")
    if {str(row.get("generator")) for row in result} != set(FAKE_GENERATORS):
        raise ValueError("fake generator set changed")
    if any(sum(str(row.get("generator")) == generator for row in result) != 5 for generator in FAKE_GENERATORS):
        raise ValueError("fake generator count changed")
    if {str(row.get("pair_key")) for row in result} != {"00015", "00018", "00033", "00042", "00044"}:
        raise ValueError("fake Pair2 ordinal set changed")
    if len({str(row.get("video_id")) for row in result}) != 10:
        raise ValueError("fake video identity is not unique")
    for row in result:
        generator = str(row["generator"])
        relative = str(row.get("official_relative_path", ""))
        if not relative.startswith(f"Pair2/{generator}/"):
            raise ValueError("fake official path does not match generator")
        if str(row.get("pair_lineage")) != "shared Pair2 ordinal and official HDVG semantic record":
            raise ValueError("fake official lineage changed")
        path = Path(str(row["video_path"]))
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(f"fake media is missing or empty: {path}")
    return result


def validate_fake_media(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    """Recheck container, RGB decode, and strictly increasing source PTS."""

    validated: list[dict[str, object]] = []
    for row in rows:
        path = Path(str(row["video_path"]))
        check = validate_video(path)
        if check.get("status") != "MEDIA_VALID":
            raise RuntimeError(f"MEDIA_DECODE_FAILURE: {path}: {check.get('error')}")
        validated.append({**row, **check, "status": "MEDIA_VALID"})
    return validated


def prepare_fake_population(output: Path, manifest_path: Path = FAKE_MANIFEST) -> tuple[Path, list[dict[str, object]]]:
    rows = validate_fake_media(validate_fake_identity(read_json(manifest_path)))
    manifest = output / "manifests" / "fake_media_manifest.json"
    write_json(manifest, rows)
    write_json(output / "manifests" / "fake_population.json", {
        "dataset": "GenVidBench",
        "revision": PAIR2_REVISION,
        "population": "V7_UNPAIRED_FAKE_RESPONSE_V1",
        "fake_count": len(rows),
        "generator_counts": {generator: sum(str(row["generator"]) == generator for row in rows) for generator in FAKE_GENERATORS},
        "pair2_ordinals": sorted({str(row["pair_key"]) for row in rows}),
        "fake_used_for_fitting": False,
        "lineage_manifest": file_identity(manifest_path),
    })
    return manifest, rows


def run(args: argparse.Namespace) -> dict[str, object]:
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    model_identity = verify_frozen_model()
    manifest, media_rows = prepare_fake_population(output, args.fake_manifest)
    representation = run_representation(
        output=output,
        tapnet_source=args.tapnet_source,
        tapnet_checkpoint=args.tapnet_checkpoint,
        depth_source=args.depth_source,
        depth_checkpoint=args.depth_checkpoint,
        media_manifest_path=manifest,
    )
    windows = read_json(output / "manifests" / "pair2_window_manifest.json")
    if not isinstance(windows, list):
        raise ValueError("representation runner did not produce a window manifest")
    shutil.copyfile(output / "manifests" / "pair2_window_manifest.json", output / "manifests" / "fake_window_manifest.json")
    if len(media_rows) != 10 or len(windows) != 30:
        raise RuntimeError("UNEXPECTED_FAKE_WINDOW_COUNT")
    if any(str(row["status"]) != "AVAILABLE" for row in windows):
        raise RuntimeError("FAKE_WINDOW_DURATION_UNAVAILABLE")
    meta_path = output / "frontend" / "run_meta.json"
    meta = read_json(meta_path)
    if not isinstance(meta, dict):
        raise ValueError("frontend metadata has invalid shape")
    meta.update({
        "population": "V7_UNPAIRED_FAKE_RESPONSE_V1",
        "source_domain": "GenVidBench_Pair2",
        "fake_count": 10,
        "generator_counts": {generator: sum(str(row["generator"]) == generator for row in media_rows) for generator in FAKE_GENERATORS},
        "pair2_ordinals": sorted({str(row["pair_key"]) for row in media_rows}),
        "fake_used_for_fitting": False,
        "normality_refit": False,
        "frozen_model": model_identity,
        "providers": {
            **dict(meta.get("providers", {})),
            "tapnet_source_sha": TAPNET_SHA,
            "depth_source_sha": DEPTH_PRO_SHA,
        },
    })
    write_json(meta_path, meta)
    write_json(output / "metrics" / "frontend_summary.json", {**dict(read_json(output / "metrics" / "frontend_summary.json")), "population": "V7_UNPAIRED_FAKE_RESPONSE_V1"})
    summary = {
        "status": "FAKE_REPRESENTATION_COMPLETE",
        "population": "V7_UNPAIRED_FAKE_RESPONSE_V1",
        "media_count": len(media_rows),
        "windows_requested": len(windows),
        "windows_run": len(representation["windows"]),
        "frozen_model": model_identity,
        "fake_used_for_fitting": False,
        "normality_refit": False,
        "frontend_unchanged": True,
        "causal_execution": True,
    }
    write_json(output / "run_summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--fake-manifest", type=Path, default=FAKE_MANIFEST)
    parser.add_argument("--tapnet-source", type=Path, required=True)
    parser.add_argument("--tapnet-checkpoint", type=Path, required=True)
    parser.add_argument("--depth-source", type=Path, required=True)
    parser.add_argument("--depth-checkpoint", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
