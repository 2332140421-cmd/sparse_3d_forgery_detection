import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image

from scripts.finalize_v7_core_domain_manifest import finalize_core_domain_review
from sparse3d_forgery.experiments.v7_core_domain import (
    ALLOWED_DECISIONS,
    ALLOWED_REASON_CODES,
    REVIEW_COLUMNS,
    CoreDomainCandidate,
    _write_contact_sheet,
    load_core_domain_candidates,
)


def write_manifest(path: Path) -> None:
    items = [
        {
            "role": "real_val",
            "official_split": "val",
            "source_video_id": "val-b",
            "video_path": "real_videos/val-b.mp4",
            "frame_indices": [10, 11],
            "component_count": 999,
        },
        {
            "role": "fake_probe",
            "official_split": "test",
            "source_video_id": "fake-a",
            "video_path": "fake_videos/fake-a.mp4",
            "frame_indices": [10, 11],
        },
        {
            "role": "real_train",
            "official_split": "train",
            "source_video_id": "train-b",
            "video_path": "real_videos/train-b.mp4",
            "frame_indices": [4, 5],
        },
        {
            "role": "real_train",
            "official_split": "train",
            "source_video_id": "train-a",
            "video_path": "real_videos/train-a.mp4",
            "frame_indices": [2, 3],
        },
    ]
    path.write_text(json.dumps({"items": items}), encoding="utf-8")


def test_population_is_sorted_split_preserving_and_ignores_fake_and_metrics(tmp_path: Path):
    manifest = tmp_path / "manifest.json"
    write_manifest(manifest)
    candidates = load_core_domain_candidates(manifest, real_train_limit=2, real_val_limit=1)
    assert [item.source_video_id for item in candidates] == ["train-a", "train-b", "val-b"]
    assert [item.split for item in candidates] == ["train", "train", "val"]
    assert all(item.split in ("train", "val") for item in candidates)


def test_review_schema_has_no_model_output_columns():
    forbidden = {"component_count", "xyz_outlier_rate", "tracking_quality", "depth_quality", "score", "label"}
    assert not forbidden.intersection(REVIEW_COLUMNS)
    assert set(ALLOWED_DECISIONS) == {"IN_DOMAIN", "OUT_OF_DOMAIN", "UNCERTAIN"}
    assert "SINGLE_STRUCTURED_MOTION" in ALLOWED_REASON_CODES


def test_contact_sheet_contains_all_sixteen_frames(tmp_path: Path):
    frames = []
    for index in range(16):
        image = np.full((12, 16, 3), index, dtype=np.uint8)
        frames.append(type("Frame", (), {"rgb": image, "source_frame_index": index, "timestamp_s": index / 10})())
    decoded = type("Decoded", (), {"frames": tuple(frames)})()
    candidate = CoreDomainCandidate("video", "train", "real_videos/video.mp4", tuple(range(16)))
    path = tmp_path / "sheet.png"
    _write_contact_sheet(decoded, candidate, path)
    with Image.open(path) as image:
        assert image.size == (1280, 1008)


def test_finalizer_excludes_uncertain_and_requires_in_domain_reason(tmp_path: Path):
    review_root = tmp_path
    candidates = [
        {"source_video_id": "a", "status": "success", "frame_indices": [0, 1], "timestamps_s": [0.0, 0.1]},
        {"source_video_id": "b", "status": "success", "frame_indices": [0, 1], "timestamps_s": [0.0, 0.1]},
        {"source_video_id": "c", "status": "success", "frame_indices": [0, 1], "timestamps_s": [0.0, 0.1]},
    ]
    (review_root / "review_manifest.json").write_text(json.dumps({"candidates": candidates}), encoding="utf-8")
    rows = [
        {"source_video_id": "a", "split": "train", "video_path": "real_videos/a.mp4", "frame_start": "0", "frame_end": "1", "preview_path": "raw_previews/a.mp4", "contact_sheet_path": "contact_sheets/a.png", "decision": "IN_DOMAIN", "reason_code": "SINGLE_STRUCTURED_MOTION", "notes": ""},
        {"source_video_id": "b", "split": "train", "video_path": "real_videos/b.mp4", "frame_start": "0", "frame_end": "1", "preview_path": "raw_previews/b.mp4", "contact_sheet_path": "contact_sheets/b.png", "decision": "UNCERTAIN", "reason_code": "OTHER", "notes": ""},
        {"source_video_id": "c", "split": "train", "video_path": "real_videos/c.mp4", "frame_start": "0", "frame_end": "1", "preview_path": "raw_previews/c.mp4", "contact_sheet_path": "contact_sheets/c.png", "decision": "OUT_OF_DOMAIN", "reason_code": "STATIC_DOMINANT", "notes": ""},
    ]
    with (review_root / "review_template.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=REVIEW_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    output = review_root / "core.json"
    result = finalize_core_domain_review(review_root, output)
    assert [row["source_video_id"] for row in result["selected"]] == ["a"]
    assert result["selected"][0]["frame_indices"] == [0, 1]
