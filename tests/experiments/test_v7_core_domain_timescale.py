import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image
import pytest

from sparse3d_forgery.experiments.v7_core_domain_timescale import (
    FULL_VIDEO_COLUMNS,
    WINDOW_COLUMNS,
    _write_contact_sheet,
    build_window_review_rows,
    centered_window_selection,
    finalize_video_domain_review,
    load_phase_a_review_manifest,
    uniform_frame_indices,
    write_window_review_csv,
)


def _phase_a_manifest(path: Path, count: int = 96) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for index in range(count):
        split = "train" if index < 64 else "val"
        rows.append(
            {
                "source_video_id": f"video-{index:03d}",
                "split": split,
                "video_path": f"real_videos/video-{index:03d}.mp4",
                "frame_indices": list(range(16)),
                "status": "success",
            }
        )
    path.write_text(json.dumps({"candidates": rows}), encoding="utf-8")
    return rows


def test_phase_a_population_is_exactly_reused(tmp_path: Path):
    path = tmp_path / "review_manifest.json"
    rows = _phase_a_manifest(path)
    candidates = load_phase_a_review_manifest(path)
    assert [candidate.source_video_id for candidate in candidates] == [
        row["source_video_id"] for row in rows
    ]
    assert [candidate.split for candidate in candidates].count("train") == 64
    assert [candidate.split for candidate in candidates].count("val") == 32
    assert all(candidate.split in {"train", "val"} for candidate in candidates)


def test_uniform_samples_span_full_video_without_duplicate_padding():
    assert uniform_frame_indices(100, 20)[0:2] == (0, 5)
    samples = uniform_frame_indices(100, 20)
    assert samples[-1] == 99
    assert len(samples) == len(set(samples)) == 20
    assert uniform_frame_indices(3, 20) == (0, 1, 2)


def test_full_contact_sheet_is_four_by_five_and_uses_first_last_samples(tmp_path: Path):
    frames = []
    for index in (0, 99):
        frames.append(
            type(
                "Frame",
                (),
                {
                    "rgb": np.full((12, 16, 3), index, dtype=np.uint8),
                    "source_frame_index": index,
                    "timestamp_s": index / 30,
                },
            )()
        )
    path = tmp_path / "full.png"
    _write_contact_sheet(frames, path)
    with Image.open(path) as image:
        assert image.size == (1280, 1260)
        assert image.mode == "RGB"


def test_window_durations_anchors_and_boundary_clipping_are_deterministic():
    timestamps = np.arange(0.0, 3.01, 0.1)
    for duration in (0.5, 1.0, 2.0):
        first = centered_window_selection(
            timestamps, anchor_fraction=0.25, requested_duration_s=duration
        )
        second = centered_window_selection(
            timestamps, anchor_fraction=0.25, requested_duration_s=duration
        )
        assert first == second
        assert first.frame_start is not None and first.frame_end is not None
        assert first.frame_end >= first.frame_start
        assert first.timestamp_start_s <= first.timestamp_end_s
        assert first.actual_duration_s <= duration + 1e-12
    clipped = centered_window_selection(
        timestamps, anchor_fraction=0.25, requested_duration_s=2.0
    )
    assert clipped.frame_start == 0
    assert clipped.frame_end < len(timestamps)


def test_short_video_records_duration_unavailable_without_padding():
    selection = centered_window_selection(
        (0.0, 0.1, 0.2), anchor_fraction=0.75, requested_duration_s=0.5
    )
    assert selection.status == "WINDOW_DURATION_UNAVAILABLE"
    assert selection.frame_start == 0
    assert selection.frame_end == 2
    assert selection.actual_duration_s == pytest.approx(0.2)


def _small_full_manifest(path: Path) -> tuple[Path, Path]:
    candidates = []
    for source_video_id, split, decision, reason in (
        ("in", "train", "IN_DOMAIN", "SINGLE_STRUCTURED_MOTION"),
        ("uncertain", "val", "UNCERTAIN", "OTHER"),
    ):
        candidates.append(
            {
                "source_video_id": source_video_id,
                "split": split,
                "source_video_path": f"real_videos/{source_video_id}.mp4",
                "video_path": f"/data/{source_video_id}.mp4",
                "frame_indices": list(range(31)),
                "timestamps_s": [round(index * 0.1, 6) for index in range(31)],
                "duration_s": 3.0,
                "frame_count": 31,
                "video_decision": decision,
                "video_reason_code": reason,
                "video_notes": "",
            }
        )
    manifest = path / "full_video_manifest.json"
    manifest.write_text(json.dumps({"candidates": candidates}), encoding="utf-8")
    review = path / "full_video_review.csv"
    with review.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FULL_VIDEO_COLUMNS)
        writer.writeheader()
        for candidate in candidates:
            writer.writerow(
                {
                    "source_video_id": candidate["source_video_id"],
                    "split": candidate["split"],
                    "video_path": candidate["video_path"],
                    "duration_s": candidate["duration_s"],
                    "frame_count": candidate["frame_count"],
                    "full_contact_sheet_path": "/data/sheet.png",
                    "video_decision": candidate["video_decision"],
                    "video_reason_code": candidate["video_reason_code"],
                    "video_notes": "",
                }
            )
    return review, manifest


def test_window_generator_only_emits_video_in_domain_and_has_no_model_fields(tmp_path: Path):
    review, manifest = _small_full_manifest(tmp_path)
    rows = build_window_review_rows(review, manifest, output_root=tmp_path / "window")
    assert len(rows) == 9
    assert {row["source_video_id"] for row in rows} == {"in"}
    assert {row["anchor_fraction"] for row in rows} == {"0.25", "0.50", "0.75"}
    assert {row["requested_duration_s"] for row in rows} == {"0.500", "1.000", "2.000"}
    forbidden = {"component", "xyz", "tracking", "depth", "pose", "score", "fake", "label"}
    assert not forbidden.intersection(WINDOW_COLUMNS)
    assert not forbidden.intersection(FULL_VIDEO_COLUMNS)


def test_video_finalizer_excludes_uncertain_and_keeps_only_explicit_in_domain(tmp_path: Path):
    review, manifest = _small_full_manifest(tmp_path)
    output = tmp_path / "video_core_domain_manifest.json"
    result = finalize_video_domain_review(review, manifest, output)
    assert [row["source_video_id"] for row in result["selected"]] == ["in"]
    assert output.is_file()


def test_window_csv_schema_is_explicit_and_round_trips(tmp_path: Path):
    row = {column: "" for column in WINDOW_COLUMNS}
    row.update(
        {
            "source_video_id": "in",
            "split": "train",
            "anchor_fraction": "0.25",
            "requested_duration_s": "0.500",
            "actual_duration_s": "0.500000000",
            "frame_start": "0",
            "frame_end": "5",
            "timestamp_start_s": "0.000000000",
            "timestamp_end_s": "0.500000000",
            "window_reason_code": "WINDOW_DURATION_UNAVAILABLE",
        }
    )
    path = tmp_path / "window_review.csv"
    write_window_review_csv([row], path)
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        assert tuple(reader.fieldnames or ()) == WINDOW_COLUMNS
        assert next(reader)["source_video_id"] == "in"
