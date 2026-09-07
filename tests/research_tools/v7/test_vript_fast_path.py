from __future__ import annotations

import json
import zipfile
from pathlib import Path

from research_tools.v7.feasibility.materialize_vript_frozen_targets import (
    build_target_map,
    build_time_window,
    exact_member,
    first_valid_targets,
    parse_target_filename,
    safe_output_name,
)
from research_tools.v7.feasibility.run_vript_frontend_probe import build_window_manifest
from sparse3d_forgery.experiments.v7_dynamic_structure_probe import ComponentConfig


def _rows():
    frozen = [
        {
            "source_id": "Vript:00001:vid-Scene-001",
            "role": "train_real",
            "source_identity": "vid-Scene-001",
            "relative_path": "Pair1/vript/vid-Scene-001.mp4",
        },
        {
            "source_id": "Vript:00002:later-Scene-002",
            "role": "train_real",
            "source_identity": "later-Scene-002",
            "relative_path": "Pair1/vript/later-Scene-002.mp4",
        },
    ]
    prior = [
        {"source_id": frozen[0]["source_id"], "archive_member": "vript/vript_long_videos_clips/clips_007_of_1095/vid/vid-Scene-001.mp4"},
        {"source_id": frozen[1]["source_id"], "archive_member": "vript/vript_long_videos_clips/clips_008_of_1095/later/later-Scene-002.mp4"},
    ]
    official = {
        "vript_long_videos_clips/clips_007_of_1095.zip": {"size": 7},
        "vript_long_videos_clips/clips_008_of_1095.zip": {"size": 8},
    }
    return frozen, prior, official


def test_frozen_target_identity_and_exact_shard_are_preserved():
    frozen, prior, official = _rows()
    rows = build_target_map(frozen, prior, official)
    assert [row["source_id"] for row in rows] == [row["source_id"] for row in frozen]
    assert rows[0]["shard"].endswith("clips_007_of_1095.zip")
    assert rows[0]["member"] == "clips_007_of_1095/vid/vid-Scene-001.mp4"


def test_target_filename_parser_does_not_fuzzy_substitute():
    assert parse_target_filename("-C_-HNTztXI-Scene-005.mp4") == ("-C_-HNTztXI", "005")
    try:
        parse_target_filename("-C_-HNTztXI-Scene-005-copy.mp4")
    except ValueError:
        pass
    else:
        raise AssertionError("non-exact scene identity was accepted")


def test_exact_member_requires_filename_and_video_identity():
    entries = [
        {"name": "clips_1/vid/vid-Scene-001.mp4"},
        {"name": "clips_1/other/vid-Scene-001.mp4"},
    ]
    assert exact_member(entries, filename="vid-Scene-001.mp4", video_id="vid") ["name"] == entries[0]["name"]
    assert exact_member(entries, filename="vid-Scene-001.mp4", video_id="missing") is None


def test_first_valid_population_keeps_frozen_order_and_real_only():
    frozen, prior, official = _rows()
    rows = build_target_map(frozen, prior, official)
    assert [row["source_id"] for row in first_valid_targets(rows, {rows[0]["source_id"]: "MEDIA_VALID", rows[1]["source_id"]: "MEDIA_INVALID"}, count=1)] == [rows[0]["source_id"]]


def test_midpoint_windows_use_true_time_without_padding():
    values = [0.0, 0.1, 0.2, 0.4, 0.7, 1.0, 1.3, 1.5, 1.8, 2.0]
    window = build_time_window(values, 1.0)
    assert window["status"] == "AVAILABLE"
    assert window["anchor_timestamp_s"] == 1.0
    assert window["frame_indices"] == [4, 5, 6, 7]
    assert window["frame_indices"] == sorted(set(window["frame_indices"]))
    assert build_time_window(values, 3.0)["status"] == "WINDOW_DURATION_UNAVAILABLE"


def test_safe_output_name_is_deterministic():
    assert safe_output_name("Vript:1:x", "x-Scene-001.mp4") == safe_output_name("Vript:1:x", "x-Scene-001.mp4")


def test_formal_src_does_not_import_research_tools():
    for path in Path("src/sparse3d_forgery").rglob("*.py"):
        assert "research_tools" not in path.read_text(encoding="utf-8")


def test_component_config_and_structure_baseline_are_unchanged():
    config = ComponentConfig(max_initial_distance=1.0, max_relative_change=0.05, minimum_overlap=8, minimum_size=3)
    assert config.max_initial_distance == 1.0
    assert config.max_relative_change == 0.05
    assert config.minimum_overlap == 8
    assert config.minimum_size == 3


def test_per_video_timescale_pairing_is_exact_and_real_only(tmp_path: Path):
    media = [{"source_id": "Vript:1:x", "video_path": str(tmp_path / "x.mp4"), "status": "MEDIA_VALID", "timestamps_s": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]}]
    rows = build_window_manifest(media, tmp_path)
    assert len(rows) == 3
    assert {row["timescale_s"] for row in rows} == {0.5, 1.0, 2.0}
    assert all(row["real_only"] and not row["fake_used"] for row in rows)
