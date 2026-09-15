from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from sparse3d_forgery.video_input import DecodedFrame, DecodedVideoSample

from research_tools.v7.periodic_requery_probe import runner
from research_tools.v7.periodic_requery_probe.runner import (
    OFFSETS_S,
    _atomic_json,
    _build_sequence,
    _persist_parent_completion,
    _support_for_sequence,
    Budget,
    _window_label,
    _choose_parents,
)


def test_undefined_support_summary_round_trips_as_null_with_reason(tmp_path: Path) -> None:
    path = tmp_path / "support.json"
    value = {
        "support_status": "NO_VALID_FIVE_TIME_UNIT",
        "support_reasons": ["COMMON_VALID_MEMBERS_LT3"],
        "valid_unit_count": 0,
        "intervals_s": None,
        "features": {"SET_A": None},
    }
    _atomic_json(path, value)
    loaded = json.loads(path.read_text())
    assert loaded["intervals_s"] is None
    assert loaded["features"]["SET_A"] is None
    assert loaded["support_reasons"] == ["COMMON_VALID_MEMBERS_LT3"]
    assert loaded["valid_unit_count"] == 0


def test_nonfinite_valid_feature_is_rejected_with_field_path(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match=r"\$\.features\.SET_A\[0\]\[0\]"):
        _atomic_json(tmp_path / "support.json", {"features": {"SET_A": np.asarray([[np.nan]])}})


def test_current_failure_replaces_stale_final_status(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _atomic_json(tmp_path / "final_status.json", {"status": "STOPPED_SAFE"})
    _atomic_json(tmp_path / "progress.json", {"stage": "features", "completed": 12, "total": 96})
    monkeypatch.setattr(runner, "_git_head", lambda: "abc")
    try:
        raise ValueError("bad support")
    except ValueError as exc:
        runner._record_current_failure(tmp_path, "features", exc)
    final = json.loads((tmp_path / "final_status.json").read_text())
    progress = json.loads((tmp_path / "progress.json").read_text())
    assert final["status"] == "FAILED"
    assert final["failed_stage"] == "features"
    assert final["failure_reason"] == "ValueError: bad support"
    assert "ValueError: bad support" in final["traceback"]
    assert progress["status"] == "FAILED"
    assert progress["completed"] == 12
    assert progress["total"] == 96


def test_periodic_support_uses_five_time_units(monkeypatch: pytest.MonkeyPatch) -> None:
    sequence = SimpleNamespace(
        timestamps_s=np.asarray([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]),
        frame_indices=np.arange(10, dtype=np.int64),
        xyz=np.zeros((10, 3, 3), dtype=np.float32),
        geometry_validity=np.ones((10, 3), dtype=np.bool_),
    )
    monkeypatch.setattr(runner, "load_particle_sequence", lambda _prefix: sequence)
    monkeypatch.setattr(runner, "rebuild_components_fast", lambda *_args: ((0, 1, 2),))
    monkeypatch.setattr(
        runner,
        "build_local_groups",
        lambda *_args, **_kwargs: {
            "groups": [{"retained": True, "member_slots": [0, 1, 2], "local_group_id": 7}],
            "retained_group_count": 1,
        },
    )
    monkeypatch.setattr(
        runner,
        "build_five_time_unit",
        lambda *_args, **_kwargs: {
            "status": "VALID",
            "local_group_id": 7,
            "timestamps_s": [0.5, 0.6, 0.7, 0.8, 0.9],
            "states": np.ones((5, 4), dtype=np.float64),
        },
    )
    output = _support_for_sequence(
        Path("unused"),
        {
            "window_id": "W",
            "source_id": "S",
            "pair_id": "P",
            "role": "real",
            "kind": "MANIP",
            "label": 0,
            "annotation_category": "REAL_NEGATIVE",
            "offset_s": 0.0,
            "interval_start_s": 0.0,
            "interval_end_s": 1.0,
        },
        "R",
    )
    assert output["support_status"] == "VALID"
    assert output["valid_unit_count"] == 1
    assert output["features"]["SET_A"].shape == (1, 5, 4)
    assert output["intervals_s"] == pytest.approx([0.1, 0.1, 0.1, 0.1])
    assert "states" not in output["support"]["units"][0]


def test_periodic_support_records_no_retained_group_reason(monkeypatch: pytest.MonkeyPatch) -> None:
    sequence = SimpleNamespace(
        timestamps_s=np.asarray([0.0, 0.1]),
        frame_indices=np.arange(2, dtype=np.int64),
        xyz=np.zeros((2, 3, 3), dtype=np.float32),
        geometry_validity=np.ones((2, 3), dtype=np.bool_),
    )
    monkeypatch.setattr(runner, "load_particle_sequence", lambda _prefix: sequence)
    monkeypatch.setattr(runner, "rebuild_components_fast", lambda *_args: ())
    monkeypatch.setattr(
        runner,
        "build_local_groups",
        lambda *_args, **_kwargs: {"groups": [], "retained_group_count": 0},
    )
    output = _support_for_sequence(
        Path("unused"),
        {
            "window_id": "W",
            "source_id": "S",
            "pair_id": "P",
            "role": "real",
            "kind": "MANIP",
            "label": 0,
            "annotation_category": "REAL_NEGATIVE",
            "offset_s": 0.0,
            "interval_start_s": 0.0,
            "interval_end_s": 1.0,
        },
        "O",
    )
    assert output["intervals_s"] is None
    assert output["support_reasons"] == ["NO_RETAINED_LOCAL_GROUP"]


def test_label_mapping_is_applied_after_fixed_sampling() -> None:
    segments = [{"start_s": 10.0, "end_s": 20.0}]
    assert _window_label("real", 12.0, 13.0, segments) == (0, "REAL_NEGATIVE")
    assert _window_label("fake", 12.0, 13.0, segments) == (1, "FAKE_MANIPULATION")
    assert _window_label("fake", 9.5, 10.5, segments) == (None, "BOUNDARY_MIXED")
    assert _window_label("fake", 21.0, 22.0, segments) == (None, "OUTSIDE_ANNOTATED_MANIPULATION")


def test_frozen_parent_selection_has_three_offsets_without_score_fields() -> None:
    parents, subwindows = _choose_parents()
    assert len(parents) == 32
    assert len(subwindows) == 96
    assert sorted({float(row["offset_s"]) for row in subwindows}) == list(OFFSETS_S)
    assert all(row["selection_rule"] for row in parents if row.get("status") == "PLANNED")
    assert not any("score" in key.lower() for row in subwindows for key in row)


def test_window_ids_keep_query_cohorts_separate() -> None:
    parents, subwindows = _choose_parents()
    assert len({row["window_id"] for row in subwindows}) == len(subwindows)
    for row in subwindows:
        assert row["parent_id"] in {parent["parent_id"] for parent in parents}
        assert row["frame_indices"] == sorted(row["frame_indices"])
        assert len(row["frame_indices"]) == len(row["timestamps_s"])
        if row["timestamps_s"]:
            assert np.all(np.diff(np.asarray(row["timestamps_s"], dtype=float)) > 0)


def test_build_sequence_converts_geometry_dtype_without_repairing_missing() -> None:
    frames = tuple(
        DecodedFrame(
            source_frame_index=index,
            timestamp_s=float(index) * 0.1,
            rgb=np.zeros((4, 5, 3), dtype=np.uint8),
        )
        for index in range(3)
    )
    decoded = DecodedVideoSample("sample", "video", frames)
    visibility = np.ones((3, 289), dtype=np.bool_)
    geometry = np.ones((3, 289), dtype=np.bool_)
    visibility[1, 7] = False
    geometry[1, 7] = False
    xyz = np.zeros((3, 289, 3), dtype=np.float64)
    xyz[1, 7] = np.nan
    uv = np.zeros((3, 289, 2), dtype=np.float32)
    uv[1, 7] = np.nan
    sequence = _build_sequence(
        decoded,
        uv=uv,
        visibility=visibility,
        xyz=xyz,
        geometry_valid=geometry,
        row={"source_id": "S", "pair_id": "P", "role": "real", "window_id": "W"},
        mode="O",
        parent_id="PARENT",
        cohort_start_s=0.0,
    )
    assert sequence.xyz.dtype == np.float32
    assert np.isnan(sequence.xyz[1, 7]).all()
    assert not sequence.geometry_validity[1, 7]


def test_parent_completion_persists_results_progress_and_budget_once(tmp_path: Path) -> None:
    root = tmp_path / "pilot"
    result_path = root / "frontend" / "results.json"
    budget = Budget(root, "frontend", 100.0)
    results: dict[str, dict[str, object]] = {}
    generated = [
        {"window_id": f"P::b{index}", "status": "FRONTEND_COMPLETE"}
        for index in range(3)
    ]
    _persist_parent_completion(
        root,
        result_path,
        results,
        generated,
        parent_id="P",
        parent_elapsed_s=1.25,
        budget=budget,
        parent_completed=1,
        total_windows=3,
    )
    saved = json.loads(result_path.read_text())
    progress = json.loads((root / "progress.json").read_text())
    budget_state = json.loads((root / "state" / "frontend_budget.json").read_text())
    assert len(saved) == 3
    assert progress["status"] == "RUNNING"
    assert progress["completed"] == 3
    assert progress["total"] == 3
    assert budget_state["elapsed_before_this_process_s"] == 0.0
    assert budget_state["cumulative_s"] >= budget_state["process_elapsed_s"]
