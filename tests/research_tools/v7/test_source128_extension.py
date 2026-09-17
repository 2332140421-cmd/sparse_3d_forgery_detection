from __future__ import annotations

import json
from pathlib import Path

import pytest

from research_tools.v7.source128_extension import runner
from research_tools.v7.source128_extension.runner import _bootstrap, _effective_training_rows, _stable_key, atomic_json, report
from research_tools.v7.periodic_requery_probe.runner import Budget as PeriodicBudget


def test_selection_key_is_stable_and_order_independent() -> None:
    assert _stable_key("N79WJ") == _stable_key("N79WJ")
    assert _stable_key("N79WJ") != _stable_key("HWL2J")


def test_source_bootstrap_uses_paired_sources_and_fixed_seed() -> None:
    left = {"a": 0.5, "b": 1.0}
    right = {"a": 0.25, "b": 0.75}
    result = _bootstrap(left, right)
    assert result["source_count"] == 2
    assert result["mean"] == pytest.approx(0.25)
    assert result["seed"] == 20260909
    assert result["replicates"] == 10000


def test_atomic_json_rejects_nonfinite(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        atomic_json(tmp_path / "bad.json", {"value": float("nan")})


def test_recovery_frontend_budget_does_not_consume_or_rewrite_historical_budget(tmp_path: Path) -> None:
    historical = {
        "budget_s": 7200.0,
        "cumulative_s": 90.945,
        "estimated_reserved_s": 11161.097,
        "budget_accounted_upper_s": 11252.042,
        "process_state": "BUDGET_UNCERTAIN",
        "uncertain_attempt": {"lower_bound_s_approx": 5.0, "upper_bound_s": 11161.097, "actual_elapsed_s": None},
    }
    atomic_json(tmp_path / "state/frontend_budget.json", historical)

    budget = PeriodicBudget(tmp_path, "frontend_recovery", 5400.0)

    assert json.loads((tmp_path / "state/frontend_budget.json").read_text()) == historical
    recovery = json.loads((tmp_path / "state/frontend_recovery_budget.json").read_text())
    assert recovery["budget_s"] == 5400.0
    assert recovery["cumulative_s"] < 1.0
    assert recovery["estimated_reserved_s"] == 0.0
    assert budget.remaining() <= 5400.0
    assert budget.remaining() > 5399.0


def test_formal_training_recovery_budget_is_independent_of_exhausted_feature_budget(tmp_path: Path) -> None:
    historical = {
        "budget_s": 900.0,
        "cumulative_s": 2976.735,
        "budget_accounted_upper_s": 2976.735,
        "process_state": "TRAINING_BUDGET_EXHAUSTED",
    }
    atomic_json(tmp_path / "state/training_budget.json", historical)

    budget = runner._budget(tmp_path, 900.0, "training_recovery")

    assert json.loads((tmp_path / "state/training_budget.json").read_text()) == historical
    recovery = json.loads((tmp_path / "state/training_recovery_budget.json").read_text())
    assert recovery["budget_s"] == 900.0
    assert recovery["cumulative_s"] < 1.0
    assert recovery["estimated_reserved_s"] == 0.0
    assert budget.remaining() > 899.0


def test_effective_training_rows_keep_frozen_selection_and_report_no_support() -> None:
    rows, effective, missing = _effective_training_rows(
        [{"window_id": "w1", "source_id": "S1"}, {"window_id": "outside", "source_id": "S3"}],
        ["S1", "S2"],
    )
    assert [row["window_id"] for row in rows] == ["w1"]
    assert effective == {"S1"}
    assert missing == ["S2"]


def test_feature_cache_is_reused_only_for_current_complete_plan(tmp_path: Path) -> None:
    window = {
        "window_id": "S1::MANIP50::b00::real", "source_id": "S1", "role": "real",
        "parent_id": "S1::real::MANIP50", "offset_s": 0.0, "frame_indices": [0],
        "timestamps_s": [0.0], "label": 0, "annotation_category": "REAL", "video_path": "real.mp4",
    }
    atomic_json(tmp_path / "manifests/subwindows.json", [window])
    atomic_json(tmp_path / "frontend/results.json", [{**window, "status": "FRONTEND_COMPLETE", "o_sequence_prefix": "O", "r_sequence_prefix": "R", "o_frame_count": 1, "r_frame_count": 1}])
    atomic_json(tmp_path / "support/window_support.json", [
        {**window, "mode": mode, "support_status": "NO_VALID_SUPPORT", "particle_prefix": prefix, "features": {"SET_A": None}}
        for mode, prefix in (("O", "O"), ("R", "R"))
    ])
    atomic_json(tmp_path / "support/support_summary.json", {"subwindow_count": 1, "mode_rows": 2})

    assert runner._feature_artifacts_complete(tmp_path)
    atomic_json(tmp_path / "frontend/results.json", [{**window, "status": "FRONTEND_FAILED", "o_sequence_prefix": "O", "r_sequence_prefix": "R", "o_frame_count": 1, "r_frame_count": 1}])
    assert not runner._feature_artifacts_complete(tmp_path)


def test_feature_cache_identity_accepts_parent_sequence_covering_window(tmp_path: Path) -> None:
    window = {
        "window_id": "S1::MANIP50::b00", "source_id": "S1", "role": "real",
        "parent_id": "S1::real::MANIP50", "pair_id": "P1", "kind": "MANIP",
        "offset_s": 0.0, "interval_start_s": 0.0, "interval_end_s": 0.2,
        "frame_indices": [1, 2, 3], "timestamps_s": [0.0, 0.1, 0.2],
        "label": 0, "annotation_category": "REAL_NEGATIVE", "video_path": "real.mp4",
    }
    atomic_json(tmp_path / "manifests/subwindows.json", [window])
    frontend = {
        **window, "status": "FRONTEND_COMPLETE", "o_sequence_prefix": "O", "r_sequence_prefix": "R",
        "o_frame_count": 5, "r_frame_count": 3,
    }
    atomic_json(tmp_path / "frontend/results.json", [frontend])
    support_rows = [
        {**{key: value for key, value in window.items() if key not in {"video_path", "frame_indices", "timestamps_s"}},
         "mode": mode, "support_status": "VALID", "valid_unit_count": 1,
         "particle_prefix": prefix, "frame_indices": frames, "timestamps_s": times,
         "features": {"SET_A": [[1.0, 2.0]]}}
        for mode, prefix, frames, times in (
            ("O", "O", [0, 1, 2, 3, 4], [-0.1, 0.0, 0.1, 0.2, 0.3]),
            ("R", "R", [1, 2, 3], [0.0, 0.1, 0.2]),
        )
    ]
    atomic_json(tmp_path / "support/window_support.json", support_rows)
    atomic_json(tmp_path / "support/support_summary.json", {"subwindow_count": 1, "mode_rows": 2})

    assert runner._feature_artifacts_complete(tmp_path)

    support_rows[0]["timestamps_s"][1] = 9.0
    atomic_json(tmp_path / "support/window_support.json", support_rows)
    assert not runner._feature_artifacts_complete(tmp_path)


def test_feature_runtime_is_charged_to_resumed_combined_budget(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(runner, "features", lambda _root: {"mode_rows": 2})
    monkeypatch.setattr(runner, "_feature_artifacts_complete", lambda _root: True)
    budget = runner.Budget(tmp_path, "training", 10.0)
    result = runner._features_with_budget(tmp_path, budget)
    saved = json.loads((tmp_path / "state/training_budget.json").read_text(encoding="utf-8"))
    resumed = runner.Budget(tmp_path, "training", 10.0)
    assert result["status"] == "COMPLETE"
    assert saved["cumulative_s"] >= result["elapsed_s"]
    assert resumed.before == pytest.approx(saved["cumulative_s"])
    assert resumed.remaining() <= 10.0 - saved["cumulative_s"] + 1e-6


@pytest.mark.parametrize("budget_name", ["source128", "periodic"])
def test_interrupted_budget_resume_charges_from_fixed_process_base_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    budget_name: str,
) -> None:
    from research_tools.v7.periodic_requery_probe.runner import Budget as PeriodicBudget

    clock = {"wall": 1000.0, "monotonic": 100.0}
    monkeypatch.setattr(runner.time, "time", lambda: clock["wall"])
    monkeypatch.setattr(runner.time, "monotonic", lambda: clock["monotonic"])
    budget_type = runner.Budget if budget_name == "source128" else PeriodicBudget

    first = budget_type(tmp_path, "frontend", 100.0)
    assert json.loads((tmp_path / "state/frontend_budget.json").read_text())["process_state"] == "RUNNING"
    clock["monotonic"] = 108.0
    first.save(None)
    checkpoint = json.loads((tmp_path / "state/frontend_budget.json").read_text())
    assert checkpoint["cumulative_s"] == pytest.approx(8.0)

    # Simulate an unrecorded process that started from elapsed_before=0 and
    # was found 30 wall seconds later.  The 8 s checkpoint is measured; only
    # the remaining 22 s are conservatively reserved, not counted twice.
    clock.update(wall=1030.0, monotonic=5.0)
    resumed = budget_type(tmp_path, "frontend", 100.0)
    saved = json.loads((tmp_path / "state/frontend_budget.json").read_text())
    assert resumed.before == pytest.approx(8.0)
    assert saved["cumulative_s"] == pytest.approx(8.0)
    assert saved["estimated_reserved_s"] == pytest.approx(22.0)
    assert saved["budget_accounted_upper_s"] == pytest.approx(30.0)
    assert saved["recovered_interrupted_process"]["charged_process_elapsed_s"] == pytest.approx(30.0)
    assert resumed.remaining() == pytest.approx(70.0)


def test_supervisor_exit_records_code_and_only_marks_its_own_running_attempt(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(runner.time, "time", lambda: 120.0)
    atomic_json(tmp_path / "state/launch.json", {"pid": 456, "started_unix": 101.0})
    atomic_json(tmp_path / "final_status.json", {"status": "RUNNING", "pid": 456})
    atomic_json(tmp_path / "state/frontend_budget.json", {
        "budget_s": 100.0,
        "elapsed_before_this_process_s": 10.0,
        "process_start_unix": 100.0,
        "process_elapsed_s": 8.0,
        "cumulative_s": 18.0,
        "estimated_reserved_s": 2.0,
        "process_state": "RUNNING",
    })

    record = runner.record_supervisor_exit(
        tmp_path,
        runner_pid=456,
        wrapper_pid=123,
        exit_code=143,
        started_unix=100.0,
        log_path="/tmp/source128.log",
        screen_session="v7-source128-extension",
        command="python -u -m research_tools.v7.source128_extension.runner all",
    )

    saved_exit = json.loads((tmp_path / "state/supervisor_exit.json").read_text())
    saved_final = json.loads((tmp_path / "final_status.json").read_text())
    saved_budget = json.loads((tmp_path / "state/frontend_budget.json").read_text())
    assert saved_exit["exit_code"] == record["exit_code"] == 143
    assert saved_exit["elapsed_wall_s"] == pytest.approx(20.0)
    assert saved_exit["log_path"] == "/tmp/source128.log"
    assert saved_final["status"] == "INTERRUPTED"
    assert saved_final["wrapper_exit_code"] == 143
    assert saved_budget["cumulative_s"] == pytest.approx(18.0)
    assert saved_budget["estimated_reserved_s"] == pytest.approx(14.0)
    assert saved_budget["budget_accounted_upper_s"] == pytest.approx(32.0)
    assert saved_budget["process_state"] == "INTERRUPTED"

    atomic_json(tmp_path / "state/launch.json", {"pid": 789, "started_unix": 101.0})
    atomic_json(tmp_path / "final_status.json", {"status": "RUNNING", "pid": 789})
    runner.record_supervisor_exit(
        tmp_path,
        runner_pid=456,
        wrapper_pid=123,
        exit_code=1,
        started_unix=100.0,
        log_path="/tmp/old-wrapper.log",
        screen_session="v7-source128-extension",
        command="another run",
    )
    assert json.loads((tmp_path / "final_status.json").read_text())["pid"] == 789


def _report_fixture(tmp_path: Path, real_status: str, fake_status: str) -> None:
    atomic_json(tmp_path / "protocol.json", {"selection": {"base_sources": ["S1"], "added_sources": [], "validation_sources": []}})
    atomic_json(tmp_path / "acquisition/media_manifest.json", {"results": [
        {"source_id": "S1", "role": "real", "status": real_status},
        {"source_id": "S1", "role": "fake", "status": fake_status},
    ]})
    atomic_json(tmp_path / "final_status.json", {"status": "MEDIA_INCOMPLETE"})


def test_report_clears_stale_media_incomplete_when_media_is_complete(tmp_path: Path) -> None:
    _report_fixture(tmp_path, "REUSED_EXISTING", "DOWNLOADED")

    result = report(tmp_path)

    final = json.loads((tmp_path / "final_status.json").read_text(encoding="utf-8"))
    text = (tmp_path / "report.md").read_text(encoding="utf-8")
    assert result["status"] == "PLANNED"
    assert final["status"] == "PLANNED"
    assert final["media_download_complete"] is True
    assert final["media_manifest_count"] == final["expected_media_count"] == 2
    assert final["download_error_resolved"] is False
    assert "下载状态：COMPLETE（2/2 项）" in text


def test_report_preserves_media_incomplete_when_manifest_is_still_incomplete(tmp_path: Path) -> None:
    _report_fixture(tmp_path, "REUSED_EXISTING", "DOWNLOAD_FAILURE")

    result = report(tmp_path)

    final = json.loads((tmp_path / "final_status.json").read_text(encoding="utf-8"))
    assert result["status"] == "MEDIA_INCOMPLETE"
    assert final["status"] == "MEDIA_INCOMPLETE"
    assert final["media_download_complete"] is False
    assert final["download_error_resolved"] is False
