"""Targeted contracts for the matched two-density training pilot."""

from types import SimpleNamespace

import numpy as np
import pytest

from research_tools.v7.density_training_pilot import runner
from research_tools.v7.query_density_probe.pilot import nested_axes


def test_dense_queries_are_exact_midpoint_nested_grid() -> None:
    original, dense, _ = nested_axes(256)
    queries = runner._dense_queries()
    assert queries.shape == (1089, 2)
    assert queries.dtype == np.float32
    uu, vv = np.meshgrid(dense, dense)
    expected = np.stack((uu.ravel(), vv.ravel()), axis=1).astype(np.float32)
    assert np.array_equal(queries, expected)
    assert np.allclose(dense[::2], original)


def test_json_boundary_rejects_nonfinite_values() -> None:
    with pytest.raises(ValueError, match="non-finite JSON value"):
        runner._jsonable({"missing": float("nan")})


def test_fixed_threshold_metrics_keep_missing_class_as_none() -> None:
    result = runner._classification([0, 0], [-1.0, -0.2])
    assert result["tn"] == 2 and result["tp"] == 0
    assert result["precision"] is None
    assert result["recall"] is None
    assert result["f1"] is None


def test_sequence_resume_identity_uses_saved_sequence_frames(monkeypatch, tmp_path) -> None:
    fake = SimpleNamespace(
        num_tracks=289,
        frame_indices=np.asarray([10, 11, 12], dtype=np.int64),
        timestamps_s=np.asarray([0.0, 0.1, 0.2], dtype=np.float64),
        source_video_id="S::real",
        lineage={"window_id": "S::real::MANIP50::b00"},
        provenance={"query_count": 289, "query_cohort": "O"},
    )
    monkeypatch.setattr(runner, "load_particle_sequence", lambda _path: fake)
    row = {"source_id": "S", "role": "real", "window_id": "S::real::MANIP50::b00", "frame_indices": [10, 11, 12], "timestamps_s": [0.0, 0.1, 0.2]}
    assert runner._sequence_valid(tmp_path / "unused", row, "R17", 289)
    assert not runner._sequence_valid(tmp_path / "unused", row, "R17", 1089)
    assert not runner._sequence_valid(tmp_path / "unused", {**row, "frame_indices": [10, 13, 12]}, "R17", 289)


def test_cross_density_summary_keeps_four_cells_and_seed_mean() -> None:
    rows = []
    for train_density in ("R17", "R33"):
        for inference_density in ("R17", "R33"):
            for seed in (20260909, 20260910, 20260911):
                rows.extend({
                    "train_density": train_density,
                    "inference_density": inference_density,
                    "seed": seed,
                    "window_id": f"S::{'real' if label == 0 else 'fake'}::MANIP50::b{label}0",
                    "score": float(label + (seed % 3) * 0.01),
                } for label in (0, 1))
    summary = runner._cross_density_metrics(rows)
    assert len(summary) == 16
    assert sum(item["aggregate"] == "MEAN_LOGIT" for item in summary) == 4
    assert all(item["window_count"] == 2 for item in summary)
