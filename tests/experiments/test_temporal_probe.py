from dataclasses import replace

import numpy as np
import pytest
import torch

from sparse3d_forgery.experiments.temporal_probe import (
    HORIZONS,
    MissingAwareGRU,
    assert_disjoint_source_ids,
    build_targets,
    history_normalize,
    linear_extrapolation,
    rank_auroc,
    zero_displacement,
)
from sparse3d_forgery.particle_sequence import (
    PARTICLE_SEQUENCE_SCHEMA_VERSION,
    CoordinateSystem,
    Handedness,
    LengthUnit,
    ParticleSequence,
)


def _sequence() -> ParticleSequence:
    frames, particles = 16, 3
    timestamps = np.arange(frames, dtype=np.float64) * 0.1
    xyz = np.empty((frames, particles, 3), dtype=np.float32)
    for time in range(frames):
        xyz[time] = np.array(
            [[time, 0, 1], [0, 2 * time, 2], [time, time, 3]], dtype=np.float32
        )
    validity = np.ones((frames, particles), dtype=np.bool_)
    validity[2, 1] = False
    xyz[~validity] = np.nan
    uv = np.full((frames, particles, 2), 10.0, dtype=np.float32)
    uv[~validity] = np.nan
    return ParticleSequence(
        schema_version=PARTICLE_SEQUENCE_SCHEMA_VERSION,
        sample_id="probe-test",
        source_video_id="source-test",
        frame_indices=np.arange(frames, dtype=np.int64),
        timestamps_s=timestamps,
        frame_sizes_hw=np.full((frames, 2), 64, dtype=np.int64),
        track_ids=np.arange(particles, dtype=np.int64),
        xyz=xyz,
        uv=uv,
        visibility=validity.copy(),
        geometry_validity=validity,
        coordinate_system=CoordinateSystem(
            frame_name="test",
            handedness=Handedness.RIGHT,
            axis_directions=("right", "down", "forward"),
            length_unit=LengthUnit.ARBITRARY_SCALE,
            camera_motion_compensated=True,
            normalization={"applied": False},
        ),
        lineage={"source": "test"},
        provenance={"causal_training_eligible": True},
    )


def test_history_normalization_ignores_future_and_standardizes_history():
    sequence = _sequence()
    changed = sequence.xyz.copy()
    changed[8:] += 10000
    first = history_normalize(sequence, 8)
    second = history_normalize(replace(sequence, xyz=changed), 8)

    np.testing.assert_equal(first.centroid, second.centroid)
    assert first.rms_radius == second.rms_radius
    history = first.xyz[:8][first.validity[:8]]
    np.testing.assert_allclose(history.mean(axis=0), 0, atol=1e-6)
    assert np.sqrt(np.mean(np.sum(history**2, axis=1))) == pytest.approx(1.0, abs=1e-6)


def test_normalization_preserves_invalid_nan():
    normalized = history_normalize(_sequence(), 8)
    assert np.isnan(normalized.xyz[~normalized.validity]).all()


def test_missing_aware_gru_does_not_update_invalid_particle():
    torch.manual_seed(4)
    model = MissingAwareGRU(hidden_dim=5)
    xyz = torch.randn(1, 3, 2, 3)
    validity = torch.tensor([[[True, True], [True, False], [True, True]]])
    changed = xyz.clone()
    changed[0, 1, 1] = torch.tensor([float("nan")] * 3)

    first = model.encode(xyz, validity)
    second = model.encode(changed, validity)

    torch.testing.assert_close(first, second)


def test_targets_require_cutoff_and_target_validity_and_map_horizons():
    normalized = history_normalize(_sequence(), 8)
    targets, mask = build_targets(normalized.xyz, normalized.validity, 8)

    assert targets.shape == (3, 4, 3)
    assert mask.shape == (3, 4)
    cutoff = 7
    for offset, horizon in enumerate(HORIZONS):
        np.testing.assert_allclose(
            targets[mask[:, offset], offset],
            normalized.xyz[cutoff + horizon, mask[:, offset]]
            - normalized.xyz[cutoff, mask[:, offset]],
        )
    invalid = normalized.validity.copy()
    invalid[cutoff, 0] = False
    invalid[cutoff + 2, 1] = False
    _, changed_mask = build_targets(normalized.xyz, invalid, 8)
    assert not changed_mask[0].any()
    assert changed_mask[1].tolist() == [True, False, True, True]


def test_gru_direct_multi_horizon_shape():
    model = MissingAwareGRU(hidden_dim=5)
    output = model(torch.randn(2, 8, 3, 3), torch.ones(2, 8, 3, dtype=torch.bool))
    assert output.shape == (2, 3, len(HORIZONS), 3)


def test_zero_and_linear_diagnostic_baselines():
    normalized = history_normalize(_sequence(), 8)
    assert np.count_nonzero(zero_displacement(3)) == 0
    prediction, coverage = linear_extrapolation(
        normalized.xyz, normalized.validity, _sequence().timestamps_s, 8
    )
    targets, target_validity = build_targets(normalized.xyz, normalized.validity, 8)
    np.testing.assert_allclose(prediction[coverage], targets[coverage], atol=1e-5)
    assert coverage.shape == target_validity.shape


def test_linear_baseline_reports_missing_history_coverage():
    normalized = history_normalize(_sequence(), 8)
    validity = normalized.validity.copy()
    validity[:7, 0] = False
    _, coverage = linear_extrapolation(
        normalized.xyz, validity, _sequence().timestamps_s, 8
    )
    assert not coverage[0].any()


def test_rank_auroc_perfect_reversed_and_tied():
    assert rank_auroc(np.array([0, 1]), np.array([2, 3])) == 1.0
    assert rank_auroc(np.array([2, 3]), np.array([0, 1])) == 0.0
    assert rank_auroc(np.ones(3), np.ones(4)) == 0.5


def test_train_validation_source_ids_must_be_disjoint():
    assert_disjoint_source_ids(["a", "b"], ["c"])
    with pytest.raises(ValueError, match="disjoint"):
        assert_disjoint_source_ids(["a", "b"], ["b", "c"])
