from dataclasses import replace

import numpy as np
import pytest

from sparse3d_forgery.frontend import (
    SimilarityAlignmentError,
    fit_similarity_transform,
)
from sparse3d_forgery.frontend.vggt import _construct_history_anchored_window
from sparse3d_forgery.particle_sequence import validate_particle_sequence
from sparse3d_forgery.particle_sequence import (
    PARTICLE_SEQUENCE_SCHEMA_VERSION,
    CoordinateSystem,
    Handedness,
    LengthUnit,
    ParticleSequence,
    load_particle_sequence,
    save_particle_sequence,
)


def _proper_rotation() -> np.ndarray:
    angle = 0.47
    return np.array(
        [
            [np.cos(angle), -np.sin(angle), 0.0],
            [np.sin(angle), np.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def _valid_sequence() -> ParticleSequence:
    visibility = np.array([[True, False], [True, True], [False, True]], dtype=np.bool_)
    uv = np.array(
        [
            [[10.0, 20.0], [np.nan, np.nan]],
            [[11.0, 21.0], [30.0, 40.0]],
            [[np.nan, np.nan], [31.0, 41.0]],
        ],
        dtype=np.float32,
    )
    xyz = np.array(
        [
            [[0.0, 0.0, 0.0], [np.nan, np.nan, np.nan]],
            [[1.0, 0.0, 2.0], [2.0, 1.0, 3.0]],
            [[np.nan, np.nan, np.nan], [3.0, 1.0, 4.0]],
        ],
        dtype=np.float32,
    )
    return ParticleSequence(
        schema_version=PARTICLE_SEQUENCE_SCHEMA_VERSION,
        sample_id="alignment-test",
        source_video_id="video-test",
        frame_indices=np.array([0, 1, 2], dtype=np.int64),
        timestamps_s=np.array([0.0, 0.1, 0.2], dtype=np.float64),
        frame_sizes_hw=np.array([[64, 80]] * 3, dtype=np.int64),
        track_ids=np.array([0, 1], dtype=np.int64),
        xyz=xyz,
        uv=uv,
        visibility=visibility,
        geometry_validity=visibility.copy(),
        coordinate_system=CoordinateSystem(
            frame_name="test-gauge",
            handedness=Handedness.RIGHT,
            axis_directions=("right", "down", "forward"),
            length_unit=LengthUnit.ARBITRARY_SCALE,
            camera_motion_compensated=True,
            normalization={"applied": False},
        ),
        lineage={"source": "unit-test"},
        provenance={"elapsed_s": 1.0, "peak_gpu_memory_bytes": 10},
    )


def test_umeyama_recovers_known_proper_similarity_and_rejects_degeneracy():
    source = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.2], [0.0, 2.0, 0.5], [1.0, 1.0, 2.0]]
    )
    scale = 2.3
    rotation = _proper_rotation()
    translation = np.array([4.0, -3.0, 1.5])
    destination = scale * np.einsum("ij,kj->ki", rotation, source) + translation

    fitted = fit_similarity_transform(source, destination)

    assert fitted.scale == pytest.approx(scale)
    np.testing.assert_allclose(fitted.rotation, rotation, atol=1e-12)
    np.testing.assert_allclose(fitted.translation, translation, atol=1e-12)
    np.testing.assert_allclose(fitted.apply(source), destination, atol=1e-6)
    assert np.linalg.det(fitted.rotation) == pytest.approx(1.0)
    with pytest.raises(SimilarityAlignmentError, match="collinear or coincident"):
        fit_similarity_transform(np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]]), source[:3])


def test_dynamic_same_frame_same_track_correspondences_recover_one_gauge_transform():
    # Rows represent distinct (historical frame, track) correspondences. Particles
    # move differently between frames; no cross-time static-point assumption is used.
    source = np.array(
        [
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
            [0.4, 0.7, 1.4],
            [1.8, -0.2, 0.9],
            [-0.3, 1.5, 2.2],
            [2.2, 0.8, 1.7],
        ]
    )
    rotation = _proper_rotation()
    destination = 0.7 * np.einsum("ij,kj->ki", rotation, source) + [0.2, 1.1, -0.4]

    fitted = fit_similarity_transform(source, destination)

    np.testing.assert_allclose(fitted.apply(source), destination, atol=1e-6)
    assert fitted.diagnostics.correspondence_count == 6


def _history_and_extension(valid_sequence):
    history = replace(
        valid_sequence,
        frame_indices=valid_sequence.frame_indices[:2].copy(),
        timestamps_s=valid_sequence.timestamps_s[:2].copy(),
        frame_sizes_hw=valid_sequence.frame_sizes_hw[:2].copy(),
        xyz=valid_sequence.xyz[:2].copy(),
        uv=valid_sequence.uv[:2].copy(),
        visibility=valid_sequence.visibility[:2].copy(),
        geometry_validity=valid_sequence.geometry_validity[:2].copy(),
    )
    rotation = _proper_rotation()
    scale = 1.6
    translation = np.array([0.8, -0.5, 0.3])
    extension_xyz = valid_sequence.xyz.copy()
    valid = valid_sequence.geometry_validity
    extension_xyz[valid] = np.einsum(
        "ij,kj->ki",
        rotation.T,
        (valid_sequence.xyz[valid] - translation) / scale,
    )
    extension = replace(valid_sequence, xyz=extension_xyz)
    return history, extension


def test_causal_window_preserves_history_and_future_never_changes_alignment(tmp_path):
    valid_sequence = _valid_sequence()
    history, extension = _history_and_extension(valid_sequence)

    result = _construct_history_anchored_window(history, extension, 2)
    changed_future_xyz = extension.xyz.copy()
    changed_future_xyz[2, 1] += np.array([1000.0, -700.0, 400.0], dtype=np.float32)
    changed = _construct_history_anchored_window(
        history, replace(extension, xyz=changed_future_xyz), 2
    )

    np.testing.assert_equal(result.xyz[:2], history.xyz)
    np.testing.assert_equal(result.uv[:2], history.uv)
    np.testing.assert_equal(result.visibility[:2], history.visibility)
    np.testing.assert_equal(result.geometry_validity[:2], history.geometry_validity)
    assert result.provenance["estimated_scale"] == changed.provenance["estimated_scale"]
    assert result.provenance["rotation"] == changed.provenance["rotation"]
    assert result.provenance["translation"] == changed.provenance["translation"]
    assert result.provenance["causal_training_eligible"] is True
    assert result.provenance["common_historical_correspondence_count"] == 3
    assert result.lineage["future_points_used_for_alignment"] is False
    validate_particle_sequence(result)
    save_particle_sequence(result, tmp_path / "causal-window")
    restored = load_particle_sequence(tmp_path / "causal-window")
    np.testing.assert_equal(restored.xyz, result.xyz)
    np.testing.assert_equal(restored.geometry_validity, result.geometry_validity)
    assert restored.coordinate_system == result.coordinate_system


def test_degenerate_history_has_no_identity_fallback_and_invalidates_future():
    valid_sequence = _valid_sequence()
    history, extension = _history_and_extension(valid_sequence)
    history_xyz = history.xyz.copy()
    extension_xyz = extension.xyz.copy()
    history_xyz[history.geometry_validity] = 1.0
    extension_xyz[:2][extension.geometry_validity[:2]] = 2.0

    result = _construct_history_anchored_window(
        replace(history, xyz=history_xyz), replace(extension, xyz=extension_xyz), 2
    )

    np.testing.assert_equal(result.xyz[:2], history_xyz)
    assert not result.geometry_validity[2:].any()
    assert np.isnan(result.xyz[2:]).all()
    assert result.provenance["causal_training_eligible"] is False
    assert result.provenance["alignment_failure"] is not None
    validate_particle_sequence(result)
