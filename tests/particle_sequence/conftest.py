from dataclasses import replace

import numpy as np
import pytest

from sparse3d_forgery.particle_sequence import (
    PARTICLE_SEQUENCE_SCHEMA_VERSION,
    CoordinateSystem,
    Handedness,
    LengthUnit,
    ParticleSequence,
)


@pytest.fixture
def valid_sequence() -> ParticleSequence:
    visibility = np.array([[True, False], [True, True], [False, True]], dtype=np.bool_)
    geometry = visibility.copy()
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
        sample_id="clip-001",
        source_video_id="video-001",
        frame_indices=np.array([10, 12, 15], dtype=np.int64),
        timestamps_s=np.array([0.0, 0.08, 0.2], dtype=np.float64),
        frame_sizes_hw=np.array([[64, 80], [64, 80], [64, 80]], dtype=np.int64),
        track_ids=np.array([7, 42], dtype=np.int64),
        xyz=xyz,
        uv=uv,
        visibility=visibility,
        geometry_validity=geometry,
        coordinate_system=CoordinateSystem(
            frame_name="compensated_camera",
            handedness=Handedness.RIGHT,
            axis_directions=("right", "down", "forward"),
            length_unit=LengthUnit.ARBITRARY_SCALE,
            camera_motion_compensated=True,
            normalization={"applied": False},
        ),
        lineage={"source": "synthetic-unit-test"},
        provenance={"software_version": "test"},
    )


@pytest.fixture
def changed(valid_sequence):
    def factory(**changes):
        return replace(valid_sequence, **changes)

    return factory
