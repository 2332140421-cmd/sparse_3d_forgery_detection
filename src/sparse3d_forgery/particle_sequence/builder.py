"""Construction boundary from frontend observations to ParticleSequence."""

from collections.abc import Mapping

import numpy as np

from sparse3d_forgery.video_input import DecodedVideoSample

from .schema import (
    PARTICLE_SEQUENCE_SCHEMA_VERSION,
    CoordinateSystem,
    ParticleSequence,
)
from .validation import validate_particle_sequence


def build_particle_sequence(
    decoded: DecodedVideoSample,
    *,
    track_ids: np.ndarray,
    xyz: np.ndarray,
    uv: np.ndarray,
    visibility: np.ndarray,
    geometry_validity: np.ndarray,
    coordinate_system: CoordinateSystem,
    lineage: Mapping[str, object],
    provenance: Mapping[str, object],
) -> ParticleSequence:
    """Build and strictly validate one canonical sequence without repairing inputs."""

    sequence = ParticleSequence(
        schema_version=PARTICLE_SEQUENCE_SCHEMA_VERSION,
        sample_id=decoded.sample_id,
        source_video_id=decoded.source_video_id,
        frame_indices=decoded.frame_indices,
        timestamps_s=decoded.timestamps_s,
        frame_sizes_hw=decoded.frame_sizes_hw,
        track_ids=track_ids,
        xyz=xyz,
        uv=uv,
        visibility=visibility,
        geometry_validity=geometry_validity,
        coordinate_system=coordinate_system,
        lineage=lineage,
        provenance=provenance,
    )
    validate_particle_sequence(sequence)
    return sequence
