"""In-memory logical schema for sparse 3D particle observations."""

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum

import numpy as np

PARTICLE_SEQUENCE_SCHEMA_VERSION = "1.0.0"


class Handedness(str, Enum):
    """Coordinate-system handedness."""

    LEFT = "left"
    RIGHT = "right"


class LengthUnit(str, Enum):
    """Supported logical length-unit declarations."""

    METER = "meter"
    ARBITRARY_SCALE = "arbitrary_scale"


@dataclass(frozen=True, slots=True)
class CoordinateSystem:
    """Coordinate semantics only; this object performs no transformation."""

    frame_name: str
    handedness: Handedness
    axis_directions: tuple[str, str, str]
    length_unit: LengthUnit
    camera_motion_compensated: bool
    normalization: Mapping[str, object]


@dataclass(frozen=True, slots=True)
class ParticleSequence:
    """One clip's canonical in-memory sparse 3D observations.

    Frozen fields cannot be rebound. Contained NumPy arrays and mapping values
    are not deeply immutable.
    """

    schema_version: str
    sample_id: str
    source_video_id: str
    frame_indices: np.ndarray
    timestamps_s: np.ndarray
    frame_sizes_hw: np.ndarray
    track_ids: np.ndarray
    xyz: np.ndarray
    uv: np.ndarray
    visibility: np.ndarray
    geometry_validity: np.ndarray
    coordinate_system: CoordinateSystem
    lineage: Mapping[str, object]
    provenance: Mapping[str, object]

    @property
    def num_frames(self) -> int:
        """Return the time-axis length without validating the sequence."""

        return len(self.frame_indices)

    @property
    def num_tracks(self) -> int:
        """Return the track-axis length without validating the sequence."""

        return len(self.track_ids)

    @property
    def observation_validity(self) -> np.ndarray:
        """Return visibility AND geometry_validity as a new array."""

        return np.logical_and(self.visibility, self.geometry_validity)
