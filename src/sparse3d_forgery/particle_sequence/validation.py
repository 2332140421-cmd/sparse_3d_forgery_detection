"""Strict, non-mutating validation for ParticleSequence."""

from collections.abc import Mapping
from dataclasses import dataclass
import json

import numpy as np

from .schema import (
    PARTICLE_SEQUENCE_SCHEMA_VERSION,
    CoordinateSystem,
    Handedness,
    LengthUnit,
    ParticleSequence,
)


@dataclass(frozen=True, slots=True)
class ValidationIssue:
    """A stable validation finding associated with a logical field path."""

    path: str
    message: str


class ParticleSequenceValidationError(ValueError):
    """Raised with every issue found during one validation pass."""

    def __init__(self, issues: tuple[ValidationIssue, ...]) -> None:
        self.issues = issues
        detail = "\n".join(f"- {issue.path}: {issue.message}" for issue in issues)
        super().__init__(f"ParticleSequence validation failed:\n{detail}")


_ARRAY_SPECS = (
    ("frame_indices", np.dtype(np.int64)),
    ("timestamps_s", np.dtype(np.float64)),
    ("frame_sizes_hw", np.dtype(np.int64)),
    ("track_ids", np.dtype(np.int64)),
    ("xyz", np.dtype(np.float32)),
    ("uv", np.dtype(np.float32)),
    ("visibility", np.dtype(np.bool_)),
    ("geometry_validity", np.dtype(np.bool_)),
)


def _add(issues: list[ValidationIssue], path: str, message: str) -> None:
    issues.append(ValidationIssue(path=path, message=message))


def _json_mapping(value: object, path: str, issues: list[ValidationIssue]) -> None:
    if not isinstance(value, Mapping):
        _add(issues, path, "must be a mapping")
        return
    try:
        json.dumps(value, allow_nan=False)
    except (TypeError, ValueError, OverflowError) as exc:
        _add(issues, path, f"must be strictly JSON-serializable: {exc}")


def validate_particle_sequence(sequence: ParticleSequence) -> None:
    """Validate every frozen logical invariant without changing the input."""

    issues: list[ValidationIssue] = []
    if not isinstance(sequence, ParticleSequence):
        raise ParticleSequenceValidationError(
            (ValidationIssue("sequence", "must be a ParticleSequence"),)
        )

    if sequence.schema_version != PARTICLE_SEQUENCE_SCHEMA_VERSION:
        _add(issues, "schema_version", "must equal '1.0.0'")
    if not isinstance(sequence.sample_id, str) or not sequence.sample_id.strip():
        _add(issues, "sample_id", "must be a non-empty string")
    if not isinstance(sequence.source_video_id, str) or not sequence.source_video_id.strip():
        _add(issues, "source_video_id", "must be a non-empty string")

    arrays: dict[str, np.ndarray] = {}
    dtype_valid: dict[str, bool] = {}
    for name, expected_dtype in _ARRAY_SPECS:
        value = getattr(sequence, name)
        if not isinstance(value, np.ndarray):
            _add(issues, name, "must be a numpy.ndarray")
            continue
        arrays[name] = value
        dtype_valid[name] = value.dtype == expected_dtype
        if not dtype_valid[name]:
            _add(issues, name, f"dtype must be {expected_dtype.name}")

    frame_indices = arrays.get("frame_indices")
    track_ids = arrays.get("track_ids")
    time_size = frame_indices.shape[0] if frame_indices is not None and frame_indices.ndim == 1 else None
    track_size = track_ids.shape[0] if track_ids is not None and track_ids.ndim == 1 else None
    expected_shapes = (
        ("frame_indices", (time_size,)),
        ("timestamps_s", (time_size,)),
        ("frame_sizes_hw", (time_size, 2)),
        ("track_ids", (track_size,)),
        ("xyz", (time_size, track_size, 3)),
        ("uv", (time_size, track_size, 2)),
        ("visibility", (time_size, track_size)),
        ("geometry_validity", (time_size, track_size)),
    )
    shape_valid: dict[str, bool] = {}
    for name, expected in expected_shapes:
        value = arrays.get(name)
        if value is None:
            continue
        if value.ndim != len(expected):
            _add(issues, name, f"must have rank {len(expected)}")
            shape_valid[name] = False
        elif any(size is not None and value.shape[i] != size for i, size in enumerate(expected)):
            _add(issues, name, f"shape must match {expected}")
            shape_valid[name] = False
        else:
            shape_valid[name] = True

    def is_safe(name: str) -> bool:
        return dtype_valid.get(name, False) and shape_valid.get(name, False)

    if time_size == 0:
        _add(issues, "frame_indices", "T must be greater than zero")
    if track_size == 0:
        _add(issues, "track_ids", "N must be greater than zero")

    if is_safe("frame_indices"):
        if frame_indices.size > 1 and not np.all(np.diff(frame_indices) > 0):
            _add(issues, "frame_indices", "must be strictly increasing")
    timestamps = arrays.get("timestamps_s")
    if is_safe("timestamps_s"):
        if not np.all(np.isfinite(timestamps)):
            _add(issues, "timestamps_s", "all values must be finite")
        if timestamps.size > 1 and not np.all(np.diff(timestamps) > 0):
            _add(issues, "timestamps_s", "must be strictly increasing")
    frame_sizes = arrays.get("frame_sizes_hw")
    if is_safe("frame_sizes_hw"):
        if not np.all(frame_sizes > 0):
            _add(issues, "frame_sizes_hw", "height and width must be positive")
    if is_safe("track_ids"):
        if np.unique(track_ids).size != track_ids.size:
            _add(issues, "track_ids", "values must be unique")

    uv = arrays.get("uv")
    visibility = arrays.get("visibility")
    uv_ok = (
        is_safe("uv") and is_safe("visibility")
        and uv.shape[:2] == visibility.shape
    )
    if uv_ok:
        visible = visibility
        if np.any(visible & ~np.all(np.isfinite(uv), axis=-1)):
            _add(issues, "uv", "visible observations must be finite")
        if np.any(~visible & ~np.all(np.isnan(uv), axis=-1)):
            _add(issues, "uv", "invisible observations must be all NaN")
        if is_safe("frame_sizes_hw") and frame_sizes.shape == (uv.shape[0], 2):
            heights = frame_sizes[:, 0, None]
            widths = frame_sizes[:, 1, None]
            u, v = uv[:, :, 0], uv[:, :, 1]
            in_bounds = (u >= 0) & (u < widths) & (v >= 0) & (v < heights)
            if np.any(visible & ~in_bounds):
                _add(issues, "uv", "visible observations must be in frame bounds")

    xyz = arrays.get("xyz")
    geometry = arrays.get("geometry_validity")
    if is_safe("geometry_validity") and is_safe("visibility") and geometry.shape == visibility.shape:
        if np.any(geometry & ~visibility):
            _add(issues, "geometry_validity", "true values require visibility to be true")
    xyz_ok = (
        is_safe("xyz") and is_safe("geometry_validity")
        and xyz.shape[:2] == geometry.shape
    )
    if xyz_ok:
        valid = geometry
        if np.any(valid & ~np.all(np.isfinite(xyz), axis=-1)):
            _add(issues, "xyz", "geometry-valid observations must be finite")
        if np.any(~valid & ~np.all(np.isnan(xyz), axis=-1)):
            _add(issues, "xyz", "geometry-invalid observations must be all NaN")

    coordinate = sequence.coordinate_system
    if not isinstance(coordinate, CoordinateSystem):
        _add(issues, "coordinate_system", "must be a CoordinateSystem")
    else:
        if not isinstance(coordinate.frame_name, str) or not coordinate.frame_name.strip():
            _add(issues, "coordinate_system.frame_name", "must be a non-empty string")
        if not isinstance(coordinate.handedness, Handedness):
            _add(issues, "coordinate_system.handedness", "must be a Handedness")
        axes = coordinate.axis_directions
        if not isinstance(axes, tuple) or len(axes) != 3 or any(
            not isinstance(axis, str) or not axis.strip() for axis in axes
        ):
            _add(issues, "coordinate_system.axis_directions", "must be exactly three non-empty strings")
        if not isinstance(coordinate.length_unit, LengthUnit):
            _add(issues, "coordinate_system.length_unit", "must be a LengthUnit")
        if coordinate.camera_motion_compensated is not True:
            _add(issues, "coordinate_system.camera_motion_compensated", "must be True")
        _json_mapping(coordinate.normalization, "coordinate_system.normalization", issues)

    _json_mapping(sequence.lineage, "lineage", issues)
    _json_mapping(sequence.provenance, "provenance", issues)
    if issues:
        raise ParticleSequenceValidationError(tuple(issues))
