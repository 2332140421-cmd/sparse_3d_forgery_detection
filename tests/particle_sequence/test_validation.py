from dataclasses import replace

import numpy as np
import pytest

from sparse3d_forgery.particle_sequence import (
    ParticleSequenceValidationError,
    validate_particle_sequence,
)


def issue_paths(sequence):
    with pytest.raises(ParticleSequenceValidationError) as caught:
        validate_particle_sequence(sequence)
    return [issue.path for issue in caught.value.issues]


def test_valid_sequence_and_origin(valid_sequence):
    assert validate_particle_sequence(valid_sequence) is None
    np.testing.assert_array_equal(valid_sequence.xyz[0, 0], [0.0, 0.0, 0.0])


def test_requires_particle_sequence():
    with pytest.raises(ParticleSequenceValidationError) as caught:
        validate_particle_sequence(object())
    assert caught.value.issues[0].path == "sequence"


@pytest.mark.parametrize(
    ("changes", "path"),
    [
        ({"schema_version": "2.0.0"}, "schema_version"),
        ({"sample_id": ""}, "sample_id"),
        ({"source_video_id": "  "}, "source_video_id"),
    ],
)
def test_version_and_identity(changed, changes, path):
    assert path in issue_paths(changed(**changes))


def test_rejects_non_ndarray(changed):
    assert "frame_indices" in issue_paths(changed(frame_indices=[0, 1, 2]))


@pytest.mark.parametrize(
    ("field", "dtype"),
    [
        ("frame_indices", np.int32),
        ("timestamps_s", np.float32),
        ("frame_sizes_hw", np.int32),
        ("track_ids", np.int32),
        ("xyz", np.float64),
        ("uv", np.float64),
        ("visibility", np.uint8),
        ("geometry_validity", np.uint8),
    ],
)
def test_rejects_every_wrong_dtype(valid_sequence, changed, field, dtype):
    value = getattr(valid_sequence, field).astype(dtype)
    assert field in issue_paths(changed(**{field: value}))


@pytest.mark.parametrize(
    "field",
    ["frame_indices", "timestamps_s", "frame_sizes_hw", "uv", "xyz"],
)
@pytest.mark.parametrize("dtype", [str, object])
def test_non_numeric_dtype_is_reported_without_mutation(
    valid_sequence, changed, field, dtype
):
    value = getattr(valid_sequence, field).astype(dtype)
    before = value.copy()

    assert field in issue_paths(changed(**{field: value}))
    assert value.dtype == before.dtype
    assert value.shape == before.shape
    np.testing.assert_array_equal(value.astype(str), before.astype(str))


@pytest.mark.parametrize("field", ["visibility", "geometry_validity"])
@pytest.mark.parametrize("dtype", [str, object])
def test_mask_dtype_is_reported_without_mutation(valid_sequence, changed, field, dtype):
    value = getattr(valid_sequence, field).astype(dtype)
    before = value.copy()

    assert field in issue_paths(changed(**{field: value}))
    assert value.dtype == before.dtype
    assert value.shape == before.shape
    np.testing.assert_array_equal(value.astype(str), before.astype(str))


@pytest.mark.parametrize(
    ("field", "shape"),
    [
        ("frame_indices", (3, 1)),
        ("timestamps_s", (3, 1)),
        ("frame_sizes_hw", (3, 1)),
        ("track_ids", (2, 1)),
        ("xyz", (3, 2, 3, 1)),
        ("uv", (3, 2, 1)),
        ("visibility", (3, 2, 1)),
        ("geometry_validity", (3, 2, 1)),
    ],
)
def test_rejects_every_wrong_shape(valid_sequence, changed, field, shape):
    value = np.resize(getattr(valid_sequence, field), shape)
    assert field in issue_paths(changed(**{field: value}))


def test_rejects_empty_time_axis(changed):
    sequence = changed(
        frame_indices=np.empty((0,), np.int64),
        timestamps_s=np.empty((0,), np.float64),
        frame_sizes_hw=np.empty((0, 2), np.int64),
        xyz=np.empty((0, 2, 3), np.float32),
        uv=np.empty((0, 2, 2), np.float32),
        visibility=np.empty((0, 2), np.bool_),
        geometry_validity=np.empty((0, 2), np.bool_),
    )
    assert "frame_indices" in issue_paths(sequence)


def test_rejects_empty_track_axis(changed):
    sequence = changed(
        track_ids=np.empty((0,), np.int64),
        xyz=np.empty((3, 0, 3), np.float32),
        uv=np.empty((3, 0, 2), np.float32),
        visibility=np.empty((3, 0), np.bool_),
        geometry_validity=np.empty((3, 0), np.bool_),
    )
    assert "track_ids" in issue_paths(sequence)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("frame_indices", np.array([0, 0, 2], np.int64)),
        ("timestamps_s", np.array([0.0, np.inf, 2.0], np.float64)),
        ("timestamps_s", np.array([0.0, 0.0, 2.0], np.float64)),
        ("frame_sizes_hw", np.array([[64, 80], [0, 80], [64, 80]], np.int64)),
        ("track_ids", np.array([7, 7], np.int64)),
    ],
)
def test_time_identity_and_size_rules(changed, field, value):
    assert field in issue_paths(changed(**{field: value}))


def test_visible_uv_must_be_finite(valid_sequence, changed):
    uv = valid_sequence.uv.copy()
    uv[0, 0] = [np.nan, 1.0]
    assert "uv" in issue_paths(changed(uv=uv))


def test_invisible_uv_must_be_all_nan(valid_sequence, changed):
    uv = valid_sequence.uv.copy()
    uv[0, 1] = [1.0, np.nan]
    assert "uv" in issue_paths(changed(uv=uv))


@pytest.mark.parametrize("uv_value", [[80.0, 1.0], [1.0, 64.0], [-1.0, 1.0]])
def test_visible_uv_must_be_in_bounds(valid_sequence, changed, uv_value):
    uv = valid_sequence.uv.copy()
    uv[0, 0] = uv_value
    assert "uv" in issue_paths(changed(uv=uv))


def test_geometry_validity_requires_visibility(valid_sequence, changed):
    geometry = valid_sequence.geometry_validity.copy()
    geometry[0, 1] = True
    assert "geometry_validity" in issue_paths(changed(geometry_validity=geometry))


def test_geometry_valid_xyz_must_be_finite(valid_sequence, changed):
    xyz = valid_sequence.xyz.copy()
    xyz[0, 0, 0] = np.nan
    assert "xyz" in issue_paths(changed(xyz=xyz))


def test_geometry_invalid_xyz_must_be_all_nan(valid_sequence, changed):
    xyz = valid_sequence.xyz.copy()
    xyz[0, 1] = [0.0, np.nan, np.nan]
    assert "xyz" in issue_paths(changed(xyz=xyz))


def test_camera_motion_compensation_is_required(valid_sequence, changed):
    coordinate = replace(valid_sequence.coordinate_system, camera_motion_compensated=False)
    path = "coordinate_system.camera_motion_compensated"
    assert path in issue_paths(changed(coordinate_system=coordinate))


@pytest.mark.parametrize(
    ("changes", "path"),
    [
        ({"handedness": "right"}, "coordinate_system.handedness"),
        ({"length_unit": "meter"}, "coordinate_system.length_unit"),
        ({"axis_directions": ("right", "")}, "coordinate_system.axis_directions"),
        ({"frame_name": ""}, "coordinate_system.frame_name"),
    ],
)
def test_coordinate_system_rules(valid_sequence, changed, changes, path):
    coordinate = replace(valid_sequence.coordinate_system, **changes)
    assert path in issue_paths(changed(coordinate_system=coordinate))


@pytest.mark.parametrize(
    ("field", "value", "path"),
    [
        ("lineage", {"bad": np.nan}, "lineage"),
        ("provenance", {"bad": object()}, "provenance"),
    ],
)
def test_metadata_must_be_strict_json(changed, field, value, path):
    assert path in issue_paths(changed(**{field: value}))


def test_normalization_mapping_and_json(valid_sequence, changed):
    coordinate = replace(valid_sequence.coordinate_system, normalization={"bad": np.nan})
    assert "coordinate_system.normalization" in issue_paths(changed(coordinate_system=coordinate))
    coordinate = replace(valid_sequence.coordinate_system, normalization=[])
    assert "coordinate_system.normalization" in issue_paths(changed(coordinate_system=coordinate))


def test_collects_multiple_issues(changed):
    sequence = changed(schema_version="bad", sample_id="", source_video_id="")
    with pytest.raises(ParticleSequenceValidationError) as caught:
        validate_particle_sequence(sequence)
    assert [issue.path for issue in caught.value.issues[:3]] == [
        "schema_version", "sample_id", "source_video_id",
    ]


def test_compound_dtype_shape_and_metadata_issues_are_aggregated(
    valid_sequence, changed
):
    sequence = changed(
        timestamps_s=valid_sequence.timestamps_s.astype(str),
        frame_sizes_hw=valid_sequence.frame_sizes_hw[:, :1],
        lineage={"bad": np.nan},
    )

    assert issue_paths(sequence) == ["timestamps_s", "frame_sizes_hw", "lineage"]


def test_array_dtype_error_does_not_stop_independent_checks(valid_sequence, changed):
    sequence = changed(
        sample_id="",
        timestamps_s=valid_sequence.timestamps_s.astype(object),
        coordinate_system=object(),
        provenance={"bad": np.nan},
    )

    assert issue_paths(sequence) == [
        "sample_id",
        "timestamps_s",
        "coordinate_system",
        "provenance",
    ]


def test_validator_does_not_modify_arrays(valid_sequence):
    names = (
        "frame_indices", "timestamps_s", "frame_sizes_hw", "track_ids",
        "xyz", "uv", "visibility", "geometry_validity",
    )
    before = {name: getattr(valid_sequence, name).copy() for name in names}
    validate_particle_sequence(valid_sequence)
    for name, expected in before.items():
        actual = getattr(valid_sequence, name)
        assert np.array_equal(actual, expected, equal_nan=True)
