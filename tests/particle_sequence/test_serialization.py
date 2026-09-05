import json

import numpy as np

from sparse3d_forgery.particle_sequence import (
    load_particle_sequence,
    save_particle_sequence,
    validate_particle_sequence,
)


def test_npz_json_round_trip_preserves_contract(valid_sequence, tmp_path):
    prefix = tmp_path / "sequence"

    arrays_path, metadata_path = save_particle_sequence(valid_sequence, prefix)
    restored = load_particle_sequence(prefix)

    assert arrays_path == prefix.with_suffix(".npz")
    assert metadata_path == prefix.with_suffix(".json")
    assert restored.schema_version == valid_sequence.schema_version
    assert restored.sample_id == valid_sequence.sample_id
    assert restored.source_video_id == valid_sequence.source_video_id
    assert restored.coordinate_system == valid_sequence.coordinate_system
    assert restored.lineage == valid_sequence.lineage
    assert restored.provenance == valid_sequence.provenance
    for name in (
        "frame_indices",
        "timestamps_s",
        "frame_sizes_hw",
        "track_ids",
        "xyz",
        "uv",
        "visibility",
        "geometry_validity",
    ):
        actual = getattr(restored, name)
        expected = getattr(valid_sequence, name)
        assert actual.dtype == expected.dtype
        np.testing.assert_equal(actual, expected)
    validate_particle_sequence(restored)

    with np.load(arrays_path, allow_pickle=False) as arrays:
        assert all(array.dtype != object for array in arrays.values())
    with metadata_path.open(encoding="utf-8") as handle:
        assert json.load(handle)["coordinate_system"]["handedness"] == "right"
