"""NPZ arrays plus JSON metadata for current research artifacts."""

import json
import os
from pathlib import Path

import numpy as np

from .schema import CoordinateSystem, Handedness, LengthUnit, ParticleSequence
from .validation import validate_particle_sequence


def save_particle_sequence(sequence: ParticleSequence, prefix: str | Path) -> tuple[Path, Path]:
    """Atomically save validated arrays and JSON metadata without pickle."""

    validate_particle_sequence(sequence)
    prefix = Path(prefix)
    arrays_path = prefix.with_suffix(".npz")
    metadata_path = prefix.with_suffix(".json")
    arrays_tmp = arrays_path.with_name(f".{arrays_path.name}.tmp")
    metadata_tmp = metadata_path.with_name(f".{metadata_path.name}.tmp")
    prefix.parent.mkdir(parents=True, exist_ok=True)
    if arrays_path.exists() or metadata_path.exists() or arrays_tmp.exists() or metadata_tmp.exists():
        raise FileExistsError(f"ParticleSequence artifact already exists: {prefix}")

    with arrays_tmp.open("xb") as handle:
        np.savez_compressed(
            handle,
            frame_indices=sequence.frame_indices,
            timestamps_s=sequence.timestamps_s,
            frame_sizes_hw=sequence.frame_sizes_hw,
            track_ids=sequence.track_ids,
            xyz=sequence.xyz,
            uv=sequence.uv,
            visibility=sequence.visibility,
            geometry_validity=sequence.geometry_validity,
        )
        handle.flush()
        os.fsync(handle.fileno())
    metadata = {
        "schema_version": sequence.schema_version,
        "sample_id": sequence.sample_id,
        "source_video_id": sequence.source_video_id,
        "coordinate_system": {
            "frame_name": sequence.coordinate_system.frame_name,
            "handedness": sequence.coordinate_system.handedness.value,
            "axis_directions": list(sequence.coordinate_system.axis_directions),
            "length_unit": sequence.coordinate_system.length_unit.value,
            "camera_motion_compensated": sequence.coordinate_system.camera_motion_compensated,
            "normalization": sequence.coordinate_system.normalization,
        },
        "lineage": sequence.lineage,
        "provenance": sequence.provenance,
    }
    with metadata_tmp.open("x", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(arrays_tmp, arrays_path)
    os.replace(metadata_tmp, metadata_path)
    return arrays_path, metadata_path


def load_particle_sequence(prefix: str | Path) -> ParticleSequence:
    """Load one NPZ/JSON artifact and validate its exact logical contract."""

    prefix = Path(prefix)
    with prefix.with_suffix(".json").open(encoding="utf-8") as handle:
        metadata = json.load(handle)
    coordinate = metadata["coordinate_system"]
    with np.load(prefix.with_suffix(".npz"), allow_pickle=False) as arrays:
        sequence = ParticleSequence(
            schema_version=metadata["schema_version"],
            sample_id=metadata["sample_id"],
            source_video_id=metadata["source_video_id"],
            frame_indices=arrays["frame_indices"],
            timestamps_s=arrays["timestamps_s"],
            frame_sizes_hw=arrays["frame_sizes_hw"],
            track_ids=arrays["track_ids"],
            xyz=arrays["xyz"],
            uv=arrays["uv"],
            visibility=arrays["visibility"],
            geometry_validity=arrays["geometry_validity"],
            coordinate_system=CoordinateSystem(
                frame_name=coordinate["frame_name"],
                handedness=Handedness(coordinate["handedness"]),
                axis_directions=tuple(coordinate["axis_directions"]),
                length_unit=LengthUnit(coordinate["length_unit"]),
                camera_motion_compensated=coordinate["camera_motion_compensated"],
                normalization=coordinate["normalization"],
            ),
            lineage=metadata["lineage"],
            provenance=metadata["provenance"],
        )
    validate_particle_sequence(sequence)
    return sequence
