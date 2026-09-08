"""Build exact ActivityForensics-to-Charades lineage manifests.

The current experiment needs an auditable candidate population, not a dataset
framework.  This module reads the official CSVs, annotation archive and
Charades IDs already downloaded under the data root.  It never downloads,
guesses a pairing, or filters by visual or model behaviour.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable
from zipfile import ZipFile


_EDIT_RE = re.compile(r"(?P<start>[0-9]+(?:\.[0-9]+)?)=(?P<end>[0-9]+(?:\.[0-9]+)?)=(?P<payload>[^+]+)$")


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def parse_activityforensics_filename(file_name: str) -> dict[str, Any]:
    """Parse all official edit segments without dropping multi-edit intervals."""

    name = Path(file_name).name
    stem = Path(name).stem
    if "+" not in stem:
        raise ValueError(f"missing manipulation segment: {file_name}")
    source_video_id, *encoded_segments = stem.split("+")
    segments: list[dict[str, Any]] = []
    for encoded in encoded_segments:
        match = _EDIT_RE.fullmatch(encoded)
        if match is None:
            raise ValueError(f"invalid official manipulation segment: {encoded}")
        values = match.groupdict()
        parts = values.pop("payload").split("@")
        if len(parts) not in (4, 5):
            raise ValueError(f"invalid official manipulation payload: {encoded}")
        source_dataset, split_operation, source_id = parts[:3]
        operation_id, generator = (parts[3], parts[4]) if len(parts) == 5 else (None, parts[3])
        if "_" not in split_operation:
            raise ValueError(f"invalid split/operation: {split_operation}")
        official_split, operation = split_operation.split("_", 1)
        segments.append(
            {
                "start_s": float(values["start"]),
                "end_s": float(values["end"]),
                "source_dataset": source_dataset,
                "official_split": official_split,
                "manipulation_operation": operation,
                "operation_id": operation_id,
                "generator": generator,
                "charades_source_id": source_id
                if source_dataset == "charades"
                else None,
            }
        )
    return {"source_video_id": source_video_id, "segments": segments}


def iter_metadata_rows(metadata_dir: Path) -> Iterable[dict[str, str]]:
    for official_file in ("train.csv", "test.csv"):
        with (metadata_dir / official_file).open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                file_name = row.get("file_name")
                if not file_name:
                    raise ValueError(f"missing file_name in {official_file}")
                yield {"activityforensics_split": official_file.removesuffix(".csv"), "file_name": file_name}


def read_official_charades_ids(metadata_dir: Path) -> dict[str, set[str]]:
    result: dict[str, set[str]] = {}
    for split in ("train", "test"):
        with (metadata_dir / f"Charades_v1_{split}.csv").open(newline="", encoding="utf-8") as handle:
            result[split] = {row["id"] for row in csv.DictReader(handle)}
    if result["train"] & result["test"]:
        raise ValueError("official Charades train/test IDs overlap")
    return result


def read_activity_annotations(annotation_zip: Path) -> dict[str, dict[str, Any]]:
    """Read official duration and interval annotations keyed by basename."""

    result: dict[str, dict[str, Any]] = {}
    with ZipFile(annotation_zip) as archive:
        for info in archive.infolist():
            if info.is_dir():
                continue
            annotation_file = info.filename
            for line_number, line in enumerate(archive.read(info).decode("utf-8").splitlines(), 1):
                columns = line.split()
                if not columns:
                    continue
                if len(columns) < 3:
                    raise ValueError(f"short annotation row: {annotation_file}:{line_number}")
                name, duration, encoded_intervals = columns[:3]
                intervals = []
                for interval in encoded_intervals.split("+"):
                    start, end = interval.split("=", 1)
                    intervals.append({"start_s": float(start), "end_s": float(end)})
                result[name] = {
                    "duration_s": float(duration),
                    "manipulation_intervals": intervals,
                    "annotation_file": annotation_file,
                    "annotation_line": line_number,
                }
    return result


def _lineage_status(segments: list[dict[str, Any]], official_ids: dict[str, set[str]]) -> tuple[str, str | None, list[str]]:
    charades = [segment for segment in segments if segment["source_dataset"] == "charades"]
    if not charades:
        return "UNRESOLVED", None, ["source dataset is not charades"]
    source_ids = sorted({str(segment["charades_source_id"]) for segment in charades})
    evidence: list[str] = []
    statuses: list[str] = []
    for source_id in source_ids:
        present = [split for split, ids in official_ids.items() if source_id in ids]
        referenced = sorted({str(segment["official_split"]) for segment in charades if segment["charades_source_id"] == source_id})
        evidence.append(f"Charades_v1_{present[0]}.csv:id={source_id}" if len(present) == 1 else f"Charades ID lookup={source_id}:{present}")
        if len(present) == 1 and referenced == present:
            statuses.append("EXACT")
        elif len(present) > 1:
            statuses.append("AMBIGUOUS")
        else:
            statuses.append("UNRESOLVED")
    if len(source_ids) != 1:
        statuses.append("AMBIGUOUS")
    status = "EXACT" if statuses and all(item == "EXACT" for item in statuses) else ("AMBIGUOUS" if "AMBIGUOUS" in statuses else "UNRESOLVED")
    return status, source_ids[0] if len(source_ids) == 1 else None, evidence


def build_mapping(metadata_dir: Path, annotation_zip: Path, charades_metadata_dir: Path) -> list[dict[str, Any]]:
    official_ids = read_official_charades_ids(charades_metadata_dir)
    annotations = read_activity_annotations(annotation_zip)
    rows: list[dict[str, Any]] = []
    for metadata_row in iter_metadata_rows(metadata_dir):
        file_name = metadata_row["file_name"]
        parsed = parse_activityforensics_filename(file_name)
        segments = parsed["segments"]
        annotation = annotations.get(Path(file_name).name)
        status, source_id, lookup_evidence = _lineage_status(segments, official_ids)
        generators = sorted({segment["generator"] for segment in segments})
        operations = sorted({segment["manipulation_operation"] for segment in segments})
        source_datasets = sorted({segment["source_dataset"] for segment in segments})
        official_splits = sorted({segment["official_split"] for segment in segments})
        evidence = [
            f"source/activityforensics/metadata/{metadata_row['activityforensics_split']}.csv:file_name",
            *(f"filename segment: {segment['source_dataset']}@{segment['official_split']}_{segment['manipulation_operation']}@{segment.get('charades_source_id') or segment['operation_id']}" for segment in segments),
            *lookup_evidence,
        ]
        if annotation is None:
            status = "UNRESOLVED"
            evidence.append("official annot.zip basename lookup: missing")
            intervals = [
                {"start_s": segment["start_s"], "end_s": segment["end_s"]}
                for segment in segments
            ]
        else:
            evidence.append(f"source/activityforensics/raw/annot.zip:{annotation['annotation_file']}:{annotation['annotation_line']}")
            intervals = annotation["manipulation_intervals"]
        rows.append(
            {
                "activityforensics_file": file_name,
                "activityforensics_split": metadata_row["activityforensics_split"],
                "official_split": official_splits[0] if len(official_splits) == 1 else official_splits,
                "generator": generators[0] if len(generators) == 1 else generators,
                "source_dataset": source_datasets[0] if len(source_datasets) == 1 else source_datasets,
                "charades_source_id": source_id,
                "manipulation_intervals": intervals,
                "manipulation_operation": operations[0] if len(operations) == 1 else operations,
                "manipulation_segments": segments,
                "lineage_evidence": evidence,
                "lineage_status": status,
                "annotation_duration_s": annotation["duration_s"] if annotation else None,
            }
        )
    return rows


def inventory(rows: list[dict[str, Any]]) -> dict[str, Any]:
    exact = [row for row in rows if row["lineage_status"] == "EXACT"]
    exact_sources = sorted({row["charades_source_id"] for row in exact if row["charades_source_id"]})
    return {
        "activityforensics_fake_files": len(rows),
        "unique_charades_source_ids_exact": len(exact_sources),
        "lineage_status_counts": dict(sorted(Counter(row["lineage_status"] for row in rows).items())),
        "activityforensics_split_counts": dict(sorted(Counter(row["activityforensics_split"] for row in rows).items())),
        "official_split_counts": dict(sorted(Counter(segment["official_split"] for row in rows for segment in row["manipulation_segments"]).items())),
        "generator_counts": dict(sorted(Counter(segment["generator"] for row in rows for segment in row["manipulation_segments"]).items())),
        "operation_counts": dict(sorted(Counter(segment["manipulation_operation"] for row in rows for segment in row["manipulation_segments"]).items())),
        "single_vs_multi_edit_counts": dict(sorted(Counter(len(row["manipulation_segments"]) for row in rows).items())),
        "per_source_fake_variant_counts": dict(sorted(Counter(row["charades_source_id"] for row in exact if row["charades_source_id"]).items())),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    args = parser.parse_args()
    base: Path = args.base
    rows = build_mapping(
        base / "source/activityforensics/metadata",
        base / "source/activityforensics/raw/annot.zip",
        base / "source/charades/metadata",
    )
    mapping_path = base / "paired/manifests/activityforensics_source_mapping.json"
    write_json(mapping_path, rows)
    write_json(base / "validation/lineage_validation.json", {"inventory": inventory(rows), "mapping_path": str(mapping_path), "official_ids_verified": True})
    print(json.dumps(inventory(rows), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
