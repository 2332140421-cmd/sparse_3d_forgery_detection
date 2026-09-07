"""Minimal, source-level GenVidBench metadata parser and pilot selector.

The parser is deliberately tied to the fields present in the official files;
it does not infer labels from fake content or model outputs.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
from typing import Iterable


TARGET_GENERATORS = ("musev", "svd", "mora", "cogvideo")
TRAIN_REAL_SOURCE = "Vript"
TEST_REAL_SOURCE = "HD-VG-130M"


@dataclass(frozen=True, slots=True)
class SemanticRecord:
    source: str
    ordinal: str
    source_identity: str
    object_label: str
    action_label: str
    location_label: str
    caption: str


@dataclass(frozen=True, slots=True)
class LabeledVideo:
    relative_path: str
    subset: str
    label: int
    ordinal: str | None
    source_identity: str | None


@dataclass(frozen=True, slots=True)
class RealCandidate:
    source_id: str
    role: str
    real_source: str
    relative_path: str
    ordinal: str
    source_identity: str
    semantic_object: str
    semantic_action: str
    semantic_location: str
    caption: str
    prefilter_rule: str


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _taxonomy(classes_path: Path) -> dict[str, dict[int, str]]:
    text = classes_path.read_text(encoding="utf-8")
    result: dict[str, dict[int, str]] = {}
    for name in ("Object_dict", "Action_dict", "Location_dict"):
        match = re.search(rf"{name}\s*=\s*\{{(.*?)\}}", text, re.S)
        if match is None:
            raise ValueError(f"missing taxonomy mapping: {name}")
        values: dict[int, str] = {}
        for label, value in re.findall(r'"([^"]+)"\s*:\s*(\d+)', match.group(1)):
            values[int(value)] = label
        result[name.removesuffix("_dict").lower()] = values
    return result


def parse_semantic_file(path: Path, source: str, classes_path: Path) -> tuple[SemanticRecord, ...]:
    taxonomy = _taxonomy(classes_path)
    records: list[SemanticRecord] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        fields = line.split("|||")
        if len(fields) != 4:
            raise ValueError(f"{path}:{line_number} must contain four ||| fields")
        ordinal, source_identity, code, caption = fields
        parts = code.split("_")
        if len(parts) != 3 or not all(part.isdigit() for part in parts):
            raise ValueError(f"{path}:{line_number} has invalid semantic code")
        object_label = taxonomy["object"][int(parts[0])]
        action_label = taxonomy["action"][int(parts[1])]
        location_label = taxonomy["location"][int(parts[2])]
        records.append(
            SemanticRecord(
                source=source,
                ordinal=ordinal,
                source_identity=source_identity,
                object_label=object_label,
                action_label=action_label,
                location_label=location_label,
                caption=caption,
            )
        )
    return tuple(records)


def parse_label_file(path: Path) -> tuple[LabeledVideo, ...]:
    records: list[LabeledVideo] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        relative_path, label_text = line.rsplit(maxsplit=1)
        try:
            label = int(label_text)
        except ValueError as exc:
            raise ValueError(f"{path}:{line_number} has invalid label") from exc
        parts = PurePosixPath(relative_path).parts
        if len(parts) < 3:
            raise ValueError(f"{path}:{line_number} has invalid relative path")
        subset = parts[1]
        filename = parts[-1]
        stem = filename.removesuffix(".mp4")
        ordinal: str | None = None
        source_identity: str | None = None
        if "___" in stem:
            ordinal_text, payload = stem.split("___", 1)
            if ordinal_text.isdigit():
                ordinal = ordinal_text
                if subset == "hd_vg_130m":
                    source_identity = payload.split(".", 1)[0]
        elif subset == "vript":
            source_identity = stem
        if ordinal is None and source_identity is None and label == 0:
            raise ValueError(f"{path}:{line_number} has no source identity key")
        records.append(
            LabeledVideo(
                relative_path=relative_path,
                subset=subset,
                label=label,
                ordinal=ordinal,
                source_identity=source_identity,
            )
        )
    return tuple(records)


def metadata_summary(
    *,
    pair1_labels: Iterable[LabeledVideo],
    pair2_labels: Iterable[LabeledVideo],
    vript: Iterable[SemanticRecord],
    hdvg: Iterable[SemanticRecord],
    vidprom: Iterable[SemanticRecord],
) -> dict[str, object]:
    pair1 = tuple(pair1_labels)
    pair2 = tuple(pair2_labels)
    return {
        "metadata_scope": "official V-143k label and semantic files",
        "pair1": {
            "total": len(pair1),
            "real": sum(row.label == 0 for row in pair1),
            "fake": sum(row.label == 1 for row in pair1),
            "subsets": dict(Counter(row.subset for row in pair1)),
        },
        "pair2": {
            "total": len(pair2),
            "real": sum(row.label == 0 for row in pair2),
            "fake": sum(row.label == 1 for row in pair2),
            "subsets": dict(Counter(row.subset for row in pair2)),
        },
        "semantic_rows": {
            "Vript": len(tuple(vript)),
            "HD-VG-130M": len(tuple(hdvg)),
            "VidProM": len(tuple(vidprom)),
        },
        "target_fake_generators": list(TARGET_GENERATORS),
        "pairing_key": "Pair2 ordinal plus shared source semantic record",
    }


def _prefilter_rule(record: SemanticRecord) -> str:
    if record.object_label == "Vehicles" and record.action_label != "Static Postures":
        return "VEHICLE_NONSTATIC_SEMANTIC"
    if record.action_label == "Static Postures":
        return "MANUAL_REVIEW_REQUIRED_STATIC_METADATA"
    return "MANUAL_REVIEW_REQUIRED_NONVEHICLE"


def select_real_candidates(
    records: Iterable[SemanticRecord],
    available_real: Iterable[LabeledVideo],
    *,
    role: str,
    real_source: str,
) -> tuple[RealCandidate, ...]:
    semantics_by_ordinal = {record.ordinal: record for record in records}
    semantics_by_identity = {record.source_identity: record for record in records}
    candidates: list[RealCandidate] = []
    for video in available_real:
        record = None
        if video.source_identity is not None:
            record = semantics_by_identity.get(video.source_identity)
        if record is None and video.ordinal is not None:
            record = semantics_by_ordinal.get(video.ordinal)
        if record is None:
            continue
        rule = _prefilter_rule(record)
        if rule != "VEHICLE_NONSTATIC_SEMANTIC":
            continue
        candidates.append(
            RealCandidate(
                source_id=f"{real_source}:{record.ordinal}:{record.source_identity}",
                role=role,
                real_source=real_source,
                relative_path=video.relative_path,
                ordinal=record.ordinal,
                source_identity=record.source_identity,
                semantic_object=record.object_label,
                semantic_action=record.action_label,
                semantic_location=record.location_label,
                caption=record.caption,
                prefilter_rule=rule,
            )
        )
    return tuple(sorted(candidates, key=lambda candidate: candidate.source_id))


def build_pilot_selection(
    *,
    pair1_labels: Iterable[LabeledVideo],
    pair2_labels: Iterable[LabeledVideo],
    vript_records: Iterable[SemanticRecord],
    hdvg_records: Iterable[SemanticRecord],
    train_limit: int = 32,
    test_limit: int = 16,
) -> dict[str, object]:
    pair1 = tuple(pair1_labels)
    pair2 = tuple(pair2_labels)
    train_real = select_real_candidates(
        vript_records,
        (row for row in pair1 if row.subset == "vript" and row.label == 0),
        role="train_real",
        real_source=TRAIN_REAL_SOURCE,
    )[:train_limit]
    test_real = select_real_candidates(
        hdvg_records,
        (row for row in pair2 if row.subset == "hd_vg_130m" and row.label == 0),
        role="test_real_source",
        real_source=TEST_REAL_SOURCE,
    )[:test_limit]
    fake_by_ordinal: dict[str, list[LabeledVideo]] = defaultdict(list)
    for row in pair2:
        if row.label == 1 and row.subset in TARGET_GENERATORS and row.ordinal is not None:
            fake_by_ordinal[row.ordinal].append(row)
    paired_fake: list[dict[str, object]] = []
    for source in test_real:
        fakes = sorted(fake_by_ordinal.get(source.ordinal, []), key=lambda row: row.subset)
        for fake in fakes:
            paired_fake.append(
                {
                    "source_id": source.source_id,
                    "real_video_path": source.relative_path,
                    "real_source": source.real_source,
                    "source_identity": source.source_identity,
                    "source_prompt_or_caption": source.caption,
                    "generator": fake.subset,
                    "fake_path": fake.relative_path,
                    "pair_key": source.ordinal,
                    "pair_lineage": "shared Pair2 ordinal and official HDVG semantic record",
                }
            )
    return {
        "protocol": "V7 GenVidBench metadata-first core pilot",
        "selection_rules": {
            "ordering": "source_id ascending",
            "train_real_limit": train_limit,
            "test_real_source_limit": test_limit,
            "rule": "object_label == Vehicles and action_label != Static Postures",
            "fake_informed": False,
            "model_informed": False,
        },
        "train_real": [asdict(row) for row in train_real],
        "test_real_source": [asdict(row) for row in test_real],
        "test_fake_paired": paired_fake,
    }


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
