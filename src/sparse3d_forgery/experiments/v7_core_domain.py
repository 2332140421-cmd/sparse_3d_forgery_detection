"""Deterministic V7 Core Domain candidate selection and human-review materials."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from fractions import Fraction
import hashlib
import json
import os
from pathlib import Path
from typing import Iterable

import av
import numpy as np
from PIL import Image, ImageDraw

from sparse3d_forgery.video_input import VideoSource, decode_video


ALLOWED_DECISIONS = ("IN_DOMAIN", "OUT_OF_DOMAIN", "UNCERTAIN")
ALLOWED_REASON_CODES = (
    "SINGLE_STRUCTURED_MOTION",
    "MULTI_SUBJECT_COMPLEX",
    "ARTICULATED_COMPLEX",
    "STATIC_DOMINANT",
    "STRONG_OCCLUSION",
    "TOPOLOGY_CHANGE",
    "FLUID_SMOKE_FIRE",
    "CAMERA_MOTION_DOMINANT",
    "NO_PERSISTENT_STRUCTURE",
    "SUBJECT_TOO_SMALL",
    "OTHER",
)
REVIEW_COLUMNS = (
    "source_video_id",
    "split",
    "video_path",
    "frame_start",
    "frame_end",
    "preview_path",
    "contact_sheet_path",
    "decision",
    "reason_code",
    "notes",
)


@dataclass(frozen=True, slots=True)
class CoreDomainCandidate:
    """One manifest-defined review window, independent of model outputs."""

    source_video_id: str
    split: str
    video_path: str
    frame_indices: tuple[int, ...]


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_core_domain_candidates(
    manifest_path: Path,
    *,
    real_train_limit: int = 64,
    real_val_limit: int = 32,
) -> tuple[CoreDomainCandidate, ...]:
    """Select the fixed real-only population by source_video_id ordering."""

    with manifest_path.open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    items = manifest.get("items")
    if not isinstance(items, list):
        raise ValueError("pilot manifest must contain an items list")

    selected: list[CoreDomainCandidate] = []
    for role, split, limit in (
        ("real_train", "train", real_train_limit),
        ("real_val", "val", real_val_limit),
    ):
        if isinstance(limit, bool) or not isinstance(limit, int) or limit < 0:
            raise ValueError("population limits must be non-negative integers")
        population = [
            item
            for item in items
            if item.get("role") == role and item.get("official_split") == split
        ]
        population.sort(key=lambda item: str(item.get("source_video_id", "")))
        for item in population[:limit]:
            source_video_id = item.get("source_video_id")
            video_path = item.get("video_path")
            indices = item.get("frame_indices")
            if not isinstance(source_video_id, str) or not source_video_id:
                raise ValueError("candidate source_video_id must be non-empty")
            if not isinstance(video_path, str) or not video_path:
                raise ValueError("candidate video_path must be non-empty")
            if not isinstance(indices, list) or not indices:
                raise ValueError("candidate frame_indices must be a non-empty list")
            if any(isinstance(index, bool) or not isinstance(index, int) or index < 0 for index in indices):
                raise ValueError("candidate frame_indices must be non-negative integers")
            if any(right <= left for left, right in zip(indices, indices[1:])):
                raise ValueError("candidate frame_indices must be strictly increasing")
            selected.append(
                CoreDomainCandidate(
                    source_video_id=source_video_id,
                    split=split,
                    video_path=video_path,
                    frame_indices=tuple(indices),
                )
            )

    identifiers = [candidate.source_video_id for candidate in selected]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("candidate source_video_id values must be unique")
    return tuple(selected)


def _preview_rate(timestamps_s: np.ndarray) -> Fraction:
    if timestamps_s.size < 2:
        return Fraction(30, 1)
    delta = float(np.median(np.diff(timestamps_s)))
    if not np.isfinite(delta) or delta <= 0:
        return Fraction(30, 1)
    return Fraction(1.0 / delta).limit_denominator(1000)


def _write_preview_mp4(decoded, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    rate = _preview_rate(decoded.timestamps_s)
    try:
        with av.open(str(temporary), mode="w", format="mp4") as container:
            stream = container.add_stream("mpeg4", rate=rate)
            height, width = decoded.frames[0].rgb.shape[:2]
            stream.width = width
            stream.height = height
            stream.pix_fmt = "yuv420p"
            for index, decoded_frame in enumerate(decoded.frames):
                frame = av.VideoFrame.from_ndarray(decoded_frame.rgb, format="rgb24")
                frame.pts = index
                for packet in stream.encode(frame):
                    container.mux(packet)
            for packet in stream.encode():
                container.mux(packet)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def _write_contact_sheet(decoded, candidate: CoreDomainCandidate, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    tile_width, tile_height, image_height = 320, 252, 210
    sheet = Image.new("RGB", (tile_width * 4, tile_height * 4), "white")
    draw = ImageDraw.Draw(sheet)
    for index, decoded_frame in enumerate(decoded.frames):
        row, column = divmod(index, 4)
        image = Image.fromarray(decoded_frame.rgb, mode="RGB")
        image.thumbnail((tile_width - 8, image_height - 8), Image.Resampling.LANCZOS)
        left = column * tile_width + (tile_width - image.width) // 2
        top = row * tile_height + 4
        sheet.paste(image, (left, top))
        text = (
            f"{candidate.source_video_id}\n"
            f"split={candidate.split} frame={decoded_frame.source_frame_index}\n"
            f"t={decoded_frame.timestamp_s:.6f}s"
        )
        draw.multiline_text((column * tile_width + 4, row * tile_height + image_height + 5), text, fill="black")
    sheet.save(destination, format="PNG")


def _review_row(candidate: CoreDomainCandidate, review_root: Path) -> dict[str, str]:
    stem = candidate.source_video_id
    return {
        "source_video_id": candidate.source_video_id,
        "split": candidate.split,
        "video_path": candidate.video_path,
        "frame_start": str(candidate.frame_indices[0]),
        "frame_end": str(candidate.frame_indices[-1]),
        "preview_path": str(Path("raw_previews") / f"{stem}.mp4"),
        "contact_sheet_path": str(Path("contact_sheets") / f"{stem}.png"),
        "decision": "",
        "reason_code": "",
        "notes": "",
    }


def build_review_materials(
    candidates: Iterable[CoreDomainCandidate],
    *,
    extracted_root: Path,
    output_root: Path,
) -> dict[str, object]:
    """Decode each fixed window and build materials without domain decisions."""

    candidates = tuple(candidates)
    output_root.mkdir(parents=True, exist_ok=True)
    raw_root = output_root / "raw_previews"
    sheet_root = output_root / "contact_sheets"
    results: list[dict[str, object]] = []
    rows: list[dict[str, str]] = []

    for candidate in candidates:
        row = _review_row(candidate, output_root)
        result: dict[str, object] = {
            "source_video_id": candidate.source_video_id,
            "split": candidate.split,
            "video_path": candidate.video_path,
            "frame_indices": list(candidate.frame_indices),
            "status": "failed",
            "preview_path": row["preview_path"],
            "contact_sheet_path": row["contact_sheet_path"],
        }
        try:
            decoded = decode_video(
                VideoSource(
                    sample_id=f"v7-core-domain-{candidate.source_video_id}",
                    source_video_id=candidate.source_video_id,
                    source_locator=extracted_root / candidate.video_path,
                ),
                candidate.frame_indices,
            )
            result["timestamps_s"] = decoded.timestamps_s.tolist()
            _write_preview_mp4(decoded, output_root / row["preview_path"])
            _write_contact_sheet(decoded, candidate, output_root / row["contact_sheet_path"])
            result["status"] = "success"
        except Exception as exc:
            result["error"] = f"{type(exc).__name__}: {exc}"
        results.append(result)
        rows.append(row)

    review_manifest = {
        "protocol": "V7 Core Domain Phase-2 human review",
        "selection": {
            "real_train": 64,
            "real_val": 32,
            "ordering": "source_video_id ascending",
            "model_independent": True,
            "fake_included": False,
        },
        "candidates": results,
    }
    _atomic_json(output_root / "review_manifest.json", review_manifest)
    _atomic_json(
        output_root / "review_schema.json",
        {
            "allowed_decisions": list(ALLOWED_DECISIONS),
            "allowed_reason_codes": list(ALLOWED_REASON_CODES),
            "review_columns": list(REVIEW_COLUMNS),
            "decision_is_manual": True,
            "uncertain_is_excluded_by_default": True,
        },
    )
    with (output_root / "review_template.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=REVIEW_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    success = sum(result["status"] == "success" for result in results)
    summary = {
        "candidate_count": len(results),
        "split_counts": {
            "train": sum(candidate.split == "train" for candidate in candidates),
            "val": sum(candidate.split == "val" for candidate in candidates),
        },
        "preview_success": success,
        "preview_failure": len(results) - success,
        "selection_is_model_independent": True,
        "fake_candidate_count": 0,
        "no_domain_decision_made": True,
    }
    _atomic_json(output_root / "population_summary.json", summary)
    return summary
