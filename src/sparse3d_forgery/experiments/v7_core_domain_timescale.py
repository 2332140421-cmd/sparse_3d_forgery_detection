"""V7 video-domain and multi-timescale human-review materials.

This module deliberately consumes only the Phase-A review manifest and raw RGB
videos.  It never reads frontend, component, score, label, or fake artifacts.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Iterable, Sequence

import av
import numpy as np
from PIL import Image, ImageDraw

from sparse3d_forgery.video_input import VideoSource, decode_video


VIDEO_DECISIONS = ("IN_DOMAIN", "OUT_OF_DOMAIN", "UNCERTAIN")
VIDEO_REASON_CODES = (
    "SINGLE_STRUCTURED_MOTION",
    "MULTI_SUBJECT_COMPLEX",
    "ARTICULATED_COMPLEX",
    "STATIC_VIDEO",
    "STRONG_OCCLUSION",
    "TOPOLOGY_CHANGE",
    "FLUID_SMOKE_FIRE",
    "CAMERA_MOTION_DOMINANT",
    "NO_PERSISTENT_STRUCTURE",
    "SUBJECT_TOO_SMALL",
    "OTHER",
)
WINDOW_DECISIONS = (
    "DYNAMIC_ADEQUATE",
    "DYNAMIC_TOO_WEAK",
    "MOTION_EVENT_PARTIAL",
    "OCCLUSION_DOMINANT",
    "CAMERA_MOTION_DOMINANT",
    "UNCERTAIN",
)
WINDOW_REASON_CODES = WINDOW_DECISIONS + ("WINDOW_DURATION_UNAVAILABLE",)
TEMPORAL_DURATIONS_S = (0.5, 1.0, 2.0)
TEMPORAL_ANCHORS = (0.25, 0.50, 0.75)
FULL_VIDEO_COLUMNS = (
    "source_video_id",
    "split",
    "video_path",
    "duration_s",
    "frame_count",
    "full_contact_sheet_path",
    "video_decision",
    "video_reason_code",
    "video_notes",
)
WINDOW_COLUMNS = (
    "source_video_id",
    "split",
    "anchor_fraction",
    "requested_duration_s",
    "actual_duration_s",
    "frame_start",
    "frame_end",
    "timestamp_start_s",
    "timestamp_end_s",
    "preview_path",
    "contact_sheet_path",
    "window_decision",
    "window_reason_code",
    "window_notes",
)


@dataclass(frozen=True, slots=True)
class PhaseAReviewCandidate:
    source_video_id: str
    split: str
    video_path: str
    frame_indices: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class VideoTimeline:
    frame_indices: tuple[int, ...]
    timestamps_s: tuple[float, ...]
    duration_s: float

    @property
    def frame_count(self) -> int:
        return len(self.frame_indices)

    @property
    def timestamp_start_s(self) -> float:
        return self.timestamps_s[0]

    @property
    def timestamp_end_s(self) -> float:
        return self.timestamps_s[-1]

    @property
    def span_s(self) -> float:
        return self.timestamp_end_s - self.timestamp_start_s


@dataclass(frozen=True, slots=True)
class WindowSelection:
    anchor_fraction: float
    requested_duration_s: float
    actual_duration_s: float
    frame_start: int | None
    frame_end: int | None
    timestamp_start_s: float | None
    timestamp_end_s: float | None
    status: str


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


def load_phase_a_review_manifest(
    manifest_path: Path,
    *,
    expected_count: int = 96,
    expected_train: int = 64,
    expected_val: int = 32,
) -> tuple[PhaseAReviewCandidate, ...]:
    """Reuse Phase-A identities exactly; never reselect from a pilot manifest."""

    with manifest_path.open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    rows = manifest.get("candidates")
    if not isinstance(rows, list):
        raise ValueError("Phase-A review manifest must contain candidates")
    candidates: list[PhaseAReviewCandidate] = []
    for row in rows:
        if not isinstance(row, dict) or row.get("status") != "success":
            raise ValueError("all Phase-A candidates must have successful review material")
        source_video_id = row.get("source_video_id")
        split = row.get("split")
        video_path = row.get("video_path")
        frame_indices = row.get("frame_indices")
        if not isinstance(source_video_id, str) or not source_video_id:
            raise ValueError("source_video_id must be non-empty")
        if split not in {"train", "val"}:
            raise ValueError("Phase-A population may only contain train and val")
        if not isinstance(video_path, str) or not video_path:
            raise ValueError("video_path must be non-empty")
        if not isinstance(frame_indices, list) or not frame_indices:
            raise ValueError("frame_indices must be a non-empty list")
        if any(
            isinstance(index, bool) or not isinstance(index, int) or index < 0
            for index in frame_indices
        ):
            raise ValueError("frame_indices must contain non-negative integers")
        if any(right <= left for left, right in zip(frame_indices, frame_indices[1:])):
            raise ValueError("Phase-A frame_indices must be strictly increasing")
        candidates.append(
            PhaseAReviewCandidate(
                source_video_id=source_video_id,
                split=split,
                video_path=video_path,
                frame_indices=tuple(frame_indices),
            )
        )

    identifiers = [candidate.source_video_id for candidate in candidates]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("Phase-A source_video_id values must be unique")
    if len(candidates) != expected_count:
        raise ValueError(f"expected exactly {expected_count} Phase-A candidates")
    if sum(candidate.split == "train" for candidate in candidates) != expected_train:
        raise ValueError("Phase-A train population does not match the frozen count")
    if sum(candidate.split == "val" for candidate in candidates) != expected_val:
        raise ValueError("Phase-A validation population does not match the frozen count")
    return tuple(candidates)


def read_video_timeline(video_path: Path) -> VideoTimeline:
    """Read frame ordinals and true PTS timestamps without any model output."""

    frame_indices: list[int] = []
    timestamps_s: list[float] = []
    try:
        with av.open(str(video_path), mode="r") as container:
            if not container.streams.video:
                raise ValueError(f"video has no video stream: {video_path}")
            stream = container.streams.video[0]
            for frame_index, frame in enumerate(container.decode(stream)):
                if frame.pts is None or frame.time_base is None:
                    raise ValueError(f"frame {frame_index} has no PTS time base: {video_path}")
                timestamp_s = float(frame.pts * frame.time_base)
                if not math.isfinite(timestamp_s):
                    raise ValueError(f"frame {frame_index} PTS is not finite: {video_path}")
                if timestamps_s and timestamp_s <= timestamps_s[-1]:
                    raise ValueError(f"video PTS is not strictly increasing: {video_path}")
                frame_indices.append(frame_index)
                timestamps_s.append(timestamp_s)
            if not timestamps_s:
                raise ValueError(f"video contains no decodable frames: {video_path}")
            stream_duration = None
            if stream.duration is not None and stream.time_base is not None:
                candidate_duration = float(stream.duration * stream.time_base)
                if math.isfinite(candidate_duration) and candidate_duration > 0:
                    stream_duration = candidate_duration
            duration_s = max(
                timestamps_s[-1],
                stream_duration if stream_duration is not None else timestamps_s[-1],
            )
    except (av.FFmpegError, OSError) as exc:
        raise ValueError(f"failed to read video timeline: {video_path}") from exc
    return VideoTimeline(tuple(frame_indices), tuple(timestamps_s), float(duration_s))


def uniform_frame_indices(frame_count: int, count: int = 20) -> tuple[int, ...]:
    """Return deterministic unique samples spanning the first and last frame."""

    if isinstance(frame_count, bool) or not isinstance(frame_count, int) or frame_count <= 0:
        raise ValueError("frame_count must be a positive integer")
    if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
        raise ValueError("count must be a positive integer")
    if frame_count <= count:
        return tuple(range(frame_count))
    indices = np.rint(np.linspace(0, frame_count - 1, count)).astype(np.int64)
    unique = tuple(int(index) for index in np.unique(indices))
    if unique[0] != 0 or unique[-1] != frame_count - 1:
        raise AssertionError("uniform samples must span the full timeline")
    return unique


def _write_contact_sheet(frames: Sequence[object], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    tile_width, tile_height, image_height = 320, 252, 210
    sheet = Image.new("RGB", (tile_width * 4, tile_height * 5), "white")
    draw = ImageDraw.Draw(sheet)
    for position, decoded_frame in enumerate(frames):
        row, column = divmod(position, 4)
        image = Image.fromarray(decoded_frame.rgb, mode="RGB")
        image.thumbnail((tile_width - 8, image_height - 8), Image.Resampling.LANCZOS)
        left = column * tile_width + (tile_width - image.width) // 2
        top = row * tile_height + 4
        sheet.paste(image, (left, top))
        text = f"frame={decoded_frame.source_frame_index}\nPTS={decoded_frame.timestamp_s:.6f}s"
        draw.multiline_text(
            (column * tile_width + 4, row * tile_height + image_height + 5),
            text,
            fill="black",
        )
    temporary = destination.with_name(f".{destination.name}.tmp")
    sheet.save(temporary, format="PNG")
    os.replace(temporary, destination)


def _decode_samples(video_path: Path, source_video_id: str, indices: Sequence[int]):
    return decode_video(
        VideoSource(
            sample_id=f"v7-full-video-review-{source_video_id}",
            source_video_id=source_video_id,
            source_locator=video_path,
        ),
        indices,
    )


def build_full_video_review(
    candidates: Iterable[PhaseAReviewCandidate],
    *,
    extracted_root: Path,
    output_root: Path,
    sample_count: int = 20,
) -> dict[str, object]:
    """Build blank video-level review rows and full-duration contact sheets."""

    candidates = tuple(candidates)
    video_root = output_root / "video_review"
    sheet_root = video_root / "full_contact_sheets"
    rows: list[dict[str, str]] = []
    manifest_rows: list[dict[str, object]] = []
    for candidate in candidates:
        source_path = Path(candidate.video_path)
        if not source_path.is_absolute():
            source_path = extracted_root / source_path
        source_path = source_path.resolve()
        if not source_path.is_file():
            raise FileNotFoundError(source_path)
        timeline = read_video_timeline(source_path)
        sample_indices = uniform_frame_indices(timeline.frame_count, sample_count)
        decoded = _decode_samples(source_path, candidate.source_video_id, sample_indices)
        sheet_path = sheet_root / f"{candidate.source_video_id}.png"
        _write_contact_sheet(decoded.frames, sheet_path)
        sheet_path = sheet_path.resolve()
        rows.append(
            {
                "source_video_id": candidate.source_video_id,
                "split": candidate.split,
                "video_path": str(source_path),
                "duration_s": f"{timeline.duration_s:.9f}",
                "frame_count": str(timeline.frame_count),
                "full_contact_sheet_path": str(sheet_path),
                "video_decision": "",
                "video_reason_code": "",
                "video_notes": "",
            }
        )
        manifest_rows.append(
            {
                "source_video_id": candidate.source_video_id,
                "split": candidate.split,
                "source_video_path": candidate.video_path,
                "video_path": str(source_path),
                "frame_indices": list(timeline.frame_indices),
                "timestamps_s": list(timeline.timestamps_s),
                "duration_s": timeline.duration_s,
                "frame_count": timeline.frame_count,
                "full_contact_sheet_path": str(sheet_path),
                "video_decision": "",
                "video_reason_code": "",
                "video_notes": "",
            }
        )

    manifest_path = video_root / "full_video_manifest.json"
    _atomic_json(
        manifest_path,
        {
            "protocol": "V7 video-level domain review and multi-timescale window review",
            "selection_source": "Phase-A review_manifest.json",
            "selection": {
                "candidate_count": len(candidates),
                "train_count": sum(candidate.split == "train" for candidate in candidates),
                "val_count": sum(candidate.split == "val" for candidate in candidates),
                "fake_count": 0,
                "model_independent": True,
            },
            "candidates": manifest_rows,
        },
    )
    review_csv = video_root / "full_video_review.csv"
    review_csv.parent.mkdir(parents=True, exist_ok=True)
    with review_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FULL_VIDEO_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "candidate_count": len(candidates),
        "train_count": sum(candidate.split == "train" for candidate in candidates),
        "val_count": sum(candidate.split == "val" for candidate in candidates),
        "fake_count": 0,
        "full_contact_sheet_success": len(rows),
        "full_contact_sheet_failure": 0,
        "decision_is_manual": True,
        "window_review_deferred_until_video_review": True,
    }
    _atomic_json(output_root / "reports" / "population_summary.json", summary)
    return summary


def _validate_timestamps(timestamps_s: Sequence[float]) -> np.ndarray:
    values = np.asarray(timestamps_s, dtype=np.float64)
    if values.ndim != 1 or values.size == 0 or not np.all(np.isfinite(values)):
        raise ValueError("timestamps_s must be a non-empty finite one-dimensional sequence")
    if values.size > 1 and not np.all(np.diff(values) > 0):
        raise ValueError("timestamps_s must be strictly increasing")
    return values


def centered_window_selection(
    timestamps_s: Sequence[float],
    *,
    anchor_fraction: float,
    requested_duration_s: float,
) -> WindowSelection:
    """Construct a legal centered window from true timestamps, without padding."""

    values = _validate_timestamps(timestamps_s)
    if not math.isfinite(anchor_fraction) or not 0 < anchor_fraction < 1:
        raise ValueError("anchor_fraction must be strictly between zero and one")
    if not math.isfinite(requested_duration_s) or requested_duration_s <= 0:
        raise ValueError("requested_duration_s must be positive and finite")
    start_time = float(values[0])
    end_time = float(values[-1])
    span = end_time - start_time
    if span < requested_duration_s:
        return WindowSelection(
            anchor_fraction=anchor_fraction,
            requested_duration_s=requested_duration_s,
            actual_duration_s=span,
            frame_start=0,
            frame_end=int(values.size - 1),
            timestamp_start_s=start_time,
            timestamp_end_s=end_time,
            status="WINDOW_DURATION_UNAVAILABLE",
        )

    center = start_time + anchor_fraction * span
    requested_start = max(start_time, center - requested_duration_s / 2.0)
    requested_end = min(end_time, center + requested_duration_s / 2.0)
    left = int(np.searchsorted(values, requested_start, side="left"))
    right = int(np.searchsorted(values, requested_end, side="right") - 1)
    if left > right:
        nearest = int(np.argmin(np.abs(values - center)))
        left = right = nearest
    return WindowSelection(
        anchor_fraction=anchor_fraction,
        requested_duration_s=requested_duration_s,
        actual_duration_s=float(values[right] - values[left]),
        frame_start=left,
        frame_end=right,
        timestamp_start_s=float(values[left]),
        timestamp_end_s=float(values[right]),
        status="AVAILABLE",
    )


def _read_csv(path: Path, columns: Sequence[str]) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != tuple(columns):
            raise ValueError(f"{path} columns do not match the review schema")
        return list(reader)


def build_window_review_rows(
    full_video_review_csv: Path,
    full_video_manifest_path: Path,
    *,
    output_root: Path,
) -> list[dict[str, str]]:
    """Prepare windows only for rows manually marked ``IN_DOMAIN``."""

    review_rows = _read_csv(full_video_review_csv, FULL_VIDEO_COLUMNS)
    with full_video_manifest_path.open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    by_id = {row["source_video_id"]: row for row in manifest.get("candidates", [])}
    rows: list[dict[str, str]] = []
    for review in review_rows:
        decision = review["video_decision"].strip()
        if decision not in VIDEO_DECISIONS:
            raise ValueError(
                f"video_decision must be one of {VIDEO_DECISIONS}; blank rows await human review"
            )
        if decision != "IN_DOMAIN":
            continue
        if review["video_reason_code"].strip() != "SINGLE_STRUCTURED_MOTION":
            raise ValueError("IN_DOMAIN requires SINGLE_STRUCTURED_MOTION")
        candidate = by_id.get(review["source_video_id"])
        if candidate is None:
            raise ValueError(f"review row is not present in full video manifest: {review['source_video_id']}")
        timeline = _validate_timestamps(candidate["timestamps_s"])
        video_id = review["source_video_id"]
        for anchor in TEMPORAL_ANCHORS:
            for duration in TEMPORAL_DURATIONS_S:
                selection = centered_window_selection(
                    timeline,
                    anchor_fraction=anchor,
                    requested_duration_s=duration,
                )
                stem = f"{video_id}__a{int(anchor * 100):02d}__d{str(duration).replace('.', 'p')}"
                rows.append(
                    {
                        "source_video_id": video_id,
                        "split": review["split"],
                        "anchor_fraction": f"{anchor:.2f}",
                        "requested_duration_s": f"{duration:.3f}",
                        "actual_duration_s": f"{selection.actual_duration_s:.9f}",
                        "frame_start": "" if selection.frame_start is None else str(selection.frame_start),
                        "frame_end": "" if selection.frame_end is None else str(selection.frame_end),
                        "timestamp_start_s": ""
                        if selection.timestamp_start_s is None
                        else f"{selection.timestamp_start_s:.9f}",
                        "timestamp_end_s": ""
                        if selection.timestamp_end_s is None
                        else f"{selection.timestamp_end_s:.9f}",
                        "preview_path": str(Path("previews") / f"{stem}.mp4"),
                        "contact_sheet_path": str(Path("contact_sheets") / f"{stem}.png"),
                        "window_decision": "",
                        "window_reason_code": selection.status
                        if selection.status == "WINDOW_DURATION_UNAVAILABLE"
                        else "",
                        "window_notes": "",
                    }
                )
    return rows


def write_window_review_csv(rows: Iterable[dict[str, str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=WINDOW_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def finalize_video_domain_review(
    full_video_review_csv: Path,
    full_video_manifest_path: Path,
    output_path: Path,
) -> dict[str, object]:
    """Create the core-domain manifest after explicit human video review."""

    rows = _read_csv(full_video_review_csv, FULL_VIDEO_COLUMNS)
    with full_video_manifest_path.open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    by_id = {row["source_video_id"]: row for row in manifest.get("candidates", [])}
    selected: list[dict[str, object]] = []
    seen: set[str] = set()
    for row in rows:
        source_video_id = row["source_video_id"]
        if source_video_id in seen:
            raise ValueError(f"duplicate video review row: {source_video_id}")
        seen.add(source_video_id)
        decision = row["video_decision"].strip()
        reason = row["video_reason_code"].strip()
        if decision not in VIDEO_DECISIONS:
            raise ValueError(f"invalid or blank video_decision: {source_video_id}")
        if reason not in VIDEO_REASON_CODES:
            raise ValueError(f"invalid or blank video_reason_code: {source_video_id}")
        if decision == "IN_DOMAIN":
            if reason != "SINGLE_STRUCTURED_MOTION":
                raise ValueError("IN_DOMAIN requires SINGLE_STRUCTURED_MOTION")
            source = by_id.get(source_video_id)
            if source is None:
                raise ValueError(f"video review row is not in full manifest: {source_video_id}")
            selected.append(
                {
                    "source_video_id": source_video_id,
                    "split": row["split"],
                    "source_video_path": source["source_video_path"],
                    "video_path": source["video_path"],
                    "frame_indices": source["frame_indices"],
                    "timestamps_s": source["timestamps_s"],
                    "duration_s": source["duration_s"],
                    "video_decision": decision,
                    "video_reason_code": reason,
                    "video_notes": row["video_notes"],
                }
            )
    result = {
        "protocol": "V7 video-level domain review",
        "review_csv_sha256": sha256_file(full_video_review_csv),
        "uncertain_excluded_by_default": True,
        "selected": selected,
    }
    _atomic_json(output_path, result)
    return result


def finalize_window_review(
    window_review_csv: Path,
    video_core_domain_manifest_path: Path,
    output_path: Path,
) -> dict[str, object]:
    """Finalize reviewed windows only for already selected IN_DOMAIN videos."""

    rows = _read_csv(window_review_csv, WINDOW_COLUMNS)
    with video_core_domain_manifest_path.open(encoding="utf-8") as handle:
        video_manifest = json.load(handle)
    allowed_video_ids = {row["source_video_id"] for row in video_manifest.get("selected", [])}
    finalized: list[dict[str, str]] = []
    seen: set[tuple[str, str, str]] = set()
    for row in rows:
        key = (row["source_video_id"], row["anchor_fraction"], row["requested_duration_s"])
        if key in seen:
            raise ValueError(f"duplicate window review row: {key}")
        seen.add(key)
        if row["source_video_id"] not in allowed_video_ids:
            raise ValueError("window review contains a video that is not IN_DOMAIN")
        decision = row["window_decision"].strip()
        reason = row["window_reason_code"].strip()
        if decision not in WINDOW_DECISIONS and not (
            decision == "" and reason == "WINDOW_DURATION_UNAVAILABLE"
        ):
            raise ValueError(f"invalid window review decision: {key}")
        if reason and reason not in WINDOW_REASON_CODES:
            raise ValueError(f"invalid window review reason: {key}")
        if decision == "UNCERTAIN":
            continue
        finalized.append(dict(row))
    result = {
        "protocol": "V7 multi-timescale window adequacy review",
        "review_csv_sha256": sha256_file(window_review_csv),
        "uncertain_excluded_by_default": True,
        "windows": finalized,
    }
    _atomic_json(output_path, result)
    return result

def _preview_rate(timestamps_s: Sequence[float]):
    if len(timestamps_s) < 2:
        return 30
    delta = float(np.median(np.diff(np.asarray(timestamps_s, dtype=np.float64))))
    if not math.isfinite(delta) or delta <= 0:
        return 30
    from fractions import Fraction

    return Fraction(1.0 / delta).limit_denominator(1000)


def _write_preview_mp4(decoded, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    try:
        with av.open(str(temporary), mode="w", format="mp4") as container:
            stream = container.add_stream("mpeg4", rate=_preview_rate(decoded.timestamps_s))
            height, width = decoded.frames[0].rgb.shape[:2]
            stream.width = width
            stream.height = height
            stream.pix_fmt = "yuv420p"
            for position, decoded_frame in enumerate(decoded.frames):
                frame = av.VideoFrame.from_ndarray(decoded_frame.rgb, format="rgb24")
                frame.pts = position
                for packet in stream.encode(frame):
                    container.mux(packet)
            for packet in stream.encode():
                container.mux(packet)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def generate_window_review_materials(
    rows: Sequence[dict[str, str]],
    full_video_manifest_path: Path,
    *,
    output_root: Path,
) -> None:
    """Generate optional RGB-only previews/sheets for selected rows."""

    with full_video_manifest_path.open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    by_id = {row["source_video_id"]: row for row in manifest.get("candidates", [])}
    for row in rows:
        candidate = by_id.get(row["source_video_id"])
        if candidate is None:
            raise ValueError(f"window source is not in full video manifest: {row['source_video_id']}")
        start = int(row["frame_start"])
        end = int(row["frame_end"])
        decoded = _decode_samples(
            Path(candidate["video_path"]),
            row["source_video_id"],
            tuple(range(start, end + 1)),
        )
        _write_preview_mp4(decoded, output_root / row["preview_path"])
        positions = uniform_frame_indices(len(decoded.frames), 20)
        _write_contact_sheet(
            tuple(decoded.frames[position] for position in positions),
            output_root / row["contact_sheet_path"],
        )


    if set(seen) != set(by_id):
        raise ValueError("full video review must contain exactly one row for every candidate")
