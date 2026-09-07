"""Resolve and materialize only frozen Vript target members.

This module is deliberately outside ``src/``.  It is an experiment adapter for
the V7 fast path, not a general dataset framework.  It uses exact target
identity and exact ZIP members; it never substitutes a similar filename.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import re
import struct
import subprocess
from typing import Iterable, Mapping, Sequence
import urllib.request
import zlib


REVISION = "acc278efb0ee249d646eef8a6b023595ca7efb93"
REPO_ID = "Mutonix/Vript"
SHARD_ROOT = "vript_long_videos_clips"
REVIEW_FIELDS = (
    "source_id",
    "video_path",
    "semantic_object",
    "semantic_action",
    "semantic_location",
    "video_decision",
    "video_reason_code",
    "notes",
)
_SCENE_RE = re.compile(r"^(?P<video_id>.+)-Scene-(?P<scene_id>[0-9]+)\.mp4$")
_SHARD_RE = re.compile(r"clips_(?P<number>[0-9]+)_of_1095(?:\.zip)?$")


def read_json(path: Path) -> object:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def frozen_targets(manifest_path: Path) -> list[dict[str, object]]:
    """Read the already frozen train-real population without re-sampling it."""

    rows = read_json(manifest_path)
    if not isinstance(rows, list):
        raise ValueError("frozen train manifest must be a list")
    result: list[dict[str, object]] = []
    seen: set[str] = set()
    for row in rows:
        if not isinstance(row, dict) or row.get("role") != "train_real":
            continue
        source_id = str(row["source_id"])
        if source_id in seen:
            raise ValueError(f"duplicate frozen source_id: {source_id}")
        seen.add(source_id)
        result.append(dict(row))
    return result


def parse_target_filename(filename: str) -> tuple[str, str]:
    match = _SCENE_RE.fullmatch(filename)
    if match is None:
        raise ValueError(f"not an exact Vript scene filename: {filename}")
    return match.group("video_id"), match.group("scene_id")


def _archive_shard_number(archive_member: str) -> str | None:
    for part in archive_member.split("/"):
        match = _SHARD_RE.fullmatch(part)
        if match:
            return match.group("number")
    return None


def _source_filename(row: Mapping[str, object]) -> str:
    source_identity = row.get("source_identity")
    if isinstance(source_identity, str) and source_identity:
        return f"{source_identity}.mp4"
    relative = str(row.get("relative_path", ""))
    return Path(relative).name


def build_target_map(
    frozen_rows: Sequence[Mapping[str, object]],
    prior_archive_rows: Sequence[Mapping[str, object]],
    official_shards: Mapping[str, Mapping[str, object]],
) -> list[dict[str, object]]:
    """Map frozen identities to exact Mutonix shard/member names.

    ``prior_archive_rows`` contains the previously resolved official
    GenVidBench-to-Vript identity map.  We use only its exact shard number and
    filename, never fuzzy matching or a new population selection.
    """

    prior_by_id = {str(row["source_id"]): row for row in prior_archive_rows}
    result: list[dict[str, object]] = []
    for row in frozen_rows:
        source_id = str(row["source_id"])
        filename = _source_filename(row)
        video_id, scene_id = parse_target_filename(filename)
        prior = prior_by_id.get(source_id)
        number = _archive_shard_number(str(prior.get("archive_member", ""))) if prior else None
        shard = f"{SHARD_ROOT}/clips_{number}_of_1095.zip" if number is not None else None
        member = f"clips_{number}_of_1095/{video_id}/{filename}" if number is not None else None
        status = "RESOLVED" if shard in official_shards else "UNRESOLVED"
        result.append(
            {
                "source_id": source_id,
                "target_filename": filename,
                "video_id": video_id,
                "scene_id": scene_id,
                "shard": shard,
                "member": member,
                "status": status,
                "official_revision": REVISION,
                "shard_bytes": official_shards.get(shard, {}).get("size") if shard else None,
            }
        )
    return result


def first_valid_targets(rows: Sequence[Mapping[str, object]], media_status: Mapping[str, str], count: int = 8) -> list[dict[str, object]]:
    """Return the first valid rows in frozen order, skipping only explicit failures."""

    selected: list[dict[str, object]] = []
    for row in rows:
        if media_status.get(str(row["source_id"])) == "MEDIA_VALID":
            selected.append(dict(row))
            if len(selected) == count:
                break
    return selected


def safe_output_name(source_id: str, filename: str) -> str:
    digest = hashlib.sha1(source_id.encode("utf-8")).hexdigest()[:10]
    return f"{digest}__{filename}"


class RemoteZipRange:
    """Read a remote ZIP central directory and one member by HTTP ranges."""

    def __init__(self, url: str, *, timeout: float = 120.0):
        self.url = url
        self.timeout = timeout
        self.size: int | None = None
        self.etag: str | None = None

    def _request(self, headers: Mapping[str, str] | None = None) -> tuple[bytes, Mapping[str, str]]:
        request_headers = dict(headers or {})
        try:
            request = urllib.request.Request(self.url, headers=request_headers)
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                return response.read(), dict(response.headers.items())
        except OSError:
            curl_headers = []
            for key, value in request_headers.items():
                curl_headers.extend(["-H", f"{key}: {value}"])
            completed = subprocess.run(
                ["curl", "-L", "--fail", "--retry", "3", "--retry-delay", "2", "--max-time", str(int(self.timeout)), *curl_headers, self.url],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            return completed.stdout, {}

    def head(self) -> dict[str, object]:
        try:
            request = urllib.request.Request(self.url, method="HEAD")
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                headers = dict(response.headers.items())
        except OSError:
            completed = subprocess.run(
                ["curl", "-L", "--fail", "--retry", "3", "--retry-delay", "2", "--max-time", str(int(self.timeout)), "-I", self.url],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            headers = {}
            for line in completed.stdout.splitlines():
                if ":" in line:
                    key, value = line.split(":", 1)
                    headers[key.strip().lower()] = value.strip()
        length = headers.get("Content-Length") or headers.get("content-length")
        self.size = int(length) if length is not None else None
        self.etag = headers.get("ETag") or headers.get("etag")
        return {"bytes": self.size, "etag": self.etag, "url": self.url}

    def range(self, start: int, end: int) -> bytes:
        if start < 0 or end < start:
            raise ValueError("invalid HTTP range")
        body, headers = self._request({"Range": f"bytes={start}-{end}"})
        expected = end - start + 1
        content_range = headers.get("Content-Range", "")
        if len(body) != expected:
            raise RuntimeError(f"range length mismatch: wanted {expected}, got {len(body)} ({content_range})")
        return body

    def central_directory(self) -> tuple[list[dict[str, object]], dict[str, object]]:
        info = self.head()
        if self.size is None:
            raise RuntimeError("remote ZIP has no Content-Length")
        tail_size = min(self.size, 4 * 1024 * 1024)
        tail_start = self.size - tail_size
        tail = self.range(tail_start, self.size - 1)
        end = tail.rfind(b"PK\x05\x06")
        if end < 0:
            raise RuntimeError("ZIP end-of-central-directory not found")
        _, disk, cd_disk, disk_entries, entries, cd_size, cd_offset, comment_len = struct.unpack_from(
            "<4s4H2LH", tail, end
        )
        if any(value == 0xFFFF for value in (disk_entries, entries)) or cd_size == 0xFFFFFFFF or cd_offset == 0xFFFFFFFF:
            raise RuntimeError("ZIP64 central directory is not supported by this bounded adapter")
        central = self.range(cd_offset, cd_offset + cd_size - 1)
        parsed: list[dict[str, object]] = []
        offset = 0
        while offset < len(central):
            if central[offset : offset + 4] != b"PK\x01\x02":
                raise RuntimeError("invalid ZIP central-directory signature")
            values = struct.unpack_from("<4s6H3I5H2I", central, offset)
            _, version_made, version_needed, flags, method, mtime, mdate, crc, compressed, uncompressed, name_len, extra_len, comment_len, disk_start, int_attr, ext_attr, local_offset = values
            start = offset + 46
            name = central[start : start + name_len].decode("utf-8")
            parsed.append(
                {
                    "name": name,
                    "flags": flags,
                    "method": method,
                    "crc32": crc,
                    "compressed_size": compressed,
                    "uncompressed_size": uncompressed,
                    "local_offset": local_offset,
                }
            )
            offset = start + name_len + extra_len + comment_len
        return parsed, {"archive": info, "entry_count": entries, "central_directory_bytes": cd_size}

    def read_member(self, entry: Mapping[str, object]) -> bytes:
        offset = int(entry["local_offset"])
        local = self.range(offset, offset + 30 - 1)
        if local[:4] != b"PK\x03\x04":
            raise RuntimeError("invalid ZIP local-header signature")
        _, version, flags, method, mtime, mdate, crc, compressed, uncompressed, name_len, extra_len = struct.unpack_from(
            "<4s5H3I2H", local
        )
        data_start = offset + 30 + name_len + extra_len
        data_end = data_start + int(entry["compressed_size"]) - 1
        payload = self.range(data_start, data_end) if data_end >= data_start else b""
        if int(entry["method"]) == 0:
            data = payload
        elif int(entry["method"]) == 8:
            data = zlib.decompress(payload, -15)
        else:
            raise RuntimeError(f"unsupported ZIP compression method: {entry['method']}")
        if len(data) != int(entry["uncompressed_size"]):
            raise RuntimeError("ZIP member size mismatch")
        if (zlib.crc32(data) & 0xFFFFFFFF) != int(entry["crc32"]):
            raise RuntimeError("ZIP member CRC mismatch")
        return data


def exact_member(entries: Iterable[Mapping[str, object]], *, filename: str, video_id: str) -> Mapping[str, object] | None:
    candidates = [
        entry
        for entry in entries
        if str(entry["name"]).rsplit("/", 1)[-1] == filename
        and video_id in str(entry["name"]).split("/")
    ]
    if len(candidates) != 1:
        return None
    return candidates[0]


def materialize_member(remote: RemoteZipRange, row: Mapping[str, object], output_dir: Path) -> dict[str, object]:
    filename = str(row["target_filename"])
    video_id = str(row["video_id"])
    entries, archive_info = remote.central_directory()
    entry = exact_member(entries, filename=filename, video_id=video_id)
    if entry is None:
        return {**row, "status": "MEMBER_MISSING", "archive": archive_info}
    data = remote.read_member(entry)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / safe_output_name(str(row["source_id"]), filename)
    partial = path.with_suffix(path.suffix + ".part")
    partial.write_bytes(data)
    partial.replace(path)
    digest = hashlib.sha256(data).hexdigest()
    return {
        **row,
        "status": "MATERIALIZED",
        "member": entry["name"],
        "video_path": str(path),
        "video_bytes": len(data),
        "video_sha256": digest,
        "archive": archive_info,
    }


def validate_video(path: Path) -> dict[str, object]:
    """Decode every frame and require finite, strictly increasing true PTS."""

    try:
        import av
    except ImportError as exc:  # pragma: no cover - environment guard
        raise RuntimeError("PyAV is required for media validation") from exc
    try:
        container = av.open(str(path))
        stream = next(stream for stream in container.streams if stream.type == "video")
        time_base = float(stream.time_base)
        timestamps: list[float] = []
        width = height = None
        for frame in container.decode(stream):
            if frame.pts is None:
                raise ValueError("frame PTS is missing")
            timestamp = float(frame.pts) * time_base
            if timestamps and timestamp <= timestamps[-1]:
                raise ValueError("frame timestamps are not strictly increasing")
            rgb = frame.to_rgb().to_ndarray()
            if rgb.dtype.name != "uint8" or rgb.ndim != 3 or rgb.shape[-1] != 3:
                raise ValueError("decoded RGB frame does not have uint8 [H,W,3] shape")
            height, width = rgb.shape[:2]
            timestamps.append(timestamp)
        container.close()
        if not timestamps or width is None or height is None:
            raise ValueError("video contains no decodable frames")
        deltas = [b - a for a, b in zip(timestamps, timestamps[1:])]
        fps = (1.0 / (sorted(deltas)[len(deltas) // 2]) if deltas else None)
        return {
            "status": "MEDIA_VALID",
            "video_path": str(path),
            "frames": len(timestamps),
            "width": width,
            "height": height,
            "timestamps_s": timestamps,
            "duration_s": timestamps[-1] - timestamps[0],
            "effective_fps": fps,
            "time_base": time_base,
        }
    except Exception as exc:
        return {"status": "MEDIA_DECODE_FAILURE", "video_path": str(path), "error": f"{type(exc).__name__}: {exc}"}


def contact_sheet(path: Path, output: Path, *, columns: int = 4, rows: int = 5) -> dict[str, object]:
    """Create a 20-frame full-timeline sheet with index and true timestamp labels."""

    import av
    from PIL import Image, ImageDraw, ImageFont

    validation = validate_video(path)
    if validation.get("status") != "MEDIA_VALID":
        raise ValueError("contact sheet requires MEDIA_VALID video")
    count = int(validation["frames"])
    indices = sorted(set(round(i * (count - 1) / (columns * rows - 1)) for i in range(columns * rows)))
    wanted = set(indices)
    frames: dict[int, tuple[object, float]] = {}
    container = av.open(str(path))
    stream = next(stream for stream in container.streams if stream.type == "video")
    for index, frame in enumerate(container.decode(stream)):
        if index in wanted:
            frames[index] = (frame.to_rgb().to_ndarray(), float(frame.pts) * float(stream.time_base))
    container.close()
    if len(frames) != len(indices):
        raise ValueError("contact sheet did not decode all requested positions")
    thumb_w, thumb_h = 320, 200
    sheet = Image.new("RGB", (columns * thumb_w, rows * (thumb_h + 24)), "black")
    draw = ImageDraw.Draw(sheet)
    for slot, index in enumerate(indices):
        image = Image.fromarray(frames[index][0]).convert("RGB")
        image.thumbnail((thumb_w, thumb_h))
        left = (slot % columns) * thumb_w + (thumb_w - image.width) // 2
        top = (slot // columns) * (thumb_h + 24) + (thumb_h - image.height) // 2
        sheet.paste(image, (left, top))
        draw.text(((slot % columns) * thumb_w + 4, (slot // columns) * (thumb_h + 24) + thumb_h + 3), f"frame={index} t={frames[index][1]:.3f}s", fill="white")
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output)
    return {"path": str(output), "indices": indices, "timestamps_s": [frames[i][1] for i in indices]}


def write_review_subset(rows: Sequence[Mapping[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=REVIEW_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "source_id": row["source_id"],
                    "video_path": row.get("video_path", ""),
                    "semantic_object": row.get("semantic_object", ""),
                    "semantic_action": row.get("semantic_action", ""),
                    "semantic_location": row.get("semantic_location", ""),
                    "video_decision": "",
                    "video_reason_code": "",
                    "notes": "",
                }
            )


def build_time_window(timestamps_s: Sequence[float], duration_s: float, *, anchor_fraction: float = 0.5) -> dict[str, object]:
    """Construct a true-time, no-padding midpoint window."""

    values = [float(value) for value in timestamps_s]
    if len(values) < 2 or any(b <= a for a, b in zip(values, values[1:])):
        raise ValueError("timestamps must be finite and strictly increasing")
    if not 0.0 <= anchor_fraction <= 1.0 or duration_s <= 0:
        raise ValueError("invalid anchor or duration")
    start_time, end_time = values[0], values[-1]
    center = start_time + (end_time - start_time) * anchor_fraction
    if end_time - start_time < duration_s:
        return {"status": "WINDOW_DURATION_UNAVAILABLE", "duration_s": duration_s, "anchor_timestamp_s": center, "frame_indices": []}
    left, right = center - duration_s / 2.0, center + duration_s / 2.0
    indices = [index for index, value in enumerate(values) if left <= value <= right]
    if len(indices) < 2:
        return {"status": "WINDOW_DURATION_UNAVAILABLE", "duration_s": duration_s, "anchor_timestamp_s": center, "frame_indices": []}
    return {
        "status": "AVAILABLE",
        "duration_s": duration_s,
        "anchor_timestamp_s": center,
        "left_timestamp_s": left,
        "right_timestamp_s": right,
        "frame_indices": indices,
        "timestamps_s": [values[index] for index in indices],
        "actual_span_s": values[indices[-1]] - values[indices[0]],
    }
