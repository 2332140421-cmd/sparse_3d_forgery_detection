"""Validate downloaded candidate media with PyAV only.

This is a bounded container/decode check for data preparation.  It does not
run tracking, depth, pose, scene-cut detection, or any V7 model.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path
from typing import Any, Iterable

import av


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def _stream_rate(stream: Any) -> float | None:
    rate = stream.average_rate or stream.base_rate
    if rate is None:
        return None
    try:
        value = float(rate)
    except (TypeError, ValueError, ZeroDivisionError):
        return None
    return value if math.isfinite(value) else None


def validate_video(path: Path, *, source_kind: str, identity: str) -> dict[str, Any]:
    started = time.monotonic()
    result: dict[str, Any] = {
        "identity": identity,
        "source_kind": source_kind,
        "path": str(path),
        "exists": path.exists(),
        "nonzero": path.is_file() and path.stat().st_size > 0 if path.exists() else False,
        "status": "MEDIA_MISSING",
        "frame_count": 0,
        "duration_s": None,
        "width": None,
        "height": None,
        "fps": None,
        "pts_first_s": None,
        "pts_last_s": None,
        "pts_finite": False,
        "pts_monotonic": False,
        "rgb_frames_checked": 0,
        "error": None,
        "elapsed_s": None,
    }
    if not result["exists"] or not result["nonzero"]:
        result["elapsed_s"] = time.monotonic() - started
        return result
    previous_pts: float | None = None
    pts_finite = True
    pts_monotonic = True
    try:
        with av.open(str(path)) as container:
            streams = [stream for stream in container.streams if stream.type == "video"]
            if not streams:
                raise ValueError("no video stream")
            stream = streams[0]
            result["width"] = int(stream.width or 0)
            result["height"] = int(stream.height or 0)
            result["fps"] = _stream_rate(stream)
            if stream.duration is not None and stream.time_base is not None:
                result["duration_s"] = float(stream.duration * stream.time_base)
            for frame in container.decode(stream):
                result["frame_count"] += 1
                if frame.pts is None or frame.time_base is None:
                    pts_finite = False
                else:
                    timestamp = float(frame.pts * frame.time_base)
                    if not math.isfinite(timestamp):
                        pts_finite = False
                    else:
                        if result["pts_first_s"] is None:
                            result["pts_first_s"] = timestamp
                        if previous_pts is not None and timestamp < previous_pts:
                            pts_monotonic = False
                        previous_pts = timestamp
                        result["pts_last_s"] = timestamp
                frame.to_ndarray(format="rgb24")
                result["rgb_frames_checked"] += 1
            if result["frame_count"] == 0:
                raise ValueError("video stream decoded zero frames")
            result["pts_finite"] = pts_finite
            result["pts_monotonic"] = pts_monotonic
            if not pts_finite:
                raise ValueError("missing or non-finite frame PTS")
            if not pts_monotonic:
                raise ValueError("frame PTS is not monotonic")
            if result["width"] <= 0 or result["height"] <= 0:
                raise ValueError("invalid video dimensions")
            result["status"] = "MEDIA_VALID"
    except Exception as exc:  # noqa: BLE001 - classify a media item, never hide the path
        result["error"] = f"{type(exc).__name__}: {exc}"
        result["status"] = "MEDIA_DECODE_FAILURE"
    result["elapsed_s"] = time.monotonic() - started
    return result


def _activity_paths(base: Path, mapping: Iterable[dict[str, Any]]) -> list[tuple[str, Path, str]]:
    return [("activityforensics", base / "source/activityforensics/raw" / row["activityforensics_file"], row["activityforensics_file"]) for row in mapping]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--mapping", type=Path)
    parser.add_argument("--only-exact", action="store_true")
    args = parser.parse_args()
    base: Path = args.base
    mapping_path = args.mapping or base / "paired/manifests/activityforensics_source_mapping.json"
    mapping = json.loads(mapping_path.read_text(encoding="utf-8"))
    rows: list[tuple[str, Path, str]] = []
    for row in mapping:
        if args.only_exact and row["lineage_status"] != "EXACT":
            continue
        rows.extend(_activity_paths(base, [row]))
    for source_id in sorted({row["charades_source_id"] for row in mapping if row["lineage_status"] == "EXACT" and row["charades_source_id"]}):
        rows.append(("charades", base / "source/charades/videos" / f"{source_id}.mp4", source_id))
    results = [validate_video(path, source_kind=kind, identity=identity) for kind, path, identity in rows]
    write_json(base / "validation/media_validation.json", results)
    with (base / "validation/media_validation.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = sorted({key for result in results for key in result})
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(results)
    counts: dict[str, int] = {}
    for result in results:
        counts[result["status"]] = counts.get(result["status"], 0) + 1
    print(json.dumps({"count": len(results), "status_counts": counts}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
