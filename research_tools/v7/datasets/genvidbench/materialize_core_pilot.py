"""Materialize at most 48 local real review sheets when media is available."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from PIL import Image, ImageDraw

from sparse3d_forgery.video_input import VideoSource, decode_video

from .audit_core_pilot import REVIEW_FIELDS
from .metadata import write_json

def _timeline(path: Path) -> tuple[int, list[float]]:
    import av
    timestamps: list[float] = []
    with av.open(str(path), mode="r") as container:
        if not container.streams.video:
            raise RuntimeError("media contains no video stream")
        stream = container.streams.video[0]
        for frame in container.decode(stream):
            if frame.pts is None or frame.time_base is None:
                raise RuntimeError("frame has no PTS time base")
            timestamps.append(float(frame.pts * frame.time_base))
    if not timestamps:
        raise RuntimeError("video has no decodable frames")
    return len(timestamps), timestamps

def _contact_sheet(sample, path: Path) -> None:
    cell_w, cell_h = 320, 220
    sheet = Image.new("RGB", (cell_w * 4, cell_h * 5), "white")
    draw = ImageDraw.Draw(sheet)
    for i, frame in enumerate(sample.frames):
        image = Image.fromarray(frame.rgb, mode="RGB")
        image.thumbnail((cell_w - 8, cell_h - 28))
        x = (i % 4) * cell_w
        y = (i // 4) * cell_h
        sheet.paste(image, (x + (cell_w - image.width) // 2, y + 2))
        draw.text((x + 4, y + cell_h - 22), f"idx={frame.source_frame_index} pts={frame.timestamp_s:.6f}s", fill="black")
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path, format="PNG", optimize=False)

def run(media_root: Path, output_root: Path) -> dict[str, object]:
    review_path = output_root / "real_review" / "review_core.csv"
    rows = list(csv.DictReader(review_path.open(encoding="utf-8", newline="")))
    success = 0
    failures: list[dict[str, str]] = []
    measurements: list[dict[str, object]] = []
    for row in rows:
        path = media_root / row["video_path"]
        if not path.is_file():
            failures.append({"source_id": row["source_id"], "reason": "VIDEO_NOT_FOUND", "path": str(path)})
            continue
        try:
            frame_count, timestamps = _timeline(path)
            positions = [round(i * (frame_count - 1) / 19) for i in range(20)] if frame_count > 1 else [0]
            positions = list(dict.fromkeys(int(i) for i in positions))
            sample = decode_video(VideoSource(row["source_id"], row["source_id"], path), positions)
            sheet = output_root / "real_review" / "contact_sheets" / f"{row['source_id'].replace(':', '_')}.png"
            _contact_sheet(sample, sheet)
            row["duration_s"] = f"{timestamps[-1] - timestamps[0]:.9f}"
            row["fps_or_timestamp_info"] = f"decoded PTS; frame_count={frame_count}"
            row["notes"] = f"contact_sheet={sheet.name}"
            measurements.append({"source_id": row["source_id"], "duration_s": float(row["duration_s"]), "frame_count": frame_count, "resolution": list(sample.frames[0].rgb.shape[:2][::-1])})
            success += 1
        except Exception as exc:
            failures.append({"source_id": row["source_id"], "reason": type(exc).__name__, "detail": str(exc)})
    with review_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=REVIEW_FIELDS)
        writer.writeheader(); writer.writerows(rows)
    summary = {"status": "CORE_PILOT_HUMAN_REVIEW_REQUIRED" if success else "GENVIDBENCH_DATA_ACCESS_BLOCKED", "media_root": str(media_root), "review_rows": len(rows), "decode_success": success, "decode_failure": len(failures), "failures": failures, "measurements": measurements, "contact_sheet_count": success}
    write_json(output_root / "summary" / "materialization_summary.json", summary)
    return summary

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--media-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.media_root, args.output_root), indent=2, sort_keys=True))

if __name__ == "__main__":
    main()
