#!/usr/bin/env python3
"""Build future V7 multi-timescale review rows after video-level review."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from sparse3d_forgery.experiments.v7_core_domain_timescale import (
    build_window_review_rows,
    generate_window_review_materials,
    write_window_review_csv,
)


DEFAULT_DATA_ROOT = Path("/root/autodl-tmp/data/sparse_3d_forgery_detection")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--full-video-review-csv",
        type=Path,
        required=True,
        help="human-reviewed video_review/full_video_review.csv",
    )
    parser.add_argument(
        "--full-video-manifest",
        type=Path,
        required=True,
        help="video_review/full_video_manifest.json",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_DATA_ROOT / "derived/v7_core_domain_review_v2/window_review",
    )
    parser.add_argument(
        "--generate-materials",
        action="store_true",
        help="also decode selected windows and write RGB-only previews/sheets",
    )
    args = parser.parse_args()
    rows = build_window_review_rows(
        args.full_video_review_csv,
        args.full_video_manifest,
        output_root=args.output_root,
    )
    output = args.output_root / "window_review.csv"
    write_window_review_csv(rows, output)
    if args.generate_materials:
        generate_window_review_materials(
            rows,
            args.full_video_manifest,
            output_root=args.output_root,
        )
    print(json.dumps({"window_count": len(rows), "output": str(output)}, indent=2))


if __name__ == "__main__":
    main()
