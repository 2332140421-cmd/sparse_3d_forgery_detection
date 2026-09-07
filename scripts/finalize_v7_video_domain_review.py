#!/usr/bin/env python3
"""Finalize manually reviewed V7 video-domain decisions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from sparse3d_forgery.experiments.v7_core_domain_timescale import (
    finalize_video_domain_review,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--full-video-review-csv", type=Path, required=True)
    parser.add_argument("--full-video-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = finalize_video_domain_review(
        args.full_video_review_csv,
        args.full_video_manifest,
        args.output,
    )
    print(json.dumps({"selected_count": len(result["selected"]), "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
