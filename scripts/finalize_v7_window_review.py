#!/usr/bin/env python3
"""Finalize manually reviewed V7 temporal-window adequacy decisions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from sparse3d_forgery.experiments.v7_core_domain_timescale import finalize_window_review


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--window-review-csv", type=Path, required=True)
    parser.add_argument("--video-core-domain-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = finalize_window_review(
        args.window_review_csv,
        args.video_core_domain_manifest,
        args.output,
    )
    print(json.dumps({"window_count": len(result["windows"]), "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
