#!/usr/bin/env python3
"""Build V7 full-video domain-review materials from the frozen Phase-A set."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from sparse3d_forgery.experiments.v7_core_domain_timescale import (
    build_full_video_review,
    load_phase_a_review_manifest,
)


DEFAULT_DATA_ROOT = Path("/root/autodl-tmp/data/sparse_3d_forgery_detection")
DEFAULT_REVISION = "92e76e78e8c90a1ff7ec9354bee44eb024265e79"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--phase-a-manifest",
        type=Path,
        default=DEFAULT_DATA_ROOT / "derived/v7_core_domain_review_v1/review_manifest.json",
    )
    parser.add_argument(
        "--extracted-root",
        type=Path,
        default=DEFAULT_DATA_ROOT / "extracted/deeptrace_reward" / DEFAULT_REVISION,
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_DATA_ROOT / "derived/v7_core_domain_review_v2",
    )
    parser.add_argument("--sample-count", type=int, default=20)
    args = parser.parse_args()
    candidates = load_phase_a_review_manifest(args.phase_a_manifest)
    summary = build_full_video_review(
        candidates,
        extracted_root=args.extracted_root,
        output_root=args.output_root,
        sample_count=args.sample_count,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
