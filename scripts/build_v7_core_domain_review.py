#!/usr/bin/env python3
"""Build deterministic V7 Core Domain human-review materials."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from sparse3d_forgery.experiments.v7_core_domain import (
    build_review_materials,
    load_core_domain_candidates,
)


DEFAULT_DATA_ROOT = Path("/root/autodl-tmp/data/sparse_3d_forgery_detection")
DEFAULT_REVISION = "92e76e78e8c90a1ff7ec9354bee44eb024265e79"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_DATA_ROOT / "derived/temporal_learnability_probe_v1/pilot_manifest.json",
    )
    parser.add_argument(
        "--extracted-root",
        type=Path,
        default=DEFAULT_DATA_ROOT / "extracted/deeptrace_reward" / DEFAULT_REVISION,
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_DATA_ROOT / "derived/v7_core_domain_review_v1",
    )
    args = parser.parse_args()
    candidates = load_core_domain_candidates(args.manifest)
    summary = build_review_materials(
        candidates,
        extracted_root=args.extracted_root,
        output_root=args.output_root,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
