#!/usr/bin/env python3
"""Finalize a V7 Core Domain manifest from a human-reviewed CSV."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

from sparse3d_forgery.experiments.v7_core_domain import (
    ALLOWED_DECISIONS,
    ALLOWED_REASON_CODES,
    _atomic_json,
    sha256_file,
)


def finalize_core_domain_review(review_root: Path, output_path: Path) -> dict[str, object]:
    template_path = review_root / "review_template.csv"
    manifest_path = review_root / "review_manifest.json"
    with manifest_path.open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    by_id = {row["source_video_id"]: row for row in manifest.get("candidates", [])}

    with template_path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != (
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
        ):
            raise ValueError("review_template.csv columns do not match the review schema")
        rows = list(reader)

    selected = []
    seen: set[str] = set()
    for row in rows:
        source_video_id = row["source_video_id"]
        if source_video_id in seen:
            raise ValueError(f"duplicate review row: {source_video_id}")
        seen.add(source_video_id)
        decision = row["decision"].strip()
        reason = row["reason_code"].strip()
        if decision not in ALLOWED_DECISIONS:
            raise ValueError(f"invalid decision for {source_video_id}: {decision}")
        if reason not in ALLOWED_REASON_CODES:
            raise ValueError(f"invalid reason_code for {source_video_id}: {reason}")
        if decision == "IN_DOMAIN":
            if reason != "SINGLE_STRUCTURED_MOTION":
                raise ValueError("IN_DOMAIN requires SINGLE_STRUCTURED_MOTION")
            source = by_id.get(source_video_id)
            if source is None or source.get("status") != "success":
                raise ValueError(f"missing successful review material: {source_video_id}")
            selected.append(
                {
                    "source_video_id": source_video_id,
                    "split": row["split"],
                    "video_path": row["video_path"],
                    "frame_indices": source["frame_indices"],
                    "timestamps_s": source.get("timestamps_s"),
                    "decision": decision,
                    "reason_code": reason,
                    "notes": row["notes"],
                }
            )

    result = {
        "protocol": "V7 Core Domain Phase-2 human review",
        "review_csv_sha256": sha256_file(template_path),
        "uncertain_excluded_by_default": True,
        "selected": selected,
    }
    _atomic_json(output_path, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--review-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = finalize_core_domain_review(args.review_root, args.output)
    print(json.dumps({"selected_count": len(result["selected"]), "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
