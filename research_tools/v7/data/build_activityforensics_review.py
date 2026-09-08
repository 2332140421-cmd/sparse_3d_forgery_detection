"""Create deterministic symlinked human-review material for exact pairs."""

from __future__ import annotations

import argparse
import csv
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any


HUMAN_FIELDS = (
    "continuous_single_shot",
    "scene_cut_or_transition",
    "camera_motion",
    "dominant_dynamic_subject",
    "subject_persistent",
    "nontrivial_subject_motion",
    "nontrivial_structural_change",
    "strong_occlusion",
    "multi_subject_complex",
    "realistic_physical_scene",
    "fake_looks_realistic",
    "fake_has_obvious_game_or_cg_style",
    "manipulation_affects_dynamic_structure",
    "notes",
    "V7_CORE_DECISION",
)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def select_review_sources(mapping: list[dict[str, Any]], media: list[dict[str, Any]], limit: int = 100) -> list[str]:
    media_by_path = {row["path"]: row for row in media}
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in mapping:
        if row["lineage_status"] != "EXACT" or not row["charades_source_id"]:
            continue
        path = row["activityforensics_file"]
        activity_status = media_by_path.get(str(Path(row.get("activity_path", ""))), {}).get("status")
        if activity_status is not None and activity_status != "MEDIA_VALID":
            continue
        groups[row["charades_source_id"]].append(row)
    keys = sorted({(str(row["generator"]), str(row["manipulation_operation"])) for rows in groups.values() for row in rows})
    selected: list[str] = []
    remaining = set(groups)
    while remaining and len(selected) < limit:
        progressed = False
        for key in keys:
            candidates = sorted(source for source in remaining if any((str(row["generator"]), str(row["manipulation_operation"])) == key for row in groups[source]))
            if candidates:
                selected.append(candidates[0])
                remaining.remove(candidates[0])
                progressed = True
                break
        if not progressed:
            selected.extend(sorted(remaining)[: limit - len(selected)])
            break
    return selected


def _blank_human_fields() -> dict[str, str]:
    return {field: "" for field in HUMAN_FIELDS}


def _relative_symlink(target: Path, link: Path) -> None:
    link.parent.mkdir(parents=True, exist_ok=True)
    link.symlink_to(os.path.relpath(target, link.parent))


def build_review(base: Path, *, limit: int = 100) -> dict[str, Any]:
    mapping = json.loads((base / "paired/manifests/activityforensics_source_mapping.json").read_text(encoding="utf-8"))
    media = json.loads((base / "validation/media_validation.json").read_text(encoding="utf-8"))
    valid = {row["path"]: row for row in media if row["status"] == "MEDIA_VALID"}
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in mapping:
        fake_path = base / "source/activityforensics/raw" / row["activityforensics_file"]
        real_path = base / "source/charades/videos" / f"{row['charades_source_id']}.mp4"
        if row["lineage_status"] == "EXACT" and str(fake_path) in valid and str(real_path) in valid:
            enriched = dict(row)
            enriched.update({"activity_path": str(fake_path), "real_path": str(real_path)})
            groups[row["charades_source_id"]].append(enriched)
    selection_rows = [
        {**row, "activity_path": item["activity_path"], "real_path": item["real_path"]}
        for source_rows in groups.values()
        for item in source_rows
    ]
    selected_sources = select_review_sources(selection_rows, media, limit)
    review_root = base / "review/batch_001"
    manifest_rows: list[dict[str, Any]] = []
    for ordinal, source_id in enumerate(selected_sources, 1):
        candidates = sorted(groups[source_id], key=lambda row: (str(row["generator"]), str(row["manipulation_operation"]), row["activityforensics_file"]))[:2]
        unit_dir = review_root / f"{ordinal:04d}"
        real_path = Path(candidates[0]["real_path"])
        _relative_symlink(real_path, unit_dir / f"REAL__{source_id}.mp4")
        fake_names: list[str] = []
        for fake_index, row in enumerate(candidates):
            fake_name = f"FAKE_{chr(65 + fake_index)}__{Path(row['activityforensics_file']).name}"
            fake_names.append(fake_name)
            _relative_symlink(Path(row["activity_path"]), unit_dir / fake_name)
            fake_media = valid[str(row["activity_path"])]
            real_media = valid[str(real_path)]
            segments = row["manipulation_segments"]
            first = segments[0]
            manifest_rows.append(
                {
                    "review_id": f"{ordinal:04d}-{chr(65 + fake_index)}",
                    "charades_source_id": source_id,
                    "real_video_path": str(real_path),
                    "fake_video_path": row["activity_path"],
                    "generator": row["generator"],
                    "manipulation_operation": row["manipulation_operation"],
                    "manipulation_start_s": first["start_s"],
                    "manipulation_end_s": first["end_s"],
                    "multiple_manipulations": len(row["manipulation_intervals"]) > 1,
                    "real_duration_s": real_media.get("duration_s"),
                    "fake_duration_s": fake_media.get("duration_s"),
                    "real_resolution": f"{real_media.get('width')}x{real_media.get('height')}",
                    "fake_resolution": f"{fake_media.get('width')}x{fake_media.get('height')}",
                    "lineage_status": row["lineage_status"],
                    "media_status_real": real_media["status"],
                    "media_status_fake": fake_media["status"],
                    **_blank_human_fields(),
                }
            )
        write_json(unit_dir / "INFO.json", {"charades_source_id": source_id, "real_video_path": str(real_path), "fake_video_names": fake_names, "fake_count": len(candidates), "human_review_required": True})
    fields = ["review_id", "charades_source_id", "real_video_path", "fake_video_path", "generator", "manipulation_operation", "manipulation_start_s", "manipulation_end_s", "multiple_manipulations", "real_duration_s", "fake_duration_s", "real_resolution", "fake_resolution", "lineage_status", "media_status_real", "media_status_fake", *HUMAN_FIELDS]
    manifest_csv = base / "review/review_manifest.csv"
    manifest_csv.parent.mkdir(parents=True, exist_ok=True)
    with manifest_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(manifest_rows)
    write_json(base / "review/review_manifest.json", manifest_rows)
    readme = """# REVIEW_BATCH_001\n\nThis is a deterministic human-review package, not an automatic V7 decision. Open the original MP4 symlink and each fake MP4 symlink in every numbered directory. The `review_manifest.csv` and `INFO.json` files record provenance and media checks only.\n\n## Human fields\n\n- `continuous_single_shot`: YES only when the viewed interval is one continuous shot without a hard cut, transition, or time jump.\n- `scene_cut_or_transition`: record YES/NO and explain any uncertain case.\n- `camera_motion`: `STATIC`, `WEAK_SMOOTH`, or `DOMINANT`.\n- `dominant_dynamic_subject`: whether the main moving subject is clearly visible.\n- `subject_persistent`: whether the subject remains trackable through the interval.\n- `nontrivial_subject_motion`: motion is not only a static subject plus camera motion.\n- `nontrivial_structural_change`: visible change in body/object parts, pose, or relative structure.\n- `strong_occlusion`: record whether occlusion prevents a reliable judgment.\n- `multi_subject_complex`: whether multiple interacting subjects make the case complex.\n- `realistic_physical_scene`: real-world capture, not game, anime, obvious CGI, or synthetic render.\n- `fake_looks_realistic`: fake remains in the realistic-video-forgery domain.\n- `fake_has_obvious_game_or_cg_style`: record obvious synthetic style separately.\n- `manipulation_affects_dynamic_structure`: whether the edit appears to affect action, structure, interaction, or temporal process rather than only static texture/color/style.\n- `notes`: concise evidence for the human entries.\n- `V7_CORE_DECISION`: `ACCEPT`, `REJECT`, or `UNCERTAIN`; this must be filled by the human reviewer.\n\nDo not infer a V7 decision from filenames, generator, metadata, or this package. No model, frontend, tracking, depth, pose, normality score, or AUROC was run to create it.\n"""
    (base / "review/README.md").write_text(readme, encoding="utf-8")
    return {"unique_sources": len(selected_sources), "fake_videos": len(manifest_rows), "manifest_csv": str(manifest_csv), "review_root": str(review_root)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=100)
    args = parser.parse_args()
    print(json.dumps(build_review(args.base, limit=args.limit), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
