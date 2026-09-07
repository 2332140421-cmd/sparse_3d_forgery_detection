"""Build a bounded, metadata-first GenVidBench Core Domain pilot manifest."""
from __future__ import annotations

import argparse
import csv
from collections import Counter
from dataclasses import asdict
import json
from pathlib import Path

from .metadata import (
    TARGET_GENERATORS,
    build_pilot_selection,
    metadata_summary,
    parse_label_file,
    parse_semantic_file,
    sha256_file,
    write_json,
)

OFFICIAL = {
    "official_repo": "genvidbench/GenVidBench",
    "official_repo_revision": "de027df (web-visible main commit; network clone unavailable)",
    "hf_dataset": "jian-0/GenVidBench",
    "hf_revision": "701cafb6f999d7ea0cbf3c354df6177311a4d824",
    "release_license": "CC BY-NC 4.0 (official README)",
    "upstream_license_detail": "LICENSE_DETAIL_UNRESOLVED",
}

REVIEW_FIELDS = (
    "source_id", "role", "real_source", "video_path", "duration_s",
    "fps_or_timestamp_info", "semantic_object", "semantic_action",
    "semantic_location", "video_decision", "video_reason_code", "notes",
)

def load_metadata(metadata_root: Path) -> dict[str, object]:
    classes = metadata_root / "classes_list.txt"
    pair1 = parse_label_file(metadata_root / "Pair1_labels.txt")
    pair2 = parse_label_file(metadata_root / "Pair2_labels.txt")
    vript = parse_semantic_file(metadata_root / "Vript_20k_classes.txt", "Vript", classes)
    hdvg = parse_semantic_file(metadata_root / "HDVG_14k_classes.txt", "HD-VG-130M", classes)
    vidprom = parse_semantic_file(metadata_root / "VidProM_13k_classes.txt", "VidProM", classes)
    selection = build_pilot_selection(
        pair1_labels=pair1, pair2_labels=pair2,
        vript_records=vript, hdvg_records=hdvg,
        train_limit=32, test_limit=16,
    )
    return {"classes": classes, "pair1": pair1, "pair2": pair2,
            "vript": vript, "hdvg": hdvg, "vidprom": vidprom,
            "selection": selection}

def _review_rows(selection: dict[str, object]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for key in ("train_real", "test_real_source"):
        for item in selection[key]:
            rows.append({
                "source_id": str(item["source_id"]),
                "role": str(item["role"]),
                "real_source": str(item["real_source"]),
                "video_path": str(item["relative_path"]),
                "duration_s": "",
                "fps_or_timestamp_info": "30 fps declared by official paper; local media not materialized",
                "semantic_object": str(item["semantic_object"]),
                "semantic_action": str(item["semantic_action"]),
                "semantic_location": str(item["semantic_location"]),
                "video_decision": "",
                "video_reason_code": "",
                "notes": "MEDIA_NOT_MATERIALIZED; human review requires original RGB",
            })
    return rows

def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=REVIEW_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

def run(metadata_root: Path, output_root: Path) -> dict[str, object]:
    loaded = load_metadata(metadata_root)
    pair1, pair2 = loaded["pair1"], loaded["pair2"]
    vript, hdvg, vidprom = loaded["vript"], loaded["hdvg"], loaded["vidprom"]
    selection = loaded["selection"]
    output_root.mkdir(parents=True, exist_ok=True)
    file_sha = {p.name: sha256_file(p) for p in sorted(metadata_root.iterdir()) if p.is_file()}
    source_audit = {
        **OFFICIAL,
        "metadata_root": str(metadata_root),
        "metadata_file_sha256": file_sha,
        "counts": metadata_summary(pair1_labels=pair1, pair2_labels=pair2, vript=vript, hdvg=hdvg, vidprom=vidprom),
        "semantic_distributions": {
            "Vript_object": dict(Counter(r.object_label for r in vript)),
            "Vript_action": dict(Counter(r.action_label for r in vript)),
            "Vript_location": dict(Counter(r.location_label for r in vript)),
            "HD_VG_130M_object": dict(Counter(r.object_label for r in hdvg)),
            "HD_VG_130M_action": dict(Counter(r.action_label for r in hdvg)),
            "HD_VG_130M_location": dict(Counter(r.location_label for r in hdvg)),
        },
        "selection_source": "official semantic metadata plus real label rows only",
        "fake_informed": False,
        "model_output_informed": False,
    }
    write_json(output_root / "metadata" / "source_audit.json", source_audit)
    write_json(output_root / "metadata" / "core_candidate_rules.json", {
        "rule": "object_label == Vehicles and action_label != Static Postures",
        "taxonomy_source": "classes_list.txt",
        "manual_review_required_for": ["Static Postures", "non-Vehicles"],
        "selection_does_not_use": ["fake labels", "generator identity", "model output", "AUC", "motion features"],
    })
    write_json(output_root / "manifests" / "train_real_pilot.json", selection["train_real"])
    write_json(output_root / "manifests" / "test_real_source_pilot.json", selection["test_real_source"])
    write_json(output_root / "paired_test" / "paired_test_manifest.json", selection["test_fake_paired"])
    review_rows = _review_rows(selection)
    _write_csv(output_root / "real_review" / "review_core.csv", review_rows)
    pair_counts = Counter(row["generator"] for row in selection["test_fake_paired"])
    summary = {
        **OFFICIAL,
        "status": "GENVIDBENCH_DATA_ACCESS_BLOCKED",
        "media_status": "GENVIDBENCH_MEDIA_ARCHIVE_ONLY",
        "media_note": "Official HF repository exposes large archive parts; no individual local videos were materialized in this metadata-first phase.",
        "pilot_requested": {"train_real": 32, "test_real_source": 16},
        "pilot_selected": {"train_real": len(selection["train_real"]), "test_real_source": len(selection["test_real_source"]), "paired_fake": len(selection["test_fake_paired"])},
        "paired_fake_by_generator": dict(pair_counts),
        "review_rows": len(review_rows),
        "decode_success": 0,
        "decode_failure": 0,
        "contact_sheets": [],
        "finalizer_eligible_rows": 0,
        "causal_training_eligible": False,
        "reason": "Human RGB video review and decode cannot start until a bounded official media subset is made available.",
    }
    write_json(output_root / "summary" / "pilot_summary.json", summary)
    return summary

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.metadata_root, args.output_root), indent=2, sort_keys=True))

if __name__ == "__main__":
    main()
