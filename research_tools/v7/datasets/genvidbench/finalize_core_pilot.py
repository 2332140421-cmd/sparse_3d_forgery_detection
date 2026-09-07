"""Finalize only explicitly human-reviewed IN_DOMAIN GenVidBench pilot rows."""
from __future__ import annotations

import argparse
import csv
from hashlib import sha256
import json
from pathlib import Path

from .metadata import write_json

REQUIRED = {"IN_DOMAIN", "OUT_OF_DOMAIN", "UNCERTAIN"}

def file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

def run(output_root: Path) -> dict[str, object]:
    review_path = output_root / "real_review" / "review_core.csv"
    rows = list(csv.DictReader(review_path.open(encoding="utf-8", newline="")))
    if not rows:
        raise ValueError("review_core.csv is empty")
    decisions = {row["video_decision"] for row in rows}
    invalid = decisions - {"", *REQUIRED}
    if invalid:
        raise ValueError(f"invalid video decisions: {sorted(invalid)}")
    if "" in decisions:
        raise ValueError("human review is incomplete; blank video_decision remains")
    # UNCERTAIN is intentionally excluded by the finalizer; it is not a failure.
    selected = {row["source_id"] for row in rows if row["video_decision"] == "IN_DOMAIN"}
    train = json.loads((output_root / "manifests" / "train_real_pilot.json").read_text())
    test = json.loads((output_root / "manifests" / "test_real_source_pilot.json").read_text())
    paired = json.loads((output_root / "paired_test" / "paired_test_manifest.json").read_text())
    train_final = [row for row in train if row["source_id"] in selected and row["role"] == "train_real"]
    test_final = [row for row in test if row["source_id"] in selected and row["role"] == "test_real_source"]
    test_ids = {row["source_id"] for row in test_final}
    paired_final = [row for row in paired if row["source_id"] in test_ids]
    manifest = {
        "status": "CORE_PILOT_HUMAN_REVIEWED",
        "review_csv": str(review_path),
        "review_csv_sha256": file_sha256(review_path),
        "selection_policy": "only explicit IN_DOMAIN; UNCERTAIN excluded",
        "train_real": train_final,
        "test_real_source": test_final,
        "test_fake_paired": paired_final,
        "fake_is_not_used_for_core_selection": True,
        "causal_training_eligible": False,
        "causal_note": "This manifest establishes source review only; no 3D frontend or causal materialization was run.",
    }
    path = output_root / "manifests" / "genvidbench_core_pilot_manifest.json"
    write_json(path, manifest)
    return manifest

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.output_root), indent=2, sort_keys=True))

if __name__ == "__main__":
    main()
