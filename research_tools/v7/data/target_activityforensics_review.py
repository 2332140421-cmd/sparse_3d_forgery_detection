"""Freeze a deterministic 100-primary/30-reserve exact-lineage population."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def _cells(row: dict[str, Any]) -> set[tuple[str, str]]:
    generator = str(row["generator"])
    operations = row["manipulation_operation"]
    if isinstance(operations, list):
        values = [str(value) for value in operations]
    else:
        values = [str(operations)]
    return {(generator, operation) for operation in values}


def build_candidate_sources(mapping: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in mapping:
        if row["lineage_status"] == "EXACT" and row["charades_source_id"]:
            groups[str(row["charades_source_id"])].append(row)
    result = []
    for source_id in sorted(groups):
        variants = sorted(groups[source_id], key=lambda row: (str(row["generator"]), str(row["manipulation_operation"]), str(row["activityforensics_file"])))
        result.append({"charades_source_id": source_id, "fake_variants": variants, "available_cells": sorted([list(cell) for row in variants for cell in _cells(row)])})
    return result


def _round_robin(sources: list[dict[str, Any]], count: int) -> list[dict[str, Any]]:
    by_source = {row["charades_source_id"]: row for row in sources}
    cells = sorted({tuple(cell) for row in sources for cell in row["available_cells"]})
    remaining = set(by_source)
    selected: list[dict[str, Any]] = []
    cell_index = 0
    while remaining and len(selected) < count:
        if not cells:
            break
        cell = cells[cell_index % len(cells)]
        cell_index += 1
        candidates = sorted(source for source in remaining if list(cell) in by_source[source]["available_cells"])
        if not candidates:
            if cell_index >= len(cells) and not any(list(other) in by_source[source]["available_cells"] for source in remaining for other in cells):
                break
            continue
        source_id = candidates[0]
        row = by_source[source_id]
        variants = sorted(row["fake_variants"], key=lambda item: (str(item["generator"]), str(item["manipulation_operation"]), str(item["activityforensics_file"])))
        matching = [item for item in variants if cell in _cells(item)]
        selected.append({"charades_source_id": source_id, "selected_cell": list(cell), "fake_variant": matching[0]})
        remaining.remove(source_id)
    return selected


def freeze_selection(mapping_path: Path, output_dir: Path, *, primary_count: int = 100, reserve_count: int = 30) -> dict[str, Any]:
    mapping = json.loads(mapping_path.read_text(encoding="utf-8"))
    candidates = build_candidate_sources(mapping)
    primary = _round_robin(candidates, primary_count)
    chosen = {row["charades_source_id"] for row in primary}
    reserve_candidates = [row for row in candidates if row["charades_source_id"] not in chosen]
    reserve = _round_robin(reserve_candidates, reserve_count)
    algorithm = "sort source IDs; round-robin sorted (generator, manipulation_operation) cells; one source and one fake variant per primary/reserve row"
    write_json(output_dir / "candidate_sources.json", {"algorithm": algorithm, "candidate_source_count": len(candidates), "sources": candidates})
    write_json(output_dir / "selected_sources.json", {"algorithm": algorithm, "target_count": primary_count, "sources": primary})
    write_json(output_dir / "reserve_sources.json", {"algorithm": algorithm, "target_count": reserve_count, "sources": reserve})
    return {"candidate_sources": len(candidates), "primary": len(primary), "reserve": len(reserve), "algorithm": algorithm}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    args = parser.parse_args()
    result = freeze_selection(args.base / "paired/manifests/activityforensics_source_mapping.json", args.base / "acquisition/targeted_review_v1")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
