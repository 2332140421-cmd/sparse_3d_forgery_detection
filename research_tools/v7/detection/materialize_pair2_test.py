"""Materialize the frozen Pair2 media members from official archives.

Only exact manifest paths (with a fixed archive-root prefix removed) are
accepted.  This is a bounded experiment adapter, not a dataset downloader or
general archive framework.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Iterable

from research_tools.v7.feasibility.materialize_vript_frozen_targets import validate_video

from .protocol import (
    DATA_ROOT,
    GENERATORS,
    PAIR2_REVISION,
    PAIR2_ROOT,
    frozen_pair2_population,
    read_json,
    write_json,
)


ARCHIVE_NAMES = {
    "HD-VG-130M": "hd_vg_130m.7z.001",
    "cogvideo": "cogvideo.rar",
    "mora": "mora.rar",
    "musev": "musev.rar",
    "svd": "svd.rar",
}


def _archive_name(relative_path: str) -> str:
    parts = relative_path.split("/")
    if len(parts) < 2 or parts[0] != "Pair2":
        raise ValueError(f"not an exact Pair2 path: {relative_path}")
    category = "HD-VG-130M" if parts[1] == "hd_vg_130m" else parts[1]
    if category not in ARCHIVE_NAMES:
        raise ValueError(f"unsupported Pair2 category: {category}")
    return ARCHIVE_NAMES[category]


def _archive_candidates(relative_path: str) -> set[str]:
    if not relative_path.startswith("Pair2/"):
        raise ValueError(f"unexpected Pair2 path: {relative_path}")
    without_root = relative_path.removeprefix("Pair2/")
    # GenVidBench archives have used either the Pair2/ prefix or the archive
    # directory itself as their root.  These are exact paths, never basename
    # or fuzzy matches.
    return {relative_path, without_root, "./" + relative_path, "./" + without_root}


def list_archive_members(archive: Path) -> list[str]:
    completed = subprocess.run(
        ["bsdtar", "-tf", str(archive)],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return [line.rstrip("\n") for line in completed.stdout.splitlines() if line and not line.endswith("/")]


def resolve_exact_member(members: Iterable[str], relative_path: str) -> str:
    candidates = _archive_candidates(relative_path)
    matches = [member for member in members if member in candidates]
    if len(matches) != 1:
        raise RuntimeError(f"expected exactly one archive member for {relative_path!r}, found {matches!r}")
    return matches[0]


def _safe_name(source_id: str, role: str, generator: str | None, relative_path: str) -> str:
    token = f"{source_id}|{role}|{generator or 'real'}|{relative_path}".encode("utf-8")
    digest = hashlib.sha256(token).hexdigest()[:16]
    return f"{digest}__{Path(relative_path).name}"


def _extract_member(archive: Path, member: str, target: Path) -> None:
    partial = target.with_suffix(target.suffix + ".part")
    partial.parent.mkdir(parents=True, exist_ok=True)
    with partial.open("wb") as handle:
        subprocess.run(
            ["bsdtar", "-xOf", str(archive), member],
            check=True,
            stdout=handle,
            stderr=subprocess.PIPE,
        )
    if partial.stat().st_size == 0:
        partial.unlink()
        raise RuntimeError(f"archive member extracted as an empty file: {member}")
    partial.replace(target)


def _base_row(source: dict[str, object], *, role: str, generator: str | None, relative_path: str, archive: Path, member: str, video_path: Path) -> dict[str, object]:
    return {
        "source_id": str(source["source_id"]),
        "source_identity": source.get("source_identity"),
        "pair_key": source.get("ordinal"),
        "source_prompt_or_caption": source.get("caption"),
        "role": role,
        "generator": generator,
        "video_id": f"{source['source_id']}::{role}::{generator or 'real'}",
        "official_relative_path": relative_path,
        "video_path": str(video_path),
        "archive_path": str(archive),
        "archive_member": member,
        "pair_lineage": "shared Pair2 ordinal and official HDVG semantic record",
        "dataset": "GenVidBench",
        "revision": PAIR2_REVISION,
    }


def materialize_pair2(*, archive_dir: Path, output: Path, source_count: int = 8) -> list[dict[str, object]]:
    sources, pairs = frozen_pair2_population(source_count=source_count)
    source_by_id = {str(row["source_id"]): row for row in sources}
    selected: list[dict[str, object]] = []
    for source in sources:
        selected.append({"source": source, "role": "real", "generator": None, "relative_path": str(source["relative_path"])})
        for pair in pairs:
            if str(pair["source_id"]) == str(source["source_id"]):
                selected.append({"source": source, "role": "fake", "generator": str(pair["generator"]), "relative_path": str(pair["fake_path"])})
    if len(selected) != source_count * (1 + len(GENERATORS)):
        raise ValueError("Pair2 selection did not form the expected exact population")

    members: dict[str, list[str]] = {}
    for row in selected:
        archive_name = _archive_name(str(row["relative_path"]))
        archive = archive_dir / archive_name
        if not archive.is_file() or archive.stat().st_size == 0:
            raise FileNotFoundError(f"missing official Pair2 archive: {archive}")
        if archive_name not in members:
            members[archive_name] = list_archive_members(archive)

    media_dir = output / "media"
    result: list[dict[str, object]] = []
    for row in selected:
        source = row["source"]
        role = str(row["role"])
        generator = row["generator"]
        relative_path = str(row["relative_path"])
        archive_name = _archive_name(relative_path)
        archive = archive_dir / archive_name
        member = resolve_exact_member(members[archive_name], relative_path)
        target = media_dir / _safe_name(str(source["source_id"]), role, generator, relative_path)
        if not target.exists():
            _extract_member(archive, member, target)
        validation = validate_video(target)
        output_row = _base_row(source, role=role, generator=generator, relative_path=relative_path, archive=archive, member=member, video_path=target)
        output_row.update(validation)
        output_row["status"] = validation.get("status", "MEDIA_DECODE_FAILURE")
        result.append(output_row)
    result.sort(key=lambda row: (str(row["source_id"]), 0 if row["role"] == "real" else 1, str(row.get("generator") or "")))
    write_json(output / "manifests" / "pair2_media_manifest.json", result)
    write_json(output / "manifests" / "pair2_population.json", {
        "dataset": "GenVidBench",
        "revision": PAIR2_REVISION,
        "source_count_requested": source_count,
        "source_ids": [str(row["source_id"]) for row in sources],
        "generator_order": list(GENERATORS),
        "fake_never_used_for_fitting": True,
    })
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archives", type=Path, default=DATA_ROOT / "derived/v7_pair2_detection_pilot_v1/archives")
    parser.add_argument("--output", type=Path, default=DATA_ROOT / "derived/v7_pair2_detection_pilot_v1")
    parser.add_argument("--source-count", type=int, default=8)
    args = parser.parse_args()
    rows = materialize_pair2(archive_dir=args.archives, output=args.output, source_count=args.source_count)
    counts: dict[str, int] = {}
    for row in rows:
        counts[str(row["status"])] = counts.get(str(row["status"]), 0) + 1
    print(json.dumps({"rows": len(rows), "statuses": counts}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
