"""Resolve frozen real pilot identities against official GenVidBench archives."""
from __future__ import annotations

import argparse
import json
from pathlib import Path, PurePosixPath
import shutil
import subprocess
from typing import Iterable
from urllib.request import Request, urlopen

from .metadata import write_json

HF_DATASET = "jian-0/GenVidBench"
HF_REVISION = "701cafb6f999d7ea0cbf3c354df6177311a4d824"
API_URL = f"https://huggingface.co/api/datasets/{HF_DATASET}/tree/{HF_REVISION}/GenVidBench"
DOWNLOAD_GATE = 20 * 1024**3

def fetch_inventory() -> dict[str, object]:
    query = API_URL + "?recursive=true"
    request = Request(query, headers={"Accept": "application/json"})
    entries = json.load(urlopen(request, timeout=60))
    files = []
    for entry in entries:
        if entry.get("type") != "file":
            continue
        files.append({"path": entry.get("path"), "size": entry.get("size"), "oid": entry.get("oid")})
    return {
        "dataset": HF_DATASET,
        "revision": HF_REVISION,
        "api_url": query,
        "files": sorted(files, key=lambda row: row["path"]),
        "archive_listing": {
            "Pair1/vript.rar": {"format": "RAR5", "compression": "none", "range_probe": "member headers readable; prefix vript/ confirmed"},
            "Pair2/hd_vg_130m.7z.001-004": {"format": "7z multi-volume", "range_probe": "member table not independently available without the complete volume set"},
        },
    }

def extract_requested_members(archive_path: Path, members: Iterable[str], output_root: Path) -> list[Path]:
    """Extract only explicitly named safe members from a local archive."""
    extracted: list[Path] = []
    for member in members:
        pure = PurePosixPath(member)
        if pure.is_absolute() or ".." in pure.parts or not pure.parts:
            raise ValueError(f"unsafe archive member: {member}")
        destination = output_root.joinpath(*pure.parts)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("wb") as handle:
            result = subprocess.run(
                ["bsdtar", "-xOf", str(archive_path), member],
                check=True, stdout=handle, stderr=subprocess.PIPE,
            )
        del result
        extracted.append(destination)
    return extracted

def _by_path(inventory: dict[str, object]) -> dict[str, dict[str, object]]:
    return {str(row["path"]): row for row in inventory["files"]}

def _verify_rows(path: Path) -> dict[str, list[str]]:
    result: dict[str, list[str]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        fields = line.split()
        if not fields:
            continue
        result.setdefault(Path(fields[0]).name, []).append(fields[0])
    return result

def _target_rows(output_root: Path) -> list[dict[str, object]]:
    train = json.loads((output_root / "manifests" / "train_real_pilot.json").read_text())
    test = json.loads((output_root / "manifests" / "test_real_source_pilot.json").read_text())
    return [*train, *test]

def resolve_targets(metadata_root: Path, output_root: Path, inventory: dict[str, object]) -> list[dict[str, object]]:
    pair1_verify = _verify_rows(metadata_root / "Pair1_verify.txt")
    hdvg_verify = _verify_rows(metadata_root / "HD_VG_130M_verify.txt")
    archive_parts = [
        "GenVidBench/Pair2/hd_vg_130m.7z.001",
        "GenVidBench/Pair2/hd_vg_130m.7z.002",
        "GenVidBench/Pair2/hd_vg_130m.7z.003",
        "GenVidBench/Pair2/hd_vg_130m.7z.004",
    ]
    resolved: list[dict[str, object]] = []
    for target in _target_rows(output_root):
        source = str(target["real_source"])
        relative = str(target["relative_path"])
        filename = Path(relative).name
        base = {
            "source_id": target["source_id"],
            "role": target["role"],
            "real_source": source,
            "expected_video_identity": target["source_identity"],
            "source_video_path": relative,
            "archive_shard": None,
            "archive_member": None,
            "resolution_method": None,
            "video_sha256": None,
            "status": "UNRESOLVED",
        }
        if source == "Vript":
            matches = pair1_verify.get(filename, [])
            shard = "GenVidBench/Pair1/vript.rar"
            base["archive_shard"] = shard
            if len(matches) == 1:
                verify_path = matches[0]
                base["archive_member"] = verify_path.removeprefix("GenVidBench/")
                base["resolution_method"] = "official Pair1_verify exact filename + bounded RAR member-prefix probe"
                base["status"] = "RESOLVED"
            elif not matches:
                base["resolution_method"] = "official Pair1_verify has no exact filename"
            else:
                base["resolution_method"] = "multiple official Pair1_verify matches"
                base["status"] = "AMBIGUOUS"
        elif source == "HD-VG-130M":
            base["archive_shard"] = archive_parts
            matches = hdvg_verify.get(filename, [])
            base["resolution_method"] = "official HD_VG_130M_verify basename only; multi-volume 7z member table unavailable"
            if len(matches) == 1:
                base["archive_member"] = None
            elif len(matches) > 1:
                base["status"] = "AMBIGUOUS"
        resolved.append(base)
    return resolved

def build_plan(inventory: dict[str, object], targets: list[dict[str, object]], *, metadata_root: Path, output_root: Path, range_probe_root: Path | None = None) -> dict[str, object]:
    by_path = _by_path(inventory)
    unique: set[str] = set()
    for row in targets:
        shard = row["archive_shard"]
        if isinstance(shard, list):
            unique.update(shard)
        elif shard:
            unique.add(str(shard))
    per_shard = {path: by_path.get(path, {}).get("size") for path in sorted(unique)}
    total = sum(int(size or 0) for size in per_shard.values())
    train = [row for row in targets if row["role"] == "train_real"]
    test = [row for row in targets if row["role"] == "test_real_source"]
    minimum_shards = sorted({str(row["archive_shard"]) for row in train[:16]})
    for row in test[:8]:
        if isinstance(row["archive_shard"], list): minimum_shards.extend(row["archive_shard"])
        elif row["archive_shard"]: minimum_shards.append(str(row["archive_shard"]))
    minimum_shards = sorted(set(minimum_shards))
    minimum_total = sum(int(by_path.get(path, {}).get("size") or 0) for path in minimum_shards)
    free = shutil.disk_usage(output_root).free
    probe_bytes = 0
    if range_probe_root and range_probe_root.exists():
        probe_bytes = sum(p.stat().st_size for p in range_probe_root.glob("*") if p.is_file())
    return {
        "target_count": len(targets),
        "resolved_count": sum(row["status"] == "RESOLVED" for row in targets),
        "unresolved_count": sum(row["status"] == "UNRESOLVED" for row in targets),
        "ambiguous_count": sum(row["status"] == "AMBIGUOUS" for row in targets),
        "unique_shards": sorted(unique),
        "per_shard_size": per_shard,
        "total_download_size_bytes": total,
        "total_download_size_gib": total / 1024**3,
        "minimum_tranche": {"train_real": min(16, len(train)), "test_real_source": min(8, len(test)), "unique_shards": minimum_shards, "download_size_bytes": minimum_total, "download_size_gib": minimum_total / 1024**3},
        "download_gate_bytes": DOWNLOAD_GATE,
        "free_disk_before_bytes": free,
        "expected_disk_after_full_shards_bytes": free - total,
        "actual_archive_download_bytes": 0,
        "range_probe_bytes": probe_bytes,
        "decision": "SELECTIVE_MATERIALIZATION_TOO_EXPENSIVE" if minimum_total > DOWNLOAD_GATE else "WITHIN_DOWNLOAD_GATE",
        "reason": "Even the mandated 16 train + 8 test minimum touches the complete Vript RAR and all four HD-VG 7z parts." if minimum_total > DOWNLOAD_GATE else "minimum tranche is within the bounded gate",
    }

def run(metadata_root: Path, output_root: Path, range_probe_root: Path | None = None) -> dict[str, object]:
    inventory = fetch_inventory()
    output_root.joinpath("materialization").mkdir(parents=True, exist_ok=True)
    write_json(output_root / "materialization" / "hf_file_inventory.json", inventory)
    targets = resolve_targets(metadata_root, output_root, inventory)
    write_json(output_root / "materialization" / "target_archive_map.json", {"dataset": HF_DATASET, "revision": HF_REVISION, "targets": targets})
    plan = build_plan(inventory, targets, metadata_root=metadata_root, output_root=output_root, range_probe_root=range_probe_root)
    write_json(output_root / "materialization" / "materialization_plan.json", plan)
    return plan

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--range-probe-root", type=Path)
    args = parser.parse_args()
    print(json.dumps(run(args.metadata_root, args.output_root, args.range_probe_root), indent=2, sort_keys=True))

if __name__ == "__main__":
    main()
