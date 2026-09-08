"""Range-extract exact Charades MP4 members without downloading the archive."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .remote_zip64 import CentralEntry, RangeClient, exact_basename, read_central_directory, extract_member


URL = "https://ai2-public-datasets.s3-us-west-2.amazonaws.com/charades/Charades_v1.zip"


def _entry_dict(entry: CentralEntry) -> dict[str, Any]:
    return {
        "name": entry.name,
        "flags": entry.flags,
        "method": entry.method,
        "crc32": entry.crc32,
        "compressed_size": entry.compressed_size,
        "uncompressed_size": entry.uncompressed_size,
        "local_header_offset": entry.local_header_offset,
    }


def materialize(base: Path, *, roles: tuple[str, ...] = ("primary", "reserve"), timeout: float = 180.0) -> list[dict[str, Any]]:
    target_dir = base / "acquisition/targeted_review_v1"
    sources: list[dict[str, Any]] = []
    for role, filename in (("primary", "selected_sources.json"), ("reserve", "reserve_sources.json")):
        if role not in roles:
            continue
        payload = json.loads((target_dir / filename).read_text(encoding="utf-8"))
        sources.extend({"role": role, **row} for row in payload["sources"])
    client = RangeClient(URL, timeout=timeout)
    client.head()
    entries, archive_meta = read_central_directory(client)
    (target_dir / "charades_range_index.json").write_text(json.dumps({"url": URL, "archive": archive_meta, "entries": [_entry_dict(entry) for entry in entries]}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    results: list[dict[str, Any]] = []
    output_root = base / "source/charades/videos"
    for source in sorted(sources, key=lambda row: (row["role"], row["charades_source_id"])):
        source_id = source["charades_source_id"]
        result = {"role": source["role"], "charades_source_id": source_id, "status": "SOURCE_MEMBER_MISSING"}
        try:
            entry = exact_basename(entries, f"{source_id}.mp4")
            result["archive_member"] = _entry_dict(entry)
            final = output_root / f"{source_id}.mp4"
            if final.is_file() and final.stat().st_size == entry.uncompressed_size:
                result.update({"status": "REUSED_EXISTING", "path": str(final)})
            else:
                extract_member(client, entry, final.with_suffix(final.suffix + ".part"))
                result.update({"status": "MATERIALIZED", "path": str(final), "local_bytes": final.stat().st_size})
        except LookupError as exc:
            result.update({"status": "SOURCE_MEMBER_AMBIGUOUS" if "found 0" not in str(exc) else "SOURCE_MEMBER_MISSING", "error": str(exc)})
        except Exception as exc:  # noqa: BLE001 - preserve per-source failure for reserve substitution
            result.update({"status": "MATERIALIZATION_FAILURE", "error": f"{type(exc).__name__}: {exc}"})
        results.append(result)
    (target_dir / "charades_materialization.json").write_text(json.dumps({"url": URL, "roles": list(roles), "results": results}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--roles", nargs="+", choices=("primary", "reserve"), default=("primary", "reserve"))
    args = parser.parse_args()
    results = materialize(args.base, roles=tuple(args.roles))
    from collections import Counter

    print(json.dumps(dict(Counter(row["status"] for row in results)), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
