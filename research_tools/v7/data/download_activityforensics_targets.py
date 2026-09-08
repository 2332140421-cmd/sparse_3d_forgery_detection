"""Download only the frozen ActivityForensics review targets."""

from __future__ import annotations

import argparse
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from huggingface_hub import HfApi, hf_hub_download


REPO_ID = "ActivityForensics/ActivityForensics"
REVISION = "a34d4b7b04b0f3f3e26ba900adc367218667c581"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _target_rows(base: Path) -> list[dict[str, Any]]:
    root = base / "acquisition/targeted_review_v1"
    result: list[dict[str, Any]] = []
    for role, name in (("primary", "selected_sources.json"), ("reserve", "reserve_sources.json")):
        payload = json.loads((root / name).read_text(encoding="utf-8"))
        for row in payload["sources"]:
            variant = dict(row["fake_variant"])
            result.append({"role": role, "charades_source_id": row["charades_source_id"], "selected_cell": row["selected_cell"], "activityforensics_file": variant["activityforensics_file"], "generator": variant["generator"], "manipulation_operation": variant["manipulation_operation"]})
    return result


def download_targets(base: Path, *, roles: tuple[str, ...] = ("primary", "reserve"), workers: int = 8) -> list[dict[str, Any]]:
    api = HfApi()
    info = api.dataset_info(REPO_ID, revision=REVISION, files_metadata=True)
    metadata = {item.rfilename: item for item in info.siblings if item.rfilename.startswith("video/") and item.rfilename.endswith(".mp4")}
    raw = base / "source/activityforensics/raw"
    rows = [row for row in _target_rows(base) if row["role"] in roles]

    def one(row: dict[str, Any]) -> dict[str, Any]:
        filename = row["activityforensics_file"]
        expected = metadata.get(filename)
        result = dict(row)
        result.update({"revision": REVISION, "expected_bytes": expected.size if expected else None, "expected_sha256": expected.lfs.sha256 if expected and expected.lfs else None})
        path = raw / filename
        if expected is None:
            result.update({"status": "REMOTE_MEMBER_MISSING", "path": str(path)})
            return result
        if path.is_file() and path.stat().st_size == expected.size and sha256(path) == expected.lfs.sha256:
            result.update({"status": "REUSED_EXISTING", "path": str(path), "local_bytes": path.stat().st_size, "local_sha256": sha256(path)})
            return result
        try:
            downloaded = Path(hf_hub_download(repo_id=REPO_ID, filename=filename, revision=REVISION, repo_type="dataset", local_dir=str(raw)))
            local_hash = sha256(downloaded)
            status = "DOWNLOADED" if local_hash == expected.lfs.sha256 and downloaded.stat().st_size == expected.size else "CHECKSUM_FAILURE"
            result.update({"status": status, "path": str(downloaded), "local_bytes": downloaded.stat().st_size, "local_sha256": local_hash})
        except Exception as exc:  # noqa: BLE001 - keep per-target failure for reserve logic
            result.update({"status": "DOWNLOAD_FAILURE", "path": str(path), "error": f"{type(exc).__name__}: {exc}"})
        return result

    results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(one, row) for row in rows]
        for future in as_completed(futures):
            results.append(future.result())
    results.sort(key=lambda row: (row["role"], row["charades_source_id"], row["activityforensics_file"]))
    output = base / "acquisition/targeted_review_v1/activityforensics_downloads.json"
    output.write_text(json.dumps({"repo_id": REPO_ID, "revision": REVISION, "roles": list(roles), "results": results}, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--roles", nargs="+", choices=("primary", "reserve"), default=("primary", "reserve"))
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    results = download_targets(args.base, roles=tuple(args.roles), workers=args.workers)
    from collections import Counter

    print(json.dumps(dict(Counter(row["status"] for row in results)), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
