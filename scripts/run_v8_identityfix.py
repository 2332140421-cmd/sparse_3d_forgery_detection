#!/usr/bin/env python3
"""Run the V8 source-aware identity repair without touching the prior run.

This is a one-off continuation for the documented SL0061 collision.  It
reuses identity-verified frontend artifacts through independent hard links,
re-extracts only rows whose resolved video changed, rebuilds all derived V8
features/statistics, and starts all six frozen models from scratch.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from research_tools.v8.full_coverage_observation_field.common import read_json, write_json_atomic
from research_tools.v8.full_coverage_observation_field.frontend import CoTrackerShortPrefix, MogeV2, extract_window
from research_tools.v8.full_coverage_observation_field.pipeline import (
    _load_status,
    _safe_name,
    _stage,
    build_pair_index,
    build_manifest,
    resolve_pair_video,
    run_evaluate,
    run_features,
    run_report,
    run_train,
    run_visualizations,
    sha256_file,
    write_protocol,
)


def _arg() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--old-output", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--identity-root", required=True)
    p.add_argument("--source-root", required=True)
    p.add_argument("--moge-checkpoint", required=True)
    p.add_argument("--tracker-checkpoint", required=True)
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def _safe_copy(src: Path, dst: Path, hardlink: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    if hardlink:
        try:
            os.link(src, dst)
            return
        except OSError:
            pass
    shutil.copy2(src, dst)


def _reuse_tree(old: Path, new: Path, excluded_frontend: set[str]) -> None:
    """Reuse old raw/frontend files without giving the new run write access."""
    for rel in ["raw_frontend"]:
        root = old / rel
        if not root.exists():
            continue
        for src in root.rglob("*"):
            if src.is_file():
                _safe_copy(src, new / rel / src.relative_to(root), hardlink=src.suffix == ".npz")
    root = old / "frontend"
    if root.exists():
        for src in root.iterdir():
            if not src.is_file():
                continue
            if src.stem in excluded_frontend:
                continue
            # Metadata is copied (not hard-linked) so the repaired run can add
            # its resolved pair identity without mutating the prior run.
            _safe_copy(src, new / "frontend" / src.name, hardlink=src.suffix == ".npz")


def _write_identity_audit(old_rows: list[dict], new_rows: list[dict], out: Path, source_root: Path) -> list[dict]:
    old_by = {r["window_id"]: r for r in old_rows}
    records = []
    for row in new_rows:
        old = old_by.get(row["window_id"], {})
        changed = old.get("video_path") != row.get("video_path") or old.get("video_sha256") != row.get("video_sha256")
        records.append({
            "window_id": row["window_id"], "source_id": row["source_id"], "pair_id": row["pair_id"],
            "role": row["role"], "split": row["split"], "resolved_video_path": row["video_path"],
            "resolved_video_sha256": row.get("video_sha256"), "old_video_path": old.get("video_path"),
            "old_video_sha256": old.get("video_sha256"), "identity_changed": changed,
            "status": "REEXTRACT_REQUIRED" if changed else "IDENTITY_MATCH_REUSE_ALLOWED",
        })
    with (out / "identity_audit.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(records[0].keys())); w.writeheader(); w.writerows(records)
    write_json_atomic(out / "identity_audit.json", {
        "window_count": len(records), "changed_count": sum(r["identity_changed"] for r in records),
        "changed_windows": [r["window_id"] for r in records if r["identity_changed"]],
        "policy": "(pair_id,source_id,role) resolver; no pair_id fallback",
        "source_manifest": str(source_root / "manifests" / "input_pairs.json"),
    })
    return [r for r in records if r["identity_changed"]]


def _write_run_protocol(out: Path, run_id: str, old: Path, changed: list[dict]) -> None:
    p = out / "protocol.json"
    d = read_json(p)
    d.update({
        "run_id": run_id,
        "parent_run": str(old),
        "identity_fix": {
            "status": "PRE_FIX_TRAIN_INPUT_IDENTITY_MISMATCH_REPAIRED",
            "resolver_key": ["pair_id", "source_id"],
            "changed_window_count": len(changed),
            "changed_windows": [r["window_id"] for r in changed],
            "old_results_preserved": True,
        },
    })
    write_json_atomic(p, d)


def main() -> None:
    args = _arg()
    old = Path(args.old_output).resolve()
    out = Path(args.output).resolve()
    previous_status = None
    if out.exists() and (out / "final_status.json").exists():
        previous_status = read_json(out / "final_status.json")
        # A failed attempt may be restarted in-place, but never overwrite a
        # completed result.  The failed record remains in run_history below.
        if previous_status.get("status") == "COMPLETE":
            raise RuntimeError(f"REFUSING_TO_OVERWRITE_COMPLETED_IDENTITYFIX:{out}")
    out.mkdir(parents=True, exist_ok=True)
    run_id = f"v8-identityfix-{int(time.time())}"
    if previous_status is not None:
        history_path = out / "run_history.jsonl"
        with history_path.open("a", encoding="utf-8") as hf:
            hf.write(json.dumps({"observed_final_status": previous_status, "observed_unix": time.time()}, sort_keys=True) + "\n")
    write_json_atomic(out / "run.json", {"run_id": run_id, "status": "RUNNING", "started_unix": time.time(), "pid": os.getpid(), "command": " ".join(sys.argv), "old_output": str(old), "previous_status": previous_status})
    try:
        rows = build_manifest(out, Path(args.identity_root), Path(args.source_root))
        write_protocol(out, rows, Path(args.identity_root), args.moge_checkpoint, args.tracker_checkpoint)
        old_rows = read_json(old / "data_manifest.json")["rows"]
        changed = _write_identity_audit(old_rows, rows, out, Path(args.source_root))
        if len(changed) != 5:
            raise RuntimeError(f"UNEXPECTED_IDENTITY_CHANGE_COUNT:{len(changed)}")
        _write_run_protocol(out, run_id, old, changed)
        _stage(out, "plan", "COMPLETE", planned=len(rows), train=sum(r["split"] == "train" for r in rows), validation=sum(r["split"] == "validation" for r in rows), identity_changed=len(changed))

        changed_wids = {r["window_id"] for r in changed}
        _reuse_tree(old, out, {_safe_name(w) for w in changed_wids})
        # Reused metadata must point into this independent run and carry the
        # newly resolved source-aware identity; no old absolute cache path is
        # allowed to become the provenance of the repaired run.
        for row in rows:
            meta_path = out / "frontend" / f"{_safe_name(row['window_id'])}.json"
            if meta_path.exists():
                meta = read_json(meta_path)
                meta["npz"] = str(out / "frontend" / f"{_safe_name(row['window_id'])}.npz")
                meta["pair_id"] = row.get("pair_id")
                meta["source_id"] = row.get("source_id")
                meta["video_identity"] = row.get("video_identity")
                write_json_atomic(meta_path, meta)
        _stage(out, "frontend", "RUNNING", planned=len(rows), completed=len(rows) - len(changed), failed=0, identity_reuse=len(rows) - len(changed), reextract=len(changed))
        moge = MogeV2(args.moge_checkpoint, device=args.device)
        tracker = CoTrackerShortPrefix(args.tracker_checkpoint, device=args.device, grid_size=32)
        by = {r["window_id"]: r for r in rows}
        for record in changed:
            # overwrite=True is intentional only for the five identity-changed
            # rows, whose old frontend artifacts were not copied above.
            extract_window(by[record["window_id"]], out, moge, tracker, overwrite=True)
        frontend_results = []
        for row in rows:
            meta_path = out / "frontend" / f"{_safe_name(row['window_id'])}.json"
            if not meta_path.exists():
                raise FileNotFoundError(f"FRONTEND_REUSE_MISSING:{row['window_id']}")
            meta = read_json(meta_path)
            if meta.get("video_sha256") != row.get("video_sha256") or meta.get("source_id") != row.get("source_id") or meta.get("role") != row.get("role"):
                raise RuntimeError(f"FRONTEND_IDENTITY_POSTCHECK_FAILED:{row['window_id']}")
            frontend_results.append({"window_id": row["window_id"], "status": meta.get("status"), "npz": meta.get("npz")})
        write_json_atomic(out / "frontend_manifest.json", {"planned": len(rows), "completed": len(frontend_results), "failed": 0, "reused": len(rows) - len(changed), "reextracted": len(changed), "results": frontend_results})
        _stage(out, "frontend", "COMPLETE", planned=len(rows), completed=len(rows), failed=0, reused=len(rows) - len(changed), reextracted=len(changed))
        write_json_atomic(out / "smoke.json", {"status": "IDENTITY_REAL_FAKE_SMOKE_PASS", "checked_windows": [r["window_id"] for r in changed if r["role"] == "real"] + [r["window_id"] for r in changed if r["role"] == "fake"], "note": "Corrected real/fake frontend metadata and cache identities verified before feature rebuild."})
        # Release the large frontend networks before loading the RGB backbone
        # used by feature extraction.  This is a memory-lifetime change only;
        # it does not alter any frozen input or training protocol.
        del moge, tracker
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

        # Recompute grouping scales and every derived feature from the complete
        # repaired frontend set; this prevents old train-fitted scales from
        # mixing with the repaired source.
        run_features(out, rows, args.device)
        models_dir = out / "models"
        if any(models_dir.rglob("*.pt")) or any(models_dir.rglob("*.pth")):
            raise RuntimeError("IDENTITYFIX_MODEL_OUTPUT_ALREADY_PRESENT_RESTART_REQUIRED")
        run_train(out, rows, args.device)
        run_evaluate(out, rows, args.device)
        run_visualizations(out, rows, args.device)
        run_report(out, rows)
        status = {"status": "COMPLETE", "run_id": run_id, "changed_windows": len(changed), "completed_models": 6, "updated_unix": time.time()}
        write_json_atomic(out / "final_status.json", status)
        write_json_atomic(out / "run.json", {"run_id": run_id, "status": "COMPLETE", "started_unix": read_json(out / "run.json").get("started_unix"), "finished_unix": time.time(), "pid": os.getpid(), "changed_windows": len(changed)})
    except Exception as exc:
        write_json_atomic(out / "final_status.json", {"status": "FAILED", "run_id": run_id, "error": f"{type(exc).__name__}: {exc}", "updated_unix": time.time()})
        write_json_atomic(out / "run.json", {"run_id": run_id, "status": "FAILED", "error": f"{type(exc).__name__}: {exc}", "finished_unix": time.time()})
        raise


if __name__ == "__main__":
    main()
