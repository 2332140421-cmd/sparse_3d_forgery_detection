"""Independent runner for the V7 boundary/pooling matched pilot.

The runner reuses only the already materialized 289-point ParticleSequence
artifacts and the frozen local-organization representation.  It writes small
JSON/CSV state files so a process can be resumed without a second scheduler.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import time
import traceback
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from sparse3d_forgery.particle_sequence import load_particle_sequence
from sparse3d_forgery.video_input import VideoSource, decode_video
from research_tools.v7.local_organization_probe.grouping import (
    build_local_groups,
    build_local_support,
    build_support_from_components,
    rebuild_components_fast,
)
from research_tools.v7.local_structural_temporal_probe.model import (
    ARM_NAMES as ALL_ARM_NAMES,
    MODEL_CONFIG,
    build_batch,
    fit_weighted_standardizer,
    window_label,
)
from research_tools.v7.local_structural_temporal_probe.representation import COMPONENT_CONFIG, support_arrays

from .grouping import assign_uv_to_masks, split_local_groups_by_assignment, validate_boundary_partition
from .model import score_pooling_model, serialize_pooling_model, train_pooling_model


REPO_ROOT = Path(__file__).resolve().parents[3]
DATA_ROOT = Path("/root/autodl-tmp/data/sparse_3d_forgery_detection")
SOURCE_ROOT = DATA_ROOT / "derived/v7_activityforensics_local_organization_pilot_v1"
FRONTEND_ROOT = DATA_ROOT / "derived/v7_activityforensics_density_matched_frontend_v1"
PAIRED_ROOT = DATA_ROOT / "derived/v7_activityforensics_paired_second_order_pilot_v1"
OUTPUT_ROOT = DATA_ROOT / "derived/v7_activityforensics_boundary_pooling_pilot_v1"
SEG_WEIGHT = DATA_ROOT / "external/yolo26_depth/yolo26m-seg.pt"
SEEDS = (20260909, 20260910, 20260911)
BOOTSTRAP_SEED = 20260909
BOOTSTRAP_REPLICATES = 10_000
ARMS = ("A", "C", "D")
CONDITIONS = tuple(f"{organization}_{pooling}_{arm}" for organization in ("H", "B") for pooling in ("MEAN", "MAX") for arm in ARMS)
ARM_TO_EXISTING = {"A": "UNORDERED_STATE", "C": "ORDERED_SECOND", "D": "PERMUTED_SECOND"}


def _safe(value: str) -> str:
    return "".join(char if char.isalnum() or char in "._-" else "_" for char in value)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_head() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.int64)):
        return int(value)
    if isinstance(value, (np.floating, np.float64)):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    raise TypeError(type(value).__name__)


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False, default=_json_default) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _progress(root: Path, phase: str, completed: int, total: int, status: str = "RUNNING", **extra: Any) -> None:
    _atomic_json(root / "progress.json", {"phase": phase, "completed": int(completed), "total": int(total), "status": status, "updated_unix": time.time(), **extra})


def _load_input_rows() -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]], dict[str, Any]]:
    path = SOURCE_ROOT / "manifests/input_manifest.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    rows = list(data.get("rows", []))
    if len(rows) != 192:
        raise RuntimeError(f"expected 192 frozen windows, found {len(rows)}")
    paired_rows = json.loads((PAIRED_ROOT / "manifests/window_manifest.json").read_text(encoding="utf-8"))
    by_id = {str(row["window_id"] + "::" + row["role"]): row for row in paired_rows}
    if len(by_id) != 192:
        raise RuntimeError("paired manifest is not the frozen 192-window population")
    return rows, by_id, data.get("population", {})


def _compact_grouping(grouping: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "local_diameter_m": float(grouping.get("local_diameter_m", 0.30)),
        "component_config": dict(grouping.get("component_config", {})),
        "history_array_indices": [int(x) for x in grouping.get("history_array_indices", [])],
        "parent_components": [
            {"parent_component_id": int(row["parent_component_id"]), "member_slots": [int(x) for x in row.get("member_slots", [])], "track_ids": [int(x) for x in row.get("track_ids", [])]}
            for row in grouping.get("parent_components", [])
        ],
        "groups": [
            {key: value for key, value in row.items() if key not in {"pair_evidence"}}
            for row in grouping.get("groups", [])
        ],
        "retained_group_count": int(grouping.get("retained_group_count", 0)),
        "support_insufficient_group_count": int(grouping.get("support_insufficient_group_count", 0)),
    }


def _compact_support(support: Mapping[str, Any]) -> dict[str, Any]:
    prepared = support_arrays(support)
    return {
        "support_status": str(support.get("support_status")),
        "valid_triplet_count": int(support.get("valid_triplet_count", 0)),
        "invalid_reasons": list(support.get("invalid_reasons", [])),
        "triplets": [
            {**{key: value for key, value in triplet.items() if key not in {"states", "timestamps_s"}}, "states": np.asarray(triplet["states"], dtype=np.float64).tolist(), "timestamps_s": np.asarray(triplet["timestamps_s"], dtype=np.float64).tolist()}
            for triplet in prepared["triplets"]
        ],
    }


def _feature_path(root: Path, window_id: str) -> Path:
    return root / "features" / f"{_safe(window_id)}.json"


def _load_feature(root: Path, row: Mapping[str, Any]) -> dict[str, Any]:
    return json.loads(_feature_path(root, str(row["window_id"])).read_text(encoding="utf-8"))


def prepare(root: Path, *, resume: bool) -> None:
    rows, paired_by_id, population = _load_input_rows()
    root.mkdir(parents=True, exist_ok=True)
    manifest_rows: list[dict[str, Any]] = []
    _progress(root, "prepare", 0, len(rows))
    for index, row in enumerate(rows, 1):
        path = _feature_path(root, str(row["window_id"]))
        if resume and path.is_file():
            manifest_rows.append(json.loads(path.read_text(encoding="utf-8"))["identity"])
            _progress(root, "prepare", index, len(rows))
            continue
        prefix = Path(str(row["particle_prefix"]))
        sequence = load_particle_sequence(prefix)
        if sequence.num_tracks != 289:
            raise RuntimeError(f"{row['window_id']} is not the frozen 289-point artifact")
        history_indices = np.flatnonzero(sequence.timestamps_s < float(row["interval_start_s"]) + 0.5).astype(np.int64)
        old_components = rebuild_components_fast(sequence.xyz, sequence.geometry_validity, history_indices, COMPONENT_CONFIG)
        h_grouping = build_local_groups(sequence, history_indices, old_components=old_components)
        h_support = build_local_support(sequence, window_start_s=float(row["interval_start_s"]), grouping=h_grouping)
        paired = paired_by_id.get(str(row["window_id"]) + "::" + str(row["role"]))
        if paired is None:
            raise RuntimeError(f"missing paired video row for {row['window_id']}")
        identity = {
            "window_id": str(row["window_id"]), "source_id": str(row["source_id"]), "pair_id": str(row["pair_id"]),
            "role": str(row["role"]), "kind": str(row["kind"]), "label": window_label(row),
            "anchor_fraction": float(row["anchor_fraction"]), "interval_start_s": float(row["interval_start_s"]), "interval_end_s": float(row["interval_end_s"]),
            "particle_prefix": str(prefix), "particle_npz_sha256": str(row["particle_npz_sha256"]), "particle_json_sha256": str(row["particle_json_sha256"]),
            "frame_indices": [int(x) for x in sequence.frame_indices], "timestamps_s": [float(x) for x in sequence.timestamps_s],
            "video_path": str(paired.get("video_path", "")), "source_frame_index": int(sequence.frame_indices[0]), "source_timestamp_s": float(sequence.timestamps_s[0]),
            "track_count": int(sequence.num_tracks), "h_group_count": int(len(h_grouping.get("groups", []))),
            "h_retained_group_count": int(h_grouping.get("retained_group_count", 0)), "h_support_status": str(h_support.get("support_status")),
            "h_valid_triplet_count": int(h_support.get("valid_triplet_count", 0)),
        }
        _atomic_json(path, {"identity": identity, "h_grouping": _compact_grouping(h_grouping), "h_support": _compact_support(h_support), "segmentation": {"status": "PENDING"}})
        manifest_rows.append(identity)
        _progress(root, "prepare", index, len(rows))
    _atomic_json(root / "manifests/input_manifest.json", {"population": {"frozen_windows": 192, "frozen_sources": 16, "source_population": population, "git_head": _git_head()}, "rows": manifest_rows})
    _atomic_json(root / "protocol.json", {"experiment": "V7 boundary partition and local pooling matched pilot", "git_head": _git_head(), "source_artifact_root": str(SOURCE_ROOT), "frontend_root": str(FRONTEND_ROOT), "population": "16 sources, 192 windows, 289-point density-matched ParticleSequence", "representation": "existing H local grouping; B first-frame instance-mask partition of H; same S(t), timestamp-aware first/second derivatives and A/C/D arms", "boundaries": ["development pilot only", "no ROI gate", "no formal src changes", "no dense branch inputs"]})


def _mask_cache_key(video_path: str, frame_index: int) -> str:
    weight_hash = getattr(_mask_cache_key, "weight_hash", None)
    if weight_hash is None:
        weight_hash = _sha256(SEG_WEIGHT) if SEG_WEIGHT.is_file() else "missing"
        setattr(_mask_cache_key, "weight_hash", weight_hash)
    text = f"{video_path}\0{frame_index}\0{weight_hash}\0ultralytics-default-seg"
    return hashlib.sha256(text.encode()).hexdigest()[:24]


def _predict_masks(video_path: Path, frame_index: int, model: Any) -> dict[str, Any]:
    decoded = decode_video(VideoSource(sample_id=f"boundary-{video_path.name}-{frame_index}", source_video_id=video_path.stem, source_locator=video_path), [int(frame_index)])
    image = np.asarray(decoded.frames[0].rgb)
    result = model.predict(source=image, verbose=False, device=0)[0]
    polygons: list[list[list[float]]] = []
    areas: list[float] = []
    if getattr(result, "masks", None) is not None and result.masks.xy is not None:
        try:
            import cv2
        except ImportError as exc:
            raise RuntimeError("opencv is required only to rasterize first-frame segmentation masks") from exc
        for poly in result.masks.xy:
            points = np.asarray(poly, dtype=np.float64)
            if points.ndim != 2 or points.shape[1] != 2 or points.shape[0] < 3:
                continue
            polygons.append(points.tolist())
            areas.append(float(abs(cv2.contourArea(points.astype(np.float32)))))
    return {"source_video": str(video_path), "frame_index": int(frame_index), "image_shape_hw": [int(image.shape[0]), int(image.shape[1])], "polygons": polygons, "areas": areas, "provider": "ultralytics YOLO26m-seg", "weight": str(SEG_WEIGHT), "weight_sha256": _sha256(SEG_WEIGHT), "predict": {"device": "cuda:0", "defaults": True}}


def _polygons_to_masks(record: Mapping[str, Any]) -> list[np.ndarray]:
    import cv2

    height, width = (int(record["image_shape_hw"][0]), int(record["image_shape_hw"][1]))
    masks: list[np.ndarray] = []
    for polygon in record.get("polygons", []):
        canvas = np.zeros((height, width), dtype=np.uint8)
        points = np.rint(np.asarray(polygon, dtype=np.float32)).astype(np.int32)
        if points.shape[0] >= 3:
            cv2.fillPoly(canvas, [points], 1)
        masks.append(canvas.astype(bool))
    return masks


def segmentation(root: Path, *, resume: bool) -> None:
    if not SEG_WEIGHT.is_file():
        raise FileNotFoundError(SEG_WEIGHT)
    rows = json.loads((root / "manifests/input_manifest.json").read_text(encoding="utf-8"))["rows"]
    mask_dir = root / "masks"
    mask_dir.mkdir(parents=True, exist_ok=True)
    _progress(root, "segmentation", 0, len(rows))
    model = None
    cache: dict[str, dict[str, Any]] = {}
    for index, row in enumerate(rows, 1):
        feature_path = _feature_path(root, str(row["window_id"]))
        feature = json.loads(feature_path.read_text(encoding="utf-8"))
        video = Path(str(row.get("video_path", "")))
        key = _mask_cache_key(str(video), int(row["source_frame_index"]))
        mask_path = mask_dir / f"{key}.json"
        if key not in cache:
            if resume and mask_path.is_file():
                cache[key] = json.loads(mask_path.read_text(encoding="utf-8"))
            elif not video.is_file():
                cache[key] = {"status": "SOURCE_MISSING", "source_video": str(video), "frame_index": int(row["source_frame_index"]), "polygons": [], "areas": []}
                _atomic_json(mask_path, cache[key])
            else:
                if model is None:
                    from ultralytics import YOLO
                    model = YOLO(str(SEG_WEIGHT))
                try:
                    cache[key] = {"status": "COMPLETE", **_predict_masks(video, int(row["source_frame_index"]), model)}
                except Exception as exc:  # preserve the exact failure and let B remain unsupported
                    cache[key] = {"status": "DECODE_FAILED", "source_video": str(video), "frame_index": int(row["source_frame_index"]), "polygons": [], "areas": [], "error": f"{type(exc).__name__}: {exc}"}
                _atomic_json(mask_path, cache[key])
        record = cache[key]
        prefix = Path(str(row["particle_prefix"]))
        sequence = load_particle_sequence(prefix)
        masks = _polygons_to_masks(record) if record.get("status") == "COMPLETE" else []
        assignments = assign_uv_to_masks(np.asarray(sequence.uv)[0], np.asarray(sequence.visibility)[0], masks, record.get("areas", []))
        feature["segmentation"] = {"status": str(record.get("status")), "cache_key": key, "cache_path": str(mask_path), "assignment_counts": {str(int(k)): int(v) for k, v in zip(*np.unique(assignments, return_counts=True))}, "assignments": assignments.tolist()}
        _atomic_json(feature_path, feature)
        _progress(root, "segmentation", index, len(rows), source_missing=sum(record.get("status") == "SOURCE_MISSING" for record in cache.values()), decode_failed=sum(record.get("status") == "DECODE_FAILED" for record in cache.values()))
    _atomic_json(root / "manifests/segmentation.json", {"weight": str(SEG_WEIGHT), "weight_sha256": _sha256(SEG_WEIGHT), "cache_count": len(cache), "statuses": dict(Counter(str(item.get("status")) for item in cache.values())), "ultralytics_defaults": True})


def features(root: Path, *, resume: bool) -> None:
    manifest = json.loads((root / "manifests/input_manifest.json").read_text(encoding="utf-8"))
    rows = manifest["rows"]
    coverage: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []
    common: list[str] = []
    _progress(root, "features", 0, len(rows))
    for index, row in enumerate(rows, 1):
        path = _feature_path(root, str(row["window_id"]))
        feature = json.loads(path.read_text(encoding="utf-8"))
        if not (resume and feature.get("b_support") is not None):
            sequence = load_particle_sequence(Path(str(row["particle_prefix"])))
            h_grouping = feature["h_grouping"]
            assignments = np.asarray(feature.get("segmentation", {}).get("assignments", [-1] * sequence.num_tracks), dtype=np.int64)
            b_grouping = split_local_groups_by_assignment(h_grouping, assignments, minimum_size=int(COMPONENT_CONFIG.minimum_size))
            for child in b_grouping["groups"]:
                child["track_ids"] = [int(sequence.track_ids[int(slot)]) for slot in child["member_slots"]]
            checks = validate_boundary_partition(h_grouping, b_grouping)
            if not checks["all_pass"]:
                raise RuntimeError(f"boundary partition invariant failed: {row['window_id']}: {checks}")
            b_support = build_local_support(sequence, window_start_s=float(row["interval_start_s"]), grouping=b_grouping)
            feature["b_grouping"] = _compact_grouping(b_grouping)
            feature["b_support"] = _compact_support(b_support)
            feature["boundary_checks"] = checks
            _atomic_json(path, feature)
        h_valid = str(feature["h_support"]["support_status"]) == "VALID"
        b_valid = str(feature["b_support"]["support_status"]) == "VALID"
        common_valid = h_valid and b_valid
        if common_valid:
            common.append(str(row["window_id"]))
        row_out = {"window_id": str(row["window_id"]), "source_id": str(row["source_id"]), "pair_id": str(row["pair_id"]), "kind": str(row["kind"]), "role": str(row["role"]), "label": row.get("label"), "h_support": "VALID" if h_valid else "NO_VALID_TRIPLET", "b_support": "VALID" if b_valid else "NO_VALID_TRIPLET", "common_support": common_valid, "h_group_count": int(feature["h_grouping"].get("retained_group_count", 0)), "b_group_count": int(feature["b_grouping"].get("retained_group_count", 0)), "segmentation_status": str(feature.get("segmentation", {}).get("status"))}
        coverage.append(row_out)
        diagnostics.append({"window_id": str(row["window_id"]), "source_id": str(row["source_id"]), "h_groups": int(feature["h_grouping"].get("retained_group_count", 0)), "b_groups": int(feature["b_grouping"].get("retained_group_count", 0)), "h_triplets": int(feature["h_support"].get("valid_triplet_count", 0)), "b_triplets": int(feature["b_support"].get("valid_triplet_count", 0)), "cut_group_count": int(feature["h_grouping"].get("retained_group_count", 0)) - int(feature["b_grouping"].get("retained_group_count", 0)), "segmentation_status": str(feature.get("segmentation", {}).get("status"))})
        _progress(root, "features", index, len(rows))
    _write_csv(root / "evaluation/coverage.csv", coverage)
    _write_csv(root / "evaluation/group_diagnostics.csv", diagnostics)
    _atomic_json(root / "manifests/common_windows.json", {"window_ids": common, "count": len(common), "source_ids": sorted({row["source_id"] for row in coverage if row["common_support"]}), "rule": "H and B both have valid representation; no score-dependent exclusion"})


def _examples(root: Path, rows: Sequence[Mapping[str, Any]], organization: str) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for row in rows:
        feature = _load_feature(root, row)
        support = feature["h_support"] if organization == "H" else feature["b_support"]
        triplets = support.get("triplets", [])
        if not triplets:
            continue
        result.append({"window_id": str(row["window_id"]), "pair_id": str(row["pair_id"]), "source_id": str(row["source_id"]), "role": str(row["role"]), "kind": str(row["kind"]), "anchor_fraction": float(row["anchor_fraction"]), "triplets": triplets})
    return result


def _load_common_rows(root: Path) -> list[dict[str, Any]]:
    rows = json.loads((root / "manifests/input_manifest.json").read_text(encoding="utf-8"))["rows"]
    ids = set(json.loads((root / "manifests/common_windows.json").read_text(encoding="utf-8"))["window_ids"])
    return [row for row in rows if str(row["window_id"]) in ids]


def _weight_audit(examples: Sequence[Mapping[str, Any]], weights: np.ndarray) -> dict[str, Any]:
    sums: defaultdict[str, float] = defaultdict(float)
    counts: Counter[str] = Counter()
    for example, weight in zip(examples, weights):
        label = window_label(example)
        key = f"{example['source_id']}::{'real' if label == 0 else 'fake'}"
        sums[key] += float(weight); counts[key] += 1
    return {"source_class_counts": dict(sorted(counts.items())), "source_class_weight_sums": {key: float(value) for key, value in sorted(sums.items())}, "mean": float(np.mean(weights)), "min": float(np.min(weights)), "max": float(np.max(weights))}


def _load_records(root: Path) -> dict[str, Any]:
    path = root / "models/fold_models.json"
    if path.is_file():
        return json.loads(path.read_text(encoding="utf-8"))
    return {"model_config": MODEL_CONFIG, "seeds": list(SEEDS), "records": []}


def _save_oof(root: Path, oof: Mapping[str, Mapping[str, Any]]) -> None:
    rows = list(oof.values())
    _write_csv(root / "scores/oof_window_scores.csv", rows)


def train(root: Path, *, resume: bool) -> None:
    if not __import__("torch").cuda.is_available():
        raise RuntimeError("boundary pooling pilot requires CUDA; refusing CPU fallback")
    rows = _load_common_rows(root)
    all_sources = sorted({str(row["source_id"]) for row in json.loads((root / "manifests/input_manifest.json").read_text())["rows"]})
    records_file = root / "models/fold_models.json"
    records_data = _load_records(root)
    records = records_data.setdefault("records", [])
    done_keys = {(str(item.get("condition")), str(item.get("held_out_source")), int(item.get("seed", -1))) for item in records}
    oof: dict[str, dict[str, Any]] = {}
    oof_file = root / "scores/oof_window_scores.csv"
    if resume and oof_file.is_file():
        for row in _read_csv(oof_file):
            oof[str(row["window_id"])] = {key: (None if value == "" else value) for key, value in row.items()}
    total = len(CONDITIONS) * len(all_sources) * len(SEEDS); completed = len(done_keys)
    _progress(root, "train", completed, total, device="cuda")
    for condition in CONDITIONS:
        organization, pooling, arm = condition.split("_")
        existing_arm = ARM_TO_EXISTING[arm]
        examples = _examples(root, rows, organization)
        main = [item for item in examples if window_label(item) is not None]
        for held_out in all_sources:
            heldout = [item for item in examples if str(item["source_id"]) == held_out]
            training = [item for item in main if str(item["source_id"]) != held_out]
            if not heldout or not training or any(len({window_label(x) for x in training if str(x["source_id"]) == source}) != 2 for source in {str(x["source_id"]) for x in training}):
                continue
            raw_batch, obs = build_batch(training, existing_arm, observation_weights=True)
            if obs is None: raise RuntimeError("missing observation weights")
            standardizer = fit_weighted_standardizer(raw_batch.inputs, obs)
            train_batch, _ = build_batch(training, existing_arm, standardizer=standardizer, observation_weights=True)
            held_batch, _ = build_batch(heldout, existing_arm, standardizer=standardizer)
            audit = _weight_audit(training, train_batch.window_weights)
            for seed in SEEDS:
                key = (condition, held_out, int(seed))
                if resume and key in done_keys:
                    completed += 1; _progress(root, "train", completed, total, device="cuda"); continue
                model, fit = train_pooling_model(train_batch, pooling=pooling.lower(), seed=seed, device="cuda")
                scores = score_pooling_model(model, held_batch)
                for example, score in zip(heldout, scores):
                    item = oof.setdefault(str(example["window_id"]), {"window_id": str(example["window_id"]), "source_id": str(example["source_id"]), "pair_id": str(example["pair_id"]), "kind": str(example["kind"]), "role": str(example["role"]), "label": window_label(example)})
                    item[f"{condition}_seed_{seed}"] = float(score)
                records.append({"condition": condition, "organization": organization, "pooling": pooling, "arm": arm, "existing_arm": existing_arm, "held_out_source": held_out, "seed": int(seed), "training_real_count": int(sum(window_label(x) == 0 for x in training)), "training_fake_count": int(sum(window_label(x) == 1 for x in training)), "held_out_window_count": len(heldout), "train_triplet_count": int(train_batch.n_triplets), "held_triplet_count": int(held_batch.n_triplets), "standardization": standardizer.as_dict(), "weight_audit": audit, "fit": fit, "model": serialize_pooling_model(model, standardizer)})
                _atomic_json(records_file, {"model_config": MODEL_CONFIG, "seeds": list(SEEDS), "conditions": list(CONDITIONS), "records": records})
                _save_oof(root, oof)
                completed += 1; _progress(root, "train", completed, total, device="cuda", last_condition=condition, last_source=held_out, last_seed=int(seed))
                del model
                if __import__("torch").cuda.is_available(): __import__("torch").cuda.empty_cache()
    _atomic_json(records_file, {"model_config": MODEL_CONFIG, "seeds": list(SEEDS), "conditions": list(CONDITIONS), "records": records})


def _auroc(fake: Iterable[float], real: Iterable[float]) -> float | None:
    pos = np.asarray([x for x in fake if x is not None and np.isfinite(float(x))], dtype=np.float64)
    neg = np.asarray([x for x in real if x is not None and np.isfinite(float(x))], dtype=np.float64)
    if pos.size == 0 or neg.size == 0: return None
    comparisons = (pos[:, None] > neg[None, :]).astype(np.float64) + 0.5 * (pos[:, None] == neg[None, :])
    return float(np.mean(comparisons))


def _source_metrics(oof_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in oof_rows:
        if str(row.get("kind")) != "MANIP": continue
        for condition in CONDITIONS:
            value = row.get(f"{condition}_score")
            if value not in (None, ""):
                grouped[str(row["source_id"])][f"{condition}_{row['role']}"] .append(float(value))
    output: list[dict[str, Any]] = []
    for source in sorted(grouped):
        result: dict[str, Any] = {"source_id": source}
        for condition in CONDITIONS:
            fake = grouped[source].get(f"{condition}_fake", []); real = grouped[source].get(f"{condition}_real", [])
            result[f"{condition}_fake_count"] = len(fake); result[f"{condition}_real_count"] = len(real); result[f"{condition}_auroc"] = _auroc(fake, real)
        output.append(result)
    return output


def _bootstrap(source_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    complete = [row for row in source_rows if all(row.get(f"{condition}_auroc") is not None for condition in CONDITIONS)]
    if not complete: return {"N": 0, "seed": BOOTSTRAP_SEED, "replicates": BOOTSTRAP_REPLICATES, "conditions": {}, "comparisons": {}}
    values = {condition: np.asarray([float(row[f"{condition}_auroc"]) for row in complete]) for condition in CONDITIONS}
    rng = np.random.default_rng(BOOTSTRAP_SEED); indices = rng.integers(0, len(complete), size=(BOOTSTRAP_REPLICATES, len(complete)))
    means = {condition: np.mean(values[condition][indices], axis=1) for condition in CONDITIONS}
    comparisons: dict[str, tuple[str, str]] = {}
    for arm in ARMS:
        comparisons[f"H_MAX-H_MEAN_{arm}"] = (f"H_MAX_{arm}", f"H_MEAN_{arm}")
        comparisons[f"B_MAX-B_MEAN_{arm}"] = (f"B_MAX_{arm}", f"B_MEAN_{arm}")
        comparisons[f"B_MEAN-H_MEAN_{arm}"] = (f"B_MEAN_{arm}", f"H_MEAN_{arm}")
        comparisons[f"B_MAX-H_MAX_{arm}"] = (f"B_MAX_{arm}", f"H_MAX_{arm}")
        comparisons[f"B_MAX-H_MEAN_{arm}"] = (f"B_MAX_{arm}", f"H_MEAN_{arm}")
        comparisons[f"H_MEAN_C-A"] = ("H_MEAN_C", "H_MEAN_A"); comparisons[f"H_MEAN_C-D"] = ("H_MEAN_C", "H_MEAN_D")
        comparisons[f"H_MAX_C-A"] = ("H_MAX_C", "H_MAX_A"); comparisons[f"H_MAX_C-D"] = ("H_MAX_C", "H_MAX_D")
        comparisons[f"B_MEAN_C-A"] = ("B_MEAN_C", "B_MEAN_A"); comparisons[f"B_MEAN_C-D"] = ("B_MEAN_C", "B_MEAN_D")
        comparisons[f"B_MAX_C-A"] = ("B_MAX_C", "B_MAX_A"); comparisons[f"B_MAX_C-D"] = ("B_MAX_C", "B_MAX_D")
    return {"N": len(complete), "source_ids": [str(x["source_id"]) for x in complete], "seed": BOOTSTRAP_SEED, "replicates": BOOTSTRAP_REPLICATES, "conditions": {condition: {"mean": float(np.mean(values[condition])), "ci95": [float(np.percentile(means[condition], 2.5)), float(np.percentile(means[condition], 97.5))]} for condition in CONDITIONS}, "comparisons": {name: {"mean": float(np.mean(values[left] - values[right])), "ci95": [float(np.percentile(means[left] - means[right], 2.5)), float(np.percentile(means[left] - means[right], 97.5))]} for name, (left, right) in comparisons.items()}}


def evaluate(root: Path, *, resume: bool) -> None:
    rows = _read_csv(root / "scores/oof_window_scores.csv")
    for row in rows:
        for condition in CONDITIONS:
            values = [row.get(f"{condition}_seed_{seed}") for seed in SEEDS]
            finite = [float(x) for x in values if x not in (None, "") and np.isfinite(float(x))]
            row[f"{condition}_score"] = float(np.mean(finite)) if len(finite) == len(SEEDS) else None
            row[f"{condition}_seed_std"] = float(np.std(finite)) if len(finite) == len(SEEDS) else None
    _write_csv(root / "scores/oof_window_scores.csv", rows)
    source_rows = _source_metrics(rows); bootstrap = _bootstrap(source_rows)
    _write_csv(root / "evaluation/per_source_metrics.csv", source_rows)
    paired_rows = [{"comparison": name, "mean": value["mean"], "ci95_low": value["ci95"][0], "ci95_high": value["ci95"][1]} for name, value in bootstrap.get("comparisons", {}).items()]
    _write_csv(root / "evaluation/paired_gains.csv", paired_rows)
    coverage = _read_csv(root / "evaluation/coverage.csv")
    complete = int(bootstrap.get("N", 0))
    status = "BOUNDARY_POOLING_EVALUATION_COMPLETE" if complete >= 12 and all(row.get(f"{c}_auroc") is not None for row in source_rows for c in CONDITIONS) else "BOUNDARY_POOLING_SUPPORT_INSUFFICIENT"
    summary = {"status": status, "git_head": _git_head(), "conditions": list(CONDITIONS), "coverage": {"frozen_windows": len(coverage), "h_valid": sum(x["h_support"] == "VALID" for x in coverage), "b_valid": sum(x["b_support"] == "VALID" for x in coverage), "common": sum(x["common_support"] == "True" for x in coverage), "source_count": len({x["source_id"] for x in coverage})}, "source_level_complete_count": complete, "source_metrics": source_rows, "bootstrap": bootstrap, "paired_comparisons": "evaluation/paired_gains.csv", "input": "289-point local organization only; no dense branch, labels/source not numeric input", "formal_src_modified": False}
    _atomic_json(root / "evaluation/summary.json", summary); _atomic_json(root / "run_summary.json", {"status": status, "summary": "evaluation/summary.json", "artifact_root": str(root), "updated_unix": time.time()})
    _progress(root, "evaluate", 1, 1, status=status, complete_sources=complete)


def _visual_report(root: Path) -> None:
    try:
        import cv2
    except ImportError:
        _atomic_json(root / "visualizations/manifest.json", {"status": "UNAVAILABLE", "reason": "opencv unavailable"}); return
    rows = json.loads((root / "manifests/input_manifest.json").read_text(encoding="utf-8"))["rows"]
    selected: list[dict[str, Any]] = []
    for source in sorted({str(row["source_id"]) for row in rows}):
        for role in ("real", "fake"):
            candidates = [row for row in rows if str(row["source_id"]) == source and str(row["role"]) == role and str(row["kind"]) == "MANIP"]
            if candidates: selected.append(sorted(candidates, key=lambda x: float(x["anchor_fraction"]))[0])
        if len(selected) >= 8: break
    records: list[dict[str, Any]] = []
    colors = [(0, 220, 255), (255, 100, 180), (100, 255, 120), (255, 170, 40), (180, 120, 255), (80, 220, 220)]
    for row in selected:
        feature = _load_feature(root, row); video = Path(str(row["video_path"]))
        if not video.is_file(): records.append({"window_id": row["window_id"], "status": "SOURCE_MISSING"}); continue
        try:
            image = cv2.cvtColor(np.asarray(decode_video(VideoSource(sample_id=f"report-{row['window_id']}", source_video_id=str(row["source_id"]), source_locator=video), [int(row["source_frame_index"]) ]).frames[0].rgb), cv2.COLOR_RGB2BGR)
            sequence = load_particle_sequence(Path(str(row["particle_prefix"])))
            h_groups = [x for x in feature["h_grouping"].get("groups", []) if x.get("retained")]
            b_groups = [x for x in feature["b_grouping"].get("groups", []) if x.get("retained")]
            panels = []
            for title, groups, key in (("H local groups", h_groups, "member_slots"), ("B boundary children", b_groups, "member_slots")):
                canvas = image.copy(); assignment = {int(slot): idx for idx, group in enumerate(groups) for slot in group.get(key, [])}
                for slot, uv in enumerate(np.asarray(sequence.uv)[0]):
                    if slot not in assignment or not bool(sequence.visibility[0, slot]) or not np.all(np.isfinite(uv)): continue
                    point = (int(round(float(uv[0]))), int(round(float(uv[1])))); cv2.circle(canvas, point, 4, colors[assignment[slot] % len(colors)], -1)
                cv2.putText(canvas, f"{title}: {len(groups)} groups (not pixel GT)", (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 2, cv2.LINE_AA); panels.append(canvas)
            out = root / "visualizations" / f"{_safe(row['window_id'])}.png"; out.parent.mkdir(parents=True, exist_ok=True); cv2.imwrite(str(out), np.concatenate(panels, axis=1)); records.append({"window_id": row["window_id"], "source_id": row["source_id"], "role": row["role"], "status": "WRITTEN", "path": str(out), "frame_index": row["source_frame_index"], "timestamp_s": row["source_timestamp_s"]})
        except Exception as exc:
            records.append({"window_id": row["window_id"], "status": "DECODE_FAILED", "error": f"{type(exc).__name__}: {exc}"})
    _atomic_json(root / "visualizations/manifest.json", {"selection": "earliest MANIP real/fake per source, first eight sources", "records": records})


def report(root: Path, *, resume: bool) -> None:
    _visual_report(root)
    summary_path = root / "evaluation/summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.is_file() else {"status": "NOT_EVALUATED"}
    text = f"""# V7 Boundary and Local-Pooling Matched Pilot\n\nStatus: `{summary.get('status')}`\n\nThis is a development-only matched pilot on the frozen 289-point artifacts. H is the historical local-group representation from the prior pilot. B partitions each H group by a first-frame YOLO26m-seg instance boundary; it never merges parents or reassigns points in later frames. Instance category, confidence and mask area are not numeric model inputs.\n\nThe matrix is H/B × MEAN/MAX × A/C/D, with the common H∩B representation window set and source-disjoint LOSO. A/C/D are the existing UNORDERED_STATE, ORDERED_SECOND and PERMUTED_SECOND arms. MEAN uses the existing local encoder and window mean; MAX applies the same linear head to local-group representations and takes the maximum logit before any sigmoid.\n\nCoverage, per-source metrics, paired source bootstrap, fold/model audit and OOF scores are written under this artifact directory. Visualizations are sparse group-coverage illustrations, not pixel-level localization truth.\n\nNo dense branch, frontend rerun, formal `src` chain, ROI label, quality feature or new architecture is used.\n"""
    (root / "report.md").write_text(text, encoding="utf-8")
    _progress(root, "report", 1, 1, status=str(summary.get("status", "NOT_EVALUATED")))


def run_phase(phase: str, root: Path, *, resume: bool) -> None:
    started = time.time()
    try:
        if phase == "prepare": prepare(root, resume=resume)
        elif phase == "segmentation": segmentation(root, resume=resume)
        elif phase == "features": features(root, resume=resume)
        elif phase == "train": train(root, resume=resume)
        elif phase == "evaluate": evaluate(root, resume=resume)
        elif phase == "report": report(root, resume=resume)
        else: raise ValueError(f"unknown phase: {phase}")
    except Exception as exc:
        _atomic_json(root / "error.json", {"phase": phase, "status": "FAILED", "error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc(), "updated_unix": time.time()})
        _progress(root, phase, 0, 1, status="FAILED", error=f"{type(exc).__name__}: {exc}")
        raise
    _progress(root, phase, 1, 1, status="SUCCESS")
    print(json.dumps({"phase": phase, "status": "SUCCESS", "elapsed_s": time.time() - started}, ensure_ascii=False), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=["prepare", "segmentation", "features", "train", "evaluate", "report", "all"], default="all")
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    phases = ["prepare", "segmentation", "features", "train", "evaluate", "report"] if args.phase == "all" else [args.phase]
    for phase in phases: run_phase(phase, args.output_root, resume=args.resume)


if __name__ == "__main__":
    main()
