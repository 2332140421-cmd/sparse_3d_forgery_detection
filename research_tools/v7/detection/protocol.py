"""Frozen Pair2 population, windows, model loading, and metric helpers.

This module is experiment tooling only.  It never fits a model and never
turns labels or provenance into numeric representation features.
"""

from __future__ import annotations

from collections import defaultdict
import hashlib
import json
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np

from research_tools.v7.normality.protocol import (
    ANCHOR_FRACTIONS,
    MODEL_NAMES,
    TIMESCALE_S,
    GaussianNormality,
    video_score,
)


DATA_ROOT = Path("/root/autodl-tmp/data/sparse_3d_forgery_detection")
PAIR2_ROOT = DATA_ROOT / "derived/v7_genvidbench_core_pilot_v1"
PAIR2_MANIFEST = PAIR2_ROOT / "paired_test/paired_test_manifest.json"
PAIR2_SOURCE_MANIFEST = PAIR2_ROOT / "manifests/test_real_source_pilot.json"
FROZEN_MODEL_PATH = DATA_ROOT / "derived/v7_real_only_normality_pilot_v1/models/normality_models.json"
FROZEN_VAL_SCORE_PATH = DATA_ROOT / "derived/v7_real_only_normality_pilot_v1/scores/val_window_scores.json"
PAIR2_REVISION = "701cafb6f999d7ea0cbf3c354df6177311a4d824"
GENERATORS = ("musev", "svd", "mora", "cogvideo")


def read_json(path: Path) -> object:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_identity(path: Path) -> dict[str, object]:
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def frozen_pair2_population(
    *,
    pair_manifest_path: Path = PAIR2_MANIFEST,
    source_manifest_path: Path = PAIR2_SOURCE_MANIFEST,
    source_count: int = 8,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Return the first frozen sources and their exact official pair rows."""

    sources = read_json(source_manifest_path)
    pairs = read_json(pair_manifest_path)
    if not isinstance(sources, list) or not isinstance(pairs, list):
        raise ValueError("Pair2 manifests must be JSON lists")
    selected = [dict(row) for row in sources[:source_count]]
    if len(selected) != source_count:
        raise ValueError("frozen Pair2 source manifest is shorter than requested")
    source_ids = [str(row["source_id"]) for row in selected]
    if len(set(source_ids)) != len(source_ids):
        raise ValueError("duplicate frozen Pair2 source identity")
    selected_pairs = [dict(row) for row in pairs if str(row["source_id"]) in set(source_ids)]
    expected = {(source_id, generator) for source_id in source_ids for generator in GENERATORS}
    actual = {(str(row["source_id"]), str(row["generator"])) for row in selected_pairs}
    if actual != expected or len(selected_pairs) != len(expected):
        raise ValueError("Pair2 fake lineage does not exactly cover frozen source/generator pairs")
    order = {source_id: index for index, source_id in enumerate(source_ids)}
    selected_pairs.sort(key=lambda row: (order[str(row["source_id"])], GENERATORS.index(str(row["generator"]))))
    for row in selected_pairs:
        if str(row.get("real_source")) != "HD-VG-130M":
            raise ValueError("unexpected Pair2 real source")
        if str(row.get("pair_lineage")) != "shared Pair2 ordinal and official HDVG semantic record":
            raise ValueError("unexpected Pair2 lineage")
    return selected, selected_pairs


def _window_indices(timestamps_s: Sequence[float], anchor_fraction: float) -> list[int]:
    values = np.asarray(timestamps_s, dtype=np.float64)
    if values.ndim != 1 or values.size < 2 or not np.all(np.isfinite(values)) or not np.all(np.diff(values) > 0):
        raise ValueError("timestamps must be finite and strictly increasing")
    if values[-1] - values[0] < TIMESCALE_S:
        return []
    center = float(values[0] + (values[-1] - values[0]) * anchor_fraction)
    left, right = center - TIMESCALE_S / 2.0, center + TIMESCALE_S / 2.0
    return [int(index) for index, value in enumerate(values) if left <= value <= right]


def build_pair2_window_manifest(media_rows: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    """Build unchanged 1.0 s true-time windows for real and fake videos."""

    rows: list[dict[str, object]] = []
    for row in media_rows:
        if row.get("status") != "MEDIA_VALID":
            continue
        source_id = str(row["source_id"])
        role = str(row["role"])
        generator = row.get("generator")
        if role == "real" and generator is not None:
            raise ValueError("real row must not carry generator identity")
        if role == "fake" and str(generator) not in GENERATORS:
            raise ValueError("fake row has unknown generator")
        for fraction in ANCHOR_FRACTIONS:
            indices = _window_indices(row["timestamps_s"], fraction)
            rows.append(
                {
                    "window_id": f"{source_id}::{role}::{generator or 'real'}::anchor-{int(fraction * 100):02d}",
                    "source_id": source_id,
                    "video_id": str(row["video_id"]),
                    "video_path": str(row.get("video_path", "")),
                    "role": role,
                    "generator": generator,
                    "pair_key": row.get("pair_key"),
                    "pair_lineage": row.get("pair_lineage"),
                    "source_identity": row.get("source_identity"),
                    "source_prompt_or_caption": row.get("source_prompt_or_caption"),
                    "official_relative_path": row.get("official_relative_path"),
                    "revision": row.get("revision", PAIR2_REVISION),
                    "anchor_fraction": fraction,
                    "timescale_s": TIMESCALE_S,
                    "frame_indices": indices,
                    "timestamps_s": [float(row["timestamps_s"][index]) for index in indices],
                    "status": "AVAILABLE" if len(indices) >= 2 else "WINDOW_DURATION_UNAVAILABLE",
                    "fake_used": role == "fake",
                }
            )
    return rows


def load_frozen_models(path: Path = FROZEN_MODEL_PATH) -> tuple[dict[str, GaussianNormality], dict[str, object]]:
    """Load, but never fit or alter, the previously frozen real-only models."""

    payload = read_json(path)
    if not isinstance(payload, dict) or payload.get("fake_count") != 0 or payload.get("fitting_population") != "real_train_only":
        raise ValueError("frozen model artifact is not a real-only fit")
    model_rows = payload.get("models")
    if not isinstance(model_rows, dict) or tuple(model_rows) != MODEL_NAMES:
        raise ValueError("frozen model artifact has an unexpected model schema")
    models: dict[str, GaussianNormality] = {}
    for name in MODEL_NAMES:
        row = model_rows[name]
        models[name] = GaussianNormality(
            mean=np.asarray(row["mean"], dtype=np.float64),
            covariance=np.asarray(row["covariance"], dtype=np.float64),
            covariance_type=str(row["covariance_type"]),
            regularization_lambda=float(row["regularization_lambda"]),
            condition_number=float(row["condition_number"]),
            feature_count=int(row["feature_count"]),
        )
    identity = {
        "artifact": file_identity(path),
        "fitting_population": payload["fitting_population"],
        "fake_count": payload["fake_count"],
        "feature_schema": payload.get("feature_schema"),
        "model_names": list(MODEL_NAMES),
        "model_feature_counts": {name: models[name].feature_count for name in MODEL_NAMES},
        "refit": False,
    }
    return models, identity


def score_frontend_results(
    results: Sequence[Mapping[str, object]],
    models: Mapping[str, GaussianNormality],
) -> dict[str, object]:
    """Score frozen frontend observations and aggregate each complete video."""

    window_scores: list[dict[str, object]] = []
    by_video: dict[str, dict[str, object]] = {}
    for result in results:
        window = result.get("window", result)
        video_id = str(window["video_id"])
        video = by_video.setdefault(
            video_id,
            {
                "video_id": video_id,
                "source_id": str(window["source_id"]),
                "role": str(window["role"]),
                "generator": window.get("generator"),
                "window_ids": [],
                "scores": {name: [] for name in models},
                "feature_counts": {name: 0 for name in models},
            },
        )
        video["window_ids"].append(str(window["window_id"]))
        row: dict[str, object] = {
            "window_id": str(window["window_id"]),
            "video_id": video_id,
            "source_id": str(window["source_id"]),
            "role": str(window["role"]),
            "generator": window.get("generator"),
            "status": result.get("status"),
            "coverage": result.get("coverage"),
        }
        for name, model in models.items():
            observations = result.get("observations", {}).get(name, [])
            values = np.asarray([item["values"] for item in observations], dtype=np.float64)
            if values.size == 0:
                scores = np.asarray([], dtype=np.float64)
            else:
                scores = model.score(values)
            video["scores"][name].extend(float(value) for value in scores if np.isfinite(value))
            video["feature_counts"][name] += int(scores.size)
            row[name] = {
                "feature_count": int(scores.size),
                "window_p95": video_score(scores, "p95"),
                "window_median": video_score(scores, "median"),
            }
        window_scores.append(row)
    video_scores: list[dict[str, object]] = []
    for video in by_video.values():
        output = {key: value for key, value in video.items() if key not in {"scores"}}
        output.update(
            {
                name: {
                    "feature_count": int(video["feature_counts"][name]),
                    "video_p95": video_score(video["scores"][name], "p95"),
                    "video_median": video_score(video["scores"][name], "median"),
                }
                for name in models
            }
        )
        video_scores.append(output)
    video_scores.sort(key=lambda row: str(row["video_id"]))
    return {"window_scores": window_scores, "video_scores": video_scores, "aggregation": {"primary": "p95", "sensitivity": "median", "sum_used": False}}


def distribution(values: Iterable[float]) -> dict[str, float | int | None]:
    array = np.asarray(list(values), dtype=np.float64)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return {"count": 0, "median": None, "iqr": None, "p10": None, "p90": None}
    return {
        "count": int(array.size),
        "median": float(np.median(array)),
        "iqr": float(np.percentile(array, 75) - np.percentile(array, 25)),
        "p10": float(np.percentile(array, 10)),
        "p90": float(np.percentile(array, 90)),
    }


def _valid_metric_rows(rows: Sequence[Mapping[str, object]], model: str) -> list[Mapping[str, object]]:
    return [row for row in rows if row.get(model, {}).get("video_p95") is not None]


def auroc(labels: Sequence[int], scores: Sequence[float]) -> float | None:
    y = np.asarray(labels, dtype=np.int64)
    s = np.asarray(scores, dtype=np.float64)
    if y.size == 0 or np.unique(y).size != 2:
        return None
    order = np.argsort(s, kind="mergesort")
    sorted_scores = s[order]
    ranks = np.empty_like(sorted_scores, dtype=np.float64)
    start = 0
    while start < len(sorted_scores):
        end = start + 1
        while end < len(sorted_scores) and sorted_scores[end] == sorted_scores[start]:
            end += 1
        ranks[start:end] = (start + end - 1) / 2.0 + 1.0
        start = end
    rank_sum = float(np.sum(ranks[y[order] == 1]))
    positive = int(np.sum(y == 1))
    negative = int(np.sum(y == 0))
    return (rank_sum - positive * (positive + 1) / 2.0) / (positive * negative)


def auprc(labels: Sequence[int], scores: Sequence[float]) -> float | None:
    y = np.asarray(labels, dtype=np.int64)
    s = np.asarray(scores, dtype=np.float64)
    positive = int(np.sum(y == 1))
    if y.size == 0 or positive == 0 or positive == y.size:
        return None
    order = np.argsort(-s, kind="mergesort")
    y = y[order]
    tp = np.cumsum(y == 1)
    fp = np.cumsum(y == 0)
    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / positive
    return float(np.sum(precision[y == 1] * np.diff(np.concatenate(([0.0], recall[y == 1])))))


def paired_deltas(rows: Sequence[Mapping[str, object]], model: str) -> dict[str, object]:
    grouped: dict[tuple[str, str], dict[str, float]] = defaultdict(dict)
    for row in rows:
        source = str(row["source_id"])
        role = str(row["role"])
        generator = str(row.get("generator") or "real")
        value = row.get(model, {}).get("video_p95")
        if value is not None:
            grouped[(source, generator)][role] = float(value)
    deltas = []
    by_generator: dict[str, list[float]] = defaultdict(list)
    for (source, generator), values in sorted(grouped.items()):
        if generator != "real" and "real" in grouped.get((source, "real"), {}) and "fake" in values:
            delta = values["fake"] - grouped[(source, "real")]["real"]
            deltas.append(delta)
            by_generator[generator].append(delta)
    return {
        "count": len(deltas),
        "median": float(np.median(deltas)) if deltas else None,
        "iqr": float(np.percentile(deltas, 75) - np.percentile(deltas, 25)) if deltas else None,
        "positive_fraction": float(np.mean(np.asarray(deltas) > 0)) if deltas else None,
        "by_generator": {
            name: {
                "count": len(values),
                "median": float(np.median(values)) if values else None,
                "iqr": float(np.percentile(values, 75) - np.percentile(values, 25)) if values else None,
                "positive_fraction": float(np.mean(np.asarray(values) > 0)) if values else None,
            }
            for name, values in sorted(by_generator.items())
        },
    }


def bootstrap_source_cluster(rows: Sequence[Mapping[str, object]], model: str, *, seed: int = 20260908, repeats: int = 2000) -> dict[str, object]:
    """Bootstrap AUROC by source video, never by windows/components."""

    valid = _valid_metric_rows(rows, model)
    source_ids = sorted({str(row["source_id"]) for row in valid})
    if len(source_ids) < 4:
        return {"status": "INSUFFICIENT_SOURCE_CLUSTER_N", "source_count": len(source_ids)}
    grouped = {source: [row for row in valid if str(row["source_id"]) == source] for source in source_ids}
    rng = np.random.default_rng(seed)
    values: list[float] = []
    for _ in range(repeats):
        sample_ids = rng.choice(source_ids, size=len(source_ids), replace=True)
        sample = [row for source in sample_ids for row in grouped[str(source)]]
        labels = [int(str(row["role"]) == "fake") for row in sample]
        scores = [float(row[model]["video_p95"]) for row in sample]
        value = auroc(labels, scores)
        if value is not None:
            values.append(value)
    return {
        "status": "OK" if values else "NO_VALID_RESAMPLE",
        "source_count": len(source_ids),
        "repeats": repeats,
        "unit": "source_video",
        "lower_95": float(np.percentile(values, 2.5)) if values else None,
        "upper_95": float(np.percentile(values, 97.5)) if values else None,
    }


def calibrate_real_validation_threshold(path: Path = FROZEN_VAL_SCORE_PATH) -> dict[str, object]:
    """Use only frozen real-validation video scores to set the operating point."""

    rows = read_json(path)
    if not isinstance(rows, list):
        raise ValueError("frozen validation scores must be a list")
    by_source: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        if str(row.get("role")) != "real_val":
            continue
        source = str(row["source_id"])
        for model in MODEL_NAMES:
            value = row.get(model, {}).get("video_p95")
            if value is not None:
                by_source[source][model].append(float(value))
    thresholds = {}
    source_count = len(by_source)
    for model in MODEL_NAMES:
        values = [float(np.median(row[model])) for row in by_source.values() if row.get(model)]
        thresholds[model] = {
            "threshold": float(np.percentile(values, 95)) if values else None,
            "source_count": len(values),
            "aggregation": "source median of frozen real_val window p95 scores",
        }
    return {"label": "REAL-ONLY CALIBRATED THRESHOLD", "percentile": 95, "source_count": source_count, "source": file_identity(path), "models": thresholds}
