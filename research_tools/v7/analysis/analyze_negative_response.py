"""Explain the negative fake response using frozen V7 artifacts only.

This module is a CPU-only descriptive diagnostic. It never fits or changes the
normality detector, never uses fake data as a reference cloud, and never writes
large artifacts into the repository.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np

from sparse3d_forgery.particle_sequence import load_particle_sequence

from research_tools.v7.detection.evaluate_unpaired_fake_response import aggregate_quality
from research_tools.v7.detection.protocol import (
    DATA_ROOT,
    FROZEN_MODEL_PATH,
    MODEL_NAMES,
    auroc,
    file_identity,
    load_frozen_models,
    read_json,
    sha256_file,
    write_json,
)
from research_tools.v7.detection.run_unpaired_fake_response import (
    FAKE_MANIFEST,
    FROZEN_MODEL_SHA256,
    validate_fake_identity,
)

REAL_ROOT = DATA_ROOT / "derived/v7_real_only_normality_pilot_v1"
FAKE_ROOT = DATA_ROOT / "derived/v7_unpaired_fake_response_v1"
OUTPUT_ROOT = DATA_ROOT / "derived/v7_negative_response_mechanism_v1"
REAL_FRONTEND = REAL_ROOT / "frontend/window_results.json"
FAKE_FRONTEND = FAKE_ROOT / "frontend/window_results.json"
EPSILON = 1e-12
BOOTSTRAP_SEED = 20260908
BOOTSTRAP_REPEATS = 10_000
QUALITY_KEYS = ("geometry_coverage", "tracking_persistence", "component_success_fraction", "s_valid_fraction")


def _finite(values: Iterable[float]) -> np.ndarray:
    array = np.asarray(list(values), dtype=np.float64)
    return array[np.isfinite(array)]


def _stats(values: Iterable[float]) -> dict[str, float | int | None]:
    array = _finite(values)
    if array.size == 0:
        return {"N": 0, "median": None, "IQR": None, "p10": None, "p75": None, "p90": None, "p95": None, "min": None, "max": None}
    return {
        "N": int(array.size),
        "median": float(np.median(array)),
        "IQR": float(np.percentile(array, 75) - np.percentile(array, 25)),
        "p10": float(np.percentile(array, 10)),
        "p75": float(np.percentile(array, 75)),
        "p90": float(np.percentile(array, 90)),
        "p95": float(np.percentile(array, 95)),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
    }


def _mad(values: Iterable[float]) -> float | None:
    array = _finite(values)
    if array.size == 0:
        return None
    median = float(np.median(array))
    return float(np.median(np.abs(array - median)))


def _skewness(values: Iterable[float]) -> float | None:
    array = _finite(values)
    if array.size < 2:
        return None
    centered = array - np.mean(array)
    scale = float(np.std(array))
    if scale == 0.0:
        return 0.0
    return float(np.mean(centered**3) / (scale**3))


def _spearman(x: Sequence[float], y: Sequence[float]) -> float | None:
    if len(x) != len(y) or len(x) < 2:
        return None
    xa, ya = _finite(x), _finite(y)
    if xa.size != len(x) or ya.size != len(y):
        return None

    def ranks(values: np.ndarray) -> np.ndarray:
        order = np.argsort(values, kind="mergesort")
        result = np.empty(values.size, dtype=np.float64)
        start = 0
        while start < values.size:
            end = start + 1
            while end < values.size and values[order[end]] == values[order[start]]:
                end += 1
            result[order[start:end]] = (start + end - 1) / 2.0 + 1.0
            start = end
        return result

    xr, yr = ranks(xa), ranks(ya)
    if np.std(xr) == 0 or np.std(yr) == 0:
        return None
    return float(np.corrcoef(xr, yr)[0, 1])


def _video_id(window: Mapping[str, object]) -> str:
    role = str(window.get("role"))
    return str(window.get("video_id") if role == "fake" else window.get("source_id"))


def _metadata(window: Mapping[str, object]) -> dict[str, object]:
    raw_role = str(window.get("role"))
    role = "fake" if raw_role == "fake" else "real"
    return {
        "video_id": _video_id(window),
        "source_id": str(window.get("source_id")),
        "role": role,
        "source_role": raw_role,
        "generator": window.get("generator") if role == "fake" else None,
        "pair2_ordinal": str(window.get("pair_key")) if role == "fake" and window.get("pair_key") is not None else None,
    }


def _load_inputs() -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    real_rows = read_json(REAL_FRONTEND)
    fake_rows = read_json(FAKE_FRONTEND)
    manifest_rows = read_json(FAKE_MANIFEST)
    if not isinstance(real_rows, list) or not isinstance(fake_rows, list):
        raise ValueError("frozen frontend artifacts must be JSON lists")
    validate_fake_identity(manifest_rows)
    train = [dict(row) for row in real_rows if str(row.get("window", {}).get("role")) == "real_train"]
    val = [dict(row) for row in real_rows if str(row.get("window", {}).get("role")) == "real_val"]
    fake = [dict(row) for row in fake_rows if str(row.get("window", {}).get("role")) == "fake"]
    if len(train) != 72 or len(val) != 24 or len(fake) != 30:
        raise ValueError(f"unexpected frozen population windows train={len(train)} val={len(val)} fake={len(fake)}")
    if len({_video_id(row["window"]) for row in val}) != 8:
        raise ValueError("real_val population is not eight sources")
    if len({_video_id(row["window"]) for row in fake}) != 10:
        raise ValueError("fake population is not ten videos")
    return train, val, fake


def _observation_data(rows: Sequence[Mapping[str, object]]) -> dict[str, dict[str, object]]:
    data: dict[str, dict[str, object]] = {}
    for row in rows:
        window = row["window"]
        metadata = _metadata(window)
        entry = data.setdefault(
            str(metadata["video_id"]),
            {"metadata": metadata, "observations": {name: [] for name in MODEL_NAMES}},
        )
        for model in MODEL_NAMES:
            for item in row.get("observations", {}).get(model, []):
                values = np.asarray(item.get("values", []), dtype=np.float64)
                expected = 8 if model == "M3_delta_s_delta2_s" else 4
                if values.shape != (expected,) or not np.all(np.isfinite(values)):
                    continue
                entry["observations"][model].append(values)
    return data


def _train_statistics(train_data: Mapping[str, Mapping[str, object]]) -> dict[str, dict[str, np.ndarray]]:
    result: dict[str, dict[str, np.ndarray]] = {}
    for model in MODEL_NAMES:
        values = [value for entry in train_data.values() for value in entry["observations"][model]]
        array = np.asarray(values, dtype=np.float64)
        if array.ndim != 2 or array.shape[0] == 0:
            raise ValueError(f"no frozen real_train observations for {model}")
        mean = np.mean(array, axis=0)
        std = np.std(array, axis=0, ddof=1) if array.shape[0] > 1 else np.ones(array.shape[1], dtype=np.float64)
        std = np.where(np.isfinite(std) & (std > EPSILON), std, EPSILON)
        result[model] = {"mean": mean, "std": std, "cloud": (array - mean) / std, "count": np.asarray([array.shape[0]])}
    return result


def _knn(query: np.ndarray, cloud: np.ndarray, k: int) -> np.ndarray:
    if query.ndim != 2 or query.shape[0] == 0:
        return np.empty(0, dtype=np.float64)
    if cloud.ndim != 2 or cloud.shape[0] < k:
        raise ValueError("reference cloud is smaller than requested k")
    output: list[np.ndarray] = []
    for start in range(0, query.shape[0], 256):
        block = query[start:start + 256]
        distances = np.sqrt(np.sum((block[:, None, :] - cloud[None, :, :]) ** 2, axis=2))
        nearest = np.partition(distances, kth=k - 1, axis=1)[:, :k]
        output.append(np.mean(nearest, axis=1))
    return np.concatenate(output)


def _dispersion(values: np.ndarray) -> dict[str, object]:
    if values.ndim != 2 or values.shape[0] == 0:
        return {"N": 0, "per_dimension_MAD": [], "per_dimension_IQR": [], "norm_MAD": None, "norm_IQR": None, "covariance_trace": None}
    norms = np.linalg.norm(values, axis=1)
    covariance_trace = None
    if values.shape[0] > 1:
        covariance_trace = float(np.trace(np.atleast_2d(np.cov(values, rowvar=False, ddof=1))))
    return {
        "N": int(values.shape[0]),
        "per_dimension_MAD": [_mad(values[:, index]) for index in range(values.shape[1])],
        "per_dimension_IQR": [_stats(values[:, index])["IQR"] for index in range(values.shape[1])],
        "norm_MAD": _mad(norms),
        "norm_IQR": _stats(norms)["IQR"],
        "covariance_trace": covariance_trace,
    }


def _model_video_metrics(entry: Mapping[str, object], model, train_stats: Mapping[str, np.ndarray]) -> dict[str, object]:
    values = np.asarray(entry["observations"], dtype=np.float64)
    if values.ndim != 2 or values.shape[0] == 0:
        empty = _stats([])
        return {"N": 0, "norm": empty, "abs_dimensions": [], "dispersion": _dispersion(values), "center_distance": empty, "train_standardized_distance": empty, "mahalanobis_squared": empty, "mahalanobis_distance": empty, "knn1_distance": empty, "knn5_distance": empty}
    norms = np.linalg.norm(values, axis=1)
    center = np.linalg.norm(values - model.mean[None, :], axis=1)
    standardized = np.linalg.norm((values - train_stats["mean"][None, :]) / train_stats["std"][None, :], axis=1)
    mahalanobis_squared = model.score(values)
    mahalanobis_distance = np.sqrt(np.maximum(mahalanobis_squared, 0.0))
    standardized_values = (values - train_stats["mean"][None, :]) / train_stats["std"][None, :]
    return {
        "N": int(values.shape[0]),
        "norm": _stats(norms),
        "abs_dimensions": [_stats(np.abs(values[:, index])) for index in range(values.shape[1])],
        "dispersion": _dispersion(values),
        "center_distance": _stats(center),
        "train_standardized_distance": _stats(standardized),
        "mahalanobis_squared": _stats(mahalanobis_squared),
        "mahalanobis_distance": _stats(mahalanobis_distance),
        "knn1_distance": _stats(_knn(standardized_values, train_stats["cloud"], 1)),
        "knn5_distance": _stats(_knn(standardized_values, train_stats["cloud"], 5)),
    }


def _cluster_bootstrap(rows: Sequence[Mapping[str, object]], key: str) -> dict[str, object]:
    valid = [row for row in rows if row.get(key) is not None and np.isfinite(float(row[key]))]
    real_groups: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    fake_groups: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    for row in valid:
        if row["role"] == "real":
            real_groups[str(row["video_id"])].append(row)
        elif row["role"] == "fake":
            fake_groups[str(row.get("pair2_ordinal"))].append(row)
    if len(real_groups) != 8 or len(fake_groups) != 5:
        return {"status": "INSUFFICIENT_CLUSTER_N", "real_clusters": len(real_groups), "fake_clusters": len(fake_groups)}
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    real_ids, fake_ids = sorted(real_groups), sorted(fake_groups)
    values: list[float] = []
    for _ in range(BOOTSTRAP_REPEATS):
        sample_real = [row for index in rng.integers(0, len(real_ids), len(real_ids)) for row in real_groups[real_ids[int(index)]]]
        sample_fake = [row for index in rng.integers(0, len(fake_ids), len(fake_ids)) for row in fake_groups[fake_ids[int(index)]]]
        sample = sample_real + sample_fake
        labels = [int(row["role"] == "fake") for row in sample]
        scores = [float(row[key]) for row in sample]
        value = auroc(labels, scores)
        if value is not None:
            values.append(float(value))
    return {
        "status": "OK" if values else "NO_VALID_RESAMPLE",
        "seed": BOOTSTRAP_SEED,
        "repeats": BOOTSTRAP_REPEATS,
        "real_cluster_unit": "Vript real_val source video",
        "fake_cluster_unit": "Pair2 ordinal preserving generator variants",
        "real_clusters": 8,
        "fake_clusters": 5,
        "lower_95": float(np.percentile(values, 2.5)) if values else None,
        "upper_95": float(np.percentile(values, 97.5)) if values else None,
    }


def _effect(rows: Sequence[Mapping[str, object]], key: str) -> dict[str, object]:
    real = [float(row[key]) for row in rows if row["role"] == "real" and row.get(key) is not None and np.isfinite(float(row[key]))]
    fake = [float(row[key]) for row in rows if row["role"] == "fake" and row.get(key) is not None and np.isfinite(float(row[key]))]
    labels = [0] * len(real) + [1] * len(fake)
    auc = auroc(labels, real + fake) if real and fake else None
    return {
        "real": _stats(real),
        "fake": _stats(fake),
        "svd": _stats(float(row[key]) for row in rows if row["role"] == "fake" and row.get("generator") == "svd" and row.get(key) is not None and np.isfinite(float(row[key]))),
        "cogvideo": _stats(float(row[key]) for row in rows if row["role"] == "fake" and row.get("generator") == "cogvideo" and row.get(key) is not None and np.isfinite(float(row[key]))),
        "AUROC_fake_higher": auc,
        "Cliffs_delta_fake_higher": 2.0 * auc - 1.0 if auc is not None else None,
        "cluster_bootstrap_95": _cluster_bootstrap(rows, key),
    }


def _metric_summary(rows: Sequence[Mapping[str, object]], keys: Sequence[str]) -> dict[str, object]:
    return {key: _effect(rows, key) for key in keys}


def _compression_stats(values: Sequence[Mapping[str, object]]) -> dict[str, object]:
    keys = ("p10", "p90", "range", "MAD", "skewness", "pair_temporal_MAD", "pair_temporal_IQR")
    return {key: _stats(float(row[key]) for row in values if row.get(key) is not None and np.isfinite(float(row[key]))) for key in keys}


def _compression_for_window(result: Mapping[str, object]) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    prefix = Path(str(result.get("particle_prefix", "")))
    if not prefix.with_suffix(".json").is_file() or not prefix.with_suffix(".npz").is_file():
        return [], []
    sequence = load_particle_sequence(prefix)
    component_time: list[dict[str, object]] = []
    component_temporal: list[dict[str, object]] = []
    for component_row in result.get("component_rows", []):
        members = np.asarray(component_row.get("members", []), dtype=np.int64)
        if members.size < 3:
            continue
        pair_series: dict[tuple[int, int], list[float]] = defaultdict(list)
        for time_index in range(sequence.xyz.shape[0]):
            valid_members = members[sequence.geometry_validity[time_index, members]]
            if valid_members.size < 3:
                continue
            points = sequence.xyz[time_index, valid_members].astype(np.float64)
            distances = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=2)
            values = distances[np.triu_indices(valid_members.size, k=1)]
            scale = float(np.median(values))
            if not np.isfinite(scale) or scale <= 0:
                continue
            normalized = values / scale
            component_time.append({
                "p10": float(np.percentile(normalized, 10)),
                "p90": float(np.percentile(normalized, 90)),
                "range": float(np.percentile(normalized, 90) - np.percentile(normalized, 10)),
                "MAD": _mad(normalized),
                "skewness": _skewness(normalized),
            })
            upper_i, upper_j = np.triu_indices(valid_members.size, k=1)
            for pair_i, pair_j, value in zip(upper_i, upper_j, normalized):
                pair = (int(valid_members[pair_i]), int(valid_members[pair_j]))
                pair_series[pair].append(float(value))
        temporal_mad = [_mad(values) for values in pair_series.values() if len(values) >= 2]
        temporal_iqr = [_stats(values)["IQR"] for values in pair_series.values() if len(values) >= 2]
        if component_time and temporal_mad:
            component_temporal.append({"pair_temporal_MAD": float(np.median(_finite(temporal_mad))), "pair_temporal_IQR": float(np.median(_finite(temporal_iqr)))})
    if component_time and component_temporal:
        for row in component_time:
            row.update({
                "pair_temporal_MAD": float(np.median([value["pair_temporal_MAD"] for value in component_temporal])),
                "pair_temporal_IQR": float(np.median([value["pair_temporal_IQR"] for value in component_temporal])),
            })
    return component_time, component_temporal


def _compression_diagnostic(rows: Sequence[Mapping[str, object]]) -> tuple[list[dict[str, object]], dict[str, object]]:
    by_video: dict[str, list[dict[str, object]]] = defaultdict(list)
    temporal_by_video: dict[str, list[dict[str, object]]] = defaultdict(list)
    for result in rows:
        metadata = _metadata(result["window"])
        component_time, component_temporal = _compression_for_window(result)
        by_video[str(metadata["video_id"])].extend(component_time)
        temporal_by_video[str(metadata["video_id"])].extend(component_temporal)
    output: list[dict[str, object]] = []
    for video_id, values in sorted(by_video.items()):
        temporal = temporal_by_video.get(video_id, [])
        row = {"video_id": video_id, **_metadata(next(result["window"] for result in rows if _video_id(result["window"]) == video_id)), "component_time_count": len(values), "component_temporal_count": len(temporal), **_compression_stats(values)}
        for key in ("pair_temporal_MAD", "pair_temporal_IQR"):
            row[key] = _stats(float(value[key]) for value in temporal if value.get(key) is not None)
        output.append(row)
    expected_videos = len({_video_id(row["window"]) for row in rows})
    status = "IDENTIFIABLE_WITH_CURRENT_ARTIFACTS" if output else "REPRESENTATION_COMPRESSION_NOT_IDENTIFIABLE_WITH_CURRENT_ARTIFACTS"
    return output, {"status": status, "videos_with_pairwise_diagnostics": len(output), "videos_expected": expected_videos, "component_time_rows": sum(len(values) for values in by_video.values())}


def _quality_map(real_val: Sequence[Mapping[str, object]], fake: Sequence[Mapping[str, object]]) -> dict[str, dict[str, object]]:
    values = aggregate_quality(real_val, role="real") + aggregate_quality(fake, role="fake")
    return {str(row["video_id"]): dict(row) for row in values}


def _score_map() -> dict[str, dict[str, object]]:
    values = read_json(FAKE_ROOT / "metrics/per_video_response.json")
    if not isinstance(values, list) or len(values) != 18:
        raise ValueError("frozen per-video response table must contain 18 rows")
    return {str(row["video_id"]): dict(row) for row in values}


def _flatten_model_metrics(details: Mapping[str, object], model: str, prefix: str) -> dict[str, object]:
    model_detail = details[model]
    return {
        f"{prefix}_norm_median": model_detail["norm"]["median"],
        f"{prefix}_norm_p95": model_detail["norm"]["p95"],
        f"{prefix}_dispersion": model_detail["dispersion"]["norm_IQR"],
        f"{prefix}_center_distance_median": model_detail["center_distance"]["median"],
        f"{prefix}_standardized_distance_median": model_detail["train_standardized_distance"]["median"],
        f"{prefix}_mahalanobis_median": model_detail["mahalanobis_distance"]["median"],
        f"{prefix}_knn1_distance": model_detail["knn1_distance"]["median"],
        f"{prefix}_knn5_distance": model_detail["knn5_distance"]["median"],
    }


def _per_video_rows(data: Mapping[str, Mapping[str, object]], models, train_stats, quality, scores) -> tuple[list[dict[str, object]], dict[str, object]]:
    detail_rows: list[dict[str, object]] = []
    flat_rows: list[dict[str, object]] = []
    for video_id, entry in sorted(data.items()):
        details = {model: _model_video_metrics({"observations": entry["observations"][model]}, models[model], train_stats[model]) for model in MODEL_NAMES}
        m1 = details["M1_delta_s"]["norm"]["median"]
        m2 = details["M2_delta2_s"]["norm"]["median"]
        roughness = float(m2 / (m1 + EPSILON)) if m1 is not None and m2 is not None else None
        meta = dict(entry["metadata"])
        q = quality.get(video_id, {})
        score = scores.get(video_id, {})
        row = {**meta, **{key: q.get(key) for key in QUALITY_KEYS}, "s_valid_fraction": q.get("s_valid_fraction"), "M1_score": score.get("M1_video_score"), "M2_score": score.get("M2_video_score"), "M3_score": score.get("M3_video_score"), "relative_roughness": roughness, "details": details}
        row.update(_flatten_model_metrics(details, "M1_delta_s", "delta_s"))
        row.update(_flatten_model_metrics(details, "M2_delta2_s", "delta2_s"))
        row.update(_flatten_model_metrics(details, "M3_delta_s_delta2_s", "M3"))
        detail_rows.append(row)
        flat = {key: value for key, value in row.items() if key != "details"}
        flat["delta_s_norm_median"] = details["M1_delta_s"]["norm"]["median"]
        flat["delta_s_norm_p95"] = details["M1_delta_s"]["norm"]["p95"]
        flat["delta2_s_norm_median"] = details["M2_delta2_s"]["norm"]["median"]
        flat["delta2_s_norm_p95"] = details["M2_delta2_s"]["norm"]["p95"]
        flat["delta_s_dispersion"] = details["M1_delta_s"]["dispersion"]["norm_IQR"]
        flat["delta2_s_dispersion"] = details["M2_delta2_s"]["dispersion"]["norm_IQR"]
        flat["component_success"] = q.get("component_success_fraction")
        flat["M1_center_distance_median"] = details["M1_delta_s"]["center_distance"]["median"]
        flat["M2_center_distance_median"] = details["M2_delta2_s"]["center_distance"]["median"]
        flat["M3_center_distance_median"] = details["M3_delta_s_delta2_s"]["center_distance"]["median"]
        flat["M1_knn1_distance"] = details["M1_delta_s"]["knn1_distance"]["median"]
        flat["M1_knn5_distance"] = details["M1_delta_s"]["knn5_distance"]["median"]
        flat["M2_knn1_distance"] = details["M2_delta2_s"]["knn1_distance"]["median"]
        flat["M2_knn5_distance"] = details["M2_delta2_s"]["knn5_distance"]["median"]
        flat["M3_knn1_distance"] = details["M3_delta_s_delta2_s"]["knn1_distance"]["median"]
        flat["M3_knn5_distance"] = details["M3_delta_s_delta2_s"]["knn5_distance"]["median"]
        flat_rows.append(flat)
    return detail_rows, {"flat_rows": flat_rows, "detail_rows": detail_rows}


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        return
    preferred = [
        "video_id", "role", "generator", "pair2_ordinal", "geometry_coverage", "tracking_persistence", "component_success_fraction", "component_success", "s_valid_fraction", "M1_score", "M2_score", "M3_score",
        "delta_s_norm_median", "delta_s_norm_p95", "delta2_s_norm_median", "delta2_s_norm_p95", "relative_roughness", "delta_s_dispersion", "delta2_s_dispersion",
        "M1_center_distance_median", "M2_center_distance_median", "M3_center_distance_median", "M1_knn1_distance", "M1_knn5_distance", "M2_knn1_distance", "M2_knn5_distance", "M3_knn1_distance", "M3_knn5_distance",
        "delta_s_center_distance_median", "delta2_s_center_distance_median", "delta_s_knn1_distance", "delta_s_knn5_distance", "delta2_s_knn1_distance", "delta2_s_knn5_distance",
    ]
    fields = [field for field in preferred if any(field in row for row in rows)]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: row.get(field) for field in fields} for row in rows)


def _dimension_summary(detail_rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for model in ("M1_delta_s", "M2_delta2_s"):
        dimensions = []
        for index in range(4):
            key = "delta_s" if model == "M1_delta_s" else "delta2_s"
            values = []
            for row in detail_rows:
                dimension_values = row["details"][model]["abs_dimensions"]
                value = dimension_values[index]["median"] if index < len(dimension_values) else None
                if value is not None:
                    values.append((row["role"], row.get("generator"), value))
            dimension = {"dimension": index, "semantic": ("mean", "std", "p25", "p75")[index], "groups": {}}
            for group in ("real", "fake", "svd", "cogvideo"):
                selected = [value for role, generator, value in values if (group == role) or (group in ("svd", "cogvideo") and generator == group)]
                dimension["groups"][group] = _stats(selected)
            dimension["absolute_component"] = key
            dimensions.append(dimension)
        result[model] = {"dimensions": dimensions, "definition": "video-level median of per-observation absolute derivative coordinate"}
    return result


def _write_dimension_csv(path: Path, summary: Mapping[str, object]) -> None:
    rows: list[dict[str, object]] = []
    for model, payload in summary.items():
        for dimension in payload["dimensions"]:
            for group, stats in dimension["groups"].items():
                rows.append(
                    {
                        "model": model,
                        "dimension": dimension["dimension"],
                        "semantic": dimension["semantic"],
                        "absolute_component": dimension["absolute_component"],
                        "group": group,
                        **stats,
                    }
                )
    fields = ["model", "dimension", "semantic", "absolute_component", "group", "N", "median", "IQR", "p10", "p75", "p90", "p95", "min", "max"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _quality_correlations(flat_rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    mechanism_keys = ("delta_s_norm_median", "delta2_s_norm_median", "relative_roughness", "delta_s_center_distance_median", "delta2_s_center_distance_median", "delta_s_knn1_distance", "delta2_s_knn1_distance")
    output: dict[str, object] = {}
    for metric in mechanism_keys:
        output[metric] = {}
        for quality_key in ("geometry_coverage", "tracking_persistence"):
            pairs = [(row.get(metric), row.get(quality_key)) for row in flat_rows if row.get(metric) is not None and row.get(quality_key) is not None]
            output[metric][quality_key] = {"N": len(pairs), "rho": _spearman([float(pair[0]) for pair in pairs], [float(pair[1]) for pair in pairs])}
            fake_pairs = [(row.get(metric), row.get(quality_key)) for row in flat_rows if row["role"] == "fake" and row.get(metric) is not None and row.get(quality_key) is not None]
            output[metric][f"fake_only_{quality_key}"] = {"N": len(fake_pairs), "rho": _spearman([float(pair[0]) for pair in fake_pairs], [float(pair[1]) for pair in fake_pairs])}
    return output


def _reconcile_aggregation() -> dict[str, object]:
    metrics = read_json(REAL_ROOT / "metrics/normality_metrics.json")
    response = read_json(FAKE_ROOT / "metrics/unpaired_response_metrics.json")
    result: dict[str, object] = {
        "observation_level": "Each finite raw ΔS/Δ²S (or concatenated M3) component-time vector; frozen Gaussian scores are squared Mahalanobis values per observation.",
        "window_level": "For each window, component-time scores are summarized by p95 (primary) and median (sensitivity).",
        "real_only_source_level_report": "The real-only report's source_level_val uses the median of the three window medians; its normality_metrics also stores window p95 distributions separately.",
        "unpaired_primary_source_level": "The unpaired response uses the median of each source's window p95 values to match the frozen primary p95 operation.",
        "not_identical_display_layers": True,
        "real_only_source_level_values": {},
        "unpaired_real_primary_values": {},
    }
    for model in MODEL_NAMES:
        source_values = [row[model]["source_median_score"] for row in metrics["source_level_val"].values() if row.get(model, {}).get("source_median_score") is not None]
        result["real_only_source_level_values"][model] = _stats(source_values)
        result["unpaired_real_primary_values"][model] = response["models"][model]["real_distribution"]
    return result


def _mechanism_interpretation(summary: Mapping[str, object]) -> dict[str, object]:
    activity = summary["activity"]
    def median(metric: str, group: str) -> float | None:
        return activity[metric][group]["median"]
    low_dirs = []
    for metric in ("delta_s_norm_median", "delta2_s_norm_median"):
        real = median(metric, "real")
        fake = median(metric, "fake")
        svd = median(metric, "svd")
        cog = median(metric, "cogvideo")
        low_dirs.append(real is not None and fake is not None and fake < real and svd is not None and cog is not None and svd < real and cog < real)
    rough_real = median("relative_roughness", "real")
    rough_fake = median("relative_roughness", "fake")
    first_real = median("delta_s_norm_median", "real")
    first_fake = median("delta_s_norm_median", "fake")
    low_activity = all(low_dirs)
    over_smooth = rough_real is not None and rough_fake is not None and rough_fake < rough_real and first_real is not None and first_fake is not None and first_fake >= 0.8 * first_real
    center = summary["distance_diagnostics"]
    gaussian_mismatch = any(
        center[model]["center_distance"]["fake"]["median"] is not None
        and center[model]["center_distance"]["real"]["median"] is not None
        and center[model]["center_distance"]["fake"]["median"] < center[model]["center_distance"]["real"]["median"]
        and center[model]["knn1_distance"]["fake"]["median"] is not None
        and center[model]["knn1_distance"]["real"]["median"] is not None
        and center[model]["knn1_distance"]["fake"]["median"] >= center[model]["knn1_distance"]["real"]["median"]
        for model in MODEL_NAMES
    )
    manifold = all(
        center[model]["mahalanobis_distance"]["fake"]["median"] is not None
        and center[model]["mahalanobis_distance"]["real"]["median"] is not None
        and center[model]["mahalanobis_distance"]["fake"]["median"] <= center[model]["mahalanobis_distance"]["real"]["median"]
        and center[model]["knn1_distance"]["fake"]["median"] is not None
        and center[model]["knn1_distance"]["real"]["median"] is not None
        and center[model]["knn1_distance"]["fake"]["median"] <= center[model]["knn1_distance"]["real"]["median"]
        for model in MODEL_NAMES
    )
    compression = summary["compression"]
    return {
        "LOW_STRUCTURAL_ACTIVITY_SUPPORTED": bool(low_activity),
        "OVER_SMOOTH_EVOLUTION_SUPPORTED": bool(over_smooth and not low_activity),
        "GAUSSIAN_CENTER_MISMATCH_SUPPORTED": bool(gaussian_mismatch),
        "CURRENT_REPRESENTATION_COLLAPSE_SUSPECTED": bool(compression.get("status") == "IDENTIFIABLE_WITH_CURRENT_ARTIFACTS" and not low_activity),
        "FAKE_LIES_INSIDE_CURRENT_REAL_FEATURE_MANIFOLD": bool(manifold),
        "MECHANISM_UNRESOLVED": True,
        "rules": {
            "activity": "Both derivative-magnitude video medians are lower for fake and both generators than real.",
            "roughness": "R is lower while first-order activity remains at least 0.8 of real; this is only a descriptive gate.",
            "gaussian_mismatch": "At least one model has lower center distance but non-lower kNN-1 distance.",
            "manifold": "All three models have non-higher frozen Mahalanobis distance and kNN-1 distance for fake.",
        },
    }


def analyze(output: Path = OUTPUT_ROOT) -> dict[str, object]:
    train_rows, real_val_rows, fake_rows = _load_inputs()
    models, model_identity = load_frozen_models(FROZEN_MODEL_PATH)
    if sha256_file(FROZEN_MODEL_PATH) != FROZEN_MODEL_SHA256:
        raise RuntimeError("FROZEN_MODEL_IDENTITY_MISMATCH")
    train_data = _observation_data(train_rows)
    eval_data = _observation_data(real_val_rows + fake_rows)
    train_stats = _train_statistics(train_data)
    quality = _quality_map(real_val_rows, fake_rows)
    scores = _score_map()
    detail_rows, row_payload = _per_video_rows(eval_data, models, train_stats, quality, scores)
    flat_rows = row_payload["flat_rows"]
    activity_keys = ("delta_s_norm_median", "delta_s_norm_p95", "delta2_s_norm_median", "delta2_s_norm_p95", "relative_roughness", "delta_s_dispersion", "delta2_s_dispersion")
    center_keys = ("delta_s_center_distance_median", "delta2_s_center_distance_median", "M3_center_distance_median", "delta_s_knn1_distance", "delta_s_knn5_distance", "delta2_s_knn1_distance", "delta2_s_knn5_distance", "M3_knn1_distance", "M3_knn5_distance")
    activity = _metric_summary(flat_rows, activity_keys)
    distance_rows = []
    for row in detail_rows:
        distance_rows.append({
            "video_id": row["video_id"], "role": row["role"], "generator": row["generator"], "pair2_ordinal": row["pair2_ordinal"],
            **{f"{model}_{metric}": row["details"][model][metric]["median"] for model in MODEL_NAMES for metric in ("center_distance", "mahalanobis_distance", "knn1_distance", "knn5_distance")},
        })
    distance_diagnostics: dict[str, object] = {}
    for model in MODEL_NAMES:
        distance_diagnostics[model] = {
            metric: _effect([{**row, "value": row[f"{model}_{metric}"]} for row in distance_rows], "value")
            for metric in ("center_distance", "mahalanobis_distance", "knn1_distance", "knn5_distance")
        }
    compression_rows, compression = _compression_diagnostic(real_val_rows + fake_rows)
    compression_effects = {key: _effect([{**row, key: row[key]["median"] if isinstance(row[key], Mapping) else row[key]} for row in compression_rows], key) for key in ("p10", "p90", "range", "MAD", "skewness", "pair_temporal_MAD", "pair_temporal_IQR")}
    summary: dict[str, object] = {
        "status": "MECHANISM_ANALYSIS_COMPLETE",
        "frozen_model": {**model_identity, "sha256": FROZEN_MODEL_SHA256, "fake_count": 0, "refit": False},
        "population": {"real_train_sources": 24, "real_train_windows": 72, "real_val_sources": 8, "real_val_windows": 24, "fake_videos": 10, "svd_videos": 5, "cogvideo_videos": 5, "fake_pair2_ordinals": 5, "fake_unscoreable_videos": 1},
        "aggregation_reconciliation": _reconcile_aggregation(),
        "activity": activity,
        "distance_diagnostics": distance_diagnostics,
        "per_dimension": _dimension_summary(detail_rows),
        "compression": {**compression, "effects": compression_effects, "note": "Descriptive pairwise diagnostics reconstructed from saved ParticleSequence and component members; never fitted or added to S_t."},
        "quality_correlations": _quality_correlations(flat_rows),
        "mechanism_interpretation": {},
        "artifacts": {"real_frontend": file_identity(REAL_FRONTEND), "fake_frontend": file_identity(FAKE_FRONTEND), "frozen_model": file_identity(FROZEN_MODEL_PATH), "fake_manifest": file_identity(FAKE_MANIFEST)},
        "per_video": detail_rows,
    }
    summary["mechanism_interpretation"] = _mechanism_interpretation(summary)
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "mechanism_summary.json", {key: value for key, value in summary.items() if key != "per_video"})
    write_json(output / "per_video_mechanism.json", detail_rows)
    write_json(output / "per_dimension_summary.json", summary["per_dimension"])
    _write_dimension_csv(output / "per_dimension_summary.csv", summary["per_dimension"])
    write_json(output / "compression_diagnostics.json", {"status": compression, "rows": compression_rows})
    _write_csv(output / "per_video_mechanism.csv", flat_rows)
    write_json(output / "run_summary.json", {"status": summary["status"], "population": summary["population"], "frozen_model": summary["frozen_model"], "mechanism_interpretation": summary["mechanism_interpretation"]})
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUTPUT_ROOT)
    args = parser.parse_args()
    result = analyze(args.output)
    print(json.dumps({key: value for key, value in result.items() if key not in {"per_video"}}, indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
