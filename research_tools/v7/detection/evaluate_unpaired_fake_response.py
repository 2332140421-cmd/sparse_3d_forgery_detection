"""Evaluate the frozen real-only normality response on unpaired fake media.

The real validation scores are read from the frozen artifact.  Fake features
are scored with the same frozen Gaussian/Mahalanobis models; no fitting,
calibration, filtering, or threshold tuning uses fake media.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np

from .protocol import (
    FROZEN_VAL_SCORE_PATH,
    MODEL_NAMES,
    file_identity,
    load_frozen_models,
    read_json,
    score_frontend_results,
    sha256_file,
    auroc,
    distribution,
    write_json,
)
from .run_unpaired_fake_response import FAKE_GENERATORS, FROZEN_MODEL_PATH, FROZEN_MODEL_SHA256, verify_frozen_model


DATA_ROOT = Path("/root/autodl-tmp/data/sparse_3d_forgery_detection")
OUTPUT_ROOT = DATA_ROOT / "derived/v7_unpaired_fake_response_v1"
REAL_FRONTEND_PATH = DATA_ROOT / "derived/v7_real_only_normality_pilot_v1/frontend/window_results.json"
FAKE_FRONTEND_PATH = OUTPUT_ROOT / "frontend/window_results.json"

QUALITY_KEYS = (
    "geometry_coverage",
    "tracking_persistence",
    "pose_success",
    "component_success_fraction",
    "s_valid_fraction",
)


def _finite(values: Iterable[float]) -> list[float]:
    return [float(value) for value in values if np.isfinite(float(value))]


def _median(values: Iterable[float]) -> float | None:
    values = _finite(values)
    return float(np.median(values)) if values else None


def _iqr(values: Iterable[float]) -> float | None:
    values = _finite(values)
    return float(np.percentile(values, 75) - np.percentile(values, 25)) if values else None


def _p10_p90_min_max(values: Iterable[float]) -> dict[str, float | int | None]:
    values = _finite(values)
    if not values:
        return {"N": 0, "median": None, "IQR": None, "p10": None, "p90": None, "min": None, "max": None}
    array = np.asarray(values, dtype=np.float64)
    return {
        "N": int(array.size),
        "median": float(np.median(array)),
        "IQR": float(np.percentile(array, 75) - np.percentile(array, 25)),
        "p10": float(np.percentile(array, 10)),
        "p90": float(np.percentile(array, 90)),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
    }


def _video_id(row: Mapping[str, object]) -> str:
    window = row.get("window")
    if isinstance(window, Mapping):
        return str(window.get("video_id") or window.get("source_id"))
    return str(row.get("video_id") or row.get("source_id"))


def aggregate_window_scores(rows: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    """Use the prior real-val source aggregation: median of window p95 scores."""

    grouped: dict[str, dict[str, object]] = {}
    for row in rows:
        video_id = _video_id(row)
        group = grouped.setdefault(video_id, {
            "video_id": video_id,
            "source_id": str(row.get("source_id") or row.get("window", {}).get("source_id")),
            "role": "real" if str(row.get("role") or row.get("window", {}).get("role")) in {"real", "real_val", "real_train"} else "fake",
            "generator": row.get("generator") or row.get("window", {}).get("generator"),
            "pair2_ordinal": row.get("pair2_ordinal") or row.get("window", {}).get("pair_key"),
            "window_ids": [],
            "models": {name: {"p95": [], "median": [], "feature_count": 0} for name in MODEL_NAMES},
        })
        window_id = row.get("window_id") or row.get("window", {}).get("window_id")
        group["window_ids"].append(str(window_id))
        for name in MODEL_NAMES:
            model_row = row.get(name, {})
            if not isinstance(model_row, Mapping):
                continue
            for output_key, candidate_keys in (("p95", ("window_p95", "video_p95")), ("median", ("window_median", "video_median"))):
                for key in candidate_keys:
                    value = model_row.get(key)
                    if value is not None and np.isfinite(float(value)):
                        group["models"][name][output_key].append(float(value))
                        break
            group["models"][name]["feature_count"] += int(model_row.get("feature_count", 0))
    result: list[dict[str, object]] = []
    for group in grouped.values():
        output = {
            key: group[key]
            for key in ("video_id", "source_id", "role", "generator", "pair2_ordinal", "window_ids")
        }
        output["window_count"] = len(group["window_ids"])
        for name in MODEL_NAMES:
            values = group["models"][name]
            output[name] = {
                "video_p95": _median(values["p95"]),
                "video_median": _median(values["median"]),
                "feature_count": int(values["feature_count"]),
            }
        result.append(output)
    return sorted(result, key=lambda row: str(row["video_id"]))


def real_window_score_rows(path: Path = FROZEN_VAL_SCORE_PATH) -> list[dict[str, object]]:
    raw = read_json(path)
    if not isinstance(raw, list) or len(raw) != 24:
        raise ValueError("frozen real_val score artifact must contain 24 windows")
    rows: list[dict[str, object]] = []
    for row in raw:
        if str(row.get("role")) != "real_val":
            raise ValueError("real score artifact contains a non-real_val row")
        rows.append({
            "window_id": str(row["window_id"]),
            "video_id": str(row["source_id"]),
            "source_id": str(row["source_id"]),
            "role": "real",
            "generator": None,
            "pair2_ordinal": None,
            **{name: dict(row[name]) for name in MODEL_NAMES},
        })
    return rows


def _quality_from_window(row: Mapping[str, object], *, role: str) -> dict[str, object]:
    coverage = row.get("coverage", {})
    tracking = row.get("tracking", {})
    components = int(row.get("component_count", 0))
    timestamps = row.get("timestamps_s", [])
    slots = len(timestamps) * components
    valid_states = int(row.get("valid_s_states", 0))
    window = row.get("window", {})
    if not isinstance(coverage, Mapping) or not isinstance(tracking, Mapping):
        raise ValueError("frontend quality fields are missing")
    return {
        "video_id": str(window.get("video_id") or window.get("source_id")) if isinstance(window, Mapping) else str(row.get("video_id") or row.get("source_id")),
        "source_id": str(window.get("source_id")) if isinstance(window, Mapping) else str(row.get("source_id")),
        "role": role,
        "generator": window.get("generator") if isinstance(window, Mapping) else row.get("generator"),
        "pair2_ordinal": window.get("pair_key") if isinstance(window, Mapping) else row.get("pair2_ordinal"),
        "window_id": str(window.get("window_id")) if isinstance(window, Mapping) else str(row.get("window_id")),
        "geometry_coverage": float(coverage["final_geometry_valid_fraction"]),
        "tracking_persistence": float(tracking["persistent_8_fraction"]),
        "pose_success": float(coverage["pose_valid_fraction"]),
        "component_success": float(components > 0),
        "s_valid_fraction": float(valid_states / slots) if slots else 0.0,
        "delta_s_valid_count": int(row.get("valid_delta_s", 0)),
        "delta2_s_valid_count": int(row.get("valid_delta2_s", 0)),
        "status": str(row.get("status", "UNKNOWN")),
    }


def aggregate_quality(rows: Sequence[Mapping[str, object]], *, role: str) -> list[dict[str, object]]:
    grouped: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        quality = _quality_from_window(row, role=role)
        grouped.setdefault(str(quality["video_id"]), []).append(quality)
    result: list[dict[str, object]] = []
    for video_id, values in grouped.items():
        first = values[0]
        output: dict[str, object] = {
            "video_id": video_id,
            "source_id": first["source_id"],
            "role": role,
            "generator": first["generator"],
            "pair2_ordinal": first["pair2_ordinal"],
            "window_count": len(values),
            "complete_window_count": sum(value["status"] == "COMPLETE" for value in values),
            "delta_s_valid_count": int(sum(int(value["delta_s_valid_count"]) for value in values)),
            "delta2_s_valid_count": int(sum(int(value["delta2_s_valid_count"]) for value in values)),
        }
        for key in QUALITY_KEYS:
            if key == "component_success_fraction":
                output[key] = float(np.mean([value["component_success"] for value in values]))
            else:
                output[key] = _median(value[key] for value in values)
        result.append(output)
    return sorted(result, key=lambda row: str(row["video_id"]))


def spearman(x: Sequence[float], y: Sequence[float]) -> float | None:
    if len(x) != len(y) or len(x) < 2:
        return None
    x_array = np.asarray(x, dtype=np.float64)
    y_array = np.asarray(y, dtype=np.float64)
    if not np.all(np.isfinite(x_array)) or not np.all(np.isfinite(y_array)):
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
    xr, yr = ranks(x_array), ranks(y_array)
    if np.std(xr) == 0 or np.std(yr) == 0:
        return None
    return float(np.corrcoef(xr, yr)[0, 1])


def cluster_bootstrap(rows: Sequence[Mapping[str, object]], model: str, *, seed: int = 20260908, repeats: int = 10000) -> dict[str, object]:
    real = [row for row in rows if row["role"] == "real" and row[model]["video_p95"] is not None]
    fake = [row for row in rows if row["role"] == "fake" and row[model]["video_p95"] is not None]
    real_groups = {str(row["video_id"]): [row] for row in real}
    fake_groups: dict[str, list[Mapping[str, object]]] = {}
    for row in fake:
        fake_groups.setdefault(str(row["pair2_ordinal"]), []).append(row)
    if len(real_groups) != 8 or len(fake_groups) != 5:
        return {"status": "INSUFFICIENT_CLUSTER_N", "real_clusters": len(real_groups), "fake_clusters": len(fake_groups)}
    rng = np.random.default_rng(seed)
    real_ids, fake_ids = sorted(real_groups), sorted(fake_groups)
    values: list[float] = []
    for _ in range(repeats):
        sample_real = [row for cluster in rng.choice(real_ids, len(real_ids), replace=True) for row in real_groups[str(cluster)]]
        sample_fake = [row for cluster in rng.choice(fake_ids, len(fake_ids), replace=True) for row in fake_groups[str(cluster)]]
        sample = sample_real + sample_fake
        labels = [int(row["role"] == "fake") for row in sample]
        scores = [float(row[model]["video_p95"]) for row in sample]
        value = auroc(labels, scores)
        if value is not None:
            values.append(value)
    return {
        "status": "OK" if values else "NO_VALID_RESAMPLE",
        "repeats": repeats,
        "seed": seed,
        "real_cluster_unit": "Vript real_val source video",
        "fake_cluster_unit": "Pair2 source ordinal preserving generator variants",
        "real_clusters": len(real_groups),
        "fake_clusters": len(fake_groups),
        "lower_95": float(np.percentile(values, 2.5)) if values else None,
        "upper_95": float(np.percentile(values, 97.5)) if values else None,
    }


def _model_metrics(rows: Sequence[Mapping[str, object]], model: str) -> dict[str, object]:
    valid = [row for row in rows if row[model]["video_p95"] is not None]
    real = [row for row in valid if row["role"] == "real"]
    fake = [row for row in valid if row["role"] == "fake"]
    labels = [int(row["role"] == "fake") for row in valid]
    scores = [float(row[model]["video_p95"]) for row in valid]
    auc = auroc(labels, scores)
    per_generator: dict[str, object] = {}
    for generator in FAKE_GENERATORS:
        subset = real + [row for row in fake if row.get("generator") == generator]
        sub_labels = [int(row["role"] == "fake") for row in subset]
        sub_scores = [float(row[model]["video_p95"]) for row in subset]
        sub_auc = auroc(sub_labels, sub_scores)
        total_generator_fake = sum(row["role"] == "fake" and row.get("generator") == generator for row in rows)
        per_generator[generator] = {
            "N_fake": total_generator_fake,
            "scored_fake_N": sum(row["role"] == "fake" for row in subset),
            "fake_distribution": _p10_p90_min_max(float(row[model]["video_p95"]) for row in subset if row["role"] == "fake"),
            "AUROC": sub_auc,
            "Cliffs_delta": 2 * sub_auc - 1 if sub_auc is not None else None,
        }
    all_real = [row for row in rows if row["role"] == "real"]
    all_fake = [row for row in rows if row["role"] == "fake"]
    return {
        "N": len(rows),
        "score_available_N": len(valid),
        "real_N": len(all_real),
        "real_scored_N": len(real),
        "fake_N": len(all_fake),
        "fake_scored_N": len(fake),
        "real_distribution": _p10_p90_min_max(float(row[model]["video_p95"]) for row in real),
        "fake_distribution": _p10_p90_min_max(float(row[model]["video_p95"]) for row in fake),
        "AUROC": auc,
        "Cliffs_delta": 2 * auc - 1 if auc is not None else None,
        "cluster_bootstrap_95": cluster_bootstrap(valid, model),
        "per_generator": per_generator,
    }


def _quality_correlations(rows: Sequence[Mapping[str, object]]) -> dict[str, dict[str, float | None]]:
    result: dict[str, dict[str, float | None]] = {}
    for model in MODEL_NAMES:
        model_result: dict[str, float | None] = {}
        for quality in QUALITY_KEYS:
            paired = [(float(row[model]["video_p95"]), float(row[quality])) for row in rows if row[model]["video_p95"] is not None and row.get(quality) is not None]
            model_result[quality] = spearman([pair[0] for pair in paired], [pair[1] for pair in paired])
        result[model] = model_result
    return result


def _tail_metrics(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    real = [row for row in rows if row["role"] == "real"]
    fake = [row for row in rows if row["role"] == "fake"]
    output: dict[str, object] = {}
    for model in MODEL_NAMES:
        real_values = [float(row[model]["video_p95"]) for row in real if row[model]["video_p95"] is not None]
        fake_values = [row for row in fake if row[model]["video_p95"] is not None]
        tau95 = float(np.percentile(real_values, 95)) if real_values else None
        real_max = float(np.max(real_values)) if real_values else None
        def fraction(values: Sequence[Mapping[str, object]], threshold: float | None) -> float | None:
            if threshold is None or not values:
                return None
            return float(np.mean([float(row[model]["video_p95"]) > threshold for row in values]))
        output[model] = {
            "tau95_real_only": tau95,
            "tau_max_real_only": real_max,
            "overall_fake_gt_tau95_fraction": fraction(fake_values, tau95),
            "overall_fake_gt_real_max_fraction": fraction(fake_values, real_max),
            "by_generator": {
                generator: {
                    "fake_gt_tau95_fraction": fraction([row for row in fake_values if row.get("generator") == generator], tau95),
                    "fake_gt_real_max_fraction": fraction([row for row in fake_values if row.get("generator") == generator], real_max),
                }
                for generator in FAKE_GENERATORS
            },
        }
    return output


def _quality_summary(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    return {
        role: {key: _p10_p90_min_max(row[key] for row in rows if row["role"] == role) for key in QUALITY_KEYS}
        for role in ("real", "fake")
    }


def classify_status(rows: Sequence[Mapping[str, object]], model_metrics: Mapping[str, Mapping[str, object]], correlations: Mapping[str, Mapping[str, float | None]]) -> str:
    if not any(row["role"] == "fake" and row["M1_delta_s"]["video_p95"] is not None for row in rows):
        return "EXPERIMENT_EXECUTION_BLOCKED"
    deltas = [float(model_metrics[name]["Cliffs_delta"]) for name in MODEL_NAMES if model_metrics[name]["Cliffs_delta"] is not None]
    positive_generators = []
    for generator in FAKE_GENERATORS:
        generator_deltas = [float(model_metrics[name]["per_generator"][generator]["Cliffs_delta"]) for name in MODEL_NAMES if model_metrics[name]["per_generator"][generator]["Cliffs_delta"] is not None]
        positive_generators.append(bool(generator_deltas) and np.mean(np.asarray(generator_deltas) > 0) >= 0.5)
    quality = _quality_summary(rows)
    collapsed = sum(
        quality["fake"][key]["median"] is not None
        and quality["real"][key]["median"] is not None
        and quality["fake"][key]["median"] < quality["real"][key]["median"] * 0.75
        for key in ("geometry_coverage", "tracking_persistence", "pose_success", "component_success_fraction", "s_valid_fraction")
    ) >= 2
    strong_quality_link = any(abs(float(value)) >= 0.7 for model in correlations.values() for value in model.values() if value is not None)
    response = sum(delta > 0.2 for delta in deltas) >= 2 and sum(
        model_metrics[name]["fake_distribution"]["median"] > model_metrics[name]["real_distribution"]["median"]
        for name in MODEL_NAMES
    ) >= 2
    if response and collapsed and strong_quality_link:
        return "FAKE_RESPONSE_FRONTEND_CONFOUNDED"
    if response and all(positive_generators):
        return "UNPAIRED_FAKE_RESPONSE_OBSERVED"
    if not deltas or max(abs(delta) for delta in deltas) < 0.2:
        return "NO_OBVIOUS_UNPAIRED_FAKE_RESPONSE"
    return "WEAK_OR_GENERATOR_SPECIFIC_FAKE_RESPONSE"


def _merge_rows(scores: Sequence[Mapping[str, object]], quality: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    quality_by_id = {str(row["video_id"]): row for row in quality}
    result: list[dict[str, object]] = []
    for score in scores:
        video_id = str(score["video_id"])
        if video_id not in quality_by_id:
            raise ValueError(f"quality row missing for {video_id}")
        row = {**quality_by_id[video_id]}
        row.update({"video_id": video_id, "role": score["role"], "generator": score.get("generator"), "pair2_ordinal": score.get("pair2_ordinal") or row.get("pair2_ordinal"), "source_domain": "Vript" if score["role"] == "real" else "GenVidBench_Pair2"})
        row["window_count"] = score["window_count"]
        for model in MODEL_NAMES:
            row[model] = {"video_p95": score[model]["video_p95"], "video_median": score[model]["video_median"], "feature_count": score[model]["feature_count"]}
            row[f"{model}_video_score"] = score[model]["video_p95"]
            row[f"{model}_sensitivity_score"] = score[model]["video_median"]
        result.append(row)
    return sorted(result, key=lambda row: str(row["video_id"]))


def write_per_video(output: Path, rows: Sequence[Mapping[str, object]], tails: Mapping[str, object]) -> None:
    fields = ["video_id", "role", "source_domain", "generator", "pair2_ordinal", "window_count", *QUALITY_KEYS, "delta_s_valid_count", "delta2_s_valid_count", "M1_video_score", "M2_video_score", "M3_video_score", "above_M1_tau95", "above_M2_tau95", "above_M3_tau95", "above_M1_real_max", "above_M2_real_max", "above_M3_real_max"]
    serialized: list[dict[str, object]] = []
    for row in rows:
        value = dict(row)
        value["source_domain"] = "Vript" if row["role"] == "real" else "GenVidBench_Pair2"
        short_names = {"M1_delta_s": "M1", "M2_delta2_s": "M2", "M3_delta_s_delta2_s": "M3"}
        for model in MODEL_NAMES:
            short = short_names[model]
            score = row[f"{model}_video_score"]
            value[f"{short}_video_score"] = score
            value[f"above_{short}_tau95"] = bool(score is not None and tails[model]["tau95_real_only"] is not None and score > tails[model]["tau95_real_only"])
            value[f"above_{short}_real_max"] = bool(score is not None and tails[model]["tau_max_real_only"] is not None and score > tails[model]["tau_max_real_only"])
        serialized.append({field: value.get(field) for field in fields})
    write_json(output / "metrics" / "per_video_response.json", serialized)
    with (output / "metrics" / "per_video_response.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(serialized)


def _fmt(value: object) -> str:
    return "NA" if value is None else f"{float(value):.6g}" if isinstance(value, (float, int)) else str(value)


def render_report(output: Path, result: Mapping[str, object]) -> None:
    metrics = result
    lines = [
        "# V7 real-only structural normality — unpaired generated-video response",
        "",
        "## 1. Research Question",
        "",
        "冻结的 real-only structural-evolution normality 是否对真实生成视频产生异常响应？",
        "",
        "## 2. Scientific Boundary",
        "",
        "这是 Vript held-out real_val 与 GenVidBench Pair2 SVD/CogVideo 的 unpaired source/domain comparison；不是最终 paired benchmark，也没有 structural-anomaly ground truth。",
        "",
        "## 3. Frozen Method",
        "",
        f"M1=ΔS、M2=Δ²S、M3=[ΔS,Δ²S]；model SHA256=`{result['frozen_model']['sha256']}`；fitting population=real_train_only；fake fitting count=0；无 refit。",
        "",
        "1.0 s true-PTS windows、25/50/75% anchors、现有 ComponentConfig、S_t/ΔS/Δ²S 和前端均保持冻结。",
        "",
        "## 4. Population and Representation",
        "",
        "real N=8 held-out Vript sources；fake N=10（SVD=5、CogVideo=5），fake underlying Pair2 ordinal N=5。两类均使用 geometry → tracking → component → S validity → frozen score；所有 10 个 fake 均保留在 per-video 表，00042 CogVideo 因三窗口均无 component 而无可评分 M1/M2/M3 值。",
        "",
        f"real quality summary: `{json.dumps(result['quality_summary']['real'], ensure_ascii=False, sort_keys=True)}`",
        f"fake quality summary: `{json.dumps(result['quality_summary']['fake'], ensure_ascii=False, sort_keys=True)}`",
        "",
        "## 5. Fake Response Results",
        "",
        "主 video/source score 按已有 real-val 实际 artifact 使用：每个视频先取每个窗口的 component-time p95，再对窗口 p95 取 source median；这与 ADR 中简写的 primary p95 存在表述差异，未静默改用跨窗口 pooled p95。",
        "",
        "| model | real median/IQR | fake median/IQR | AUROC | Cliff's delta | cluster bootstrap 95% CI |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for model in MODEL_NAMES:
        row = metrics["models"][model]
        boot = row["cluster_bootstrap_95"]
        ci = f"{_fmt(boot.get('lower_95'))}–{_fmt(boot.get('upper_95'))}" if boot.get("status") == "OK" else str(boot.get("status"))
        lines.append(f"| {model} | {_fmt(row['real_distribution']['median'])}/{_fmt(row['real_distribution']['IQR'])} | {_fmt(row['fake_distribution']['median'])}/{_fmt(row['fake_distribution']['IQR'])} | {_fmt(row['AUROC'])} | {_fmt(row['Cliffs_delta'])} | {ci} |")
    lines += ["", "## 6. Generator-specific Results", ""]
    for generator in FAKE_GENERATORS:
        lines.append(f"### {generator} (N=5 fake; same N=8 real_val reference)")
        for model in MODEL_NAMES:
            row = metrics["models"][model]["per_generator"][generator]
            lines.append(f"- {model}: fake N={row['N_fake']} (scored={row['scored_fake_N']}), median={_fmt(row['fake_distribution']['median'])}, IQR={_fmt(row['fake_distribution']['IQR'])}, AUROC={_fmt(row['AUROC'])}, Cliff's delta={_fmt(row['Cliffs_delta'])}")
        lines.append("")
    lines += ["## 7. Real-only Empirical Tail Response", ""]
    for model in MODEL_NAMES:
        row = metrics["tails"][model]
        lines.append(f"- {model}: tau95={_fmt(row['tau95_real_only'])}, real-max={_fmt(row['tau_max_real_only'])}, overall fake>tau95={_fmt(row['overall_fake_gt_tau95_fraction'])}, overall fake>real-max={_fmt(row['overall_fake_gt_real_max_fraction'])}; SVD={json.dumps(row['by_generator']['svd'], sort_keys=True)}, CogVideo={json.dumps(row['by_generator']['cogvideo'], sort_keys=True)}")
    lines += ["", "## 8. Measurement-quality Confounding", "", f"score-quality Spearman: `{json.dumps(metrics['quality_correlations'], sort_keys=True)}`", "", "The per-video table is `metrics/per_video_response.csv` and `.json`; no video was dropped by anomaly score. Quality is reported descriptively and was not used to refit, filter, or tune the model.", ""]
    lines += ["## 9. First- vs Second-order Evidence", ""]
    lines.append("M1=ΔS、M2=Δ²S、M3=[ΔS,Δ²S] 的比较只按上述 frozen scores 解读；M3 的 condition number 保持 frozen，不因 fake 结果重训或重新 regularize。")
    lines += ["", "## 10. Interpretation and Limitations", "", "该结果只能回答 frozen structural normality 是否对当前真实生成视频产生 exploratory response，不能证明结构异常 GT、最终跨域检测能力或定位能力。限制包括 unpaired source/domain、real N=8、fake N=10/5 ordinals、仅 SVD/CogVideo、M3 高 covariance condition、无 spatial anomaly GT、无 paired HD-VG real，以及当前 S_t/component baseline。", "", "## 11. Conclusion", "", f"`{result['status']}`", ""]
    (output / "docs").mkdir(parents=True, exist_ok=True)
    (output / "docs" / "v7-unpaired-generated-video-response.md").write_text("\n".join(lines), encoding="utf-8")


def evaluate(output: Path = OUTPUT_ROOT) -> dict[str, object]:
    frozen_model = verify_frozen_model()
    models, identity = load_frozen_models(FROZEN_MODEL_PATH)
    if identity["artifact"]["sha256"] != FROZEN_MODEL_SHA256:
        raise RuntimeError("FROZEN_MODEL_IDENTITY_MISMATCH")
    fake_raw = read_json(output / "frontend" / "window_results.json")
    real_frontend = read_json(REAL_FRONTEND_PATH)
    if not isinstance(fake_raw, list) or not isinstance(real_frontend, list):
        raise ValueError("frontend artifacts must be lists")
    fake_scored = score_frontend_results(fake_raw, models)
    write_json(output / "scores" / "unpaired_window_scores.json", fake_scored["window_scores"])
    write_json(output / "scores" / "unpaired_video_scores.json", fake_scored["video_scores"])
    fake_score_windows = fake_scored["window_scores"]
    fake_pair_by_window = {str(row["window"]["window_id"]): row["window"].get("pair_key") for row in fake_raw}
    for row in fake_score_windows:
        row["pair2_ordinal"] = fake_pair_by_window.get(str(row["window_id"]))
    real_score_windows = real_window_score_rows()
    scores = aggregate_window_scores(real_score_windows + fake_score_windows)
    real_quality = aggregate_quality([row for row in real_frontend if str(row.get("window", {}).get("role")) == "real_val"], role="real")
    fake_quality = aggregate_quality(fake_raw, role="fake")
    rows = _merge_rows(scores, real_quality + fake_quality)
    model_metrics = {model: _model_metrics(rows, model) for model in MODEL_NAMES}
    quality_correlations = _quality_correlations(rows)
    tails = _tail_metrics(rows)
    result: dict[str, object] = {
        "status": classify_status(rows, model_metrics, quality_correlations),
        "frozen_model": frozen_model,
        "frozen_model_identity": identity,
        "population": {"real_N": 8, "fake_N": 10, "svd_N": 5, "cogvideo_N": 5, "fake_pair2_ordinal_N": 5, "score_available_by_model": {model: sum(row[model]["video_p95"] is not None for row in rows) for model in MODEL_NAMES}, "fake_score_unavailable_by_model": {model: sum(row["role"] == "fake" and row[model]["video_p95"] is None for row in rows) for model in MODEL_NAMES}},
        "protocol": {"timescale_s": 1.0, "anchors": [0.25, 0.5, 0.75], "component_config_unchanged": True, "fake_used_for_fitting": False, "normality_refit": False, "aggregation_primary": "source_median_of_window_component_p95", "aggregation_sensitivity": "source_median_of_window_component_median", "aggregation_artifact_note": "frozen real_val score artifact stores window p95; source-level real metric is median across windows"},
        "models": model_metrics,
        "tails": tails,
        "quality_summary": _quality_summary(rows),
        "quality_correlations": quality_correlations,
        "representation": {
            "real": {"video_count": 8, "window_count": len(real_frontend), "complete_windows": sum(row.get("status") == "COMPLETE" for row in real_frontend), "geometry_coverage": _p10_p90_min_max(row["geometry_coverage"] for row in real_quality), "tracking_persistence": _p10_p90_min_max(row["tracking_persistence"] for row in real_quality), "pose_success": _p10_p90_min_max(row["pose_success"] for row in real_quality), "component_success_fraction": _p10_p90_min_max(row["component_success_fraction"] for row in real_quality), "s_valid_fraction": _p10_p90_min_max(row["s_valid_fraction"] for row in real_quality), "delta_s_valid_count": sum(int(row["delta_s_valid_count"]) for row in real_quality), "delta2_s_valid_count": sum(int(row["delta2_s_valid_count"]) for row in real_quality)},
            "fake": {"video_count": 10, "window_count": len(fake_raw), "complete_windows": sum(row.get("status") == "COMPLETE" for row in fake_raw), "geometry_coverage": _p10_p90_min_max(row["geometry_coverage"] for row in fake_quality), "tracking_persistence": _p10_p90_max(row["tracking_persistence"] for row in fake_quality), "pose_success": _p10_p90_max(row["pose_success"] for row in fake_quality), "component_success_fraction": _p10_p90_max(row["component_success_fraction"] for row in fake_quality), "s_valid_fraction": _p10_p90_max(row["s_valid_fraction"] for row in fake_quality), "delta_s_valid_count": sum(int(row["delta_s_valid_count"]) for row in fake_quality), "delta2_s_valid_count": sum(int(row["delta2_s_valid_count"]) for row in fake_quality)},
        },
        "per_video_rows": rows,
        "fake_used_for_fitting": False,
        "normality_refit": False,
        "spatial_localization_ground_truth": "SPATIAL_LOCALIZATION_GT_NOT_AVAILABLE",
    }
    # Keep the representation summary construction compact while retaining the
    # same distribution function for every quality field.
    for role in ("real", "fake"):
        for key in QUALITY_KEYS:
            result["representation"][role][key] = _p10_p90_min_max(row[key] for row in (real_quality if role == "real" else fake_quality))
    write_per_video(output, rows, tails)
    write_json(output / "metrics" / "unpaired_response_metrics.json", {key: value for key, value in result.items() if key != "per_video_rows"})
    write_json(output / "models_frozen" / "frozen_model_identity.json", frozen_model)
    write_json(output / "run_evaluation.json", {"status": result["status"], "frozen_model": frozen_model, "fake_used_for_fitting": False, "normality_refit": False})
    summary_path = output / "run_summary.json"
    summary = read_json(summary_path) if summary_path.is_file() else {}
    if not isinstance(summary, dict):
        summary = {}
    summary.update({"status": result["status"], "population": result["population"], "fake_used_for_fitting": False, "normality_refit": False, "metrics_path": str(output / "metrics" / "unpaired_response_metrics.json"), "per_video_path": str(output / "metrics" / "per_video_response.csv")})
    write_json(summary_path, summary)
    render_report(output, result)
    return result


def _p10_p90_max(values: Iterable[float]) -> dict[str, float | int | None]:
    return _p10_p90_min_max(values)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUTPUT_ROOT)
    args = parser.parse_args()
    result = evaluate(args.output)
    print(json.dumps({key: value for key, value in result.items() if key != "per_video_rows"}, indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
