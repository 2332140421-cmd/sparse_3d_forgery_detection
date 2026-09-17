"""Read-only analysis for the completed V7 observation-support pilot.

This entry point never loads a model and never runs a frontend.  It consumes
the saved logits, feature NPZ files, feature manifest, and the source support
records, then writes small CSV summaries and a markdown report under the
experiment's ``analysis_v1`` directory.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


CONDITIONS = (
    "STRUCTURE_ONLY",
    "SUPPORT_ONLY",
    "STRUCTURE_SUPPORT",
    "STRUCTURE_DUP_CONTROL",
)
SEEDS = (20260909, 20260910, 20260911)
BOOTSTRAP_SEED = 20260909
BOOTSTRAP_REPLICATES = 10000


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: "" if row.get(key) is None else row.get(key) for key in fields})


def _finite(value: Any) -> float | None:
    if value is None or value == "":
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _auroc(labels: Sequence[int], scores: Sequence[float]) -> float | None:
    y = np.asarray(labels, dtype=np.int64)
    s = np.asarray(scores, dtype=np.float64)
    if np.sum(y == 0) == 0 or np.sum(y == 1) == 0:
        return None
    positive = s[y == 1]
    negative = s[y == 0]
    return float(np.mean((positive[:, None] > negative[None, :]) + 0.5 * (positive[:, None] == negative[None, :])))


def _ap(labels: Sequence[int], scores: Sequence[float]) -> float | None:
    y = np.asarray(labels, dtype=np.int64)
    s = np.asarray(scores, dtype=np.float64)
    if not np.any(y == 1) or not np.any(y == 0):
        return None
    order = np.argsort(-s, kind="mergesort")
    ordered = y[order]
    true_positive = np.cumsum(ordered == 1)
    precision = true_positive / np.arange(1, len(ordered) + 1)
    return float(np.sum(precision[ordered == 1]) / np.sum(ordered == 1))


def _classification(labels: Sequence[int], scores: Sequence[float]) -> dict[str, Any]:
    y = np.asarray(labels, dtype=np.int64)
    predicted = np.asarray(scores, dtype=np.float64) >= 0.0
    tn = int(np.sum((y == 0) & ~predicted))
    fp = int(np.sum((y == 0) & predicted))
    fn = int(np.sum((y == 1) & ~predicted))
    tp = int(np.sum((y == 1) & predicted))
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    f1 = 2 * precision * recall / (precision + recall) if precision is not None and recall is not None and precision + recall else None
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": (tn + tp) / len(y) if len(y) else None,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    }


def _bootstrap(values: Mapping[str, float], other: Mapping[str, float] | None = None) -> dict[str, Any]:
    keys = sorted(set(values) & (set(other) if other is not None else set(values)))
    if not keys:
        return {"source_count": 0, "sources": [], "mean": None, "ci95": [None, None], "seed": BOOTSTRAP_SEED, "replicates": BOOTSTRAP_REPLICATES}
    left = np.asarray([values[key] for key in keys], dtype=np.float64)
    delta = left if other is None else left - np.asarray([other[key] for key in keys], dtype=np.float64)
    draws = delta[np.random.default_rng(BOOTSTRAP_SEED).integers(0, len(keys), size=(BOOTSTRAP_REPLICATES, len(keys)))].mean(axis=1)
    return {
        "source_count": len(keys),
        "sources": keys,
        "mean": float(np.mean(delta)),
        "ci95": [float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))],
        "seed": BOOTSTRAP_SEED,
        "replicates": BOOTSTRAP_REPLICATES,
    }


def _metric_bundle(rows: Sequence[Mapping[str, Any]], score_key: str) -> tuple[dict[str, Any], dict[str, float]]:
    labels = [int(row["label"]) for row in rows]
    scores = [float(row[score_key]) for row in rows]
    by_source: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        by_source[str(row["source_id"])].append(row)
    per_source = {
        source: _auroc([int(row["label"]) for row in subset], [float(row[score_key]) for row in subset])
        for source, subset in by_source.items()
    }
    per_source = {source: float(value) for source, value in per_source.items() if value is not None}
    metrics = {
        "window_count": len(rows),
        "real_count": labels.count(0),
        "fake_count": labels.count(1),
        "source_count": len(by_source),
        "dual_role_source_count": len(per_source),
        "source_macro_auroc": _bootstrap(per_source)["mean"],
        "source_macro_ci_low": _bootstrap(per_source)["ci95"][0],
        "source_macro_ci_high": _bootstrap(per_source)["ci95"][1],
        "pooled_auroc": _auroc(labels, scores),
        "pooled_ap": _ap(labels, scores),
    }
    metrics.update(_classification(labels, scores))
    return metrics, per_source


def _direction(value: float, eps: float = 1e-12) -> str:
    if value > eps:
        return "POSITIVE"
    if value < -eps:
        return "NEGATIVE"
    return "ZERO"


def _direction_class(values: Sequence[float]) -> str:
    directions = [_direction(float(value)) for value in values]
    if all(item == "ZERO" for item in directions):
        return "ALL_ZERO"
    if all(item == "POSITIVE" for item in directions):
        return "ALL_POSITIVE"
    if all(item == "NEGATIVE" for item in directions):
        return "ALL_NEGATIVE"
    if any(item == "ZERO" for item in directions):
        return "CONTAINS_ZERO"
    return "MIXED"


def _transition(label: int, base_score: float, fused_score: float) -> str:
    base_positive = base_score >= 0.0
    fused_positive = fused_score >= 0.0
    base_correct = base_positive == bool(label)
    fused_correct = fused_positive == bool(label)
    if label == 1 and not base_positive and fused_positive:
        return "FN_TO_TP"
    if label == 1 and base_positive and not fused_positive:
        return "TP_TO_FN"
    if label == 0 and base_positive and not fused_positive:
        return "FP_TO_TN"
    if label == 0 and not base_positive and fused_positive:
        return "TN_TO_FP"
    if base_correct and fused_correct:
        return "CORRECT_UNCHANGED"
    return "ERROR_UNCHANGED"


def _integer_counts(q: np.ndarray, raw_counts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    # q is saved as float32.  Fractions are generated from integer masks, so
    # rounding recovers the count without inventing a floating-point change.
    v = np.rint(q[:, :, 0] * raw_counts[:, None]).astype(np.int64)
    g = np.rint(q[:, :, 1] * raw_counts[:, None]).astype(np.int64)
    if np.any(v < 0) or np.any(g < 0) or np.any(v > raw_counts[:, None]) or np.any(g > raw_counts[:, None]):
        raise ValueError("Q_COUNT_OUT_OF_RANGE")
    return v, g


def _window_q(item: Mapping[str, Any], npz_path: Path) -> dict[str, Any]:
    with np.load(npz_path, allow_pickle=False) as archive:
        q = np.asarray(archive["q"], dtype=np.float64)
    identities = list(item["unit_identities"])
    raw_counts = np.asarray([int(identity["raw_member_count"]) for identity in identities], dtype=np.int64)
    if q.ndim != 3 or q.shape != (len(identities), 5, 4):
        raise ValueError(f"Q_SHAPE_MISMATCH:{item['window_id']}:{q.shape}")
    v, g = _integer_counts(q, raw_counts)
    q_changed = np.any(np.abs(q[:, :, 2:]) > 1e-8, axis=(1, 2))
    target_changed = (np.ptp(v, axis=1) > 0) | (np.ptp(g, axis=1) > 0)
    target_constant = ~target_changed
    fully_one = np.all(v == raw_counts[:, None], axis=1) & np.all(g == raw_counts[:, None], axis=1)
    constant_low_any = target_constant & ((v[:, 0] < raw_counts) | (g[:, 0] < raw_counts))
    constant_low_both = target_constant & ((v[:, 0] < raw_counts) & (g[:, 0] < raw_counts))
    history_only = target_constant & q_changed
    visibility_geometry_difference = np.any(v != g, axis=1)
    ages = np.asarray([float(age) for identity in identities for age in identity["query_age_s"]], dtype=np.float64)
    return {
        "window_id": str(item["window_id"]),
        "source_id": str(item["source_id"]),
        "role": str(item["role"]),
        "label": int(item["label"]),
        "annotation_category": str(item.get("annotation_category", "")),
        "b": float(item.get("offset_s", 0.0)),
        "unit_count": len(identities),
        "raw_common_diff_units": int(sum(int(identity["raw_member_count"]) != int(identity["common_member_count"]) for identity in identities)),
        "q_changed_units": int(np.sum(q_changed)),
        "target_changed_units": int(np.sum(target_changed)),
        "history_only_units": int(np.sum(history_only)),
        "fully_one_units": int(np.sum(fully_one)),
        "constant_low_any_units": int(np.sum(constant_low_any)),
        "constant_low_both_units": int(np.sum(constant_low_both)),
        "visibility_geometry_diff_units": int(np.sum(visibility_geometry_difference)),
        "q_changed_fraction": float(np.mean(q_changed)),
        "target_changed_fraction": float(np.mean(target_changed)),
        "history_only_fraction": float(np.mean(history_only)),
        "constant_low_any_fraction": float(np.mean(constant_low_any)),
        "constant_low_both_fraction": float(np.mean(constant_low_both)),
        "mean_v_fraction": float(np.mean(q[:, :, 0])),
        "mean_g_fraction": float(np.mean(q[:, :, 1])),
        "mean_v_history_change": float(np.mean(q[:, :, 2])),
        "mean_g_history_change": float(np.mean(q[:, :, 3])),
        "mean_abs_history_change": float(np.mean(np.abs(q[:, :, 2:]))),
        "query_age_min_s": float(np.min(ages)),
        "query_age_max_s": float(np.max(ages)),
        "query_age_mean_s": float(np.mean(ages)),
    }


def _coverage_rows(window_stats: Sequence[Mapping[str, Any]], scope: str, scope_type: str, key_fn) -> dict[str, Any]:
    selected = [row for row in window_stats if key_fn(row)]
    if not selected:
        return {"scope_type": scope_type, "scope": scope, "unit_count": 0, "window_count": 0}
    unit_count = sum(int(row["unit_count"]) for row in selected)
    def units(field: str) -> int:
        return sum(int(row[field]) for row in selected)
    def windows(field: str) -> int:
        return sum(int(row[field]) > 0 for row in selected)
    return {
        "scope_type": scope_type,
        "scope": scope,
        "source_id": "" if scope_type != "source" else str(selected[0]["source_id"]),
        "role": "" if scope_type != "role" else str(selected[0]["role"]),
        "b": "" if scope_type != "b" else str(selected[0]["b"]),
        "unit_count": unit_count,
        "window_count": len(selected),
        "raw_common_diff_units": units("raw_common_diff_units"),
        "q_changed_units": units("q_changed_units"),
        "target_changed_units": units("target_changed_units"),
        "history_only_units": units("history_only_units"),
        "fully_one_units": units("fully_one_units"),
        "constant_low_any_units": units("constant_low_any_units"),
        "constant_low_both_units": units("constant_low_both_units"),
        "visibility_geometry_diff_units": units("visibility_geometry_diff_units"),
        "windows_with_q_change": windows("q_changed_units"),
        "windows_with_target_change": windows("target_changed_units"),
        "windows_with_history_only": windows("history_only_units"),
        "windows_with_constant_low": windows("constant_low_any_units"),
        "mean_changed_unit_fraction": float(np.mean([row["q_changed_fraction"] for row in selected])),
        "mean_target_changed_unit_fraction": float(np.mean([row["target_changed_fraction"] for row in selected])),
        "mean_history_only_unit_fraction": float(np.mean([row["history_only_fraction"] for row in selected])),
        "mean_constant_low_any_fraction": float(np.mean([row["constant_low_any_fraction"] for row in selected])),
        "mean_constant_low_both_fraction": float(np.mean([row["constant_low_both_fraction"] for row in selected])),
        "mean_v_fraction": float(np.mean([row["mean_v_fraction"] for row in selected])),
        "mean_g_fraction": float(np.mean([row["mean_g_fraction"] for row in selected])),
        "mean_v_history_change": float(np.mean([row["mean_v_history_change"] for row in selected])),
        "mean_g_history_change": float(np.mean([row["mean_g_history_change"] for row in selected])),
        "query_age_min_s": float(min(row["query_age_min_s"] for row in selected)),
        "query_age_max_s": float(max(row["query_age_max_s"] for row in selected)),
        "query_age_mean_s": float(np.mean([row["query_age_mean_s"] for row in selected])),
    }


def _fmt(value: Any, digits: int = 6) -> str:
    if value is None or value == "":
        return "NA"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    return f"{float(value):.{digits}f}"


def analyze(root: Path, output: Path) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    protocol = _read_json(root / "protocol.json")
    feature_manifest = _read_json(root / "inputs/feature_manifest.json")
    validation_rows = _read_csv(root / "scores/validation_window_scores.csv")
    train_rows = _read_csv(root / "scores/train_window_scores.csv")
    saved_metrics = _read_csv(root / "evaluation/metrics.csv")
    if len(validation_rows) != 83 or len({row["window_id"] for row in validation_rows}) != 83:
        raise ValueError("VALIDATION_WINDOW_COUNT_OR_DUPLICATE")
    if (sum(int(row["label"]) == 0 for row in validation_rows), sum(int(row["label"]) == 1 for row in validation_rows)) != (42, 41):
        raise ValueError("VALIDATION_LABEL_COUNT_MISMATCH")
    if len({row["source_id"] for row in validation_rows}) != 14:
        raise ValueError("VALIDATION_SOURCE_COUNT_MISMATCH")
    for condition in CONDITIONS:
        for seed in SEEDS:
            key = f"{condition}_seed_{seed}"
            if any(key not in row for row in validation_rows):
                raise ValueError(f"MISSING_SCORE_COLUMN:{key}")
    validation_ids = {row["window_id"] for row in validation_rows}
    train_ids = {row["window_id"] for row in train_rows}
    manifest_by_id = {str(item["window_id"]): item for item in feature_manifest}
    if not validation_ids.issubset(manifest_by_id) or not train_ids.issubset(manifest_by_id):
        raise ValueError("FEATURE_MANIFEST_WINDOW_MISMATCH")

    # Recompute the primary metrics from the saved three-seed logits.
    metric_rows: list[dict[str, Any]] = []
    saved_by_key = {(row["split"], row["condition"], str(row["seed"])): row for row in saved_metrics}
    reproduction: dict[str, dict[str, Any]] = {}
    metric_names = ("source_macro_auroc", "source_macro_ci_low", "source_macro_ci_high", "pooled_auroc", "pooled_ap", "precision", "recall", "f1", "accuracy", "tn", "fp", "fn", "tp")
    mean_source_values: dict[str, dict[str, float]] = {}
    per_seed_source_values: dict[int, dict[str, dict[str, float]]] = {seed: {} for seed in SEEDS}
    for condition in CONDITIONS:
        mean_key = f"{condition}_MEAN_LOGIT"
        metrics, source_values = _metric_bundle(validation_rows, mean_key)
        mean_source_values[condition] = source_values
        for metric_name in metric_names:
            reproduced = metrics.get(metric_name)
            saved = saved_by_key.get(("validation", condition, "MEAN_LOGIT"), {}).get(metric_name)
            saved_value = _finite(saved)
            error = abs(float(reproduced) - saved_value) if reproduced is not None and saved_value is not None else None
            metric_rows.append({"split": "validation", "condition": condition, "seed": "MEAN_LOGIT", "metric": metric_name, "reproduced": reproduced, "saved": saved_value, "absolute_error": error, "match": error is not None and error <= 1e-12 or reproduced is None and saved_value is None})
        for seed in SEEDS:
            key = f"{condition}_seed_{seed}"
            metrics_seed, source_seed = _metric_bundle(validation_rows, key)
            per_seed_source_values[seed][condition] = source_seed

    _write_csv(output / "metric_reproduction.csv", metric_rows)

    # Source-level mean-logit differences and per-seed direction checks.
    per_source_rows: list[dict[str, Any]] = []
    for source in sorted(mean_source_values["STRUCTURE_SUPPORT"]):
        subset = [row for row in validation_rows if row["source_id"] == source]
        item: dict[str, Any] = {
            "source_id": source,
            "real_count": sum(int(row["label"]) == 0 for row in subset),
            "fake_count": sum(int(row["label"]) == 1 for row in subset),
        }
        for condition in CONDITIONS:
            item[f"{condition}_mean_logit_auroc"] = mean_source_values[condition].get(source)
            item[f"{condition}_mean_of_seed_auroc"] = float(np.mean([per_seed_source_values[seed][condition][source] for seed in SEEDS]))
        item["A_support_minus_dup"] = item["STRUCTURE_SUPPORT_mean_logit_auroc"] - item["STRUCTURE_DUP_CONTROL_mean_logit_auroc"]
        item["B_support_minus_structure"] = item["STRUCTURE_SUPPORT_mean_logit_auroc"] - item["STRUCTURE_ONLY_mean_logit_auroc"]
        for seed in SEEDS:
            a = per_seed_source_values[seed]["STRUCTURE_SUPPORT"].get(source)
            d = per_seed_source_values[seed]["STRUCTURE_DUP_CONTROL"].get(source)
            s = per_seed_source_values[seed]["STRUCTURE_ONLY"].get(source)
            item[f"A_seed_{seed}"] = a - d if a is not None and d is not None else None
            item[f"B_seed_{seed}"] = a - s if a is not None and s is not None else None
        item["A_seed_direction_class"] = _direction_class([item[f"A_seed_{seed}"] for seed in SEEDS])
        item["B_seed_direction_class"] = _direction_class([item[f"B_seed_{seed}"] for seed in SEEDS])
        per_source_rows.append(item)
    for item in per_source_rows:
        sources_without = {row["source_id"] for row in per_source_rows} - {item["source_id"]}
        item["A_leave_one_out_mean"] = float(np.mean([row["A_support_minus_dup"] for row in per_source_rows if row["source_id"] in sources_without])) if sources_without else None
        item["B_leave_one_out_mean"] = float(np.mean([row["B_support_minus_structure"] for row in per_source_rows if row["source_id"] in sources_without])) if sources_without else None
    _write_csv(output / "per_source_seed_differences.csv", per_source_rows)

    paired_a = _bootstrap(mean_source_values["STRUCTURE_SUPPORT"], mean_source_values["STRUCTURE_DUP_CONTROL"])
    paired_b = _bootstrap(mean_source_values["STRUCTURE_SUPPORT"], mean_source_values["STRUCTURE_ONLY"])
    leave_a = [row["A_leave_one_out_mean"] for row in per_source_rows]
    leave_b = [row["B_leave_one_out_mean"] for row in per_source_rows]

    # Q coverage comes from the saved feature arrays and identities.  The
    # support source has O/R duplicate rows; choose O for b=0 and R for b>0.
    window_stats: list[dict[str, Any]] = []
    for item in feature_manifest:
        path = root / str(item["input_path"])
        window_stats.append(_window_q(item, path))
    window_by_id = {row["window_id"]: row for row in window_stats}
    coverage_rows: list[dict[str, Any]] = []
    for split, ids in (("train", train_ids), ("validation", validation_ids)):
        selected = [row for row in window_stats if row["window_id"] in ids]
        current = _coverage_rows(selected, split, "overall", lambda row, ids=ids: row["window_id"] in ids); current["split"] = split; coverage_rows.append(current)
        for source in sorted({row["source_id"] for row in selected}):
            current = _coverage_rows(selected, source, "source", lambda row, source=source: row["source_id"] == source); current["split"] = split; coverage_rows.append(current)
        for role in ("real", "fake"):
            current = _coverage_rows(selected, role, "role", lambda row, role=role: row["role"] == role); current["split"] = split; coverage_rows.append(current)
        for b in (0.0, 0.5, 1.0):
            current = _coverage_rows(selected, str(b), "b", lambda row, b=b: abs(row["b"] - b) < 1e-9); current["split"] = split; coverage_rows.append(current)
    _write_csv(output / "q_coverage_summary.csv", coverage_rows)

    prediction_rows: list[dict[str, Any]] = []
    transition_rows: list[dict[str, Any]] = []
    for row in validation_rows:
        q = window_by_id[row["window_id"]]
        base = dict(row)
        base.update({
            "q_changed_unit_fraction": q["q_changed_fraction"],
            "target_changed_unit_fraction": q["target_changed_fraction"],
            "history_only_unit_fraction": q["history_only_fraction"],
            "constant_low_any_unit_fraction": q["constant_low_any_fraction"],
            "constant_low_both_unit_fraction": q["constant_low_both_fraction"],
            "mean_v_fraction": q["mean_v_fraction"],
            "mean_g_fraction": q["mean_g_fraction"],
            "mean_v_history_change": q["mean_v_history_change"],
            "mean_g_history_change": q["mean_g_history_change"],
            "mean_abs_history_change": q["mean_abs_history_change"],
            "matched_unit_count": q["unit_count"],
            "visibility_geometry_diff_units": q["visibility_geometry_diff_units"],
        })
        for comparator, score_key in (("STRUCTURE_ONLY", "STRUCTURE_ONLY_MEAN_LOGIT"), ("STRUCTURE_DUP_CONTROL", "STRUCTURE_DUP_CONTROL_MEAN_LOGIT")):
            fusion = float(row["STRUCTURE_SUPPORT_MEAN_LOGIT"])
            comparison = f"STRUCTURE_SUPPORT_minus_{comparator}"
            transition = _transition(int(row["label"]), float(row[score_key]), fusion)
            base[f"fusion_minus_{comparator}"] = fusion - float(row[score_key])
            aligned_change = (2 * int(row["label"]) - 1) * (fusion - float(row[score_key]))
            base[f"label_aligned_change_{comparator}"] = aligned_change
            base[f"transition_{comparator}"] = transition
            transition_rows.append({"comparison": comparison, "transition": transition, "q_group": "Q_CHANGED" if q["q_changed_units"] > 0 else "Q_UNCHANGED", "label_aligned_change": aligned_change, "real_count": int(row["label"]) == 0, "fake_count": int(row["label"]) == 1})
        prediction_rows.append(base)
    _write_csv(output / "window_prediction_changes.csv", prediction_rows)
    transition_summary: list[dict[str, Any]] = []
    for comparison in ("STRUCTURE_SUPPORT_minus_STRUCTURE_ONLY", "STRUCTURE_SUPPORT_minus_STRUCTURE_DUP_CONTROL"):
        subset = [row for row in transition_rows if row["comparison"] == comparison]
        for q_group in ("ALL", "Q_CHANGED", "Q_UNCHANGED"):
            grouped = subset if q_group == "ALL" else [row for row in subset if row["q_group"] == q_group]
            for transition in ("FN_TO_TP", "TP_TO_FN", "FP_TO_TN", "TN_TO_FP", "CORRECT_UNCHANGED", "ERROR_UNCHANGED"):
                rows = [row for row in grouped if row["transition"] == transition]
                changes = [float(row["label_aligned_change"]) for row in grouped]
                transition_summary.append({"comparison": comparison, "q_group": q_group, "transition": transition, "count": len(rows), "real_count": sum(int(row["real_count"]) for row in rows), "fake_count": sum(int(row["fake_count"]) for row in rows), "group_window_count": len(grouped), "group_real_count": sum(int(row["real_count"]) for row in grouped), "group_fake_count": sum(int(row["fake_count"]) for row in grouped), "label_aligned_change_mean": float(np.mean(changes)) if changes else None, "label_aligned_change_median": float(np.median(changes)) if changes else None})
    _write_csv(output / "error_transitions.csv", transition_summary)

    # Resolve the 30,980 count from the same O-at-b=0/R-at-b>0 source rows.
    source_support = _read_json(Path(protocol["source_root"]) / "support/window_support.json")
    support_by_id: dict[str, dict[str, Any]] = {}
    for item in source_support:
        mode = str(item.get("mode"))
        if mode == "R":
            support_by_id[str(item["window_id"])] = item
        elif mode == "O":
            support_by_id.setdefault(str(item["window_id"]), item)
    unformed = Counter()
    unformed_split = Counter()
    for item in feature_manifest:
        support = support_by_id[str(item["window_id"])]
        split = "validation" if item["window_id"] in validation_ids else "train"
        for group in support.get("grouping", {}).get("groups", []):
            if not bool(group.get("retained")):
                unformed[str(group.get("status", "UNSPECIFIED"))] += 1
                unformed_split[(split, str(group.get("status", "UNSPECIFIED")))] += 1
    if sum(unformed.values()) != 30980:
        raise ValueError(f"UNFORMED_GROUP_COUNT_MISMATCH:{sum(unformed.values())}")

    # Query-age summaries for the requested role split, without treating age as
    # a feature or claiming frame alignment between real and fake.
    age_rows = []
    for split in ("train", "validation"):
        for role in ("real", "fake"):
            subset = [row for row in window_stats if (row["window_id"] in (train_ids if split == "train" else validation_ids)) and row["role"] == role]
            ages = [row["query_age_mean_s"] for row in subset]
            age_rows.append((split, role, min(ages) if ages else None, max(ages) if ages else None, float(np.mean(ages)) if ages else None, len(subset)))

    # A compact report keeps all source rows in the CSV while exposing the
    # decision-relevant numbers and the requested limitations in prose.
    lines: list[str] = [
        "# V7 observation-support pilot: finite read-only analysis",
        "",
        "本分析只读取已保存的 logits、Q 数组、manifest 和 support 记录；没有重跑 frontend、tracking、depth、pose、segmentation、训练或模型前向。所有结论是开发集上的描述性结果。",
        "",
        "## 结论先行",
        f"- 主配对 A（STRUCTURE_SUPPORT − STRUCTURE_DUP_CONTROL）点估计为 **{_fmt(paired_a['mean'])}**，95% CI [{_fmt(paired_a['ci95'][0])}, {_fmt(paired_a['ci95'][1])}]；B（STRUCTURE_SUPPORT − STRUCTURE_ONLY）为 **{_fmt(paired_b['mean'])}**，95% CI [{_fmt(paired_b['ci95'][0])}, {_fmt(paired_b['ci95'][1])}]。两者区间均跨 0。",
        f"- A 的 source 方向为 {sum(float(row['A_support_minus_dup']) > 0 for row in per_source_rows)} 正、{sum(float(row['A_support_minus_dup']) < 0 for row in per_source_rows)} 负、{sum(float(row['A_support_minus_dup']) == 0 for row in per_source_rows)} 平；B 为 {sum(float(row['B_support_minus_structure']) > 0 for row in per_source_rows)} 正、{sum(float(row['B_support_minus_structure']) < 0 for row in per_source_rows)} 负、{sum(float(row['B_support_minus_structure']) == 0 for row in per_source_rows)} 平。seed 方向不是完全一致：seed 20260911 的 A/B 均含反向 source。",
        f"- 验证集 83 个窗口中，{sum(window_by_id[row['window_id']]['q_changed_units'] > 0 for row in validation_rows)} 个（{sum(window_by_id[row['window_id']]['q_changed_units'] > 0 for row in validation_rows)/83:.1%}）至少一个 unit 的 Q 相对历史发生变化；{sum(window_by_id[row['window_id']]['target_changed_units'] > 0 for row in validation_rows)} 个目标阶段本身发生 v/g 变化；{sum(window_by_id[row['window_id']]['constant_low_any_units'] > 0 for row in validation_rows)} 个含恒定但低于 1 的 unit。",
        "- 综合判断：**证据不足，暂不扩展实验**。正向点估计分散在多个 source，但 seed/source 方向不完全一致且 bootstrap CI 跨 0；Q 是保存的观测支撑状态，不是伪造空间真值。若未来继续，应先在未反复开发的独立数据上验证，而不是据此扩大当前实验。",
        "",
        "## 评价口径与复现",
        f"验证总体：14 source、83 windows、42 real / 41 fake；四条件窗口 ID、source 和标签列均在 logits 表中逐行共用。主分数是三个 seed logit 先逐窗口平均，再计算 AUROC；另行保留逐 seed 的 source AUROC，不将两者混为一谈。阈值固定为 logit >= 0。",
        "",
        "| 条件 | source-macro AUROC (95% CI) | pooled AUROC | AP | Precision | Recall | F1 | ACC | TN/FP/FN/TP |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for condition in CONDITIONS:
        metrics, _ = _metric_bundle(validation_rows, f"{condition}_MEAN_LOGIT")
        lines.append(f"| {condition} | {_fmt(metrics['source_macro_auroc'])} [{_fmt(metrics['source_macro_ci_low'])}, {_fmt(metrics['source_macro_ci_high'])}] | {_fmt(metrics['pooled_auroc'])} | {_fmt(metrics['pooled_ap'])} | {_fmt(metrics['precision'])} | {_fmt(metrics['recall'])} | {_fmt(metrics['f1'])} | {_fmt(metrics['accuracy'])} | {metrics['tn']}/{metrics['fp']}/{metrics['fn']}/{metrics['tp']} |")
    max_error = max((float(row["absolute_error"]) for row in metric_rows if row["absolute_error"] not in (None, "")), default=0.0)
    lines += ["", f"复算与原 evaluation/metrics.csv 的最大有限字段绝对误差为 {_fmt(max_error, 12)}；详细逐字段结果见 `metric_reproduction.csv`。", ""]

    lines += ["## source / seed 收益分布", "", "| 比较 | 点估计 | 95% CI | source 正/负/平 | 中位数 | 范围 |", "|---|---:|---:|---:|---:|---:|"]
    for name, values, paired in (("A: SUPPORT−DUP", [row["A_support_minus_dup"] for row in per_source_rows], paired_a), ("B: SUPPORT−STRUCTURE", [row["B_support_minus_structure"] for row in per_source_rows], paired_b)):
        lines.append(f"| {name} | {_fmt(paired['mean'])} | [{_fmt(paired['ci95'][0])}, {_fmt(paired['ci95'][1])}] | {sum(x > 0 for x in values)}/{sum(x < 0 for x in values)}/{sum(x == 0 for x in values)} | {_fmt(np.median(values))} | [{_fmt(min(values))}, {_fmt(max(values))}] |")
    max_a = max(per_source_rows, key=lambda row: float(row["A_support_minus_dup"]))
    min_a = min(per_source_rows, key=lambda row: float(row["A_support_minus_dup"]))
    max_b = max(per_source_rows, key=lambda row: float(row["B_support_minus_structure"]))
    min_b = min(per_source_rows, key=lambda row: float(row["B_support_minus_structure"]))
    lines.append(f"最大/最小 source：A {max_a['source_id']} ({_fmt(max_a['A_support_minus_dup'])}) / {min_a['source_id']} ({_fmt(min_a['A_support_minus_dup'])})；B {max_b['source_id']} ({_fmt(max_b['B_support_minus_structure'])}) / {min_b['source_id']} ({_fmt(min_b['B_support_minus_structure'])})。")
    lines += ["", "逐 source、逐 seed 的 A/B 差值和 leave-one-source-out 影响范围见 `per_source_seed_differences.csv`。删除一个 source 的 A 平均差范围为 [" + _fmt(min(leave_a)) + ", " + _fmt(max(leave_a)) + f"]，B 为 [{_fmt(min(leave_b))}, {_fmt(max(leave_b))}]；这只是影响分析，不改变主评价总体。", ""]
    lines += ["逐 source 的 `mean_of_seed_auroc` 也写入 `per_source_seed_differences.csv`；它是先逐 seed 计算 AUROC 再平均，和主口径（先平均 logits 后算 AUROC）不同，不能互换。主口径的三个 seed 方向计数如下："]
    for seed in SEEDS:
        a_values = [per_source_rows[i][f"A_seed_{seed}"] for i in range(len(per_source_rows))]
        b_values = [per_source_rows[i][f"B_seed_{seed}"] for i in range(len(per_source_rows))]
        lines.append(f"- seed {seed}: A {_fmt(np.mean(a_values))}，方向 {_direction_class(a_values)}（正/负/零 {sum(x > 0 for x in a_values)}/{sum(x < 0 for x in a_values)}/{sum(x == 0 for x in a_values)}）；B {_fmt(np.mean(b_values))}，方向 {_direction_class(b_values)}（正/负/零 {sum(x > 0 for x in b_values)}/{sum(x < 0 for x in b_values)}/{sum(x == 0 for x in b_values)}）。")
    lines.append("")

    lines += ["## Q 覆盖与查询年龄", "", "Q 定义为 `[visibility_fraction, geometry_fraction, visibility_minus_history, geometry_minus_history]`，分母是原始历史组成员；S 使用共同有效成员。目标阶段恒定但相对历史非零的 unit 单独记为 `history_only_change`（通过 q_changed 且 target_changed=false 的计数可得），不把它称为目标阶段动态。", ""]
    for scope in ("train", "validation"):
        row = next(item for item in coverage_rows if item["scope_type"] == "overall" and item["scope"] == scope)
        lines.append(f"- {scope}: {row['unit_count']} units，{row['q_changed_units']} Q changed ({row['q_changed_units']/row['unit_count']:.2%})，{row['target_changed_units']} target v/g changed ({row['target_changed_units']/row['unit_count']:.2%})，{row['history_only_units']} history-only，{row['fully_one_units']} fully-one，{row['constant_low_any_units']} constant-low-any，{row['raw_common_diff_units']} raw/common member count different；有 Q change 的窗口 {row['windows_with_q_change']}/{row['window_count']}，有 target change 的窗口 {row['windows_with_target_change']}/{row['window_count']}，含 constant-low 的窗口 {row['windows_with_constant_low']}/{row['window_count']}，平均 changed-unit fraction {_fmt(row['mean_changed_unit_fraction'])}。")
    lines.append("")
    lines.append("验证集按 b 的覆盖摘要（每行按窗口聚合，完整数值见 q_coverage_summary.csv）：")
    lines += ["", "| b | windows | units | Q changed units | target changed units | constant-low units | mean Q-changed fraction |", "|---:|---:|---:|---:|---:|---:|---:|"]
    for row in coverage_rows:
        if row["split"] == "validation" and row["scope_type"] == "b" and row["scope"] in {"0.0", "0.5", "1.0"}:
            lines.append(f"| {row['scope']} | {row['window_count']} | {row['unit_count']} | {row['q_changed_units']} | {row['target_changed_units']} | {row['constant_low_any_units']} | {_fmt(row['mean_changed_unit_fraction'])} |")
    lines += ["", "查询年龄按角色（实际 target PTS 减 query 初始化 PTS）如下；real/fake 的动作时间并未被假设逐帧对应：", "", "| split | role | windows | age min | age max | age mean |", "|---|---|---:|---:|---:|---:|"]
    for split, role, minimum, maximum, mean, count in age_rows:
        lines.append(f"| {split} | {role} | {count} | {_fmt(minimum)} | {_fmt(maximum)} | {_fmt(mean)} |")
    lines += ["", "现有年龄范围在 real/fake 间相近；本分析没有证据显示明显的角色-查询年龄混淆。visibility 与 geometry 的逐成员差异极少（原 summary 相等比例约 0.99999），但它们仍只是前端保存的判断，不是真实遮挡/几何真值。", ""]

    lines += ["## 预测改变与错误转移", "", "| comparison | transition | count | real | fake |", "|---|---|---:|---:|---:|"]
    for row in transition_summary:
        if row["q_group"] != "ALL":
            continue
        lines.append(f"| {row['comparison']} | {row['transition']} | {row['count']} | {row['real_count']} | {row['fake_count']} |")
    lines += ["", "按 Q_CHANGED/Q_UNCHANGED 分层的转移与 label-aligned change（每组同时给出窗口和 real/fake 数）见下表：", "", "| comparison | Q group | group n (real/fake) | FN→TP | TP→FN | FP→TN | TN→FP | aligned mean/median |", "|---|---|---:|---:|---:|---:|---:|---:|"]
    for comparison in ("STRUCTURE_SUPPORT_minus_STRUCTURE_ONLY", "STRUCTURE_SUPPORT_minus_STRUCTURE_DUP_CONTROL"):
        for q_group in ("Q_CHANGED", "Q_UNCHANGED"):
            grouped = [row for row in transition_summary if row["comparison"] == comparison and row["q_group"] == q_group]
            by_transition = {row["transition"]: row for row in grouped}
            first = grouped[0]
            aligned = [float(row["label_aligned_change_mean"]) for row in grouped if row["label_aligned_change_mean"] is not None]
            medians = [float(row["label_aligned_change_median"]) for row in grouped if row["label_aligned_change_median"] is not None]
            lines.append(f"| {comparison} | {q_group} | {first['group_window_count']} ({first['group_real_count']}/{first['group_fake_count']}) | {by_transition['FN_TO_TP']['count']} | {by_transition['TP_TO_FN']['count']} | {by_transition['FP_TO_TN']['count']} | {by_transition['TN_TO_FP']['count']} | {_fmt(np.mean(aligned) if aligned else None)}/{_fmt(np.mean(medians) if medians else None)} |")
    lines += ["", "逐窗口 logits、Q 变化比例、平均 v/g、相对历史变化和 label-aligned logit change 见 `window_prediction_changes.csv`。label-aligned change 只是描述量；融合重新训练后，Q 不变窗口也可能改变预测，不能据此建立因果关系。", ""]

    lines += ["## 未形成结构的原始组", f"报告中的 30,980 是当前 801 个已选 feature window 中，按 b=0 取 O、非零 b 取 R 的 `grouping.groups` 内 `retained=false` 记录数，即组-窗口记录的总数，不是去重实体、像素区域或失真区域。唯一主失败状态为 `SUPPORT_INSUFFICIENT_GROUP_SIZE`；train {unformed_split.get(('train', 'SUPPORT_INSUFFICIENT_GROUP_SIZE'), 0)}，validation {unformed_split.get(('validation', 'SUPPORT_INSUFFICIENT_GROUP_SIZE'), 0)}。这些组未进入本轮分类器，不能转换成空间漏检率。", ""]

    lines += ["## 限制与建议", "- 14-source/83-window 验证集已被多轮开发使用，不能视为 sealed test。", "- 主配对 CI 均跨 0；source 正向点估计分散但并非所有 seed 同向，不能写成稳定增益。", "- 共同有效 unit 与已保存特征造成 survivor/measurement selection；Q 变化表示观测支撑状态变化，不等于局部伪造真值。", "- 无空间真值、未形成结构的组没有进入分类器，本轮不能回答失真部位覆盖。", "- 建议状态：证据不足，暂不扩展当前实验；若继续，优先使用未参与开发的独立 source/时间总体作一次冻结验证，而不是据本分析调参或筛 source。", ""]
    (output / "report.md").write_text("\n".join(lines), encoding="utf-8")
    summary = {
        "validation_window_count": len(validation_rows),
        "validation_source_count": len({row["source_id"] for row in validation_rows}),
        "paired_A": paired_a,
        "paired_B": paired_b,
        "source_A_positive_negative_tie": [sum(float(row["A_support_minus_dup"]) > 0 for row in per_source_rows), sum(float(row["A_support_minus_dup"]) < 0 for row in per_source_rows), sum(float(row["A_support_minus_dup"]) == 0 for row in per_source_rows)],
        "source_B_positive_negative_tie": [sum(float(row["B_support_minus_structure"]) > 0 for row in per_source_rows), sum(float(row["B_support_minus_structure"]) < 0 for row in per_source_rows), sum(float(row["B_support_minus_structure"]) == 0 for row in per_source_rows)],
        "q_validation_windows_changed": sum(window_by_id[row["window_id"]]["q_changed_units"] > 0 for row in validation_rows),
        "q_validation_windows_target_changed": sum(window_by_id[row["window_id"]]["target_changed_units"] > 0 for row in validation_rows),
        "q_validation_windows_constant_low": sum(window_by_id[row["window_id"]]["constant_low_any_units"] > 0 for row in validation_rows),
        "q_validation_history_only_units": sum(window_by_id[row["window_id"]]["history_only_units"] for row in validation_rows),
        "unformed_group_count": int(sum(unformed.values())),
        "unformed_group_reasons": dict(unformed),
        "max_metric_reproduction_error": max_error,
        "source_count_and_window_count_verified": True,
    }
    (output / "analysis_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("/root/autodl-tmp/data/sparse_3d_forgery_detection/derived/v7_activityforensics_observation_support_pilot_v1"))
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    output = args.output or args.root / "analysis_v1"
    summary = analyze(args.root, output)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
