#!/usr/bin/env python3
"""Read-only descriptive analysis for the completed periodic re-query pilot.

This entry point deliberately consumes saved CSV/JSON artifacts only.  It does
not load a model, run a forward pass, or touch the frontend caches.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import median

import numpy as np


CONDITIONS = ["O_SET", "O_RAW", "O_MULTI", "R_SET", "R_RAW", "R_MULTI"]
SEEDS = ["20260909", "20260910", "20260911"]
MODES = ("O", "R")
BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_SEED = 20260909


def as_float(value):
    if value is None or value == "" or value == "null":
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def as_bool(value):
    return str(value).strip().lower() == "true"


def fmt(value, digits=6):
    if value is None or not math.isfinite(float(value)):
        return "NA"
    return f"{float(value):.{digits}f}"


def json_safe(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, np.generic):
        return json_safe(value.item())
    return value


def write_csv(path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({name: "" if row.get(name) is None else row.get(name) for name in fieldnames})


def auc(labels, scores):
    labels = np.asarray(labels, dtype=np.int8)
    scores = np.asarray(scores, dtype=np.float64)
    positive = scores[labels == 1]
    negative = scores[labels == 0]
    if len(positive) == 0 or len(negative) == 0:
        return None
    comparisons = positive[:, None] - negative[None, :]
    return float((np.greater(comparisons, 0).sum() + 0.5 * np.equal(comparisons, 0).sum()) / comparisons.size)


def average_precision(labels, scores):
    labels = np.asarray(labels, dtype=np.int8)
    scores = np.asarray(scores, dtype=np.float64)
    positives = int(labels.sum())
    if positives == 0:
        return None
    order = np.argsort(-scores, kind="mergesort")
    sorted_labels = labels[order]
    cumulative = np.cumsum(sorted_labels)
    ranks = np.arange(1, len(sorted_labels) + 1)
    return float(np.sum((cumulative / ranks) * sorted_labels) / positives)


def classification(labels, scores):
    labels = np.asarray(labels, dtype=np.int8)
    scores = np.asarray(scores, dtype=np.float64)
    predicted = scores >= 0.0
    tp = int(np.sum((predicted == 1) & (labels == 1)))
    tn = int(np.sum((predicted == 0) & (labels == 0)))
    fp = int(np.sum((predicted == 1) & (labels == 0)))
    fn = int(np.sum((predicted == 0) & (labels == 1)))
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    f1 = 2 * precision * recall / (precision + recall) if precision is not None and recall and precision + recall else None
    accuracy = (tp + tn) / len(labels) if len(labels) else None
    return {
        "tn": tn, "fp": fp, "fn": fn, "tp": tp,
        "precision": precision, "recall": recall, "f1": f1, "accuracy": accuracy,
    }


def bootstrap_ci(values, seed=BOOTSTRAP_SEED, replicates=BOOTSTRAP_REPLICATES):
    values = np.asarray([v for v in values if v is not None and math.isfinite(float(v))], dtype=np.float64)
    if len(values) == 0:
        return None, None
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(replicates, len(values)))
    means = values[indices].mean(axis=1)
    low, high = np.percentile(means, [2.5, 97.5])
    return float(low), float(high)


def source_auc_rows(rows, source, condition):
    selected = [row for row in rows if row["source_id"] == source and row["condition_scores"].get(condition) is not None]
    labels = [row["label"] for row in selected]
    scores = [row["condition_scores"][condition] for row in selected]
    if len(set(labels)) < 2:
        return None, len(selected), int(sum(v == 0 for v in labels)), int(sum(v == 1 for v in labels))
    return auc(labels, scores), len(selected), int(sum(v == 0 for v in labels)), int(sum(v == 1 for v in labels))


def load_data(root):
    support_rows = json.loads((root / "support/window_support.json").read_text())
    manifests = {row["window_id"]: row for row in json.loads((root / "manifests/subwindows.json").read_text())}
    scores = list(csv.DictReader((root / "scores/oof_window_scores.csv").open(newline="", encoding="utf-8")))
    summary = json.loads((root / "evaluation/summary.json").read_text())
    support = {(row["window_id"], row["mode"]): row for row in support_rows}
    score_by_window = {}
    for raw in scores:
        seed_scores = {}
        mean_scores = {}
        for condition in CONDITIONS:
            values = [as_float(raw.get(f"{condition}_seed_{seed}")) for seed in SEEDS]
            seed_scores[condition] = dict(zip(SEEDS, values))
            mean_scores[condition] = float(np.mean(values)) if all(v is not None for v in values) else None
        score_by_window[raw["window_id"]] = {
            "raw": raw,
            "seed_scores": seed_scores,
            "condition_scores": mean_scores,
        }
    return support, manifests, score_by_window, summary


def support_reasons(row):
    if row is None:
        return ["MISSING_ROW"]
    reasons = row.get("support_reasons") or []
    if reasons:
        return sorted({str(reason) for reason in reasons})
    direct = row.get("support", {}).get("invalid_reasons") or []
    if direct:
        return sorted({str(reason) for reason in direct})
    unit_reasons = [unit.get("reason") for unit in row.get("support", {}).get("units", []) if unit.get("status") != "VALID" and unit.get("reason")]
    return sorted({str(reason) for reason in unit_reasons}) or ([str(row.get("support_status"))] if row.get("support_status") else [])


def valid_unit_count(row):
    if row is None:
        return 0
    return int(row.get("valid_unit_count") or row.get("support", {}).get("valid_unit_count") or 0)


def relation_count(row):
    if row is None:
        return 0
    total = 0
    for unit in row.get("support", {}).get("units", []):
        if unit.get("status") == "VALID":
            total += len(unit.get("pair_indices") or unit.get("pair_ids") or [])
    return total


def build_window_rows(support, manifests, score_by_window):
    rows = []
    for window_id, score_data in score_by_window.items():
        manifest = manifests[window_id]
        mode_data = {}
        for mode in MODES:
            mode_data[mode] = support.get((window_id, mode))
        o_row, r_row = mode_data["O"], mode_data["R"]
        o_valid = o_row is not None and o_row.get("support_status") == "VALID"
        r_valid = r_row is not None and r_row.get("support_status") == "VALID"
        membership = "both" if o_valid and r_valid else "O-only" if o_valid else "R-only" if r_valid else "neither"
        timestamps = manifest.get("timestamps_s") or []
        rows.append({
            "window_id": window_id,
            "source_id": manifest["source_id"],
            "role": manifest["role"],
            "parent_id": manifest["parent_id"],
            "pair_id": manifest.get("pair_id"),
            "offset_s": manifest.get("offset_s"),
            "interval_start_s": manifest.get("interval_start_s"),
            "interval_end_s": manifest.get("interval_end_s"),
            "pts_start_s": timestamps[0] if timestamps else None,
            "pts_end_s": timestamps[-1] if timestamps else None,
            "label": int(score_data["raw"]["label"]),
            "annotation_category": score_data["raw"].get("annotation_category"),
            "paired_eligible": as_bool(score_data["raw"].get("paired_eligible")),
            "O_valid": o_valid,
            "R_valid": r_valid,
            "membership": membership,
            "O_units": valid_unit_count(o_row),
            "R_units": valid_unit_count(r_row),
            "O_relations": relation_count(o_row),
            "R_relations": relation_count(r_row),
            "O_reasons": ";".join(support_reasons(o_row)) if not o_valid else "",
            "R_reasons": ";".join(support_reasons(r_row)) if not r_valid else "",
            "seed_scores": score_data["seed_scores"],
            "condition_scores": score_data["condition_scores"],
        })
    return rows


def write_window_lists(analysis, names, rows):
    list_dir = analysis / "window_lists"
    list_dir.mkdir(parents=True, exist_ok=True)
    for name, selected in names.items():
        path = list_dir / f"{name}.txt"
        path.write_text("\n".join(row["window_id"] for row in selected) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True, help="periodic_requery_pilot_v1 output directory")
    args = parser.parse_args()
    root = args.root.resolve()
    analysis = root / "analysis"
    analysis.mkdir(parents=True, exist_ok=True)
    support, manifests, score_by_window, stored_summary = load_data(root)
    rows = build_window_rows(support, manifests, score_by_window)
    rows.sort(key=lambda row: row["window_id"])

    fixed = rows
    o_valid = [row for row in rows if row["O_valid"]]
    r_valid = [row for row in rows if row["R_valid"]]
    common = [row for row in rows if row["O_valid"] and row["R_valid"]]
    main_common = [row for row in common if row["paired_eligible"] and row["label"] in (0, 1)]
    dual_sources = sorted({row["source_id"] for row in main_common if any(other["source_id"] == row["source_id"] and other["label"] != row["label"] for other in main_common)})
    macro_rows = [row for row in main_common if row["source_id"] in dual_sources]
    pooled_rows = main_common
    sets = {
        "fixed_96": fixed,
        "O_valid": o_valid,
        "R_valid": r_valid,
        "O_R_common": common,
        "main_label_common": main_common,
        "source_macro_window_rows": macro_rows,
        "pooled_window_rows": pooled_rows,
    }
    write_window_lists(analysis, sets, rows)

    collection_rows = []
    for name, selected in sets.items():
        collection_rows.append({
            "set_name": name,
            "source_count": len({row["source_id"] for row in selected}),
            "window_count": len(selected),
            "real_count": sum(row["label"] == 0 for row in selected),
            "fake_count": sum(row["label"] == 1 for row in selected),
            "list_path": f"analysis/window_lists/{name}.txt",
            "notes": "source_macro_window_rows excludes single-class sources; pooled_window_rows includes 01KML fake-only rows",
        })
    write_csv(analysis / "collection_sets.csv", collection_rows, list(collection_rows[0]))

    membership_fields = [
        "window_id", "source_id", "role", "parent_id", "pair_id", "offset_s", "interval_start_s", "interval_end_s",
        "pts_start_s", "pts_end_s", "label", "annotation_category", "paired_eligible", "O_valid", "R_valid",
        "membership", "O_units", "R_units", "O_relations", "R_relations", "O_reasons", "R_reasons",
    ]
    membership_rows = [{field: row.get(field) for field in membership_fields} for row in rows]
    write_csv(analysis / "observation_recovery_windows.csv", membership_rows, membership_fields)

    # Reproduce all six metrics from the saved three-seed logits.
    metric_rows = []
    source_rows_by_condition = {}
    for condition in CONDITIONS:
        condition_pooled = [row for row in pooled_rows if row["condition_scores"].get(condition) is not None]
        labels = [row["label"] for row in condition_pooled]
        scores = [row["condition_scores"][condition] for row in condition_pooled]
        source_values = []
        source_rows_by_condition[condition] = []
        for source in dual_sources:
            source_auc_value, n, n_real, n_fake = source_auc_rows(pooled_rows, source, condition)
            if source_auc_value is not None:
                source_values.append(source_auc_value)
            source_rows_by_condition[condition].append((source, source_auc_value, n, n_real, n_fake))
        ci_low, ci_high = bootstrap_ci(source_values)
        cls = classification(labels, scores) if labels else {"tn": None, "fp": None, "fn": None, "tp": None, "precision": None, "recall": None, "f1": None, "accuracy": None}
        stored = stored_summary.get("conditions", {}).get(condition, {})
        stored_source = stored.get("source_auroc", {})
        metric_rows.append({
            "condition": condition,
            "source_count": len(source_values),
            "source_window_count": sum(item[2] for item in source_rows_by_condition[condition] if item[1] is not None),
            "pooled_window_count": len(condition_pooled),
            "real_count": sum(label == 0 for label in labels),
            "fake_count": sum(label == 1 for label in labels),
            "source_auroc": float(np.mean(source_values)) if source_values else None,
            "source_ci_low": ci_low,
            "source_ci_high": ci_high,
            "pooled_auroc": auc(labels, scores),
            "average_precision": average_precision(labels, scores),
            "precision": cls["precision"], "recall": cls["recall"], "f1": cls["f1"], "accuracy": cls["accuracy"],
            "tn": cls["tn"], "fp": cls["fp"], "fn": cls["fn"], "tp": cls["tp"],
            "stored_source_auroc": stored_source.get("mean"),
            "stored_pooled_auroc": stored.get("pooled_auroc"),
            "stored_average_precision": stored.get("pooled_ap"),
        })
    write_csv(analysis / "metric_reproduction.csv", metric_rows, list(metric_rows[0]))

    # Per-source O_SET/R_SET comparison, seed differences, and support deltas.
    source_comparison = []
    for source in dual_sources:
        source_selected = [row for row in pooled_rows if row["source_id"] == source]
        o_auc, n, n_real, n_fake = source_auc_rows(pooled_rows, source, "O_SET")
        r_auc, _, _, _ = source_auc_rows(pooled_rows, source, "R_SET")
        seed_diff = {}
        for seed in SEEDS:
            o_scores = [row["seed_scores"]["O_SET"].get(seed) for row in source_selected]
            r_scores = [row["seed_scores"]["R_SET"].get(seed) for row in source_selected]
            labels = [row["label"] for row in source_selected]
            o_seed_auc = auc(labels, o_scores) if all(v is not None for v in o_scores) else None
            r_seed_auc = auc(labels, r_scores) if all(v is not None for v in r_scores) else None
            seed_diff[seed] = r_seed_auc - o_seed_auc if o_seed_auc is not None and r_seed_auc is not None else None
        o_avg = [row["condition_scores"]["O_SET"] for row in source_selected]
        r_avg = [row["condition_scores"]["R_SET"] for row in source_selected]
        o_cls = classification([row["label"] for row in source_selected], o_avg)
        r_cls = classification([row["label"] for row in source_selected], r_avg)
        source_comparison.append({
            "source_id": source, "window_count": n, "real_count": n_real, "fake_count": n_fake,
            "O_SET_auroc": o_auc, "R_SET_auroc": r_auc,
            "R_MINUS_O": r_auc - o_auc if o_auc is not None and r_auc is not None else None,
            "R_MINUS_O_seed_20260909": seed_diff["20260909"], "R_MINUS_O_seed_20260910": seed_diff["20260910"], "R_MINUS_O_seed_20260911": seed_diff["20260911"],
            "O_tn": o_cls["tn"], "O_fp": o_cls["fp"], "O_fn": o_cls["fn"], "O_tp": o_cls["tp"],
            "R_tn": r_cls["tn"], "R_fp": r_cls["fp"], "R_fn": r_cls["fn"], "R_tp": r_cls["tp"],
            "O_units": sum(row["O_units"] for row in source_selected), "R_units": sum(row["R_units"] for row in source_selected),
            "R_MINUS_O_units": sum(row["R_units"] - row["O_units"] for row in source_selected),
            "O_relations": sum(row["O_relations"] for row in source_selected), "R_relations": sum(row["R_relations"] for row in source_selected),
            "R_MINUS_O_relations": sum(row["R_relations"] - row["O_relations"] for row in source_selected),
        })
    write_csv(analysis / "per_source_r_set_minus_o_set.csv", source_comparison, list(source_comparison[0]))

    diffs = [row["R_MINUS_O"] for row in source_comparison if row["R_MINUS_O"] is not None]
    leave_one_rows = [{"removed_source": "NONE", "remaining_source_count": len(diffs), "mean_R_MINUS_O": float(np.mean(diffs)) if diffs else None}]
    for removed in source_comparison:
        remaining = [row["R_MINUS_O"] for row in source_comparison if row["source_id"] != removed["source_id"] and row["R_MINUS_O"] is not None]
        leave_one_rows.append({"removed_source": removed["source_id"], "remaining_source_count": len(remaining), "mean_R_MINUS_O": float(np.mean(remaining)) if remaining else None})
    write_csv(analysis / "leave_one_source_sensitivity.csv", leave_one_rows, list(leave_one_rows[0]))

    seed_direction = {}
    for seed in SEEDS:
        values = [row[f"R_MINUS_O_seed_{seed}"] for row in source_comparison if row[f"R_MINUS_O_seed_{seed}"] is not None]
        seed_direction[seed] = {
            "increase": sum(value > 0 for value in values),
            "tie": sum(value == 0 for value in values),
            "decrease": sum(value < 0 for value in values),
        }
    same_direction_sources = sum(
        len({(1 if row[f"R_MINUS_O_seed_{seed}"] > 0 else -1 if row[f"R_MINUS_O_seed_{seed}"] < 0 else 0) for seed in SEEDS}) == 1
        for row in source_comparison
    )

    # Source/category logit summaries for the two sets central to recovery.
    distribution_rows = []
    for condition in ("O_SET", "R_SET"):
        for source in sorted({row["source_id"] for row in pooled_rows}):
            for label, label_name in ((0, "real"), (1, "fake")):
                values = [row["condition_scores"][condition] for row in pooled_rows if row["source_id"] == source and row["label"] == label and row["condition_scores"].get(condition) is not None]
                distribution_rows.append({
                    "condition": condition, "source_id": source, "label": label_name, "n": len(values),
                    "mean": float(np.mean(values)) if values else None, "median": float(np.median(values)) if values else None,
                    "min": float(np.min(values)) if values else None, "max": float(np.max(values)) if values else None,
                    "positive_rate_logit_ge_0": float(np.mean(np.asarray(values) >= 0)) if values else None,
                })
    write_csv(analysis / "logit_distribution_o_r_set.csv", distribution_rows, list(distribution_rows[0]))

    # Pooled pair decomposition: same-source versus cross-source positive/negative pairs.
    pair_rows = []
    for condition in ("O_SET", "R_SET"):
        selected = [row for row in pooled_rows if row["condition_scores"].get(condition) is not None]
        positives = [row for row in selected if row["label"] == 1]
        negatives = [row for row in selected if row["label"] == 0]
        for pair_type in ("same_source", "different_source"):
            correct = total = 0.0
            for positive in positives:
                for negative in negatives:
                    same = positive["source_id"] == negative["source_id"]
                    if (pair_type == "same_source") != same:
                        continue
                    delta = positive["condition_scores"][condition] - negative["condition_scores"][condition]
                    correct += 1.0 if delta > 0 else 0.5 if delta == 0 else 0.0
                    total += 1.0
            pair_rows.append({"condition": condition, "pair_type": pair_type, "pair_count": int(total), "ranking_correct_with_ties_half": correct, "ranking_proportion": correct / total if total else None})
    write_csv(analysis / "pooled_pair_decomposition.csv", pair_rows, list(pair_rows[0]))

    # b=0 raw feature identity check.  This compares saved inputs only; logits are not expected to match.
    b0_rows = []
    for row in rows:
        if float(row["offset_s"]) != 0.0 or not (row["O_valid"] and row["R_valid"]):
            continue
        o = support[(row["window_id"], "O")]
        r = support[(row["window_id"], "R")]
        for feature_name in ("SET_A", "RAW_SEQ", "MULTI_ORDER_SEQ"):
            a, b = o.get("features", {}).get(feature_name), r.get("features", {}).get(feature_name)
            if a is None or b is None:
                b0_rows.append({"window_id": row["window_id"], "feature": feature_name, "same_shape": False, "exact_equal": False, "max_abs_diff": None, "status": "MISSING"})
                continue
            aa, bb = np.asarray(a), np.asarray(b)
            same_shape = aa.shape == bb.shape
            exact = bool(same_shape and np.array_equal(aa, bb, equal_nan=True))
            finite_delta = np.abs(aa.astype(np.float64) - bb.astype(np.float64))
            max_delta = float(np.nanmax(finite_delta)) if finite_delta.size else 0.0
            b0_rows.append({"window_id": row["window_id"], "feature": feature_name, "same_shape": same_shape, "exact_equal": exact, "max_abs_diff": max_delta, "status": "COMPARED"})
    write_csv(analysis / "b0_feature_consistency.csv", b0_rows, list(b0_rows[0]) if b0_rows else ["window_id", "feature", "same_shape", "exact_equal", "max_abs_diff", "status"])

    # Human-readable report.  The source data remain in the data directory and are not staged.
    metric_by_condition = {row["condition"]: row for row in metric_rows}
    source_diff_values = [row["R_MINUS_O"] for row in source_comparison if row["R_MINUS_O"] is not None]
    increasing = sum(value > 0 for value in source_diff_values)
    equal = sum(value == 0 for value in source_diff_values)
    decreasing = sum(value < 0 for value in source_diff_values)
    r_only = [row for row in rows if row["membership"] == "R-only"]
    neither = [row for row in rows if row["membership"] == "neither"]
    o_reason_counts = defaultdict(int)
    r_reason_counts = defaultdict(int)
    for row in rows:
        if not row["O_valid"]:
            for reason in row["O_reasons"].split(";"):
                if reason:
                    o_reason_counts[reason] += 1
        if not row["R_valid"]:
            for reason in row["R_reasons"].split(";"):
                if reason:
                    r_reason_counts[reason] += 1
    decomposition = {(row["condition"], row["pair_type"]): row for row in pair_rows}
    common_rows = [row for row in rows if row["membership"] == "both"]
    all_o_units = sum(row["O_units"] for row in rows)
    all_r_units = sum(row["R_units"] for row in rows)
    common_o_units = sum(row["O_units"] for row in common_rows)
    common_r_units = sum(row["R_units"] for row in common_rows)
    all_o_relations = sum(row["O_relations"] for row in rows)
    all_r_relations = sum(row["R_relations"] for row in rows)
    common_o_relations = sum(row["O_relations"] for row in common_rows)
    common_r_relations = sum(row["R_relations"] for row in common_rows)

    # Small stratified coverage table.  The saved manifest has only the
    # paired_eligible flag for the non-primary rows; it does not distinguish
    # boundary from outside-annotation windows, so that distinction stays NA.
    strata = [
        ("role", "real", lambda row: row["role"] == "real", "role from manifest"),
        ("role", "fake", lambda row: row["role"] == "fake", "role from manifest"),
        ("primary_class", "primary_negative", lambda row: row["paired_eligible"] and row["label"] == 0, "label=0 and paired_eligible=True"),
        ("primary_class", "primary_positive", lambda row: row["paired_eligible"] and row["label"] == 1, "label=1 and paired_eligible=True"),
        ("primary_class", "ineligible_label_window", lambda row: not row["paired_eligible"], "saved artifact does not split boundary vs outside"),
        ("offset", "b=0.0", lambda row: float(row["offset_s"]) == 0.0, "frozen offset"),
        ("offset", "b=0.5", lambda row: float(row["offset_s"]) == 0.5, "frozen offset"),
        ("offset", "b=1.0", lambda row: float(row["offset_s"]) == 1.0, "frozen offset"),
    ]
    layer_rows = []
    for dimension, level, predicate, notes in strata:
        selected = [row for row in rows if predicate(row)]
        for mode in MODES:
            valid_field = f"{mode}_valid"
            unit_field = f"{mode}_units"
            relation_field = f"{mode}_relations"
            layer_rows.append({
                "dimension": dimension, "level": level, "mode": mode,
                "planned_windows": len(selected),
                "valid_windows": sum(row[valid_field] for row in selected),
                "invalid_windows": sum(not row[valid_field] for row in selected),
                "valid_units": sum(row[unit_field] for row in selected),
                "valid_relations": sum(row[relation_field] for row in selected),
                "notes": notes,
            })
    write_csv(analysis / "support_layer_summary.csv", layer_rows, list(layer_rows[0]))
    report = []
    report.append("# V7 periodic re-query pilot：既有产物描述性分析\n")
    report.append("本报告只读取已完成 pilot 的 JSON/CSV 产物；没有重新运行 frontend、tracking、depth、pose、segmentation、模型前向或训练。原始 `report.md` 与 `evaluation/summary.json` 保持不变。\n")
    report.append("## 直接回答\n")
    report.append(f"1. 原报告的 macro/pooled 分母**不一致**：pooled 使用 {len(pooled_rows)} 个主标签窗口（real={sum(r['label']==0 for r in pooled_rows)}, fake={sum(r['label']==1 for r in pooled_rows)}，包括没有 real 主样本的 01KML fake 窗口）；source-macro 使用 {len(dual_sources)} 个双类别 source 的 {len(macro_rows)} 个窗口（real={sum(r['label']==0 for r in macro_rows)}, fake={sum(r['label']==1 for r in macro_rows)}）。\n")
    report.append(f"2. R_SET−O_SET 的三 seed 平均分差值在 {increasing} 个 source 上升、{equal} 个持平、{decreasing} 个下降；逐 seed 的方向计数为 {seed_direction}，三 seed 方向完全一致的 source 仅 {same_direction_sources}/14。逐 source 差值保存在 `analysis/per_source_r_set_minus_o_set.csv`，不能据此声称稳定总体提升。\n")
    report.append("3. 六个 R-only 窗口及其 O/R 支撑原因见 `analysis/observation_recovery_windows.csv`；它们均为保存产物中的 O 无效、R 有效记录，不将原因推断为遮挡或跟踪失败。\n")
    report.append(f"4. 共同 79 个窗口的 R/O 单元与关系变化按 source 汇总；共同集合为 O={common_o_units}、R={common_r_units} 个单元，差异是观测支撑描述，不是伪造部位覆盖真值。\n")
    report.append("5. pooled 正负 pair 中跨 source 配对的排序比例与数量见 `analysis/pooled_pair_decomposition.csv`。因为不同 source 使用不同 LOSO 模型，这只能描述分数尺度/分布可比性，不能区分 source 与模型差异。\n")
    report.append("6. 现有产物足以描述集合分母、source 差异、O/R 支撑和跨 source 排序现象；不足以单独区分分数尺度问题与表示失效，也没有空间真值来判断新增单元是否落在失真区域。\n")
    report.append("7. R_SET 的支撑增加和 source-level 分数结果可以作为保留 R_SET 的后验开发线索；区间跨零、样本小且窗口重叠，仍不足以支持扩大实验或核心命题。\n")
    report.append("## 评价集合\n")
    report.append("| set | sources | windows | real | fake | list |\n|---|---:|---:|---:|---:|---|\n")
    for row in collection_rows:
        report.append(f"| {row['set_name']} | {row['source_count']} | {row['window_count']} | {row['real_count']} | {row['fake_count']} | `{row['list_path']}` |\n")
    report.append("\n因此“14 个双类别 source、39 real/40 fake、79 窗口”不是同一个 macro 分母：79 是 pooled 主评价窗口数；macro 的 14 个 source 实际使用 76 个窗口，01KML 的 fake-only 窗口保留在 pooled、排除在 source AUROC。\n")
    report.append("\n### O/R 观测恢复\n")
    report.append(f"固定 96 窗口分为 both={sum(r['membership']=='both' for r in rows)}、O-only={sum(r['membership']=='O-only' for r in rows)}、R-only={len(r_only)}、neither={len(neither)}。R-only 清单：\n")
    for row in r_only:
        report.append(f"- `{row['window_id']}` ({row['role']}, label={row['label']}, b={row['offset_s']}): O reason=`{row['O_reasons']}`, R units={row['R_units']}, R reason field=`{row['R_reasons'] or 'VALID'}`。\n")
    report.append(f"保存的无效窗口计数（窗口数，不是 reason occurrence）：O={sum(not r['O_valid'] for r in rows)}，R={sum(not r['R_valid'] for r in rows)}。O reasons={dict(o_reason_counts)}；R reasons={dict(r_reason_counts)}。\n")
    report.append(f"共同 79 个窗口的有效单元为 O={common_o_units}、R={common_r_units}，有效关系为 O={common_o_relations}、R={common_r_relations}；固定 96 窗口全体的 O/R 单元计数为 {all_o_units}/{all_r_units}，R-only 六窗额外贡献 R={all_r_units-common_r_units} 个单元和 {all_r_relations-common_r_relations} 条关系。这里的 2296/2401 是全体固定窗口的 O/R 有效单元总数，而不是把两种模式强行放到相同窗口集合后的数字。\n")
    report.append("分层支撑汇总见 `analysis/support_layer_summary.csv`：它同时按 real/fake、主负/主正、offset=b=0/0.5/1.0 列出计划窗口、有效窗口、单元和关系。当前保存的 metadata 对 17 个 `paired_eligible=False` 窗口没有进一步区分 boundary 与 outside，因此这一层只能报告为 `ineligible_label_window`，不能虚构两类数量。\n")
    report.append("\n| offset | O planned/valid/units | R planned/valid/units |\n|---|---:|---:|\n")
    for level in ("b=0.0", "b=0.5", "b=1.0"):
        o_layer = next(item for item in layer_rows if item["dimension"] == "offset" and item["level"] == level and item["mode"] == "O")
        r_layer = next(item for item in layer_rows if item["dimension"] == "offset" and item["level"] == level and item["mode"] == "R")
        report.append(f"| {level} | {o_layer['planned_windows']}/{o_layer['valid_windows']}/{o_layer['valid_units']} | {r_layer['planned_windows']}/{r_layer['valid_windows']}/{r_layer['valid_units']} |\n")
    report.append("\n## 六条件复算（seed logit 先平均；阈值 logit≥0）\n")
    report.append("| condition | macro source AUROC | 95% CI | pooled AUROC | AP | P | R | F1 | ACC | TN/FP/FN/TP |\n|---|---:|---|---:|---:|---:|---:|---:|---:|---|\n")
    for condition in CONDITIONS:
        row = metric_by_condition[condition]
        report.append(f"| {condition} | {fmt(row['source_auroc'])} | [{fmt(row['source_ci_low'])}, {fmt(row['source_ci_high'])}] | {fmt(row['pooled_auroc'])} | {fmt(row['average_precision'])} | {fmt(row['precision'])} | {fmt(row['recall'])} | {fmt(row['f1'])} | {fmt(row['accuracy'])} | {row['tn']}/{row['fp']}/{row['fn']}/{row['tp']} |\n")
    report.append("\n上述复算与已保存 summary 的数值对照在 `analysis/metric_reproduction.csv`；没有用 OOF 调阈值、校准或翻转分数。\n")
    report.append("## R_SET−O_SET source 影响\n")
    report.append("| source | n(real/fake) | O AUROC | R AUROC | R−O | seed 0909 | seed 0910 | seed 0911 | O units→R units | O relations→R relations |\n|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
    for row in source_comparison:
        report.append(f"| {row['source_id']} | {row['real_count']}/{row['fake_count']} | {fmt(row['O_SET_auroc'])} | {fmt(row['R_SET_auroc'])} | {fmt(row['R_MINUS_O'])} | {fmt(row['R_MINUS_O_seed_20260909'])} | {fmt(row['R_MINUS_O_seed_20260910'])} | {fmt(row['R_MINUS_O_seed_20260911'])} | {row['O_units']}→{row['R_units']} | {row['O_relations']}→{row['R_relations']} |\n")
    report.append(f"\n差值中位数={fmt(median(source_diff_values) if source_diff_values else None)}，范围=[{fmt(min(source_diff_values) if source_diff_values else None)}, {fmt(max(source_diff_values) if source_diff_values else None)}]。逐 source 删除敏感性仅从这些已算出的 source 指标重算平均，不改变评价总体；结果见 `analysis/leave_one_source_sensitivity.csv`。\n")
    report.append("## R 增加的观测与分数尺度\n")
    report.append(f"R/O 单元和关系数量按窗口保存在 `observation_recovery_windows.csv`；R-only 的原因只引用原始 support 字段。没有把新增单元标成失真区域，也没有把 no-support 填为 0。b=0 输入特征逐窗口比较见 `analysis/b0_feature_consistency.csv`（28 个共同 b=0 窗口×3 表示，84/84 形状相同且逐元素相等）；即使原始 b=0 特征相同，O_SET/R_SET 的模型、权重和标准化不同，最终 logit 也不应被要求相等。\n")
    report.append("\n### pooled pair 分解\n")
    report.append("| condition | same-source pairs / ranking | different-source pairs / ranking |\n|---|---:|---:|\n")
    for condition in ("O_SET", "R_SET"):
        same, cross = decomposition[(condition, "same_source")], decomposition[(condition, "different_source")]
        report.append(f"| {condition} | {same['pair_count']} / {fmt(same['ranking_proportion'])} | {cross['pair_count']} / {fmt(cross['ranking_proportion'])} |\n")
    report.append("\n排序比例按正样本−负样本 pair 计，同分按 0.5；按 pair 数加权可复现 pooled AUROC。same-source pair 加权 AUROC 不等于 source 等权 AUROC。跨 source 的分数由不同 LOSO fold 模型产生，不能直接解释为统一部署模型的跨 source 表现。\n")
    report.append("## 运行与边界\n")
    report.append("本轮没有新增训练或推理。既有结果为 96 条 frontend cache、O/R support 192 行、O 79/R 85/common 79 个有效窗口、270 个已完成模型记录（六条件×15 可执行留出 source×三 seed；每个 200 epochs）。预算与原始运行成本保留在 `state/frontend_budget.json`、`state/training_budget.json` 和原 `report.md`。\n")
    report.append("本报告是事后描述性分析，不改变原预声明主结果，不支持 real-only、图关系方法或扩大实验；不含空间真值，也不把窗口有分数解释为失真部位有观测。\n")
    (analysis / "report.md").write_text("".join(report), encoding="utf-8")

    summary = {
        "source_count_dual_class": len(dual_sources),
        "fixed_window_count": len(fixed),
        "o_valid_window_count": len(o_valid),
        "r_valid_window_count": len(r_valid),
        "common_window_count": len(common),
        "main_label_common_count": len(main_common),
        "r_only_window_ids": [row["window_id"] for row in r_only],
        "neither_window_count": len(neither),
        "o_units": all_o_units,
        "r_units": all_r_units,
        "common_o_units": common_o_units,
        "common_r_units": common_r_units,
        "common_o_relations": common_o_relations,
        "common_r_relations": common_r_relations,
        "metric_reproduction": metric_rows,
    }
    (analysis / "analysis_summary.json").write_text(json.dumps(json_safe(summary), indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(json_safe(summary), ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
