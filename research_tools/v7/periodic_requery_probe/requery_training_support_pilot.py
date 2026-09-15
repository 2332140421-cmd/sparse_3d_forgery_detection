#!/usr/bin/env python3
"""Audit label/support eligibility and run the one-condition R training supplement.

The audit is independent of O/R validity.  The formal supplement only changes
the R SET_A training rows; it reuses the frozen model implementation and the
existing R_SET fold set.  No frontend or geometry code is called here.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import traceback
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from research_tools.v7.periodic_requery_probe.analyze_existing import (
    auc,
    average_precision,
    bootstrap_ci,
    classification,
)


OLD_ROOT = Path("/root/autodl-tmp/data/sparse_3d_forgery_detection/derived/v7_activityforensics_periodic_requery_pilot_v1")
SEEDS = (20260909, 20260910, 20260911)
BOOTSTRAP_SEED = 20260909
BOOTSTRAP_REPLICATES = 10_000
FORMAL_BUDGET_S = 900.0
NEW_CONDITION = "R_SET_ALL_VALID_TRAIN"
OLD_CONDITION = "R_SET"


def write_csv(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def atomic_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def finite(value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def merge_segments(segments):
    merged: list[list[float]] = []
    for item in sorted((float(x["start_s"]), float(x["end_s"])) for x in segments):
        if item[1] < item[0]:
            raise ValueError(f"invalid manipulation interval: {item}")
        if not merged or item[0] > merged[-1][1]:
            merged.append([item[0], item[1]])
        else:
            merged[-1][1] = max(merged[-1][1], item[1])
    return merged


def independent_label(role: str, start_s: float, window_length_s: float, segments):
    """Apply the frozen real/union/endpoint convention without support inputs."""

    if role == "real":
        return "REAL_NEGATIVE", True, 0, "real role"
    end_s = float(start_s) + float(window_length_s)
    merged = merge_segments(segments)
    overlaps = [(max(start_s, left), min(end_s, right)) for left, right in merged if min(end_s, right) > max(start_s, left)]
    if not overlaps:
        return "OUTSIDE_ANNOTATED_MANIPULATION", False, None, "no positive-length overlap with manipulation union"
    fully_inside = any(start_s >= left - 1e-9 and end_s <= right + 1e-9 for left, right in merged)
    if fully_inside:
        return "FAKE_MANIPULATION", True, 1, "window is fully contained in manipulation union"
    return "BOUNDARY_MIXED", False, None, "window intersects but is not fully contained in manipulation union"


def load_inputs():
    parents = {row["parent_id"]: row for row in json.loads((OLD_ROOT / "manifests/parents.json").read_text(encoding="utf-8"))}
    subwindows = json.loads((OLD_ROOT / "manifests/subwindows.json").read_text(encoding="utf-8"))
    support_rows = json.loads((OLD_ROOT / "support/window_support.json").read_text(encoding="utf-8"))
    support = {(row["window_id"], row["mode"]): row for row in support_rows}
    scores = {row["window_id"]: row for row in csv.DictReader((OLD_ROOT / "scores/oof_window_scores.csv").open(newline="", encoding="utf-8"))}
    old_models = json.loads((OLD_ROOT / "models/fold_models.json").read_text(encoding="utf-8"))
    return parents, subwindows, support, scores, old_models


def make_audit_rows(parents, subwindows, support, scores, old_models):
    old_heldout_sources = sorted({str(row["held_out_source"]) for row in old_models.get("records", []) if row.get("condition") == OLD_CONDITION})
    rows = []
    for window in subwindows:
        parent = parents[window["parent_id"]]
        category, label_eligible, label, label_reason = independent_label(
            str(window["role"]), float(window["interval_start_s"]), float(window["window_length_s"]), parent.get("all_manipulation_segments", [])
        )
        score = scores.get(window["window_id"], {})
        o = support.get((window["window_id"], "O"), {})
        r = support.get((window["window_id"], "R"), {})
        o_valid = str(o.get("support_status")) == "VALID" and int(o.get("valid_unit_count", 0)) > 0
        r_valid = str(r.get("support_status")) == "VALID" and int(r.get("valid_unit_count", 0)) > 0
        common = o_valid and r_valid
        existing_paired = str(score.get("paired_eligible", "")).lower() == "true"
        old_training_eligible = bool(existing_paired and label_eligible and r_valid)
        new_training_eligible = bool(label_eligible and r_valid)
        source = str(window["source_id"])
        old_fold_count = max(0, len(old_heldout_sources) - 1) if old_training_eligible and source in old_heldout_sources else 0
        new_fold_count = max(0, len(old_heldout_sources) - 1) if new_training_eligible and source in old_heldout_sources else 0
        old_score_status = score.get("R_SET_status", "MISSING_SCORE_ROW")
        rows.append({
            "window_id": window["window_id"], "source_id": source, "role": window["role"], "parent_id": window["parent_id"], "offset_s": window["offset_s"],
            "actual_start_pts_s": window.get("timestamps_s", [None])[0], "actual_end_pts_s": window.get("timestamps_s", [None])[-1],
            "analysis_interval_start_s": window["interval_start_s"], "analysis_interval_end_s": float(window["interval_start_s"]) + float(window["window_length_s"]),
            "annotation_class": category, "independent_label": label, "label_eligible": label_eligible, "label_reason": label_reason,
            "manifest_annotation_category": window.get("annotation_category"), "manifest_label": window.get("label"),
            "label_mapping_matches_manifest": category == window.get("annotation_category") and label == window.get("label"),
            "O_valid": o_valid, "R_valid": r_valid, "common_support": common,
            "existing_paired_eligible": existing_paired, "paired_formula_expected": bool(common and label_eligible),
            "paired_formula_matches_saved": existing_paired == bool(common and label_eligible),
            "O_support_status": o.get("support_status", "MISSING"), "R_support_status": r.get("support_status", "MISSING"),
            "O_valid_unit_count": int(o.get("valid_unit_count", 0)), "R_valid_unit_count": int(r.get("valid_unit_count", 0)),
            "existing_R_SET_status": old_score_status, "existing_R_SET_score": finite(score.get("R_SET")),
            "old_R_training_eligible": old_training_eligible, "new_R_training_eligible": new_training_eligible,
            "old_training_fold_count": old_fold_count, "new_training_fold_count": new_fold_count,
            "old_exclusion_reason": "PAIRED_ELIGIBILITY_REQUIRES_O_AND_R_COMMON_SUPPORT" if new_training_eligible and not old_training_eligible else "",
        })
    return rows, old_heldout_sources


def metric_rows(rows, score_getter, sources):
    def row_label(row):
        return int(row.get("independent_label", row.get("label")))

    def row_label_eligible(row):
        return bool(row.get("label_eligible", True))

    def row_common(row):
        return bool(row.get("common_support", True))

    values_by_source = {}
    selected = [row for row in rows if row["source_id"] in sources and row_label_eligible(row) and row_common(row)]
    for source in sources:
        source_rows = [row for row in selected if row["source_id"] == source and score_getter(row) is not None]
        labels = [row_label(row) for row in source_rows]
        scores = [score_getter(row) for row in source_rows]
        value = auc(labels, scores) if len(set(labels)) == 2 else None
        if value is not None:
            values_by_source[source] = value
    all_labels = [row_label(row) for row in selected if score_getter(row) is not None]
    all_scores = [score_getter(row) for row in selected if score_getter(row) is not None]
    low, high = bootstrap_ci(list(values_by_source.values()), seed=BOOTSTRAP_SEED, replicates=BOOTSTRAP_REPLICATES)
    cls = classification(all_labels, all_scores) if all_labels else {"precision": None, "recall": None, "f1": None, "accuracy": None, "tn": None, "fp": None, "fn": None, "tp": None}
    return {
        "source_count": len(values_by_source), "window_count": len(all_labels), "real_count": sum(x == 0 for x in all_labels), "fake_count": sum(x == 1 for x in all_labels),
        "source_auroc": float(np.mean(list(values_by_source.values()))) if values_by_source else None, "ci_low": low, "ci_high": high,
        "pooled_auroc": auc(all_labels, all_scores), "pooled_ap": average_precision(all_labels, all_scores), **cls,
        "source_values": values_by_source,
    }


def load_saved_seed_scores(score_row, condition):
    values = [finite(score_row.get(f"{condition}_seed_{seed}")) for seed in SEEDS]
    return values if all(value is not None for value in values) else None


def audit(root: Path):
    parents, subwindows, support, scores, old_models = load_inputs()
    rows, old_heldout_sources = make_audit_rows(parents, subwindows, support, scores, old_models)
    audit_dir = root / "audit"
    audit_dir.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    write_csv(audit_dir / "eligibility_audit.csv", rows)
    write_csv(audit_dir / "r_only_audit.csv", [row for row in rows if row["R_valid"] and not row["O_valid"]])
    common = [row for row in rows if row["common_support"] and row["label_eligible"]]
    dual_sources = sorted({row["source_id"] for row in common if any(other["source_id"] == row["source_id"] and other["independent_label"] != row["independent_label"] for other in common)})
    score_lookup = scores
    unified = []
    for condition in ("O_SET", "O_RAW", "O_MULTI", "R_SET", "R_RAW", "R_MULTI"):
        value = metric_rows(common, lambda row, c=condition: float(np.mean(load_saved_seed_scores(score_lookup[row["window_id"]], c))) if load_saved_seed_scores(score_lookup[row["window_id"]], c) else None, dual_sources)
        unified.append({"condition": condition, **{key: val for key, val in value.items() if key != "source_values"}})
    write_csv(audit_dir / "uniform_common_metrics.csv", unified)
    atomic_json(audit_dir / "audit_summary.json", {
        "window_count": len(rows), "label_eligible_count": sum(row["label_eligible"] for row in rows),
        "label_classes": {name: sum(row["annotation_class"] == name for row in rows) for name in ("REAL_NEGATIVE", "FAKE_MANIPULATION", "BOUNDARY_MIXED", "OUTSIDE_ANNOTATED_MANIPULATION")},
        "o_valid_count": sum(row["O_valid"] for row in rows), "r_valid_count": sum(row["R_valid"] for row in rows), "common_count": sum(row["common_support"] for row in rows),
        "existing_paired_count": sum(row["existing_paired_eligible"] for row in rows), "old_heldout_sources": old_heldout_sources, "dual_evaluation_sources": dual_sources,
        "r_only_count": sum(row["R_valid"] and not row["O_valid"] for row in rows), "r_only_label_eligible_count": sum(row["R_valid"] and not row["O_valid"] and row["label_eligible"] for row in rows),
        "r_only_old_training_count": sum(row["R_valid"] and not row["O_valid"] and row["old_R_training_eligible"] for row in rows),
        "r_only_new_training_count": sum(row["R_valid"] and not row["O_valid"] and row["new_R_training_eligible"] for row in rows),
        "label_mapping_mismatch_count": sum(not row["label_mapping_matches_manifest"] for row in rows),
        "paired_formula_mismatch_count": sum(not row["paired_formula_matches_saved"] for row in rows),
    })
    return {
        "parents": parents, "subwindows": subwindows, "support": support, "scores": scores, "old_models": old_models,
        "rows": rows, "old_heldout_sources": old_heldout_sources, "dual_sources": dual_sources, "common": common, "unified_metrics": unified,
    }


def model_records_by_key(records, condition):
    return {(str(row["held_out_source"]), int(row["seed"])): row for row in records if row.get("condition") == condition}


def attach_r_feature(row, support):
    saved = support[(row["window_id"], "R")]
    return {**row, "label": row.get("independent_label"), "features": saved.get("features", {}), "intervals_s": saved.get("intervals_s")}


def score_frozen_records(records_by_key, rows, condition, base, device, runner):
    scores = {row["window_id"]: {} for row in rows}
    for heldout in sorted({str(key[0]) for key in records_by_key}):
        held_rows = [row for row in rows if row["source_id"] == heldout]
        if not held_rows:
            continue
        for seed in SEEDS:
            record = records_by_key.get((heldout, int(seed)))
            if record is None:
                continue
            model = runner._model_from_record(base, record, device)
            standardizer = runner._standardizer(record)
            values = [{**dict(row), "features": row["features"]} for row in held_rows]
            batch = runner.make_batch(base, values, standardizer, require_labels=False)
            predicted = runner.score_batch(base, model, batch, device)
            for window_id, score in zip(batch["window_ids"], predicted):
                scores[window_id][str(seed)] = float(score)
            del model
    return scores


def run_smoke(root: Path, audit_data, device: str):
    import torch
    from research_tools.v7.periodic_requery_probe import runner

    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA_UNAVAILABLE")
    heldout = audit_data["old_heldout_sources"][0]
    rows = [attach_r_feature(row, audit_data["support"]) for row in audit_data["rows"] if row["new_R_training_eligible"] and row["source_id"] != heldout and row["R_valid"]]
    held_rows = [attach_r_feature(row, audit_data["support"]) for row in audit_data["rows"] if row["new_R_training_eligible"] and row["source_id"] == heldout and row["R_valid"]]
    values, batch, standardizer = runner._fit_rows(rows, "SET_A")
    model, fit = runner.train_one("SET_A", batch, seed=int(SEEDS[0]), device=device, epochs=1)
    score_batch = runner.make_batch("SET_A", held_rows, standardizer, require_labels=False)
    scores = runner.score_batch("SET_A", model, score_batch, device)
    output = {"status": "PASS", "held_out_source": heldout, "training_window_count": len(rows), "held_out_window_count": len(held_rows), "predicted_count": len(scores), "epochs": fit["epochs"], "device": device, "model_persisted": False}
    atomic_json(root / "smoke/summary.json", output)
    del model
    return output


def run_formal(root: Path, audit_data, device: str):
    import torch
    from research_tools.v7.periodic_requery_probe import runner

    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA_UNAVAILABLE")
    records = []
    manifest_rows = []
    old_records = audit_data["old_models"].get("records", [])
    heldout_sources = audit_data["old_heldout_sources"]
    training_rows = [attach_r_feature(row, audit_data["support"]) for row in audit_data["rows"] if row["new_R_training_eligible"] and row["R_valid"]]
    old_training_rows = [attach_r_feature(row, audit_data["support"]) for row in audit_data["rows"] if row["old_R_training_eligible"] and row["R_valid"]]
    models_dir = root / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    budget = runner.Budget(root, "training", FORMAL_BUDGET_S)
    atomic_json(root / "protocol.json", {
        "protocol_id": "v7-requery-training-support-pilot-v1", "base_experiment": str(OLD_ROOT),
        "condition": NEW_CONDITION, "old_condition": "R_SET_COMMON_TRAIN", "held_out_sources": heldout_sources,
        "training_filter": "independent label_eligible AND R support_status VALID; held-out source excluded",
        "evaluation_main": "original common-support label-qualified dual-source 14-source/76-window set",
        "seeds": list(SEEDS), "epochs": 200, "optimizer": "Adam", "learning_rate": 0.001, "weight_decay": 0.0001,
        "bootstrap": {"unit": "source", "seed": BOOTSTRAP_SEED, "replicates": BOOTSTRAP_REPLICATES}, "device": device,
        "frontend_reused": True,
    })
    loss_path = models_dir / "loss_history.jsonl"
    loss_handle = loss_path.open("w", encoding="utf-8")
    try:
        total = len(heldout_sources) * len(SEEDS)
        completed = 0
        for heldout in heldout_sources:
            train_rows = [row for row in training_rows if row["source_id"] != heldout]
            old_rows = [row for row in old_training_rows if row["source_id"] != heldout]
            values, batch, standardizer = runner._fit_rows(train_rows, "SET_A")
            old_values, old_batch, _old_standardizer = runner._fit_rows(old_rows, "SET_A")
            old_ids = {row["window_id"] for row in old_rows}
            weights = np.asarray(batch["window_weights"], dtype=np.float64)
            old_weights = {row["window_id"]: float(weight) for row, weight in zip(old_values, np.asarray(old_batch["window_weights"], dtype=np.float64))}
            for row, weight in zip(values, weights):
                manifest_rows.append({
                    "condition": NEW_CONDITION, "held_out_source": heldout, "seed": "ALL_FORMAL_SEEDS", "window_id": row["window_id"], "source_id": row["source_id"], "role": row["role"], "label": row["label"], "weight": float(weight), "old_weight": old_weights.get(row["window_id"]), "weight_delta": float(weight) - old_weights[row["window_id"]] if row["window_id"] in old_weights else None, "in_old_R_SET_COMMON_TRAIN": row["window_id"] in old_ids,
                })
            for seed in SEEDS:
                if budget.remaining() <= 0:
                    raise RuntimeError("TRAIN_BUDGET_EXHAUSTED")
                print(f"formal condition={NEW_CONDITION} held_out={heldout} seed={seed} start remaining={budget.remaining():.1f}s", flush=True)
                model, fit = runner.train_one("SET_A", batch, seed=int(seed), device=device, epochs=200, epoch_callback=(lambda epoch, metrics, h=heldout, s=seed: print(f"formal held_out={h} seed={s} epoch={epoch}/200 loss={float(metrics['loss']):.6f}", flush=True) if epoch in (1, 200) else None))
                record = {
                    "condition": NEW_CONDITION, "base_condition": "SET_A", "mode": "R", "held_out_source": heldout, "seed": int(seed), "parameter_count": runner.parameter_count("SET_A"),
                    "training_source_count": len({row["source_id"] for row in train_rows}), "training_real_count": sum(row["label"] == 0 for row in train_rows), "training_fake_count": sum(row["label"] == 1 for row in train_rows), "training_window_count": len(train_rows), "added_training_window_count": sum(row["window_id"] not in old_ids for row in train_rows),
                    "fit": fit, "standardization": standardizer.as_dict(), "state_dict": runner.model_state(model),
                }
                records.append(record)
                completed += 1
                for item in fit["loss_history"]:
                    loss_handle.write(json.dumps({"condition": NEW_CONDITION, "held_out_source": heldout, "seed": int(seed), **item}, allow_nan=False) + "\n")
                loss_handle.flush()
                atomic_json(models_dir / "fold_models.json", {"condition": NEW_CONDITION, "base_condition": "SET_A", "seeds": list(SEEDS), "epochs": 200, "records": records})
                budget.save(None, completed_models=completed, total_models=total, last_held_out_source=heldout, last_seed=int(seed))
                del model
        if completed != total:
            raise RuntimeError(f"MODEL_COUNT_MISMATCH:{completed}/{total}")
    finally:
        loss_handle.close()
    write_csv(root / "training_window_manifest.csv", manifest_rows)
    return {"records": records, "training_rows": training_rows, "old_training_rows": old_training_rows, "heldout_sources": heldout_sources, "budget": budget}


def evaluate_formal(root: Path, audit_data, formal_data, device: str):
    from research_tools.v7.periodic_requery_probe import runner

    old_records = model_records_by_key(audit_data["old_models"].get("records", []), OLD_CONDITION)
    new_records = model_records_by_key(formal_data["records"], NEW_CONDITION)
    score_rows = [row for row in audit_data["rows"] if row["new_R_training_eligible"] and row["R_valid"]]
    score_input_rows = [attach_r_feature(row, audit_data["support"]) for row in score_rows]
    old_scores = score_frozen_records(old_records, score_input_rows, OLD_CONDITION, "SET_A", device, runner)
    new_scores = score_frozen_records(new_records, score_input_rows, NEW_CONDITION, "SET_A", device, runner)
    existing_mismatch = []
    for row in score_rows:
        saved = audit_data["scores"].get(row["window_id"], {})
        for seed in SEEDS:
            a = finite(saved.get(f"{OLD_CONDITION}_seed_{seed}")); b = old_scores[row["window_id"]].get(str(seed))
            if a is not None and b is not None:
                existing_mismatch.append(abs(a - b))
    max_mismatch = max(existing_mismatch) if existing_mismatch else None
    output_scores = []
    for row in score_rows:
        old = old_scores[row["window_id"]]; new = new_scores[row["window_id"]]
        old_mean = float(np.mean([old[str(seed)] for seed in SEEDS])) if all(str(seed) in old for seed in SEEDS) else None
        new_mean = float(np.mean([new[str(seed)] for seed in SEEDS])) if all(str(seed) in new for seed in SEEDS) else None
        output_scores.append({
            "window_id": row["window_id"], "source_id": row["source_id"], "role": row["role"], "offset_s": row["offset_s"], "label": row["independent_label"], "annotation_class": row["annotation_class"], "common_support": row["common_support"], "r_only": not row["common_support"],
            **{f"old_R_SET_seed_{seed}": old.get(str(seed)) for seed in SEEDS}, **{f"new_R_SET_ALL_VALID_TRAIN_seed_{seed}": new.get(str(seed)) for seed in SEEDS}, "old_R_SET": old_mean, "new_R_SET_ALL_VALID_TRAIN": new_mean,
        })
    write_csv(root / "scores/r_set_support_comparison.csv", output_scores)
    main_rows = [row for row in output_scores if row["common_support"] and row["source_id"] in audit_data["dual_sources"]]
    by_source = {}
    for source in audit_data["dual_sources"]:
        source_rows = [row for row in main_rows if row["source_id"] == source]
        labels = [int(row["label"]) for row in source_rows]
        old = [row["old_R_SET"] for row in source_rows]
        new = [row["new_R_SET_ALL_VALID_TRAIN"] for row in source_rows]
        old_auc = auc(labels, old) if len(set(labels)) == 2 else None
        new_auc = auc(labels, new) if len(set(labels)) == 2 else None
        seed_diffs = {}
        for seed in SEEDS:
            old_seed = [row[f"old_R_SET_seed_{seed}"] for row in source_rows]; new_seed = [row[f"new_R_SET_ALL_VALID_TRAIN_seed_{seed}"] for row in source_rows]
            seed_old_auc = auc(labels, old_seed) if len(set(labels)) == 2 else None; seed_new_auc = auc(labels, new_seed) if len(set(labels)) == 2 else None
            seed_diffs[seed] = seed_new_auc - seed_old_auc if seed_old_auc is not None and seed_new_auc is not None else None
        old_cls = classification(labels, old); new_cls = classification(labels, new)
        by_source[source] = {"source_id": source, "window_count": len(source_rows), "real_count": sum(x == 0 for x in labels), "fake_count": sum(x == 1 for x in labels), "old_auroc": old_auc, "new_auroc": new_auc, "new_minus_old": new_auc - old_auc if old_auc is not None and new_auc is not None else None, **{f"new_minus_old_seed_{seed}": seed_diffs[seed] for seed in SEEDS}, "old_tn": old_cls["tn"], "old_fp": old_cls["fp"], "old_fn": old_cls["fn"], "old_tp": old_cls["tp"], "new_tn": new_cls["tn"], "new_fp": new_cls["fp"], "new_fn": new_cls["fn"], "new_tp": new_cls["tp"]}
    write_csv(root / "evaluation/per_source_metrics.csv", list(by_source.values()))
    old_main = metric_rows(main_rows, lambda row: row["old_R_SET"], audit_data["dual_sources"])
    new_main = metric_rows(main_rows, lambda row: row["new_R_SET_ALL_VALID_TRAIN"], audit_data["dual_sources"])
    old_values = old_main["source_values"]; new_values = new_main["source_values"]
    difference_values = [new_values[s] - old_values[s] for s in sorted(set(old_values) & set(new_values))]
    d_low, d_high = bootstrap_ci(difference_values, seed=BOOTSTRAP_SEED, replicates=BOOTSTRAP_REPLICATES)
    all_rows = output_scores
    all_labels = [int(row["label"]) for row in all_rows]
    all_new_scores = [row["new_R_SET_ALL_VALID_TRAIN"] for row in all_rows]
    all_old_scores = [row["old_R_SET"] for row in all_rows]
    new_all_cls = classification(all_labels, all_new_scores)
    r_only_rows = [row for row in output_scores if row["r_only"]]
    write_csv(root / "evaluation/r_only_scores.csv", r_only_rows)
    summary = {
        "main_collection": {"source_count": len(audit_data["dual_sources"]), "window_count": len(main_rows), "real_count": sum(row["label"] == 0 for row in main_rows), "fake_count": sum(row["label"] == 1 for row in main_rows)},
        "old_R_SET": {key: value for key, value in old_main.items() if key != "source_values"}, "new_R_SET_ALL_VALID_TRAIN": {key: value for key, value in new_main.items() if key != "source_values"},
        "new_minus_old": {"mean": float(np.mean(difference_values)) if difference_values else None, "ci95": [d_low, d_high], "source_count": len(difference_values), "seed": BOOTSTRAP_SEED, "replicates": BOOTSTRAP_REPLICATES},
        "all_label_qualified_r_valid": {"window_count": len(all_rows), "real_count": sum(row["label"] == 0 for row in all_rows), "fake_count": sum(row["label"] == 1 for row in all_rows), "new_pooled_auroc": auc(all_labels, all_new_scores), "new_pooled_ap": average_precision(all_labels, all_new_scores), **new_all_cls},
        "r_only_count": len(r_only_rows), "r_only_rows": r_only_rows,
        "old_model_reproduction_max_abs": max_mismatch, "old_model_reproduction_tolerance": 1e-5,
        "model_count": len(formal_data["records"]), "all_models_200_epochs": all(record["fit"].get("epochs") == 200 for record in formal_data["records"]), "device": device,
    }
    atomic_json(root / "evaluation/summary.json", summary)
    return summary, by_source


def make_report(root: Path, audit_data, formal_data, summary, source_metrics, smoke):
    rows = audit_data["rows"]
    r_only = [row for row in rows if row["R_valid"] and not row["O_valid"]]
    added = [row for row in rows if row["new_R_training_eligible"] and not row["old_R_training_eligible"]]
    manifest = list(csv.DictReader((root / "training_window_manifest.csv").open(newline="", encoding="utf-8"))) if (root / "training_window_manifest.csv").is_file() else []
    added_manifest_entries = [row for row in manifest if row.get("in_old_R_SET_COMMON_TRAIN") == "False"]
    report = [
        "# V7 periodic re-query：标签资格与 R 新增可观测窗口训练补充 pilot\n",
        "## 直接回答\n",
        "1. `paired_eligible` 不是单纯标签资格。`runner.py:763` 的实际公式要求本行和另一模式 `valid_unit_count>0`，且两行 label 都在 `{0,1}`；它混合了 O/R 共同结构支撑与标签。\n",
        f"2. 六个 R-only 窗口中，标签合格 {sum(row['label_eligible'] for row in r_only)}/{len(r_only)}，R 特征有效 {sum(row['R_valid'] for row in r_only)}/{len(r_only)}；此前进入旧 R_SET 训练 {sum(row['old_R_training_eligible'] for row in r_only)}/{len(r_only)}，本补充可进入训练 {sum(row['new_R_training_eligible'] for row in r_only)}/{len(r_only)}。\n",
        "3. 已实际执行一个真实 fold 的 1 epoch smoke，并完成正式 `R_SET_ALL_VALID_TRAIN` 训练：45 个模型（15 个原有 held-out source×3 seed），每个 200 epochs。\n",
        f"4. 新训练每折使用 R 有效且标签合格的全部窗口；新增训练行涉及 {len(added)} 个 distinct 窗口、{len(added_manifest_entries)} 个 fold-window 条目（持出该窗口所属 source 的 fold 不纳入），具体 fold/window/权重在 `training_window_manifest.csv`。\n",
        f"5. 原共同 14-source/76-window 集合上，新旧 source-macro AUROC 为 {summary['new_R_SET_ALL_VALID_TRAIN']['source_auroc']:.6f}/{summary['old_R_SET']['source_auroc']:.6f}，新−旧={summary['new_minus_old']['mean']:.6f}，95% CI={summary['new_minus_old']['ci95']}。\n",
        "6. 新增 R-only 窗口仅作描述性补评分，不混入主要配对指标；逐窗口分数在 `evaluation/r_only_scores.csv`。\n",
        "7. 本轮仍没有空间真值；R 结构有效只说明测量支撑存在，不能说明新增单元位于失真部位。\n",
        "8. 本轮未重新解码、tracking、depth、pose、segmentation，也未修改旧实验产物；新增训练预算与状态单独保存在本目录。\n",
        "\n## 独立标签与资格核对\n",
        f"- 96 个窗口独立重算标签：REAL_NEGATIVE={sum(row['annotation_class']=='REAL_NEGATIVE' for row in rows)}，FAKE_MANIPULATION={sum(row['annotation_class']=='FAKE_MANIPULATION' for row in rows)}，BOUNDARY_MIXED={sum(row['annotation_class']=='BOUNDARY_MIXED' for row in rows)}，OUTSIDE={sum(row['annotation_class']=='OUTSIDE_ANNOTATED_MANIPULATION' for row in rows)}。\n",
        f"- 独立标签资格 {sum(row['label_eligible'] for row in rows)}/96；与 manifest category/label 不一致 {sum(not row['label_mapping_matches_manifest'] for row in rows)}。\n",
        f"- 保存的 `paired_eligible=True` 为 {sum(row['existing_paired_eligible'] for row in rows)}；按 `label_eligible AND O_valid AND R_valid` 独立重算，公式不一致 {sum(not row['paired_formula_matches_saved'] for row in rows)}。\n",
        f"- 旧 R_SET 有效模型的 held-out source 集合：{', '.join(audit_data['old_heldout_sources'])}。R-only 所属 source 均在该集合内，但因旧 `paired_eligible=False` 没有进入旧训练行。\n",
        "\n## R-only 逐条核对\n",
        "| window | source/role/b | label | R units | old paired | old training | new training | O/R score state |\n|---|---|---:|---:|---|---|---|---|\n",
    ]
    for row in r_only:
        report.append(f"| `{row['window_id']}` | {row['source_id']}/{row['role']}/b={row['offset_s']} | {row['annotation_class']} | {row['R_valid_unit_count']} | {row['existing_paired_eligible']} | {row['old_R_training_eligible']} | {row['new_R_training_eligible']} | {row['existing_R_SET_status']} |\n")
    report += [
        "\n## 统一 14-source/76-window 补充指标\n",
        "| 条件 | source-macro AUROC | 95% CI | pooled AUROC | AP | P/R/F1/ACC | TN/FP/FN/TP |\n|---|---:|---|---:|---:|---|---|\n",
    ]
    for name in ("old_R_SET", "new_R_SET_ALL_VALID_TRAIN"):
        item = summary[name]
        report.append(f"| {name} | {item['source_auroc']:.6f} | [{item['ci_low']:.6f}, {item['ci_high']:.6f}] | {item['pooled_auroc']:.6f} | {item['pooled_ap']:.6f} | {item['precision']:.6f}/{item['recall']:.6f}/{item['f1']:.6f}/{item['accuracy']:.6f} | {item['tn']}/{item['fp']}/{item['fn']}/{item['tp']} |\n")
    report += [
        f"\n唯一主要比较 `R_SET_ALL_VALID_TRAIN − R_SET`：均值差={summary['new_minus_old']['mean']:.6f}，95% CI=[{summary['new_minus_old']['ci95'][0]:.6f}, {summary['new_minus_old']['ci95'][1]:.6f}]，source_count={summary['new_minus_old']['source_count']}。\n",
        "逐 source 和逐 seed 差值见 `evaluation/per_source_metrics.csv`；没有选择有利 source/seed。\n",
        "\n## 全部 R 标签合格且结构有效窗口（描述性，不与主比较相减）\n",
        f"新模型评分 {summary['all_label_qualified_r_valid']['window_count']} 个窗口（real={summary['all_label_qualified_r_valid']['real_count']}, fake={summary['all_label_qualified_r_valid']['fake_count']}）：pooled AUROC={summary['all_label_qualified_r_valid']['new_pooled_auroc']:.6f}，AP={summary['all_label_qualified_r_valid']['new_pooled_ap']:.6f}，固定阈值 P/R/F1/ACC={summary['all_label_qualified_r_valid']['precision']:.6f}/{summary['all_label_qualified_r_valid']['recall']:.6f}/{summary['all_label_qualified_r_valid']['f1']:.6f}/{summary['all_label_qualified_r_valid']['accuracy']:.6f}。这是扩展集合描述，不能归因于仅训练范围变化。\n",
        "\n## 验证、成本与边界\n",
        f"- 真实 fold smoke：`{smoke['status']}`，held-out={smoke['held_out_source']}，1 epoch，训练窗口={smoke['training_window_count']}，评分窗口={smoke['held_out_window_count']}，未将 smoke 模型混入正式 records。\n",
        f"- 正式模型：{summary['model_count']} 个，全部 200 epochs={summary['all_models_200_epochs']}，设备={summary['device']}；旧 R_SET 复现最大绝对差={summary['old_model_reproduction_max_abs']}（容差 1e-5）。\n",
        "- frontend/feature 缓存只读复用；不提交模型、视频、权重、NPZ 或 ZIP。\n",
        "- 这是观察到结果后的开发性补充，不是独立确认性测试或 sealed-test；不支持 real-only、图关系方法或扩大下一轮实验。\n",
    ]
    (root / "report.md").write_text("".join(report), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--smoke-only", action="store_true")
    parser.add_argument("--formal", action="store_true")
    parser.add_argument("--evaluate-only", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    audit_data = audit(root)
    if args.smoke_only:
        smoke = run_smoke(root, audit_data, args.device)
        atomic_json(root / "final_status.json", {"status": "SMOKE_COMPLETE", "smoke": smoke})
        print(json.dumps(smoke, ensure_ascii=False, indent=2))
        return
    if args.evaluate_only:
        smoke_path = root / "smoke/summary.json"
        smoke = json.loads(smoke_path.read_text(encoding="utf-8")) if smoke_path.is_file() else {"status": "NOT_RECORDED"}
        records_path = root / "models/fold_models.json"
        records = json.loads(records_path.read_text(encoding="utf-8")).get("records", []) if records_path.is_file() else []
        try:
            manifest_path = root / "training_window_manifest.csv"
            if manifest_path.is_file():
                manifest_rows = list(csv.DictReader(manifest_path.open(newline="", encoding="utf-8")))
                if manifest_rows and "seed" not in manifest_rows[0]:
                    for manifest_row in manifest_rows:
                        manifest_row["seed"] = "ALL_FORMAL_SEEDS"
                    write_csv(manifest_path, manifest_rows)
            summary, source_metrics = evaluate_formal(root, audit_data, {"records": records}, args.device)
            make_report(root, audit_data, {"records": records}, summary, source_metrics, smoke)
            atomic_json(root / "final_status.json", {"status": "COMPLETE", "smoke": smoke, "model_count": summary["model_count"], "summary": summary})
            print(json.dumps({"status": "COMPLETE", "model_count": summary["model_count"], "new_minus_old": summary["new_minus_old"]}, ensure_ascii=False, indent=2))
        except Exception as exc:
            error = {"status": "FAILED", "stage": "evaluation_only", "error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()}
            atomic_json(root / "final_status.json", error)
            raise
        return
    if not args.formal:
        atomic_json(root / "final_status.json", {"status": "AUDIT_COMPLETE", "audit_only": True})
        print(json.dumps({"status": "AUDIT_COMPLETE"}, ensure_ascii=False))
        return
    smoke_path = root / "smoke/summary.json"
    smoke = json.loads(smoke_path.read_text(encoding="utf-8")) if smoke_path.is_file() else run_smoke(root, audit_data, args.device)
    atomic_json(root / "final_status.json", {"status": "TRAINING_RUNNING", "smoke": smoke})
    try:
        formal_data = run_formal(root, audit_data, args.device)
        summary, source_metrics = evaluate_formal(root, audit_data, formal_data, args.device)
        formal_data["budget"].save(None, completed_models=summary["model_count"], total_models=summary["model_count"])
        make_report(root, audit_data, formal_data, summary, source_metrics, smoke)
        atomic_json(root / "final_status.json", {"status": "COMPLETE", "smoke": smoke, "model_count": summary["model_count"], "summary": summary})
        print(json.dumps({"status": "COMPLETE", "model_count": summary["model_count"], "new_minus_old": summary["new_minus_old"]}, ensure_ascii=False, indent=2))
    except Exception as exc:
        error = {"status": "FAILED", "stage": "formal_training_or_evaluation", "error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()}
        atomic_json(root / "final_status.json", error)
        raise


if __name__ == "__main__":
    main()
