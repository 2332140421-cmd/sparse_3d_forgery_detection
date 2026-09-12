"""Build the frozen boundary-pilot review page and forward-only train/held-out audit.

The immediate caller is the V7 boundary-pooling pilot artifact.  This module
reads saved H/B features, ParticleSequence arrays, and serialized fold models;
it never calls an optimizer or a frontend provider.  One explicit script is
smaller and easier to audit here than a review web framework or a new task
runner.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from sparse3d_forgery.particle_sequence import load_particle_sequence
from sparse3d_forgery.video_input import VideoSource, decode_video
from research_tools.v7.boundary_pooling_probe.model import PoolingWindowMLP, score_pooling_model
from research_tools.v7.boundary_pooling_probe.pipeline import _feature_path
from research_tools.v7.boundary_pooling_probe.summarize_results import (
    SEEDS,
    _average_precision,
    _roc_auc,
)
from research_tools.v7.local_structural_temporal_probe.model import (
    WeightedStandardizer,
    build_batch,
)
from research_tools.v7.local_structural_temporal_probe.representation import (
    compute_local_derivatives,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
DATA_ROOT = Path("/root/autodl-tmp/data/sparse_3d_forgery_detection")
DEFAULT_ROOT = DATA_ROOT / "derived/v7_activityforensics_boundary_pooling_pilot_v1"
FORWARD_CONDITIONS = ("H_MEAN_A", "B_MEAN_A", "B_MEAN_C")
DEFAULT_SOURCE_COUNT = 4


def _safe(value: str) -> str:
    return "".join(char if char.isalnum() or char in "._-" else "_" for char in value)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
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
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _read_rows(root: Path) -> list[dict[str, Any]]:
    return json.loads((root / "manifests/input_manifest.json").read_text(encoding="utf-8"))["rows"]


def _load_feature(root: Path, window_id: str) -> dict[str, Any]:
    return json.loads(_feature_path(root, window_id).read_text(encoding="utf-8"))


def _examples(root: Path, rows: Sequence[Mapping[str, Any]], organization: str) -> list[dict[str, Any]]:
    common_ids = set(json.loads((root / "manifests/common_windows.json").read_text(encoding="utf-8"))["window_ids"])
    output: list[dict[str, Any]] = []
    for row in rows:
        window_id = str(row["window_id"])
        if window_id not in common_ids:
            continue
        feature = _load_feature(root, window_id)
        support = feature[f"{organization.lower()}_support"]
        if not support.get("triplets"):
            continue
        output.append({
            "window_id": window_id,
            "pair_id": str(row["pair_id"]),
            "source_id": str(row["source_id"]),
            "role": str(row["role"]),
            "kind": str(row["kind"]),
            "label": row.get("label"),
            "anchor_fraction": float(row["anchor_fraction"]),
            "triplets": support["triplets"],
        })
    return output


def _main_examples(examples: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [row for row in examples if str(row.get("kind")) == "MANIP" and str(row.get("role")) in {"real", "fake"}]


def _standardizer(record: Mapping[str, Any]) -> WeightedStandardizer:
    value = record["standardization"]
    return WeightedStandardizer(
        mean=np.asarray(value["mean"], dtype=np.float64),
        scale=np.asarray(value["scale"], dtype=np.float64),
        zero_variance_dimensions=tuple(int(x) for x in value.get("zero_variance_dimensions", [])),
    )


def _restore_model(record: Mapping[str, Any]) -> tuple[PoolingWindowMLP, WeightedStandardizer]:
    model_data = record["model"]
    model = PoolingWindowMLP(str(record["pooling"]).lower()).to("cpu")
    state = {name: torch.as_tensor(values, dtype=torch.float32) for name, values in model_data["state_dict"].items()}
    model.load_state_dict(state)
    model.eval()
    return model, _standardizer(record)


def _classification_metrics(labels: Sequence[int], scores: Sequence[float]) -> dict[str, Any]:
    y = np.asarray(labels, dtype=np.int64)
    s = np.asarray(scores, dtype=np.float64)
    if y.size != s.size:
        raise ValueError("labels and scores must have equal length")
    valid = np.isfinite(s)
    y, s = y[valid], s[valid]
    predicted = s >= 0.0
    tp = int(np.sum((y == 1) & predicted)); fp = int(np.sum((y == 0) & predicted))
    tn = int(np.sum((y == 0) & ~predicted)); fn = int(np.sum((y == 1) & ~predicted))
    precision = float(tp / (tp + fp)) if tp + fp else None
    recall = float(tp / (tp + fn)) if tp + fn else None
    f1 = float(2 * precision * recall / (precision + recall)) if precision is not None and recall is not None and precision + recall else None
    return {
        "n_total": int(y.size),
        "n_fake": int(np.sum(y == 1)),
        "n_real": int(np.sum(y == 0)),
        "roc_auc": _roc_auc(y, s),
        "average_precision": _average_precision(y, s),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": float((tp + tn) / y.size) if y.size else None,
        "tn": tn, "fp": fp, "fn": fn, "tp": tp,
        "threshold": "logit >= 0",
    }


_METRIC_FIELDS = ("roc_auc", "average_precision", "precision", "recall", "f1", "accuracy", "tn", "fp", "fn", "tp")


def _mean_metric_dict(metrics: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Average defined metrics without converting undefined values to zero."""

    output: dict[str, Any] = {}
    for key in _METRIC_FIELDS:
        values = [float(item[key]) for item in metrics if item.get(key) is not None]
        output[key] = float(np.mean(values)) if values else None
    return output


def _aggregate_forward_scores(score_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Write pooled and source-macro summaries from saved forward scores.

    A pooled row keeps every fold/seed prediction (so its duplicate structure is
    explicit).  A source-macro row first averages scores for each source/window
    over the available saved fold/seed predictions, then computes metrics per
    source and averages those metrics equally.
    """

    output: list[dict[str, Any]] = []
    for condition in FORWARD_CONDITIONS:
        for split in ("train", "heldout"):
            selected = [x for x in score_rows if x["condition"] == condition and x["split"] == split]
            labels = [int(x["label"]) for x in selected]
            scores = [float(x["logit"]) for x in selected]
            pooled = _classification_metrics(labels, scores)
            output.append({
                "condition": condition, "split": split, "aggregation": "pooled",
                "source_count": len({str(x["source_id"]) for x in selected}),
                "fold_seed_rows": len({(str(x["held_out_source"]), int(x["seed"])) for x in selected}),
                "score_rows": len(selected), **pooled,
                "score_note": "all saved fold/seed predictions; training rows repeat a window across folds",
            })
            by_source_window: dict[tuple[str, str], list[float]] = defaultdict(list)
            labels_by_source_window: dict[tuple[str, str], int] = {}
            for item in selected:
                key = (str(item["source_id"]), str(item["window_id"]))
                by_source_window[key].append(float(item["logit"]))
                labels_by_source_window[key] = int(item["label"])
            per_source: dict[str, dict[str, Any]] = {}
            for (source_id, _window_id), values in by_source_window.items():
                per_source.setdefault(source_id, {"labels": [], "scores": []})
                per_source[source_id]["labels"].append(labels_by_source_window[(source_id, _window_id)])
                per_source[source_id]["scores"].append(float(np.mean(values)))
            source_metrics = [_classification_metrics(value["labels"], value["scores"]) for value in per_source.values()]
            macro = _mean_metric_dict(source_metrics)
            output.append({
                "condition": condition, "split": split, "aggregation": "source_macro",
                "source_count": len(per_source),
                "fold_seed_rows": len({(str(x["held_out_source"]), int(x["seed"])) for x in selected}),
                "score_rows": sum(len(value["labels"]) for value in per_source.values()), **macro,
                "score_note": "mean seed/fold logit per source/window, then equal mean of source metrics",
            })
    return output


def _forward_audit(root: Path, rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    model_data = json.loads((root / "models/fold_models.json").read_text(encoding="utf-8"))
    records = model_data["records"]
    record_map = {(str(x["condition"]), str(x["held_out_source"]), int(x["seed"])): x for x in records}
    oof = {str(x["window_id"]): x for x in _read_csv(root / "scores/oof_window_scores.csv")}
    rows_out: list[dict[str, Any]] = []
    score_rows: list[dict[str, Any]] = []
    summary: list[dict[str, Any]] = []
    for condition in FORWARD_CONDITIONS:
        organization, _pooling, arm = condition.split("_")
        examples = _examples(root, rows, organization)
        main = _main_examples(examples)
        fold_rows: list[dict[str, Any]] = []
        for held_out in sorted({str(x["held_out_source"]) for x in records if str(x["condition"]) == condition}):
            training = [x for x in main if str(x["source_id"]) != held_out]
            heldout = [x for x in main if str(x["source_id"]) == held_out]
            if not training or not heldout:
                continue
            for seed in SEEDS:
                record = record_map[(condition, held_out, int(seed))]
                model, standardizer = _restore_model(record)
                existing_arm = {"A": "UNORDERED_STATE", "C": "ORDERED_SECOND"}[arm]
                train_batch, _ = build_batch(training, existing_arm, standardizer=standardizer, observation_weights=True)
                held_batch, _ = build_batch(heldout, {"A": "UNORDERED_STATE", "C": "ORDERED_SECOND"}[arm], standardizer=standardizer)
                train_scores = score_pooling_model(model, train_batch)
                held_scores = score_pooling_model(model, held_batch)
                train_labels = [int(x["label"]) for x in training]
                held_labels = [int(x["label"]) for x in heldout]
                train_metrics = _classification_metrics(train_labels, train_scores)
                held_metrics = _classification_metrics(held_labels, held_scores)
                for example, score in zip(training, train_scores):
                    score_rows.append({"condition": condition, "split": "train", "held_out_source": held_out, "seed": int(seed), "source_id": str(example["source_id"]), "window_id": str(example["window_id"]), "label": int(example["label"]), "logit": float(score)})
                for example, score in zip(heldout, held_scores):
                    score_rows.append({"condition": condition, "split": "heldout", "held_out_source": held_out, "seed": int(seed), "source_id": str(example["source_id"]), "window_id": str(example["window_id"]), "label": int(example["label"]), "logit": float(score)})
                diffs: list[float] = []
                for example, score in zip(heldout, held_scores):
                    saved = oof[str(example["window_id"])].get(f"{condition}_seed_{seed}")
                    if saved not in (None, ""):
                        diffs.append(abs(float(score) - float(saved)))
                row = {
                    "condition": condition, "organization": organization, "arm": arm,
                    "held_out_source": held_out, "seed": int(seed),
                    "train_windows": len(training), "heldout_windows": len(heldout),
                    "train_roc_auc": train_metrics["roc_auc"], "heldout_roc_auc": held_metrics["roc_auc"],
                    "train_average_precision": train_metrics["average_precision"], "heldout_average_precision": held_metrics["average_precision"],
                    "train_precision": train_metrics["precision"], "heldout_precision": held_metrics["precision"],
                    "train_recall": train_metrics["recall"], "heldout_recall": held_metrics["recall"],
                    "train_f1": train_metrics["f1"], "heldout_f1": held_metrics["f1"],
                    "train_accuracy": train_metrics["accuracy"], "heldout_accuracy": held_metrics["accuracy"],
                    "heldout_tn": held_metrics["tn"], "heldout_fp": held_metrics["fp"],
                    "heldout_fn": held_metrics["fn"], "heldout_tp": held_metrics["tp"],
                    "oof_compare_count": len(diffs), "max_abs_oof_logit_diff": max(diffs, default=None),
                    "forward_only": True, "optimizer_called": False,
                }
                rows_out.append(row); fold_rows.append(row)
        numeric = [key for key in fold_rows[0] if key.startswith(("train_", "heldout_")) and key not in {"train_windows", "heldout_windows"}]
        result: dict[str, Any] = {"condition": condition, "fold_seed_count": len(fold_rows), "train_windows_mean": float(np.mean([x["train_windows"] for x in fold_rows])) if fold_rows else None, "heldout_windows_mean": float(np.mean([x["heldout_windows"] for x in fold_rows])) if fold_rows else None}
        for key in numeric:
            values = [float(x[key]) for x in fold_rows if x[key] is not None]
            result[f"{key}_mean"] = float(np.mean(values)) if values else None
        diffs = [float(x["max_abs_oof_logit_diff"]) for x in fold_rows if x["max_abs_oof_logit_diff"] is not None]
        result["max_abs_oof_logit_diff"] = max(diffs, default=None)
        summary.append(result)
    _write_csv(root / "evaluation/train_heldout_forward_metrics.csv", rows_out)
    _write_csv(root / "evaluation/train_heldout_forward_summary.csv", summary)
    _write_csv(root / "evaluation/train_heldout_forward_scores.csv", score_rows)
    aggregate = _aggregate_forward_scores(score_rows)
    _write_csv(root / "evaluation/train_heldout_forward_aggregate.csv", aggregate)
    for name in ("train_heldout_forward_metrics.csv", "train_heldout_forward_summary.csv", "train_heldout_forward_aggregate.csv"):
        (REPO_ROOT / "docs/experiments" / f"v7-boundary-pooling-{name}").write_text((root / "evaluation" / name).read_text(encoding="utf-8"), encoding="utf-8")
    return {"rows": rows_out, "summary": summary, "score_rows": score_rows, "aggregate": aggregate}


def _track_slot_map(sequence: Any) -> dict[int, int]:
    """Return the explicit track-id to ParticleSequence slot mapping.

    Track IDs are identities; slots are array positions.  They happen to be
    equal in some cached examples, but the review page must never rely on
    that incidental property.
    """

    return {int(track_id): int(slot) for slot, track_id in enumerate(sequence.track_ids.tolist())}


def _group_owner_ids(
    triplet: Mapping[str, Any],
    groups: Sequence[Mapping[str, Any]],
    track_to_slot: Mapping[int, int],
) -> list[int]:
    considered_slots = {int(x) for x in triplet.get("members_considered", [])}
    common_ids = {int(x) for x in triplet.get("common_track_ids", [])}
    common_slots = {track_to_slot[x] for x in common_ids if x in track_to_slot}
    required_slots = considered_slots or common_slots
    owners: list[int] = []
    for group in groups:
        slots = {int(x) for x in group.get("member_slots", [])}
        if required_slots and required_slots.issubset(slots):
            owners.append(int(group["local_group_id"]))
    return sorted(owners)


def _triplet_display(
    triplet: Mapping[str, Any],
    *,
    track_to_slot: Mapping[int, int] | None = None,
    owner_group_ids: Sequence[int] = (),
) -> dict[str, Any]:
    states = np.asarray(triplet["states"], dtype=np.float64)
    timestamps = np.asarray(triplet["timestamps_s"], dtype=np.float64)
    center, first, second = compute_local_derivatives(states, timestamps)
    common_track_ids = [int(x) for x in triplet["common_track_ids"]]
    pair_ids = [[int(a), int(b)] for a, b in triplet["pair_ids"]]
    if track_to_slot is None:
        track_to_slot = {}
    common_member_slots = [int(track_to_slot[x]) for x in common_track_ids if x in track_to_slot]
    pair_member_slots = [
        [int(track_to_slot[a]), int(track_to_slot[b])]
        for a, b in pair_ids
        if a in track_to_slot and b in track_to_slot
    ]
    return {
        "triplet_id": int(triplet["triplet_id"]), "component_index": int(triplet["component_index"]),
        "common_track_ids": common_track_ids, "common_member_slots": common_member_slots,
        "pair_ids": pair_ids, "pair_member_slots": pair_member_slots,
        "members_considered": [int(x) for x in triplet.get("members_considered", [])],
        "track_ids_considered": [int(x) for x in triplet.get("track_ids_considered", [])],
        "owner_group_ids": [int(x) for x in owner_group_ids],
        "states": states.tolist(), "timestamps_s": timestamps.tolist(),
        "center_state": center.tolist(), "first_derivative": first.tolist(), "second_derivative": second.tolist(),
        "target_slots": [int(x) for x in triplet["target_slots"]],
    }


def _group_pairs(group: Mapping[str, Any]) -> list[tuple[int, int]]:
    members = [int(x) for x in group.get("member_slots", [])]
    return [(int(a), int(b)) for a, b in combinations(members, 2)]


def _boundary_relation_stats(h_group: Mapping[str, Any], b_groups: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    members = set(int(x) for x in h_group.get("member_slots", []))
    children = [g for g in b_groups if int(g.get("parent_local_group_id", -1)) == int(h_group.get("local_group_id", -1))]
    assignment: dict[int, int] = {}
    for child in children:
        for slot in child.get("member_slots", []): assignment[int(slot)] = int(child.get("local_group_id", -1))
    kept: list[list[int]] = []; cut: list[list[int]] = []; unassigned_pairs: list[list[int]] = []
    for left, right in _group_pairs(h_group):
        a, b = assignment.get(left, -1), assignment.get(right, -1)
        if a >= 0 and a == b: kept.append([left, right])
        else: cut.append([left, right])
        if a < 0 or b < 0: unassigned_pairs.append([left, right])
    return {
        "h_group_id": int(h_group.get("local_group_id", -1)),
        "h_member_slots": sorted(members),
        "b_child_group_ids": [int(g.get("local_group_id", -1)) for g in children],
        "h_pair_total": len(_group_pairs(h_group)),
        "pairs_retained_within_one_b_child": len(kept),
        "pairs_cut_by_boundary": len(cut),
        "pairs_with_unassigned_endpoint": len(unassigned_pairs),
        "retained_pair_examples": kept[:80], "cut_pair_examples": cut[:80],
        "unassigned_pair_examples": unassigned_pairs[:80],
    }


def _model_responses(root: Path, examples_by_org: Mapping[str, Sequence[Mapping[str, Any]]], source_id: str, window_id: str) -> dict[str, Any]:
    model_data = json.loads((root / "models/fold_models.json").read_text(encoding="utf-8"))
    records = {(str(x["condition"]), str(x["held_out_source"]), int(x["seed"])): x for x in model_data["records"]}
    output: dict[str, Any] = {}
    for condition in FORWARD_CONDITIONS:
        organization, _pooling, arm_code = condition.split("_")
        arm = {"A": "UNORDERED_STATE", "C": "ORDERED_SECOND"}[arm_code]
        example = next((x for x in examples_by_org[organization] if str(x["window_id"]) == window_id), None)
        if example is None:
            output[condition] = {"status": "NO_VALID_SUPPORT"}; continue
        seed_scores: list[float] = []; q_by_group: dict[int, list[float]] = defaultdict(list)
        for seed in SEEDS:
            record = records.get((condition, source_id, int(seed)))
            if record is None:
                continue
            model, standardizer = _restore_model(record)
            batch, _ = build_batch([example], arm, standardizer=standardizer)
            seed_scores.append(float(score_pooling_model(model, batch)[0]))
            q = model.group_logits(batch).numpy().tolist()
            keys = sorted({int(t["component_index"]) for t in example["triplets"]})
            for key, value in zip(keys, q): q_by_group[key].append(float(value))
        groups = [{"group_id": int(key), "component_index": int(key), "q_mean": float(np.mean(values)), "seed_count": len(values), "aggregation_weight": 1.0 / len(q_by_group), "weighted_contribution": float(np.mean(values) / len(q_by_group)), "scope": "organization_local_group_component"} for key, values in sorted(q_by_group.items())]
        output[condition] = {"status": "AVAILABLE" if seed_scores else "MODEL_MISSING", "window_logit_mean": float(np.mean(seed_scores)) if seed_scores else None, "seed_count": len(seed_scores), "groups": groups, "score_semantics": "classifier logit response; not calibrated probability or localization truth", "group_response_scope": "organization local-group component; component_index follows the frozen support triplet, while parent_component_id remains a separate grouping identity"}
    return output


def _frame_records(sequence: Any, model_frame_indices: set[int]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for index, frame_index in enumerate(sequence.frame_indices.tolist()):
        uv = np.asarray(sequence.uv[index], dtype=np.float64)
        vis = np.asarray(sequence.visibility[index], dtype=bool)
        geo = np.asarray(sequence.geometry_validity[index], dtype=bool)
        points = [[
            int(slot), int(sequence.track_ids[slot]),
            None if not np.isfinite(uv[slot, 0]) else float(uv[slot, 0]),
            None if not np.isfinite(uv[slot, 1]) else float(uv[slot, 1]),
            bool(vis[slot]), bool(geo[slot]),
        ] for slot in range(sequence.num_tracks)]
        output.append({"array_index": int(index), "source_frame_index": int(frame_index), "timestamp_s": float(sequence.timestamps_s[index]), "height": int(sequence.frame_sizes_hw[index, 0]), "width": int(sequence.frame_sizes_hw[index, 1]), "visible_count": int(np.sum(vis)), "geometry_count": int(np.sum(geo)), "model_used": int(frame_index) in model_frame_indices, "points": points})
    return output


def _observation_frame_match(
    frame_records: Sequence[Mapping[str, Any]],
    media_time_s: float,
    tolerance_s: float | None = None,
) -> dict[str, Any]:
    """Match browser media time to a saved PTS without crossing gaps."""

    records = sorted(frame_records, key=lambda item: float(item["timestamp_s"]))
    if not records:
        return {"status": "NO_OBSERVATION", "reason": "empty_saved_observation"}
    times = np.asarray([float(item["timestamp_s"]) for item in records], dtype=np.float64)
    value = float(media_time_s)
    if not np.isfinite(value) or value < times[0] or value > times[-1]:
        return {
            "status": "OUTSIDE_OBSERVATION",
            "reason": "media_time_outside_saved_observation_range",
            "observation_start_s": float(times[0]), "observation_end_s": float(times[-1]),
        }
    positive = np.diff(times); positive = positive[positive > 0]
    default_tolerance = float(0.75 * np.median(positive)) if positive.size else 1e-3
    tolerance = float(tolerance_s if tolerance_s is not None else max(default_tolerance, 1e-3))
    index = int(np.argmin(np.abs(times - value)))
    delta = float(value - times[index])
    # A browser time in a genuinely missing interval must not be assigned to
    # either endpoint.  The saved sequence is the only observation timeline.
    if index and value < times[index] and times[index] - times[index - 1] > 2.0 * tolerance:
        return {"status": "NO_OBSERVATION", "reason": "saved_observation_gap", "delta_s": delta, "tolerance_s": tolerance}
    if index + 1 < len(times) and value > times[index] and times[index + 1] - times[index] > 2.0 * tolerance:
        return {"status": "NO_OBSERVATION", "reason": "saved_observation_gap", "delta_s": delta, "tolerance_s": tolerance}
    if abs(delta) > tolerance:
        return {"status": "NO_OBSERVATION", "reason": "no_saved_pts_within_tolerance", "delta_s": delta, "tolerance_s": tolerance}
    return {"status": "MATCHED", "record": records[index], "delta_s": delta, "tolerance_s": tolerance}


def _make_screenshot(video_path: Path, frame_index: int, output: Path, h_slots: set[int], b_slots: set[int], sequence: Any, label: str) -> None:
    from PIL import Image, ImageDraw

    decoded = decode_video(VideoSource(sample_id=f"review-{output.stem}", source_video_id=label, source_locator=video_path), [int(frame_index)])
    image = Image.fromarray(decoded.frames[0].rgb)
    draw = ImageDraw.Draw(image)
    # This is a trajectory point overlay. No mask is propagated beyond the
    # first-frame reference; later frames only use frozen slot identities.
    uv = np.asarray(sequence.uv[int(np.flatnonzero(sequence.frame_indices == frame_index)[0])], dtype=np.float64)
    vis = np.asarray(sequence.visibility[int(np.flatnonzero(sequence.frame_indices == frame_index)[0])], dtype=bool)
    geo = np.asarray(sequence.geometry_validity[int(np.flatnonzero(sequence.frame_indices == frame_index)[0])], dtype=bool)
    for slot in range(sequence.num_tracks):
        if not vis[slot] or not np.all(np.isfinite(uv[slot])): continue
        x, y = float(uv[slot, 0]), float(uv[slot, 1]); color = (80, 210, 100) if geo[slot] else (150, 150, 150)
        if slot in h_slots: color = (20, 180, 240)
        if slot in b_slots: color = (255, 190, 40)
        radius = 5 if slot in h_slots or slot in b_slots else 2
        draw.ellipse((x-radius, y-radius, x+radius, y+radius), fill=color)
    draw.rectangle((0, 0, 760, 42), fill=(0, 0, 0))
    draw.text((10, 10), f"frame {frame_index} | {label} | H cyan / B yellow; first-frame mask reference only", fill=(255, 255, 255))
    output.parent.mkdir(parents=True, exist_ok=True); image.save(output, format="PNG")


def _source_frames_for_timestamps(sequence: Any, timestamps_s: Sequence[float]) -> list[int]:
    """Map stored triplet PTS to the exact source frame indices in the sequence."""

    sequence_times = np.asarray(sequence.timestamps_s, dtype=np.float64)
    sequence_frames = np.asarray(sequence.frame_indices, dtype=np.int64)
    if sequence_times.size == 0:
        raise ValueError("ParticleSequence has no timestamps")
    output: list[int] = []
    for timestamp in timestamps_s:
        nearest = int(np.argmin(np.abs(sequence_times - float(timestamp))))
        if not np.isclose(sequence_times[nearest], float(timestamp), rtol=0.0, atol=1e-6):
            raise ValueError(f"triplet timestamp {timestamp!r} is not present in ParticleSequence")
        output.append(int(sequence_frames[nearest]))
    return output


def _detail(
    root: Path,
    row: Mapping[str, Any],
    examples_by_org: Mapping[str, Sequence[Mapping[str, Any]]],
    default_case: bool,
    *,
    cached_model_response: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    window_id = str(row["window_id"]); feature = _load_feature(root, window_id); identity = feature["identity"]
    sequence = load_particle_sequence(Path(str(identity["particle_prefix"])))
    h_groups = [g for g in feature["h_grouping"].get("groups", []) if bool(g.get("retained"))]
    b_groups = [g for g in feature["b_grouping"].get("groups", []) if bool(g.get("retained"))]
    track_to_slot = _track_slot_map(sequence)
    h_triplets = [
        _triplet_display(
            t,
            track_to_slot=track_to_slot,
            owner_group_ids=_group_owner_ids(t, h_groups, track_to_slot),
        ) for t in feature["h_support"].get("triplets", [])
    ]
    b_triplets = [
        _triplet_display(
            t,
            track_to_slot=track_to_slot,
            owner_group_ids=_group_owner_ids(t, b_groups, track_to_slot),
        ) for t in feature["b_support"].get("triplets", [])
    ]
    for organization_groups, organization_triplets in ((h_groups, h_triplets), (b_groups, b_triplets)):
        for group in organization_groups:
            group_id = int(group.get("local_group_id", -1))
            group["response_component_indices"] = sorted({
                int(triplet["component_index"])
                for triplet in organization_triplets
                if group_id in {int(x) for x in triplet.get("owner_group_ids", [])}
            })
    model_frame_indices = {frame for triplet in feature["h_support"].get("triplets", []) for frame in _source_frames_for_timestamps(sequence, triplet["timestamps_s"])}
    selected_h = h_groups[0] if h_groups else None
    selected_b = next((g for g in b_groups if selected_h is not None and int(g.get("parent_local_group_id", -1)) == int(selected_h.get("local_group_id", -2))), None)
    screenshots: list[dict[str, Any]] = []
    video_path = Path(str(identity["video_path"]))
    media_path = None
    if video_path.is_file():
        media_dir = root / "review/media"; media_dir.mkdir(parents=True, exist_ok=True)
        media_path = media_dir / f"{_safe(window_id)}__{identity['role']}.mp4"
        if not media_path.exists(): media_path.symlink_to(os.path.relpath(video_path, media_path.parent))
    if default_case and video_path.is_file() and selected_h is not None and h_triplets:
        screenshot_dir = root / "review/screenshots"; h_slots = set(int(x) for x in selected_h.get("member_slots", [])); b_slots = set(int(x) for x in selected_b.get("member_slots", [])) if selected_b else set()
        target_frames = _source_frames_for_timestamps(sequence, h_triplets[0]["timestamps_s"])
        for position, frame_index in enumerate(target_frames):
            path = screenshot_dir / f"{_safe(window_id)}__displayfix__t{position}.png"
            if not path.is_file():
                _make_screenshot(
                    video_path, frame_index, path, h_slots, b_slots, sequence,
                    f"{identity['source_id']} {identity['role']} | {window_id} | t{position}",
                )
            screenshots.append({"position": position, "source_frame_index": frame_index, "timestamp_s": float(identity["timestamps_s"][int(np.flatnonzero(sequence.frame_indices == frame_index)[0])]), "path": str(path.relative_to(root / "review"))})
    mappings = [_boundary_relation_stats(group, feature["b_grouping"].get("groups", [])) for group in feature["h_grouping"].get("groups", []) if bool(group.get("retained"))]
    model_response = dict(cached_model_response) if cached_model_response is not None else _model_responses(root, examples_by_org, str(identity["source_id"]), window_id)
    for response in model_response.values():
        if not isinstance(response, dict):
            continue
        response["group_response_scope"] = "organization local-group component; component_index follows the frozen support triplet, while parent_component_id remains a separate grouping identity"
        for group in response.get("groups", []):
            if isinstance(group, dict):
                group["scope"] = "organization_local_group_component"
    return {
        "identity": {key: identity[key] for key in ("window_id", "source_id", "pair_id", "kind", "role", "label", "anchor_fraction", "interval_start_s", "interval_end_s", "frame_indices", "timestamps_s", "track_count", "video_path", "source_frame_index", "source_timestamp_s")},
        "video": {"status": "AVAILABLE" if video_path.is_file() else "SOURCE_MISSING", "materialization_status": "AVAILABLE" if media_path else "NOT_MATERIALIZED", "path": str(video_path), "relative_media": str(media_path.relative_to(root / "review")) if media_path else None},
        "support": {"H": {"status": feature["h_support"].get("support_status"), "groups": h_groups, "triplets": h_triplets}, "B": {"status": feature["b_support"].get("support_status"), "groups": b_groups, "triplets": b_triplets}},
        "boundary": {"checks": feature.get("boundary_checks"), "segmentation": {"status": feature.get("segmentation", {}).get("status"), "assignment_counts": feature.get("segmentation", {}).get("assignment_counts"), "reference_note": "assignment/mask is a first-frame boundary reference; never propagated as a later-frame mask"}, "mappings": mappings},
        "frame_records": _frame_records(sequence, model_frame_indices),
        "observation_range": {
            "start_s": float(sequence.timestamps_s[0]),
            "end_s": float(sequence.timestamps_s[-1]),
            "frame_count": int(sequence.num_frames),
            "source_frame_start": int(sequence.frame_indices[0]),
            "source_frame_end": int(sequence.frame_indices[-1]),
            "time_origin": "saved PyAV frame PTS * stream time_base; no FPS fallback",
        },
        "window_range": {"start_s": float(identity["interval_start_s"]), "end_s": float(identity["interval_end_s"])},
        "model_target_timestamps_s": sorted({float(x) for t in h_triplets + b_triplets for x in t["timestamps_s"]}),
        "model_frame_indices": sorted(model_frame_indices),
        "track_id_to_slot": {str(track_id): int(slot) for track_id, slot in track_to_slot.items()},
        "model_response": model_response,
        "default_selection": {"h_group_id": int(selected_h.get("local_group_id")) if selected_h else None, "b_group_id": int(selected_b.get("local_group_id")) if selected_b else None, "triplet_id": int(h_triplets[0]["triplet_id"]) if h_triplets else None},
        "screenshots": screenshots,
        "display_boundaries": ["colors identify local group membership, not real/fake", "q is an organization-local-group component classifier response, not a calibrated probability or localization truth", "visible, geometry-valid, triplet-common and model-used remain separate", "first-frame segmentation is reference-only; later overlays use frozen track IDs"],
    }


def materialize_windows(root: Path, window_ids: Sequence[str]) -> list[dict[str, Any]]:
    """Create only lightweight review links for explicitly requested windows."""

    rows = _read_rows(root)
    by_id = {str(row["window_id"]): row for row in rows}
    review_root = root / "review"
    results: list[dict[str, Any]] = []
    for window_id in window_ids:
        row = by_id.get(str(window_id))
        if row is None:
            results.append({"window_id": str(window_id), "status": "WINDOW_NOT_FOUND"})
            continue
        detail_path = review_root / "details" / f"{_safe(str(window_id))}.json"
        if not detail_path.is_file():
            results.append({"window_id": str(window_id), "status": "DETAIL_NOT_FOUND"})
            continue
        detail = json.loads(detail_path.read_text(encoding="utf-8"))
        video_path = Path(str(detail["video"]["path"]))
        if not video_path.is_file():
            detail["video"]["materialization_status"] = "SOURCE_MISSING"
            _write_json(detail_path, detail)
            results.append({"window_id": str(window_id), "status": "SOURCE_MISSING"})
            continue
        media_dir = review_root / "media"
        media_dir.mkdir(parents=True, exist_ok=True)
        media_path = media_dir / f"{_safe(str(window_id))}__{detail['identity']['role']}.mp4"
        if media_path.exists() and not media_path.is_symlink():
            raise FileExistsError(f"refusing to overwrite non-symlink media path: {media_path}")
        if not media_path.exists():
            media_path.symlink_to(os.path.relpath(video_path, media_path.parent))
        detail["video"]["status"] = "AVAILABLE"
        detail["video"]["materialization_status"] = "AVAILABLE"
        detail["video"]["relative_media"] = str(media_path.relative_to(review_root))
        _write_json(detail_path, detail)
        results.append({"window_id": str(window_id), "status": "AVAILABLE", "media": str(media_path)})
    index_path = review_root / "index_data.json"
    if index_path.is_file():
        index = json.loads(index_path.read_text(encoding="utf-8"))
        requested = {str(x) for x in window_ids}
        for entry in index.get("windows", []):
            if str(entry.get("window_id")) in requested:
                detail = json.loads((review_root / entry["detail_path"]).read_text(encoding="utf-8"))
                entry["video_status"] = detail["video"]["status"]
                entry["materialization_status"] = detail["video"].get("materialization_status")
        _write_json(index_path, index)
    return results


def _build_index(root: Path, rows: Sequence[Mapping[str, Any]], details: Mapping[str, Mapping[str, Any]], default_ids: set[str]) -> dict[str, Any]:
    entries = []
    for row in rows:
        window_id = str(row["window_id"]); feature = _load_feature(root, window_id); identity = feature["identity"]
        entries.append({"window_id": window_id, "source_id": str(row["source_id"]), "pair_id": str(row["pair_id"]), "kind": str(row["kind"]), "role": str(row["role"]), "anchor_fraction": float(row["anchor_fraction"]), "support_h": str(feature["h_support"].get("support_status")), "support_b": str(feature["b_support"].get("support_status")), "h_group_count": int(feature["h_grouping"].get("retained_group_count", 0)), "b_group_count": int(feature["b_grouping"].get("retained_group_count", 0)), "video_status": details[window_id]["video"]["status"], "materialization_status": details[window_id]["video"]["materialization_status"], "default_case": window_id in default_ids, "detail_path": f"details/{_safe(window_id)}.json"})
    return {"windows": entries, "default_case_count": len(default_ids), "page_notes": ["index covers all 192 frozen windows", "default cases are earliest MANIP/CTRL real/fake for four deterministic sources", "a browser visual check is not implied by generation"]}


def _html_page() -> str:
    # The page is deliberately dependency-free: local HTML, fetch, canvas and
    # the browser's native video element are sufficient for this review task.
    return r'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>V7 H/B Boundary Observation Review</title><style>
body{font-family:system-ui,sans-serif;margin:0;background:#111827;color:#e5e7eb}main{max-width:1600px;margin:auto;padding:16px}.panel{background:#1f2937;border:1px solid #374151;border-radius:8px;padding:12px;margin:10px 0}.controls{display:flex;gap:8px;flex-wrap:wrap;align-items:center}select,button{background:#0f172a;color:#e5e7eb;border:1px solid #64748b;border-radius:4px;padding:5px}.grid{display:grid;grid-template-columns:1fr 1fr;gap:12px}.side{background:#0f172a;padding:8px;border-radius:7px;border:1px solid #334155}.video{position:relative;background:#000}.video video{width:100%;display:block}.video canvas{position:absolute;inset:0;width:100%;height:100%;pointer-events:none}.small{font-size:12px;color:#cbd5e1}.warn{color:#fca5a5}.good{color:#86efac}.table{overflow:auto;max-height:330px}table{border-collapse:collapse;width:100%;font-size:12px}th,td{border:1px solid #475569;padding:4px;text-align:left;vertical-align:top}.thumbs img{max-width:240px;max-height:140px;margin:4px;border:1px solid #64748b}.cols{columns:2}.metric{display:inline-block;margin:4px 10px 4px 0}.badge{padding:2px 5px;border:1px solid #64748b;border-radius:4px}.mono{font-family:ui-monospace,monospace;word-break:break-all}
</style></head><body><main><h1>V7 H/B 观测对照审查页</h1><div class="panel small">这是冻结 boundary/pooling pilot 的离线观测审查，不是新 detector。页面只读取已保存的 289 点、H/B 局部组和已有模型前向响应。颜色可切换为观测状态或稳定的局部组身份，不表示 real/fake；局部分数是分类 logit 响应，不是校准概率或空间真值。首帧 segmentation assignment 只作边界参考，后续帧只画冻结 track ID 轨迹，不传播 mask。</div><div class="panel controls"><label>source <select id="source"></select></label><label>kind <select id="kind"><option>MANIP</option><option>CTRL</option></select></label><label>real window <select id="realWindow"></select></label><label>fake window <select id="fakeWindow"></select></label><label>model <select id="model"><option>H_MEAN_A</option><option>B_MEAN_A</option><option>B_MEAN_C</option></select></label><label>颜色 <select id="displayMode"><option value="status">观测状态</option><option value="group">分组身份</option></select></label><label><input id="showAll" type="checkbox" checked> visible</label><label><input id="showGeo" type="checkbox" checked> geometry-valid</label><label><input id="showCommon" type="checkbox" checked> selected triplet-common</label><label><input id="showPairs" type="checkbox" checked> selected pair</label></div><div class="panel small" id="status">加载索引…</div><div class="grid"><section class="side"><h2>real（独立选择）</h2><div class="video"><video id="realVideo" controls preload="metadata"></video><canvas id="realCanvas"></canvas></div><div id="realInfo" class="small"></div><div class="controls"><label>H parent <select id="realH"></select></label><label>B child <select id="realB"></select></label><label>triplet <select id="realTriplet"></select></label><button id="realObs">观测起点</button><button id="realT0">t0</button><button id="realT1">t1</button><button id="realT2">t2</button></div><div id="realMetrics" class="small"></div></section><section class="side"><h2>fake（独立选择）</h2><div class="video"><video id="fakeVideo" controls preload="metadata"></video><canvas id="fakeCanvas"></canvas></div><div id="fakeInfo" class="small"></div><div class="controls"><label>H parent <select id="fakeH"></select></label><label>B child <select id="fakeB"></select></label><label>triplet <select id="fakeTriplet"></select></label><button id="fakeObs">观测起点</button><button id="fakeT0">t0</button><button id="fakeT1">t1</button><button id="fakeT2">t2</button></div><div id="fakeMetrics" class="small"></div></section></div><div class="panel"><h2>精确帧与覆盖计数</h2><p class="small">播放器 seek 只作浏览；下表的源 frame index、PTS 和三时刻截图才是精确核对依据。model-used 是已保存 triplet 目标帧，不等于每一帧都进入模型。</p><div class="grid"><div class="table" id="realFrames"></div><div class="table" id="fakeFrames"></div></div></div><div class="panel"><h2>选定 triplet：S(t)、一阶/二阶量</h2><div class="grid"><div class="table" id="realCurve"></div><div class="table" id="fakeCurve"></div></div></div><div class="panel"><h2>H→B 边界关系</h2><div class="grid"><div id="realBoundary"></div><div id="fakeBoundary"></div></div></div><div class="panel"><h2>精确三时刻截图（默认案例会物化）</h2><div class="grid"><div id="realShots" class="thumbs"></div><div id="fakeShots" class="thumbs"></div></div></div><div class="panel small"><h2>限制</h2><div class="cols"><p>当前页面显示的是稀疏观测点，不是像素分割。visible、geometry-valid、triplet-common、model-used 分开统计；没有支撑不补成 0。real/fake 的 component、H parent、B child、triplet 和 pair 不假定一一对应。MANIP/CTRL 是数据集时间窗口标签；CTRL 不参加主训练。</p><p>默认案例：按 source 排序取前四个 source，各取最早 MANIP/CTRL 的 real/fake。其他窗口在索引中可选，但如果未物化媒体则显示 SOURCE_MISSING。页面没有外部 CDN。</p></div></div></main><script>
const state={index:null,details:new Map(),current:{real:null,fake:null},sel:{real:{h:null,b:null,t:null},fake:{h:null,b:null,t:null}}};const $=id=>document.getElementById(id);const esc=x=>String(x??'').replace(/[&<>"']/g,m=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[m]));
async function loadDetail(id){if(state.details.has(id))return state.details.get(id);const d=await fetch(state.indexBase+'/'+state.index.windows.find(x=>x.window_id===id).detail_path).then(r=>r.json());state.details.set(id,d);return d}
function sourceList(){return [...new Set(state.index.windows.map(x=>x.source_id))].sort()}
function choices(role){const source=$("source").value,kind=$("kind").value;return state.index.windows.filter(x=>x.source_id===source&&x.kind===kind&&x.role===role).sort((a,b)=>a.anchor_fraction-b.anchor_fraction||a.window_id.localeCompare(b.window_id))}
function fillWindow(role){const sel=$(role+'Window'),old=sel.value;const list=choices(role);sel.innerHTML=list.map(x=>`<option value="${esc(x.window_id)}">${esc(x.window_id)} (${esc(x.support_h)}/${esc(x.support_b)})</option>`).join('');if(list.some(x=>x.window_id===old))sel.value=old;else if(list.length)sel.value=list[0].window_id;state.current[role]=sel.value;}
function fillSources(){const s=$("source"),previous=s.value,sources=sourceList();s.innerHTML=sources.map(x=>`<option>${esc(x)}</option>`).join('');if(sources.includes(previous))s.value=previous;else if(s.options.length)s.value=sources[0].value;fillWindow('real');fillWindow('fake');loadBoth()}
function detailGroups(d,org){return (d.support[org].groups||[]).filter(x=>x.retained)}
function fillGroups(role,d){const sel=$(role+'H'),old=state.sel[role].h;const gs=detailGroups(d,'H');sel.innerHTML=gs.map(g=>`<option value="${g.local_group_id}">H${g.local_group_id} (${g.member_slots.length})</option>`).join('');state.sel[role].h=gs.some(g=>String(g.local_group_id)===String(old))?old:(gs[0]?.local_group_id??null);sel.value=state.sel[role].h??'';fillB(role,d);fillTriplet(role,d)}
function fillB(role,d){const sel=$(role+'B'),h=Number(state.sel[role].h),bs=detailGroups(d,'B').filter(x=>Number(x.parent_local_group_id)===h),old=state.sel[role].b;sel.innerHTML=bs.map(g=>`<option value="${g.local_group_id}">B${g.local_group_id} (${g.member_slots.length})</option>`).join('');state.sel[role].b=bs.some(g=>String(g.local_group_id)===String(old))?old:(bs[0]?.local_group_id??null);sel.value=state.sel[role].b??''}
function fillTriplet(role,d){const ts=d.support.H.triplets||[],old=state.sel[role].t;$(role+'Triplet').innerHTML=ts.map(t=>`<option value="${t.triplet_id}">H triplet ${t.triplet_id} / group ${t.component_index}</option>`).join('');state.sel[role].t=ts.some(t=>String(t.triplet_id)===String(old))?old:(ts[0]?.triplet_id??null);$(role+'Triplet').value=state.sel[role].t??''}
function selectedTriplet(role,d){return (d.support.H.triplets||[]).find(t=>String(t.triplet_id)===String(state.sel[role].t))||null}
function selectedGroup(d,org,id){return detailGroups(d,org).find(g=>String(g.local_group_id)===String(id))||null}
function frameTable(role,d){const fs=d.frame_records||[];return `<table><tr><th>array</th><th>source frame</th><th>PTS(s)</th><th>visible</th><th>geometry</th><th>model-used</th></tr>${fs.map(f=>`<tr class="${f.model_used?'good':''}"><td>${f.array_index}</td><td>${f.source_frame_index}</td><td>${f.timestamp_s.toFixed(6)}</td><td>${f.visible_count}</td><td>${f.geometry_count}</td><td>${f.model_used?'yes':''}</td></tr>`).join('')}</table>`}
function curveTable(role,d){const t=selectedTriplet(role,d);if(!t)return '<p class="warn">无有效 triplet 支撑</p>';return `<table><tr><th>PTS(s)</th><th>mean</th><th>std</th><th>p25</th><th>p75</th><th>一阶</th><th>二阶</th></tr>${t.states.map((s,i)=>`<tr><td>${t.timestamps_s[i].toFixed(6)}</td>${s.map(x=>`<td>${Number(x).toFixed(6)}</td>`).join('')}<td>${i===1?t.first_derivative.map(x=>Number(x).toFixed(6)).join(', '):''}</td><td>${i===1?t.second_derivative.map(x=>Number(x).toFixed(6)).join(', '):''}</td></tr>`).join('')}</table><p class="small">common track IDs: ${t.common_track_ids.join(', ')}；pair 总数：${t.pair_ids.length}。一阶/二阶仅是已保存表示的派生量。</p>`}
function boundaryTable(role,d){const h=selectedGroup(d,'H',state.sel[role].h);const b=selectedGroup(d,'B',state.sel[role].b);const stat=(d.boundary.mappings||[]).find(x=>Number(x.h_group_id)===Number(state.sel[role].h));if(!h)return '<p class="warn">无 H group</p>';return `<p>H${h.local_group_id} 成员 ${h.member_slots.length}；B child：${stat?.b_child_group_ids?.join(', ')||'无'}；选择 B${b?.local_group_id??'—'}。</p><p>H pair 总数 ${stat?.h_pair_total??0}；边界内保留 ${stat?.pairs_retained_within_one_b_child??0}；被切断 ${stat?.pairs_cut_by_boundary??0}；含未分配端点 ${stat?.pairs_with_unassigned_endpoint??0}。</p><p class="small">保留示例：${esc((stat?.retained_pair_examples||[]).slice(0,8).map(x=>`(${x[0]},${x[1]})`).join(' '))||'无'}；被切断示例：${esc((stat?.cut_pair_examples||[]).slice(0,8).map(x=>`(${x[0]},${x[1]})`).join(' '))||'无'}</p><p class="small">H→B 是首帧分组映射；颜色不表示真假。首帧 assignment/mask 只在首帧参考，不传播为后续 mask。</p>`}
function modelText(role,d){const cond=$("model").value,r=d.model_response?.[cond];if(!r||r.status!=='AVAILABLE')return '<span class="warn">没有该 source 的已保存 fold 模型</span>';const h=selectedGroup(d,'H',state.sel[role].h),b=selectedGroup(d,'B',state.sel[role].b),selected=b||h,component=selected?.parent_component_id,g=r.groups.find(x=>Number(x.component_index)===Number(component));return `<span class="metric">window logit ${Number(r.window_logit_mean).toFixed(6)}</span><span class="metric">selected q ${g?Number(g.q_mean).toFixed(6):'NA'}</span><span class="metric">weight ${g?Number(g.aggregation_weight).toFixed(6):'NA'}</span><span class="metric">contribution ${g?Number(g.weighted_contribution).toFixed(6):'NA'}</span><br><span class="small">classifier response；不是校准概率或定位真值；seed count ${r.seed_count}；component ${component??'NA'}</span>`}
function shots(role,d){return d.screenshots?.length?d.screenshots.map(x=>`<a href="${esc(x.path)}" target="_blank"><img src="${esc(x.path)}" title="frame ${x.source_frame_index} PTS ${x.timestamp_s}"></a>`).join(''):'<span class="small">该窗口未物化三时刻截图</span>'}
function info(role,d){const i=d.identity;return `<div class="mono">${esc(i.window_id)} | source ${esc(i.source_id)} | ${esc(i.role)} | ${esc(i.kind)} | anchor ${i.anchor_fraction}</div><div>video: ${esc(d.video.status)}；media: ${esc(d.video.materialization_status||'NA')}；H ${esc(d.support.H.status)} / B ${esc(d.support.B.status)}；model frames: ${d.model_frame_indices.join(', ')||'none'}</div><div class="small">${esc(d.video.path)}</div>`}
function draw(role){const d=state.details.get(state.current[role]),v=$(role+'Video'),c=$(role+'Canvas');if(!d||!v.videoWidth||!v.clientWidth)return;c.width=v.clientWidth*devicePixelRatio;c.height=v.clientHeight*devicePixelRatio;const ctx=c.getContext('2d'),fr=(d.frame_records||[]).reduce((a,b)=>Math.abs(b.timestamp_s-v.currentTime)<Math.abs(a.timestamp_s-v.currentTime)?b:a);const sx=c.width/v.videoWidth,sy=c.height/v.videoHeight;ctx.clearRect(0,0,c.width,c.height);const h=selectedGroup(d,'H',state.sel[role].h),b=selectedGroup(d,'B',state.sel[role].b),hs=new Set((h?.member_slots||[]).map(Number)),bs=new Set((b?.member_slots||[]).map(Number)),t=selectedTriplet(role,d),ts=new Set((t?.common_track_ids||[]).map(Number));for(const p of fr.points||[]){const [slot,u,y,vis,geo]=p;if(!vis||u===null||y===null)continue;let color='#94a3b8',r=2.5;if(bs.has(slot)){color='#fbbf24';r=5}else if(hs.has(slot)){color='#22d3ee';r=5}else if(ts.has(slot)&&$('showCommon').checked){color='#f472b6';r=4}else if(geo&&$('showGeo').checked){color='#4ade80';r=3}else if(!$('showAll').checked)continue;ctx.fillStyle=color;ctx.beginPath();ctx.arc(u*sx,y*sy,r,0,Math.PI*2);ctx.fill()}if($('showPairs').checked&&t){const pos=t.timestamps_s.reduce((best,x,i)=>Math.abs(x-fr.timestamp_s)<Math.abs(t.timestamps_s[best]-fr.timestamp_s)?i:best,0);const pts=Object.fromEntries((fr.points||[]).filter(p=>p[1]!==null&&p[2]!==null&&p[3]).map(p=>[p[0],[p[1],p[2]]]));for(const pair of t.pair_ids||[]){const a=pts[pair[0]],bb=pts[pair[1]];if(!a||!bb)continue;ctx.strokeStyle='#64748b';ctx.lineWidth=1;ctx.beginPath();ctx.moveTo(a[0]*sx,a[1]*sy);ctx.lineTo(bb[0]*sx,bb[1]*sy);ctx.stroke()}}}
async function renderRole(role){const d=await loadDetail(state.current[role]);fillGroups(role,d);$(role+'Info').innerHTML=info(role,d);$(role+'Frames').innerHTML=frameTable(role,d);$(role+'Curve').innerHTML=curveTable(role,d);$(role+'Boundary').innerHTML=boundaryTable(role,d);$(role+'Metrics').innerHTML=modelText(role,d);$(role+'Shots').innerHTML=shots(role,d);const v=$(role+'Video');v.src=d.video.relative_media||'';v.dataset.windowId=state.current[role];v.onloadedmetadata=()=>draw(role);v.ontimeupdate=()=>draw(role)}
async function loadBoth(){if(!state.index)return;await Promise.all([renderRole('real'),renderRole('fake')]);$('status').textContent=`索引 ${state.index.windows.length} 窗口；默认物化 ${state.index.default_case_count} 案例。实际点数和 PTS 见两侧表格。`}
async function init(){state.indexBase='.';state.index=await fetch('index_data.json').then(r=>r.json());$('source').innerHTML=sourceList().map(x=>`<option>${esc(x)}</option>`).join('');fillSources();for(const role of ['real','fake']){$(role+'Window').onchange=()=>{state.current[role]=$(role+'Window').value;renderRole(role)};$(role+'H').onchange=()=>{state.sel[role].h=$(role+'H').value;renderRole(role)};$(role+'B').onchange=()=>{state.sel[role].b=$(role+'B').value;renderRole(role)};$(role+'Triplet').onchange=()=>{state.sel[role].t=$(role+'Triplet').value;renderRole(role)}}$('source').onchange=fillSources;$('kind').onchange=fillSources;$('model').onchange=()=>{renderRole('real');renderRole('fake')};for(const id of ['showAll','showGeo','showCommon','showPairs'])$(id).onchange=()=>{draw('real');draw('fake')};window.addEventListener('resize',()=>{draw('real');draw('fake')})}init().catch(e=>$('status').textContent='加载失败：'+e);</script></body></html>'''


def _fixed_review_html() -> str:
    """Dependency-free review page with explicit time and identity mapping."""

    return r'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>V7 H/B Boundary Observation Review</title><style>
body{font-family:system-ui,sans-serif;margin:0;background:#111827;color:#e5e7eb}main{max-width:1600px;margin:auto;padding:16px}.panel{background:#1f2937;border:1px solid #374151;border-radius:8px;padding:12px;margin:10px 0}.controls{display:flex;gap:8px;flex-wrap:wrap;align-items:center}select,button{background:#0f172a;color:#e5e7eb;border:1px solid #64748b;border-radius:4px;padding:5px}.grid{display:grid;grid-template-columns:1fr 1fr;gap:12px}.side{background:#0f172a;padding:8px;border-radius:7px;border:1px solid #334155}.video{position:relative;background:#000}.video video{width:100%;display:block}.video canvas{position:absolute;inset:0;width:100%;height:100%;pointer-events:none}.small{font-size:12px;color:#cbd5e1}.warn{color:#fca5a5}.good{color:#86efac}.table{overflow:auto;max-height:330px}table{border-collapse:collapse;width:100%;font-size:12px}th,td{border:1px solid #475569;padding:4px;text-align:left;vertical-align:top}.thumbs img{max-width:240px;max-height:140px;margin:4px;border:1px solid #64748b}.cols{columns:2}.metric{display:inline-block;margin:4px 10px 4px 0}.mono{font-family:ui-monospace,monospace;word-break:break-all}
</style></head><body><main><h1>V7 H/B 观测对照审查页</h1><div class="panel small">这是冻结 boundary/pooling pilot 的离线观测审查，不是新 detector。页面只读取已保存的 289 点、H/B 局部组和已有模型前向响应。颜色可切换为观测状态或稳定的局部组身份，不表示 real/fake；局部分数是分类 logit 响应，不是校准概率或空间真值。首帧 segmentation assignment 只作边界参考，后续帧只画冻结 track ID 轨迹，不传播 mask。</div><div class="panel controls"><label>source <select id="source"></select></label><label>kind <select id="kind"><option>MANIP</option><option>CTRL</option></select></label><label>real window <select id="realWindow"></select></label><label>fake window <select id="fakeWindow"></select></label><label>model <select id="model"><option>H_MEAN_A</option><option>B_MEAN_A</option><option>B_MEAN_C</option></select></label><label>颜色 <select id="displayMode"><option value="status">观测状态</option><option value="group">分组身份</option></select></label><label><input id="showAll" type="checkbox" checked> visible</label><label><input id="showGeo" type="checkbox" checked> geometry-valid</label><label><input id="showCommon" type="checkbox" checked> selected triplet-common</label><label><input id="showPairs" type="checkbox" checked> selected pair</label></div><div class="panel small" id="status">加载索引…</div><div class="grid"><section class="side"><h2>real（独立选择）</h2><div class="video"><video id="realVideo" controls preload="metadata"></video><canvas id="realCanvas"></canvas></div><div id="realInfo" class="small"></div><div class="controls"><label>H parent <select id="realH"></select></label><label>B child <select id="realB"></select></label><label>triplet <select id="realTriplet"></select></label><button id="realObs">观测起点</button><button id="realT0">t0</button><button id="realT1">t1</button><button id="realT2">t2</button></div><div id="realMetrics" class="small"></div></section><section class="side"><h2>fake（独立选择）</h2><div class="video"><video id="fakeVideo" controls preload="metadata"></video><canvas id="fakeCanvas"></canvas></div><div id="fakeInfo" class="small"></div><div class="controls"><label>H parent <select id="fakeH"></select></label><label>B child <select id="fakeB"></select></label><label>triplet <select id="fakeTriplet"></select></label><button id="fakeObs">观测起点</button><button id="fakeT0">t0</button><button id="fakeT1">t1</button><button id="fakeT2">t2</button></div><div id="fakeMetrics" class="small"></div></section></div><div class="panel"><h2>精确帧与覆盖计数</h2><p class="small">播放器 seek 只作浏览；下表的源 frame index、PTS 和三时刻截图才是精确核对依据。model-used 是已保存 triplet 目标帧，不等于每一帧都进入模型。</p><div class="grid"><div class="table" id="realFrames"></div><div class="table" id="fakeFrames"></div></div></div><div class="panel"><h2>选定 triplet：S(t)、一阶/二阶量</h2><div class="grid"><div class="table" id="realCurve"></div><div class="table" id="fakeCurve"></div></div></div><div class="panel"><h2>H→B 边界关系</h2><div class="grid"><div id="realBoundary"></div><div id="fakeBoundary"></div></div></div><div class="panel"><h2>精确三时刻截图（默认案例会物化）</h2><div class="grid"><div id="realShots" class="thumbs"></div><div id="fakeShots" class="thumbs"></div></div></div><div class="panel small"><h2>限制</h2><div class="cols"><p>当前页面显示的是稀疏观测点，不是像素分割。visible、geometry-valid、triplet-common、model-used 分开统计；没有支撑不补成 0。real/fake 的 component、H parent、B child、triplet 和 pair 不假定一一对应。MANIP/CTRL 是数据集时间窗口标签；CTRL 不参加主训练。</p><p>默认案例：按 source 排序取前四个 source，各取最早 MANIP/CTRL 的 real/fake。其他窗口在索引中可选，但如果未物化媒体则显示 SOURCE_MISSING。页面没有外部 CDN。</p></div></div></main><script>
const state={index:null,details:new Map(),current:{real:null,fake:null},sel:{real:{h:null,b:null,t:null},fake:{h:null,b:null,t:null}}};const $=id=>document.getElementById(id);const esc=x=>String(x??'').replace(/[&<>"']/g,m=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[m]));const runtime={token:{real:0,fake:0},callback:{real:false,fake:false}};
async function loadDetail(id){if(state.details.has(id))return state.details.get(id);const entry=state.index.windows.find(x=>x.window_id===id);const d=await fetch('./'+entry.detail_path).then(r=>r.json());state.details.set(id,d);return d}
function selectedOrganization(){return String($('model').value||'H').startsWith('B_')?'B':'H'}function sourceList(){return [...new Set(state.index.windows.map(x=>x.source_id))].sort()}function choices(role){const s=$('source').value,k=$('kind').value;return state.index.windows.filter(x=>x.source_id===s&&x.kind===k&&x.role===role).sort((a,b)=>a.anchor_fraction-b.anchor_fraction||a.window_id.localeCompare(b.window_id))}function fillWindow(role){const sel=$(role+'Window'),old=sel.value,list=choices(role);sel.innerHTML=list.map(x=>`<option value="${esc(x.window_id)}">${esc(x.window_id)} (${esc(x.support_h)}/${esc(x.support_b)})</option>`).join('');if(list.some(x=>x.window_id===old))sel.value=old;else if(list.length)sel.value=list[0].window_id;state.current[role]=sel.value}
function fillSources(){const s=$('source'),old=s.value,list=sourceList();s.innerHTML=list.map(x=>`<option value="${esc(x)}">${esc(x)}</option>`).join('');if(list.includes(old))s.value=old;else if(s.options.length)s.value=list[0];fillWindow('real');fillWindow('fake');loadBoth()}
function groups(d,org){return (d.support?.[org]?.groups||[]).filter(x=>x.retained)}function group(d,org,id){return groups(d,org).find(x=>String(x.local_group_id)===String(id))||null}function selectedGroup(role,d){const org=selectedOrganization();return {org,group:group(d,org,org==='B'?state.sel[role].b:state.sel[role].h)}}
function fillGroups(role,d){const hs=$(role+'H'),ho=state.sel[role].h,hgs=groups(d,'H');hs.innerHTML=hgs.map(g=>`<option value="${g.local_group_id}">H${g.local_group_id} (${g.member_slots.length})</option>`).join('');state.sel[role].h=hgs.some(g=>String(g.local_group_id)===String(ho))?ho:(hgs[0]?.local_group_id??null);hs.value=state.sel[role].h??'';const bs=$(role+'B'),bo=state.sel[role].b,bgs=groups(d,'B').filter(g=>Number(g.parent_local_group_id)===Number(state.sel[role].h));bs.innerHTML=bgs.map(g=>`<option value="${g.local_group_id}">B${g.local_group_id} (${g.member_slots.length})</option>`).join('');state.sel[role].b=bgs.some(g=>String(g.local_group_id)===String(bo))?bo:(bgs[0]?.local_group_id??null);bs.value=state.sel[role].b??'';fillTriplet(role,d)}
function triplets(role,d){const x=selectedGroup(role,d);if(!x.group)return[];return(d.support?.[x.org]?.triplets||[]).filter(t=>(t.owner_group_ids||[]).map(Number).includes(Number(x.group.local_group_id)))}function fillTriplet(role,d){const sel=$(role+'Triplet'),old=state.sel[role].t,ts=triplets(role,d);if(!ts.length){sel.innerHTML='<option value="">NO_VALID_TRIPLET</option>';sel.disabled=true;state.sel[role].t=null;return}sel.disabled=false;sel.innerHTML=ts.map(t=>`<option value="${t.triplet_id}">${selectedOrganization()} triplet ${t.triplet_id} / component ${t.component_index}</option>`).join('');state.sel[role].t=ts.some(t=>String(t.triplet_id)===String(old))?old:String(ts[0].triplet_id);sel.value=state.sel[role].t}function triplet(role,d){return triplets(role,d).find(t=>String(t.triplet_id)===String(state.sel[role].t))||null}
function frameTable(d){return`<table><tr><th>array</th><th>source frame</th><th>PTS(s)</th><th>visible</th><th>geometry</th><th>model-used</th></tr>${(d.frame_records||[]).map(f=>`<tr class="${f.model_used?'good':''}"><td>${f.array_index}</td><td>${f.source_frame_index}</td><td>${Number(f.timestamp_s).toFixed(6)}</td><td>${f.visible_count}</td><td>${f.geometry_count}</td><td>${f.model_used?'yes':''}</td></tr>`).join('')}</table>`}
function curveTable(role,d){const t=triplet(role,d);if(!t)return'<p class="warn">NO_VALID_TRIPLET：当前组织和局部组没有共同支撑；不补成 0 分。</p>';return`<table><tr><th>PTS(s)</th><th>mean</th><th>std</th><th>p25</th><th>p75</th><th>一阶</th><th>二阶</th></tr>${t.states.map((s,i)=>`<tr><td>${Number(t.timestamps_s[i]).toFixed(6)}</td>${s.map(x=>`<td>${Number(x).toFixed(6)}</td>`).join('')}<td>${i===1?t.first_derivative.map(x=>Number(x).toFixed(6)).join(', '):''}</td><td>${i===1?t.second_derivative.map(x=>Number(x).toFixed(6)).join(', '):''}</td></tr>`).join('')}</table><p class="small">${selectedOrganization()} common track IDs: ${t.common_track_ids.join(', ')}；common slots: ${t.common_member_slots.join(', ')}；pair 总数：${t.pair_member_slots.length}。</p>`}
function boundaryTable(role,d){const h=group(d,'H',state.sel[role].h),b=group(d,'B',state.sel[role].b),m=(d.boundary.mappings||[]).find(x=>Number(x.h_group_id)===Number(state.sel[role].h));if(!h)return'<p class="warn">无 H group</p>';return`<p>H${h.local_group_id} 成员 ${h.member_slots.length}；B child：${m?.b_child_group_ids?.join(', ')||'无'}；选择 B${b?.local_group_id??'—'}。</p><p>H pair 总数 ${m?.h_pair_total??0}；边界内保留 ${m?.pairs_retained_within_one_b_child??0}；被切断 ${m?.pairs_cut_by_boundary??0}；含未分配端点 ${m?.pairs_with_unassigned_endpoint??0}。</p><p class="small">保留示例：${esc((m?.retained_pair_examples||[]).slice(0,8).map(x=>`(${x[0]},${x[1]})`).join(' '))||'无'}；被切断示例：${esc((m?.cut_pair_examples||[]).slice(0,8).map(x=>`(${x[0]},${x[1]})`).join(' '))||'无'}</p><p class="small">首帧 assignment/mask 只作首帧参考，后续只使用冻结 track ID。</p>`}
function modelText(role,d){const c=$('model').value,r=d.model_response?.[c],x=selectedGroup(role,d);if(!r||r.status!=='AVAILABLE')return'<span class="warn">没有该 source 的已保存 fold 模型</span>';const component=x.group?.parent_component_id,q=(r.groups||[]).find(g=>Number(g.component_index)===Number(component));return`<span class="metric">window logit ${r.window_logit_mean==null?'NA':Number(r.window_logit_mean).toFixed(6)}</span><span class="metric">父 component q ${q?Number(q.q_mean).toFixed(6):'NA'}</span><span class="metric">weight ${q?Number(q.aggregation_weight).toFixed(6):'NA'}</span><span class="metric">contribution ${q?Number(q.weighted_contribution).toFixed(6):'NA'}</span><br><span class="small">${x.org} local_group ${x.group?.local_group_id??'NA'} → parent component ${component??'NA'}；q 是 parent-component 分类响应，不是 local-group 概率或定位真值；seed count ${r.seed_count}</span>`}
function info(role,d){const i=d.identity,f=d.frame_records||[],x=selectedGroup(role,d),t=triplet(role,d),q=(d.model_response?.[$('model').value]?.groups||[]);return`<div class="mono">${esc(i.window_id)} | source ${esc(i.source_id)} | ${esc(i.role)} | ${esc(i.kind)} | anchor ${i.anchor_fraction}</div><div>window ${Number(d.window_range.start_s).toFixed(3)}–${Number(d.window_range.end_s).toFixed(3)} s；saved observation ${f[0]?Number(f[0].timestamp_s).toFixed(3):'NA'}–${f.at(-1)?Number(f.at(-1).timestamp_s).toFixed(3):'NA'} s；H ${esc(d.support.H.status)} / B ${esc(d.support.B.status)}；model target frames ${d.model_frame_indices.join(', ')||'none'}</div><div>organization ${x.org}；local group ${x.group?.local_group_id??'NA'}；selected triplet ${t?.triplet_id??'NO_VALID_TRIPLET'}；groups with q ${q.length}</div><div class="small">video ${esc(d.video.status)}；media ${esc(d.video.materialization_status||'NA')}；${esc(d.observation_range.time_origin)}</div><div class="small">${esc(d.video.path)}</div>`}
function groupColor(id){const p=['#38bdf8','#fbbf24','#a78bfa','#fb7185','#34d399','#f472b6','#facc15','#60a5fa','#c084fc','#fb923c'];return p[Math.abs(Number(id)||0)%p.length]}
function matchFrame(d,time){const fs=d.frame_records||[];if(!fs.length)return{status:'NO_OBSERVATION',reason:'empty_saved_observation'};if(time<fs[0].timestamp_s||time>fs.at(-1).timestamp_s)return{status:'OUTSIDE_OBSERVATION',reason:'outside_saved_observation_range'};let b=0;for(let i=1;i<fs.length;i++)if(Math.abs(fs[i].timestamp_s-time)<Math.abs(fs[b].timestamp_s-time))b=i;const delta=time-fs[b].timestamp_s,dt=fs.length>1?Math.max(.001,.75*(fs[1].timestamp_s-fs[0].timestamp_s)):.001;return Math.abs(delta)<=dt?{status:'MATCHED',record:fs[b],delta_s:delta,tolerance_s:dt}:{status:'NO_OBSERVATION',reason:'no_saved_pts_within_tolerance',delta_s:delta,tolerance_s:dt}}
function draw(role){const d=state.details.get(state.current[role]),v=$(role+'Video'),c=$(role+'Canvas');if(!d||!v.videoWidth||!v.clientWidth)return;c.width=v.clientWidth*devicePixelRatio;c.height=v.clientHeight*devicePixelRatio;const ctx=c.getContext('2d'),m=matchFrame(d,v.currentTime);ctx.clearRect(0,0,c.width,c.height);if(m.status!=='MATCHED'){$(role+'Metrics').innerHTML=`<span class="warn">${m.status}：当前播放器时间不在已保存观测范围；不显示点、组或关系。</span>`;return}const f=m.record,sx=c.width/v.videoWidth,sy=c.height/v.videoHeight,x=selectedGroup(role,d),gset=new Set((x.group?.member_slots||[]).map(Number)),t=triplet(role,d),common=new Set((t?.common_member_slots||[]).map(Number)),groups0=groups(d,x.org),points=new Map(),mode=$('displayMode').value;let visible=0,geo=0,gvisible=0,cvisible=0;for(const p of f.points||[]){const[slot,track,u,y,vis,valid]=p;points.set(Number(slot),p);if(!vis||u===null||y===null)continue;visible++;if(valid)geo++;if(gset.has(Number(slot)))gvisible++;if(common.has(Number(slot)))cvisible++;let col='#94a3b8',r=2.5;if(mode==='group'){const owner=groups0.find(z=>(z.member_slots||[]).map(Number).includes(Number(slot)));col=owner?groupColor(owner.local_group_id):'#64748b';if(gset.has(Number(slot)))r=5}else if(common.has(Number(slot))&&$('showCommon').checked){col='#f472b6';r=5}else if(gset.has(Number(slot))){col=x.org==='B'?'#fbbf24':'#22d3ee';r=5}else if(valid&&$('showGeo').checked){col='#4ade80';r=3}else if(!$('showAll').checked)continue;ctx.fillStyle=col;ctx.beginPath();ctx.arc(u*sx,y*sy,r,0,Math.PI*2);ctx.fill()}let drawn=0;if($('showPairs').checked&&t)for(const pair of t.pair_member_slots||[]){const a=points.get(Number(pair[0])),b=points.get(Number(pair[1]));if(!a||!b||a[2]===null||a[3]===null||b[2]===null||b[3]===null||!a[4]||!b[4])continue;drawn++;ctx.strokeStyle='#64748b';ctx.lineWidth=1;ctx.beginPath();ctx.moveTo(a[2]*sx,a[3]*sy);ctx.lineTo(b[2]*sx,b[3]*sy);ctx.stroke()}const q=(d.model_response?.[$('model').value]?.groups||[]).filter(z=>groups0.some(z2=>Number(z2.parent_component_id)===Number(z.component_index))).length;$(role+'Metrics').innerHTML=`<span class="metric">playback ${v.currentTime.toFixed(3)} s</span><span class="metric">matched PTS ${f.timestamp_s.toFixed(6)} s</span><span class="metric">Δ ${m.delta_s.toFixed(6)} s</span><span class="metric">source frame ${f.source_frame_index}</span><span class="metric">model-used ${f.model_used?'yes':'no'}</span><br><span class="metric">visible ${visible}</span><span class="metric">geometry-valid ${geo}</span><span class="metric">${x.org} group members ${x.group?.member_slots?.length??0} / visible ${gvisible}</span><span class="metric">triplet-common ${cvisible}</span><span class="metric">pairs total ${t?.pair_member_slots?.length??0} / drawn ${drawn}</span><span class="metric">groups with q ${q}</span>`}
function shots(role,d){return d.screenshots?.length?d.screenshots.map(x=>`<a href="${esc(x.path)}" target="_blank"><img src="${esc(x.path)}" title="frame ${x.source_frame_index} PTS ${x.timestamp_s}"></a>`).join(''):'<span class="small">该窗口未物化三时刻截图</span>'}function panels(role,d){fillGroups(role,d);$(role+'Info').innerHTML=info(role,d);$(role+'Frames').innerHTML=frameTable(d);$(role+'Curve').innerHTML=curveTable(role,d);$(role+'Boundary').innerHTML=boundaryTable(role,d);$(role+'Metrics').innerHTML=modelText(role,d);$(role+'Shots').innerHTML=shots(role,d);draw(role)}async function renderRole(role){const id=state.current[role];if(id)panels(role,await loadDetail(id))}async function loadWindow(role,id){const tok=++runtime.token[role];state.current[role]=id;const d=await loadDetail(id);if(tok!==runtime.token[role])return;panels(role,d);const v=$(role+'Video');if(v.dataset.windowId!==id){v.dataset.windowId=id;v.dataset.seekedWindow='';v.src=d.video.relative_media||'';v.load();v.onloadedmetadata=()=>{if(tok!==runtime.token[role])return;if(v.dataset.seekedWindow!==id){v.currentTime=Number(d.observation_range.start_s);v.dataset.seekedWindow=id}draw(role);schedule(role)};v.ontimeupdate=()=>draw(role);v.onseeked=()=>draw(role);v.onplay=()=>schedule(role)}else draw(role)}async function loadBoth(){if(!state.index)return;await Promise.all([loadWindow('real',$('realWindow').value),loadWindow('fake',$('fakeWindow').value)]);$('status').textContent=`索引 ${state.index.windows.length} 窗口；两侧 source/window 独立选择；只在保存观测范围内绘制。`}
function schedule(role){const v=$(role+'Video');if(!v.requestVideoFrameCallback||v.paused||runtime.callback[role])return;runtime.callback[role]=true;v.requestVideoFrameCallback(()=>{runtime.callback[role]=false;draw(role);schedule(role)})}function jump(role,pos){const d=state.details.get(state.current[role]),t=triplet(role,d),v=$(role+'Video'),x=pos==='obs'?d?.observation_range?.start_s:t?.timestamps_s?.[Number(pos)];if(Number.isFinite(x)&&v.readyState>=1){v.currentTime=x;draw(role)}}async function start(){state.index=await fetch('./index_data.json').then(r=>r.json());$('source').innerHTML=sourceList().map(x=>`<option value="${esc(x)}">${esc(x)}</option>`).join('');fillSources();for(const role of['real','fake']){$(role+'Window').onchange=()=>loadWindow(role,$(role+'Window').value);$(role+'H').onchange=()=>{state.sel[role].h=$(role+'H').value;renderRole(role)};$(role+'B').onchange=()=>{state.sel[role].b=$(role+'B').value;renderRole(role)};$(role+'Triplet').onchange=()=>{state.sel[role].t=$(role+'Triplet').value;renderRole(role)};$(role+'Obs').onclick=()=>jump(role,'obs');['0','1','2'].forEach(i=>$(role+'T'+i).onclick=()=>jump(role,i))}$('source').onchange=fillSources;$('kind').onchange=fillSources;$('model').onchange=()=>{renderRole('real');renderRole('fake')};$('displayMode').onchange=()=>{draw('real');draw('fake')};for(const id of['showAll','showGeo','showCommon','showPairs'])$(id).onchange=()=>{draw('real');draw('fake')};window.addEventListener('resize',()=>{draw('real');draw('fake')})}start().catch(e=>$('status').textContent='加载失败：'+e);
function modelText(role,d){const c=$('model').value,r=d.model_response?.[c],x=selectedGroup(role,d);if(!r||r.status!=='AVAILABLE')return'<span class="warn">没有该 source 的已保存 fold 模型</span>';const parent=x.group?.parent_component_id,ids=(x.group?.response_component_indices||[]).map(Number),qs=(r.groups||[]).filter(g=>ids.includes(Number(g.component_index)));const q=qs[0];return`<span class="metric">window logit ${r.window_logit_mean==null?'NA':Number(r.window_logit_mean).toFixed(6)}</span><span class="metric">model response q ${q?Number(q.q_mean).toFixed(6):'NA'}</span><span class="metric">weight ${q?Number(q.aggregation_weight).toFixed(6):'NA'}</span><span class="metric">contribution ${q?Number(q.weighted_contribution).toFixed(6):'NA'}</span><br><span class="small">${x.org} local_group ${x.group?.local_group_id??'NA'}；original parent_component_id ${parent??'NA'}；triplet/model component_index ${ids.join(', ')||'NA'}；q scope=${q?.scope||'organization_local_group_component'}；不是校准概率或定位真值；seed count ${r.seed_count}</span>`}
</script></body></html>'''


def _html_page() -> str:
    """Return the current fixed review page (kept as the public helper name)."""

    return _fixed_review_html()


def _default_ids(rows: Sequence[Mapping[str, Any]], source_count: int = DEFAULT_SOURCE_COUNT) -> set[str]:
    sources = sorted({str(x["source_id"]) for x in rows})[:source_count]
    selected: set[str] = set()
    for source in sources:
        for kind in ("MANIP", "CTRL"):
            for role in ("real", "fake"):
                choices = [x for x in rows if str(x["source_id"]) == source and str(x["kind"]) == kind and str(x["role"]) == role]
                if choices: selected.add(str(sorted(choices, key=lambda x: (float(x["anchor_fraction"]), str(x["window_id"])))[0]["window_id"]))
    return selected


def build(root: Path) -> dict[str, Any]:
    rows = _read_rows(root); default_ids = _default_ids(rows)
    examples_by_org = {org: _examples(root, rows, org) for org in ("H", "B")}
    forward = _forward_audit(root, rows)
    review_root = root / "review"; detail_root = review_root / "details"; detail_root.mkdir(parents=True, exist_ok=True)
    details: dict[str, dict[str, Any]] = {}
    for index, row in enumerate(rows, 1):
        window_id = str(row["window_id"])
        value = _detail(root, row, examples_by_org, window_id in default_ids)
        details[window_id] = value
        _write_json(detail_root / f"{_safe(window_id)}.json", value)
        if index % 24 == 0: print(f"review details {index}/{len(rows)}", flush=True)
    _write_json(review_root / "index_data.json", _build_index(root, rows, details, default_ids))
    (review_root / "index.html").write_text(_fixed_review_html(), encoding="utf-8")
    summary = {"status": "FORWARD_REVIEW_READY", "artifact_root": str(root), "review_html": str(review_root / "index.html"), "index_windows": len(rows), "default_case_count": len(default_ids), "forward_conditions": list(FORWARD_CONDITIONS), "forward_rows": len(forward["rows"]), "forward_score_rows": len(forward["score_rows"]), "forward_aggregate": str(root / "evaluation/train_heldout_forward_aggregate.csv"), "browser_visual_verification": False, "boundaries": ["no optimizer", "no frontend/provider rerun", "no tracking/depth/pose/segmentation rerun", "no dense branch", "no video/model artifact copied into Git"]}
    _write_json(root / "review/run_summary.json", summary)
    return {"summary": summary, "forward": forward}


def rebuild_review_only(root: Path) -> dict[str, Any]:
    """Refresh display mappings and HTML from saved details without model work.

    The boundary-pooling forward audit is deliberately not called here.  An
    existing detail's serialized model response is reused, while sequence and
    feature files are read again to correct track/slot and H/B ownership
    mappings.  This keeps a display fix from changing the experiment inputs.
    """

    rows = _read_rows(root)
    default_ids = _default_ids(rows)
    examples_by_org = {org: _examples(root, rows, org) for org in ("H", "B")}
    review_root = root / "review"; detail_root = review_root / "details"; detail_root.mkdir(parents=True, exist_ok=True)
    details: dict[str, dict[str, Any]] = {}
    for index, row in enumerate(rows, 1):
        window_id = str(row["window_id"])
        old_path = detail_root / f"{_safe(window_id)}.json"
        cached = json.loads(old_path.read_text(encoding="utf-8")).get("model_response") if old_path.is_file() else None
        value = _detail(root, row, examples_by_org, window_id in default_ids, cached_model_response=cached)
        details[window_id] = value
        _write_json(old_path, value)
        if index % 24 == 0:
            print(f"review-only details {index}/{len(rows)}", flush=True)
    _write_json(review_root / "index_data.json", _build_index(root, rows, details, default_ids))
    (review_root / "index.html").write_text(_fixed_review_html(), encoding="utf-8")
    summary = {
        "status": "DISPLAY_REVIEW_REFRESHED",
        "artifact_root": str(root), "review_html": str(review_root / "index.html"),
        "index_windows": len(rows), "default_case_count": len(default_ids),
        "forward_audit_reused": True, "browser_visual_verification": False,
        "boundaries": ["no optimizer", "no frontend/provider rerun", "no tracking/depth/pose/segmentation rerun", "no model forward recomputation"],
    }
    _write_json(root / "review/run_summary.json", summary)
    return {"summary": summary}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--materialize-window", action="append", default=[], help="create a review symlink for this existing window; may be repeated")
    parser.add_argument("--rebuild-review-only", action="store_true", help="refresh display mappings without model forward or training")
    args = parser.parse_args()
    if args.materialize_window:
        print(json.dumps(materialize_windows(args.root, args.materialize_window), indent=2, ensure_ascii=False))
        return
    if args.rebuild_review_only:
        print(json.dumps(rebuild_review_only(args.root), indent=2, ensure_ascii=False))
        return
    started = time.time(); result = build(args.root); result["elapsed_s"] = time.time() - started
    print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
