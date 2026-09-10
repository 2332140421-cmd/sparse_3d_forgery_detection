"""Read-only joint audit of the frozen V7 structural-temporal pilot.

The immediate caller is the command line entry point at the bottom of this
file.  It restores saved checkpoints and standardizers for forward scoring,
reconstructs persisted pair relations, and builds a local review page.  It
does not call ``train_model``, perform an optimizer step, rerun any provider,
or change the detector.  The smallest useful implementation is one explicit
reader/audit module; a generic experiment framework would obscure the frozen
contracts this diagnostic is intended to verify.
"""

from __future__ import annotations

import argparse
import base64
import csv
from fractions import Fraction
import hashlib
import io
import json
import math
import re
import statistics
import struct
import subprocess
import time
import zlib
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch

from sparse3d_forgery.particle_sequence import load_particle_sequence

from research_tools.v7.local_structural_temporal_probe.model import (
    ARM_NAMES,
    Batch,
    WeightedStandardizer,
    WindowMLP,
    build_batch,
    fit_weighted_standardizer,
    score_model,
    weighted_window_bce,
    window_label,
)
from research_tools.v7.local_structural_temporal_probe.representation import (
    arm_inputs_for_triplet,
    compute_local_derivatives,
)
from research_tools.v7.local_structural_temporal_probe.run_pilot import (
    DATA_ROOT,
    SOURCE_ARTIFACT_ROOT,
    load_frozen_examples,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
PILOT_ROOT = DATA_ROOT / "derived/v7_activityforensics_local_structural_temporal_supervised_pilot_weightfix_v1"
DIAGNOSTIC_OUTPUT_ROOT = DATA_ROOT / "derived/v7_activityforensics_local_structural_temporal_joint_diagnostic_v1"
SEEDS = (20260909, 20260910, 20260911)
DIAGNOSTIC_BOOTSTRAP_SEED = 20260909
DIAGNOSTIC_BOOTSTRAP_REPLICATES = 10_000
REPRO_TOL_ABS = 1e-5
REPRO_TOL_REL = 1e-5
ROW_NAMES = {
    "UNORDERED_STATE": ("state_0", "state_1", "state_2"),
    "ORDERED_FIRST": ("triplet",),
    "ORDERED_SECOND": ("triplet",),
    "PERMUTED_SECOND": ("perm_012", "perm_021", "perm_102", "perm_120", "perm_201", "perm_210"),
}


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_head() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()


def _finite(values: Iterable[float]) -> np.ndarray:
    array = np.asarray(list(values), dtype=np.float64)
    return array[np.isfinite(array)]


def auroc(positive: Iterable[float], negative: Iterable[float]) -> float | None:
    """Tie-aware AUROC used only for training/held-out diagnostics."""

    pos, neg = _finite(positive), _finite(negative)
    if pos.size == 0 or neg.size == 0:
        return None
    comparison = (pos[:, None] > neg[None, :]).astype(np.float64)
    comparison += 0.5 * (pos[:, None] == neg[None, :])
    return float(np.mean(comparison))


def _summary(values: Iterable[float]) -> dict[str, Any]:
    array = _finite(values)
    if array.size == 0:
        return {"N": 0, "mean": None, "median": None, "p99": None, "min": None, "max": None}
    return {
        "N": int(array.size),
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "p99": float(np.percentile(array, 99)),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
    }


def aggregate_window_logit(component_logits: Sequence[Sequence[float]]) -> float:
    """Exact frozen aggregation: mean over triplets, then components."""

    if not component_logits or any(not values for values in component_logits):
        raise ValueError("at least one component and triplet are required")
    component_means = [float(np.mean(np.asarray(values, dtype=np.float64))) for values in component_logits]
    return float(np.mean(np.asarray(component_means, dtype=np.float64)))


def nearest_frame_index(frame_records: Sequence[Mapping[str, Any]], relative_time_s: float) -> int:
    """Return the array index nearest a player-relative time."""

    if not frame_records:
        raise ValueError("frame_records are empty")
    return int(min(frame_records, key=lambda row: abs(float(row["relative_time_s"]) - float(relative_time_s)))["array_index"])


def _restore_standardizer(record: Mapping[str, Any]) -> WeightedStandardizer:
    data = record["model"]["standardization"]
    return WeightedStandardizer(
        mean=np.asarray(data["mean"], dtype=np.float64),
        scale=np.asarray(data["scale"], dtype=np.float64),
        zero_variance_dimensions=tuple(int(item) for item in data.get("zero_variance_dimensions", [])),
    )


def _restore_model(record: Mapping[str, Any]) -> WindowMLP:
    model = WindowMLP()
    state = {
        name: torch.as_tensor(values, dtype=torch.float32)
        for name, values in record["model"]["state_dict"].items()
    }
    model.load_state_dict(state)
    model.eval()
    return model


def _load_oof(path: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for raw in csv.DictReader(handle):
            row: dict[str, Any] = {key: value for key, value in raw.items()}
            for key, value in list(row.items()):
                if key.endswith("_score") or "_score_seed_" in key or key.endswith("_seed_std"):
                    row[key] = float(value) if value not in (None, "") else None
            row["label"] = int(row["label"]) if row.get("label") not in (None, "") else None
            rows[str(row["window_id"])] = row
    return rows


def _load_model_records() -> dict[tuple[str, str, int], dict[str, Any]]:
    data = json.loads((PILOT_ROOT / "models/fold_models.json").read_text(encoding="utf-8"))
    records = {}
    for record in data.get("folds", []):
        key = (str(record["held_out_source"]), str(record["arm"]), int(record["seed"]))
        if key in records:
            raise ValueError(f"duplicate frozen model record: {key}")
        records[key] = record
    return records


def _triplets_for_example(example: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    values = example.get("triplets")
    if values is None:
        values = example.get("support", {}).get("triplets", [])
    return list(values or [])


def _batch_triplet_map(examples: Sequence[Mapping[str, Any]], arm: str) -> list[dict[str, Any]]:
    """Mirror build_batch iteration to retain human-readable component IDs."""

    result: list[dict[str, Any]] = []
    for example in examples:
        triplets = _triplets_for_example(example)
        by_component: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
        for triplet in triplets:
            by_component[int(triplet["component_index"])].append(triplet)
        for component_index in sorted(by_component):
            for triplet in by_component[component_index]:
                rows = arm_inputs_for_triplet(triplet)[arm]
                result.append(
                    {
                        "window_id": str(example["window_id"]),
                        "component_index": int(component_index),
                        "triplet_id": int(triplet["triplet_id"]),
                        "target_slots": [int(item) for item in triplet["target_slots"]],
                        "row_count": int(rows.shape[0]),
                    }
                )
    return result


def _decompose_model(model: WindowMLP, batch: Batch, examples: Sequence[Mapping[str, Any]], arm: str) -> dict[str, dict[str, Any]]:
    """Compute q values and the exact algebraic component/window means."""

    triplet_map = _batch_triplet_map(examples, arm)
    if len(triplet_map) != batch.n_triplets:
        raise ValueError("batch triplet map does not match model batch")
    with torch.no_grad():
        values = torch.as_tensor(batch.inputs, dtype=torch.float32)
        hidden = model.encoder(values)
        row_q = model.head(hidden).squeeze(-1).cpu().numpy().astype(np.float64)
        direct = model(batch).cpu().numpy().astype(np.float64)
    triplet_values: list[dict[str, Any]] = []
    row_offset = 0
    for triplet_id, info in enumerate(triplet_map):
        row_count = int(info["row_count"])
        q_rows = row_q[row_offset : row_offset + row_count]
        row_offset += row_count
        triplet_values.append(
            {
                **info,
                "q": float(np.mean(q_rows)),
                "row_q": [float(item) for item in q_rows],
                "row_weight": 1.0 / row_count,
                "row_names": list(ROW_NAMES[arm]),
            }
        )
    by_window: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in triplet_values:
        by_window[item["window_id"]].append(item)
    result: dict[str, dict[str, Any]] = {}
    for window_id, triplets in by_window.items():
        by_component: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for triplet in triplets:
            by_component[int(triplet["component_index"])].append(triplet)
        components: list[dict[str, Any]] = []
        component_logits: list[list[float]] = []
        for component_index in sorted(by_component):
            component_triplets = by_component[component_index]
            component_qs = [float(item["q"]) for item in component_triplets]
            component_q = float(np.mean(component_qs))
            triplet_count = len(component_triplets)
            component_logits.append(component_qs)
            components.append(
                {
                    "component_index": int(component_index),
                    "triplet_count": int(triplet_count),
                    "component_q": component_q,
                    "component_weight": 1.0 / len(by_component),
                    "component_contribution": component_q / len(by_component),
                    "triplets": [
                        {
                            **item,
                            "triplet_weight": 1.0 / triplet_count,
                            "weighted_contribution": float(item["q"]) / len(by_component) / triplet_count,
                            "local_row_contributions": [
                                float(q) / len(by_component) / triplet_count / int(item["row_count"])
                                for q in item["row_q"]
                            ],
                        }
                        for item in component_triplets
                    ],
                }
            )
        reconstructed = aggregate_window_logit(component_logits)
        # The window index is obtained from the first matching batch row.  This
        # avoids relying on window IDs being numerically formatted.
        first_tid = next(index for index, item in enumerate(triplet_map) if item["window_id"] == window_id)
        window_index = int(batch.row_window_ids[np.flatnonzero(batch.row_triplet_ids == first_tid)[0]])
        result[window_id] = {
            "window_id": window_id,
            "window_logit": float(direct[window_index]),
            "decomposed_window_logit": reconstructed,
            "decomposition_abs_error": abs(float(direct[window_index]) - reconstructed),
            "component_count": len(components),
            "components": components,
        }
    return result


def _source_train_metrics(examples: Sequence[Mapping[str, Any]], scores: np.ndarray) -> tuple[float | None, dict[str, Any]]:
    grouped: dict[str, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    for example, score in zip(examples, scores):
        label = window_label(example)
        if label is None:
            continue
        grouped[str(example["source_id"])][int(label)].append(float(score))
    values = [auroc(groups.get(1, []), groups.get(0, [])) for groups in grouped.values()]
    finite = [float(value) for value in values if value is not None]
    return (float(np.mean(finite)) if finite else None), {
        "source_count": len(finite),
        "source_aurocs": {source: auroc(groups.get(1, []), groups.get(0, [])) for source, groups in sorted(grouped.items())},
    }


def _bce(logits: np.ndarray, examples: Sequence[Mapping[str, Any]], weights: np.ndarray | None = None) -> float | None:
    labels = [window_label(example) for example in examples]
    keep = [index for index, label in enumerate(labels) if label is not None]
    if not keep:
        return None
    target = torch.as_tensor([float(labels[index]) for index in keep], dtype=torch.float32)
    values = torch.as_tensor(np.asarray(logits, dtype=np.float64)[keep], dtype=torch.float32)
    if weights is None:
        weight_tensor = torch.ones_like(target)
    else:
        weight_tensor = torch.as_tensor(np.asarray(weights, dtype=np.float64)[keep], dtype=torch.float32)
    return float(weighted_window_bce(values, target, weight_tensor).item())


def _model_diagnostics(
    examples: Sequence[Mapping[str, Any]],
    all_source_ids: Sequence[str],
    records: Mapping[tuple[str, str, int], Mapping[str, Any]],
    oof: Mapping[str, Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, dict[str, Any]], dict[tuple[str, str], list[dict[str, Any]]]]:
    main_examples = [example for example in examples if window_label(example) is not None]
    by_source = {source: [example for example in examples if str(example["source_id"]) == source] for source in all_source_ids}
    diagnostics: list[dict[str, Any]] = []
    loss_curves: list[dict[str, Any]] = []
    decomp_runs: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    errors: list[float] = []
    tolerance_failures: list[dict[str, Any]] = []
    standardizer_errors: list[float] = []
    expected_model_keys = {(source, arm, seed) for source in all_source_ids for arm in ARM_NAMES for seed in SEEDS}
    missing_model_keys = sorted(expected_model_keys - set(records))
    for held_out in all_source_ids:
        heldout = by_source.get(held_out, [])
        training = [example for example in main_examples if str(example["source_id"]) != held_out]
        if not heldout:
            continue
        for arm in ARM_NAMES:
            raw_batch, observation_weights = build_batch(training, arm, observation_weights=True)
            if observation_weights is None:
                raise RuntimeError("frozen diagnostic did not receive observation weights")
            recomputed = fit_weighted_standardizer(raw_batch.inputs, observation_weights)
            train_batch_for_recomputed, _ = build_batch(training, arm, standardizer=recomputed, observation_weights=True)
            del train_batch_for_recomputed
            heldout_batch_cache: dict[str, Any] = {}
            for seed in SEEDS:
                key = (held_out, arm, seed)
                if key not in records:
                    continue
                record = records[key]
                stored = _restore_standardizer(record)
                standardizer_error = max(
                    float(np.max(np.abs(stored.mean - recomputed.mean))),
                    float(np.max(np.abs(stored.scale - recomputed.scale))),
                )
                standardizer_errors.append(standardizer_error)
                train_batch, _ = build_batch(training, arm, standardizer=stored, observation_weights=True)
                held_batch = heldout_batch_cache.get("batch")
                if held_batch is None:
                    held_batch, _ = build_batch(heldout, arm, standardizer=stored)
                    heldout_batch_cache["batch"] = held_batch
                model = _restore_model(record)
                train_scores = score_model(model, train_batch)
                held_scores = score_model(model, held_batch)
                decomposition = _decompose_model(model, held_batch, heldout, arm)
                decomp_runs[(held_out, arm)].append({"seed": seed, "decomposition": decomposition})
                for example, score in zip(heldout, held_scores):
                    reference = oof.get(str(example["window_id"]), {}).get(f"{arm}_score_seed_{seed}")
                    if reference is None:
                        continue
                    error = abs(float(score) - float(reference))
                    errors.append(error)
                    allowed = REPRO_TOL_ABS + REPRO_TOL_REL * abs(float(reference))
                    if error > allowed:
                        tolerance_failures.append({"window_id": str(example["window_id"]), "arm": arm, "seed": seed, "error": error, "allowed": allowed})
                train_auroc, train_source = _source_train_metrics(training, train_scores)
                held_manip = [example for example in heldout if window_label(example) is not None]
                held_manip_scores = [float(score) for example, score in zip(heldout, held_scores) if window_label(example) is not None]
                held_auroc = auroc(
                    [score for example, score in zip(heldout, held_scores) if window_label(example) == 1],
                    [score for example, score in zip(heldout, held_scores) if window_label(example) == 0],
                )
                fit_info = record.get("fit", {})
                checkpoint_bce = _bce(train_scores, training, train_batch.window_weights)
                held_bce = _bce(np.asarray(held_manip_scores, dtype=np.float64), held_manip)
                diagnostics.append(
                    {
                        "held_out_source": held_out,
                        "arm": arm,
                        "seed": seed,
                        "training_source_count": len({str(item["source_id"]) for item in training}),
                        "training_window_count": len(training),
                        "training_real_count": sum(window_label(item) == 0 for item in training),
                        "training_fake_count": sum(window_label(item) == 1 for item in training),
                        "held_out_window_count": len(heldout),
                        "held_out_manip_count": len(held_manip),
                        "training_auroc_source_equal": train_auroc,
                        "training_auroc_source_count": train_source["source_count"],
                        "held_out_auroc_manip": held_auroc,
                        "training_weighted_bce": checkpoint_bce,
                        "held_out_bce_manip_uncalibrated_logit": held_bce,
                        "recorded_initial_loss": fit_info.get("initial_loss"),
                        "recorded_final_loss_before_update": fit_info.get("final_loss"),
                        "checkpoint_rescored_training_weighted_bce": checkpoint_bce,
                        "checkpoint_minus_recorded_final_loss": None if checkpoint_bce is None or fit_info.get("final_loss") is None else checkpoint_bce - float(fit_info["final_loss"]),
                        "max_oof_abs_error_for_model": max((abs(float(score) - float(oof[str(example["window_id"])] [f"{arm}_score_seed_{seed}"])) for example, score in zip(heldout, held_scores) if oof.get(str(example["window_id"]), {}).get(f"{arm}_score_seed_{seed}") is not None), default=0.0),
                        "standardizer_recomputed_max_abs_error": standardizer_error,
                        "zero_variance_dimensions": list(stored.zero_variance_dimensions),
                        "loss_history_length": len(fit_info.get("loss_history", [])),
                    }
                )
                loss_curves.append({"held_out_source": held_out, "arm": arm, "seed": seed, "loss_history": [float(value) for value in fit_info.get("loss_history", [])]})
                del model
    error_summary = {
        "comparison": "saved held-out per-seed logits against restored checkpoint forward pass",
        "tolerance": {"absolute": REPRO_TOL_ABS, "relative": REPRO_TOL_REL, "formula": "abs(error) <= 1e-5 + 1e-5*abs(reference)"},
        "score_count": len(errors),
        "absolute_error": _summary(errors),
        "tolerance_failure_count": len(tolerance_failures),
        "tolerance_failures": tolerance_failures[:20],
        "model_record_count": len(diagnostics),
        "expected_model_record_count_for_scored_sources": len(records),
        "missing_model_keys": [list(key) for key in missing_model_keys],
        "standardizer_recomputed_max_abs_error": max(standardizer_errors) if standardizer_errors else None,
    }
    return diagnostics, {"oof_reproduction": error_summary, "loss_curves": loss_curves}, loss_curves, decomp_runs


def _load_window_results() -> dict[str, dict[str, Any]]:
    rows = json.loads((SOURCE_ARTIFACT_ROOT / "frontend/window_results.json").read_text(encoding="utf-8"))
    return {str(row["window"]["window_id"]): row for row in rows}


def _load_window_manifest() -> dict[str, dict[str, Any]]:
    rows = json.loads((SOURCE_ARTIFACT_ROOT / "manifests/window_manifest.json").read_text(encoding="utf-8"))
    return {str(row["window_id"]): row for row in rows}


def _load_selected_pairs() -> list[dict[str, Any]]:
    return json.loads((SOURCE_ARTIFACT_ROOT / "manifests/selected_pairs.json").read_text(encoding="utf-8"))


def _frame_index_map(sequence: Any) -> dict[int, int]:
    return {int(value): index for index, value in enumerate(np.asarray(sequence.frame_indices, dtype=np.int64))}


def _component_for_slots(support: Mapping[str, Any]) -> dict[int, int]:
    result: dict[int, int] = {}
    for component in support.get("components", []):
        for member in component.get("member_indices", []):
            result.setdefault(int(member), int(component["component_index"]))
    return result


def _coverage_and_structure(
    input_rows: Sequence[Mapping[str, Any]],
    support_rows: Sequence[Mapping[str, Any]],
    window_results: Mapping[str, Mapping[str, Any]],
    window_manifest: Mapping[str, Mapping[str, Any]],
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]], dict[str, Any], dict[str, Any]]:
    support_by_id = {str(row["window_id"]): row for row in support_rows}
    coverage: dict[str, dict[str, Any]] = {}
    structure: dict[str, dict[str, Any]] = {}
    reconstruction_errors: list[float] = []
    reconstruction_failures: list[dict[str, Any]] = []
    native_curve_keys: dict[tuple[str, int, int, float], int] = {}
    native_curve_rows: list[dict[str, Any]] = []
    for input_row in input_rows:
        window_id = str(input_row["window_id"])
        support = support_by_id[window_id]
        sequence = load_particle_sequence(Path(input_row["particle_prefix"]))
        xyz = np.asarray(sequence.xyz, dtype=np.float64)
        uv = np.asarray(sequence.uv, dtype=np.float64)
        visible = np.asarray(sequence.visibility, dtype=bool)
        valid = np.asarray(sequence.geometry_validity, dtype=bool)
        timestamps = np.asarray(sequence.timestamps_s, dtype=np.float64)
        frame_indices = np.asarray(sequence.frame_indices, dtype=np.int64)
        frame_sizes = np.asarray(sequence.frame_sizes_hw, dtype=np.int64)
        track_ids = np.asarray(sequence.track_ids, dtype=np.int64)
        frame_map = _frame_index_map(sequence)
        component_for_slot = _component_for_slots(support)
        model_arrays = sorted({int(frame_map[int(frame)]) for triplet in support.get("triplets", []) for frame in triplet.get("frame_indices", []) if frame is not None and int(frame) in frame_map})
        frame_records: list[dict[str, Any]] = []
        for array_index in range(xyz.shape[0]):
            h, w = [int(value) for value in frame_sizes[array_index]]
            points = []
            for slot in range(xyz.shape[1]):
                if np.all(np.isfinite(uv[array_index, slot])):
                    points.append(
                        [
                            int(slot),
                            int(track_ids[slot]),
                            float(uv[array_index, slot, 0]),
                            float(uv[array_index, slot, 1]),
                            bool(visible[array_index, slot]),
                            bool(valid[array_index, slot]),
                            component_for_slot.get(slot),
                        ]
                    )
            frame_records.append(
                {
                    "array_index": int(array_index),
                    "source_frame_index": int(frame_indices[array_index]),
                    "timestamp_s": float(timestamps[array_index]),
                    "relative_time_s": float(timestamps[array_index] - timestamps[0]),
                    "height": h,
                    "width": w,
                    "initialized_track_slots": int(track_ids.size),
                    "visible_count": int(np.sum(visible[array_index])),
                    "geometry_valid_count": int(np.sum(valid[array_index])),
                    "uv_finite_count": int(np.sum(np.all(np.isfinite(uv[array_index]), axis=1))),
                    "xyz_finite_count": int(np.sum(np.all(np.isfinite(xyz[array_index]), axis=1))),
                    "model_used": int(array_index) in model_arrays,
                    "uv_points": points,
                }
            )
        triplet_records: list[dict[str, Any]] = []
        for triplet in support.get("triplets", []):
            frames = [int(item) for item in triplet["frame_indices"]]
            arrays = [int(frame_map[item]) for item in frames]
            times = [float(item) for item in triplet["timestamps_s"]]
            saved_states = np.asarray(triplet["states"], dtype=np.float64)
            reconstructed_distances: list[list[float]] = []
            triplet_pair_records: list[dict[str, Any]] = []
            for pair_index, pair in zip(triplet["pair_indices"], triplet["pair_ids"]):
                left, right = [int(item) for item in pair_index]
                scale = float(triplet["history_scale"])
                difference = xyz[:, right, :] - xyz[:, left, :]
                all_distances = np.linalg.norm(difference, axis=1)
                raw_target = [float(all_distances[array_index]) for array_index in arrays]
                normalized_target = [float(value / scale) for value in raw_target]
                reconstructed_distances.append(normalized_target)
                curve_key = (window_id, left, right, scale)
                curve_id = native_curve_keys.get(curve_key)
                if curve_id is None:
                    curve_id = len(native_curve_rows)
                    native_curve_keys[curve_key] = curve_id
                    valid_pair = valid[:, left] & valid[:, right] & np.all(np.isfinite(difference), axis=1) & np.isfinite(all_distances)
                    native_curve_rows.append(
                        {
                            "curve_id": int(curve_id),
                            "window_id": window_id,
                            "member_indices": [left, right],
                            "history_scale": scale,
                            "timestamps_s": [float(item) for item in timestamps],
                            "raw_distance": [float(value) if bool(ok) else None for value, ok in zip(all_distances, valid_pair)],
                        }
                    )
                triplet_pair_records.append(
                    {
                        "pair_id": [int(pair[0]), int(pair[1])],
                        "member_indices": [left, right],
                        "raw_distance": raw_target,
                        "normalized_distance": normalized_target,
                        "native_curve_id": int(curve_id),
                        "history_scale": scale,
                    }
                )
            reconstructed_states = np.asarray(
                [
                    [
                        np.mean(row),
                        np.std(row),
                        np.percentile(row, 25),
                        np.percentile(row, 75),
                    ]
                    for row in np.asarray(reconstructed_distances, dtype=np.float64).T
                ],
                dtype=np.float64,
            )
            state_error = float(np.max(np.abs(saved_states - reconstructed_states)))
            reconstruction_errors.append(state_error)
            if state_error > 1e-6:
                reconstruction_failures.append({"window_id": window_id, "triplet_id": int(triplet["triplet_id"]), "max_abs_error": state_error})
            center, first, second = compute_local_derivatives(saved_states, times)
            triplet_records.append(
                {
                    "triplet_id": int(triplet["triplet_id"]),
                    "component_index": int(triplet["component_index"]),
                    "target_slots": [int(item) for item in triplet["target_slots"]],
                    "source_frame_indices": frames,
                    "array_indices": arrays,
                    "timestamps_s": times,
                    "target_times_s": [float(item) for item in triplet["target_times_s"]],
                    "match_errors_s": [None if item is None else float(item) for item in triplet["match_errors_s"]],
                    "common_member_indices": [int(item) for item in triplet["common_member_indices"]],
                    "common_track_ids": [int(item) for item in triplet["common_track_ids"]],
                    "pair_count": len(triplet_pair_records),
                    "history_scale": float(triplet["history_scale"]),
                    "saved_states": saved_states.tolist(),
                    "reconstructed_states": reconstructed_states.tolist(),
                    "state_max_abs_error": state_error,
                    "center_state": center.tolist(),
                    "signed_first_difference": first.tolist(),
                    "signed_second_difference": second.tolist(),
                    "pair_records": triplet_pair_records,
                    "status": str(triplet.get("status", "VALID")),
                }
            )
        result_meta = window_results.get(window_id, {})
        tracking = result_meta.get("tracking", {})
        coverage[window_id] = {
            "window_id": window_id,
            "source_id": str(input_row["source_id"]),
            "pair_id": str(input_row["pair_id"]),
            "role": str(input_row["role"]),
            "kind": str(input_row["kind"]),
            "support_status": str(support["support_status"]),
            "invalid_reasons": [dict(item) for item in support.get("invalid_reasons", [])],
            "initial_query": {
                "requested_tracks": tracking.get("requested_tracks"),
                "initialized_track_slots": tracking.get("initialized_tracks", int(track_ids.size)),
                "track_ids_persisted": [int(item) for item in track_ids],
                "original_query_coordinates_persisted": False,
            },
            "history_frame_indices": [int(item) for item in support.get("history_frame_indices", [])],
            "history_array_indices": [int(item) for item in support.get("history_array_indices", [])],
            "evaluation_frame_indices": [int(item) for item in support.get("evaluation_frame_indices", [])],
            "evaluation_array_indices": [int(item) for item in support.get("evaluation_array_indices", [])],
            "target_matches": support.get("target_matches", []),
            "components": [
                {
                    "component_index": int(component["component_index"]),
                    "member_indices": [int(item) for item in component.get("member_indices", [])],
                    "track_ids": [int(item) for item in component.get("track_ids", [])],
                    "history_frame_count": int(component.get("history_frame_count", 0)),
                    "history_scale": component.get("history_scale"),
                    "valid_triplet_count": len(component.get("triplet_ids", [])),
                    "invalid_triplets": component.get("invalid_triplets", []),
                    "validity_source": component.get("validity_source"),
                }
                for component in support.get("components", [])
            ],
            "triplet_count": len(triplet_records),
            "model_frame_indices": [int(frame_indices[item]) for item in model_arrays],
            "model_array_indices": model_arrays,
            "frame_records": frame_records,
            "actual_model_time_records": [
                {
                    "triplet_id": int(triplet["triplet_id"]),
                    "source_frame_indices": [int(item) for item in triplet["frame_indices"]],
                    "array_indices": [int(frame_map[int(item)]) for item in triplet["frame_indices"]],
                    "timestamps_s": [float(item) for item in triplet["timestamps_s"]],
                    "target_times_s": [float(item) for item in triplet["target_times_s"]],
                    "match_errors_s": [None if item is None else float(item) for item in triplet["match_errors_s"]],
                }
                for triplet in support.get("triplets", [])
            ],
            "provider_elapsed_s": result_meta.get("elapsed_s"),
            "frontend_causal_training_eligible": result_meta.get("causal_training_eligible"),
        }
        structure[window_id] = {
            "window_id": window_id,
            "triplets": triplet_records,
            "background_rule": "native same-window timestamps only; no interpolation and no background value enters model inputs",
        }
    relation_summary = {
        "triplet_count": len(reconstruction_errors),
        "state_reconstruction_error": _summary(reconstruction_errors),
        "state_reconstruction_failure_count": len(reconstruction_failures),
        "state_reconstruction_failures": reconstruction_failures[:20],
        "uses_exact_saved_pair_support": True,
        "derivative_semantics": "signed timestamp-aware first/second differences; not physical velocity/acceleration",
    }
    if native_curve_rows:
        max_length = max(len(row["timestamps_s"]) for row in native_curve_rows)
        curve_times = np.full((len(native_curve_rows), max_length), np.nan, dtype=np.float64)
        curve_raw = np.full((len(native_curve_rows), max_length), np.nan, dtype=np.float32)
        curve_lengths = np.zeros(len(native_curve_rows), dtype=np.int64)
        curve_windows: list[str] = []
        curve_members = np.zeros((len(native_curve_rows), 2), dtype=np.int64)
        curve_scales = np.zeros(len(native_curve_rows), dtype=np.float64)
        for row in native_curve_rows:
            index = int(row["curve_id"])
            length = len(row["timestamps_s"])
            curve_lengths[index] = length
            curve_times[index, :length] = np.asarray(row["timestamps_s"], dtype=np.float64)
            curve_raw[index, :length] = np.asarray([np.nan if value is None else value for value in row["raw_distance"]], dtype=np.float32)
            curve_windows.append(str(row["window_id"]))
            curve_members[index] = np.asarray(row["member_indices"], dtype=np.int64)
            curve_scales[index] = float(row["history_scale"])
        curve_data = {"curve_id": np.arange(len(native_curve_rows), dtype=np.int64), "window_id": np.asarray(curve_windows, dtype="U64"), "member_indices": curve_members, "history_scale": curve_scales, "timestamps_s": curve_times, "raw_distance": curve_raw, "lengths": curve_lengths}
    else:
        curve_data = {"curve_id": np.empty((0,), dtype=np.int64), "window_id": np.empty((0,), dtype="U1"), "member_indices": np.empty((0, 2), dtype=np.int64), "history_scale": np.empty((0,), dtype=np.float64), "timestamps_s": np.empty((0, 0), dtype=np.float64), "raw_distance": np.empty((0, 0), dtype=np.float32), "lengths": np.empty((0,), dtype=np.int64)}
    relation_summary["native_curve_count"] = len(native_curve_rows)
    relation_summary["native_curve_storage"] = "coverage/native_pair_curves.npz (numeric arrays; no object dtype)"
    return coverage, structure, relation_summary, curve_data


def _reconstruct_model_inputs(examples: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Compare raw ``build_batch`` rows with inputs rebuilt from saved support."""

    per_arm: dict[str, float] = {}
    row_counts: dict[str, int] = {}
    for arm in ARM_NAMES:
        batch, _ = build_batch(examples, arm)
        expected_rows: list[np.ndarray] = []
        for example in examples:
            for triplet in _triplets_for_example(example):
                expected_rows.append(np.asarray(arm_inputs_for_triplet(triplet)[arm], dtype=np.float64))
        expected = np.concatenate(expected_rows, axis=0) if expected_rows else np.empty((0, 12), dtype=np.float64)
        if expected.shape != batch.inputs.shape:
            raise ValueError(f"model input shape mismatch for {arm}: {expected.shape} != {batch.inputs.shape}")
        per_arm[arm] = float(np.max(np.abs(expected - batch.inputs))) if expected.size else 0.0
        row_counts[arm] = int(expected.shape[0])
    return {"per_arm_max_abs_error": per_arm, "max_abs_error": max(per_arm.values(), default=0.0), "row_counts": row_counts}


def _average_decompositions(
    decomp_runs: Mapping[tuple[str, str], Sequence[Mapping[str, Any]]],
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    output: dict[str, dict[str, Any]] = defaultdict(dict)
    errors: list[float] = []
    for (held_out, arm), runs in sorted(decomp_runs.items()):
        by_window: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
        for run_record in runs:
            for window_id, decomposition in run_record["decomposition"].items():
                by_window[window_id].append(decomposition)
                errors.append(float(decomposition["decomposition_abs_error"]))
        for window_id, values in by_window.items():
            first = values[0]
            response = {
                "window_id": window_id,
                "held_out_source": held_out,
                "arm": arm,
                "seed_scores": [float(item["window_logit"]) for item in values],
                "mean_score": float(np.mean([item["window_logit"] for item in values])),
                "seed_std": float(np.std([item["window_logit"] for item in values])),
                "decomposition_max_abs_error": float(max(item["decomposition_abs_error"] for item in values)),
                "components": [],
            }
            for component_position, component in enumerate(first["components"]):
                component_runs = [item["components"][component_position] for item in values]
                triplets: list[dict[str, Any]] = []
                for triplet_position, triplet in enumerate(component["triplets"]):
                    triplet_runs = [item["triplets"][triplet_position] for item in component_runs]
                    triplets.append(
                        {
                            "triplet_id": int(triplet["triplet_id"]),
                            "target_slots": [int(item) for item in triplet["target_slots"]],
                            "q_mean": float(np.mean([item["q"] for item in triplet_runs])),
                            "q_seed_values": [float(item["q"]) for item in triplet_runs],
                            "triplet_weight": float(triplet["triplet_weight"]),
                            "weighted_contribution": float(np.mean([item["weighted_contribution"] for item in triplet_runs])),
                            "row_q_mean": [float(np.mean([item["row_q"][row] for item in triplet_runs])) for row in range(len(triplet["row_q"]))],
                            "row_names": list(triplet["row_names"]),
                            "local_row_contributions": [float(np.mean([item["local_row_contributions"][row] for item in triplet_runs])) for row in range(len(triplet["local_row_contributions"]))],
                        }
                    )
                response["components"].append(
                    {
                        "component_index": int(component["component_index"]),
                        "triplet_count": int(component["triplet_count"]),
                        "component_q": float(np.mean([item["component_q"] for item in component_runs])),
                        "component_weight": float(component["component_weight"]),
                        "component_contribution": float(np.mean([item["component_contribution"] for item in component_runs])),
                        "triplets": triplets,
                    }
                )
            output[window_id][arm] = response
    return dict(output), {"decomposition_abs_error": _summary(errors), "decomposition_run_count": sum(len(values) for values in decomp_runs.values())}


def _safe_slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")


def _png_bytes(rgb: np.ndarray) -> bytes:
    array = np.asarray(rgb, dtype=np.uint8)
    if array.ndim != 3 or array.shape[2] != 3:
        raise ValueError("RGB image must have shape [H,W,3]")
    height, width, _ = array.shape
    raw = b"".join(b"\x00" + array[row].tobytes() for row in range(height))
    def chunk(kind: bytes, payload: bytes) -> bytes:
        return struct.pack(">I", len(payload)) + kind + payload + struct.pack(">I", zlib.crc32(kind + payload) & 0xFFFFFFFF)
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)) + chunk(b"IDAT", zlib.compress(raw, 6)) + chunk(b"IEND", b"")


def _decode_frame(video_path: Path, frame_index: int) -> np.ndarray:
    try:
        import av
    except ImportError as exc:  # pragma: no cover - environment check is in the command
        raise RuntimeError("PyAV is required for review media") from exc
    container = av.open(str(video_path))
    try:
        stream = container.streams.video[0]
        for index, frame in enumerate(container.decode(stream)):
            if index == int(frame_index):
                return frame.to_ndarray(format="rgb24")
    finally:
        container.close()
    raise ValueError(f"frame {frame_index} not found in {video_path}")


def _encode_clip(video_path: Path, frame_indices: Sequence[int], timestamps_s: Sequence[float], output_path: Path) -> None:
    try:
        import av
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("PyAV is required for review media") from exc
    wanted = {int(item) for item in frame_indices}
    if not wanted:
        raise ValueError("clip frame list is empty")
    container = av.open(str(video_path))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output = None
    try:
        stream = container.streams.video[0]
        output = av.open(str(output_path), mode="w")
        deltas = np.diff(np.asarray(timestamps_s, dtype=np.float64))
        fps = float(1.0 / np.median(deltas)) if deltas.size and np.median(deltas) > 0 else 30.0
        fps = min(120.0, max(1.0, fps))
        out_stream = output.add_stream("mpeg4", rate=Fraction(fps).limit_denominator(1000))
        first_frame = None
        selected = 0
        for index, frame in enumerate(container.decode(stream)):
            if index not in wanted:
                continue
            rgb = frame.to_ndarray(format="rgb24")
            if first_frame is None:
                first_frame = rgb
                out_stream.width = int(rgb.shape[1])
                out_stream.height = int(rgb.shape[0])
                out_stream.pix_fmt = "yuv420p"
            video_frame = av.VideoFrame.from_ndarray(rgb, format="rgb24")
            for packet in out_stream.encode(video_frame):
                output.mux(packet)
            selected += 1
            if selected == len(wanted):
                break
        for packet in out_stream.encode():
            output.mux(packet)
        if selected != len(wanted):
            raise ValueError(f"clip requested {len(wanted)} frames but encoded {selected}")
    except Exception:
        if output_path.exists():
            output_path.unlink()
        raise
    finally:
        container.close()
        if output is not None:
            output.close()


def _svg_overlay(rgb: np.ndarray, points: Sequence[Sequence[Any]], pair_lines: Sequence[Mapping[str, Any]], output_path: Path) -> None:
    scale = min(1.0, 960.0 / max(1, rgb.shape[1]))
    height = max(1, int(round(rgb.shape[0] * scale)))
    width = max(1, int(round(rgb.shape[1] * scale)))
    small = rgb[:: max(1, int(round(1.0 / scale))), :: max(1, int(round(1.0 / scale)))]
    small = small[:height, :width]
    encoded = base64.b64encode(_png_bytes(small)).decode("ascii")
    colors = ["#22c55e", "#38bdf8", "#f59e0b", "#e879f9", "#f43f5e", "#a3e635", "#fb7185", "#2dd4bf"]
    shapes: list[str] = []
    for point in points:
        _, _, u, v, is_visible, is_geometry, component = point
        if component is None:
            color = "#facc15"
        else:
            color = colors[int(component) % len(colors)]
        radius = 4.0 if is_geometry else 3.0
        opacity = 0.95 if is_visible else 0.35
        shapes.append(f'<circle cx="{float(u) * scale:.2f}" cy="{float(v) * scale:.2f}" r="{radius}" fill="{color}" fill-opacity="{opacity}" stroke="#111827" stroke-width="1"/>')
    for line in pair_lines:
        left, right = line.get("uv", [None, None])
        if left is None or right is None:
            continue
        delta = float(line.get("delta", 0.0))
        color = "#ef4444" if delta > 0 else "#2563eb"
        shapes.append(f'<line x1="{left[0] * scale:.2f}" y1="{left[1] * scale:.2f}" x2="{right[0] * scale:.2f}" y2="{right[1] * scale:.2f}" stroke="{color}" stroke-width="2" opacity="0.65"/>')
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<title>coverage and structure overlay; red grows, blue shortens</title>
<image href="data:image/png;base64,{encoded}" width="{width}" height="{height}"/>
{''.join(shapes)}
</svg>'''
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(svg, encoding="utf-8")


def _materialize_review_media(
    output_root: Path,
    groups: Mapping[str, Mapping[str, Any]],
    coverage: Mapping[str, Mapping[str, Any]],
    structure: Mapping[str, Mapping[str, Any]],
    requested_cases: Mapping[str, str] | None = None,
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    """Create only short review clips and frame overlays; never save payloads."""

    media_by_window: dict[str, dict[str, Any]] = {}
    screenshot_rows: list[dict[str, Any]] = []
    case_groups: dict[str, str] = dict(requested_cases or {})
    if not case_groups:
        valid_groups = [gid for gid, group in groups.items() if all(coverage.get(group.get(role, ""), {}).get("support_status") == "VALID" for role in ("real", "fake"))]
        if valid_groups:
            case_groups["valid_MANIP"] = next((gid for gid in valid_groups if groups[gid].get("kind") == "MANIP"), valid_groups[0])
        ctrl_groups = [gid for gid, group in groups.items() if group.get("kind") == "CTRL"]
        if ctrl_groups:
            case_groups["CTRL"] = next((gid for gid in ctrl_groups if all(coverage.get(groups[gid].get(role, ""), {}).get("support_status") == "VALID" for role in ("real", "fake"))), ctrl_groups[0])
        invalid_windows = [wid for wid, row in sorted(coverage.items()) if row.get("support_status") != "VALID"]
        if invalid_windows:
            case_groups["no_support"] = next((gid for gid, group in groups.items() if group.get("real") in invalid_windows or group.get("fake") in invalid_windows), "")
        partial_windows = [wid for wid, row in sorted(coverage.items()) if row.get("support_status") == "VALID" and any(component.get("invalid_triplets") for component in row.get("components", []))]
        if partial_windows:
            case_groups["partial_missing"] = next((gid for gid, group in groups.items() if group.get("real") in partial_windows or group.get("fake") in partial_windows), "")
    for case, group_id in case_groups.items():
        if not group_id or group_id not in groups:
            continue
        group = groups[group_id]
        for role in ("real", "fake"):
            window_id = str(group.get(role, ""))
            if not window_id or window_id not in coverage:
                continue
            window = coverage[window_id]
            manifest = group["manifest"][role]
            source_path = Path(str(manifest["video_path"]))
            frame_indices = [int(item) for item in manifest.get("frame_indices", [])]
            timestamps_s = [float(item) for item in manifest.get("timestamps_s", [])]
            slug = f"{_safe_slug(case)}__{_safe_slug(window_id)}__{role}"
            clip_path = output_root / "review/media" / f"{slug}.mp4"
            screenshot_path = output_root / "review/screenshots" / f"{slug}.png"
            overlay_path = output_root / "review/screenshots" / f"{slug}__coverage.svg"
            try:
                _encode_clip(source_path, frame_indices, timestamps_s, clip_path)
                target_source_frame = frame_indices[len(frame_indices) // 2]
                rgb = _decode_frame(source_path, target_source_frame)
                scale_step = max(1, int(math.ceil(rgb.shape[1] / 960.0)))
                rgb_small = rgb[::scale_step, ::scale_step]
                screenshot_path.parent.mkdir(parents=True, exist_ok=True)
                screenshot_path.write_bytes(_png_bytes(rgb_small))
                target_record = next((record for record in window["frame_records"] if record["source_frame_index"] == target_source_frame), window["frame_records"][0])
                point_by_slot = {int(point[0]): [float(point[2]), float(point[3])] for point in target_record.get("uv_points", [])}
                lines: list[dict[str, Any]] = []
                for triplet in structure.get(window_id, {}).get("triplets", []):
                    for pair in triplet.get("pair_records", []):
                        if target_source_frame in triplet.get("source_frame_indices", []):
                            position = triplet["source_frame_indices"].index(target_source_frame)
                            left, right = [int(item) for item in pair["member_indices"]]
                            uv_pair = [point_by_slot[left], point_by_slot[right]] if left in point_by_slot and right in point_by_slot else None
                            delta = float(pair["normalized_distance"][position] - pair["normalized_distance"][0])
                            lines.append({"uv": uv_pair, "delta": delta})
                _svg_overlay(rgb, target_record.get("uv_points", []), lines[:256], overlay_path)
                media_by_window[window_id] = {"status": "AVAILABLE", "clip_path": str(clip_path.relative_to(output_root / "review")), "screenshot_path": str(screenshot_path.relative_to(output_root / "review")), "overlay_path": str(overlay_path.relative_to(output_root / "review")), "source_video_path": str(source_path), "display_source_frame_index": target_source_frame}
                screenshot_rows.append({"case": case, "window_id": window_id, "role": role, **media_by_window[window_id]})
            except Exception as exc:  # media is optional; retain a truthful unavailable marker
                media_by_window[window_id] = {"status": "MEDIA_UNAVAILABLE", "reason": f"{type(exc).__name__}: {exc}", "source_video_path": str(source_path)}
    return media_by_window, screenshot_rows


def _minimal_protocol(head: str, input_rows: Sequence[Mapping[str, Any]], support_rows: Sequence[Mapping[str, Any]], model_records: Mapping[tuple[str, str, int], Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "protocol_version": "v1",
        "experiment": "V7 frozen local structural-temporal joint diagnostic",
        "git_head": head,
        "source_artifact_root": str(SOURCE_ARTIFACT_ROOT),
        "pilot_artifact_root": str(PILOT_ROOT),
        "source_artifact_hashes": {
            relative: _sha256(SOURCE_ARTIFACT_ROOT / relative)
            for relative in ("frontend/window_results.json", "manifests/selected_pairs.json", "manifests/window_manifest.json")
        },
        "pilot_artifact_hashes": {
            relative: _sha256(PILOT_ROOT / relative)
            for relative in ("manifests/input_manifest.json", "manifests/window_support.json", "scores/oof_window_scores.csv", "models/fold_models.json")
        },
        "population": {"windows": len(input_rows), "support_rows": len(support_rows), "model_records": len(model_records), "all_model_seeds": list(SEEDS), "arms": list(ARM_NAMES)},
        "training_reconstruction": "non-held-out source, MANIP real/fake only, original source/class-balanced window weights; stored fold standardizer and model state; no optimizer or train_model call",
        "forward_reproduction_tolerance": {"absolute": REPRO_TOL_ABS, "relative": REPRO_TOL_REL},
        "representation": "saved local pair support -> normalized distance -> S(t)=[mean,std,p25,p75] -> signed timestamp-aware first/second differences; no extra pairs",
        "aggregation": "within triplet row mean, component mean over valid triplets, window mean over valid components; A has 3 state rows and D has 6 permutation rows",
        "missing": "invalid windows remain indexed; missing UV/XYZ are not filled; no model score is invented",
        "display": {"default_order": "source_id then MANIP/CTRL earliest anchor, independent of scores", "layers": ["coverage", "structure", "model response"], "structure_color_rule": "blue=normalized pair shortening, red=growth; display only", "player_time_note": "clip seek is approximate; source frame index and PTS table are authoritative"},
        "bootstrap": {"not_recomputed": True, "seed": DIAGNOSTIC_BOOTSTRAP_SEED, "replicates": DIAGNOSTIC_BOOTSTRAP_REPLICATES, "reason": "this is a reproduction/coverage diagnostic, not a new detector evaluation"},
        "boundaries": ["no new training", "no frontend/provider rerun", "CPU-only", "no formal detector claim", "180 model records are repeated fold/seed fits, not independent samples"],
    }


def _build_groups(window_manifest: Mapping[str, Mapping[str, Any]]) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    groups: dict[str, dict[str, Any]] = {}
    for window_id, row in window_manifest.items():
        base = str(window_id).split("::", 1)[0]
        group = groups.setdefault(base, {"group_id": base, "source_id": str(row["source_id"]), "pair_id": str(row["pair_id"]), "kind": str(row["kind"]), "label": str(row["label"]), "anchor_fraction": float(row["anchor_fraction"]), "real": None, "fake": None, "manifest": {}})
        role = str(row["role"])
        group[role] = window_id
        group["manifest"][role] = dict(row)
    ordered = sorted(groups.values(), key=lambda row: (str(row["source_id"]), 0 if row["kind"] == "MANIP" else 1, float(row["anchor_fraction"]), str(row["pair_id"])))
    for index, row in enumerate(ordered):
        row["default_order_index"] = index
    return {str(row["group_id"]): row for row in ordered}, ordered


def build_review_html(payload: Mapping[str, Any]) -> str:
    """Return an offline HTML/JS page with embedded JSON and no CDN."""

    encoded = json.dumps(payload, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
    encoded = encoded.replace("<", "\\u003c").replace("&", "\\u0026")
    return """<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>V7 local structural-temporal joint diagnostic</title>
<style>
body{font-family:system-ui,-apple-system,sans-serif;margin:0;background:#111827;color:#e5e7eb}main{max-width:1500px;margin:auto;padding:18px}h1,h2{margin:.3em 0}.controls,.notice,.panel{background:#1f2937;border:1px solid #374151;border-radius:8px;padding:12px;margin:10px 0}label{margin-right:12px}select,button,input,textarea{background:#111827;color:#e5e7eb;border:1px solid #4b5563;border-radius:4px;padding:5px}button{cursor:pointer}.sides{display:grid;grid-template-columns:1fr 1fr;gap:10px}.side{background:#0f172a;border:1px solid #334155;padding:8px;border-radius:8px}.video-wrap{position:relative;background:#000;min-height:180px}.video-wrap video{width:100%;display:block}.video-wrap canvas{position:absolute;inset:0;width:100%;height:100%;pointer-events:none}.muted{color:#9ca3af}.bad{color:#fca5a5}.good{color:#86efac}.table-wrap{overflow:auto;max-height:330px}table{border-collapse:collapse;width:100%;font-size:12px}th,td{border:1px solid #374151;padding:4px;text-align:left;vertical-align:top}.bar{height:8px;background:#374151}.bar span{display:block;height:100%;background:#f59e0b}.hidden-info .identity,.hidden-info .label,.hidden-info .score{display:none}.layers{display:flex;gap:10px;flex-wrap:wrap}.small{font-size:12px}.annotation-grid{display:grid;grid-template-columns:repeat(4,1fr);gap:6px}.annotation-grid textarea{grid-column:span 4;min-height:45px}.source-path{word-break:break-all}.case-links a{margin-right:10px;color:#93c5fd}
@media(max-width:900px){.sides{grid-template-columns:1fr}.annotation-grid{grid-template-columns:repeat(2,1fr)}.annotation-grid textarea{grid-column:span 2}}
</style></head><body><main>
<h1>V7 联合诊断审查页</h1>
<div class="notice">这是冻结 pilot 的离线人工审查工具，不是新的 detector。默认顺序按 source、MANIP/CTRL 和 anchor 固定，不按分数排序。real/fake 两侧同编号 component 只是在各自视频内的局部编号，不代表物理对应。</div>
<div class="controls"><label>source <select id="source"></select></label><label>kind <select id="kind"><option>MANIP</option><option>CTRL</option></select></label><label>window <select id="window"></select></label><button id="prev">上一窗口</button><button id="next">下一窗口</button><label><input id="showIds" type="checkbox" checked> IDs</label><label><input id="showLabels" type="checkbox" checked> labels</label><label><input id="showScores" type="checkbox" checked> scores</label></div>
<div class="controls layers"><strong>图层：</strong><label><input id="layerCoverage" type="checkbox" checked> 观测覆盖</label><label><input id="layerStructure" type="checkbox" checked> 结构变化</label><label><input id="layerModel" type="checkbox" checked> 模型响应</label><span class="small">结构变化只画实际三时刻/原生背景点；没有测量覆盖的区域保持未测量。</span></div>
<div class="controls"><button id="play">两侧播放</button><button id="pause">暂停</button><button id="step">下一源帧</button><button id="seekTarget">定位首个模型目标</button><span id="timeline" class="small"></span></div>
<div class="sides"><section class="side"><h2 id="realTitle">real</h2><div class="video-wrap"><video id="realVideo" controls preload="metadata"></video><canvas id="realCanvas"></canvas></div><p id="realMedia" class="small source-path"></p></section><section class="side"><h2 id="fakeTitle">fake</h2><div class="video-wrap"><video id="fakeVideo" controls preload="metadata"></video><canvas id="fakeCanvas"></canvas></div><p id="fakeMedia" class="small source-path"></p></section></div>
<section class="panel"><h2>覆盖与时间事实</h2><div id="coverageSummary"></div><div class="table-wrap"><table id="framesTable"></table></div></section>
<section class="panel"><h2>结构变化</h2><p class="small">红色表示相对该 pair 的历史归一化距离增长，蓝色表示缩短。三时刻外的 native 曲线只是背景，不进入模型输入；这不是真假概率。</p><label>component <select id="component"></select></label><label>triplet <select id="triplet"></select></label><label>pair <select id="pair"></select></label><div class="table-wrap"><table id="relationTable"></table></div></section>
<section class="panel"><h2>模型响应与平均聚合</h2><p class="small">q 是局部 logit 响应，不是局部概率；贡献严格按当前 component/window 平均规则计算。不同 held-out fold 的 logit 尺度不作为统一异常刻度。</p><div class="table-wrap"><table id="modelTable"></table></div></section>
<section class="panel"><h2>开发人工标记（不进入训练或筛选）</h2><div class="annotation-grid"><label>可见失真 <select id="annVisible"><option>uncertain</option><option>yes</option><option>no</option></select></label><label>开始秒 <input id="annStart" type="number" step=".001"></label><label>结束秒 <input id="annEnd" type="number" step=".001"></label><label>区域 x,y,w,h <input id="annRect" placeholder="x,y,w,h"></label><label>点 x,y <input id="annPoint" placeholder="x,y"></label><label>覆盖 <select id="annCoverage"><option>uncertain</option><option>yes</option><option>no</option></select></label><label>raw relation <select id="annRaw"><option>uncertain</option><option>yes</option><option>no</option></select></label><label>S response <select id="annState"><option>uncertain</option><option>yes</option><option>no</option></select></label><label>model response <select id="annModel"><option>uncertain</option><option>yes</option><option>no</option></select></label><textarea id="annNotes" placeholder="notes"></textarea></div><button id="saveAnn">保存标记</button><button id="exportJson">导出 JSON</button><button id="exportCsv">导出 CSV</button><label>导入 <input id="importAnn" type="file" accept="application/json"></label><div id="annStatus" class="small"></div></section>
<section class="panel"><h2>默认人工检查样本</h2><div id="caseLinks" class="case-links"></div><p class="small">播放器 seek 不是源帧级证据；请以页面表格中的源 frame index、array index 和真实 PTS 为准。</p></section>
<script id="payload" type="application/json">""" + encoded + """</script>
<script>
const DATA=JSON.parse(document.getElementById('payload').textContent);const windows=DATA.windows||[];const details=DATA.details||{};const groups=DATA.groups||{};const byId=Object.fromEntries(windows.map(x=>[x.window_id,x]));let currentGroup=null,annotations=[];let selectedComponent=null,selectedTriplet=null,selectedPair=null;
const $=id=>document.getElementById(id);function dl(name,text,type){const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([text],{type}));a.download=name;a.click();setTimeout(()=>URL.revokeObjectURL(a.href),1000)}
function groupList(){return Object.values(groups).filter(g=>g.source_id===$('source').value&&g.kind===$('kind').value).sort((a,b)=>a.default_order_index-b.default_order_index)}
function populate(){const sources=[...new Set(Object.values(groups).map(g=>g.source_id))].sort();$('source').innerHTML=sources.map(x=>`<option>${x}</option>`).join('');if(!sources.length)return;let list=groupList();if(!list.length){$('kind').value=$('kind').value==='MANIP'?'CTRL':'MANIP';list=groupList()}$('window').innerHTML=list.map(g=>`<option value="${g.group_id}">${g.label} (${g.group_id})</option>`).join('');render()}
function setGroup(){const list=groupList();if(!list.length)return;currentGroup=groups[$('window').value]||list[0];$('window').value=currentGroup.group_id;render()}
function mediaFor(wid){return (byId[wid]||{}).media||{status:'MEDIA_UNAVAILABLE'} }
function setVideo(role,wid){const v=$(role+'Video'),c=$(role+'Canvas'),m=$(role+'Media'),media=mediaFor(wid);v.pause();v.removeAttribute('src');v.load();if(media.status==='AVAILABLE'){v.src=media.clip_path;m.textContent='展示短片：'+media.clip_path+'；源：'+media.source_video_path}else{m.textContent='MEDIA_UNAVAILABLE；源：'+(media.source_video_path||'未记录')+(media.reason?'；'+media.reason:'')}v.dataset.windowId=wid;v.onloadedmetadata=()=>draw(role);v.ontimeupdate=()=>{draw(role);sync(role)};c.width=1;c.height=1}
function sync(role){const other=role==='real'?'fake':'real',a=$(role+'Video'),b=$(other+'Video');if(a.seeking||b.seeking||!a.duration||!b.duration)return;const t=Math.min(a.currentTime,b.duration);if(Math.abs(b.currentTime-t)>.08)b.currentTime=t}
function nearest(frames,t){if(!frames.length)return null;return frames.reduce((a,b)=>Math.abs(b.relative_time_s-t)<Math.abs(a.relative_time_s-t)?b:a)}
function draw(role){const v=$(role+'Video'),canvas=$(role+'Canvas'),wid=v.dataset.windowId,d=details[wid];if(!d||!v.videoWidth)return;const fr=nearest(d.coverage.frame_records,v.currentTime);if(!fr)return;canvas.width=v.clientWidth*devicePixelRatio;canvas.height=v.clientHeight*devicePixelRatio;const ctx=canvas.getContext('2d');ctx.clearRect(0,0,canvas.width,canvas.height);const sx=canvas.width/v.videoWidth,sy=canvas.height/v.videoHeight;const colors=['#22c55e','#38bdf8','#f59e0b','#e879f9','#f43f5e','#a3e635','#fb7185','#2dd4bf'];if($('layerCoverage').checked){for(const p of fr.uv_points){const [slot,track,u,vv,vis,geo,comp]=p;ctx.fillStyle=comp==null?'#facc15':colors[comp%colors.length];ctx.globalAlpha=geo?(vis?.95:.35):.55;ctx.beginPath();ctx.arc(u*sx,vv*sy,geo?3.5:2.5,0,Math.PI*2);ctx.fill();if($('showIds').checked){ctx.globalAlpha=.8;ctx.fillStyle='#fff';ctx.font='10px sans-serif';ctx.fillText(String(track),u*sx+4,vv*sy-3)}}}ctx.globalAlpha=1;if($('layerStructure').checked&&selectedPair&&d.structure){const points=Object.fromEntries(fr.uv_points.map(p=>[p[0],[p[2],p[3]]]));for(const tr of d.structure.triplets.filter(t=>selectedTriplet==null||t.triplet_id==selectedTriplet)){const pos=tr.source_frame_indices.indexOf(fr.source_frame_index);if(pos<0)continue;for(const p of tr.pair_records.filter(x=>selectedPair==null||x.pair_id.join('-')===selectedPair)){const left=p.member_indices[0],right=p.member_indices[1],uv=points[left]&&points[right]?[points[left],points[right]]:null;if(!uv)continue;const delta=p.normalized_distance[pos]-p.normalized_distance[0];ctx.strokeStyle=delta>0?'#ef4444':'#2563eb';ctx.lineWidth=2;ctx.beginPath();ctx.moveTo(uv[0][0]*sx,uv[0][1]*sy);ctx.lineTo(uv[1][0]*sx,uv[1][1]*sy);ctx.stroke()}}}}
function render(){if(!currentGroup){setGroup();return}const real=currentGroup.real,fake=currentGroup.fake;const rw=byId[real],fw=byId[fake];$('realTitle').textContent=($('showLabels').checked?'real '+(rw?.label||''):'real');$('fakeTitle').textContent=($('showLabels').checked?'fake '+(fw?.label||''):'fake');setVideo('real',real);setVideo('fake',fake);const rd=details[real],fd=details[fake];const rtime=rd?.coverage?.frame_records?.[0]?.timestamp_s,fTime=fd?.coverage?.frame_records?.[0]?.timestamp_s;$('timeline').textContent=`source=${currentGroup.source_id}; interval=${rw?.interval_start_s}–${rw?.interval_end_s}s; H boundary and actual model times are in tables`;renderCoverage(rd,fd);renderStructure(rd||{});renderModels(rd,fd)}
function renderCoverage(rd,fd){const cov=[['real',rd],['fake',fd]];$('coverageSummary').innerHTML=cov.map(([role,d])=>{if(!d)return'';const c=d.coverage;return `<b>${role}</b>: support=${c.support_status}, initialized=${c.initial_query.initialized_track_slots}, frames=${c.frame_records.length}, model frames=${c.model_frame_indices.length}, components=${c.components.length}, triplets=${c.triplet_count}, invalid=${(c.invalid_reasons||[]).map(x=>x.reason).join(',')||'none'}`}).join('　');const d=rd||fd;if(!d)return;$('framesTable').innerHTML='<tr><th>array</th><th>source frame</th><th>PTS(s)</th><th>visible</th><th>geometry</th><th>model used</th></tr>'+d.coverage.frame_records.map(f=>`<tr><td>${f.array_index}</td><td>${f.source_frame_index}</td><td>${f.timestamp_s.toFixed(6)}</td><td>${f.visible_count}/${f.uv_finite_count}</td><td>${f.geometry_valid_count}/${f.xyz_finite_count}</td><td>${f.model_used?'yes':'—'}</td></tr>`).join('')}
function renderStructure(d){selectedComponent=selectedComponent??null;const ts=d.structure?.triplets||[];const cs=[...new Set(ts.map(t=>t.component_index))];if(selectedComponent==null||!cs.includes(+selectedComponent))selectedComponent=cs[0]??null;$('component').innerHTML=cs.map(c=>`<option ${c==selectedComponent?'selected':''}>${c}</option>`).join('');const ct=ts.filter(t=>t.component_index==selectedComponent);const ids=ct.map(t=>t.triplet_id);if(selectedTriplet==null||!ids.includes(+selectedTriplet))selectedTriplet=ids[0]??null;$('triplet').innerHTML=ct.map(t=>`<option ${t.triplet_id==selectedTriplet?'selected':''}>${t.triplet_id} slots=${t.target_slots.join(',')}</option>`).join('');const tr=ct.find(t=>t.triplet_id==selectedTriplet);const pairs=tr?.pair_records||[];const pids=pairs.map(p=>p.pair_id.join('-'));if(selectedPair==null||!pids.includes(selectedPair))selectedPair=pids[0]??null;$('pair').innerHTML=pids.map(p=>`<option ${p===selectedPair?'selected':''}>${p}</option>`).join('');$('relationTable').innerHTML=tr?'<tr><th>pair</th><th>source frames</th><th>raw distance</th><th>normalized</th><th>background curve</th></tr>'+pairs.filter(p=>selectedPair==null||p.pair_id.join('-')===selectedPair).map(p=>`<tr><td>${p.pair_id.join('-')}</td><td>${p.source_frame_indices.join(', ')}</td><td>${p.raw_distance.map(x=>x.toFixed(6)).join(', ')}</td><td>${p.normalized_distance.map(x=>x.toFixed(6)).join(', ')}</td><td>native_curve_id=${p.native_curve_id}; full native curve in coverage/native_pair_curves.npz</td></tr>`).join(''):'<tr><td>无实际 triplet 支撑，结构未测量。</td></tr>'}
function renderModels(rd,fd){const table=$('modelTable');if(!$('layerModel').checked){table.style.display='none';table.innerHTML='';return}table.style.display='table';const rows=[];for(const [role,d] of [['real',rd],['fake',fd]]){if(!d)continue;const response=d.model_response||{};for(const arm of Object.keys(response)){if(!$('showScores').checked)continue;const r=response[arm];rows.push(`<tr><td>${role}</td><td>${arm}</td><td class="score">${r.mean_score.toFixed(6)}</td><td>${r.seed_scores.map(x=>x.toFixed(5)).join(', ')}</td><td>${r.components.map(c=>`c${c.component_index}: q=${c.component_q.toFixed(5)}, contribution=${c.component_contribution.toFixed(5)}; `+c.triplets.map(t=>`t${t.triplet_id} q=${t.q_mean.toFixed(5)} w=${t.triplet_weight.toFixed(3)} contrib=${t.weighted_contribution.toFixed(5)}`).join(' ')).join('<br>')}</td></tr>`)}}table.innerHTML='<tr><th>side</th><th>arm</th><th>window logit</th><th>3 seeds</th><th>component/triplet decomposition</th></tr>'+rows.join('')||'<tr><td>未评分</td></tr>'}
function currentDetails(){return details[currentGroup?.real]||details[currentGroup?.fake]||{}}
$('source').onchange=()=>{const list=groupList();$('window').innerHTML=list.map(g=>`<option value="${g.group_id}">${g.label} (${g.group_id})</option>`).join('');setGroup()};$('kind').onchange=()=>{$('source').onchange()};$('window').onchange=setGroup;$('component').onchange=e=>{selectedComponent=e.target.value;selectedTriplet=null;selectedPair=null;render()};$('triplet').onchange=e=>{selectedTriplet=e.target.value;selectedPair=null;render()};$('pair').onchange=e=>{selectedPair=e.target.value;render()};$('showIds').onchange=render;$('showLabels').onchange=render;$('showScores').onchange=render;for(const id of ['layerCoverage','layerStructure','layerModel'])$(id).onchange=()=>{render();draw('real');draw('fake')};$('prev').onclick=()=>move(-1);$('next').onclick=()=>move(1);function move(delta){const list=groupList(),i=Math.max(0,Math.min(list.length-1,list.findIndex(g=>g.group_id===currentGroup.group_id)+delta));$('window').value=list[i]?.group_id||'';setGroup()}
$('play').onclick=()=>{$('realVideo').play();$('fakeVideo').play()};$('pause').onclick=()=>{$('realVideo').pause();$('fakeVideo').pause()};$('step').onclick=()=>{for(const role of ['real','fake']){const v=$(role+'Video'),d=details[v.dataset.windowId];if(!d||!v.duration)continue;const fr=nearest(d.coverage.frame_records,v.currentTime+.034);v.currentTime=fr?fr.relative_time_s:v.currentTime+.034}};$('seekTarget').onclick=()=>{for(const role of ['real','fake']){const v=$(role+'Video'),d=details[v.dataset.windowId],tr=d?.structure?.triplets?.[0];if(tr&&v.duration){const t=tr.timestamps_s[1]-d.coverage.frame_records[0].timestamp_s;v.currentTime=Math.max(0,Math.min(v.duration,t))}}};
$('saveAnn').onclick=()=>{const d=currentDetails();annotations.push({window_id:currentGroup?.group_id,real_window_id:currentGroup?.real,fake_window_id:currentGroup?.fake,visible_distortion:$('annVisible').value,time_interval_s:[Number($('annStart').value)||null,Number($('annEnd').value)||null],rectangle:$('annRect').value||null,point:$('annPoint').value||null,coverage:$('annCoverage').value,raw_relation:$('annRaw').value,state_response:$('annState').value,model_response:$('annModel').value,notes:$('annNotes').value||'',created_at:new Date().toISOString()});$('annStatus').textContent=`已保存 ${annotations.length} 条（仅本地浏览器内存）`};$('exportJson').onclick=()=>dl('v7_annotations.json',JSON.stringify(annotations,null,2),'application/json');$('exportCsv').onclick=()=>{const keys=['window_id','visible_distortion','time_interval_s','rectangle','point','coverage','raw_relation','state_response','model_response','notes'];const esc=x=>'"'+String(x??'').replaceAll('"','""')+'"';dl('v7_annotations.csv',[keys.join(','),...annotations.map(a=>keys.map(k=>esc(Array.isArray(a[k])?a[k].join(';'):a[k])).join(','))].join('\n'),'text/csv')};$('importAnn').onchange=async e=>{const text=await e.target.files[0].text();try{const x=JSON.parse(text);if(!Array.isArray(x))throw Error('JSON must be array');annotations=x;$('annStatus').textContent=`已导入 ${annotations.length} 条`}catch(err){$('annStatus').textContent='导入失败：'+err}};
function initCases(){const cases=DATA.sample_cases||{};$('caseLinks').innerHTML=Object.entries(cases).map(([name,gid])=>`<a href="#" data-g="${gid}">${name} → ${gid}</a>`).join('');document.querySelectorAll('#caseLinks a').forEach(a=>a.onclick=e=>{e.preventDefault();const g=groups[a.dataset.g];$('source').value=g.source_id;$('kind').value=g.kind;$('source').onchange();$('window').value=g.group_id;setGroup()})}
populate();initCases();
</script></main></body></html>"""


def _build_index(
    input_rows: Sequence[Mapping[str, Any]],
    coverage: Mapping[str, Mapping[str, Any]],
    model_response: Mapping[str, Mapping[str, Any]],
    window_manifest: Mapping[str, Mapping[str, Any]],
    media_by_window: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in input_rows:
        wid = str(row["window_id"])
        manifest = window_manifest[wid]
        cov = coverage[wid]
        rows.append(
            {
                "window_id": wid,
                "group_id": wid.split("::", 1)[0],
                "source_id": str(row["source_id"]),
                "pair_id": str(row["pair_id"]),
                "role": str(row["role"]),
                "kind": str(row["kind"]),
                "label": str(manifest.get("label", "")),
                "anchor_fraction": float(manifest["anchor_fraction"]),
                "interval_start_s": float(manifest["interval_start_s"]),
                "interval_end_s": float(manifest["interval_end_s"]),
                "support_status": str(cov["support_status"]),
                "invalid_reasons": cov["invalid_reasons"],
                "model_available": wid in model_response,
                "media": dict(media_by_window.get(wid, {"status": "MEDIA_UNAVAILABLE", "source_video_path": str(manifest["video_path"])})),
                "source_video_path": str(manifest["video_path"]),
                "default_order_key": [str(row["source_id"]), 0 if row["kind"] == "MANIP" else 1, float(row["anchor_fraction"]), str(row["role"])],
            }
        )
    return sorted(rows, key=lambda row: tuple(row["default_order_key"]))


def run(
    *,
    source_root: Path = SOURCE_ARTIFACT_ROOT,
    pilot_root: Path = PILOT_ROOT,
    output_root: Path = DIAGNOSTIC_OUTPUT_ROOT,
    make_media: bool = True,
) -> dict[str, Any]:
    """Run the frozen, CPU-only diagnostic and write its independent artifact."""

    del source_root, pilot_root  # paths are fixed by the frozen pilot imports
    started = time.perf_counter()
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"diagnostic output already exists: {output_root}")
    examples, support_rows, details = load_frozen_examples(SOURCE_ARTIFACT_ROOT)
    input_rows = details["input_rows"]
    all_source_ids = list(details["population"]["all_source_ids"])
    records = _load_model_records()
    oof = _load_oof(PILOT_ROOT / "scores/oof_window_scores.csv")
    window_results = _load_window_results()
    window_manifest = _load_window_manifest()
    selected_pairs = _load_selected_pairs()
    head = _git_head()
    protocol = _minimal_protocol(head, input_rows, support_rows, records)
    output_root.mkdir(parents=True, exist_ok=True)
    _write_json(output_root / "protocol.json", protocol)
    diagnostics, diagnostic_meta, loss_curves, decomp_runs = _model_diagnostics(examples, all_source_ids, records, oof)
    _write_csv(output_root / "evaluation/model_reproduction.csv", diagnostics)
    _write_json(output_root / "evaluation/loss_curves.json", loss_curves)
    _write_json(output_root / "evaluation/oof_reproduction_summary.json", diagnostic_meta["oof_reproduction"])
    coverage, structure, relation_summary, native_curve_data = _coverage_and_structure(input_rows, support_rows, window_results, window_manifest)
    relation_summary["model_input_reconstruction"] = _reconstruct_model_inputs(examples)
    _write_json(output_root / "coverage/coverage_windows.json", coverage)
    _write_json(output_root / "coverage/structure_records.json", structure)
    np.savez_compressed(output_root / "coverage/native_pair_curves.npz", **native_curve_data)
    _write_json(output_root / "evaluation/relation_reconstruction_summary.json", relation_summary)
    model_response, decomposition_summary = _average_decompositions(decomp_runs)
    _write_json(output_root / "evaluation/contribution_decomposition_summary.json", decomposition_summary)
    _write_json(output_root / "evaluation/contribution_responses.json", model_response)
    groups, ordered_groups = _build_groups(window_manifest)
    media_by_window: dict[str, dict[str, Any]] = {}
    screenshot_rows: list[dict[str, Any]] = []
    if make_media:
        media_by_window, screenshot_rows = _materialize_review_media(output_root, groups, coverage, structure)
    for window_id, media in media_by_window.items():
        if window_id in coverage:
            coverage[window_id]["media"] = media
    index_rows = _build_index(input_rows, coverage, model_response, window_manifest, media_by_window)
    for row in index_rows:
        groups[row["group_id"]]["side_index"] = groups[row["group_id"]].get("side_index", {}) | {row["role"]: row["window_id"]}
    details_payload: dict[str, dict[str, Any]] = {}
    for row in index_rows:
        wid = row["window_id"]
        details_payload[wid] = {"coverage": coverage[wid], "structure": structure[wid], "model_response": model_response.get(wid)}
    sample_cases: dict[str, str] = {}
    valid_manip = [group for group in ordered_groups if group["kind"] == "MANIP" and group.get("real") and group.get("fake") and coverage[group["real"]]["support_status"] == "VALID" and coverage[group["fake"]]["support_status"] == "VALID"]
    valid_ctrl = [group for group in ordered_groups if group["kind"] == "CTRL" and group.get("real") and group.get("fake") and coverage[group["real"]]["support_status"] == "VALID" and coverage[group["fake"]]["support_status"] == "VALID"]
    no_support = [group for group in ordered_groups if (group.get("real") and coverage[group["real"]]["support_status"] != "VALID") or (group.get("fake") and coverage[group["fake"]]["support_status"] != "VALID")]
    partial = [
        group
        for group in ordered_groups
        if any(
            wid
            and coverage[wid]["support_status"] == "VALID"
            and any(component.get("invalid_triplets") for component in coverage[wid].get("components", []))
            for wid in (group.get("real"), group.get("fake"))
        )
    ]
    if valid_manip:
        sample_cases["valid_MANIP"] = valid_manip[0]["group_id"]
    if valid_ctrl:
        sample_cases["CTRL"] = valid_ctrl[0]["group_id"]
    if no_support:
        sample_cases["no_support"] = no_support[0]["group_id"]
    if partial:
        sample_cases["partial_missing"] = partial[0]["group_id"]
    _write_json(output_root / "review/screenshot_index.json", screenshot_rows)
    _write_json(output_root / "review/index_data.json", {"windows": index_rows, "groups": groups, "sample_cases": sample_cases})
    page_payload = {
        "protocol": protocol,
        "windows": index_rows,
        "groups": groups,
        "details": details_payload,
        "sample_cases": sample_cases,
        "annotation_schema": {"status": "development-only; not training or population selection", "fields": ["visible_distortion", "time_interval_s", "rectangle", "point", "coverage", "raw_relation", "state_response", "model_response", "notes"]},
        "selected_pairs_count": len(selected_pairs),
    }
    _write_json(output_root / "review/data.json", page_payload)
    review_path = output_root / "review/index.html"
    review_path.parent.mkdir(parents=True, exist_ok=True)
    review_path.write_text(build_review_html(page_payload), encoding="utf-8")
    summary = {
        "status": "JOINT_DIAGNOSTIC_COMPLETE",
        "git_head": head,
        "population": protocol["population"],
        "model_reproduction": diagnostic_meta["oof_reproduction"],
        "relation_reconstruction": relation_summary,
        "contribution_decomposition": decomposition_summary,
        "coverage": {
            "window_count": len(coverage),
            "valid_support_windows": sum(row["support_status"] == "VALID" for row in coverage.values()),
            "invalid_support_windows": sum(row["support_status"] != "VALID" for row in coverage.values()),
            "invalid_reason_counts": {reason: sum(1 for row in coverage.values() for item in row.get("invalid_reasons", []) if item.get("reason") == reason) for reason in sorted({item.get("reason") for row in coverage.values() for item in row.get("invalid_reasons", [])})},
        },
        "review": {"html": str(review_path), "index_entries": len(index_rows), "media_entries": len(media_by_window), "screenshots": len(screenshot_rows), "browser_visual_verification": False},
        "elapsed_s": time.perf_counter() - started,
        "boundaries": ["read-only frozen model forward passes", "no optimizer.step", "no frontend/provider rerun", "no GPU", "not a formal detector result", "180 fold/seed models are not independent samples"],
    }
    _write_json(output_root / "evaluation/summary.json", summary)
    _write_json(output_root / "run_summary.json", {"status": summary["status"], "artifact_root": str(output_root), "summary": "evaluation/summary.json", "review_html": str(review_path), "window_count": len(coverage), "model_record_count": len(diagnostics), "media_entry_count": len(media_by_window)})
    return summary


def attach_media(output_root: Path = DIAGNOSTIC_OUTPUT_ROOT) -> dict[str, Any]:
    """Attach deterministic short review clips to an existing numeric audit."""

    if not output_root.exists():
        raise FileNotFoundError(output_root)
    coverage = json.loads((output_root / "coverage/coverage_windows.json").read_text(encoding="utf-8"))
    structure = json.loads((output_root / "coverage/structure_records.json").read_text(encoding="utf-8"))
    index_data = json.loads((output_root / "review/index_data.json").read_text(encoding="utf-8"))
    response = json.loads((output_root / "evaluation/contribution_responses.json").read_text(encoding="utf-8"))
    input_rows = json.loads((PILOT_ROOT / "manifests/input_manifest.json").read_text(encoding="utf-8"))["rows"]
    window_manifest = _load_window_manifest()
    groups = index_data["groups"]
    ordered_groups = sorted(groups.values(), key=lambda row: int(row.get("default_order_index", 0)))
    partial_groups = [
        group
        for group in ordered_groups
        if any(
            wid
            and coverage[wid]["support_status"] == "VALID"
            and any(component.get("invalid_triplets") for component in coverage[wid].get("components", []))
            for wid in (group.get("real"), group.get("fake"))
        )
    ]
    sample_cases = dict(index_data.get("sample_cases", {}))
    if partial_groups:
        sample_cases["partial_missing"] = partial_groups[0]["group_id"]
    media_by_window, screenshot_rows = _materialize_review_media(output_root, groups, coverage, structure, sample_cases)
    for window_id, media in media_by_window.items():
        coverage[window_id]["media"] = media
    index_rows = _build_index(input_rows, coverage, response, window_manifest, media_by_window)
    details_payload = {row["window_id"]: {"coverage": coverage[row["window_id"]], "structure": structure[row["window_id"]], "model_response": response.get(row["window_id"])} for row in index_rows}
    protocol = json.loads((output_root / "protocol.json").read_text(encoding="utf-8"))
    page_payload = {
        "protocol": protocol,
        "windows": index_rows,
        "groups": groups,
        "details": details_payload,
        "sample_cases": sample_cases,
        "annotation_schema": {"status": "development-only; not training or population selection", "fields": ["visible_distortion", "time_interval_s", "rectangle", "point", "coverage", "raw_relation", "state_response", "model_response", "notes"]},
        "selected_pairs_count": 16,
    }
    _write_json(output_root / "coverage/coverage_windows.json", coverage)
    _write_json(output_root / "review/screenshot_index.json", screenshot_rows)
    _write_json(output_root / "review/index_data.json", {"windows": index_rows, "groups": groups, "sample_cases": sample_cases})
    _write_json(output_root / "review/data.json", page_payload)
    (output_root / "review/index.html").write_text(build_review_html(page_payload), encoding="utf-8")
    summary_path = output_root / "evaluation/summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["review"] = {"html": str(output_root / "review/index.html"), "index_entries": len(index_rows), "media_entries": len(media_by_window), "screenshots": len(screenshot_rows), "browser_visual_verification": False}
    _write_json(summary_path, summary)
    _write_json(output_root / "run_summary.json", {"status": summary["status"], "artifact_root": str(output_root), "summary": "evaluation/summary.json", "review_html": str(output_root / "review/index.html"), "window_count": len(coverage), "model_record_count": summary.get("population", {}).get("model_records"), "media_entry_count": len(media_by_window)})
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DIAGNOSTIC_OUTPUT_ROOT)
    parser.add_argument("--no-media", action="store_true", help="skip optional short clips and screenshots")
    parser.add_argument("--attach-media", action="store_true", help="attach review media to an existing numeric diagnostic")
    args = parser.parse_args()
    result = attach_media(args.output_root) if args.attach_media else run(output_root=args.output_root, make_media=not args.no_media)
    print(json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
