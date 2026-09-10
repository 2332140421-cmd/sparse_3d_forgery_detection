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
    COMPONENT_CONFIG,
    MAX_TARGET_ERROR_S,
    TARGET_OFFSETS_S,
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


def _annotation_inventory_rows() -> list[dict[str, Any]]:
    """Build a small, source-grounded inventory of annotation evidence.

    This deliberately inventories only the files already used by the V7
    materialisation.  It does not claim that a local absence proves a dataset
    wide absence of spatial annotations.
    """

    dataset_root = DATA_ROOT / "datasets/v7_core_candidates/activityforensics_charades_v1"
    mapping_path = dataset_root / "paired/manifests/activityforensics_source_mapping.json"
    rows: list[dict[str, Any]] = []

    def add(
        path: Path,
        source_version: str,
        identity: str,
        level: str,
        field: str,
        time_convention: str,
        used: str,
        evaluable: str,
        questions: str,
        *,
        evidence_status: str = "LOCAL_FILE",
        notes: str = "",
    ) -> None:
        rows.append(
            {
                "annotation_file_path": str(path),
                "sha256": _sha256(path) if path.is_file() else "",
                "data_source_version": source_version,
                "source_video_identity": identity,
                "annotation_level": level,
                "field_name": field,
                "time_unit_or_frame_convention": time_convention,
                "pipeline_used": used,
                "usable_for_evaluation": evaluable,
                "evidence_status": evidence_status,
                "unverified_questions": questions,
                "notes": notes,
            }
        )

    train_csv = dataset_root / "source/activityforensics/metadata/train.csv"
    test_csv = dataset_root / "source/activityforensics/metadata/test.csv"
    annot_zip = dataset_root / "source/activityforensics/raw/annot.zip"
    add(train_csv, "ActivityForensics HF revision a34d4b7b04b0f3f3e26ba900adc367218667c581", "all train video/file_name rows", "video", "file_name", "filename encodes decimal seconds; exact frame convention not encoded", "yes (lineage/mapping)", "yes for split/file lineage; not spatial GT", "whether an external release contains additional spatial annotations", notes="local metadata snapshot")
    add(test_csv, "ActivityForensics HF revision a34d4b7b04b0f3f3e26ba900adc367218667c581", "all test video/file_name rows", "video", "file_name", "filename encodes decimal seconds; exact frame convention not encoded", "yes (lineage/mapping)", "yes for split/file lineage; not spatial GT", "whether an external release contains additional spatial annotations", notes="local metadata snapshot")
    add(annot_zip, "ActivityForensics official annotation archive (local snapshot)", "train/test videos grouped by generator", "time", "duration, manipulation segments", "seconds; lines contain duration and start=end segments", "yes (mapping and MANIP windows)", "yes for temporal-window evaluation", "whether segment boundaries are inclusive/exclusive at decoded-frame level; no bbox/mask field in these files", notes="zip central directory read; not extracted")
    add(mapping_path, "local paired mapping derived from the ActivityForensics snapshot", "16 selected source IDs and paired real/fake videos", "video+time", "source_id, generator, operation, manipulation_intervals, manipulation_segments", "intervals in seconds; selected windows carry source frame indices and PTS", "yes", "yes for window labels and source-disjoint grouping", "real/fake same-source physical correspondence is an engineering pairing, not pixel identity", notes="selected-pair lineage")

    for relative, identity, level, field, convention, used, evaluable, questions, notes in (
        (
            "review/review_manifest.json",
            "30 manually selected development review pairs",
            "spatial-review template",
            "human review fields (currently blank)",
            "frame/PTS fields are to be filled by reviewer",
            "no (template only)",
            "no until human entries are made and audited",
            "which local human observations will be accepted as development evidence",
            "not official spatial ground truth",
        ),
        ):
        path = dataset_root / relative
        add(path, "V7 local derived artifact", identity, level, field, convention, used, evaluable, questions, evidence_status="LOCAL_FILE" if path.is_file() else "LOCAL_NOT_FOUND", notes=notes)

    window_manifest_path = DATA_ROOT / "derived/v7_activityforensics_paired_second_order_pilot_v1/manifests/window_manifest.json"
    add(window_manifest_path, "V7 local derived artifact", "16 selected source pairs", "time", "kind, label, interval_start_s, interval_end_s, frame_indices, timestamps_s", "source frame indices plus decoder PTS seconds", "yes (MANIP/CTRL windows)", "yes for time-window evaluation; not spatial GT", "whether a human temporal segment should use a different boundary convention", evidence_status="LOCAL_FILE" if window_manifest_path.is_file() else "LOCAL_NOT_FOUND", notes="selected-window manifest")

    add(
        dataset_root / "source/activityforensics/raw/annot.zip::spatial_search",
        "ActivityForensics local snapshot",
        "all locally materialized ActivityForensics videos",
        "spatial",
        "bbox, mask, polygon, edit-region files",
        "not found locally; no frame convention to record",
        "no",
        "no",
        "an official spatial annotation release may exist outside this snapshot; local absence is not dataset-wide proof",
        evidence_status="LOCAL_NOT_FOUND",
        notes="bounded search of the current dataset material; annot.zip is temporal-only",
    )
    add(
        dataset_root / "source/activityforensics/source_info/dataset_card_README.md",
        "ActivityForensics documentation snapshot",
        "ActivityForensics dataset",
        "documentation",
        "annotation/repository/license description",
        "not applicable",
        "yes (provenance only)",
        "no (not a ground-truth file)",
        "documentation does not establish a per-frame spatial GT mapping for this snapshot",
        evidence_status="OFFICIAL_DOCUMENTATION",
        notes="official links and research-use statement preserved locally",
    )
    return rows


def _write_annotation_inventory(output_root: Path) -> Path:
    path = output_root / "annotation_inventory.csv"
    rows = _annotation_inventory_rows()
    _write_csv(path, rows)
    _write_csv(output_root / "review/annotation_inventory.csv", rows)
    return path


def _write_measurement_contract(output_root: Path, coverage: Mapping[str, Mapping[str, Any]], structure: Mapping[str, Mapping[str, Any]]) -> Path:
    path = output_root / "measurement_contract.json"
    value = _measurement_contract(coverage, structure)
    _write_json(path, value)
    _write_json(output_root / "review/measurement_contract.json", value)
    return path


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


def _decode_frames(video_path: Path, frame_indices: Sequence[int]) -> dict[int, np.ndarray]:
    try:
        import av
    except ImportError as exc:  # pragma: no cover - environment check is in the command
        raise RuntimeError("PyAV is required for review media") from exc
    wanted = {int(item) for item in frame_indices}
    if not wanted:
        return {}
    decoded: dict[int, np.ndarray] = {}
    container = av.open(str(video_path))
    try:
        stream = container.streams.video[0]
        for index, frame in enumerate(container.decode(stream)):
            if index in wanted:
                decoded[index] = frame.to_ndarray(format="rgb24")
                if len(decoded) == len(wanted):
                    break
    finally:
        container.close()
    missing = sorted(wanted.difference(decoded))
    if missing:
        raise ValueError(f"frame(s) {missing} not found in {video_path}")
    return decoded


def _decode_frame(video_path: Path, frame_index: int) -> np.ndarray:
    return _decode_frames(video_path, [frame_index])[int(frame_index)]


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


def _default_review_cases(
    groups: Mapping[str, Mapping[str, Any]],
    coverage: Mapping[str, Mapping[str, Any]],
) -> dict[str, str]:
    """Select earliest MANIP and CTRL group for every source, score-blind."""

    ordered = sorted(groups.values(), key=lambda row: int(row.get("default_order_index", 0)))
    cases: dict[str, str] = {}
    for source_id in sorted({str(group.get("source_id")) for group in ordered}):
        for kind in ("MANIP", "CTRL"):
            candidates = [
                group
                for group in ordered
                if str(group.get("source_id")) == source_id
                and str(group.get("kind")) == kind
                and all(group.get(role) in coverage for role in ("real", "fake"))
            ]
            if candidates:
                cases[f"source_{_safe_slug(source_id)}_{kind}"] = str(candidates[0]["group_id"])
    return cases


def _measurement_contract(
    coverage: Mapping[str, Mapping[str, Any]],
    structure: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Describe what the persisted frontend and model actually measured."""

    frame_counts = [len(row.get("frame_records", [])) for row in coverage.values()]
    dimensions = sorted({(int(row["frame_records"][0]["height"]), int(row["frame_records"][0]["width"])) for row in coverage.values() if row.get("frame_records")})
    model_frame_counts = [len(row.get("model_frame_indices", [])) for row in coverage.values()]
    history_counts = sorted({len(row.get("history_frame_indices", [])) for row in coverage.values()})
    triplets = [triplet for record in structure.values() for triplet in record.get("triplets", [])]
    first = triplets[0] if triplets else None
    actual_frames = sorted({int(frame) for row in coverage.values() for frame in row.get("model_frame_indices", [])})
    processed_frames = sorted({int(frame) for row in coverage.values() for frame in row.get("frame_records", []) for frame in [frame["source_frame_index"]]})
    return {
        "contract_version": "v1",
        "query": {
            "requested_tracks": sorted({int(row["initial_query"].get("requested_tracks")) for row in coverage.values() if row["initial_query"].get("requested_tracks") is not None}),
            "initialized_track_slots": sorted({int(row["initial_query"]["initialized_track_slots"]) for row in coverage.values()}),
            "initialization": "query is made at the persisted window start; track_ids are then carried through the window",
            "new_points_join_mid_window": False,
            "evidence": "initial_query.track_ids_persisted and constant initialized_track_slots in coverage_windows.json",
        },
        "frame_layers": {
            "video_window_frame_indices": "window manifest frame_indices are the requested source decode sequence",
            "frontend_processed": {"count_range": [min(frame_counts, default=0), max(frame_counts, default=0)], "unique_source_frame_count": len(processed_frames), "source": "coverage.frame_records"},
            "model_actual_frames": {"count_range": [min(model_frame_counts, default=0), max(model_frame_counts, default=0)], "unique_source_frame_count": len(actual_frames), "source": "coverage.model_frame_indices and actual_model_time_records"},
            "distinction": "a decoded/frontend frame is not asserted to be a classifier input unless it appears in model_frame_indices or a triplet's actual_model_time_records",
        },
        "geometry": {
            "source_dimensions_hw": dimensions,
            "tracking_coordinate_system": "persisted ParticleSequence UV in original decoded-video pixel coordinates",
            "display_coordinate_system": "browser canvas scales original UV to the video element; no coordinate rewrite",
            "visibility": "visibility and finite UV in the persisted ParticleSequence coverage records",
            "geometry_validity": "ParticleSequence.geometry_validity with finite XYZ; invalid observations remain absent from current coverage",
        },
        "component_formation": {
            "history_boundary": "timestamps_s < window_start_s + 0.5 s",
            "history_frame_count_range": [history_counts[0], history_counts[-1]] if history_counts else [0, 0],
            "component_rule": "motion_coherent_components on history XYZ/geometry_validity using the frozen ComponentConfig",
            "component_config": {"max_initial_distance": COMPONENT_CONFIG.max_initial_distance, "max_relative_change": COMPONENT_CONFIG.max_relative_change, "minimum_size": COMPONENT_CONFIG.minimum_size, "minimum_overlap": COMPONENT_CONFIG.minimum_overlap},
            "validity": "a triplet retains only members geometry-valid and finite at all three matched target frames; no interpolation or last-visible fill",
        },
        "target_slots": {
            "target_offsets_s": list(TARGET_OFFSETS_S),
            "selection": "evaluation timestamps at or after the 0.5 s boundary, one unused nearest frame per target",
            "max_match_error_s": MAX_TARGET_ERROR_S,
            "matched_records_source": "coverage.actual_model_time_records and target_matches",
            "example": first.get("timestamps_s") if first else [],
        },
        "triplet_and_pair": {
            "common_members": "same track slot must be geometry-valid with finite XYZ at all three matched frames",
            "minimum_common_members": 3,
            "pair_rule": "all unordered combinations of the common members",
            "history_scale": "median of finite pair distances across the history frames and component members; one scale per component/triplet",
        },
        "representation": {
            "normalized_pair": "d_ij(t) = ||X_j(t)-X_i(t)|| / history_scale",
            "S": "S(t) = [mean(d_ij), std(d_ij), percentile25(d_ij), percentile75(d_ij)] over the triplet common-member pairs",
            "v_minus": "(S(t1)-S(t0))/(t1-t0)",
            "v_plus": "(S(t2)-S(t1))/(t2-t1)",
            "a": "2*(v_plus-v_minus)/((t1-t0)+(t2-t1)); signed timestamp-aware finite difference, not physical acceleration",
            "missing": "no pair/triplet line is joined across a missing target or invalid observation",
        },
        "aggregation": {
            "triplet": "model arm rows are evaluated per triplet",
            "component": "mean over valid triplets in the component",
            "window": "mean over valid components",
            "local_contribution": "component and triplet q/weight/contribution are the saved decomposition of the existing window logit",
        },
    }


def _materialize_review_media(
    output_root: Path,
    groups: Mapping[str, Mapping[str, Any]],
    coverage: Mapping[str, Mapping[str, Any]],
    structure: Mapping[str, Mapping[str, Any]],
    requested_cases: Mapping[str, str] | None = None,
    requested_window_ids: set[str] | None = None,
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    """Create only short review clips and frame overlays; never save payloads."""

    media_by_window: dict[str, dict[str, Any]] = {}
    screenshot_rows: list[dict[str, Any]] = []
    case_groups: dict[str, str] = dict(requested_cases or {})
    seen_windows: set[str] = set()
    for case, group_id in case_groups.items():
        if not group_id or group_id not in groups:
            continue
        group = groups[group_id]
        roles = ("real", "fake")
        if requested_window_ids is not None:
            roles = tuple(role for role in roles if str(group.get(role, "")) in requested_window_ids)
        for role in roles:
            window_id = str(group.get(role, ""))
            if not window_id or window_id not in coverage or window_id in seen_windows:
                continue
            seen_windows.add(window_id)
            window = coverage[window_id]
            manifest = group["manifest"][role]
            source_path = Path(str(manifest["video_path"]))
            frame_indices = [int(item) for item in manifest.get("frame_indices", [])]
            timestamps_s = [float(item) for item in manifest.get("timestamps_s", [])]
            slug = f"{_safe_slug(case)}__{_safe_slug(window_id)}__{role}"
            clip_path = output_root / "review/media" / f"{slug}.mp4"
            screenshot_path = output_root / "review/screenshots" / f"{slug}.png"
            overlay_path = output_root / "review/screenshots" / f"{slug}__coverage.svg"
            model_frame_indices = [int(item) for item in window.get("model_frame_indices", [])]
            try:
                if not source_path.is_file():
                    media_by_window[window_id] = {"status": "SOURCE_MISSING", "source_video_path": str(source_path), "case": case}
                    screenshot_rows.append({"case": case, "window_id": window_id, "role": role, **media_by_window[window_id]})
                    continue
                _encode_clip(source_path, frame_indices, timestamps_s, clip_path)
                target_source_frame = frame_indices[len(frame_indices) // 2]
                decoded = _decode_frames(source_path, sorted(set(frame_indices + model_frame_indices)))
                rgb = decoded[target_source_frame]
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
                exact_screenshots: list[dict[str, Any]] = []
                for source_frame in model_frame_indices:
                    frame_rgb = decoded[source_frame]
                    frame_record = next((item for item in window["frame_records"] if int(item["source_frame_index"]) == source_frame), None)
                    if frame_record is None:
                        continue
                    exact_path = output_root / "review/screenshots" / f"{slug}__model_frame_{source_frame}.png"
                    exact_step = max(1, int(math.ceil(frame_rgb.shape[1] / 960.0)))
                    exact_path.write_bytes(_png_bytes(frame_rgb[::exact_step, ::exact_step]))
                    exact_screenshots.append(
                        {
                            "path": str(exact_path.relative_to(output_root / "review")),
                            "source_frame_index": source_frame,
                            "array_index": int(frame_record["array_index"]),
                            "timestamp_s": float(frame_record["timestamp_s"]),
                        }
                    )
                media_by_window[window_id] = {"status": "AVAILABLE", "clip_path": str(clip_path.relative_to(output_root / "review")), "screenshot_path": str(screenshot_path.relative_to(output_root / "review")), "overlay_path": str(overlay_path.relative_to(output_root / "review")), "source_video_path": str(source_path), "display_source_frame_index": target_source_frame, "display_timestamp_s": float(target_record["timestamp_s"]), "model_frame_screenshots": exact_screenshots, "case": case}
                screenshot_rows.append({"case": case, "window_id": window_id, "role": role, **media_by_window[window_id]})
            except Exception as exc:  # media is optional; retain a truthful unavailable marker
                media_by_window[window_id] = {"status": "DECODE_FAILED", "reason": f"{type(exc).__name__}: {exc}", "source_video_path": str(source_path), "case": case}
                screenshot_rows.append({"case": case, "window_id": window_id, "role": role, **media_by_window[window_id]})
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
body{font-family:system-ui,-apple-system,sans-serif;margin:0;background:#111827;color:#e5e7eb}main{max-width:1500px;margin:auto;padding:18px}h1,h2{margin:.3em 0}.controls,.notice,.panel{background:#1f2937;border:1px solid #374151;border-radius:8px;padding:12px;margin:10px 0}label{margin-right:12px}select,button,input,textarea{background:#111827;color:#e5e7eb;border:1px solid #4b5563;border-radius:4px;padding:5px}button{cursor:pointer}.sides{display:grid;grid-template-columns:1fr 1fr;gap:10px}.side{background:#0f172a;border:1px solid #334155;padding:8px;border-radius:8px}.video-wrap{position:relative;background:#000;min-height:180px}.video-wrap video{width:100%;display:block}.video-wrap canvas{position:absolute;inset:0;width:100%;height:100%;pointer-events:auto}.muted{color:#9ca3af}.bad{color:#fca5a5}.good{color:#86efac}.table-wrap{overflow:auto;max-height:330px}table{border-collapse:collapse;width:100%;font-size:12px}th,td{border:1px solid #374151;padding:4px;text-align:left;vertical-align:top}.bar{height:8px;background:#374151}.bar span{display:block;height:100%;background:#f59e0b}.hidden-info .identity,.hidden-info .label,.hidden-info .score{display:none}.layers{display:flex;gap:10px;flex-wrap:wrap}.small{font-size:12px}.annotation-grid{display:grid;grid-template-columns:repeat(4,1fr);gap:6px}.annotation-grid textarea{grid-column:span 4;min-height:45px}.source-path{word-break:break-all}.case-links a{margin-right:10px;color:#93c5fd}
@media(max-width:900px){.sides{grid-template-columns:1fr}.annotation-grid{grid-template-columns:repeat(2,1fr)}.annotation-grid textarea{grid-column:span 2}}
</style></head><body><main>
<h1>V7 联合诊断审查页</h1>
<div class="notice">这是冻结 pilot 的离线人工审查工具，不是新的 detector。默认顺序按 source、MANIP/CTRL 和 anchor 固定，不按分数排序。real/fake 两侧同编号 component 只是在各自视频内的局部编号，不代表物理对应。</div>
<div class="controls"><label>source <select id="source"></select></label><label>kind <select id="kind"><option>MANIP</option><option>CTRL</option></select></label><label>window <select id="window"></select></label><button id="prev">上一窗口</button><button id="next">下一窗口</button><label><input id="showIds" type="checkbox" checked> IDs</label><label><input id="showLabels" type="checkbox" checked> labels</label><label><input id="showScores" type="checkbox" checked> scores</label></div>
<div class="controls layers"><strong>图层：</strong><label><input id="layerTracks" type="checkbox" checked> 当前可定位跟踪点</label><label><input id="layerGeometry" type="checkbox" checked> geometry-valid 点</label><label><input id="layerTriplet" type="checkbox" checked> 所选 triplet 共同成员</label><label><input id="layerPair" type="checkbox" checked> 所选 pair</label><label><input id="layerModel" type="checkbox" checked> component 局部模型响应</label><input id="layerCoverage" type="checkbox" checked hidden><input id="layerStructure" type="checkbox" checked hidden><span class="small">无效/不可见点不作为当前有效观测；没有测量覆盖的区域保持未测量。</span></div>
<div class="controls"><button id="play">两侧播放</button><button id="pause">暂停</button><button id="step">下一源帧</button><button id="seekTarget">定位首个模型目标</button><span id="timeline" class="small"></span></div>
<div class="sides"><section class="side"><h2 id="realTitle">real</h2><div class="video-wrap"><video id="realVideo" controls preload="metadata"></video><canvas id="realCanvas"></canvas></div><p id="realMedia" class="small source-path"></p></section><section class="side"><h2 id="fakeTitle">fake</h2><div class="video-wrap"><video id="fakeVideo" controls preload="metadata"></video><canvas id="fakeCanvas"></canvas></div><p id="fakeMedia" class="small source-path"></p></section></div>
<section class="panel"><h2>覆盖与时间事实</h2><div id="coverageSummary"></div><div id="timeFacts" class="small"></div><div class="table-wrap"><table id="framesTable"></table></div><p class="small">源 frame index、array index 和 PTS 是精确核对依据；播放器 seek 仅作浏览。</p></section>
<section class="panel"><h2>结构变化（real/fake 独立选择）</h2><p class="small">显示定义：相对当前 triplet 首时刻的归一化距离变化；红色表示增长，蓝色表示缩短。这里不是“相对历史”，也不是真假概率。两侧同编号仅为各自视频内编号，不代表物理对应。</p><div class="sides"><div class="side"><h3>real 局部选择</h3><label>component <select id="realComponent"></select></label><label>triplet <select id="realTriplet"></select></label><label>pair <select id="realPair"></select></label><div class="table-wrap"><table id="realRelationTable"></table></div><div class="table-wrap"><table id="realCurveTable"></table></div></div><div class="side"><h3>fake 局部选择</h3><label>component <select id="fakeComponent"></select></label><label>triplet <select id="fakeTriplet"></select></label><label>pair <select id="fakePair"></select></label><div class="table-wrap"><table id="fakeRelationTable"></table></div><div class="table-wrap"><table id="fakeCurveTable"></table></div></div></div></section>
<section class="panel"><h2>模型响应与平均聚合</h2><p class="small">q 是局部 logit 响应，不是局部概率；贡献严格按当前 component/window 平均规则计算。不同 held-out fold 的 logit 尺度不作为统一异常刻度。</p><div class="table-wrap"><table id="modelTable"></table></div></section>
<section class="panel"><h2>开发人工标记（不进入训练或筛选）</h2><p class="small">先选择 real/fake，再在对应画面拖框或点选；坐标记录为原始图像坐标。统计是“选定时刻/区间内每帧的最大值”，不是检测准确率。没有区域时不输出覆盖率。</p><div class="controls"><label><input type="radio" name="regionMode" id="modeRect" checked> 鼠标框选</label><label><input type="radio" name="regionMode" id="modePoint"> 鼠标点选</label><label>标记侧 <select id="annRole"><option>real</option><option>fake</option></select></label><button id="clearRegion">清除区域</button></div><div class="annotation-grid"><label>可见失真 <select id="annVisible"><option>uncertain</option><option>yes</option><option>no</option></select></label><label>类型 <select id="annType"><option>uncertain</option><option>internal_deformation</option><option>overall_motion</option><option>other</option></select></label><label>开始秒 <input id="annStart" type="number" step=".001"></label><label>结束秒 <input id="annEnd" type="number" step=".001"></label><label>区域 x,y,w,h <input id="annRect" placeholder="鼠标拖框后自动填充"></label><label>点 x,y <input id="annPoint" placeholder="鼠标点选后自动填充"></label><textarea id="annNotes" placeholder="notes"></textarea></div><div id="annContext" class="small"></div><pre id="annCoverageStats" class="small"></pre><button id="saveAnn">保存标记</button><button id="exportJson">导出 JSON</button><button id="exportCsv">导出 CSV</button><label>导入 JSON/CSV <input id="importAnn" type="file" accept="application/json,text/csv,.csv"></label><div id="annStatus" class="small">标记只保存在当前浏览器，导出后再长期保存。</div></section>
<section class="panel"><h2>默认人工检查样本</h2><div id="caseLinks" class="case-links"></div><p class="small">播放器 seek 不是源帧级证据；请以页面表格中的源 frame index、array index 和真实 PTS 为准。</p></section>
<script id="payload" type="application/json">""" + encoded + """</script>
<script>
const DATA=JSON.parse(document.getElementById('payload').textContent);const windows=DATA.windows||[];const details=DATA.details||{};const groups=DATA.groups||{};const byId=Object.fromEntries(windows.map(x=>[x.window_id,x]));let currentGroup=null,annotations=[];const selection={real:{component:null,triplet:null,pair:null},fake:{component:null,triplet:null,pair:null}};let regionMode='rect',regionSelection=null,dragStart=null;
const $=id=>document.getElementById(id);const esc=x=>String(x??'').replaceAll('&','&amp;').replaceAll('<','&lt;').replaceAll('>','&gt;').replaceAll('"','&quot;');const fmt=x=>Number.isFinite(Number(x))?Number(x).toFixed(6):'—';const numberOrNull=x=>{if(x===null||x===undefined||String(x).trim()==='')return null;const n=Number(x);return Number.isFinite(n)?n:null};function dl(name,text,type){const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([text],{type}));a.download=name;a.click();setTimeout(()=>URL.revokeObjectURL(a.href),1000)}
function groupList(){return Object.values(groups).filter(g=>g.source_id===$('source').value&&g.kind===$('kind').value).sort((a,b)=>a.default_order_index-b.default_order_index)}
function populate(){const sources=[...new Set(Object.values(groups).map(x=>x.source_id))].sort();$('source').innerHTML=sources.map(x=>`<option>${esc(x)}</option>`).join('');if(!sources.length)return;let list=groupList();if(!list.length){$('kind').value=$('kind').value==='MANIP'?'CTRL':'MANIP';list=groupList()}$('window').innerHTML=list.map(g=>`<option value="${esc(g.group_id)}">${esc(g.label)} (${esc(g.group_id)})</option>`).join('');render()}
function setGroup(){const list=groupList();if(!list.length)return;currentGroup=groups[$('window').value]||list[0];$('window').value=currentGroup.group_id;for(const role of ['real','fake'])selection[role]={component:null,triplet:null,pair:null};regionSelection=null;dragStart=null;render()}
function mediaFor(wid){return (byId[wid]||{}).media||{status:'NOT_MATERIALIZED'} }
function setVideo(role,wid){const v=$(role+'Video'),c=$(role+'Canvas'),m=$(role+'Media'),media=mediaFor(wid),same=v.dataset.windowId===wid;v.onloadedmetadata=()=>draw(role);v.ontimeupdate=()=>{draw(role);sync(role)};if(!same){v.pause();v.removeAttribute('src');v.load();if(media.status==='AVAILABLE'){v.src=media.clip_path;const links=(media.model_frame_screenshots||[]).map(x=>`<a href="${x.path}" target="_blank">精确帧 ${x.source_frame_index} / PTS ${fmt(x.timestamp_s)}</a>`).join(' ');m.innerHTML=`展示短片：${esc(media.clip_path)}；源：${esc(media.source_video_path)}；${links?'模型实际采样截图：'+links:'暂无模型采样截图'}`}else if(media.status==='NOT_MATERIALIZED'||media.status==='SOURCE_MISSING'||media.status==='DECODE_FAILED'){m.textContent=`${media.status}；源：${media.source_video_path||'未记录'}${media.reason?'；'+media.reason:''}`}v.dataset.windowId=wid;c.width=1;c.height=1}else if(media.status==='AVAILABLE'){const links=(media.model_frame_screenshots||[]).map(x=>`<a href="${x.path}" target="_blank">精确帧 ${x.source_frame_index} / PTS ${fmt(x.timestamp_s)}</a>`).join(' ');m.innerHTML=`展示短片：${esc(media.clip_path)}；源：${esc(media.source_video_path)}；${links?'模型实际采样截图：'+links:'暂无模型采样截图'}`}}
function sync(role){const other=role==='real'?'fake':'real',a=$(role+'Video'),b=$(other+'Video');if(a.seeking||b.seeking||!a.duration||!b.duration)return;const t=Math.min(a.currentTime,b.duration);if(Math.abs(b.currentTime-t)>.08)b.currentTime=t}
function nearest(frames,t){if(!frames?.length)return null;return frames.reduce((a,b)=>Math.abs(b.relative_time_s-t)<Math.abs(a.relative_time_s-t)?b:a)}
function selectedTriplet(role,d){const s=selection[role],ts=d?.structure?.triplets||[];return ts.find(t=>String(t.triplet_id)===String(s.triplet))||null}
function draw(role){const v=$(role+'Video'),canvas=$(role+'Canvas'),wid=v.dataset.windowId,d=details[wid];if(!d||!v.videoWidth||!v.clientWidth)return;const fr=nearest(d.coverage.frame_records,v.currentTime);if(!fr)return;canvas.width=Math.max(1,v.clientWidth*devicePixelRatio);canvas.height=Math.max(1,v.clientHeight*devicePixelRatio);const ctx=canvas.getContext('2d');ctx.clearRect(0,0,canvas.width,canvas.height);const sx=canvas.width/v.videoWidth,sy=canvas.height/v.videoHeight;const colors=['#22c55e','#38bdf8','#f59e0b','#e879f9','#f43f5e','#a3e635','#fb7185','#2dd4bf'];const tr=selectedTriplet(role,d),s=selection[role],slots=new Set((tr?.target_slots||[]).map(Number));if(s.pair&&tr)for(const p of tr.pair_records||[]){if(p.pair_id.join('-')===s.pair)for(const slot of p.member_indices||[])slots.add(Number(slot))}if($('layerCoverage').checked){for(const p of fr.uv_points||[]){const [slot,track,u,vv,vis,geo,comp]=p;ctx.fillStyle=comp==null?'#facc15':colors[comp%colors.length];ctx.globalAlpha=geo?(vis?.95:.35):.55;ctx.beginPath();ctx.arc(u*sx,vv*sy,geo?3.5:2.5,0,Math.PI*2);ctx.fill();if(slots.has(Number(slot))){ctx.globalAlpha=1;ctx.strokeStyle='#fff';ctx.lineWidth=2;ctx.stroke()}if($('showIds').checked){ctx.globalAlpha=.8;ctx.fillStyle='#fff';ctx.font='10px sans-serif';ctx.fillText(String(track),u*sx+4,vv*sy-3)}}}ctx.globalAlpha=1;if($('layerStructure').checked&&tr&&s.pair){const points=Object.fromEntries((fr.uv_points||[]).map(p=>[p[0],[p[2],p[3]]]));const pos=tr.source_frame_indices.indexOf(fr.source_frame_index);if(pos>=0)for(const p of tr.pair_records||[]){if(p.pair_id.join('-')!==s.pair)continue;const left=p.member_indices[0],right=p.member_indices[1],uv=points[left]&&points[right]?[points[left],points[right]]:null;if(!uv)continue;const delta=p.normalized_distance[pos]-p.normalized_distance[0];ctx.strokeStyle=delta>0?'#ef4444':'#2563eb';ctx.lineWidth=3;ctx.beginPath();ctx.moveTo(uv[0][0]*sx,uv[0][1]*sy);ctx.lineTo(uv[1][0]*sx,uv[1][1]*sy);ctx.stroke()}}if(regionSelection&&regionSelection.role===role){ctx.strokeStyle='#facc15';ctx.lineWidth=2;ctx.setLineDash([6,4]);if(regionSelection.kind==='rect')ctx.strokeRect(regionSelection.x*sx,regionSelection.y*sy,regionSelection.w*sx,regionSelection.h*sy);else{ctx.beginPath();ctx.arc(regionSelection.x*sx,regionSelection.y*sy,5*sx,0,Math.PI*2);ctx.stroke()}ctx.setLineDash([])} }
function draw(role){const v=$(role+'Video'),canvas=$(role+'Canvas'),wid=v.dataset.windowId,d=details[wid];if(!d||!v.videoWidth||!v.clientWidth)return;const fr=nearest(d.coverage.frame_records,v.currentTime);if(!fr)return;canvas.width=Math.max(1,v.clientWidth*devicePixelRatio);canvas.height=Math.max(1,v.clientHeight*devicePixelRatio);const ctx=canvas.getContext('2d');ctx.clearRect(0,0,canvas.width,canvas.height);const sx=canvas.width/v.videoWidth,sy=canvas.height/v.videoHeight;const colors=['#22c55e','#38bdf8','#f59e0b','#e879f9','#f43f5e','#a3e635','#fb7185','#2dd4bf'];const tr=selectedTriplet(role,d),s=selection[role],slots=new Set((tr?.target_slots||[]).map(Number));if(s.pair&&tr)for(const p of tr.pair_records||[])if(p.pair_id.join('-')===s.pair)for(const slot of p.member_indices||[])slots.add(Number(slot));for(const p of fr.uv_points||[]){const [slot,track,u,vv,vis,geo,comp]=p;if(!vis)continue;const selected=slots.has(Number(slot));if((geo&&$('layerGeometry').checked)||(!geo&&$('layerTracks').checked)){ctx.fillStyle=comp==null?'#facc15':colors[comp%colors.length];ctx.globalAlpha=geo?.95:.7;ctx.beginPath();ctx.arc(u*sx,vv*sy,selected?5:geo?3.5:2.5,0,Math.PI*2);ctx.fill();if(selected&&$('layerTriplet').checked){ctx.globalAlpha=1;ctx.strokeStyle='#fff';ctx.lineWidth=2;ctx.stroke()}if($('showIds').checked){ctx.globalAlpha=.85;ctx.fillStyle='#fff';ctx.font='10px sans-serif';ctx.fillText(String(track),u*sx+4,vv*sy-3)}}}ctx.globalAlpha=1;if($('layerPair').checked&&$('layerStructure').checked&&tr&&s.pair){const points=Object.fromEntries((fr.uv_points||[]).map(p=>[p[0],[p[2],p[3]]]));const pos=tr.source_frame_indices.indexOf(fr.source_frame_index);if(pos>=0)for(const p of tr.pair_records||[]){if(p.pair_id.join('-')!==s.pair)continue;const a=points[p.member_indices[0]],b=points[p.member_indices[1]];if(!a||!b)continue;const delta=p.normalized_distance[pos]-p.normalized_distance[0];ctx.strokeStyle=delta>0?'#ef4444':'#2563eb';ctx.lineWidth=3;ctx.beginPath();ctx.moveTo(a[0]*sx,a[1]*sy);ctx.lineTo(b[0]*sx,b[1]*sy);ctx.stroke()}}if(regionSelection&&regionSelection.role===role){ctx.strokeStyle='#facc15';ctx.lineWidth=2;ctx.setLineDash([6,4]);if(regionSelection.kind==='rect')ctx.strokeRect(regionSelection.x*sx,regionSelection.y*sy,regionSelection.w*sx,regionSelection.h*sy);else{ctx.beginPath();ctx.arc(regionSelection.x*sx,regionSelection.y*sy,5*sx,0,Math.PI*2);ctx.stroke()}ctx.setLineDash([])}}
function render(){if(!currentGroup){setGroup();return}const real=currentGroup.real,fake=currentGroup.fake,rw=byId[real],fw=byId[fake];$('realTitle').textContent=$('showLabels').checked?'real '+(rw?.label||''):'real';$('fakeTitle').textContent=$('showLabels').checked?'fake '+(fw?.label||''):'fake';setVideo('real',real);setVideo('fake',fake);$('timeline').textContent=`source=${currentGroup.source_id}; interval=${rw?.interval_start_s}–${rw?.interval_end_s}s；双方选择独立；时间以各自 PTS 表为准`;const rd=details[real],fd=details[fake];renderCoverage(rd,fd);renderStructure('real',rd||{});renderStructure('fake',fd||{});renderModels(rd,fd);updateAnnotationContext();draw('real');draw('fake')}
function renderCoverage(rd,fd){const cov=[['real',rd],['fake',fd]];$('coverageSummary').innerHTML=cov.map(([role,d])=>{if(!d)return'';const c=d.coverage;return `<b>${role}</b>: status=${c.support_status}, query=${c.initial_query.initialized_track_slots}, frames=${c.frame_records.length}, model frames=${c.model_frame_indices.length}, components=${c.components.length}, triplets=${c.triplet_count}, invalid=${(c.invalid_reasons||[]).map(x=>esc(x.reason)).join(',')||'none'}`}).join('　');const rows=[];for(const [role,d] of cov){if(!d)continue;for(const f of d.coverage.frame_records||[])rows.push(`<tr><td>${role}</td><td>${f.array_index}</td><td>${f.source_frame_index}</td><td>${fmt(f.timestamp_s)}</td><td>${f.visible_count}/${f.uv_finite_count}</td><td>${f.geometry_valid_count}/${f.xyz_finite_count}</td><td>${f.model_used?'yes':'—'}</td></tr>`)}$('framesTable').innerHTML='<tr><th>side</th><th>array</th><th>source frame</th><th>PTS(s)</th><th>visible</th><th>geometry</th><th>model used</th></tr>'+rows.join('')}
function renderCoverage(rd,fd){const cov=[['real',rd],['fake',fd]];$('coverageSummary').innerHTML=cov.map(([role,d])=>{if(!d)return'';const c=d.coverage;return `<b>${role}</b>: status=${c.support_status}, query=${c.initial_query.initialized_track_slots}, frames=${c.frame_records.length}, history=${c.history_frame_indices.length}, evaluation=${c.evaluation_frame_indices.length}, model slots=${c.model_frame_indices.length}, components=${c.components.length}, triplets=${c.triplet_count}, invalid=${(c.invalid_reasons||[]).map(x=>esc(x.reason)).join(',')||'none'}`}).join('　');$('timeFacts').innerHTML=cov.map(([role,d])=>{if(!d)return'';const c=d.coverage,matches=(c.target_matches||[]).map(x=>`${x.status==='MATCHED'?x.frame_index:'缺失'}@${x.target_time_s.toFixed(3)}s`).join(', ');return `<div><b>${role}</b> 历史=${c.history_frame_indices[0]??'—'}…${c.history_frame_indices.at(-1)??'—'}；评估=${c.evaluation_frame_indices[0]??'—'}…${c.evaluation_frame_indices.at(-1)??'—'}；目标槽=${matches}；有效 triplet=${c.triplet_count}</div>`}).join('');const rows=[];for(const [role,d] of cov){if(!d)continue;const records=d.coverage.frame_records||[],modelTimes=[].concat(...(d.coverage.actual_model_time_records||[]).map(t=>t.timestamps_s||[]));for(const f of records){const near=modelTimes.length?modelTimes.reduce((a,b)=>Math.abs(b-f.timestamp_s)<Math.abs(a-f.timestamp_s)?b:a):null;rows.push(`<tr><td>${role}</td><td>${f.array_index}</td><td>${f.source_frame_index}</td><td>${fmt(f.timestamp_s)}</td><td>${fmt(f.relative_time_s)}</td><td>${near===null?'—':fmt(near)}</td><td>${f.visible_count}/${f.uv_finite_count}</td><td>${f.geometry_valid_count}/${f.xyz_finite_count}</td><td>${f.model_used?'yes':'—'}</td></tr>`)}}$('framesTable').innerHTML='<tr><th>side</th><th>array</th><th>source frame</th><th>PTS(s)</th><th>window relative(s)</th><th>nearest model PTS(s)</th><th>visible</th><th>geometry</th><th>model used</th></tr>'+rows.join('')}
function renderStructure(role,d){const s=selection[role],ts=d.structure?.triplets||[],cs=[...new Set(ts.map(t=>Number(t.component_index)))].sort((a,b)=>a-b);if(s.component===null||!cs.includes(Number(s.component)))s.component=cs[0]??null;const csel=$(role+'Component'),tsel=$(role+'Triplet'),psel=$(role+'Pair');csel.innerHTML=cs.map(c=>`<option value="${c}" ${Number(s.component)===c?'selected':''}>${c}</option>`).join('');const ct=ts.filter(t=>Number(t.component_index)===Number(s.component));const tids=ct.map(t=>String(t.triplet_id));if(s.triplet===null||!tids.includes(String(s.triplet)))s.triplet=tids[0]??null;tsel.innerHTML=ct.map(t=>`<option value="${t.triplet_id}" ${String(s.triplet)===String(t.triplet_id)?'selected':''}>t${t.triplet_id} slots=${t.target_slots.join(',')}</option>`).join('');const tr=ct.find(t=>String(t.triplet_id)===String(s.triplet)),pairs=tr?.pair_records||[],pids=pairs.map(p=>p.pair_id.join('-'));if(s.pair===null||!pids.includes(String(s.pair)))s.pair=pids[0]??null;psel.innerHTML=pids.map(p=>`<option value="${p}" ${String(s.pair)===String(p)?'selected':''}>${p}</option>`).join('');const rel=$(role+'RelationTable'),curve=$(role+'CurveTable');if(!tr){rel.innerHTML='<tr><td>无实际 triplet 支撑，结构未测量。</td></tr>';curve.innerHTML='';return}rel.innerHTML='<tr><th>pair</th><th>triplet source frames / PTS</th><th>raw distance</th><th>fixed-scale normalized</th></tr>'+pairs.filter(p=>!s.pair||p.pair_id.join('-')===s.pair).map(p=>`<tr><td>${p.pair_id.join('-')}</td><td>${tr.source_frame_indices.join(', ')}<br>${tr.timestamps_s.map(fmt).join(', ')}</td><td>${p.raw_distance.map(fmt).join(', ')}</td><td>${p.normalized_distance.map(fmt).join(', ')}</td></tr>`).join('');const state=tr.saved_states||[],first=tr.signed_first_difference||[],second=tr.signed_second_difference||[];curve.innerHTML='<tr><th colspan="5">St=[mean,std,p25,p75] and timestamp-aware signed differences</th></tr><tr><th>t</th><th>PTS(s)</th><th>mean</th><th>std</th><th>p25 / p75</th></tr>'+state.map((row,i)=>`<tr><td>${i}</td><td>${fmt(tr.timestamps_s[i])}</td><td>${fmt(row[0])}</td><td>${fmt(row[1])}</td><td>${fmt(row[2])} / ${fmt(row[3])}</td></tr>`).join('')+`<tr><td colspan="5">signed first: [${first.map(fmt).join(', ')}]; signed second: [${second.map(fmt).join(', ')}]</td></tr>`}
function renderModels(rd,fd){const table=$('modelTable');if(!$('layerModel').checked){table.style.display='none';table.innerHTML='';return}table.style.display='table';const rows=[];for(const [role,d] of [['real',rd],['fake',fd]]){if(!d||!$('showScores').checked)continue;const response=d.model_response||{},s=selection[role];for(const arm of Object.keys(response)){const r=response[arm],parts=(r.components||[]).filter(c=>s.component===null||Number(c.component_index)===Number(s.component));const detail=parts.map(c=>`c${c.component_index}: q=${fmt(c.component_q)}, component contribution=${fmt(c.component_contribution)}; `+(c.triplets||[]).filter(t=>s.triplet===null||String(t.triplet_id)===String(s.triplet)).map(t=>`t${t.triplet_id}: q=${fmt(t.q_mean)}, weight=${fmt(t.triplet_weight)}, weighted contribution=${fmt(t.weighted_contribution)}`).join(' ')).join('<br>');rows.push(`<tr><td>${role}</td><td>${arm}</td><td>${fmt(r.mean_score)}</td><td>${(r.seed_scores||[]).map(fmt).join(', ')}</td><td>${detail||'未找到所选局部响应'}</td></tr>`)}}table.innerHTML='<tr><th>side</th><th>arm</th><th>window logit</th><th>seed logits</th><th>selected component/triplet q, weight, contribution</th></tr>'+rows.join('')||'<tr><td>未评分</td></tr>'}
function currentFrame(role){const v=$(role+'Video'),d=details[v.dataset.windowId];return d?nearest(d.coverage.frame_records,v.currentTime)||d.coverage.frame_records[0]:null}
function inside(point){if(!regionSelection)return false;const x=Number(point[2]),y=Number(point[3]);if(regionSelection.kind==='point')return Math.hypot(x-regionSelection.x,y-regionSelection.y)<=5;return x>=regionSelection.x&&x<=regionSelection.x+regionSelection.w&&y>=regionSelection.y&&y<=regionSelection.y+regionSelection.h}
function computeCoverage(role){const v=$(role+'Video'),d=details[v.dataset.windowId];if(!d||!regionSelection||regionSelection.role!==role)return {status:'未选择区域'};const t0=numberOrNull($('annStart').value),t1=numberOrNull($('annEnd').value);if(t0!==null&&t1!==null&&t1<t0)return {status:'时间区间无效：结束早于开始'};const frames=(d.coverage.frame_records||[]).filter(f=>(t0===null||f.timestamp_s>=t0)&&(t1===null||f.timestamp_s<=t1));if(!frames.length)return {status:'选定时间内没有采样帧'};const per=[];for(const f of frames){const points=f.uv_points||[],visible=points.filter(p=>p[4]&&inside(p)),geo=points.filter(p=>p[5]&&inside(p));let common=0,both=0,one=0;for(const tr of d.structure?.triplets||[]){const pos=tr.source_frame_indices.indexOf(f.source_frame_index);if(pos<0)continue;const memberSet=new Set((tr.common_member_indices||[]).map(Number));common=Math.max(common,[...memberSet].filter(slot=>{const p=points.find(q=>Number(q[0])===slot);return p&&inside(p)}).length);for(const p of tr.pair_records||[]){const a=points.find(q=>Number(q[0])===Number(p.member_indices[0])),b=points.find(q=>Number(q[0])===Number(p.member_indices[1]));const ia=!!a&&inside(a),ib=!!b&&inside(b);if(ia&&ib)both++;else if(ia||ib)one++}}per.push({source_frame_index:f.source_frame_index,timestamp_s:f.timestamp_s,visible:visible.length,geometry:geo.length,common,both,one,model_used:!!f.model_used})}const max=k=>Math.max(...per.map(x=>x[k]),0);return {status:'ok; selected-frame maximums',frames_considered:per.length,max_visible_query_points:max('visible'),max_geometry_valid_points:max('geometry'),max_triplet_common_members:max('common'),max_pairs_both_endpoints:max('both'),max_pairs_one_endpoint:max('one'),model_frame_count:per.filter(x=>x.model_used).length,model_sample_range:(d.coverage.model_frame_indices||[]).map(Number),per_frame:per}}
function computeCoverage(role){const v=$(role+'Video'),d=details[v.dataset.windowId];if(!d||!regionSelection||regionSelection.role!==role)return {status:'未选择区域'};const t0=numberOrNull($('annStart').value),t1=numberOrNull($('annEnd').value);if(t0!==null&&t1!==null&&t1<t0)return {status:'时间区间无效：结束早于开始'};const frames=(d.coverage.frame_records||[]).filter(f=>(t0===null||f.timestamp_s>=t0)&&(t1===null||f.timestamp_s<=t1));if(!frames.length)return {status:'选定时间内没有采样帧'};const s=selection[role],tr=selectedTriplet(role,d),component=(d.coverage.components||[]).find(c=>Number(c.component_index)===Number(s.component));const per=[];for(const f of frames){const points=f.uv_points||[],visible=points.filter(p=>p[4]&&inside(p)),geo=points.filter(p=>p[5]&&inside(p));let common=0,both=0,one=0,tripletIds=[];if(tr){const pos=tr.source_frame_indices.indexOf(f.source_frame_index);if(pos>=0){tripletIds=[tr.triplet_id];const memberSet=new Set((tr.common_member_indices||[]).map(Number));common=[...memberSet].filter(slot=>{const p=points.find(q=>Number(q[0])===slot);return p&&inside(p)}).length;for(const p of tr.pair_records||[]){const a=points.find(q=>Number(q[0])===Number(p.member_indices[0])),b=points.find(q=>Number(q[0])===Number(p.member_indices[1]));const ia=!!a&&inside(a),ib=!!b&&inside(b);if(ia&&ib)both++;else if(ia||ib)one++}}}per.push({source_frame_index:f.source_frame_index,timestamp_s:f.timestamp_s,visible:visible.length,geometry:geo.length,common,both,one,tripletIds,model_used:!!f.model_used})}const max=k=>Math.max(...per.map(x=>x[k]),0),modelFrames=per.filter(x=>x.model_used);return {status:'ok; selected-frame maximums',aggregation:'maximum across selected frames; counts are not accuracy',frames_considered:per.length,max_visible_query_points:max('visible'),max_geometry_valid_points:max('geometry'),max_selected_triplet_common_members:max('common'),max_pairs_both_endpoints:max('both'),max_pairs_one_endpoint:max('one'),component_index:s.component,component_total_members:component?.member_indices?.length??0,component_total_pairs:component?Math.floor(component.member_indices.length*(component.member_indices.length-1)/2):0,model_frame_count:modelFrames.length,model_sample_frame_indices:modelFrames.map(x=>x.source_frame_index),model_sample_timestamps_s:modelFrames.map(x=>x.timestamp_s),effective_triplet_ids:[...new Set(per.flatMap(x=>x.tripletIds))],per_frame:per}}
function updateAnnotationContext(){const role=$('annRole').value,fr=currentFrame(role),d=fr?details[$(role+'Video').dataset.windowId]:null;if(fr){if($('annStart').value==='')$('annStart').value=String(fr.timestamp_s);if($('annEnd').value==='')$('annEnd').value=String(fr.timestamp_s);$('annContext').textContent=`${role} | window=${d.coverage.window_id} | source frame=${fr.source_frame_index} | PTS=${fmt(fr.timestamp_s)} | array=${fr.array_index}`}else $('annContext').textContent='尚无可用视频帧';$('annCoverageStats').textContent=JSON.stringify(computeCoverage(role),null,2)}
function canvasPoint(role,event){const canvas=$(role+'Canvas'),v=$(role+'Video'),rect=canvas.getBoundingClientRect();return {x:(event.clientX-rect.left)*v.videoWidth/rect.width,y:(event.clientY-rect.top)*v.videoHeight/rect.height}}
function updateRegionInputs(){if(!regionSelection)return;if(regionSelection.kind==='rect'){$('annRect').value=[regionSelection.x,regionSelection.y,regionSelection.w,regionSelection.h].map(x=>Number(x.toFixed(3))).join(',');$('annPoint').value=''}else{$('annPoint').value=[regionSelection.x,regionSelection.y].map(x=>Number(x.toFixed(3))).join(',');$('annRect').value=''}$('annRole').value=regionSelection.role;updateAnnotationContext();draw('real');draw('fake')}
function attachCanvas(role){const c=$(role+'Canvas');c.addEventListener('pointerdown',e=>{if(!$(role+'Video').videoWidth)return;const p=canvasPoint(role,e);if(regionMode==='point'){regionSelection={role,kind:'point',x:p.x,y:p.y};updateRegionInputs();return}dragStart=p;c.setPointerCapture(e.pointerId)});c.addEventListener('pointermove',e=>{if(!dragStart||regionMode!=='rect')return;const p=canvasPoint(role,e);regionSelection={role,kind:'rect',x:Math.min(dragStart.x,p.x),y:Math.min(dragStart.y,p.y),w:Math.abs(p.x-dragStart.x),h:Math.abs(p.y-dragStart.y)};updateRegionInputs()});c.addEventListener('pointerup',e=>{if(!dragStart||regionMode!=='rect')return;const p=canvasPoint(role,e);regionSelection={role,kind:'rect',x:Math.min(dragStart.x,p.x),y:Math.min(dragStart.y,p.y),w:Math.abs(p.x-dragStart.x),h:Math.abs(p.y-dragStart.y)};dragStart=null;updateRegionInputs()})}
function parseCsv(text){const rows=[];let row=[],field='',quoted=false;for(let i=0;i<text.length;i++){const ch=text[i];if(ch==='"'){if(quoted&&text[i+1]==='"'){field+='"';i++}else quoted=!quoted}else if(ch===','&&!quoted){row.push(field);field=''}else if((ch==='\n'||ch==='\r')&&!quoted){if(ch==='\r'&&text[i+1]==='\n')i++;row.push(field);if(row.some(x=>x!==''))rows.push(row);row=[];field=''}else field+=ch}if(field||row.length){row.push(field);rows.push(row)}const keys=rows.shift()||[];return rows.map(r=>Object.fromEntries(keys.map((k,i)=>[k,r[i]??''])))}
function decodeAnnotationRow(a){for(const k of ['time_interval_s','region','point','coverage_counts','model_sample_range'])if(typeof a[k]==='string'&&a[k]){try{a[k]=JSON.parse(a[k])}catch(_){}}return a}
function currentAnnotation(){const role=$('annRole').value,v=$(role+'Video'),d=details[v.dataset.windowId],fr=currentFrame(role),region=regionSelection&&regionSelection.role===role?(regionSelection.kind==='rect'?{x:regionSelection.x,y:regionSelection.y,w:regionSelection.w,h:regionSelection.h}:null):null,point=regionSelection&&regionSelection.role===role&&regionSelection.kind==='point'?{x:regionSelection.x,y:regionSelection.y}:null;return {source_id:currentGroup?.source_id,window_id:currentGroup?.group_id,role,real_window_id:currentGroup?.real,fake_window_id:currentGroup?.fake,source_frame_index:fr?.source_frame_index??null,array_index:fr?.array_index??null,timestamp_s:fr?.timestamp_s??null,visible_distortion:$('annVisible').value,distortion_type:$('annType').value,time_interval_s:[numberOrNull($('annStart').value),numberOrNull($('annEnd').value)],region:region,point:point,selected_component:selection[role].component,selected_triplet:selection[role].triplet,selected_pair:selection[role].pair,coverage_counts:computeCoverage(role),model_sample_range:d?.coverage?.model_frame_indices||[],notes:$('annNotes').value||'',created_at:new Date().toISOString()}}
function currentAnnotation(){const role=$('annRole').value,v=$(role+'Video'),d=details[v.dataset.windowId],fr=currentFrame(role),region=regionSelection&&regionSelection.role===role?(regionSelection.kind==='rect'?{x:regionSelection.x,y:regionSelection.y,w:regionSelection.w,h:regionSelection.h}:null):null,point=regionSelection&&regionSelection.role===role&&regionSelection.kind==='point'?{x:regionSelection.x,y:regionSelection.y}:null;return {source_id:currentGroup?.source_id,window_id:currentGroup?.group_id,video_identity:d?.coverage?.source_id?`${role}:${d.coverage.source_id}:${d.coverage.window_id}`:null,role,real_window_id:currentGroup?.real,fake_window_id:currentGroup?.fake,source_frame_index:fr?.source_frame_index??null,array_index:fr?.array_index??null,timestamp_s:fr?.timestamp_s??null,frame_size_hw:fr?[fr.height,fr.width]:null,visible_distortion:$('annVisible').value,distortion_type:$('annType').value,time_interval_s:[numberOrNull($('annStart').value),numberOrNull($('annEnd').value)],region:region,point:point,selected_component:selection[role].component,selected_triplet:selection[role].triplet,selected_pair:selection[role].pair,coverage_counts:computeCoverage(role),model_sample_range:d?.coverage?.model_frame_indices||[],notes:$('annNotes').value||'',created_at:new Date().toISOString()}}
function csvEscape(x){return '"'+String(x??'').replaceAll('"','""')+'"'}
$('source').onchange=()=>{const list=groupList();$('window').innerHTML=list.map(g=>`<option value="${esc(g.group_id)}">${esc(g.label)} (${esc(g.group_id)})</option>`).join('');setGroup()};$('kind').onchange=()=>$('source').onchange();$('window').onchange=setGroup;for(const role of ['real','fake']){for(const field of ['Component','Triplet','Pair'])$(role+field).onchange=e=>{selection[role][field.toLowerCase()]=e.target.value;if(field!=='Pair'&&field!=='Triplet')selection[role].triplet=null;if(field==='Component'){selection[role].triplet=null;selection[role].pair=null}if(field==='Triplet')selection[role].pair=null;render()};attachCanvas(role)}$('showIds').onchange=render;$('showLabels').onchange=render;$('showScores').onchange=render;for(const id of ['layerCoverage','layerStructure','layerModel'])$(id).onchange=()=>{render();draw('real');draw('fake')};$('prev').onclick=()=>move(-1);$('next').onclick=()=>move(1);function move(delta){const list=groupList(),i=Math.max(0,Math.min(list.length-1,list.findIndex(g=>g.group_id===currentGroup.group_id)+delta));$('window').value=list[i]?.group_id||'';setGroup()}
$('play').onclick=()=>{$('realVideo').play();$('fakeVideo').play()};$('pause').onclick=()=>{$('realVideo').pause();$('fakeVideo').pause()};$('step').onclick=()=>{for(const role of ['real','fake']){const v=$(role+'Video'),d=details[v.dataset.windowId];if(!d||!v.duration)continue;const fr=nearest(d.coverage.frame_records,v.currentTime+.034);if(fr)v.currentTime=fr.relative_time_s}};$('seekTarget').onclick=()=>{for(const role of ['real','fake']){const v=$(role+'Video'),d=details[v.dataset.windowId],tr=selectedTriplet(role,d);if(tr&&v.duration){const t=tr.timestamps_s[1]-d.coverage.frame_records[0].timestamp_s;v.currentTime=Math.max(0,Math.min(v.duration,t))}}};$('modeRect').onchange=()=>{regionMode='rect'};$('modePoint').onchange=()=>{regionMode='point'};$('annRole').onchange=()=>{if(regionSelection)regionSelection.role=$('annRole').value;updateRegionInputs();render()};$('clearRegion').onclick=()=>{regionSelection=null;dragStart=null;$('annRect').value='';$('annPoint').value='';updateAnnotationContext();draw('real');draw('fake')};for(const id of ['annStart','annEnd'])$(id).oninput=updateAnnotationContext;
$('saveAnn').onclick=()=>{if(!currentGroup){$('annStatus').textContent='没有选定窗口';return}const a=currentAnnotation();annotations.push(a);$('annStatus').textContent=`已保存 ${annotations.length} 条（开发解释用途；请导出后长期保存）`};$('exportJson').onclick=()=>dl('v7_annotations.json',JSON.stringify(annotations,null,2),'application/json');$('exportCsv').onclick=()=>{const keys=['source_id','window_id','role','real_window_id','fake_window_id','source_frame_index','array_index','timestamp_s','visible_distortion','distortion_type','time_interval_s','region','point','selected_component','selected_triplet','selected_pair','coverage_counts','model_sample_range','notes','created_at'];const lines=[keys.join(','),...annotations.map(a=>keys.map(k=>csvEscape(typeof a[k]==='object'?JSON.stringify(a[k]):a[k])).join(','))];dl('v7_annotations.csv',lines.join('\n'),'text/csv')};$('importAnn').onchange=async e=>{const file=e.target.files[0];if(!file)return;const text=await file.text();try{const x=file.name.toLowerCase().endsWith('.csv')?parseCsv(text):JSON.parse(text);if(!Array.isArray(x))throw Error('文件必须是数组');annotations=x.map(decodeAnnotationRow);$('annStatus').textContent=`已导入 ${annotations.length} 条`}catch(err){$('annStatus').textContent='导入失败：'+err.message}};
function initCases(){const cases=DATA.sample_cases||{};$('caseLinks').innerHTML=Object.entries(cases).map(([name,gid])=>`<a href="#" data-g="${esc(gid)}">${esc(name)} → ${esc(gid)}</a>`).join('');document.querySelectorAll('#caseLinks a').forEach(a=>a.onclick=e=>{e.preventDefault();const g=groups[a.dataset.g];if(!g)return;$('source').value=g.source_id;$('kind').value=g.kind;$('source').onchange();$('window').value=g.group_id;setGroup()})}
function stepToAdjacentSourceFrame(role){const v=$(role+'Video'),d=details[v.dataset.windowId],fr=currentFrame(role);if(!d||!fr)return;const records=d.coverage.frame_records||[],i=records.findIndex(x=>x.array_index===fr.array_index),next=records[Math.max(0,Math.min(records.length-1,i+1))];if(next)v.currentTime=next.relative_time_s}
$('step').onclick=()=>{for(const role of ['real','fake'])stepToAdjacentSourceFrame(role)};
for(const id of ['layerTracks','layerGeometry','layerTriplet','layerPair'])$(id).onchange=()=>{draw('real');draw('fake')};
/* Final drawing override: target_slots are temporal target slots; particle
   highlighting must use common_member_indices instead. */
function draw(role){const v=$(role+'Video'),canvas=$(role+'Canvas'),wid=v.dataset.windowId,d=details[wid];if(!d||!v.videoWidth||!v.clientWidth)return;const fr=nearest(d.coverage.frame_records,v.currentTime);if(!fr)return;canvas.width=Math.max(1,v.clientWidth*devicePixelRatio);canvas.height=Math.max(1,v.clientHeight*devicePixelRatio);const ctx=canvas.getContext('2d');ctx.clearRect(0,0,canvas.width,canvas.height);const sx=canvas.width/v.videoWidth,sy=canvas.height/v.videoHeight,colors=['#22c55e','#38bdf8','#f59e0b','#e879f9','#f43f5e','#a3e635','#fb7185','#2dd4bf'],tr=selectedTriplet(role,d),s=selection[role],slots=new Set((tr?.common_member_indices||[]).map(Number));if(s.pair&&tr)for(const p of tr.pair_records||[])if(p.pair_id.join('-')===s.pair)for(const slot of p.member_indices||[])slots.add(Number(slot));for(const p of fr.uv_points||[]){const [slot,track,u,vv,vis,geo,comp]=p;if(!vis||!Number.isFinite(u)||!Number.isFinite(vv))continue;const selected=slots.has(Number(slot)),componentSelected=s.component!==null&&Number(comp)===Number(s.component),showPoint=$('layerTracks').checked||(geo&&$('layerGeometry').checked);if(!showPoint)continue;ctx.fillStyle=comp==null?'#facc15':colors[Number(comp)%colors.length];ctx.globalAlpha=geo?.95:.7;ctx.beginPath();ctx.arc(u*sx,vv*sy,selected?5:geo?3.5:2.5,0,Math.PI*2);ctx.fill();if(selected&&$('layerTriplet').checked){ctx.globalAlpha=1;ctx.strokeStyle='#fff';ctx.lineWidth=2;ctx.stroke()}else if(componentSelected){ctx.globalAlpha=1;ctx.strokeStyle='#fbbf24';ctx.lineWidth=1.5;ctx.stroke()}if($('showIds').checked){ctx.globalAlpha=.85;ctx.fillStyle='#fff';ctx.font='10px sans-serif';ctx.fillText(String(track),u*sx+4,vv*sy-3)}}ctx.globalAlpha=1;if($('layerPair').checked&&tr&&s.pair){const points=Object.fromEntries((fr.uv_points||[]).filter(p=>p[4]&&Number.isFinite(p[2])&&Number.isFinite(p[3])).map(p=>[p[0],[p[2],p[3]]]));const pos=tr.source_frame_indices.indexOf(fr.source_frame_index);if(pos>=0)for(const p of tr.pair_records||[]){if(p.pair_id.join('-')!==s.pair)continue;const a=points[p.member_indices[0]],b=points[p.member_indices[1]];if(!a||!b)continue;const delta=p.normalized_distance[pos]-p.normalized_distance[0];ctx.strokeStyle=delta>0?'#ef4444':'#2563eb';ctx.lineWidth=3;ctx.beginPath();ctx.moveTo(a[0]*sx,a[1]*sy);ctx.lineTo(b[0]*sx,b[1]*sy);ctx.stroke()}}if(regionSelection&&regionSelection.role===role){ctx.strokeStyle='#facc15';ctx.lineWidth=2;ctx.setLineDash([6,4]);if(regionSelection.kind==='rect')ctx.strokeRect(regionSelection.x*sx,regionSelection.y*sy,regionSelection.w*sx,regionSelection.h*sy);else{ctx.beginPath();ctx.arc(regionSelection.x*sx,regionSelection.y*sy,5*sx,0,Math.PI*2);ctx.stroke()}ctx.setLineDash([])}}
function canvasPoint(role,event){const canvas=$(role+'Canvas'),v=$(role+'Video'),rect=canvas.getBoundingClientRect(),clamp=(x,lo,hi)=>Math.max(lo,Math.min(hi,x));return {x:clamp((event.clientX-rect.left)*v.videoWidth/rect.width,0,v.videoWidth),y:clamp((event.clientY-rect.top)*v.videoHeight/rect.height,0,v.videoHeight)}}
const previousAnnotationRoleChange=$('annRole').onchange;$('annRole').onchange=()=>{if(regionSelection&&regionSelection.role!==$('annRole').value)regionSelection=null;if(typeof previousAnnotationRoleChange==='function')previousAnnotationRoleChange();updateRegionInputs();render()};
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
        source_path = Path(str(manifest["video_path"]))
        fallback_media = {
            "status": "NOT_MATERIALIZED" if source_path.is_file() else "SOURCE_MISSING",
            "source_video_path": str(source_path),
        }
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
                "media": dict(media_by_window.get(wid, fallback_media)),
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
        media_by_window, screenshot_rows = _materialize_review_media(output_root, groups, coverage, structure, _default_review_cases(groups, coverage))
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
    _write_annotation_inventory(output_root)
    _write_measurement_contract(output_root, coverage, structure)
    _write_json(output_root / "review/index_data.json", {"windows": index_rows, "groups": groups, "sample_cases": sample_cases})
    page_payload = {
        "protocol": protocol,
        "windows": index_rows,
        "groups": groups,
        "details": details_payload,
        "sample_cases": sample_cases,
        "annotation_schema": {"status": "development-only; not training or population selection", "fields": ["source_id", "window_id", "role", "source_frame_index", "timestamp_s", "visible_distortion", "distortion_type", "time_interval_s", "region", "point", "coverage_counts", "model_sample_range", "notes"]},
        "annotation_inventory_path": "../annotation_inventory.csv",
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
        "review": {"html": str(review_path), "index_entries": len(index_rows), "media_entries": len(media_by_window), "screenshots": len(screenshot_rows), "browser_visual_verification": False, "annotation_inventory": str(output_root / "annotation_inventory.csv")},
        "elapsed_s": time.perf_counter() - started,
        "boundaries": ["read-only frozen model forward passes", "no optimizer.step", "no frontend/provider rerun", "no GPU", "not a formal detector result", "180 fold/seed models are not independent samples"],
    }
    _write_json(output_root / "evaluation/summary.json", summary)
    _write_json(output_root / "run_summary.json", {"status": summary["status"], "artifact_root": str(output_root), "summary": "evaluation/summary.json", "review_html": str(review_path), "annotation_inventory": str(output_root / "annotation_inventory.csv"), "measurement_contract": str(output_root / "measurement_contract.json"), "window_count": len(coverage), "model_record_count": len(diagnostics), "media_entry_count": len(media_by_window)})
    return summary


def attach_media(
    output_root: Path = DIAGNOSTIC_OUTPUT_ROOT,
    window_ids: Sequence[str] | None = None,
) -> dict[str, Any]:
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
    sample_cases = dict(index_data.get("sample_cases", {}))
    requested_window_ids = None if window_ids is None else {str(item) for item in window_ids}
    if window_ids is None:
        requested_cases = _default_review_cases(groups, coverage)
        requested_cases.update({name: gid for name, gid in sample_cases.items() if gid in groups})
        media_by_window, screenshot_rows = _materialize_review_media(output_root, groups, coverage, structure, requested_cases)
    else:
        requested_cases = {}
        for value in requested_window_ids:
            group_id = value.split("::", 1)[0]
            if group_id in groups:
                requested_cases[f"on_demand_{_safe_slug(value)}"] = group_id
        media_by_window, screenshot_rows = _materialize_review_media(output_root, groups, coverage, structure, requested_cases, requested_window_ids=requested_window_ids)
    old_media = {
        str(row["window_id"]): dict(row.get("media", {}))
        for row in index_data.get("windows", [])
        if row.get("media", {}).get("status") in {"AVAILABLE", "SOURCE_MISSING", "DECODE_FAILED"}
    }
    merged_media = old_media | media_by_window
    for window_id, media in merged_media.items():
        coverage[window_id]["media"] = media
    old_screenshot_rows = json.loads((output_root / "review/screenshot_index.json").read_text(encoding="utf-8")) if (output_root / "review/screenshot_index.json").is_file() else []
    screenshot_by_window = {str(row["window_id"]): row for row in old_screenshot_rows}
    screenshot_by_window.update({str(row["window_id"]): row for row in screenshot_rows})
    screenshot_rows = [screenshot_by_window[key] for key in sorted(screenshot_by_window)]
    index_rows = _build_index(input_rows, coverage, response, window_manifest, merged_media)
    details_payload = {row["window_id"]: {"coverage": coverage[row["window_id"]], "structure": structure[row["window_id"]], "model_response": response.get(row["window_id"])} for row in index_rows}
    protocol = json.loads((output_root / "protocol.json").read_text(encoding="utf-8"))
    page_payload = {
        "protocol": protocol,
        "windows": index_rows,
        "groups": groups,
        "details": details_payload,
        "sample_cases": sample_cases,
        "annotation_schema": {"status": "development-only; not training or population selection", "fields": ["source_id", "window_id", "role", "source_frame_index", "timestamp_s", "visible_distortion", "distortion_type", "time_interval_s", "region", "point", "coverage_counts", "model_sample_range", "notes"]},
        "annotation_inventory_path": "../annotation_inventory.csv",
        "selected_pairs_count": 16,
    }
    _write_json(output_root / "coverage/coverage_windows.json", coverage)
    _write_annotation_inventory(output_root)
    _write_measurement_contract(output_root, coverage, structure)
    _write_json(output_root / "review/screenshot_index.json", screenshot_rows)
    _write_json(output_root / "review/index_data.json", {"windows": index_rows, "groups": groups, "sample_cases": sample_cases})
    _write_json(output_root / "review/data.json", page_payload)
    (output_root / "review/index.html").write_text(build_review_html(page_payload), encoding="utf-8")
    summary_path = output_root / "evaluation/summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["review"] = {"html": str(output_root / "review/index.html"), "index_entries": len(index_rows), "media_entries": len(merged_media), "screenshots": len(screenshot_rows), "browser_visual_verification": False, "annotation_inventory": str(output_root / "annotation_inventory.csv")}
    _write_json(summary_path, summary)
    _write_json(output_root / "run_summary.json", {"status": summary["status"], "artifact_root": str(output_root), "summary": "evaluation/summary.json", "review_html": str(output_root / "review/index.html"), "annotation_inventory": str(output_root / "annotation_inventory.csv"), "measurement_contract": str(output_root / "measurement_contract.json"), "window_count": len(coverage), "model_record_count": summary.get("population", {}).get("model_records"), "media_entry_count": len(merged_media)})
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DIAGNOSTIC_OUTPUT_ROOT)
    parser.add_argument("--no-media", action="store_true", help="skip optional short clips and screenshots")
    parser.add_argument("--attach-media", action="store_true", help="attach review media to an existing numeric diagnostic")
    parser.add_argument("--window-id", action="append", default=None, help="materialise one existing group/window (repeatable); only used with --attach-media")
    args = parser.parse_args()
    if args.window_id and not args.attach_media:
        parser.error("--window-id requires --attach-media")
    result = attach_media(args.output_root, args.window_id) if args.attach_media else run(output_root=args.output_root, make_media=not args.no_media)
    print(json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
