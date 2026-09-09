"""Run the fixed local-relation temporal supervised V7 development pilot."""

from __future__ import annotations

import csv
import hashlib
import json
import subprocess
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from sparse3d_forgery.particle_sequence import load_particle_sequence

from .model import (
    ARM_NAMES,
    MODEL_CONFIG,
    build_batch,
    fit_weighted_standardizer,
    score_model,
    serialize_model,
    source_class_window_weights,
    train_model,
    validate_training_examples,
    window_label,
)
from .representation import (
    COMPONENT_CONFIG,
    MAX_TARGET_ERROR_S,
    TARGET_OFFSETS_S,
    build_window_support,
    json_ready_support,
    support_arrays,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
DATA_ROOT = Path("/root/autodl-tmp/data/sparse_3d_forgery_detection")
SOURCE_ARTIFACT_ROOT = DATA_ROOT / "derived/v7_activityforensics_paired_second_order_pilot_v1"
OUTPUT_ROOT = DATA_ROOT / "derived/v7_activityforensics_local_structural_temporal_supervised_pilot_v4"
BOOTSTRAP_SEED = 20260909
BOOTSTRAP_REPLICATES = 10_000
SEEDS = (20260909, 20260910, 20260911)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


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


def _input_manifest_row(result: Mapping[str, Any], support: Mapping[str, Any], sequence: Any) -> dict[str, Any]:
    window = result["window"]
    matches = support["target_matches"]
    errors = [item["match_error_s"] for item in matches if item["match_error_s"] is not None]
    intervals = [
        float(item["timestamps_s"][index + 1] - item["timestamps_s"][index])
        for item in support["triplets"]
        for index in (0, 1)
        if item["timestamps_s"][index] is not None and item["timestamps_s"][index + 1] is not None
    ]
    return {
        "window_id": str(window["window_id"]),
        "pair_id": str(window["pair_id"]),
        "source_id": str(window["source_id"]),
        "role": str(window["role"]),
        "kind": str(window["kind"]),
        "label": window_label(window),
        "anchor_fraction": float(window["anchor_fraction"]),
        "window_start_s": float(window["interval_start_s"]),
        "window_end_s": float(window["interval_end_s"]),
        "particle_prefix": str(result["particle_prefix"]),
        "particle_npz_sha256": _sha256(Path(result["particle_prefix"]).with_suffix(".npz")),
        "particle_json_sha256": _sha256(Path(result["particle_prefix"]).with_suffix(".json")),
        "sequence_schema_version": str(sequence.schema_version),
        "sequence_frame_count": int(sequence.num_frames),
        "sequence_track_count": int(sequence.num_tracks),
        "history_frame_count": len(support["history_frame_indices"]),
        "evaluation_frame_count": len(support["evaluation_frame_indices"]),
        "matched_target_count": sum(item["status"] == "MATCHED" for item in matches),
        "valid_triplet_count": int(support["valid_triplet_count"]),
        "support_status": str(support["support_status"]),
        "invalid_reason_counts": dict(Counter(str(item["reason"]) for item in support["invalid_reasons"])),
        "match_error_summary_s": {
            "count": len(errors),
            "max": float(max(errors)) if errors else None,
            "median": float(np.median(errors)) if errors else None,
        },
        "actual_interval_summary_s": {
            "count": len(intervals),
            "min": float(min(intervals)) if intervals else None,
            "max": float(max(intervals)) if intervals else None,
            "median": float(np.median(intervals)) if intervals else None,
        },
    }


def load_frozen_examples(source_root: Path = SOURCE_ARTIFACT_ROOT) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Read all frozen particle artifacts and construct support without scoring."""

    results_path = source_root / "frontend/window_results.json"
    selected_path = source_root / "manifests/selected_pairs.json"
    windows_path = source_root / "manifests/window_manifest.json"
    results = json.loads(results_path.read_text(encoding="utf-8"))
    selected = json.loads(selected_path.read_text(encoding="utf-8"))
    frozen_windows = json.loads(windows_path.read_text(encoding="utf-8"))
    if len(selected) != 16 or len(frozen_windows) != 192 or len(results) != 192:
        raise ValueError("frozen input must contain exactly 16 pairs and 192 completed windows")
    examples: list[dict[str, Any]] = []
    support_rows: list[dict[str, Any]] = []
    input_rows: list[dict[str, Any]] = []
    for result in results:
        if result.get("status") != "COMPLETE":
            raise ValueError(f"incomplete frozen window: {result.get('window', {}).get('window_id')}")
        window = result["window"]
        prefix = Path(result["particle_prefix"])
        sequence = load_particle_sequence(prefix)
        if str(sequence.lineage.get("source_id")) != str(window["source_id"]):
            raise ValueError(f"source identity mismatch: {window['window_id']}")
        support = build_window_support(sequence, window_start_s=float(window["interval_start_s"]))
        support_json = json_ready_support(support)
        support_row = {
            "window_id": str(window["window_id"]),
            "source_id": str(window["source_id"]),
            "pair_id": str(window["pair_id"]),
            "role": str(window["role"]),
            "kind": str(window["kind"]),
            "label": window_label(window),
            **support_json,
        }
        support_rows.append(support_row)
        input_rows.append(_input_manifest_row(result, support, sequence))
        prepared = support_arrays(support)
        if prepared["triplets"]:
            examples.append(
                {
                    "window_id": str(window["window_id"]),
                    "pair_id": str(window["pair_id"]),
                    "source_id": str(window["source_id"]),
                    "role": str(window["role"]),
                    "kind": str(window["kind"]),
                    "anchor_fraction": float(window["anchor_fraction"]),
                    "triplets": prepared["triplets"],
                    "support": support,
                }
            )
    population = {
        "selected_pairs": len(selected),
        "frozen_windows": len(frozen_windows),
        "completed_frontend_windows": len(results),
        "valid_support_windows": len(examples),
        "all_source_ids": sorted({str(row["source_id"]) for row in frozen_windows}),
        "support_source_ids": sorted({str(row["source_id"]) for row in examples}),
        "invalid_support_sources": sorted({str(row["source_id"]) for row in frozen_windows} - {str(row["source_id"]) for row in examples}),
        "source_artifact_root": str(source_root),
        "frontend_window_results_sha256": _sha256(results_path),
        "selected_pairs_sha256": _sha256(selected_path),
        "window_manifest_sha256": _sha256(windows_path),
    }
    return examples, support_rows, {"population": population, "input_rows": input_rows}


def protocol_for(population: Mapping[str, Any], *, input_manifest_sha256: str, support_manifest_sha256: str, head: str) -> dict[str, Any]:
    """Return the predeclared protocol; no result-dependent value is included."""

    return {
        "experiment": "V7 Fixed Local-Relation-Supported Temporal Structural Supervised Pilot",
        "research_question": "Whether fixed local relation identity supports cross-source temporal structural discrimination beyond unordered state, and whether second order adds information, in the frozen development population.",
        "git_head": head,
        "input": {
            "source_artifact_root": str(SOURCE_ARTIFACT_ROOT),
            "input_manifest": "manifests/input_manifest.json",
            "input_manifest_sha256": input_manifest_sha256,
            "support_manifest": "manifests/window_support.json",
            "support_manifest_sha256": support_manifest_sha256,
            "population": "16 frozen ActivityForensics/Charades source pairs, 192 role windows",
            "window_identity": "existing MANIP/CTRL windows unchanged",
            "numeric_input": "local fixed-support relation state only; no labels, source, role, generator, path, or provenance",
        },
        "history_and_evaluation": {
            "history": "first 0.5 seconds of each one-second window using true PTS",
            "evaluation": "target offsets 0.5, 0.6, 0.7, 0.8, 0.9 seconds from window start",
            "target_matching": "nearest source PTS within 0.05 seconds; ties choose earlier frame; no duplicate source frame, interpolation, extrapolation, or gap bridging",
            "local_triplets": "only contiguous target slots (0,1,2), (1,2,3), (2,3,4); all three must be matched",
        },
        "component_and_relation_support": {
            "component_algorithm": "existing motion_coherent_components on history only",
            "component_config": {
                "max_initial_distance": COMPONENT_CONFIG.max_initial_distance,
                "max_relative_change": COMPONENT_CONFIG.max_relative_change,
                "minimum_size": COMPONENT_CONFIG.minimum_size,
                "minimum_overlap": COMPONENT_CONFIG.minimum_overlap,
            },
            "candidate_pairs": "all undirected pairs of component members",
            "common_observation": "the same geometry-valid members and all their pair identities at all three triplet times; at least three members",
            "history_scale": "pooled median of all finite valid H pair-time distances in that component; non-finite or non-positive scale invalid",
            "missing": "invalid triplets remain explicitly recorded and are never filled or concatenated across missing slots",
        },
        "state_and_derivatives": {
            "state": "S(t)=[mean,std,p25,p75] of pair distances divided by the fixed history component scale",
            "derivatives": "h_minus=t-t_minus, h_plus=t_plus-t; v_minus=(S_t-S_minus)/h_minus; v_plus=(S_plus-S_t)/h_plus; a_t=2*(v_plus-v_minus)/(h_minus+h_plus)",
            "interpretation": "signed structural changes, not physical velocity/acceleration and not future prediction",
            "full_input_shape": [12],
        },
        "arms": {
            "UNORDERED_STATE": "each of the three S values is encoded as [S,S,S], then the three shared-MLP outputs are averaged",
            "ORDERED_FIRST": "[S_t,v_minus,0]",
            "ORDERED_SECOND": "[S_t,v_minus,a_t]",
            "PERMUTED_SECOND": "all six joint permutations of the three S values, recompute derivatives with unchanged time slots, and average outputs; train and test use the same six-way average",
            "shared_structure": "each arm uses the same 12->16->8 ReLU MLP, component mean, window mean, and one linear window head",
            "support": "all arms use exactly the same valid window/component/triplet support",
        },
        "training": {
            "labels": {"real_MANIP": 0, "fake_MANIP": 1, "CTRL": "excluded from fitting; descriptive evaluation only"},
            "split": "leave-one-source-out; held-out source windows never enter fitting, standardization, or weights",
            "standardization": "per arm, weighted training inputs only; zero variance uses fixed scale 1.0 and is recorded",
            "weights": "equal total weight per training source/class, equal within source/class window weight, then equal component/triplet/within-triplet observation weight; A three states and D six permutations split weight equally; window and observation weights normalized to mean one",
            "model": MODEL_CONFIG,
            "seeds": list(SEEDS),
            "device": "CPU",
        },
        "evaluation": {
            "primary": "source-equal mean AUROC of fake MANIP vs real MANIP for ORDERED_SECOND",
            "contrasts": {"C-A": "ORDERED_SECOND - UNORDERED_STATE", "C-D": "ORDERED_SECOND - PERMUTED_SECOND", "C-B": "ORDERED_SECOND - ORDERED_FIRST"},
            "pooled": "auxiliary only because fold logits have separate scales",
            "bootstrap": {"unit": "complete source resampling with all source windows and all four arms", "seed": BOOTSTRAP_SEED, "replicates": BOOTSTRAP_REPLICATES, "same_indices_for_all_arms": True},
            "state_rules": {
                "MEASUREMENT_SUPPORT_INSUFFICIENT": "fewer than 12 source-complete two-class sources",
                "LOCAL_TEMPORAL_DISCRIMINATION_SUPPORTED_IN_DEVELOPMENT": "C AUROC CI lower > 0.5 and C-A and C-D CI lower > 0",
                "DISCRIMINATION_WITHOUT_ESTABLISHED_TEMPORAL_GAIN": "C AUROC CI lower > 0.5 but either temporal gain CI lower <= 0",
                "LOCAL_TEMPORAL_DISCRIMINATION_NOT_ESTABLISHED": "otherwise",
            },
            "secondary_second_order": "SECOND_ORDER_INCREMENT_SUPPORTED only when C-B CI lower > 0; otherwise NOT_ESTABLISHED",
        },
        "boundaries": [
            "development population only; not independent, sealed-test, full-video, or cross-generator guarantee",
            "local offline three-observation discrimination; not future prediction or calibrated probability",
            "no frontend rerun, GPU, NSI, dynamic fusion, feature search, or model search",
        ],
    }


def _finite(values: Iterable[float]) -> np.ndarray:
    array = np.asarray(list(values), dtype=np.float64)
    return array[np.isfinite(array)]


def auroc(positive: Iterable[float], negative: Iterable[float]) -> float | None:
    pos, neg = _finite(positive), _finite(negative)
    if pos.size == 0 or neg.size == 0:
        return None
    comparison = (pos[:, None] > neg[None, :]).astype(np.float64)
    comparison += 0.5 * (pos[:, None] == neg[None, :])
    return float(np.mean(comparison))


def _summary(values: Iterable[float]) -> dict[str, Any]:
    array = _finite(values)
    if array.size == 0:
        return {"N": 0, "mean": None, "median": None, "min": None, "max": None}
    return {"N": int(array.size), "mean": float(np.mean(array)), "median": float(np.median(array)), "min": float(np.min(array)), "max": float(np.max(array))}


def _source_metrics(oof: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in oof:
        if row.get("kind") != "MANIP":
            continue
        source = str(row["source_id"])
        for arm in ARM_NAMES:
            value = row.get(f"{arm}_score")
            if value is not None:
                grouped[source][f"{arm}_{row['role']}"] .append(float(value))
    output: list[dict[str, Any]] = []
    for source in sorted(grouped):
        row: dict[str, Any] = {"source_id": source}
        for arm in ARM_NAMES:
            real = grouped[source].get(f"{arm}_real", [])
            fake = grouped[source].get(f"{arm}_fake", [])
            row[f"{arm}_real_count"] = len(real)
            row[f"{arm}_fake_count"] = len(fake)
            row[f"{arm}_auroc"] = auroc(fake, real)
        row["C-A"] = row["ORDERED_SECOND_auroc"] - row["UNORDERED_STATE_auroc"] if row["ORDERED_SECOND_auroc"] is not None and row["UNORDERED_STATE_auroc"] is not None else None
        row["C-D"] = row["ORDERED_SECOND_auroc"] - row["PERMUTED_SECOND_auroc"] if row["ORDERED_SECOND_auroc"] is not None and row["PERMUTED_SECOND_auroc"] is not None else None
        row["C-B"] = row["ORDERED_SECOND_auroc"] - row["ORDERED_FIRST_auroc"] if row["ORDERED_SECOND_auroc"] is not None and row["ORDERED_FIRST_auroc"] is not None else None
        output.append(row)
    return output


def paired_source_bootstrap(source_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    complete = [row for row in source_rows if all(row.get(f"{arm}_auroc") is not None for arm in ARM_NAMES)]
    if not complete:
        return {"N": 0, "seed": BOOTSTRAP_SEED, "replicates": BOOTSTRAP_REPLICATES, "arms": {}, "gains": {}}
    arm_values = {arm: np.asarray([float(row[f"{arm}_auroc"]) for row in complete], dtype=np.float64) for arm in ARM_NAMES}
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    indices = rng.integers(0, len(complete), size=(BOOTSTRAP_REPLICATES, len(complete)))
    means = {arm: np.mean(values[indices], axis=1) for arm, values in arm_values.items()}
    gains = {name: means["ORDERED_SECOND"] - means[other] for name, other in (("C-A", "UNORDERED_STATE"), ("C-D", "PERMUTED_SECOND"), ("C-B", "ORDERED_FIRST"))}
    return {
        "N": len(complete),
        "source_ids": [str(row["source_id"]) for row in complete],
        "seed": BOOTSTRAP_SEED,
        "replicates": BOOTSTRAP_REPLICATES,
        "arms": {arm: {"mean": float(np.mean(values)), "ci95": [float(np.percentile(means[arm], 2.5)), float(np.percentile(means[arm], 97.5))]} for arm, values in arm_values.items()},
        "gains": {name: {"mean": float(np.mean(values)), "ci95": [float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))]} for name, values in gains.items()},
    }


def _pooled(oof: Sequence[Mapping[str, Any]], arm: str, positive_role: str = "fake", negative_kind: str | None = "MANIP") -> float | None:
    positive = [row[f"{arm}_score"] for row in oof if row.get("role") == positive_role and row.get("kind") == "MANIP"]
    negative = [row[f"{arm}_score"] for row in oof if row.get("role") == "real" and (negative_kind is None or row.get("kind") == negative_kind)]
    return auroc(positive, negative)


def _arm_descriptives(oof: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for arm in ARM_NAMES:
        output[arm] = {}
        for role in ("real", "fake"):
            for kind in ("MANIP", "CTRL"):
                values = [row[f"{arm}_score"] for row in oof if row.get("role") == role and row.get("kind") == kind]
                output[arm][f"{role}_{kind}"] = _summary(values)
    return output


def _primary_status(bootstrap: Mapping[str, Any], *, complete_sources: int) -> tuple[str, str]:
    if complete_sources < 12:
        return "MEASUREMENT_SUPPORT_INSUFFICIENT", "fewer than 12 source-complete two-class sources"
    arms = bootstrap.get("arms", {})
    gains = bootstrap.get("gains", {})
    c = arms.get("ORDERED_SECOND", {}).get("ci95")
    ca = gains.get("C-A", {}).get("ci95")
    cd = gains.get("C-D", {}).get("ci95")
    if c and ca and cd and c[0] > 0.5 and ca[0] > 0 and cd[0] > 0:
        return "LOCAL_TEMPORAL_DISCRIMINATION_SUPPORTED_IN_DEVELOPMENT", "C lower CI > 0.5 and C-A/C-D lower CIs > 0"
    if c and c[0] > 0.5:
        return "DISCRIMINATION_WITHOUT_ESTABLISHED_TEMPORAL_GAIN", "C lower CI > 0.5 but at least one temporal gain lower CI <= 0"
    return "LOCAL_TEMPORAL_DISCRIMINATION_NOT_ESTABLISHED", "predeclared support conditions not met"


def _build_protocol_and_manifests(output_root: Path, examples: list[dict[str, Any]], support_rows: list[dict[str, Any]], details: Mapping[str, Any], head: str) -> dict[str, Any]:
    input_rows = details["input_rows"]
    _write_json(output_root / "manifests/input_manifest.json", {"rows": input_rows, "population": details["population"]})
    _write_json(output_root / "manifests/window_support.json", support_rows)
    _write_json(output_root / "manifests/coverage.json", {
        "total_windows": len(input_rows),
        "valid_support_windows": len(examples),
        "by_support_status": dict(Counter(str(row["support_status"]) for row in input_rows)),
        "by_role_kind": {f"{role}_{kind}": {"total": sum(row["role"] == role and row["kind"] == kind for row in input_rows), "valid": sum(row["role"] == role and row["kind"] == kind and row["support_status"] == "VALID" for row in input_rows)} for role in ("real", "fake") for kind in ("MANIP", "CTRL")},
        "invalid_reason_counts": dict(Counter(reason for row in support_rows for item in row.get("invalid_reasons", []) for reason in [str(item["reason"])])),
    })
    protocol = protocol_for(details["population"], input_manifest_sha256=_sha256(output_root / "manifests/input_manifest.json"), support_manifest_sha256=_sha256(output_root / "manifests/window_support.json"), head=head)
    _write_json(output_root / "protocol.json", protocol)
    return protocol


def run(source_root: Path = SOURCE_ARTIFACT_ROOT, output_root: Path = OUTPUT_ROOT) -> dict[str, Any]:
    started = time.perf_counter()
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"pilot output already exists: {output_root}")
    examples, support_rows, details = load_frozen_examples(source_root)
    output_root.mkdir(parents=True, exist_ok=True)
    head = _git_head()
    _build_protocol_and_manifests(output_root, examples, support_rows, details, head)

    all_sources = list(details["population"]["all_source_ids"])
    main_examples = [example for example in examples if window_label(example) is not None]
    oof_by_id: dict[str, dict[str, Any]] = {
        str(example["window_id"]): {
            "window_id": str(example["window_id"]),
            "pair_id": str(example["pair_id"]),
            "source_id": str(example["source_id"]),
            "role": str(example["role"]),
            "kind": str(example["kind"]),
            "label": window_label(example),
        }
        for example in examples
    }
    fold_support: list[dict[str, Any]] = []
    model_records: list[dict[str, Any]] = []
    for held_out in all_sources:
        heldout = [example for example in examples if str(example["source_id"]) == held_out]
        training = [example for example in main_examples if str(example["source_id"]) != held_out]
        if not heldout:
            fold_support.append({"held_out_source": held_out, "status": "NO_VALID_SUPPORT_WINDOWS", "held_out_window_count": 0, "training_real_count": sum(window_label(item) == 0 for item in training), "training_fake_count": sum(window_label(item) == 1 for item in training)})
            continue
        validate_training_examples(training)
        support_by_arm: dict[str, tuple[Any, Any, Any]] = {}
        for arm in ARM_NAMES:
            raw_batch, observation_weights = build_batch(training, arm, observation_weights=True)
            if observation_weights is None:
                raise RuntimeError("observation weights were not returned")
            standardizer = fit_weighted_standardizer(raw_batch.inputs, observation_weights)
            train_batch, _ = build_batch(training, arm, standardizer=standardizer)
            held_batch, _ = build_batch(heldout, arm, standardizer=standardizer)
            support_by_arm[arm] = (standardizer, train_batch, held_batch)
        for arm in ARM_NAMES:
            standardizer, train_batch, held_batch = support_by_arm[arm]
            for seed in SEEDS:
                model, fit_info = train_model(train_batch, seed=seed)
                scores = score_model(model, held_batch)
                for example, score in zip(heldout, scores):
                    oof_by_id[str(example["window_id"])][f"{arm}_score_seed_{seed}"] = float(score)
                model_records.append({"held_out_source": held_out, "arm": arm, "seed": seed, "training_source_count": len({str(item["source_id"]) for item in training}), "training_real_count": sum(window_label(item) == 0 for item in training), "training_fake_count": sum(window_label(item) == 1 for item in training), "held_out_window_count": len(heldout), "training_triplet_count": train_batch.n_triplets, "held_out_triplet_count": held_batch.n_triplets, "zero_variance_dimensions": list(standardizer.zero_variance_dimensions), "fit": fit_info, "model": serialize_model(model, standardizer)})
                del model
        fold_support.append({"held_out_source": held_out, "status": "SCORED", "held_out_window_count": len(heldout), "training_source_count": len({str(item["source_id"]) for item in training}), "training_real_count": sum(window_label(item) == 0 for item in training), "training_fake_count": sum(window_label(item) == 1 for item in training), "held_out_main_windows": sum(window_label(item) is not None for item in heldout)})

    oof_rows: list[dict[str, Any]] = []
    for row in oof_by_id.values():
        for arm in ARM_NAMES:
            values = [row.get(f"{arm}_score_seed_{seed}") for seed in SEEDS]
            finite = [float(value) for value in values if value is not None and np.isfinite(value)]
            row[f"{arm}_score"] = float(np.mean(finite)) if len(finite) == len(SEEDS) else None
            row[f"{arm}_seed_std"] = float(np.std(finite)) if len(finite) == len(SEEDS) else None
        oof_rows.append(row)
    _write_csv(output_root / "scores/oof_window_scores.csv", oof_rows)
    _write_csv(output_root / "evaluation/fold_support.csv", fold_support)
    _write_json(output_root / "models/fold_models.json", {"model_config": MODEL_CONFIG, "seeds": list(SEEDS), "folds": model_records})

    source_rows = _source_metrics(oof_rows)
    bootstrap = paired_source_bootstrap(source_rows)
    _write_csv(output_root / "evaluation/per_source_metrics.csv", source_rows)
    complete_sources = [row for row in source_rows if all(row.get(f"{arm}_auroc") is not None for arm in ARM_NAMES)]
    primary_status, status_reason = _primary_status(bootstrap, complete_sources=len(complete_sources))
    second_order = "SECOND_ORDER_INCREMENT_SUPPORTED" if bootstrap.get("gains", {}).get("C-B", {}).get("ci95", [None])[0] is not None and bootstrap["gains"]["C-B"]["ci95"][0] > 0 else "SECOND_ORDER_INCREMENT_NOT_ESTABLISHED"
    coverage_rows = json.loads((output_root / "manifests/input_manifest.json").read_text(encoding="utf-8"))["rows"]
    invalid_reason_counts = Counter(reason for row in support_rows for item in row.get("invalid_reasons", []) for reason in [str(item["reason"])])
    summary = {
        "status": primary_status,
        "status_reason": status_reason,
        "secondary_second_order_status": second_order,
        "experiment": "V7 Fixed Local-Relation-Supported Temporal Structural Supervised Pilot",
        "git_head": head,
        "population": details["population"],
        "coverage": {
            "frozen_windows": len(coverage_rows),
            "valid_support_windows": len(oof_rows),
            "invalid_support_windows": len(coverage_rows) - len(oof_rows),
            "valid_support_sources": sorted({str(row["source_id"]) for row in oof_rows}),
            "invalid_support_sources": details["population"]["invalid_support_sources"],
            "source_complete_two_class_count": len(complete_sources),
            "all_source_ids": all_sources,
            "invalid_reason_counts": dict(invalid_reason_counts),
            "by_role_kind": {f"{role}_{kind}": {"total": sum(row["role"] == role and row["kind"] == kind for row in coverage_rows), "valid": sum(row["role"] == role and row["kind"] == kind for row in oof_rows)} for role in ("real", "fake") for kind in ("MANIP", "CTRL")},
            "target_match_error_s": _summary(item["match_error_summary_s"]["max"] for item in coverage_rows if item["match_error_summary_s"]["max"] is not None),
            "actual_interval_s": _summary(item["actual_interval_summary_s"]["median"] for item in coverage_rows if item["actual_interval_summary_s"]["median"] is not None),
        },
        "training": {
            "main_real_windows": sum(window_label(item) == 0 for item in main_examples),
            "main_fake_windows": sum(window_label(item) == 1 for item in main_examples),
            "control_windows_excluded": sum(item["kind"] == "CTRL" for item in examples),
            "fold_support": "evaluation/fold_support.csv",
            "seeds": list(SEEDS),
            "model_records": len(model_records),
            "device": "CPU",
        },
        "source_level_metrics": source_rows,
        "source_bootstrap": bootstrap,
        "pooled_auroc_auxiliary": {arm: {"fake_MANIP_vs_real_MANIP": _pooled(oof_rows, arm), "fake_MANIP_vs_all_real": _pooled(oof_rows, arm, negative_kind=None)} for arm in ARM_NAMES},
        "arm_score_descriptives": _arm_descriptives(oof_rows),
        "contrasts": {"C-A": "ORDERED_SECOND - UNORDERED_STATE", "C-D": "ORDERED_SECOND - PERMUTED_SECOND", "C-B": "ORDERED_SECOND - ORDERED_FIRST"},
        "support_manifest": "manifests/window_support.json",
        "input_manifest": "manifests/input_manifest.json",
        "oof_scores": "scores/oof_window_scores.csv",
        "fold_models": "models/fold_models.json",
        "protocol": "protocol.json",
        "paired_reference_used": False,
        "frontend_rerun": False,
        "full_video": False,
        "sealed_test": False,
        "normality_model": False,
        "formal_src_modified": False,
        "elapsed_s": time.perf_counter() - started,
    }
    _write_json(output_root / "evaluation/summary.json", summary)
    _write_json(output_root / "run_summary.json", {"status": primary_status, "secondary_second_order_status": second_order, "artifact_root": str(output_root), "summary": "evaluation/summary.json", "oof_window_count": len(oof_rows), "source_complete_two_class_count": len(complete_sources), "formal_src_modified": False})
    return summary


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, default=SOURCE_ARTIFACT_ROOT)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    args = parser.parse_args()
    print(json.dumps(run(args.source_root, args.output_root), indent=2, sort_keys=True, ensure_ascii=False))


if __name__ == "__main__":
    main()
