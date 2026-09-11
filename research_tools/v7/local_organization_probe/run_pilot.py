"""Run the frozen V7 historical-local-organization matched pilot.

The runner consumes the already materialized 289-point ParticleSequence
artifacts.  It rebuilds the old support and a history-only local grouping,
then trains the existing CPU MLP on the predeclared matched population.  No
frontend, tracking, depth, pose, or formal ``src`` pipeline is changed.
"""

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
from sparse3d_forgery.video_input import VideoSource, decode_video
from research_tools.v7.local_structural_temporal_probe.model import (
    ARM_NAMES,
    MODEL_CONFIG,
    build_batch,
    fit_weighted_standardizer,
    score_model,
    serialize_model,
    train_model,
    validate_training_examples,
    window_label,
)
from research_tools.v7.local_structural_temporal_probe.representation import (
    COMPONENT_CONFIG,
    json_ready_support,
    support_arrays,
)

from .grouping import (
    LOCAL_DIAMETER_M,
    build_local_groups,
    build_local_support,
    build_support_from_components,
    json_ready_local_support,
    rebuild_components_fast,
    validate_grouping,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
DATA_ROOT = Path("/root/autodl-tmp/data/sparse_3d_forgery_detection")
PILOT_INPUT_ROOT = DATA_ROOT / "derived/v7_activityforensics_density_supervised_pilot_v1"
FRONTEND_ROOT = DATA_ROOT / "derived/v7_activityforensics_density_matched_frontend_v1"
OUTPUT_ROOT = DATA_ROOT / "derived/v7_activityforensics_local_organization_pilot_v1"
BOOTSTRAP_SEED = 20260909
BOOTSTRAP_REPLICATES = 10_000
SEEDS = (20260909, 20260910, 20260911)
PILOT_ARMS = ("G_C", "L_C", "L_A", "L_D")
ARM_TO_MODEL = {
    "G_C": "ORDERED_SECOND",
    "L_C": "ORDERED_SECOND",
    "L_A": "UNORDERED_STATE",
    "L_D": "PERMUTED_SECOND",
}


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    raise TypeError(type(value).__name__)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False, default=_json_default) + "\n",
        encoding="utf-8",
    )


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_head() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()


def _safe(value: str) -> str:
    return "".join(char if char.isalnum() or char in "._-" else "_" for char in value)


def _resolve_frontend_root() -> tuple[Path, dict[str, Any]]:
    """Resolve the sibling frontend through the completed pilot identity hash."""

    protocol_path = PILOT_INPUT_ROOT / "protocol.json"
    if not protocol_path.is_file():
        raise FileNotFoundError(protocol_path)
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    expected_hash = str(protocol["coverage"]["frontend_results_sha256"])
    frontend_results = FRONTEND_ROOT / "frontend_results.json"
    if not frontend_results.is_file() or _sha256(frontend_results) != expected_hash:
        raise ValueError("289-point frontend artifact does not match the completed density pilot hash")
    return FRONTEND_ROOT, protocol


def _compact_grouping(grouping: Mapping[str, Any]) -> dict[str, Any]:
    parents: list[dict[str, Any]] = []
    for parent in grouping.get("parent_components", []):
        evidence = list(parent.get("pair_evidence", []))
        reasons = Counter(str(item["rejection_reason"]) for item in evidence if item.get("rejection_reason"))
        parents.append(
            {
                "parent_component_id": int(parent["parent_component_id"]),
                "member_slots": [int(item) for item in parent["member_slots"]],
                "track_ids": [int(item) for item in parent["track_ids"]],
                "pair_count": len(evidence),
                "pair_merge_allowed_count": int(sum(bool(item.get("merge_allowed")) for item in evidence)),
                "pair_rejection_reason_counts": dict(sorted(reasons.items())),
            }
        )
    return {
        "local_diameter_m": float(grouping["local_diameter_m"]),
        "component_config": dict(grouping["component_config"]),
        "history_array_indices": [int(item) for item in grouping["history_array_indices"]],
        "parent_components": parents,
        "groups": [dict(row) for row in grouping.get("groups", [])],
        "retained_group_count": int(grouping["retained_group_count"]),
        "support_insufficient_group_count": int(grouping["support_insufficient_group_count"]),
        "retained_track_ids": [int(item) for item in grouping["retained_track_ids"]],
    }


def _support_summary(support: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "support_status": str(support["support_status"]),
        "valid_triplet_count": int(support["valid_triplet_count"]),
        "component_count": int(len(support.get("components", []))),
        "history_frame_count": int(len(support.get("history_array_indices", []))),
        "invalid_reason_counts": dict(Counter(str(item["reason"]) for item in support.get("invalid_reasons", []))),
    }


def _input_row(row: Mapping[str, Any], sequence: Any, prefix: Path, old_support: Mapping[str, Any], local_support: Mapping[str, Any], grouping: Mapping[str, Any], prior: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "window_id": str(row["window_id"]),
        "pair_id": str(row["pair_id"]),
        "source_id": str(row["source_id"]),
        "role": str(row["role"]),
        "kind": str(row["kind"]),
        "label": window_label(row),
        "anchor_fraction": float(row["anchor_fraction"]),
        "interval_start_s": float(row["interval_start_s"]),
        "interval_end_s": float(row["interval_end_s"]),
        "particle_prefix": str(prefix),
        "particle_npz_sha256": _sha256(prefix.with_suffix(".npz")),
        "particle_json_sha256": _sha256(prefix.with_suffix(".json")),
        "frame_indices": [int(item) for item in sequence.frame_indices],
        "timestamps_s": [float(item) for item in sequence.timestamps_s],
        "source_frame_contract_equal": bool(np.array_equal(sequence.frame_indices, np.asarray(row["frame_indices"], dtype=np.int64))),
        "source_timestamp_max_abs_error_s": float(np.max(np.abs(sequence.timestamps_s - np.asarray(row["timestamps_s"], dtype=np.float64)))),
        "old_component_count": int(len(old_support.get("components", []))),
        "old_valid_triplet_count": int(old_support["valid_triplet_count"]),
        "old_support_status": str(old_support["support_status"]),
        "local_parent_component_count": int(len(grouping.get("parent_components", []))),
        "local_group_count": int(len(grouping.get("groups", []))),
        "local_retained_group_count": int(grouping["retained_group_count"]),
        "local_support_insufficient_group_count": int(grouping["support_insufficient_group_count"]),
        "local_valid_triplet_count": int(local_support["valid_triplet_count"]),
        "local_support_status": str(local_support["support_status"]),
        "prior_density289_support_status": prior.get("density289_support_status"),
        "prior_density289_triplet_count": prior.get("density289_triplet_count"),
    }


def load_population(
    pilot_input_root: Path = PILOT_INPUT_ROOT,
    frontend_root: Path = FRONTEND_ROOT,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Load all 192 windows and build old/local support without model scores."""

    protocol = json.loads((pilot_input_root / "protocol.json").read_text(encoding="utf-8"))
    coverage_rows = json.loads((pilot_input_root / "manifests/coverage.json").read_text(encoding="utf-8"))["coverage"]
    prior_by_id = {str(row["window_id"]): row for row in coverage_rows}
    manifest = json.loads((frontend_root / "window_manifest.json").read_text(encoding="utf-8"))
    result_rows = json.loads((frontend_root / "frontend_results.json").read_text(encoding="utf-8"))["results"]
    result_by_id = {(str(row["window_id"]), str(row["density"])): row for row in result_rows}
    if len(manifest) != 192 or len(result_by_id) != 384:
        raise ValueError("matched frontend must contain 192 windows and 384 density results")

    examples: list[dict[str, Any]] = []
    support_rows: list[dict[str, Any]] = []
    input_rows: list[dict[str, Any]] = []
    grouping_rows: list[dict[str, Any]] = []
    old_support_count = local_support_count = common_count = 0
    old_source_ids: set[str] = set()
    local_source_ids: set[str] = set()
    common_source_ids: set[str] = set()
    for row in manifest:
        window_id = str(row["window_id"])
        result = result_by_id.get((window_id, "density289"))
        if result is None:
            raise ValueError(f"missing density289 result: {window_id}")
        prefix = Path(result["sequence_prefix"])
        if not prefix.with_suffix(".npz").is_file() or not prefix.with_suffix(".json").is_file():
            raise FileNotFoundError(prefix)
        sequence = load_particle_sequence(prefix)
        if not np.array_equal(sequence.frame_indices, np.asarray(row["frame_indices"], dtype=np.int64)):
            raise ValueError(f"frame identity mismatch: {window_id}")
        timestamp_error = float(np.max(np.abs(sequence.timestamps_s - np.asarray(row["timestamps_s"], dtype=np.float64))))
        if timestamp_error > 1e-7:
            raise ValueError(f"timestamp mismatch: {window_id} ({timestamp_error})")

        history_indices = np.flatnonzero(sequence.timestamps_s < float(row["interval_start_s"]) + 0.5).astype(np.int64)
        expected_components = result.get("diagnostic", {}).get("component_members", [])
        rebuilt_components = rebuild_components_fast(sequence.xyz, sequence.geometry_validity, history_indices, COMPONENT_CONFIG)
        expected_canonical = sorted(tuple(sorted(int(item) for item in component)) for component in expected_components)
        rebuilt_canonical = sorted(tuple(sorted(int(item) for item in component)) for component in rebuilt_components)
        if expected_canonical != rebuilt_canonical:
            raise ValueError(f"old component reconstruction mismatch: {window_id}")
        old_support = build_support_from_components(sequence, window_start_s=float(row["interval_start_s"]), components=expected_components)
        prior = prior_by_id.get(window_id)
        if prior is None:
            raise ValueError(f"missing completed pilot coverage row: {window_id}")
        if str(old_support["support_status"]) != str(prior["density289_support_status"]) or int(old_support["valid_triplet_count"]) != int(prior["density289_triplet_count"]):
            raise ValueError(f"old support reconstruction mismatch: {window_id}")
        history_indices = np.asarray(old_support["history_array_indices"], dtype=np.int64)
        grouping = build_local_groups(sequence, history_indices, old_components=expected_components)
        grouping_checks = validate_grouping(grouping, sequence.track_ids, minimum_size=COMPONENT_CONFIG.minimum_size)
        if not grouping_checks["all_pass"]:
            raise ValueError(f"local grouping invariant failed: {window_id}: {grouping_checks}")
        local_support = build_local_support(sequence, window_start_s=float(row["interval_start_s"]), grouping=grouping)
        old_valid = str(old_support["support_status"]) == "VALID"
        local_valid = str(local_support["support_status"]) == "VALID"
        common = old_valid and local_valid
        if old_valid:
            old_support_count += 1
            old_source_ids.add(str(row["source_id"]))
        if local_valid:
            local_support_count += 1
            local_source_ids.add(str(row["source_id"]))
        if common:
            common_count += 1
            common_source_ids.add(str(row["source_id"]))
            examples.append(
                {
                    "window_id": window_id,
                    "pair_id": str(row["pair_id"]),
                    "source_id": str(row["source_id"]),
                    "role": str(row["role"]),
                    "kind": str(row["kind"]),
                    "anchor_fraction": float(row["anchor_fraction"]),
                    "old_triplets": support_arrays(old_support)["triplets"],
                    "local_triplets": support_arrays(local_support)["triplets"],
                }
            )
        compact_grouping = _compact_grouping(grouping)
        input_rows.append(_input_row(row, sequence, prefix, old_support, local_support, grouping, prior))
        support_rows.append(
            {
                "window_id": window_id,
                "source_id": str(row["source_id"]),
                "pair_id": str(row["pair_id"]),
                "role": str(row["role"]),
                "kind": str(row["kind"]),
                "common_support": bool(common),
                "old": json_ready_support(old_support),
                "local": json_ready_local_support(local_support),
                "local_grouping": compact_grouping,
                "grouping_checks": grouping_checks,
            }
        )
        for group in compact_grouping["groups"]:
            grouping_rows.append(
                {
                    "window_id": window_id,
                    "source_id": str(row["source_id"]),
                    "kind": str(row["kind"]),
                    "role": str(row["role"]),
                    "parent_component_id": int(group["parent_component_id"]),
                    "local_group_id": int(group["local_group_id"]),
                    "member_count": len(group["member_slots"]),
                    "member_slots": json.dumps(group["member_slots"]),
                    "track_ids": json.dumps(group["track_ids"]),
                    "retained": bool(group["retained"]),
                    "status": str(group["status"]),
                }
            )

    details = {
        "population": {
            "frozen_windows": len(manifest),
            "frozen_sources": len({str(row["source_id"]) for row in manifest}),
            "old_valid_support_windows": old_support_count,
            "local_valid_support_windows": local_support_count,
            "common_support_windows": common_count,
            "old_support_sources": sorted(old_source_ids),
            "local_support_sources": sorted(local_source_ids),
            "common_support_sources": sorted(common_source_ids),
            "all_source_ids": sorted({str(row["source_id"]) for row in manifest}),
            "source_manifest_sha256": _sha256(frontend_root / "window_manifest.json"),
            "frontend_results_sha256": _sha256(frontend_root / "frontend_results.json"),
            "completed_density_pilot_protocol_sha256": _sha256(pilot_input_root / "protocol.json"),
            "density_pilot_git_head": protocol.get("git_head"),
        },
        "input_rows": input_rows,
        "grouping_rows": grouping_rows,
    }
    return examples, support_rows, details


def _method_examples(examples: Sequence[Mapping[str, Any]], method: str) -> list[dict[str, Any]]:
    key = "old_triplets" if method == "old" else "local_triplets"
    output: list[dict[str, Any]] = []
    for example in examples:
        row = {key_name: value for key_name, value in example.items() if key_name not in {"old_triplets", "local_triplets"}}
        row["triplets"] = example[key]
        output.append(row)
    return output


def _auroc(positive: Iterable[float], negative: Iterable[float]) -> float | None:
    pos = np.asarray([float(item) for item in positive if item is not None and np.isfinite(item)], dtype=np.float64)
    neg = np.asarray([float(item) for item in negative if item is not None and np.isfinite(item)], dtype=np.float64)
    if pos.size == 0 or neg.size == 0:
        return None
    comparison = (pos[:, None] > neg[None, :]).astype(np.float64)
    comparison += 0.5 * (pos[:, None] == neg[None, :])
    return float(np.mean(comparison))


def _source_metrics(oof: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in oof:
        if row.get("kind") != "MANIP":
            continue
        for arm in PILOT_ARMS:
            score = row.get(f"{arm}_score")
            if score is not None:
                grouped[str(row["source_id"])][f"{arm}_{row['role']}"] .append(float(score))
    rows: list[dict[str, Any]] = []
    for source in sorted(grouped):
        row: dict[str, Any] = {"source_id": source}
        for arm in PILOT_ARMS:
            real = grouped[source].get(f"{arm}_real", [])
            fake = grouped[source].get(f"{arm}_fake", [])
            row[f"{arm}_real_count"] = len(real)
            row[f"{arm}_fake_count"] = len(fake)
            row[f"{arm}_auroc"] = _auroc(fake, real)
        for name, other in (("L_C-G_C", "G_C"), ("L_C-L_A", "L_A"), ("L_C-L_D", "L_D")):
            row[name] = row["L_C_auroc"] - row[f"{other}_auroc"] if row["L_C_auroc"] is not None and row[f"{other}_auroc"] is not None else None
        rows.append(row)
    return rows


def _paired_bootstrap(source_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    complete = [row for row in source_rows if all(row.get(f"{arm}_auroc") is not None for arm in PILOT_ARMS)]
    if not complete:
        return {"N": 0, "seed": BOOTSTRAP_SEED, "replicates": BOOTSTRAP_REPLICATES, "arms": {}, "gains": {}}
    values = {arm: np.asarray([float(row[f"{arm}_auroc"]) for row in complete], dtype=np.float64) for arm in PILOT_ARMS}
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    indices = rng.integers(0, len(complete), size=(BOOTSTRAP_REPLICATES, len(complete)))
    means = {arm: np.mean(value[indices], axis=1) for arm, value in values.items()}
    gains = {name: values["L_C"] - values[other] for name, other in (("L_C-G_C", "G_C"), ("L_C-L_A", "L_A"), ("L_C-L_D", "L_D"))}
    boot_gains = {name: means["L_C"] - means[other] for name, other in (("L_C-G_C", "G_C"), ("L_C-L_A", "L_A"), ("L_C-L_D", "L_D"))}
    return {
        "N": len(complete),
        "source_ids": [str(row["source_id"]) for row in complete],
        "seed": BOOTSTRAP_SEED,
        "replicates": BOOTSTRAP_REPLICATES,
        "arms": {arm: {"mean": float(np.mean(value)), "ci95": [float(np.percentile(means[arm], 2.5)), float(np.percentile(means[arm], 97.5))]} for arm, value in values.items()},
        "gains": {name: {"mean": float(np.mean(gains[name])), "ci95": [float(np.percentile(boot_gains[name], 2.5)), float(np.percentile(boot_gains[name], 97.5))]} for name in gains},
    }


def _summary(values: Iterable[float]) -> dict[str, Any]:
    array = np.asarray([float(item) for item in values if item is not None and np.isfinite(item)], dtype=np.float64)
    if array.size == 0:
        return {"N": 0, "mean": None, "median": None, "min": None, "max": None}
    return {"N": int(array.size), "mean": float(np.mean(array)), "median": float(np.median(array)), "min": float(np.min(array)), "max": float(np.max(array))}


def _training_weight_audit(examples: Sequence[Mapping[str, Any]], weights: np.ndarray) -> dict[str, Any]:
    values = np.asarray(weights, dtype=np.float64)
    if values.shape != (len(examples),):
        raise ValueError("training window weights do not match examples")
    groups: defaultdict[str, float] = defaultdict(float)
    counts: Counter[str] = Counter()
    for example, value in zip(examples, values):
        label = window_label(example)
        if label is None:
            raise ValueError("weight audit requires MANIP labels")
        key = f"{example['source_id']}::{'real' if label == 0 else 'fake'}"
        groups[key] += float(value)
        counts[key] += 1
    return {
        "source_class_window_counts": dict(sorted(counts.items())),
        "source_class_weight_sums": {key: float(value) for key, value in sorted(groups.items())},
        "window_weight_min": float(np.min(values)),
        "window_weight_max": float(np.max(values)),
        "window_weight_mean": float(np.mean(values)),
        "loss_weight_source": "Batch.window_weights passed to existing weighted_window_bce",
    }


def _control_descriptives(oof: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for arm in PILOT_ARMS:
        for role in ("real", "fake"):
            values = [row.get(f"{arm}_score") for row in oof if row.get("kind") == "CTRL" and row.get("role") == role]
            output[f"{arm}_{role}"] = _summary(values)
    return output


def _seed_stability(oof: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for arm in PILOT_ARMS:
        values = [row.get(f"{arm}_seed_std") for row in oof]
        output[arm] = {"window_count": sum(value is not None for value in values), "mean_seed_std": float(np.mean([value for value in values if value is not None])) if any(value is not None for value in values) else None}
    return output


def _status(bootstrap: Mapping[str, Any], complete_sources: int) -> dict[str, str]:
    if complete_sources < 12:
        value = "SUPPORT_INSUFFICIENT"
        return {"overall": value, "local_discrimination": value, "local_organization_gain": value, "local_temporal_gain": value}
    arms = bootstrap.get("arms", {})
    gains = bootstrap.get("gains", {})
    lc = arms.get("L_C", {}).get("ci95")
    lg = gains.get("L_C-G_C", {}).get("ci95")
    la = gains.get("L_C-L_A", {}).get("ci95")
    ld = gains.get("L_C-L_D", {}).get("ci95")
    discrimination = "L_C_DISCRIMINATION_SUPPORTED_IN_DEVELOPMENT" if lc and lc[0] > 0.5 else "L_C_DISCRIMINATION_NOT_ESTABLISHED"
    organization = "LOCAL_ORGANIZATION_GAIN_SUPPORTED_IN_DEVELOPMENT" if lg and lg[0] > 0 else "LOCAL_ORGANIZATION_GAIN_NOT_ESTABLISHED"
    temporal = "LOCAL_TEMPORAL_GAIN_SUPPORTED_IN_DEVELOPMENT" if lc and la and ld and lc[0] > 0.5 and la[0] > 0 and ld[0] > 0 else "LOCAL_TEMPORAL_GAIN_NOT_ESTABLISHED"
    return {"overall": temporal, "local_discrimination": discrimination, "local_organization_gain": organization, "local_temporal_gain": temporal}


def _protocol(population: Mapping[str, Any], *, head: str, input_hash: str, support_hash: str, grouping_hash: str, frontend_protocol: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "experiment": "V7 Historical Local-Organization Matched Temporal Structural Pilot",
        "research_question": "Whether history-local relation groups improve cross-source structural discrimination over the old component relation set under the same 289-point observations.",
        "git_head": head,
        "input": {
            "completed_density_pilot": str(PILOT_INPUT_ROOT),
            "frontend_root": str(FRONTEND_ROOT),
            "input_manifest": "manifests/input_manifest.json",
            "input_manifest_sha256": input_hash,
            "support_manifest": "manifests/window_support.json",
            "support_manifest_sha256": support_hash,
            "group_mapping": "manifests/local_group_mapping.csv",
            "group_mapping_sha256": grouping_hash,
            "population": "16 frozen source pairs, 192 unchanged windows, shared 289-point Depth Pro artifacts",
            "numeric_input": "same S(t), timestamp-aware v_minus/a, existing MLP; no label/source/path/generator/provenance input",
        },
        "history_and_local_groups": {
            "history": "timestamps_s < window_start_s + 0.5 seconds",
            "old_component_algorithm": "existing motion_coherent_components on history only",
            "component_config": {
                "max_initial_distance": float(COMPONENT_CONFIG.max_initial_distance),
                "max_relative_change": float(COMPONENT_CONFIG.max_relative_change),
                "minimum_size": int(COMPONENT_CONFIG.minimum_size),
                "minimum_overlap": int(COMPONENT_CONFIG.minimum_overlap),
            },
            "local_diameter_m": LOCAL_DIAMETER_M,
            "pair_distance": "dH_ij=max over jointly geometry-valid finite history frames of 3D distance",
            "merge_condition": "existing direct edge E_ij and every cross-group pair finite with dH<=0.30 m",
            "algorithm": "deterministic complete-linkage-style constrained agglomeration within each old component; stable track-ID tie break",
            "small_groups": "retained in mapping as SUPPORT_INSUFFICIENT_GROUP_SIZE and excluded from triplet support",
            "evaluation_identity": "groups and history scales are frozen; no evaluation re-grouping, deletion, or split",
        },
        "state_and_arms": {
            "state": "existing S(t)=[mean,std,p25,p75] over pair distances divided by per-group fixed history scale",
            "derivatives": "existing timestamp-aware v_minus and second-order a formulas",
            "groups": {"G_C": "old component + ORDERED_SECOND", "L_C": "local group + ORDERED_SECOND", "L_A": "local group + UNORDERED_STATE", "L_D": "local group + PERMUTED_SECOND"},
            "not_run": ["ORDERED_FIRST", "YOLO26 depth in main experiment", "new point sampling", "frontend rerun"],
        },
        "training": {
            "labels": {"MANIP real": 0, "MANIP fake": 1, "CTRL": "excluded from fitting; descriptive only"},
            "split": "source-disjoint LOSO; held-out source absent from support fitting, standardization, weights, and model parameters",
            "model": MODEL_CONFIG,
            "seeds": list(SEEDS),
            "weights": "existing source/class-balanced window weights and weighted BCE",
            "device": "CPU",
        },
        "evaluation": {
            "primary": "source-equal mean AUROC of L_C fake MANIP vs real MANIP",
            "contrasts": {"L_C-G_C": "local ordered second minus old component ordered second", "L_C-L_A": "local ordered second minus local unordered state", "L_C-L_D": "local ordered second minus local permuted second"},
            "bootstrap": {"unit": "complete source resampling with the same source indices for all four arms", "seed": BOOTSTRAP_SEED, "replicates": BOOTSTRAP_REPLICATES},
            "rules": {"support": "fewer than 12 complete source two-class rows -> SUPPORT_INSUFFICIENT", "local_discrimination": "L_C CI lower > 0.5", "local_organization_gain": "L_C-G_C CI lower > 0", "local_temporal_gain": "L_C CI lower > 0.5 and L_C-L_A/L_C-L_D lower CIs > 0"},
        },
        "frontend_protocol_sha256": _sha256(PILOT_INPUT_ROOT / "protocol.json"),
        "frontend_protocol_git_head": frontend_protocol.get("git_head"),
        "boundaries": ["development matched pilot only", "not full-video, sealed-test, pixel localization, or cross-generator guarantee", "complete-linkage is an existing clustering idea; no claim of algorithmic originality", "no ROI gate"],
    }


def _visual_member_slots(group: Mapping[str, Any], *, local: bool) -> tuple[int, ...]:
    """Read the member field used by one of the two fixed visualization schemas."""

    key = "member_slots" if local else "member_indices"
    return tuple(int(item) for item in group[key])


def _write_visualizations(output_root: Path, rows: Sequence[Mapping[str, Any]], support_by_id: Mapping[str, Mapping[str, Any]], group_by_id: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Write deterministic first-frame old/local group panels for the fixed 16 windows."""

    try:
        import cv2
    except ImportError:
        return [{"status": "UNAVAILABLE", "reason": "opencv is not available"}]
    selected = sorted(rows, key=lambda row: (str(row["source_id"]), 0 if row["kind"] == "MANIP" else 1, 0 if row["role"] == "real" else 1, str(row["window_id"])))
    visual_dir = output_root / "visualizations"
    visual_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    colors = [(0, 220, 255), (255, 100, 180), (100, 255, 120), (255, 170, 40), (180, 120, 255), (80, 220, 220)]
    for row in selected:
        window_id = str(row["window_id"])
        source = Path(str(row["video_path"]))
        if not source.is_file():
            records.append({"window_id": window_id, "status": "SOURCE_MISSING", "path": str(source)})
            continue
        try:
            decoded = decode_video(VideoSource(sample_id=f"local-org-{window_id}", source_video_id=str(row["source_id"]), source_locator=source), [int(row["frame_indices"][0])])
            image = cv2.cvtColor(np.asarray(decoded.frames[0].rgb), cv2.COLOR_RGB2BGR)
            sequence = support_by_id[window_id]["sequence"]
            old_support = support_by_id[window_id]["old"]
            grouping = group_by_id[window_id]
            local_groups = [item for item in grouping.get("groups", []) if bool(item.get("retained"))]

            def panel(groups: Sequence[Mapping[str, Any]], title: str, *, local: bool) -> np.ndarray:
                canvas = image.copy()
                assignment = {slot: int(index) for index, group in enumerate(groups) for slot in _visual_member_slots(group, local=local)}
                for slot, uv in enumerate(np.asarray(sequence.uv)[0]):
                    if not np.all(np.isfinite(uv)) or not bool(sequence.visibility[0, slot]):
                        continue
                    point = (int(round(float(uv[0]))), int(round(float(uv[1]))))
                    group_index = assignment.get(slot)
                    if group_index is None:
                        cv2.circle(canvas, point, 3, (140, 140, 140), -1)
                    else:
                        cv2.circle(canvas, point, 4, colors[group_index % len(colors)], -1)
                if local:
                    group_count = len(groups)
                    total_members = sum(len(group["member_slots"]) for group in groups)
                    unsupported = sum(not bool(group.get("retained")) for group in grouping.get("groups", []))
                else:
                    group_count = len(groups)
                    total_members = sum(len(group["member_indices"]) for group in groups)
                    unsupported = 0
                text = f"{title} groups={group_count} members={total_members} unsupported={unsupported}"
                cv2.putText(canvas, text, (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
                return canvas

            old_panel = panel(old_support.get("components", []), "OLD component", local=False)
            local_panel = panel(grouping.get("groups", []), "LOCAL history groups", local=True)
            output_image = np.concatenate([old_panel, local_panel], axis=1)
            path = visual_dir / f"{_safe(window_id)}.png"
            if not cv2.imwrite(str(path), output_image):
                raise RuntimeError("cv2.imwrite returned false")
            records.append({"window_id": window_id, "source_id": str(row["source_id"]), "kind": str(row["kind"]), "role": str(row["role"]), "source_frame_index": int(row["frame_indices"][0]), "timestamp_s": float(row["timestamps_s"][0]), "path": str(path), "status": "WRITTEN"})
        except Exception as exc:  # noqa: BLE001 - record one fixed-case visualization failure without changing the pilot
            records.append({"window_id": window_id, "status": "DECODE_FAILED", "error": f"{type(exc).__name__}: {exc}"})
    _write_json(output_root / "visualizations/manifest.json", {"selection": "all fixed earliest MANIP/CTRL real/fake rows in the 16-window manifest", "records": records})
    return records


def run(
    pilot_input_root: Path = PILOT_INPUT_ROOT,
    output_root: Path = OUTPUT_ROOT,
    *,
    preloaded: tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    started = time.perf_counter()
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(output_root)
    frontend_root, frontend_protocol = _resolve_frontend_root()
    if preloaded is None:
        examples, support_rows, details = load_population(pilot_input_root, frontend_root)
    else:
        examples, support_rows, details = preloaded
    output_root.mkdir(parents=True, exist_ok=True)
    head = _git_head()

    _write_json(output_root / "manifests/input_manifest.json", {"rows": details["input_rows"], "population": details["population"]})
    _write_json(output_root / "manifests/window_support.json", support_rows)
    _write_csv(output_root / "manifests/local_group_mapping.csv", details["grouping_rows"])
    input_hash = _sha256(output_root / "manifests/input_manifest.json")
    support_hash = _sha256(output_root / "manifests/window_support.json")
    grouping_hash = _sha256(output_root / "manifests/local_group_mapping.csv")
    protocol = _protocol(details["population"], head=head, input_hash=input_hash, support_hash=support_hash, grouping_hash=grouping_hash, frontend_protocol=frontend_protocol)
    _write_json(output_root / "protocol.json", protocol)
    _write_csv(output_root / "evaluation/coverage.csv", details["input_rows"])

    common_ids = {str(example["window_id"]) for example in examples}
    main_examples = [example for example in examples if window_label(example) is not None]
    method_examples = {"old": _method_examples(examples, "old"), "local": _method_examples(examples, "local")}
    examples_by_id = {str(example["window_id"]): example for example in examples}
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
    all_sources = sorted(details["population"]["all_source_ids"])
    fold_rows: list[dict[str, Any]] = []
    model_records: list[dict[str, Any]] = []
    for held_out in all_sources:
        heldout = [example for example in examples if str(example["source_id"]) == held_out]
        training = [example for example in main_examples if str(example["source_id"]) != held_out]
        if not heldout or not training:
            fold_rows.append({"held_out_source": held_out, "status": "NO_VALID_SUPPORT_WINDOWS", "held_out_window_count": len(heldout), "training_real_count": sum(window_label(item) == 0 for item in training), "training_fake_count": sum(window_label(item) == 1 for item in training)})
            continue
        validate_training_examples(training)
        support_by_arm: dict[str, tuple[Any, Any, Any]] = {}
        audit: dict[str, Any] | None = None
        for arm in PILOT_ARMS:
            method = "old" if arm == "G_C" else "local"
            training_method = [item for item in method_examples[method] if str(item["source_id"]) != held_out and window_label(item) is not None]
            held_method = [item for item in method_examples[method] if str(item["source_id"]) == held_out]
            raw_batch, observation_weights = build_batch(training_method, ARM_TO_MODEL[arm], observation_weights=True)
            if observation_weights is None:
                raise RuntimeError("existing batch builder did not return observation weights")
            standardizer = fit_weighted_standardizer(raw_batch.inputs, observation_weights)
            train_batch, _ = build_batch(training_method, ARM_TO_MODEL[arm], standardizer=standardizer, observation_weights=True)
            held_batch, _ = build_batch(held_method, ARM_TO_MODEL[arm], standardizer=standardizer)
            current_audit = _training_weight_audit(training_method, train_batch.window_weights)
            if audit is None:
                audit = current_audit
            elif not np.array_equal(train_batch.window_weights, support_by_arm[next(iter(support_by_arm))][1].window_weights):
                raise RuntimeError("training window weights differ between matched arms")
            support_by_arm[arm] = (standardizer, train_batch, held_batch)
        if audit is None:
            raise RuntimeError("training weight audit missing")
        for arm in PILOT_ARMS:
            standardizer, train_batch, held_batch = support_by_arm[arm]
            for seed in SEEDS:
                model, fit_info = train_model(train_batch, seed=seed)
                scores = score_model(model, held_batch)
                for example, score in zip([item for item in method_examples["old" if arm == "G_C" else "local"] if str(item["source_id"]) == held_out], scores):
                    oof_by_id[str(example["window_id"])][f"{arm}_score_seed_{seed}"] = float(score)
                model_records.append({"held_out_source": held_out, "arm": arm, "model_arm": ARM_TO_MODEL[arm], "seed": seed, "training_source_count": len({str(item["source_id"]) for item in training}), "training_real_count": sum(window_label(item) == 0 for item in training), "training_fake_count": sum(window_label(item) == 1 for item in training), "held_out_window_count": len(held_method), "training_triplet_count": train_batch.n_triplets, "held_out_triplet_count": held_batch.n_triplets, "zero_variance_dimensions": list(standardizer.zero_variance_dimensions), "fit": fit_info, "model": serialize_model(model, standardizer)})
                del model
        fold_rows.append({"held_out_source": held_out, "status": "SCORED", "held_out_window_count": len(heldout), "training_source_count": len({str(item["source_id"]) for item in training}), "training_real_count": sum(window_label(item) == 0 for item in training), "training_fake_count": sum(window_label(item) == 1 for item in training), "held_out_main_windows": sum(window_label(item) is not None for item in heldout), "old_support_windows": sum(str(item["source_id"]) == held_out for item in examples), "local_support_windows": sum(str(item["source_id"]) == held_out for item in examples), **(audit or {})})
        print(f"local-organization fold {held_out} complete", flush=True)

    oof_rows: list[dict[str, Any]] = []
    for row in oof_by_id.values():
        for arm in PILOT_ARMS:
            values = [row.get(f"{arm}_score_seed_{seed}") for seed in SEEDS]
            finite = [float(item) for item in values if item is not None and np.isfinite(item)]
            row[f"{arm}_score"] = float(np.mean(finite)) if len(finite) == len(SEEDS) else None
            row[f"{arm}_seed_std"] = float(np.std(finite)) if len(finite) == len(SEEDS) else None
        oof_rows.append(row)
    _write_csv(output_root / "scores/oof_window_scores.csv", oof_rows)
    _write_csv(output_root / "evaluation/fold_support.csv", fold_rows)
    _write_json(output_root / "models/fold_models.json", {"model_config": MODEL_CONFIG, "seeds": list(SEEDS), "arms": PILOT_ARMS, "arm_to_model": ARM_TO_MODEL, "folds": model_records})

    source_rows = _source_metrics(oof_rows)
    bootstrap = _paired_bootstrap(source_rows)
    _write_csv(output_root / "evaluation/per_source_metrics.csv", source_rows)
    complete_sources = [row for row in source_rows if all(row.get(f"{arm}_auroc") is not None for arm in PILOT_ARMS)]
    statuses = _status(bootstrap, len(complete_sources))
    pooled = {arm: {"fake_MANIP_vs_real_MANIP": _auroc([row.get(f"{arm}_score") for row in oof_rows if row.get("kind") == "MANIP" and row.get("role") == "fake"], [row.get(f"{arm}_score") for row in oof_rows if row.get("kind") == "MANIP" and row.get("role") == "real"])} for arm in PILOT_ARMS}
    source_by_id = {str(row["source_id"]): row for row in source_rows}
    coverage_by_source: list[dict[str, Any]] = []
    for source in all_sources:
        source_inputs = [row for row in details["input_rows"] if str(row["source_id"]) == source]
        common = [row for row in source_inputs if bool(next(item for item in support_rows if item["window_id"] == row["window_id"])["common_support"])]
        coverage_by_source.append({"source_id": source, "old_valid_windows": sum(str(row["old_support_status"]) == "VALID" for row in source_inputs), "local_valid_windows": sum(str(row["local_support_status"]) == "VALID" for row in source_inputs), "common_windows": len(common), "main_common_real": sum(row["kind"] == "MANIP" and row["role"] == "real" and row["window_id"] in common_ids for row in source_inputs), "main_common_fake": sum(row["kind"] == "MANIP" and row["role"] == "fake" and row["window_id"] in common_ids for row in source_inputs), "source_auroc_complete": source in source_by_id and all(source_by_id[source].get(f"{arm}_auroc") is not None for arm in PILOT_ARMS)})
    _write_csv(output_root / "evaluation/coverage_by_source.csv", coverage_by_source)

    support_rows_by_id = {str(row["window_id"]): row for row in support_rows}
    visualization_support = {}
    visualization_groups = {}
    for row in details["input_rows"]:
        item = support_rows_by_id[str(row["window_id"])]
        # Visualization only needs the original sequence and compact group rows.
        prefix = Path(str(row["particle_prefix"]))
        visualization_support[str(row["window_id"])] = {"sequence": load_particle_sequence(prefix), "old": item["old"]}
        visualization_groups[str(row["window_id"])] = item["local_grouping"]
    visualization_rows = [row for row in json.loads((frontend_root / "window_manifest.json").read_text(encoding="utf-8")) if str(row["window_id"]) in support_rows_by_id]
    visualization = _write_visualizations(output_root, visualization_rows, visualization_support, visualization_groups)
    _write_json(output_root / "evaluation/visualization_summary.json", {"records": visualization, "fixed_selection": "manifest order by source, MANIP/CTRL, real/fake; no score-based selection"})

    summary = {
        "experiment": "V7 Historical Local-Organization Matched Temporal Structural Pilot",
        "statuses": statuses,
        "git_head": head,
        "population": details["population"],
        "coverage": {"frozen_windows": len(details["input_rows"]), "old_valid_support_windows": details["population"]["old_valid_support_windows"], "local_valid_support_windows": details["population"]["local_valid_support_windows"], "common_support_windows": details["population"]["common_support_windows"], "source_complete_two_class_count": len(complete_sources), "invalid_reason_counts": dict(Counter(str(item["reason"]) for row in support_rows for item in row["local"].get("invalid_reasons", [])))},
        "training": {"arms": PILOT_ARMS, "arm_to_model": ARM_TO_MODEL, "model_records": len(model_records), "seeds": list(SEEDS), "device": "CPU", "main_common_windows": sum(window_label(item) is not None for item in examples), "labels": {"MANIP real": 0, "MANIP fake": 1, "CTRL": "excluded"}},
        "source_level_metrics": source_rows,
        "source_bootstrap": bootstrap,
        "pooled_auroc_auxiliary": pooled,
        "control_descriptives": _control_descriptives(oof_rows),
        "seed_stability": _seed_stability(oof_rows),
        "coverage_by_source": "evaluation/coverage_by_source.csv",
        "fold_support": "evaluation/fold_support.csv",
        "oof_scores": "scores/oof_window_scores.csv",
        "support_manifest": "manifests/window_support.json",
        "local_group_mapping": "manifests/local_group_mapping.csv",
        "visualizations": "evaluation/visualization_summary.json",
        "protocol": "protocol.json",
        "paired_reference_used": False,
        "frontend_rerun": False,
        "formal_src_modified": False,
        "full_video": False,
        "sealed_test": False,
        "roi_gate": False,
        "elapsed_s": time.perf_counter() - started,
    }
    _write_json(output_root / "evaluation/summary.json", summary)
    _write_json(output_root / "run_summary.json", {"statuses": statuses, "artifact_root": str(output_root), "summary": "evaluation/summary.json", "common_support_windows": details["population"]["common_support_windows"], "source_complete_two_class_count": len(complete_sources)})
    return summary


def main() -> None:
    import argparse

    global FRONTEND_ROOT
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot-input-root", type=Path, default=PILOT_INPUT_ROOT)
    parser.add_argument("--frontend-root", type=Path, default=FRONTEND_ROOT)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    args = parser.parse_args()
    FRONTEND_ROOT = args.frontend_root
    print(json.dumps(run(args.pilot_input_root, args.output_root), indent=2, sort_keys=True, ensure_ascii=False))


if __name__ == "__main__":
    main()
