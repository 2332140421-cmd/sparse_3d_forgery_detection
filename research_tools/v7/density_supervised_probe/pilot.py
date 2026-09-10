"""Source-disjoint supervised pilot for the frozen V7 density comparison.

The module consumes already materialized 64/289 ``ParticleSequence`` artifacts
and reuses the established local structural temporal model.  It does not
change the formal ``src`` detection chain, component rule, state definition,
or training implementation.  Its only new responsibility is to keep the
matched-density population and the predeclared three-arm comparisons
auditable in one independent artifact directory.
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
from research_tools.v7.local_structural_temporal_probe.model import (
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
from research_tools.v7.local_structural_temporal_probe.representation import (
    ARM_NAMES,
    build_window_support,
    support_arrays,
)


DATA_ROOT = Path("/root/autodl-tmp/data/sparse_3d_forgery_detection")
BASE_ROOT = DATA_ROOT / "derived/v7_activityforensics_paired_second_order_pilot_v1"
FRONTEND_ROOT = DATA_ROOT / "derived/v7_activityforensics_density_matched_frontend_v1"
OUTPUT_ROOT = DATA_ROOT / "derived/v7_activityforensics_density_supervised_pilot_v1"
SOURCE_DENSITY_ROOT = DATA_ROOT / "derived/v7_activityforensics_observation_density_diagnostic_v1"
DENSITIES = ("density64", "density289")
PILOT_ARMS = ("UNORDERED_STATE", "ORDERED_SECOND", "PERMUTED_SECOND")
SEEDS = (20260909, 20260910, 20260911)
BOOTSTRAP_SEED = 20260909
BOOTSTRAP_REPLICATES = 10_000


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False, default=_json_default)
        + "\n",
        encoding="utf-8",
    )


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
    repo = Path(__file__).resolve().parents[3]
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()


def _safe_window_id(window_id: str) -> str:
    return "".join(char if char.isalnum() or char in "._-" else "_" for char in window_id)


def _front_prefix(frontend_root: Path, window_id: str, density: str) -> Path:
    return frontend_root / "particles" / f"{_safe_window_id(window_id)}__{density}"


def _base_prefix(window_id: str) -> Path:
    return BASE_ROOT / "particles" / window_id.replace("::", "__")


def _density_prefix(window_id: str, density: str, frontend_root: Path) -> Path:
    return _front_prefix(frontend_root, window_id, density)


def _load_manifest(frontend_root: Path) -> list[dict[str, Any]]:
    path = frontend_root / "window_manifest.json"
    rows = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(rows, list) or len(rows) != 192:
        raise ValueError(f"density frontend manifest must contain 192 rows: {path}")
    return [dict(row) for row in rows]


def _load_front_results(frontend_root: Path) -> dict[tuple[str, str], dict[str, Any]]:
    path = frontend_root / "frontend_results.json"
    values = json.loads(path.read_text(encoding="utf-8")).get("results", [])
    result: dict[tuple[str, str], dict[str, Any]] = {}
    for row in values:
        key = (str(row["window_id"]), str(row["density"]))
        result[key] = dict(row)
    return result


def load_density_population(frontend_root: Path = FRONTEND_ROOT) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Load all windows, retaining only compact support arrays in memory."""

    manifest = _load_manifest(frontend_root)
    results = _load_front_results(frontend_root)
    examples: list[dict[str, Any]] = []
    coverage: list[dict[str, Any]] = []
    for row in manifest:
        by_density: dict[str, dict[str, Any]] = {}
        statuses: dict[str, str] = {}
        for density in DENSITIES:
            result = results.get((str(row["window_id"]), density))
            if result is None:
                raise ValueError(f"missing frontend result: {row['window_id']} {density}")
            prefix = Path(result["sequence_prefix"])
            if not prefix.with_suffix(".npz").is_file():
                raise FileNotFoundError(prefix.with_suffix(".npz"))
            sequence = load_particle_sequence(prefix)
            support = build_window_support(sequence, window_start_s=float(row["interval_start_s"]))
            prepared = support_arrays(support)
            by_density[density] = {
                "triplets": prepared["triplets"],
                "result_key": result["result_key"],
                "sequence_schema_version": str(sequence.schema_version),
                "sequence_frame_count": int(sequence.num_frames),
                "sequence_track_count": int(sequence.num_tracks),
                "geometry_valid_fraction": float(np.mean(sequence.geometry_validity)),
                "visible_fraction": float(np.mean(sequence.visibility)),
                "valid_triplet_count": int(support["valid_triplet_count"]),
                "component_count": int(support["component_count"]),
            }
            statuses[density] = str(support["support_status"])
        common = statuses["density64"] == "VALID" and statuses["density289"] == "VALID"
        coverage.append(
            {
                "window_id": str(row["window_id"]),
                "source_id": str(row["source_id"]),
                "kind": str(row["kind"]),
                "role": str(row["role"]),
                "density64_support_status": statuses["density64"],
                "density289_support_status": statuses["density289"],
                "density64_triplet_count": by_density["density64"]["valid_triplet_count"],
                "density289_triplet_count": by_density["density289"]["valid_triplet_count"],
                "common_support": common,
                "label": window_label(row),
            }
        )
        if common:
            examples.append(
                {
                    "window_id": str(row["window_id"]),
                    "pair_id": str(row["pair_id"]),
                    "source_id": str(row["source_id"]),
                    "role": str(row["role"]),
                    "kind": str(row["kind"]),
                    "anchor_fraction": float(row["anchor_fraction"]),
                    "density": by_density,
                }
            )
    details = {
        "population": {
            "frozen_windows": len(manifest),
            "frozen_sources": len({str(row["source_id"]) for row in manifest}),
            "common_support_windows": len(examples),
            "common_support_sources": sorted({str(row["source_id"]) for row in examples}),
            "source_manifest_sha256": _sha256(BASE_ROOT / "manifests/window_manifest.json"),
            "frontend_results_sha256": _sha256(frontend_root / "frontend_results.json"),
        },
        "coverage": coverage,
    }
    return examples, details


def _examples_for_density(examples: Sequence[Mapping[str, Any]], density: str) -> list[dict[str, Any]]:
    values: list[dict[str, Any]] = []
    for example in examples:
        row = {key: value for key, value in example.items() if key != "density"}
        row["triplets"] = example["density"][density]["triplets"]
        values.append(row)
    return values


def _auroc(positive: Iterable[float], negative: Iterable[float]) -> float | None:
    pos = np.asarray([float(value) for value in positive if np.isfinite(value)], dtype=np.float64)
    neg = np.asarray([float(value) for value in negative if np.isfinite(value)], dtype=np.float64)
    if pos.size == 0 or neg.size == 0:
        return None
    comparisons = (pos[:, None] > neg[None, :]).astype(np.float64)
    comparisons += 0.5 * (pos[:, None] == neg[None, :])
    return float(np.mean(comparisons))


def _fit_weight_audit(examples: Sequence[Mapping[str, Any]], weights: np.ndarray) -> dict[str, Any]:
    validate_training_examples(examples)
    values = np.asarray(weights, dtype=np.float64)
    if values.shape != (len(examples),):
        raise ValueError("window weights must match examples")
    groups: defaultdict[str, float] = defaultdict(float)
    counts: Counter[str] = Counter()
    for example, value in zip(examples, values):
        label = int(window_label(example))
        key = f"{example['source_id']}::{'real' if label == 0 else 'fake'}"
        groups[key] += float(value)
        counts[key] += 1
    return {
        "source_class_window_counts": dict(sorted(counts.items())),
        "source_class_weight_sums": {key: float(value) for key, value in sorted(groups.items())},
        "window_weight_min": float(np.min(values)),
        "window_weight_max": float(np.max(values)),
        "window_weight_mean": float(np.mean(values)),
        "weights_expected": "equal total mass per source/class, normalized mean one",
    }


def _source_metric_rows(oof: Sequence[Mapping[str, Any]], densities: Sequence[str] = DENSITIES) -> list[dict[str, Any]]:
    grouped: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in oof:
        if row.get("kind") != "MANIP":
            continue
        source = str(row["source_id"])
        for density in densities:
            for arm in PILOT_ARMS:
                key = f"{density}_{arm}_score"
                if row.get(key) is not None:
                    grouped[source][f"{density}_{arm}_{row['role']}"] .append(float(row[key]))
    rows: list[dict[str, Any]] = []
    for source in sorted(grouped):
        row: dict[str, Any] = {"source_id": source}
        for density in densities:
            for arm in PILOT_ARMS:
                real = grouped[source].get(f"{density}_{arm}_real", [])
                fake = grouped[source].get(f"{density}_{arm}_fake", [])
                row[f"{density}_{arm}_real_count"] = len(real)
                row[f"{density}_{arm}_fake_count"] = len(fake)
                row[f"{density}_{arm}_auroc"] = _auroc(fake, real)
        c = f"{densities[1]}_ORDERED_SECOND_auroc"
        row["C289-A289"] = row[c] - row[f"{densities[1]}_UNORDERED_STATE_auroc"] if row[c] is not None and row[f"{densities[1]}_UNORDERED_STATE_auroc"] is not None else None
        d289 = row.get(f"{densities[1]}_PERMUTED_SECOND_auroc")
        row["C289-D289"] = row[c] - d289 if row[c] is not None and d289 is not None else None
        c64 = row.get(f"{densities[0]}_ORDERED_SECOND_auroc")
        row["C289-C64"] = row[c] - c64 if row[c] is not None and c64 is not None else None
        rows.append(row)
    return rows


def _aggregate_source_metrics(source_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Summarize all six density/arm results with source as the unit."""

    output: dict[str, Any] = {}
    for density in DENSITIES:
        for arm in PILOT_ARMS:
            name = f"{density}_{arm}_auroc"
            values = [float(row[name]) for row in source_rows if row.get(name) is not None]
            output[name] = {
                "source_count": len(values),
                "source_equal_mean": float(np.mean(values)) if values else None,
                "source_equal_std": float(np.std(values)) if values else None,
            }
    return output


def _pooled_metrics(oof: Sequence[Mapping[str, Any]]) -> dict[str, float | None]:
    output: dict[str, float | None] = {}
    for density in DENSITIES:
        for arm in PILOT_ARMS:
            key = f"{density}_{arm}_score"
            fake = [row[key] for row in oof if row.get("kind") == "MANIP" and row.get("role") == "fake" and row.get(key) is not None]
            real = [row[key] for row in oof if row.get("kind") == "MANIP" and row.get("role") == "real" and row.get(key) is not None]
            output[f"{density}_{arm}"] = _auroc(fake, real)
    return output


def _control_descriptives(oof: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for density in DENSITIES:
        for arm in PILOT_ARMS:
            key = f"{density}_{arm}_score"
            for role in ("real", "fake"):
                values = [float(row[key]) for row in oof if row.get("kind") == "CTRL" and row.get("role") == role and row.get(key) is not None and np.isfinite(row[key])]
                output[f"{density}_{arm}_{role}"] = {
                    "count": len(values),
                    "median": float(np.median(values)) if values else None,
                    "mean": float(np.mean(values)) if values else None,
                }
    return output


def _paired_bootstrap(source_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    metric_names = [
        "density289_ORDERED_SECOND_auroc",
        "density289_UNORDERED_STATE_auroc",
        "density289_PERMUTED_SECOND_auroc",
        "density64_ORDERED_SECOND_auroc",
    ]
    complete = [row for row in source_rows if all(row.get(name) is not None for name in metric_names)]
    if not complete:
        return {"N": 0, "seed": BOOTSTRAP_SEED, "replicates": BOOTSTRAP_REPLICATES, "arms": {}, "gains": {}}
    values = {name: np.asarray([float(row[name]) for row in complete], dtype=np.float64) for name in metric_names}
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    indices = rng.integers(0, len(complete), size=(BOOTSTRAP_REPLICATES, len(complete)))
    means = {name: np.mean(value[indices], axis=1) for name, value in values.items()}
    gains = {
        "C289-A289": values[metric_names[0]] - values[metric_names[1]],
        "C289-D289": values[metric_names[0]] - values[metric_names[2]],
        "C289-C64": values[metric_names[0]] - values[metric_names[3]],
    }
    boot_gains = {
        "C289-A289": means[metric_names[0]] - means[metric_names[1]],
        "C289-D289": means[metric_names[0]] - means[metric_names[2]],
        "C289-C64": means[metric_names[0]] - means[metric_names[3]],
    }
    return {
        "N": len(complete),
        "source_ids": [str(row["source_id"]) for row in complete],
        "seed": BOOTSTRAP_SEED,
        "replicates": BOOTSTRAP_REPLICATES,
        "arms": {
            name: {"mean": float(np.mean(value)), "ci95": [float(np.percentile(means[name], 2.5)), float(np.percentile(means[name], 97.5))]}
            for name, value in values.items()
        },
        "gains": {
            name: {"mean": float(np.mean(gains[name])), "ci95": [float(np.percentile(boot_gains[name], 2.5)), float(np.percentile(boot_gains[name], 97.5))]}
            for name in gains
        },
    }


def _predeclared_status(bootstrap: Mapping[str, Any], complete_sources: int) -> dict[str, str]:
    if complete_sources < 12:
        value = "MEASUREMENT_SUPPORT_INSUFFICIENT"
        return {"overall": value, "classification": value, "temporal_organization": value, "density_gain": value}
    c = bootstrap.get("arms", {}).get("density289_ORDERED_SECOND_auroc", {}).get("ci95")
    ca = bootstrap.get("gains", {}).get("C289-A289", {}).get("ci95")
    cd = bootstrap.get("gains", {}).get("C289-D289", {}).get("ci95")
    cg = bootstrap.get("gains", {}).get("C289-C64", {}).get("ci95")
    classification = "C289_DISCRIMINATION_SUPPORTED_IN_PILOT" if c and c[0] > 0.5 else "C289_DISCRIMINATION_NOT_ESTABLISHED"
    temporal = "C289_TEMPORAL_ORGANIZATION_SUPPORTED_IN_PILOT" if c and ca and cd and c[0] > 0.5 and ca[0] > 0 and cd[0] > 0 else "C289_TEMPORAL_ORGANIZATION_NOT_ESTABLISHED"
    density_gain = "DENSITY_289_GAIN_SUPPORTED_IN_PILOT" if cg and cg[0] > 0 else "DENSITY_289_GAIN_NOT_ESTABLISHED"
    return {"overall": temporal, "classification": classification, "temporal_organization": temporal, "density_gain": density_gain}


def _protocol(details: Mapping[str, Any], head: str) -> dict[str, Any]:
    return {
        "experiment": "V7 matched 64/289-point supervised structural pilot",
        "git_head": head,
        "question": "Whether 289 nested observations add cross-source supervised information and temporal organization under the frozen three-arm model.",
        "input": {
            "frozen_population": "16 source pairs, 192 windows",
            "common_population": "only windows with valid support in both densities; determined before labels or scores",
            "densities": {"density64": 64, "density289": 289},
            "state": "existing S(t)=[mean,std,p25,p75] with existing fixed component scale",
            "arms": list(PILOT_ARMS),
            "excluded": ["paired differences", "source/generator/path", "NSI/dynamics/quality", "control labels"],
        },
        "labels": {"MANIP real": 0, "MANIP fake": 1, "CTRL": "excluded from fitting; descriptive scores only"},
        "training": {
            "split": "source-disjoint LOSO",
            "weights": "source/class balanced Batch.window_weights, mean normalized to one",
            "standardization": "fold training real+fake only, weighted; zero-variance scale one",
            "model": MODEL_CONFIG,
            "seeds": list(SEEDS),
            "arms": list(PILOT_ARMS),
            "epochs": MODEL_CONFIG["epochs"],
            "device": "CPU",
        },
        "evaluation": {
            "primary": "source-equal mean AUROC of C289 fake MANIP vs real MANIP",
            "contrasts": {"C289-A289": "ORDERED_SECOND289 - UNORDERED_STATE289", "C289-D289": "ORDERED_SECOND289 - PERMUTED_SECOND289", "C289-C64": "ORDERED_SECOND289 - ORDERED_SECOND64"},
            "bootstrap": {"unit": "complete source resampling; paired metrics use same indices", "seed": BOOTSTRAP_SEED, "replicates": BOOTSTRAP_REPLICATES},
            "state_rules": {
                "classification": "C289 lower CI > 0.5",
                "temporal_organization": "C289 lower CI > 0.5 and C289-A289/C289-D289 lower CIs > 0",
                "density_gain": "C289-C64 lower CI > 0",
            },
        },
        "coverage": details["population"],
        "boundaries": ["development pilot only", "not full-video or sealed-test", "no ROI required", "no model or frontend search"],
    }


def run_pilot(frontend_root: Path = FRONTEND_ROOT, output_root: Path = OUTPUT_ROOT) -> dict[str, Any]:
    started = time.perf_counter()
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(output_root)
    examples, details = load_density_population(frontend_root)
    output_root.mkdir(parents=True, exist_ok=True)
    head = _git_head()
    main_examples = [example for example in examples if window_label(example) is not None]
    common_ids = {str(example["window_id"]) for example in examples}
    _write_json(output_root / "manifests/coverage.json", details)
    _write_json(output_root / "protocol.json", _protocol(details, head))
    _write_csv(output_root / "manifests/coverage.csv", details["coverage"])

    oof: dict[str, dict[str, Any]] = {}
    for example in examples:
        oof[str(example["window_id"])] = {
            "window_id": str(example["window_id"]),
            "pair_id": str(example["pair_id"]),
            "source_id": str(example["source_id"]),
            "role": str(example["role"]),
            "kind": str(example["kind"]),
            "label": window_label(example),
        }
    fold_rows: list[dict[str, Any]] = []
    model_records: list[dict[str, Any]] = []
    all_sources = sorted({str(row["source_id"]) for row in details["coverage"]})
    for held_out in all_sources:
        heldout = [example for example in main_examples if str(example["source_id"]) == held_out]
        training = [example for example in main_examples if str(example["source_id"]) != held_out]
        if not heldout:
            fold_rows.append({"held_out_source": held_out, "status": "NO_VALID_MAIN_SUPPORT", "training_real_count": sum(window_label(item) == 0 for item in training), "training_fake_count": sum(window_label(item) == 1 for item in training)})
            continue
        validate_training_examples(training)
        fold_row: dict[str, Any] = {
            "held_out_source": held_out,
            "status": "SCORED",
            "training_source_count": len({str(item["source_id"]) for item in training}),
            "training_real_count": sum(window_label(item) == 0 for item in training),
            "training_fake_count": sum(window_label(item) == 1 for item in training),
            "held_out_main_count": len(heldout),
        }
        for density in DENSITIES:
            density_examples = _examples_for_density(examples, density)
            training_density = [item for item in density_examples if str(item["source_id"]) != held_out and window_label(item) is not None]
            held_density = [item for item in density_examples if str(item["source_id"]) == held_out]
            for arm in PILOT_ARMS:
                raw_batch, observation_weights = build_batch(training_density, arm, observation_weights=True)
                if observation_weights is None:
                    raise RuntimeError("missing observation weights")
                standardizer = fit_weighted_standardizer(raw_batch.inputs, observation_weights)
                train_batch, _ = build_batch(training_density, arm, standardizer=standardizer, observation_weights=True)
                held_batch, _ = build_batch(held_density, arm, standardizer=standardizer)
                audit = _fit_weight_audit(training_density, train_batch.window_weights)
                fold_row[f"{density}_{arm}_weight_audit"] = audit
                for seed in SEEDS:
                    model, fit_info = train_model(train_batch, seed=seed)
                    scores = score_model(model, held_batch)
                    for item, score in zip(held_density, scores):
                        oof[str(item["window_id"])][f"{density}_{arm}_score_seed_{seed}"] = float(score)
                    model_records.append(
                        {
                            "held_out_source": held_out,
                            "density": density,
                            "arm": arm,
                            "seed": seed,
                            "training_real_count": sum(window_label(item) == 0 for item in training_density),
                            "training_fake_count": sum(window_label(item) == 1 for item in training_density),
                            "held_out_window_count": len(held_density),
                            "training_triplet_count": train_batch.n_triplets,
                            "held_out_triplet_count": held_batch.n_triplets,
                            "zero_variance_dimensions": list(standardizer.zero_variance_dimensions),
                            "fit": fit_info,
                            "model": serialize_model(model, standardizer),
                        }
                    )
                    del model
        fold_rows.append(fold_row)

    oof_rows: list[dict[str, Any]] = []
    for row in oof.values():
        for density in DENSITIES:
            for arm in PILOT_ARMS:
                values = [row.get(f"{density}_{arm}_score_seed_{seed}") for seed in SEEDS]
                finite = [float(value) for value in values if value is not None and np.isfinite(value)]
                row[f"{density}_{arm}_score"] = float(np.mean(finite)) if len(finite) == len(SEEDS) else None
                row[f"{density}_{arm}_seed_std"] = float(np.std(finite)) if len(finite) == len(SEEDS) else None
        oof_rows.append(row)
    _write_csv(output_root / "scores/oof_window_scores.csv", oof_rows)
    _write_csv(output_root / "evaluation/fold_support.csv", fold_rows)
    _write_json(output_root / "models/fold_models.json", {"model_config": MODEL_CONFIG, "seeds": list(SEEDS), "arms": list(PILOT_ARMS), "folds": model_records})
    source_rows = _source_metric_rows(oof_rows)
    bootstrap = _paired_bootstrap(source_rows)
    _write_csv(output_root / "evaluation/per_source_metrics.csv", source_rows)
    complete_sources = [row for row in source_rows if all(row.get(name) is not None for name in ("density289_ORDERED_SECOND_auroc", "density289_UNORDERED_STATE_auroc", "density289_PERMUTED_SECOND_auroc", "density64_ORDERED_SECOND_auroc"))]
    status = _predeclared_status(bootstrap, len(complete_sources))
    aggregate = _aggregate_source_metrics(source_rows)
    pooled = _pooled_metrics(oof_rows)
    control = _control_descriptives(oof_rows)
    seed_stability: dict[str, Any] = {}
    for density in DENSITIES:
        for arm in PILOT_ARMS:
            values = [float(row[f"{density}_{arm}_seed_std"]) for row in oof_rows if row.get(f"{density}_{arm}_seed_std") is not None]
            seed_stability[f"{density}_{arm}"] = {"window_count": len(values), "mean_seed_std": float(np.mean(values)) if values else None}
    fit_summary: dict[str, Any] = {}
    for density in DENSITIES:
        for arm in PILOT_ARMS:
            values = [float(record["fit"]["final_loss"]) for record in model_records if record["density"] == density and record["arm"] == arm]
            fit_summary[f"{density}_{arm}"] = {"model_count": len(values), "final_loss_mean": float(np.mean(values)) if values else None, "final_loss_min": float(np.min(values)) if values else None}
    summary = {
        "experiment": "V7 matched 64/289-point supervised structural pilot",
        "status": status,
        "git_head": head,
        "coverage": {
            **details["population"],
            "common_main_windows": len(main_examples),
            "common_source_complete_two_class": len(complete_sources),
            "common_window_ids": sorted(common_ids),
            "by_support_status": dict(
                Counter(
                    f"{row['density64_support_status']}__{row['density289_support_status']}"
                    for row in details["coverage"]
                )
            ),
        },
        "training": {
            "arms": list(PILOT_ARMS),
            "densities": {"density64": 64, "density289": 289},
            "model_records": len(model_records),
            "seeds": list(SEEDS),
            "device": "CPU",
            "labels": {"MANIP real": 0, "MANIP fake": 1, "CTRL": "excluded"},
        },
        "source_metrics": source_rows,
        "source_equal_aggregate": aggregate,
        "pooled_auroc": pooled,
        "control_descriptives": control,
        "seed_stability": seed_stability,
        "fit_summary": fit_summary,
        "source_bootstrap": bootstrap,
        "oof_scores": "scores/oof_window_scores.csv",
        "fold_support": "evaluation/fold_support.csv",
        "fold_models": "models/fold_models.json",
        "protocol": "protocol.json",
        "elapsed_s": time.perf_counter() - started,
        "paired_reference_used": False,
        "formal_src_modified": False,
        "full_video": False,
        "sealed_test": False,
    }
    _write_json(output_root / "evaluation/summary.json", summary)
    _write_json(output_root / "run_summary.json", {"status": status, "artifact_root": str(output_root), "summary": "evaluation/summary.json", "common_support_windows": len(examples), "source_complete_two_class_count": len(complete_sources)})
    return summary


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--frontend-root", type=Path, default=FRONTEND_ROOT)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    args = parser.parse_args()
    print(json.dumps(run_pilot(args.frontend_root, args.output_root), indent=2, sort_keys=True, ensure_ascii=False))


if __name__ == "__main__":
    main()
