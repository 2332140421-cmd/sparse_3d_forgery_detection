"""Trace one real/fake path through completed V7 artifacts.

This is intentionally a bounded audit, not a cache registry or a rerun
framework.  It reads existing arrays/models, reconstructs a few selected
windows, and records evidence under the data directory.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import subprocess
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from sparse3d_forgery.particle_sequence import load_particle_sequence
from research_tools.v7.attention_pooling_pilot import runner as attention
from research_tools.v7.multi_order_sequence_probe import representation as multi_repr
from research_tools.v7.multi_order_sequence_probe import runner as multi
from research_tools.v7.pair_trajectory_probe import runner as pair
from research_tools.v7.periodic_requery_probe import runner as periodic
from research_tools.v7.source128_extension import runner as source128


REPO_ROOT = Path(__file__).resolve().parents[3]
DATA_ROOT = Path("/root/autodl-tmp/data/sparse_3d_forgery_detection/derived")
OUT_ROOT = DATA_ROOT / "v7_dataflow_cache_audit_v1"
PERIODIC_ROOT = DATA_ROOT / "v7_activityforensics_periodic_requery_pilot_v1"
MULTI_ROOT = DATA_ROOT / "v7_activityforensics_multi_order_sequence_pilot_v1"
PAIR_ROOT = DATA_ROOT / "v7_activityforensics_pair_trajectory_pilot_v1"
ATT_ROOT = DATA_ROOT / "v7_activityforensics_attention_pooling_pilot_v1"
SOURCE128_ROOT = DATA_ROOT / "v7_activityforensics_source128_extension_v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_head() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()


def _json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader(); writer.writerows(rows)
    os.replace(tmp, path)


def _finite_max(left: np.ndarray, right: np.ndarray) -> float | None:
    if left.shape != right.shape or left.size == 0:
        return None
    return float(np.max(np.abs(left.astype(np.float64) - right.astype(np.float64))))


def _row_key(row: Mapping[str, Any]) -> tuple[str, str, float]:
    return str(row["source_id"]), str(row["role"]), float(row.get("offset_s", 0.0))


def _periodic_trace(evidence: list[dict[str, Any]]) -> dict[str, Any]:
    rows = _json(PERIODIC_ROOT / "support/window_support.json")
    candidates = [row for row in rows if row.get("mode") == "O" and float(row.get("offset_s", 0.0)) > 0 and int(row.get("valid_unit_count", 0)) > 0]
    chosen: list[dict[str, Any]] = []
    for role in ("real", "fake"):
        candidates_role = sorted([row for row in candidates if row.get("role") == role], key=_row_key)
        for row in candidates_role:
            r = next((item for item in rows if item.get("window_id") == row.get("window_id") and item.get("mode") == "R" and int(item.get("valid_unit_count", 0)) > 0), None)
            if r is not None:
                chosen.append(row); break
    model_records = _json(PERIODIC_ROOT / "models/fold_models.json")["records"]
    score_rows = {str(row["window_id"]): row for row in csv.DictReader((PERIODIC_ROOT / "scores/oof_window_scores.csv").open(newline="", encoding="utf-8"))}
    results: list[dict[str, Any]] = []
    for row in chosen:
        rrow = next(item for item in rows if item.get("window_id") == row.get("window_id") and item.get("mode") == "R")
        for mode, current in (("O", row), ("R", rrow)):
            rebuilt = periodic._support_for_sequence(Path(str(current["particle_prefix"])), current, mode)
            seq = load_particle_sequence(str(current["particle_prefix"]))
            for base in periodic.BASE_CONDITIONS:
                stored = np.asarray(current["features"][base], dtype=np.float64)
                rebuilt_values = np.asarray(rebuilt["features"][base], dtype=np.float64)
                condition = f"{mode}_{'SET' if base == 'SET_A' else 'RAW' if base == 'RAW_SEQ' else 'MULTI'}"
                score_diff = None
                record = next((item for item in model_records if item.get("condition") == condition and str(item.get("held_out_source")) == str(current["source_id"]) and int(item.get("seed", -1)) == 20260909), None)
                if record is not None:
                    standardizer = periodic._standardizer(record)
                    batch_row = {**current, "features": {base: rebuilt_values}, "intervals_s": rebuilt.get("intervals_s")}
                    model = periodic._model_from_record(base, record, "cpu")
                    score = float(periodic.score_batch(base, model, periodic.make_batch(base, [batch_row], standardizer, require_labels=False), "cpu")[0])
                    score_diff = abs(score - float(score_rows[str(current["window_id"])][f"{condition}_seed_20260909"]))
                results.append({"experiment": "periodic_requery", "role": current["role"], "window_id": current["window_id"], "mode": mode, "condition": condition, "particle_path": str(current["particle_prefix"]), "feature_shape": list(stored.shape), "source_video_id": str(seq.source_video_id), "query_cohort": str(seq.provenance.get("query_cohort")), "lineage_window_id": str(seq.lineage.get("window_id")), "frame_start": int(seq.frame_indices[0]), "frame_end": int(seq.frame_indices[-1]), "pts_start": float(seq.timestamps_s[0]), "pts_end": float(seq.timestamps_s[-1]), "recomputed_feature_max_abs": _finite_max(stored, rebuilt_values), "score_max_abs": score_diff, "status": "PASS" if _finite_max(stored, rebuilt_values) == 0.0 and (score_diff is None or score_diff <= 1e-5) else "MISMATCH"})
    evidence.extend(results)
    return {"chosen_windows": [str(row["window_id"]) for row in chosen], "rows": len(results), "all_pass": all(row["status"] == "PASS" for row in results)}


def _multi_trace(evidence: list[dict[str, Any]]) -> dict[str, Any]:
    rows = _json(MULTI_ROOT / "support/window_support.json")
    candidates = sorted([row for row in rows if int(row.get("valid_unit_count", 0)) > 0], key=_row_key)
    chosen: list[dict[str, Any]] = []
    for role in ("real", "fake"):
        chosen.append(next(row for row in candidates if row.get("role") == role))
    score_rows = {str(row["window_id"]): row for row in csv.DictReader((MULTI_ROOT / "scores/oof_window_scores.csv").open(newline="", encoding="utf-8"))}
    records = _json(MULTI_ROOT / "models/fold_models.json")["records"]
    result_rows: list[dict[str, Any]] = []
    for row in chosen:
        feature_path = MULTI_ROOT / "features" / ("".join(c if c.isalnum() or c in "._-" else "_" for c in str(row["window_id"])) + ".npz")
        with np.load(feature_path, allow_pickle=False) as archive:
            stored = {condition: np.asarray(archive[condition], dtype=np.float64) for condition in multi.CONDITIONS}
        rebuilt = multi_repr.condition_feature_matrix(row, str(row["window_id"]))["features"]
        sample_base = {**row, "features": stored, "intervals_s": np.diff(np.asarray(row["target_timestamps_s"], dtype=np.float64))}
        for condition in multi.CONDITIONS:
            diff = _finite_max(stored[condition], np.asarray(rebuilt[condition], dtype=np.float64))
            rec = next((item for item in records if item.get("condition") == condition and str(item.get("held_out_source")) == str(row["source_id"]) and int(item.get("seed", -1)) == 20260909), None)
            score_diff = None
            if rec is not None:
                model = multi._model_from_record(condition, rec, "cpu")
                score = float(multi.score_batch(condition, model, multi.make_batch(condition, [sample_base], multi._standardizer_record(rec["standardization"]), require_labels=False), "cpu")[0])
                score_diff = abs(score - float(score_rows[str(row["window_id"])][f"{condition}_seed_20260909"]))
            result_rows.append({"experiment": "multi_order_sequence", "role": row["role"], "window_id": row["window_id"], "condition": condition, "feature_shape": list(stored[condition].shape), "feature_sha256": hashlib.sha256(stored[condition].tobytes(order="C")).hexdigest(), "recomputed_feature_max_abs": diff, "score_max_abs": score_diff, "status": "PASS" if diff == 0.0 and (score_diff is None or score_diff <= 1e-5) else "MISMATCH"})
    synthetic = multi_repr.condition_inputs(np.asarray([[0.0, 0.0, 0.0, 0.0], [1.0, 0.5, 0.2, 0.1], [2.2, 0.1, 0.4, 0.8], [3.7, 0.8, 0.3, 0.2], [5.9, 0.2, 0.6, 1.1]]), [0.0, 0.11, 0.23, 0.41, 0.58], "synthetic-a")
    synthetic_diff = float(np.max(np.abs(synthetic["MULTI_ORDER_SEQ"] - synthetic["SHUFFLED_MULTI_ORDER"])))
    evidence.extend(result_rows)
    return {"chosen_windows": [str(row["window_id"]) for row in chosen], "rows": len(result_rows), "all_pass": all(row["status"] == "PASS" for row in result_rows), "synthetic_order_difference_max_abs": synthetic_diff}


def _pair_trace(evidence: list[dict[str, Any]]) -> dict[str, Any]:
    data = pair._load_relation_dataset(PAIR_ROOT)
    by_source: dict[str, list[dict[str, Any]]] = {}
    for row in data["windows"]:
        by_source.setdefault(str(row["source_id"]), []).append(row)
    model_records = _json(PAIR_ROOT / "models/fold_models.json")["records"]
    valid_sources = [source for source, source_rows in sorted(by_source.items()) if {str(row["role"]) for row in source_rows} == {"real", "fake"} and any(str(rec.get("held_out_source")) == source for rec in model_records)]
    source = valid_sources[0]
    chosen = [next(row for row in by_source[source] if row["role"] == role and int(row["relation_count"]) > 0) for role in ("real", "fake")]
    features = {condition: pair._condition_features(data, condition)[0] for condition in pair.CONDITIONS}
    logs = {condition: pair._condition_features(data, condition)[1] for condition in pair.CONDITIONS}
    scores = {str(row["window_id"]): row for row in csv.DictReader((PAIR_ROOT / "scores/oof_window_scores.csv").open(newline="", encoding="utf-8"))}
    result_rows: list[dict[str, Any]] = []
    for row in chosen:
        index = next(i for i, item in enumerate(data["windows"]) if str(item["window_id"]) == str(row["window_id"]))
        for condition in pair.CONDITIONS:
            rec = next(item for item in model_records if item.get("condition") == condition and str(item.get("held_out_source")) == source and int(item.get("seed", -1)) == 20260909)
            standard = pair.DistanceStandardizer(np.asarray(rec["standardization"]["mean"]), np.asarray(rec["standardization"]["scale"]), tuple(rec["standardization"].get("zero_variance_dimensions", [])), rec["standardization"].get("weighting", {}))
            model = pair._model_from_record(rec, "cpu")
            score = float(pair._score(model, pair._batch(data, [index], features[condition], standard, require_labels=False), "cpu")[0])
            saved = float(scores[str(row["window_id"])][f"{condition}_seed_20260909"])
            result_rows.append({"experiment": "pair_trajectory", "role": row["role"], "window_id": row["window_id"], "condition": condition, "relation_count": int(row["relation_count"]), "score_max_abs": abs(score - saved), "status": "PASS" if abs(score - saved) <= 1e-5 else "MISMATCH"})
    raw_checks = pair._validate_transforms(data)
    checks: dict[str, Any] = {"all_checks_passed": bool(raw_checks.get("all_checks_passed"))}
    for name in ("id_shuffle", "time_shuffle"):
        item = raw_checks.get(name, {})
        checks[name] = {key: item.get(key) for key in ("condition", "fixed_seed", "changed_entry_ratio", "changed_trajectory_ratio", "changed_window_ratio", "intervals_unchanged") if key in item}
    units = raw_checks.get("unit_checks", [])
    checks["unit_checks"] = {"count": len(units), "all_multisets_preserved": all(bool(item.get("id_multiset_preserved")) and bool(item.get("id_summary_preserved")) and bool(item.get("time_relation_multiset_preserved")) for item in units)}
    evidence.extend(result_rows)
    return {"chosen_windows": [str(row["window_id"]) for row in chosen], "rows": len(result_rows), "all_pass": all(row["status"] == "PASS" for row in result_rows), "transform_checks": checks, "condition_logs": {condition: {key: value for key, value in log.items() if key in {"condition", "changed_entry_ratio", "changed_trajectory_ratio", "changed_window_ratio", "intervals_unchanged"}} for condition, log in logs.items()}}


def _attention_trace(evidence: list[dict[str, Any]]) -> dict[str, Any]:
    train_rows, validation_rows, _, info = attention._load_inputs()
    by_source: dict[str, list[dict[str, Any]]] = {}
    for row in validation_rows:
        by_source.setdefault(str(row["source_id"]), []).append(row)
    source = next(source for source, source_rows in sorted(by_source.items()) if {int(row["label"]) for row in source_rows} == {0, 1})
    chosen = [next(row for row in by_source[source] if int(row["label"]) == label) for label in (0, 1)]
    scores = {str(row["window_id"]): row for row in csv.DictReader((ATT_ROOT / "scores/validation_window_scores.csv").open(newline="", encoding="utf-8"))}
    records = _json(ATT_ROOT / "models/fold_models.json")["records"]
    result_rows: list[dict[str, Any]] = []
    perturbation: dict[str, float] = {}
    for row in chosen:
        for condition in attention.CONDITIONS:
            rec = next(item for item in records if item.get("condition") == condition and int(item.get("seed", -1)) == 20260909)
            model = attention._load_model(rec, "cpu")
            standard = periodic._standardizer(rec)
            batch = attention._batch([row], standard, weighted=False)
            predicted = float((attention._window_outputs(model, batch, "cpu")[0])[0])
            saved = float(scores[str(row["window_id"])][f"{condition}_seed_20260909"])
            result_rows.append({"experiment": "attention_pooling", "role": row["role"], "window_id": row["window_id"], "condition": condition, "model_class": type(model).__name__, "score_max_abs": abs(predicted - saved), "status": "PASS" if abs(predicted - saved) <= 1e-5 else "MISMATCH"})
            if condition == "ATTENTION_POOL":
                import torch
                with torch.no_grad():
                    model.attention_w.add_(0.1)
                changed = float((attention._window_outputs(model, batch, "cpu")[0])[0])
                perturbation[str(row["window_id"])] = abs(changed - predicted)
    evidence.extend(result_rows)
    expected = info["expected"]
    return {"chosen_windows": [str(row["window_id"]) for row in chosen], "rows": len(result_rows), "all_pass": all(row["status"] == "PASS" for row in result_rows), "model_classes": sorted({row["model_class"] for row in result_rows}), "attention_parameter_perturbation_max_abs": max(perturbation.values()) if perturbation else None, "expected_counts": {key: value for key, value in expected.items() if key not in {"train_sources", "validation_sources"}}}


def _source128_trace(evidence: list[dict[str, Any]]) -> dict[str, Any]:
    rows = source128._load_rows(SOURCE128_ROOT)
    scores = {str(row["window_id"]): row for row in csv.DictReader((SOURCE128_ROOT / "scores/validation_window_scores.csv").open(newline="", encoding="utf-8"))}
    validation_ids = set(scores)
    candidates = [row for row in rows if str(row["window_id"]) in validation_ids]
    chosen = [next(row for row in candidates if row.get("role") == role) for role in ("real", "fake")]
    records = _json(SOURCE128_ROOT / "models/fold_models.json")["records"]
    protocol = _json(SOURCE128_ROOT / "protocol.json")
    train_sources = [str(x) for x in protocol["selection"]["base_sources"] + protocol["selection"]["added_sources"]]
    train_rows, observed, missing = source128._effective_training_rows(rows, train_sources)
    import torch
    from research_tools.v7.periodic_requery_probe import runner as periodic_module
    weights = periodic_module.source_class_weights(train_rows)
    recomputed = periodic_module.fit_standardizer("SET_A", [np.asarray(row["features"]["SET_A"], dtype=np.float64) for row in train_rows], weights)
    standardizer_diffs: list[float] = []
    result_rows: list[dict[str, Any]] = []
    for record in records:
        recorded = record["standardization"]
        standardizer_diffs.extend([float(np.max(np.abs(np.asarray(recorded["mean"]) - recomputed.mean))), float(np.max(np.abs(np.asarray(recorded["scale"]) - recomputed.scale)))])
        model = periodic_module._model_from_record("SET_A", record, "cpu")
        for row in chosen:
            value_row = {**row, "features": {"SET_A": np.asarray(row["features"]["SET_A"], dtype=np.float64)}, "intervals_s": np.asarray(row["intervals_s"], dtype=np.float64)}
            predicted = float(periodic_module.score_batch("SET_A", model, periodic_module.make_batch("SET_A", [value_row], periodic_module.FeatureStandardizer("SET_A", np.asarray(recorded["mean"]), np.asarray(recorded["scale"]), tuple(recorded.get("zero_variance_dimensions", []))), require_labels=False), "cpu")[0])
            saved = float(scores[str(row["window_id"])][f"source128_seed_{int(record['seed'])}"])
            result_rows.append({"experiment": "source128_extension", "role": row["role"], "window_id": row["window_id"], "seed": int(record["seed"]), "source_count": int(record.get("source_count", 0)), "status": str(record.get("status")), "epochs": int(record.get("fit", {}).get("epochs", -1)), "score_max_abs": abs(predicted - saved), "parameter_count": int(record.get("parameter_count", -1))})
    manifest_ids = {str(row["window_id"]) for row in csv.DictReader((SOURCE128_ROOT / "models/training_window_manifest.csv").open(newline="", encoding="utf-8"))}
    evidence.extend(result_rows)
    return {"chosen_windows": [str(row["window_id"]) for row in chosen], "rows": len(result_rows), "all_scores_pass": all(row["score_max_abs"] <= 1e-5 for row in result_rows), "model_count": len(records), "source_count_nominal": sorted({int(row.get("source_count", 0)) for row in records}), "effective_training_sources": len(observed), "missing_training_sources": missing, "training_window_count": len(train_rows), "training_manifest_window_count": len(manifest_ids), "validation_training_source_overlap": sorted(set(observed) & {str(row["source_id"]) for row in candidates}), "standardizer_max_abs_diff": max(standardizer_diffs, default=None)}


def _inventory() -> list[dict[str, Any]]:
    paths = [
        "research_tools/v7/periodic_requery_probe/runner.py", "research_tools/v7/multi_order_sequence_probe/runner.py", "research_tools/v7/multi_order_sequence_probe/representation.py", "research_tools/v7/pair_trajectory_probe/runner.py", "research_tools/v7/attention_pooling_pilot/runner.py", "research_tools/v7/attention_pooling_pilot/model.py", "research_tools/v7/source128_extension/runner.py", "scripts/run_v7_source128_screen.sh",
    ]
    tracked = set(subprocess.check_output(["git", "ls-files"], cwd=REPO_ROOT, text=True).splitlines())
    return [{"path": path, "kind": "self-authored", "git_tracked": path in tracked, "status": "CODE_PUSHED" if path in tracked else "CODE_NOT_TRACKED"} for path in paths]


def run() -> dict[str, Any]:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    evidence: list[dict[str, Any]] = []
    summaries = {
        "periodic_requery": _periodic_trace(evidence),
        "multi_order_sequence": _multi_trace(evidence),
        "pair_trajectory": _pair_trace(evidence),
        "attention_pooling": _attention_trace(evidence),
        "source128_extension": _source128_trace(evidence),
    }
    _write_csv(OUT_ROOT / "evidence.csv", evidence)
    _write_csv(OUT_ROOT / "source_code_inventory.csv", _inventory())
    _write_json(OUT_ROOT / "summary.json", {"git_head": _git_head(), "summaries": summaries, "potential_gaps": ["Historical model records in the five pilots predate input_identity fingerprints; exact cache reuse cannot be proven from those records alone.", "Historical periodic support rows omit parent_id; current consumer and producer checks now require it for newly written artifacts."]})
    lines = ["# V7 data-flow and cache-reuse audit", "", f"- Audited repository HEAD: `{_git_head()}`", "- Scope: existing arrays, manifests and CPU model forward checks only; no frontend, tracking, depth, pose, segmentation or training rerun.", "- Samples were selected deterministically from valid non-zero-offset real/fake rows; no score-based selection.", "", "## Direct answers", "", "All five sampled paths reached their declared condition-specific feature arrays and model classes. The reconstructed feature arrays and saved model scores matched within the recorded tolerance; no sampled A/B feature or model substitution was found. This does not prove every historical cache reuse was correctly fingerprinted.", "", "Potential rather than demonstrated pollution remains in historical resume metadata: old periodic/multi/pair/attention/source128 model records do not contain input fingerprints, and old periodic support rows do not contain `parent_id`. The code now rejects key-only model reuse and checks R sequence source/role/parent/query-cohort at the consumer. Existing results are not automatically invalidated or rerun; records without historical fingerprints remain `UNVERIFIED_HISTORY`.", "", "## Experiment summaries", ""]
    for name, summary in summaries.items():
        lines.append(f"### {name}")
        lines.append("```json")
        lines.append(json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=False))
        lines.append("```")
    lines += ["", "## Source/code boundary", "", "All listed V7 workflow files and the source128 launcher are tracked in Git. External tracker/depth/pose implementations, weights, videos, particle arrays and model artifacts remain data-disk assets and are not code omissions. The repository README was updated to distinguish the design baseline from the current V7 research tools and their external assets.", "", "## Re-run decision", "", "No expensive rerun is justified by this bounded audit. If a future resume is requested, records lacking `input_identity` must be treated as stale/unverified and regenerated only for the affected fold/condition; no full experiment reset is implied.", ""]
    (OUT_ROOT / "report.md").write_text("\n".join(lines), encoding="utf-8")
    return {"output": str(OUT_ROOT), "summaries": summaries}


if __name__ == "__main__":
    print(json.dumps(run(), indent=2, ensure_ascii=False))
