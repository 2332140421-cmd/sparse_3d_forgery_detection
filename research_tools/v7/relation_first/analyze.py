"""Compare relation-first representations with frozen pool-first baselines."""

from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from research_tools.v7.paired_signal.analyze import (
    BOOTSTRAP_REPLICATES,
    BOOTSTRAP_SEED,
    _auroc,
    _bootstrap_median,
    _rankdata,
    _sign_test,
    _spearman,
    finite_summary,
)
from research_tools.v7.paired_signal.protocol import window_observation_median

from .representation import robust_scale


PREVIOUS_ROOT = Path("/root/autodl-tmp/data/sparse_3d_forgery_detection/derived/v7_activityforensics_paired_second_order_pilot_v1")
OUTPUT_ROOT = Path("/root/autodl-tmp/data/sparse_3d_forgery_detection/derived/v7_activityforensics_relation_first_pilot_v1")
RELATION_NAMES = ("R1", "R2")
BASELINE_NAMES = ("B0", "P1", "P2")
ALL_NAMES = BASELINE_NAMES + RELATION_NAMES


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False, ensure_ascii=False) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _json_vector(value: Any) -> str | None:
    return json.dumps(value, separators=(",", ":"), allow_nan=False) if value is not None else None


def _vector(value: Any, dimension: int) -> np.ndarray | None:
    if value is None:
        return None
    array = np.asarray(value, dtype=np.float64)
    return array if array.shape == (dimension,) and np.all(np.isfinite(array)) else None


def _summary(rows: list[dict[str, Any]], name: str) -> dict[str, Any]:
    manip = [row.get(f"Dmanip_{name}") for row in rows]
    control = [row.get(f"Dctrl_{name}") for row in rows]
    gaps = [row.get(f"G_{name}") for row in rows]
    sign = _sign_test(gaps)
    nonzero = sign["N_nonzero"]
    return {
        "D_manip": finite_summary(manip),
        "D_ctrl": finite_summary(control),
        "G": {
            **finite_summary(gaps),
            "positive_fraction": float(np.mean(np.asarray([x for x in gaps if x is not None], dtype=np.float64) > 0)) if any(x is not None for x in gaps) else None,
            "sign_test": sign,
            "paired_rank_biserial": (sign["positive"] - sign["negative"]) / nonzero if nonzero else None,
            "bootstrap_median": _bootstrap_median(gaps),
        },
        "exploratory_auroc_Dmanip_vs_Dctrl": _auroc(manip, control),
    }


def _contrast(rows: list[dict[str, Any]], left: str, right: str) -> dict[str, Any]:
    values = [row.get(f"G_{left}") - row.get(f"G_{right}") for row in rows if row.get(f"G_{left}") is not None and row.get(f"G_{right}") is not None]
    sign = _sign_test(values)
    return {**finite_summary(values), "positive_fraction": float(np.mean(np.asarray(values) > 0)) if values else None, "sign_test": sign, "paired_rank_biserial": (sign["positive"] - sign["negative"]) / sign["N_nonzero"] if sign["N_nonzero"] else None, "bootstrap_median": _bootstrap_median(values)}


def _relation_window_rows(previous_results: list[dict[str, Any]], relation_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    previous_by_id = {row["window"]["window_id"]: row for row in previous_results}
    rows: list[dict[str, Any]] = []
    for relation_row in relation_rows:
        window = relation_row["window"]
        previous = previous_by_id[window["window_id"]]
        features = previous["features"]
        vectors: dict[str, list[float] | None] = {}
        dimensions = {"B0": 4, "P1": 4, "P2": 4, "R1": 8, "R2": 8}
        for name, key in (("B0", "K0_S"), ("P1", "K1_delta_s"), ("P2", "K2_delta2_s")):
            vector, _ = window_observation_median(features, key)
            vectors[name] = vector.tolist() if vector is not None else None
        vectors["R1"] = relation_row["relation"]["r1_window"]
        vectors["R2"] = relation_row["relation"]["r2_window"]
        row = {"window_id": window["window_id"], "pair_id": window["pair_id"], "source_id": window["source_id"], "role": window["role"], "kind": window["kind"], "label": window["label"], "anchor_fraction": window["anchor_fraction"], "geometry_coverage": previous["quality"]["geometry_coverage"], "tracking_persistence": previous["quality"]["tracking_persistence"], "component_count": relation_row["relation"]["component_count"], "component_with_valid_r1": relation_row["relation"]["component_with_valid_r1"], "component_with_valid_r2": relation_row["relation"]["component_with_valid_r2"], "persistent_pair_count": relation_row["relation"]["persistent_pair_count"], "r1_eligible_pair_count": relation_row["relation"]["r1_eligible_pair_count"], "r2_eligible_pair_count": relation_row["relation"]["r2_eligible_pair_count"], "r1_eligible_fraction": relation_row["relation"]["r1_eligible_fraction"], "r2_eligible_fraction": relation_row["relation"]["r2_eligible_fraction"]}
        for name, dimension in dimensions.items():
            row[f"{name}_vector"] = _json_vector(vectors[name])
            row[f"{name}_valid"] = _vector(vectors[name], dimension) is not None
        rows.append(row)
    return rows


def _scaled_relation_gains(window_rows: list[dict[str, Any]], selected: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    scales: dict[str, Any] = {}
    scale_arrays: dict[str, tuple[np.ndarray, dict[str, Any]]] = {}
    for name in RELATION_NAMES:
        vectors = [_vector(json.loads(row[f"{name}_vector"]), 8) for row in window_rows if row["role"] == "real" and row["kind"] == "CTRL" and row[f"{name}_valid"]]
        if vectors:
            scale_arrays[name] = robust_scale([item.tolist() for item in vectors])
            scales[name] = {"N_real_control_windows": len(vectors), "details": scale_arrays[name][1]}
        else:
            scales[name] = {"N_real_control_windows": 0, "details": None}
    sides: dict[tuple[str, str, str, float], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in window_rows:
        sides[(row["pair_id"], row["kind"], row["label"], float(row["anchor_fraction"]))][row["role"]] = row
    values = {pair["pair_id"]: {"pair_id": pair["pair_id"], "source_id": pair["source_id"], "generator": pair["generator"], "manipulation_operation": pair["manipulation_operation"], "D": {name: {"MANIP": [], "CTRL": []} for name in RELATION_NAMES}, "coverage": {kind: {role: [] for role in ("real", "fake")} for kind in ("MANIP", "CTRL")}} for pair in selected}
    for (pair_id, kind, label, anchor), role_rows in sides.items():
        if pair_id not in values or "real" not in role_rows or "fake" not in role_rows:
            continue
        target = values[pair_id]
        for role in ("real", "fake"):
            target["coverage"][kind][role].append(role_rows[role]["r2_eligible_pair_count"])
        for name in RELATION_NAMES:
            if name not in scale_arrays:
                continue
            real = _vector(json.loads(role_rows["real"][f"{name}_vector"]), 8) if role_rows["real"][f"{name}_valid"] else None
            fake = _vector(json.loads(role_rows["fake"][f"{name}_vector"]), 8) if role_rows["fake"][f"{name}_valid"] else None
            if real is None or fake is None:
                continue
            scale = scale_arrays[name][0]
            target["D"][name][kind].append(float(np.linalg.norm((fake - real) / (scale + 1e-12))))
    pair_rows: list[dict[str, Any]] = []
    for value in values.values():
        row = {key: item for key, item in value.items() if key not in ("D", "coverage")}
        for name in RELATION_NAMES:
            manip = np.asarray(value["D"][name]["MANIP"], dtype=np.float64)
            ctrl = np.asarray(value["D"][name]["CTRL"], dtype=np.float64)
            row[f"Dmanip_{name}"] = float(np.median(manip)) if manip.size else None
            row[f"Dctrl_{name}"] = float(np.median(ctrl)) if ctrl.size else None
            row[f"G_{name}"] = float(np.median(manip) - np.median(ctrl)) if manip.size and ctrl.size else None
        for kind in ("MANIP", "CTRL"):
            real = np.asarray(value["coverage"][kind]["real"], dtype=np.float64)
            fake = np.asarray(value["coverage"][kind]["fake"], dtype=np.float64)
            row[f"r2_pair_count_gap_{kind}"] = float(np.median(fake) - np.median(real)) if real.size and fake.size else None
        pair_rows.append(row)
    return pair_rows, scales


def _baseline_pair_rows(selected: list[dict[str, Any]], previous_results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    from research_tools.v7.paired_signal.analyze import _pair_rows

    rows, _ = _pair_rows(selected, previous_results)
    output: list[dict[str, Any]] = []
    for row in rows:
        item = {key: row.get(key) for key in ("pair_id", "source_id", "generator", "manipulation_operation")}
        for old, new in (("K0_S", "B0"), ("K1_delta_s", "P1"), ("K2_delta2_s", "P2")):
            item[f"Dmanip_{new}"] = row.get(f"Dmanip_{old}")
            item[f"Dctrl_{new}"] = row.get(f"Dctrl_{old}")
            item[f"G_{new}"] = row.get(f"G_{old}")
        item.update({key: row[key] for key in row if key.startswith("quality_gap_")})
        output.append(item)
    return output


def _coverage_summary(window_rows: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in window_rows:
        grouped[f"{row['role']}::{row['kind']}"].append(row)
    output: dict[str, Any] = {}
    for key, rows in sorted(grouped.items()):
        output[key] = {name: finite_summary(row[name] for row in rows) for name in ("persistent_pair_count", "r1_eligible_pair_count", "r2_eligible_pair_count", "r1_eligible_fraction", "r2_eligible_fraction", "component_count", "component_with_valid_r1", "component_with_valid_r2")}
    return output


def analyze(output: Path = OUTPUT_ROOT) -> dict[str, Any]:
    selected = json.loads((PREVIOUS_ROOT / "manifests/selected_pairs.json").read_text(encoding="utf-8"))
    relation_rows = json.loads((output / "relations/per_window_relation_summary.json").read_text(encoding="utf-8"))
    previous_results = json.loads((PREVIOUS_ROOT / "frontend/window_results.json").read_text(encoding="utf-8"))
    window_rows = _relation_window_rows(previous_results, relation_rows)
    _write_csv(output / "metrics/per_window_representation.csv", window_rows)
    relation_pairs, scales = _scaled_relation_gains(window_rows, selected)
    baseline_pairs = _baseline_pair_rows(selected, previous_results)
    baseline_by_id = {row["pair_id"]: row for row in baseline_pairs}
    relation_by_id = {row["pair_id"]: row for row in relation_pairs}
    pair_rows: list[dict[str, Any]] = []
    for pair in selected:
        pair_id = pair["pair_id"]
        row = {"pair_id": pair_id, "source_id": pair["source_id"], "generator": pair["generator"], "manipulation_operation": pair["manipulation_operation"]}
        row.update({key: baseline_by_id[pair_id].get(key) for key in baseline_by_id[pair_id] if key.startswith("D") or key.startswith("G") or key.startswith("quality_gap")})
        row.update({key: relation_by_id[pair_id].get(key) for key in relation_by_id[pair_id] if key.startswith("D") or key.startswith("G") or key.startswith("r2_pair_count_gap")})
        pair_rows.append(row)
    _write_csv(output / "metrics/per_pair_gain.csv", pair_rows)
    summaries = {name: _summary(pair_rows, name) for name in ALL_NAMES}
    contrasts = {"A_R2_P2": _contrast(pair_rows, "R2", "P2"), "A_R2_R1": _contrast(pair_rows, "R2", "R1"), "A_R2_B0": _contrast(pair_rows, "R2", "B0")}
    previous_summary = json.loads((PREVIOUS_ROOT / "metrics/structural_order_summary.json").read_text(encoding="utf-8"))
    b0_current = summaries["B0"]
    b0_previous = previous_summary["orders"]["K0_S"]
    b0_diff = {"G_median": (b0_current["G"]["median"] or 0) - (b0_previous["G"]["median"] or 0), "positive_fraction": (b0_current["G"]["positive_fraction"] or 0) - (b0_previous["G"]["positive_fraction"] or 0), "AUROC": (b0_current["exploratory_auroc_Dmanip_vs_Dctrl"] or 0) - (b0_previous["exploratory_auroc_Dmanip_vs_Dctrl"] or 0)}
    b0_reproduced = all(abs(value) <= 1e-12 for value in b0_diff.values())
    coverage = _coverage_summary(window_rows)
    coverage_by_pair = {pair["pair_id"]: {kind: {role: [] for role in ("real", "fake")} for kind in ("MANIP", "CTRL")} for pair in selected}
    for row in window_rows:
        coverage_by_pair[row["pair_id"]][row["kind"]][row["role"]].append(row["r2_eligible_pair_count"])
    pair_count_gap = {pair_id: float(np.median(value["MANIP"]["fake"]) - np.median(value["MANIP"]["real"])) if value["MANIP"]["fake"] and value["MANIP"]["real"] else None for pair_id, value in coverage_by_pair.items()}
    r2_g = [row.get("G_R2") for row in pair_rows]
    quality_confounds = {"r2_vs_manipulation_r2_pair_count_gap": _spearman(r2_g, [pair_count_gap[row["pair_id"]] for row in pair_rows]), "r2_vs_geometry_gap": _spearman(r2_g, [row.get("quality_gap_MANIP_geometry_coverage") for row in pair_rows]), "r2_vs_tracking_gap": _spearman(r2_g, [row.get("quality_gap_MANIP_tracking_persistence") for row in pair_rows]), "strong_threshold": 0.7}
    previous_pairwise = json.loads((PREVIOUS_ROOT / "metrics/pairwise_diagnostic.json").read_text(encoding="utf-8"))["per_pair"]
    raw_by_id = {row["pair_id"]: row.get("G_second_difference_magnitude") for row in previous_pairwise}
    aligned = [(row.get("G_R2"), raw_by_id.get(row["pair_id"])) for row in pair_rows if row.get("G_R2") is not None and raw_by_id.get(row["pair_id"]) is not None]
    raw_alignment = {"N": len(aligned), "spearman": _spearman([x[0] for x in aligned], [x[1] for x in aligned]), "direction_agreement": float(np.mean([np.sign(x[0]) == np.sign(x[1]) for x in aligned])) if aligned else None, "previous_raw_metric": "G_second_difference_magnitude"}
    _write_json(output / "metrics/quality_confound.json", {"window_summary": coverage, "pair_count_gap": pair_count_gap, "spearman": quality_confounds})
    _write_json(output / "metrics/raw_alignment.json", raw_alignment)
    comparison = {"representations": summaries, "contrasts": contrasts, "B0_reproduction": {"reproduced": b0_reproduced, "differences": b0_diff, "previous": b0_previous}, "scales": scales, "bootstrap_seed": BOOTSTRAP_SEED, "bootstrap_replicates": BOOTSTRAP_REPLICATES, "fake_count_in_fitting": 0, "normality_model": "NONE"}
    _write_json(output / "metrics/representation_comparison.json", comparison)
    r2 = summaries["R2"]["G"]
    raw_previous = json.loads((PREVIOUS_ROOT / "metrics/pairwise_diagnostic.json").read_text(encoding="utf-8"))["paired"]["second_difference_magnitude"]["G"]
    raw_stable = raw_previous["N"] >= 12 and raw_previous["median"] > 0 and raw_previous["positive_fraction"] >= 0.70 and raw_previous["bootstrap_median"]["ci95"][0] > 0
    status = "PILOT_INCONCLUSIVE"
    if not b0_reproduced:
        status = "B0_REGRESSION_MISMATCH"
    elif r2["N"] < 12:
        status = "PILOT_INCONCLUSIVE"
    elif r2["median"] is not None and r2["median"] > 0 and r2["positive_fraction"] >= 0.70 and r2["bootstrap_median"]["ci95"][0] >= 0:
        if contrasts["A_R2_P2"]["median"] is not None and contrasts["A_R2_P2"]["median"] > 0 and contrasts["A_R2_P2"]["positive_fraction"] >= 0.70:
            status = "DERIVATIVE_BEFORE_AGGREGATION_SUPPORTED"
            if contrasts["A_R2_R1"]["median"] is not None and contrasts["A_R2_R1"]["median"] > 0 and contrasts["A_R2_R1"]["positive_fraction"] >= 0.60:
                status = "RELATION_FIRST_SECOND_ORDER_ADVANTAGE_PRESENT"
        else:
            status = "RELATION_FIRST_SECOND_ORDER_SIGNAL_PRESENT"
    elif raw_stable:
        status = "FIXED_DIMENSION_RELATION_AGGREGATION_STILL_LOSSY"
    else:
        status = "RAW_SIGNAL_REPRODUCTION_FAILURE"
    tags = []
    if any(abs(quality_confounds[name]) >= 0.7 for name in ("r2_vs_manipulation_r2_pair_count_gap", "r2_vs_geometry_gap", "r2_vs_tracking_gap") if quality_confounds[name] is not None):
        tags.append("STRONG_REPRESENTATION_QUALITY_CONFOUND")
    final = {"status": status, "secondary_tags": tags, "selected_pairs": 16, "windows": 192, "frontend_rerun": False, "formal_src_modified": False, "b0_reproduced": b0_reproduced, "fake_count_in_fitting": 0, "normality_model": "NONE", "raw_previous_stable": raw_stable, "metrics": {"per_window": "metrics/per_window_representation.csv", "per_pair": "metrics/per_pair_gain.csv", "comparison": "metrics/representation_comparison.json", "quality": "metrics/quality_confound.json", "raw_alignment": "metrics/raw_alignment.json"}}
    _write_json(output / "run_summary.json", final)
    return final


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUTPUT_ROOT)
    args = parser.parse_args()
    print(json.dumps(analyze(args.output), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
