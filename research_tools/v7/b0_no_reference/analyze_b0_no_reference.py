"""Analysis helpers for the frozen B0 no-reference development pilot."""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Iterable, Sequence

import numpy as np


BOOTSTRAP_SEED = 20260909
BOOTSTRAP_REPLICATES = 10_000


def finite_summary(values: Iterable[float]) -> dict[str, Any]:
    array = np.asarray(list(values), dtype=np.float64)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return {"N": 0, "median": None, "iqr": None, "p10": None, "p90": None}
    return {
        "N": int(array.size),
        "median": float(np.median(array)),
        "iqr": float(np.percentile(array, 75) - np.percentile(array, 25)),
        "p10": float(np.percentile(array, 10)),
        "p90": float(np.percentile(array, 90)),
    }


def positive_fraction(values: Iterable[float]) -> float | None:
    array = np.asarray(list(values), dtype=np.float64)
    array = array[np.isfinite(array)]
    return float(np.mean(array > 0)) if array.size else None


def bootstrap_median(values: Iterable[float]) -> dict[str, Any]:
    array = np.asarray(list(values), dtype=np.float64)
    array = array[np.isfinite(array)]
    result: dict[str, Any] = {
        "N": int(array.size),
        "seed": BOOTSTRAP_SEED,
        "replicates": BOOTSTRAP_REPLICATES,
        "median": float(np.median(array)) if array.size else None,
        "ci95": None,
    }
    if array.size:
        rng = np.random.default_rng(BOOTSTRAP_SEED)
        samples = rng.integers(0, array.size, size=(BOOTSTRAP_REPLICATES, array.size))
        medians = np.median(array[samples], axis=1)
        result["ci95"] = [float(np.percentile(medians, 2.5)), float(np.percentile(medians, 97.5))]
    return result


def auroc(positive: Iterable[float], negative: Iterable[float]) -> float | None:
    pos = np.asarray(list(positive), dtype=np.float64)
    neg = np.asarray(list(negative), dtype=np.float64)
    pos, neg = pos[np.isfinite(pos)], neg[np.isfinite(neg)]
    if pos.size == 0 or neg.size == 0:
        return None
    comparisons = (pos[:, None] > neg[None, :]).astype(np.float64)
    comparisons += 0.5 * (pos[:, None] == neg[None, :])
    return float(np.mean(comparisons))


def window_score_rows(observation_rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Aggregate scored B0 observations by the frozen window median rule."""

    grouped: dict[str, list[float]] = defaultdict(list)
    metadata: dict[str, dict[str, Any]] = {}
    for row in observation_rows:
        score = row.get("score")
        if score is None or not np.isfinite(float(score)):
            continue
        window_id = str(row["window_id"])
        grouped[window_id].append(float(score))
        metadata[window_id] = {
            "window_id": window_id,
            "pair_id": str(row["pair_id"]),
            "source_id": str(row["source_id"]),
            "role": str(row["role"]),
            "kind": str(row["kind"]),
            "anchor_fraction": float(row["anchor_fraction"]),
        }
    output: list[dict[str, Any]] = []
    for window_id in sorted(metadata):
        values = np.asarray(grouped[window_id], dtype=np.float64)
        output.append({
            **metadata[window_id],
            "score": float(np.median(values)),
            "score_iqr": float(np.percentile(values, 75) - np.percentile(values, 25)),
            "valid_observation_count": int(values.size),
        })
    return output


def source_score_rows(window_rows: Sequence[dict[str, Any]], all_sources: Sequence[str]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for row in window_rows:
        score = row.get("score")
        if score is not None and np.isfinite(float(score)):
            grouped[(str(row["source_id"]), str(row["role"]), str(row["kind"]))].append(float(score))
    output: list[dict[str, Any]] = []
    for source in sorted({str(item) for item in all_sources}):
        for role in ("real", "fake"):
            for kind in ("MANIP", "CTRL"):
                values = np.asarray(grouped.get((source, role, kind), []), dtype=np.float64)
                values = values[np.isfinite(values)]
                output.append({
                    "source_id": source,
                    "role": role,
                    "kind": kind,
                    "window_count": int(values.size),
                    "score_median": float(np.median(values)) if values.size else None,
                    "score_iqr": float(np.percentile(values, 75) - np.percentile(values, 25)) if values.size else None,
                    "score_p90": float(np.percentile(values, 90)) if values.size else None,
                    "score_p10": float(np.percentile(values, 10)) if values.size else None,
                })
    return output


def real_behavior(window_rows: Sequence[dict[str, Any]], all_sources: Sequence[str]) -> dict[str, Any]:
    """Summarize held-out real scores with one equal-weight source unit."""

    per_source: list[dict[str, Any]] = []
    for source in sorted({str(item) for item in all_sources}):
        values = np.asarray(
            [
                row["score"]
                for row in window_rows
                if str(row["source_id"]) == source and row["role"] == "real" and row.get("score") is not None
            ],
            dtype=np.float64,
        )
        values = values[np.isfinite(values)]
        per_source.append(
            {
                "source_id": source,
                "window_count": int(values.size),
                "score_median": float(np.median(values)) if values.size else None,
                "score_iqr": float(np.percentile(values, 75) - np.percentile(values, 25)) if values.size else None,
                "score_p90": float(np.percentile(values, 90)) if values.size else None,
                "score_p10": float(np.percentile(values, 10)) if values.size else None,
            }
        )
    medians = [row["score_median"] for row in per_source if row["score_median"] is not None]
    values = np.asarray(medians, dtype=np.float64)
    return {
        "per_source": per_source,
        "cross_source_median_of_source_medians": float(np.median(values)) if values.size else None,
        "cross_source_mad_of_source_medians": float(np.median(np.abs(values - np.median(values)))) if values.size else None,
        "cross_source_p10": float(np.percentile(values, 10)) if values.size else None,
        "cross_source_p90": float(np.percentile(values, 90)) if values.size else None,
        "valid_source_count": int(values.size),
        "all_sources": list(sorted({str(item) for item in all_sources})),
    }


def evaluation(source_rows: Sequence[dict[str, Any]], observation_rows: Sequence[dict[str, Any]], window_rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    by_source: dict[str, dict[tuple[str, str], float]] = defaultdict(dict)
    for row in source_rows:
        if row["score_median"] is not None:
            by_source[str(row["source_id"])][(str(row["role"]), str(row["kind"]))] = float(row["score_median"])
    deltas: list[dict[str, Any]] = []
    for source in sorted(by_source):
        values = by_source[source]
        required = [("real", "MANIP"), ("fake", "MANIP"), ("real", "CTRL"), ("fake", "CTRL")]
        if not all(key in values for key in required):
            continue
        delta_manip = values[("fake", "MANIP")] - values[("real", "MANIP")]
        delta_control = values[("fake", "CTRL")] - values[("real", "CTRL")]
        deltas.append({
            "source_id": source,
            "A_RM": values[("real", "MANIP")],
            "A_FM": values[("fake", "MANIP")],
            "A_RC": values[("real", "CTRL")],
            "A_FC": values[("fake", "CTRL")],
            "Delta_manip": delta_manip,
            "Delta_control": delta_control,
            "J_B0": delta_manip - delta_control,
        })
    def stat(name: str) -> dict[str, Any]:
        values = [row[name] for row in deltas]
        return {**finite_summary(values), "positive_fraction": positive_fraction(values), "bootstrap": bootstrap_median(values)}

    by_group = lambda role, kind: [row["score"] for row in window_rows if row["role"] == role and row["kind"] == kind and row.get("score") is not None]
    real_all = [row["score"] for row in window_rows if row["role"] == "real" and row.get("score") is not None]
    fake_all = [row["score"] for row in window_rows if row["role"] == "fake" and row.get("score") is not None]
    return {
        "per_source_deltas": deltas,
        "Delta_manip": stat("Delta_manip"),
        "Delta_control": stat("Delta_control"),
        "J_B0": stat("J_B0"),
        "group_source_medians": {
            "A_RM": {row["source_id"]: row["A_RM"] for row in deltas},
            "A_FM": {row["source_id"]: row["A_FM"] for row in deltas},
            "A_RC": {row["source_id"]: row["A_RC"] for row in deltas},
            "A_FC": {row["source_id"]: row["A_FC"] for row in deltas},
        },
        "exploratory_auroc": {
            "fake_manip_vs_real_manip": auroc(by_group("fake", "MANIP"), by_group("real", "MANIP")),
            "fake_manip_vs_all_heldout_real": auroc(by_group("fake", "MANIP"), real_all),
            "all_fake_vs_all_real": auroc(fake_all, real_all),
        },
        "source_statistical_unit": True,
        "window_level_auroc_is_exploratory": True,
        "observation_score_count": len([row for row in observation_rows if row.get("score") is not None]),
    }


def choose_status(
    delta: dict[str, Any],
    contrast: dict[str, Any],
    valid_source_count: int,
    historical_paired_positive: bool,
    pooled_auroc: float | None,
) -> str:
    if valid_source_count < 8:
        return "B0_SUPPORT_INSUFFICIENT"
    delta_ci = delta.get("bootstrap", {}).get("ci95")
    contrast_ci = contrast.get("bootstrap", {}).get("ci95")
    delta_gate = bool(delta.get("median") is not None and delta["median"] > 0 and (delta.get("positive_fraction") or 0) >= 0.70 and delta_ci and delta_ci[0] >= 0)
    contrast_gate = bool(contrast.get("median") is not None and contrast["median"] > 0 and (contrast.get("positive_fraction") or 0) >= 0.70 and contrast_ci and contrast_ci[0] >= 0)
    if delta_gate and contrast_gate:
        return "B0_NO_REFERENCE_SIGNAL_PRESENT"
    if pooled_auroc is not None and pooled_auroc >= 0.70 and not (delta_gate and contrast_gate):
        return "B0_SOURCE_HETEROGENEITY_DOMINANT"
    if historical_paired_positive:
        return "B0_PAIRED_ONLY_NOT_NO_REFERENCE"
    return "PILOT_INCONCLUSIVE"


def sign_test(values: Iterable[float]) -> dict[str, Any]:
    array = np.asarray(list(values), dtype=np.float64)
    array = array[np.isfinite(array)]
    array = array[array != 0]
    n = int(array.size)
    if not n:
        return {"N_nonzero": 0, "positive": 0, "negative": 0, "p_two_sided": None}
    positive = int(np.sum(array > 0))
    negative = n - positive
    tail = sum(math.comb(n, k) for k in range(min(positive, negative) + 1)) / (2.0**n)
    return {"N_nonzero": n, "positive": positive, "negative": negative, "p_two_sided": float(min(1.0, 2 * tail))}
