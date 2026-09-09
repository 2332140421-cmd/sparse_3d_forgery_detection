"""Fixed, source-disjoint logistic probe over the frozen four-dimensional B0."""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression

from research_tools.v7.b0_no_reference.analyze_b0_no_reference import (
    BOOTSTRAP_REPLICATES,
    BOOTSTRAP_SEED,
    auroc,
    bootstrap_median,
    finite_summary,
    positive_fraction,
)


MODEL_CONFIG = {
    "model": "logistic_regression",
    "penalty": "l2",
    "C": 1.0,
    "solver": "lbfgs",
    "fit_intercept": True,
    "max_iter": 1000,
    "tol": 1e-6,
    "class_weight": None,
    "random_state": None,
}
STANDARDIZATION_VARIANCE_FLOOR = 1e-12


def main_training_label(row: Mapping[str, Any]) -> int | None:
    """Return the predeclared label for a manipulation window only."""

    if str(row.get("kind")) != "MANIP":
        return None
    role = str(row.get("role"))
    if role == "real":
        return 0
    if role == "fake":
        return 1
    return None


def validate_main_rows(rows: Sequence[Mapping[str, Any]]) -> None:
    labels = [main_training_label(row) for row in rows]
    if not rows or any(label is None for label in labels):
        raise ValueError("main probe rows must be real/fake MANIP windows")
    source_classes: dict[str, set[int]] = {}
    for row, label in zip(rows, labels):
        source_classes.setdefault(str(row["source_id"]), set()).add(int(label))
    if any(classes != {0, 1} for classes in source_classes.values()):
        raise ValueError("every training source must have both real and fake MANIP windows")


def source_class_weights(rows: Sequence[Mapping[str, Any]]) -> np.ndarray:
    """Equalize source totals and give real/fake half of each source total."""

    validate_main_rows(rows)
    sources = sorted({str(row["source_id"]) for row in rows})
    source_count = len(sources)
    counts: dict[tuple[str, int], int] = {}
    for row in rows:
        key = (str(row["source_id"]), int(main_training_label(row)))
        counts[key] = counts.get(key, 0) + 1
    weights = np.asarray(
        [1.0 / (2.0 * source_count * counts[(str(row["source_id"]), int(main_training_label(row)))]) for row in rows],
        dtype=np.float64,
    )
    return weights * (weights.size / np.sum(weights))


@dataclass(frozen=True)
class WeightedStandardizer:
    mean: np.ndarray
    scale: np.ndarray
    zero_variance_dimensions: tuple[int, ...]

    def transform(self, values: Iterable[Iterable[float]]) -> np.ndarray:
        array = np.asarray(values, dtype=np.float64)
        if array.ndim != 2 or array.shape[1] != 4:
            raise ValueError("B0 matrix must have shape [N,4]")
        return (array - self.mean) / self.scale

    def as_dict(self) -> dict[str, Any]:
        return {
            "mean": self.mean.tolist(),
            "scale": self.scale.tolist(),
            "zero_variance_dimensions": list(self.zero_variance_dimensions),
            "variance_floor": STANDARDIZATION_VARIANCE_FLOOR,
        }


def fit_weighted_standardizer(values: Iterable[Iterable[float]], weights: Iterable[float]) -> WeightedStandardizer:
    array = np.asarray(values, dtype=np.float64)
    weight = np.asarray(list(weights), dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != 4 or weight.shape != (array.shape[0],) or not np.all(np.isfinite(array)):
        raise ValueError("finite B0 matrix and matching weights are required")
    if np.any(weight < 0) or not np.isfinite(np.sum(weight)) or np.sum(weight) <= 0:
        raise ValueError("weights must be non-negative with positive finite sum")
    total = float(np.sum(weight))
    mean = np.sum(array * weight[:, None], axis=0) / total
    variance = np.sum(((array - mean) ** 2) * weight[:, None], axis=0) / total
    zero = tuple(int(index) for index in np.flatnonzero(~np.isfinite(variance) | (variance <= STANDARDIZATION_VARIANCE_FLOOR)))
    scale = np.sqrt(np.maximum(variance, STANDARDIZATION_VARIANCE_FLOOR))
    scale[list(zero)] = 1.0
    return WeightedStandardizer(mean=mean, scale=scale, zero_variance_dimensions=zero)


@dataclass(frozen=True)
class FoldModel:
    held_out_source: str
    standardizer: WeightedStandardizer
    coefficient: np.ndarray
    intercept: float
    training_source_count: int
    training_real_count: int
    training_fake_count: int
    sample_weight_mean: float
    n_iter: int
    convergence_status: str

    def score(self, values: Iterable[Iterable[float]]) -> np.ndarray:
        transformed = self.standardizer.transform(values)
        return transformed @ self.coefficient + self.intercept

    def as_dict(self) -> dict[str, Any]:
        return {
            "held_out_source": self.held_out_source,
            "standardization": self.standardizer.as_dict(),
            "coefficient": self.coefficient.tolist(),
            "intercept": self.intercept,
            "training_source_count": self.training_source_count,
            "training_real_count": self.training_real_count,
            "training_fake_count": self.training_fake_count,
            "sample_weight_mean": self.sample_weight_mean,
            "model_config": dict(MODEL_CONFIG),
            "n_iter": self.n_iter,
            "convergence_status": self.convergence_status,
        }


def fit_fold(rows: Sequence[Mapping[str, Any]], held_out_source: str) -> FoldModel:
    train = [row for row in rows if str(row["source_id"]) != str(held_out_source)]
    validate_main_rows(train)
    values = np.asarray([row["b0"] for row in train], dtype=np.float64)
    labels = np.asarray([main_training_label(row) for row in train], dtype=np.int64)
    weights = source_class_weights(train)
    standardizer = fit_weighted_standardizer(values, weights)
    transformed = standardizer.transform(values)
    sklearn_config = {key: value for key, value in MODEL_CONFIG.items() if key != "model"}
    model = LogisticRegression(**sklearn_config)
    warning_messages: list[str] = []
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        model.fit(transformed, labels, sample_weight=weights)
    warning_messages.extend(str(item.message) for item in caught)
    needs_retry = bool(warning_messages) or int(np.max(model.n_iter_)) >= MODEL_CONFIG["max_iter"]
    status = "CONVERGED" if not needs_retry else "MAX_ITER_RETRY"
    if needs_retry:
        retry_config = dict(sklearn_config)
        retry_config["max_iter"] = 2000
        model = LogisticRegression(**retry_config)
        with warnings.catch_warnings(record=True) as caught_retry:
            warnings.simplefilter("always", ConvergenceWarning)
            model.fit(transformed, labels, sample_weight=weights)
        warning_messages.extend(str(item.message) for item in caught_retry)
        if caught_retry or int(np.max(model.n_iter_)) >= retry_config["max_iter"]:
            status = "NOT_CONVERGED_AFTER_ALLOWED_RETRY"
        else:
            status = "CONVERGED_AFTER_MAX_ITER_RETRY"
    if not hasattr(model, "coef_") or model.coef_.shape != (1, 4):
        raise ValueError("binary logistic probe did not produce four coefficients")
    n_iter = int(np.max(model.n_iter_))
    return FoldModel(
        held_out_source=str(held_out_source),
        standardizer=standardizer,
        coefficient=np.asarray(model.coef_[0], dtype=np.float64),
        intercept=float(model.intercept_[0]),
        training_source_count=len({str(row["source_id"]) for row in train}),
        training_real_count=int(np.sum(labels == 0)),
        training_fake_count=int(np.sum(labels == 1)),
        sample_weight_mean=float(np.mean(weights)),
        n_iter=n_iter,
        convergence_status=status,
    )


def source_auc(rows: Sequence[Mapping[str, Any]], score_key: str = "score") -> list[dict[str, Any]]:
    grouped: dict[str, dict[str, list[float]]] = {}
    for row in rows:
        if row.get("kind") != "MANIP" or row.get(score_key) is None:
            continue
        grouped.setdefault(str(row["source_id"]), {"real": [], "fake": []})[str(row["role"])].append(float(row[score_key]))
    output: list[dict[str, Any]] = []
    for source in sorted(grouped):
        real, fake = grouped[source]["real"], grouped[source]["fake"]
        output.append({
            "source_id": source,
            "real_manip_count": len(real),
            "fake_manip_count": len(fake),
            "auroc": auroc(fake, real),
        })
    return output


def paired_source_bootstrap(model_rows: Sequence[Mapping[str, Any]], model_key: str = "model_auroc", baseline_key: str = "baseline_auroc") -> dict[str, Any]:
    pairs = [(float(row[model_key]), float(row[baseline_key])) for row in model_rows if row.get(model_key) is not None and row.get(baseline_key) is not None]
    if not pairs:
        return {"N": 0, "seed": BOOTSTRAP_SEED, "replicates": BOOTSTRAP_REPLICATES, "model_mean": None, "baseline_mean": None, "delta_mean": None, "model_ci95": None, "baseline_ci95": None, "delta_ci95": None}
    model = np.asarray([pair[0] for pair in pairs], dtype=np.float64)
    baseline = np.asarray([pair[1] for pair in pairs], dtype=np.float64)
    delta = model - baseline
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    indices = rng.integers(0, len(pairs), size=(BOOTSTRAP_REPLICATES, len(pairs)))
    model_boot = np.mean(model[indices], axis=1)
    baseline_boot = np.mean(baseline[indices], axis=1)
    delta_boot = np.mean(delta[indices], axis=1)
    return {
        "N": int(len(pairs)),
        "seed": BOOTSTRAP_SEED,
        "replicates": BOOTSTRAP_REPLICATES,
        "model_mean": float(np.mean(model)),
        "baseline_mean": float(np.mean(baseline)),
        "delta_mean": float(np.mean(delta)),
        "model_ci95": [float(np.percentile(model_boot, 2.5)), float(np.percentile(model_boot, 97.5))],
        "baseline_ci95": [float(np.percentile(baseline_boot, 2.5)), float(np.percentile(baseline_boot, 97.5))],
        "delta_ci95": [float(np.percentile(delta_boot, 2.5)), float(np.percentile(delta_boot, 97.5))],
    }


def source_group_medians(rows: Sequence[Mapping[str, Any]], score_key: str = "score") -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[float]] = {}
    for row in rows:
        value = row.get(score_key)
        if value is None:
            continue
        grouped.setdefault((str(row["source_id"]), str(row["role"]), str(row["kind"])), []).append(float(value))
    sources = sorted({key[0] for key in grouped})
    output: list[dict[str, Any]] = []
    for source in sources:
        values = {(role, kind): grouped.get((source, role, kind), []) for role in ("real", "fake") for kind in ("MANIP", "CTRL")}
        row: dict[str, Any] = {"source_id": source}
        for (role, kind), group in values.items():
            row[f"A_{role[0].upper()}M" if kind == "MANIP" else f"A_{role[0].upper()}C"] = float(np.median(group)) if group else None
        if all(row.get(key) is not None for key in ("A_RM", "A_FM", "A_RC", "A_FC")):
            row["Delta_manip"] = row["A_FM"] - row["A_RM"]
            row["Delta_control"] = row["A_FC"] - row["A_RC"]
            row["J_B0"] = row["Delta_manip"] - row["Delta_control"]
        else:
            row["Delta_manip"] = row["Delta_control"] = row["J_B0"] = None
        output.append(row)
    return output


def summary_for_deltas(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key in ("Delta_manip", "Delta_control", "J_B0"):
        values = [row[key] for row in rows if row.get(key) is not None]
        output[key] = {**finite_summary(values), "positive_fraction": positive_fraction(values), "bootstrap": bootstrap_median(values)}
    return output


def primary_status(bootstrap: Mapping[str, Any], blocked: bool = False) -> str:
    if blocked:
        return "B0_LINEAR_PROBE_BLOCKED"
    model_ci = bootstrap.get("model_ci95")
    delta_ci = bootstrap.get("delta_ci95")
    if model_ci and delta_ci and model_ci[0] > 0.5 and delta_ci[0] > 0:
        return "B0_LINEAR_DISCRIMINATION_SUPPORTED_IN_PILOT"
    if model_ci and model_ci[0] > 0.5:
        return "B0_LINEAR_SIGNAL_WITHOUT_CLEAR_BASELINE_GAIN"
    return "B0_LINEAR_DISCRIMINATION_NOT_ESTABLISHED"
