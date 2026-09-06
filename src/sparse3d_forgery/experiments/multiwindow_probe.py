"""Label-blind fixed multi-window coverage probe helpers."""

import numpy as np

from .temporal_probe import rank_auroc


def fixed_window_starts(frame_count: int, window_length: int = 16, count: int = 5) -> tuple[int, ...]:
    """Return unique floor-spaced starts without consulting labels or scores."""

    if frame_count < window_length:
        return ()
    if count < 2:
        raise ValueError("count must be at least two")
    span = frame_count - window_length
    return tuple(dict.fromkeys((index * span) // (count - 1) for index in range(count)))


def exact_artifact_match(frame_indices: np.ndarray, requested: tuple[int, ...]) -> bool:
    return np.array_equal(frame_indices, np.asarray(requested, dtype=np.int64))


def eligible_window_records(records: list[dict]) -> list[dict]:
    """Keep only structurally causal-eligible windows, never residual-filtered ones."""

    return [record for record in records if str(record.get("status", "")).startswith("eligible")]


def arithmetic_window_score(errors: np.ndarray, validity: np.ndarray) -> float:
    selected = np.asarray(errors)[np.asarray(validity, dtype=np.bool_)]
    if not selected.size:
        raise ValueError("window has no valid particle-horizon errors")
    return float(selected.mean())


def aggregate_video_scores(scores: list[float]) -> dict[str, float | int]:
    if not scores:
        raise ValueError("video has no eligible window score")
    values = np.asarray(scores, dtype=np.float64)
    return {
        "score_max": float(values.max()),
        "score_mean_secondary": float(values.mean()),
        "max_window_offset": int(values.argmax()),
    }


def actual_time_spans(timestamps_s: np.ndarray, history_count: int = 8) -> dict:
    timestamps = np.asarray(timestamps_s, dtype=np.float64)
    if (
        timestamps.shape != (16,)
        or not np.isfinite(timestamps).all()
        or not np.all(np.diff(timestamps) > 0)
    ):
        raise ValueError("timestamps must be 16 finite strictly increasing values")
    cutoff = history_count - 1
    return {
        "window_duration_s": float(timestamps[-1] - timestamps[0]),
        "history_duration_s": float(timestamps[cutoff] - timestamps[0]),
        "horizon_delta_s": {
            str(horizon): float(timestamps[cutoff + horizon] - timestamps[cutoff])
            for horizon in (1, 2, 4, 8)
        },
    }


def bootstrap_auroc_difference(
    real_candidate: np.ndarray,
    fake_candidate: np.ndarray,
    real_baseline: np.ndarray,
    fake_baseline: np.ndarray,
    seed: int = 20260906,
    replicates: int = 10_000,
) -> dict:
    arrays = [
        np.asarray(value, dtype=np.float64)
        for value in (real_candidate, fake_candidate, real_baseline, fake_baseline)
    ]
    if arrays[0].shape != arrays[2].shape or arrays[1].shape != arrays[3].shape:
        raise ValueError("candidate and baseline video scores must be paired")
    generator = np.random.default_rng(seed)
    deltas = np.empty(replicates, dtype=np.float64)
    for index in range(replicates):
        real_indices = generator.integers(0, arrays[0].size, arrays[0].size)
        fake_indices = generator.integers(0, arrays[1].size, arrays[1].size)
        deltas[index] = rank_auroc(
            arrays[0][real_indices], arrays[1][fake_indices]
        ) - rank_auroc(arrays[2][real_indices], arrays[3][fake_indices])
    return {
        "point_estimate": rank_auroc(arrays[0], arrays[1])
        - rank_auroc(arrays[2], arrays[3]),
        "ci95": [float(value) for value in np.percentile(deltas, (2.5, 97.5))],
    }


def classify_probe(aurocs: dict[str, dict[str, float]], fake_ratio_higher: bool) -> str:
    t0_max = aurocs["t0"]["max"]
    t0_center = aurocs["t0"]["center"]
    zero_max = aurocs["zero"]["max"]
    spatial_max = aurocs["spatial"]["max"]
    if (
        t0_max >= 0.60
        and t0_max - t0_center >= 0.10
        and t0_max - zero_max >= 0.05
        and fake_ratio_higher
    ):
        return "CENTER_WINDOW_COVERAGE_SUPPORTED"
    if zero_max >= 0.60 and t0_max - zero_max < 0.05 and spatial_max - zero_max < 0.05:
        return "MOTION_MAGNITUDE_CONFOUND"
    if t0_max <= 0.55 and t0_max - t0_center < 0.05 and spatial_max <= 0.55:
        return "CENTER_WINDOW_NOT_MAIN_CAUSE"
    return "MIXED_INCONCLUSIVE"
