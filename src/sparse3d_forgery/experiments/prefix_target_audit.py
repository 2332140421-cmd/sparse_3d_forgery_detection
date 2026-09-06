"""Per-horizon causal target reliability audit helpers."""

from dataclasses import replace

import numpy as np

from sparse3d_forgery.particle_sequence import ParticleSequence, validate_particle_sequence


HORIZONS = (1, 2, 4, 8)
PREFIX_INFERENCE_HORIZONS = (1, 2, 4)


def prefix_frame_indices(
    frame_indices: np.ndarray, history_count: int, horizon: int
) -> tuple[int, ...]:
    """Select history through exactly its target, excluding later future frames."""

    if horizon not in HORIZONS:
        raise ValueError("horizon must be one of 1, 2, 4, 8")
    indices = np.asarray(frame_indices)
    required = history_count + horizon
    if indices.ndim != 1 or history_count <= 0 or len(indices) < required:
        raise ValueError("frame_indices cannot supply the requested prefix")
    return tuple(int(value) for value in indices[:required])


def fixed_history_anchor(sequence: ParticleSequence, history_count: int = 8) -> ParticleSequence:
    """Recover the immutable Pass-A rows already embedded in a causal artifact."""

    validate_particle_sequence(sequence)
    if sequence.lineage.get("construction") != "history-anchored causal VGGT window":
        raise ValueError("artifact is not a history-anchored causal window")
    if sequence.lineage.get("history_count") != history_count:
        raise ValueError("artifact history_count is incompatible")
    anchor = replace(
        sequence,
        sample_id=f"{sequence.sample_id}:fixed-pass-a",
        frame_indices=sequence.frame_indices[:history_count].copy(),
        timestamps_s=sequence.timestamps_s[:history_count].copy(),
        frame_sizes_hw=sequence.frame_sizes_hw[:history_count].copy(),
        xyz=sequence.xyz[:history_count].copy(),
        uv=sequence.uv[:history_count].copy(),
        visibility=sequence.visibility[:history_count].copy(),
        geometry_validity=sequence.geometry_validity[:history_count].copy(),
        lineage=sequence.lineage["pass_a_history_only"],
        provenance={"fixed_anchor_reused": True},
    )
    validate_particle_sequence(anchor)
    return anchor


def compatibility_signature(sequence: ParticleSequence) -> dict:
    lineage = sequence.lineage["pass_a_history_only"]
    return {
        "provider": lineage.get("provider"),
        "code_revision": lineage.get("code_revision"),
        "weight_revision": lineage.get("weight_revision"),
        "weight_sha256": lineage.get("weight_sha256"),
        "query_initialization": lineage.get("query_initialization"),
        "history_frame_indices": tuple(sequence.lineage.get("history_frame_indices", ())),
        "history_count": sequence.lineage.get("history_count"),
        "track_count": sequence.num_tracks,
    }


def common_target_measurements(
    prefix_xyz: np.ndarray,
    prefix_validity: np.ndarray,
    full_xyz: np.ndarray,
    full_validity: np.ndarray,
    cutoff_xyz: np.ndarray,
    cutoff_validity: np.ndarray,
    epsilon: float = 1e-12,
) -> dict:
    """Compare prefix/full targets and motion on one identical validity set."""

    common_target = prefix_validity & full_validity
    common_motion = common_target & cutoff_validity
    disagreement = np.linalg.norm(
        prefix_xyz[common_target] - full_xyz[common_target], axis=-1
    ).astype(
        np.float64
    )
    motion = np.linalg.norm(
        prefix_xyz[common_motion] - cutoff_xyz[common_motion], axis=-1
    ).astype(np.float64)
    paired_disagreement = np.linalg.norm(
        prefix_xyz[common_motion] - full_xyz[common_motion], axis=-1
    ).astype(np.float64)
    ratio = None
    if motion.size:
        motion_rms = float(np.sqrt(np.mean(motion**2)))
        if motion_rms > epsilon:
            ratio = float(np.sqrt(np.mean(paired_disagreement**2))) / motion_rms
    return {
        "validity": common_target,
        "motion_validity": common_motion,
        "disagreement": disagreement,
        "paired_disagreement": paired_disagreement,
        "motion": motion,
        "q_target": ratio,
    }


def diagnostic_ratio(numerator: float, denominator: float, epsilon: float = 1e-12):
    return None if denominator <= epsilon else float(numerator / denominator)


def summarize(values: list[float]) -> dict:
    array = np.asarray(values, dtype=np.float64)
    if not array.size:
        return {"count": 0}
    return {
        "count": int(array.size),
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "rms": float(np.sqrt(np.mean(array**2))),
        "p75": float(np.percentile(array, 75)),
        "p90": float(np.percentile(array, 90)),
        "max": float(array.max()),
    }


def summarize_ratios(values: list[float]) -> dict:
    result = summarize(values)
    if not values:
        return result
    array = np.asarray(values, dtype=np.float64)
    result.update(
        {
            "fraction_lt_0_25": float(np.mean(array < 0.25)),
            "fraction_lt_0_5": float(np.mean(array < 0.5)),
            "fraction_lt_1": float(np.mean(array < 1.0)),
            "fraction_ge_1": float(np.mean(array >= 1.0)),
        }
    )
    return result


def classify_real(metrics: dict, repeatability_blocker: bool) -> str:
    if repeatability_blocker:
        return "RUNTIME_REPEATABILITY_BLOCKER"
    horizons = ("1", "2", "4")
    n8 = metrics["alignment"]["8"]["median"]
    alignment_25 = sum(metrics["alignment"][h]["median"] <= 0.75 * n8 for h in horizons)
    q_below = sum(metrics["q_target"][h]["median"] < 0.5 for h in horizons)
    correlation_drop = sum(
        abs(metrics["correlation"][h]["prefix"])
        <= abs(metrics["correlation"][h]["full"]) - 0.15
        for h in horizons
    )
    if alignment_25 >= 2 and q_below >= 2 and correlation_drop >= 2:
        return "PREFIX_TARGET_STABILITY_SUPPORTED"
    alignment_under_10 = sum(
        1.0 - metrics["alignment"][h]["median"] / n8 < 0.10 for h in horizons
    )
    q_high = sum(metrics["q_target"][h]["median"] >= 1.0 for h in horizons)
    correlation_bad = sum(
        abs(metrics["correlation"][h]["prefix"]) >= 0.75
        and abs(metrics["correlation"][h]["full"])
        - abs(metrics["correlation"][h]["prefix"])
        < 0.10
        for h in horizons
    )
    if alignment_under_10 >= 2 or q_high >= 2 or correlation_bad >= 2:
        return "FRONTEND_CONTEXT_INSTABILITY_PERSISTS"
    return "MIXED_INCONCLUSIVE"
