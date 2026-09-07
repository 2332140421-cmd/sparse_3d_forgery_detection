# ADR 0011: V7 real-only structural normality pilot

## Status

Accepted for the bounded V7 H3 experiment on branch `v7-dynamic-structure`.
This record describes the reproducible pilot protocol; it does not freeze the
V7 method beyond the existing candidate contract.

## Context

The preceding private feasibility run supported continuing the current explicit
frontend at a 1.0 s physical window. The next falsifiable question is whether
current `S_t`, `ΔS_t`, and `Δ²S_t` observations form a real-only normality
distribution and whether that distribution can later be tested on paired fake
media.

## Decision

- Use the existing causal Online BootsTAPIR tracker, Apple Depth Pro, causal
  first-frame focal/intrinsics assumption, and adjacent Open3D RGB-D odometry.
  No provider registry or alternative backend is introduced.
- Use 64 regular-grid particles, the existing ComponentConfig
  (`1.0 m`, `0.05 m`, minimum overlap `8`, minimum size `3`), the existing
  normalized pairwise-distance `S_t`, and the existing timestamp-aware
  `ΔS_t`/`Δ²S_t` implementation without modification.
- Use 1.0 s true-PTS windows at source-relative anchors 25%, 50%, and 75%; no
  padding, sorting, or event-based selection is allowed.
- Fit three independent real-only Gaussian/Mahalanobis baselines: `M1=ΔS`,
  `M2=Δ²S`, and `M3=[ΔS,Δ²S]`. Fake media never enters fitting. Covariance
  regularization is `λ = 1e-5 * max(mean(diag(Σ)), 1e-12)`; a diagonal fallback
  is deterministic when the regularized full covariance is underdetermined or
  has condition number above `1e12`.
- The primary video aggregation is p95 of component-time scores; median is a
  sensitivity result. Scores are not summed over components.
- Coordinates remain the existing first-camera world, right-handed
  `(right, down, forward)` axes, metres, and optical-axis z-depth interpretation.
  The pose convention remains target-camera-from-source-camera with inversion
  during world accumulation.
- Particle artifacts use the existing NPZ plus JSON serialization on the data
  disk. This ADR does not establish a universal artifact format.

## Causal boundary

The pilot executes tracking with causal state, depth per frame, and adjacent
pose accumulation in source-time order. A window is eligible only when its
history is produced without future frames. This is an experiment eligibility
claim, not a proof that every future provider is causal.

## Consequences

This protocol makes the real-only fitting population, feature comparison,
aggregation, coordinate declaration, and reproducible artifacts explicit. It
does not add a supervised detector, a synthetic perturbation, spatial ground
truth, or a new formal `src/` method. Pair2 fake media remains a separate
bounded evaluation input and must not alter fitting choices.

## Supersedes

None. This experiment record does not modify ADR 0010 or any earlier V7
decision.
