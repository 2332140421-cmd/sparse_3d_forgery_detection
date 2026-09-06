# ADR 0009: History-anchored causal VGGT window

## Status

Accepted

## Context

VGGT jointly aggregates all supplied frames, so a temporal model's causal mask cannot prevent future images from changing historical frontend coordinates. Separate history-only and extended runs also use different arbitrary-scale gauges, which must be aligned without removing real future motion.

## Decision

- Pass A sees only the recorded history through one cutoff; its XYZ, UV, visibility, and geometry validity are the final history arrays.
- Pass B sees the complete window but contributes only future rows to the final artifact.
- Fit one deterministic global Umeyama Sim(3), mapping Pass B to Pass A, from same-track, same-timestamp, jointly geometry-valid historical observations only. Dynamic historical particles are valid correspondences because each observation is matched to itself at the same time; no static-scene assumption is made.
- Sim(3), rather than SE(3), is required because separate VGGT runs can choose different arbitrary scales. The fit requires non-degenerate geometry, positive scale, finite parameters, and a proper rotation.
- Future points, labels, anomaly scores, provider confidence, and motion-based filtering are forbidden in the fit. Per-frame, per-particle, ICP, and silent identity fallback are forbidden.
- On fit failure, preserve Pass A history, invalidate all future geometry, retain NaN missing semantics, and set `causal_training_eligible=false`.
- Eligibility applies only to the cutoff and future frame set recorded in lineage/provenance. Alignment residuals are label-free frontend diagnostics, never anomaly evidence.
- Normalized alignment RMSE is divided by the Pass A historical correspondence cloud's RMS radius about its centroid.

## Consequences

Successful artifacts satisfy the intended information-flow and coordinate-gauge boundary for their specific cutoff. This construction does not establish VGGT three-dimensional accuracy and does not by itself make the complete training pipeline ready.

## Supersedes

This ADR supersedes only ADR 0008's statement that the causal prefix/window alignment path remains unresolved. It does not change VGGT's replaceable status, the Particle definition, XYZ-only continuous input, mask boundary, unfrozen spatial topology, or future-displacement prediction chain. ADR 0008 otherwise remains accepted.
