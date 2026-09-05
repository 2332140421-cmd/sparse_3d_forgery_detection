# ADR 0008: VGGT feasibility frontend

## Status

Accepted

## Context

The first displacement-prediction experiment needs real sparse tracks, depth, camera motion compensation, and reusable validated `ParticleSequence` artifacts. The implementation must remain small and must not turn a replaceable provider into the research method.

## Decision

- Use official VGGT code revision `a288dd0f14786c93483e45524328726ab7b1b4ce` and `facebook/VGGT-1B` weight revision `860abec7937da0a4c03c41d3c269c366e82abdf9` for frontend feasibility.
- Interpret VGGT depth as camera z-depth in its sequence-level arbitrary scale and its OpenCV extrinsics as world-to-camera. Invert those extrinsics to lift tracked pixels into the VGGT world gauge, with axes anchored to the first-camera right, down, and forward convention.
- Preserve decoder RGB. The adapter explicitly applies aspect-preserving resize and centered padding, maps deterministic first-frame query points into provider coordinates, and maps resulting UV back to original video coordinates.
- Visibility is thresholded from the provider visibility output plus finite source-image bounds. Geometry validity additionally requires positive finite depth, valid intrinsics and extrinsics, and finite lifted coordinates. Provider confidence is not a model input.
- VGGT jointly aggregates every supplied frame. These artifacts are therefore marked `causal_training_eligible=false`; they demonstrate frontend feasibility but cannot be used for formal future-prediction training until a prefix/window path prevents future frames from revising historical inputs and aligns future targets to the historical coordinate gauge.
- Persist current research artifacts as numeric NPZ arrays plus JSON metadata. Object arrays and pickle are forbidden, and every round trip must pass the existing validator.

## Consequences

Real videos can produce reusable, camera-compensated sparse 3D observations with explicit time, coordinates, validity, lineage, and provenance. VGGT's arbitrary scale and joint-sequence inference remain limitations. Model architecture, prediction horizons, particle count, visibility threshold, provider replacement, and a causal training pipeline remain unfrozen.

## Supersedes

This ADR resolves the initial feasibility provider and current artifact-format questions only. It does not supersede ADR 0001 through ADR 0007 or alter the frozen research chain, `ParticleSequence` schema, mask boundary, or spatial candidate-topology decision.
