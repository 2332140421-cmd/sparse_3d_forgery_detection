# V7 Phase 1: Explicit geometry frontend feasibility plan

## Purpose

This phase will determine whether a minimal explicit geometry frontend can measure causal, common-coordinate persistent particle XYZ accurately enough to study dynamic-structure evolution. It is a measurement feasibility test, not detector training and not a provider-selection result.

No model, weight or dataset is downloaded by this plan.

## Candidate audit scope

The next task must inspect a small runnable set of official candidates for:

- fixed or causal two-dimensional tracking;
- per-frame depth;
- camera intrinsics;
- causal camera pose.

A single provider may cover multiple responsibilities, but the implementation must remain an explicit, inspectable chain rather than a provider registry or generalized framework.

Each candidate must have:

- official source and pinned full revision;
- explicit academic/noncommercial research license covering source and required weights;
- checkpoint identity, byte size and SHA-256 before runtime acceptance;
- compatibility with the current server or a justified isolated environment;
- information flow in which history is never revised by later images;
- explicit image resize/crop and original-UV mapping;
- output sufficient for same tracked UV to obtain XYZ;
- a documented coordinate frame, handedness, axes and length/scale semantics;
- a clear conversion into validated ParticleSequence;
- bounded runtime, peak-memory and preprocessing cost.

## Fixed scientific questions

1. Does the 2D tracker provide sufficient persistent coverage over the intended structured-motion windows?
2. How large is per-frame depth temporal jitter for the same observation?
3. How large is causal camera-pose drift?
4. After explicit back-projection and camera compensation, is XYZ repeatable, causal, common-coordinate and temporally stable?
5. How large is frontend noise relative to particle motion and component structural change?
6. Is the result materially better than the existing V6 VGGT baseline?

The preserved VGGT reference is real-primary median `Q_target`: h1=`1.6056`, h2=`0.9326`, h4=`0.4278`, together with the accepted repeatability, prefix-alignment and correlation reports. No VGGT detector score or fake AUROC selects a geometry provider.

## Minimum comparison protocol

- Use only a preregistered real-only sample for provider selection.
- Decode the same source frames with true PTS timestamps.
- Supply the same deterministic UV tracks to every compatible geometry path.
- Run each candidate twice on identical input for repeatability.
- Compare history-only and longer processing to verify future invariance without first applying alignment.
- Measure finite depth/XYZ coverage, persistent-track lengths and horizon-specific joint validity.
- Test depth jitter and pose drift separately before attributing error to particle motion.
- Verify synthetic back-projection, transform direction and pixel-center mapping.
- Permit at most one global proper relation only when the provider's declared scale requires it; never fit future points, local warps, ICP or BA to pass the gate.
- Report elapsed time, peak GPU memory, environment and checkpoint metadata.

Reprojection may diagnose calibration or transform failure, but it cannot become anomaly evidence or determine a validity threshold using fake performance.

## Required outputs

The feasibility audit must produce a compact, reproducible manifest and report:

- candidate source/license/checkpoint evidence;
- selected real videos, exact frames and UV-track artifact identities;
- preprocessing and coordinate transformations;
- tracking coverage and re-observation consistency;
- depth jitter and pose drift;
- XYZ repeatability, future invariance and common-coordinate diagnostics;
- frontend-noise-to-particle-motion and frontend-noise-to-structural-change ratios, described as diagnostics rather than anomaly scores;
- runtime, memory, failures and explicit invalid masks;
- comparison with V6 VGGT and a preregistered pass/fail/inconclusive result.

## Stop conditions

Stop before download or runtime if source/weight license is unresolved. Stop a provider path if it revises history with future frames, cannot map original same-UV observations, lacks a usable common coordinate relation, silently fills invalid geometry, or exceeds bounded server resources.

## Excluded work

Phase 1 does not implement component clustering, `S_t`, `ΔS_t`, `Δ²S_t`, a density/normality model, anomaly aggregation, detector training or fake evaluation. It does not restore handcrafted velocity, acceleration, curvature, reprojection anomaly, occlusion anomaly or physical residual rules. It does not select a provider in advance.
