# ADR 0012: V7 paired second-order structural signal pilot

## Status

Accepted for the bounded ActivityForensics--Charades paired pilot on
`v7-dynamic-structure`. This is an experiment protocol and result record; it
does not freeze the V7 method or authorize formal normality training.

## Context

The reviewed population contains 30 exact real/fake pairs with stable timeline
metadata and official manipulation intervals. The first falsifiable V7 question
is whether matched manipulation windows show a reproducible structural
discrepancy beyond matched unmodified control windows.

## Decision

- Select the first 16 technically eligible pairs in frozen review-manifest order;
  do not use frontend quality or measured signal to select examples.
- Use the unchanged causal Online BootsTAPIR, Apple Depth Pro, causal
  first-frame focal assumption, and adjacent Open3D RGB-D odometry baseline.
- Use one-second true-PTS windows at 25%, 50%, and 75% anchors in the longest
  official manipulation segment and the nearest valid unmodified control block.
- Compare matched real/fake windows with the existing `S_t`, `Delta S_t`, and
  `Delta2 S_t` observations. Robust component scales are fitted from real
  control windows only; fake observations have fitting count zero.
- Report paired `D_manip`, `D_ctrl`, `G = D_manip - D_ctrl`, `A21 = G2 - G1`,
  and `A20 = G2 - G0`, with deterministic pair-level bootstrap seed
  `20260909` and 10,000 replicates. AUROC is exploratory only.
- Preserve every frontend failure and quality measurement. No score-based
  filtering, Gaussian normality model, supervised detector, or formal training
  is introduced.

## Causal boundary

Only causal frontend windows are eligible for the pilot. This status does not
claim that a future provider or a later normality model is causal.

## Consequences

The pilot produces auditable data-disk manifests, NPZ/JSON particle artifacts,
per-window and per-pair metrics, quality confounds, and generator/operation
descriptives. A negative or mixed result is a representation/measurement
falsification outcome, not a license to tune the protocol on fake labels.

## Supersedes

None. This ADR supplements ADR 0010 and ADR 0011 without changing the V7
contract, the explicit frontend baseline, or the formal `src/` package.
