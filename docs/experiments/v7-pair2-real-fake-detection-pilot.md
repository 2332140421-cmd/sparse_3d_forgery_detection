# V7 Pair2 real/fake detection pilot

This report records the frozen Pair2 real/fake detection experiment.  The
final numeric result is written after the materialized media run; this file is
not a design change.

## Frozen protocol

The real-only M1/M2/M3 Gaussian normality artifacts under
`v7_real_only_normality_pilot_v1` are read without refitting.  M1 is `ΔS`, M2
is `Δ²S`, and M3 is their concatenation.  The experiment uses the first eight
frozen HD-VG-130M source identities, exact official Pair2 lineage, 1.0 s true
timestamp windows at 25%, 50%, and 75%, unchanged frontend/components, and
video p95 as the primary aggregation.  Fake media never enters fitting.

## Results

The generated data-disk report is copied here after the run.  If the required
media population cannot be materialized, the status is
`PAIR2_MEDIA_ACCESS_BLOCKED`; no H3 conclusion is claimed from missing media.

Spatial localization is not evaluated because Pair2 has no reliable spatial
fake ground truth (`SPATIAL_LOCALIZATION_GT_NOT_AVAILABLE`).  No synthetic
perturbation, localization proxy, fake-aware threshold, model refit, component
change, or method tuning is performed.
