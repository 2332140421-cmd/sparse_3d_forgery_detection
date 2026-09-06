# Multi-window coverage falsification probe

## Scope and frozen protocol

Source HEAD: `9e67ca5fdc33398a29dab0e2f2369b8a3e35d399`.

This label-blind probe asks whether the previous single center window was the
main reason frozen 3D prediction evidence did not separate real and fake
videos. It reused the exact 32 real-validation and 32 fake-probe video IDs,
T0/T1/S checkpoints, normalization, 8-frame history, horizons `{1,2,4,8}`,
128 particles, loss, and arithmetic-mean window score. No model was trained.

For declared frame count `F`, the five starts were
`floor(k*(F-16)/4), k=0..4`, followed only by deterministic deduplication. All
64 videos had five unique windows. Each k=2 window exactly matched and reused
the preceding center artifact. The other 256 windows were generated with the
accepted history-anchored causal VGGT construction. All 320 windows were
causal eligible; none failed. Fake annotations were never read.

The pre-registered primary video score was the maximum of eligible window
scores. Per-video mean remained secondary and was not used to select the
reported result.

## Actual time scale

All values come from `ParticleSequence.timestamps_s`, not FPS arithmetic.

| Span | min | median | mean | p90 | max (seconds) |
| --- | ---: | ---: | ---: | ---: | ---: |
| 16-frame window | 0.500000 | 0.500250 | 0.517400 | 0.625000 | 0.625521 |
| 8-frame history | 0.233333 | 0.233450 | 0.241453 | 0.291667 | 0.291910 |
| h=1 | 0.033333 | 0.033350 | 0.034493 | 0.041667 | 0.041701 |
| h=2 | 0.066667 | 0.066700 | 0.068987 | 0.083333 | 0.083403 |
| h=4 | 0.133333 | 0.133400 | 0.137973 | 0.166667 | 0.166806 |
| h=8 | 0.266667 | 0.266800 | 0.275947 | 0.333333 | 0.333611 |

`SHORT_TEMPORAL_SCALE_LIMITATION` is **undetermined** because no post-hoc
duration threshold was introduced. Objectively, this probe tests windows no
longer than 0.626 seconds and h=8 no longer than 0.334 seconds; longer temporal
scales remain untested.

## Frontend diagnostics

Normalized Sim(3) aligned RMSE is summarized as min / median / mean / p90 /
max. Real validation was `0.00738 / 0.05884 / 0.07701 / 0.15465 / 0.38126`;
fake probe was `0.00823 / 0.06568 / 0.10639 / 0.24165 / 0.67930`.
Correspondence counts were 538--1024 for real and 997--1024 for fake. All
rotation determinants were numerically one. Edge-window normalized residuals
were not higher than interior-window residuals as a group, so no edge-specific
frontend instability was observed.

Window scores nevertheless correlated strongly with normalized alignment
residual: T0 correlations were `0.8522` for real and `0.9380` for fake (Zero:
`0.8550/0.9393`; S: `0.8569/0.9359`). This diagnostic is not an anomaly score
and was not used to filter windows. It limits any claim that larger max-window
scores represent learned normality rather than frontend/gauge instability.

## Detection results

| Predictor | Center AUROC | Multi-window max AUROC | Delta |
| --- | ---: | ---: | ---: |
| Zero | 0.4902 | 0.5547 | +0.0645 |
| Linear | 0.4619 | 0.5146 | +0.0527 |
| T0 | 0.4824 | 0.5391 | +0.0566 |
| T1 | 0.4824 | 0.5381 | +0.0557 |
| S | 0.4775 | 0.5293 | +0.0518 |

The learned-normality contrasts were T0 max minus Zero max `-0.0156` and S max
minus Zero max `-0.0254`.

With 10,000 deterministic video-level bootstrap replicates:

- T0 max minus center: `+0.0566`, 95% CI `[-0.0596, 0.1768]`.
- S max minus center: `+0.0518`, 95% CI `[-0.0625, 0.1690]`.
- T0 max minus Zero max: `-0.0156`, 95% CI `[-0.0342, -0.0020]`.
- S max minus Zero max: `-0.0254`, 95% CI `[-0.0508, -0.0049]`.

For T0, max/center ratios (median / p75 / p90) were
`1.3951 / 2.0739 / 3.5298` for real and
`1.7608 / 2.3550 / 2.8752` for fake. T0 maximum-window position counts
`w0..w4` were `6/2/10/10/4` for real and `5/11/4/7/5` for fake. These are
localization-free diagnostics and are not ground-truth temporal claims.

## Pre-registered interpretation

Classification: **MIXED_INCONCLUSIVE**.

Scanning five windows raised T0 AUROC by about 0.057, so the strict
`CENTER_WINDOW_NOT_MAIN_CAUSE` condition was not met. However, T0 max AUROC
remained 0.539, below Zero at 0.555, and the bootstrap learned-minus-Zero
contrast was negative. The requirements for `CENTER_WINDOW_COVERAGE_SUPPORTED`
and `MOTION_MAGNITUDE_CONFOUND` were also not met. Thus center coverage may
affect scores, but this experiment does not show that it was the main cause of
the earlier failures or that the change reflects learned normality.

The experiment did not use annotations, change any model or method setting,
choose aggregation after seeing results, or establish detector effectiveness.
It tests only the fixed short-time, five-window protocol above.
