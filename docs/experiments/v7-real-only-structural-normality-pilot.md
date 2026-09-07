# V7 real-only structural normality pilot

## 1. Research question and hypothesis

**RQ1:** do current component relation states `S_t`, first-order change `ΔS_t`,
and second-order evolution `Δ²S_t` form a learnable real-video normality
distribution that remains stable on held-out real source videos?

**RQ2:** can a normality model fitted only on real videos separate paired
AI-generated videos? The hypothesis under test is H3 — Evolution Normality.

## 2. Data and split

The frozen 32 Vript train-real identities were materialized by exact official
Mutonix/Vript shard/member range extraction. All 32/32 were `MEDIA_VALID`; no
source was removed for component quality or model behavior. Frozen source order
produced 24 real-train sources and 8 real-validation sources. Each source has
three 1.0 s true-PTS windows at 25%, 50%, and 75% of its timeline, giving 72
train windows and 24 validation windows. The authoritative manifests are under
`/root/autodl-tmp/data/sparse_3d_forgery_detection/derived/v7_real_only_normality_pilot_v1/manifests/`.

## 3. Representation and unchanged baseline

The run uses the existing causal Online BootsTAPIR, Apple Depth Pro, first-frame
K assumption, adjacent Open3D RGB-D pose chain, 64 particles, current
`ComponentConfig`, normalized pairwise-distance `S_t`, and timestamp-aware
`ΔS_t`/`Δ²S_t`. No component, state, frontend, or derivative implementation
was changed. The output coordinate declaration is first-camera world,
right-handed `(right, down, forward)`, metres; depth is interpreted along the
optical z axis.

## 4. Real-only training protocol

For each valid component-time observation, the compared feature schemas are:

| model | feature | dimension | train fitting |
|---|---|---:|---|
| M1 | `ΔS_t` | 4 | real-train only |
| M2 | `Δ²S_t` | 4 | real-train only |
| M3 | `[ΔS_t, Δ²S_t]` | 8 | real-train only |

Each model is a Gaussian/Mahalanobis baseline with fixed covariance
regularization. The fitted models and score files are in `models/` and
`scores/` below the derived output root. No fake feature or label entered fitting.
The primary video score is component-time p95; median is retained as a
sensitivity aggregation and no score sum is used.

## 5. Frontend and representation coverage

All 96 windows completed. There were 88/96 windows with at least one current
component (91.7%); 8 windows had no component and therefore no normality feature
for that window. Across component-time slots, `S_t` validity was 4,160/4,458
(93.3%). There were 3,978 valid `ΔS` observations and 3,806 valid `Δ²S`
observations. Final geometry
coverage median was 0.931 (IQR 0.196; p10–p90 0.671–1.000). Total frontend
wall time was 939.5 s and peak allocated GPU memory was 4,200,524,288 bytes
(about 4.20 GB). Details are in `metrics/frontend_summary.json`.

## 6. Held-out real results

The following distributions are source-level medians over the three windows;
therefore a source with more components does not receive a larger score merely
by having more observations.

| model | train features | val features | covariance | condition number | train source-median (median/IQR/p10–p90) | val source-median (median/IQR/p10–p90) |
|---|---:|---:|---|---:|---|---|
| M1 | 3,200 | 778 | full | 177.8 | 0.121 / 0.416 / 0.011–0.773 | 0.030 / 0.152 / 0.005–0.211 |
| M2 | 3,061 | 745 | full | 121.2 | 0.137 / 0.378 / 0.011–0.707 | 0.025 / 0.156 / 0.005–0.222 |
| M3 | 3,061 | 745 | full | 414,824.7 | 0.310 / 0.904 / 0.024–1.535 | 0.071 / 0.344 / 0.014–0.556 |

At the window p95 aggregation, train/validation score summaries are retained
in `metrics/normality_metrics.json`. The validation population does not show
catastrophic score divergence in this pilot, but this is not a fake-separation
result and does not validate H3 by itself.

## 7. Paired fake test

The frozen Pair2 lineage was checked against its existing bounded materialization
plan. The mandated real/fake minimum touches the complete Vript RAR and all four
HD-VG multi-volume 7z parts, beyond the 20 GiB pilot cap. No fake video entered
fitting and no fake detection metric was fabricated. The run records
`PAIRED_TEST_MEDIA_BLOCKED` in `paired_test/status.json`; real-only training is
unaffected. Therefore AUROC, AUPRC, cross-generator results, and paired
fake-minus-real differences are not reported.

There is no reliable spatial fake ground truth in this pilot: `SPATIAL_LOCALIZATION_GT_NOT_AVAILABLE`.

## 8. Conclusion

The completed outcome is:

```text
REAL_ONLY_NORMALITY_TRAINED_PAIRED_TEST_BLOCKED
```

This run establishes a reproducible real-only Gaussian baseline and held-out
real statistics, but it cannot classify H3 as supported or unsupported because
the bounded paired fake evaluation was unavailable. The current component and
`S_t` representation remain baseline research code, not a validated final
method.

## 9. Artifacts and limitations

Derived artifacts are under
`/root/autodl-tmp/data/sparse_3d_forgery_detection/derived/v7_real_only_normality_pilot_v1/`:

- `manifests/population_split.json`, `manifests/window_manifest.json`;
- `frontend/window_results.json`, `particles/`;
- `models/normality_models.json`;
- `scores/train_window_scores.json`, `scores/val_window_scores.json`;
- `metrics/frontend_summary.json`, `metrics/normality_metrics.json`;
- `run_summary.json`.

This is a small frozen pilot, not formal training at scale. RGB manual review of
the full population remains limited, provider noise and depth-scale assumptions
remain, and no spatial localization GT is available.
