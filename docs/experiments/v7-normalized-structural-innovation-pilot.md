# V7 normalized structural innovation pilot

Primary status: **NSI_SENSITIVE_BUT_NOT_NORMALIZED**

This is a bounded mechanism pilot. It is not formal normality training, a
blind detector evaluation, or a sealed test result.

## 1. Research question

With the V7 ActivityForensics/Charades population and frontend frozen, can a
single dimensionless normalized structural innovation preserve a
manipulation-specific second-order relation signal while providing a
cross-source real-only ruler?

The experiment tests only the scalar mechanism

```text
I_k = ||v_plus - v_minus|| /
      (||v_plus|| + ||v_minus|| + 1e-8)
```

No magnitude, coherence, persistence, semantic, B0-fusion, or stacked feature
was added.

## 2. Frozen data and frontend

The exact previous artifacts were reused: 16 source pairs, 192 annotated role
windows, identical timestamps, particles, track identities, validity masks, and
dynamic components. No population, anchor, manipulation/control window, or
component was reselected. No frontend was rerun and no GPU was used.

The output is under
`/root/autodl-tmp/data/sparse_3d_forgery_detection/derived/v7_activityforensics_normalized_structural_innovation_pilot_v1/`.
Only CSV/JSON summaries were written; particle NPZ, video, and frontend caches
were not copied.

## 3. Structural relation space

For every frozen component, normalized pair relations remain

```text
r_p(t) = ||X_i(t) - X_j(t)|| / (s_c(t) + epsilon)
```

where `s_c(t)` is the median valid component pair distance. For each local
triplet, the same persistent pair identities are sorted by `(track_i, track_j)`
and must be valid at all three timestamps. Missing relations are not
interpolated and no derivative crosses a gap.

The PTS-aware vectors are

```text
v_minus = (r0 - r_minus) / h0
v_plus  = (r_plus - r0) / h1
```

The component score is `Q90({I_k})`; window score is the median across valid
components. Component size and pair count do not receive extra weight.

## 4. Numerical sanity

The extraction produced 9,101 valid triplets and zero range violations. Across
triplets:

| statistic | value |
|---|---:|
| minimum I | 0.011153 |
| median I | 0.845514 |
| p90 I | 0.991707 |
| maximum I | 1.000000 |

The unit tests verify constant structural velocity gives `I≈0`, velocity
reversal gives `I≈1`, common positive velocity scaling leaves `I` unchanged,
duplicating relation coordinates leaves `I` unchanged, and `0≤I≤1` within
numerical tolerance.

## 5. Paired mechanism evaluation

For each source, `D_manip` and `D_ctrl` are the median absolute fake--real
window differences at the three fixed anchors. `G_I = D_manip - D_ctrl`.
The statistical unit is the source pair; bootstrap uses seed `20260909` and
10,000 source-level resamples.

| quantity | N | median | IQR | p10--p90 | positive fraction | 95% CI |
|---|---:|---:|---:|---:|---:|---:|
| D_manip | 15 | 0.029279 | 0.023275 | [0.022127, 0.064060] | — | — |
| D_ctrl | 14 | 0.017488 | 0.005913 | [0.009465, 0.029905] | — | — |
| G_I | 14 | 0.015500 | 0.023321 | [-0.012938, 0.043403] | 0.785714 | [0.006466, 0.030193] |

Exploratory paired `D_manip` versus `D_ctrl` AUROC is `0.838095`. This is not
video-level detector AUROC.

The paired gate is therefore positive for this pilot, but it remains a
paired-real sensitivity result.

## 6. Raw NSI distributions

These are window-level `I_window` values before any paired discrepancy:

| group | N | median | IQR | p10--p90 |
|---|---:|---:|---:|---:|
| real/manip | 44 | 0.944007 | 0.047513 | [0.876539, 0.988206] |
| fake/manip | 45 | 0.973114 | 0.024072 | [0.949127, 0.993814] |
| real/control | 42 | 0.960716 | 0.047778 | [0.908152, 0.987929] |
| fake/control | 43 | 0.957234 | 0.047187 | [0.899658, 0.989775] |

Missing windows were retained, not imputed or filtered. The previous frontend
artifact has no component rows for source `0HV07`, and a few additional
windows from `01KML`/`0LNLR` have no valid NSI; this accounts for the reduced
valid N in the summaries.

## 7. Cross-source real-only calibration probe

This is explicitly a `CROSS_SOURCE_RULER_PROBE`, not a blind detector result.
For each held-out source, calibration used only real windows from the other
15 sources. Fake calibration count was exactly zero. Calibration used median
and `1.4826*MAD`, with the predeclared IQR/floor fallback only if needed.

Valid source-level summaries contain N=15 because one frozen source has no
component-derived NSI. No missing source was silently filled.

| group | N | median Z | IQR | p10--p90 |
|---|---:|---:|---:|---:|
| real/manip | 15 | 0.692372 | 0.667360 | [0.252365, 1.261351] |
| real/control | 15 | 0.680117 | 0.341776 | [0.438468, 1.001893] |
| fake/manip | 15 | 0.567508 | 0.175804 | [0.177478, 0.957651] |
| fake/control | 15 | 0.789599 | 0.496924 | [0.183460, 1.298922] |

The evaluation statistic

```text
H_I = (Z_fake_manip - Z_real_manip)
      - (Z_fake_ctrl - Z_real_ctrl)
```

has:

| N | median | IQR | p10--p90 | positive fraction | 95% CI |
|---:|---:|---:|---:|---:|---:|
| 15 | -0.008534 | 0.716789 | [-0.715385, 0.535377] | 0.466667 | [-0.592185, 0.124886] |

Exploratory AUROC of held-out fake/manipulation windows versus held-out real
windows is `0.405168`. Real manipulation minus control Z has median `-0.185758`
and positive fraction `0.333333`; it is not a systematic real false-alarm rise.
Fake manipulation minus control Z has median `-0.190731` and positive fraction
`0.400000`.

The cross-source ruler gate is not met.

## 8. Activity normalization

Window-level Spearman correlations are:

| score | speed sum | raw `||Δr||` | raw second difference |
|---|---:|---:|---:|
| I | -0.363394 | -0.399821 | -0.345995 |
| Z | -0.117790 | -0.110555 | -0.130342 |

On real windows only, I correlations are `-0.356724`, `-0.389556`, and
`-0.347875`; Z correlations are `-0.080938`, `-0.086579`, and `-0.088561`.
Source-level correlations are also below the `|rho|=0.7` diagnostic threshold.
This pilot does not trigger `ACTIVITY_NORMALIZATION_FAILED`.

## 9. Pair-dimension and quality confounding

The common-pair dimension diagnostics were:

| diagnostic | Spearman rho |
|---|---:|
| window I vs common-pair dimension | -0.624511 |
| window Z vs common-pair dimension | -0.101292 |
| source G_I vs manipulation dimension gap | -0.345055 |
| source H_I vs manipulation dimension gap | 0.178571 |

No value reaches `|rho|>=0.7`; `PAIR_DIMENSION_CONFOUND` is not triggered.

## 10. Comparison with frozen results

Only direction and positive-fraction comparisons are made across the different
scales:

| representation | G median | positive fraction |
|---|---:|---:|
| B0 structural state | 1.117231 | 0.928571 |
| P2 pool-first second order | -0.310721 | 0.428571 |
| R2 fixed-dimensional relation-first | 0.396984 | 0.571429 |
| raw pairwise second difference | 11.775985 | 0.857143 |
| NSI paired gain | 0.015500 | 0.785714 |

The raw pairwise diagnostic remains a different, non-fixed-dimensional
quantity. Magnitudes are not mechanically comparable.

## 11. Scientific conclusion

1. NSI retains a positive paired manipulation-specific signal in this pilot.
2. That signal still depends on the frozen paired-real protocol; the LOSO ruler
   does not produce a positive manipulation-specific `H_I`.
3. The real-only cross-source scalar does not yet show comparable anomaly
   calibration potential.
4. NSI is not primarily explained by the tested activity measures, but this is
   only a mechanism diagnostic, not a general proof of normalization.
5. No strong pair-dimension confound was observed; missing component coverage
   remains an explicit limitation.
6. The mechanism is not sufficient to authorize a real-only, no-reference,
   full-video blind detection experiment.

Primary status: **NSI_SENSITIVE_BUT_NOT_NORMALIZED**.

## 12. Boundaries

No world model, action semantics, purpose model, cross-component coherence
feature, feature stacking, B0 fusion, frontend rerun, data reselection,
formal training, fake fitting, full-video blind claim, or formal `src/`
modification was performed. The run stops for scientific review.
