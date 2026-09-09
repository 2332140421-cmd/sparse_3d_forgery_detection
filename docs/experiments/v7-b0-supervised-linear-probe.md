# V7 Frozen B0 supervised linear probe

Primary status: **B0_LINEAR_DISCRIMINATION_NOT_ESTABLISHED**

This is a fixed, finite development probe. It is not a formal training-route
switch, full-video evaluation, sealed test, probability calibration, or claim
of cross-generator generalization.

## 1. Research question

Can the frozen single-video B0 vector

```text
S = [mean, std, p25, p75]
```

support cross-source linear real/fake discrimination once the authorized
MANIP-window labels are supplied, and does it improve on the preceding fixed
real-only center-distance score?

## 2. Frozen input and label protocol

The exact 16 source pairs and 192 annotation-selected windows from the paired
B0 artifact were reused. The component distance normalization, validity rule,
and window-level coordinate-wise median aggregation were not changed. The
persisted B0 window medians reproduced the historical values for 192/192
windows with maximum absolute error `0.0`.

For the main probe only:

```text
real MANIP window -> label 0
fake MANIP window -> label 1
```

The label describes a preselected manipulation location; it does not mean the
real video was modified. CTRL windows are excluded from fitting and are only a
separate descriptive evaluation. Window selection uses existing annotations;
the numerical score for an individual video does not use a paired real.

There are 174 valid B0 windows from 15 valid sources. `0HV07` remains in the
coverage accounting with no finite B0 score. No source, window, anchor, or
external condition was reselected, and no frontend was rerun.

## 3. Protocol fixed before held-out scoring

The protocol is saved at
`/root/autodl-tmp/data/sparse_3d_forgery_detection/derived/v7_activityforensics_b0_supervised_linear_probe_v1/protocol.json`
before OOF held-out scores were written. The only model is a binary logistic
regression with L2 penalty, `C=1.0`, `solver=lbfgs`, intercept, `tol=1e-6`,
`max_iter=1000`, and no class-weight option. The reported score is
`decision_function`; it is not a calibrated probability.

For each fold, every training source has equal total weight, its real and fake
classes each receive half of that weight, and observations within a
source/class are equal. Weights are rescaled to mean one. Weighted mean and
variance standardization is fit on that fold's training real/fake MANIP
windows only; variance at or below `1e-12` uses fixed scale `1.0` and is
recorded. No held-out source contributes to weights, standardization, or model
fit.

## 4. Source-disjoint LOSO

All 16 source folds are retained. Fifteen valid held-out sources are scored by
their own model trained on the other 14 sources. The `0HV07` fold is recorded
as `NO_HELDOUT_B0`, with no imputed score. Training counts are recorded in
`calibration/per_fold_support.csv`; across valid folds they are 41 or 42 real
and 42 fake MANIP windows. All fifteen fitted folds converged in 10--14
iterations, without the permitted retry.

No paired real feature, source identity, generator, filename, operation,
quality feature, NSI, dynamic feature, or control label enters the numeric
input. Fake labels are used only in the explicitly authorized training probe.

## 5. Primary comparison

The primary evaluation is source-equal mean AUROC for fake MANIP versus real
MANIP. It is compared on exactly the same 15-source valid windows as the
preceding B0 real-only scorer. Source-level paired bootstrap resamples complete
sources with seed `20260909` for 10,000 replicates and uses identical indices
for both scores.

| score | source mean AUROC | bootstrap 95% CI |
|---|---:|---:|
| supervised B0 logistic probe | 0.455556 | [0.300000, 0.614815] |
| frozen B0 real-only score | 0.496296 | [0.318519, 0.666667] |
| supervised minus real-only | -0.040741 | [-0.277778, 0.192593] |

The pooled AUROC is only exploratory because fold decision-function scales are
not shared:

- supervised fake MANIP vs real MANIP: `0.451515`;
- real-only baseline on the same windows: `0.442929`;
- supervised fake MANIP vs all eligible held-out real: `0.417571`;
- baseline fake MANIP vs all eligible held-out real: `0.410336`.

## 6. Per-source stability

| source | real N | fake N | supervised AUROC | baseline AUROC | delta |
|---|---:|---:|---:|---:|---:|
| 01KML | 2 | 3 | 0.167 | 1.000 | -0.833 |
| 04LAX | 3 | 3 | 0.000 | 0.778 | -0.778 |
| 0AGCS | 3 | 3 | 0.889 | 0.222 | +0.667 |
| 0BX9N | 3 | 3 | 0.333 | 0.222 | +0.111 |
| 0CG15 | 3 | 3 | 1.000 | 0.667 | +0.333 |
| 0CGMQ | 3 | 3 | 0.667 | 0.778 | -0.111 |
| 0DVVD | 3 | 3 | 0.000 | 0.778 | -0.778 |
| 0ET8W | 3 | 3 | 0.667 | 1.000 | -0.333 |
| 0FM93 | 3 | 3 | 0.333 | 0.000 | +0.333 |
| 0FO58 | 3 | 3 | 0.111 | 0.000 | +0.111 |
| 0G2SC | 3 | 3 | 0.444 | 0.222 | +0.222 |
| 0KTWY | 3 | 3 | 0.778 | 0.333 | +0.444 |
| 0LNLR | 3 | 3 | 0.222 | 0.333 | -0.111 |
| 0PU21 | 3 | 3 | 0.556 | 0.889 | -0.333 |
| 0QA8P | 3 | 3 | 0.667 | 0.222 | +0.444 |

The supervised score improves over baseline on 7/15 sources and is worse on
8/15. The small per-source sample sizes make these values descriptive only.
The paired bootstrap gain interval crosses zero, so the apparent pooled
improvement is not stable source-level evidence.

## 7. Control behavior and Delta definitions

CTRL windows were not training examples and are not assumed to be definitely
modified or definitely clean. On the supervised model, source-level control
score medians have:

- real CTRL median `0.061350`, IQR `0.367802`;
- fake CTRL median `0.063398`, IQR `0.206299`.

The previous real-only score has different raw scale (real CTRL median
`1.185991`, fake CTRL median `0.969658`), so score magnitudes are not compared.

For descriptive source-level evaluation only, the existing definitions remain:

```text
Delta_manip  = A_FM - A_RM
Delta_control = A_FC - A_RC
J_B0 = Delta_manip - Delta_control
```

For the supervised probe, respectively, the medians are `+0.012197`,
`+0.028872`, and `+0.004606`; their positive fractions are `0.5333`, `0.6667`,
and `0.5333`, with bootstrap intervals crossing zero. These are evaluation
contrasts, not the inference score.

## 8. Coverage and limitations

The development population is annotation-selected and small: 16 source pairs,
15 scoreable sources, 174 valid windows, and 85 valid CTRL windows excluded
from training. `0HV07` is not filled or removed. This is not full-video, not a
sealed test, and not a probability-calibrated detector. It does not test
cross-generator generalization beyond this frozen development population.

## 9. Scientific conclusion and status

The frozen B0 four-dimensional representation does not show established
cross-source linear discrimination in this probe. The source-equal mean AUROC
is below 0.5 and its bootstrap interval crosses 0.5. The supervised model is
not stably better than the preceding real-only score; the paired gain interval
crosses zero. The result is therefore:

```text
B0_LINEAR_DISCRIMINATION_NOT_ESTABLISHED
```

This does not prove that B0 contains no information, that nonlinear models are
impossible, or that every 3D representation fails. It only falsifies the
stronger claim that this frozen B0 plus one fixed linear classifier already
provides a stable cross-source development detector.

## 10. Boundaries and next stage

No hyperparameter search, classifier comparison, neural network, feature
interaction, feature selection, NSI, dynamic fusion, multi-order study,
frontend rerun, data expansion, full-video evaluation, sealed test, or
probability calibration was performed. Formal `src/sparse3d_forgery/` was not
modified. This probe does not automatically authorize any subsequent method;
next steps require explicit discussion and authorization.

Artifacts are under
`/root/autodl-tmp/data/sparse_3d_forgery_detection/derived/v7_activityforensics_b0_supervised_linear_probe_v1/`:
protocol, frozen manifests, per-fold model parameters, OOF window scores,
per-fold support, per-source metrics, and summary JSON. Videos, NPZ files, and
large frontend caches were not copied.
