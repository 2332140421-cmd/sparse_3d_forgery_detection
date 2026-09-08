# V7 relation-first structural dynamics representation pilot

Primary status: **FIXED_DIMENSION_RELATION_AGGREGATION_STILL_LOSSY**

This is a bounded representation ablation. It is not formal normality training
and does not authorize the next stage automatically.

## 1. Research question

With the ActivityForensics/Charades population, windows, frontend, particles,
and dynamic components frozen, does taking temporal derivatives on persistent
pair relations before aggregation recover the raw pairwise second-order signal?

## 2. Motivation

The previous pool-first representation produced a positive B0 state discrepancy
but no stable P2 second-order gain, while the independent raw pair diagnostic
showed manipulation-specific temporal differences. The present experiment tests
that explanation without changing data or frontend choices.

## 3. Frozen population and frontend

The exact previous pilot artifacts were reused: 16 source pairs, 192 role
windows, identical real/fake timestamps, manipulation/control anchors, particle
NPZ artifacts, component membership, and quality records. No pair, segment,
control block, or timestamp was reselected. BootsTAPIR, Depth Pro, Open3D,
tracking, depth, and pose were not rerun. The relation artifact root links to
`v7_activityforensics_paired_second_order_pilot_v1` and does not copy its NPZ.

## 4. Old pool-first representations

The unchanged baselines are:

- B0: `S_t = [mean, std, p25, p75]`;
- P1: timestamp-aware `Delta S_t`;
- P2: timestamp-aware `Delta2 S_t`.

They retain the previous paired discrepancy protocol and are used as an
ablation reference, not refit with fake data.

## 5. Relation-first representation

For each frozen component and persistent particle identity pair `(i,j)`, the
same normalized relation as B0 is formed:

```text
r_ij,t = ||X_i,t - X_j,t|| / (median valid component distance_t + epsilon)
```

The order is then:

```text
Delta2 Psi({r_ij,t})       # old pool-first conceptual order
Psi({Delta2 r_ij,t})       # relation-first order tested here
```

Only pairs with at least the frozen `minimum_overlap=8` valid observations are
eligible. Missing relations are not interpolated, filled, or differentiated
across gaps. Derivatives use true PTS and the nonuniform central formula
`2 * (v_right - v_left) / (h0 + h1)`.

## 6. Pair and component descriptors

Each eligible pair independently receives the fixed four-dimensional descriptor
`[median(signed), MAD(signed), median(abs), q90(abs)]` for R1 and R2. Pair
descriptors are equally weighted within each component; each component retains
the across-pair median and IQR, giving an 8-D component vector. Components are
then equally weighted within a window. No feature was selected after inspecting
the result.

## 7. B0 structural-state baseline preservation

B0 was reproduced exactly from the frozen artifacts:

- `G_B0` median: **1.117**;
- positive fraction: **0.929**;
- exploratory paired AUROC: **0.748**.

Differences from the previous summary for these three values were all zero.
This remains paired-pilot evidence of measurable 3D structural-state change,
not a final detector result.

## 8. Pair-level results

All values use source pair as the statistical unit. Scales are fitted from real
control windows only; fake fitting count is zero.

| representation | N | G median | IQR | p10--p90 | positive fraction | 95% bootstrap CI | exploratory AUROC |
|---|---:|---:|---:|---:|---:|---:|---:|
| B0 | 14 | 1.117 | 1.393 | [0.339, 2.860] | 0.929 | [0.775, 2.233] | 0.748 |
| P1 | 14 | -0.062 | 1.981 | [-2.324, 2.005] | 0.500 | [-1.358, 0.839] | 0.414 |
| P2 | 14 | -0.311 | 2.264 | [-1.823, 2.488] | 0.429 | [-1.093, 1.242] | 0.467 |
| R1 | 14 | 0.599 | 2.889 | [-3.999, 2.415] | 0.571 | [-0.950, 1.736] | 0.571 |
| R2 | 14 | 0.397 | 2.823 | [-1.974, 2.018] | 0.571 | [-1.546, 1.325] | 0.514 |

The AUROC values are manipulation-versus-control signal diagnostics, not
standalone fake-video detector performance.

## 9. Relation-first contrasts

| contrast | median | IQR | positive fraction | 95% bootstrap CI |
|---|---:|---:|---:|---:|
| `A_R2_P2 = G_R2 - G_P2` | 0.232 | 2.791 | 0.500 | [-1.514, 1.255] |
| `A_R2_R1 = G_R2 - G_R1` | -0.579 | 2.435 | 0.429 | [-1.417, 1.159] |
| `A_R2_B0 = G_R2 - G_B0` | -1.875 | 2.998 | 0.286 | [-2.875, -0.212] |

R2 is not stable under the predeclared `0.70` positive-fraction and bootstrap
gate, does not beat P2, and does not beat R1.

## 10. Pair coverage

Window medians (persistent pairs / R1 eligible / R2 eligible; eligible fraction)
were:

| role/kind | persistent | R1 eligible | R2 eligible | R1 fraction | R2 fraction |
|---|---:|---:|---:|---:|---:|
| real/manip | 1100.5 | 1100.5 | 1100.5 | 1.000 | 1.000 |
| fake/manip | 1272.0 | 1272.0 | 1272.0 | 1.000 | 1.000 |
| real/control | 886.5 | 886.5 | 886.5 | 1.000 | 1.000 |
| fake/control | 773.5 | 773.5 | 773.5 | 1.000 | 1.000 |

No source was filtered because of coverage. The R2 manipulation fake-minus-real
pair-count gap had Spearman rho `-0.182` with `G_R2`.

## 11. Raw diagnostic alignment

Against the previous raw pairwise second-difference `G`, R2 had N=14,
Spearman rho **0.165**, and sign-direction agreement **0.714**. This is partial
alignment, not recovery of the raw signal.

## 12. Quality confounding

Spearman correlations with `G_R2` were:

- eligible-pair-count gap: `-0.182`;
- geometry coverage gap: `0.363`;
- tracking persistence gap: `0.205`.

None reached the predeclared `|rho| >= 0.7` strong-confound flag.

## 13. Scientific conclusion

1. B0 structural-state evidence remains reproduced.
2. Relation-first R2 does not recover a stable second-order signal under the
   frozen fixed-dimensional descriptor.
3. Derivative-before-aggregation is not supported over P2 in this pilot.
4. R2 is not supported as superior to R1.
5. R2 does not yet provide a demonstrated dynamic contribution distinct from B0.
6. The raw diagnostic remains positive while R2 is not, so fixed-dimensional
   relation aggregation is still lossy.

The result is **FIXED_DIMENSION_RELATION_AGGREGATION_STILL_LOSSY**, not a claim
that all relation-first representations are impossible.

## 14. Artifacts and boundaries

New artifacts are under
`/root/autodl-tmp/data/sparse_3d_forgery_detection/derived/v7_activityforensics_relation_first_pilot_v1/`:
source links, frozen manifests, per-window relation summaries, pair coverage,
per-window representations, per-pair gains, comparison, quality-confound, raw
alignment, and run summary. Large particle NPZ files were not copied.

No data reselection, frontend redesign or rerun, formal normality fitting, fake
fitting, classifier, conditional model, formal `src/` modification, or automatic
next stage was performed. The previous pilot and its artifacts were not
overwritten.

## 15. Authorization for next stage

No authorization is granted to enter real-only training. This pilot stops here
for user and research-design review.
