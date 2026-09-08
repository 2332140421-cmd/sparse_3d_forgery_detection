# V7 ActivityForensics paired second-order structural signal pilot

Status: **PAIRED_SECOND_ORDER_SIGNAL_NOT_SUPPORTED_IN_CURRENT_BASELINE**

This was a bounded falsification pilot, not formal training. The artifact root is
`/root/autodl-tmp/data/sparse_3d_forgery_detection/derived/v7_activityforensics_paired_second_order_pilot_v1/`.

## A. Git and execution boundary

The run used branch `v7-dynamic-structure`, starting at
`9d89deb484d8934668e18b27436980be43d62c02`, which was one commit ahead of
`origin/v7-dynamic-structure`. The formal `src/sparse3d_forgery/` package was
not changed. No old R7/V5 repository was accessed.

## B. Frozen population

The first 16 technically eligible pairs in frozen review-manifest order were
selected from 30 `EXACT` review pairs; exclusions: 0. Selection used only
lineage, media validity, timeline safety, official segment length, and an
unmodified control block. The selected pair IDs are `0001`--`0016`. Generator
counts are `wan=7`, `fcvg=5`, `vidu=4`; operation values are `add=8`,
`delete=6`, and two rows contain the official `['add', 'delete']` value.

## C. Window population

Each pair has three one-second true-PTS manipulation anchors and three matched
unmodified controls, for 192 role windows (16 pairs × 6 windows × real/fake).
All 192 completed; no window was filtered after frontend execution. Source frame
indices and timestamps are retained in `manifests/window_manifest.json` and
`frontend/window_results.json`.

## D. Frontend feasibility

The unchanged causal Online BootsTAPIR + Apple Depth Pro + adjacent Open3D RGB-D
odometry baseline passed the mandated first-four-pair gate and completed all 16
pairs. Quality medians by role/kind were:

| role/kind | geometry coverage | tracking persistence | pose success | valid Delta2 observations |
|---|---:|---:|---:|---:|
| real/manipulation | 0.958 | 0.875 | 1.000 | 28 |
| fake/manipulation | 0.925 | 0.766 | 1.000 | 28 |
| real/control | 0.950 | 0.867 | 1.000 | 48 |
| fake/control | 0.944 | 0.859 | 1.000 | 56 |

No preview media was read; every input was asserted to be an original path and
its SHA-256 was recorded. Particle artifacts are NPZ + JSON on the data disk.

## E. Matched structural signal

`D_manip` and `D_ctrl` are normalized fake--real discrepancies matched by pair
and anchor. Scales use real control windows only (`fake_count_in_fitting=0`).
`G=D_manip-D_ctrl`; N is the number of pairs with both medians valid.

| order | D_manip median | D_ctrl median | G median [95% bootstrap CI] | positive fraction | exploratory AUROC |
|---|---:|---:|---:|---:|---:|
| K0 `S_t` | 2.294 | 1.197 | 1.117 [0.775, 2.233] | 0.929 | 0.748 |
| K1 `Delta S_t` | 3.027 | 3.674 | -0.062 [-1.358, 0.839] | 0.500 | 0.414 |
| K2 `Delta2 S_t` | 2.385 | 2.800 | -0.311 [-1.093, 1.242] | 0.429 | 0.467 |

K0 has a positive paired discrepancy, but K2 does not: its two-sided sign-test
p-value is 0.791 and its bootstrap interval crosses zero. AUROC is exploratory
and is not a detector result.

## F. Second-order comparison

`A21=G2-G1` has median 0.166, 95% CI `[-0.850, 1.816]`, positive fraction
0.571. `A20=G2-G0` has median -2.471, 95% CI `[-3.083, 0.450]`, positive
fraction 0.357. Neither supports a second-order advantage in this baseline.

## G. Raw activity and secondary diagnostics

The descriptive norm medians (real/fake, manipulation/control) were:

- `K0`: real 1.924/1.944; fake 1.922/1.920.
- `K1`: real 0.053/0.050; fake 0.031/0.054.
- `K2`: real 2.350/2.489; fake 1.258/2.445.

Pairwise normalized-distance temporal diagnostics covered 174 windows; temporal
MAD median 0.369 and temporal-IQR median 0.841. They are secondary and were not
added to the core `S_t` representation.

## H. Quality and generator checks

Spearman correlations between pair `G2` and manipulation fake-minus-real
quality gaps were: geometry coverage `0.279`, tracking persistence `0.181`,
valid Delta2 count `0.352`; pose success was constant and therefore undefined.
These are controls, not causal corrections. K2 `G` medians by generator/operation
were `fcvg/delete=-0.437` (N=4), `vidu/add=-1.521` (N=2),
`wan/add=1.024` (N=3), `wan/delete=-0.677` (N=3), and mixed-operation wan=2.030
(N=2), showing heterogeneity.

## I. Primary conclusion

In this 16-pair, 192-window pilot, the frozen K2 second-order structural
discrepancy is not supported as a robust paired signal beyond controls. A K0
window discrepancy is present descriptively, but it does not establish the V7
second-order hypothesis. The result is therefore a falsification of the current
baseline for this question, not evidence that all representations or generators
are impossible.

Artifacts include `manifests/`, `frontend/`, `particles/`,
`metrics/per_window_signal.csv`, `metrics/per_pair_signal.csv`,
`metrics/structural_order_summary.json`, `metrics/frontend_quality.json`,
`metrics/pairwise_diagnostic.json`, and `run_summary.json`.

## J. Boundaries

No Gaussian normality model or detector was fitted; fake data entered no fitting
step. No preview, old V6/V5/R7 code, labels, generator identity, AUROC-driven
selection, velocity/acceleration/jerk/residual, or Part hierarchy was introduced.
No formal training, depth/tracking/pose redesign, GPU experiment beyond the
short frontend pilot, or automatic next-stage run was started.
