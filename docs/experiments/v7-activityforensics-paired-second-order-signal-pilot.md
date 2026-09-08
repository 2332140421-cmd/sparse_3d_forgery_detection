# V7 ActivityForensics paired second-order structural signal pilot

Status: **REPRESENTATION_REVISION_NEEDED_BEFORE_FORMAL_TRAINING**

This was a bounded falsification pilot, not formal training. The artifact root is
`/root/autodl-tmp/data/sparse_3d_forgery_detection/derived/v7_activityforensics_paired_second_order_pilot_v1/`.

## Research question and hypothesis

The question was whether matched manipulation windows show a structural
second-order evolution discrepancy beyond matched unmodified controls. The
frozen hypothesis was that `Delta2 S_t` would provide a manipulation-specific
paired gain without assuming that fake magnitude is necessarily higher or lower.
The user-confirmed core domain is a continuous, stable-camera, dynamically
structured ActivityForensics/Charades subset; this is not a claim about the whole
dataset.

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
counts are `wan=9`, `fcvg=4`, `vidu=3`; operation values are `add=7`,
`delete=7`, and two rows contain the official `['add', 'delete']` value.

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

| role/kind | geometry coverage | tracking persistence | component success | pose success | valid Delta2 observations |
|---|---:|---:|---:|---:|---:|
| real/manipulation | 0.958 | 0.875 | 1.000 | 1.000 | 28 |
| fake/manipulation | 0.925 | 0.766 | 1.000 | 1.000 | 28 |
| real/control | 0.950 | 0.867 | 1.000 | 1.000 | 48 |
| fake/control | 0.944 | 0.859 | 1.000 | 1.000 | 56 |

No preview media was read; every input was asserted to be an original path and
its SHA-256 was recorded. Total frontend wall time was 1766.6 s (median 10.16 s
per window, maximum 15.62 s); peak allocated GPU memory was 4,016 MiB on the
RTX 4090 D. Particle artifacts are NPZ + JSON on the data disk.

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
p-value is 0.791 and its bootstrap interval crosses zero. Full pair-level
descriptives are `G0: N=14, IQR=1.393, p10--p90=[0.339,2.860]`,
`G1: N=14, IQR=1.981, p10--p90=[-2.324,2.005]`, and
`G2: N=14, IQR=2.264, p10--p90=[-1.823,2.488]`. AUROC is exploratory and
is not a detector result.

## F. Second-order comparison

`A21=G2-G1` has N=14, median 0.166, IQR 2.538, p10--p90
`[-1.986,3.622]`, 95% CI `[-0.850,1.816]`, positive fraction 0.571.
`A20=G2-G0` has N=14, median -2.471, IQR 3.504, p10--p90
`[-3.557,1.727]`, 95% CI `[-3.083,0.450]`, positive fraction 0.357.
Neither supports a second-order advantage in this baseline.

## G. Raw activity and secondary diagnostics

The descriptive norm medians (real/fake, manipulation/control) were:

- `K0`: real 1.924/1.944; fake 1.922/1.920.
- `K1`: real 0.053/0.050; fake 0.031/0.054.
- `K2`: real 2.350/2.489; fake 1.258/2.445.

Pairwise normalized-distance temporal diagnostics covered 174 windows; temporal
MAD median 0.369 and temporal-IQR median 0.841. In the matched pair diagnostic,
raw temporal-MAD `G` was 0.152 (95% CI [0.104,0.204], positive fraction 0.857),
temporal-IQR `G` was 0.211 (CI [0.091,0.531], positive fraction 0.929), and
raw second-difference magnitude `G` was 11.776 (CI [6.840,16.002], positive
fraction 0.857). These are secondary representation evidence, not detector
features; they trigger the predeclared representation-revision gate.

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
discrepancy is not supported as a robust paired signal beyond controls, and
A21/A20 do not support a second-order advantage. However, the independent raw
pairwise temporal diagnostics show stable manipulation-specific gaps. Under the
predeclared gate, the primary conclusion is
`REPRESENTATION_REVISION_NEEDED_BEFORE_FORMAL_TRAINING`: revise or expand the
structural representation before fitting a real-only normality model. This is
not evidence that all representations or generators are impossible.

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
