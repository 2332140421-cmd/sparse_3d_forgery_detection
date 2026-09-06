# Per-horizon causal target frontend audit

## Protocol

- Source HEAD: `8c21aa6f737e815fdf523555f77d0bc05b6e8998`.
- DeeptraceReward revision: `92e76e78e8c90a1ff7ec9354bee44eb024265e79`.
- Reused population: 160 real validation windows (primary) and 160 fixed fake
  probe windows (supplementary), five deterministic windows from each of 32
  videos per role.
- The existing eight-frame Pass-A history was reused exactly. It was not rerun.
- B1, B2, and B4 used exact prefixes through their respective target frames;
  960 new VGGT inferences were run. Existing B8/full targets were reused for
  all 320 windows and were not inferred again.
- Each prefix reconstruction was mapped into the fixed Pass-A gauge using the
  accepted global proper history-only Sim(3). Target points never entered the
  fit. Prefix/full disagreement used jointly valid same-track target points;
  motion ratios additionally required a valid cutoff observation.
- T0 was loaded frozen, without an optimizer or parameter updates. Fake results
  did not affect thresholds, configuration, or the real-only classification.

Derived numeric results are under
`derived/prefix_target_frontend_audit_v1/` in the project data tree and are not
tracked by Git.

## Real primary results

| h | windows | n_h median / mean / p90 | median reduction vs n8 | target disagreement mean / median / RMSE / p90 | motion mean / median / RMS / p90 |
|---:|---:|---:|---:|---:|---:|
| 1 | 160 | 0.01613 / 0.02151 / 0.04080 | 72.58% | 0.01660 / 0.00918 / 0.03325 / 0.03660 | 0.00875 / 0.00342 / 0.02670 / 0.01770 |
| 2 | 159 | 0.02509 / 0.03605 / 0.06958 | 57.67% | 0.01460 / 0.00798 / 0.03020 / 0.03157 | 0.01315 / 0.00575 / 0.03767 / 0.02649 |
| 4 | 155 | 0.04045 / 0.05503 / 0.11608 | 30.74% | 0.01022 / 0.00547 / 0.02153 / 0.02270 | 0.02001 / 0.00883 / 0.05702 / 0.03895 |
| 8 | 156 | 0.05884 / 0.07564 / 0.15340 | 0% | 0 / 0 / 0 / 0 | 0.02850 / 0.01243 / 0.07614 / 0.05606 |

The disagreement and motion columns are in the frontend coordinate unit. Full
normalized distributions, common counts, Sim(3) scale, `det(R)`, and raw/aligned
history residual diagnostics are retained in the JSON artifacts. The accepted
alignment implementation did not retain raw-history RMSE, so that requested
quantity is explicitly marked unavailable rather than reconstructed or guessed.

| h | Q_target median / mean / p75 / p90 | fractions <.25 / <.5 / <1 / >=1 | Q_history median / mean / p75 / p90 | fractions <.25 / <.5 / <1 / >=1 |
|---:|---:|---:|---:|---:|
| 1 | 1.6056 / 2.1623 / 2.5934 / 3.9135 | .0188 / .0875 / .2375 / .7625 | .4926 / .5348 / .6979 / .8592 | .1500 / .5062 / .9500 / .0500 |
| 2 | .9326 / 1.1552 / 1.3863 / 2.0894 | .0377 / .1824 / .5283 / .4717 | .5405 / .5848 / .7654 / .9214 | .1258 / .4465 / .9371 / .0629 |
| 4 | .4278 / .5132 / .6152 / .8719 | .2323 / .5742 / .9355 / .0645 | .5872 / .5980 / .7997 / .9392 | .1226 / .4323 / .9290 / .0710 |
| 8 | 0 / 0 / 0 / 0 | 1 / 1 / 1 / 0 | .5891 / .6173 / .8202 / .9651 | .0962 / .3782 / .9167 / .0833 |

T0 full/prefix Pearson error-residual correlations were 0.8486/0.6967,
0.8474/0.7833, 0.8563/0.7355, and 0.7883/0.7883 for h=1/2/4/8. Zero produced
0.8515/0.7543, 0.8476/0.8031, 0.8687/0.7593, and 0.8071/0.8071.

## Repeatability and supplementary result

Eight preselected B1 center windows (four real and four fake) produced exactly
zero repeat-run target disagreement on 116--128 common particles. Their
B1-versus-B8 context disagreement RMSE ranged from 0.01112 to 0.11861. The
repeat/context ratio was therefore zero and there is no
`RUNTIME_REPEATABILITY_BLOCKER`.

Fake supplementary results showed the same directional reduction in median
alignment residual (78.72%, 61.58%, and 31.88% for h=1/2/4), while median
Q_target remained 1.8187, 1.1530, and 0.5281. T0 full/prefix correlations were
0.9537/0.7637, 0.9462/0.8166, and 0.9310/0.8876. These values are descriptive;
they were not used in classification and no detection AUROC was computed.

## Classification and limits

The pre-registered real-only classification is **MIXED_INCONCLUSIVE**. All three
short horizons met the alignment-reduction condition, but only h=4 had median
Q_target below 0.5, and only h=1 reduced the absolute T0 correlation by at least
0.15. No pre-registered positive or negative rule is fully met.

Accordingly, whether VGGT should continue as the V6 frontend is **INCONCLUSIVE**.
Prefix target construction has not become the formal V6 target construction.
This audit did not train or tune a model, did not prove 3D accuracy, and did not
prove final forgery-detector effectiveness. Alignment residual is a frontend
diagnostic, not anomaly evidence.
