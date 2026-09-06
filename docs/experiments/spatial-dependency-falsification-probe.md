# Spatial-dependency falsification probe

## Scope

This bounded experiment tests whether one learned, per-frame soft spatial relation
layer adds useful information beyond the temporal-only predictor. It is not formal
training or evidence of final forgery-detection effectiveness.

- Source HEAD: `f24ca1566b010fecd39ebda8714c50e5121563e6`
- Reused manifest: `/root/autodl-tmp/data/sparse_3d_forgery_detection/derived/temporal_learnability_probe_v1/pilot_manifest.json`
- Reused windows: 126 usable real train, 32 real validation, 32 frozen fake probe
- No causal particle artifact was regenerated or copied.

## Fixed protocol

The probe retained history count 8, horizons `{1,2,4,8}`, 128 track slots,
history-only centroid and RMS-radius normalization, target construction,
geometry-valid masking, and arithmetic-mean `score_mean` from the preceding
probe. T1 and S used real data only for training and checkpoint selection:
AdamW, learning rate `1e-3`, weight decay `1e-4`, batch size 8, at most 30
epochs, patience 5, and seed `20260906`. The saved T0 checkpoint was loaded
without retraining. Fake artifacts were read only after the T1 and S
checkpoints were frozen.

The experimental candidate topology was dense, directed, valid-only, and had
no self edges (`probe_dense_no_self`). Each relation input was exactly
`X_j - X_i`. A `3 -> 32 -> 32` GELU relation MLP produced a scalar softmax
attention logit and relation message. A matching self MLP and LayerNorm formed
the 32-dimensional spatial state. T1 omitted all pairs; S added the relation
message. Both used a single `GRUCell(32,64)` and direct 12-value output. T0 was
the prior `GRUCell(3,64)` model.

Trainable parameter counts were T0 14,028, T1 20,844, and S 22,061. T1 selected
epoch 10 (real-validation masked MSE `0.008991`); S selected epoch 4
(`0.009089`).

## Real-side results

Overall metrics are shown as valid targets / masked MSE / L2 mean / median /
RMSE / p90.

| Split | Predictor | Overall metrics |
| --- | --- | --- |
| train | Zero | 53,844 / 0.010732 / 0.083734 / 0.037423 / 0.179434 / 0.187801 |
| train | Linear | 53,844 / 0.041299 / 0.123166 / 0.051882 / 0.351990 / 0.263540 |
| train | T0 | 53,844 / 0.010457 / 0.087090 / 0.043735 / 0.177120 / 0.185270 |
| train | T1 | 53,844 / 0.010521 / 0.088970 / 0.045721 / 0.177656 / 0.185335 |
| train | S | 53,844 / 0.010675 / 0.090467 / 0.047471 / 0.178952 / 0.186171 |
| validation | Zero | 14,673 / 0.009170 / 0.068129 / 0.033892 / 0.165859 / 0.129676 |
| validation | Linear | 14,673 / 0.072692 / 0.121191 / 0.043391 / 0.466986 / 0.207720 |
| validation | T0 | 14,673 / 0.008922 / 0.071962 / 0.039368 / 0.163606 / 0.130544 |
| validation | T1 | 14,673 / 0.008991 / 0.074175 / 0.041928 / 0.164233 / 0.130187 |
| validation | S | 14,673 / 0.009089 / 0.075789 / 0.043521 / 0.165123 / 0.131515 |

Validation L2 RMSE by horizon `{1,2,4,8}`:

| Predictor | h=1 | h=2 | h=4 | h=8 |
| --- | ---: | ---: | ---: | ---: |
| Zero | 0.112753 | 0.128323 | 0.184492 | 0.219499 |
| Linear | 0.144653 | 0.230858 | 0.421117 | 0.802501 |
| T0 | 0.112083 | 0.127593 | 0.182072 | 0.215337 |
| T1 | 0.113771 | 0.128751 | 0.182384 | 0.215376 |
| S | 0.114515 | 0.128902 | 0.184196 | 0.216096 |

S relative validation-RMSE improvement was `-0.927%` versus T0 and `-0.542%`
versus T1; its validation/train RMSE ratio was `0.9227`. For paired validation
windows, S minus T0 had mean difference `0.003921`, median `0.003745`, only
`21.875%` of windows better, and 95% bootstrap CI `[0.002499, 0.005386]`.
S minus T1 had mean `0.001823`, median `0.001603`, `25.0%` better, and CI
`[0.000113, 0.003564]`. Both used 10,000 deterministic paired resamples.

## Frozen fake probe

| Predictor | AUROC | Median fake/real | Fake above real-val p95 |
| --- | ---: | ---: | ---: |
| Zero | 0.4902 | 1.0465 | 0.15625 |
| Linear | 0.4619 | 0.9182 | 0.03125 |
| T0 | 0.4824 | 1.0566 | 0.15625 |
| T1 | 0.4824 | 1.0520 | 0.15625 |
| S | 0.4775 | 0.9989 | 0.15625 |

With 10,000 deterministic resamples, AUROC deltas for S were `-0.00488`
versus T0 (95% CI `[-0.02832, 0.01563]`), `-0.00488` versus T1
(`[-0.02930, 0.01758]`), and `-0.01270` versus Zero
(`[-0.05078, 0.02051]`).

## Pre-registered decision and limitations

Decision: **FAIL**. S stayed within 2% of T0 on validation RMSE, but failed the
other pre-registered PASS conditions: it beat T0 in fewer than 55% of windows,
had AUROC below 0.60, did not exceed T0/Zero AUROC by 0.05, and its median
fake/real ratio was not above one. It also met the pre-registered FAIL pattern:
S was worse than T0 and T1 on real validation, AUROC was at most 0.50, and no
fake/real upward trend appeared.

None of Cases A, B, or C applies: T1 did not improve on T0, S did not improve on
T1, and S did not improve real prediction. In this bounded pilot, the tested
dense learned dependency did not support the V6 hypothesis. This result neither
freezes dense topology nor disproves every possible spatial encoder. The small
fixed sample, one seed, one topology, provider-derived geometry, and clip-level
fake evaluation limit generalization; no temporal localization or official test
benchmark was performed.
