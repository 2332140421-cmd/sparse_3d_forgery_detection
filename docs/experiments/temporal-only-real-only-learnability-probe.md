# Temporal-only real-only learnability probe

## Protocol

- Dataset: DeeptraceReward revision `92e76e78e8c90a1ff7ec9354bee44eb024265e79`.
- Selection: lexicographically first unique `source_video_id` within each official split/type; official train real 128, official validation real 32, and—only after the real-only gate passed—official validation fake 32.
- Window: declared center 16 consecutive frames, history count 8, horizons `{1, 2, 4, 8}` observation offsets, 128 tracks, VGGT input 518.
- Normalization: for each window, compute history-valid centroid `c_H` and RMS radius `r_H`, then apply `(X-c_H)/r_H` to all valid rows. Future observations do not affect either statistic.
- Predictor: particle-wise single-layer `GRUCell(3, 64)` with direct 4×3 output. Missing observations carry hidden state without entering the cell. Seed `20260906`, AdamW, learning rate `1e-3`, weight decay `1e-4`, batch size 8, at most 30 epochs, patience 5 on real-validation masked MSE.
- Diagnostics: zero displacement and timestamp-aware two-point linear extrapolation. They are comparators, not model inputs or frozen method components.
- Strict CUDA deterministic algorithms were disabled because VGGT's cuBLAS operations require external workspace configuration. Python, NumPy, Torch CPU/CUDA seeds and deterministic cuDNN settings were retained.

## Data outcome

All 128 real-train, 32 real-validation, and 32 fake-probe frontend windows were causal eligible. Two eligible real-train windows had no valid particle-target at the four fixed horizons and were excluded from model fitting/evaluation without padding or zero repair. The effective counts were 126 real train, 32 real validation, and 32 frozen fake probe.

## Real-only result

The best checkpoint was epoch 16; early stopping completed after epoch 21.

| Split | Predictor | Targets | MSE | L2 mean | L2 median | L2 RMSE | L2 p90 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| real train | zero | 53,844 | 0.010732 | 0.083734 | 0.037423 | 0.179434 | 0.187801 |
| real train | linear | 53,844 | 0.041299 | 0.123166 | 0.051882 | 0.351990 | 0.263540 |
| real train | GRU | 53,844 | 0.010457 | 0.087090 | 0.043735 | 0.177120 | 0.185270 |
| real val | zero | 14,673 | 0.009170 | 0.068129 | 0.033892 | 0.165859 | 0.129676 |
| real val | linear | 14,673 | 0.072692 | 0.121191 | 0.043391 | 0.466986 | 0.207720 |
| real val | GRU | 14,673 | 0.008922 | 0.071962 | 0.039368 | 0.163606 | 0.130544 |

GRU real-validation RMSE was 1.36% lower than zero and 64.97% lower than linear. The validation/train GRU RMSE ratio was `0.9237`, so the preregistered Stage D gate passed. The gain over zero was small and appeared in RMSE rather than mean/median L2 error.

## Frozen fake probe

| Predictor | AUROC | Median fake/real score | Real-val p95 | Fake exceedance rate |
| --- | ---: | ---: | ---: | ---: |
| zero | 0.4902 | 1.0465 | 0.15289 | 0.15625 |
| linear | 0.4619 | 0.9182 | 0.36941 | 0.03125 |
| GRU | 0.4824 | 1.0566 | 0.14831 | 0.15625 |

Fake data did not participate in training, early stopping, normalization design, checkpoint selection, or hyperparameter selection. GRU AUROC did not exceed the zero baseline, and none of the three scores showed useful clip-level separation in this bounded sample.

## Decision and limitations

Result: **INCONCLUSIVE**. The preregistered real-only learnability gate passed narrowly and there was no train-to-validation degradation, but learned temporal normality did not improve real/fake separation over zero displacement. This result does not prove final forgery detection, freeze spatial topology, horizons, or aggregation, validate temporal localization, or establish VGGT three-dimensional accuracy. A spatial-dependency experiment may test the missing part of the frozen hypothesis, but this pilot alone is not evidence that spatial modeling will create separation.
