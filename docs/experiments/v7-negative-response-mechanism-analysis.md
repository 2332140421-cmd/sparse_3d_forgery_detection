# V7 Structural-Normality Negative-Response Mechanism Analysis

## 1. Research Question

本实验解释：为什么 frozen real-only 三维结构演化 normality 在 generated videos 上产生更低的 anomaly score。它是机制分析，不是重新训练、调参或新的 detector。

## 2. Prior Observation

此前 unpaired response 的窗口/视频评估为：

| model | AUROC(fake higher) | Cliffs delta |
|---|---:|---:|
| M1 = ΔS | 0.3333 | -0.3333 |
| M2 = Δ²S | 0.1944 | -0.6111 |
| M3 = [ΔS, Δ²S] | 0.3472 | -0.3056 |

本轮不翻转 score，不把 1-AUROC 作为检测结果，也不对 H3 作最终真假判定。

## 3. Frozen Population and Artifacts

| population | count | use |
|---|---:|---|
| Vript real_train sources | 24 | 仅用于 frozen mean/std/reference cloud |
| real_train windows | 72 | 仅用于 reference statistics |
| Vript real_val sources | 8 | held-out real comparison |
| real_val windows | 24 | held-out real comparison |
| generated videos | 10 | 5 SVD + 5 CogVideo |
| underlying Pair2 ordinals | 5 | fake cluster bootstrap unit |
| scoreable fake videos | 9 | score-dependent quantities |
| unscoreable fake videos | 1 | 保留在人口统计；CogVideo ordinal 00042 三窗口无 component |

frozen model 文件 SHA-256：

4e02e45bd56be8a7081c2c2af5046fbe7697434b110d016313b4d502b3d32847

fake 未进入 reference fitting，未 refit normality model。分析工具仅读取：

- derived/v7_real_only_normality_pilot_v1/
- derived/v7_unpaired_fake_response_v1/

分析输出写入数据盘：

/root/autodl-tmp/data/sparse_3d_forgery_detection/derived/v7_negative_response_mechanism_v1/

其中包括 mechanism_summary.json、per_video_mechanism.json/csv、per_dimension_summary.json/csv、compression_diagnostics.json 和 run_summary.json；这些派生文件未加入 Git。

## 4. Aggregation Reconciliation

三个层次严格区分：

1. observation level：每个有效 component-time 的原始 ΔS、Δ²S 或拼接 M3 向量；frozen Gaussian score 是 observation 的 squared Mahalanobis。
2. window level：窗口内 score 的 primary 聚合为 p95，median 为 sensitivity 聚合。
3. source/video level：unpaired primary 使用每个 source 的 window-p95 中位数；real-only 报告的 source_level_val 使用三个 window median 的中位数。两者不是相同显示层。

因此旧报告中的 real-val source median 与 unpaired 报告的 real primary median 不应直接比较。例如：

| model | real-only source median of window medians | unpaired real primary median of window p95 |
|---|---:|---:|
| M1 | 0.02990 | 0.42753 |
| M2 | 0.02496 | 0.32759 |
| M3 | 0.07116 | 0.67286 |

本轮所有 real/fake effect size 均先聚合到 video/source，不把 component-time 当独立 video 样本。cluster bootstrap 固定 seed 20260908、10,000 次；real cluster 为 8 个 real source，fake cluster 为 5 个 Pair2 ordinal，并将同一 ordinal 的 SVD/CogVideo variant 一起采样。主要 bootstrap 区间也保存在 JSON：Δ²S median 的 AUROC(fake higher) 95% CI 为 [0, 0.475]，R 为 [0, 0]，M1 kNN-1 为 [0.214, 0.819]，pair temporal IQR 为 [0.5, 1.0]。这些是探索性 cluster 区间，不是显著性结论。

## 5. First-order Structural Activity

以下为 video-level 统计；单元格为 median / IQR [p10, p90]。fake 的 effect 是 fake higher，不是 detector 结果。

| quantity | real (N=8) | fake (N=9) | SVD (N=5) | CogVideo (N=4) | fake Cliffs delta |
|---|---:|---:|---:|---:|---:|
| L2(ΔS) median | 0.3174 / 0.5858 [0.1611, 0.9060] | 0.3022 / 0.2328 [0.1824, 0.6971] | 0.2321 / 0.2650 [0.1473, 0.7043] | 0.3726 / 0.2036 [0.2748, 0.5917] | -0.1111 |
| L2(ΔS) p95 | 2.0990 / 2.6962 [1.1811, 7.3859] | 1.3271 / 1.5340 [0.5151, 9.2914] | 0.7630 / 6.4680 [0.4219, 13.8148] | 1.5474 / 0.6696 [1.0563, 2.1382] | -0.2778 |

一阶活动的总体 fake 方向略低，但 SVD 与 CogVideo 不一致：SVD 较低，CogVideo 的 median 高于 real。因此不能把低一阶活动称为跨 generator 的统一机制。

## 6. Second-order Structural Activity

| quantity | real (N=8) | fake (N=9) | SVD (N=5) | CogVideo (N=4) | fake Cliffs delta |
|---|---:|---:|---:|---:|---:|
| L2(Δ²S) median | 14.3295 / 28.7567 [5.8933, 42.3017] | 4.1906 / 6.2247 [3.2861, 12.9088] | 4.1906 / 8.8218 [2.8597, 14.0842] | 5.1254 / 3.5714 [3.6987, 8.8621] | -0.5833 |
| L2(Δ²S) p95 | 73.9767 / 109.4074 [53.6088, 268.4842] | 21.3903 / 15.6878 [8.4074, 131.9936] | 10.1340 / 74.9204 [8.2015, 229.1194] | 22.6234 / 4.9749 [15.7414, 25.2322] | -0.6667 |

二阶活动在两个 generator 上都低于 real，方向比一阶更一致，但 fake source 数量仍为 5 个 underlying ordinals。

## 7. Relative Structural Roughness

固定 epsilon = 1e-12，仅用于数值稳定性：

R = median_t(||Δ²S_t||₂) / (median_t(||ΔS_t||₂) + epsilon)。

| quantity | real (N=8) | fake (N=9) | SVD (N=5) | CogVideo (N=4) | fake Cliffs delta |
|---|---:|---:|---:|---:|---:|
| R | 46.2014 / 6.3145 [39.3954, 49.7546] | 17.5040 / 3.4501 [13.6407, 22.6433] | 18.0542 / 4.0153 [17.5677, 24.5727] | 14.2632 / 1.1545 [12.9369, 14.9420] | -1.0000 |

fake 的一阶 median 并未按同等比例下降，而二阶和 R 均明显更低，故描述性上支持 OVER_SMOOTH_EVOLUTION_SUPPORTED。不过 quality correlation、unpaired source mismatch 与小样本使该解释仍非因果证明。

## 8. Within-video Dispersion

下表为每个 video 先计算 observation dispersion 后的 video-level median / IQR；covariance trace 仅作 descriptive。

| quantity | real | fake | SVD | CogVideo |
|---|---:|---:|---:|---:|
| ΔS vector-norm MAD | 0.1734 / 0.3033 | 0.1473 / 0.2352 | 0.0750 / 0.2360 | 0.1824 / 0.1306 |
| ΔS vector-norm IQR | 0.4477 / 0.6360 | 0.4102 / 0.6132 | 0.1557 / 0.8246 | 0.4124 / 0.1277 |
| ΔS covariance trace | 0.9520 / 3.2778 | 0.4363 / 0.8706 | 0.1084 / 6.5096 | 0.6678 / 0.5201 |
| Δ²S vector-norm MAD | 7.6065 / 17.1040 | 1.9917 / 4.1577 | 1.6351 / 7.3419 | 2.8061 / 2.1850 |
| Δ²S vector-norm IQR | 19.7333 / 30.0068 | 6.3355 / 7.1688 | 3.4651 / 15.0333 | 7.1603 / 2.3355 |
| Δ²S covariance trace | 2005.6585 / 6728.7657 | 94.0462 / 163.2870 | 31.9021 / 2115.3519 | 139.2923 / 102.3620 |

逐维 MAD、逐维 IQR 和每个 video 的完整 covariance trace 已保留在 per_video_mechanism.json。总体上 Δ²S 的 fake dispersion 较低；ΔS 的方向不完全一致，CogVideo 尤其不支持简单的统一“更集中”表述。

## 9. Per-dimension Analysis

每项为 video-level median / IQR；维度语义来自当前 S_t=[mean,std,p25,p75]，没有根据结果重新选择维度。

### ΔS

| dim / semantic | real | fake | SVD | CogVideo |
|---|---:|---:|---:|---:|
| dim0 / mean | 0.1330 / 0.1305 | 0.1072 / 0.0921 | 0.0662 / 0.1692 | 0.1177 / 0.0374 |
| dim1 / std | 0.0918 / 0.1246 | 0.1091 / 0.0299 | 0.0867 / 0.0264 | 0.1131 / 0.0299 |
| dim2 / p25 | 0.1248 / 0.2462 | 0.0920 / 0.1004 | 0.0521 / 0.0822 | 0.1222 / 0.1225 |
| dim3 / p75 | 0.1728 / 0.2695 | 0.1801 / 0.1303 | 0.1413 / 0.1303 | 0.1840 / 0.0417 |

### Δ²S

| dim / semantic | real | fake | SVD | CogVideo |
|---|---:|---:|---:|---:|
| dim0 / mean | 6.0399 / 5.8573 | 1.5513 / 2.1342 | 1.5513 / 2.3443 | 2.0489 / 1.7537 |
| dim1 / std | 4.1794 / 5.9040 | 1.6408 / 1.1336 | 2.0001 / 0.9267 | 1.4990 / 0.5242 |
| dim2 / p25 | 4.6826 / 9.3637 | 1.4313 / 1.1245 | 0.8560 / 0.9152 | 1.7059 / 1.8944 |
| dim3 / p75 | 8.1693 / 14.2416 | 2.9779 / 1.5740 | 2.9779 / 1.5740 | 2.8167 / 1.4026 |

主要变化不是单一维度：ΔS 中 dim1 fake 略高且 dim3 接近/略高，dim0、dim2 较低；Δ²S 四维均低于 real，支持二阶整体变化减弱而不是某个单独统计量解释全部结果。

## 10. Distance to Frozen Real Center

下表为 raw Euclidean distance to frozen real-train mean 的 video median：

| model | real | fake | SVD | CogVideo | fake Cliffs delta |
|---|---:|---:|---:|---:|---:|
| M1 / ΔS | 0.3254 | 0.3202 | 0.2193 | 0.3693 | -0.1111 |
| M2 / Δ²S | 15.0094 | 5.4157 | 4.9158 | 6.2461 | -0.6389 |
| M3 / concat | 15.0176 | 5.4170 | 4.9275 | 6.2521 | -0.6389 |

对应 frozen Mahalanobis distance medians：

| model | real | fake | SVD | CogVideo |
|---|---:|---:|---:|---:|
| M1 | 0.1516 | 0.1223 | 0.0738 | 0.1473 |
| M2 | 0.1456 | 0.0413 | 0.0329 | 0.0524 |
| M3 | 0.2099 | 0.1389 | 0.0926 | 0.1777 |

Train-standardized Euclidean distance、p95 以及每 video 的完整分布在 JSON 中。fake 的 M2/M3 确实更接近 frozen center；M1 的整体中心差异很弱且 generator-specific。

## 11. Distance to Real Observation Cloud

kNN 仅使用 24 real_train 的 train-standardized observation cloud，固定 k=1,5，不是新 detector、没有 k 调参、没有 fake fitting。

| model / diagnostic | real | fake | SVD | CogVideo | fake Cliffs delta |
|---|---:|---:|---:|---:|---:|
| M1 1NN | 0.01742 | 0.01826 | 0.01653 | 0.01900 | +0.0556 |
| M1 5NN mean | 0.02439 | 0.02634 | 0.02293 | 0.02648 | +0.1111 |
| M2 1NN | 0.01832 | 0.01057 | 0.00950 | 0.01075 | -0.6111 |
| M2 5NN mean | 0.02673 | 0.01342 | 0.01290 | 0.01427 | -0.5833 |
| M3 1NN | 0.05684 | 0.04993 | 0.03335 | 0.05165 | -0.2500 |
| M3 5NN mean | 0.06871 | 0.05921 | 0.04226 | 0.06214 | -0.3333 |

判别结果不是单一模式：M1 中 fake 的 Mahalanobis/center distance 较低但 kNN-1/5NN 略高，支持 M1-specific GAUSSIAN_CENTER_MISMATCH_SUPPORTED；M2 中 fake 同时更靠近 Gaussian 和 kNN cloud，M3 也更接近 cloud。因此不能说所有 fake 只是“靠近 Gaussian center”，也不能说所有 fake 已经位于当前 real manifold 内。

## 12. Representation Compression Diagnostic

现有 ParticleSequence/component artifact 可以重建 normalized pairwise distances。分析覆盖 18 个 video、1,105 个 component-time rows；没有重新运行 frontend，也没有把诊断量加入 S_t。

| diagnostic | real | fake | SVD | CogVideo | fake Cliffs delta |
|---|---:|---:|---:|---:|---:|
| pairwise p10 | 0.4183 | 0.4016 | 0.3750 | 0.4065 | -0.3889 |
| pairwise p90 | 1.7728 | 1.7951 | 1.8997 | 1.6229 | +0.1667 |
| range p90-p10 | 1.3596 | 1.4201 | 1.5217 | 1.2594 | +0.3333 |
| pairwise MAD | 0.3540 | 0.3798 | 0.3824 | 0.3683 | +0.4444 |
| pairwise skewness | 0.4845 | 0.4990 | 0.6931 | 0.3393 | +0.1111 |
| pair temporal MAD | 0.0130 | 0.0236 | 0.0224 | 0.0269 | +0.5556 |
| pair temporal IQR | 0.0240 | 0.0478 | 0.0478 | 0.0544 | +0.5556 |

预先规定的 pairwise diagnostics 出现多个方向不一致但 temporal pair dispersion 在 fake 较高的现象。因此它只能支持 CURRENT_REPRESENTATION_COLLAPSE_SUSPECTED 这一弱候选，不能证明 4D S_t 丢失了决定性信息。完整逐 video 行在 compression_diagnostics.json 和 per_video_mechanism.csv。

## 13. Generator Consistency

SVD/CogVideo 在 Δ²S 和 R 的方向一致，且 M2/M3 center/cloud distances 多为更低；但 ΔS median 的方向相反，M1 kNN 方向也不同。故任何结论都标记为描述性或 model-specific，不能声称 universal fake mechanism。

## 14. Measurement-quality Control

复用既有 geometry/tracking quality，不过滤视频、不按相关性调阈值。Spearman rho（overall N=17 scored videos；括号为 fake-only N=9）：

| quantity | geometry overall (fake-only) | tracking overall (fake-only) |
|---|---:|---:|
| L2(ΔS) median | -0.597 (-0.469) | -0.440 (-0.600) |
| L2(Δ²S) median | -0.324 (-0.218) | +0.012 (-0.383) |
| R | +0.253 (+0.619) | +0.645 (+0.633) |
| M1 center distance | -0.603 (-0.469) | -0.450 (-0.600) |
| M2 center distance | -0.311 (-0.268) | +0.046 (-0.433) |
| M1 kNN-1 | -0.479 (-0.318) | -0.416 (-0.450) |
| M2 kNN-1 | -0.340 (-0.402) | -0.004 (-0.533) |

这些是控制性相关，不是因果校正。尤其 R 与 tracking persistence 的相关性意味着质量因素不能被排除；因此“过度平滑”仍是候选解释而非已证实原因。

## 15. Mechanism Interpretation

本轮标签：

- LOW_STRUCTURAL_ACTIVITY_SUPPORTED：**false**。一阶 median 的 SVD/CogVideo 方向不一致，且不能同时满足两个 generator 的统一低活动条件。
- OVER_SMOOTH_EVOLUTION_SUPPORTED：**true（描述性）**。一阶总体相近/仅略低，而二阶和 R 明显较低；但质量相关、unpaired domain 和小样本限制其强度。
- GAUSSIAN_CENTER_MISMATCH_SUPPORTED：**true（M1-specific）**。M1 center distance 降低但 kNN 距离不降低；M2/M3 不满足这一全局模式。
- CURRENT_REPRESENTATION_COLLAPSE_SUSPECTED：**true（仅 suspected）**。现有 pairwise artifact 显示与四维摘要不同的分布/时间离散信息，但方向混合且未做新的表示验证。
- FAKE_LIES_INSIDE_CURRENT_REAL_FEATURE_MANIFOLD：**false（全模型规则）**。M2/M3 的 kNN 更低，但 M1 的 kNN 不更低，不能对全部模型成立。
- MECHANISM_UNRESOLVED：**true**。source mismatch、real N=8、fake 仅 5 个 underlying ordinals、generator inconsistency 和 1 个不可评分 fake 共同阻止机制定论。

## 16. Implication for V7

当前最稳妥的下一阶段优先级是先审查/扩展 S_t representation 与 conditional structural evolution 的可辨识性，再评估 density model。原因是二阶演化的负向响应较稳定，但 M1 的 Gaussian-center mismatch 和 pairwise compression signal 同时存在；仅替换单 Gaussian 不能解释全部模型。这里不实现任何方法，不冻结组件、表示或 density model。

## 17. Limitations

- real/fake 为 unpaired source domains；
- held-out real 只有 8 个 source；
- fake 只有 5 个 underlying Pair2 ordinals；
- 只有 SVD 与 CogVideo 两个 generator；
- 1 个 CogVideo fake 无 component，无法计算 M1/M2/M3 score；
- 当前 component 与四维 S_t 是既有 baseline，不代表最终 V7 表示；
- quality correlation 不能替代受控配对实验；
- cluster bootstrap CI 在小 cluster 数下只作探索性不确定性范围；
- pairwise compression diagnostics 只从既有 artifact 重建，未形成新的 detector。

## 18. Conclusion

当前 fake score 更低最符合“二阶 structural evolution 较低、相对 roughness 较低”的描述性现象，并伴随一个仅在 M1 明显的 Gaussian-center mismatch；pairwise diagnostics 还提示四维摘要可能遗漏关系分布/时间离散信息。由于 generator、source 和样本量限制，**尚不能确定唯一机制**，也不能把该结果表述为 H3 的支持或拒绝。
