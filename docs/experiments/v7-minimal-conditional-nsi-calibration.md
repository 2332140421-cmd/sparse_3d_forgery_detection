# V7 Minimal Conditional NSI Calibration

## 1. Research Question

本诊断只检验一个问题：NSI 的跨 source 基线漂移是否可以由 lagged first-order structural activity 解释，以及该条件化是否同时保留 fake discrimination。它不是新方法、训练或完整视频检测实验。

## 2. Previous Artifact Limitation

上一轮冻结的 `metrics/per_triplet_nsi.csv` 只保存了 `I`，没有保存 `v_minus` 或其范数，因此条件 `C` 不能直接读取。本轮没有修改上一轮 artifact，而是从冻结 XYZ、validity、PTS 和 component membership 重建所需标量。

## 3. Frozen Artifact Reconstruction

- population：16 个 source pairs；window manifest：192 个 annotation-selected windows；
- 输入来自既有 paired-second-order、relation-first 和 normalized-NSI 产物；
- 未重新选择 population、window、anchor 或 component，未重跑 frontend；
- 输出只保留 `I_stored`、`I_reconstructed`、误差、范数、`M` 和 `C` 等小型标量 CSV。

## 4. Reconstruction Fidelity

`9101/9101` 条历史 triplet 一一对应，identity coverage 为 `100%`；重复 identity `0`，缺失 identity `0`，非有限值 `0`。`I_reconstructed` 与 `I_stored` 的最大、median、p99 和 RMSE 误差均为 `0.0`，状态为 `NSI_RECONSTRUCTION_VERIFIED`。

## 5. Recovered Lagged Activity Condition

唯一 condition 固定为：

`C = ||v_minus|| / sqrt(M)`

它只使用 lagged `v_minus` 范数和共同 persistent pair 数 `M`，不使用 `v_plus`、speed sum、future average、tracking、geometry、scene 或任何标签信息。C 的 min/median/p90/max 为 `3.7905e-10 / 0.55226 / 2.34371 / 164.34157`。

## 6. Real-only I-C Dependence

每个 source 仅使用 real triplets 计算 Spearman `rho(I,C)`；16 个 source 中 15 个有足够有限 real triplets，`0HV07` 没有有效 real triplet。有限 source 的 rho median/IQR/p10-p90 为 `0.16363 / 0.17792 / [-0.01776, 0.23511]`，positive fraction `0.80`。这提示存在偏弱的正向 activity 关联，但不是稳定的强关系。

## 7. U0 Unconditional Ruler

每个 LOSO fold 的 U0 仅用其他 real source 的全部 `I` 建立 source-balanced smoothed ECDF；每个 source 总权重相同。查询使用半秩平滑，之后固定使用 `A=-log(2*min(q,1-q))`。

## 8. U1 Minimal Conditional Ruler

U1 与 U0 完全复用 NSI、半秩平滑、two-sided transform、component-Q90 和 window-level component median；唯一变化是使用训练 real 的 C tertiles（LOW/MID/HIGH）选择 reference ECDF。每个 held-out source 的 cutpoints 只来自其他 15 个 real source，fake 和 held-out real 均不参与拟合。

48 个 fold/bin 支持记录均已生成；training source count 为 `13–15`，observation count 为 `1282–1447`，effective N 为 `619.82–1086.76`，弱支持记录（distinct training source `<5`）为 `0`。

## 9. Held-out Real Calibration

有限的 15 个 held-out real source 上，KS(U0) median 为 `0.09159`，KS(U1) median 为 `0.07212`；`Delta_KS=KS1-KS0` median 为 `-0.01295`，改善 fraction 为 `0.80`，source bootstrap（seed `20260909`，`10000` 次）95% CI 为 `[-0.01826, -0.00608]`。

## 10. Residual Activity Dependence

`Delta_rho=|rho1|-|rho0|` median 为 `-0.08094`，减少 fraction 为 `0.6667`，bootstrap 95% CI 为 `[-0.11414, 0.07403]`。real source window-score MAD 从 U0 的 `0.34868` 降到 U1 的 `0.21264`（delta `-0.13604`）；该 dispersion 结果仅作描述性信息。

## 11. Fake Evaluation After Scoring

所有 fake 仅在 real-only reference 固定后评分，fake fitting count 为 `0`。利用 frozen MANIP/CTRL pairing 计算的 source-level evaluation statistic（不是 detector inference formula）为：

- `J_U0`：N=`15`，median=`0.17166`，IQR=`0.77108`，positive fraction=`0.6667`，bootstrap CI=`[-0.06269, 0.66494]`；
- `J_U1`：N=`15`，median=`0.20313`，IQR=`0.79844`，positive fraction=`0.6000`，bootstrap CI=`[-0.23664, 0.56155]`；
- `Delta_J=J_U1-J_U0`：median=`-0.00386`，IQR=`0.41003`，positive fraction=`0.4667`，bootstrap CI=`[-0.24880, 0.06096]`。

探索性 AUROC（仅 development diagnostic）：fake manipulation vs real manipulation 为 U0=`0.5919`、U1=`0.6167`；fake manipulation vs all held-out real windows 为 U0=`0.5401`、U1=`0.5592`。

## 12. Conditionalization Effect

U1 对 real KS 和 source-score dispersion 有描述性改善，但 residual activity 的改善 fraction 未达到预声明的 `0.70`，`J_U1` positive fraction 也未达到 `0.70`，且 `Delta_J` median 非正。因此没有同时满足“real calibration、activity dependence、fake information retained”三项门槛。

## 13. Limitations

本轮仍是 16-source development diagnostic，不是 full-video、不是 sealed test。冻结 artifact 中有 18 个无有效 triplet 的 window，实际产生 174 个 window score；`0HV07` 没有有效 real triplet。未对缺失窗口进行插值、删除或修复，也未重新运行 frontend。

## 14. Scientific Conclusion

冻结 NSI 被精确重建。lagged activity 对 NSI baseline 存在弱的 source-level 关联，且最小条件化确实改善部分 real calibration；但这种改善没有稳定转化为 fake discrimination。综合当前有限开发样本，证据更接近 **NSI INFORMATION LIMITATION**，而不是已被充分证明的 BASELINE CONFOUND；由于有效 source/窗口缺失和预声明门槛未同时满足，主状态保守记为 `PILOT_INCONCLUSIVE`，不把它宣称为正式结论。

## 15. Next-stage Authorization

当前不授权增加第二 condition、不修改 NSI、不进入 no-reference full-video experiment，也不进入正式训练或 sealed test。任何后续方法变化须由新的明确设计决定授权。

本报告对应的产物位于数据盘：
`/root/autodl-tmp/data/sparse_3d_forgery_detection/derived/v7_activityforensics_conditional_nsi_pilot_v1/`。
