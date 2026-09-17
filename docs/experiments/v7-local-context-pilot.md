# V7 历史邻域局部时空关系 pilot

本实验检验：在同一局部自身的五时刻四维状态之外，历史半段构建的局部邻居在对应时刻的状态，是否提供稳定的跨 source 判别增量。它是开发性 pilot，不是空间伪造定位、物理规律验证或独立 sealed test。

## 输入与冻结协议

- 输入只来自 `v7_activityforensics_source128_extension_v1` 的 R `SET_A` 五时刻状态（`[local_unit, 5, 4]`）、四个实际 PTS 间隔、历史局部组、标签和 lineage；未重跑前端、tracking、depth、pose 或 segmentation。
- 训练/验证窗口分别为 718（363 real、355 fake、122 source）和 83（42 real、41 fake、14 双类别 source），source 不交叉。五个条件使用完全相同的窗口和有效 local unit；没有邻居的节点保留，缺失状态不填零。
- 历史图只用历史帧的至少三点质心；组间历史重叠至少 8 帧，窗口级 `d0` 的 2 倍内最多取 3 个邻居。边在有效五时刻节点上取诱导子图。`NEIGHBOR_REWIRED` 为固定有向双边交换，`NEIGHBOR_TIME_SHUFFLED` 为固定非恒等五时刻排列；变换不读标签或目标时刻状态。
- `SUMMARY_BASELINE` 使用本目录独立训练的原 `SetAModel/SUMMARY_SET`；其余四条件使用相同 2,793 参数的局部关系/时间模型（状态 4→16→8，关系 32→16→8，两个 Conv1d，44→16→1）。三个 seed 为 20260909/10/11，200 epochs，Adam，lr=0.001，weight decay=0.0001，source/class weighted BCE，阈值为 logit≥0。标准化只由训练窗口拟合。

## 实际结果

三 seed 先平均窗口 logit，再计算验证指标：

| 条件 | source-macro AUROC | pooled AUROC | AP | Precision | Recall | F1 | ACC |
|---|---:|---:|---:|---:|---:|---:|---:|
| SUMMARY_BASELINE | 0.6032 | 0.6161 | 0.6728 | 0.6207 | 0.4390 | 0.5143 | 0.5904 |
| SELF_TEMPORAL | 0.5794 | 0.5790 | 0.6381 | 0.5000 | 0.3171 | 0.3881 | 0.5060 |
| NEIGHBOR_TEMPORAL | 0.5516 | 0.5557 | 0.6203 | 0.5667 | 0.4146 | 0.4789 | 0.5542 |
| NEIGHBOR_REWIRED | 0.5595 | 0.5511 | 0.6141 | 0.5152 | 0.4146 | 0.4595 | 0.5181 |
| NEIGHBOR_TIME_SHUFFLED | 0.5476 | 0.5296 | 0.6020 | 0.5000 | 0.3171 | 0.3881 | 0.5060 |

Source-level paired bootstrap（10,000 次，seed=20260909）结果：

| 比较 | 均值差 | 95% CI | 正/负/平 source 数 |
|---|---:|---|---:|
| NEIGHBOR_TEMPORAL − SELF_TEMPORAL | −0.0278 | [−0.1071, 0.0556] | 3 / 7 / 4 |
| NEIGHBOR_TEMPORAL − NEIGHBOR_REWIRED | −0.0079 | [−0.0952, 0.1032] | 2 / 4 / 8 |
| NEIGHBOR_TEMPORAL − NEIGHBOR_TIME_SHUFFLED | +0.0040 | [−0.0635, 0.0794] | 3 / 4 / 7 |
| NEIGHBOR_TEMPORAL − SUMMARY_BASELINE | −0.0516 | [−0.1786, 0.0794] | 3 / 6 / 5 |

训练集三 seed 平均的 source-macro AUROC 为：SUMMARY 0.6384、SELF 0.7049、NEIGHBOR 0.7227、REWIRED 0.7072、TIME_SHUFFLED 0.7122；验证集没有同步改善。验证集逐 seed 的 `NEIGHBOR_TEMPORAL − SELF_TEMPORAL` 为 +0.0040、−0.0278、−0.0516，方向不一致。

图覆盖为 801 个选定窗口；其中 778 个窗口发生了实际重连，801 个时间排列均为非恒等。R 支持缓存总体为 805/864 行有效；其余行保留原缺失原因。以上计数不把“有邻居”当作有效性筛选，也不表示局部物理真值。

## 结论边界

在本 pilot 的固定表示、历史图规则和小模型下，正确邻居没有显示出相对自身关系或摘要基线的稳定增量；时间打乱对照的差值也跨零。因此不能据此声称邻域或共同时间演化已提供额外判别信息。训练拟合提高而验证下降，符合开发验证上过拟合/跨 source 不稳定的可能，但不能由此区分唯一原因。验证集来自既有开发流程，窗口有重叠且 source 数有限；五时刻支撑是离线事后筛选，不是严格因果输入。消息、局部 logit 和邻居距离不等同于空间定位或校准概率。

本轮建议停止扩展模型家族；若要继续，只应先提出新的、可证伪的表示问题，而不是增加更多邻居或调参。

## 产物与复核

数据产物（不进 Git）：

- `/root/autodl-tmp/data/sparse_3d_forgery_detection/derived/v7_activityforensics_local_context_pilot_v1/report.md`
- `protocol.json`、`inputs/input_manifest.json`、`inputs/graph_manifest.json`、`coverage/r_support_coverage.csv`
- `models/fold_models.json`、`scores/train_window_scores.csv`、`scores/validation_window_scores.csv`
- `evaluation/summary.json`、`evaluation/metrics.csv`、`evaluation/per_source_metrics.csv`、`evaluation/paired_comparisons.csv`
- `final_status.json`（COMPLETE，15/15 模型）

逐 seed、逐 source 指标和每个窗口 logit 以 CSV 保存；报告中的 CI 是 source bootstrap 描述，不是最终泛化保证。运行时 PyTorch 为 2.3.1+cu121，设备为 CUDA；CUDA deterministic flags 按实际运行状态记录，未进行额外确定性审计。
