# V7 同支撑二维／三维信息增量 pilot

## 目的与边界

本 pilot 在同一批 R query cohort、局部组、成员、canonical pair、五个目标时刻和共同有效性上，对比二维 UV 距离摘要与三维 XYZ 距离摘要。二维分支直接消费 `ParticleSequence` 中保存的源像素 UV；它复用了三维流程提供的分组和支撑，因此不是一个去掉三维前端的独立二维端到端系统。

本轮没有访问旧 R7/V5，没有下载数据，也没有重跑 tracking、depth、pose、segmentation 或前端；没有改变正式 `src/` 检测链、标签、训练/验证划分或阈值。

## 冻结输入与共同支撑

输入来自 source128 extension 的 R 缓存：

- 训练计划 718 窗口、122 个有效训练 source；
- 固定验证 83 窗口、14 个 source（42 real、41 fake）；
- 训练与验证 source 不交叉；
- 四个条件实际保留 718/83 窗口，原始与匹配 unit 均为 23,767/3,148，损失 0。

每个 unit 保留原成员、pair、五个目标帧和 PTS。目标时刻同时要求 UV visibility/有限性和 XYZ geometry validity/有限性；历史尺度使用同一历史 pair-time 有效集合，分别计算二维和三维中位数尺度，不使用逐帧或目标时刻重定尺度。原保存的 SET_A 三维状态按原尺度重建，逐 unit 最大误差不超过 `1e-5`；身份、尺度、PTS 和重建误差在 `inputs/feature_manifest.json` 中保存。

## 条件与训练

| 条件 | 输入 |
|---|---|
| `UV_2D` | S2（4 维） |
| `XYZ_3D` | S3（4 维） |
| `UV_XYZ` | 标准化 S2 + 标准化 S3（8 维） |
| `UV_UV_CONTROL` | 标准化 S2 + 标准化 S2（8 维宽度控制） |

四个条件各训练 seed `20260909/20260910/20260911`、200 epochs。模型为既有 SUMMARY_SET 风格的五时刻编码器和窗口内局部等权平均；Adam、学习率 `0.001`、weight decay `0.0001`、窗口级 source/class weighted BCE，阈值固定为 `logit >= 0`。4 维模型参数量 569，8 维模型参数量 633。标准化仅用训练窗口拟合，三 seed 先平均 window logit 再计算主指标。

## 结果

正式模型 12/12，CUDA 运行，匹配 smoke 通过。验证集三 seed 平均 logit：

| 条件 | source-macro AUROC (95% CI) | pooled AUROC | AP | Precision | Recall | F1 | ACC |
|---|---:|---:|---:|---:|---:|---:|---:|
| UV_2D | 0.4960 [0.3452, 0.6508] | 0.4779 | 0.5143 | 0.5000 | 0.4146 | 0.4533 | 0.5060 |
| XYZ_3D | 0.6032 [0.4127, 0.7857] | 0.6173 | 0.6746 | 0.6207 | 0.4390 | 0.5143 | 0.5904 |
| UV_XYZ | 0.5278 [0.3730, 0.6865] | 0.5052 | 0.5642 | 0.4595 | 0.4146 | 0.4359 | 0.4699 |
| UV_UV_CONTROL | 0.4921 [0.3175, 0.6746] | 0.4808 | 0.5091 | 0.4865 | 0.4390 | 0.4615 | 0.4940 |

训练集三 seed 平均 logit 的 source-macro / pooled AUROC 分别为：`UV_2D` 0.6230 / 0.5986，`XYZ_3D` 0.6412 / 0.6006，`UV_XYZ` 0.6794 / 0.6646，`UV_UV_CONTROL` 0.6116 / 0.6019。训练指标仅是拟合诊断，不是泛化证据。

预声明 source 配对差（验证 source-macro AUROC，10,000 次 source bootstrap，seed `20260909`）：

| 比较 | 均值差 | 95% CI | 正/负/平 source |
|---|---:|---:|---:|
| `UV_XYZ - UV_UV_CONTROL`（主要） | +0.0357 | [-0.1151, 0.1786] | 6 / 5 / 3 |
| `XYZ_3D - UV_2D` | +0.1071 | [-0.1032, 0.3135] | 9 / 4 / 1 |
| `UV_XYZ - UV_2D` | +0.0317 | [-0.1190, 0.1667] | 7 / 5 / 2 |
| `UV_UV_CONTROL - UV_2D` | -0.0040 | [-0.0556, 0.0476] | 3 / 4 / 7 |

## 结论边界

1. `XYZ_3D` 的点估计高于 `UV_2D`，但区间跨 0，不能称为稳定的三维单模态优势。
2. 主要的 `UV_XYZ` 相对二维重复控制只有小幅点估计增益，区间跨 0；本轮没有建立三维加入二维后的稳定跨 source 判别增量。
3. 共同支撑没有损失窗口或 unit，因此没有因支撑差异造成的条件间样本替换；结果仍带有同时要求 UV 与 XYZ 有效的 survivor bias。
4. 结果不足以判断三维测量误差、摘要压缩、窗口级监督或 source 异质性中的唯一原因，也不回答纯二维独立前端、因果时序、空间定位真值或 sealed-test 泛化。
5. 因此本轮只支持保留该对照作为有限测量证据；不自动启动新的模型、前端或扩容实验。

## 复核与产物

定向测试：`5 passed`（参数量/初始化退化、尺度不变性、刚体距离不变、固定 UV 的深度增量通路、8 维新增通道梯度）。smoke 验证四条件输出有限、保存重载误差小于 `1e-5`、8 维融合初始化退化为二维输出。

结果目录（数据和模型不进 Git）：

`/root/autodl-tmp/data/sparse_3d_forgery_detection/derived/v7_activityforensics_2d3d_information_pilot_v1/`

- `report.md`：运行时报告；
- `protocol.json`、`inputs/input_manifest.json`、`inputs/feature_manifest.json`：协议和输入身份；
- `support/matching_summary.json`、`support/matching_coverage.csv`：共同支撑；
- `scores/train_window_scores.csv`、`scores/validation_window_scores.csv`：逐窗口 seed/平均 logit；
- `evaluation/summary.json`、`evaluation/metrics.csv`、`evaluation/per_seed_metrics.csv`、`evaluation/per_source_metrics.csv`、`evaluation/paired_comparisons.csv`：评价；
- `models/fold_models.json`、`final_status.json`：模型记录和最终状态。

阶段总耗时约 8.1 秒（匹配 0.49、模型恢复/训练阶段 1.47、评价 5.66；smoke 已从独立产物复用）。最终状态为 `COMPLETE`，无后台进程。
