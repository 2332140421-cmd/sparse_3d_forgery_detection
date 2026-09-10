# V7 289 点匹配监督 pilot

状态：`C289_DISCRIMINATION_NOT_ESTABLISHED`；时间组织和增密增益也未达到预声明门槛。

本实验使用提交 `007753d503e23a7a0dfb1db02e6f3596624948d9` 下的冻结实现，产物位于：

`/root/autodl-tmp/data/sparse_3d_forgery_detection/derived/v7_activityforensics_density_supervised_pilot_v1/`

## 协议

- 16 个 source、192 个窗口；64/289 使用同一源视频、源帧、Depth Pro 深度、内参和位姿。
- 289 网格为 17×17；剩余窗口的 64 网格是 289 的奇行奇列嵌套子集。已完成的 16 窗口密度结果原样复用。
- 共同技术支撑窗口由分数前确定：166/192；26 个窗口两种密度均为 `NO_VALID_TRIPLET`，不填零、不改成正常样本。
- 主训练只用 MANIP 窗口，real=0、fake=1；CTRL 不进入拟合，只做描述性评分。
- 三个臂为 `UNORDERED_STATE`、`ORDERED_SECOND`、`PERMUTED_SECOND`；不重复 `ORDERED_FIRST`。
- source-disjoint LOSO、fold 内 real+fake 加权标准化、source/class 平衡窗口权重、Adam 200 epochs、三个 seed（20260909/10/11）。共 252 条 fold-model 记录（14 个可评分 fold）。
- source-level paired bootstrap：seed=20260909、10000 次；点估计来自原始 source 结果。

## 主结果

| 指标 | source 等权点估计 | 95% CI |
| --- | ---: | ---: |
| C289 `ORDERED_SECOND` | 0.571429 | [0.412698, 0.722222] |
| A289 `UNORDERED_STATE` | 0.587302 | [0.404762, 0.761905] |
| D289 `PERMUTED_SECOND` | 0.507937 | [0.373016, 0.650794] |
| C64 `ORDERED_SECOND` | 0.484127 | [0.373016, 0.595238] |
| C289−A289 | −0.015873 | [−0.222222, 0.190476] |
| C289−D289 | 0.063492 | [−0.134921, 0.253968] |
| C289−C64 | 0.087302 | [−0.039683, 0.222421] |

C289 的下界没有超过 0.5，三个差值区间也都跨 0。因此本轮没有建立跨 source 的稳定线性/监督判别、时间组织增益或匹配条件下的密度增益。逐 source 六臂结果、fold 权重审计、OOF logit 和 seed 波动见 `evaluation/per_source_metrics.csv`、`evaluation/fold_support.csv`、`scores/oof_window_scores.csv` 与 `evaluation/summary.json`。Pooled AUROC 仅作辅助，不能替代 source 等权结论。

这仍是有限开发 population pilot，不是 full-video、sealed-test、像素定位或未知生成器泛化结果；没有使用 paired reference、ROI、source 或 generator 作为数值输入。CTRL 的分数不被解释为确定正/负类。

用户肉眼认为增密改善明显，但没有正式 ROI 覆盖证据；本轮不以该观察替代覆盖测量，也不把它写成已经验证的定位能力。
