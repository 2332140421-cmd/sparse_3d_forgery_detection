# V7 当前数据集内跨 source 训练规模学习曲线

本实验是开发性 validation pilot，用来描述在固定 R 前端、固定 `SET_A`
模型和固定 validation source 下，增加独立训练 source 的表现变化。它不是
sealed test、full-video 检测或未知生成器泛化验证。

## 冻结协议

- 官方 train split、ActivityForensics revision
  `a34d4b7b04b0f3f3e26ba900adc367218667c581`；历史 16 个 development
  source 排除。
- 冻结 16 个 validation source 与 64 个训练池 source；训练池由
  `20260908` 的 generator/operation cell round-robin 选择，再用
  `20260909`、`20260910` 两个标签无关 ordering 构造嵌套的 8/16/32/64
  source 前缀。选择清单在数据目录的
  `manifests/source_split_manifest.csv` 和
  `manifests/training_subsets.json`。
- 每个 source 使用 real/fake 的 MANIP50 周期父片段和 `b=0,0.5,1.0`
  三个 1 秒子窗口；标签只把 real 记为 0、完整落在合并官方 manipulation
  区间内的 fake 记为 1。边界和区间外不进入监督训练。
- 前端为既有 289-query periodic re-query 路径；训练只使用 R 特征，未用
  O/R paired 资格限制训练。每个训练 subset 内重新拟合标准化和
  source/class weighted BCE；`SetAModel`/`SET_A`、Adam (`lr=1e-3`,
  `weight_decay=1e-4`)、200 epochs、seed `20260909/10/11` 固定不变。
- validation 不参与标准化、权重、epoch 或阈值选择。主指标在标签合格、
  R-valid 且 source 同时有 real/fake 的 validation source 上计算；本次实际
  为 14 个 source、83 个窗口（42 real、41 fake）。source-macro 与 pooled
  使用同一窗口集合，bootstrap 按 source、10000 次、seed `20260909`。
  `AP@tIoU` 未评价。

## 实际完成度

数据目录
`/root/autodl-tmp/data/sparse_3d_forgery_detection/derived/v7_activityforensics_source_learning_curve_v1/`
保存了冻结协议、媒体身份、480 个前端子窗口、960 个 O/R 支撑行、24 个
训练模型记录、逐窗口分数和评价。前端累计耗时约 4912.1 s（7200 s 预算），
训练累计约 33.0 s（1800 s 预算）；端点 smoke（8 source、200 epochs、三
seed 的最小完整链）通过。最终状态为 `COMPLETE`。大数组、视频和模型权重
均未加入 Git。

## 学习曲线主结果

下表是三个 seed 先平均 logit 后的结果；CI 是 source-level bootstrap，不能
被解读为最终泛化保证。

| ordering seed | train source | macro AUROC (95% CI) | pooled AUROC | AP | real/fake windows |
|---:|---:|---:|---:|---:|---:|
| 20260909 | 8  | 0.294 (0.143, 0.452) | 0.326 | 0.413 | 42/41 |
| 20260909 | 16 | 0.397 (0.230, 0.579) | 0.376 | 0.472 | 42/41 |
| 20260909 | 32 | 0.571 (0.389, 0.746) | 0.533 | 0.598 | 42/41 |
| 20260909 | 64 | 0.635 (0.429, 0.817) | 0.631 | 0.696 | 42/41 |
| 20260910 | 8  | 0.675 (0.508, 0.825) | 0.591 | 0.641 | 42/41 |
| 20260910 | 16 | 0.563 (0.381, 0.738) | 0.588 | 0.612 | 42/41 |
| 20260910 | 32 | 0.556 (0.357, 0.754) | 0.552 | 0.624 | 42/41 |
| 20260910 | 64 | 0.643 (0.444, 0.825) | 0.632 | 0.698 | 42/41 |

两种 ordering 的平均 macro AUROC 约为 0.484、0.480、0.563、0.639（8/16/32/64
source）；对应平均 pooled AUROC 约为 0.459、0.482、0.542、0.632。8-source
结果在两个 ordering 间差异很大，说明当前 source 数量很少时对具体 source 组合
敏感；64-source 的两个 ordering 较接近，但 CI 仍很宽且下界未形成确认性证据。

固定 `logit >= 0` 的 Precision/Recall/F1/ACC 和混淆矩阵，以及逐 seed 行，
见数据目录 `evaluation/learning_curve_metrics.csv`；完整 JSON 在
`evaluation/summary.json`，逐窗口 logit 在 `scores/validation_window_scores.csv`。

## 解释边界

结果支持在本地候选数据上继续观察“训练 source 数量与跨 source 表现”的关系，
但不支持把上升曲线解释为方法已具备稳定泛化，也不支持选择某一个 ordering、
source 子集或阈值。14 个双类别 source、重叠时间窗口、开发数据复用和没有空间
真值均限制结论；没有像素级定位、物理机制或未知生成器证据。本轮未访问旧
R7/V5，未修改正式 `src` 检测链，也未运行测试 split。
