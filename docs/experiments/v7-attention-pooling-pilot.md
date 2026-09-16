# V7 局部注意力聚合最小对照

本实验只改变 SET_A 的局部单元到窗口聚合，复用
`v7_activityforensics_source_learning_curve_v1` 的 R 特征、标签、标准化、
ordering=20260909 的 64-source 训练集合和 14-source/83-window 验证集合。
没有重跑前端，也没有修改正式 `src` 检测链。

结果目录：

`/root/autodl-tmp/data/sparse_3d_forgery_detection/derived/v7_activityforensics_attention_pooling_pilot_v1/`

## 结果

| 条件 | 训练 macro AUROC | 验证 macro AUROC | 验证 pooled AUROC | 验证 AP | 验证 F1 |
|---|---:|---:|---:|---:|---:|
| MEAN_BASELINE | 0.648452 | 0.634921 | 0.632404 | 0.695254 | 0.529412 |
| ATTENTION_POOL | 0.650273 | 0.611111 | 0.637050 | 0.702503 | 0.529412 |

主 source-macro 差 `ATTENTION_POOL − MEAN_BASELINE` 为 `−0.023810`，
source bootstrap 95% CI `[-0.055556, 0.007937]`。14 个 source 中提高 1、
持平 9、下降 4；逐 seed 和逐 source 表在数据目录的
`evaluation/per_source_metrics.csv`。

训练差为 `+0.001821`，验证差为 `−0.023810`，因此没有形成训练与验证同时改善
的证据，本候选在本轮结束，不继续增加模型或调参。验证 AP 略有上升，但主指标
source-macro AUROC 下降且区间跨 0，不能称为稳定收益。

## 实现与验证

- `MEAN_BASELINE` 保留原 SET_A 局部编码和局部 logit 的等权平均。
- `ATTENTION_POOL` 使用 `h_g`（分类头前的 8 维表示）计算一头 softmax，温度 1；
  V 使用固定 seed `20260912`，b/w 为零初始化。padding/无效单元不参与聚合。
- 两条件每个 seed 共用相同 encoder/head 初始参数；参数量为 569 与 649。
- 均匀注意力退化误差 `1.86e-9`，排列不变性误差 `9.31e-10`；梯度有限且注意力
  分支梯度非零；旧 SUMMARY_SET 验证分数复现最大误差 `2.38e-7`。
- 6/6 模型完成 200 epochs，实际 CUDA 训练累计（含一次收尾失败重试）约 14.385 s。
  第一次运行在写训练分数时因 split key 错误退出，6 个模型已原子保存；修复后只
  复用模型完成评分和报告，失败记录保存在数据目录 `state/retry_history.json`。

注意力最大权重的分布只作聚合行为描述，不是伪造位置或概率：seed 平均的训练
中位数/P95/最大值为 `0.0374/0.1228/1.0`，验证为 `0.0294/0.1524/0.3708`。
