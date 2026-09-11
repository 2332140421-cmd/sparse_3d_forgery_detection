# V7 boundary/pooling pilot：已有产物汇总

状态：`BOUNDARY_POOLING_EVALUATION_COMPLETE`。本次收尾只读取已经保存的
OOF 分数、fold 模型记录和运行日志，补充分类指标、完成度核对和 loss
摘要；没有重跑前端、分组或训练。

结果根目录：
`/root/autodl-tmp/data/sparse_3d_forgery_detection/derived/v7_activityforensics_boundary_pooling_pilot_v1/`

## 1. 冻结总体与模型协议

输入是已经物化的 289 点局部结构表示，冻结总体为 16 个 source、192 个
窗口。`H` 是既有历史局部分组，`B` 是在 H 内按首帧 YOLO26m-seg 实例边界
作的子分组；B 不合并 H 父组，也不在后续帧重分配点。矩阵为
`H/B × MEAN/MAX × A/C/D` 共 12 个条件：A=`UNORDERED_STATE`、
C=`ORDERED_SECOND`、D=`PERMUTED_SECOND`。H/B 的训练和主评价均限于相同的
共同有效窗口集合。

每个 fold 是 source-disjoint LOSO；训练只使用 MANIP 窗口（real=0、fake=1），
CTRL 不参加训练或主评价。每个条件使用 seed
`20260909, 20260910, 20260911`，fold 内标准化，只用训练 fold 拟合；窗口权重
使 source/class 组等权并归一化到训练样本平均权重 1。模型配置来自保存的
`models/fold_models.json`：共享 `12→16→8` MLP、ReLU、线性头、Adam、
learning rate `1e-3`、weight decay `1e-4`、full batch、200 epochs。

主 source-level AUROC、置信区间和预声明配对 bootstrap 均沿用原有
`evaluation/summary.json`，没有更换主指标。bootstrap 是 source 等权、配对、
10,000 次重采样，seed=`20260909`。

## 2. 实际完成度与覆盖

计划任务槽位为 `12 × 16 × 3 = 576`，不是视频数量。按唯一键
`(condition, held_out_source, seed)` 统计，实际完成 `540/576`，无重复键；每个
条件均完成 45 个键（15 个 source fold × 3 seed），所有已执行 fold 的三个 seed
均存在。缺失的是 source `0HV07` 的三个 seed，12 个条件各缺 3 个槽位；没有用
零或正常分数补齐。

冻结 192 窗口中，H 有效 165、B 有效 165，共同有效 165；其中主 MANIP 评价窗口
为 81 个（fake 42、real 39），CTRL 仅作描述性记录（84 个，共 fake/real 各
42）。source-level 主 AUROC 只有 14 个 source 同时具备有效 MANIP real/fake；
`0HV07` 无执行记录，`0CG15` 的 MANIP 没有有效共同支撑，另有 `01KML`、`0LNLR`
存在部分 real MANIP 窗口缺失。覆盖明细见
[`v7-boundary-pooling-completion-audit.csv`](v7-boundary-pooling-completion-audit.csv)、
[`v7-boundary-pooling-per-source-metrics.csv`](v7-boundary-pooling-per-source-metrics.csv)。

## 3. 12 条件 source 等权平均 AUROC

区间为 source-level paired bootstrap 95% CI；点估计来自原始 14 个有效 source
的 source AUROC，不是 bootstrap 重复均值。

| 条件 | 平均 AUROC | CI 下界 | CI 上界 |
| --- | ---: | ---: | ---: |
| H_MEAN_A | 0.626984 | 0.444444 | 0.793651 |
| H_MEAN_C | 0.496032 | 0.341270 | 0.650794 |
| H_MEAN_D | 0.563492 | 0.388889 | 0.730159 |
| H_MAX_A | 0.630952 | 0.468254 | 0.789683 |
| H_MAX_C | 0.555556 | 0.388889 | 0.722222 |
| H_MAX_D | 0.547619 | 0.388889 | 0.714484 |
| B_MEAN_A | 0.666667 | 0.476190 | 0.833333 |
| B_MEAN_C | 0.571429 | 0.388889 | 0.753968 |
| B_MEAN_D | 0.563492 | 0.380952 | 0.730159 |
| B_MAX_A | 0.615079 | 0.472222 | 0.742063 |
| B_MAX_C | 0.392857 | 0.246032 | 0.535714 |
| B_MAX_D | 0.563492 | 0.436508 | 0.698413 |

没有条件的 CI 下界高于 0.5，因此这组开发数据不能建立稳定的总体判别结论。

## 4. 全部预声明配对比较

每行表示左侧条件减右侧条件，区间仍是同一 source 重采样中的配对差值。

| 比较 | 平均差值 | CI 下界 | CI 上界 |
| --- | ---: | ---: | ---: |
| H_MAX−H_MEAN_A | 0.003968 | -0.146825 | 0.158730 |
| B_MAX−B_MEAN_A | -0.051587 | -0.174603 | 0.079365 |
| B_MEAN−H_MEAN_A | 0.039683 | 0.007937 | 0.079365 |
| B_MAX−H_MAX_A | -0.015873 | -0.111111 | 0.071429 |
| B_MAX−H_MEAN_A | -0.011905 | -0.123016 | 0.103175 |
| H_MEAN_C−A | -0.130952 | -0.226190 | -0.039683 |
| H_MEAN_C−D | -0.067460 | -0.166667 | 0.027778 |
| H_MAX_C−A | -0.075397 | -0.222222 | 0.067460 |
| H_MAX_C−D | 0.007937 | -0.142857 | 0.142857 |
| B_MEAN_C−A | -0.095238 | -0.182540 | -0.015873 |
| B_MEAN_C−D | 0.007937 | -0.071429 | 0.087302 |
| B_MAX_C−A | -0.222222 | -0.388889 | -0.079365 |
| B_MAX_C−D | -0.170635 | -0.333333 | -0.043651 |
| H_MAX−H_MEAN_C | 0.059524 | -0.150794 | 0.257937 |
| B_MAX−B_MEAN_C | -0.178571 | -0.392857 | 0.023810 |
| B_MEAN−H_MEAN_C | 0.075397 | -0.007937 | 0.162698 |
| B_MAX−H_MAX_C | -0.162698 | -0.337302 | -0.019841 |
| B_MAX−H_MEAN_C | -0.103175 | -0.285714 | 0.079365 |
| H_MAX−H_MEAN_D | -0.015873 | -0.150794 | 0.111111 |
| B_MAX−B_MEAN_D | 0.000000 | -0.103175 | 0.103175 |
| B_MEAN−H_MEAN_D | 0.000000 | -0.039683 | 0.039683 |
| B_MAX−H_MAX_D | 0.015873 | -0.071429 | 0.111111 |
| B_MAX−H_MEAN_D | 0.000000 | -0.095238 | 0.095238 |

可见仅 `B_MEAN−H_MEAN_A` 的区间完全为正；这不是对所有 arm 或所有表示都
一致的增益。`C` arm 在 H_MEAN、B_MEAN、B_MAX 中相对 A 为负且区间不跨 0，
但这只说明本 pilot 的 arm 间差异，不能单独解释为物理规律。

完整数值见 [`v7-boundary-pooling-paired-gains.csv`](v7-boundary-pooling-paired-gains.csv)。

## 5. 相同 MANIP 窗口上的 pooled 分类指标

这里的样本单位是窗口；仅选 81 个共同有效 MANIP 窗口，fake 是正类，real 是负类，
CTRL 没有混入。每个窗口的分数是三个 source-disjoint seed OOF logit 的算术平均，
固定阈值为 `logit >= 0`。没有根据 OOF 标签搜索阈值；缺失分数不填 0，而是排除并
记录。本次 81 个窗口均有分数。AP 是 average precision；混淆矩阵顺序为
`TN/FP/FN/TP`。这些 pooled 指标是辅助描述，不取代 source 等权主 AUROC。

| 条件 | ROC-AUC | AP | Precision | Recall | F1 | ACC | TN/FP/FN/TP |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| H_MEAN_A | 0.556777 | 0.611723 | 0.607143 | 0.404762 | 0.485714 | 0.555556 | 28/11/25/17 |
| H_MEAN_C | 0.525031 | 0.607805 | 0.513514 | 0.452381 | 0.481013 | 0.493827 | 21/18/23/19 |
| H_MEAN_D | 0.537851 | 0.594852 | 0.607143 | 0.404762 | 0.485714 | 0.555556 | 28/11/25/17 |
| H_MAX_A | 0.556166 | 0.574611 | 0.611111 | 0.523810 | 0.564103 | 0.580247 | 25/14/20/22 |
| H_MAX_C | 0.498168 | 0.511893 | 0.555556 | 0.357143 | 0.434783 | 0.518519 | 27/12/27/15 |
| H_MAX_D | 0.478022 | 0.499987 | 0.517241 | 0.357143 | 0.422535 | 0.493827 | 25/14/27/15 |
| B_MEAN_A | 0.579976 | 0.631829 | 0.633333 | 0.452381 | 0.527778 | 0.580247 | 28/11/23/19 |
| B_MEAN_C | 0.551893 | 0.631143 | 0.567568 | 0.500000 | 0.531646 | 0.543210 | 23/16/21/21 |
| B_MEAN_D | 0.555556 | 0.612856 | 0.586207 | 0.404762 | 0.478873 | 0.543210 | 27/12/25/17 |
| B_MAX_A | 0.578755 | 0.587901 | 0.600000 | 0.571429 | 0.585366 | 0.580247 | 23/16/18/24 |
| B_MAX_C | 0.426740 | 0.468695 | 0.464286 | 0.309524 | 0.371429 | 0.456790 | 24/15/29/13 |
| B_MAX_D | 0.490842 | 0.510939 | 0.500000 | 0.309524 | 0.382353 | 0.481481 | 26/13/29/13 |

CSV：[`v7-boundary-pooling-classification-metrics.csv`](v7-boundary-pooling-classification-metrics.csv)。
本轮不计算时间定位 AP@IoU，也不计算空间定位 AP。

## 6. source/seed 稳定性与 loss

逐 source/seed 的 MANIP AUROC 共 `14 × 12 × 3 = 504` 个描述性记录，精确文件在
实验数据目录的 `evaluation/per_source_seed_metrics.csv`；仓库中提交的是汇总表
[`v7-boundary-pooling-seed-stability.csv`](v7-boundary-pooling-seed-stability.csv)。
下表给出每个条件 14 个 source、每个 source 3 个 seed 的 AUROC 总体均值，以及
先在 source 内计算三 seed 标准差后再求均值的稳定性描述：

| 条件 | source×seed AUROC 均值 | source×seed AUROC 范围 | source 内 seed std 均值 |
| --- | ---: | ---: | ---: |
| H_MEAN_A | 0.634921 | 0.000000–1.000000 | 0.018707 |
| H_MEAN_C | 0.502646 | 0.000000–1.000000 | 0.068212 |
| H_MEAN_D | 0.576720 | 0.000000–1.000000 | 0.056380 |
| H_MAX_A | 0.582011 | 0.000000–1.000000 | 0.110545 |
| H_MAX_C | 0.544974 | 0.000000–1.000000 | 0.191223 |
| H_MAX_D | 0.531746 | 0.000000–1.000000 | 0.160477 |
| B_MEAN_A | 0.640212 | 0.000000–1.000000 | 0.037413 |
| B_MEAN_C | 0.589947 | 0.000000–1.000000 | 0.075237 |
| B_MEAN_D | 0.592593 | 0.000000–1.000000 | 0.055528 |
| B_MAX_A | 0.550265 | 0.000000–1.000000 | 0.132431 |
| B_MAX_C | 0.447090 | 0.000000–1.000000 | 0.179745 |
| B_MAX_D | 0.551587 | 0.000000–1.000000 | 0.153879 |

每个条件的 45 条保存 loss history 均有限且长度为 200，45/45 的最终 loss 低于
初始 loss。这里的“收敛”仅是该描述性判据，不声称优化达到全局最优；保存记录中
没有逐 epoch AUROC/AP/F1，因此没有补造这些曲线。全部 loss 摘要见
[`v7-boundary-pooling-loss-summary.csv`](v7-boundary-pooling-loss-summary.csv)，
代表性四条件的中位数及 10–90% 带见
[`v7-boundary-pooling-loss-curves.png`](assets/v7-boundary-pooling-loss-curves.png)。

## 7. 运行成本与产物边界

已有运行日志记录的阶段耗时为：prepare `0.53 s`（恢复已有输入）、segmentation
`83.03 s`、features `24.46 s`、540 个模型训练 `3530.22 s`、evaluate `0.14 s`、
report `7.92 s`，合计约 `3646.30 s`（60.77 分钟）。fold 记录标明训练设备为
`cuda`；本次没有保存可复核的峰值显存数字，故不报告显存峰值。

提交的仅是汇总 CSV、代表性 loss PNG、报告、后处理脚本及其小型测试；不提交
`fold_models.json`、OOF/NPZ、视频、权重或 ZIP。

## 8. 结论与限制

在这 14 个可做 source-level 主评价的开发 source 上，所有 12 条件的 source 等权
AUROC CI 都跨过 0.5；因此不能宣布稳定检测能力。`B_MEAN_A` 的点估计最高
（0.666667），但 CI 下界为 0.476190。唯一完全为正的预声明表示比较是
`B_MEAN−H_MEAN_A`，其平均差值 0.039683；它没有在其他 arm 上复现，不能概括为
普遍的边界或 pooling 增益。`C` arm 在 H_MEAN、B_MEAN、B_MAX 中相对 A 为负且
区间不跨 0，但这只说明本 pilot 的 arm 间差异，不能单独解释为物理规律。

这些结果支持：当前冻结输入、局部分组和三臂训练的执行记录是可追溯的，并可用于
开发期比较。它们不支持跨生成器泛化、独立 sealed-test、时间定位、像素/空间定位、
概率校准或因果检测结论。CTRL 只保留为描述性窗口，不应被解释为伪造标签。
后续如需推进，应先由用户决定如何处理 source/窗口覆盖和独立验证；本报告不自动
启动下一阶段。
