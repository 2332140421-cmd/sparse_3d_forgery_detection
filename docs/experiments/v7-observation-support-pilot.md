# V7 结构状态与观测支撑信息增量 pilot

## 目的与边界

本轮只在已完成的 R 缓存上检验：结构状态 `S` 之外，原始历史组成员的 visibility、geometry validity 及相对历史变化 `Q` 是否提供增量。没有重跑 tracking、depth、pose、segmentation 或前端，没有下载数据，没有访问旧 R7/V5，也没有修改正式 `src/` 检测链。

## 输入身份与成员口径

冻结输入为 source128 extension / 2D-3D pilot 的同一集合：718 个训练窗口、122 个训练 source；83 个主验证窗口、14 个双类别 source（42 real、41 fake）。训练和验证 source 不交叉。

`S` 直接复用既有五时刻 XYZ 摘要和共同有效 unit。`Q` 使用同一 unit 对应的 `grouping.groups.member_slots/track_ids` 原始历史成员，不能用 `support.units` 的收缩成员替代分母。每个目标时刻：

```text
v = mean(visibility[t, raw_members])
g = mean(geometry_validity[t, raw_members])
Q(t) = [v, g, v-v(history_ref), g-g(history_ref)]
```

历史参照是 `history_array_indices` 中严格早于第一个实际目标 PTS 的最后一帧。`visibility` 和 `geometry_validity` 是保存的前端判断，不是真实遮挡或几何真值；Q 的差值不是速度、加速度或物理形变。

共同集合未改变：训练/验证仍为 718/83 窗口，23,767/3,148 unit；无支撑或未形成结构的组没有填 0，也未进入分类器。原始组与共同组成员、track ID、目标帧/PTS、查询年龄和历史参照保存在产物 `inputs/feature_manifest.json`。

## 条件与训练

| 条件 | 输入 |
|---|---|
| `STRUCTURE_ONLY` | S（4 维） |
| `SUPPORT_ONLY` | Q（4 维） |
| `STRUCTURE_SUPPORT` | 标准化 S + 标准化 Q（8 维） |
| `STRUCTURE_DUP_CONTROL` | 标准化 S + 标准化 S（8 维宽度控制） |

四条件各使用 seed `20260909/20260910/20260911`、200 epochs、Adam (`lr=0.001`, `weight_decay=0.0001`)、窗口级 source/class weighted BCE 和固定 `logit >= 0`。标准化只在训练 source 拟合，三 seed 平均 logit 后评价。4 维模型 569 参数，8 维模型 633 参数。

## Q 可用性

- 有效 Q unit：26,915；目标阶段存在支撑变化的 unit：2,246。
- 原始成员数范围/均值/最大值：3 / 5.3124 / 85。
- 共同成员数范围/均值/最大值：3 / 5.0810 / 85。
- 查询年龄范围：0.5000–0.9333 s；历史参照年龄范围：0.4444–0.5000 s。
- visibility 与 geometry validity 在原始组成员/目标时刻的相等比例为 0.9999902，并非逻辑上完全相同。
- 四通道完整范围、标准差和恒定比例见 `support/summary.json`；source/role/b 粗覆盖见 `support/q_coverage.csv`。
- 选定窗口内未形成结构的原始组共 30,980，只作为观测盲区描述。

## 验证结果

主验证集三 seed 平均 logit：

| 条件 | source-macro AUROC (95% CI) | pooled AUROC | AP | Precision | Recall | F1 | ACC |
|---|---:|---:|---:|---:|---:|---:|---:|
| STRUCTURE_ONLY | 0.6032 [0.4127, 0.7857] | 0.6156 | 0.6727 | 0.6207 | 0.4390 | 0.5143 | 0.5904 |
| SUPPORT_ONLY | 0.6349 [0.4921, 0.7698] | 0.5833 | 0.6320 | 0.6500 | 0.3171 | 0.4262 | 0.5783 |
| STRUCTURE_SUPPORT | 0.6667 [0.5000, 0.8175] | 0.6283 | 0.6759 | 0.6452 | 0.4878 | 0.5556 | 0.6145 |
| STRUCTURE_DUP_CONTROL | 0.6111 [0.4206, 0.7857] | 0.6289 | 0.6845 | 0.5806 | 0.4390 | 0.5000 | 0.5663 |

source 配对 bootstrap（10,000 次，seed `20260909`）：

| 比较 | 均值差 | 95% CI | 正/负/平 source |
|---|---:|---:|---:|
| STRUCTURE_SUPPORT − STRUCTURE_DUP_CONTROL（主要） | +0.0556 | [-0.0238, 0.1429] | 6 / 3 / 5 |
| STRUCTURE_SUPPORT − STRUCTURE_ONLY | +0.0635 | [-0.0079, 0.1429] | 6 / 3 / 5 |
| SUPPORT_ONLY − STRUCTURE_ONLY | +0.0317 | [-0.1548, 0.2341] | 5 / 7 / 2 |

主要比较的三个 seed source-macro 平均差分别为 +0.0397、+0.0516、+0.0238，方向均为正，但 source 方向并不一致且区间跨 0。SUPPORT_ONLY 的点估计较高不能解释为物理伪造规律；其 pooled 指标和固定阈值召回也没有同步改善。

## 结论边界

1. 结构加支撑相对重复输入和结构基线都有正的点估计增益，但尚未建立稳定的跨 source 增量证据。
2. Q 确实存在有限变化，visibility 与 geometry validity 高度相关但不是完全相同；这支持把本轮视为观测支撑诊断，而不是独立物理量验证。
3. 未形成结构的组、无有效 unit 的区域和共同 UV/XYZ 支撑筛选造成 survivor bias；结果不支持空间定位、真实遮挡判断、因果时序或 sealed-test 泛化结论。
4. 本轮最多支持保留该有限对照供后续表示核查；不自动扩展模型、窗口或前端。若要继续，应先处理观测盲区与表示/支撑解释，而不是把单次点估计当成稳定收益。

## 产物与验收

产物目录：

`/root/autodl-tmp/data/sparse_3d_forgery_detection/derived/v7_activityforensics_observation_support_pilot_v1/`

主要文件：`protocol.json`、`inputs/input_manifest.json`、`inputs/feature_manifest.json`、`support/summary.json`、`support/coverage.csv`、`support/q_coverage.csv`、`models/fold_models.json`、`scores/`、`evaluation/summary.json`、`evaluation/per_source_metrics.csv`、`evaluation/per_seed_metrics.csv`、`evaluation/paired_comparisons.csv`、`report.md`、`final_status.json`。

smoke 通过，定向测试 10 项通过，正式模型 12/12，最终状态 `COMPLETE`，无残留进程。阶段耗时见 `final_status.json`；本轮代码未提交数据、NPZ、模型或视频。
