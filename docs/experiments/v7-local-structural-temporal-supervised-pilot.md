# V7 fixed local-relation-supported temporal structural supervised pilot

状态：**LOCAL_TEMPORAL_DISCRIMINATION_NOT_ESTABLISHED**

二阶状态：**SECOND_ORDER_INCREMENT_NOT_ESTABLISHED**

这是一个冻结的、有限开发 population pilot，不是正式训练路线、独立测试、
sealed-test、full-video detector 或跨生成器泛化结论。结果来自：

`/root/autodl-tmp/data/sparse_3d_forgery_detection/derived/v7_activityforensics_local_structural_temporal_supervised_pilot_v4/`

## 1. 问题和输入

本轮问题是：在稳定的开发 population 中，固定局部关系身份后的结构演化是否
超出无序结构状态，并且二阶项是否有增量。复用上一个 pilot 的 16 个 source
pairs、192 个 MANIP/CTRL role windows 和已有 ParticleSequence NPZ/JSON；没有
重跑 frontend、tracking、depth 或 pose。

每个一秒窗口以真实 PTS 划分：前 0.5 秒为历史 `H`，后半秒目标时间为
`0.5, 0.6, 0.7, 0.8, 0.9` 秒。每个目标用最近源帧（容差 0.05 秒、同距取较早、
不重复）匹配；仅连续三点槽形成 triplet，不插值、不外推、不跨缺失拼接。
组件只由 `H` 的既有 `motion_coherent_components`（参数未变）构造。每个
triplet 使用三个时刻共同 geometry-valid 的相同成员及全部无向 pair；历史
尺度是该组件 H 内所有有效 pair-time 距离的 pooled median。

每个时刻状态为固定历史尺度下的
`S(t)=[mean,std,p25,p75]`。输入导数使用真实不等时间间隔；它们是有符号的
结构变化，不称为物理速度/加速度，也不是未来预测。

## 2. 四个固定实验臂

所有臂使用相同的有效 window/component/triplet 支撑和窗口标签，仅数值输入不同：

| 臂 | 输入 |
|---|---|
| `UNORDERED_STATE` (A) | 三个 `S` 分别构成 `[S,S,S]`，共享 MLP 后平均 |
| `ORDERED_FIRST` (B) | `[S_t,v_minus,0]` |
| `ORDERED_SECOND` (C) | `[S_t,v_minus,a_t]` |
| `PERMUTED_SECOND` (D) | 三个完整 `S` 的六个联合排列，按同一时间槽重算导数并平均 |

每臂均为 `12 -> 16 -> 8` ReLU MLP、component mean、window mean 和线性
window head。A 的三状态和 D 的六排列等权分摊原观测权重；标准化只用训练
source 的该臂输入。训练使用 real MANIP=0、fake MANIP=1，CTRL 不参与拟合，
仅作描述性评分；窗口标签没有下放为 component 标签。

LOSO 按 source 隔离，Adam、`lr=1e-3`、`weight_decay=1e-4`、full-batch 200 epochs，
seeds=`20260909,20260910,20260911`。每个 source/class 总权重相等，再在
window/component/triplet/臂内观测层级等权。所有训练拟合排除 held-out source。

## 3. 覆盖和训练

| 项目 | 结果 |
|---|---:|
| 冻结窗口 | 192 |
| 有效局部支撑窗口 | 166 |
| 无效窗口 | 26 |
| 有效 triplet 总数 | 867 |
| 完整双类别 source（主指标） | 14 |
| 主训练 real/fake MANIP 窗口 | 40 / 42 |
| CTRL（未进入训练） | 84 |

无效原因事件记录为 `COMMON_VALID_MEMBERS_LT3=87`、`TARGET_FRAME_MISSING=18`、
`NO_HISTORY_COMPONENT=23`。`0HV07` 的所有窗口均无本协议有效支撑，保留在
覆盖报告中，不填补；`0CG15` 只有 CTRL 支撑，不能形成主 AUROC source。目标
匹配误差最大 0.04738 秒（所有 192 窗口），有效 triplet 的实际间隔中位数
约 0.10010 秒（范围 0.06667--0.10100 秒）。

共保存 180 个 fold/臂/seed 模型记录（15 个有支撑的 held-out source × 4 × 3），
每次严格 200 epochs，无 early stopping 或搜索。固定零通道导致的零方差维度
被记录，未按标签删维度。

## 4. Source-level 主结果

source-level paired bootstrap 重采样完整 source，seed=`20260909`、10,000 次，
四臂使用相同索引。

| 臂 | source 等权 AUROC 均值 | 95% CI |
|---|---:|---:|
| A `UNORDERED_STATE` | 0.531746 | [0.404762, 0.658730] |
| B `ORDERED_FIRST` | 0.452381 | [0.317460, 0.579365] |
| C `ORDERED_SECOND` | 0.531746 | [0.412698, 0.635119] |
| D `PERMUTED_SECOND` | 0.539683 | [0.404762, 0.682540] |

配对增益（同一 source bootstrap）：

| 对比 | 均值 | 95% CI |
|---|---:|---:|
| C-A | -0.001240 | [-0.166667, 0.150794] |
| C-D | -0.009298 | [-0.111111, 0.095238] |
| C-B | +0.078983 | [-0.007937, 0.190476] |

C 的均值 CI 下界没有超过 0.5，且 C-A、C-D 下界均未大于 0，因此按预声明
规则主状态为 `LOCAL_TEMPORAL_DISCRIMINATION_NOT_ESTABLISHED`。C-B 的区间也
跨越 0，故二阶增量为 `SECOND_ORDER_INCREMENT_NOT_ESTABLISHED`。

逐 source 的四臂 AUROC 和三组差值保存在
`evaluation/per_source_metrics.csv`；结果存在明显 source 异质性，例如
`0ET8W` 的 C-A/C-B/C-D 为 `+0.556/+0.667/+0.444`，而 `04LAX` 的 C-A 为
`-0.778`。因此平均值不能解释为所有 source 的一致效应。

三个 seed 的窗口 logit 先逐窗口算术平均后才进行主评价。平均 seed 标准差
（A/B/C/D）约为 `0.112/0.108/0.082/0.061`；这些是开发诊断，不是校准误差。
不同 LOSO fold 的原始 logit 尺度不相同，pooled AUROC 仅作辅助：A 0.497、
B 0.451、C 0.514、D 0.525（fake MANIP vs real MANIP）。

## 5. CTRL 行为和限制

CTRL 没有被当作确定负类。有效窗口上的模型分数中位数（real/fake）分别为：

| 臂 | real CTRL | fake CTRL |
|---|---:|---:|
| A | -0.083 | -0.126 |
| B | -0.009 | -0.035 |
| C | +0.008 | -0.028 |
| D | +0.070 | -0.009 |

它们只是控制窗口描述，不能转成额外监督证据。当前总体来自历史上多次用于开发
的 16 source pairs；没有独立 source 验证，不能声称跨生成器泛化，也不能排除
观测、重建或缺失覆盖差异而非真实伪造结构导致的响应。

## 6. 产物和边界

协议、输入/覆盖 manifest、逐窗口 support（含 component/pair/triplet 身份、实际
源帧索引/时间戳、`geometry_validity` 有效性来源和无效原因）、OOF scores、逐
source 指标、180 个小型模型/标准化记录和 summary 均在上述 derived 目录；没有
复制视频或 NPZ。协议固定在 held-out 分数生成前。

本轮只做局部离线三时刻监督判别；不实现未来预测、NSI、动态融合、关系网络、
full-video、sealed-test 或正式模型路线。旧 B0/NSI 结果仅作历史背景，没有把
本轮新状态冒充旧 B0 的精确重构。
