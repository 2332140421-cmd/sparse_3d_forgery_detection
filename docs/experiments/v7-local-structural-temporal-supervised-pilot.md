# V7 fixed local-relation-supported temporal structural supervised pilot

状态：**LOCAL_TEMPORAL_DISCRIMINATION_NOT_ESTABLISHED**

二阶状态：**SECOND_ORDER_INCREMENT_NOT_ESTABLISHED**

本次是一次严格等协议的实现纠错重跑，不是新方法或新实验臂。旧产物保留在
`v7_activityforensics_local_structural_temporal_supervised_pilot_v4/`；修正后产物在：

`/root/autodl-tmp/data/sparse_3d_forgery_detection/derived/v7_activityforensics_local_structural_temporal_supervised_pilot_weightfix_v1/`

重跑时相对已提交 `b46d4f8ac839b6620c88acaccba0312d0812e973` 的代码/测试差异
SHA-256 为 `57a976f641ef7128010231126894b59aadf66c647c0bd09af0c595fbf9be32c2`；
重跑时工作树尚未提交，因此新 `protocol.json` 的 `git_head` 保留为该父提交，差异
hash 作为本次实现快照记录。

这是冻结的、有限开发 population pilot，不是正式训练路线、独立测试、sealed-test、
full-video detector 或跨生成器泛化结论。

## 1. 冻结问题、输入和四个臂

复用同一份 16 个 source pairs、192 个 MANIP/CTRL role windows 和已有
ParticleSequence NPZ/JSON；没有重跑 frontend、tracking、depth 或 pose。每个一秒窗口
以前 0.5 秒历史 `H` 和后半秒目标槽（`0.5, 0.6, 0.7, 0.8, 0.9` 秒）构造；目标用
最近真实 PTS（容差 0.05 秒、同距取较早、不重复）匹配，仅连续三点槽形成 triplet，
不插值、不外推、不跨缺失拼接。组件只由历史 `motion_coherent_components` 构造，
triplet 使用三时刻共同 geometry-valid 的同一成员和全部无向 pair，历史尺度为组件内
有效 pair-time 距离 pooled median。

每时刻状态仍为固定历史尺度下的 `S(t)=[mean,std,p25,p75]`；导数使用真实不等时间
间隔，是有符号结构变化，不称为物理速度/加速度，也不是未来预测。

| 臂 | 输入 |
|---|---|
| `UNORDERED_STATE` (A) | 三个 `S` 分别组成 `[S,S,S]`，共享 MLP 后平均 |
| `ORDERED_FIRST` (B) | `[S_t,v_minus,0]` |
| `ORDERED_SECOND` (C) | `[S_t,v_minus,a_t]` |
| `PERMUTED_SECOND` (D) | 三个完整 `S` 的六个联合排列，按同一时间槽重算导数并平均 |

所有臂均为 `12 -> 16 -> 8` ReLU MLP、component mean、window mean 和线性 window
head；LOSO 按 source 隔离，Adam，`lr=1e-3`，`weight_decay=1e-4`，full-batch 200
epochs，seeds=`20260909,20260910,20260911`。训练标签仍为 real MANIP=0、fake
MANIP=1，CTRL 不参与拟合，仅作描述性评分。

## 2. 两处实现问题与最小修复

### 2.1 实际损失权重

旧调用链先用 `build_batch(..., observation_weights=True)` 拟合了加权标准化，但实际
训练 batch 使用默认 `observation_weights=False`，所以 `Batch.window_weights` 全为
1；`train_model` 虽然计算了加权 BCE，收到的却是窗口等权。这是协议实现偏差，不是
held-out 泄漏，source-disjoint 划分仍保留。

修正为以同一标准化器构造
`build_batch(training, arm, standardizer=standardizer, observation_weights=True)`，并
保留原 raw batch 的观测权重用于标准化。训练损失仍为：

`sum(BCE_per_window * Batch.window_weights) / sum(Batch.window_weights)`。

`model.py` 抽取的 `weighted_window_bce` 只复用该公式，没有改变模型、聚合、优化器、
epoch 或 seed。新 `evaluation/fold_support.csv` 记录了每个 fold 的 source/class 窗口
数、实际 group weight sums、window weight min/max/mean 和标准化权重来源。所有 scored
fold 的 source/class 组总权重在浮点误差内相等，窗口平均权重为 1；全体 scored fold 的
窗口权重范围为约 `0.974359--2.928571`。旧 v4 的 fold_support 没有保存权重审计，
但其调用路径可确认实际损失为全 1 窗口等权；这不否定其 source-disjoint 隔离。

### 2.2 配对增益点估计

旧 `paired_source_bootstrap` 用 bootstrap replicate 差值的均值作为增益点估计。修正
后先在完整 source 上计算逐 source `C-A`、`C-D`、`C-B`，点估计为这些原始差值的
均值；置信区间仍是同一 `seed=20260909`、`replicates=10000`、同一 source 重采样
索引下的配对差值分布。该修正只改变统计输出，不改变模型或分数。

## 3. 表示和支撑一致性

重跑前从同一 frozen source artifact 重新构造 support，并与 v4 逐窗口核对：

| 项目 | v4 / 修正后 |
|---|---:|
| 冻结窗口 | 192 |
| 有效支撑窗口 | 166 |
| 无效窗口 | 26 |
| 有效 triplet | 867 |
| 完整双类别 source | 14 |
| 主训练 real/fake MANIP 窗口 | 40 / 42 |
| CTRL（不进训练） | 84 |

`manifests/input_manifest.json` SHA-256 为
`74240c69ac90d3bf470e6e9b5de3d2c1ebeb6c5e3029d55c154ac0f225f2bb10`，
`manifests/window_support.json` SHA-256 为
`e2032379590bdc26aab05c11ca3be08bc4b9cf2f83802e528294579ab8f80b86`，与 v4 相同。
component、pair、triplet 身份、状态、导数、时间戳和历史尺度均一致；标准化参数也与
v4 的 180 条模型记录一致。唯一改变来自实际训练损失权重和增益点估计。

## 4. 修正后四臂结果

主评价为 fake MANIP 对 real MANIP 的 source 等权 AUROC；三 seed 窗口 logit 先逐窗口
算术平均。source-level paired bootstrap 使用完整 source，四臂共享重采样索引。

| 臂 | v4 均值 | v4 95% CI | 修正后均值 | 修正后 95% CI |
|---|---:|---:|---:|---:|
| A `UNORDERED_STATE` | 0.531746 | [0.404762, 0.658730] | 0.563492 | [0.444444, 0.682540] |
| B `ORDERED_FIRST` | 0.452381 | [0.317460, 0.579365] | 0.444444 | [0.309524, 0.579365] |
| C `ORDERED_SECOND` | 0.531746 | [0.412698, 0.635119] | 0.531746 | [0.412698, 0.634921] |
| D `PERMUTED_SECOND` | 0.539683 | [0.404762, 0.682540] | 0.531746 | [0.404762, 0.666667] |

配对增益如下。v4 原报告中的点估计也同时列出；其 CI 不因点估计修正而改变。

| 对比 | v4 原报告均值 | v4 正确原始点估计 | v4 CI | 修正后点估计 | 修正后 CI |
|---|---:|---:|---:|---:|---:|
| C-A | -0.001240 | 0.000000 | [-0.166667, 0.150794] | -0.031746 | [-0.206349, 0.126984] |
| C-D | -0.009298 | -0.007937 | [-0.111111, 0.095238] | 0.000000 | [-0.095238, 0.095238] |
| C-B | +0.078983 | +0.079365 | [-0.007937, 0.190476] | +0.087302 | [-0.015873, 0.214286] |

修正后 C 的 CI 下界仍未超过 0.5，且 C-A、C-D 的 CI 下界均不大于 0，因此主状态
仍为 `LOCAL_TEMPORAL_DISCRIMINATION_NOT_ESTABLISHED`；C-B 的 CI 也跨 0，二阶状态
仍为 `SECOND_ORDER_INCREMENT_NOT_ESTABLISHED`。pooled AUROC 仅为辅助（A/B/C/D：
`0.518452/0.455952/0.516071/0.527976`），不作为唯一主结论。

修正后逐 source 差值的范围和符号并不一致：C-A 范围 `-0.778--+0.444`（5 正、5
负），C-D 范围 `-0.333--+0.444`（5 正、4 负），C-B 范围 `-0.222--+0.667`
（6 正、3 负；其余为 0）。例如 `0ET8W` 的 C-A/C-B/C-D 为 `+0.444/+0.667/+0.444`，
`04LAX` 的 C-A 为 `-0.778`，因此改善不能归结为所有 source 的一致效应。三 seed
在 166 个有效窗口上的平均 logit 标准差（A/B/C/D）约为
`0.1473/0.1645/0.1490/0.0961`，这是开发稳定性描述，不是校准误差。

## 5. CTRL 和覆盖限制

CTRL 没有被当作确定负类。修正后有效窗口模型分数中位数（real/fake）为：

| 臂 | real CTRL | fake CTRL |
|---|---:|---:|
| A | -0.144 | -0.199 |
| B | -0.070 | -0.132 |
| C | -0.075 | -0.081 |
| D | -0.012 | -0.092 |

控制窗口只作描述，不能转成额外监督证据。`0HV07` 全部无本协议有效支撑，保留在
覆盖报告中；`0CG15` 只有 CTRL 支撑，不能形成主 AUROC source。无效原因仍为
`COMMON_VALID_MEMBERS_LT3=87`、`TARGET_FRAME_MISSING=18`、`NO_HISTORY_COMPONENT=23`。
没有独立 source 验证，不能声称跨生成器泛化，也不能排除观测、重建或缺失覆盖差异。

## 6. v1--v4 产物演变审计

四个目录均存在 `protocol.json`、input/support/coverage manifest、OOF scores、逐
source metrics、fold models 和 summary；均有 166 个 OOF 窗口、180 条模型记录和完整
四臂结果。因此按产物证据它们都是“已运行有结果”，不是运行失败。是否在运行前查看
过某版分数，当前文件不可核实。

| 版本 | 可核查执行证据 | 内容差异及替代原因 | source/window、模型和预算 | 是否结果驱动 |
|---|---|---|---|---|
| v1 | `run_summary.json`、OOF、models、summary | 初始支撑记录；缺少 `NO_HISTORY_COMPONENT` 细分 | 192/166/867/14，四臂、180 记录均同 | 未见证据；操作者是否查看不可核实 |
| v2 | 同上，summary 与 v1 结果相同 | support 明确记录 23 个 `NO_HISTORY_COMPONENT`，属于支撑原因审计修正 | 与 v1 相同 | 未见证据；不能证明不存在结果影响 |
| v3 | 同上，OOF 与逐 source 文件 hash 与 v2 相同 | support 增加 `history_array_indices`/`evaluation_array_indices`，并区分源帧索引与数组索引 | 与 v2 相同 | 未见证据；不能证明不存在结果影响 |
| v4 | 同上，OOF 与逐 source 文件 hash 仍相同 | component/triplet 增加 `validity_source` provenance 字段 | 与 v3 相同 | 未见证据；不能证明不存在结果影响 |

四版 `protocol.json` 的 `git_head` 均为 `ab2e19ab0ba1157ca83ff55a795618ca026c54ed`；
当前 Git 历史没有保存 v1--v4 每次运行的独立代码快照或日志。由文件内容可确认这些
变更是实现/支撑元数据演变，而不是 population、表示、模型或训练预算改变；但无法完整
证明每次替代决定是否发生在查看 held-out 结果之前，也不能宣称所有调整都与结果无关。

## 7. 产物和边界

修正后产物包含 `protocol.json`、输入/覆盖/support manifest、OOF window scores、
逐 source metrics、fold support 权重审计、180 个小型模型/标准化记录和 summary；没有
复制视频或 NPZ。关键文件 SHA-256：

```text
protocol.json                         c144c3d75c9af942c21a72852058528eb3b38e3e8fff1fb8e2d1cd934b896fdd
manifests/input_manifest.json         74240c69ac90d3bf470e6e9b5de3d2c1ebeb6c5e3029d55c154ac0f225f2bb10
manifests/window_support.json         e2032379590bdc26aab05c11ca3be08bc4b9cf2f83802e528294579ab8f80b86
scores/oof_window_scores.csv          75ec407f87e5a0a144c89b6f672500a0851b670c771f6366d75b83d72ee57749
evaluation/per_source_metrics.csv     f8068e0e9bd3a0e2921f83d33fc162c9b63129df55dd91316f8b8be486dd0658
evaluation/fold_support.csv           01a7df7c38ccf0941ab60445c45fcc6ca1b9b5538c61781360c9d5b8455ec746
evaluation/summary.json               477350855ed53dadf98e1a1461af418c5f81d673a9cab338abedcc996472317b
```

本轮只做局部离线三时刻监督判别纠错；不实现未来预测、NSI、动态融合、关系网络、
full-video、sealed-test 或正式模型路线。修正结果支持的是：在冻结开发 population 上，
按预声明 source/class 损失权重重新训练后，四臂 pilot 的可复核统计结果及其不确定性。
它不支持未知生成器泛化、物理机制证明、正式监督检测完成或四维表示“彻底无信息”。
