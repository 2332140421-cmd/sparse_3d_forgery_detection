# V7 R 固定点对距离轨迹匹配对照 pilot

本实验使用已经完成的 periodic re-query R 序列和五时刻支撑缓存，不重跑 tracking、depth、pose、segmentation 或查询。它是小规模开发性机制验证，不修改正式 `src` 检测链。

## 冻结输入

每个有效 local unit 保留原五个实际 PTS、共同成员、canonical pair 和历史尺度。由 R 序列复算每条无向关系的五时刻历史尺度归一化距离，得到 `[pair_count, 5]`，并附加四个相邻真实秒间隔。复算的 `S(t)=[mean,std,p25,p75]` 必须与既有缓存在 `1e-5` 内一致。

主评价集合由当前 R 有效且标签合格、并且同一 source 同时有 real/fake 的所有窗口生成，不套用旧 76/79 分母；原 14-source/76-window 集合作为次要复现。

## 条件

- `SUMMARY_SET`：复用已有 `R_SET_ALL_VALID_TRAIN` 三 seed OOF 平均 logit，仅作实用基准。
- `PAIR_SEQ`：原始固定 pair 五时刻距离轨迹。
- `PAIR_ID_SHUFFLE`：首个距离列保持，后四列在每个 local unit 内分别按固定标签无关哈希排列 pair 行；逐时刻距离多重集合和旧 `S(t)` 保持不变。
- `PAIR_TIME_SHUFFLE`：每个窗口使用一个固定非恒等五时刻排列，所有 local unit 同步；距离值置于原递增时间位置，间隔输入保持原值。

三种新条件使用相同的 `9 -> 16 -> 8 -> 1` relation MLP。关系 logit 先在 unit 内平均，再在 window 内平均，loss 只在 window 级按 source/class 加权。距离标准化仅由训练 fold 拟合，三种新条件共享；实际秒间隔不标准化。

## 解释边界

`PAIR_SEQ` 超过 ID shuffle 只能说明本 pilot 的固定关系轨迹携带了额外可用信息，不能证明跨查询物理对应、空间伪造真值、完整拓扑建模、未知生成器泛化或正式检测收益。五时刻支持是离线 survivor filter，窗口重叠和开发 source 数量有限。

数值报告和模型记录保存在数据盘：
`/root/autodl-tmp/data/sparse_3d_forgery_detection/derived/v7_activityforensics_pair_trajectory_pilot_v1/`。

## 实际执行结果

本轮已完成 `prepare → smoke → train → evaluate → report`。短验证为一个
held-out source、一个 seed、三个新条件各 1 epoch；正式阶段完成
`15 held-out source × 3 condition × 3 seed = 135` 个模型，每个 200 epoch，
没有失败模型。关系重建为 85 个有效窗口、2401 个 local unit、48859 条
canonical pair；主评价为 14 个双角色 source、82 个窗口（real/fake=41/41）。

主评价（先对三个 seed 的 OOF logit 求平均）如下：

| condition | source-macro AUROC | pooled AUROC | AP |
|---|---:|---:|---:|
| SUMMARY_SET | 0.8333 | 0.6568 | 0.6808 |
| PAIR_SEQ | 0.6667 | 0.5598 | 0.6339 |
| PAIR_ID_SHUFFLE | 0.6587 | 0.6086 | 0.6466 |
| PAIR_TIME_SHUFFLE | 0.6786 | 0.6050 | 0.6977 |

`PAIR_SEQ − PAIR_ID_SHUFFLE` 的 source-macro AUROC 差为 0.0079，
source bootstrap 95% CI 为 [-0.1190, 0.1587]；
`PAIR_SEQ − PAIR_TIME_SHUFFLE` 为 -0.0119，CI 为 [-0.1627, 0.1548]；
`PAIR_SEQ − SUMMARY_SET` 为 -0.1667，CI 为 [-0.3016, -0.0476]。
因此本 pilot 没有建立固定关系对应或正确时间顺序的稳定增益，
也没有显示新关系模型超过实用 `SUMMARY_SET` 基准。结果、逐 source/seed
指标和训练窗口清单均在上述数据盘目录的 `report.md`、`evaluation/` 和
`models/training_window_manifest.csv` 中；数据盘产物不进入 Git。

复现入口（使用已准备产物时）：

```bash
PYTHONPATH=src .venv/bin/python research_tools/v7/pair_trajectory_probe/runner.py all \
  --device cuda --budget-s 1800 --resume
```
