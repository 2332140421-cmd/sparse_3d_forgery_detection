# V7 两密度匹配训练 pilot

状态：`COMPLETE`（2026-09-18）。本实验只比较同一训练/验证协议下的
17×17（289 查询，`R17`）和嵌套 33×33（1089 查询，`R33`）。验证集合是
反复开发使用的集合，不是 sealed test。

## 冻结协议

- 输入来自 `v7_activityforensics_source128_extension_v1` 的合法 R 缓存；
  按 `sha256("v7-density-train-20260909:" + source_id)` 从有效训练池固定
  32 个 source，保留既有 16 个 validation source。04LAX 不在本 pilot 中。
- 每个 source 使用既有 MANIP50 的 real/fake 和 `b=0,0.5,1.0` 三个窗口；
  计划为 48 source、96 父片段、288 窗口。标签只使用既有 real=0、合格
  fake=1 规则。
- R17 复用身份匹配的 289 点 R 轨迹；R33 以 R17 坐标轴插入中点形成精确
  嵌套网格并重新查询。每个父片段共用 depth、内参和 pose，仍使用既有
  分组、五时刻支撑、S/Q 和无零填充规则。R33 可能改变原 289 点的轨迹，
  因而本实验不是“固定轨迹只增加点数”的因果对照。
- 两个条件为 `D17_TRAIN` 和 `D33_TRAIN`，均使用现有
  `STRUCTURE_SUPPORT`、局部均值聚合、source/class 加权 BCE、Adam
  (`lr=1e-3`, `weight_decay=1e-4`)、200 epochs 和 seeds
  `20260909/20260910/20260911`。标准化只用各自密度的共同训练窗口拟合。
  主指标先平均三个 seed 的窗口 logit，再计算 source-macro AUROC；CI 为
  source bootstrap（10,000 次，seed=20260909）。阈值固定为 logit≥0。

## 实际完成度与覆盖

- 前端：96/96 父片段、288/288 窗口；正式模型：6/6；smoke 输入→训练→
  保存→重载→读出：`PASS`。
- 两密度的有效窗口分别为 R17=276、R33=279，有效 unit 分别为 9,793 和
  27,776。逐窗口交集为 both=276、R17-only=0、R33-only=3、neither=9。
  3 个 R33-only 和 9 个 neither 窗口的具体 ID 与 `NO_VALID_FIVE_TIME_UNIT`
  原因见数据目录的 `evaluation/coverage_summary.csv` 与 `report.md`；缺失
  没有被填成零分。
- 为匹配训练/评价，实际共同训练为 190 窗口（96 real/94 fake），主验证
  为 83 窗口（42 real/41 fake、14 个双类别 source）。

## 主结果（seed-mean logit）

| 条件 | split | 窗口 (real/fake) | source-macro AUROC | pooled AUROC | AP | P | R | F1 | ACC |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| D17_TRAIN/R17 | train | 190 (96/94) | 0.690972 | 0.634973 | 0.672048 | 0.625000 | 0.478723 | 0.542169 | 0.600000 |
| D33_TRAIN/R33 | train | 190 (96/94) | 0.659722 | 0.627992 | 0.680682 | 0.630769 | 0.436170 | 0.515723 | 0.594737 |
| D17_TRAIN/R17 | validation | 83 (42/41) | **0.698413** | **0.665505** | **0.691513** | **0.656250** | 0.512195 | **0.575342** | **0.626506** |
| D33_TRAIN/R33 | validation | 83 (42/41) | 0.587302 | 0.604530 | 0.645299 | 0.555556 | 0.365854 | 0.441176 | 0.542169 |

主配对 `D33/D33 − D17/D17`：source-macro AUROC 差 **−0.111111**，
95% CI **[−0.293651, 0.039683]**；14 个 source 中 3 个上升、8 个持平、
3 个下降。三个 seed 的差分别为 −0.103175、−0.158730、−0.111111，方向一致
为下降，但区间跨 0。逐 source 与逐 seed 结果在
`evaluation/paired_comparisons.csv`、`evaluation/per_source_metrics.csv`。

## 2×2 交叉读出（辅助）

四格共用上述 83 个验证窗口和各训练密度自己的标准化；它只描述分布适应：

| 训练密度 | 推理密度 | source-macro AUROC | pooled AUROC | AP | F1 |
|---|---|---:|---:|---:|---:|
| R17 | R17 | 0.698413 | 0.665505 | 0.691513 | 0.575342 |
| R17 | R33 | 0.579365 | 0.615563 | 0.644923 | 0.454545 |
| R33 | R17 | 0.690476 | 0.663182 | 0.707990 | 0.637363 |
| R33 | R33 | 0.587302 | 0.604530 | 0.645299 | 0.441176 |

R17→R33 下跌、R33→R17 接近 R17 对角线，说明密度/跟踪改变了输入分布；
不能把它分解为单独的“新增查询点”效果。

## 成本与边界

`final_status.json` 记录本次进程墙钟 13,358.12 s（约 3 h 42 min）；6 个
正式模型均完成 200 epochs，模型拟合计时合计约 3.62 s，设备为 CUDA。
详细结果、逐窗口 logit、cross-density 逐 seed 读出和覆盖表位于：

`/root/autodl-tmp/data/sparse_3d_forgery_detection/derived/v7_activityforensics_density_training_pilot_v1/`

报告文件为该目录的 `report.md`；主要小表为
`evaluation/metrics.csv`、`evaluation/per_source_metrics.csv`、
`evaluation/paired_comparisons.csv`、`evaluation/cross_density_metrics.csv`
和 `evaluation/coverage_summary.csv`。ParticleSequence、特征、模型和视频
均只保留在数据盘，没有提交 Git。

结论仅是本开发验证集上的 pilot 结果：R33 的匹配训练没有显示相对 R17 的稳定
检测增益，虽然多得到 3 个可评分窗口。不能据此宣称空间定位、物理机制、未知
生成器泛化或完整视频检测；本轮不继续更高密度实验。
