# V7 固定重新查询观测与持续旧查询匹配监督对照 pilot

本文件记录代码与冻结协议；运行结果只写入数据盘
`/root/autodl-tmp/data/sparse_3d_forgery_detection/derived/v7_activityforensics_periodic_requery_pilot_v1/`，不进入 Git。

实验从既有 16 个 source、192 个 MANIP/CTRL 窗口中按确定规则选择每个 source 的中间
MANIP 槽位（`anchor_fraction=0.50`）。每个 real/fake 视频从该槽位起点取最多 2 秒，
再固定建立 `b=0.0,0.5,1.0` 三个 1 秒分析窗口。标签在时间表冻结后才映射：real 为负类，
fake 完全落入标注区间并集才为主正类，边界和区间外仅作描述。

O 在父片段 `t0` 初始化 17×17=289 个查询并持续跟踪；三个窗口只切片这批旧查询。
R 在每个窗口起点独立初始化同样的 289 个查询；不同 cohort 不共享 track identity、
历史或导数。b=0 可以复用 O 的数组，但不因此宣称物理点对应。

同一父片段的 O/R 共享一次 Depth Pro、内参和 Open3D RGB-D 位姿计算；UV、visibility、
geometry-validity 分开保存，缺失不插值、不以零填充。每个模式每个窗口都重新应用既有 H
历史分组、历史尺度和五时刻共同成员规则，使用现有 SET_A、RAW_SEQ、MULTI_ORDER_SEQ。

六个模型条件为 O/R × 三个既有表示，采用 source-disjoint LOSO、三个固定 seed、200
epoch Adam 和 fold 内标准化。主比较为 paired common support 上的 `R_SET−O_SET`，其余
差值是预声明次要比较。source bootstrap 只描述这个小型开发 pilot，不是 sealed-test、
空间定位验证或跨生成器泛化结论。

启动脚本：

```bash
cd /root/autodl-tmp/projects/sparse_3d_forgery_detection
research_tools/v7/periodic_requery_probe/run_all.sh --resume
```

查看 `progress.json`、`final_status.json`、`report.md` 和 `evaluation/summary.json`；
独立运行会在前端或训练预算耗尽时保存已有产物并给出明确状态。

## 本次下游恢复修复记录

此前 features 阶段失败有两个可复现的实现原因：无有效支撑时使用
`np.empty(4)` 构造 `intervals_s`，导致未定义的 NaN 进入严格 JSON；同时
周期重查适配器把三时刻局部支持直接交给要求 `[5, 4]` 的既有条件接口，
使 169/192 个 O/R 行被错误标记为 `FEATURE_FAILED`。修复后统一调用既有
五时刻 `build_five_time_unit`，无支撑的 `intervals_s` 保存为 JSON `null`
并保留明确原因；严格序列化仍拒绝有效字段中的 NaN/Inf，原始 NPZ 的
NaN 与有效性 mask 不变。

本次恢复复用全部 96 条 `FRONTEND_COMPLETE` 缓存，生成 192 条 O/R 摘要，
O/R 有效窗口为 79/85，共同有效窗口为 79；训练完成 270 个唯一
condition×held-out-source×seed 键，每个 200 epochs。完整数值结果和成本
记录在数据盘的 `report.md`、`evaluation/summary.json`、
`evaluation/per_source_metrics.csv` 及 `state/*_budget.json` 中。
