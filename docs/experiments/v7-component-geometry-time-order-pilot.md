# V7 component geometry time-order pilot

入口：

```bash
research_tools/v7/component_geometry_time_order_pilot/run_all.sh \
  --output-root /root/autodl-tmp/data/sparse_3d_forgery_detection/derived/v7_activityforensics_component_geometry_time_order_pilot_v1 \
  --device cuda --resume
```

本 pilot 只读取已经完成的 `v7_activityforensics_component_geometry_pilot_v1`
匹配特征缓存，不打开视频，也不调用 tracking、depth、pose 或 segmentation。
三条件共享 801 个窗口、component/common unit、历史固定边、Q、标签、source
划分、unit 到 window 的等权平均、训练规则和标准化原则。

| 条件 | 时间处理 |
|---|---|
| `G3D_SET` | 与已有 raw 3-D edge encoder 相同，五个逐时刻状态均值；四个实际 PTS 间隔作为旁路。 |
| `G3D_ORDERED` | 每时刻空间状态与 `tau=t-t0` 拼接，按真实递增 PTS 槽位固定拼接，经 `Linear(45,8)-ReLU` 后接相同 head。 |
| `G3D_ORDERED_SHUFFLED` | 与有序条件同架构；窗口级冻结非恒等置换同步重排 edge、mask、Q，时间槽和 intervals 保持固定。 |

主比较是 `G3D_ORDERED−G3D_SET`，次比较是
`G3D_ORDERED−G3D_ORDERED_SHUFFLED`。两者均按 source 配对 bootstrap 10,000
次；三个 seed 先平均窗口 logit，再计算主指标。该实验仍是开发集 pilot，不能
证明物理机制、固定 pair 的跨时刻对应、像素定位或未知 source 泛化。

完整结果位于数据目录的 `report.md`、`evaluation/summary.json` 和
`evaluation/paired_comparisons.csv`。模型、checkpoint 和大数组不进入 Git。
