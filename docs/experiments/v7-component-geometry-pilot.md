# V7 component 可学习几何表示最小验证

本实验的实现入口是 `research_tools/v7/component_geometry_pilot/runner.py`，运行：

```bash
cd /root/autodl-tmp/projects/sparse_3d_forgery_detection
research_tools/v7/component_geometry_pilot/run_all.sh
```

它只读取已完成的 R `ParticleSequence`、component/support 和 S/Q 特征缓存，不打开视频，也不调用 tracking、depth、pose 或 segmentation。默认产物在数据目录 `derived/v7_activityforensics_component_geometry_pilot_v1/`。

三臂固定为：`SUMMARY_SET`（匹配的 S+Q 四维摘要）、`GEOMETRY_3D`（可学习局部三维关系）和 `GEOMETRY_2D`（相同三维分组/固定边上的二维输入对照）。三臂使用相同窗口、标签、五个 PTS、共同 unit 支撑、source-disjoint 划分和窗口级 source/class weighted BCE；`GEOMETRY_2D` 不是独立的纯二维前端。

## 实现审计边界

| 项目 | 当前已有/可复用 | 本轮新增 | 尚不具备 |
|---|---|---|---|
| 观测与 component | R `ParticleSequence` 的 289 track、UV/XYZ、visibility、geometry-validity、历史组和共同成员 | 无 | 全画面稠密逐帧点集 |
| 表示 | observation-support 的 S/Q；`2d3d_information_pilot` 的 UV/XYZ 距离摘要（相关但不等价） | 固定历史邻边的可学习 3D/2D 局部关系编码 | 已验证的物理异常模型、旋转不变性 |
| 训练/评价 | 既有 source-disjoint 窗口清单、标签、权重和 bootstrap 规则 | A/B/C 三条件各 3 seed | sealed-test、帧/像素定位真值 |
| 无显式跟踪 | 稀疏前端仍依赖 query cohort/track ID | 只读可行性说明 | 无身份 dense anchor、跨帧尺度合同 |

局部边在保存的历史参照帧按距离、slot-ID 稳定排序，每个成员最多八个邻居；边输入为 `[x_p-c_i, x_q-x_p, ||x_q-x_p||]`（二维使用对应 UV），每个目标时刻按 visibility/geometry_validity 和有限性生成 mask。无效或 padding 边只通过 mask 排除，不以零坐标替代观测；不逐帧缩放或刚体配准。中心化移除整体平移，方向仍保留，未宣称旋转不变。

正式训练沿用三 seed、200 epochs、Adam、`lr=1e-3`、`weight_decay=1e-4` 及固定 `logit>=0`。主比较是 `GEOMETRY_3D−SUMMARY_SET` 的 source-macro AUROC；次比较为 `GEOMETRY_3D−GEOMETRY_2D`，source bootstrap 10,000 次。验证集是开发集，不是 sealed-test，也不支持帧/像素定位因果结论。

无显式跟踪可行性只读审计写入产物 `no_tracking_feasibility.md`。当前缓存是稀疏跟踪槽位，不是覆盖整幅画面的逐帧稠密深度/锚点合同；模型不读取 ID 不等于全系统无跟踪。未来接口需新增 dense/regular depth+intrinsics+pose、质量和尺度 provenance、身份无关的局部时空邻域及窗口到帧标签映射。
