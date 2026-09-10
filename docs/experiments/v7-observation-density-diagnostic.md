# V7 observation density and local grouping diagnostic v1

本轮是固定前端的测量诊断，不是检测器训练或指标改进实验。输出位于
`/root/autodl-tmp/data/sparse_3d_forgery_detection/derived/v7_activityforensics_observation_density_diagnostic_v1/`，不进入 Git。

## 直接回答

- **更多点是否真正增加了可靠局部关系？** 在固定的 16 个窗口中，289 点把三时刻共同成员的平均数从 37.1 增至 178.9，说明可测支撑数量增加；但最大 component 平均也从 44.0 增至 211.8，内部 pair 从 1,178.6 增至 27,299.8，直接边占 component 内部 pair 的比例反而由 0.576 降到 0.529。因此只能说“观测支撑变多”，不能据此宣称可靠、局部或更适合检测的关系已经增加。
- **当前大 component 主要暴露了什么？** 当前 component 是固定直接边规则后的连通分量，而模型统计的是分量内部所有 pair。289 点中大量新增 pair 是连通集合内部的关系，不等于新增的直接边或语义部位；这暴露了连通性扩张和局部性之间的差异。全量历史 192 窗口重建与冻结成员集合 192/192 一致，故不是本轮重建错误。
- **哪些失踪仍无法由现有证据解释？** 现有 `ParticleSequence` 只有阈值后的 visibility 和 geometry-validity：不能把真实遮挡与可见表面跟踪失败分开，也不能把 depth 无效与 pose 无效分开。`tracking_and_validity.csv` 对这些状态保留 `UNKNOWN`，没有把点消失当作 fake 证据。

## 固定协议与输入

- Part A：复用既有 192 个窗口和 64 点 NPZ，仅重建历史半段 component、直接图边、内部 pair、triplet 共同成员及覆盖状态；不运行模型。
- Part B：按 `source_id` 排序选取 01KML、04LAX、0AGCS、0BX9N；每个 source 最早 MANIP 和 CTRL，各取 real/fake，共 16 个窗口。当前派生目录仍没有可核验的人工 ROI，因此本次收尾状态为 `ROI_ANNOTATION_READY`，不把便利样本或模型响应当作局部真值。
- 64 点为现有 `grid_size=8`；289 点为相同 `process_size=256` 下 `grid_size=17`。两者使用 `linspace(0,255,grid_size+2)[1:-1]`，64 点在 289 网格的奇数行/列位置，映射见 `query_mapping.csv`。
- 两密度在同一窗口中共享一次 depth、首帧固定焦距内参和 Open3D RGB-D pose 结果；tracker 只改变 query grid。provider、checkpoint、query chunk=64、分辨率 256 和 component 阈值均不变。
- 289 点的嵌套 64 点与本轮匹配 64 点在 16 个窗口均 visibility 一致率 1.0、UV 中位误差 0、XYZ 中位误差 0；历史 64 与本轮匹配 64 的 UV/visibility 也一致，但 01KML CTRL real 的 XYZ 中位差异为 56.68 m，故历史 64→289 不是纯密度比较，主比较使用本轮匹配 64→289。
- 32 个密度臂的记录耗时合计约 375.88 s，单臂范围约 5.21–18.70 s；GPU 峰值显存约 4.21 GB（4090 D，外部前端环境）。

## 结果摘要

| arm | 几何有效率均值 | 有效 triplet 窗口数 | 三时刻共同成员均值 | 最大 component 均值 | 直接边/内部 pair |
|---|---:|---:|---:|---:|---:|
| matched 64 | 0.759 | 14/16 | 37.1 | 44.0 | 0.576 |
| dense 289 | 0.760 | 14/16 | 178.9 | 211.8 | 0.529 |

按 source 的共同成员均值（64 → 289）为：01KML 14.3 → 54.3、04LAX 53.0 → 239.8、0AGCS 46.0 → 228.3、0BX9N 35.3 → 193.3。01KML 两个 CTRL 窗口均因后半段 pose 链失效而没有有效 triplet；这只能说明当前几何支撑不足，不能区分 pose、深度或跟踪原因。

Part A 的 192 窗口中，166 个有有效 triplet，26 个为 `NO_VALID_TRIPLET`；冻结 component 成员全部复现。间接连通示例在 `indirect_connectivity_examples.json` 中按固定顺序保留，不能据此自动判定 component 错误。

## 产物与审查页面

- `protocol.json`：实际查询、几何、component 和缺失状态契约。
- `window_manifest.csv`：16 个固定窗口及选择理由。
- `query_mapping.csv`：64→289 精确嵌套映射。
- `tracking_and_validity.csv`：逐点逐帧的可见、UV、几何、component/triplet 支撑状态；无法区分的原因为 `UNKNOWN`。
- `component_diagnostics.csv`：192 历史窗口及 32 个新密度结果，区分图边和内部 pair。
- `density_comparison.csv`：两密度有效率、共同成员、component 和匹配误差。
- `review/index.html`：无外部 CDN 的静态页面，可在派生目录上运行 `python3 -m http.server 8765` 后访问。页面可在同一 source/role/PTS 下切换 64/289、component 和精确帧；黄色点表示未归入 component，紫/青色只表示密度网格，白圈表示 selected triplet common member。页面引用既有短片副本，不复制完整视频。
- `review/index.html` 现在提供最小的原始像素矩形圈选、JSON 导入/导出和 64/289 同帧切换入口；人工导出 JSON 并放入 `roi_review/annotations.json` 后，离线程序才会重新校验 source、frame、PTS、尺寸和 ROI 边界。
- `roi_review/`：`roi_validation.csv`、`per_frame_coverage.csv`、`per_component_triplet_coverage.csv` 和 `density_comparison.csv` 只在人工输入通过校验后填充实际计数；当前没有有效人工记录，计数表保持表头并不输出虚构结果。单帧 ROI 不自动传播到其他 triplet 时刻，`TEMPORAL_ROI_PENDING` 只表示仍需人工逐帧确认。

服务器本轮未实际执行浏览器视觉验收，只完成静态 HTTP/JSON 路径检查；因此不能把页面存在等同于已完成浏览器验收。

## 科学边界与下一步

本轮没有人工空间 ROI，不能支持像素级定位、局部篡改真值或“某 component 就是编辑部位”的主张。64 个初始查询也不等于每帧 64 个有效三维点；geometry-validity 不代表测量精确；component 连边阈值不是伪造判定阈值；MANIP 时间段不表示每个局部都有可见内部形变。未来有效 ROI 的 64/289 差值只能称为局部测量支撑差异，不是准确率、异常分数或检测概率。

唯一优先建议：若后续仍需要局部检测验证，先以本轮 matched 64/289 结果中共同成员与 component 局部性为依据做有限的局部测量组织/追踪输入审查；不要把缺失率直接加入分类器，也不要因本轮支撑增加就跳过匹配检测验证。

本轮状态：`OBSERVATION_DENSITY_DIAGNOSTIC_COMPLETE`。前端产物尚不具备正式因果预测训练资格，未训练新模型。
