# V7 YOLO26m-depth 独立测量对照

状态：`YOLO26_DEPTH_COMPARISON_COMPLETE`。这是深度测量行为对照，不是新检测器训练，也没有深度真值。

产物位于：

`/root/autodl-tmp/data/sparse_3d_forgery_detection/derived/v7_activityforensics_yolo26_depth_comparison_v1/`

## 固定设置

- 使用官方 [Ultralytics depth 文档](https://docs.ultralytics.com/tasks/depth/) 所述的 `yolo26m-depth`，`imgsz=768`，`ultralytics==8.4.146`。
- 权重：`external/yolo26_depth/yolo26m-depth.pt`，SHA-256：`c0ff310571f0f159436e95a2a4fa3f8c40e1fbf36fac66a211347bd189990165`。
- 固定 4 个 source、16 个已有密度诊断窗口；复用保存的 289 点 UV/visibility，不重新 tracking。
- 两种深度共用同一窗口 RGB、Depth Pro-derived 首帧焦距内参和 Open3D 位姿；历史粒子产物未持久化位姿矩阵，因此位姿由既有 helper 每窗口确定性重算一次，再同时用于两种深度。没有逐帧 scale fitting、未来帧拟合或 fake 校准。
- `result.depth.data` 直接作为正深度解释；输出通过原始 RGB 尺寸检查后采样。短片色标在同一视频内对两种方法一次性固定，不逐帧自适应。

## 观察结果

16 个窗口的同查询正深度支持均为 1.0；Depth Pro 与 YOLO26 的平均查询深度中位数分别为约 1.690 m 和 1.596 m。按每帧整图中位深度的 max/min 比，平均分别为 1.215 和 1.471；这只是尺度/时间行为描述，不能解释为精度结论。两者在该固定 component 规则下均有 14/16 个窗口得到有效 triplet 支撑。

Depth Pro 的平均 component 数为 1.31、最大 component 平均 211.8 点；YOLO26 的平均 component 数为 2.38、最大 component 平均 192.5 点。个别窗口仍出现接近全 289 点的大 component，YOLO 的 component 增多也可能是深度噪声碎裂，不能直接当作分组改善。

总耗时约 602.3 s，单窗口平均约 37.1 s；峰值显存约 4.12 GB，设备为 RTX 4090 D（外部隔离环境 torch 2.12.1+cu130）。固定案例短片在 `media/`，`window_results.json` 保存同一 pair 身份的距离曲线和逐窗口诊断。

因此本轮只能说 YOLO26-depth 可运行，并改变了深度尺度、时间波动和 component 行为；没有证据证明它比 Depth Pro 更准确，也没有把它加入 A 的监督检测。该对照不具备正式因果预测训练资格。
