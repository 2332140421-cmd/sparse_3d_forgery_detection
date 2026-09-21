# V8 身份修复与冻结协议重训

本记录对应独立产物目录 `v8_full_coverage_observation_field_identityfix_v1`，不覆盖原始 V8 结果。原始结果标记为 `PRE_FIX_TRAIN_INPUT_IDENTITY_MISMATCH`：其训练输入中存在 `SL0061` 的 `pair_id` 碰撞，不能继续作为干净训练结果。

## 状态与修复

- 运行：`COMPLETE`，801 窗口（718 train、83 validation），前端 801/801，派生特征 801/801，正式模型 6/6（FULL/RGB_2D × 3 seed，100 epoch）。
- 解析键改为 `(pair_id, source_id)` 后再按 `role` 取视频；同复合键冲突显式报错，缺失 source 不回退到 `pair_id`。源码证据：`pipeline.py:100-153,156-187`。
- 801 窗口核对发现且仅发现 5 个身份变化：`5H1P1` 的 2 fake + 3 real 窗口。原来实际读到 `5QJNP` 的路径；修复后按正确 `5H1P1` real/fake 视频重提。796 个身份一致窗口复用只读前端缓存。
- 修复前端缓存命中同时核对 window/source/role、视频 SHA 和源帧索引（`frontend.py:172-187`）；新运行的复用 metadata 指向独立目录。
- 5 个窗口之外不重跑前端；但全部 801 个派生特征、分组尺度、关联摘要和两个条件的标准化均从修复后前端重新生成，避免混用训练统计。新分组统计见产物 `grouping_config.json`。

## 修复后/修复前结果

主评价固定为 83 窗口、14 validation source，三 seed 先平均 logit，阈值 `logit >= 0`。修复前数值仅作受污染历史对照。

| 条件/指标（validation） | 修复前 FULL | 修复后 FULL | 修复前 RGB_2D | 修复后 RGB_2D |
|---|---:|---:|---:|---:|
| source-macro AUROC | 0.817460 | 0.785714 | 0.753968 | 0.761905 |
| pooled AUROC | 0.698026 | 0.731127 | 0.685830 | 0.656794 |
| AP | 0.729294 | 0.760273 | 0.723132 | 0.677822 |
| Precision | 0.653846 | 0.677419 | 0.700000 | 0.620690 |
| Recall | 0.414634 | 0.512195 | 0.512195 | 0.439024 |
| F1 | 0.507463 | 0.583333 | 0.591549 | 0.514286 |
| Accuracy | 0.602410 | 0.638554 | 0.650602 | 0.590361 |
| TN/FP/FN/TP | 33/9/24/17 | 32/10/20/21 | 33/9/20/21 | 31/11/23/18 |

修复后 `FULL − RGB_2D` 的 source-macro 差为 `0.023810`，source bootstrap（10000 次、seed=20260909）95% CI `[-0.190476, 0.230159]`，14 个 source 中 4 上升、8 持平、2 下降。seed 级和逐 source 表在 `evaluation/metrics.csv`、`evaluation/per_source_metrics.csv` 与 `evaluation/per_source_differences.csv`。

训练集三 seed 平均 logit 的保存分数组合复算为两条件均 pooled/source-macro AUROC=1、AP=1；这只是拟合表现，不是泛化证据。六个模型的 loss、epoch、参数量和耗时在 `models/fold_models.json`。

## 帧内、全帧与时间路径审计

实际路径为：

`原视频 → 16 个实际 PTS 帧、256×256 letterbox → MoGe cell XYZ/RGB + 32×32 CoTracker 查询状态 → 16×16 基础格 → 4 邻接动态 component/最多4格合并及多对多前驱 → x_full(16,256,77) 或 x_rgb2d(16,256,71) → cell encoder → 一层固定网格 edge MLP → FULL component mean/residual（RGB_2D 不做）→ 每时刻 GRUCell → cell logits → 按 cell_area 对 T×C 做 logsumexp → 窗口 logit`。

源码证据：

- 前端与实际 PTS、几何/跟踪状态：`frontend.py:189-310`。
- 训练拟合的空间/运动尺度、动态 component 与多对多摘要：`pipeline.py:305-356,390-476`。
- 两条件输入形状：`pipeline.py:457-475`。
- 固定网格空间消息、FULL component pooling：`model.py:56-75`。
- 16 时刻、实际 `delta_t`、最多4个前驱、GRU 与最终 area-weighted logsumexp：`model.py:77-138`。

全量轻量统计（见 `structure_audit.json`）：非 padding 像素覆盖率中位数 0.5625（受 letterbox 纵横比影响）；本次所有保存 cell 的 geometry-valid 面积比例为 1.0；每帧 component 数中位数 152、组大小中位数 1、上限4。797/801 窗口出现至少一次相邻时刻成员分区变化（0.995）；相邻时刻分区比较不是简单编号变化。t>0 的多对多关联有效覆盖 train 0.99935、validation 0.99912；关联和无历史状态仍是观测属性，不是伪造真值。

该模型有帧内固定网格的一层消息与当前 component 内 pooling，但没有跨全帧 component 图的显式消息传播。时间顺序通过逐时刻 GRU、真实 `delta_t` 和前一时刻多对多关联进入；监督和正式评价仍是窗口级，没有经验证的 frame-level 检测输出。逐格可视化不能替代帧级标签或像素真值定位。

FULL−RGB_2D 同时改变 XYZ、geometry-valid、动态 component 和观测状态输入，因此差值不能归因为纯三维规律；本修复结果也不能解释为方法创新增益，因为修复前训练输入已污染。

## 复现与边界

复现需要 V8 外部 MoGe/CoTracker 权重和数据根，不能仅靠 clone：

```bash
cd /root/autodl-tmp/projects/sparse_3d_forgery_detection
PYTHONPATH=/root/autodl-tmp/data/sparse_3d_forgery_detection/external/v8_full_coverage/MoGe:/root/autodl-tmp/data/sparse_3d_forgery_detection/external/v8_full_coverage/python_pkgs:/root/.cache/torch/hub/facebookresearch_co-tracker_main:$PWD \
PYTHONUNBUFFERED=1 .venv/bin/python -u scripts/run_v8_identityfix.py \
  --old-output /root/autodl-tmp/data/sparse_3d_forgery_detection/derived/v8_full_coverage_observation_field_v1 \
  --output /root/autodl-tmp/data/sparse_3d_forgery_detection/derived/v8_full_coverage_observation_field_identityfix_v1 \
  --identity-root /root/autodl-tmp/data/sparse_3d_forgery_detection/derived/v7_activityforensics_component_geometry_pilot_v1 \
  --source-root /root/autodl-tmp/data/sparse_3d_forgery_detection/derived/v7_activityforensics_source128_extension_v1 \
  --moge-checkpoint /root/autodl-tmp/data/sparse_3d_forgery_detection/external/v8_full_coverage/checkpoints/moge-2-vits-normal-model.pt \
  --tracker-checkpoint /root/.cache/torch/hub/checkpoints/scaled_online.pth --device cuda
```

不提交视频、前端 NPZ、特征、模型或权重；它们保留在数据目录。当前审计没有启动新进程，工作区只包含本次 V8 修复入口、审计脚本、测试与本说明文档。
