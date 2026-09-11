# V7 boundary/pooling 观测对照与已有模型前向复核

状态：`FORWARD_REVIEW_READY`

本轮只补观测对照页面和已有 fold 模型的 train/held-out 前向表现。没有调用
optimizer，没有重跑 tracking、depth、pose、segmentation 或稠密前端，也没有改变
原模型、输入表示、窗口总体或主评价定义。

## 1. 输入、页面和使用方式

输入根目录：

`/root/autodl-tmp/data/sparse_3d_forgery_detection/derived/v7_activityforensics_boundary_pooling_pilot_v1/`

页面：

`review/index.html`

启动本地页面（不需要外部 CDN）：

```bash
cd /root/autodl-tmp/data/sparse_3d_forgery_detection/derived/v7_activityforensics_boundary_pooling_pilot_v1/review
python3 -m http.server 8765 --bind 127.0.0.1
```

然后访问 `http://127.0.0.1:8765/`。索引覆盖冻结的 192 个窗口；默认案例是按 source
排序取前四个 source、各取最早 MANIP/CTRL 的 real/fake，共 16 个。原视频只以外部
数据目录下的轻量 symlink 引用，未复制进仓库；默认案例生成三时刻精确帧截图。其余
窗口在索引中可选，若需要单独物化，可按窗口 ID 运行（参数可重复）：

```bash
PYTHONPATH=/root/autodl-tmp/projects/sparse_3d_forgery_detection/src \
/root/autodl-tmp/projects/sparse_3d_forgery_detection/.venv/bin/python \
-m research_tools.v7.boundary_pooling_probe.review_and_forward \
--root /root/autodl-tmp/data/sparse_3d_forgery_detection/derived/v7_activityforensics_boundary_pooling_pilot_v1 \
--materialize-window '0001_MANIP_25::real'
```

页面两侧的 real/fake window、H parent、B child、triplet 和模型条件均独立选择；切换
选择不会把另一侧编号当成物理对应。画面层明确区分 visible、geometry-valid、所选
triplet-common 和 model-used；H→B 的 pair 统计报告总数、边界内保留数、被边界切断数
及未分配端点。颜色只表示局部组身份，不表示真假。首帧 assignment/mask 只作为首帧
边界参考，后续画的是冻结 track slot 的点，不把首帧 mask 静态覆盖到后续帧。

triplet 的 PTS 按保存的 `timestamps_s` 映射回 `ParticleSequence.frame_indices`，而不是
把局部 `target_slots` 误当作整段视频数组下标。页面同时给出三时刻的 `S(t)`、已保存
的一阶/二阶量、源 PTS、点数和模型实际目标帧。模型局部分数标为 classifier logit
响应，不是校准概率或空间定位真值；没有支撑不补成零分。

本机未发现 Chromium、Chrome 或 Firefox 可执行文件，因此本轮未完成真实浏览器视觉
验收。已完成页面生成后的结构核对：192 个详情、192 个源视频链接、16 个默认案例、
42 张实际截图均可从相对路径读取；需在有浏览器的环境打开上述 URL 完成人工视觉确认。

## 2. 已有模型的 forward-only 复核

复核条件仅为 `H_MEAN_A`、`B_MEAN_A`、`B_MEAN_C`。每个条件使用保存的
`models/fold_models.json` 中的 14 个可用 held-out source × 3 seed，共 42 条
fold/seed 记录；未调用优化器。训练输入和 fold 内标准化参数均直接从保存记录恢复，
训练样本仍是其他 source 的 MANIP real/fake，CTRL 不进入训练或主评价，阈值固定为
`logit >= 0`。

每条 forward 记录的 OOF 复核均有对应保存分数：

| 条件 | fold/seed 行数 | train 窗口均值 | held-out 窗口均值 | 最大 OOF logit 差 |
| --- | ---: | ---: | ---: | ---: |
| H_MEAN_A | 42 | 75.214 | 5.786 | 4.77e-7 |
| B_MEAN_A | 42 | 75.214 | 5.786 | 4.77e-7 |
| B_MEAN_C | 42 | 75.214 | 5.786 | 1.91e-6 |

汇总表中 `pooled` 保留每个 fold/seed 预测（训练窗口因此会跨 fold 重复），
`source_macro` 先在每个 source/window 上平均可用 seed/fold logit，再等权平均 source
指标。held-out source_macro 是与 source-level 主 AUROC 最接近的口径；不能把 pooled
窗口数当作独立 source 数。

### Train / held-out pooled 结果

| 条件 | split | score rows | fake/real | ROC-AUC | AP | F1 | ACC |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| H_MEAN_A | train | 3159 | 1638/1521 | 0.633012 | 0.696488 | 0.549530 | 0.605571 |
| H_MEAN_A | held-out | 243 | 126/117 | 0.551486 | 0.598179 | 0.492891 | 0.559671 |
| B_MEAN_A | train | 3159 | 1638/1521 | 0.645872 | 0.703483 | 0.561416 | 0.615701 |
| B_MEAN_A | held-out | 243 | 126/117 | 0.572582 | 0.617130 | 0.500000 | 0.572016 |
| B_MEAN_C | train | 3159 | 1638/1521 | 0.711125 | 0.760365 | 0.640514 | 0.663501 |
| B_MEAN_C | held-out | 243 | 126/117 | 0.557591 | 0.623522 | 0.538462 | 0.555556 |

### Source-macro 结果

| 条件 | split | source 数 | ROC-AUC | AP | F1 | ACC |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| H_MEAN_A | train | 14 | 0.690476 | 0.791667 | 0.646032 | 0.594048 |
| H_MEAN_A | held-out | 14 | 0.626984 | 0.738492 | 0.649702 | 0.558333 |
| B_MEAN_A | train | 14 | 0.698413 | 0.805952 | 0.669524 | 0.620238 |
| B_MEAN_A | held-out | 14 | 0.666667 | 0.766270 | 0.694345 | 0.582143 |
| B_MEAN_C | train | 14 | 0.722222 | 0.817460 | 0.737143 | 0.653571 |
| B_MEAN_C | held-out | 14 | 0.571429 | 0.719048 | 0.609762 | 0.546429 |

held-out 主评价窗口是 81 个共同有效 MANIP 窗口（fake 42、real 39），CTRL 没有混入。
每个 source 的有效窗口数不完全相同；无效 source/窗口不填零，而在覆盖记录中保留
原因。训练与留出数量不同，训练指标不能解释为独立泛化指标。

逐 fold/seed 明细：

- `evaluation/train_heldout_forward_metrics.csv`
- 仓库小型副本：`docs/experiments/v7-boundary-pooling-train_heldout_forward_metrics.csv`

汇总：

- `evaluation/train_heldout_forward_summary.csv`
- `evaluation/train_heldout_forward_aggregate.csv`
- 仓库小型副本：`docs/experiments/v7-boundary-pooling-train_heldout_forward_summary.csv`
- 仓库小型副本：`docs/experiments/v7-boundary-pooling-train_heldout_forward_aggregate.csv`

逐窗口前向分数保留在数据目录的 `evaluation/train_heldout_forward_scores.csv`，不提交
Git。当前结果只证明保存模型可以被无优化器地重建并复现 OOF；不等于新增训练，不等于
跨生成器泛化或 sealed-test 结果，也不产生时间/空间定位真值。

## 3. 代码、测试与边界

新增的研究工具是
`research_tools/v7/boundary_pooling_probe/review_and_forward.py`，只负责读取已有
feature、ParticleSequence 和 fold model，生成离线页面及前向审计表。新增测试
`tests/research_tools/v7/test_boundary_pooling_review.py` 覆盖固定 logit 阈值、H→B
边界 pair 计数、PTS 到源帧映射和 pooled/source-macro 区分。

本轮没有修改正式 `src/` 检测链，没有增加点数、没有改变 St/导数/聚合、没有下载或
提交视频、NPZ、模型、权重或 ZIP，也没有访问旧 R7/V5。下一步不自动切换模型、密度、
分割或启动新实验。
