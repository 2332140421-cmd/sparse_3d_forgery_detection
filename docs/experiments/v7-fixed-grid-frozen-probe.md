# V7 fixed-grid observation and frozen-model scoring pilot

本 pilot 以当前分支的 289 点前端、H/B 局部分组和已保存的
`H_MEAN_A`、`B_MEAN_A`、`B_MEAN_C` LOSO 模型为输入。它不是新模型训练，
也不是 sealed-test。

## 冻结协议

- 每个真实视频先按实际解码 PTS 建立时间原点；窗口长度 1.0 s、步长 0.5 s。
- 尾部不足一个步长时追加一个结束于最后 PTS 的 tail window；短视频不复制帧或填充。
- 网格、帧索引和 PTS 在标签应用之前保存。每个窗口重新初始化 17×17=289 查询点，
  不把相邻窗口的 track ID 当成同一身份。
- H 使用现有历史局部分组；B 只使用窗口首帧分割对 H 的子分组，不传播首帧 mask。
- 评分只读取 held-out source 自己的旧 fold 模型、原标准化和三个 seed；平均的是
  seed logit，不重拟合、不校准、不填补缺失分数。

## 时间标签

real 网格窗口作负类。fake 只有在整个一秒区间落入已有 manipulation 区间并集时
作主正类；部分重叠标记 `BOUNDARY_MIXED`，不重叠标记
`OUTSIDE_ANNOTATED_MANIPULATION`，两者只作描述。标签不参与网格、前端、分组或
模型输入选择。

## 运行方式

```bash
cd /root/autodl-tmp/projects/sparse_3d_forgery_detection
./research_tools/v7/fixed_grid_frozen_probe/run_all.sh --resume
```

所有运行产物写入数据盘的
`derived/v7_activityforensics_fixed_grid_frozen_probe_v1/`，包括协议、视频与网格
manifest、覆盖、窗口分数、评价、review 页面和 `progress.json`。后台脚本不执行
Git 操作；`BUDGET_EXHAUSTED`、`STOPPED_SAFE` 和 `COMPLETE` 必须按实际状态解读。

## 当前收尾状态

本次运行的固定计划为 32 个独立视频、2007 个网格窗口，但后台进程实际只留下
01KML 的首个 real/fake 基准窗口（2/2007）及其 H/B 特征和三条件分数；其余
2005 个窗口在前端阶段未运行。此前 `progress.json` 中的 `phase=score,
completed=2007` 是基准前向产生的评分行数，不代表 2007 个视频窗口已完成，已在
数据盘 `final_status.json` 标记为 `FRONTEND_INCOMPLETE`。因此本次 source 完整
覆盖数为 0，主 AUROC、配对差值和 bootstrap CI 均为 NA，不能据此判断冻结模型对
修改过程的响应。唯一的旧 OOF 轻量复现检查通过，绝对误差为 `5.96e-08`。

## 解释边界

可评分只表示离散 model-used PTS 具备输入支撑，不表示连续时间或像素区域覆盖。
无有效 triplet、前端失败、缺失 held-out 模型分别保留为缺失原因；不把缺失率当作
伪造证据。没有空间真值，因此页面中的 annotation 区间不是模型定位结果。
