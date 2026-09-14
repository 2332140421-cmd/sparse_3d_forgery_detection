# V7 五时刻原始状态序列与多阶状态序列匹配对照 pilot

这是独立的 V7 research-only 表示对照，不修改正式 `src` 检测链。运行入口为：

```bash
research_tools/v7/multi_order_sequence_probe/run_all.sh --resume
```

产物写入 `/root/autodl-tmp/data/sparse_3d_forgery_detection/derived/v7_activityforensics_multi_order_sequence_pilot_v1/`。
长任务应在独立 `tmux` 中运行；`progress.json`、`final_status.json` 和 `report.md`
记录阶段、累计预算和实际结果。

## 冻结问题与输入

本轮复用既有 16 个 source、192 个 MANIP/CTRL 窗口、289-query
`ParticleSequence` 和历史 H 局部组；不重跑 tracking、depth、pose 或
segmentation，也不把 04LAX 的独立 R 轨迹混入总体。每个 H 组只在五个冻结
目标偏移 `0.5, 0.6, 0.7, 0.8, 0.9 s` 都有严格递增的实际 PTS、相同共同成员和
相同 pair 时保留为一个 local unit。历史尺度由同一共同 pair 集合的历史有效距离
中位数得到，不逐帧归一化。五时刻筛选是离线持续可测关系的 survivor filter，
不是严格在线因果输入。

设 `S(t)=[mean,std,p25,p75]`。用实际秒间隔计算 `v` 和 `a`；它们是结构状态
的差分描述量，不是物理速度或加速度。四个冻结条件为：

* `SET_A`：共享 `4→16→8` 状态编码器，对五个状态平均后再接时间间隔和
  `12→16→8→1` head，保持对状态排列不变；
* `RAW_SEQ`：`[S0,S1,S2]、[S1,S2,S3]、[S2,S3,S4]` 展平为 36 维并加四个
  实际间隔，共 40 维；
* `MULTI_ORDER_SEQ`：把 `[S_i,v_i,a_i]`（`i=2,3,4`）联合展平为 36 维并加
  间隔，共 40 维；
* `SHUFFLED_MULTI_ORDER`：每个窗口冻结一个标签无关的非恒等五状态排列，
  在排列后的 `S` 上重新计算 `v/a`，使用同一 40→16→8→1 网络。

三个序列条件使用 Adam、`lr=1e-3`、`weight_decay=1e-4`、200 epochs、三个
固定 seed；LOSO 标准化只使用训练 source 的 MANIP real/fake，窗口 source/class
平衡 weighted BCE 真正作用于窗口损失。CTRL 不参与训练，仅作描述性评分。

## 评价和边界

先对三个 seed 的 OOF logit 做窗口平均，再对每个 source 计算 fake MANIP 对 real
MANIP 的 AUROC，最后 source 等权平均。主要比较是
`MULTI_ORDER_SEQ−RAW_SEQ` 和 `MULTI_ORDER_SEQ−SHUFFLED_MULTI_ORDER`；source
bootstrap 使用固定 seed `20260909`、10000 次且条件共享重采样索引。窗口重叠、同
source 相关性和五时刻共同支撑筛选均限制泛化解释。

结果支持的范围仅是当前开发总体中、持续可测局部关系上的表示学习比较；不支持
像素级定位、伪造区域真值、物理规律、旧轨迹跨失踪事件对应或未知生成器泛化。
