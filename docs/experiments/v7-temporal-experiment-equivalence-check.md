# V7 Temporal Experiment Equivalence Check

状态：只读审查结论（当前分支 `v7-dynamic-structure`）

本审查只比较当前仓库已经实现并有产物证据的实验。测试代码的通过不被当作
科学实验已经运行的证据；没有访问视频、前端缓存或大型数组。

## 1. 证据表

| 实验 | 代码、协议与结果 | 输入与时间处理 | 方法、划分与配对 | 顺序破坏对照 | 实际状态与结果 |
|---|---|---|---|---|---|
| V7 B0 minimal no-reference | `research_tools/v7/b0_no_reference/`; `docs/experiments/v7-b0-minimal-no-reference-baseline.md`; `derived/v7_activityforensics_b0_no_reference_pilot_v1/` | 窗口级 `S_t=[mean,std,p25,p75]`，先对 component/time 做 median；窗口向量不保留时间顺序 | real-only source-disjoint LOSO 标尺；不使用 paired reference | 无 | **已运行有结果**：`B0_PAIRED_ONLY_NOT_NO_REFERENCE`，探索性 fake-MANIP vs real-MANIP AUROC `0.442929` |
| V7 Frozen B0 supervised linear probe | `research_tools/v7/b0_supervised_linear_probe/`; `docs/experiments/v7-b0-supervised-linear-probe.md`; `derived/v7_activityforensics_b0_supervised_linear_probe_v1/` | 同一 4D 窗口 B0 向量；无序列输入 | real/fake MANIP 标签、source-disjoint LOSO logistic regression；不使用 paired 数值作为输入 | 无 | **已运行有结果**：source 等权 AUROC `0.455556`，状态 `B0_LINEAR_DISCRIMINATION_NOT_ESTABLISHED` |
| V7 real-only structural normality pilot | `research_tools/v7/normality/protocol.py`（`extract_window_features`、`GaussianNormality`）；`docs/experiments/v7-real-only-structural-normality-pilot.md`; `derived/v7_real_only_normality_pilot_v1/` | 按真实 PTS 保留 `S_t` 顺序，再计算 timestamp-aware `ΔS_t`、`Δ²S_t`；逐 component-time 展平为 4D/8D 观测 | real-train → real-val source split；Gaussian/Mahalanobis real-only；Pair2 fake 因媒体阻塞未评分 | 无 | **已运行有结果，但与序列分类器不等价**：完成 real-only normality baseline；`PAIRED_TEST_MEDIA_BLOCKED` |
| V7 paired K0/K1/K2 and relation-first R1/R2 | `research_tools/v7/paired_signal/`、`research_tools/v7/relation_first/`；`docs/experiments/v7-activityforensics-paired-second-order-signal-pilot.md`、`docs/experiments/v7-activityforensics-relation-first-representation-pilot.md`; 对应 derived roots | K0 为 `S_t`；P1/P2 为顺序差分；R1/R2 对 persistent pair trajectory 先做导数再固定维度聚合；均使用真实 PTS 和 track-pair identity | 配对 real/fake MANIP/CTRL，real-control 拟合尺度；直接统计 fake-real discrepancy，不训练 sequence classifier | 无 | **已运行有结果，但与完整序列模型不等价**：K0 paired AUROC `0.748`；P2 `0.467`；R2 `0.514`；R2 未显示稳定动态增益 |
| V7 NSI / conditional NSI | `research_tools/v7/normalized_innovation/`、`research_tools/v7/conditional_nsi/`；`docs/experiments/v7-normalized-structural-innovation-pilot.md`、`docs/experiments/v7-minimal-conditional-nsi-calibration.md`; `derived/v7_activityforensics_conditional_nsi_pilot_v1/` | 保留局部三时刻 persistent pair triplet 顺序，计算 `v_minus`、`v_plus` 和 NSI；不是整段 `S_t` 序列输入 | paired mechanism statistic 或 real-only LOSO scalar ruler；不拟合监督序列模型 | 无 | **已运行有结果，但只回答局部 triplet/标尺问题**：NSI paired-sensitive，cross-source conditional ruler 未建立稳定 discrimination |
| V6 temporal-only GRU / spatial probe（历史基线） | `src/sparse3d_forgery/experiments/temporal_probe.py`、`spatial_probe.py`; `docs/experiments/temporal-only-real-only-learnability-probe.md`、`docs/experiments/spatial-dependency-falsification-probe.md` | 保留 per-particle XYZ history，GRU 按时间更新并预测未来位移；不是 V7 的 `S_t` | real-only predictive training；固定 fake probe；无 matched order control | 无 | **已运行但不是 V7 等价实验**：GRU fake-probe AUROC `0.4824`；spatial candidate `FAIL` |
| 同一表示、匹配模型的顺序破坏对照 | 在限定的 `docs/experiments/`、`research_tools/v7/`、相关 tests 和小型产物索引中未找到 shuffle/permutation/order-destruction 协议、实现或结果文件 | 未确认 | 未确认 | 未确认 | **当前证据未找到**，不能把缺失写成“从未做过” |

## 2. 等价性判断

1. **静态 B0：已有等价实验。** 无参考 B0 和 B0 监督 linear probe 都把每个窗口压成同一个 4D 向量；它们不测量顺序贡献。
2. **保留时间顺序的三维结构：已有相关但不等价实验。** V7 real-only pilot、P1/P2、R1/R2 和 NSI 都使用有序 PTS/track-pair 观测或局部 triplet，并已经有结果；但它们是差分、triplet、直接统计或 real-only Gaussian/标尺，不是“同一 `S_t` 序列输入、同一匹配模型”的监督时序分类器。
3. **顺序破坏匹配对照：当前证据未找到。** 当前没有证明训练和测试均按预声明方式破坏顺序、保持相同模型和训练预算、再与原顺序结果配对比较的运行记录。测试中出现的 `order`/`R1`/`R2` 是表示或冻结顺序名称，不是 shuffle 对照。

## 3. 对此前时序 CNN 提议的审查

- 现有产物可以构造单视频有序序列：原始 paired artifact 的
  `frontend/window_results.json`、ParticleSequence NPZ、PTS、组件成员和
  persistent pair trajectory 足以重建 `S_t` 或关系序列；B0 派生目录本身只
  保存窗口级摘要，不能单独恢复完整序列。
- 直接再训练一个 `S_t` 四维时序 CNN 会重复已经测过的“时间差分/局部 triplet
  统计”问题，并继续使用已经被 relation-first/negative-response 诊断怀疑
  可能有信息压缩损失的紧凑表示。仅更换模型名称不能产生新的科学问题。
- 因此，**撤销“直接启动 CNN”这一执行要求，保留其研究问题为未授权候选**。若要补最小缺项，应先预注册同一表示、同一模型、同一总体下的顺序破坏训练/测试匹配对照；在该对照之前不应重建整套 CNN 流程。

## 4. 当前最小缺项与建议

唯一清晰的等价性缺口是：在不改变 B0/`S_t` 表示、source 划分、模型和预算的条件下，定义并运行一个顺序保留 vs 顺序破坏的匹配对照。该对照必须明确打乱对象、发生在求导前还是后，以及训练和测试是否都打乱。

本轮不执行该补项，也不执行 CNN、前端、GPU、特征重算、full-video 或 sealed-test。若未来对照未显示顺序信息增量，则没有理由继续扩大到新的时序网络。
