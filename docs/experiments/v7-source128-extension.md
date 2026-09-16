# V7 64→128 source 训练扩容对照

本实验固定已有 R 观测、SET_A/SUMMARY_SET、窗口、标签、标准化和平均局部聚合，仅将训练 source 池从 ordering=20260909 的 64 个扩展到原 64 加新增 64。新增 source 以固定 seed=20260909 的稳定 SHA-256 顺序从官方 train、exact-lineage 候选中选择，排除原训练 source 和冻结验证 source，不依据分数或前端支撑选择。

运行入口：

```bash
.venv/bin/python -m research_tools.v7.source128_extension.runner all --resume --device cuda
```

大型视频、ParticleSequence、前端缓存和模型记录保存在数据盘：
`/root/autodl-tmp/data/sparse_3d_forgery_detection/derived/v7_activityforensics_source128_extension_v1/`。

正式比较使用与固定 64-source MEAN_BASELINE 完全相同的 14 个双类别验证 source、83 个窗口；三 seed 先平均窗口 logit，再计算 source-macro AUROC。训练使用 200 epochs、Adam、`lr=1e-3`、`weight_decay=1e-4`、原 source/class weighted BCE 和 fold 内标准化。source bootstrap 为 10,000 次、seed=20260909。

该 pilot 不是 sealed test，不证明未知 generator 泛化或空间定位能力。未访问旧 R7/V5，不改变正式 `src` 检测链。

## 执行状态与恢复记录（2026-09-16）

最初的有界下载曾停在 209/288 并写入 `MEDIA_INCOMPLETE`；该历史记录保留，但已被后续用户确认的下载完成状态取代。本次恢复不重新下载：数据盘媒体清单现为 288/288 有效，144 个 source 均有 real/fake 媒体，状态为 61 `DOWNLOADED`、47 `MATERIALIZED`、180 `REUSED_EXISTING`，本地字节数与冻结 manifest 一致。

冻结计划已按 144 source、288 个父片段、864 个时间子窗口展开，既有 540 个窗口 ID/PTS/帧身份未改变。旧 80 source 的 160 个父片段缓存通过身份核验；新增 source `VG94P` 的 2 个 smoke 父片段有效。计划剩余 126 个新父片段前端计算。前端累计预算已消费 90.9448/7200 秒，剩余约 7109.0552 秒；此前端到端 smoke 为 PASS（CUDA，1 source、6 窗口、200 epochs），不进入正式结果。正式特征待按完整 864 窗口计划重建；旧 support 文件只有 540 窗口/1080 个 O/R 行，不满足当前计划身份，不能复用。正式模型仍为 0/3；固定验证为 14 source/83 窗口；特征加正式训练的 900 秒预算此前尚未启动。

恢复入口为 `.venv/bin/python -u -m research_tools.v7.source128_extension.runner all --resume --output-root /root/autodl-tmp/data/sparse_3d_forgery_detection/derived/v7_activityforensics_source128_extension_v1 --device cuda`。它在单一运行锁内依次完成前端、完整性核验后的特征、三 seed 正式训练、固定验证评分及报告。前端完成前不生成缺失特征占位；前端和特征缓存按固定身份恢复。特征耗时与训练共用且累计 900 秒上限；预算或运行失败会保留阶段状态并且不会标记 `COMPLETE`。训练只纳入冻结 128 source 池内标签合格、R 特征有效的窗口，不替换无支撑 source，报告同时给出计划池和实际有效覆盖。

恢复前最后核验：无同一实验运行进程，未发现正在运行的下载；源代码小修针对恢复完整性、有效训练覆盖记录和训练集评估，不改变冻结研究方法。启动状态、PID、阶段进度、累计预算与最终报告均写在上述输出目录的数据盘路径中。
