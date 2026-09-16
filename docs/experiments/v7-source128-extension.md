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
