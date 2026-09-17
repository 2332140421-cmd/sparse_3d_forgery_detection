# V7 表示敏感性与历史参照吸收探针

本 pilot 检查现有五时刻局部结构表示对受控几何和观测掩码操作的响应，不改变正式检测链，不训练新模型，也不把扰动样本加入真假评价。

## 固定范围

- 合成：8 个非共线点，5 个历史/5 个目标时刻，检查解析预期。
- 缓存：按 `sha256("representation-sensitivity|20260909|source")` 选 12 个训练 source，每个 source 一个 real/fake 窗口，每窗最多 3 个用于干预的 unit；冻结模型基线仍使用窗口全部有效 unit。
- 表示沿用现有实现：共同成员的五时刻 XYZ pair distance，以重算的历史中位数尺度归一化后得到 `S=[mean,std,p25,p75]`；`Q=[visibility,geometry,visibility-history,geometry-history]` 使用原始组成员。
- 仅作二次读出的模型为已有 observation-support 的 `STRUCTURE_ONLY`、`SUPPORT_ONLY`、`STRUCTURE_SUPPORT` 三 seed，复用原标准化和权重，不计算 AUROC。

## 操作

`G1` 整段刚体变换、`G2` 目标阶段刚体变换、`G3` 整段统一缩放并重算历史尺度应保持 `S/Q`；`G4` 目标阶段缩放用于检查历史参照敏感性。`G5` 是固定成员子集的渐进目标形变，`G6` 比较目标-only 与历史+目标的同一成员偏移。`M1` 改变非共同原始成员的目标观测，`M2` 改变共同成员集合，`M3` 只在合成样本检查中间时刻 Q 下降后恢复但该成员不重新进入共同结构。

## 产物

运行入口：

```bash
cd /root/autodl-tmp/projects/sparse_3d_forgery_detection
PYTHONPATH=. .venv/bin/python research_tools/v7/representation_sensitivity_probe/probe.py
```

结果写入数据盘：
`/root/autodl-tmp/data/sparse_3d_forgery_detection/derived/v7_activityforensics_representation_sensitivity_probe_v1/`。

`protocol.json` 和 `sample_manifest.json` 固定定义及样本身份；`feature_responses.csv` 保存 unit 级距离、尺度、S/Q 与掩码响应；`model_responses.csv` 保存冻结模型的 unit/window 二次读出、均值聚合核对和原分数复现；`report.md` 和 `final_status.json` 保存结论与状态。

结果只说明该表示在固定已有分组和共同成员下的数学不变性/敏感性边界，不证明跟踪、三维测量、跨查询物理对应、空间真值或未知来源泛化。
