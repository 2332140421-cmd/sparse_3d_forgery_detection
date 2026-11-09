# ADR-0037：同深度分片下的 RGB-D / RGB 特征对照

用户2026-11-09要求深度只决定像素分片、片内特征仅来自RGB，并与当前版本配对比较。
基于 `cbc8c4d7162077f68bb4a411f436ed44358f829b` 新建 `pvst-rgb-partition-compare`。
本项只扩展ADR-0036的限定拟合诊断，不修改原实验及默认入口。

| 条件 | 整帧编码输入 | 片内打分输入 |
| --- | --- | --- |
| RGBD | RGB和现有归一化深度 | 局部特征和XYZ |
| RGB_DEPTH_PARTITION | RGB；深度通道为固定0 | 局部特征；XYZ三通道为固定0 |

先整帧编码，再于原生有效像素位置取特征和按深度归组。两条件使用相同DFormer-S结构
及预训练初始化；固定常量是移除样本几何信息的干预，不用于补写无效几何。保留所有有效点、
绝对片号位置编码、真实PTS、集合交互、帧头和像素头。该对照比较片内能否直接利用深度/XYZ，
不比较有无深度分片，也不改变整帧编码感受野。

使用原 `an:7ogg04`、`an:728qx7` 各real/fake首24帧。两条件从相同权重/种子重跑，
每条件原固定顺序80次AdamW、1800秒上限；同一GPU依次执行。初始化全模型hash、实际样本、
训练配置一致且完成相同更新数后才比较。用FP32帧BCE、同源真假logit、Dice、IoU及real
像素误报衡量拟合；两对训练样本不提供泛化结论。不重提前端，不访问开发或测试。

在实验室已有训练环境、仓库根目录运行：

```bash
PYTHONPATH=. python -m unittest discover -s tests -p test_rgbd_set_relations.py
PYTHONPATH=. python -u -m research_tools.rgbd_slice.rgb_partition_compare
```

结果位于已有数据目录的 `derived/pvst_rgb_partition_compare_v1/`，分条件保存日志、
指标和 `final.pt`，根目录生成 `summary.json` 和 `report.md`。
输出目录须为新目录；中断后保留证据，不自动覆盖或重训。权重不进入Git。
