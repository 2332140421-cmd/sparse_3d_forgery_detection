# 固定decoder BN运行统计的匹配对照

2026-10-29。PREFLIGHT_COMPLETE：20项相关测试及两条件真实完整pair smoke通过；尚无正式新分数。
起点`acbd539`，分支`v8-spatial-context-frozen-bn`。本轮只改BN统计策略，
不预设固定BN改善定位，旧结果保持原样。

[合同](../v8/spatial_context_frozen_bn_contract.md)、ADR-0025、
[配置](../v8/pretrained_spatial_context_frozen_bn_config.json)。
原BN策略结果及原完整身份清单见[原实验](v8-pretrained-spatial-context.md)，
限定BN模式诊断见[诊断记录](v8-pretrained-spatial-bn-diagnosis.md)。

## 策略与资产

SPATIAL_TRANSFER/SPATIAL_3D_CONTEXT均从原官方预训练checkpoint重新开始；
BN三个buffers固定官方初始值，仿射/卷积可训练，Dropout训练时开启。
模型结构、稠密XYZ/component/全帧关系/完整时序梯度、所有损失和训练规则不变。
默认开关False，旧配置保持原行为。

48训练/16开发、6178/1806帧与原manifest精确一致。视觉特征直接引用
`.../v8_pretrained_spatial_context_pilot_v1/features`；原joint缓存/96训练统计只读，
不下载、不构建或运行DINO、不复制大缓存、不做全量重新解码或大文件扫描。
epoch0复用旧完整评价并按固定样例核对，原best/last只作元信息保护，不加载微调。

输出绝对路径：
`/media/njupt/8FCD3869E2FAB06F/chenyh/data/sparse_3d_forgery_detection/derived/v8_pretrained_spatial_context_frozen_bn_v1/`

同一单入口，正式启动前先测试、smoke、提交和普通push；独立执行时Python自动完成
训练、最终AP、曲线、两条件固定样例和中文报告。
`reference_reuse.json`记录缓存来源/元信息，`epoch0_reuse.json`记录固定前向核对；
每轮history含BN状态，结果比较写`bn_strategy_comparison.json`与`bn_strategy_curves.png`。
完整结果尚未产生，不将smoke或旧分数写成本轮性能。

唯一后续建议等待本轮实际结果；不自动执行后续实验。

## 启动前实测

[小型验证清单](../v8/spatial_context_frozen_bn_preflight.json)保留配置/原manifest/epoch0哈希、
固定前向与真实smoke结果。64视频缓存身份及元信息核对通过，epoch0四次核对最大logit差均为0。
两条件各在完整`an:95qlg9` pair（180帧）更新两次后丢弃，5层BN始终eval，
三个buffers精确保持官方初始化，BN仿射和卷积实际更新；增强第二步几何/GRU梯度绝对和
为0.002498/0.017992。重算、跨块梯度、train/eval、恢复、旧默认、缓存只读及报告配对测试通过。

|条件|两次更新耗时(s)|常规/精确AP评价(s)|峰值allocated(MiB)|
|---|---:|---:|---:|
|SPATIAL_TRANSFER|1.087 / 0.985|0.499 / 1.809|359.18|
|SPATIAL_3D_CONTEXT|2.032 / 2.000|0.836 / 2.164|633.06|

依该单pair速率外推，两条件各满30轮的训练、常规评价、周期开发AP合计约84.65分钟；
不含最终完整best/last AP、样例导出及checkpoint I/O，早停可能缩短。这是预算外推，非完成耗时。
未新增特征存储，直接引用原20.85GiB特征与前端缓存。正式checkpoint与最终指标尚未产生。
