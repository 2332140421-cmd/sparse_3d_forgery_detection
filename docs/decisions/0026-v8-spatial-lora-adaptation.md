# ADR-0026：V8空间定位LoRA适配

2026-10-30，Accepted，用户明确授权。

问题：冻结预训练视觉特征是否限制空间定位，训练官方已有LoRA能否改善覆盖与选择性？

决定：只新增SPATIAL_LORA_ADAPT，对照直接引用ADR-0025固定BN迁移基线。
从同一官方完整初始化开始，原DINOv2冻结，已有LoRA与decoder训练，骨干eval、
decoder BN buffers固定而仿射训练。具体冻结设计、输入/损失/梯度权重、
monitor/评价/恢复边界统一记于[实验文档](../experiments/v8-spatial-lora-adaptation.md)，
不新增另一套合同或重复清单。

本新条件取代ADR-0024/0025的LoRA冻结和训练/评价缓存视觉输入限制；
原条件、缓存、历史结果和旧入口默认不变。不引入三维或component新贡献主张。
