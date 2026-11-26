# ADR-0058 官方MumPy架构在ANAKIN上的增广策略对照

状态：用户授权，2026-11-26。

问题：在既有SingleAugment之外，官方默认DoubleAugment是否改善多来源定位与后期稳定性？

决定：沿用ADR-0057完整模型、相同初值/来源/目标序列/预算/损失/优化器/FP32/224。
只新增显式augmentation_policy=double；旧默认single。DoubleAugment直接调用固定官方策略，
保留随机顺序、范围和RGB路径，只将旋转、缩放及CornerCrop回缩mask修正为类别最近邻。
训练GT可用于作者裁剪；开发不裁剪、不用GT驱动前向。空裁剪照常训练并计数，不重抽。

必要模块：double_augment由Clips调用，输出同步三帧与中帧mask并汇总轻量统计；
官方函数将mask按RGB插值，必须用局部修正，不修改官方checkout。
double_comparison由已有report调用，仅从两实验保存的JSON输出逐轮/分组/逐来源表，
不能重跑SingleAugment推理。新launcher委托旧日志/环境/退出码包装，不建新调度平台。

只读旧rgb224；正式初始化hash核对旧initialization记录，不加载旧训练终点。
仅新输出目录和新身份，不迁移旧身份或关闭严格恢复。阶段结束检查点可恢复三个SGD及随机状态。
Codex仅CPU检查和两次有效12短预检，正式10轮由用户启动；失败保留、不自动重试。

仅在本新条件中取代ADR-0057“避免GT引导训练裁剪”的SingleAugment选择；
旧实验与其结果、默认策略、所有其他方法边界保留。对照不称作者全部训练协议复现。
