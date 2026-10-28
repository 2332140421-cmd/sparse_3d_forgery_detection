# ADR-0024：预训练空间定位基础与完整三维时序增强

状态：接受；日期2026-10-29。

## Context

随机浅层像素分支定位与开发泛化较弱，输出分离未改善定位。
用户授权用已训练的官方AINPAINT spatial基础，检验整个三维时序增强的附加价值。

## Decision

按`../v8/pretrained_spatial_context_contract.md`，只运行SPATIAL_TRANSFER和
SPATIAL_3D_CONTEXT：相同冻结官方骨干/已训练LoRA、decoder初值、ANAKIN48/16完整视频、
空间BCE+Dice、训练/选模规则；后者在decoder前加dense7通道几何及完整component/attention/GRU，
残差投影末层零初始化。新支路随机未训练，不使用旧定位checkpoint。
512补到518并裁回512；原几何映射按实际metadata对齐，缺失不删RGB监督。
只读复用原缓存/原96训练统计，新增共享冻结视觉特征，不重跑V8前端。

## Consequences

容量不匹配，报告全模块增益不归因于单独XYZ/component。
必须定位路径权重完整、官方forward和量化/初值一致性通过，不能随机替代缺失decoder。
最多30轮及dev Dice早停，best允许epoch0。独立入口自动评价和报告，不逐轮人工调参。

## Supersedes

仅在本新实验取代ADR-0022随机像素基础/联合检测监督与ADR-0023 monitor16/固定30轮限制。
保留原实验、默认入口和旧分数；V7及其他V8方法不改。
