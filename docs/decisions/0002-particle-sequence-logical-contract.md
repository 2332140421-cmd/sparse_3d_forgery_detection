# ADR 0002: ParticleSequence logical contract

## Status

Accepted

## Context

三维前端可能来自不同的 depth、tracking 和 pose provider，而学习模型需要一个与 provider 私有格式解耦、能够显式表达轨迹身份、有效观测、坐标语义和可追溯性的统一边界。若不先冻结这一逻辑边界，物理存储或 provider 选择可能反向固化方法语义，并使 missing、轨迹槽位和模型输入职责混淆。

## Decision

- `ParticleSequence` 是三维前端与学习模型之间唯一规范数据边界，表示一个视频片段内具有稳定轨迹槽位的稀疏三维观测序列。
- 一条序列对应一个 clip；时间长度 $T$ 和轨迹数 $N$ 可在序列间变化。
- 冻结以下逻辑字段及其 dtype、shape 和语义：
  - `schema_version: string`；
  - `sample_id: string`；
  - `source_video_id: string`；
  - `frame_indices: int64 [T]`；
  - `timestamps_s: float64 [T]`；
  - `frame_sizes_hw: int64 [T, 2]`；
  - `track_ids: int64 [N]`；
  - `xyz: float32 [T, N, 3]`；
  - `uv: float32 [T, N, 2]`；
  - `visibility: bool [T, N]`；
  - `geometry_validity: bool [T, N]`；
  - `coordinate_system` metadata object；
  - `lineage` 与 `provenance` JSON-compatible mappings。
- 默认几何主模型的连续输入只有规范三维坐标 `xyz`。其余字段只承担索引、门控、对齐、定位或审计职责。
- 标签、split 和评估真值由独立 manifest 管理，不属于 `ParticleSequence`。
- context/target 窗口从可变长规范序列派生；padding 仅在 batch collate 时产生。

## Invariants

- `frame_indices` 严格递增；`timestamps_s` 有限且严格递增；`frame_sizes_hw` 的高度和宽度均为正。
- `track_ids` 在序列内唯一，轨迹槽位身份保持一致，不因粒子出生、消失或重现而静默复用。
- $M^{obs}=M^{vis}\land M^{geo}$，且 `geometry_validity=True` 蕴含 `visibility=True`。
- 不可见的 `uv` 和几何无效的 `xyz` 使用 NaN；零向量不表示 missing，不静默插值或覆盖规范观测。
- 模型就绪序列必须声明坐标系与单位，并满足 `camera_motion_compensated=true`。
- 不在没有显式转换时混合不同坐标系或单位。
- mask 不作为普通连续特征；track ID 不建立 trainable identity embedding。
- lineage、provenance、provider 身份和私有诊断信息不进入模型数值输入。
- `ParticleSequence` 不包含标签、split、人工运动特征、人工物理残差、结构层级标签、分类目标或异常分数。
- padding mask 与观测缺失 mask 必须区分；预测目标必须真实存在且几何有效。

## Consequences

- 所有 provider 输出必须先经适配和验证成为规范 `ParticleSequence`，模型不得读取 provider 私有格式。
- validation 和后续 artifact 实现必须维护字段 shape、dtype、时间单调性、轨迹唯一性、NaN/mask 一致性以及坐标声明。
- 插值或轨迹修复若以后引入，必须形成带 lineage 的独立派生数据，不能覆盖原始规范观测。
- 训练窗口和 batch padding 不得污染规范序列 artifact。

## Deferred decisions

- depth、point-tracking 和 pose provider；
- 相机补偿算法；
- 坐标归一化公式和尺度估计；
- 粒子数量与采样策略；
- 历史长度、未来跨度和窗口步长；
- 物理存储格式、shard 大小、压缩算法和 cache 策略；
- schema validation 与 artifact serialization 的具体实现技术。

## Supersedes

本 ADR 不取代 ADR 0001；它细化 `docs/design_contract.md` 中此前尚未冻结的 `ParticleSequence` 逻辑边界，但不冻结物理存储格式。
