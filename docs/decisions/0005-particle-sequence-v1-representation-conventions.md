# ADR 0005: ParticleSequence v1 representation conventions

## Status

Accepted

## Context

ADR 0002 冻结了 `ParticleSequence` 的逻辑字段、shape 和语义，但 Python 内存实现需要明确 schema 初始版本，以及 handedness 和三轴方向声明的精确表示。若这些约定只存在于实现或测试中，后续无法区分正式契约与实现假设。

## Decision

- `PARTICLE_SEQUENCE_SCHEMA_VERSION` 的初始正式值为 `"1.0.0"`。
- `Handedness` 严格且仅允许 `left` 与 `right`，不接受任意自定义手性字符串。
- `axis_directions` 必须是 Python tuple，长度严格为 3，三个元素均为非空字符串，顺序依次对应 x、y、z 轴。
- 本阶段不新增轴方向字符串的额外词表。
- 这些决定只约束 `ParticleSequence` Python 内存 schema。
- NPZ、JSON、Zarr、HDF5 及其他物理存储表示继续保持未冻结。
- Block、Graph、Dataset、Model、Loss 和 Score 仍未实现。

## Consequences

- Validation 必须拒绝字符串形式的非枚举 handedness。
- Validation 不得把 list 自动转换为 axis tuple，也不得修复 tuple 长度或空字符串。
- Schema version 不匹配 `1.0.0` 时必须明确报错。
- 后续物理序列化设计必须忠实表达这些逻辑语义，但本 ADR 不选择存储容器或编码。

## Supersedes

本 ADR 不取代 ADR 0001、0002、0003 或 0004；它细化 ADR 0002 的 Python 内存表示约定，不改变 `ParticleSequence` 字段、shape、missing、mask、轨迹或空间候选关系语义。
