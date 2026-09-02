# ADR 0003: Mask validity boundary

## Status

Accepted

## Context

ADR 0002 冻结了 `visibility`、`geometry_validity` 及 missing 表示，但没有完整规定 mask 的生成条件、阈值纪律和 provider-failure 捷径风险。若将运动合理性、伪造标签或测试效果用于 mask，数据有效性门控会变成人工异常特征，并污染正常性学习与评估。

## Decision

- `visibility` 和 `geometry_validity` 是人工定义的数据有效性门控，不是人工异常特征。
- Mask 只判断观测是否满足最低可计算条件，不判断运动是否正常，也不判断视频是否伪造。
- Mask 策略对 real/fake 以及 train/validation/test 必须一致，并在下游检测评估前固定。
- Mask 只用于空间交互门控、时间缺失处理、prediction loss masking、evidence validity、localization validity 和 padding 区分。
- Mask 不与 `xyz` 拼接为普通连续特征，不建立用于学习数据源或真假差异的 mask embedding，也不直接把缺失率解释为伪造证据。

## Allowed validity conditions

Visibility 可以依据 tracker visibility/confidence、帧解码成功、`uv` 有限且在图像有效范围，以及 tracker 报告的遮挡、出界或跟踪失败。连续置信度二值化阈值不得依据 fake AUC、test 或个别视频调整；provider、阈值及版本记录在 lineage/provenance，连续 confidence 默认仅供上游审计，不进入几何主模型。

Geometry validity 只允许组合可信二维观测、有效且满足三维提升定义域的深度、可用内参、有效相机运动补偿信息、成功坐标变换、有限三维坐标及已声明坐标系和单位。相机补偿失败时，不得静默使用未补偿坐标并标记为有效；未来正式降级路径必须由新 ADR 决定。

## Forbidden anomaly conditions

速度、加速度、jerk、方向或曲率合理性、刚性、邻居协同运动、局部形变异常、运动类别、人工物理 residual、为伪造检测设计的 reprojection threshold、real/fake label、split、数据集类别、生成器身份、anomaly score，以及根据测试效果调整的阈值，均不得决定 mask。

## Shortcut risk

即使 mask 不作为连续输入，缺失模式仍可通过有效节点数量、有效时间长度和 loss/evidence 选择影响模型。后续必须审计 real/fake 间的 visibility rate、geometry validity rate 和有效轨迹长度分布，防止 provider failure 成为检测捷径。本 ADR 只记录该风险，不建立新的检测规则。

## Consequences

- Mask 实现需要明确记录 provider、阈值、版本和适用策略。
- validation 必须验证 mask 与有限值、相机补偿状态、坐标声明之间的一致性。
- provider 或阈值更换不能通过特定 split 或检测效果进行隐式调优。
- 具体 tracker、阈值、provider、网络、horizon、聚合和训练参数仍未冻结。

## Supersedes

本 ADR 不取代 ADR 0001，也不取代 ADR 0002；它补充 ADR 0002 中 mask 的生成与使用边界，不改变 `ParticleSequence` 字段、shape 或 missing 语义。
