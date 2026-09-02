# ADR 0004: Spatial candidate topology remains unfrozen

## Status

Accepted

## Context

原设计契约把所有有效粒子对直接写为候选关系空间，并称全连接为候选关系空间。这过早固化了候选拓扑，将“哪些粒子对参与计算”与“模型在候选粒子对间学习多强的软依赖”混为一体，也在粒子数量尚未决定时引入固定的计算和显存假设。

## Decision

- 每帧候选集合 $\mathcal C_t$ 只包含该帧的有效粒子观测对，并作为学习式软空间依赖的计算域。
- 模型在 $\mathcal C_t$ 上依据相对三维位置 $\Delta X_{ij,t}=X_{j,t}-X_{i,t}$ 学习 $A_{ij,t}$，再形成粒子空间状态 $h^S_{i,t}$。
- 学习式软空间依赖继续作为冻结的方法核心；候选集合的具体构造策略恢复为未冻结。
- 候选策略不得使用真假标签、生成器身份、anomaly score、test AUC 或人工运动与 residual 条件筛边，并须跨 real/fake 和 train/validation/test 保持一致。
- 候选策略及参数必须写入配置和 provenance。

## Frozen principles

- 空间结构由粒子之间的学习式软依赖表示，不生成 Block、Part、Region 或 Surface。
- 候选边不解释为物理连接；attention 不解释为因果关系、永久结构或已发现的物理部位。
- 候选关系只在有效观测之间构造，基本几何关系输入仍是相对三维位置。
- 不加入速度、加速度、jerk、刚性 residual 或运动类别。
- 时间编码仍沿同一粒子身份处理空间状态历史。
- 未来三维位移预测和粒子、帧、视频三级异常证据主链不变。

## Deferred candidate strategies

以下选项及其参数均未冻结：full connection、3D kNN、kNN 的 $k$、radius graph、kNN + radius cap、approximate sparse attention、learned top-k、directed 或 undirected candidate edges、self-edge 策略、多层消息传递深度、每帧粒子数量、candidate graph 是否逐帧重建，以及具体计算和显存优化。本 ADR 不选择其中任何一种。

## Computational consequences

Candidate topology 决定哪些粒子对有资格进入关系学习计算，属于计算图、局部性归纳偏置、复杂度控制和待实验验证的模型选择。Learned soft dependency 决定模型在候选粒子对中应当多大程度参考另一个粒子，是冻结的方法核心。

kNN 或 radius 候选关系不等于人工 Part 或人工伪造规则，但会引入局部性偏置，必须作为模型选择或消融因素，不能解释为真实物理结构。全连接也不天然比稀疏候选更少人工假设；它取消局部性限制，但具有 $O(N^2)$ 计算、噪声关系和粒子数量受限风险。kNN 候选的关系计算规模为 $O(Nk)$。这些说明不冻结任何方案。

## Experimental consequences

候选策略应在相同粒子输入、时间编码器、预测目标、训练数据和 evidence 计算下比较，并使用相同参数预算或报告参数差异。概念上保留 temporal-only、dense soft relation 与 sparse geometric candidate + soft relation 的比较可能；是否全部实施，等待粒子数量、显存和初步实验后决定。

## Supersedes

本 ADR 不取代 ADR 0002 的 `ParticleSequence` 契约，也不取代 ADR 0003 的 mask 边界；它部分取代 ADR 0001 和原 `design_contract.md` 中“全连接候选关系已经冻结”的表述，但不修改旧 ADR 文件。它只将候选拓扑恢复为未冻结状态，学习式软空间依赖仍然是冻结的方法原则。
