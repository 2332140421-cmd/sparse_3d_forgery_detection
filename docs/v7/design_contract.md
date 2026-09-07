# 三维时空伪造 V7 设计契约

状态：V7 候选研究设计冻结基线

适用分支：`v7-dynamic-structure`

V7 从 V6 最新基线分支化，但不覆盖 `docs/design_contract.md`。V6 的实验结论是重新定义研究假设的动机，不证明 V7 正确。

## 1. 研究目标

V7 研究持续结构化运动域中的三维动态结构连续演化正常性。核心问题是：在具有稳定跨帧对应、持续三维可观测结构，且没有占主导地位的极端拓扑生成或销毁的动态过程中，真实视频的 particle motion、组件内部三维结构及组件之间三维关系是否具有可学习的一阶和二阶连续演化规律，以及生成视频是否偏离这种 real-only normality。

V7 不再把“确定性预测每个 particle 的具体未来三维位移”作为主研究任务。

## 2. 研究域：持续结构化运动域

Persistent Structured Motion Domain 包括刚体平移、旋转、关节运动、人或动物运动、车辆/工具/普通物体运动、中等非刚性形变、有限多物体交互和手物交互。

第一阶段不以液体飞溅、烟、火焰、大规模碎裂、爆炸、严重合并/分裂和占主导地位的拓扑出生/死亡为主要对象。原因不是这些过程“不物理”，而是当前 persistent-particle 表示依赖持续 correspondence。

物理域限制不等于语义场景限制。V7 不限定只研究人、道路或厨房，而限定能够由 persistent 3D particles 合理观测的动态结构过程。

## 3. 候选主链

```text
Video
  ↓
Stable Explicit 3D Frontend
  ↓
Persistent 3D Particles
  ↓
Motion-Coherent Dynamic Components
  ↓
Component Intra / Inter 3D Relation State S_t
  ↓
First-Order Structural Change ΔS_t
  ↓
Second-Order Structural Evolution Δ²S_t
  ↓
Real-Only Normality Modeling
  ↓
Particle / Component / Frame / Video Anomaly Evidence
```

组件算法、`S_t` 表示、距离度量、正常性模型和证据聚合均未冻结。

## 4. Persistent 3D Particle

Particle 仍是具有稳定跨帧对应关系的稀疏三维观测点。track identity 只表示 correspondence，不是语义标签、对象身份或模型连续输入。无效观测继续使用 NaN 和显式 visibility/geometry-validity mask，不得静默填零、插值或修复。

## 5. 运动一致动态组件

Motion-Coherent Dynamic Component 是一组在一定时间窗口内具有持续三维对应，并表现出相干运动或结构演化的 particles。

它不是 semantic object、body-part label、rigid-body ground truth、segmentation class 或 fake region。组件成员关系不得由 fake score、anomaly residual、真实性标签、split 或 generator identity 决定。Component discovery 与 anomaly detection 必须严格分离。

## 6. 结构状态及演化

`S_t` 表示当前时刻动态组件的三维关系状态，至少区分：

- intra-component relation：组件内部 particles 之间的三维结构；
- inter-component relation：不同动态组件之间的三维关系。

后续可以实验 relative XYZ、距离结构、局部坐标表示、旋转/平移不变表示或图关系，但本契约不选择最终表示。

`ΔS_t` 表示结构状态从上一时刻到当前时刻的变化，不等同于人工 velocity feature。`Δ²S_t` 表示结构变化本身如何继续改变，即 second-order structural evolution；它不宣称是真实物理 acceleration。

V7 判断连续结构变化过程是否属于真实视频正常演化分布，而不是规定下一帧必须位于唯一位置，也不规定二阶变化大或小对应真假。

## 7. Real-only normality

正常性模型只能使用 real training domain 拟合 `P(ΔS, Δ²S | real domain)` 或后续经实验确定的正常性估计。具体 density model 未冻结。组件发现、mask 和 frontend diagnostics 均不得读取 fake score、label 或 generator identity。

## 8. Stable Explicit 3D Frontend

第一阶段候选测量链为：

```text
Video
  -> fixed/causal 2D tracking
  -> per-frame depth
  -> camera intrinsics
  -> causal camera pose
  -> explicit back-projection
  -> camera-motion-compensated XYZ
  -> ParticleSequence
```

显式几何只负责把 XYZ 测准，不负责定义 fake。Reprojection 可以作为 frontend quality diagnostic，但不得直接成为 fake evidence。禁止将 reprojection anomaly、显式 velocity/acceleration/curvature anomaly、handcrafted physical residual 或 occlusion fake rule 引入检测主链。

历史观测不得依赖未来图像；不同时间的 XYZ 必须处于明确、可记录、可比较的坐标关系中；provider 输出必须先转换为规范 ParticleSequence，不能直接进入正常性模型。

Tracking、depth、intrinsics 和 pose provider 均未选择。

## 9. 四项研究假设

### H1 — Observable Geometry

在持续结构化运动域中，三维 frontend noise 必须显著低于需要研究的结构演化幅度。

### H2 — Component Representation

运动一致动态组件的关系状态应比完全独立的单 particle trajectory 提供更稳定、更可学习的结构描述。

### H3 — Evolution Normality

一阶和二阶结构演化应比确定性具体未来预测更少依赖具体语义 outcome，并在 real videos 中形成可学习 normality。

### H4 — Generalization Boundary

V7 优先追求 cross-object、cross-background 和 cross-generator 泛化；第一阶段不要求跨所有物态和所有拓扑过程。

以上均为可证伪假设，不是既成结论。

## 10. 从 V6 继承的工程与证据资产

V7 精确继承：视频解码、真实 PTS/timestamps、ParticleSequence 基础思想、track identity 语义、visibility/geometry-validity mask 语义、dataset lineage、可复现性基础设施，以及 V6 负实验作为 baseline evidence。

V7 不把以下 V6 项目直接继承为正式方法：确定性未来位移预测器、T0/T1/S 架构、dense spatial topology、V6 probe horizons、V6 loss 和 V6 aggregation。它们仅是历史 baseline。

保留的前端基线包括 VGGT real-primary `Q_target` median：h1=`1.6056`、h2=`0.9326`、h4=`0.4278`，以及仓库中既有 VGGT stability reports。

## 11. 当前未冻结

- 组件发现算法、组件数量和时间窗口；
- `S_t`、`ΔS_t`、`Δ²S_t` 的数值表示；
- component 内外关系和距离度量；
- normality/density model；
- particle/component/frame/video evidence 聚合；
- tracking、depth、intrinsics 和 pose provider；
- 采样率、粒子数量、训练数据与超参数；
- 定位、校准和阈值。

## 12. 第一阶段完成门槛

Phase 1 必须先证明显式前端具有足够 persistent coverage，并量化 depth jitter、pose drift、因果性、repeatability、共同坐标稳定性、same-UV XYZ coverage、frontend noise 相对 particle motion/结构变化的比例、运行成本，以及相对 V6 VGGT baseline 的改善。未通过该门槛前不得实现组件、结构状态或正常性模型。
