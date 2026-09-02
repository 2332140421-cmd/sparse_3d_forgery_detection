# Sparse 3D Forgery Detection

本项目面向通用生成视频的伪造检测与时空定位，研究基于稀疏三维粒子的正常时空演化学习。目标是从三维运动与空间关系的正常演化偏差中形成异常证据，而不是构建完整三维世界模型。

冻结主链：

```text
Video
→ Depth + point tracking + camera-motion compensation
→ Tracked sparse 3D particle observations
→ ParticleSequence
→ Per-frame learned soft spatial dependency
→ Particle spatial state
→ Same-particle causal missing-aware temporal encoding
→ Particle spatiotemporal state
→ Deterministic direct multi-horizon future 3D displacement prediction
→ 3D prediction error
→ Particle/frame/video anomaly evidence
```

当前状态：仅建立设计基线，尚未实现代码，也没有可声明的实验结果。

权威方法契约见 [`docs/design_contract.md`](docs/design_contract.md)。旧仓库不是本项目的依赖或代码基线。
