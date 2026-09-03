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

当前状态：已建立独立设计基线并冻结 `ParticleSequence` 逻辑契约；当前实现仅包含内存 schema 和严格 validation，尚未实现 artifact、Dataset、provider、模型、训练或实验。

运行单元测试：

```bash
.venv/bin/python -m pytest -q
```

权威方法契约见 [`docs/design_contract.md`](docs/design_contract.md)。旧仓库不是本项目的依赖或代码基线。
