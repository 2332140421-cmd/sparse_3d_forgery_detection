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

当前状态：正式 `src` 主链仍保持最小 schema/validation 契约；V7 的研究工具另行提供了已审计的周期重查询、五时刻序列、点对轨迹、局部聚合和 source 扩容 pilot 入口（`research_tools/v7/`）。这些工具读取数据盘上的视频、粒子数组、外部深度/跟踪权重与模型产物，clone 后不能在没有这些外部资产的情况下直接复现实验；它们不等同于正式检测链，也不代表设计契约已冻结为某个模型。

运行单元测试：

```bash
.venv/bin/python -m pytest -q
```

权威方法契约见 [`docs/v7/design_contract.md`](docs/v7/design_contract.md)。V6/历史说明与当前 V7 研究工具分开维护；旧仓库不是本项目的依赖或代码基线。
