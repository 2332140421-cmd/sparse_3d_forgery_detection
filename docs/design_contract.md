# 三维时空伪造方法设计契约

状态：当前冻结设计基线

适用范围：三维时空伪造新仓库

事实优先级：

1. 用户当前明确要求
2. 当前 `design_contract.md`
3. 当前仓库 HEAD 源码与测试
4. 当前仓库中的决策记录
5. 旧仓库及历史材料仅供审计，不具有设计约束力

## 1. 研究目标

本研究面向通用生成视频的伪造检测与时空定位，提出一种基于稀疏三维粒子的正常时空演化学习方法。研究目标是利用三维运动和空间关系的正常演化偏差发现伪造，而不是构建完整三维世界模型。

## 2. 核心思想

- 不预先定义速度。
- 不预先定义加速度。
- 不预先定义 jerk。
- 不构造人工物理残差。
- 不定义运动类别。
- 不定义显式 Block、Part、Region、Surface。
- 不进行显式或隐式 Part discovery。
- 不把 attention 解释为真实物理连接。
- 以具有跨帧对应关系的稀疏三维跟踪观测点作为最小分析单元。
- 让模型从真实训练视频中学习正常三维运动与空间关系演化。

## 3. Particle 定义

$$
\text{Particle}=\text{具有跨帧对应关系的稀疏三维观测点}
$$

Particle 是视频三维观测空间中的时序对应单元，不宣称是真实物理粒子。粒子身份只表示轨迹对应关系；track ID 不是语义标签，也不是连续模型特征。

## 4. 冻结方法主链

```text
Video
  ↓
Depth + point tracking + camera-motion compensation
  ↓
Tracked sparse 3D particle observations
  ↓
ParticleSequence
  ↓
Per-frame learned soft spatial dependency
  ↓
Particle spatial state
  ↓
Same-particle causal missing-aware temporal encoding
  ↓
Particle spatiotemporal state
  ↓
Deterministic direct multi-horizon future 3D displacement prediction
  ↓
3D prediction error
  ↓
Particle/frame/video anomaly evidence
```

## 5. 三维前端边界

三维前端负责产生：

- 粒子轨迹身份；
- 三维位置；
- 二维投影位置；
- 时间信息；
- visibility；
- geometry validity；
- 坐标系和单位元数据；
- 相机运动补偿结果及其有效性信息。

depth、tracker、pose 的具体 provider 尚未冻结，相机运动补偿算法也尚未冻结。前端 provider 不得成为学习模型的隐式依赖。任何 provider 输出必须先转换为规范 `ParticleSequence`，模型不得直接读取 provider 私有格式。

## 6. Mask 边界

$$
m_{i,t}=m^{vis}_{i,t}\land m^{geo}_{i,t}
$$

Mask 仅用于：

- 空间交互门控；
- 时间序列缺失处理；
- loss masking；
- evidence validity。

Mask 不与三维坐标拼接成普通连续特征。缺失状态不得成为伪造标签捷径；不允许通过隐式填零掩盖无效观测。

## 7. 空间关系

有效粒子的候选关系空间为：

$$
(i,j),\qquad m_{i,t}=m_{j,t}=1
$$

基本关系输入为：

$$
\Delta X_{ij,t}=X_{j,t}-X_{i,t}
$$

模型学习：

$$
\Delta X_{ij,t}\rightarrow A_{ij,t}\rightarrow h^S_{i,t}
$$

- $A_{ij,t}$ 是学习得到的软空间依赖。
- 它不代表物理连接、因果关系、永久结构或 Part。
- 当前不通过 radius、kNN 或语义标签定义物理部位。
- 全连接只是候选关系空间，不表示所有粒子属于同一结构。
- 当前空间关系主要描述单帧几何依赖。
- 不声称 $A_{ij,t}$ 已根据历史运动发现“相似物理态粒子”。
- 空间关系的时间演化由后续空间状态历史编码体现。

## 8. 时间编码

$$
h^S_{i,t-L+1:t}\rightarrow z^T_{i,t}
$$

- 沿同一粒子身份进行因果编码，不跨未来泄漏。
- 对缺失观测显式感知。
- 时间编码输入是粒子的空间状态历史。
- $z^T$ 联合描述粒子自身运动与周围空间关系变化。
- 不人工输入速度、加速度、方向、曲率或运动类别。
- 不声称模型构建了完整物理状态机。

## 9. 未来预测

$$
\Delta X^{(h)}_{i,t}=X_{i,t+h}-X_{i,t}
$$

$$
z^T_{i,t}\rightarrow\widehat{\Delta X}^{(h)}_{i,t}
$$

- 第一版采用确定性预测。
- 采用多个未来时间跨度，各跨度直接预测，不进行递推 rollout。
- 预测对象是未来三维位移。
- 不预测 RGB、完整场景或显式 Part。
- 不以内部 latent 作为第一版正式预测目标。
- 具体 horizon 数值尚未冻结。

## 10. 正常性学习与异常证据

模型使用真实训练视频学习正常演化。粒子证据为：

$$
e_{i,t,h}=\left\|\widehat{\Delta X}^{(h)}_{i,t}-\Delta X^{(h)}_{i,t}\right\|_2
$$

Prediction error 是基础异常证据。误差归属于被预测的目标未来帧 $t+h$，不是历史截止帧 $t$。第一版不设置真假分类头或独立定位头，也不将训练集真假标签写入 `ParticleSequence` 数值输入。

## 11. 检测与定位

正式输出层级：

```text
particle anomaly evidence
    ↓
frame anomaly sequence
    ↓
video anomaly score
```

空间定位：

$$
a_{i,\tau}\rightarrow(u_{i,\tau},v_{i,\tau})
$$

- 正式空间输出是稀疏粒子异常点。
- Dense heatmap 是插值或核扩散得到的派生可视化，不是模型直接预测的 dense mask。
- 视频输出称为异常分数或伪造分数。
- 若需要真实性分数，必须明确规定方向转换或校准。
- 粒子、跨度、窗口、帧和视频的具体聚合算子尚未冻结。
- 阈值与校准协议尚未冻结。

## 12. 当前明确排除

- 完整三维世界重建；
- world model；
- RGB future generation；
- diffusion future generation；
- 概率多未来模型作为第一版；
- 显式 Part discovery；
- Block、Part、Region、Surface 层级；
- 人工物理残差；
- 人工运动特征；
- 人工事件 taxonomy；
- 旧 feature schema；
- 旧 fusion；
- 旧 residual prediction；
- 独立分类头；
- 独立定位头；
- 旧仓库兼容层。

## 13. 尚未冻结

以下事项尚未冻结，禁止实现者自行补全：

- depth provider；
- point-tracking provider；
- pose provider；
- 相机补偿具体算法；
- 坐标归一化具体公式和尺度估计；
- `ParticleSequence` 物理存储格式；
- 粒子数量和采样策略；
- attention 具体网络；
- hidden dimension；
- temporal encoder 架构；
- prediction horizons；
- loss 权重；
- evidence 聚合比例；
- calibration 和 threshold；
- 数据集与 split；
- 训练超参数。
