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

### 6.1 `ParticleSequence` 逻辑数据契约

`ParticleSequence` 是三维前端与学习模型之间唯一规范数据边界，表示一个视频片段内具有稳定轨迹槽位的稀疏三维观测序列。它不是 provider 原始输出、完整视频世界表示、对象或部位集合、固定长度训练 batch、带标签的训练样本或人工运动特征集合。

一条 `ParticleSequence` 对应一个 clip：

$$
\mathcal P=\left\{X,U,M^{vis},M^{geo},I,T,Q\right\}.
$$

时间长度 $T$ 和轨迹数 $N$ 可以在不同序列间变化。规范逻辑字段如下：

| 类别 | 字段 | dtype / shape | 语义 |
| --- | --- | --- | --- |
| 身份与版本 | `schema_version` | `string` | 数据契约版本，不是模型版本 |
| 身份与版本 | `sample_id` | `string` | 当前 clip artifact 的唯一标识 |
| 身份与版本 | `source_video_id` | `string` | 源视频标识，不直接使用敏感绝对路径 |
| 时间轴 | `frame_indices` | `int64 [T]` | 严格递增的源帧索引 |
| 时间轴 | `timestamps_s` | `float64 [T]` | 有限且严格递增的秒时间戳 |
| 时间轴 | `frame_sizes_hw` | `int64 [T, 2]` | 各帧正数高度和宽度 |
| 轨迹身份 | `track_ids` | `int64 [N]` | 当前序列内唯一的稳定轨迹身份 |
| 粒子观测 | `xyz` | `float32 [T, N, 3]` | 相机运动补偿后、位于声明坐标系中的规范三维观测 |
| 粒子观测 | `uv` | `float32 [T, N, 2]` | 对应源视频帧中的像素坐标，用于审计和空间定位 |
| 粒子观测 | `visibility` | `bool [T, N]` | 二维轨迹观测是否受到上游跟踪结果支持 |
| 粒子观测 | `geometry_validity` | `bool [T, N]` | 三维坐标是否通过所需几何有效性检查 |
| 坐标语义 | `coordinate_system` | metadata object | 坐标系、单位、补偿和归一化声明 |
| 可追溯信息 | `lineage` | JSON-compatible mapping | 数据来源、帧选择、上游 artifact 引用和转换链 |
| 可追溯信息 | `provenance` | JSON-compatible mapping | 软件版本、配置摘要、运行标识和契约版本 |

时间轴不假设固定 FPS；时间跨度应依据 `timestamps_s` 解释，不能只依赖原始帧号。

#### 6.1.1 轨迹槽位与缺失语义

- `track_ids` 在当前序列内唯一，第二维槽位在整个序列中保持身份一致，但不宣称跨视频全局一致。
- 粒子不必在首帧出现，可以中途出生和消失；只有上游能够可靠保持身份时才允许重现。
- 重现不得静默复用其他粒子的 ID；track ID 不进入连续模型特征，也不建立 trainable identity embedding。
- `geometry_validity=True` 必须同时满足 `visibility=True`。
- `visibility=True` 时，`uv` 必须有限并处于依据对应 `frame_sizes_hw` 可解释的图像坐标范围；`visibility=False` 时，规范序列中的 `uv` 使用 NaN。
- `geometry_validity=True` 时，`xyz` 必须有限；`geometry_validity=False` 时，规范序列中的 `xyz` 使用 NaN。
- 遮挡期间由 tracker 外推的位置不保存为有效观测；零向量不表示 missing；missing 坐标不得静默插值。
- 若以后引入插值或轨迹修复，必须生成记录 lineage 的独立派生数据，不得覆盖原始规范观测。

#### 6.1.2 坐标语义

`coordinate_system` 至少记录：

```text
frame_name
handedness
axis_directions
length_unit
camera_motion_compensated
normalization
```

- 模型就绪的 `ParticleSequence` 必须满足 `camera_motion_compensated=true`。
- `length_unit` 至少区分 `meter` 与 `arbitrary_scale`。
- `normalization` 只记录是否执行及其参数；本阶段不冻结归一化算法。
- 未声明坐标系的数据不得进入模型；不同坐标系或单位不得在没有显式转换时混合。
- 原始 depth、pose、intrinsics 和 provider 私有张量不属于核心粒子数组，应由 lineage 指向其上游来源。

#### 6.1.3 Lineage 与 provenance

- lineage 记录源视频身份、帧选择、上游 tracking/depth/pose artifact 引用和转换链。
- provenance 记录生成软件版本、配置摘要、运行标识和数据契约版本。
- 二者均不作为模型数值输入，也不要求保存密钥、令牌或机器敏感信息。
- provider 置信度和诊断信息如需保留，应属于上游审计信息，不自动成为模型特征。

### 6.2 模型输入边界

默认几何主模型的连续输入只有规范三维坐标：

$$
q_{i,t}=X_{i,t}.
$$

`track_ids`、`frame_indices`、`timestamps_s`、`uv`、`visibility`、`geometry_validity`、lineage 和 provenance 只承担索引、门控、对齐或定位职责。时间戳可以解释真实时间间隔，但在经过专门设计前不得自动拼接为粒子外观或真假特征。

`ParticleSequence` 明确不包含 real/fake label、split 名称、数据集类别、生成器身份、velocity、acceleration、jerk、direction、curvature、physical residual、motion class、Part/Block/Region/Surface label、RGB embedding、classification target 或 anomaly score。标签、split 和评估真值由独立 manifest 管理。

### 6.3 序列与训练窗口边界

- `ParticleSequence` 是可变长规范观测；context/target 窗口从序列中派生，多个窗口可以引用同一规范序列。
- padding 只在 batch collate 时产生，不写入规范 artifact；padding mask 与观测缺失 mask 必须区分。
- 预测目标帧必须真实存在，且对应 `geometry_validity=True`。
- 具体历史长度、未来跨度和窗口步长仍未冻结。

### 6.4 物理存储边界

当前只冻结逻辑字段、shape、dtype 和语义，不冻结多 `.npy`、单 `.npz`、HDF5、Zarr、LMDB、WebDataset、shard 大小、压缩算法或 cache 策略。后续物理格式必须忠实保存本逻辑契约，不得反向改变方法语义。

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
