# 三维时空伪造方法设计契约

状态：当前冻结设计基线

适用范围：三维时空伪造新仓库

事实优先级：

1. 用户当前明确要求
2. 当前 `design_contract.md`
3. 当前仓库 HEAD 源码与测试
4. 当前仓库中的决策记录
5. 旧仓库及历史材料仅供审计，不具有设计约束力

本项目代码设计和实现服务于学术论文及科学假设验证，不以建设复杂、通用或生产级工程系统为目标。代码复杂度必须与论文证据链成比例；工程内容以实现方法、支撑实验、保证正确性和可复现性为边界。

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

当前可行性前端使用固定 revision 的官方 VGGT，同时获得相机内外参、z-depth 和点轨迹。该选择是可替换的实验实现，不是论文创新或永久 provider 架构。VGGT 会联合聚合输入序列，因此整段 joint-sequence 输出不得标记为因果预测训练就绪。正式因果窗口必须以只看历史帧的 Pass A 作为全部历史数组，以包含目标未来帧的 Pass B 仅提供未来数组，并只使用两次运行中同帧同 track 的共同有效历史观测拟合一个从 Pass B gauge 到 Pass A gauge 的全局 proper Sim(3)。拟合禁止使用未来点、标签、异常分数、provider confidence 或运动大小筛选；禁止逐帧、逐粒子对齐和 identity fallback。只有拟合非退化、尺度为正、旋转 proper 且未来有效三维点成功映射时，该特定 cutoff artifact 才可标记为因果构造合格。对齐残差仅是前端几何诊断，不是异常特征。前端 provider 不得成为学习模型的隐式依赖。任何 provider 输出必须先转换为规范 `ParticleSequence`，模型不得直接读取 provider 私有格式。

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

### 6.1 Mask validity boundary

#### 6.1.1 性质

`visibility` 和 `geometry_validity` 是人工定义的数据有效性门控，但不是人工异常特征。它们只判断观测是否具备最低可计算条件，不判断运动是否正常，也不判断视频是否伪造。

$$
\boxed{\text{mask 只判断观测能否计算，不判断运动是否合理}}
$$

#### 6.1.2 Visibility

`visibility` 允许依据 tracker 的 visibility/confidence、帧是否成功解码、`uv` 是否有限、`uv` 是否位于对应图像有效范围，以及 tracker 是否报告遮挡、出界或跟踪失败。

若将连续置信度二值化：

$$
m^{vis}_{i,t}=\mathbf 1\!\left[c^{vis}_{i,t}\ge\tau_{vis}\right].
$$

阈值策略必须对 real/fake 相同，对 train/validation/test 相同，并在下游检测评估前固定；不得依据 fake AUC、test 或个别视频调整。provider、阈值和版本必须记录在 lineage/provenance。连续 confidence 默认仅用于上游审计，不进入几何主模型。具体 tracker 和阈值仍未冻结。

#### 6.1.3 Geometry validity

$$
m^{geo}_{i,t}=m^{vis}_{i,t}\land m^{depth}_{i,t}\land m^{camera}_{t}\land m^{transform}_{i,t}.
$$

`geometry_validity` 只允许包含可信二维观测、存在且有限并满足三维提升基本定义域的深度、可用于反投影的内参、有效的相机运动补偿所需信息、成功的坐标变换、全部有限的三维坐标，以及已声明的坐标系和单位。

相机补偿失败时，不得静默使用未补偿坐标并继续标记为有效。未来若设计正式降级路径，必须新增 ADR。

#### 6.1.4 禁止条件

以下内容不得决定 mask：

- 速度、加速度或 jerk；
- 方向或曲率是否合理；
- 刚性、邻居协同运动或局部形变是否异常；
- 运动类别或人工物理 residual；
- 为伪造检测设计的 reprojection threshold；
- real/fake label、split、数据集类别或生成器身份；
- anomaly score；
- 根据测试效果调整的阈值。

#### 6.1.5 模型用途与捷径风险

Mask 只用于空间交互门控、时间缺失处理、prediction loss masking、evidence validity、localization validity 和 padding 区分。

禁止将 mask 与 `xyz` 拼接为普通连续特征，禁止建立用于学习数据源或真假差异的 mask embedding，也禁止直接把缺失率解释为伪造证据。

即使 mask 不作为连续输入，缺失模式仍可能通过有效节点数量、有效时间长度和 loss/evidence 选择影响模型。后续必须审计 real/fake 之间的 visibility rate、geometry validity rate 和有效轨迹长度分布，防止 provider failure 成为检测捷径。当前只记录该风险，不据此设计新的检测规则。

### 6.2 `ParticleSequence` 逻辑数据契约

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

`PARTICLE_SEQUENCE_SCHEMA_VERSION = "1.0.0"` 是 `ParticleSequence` Python 内存契约的初始正式 schema 版本。该版本只标识逻辑内存契约，不决定任何物理存储格式。

时间轴不假设固定 FPS；时间跨度应依据 `timestamps_s` 解释，不能只依赖原始帧号。

#### 6.2.1 轨迹槽位与缺失语义

- `track_ids` 在当前序列内唯一，第二维槽位在整个序列中保持身份一致，但不宣称跨视频全局一致。
- 粒子不必在首帧出现，可以中途出生和消失；只有上游能够可靠保持身份时才允许重现。
- 重现不得静默复用其他粒子的 ID；track ID 不进入连续模型特征，也不建立 trainable identity embedding。
- `geometry_validity=True` 必须同时满足 `visibility=True`。
- `visibility=True` 时，`uv` 必须有限并处于依据对应 `frame_sizes_hw` 可解释的图像坐标范围；`visibility=False` 时，规范序列中的 `uv` 使用 NaN。
- `geometry_validity=True` 时，`xyz` 必须有限；`geometry_validity=False` 时，规范序列中的 `xyz` 使用 NaN。
- 遮挡期间由 tracker 外推的位置不保存为有效观测；零向量不表示 missing；missing 坐标不得静默插值。
- 若以后引入插值或轨迹修复，必须生成记录 lineage 的独立派生数据，不得覆盖原始规范观测。

#### 6.2.2 坐标语义

`coordinate_system` 至少记录：

```text
frame_name
handedness
axis_directions
length_unit
camera_motion_compensated
normalization
```

- `handedness` 严格且仅允许 `left` 或 `right`，不接受自定义手性字符串。
- `axis_directions` 在 Python 内存契约中必须是长度严格为 3 的 tuple，三个元素均为非空字符串，并依次描述 x、y、z 轴方向；本阶段不冻结轴方向字符串词表。
- 模型就绪的 `ParticleSequence` 必须满足 `camera_motion_compensated=true`。
- `length_unit` 至少区分 `meter` 与 `arbitrary_scale`。
- `normalization` 只记录是否执行及其参数；本阶段不冻结归一化算法。
- 未声明坐标系的数据不得进入模型；不同坐标系或单位不得在没有显式转换时混合。
- 原始 depth、pose、intrinsics 和 provider 私有张量不属于核心粒子数组，应由 lineage 指向其上游来源。

#### 6.2.3 Lineage 与 provenance

- lineage 记录源视频身份、帧选择、上游 tracking/depth/pose artifact 引用和转换链。
- provenance 记录生成软件版本、配置摘要、运行标识和数据契约版本。
- 二者均不作为模型数值输入，也不要求保存密钥、令牌或机器敏感信息。
- provider 置信度和诊断信息如需保留，应属于上游审计信息，不自动成为模型特征。

### 6.3 模型输入边界

默认几何主模型的连续输入只有规范三维坐标：

$$
q_{i,t}=X_{i,t}.
$$

`track_ids`、`frame_indices`、`timestamps_s`、`uv`、`visibility`、`geometry_validity`、lineage 和 provenance 只承担索引、门控、对齐或定位职责。时间戳可以解释真实时间间隔，但在经过专门设计前不得自动拼接为粒子外观或真假特征。

`ParticleSequence` 明确不包含 real/fake label、split 名称、数据集类别、生成器身份、velocity、acceleration、jerk、direction、curvature、physical residual、motion class、Part/Block/Region/Surface label、RGB embedding、classification target 或 anomaly score。标签、split 和评估真值由独立 manifest 管理。

### 6.4 序列与训练窗口边界

- `ParticleSequence` 是可变长规范观测；context/target 窗口从序列中派生，多个窗口可以引用同一规范序列。
- padding 只在 batch collate 时产生，不写入规范 artifact；padding mask 与观测缺失 mask 必须区分。
- 预测目标帧必须真实存在，且对应 `geometry_validity=True`。
- 具体历史长度、未来跨度和窗口步长仍未冻结。

### 6.5 当前研究 artifact

昂贵前端输出当前使用数值数组 NPZ 与 JSON metadata 成对保存。NPZ 不包含 object array，读取必须使用 `allow_pickle=False`；JSON 保存身份、坐标声明、lineage 和 provenance。保存与加载后都必须通过 `ParticleSequence` validator，并保持 dtype、shape、NaN 和 mask 语义。该格式仅是当前研究的最小持久化选择，不建立多格式、shard 或通用缓存框架。

### 6.6 物理存储边界

当前 NPZ+JSON 是首个研究实现，不冻结长期存储架构、多 `.npy`、HDF5、Zarr、LMDB、WebDataset、shard 或通用 cache 策略。后续物理格式必须忠实保存本逻辑契约，不得反向改变方法语义。

## 7. 空间关系

每一帧首先定义一个仅包含有效粒子观测的候选关系集合 $\mathcal C_t$，随后模型在该候选集合上学习软空间依赖。软依赖的学习是冻结的方法原则，但候选集合的具体构造策略尚未冻结。

$$
\mathcal C_t
\subseteq
\left\{(i,j)\mid m^{obs}_{i,t}=m^{obs}_{j,t}=1\right\}.
$$

基本关系输入为：

$$
\Delta X_{ij,t}=X_{j,t}-X_{i,t},
\qquad (i,j)\in\mathcal C_t.
$$

模型在候选粒子对上学习关系：

$$
\Delta X_{ij,t}\rightarrow A_{ij,t}\rightarrow h^S_{i,t}
$$

- 空间结构由粒子之间学习得到的软依赖 $A_{ij,t}$ 表示。
- 候选边不代表物理连接；attention 不代表因果关系、永久结构或已发现的物理部位。
- 不生成 Block、Part、Region 或 Surface。
- 候选关系只在有效观测之间构造，基本几何关系输入仍是相对三维位置。
- 不加入速度、加速度、jerk、刚性 residual 或运动类别。
- 当前空间关系主要描述单帧几何依赖。
- 不声称 $A_{ij,t}$ 已根据历史运动发现“相似物理态粒子”。
- 空间关系的时间演化由后续空间状态历史编码体现。
- 时间编码仍沿同一粒子身份处理空间状态历史；未来三维位移预测及粒子、帧、视频三级异常证据主链保持不变。

### 7.1 Candidate topology 与 learned soft dependency

Candidate topology 回答“哪些粒子对有资格进入关系学习计算？”它属于计算图、局部性归纳偏置、复杂度控制以及待实验验证的模型选择。

Learned soft dependency 回答“在候选粒子对中，模型应当多大程度参考另一个粒子？”它属于当前冻结的方法核心。

kNN 或 radius 候选关系本身不等于人工 Part，也不属于人工伪造规则；但它们会引入局部性偏置，因此必须作为模型选择或消融因素，不能被描述为真实物理结构。

全连接也不天然比稀疏候选更少人工假设。它取消局部性限制，但带来 $O(N^2)$ 计算、噪声关系和粒子数量受限风险。

复杂度权衡为：

$$
\text{dense candidate}=O(N^2),
$$

$$
\text{kNN candidate}=O(Nk).
$$

这些复杂度只用于解释设计权衡，不据此冻结候选方案。

### 7.2 候选拓扑不得成为异常规则

无论后续采用何种候选策略，都必须：

- 对 real/fake 使用相同构造策略；
- 对 train/validation/test 使用相同构造策略；
- 不使用真假标签或生成器身份；
- 不依据 anomaly score 选择边；
- 不根据 test AUC 调整候选图；
- 不用速度、加速度、刚性或 residual 筛边；
- 将候选策略及参数写入配置和 provenance；
- 排除所有无效粒子观测。

### 7.3 后续实验原则

候选策略应在相同粒子输入、相同时间编码器、相同预测目标、相同训练数据和相同 evidence 计算下比较，并使用相同参数预算或明确报告参数差异。

至少保留 temporal-only、dense soft relation 与 sparse geometric candidate + soft relation 的概念性比较可能。具体是否全部实施，等待粒子数量、显存和初步实验后决定；本阶段不冻结实验配置。

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
- full connection；
- 3D kNN；
- kNN 的 $k$；
- radius graph；
- kNN + radius cap；
- approximate sparse attention；
- learned top-k；
- directed 或 undirected candidate edges；
- self-edge 策略；
- 多层消息传递深度；
- 粒子数量和采样策略；
- candidate graph 是否逐帧重建；
- 具体计算和显存优化；
- attention 具体网络；
- hidden dimension；
- temporal encoder 架构；
- prediction horizons；
- loss 权重；
- evidence 聚合比例；
- calibration 和 threshold；
- 数据集与 split；
- 训练超参数。

## 14. V8 Full-Coverage Observation Field 的隔离边界

V8 是当前项目内用户明确授权的独立方法分支，不修改或替换 V7 稀疏
ParticleSequence 主链。V8 的输入、前端、区域定义、缺失语义和模型合同以
`docs/v8/design_contract.md` 及其 ADR 为准；V7 的固定粒子、固定 component、
查询 cohort、S/Q 特征和模型缓存不得作为 V8 输入。V8 只可复用原始视频、官方
标签、source/split 身份以及无旧方法语义的窗口定位信息。

该段明确的是工程隔离和版本优先级，不把 V8 的全覆盖区域解释为物理部位，也
不把其前端 mask 或模型热图解释为像素级伪造真值。

### 14.1 V8 局部几何匹配 pilot

在用户授权的 `v8-local-geometry-encoding` 分支中，独立执行真实格内XYZ与
格均值复制的两条件对照。该分支新增可学习patch编码和公共padding隔离，以
`docs/v8/local_geometry_contract.md`、ADR-0014为准；不改变identity-fix六模型
的实现、产物或历史结果。V8范围内显式geometry状态是已授权的模型输入，
不套用V7稀疏ParticleSequence主链的mask输入限制。

### 14.2 V8 全视频流式工程 pilot

用户授权的 `v8-full-video-streaming` 分支按ADR-0015建立全部原始解码帧队列与
冻结V8模型的逐帧状态接口，仅完整处理一对开发训练视频。不改变模型参数、旧训练
协议或旧结果；不将连续帧工程响应宣称为经验证的全视频检测或帧/像素定位性能。

### 14.3 V8 全帧区域关系匹配对照

用户授权的 `v8-component-global-context` 按ADR-0016与
`docs/v8/component_global_context_contract.md`，在同MEAN_REPLICATED输入和cell时间路径下，
以共用单层同帧attention比较动态component与singleton格分区。仅扩展独立V8实验；
不改旧方法、全视频接口或正式src，不恢复V7数值资产。

### 14.4 V8 全视频全帧关系冻结评价

用户授权的`v8-full-video-global-context`按ADR-0017，将六个已完成全帧关系模型应用
于272视频全部解码帧。复用真实attention与原因果GRU，独立紧凑缓存；无训练或统计
修改。新视频级评价不替代旧窗口主指标，不展开帧/像素标签。

### 14.5 V8 全视频级弱监督微调

用户授权的`v8-full-video-training`按ADR-0018与
`docs/v8/full_video_training_contract.md`，在原272视频紧凑缓存上进行第二阶段
完整视频微调。保持模型、统计、区域与输入定义，梯度检查点跨块不截断；
一个视频只计算一个面积LSE及视频BCE。不替代旧冻结结果，不展开帧/格监督。

用户随后按ADR-0019收缩为主方法GLOBAL_COMPONENT三seed、固定5轮适配，
只改计划范围和主要比较，保留原20轮六模型协议及暂停checkpoint。全部视频、
完整跨块梯度、统计、模型和训练数值不变；不称两条件同样微调的匹配消融。

### 14.6 V8 官方时间区间辅助监督探索

用户授权`v8-temporal-supervision-pilot`按ADR-0020，在固定GLOBAL_COMPONENT、
seed20260909、5轮完整视频适配中，仅增加聚合frame logit的时间区间辅助BCE。
原视频BCE/模型/缓存/统计不变；不提供逐格空间标签，不替换旧视频弱监督结果。

### 14.7 V8 新服务器定位准备

用户授权`v8-spatial-localization`按ADR-0021建立独立环境与小规模原始媒体入口，
审计官方空间真值，并验证现有V8前端、随机模型接口和坐标映射。
人工标注入口的目标为可见伪造异常区域；没有空间真值时不生成占位负类。
本轮不启动全量前端、骨干重训或空间定位训练；后续两条件设计仅是提案，见
`docs/v8/spatial_localization_plan.md`，不改变既有V8模型或V7边界。

### 14.8 V8 完整双分辨率联合空间定位

用户授权按ADR-0022与`docs/v8/joint_spatial_contract.md`实现CELL_LOC和COMPONENT_LOC，
256结构/真实dense几何、512像素分支，按完整视频可微时序联合学习编辑区域。
本次取代14.7仅准备接口和可见异常目标的当前阶段限制，保留其历史事实；旧V7/V8
模型默认行为和历史分数不变。前端和ImageNet工具冻结，新模型配对随机初始化。

### 14.9 V8检测／空间输出分离对照

用户授权按ADR-0023与`docs/v8/head_decoupling_contract.md`比较完整COMPONENT_LOC的
COUPLED/SEPARATE_HEADS。仅新对照中允许独立GRU检测头，pixel定位分支不变；
固定原monitor16训练/原dev32和原96训练统计，缓存只读复用，固定30轮不早停。
本轮取代14.8在该对照中的单输出检测限制，保留原实验和其他默认路径。

### 14.10 V8预训练空间定位与三维时序增强

用户授权按ADR-0024与`docs/v8/pretrained_spatial_context_contract.md`比较
SPATIAL_TRANSFER与SPATIAL_3D_CONTEXT。完整官方AINPAINT spatial骨干/已训练LoRA冻结，
相同定位decoder可训练；B在decoder前加入dense几何和完整component/attention/GRU，
投影末层零初始化。全部原ANAKIN48训练/16开发、空间BCE+Dice、最多30轮早停，
原96训练统计与缓存只读。仅在新对照取代14.8随机像素基础/检测监督、14.9小队列固定轮数；
保留旧方法、历史分数与其他默认路径。

### 14.11 V8固定decoder BN运行统计匹配对照

用户授权按ADR-0025与`docs/v8/spatial_context_frozen_bn_contract.md`，仅在独立新实验
固定两条件decoder的官方初始BN buffers；仿射/卷积仍训练，Dropout策略不变。
从同一官方初始化重新训练，直接引用原视觉与前端缓存，数据/结构/loss/预算不变。
仅取代本新实验中14.10的BN更新策略，默认False保留旧入口行为和历史成绩。

### 14.12 V8空间定位LoRA适配

用户授权按ADR-0026及`docs/experiments/v8-spatial-lora-adaptation.md`仅训练一个
SPATIAL_LORA_ADAPT，从官方已有LoRA/decoder初始化，冻结原DINOv2、训练LoRA与decoder，
骨干始终eval、decoder官方BN统计固定。训练/评分使用在线FP32视觉及可微FP16量化，
逐chunk即时反传并按完整视频帧数累计pair梯度；完整输入/全部帧/原空间loss不变。
仅取代本新条件中14.10/14.11的LoRA冻结与缓存视觉路径，不改旧条件和历史结果。

### 14.13 V8空间监督扩量与匹配视觉适配

用户授权按ADR-0027与`docs/experiments/v8-spatial-supervision-expansion.md`执行
FROZEN_VISUAL/LORA_ADAPT，同一官方初始化、固定BN、全部帧与空间loss；source隔离扩量，
实际资格不足冻结实际规模，train/dev均扩充才训练，test_reserved不评分。
新增纯视觉RGB/GT缓存不依赖几何/统计；最多四个计算配置探针后冻结完整pair执行配置。
旧24pair模型只在新dev一次冻结参考推理。仅取代14.12中本新实验的规模、单条件和计算配置；
旧方法、旧结果与默认入口保持。实验COMPLETE先记录，Git/网络失败单列，不改原退出历史。

### 14.14 V8二维／三维时序结构匹配检测

用户授权按ADR-0028与`docs/experiments/v8-3d-temporal-detection-core.md`运行
STRUCTURE_2D/STRUCTURE_3D各三配对seed。在同RGB、容量、完整全帧时序和预算下，
比较UV/二维对应位移与XYZ/三维对应位移及其派生component/关系。2D路径不读取XYZ派生量。
48/20/20 AF source pair、原生PTS官方时间监督、全视频跨块梯度，从头训练而非加载旧窗口权重；
六模型完成后仅best统一测试一次。仅本新实验取代14.5/14.6旧初始化/小预算范围；
其他历史、定位实验和V7默认限制不变。没有几何/对应时仍保留RGB、当前UV及全部内容格。

## 15. PVST点—体素新方法独立边界

用户2026-11-02明确授权ADR-0029和`docs/pvst/design_v02.md`的UV_PVST/XYZ_PVST。
独立新方法，不改写V7/V8设计或历史分数。允许原生逐点几何、XYZ双尺度体素、
VFE和三层因果稀疏时空attention、独立帧/像素头、冻结无LoRA DINO及可训练官方decoder。
完整固定宽度128/256/512、全部内容像素、17帧最大感受野（左halo16）、原生旁路必须保留；
本新范围取代早期禁止Block/独立分类及定位头的限制，体素不解释为物理部位。
AF8/4及ANAKIN8/4 pair，全帧单seed；不加载旧格均值、component、检测checkpoint或旧统计。
资源不足完成等价优化及真实尺寸测量后停止正式训练，不缩减模型。

### 15.1 PVST v0.3 统一体素尺度与连续片段更新

用户2026-11-03按ADR-0030和`docs/pvst/design_v03.md`授权在独立v0.3中改变
XYZ统一体素尺度与训练目标/更新粒度：仅训练坐标预算从1/32、1/16、1/8取最细合格值，
UV保留1/32，每训练视频最多4个不同stride8目标段，AF/AN片段等权顺序累积更新。
全像素、所有有效点、点旁路、128/256/512、双尺度三层因果时空、真实PTS、8目标/16历史保持。
本轮仅两条件各一轮及完整开发/固定monitor评价，无best/早停/30轮追加。
此项仅取代v0.3中的v0.2固定XYZ1/64、全视频更新及全帧epoch覆盖/预算规则，
不称数学等价执行优化，旧方法、配置、失败记录和历史结论保留。

## 16. 观看深度整片关系研究（独立设计阶段）

用户本轮授权ADR-0031与`docs/pvst/depth_slice_relation_design.md`，在
`pvst-depth-slice-design`分支进行文档纠正，不授权模型实现、训练或GPU测试。
仅沿观看深度共享d0/h切整片，保留原生UV、有效可见XYZ及可学习点特征，
形成深度片×时间描述C与观测掩膜；不等同XYZ三轴体素，不以片均值XYZ替代学习。
新提案允许原生稠密可见观测和深度整片单元，不要求冒充旧ParticleSequence跟踪粒子，
不把片解释为物理部位/物体；缺失仍显式，不虚构遮挡后表面或跨帧配准。
2026-11-06按ADR-0032修订主线：片内先联合学习再聚合C，时间/深度独立参数的
显式成对关系先时间后深度聚合，真伪监督检测可看当前与历史但不读未来。
仅有视频fake标签不得复制到全帧；原real-only整帧遮挡预测移为备选，不能混作主线。
本轮另授权最多6段训练real、每段24连续帧的CPU可比性/规模审计，不实施模型。
具体参数与实验协议待验证冻结，片分数不等于精确像素定位。
本项只取代“立方体调粗即深度整片”的概念混同，不改写ADR-0029/0030、
旧PVST/V7/V8实现、正式配置或历史结果，也不将本稿建议称为已验证方法。

### 16.1 深度切片首轮监督训练授权

用户2026-11-07明确授权ADR-0033及`docs/pvst/depth_slice_pilot_config.json`的
单条件DEPTH_SLICE_RELATION完整实现、必要检查、训练与评价，取代§16仅设计/CPU审计限制。
允许现有相机运动而不做新场景过滤，不意味着已解决深度可比性。保留原生像素及有效XYZ，
只按+Z/s分片，片内先学习后聚合；两组独立显式时间/深度关系、像素旁路及独立帧头。
8/4既有ANAKIN pair、固定h1/8、D512、24帧片段、最多30轮及开发Dice早停见ADR-0033。
无旧体素模型、无取证预训练decoder、无新数据/前端下载，不覆盖旧PVST或V8历史。

### 16.2 预训练RGB-D与深度切片关系匹配实验

用户2026-11-07明确授权ADR-0034与`docs/pvst/rgbd_slice_config_v1.json`，
在pvst-rgbd-slice-relations仅实施RGBD_LOCAL/RGBD_SLICE_RELATION单seed。
共同官方原版DFormer-S RGB-D预训练全编码器微调、原生多尺度卷积解码及独立帧头，
B额外全点学习式深度片池化、两组显式时空关系与零初始化上下文残差。
复用原ANAKIN8/4及24帧清单，授权仅这些视频MoGe较高分辨率新观测和必需官方权重/依赖，
不重用256放大场。拟合检查先行，失败停止正式训练；两条件共同初始化逐张量复制。
两者都使用深度，不作为纯二维/三维增益证明；本新合同不改旧实验或模型。

### 16.3 最终关系上下文通道归一化单条件

2026-11-08用户按ADR-0035授权RGBD_SLICE_RELATION_NORM单seed匹配实验。
仅在完整关系C之后共享LayerNorm(512,eps1e-5,affine)，先逐token再均值进入帧残差；
像素残差消费同一C_norm，原零初始化保持。其余沿ADR-0034，旧A/B默认路径兼容。
复用原身份/48片段及全部缓存，不重跑前端/旧条件/单pair拟合；只新增1024参数。
两次必要更新检查后，只训练一个新条件max30/min10/patience8并评价best/last。
训练尺度为诊断，不替代空间/检测指标；无改善不追加修改。

### 16.4 自适应集合对集合限定拟合

用户2026-11-08按ADR-0036与`docs/pvst/set_to_set_design_v1.md`授权独立
pvst-set-to-set-relations：保留RGB-D全原生路径和SlabPool，以8潜在单元、A/B重叠sigmoid集合、
每角色4×512诱导状态、两块512/8head联合交互替代新入口的固定行列关系。
绝对深度和真实PTS差，只看当前及过去；逐片返回关系上下文，帧头仅由关系readout与512因果GRU驱动。
不是Part发现、互斥分区或物理因果；不套用早期禁止潜在集合的限制，也不改变旧方法合同。
仅训练an:7ogg04/an:728qx7各fake/real首24帧，最多80次总更新或worker1800秒，
旧共同初始化共享Local strict加载、新模块固定seed，官方空间GT与三项等权loss不变。
必要测试及一次关系消息干预、逐pair拟合报告；不启动正式对照，不读开发/测试，不重提前端。
旧A/B/NORM、失败历史、正式配置与结果保留；计算和发布状态独立。

### 16.5 仅分片使用深度的配对拟合对照

用户2026-11-09授权ADR-0037的RGBD / RGB_DEPTH_PARTITION对照。两条件共享§16.4
切片归属、有效点、绝对片号、PTS、关系模块、局部解码和三项损失。新条件整帧RGB编码时
将深度通道固定为常量，片内加权打分不使用实际XYZ；保留参数形状及共同初始化。
只对原两对四个24帧训练片段各从同一起点更新80次、各1800秒上限。分别保存输出和最终权重，
仅在身份、初始化、预算匹配时比较；不重提前端、不读开发或测试。该授权仅扩展新入口为
两条件对照，原§16.4入口和历史结果不变。

### 16.6 ANAKIN同数据集来源隔离RGB分片对照

2026-11-09用户按ADR-0038授权pvst-rgb-partition-anakin-dev：原train8/dev4、冻结48段及高分辨率缓存，
RGBD与RGB_DEPTH_PARTITION均从共同初始化重新训练12轮×48更新；完整集合关系与像素消息开启。
原生输入、valid/分片/绝对片号、PTS、宽度、BN、Dropout、BF16及三项等权loss保持ADR-0037。
同源real及编辑版本不得跨train/dev，逐源URL/媒体SHA补核；不读保留测试，不重提前端。
主比较epoch12，另列epoch1–12开发fake视频宏Dice best（平分取早）；各4小时时限，不自动重试或降级。
仅本独立实验取代16.5的两对80次更新范围；原拟合、干预和历史结果不变。旧实验无隐式恢复。

### 16.7 冻结切片特征的当前／历史关系读出对照

2026-11-11用户按ADR-0039授权pvst-slice-relation-readout独立训练路径。
仅原RGB_DEPTH_PARTITION last epoch12/576冻结RGB局部编码与SlabPool，提取一次原train8/48段1152唯一帧、dev4/924唯一帧的全部C512。
不运行旧sets/GRU/帧像素头作为新输入。CURRENT_ONLY与CURRENT_HISTORY共享一层512/8head交叉注意力、初始化及原12轮顺序；只差同帧其他片／加窗口全部历史片候选。
以(帧,片)排除自身，历史同号片可读；无候选才用共享可学习空参照。绝对片号位置和真实有符号PTS差共用设计，无未来/跨视频。
分别归一化C/R后仅差与乘积进入1024→128→1，无C分类旁路；正片为任一有效官方编辑像素，train固定负/正比加权BCE、逐段全部有效片归一化。
两组各12×48=576次AdamW(lr1e-4,wd.01)，第576步主比较，无best替换或开发调参。
主指标同画面正负片均存在的fake帧slice AP→视频宏平均；有效像素回填AP单列，不能称精细定位或与旧全画面AP直接比较。
只扩展本独立探针的授权范围，不取代16.4–16.6旧模型默认行为、历史结果或正式协议；不读保留测试、不重提前端、不加方格。

### 16.8 CPU切片定位表达上限（无模型）

用户授权ADR-0040，在pvst-slice-localization-oracle仅计算原开发四fake/462帧的GT整单元Dice/IoU上限。
保持原高分辨率几何回映、valid、Z/s、h=.125；唯一候选细分是原生L=ceil(min(H,W)/16)方格。
GT只用于排序单元密度及最优掩膜，不参与划分；该结果不是推理规则、模型成绩或方格训练授权。
仅为当前CPU诊断解除16.7“不加方格”的范围限制，不修改旧模型、配置、协议、结果或训练路线。

### 16.9 整片与片内固定方格匹配定位

用户授权ADR-0041/pvst-slab-grid-localization：仅原train8/48段与dev4，冻结原last局部RGB编码/SlabPool。
一帧仅编码一次，两组同score/value权重但各自单元完整四头softmax；GRID严格沿16.8固定格长，无新XY编码。
两组共享CURRENT_ONLY 512/8head读出与新初始化，仅当前同帧其他单元，无历史。查询分块SDPA不丢候选，保留完整梯度。
损失统一为原有效像素计数BCE，train负/正像素比共享，不再使用16.7任一GT像素即正片的监督。
先小图等价与真实最大训练片段完整更新验收，再各12轮/576步；原顺序、AdamW1e-4/wd.01，最后权重主比较。
以同一有效范围回填pixel AP及固定阈值空间指标评价。仅本独立实验扩展16.8无方格模型限制，不改旧默认与历史。

### 16.10 缓存XY格历史标量的CPU线性探针

ADR-0042授权本独立train8/dev4缓存诊断：原L、几何、C不变，按counts合并XY格；H3共有非零格，训练片段独立起窗。A=当前F，B=F加三段逐格余弦距离和，仅训练集加权标准化及像素平衡L2拟合。只比较相同当前GT与有效范围。旧模型、配置、划分及结果不变；无神经前向、前端或开发调参，不替代16.9正式实验。

### 17. 紧凑等尺寸XYZ块学习式时序探针

ADR-0043/xyz_temporal_config_v1.json登记用户明确授权的独立train7/dev4实验。通用冻结视觉、等尺寸XYZ块、几何参与可训练块编码、学习式空间及跨时刻跨位置交互、全图局部旁路和空间定位保留。A当前/B当前与历史同构同初始化同576步；不要求历史共有键，不用手工距离代替网络。工作分辨率、训练尺度和h选择规则预先登记。40y1x9 fake6–9仅无标签历史。仅本探针允许紧凑D128单尺度，旧方案、深度片/方格替代实验和历史结果保持；不称世界米制或三维独立贡献。一次真实资源验收后执行授权训练和末次评价，失败保留证据，不自动降规格或追加架构。

### 17.1 两层窗口联合关系匹配实验

ADR-0044与xyz_joint_relation_config_v1.json登记用户本轮明确授权的独立pvst-xyz-joint-relation-v1。
复用§17的train7/dev4、通用视觉/XYZ观测、scale/h与8槽PTS，新增前端计算为0。
新可训练块编码输出C0，不经过旧space；两层D128/4head窗口自注意力接替space/context，
XYZ中心与相对PTS仅加入各层Q/K，归一化内容作V，全部有效token参与更新。
A按实际观测帧身份只开放同帧连接；B窗口内双向，只输出截止目标，禁止目标之后观测。
重复槽保留实际观测身份，缺失不伪造；无共有块键、数量输入或状态监督。
逐块R保留定位映射/全画面视觉旁路；Linear块证据以logsumexp−logN产生帧logit，
替代旧均值帧MLP，空集使用可学习空表示。视频均值帧logit仅辅助口径。
两条件相同新初始化、三项等权loss及采样序列，各14700目标更新，无best/早停/搜索。
训练56帧观察于0/576/4410/14700，开发824帧于576/4410/14700；一次最大窗口更新检查后自动训练。
末次B仅五个目标做一次C0提取后的FP32同输入重复/历史时间槽反绑定对照，不作额外训练。
本新实验仅取代§17/ADR-0043在其范围内的旧关系/帧读出与576步预算，旧入口/协议/结果保留。

### 17.2 同来源前后片段定位探针

ADR-0045/pvst-xyz-insource-probe-v1授权仅原train7来源，配对共用实际PTS跨度70%边界，
前段训练、后段评价，所有读取窗口严格在各分区内；原dev4不参与。40y1x9_fake6–9仍不监督计分。
统一scale仅从新训练段按原256点/帧正Z中位数规则估计，h=.5不搜索，旧通用缓存只读，无新前端。
I为相同视觉多尺度/细节/像素解码，context通道全零，不读取XYZ或运行块/关系路径，无无效参数填充。
A/B共用XYZ联合编码与两层128/4头关系，同帧／同分区8槽历史；移除未监督帧读出参数。
仅pixel BCE+soft Dice等权，三组公共参数同初始值；相同全部训练目标无放回打乱完整遍历10轮，
第3/10轮固定观察及完整后段评价，主结果第10轮，不选best/调阈值。观察每训练视频约1/3、2/3合法目标。
全原生空间计分含real FPR与fake非GT背景FPR，无帧头成绩或编辑起止结论。
本独立实验仅取代§17.1在其范围内的数据划分、采样预算、三条件与空间监督，不改旧协议/结果。

### 17.3 固定权重时间读出诊断

ADR-0046/pvst-xyz-temporal-readout-diagnosis-v1授权仅§17.2的A/B第10轮权重和444后段目标，
A_CURRENT、B_FULL、B_CURRENT三读出；后两者同一B权重。无训练、模型结构修改或新前端。
先按H0/H1–6/H7汇总旧逐帧结果，槽数与唯一观测数分开；分层仅关联分析，不替换主成绩。
保持原scale/h/分区、BF16与FP32原生恢复；A/B不共享C0，B内部只复用冻结C0。
概率阈值预先固定.01–.99步长.01，含.5；GT分区仅用于输出统计。流式计数，不保存全队列预测。
原生FP32排序AP不由阈值/PNG重算。曲线不选部署阈值；诊断上界明确使用评价标签，不作新主结果。
保留原42固定预览，无帧头、保留测试或起止任务；旧协议和产物不改。

### 17.4 同源B定位读出适配

ADR-0047授权仅从§17.2 B epoch10完整模型/AdamW/随机状态独立恢复两组：
READOUT_ONLY只更新pixel_context/local_out/detail/pixel；FULL_CONTINUE更新原全部可训练参数。
先完整顺序创建并恢复原AdamW，再冻结；冻结参数grad=None，读出动量/方差/step保留。
两组显式读取XYZ_TEMPORAL完整历史，原protocol/scale/h/监督范围和空间loss不变，无新前端。
共同起点仅回放一次；追加原第11–13轮共同无放回顺序，各3078次更新，累计13338。
每轮仅评价原28观察目标，末次评价原444后段目标；42固定预览，不选best或调阈值。
本次仅扩展§17.3的无训练限制以执行本独立适配对照，不更改旧默认、结果或数据角色。

### 17.5 冻结定位权重的同数据集独立来源开发评价

ADR-0048仅评价§17.2的I/A/B epoch10（训练原7来源前70%，各10260步），
以及§17.4 FULL_CONTINUE末次13338步的附表；原dev4完整824帧，不再次70/30切分。
固定原训练scale/h、模型、BF16/FP32计分与完整截止历史，独立输出，不更新参数或前端。
只补标签降采样恢复重叠与GT几何有效覆盖参考，不称模型理论上限。
新增候选仅从已有第二批池准备train/unassigned人工页，不按分数挑选，不自动冻结划分。
只扩展本轮评价范围，不修改原协议、模型默认或结果；开发不称最终测试。

### 17.6 全视频训练来源扩充配对

ADR-0049授权TRAIN7_FULL与TRAIN_EXPANDED_FULL，固定现有XYZLocalization完整历史路径。
原7来源与原7＋第三批指定技术PASS四来源（明确逐IDmask排除优先于本轮批次授权），
训练全视频合法目标，fake40y1x9 6–9只作无标签历史。原dev4完整824帧仍来源隔离。
仅新来源缺失帧可按原官方DFormer零depth/MoGe流程补观测；旧缓存只读，C/R必须在线更新。
共同固定scale=12.880994319915771、h=.5、D128/2层4头/8槽.1秒及原空间loss/优化器。
两组严格复用原随机initial.pt/XYZ，优化器全新；U=10*Nexp在训练前冻结。
扩充组10完整无放回轮，原7同算法循环至U，尾轮可部分；不同集合不宣称同序列。
U/2和U检查点/评价，原7共同28观察与新增观察分列，末次主结果及24固定预览。
不按开发调整预算/架构/阈值。比较同预算覆盖与重复策略，不解释为独立样本数量效应。
本独立协议取代§17.2在其范围内的70/30训练限制、§17.5仅评价限制，不修改旧协议/成绩。

### 17.7 官方DFormer真实末级适配开发筛选

ADR-0050授权仅DFORMER_STAGE4_TRAINABLE一组，复用§17.6扩充11来源、原开发4与原订单前12845项。
真实阶段边界由官方forward确定：缓存前三阶段后原始RGB/depth状态，末级RGB/depth下采样及两个block可训练。
冻结前缀、全部运行统计和DropPath维持原eval行为；eval不关闭末级autograd，不增加输出后适配器或分类尾部。
所有唯一当前/历史观测逐更新重算末级，保留历史梯度；原low3特征与MoGe/XYZ复用，原half存储路径以可微cast匹配。
PVST原initial.pt/XYZ、官方末级初值、全新AdamW，PVST lr1e-4/末级1e-5，原loss/clip/尺度/结构不变。
先固定少量train/dev校验切分、像素初值、梯度与冻结边界，丢弃检查更新；再5完整遍历12845更新。
每遍历固定共同/新增观察分列，仅末次824开发评价；冻结同预算参照只读引用，24预览可仅补固定目标前向。
本独立实验仅解除§17.6完全冻结视觉的限制，不变更旧实验、默认路径、预算或数据角色。

§17.7预检保留同输入普通整链像素差与原池化归约自重复证据；逐段验证输入、路由、local/C0后，仅下游读出核验共享该共同归约的C0，数值容差不变。共享不进入正式训练，首次失败不删除。

### 17.8 官方AINPAINT空间结构能力参照

ADR-0051授权一个独立AINPAINT_SPATIAL：通用dinov2_vitl14_reg，LoRA4/8/.25及官方HighResContextHead。
只训练LoRA/head，原始DINO冻结；T1、不读XYZ/历史，不加载取证成品或RAFT。原扩充11/开发4、
原订单前12845，完整RGB长边384归一化右下pad14，严格patch网格，head先输出pad尺寸再裁回。
原BCE+Dice各1、AdamW1e-4/wd.01/clip1/BF16，五轮无放回曝光各5；每轮前100训练目标额外BN更新，
训练观察28/16分列、末次824评价及固定24预览。无best/搜索/新增分支，不改旧结果。
只在本独立参照扩展17.7的单DFormer范围，不宣称XYZ单变量消融或作者完整训练复现。

### 17.9 AINPAINT固定权重BN统计诊断

ADR-0052授权原空间模型final12845、原dev824的RUNNING_BN/IMAGE_BN各一次。
整体eval/no_grad、dropout关闭；只临时替换head五BN为函数式当前单帧空间统计，
保留affine/eps、不读写running统计或计数；结束恢复。无训练/优化器/新校准/阈值搜索。
预定少量回放及重复，原生指标与24固定图/统一差分保持；不替换17.8原报告或推广正式目标。

### 17.10 原生局部细节匹配输入探针

ADR-0053授权NATIVE_PATCH/LOWDETAIL_PATCH、原11/4来源和订单，各12845单crop更新。
共享288原生坐标GT采样计划；仅LOW整帧INTER_AREA至原work_hw再INTER_LINEAR恢复后裁同区域。
原生GT/valid不退化，padding排除loss；相同新通用DINO+LoRA/head初值及原训练参数/500额外BN前向。
原824帧tile288/stride144/贴边全覆盖，FP32 logit均匀融合，不依GT路由。只末次评价和固定24图。
少量真实预检后训练两组；不改C/XYZ/关系、不提前端。旧全图384仅历史，不当本轮单因果对照。

### 17.11 扩充来源的整帧空间训练

ADR-0054授权仅EXPANDED_SPATIAL：上一轮93个技术PASS按既有train/dev草案分流，合并原11训练/4开发并按原视频隔离。
取消逐条人工确认前置；保留原文且不宣称语义验收，明确mask反证、技术待核、角色冲突、未知身份及94测试保护来源仍排除。
沿用17.8通用DINOv2+原LoRA/head、完整长边384/pad14和BCE+Dice。单组seed20260909新初始化，10完整无放回epoch，U=10*Ntrain。
每轮既定训练订单前100额外BN前向，仅训练统计；第1/3/5/10轮完整开发及固定观察，末次主结果，ADDITIONAL_DEV AP选开发补充best。
旧11来源空间权重只在同一扩充开发回放，不训练/更新BN。仅准备缺失工作RGB/GT，无MoGe/DFormer或神经特征预计算。
本轮仅取代17.8在其独立范围的数据规模/预算/评价时点及旧候选“待逐条人工确认”限制，不改模型、旧协议与结果，不称等预算纯数据量因果或PVST三维验证。

### 17.12 完整视频原始XYZ块原型与限定资源预检

ADR-0055登记用户授权的独立pvst-fullvideo-direct-prototype-v1：冻结扩充空间epoch01的DINO/LoRA/官方head，
一次前向缓存patch1024、predictor输入detail32及原工作logit；自身已有MoGe只读，无新MoGe。
固定scale12.880994319915771/h.5/origin0，FP32三轴路由、保留全部点和非空块；新视觉投影与旧几何调制/逐点学习池化形成C128。
两层128/4head/FF256、dropout0，XYZ中心与实际PTS入Q/K、内容作V；FRAME_ONLY逐帧，FULL_VIDEO全片原始块直接双向，无因果/中继/筛块。
所有帧R回填32，与冻结detail32融合，零初始化末层delta叠加z0；无帧头。全片合法帧BCE+Dice平均，一次反向更新。
Codex仅完整低token病例4ym5n4_fake、两模式各2次可丢弃AdamW预检；40y1x9/9mygy9完整测量由用户终端启动。
可非重入checkpoint/精确Q分块1024，不缩片、不丢块、不detach；失败记录，不自动重试或降级。无正式训练/开发效果评价。
仅在本独立范围取代17.1/17.2的截止8槽、单目标输出及旧视觉接口；不改旧协议/成绩。取代设计对齐文档中未批准的中继推荐，采用用户明确指定原始块直接交互。

### 17.13 完整视频匹配定位脚本交付与用户启动协议

ADR-0056授权pvst-fullvideo-matched-localization-v1：直接复用§17.11冻结protocol的89训练/19开发来源，
FRAME_ONLY/FULL_VIDEO均完整单视频输入/全部帧定位，仅跨帧连接不同。复用§17.12模型数学及epoch01冻结底座。
共享一次观测准备：两种工作RGB/GT格式，自身兼容MoGe或原provider补缺，冻结视觉同次patch/detail/z0，不提DFormer；缓存恒定路由而不缓存C/R。
统一scale12.880994319915771/h.5；全队列统计后最大训练M完整前反向、最大开发M完整推理，默认Q0，明确OOM才精确Q1024等价检查并两组共用。
每组3完整epoch、178视频/轮、534更新、68634合法帧呈现；公共新初值、同视频顺序、AdamW1e-4/wd.01/clip1、原BCE+Dice，额外BN0。
每轮原5152开发评价，epoch3主比较，各自新增15开发AP选best补充；继承分类/54固定预览、四视频训练观察，不访问测试或重新人工筛选。
逐视频原子last保存模型/优化器/位置/RNG/身份，显式resume按完整视频边界恢复；失败保留日志，不无限重试/降级。
本轮Codex仅实现、元数据计划、CPU检查及一个已有小病例训练/恢复/评价接口短预检；全量prepare、极值与正式训练由用户终端启动。
仅在本独立脚本范围取代17.12的三病例/两步预算与禁止新增MoGe限制，不改原型默认或旧实验结果。


### 17.14 官方MumPy的ANAKIN定位迁移与脚本交付

ADR-0057登记一个独立MumPy基线：直接使用固定官方commit的完整三视图Encoder、DWTI、全局时间模块、DCT与多金字塔Decoder，不用BaselineEncoder或PVST自建注意力替代。
沿用89训练/19开发、22878合法训练目标/5152开发帧和94测试保护隔离；官方3帧夹取、监督中帧，40y1x9 fake6–9仅可作RGB上下文。
只读旧RGB/官方GT，允许独立224整图缓存；无MoGe/DINO/DFormer提取、XYZ/光流或分类支路。仅官方通用初始化，不加载旧取证权重。
10完整无放回轮、seed20260909；作者三组SGD/多项式调度、focal+softIoU，官方可选SingleAugment共享同片变换。micro2累积至有效12（官方collate实际批量），末批6目标不丢；1907更新/轮、总19070；FP32是本基线配置，不称原论文完整训练复现。
每轮开发/固定32训练观察，末次与新增15开发AP选择结果分列，原生连续logit计分及既定54预览。完整检查点含三个优化器/调度器与RNG，可独立评价/报告。
Codex本轮只代码、CPU检查及少量真实目标的可丢弃短预检；完整准备/10轮由用户终端或tmux启动，无后台长训练/持续轮询。旧PVST默认、协议与产物不变；本独立基线不替代XYZ方法，也不实现DETAIL_ONLY。
