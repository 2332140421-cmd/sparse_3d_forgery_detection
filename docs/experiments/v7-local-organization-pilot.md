# V7 历史局部组织匹配对照 pilot

状态：`LOCAL_TEMPORAL_GAIN_NOT_ESTABLISHED`；局部组织增益和局部有序判别也未达到预声明门槛。

本实验使用提交 `7a58bd109cefa9f87ae5bb3dcef0c4bbc38563c1` 下的实现，产物位于：

`/root/autodl-tmp/data/sparse_3d_forgery_detection/derived/v7_activityforensics_local_organization_pilot_v1/`

本轮只在已经物化的 289 点 Depth Pro `ParticleSequence` 上进行 CPU 侧关系重建和既有小模型 pilot；未重跑 frontend、tracking、depth、pose 或 GPU。

## 研究问题与边界

问题是：在同一 289 点观测和同一动态表示下，把历史大 component 限制为局部测量组，是否改善跨 source 的结构判别？本轮改变的是关系集合及其历史固定尺度，不是颜色显示或旧 `S(t)` 的精确复现。

complete-linkage 是已有的聚类思想；本实验检验的是历史局部测量约束的作用，不宣称聚类算法原创，也不把局部组解释为身体部位、语义对象、表面或物理刚体。

用户肉眼认为增密改善明显，但没有正式 ROI 覆盖证据；该观察不作为本轮执行门槛，也不被写成已经验证的定位能力。

## 冻结协议

- population 固定为 16 个 source、192 个窗口，复用已完成的 289 点 Depth Pro 轨迹、深度、内参、位姿、源帧和 PTS。窗口、anchor、component 配置和目标时间槽均不重选。
- 历史区间为 `timestamps_s < window_start_s + 0.5`。原 component 仍使用既有 `motion_coherent_components`：初始距离 `<=1.0 m`、中位相对变化 `<=0.05`（当前三维单位下的绝对量）、最小历史重叠 8、最小成员数 3。
- 在每个旧 component 内，从单点组开始做确定性的受约束 complete-linkage 式合并。点对必须同时满足旧直接边 `E_ij`，且历史共同有效帧上的 `dH_ij=max ||X_i-X_j|| <=0.30 m`；组间距离是所有跨组点对 `dH` 的最大值。相同距离用稳定 track ID 打破平局，不跨旧 component 合并。
- 组只由历史形成；评估期不重分组、不因异常拉伸拆组、不逐帧改历史尺度。小于 3 点的组保留在映射中并标为 `SUPPORT_INSUFFICIENT_GROUP_SIZE`，不强行并入其他组。三时刻共同成员不足时不填 0。
- 四个匹配臂使用完全相同的共同窗口、模型和训练预算：`G_C=旧 component+ORDERED_SECOND`，`L_C=局部组+ORDERED_SECOND`，`L_A=局部组+UNORDERED_STATE`，`L_D=局部组+PERMUTED_SECOND`。不运行 `ORDERED_FIRST`。
- MANIP real=0、fake=1；CTRL 不参与拟合，仅作描述性分数。source-disjoint LOSO，fold 内标准化和 source/class 平衡窗口权重，既有 weighted BCE、Adam 200 epochs、seed `20260909/20260910/20260911`。
- 主分数为三个 seed 的 OOF logit 先逐窗口平均，再按 source 计算 MANIP AUROC，最后对 source 等权平均。bootstrap 以完整 source 为单位，seed `20260909`、10000 次，四个臂使用相同重采样索引。

输入、支撑映射和协议的哈希记录在 `protocol.json`；其中 frontend 结果 SHA-256 为 `68beb854fab9540da297ec26b3359003a4f02238d7d995b023d33178e77d22ec`，上游 density pilot protocol SHA-256 为 `7d842c2cb0aa17fe131996d1ead21b1a252b1098563b898a4759bb3e3d32db58`。

## 支撑与局部组织结果

| 项目 | 数量/结果 |
| --- | ---: |
| 冻结窗口 | 192 |
| 旧 component 有效支撑 | 166/192 |
| 局部组有效支撑 | 165/192 |
| 共同主匹配窗口 | 165/192 |
| 完整 real/fake MANIP source | 14 |
| 有局部组映射的窗口 | 169 |
| 局部组映射行 | 11,985 |
| 保留组（成员数至少 3） | 5,814 |
| 支撑不足小组（成员数 1–2） | 6,171 |
| 保留组成员数中位数（最小–最大） | 4（3–59） |

局部规则没有增加可评分窗口，并使一个原本可评分的旧 component 窗口失去支撑；因此主比较严格使用 165 个两路径共同窗口，而不是把额外或失败窗口填成正常分数。局部 support manifest 中记录的无效原因总数为 `COMMON_VALID_MEMBERS_LT3=2,134`、`TARGET_FRAME_MISSING=201`；这是组/target 支撑记录数，不是伪造标签数量。

`coverage_by_source.csv` 保留全部 16 个 source，包括无有效支撑的 `0HV07` 和只有 CTRL 支撑的 `0CG15`。主 AUROC 的 14 个 source 为：`01KML, 04LAX, 0AGCS, 0BX9N, 0CGMQ, 0DVVD, 0ET8W, 0FM93, 0FO58, 0G2SC, 0KTWY, 0LNLR, 0PU21, 0QA8P`。

## 主结果

source 等权 AUROC 的点估计来自原始 source 结果，括号为同一 source bootstrap 的 95% 区间：

| 臂/比较 | 平均 AUROC 或差值 | 95% CI |
| --- | ---: | ---: |
| `G_C` | 0.575397 | [0.424603, 0.726190] |
| `L_C` | 0.496032 | [0.341270, 0.650794] |
| `L_A` | 0.626984 | [0.444444, 0.793651] |
| `L_D` | 0.563492 | [0.388889, 0.730159] |
| `L_C-G_C` | −0.079365 | [−0.269841, 0.119048] |
| `L_C-L_A` | −0.130952 | [−0.226190, −0.039683] |
| `L_C-L_D` | −0.067460 | [−0.166667, 0.027778] |

因此：

- `L_C` 区间下界没有超过 0.5，当前局部有序二阶表示没有建立跨 source 判别信息；
- `L_C-G_C` 跨 0，未建立历史局部组织相对旧 component 的增益；
- `L_C-L_A` 为负且区间不跨 0，当前数据反而不支持有序臂超过局部无序臂；
- `L_C-L_D` 跨 0，未建立有序结构超过排列破坏对照的稳定证据；
- 总体状态按预声明规则为 `LOCAL_TEMPORAL_GAIN_NOT_ESTABLISHED`，不是“方法失败”的普遍性结论。

辅助 pooled AUROC（不作为主结论）为 `G_C=0.615995`、`L_C=0.524420`、`L_A=0.556777`、`L_D=0.537851`。三个 seed 的窗口平均标准差分别为 `G_C=0.154285`、`L_C=0.084099`、`L_A=0.111820`、`L_D=0.091172`。CTRL 每个臂每个 role 有 42 个描述性分数；它们没有被当作确定的训练正负标签。

逐 source 的完整结果（AUROC 和配对差值）在 `evaluation/per_source_metrics.csv`。例如，`L_C-G_C` 为正的 source 包括 `01KML=+0.667`、`0DVVD=+0.111`、`0FM93=+0.222`、`0KTWY=+0.111`、`0LNLR=+0.333`、`0QA8P=+0.111`，但负向 source 同样存在，未形成稳定总体增益；不据少数 source 的方向宣称方法成立。

## 训练与泄漏核对

每个可评分 held-out source 都有 4 个臂×3 个 seed 的模型记录，共 180 条（15 个有 common support 的 source fold；`0HV07` 无支撑）。held-out source 的 MANIP 窗口没有进入训练、标准化、权重拟合或模型参数；`0CG15` 保留在总体但无主 MANIP 双类别，不能贡献 source AUROC。`evaluation/fold_support.csv` 记录每 fold 的训练 real/fake 数、source/class 权重和 weighted BCE 输入来源；训练 source/class 权重和在每个 fold 相等，窗口平均权重为 1。

没有 paired reference、source、generator、文件名、ROI、缺失率或其他标签进入数值输入。模型使用的仍是既有 `S(t)=[mean,std,p25,p75]`、时间戳感知的一阶/二阶状态和既有 MLP；本轮没有把局部组当成语义部件，也没有增加人工速度、加速度、jerk 或 residual。

## 固定案例可视化与产物

固定选择规则是按 manifest 的 source、MANIP/CTRL、real/fake 顺序，不按分数挑图。`visualizations/manifest.json` 记录 192/192 张成功生成的并排 PNG；左侧为旧 component，右侧为历史局部组，灰点表示未落入当前显示组，颜色只表示组编号，不表示真假或概率。它是组覆盖审查，不是像素级定位热力图。

主要产物：

- `protocol.json`：冻结参数、输入身份、哈希、训练和评价规则；
- `manifests/input_manifest.json`：192 个窗口的帧、PTS 和粒子 artifact 身份；
- `manifests/window_support.json`：旧/局部支撑、共同成员和无效原因；
- `manifests/local_group_mapping.csv`：逐窗口局部组、track 成员和支撑不足状态；
- `scores/oof_window_scores.csv`：四臂逐窗口、逐 seed 及聚合 OOF logit；
- `models/fold_models.json`：标准化、模型参数和 180 条 fold/seed 记录；
- `evaluation/fold_support.csv`、`per_source_metrics.csv`、`coverage_by_source.csv`、`summary.json`；
- `visualizations/manifest.json` 及固定案例 PNG。

## 结论与限制

1. 历史测量范围确实被限制：新组内所有保留成员点对都满足旧直接边和 `dH<=0.30 m`，且不跨旧 component。
2. 局部组并非普遍有足够持续支撑：保留组数量虽为 5,814，但约半数组只有 1–2 点而被排除，最终有效窗口为 165/192；这不是把缺失填零后的覆盖率。
3. 相同窗口上，`L_C` 没有超过 `G_C`：点估计下降 0.079，区间跨 0。
4. `L_C` 没有超过无序/排列对照；`L_C-L_A` 明确为负，`L_C-L_D` 跨 0。不能把本轮结果解释为二阶规律或时间因果结构已成立。
5. 性能变化伴随轻微但实际的覆盖损失（166→165；共同 165），局部规则没有带来覆盖收益。

本 pilot 只支持对当前 16-source 开发总体和固定 289 点测量的上述判断，不支持 full-video、sealed-test、未知生成器泛化、像素级定位、身体部位发现或真实物理连接结论。下一步若继续，应先针对支撑/表示缺口提出唯一可证伪补项；本轮不自动改尺度、加点、换模型或扩充数据。
