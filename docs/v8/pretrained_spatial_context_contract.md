# V8预训练空间定位与完整三维时序上下文合同

2026-10-29，用户授权；ADR-0024。本轮只检验整个增强模块的附加价值，
不将其差异归因于纯XYZ或component单独作用。

## 队列、资产与隔离

原joint_manifest全部ANAKIN：train24pair/48视频，dev8pair/16视频，全部7984实际帧。
沿用身份、SHA、PTS、官方编辑区域mask、split和技术排除。real为空背景。
无AF媒体更新、无test选模、无重新抽样。原96训练视频拟合的statistics/grouping只读。
原约130GiB缓存只读，不复制、不提取MoGe/CoTracker/ResNet，不读旧训练权重。

官方AINPAINT revision `753ae1b700b054c97e15569718690211e36cdf53`，
官方DINOv2 revision `7764ea0f912e53c92e82eb78a2a1631e92725fc8`。
只下载官方spatial_opn_sttn_noreal_loconly_v1.pt；不引入双RAFT。
加载完整定位骨干、已训练LoRA(rank4/alpha8/dropout0.05)、定位decoder；
仅明确排除encoder.head_features/encoder.classifier，不容忍其余缺失/多余键或尺寸错配。
SHA、权重完整性和官方forward必须实际验证；资产未验证前不启动正式训练。
兼容包放工作区独立overlay，不升级原V8环境。

## 输入与两个条件

RGB512/255，右/下各黑色补6至518，再官方ImageNet归一化；37×37 patch特征。
decoder输出518，再裁回左上512；原图连续回映后固定logit≥0。无504中心裁边。
GT沿用原nearest mapping；额外padding无损失/指标，原content mask保持。
G/C按每视频实际256/512 content_hw、pad_xy及37个patch中心连续对齐；不假设严格2倍。

SPATIAL_TRANSFER仅训练官方定位decoder。视觉骨干及已训练LoRA冻结、始终eval。
SPATIAL_3D_CONTEXT使用同一冻结骨干和逐参数相同的decoder初始化；
新增未训练dense局部几何(原7通道，7→32→32卷积)、完整GLOBAL_COMPONENT
(动态component消息、全帧attention、多前驱因果128GRU、实际delta_t)，
F_fused=F+P(concat(G32,C128))。P为160→128→1024，末层全零，初始像素输出一致。
原结构cell_head不进optimizer，未引入检测loss、分类logit、GT输入或预测门控。
geometry缺失保留视觉预测和监督；纯padding不入池化/attention。

冻结视觉骨干一次提取64视频FP16 patch缓存，读取转FP32，两条件共享。
记录checkpoint SHA、视频SHA、PTS SHA、形状、dtype、预处理和文件SHA。
先横/竖屏官方forward与缓存量化探针，再全队列提取，再核对实际缓存。
内容区量化限制mean abs≤0.01、max abs≤0.1、阈值翻转≤0.005，不能超标后静默放宽。
同一次官方forward捕获的decoder输入重解码须1e-5一致；另比较特征接口的重复编码，
特征atol/rtol1e-4、logit atol1e-3/rtol1e-4，分别保存实际误差。
原因是实测CUDA SDP重复FP32编码有约1e-5特征舍入差，同token重解码为0；
重复编码误差与FP16缓存量化误差分列，量化门限保持不变。

## 损失与训练

每帧内容内BCE+非空GT softDice(eps1)；空GT Dice=0且背景BCE保留。
先帧平均为视频loss，再pair real/fake各0.5。仅空间监督。
seed20260909；AdamW lr=wd=1e-4，clip1，BF16；既有scatter/attention FP32。
完整视频checkpoint跨块梯度，不detach；视频边界显式清零状态。
官方decoder的BN随训练更新；checkpoint重算临时克隆BN buffers，避免二次更新。
相同确定性pair顺序，最多30轮；dev Dice patience8/min_delta0.002。
best严格最高Dice、epoch0可最佳；min_delta只控制patience。保存last及完整恢复状态。
真实pair两次更新检查decoder/P更新及上游几何、GRU梯度；smoke参数丢弃。

## 评价、报告与入口

正GT帧先视频内再视频间平均Dice/IoU/AP；real及空GT背景FP单列。
每轮online与轮末train/dev指标分开，dev AP每5轮；未算为null。
epoch0、最终best/last在完整train/dev计算精确AP；不混拼轮次。
原monitor训练及全部dev固定25/50/75%导出原始F32 logits、GT、RGB、PTS和mapping。
共用已有静态索引，绝对色阶[-6,6]，二值阈值0；不按峰值选帧、不存全帧logit。
参数、冻结/训练/新增量、下载/特征/训练成本分列，单seed不作显著性推断。

入口`research_tools.v8.pretrained_spatial_context.run`：资产→特征→smoke→epoch0→
两个条件→最终评价/图/中文报告。`--phase assets|smoke`用于正式启动前验证，
`--resume`读取同身份checkpoint；训练前提交推送，再独立tmux执行，不逐轮轮询。
只完成本轮，不追加模型、数据、seed或LoRA微调。

新增精简子目录直接服务本对照，调用者为上述单入口，输出独立实验目录。
既有像素模型不具备预训练patch/decoder及518映射，不能仅改配置实现；
复用既有只读Video、训练checkpoint、V8 hidden接口、指标/可视化，避免复制通用框架。
