# ADR-0034：预训练RGB-D局部基础与切片关系增量

状态：2026-11-07用户授权实现、检查、拟合诊断及两个条件各一次正式训练。
分支pvst-rgbd-slice-relations，起点0570938aa600072a0cc780a897dc8f8eedef51c3。
本轮仅取代ADR-0033中新实验的冻结RGB骨干/随机XYZ编码、mean-max池化、单条件及旧256几何输入，不改旧代码、配置、成绩或失败历史。

## 问题与范围

RGBD_LOCAL和RGBD_SLICE_RELATION都使用深度；比较完整切片时空分支的增量，不是纯二维/三维，也不作等参数宣称。独立新代码research_tools/rgbd_slice按数据/模型/训练评价分三个模块；立即调用者为experiment，输出独立geometry、训练产物及报告。旧入口只能读已冻结RGB/GT/身份与复用PairRelation/terms/指标，不能提供旧模型或已训练特征。

## 输入与官方初始化

严格复用上轮manifest、源隔离ANAKIN8训练/4开发pair及48条首/中/尾24原生帧训练片段。无场景审查、无新数据、无测试读取。frame_label由已确认逐帧二值编辑mask是否非空定义，real为合法空GT。保留所有原生像素和PTS；无深度像素仍有图像定位/监督。

官方原版DFormer-S（非v2/++），源码e3273009b759b578945483828ff315d560be94c9；官方预训练目录Small文件Google ID为1n-eLQMlGaaJEiMqAUA7g6D2u_TA_AG7E，大小229183347，SHA256=8a086a3fb4e7ebda0806172938dd199bb7d8394593a8176640ed79735e35a0c0。epoch297的state_dict；保留编码器530键strict加载，RGB/深度stem各35键、交互270键，无missing/unexpected。分类pred及stages.3.1末端仅输出分类深度更新12键不在官方分割编码器中；CPU小图同权重补回该更新后，四尺度RGB输出与分割版一致。不是strict=False容忍主干随机。官方文件不改；mmcv-lite2.1.0/mmengine0.10.7等纯Python依赖隔离安装，既有PyTorch2.5.1/驱动不变，不使用其未安装的deformable算子。

通道64/128/256/512、stride4/8/16/32；编码器全体参与微调，预训练BN running stats固定、仿射可训练。RGB-D预训练代码12947c1ca521126fa882ad5e1859ac9333b8ac4b确认RGB顺序、RGB ImageNet标准化、深度通道mean.48/std.28（四通道预训练）；官方分割README的深度Greys_r逐图灰度适配沿用到MoGe。精确变换：valid有限Z逐图minmax→Greys_r八位灰度→/255→(x-.48)/.28；无效黑填充只供编码器，不是假观测。该显示映射不是物理深度，不宣称预训练数据分布与MoGe完全一致。

独立MoGe-2 vits-normal几何缓存仅24个现有视频，输入长边≤768、不放大小视频。保持宽高比，num_tokens=round(input_h*input_w/14²)，明确记录官方内部base网格×14（有少量patch取整，不人为改输出冒充高分辨率）。apply_mask=False保留结果并显式存mask，force_projection=True,use_fp16=True。每视频SHA、PTS、帧号、输入/内部/输出尺寸、权重SHA及partial计数可核验；仅缺帧重算。不复用256放大场。插值回原生网格仍是插值估计。slice_depth始终相机+Z/1.9888148307800293，h=.125,d0=0，绝对floor及空洞保留，非米。与encoder_depth分开。

## 共同路径与增量

共同路径：四尺度lateral256→逐级上采样+skip+3×3/GN细化，得到1/4局部256；再到1/2的64、融合原生RGB两层16维细节支路→原生32维空间卷积→pixel logit。独立帧头读取最后RGB-D特征均值（LayerNorm→256→logit）。不再是stride32插值+逐点MLP。

B：每个原生有效像素从局部256按像素中心双线性对应，score(F,规范XYZ)为4个独立片内softmax，V(F)4×128，加权汇聚C512；每头在同片全体有效点归一化，不互斥、不加面积或分区损失。分块精确softmax、分组求和与checkpoint是存储组织，所有点均参与。真实整数片号和原生pixel_to_slice映射保留，不截断远点。两组已有PairRelation（D512/8head）先同片严格历史PTS、再同帧不同片实际深度距离；不读未来，不加斜向边或新物理特征。空片不参与，无历史保留当前证据。

B的片上下文经512→32零初始化投影映射回原生细化特征，另有512→1零初始化帧残差。广播上下文而非最终真假标签；局部像素仍可预测不同。无帧标签门控像素。A/B共同local整份state_dict从一次生成的common_initial.pt严格复制并记录哈希；B附加参数单列，不称等参数。两条件训练中始终按固定策略，不detach/冻结应训练编码器。

## 检查、训练与评价

先检查实际GT/PTS/新geometry身份、floor及原生像素对应、全点学习式池化对稠密参考输出/梯度、官方末端分类分支不影响编码输出、共享初始化零残差输出、固定BN和两步上游梯度路径。CPU FP32容差atol2e-5/rtol3e-5，真实GPU零残差BF16对照atol/rtol均.002，运行前固定。

B真实含非空GT24帧两次更新；每步loss/梯度有限/存在、两步内核心分支至少一次非零、AdamW状态和参数变化。第一步零投影允许上游关系零梯度。检查权重丢弃。

随后A固定第一个合法训练pair的real/fake首24帧反复拟合，最多150次片段更新或2700秒（不搜索参数）。每次交替fake/real片段各一次更新；每10步评价两个固定片段，目标Dice≥.8且real FPR≤.05可提前停止。预算结束如Dice≥.6、GT召回≥.6且real FPR≤.05，则单列“明显拟合但目标未达”允许正式训练；这是运行前冻结的显著拟合判据，而非放宽GT或挑样本。未达到明显拟合则FIT_BLOCKED，不自动跑正式30轮；不能把基本全背景当学习通过。拟合权重全部丢弃。

两条件相同48clips及每epoch顺序、共同初始参数、loss、优化策略、预算。每片段全部24帧，frame BCE+pixel BCE+softDice逐帧均值、固定等权，Dice epsilon1含空GT；一次AdamW更新，A允许独立逐帧立即backward后累积，B整片跨帧梯度。lr编码器1e-5/新模块1e-4,wd.01,clip1,BF16；不搜索。逐帧checkpoint、可微CPU暂存、标准torch关系模块；不构建原生像素全连接图。最多30轮/最少10轮/patience8，best开发Dice严格提高，保存last、optimizer/RNG/配置/身份。

监测固定第一个训练pair真正反向首段，不混入未训练时段。开发完整合法帧，24帧因果窗口，尾段补左历史只评分新帧，常规边界重置，每帧一次。各自同一个best全任务评价；Dice/IoU正GT帧均值再fake视频宏平均，pixel AP正GT帧AP再fake视频宏平均；real平均像素FPR及编辑区召回分列。帧/视频标准AUROC/AP/P/R/F1/Accuracy及混淆矩阵，阈值logit≥0，视频frame log-mean-exp温度1。固定相同身份/PTS25/50/75%、色阶0–1，输出B−A及逐fake差值。开发全程编辑不能检验复杂边界。

相机运动与深度漂移仍是局限。两个条件/单seed，不加消融；如资源不能跑，最多一次等价执行调整后如实报告，不删点降维。计算与发布状态分开。旧文件和用户JSON不改。
