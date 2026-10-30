# V8 当前状态与边界

更新2026-10-30。当前工作分支`v8-spatial-lora-adaptation`，既有V8历史基线为远端
`v8-temporal-supervision-pilot` / `9a9b6b8efc41a0c752b070c440ab3c6a69a4a663`。
本页取代旧版“当前仍只有窗口队列、下一步先做局部几何”的过时状态说明；
以下历史实验与新机验证分开陈述，不改历史报告和模型。

## 已完成：空间定位LoRA适配

单条件SPATIAL_LORA_ADAPT已完成训练、最终评价和报告。实际20轮/480更新，best第12轮。
开发Dice/IoU/AP/real FP=0.312192/0.246006/0.423351/0.032730；
相对固定BN冻结视觉Dice差+0.066420。详见[完成报告](../experiments/v8-spatial-lora-adaptation.md)。
仅视觉适配，不归因于三维；旧资产只读核对通过，不自动追加实验。
结果提交`d32930d`已push并事后核对本地/远端一致。原流程随后在远端HEAD核验时遇HTTP/2错误，
原进程退出码1及错误日志保留；现仅将derived状态纠正为COMPLETE/report_complete，
归档原FAILED记录并补记事后publication核验，不重跑实验或改指标。

## 已完成：固定decoder BN匹配对照

用户授权ADR-0025/[固定BN合同](spatial_context_frozen_bn_contract.md)，两条件从官方初始化
重新训练，仅固定decoder BN三个buffers，仿射/卷积及Dropout策略保持。
原64视频7984帧、原96训练统计与已有视觉/前端缓存直接引用，不重提或复制缓存。
[本轮实验](../experiments/v8-spatial-context-frozen-bn.md)已通过20项相关测试与真实完整pair两次更新smoke，
代码`2eeb38d`正式运行已COMPLETE/report_complete，两条件均15轮/360更新、best第7轮。
固定迁移开发Dice/IoU/AP/real FP=0.245772/0.189581/0.350134/0.014186；
增强为0.254395/0.195870/0.353108/0.014478。固定BN内增强Dice+0.008623，
两者未超过原BN迁移Dice0.274765。上述只读结果供本轮对照，不重跑旧模型。

## 已完成：原BN策略预训练空间定位

2026-10-29用户授权[预训练空间定位与三维时序增强](../experiments/v8-pretrained-spatial-context.md)，
协议[pretrained_spatial_context_contract.md](pretrained_spatial_context_contract.md)、ADR-0024。
仅SPATIAL_TRANSFER/SPATIAL_3D_CONTEXT，全部原ANAKIN训练48/开发16，共7984帧；
原缓存/原96训练统计只读。官方AINPAINT完整定位权重已匹配，横竖屏官方forward与量化、
零修正一致性通过；64视频共享冻结特征已完成，两个条件完整pair两次实际更新及评价通过，
代码`d53fb5a`先提交推送后，2026-10-29 04:20:38在独立tmux
`v8_pretrained_spatial_context_pilot_v1`启动单入口，05:26:09正常完成，状态COMPLETE。
迁移基线20轮/480更新，best第12轮；增强12轮/288更新，best第4轮，均按patience8早停。
官方初始化dev Dice/AP为0.112233/0.220394；迁移best为0.274765/0.349170，
增强best为0.250421/0.353893。增强Dice差−0.024344，8条开发fake为3提高/0持平/5下降；
抑制背景同时损失GT覆盖，未见平均重叠或边界增益。inpainting/splicing均值差−0.035501/−0.005750。
训练与开发存在明显差距、后期评价波动，不能仅以最佳轮次提前证明过拟合。
最终指标、逐视频配对、实际成本及证据范围见[完成审阅](../experiments/v8-pretrained-spatial-context.md)。
该完成审阅原建议固定已适配decoder、只学习上下文残差，仍为后续候选。
上述完成结果审阅仅更新文档，未重启训练、推理或前端；后续限定BN诊断另述，不改历史分数。

随后授权的[限定BN状态诊断](../experiments/v8-pretrained-spatial-bn-diagnosis.md)已完成：
只用两个best与原训练2pair/4视频480帧，读取已有特征，仅切换decoder BN，Dropout始终关闭。
RUNNING_STATS复现最大偏差4.44e-7；原checkpoint/缓存/正式结果前后哈希一致，参数未更新。
BATCH_STATS使竖屏fake迁移/增强Dice从0.501870/0.432410变为0.598243/0.451220，
横屏仅小幅增加，增强real误报反而上升。输出统计敏感性不能忽略，但未证明BN为唯一瓶颈。
当前优先建议控制BN训练/评价统计一致性；本轮没有校准、改训练策略或启动新训练。

2026-10-27新增用户授权[输出分离对照](../experiments/v8-spatial-head-decoupling.md)，
分支`v8-spatial-head-decoupling`；原monitor16完整训练/原dev32，固定30轮，原缓存只读。
COUPLED与SEPARATE_HEADS均为完整COMPONENT_LOC，已COMPLETE，各30轮/240更新；原实验只读核验通过。
第30轮ANAKIN train Dice 0.535687/0.534415，dev Dice 0.052465/0.037654；分离未改善定位，
两库dev视频均被分离头判real。背景误报减少但GT召回也下降，不支持输出耦合为主要瓶颈。
下一步仅建议核查完整定位路径的小样本可拟合性，不扩大该修改，不自动启动新实验。以下原pilot成绩不变。

用户已授权ADR-0022：CELL_LOC/COMPONENT_LOC共享256结构与dense几何、512像素分支，
同初始化、数据及预算，唯一条件差异为成员分区；目标更新为编辑区域。
代码与固定队列已完成，两库真实完整pair在两个模型上均完成实际更新/评价，smoke权重已丢弃。
固定96训练/32开发视频、59,909帧；两模型各362,658参数。独立任务已COMPLETE，best/last评价及报告齐全。
CELL_LOC最佳epoch6、末轮14；COMPONENT_LOC最佳epoch5、末轮13，均按patience8早停。
ANAKIN最佳dev Dice为0.120542/0.114502，component差值−0.006039；未显示定位增益。
固定阈值下两模型均漏检全部8个ANAKIN开发fake；当前应先审阅固定预测图与曲线，不自动重训。
2026-10-27[离线诊断已完成](../experiments/v8-joint-spatial-diagnosis.md)：32条fake的3991帧mask/缓存GT对应通过；
固定样例多见背景响应，monitor改善而dev退化。下一轮仅建议验证输出解耦，不自动实施。
资源与启动说明见[联合定位pilot](../experiments/v8-joint-spatial-pilot.md)。
用户已确认本批ANAKIN mask位置和像素细节，取代旧粗矩形概括。
当前协议见[joint_spatial_contract.md](joint_spatial_contract.md)，旧结果保持原样。

## 既有V8方法与实现

纯V8主方法为GLOBAL_COMPONENT：全视频真实解码帧/原生PTS→256×256 letterbox→
MoGe-2 dense XYZ与CoTracker3 online短历史对应→16×16基础格→几何/近期运动规则驱动
动态component→冻结ImageNet ResNet18格RGB及MEAN_REPLICATED几何编码→同帧全域
attention→多前驱因果cell GRU→格响应→面积加权LSE帧/视频分数。
实现位于`research_tools/v8`，不是只看`src`或早期README。

component是四邻接规则分区，每个最多4格、可逐帧重组；不是网络学习的任意像素边界，
不是物理部位。主方法在学习前已经压缩格内XYZ均值，不能宣称保留了完整稠密表面细节。
几何缺失格保留，RGB与输出不删除；纯padding排除。既有格热图是弱监督响应，不是伪造mask。
V8授权范围以根契约第14节及V8合同/ADR为准，不套用早期V7粒子输入限制。

## 旧服务器已完成的历史结果

这些不是新机复现，也不意味着新机拥有对应checkpoint。

|实验|主要开发验证结果|边界|
|---|---|---|
|identityfix窗口FULL/RGB_2D三seed|source-macro AUROC 0.785714/0.761905|718训练/83验证窗口；旧修复前身份错配结果排除|
|同阶段全帧关系窗口COMPONENT/CELL|macro 0.896825/0.904762；pooled 0.856562/0.721835|801窗口，分区匹配对照，不能单凭macro宣称稳定增益|
|窗口模型的全视频冻结COMPONENT/CELL|AUROC 0.826531/0.714286；AP 0.857998/0.707924|同阶段对照，272视频全解码帧|
|COMPONENT_ADAPTED三seed，固定5轮|AUROC 0.872449，AP 0.904242，P 0.833333，R 0.714286，F1 0.769231，ACC 0.785714|只有COMPONENT适配，不与未适配CELL作因果归因|

历史全视频数据：244训练视频/122source/206779帧；28开发验证视频/14source/27505帧。
COMPONENT适配相对冻结AUROC差+0.045918，95%CI[-0.010204,0.112245]；
同一14source开发验证反复使用，不是全新独立test。视频弱监督和热图不证明空间定位。
原六模型20轮协议已暂停，不能写成6/6完成；三seed COMPONENT固定5轮已完成。

时间区间辅助监督pilot仅seed20260909、5轮：帧AUROC从0.764115到0.836800，
real平均帧FPR从0.004863到0.084656；视频AUROC均0.846939。它是时间BCE，
不是空间mask监督，也不是无代价提高。独立source/未见Vidu及DeepAction迁移评价已有
下降记录，不能声称已证明稳定跨数据集泛化。V9稠密/query路线历史约0.59—0.64且
query集中，本轮不恢复或修复V9。

历史细节：[全帧关系](../experiments/v8-component-global-context-pilot.md)、
[全视频冻结](../experiments/v8-full-video-global-context.md)、
[固定5轮适配](../experiments/v8-full-video-training.md)、
[时间pilot](../experiments/v8-temporal-supervision-pilot.md)、
[独立source](../experiments/v8-independent-source-test.md)、
[DeepAction](../experiments/v8-deepaction-cross-dataset.md)。
历史视频身份曾错配：新接口使用source+role+完整媒体路径及SHA；pair_id不能单独索引。

## 上一轮新机接口准备（已完成）

- 新建独立Python3.12.7环境，torch2.5.1+cu124/torchvision0.20.1；RTX4090 24GB。
- 下载固定MoGe-2代码、CoTracker3 online代码及utils3d，三个预训练工具权重记录完整SHA。
- 下载官方metadata/时间标注，核对5389 fake/3730source/6生成器/3来源数据族；未发现公开空间mask/框。
- 仅下载1LARL real/fake各1个；CPU完整解码各914帧，保留原生PTS，未提取完整视频前端。
- 固定4秒开始各4个连续帧：实际前端、动态component、两种随机模型前反向、padding与原图映射通过。
- 导出8张原尺寸图和身份/PTS/标注状态JSON；未标注显式null。10项定向测试通过。
- 提交小型执行证据及新机运行说明，不提交媒体/权重/环境/大型缓存。

环境与精确命令、下载身份、吞吐和成本见[new_server_setup.md](new_server_setup.md)及
[evidence/new_server_smoke.json](evidence/new_server_smoke.json)。
新机没有历史训练好的V8模型、旧standardizer/grouping、旧前端缓存或历史source划分清单；
随机smoke明确使用identity scaling与1/1/1 grouping常量，不能叫COMPONENT_ADAPTED或正式复现。

## ActivityForensics资产准备检查点（2026-10-23）

本轮仅完成Git认证、全量官方资产及关联real准备、标注核查和历史队列身份恢复。
不进行试标、训练、全量前端或模型评分。Git专用Deploy key已获授权，上一轮提交
`8c3ba1e`已实际push并核对远端一致；本轮最终提交另行记录。
完整官方HF发布集固定revision已实际验收：5395文件、5389 fake全部通过。
关联real尚在恢复，不能把官方发布集COMPLETE写成配对数据COMPLETE。
关联real目标3730：Charades976、ActivityNet842、HC-STVG v1 1912；Charades976和HC-STVG1912均齐全；ActivityNet798/842已通过容器核验，余44继续扫描历史v1.2包。
本次统一核验3686 real、5389 fake，完整性失败0；该计数是当前检查点，待剩余恢复后再次验收。
精确进度、完整性及人工缺口见[data_assets.md](data_assets.md)。
历史仅恢复14个开发验证source名称，缺完整V8视频身份manifest，未重新抽样替代244/28队列。

## 新机准备阶段的后续提案（历史记录）

**以下为新机准备阶段的历史提案；后续已完成的训练、结果和标签合同以上方及ADR-0022/0024为准。**
当时环境和真实数据接口已准备，尚未进行空间训练；ActivityForensics无可直接使用的公开空间真值，
ANAKIN候选的公开编辑mask核验见下节。其后已有的空间性能见上方，不能沿用当时“尚无空间性能”的状态。
正式CELL_LOC/COMPONENT_LOC拟共享格均值压缩前的局部像素RGB/XYZ编码器与非整格常值
解码器，仅改变cell/component上下文；公平初始化和预算见
[spatial_localization_plan.md](spatial_localization_plan.md)。
前轮提出的“可见异常区域”8帧接口标注/状态检查及正式标注集准备均不在本轮执行。后续需冻结source隔离的
训练/开发/测试清单和完整初始化协议。需要重训时，两条件共同拟合训练统计、同阶段同预算
重新训练骨干，不能只适配COMPONENT。后续正式网络/超参尚未冻结。
本轮结束后不自动启动全库下载、全量前端、正式训练或历史全实验重跑。

## 第二数据集资产核验（2026-10-26）

ANAKIN固定revision已取20对及mask，60文件通过校验；17对帧号/数量/尺寸可对应，
3对保留alignment_unresolved。场景预审13符合、4待确认、3不符合；与空间可对应取交集为11对。
标签是公开编辑区域，不能称可见异常GT；16对全程非空、1对空/非空混合，另3对未解决。
TVIL官方网盘尚未清点，空间mask是否公开及其对应仍未核实。
完整边界、场景逐项证据、成本和外部预览入口见[第二数据集核验](secondary_dataset_review.md)。
本轮无训练、评分、前端提取或正式split冻结。旧ActivityNet恢复已自然结束，未改动其资产。
