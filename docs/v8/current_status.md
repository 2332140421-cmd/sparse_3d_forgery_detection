# V8 当前状态与边界

更新2026-10-23。当前工作分支`v8-spatial-localization`，基线为远端
`v8-temporal-supervision-pilot` / `9a9b6b8efc41a0c752b070c440ab3c6a69a4a663`。
本页取代旧版“当前仍只有窗口队列、下一步先做局部几何”的过时状态说明；
以下历史实验与新机验证分开陈述，不改历史报告和模型。

## 当前方法与实现

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

## 当前资产准备任务

本轮仅完成Git认证、全量官方资产及关联real准备、标注核查和历史队列身份恢复。
不进行试标、训练、全量前端或模型评分。Git专用Deploy key已获授权，上一轮提交
`8c3ba1e`已实际push并核对远端一致；本轮最终提交另行记录。
完整官方发布集固定revision下载中，不能把5389 metadata目标数称为已得媒体数。
关联real目标3730：Charades976、ActivityNet842、HC-STVG v1 1912；后两者有访问缺口。
精确进度、完整性及人工缺口见[data_assets.md](data_assets.md)。
历史仅恢复14个开发验证source名称，缺完整V8视频身份manifest，未重新抽样替代244/28队列。

## 尚未完成及后续研究（本轮不执行）

**本轮环境和真实数据接口准备完成；空间定位训练尚未开始。** 没有空间真值或空间性能。
正式CELL_LOC/COMPONENT_LOC拟共享格均值压缩前的局部像素RGB/XYZ编码器与非整格常值
解码器，仅改变cell/component上下文；公平初始化和预算见
[spatial_localization_plan.md](spatial_localization_plan.md)。
前轮提出的“可见异常区域”8帧接口标注/状态检查及正式标注集准备均不在本轮执行。后续需冻结source隔离的
训练/开发/测试清单和完整初始化协议。需要重训时，两条件共同拟合训练统计、同阶段同预算
重新训练骨干，不能只适配COMPONENT。后续正式网络/超参尚未冻结。
本轮结束后不自动启动全库下载、全量前端、正式训练或历史全实验重跑。
