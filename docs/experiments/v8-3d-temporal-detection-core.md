# V8二维／三维时序结构匹配检测

2026-10-31；ADR-0028；分支`v8-3d-temporal-detection-core`，从已同步`10ff8b2`建立。
准备及真实完整pair实测已通过，尚无正式检测成绩。

研究问题：相同冻结视觉、数据、容量、时间机制与预算下，3D时序结构是否优于2D？
只运行STRUCTURE_2D/STRUCTURE_3D，各seed20260909、20260910、20260911。
从头训练，不加载历史窗口或定位权重；这不是历史0.872449的直接复现。

## 队列及输入

[唯一配置](../v8/temporal_detection_core_config.json)、[精确身份清单](../v8/evidence/temporal_detection_core_manifest.json)、
[原片预审记录](../v8/evidence/temporal_detection_core_scene_review.json)。
外部完整manifest保存全部原生PTS，其SHA及逐视频PTS摘要已冻结到小型身份清单。
实际train48/dev20/test20 source pair，共176视频137397帧；保留原AF24训练/8开发角色。
train96视频74578帧、dev40视频31584帧、test40视频31235帧；Wan/FCVG pair分别24/24、10/10、10/10。
每source一个real/fake，Wan/FCVG各集合均衡；源身份隔离，不称完整官方benchmark。
新候选顺序seed20260909先冻结，原片25/50/75%一次预审100条，84符合、16排除。
预审能核对固定时刻的构图/人物活动，不能证明未采样时刻绝无短暂切镜；不根据fake或分数选样。

旧完整122训练source清单缺失。测试严格使用官方test且source不出现在官方train任何编辑中，
并排除本机已知使用source及旧14开发source、接口source1LARL；独立性依据为历史官方train协议，
不宣称已恢复完整历史交集。下载和核验本身不作为测试排除条件。

全部正常实际解码帧、原生PTS；fake帧标签来自精确官方身份的闭区间并集，区间外为负，real全负。
旧V8 RGB/XYZ只读复用；缓存没有逐点轨迹，须补算缺失跟踪以获得同点3D位移，成本单列。
新视频共享一次MoGe-2、CoTracker3 online和冻结ResNet18 layer1，三个seed不重复前端。
不运行DINO、LoRA、ANAKIN测试或空间指标。

## 匹配模型及训练

复用GlobalContextModel、VideoTraining、原帧面积LSE和类别均衡frame BCE。
2D连续结构量是content内UV位置与当前/前帧同点二维位移，补零成3维；不读XYZ/geometry validity。
3D是cell XYZ与两个几何有效端点的同点三维位移；不将2D位移冒充3D运动。
两个条件均额外保留当前UV；无几何/无历史时保留RGB及当前位置、全部内容格和帧。
80维输入字段在model.py明确列出，隐藏128、4头全帧attention、最大4格component与GRU一致。
沿用V8 MEAN_REPLICATED，不将它写成保留格内稠密细节。2D/3D的component均由当前位置和近期运动驱动，
各自仅在train拟合尺度与同25%分位阈值，四邻接/上限/关联策略一致。比较整套结构表示，不拆解因果来源。

每seed公共可学习参数完全相同初始化；固定RGB，检测网络从头训练。每完整pair一次更新、real/fake各半；
loss=video BCE+0.5frame BCE，frame项沿用视频内存在的正负类别均衡。
AdamW lr/wd均1e-4，clip1、最多30轮；dev视频AUROC严格最高保存best，8轮不改善早停。
chunk16、非重入checkpoint；跨块hidden与梯度不detach，只有视频起点重置。
无置信度分支，无阈值/学习率/损失权重搜索。

## 评价与独立执行

每轮记录online loss、固定训练4source monitor、完整dev标准指标、LR/耗时/显存/best。
视频AUROC/AP/ACC/F1/Precision/Recall/混淆矩阵，pooled帧AUROC/AP/F1；阈值logit≥0。
同一行同checkpoint。各seed独立，mean±样本std(ddof=1)及同seed 3D−2D；不以集成替代。
六模型都训练完后，对各best在冻结test统一评价一次；last仅train/dev。
恢复文件保存optimizer/RNG/pair顺序及位置，复用原原子checkpoint逻辑；不建立任务框架。

独立输出：`/media/njupt/8FCD3869E2FAB06F/chenyh/data/sparse_3d_forgery_detection/derived/v8_3d_temporal_detection_core_v1/`。
正式入口：`python -u -m research_tools.v8.temporal_detection_core.run`。
代码及实测通过后先commit/push，再tmux独立运行单入口。Python自动完成共享准备、六模型、测试、曲线和报告，
实验COMPLETE先落盘，再单独记录publication；实际非零退出历史和错误不删除。完成后停止。

## 启动前验证与预算

6项本轮关键测试覆盖二维不读取三维字段、同点运动与缺失、运动参与分组、原生PTS标签、
等容量同初始化、跨块梯度/时间顺序/padding、实际mmap完整时间轴与末块dtype，以及保存重载。
[真实pair实测](../v8/evidence/temporal_detection_core_preflight.json)：两条件同一训练pair1478帧，
各完成一次真实更新，权重丢弃。参数均320353，2D/3D完整更新3.110317/2.687136秒，
读取0.147187/0.138862秒；评价0.625264/0.585345秒。
峰值allocated 525909504/528795136字节，reserved 570425344/585105408字节；
重载最大偏差1.04e-11/2.77e-11，最后帧到首块梯度2.12e-6/2.22e-6。
chunk16、非重入checkpoint冻结，不为性能修改模型。

按单pair线性外推：共享前端准备35572.24秒（9.88小时，旧几何复用pair参考，并非新MoGe精确实测）；
六模型最多30轮完整更新26327.89秒（7.31小时），逐轮monitor+dev2731.91秒（0.76小时），
最终best/last评价上界参考675.24秒（0.19小时）。实际早停、I/O、视频长度和共享GPU会改变耗时，
新视频额外MoGe和结构拟合成本须最终另报，不能把外推写成已完成成本。

准备阶段两次预检发现并修复新缓存的重复时间维和UV均值float64问题；实际mmap+模型前向回归测试已补齐。
仅重建本轮废弃结构缓存，观测复用。原失败日志preflight.log/preflight_retry.log、失败JSON及退出1历史留在外部目录；
最终预检preflight_final.log通过，退出0，不改写失败历史。以上为启动前记录，当时正式训练尚未开始。

本轮6项新增测试和复用时间监督／全视频训练／全帧关系相关测试合计23项通过；未测试无关训练或前端。

启动前历史交叉核对发现55NRK已有旧模型评分，将其排除并按原冻结顺序补入同生成器候选；
外部manifest_initial_before_history_check.json及manifest_amendment.json保留修订前身份与原因，本轮尚无测试预测。
场景预审84条通过中1条因历史评分失去测试资格；不因仅下载／核验排除source。
