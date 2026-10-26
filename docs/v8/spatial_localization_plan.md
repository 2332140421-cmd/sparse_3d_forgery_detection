# V8 空间定位准备与下一阶段方案

2026-10-23；本轮只准备接口，不实现或启动定位训练。已发布官方时间监督不等于空间监督。

## 标注目标和人工入口

本轮导出目标固定为 **可见伪造异常区域**：在所选原始 fake 帧中，人可确认的视觉异常像素。
不标完整编辑区域，不将被删除但不可见的物体虚构成空间mask；时间区间内也允许正常空mask。
无法可靠确定边界或异常归因时标 `uncertain`，不能强迫整帧为正。
若后续改研究目标为完整编辑区域，需独立协议与新标签，不能混用。

`smoke.py`在看模型响应前按固定规则选从4秒开始4个连续原生解码帧，real/fake各4张，
导出原尺寸PNG和 `annotations.json`。这是接口开发样例，不是正式标注集、独立测试或代表性抽样。
正式标注集需先按source冻结train/dev/test，再按视频PTS均匀固定取帧；可预先定义区间内外
分层，但不可按热图或异常观感挑帧。保留失败/不确定项和选择清单，不能根据前端成功率换样本。

用现成CVAT语义分割工具或图像编辑器在原尺寸PNG上人工标注，导出单通道0/255二值PNG。
不要让工具缩放原图；背景0，异常255。为每行填写相对 `mask_path` 与状态：

|状态|mask文件|评价处理|
|---|---|---|
|unannotated|必须null|未标注，不参与|
|uncertain|必须null|无法确定，单列，不参与|
|normal_empty|必须有人工确认的全0 PNG|正常空真值，参与|
|annotated|必须有至少一个255像素的PNG|异常真值，参与|

身份字段包含source、role、完整媒体路径、视频SHA、解码序号、PTS ticks/time_base/秒、原尺寸。
以 `validate_annotation(row, export_directory)`读取；不会创建占位mask，不按真假自动造标签。
`mask_to_canvas`用最近邻，padding监督/评价权重为0；`to_source(...,semantic_mask=True)`
裁去padding后最近邻返回原图。响应图用双线性仅作显示。坐标基于实际round后的content宽高，
不能只用名义scale反变换。缩到256会丢失细边界，回变换不能恢复这些信息。
首轮人工相互复核边界和正常空mask，不把real/fake像素差或模型预测当真值。

## 两条件结构：待标注就绪后实现

科学问题：相同真实局部RGB/XYZ信息和监督下，动态component上下文相对固定cell上下文
是否改善可见异常定位？下一阶段只实现 `CELL_LOC` / `COMPONENT_LOC`，不恢复V9 query。

共同输入是256×256 letterbox RGB、MoGe真实dense XYZ、geometry validity和content mask。
轻量卷积像素编码器在格均值之前读取这些输入：RGB与几何分支各两层3×3卷积、32通道；
几何无效坐标显式0占位并带有效性状态，RGB路径和像素输出保持；各层屏蔽padding。
拼接形成64通道局部像素特征。现有冻结ResNet18 layer1格特征和MEAN_REPLICATED路径
继续产生128维cell状态，区域attention后接原多前驱因果GRU；提取GRU hidden作为上下文。
唯一条件差异是singleton cell与规则驱动dynamic component的区域成员分区；attention、
cell时间路径、hidden128、初始化、训练预算完全相同。component不是网络学出的像素分区。

将128维时序上下文广播回16×16格内，与64通道像素特征融合，经共同3×3卷积和1×1输出。
可用stride2卷积后双线性上采样至256的轻量解码器；相同skip保留局部边界。
最终logit在格内可变化，不把输出限制为整格常值；不用全像素全局attention或扩大hidden到512。
具体解码器层数、loss权重和预算在标注规模明确后预注册，本文不把候选当成已完成方法。

仅对已标注内容像素计算共同BCE+Dice；可保留视频级面积LSE辅助项，但系数要提前固定。
未标注帧不复制视频标签为像素标签；uncertain不作为负类。主要空间指标IoU、Dice、pixel AP，
正常空mask的IoU约定、阈值、像素采样和source聚合在正式评价前固定；不能报告时间AP@tIoU
冒充空间AP，也不在没有实例定义时报告实例AP@IoU。只在dev选阈值，test一次冻结评价。

## 新机无旧权重时的公平初始化

本机只有MoGe/CoTracker/ImageNet预训练工具，没有旧V8检测checkpoint、standardizer或缓存。
随机接口模型不是COMPONENT_ADAPTED。正式实验先从本机重新提取且核验的训练数据拟合
共同standardizer/grouping，仅用训练source；冻结并记录hash，不能复用本轮identity smoke常量。

建议先以相同seed、相同初始化参数、相同source顺序和预算重新训练GLOBAL_CELL与
GLOBAL_COMPONENT窗口骨干，再各自同预算完整视频适配，不能只适配COMPONENT。
随后两定位条件使用对应同阶段骨干；新像素编码器/解码器按同seed逐参数相同初始化。
若选择从随机V8骨干直接联合训练，也必须两条件都如此并另立协议，不能冒称历史适配模型。
历史122/14 source清单不在Git中，需重新取得明确清单或冻结新的source隔离划分；
新划分的数值不得称历史复现。ImageNet和前端预训练权重可以共同复用。

## 下一步人工任务

1. 查看8张接口导出帧，确认“可见异常”目标可操作，完成状态与原尺寸mask；若都不确定，保留该事实。
2. 冻结正式source清单及历史使用记录，确定标注预算、双人复核和test隔离；同source各生成版本不跨split。
3. 标注就绪后冻结两条件协议，再授权必要的骨干重训和定位训练资源smoke。本轮不自动继续。

## 第二数据集核验结果（2026-10-26，未冻结训练协议）

ANAKIN公开标签表示编辑区域，与上文人工入口的可见异常区域不同，不能混用或重命名。
本轮只核验候选资产，不改变前述人工标签合同、V8方法、模型或loss；任何未来联合训练目标
仍需单独冻结。17/20对mask结构可对应，11对同时初审符合场景；仍需全段播放器确认，
粗区域GT和外观装饰分布不能直接证明人体结构定位适用。TVIL空间资产尚未核实。
见[核验报告](secondary_dataset_review.md)；后续不自动扩大下载或启动训练。
