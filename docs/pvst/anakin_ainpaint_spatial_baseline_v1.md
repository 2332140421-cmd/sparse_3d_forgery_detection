# ANAKIN：通用DINOv2＋官方AINPAINT空间结构基线
**已完成：12,845次有效优化器更新、5次完整遍历，最终824个独立开发目标，退出码0。** 仅新增AINPAINT_SPATIAL一组，原冻结参照未重训，未接入PVST或增加其他模型。
宏AP从0.09388990升至0.18880705，Dice从0.10639928升至0.14617812；但四来源两升两降，改善主要集中在7ogeg4。猫添加样本下降，不能称为稳定、普遍的独立视频提升。
## 1. 同监督预算主表
AP：正GT fake帧的全原生FP32 logit排序→每视频均值→4来源等权宏平均，不合并real。Dice/IoU/召回/FPR统一logit≥0；fake非GT背景误报与real误报分别报告。
|指标|冻结DFormer＋PVST 12845|AINPAINT空间结构12845|新−参照|
|---|---:|---:|---:|
|像素AP|0.09388990|0.18880705|+0.09491715|
|Dice|0.10639928|0.14617812|+0.03977885|
|IoU|0.06332021|0.10519435|+0.04187415|
|编辑区召回|0.12160310|0.16673784|+0.04513475|
|real像素FPR|0.03017121|0.00055028|-0.02962093|
|fake背景FPR|0.02554721|0.00804422|-0.01750299|
|fake预测面积比例|0.02892135|0.01702063|-0.01190072|
匹配的是来源、监督目标顺序/曝光、全画面范围和计分口径；DINOv2与DFormer的预训练、容量、视觉前向和计算量不同。这不是XYZ有无单因素消融，也不是作者原始裁剪、loss和数据设置的完整复现。
## 2. 四来源完整结果
|来源|冻结AP|空间AP|AP增量|冻结Dice|空间Dice|
|---|---:|---:|---:|---:|---:|
|an:7e2e57|0.13466605|0.01781587|-0.11685018|0.18473124|0.00000000|
|an:95q8g9|0.02091876|0.04808098|+0.02716222|0.00000000|0.01074183|
|an:4xd3r9|0.01359695|0.01289294|-0.00070401|0.00329769|0.00000000|
|an:7ogeg4|0.20637783|0.67643839|+0.47006056|0.23756818|0.57397066|

|来源|条件|IoU|GT召回|配对real FPR|fake背景FPR|fake面积|
|---|---|---:|---:|---:|---:|---:|
|an:7e2e57|冻结参照|0.11084101|0.25518875|0.01968245|0.02090264|0.02464240|
|an:7e2e57|空间基线|0.00000000|0.00000000|0.00000000|0.00000000|0.00000000|
|an:95q8g9|冻结参照|0.00000000|0.00000000|0.02744741|0.01516223|0.01466607|
|an:95q8g9|空间基线|0.00618753|0.00729965|0.00000000|0.00020881|0.00040616|
|an:4xd3r9|冻结参照|0.00169123|0.00606544|0.02496672|0.02787862|0.02757722|
|an:4xd3r9|空间基线|0.00000000|0.00000000|0.00000827|0.00002139|0.00002093|
|an:7ogeg4|冻结参照|0.14074858|0.22515819|0.04858825|0.03824534|0.04879971|
|an:7ogeg4|空间基线|0.41458988|0.65965173|0.00219284|0.03194669|0.06765543|
7ogeg4的AP增量+0.47006056，约占宏AP净增量的123.8%；不含该来源的其余三条平均增量为−0.03013066，仅作敏感性说明，不从主结果剔除任何来源。95q8g9的AP也提高，但0.5操作点下召回只有0.00729965。7e2e57与4xd3r9的Dice/召回为0，不能由7ogeg4的高分掩盖。
宏real FPR和fake背景FPR均下降，同时宏GT召回有所上升，但召回收益几乎集中在7ogeg4；猫的预测前景面积为0。训练编辑以移除/背景补全为主，猫添加覆盖不足可作为解释候选，不能据此排除猫或认定唯一原因。
真实DFormer末级微调的同预算历史结果为AP0.06243255/Dice0.04881882，仅作背景，不加入本轮主模型训练或重评。
## 3. 实际模型与官方来源
官方AINPAINT revision：`753ae1b700b054c97e15569718690211e36cdf53`；官方DINOv2 revision：`7764ea0f912e53c92e82eb78a2a1631e92725fc8`。
源码：[AINPAINT](https://github.com/MMLab-unitn/AINPAINT-CVIU26)、[DINOv2](https://github.com/facebookresearch/dinov2)。读取并核对model_builder.py、modules/context_decoder.py、training_helper.py与scripts/train.py，直接实例化官方DINOv2和HighResContextHead。
只从官方通用dinov2_vitl14_reg4_pretrain.pth严格加载原DINOv2；LoRA及定位头固定seed20260909初始化。不加载AINPAINT取证成品、RAFT、DFormer特征或PVST权重。未构造未受监督的分类头，不报告帧分类成绩。
通用权重：`/media/njupt/8FCD3869E2FAB06F/chenyh/weights/dinov2/dinov2_vitl14_reg4_pretrain.pth`，SHA256 `36e4deffbaef061a2576705b0c36f93621e2ae20bf6274694821b0b492551b51`。
|参数范围|数量/设置|
|---|---:|
|原始DINOv2冻结参数|304,372,736|
|可训练LoRA|1,579,312|
|可训练官方定位头|2,094,017|
|可训练合计|3,673,329|
|LoRA rank/alpha/dropout|4 / 8 / .25|
|官方定位头dropout|.1|
|实际LoRA匹配模块|97|
匹配官方target_modules=[qkv,proj,fc1,fc2]：24个block各四处，共96；另包括同名patch_embed.proj卷积，共97。这来自官方PEFT集合的实际后缀匹配，不额外解冻普通投影。194个LoRA A/B张量＋22个head参数张量进入优化器；实际名称在model_scope.json。
逐帧T=1：工作RGB→ImageNet归一化→右下pad到14倍数→官方forward_features.x_norm_patchtokens（已排除CLS与四register）→严格H/14×W/14重排→官方HighResContextHead→pad尺寸logit→裁去右下padding→工作网格loss／原生bilinear恢复（align_corners=False）。
真实预检示例216×384工作RGB→224×392补齐→16×28空间patch、1024通道→224×392定位输出→216×384裁回。没有任意token截断、mask-aware crop、中心/随机裁剪或padding背景监督。
## 4. 数据、训练与核验
直接读取原source_expansion protocol/config/观测映射和订单。11训练来源：40y1x9、95qvg9、9l8w54、4qwgw4、4ym5n4、76rjm7、9mygy9、4ymrn4、4wge37、71gz67、4xeqd7；2569合法目标，各呈现5次，逐项等于原订单前12845。40y1x9_fake[6,10)不监督且本模型无需读取为历史。
原4开发完整824目标，412正GT fake、412 real，来源角色/PTS/原生GT保持；新旧计分身份hash一致。它们是已反复查看的开发集，不是最终独立测试集；没有保留测试访问。
观测包只选择工作RGB/GT与身份字段，不向模型传XYZ、valid、DFormer视觉或历史；现有混合包的反序列化不等于使用其中全部观测作为模型输入。新增MoGe/DFormer提取均0。DINOv2每目标在线运行，LoRA计算不在no_grad中，没有最终视觉特征缓存冒充训练。
新AdamW lr1e-4/wd.01、clip1、BF16前向/FP32 loss，pixel BCE+soft Dice各1/epsilon1；batch1、无梯度累积、无类权重、调度器、增强或超参搜索。LoRA和head正常train/dropout，开发统一eval。官方DINO默认DropPath为0，使用官方无xFormers的SDPA路径（XFORMERS_DISABLED=1），不修改官方源码。
按官方training_helper逻辑，每轮训练后使用该轮预先固定订单前100个训练目标，train/no_grad更新BN，不reset统计，不读取开发校准；合计500次额外前向。该列表可由冻结orders精确恢复，另导出BN_statistics_orders.json。
4项关键测试全部通过：完整画面及padding裁回、patch数量/顺序、官方头梯度/BN、PEFT冻结与适配状态恢复。真实预检3次副本更新后，97个A、97个B和22个head参数张量确有变化、梯度有限，原DINO未变；丢弃全部检查更新并恢复公共固定seed初值和空优化器。
完成后在CPU重新构建官方通用模型，严格核对LoRA/head状态键和形状并恢复final.pt。原始DINO参数hash不变，5个head BN计数均13345=12845优化前向+500统计前向，BN缓冲区正常改变。216个优化器状态step均12845，随机状态完整，24个固定预览与所有链接存在，无本实验残留runner。核验见verification.json。
## 5. 固定训练观察与在线loss
观察仅每视频约1/3、2/3固定目标：共同28帧和新增16帧分开，不代表完整训练集。每轮统计BN后评价，不用观察/开发选择best。
|完整遍历|共同28 AP|共同28 Dice|新增16 AP|新增16 Dice|
|---|---:|---:|---:|---:|
|1|0.90139134|0.80207324|0.79148217|0.68885378|
|2|0.94042745|0.85314957|0.89403697|0.74978012|
|3|0.95062920|0.78690773|0.82788178|0.69385758|
|4|0.98171196|0.86710769|0.91945126|0.80346541|
|5|0.98706981|0.87492935|0.96085626|0.78714437|

|完整遍历|平均在线pixel BCE|平均在线soft Dice loss|总loss|
|---|---:|---:|---:|
|1|0.14667677|0.81818323|0.96486001|
|2|0.02782781|0.64473564|0.67256344|
|3|0.01803386|0.55262536|0.57065922|
|4|0.01622496|0.11273962|0.12896458|
|5|0.01270595|0.05946501|0.07217096|
在线loss是更新过程的均值，与固定权重观察指标不同。训练观察AP较好，但独立开发大部分来源仍较弱；没有完整训练集评价，不用观察集代填完整训练成绩。
## 6. 下载、准备与实际计算成本
复用已有官方代码及v8-ainpaint-packages中的PEFT0.13.2等隔离依赖，原Python/Torch2.5.1+cu124不变；未安装作者整套requirements。新增下载仅官方通用DINOv2权重，不下载AINPAINT数据、取证checkpoint或RAFT。
|成本项目|实际值|
|---|---:|
|新增权重下载|1,217,607,321 bytes（约1.134 GiB）|
|下载耗时|约5分钟，准备过程仅保留分钟级起始记录，未独立秒级计时|
|训练优化更新/监督目标呈现|12,845 / 12,845|
|训练DINO前向|12,845|
|额外BN统计DINO前向|500|
|固定训练观察DINO前向|220|
|开发DINO前向|824|
|正式DINO前向合计|14,389|
|另计丢弃的预检DINO前向|4|
|新增MoGe / DFormer|0 / 0|
|训练更新累计|383.01s|
|BN统计累计|5.13s|
|训练观察累计|7.77s|
|最终开发评价|28.58s|
|训练/统计/评价墙钟（含I/O）|430.81s|
|准备模型/预检至最终预览主流程|444.16s，不含下载及代码准备|
|峰值allocated / reserved|2.311 / 2.391 GiB|
现有冻结PVST参照完整指标直接读取原12845步JSON；固定24图像复用上轮已核验的FROZEN_REFERENCE_12845图，不增加参照前向或重训。二者视觉计算预算不同，不把相同更新数写成相同墙钟/容量。
## 7. 产物、终态与结论
实际worktree：`/media/njupt/8FCD3869E2FAB06F/chenyh/projects/pvst-geometry-dataset-screen`；分支pvst-anakin-ainpaint-spatial-baseline-v1；执行提交`2a77801b19ea965fe138b6e50581bbc019c06bca`。
结果与中文报告：`/media/njupt/8FCD3869E2FAB06F/chenyh/data/sparse_3d_forgery_detection/derived/pvst_anakin_ainpaint_spatial_baseline_v1/report.md`；仓库报告：`/media/njupt/8FCD3869E2FAB06F/chenyh/projects/pvst-geometry-dataset-screen/docs/pvst/anakin_ainpaint_spatial_baseline_v1.md`；固定预览：`/media/njupt/8FCD3869E2FAB06F/chenyh/data/sparse_3d_forgery_detection/derived/pvst_anakin_ainpaint_spatial_baseline_v1/preview/index.html`。
final.pt保存完整LoRA、head及BN缓冲区、AdamW、更新位置与Python/NumPy/Torch/CUDA RNG；冻结通用骨干按记录SHA复用，不重复存入每个checkpoint。配置、官方revision/权重身份、原订单、逐步loss、观察、逐帧/来源CSV、恢复核验及实际命令run.sh均保留。概率0–1与0.5二值统一显示，使用原24个首/中/末目标，不按效果选图。
计算COMPLETE、实际退出码0；Git发布状态另存publication.json，不覆盖计算状态。大权重、视频、缓存和逐帧产物不提交。
本轮支持把这套视觉表示与掩膜定位训练链路作为值得借鉴的空间能力参照，但提升高度依赖7ogeg4，不能视作四来源稳定提升，也不能据此认定XYZ/时空关系无效。优先方向仅是保留该视觉定位链路供后续空间底座讨论；本轮不接入C、不增模型、seed或预算，发布后停止。
