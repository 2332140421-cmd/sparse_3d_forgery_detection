# V8联合空间定位pilot：离线诊断

2026-10-27；分析基线`79bb392337e7d53a9aa5e1ca1d9665c8d8009cdc`，分支`v8-spatial-localization`。

最主要的两个瓶颈是：**空间响应缺乏编辑位置选择性，以及训练改善未转化为开发泛化**。
前者的直接证据是8条开发fake中7条Dice不足0.15、正常real仍有约19%～20%空间误报，
固定样例中real/fake均出现相似的大块背景响应；后者是两个模型最佳轮以后监测Dice继续上升而dev下降。
训练定位本身也仍弱，不能只归为过拟合，也不能只归为欠拟合。

**唯一优先建议：下一轮只解除“最终像素logit同时承担检测”的输出耦合**，共享完整结构、
RGB/几何、component和时序特征，使用独立检测头与像素定位头。先在预先固定的少量完整pair上
做原耦合/解耦的可拟合性匹配对照，控制成本；不同时改损失权重、温度、阈值或编码器。
这是最值得验证的机制假设，不是已经证实的全部失败原因。本轮没有实现或运行该探针。

核对范围内**未发现明确的帧序、尺寸映射、GT缓存或训练/评价聚合实现错误**。
面积归一化LSE带来的小目标牵制是源码和数学可确认的机制；它是否主导当前优化，需要上述对照验证。
暂不优先扩量；扩量不能直接解除目标牵制。后续若可拟合性改善而泛化差距仍在，再增加
独立source、局部编辑形态及场景覆盖，解决训练24个ANAKIN正空间视频代表性不足的问题。

## 产物与证据范围

外部目录：
`/media/njupt/8FCD3869E2FAB06F/chenyh/data/sparse_3d_forgery_detection/derived/v8_joint_spatial_diagnosis_v1/`

小型机器可读证据见[joint_spatial_diagnosis.json](../v8/evidence/joint_spatial_diagnosis.json)。
直接查看[固定样例索引](/media/njupt/8FCD3869E2FAB06F/chenyh/data/sparse_3d_forgery_detection/derived/v8_joint_spatial_diagnosis_v1/index.html)、
[CELL损失曲线](/media/njupt/8FCD3869E2FAB06F/chenyh/data/sparse_3d_forgery_detection/derived/v8_joint_spatial_diagnosis_v1/CELL_LOC_losses.png)、
[COMPONENT任务曲线](/media/njupt/8FCD3869E2FAB06F/chenyh/data/sparse_3d_forgery_detection/derived/v8_joint_spatial_diagnosis_v1/COMPONENT_LOC_tasks.png)、
[区域面积散点](/media/njupt/8FCD3869E2FAB06F/chenyh/data/sparse_3d_forgery_detection/derived/v8_joint_spatial_diagnosis_v1/area_dice.png)。

- `report_zh.md`：本报告副本；`index.html`：全部8条ANAKIN dev fake与对应8 real，三个固定时刻，共48行四列对照。
- `videos.csv`：两条件best/last、全部64条ANAKIN real/fake身份、类型、区域统计与已有指标。
- `*_losses.png`：online、固定monitor和dev的总/video/frame/pixel损失；`*_tasks.png`：Dice/IoU、两个数据集视频AUROC、帧AUROC/AP、real空间FP。
- `*_optimization.png`：已有平均梯度范数、学习率；`area_dice.png`：连续面积散点。
- `native_area_by_frame.json`、`mask_checks.json`、`lse_examples.json`、`audit.json`：统计与核验细节。

原实验保存两条件`history.csv/history.json`，`summary.json`中有best/last的逐视频结果、全部帧logit/标签、
逐视频native像素混淆矩阵和空间指标；`samples/`仅best的JPEG RGB/GT/二值预测拼图、时间JSON/PNG。
拼图RGB是原程序将512画布回映到原尺寸后保存的edited/real画面，不是重新解码的原生RGB像素。
没有原始像素logits/概率、逐帧空间分数或预测面积序列，也没有last的固定像素图。
因此不做阈值扫描、统一概率热图、逐帧空间相关性或last图像比较；JPEG只供看形状位置，不反推指标。

固定train_monitor是每库4 pair，即每库4 real+4 fake，**不是完整训练集**。
完整ANAKIN训练24 fake中只有4条有已保存逐视频评价，其余20条仅可统计GT面积，CSV指标为空并标明原因；
train_monitor没有保存pixel AP，CSV亦为空，不补算。配置/manifest/统计来自原实验，未改动。

## 训练学到了什么

下表按各自最佳轮→末轮，使用全部14/13轮曲线判断趋势。online是更新中不同权重的损失平均；
monitor/dev是在轮末同一权重下评价，两种数值不能直接混同。

|指标|CELL_LOC：epoch6→14|COMPONENT_LOC：epoch5→13|
|---|---:|---:|
|online总损失|1.5675→1.3839|1.5515→1.3042|
|online video / frame / pixel|0.7803/0.6250/0.4746→0.7187/0.6493/0.3406|0.7774/0.6004/0.4740→0.6494/0.5799/0.3649|
|ANAKIN monitor Dice|0.2400→0.3217|0.2615→0.3711|
|ANAKIN dev Dice|0.1205→0.0724|0.1145→0.0736|
|ANAKIN monitor IoU|0.1513→0.2256|0.1614→0.2640|
|ANAKIN dev IoU|0.0752→0.0450|0.0754→0.0426|
|ANAKIN monitor pixel loss|0.7530→0.5336|0.7843→0.4956|
|ANAKIN dev pixel loss|0.7508→0.7590|0.8085→0.7340|
|ANAKIN dev video BCE / frame BCE|0.8148/0.8244→1.0941/1.1351|0.7738/0.7846→0.9599/1.1932|
|AF monitor pixel loss|0.1362→0.0274|0.1197→0.0191|
|AF dev pixel loss（仅real背景）|0.1774→0.1942|0.1652→0.3029|
|ANAKIN dev real空间FP|18.85%→7.00%|19.83%→5.61%|

两者从首轮Dice0发展到非零空间学习；后续monitor改善、dev定位退化，末轮monitor-dev Dice差已达
0.2493/0.2975。旧报告仅引用best差距小于0.15，不能据此排除过拟合。
同时monitor末轮Dice也只有0.32/0.37，且只覆盖4个训练fake，不能声称完整训练集已经拟合充分。
pixel loss是BCE+Dice，未分别记录，无法判定各自贡献。尤其COMPONENT的dev pixel loss下降但Dice变差，
说明混合损失改善不等于正GT区域重叠改善；空背景BCE也会降低混合损失。

任务并非全部同向：AF dev frame AUROC分别0.5076→0.5387、0.7230→0.7759，
COMPONENT的AF video AUROC 0.6094→0.7188，但视频/帧BCE变差；排序指标与固定阈值/校准表现不同。
ANAKIN dev video AUROC分别0.5469→0.5469、0.5156→0.5781，仍不能说明定位改善。
所有epoch的两个数据集曲线均保留，未拼接不同轮次的任务最佳值。

lr始终1e-4。记录的是`clip_grad_norm_`返回的**裁剪前范数的epoch均值**，不是裁剪后范数：
CELL从首轮0.272到末轮18.596，COMPONENT从0.273到46.965；裁剪阈值1。
这说明后期出现较大更新梯度、裁剪有实际作用，不能据均值恢复被裁剪pair比例或断言数值爆炸。
已有训练数值有限，没有NaN证据，也没有各任务梯度余弦，不能宣称测得梯度冲突。

离线按每轮Dice重放best与patience规则：两者best为6/5，停止14/13，stale均8。
最高Dice保存best，只有超过patience参考值0.002才重置等待；早停符合冻结协议，无提前中断证据。

## 固定样例：具体错在哪里

已查看两个模型全部ANAKIN开发fake和对应real的25%、50%、75%固定样例。
以下每条观察仅指这三个时刻；二值图不包含概率强弱，不能推断其余帧。

|视频|fake固定样例的GT与预测|对应real及两模型差别|
|---|---|---|
|4rvro9，帽子|GT在头顶，两个模型主要响应下方衣服/栏杆附近及右下背景；帽子基本遗漏|real也有相似低处响应；COMPONENT右侧区域更宽，未改善帽子位置|
|4wgx37，移除骑车人|GT在画面中部偏左，预测多在更低、更偏右的地面；部分时刻接触GT但形状不合|real骑车场景也有地面误报；两模型面积和碎片不同，均未追随GT形状|
|4xd3r9，移除5人|GT是数个小人体区域，预测成右下方的大块区域，局部覆盖但明显外扩|real同处也有大片响应；两模型整体位置相近，不是只有细边界错误|
|4xd5r9，项链|GT在画面较低的胸前区域；预测碎片散布衣服、手臂及周边，多数不在GT内|real也有碎片；COMPONENT在中/后时刻外扩更多。沿用官方编辑区域，不把GT强改成项链轮廓|
|71g867，移除右侧进入的人|GT沿右侧及下边缘变化，预测常与之重叠但含大块背景，未精确贴合|real右侧/地面也有相似响应；COMPONENT在此视频重叠较好，这是8条中明显较好的一条|
|7e2e57，加入猫|GT是移动小猫，预测多为靠近地面的宽带/斑块，猫位置重叠很少|对应real无猫也有宽带响应；COMPONENT更连续、更宽，不能称成功跟踪猫|
|7e2p57，衣服美元符号|GT极小，缩略图近乎一点；两者主要响应下方/侧边大块区域，漏掉符号|real也有大块响应；这是小目标遗漏叠加背景误报，不是全图背景预测|
|95q8g9，移除骑手与马|GT是中部小形状，预测成路面/右下大区域，有重叠但严重扩张|real中也有大块路面响应；COMPONENT通常更宽，两者都缺乏编辑边界选择性|

直接量化不来自JPEG，而来自已保存native像素混淆矩阵：最佳模型的8条fake均有正像素预测，
其全视频像素预测正比例范围CELL为6.50%～40.79%、COMPONENT为6.19%～43.27%，
但8条视频logit全部小于0。real的空间FP均值18.85%/19.83%，其视频也全部判real。
**因此视频判real与像素有fake响应同时存在**，不能用“全部漏检”解释为全部像素输出背景。
这些像素比例为视频内全部帧的pooled像素计数，区别于下节正GT帧均值的面积和Dice。

## 区域大小、类型与逐视频结果

面积使用官方native尺寸mask，RGB按既有规则转灰度、/255≥0.5为前景；无letterbox padding。
正GT帧先求面积比例，再逐视频给中位数/范围；AP基线用**同一批正GT帧的面积均值**，
再按视频等权平均，与保存的pixel AP口径一致。常数响应的AP即该帧前景比例。
512缓存GT另做尺寸/映射检查，不混用两种面积口径。

32 fake共3991帧、3961正GT帧；29条全程非空，3条混合：train 40y1x9空4帧、71gz67空21帧，
dev 71g867空5帧。空GT是已知背景，不是缺标。没有把缺标AF fake算正常。

训练24条的每视频正帧面积中位数之中位数3.10%，范围0.119%～24.751%；
dev8条相应3.42%，范围0.0349%～13.902%。类型train为inpainting16/splicing7/swap1，dev为5/3/0。
训练有一个swap，开发没有；面积尺度与内容有差异，但未见足以单独解释失败的整体面积突变。
本次不删样本、不重新划分。

|dev ID|GT面积中位数|CELL Dice / pixel AP|COMPONENT Dice / pixel AP|
|---|---:|---:|---:|
|4xd3r9|1.531%|0.0684 / 0.1137|0.0740 / 0.0974|
|4xd5r9|5.942%|0.1395 / 0.1085|0.1013 / 0.0901|
|7e2p57|0.0349%|0.0000 / 0.0003|0.0000 / 0.0002|
|4rvro9|3.166%|0.0002 / 0.0342|0.0000 / 0.0275|
|4wgx37|13.902%|0.1423 / 0.2165|0.0805 / 0.2091|
|71g867|9.062%|0.4769 / 0.4598|0.5346 / 0.4607|
|7e2e57|1.587%|0.0076 / 0.0166|0.0187 / 0.0168|
|95q8g9|3.680%|0.1294 / 0.0699|0.1069 / 0.0532|

dev连续散点的面积中位数与Dice Spearman为0.881/0.707，n=8，只作描述，不作因果检验。
最小符号确实完全漏掉，但3.166%的帽子也几乎完全漏掉，13.9%的移除区域也只有0.08～0.14 Dice；
不能归结为单一小目标问题。两条件均7/8低于0.15，中位Dice0.0989/0.0772，
低均值是多数视频较差，且由71g867这一条较好结果抬高，并非少数坏视频拉低。
正帧/视频等权的前景比例AP基线0.04675，对应已保存AP0.12741/0.11937，存在一定排序信息，
但不等于定位合格。逐视频min/max、video logit、IoU、real FP及未监测训练缺口见`videos.csv`。

## 聚合与监督权重：可确认机制与未证实原因

源码`model.py:loss_sums/video_loss`：最终pixel logit在内容区域上double精度LSE减log面积得到frame；
跨块累加frame+log面积后，再减全视频总面积得到video。`metrics.py:evaluate_video`调用同一
`loss_sums`，使用同样的累计量。所有已保存best/last、monitor/dev记录的frame→video复核最大差
4.27e-14；训练/评价聚合一致。原始尺寸预测先双线性回映logit再阈值0，GT是最近邻正向映射。

设GT前景α，前景logit a、背景b，`z=log(α exp(a)+(1−α) exp(b))`。
固定a=2、b=−2，不调用模型：

|前景α|帧logit|背景−2时使帧logit=0所需a|
|---|---:|---:|
|0.5%|−1.7626|5.1537|
|1%|−1.5708|4.4613|
|2%|−1.2715|3.7697|
|5%|−0.6971|2.8581|
|10%|−0.1500|2.1727|

所以局部像素正确为正也不保证帧/视频为正。对正检测标签，BCE给所有参与像素的梯度方向都是
提高logit（LSE导数为正、且按exp权重分配），不只提高GT；像素背景BCE则要求背景下降。
在当前有限响应强度下，有抬高非GT区域或放大预测面积的优化牵制，但可以通过足够高的局部a满足各目标，
并非数学上绝对不相容。真实训练中的主导因素还取决于输出分布和任务梯度，本次无原始像素logits/梯度分解，
无法证明模型确实靠抬高背景降低检测损失。已有“视频负而空间大片正”“GT错位且real也响应”与该风险相容，
不构成唯一因果解释；随机初始化、样本局限和位置/外观捷径同样未被控制实验排除。

等尺寸视频的时间归一化形式相同：少量帧为正时，可再稀释。像素和时间两层等价于对全部
像素×帧归一化，不是额外重复扣除同一面积。ANAKIN本队列29/32全程正GT，时间稀释主要不能解释其失败；
AF只有区间内正帧，存在该风险。本轮不搜温度、top-k或阈值。

空间损失事实如下，按每视频已知帧平均，BCE在各帧内容像素平均；非空GT加soft Dice损失，
空GT Dice为0但BCE保留。Dice平滑常数1。未知GT跳过，不算空mask。

|每库24训练pair|空间GT|每次该库pair更新的pixel项|
|---|---|---|
|AF real / fake|real全背景；fake未知|0.5×BCE(real)，fake为0|
|ANAKIN real / fake|real全背景；fake编辑区域|0.5×BCE(real)+0.5×平均帧[BCE(fake)+非空Dice(fake)]|

每轮两库各24次更新交替，真假各0.5；在48次更新目标的平均意义下，AF real、ANAKIN real、
ANAKIN fake的pixel名义系数各1/4，AF fake为0。这是目标系数，不是声称48次AdamW更新等价于一个大batch。
video和frame真假平衡，各视频video系数1/4、frame系数1/8；视频长度不增加该视频总权重。
ANAKIN fake空间有正帧3961/3991是全队列计数，训练实际为3063正/3088帧，仍按视频等权，不能用帧数替代权重。

训练edited按每视频全部帧平均后的前景比例再视频平均为5.111%。因此像素BCE的标签系数质量约为
前景`0.25×0.05111=0.01278`、背景`0.5+0.25×0.94889=0.73722`，约57.7:1。
这个比值是标签与归一化系数，不是实测梯度比；Dice会强调前景，预测概率也影响梯度。
事实支持“背景BCE占比较强”这一机制风险，但不能仅凭比值就决定新权重；本次优先只解除检测/定位输出牵制。

## 核验、复现与下一步边界

诊断脚本`research_tools/v8/spatial_localization/diagnose.py`仅新增离线CLI，直接调用者为人工分析，
输出上述统计、图表和索引。已有报告未包含全mask面积、固定四列索引和聚合核验，所以需要此单一脚本；
复用既有mask解码、数值排序与映射，不引入通用分析架构。

32个官方mask ZIP的全部3991帧与已有GT512逐帧相等；帧编号连续、native尺寸、已知标记与正帧标签均通过。
共同标准化身份恰为全部96训练视频，不含dev。已有空间汇总由逐视频结果复现至1e-12。
这些核验不证明官方标注语义绝对正确，也未逐帧人工看全部媒体；未重新解码视频、读取模型权重、
扫描/哈希大型前端缓存或运行模型。现存样例未见统一翻转/缩放错位的直接证据，误报也出现在real中。

```bash
MPLCONFIGDIR=/tmp/v8-diagnosis-matplotlib \
  /media/njupt/8FCD3869E2FAB06F/chenyh/envs/v8/bin/python \
  -m research_tools.v8.spatial_localization.diagnose \
  --source /media/njupt/8FCD3869E2FAB06F/chenyh/data/sparse_3d_forgery_detection/derived/v8_joint_spatial_pilot_v1 \
  --output /media/njupt/8FCD3869E2FAB06F/chenyh/data/sparse_3d_forgery_detection/derived/v8_joint_spatial_diagnosis_v1
```

必要单测覆盖正帧面积/AP基线、LSE零点、best与patience区别、ZIP乱序及padding映射/损坏拒绝。
人工观察单独记录在本报告，不冒充脚本自动判图。本轮仅诊断，旧模型、配置、划分、统计和正式分数不变。
后续独立头仅为建议，尚未冻结新方法合同；不自动启动任何训练、前端或新推理。
