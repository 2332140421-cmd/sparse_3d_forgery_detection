# V8预训练空间定位基础与三维时序上下文增强

日期2026-10-29。状态COMPLETE：05:26:09正常完成，两个条件最终评价、曲线、固定样例和报告齐全。
本次完成审阅仅读取已有结果，不启动、resume、重新推理或提取前端；仅更新相关文档。
基于审阅提交`1cafc51e01f3d0b668b99680522fa655f7320f82`建立
`v8-pretrained-spatial-context`，旧实验不变。

研究问题：同一预训练视觉特征、编辑区域监督和训练规则下，dense三维几何+
动态component全帧关系与因果时序上下文是否改善像素定位？
只有SPATIAL_TRANSFER和SPATIAL_3D_CONTEXT，见
[合同](../v8/pretrained_spatial_context_contract.md)与[配置](../v8/pretrained_spatial_context_config.json)。

## 完成结果：预训练基础可适配，完整三维增强未显示平均定位增益

以下均取已保存最终`epoch0.json`或各条件`summary.json`，与`result_summary.json`逐项核对。
轮次与计划中的best/12、last/20、best/4、last/12一致。同一行来自同一checkpoint，
不是各指标分别选优。Dice/IoU/AP按正GT帧先视频内、再视频间平均；
real空间误报为开发8条real的空间像素FP比例均值，所有数值均为比例，logit阈值固定0。
完整train48/dev16视频参与评价，未计算轮次的AP仍为null。

|版本|checkpoint/轮次|训练Dice|开发Dice|开发IoU|开发pixel AP|real空间误报（开发）|
|---|---|---:|---:|---:|---:|---:|
|官方预训练初始化|epoch0|0.182620|0.112233|0.088017|0.220394|0.013061|
|SPATIAL_TRANSFER|best/12|0.587574|0.274765|0.212297|0.349170|0.038293|
|SPATIAL_TRANSFER|last/20|0.583534|0.197140|0.158302|0.307962|0.019892|
|SPATIAL_3D_CONTEXT|best/4|0.474978|0.250421|0.193600|0.353893|0.028408|
|SPATIAL_3D_CONTEXT|last/12|0.520208|0.247409|0.174233|0.329781|0.090705|

各自best的增强−迁移：开发Dice −0.024344、IoU −0.018698、AP +0.004724、real空间FP −0.009885。
因此有轻微排序指标提升和较低背景误报，但固定阈值下重叠下降，不支持“定位整体改善”。
本比较遵守共同预算上限和选模规则，实际轮数及参数量不同；单seed/8条开发fake不能支持显著性判断。

### 8条开发fake：各自best的配对Dice

差值为SPATIAL_3D_CONTEXT−SPATIAL_TRANSFER，按未四舍五入的原始值判断方向。
类型沿用既有metadata，不凭编辑描述重新分类；身份和描述见冻结清单。

|pair_id|类型|迁移best/12 Dice|增强best/4 Dice|差值|
|---|---|---:|---:|---:|
|an:4xd3r9|inpainting|0.037143|0.004004|−0.033139|
|an:4xd5r9|inpainting|0.283025|0.246671|−0.036354|
|an:7e2p57|splicing|0.000125|0.011733|+0.011608|
|an:4rvro9|splicing|0.794990|0.755885|−0.039105|
|an:4wgx37|inpainting|0.746122|0.748379|+0.002257|
|an:71g867|inpainting|0.319217|0.209536|−0.109681|
|an:7e2e57|splicing|0.010128|0.020375|+0.010247|
|an:95q8g9|inpainting|0.007371|0.006785|−0.000586|

|范围|视频数|迁移均值|增强均值|平均差|提高/持平/下降|
|---|---:|---:|---:|---:|---|
|全部|8|0.274765|0.250421|−0.024344|3 / 0 / 5|
|inpainting|5|0.278576|0.243075|−0.035501|1 / 0 / 4|
|splicing|3|0.268414|0.262664|−0.005750|2 / 0 / 1|

全部视频的差值中位数−0.016863；逐一去掉任一视频后，平均差仍为负
（−0.029480至−0.012153）。最大下降来自71g867，去掉后总体仍−0.012153，
inpainting仍−0.016956，所以整体下降不是只由它造成。
splicing对单例敏感：去掉帽子4rvro9后，其余两条平均+0.010927；
但两者增强Dice仅0.011733/0.020375，不能把小幅增加称为可靠定位。
这些排除统计仅用于描述敏感性，不改变主指标、样本或选模。

### 位置选择性、覆盖与边界

官方初始化具有样本依赖的定位能力，尚不足以称稳定可用：开发Dice0.112233/AP0.220394，
其中4wgx37已达Dice0.726748，约占8条Dice总和的80.94%；其余7条平均仅0.024445。
所以初始化平均值不能代表广泛泛化。历史随机COMPONENT_LOC的dev Dice0.114502/AP0.119368
只作背景参照，因训练/监督不同，不能解释为单独预训练带来的匹配增益。

微调的两个best均超过epoch0：迁移dev Dice +0.162532，增强+0.138188；
AP分别+0.128775/+0.133499，且二者last的开发Dice也高于epoch0。
这支持迁移适配改善当前队列的位置选择性，但背景误报也高于epoch0，且部分编辑仍几乎漏检。

下表复用现有完整开发fake混淆计数与`fixed_boundary_statistics.json`，不生成新预测：

|best错误统计|SPATIAL_TRANSFER|SPATIAL_3D_CONTEXT|
|---|---:|---:|
|全部开发fake帧pooled像素GT召回|0.420440|0.350502|
|全部开发fake帧pooled背景FP|0.033763|0.013762|
|固定样例GT内部召回|0.329698|0.324916|
|固定样例GT边界召回|0.229516|0.187753|
|固定样例近边界背景FP|0.203768|0.176696|
|固定样例远背景FP|0.035373|0.024033|

pooled计数不替代主表的视频均值；边界仅指开发fake固定25/50/75%时刻的原生2px带，
先帧后视频平均，不是全队列边界精度。GT未修改。
增强主要呈现更保守的背景响应，同时损失GT召回；内部召回近似、边界召回下降，
没有证据说明边界精细度改善，更不能归因于纯XYZ或component的单独作用。

实际查看两张曲线、全部8条开发fake的best固定50%图、71g867的25%/75%图、
4rvro9/71g867的last中帧，以及原monitor前两条训练fake的best中帧。
4rvro9的帽子和4wgx37的移除区域有明显位置响应，但仍有边界欠覆盖或大块外溢；
7e2p57的小标记、7e2e57的猫和95q8g9的骑手区域仍有偏移/漏检，背景响应减少也未使其可靠覆盖GT。
71g867的中帧为空GT，此帧增强best较少误报不能代表整段定位提高；25%/75%时刻仍见欠覆盖，
last中帧则出现右侧大片背景响应。训练监测图也并非全部精细拟合，不能用训练均值概括所有视频。
上述画面判断只限已查看时刻，完整视频结论依据既有逐视频统计。

### 第4轮后的变化与泛化差距

增强模型轮末history如下（AP未计划计算时标为未计算，不与最终summary补算值混拼）：

|轮次|online总loss|轮末train Dice|dev Dice|dev IoU|dev AP|dev real空间FP|
|---|---:|---:|---:|---:|---:|---:|
|4|0.382198|0.474978|0.250421|0.193599|未计算|0.028408|
|5|0.350421|0.516662|0.244105|0.187251|0.338545|0.038376|
|9|0.270095|0.532279|0.250395|0.181208|未计算|0.063495|
|11|0.268469|0.311684|0.032048|0.020754|未计算|0.002791|
|12|0.235531|0.520209|0.247409|0.174233|未计算|0.090705|

第9轮Dice接近第4轮，但IoU较低、real FP更高；第11轮train/dev Dice同时下降，
伴极低real FP，符合响应过于保守而非稳定改进。第12轮Dice恢复接近best，但IoU下降、
最终AP由0.353893降至0.329781、real FP由0.028408升至0.090705。
online loss由0.382198降至0.235531，轮末dev总loss却由0.488734升至0.620110；
轮末train总loss也从0.360234升至0.407468，评价波动并非只有开发集出现。
因此证据是明显的泛化差距及训练/评价不稳定，而非仅凭“best第4轮/早停第12轮”证明过拟合。
现有数据不能分别确定小样本过拟合、decoder训练/评价状态或优化波动的作用，不将机制假设写成事实。

最终best的train−dev Dice差：迁移0.312809，增强0.224558；last差分别0.386395/0.272799。
增强较小的差距同时伴随更低训练和开发Dice，不能据此宣称泛化更好。
同轮history也不支持仅由选模轮次解释：第4轮迁移/增强dev Dice0.253254/0.250421，
第12轮0.274765/0.247409，均未见增强占优。

### 实际成本与核对边界

|条件|实际轮数|更新次数|累计pair更新秒|逐轮train/dev评价秒|最终best/last评价秒|active总秒|峰值allocated MiB|
|---|---:|---:|---:|---:|---:|---:|---:|
|SPATIAL_TRANSFER|20|480|666.55|701.66|429.43|1840.87|351.24|
|SPATIAL_3D_CONTEXT|12|288|782.21|582.62|460.34|1857.06|661.03|

两者均按patience8/min_delta0.002正常早停；每轮24pair各一次更新。active总秒包括
模型自身训练、评价及其他执行开销，不等于纯GPU训练时间；checkpoint写入/画图各自耗时未记录。
峰值为PyTorch allocated，不是含CUDA上下文/其他进程的整卡显存。
epoch0完整评价204.66秒；正式入口04:20:38.930至05:26:09.103，墙钟3930.17秒（65分30秒），
不把此前下载与特征阶段相加冒称单一墙钟。
冻结特征579.99秒/7984帧；权重成功续传阶段1527.90秒，前置探测完整耗时未记录。

报告生成前记录输出目录22,875,832,014字节（21.304779GiB），其中共享特征
22,384,844,800字节（20.847511GiB）；此外官方权重1,266,224,146字节（1.179263GiB）。
目录快照含新checkpoint、固定样例和日志，不含原约130GiB缓存；不是本次重新测量值。
外部代码及隔离依赖新增体积未记录，不重新运行补测。

汇总核对覆盖：result_summary与两summary/epoch0一致、逐视频均值回算主指标、
8条fake身份与manifest一致、轮数/24pair更新次数、best轮次与history最大值一致。
最终评价相对对应history的Dice/IoU/real FP最大差2.63e-7，迁移为0；
原始结果保留，本主表统一使用最终summary，而上述曲线表明确使用history。
本次不修改实验输出目录、训练代码、checkpoint、缓存或旧实验分数。

**唯一下一步建议：保留SPATIAL_TRANSFER定位基础，优先设计“固定已适配decoder（含评价状态）、
仅学习上下文残差”的匹配接入对照，检验新增上下文是否能在不扰动既有定位基础时提供收益。**
这是后续提案，尚无该对照结果；不据本轮宣称纯几何/component无效。本轮审阅提交后停止，不追加实验。

## 后续限定BN状态诊断（独立输出，不替换上述成绩）

随后用户仅授权两个best、固定训练pair95qlg9/40y1x9的BN统计模式诊断，
已一次完成，见[独立诊断](v8-pretrained-spatial-bn-diagnosis.md)。
同checkpoint只切换decoder BN后，竖屏fake Dice在迁移/增强上变化+0.096373/+0.018810，
横屏+0.009566/+0.003229；real误报并非一致改善。
因此当前下一步优先控制BN统计一致性，固定decoder残差仍为其后的候选；
未把批次统计结果替代正式成绩，未校准、训练或重提特征。

## 已完成准备

- 原joint_manifest全部ANAKIN身份冻结：[小型清单](../v8/pretrained_spatial_context_manifest.json)。
  train48视频/6178帧，dev16视频/1806帧，共32pair/7984帧；无重选。
- 原96训练统计/grouping、canonical缓存和已核验官方mask只读；未复制130GiB或重跑前端。
- 官方AINPAINT代码revision `753ae1b700b054c97e15569718690211e36cdf53`；
  官方DINOv2代码revision `7764ea0f912e53c92e82eb78a2a1631e92725fc8`。
  前者外部目录`workspace/external/AINPAINT-CVIU26`，后者`workspace/external/dinov2-ainpaint`。
- 官方[权重目录](https://drive.google.com/drive/folders/12p7chrWu7fY3_aObwd8_MfgmiAk0CIbX)
  中仅取[空间checkpoint](https://drive.google.com/file/d/1Wz4IZ046XTsikE6wPUc7_VDWxZJbzs9d/view)，
  文件`spatial_opn_sttn_noreal_loconly_v1.pt`，1,266,224,146字节，SHA256
  `6f6210b10a9b09aabd2dbb139e8797cd68e786a7591440af6d7dca4ba9e58cc7`。
  成功续传阶段1527.9秒；此前连通性尝试和13.6MiB部分下载未合并，不能作为全新下载吞吐。
- 真实checkpoint源epoch6，575定位必需键完整匹配，仅排除4分类头键。
  backbone305,952,048参数（其中训练过的LoRA1,579,312），decoder2,094,017。
  全骨干及LoRA冻结、eval、与新optimizer参数集合无交集。
- 横屏368×640/竖屏640×360中帧核验：原视频解码按原mapping到512与旧RGB缓存逐像素相同，
  目视无裁边/翻转；512补右下6至518后保留全部内容。
  同一次官方token重解码差0，重复CUDA骨干编码logit最大差0.000162/0.000295；
  FP16往返量化mean abs 0.00000135/0.00000149，max abs0.000077/0.000112，阈值翻转0。
  A/B零修正最大logit差0。实际落盘缓存的横竖屏real/edited四条核对也已通过。
- 原V8环境保持，隔离overlay `workspace/envs/v8-ainpaint-packages`：
  peft0.13.2/transformers4.44.2/accelerate0.34.2/huggingface-hub0.24.6/
  tokenizers0.19.1/safetensors0.4.5/regex2024.11.6，torch2.5.1+cu124不变。
  官方DINOv2本地结构可构建，无xFormers，未下载原始DINOv2权重。
- 27项相关定向测试通过（新路径10项、旧输出分离7项、旧联合路径10项）。
  覆盖严格键检查、边缘/真实尺寸mapping、GT不入模型、两次更新梯度、BN重算、
  跨块梯度/状态重置、恢复同序同参数、epoch0 best和报告口径。

## 已通过的真实资源smoke

[小型执行证据](../v8/evidence/pretrained_spatial_context_preflight.json)保留完整权重加载、
横竖屏real/edited四条实际缓存量化检查及两个条件更新证据。
64视频7984帧共享特征累计579.99秒、0.072645秒/帧，22,384,844,800字节（20.85GiB）。
未复制原缓存。新增权重与特征成本分开记录。

原首训练pair `an:95qlg9`，real90帧/fake90帧，两个条件各两次完整pair更新：

|条件|可训练参数|总参数（含冻结视觉）|pair更新秒（两次）|常规评价秒|精确AP评价秒|峰值allocated MiB|
|---|---:|---:|---|---:|---:|---:|
|SPATIAL_TRANSFER|2,094,017|308,046,065|1.461 / 0.975|0.481|1.834|357.8|
|SPATIAL_3D_CONTEXT|2,569,665|308,530,034|2.052 / 1.865|0.763|2.148|629.5|

增强新增支路483,969参数，其中8,321为不参与optimizer的旧检测head；
两条件冻结视觉均305,952,048参数，增强总冻结305,960,369。
decoder和P实际更新；第一步几何/GRU梯度0，第二步梯度绝对和0.002791/0.025179，
P末层第二步2.736927（裁剪后）。smoke权重已丢弃，正式重新构建配对官方decoder和零P。

按单pair实测外推最大30轮：A训练1254秒、常规评价639秒、额外dev AP约81秒；
B训练2017秒、常规评价1015秒、额外dev AP约83秒，合计约85分钟。
仅为单pair线性外推，不含最终完整best/last AP、epoch0 AP、样例导出与每pair checkpoint I/O；
原生分辨率、视频长度和共享服务器负载可影响耗时，早停可缩短。

## 执行及交付位置

独立输出绝对路径：
`/media/njupt/8FCD3869E2FAB06F/chenyh/data/sparse_3d_forgery_detection/derived/v8_pretrained_spatial_context_pilot_v1/`

历史执行命令（仅记录，本次不执行）：

```bash
PYTHONPATH=/media/njupt/8FCD3869E2FAB06F/chenyh/envs/v8-ainpaint-packages \
OMP_NUM_THREADS=4 \
/media/njupt/8FCD3869E2FAB06F/chenyh/envs/v8/bin/python -u \
  -m research_tools.v8.pretrained_spatial_context.run \
  --config docs/v8/pretrained_spatial_context_config.json --resume
```

正式运行前已通过完整定位权重加载、横竖屏官方forward/量化/零修正一致性、
两条件完整ANAKIN pair两次真实更新及评价smoke，再提交推送和独立tmux启动。
权重不能完整恢复则资产阻塞，不用随机decoder替代。

### 启动与完成记录

上述前置验证已全部通过。执行代码提交`d53fb5a914769ac1cfecb1d66157eb5d67d8d1d1`
已先普通push。北京时间2026-10-29 04:20:38独立启动：

```bash
tmux attach -t v8_pretrained_spatial_context_pilot_v1
```

日志绝对路径：
`/media/njupt/8FCD3869E2FAB06F/chenyh/data/sparse_3d_forgery_detection/derived/v8_pretrained_spatial_context_pilot_v1/run.log`

`tee`同时输出到tmux画面和日志，会话设置remain-on-exit，退出后保留画面。
同目录`launch.json`保存代码提交/命令/启动时刻，`progress.json`为当前执行状态。
启动先验证资产和已存在特征SHA，复用smoke证据，再完整epoch0评价、训练A/B、最终评价/报告。
05:26:09正常完成，`progress.json`为COMPLETE/report_complete，tmux退出码0；
本次已集中审阅实际早停、最终指标、曲线与固定样例，未重启进程。

Python已写`report_zh.md`、`result_summary.json`、`index.html`和固定原始F32 NPZ；
best/last完整train/dev Dice/IoU/AP/real空间FP、epoch0、参数/成本分列。
固定样例原生2像素GT边界带仅用于描述覆盖/背景/边界错误，不修改GT或训练loss。
训练中不逐轮Codex轮询，不追加模型或解冻LoRA。
