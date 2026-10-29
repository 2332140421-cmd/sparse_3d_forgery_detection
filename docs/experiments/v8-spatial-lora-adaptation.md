# V8空间定位LoRA适配对照

2026-10-30，PREFLIGHT_COMPLETE；分支`v8-spatial-lora-adaptation`，从`9aaa029`开始。
本轮只训练一个条件SPATIAL_LORA_ADAPT，检验冻结视觉是否限制ANAKIN空间定位。
不运行三维增强、额外seed或前端；完整模型、输入、全部实际帧与正式损失保持。

[配置](../v8/spatial_lora_adaptation_config.json)，ADR-0026。
入口`research_tools.v8.pretrained_spatial_context.lora`；在原模块内新增单文件，调用现有
`official/prepare/patch_features/pixel_sums/evaluate/curves/sample_index`。
旧入口默认行为不变，不建立新框架或重复数据清单。

## 冻结设计

同一官方完整spatial checkpoint及revision/SHA沿用原配置；不加载旧best/last继续训练，
不随机重置已训练LoRA，不合并LoRA。原DINOv2参数冻结，rank4/alpha8的已有
qkv/proj/fc1/fc2 LoRA及官方decoder训练；LoRA/decoder AdamW lr为1e-5/1e-4，
weight_decay1e-4、全体可训练参数clip1。骨干及LoRA模块始终eval，
decoder训练Dropout正常，5层BN的三个buffers固定官方初始化，仿射可训练。

在线FP32骨干→可微FP16再FP32量化→BF16 decoder。RGB512/255、右下黑补6至518、
官方normalize、37×37 patch、输出裁回512、原图回映与内容排除复用。
训练和评价仅读取原RGB/GT/content缓存，旧patch.npy只用于初始化横竖屏核对，
不作为新模型训练/评分输入，不保存逐轮特征。非reentrant逐ViT块检查点，初始chunk2；
若显存不足，允许chunk1，保持分辨率和全部帧；仍不足则停在资源验证。

原ANAKIN train24pair/48视频6178帧，dev8pair/16视频1806帧，原身份/PTS/source隔离。
每完整pair一次更新：fake后real，各视频chunk即时反传`.5*chunk帧loss总和/该视频总帧数`，
最后不足chunk的帧按实际帧数计。全部pair结束统一clip/step；无GRU或跨帧可学习状态。
原每帧内容归一化BCE+非空GT Dice(eps1)，空GT只背景BCE；无检测损失。

seed20260909，原`Random(seed+epoch)` pair排序，最多30轮；dev Dice patience8/min_delta.002，
best严格最大Dice且epoch0可选；delta只控制patience。每轮完整dev、原ANAKIN monitor4pair，
固定为95qlg9/76rjm7/7p8r14/40y1x9，不冒充完整train；全部24pair继续训练。
每5轮dev pixel AP，best/last完整train/dev最终AP及逐视频指标；未算指标null。
阈值logit>=0、正GT帧先视频内再视频间均值、空GT FP单列、原图指标不变。

紧凑checkpoint仅LoRA、decoder及BN buffers、optimizer、RNG、实际pair边界、早停状态与身份；
恢复先载同一官方主干，校验身份后恢复。信号只在pair边界安全结束，不宣称COMPLETE。
测试/真实smoke通过后先commit和普通push，再独立tmux执行单Python入口；
自动训练→评价→固定25/50/75%样例→曲线→中文报告，Codex不逐轮轮询调参。

## 已有对照（只读复用）

实际summary核对，三行初始化相同。主要对照固定BN冻结视觉，原BN仅历史参考。

|条件|best/结束轮次|完整训练Dice|开发Dice|IoU|pixel AP|real FP|
|---|---|---:|---:|---:|---:|---:|
|固定BN冻结视觉|7/15|0.570987|0.245772|0.189581|0.350134|0.014186|
|SPATIAL_LORA_ADAPT|尚未训练|—|—|—|—|—|
|原BN历史参考|12/20|0.587574|0.274765|0.212297|0.349170|0.038293|

上一轮固定BN已COMPLETE：两模型各15轮/360更新、best均第7轮；增强Dice0.254395，
仅比固定迁移高0.008623，未超过原BN迁移。这里只引用，不重新评价、不改旧资产。

## 输出与完成审阅

输出：`/media/njupt/8FCD3869E2FAB06F/chenyh/data/sparse_3d_forgery_detection/derived/v8_spatial_lora_adaptation_pilot_v1/`。
报告须比较8条fake与原弱6条、GT覆盖/背景/固定边界、完整train与dev、monitor曲线、
last及实际成本。样例直接引用旧固定BN原始响应，不重跑基线。
smoke、外推预算和正式实测分开记录；最终结果尚未产生，不预填收益。
完成后集中审阅并提交小型结果和必要曲线；不追加实验。

## 启动前验证（实测）

28项相关测试通过（LoRA路径8、原预训练10、固定BN6、BN诊断4）。横竖屏在线初始化与旧缓存输出
均满足原容差：mean abs 0.001603/0.001780、max abs 0.031251/0.031250、阈值翻转
0.00001329/0。完整权重恢复沿用原严格键/shape/SHA/revision核对，无下载或依赖升级。

|训练pair|帧数|一次完整pair更新(s)|常规/精确AP评价(s)|峰值allocated/reserved(GiB)|
|---|---:|---:|---:|---:|
|横屏an:95qlg9|180|26.229|8.008 / 9.239|1.689 / 1.896|
|竖屏an:40y1x9|300|43.960|13.461 / 15.301|1.689 / 1.830|

LoRA/decoder梯度绝对和分别2080.23/496.57、1204.12/380.76，参数均实际更新；
原主干无梯度且不在optimizer，选定冻结张量精确不变。两pair的5层BN buffers均保持官方值，
保存重载预测最大差0；全部smoke更新丢弃。实际参数LoRA1,579,312、decoder2,094,017、
冻结304,372,736；24个ViT块采用非reentrant检查点，chunk2可行，无需降为1。

满30轮预算外推：训练27102秒（7.53小时）、monitor/dev3545秒、周期AP额外69秒、
最终best/last完整评价816秒，合计约8.76小时，未含checkpoint/日志I/O；早停可缩短。
这是两个完整pair的外推，非正式完成耗时。资源原始记录在输出`resource_smoke.json`，
初始化核对在`initialization_parity.json`；未新增视觉特征存储。


正式入口同时启用`--publish-results`：最终报告、旧资产只读核对完成后，自动将小型结果JSON、
一张必要曲线及完成报告同步到本文档，更新current_status，普通commit/push并核对远端HEAD。
仅允许当前任务分支且Git工作区干净；若期间有其他未提交改动，停止文档推送并保留现场，
不覆盖或暂存用户文件。`publication.json`记录实际推送commit；完成状态只在最终步骤结束后写入。
这只是本次已授权结果交付，不启动新实验。
