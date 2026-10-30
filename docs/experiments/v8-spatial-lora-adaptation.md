# V8空间定位LoRA适配对照

2026-10-30，COMPLETE；分支`v8-spatial-lora-adaptation`，从`9aaa029`开始。
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
|SPATIAL_LORA_ADAPT|12/20|0.664341|0.312192|0.246006|0.423351|0.032730|
|原BN历史参考|12/20|0.587574|0.274765|0.212297|0.349170|0.038293|

上一轮固定BN已COMPLETE：两模型各15轮/360更新、best均第7轮；增强Dice0.254395，
仅比固定迁移高0.008623，未超过原BN迁移。这里只引用，不重新评价、不改旧资产。

## 输出与完成审阅

输出：`/media/njupt/8FCD3869E2FAB06F/chenyh/data/sparse_3d_forgery_detection/derived/v8_spatial_lora_adaptation_pilot_v1/`。
报告须比较8条fake与原弱6条、GT覆盖/背景/固定边界、完整train与dev、monitor曲线、
last及实际成本。样例直接引用旧固定BN原始响应，不重跑基线。
smoke、外推预算和正式实测分开记录；最终指标见下文完成结果和小型结果JSON。
已集中汇总并提交小型结果和必要曲线；不追加实验。

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

## 独立启动

代码`7a3007e0dd81ac0adc06c82ce959aa2ab58240f6`已普通推送后启动。
启动时间`2026-10-30T00:17:54.478864+08:00`，tmux `v8_spatial_lora_adaptation_pilot_v1`。
无`--resume`，只训练新条件，自动结果交付已启用。日志为输出目录`run.log`，
实时状态`progress.json`，最终退出码`exit_code.txt`，实际Git交付记录`publication.json`。
训练、最终评价、报告和push均已完成，最终指标见下文；原进程在后续远端HEAD核验时因网络错误退出1，
事后核验及状态纠正见下节。

## 完成状态与真实退出历史（2026-10-30事后核验）

运行PID `1321201`，训练代码`7a3007e`，20轮/480更新，best第12轮；现有summary、
完成报告、结果JSON及日志核对为同一次运行。结果提交
`d32930d7ef98170639851fb3675d5c9f9deae728`已成功push；事后只读核验时本地与远端HEAD一致。
随后发生的是`git ls-remote`的HTTP/2 framing错误，不是训练或评价失败。

已仅将数据目录`progress.json`纠正为`COMPLETE/report_complete`，注明事后核验原因；
原自动流程未生成`publication.json`，现补记发布已完成及事后远端核验事实。
原`FAILED`状态（含完整traceback）逐字节归档为`progress.failed.remote_verification.json`。
`exit_code.txt`仍为 **1**，`run.log`及其中网络错误原样保留；不改称原进程正常退出0。
报告、指标、history未修改，修正前后SHA256一致；未调用训练、恢复、评价、前端或完整发布入口。
上述状态与失败记录均留在derived数据目录，不纳入Git。本次仅提交状态说明文档。

## 完成结果（自动集中汇总）

# V8空间定位LoRA适配对照

仅新训练SPATIAL_LORA_ADAPT；全部条件从同一官方已训练LoRA/decoder初始化。主要对照固定BN冻结视觉，原BN仅历史参考。
完整输入/全部帧/空间BCE+Dice不变；无三维增强，无检测门控。单seed和8条开发fake仅作描述性比较。

|条件|best/结束轮次|完整训练Dice|开发Dice|IoU|pixel AP|GT召回(视频均值)|real空间FP|
|---|---|---:|---:|---:|---:|---:|---:|
|固定BN冻结视觉|7/15|0.570987|0.245772|0.189581|0.350134|0.211642|0.014186|
|LoRA适配固定BN|12/20|0.664341|0.312192|0.246006|0.423351|0.313702|0.032730|
|原BN历史参考|12/20|0.587574|0.274765|0.212297|0.349170|0.308388|0.038293|
LoRA−固定BN冻结视觉：{"spatial_dice": 0.06642011533210179, "spatial_iou": 0.05642513557743467, "pixel_ap": 0.07321712937493552, "real_spatial_fpr": 0.0185433530613359, "gt_recall": 0.10206028809144124}

|开发fake|类型|冻结视觉Dice|LoRA Dice|差|
|---|---|---:|---:|---:|
|an:4xd3r9|inpainting|0.029942|0.066325|0.036383|
|an:4xd5r9|inpainting|0.178024|0.072950|-0.105075|
|an:7e2p57|splicing|0.000000|0.000000|0.000000|
|an:4rvro9|splicing|0.745436|0.825083|0.079646|
|an:4wgx37|inpainting|0.757326|0.779656|0.022330|
|an:71g867|inpainting|0.246958|0.520010|0.273052|
|an:7e2e57|splicing|0.002451|0.008093|0.005642|
|an:95q8g9|inpainting|0.006036|0.225418|0.219382|
全部8条：{"n": 8, "mean": 0.06642011533210182, "up": 6, "tie": 1, "down": 1}
原较弱6条：{"n": 6, "mean": 0.07156407107381135, "up": 4, "tie": 1, "down": 1}
inpainting：{"n": 5, "mean": 0.08921442379792559, "up": 4, "tie": 0, "down": 1}
splicing：{"n": 3, "mean": 0.028429601222395545, "up": 2, "tie": 1, "down": 0}
完整训练Dice变化=+0.093355，开发变化=+0.066420，best训练−开发差=+0.352149；曲线中的monitor仅为固定4pair，不冒充完整训练。

|LoRA checkpoint|集合|Dice|IoU|pixel AP|real FP|
|---|---|---:|---:|---:|---:|
|best/12|train|0.664341|0.548813|0.770824|0.023271|
|best/12|dev|0.312192|0.246006|0.423351|0.032730|
|last/20|train|0.738467|0.635142|0.821310|0.013313|
|last/20|dev|0.179891|0.141637|0.421892|0.015313|
固定25/50/75%原生2px边界统计：{"FROZEN_VISUAL": {"interior_recall": 0.25221351254724866, "GT_boundary_recall": 0.1138916929085945, "near_boundary_BG_FPR": 0.08833724174243185, "far_BG_FPR": 0.011641746781605934}, "SPATIAL_LORA_ADAPT": {"interior_recall": 0.33968173226910964, "GT_boundary_recall": 0.20868283875566657, "near_boundary_BG_FPR": 0.1753771011232012, "far_BG_FPR": 0.022395220236543473}}
冻结视觉完整fake覆盖/背景：{"video_mean_GT_recall": 0.21164190731260085, "pooled_GT_recall": 0.303813445513503, "pooled_BG_FPR": 0.0107119410449408}
LoRA完整fake覆盖/背景：{"video_mean_GT_recall": 0.3137021954040421, "pooled_GT_recall": 0.48279850071321323, "pooled_BG_FPR": 0.01874723999474341}
边界只代表固定时刻；完整GT覆盖与背景统计单列，背景误报下降本身不等于定位改善。样例索引见index.html，原始F32响应未裁剪。
实际20轮/480次更新，best第12轮，停止原因patience8_delta0.002；成本：{"training_loop_s": 20545.579268482514, "train_updates_s": 17979.06698576361, "routine_evaluation_s": 2489.174528995063, "final_evaluation_s": 1095.4648316120729, "peak_allocated_bytes": 1846627328, "peak_reserved_bytes": 1954545664, "parameters": {"lora": 1579312, "decoder": 2094017, "frozen": 304372736, "checkpointed_blocks": 24}}
smoke与满30轮预算外推见resource_smoke.json；正式成本单列，不将估计当实测。旧特征仅用于初始化核对；训练/评分全用当前LoRA在线FP32视觉、可微FP16量化、BF16 decoder。
报告前新增输出存储=252154037字节（checkpoint/必要固定样例/曲线/日志，无逐轮特征，无主干副本）。
唯一下一步建议：保留视觉适配基础，先扩充空间监督覆盖以检验开发收益的稳定性。
本轮收益只能归于视觉适配条件，不归于三维或component；不自动追加实验。
