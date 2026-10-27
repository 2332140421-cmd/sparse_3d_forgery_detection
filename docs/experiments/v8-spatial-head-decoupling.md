# V8检测／空间定位输出分离对照

2026-10-27；分支`v8-spatial-head-decoupling`，起点诊断提交`0964236`。
当前：**COMPLETE**。2026-10-28核验两条件均30轮、各240更新，最终评价、曲线、原始响应和报告齐全。
冻结协议：[head_decoupling_contract.md](../v8/head_decoupling_contract.md)、ADR-0023。

完成结论：第30轮训练Dice 0.535687/0.534415、开发Dice 0.052465/0.037654（耦合/分离）。
分离未改善定位；背景误报减少但GT召回也下降。暂不扩大该修改，优先核查完整定位路径的小样本可拟合性。

本轮只改COMPONENT_LOC的检测输出路径，检验共享像素响应是否限制空间学习。
COUPLED从pixel做检测LSE；SEPARATE_HEADS从共享GRU hidden经独立head和cell面积LSE检测。
像素coarse/residual、256结构/真实XYZ、512RGB及skip、component、消息、attention、
多前驱因果GRU与完整跨块梯度均保留。检测不门控mask，也不向pixel加回分数。
共享特征仍受两个任务共同影响；不是完全隔绝任务梯度。

## 固定资产与公平性

只取原manifest明确train_monitor的8pair/16完整视频，全部dev16pair/32视频；共21,669帧，
训练7,520帧。两条件都完整处理，不下载、不解码、不开前端、不复制约130GiB缓存。
训练pair：AF 7SXQS/C9VPX/NK9RX/UVEPN；ANAKIN 40y1x9/76rjm7/7p8r14/95qlg9。
小型身份、revision、相对路径及PTS摘要见[manifest](../v8/evidence/head_decoupling_manifest.json)。
原96训练视频拟合的统计/grouping固定复用，本轮仅16视频更新参数，无开发拟合。
预检全部48视频canonical身份、header/帧数、PTS、mask SHA和已拟合component身份/SHA；
缺失即停止，不隐式重建cache。原产物小文件hash和模型大小/mtime在新目录留档，结束再核对。

共同初始化SHA为`a350fe6ee6f57095d82d895136660e8de7c925e383df609469bf7ef6dd8365ba`，
已由seed20260909随机构建验证与原paired_initialization一致，非best/last。
独立head深拷贝coarse初值但不共享参数；COUPLED362658参数，SEPARATE_HEADS370979，新增8321。

每条件固定30轮（240 pair更新），不用早停；相同确定性逐轮顺序、AdamW1e-4/wd1e-4、clip1。
L=video+0.5frame+pixel，像素BCE/Dice独立日志不改原总损失；标签及GT空/缺失语义不变。
每轮完整train16/dev32评价，AP每5轮与最终；主比较相同第30轮，best只补充。

## 实测验证和成本

34项定向测试/回归通过；原实验小文件hash与模型大小/mtime在smoke后复核未变。

梯度路径：检测损失更新新检测头及共享特征，不直接更新pixel专属卷积或coarse head；
pixel损失更新pixel及共享特征，不更新检测头。共同参数逐项相等、独立参数存储、旧COUPLED
计算路径、分块梯度/跨块历史、训练评价聚合、pair边界精确恢复和自动索引/报告均有定向验证。
真实完整AF pair NK9RX各739帧、ANAKIN 95qlg9各90帧在两条件上各更新并评价，数值有限，smoke权重丢弃。

|条件|数据集|更新秒|常规评价秒|精确AP评价秒|峰值allocated MiB|
|---|---|---:|---:|---:|---:|
|COUPLED|AF|15.37|5.21|不适用|633.26|
|COUPLED|ANAKIN|1.71|0.71|2.32|594.93|
|SEPARATE_HEADS|AF|14.00|5.32|不适用|632.85|
|SEPARATE_HEADS|ANAKIN|1.72|0.73|2.36|594.60|

完整记录：[smoke](../v8/evidence/head_decoupling_smoke.json)。
按两pair实测速率外推30轮，COUPLED约4792秒、SEPARATE_HEADS4658秒，合计约2.63小时。
包括常规评价与ANAKIN精确AP额外耗时外推，不含最终额外best/last评价、画图/压缩导出和IO波动；
实际总耗时以Python记录为准，暂估约3小时，不用删模块/降分辨率调整预算。

## 运行和产物

复用原入口，一个命令自动核验、smoke缓存命中、两条件训练、最终评价、索引和中文报告：

```bash
cd /media/njupt/8FCD3869E2FAB06F/chenyh/projects/sparse_3d_forgery_detection
MPLCONFIGDIR=/media/njupt/8FCD3869E2FAB06F/chenyh/cache/matplotlib \
  OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=0 \
  /media/njupt/8FCD3869E2FAB06F/chenyh/envs/v8/bin/python -u \
  -m research_tools.v8.spatial_localization.run \
  --config docs/v8/head_decoupling_config.json --resume
```

输出根：
`/media/njupt/8FCD3869E2FAB06F/chenyh/data/sparse_3d_forgery_detection/derived/v8_spatial_head_decoupling_v1/`

`progress.json`记录阶段、条件、epoch/pair；`run.log`每轮打印进度。
每条件保存`history.csv/json`、`epoch_metrics/`全部逐视频/帧结果、`best.pt/last.pt`及恢复状态、
`curves.png`、`summary.json`。online与完整train评价分开，不称train_monitor。
`samples/{best,last}/{train,dev}`含固定三时刻ANAKIN raw float32压缩logits、GT、PTS、身份、映射；
两条件的逐帧检测分数也保存。`index.html`合并全部固定样例，绝对logit色阶固定[-6,+6]，二值0。
`report_zh.md`首先比较epoch30，随后完整best同epoch结果；新增存储和实际成本见`costs.json`。
不会按视频real预测隐藏mask，也不自动追加seed、样本、训练轮次或阈值搜索。

原实验资产只读；结束后小文件hash与原模型大小/mtime再次核验未变。正式结果见下节；启动快照仅作历史记录。

## 完成审阅：输出分离未改善本轮定位

主比较严格使用同一第30轮，16视频完整训练集与32视频开发集；没有调阈值、挑选checkpoint或更换样本。
下表均为ANAKIN，正GT帧均值→视频均值，real空间FP是空GT像素误报比例。

|条件|集合|Dice|IoU|pixel AP|real空间FP|
|---|---|---:|---:|---:|---:|
|COUPLED|train|0.535687|0.419452|0.669810|0.019876|
|SEPARATE_HEADS|train|0.534415|0.416663|0.710957|0.018727|
|COUPLED|dev|0.052465|0.029815|0.080381|0.065298|
|SEPARATE_HEADS|dev|0.037654|0.020671|0.102253|0.030258|

分离−耦合：训练Dice−0.001273（几乎相同），开发Dice−0.014810。pixel AP有所上升，
但固定阈值0的区域重叠没有改善；排序信息提升不等于空间预测合格。
训练/开发Dice差分别0.483223/0.496760，两个条件均有显著的描述性泛化差距。
这里没有做显著性检验，“差距大”不代表统计显著或全训练集已充分拟合。

第30轮检测指标如下，均来自相同权重，非各任务最佳轮次拼接：

|条件|数据集|dev video AUROC/AP|dev frame AUROC/AP|dev视频混淆矩阵 [[TN,FP],[FN,TP]]|
|---|---|---|---|---|
|COUPLED|AF|0.656250/0.612649|0.728385/0.311120|[[5,3],[3,5]]|
|SEPARATE_HEADS|AF|0.484375/0.556845|0.653104/0.382677|[[8,0],[8,0]]|
|COUPLED|ANAKIN|0.562500/0.632585|0.528897/0.542122|[[6,2],[4,4]]|
|SEPARATE_HEADS|ANAKIN|0.531250/0.613267|0.521933/0.510494|[[8,0],[8,0]]|

分离头在两库开发集都将所有视频判real，故视频Recall/F1均0、ACC均0.5；并非像素mask全部为空。
完整train video AUROC：AF 1.0→0.625、ANAKIN 0.9375→0.875；train frame AUROC：
AF 0.992218→0.903289、ANAKIN 0.850442→0.686826（箭头为COUPLED→SEPARATE）。
AF dev真实帧FPR为0.322382→0，ANAKIN为0.343510→0，降低误报伴随检测漏检，不能称检测改善。
所有ACC/Precision/Recall/F1、混淆矩阵、逐视频指标及分项损失见各条件summary和epoch_metrics。

best仅补充：COUPLED epoch18 dev Dice0.092025，SEPARATE epoch29为0.039146；
各自best的所有任务结果在自动报告中同epoch列出，不取代第30轮主比较。

### 轨迹与固定图像证据

COUPLED从epoch16开始有非零训练硬Dice，epoch26达0.5721、末轮0.5357；dev最佳在18轮，此后总体下降。
SEPARATE的硬Dice在epoch1–28为0，29轮升到0.4185、30轮0.5344；同期ANAKIN训练pixel BCE
从首轮0.5205降到28轮0.0882，soft Dice loss从0.4307降到0.3010。
所以长期硬Dice0不能直接解释为完全没有空间学习，后期跨过固定阈值才出现明显重叠。
本预算内训练仍在变化，不能据单seed30轮宣称架构绝对无效；也不因此自动延长训练。

末轮4条ANAKIN训练fake Dice（COUPLED/SEPARATE）：95qlg9 0.6617/0.5675，
76rjm7 0.6526/0.8349，7p8r14 0.7773/0.5995，40y1x9 0.0511/0.1358。
平均约0.535掩盖了样本差异，尚未充分拟合全部4条，且不能以三个较好样本代表整体。

已查看全部24条ANAKIN train/dev real/fake在epoch30的25/50/75%二值对照，共72固定时刻。
下面只描述固定时刻，不概括未查看的全视频，也不从缩略图反推精确指标：

- 训练76rjm7分离输出填充GT区域较完整，但对应real仍有局部响应；40y1x9两者预测相对GT偏移，低Dice仍在。
- 训练7p8r14帽子区域已有响应，但轮廓粗糙、覆盖不全；95qlg9移除区域也仍有外扩或遗漏。
- 开发4rvro9帽子、7e2e57猫、7e2p57小符号均未获正确覆盖；分离常更接近空背景，并未找回目标。
- 开发4xd3r9两者仍在右下背景出现大块响应，对应real也相似；分离不保证逐视频都减少误报。
- 开发71g867分离响应较窄、GT覆盖减少；4wgx37分离出现一些耦合所无的响应，但位置/形状仍不贴合。
- 开发4xd5r9分离大部分为空、95q8g9少量分离响应偏离GT，未形成完整编辑区域。

已保存raw logits计算的固定三时刻统计也显示：train背景FPR 0.028058→0.013879，GT召回0.576825→0.487704；
dev背景FPR 0.066956→0.030655，GT召回0.136290→0.104023。这是固定样例均值，不替代全视频指标。
背景响应减少与GT召回下降并存，更符合保守输出，不能称位置选择性或GT完整性普遍改善。

### 结论与唯一后续方向

本次指定的GRU检测头输出分离没有改善训练定位，开发定位更差，检测亦无整体改善。
因此**现有证据不支持将同一像素图承担检测认定为主要瓶颈，也不支持优先扩大这项修改**。
它不排除其他共享特征任务牵制；本次比较仅涉及指定路径、一个seed和固定预算，不能泛化为所有独立头均无效。

唯一建议：先核查完整定位路径在更少预先固定样本上的空间可拟合性，保留完整目标架构和时序，
明确哪个样本/区域不能学好，再决定需要处理何种空间特征或监督限制。
当前不优先扩数据、不增加隐藏维度或自动延长训练；也不从开发样本中删难例。本轮到此停止。

### 验收与实测成本

完整30轮history、240更新和optimizer/AMP/RNG恢复状态核验通过；每轮train16/dev32结果齐全。
best/last共288个ANAKIN固定float32压缩logit NPZ，形状、有限值、帧号、PTS和映射核验通过；
统一索引144行（每行含两个条件），原图/GT/二值/固定[-6,+6]色阶均保留。
正GT逐视频均值复现汇总至1e-12；原缓存未复制，原实验小文件hash/模型大小mtime均未变化。
没有重跑训练、前端或模型推理；本次仅审阅已保存结果与图像。

实际训练及评价活跃墙钟：COUPLED6040.07秒、SEPARATE5860.09秒，合计约3.31小时。
峰值allocated分别649.02/652.00 MiB；新增输出在自动报告计量时约401.18 MiB，原缓存复制0字节。
新增检测头8321参数。此处为实际成本，取代约3小时的启动外推，不改原smoke记录。
小型机器证据：[head_decoupling_results.json](../v8/evidence/head_decoupling_results.json)。

查看[自动完整报告](/media/njupt/8FCD3869E2FAB06F/chenyh/data/sparse_3d_forgery_detection/derived/v8_spatial_head_decoupling_v1/report_zh.md)、
[固定样例索引](/media/njupt/8FCD3869E2FAB06F/chenyh/data/sparse_3d_forgery_detection/derived/v8_spatial_head_decoupling_v1/index.html)。

## 启动快照

代码`3acbf6cba31752a63d86fe0432c9db28266803d4`已push后启动；北京时间2026-10-27 21:47。
tmux：`v8_spatial_head_decoupling_v1`，Python PID **794523**。
一次启动核验：RUNNING/train，COUPLED epoch1、当前pair7/8；进程存活，未见启动错误。
记录见[head_decoupling_launch.json](../v8/evidence/head_decoupling_launch.json)。
预估约3小时，完成时间约北京时间10月28日00:47；IO、绘图及最终导出会影响实际时间。
不在Codex内逐轮轮询。运行输出重定向run.log，tmux主窗可没有文字；可另开终端查看：

```bash
V8_OUT=/media/njupt/8FCD3869E2FAB06F/chenyh/data/sparse_3d_forgery_detection/derived/v8_spatial_head_decoupling_v1
cat "$V8_OUT/progress.json"
tail -f "$V8_OUT/run.log"
```

以上启动时间和预估仅作历史记录；Python已正常结束，2026-10-28完成集中结果审阅。
