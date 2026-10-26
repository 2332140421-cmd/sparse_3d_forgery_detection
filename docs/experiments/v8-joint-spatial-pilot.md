# V8 双数据集完整空间定位 pilot v1

状态：**COMPLETE**。两模型训练、最终best/last评价和自动报告均完成；按预注册规则早停。
协议：[joint_spatial_contract.md](../v8/joint_spatial_contract.md)、ADR-0022。

## 实际结构与队列

CELL_LOC / COMPONENT_LOC 均362,658个可训练参数；共享256结构、真实dense XYZ局部几何、
512RGB/高分辨率skip及像素残差定位，完整因果GRU跨块梯度。唯一条件差异是成员分区。
新定位模型配对随机初始化；MoGe-2、CoTracker3 online、ImageNet ResNet18冻结。
没有加载旧V8 checkpoint，没有将CELL_LOC作为纯二维对照。

按seed20260909冻结96训练、32开发视频，总59,909实际解码帧：
AF train24 pair（12Wan/12FCVG）、dev8（4/4）；ANAKIN train24/dev8 pair，
使用has-masks官方train/validation，训练/开发source交集0。
新机队列不称旧244/28复现。逐视频身份、SHA、两个letterbox映射、PTS摘要及标注状态见
[evidence/joint_spatial_manifest.json](../v8/evidence/joint_spatial_manifest.json)。
完整PTS留在外部joint_manifest.json，Git内保留摘要SHA。开发未用test样本。

ANAKIN排除既有不符/未解决项，并按预先固定候选顺序排除8项新增技术冲突，未截断/复制GT。
包括7项帧数或编号冲突及1项原生mask尺寸不符；具体原始数量和尺寸留在清单。
用户已确认此前核验批mask覆盖位置且有像素级细节，本轮目标为编辑区域。
真实原片是明确背景空间负例；AF fake空间标注缺失，像素损失跳过。
新增样本未按模型分数或异常明显程度选择，未逐条索取人工审批。

## 必要验证与资源

30项CPU关键测试和旧全视频回归测试通过（含自动报告渲染），包含配对初始化、真实XYZ、同格变化、padding/
缺失标签、视频/帧/像素损失回传、checkpoint等价、跨块梯度及pair边界精确恢复。
实际CUDA BF16前反向通过；结构scatter/attention保持FP32，像素卷积使用BF16。

同一个AF完整pair NK9RX real/fake各739帧，ANAKIN完整pair real/edited各90帧，
在两个条件上均实际完成一次更新并评价；全部数值有限，smoke权重丢弃。

|条件|来源|更新秒|评价秒|CUDA allocated峰值 MiB|
|---|---|---:|---:|---:|
|CELL_LOC|AF|19.28|5.25|633.13|
|COMPONENT_LOC|AF|13.87|5.25|632.63|
|CELL_LOC|ANAKIN|2.28|0.71|594.80|
|COMPONENT_LOC|ANAKIN|1.66|0.70|594.31|

完整记录见[resource_smoke](../v8/evidence/joint_spatial_resource_smoke.json)。
30轮上限外推更新约8.52小时、常规评价约1.30小时，总约9.82小时；不含前端、精确pixel AP
和最终额外评价，早停可缩短。冷启动缓存开销使该外推不是实测总时长。
低于预注册48小时阈值，保留每库24训练pair，未缩减模块、分辨率或视频长度。
预算决定见[joint_spatial_budget.json](../v8/evidence/joint_spatial_budget.json)。

运行中按视频记录前端同步计时，按轮记录训练/评价墙钟、帧数、梯度、显存与学习率。
准备调用墙钟与前端GPU/特征阶段计时分开；未单独计时的早期解码/缓存IO不填0。
实际缓存与最终输出字节自动写costs.json。精确pixel AP每5轮及最终计算，不做抽样近似。

## 独立运行与恢复

使用原V8环境；仅为自动曲线补装Matplotlib3.10.7及其小型绘图依赖。
显式保持NumPy1.26.4，并锁ContourPy1.3.3；torch2.5.1+cu124及前端版本未变。
复建绘图依赖使用`pip install numpy==1.26.4 matplotlib==3.10.7 contourpy==1.3.3`，避免解析到需NumPy2的新依赖。
配置：[joint_spatial_config.json](../v8/joint_spatial_config.json)。一个入口完成后续全部阶段：

```bash
cd /media/njupt/8FCD3869E2FAB06F/chenyh/projects/sparse_3d_forgery_detection
TORCH_HOME=/media/njupt/8FCD3869E2FAB06F/chenyh/cache/torch \
  /media/njupt/8FCD3869E2FAB06F/chenyh/envs/v8/bin/python -u \
  -m research_tools.v8.spatial_localization.run \
  --config docs/v8/joint_spatial_config.json --resume
```

本人独立会话名`v8_joint_spatial_pilot_v1`，不用用户手动串联前端和训练。
输出根：
`/media/njupt/8FCD3869E2FAB06F/chenyh/data/sparse_3d_forgery_detection/derived/v8_joint_spatial_pilot_v1/`

查看`progress.json`、`run.log`；训练后每条件`history.csv`一轮一行，指标展开为列。
`last.pt`保存模型/optimizer/AMP/RNG/epoch及已提交pair位置；SIGTERM在pair边界停止，
意外半pair从上个已提交pair恢复，不跳过梯度。全视频hidden不detach。
模型best/last、曲线、固定25/50/75%样例及时间响应、中文report_zh.md均由Python生成。
不在Codex内逐轮轮询，不自动追加模型、seed或预算。正式结果以COMPLETE及summary为准。

## 正式结果与限制

两条件为完整256结构/512像素分支，只有成员分区不同。

|条件|checkpoint/epoch|数据集|video AUROC/AP|frame AUROC/AP|IoU/Dice|pixel AP|real空间FP|
|---|---|---|---|---|---|---|---|
|CELL_LOC|best/6|ActivityForensics|0.609375/0.659402|0.507601/0.145768|未计算/未计算|未计算|0.242310|
|CELL_LOC|best/6|ANAKIN|0.546875/0.618498|0.534477/0.521902|0.075226/0.120542|0.127411|0.188547|
|CELL_LOC|last/14|ActivityForensics|0.593750/0.639805|0.538696/0.184153|未计算/未计算|未计算|0.127042|
|CELL_LOC|last/14|ANAKIN|0.546875/0.619418|0.520177/0.543233|0.044973/0.072401|0.099066|0.069987|
|COMPONENT_LOC|best/5|ActivityForensics|0.609375/0.606944|0.723012/0.289364|未计算/未计算|未计算|0.179744|
|COMPONENT_LOC|best/5|ANAKIN|0.515625/0.600767|0.531150/0.513880|0.075411/0.114502|0.119368|0.198327|
|COMPONENT_LOC|last/13|ActivityForensics|0.718750/0.681250|0.775869/0.470352|未计算/未计算|未计算|0.159522|
|COMPONENT_LOC|last/13|ANAKIN|0.578125/0.647764|0.554065/0.551769|0.042595/0.073625|0.107310|0.056133|
## CELL_LOC
状态：COMPLETE；best epoch：6；停止：early_stop_dev_dice_patience8_delta0.002。
参数量：362658；训练/开发视频：96/32；完整任务指标、混淆矩阵及训练开发差距见该条件summary.json。

## COMPONENT_LOC
状态：COMPLETE；best epoch：5；停止：early_stop_dev_dice_patience8_delta0.002。
参数量：362658；训练/开发视频：96/32；完整任务指标、混淆矩阵及训练开发差距见该条件summary.json。

COMPONENT_LOC减CELL_LOC的开发Dice：-0.006039。单seed小队列结果；按各自best同epoch评价，不拼接任务最佳值。
CELL_LOC按编辑类别的正确率/漏检和真实误报：
- ActivityForensics/wan：fake检出1/4；real误报1/4。
- ActivityForensics/fcvg：fake检出0/4；real误报0/4。
- ANAKIN/inpainting：fake检出0/5；real误报0/5。
- ANAKIN/splicing：fake检出0/3；real误报0/3。
ANAKIN train_monitor减dev Dice=+0.119431；本轮未见超过预设0.15的明显Dice差距，不等于证明泛化。
COMPONENT_LOC按编辑类别的正确率/漏检和真实误报：
- ActivityForensics/wan：fake检出1/4；real误报1/4。
- ActivityForensics/fcvg：fake检出0/4；real误报0/4。
- ANAKIN/inpainting：fake检出0/5；real误报0/5。
- ANAKIN/splicing：fake检出0/3；real误报0/3。
ANAKIN train_monitor减dev Dice=+0.146995；本轮未见超过预设0.15的明显Dice差距，不等于证明泛化。
前端耗时与缓存字节见costs.json；训练/评价实测见各轮history及summary。后续不自动扩大数据或追加模型。

前端同步推理/特征阶段累计16417.7秒；正式准备调用墙钟累计18156.0秒。
缓存129.911GiB；输出计量检查点129.983GiB；训练与评价活跃墙钟秒：{'CELL_LOC': 9756.401061598677, 'COMPONENT_LOC': 8923.90903971903}。


小型指标、停止原因、配对身份和产物SHA见[evidence/joint_spatial_results.json](../v8/evidence/joint_spatial_results.json)。

下一步先审阅各条件`samples/`的固定开发样例、`curves.png`与`history.csv`，
排查固定阈值下ANAKIN全部fake漏检和真实帧空间误报。当前不支持component定位增益，
也未证明可用定位性能；不通过改开发阈值或挑选其他epoch拼接最佳任务分数。
后续调整需独立冻结实验，保留本次基线，不自动扩量或重训。

## 启动核验快照

代码提交`776174d6d4a97dfcf69411823217659708c653b8`已普通push后启动。
真实Python PID **3433433**；tmux `v8_joint_spatial_pilot_v1`。
一次启动核验为frontend阶段，video_index2，当前视频已处理401/704帧；这不是整个队列的完成计数。
后续进度由Python自行写入，不持续轮询。完整启动记录见
[joint_spatial_launch.json](../v8/evidence/joint_spatial_launch.json)。

```bash
tmux attach -t v8_joint_spatial_pilot_v1
V8_OUT=/media/njupt/8FCD3869E2FAB06F/chenyh/data/sparse_3d_forgery_detection/derived/v8_joint_spatial_pilot_v1
cat "$V8_OUT/progress.json"
tail -n 20 "$V8_OUT/run.log"
```

启动快照仅为历史记录；当前已COMPLETE，正式成绩以以上结果及外部summary为准。
