# V8预训练定位的限定BatchNorm状态诊断

2026-10-29；COMPLETE。基于`98425b8`，仅检查两个既有best，无训练、恢复、反向传播、
optimizer、前端或DINOv2特征提取。原正式成绩保持不变。

## 范围、口径和完整性

固定原训练pair，不抽样替换、不用开发媒体校准：

|pair|视频身份（real / fake）|每条帧数|原生尺寸|
|---|---|---:|---|
|an:95qlg9|5e281219c8a281ae6109f506 / ceb1a2c7d9ba416daf788ea4|90|368×640|
|an:40y1x9|5412ec415c6dc6000437ff00 / ac8efb162961478dd9f9af22|150|640×360|

共4视频、480唯一帧；全部实际帧、PTS、官方GT与已有缓存沿用原manifest。
两个模式从相同best各自独立严格加载：SPATIAL_TRANSFER epoch12、SPATIAL_3D_CONTEXT epoch4。
只构建官方HighResContextHead和原SpatialModel，不加载或执行DINO骨干。

- RUNNING_STATS：整模型eval，decoder BN使用保存的running statistics。
- BATCH_STATS：先整模型eval，仅decoder的5层BatchNorm2d设为训练统计模式；
  Dropout、GRU及其余模块仍eval。连续2帧、原顺序、每视频清零GRU，全程no_grad。
  只在内存更新BN buffers，结束丢弃，未保存校准checkpoint。
- 原图连续回映后logit≥0；padding排除，空GT不计正帧Dice/IoU但保留背景统计。
  GT召回与GT内/外logit为每视频原生像素pooled值，面积含全部帧；real无正GT项为null。
- RUNNING_STATS与已有逐视频Dice/IoU/空GT FP最大绝对差4.436636e-7，低于事先固定1e-5核对容差。
  迁移模型差为0，增强模型为微小数值差；未改阈值。
- 同一模型两模式参数SHA完全相同，前后未变，所有参数无梯度且requires_grad=False。
  RUNNING_STATS BN计数不变；BATCH_STATS五层计数各增加240（480帧/2），仅在内存。
  107个原输入/结果文件前后SHA一致，覆盖两个best、所选四视频缓存、mask和正式汇总等。
  4项定向测试通过，含只切换BN、Dropout关闭、参数不变、固定身份/空GT/分块及固定预览。

best文件SHA256：

- SPATIAL_TRANSFER：`d66f8a6da01c2ba60ac3eabfed331ca12d455f0ddb98a97e0e0ad341daadfbc2`
- SPATIAL_3D_CONTEXT：`8f087bfc7b4e5e8bc16fd0a6d1002357aa71130bd9c4228cc61efed89b5b503d`

## 实测指标

每视频、每模式的全部指标与耗时保存在[小型CSV](../v8/evidence/pretrained_spatial_bn_diagnosis.csv)。
下面列fake及其对应real误报，均为比例；R=RUNNING_STATS，B=BATCH_STATS。

|模型|pair|模式|Dice|IoU|GT召回|预测面积|对应real FP|GT内logit|GT外logit|
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
|SPATIAL_TRANSFER|95qlg9|R|0.830154|0.729433|0.847645|0.070701|0.002594|3.442943|−7.405924|
|SPATIAL_TRANSFER|95qlg9|B|0.839720|0.743549|0.844571|0.068663|0.002608|2.984662|−7.403828|
|SPATIAL_TRANSFER|40y1x9|R|0.501870|0.365100|0.275181|0.014557|0.006583|−2.706110|−7.818350|
|SPATIAL_TRANSFER|40y1x9|B|0.598243|0.458694|0.469355|0.020270|0.000797|−0.926159|−7.354294|
|SPATIAL_3D_CONTEXT|95qlg9|R|0.781422|0.661014|0.880993|0.086703|0.022481|3.639150|−5.983388|
|SPATIAL_3D_CONTEXT|95qlg9|B|0.784651|0.664333|0.879659|0.085698|0.035403|3.201699|−5.655702|
|SPATIAL_3D_CONTEXT|40y1x9|R|0.432410|0.304223|0.338194|0.029548|0.019801|−2.933581|−6.791953|
|SPATIAL_3D_CONTEXT|40y1x9|B|0.451220|0.314572|0.438238|0.036870|0.021326|−1.491414|−6.364603|

## 判断与唯一优先事项

BN统计模式确实会明显、且依样本改变输出；这里的“明显”指效应幅度，不是统计显著性检验。
四个模型×fake组合Dice均增加，但幅度不同：横屏迁移/增强+0.009566/+0.003229，
竖屏+0.096373/+0.018810。不能把两条fake的方向外推为开发集收益或BN是失败主因。

影响不只是整体分数平移。竖屏GT内logit分别提高1.779951/1.442167，GT外仅提高0.464056/0.427350；
GT召回分别增加0.194174/0.100045。迁移模型竖屏三个固定时刻的跨模式logit相关系数
0.288–0.741，delta空间标准差1.729–1.943、均值0.454–0.650，阈值mask相互IoU0.408–0.549，
显示响应位置/形状也变动。增强竖屏相关系数0.883–0.890，主要仍在同一区域，强度与范围发生变化。
这些位置统计只基于固定时刻，不冒称全视频位置指标。

已查看两模型竖屏fake的50%对照、迁移横屏fake的50%及增强横屏real的50%。
迁移竖屏的BATCH_STATS响应更贴近GT、减少右上侧外溢；增强竖屏覆盖扩大但仍有外溢。
横屏fake核心位置变化较小，与较小Dice差一致。横屏fake的GT召回反而略降
（−0.003073/−0.001333），说明其小幅Dice增加不等于覆盖增加。

两模型有“竖屏更敏感、横屏fake较稳定”的共同现象，但背景代价不同：
迁移竖屏real FP从0.006583降至0.000797；增强横/竖real FP从0.022481/0.019801
升至0.035403/0.021326。因而不能把BATCH_STATS当成一致更优的替代评价。
两pair不能把效应归因于横竖屏本身，内容、编辑与模型checkpoint均有差异。

**下一步优先控制并稳定BN训练/评价统计的一致性，再判断上下文附加价值；目前不宜停止排查BN、
直接把固定decoder残差对照当成已排除统计干扰的实验。** 这只是优先级建议，
本轮不校准模型、不切换正式评价、不修改训练策略、不启动新训练。
原“固定已适配decoder残差”建议仍为后续候选，BN不是已证实的唯一瓶颈。

## 输出与执行记录

独立输出：
`/media/njupt/8FCD3869E2FAB06F/chenyh/data/sparse_3d_forgery_detection/derived/v8_pretrained_spatial_bn_diagnosis_v1/`

完整中文数值报告`report_zh.md`；16条模型/模式/视频指标`metrics.csv`；
核对与参数检查`results.json`；原输入身份与SHA `input_identity.json`；
24张固定对照及索引`index.html`；仅48个固定帧F32 NPZ，不存全视频logit。
入口内部总计时14.803秒，含核对、前后hash、固定导出和渲染。
四视频评价累计秒：迁移R/B为2.085/1.910，增强R/B为2.934/2.822，含I/O及固定帧保存，
不能视作纯GPU耗时或速度比较。大图、checkpoint和缓存不进Git。

唯一小型入口：`research_tools/v8/pretrained_spatial_bn_diagnosis.py`，直接复用CachedVideo、
SpatialModel、原图回映和空间指标。正式训练/评价默认行为均未改；输出非空即拒绝重复运行。
历史执行命令（本轮已完成，不再执行）：

```bash
OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=0 \
/media/njupt/8FCD3869E2FAB06F/chenyh/envs/v8/bin/python -u \
  -m research_tools.v8.pretrained_spatial_bn_diagnosis \
  --output /media/njupt/8FCD3869E2FAB06F/chenyh/data/sparse_3d_forgery_detection/derived/v8_pretrained_spatial_bn_diagnosis_v1
```
