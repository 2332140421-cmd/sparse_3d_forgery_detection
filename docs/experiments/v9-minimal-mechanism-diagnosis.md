# V9 最小机制诊断

起点`1f4e4a24b256af7acab6ded01aa4bcdc8e777dff`，新分支
`v9-minimal-mechanism-diagnosis`。只增加diagnose.py和3项定向测试；
复用原数据读取、冻结ResNet、模型和指标，不训练、不重提几何/追踪、不改正式结果。
输出`derived/v9_minimal_mechanism_diagnosis_v1/`。

## 训练片段拟合

各条件976片段、31232 clip×frame使用（31144去重帧），不是完整训练视频评价。
fake上下文122片段仍为负；122个fake parent均具备正负帧，单类排除0。

|条件|clip AUROC/AP|clip TN/FP/FN/TP|frame AUROC/AP|real parent帧FPR|fake parent时间AP|
|---|---|---|---|---:|---:|
|RGB_QUERY|0.764364/0.683011|503/107/171/195|0.773194/0.631452|0.160156|0.902142|
|XYZ_QUERY|0.808949/0.739774|402/208/83/283|0.811752/0.690796|0.306673|0.905975|
|XYZ_PARTITION|0.794639/0.722747|359/251/64/302|0.800096/0.677384|0.358671|0.905123|

首→末总loss分别0.947971→0.931069、0.957860→0.880895、0.971211→0.978657。
原clip/frame记录是两个parent之和，报告统一除2，验证total=clip+0.5frame。
后5轮有波动、无一致持续下降。已学到部分时间区分，但并未充分拟合训练目标；
不能把训练clip与完整验证video直接相减作为泛化差距。

## 验证推理干预

仅XYZ_PARTITION，28视频/27505帧相同集合、原权重、阈值0。
正式模式直接读原响应；RESET32和ZERO_CONTEXT共享像素编码。
重置首hidden保持原FP32且delta_t=0；context在最终concat处置零，不通过带bias投影。

|模式|video AUROC/AP|TN/FP/FN/TP|frame AUROC/AP|real-video帧FPR|fake-video帧AP|
|---|---|---|---|---:|---:|
|原连续保存响应|0.642857/0.735391|6/8/4/10|0.630117/0.397832|0.301336|0.601490|
|RESET32|0.642857/0.735391|6/8/4/10|0.629757/0.397703|0.301439|0.601606|
|ZERO_CONTEXT|0.642857/0.735648|6/8/5/9|0.628889/0.397592|0.274405|0.602440|

RESET32视频排序及预测均不变，平均绝对logit变化0.00269；未见明显长序列不匹配。
context置零使所有视频logit降低，平均−0.24621，Spearman0.997263；196个真假对中
2对排序变化抵消，AUROC几乎相同。影响主要体现为分数水平/固定阈值，未显示强排序增益；
不证明分支完全无用，不是独立训练的像素消融，也不证明长记忆有害。

优先级是训练/优化目标的可拟合性，而非时序长度修补；仍不能区分优化、表示与区域学习目标
原因。没有证据支持具体的新训练改动，暂不扩展V9，保留探索结果；不自动加损失或轮次。

## 验证、成本与使用

原资产hash前后不变；未干预首3帧复现误差3.8147e−6（容差0.002）；
head上下文精确零、f不变；PTS/标签对应、视频面积LSE复算误差<1e−9。
三项定向测试通过，没有旧测试全集、bootstrap或热图重制。
旧布尔元数据被JSON写为1，仅在诊断核验时恢复语义，不改原文件或数值。
验证前修正诊断hidden dtype并保留记录，642个合法训练parent响应复用，不重算。

首次RGB/XYZ_QUERY完整前向104.16/151.21s；恢复PARTITION53.91s（含已完成缓存读取），
此前中断部分单独耗时未记录。验证两模式共享133.33s；成功恢复调用191.90s，
不是整个任务总墙钟。峰值allocated7.434GiB/reserved9.105GiB，新增约6.33MiB标量产物。

```bash
.venv/bin/python -u -m research_tools.v9.dense_learned_regions.diagnose --output <独立诊断输出目录>
.venv/bin/python -m pytest -q tests/test_v9_mechanism_diagnosis.py
```

现有结果无需重跑；完整数字、loss分项、逐parent时间排序与分数变化见输出report.md及JSON/CSV。
