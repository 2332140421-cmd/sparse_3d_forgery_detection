# V8检测／空间定位输出分离对照

2026-10-27；分支`v8-spatial-head-decoupling`，起点诊断提交`0964236`。
当前：实现、定向回归与两库完整pair smoke完成，正式任务待独立启动；没有正式成绩。
冻结协议：[head_decoupling_contract.md](../v8/head_decoupling_contract.md)、ADR-0023。

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

原实验资产只读；完成后复核并集中审阅实际结果，再更新小型结果摘要。当前无正式指标，不预填正结论。
