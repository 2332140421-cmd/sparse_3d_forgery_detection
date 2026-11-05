# PVST v0.3 目标帧2＋2＋2＋2资源验收

**状态：FAILED；真实更新0次；正式模型0/2，不启动训练。**

源码：`2e5918b7b7959140effd294456815c917fe6482d`；产物：`/media/njupt/8FCD3869E2FAB06F/chenyh/data/sparse_3d_forgery_detection/derived/pvst_target_microbatch_2_check_v1`。
沿用原模型/初始化/AdamW/Dropout和BF16外层、逐head FP32、CSR桥接、CPU卸载、head暂存、即时QKV、node_split/view清理。attention512/QKV8192/FFN8192/point32768/tile256不变。
三层因果最大回溯14帧加visual两帧=16；同帧context、固定BN、逐位置LayerNorm、冻结全局坐标scale、真实PTS保持。
frame BCE逐帧标量、pixel BCE逐帧原生像素均值、softDice逐帧全像素sum/epsilon1；clip_terms按目标帧均值。此AN real全部合法零GT，分项均以原8目标帧为分母，权重按目标数/8计算；缺GT语义、loss权重不变。以后AF/AN配对外层权重仍需保留。
每组反向后释放图和Video GPU缓存，只保留梯度；所有组成功前不clip、不step、不zero_grad。未完成组不会计作更新。
计划每更新72帧次、24唯一输入、8唯一目标；重复历史重新计算。失败时实际加载组数见下，不能把计划量当作完成量。
小图沿用同非零诊断头、全部Dropout关闭、训练/checkpoint路径与既定容差；真实资源恢复原Dropout，随机轨迹随分组变化，不称训练逐步一致。

## 语义核验
```json
{}
```

## 实测阶段
|组/阶段|秒|allocated/reserved GiB|峰值A/R GiB|RSS GiB|
|---|---:|---:|---:|---:|

## 节点/边
|组/层|N|E|拓扑字节|
|---|---:|---:|---:|

结果：成功step=0；优化器状态参数=0。
失败：[]；阶段=None；组=None
进程：None
更新记录：[]
主机内存：None
完整逐组记录/资源采样/原生错误保存在max_window。未完成的反向与完整更新时间不填0，不用旧4＋4失败耗时计算加速比。
仅规定窗口资源检查，XYZ正式配置仍null，不改变节点门槛/模型完成数；不重跑前端、旧24帧/4＋4或额外样本。本轮停止，不重试、不追加逐帧微批或算子修改。
