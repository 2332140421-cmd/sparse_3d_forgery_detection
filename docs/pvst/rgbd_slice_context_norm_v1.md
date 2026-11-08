# 最终关系上下文LayerNorm匹配实验

当前状态：RUNNING。执行源码`2b288c87568ee2a8fed0b50e3b73a3e67d085f47`；分支pvst-rgbd-slice-relations。
12份附件相同，本轮只运行一个新条件RGBD_SLICE_RELATION_NORM，不重复实验。

唯一核心改动：完整两组时间→深度关系之后，共享LayerNorm(512,eps=1e-5,affine=True)，先逐token归一化，再由同一C_norm供pixel_context和frame_context(mean(C_norm))消费。坐标/切片归属、池化、关系、原生解码旁路、损失、初始化及训练数据不变。两个残差仍零初始化。见[ADR-0035](../decisions/0035-rgbd-final-context-norm.md)、[配置](rgbd_slice_context_norm_config_v1.json)。

3项直接相关CPU检查和两次真实24帧更新检查均通过：

- 新旧B所有共有参数和缓冲逐项一致；CPU/CUDA初始随机状态一致。
- 旧B的真实best checkpoint严格加载成功，仅检查兼容，不对旧模型评分。
- 新增仅context_norm.weight/bias两键、1024参数；新条件总可训练参数33,265,335。
- 初始零残差检测/像素输出最大差均为0，沿用BF16 atol/rtol=.002。
- 逐token、不跨时间/切片统计，先norm再mean且两头共享输入，空片保持；标量观察不消费RNG、不截断反向。

真实非空GT检查沿用an:7ogg04 fake的[0,24)，没有新增拟合实验：

|检查更新|总loss|剪裁前总梯度范数|新增LayerNorm梯度范数|raw C RMS|C_norm RMS|
|---|---:|---:|---:|---:|---:|
|1|2.230198|20.866795|0|7.361800|1.000000|
|2|1.958602|16.485189|0.002332|7.381272|0.999999|

第二步池化、时间、深度及两个残差投影梯度均存在、有限且非零；第一步上游零梯度符合零初始化。统计只是检查证据，不是开发成绩或改善结论。检查权重丢弃，正式模型重新从原共同初始化开始。

实际AdamW分组：编码器17,477,856参数，lr1e-5；其他15,787,479参数，lr1e-4；两组weight_decay均.01。所有共有参数的组归属/lr/weight_decay保持，新增norm归新模块组。

独立入口已通过必要检查并进入epoch0完整开发评价，随后自动完成唯一新条件正式训练及best/last评价。复用原8训练/4开发pair、每轮48条24帧清单及全部缓存；未调用MoGe、下载、重复单pair拟合或旧A/B训练。max30/min10/patience8和开发Dice选best不改。

尚无新best/结束轮次或最终指标。最终报告将复用旧A/B JSON，列三条件统一表与四条fake差值，保留完整固定监测曲线和新条件剪裁/尺度曲线；旧B预测直接用保存JPEG拼图，概率色阶0–1，不重新评分。

产物根目录：`/media/njupt/8FCD3869E2FAB06F/chenyh/data/sparse_3d_forgery_detection/derived/pvst_rgbd_slice_context_norm_v1/`。
实时progress.json/run.log；正式更新诊断在RGBD_SLICE_RELATION_NORM/update_history.jsonl。计算结束后自动更新本页、报告并普通commit/push，发布状态单列。本轮不追加其他训练或调参。
