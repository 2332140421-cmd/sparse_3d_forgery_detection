# PVST v0.2 首次自动运行显存不足报告

本次未完成正式实验。流程在正式epoch0前的XYZ_PVST完整视频检查中因CUDA显存不足退出，状态为`RESOURCE_BLOCKED`；正式完成模型0/2，无正式训练、评价指标或最终结果发布。本报告仅读取既有产物，未启动、恢复或重跑任何计算。

## 运行身份和退出历史

- 分支：`pvst-point-voxel-v02`；运行源码commit：`8477da4c5df9e01d4e092163de537d1200dcd85c`。
- Python PID：2456391；tmux会话：`pvst_point_voxel_v02`。
- 启动：2026-11-02 05:36:31 +08:00；退出：2026-11-02 07:41:47 +08:00。两条时间记录相减约2小时5分15秒，包含准备及检查，不是正式训练耗时。
- `exit_code.txt`、`process_exit_code.txt`均为1；`exit_history.jsonl`保留本次真实退出事件。此前只读核验进程已不存在，tmux pane为dead=1、exit=1。
- `launch.json`的STARTED是启动事件，不代表当前仍运行；当前终态取`progress.json`及退出记录。没有`publication.json`，因为尚未进入最终结果发布阶段；源码此前已正常推送。

## 已完成与未完成

|环节|现有证据与边界|
|---|---|
|固定数据与身份|AF与ANAKIN各训练8 pair、开发4 pair；32训练/16开发视频，共22,074帧|
|原生输入与冻结视觉缓存|48份complete.json存在；准备记录本次新增22,026帧token，另48帧此前准备；不重新运行MoGe|
|准备耗时|本次prepare调用515.581秒；完整视频准备记录累计529.881秒，口径不同，不相加当本次总耗时|
|单块资源验收|720×1280、8目标+16上下文、完整损失、各两次更新均通过；UV/XYZ峰值allocated约16.183/16.216 GiB，见resource_evidence_v02.json|
|UV完整视频检查|源码控制流及日志表明已完成两次有效更新、检查评价和重载评价，并进入XYZ条件；逐条件结果只在内存累计，未独立落盘，不将其描述为已保存的完整验收报告|
|XYZ完整视频检查|第1次有效更新尚未完成；AF视频已累计梯度，随后的ANAKIN fake处理中OOM；未执行这次optimizer.step|
|整体验收|full_pair_check.json不存在，整体验收未通过|
|正式实验|两条件均无history.json、summary.json，未开始正式epoch0/训练，无best/last及最终指标|

## 失败位置

最后的阶段记录为`full_pair_check / XYZ_PVST / step=1`。报错前日志显示：

```text
VIDEO_BACKWARD XYZ_PVST 23ff43433738441e0208f95e 480 / 547
VIDEO_BACKWARD XYZ_PVST cb41b3d004b6640f6c3b5bb1 0 / 75
```

失败时正在处理原训练ANAKIN fake：

- pair：`an:7ogg04`；source：`anakin:https://vimeo.com/12973568`。
- video-id：`cb41b3d004b6640f6c3b5bb1`；原生H×W为360×640，共75帧。
- 日志每80帧打印一次，故`0 / 75`不能确定准确报错帧或target chunk；本次未逐块记录体素数，也不能确定具体粗尺度层号。

关键调用链：`train.main → smoke_pair → video_backward → PVST.forward → coarse_time → SparseLayer.forward → self.ffn(self.norm2(x)) → GELU`。
对应运行源码：`train.py:98,54`，`model.py:158`，`temporal.py:75,59`。异常发生在前向FFN，不能写成该块反向传播失败。

## 显存证据

原始异常为`torch.OutOfMemoryError: CUDA out of memory`：

|项目|原始异常记录|
|---|---|
|GPU总容量|23.51 GiB|
|当时空闲|1.67 GiB|
|失败申请|2.98 GiB|
|本进程显存（含非PyTorch）|20.62 GiB|
|PyTorch allocated|19.64 GiB|
|PyTorch reserved但未allocated|509.84 MiB|
|另一个ToDesk进程|PID1208337，507.28 MiB|

这是异常时刻的占用，未单独保存该次完整检查的max allocated/reserved，不冒称精确峰值。启动已设置`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`；不能仅复述错误提示就认定尚未开启。失败申请比空闲多约1.31 GiB；未分配reserved约0.50 GiB，现有证据不支持把问题仅归因于碎片，也不支持把约0.50 GiB的ToDesk进程认定为主因。

## 原因判断及其局限

确定事实是粗尺度全节点FFN的GELU需要额外大张量，超过当前可用显存。代码已按query分块attention，并对时空层做激活重算，但FFN仍一次处理全部节点；激活重算降低保留量，不消除单次前向FFN的瞬时分配。

此前资源探针按最大原生像素数挑选一个视频、一个24帧窗口。XYZ占据体素数由坐标分布决定，不只取决于图像分辨率。因此较小图像也可能有较高节点开销；原资源门槛没有证明所有视频/窗口均能容纳。这是验收覆盖不足，不能将单块PASS解释为整个数据队列已通过资源验收。

本次没有保存失败块的节点数、几何有效率或逐操作内存剖析，故“失败样本体素数具体多少”“哪个中间张量恰好占2.98 GiB”“是否还有可释放的引用”仍未确定。不据此判断模型效果、数据异常或三维方法无效。

## 恢复边界及后续建议（未执行）

48条视频的既有缓存可复用；没有正式训练进度需要恢复。完整检查的临时模型、optimizer及中途梯度未保存，不能从失败的准确帧续接；需重新执行未完成的检查。UV检查也没有独立持久化PASS记录，现入口会重新执行整体检查，不应伪称能够自动跳过。

优先候选是对逐节点独立FFN分块并重算激活，保留全部节点、宽度、层数、损失及完整梯度；这只是待验证方案，当前未实现，不能保证一定解决OOM。训练含dropout，改变执行分块可能改变随机掩码序列；需明确验证关闭dropout时输出/梯度等价、训练路径与随机性记录正确，不能承诺逐位复现旧执行轨迹。

任何续跑前先保留原失败证据，验证执行优化，再覆盖失败样本及高占据体素窗口做资源检查，通过完整pair检查后才进入原定两条件。若固定方法经等价优化仍无法适配24GB，应继续保持RESOURCE_BLOCKED并评估更大显存，不删点、不减通道、不改变方法来制造通过。

本轮仅生成本报告并更正文档入口，不修改模型、配置、指标、数据目录状态或退出历史，不启动新任务。

## 原始证据入口

数据目录绝对路径：
`/media/njupt/8FCD3869E2FAB06F/chenyh/data/sparse_3d_forgery_detection/derived/pvst_point_voxel_v02/`

保留文件：`launch.json`、`progress.json`（含完整traceback）、`run.log`、`exit_code.txt`、`process_exit_code.txt`、`exit_history.jsonl`、`protocol.json`、`manifest.json`、`preparation_cost.json`、`resource_*.json`。原始文件未改写。

本次报告读取的关键证据SHA256：

- `progress.json`：`3371b1aa3731f74c6f111b7ba19d1fd3917e77d0f2609b488fd711a49dfa11dd`
- `run.log`：`1fbd5e9f142cdb2a48644202f480fd9aab06e43db58b2de3b629b03fa1ab81d9`
- `exit_history.jsonl`：`ff12ce436f0a4dc0265324cbd61e28198a177f85cdd85e94cce2047987581e99`
- `protocol.json`：`be38704f4da87c1c078058b2a5962b7b986cbed9e2d6ee8db51367b906dabfd5`
- `preparation_cost.json`：`60e8672a9b646dfb8820c0bd1391b80633677c09f9af2cfa8e1440dea7ee5eea`
