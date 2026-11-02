# PVST v0.2 node_chunks_v2 第三次显存不足报告

本次再次发生CUDA OOM，状态为RESOURCE_BLOCKED，Python及tmux包装的真实退出码均为1。
任务已退出；正式模型0/2，未进入epoch0或正式训练，没有最终性能指标。
本轮按用户要求仅核对现有证据并生成报告，后续等待用户指示；没有修改代码、配置、协议、数据状态，没有启动/恢复计算或创建新执行版本。

## 运行身份与过程

- 执行版本：node_chunks_v2；源码commit：`60ddb0fd78fbc206916f9ae8fff7cafd9ff2f077`。
- 分支：`pvst-point-voxel-v02`；PID145289；tmux会话`pvst_point_voxel_v02_nodes`。
- 启动：2026-11-02 17:49:29 +08:00；退出：2026-11-02 18:54:01 +08:00。
- 启停记录相减3872.301秒，约1小时4分32秒；包含缓存核查及失败验收，不是正式训练耗时。
- 只读进程核验：PID已不存在，pane为dead=1、exit=1。launch.json的STARTED是历史启动事件，当前终态以progress/退出记录为准。

本次首先执行XYZ完整pair检查。每次有效更新按原规则先完整AF视频，再ANAKIN视频；本次AF部分的梯度累计仍保留，随后在ANAKIN fake的首块前向中失败，尚未执行首次optimizer.step。

|阶段|实际证据|
|---|---|
|缓存复用核查|48份complete存在；请求22,074帧，新增token帧0，调用0.836秒；未重跑前端|
|AF完整视频|video-id `23ff43433738441e0208f95e`，547帧；69个目标分块PASS，最后范围[544,547)，逐块计时合计3861.904秒|
|ANAKIN fake|video-id `cb41b3d004b6640f6c3b5bb1`，pair `an:7ogg04`，360×640、75帧；首个目标块[0,8)前向OOM|
|完整pair验收|full_pair_check.json不存在；两条件均未取得本版本完整PASS|
|全缓存节点计数及代表验收|位于完整pair检查之后，本次未到达；没有全队列最大窗口结论|
|正式实验|UV_PVST/XYZ_PVST目录均无正式产物，根及两条件下无pt文件；无正式history/summary、epoch0/best/last|
|最终结果发布|publication.json不存在，本次未进入结果发布；源码60ddb0f此前已推送|

## 精确失败位置

逐块日志已能确定本次失败窗口，区别于第一次只有0/75低频日志的情况：

|字段|记录|
|---|---|
|条件/阶段|XYZ_PVST / full_pair_check / 第1次有效更新|
|目标帧|[0,8)，即索引0–7|
|左上下文|[0,0)，片头无历史帧|
|fine节点数（窗口各帧占据节点合计）|1,020,328|
|coarse节点数（窗口各帧占据节点合计）|780,798|
|粗尺度层|第3层，dilation=4，宽度512|
|进入该层dtype|torch.bfloat16|
|最终异常源码|research_tools/pvst/temporal.py:77：`return x + torch.cat(values)`|

调用链：`train.main → smoke_pair → video_backward → PVST.forward → coarse_time → SparseLayer.forward`。
原始trace停在FFN块输出的全量拼接及最终残差表达式；该行包含cat和加法，未记录算子级内存事件，故报告不只凭该行确定究竟是哪一个申请失败。
细尺度三层及粗尺度三层的最后FFN块均出现在trace诊断中；粗尺度尾块为[2558,512]，其隐藏形状[2558,2048]。
这是前向阶段的失败，不是本块backward或optimizer.step报错。

## 显存记录

|指标|本次记录|
|---|---|
|GPU容量|23.51 GiB|
|失败时空闲|641.75 MiB|
|本次失败申请|764.00 MiB|
|本进程占用（含非PyTorch）|21.67 GiB|
|PyTorch allocated（异常时）|20.98 GiB|
|reserved但未allocated（异常时）|223.45 MiB|
|失败块max_memory_allocated|22,528,180,224字节，20.9810 GiB|
|失败块max_memory_reserved|23,148,363,776字节，21.5586 GiB|
|原始异常中的其他进程|ToDesk PID106515，507.28 MiB|

上述max值为本块统计，不能解释为整段进程的全局峰值；异常时占用与peak口径也不同。
本次仍已设置expandable_segments:True。没有关闭ToDesk或其他服务，也没有尝试释放服务器服务来继续。

仅用已记录形状计算：780,798×512个BF16元素约762.50 MiB，与764 MiB量级一致。
这说明全节点D维张量本身已达到约0.75 GiB，分块限制FFN隐藏张量后，最终全量输出/残差仍有显著瞬时开销。
该计算是形状推算，不是额外GPU测量；不能把它当成某一具体allocator事件的证明。

## 三次运行对照

|版本/运行源码|失败位置|额外申请|可追溯范围|运行墙时|
|---|---|---|---|---|
|初始 / 8477da4|粗尺度FFN/GELU|2.98 GiB|原日志未记录精确失败块|7515.452秒|
|ffn_chunks_v1 / 81312af|粗尺度第3层attention投影后的Dropout|764 MiB|ANAKIN目标[0,8)，fine1,020,328/coarse780,798|3762.245秒|
|node_chunks_v2 / 60ddb0f|粗尺度第3层FFN最终拼接/残差表达式|764 MiB|同一ANAKIN目标[0,8)，同一节点数|3872.301秒|

第一次运行包含最初特征准备，后两次复用缓存；墙时不代表等量工作，不直接作为吞吐优劣比较。
所有版本均未开始正式训练。23项正确性测试通过不等于真实大节点窗口的资源验收通过。
本次代码没有改变17,821,672参数、输入覆盖、128/256/512宽度、目标/halo、监督或划分；执行分块未充分解决现有硬件上的整体内存峰值。

## 当前可以与不能得出的结论

可以确认：两处局部中间计算完成分块后，内存瓶颈仍出现在全节点最终张量表达式；这个执行版本没有通过完整视频验收，不能放行训练。

尚未测清：全量拓扑、各层保存输入、最终张量及其他存活引用各占多少；更长24帧窗口的实际峰值；最小所需GPU容量。
本次只覆盖到失败视频首8帧，不把其资源需求当成全队列上界；也不凭此断言需要某个具体规格的新GPU或模型方法无效。
本轮不实施任何进一步优化、不提出自动执行动作；由用户决定接下来如何处理。

## 证据与资产保留

当前原始产物目录：
`/media/njupt/8FCD3869E2FAB06F/chenyh/data/sparse_3d_forgery_detection/derived/pvst_point_voxel_v02/`

本次progress.json（含完整traceback）、run.log、full_pair_XYZ_PVST_step1.jsonl、launch.json、protocol.json、exit_code.txt、process_exit_code.txt和exit_history.jsonl均保留原样。
前两次归档仍分别位于execution_history/initial_8477da4/与execution_history/ffn_chunks_v1_81312af/。
exit_history包含三个真实退出码1的事件；没有改写为正常退出0。48份缓存保留，不重新提取、不清理。

本次读取关键原始证据的SHA256：

- `progress.json`：`d6ebe2366d4f776305fc93ba08311c338c8d0fb954a861f666741262b0749f76`
- `run.log`：`a23c500942a9f0d2955b040fdad774e6b3da168af39ced0fb238d8658d3c448c`
- `full_pair_XYZ_PVST_step1.jsonl`：`350bfdf12572810649c474098ef9a023f5c976c5a4bba9045658bcd4de62c1ef`
- `protocol.json`：`416781b66ef20c29ac234b4e5c88fe261d3af969e62928e7ecb14830022b6e50`
- `exit_history.jsonl`：`8555e54cb30f2f60406b790a9369c9cccd85dfc8af6e506bab78c1d9426b14f7`
