# PVST v0.2 FFN分块执行修复

2026-11-02。研究规格仍为[design_v02.md](design_v02.md)，旧失败报告不改写。
本次只改变计算的物化方式、必要资源验收及执行身份记录；正式成绩尚未产生。

## 原因与修复

原XYZ完整pair检查在粗尺度全节点`norm2→FFN→GELU`申请额外2.98 GiB时OOM。
`temporal.SparseLayer`现沿节点维将norm2和完整FFN一起分块，拼接最终D维输出再加残差。
初始`ffn_chunk=8192`，UV/XYZ相同；训练每块使用`use_reentrant=False, preserve_rng_state=True`。
没有先构造全量归一化/4D隐藏张量，没有detach、no_grad、原地覆盖或节点截断。
保留原外层checkpoint、dropout及全部模块/参数名字；绑定方法和层默认参数避免循环闭包引用错层。

输入分辨率、像素/点/占据体素/邻接、128/256/512、target8+halo16、监督、数据身份、优化器、seed、早停及指标均不变。
改变分块可能改变训练dropout掩码序列，不宣称与旧执行逐位相同，也不改变模型表达能力。
没有结束ToDesk等服务，不以empty_cache作为修复。

## 正确性验收（已完成）

20项直接相关测试通过（CPU，5.71秒），包含原合同、非整除节点尾块、参数/键、嵌套checkpoint和dropout/RNG一致性。
原始输出保存在数据目录`ffn_correctness.txt`。

|确定性分块对比（11节点/chunk4）|前向最大绝对误差|输入梯度最大绝对误差|参数梯度最大绝对误差|
|---|---|---|---|
|FP32|4.7684e-7|5.8208e-10|3.4925e-9|
|BF16 autocast及BF16输入|0|0|3.0518e-5|

FP32使用atol=2e-6/rtol=2e-4，BF16使用atol=2e-3/rtol=3e-2；不把BF16与FP32逐位比较。
同一分块、相同初始RNG、dropout开启时，内层checkpoint开/关（外层仍开启）最大输出/梯度差2.3283e-9，RNG终态相同，梯度有限且非零。
官方decoder实测可训练参数仍为17,821,672；state_dict共252键，键序列SHA256：
`31beab46b45b34ed53ff89a70293a0e9e93abb86721dc51902e9c4c1808af665`。
正式GPU BF16完整路径的梯度、显存与重载验收仍须以下实际流程通过，不能由单元测试代替。

## 资源验收与正式训练门槛（后台执行）

1. 先XYZ，再UV，各重新初始化完整pair检查。每次有效更新先完整AF视频，再ANAKIN fake/real，保留已有AF梯度；失败fake全部75帧，不以冷启动单块代替。
2. 各完成两次更新、几何/VFE/细粗时空梯度检查、原生评价、保存重载以及逐帧检测分数和空间指标重载一致性；验收权重丢弃。
3. 两条件分别落盘`full_pair_<condition>.json`及逐块JSONL，最终两者通过才生成`full_pair_check.json`。
4. 只读取48个既有train/dev缓存坐标，逐实际target8/halo16窗口统计XYZ fine/coarse占据节点数；不运行前端、不用标签/分数选代表。
5. 原高分辨率窗口、已失败视频最高coarse窗口、全队列fine最大与coarse最大窗口去重，最多4个代表窗口，两条件都做完整前向/反向及两次更新；不对全队列做模型资源推理。
6. GPU实际节点数必须与CPU计数一致。记录目标/上下文范围、节点数、操作/layer/shape/dtype、耗时、allocated/reserved。异常保留新算子trace并停止，不自动减规格。
7. 全部通过且源码/配置/commit身份未变化、工作区干净，才自动按原顺序UV→XYZ进行正式初始化、epoch0、训练、最终评价与发布。

当前没有新完整验收显存、耗时或训练结果，不填写旧资源数字冒充本版本通过。

## 身份、缓存与成本

先确认实际HEAD为f704079、工作区初始干净，无任务进程、无根目录及两条件正式pt文件。
旧协议、run.log、状态、launch、退出码1、退出历史及准备成本按原字节归档于：
`derived/pvst_point_voxel_v02/execution_history/initial_8477da4/`。
`execution_revision.json`登记`ffn_chunks_v1`、旧失败commit、归档SHA及新配置SHA；旧exit_history事件保留并继续追加，新事件标明执行版本。
当前protocol由新代码/config/manifest/scale/源码commit重新生成，严格检查没有关闭；资源缓存身份还包括ffn_chunk、执行版本、实现hash、队列/scale及计数文件hash，旧PASS不可复用。

48份complete缓存的媒体/初始化身份、原生header、448映射、revision及无LoRA token声明均已核验兼容，新增特征帧数0。
首次缓存复用核查实测2.855秒；没有重跑MoGe、下载权重或提取DINO。
原失败版本prepare调用515.581秒，完整缓存准备累计529.881秒；旧运行总墙时7515.452秒，口径不同不相加。
新坐标计数、pair验收、代表窗口验收及正式训练/评价各自记录实测成本；未完成项保持未记录。
最终脚本报告会同时引用旧归档成本与新实测成本，不把验收更新算入正式训练。

## 自动运行

入口和环境命令仍为[current_status.md](current_status.md)，tmux新会话`pvst_point_voxel_v02_ffn`，旧失败pane保留。
当前输出目录不变；`run.log/progress.json/launch.json/exit_code.txt`记录新版本；历史原件在上述归档。
结束后自动生成`report_zh.md/result_summary.json`，结果文档普通commit/push；计算COMPLETE与publication状态独立。
如果再次资源不足，保留具体算子与退出历史；不自动改模型、缩目标/halo或重训旧结果。
