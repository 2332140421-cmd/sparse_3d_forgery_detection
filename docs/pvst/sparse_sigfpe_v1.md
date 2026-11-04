# PVST v0.3 稀疏attention SIGFPE定位与最小修复

2026-11-04；分支pvst-point-voxel-v03；修复验证源码 `18e45afb4ffd587d77cb768aeb8c6cb358ac1cf5`。

**状态：FIX_VALIDATED（限定单层修复验证通过）。65,536两个宽度及实际131,072/512均完成前向/反向；未做完整模型、AdamW或900秒窗口。候选仍默认关闭、正式XYZ=null、正式模型0/2。**

## 根因证据

1. 首次诊断保留原65,536/dim256完整4head候选、预热、dropout0.1、BF16外层/逐head FP32和内外checkpoint。原图为32×32×16坐标×4帧，PTS[0,.031,.10,.187]，实际N65,536、E3,658,104；CSR rowptr长度/单调/末值、列范围/排序/无重复、relation帧对应检查全通过，crow/col/relation均int64。
2. 算子边界同步及立即落盘显示：原层预热前向完成；进入backward后，第4个head（head=3）的sampled_addmm反向prehook之后终止，无posthook。此前该head的SpMM、布局转换、softmax反向均已完成。因此不是softmax前向报错或第一次模型构建失败。
3. 已安装gdb直接启动本任务子进程，捕获SIGFPE原生栈。关键调用链：sampled_addmm梯度处理 → sparse_mask_sparse_compressed → mul_sparse_csr / mul_sparse → mul_sparse_sparse_out_cuda_kernel → _sparse_binary_op_intersection_kernel_impl。faulthandler也记录_engine_run_backward。调试器自身退出255；gdb明确记录inferior由SIGFPE终止，不能把255写成模型正常退出。
4. 同逻辑[65536,65536]、两个各仅1个非零且值为1/2的coalesced CUDA COO张量相乘，再次在最小算子中触发SIGFPE，独立进程真实returncode=-8。这不依赖模型初始化、复杂图、NaN或梯度爆炸，也不是OOM。此为有目的的单算子根因确认，不是重复整层尝试。

匹配本机PyTorch 2.5.1+cu124的[官方v2.5.1源码](https://github.com/pytorch/pytorch/blob/v2.5.1/aten/src/ATen/native/sparse/SparseBinaryOpIntersectionCommon.h#L200-L211)中，交集kernel用`std::accumulate(..., 1, std::multiplies<int64_t>())`计算sparse_dim_numel，随后执行`larger._nnz() / sparse_dim_numel`。初值1使accumulate返回/累加类型仍为int；乘法functor虽为int64，中间结果赋回int时65536²=2³²变为0，继而整数除零。int64的CSR索引不能修复这个独立的内部累加类型问题。
直接观测包括原生栈、最后算子边界、有效索引和最小SIGFPE；具体int累加/除零机制由匹配源码和尺寸算术共同解释。没有伪造寄存器值，也没有把所有底层稀疏问题都归为同一根因。原始栈及阶段JSONL留新数据目录，不提交完整大日志/core。

## 唯一修复及语义

只在逻辑行列乘积低32位为0时，为稀疏表示增加1个**无任何边的逻辑列**，并给当前head的K/V追加显式零填充行。它是存储padding，不是新增三维观察、模型节点或缺失值修复。图仍有N个真实节点，CSR所有col仍<N，输出仍N×head_dim。
- 65,536真实节点保持65,536，算子逻辑shape变为[65536,65537]；131,072对应[131072,131073]。逻辑尺寸乘积的低32位分别为65,536/131,072，不再成为零除数。
- 每个query的所有原有效邻居、relative bias和完整行softmax均保持；填充列没有稀疏边，在softmax中不存在，不作为分数0加入。SpMM输出与原有效节点的梯度不变，填充行不进模型参数。
- 原head数/head_dim、精度、dropout、projection/residual/FFN、PTS/因果邻接均保持。仅当前head增加一次存储cat，不做每query块对完整K/V的gather，不重新引入已移除的K/V IndexBackward旧路径。
- 没有detach/no_grad、nan_to_num、epsilon或初始化修改；不升级/修改系统PyTorch/CUDA，不写内核，也未将65,536节点降为65,535。
- 这是针对已确认零除边界的等价规避，不声称上游32位启发式对所有形状普遍正确，也不是正式后端放行。

## 必要回归

只重跑与padding修改直接相关的部分，未重跑旧12组或旧baseline。小图强制使用同一padding分支：FP32/dim256和BF16外层+FP32核心/dim512，含非均匀PTS、稀疏/密邻居和空帧。沿用既有预声明容差，未事后放宽。
|精度|输出max abs|输入梯度max abs|QKV参数梯度max abs|relative bias梯度max abs|投影梯度max abs|
|---|---:|---:|---:|---:|---:|
|torch.float32|4.76837e-07|1.16415e-10|9.31323e-10|2.09548e-09|6.98492e-10|
|torch.bfloat16|0.03125|5.72205e-06|7.62939e-06|3.82587e-06|1.52588e-05|

输出误差相对参考最大幅度：FP32约1.23e-7、BF16约0.00725；输入梯度相对尺度约1.43e-7/0.01402。逐参数绝对及相对幅度完整保留regression.json，不用近零参考的相对值掩盖绝对误差。
候选dropout=0.1的checkpoint开/关输出误差0、梯度有限，RNG重计算通过；强制padding的小CUDA回归及CPU TorchDispatch回归检查均拒绝稠密N×N或N×(N+1)张量，不以小图没OOM代替layout检查。10项直接相关CPU测试通过（2.74秒），包括padding边界、真实131,072图构造和trace不改输出/梯度。无新增可学习参数或state_dict键，候选开关和原初始化保持。

## 原失败形状及唯一更大规模

各组用独立进程，前组通过才继续。真实图构造N/边数均核对；131,072使用单帧32³坐标×4帧，实际达到要求，不调用只支持旧固定N的benchmark。所有参数和输入梯度存在且有限，完成原预热后再同步计时。计时无gdb、trace/hook、CUDA_LAUNCH_BLOCKING或profiler，BF16外层/FP32核心、dropout0.1和外层checkpoint保持。
|实际N|dim/heads|E|逻辑稀疏shape|前向秒|反向秒|峰值allocated/reserved GiB|退出|
|---:|---:|---:|---|---:|---:|---:|---|
|65,536|256/4|3,658,104|[65536, 65537]|0.057783|0.154458|0.897/1.225|PASS / 0|
|65,536|512/8|3,658,104|[65536, 65537]|0.068116|0.295710|1.225/1.449|PASS / 0|
|131,072|512/8|7,475,256|[131072, 131073]|0.155698|0.852404|2.437/2.977|PASS / 0|

旧65,536/dim256 baseline的已存反向1.609591秒可作有条件参考，新结果0.154458秒；本轮未重跑baseline，且旧记录处于含profiler历史的共享worker，新记录独立进程、增加了padding，因此不强算统一加速比或净显存节省。旧16,384的9.37/7.24倍数据和旧原生-8历史均原样保留。

## 成本与决策

所有GPU子进程墙时累计 **22.165秒 / 300秒**：首次gdb诊断7.422秒、最小算子1.266秒，余下为小回归和三组独立验证。预算包含进程导入/构建等CPU开销，未自动重试。gdb调试参数仅限首次定位，不进入最终计时。
**已完成这一处SIGFPE的最小修复及限定单层验证，值得由用户下一轮授权一次完整窗口复验；本轮不执行。** 131,072通过不代表百万节点完整模型已可训练，也未证明4090能完成真实最大窗口的全部路径/AdamW。
正式候选默认关闭、XYZ尺度仍null、正式模型0/2。没有完整模型前向、AdamW、前端或DINO提取，无检测性能指标。用户无关JSON改动保留。原报告/部分结果/退出码/日志未改，所有新证据写入：`/media/njupt/8FCD3869E2FAB06F/chenyh/data/sparse_3d_forgery_detection/derived/pvst_sparse_attention_sigfpe_v1`。
关键文件：diagnostic_stages.jsonl、native_stack.log、diagnostic_graph/process、minimal/worker.log/process、regression.json、validation_*.json及validation_summary.json、budget.json、源码身份。Git只保留小型结果和此报告，不含core、权重或完整大型日志。计算状态FIX_VALIDATED与Git发布状态分别记录。
