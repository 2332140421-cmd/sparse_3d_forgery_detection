# PVST v0.3 稀疏attention反向限定修复报告

2026-11-04；执行源码 `4f365880bdb7051d255a7ee74c5398b03f30f21c`；当前分支pvst-point-voxel-v03。

**本轮验收状态：FAILED（候选65,536节点阶段SIGFPE）。小规模数值正确且中等规模反向明显改善，但候选未通过大形状可靠性门槛；完整窗口尝试0次，AdamW更新0次，正式模型仍0/2。**

## 已确认的原路径

安装的PyTorch为2.5.1+cu124；FunctionsManual.h中index_backward接收zeros_like_self。更直接的本次N16,384/dim256 CUDA profiler证据：
- 原query512实现的K/V IndexBackward0有64次，梯度输出回填完整[16384,4,64]输入；同形状zero_共96次（还含其他梯度路径），并有同形状add_累加。
- indexing_backward_kernel约374.840ms，占该组Self CUDA约92.02%；对应K/V _index_put_impl_约371.114ms。相同形状zero_合计仅约0.446ms，因此不能把耗时简单等同于清零时间。
- 候选中该K/V IndexBackward路径消失；保留的是4次共享bias索引梯度（小输入[324]或[324,4]）和每head/切片等必要梯度缓冲。同形状zero_仍有24次，不能宣称所有全尺寸缓冲都消失。
这些是单层小组的真实证据，不能把原完整窗口至少827.78秒全部归因于同一个kernel，也不能把该组速度比外推成整模型加速比。

## API、布局与实现

|检查|结果|
|---|---|
|FP32 CSR sampled_addmm|CUDA前向/反向可用|
|原生BF16 sampled_addmm|失败：sampled_addmm_out_sparse_csr not implemented for BFloat16|
|CSR values→COO softmax|梯度保留，Q/K/V/bias均存在有限梯度|
|COO sparse.mm聚合反向|极小CUDA调度记录发现稠密N×N，排除该实现|
|softmax后转CSR sparse.mm|极小CUDA调度未出现稠密N×N，Q/K/V/bias梯度可用|
|更大候选N65,536|原生SIGFPE，可靠性门槛未通过|

实际候选：精确有效边CSR sampled_addmm(Q,K.T)/sqrt64 + relative bias → COO sparse.softmax(逐查询行原有效邻居) → values上的原dropout → CSR sparse.mm(A,V)。局部关闭autocast，仅当前head Q/K/V和必要values转换FP32，输出恢复原BF16/FP32存储；整套模型没有改FP32。没有使用_values、detach或no_grad绕过训练梯度。
复用Neighborhood的整数键与searchsorted，query块内生成全部有效边后按列排序，relative关系ID随同排序，重复边检查通过；缺失边不进入稀疏softmax，等价于负无穷，不能解释为分数0。没有N×N稠密矩阵或E×head_dim显式边特征。大形状稀疏矩阵的逻辑shape是N×N，不等于物化稠密矩阵；极小调度和回归测试同时检查layout。
同帧/t-d/t-2d、27空间邻域、dilation1/2/4、真实PTS、heads/head_dim、bias参数、dropout、projection/residual/FFN及全局context均保持。新增helper只供SparseLayer调用，无可学习参数；bool sparse_edge_attention默认为False，旧入口仍走原实现，无后端注册框架、依赖升级或自定义内核。

## 正确性

12组CUDA比较全部通过：FP32/BF16 × dim256/4head或512/8head × dilation1/2/4。包含负坐标、非均匀PTS、不同帧节点数、空帧、稠密/稀疏邻居、query跨帧及不足chunk尾块；有效边及关系对应精确一致。
运行前固定容差：FP32输出atol2e-5/rtol2e-4、梯度2e-5/3e-3；BF16外层+FP32核心输出.04/.03、梯度.003/.08。没有事后放宽。
|精度|输出max abs|输入梯度max abs|QKV参数梯度max abs|bias参数梯度max abs|投影参数梯度max abs|
|---|---:|---:|---:|---:|---:|
|torch.float32|4.76837e-07|1.74623e-10|9.31323e-10|3.14321e-09|6.98492e-10|
|torch.bfloat16|0.03125|7.62939e-06|3.05176e-05|8.58307e-06|6.10352e-05|

以每组参考梯度最大幅度作尺度，QKV/bias/投影的误差相对尺度：FP32约2.71e-7/4.20e-6/1.42e-7，BF16约0.00889/0.01142/0.01235。另须如实记录：单参数归一化相对误差的最大值，bias为FP32约1.019、BF16约1.378，来自bias末层常数项的近零参考梯度；绝对误差分别仅3.14e-9/8.58e-6。该常数加到每head整行分数，softmax平移不变，理论梯度为0，故不能单凭该相对值判断数值失败。原始逐参数误差保留layer_results.json。
候选dropout=0.1的checkpoint开/关比较通过：输出误差0，梯度有限，RNG重计算正确；不要求新旧边排列使用同一mask。49项相关CPU回归通过（6.64秒），真实官方decoder的完整XYZ模型在CPU仅构建核对身份：17,821,672参数、252个state_dict键，键SHA256 `31beab46b45b34ed53ff89a70293a0e9e93abb86721dc51902e9c4c1808af665`；未前向，BN未改变。

## 单层性能

相同seed、输入、权重、BF16外层、dropout0.1、query512、QKV/FFN8192、外层checkpoint；新旧分别函数顺序运行，预热一次后非profiler同步计时。候选核心使用FP32，这是明确的计算精度差异。第一组另做唯一一组短profiler（旧/新各一次），不把profiler时间用于正式性能表。
|N|dim|实现|E/平均度|CSR拓扑MiB|构造compact/edges ms|前向ms|反向ms|peak allocated/reserved MiB|
|---:|---:|---|---:|---:|---:|---:|---:|---:|
|16384|256|旧query gather|876,024/53.468|13.492|2.442/7.077|17.188|414.589|217.469/284.000|
|16384|256|候选稀疏FP32核心|876,024/53.468|13.492|0.767/9.792|20.003|44.262|241.227/288.000|
|16384|512|旧query gather|876,024/53.468|13.492|0.955/9.868|21.685|462.825|1258.715/1530.000|
|16384|512|候选稀疏FP32核心|876,024/53.468|13.492|1.193/11.604|24.757|63.912|1167.411/1530.000|
|65536|256|旧query gather|3,658,104/55.818|56.318|0.738/42.120|79.662|1609.591|1290.522/1802.000|
|65,536|256|候选|未完成记录|未完成记录|未完成记录|未完成记录|未完成记录|未完成记录|

N16,384时，dim256反向0.414589→0.044262秒（约9.37倍），dim512为0.462825→0.063912秒（约7.24倍）。前向略慢；两组反向均达到预声明≤旧版80%的门槛。只以这些已完成组陈述改善。
CSR拓扑存储计rowptr、有效col及relation ID：8(N+1)+16E字节。score/probability、COO转换和FP32 head临时workspace另占内存，不能把拓扑字节当总峰值。新旧没有显式同时保存两套模型引用；峰值在每组reset_peak后记录，但同一worker顺序执行且未单列组起始allocated，reserved受分配器历史缓存影响，不依据这些峰值宣称候选净显存优势。

## 大形状失败与停止

单层worker墙时13.318秒，真实退出码 **-8（SIGFPE/信号8）**，未达到290秒限时；连同先行极小能力检查仍远低于约300秒预算。不是TIMEOUT，也没有CUDA OOM异常记录。
最后完整保存的是旧query版N65,536/dim256（E3,658,104，平均度55.818）。随后候选同形状阶段原生终止，没有Python异常栈或阶段边界，因此不能声称具体失败于sampled_addmm、softmax或SpMM中的哪一个。现有apport目录未找到本任务可用core，内核日志读取受限；未重试、不换layout/chunk/精度去绕过。
partial layer_results.interrupted.json及layer_process.json/日志保留；父进程状态处理已补必要回归，原生非零退出不能留下似乎通过的门槛记录。没有重跑GPU来修补数据。N65,536/dim512未执行，完整模型900秒复验**未进入（0次）**，AdamW更新**0次**。旧900秒TIMEOUT结果原样保留，不伪造总加速比。

## 决策、资产与发布

该计算方向在已完成小组有明确反向改善，也确认了原K/V索引反向的成本；但当前候选在较大形状原生SIGFPE，未满足稳定性/后端支持门槛。**不建议在4090上启动正式训练或完整窗口重跑；下一步应先定位并解决这次原生崩溃，再由用户授权有限复验。** 不升级依赖、不写CUDA/Triton内核，不追加模型或训练。
正式config_v03.json的XYZ仍null，原数据/损失/时序/宽度/模型初始化未变，旧默认入口不静默使用候选。无下载、前端或DINO提取，无检测AUROC/Dice成绩；正式模型仍0/2。
输出：`/media/njupt/8FCD3869E2FAB06F/chenyh/data/sparse_3d_forgery_detection/derived/pvst_sparse_attention_check_v1`。capability/layout_dispatch、layer_identity、layer_results及interrupted副本、layer_process、日志与测试记录留数据目录；Git只提交代码、必要测试、协议说明和此小型结果。用户原有无关JSON改动保留。计算状态与发布状态分开记录。
