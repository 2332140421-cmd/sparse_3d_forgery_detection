# PVST稀疏attention反向限定检查

2026-11-04用户授权；起点53e59c36b82e48658b5c58d13fbe5abfca70df2c，分支pvst-point-voxel-v03。
不改模型方法、输入尺度、loss、数据或正式XYZ=null；无训练或效果指标。
用户无关历史检测JSON改动原样保留，不要求整个工作区干净。

## 待验证路径与最小实现

原SparseLayer每查询块对完整K/V用index.Tensor，安装包FunctionsManual.h的index_backward声明接收zeros_like_self；是否实际反复分配完整输入尺寸buffer，以本轮短profiler核对，不能据此归因全部827秒。

本机torch2.5.1+cu124极小CUDA实测：FP32 sampled_addmm(CSR)→values→COO softmax→mm的Q/K/V/bias梯度存在且有限；原生BF16 sampled_addmm报未实现。
额外极小TorchDispatch形状核对发现COO mm反向会生成稠密N×N，CSR聚合反向未出现；因此候选必须用CSR SpMM。

新增sparse_attention.py只直接供SparseLayer调用，输出原同形状attention结果；旧逐查询gather不能通过配置本身消除其反向全量buffer，需要这一小型函数辅助。不是新模型模块，无参数或后端注册框架。

- Neighborhood原坐标/整数键/query规则不变；逐query块筛全部有效边，按列排序且relation ID同步置换；保留共享[T,81,4]相对关系。
- 保留CSR rowptr、有效col、relation ID，不保留N×N、E×head_dim特征或全部层的N×81×4展开。
- 逐head局部关闭autocast：CSR sampled_addmm(Q,K.T)/sqrt64 +原relative bias；COO sparse.softmax仅对该行原有效边归一化；对values执行原dropout；转CSR sparse.mm(A,V)。public values()保梯度，无_values/detach/no_grad绕开梯度。
- 每head仅转换必要Q/K/V为FP32，结果返回原特征dtype；不用所有head同时FP32复制。主模型其余BF16不变，必须记录这一精度差异。
- 原输出投影/残差、FFN、dilation1/2/4、时空/全局context、BN及初始化保持；bool sparse_edge_attention默认为False保持旧配置，只有独立候选设True，不能静默退回。

## 运行前确定的检查与门槛

小规模含负坐标/非均匀PTS/不同帧节点数/空帧/密稀邻居/跨帧query/尾块，dim256/4heads及512/8heads，三种dilation。
FP32输出atol2e-5/rtol2e-4、梯度atol2e-5/rtol3e-3；BF16外层+FP32稀疏核心输出atol.04/rtol.03、梯度atol.003/rtol.08。最大绝对误差及相对reference最大幅度均记录，运行后不放宽。
固定候选dropout开启，比较checkpoint开/关的RNG重算及梯度；新旧布局不强求相同mask。检查参数/state_dict不变，并加隐式N×N反向回归保护。

单层检查由sparse_attention_check.py独立父进程复用既有限时/进程组机制管理290秒，加先行极小CUDA能力检查约3秒，总预算约300秒。
先N16,384/dim256，新旧分别驻留、相同seed/输入/权重/损失/BF16/dropout0.1及外层checkpoint；预热一次，非profiler同步测量一次前向/反向。
同一首组另做短profiler，区分index/slice backward、full-size zeros及候选稀疏算子，不对更多形状生成trace。
预声明门槛：候选反向≤旧版80%才继续，否则停止；不选最好一次，不搜索chunk/精度/卸载。
通过后仅再测N16,384/dim512及N65,536/dim256/512；每组都记录E/平均度、拓扑量、构图/前向/反向时间、峰值内存。
所有门槛都通过，才接入独立1/8全模型配置并在新目录最多复验一次原窗口900秒；正式入口和旧结果不变。未完成更新不算总加速比。

输出独立目录：
`/media/njupt/8FCD3869E2FAB06F/chenyh/data/sparse_3d_forgery_detection/derived/pvst_sparse_attention_check_v1/`。
必要结果只记录小型JSON/中文文档；缓存/权重/大日志不进Git。
