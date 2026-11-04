# PVST候选SIGFPE定位与单一修复边界

2026-11-04；用户本轮明确授权。只复现原N65,536/dim256/4heads单层候选（含预热），不跑旧baseline、旧12组或完整模型/AdamW/900秒窗口。
旧证据目录pvst_sparse_attention_check_v1只读，新目录pvst_sparse_attention_sigfpe_v1。

直接入口research_tools/pvst/sparse_sigfpe.py，调用现有SparseLayer、Neighborhood、bounded_process；输出阶段JSONL和gdb原生栈。原benchmark缺少算子落盘和原生信号捕获，不能靠其finally收集SIGFPE，因此新增此小型诊断入口，不建设诊断平台。

原图单帧32×32×16，4帧，PTS[0,.031,.10,.187]；N必须65,536、E必须3,658,104。实际检查CSR长度/单调性/nnz、列范围/排序/重复、relation帧映射，记录int64索引与逻辑N²，不预设整数溢出。

sparse_head只增加默认None的trace回调；诊断进程局部绑定，正常路径无同步/文件写入。按head落盘CSR/SDDMM/COO/bias/softmax/dropout/CSR/SpMM边界，cuda synchronize仅在这些边界；有限autograd前后hook不修改梯度或读取巨大梯度。
首次复现使用已安装gdb启动自己的子进程，父进程90秒上限，PYTHONFAULTHANDLER=1与CUDA_LAUNCH_BLOCKING=1只限诊断。gdb捕获SIGFPE bt后传递原信号；不附加其他用户PID、不修改系统core/security/驱动。

所有本轮GPU子进程墙时累计预算300秒，先记录诊断支出；只有定位明确才实施一种等价表示/调用修复。若无修复则报告停止，不能以改long、N减1或改epsilon声称解决。
如形成修复：只跑修改相关的小规模输出/输入/QKV/bias梯度、FP32和BF16外层、非均匀PTS/空帧、dropout checkpoint及无稠密N×N检查，沿用既有容差；再独立进程验证65,536/256、65,536/512，均过才测实际131,072/512（单帧32³共4帧）。不复制原benchmark错误的固定N构造。
性能阶段关闭诊断回调/hook/launch blocking，必要预热，不重跑旧baseline、不强算优化比。任何组失败或预算不足就停止后续放大。候选仍默认False，正式XYZ=null、模型0/2。

源码各阶段提交后执行，运行期间不改；用户无关JSON改动保留。最终小型结果/报告普通commit/push，计算与发布独立；不提交core、大日志或权重。

## 已定位后唯一修复（验证前登记）

首次原形状前向成功，在预热backward的sampled_addmm梯度→sparse_mask→mul_sparse→intersection kernel SIGFPE，gdb取得原生栈；int64索引/CSR结构检查均通过。
官方v2.5.1 SparseBinaryOpIntersectionCommon.h的std::accumulate使用int初值1，结果类型为int，N²=2³²截断成0，再做nnz除法触发整数除零。同逻辑尺寸单非零元素的CUDA稀疏乘法最小复现也退出-8。

仅在逻辑行列乘积低32位为0时，增加1个没有任何边的逻辑列，并为该head的K/V增加显式零填充行。行数/真实节点/边/relative映射均不变；空列在稀疏softmax中不存在，输出及原输入梯度不变。它是存储padding，不是新增观察、模型节点或缺失值修复。没有每query块K/V gather。
65,536实际节点保持65,536，逻辑矩阵变为[65536,65537]；131,072对应[131072,131073]，避免触发本库除零分支。不改系统库或版本，不能称上游整数溢出已普遍修复。
小规模回归强制走同一padding分支（FP32/BF16两组、非均匀PTS/空帧、dropout checkpoint及no-dense调度检查），不重跑旧12组。
随后独立进程按65,536/256→65,536/512→131,072/512验证，任一失败停止。计时无gdb、hook、launch blocking；原图/PTS、BF16外层、FP32核心、query512、QKV/FFN8192、原预热和外层checkpoint保持。
