# 固定CSR values桥接：限定执行计划

本轮只完成CSRFromValues构造桥接，将原边values梯度从SpMM返回的CSR梯度取出。shape/crow/col逐项核对（CUDA异步断言）；不靠nnz假定顺序，不稠密化。None及空CSR零梯度单独处理。不重写SpMM、sampled_addmm或softmax设计，无新增参数。

已在极小CUDA图确认SpMM返回相同CSR结构，完整head不再_to_dense；CPU直接相关21项测试通过，含values/V FP64 gradcheck、显式零边、空行/空边、无梯度、错误结构拒绝、完整层checkpoint布局。标准CSR.values()取出反向未见稠密化，不另加取值桥接。

独立入口 `python -m research_tools.pvst.csr_values_bridge_check`：小CUDA正确性→旧COO与新CSR唯一规模131072/512/8head单层比较→门槛通过才最多一次max_window_check --values-bridge。旧COO仅放tests作本次比较参考，模型代码不依赖tests。小GPU子进程累计290秒，为前置极小探针留约10秒；每个阶段独立进程且不重试。

执行前固定门槛：既有输出/梯度容差全部通过、完整前向/反向/checkpoint无逻辑稠密矩阵、peak allocated实测下降，反向耗时≤旧1.25倍。大图比较全部参数梯度，输出/输入梯度使用固定每2048节点一行64行样本，明确不称大图逐元素核验。

完整窗口仍为ANAKIN/train/real/an:4wdo37，video782426ea0330c7fea684953c，720×1280，历史[104,120)、目标[120,128)，fine1/8/coarse1/4，节点2970629/1488042。原seed/BF16/逐head FP32、attention512/QKV8192/FFN8192/point32768/tile256、外层CPU卸载、初始化、损失与一次AdamW保持。worker由既有父进程900秒限时。记录六层实际CSRSoftmax/CSRFromValues执行及更新证据。权重丢弃，正式模型0/2，正式XYZ仍null，候选默认关闭。

所有新证据在derived/pvst_csr_values_bridge_check_v1，旧报告、退出历史、协议、缓存和用户无关JSON保持。代码先提交，运行期间不修改；报告普通提交推送，计算与发布分开。失败停止，不叠加其他优化、不自动训练。
