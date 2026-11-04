# 重复节点切片反向：限定取证与split修复

先读当前HEAD51d0f298cd4aeadcb1944d7ef3308d0f149974f0及既有日志。CPU实际小图显示[41,256]按11/7切片分别4/6次完整输入slice_backward，split为共享SplitBackward0；实际SparseLayer原路径40次对应分配。没有据形状就认定OOM根因。

原HEAD完整诊断一次：同原head QKV/CPU head/外层CPU卸载、BF16/CSR/chunk/初始化/数据/损失，仅启动前TORCH_SHOW_CPP_STACKTRACES=1。原窗口再次OOM，C++栈明确SliceBackward0→slice_backward_wrapper→at::native::slice_backward→zeros→CUDA allocator malloc，确认1.42GiB请求来自切片完整梯度零缓冲。未开allocator history/profiler，异常文本不含确切input_sizes，不能指定哪一个入口或fine/coarse；保留证据边界。原诊断进程已结束后才修改源码。

本轮只增加默认False的node_split及node_parts共享小方法，立即调用者仅streamed_head norm+QKV、attention投影/残差两个输入、FFN三处大激活分块。一次原生split维持顺序/chunk/尾块与原checkpoint/RNG，不复制输入、不改多输出view、不用CopySlices或自定义反向。不替换参数切片或全仓切片。

11项CPU直接相关检查通过：256/512、FP32/BF16、dropout内外checkpoint、尾块、全部参数和输入梯度、参数身份/数量/键与默认关闭；共享native split视图、目标全节点slice_backward消除。保留原容差，无初始化/损失修改。

独立入口python -m research_tools.pvst.split_backward_check复用已有bounded_process、benchmark和max_window worker，读取已完成原诊断，绝不再启动第二次诊断。四组必要CUDA小图→唯一N131072/E7475256、dim512/8head匹配单层，二者均逐headQKV/CPU head/CSR＋桥接，只node_split不同。通过正确性/布局/配对后，同任务一次max_window_check --node-split，worker900秒，无重试，无第三次窗口。

唯一窗口仍video782426ea0330c7fea684953c，ANAKIN/train/real/an:4wdo37，720×1280，history[104,120)、target[120,128)，fine1/8/coarse1/4，节点2970629/1488042。seed20260909、attention512/QKV8192/FFN8192/point32768/tile256、128/256/512、全部点/像素/边、原AdamW/监督保持。正式XYZ=null，资源权重丢弃无checkpoint，正式模型0/2。

原诊断用旧HEAD，候选用提交后新HEAD；运行期间不改源码。derived/pvst_split_backward_check_v1分目录保留两份协议/实际退出码及日志。原资产、退出历史/config及用户JSON保留。自动中文报告、小JSON、状态页及普通push，计算/发布分列；本轮不训练、不叠加其他优化。
