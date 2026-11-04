# PVST逐head输出CPU暂存：限定验收计划

本轮只增加默认False的head_output_offload存储开关。原sparse_head/checkpoint返回后，同步to(cpu)保留ToCopyBackward连接，CPU列表仅存CPU输出并删除h局部引用；CPU stack/reshape维持原head顺序，完整attended一次回原GPU。dtype不变，不改CSRSoftmax、CSRFromValues、SpMM、QKV、尺度、节点、边、128/256/512、24帧上下文或损失。

CPU直接相关8项检查通过：输出/全参数梯度、dropout与内外checkpoint、顺序/reshape、默认关闭、权重/state_dict。CUDA再对照FP32无dropout/有dropout和BF16有dropout，检查CPU requires_grad/ToCopyBackward0及完整布局；不重跑旧CSR或SIGFPE矩阵。

复用csr_values_bridge_check.benchmark，仅给它增加本轮out和head_output_offload可选参数；默认调用保留旧行为。唯一匹配规模实际N131072/E7475256、dim512/8head/64，每组独立进程、一次预热后同步计时。两组均用当前CSR＋桥接，不用旧COO。大图核对全部参数梯度，输出/输入梯度固定每2048节点一行64行；小图全量核对。

本轮专用入口 `python -m research_tools.pvst.head_output_offload_check` 复用已有bounded_process，累计小型检查限300秒。正确性/布局/一致性/真实CPU存储通过，无OOM且单层峰值RSS增长≤预先记录1GiB安全界后，继续一次max_window_check --head-output-offload。全局峰值是否下降如实报告，不以调用cpu或reserved变化放大结论；不搜索其他规模/chunk/方案。

唯一完整窗口仍video782426ea0330c7fea684953c、ANAKIN/train/real/an:4wdo37、720×1280，history[104,120)、target[120,128)，fine1/8/coarse1/4；fine2970629/coarse1488042。原seed/BF16/逐head FP32、attention512/QKV8192/FFN8192/point32768/tile256、外层saved-tensor CPU卸载与一次AdamW保持，head暂存显式开启。worker启动起900秒由原父进程限时，无重试。六层前向实际CPU存储和拼接设备、原形状/dtype/梯度连接、复制payload、回传前后显存均核对；关键阶段及失败上下文复用原记录。

源码在GPU检查前提交，运行期间不改代码。所有证据写derived/pvst_head_output_offload_check_v1，自动报告和普通commit/push，计算/发布分列。旧产物/协议/退出码/config不改；用户无关JSON保留，不提交/stash/reset；正式XYZ仍null，资源模型不保存checkpoint，正式模型0/2。不追加训练或其他资源优化。
