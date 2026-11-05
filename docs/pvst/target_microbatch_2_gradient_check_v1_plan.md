# PVST v0.3 两帧微批梯度归因与限定资源检查规则

复用既有小图夹具与微批入口，新增一个诊断选项，不新增模型/调度模块。
FP32优先，再BF16；总小图worker限300秒。只有两者都通过才允许唯一900秒最大窗口worker，最多一次AdamW。
旧FAILED与旧目录只读保留，新目录pvst_target_microbatch_2_gradient_check_v1。

A：24帧联合损失梯度。B：独立同权重24帧前向图，每2目标帧损失乘2/8，
autograd.grad求四组梯度，CPU FP64累加。C：四个18帧图各反向，叶参数hook记录原始组贡献，CPU FP64累加，同时保留生产式.grad。
仅小图B允许retain_graph，真实worker保持逐组释放/累计梯度、不保留图。
所有Dropout含attention/MHA关闭、同非零诊断头；资源恢复规定初始化/Dropout。

比较固定沿用旧设置：FP32输出atol2e-5/rtol1e-5、梯度2e-5/3e-5；
BF16输出0.02/0.02、梯度0.2/0.03。actual为第一个参数、reference为第二个参数，
逐元素abs(actual-reference)≤atol+rtol×abs(reference)，有限性与缺失梯度分别记录。
先完成所有参数摘要及超差元素再判定，既有偏置index6/19即使未超差也保留数值。

必要放行条件：B/C逐组输出、加权loss与参数梯度，B_sum/C_sum、C实际.grad/C_sum及C实际.grad/B_sum均通过。
原A/C通过则按原参考通过；若A/C超差，对每一超差元素记录
原差C-A、归约差B-A、剩余C-B。仅当归约差与原差同向且剩余满足原梯度容差（reference=B_sum）时，
可标记基于分组参考通过。A/C原结果不改写，不忽略任何偏置或后续参数。
保存所有参数的max/count/缺失/零梯度摘要和必要超差元素（含A/B/C总值及四组贡献），不存完整梯度张量。

真实窗口仍为782426ea0330c7fea684953c，目标[120,128)，四组各16历史＋2目标。
原模型/全部节点边/像素/PTS/全局scale/CSR/K.float/精度/config不变；XYZ正式尺度null，正式模型0/2。
失败不重试、不改组大小、不启动训练。计算和Git发布分列。
