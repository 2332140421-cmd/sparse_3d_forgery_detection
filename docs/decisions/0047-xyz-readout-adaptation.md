# ADR-0047：同源B定位读出适配与完整继续训练

状态：用户授权，固定预算匹配实验。

## Context

固定B加入历史改善多数来源，不能由旧A/B差值认定历史导致前景收缩。现在比较冻结表示后的定位读出适配与相同预算完整继续训练。

## Decision

唯一共同起点为ADR-0045的XYZ_TEMPORAL_epoch10（10260步）。每组独立严格恢复模型、完整原顺序AdamW及Python/NumPy/Torch/CUDA随机状态，再设置冻结范围。READOUT_ONLY仅四模块pixel_context/local_out/detail/pixel，FULL_CONTINUE原全部可训练参数；不重新初始化或重置公共动量，不新增结构。冻结参数grad=None，保护权重及优化器状态。
两组都显式用原XYZ_TEMPORAL窗口/temporal=True，原7来源70/30协议、scale/h、BF16、pixel BCE+Dice各1、AdamW和clip1不变。原第11–13轮按seed+epoch无放回同序列，共3078额外更新，末次累计13338。预检副本更新丢弃，正式重新恢复。
共同B仅回放一次，宏/逐来源与原结果绝对偏差超过.001先核查；原A只引用保存结果。各轮固定28训练观察，最终444目标及42固定预览。预测图捕获可选FP32，旧评价默认FP16不变，正式AP始终原生FP32排序。

## Consequences

比较适配策略，不证明读出实现错误或某层存在缺陷。报告AP、固定阈值空间指标和误报，不选best、不新增阈值曲线、前端、来源或训练条件。计算与Git发布状态分开。

## Supersedes

仅本独立对照解除ADR-0046无训练的执行范围限制；不替换ADR-0045/0046的协议、默认行为或历史结果。
