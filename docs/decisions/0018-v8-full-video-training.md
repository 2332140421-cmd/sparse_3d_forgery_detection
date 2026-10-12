# ADR-0018：完整视频监督与跨块梯度检查点

用户授权；独立分支`v8-full-video-training`。不废弃ADR-0017冻结评价。

本次问题是窗口预训练后，完整视频监督是否改善视频排序与固定阈值分类。
调用方为`research_tools.v8.full_video_training.run`；输出为六个epoch20模型、
完整视频分数与报告。旧inference_mode逐帧接口无法保留梯度，因此新增薄的可微
chunk适配，直接调用已有patch编码、真实spatial/region attention、GRU与head。

checkpoint(use_reentrant=False)显式传入CPU块、上块hidden/active与首块标识，
不detach。完整视频累计未归一化LSE后计算一次BCE；同source两视频各乘1/2，
统一更新。纯计算函数不写状态；每个source pair完成后原子保存恢复状态。
这不是TBPTT，也不是块级弱监督。不增加模型参数或新缓存框架。

AGENTS无需改写；延续V8隔离边界。完整资源测试失败时如实停止，不改变监督。
