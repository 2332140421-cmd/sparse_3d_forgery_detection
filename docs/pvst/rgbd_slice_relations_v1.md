# 预训练RGB-D编码与深度切片关系匹配实验

状态：FAILED。两个条件均使用深度，非纯二维/三维对照。

数据：{"train": {"pairs": 8, "videos": 16, "frames": 2138, "positive_frames": 1065, "all_nonempty_fake": 7, "mixed_fake": 1}, "dev": {"pairs": 4, "videos": 8, "frames": 924, "positive_frames": 462, "all_nonempty_fake": 4, "mixed_fake": 0}}

官方原版DFormer-S RGB-D ImageNet预训练，保留多尺度64/128/256/512、stride4/8/16/32，编码器1e-5微调，BN统计固定且仿射可训练；多尺度局部融合256，逐级上采样/跳连、原生RGB细节卷积及独立帧头。共同初始化精确复制，B额外4×128池化和两组512维8head显式时间→深度关系，像素/帧增量投影零初始化。

官方加载记录：/media/njupt/8FCD3869E2FAB06F/chenyh/data/sparse_3d_forgery_detection/derived/pvst_rgbd_slice_relations_v1/official_load.json；全部保留编码器strict加载；忽略分类pred及官方分割本就删除的末端仅分类深度更新，已对照证明RGB多尺度输出一致。

新MoGe-2：完整宽高比输入长边≤768、不放大小视频；num_tokens=round(input_h*input_w/14²)，实际内部网格与输出记录在每视频complete.json。旧256场不进入模型。原生插值回映不是独立测量。encoder_depth采用官方Greys_r灰度处理及(.48,.28)标准化；slice_depth始终Z/固定s，h.125,d0=0，非米。

片段清单严格复用上轮首/中/尾24帧，每片段全帧等权损失并只更新一次；A逐帧梯度累积，B完整因果关系反向。frame BCE+pixel BCE+softDice等权，空GT沿用epsilon1。最多30轮、最少10轮、开发Dice耐心8。阈值logit≥0，视频frame log-mean-exp温度1。

训练监测仅固定第一个训练pair的真实反向首段，不将未训练时段冒充拟合。开发完整视频24帧因果窗口，尾段补左上下文、每帧一次、边界重置；不声称无限历史/精确时间边界。pixel AP为正GT帧AP再fake视频宏平均，不是AP@IoU。

必要检查：/media/njupt/8FCD3869E2FAB06F/chenyh/data/sparse_3d_forgery_detection/derived/pvst_rgbd_slice_relations_v1/checks.json；独立拟合诊断：/media/njupt/8FCD3869E2FAB06F/chenyh/data/sparse_3d_forgery_detection/derived/pvst_rgbd_slice_relations_v1/fit/result.json（检查权重丢弃）。

|条件|best/结束epoch|Dice|IoU|pixel AP|GT召回|real像素FPR|frame AUROC/AP|video AUROC/AP|

|---|---|---|---|---|---|---|---|---|

B−A：{}

逐fake视频：[]

定位错误证据：分别检查编辑区召回（漏覆盖）、real像素FPR（背景误报）及固定样例。低误报不能单独证明定位成功；本报告不凭数值虚构肉眼边界结论。共同样例身份/PTS/概率色阶固定，不按预测挑选。

限制：相机运动与逐帧几何可比性未解决；单seed小队列不能证明三维增益，或把切片关系收益外推到其他数据。开发fake全程编辑，无法检验复杂时间切换。

完整产物：`/media/njupt/8FCD3869E2FAB06F/chenyh/data/sparse_3d_forgery_detection/derived/pvst_rgbd_slice_relations_v1`；计算与Git发布状态分开。错误：Traceback (most recent call last):
  File "/media/njupt/8FCD3869E2FAB06F/chenyh/projects/sparse_3d_forgery_detection/research_tools/rgbd_slice/experiment.py", line 292, in main
    checks(c,out,manifest,progress);fit=fitting(c,out,manifest,progress)
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/media/njupt/8FCD3869E2FAB06F/chenyh/projects/sparse_3d_forgery_detection/research_tools/rgbd_slice/experiment.py", line 102, in checks
    with torch.no_grad():pa=forward(a,fs)
                            ^^^^^^^^^^^^^
  File "/media/njupt/8FCD3869E2FAB06F/chenyh/projects/sparse_3d_forgery_detection/research_tools/rgbd_slice/experiment.py", line 52, in forward
    with torch.autograd.graph.save_on_cpu(pin_memory=True),torch.autocast('cuda',dtype=torch.bfloat16):return model(fs)
                                                                                                              ^^^^^^^^^
  File "/media/njupt/8FCD3869E2FAB06F/chenyh/envs/v8/lib/python3.12/site-packages/torch/nn/modules/module.py", line 1736, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/media/njupt/8FCD3869E2FAB06F/chenyh/envs/v8/lib/python3.12/site-packages/torch/nn/modules/module.py", line 1747, in _call_impl
    return forward_call(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/media/njupt/8FCD3869E2FAB06F/chenyh/projects/sparse_3d_forgery_detection/research_tools/rgbd_slice/model.py", line 138, in forward
    args=(local,f['rgb'],ctx);pixel=checkpoint(self.local.decode,*args,use_reentrant=False) if self.training else self.local.decode(*args)
                                                                                                                  ^^^^^^^^^^^^^^^^^^^^^^^^
  File "/media/njupt/8FCD3869E2FAB06F/chenyh/projects/sparse_3d_forgery_detection/research_tools/rgbd_slice/model.py", line 64, in decode
    x=self.native(torch.cat((x,detail),1));x=x+context.to(x.dtype)
                                             ~^~~~~~~~~~~~~~~~~~~~
RuntimeError: Expected all tensors to be on the same device, but found at least two devices, cuda:0 and cpu!

