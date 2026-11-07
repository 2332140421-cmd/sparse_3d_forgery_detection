# 预训练RGB-D编码与深度切片关系匹配实验

状态：COMPLETE。两个条件均使用深度，非纯二维/三维对照。

数据：{"train": {"pairs": 8, "videos": 16, "frames": 2138, "positive_frames": 1065, "all_nonempty_fake": 7, "mixed_fake": 1}, "dev": {"pairs": 4, "videos": 8, "frames": 924, "positive_frames": 462, "all_nonempty_fake": 4, "mixed_fake": 0}}

官方原版DFormer-S RGB-D ImageNet预训练，保留多尺度64/128/256/512、stride4/8/16/32，编码器1e-5微调，BN统计固定且仿射可训练；多尺度局部融合256，逐级上采样/跳连、原生RGB细节卷积及独立帧头。共同初始化精确复制，B额外4×128池化和两组512维8head显式时间→深度关系，像素/帧增量投影零初始化。

官方加载记录：/media/njupt/8FCD3869E2FAB06F/chenyh/data/sparse_3d_forgery_detection/derived/pvst_rgbd_slice_relations_v1/official_load.json；全部保留编码器strict加载；忽略分类pred及官方分割本就删除的末端仅分类深度更新，已对照证明RGB多尺度输出一致。

新MoGe-2：完整宽高比输入长边≤768、不放大小视频；num_tokens=round(input_h*input_w/14²)，实际内部网格与输出记录在每视频complete.json。旧256场不进入模型。原生插值回映不是独立测量。encoder_depth采用官方Greys_r灰度处理及(.48,.28)标准化；slice_depth始终Z/固定s，h.125,d0=0，非米。

片段清单严格复用上轮首/中/尾24帧，每片段全帧等权损失并只更新一次；A逐帧梯度累积，B完整因果关系反向。frame BCE+pixel BCE+softDice等权，空GT沿用epsilon1。最多30轮、最少10轮、开发Dice耐心8。阈值logit≥0，视频frame log-mean-exp温度1。

训练监测仅固定第一个训练pair的真实反向首段，不将未训练时段冒充拟合。开发完整视频24帧因果窗口，尾段补左上下文、每帧一次、边界重置；不声称无限历史/精确时间边界。pixel AP为正GT帧AP再fake视频宏平均，不是AP@IoU。

必要检查：/media/njupt/8FCD3869E2FAB06F/chenyh/data/sparse_3d_forgery_detection/derived/pvst_rgbd_slice_relations_v1/checks.json；独立拟合诊断：/media/njupt/8FCD3869E2FAB06F/chenyh/data/sparse_3d_forgery_detection/derived/pvst_rgbd_slice_relations_v1/fit/result.json（检查权重丢弃）。

|条件|best/结束epoch|Dice|IoU|pixel AP|GT召回|real像素FPR|frame AUROC/AP|video AUROC/AP|

|---|---|---|---|---|---|---|---|---|

|RGBD_LOCAL|5/13|0.1582535901668644|0.10959719247374318|0.23274230259678305|0.13096429560401388|0.015151872196903934|0.5143527107812822/0.5131029368988097|0.5/0.5595238095238095|

该条件全部同best指标及成本：{"status": "COMPLETE", "condition": "RGBD_LOCAL", "best_epoch": 5, "last_epoch": 13, "updates": 624, "best_dev": {"dice": 0.1582535901668644, "iou": 0.10959719247374318, "pixel_ap": 0.23274230259678305, "real_fpr": 0.015151872196903934, "GT_recall": 0.13096429560401388, "pred_area": 0.023241507087780777, "frame": {"auroc": 0.5143527107812822, "ap": 0.5131029368988097, "acc": 0.43506493506493504, "precision": 0.45677233429394815, "recall": 0.6861471861471862, "f1": 0.5484429065743944, "confusion_matrix": [[85, 377], [145, 317]], "undefined_reason": null}, "video": {"auroc": 0.5, "ap": 0.5595238095238095, "acc": 0.5, "precision": 0.5, "recall": 0.75, "f1": 0.6, "confusion_matrix": [[1, 3], [1, 3]], "undefined_reason": null}, "loss": 1.835012935159935}, "best_train_monitor_clips": {"dice": 0.6436738624544424, "iou": 0.4749842691256574, "pixel_ap": null, "real_fpr": 0.007859700520833334, "GT_recall": 0.7133529469156135, "pred_area": 0.07175184461805555, "frame": {"auroc": 1.0, "ap": 1.0, "acc": 0.5, "precision": 0.0, "recall": 0.0, "f1": 0.0, "confusion_matrix": [[24, 0], [24, 0]], "undefined_reason": null}, "video": {"auroc": 1.0, "ap": 1.0, "acc": 0.5, "precision": 0.0, "recall": 0.0, "f1": 0.0, "confusion_matrix": [[1, 0], [1, 0]], "undefined_reason": null}, "loss": 1.6145828068256378}, "parameters": 20402578, "trainable": 20402578, "seconds": 2425.9403315884992, "peak_allocated": 2761790464, "peak_reserved": 2946498560, "peak_rss": 14470848512}

相对epoch0的开发Dice变化：0.07521293833844413；训练监测Dice：0.6436738624544424。

|RGBD_SLICE_RELATION|12/20|0.21836327576841308|0.17624700211499955|0.2951098366623513|0.26077065258600723|0.008210221387656464|0.47167875414628657/0.4519823489901186|0.5/0.5595238095238095|

该条件全部同best指标及成本：{"status": "COMPLETE", "condition": "RGBD_SLICE_RELATION", "best_epoch": 12, "last_epoch": 20, "updates": 960, "best_dev": {"dice": 0.21836327576841308, "iou": 0.17624700211499955, "pixel_ap": 0.2951098366623513, "real_fpr": 0.008210221387656464, "GT_recall": 0.26077065258600723, "pred_area": 0.015637847272456918, "frame": {"auroc": 0.47167875414628657, "ap": 0.4519823489901186, "acc": 0.4805194805194805, "precision": 0.48717948717948717, "recall": 0.7402597402597403, "f1": 0.5876288659793815, "confusion_matrix": [[102, 360], [120, 342]], "undefined_reason": null}, "video": {"auroc": 0.5, "ap": 0.5595238095238095, "acc": 0.5, "precision": 0.5, "recall": 0.75, "f1": 0.6, "confusion_matrix": [[1, 3], [1, 3]], "undefined_reason": null}, "loss": 2.3804389246967106}, "best_train_monitor_clips": {"dice": 0.12626600521983058, "iou": 0.07032849924674926, "pixel_ap": null, "real_fpr": 1.808449074074074e-07, "GT_recall": 0.07090510695866943, "pred_area": 0.004649703414351851, "frame": {"auroc": 1.0, "ap": 0.9999999999999998, "acc": 0.5208333333333334, "precision": 1.0, "recall": 0.041666666666666664, "f1": 0.08, "confusion_matrix": [[24, 0], [23, 1]], "undefined_reason": null}, "video": {"auroc": 1.0, "ap": 1.0, "acc": 0.5, "precision": 0.0, "recall": 0.0, "f1": 0.0, "confusion_matrix": [[1, 0], [1, 0]], "undefined_reason": null}, "loss": 1.9225237319866815}, "parameters": 33264311, "trainable": 33264311, "seconds": 12265.46726849489, "peak_allocated": 3182884864, "peak_reserved": 3640655872, "peak_rss": 31536898048}

相对epoch0的开发Dice变化：0.13532262393999278；训练监测Dice：0.12626600521983058。

B−A：{"dice": 0.06010968560154867, "iou": 0.06664980964125637, "pixel_ap": 0.06236753406556825, "real_fpr": -0.00694165080924747, "GT_recall": 0.12980635698199336}

逐fake视频：[{"video_id": "e73bc60d4e2c5b994ee45ba4", "pair_id": "an:7e2e57", "generator": "splicing", "A_Dice": 0.0, "B_Dice": 0.06759553813129342, "difference": 0.06759553813129342}, {"video_id": "a2a08b465a8cea8a95403ee7", "pair_id": "an:4rvro9", "generator": "splicing", "A_Dice": 0.5083187910062422, "B_Dice": 0.7968243859360006, "difference": 0.28850559492975836}, {"video_id": "f2cd41132415857399b19518", "pair_id": "an:4wgx37", "generator": "inpainting", "A_Dice": 0.1245313904327252, "B_Dice": 0.008896250144694726, "difference": -0.11563514028803047}, {"video_id": "37b22da558e1daee1e72b707", "pair_id": "an:95q8g9", "generator": "inpainting", "A_Dice": 0.00016417922849022284, "B_Dice": 0.00013692886166365674, "difference": -2.7250366826566096e-05}]

定位错误证据：分别检查编辑区召回（漏覆盖）、real像素FPR（背景误报）及固定样例。低误报不能单独证明定位成功；本报告不凭数值虚构肉眼边界结论。共同样例身份/PTS/概率色阶固定，不按预测挑选。

限制：相机运动与逐帧几何可比性未解决；单seed小队列不能证明三维增益，或把切片关系收益外推到其他数据。开发fake全程编辑，无法检验复杂时间切换。

完整产物：`/media/njupt/8FCD3869E2FAB06F/chenyh/data/sparse_3d_forgery_detection/derived/pvst_rgbd_slice_relations_v1`；计算与Git发布状态分开。
