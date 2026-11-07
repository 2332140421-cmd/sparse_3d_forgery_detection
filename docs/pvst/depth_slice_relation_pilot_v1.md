# 深度切片关系学习首轮实验



状态：COMPLETE。独立方法DEPTH_SLICE_RELATION，单seed20260909；不是旧PVST体素实验。

数据：{"train": {"pairs": 8, "videos": 16, "frames": 2138, "positive_frames": 1065, "all_nonempty_fake": 7, "mixed_fake": 1}, "dev": {"pairs": 4, "videos": 8, "frames": 924, "positive_frames": 462, "all_nonempty_fake": 4, "mixed_fake": 0}}

工作流：冻结官方ImageNet ResNet18原生stride32局部视觉→回映每个原生像素；XYZ/UV逐点128→256编码→融合128→同深度片mean/max→512维C→两组时间/深度显式8head成对关系→像素局部旁路+片上下文定位/独立帧头。无效几何保留视觉路径。d=Z/1.9888148307800293，h=1/8，d0=0；单位非米，无新增配准。

训练片段：每视频预先固定首、中、尾最多3段24连续原生帧。开发完整原生帧，每帧一次；24帧因果窗口，最后尾段补左历史，常规边界重置历史。frame BCE+pixel BCE+softDice等权，逐帧/片段均值；空GT Dice使用epsilon1仍参与。

全部指标固定logit≥0；视频log-mean-exp温度1；空间Dice/IoU：正GT帧均值后fake视频宏平均；pixel AP：正GT帧AP均值后fake视频宏平均，并非AP@IoU。

监测：固定前两个训练pair的完整视频，称训练监测子集，不代表全训练集。best仅按开发Dice，min10/max30/patience8。

完成摘要：{"status": "COMPLETE", "best_epoch": 10, "last_epoch": 18, "updates": 864, "parameters": 13789378, "D": 512, "best_dev": {"dice": 0.07263686413452544, "iou": 0.04540065552366088, "pixel_ap": 0.19427542041815773, "real_fpr": 0.0, "GT_recall": 0.04822143645625238, "pred_area": 0.001884620949074074, "frame": {"auroc": 0.47001789696594887, "ap": 0.48807045122123643, "acc": 0.4945887445887446, "precision": 0.4966887417218543, "recall": 0.8116883116883117, "f1": 0.6162695152013147, "confusion_matrix": [[82, 380], [87, 375]], "undefined_reason": null}, "video": {"auroc": 0.4375, "ap": 0.5416666666666666, "acc": 0.5, "precision": 0.5, "recall": 0.75, "f1": 0.6, "confusion_matrix": [[1, 3], [1, 3]], "undefined_reason": null}, "loss": 1.9074383495416907}, "best_monitor": {"dice": 0.0, "iou": 0.0, "pixel_ap": 0.21752527667122068, "real_fpr": 0.0, "GT_recall": 0.0, "pred_area": 0.0, "frame": {"auroc": 0.6711802469135802, "ap": 0.5811942440542458, "acc": 0.5, "precision": 0.5, "recall": 0.3333333333333333, "f1": 0.4, "confusion_matrix": [[150, 75], [150, 75]], "undefined_reason": null}, "video": {"auroc": 0.75, "ap": 0.8333333333333333, "acc": 0.5, "precision": 0.5, "recall": 0.5, "f1": 0.5, "confusion_matrix": [[1, 1], [1, 1]], "undefined_reason": null}, "loss": 1.920107186436653}, "wall_s": 9945.8154096473, "peak_allocated": 5167507968, "peak_reserved": 6169821184, "execution_head": "35914bb1f469d896f77b91954780a5116179bc49", "peak_rss_bytes": 19283095552, "C_max_occupied_per_frame": 552, "C_max_window_units_upper_bound": 13248, "data": {"train": {"pairs": 8, "videos": 16, "frames": 2138, "positive_frames": 1065, "all_nonempty_fake": 7, "mixed_fake": 1}, "dev": {"pairs": 4, "videos": 8, "frames": 924, "positive_frames": 462, "all_nonempty_fake": 4, "mixed_fake": 0}}, "storage_bytes": 2067202299}



## 结论边界

相机运动本轮允许，不证明深度可比性已解决；此前CPU审计显示漂移/跨片问题。没有匹配二维对照，不主张三维优于二维。全程编辑样本不能检验复杂时间边界。检测监督只来自逐帧GT是否含编辑区，ANAKIN裁剪边界未作为伪造区间。



best开发：{"dice": 0.07263686413452544, "iou": 0.04540065552366088, "pixel_ap": 0.19427542041815773, "real_fpr": 0.0, "GT_recall": 0.04822143645625238, "pred_area": 0.001884620949074074, "frame": {"auroc": 0.47001789696594887, "ap": 0.48807045122123643, "acc": 0.4945887445887446, "precision": 0.4966887417218543, "recall": 0.8116883116883117, "f1": 0.6162695152013147, "confusion_matrix": [[82, 380], [87, 375]], "undefined_reason": null}, "video": {"auroc": 0.4375, "ap": 0.5416666666666666, "acc": 0.5, "precision": 0.5, "recall": 0.75, "f1": 0.6, "confusion_matrix": [[1, 3], [1, 3]], "undefined_reason": null}, "loss": 1.9074383495416907}

best训练监测子集：{"dice": 0.0, "iou": 0.0, "pixel_ap": 0.21752527667122068, "real_fpr": 0.0, "GT_recall": 0.0, "pred_area": 0.0, "frame": {"auroc": 0.6711802469135802, "ap": 0.5811942440542458, "acc": 0.5, "precision": 0.5, "recall": 0.3333333333333333, "f1": 0.4, "confusion_matrix": [[150, 75], [150, 75]], "undefined_reason": null}, "video": {"auroc": 0.75, "ap": 0.8333333333333333, "acc": 0.5, "precision": 0.5, "recall": 0.5, "f1": 0.5, "confusion_matrix": [[1, 1], [1, 1]], "undefined_reason": null}, "loss": 1.920107186436653}

epoch0开发：{"dice": 0.009630645389795005, "iou": 0.005101931697062507, "pixel_ap": 0.058104379338320855, "real_fpr": 0.06969183877073688, "GT_recall": 0.010613632143187293, "pred_area": 0.0705045654296875, "frame": {"auroc": 0.34483283671595355, "ap": 0.4015775094761759, "acc": 0.4945887445887446, "precision": 0.4972796517954298, "recall": 0.9891774891774892, "f1": 0.6618392469225199, "confusion_matrix": [[0, 462], [5, 457]], "undefined_reason": null}, "video": {"auroc": 0.3125, "ap": 0.4583333333333333, "acc": 0.5, "precision": 0.5, "recall": 1.0, "f1": 0.6666666666666666, "confusion_matrix": [[0, 4], [0, 4]], "undefined_reason": null}, "loss": 2.2399639257457524}

学习判断：best监测Dice=0.0，开发Dice=0.07263686413452544；相对epoch0开发Dice变化=0.06300621874473043。这些只证明此队列的拟合与开发差异，不证明泛化。

定位错误证据：best开发编辑区召回=0.04822143645625238，fake预测面积比例=0.001884620949074074，real误报面积=0.0。1−召回衡量漏覆盖，real面积衡量背景误报，二者分别解释；固定样例需人工查看，不将数字当成精确边界证据。

训练监测/开发差距及固定样例见best_monitor.json、best_dev.json和samples/index.html。降低real误报不自动意味着GT覆盖提升。固定样例只展示预先选择身份及PTS，不按预测挑帧。

完整资产与固定样例：`/media/njupt/8FCD3869E2FAB06F/chenyh/data/sparse_3d_forgery_detection/derived/pvst_depth_slice_relation_pilot_v1`。
