# ANAKIN同数据集、来源隔离RGB-D分片匹配对照
状态：COMPLETE。主比较固定第12轮；best单列；不是保留测试集结论。
执行提交：5ffe208bc9bcac16ccb0429ebc8659b6bea36267
数据：{"train_videos": 16, "train_pairs": 8, "dev_videos": 8, "dev_pairs": 4, "train_full_frames": 2138, "dev_frames": 924, "clips": 48, "frame_presentations_per_epoch": 1152, "unique_training_clip_frames": 1152, "unique_frames_per_video": {"cb41b3d004b6640f6c3b5bb1": 72, "c26e3fc1fdc6aabda308ae84": 72, "f02cfe65b813132c7e61618f": 72, "8a9691dcdc0248c8ab03723f": 72, "d74a0e91fefb7a10107172c0": 72, "adcb32286a41e0e3a6fdfdeb": 72, "87664e1174a82a8f26ee463f": 72, "782426ea0330c7fea684953c": 72, "ac8efb162961478dd9f9af22": 72, "5412ec415c6dc6000437ff00": 72, "f0ba19d9d7098e420eff20ab": 72, "7f6c4de78fd5220b6dbbea63": 72, "473cef2b50cd96215bcdcbed": 72, "7de955b21023a518f9dd2ce0": 72, "03940cfef98b2a38304cdd0e": 72, "9d465eb0c15946cd37ad1e72": 72}, "source_id_url_media_sha_isolation": true}
共同初始化、原始48段、seed+epoch同序；完整原生像素/有效点、固定BN、BF16、三项等权loss。两组都使用相同深度分片，不能据此证明深度分片本身的增益。

|第12轮|Dice|IoU|pixel AP|GT召回|real FPR|FP32帧BCE|frame AUROC/AP|video AUROC/AP|
|---|---:|---:|---:|---:|---:|---:|---|---|
|RGBD|0.032718|0.018715|0.116857|0.023416|0.001976|1.653667|0.562876/0.603303|0.500000/0.642857|
|RGB_DEPTH_PARTITION|0.191751|0.139499|0.302668|0.147274|0.003645|1.355967|0.489135/0.494619|0.500000/0.625000|
|新版−旧版|+0.159033|+0.120784|+0.185811|+0.123858|+0.001669|-0.297700|-0.073741/-0.108684|+0.000000/-0.017857|

|开发pair|旧Dice|新Dice|ΔDice|旧IoU|新IoU|ΔIoU|旧real FPR|新real FPR|
|---|---:|---:|---:|---:|---:|---:|---:|---:|
|an:7e2e57|0.000000|0.000199|0.000199|0.000000|0.000101|0.000101|0.000000|0.008569|
|an:4rvro9|0.000000|0.656190|0.656190|0.000000|0.494794|0.494794|0.000000|0.000000|
|an:4wgx37|0.130874|0.110615|-0.020259|0.074859|0.063099|-0.011759|0.007905|0.006012|
|an:95q8g9|0.000000|0.000000|0.000000|0.000000|0.000000|0.000000|0.000000|0.000000|

Dice改善2/4，持平1，下降1。
最大单对绝对变化/总变化绝对值：1.0315334768835125；该比率可能因正负抵消超过1。
第12轮宏平均Dice变化+0.159033，real误报变化+0.001669。只判断本4对开发视频；检测与空间指标分开判断。

## 各自best（不与第12轮主表拼接）
RGBD：{"status": "COMPLETE", "updates": 576, "last_epoch": 12, "best_epoch": 10, "best_dev": {"dice": 0.2493140646215881, "iou": 0.1668895190313984, "pixel_ap": 0.2543612593520033, "real_fpr": 0.06664055991257824, "GT_recall": 0.2970810635838858, "pred_area": 0.08667147128048912, "frame": {"auroc": 0.5112699349712336, "ap": 0.4968047200997433, "acc": 0.5476190476190477, "precision": 0.5294906166219839, "recall": 0.854978354978355, "f1": 0.6539735099337748, "confusion_matrix": [[111, 351], [67, 395]], "undefined_reason": null}, "video": {"auroc": 0.5625, "ap": 0.6678571428571428, "acc": 0.5, "precision": 0.5, "recall": 1.0, "f1": 0.6666666666666666, "confusion_matrix": [[0, 4], [0, 4]], "undefined_reason": null}, "loss": 3.597323600653973, "frame_bce_fp32": 2.3526760943584724}, "wall_s": 5940.987753789872, "peak_allocated": 3289728512, "peak_reserved": 22265462784, "stop_reason": "12_EPOCHS_576_UPDATES"}
RGB_DEPTH_PARTITION：{"status": "COMPLETE", "updates": 576, "last_epoch": 12, "best_epoch": 6, "best_dev": {"dice": 0.2969638299941371, "iou": 0.2175790938759564, "pixel_ap": 0.35902189924110345, "real_fpr": 0.01948410674672068, "GT_recall": 0.35222686735021586, "pred_area": 0.049302243397821285, "frame": {"auroc": 0.538544067764847, "ap": 0.5930595207064995, "acc": 0.5216450216450217, "precision": 1.0, "recall": 0.04329004329004329, "f1": 0.08298755186721991, "confusion_matrix": [[462, 0], [442, 20]], "undefined_reason": null}, "video": {"auroc": 0.5625, "ap": 0.6428571428571428, "acc": 0.625, "precision": 1.0, "recall": 0.25, "f1": 0.4, "confusion_matrix": [[4, 0], [3, 1]], "undefined_reason": null}, "loss": 2.1191922468526494, "frame_bce_fp32": 1.0380723819910707}, "wall_s": 5871.738845240325, "peak_allocated": 3288146432, "peak_reserved": 22947037184, "stop_reason": "12_EPOCHS_576_UPDATES"}

![同轴开发Dice](/media/njupt/8FCD3869E2FAB06F/chenyh/data/sparse_3d_forgery_detection/derived/pvst_rgb_partition_anakin_dev_v1/dev_dice.png)
完整产物：/media/njupt/8FCD3869E2FAB06F/chenyh/data/sparse_3d_forgery_detection/derived/pvst_rgb_partition_anakin_dev_v1
只读取原train/dev；不重提前端、不加载历史拟合权重、不访问保留测试。
