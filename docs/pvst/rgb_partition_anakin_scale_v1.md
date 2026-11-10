# ANAKIN固定576步训练来源扩量对照
状态：COMPLETE
原8对/48段训练12轮，扩量16对/96段训练6轮；均576次AdamW、13824帧次。原开发4对8视频924帧不变。两组匹配更新与输入帧次，实际墙钟不同。
扩量固定新增4 inpainting/4 splicing，均来自历史登记train，排除原train/dev及保留test来源；未用面积/模型分数/几何异常挑选。新视频含1080p样本，未按资源负载替换。
准备预算及实测：{"status": "PASS", "reused_original_videos": 24, "reused_original_frames": 3062, "new_videos": 16, "new_frames": 1903, "new_RGB_videos": 16, "new_geometry_videos": 16, "new_cache_bytes_estimate": 13210804224, "new_cache_GiB_estimate": 12.303520202636719, "preparation_seconds_estimate": 274.1557882699193, "estimate_basis": {"seconds": 82.68085893895477, "new_frontend_frames": 3062}, "estimate_rule": "historical MoGe seconds/frame x new frames x3 plus120s; not guaranteed; parent hard cap3600s", "per_video": [{"video_id": "1f2917e9f5fddfc574f0c503", "frames": 150, "native_hw": [360, 640], "geometry_hw": [360, 640], "rgb_bytes": 103680000, "geometry_bytes": 449280000, "reused": false}, {"video_id": "0a93d3d27cddd679868fd417", "frames": 150, "native_hw": [360, 640], "geometry_hw": [360, 640], "rgb_bytes": 103680000, "geometry_bytes": 449280000, "reused": false}, {"video_id": "43bf3b5843f9f0fefd6445c7", "frames": 150, "native_hw": [720, 1280], "geometry_hw": [432, 768], "rgb_bytes": 414720000, "geometry_bytes": 646963200, "reused": false}, {"video_id": "dc78bade65a643b123bdece6", "frames": 150, "native_hw": [720, 1280], "geometry_hw": [432, 768], "rgb_bytes": 414720000, "geometry_bytes": 646963200, "reused": false}, {"video_id": "5f8e607584c82ad726e61a7b", "frames": 96, "native_hw": [1080, 1920], "geometry_hw": [432, 768], "rgb_bytes": 597196800, "geometry_bytes": 414056448, "reused": false}, {"video_id": "001bc9db0f06b3c847090d4c", "frames": 96, "native_hw": [1080, 1920], "geometry_hw": [432, 768], "rgb_bytes": 597196800, "geometry_bytes": 414056448, "reused": false}, {"video_id": "5e281219c8a281ae6109f506", "frames": 90, "native_hw": [368, 640], "geometry_hw": [368, 640], "rgb_bytes": 63590400, "geometry_bytes": 275558400, "reused": false}, {"video_id": "ceb1a2c7d9ba416daf788ea4", "frames": 90, "native_hw": [368, 640], "geometry_hw": [368, 640], "rgb_bytes": 63590400, "geometry_bytes": 275558400, "reused": false}, {"video_id": "1ba064388272863e75d92e59", "frames": 180, "native_hw": [720, 1280], "geometry_hw": [432, 768], "rgb_bytes": 497664000, "geometry_bytes": 776355840, "reused": false}, {"video_id": "d9d49a14fc24303c96df1646", "frames": 179, "native_hw": [720, 1280], "geometry_hw": [432, 768], "rgb_bytes": 494899200, "geometry_bytes": 772042752, "reused": false}, {"video_id": "352bc88a442731f9c74d90c9", "frames": 96, "native_hw": [720, 1280], "geometry_hw": [432, 768], "rgb_bytes": 265420800, "geometry_bytes": 414056448, "reused": false}, {"video_id": "f75c5c0dafe2d1bc889b4dd4", "frames": 96, "native_hw": [720, 1280], "geometry_hw": [432, 768], "rgb_bytes": 265420800, "geometry_bytes": 414056448, "reused": false}, {"video_id": "2781886ae295fb7592043d62", "frames": 100, "native_hw": [1080, 1920], "geometry_hw": [432, 768], "rgb_bytes": 622080000, "geometry_bytes": 431308800, "reused": false}, {"video_id": "e64dea5e0e1c0f9c9b6b1756", "frames": 100, "native_hw": [1080, 1920], "geometry_hw": [432, 768], "rgb_bytes": 622080000, "geometry_bytes": 431308800, "reused": false}, {"video_id": "e6946eddd169928939058b24", "frames": 90, "native_hw": [720, 1280], "geometry_hw": [432, 768], "rgb_bytes": 248832000, "geometry_bytes": 388177920, "reused": false}, {"video_id": "afdc517ce94183050f471883", "frames": 90, "native_hw": [720, 1280], "geometry_hw": [432, 768], "rgb_bytes": 248832000, "geometry_bytes": 388177920, "reused": false}], "limits": {"seconds": 3600, "bytes": 32212254720}}
{"status": "PASS", "wall_s": 80.19891512487084, "new_cache_bytes": 13212223669, "new_frontend_frames": 1903}
原缓存通过只读符号链接引用；只补新增16条native RGB/高分辨率MoGe。frame_identity只是帧身份适配元数据，不是旧256几何。

|576步开发|Dice|IoU|pixel AP|GT召回|real FPR|
|---|---:|---:|---:|---:|---:|
|原8对|0.191751|0.139499|0.302668|0.147274|0.003645|
|扩量16对|0.057268|0.033941|0.181692|0.034663|0.000241|
|扩量−基线|-0.134483|-0.105557|-0.120976|-0.112611|-0.003404|

|开发pair|旧Dice|新Dice|ΔDice|旧IoU|新IoU|ΔIoU|旧召回|新召回|旧real FPR|新real FPR|
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|an:7e2e57|0.000199|0.000000|-0.000199|0.000101|0.000000|-0.000101|0.000137|0.000000|0.008569|0.000000|
|an:4rvro9|0.656190|0.000000|-0.656190|0.494794|0.000000|-0.494794|0.518603|0.000000|0.000000|0.000000|
|an:4wgx37|0.110615|0.229073|0.118458|0.063099|0.135766|0.072666|0.070358|0.138653|0.006012|0.000965|
|an:95q8g9|0.000000|0.000000|0.000000|0.000000|0.000000|0.000000|0.000000|0.000000|0.000000|0.000000|

Dice改善1/4，下降2/4；剔除绝对变化最大一对的其他3对平均差+0.039420（仅敏感性描述，主表保留4对）。
## 原48个共同训练片段（同集合比较）
原基线训练Dice 0.514184、开发0.191751、差距0.322433；扩量末轮共同片段Dice 0.449778、开发0.057268、差距0.392510。
若差距缩小仅来自训练分数下降，不视为开发改善。实际是否受益看开发Dice/IoU/AP及逐pair，不凭差距一项判断。

## 检测单列
原开发帧AUROC/AP 0.489135/0.494619，正GT帧FP32 BCE 2.637695；扩量0.500000/0.500000，正GT帧BCE 2.421140。
## best补充（不替代576步主表）
{"best_epoch": 4, "best_dev": {"dice": 0.12102743421604176, "iou": 0.07480171972892155, "pixel_ap": 0.2731370308158486, "real_fpr": 0.0016060245131923013, "GT_recall": 0.08077625685198193, "pred_area": 0.009164429665396734, "frame": {"auroc": 0.5, "ap": 0.5, "acc": 0.5, "precision": 0.0, "recall": 0.0, "f1": 0.0, "confusion_matrix": [[462, 0], [462, 0]], "undefined_reason": null}, "video": {"auroc": 0.5, "ap": 0.5, "acc": 0.5, "precision": 0.0, "recall": 0.0, "f1": 0.0, "confusion_matrix": [[4, 0], [4, 0]], "undefined_reason": null}, "loss": 2.2700421911726396, "frame_bce_fp32": 1.1753222942352295, "positive_frame_bce": 2.2378220558166504, "negative_frame_bce": 0.1128222867846489}}

固定576步下开发定位未提高；Dice改善1/4。只反映更多来源替代片段重复的固定预算策略；不证明数据已充分，也不归因于单独样本量效应。

![同更新数开发Dice](/media/njupt/8FCD3869E2FAB06F/chenyh/data/sparse_3d_forgery_detection/derived/pvst_rgb_partition_anakin_scale_v1/dev_dice_by_updates.png)
资源：{"wall_s": 10206.845383245498, "peak_allocated": 8547988480, "peak_reserved": 23116906496, "stop_reason": "6_EPOCHS_576_UPDATES", "updates": 576, "last_epoch": 6}
执行代码：e60b8ce5f57866fe59a151a2709c95253b86a204
产物：/media/njupt/8FCD3869E2FAB06F/chenyh/data/sparse_3d_forgery_detection/derived/pvst_rgb_partition_anakin_scale_v1
计算状态与发布状态独立；不追加训练、模型、seed或规模。
