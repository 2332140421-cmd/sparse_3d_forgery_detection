# ANAKIN帧BCE关闭消融

状态：COMPLETE

唯一改动：帧BCE权重1→0；pixel BCE/soft Dice各1，未重归一化。完整RGB_DEPTH_PARTITION及帧头/像素消息保留，原参数分组/初始化/seed/BN/Dropout/BF16/clip1不变。

原8对训练16视频48段，每轮1152帧次，固定12轮576步；直接复制基线manifest及orders，原4对完整开发924帧不变。未使用扩量清单、权重或新缓存。

执行提交：dc171214051db9b9d3b67d67f88bf59641182b6b

预检：{"status": "PASS", "clip": {"video_id": "cb41b3d004b6640f6c3b5bb1", "start": 0, "end": 24}, "frames": 24, "initial_model_sha256": "b08ed76adb433e8aaab631001240a10faff572a9e8ef166e82da4cdcffbb07ae", "terms": [0.7076897621154785, 0.7176088094711304, 0.8933441638946533], "spatial_gradients": {"local.encoder": {"present": true, "finite": true, "norm": 1.5972150564193726}, "local.refine": {"present": true, "finite": true, "norm": 3.687112331390381}, "local.final": {"present": true, "finite": true, "norm": 2.2944462299346924}, "pool": {"present": true, "finite": true, "norm": 0.0}, "sets.current_readout": {"present": true, "finite": true, "norm": 0.0}, "pixel_context": {"present": true, "finite": true, "norm": 2.9936540126800537}, "sets.frame_head": {"present": true, "finite": true, "norm": 0.0}}, "optimizer_updates": 0, "zero_upstream_first_step_allowed": true, "peak_allocated": 2016861184, "peak_reserved": 11788091392}



## 第576步开发主比较

|条件|Dice|IoU|pixel AP|GT召回|real FPR|
|---|---|---|---|---|---|
|原帧BCE_ON基线|0.191751|0.139499|0.302668|0.147274|0.003645|
|FRAME_BCE_OFF|0.230979|0.165241|0.302943|0.192837|0.016198|
|OFF−ON|0.039228|0.025742|0.000275|0.045562|0.012553|



## 逐开发pair（全部4对）

|pair|on_Dice|off_Dice|delta_Dice|on_IoU|off_IoU|delta_IoU|on_recall|off_recall|on_real_FPR|off_real_FPR|
|---|---|---|---|---|---|---|---|---|---|---|
|an:7e2e57|0.000199|0.000000|-0.000199|0.000101|0.000000|-0.000101|0.000137|0.000000|0.008569|0.016511|
|an:4rvro9|0.656190|0.654951|-0.001239|0.494794|0.497423|0.002630|0.518603|0.534577|0.000000|0.000000|
|an:4wgx37|0.110615|0.268964|0.158349|0.063099|0.163540|0.100441|0.070358|0.236770|0.006012|0.048281|
|an:95q8g9|0.000000|0.000000|0.000000|0.000000|0.000000|0.000000|0.000000|0.000000|0.000000|0.000000|

开发Dice改善1/4，持平1，下降2；其他3对（不含4rvro9）平均ΔDice +0.052717，仅作敏感性说明，不改变主表。



## 原48段同口径训练回测

|条件|Dice|IoU|GT召回|fake预测面积|real FPR|fake非编辑区FPR|
|---|---|---|---|---|---|---|
|原基线|0.514184|0.440020|0.571029|0.039256|0.000007|0.005331|
|FRAME_BCE_OFF|0.700751|0.584892|0.910279|0.045361|0.000856|0.009516|
|OFF−ON|0.186567|0.144872|0.339250|0.006105|0.000850|0.004185|

训练指标按视频聚合再fake宏平均；训练只覆盖反向使用的原48段，开发覆盖完整视频。fake非编辑区FPR分母是GT负像素数，每帧→视频→fake宏平均，不能以real FPR代替。



|pair|on_Dice|off_Dice|delta_Dice|off_IoU|off_recall|off_pred_area|off_fake_nonedit_FPR|off_real_FPR|
|---|---|---|---|---|---|---|---|---|
|an:7ogg04|0.797437|0.811674|0.014237|0.684128|0.954841|0.085824|0.026785|0.000000|
|an:728qx7|0.000000|0.507869|0.507869|0.340393|0.909125|0.004556|0.002955|0.000000|
|an:95qvg9|0.907173|0.928771|0.021599|0.867185|0.974537|0.096196|0.011939|0.000000|
|an:4wdo37|0.307473|0.397502|0.090029|0.249870|0.966092|0.029630|0.022206|0.006850|
|an:40y1x9|0.510766|0.584124|0.073358|0.479476|0.520171|0.036766|0.005275|0.000000|
|an:7p8r14|0.984601|0.985230|0.000629|0.970937|0.982725|0.099226|0.001373|0.000000|
|an:7e2557|0.000000|0.776165|0.776165|0.642884|0.984723|0.001800|0.000639|0.000000|
|an:4z1827|0.606025|0.614676|0.008650|0.444260|0.990019|0.008890|0.004955|0.000000|



## 帧指标仅诊断：OFF未接受帧BCE监督

|条件|开发帧AUROC|开发帧AP|开发正GT帧BCE|
|---|---|---|---|
|基线|0.489135|0.494619|2.637695|
|OFF（未帧监督）|0.429426|0.436113|0.756980|

帧头仍计算并留在原AdamW参数组；0权重产生的零梯度是预期，不作为路径缺失，不把该头指标当正式检测能力或消融胜负标准。

## best补充，不替代第12轮

{"baseline_best_epoch": 6, "baseline_best_dev": {"dice": 0.2969638299941371, "iou": 0.2175790938759564, "pixel_ap": 0.35902189924110345, "real_fpr": 0.01948410674672068, "GT_recall": 0.35222686735021586, "pred_area": 0.049302243397821285, "frame": {"auroc": 0.538544067764847, "ap": 0.5930595207064995, "acc": 0.5216450216450217, "precision": 1.0, "recall": 0.04329004329004329, "f1": 0.08298755186721991, "confusion_matrix": [[462, 0], [442, 20]], "undefined_reason": null}, "video": {"auroc": 0.5625, "ap": 0.6428571428571428, "acc": 0.625, "precision": 1.0, "recall": 0.25, "f1": 0.4, "confusion_matrix": [[4, 0], [3, 1]], "undefined_reason": null}, "loss": 2.1191922468526494, "frame_bce_fp32": 1.0380723819910707}, "OFF_best_epoch": 8, "OFF_best_dev": {"dice": 0.2609542048837805, "iou": 0.21170940557496593, "pixel_ap": 0.3342281230472538, "real_fpr": 0.002785866066261574, "GT_recall": 0.24956995803275084, "pred_area": 0.018265131645045656, "frame": {"auroc": 0.4374543205712037, "ap": 0.438284741342704, "acc": 0.37987012987012986, "precision": 0.06299212598425197, "recall": 0.017316017316017316, "f1": 0.027164685908319185, "confusion_matrix": [[343, 119], [454, 8]], "undefined_reason": null}, "video": {"auroc": 0.4375, "ap": 0.525, "acc": 0.375, "precision": 0.0, "recall": 0.0, "f1": 0.0, "confusion_matrix": [[3, 1], [4, 0]], "undefined_reason": null}, "loss": 1.760774376756615, "frame_bce_fp32": 0.6982226371765137, "positive_frame_bce": 0.7508245706558228, "negative_frame_bce": 0.6456207036972046}}



训练Dice变化+0.186567；开发Dice变化+0.039228，改善1/4，real误报变化+0.012553。当前配置下帧监督可能拖累定位，值得研究任务协调；仍须看多视频方向及误报，不确定具体层的梯度冲突。



![同更新数曲线](/media/njupt/8FCD3869E2FAB06F/chenyh/data/sparse_3d_forgery_detection/derived/pvst_rgb_partition_frame_loss_off_v1/curves.png)

评价记录中的metrics.loss沿用原三项诊断和，不是OFF优化目标，不据此比较总loss。曲线分开比较两项原始空间loss，保留未加权frame BCE；新优化目标是pixel BCE+soft Dice，没有与旧三项总loss直接比大小。逐次更新日志记录原始terms和实际loss，逐轮history记录optimization_loss/spatial_loss/unweighted_frame_BCE。

资源：{"updates": 576, "last_epoch": 12, "wall_s": 6239.076755112037, "peak_allocated": 3288058880, "peak_reserved": 22951231488, "stop_reason": "12_EPOCHS_576_UPDATES"}

产物：/media/njupt/8FCD3869E2FAB06F/chenyh/data/sparse_3d_forgery_detection/derived/pvst_rgb_partition_frame_loss_off_v1

本轮只运行一个新条件，不重训基线、不追加其他权重/模型/seed/数据。计算与发布状态独立。
