# 冻结模型：像素上下文投影后清零的开发检测对照

COMPLETE；只用原RGB_DEPTH_PARTITION epoch12/576的last.pt，原4对8视频924唯一计分帧（462正/462负）。eval/no_grad、原BF16，保留原生画面、PTS、24帧窗口及尾段左上下文；上下文不重复计分。

## 主表：像素top-1%检测

|条件|帧AUROC|帧AP|视频AUROC|视频AP|
|---|---|---|---|---|
|NORMAL|0.732279|0.766790|0.687500|0.792857|
|PIXEL_MESSAGE_ZERO|0.732054|0.766300|0.687500|0.792857|
|NORMAL−ZERO|0.000225|0.000490|0.000000|0.000000|

## 每对视频分数及方向

|pair|normal_real|normal_fake|normal_gap|zero_real|zero_fake|zero_gap|normal_minus_zero_gap|direction_flipped|
|---|---|---|---|---|---|---|---|---|
|an:7e2e57|1.415016|0.888985|-0.526031|1.511030|0.983757|-0.527274|0.001243|False|
|an:4rvro9|-2.221082|2.428423|4.649505|-2.226517|2.482814|4.709331|-0.059826|False|
|an:4wgx37|0.935636|2.072501|1.136865|0.991438|2.136129|1.144692|-0.007826|False|
|an:95q8g9|-2.416090|-2.356383|0.059707|-2.389126|-2.326610|0.062516|-0.002809|False|

## 核验

基线复现：{"max_abs_pixel_score_difference": 0.0006678104400634766, "metric_delta": {"frame": {"auroc": 2.3425348100181154e-06, "ap": -2.988391412506175e-05}, "video": {"auroc": 0.0, "ap": 0.0}}, "pair_direction_flips": 0, "saved_metrics": {"frame": {"auroc": 0.7322763816270309, "ap": 0.7668200599099986, "acc": 0.7142857142857143, "precision": 0.698, "recall": 0.7554112554112554, "f1": 0.7255717255717256, "confusion_matrix": [[311, 151], [113, 349]], "undefined_reason": null}, "video": {"auroc": 0.6875, "ap": 0.7928571428571428, "acc": 0.625, "precision": 0.6, "recall": 0.75, "f1": 0.6666666666666666, "confusion_matrix": [[2, 2], [1, 3]], "undefined_reason": null}}}

原帧头一致性：{"max_abs_logit_difference": 0.015625, "changed_frame_scores": 2, "normal_minus_zero_metrics": {"frame": {"auroc": 0.002162159629692073, "ap": 0.0010587990206313846}, "video": {"auroc": 0.0, "ap": 0.0}}, "head_pair_flips": 0, "scope_check": "Only pixel_context forward output replaced; frame_logits computed beforehand; weights/buffers unchanged and hook removed"}

干预是pixel_context(C+M)完整投影输出（含bias）→zeros_like，再沿原代码回填像素；不是投影前置零，也不是上次保留C的pixel_context(C)。本轮不拆分C/M/bias各自贡献。原集合关系与GRU完整执行，zero_messages/capture始终关闭。

top1%口径已按旧脚本核对：所有原生像素FP32 logit、k=max(1,ceil(.01×H×W))、topk均值；视频log-mean-exp。GT仅作标签，不选像素或按valid裁画面。没有调阈值/比例。开发fake全程mask非空，此为真假排序，不检验编辑起止定位。

本表尚需结合基线重算误差与逐pair方向解释：微小BF16路径重算差异本身不判失败，但若NORMAL偏差足以改变关键排序结论，则不作消息归因。仅此权重的推理依赖，不能分离深度/时序/集合构造贡献，也不能替代移除模块后的匹配重训。

资源：{"NORMAL": {"seconds": 79.66948225814849, "peak_allocated": 1747005440, "peak_reserved": 20721958912, "peak_RSS": 4817260544, "delivered_projection": {"calls": 960, "elements": 11416896, "delivered_nonzero_elements": 11416108, "first_shape": [224, 32], "dtype": "torch.bfloat16", "device": "cuda:0"}}, "PIXEL_MESSAGE_ZERO": {"seconds": 77.56866041943431, "peak_allocated": 1747005440, "peak_reserved": 20721958912, "peak_RSS": 5174824960, "delivered_projection": {"calls": 960, "elements": 11416896, "delivered_nonzero_elements": 0, "first_shape": [224, 32], "dtype": "torch.bfloat16", "device": "cuda:0"}}}

总墙钟161.873秒；真实退出码0；执行HEAD 84ffad98e33ae8f1c2fb532e5dad7554cb9af8db；完整产物/media/njupt/8FCD3869E2FAB06F/chenyh/data/sparse_3d_forgery_detection/derived/pvst_pixel_message_dev_check_v1。计算与发布状态独立。


## 完成后的判读与下一步

NORMAL−ZERO的帧AUROC仅+0.000224883、帧AP仅+0.000490176；视频AUROC/AP均0差。4对均未改变fake−real的方向，仍3对正向、7e2e57反向。关闭后4rvro9、4wgx37、95q8g9的正分差分别略增0.059826、0.007826、0.002809，7e2e57的负分差略恶化0.001243；不符合“多个pair同时因关闭而排序/分差变差”的支持模式。

NORMAL复现检查：与旧像素基线帧AUROC差+0.000002343、帧AP差−0.000029884，单帧像素分数最大差0.00066781，视频指标相同且逐对方向无翻转。没有发现足以改变本轮“总体差异很小”判断的基线口径偏差。

帧头核验并非逐值完全一致：2/924帧有数值变化，max|Δlogit|=0.015625，NORMAL−ZERO帧AUROC/AP差为+0.00216216/+0.00105880；两条件原帧头视频AUROC/AP相同，pair方向无翻转。源码中frame_logits在pixel_context调用前已计算完；hook只作用于该Linear输出，关键测试证明CPU小图帧头不变，实际权重/buffers hash不变、原zero_messages/capture关闭、hook已移除。未发现误改其他路径的证据，但本轮未独立确定数值差异原因，不自动归因BF16，不将其解释为像素消息影响了帧头。

ZERO对960次投影调用（含尾段36帧额外上下文）的11,416,896个输出元素清零，非零交付元素数为0；最终两条件均924个唯一计分帧（462/462）。完整关系、GRU及局部RGB路径均执行。

**结论与下一步判断：保留像素top-1%读出为候选基线，暂不把该检测成绩归给集合关系消息。** 不据这几个微小指标差宣称正常消息有可复现收益。本干预去掉的是pixel_context(C+M)完整仿射输出，包括C分量和bias；不能单独拆分深度、时序、集合构造或M自身的贡献，也不能证明该分支在训练时无作用。匹配重训不在本轮范围内，不自动启动。

实际正常前向各一次；计算退出码0，总墙钟约161.87秒。原实验、旧两份诊断、权重及用户JSON均未修改。本次报告发布状态另记publication.json，不影响计算COMPLETE。
