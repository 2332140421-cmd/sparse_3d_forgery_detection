# PVST等尺寸XYZ块：当前空间 vs 加入历史

状态：COMPLETE；末次576步主比较，未调参/选best。

|指标|XYZ_CURRENT|XYZ_TEMPORAL|B−A|

|---|---:|---:|---:|

|pixel_AP|0.06939956|0.07098718|+0.00158762|

|Dice|0.06643990|0.07151388|+0.00507398|

|IoU|0.03800968|0.04092757|+0.00291789|

|GT_recall|0.19356743|0.23068951|+0.03712207|

|real_FPR|0.06397724|0.07626510|+0.01228786|


|开发来源|A AP|B AP|B−A|A Dice|B Dice|配对real FPR A/B|

|---|---:|---:|---:|---:|---:|---|

|an:7e2e57|0.02543235|0.01806305|-0.00736930|0.00493550|0.00053419|0.00458898/0.00648353|

|an:95q8g9|0.02479678|0.02221503|-0.00258175|0.00000000|0.00000000|0.00013744/0.00002216|

|an:4xd3r9|0.01725572|0.02293805|+0.00568233|0.02088594|0.04322440|0.09257091/0.10963251|

|an:7ogeg4|0.21011338|0.22073259|+0.01061921|0.23993816|0.24229694|0.15861162/0.18892219|

改善2/4，下降2/4；四来源始终全部保留。

7e2e57猫添加是跨编辑类型；95q8g9骑马人物移除和4xd3r9多人过街移除为已明确运动内容；7ogeg4为站立对象移除参照，运动程度未获人工确认。逐来源不能以宏均值替代。

## 结构/数据/评价对应

通用原版DFormer-S冻结eval，仅官方通用预训练权重，不加载旧伪造训练last/common-C；新多尺度投影、几何条件调制、块内学习式softmax、空间及跨候选关系、像素/帧头训练。块为统一XYZ floor、不是深度整片或图像格；保留全部工作网格有效点/占据块，模型无GT选块、无共有键或追踪前提。

单一训练尺度与边长：{"scale": 12.699031352996826, "h": 0.5, "D": 128, "history_slots": 8, "history_step_seconds": 0.1, "work_long_edge": 384}。原点0，无米制/世界对齐保证；单目帧间尺度漂移仍是限制。工作长边384，等比resize、右/下pad32；输出bilinear恢复原生，全画面评分，不按几何valid裁评价像素。几何无效处仍有视觉局部旁路。

固定7/4来源，无测试访问。40y1x9 fake6–9不作为目标/标签，可进入无标签历史；real原零GT合法。开发所有原生帧各计一次，AP仅fake正GT帧：逐帧全画面AP→视频均值→4视频宏平均，real不混入AP，固定logit≥0的FPR单列。

训练采用同初始化/576目标顺序、AdamW1e-4 wd.01 clip1；pixelBCE+softDice+frameBCE各1。工作GT nearest；帧标签按原生GT非空，不因下采样消失改变帧标签。每步目标一帧，过去7个PTS槽是约0.7秒；不足历史显式None不补未来，真实相对PTS进入注意力key。A只当前，B允许当前和过去全部跨位置块。

## 一次资源验收与成本

{"loss": 2.2884738445281982, "pixel_BCE": 0.7498311400413513, "soft_Dice": 0.9558321833610535, "frame_BCE": 0.5828105211257935, "gradient_norm": 8.453596115112305, "load_seconds": 0.014104639005381614, "forward_seconds": 0.3165876349667087, "backward_seconds": 0.24652618798427284, "optimizer_seconds": 0.09902112203417346, "seconds": 0.6762395839905366, "target_blocks": 447, "gradients": {"geometry": {"present": true, "finite": true, "norm": 0.02703934535384178}, "point": {"present": true, "finite": true, "norm": 0.41952332854270935}, "score": {"present": true, "finite": true, "norm": 0.002248878590762615}, "value": {"present": true, "finite": true, "norm": 0.48058849573135376}, "space": {"present": true, "finite": true, "norm": 1.281915545463562}, "context": {"present": true, "finite": true, "norm": 1.7172781229019165}, "time": {"present": true, "finite": true, "norm": 0.008535631000995636}, "pixel": {"present": true, "finite": true, "norm": 5.819377899169922}, "frame": {"present": true, "finite": true, "norm": 2.2421960830688477}}, "peak_allocated": 2479110656, "peak_reserved": 2761949184, "status": "PASS", "target_video": "9mygy9_fake", "target_frame": 100, "window_blocks": 3575, "free_before": 18721079296, "total_gpu": 25241190400, "optimizer_states": 76, "initial_sha256": "43323d8e2229a0202c266884edb03d020409e16e34041c4bfc6f0e29fbf1a6f9"}

准备成本：{"status": "COMPLETE", "exit_code": 0, "seconds": 158.86115753097693, "new_geometry_frames": 1222, "reused_geometry_frames": 1076, "new_frozen_visual_frames": 2298, "generic_encoder_unchanged": true, "per_video": [{"video": "40y1x9_real", "frames": 150, "work_hw": [384, 216], "geometry_reused_from": "/media/njupt/8FCD3869E2FAB06F/chenyh/data/sparse_3d_forgery_detection/derived/pvst_rgbd_slice_relations_v1/geometry/5412ec415c6dc6000437ff00", "new_geometry_calls": 0, "new_visual_calls": 150, "seconds": 4.142934409959707}, {"video": "40y1x9_fake", "frames": 150, "work_hw": [384, 216], "geometry_reused_from": "/media/njupt/8FCD3869E2FAB06F/chenyh/data/sparse_3d_forgery_detection/derived/pvst_rgbd_slice_relations_v1/geometry/ac8efb162961478dd9f9af22", "new_geometry_calls": 0, "new_visual_calls": 150, "seconds": 4.1815869199926965}, {"video": "95qvg9_real", "frames": 100, "work_hw": [216, 384], "geometry_reused_from": "/media/njupt/8FCD3869E2FAB06F/chenyh/data/sparse_3d_forgery_detection/derived/pvst_rgbd_slice_relations_v1/geometry/adcb32286a41e0e3a6fdfdeb", "new_geometry_calls": 0, "new_visual_calls": 100, "seconds": 3.0192263610078953}, {"video": "95qvg9_fake", "frames": 100, "work_hw": [216, 384], "geometry_reused_from": "/media/njupt/8FCD3869E2FAB06F/chenyh/data/sparse_3d_forgery_detection/derived/pvst_rgbd_slice_relations_v1/geometry/d74a0e91fefb7a10107172c0", "new_geometry_calls": 0, "new_visual_calls": 100, "seconds": 3.44004178699106}, {"video": "9l8w54_real", "frames": 96, "work_hw": [216, 384], "geometry_reused_from": "/media/njupt/8FCD3869E2FAB06F/chenyh/data/sparse_3d_forgery_detection/derived/pvst_rgb_partition_anakin_scale_v1/geometry/5f8e607584c82ad726e61a7b", "new_geometry_calls": 0, "new_visual_calls": 96, "seconds": 3.119598948978819}, {"video": "9l8w54_fake", "frames": 96, "work_hw": [216, 384], "geometry_reused_from": "/media/njupt/8FCD3869E2FAB06F/chenyh/data/sparse_3d_forgery_detection/derived/pvst_rgb_partition_anakin_scale_v1/geometry/001bc9db0f06b3c847090d4c", "new_geometry_calls": 0, "new_visual_calls": 96, "seconds": 4.108607607020531}, {"video": "4qwgw4_real", "frames": 75, "work_hw": [216, 384], "geometry_reused_from": null, "new_geometry_calls": 75, "new_visual_calls": 75, "seconds": 3.2660461879568174}, {"video": "4qwgw4_fake", "frames": 75, "work_hw": [216, 384], "geometry_reused_from": null, "new_geometry_calls": 75, "new_visual_calls": 75, "seconds": 2.8067811640212312}, {"video": "4ym5n4_real", "frames": 96, "work_hw": [216, 384], "geometry_reused_from": null, "new_geometry_calls": 96, "new_visual_calls": 96, "seconds": 3.346415668958798}, {"video": "4ym5n4_fake", "frames": 96, "work_hw": [216, 384], "geometry_reused_from": null, "new_geometry_calls": 96, "new_visual_calls": 96, "seconds": 3.6865536409895867}, {"video": "76rjm7_real", "frames": 100, "work_hw": [216, 384], "geometry_reused_from": null, "new_geometry_calls": 100, "new_visual_calls": 100, "seconds": 4.405859787017107}, {"video": "76rjm7_fake", "frames": 100, "work_hw": [216, 384], "geometry_reused_from": null, "new_geometry_calls": 100, "new_visual_calls": 100, "seconds": 4.727752087987028}, {"video": "9mygy9_real", "frames": 120, "work_hw": [216, 384], "geometry_reused_from": null, "new_geometry_calls": 120, "new_visual_calls": 120, "seconds": 5.137231156986672}, {"video": "9mygy9_fake", "frames": 120, "work_hw": [216, 384], "geometry_reused_from": null, "new_geometry_calls": 120, "new_visual_calls": 120, "seconds": 5.7213258820120245}, {"video": "7e2e57_real", "frames": 120, "work_hw": [216, 384], "geometry_reused_from": "/media/njupt/8FCD3869E2FAB06F/chenyh/data/sparse_3d_forgery_detection/derived/pvst_rgbd_slice_relations_v1/geometry/d7b7aa3915ac3ddc0bf94b22", "new_geometry_calls": 0, "new_visual_calls": 120, "seconds": 2.748736531008035}, {"video": "7e2e57_fake", "frames": 120, "work_hw": [216, 384], "geometry_reused_from": "/media/njupt/8FCD3869E2FAB06F/chenyh/data/sparse_3d_forgery_detection/derived/pvst_rgbd_slice_relations_v1/geometry/e73bc60d4e2c5b994ee45ba4", "new_geometry_calls": 0, "new_visual_calls": 120, "seconds": 3.537681815039832}, {"video": "95q8g9_real", "frames": 72, "work_hw": [216, 384], "geometry_reused_from": "/media/njupt/8FCD3869E2FAB06F/chenyh/data/sparse_3d_forgery_detection/derived/pvst_rgbd_slice_relations_v1/geometry/2ad19f5f828cec54da51b511", "new_geometry_calls": 0, "new_visual_calls": 72, "seconds": 1.7042153920046985}, {"video": "95q8g9_fake", "frames": 72, "work_hw": [216, 384], "geometry_reused_from": "/media/njupt/8FCD3869E2FAB06F/chenyh/data/sparse_3d_forgery_detection/derived/pvst_rgbd_slice_relations_v1/geometry/37b22da558e1daee1e72b707", "new_geometry_calls": 0, "new_visual_calls": 72, "seconds": 2.0113993249833584}, {"video": "4xd3r9_real", "frames": 100, "work_hw": [216, 384], "geometry_reused_from": null, "new_geometry_calls": 100, "new_visual_calls": 100, "seconds": 4.364927186979912}, {"video": "4xd3r9_fake", "frames": 100, "work_hw": [216, 384], "geometry_reused_from": null, "new_geometry_calls": 100, "new_visual_calls": 100, "seconds": 4.894095739990007}, {"video": "7ogeg4_real", "frames": 120, "work_hw": [287, 384], "geometry_reused_from": null, "new_geometry_calls": 120, "new_visual_calls": 120, "seconds": 5.0199667059932835}, {"video": "7ogeg4_fake", "frames": 120, "work_hw": [287, 384], "geometry_reused_from": null, "new_geometry_calls": 120, "new_visual_calls": 120, "seconds": 5.187112862011418}]}

训练/评价成本：{"XYZ_CURRENT": {"updates": 576, "train_seconds": 7.775019744003657, "peak_allocated": 466261504, "peak_reserved": 914358272}, "XYZ_TEMPORAL": {"updates": 576, "train_seconds": 35.38531978696119, "peak_allocated": 2678131200, "peak_reserved": 3053453312}}

## 固定历史倒序（不训练）

B倒序评价：{"macro": {"pixel_AP": 0.07099418269775494, "Dice": 0.07151398810789143, "IoU": 0.040928004642241785, "GT_recall": 0.2307029882798796, "real_FPR": 0.07627356738583639, "frame_metrics": {"auroc": 0.4852159722876803, "ap": 0.48413036878898635, "acc": 0.5, "precision": 0.5, "recall": 0.5339805825242718, "f1": 0.5164319248826291, "confusion_matrix": [[192, 220], [192, 220]], "undefined_reason": null}, "video_metrics": {"auroc": 0.5, "ap": 0.5666666666666667, "acc": 0.5, "precision": 0.5, "recall": 0.5, "f1": 0.5, "confusion_matrix": [[2, 2], [2, 2]], "undefined_reason": null}}, "macro_pixel_AP_delta": 7.001809727610886e-06, "max_abs_frame_logit_change": 0.03125, "mean_abs_frame_logit_change": 0.0009386377427184466, "frames": 824, "seconds": 27.642583712004125, "fixed_frame_max_abs_pixel_logit_change": 0.046875}。仅重新绑定历史内容到原时间槽，当前/历史集合及次数不变。顺序变化不预设提高异常分数。

帧/视频指标仅辅助，开发来源少且无独立内容编辑起止真值，不称已验证时间边界定位。A/B只回答本模型历史关系增量；不能单独证明三维优于二维，负结果也不否定全部时空关系假设。旧结果不与本协议混表。

固定首/中/末RGB/GT/A/B：preview/index.html。

执行身份：{"execution_head": "a1f320a1586078d7613c3415dc9d5df688040740", "config_sha256": "981ff022f1f51daeae6228eff2f15f5267c4e6942efa1c7b220c68138d32d5c1", "manifest_sha256": "c181b174c86a79de8216399a98bbb574287a161e2c8aa0f7c9a8dd884d97736b", "prepared_identity": {"source_head": "8da2b614ee776b48d66c3341fb030bb36b35a050", "config_sha256": "996fb1d3f8c0b4cdd33fd4c02933ab9dd157a1ac30ada479d4e2282a1aa7abf4", "manifest_sha256": "c181b174c86a79de8216399a98bbb574287a161e2c8aa0f7c9a8dd884d97736b", "DFormer_generic_weight_sha256": "8a086a3fb4e7ebda0806172938dd199bb7d8394593a8176640ed79735e35a0c0", "DFormer_revision": "e3273009b759b578945483828ff315d560be94c9", "MoGe_weight_sha256": "79a16621928c2bf0ed04659218c55c01075e950507f40bb3332fb4c873d3e1dc", "MoGe_revision": "07444410f1e33f402353b99d6ccd26bd31e469e8", "forensic_weights_used": false}, "model_source_sha256": "1612716ee66ff986b01758ae1d925ac382bc339b69c9e3c342b0b418adba3fa0"}

本轮停止，不扩数据、seed或架构。完整产物：/media/njupt/8FCD3869E2FAB06F/chenyh/data/sparse_3d_forgery_detection/derived/pvst_xyz_temporal_pilot_v1


## 补充核验与直接结论

- 实际工作区：`/media/njupt/8FCD3869E2FAB06F/chenyh/projects/pvst-geometry-dataset-screen`，独立分支`pvst-xyz-temporal-pilot-v1`。模型/配置执行提交：`a1f320a1586078d7613c3415dc9d5df688040740`，未在运行期间更改执行模型。
- 两条件各640,387个可训练参数；同一初始化SHA `43323d8e2229a0202c266884edb03d020409e16e34041c4bfc6f0e29fbf1a6f9`。两份末次权重、JSONL及顺序独立复核，各576次有限loss/梯度更新，real/fake各288目标，476个唯一目标帧；geometry、点编码、pool score/value、space/context、像素/帧头参数均实际改变。验收更新未继承。完整核对见final_integrity_checks.json。
- 训练来源14视频共1474个原生帧，fake40y1x9的4帧排除监督，仍可作历史；训练是固定采样576步，不声称所有可监督帧都作过目标。开发8视频824个唯一原生帧：412 fake正GT、412 real负GT。每条件计分553,574,400个原生像素观察，覆盖100%画面，没有以geometry valid裁评价。A/B身份、PTS、GT和像素范围一致。没有开发排除帧。
- 工作分辨率长边≤384、padding右/下至32倍数；所有有效工作网格点参与XYZ路由和学习池化，无GT ROI。恢复原生输出**不等于原生分辨率建模**，下采样对细小编辑及边界的损失是本紧凑探针限制。真实XYZ块坐标保留三元整数，j可由Z块号表示、k对应XY块坐标；展平顺序不作空间距离。
- 训练几何尺度12.699031352996826（训练采样数量见scale_selection.json）统一应用于所有源；h=0.5。候选h=0.125/0.25/0.5/1/2/4的训练最大占据数分别5869/1914/456/106/37/14。预算仅决定选哪个统一h，不截断或丢弃点/块。开发数据未决定scale/h。没有逐帧min-max坐标缩放。

**主要结果不支持跨运动来源一致的历史增益。** 宏AP B−A=+0.00158762，2升2降；3个明确运动开发来源平均差为-0.00142291，下降。7ogeg4对象移除参照提高+0.01061921，是宏平均小幅提高的重要来源。猫添加AP下降属于跨编辑类型表现，但单条结果不能将原因唯一归为编辑类型。real宏FPR从6.397724%升至7.626510%，故不能将召回上升写成无代价定位改善。空间和帧/视频检测分别判断，帧AUROC仍较弱，视频AUROC两组均0.5；本轮没有独立时间起止真值验证。

## 倒序评价的数值限制

倒序保持每个目标当前帧不变、历史源帧集合及次数一致，内容（含块空间位置）重新绑定到原相对PTS槽，无额外训练。宏pixel AP变化+0.00000700，帧logit最大变化0.03125000，固定图像logit最大变化0.04687500。

另从已保存记录核对：有44个目标的历史内容绑定实际上未改变（起始历史不足等），这些目标在两次评价之间也有最大0.01562500的帧logit差异。因此不能把所有微小倒序变化单独归于时间顺序；具体数值差异来源本轮未独立定位，不直接归因于BF16。没有额外开启精度对照、重复前向或重训。当前倒序结果不足以证明明显的顺序依赖。

## 实测资源、耗时与退出

- 一次最大训练块负载窗口9mygy9/fake目标100，窗口3,575块。前向0.3166s、反向0.2465s、AdamW 0.0990s，总0.6762s；真实更新PASS、优化器状态建立，权重丢弃。
- 资源验收allocated/reserved 2.309/2.572GiB；开跑可用17.435GiB。训练/末次评价峰值A 0.434/0.852GiB，B 2.494/2.844GiB。运行进程RSS峰值1.850GiB。共享服务器其他进程未干预，本任务GPU进程已释放。
- 准备158.861s；复用MoGe 1076帧、新增MoGe1222帧、新增官方通用视觉缓存2298帧，逻辑磁盘文件合计5.971GiB。无下载/新依赖，未读取旧取证checkpoint或缓存C。
- 训练A 7.775s，B 35.385s；末次开发评价分别27.472s/27.462s，倒序评价27.643s。资源验收冷启动估计训练约779s仅为保守估计，未冒充实测；正式逐24步进度给出实际吞吐。主运行总128.406s；准备加主运行约287.267s，不含实现、测试、Git与训练后只读核验。
- 4项必要CPU合同测试通过；额外只读核对更新、源身份、参数变化和倒序集合不变均通过。无OOM、无超时、无非有限更新，主程序真实退出0，计算COMPLETE。发布状态另记publication.json，网络失败不覆盖计算状态。

本轮不追加实验。低定位成绩不是程序FAILED，也不足以确定是拟合、数据、工作分辨率或关系结构中的哪一项单独造成。
